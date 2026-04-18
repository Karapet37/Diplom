"""
SpeechPlanner — cognitive pipeline output → structured speech plan → LLM verbalization.

Architecture
------------
OLD:  user_text → staged prompts (INTERACTION_ROUTER → ... → FINAL_GENERATOR) → LLM writes everything
NEW:  user_text → CognitivePipeline.forward() → SpeechPlan → LLM verbalize(plan)

The cognitive pipeline (P1–P6 + PersonalityGenome) is the decision-maker.
The LLM's only job: turn a structured SpeechPlan into fluent natural language.

Integration point in chat_engine.py
-------------------------------------
After `built` is computed (context_building stage) and `_cog_output` is available:

    from .speech_planner import SpeechPlanner, verbalizer_prompt
    _plan = SpeechPlanner().build(_cog_output, built=built, user_text=clean_message,
                                   language=response_language)
    prompt = verbalizer_prompt(_plan)
    # → replaces _call_build_chat_prompt_compat(...)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cognitive_pipeline import (
    ACTION_FAMILIES, RESOLUTION_NAMES, CognitiveTurnOutput,
)


# ─── Semantic maps ─────────────────────────────────────────────────────────────

_ACTION_TO_GOAL: dict[str, str] = {
    'approach':        'be open and engage with the topic directly',
    'avoid':           'stay on the surface, do not go deeper into the issue',
    'freeze':          'pause and acknowledge the weight without rushing to answer',
    'attack':          'push back clearly, express disagreement without softening',
    'placate':         'soften the tension, acknowledge the other person',
    'analyze':         'break down the situation calmly and clearly',
    'ask_for_help':    'express a genuine need, invite support or understanding',
    'seek_control':    'establish a clear frame or boundary for this exchange',
    'reduce_exposure': 'keep the reply brief, do not over-share',
    'reframe':         'introduce a different perspective on the situation',
    'self_protect':    'maintain boundaries, give only what is necessary',
    'connect':         'build closeness, reflect and empathize',
    'withdraw':        'step back, reply minimally, signal need for space',
    'plan_small_step': 'offer one concrete, modest next step',
}

_RESOLUTION_TO_TONE: dict[str, str] = {
    'avoidance':        'cautious, slightly evasive — avoids naming the core issue',
    'overcompensation': 'over-eager, trying harder than needed',
    'attack':           'sharp and direct, does not soften edges',
    'freeze':           'slow and careful — every word chosen deliberately',
    'planning':         'structured, practical, forward-looking',
    'support_seeking':  'warm but uncertain, reaching out',
    'self_deception':   'reassuring, glosses over contradictions',
}

_BLOCKED_TOPIC_NAMES: dict[str, str] = {
    'approach':     'initiating directness or closeness',
    'attack':       'confrontation or sharp criticism',
    'connect':      'emotional disclosure or intimacy',
    'placate':      'appeasing or over-accommodating',
    'ask_for_help': 'admitting need or asking for support',
    'analyze':      'analytical breakdown',
    'reframe':      'reinterpreting the situation',
    'plan_small_step': 'concrete planning',
}

# Thought vector layout (mirrors cognitive_pipeline.py P4 output)
_T_RISK       = 0
_T_CONFIDENCE = 1
_T_NEED       = slice(2, 5)    # softmax: connection / achievement / safety
_T_FRAME      = slice(5, 8)    # softmax: approach / hold / retreat


# ─── SpeechPlan ────────────────────────────────────────────────────────────────

@dataclass
class SpeechPlan:
    """
    Complete specification of WHAT to say.
    The LLM receives this and produces fluent prose — it invents nothing.
    """
    action_name:     str
    speech_goal:     str
    tone:            str
    perceived_risk:  float
    confidence:      float
    intensity:       float
    primary_event:   str
    key_points:      list[str]
    blocked_topics:  list[str]
    style_hints:     list[str]
    language:        str = 'en'
    max_tokens:      int = 200
    _cog_snapshot:   dict[str, Any] = field(default_factory=dict, repr=False)

    def as_directive(self) -> str:
        """Compact directive block for the LLM verbalization prompt."""
        length_hint = 'very short (1–2 sentences)' if self.max_tokens <= 80 else \
                      'brief (2–4 sentences)' if self.max_tokens <= 160 else \
                      'moderate length'
        blocked_str = ', '.join(self.blocked_topics) if self.blocked_topics else 'none'
        hedge = ' Use hedging language where uncertain.' if self.confidence < 0.4 else ''
        points_str = '\n'.join(f'  {i+1}. {p}' for i, p in enumerate(self.key_points)) \
                     if self.key_points else '  1. express the persona\'s current state authentically'

        return (
            f"── SPEECH DIRECTIVE ──\n"
            f"Goal         : {self.speech_goal}\n"
            f"Tone         : {self.tone}\n"
            f"Risk level   : {self.perceived_risk:.2f}  |  Confidence: {self.confidence:.2f}  |  Intensity: {self.intensity:.2f}\n"
            f"Event context: {self.primary_event}\n"
            f"Points to express (in order):\n{points_str}\n"
            f"Avoid touching: {blocked_str}\n"
            f"Reply length : {length_hint}.{hedge}\n"
            f"──────────────────────"
        )


# ─── SpeechPlanner ─────────────────────────────────────────────────────────────

class SpeechPlanner:
    """
    Builds a SpeechPlan from CognitiveTurnOutput + the built context dict.

    The `built` dict is the output of context_builder.build_context() and contains:
      - 'persona_block'   : persona description (traits, voice, knowledge)
      - 'recent_dialogue' : last N turns of session history
      - 'graph_context'   : graph-grounded facts about current topic
    """

    def build(
        self,
        cog: CognitiveTurnOutput,
        built: dict[str, Any] | None = None,
        user_text: str = '',
        language: str = 'en',
    ) -> SpeechPlan:
        built = built or {}
        thought = np.array(cog.thought_vec, dtype=np.float32)

        speech_goal = _ACTION_TO_GOAL.get(cog.action_name, cog.action_name)
        tone = self._build_tone(cog, thought)
        blocked = [_BLOCKED_TOPIC_NAMES[a] for a in cog.blocked_actions if a in _BLOCKED_TOPIC_NAMES]
        style_hints = self._build_style_hints(thought, language, built)
        key_points = self._build_key_points(cog, thought, user_text, built)

        # Token budget: tighter when guarded or intense
        if cog.perceived_risk > 0.65 or cog.intensity > 0.75:
            max_tokens = 80
        elif cog.perceived_risk > 0.4:
            max_tokens = 140
        else:
            max_tokens = 220

        return SpeechPlan(
            action_name    = cog.action_name,
            speech_goal    = speech_goal,
            tone           = tone,
            perceived_risk = cog.perceived_risk,
            confidence     = float(thought[_T_CONFIDENCE]),
            intensity      = cog.intensity,
            primary_event  = cog.primary_event,
            key_points     = key_points,
            blocked_topics = blocked,
            style_hints    = style_hints,
            language       = language,
            max_tokens     = max_tokens,
            _cog_snapshot  = cog.to_dict(),
        )

    # ── private helpers ──────────────────────────────────────────────────────

    def _build_tone(self, cog: CognitiveTurnOutput, thought: np.ndarray) -> str:
        base = _RESOLUTION_TO_TONE.get(cog.dominant_resolution, 'measured')
        if cog.intensity > 0.75:
            return 'intense and charged — ' + base
        if cog.intensity < 0.2:
            return 'calm and low-key — ' + base
        return base

    def _build_style_hints(
        self,
        thought: np.ndarray,
        language: str,
        built: dict[str, Any],
    ) -> list[str]:
        hints: list[str] = []
        if language and language != 'en':
            hints.append(f'reply in {language}')

        need = thought[_T_NEED]
        if float(need[0]) > 0.5:      # connection dominant
            hints.append('use warm, personal phrasing')
        elif float(need[2]) > 0.5:    # safety dominant
            hints.append('keep phrasing stable and reassuring')

        # If persona block mentions specific speech patterns, pass that through
        pb = str(built.get('persona_block') or '')
        if 'formal' in pb.lower():
            hints.append('maintain formal register')
        elif 'casual' in pb.lower() or 'informal' in pb.lower():
            hints.append('keep casual, natural register')

        return hints

    def _build_key_points(
        self,
        cog: CognitiveTurnOutput,
        thought: np.ndarray,
        user_text: str,
        built: dict[str, Any],
    ) -> list[str]:
        """
        Builds ordered content points from:
          1. Cognitive state signals (always present)
          2. Graph facts relevant to the current topic (if available)
          3. Persona knowledge/traits (from persona_block)
          4. Session continuity (from recent_dialogue)
        """
        points: list[str] = []
        frame = thought[_T_FRAME]  # [approach, hold, retreat]

        # ── 1. Acknowledge the event signal ─────────────────────────────────
        if cog.primary_event not in ('neutral', 'reward'):
            points.append(f'acknowledge the {cog.primary_event} in what was said')

        # ── 2. Internal state visibility (gated by risk) ─────────────────────
        if cog.perceived_risk < 0.35:
            points.append("share the persona's genuine perspective on this")
        elif cog.perceived_risk < 0.6:
            points.append('respond thoughtfully, keep some things internal')
        else:
            points.append('be careful — do not over-disclose; protect the core')

        # ── 3. Graph facts — extract sentences most relevant to user_text ───
        graph_ctx = str(built.get('graph_context') or '').strip()
        if graph_ctx:
            relevant = self._extract_relevant_sentences(graph_ctx, user_text, n=2)
            for fact in relevant:
                points.append(f'ground in known fact: "{fact}"')

        # ── 4. Resolution-driven content ─────────────────────────────────────
        res = cog.dominant_resolution
        if res == 'planning':
            points.append('offer a concrete direction or next step')
        elif res == 'support_seeking':
            points.append('express that understanding or support would be welcome')
        elif res == 'freeze':
            points.append('let the weight of the moment be present — do not rush')
        elif res == 'overcompensation':
            points.append('pull back slightly — do not try too hard to fix it')

        # ── 5. Frame direction shapes the closing ───────────────────────────
        if float(frame[0]) > 0.5:     # approach-dominant frame
            points.append('end with an opening — leave room for the conversation to continue')
        elif float(frame[2]) > 0.5:   # retreat-dominant frame
            points.append('close gently but clearly')

        # ── 6. Persona voice hint (from persona_block excerpt) ───────────────
        pb = str(built.get('persona_block') or '').strip()
        if pb:
            first_line = pb.split('\n')[0][:120].strip()
            if first_line:
                points.append(f'stay consistent with: {first_line}')

        return points

    def _extract_relevant_sentences(
        self,
        text: str,
        query: str,
        n: int = 2,
    ) -> list[str]:
        """
        Simple keyword-overlap ranking to find sentences in `text`
        most relevant to `query`. No external dependencies.
        """
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]
        if not sentences:
            return []
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        scored: list[tuple[float, str]] = []
        for s in sentences:
            s_words = set(re.findall(r'\b\w+\b', s.lower()))
            overlap = len(query_words & s_words) / (len(query_words) + 1)
            scored.append((overlap, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:n] if scored[0][0] > 0.0]


# ─── verbalizer_prompt ─────────────────────────────────────────────────────────

def verbalizer_prompt(
    plan: SpeechPlan,
    persona_voice: str = '',
    recent_exchange: str = '',
) -> str:
    """
    Build the complete LLM prompt for verbalization.
    This is the only prompt the LLM receives — no staging, no chain.

    Args:
        plan:            SpeechPlan from SpeechPlanner.build()
        persona_voice:   persona_block[:500] — voice/character description
        recent_exchange: recent_dialogue[:600] — last turns for continuity
    """
    system_parts: list[str] = []
    user_parts: list[str] = []

    if persona_voice:
        system_parts.append(f"PERSONA\n{persona_voice.strip()[:500]}")

    if recent_exchange:
        system_parts.append(f"RECENT EXCHANGE\n{recent_exchange.strip()[:600]}")

    user_parts.append(plan.as_directive())

    if plan.style_hints:
        user_parts.append("STYLE REQUIREMENTS\n" + '\n'.join(f'• {h}' for h in plan.style_hints))

    user_parts.append(
        "Write the reply now. Follow the directive exactly. "
        "Use natural speech for the persona. "
        "Do not add topics not listed above. "
        "Do not explain your reasoning or reference this directive. "
        "Do not break character."
    )

    user_block = '\n\n'.join(user_parts)

    if system_parts:
        # Use the "User question:" separator so _split_chat_prompt_messages puts
        # PERSONA + RECENT EXCHANGE in the system role and the directive in the user role.
        return '\n\n'.join(system_parts) + '\n\nUser question:\n' + user_block

    return user_block
