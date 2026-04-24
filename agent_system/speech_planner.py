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
    action_name:      str
    speech_goal:      str
    tone:             str
    perceived_risk:   float
    confidence:       float
    intensity:        float
    primary_event:    str
    key_points:       list[str]
    blocked_topics:   list[str]
    style_hints:      list[str]
    language:         str   = 'en'
    max_tokens:       int   = 200
    conflict_tension: float = 0.0   # entropy of resolution probs: high → genuine ambivalence
    _cog_snapshot:    dict[str, Any] = field(default_factory=dict, repr=False)

    def as_directive(self) -> str:
        """Compact directive block for the LLM verbalization prompt."""
        length_hint = 'very short (1–2 sentences)' if self.max_tokens <= 80 else \
                      'brief (2–4 sentences)' if self.max_tokens <= 160 else \
                      'moderate length'
        blocked_str = ', '.join(self.blocked_topics) if self.blocked_topics else 'none'
        hedge = ' Use hedging language where uncertain.' if self.confidence < 0.4 else ''
        points_str = '\n'.join(f'  {i+1}. {p}' for i, p in enumerate(self.key_points)) \
                     if self.key_points else '  1. express the persona\'s current state authentically'

        tension_note = ''
        if self.conflict_tension > 0.7:
            tension_note = ' Internal conflict is high — reflect genuine ambivalence, do not force resolution.'
        elif self.conflict_tension > 0.45:
            tension_note = ' Mild internal tension present — allow slight hesitation or nuance.'

        return (
            f"── SPEECH DIRECTIVE ──\n"
            f"Goal         : {self.speech_goal}\n"
            f"Tone         : {self.tone}\n"
            f"Risk level   : {self.perceived_risk:.2f}  |  Confidence: {self.confidence:.2f}  |  Intensity: {self.intensity:.2f}\n"
            f"Conflict     : {self.conflict_tension:.2f}  (0=resolved · 1=ambivalent)\n"
            f"Event context: {self.primary_event}\n"
            f"Points to express (in order):\n{points_str}\n"
            f"Avoid touching: {blocked_str}\n"
            f"Reply length : {length_hint}.{hedge}{tension_note}\n"
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

        # Conflict tension = entropy of resolution probability distribution.
        # High entropy means no clear resolution dominates → genuine ambivalence.
        conflict_vec = np.array(cog.conflict_vec, dtype=np.float32)
        res_probs = conflict_vec[1:] if len(conflict_vec) > 1 else conflict_vec
        res_probs = np.clip(res_probs, 1e-9, None)
        res_probs = res_probs / res_probs.sum()
        max_entropy = float(np.log(len(res_probs)))
        raw_entropy = float(-np.sum(res_probs * np.log(res_probs)))
        conflict_tension = float(np.clip(raw_entropy / max_entropy if max_entropy > 0 else 0.0, 0.0, 1.0))

        # Token budget: tighter when guarded, intense, or highly conflicted
        if cog.perceived_risk > 0.65 or cog.intensity > 0.75 or conflict_tension > 0.7:
            max_tokens = 110
        elif cog.perceived_risk > 0.4 or conflict_tension > 0.45:
            max_tokens = 180
        else:
            max_tokens = 280

        return SpeechPlan(
            action_name      = cog.action_name,
            speech_goal      = speech_goal,
            tone             = tone,
            perceived_risk   = cog.perceived_risk,
            confidence       = float(thought[_T_CONFIDENCE]),
            intensity        = cog.intensity,
            primary_event    = cog.primary_event,
            key_points       = key_points,
            blocked_topics   = blocked,
            style_hints      = style_hints,
            language         = language,
            max_tokens       = max_tokens,
            conflict_tension = conflict_tension,
            _cog_snapshot    = cog.to_dict(),
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

        for point in self._vector_guidance_points(built):
            if point not in points:
                points.append(point)

        return points

    def _vector_guidance_points(self, built: dict[str, Any]) -> list[str]:
        payload = dict(built.get('message_vector_runtime') or {})
        vector = dict(payload.get('current_vector') or {})
        transition = dict(payload.get('transition_interpretation') or {})
        flow = [str(item).strip() for item in list(payload.get('history_flow') or []) if str(item).strip()]
        points: list[str] = []

        def main(pid: str) -> str:
            return str((vector.get(pid) or {}).get('main') or '').strip()

        if main('P24') in {'sarcasm', 'dry_sarcasm', 'false_praise'} or main('P41') == 'masking':
            points.append('treat the surface wording as masked — answer the underlying social move, not the literal phrasing')
        if main('P13') == 'attack' or main('P30') == 'pressure':
            points.append('hold boundaries and do not reward pressure with extra openness')
        if main('P16') == 'care' or main('P20') == 'friendliness':
            points.append('preserve the contact instead of turning cold or mechanical')
        if main('P34') == 'reconciliation' or main('P38') == 'softening':
            points.append('leave a small opening for repair')
        if main('P33') == 'distancing' or main('P37') == 'rupture':
            points.append('signal distance clearly instead of pretending the tension is gone')
        if str(transition.get('type') or '').strip() == 'masking':
            points.append('the transition suggests masking — do not misread it as straightforward sincerity')
        elif str(transition.get('type') or '').strip() == 'repair':
            points.append('the transition suggests repair — allow some de-escalation in tone')
        elif str(transition.get('type') or '').strip() == 'escalation':
            points.append('the transition suggests escalation — keep the answer controlled and grounded')
        if flow:
            points.append(f'keep continuity with the recent dialogue flow: {" -> ".join(flow[:4])}')
        return points[:4]

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


# ─── qwen_fast_respond ────────────────────────────────────────────────────────

def qwen_fast_respond(
    plan: 'SpeechPlan',
    user_message: str,
    llm: Any,
    persona_system: str = '',
    history: list[tuple[str, str]] | None = None,
) -> str:
    """
    Fast verbalization for Qwen3.x models using raw completion with
    pre-filled <think>...</think> to skip the reasoning phase entirely.

    Achieves 1.5–4s response time vs 13–22s with full thinking.

    Args:
        plan:           SpeechPlan from SpeechPlanner.build()
        user_message:   the original user text
        llm:            llama_cpp.Llama instance
        persona_system: persona description for system prompt
        history:        [(user, assistant), ...] recent turns (last 2 used)

    Returns:
        clean Russian response string
    """
    import re as _re
    from src.utils.local_llm_provider import _strip_think_blocks

    lang = _lang2(plan) if hasattr(plan, 'language') else 'ru'
    action_instr = _ACTION_TO_GOAL.get(plan.action_name, plan.action_name)

    _length_hints: dict[str, tuple[str, str, str]] = {
        'ru': ('1 предложение', '1-2 предложения', '2-3 предложения'),
        'hy': ('1 нахадас', '1-2 нахадас', '2-3 нахадас'),
        'zh': ('1句', '1-2句', '2-3句'),
    }
    lh = _length_hints.get(lang, _length_hints['ru'])
    length_hint = lh[0] if plan.max_tokens <= 80 else (lh[1] if plan.max_tokens <= 160 else lh[2])

    tension_note = _TENSION_HIGH.get(lang, _TENSION_HIGH['ru']) if plan.conflict_tension > 0.7 else (
        _TENSION_MID.get(lang, _TENSION_MID['ru']) if plan.conflict_tension > 0.45 else ''
    )

    _lang_desc: dict[str, str] = {
        'ru': f'Ты — живой тёплый эмпатичный собеседник. Отвечаешь по-русски, {length_hint}.{tension_note}',
        'hy': f'Ду — кенданի, джерм ев кареккан зрутсяких: Патасхани hайерен, {length_hint}.{tension_note}',
        'zh': f'你是一个温暖、真实、有同理心的对话者。用中文回答，{length_hint}。{tension_note}',
    }
    think_line: dict[str, str] = {
        'ru': 'Отвечаю кратко и тепло по-русски.',
        'hy': 'Կarчрам кармас ев джерм hайерен.',
        'zh': '我简短温暖地用中文回答。',
    }

    sys_parts = [_lang_desc.get(lang, _lang_desc['ru'])]
    if persona_system:
        sys_parts.append(persona_system.strip()[:300])
    sys_parts.append(f'Сейчас: {action_instr}' if lang == 'ru' else
                     f'现在：{action_instr}' if lang == 'zh' else
                     f'Hиmer: {action_instr}')
    system = ' '.join(sys_parts)

    think_txt = think_line.get(lang, think_line['ru'])

    # Build prompt with Qwen3 chat template
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n"

    for h_user, h_sys in (history or [])[-2:]:
        prompt += (
            f"<|im_start|>user\n{h_user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<think>\n{think_txt}\n</think>\n\n"
            f"{h_sys}<|im_end|>\n"
        )

    prompt += (
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n{think_txt}\n</think>\n\n"
    )

    out = llm.create_completion(
        prompt=prompt,
        max_tokens=min(plan.max_tokens, 220),
        temperature=0.78,
        top_p=0.92,
        stop=["<|im_end|>", "<|im_start|>", "\n\n\n", "<think>"],
    )
    raw = out["choices"][0]["text"].strip()

    cleaned = _strip_think_blocks(raw).split('<|im_end|>')[0].strip()
    _meta = _re.compile(
        r'\b(analyze|нужно ответить|пользователь|задача:|let me|thinking process|'
        r'key point|constraint:|step \d|role:|task:)\b', _re.IGNORECASE,
    )
    if not cleaned or _meta.search(cleaned[:60]):
        # Try to salvage natural sentences in any of the three scripts
        sentences = _re.findall(
            r'(?:[А-ЯЁа-яёԱ-Ֆա-և\u4e00-\u9fff][^.!?\n]{8,200}[.!?])',
            raw,
        )
        good = [s for s in sentences if not _meta.search(s) and len(s) > 10]
        cleaned = ' '.join(good[-2:]).strip() if good else raw
    return ' '.join(_re.split(r'(?<=[.!?])\s+', cleaned.strip())[:4]).strip()


def mistral_fast_respond(
    plan: 'SpeechPlan',
    user_message: str,
    llm: Any,
    persona_system: str = '',
    history: list[tuple[str, str]] | None = None,
) -> str:
    """Fast verbalization for Mistral/LLaMA-family models using [INST]...[/INST] format.

    No thinking tokens exist in these models so no pre-fill trick needed.
    Max tokens capped at 120 to keep response under 30s on i7-1255U.
    """
    import re as _re

    lang = _lang2(plan) if hasattr(plan, 'language') else 'ru'
    action_instr = _ACTION_TO_GOAL.get(plan.action_name, plan.action_name)
    tension_note = _TENSION_HIGH.get(lang, '') if plan.conflict_tension > 0.7 else ''

    _sys_templates: dict[str, str] = {
        'ru': f'Ты — тёплый эмпатичный собеседник. Отвечаешь по-русски, 1-2 предложения.{tension_note} Сейчас: {action_instr}',
        'hy': f'Ду — джерм зрутсяких. Патасхани hайерен, 1-2 нахадас.{tension_note} Hиmer: {action_instr}',
        'zh': f'你是温暖的对话者。用中文回答，1-2句。{tension_note} 现在：{action_instr}',
    }
    system = _sys_templates.get(lang, _sys_templates['ru'])
    if persona_system:
        system += ' ' + persona_system.strip()[:200]

    # Mistral v0.3 instruct format: <s>[INST] system \n\n user [/INST]
    prompt = f"<s>[INST] {system}\n\n"
    for h_user, h_sys in (history or [])[-2:]:
        prompt += f"{h_user} [/INST] {h_sys} </s>[INST] "
    prompt += f"{user_message} [/INST]"

    out = llm.create_completion(
        prompt=prompt,
        max_tokens=min(plan.max_tokens, 200),
        temperature=0.72,
        top_p=0.90,
        stop=["</s>", "[INST]", "[/INST]", "\n\n\n"],
    )
    raw = out["choices"][0]["text"].strip()
    _meta = _re.compile(
        r'\b(i will|let me|step \d|thinking|analyze|task:|role:)\b', _re.IGNORECASE,
    )
    sentences = _re.findall(
        r'(?:[А-ЯЁа-яёԱ-Ֆա-և\u4e00-\u9fff][^.!?\n]{8,200}[.!?])', raw,
    )
    good = [s for s in sentences if not _meta.search(s) and len(s) > 10]
    if good:
        return ' '.join(good[:4]).strip()
    return ' '.join(_re.split(r'(?<=[.!?])\s+', raw.strip())[:4]).strip() or raw


# ─── simple_verbalizer_prompt ─────────────────────────────────────────────────

_ACTION_TO_RU_INSTRUCTION: dict[str, str] = {
    'connect':         'Ответь тепло и с искренней эмпатией. Просто будь рядом.',
    'ask_for_help':    'Ответь поддерживающе — дай понять что слышишь и что рядом.',
    'approach':        'Ответь открыто и тепло, вступи в контакт.',
    'placate':         'Ответь мягко, смягчи напряжение.',
    'reframe':         'Предложи другой взгляд на ситуацию, мягко.',
    'plan_small_step': 'Предложи один конкретный небольшой шаг вперёд.',
    'analyze':         'Разбери ситуацию спокойно и ясно.',
    'freeze':          'Ответь медленно и осторожно, не торопись.',
    'withdraw':        'Ответь кратко, дай пространство.',
    'self_protect':    'Ответь осторожно, защити себя.',
    'seek_control':    'Мягко установи рамки разговора.',
    'avoid':           'Ответь поверхностно, не углубляйся.',
    'attack':          'Чётко вырази несогласие.',
    'reduce_exposure': 'Ответь очень кратко.',
}

_ACTION_TO_HY_INSTRUCTION: dict[str, str] = {
    'connect':         'Ջերմությամb ев անкеղծ кarекцанутjамb патасхани. Павасапес кес нrank.',
    'ask_for_help':    'Хнамхндрвак патасхани — тур hасканал, вор лсум ес ев когкит ес.',
    'approach':        'Барьак ев джерм патасхани, капакцир капакцутюн.',
    'placate':         'Мегр патасхани, мегрецрир лара.',
    'reframe':         'Мегр жаманак аражаркир карворутюни мьюс ален нкатутюни.',
    'plan_small_step': 'Аражаркир мьек конкрет паккетя квад.',
    'analyze':         'Ихавахарутюне бацатрир анхагист ев hаски.',
    'freeze':          'Камас ев згушак патасхани, ми шататапес.',
    'withdraw':        'Камас патасхани, тур тарацутюн.',
    'self_protect':    'Згушак патасхани, паштпанир кезанд.',
    'seek_control':    'Мегрос сахмани хоски кариначе.',
    'avoid':           'Матчарайкин патасхани, ми хорини.',
    'attack':          'Хатканапен аргаканел антемутюне.',
    'reduce_exposure': 'Шат камас патасхани.',
}

_ACTION_TO_ZH_INSTRUCTION: dict[str, str] = {
    'connect':         '温暖而真诚地回应。就陪在他身边。',
    'ask_for_help':    '给予支持性回应 — 让他知道你在听，你在这里。',
    'approach':        '开放而温暖地回应，建立连接。',
    'placate':         '温和地回应，缓和紧张气氛。',
    'reframe':         '轻柔地提供另一种看待情况的视角。',
    'plan_small_step': '提出一个具体的小步骤。',
    'analyze':         '平静清晰地分析情况。',
    'freeze':          '慢慢地、谨慎地回应，不要着急。',
    'withdraw':        '简短回应，给对方空间。',
    'self_protect':    '谨慎回应，保持自己的边界。',
    'seek_control':    '温和地建立对话框架。',
    'avoid':           '浅浅地回应，不要深入。',
    'attack':          '清晰地表达不同意见。',
    'reduce_exposure': '非常简短地回应。',
}

_TONE_TO_RU: dict[str, str] = {
    'avoidance':        'осторожно, слегка уклончиво',
    'overcompensation': 'тепло, но без чрезмерного старания',
    'attack':           'прямо, без смягчений',
    'freeze':           'медленно, каждое слово на вес',
    'planning':         'структурированно, практично',
    'support_seeking':  'тепло, с долей неуверенности',
    'self_deception':   'мягко, не акцентируя противоречий',
}

_TONE_TO_ZH: dict[str, str] = {
    'avoidance':        '谨慎，略显回避',
    'overcompensation': '温暖但不过分',
    'attack':           '直接，不软化',
    'freeze':           '缓慢，字字斟酌',
    'planning':         '有条理，务实',
    'support_seeking':  '温暖，带一丝不确定',
    'self_deception':   '温和，不强调矛盾',
}

_TENSION_HIGH: dict[str, str] = {
    'ru': ' Отрази подлинную внутреннюю неоднозначность — не спеши к решению.',
    'hy': ' Арцанацрир иракан неркаин анкаюнакануютюне — ми шататапес лузуме.',
    'zh': ' 反映真实的内心矛盾 — 不要急于解决。',
    'en': ' Reflect genuine inner ambivalence — do not force resolution.',
}
_TENSION_MID: dict[str, str] = {
    'ru': ' Допусти лёгкую нотку колебания.',
    'hy': ' Тур текн таталутяни нота.',
    'zh': ' 允许一丝犹豫。',
    'en': ' Allow a slight note of hesitation.',
}


def _lang2(plan: 'SpeechPlan') -> str:
    return str(plan.language or 'ru').lower()[:2]


def simple_verbalizer_prompt(
    plan: 'SpeechPlan',
    user_message: str = '',
    persona_voice: str = '',
    recent_exchange: str = '',
) -> str:
    """Language-aware prompt builder: 'ru' (default), 'hy' (Armenian), 'zh' (Chinese)."""
    lang = _lang2(plan)
    parts: list[str] = []

    if lang == 'hy':
        if persona_voice:
            parts.append(f"Ду — {persona_voice.strip()[:300]}")
        if recent_exchange:
            parts.append(f"Хоскернал карнак:\n{recent_exchange.strip()[:400]}")
        instr = _ACTION_TO_HY_INSTRUCTION.get(plan.action_name, 'Патасхани бунаканабар.')
        length_hint = '1–2 нахадас' if plan.max_tokens <= 80 else ('2–3 нахадас' if plan.max_tokens <= 140 else '3–4 нахадас')
        tension_note = _TENSION_HIGH.get('hy', '') if plan.conflict_tension > 0.7 else (
            _TENSION_MID.get('hy', '') if plan.conflict_tension > 0.45 else '')
        if user_message:
            parts.append(f'Марде грел е: «{user_message.strip()}»')
        instr_block = f'{instr} Еркарутюн: {length_hint}.{tension_note}'
        parts.append(instr_block)
        if plan.key_points:
            parts.append('Андхаманапес арцанацрир:\n' + '\n'.join(f'— {p}' for p in plan.key_points[:3]))
        parts.append('Патасхане:')

    elif lang == 'zh':
        if persona_voice:
            parts.append(f"你是 — {persona_voice.strip()[:300]}")
        if recent_exchange:
            parts.append(f"对话背景:\n{recent_exchange.strip()[:400]}")
        instr = _ACTION_TO_ZH_INSTRUCTION.get(plan.action_name, '自然地回应。')
        length_hint = '1–2句' if plan.max_tokens <= 80 else ('2–3句' if plan.max_tokens <= 140 else '3–4句')
        tension_note = _TENSION_HIGH.get('zh', '') if plan.conflict_tension > 0.7 else (
            _TENSION_MID.get('zh', '') if plan.conflict_tension > 0.45 else '')
        if user_message:
            parts.append(f'对方说：「{user_message.strip()}」')
        zh_tone = _TONE_TO_ZH.get(plan.dominant_resolution if hasattr(plan, 'dominant_resolution') else '', '')
        instr_block = f'{instr} 长度：{length_hint}。{tension_note}'
        if zh_tone:
            instr_block += f' 语气：{zh_tone}。'
        parts.append(instr_block)
        if plan.key_points:
            parts.append('必须体现：\n' + '\n'.join(f'— {p}' for p in plan.key_points[:3]))
        parts.append('回答：')

    else:
        # Russian (default for 'ru', 'en', and any other lang)
        if persona_voice:
            parts.append(f"Ты — {persona_voice.strip()[:300]}")
        if recent_exchange:
            parts.append(f"Контекст разговора:\n{recent_exchange.strip()[:400]}")
        instr = _ACTION_TO_RU_INSTRUCTION.get(plan.action_name, 'Ответь естественно.')
        length_hint = '1–2 предложения' if plan.max_tokens <= 80 else ('2–3 предложения' if plan.max_tokens <= 140 else '3–4 предложения')
        tension_note = _TENSION_HIGH.get('ru', '') if plan.conflict_tension > 0.7 else (
            _TENSION_MID.get('ru', '') if plan.conflict_tension > 0.45 else '')
        if user_message:
            parts.append(f'Человек написал: «{user_message.strip()}»')
        ru_tone = _TONE_TO_RU.get(plan.dominant_resolution if hasattr(plan, 'dominant_resolution') else '', '')
        instr_block = f'{instr} Длина: {length_hint}.{tension_note}'
        if ru_tone:
            instr_block += f' Тон: {ru_tone}.'
        parts.append(instr_block)
        if plan.key_points:
            parts.append('Обязательно выдели:\n' + '\n'.join(f'— {p}' for p in plan.key_points[:3]))
        parts.append('Ответ:')

    return '\n\n'.join(parts)


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

    # Closing instruction adapts to target language
    if plan.language and plan.language != 'en':
        user_parts.append(
            f"Напиши ответ прямо сейчас на языке: {plan.language}. "
            "Следуй директиве точно. Пиши только живую речь персонажа. "
            "Не объясняй свои рассуждения. Не ссылайся на директиву. Не выходи из образа."
        )
    else:
        user_parts.append(
            "Write the reply now. Follow the directive exactly. "
            "Use natural speech for the persona. "
            "Do not explain your reasoning or reference this directive. "
            "Do not break character."
        )

    user_block = '\n\n'.join(user_parts)

    if system_parts:
        # Use the "User question:" separator so _split_chat_prompt_messages puts
        # PERSONA + RECENT EXCHANGE in the system role and the directive in the user role.
        return '\n\n'.join(system_parts) + '\n\nUser question:\n' + user_block

    return user_block
