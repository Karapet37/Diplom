"""
Planning engine.

Implements PLANNING MODE — the primary product mode.

The planning loop:
    user message (report / question / reflection)
    → classify the turn type
    → load notes + personality summary + recent outcomes
    → compute what matters now (forces, conflicts, overload)
    → select ONE small next step
    → select ONE fallback smaller step
    → select ONE follow-up question for feedback
    → return PlanningOutput

The LLM is used ONLY to verbalize the selected output.
The system computes the structure first.

Key principles:
- Plans are hypotheses, not orders
- Steps must be tiny enough to be testable
- Overload suppresses step intensity
- Crisis signals suppress planning entirely
- Feedback from previous steps must update the model

Feedback classification:
    'success'     → step worked, reinforce
    'partial'     → step partially worked, reduce size
    'failure'     → step failed, switch to fallback or re-assess
    'resistance'  → user resisted/modified plan
    'crisis'      → stop planning, switch to support
    'neutral'     → no clear signal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .notes_store import format_notes_for_context, load_notes
from .personality_store import load_personality_for_persona
from .personality_summarizer import generate_compact_summary
from .runtime_config import get_runtime_config


# ---------------------------------------------------------------------------
# Crisis detection
# ---------------------------------------------------------------------------

_CRISIS_MARKERS = (
    'suicide',
    'суицид',
    'kill myself',
    'убью себя',
    'end my life',
    'хочу умереть',
    'want to die',
    'can\'t go on',
    'не могу больше',
    'collapse',
    'breakdown',
    'i\'m losing control',
    'теряю контроль',
    'self-harm',
    'самоповреждение',
    'hurt myself',
    'причиняю себе вред',
    'no reason to live',
    'нет смысла жить',
    'hopeless',
    'безнадёжно',
    'helpless',
    'беспомощен',
    'overwhelmed completely',
    'полный распад',
)

_CRISIS_RESPONSE = (
    'This sounds like a very difficult moment. '
    'Please stop any plans for now — that\'s not what matters right now.\n\n'
    'If you\'re in danger or feel you may harm yourself, please contact a crisis line '
    'or someone you trust immediately.\n\n'
    'I\'m here to listen. What is happening right now?'
)

_CRISIS_RESPONSE_RU = (
    'Это звучит как очень тяжёлый момент. '
    'Планы сейчас не важны — сначала позаботьтесь о себе.\n\n'
    'Если вы в опасности или думаете о том, чтобы причинить себе вред — '
    'пожалуйста, свяжитесь с телефоном доверия или человеком, которому доверяете.\n\n'
    'Я слушаю. Что происходит прямо сейчас?'
)


def detect_crisis(text: str) -> bool:
    """Return True if message contains crisis signals."""
    lowered = str(text or '').lower()
    return any(marker in lowered for marker in _CRISIS_MARKERS)


# ---------------------------------------------------------------------------
# Feedback classification
# ---------------------------------------------------------------------------

_FEEDBACK_SUCCESS_MARKERS = (
    'i did it',
    'it worked',
    'я сделал',
    'получилось',
    'succeeded',
    'it helped',
    'помогло',
    'worked well',
    'сработало',
    'made progress',
    'продвинулся',
    'done',
    'completed',
    'сделано',
)
_FEEDBACK_FAILURE_MARKERS = (
    'i failed',
    'didn\'t work',
    'провалился',
    'не получилось',
    'couldn\'t do it',
    'не смог',
    'gave up',
    'сдался',
    'went wrong',
    'пошло не так',
    'didn\'t help',
    'не помогло',
    'worse',
    'стало хуже',
)
_FEEDBACK_PARTIAL_MARKERS = (
    'kind of',
    'partially',
    'somewhat',
    'not quite',
    'partly',
    'частично',
    'не совсем',
    'почти',
    'almost',
)
_FEEDBACK_RESISTANCE_MARKERS = (
    'i modified',
    'i changed',
    'i did it differently',
    'that\'s not how i did it',
    'я изменил',
    'сделал по-другому',
    'не совсем так',
    'адаптировал',
)
_FEEDBACK_OVERLOAD_MARKERS = (
    'too much',
    'слишком много',
    'overwhelmed',
    'перегружен',
    'exhausted',
    'истощён',
    'burned out',
    'выгорел',
    'can\'t keep up',
    'не успеваю',
    'too hard',
    'слишком тяжело',
)


FeedbackClass = str  # 'success' | 'partial' | 'failure' | 'resistance' | 'overload' | 'neutral'


def classify_feedback(text: str) -> FeedbackClass:
    """
    Classify a user message as feedback on a previous step.
    Returns a FeedbackClass string.
    """
    if detect_crisis(text):
        return 'crisis'
    lowered = str(text or '').lower()
    if any(m in lowered for m in _FEEDBACK_OVERLOAD_MARKERS):
        return 'overload'
    if any(m in lowered for m in _FEEDBACK_SUCCESS_MARKERS):
        return 'success'
    if any(m in lowered for m in _FEEDBACK_FAILURE_MARKERS):
        return 'failure'
    if any(m in lowered for m in _FEEDBACK_RESISTANCE_MARKERS):
        return 'resistance'
    if any(m in lowered for m in _FEEDBACK_PARTIAL_MARKERS):
        return 'partial'
    return 'neutral'


# ---------------------------------------------------------------------------
# Planning intent detection
# ---------------------------------------------------------------------------

_PLANNING_INTENT_MARKERS = (
    '/plan',
    'what should i do',
    'what next',
    'что делать',
    'что дальше',
    'help me plan',
    'помоги спланировать',
    'next step',
    'следующий шаг',
    'make a plan',
    'составь план',
    'how to proceed',
    'как продолжить',
    'i need a plan',
    'нужен план',
    'guide me',
    'направь меня',
    'what\'s my next move',
    'мой следующий шаг',
    'i don\'t know what to do',
    'не знаю что делать',
    'i\'m stuck',
    'застрял',
    'я застрял',
)


def detect_planning_intent(message: str) -> bool:
    """Return True if the user is asking for planning mode."""
    lowered = str(message or '').lower()
    return any(marker in lowered for marker in _PLANNING_INTENT_MARKERS)


# ---------------------------------------------------------------------------
# Overload level
# ---------------------------------------------------------------------------

def estimate_overload_level(
    recent_feedback: list[FeedbackClass],
    temp_state_values: list[str],
) -> float:
    """
    Estimate overload level as 0.0–1.0.

    Based on:
    - Recent feedback history (failures, overload signals)
    - Temporary state entries mentioning stress / exhaustion
    """
    score = 0.0
    for fb in recent_feedback[-5:]:
        if fb == 'overload':
            score += 0.35
        elif fb == 'failure':
            score += 0.15
        elif fb == 'partial':
            score += 0.08
        elif fb == 'resistance':
            score += 0.12
        elif fb == 'success':
            score = max(0.0, score - 0.1)
    overload_keywords = (
        'stress', 'exhausted', 'overwhelmed', 'burned out', 'strained',
        'стресс', 'истощён', 'перегружен', 'выгорел', 'напряжение',
    )
    for val in temp_state_values:
        if any(kw in val.lower() for kw in overload_keywords):
            score += 0.2
    return min(1.0, max(0.0, score))


# ---------------------------------------------------------------------------
# Step intensity selection
# ---------------------------------------------------------------------------

def select_step_intensity(overload: float) -> str:
    """
    Map overload level to a step intensity label.

    0.0–0.3   → normal
    0.3–0.6   → gentle
    0.6–0.8   → minimal
    0.8–1.0   → micro (tiny, ultra-safe)
    """
    if overload >= 0.8:
        return 'micro'
    if overload >= 0.6:
        return 'minimal'
    if overload >= 0.3:
        return 'gentle'
    return 'normal'


# ---------------------------------------------------------------------------
# Planning output
# ---------------------------------------------------------------------------

@dataclass
class PlanningOutput:
    """
    Structured output of the planning engine.

    Fields are filled by the system (NOT invented by the LLM).
    The LLM only verbalizes these fields into natural language.
    """
    what_matters: str = ''              # What seems most important right now
    active_forces: list[str] = field(default_factory=list)  # Forces in tension
    next_step: str = ''                 # ONE small next action
    fallback_step: str = ''             # ONE even smaller fallback if next_step fails
    follow_up_question: str = ''        # What to ask next turn for feedback
    overload_level: float = 0.0
    step_intensity: str = 'normal'
    feedback_class: str = 'neutral'
    is_crisis: bool = False
    crisis_response: str = ''
    personality_summary: str = ''
    notes_used: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'what_matters': self.what_matters,
            'active_forces': list(self.active_forces),
            'next_step': self.next_step,
            'fallback_step': self.fallback_step,
            'follow_up_question': self.follow_up_question,
            'overload_level': round(float(self.overload_level), 3),
            'step_intensity': self.step_intensity,
            'feedback_class': self.feedback_class,
            'is_crisis': bool(self.is_crisis),
            'crisis_response': self.crisis_response,
            'personality_summary': self.personality_summary,
            'notes_used': self.notes_used,
        }

    def render_planning_prompt_section(self) -> str:
        """
        Render the planning context as a prompt section
        for injection into the LLM verbalizer.
        """
        if self.is_crisis:
            return ''  # Crisis is handled directly, not via LLM planning

        parts: list[str] = []

        if self.notes_used:
            parts.append(f'Important saved notes:\n{self.notes_used}')

        if self.personality_summary:
            parts.append(f'Current personality model summary:\n{self.personality_summary}')

        instructions = [
            'Respond briefly and concretely. DO NOT produce a long essay.',
            f'Feedback on previous step: {self.feedback_class}' if self.feedback_class != 'neutral' else '',
            f'Current overload level: {self.step_intensity} (scale: micro < minimal < gentle < normal)',
            '',
            'Planning output — respond with EXACTLY these sections:',
            '1. What matters now: [1-2 sentences]',
            '2. Forces in conflict: [2-3 bullet points or "none identified"]',
            f'3. Next micro-step ({self.step_intensity} intensity): [1 sentence — specific and testable]',
            '4. Fallback smaller step: [1 sentence — even smaller, safer]',
            '5. Follow-up question: [1 question to ask next turn for feedback]',
            '',
            'Rules:',
            '- Steps must be tiny and concrete',
            '- Do not shame or moralize',
            '- Do not push through clear resistance',
            '- Acknowledge uncertainty in the model',
        ]
        parts.append('\n'.join(s for s in instructions if s or True))
        return '\n\n'.join(parts)


# ---------------------------------------------------------------------------
# Core planning logic
# ---------------------------------------------------------------------------

def build_planning_output(
    *,
    message: str,
    session_id: str,
    personality_name: str = '',
    recent_feedback: list[FeedbackClass] | None = None,
    language: str = 'en',
) -> PlanningOutput:
    """
    Build a PlanningOutput from message + context.

    This function DOES NOT call the LLM.
    It assembles structure, computes overload, selects intensity.
    The caller uses render_planning_prompt_section() to inject into LLM prompt.
    """
    recent_feedback = list(recent_feedback or [])
    output = PlanningOutput()

    # Crisis check — highest priority
    is_crisis = detect_crisis(message)
    output.is_crisis = is_crisis
    if is_crisis:
        if language == 'ru':
            output.crisis_response = _CRISIS_RESPONSE_RU
        else:
            output.crisis_response = _CRISIS_RESPONSE
        return output

    # Classify feedback from current message
    output.feedback_class = classify_feedback(message)
    recent_feedback.append(output.feedback_class)

    # Load personality for overload estimation
    temp_state_values: list[str] = []
    personality_summary = ''
    if personality_name:
        personality = load_personality_for_persona(personality_name)
        if personality is not None:
            # Extract temporary state values for overload estimation
            from .personality_schema import TemporaryStateEntry
            # personality doesn't have temp_state directly; it's session-scoped
            # We extract from profile fields that are marked is_temporary
            for field_name in personality.profile.all_field_names():
                for entry in personality.profile.get_field(field_name):
                    if entry.is_temporary:
                        temp_state_values.append(entry.value)
            # Generate compact summary
            try:
                personality_summary = generate_compact_summary(personality)
            except Exception:
                personality_summary = ''

    # Estimate overload
    overload = estimate_overload_level(recent_feedback, temp_state_values)
    output.overload_level = overload
    output.step_intensity = select_step_intensity(overload)
    output.personality_summary = personality_summary

    # Load saved notes
    notes_text = format_notes_for_context(session_id)
    output.notes_used = notes_text

    # Active forces — derived from personality profile
    if personality_name:
        personality = load_personality_for_persona(personality_name)
        if personality is not None:
            forces: list[str] = []
            # Goals vs fears tension
            goals = [e.value for e in personality.profile.get_field('core_goals') if not e.is_temporary][:2]
            fears = [e.value for e in personality.profile.get_field('fears') if not e.is_temporary][:2]
            for g in goals:
                forces.append(f'drive toward: {g}')
            for f in fears:
                forces.append(f'fear of: {f}')
            # Internal conflicts
            conflicts = [e.value for e in personality.profile.get_field('internal_conflicts') if not e.is_temporary][:2]
            for c in conflicts:
                forces.append(f'tension: {c}')
            output.active_forces = forces[:6]

    return output


# ---------------------------------------------------------------------------
# Planning prompt builder
# ---------------------------------------------------------------------------

def build_planning_prompt(
    *,
    message: str,
    planning_output: PlanningOutput,
    language: str = 'en',
) -> str:
    """
    Build the full LLM prompt for planning mode verbalization.

    The LLM receives:
    - The user's message
    - The planning context (notes, personality summary, overload level)
    - Instructions to fill out the 5-section output format

    The LLM must NOT invent new structure.
    It must fill the given sections with natural language.
    """
    if planning_output.is_crisis:
        # No LLM needed — return canned response
        return ''

    context_section = planning_output.render_planning_prompt_section()

    lang_instruction = ''
    if language == 'ru':
        lang_instruction = 'Respond in Russian.\n\n'

    prompt = f"""{lang_instruction}You are a local private planning assistant. Your job is to help the user take the smallest useful next step.

User request:
{message.strip()}

{context_section}

Remember: you are modeling a hypothesis, not issuing commands. Use hedged language ("it might help to...", "one option is..."). Do not pretend to know everything about the user.
"""
    return prompt.strip()
