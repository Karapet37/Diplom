"""
Situation regulator.

Implements the event → regulator → action architecture described in the spec.

Architecture:
    Layer 1 — Event Classifier
        Classify what kind of situation this is.
        NOT an action selector — only event labeling.

    Layer 2 — Regulator Release Policy
        Rule-based mapping: event type → regulator state changes.
        Uses personality modifiers to shape intensity.
        NOT an action selector — only regulator state update.

    Layer 3 — Action Selector
        Weighted scoring based on personality forces + regulator state.
        Chooses a behavioral response FAMILY.
        The LLM verbalizes this choice.

    Layer 4 — LLM Verbalizer  (in chat_engine.py)
        Receives the selected action + all context.
        Produces the final text response.

Regulators are NOT simulated endocrinology.
They are behaviorally useful internal state variables.

Event types:
    threat           → cortisol↑, adrenaline↑
    reward_signal    → dopamine↑
    social_shame     → cortisol↑, serotonin↓, testosterone↑
    warm_support     → oxytocin↑, cortisol↓
    challenge        → adrenaline↑, testosterone↑, dopamine↑
    failure          → cortisol↑, dopamine↓
    social_rejection → cortisol↑, serotonin↓, oxytocin↓
    attachment_activation → oxytocin↑, cortisol variable
    overload         → cortisol↑, fatigue↑, adrenaline↓
    novelty          → dopamine↑, adrenaline↑
    neutral          → no significant change
    planning_request → (no significant hormonal change)
    feedback_report  → depends on content
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

EventType = str  # one of the strings below

EVENT_TYPES = (
    'threat',
    'reward_signal',
    'social_shame',
    'warm_support',
    'challenge',
    'failure',
    'social_rejection',
    'attachment_activation',
    'overload',
    'novelty',
    'planning_request',
    'feedback_report',
    'neutral',
    'crisis',
)

# ---------------------------------------------------------------------------
# Event classification — rule-based marker system
# ---------------------------------------------------------------------------

_EVENT_MARKERS: dict[str, tuple[str, ...]] = {
    'crisis': (
        'suicide', 'суицид', 'kill myself', 'убью себя', 'end my life',
        'хочу умереть', 'want to die', 'can\'t go on', 'не могу больше',
        'self-harm', 'hurt myself', 'no reason to live', 'нет смысла жить',
        'hopeless', 'безнадёжно', 'breakdown', 'collapse',
    ),
    'threat': (
        'threat', 'угроза', 'danger', 'опасность', 'afraid', 'боюсь',
        'scared', 'испуган', 'at risk', 'под угрозой', 'warning', 'предупреждение',
        'будут последствия', 'consequences', 'punish', 'наказание',
    ),
    'social_shame': (
        'ashamed', 'стыжусь', 'shame', 'стыд', 'embarrassed', 'смущён',
        'humiliated', 'унижен', 'rejected', 'отвергнут', 'judged', 'осуждён',
        'felt stupid', 'чувствую себя идиотом', 'looked down', 'смотрят свысока',
    ),
    'warm_support': (
        'support', 'поддержка', 'thank you', 'спасибо', 'helped me',
        'помог мне', 'i feel understood', 'меня понимают', 'comfort', 'утешение',
        'caring', 'заботливый', 'safe', 'безопасно', 'i trust', 'я доверяю',
    ),
    'challenge': (
        'challenge', 'вызов', 'try', 'попробую', 'attempt', 'попытаюсь',
        'push myself', 'напрягусь', 'go for it', 'пойду на это', 'test myself',
        'проверю себя', 'compete', 'соревнование', 'prove', 'докажу',
    ),
    'failure': (
        'failed', 'провалился', 'didn\'t work', 'не получилось',
        'gave up', 'сдался', 'couldn\'t', 'не смог', 'lost', 'проиграл',
        'mistake', 'ошибка', 'went wrong', 'пошло не так',
    ),
    'social_rejection': (
        'rejected', 'отвергнут', 'ignored', 'игнорируют', 'excluded',
        'исключён', 'alone', 'одинок', 'nobody cares', 'никому не важен',
        'left out', 'забытый', 'abandoned', 'брошен',
    ),
    'attachment_activation': (
        'love', 'люблю', 'miss', 'скучаю', 'close to', 'близок с',
        'bonded', 'связан', 'trust deeply', 'глубоко доверяю',
        'attachment', 'привязанность', 'care about', 'беспокоюсь о',
    ),
    'overload': (
        'too much', 'слишком много', 'overwhelmed', 'перегружен',
        'exhausted', 'истощён', 'burned out', 'выгорел', 'can\'t keep up',
        'не успеваю', 'too hard', 'слишком тяжело', 'too fast', 'слишком быстро',
    ),
    'reward_signal': (
        'succeeded', 'успех', 'did it', 'сделал', 'worked', 'получилось',
        'accomplished', 'достиг', 'proud', 'горжусь', 'happy', 'счастлив',
        'satisfied', 'доволен', 'reward', 'награда', 'progress', 'прогресс',
    ),
    'novelty': (
        'new', 'новый', 'interesting', 'интересно', 'curious', 'любопытно',
        'discovery', 'открытие', 'surprise', 'удивление', 'first time',
        'впервые', 'never seen', 'никогда не видел', 'unexpected', 'неожиданно',
    ),
    'planning_request': (
        '/plan', 'what next', 'что дальше', 'next step', 'следующий шаг',
        'help me plan', 'помоги спланировать', 'what should i do', 'что делать',
    ),
    'feedback_report': (
        'i did it', 'я сделал', 'it helped', 'помогло', 'failed', 'провалился',
        'tried', 'попробовал', 'result', 'результат', 'update', 'обновление',
        'reporting back', 'отчитываюсь', 'outcome', 'итог',
    ),
}

# ---------------------------------------------------------------------------
# Empathetic Dialogues emotion → event_type mapping
# Source: DialogStudio / EmpatheticDialogues (32 emotion labels)
# Maps the dataset's emotion context labels to our event taxonomy.
# Used to enrich event classification when emotion context is known.
# ---------------------------------------------------------------------------

EMPATHETIC_EMOTION_TO_EVENT: dict[str, EventType] = {
    # Crisis-adjacent
    'terrified':         'threat',
    'devastated':        'failure',
    'hopeful':           'challenge',
    # Threat / fear
    'afraid':            'threat',
    'anxious':           'threat',
    'apprehensive':      'threat',
    'impressed':         'novelty',
    # Social shame / rejection
    'ashamed':           'social_shame',
    'embarrassed':       'social_shame',
    'guilty':            'social_shame',
    'lonely':            'social_rejection',
    'jealous':           'social_rejection',
    'furious':           'threat',
    'annoyed':           'threat',
    'disgusted':         'social_shame',
    # Failure / loss
    'sad':               'failure',
    'disappointed':      'failure',
    'nostalgic':         'attachment_activation',
    'sentimental':       'attachment_activation',
    'grieving':          'failure',
    # Warm / positive
    'caring':            'warm_support',
    'grateful':          'warm_support',
    'trusting':          'warm_support',
    'content':           'reward_signal',
    'faithful':          'warm_support',
    # Achievement / energy
    'proud':             'reward_signal',
    'excited':           'challenge',
    'joyful':            'reward_signal',
    'anticipating':      'planning_request',
    'surprised':         'novelty',
    'prepared':          'planning_request',
    'confident':         'challenge',
}

# Priority order for event classification (earlier = higher priority)
_EVENT_PRIORITY = (
    'crisis',
    'overload',
    'social_shame',
    'social_rejection',
    'threat',
    'failure',
    'warm_support',
    'challenge',
    'attachment_activation',
    'reward_signal',
    'planning_request',
    'feedback_report',
    'novelty',
    'neutral',
)


def classify_event(message: str, *, emotion_context: str = '') -> EventType:
    """
    Classify the primary event type from a message.

    If emotion_context is provided (e.g. from Empathetic Dialogues label or
    a detected emotion tag), it is used as a tiebreaker when the rule-based
    classifier returns 'neutral'.

    Returns the highest-priority matching event type.
    """
    lowered = str(message or '').lower()
    for event in _EVENT_PRIORITY:
        markers = _EVENT_MARKERS.get(event, ())
        if any(m in lowered for m in markers):
            return event

    # Fallback: use emotion_context mapping (Empathetic Dialogues taxonomy)
    if emotion_context:
        mapped = EMPATHETIC_EMOTION_TO_EVENT.get(emotion_context.lower().strip())
        if mapped:
            return mapped

    return 'neutral'


def classify_event_from_emotion(emotion_label: str) -> EventType:
    """
    Map an Empathetic Dialogues emotion label directly to an event type.
    Deterministic lookup. Returns 'neutral' for unknown labels.
    """
    return EMPATHETIC_EMOTION_TO_EVENT.get(
        str(emotion_label or '').lower().strip(), 'neutral'
    )


def classify_event_multi(message: str, *, top_n: int = 3) -> list[EventType]:
    """
    Return up to top_n matching event types, in priority order.
    """
    lowered = str(message or '').lower()
    results: list[str] = []
    for event in _EVENT_PRIORITY:
        if event == 'neutral':
            continue
        markers = _EVENT_MARKERS.get(event, ())
        if any(m in lowered for m in markers):
            results.append(event)
            if len(results) >= top_n:
                break
    if not results:
        results = ['neutral']
    return results


# ---------------------------------------------------------------------------
# Regulator state
# ---------------------------------------------------------------------------

@dataclass
class RegulatorState:
    """
    Internal regulator state.

    Values are in range [0.0, 1.0].
    Baseline is 0.5 for most regulators (neutral/resting state).
    """
    dopamine: float = 0.5        # reward anticipation, motivation
    cortisol: float = 0.5        # stress, threat response
    serotonin: float = 0.5       # mood stability, belonging
    adrenaline: float = 0.3      # arousal, urgency, action readiness
    oxytocin: float = 0.5        # social bonding, trust
    testosterone: float = 0.5    # dominance, challenge drive
    fatigue: float = 0.3         # depletion (inverse of energy)

    def to_dict(self) -> dict[str, Any]:
        return {
            'dopamine': round(self.dopamine, 3),
            'cortisol': round(self.cortisol, 3),
            'serotonin': round(self.serotonin, 3),
            'adrenaline': round(self.adrenaline, 3),
            'oxytocin': round(self.oxytocin, 3),
            'testosterone': round(self.testosterone, 3),
            'fatigue': round(self.fatigue, 3),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'RegulatorState':
        obj = cls()
        for key in ('dopamine', 'cortisol', 'serotonin', 'adrenaline',
                    'oxytocin', 'testosterone', 'fatigue'):
            val = data.get(key)
            if val is not None:
                setattr(obj, key, float(val))
        return obj

    def clone(self) -> 'RegulatorState':
        return RegulatorState(
            dopamine=self.dopamine,
            cortisol=self.cortisol,
            serotonin=self.serotonin,
            adrenaline=self.adrenaline,
            oxytocin=self.oxytocin,
            testosterone=self.testosterone,
            fatigue=self.fatigue,
        )

    @classmethod
    def baseline(cls) -> 'RegulatorState':
        return cls()


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


# ---------------------------------------------------------------------------
# Regulator release policy (Layer 2)
# ---------------------------------------------------------------------------

# Release profiles: event_type → {regulator: delta}
# Positive delta = increase, negative = decrease
# Personality intensity_modifier scales these deltas

_BASE_RELEASE_PROFILES: dict[str, dict[str, float]] = {
    'threat': {
        'cortisol': +0.25,
        'adrenaline': +0.20,
        'dopamine': -0.05,
        'serotonin': -0.08,
        'testosterone': +0.10,
    },
    'reward_signal': {
        'dopamine': +0.25,
        'serotonin': +0.10,
        'cortisol': -0.10,
        'fatigue': -0.08,
    },
    'social_shame': {
        'cortisol': +0.30,
        'serotonin': -0.20,
        'testosterone': +0.15,
        'oxytocin': -0.10,
        'dopamine': -0.10,
    },
    'warm_support': {
        'oxytocin': +0.25,
        'cortisol': -0.20,
        'serotonin': +0.15,
        'dopamine': +0.10,
        'fatigue': -0.05,
    },
    'challenge': {
        'adrenaline': +0.20,
        'testosterone': +0.20,
        'dopamine': +0.15,
        'cortisol': +0.08,
        'fatigue': -0.05,
    },
    'failure': {
        'cortisol': +0.20,
        'dopamine': -0.20,
        'serotonin': -0.10,
        'fatigue': +0.15,
        'testosterone': -0.05,
    },
    'social_rejection': {
        'cortisol': +0.25,
        'serotonin': -0.20,
        'oxytocin': -0.20,
        'dopamine': -0.10,
        'testosterone': +0.10,
    },
    'attachment_activation': {
        'oxytocin': +0.20,
        'dopamine': +0.10,
        'cortisol': -0.05,
        'serotonin': +0.08,
    },
    'overload': {
        'cortisol': +0.30,
        'fatigue': +0.30,
        'adrenaline': -0.10,
        'dopamine': -0.15,
        'serotonin': -0.10,
    },
    'novelty': {
        'dopamine': +0.20,
        'adrenaline': +0.10,
        'serotonin': +0.05,
        'cortisol': +0.05,
    },
    'planning_request': {},   # No significant hormonal change
    'feedback_report': {},    # Handled by feedback_class in planning engine
    'neutral': {},
    'crisis': {
        'cortisol': +0.40,
        'adrenaline': +0.25,
        'dopamine': -0.30,
        'serotonin': -0.25,
        'oxytocin': -0.10,
        'fatigue': +0.20,
    },
}


def apply_release_profile(
    state: RegulatorState,
    event_type: EventType,
    *,
    intensity_modifier: float = 1.0,
) -> RegulatorState:
    """
    Apply the release profile for an event to the regulator state.

    intensity_modifier: personality-derived amplifier (0.5–1.5 typically).
    Returns a new RegulatorState (does not mutate in place).
    """
    profile = _BASE_RELEASE_PROFILES.get(event_type, {})
    new_state = state.clone()
    modifier = max(0.1, min(2.0, float(intensity_modifier or 1.0)))
    for reg, delta in profile.items():
        current = getattr(new_state, reg, 0.5)
        setattr(new_state, reg, _clamp(current + delta * modifier))
    # Natural decay toward baseline (partial regression per turn)
    _decay_toward_baseline(new_state, rate=0.05)
    return new_state


def _decay_toward_baseline(state: RegulatorState, *, rate: float = 0.05) -> None:
    """Partially decay all regulators toward their baseline (0.5), except fatigue."""
    baseline = 0.5
    for attr in ('dopamine', 'cortisol', 'serotonin', 'adrenaline', 'oxytocin', 'testosterone'):
        current = getattr(state, attr, 0.5)
        new_val = current + (baseline - current) * rate
        setattr(state, attr, _clamp(new_val))
    # Fatigue decays toward 0.3 baseline
    state.fatigue = _clamp(state.fatigue + (0.3 - state.fatigue) * rate)


# ---------------------------------------------------------------------------
# Intensity modifier from personality
# ---------------------------------------------------------------------------

def compute_intensity_modifier(
    *,
    pressure_response: list[str],
    vulnerabilities: list[str],
    defense_mechanisms: list[str],
    event_type: EventType,
) -> float:
    """
    Compute a personality-derived intensity modifier for regulator release.

    High-sensitivity personalities amplify certain events.
    High-defense personalities dampen certain events.

    Returns float in [0.4, 1.8].
    """
    base = 1.0

    # Threat/shame sensitivity from vulnerabilities
    if event_type in ('threat', 'social_shame', 'social_rejection'):
        for v in vulnerabilities:
            v_lower = v.lower()
            if any(kw in v_lower for kw in ('sensitive', 'чувствительн', 'shame', 'стыд', 'rejection', 'отвержение')):
                base += 0.2
                break

    # Challenge amplifier from pressure_response
    if event_type == 'challenge':
        for p in pressure_response:
            p_lower = p.lower()
            if any(kw in p_lower for kw in ('competitive', 'конкурент', 'prove', 'доказ', 'fight back', 'бороться')):
                base += 0.25
                break

    # Defense dampener
    if defense_mechanisms:
        for d in defense_mechanisms:
            d_lower = d.lower()
            if any(kw in d_lower for kw in ('dissociation', 'диссоциаци', 'numbing', 'онемение', 'suppression', 'подавлени')):
                base -= 0.3
                break

    return max(0.4, min(1.8, base))


# ---------------------------------------------------------------------------
# Action scoring (Layer 3)
# ---------------------------------------------------------------------------

# Action families relevant to the coaching/planning context
PLANNING_ACTION_FAMILIES = (
    'plan_small_step',          # Propose a concrete small step
    'acknowledge_and_support',  # Validate feelings, provide emotional support
    'ask_clarifying_question',  # Probe for more information
    'celebrate_progress',       # Reinforce success
    'reduce_step_size',         # Propose an even smaller step after failure
    'suggest_environment_change', # Change environment to reduce friction
    'de_escalate',              # Crisis / overload de-escalation
    'reflect_back',             # Mirror back what was said without judgment
    'reality_check',            # Gently challenge distorted thinking
    'explore_resistance',       # Ask about resistance / modification of plan
    'provide_information',      # Factual answer to a question
)


@dataclass
class ActionScore:
    action: str
    score: float
    blocked: bool = False
    block_reason: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'action': self.action,
            'score': round(float(self.score), 4),
            'blocked': bool(self.blocked),
            'block_reason': self.block_reason,
        }


def score_action(
    action: str,
    *,
    regulator_state: RegulatorState,
    event_type: EventType,
    overload_level: float = 0.0,
    feedback_class: str = 'neutral',
    message: str = '',
) -> ActionScore:
    """
    Score a single action given the current regulator state and context.

    Scoring logic:
    - de_escalate is always top when crisis/overload
    - acknowledge_and_support rises with cortisol and low serotonin
    - plan_small_step rises with dopamine, falls with high cortisol
    - celebrate_progress rises when dopamine is high and feedback is success
    - reduce_step_size rises after failure
    - suggest_environment_change rises with repeated failure or high fatigue
    - explore_resistance rises when feedback is resistance
    - reflect_back is a safe moderate action when unsure
    """
    rs = regulator_state
    score = 0.0
    blocked = False
    block_reason = ''

    if action == 'de_escalate':
        if event_type in ('crisis', 'overload') or overload_level > 0.8:
            score = 10.0  # Always wins in crisis
        else:
            score = overload_level * 2.0

    elif action == 'acknowledge_and_support':
        score = rs.cortisol * 0.8 + (1.0 - rs.serotonin) * 0.6 + rs.oxytocin * 0.3
        if event_type in ('social_shame', 'social_rejection', 'warm_support'):
            score += 0.8

    elif action == 'plan_small_step':
        score = rs.dopamine * 0.7 + (1.0 - rs.cortisol) * 0.4 + (1.0 - overload_level) * 0.5
        if event_type == 'planning_request':
            score += 1.0
        if event_type in ('crisis', 'overload') or overload_level > 0.7:
            blocked = True
            block_reason = 'overload or crisis blocks planning'

    elif action == 'celebrate_progress':
        score = rs.dopamine * 0.8 + rs.serotonin * 0.5
        if feedback_class == 'success':
            score += 1.5

    elif action == 'reduce_step_size':
        score = (1.0 - rs.dopamine) * 0.5 + rs.fatigue * 0.6
        if feedback_class in ('failure', 'partial'):
            score += 1.2
        if event_type == 'overload':
            score += 0.8

    elif action == 'suggest_environment_change':
        score = rs.fatigue * 0.5 + (1.0 - rs.dopamine) * 0.4
        if feedback_class == 'failure':
            score += 0.6

    elif action == 'explore_resistance':
        score = 0.3
        if feedback_class == 'resistance':
            score += 1.5

    elif action == 'ask_clarifying_question':
        # Base score boosted by clarification need score from the engine
        # Imported lazily to avoid circular dependency
        try:
            from .clarification_engine import score_clarification_need as _scn
            clarification_need = _scn(message)
        except Exception:
            clarification_need = 0.0
        score = 0.3 + rs.oxytocin * 0.3 + clarification_need * 0.6
        if event_type == 'neutral':
            score += 0.3
        # Suppress clarification when user is in distress — they need support, not questions
        if rs.cortisol > 0.70 or event_type in ('crisis', 'overload', 'social_shame'):
            score *= 0.2

    elif action == 'reflect_back':
        score = 0.5 + (1.0 - rs.cortisol) * 0.3

    elif action == 'reality_check':
        score = rs.testosterone * 0.3 + (1.0 - rs.cortisol) * 0.3
        # Don't reality-check when cortisol is very high (shame/threat state)
        if rs.cortisol > 0.75:
            score *= 0.3

    elif action == 'provide_information':
        score = 0.3 + (1.0 - rs.cortisol) * 0.2

    return ActionScore(action=action, score=score, blocked=blocked, block_reason=block_reason)


def select_action(
    *,
    regulator_state: RegulatorState,
    event_type: EventType,
    overload_level: float = 0.0,
    feedback_class: str = 'neutral',
    available_actions: tuple[str, ...] = PLANNING_ACTION_FAMILIES,
    message: str = '',
) -> tuple[str, list[ActionScore]]:
    """
    Score all available actions and return (best_action, all_scores_sorted).

    The LLM will then verbalize the best_action.
    """
    scores = [
        score_action(
            action,
            regulator_state=regulator_state,
            event_type=event_type,
            overload_level=overload_level,
            feedback_class=feedback_class,
            message=message,
        )
        for action in available_actions
    ]

    # Filter blocked actions
    valid = [s for s in scores if not s.blocked]
    if not valid:
        # All blocked — fall back to de_escalate or reflect_back
        valid = [s for s in scores if s.action in ('de_escalate', 'reflect_back')]
        if not valid:
            valid = scores  # Last resort

    # Sort by score descending
    valid.sort(key=lambda s: s.score, reverse=True)
    all_sorted = sorted(scores, key=lambda s: s.score, reverse=True)

    best = valid[0].action if valid else 'reflect_back'
    return best, all_sorted


# ---------------------------------------------------------------------------
# Full regulator pipeline
# ---------------------------------------------------------------------------

@dataclass
class RegulatorOutput:
    """Output of the full event → regulator → action pipeline."""
    event_type: EventType = 'neutral'
    event_multi: list[str] = field(default_factory=list)
    regulator_state: RegulatorState = field(default_factory=RegulatorState.baseline)
    selected_action: str = 'reflect_back'
    action_scores: list[ActionScore] = field(default_factory=list)
    intensity_modifier: float = 1.0
    # Source message — stored for clarification question selection
    _source_message: str = field(default='', repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            'event_type': self.event_type,
            'event_multi': list(self.event_multi),
            'regulator_state': self.regulator_state.to_dict(),
            'selected_action': self.selected_action,
            'action_scores': [s.to_dict() for s in self.action_scores[:5]],
            'intensity_modifier': round(float(self.intensity_modifier), 3),
        }

    def render_action_directive(self) -> str:
        """
        Render the selected action as an LLM prompt directive.

        This is injected into the prompt to guide the LLM's response style
        without letting the LLM choose the action itself.
        """
        directives = {
            'plan_small_step': 'Propose one small, specific, concrete next step. Keep it brief.',
            'acknowledge_and_support': 'First acknowledge the feeling without judgment. Be warm and brief.',
            'ask_clarifying_question': self._clarification_directive(),
            'celebrate_progress': 'Briefly acknowledge this success. Be genuine, not over-effusive.',
            'reduce_step_size': 'The previous step was too large. Suggest an even smaller version.',
            'suggest_environment_change': 'Suggest changing the environment to make the action easier.',
            'de_escalate': 'De-escalate. Do not plan. Listen and offer grounding support.',
            'reflect_back': 'Briefly reflect back what you heard without adding interpretation.',
            'reality_check': 'Gently offer one concrete perspective that may differ from the user\'s.',
            'explore_resistance': 'Explore how the user adapted the plan — without judgment.',
            'provide_information': 'Answer the question factually and briefly.',
        }
        return directives.get(self.selected_action, 'Respond briefly and concretely.')

    def _clarification_directive(self) -> str:
        """
        Generate the clarification directive using the question bank.
        Falls back to a generic instruction if the engine is unavailable.
        """
        try:
            from .clarification_engine import get_clarification_for_action
            result = get_clarification_for_action(
                self._source_message,
                event_type=self.event_type,
            )
            if result.selected_question:
                return f'Ask exactly this clarifying question (verbatim or very close): "{result.selected_question}"'
        except Exception:
            pass
        return 'Ask one brief clarifying question to understand better.'


def run_regulator_pipeline(
    *,
    message: str,
    previous_state: RegulatorState | None = None,
    overload_level: float = 0.0,
    feedback_class: str = 'neutral',
    personality_pressure_response: list[str] | None = None,
    personality_vulnerabilities: list[str] | None = None,
    personality_defense_mechanisms: list[str] | None = None,
) -> RegulatorOutput:
    """
    Run the full event → regulator → action pipeline.

    Returns a RegulatorOutput with:
    - classified event type(s)
    - updated regulator state
    - selected action
    - action scores
    """
    # Layer 1: Classify event
    event_type = classify_event(message)
    event_multi = classify_event_multi(message)

    # Start from previous state or baseline
    state = previous_state.clone() if previous_state is not None else RegulatorState.baseline()

    # Layer 2: Compute intensity modifier from personality
    intensity = compute_intensity_modifier(
        pressure_response=list(personality_pressure_response or []),
        vulnerabilities=list(personality_vulnerabilities or []),
        defense_mechanisms=list(personality_defense_mechanisms or []),
        event_type=event_type,
    )

    # Apply release profile
    new_state = apply_release_profile(state, event_type, intensity_modifier=intensity)

    # Layer 3: Select action
    best_action, action_scores = select_action(
        regulator_state=new_state,
        event_type=event_type,
        overload_level=overload_level,
        feedback_class=feedback_class,
        message=message,
    )

    return RegulatorOutput(
        event_type=event_type,
        event_multi=event_multi,
        regulator_state=new_state,
        selected_action=best_action,
        action_scores=action_scores,
        intensity_modifier=intensity,
        _source_message=str(message or ''),
    )
