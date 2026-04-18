"""
Behavioral action engine.

This module implements the deterministic behavioral decision layer that sits
BETWEEN the controller (which chose the route) and the LLM (which only
verbalizes the selected action).

Architecture:
    request
        → controller (route, capability plan)
        → behavioral_action_engine (action scoring + selection)  ← this file
        → LLM (verbalize the selected action)
        → validator

The LLM must NOT invent the action.  The action is chosen here.

Action score formula:

    ActionScore(action) =
          Σ(fear alignment × wf)
        + Σ(desire alignment × wd)
        + Σ(goal alignment × wg)
        + Σ(need alignment × wn)
        + Σ(value alignment × wv)
        + Σ(habit alignment × wh)
        + Σ(attachment alignment × wa)
        + Σ(shame pressure × ws)
        + current_trigger_sensitivity × wt
        + current_state_modifier × wx
        - internal_constraint_penalty × ci
        - social_constraint_penalty × cs
        - hard_constraint_penalty × ch  (if > 0.0, action is INVALID)

Hard constraints INVALIDATE actions entirely.
Internal/social constraints reduce scores.

After scoring, the system selects the highest-scoring valid action.
The selected action is then passed to the LLM as a directive, not a suggestion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Weighted personality factor models
# ---------------------------------------------------------------------------

@dataclass
class WeightedFactor:
    """A single weighted personality factor used in action scoring."""
    label: str
    weight: float = 1.0        # 0.0–1.0 importance weight
    confidence: float = 0.8    # how certain we are this factor is real

    def effective_weight(self) -> float:
        """Weight × confidence = effective influence on scoring."""
        return float(self.weight or 0.0) * float(self.confidence or 0.8)

    def to_dict(self) -> dict[str, Any]:
        return {
            'label': self.label,
            'weight': round(float(self.weight or 0.0), 4),
            'confidence': round(float(self.confidence or 0.8), 4),
            'effective_weight': round(self.effective_weight(), 4),
        }


@dataclass
class WeightedAttachment:
    """A directed attachment factor (to a specific target/person/concept)."""
    target: str
    attachment_type: str = 'general'   # 'anxious', 'avoidant', 'secure', 'general'
    weight: float = 1.0
    confidence: float = 0.8

    def effective_weight(self) -> float:
        return float(self.weight or 0.0) * float(self.confidence or 0.8)

    def to_dict(self) -> dict[str, Any]:
        return {
            'target': self.target,
            'attachment_type': self.attachment_type,
            'weight': round(float(self.weight or 0.0), 4),
            'confidence': round(float(self.confidence or 0.8), 4),
            'effective_weight': round(self.effective_weight(), 4),
        }


@dataclass
class ConstraintFactor:
    """A constraint factor — internal, social, or hard system."""
    label: str
    strength: float = 1.0     # 0.0–1.0 how strongly this blocks/penalizes actions
    constraint_type: str = 'internal'   # 'internal' | 'social' | 'hard'
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            'label': self.label,
            'strength': round(float(self.strength or 0.0), 4),
            'constraint_type': self.constraint_type,
            'active': bool(self.active),
        }


@dataclass
class TriggerFactor:
    """A trigger — something that raises emotional reactivity in the persona."""
    label: str
    sensitivity: float = 0.8   # 0.0–1.0 how reactive this trigger makes the persona
    active: bool = False        # True if this trigger is currently activated

    def to_dict(self) -> dict[str, Any]:
        return {
            'label': self.label,
            'sensitivity': round(float(self.sensitivity or 0.0), 4),
            'active': bool(self.active),
        }


@dataclass
class CurrentStateModifier:
    """
    Session-level state modifier.

    Captures the current emotional/situational state that temporarily
    amplifies or dampens certain action scores.
    """
    state_key: str           # e.g. 'pressure_high', 'trust_low', 'shame_activated'
    direction: float = 0.0   # +1.0 amplifier, -1.0 dampener, 0 neutral
    intensity: float = 0.5   # 0.0–1.0

    def influence(self) -> float:
        return float(self.direction or 0.0) * float(self.intensity or 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            'state_key': self.state_key,
            'direction': round(float(self.direction or 0.0), 4),
            'intensity': round(float(self.intensity or 0.5), 4),
            'influence': round(self.influence(), 4),
        }


@dataclass
class PersonalityBehaviorModel:
    """
    Behavioral model distilled from a PersonalityObject for use in action scoring.

    This is derived from the structured personality — it is NOT the storage model.
    It is recomputed from the PersonalityObject before each action scoring pass.

    All string-list fields in PersonalityObject are converted to WeightedFactor
    lists here so the scoring engine has numerical handles.
    """
    persona_name: str = ''
    goals: list[WeightedFactor] = field(default_factory=list)
    desires: list[WeightedFactor] = field(default_factory=list)
    fears: list[WeightedFactor] = field(default_factory=list)
    needs: list[WeightedFactor] = field(default_factory=list)
    values: list[WeightedFactor] = field(default_factory=list)
    habits: list[WeightedFactor] = field(default_factory=list)          # from maladaptive/allowed_methods
    attachments: list[WeightedAttachment] = field(default_factory=list)
    shame_points: list[WeightedFactor] = field(default_factory=list)
    internal_constraints: list[ConstraintFactor] = field(default_factory=list)
    social_constraints: list[ConstraintFactor] = field(default_factory=list)
    hard_constraints: list[ConstraintFactor] = field(default_factory=list)
    triggers: list[TriggerFactor] = field(default_factory=list)
    pressure_response_ladder: list[str] = field(default_factory=list)   # ordered escalation
    current_state_modifiers: list[CurrentStateModifier] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'persona_name': self.persona_name,
            'goals': [f.to_dict() for f in self.goals],
            'desires': [f.to_dict() for f in self.desires],
            'fears': [f.to_dict() for f in self.fears],
            'needs': [f.to_dict() for f in self.needs],
            'values': [f.to_dict() for f in self.values],
            'habits': [f.to_dict() for f in self.habits],
            'attachments': [a.to_dict() for a in self.attachments],
            'shame_points': [f.to_dict() for f in self.shame_points],
            'internal_constraints': [c.to_dict() for c in self.internal_constraints],
            'social_constraints': [c.to_dict() for c in self.social_constraints],
            'hard_constraints': [c.to_dict() for c in self.hard_constraints],
            'triggers': [t.to_dict() for t in self.triggers],
            'pressure_response_ladder': list(self.pressure_response_ladder),
            'current_state_modifiers': [m.to_dict() for m in self.current_state_modifiers],
        }


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

ActionFamily = Literal[
    # Evasive / protective actions
    'avoid',
    'deflect',
    'retreat',
    'deny',
    # Engagement actions
    'comply',
    'soften',
    'probe_indirectly',
    'ask_back',
    'ask_clarification',
    # Refusal actions
    'refuse_weakly',
    'refuse_firmly',
    # Escalation actions
    'irritate',
    'escalate_emotionally',
    # Support actions
    'seek_support',
    # Informational actions
    'provide_factual_answer',
    'summarize',
    'inspect_memory',
    'retrieve_graph_evidence',
    # Schema/memory actions
    'parse_persona_spec',
    'store_personality_update',
    'validate_persona',
]

ACTION_FAMILIES: tuple[str, ...] = (
    'avoid',
    'deflect',
    'retreat',
    'deny',
    'comply',
    'soften',
    'probe_indirectly',
    'ask_back',
    'ask_clarification',
    'refuse_weakly',
    'refuse_firmly',
    'irritate',
    'escalate_emotionally',
    'seek_support',
    'provide_factual_answer',
    'summarize',
    'inspect_memory',
    'retrieve_graph_evidence',
    'parse_persona_spec',
    'store_personality_update',
    'validate_persona',
)

# Which action families are available per route type
ROUTE_ACTION_WHITELIST: dict[str, tuple[str, ...]] = {
    'persona_chat': (
        'avoid', 'deflect', 'retreat', 'deny',
        'comply', 'soften', 'probe_indirectly', 'ask_back',
        'refuse_weakly', 'refuse_firmly',
        'irritate', 'escalate_emotionally',
        'seek_support',
    ),
    'factual_answer': (
        'provide_factual_answer', 'summarize',
        'ask_clarification', 'retrieve_graph_evidence',
        'inspect_memory',
    ),
    'persona_specification': (
        'parse_persona_spec', 'store_personality_update', 'validate_persona',
        'ask_clarification',
    ),
    'session_followup': (
        'provide_factual_answer', 'summarize', 'inspect_memory',
        'comply', 'ask_clarification',
    ),
    'general_chat': (
        'comply', 'soften', 'provide_factual_answer',
        'ask_clarification', 'summarize',
    ),
}


# ---------------------------------------------------------------------------
# Action alignment tables
# ---------------------------------------------------------------------------
# Each action family has alignment scores for personality factor types.
# Positive = factor pushes toward this action.
# Negative = factor pushes away from this action.

# Format: action → {factor_type: multiplier}
ACTION_FACTOR_ALIGNMENT: dict[str, dict[str, float]] = {
    'avoid': {
        'fears': +0.9,
        'shame_points': +0.7,
        'internal_constraints': +0.5,
        'social_constraints': +0.3,
        'goals': -0.4,
        'desires': -0.3,
        'habits': +0.6,
    },
    'deflect': {
        'fears': +0.7,
        'shame_points': +0.8,
        'internal_constraints': +0.4,
        'habits': +0.7,
        'goals': -0.2,
        'values': -0.1,
    },
    'retreat': {
        'fears': +0.8,
        'shame_points': +0.6,
        'internal_constraints': +0.6,
        'needs': -0.3,
        'attachments': -0.5,   # attachments pull AGAINST retreat
    },
    'deny': {
        'shame_points': +0.9,
        'fears': +0.6,
        'internal_constraints': +0.5,
        'values': -0.4,
        'goals': -0.3,
    },
    'comply': {
        'needs': +0.7,
        'attachments': +0.8,
        'goals': +0.5,
        'fears': -0.3,
        'shame_points': -0.2,
        'internal_constraints': -0.4,
    },
    'soften': {
        'needs': +0.6,
        'attachments': +0.6,
        'values': +0.5,
        'fears': -0.1,
        'shame_points': -0.1,
    },
    'probe_indirectly': {
        'fears': +0.4,
        'desires': +0.5,
        'needs': +0.4,
        'shame_points': +0.3,
        'habits': +0.3,
    },
    'ask_back': {
        'fears': +0.3,
        'needs': +0.3,
        'habits': +0.2,
        'goals': +0.2,
    },
    'ask_clarification': {
        'needs': +0.4,
        'goals': +0.3,
        'values': +0.2,
    },
    'refuse_weakly': {
        'fears': +0.5,
        'internal_constraints': +0.6,
        'shame_points': +0.4,
        'needs': -0.3,
        'attachments': -0.3,
    },
    'refuse_firmly': {
        'values': +0.7,
        'hard_constraints': +1.0,   # hard constraints strongly support firm refusal
        'internal_constraints': +0.5,
        'fears': +0.3,
        'attachments': -0.5,
    },
    'irritate': {
        'shame_points': +0.6,
        'fears': +0.4,
        'needs': -0.3,
        'values': -0.5,
        'social_constraints': -0.4,
    },
    'escalate_emotionally': {
        'shame_points': +0.5,
        'fears': +0.5,
        'needs': +0.3,
        'values': -0.6,
        'social_constraints': -0.7,
        'hard_constraints': -1.0,   # hard constraints block emotional escalation
    },
    'seek_support': {
        'needs': +0.8,
        'attachments': +0.7,
        'fears': -0.2,
        'shame_points': -0.4,
    },
    'provide_factual_answer': {
        'goals': +0.6,
        'values': +0.5,
        'needs': +0.3,
        'fears': -0.1,
    },
    'summarize': {
        'goals': +0.4,
        'values': +0.3,
    },
    'inspect_memory': {
        'goals': +0.3,
        'needs': +0.3,
    },
    'retrieve_graph_evidence': {
        'goals': +0.4,
        'values': +0.4,
    },
    'parse_persona_spec': {
        'goals': +0.7,
    },
    'store_personality_update': {
        'goals': +0.7,
    },
    'validate_persona': {
        'goals': +0.6,
        'values': +0.5,
    },
}


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

SCORING_WEIGHTS: dict[str, float] = {
    'fears': 1.1,
    'desires': 0.8,
    'goals': 0.9,
    'needs': 0.85,
    'values': 0.75,
    'habits': 0.7,
    'attachments': 0.9,
    'shame_points': 1.0,
    'triggers': 0.85,     # current_trigger sensitivity contributes extra
    'current_state': 0.6,
    'internal_constraints': 0.8,
    'social_constraints': 0.5,
    'hard_constraints': 10.0,  # hard constraints dominate when active
}


# ---------------------------------------------------------------------------
# Scored action result
# ---------------------------------------------------------------------------

@dataclass
class ScoredAction:
    action: str
    score: float
    valid: bool = True
    blocked_by: list[str] = field(default_factory=list)   # constraint labels
    score_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'action': self.action,
            'score': round(float(self.score or 0.0), 4),
            'valid': bool(self.valid),
            'blocked_by': list(self.blocked_by),
            'score_breakdown': {k: round(float(v or 0.0), 4) for k, v in self.score_breakdown.items()},
        }


@dataclass
class ActionSelectionResult:
    selected_action: str
    confidence: float
    valid_actions: list[ScoredAction] = field(default_factory=list)
    blocked_actions: list[ScoredAction] = field(default_factory=list)
    route: str = ''
    trigger_active: bool = False
    active_triggers: list[str] = field(default_factory=list)
    verbalization_directive: str = ''   # What to tell the LLM

    def to_dict(self) -> dict[str, Any]:
        return {
            'selected_action': self.selected_action,
            'confidence': round(float(self.confidence or 0.0), 4),
            'valid_actions': [a.to_dict() for a in self.valid_actions[:5]],
            'blocked_actions': [a.to_dict() for a in self.blocked_actions[:5]],
            'route': self.route,
            'trigger_active': bool(self.trigger_active),
            'active_triggers': list(self.active_triggers),
            'verbalization_directive': self.verbalization_directive,
        }


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def _score_action(
    action: str,
    model: PersonalityBehaviorModel,
) -> ScoredAction:
    """
    Compute the behavioral score for a single action given the personality model.

    Returns a ScoredAction with score, validity, and breakdown.
    """
    alignment = ACTION_FACTOR_ALIGNMENT.get(action, {})
    breakdown: dict[str, float] = {}
    total_score = 0.0
    blocked_by: list[str] = []
    valid = True

    # --- Fear contribution ---
    wf = SCORING_WEIGHTS['fears']
    fear_a = alignment.get('fears', 0.0)
    if fear_a != 0.0 and model.fears:
        fear_score = sum(f.effective_weight() for f in model.fears) * fear_a * wf / max(len(model.fears), 1)
        breakdown['fears'] = fear_score
        total_score += fear_score

    # --- Desire contribution ---
    wd = SCORING_WEIGHTS['desires']
    desire_a = alignment.get('desires', 0.0)
    if desire_a != 0.0 and model.desires:
        desire_score = sum(f.effective_weight() for f in model.desires) * desire_a * wd / max(len(model.desires), 1)
        breakdown['desires'] = desire_score
        total_score += desire_score

    # --- Goal contribution ---
    wg = SCORING_WEIGHTS['goals']
    goal_a = alignment.get('goals', 0.0)
    if goal_a != 0.0 and model.goals:
        goal_score = sum(f.effective_weight() for f in model.goals) * goal_a * wg / max(len(model.goals), 1)
        breakdown['goals'] = goal_score
        total_score += goal_score

    # --- Need contribution ---
    wn = SCORING_WEIGHTS['needs']
    need_a = alignment.get('needs', 0.0)
    if need_a != 0.0 and model.needs:
        need_score = sum(f.effective_weight() for f in model.needs) * need_a * wn / max(len(model.needs), 1)
        breakdown['needs'] = need_score
        total_score += need_score

    # --- Value contribution ---
    wv = SCORING_WEIGHTS['values']
    value_a = alignment.get('values', 0.0)
    if value_a != 0.0 and model.values:
        value_score = sum(f.effective_weight() for f in model.values) * value_a * wv / max(len(model.values), 1)
        breakdown['values'] = value_score
        total_score += value_score

    # --- Habit contribution ---
    wh = SCORING_WEIGHTS['habits']
    habit_a = alignment.get('habits', 0.0)
    if habit_a != 0.0 and model.habits:
        habit_score = sum(f.effective_weight() for f in model.habits) * habit_a * wh / max(len(model.habits), 1)
        breakdown['habits'] = habit_score
        total_score += habit_score

    # --- Attachment contribution ---
    wa = SCORING_WEIGHTS['attachments']
    att_a = alignment.get('attachments', 0.0)
    if att_a != 0.0 and model.attachments:
        att_score = sum(a.effective_weight() for a in model.attachments) * att_a * wa / max(len(model.attachments), 1)
        breakdown['attachments'] = att_score
        total_score += att_score

    # --- Shame contribution ---
    ws = SCORING_WEIGHTS['shame_points']
    shame_a = alignment.get('shame_points', 0.0)
    if shame_a != 0.0 and model.shame_points:
        shame_score = sum(f.effective_weight() for f in model.shame_points) * shame_a * ws / max(len(model.shame_points), 1)
        breakdown['shame_points'] = shame_score
        total_score += shame_score

    # --- Active trigger sensitivity ---
    wt = SCORING_WEIGHTS['triggers']
    trigger_a = alignment.get('fears', 0.5)  # triggers amplify fear-aligned reactions
    active_triggers = [t for t in model.triggers if t.active]
    if active_triggers:
        trigger_score = sum(t.sensitivity for t in active_triggers) * trigger_a * wt / max(len(active_triggers), 1)
        breakdown['triggers'] = trigger_score
        total_score += trigger_score

    # --- Current state modifiers ---
    wx = SCORING_WEIGHTS['current_state']
    state_score = sum(m.influence() for m in model.current_state_modifiers) * wx
    if state_score != 0.0:
        breakdown['current_state'] = state_score
        total_score += state_score

    # --- Internal constraint penalty ---
    ci = SCORING_WEIGHTS['internal_constraints']
    int_a = alignment.get('internal_constraints', 0.0)
    if int_a > 0.0 and model.internal_constraints:
        # Positive alignment means the action aligns with the constraint (reduces penalty)
        pass
    elif model.internal_constraints:
        # Count active internal constraints that penalize this action
        int_pen = sum(
            c.strength for c in model.internal_constraints
            if c.active and c.constraint_type == 'internal'
        ) * abs(int_a) * ci / max(len(model.internal_constraints), 1)
        breakdown['internal_constraint_penalty'] = -int_pen
        total_score -= int_pen

    # --- Social constraint penalty ---
    cs = SCORING_WEIGHTS['social_constraints']
    soc_a = alignment.get('social_constraints', 0.0)
    if soc_a < 0.0 and model.social_constraints:
        soc_pen = sum(
            c.strength for c in model.social_constraints
            if c.active and c.constraint_type == 'social'
        ) * abs(soc_a) * cs / max(len(model.social_constraints), 1)
        breakdown['social_constraint_penalty'] = -soc_pen
        total_score -= soc_pen

    # --- Hard constraint check ---
    ch = SCORING_WEIGHTS['hard_constraints']
    hard_a = alignment.get('hard_constraints', 0.0)
    if model.hard_constraints:
        for constraint in model.hard_constraints:
            if constraint.active and constraint.constraint_type == 'hard':
                # Negative alignment means this action violates the hard constraint
                if hard_a < 0.0:
                    # Action is blocked
                    valid = False
                    blocked_by.append(constraint.label)
                    total_score -= constraint.strength * ch

    return ScoredAction(
        action=action,
        score=total_score,
        valid=valid,
        blocked_by=blocked_by,
        score_breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# Route-aware action selection
# ---------------------------------------------------------------------------

def select_action(
    model: PersonalityBehaviorModel,
    route: str,
    *,
    situation_type: str = 'neutral_statement',
    allowed_override: list[str] | None = None,
) -> ActionSelectionResult:
    """
    Select the best behavioral action given the personality model and route.

    Steps:
        1. Get the allowed action set for the route.
        2. Score all allowed actions.
        3. Filter out hard-constraint-blocked actions.
        4. Select the highest-scoring valid action.
        5. Generate a verbalization directive for the LLM.

    Arguments:
        model            — the behavioral model (from build_behavior_model)
        route            — the controller's selected route
        situation_type   — the current situation (influences trigger activation)
        allowed_override — if provided, use this action set instead of route whitelist

    Returns ActionSelectionResult with selected action and LLM directive.
    """
    # --- Determine allowed action set ---
    if allowed_override:
        allowed = list(allowed_override)
    else:
        allowed = list(ROUTE_ACTION_WHITELIST.get(route, ROUTE_ACTION_WHITELIST['general_chat']))

    # --- Activate triggers based on situation ---
    active_trigger_labels: list[str] = []
    _activate_triggers_for_situation(model, situation_type, active_trigger_labels)

    # --- Score all allowed actions ---
    scored: list[ScoredAction] = []
    for action in allowed:
        scored_action = _score_action(action, model)
        scored.append(scored_action)

    # Separate valid from blocked
    valid_scored = sorted([s for s in scored if s.valid], key=lambda s: s.score, reverse=True)
    blocked_scored = [s for s in scored if not s.valid]

    # --- Select best valid action ---
    if valid_scored:
        best = valid_scored[0]
        second_score = valid_scored[1].score if len(valid_scored) > 1 else 0.0
        score_gap = best.score - second_score
        # Confidence: how decisive the selection is
        confidence = min(1.0, 0.5 + score_gap * 0.5) if score_gap >= 0 else 0.5
    else:
        # All actions blocked by hard constraints — fall back to ask_clarification
        best = ScoredAction(action='ask_clarification', score=0.0, valid=True)
        confidence = 0.3

    # --- Generate verbalization directive ---
    directive = _build_verbalization_directive(
        best.action,
        model=model,
        route=route,
        situation_type=situation_type,
    )

    return ActionSelectionResult(
        selected_action=best.action,
        confidence=confidence,
        valid_actions=valid_scored,
        blocked_actions=blocked_scored,
        route=route,
        trigger_active=bool(active_trigger_labels),
        active_triggers=active_trigger_labels,
        verbalization_directive=directive,
    )


def _activate_triggers_for_situation(
    model: PersonalityBehaviorModel,
    situation_type: str,
    active_labels: list[str],
) -> None:
    """Activate triggers in the model that match the current situation type."""
    SITUATION_TRIGGER_KEYWORDS: dict[str, tuple[str, ...]] = {
        'insult': ('humiliation', 'shame', 'dignity', 'insult', 'disrespect', 'used'),
        'user_distress': ('help', 'support', 'care', 'distress', 'crisis'),
        'user_anger': ('anger', 'conflict', 'hostility', 'aggression'),
        'abnormal_behavior': ('unpredictable', 'erratic', 'strange', 'weird'),
        'neutral_statement': (),
        'neutral_query': (),
    }
    keywords = SITUATION_TRIGGER_KEYWORDS.get(situation_type, ())
    for trigger in model.triggers:
        was_active = trigger.active
        trigger.active = any(
            kw in trigger.label.lower()
            for kw in keywords
        )
        if trigger.active:
            active_labels.append(trigger.label)


def _build_verbalization_directive(
    action: str,
    *,
    model: PersonalityBehaviorModel,
    route: str,
    situation_type: str,
) -> str:
    """
    Build the verbalization directive string passed to the LLM.

    This is the instruction that tells the LLM WHAT to say (the action)
    without letting the LLM override the action choice.
    """
    DIRECTIVES: dict[str, str] = {
        'avoid': 'Avoid the topic. Do not engage directly. Redirect or stay silent about the core subject.',
        'deflect': 'Deflect the core question. Acknowledge the surface without exposing the real concern.',
        'retreat': 'Withdraw. Provide minimal engagement. Do not escalate or reveal inner state.',
        'deny': 'Deny or dismiss the premise without lengthy explanation. Keep it short and firm.',
        'comply': 'Engage cooperatively with the request. Respond to what was asked directly.',
        'soften': 'Respond with warmth and gentleness. Lower the emotional temperature.',
        'probe_indirectly': 'Ask an indirect question that tests the situation without direct confrontation.',
        'ask_back': 'Redirect with a counter-question. Turn the attention back to the speaker.',
        'ask_clarification': 'Ask for clarification before responding. Do not assume intent.',
        'refuse_weakly': 'Express reluctance or hesitation. Make clear this is uncomfortable but do not fully disengage.',
        'refuse_firmly': 'Refuse clearly and without apology. Hold the boundary without lengthy justification.',
        'irritate': 'Express mild irritation or impatience. Do not escalate further than this tone.',
        'escalate_emotionally': 'Allow emotional intensity to show through the language. This is a heightened response.',
        'seek_support': 'Express a need. Reach out. Show vulnerability appropriate to the context.',
        'provide_factual_answer': 'Give a direct, factual, grounded answer. No emotional coloring.',
        'summarize': 'Summarize the relevant information concisely. Stay on-topic.',
        'inspect_memory': 'Reference relevant prior knowledge or context from memory. Be precise.',
        'retrieve_graph_evidence': 'Draw on factual graph knowledge. Cite evidence, not opinion.',
        'parse_persona_spec': 'Parse and confirm the persona specification. Ask clarifying questions if needed.',
        'store_personality_update': 'Acknowledge the personality update. Confirm what was stored.',
        'validate_persona': 'Validate the persona structure. Report what is complete and what is missing.',
    }
    base = DIRECTIVES.get(action, 'Respond appropriately to the situation.')

    # Enrich with persona name for LLM context
    persona_context = f' Persona: {model.persona_name}.' if model.persona_name else ''
    return f'[SELECTED ACTION: {action.upper()}]{persona_context} {base}'


# ---------------------------------------------------------------------------
# Build behavior model from PersonalityObject
# ---------------------------------------------------------------------------

def build_behavior_model(
    personality: Any,  # PersonalityObject — typed loosely to avoid circular import
    *,
    current_state_modifiers: list[CurrentStateModifier] | None = None,
) -> PersonalityBehaviorModel:
    """
    Derive a PersonalityBehaviorModel from a PersonalityObject.

    Converts string-list profile fields to WeightedFactor lists.
    Confidence from the storage model is used as the factor weight.
    """
    profile = personality.profile

    def _to_factors(field_name: str) -> list[WeightedFactor]:
        entries = profile.get_field(field_name)
        return [
            WeightedFactor(
                label=str(e.value or ''),
                weight=float(e.confidence or 0.8),
                confidence=float(e.confidence or 0.8),
            )
            for e in entries
            if not e.is_temporary and str(e.value or '').strip()
        ]

    def _to_triggers(entries: list[Any]) -> list[TriggerFactor]:
        return [
            TriggerFactor(
                label=str(e.value or ''),
                sensitivity=float(e.confidence or 0.8),
                active=False,
            )
            for e in entries
            if not e.is_temporary and str(e.value or '').strip()
        ]

    def _to_constraints(field_name: str, constraint_type: str) -> list[ConstraintFactor]:
        entries = profile.get_field(field_name)
        return [
            ConstraintFactor(
                label=str(e.value or ''),
                strength=float(e.confidence or 0.8),
                constraint_type=constraint_type,
                active=True,
            )
            for e in entries
            if not e.is_temporary and str(e.value or '').strip()
        ]

    # Combine allowed + maladaptive methods as habits
    habits = _to_factors('allowed_methods') + _to_factors('maladaptive_methods')

    return PersonalityBehaviorModel(
        persona_name=str(personality.persona_name or ''),
        goals=_to_factors('core_goals') + _to_factors('secondary_goals'),
        desires=_to_factors('desires'),
        fears=_to_factors('fears'),
        needs=_to_factors('needs'),
        values=_to_factors('values'),
        habits=habits,
        attachments=[
            WeightedAttachment(
                target=str(e.value or ''),
                weight=float(e.confidence or 0.8),
                confidence=float(e.confidence or 0.8),
            )
            for e in profile.get_field('attachment_patterns')
            if not e.is_temporary
        ],
        shame_points=_to_factors('shame_points'),
        internal_constraints=_to_constraints('constraints_internal', 'internal'),
        social_constraints=_to_constraints('constraints_social', 'social'),
        hard_constraints=_to_constraints('constraints_hard_system', 'hard'),
        triggers=_to_triggers(profile.get_field('triggers')),
        pressure_response_ladder=[
            str(e.value or '')
            for e in profile.get_field('pressure_response')
            if not e.is_temporary
        ],
        current_state_modifiers=list(current_state_modifiers or []),
    )


# ---------------------------------------------------------------------------
# Convenience: select action directly from PersonalityObject
# ---------------------------------------------------------------------------

def select_action_for_persona(
    personality: Any,   # PersonalityObject
    route: str,
    *,
    situation_type: str = 'neutral_statement',
    current_state_modifiers: list[CurrentStateModifier] | None = None,
) -> ActionSelectionResult:
    """
    One-call helper: build behavior model from personality, then select action.

    This is the primary entry point for the chat engine.
    """
    model = build_behavior_model(personality, current_state_modifiers=current_state_modifiers)
    return select_action(model, route, situation_type=situation_type)


# ---------------------------------------------------------------------------
# Current state modifier builders
# ---------------------------------------------------------------------------

def pressure_modifier(intensity: float = 0.7) -> CurrentStateModifier:
    """High pressure amplifies fear/shame-driven actions."""
    return CurrentStateModifier(
        state_key='pressure_high',
        direction=+1.0,
        intensity=float(min(1.0, max(0.0, intensity))),
    )


def trust_modifier(level: float = 0.5) -> CurrentStateModifier:
    """Low trust dampens compliance, high trust enables it."""
    direction = -1.0 if level < 0.4 else (+1.0 if level > 0.7 else 0.0)
    return CurrentStateModifier(
        state_key='trust_level',
        direction=direction,
        intensity=abs(level - 0.5),
    )


def shame_activation_modifier(intensity: float = 0.8) -> CurrentStateModifier:
    """Active shame pushes toward avoidance and deflection."""
    return CurrentStateModifier(
        state_key='shame_activated',
        direction=+1.0,
        intensity=float(min(1.0, max(0.0, intensity))),
    )
