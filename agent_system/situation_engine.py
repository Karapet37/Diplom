from __future__ import annotations

from typing import Any

from .classifier_forest import DEFAULT_SITUATION_CLASSIFIER
from .duplicate_resolver import normalize_name
from .models import Situation, SituationDecision, SituationFeatures, UserState
from .text_resources import SECOND_PERSON_TOKENS, SELF_REFERENCE_TOKENS

def _clean_message(message: str) -> str:
    return ' '.join(str(message or '').strip().split())


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _normalized_tokens(text: str) -> set[str]:
    return {token for token in normalize_name(text).split() if token}


def _detect_target(message: str, *, primary_entity: str = '', selected_head: str = '', signals: dict[str, float] | None = None) -> str:
    signal_map = dict(signals or {})
    token_set = _normalized_tokens(message)
    persona_token = normalize_name(selected_head or primary_entity)
    mentions_persona = bool(persona_token and persona_token in normalize_name(message))
    mentions_second_person = bool(token_set & SECOND_PERSON_TOKENS)
    mentions_self = bool(token_set & SELF_REFERENCE_TOKENS)

    if signal_map.get('contains_insult') and (mentions_persona or mentions_second_person or signal_map.get('contains_persona_reference')):
        return 'persona'
    if signal_map.get('contains_distress') and mentions_self:
        return 'self'
    if signal_map.get('contains_question') and (mentions_persona or mentions_second_person):
        return 'persona'
    if mentions_self:
        return 'self'
    if mentions_persona or mentions_second_person:
        return 'persona'
    return 'external'


def _build_situation_features(message: str, *, target: str, user_state: UserState) -> SituationFeatures:
    signals = dict(user_state.signals)
    feature_map = {
        'insult_signal': float(signals.get('contains_insult') or 0.0),
        'anger_signal': float(signals.get('contains_anger') or 0.0),
        'distress_signal': float(signals.get('contains_distress') or 0.0),
        'question_signal': float(signals.get('contains_question') or 0.0),
        'help_signal': float(signals.get('contains_help_request') or 0.0),
        'moral_violation_signal': float(signals.get('contains_moral_violation') or 0.0),
        'celebratory_signal': float(signals.get('contains_celebratory') or 0.0),
        'persona_target_signal': float(target == 'persona'),
        'self_target_signal': float(target == 'self'),
        'external_target_signal': float(target == 'external'),
    }
    evidence = [f'{key}={value}' for key, value in feature_map.items() if value > 0]
    if normalize_name(message):
        evidence.append(f'tone={user_state.tone}')
        evidence.append(f'intent={user_state.intent}')
    return SituationFeatures(feature_map=feature_map, evidence=evidence)


def _rule_driven_type(*, classifier_decision: SituationDecision, user_state: UserState, target: str) -> str:
    signals = dict(user_state.signals)
    if signals.get('contains_moral_violation') and signals.get('contains_celebratory'):
        return 'abnormal_behavior'
    if signals.get('contains_insult') and target == 'persona':
        return 'insult'
    if signals.get('contains_distress'):
        return 'user_distress'
    if signals.get('contains_anger'):
        return 'user_anger'
    if signals.get('contains_question'):
        return 'neutral_query'
    classifier_type = str(classifier_decision.situation_type or '').strip().lower()
    if classifier_type == 'user_distress':
        # A calm world-building or persona-setting statement should not be escalated
        # into distress handling just because the statistical classifier is unsure.
        if not signals.get('contains_help_request') and user_state.intent == 'statement' and user_state.tone == 'neutral':
            return 'neutral_statement'
    return classifier_decision.situation_type or 'neutral_statement'


def _severity_for(situation_type: str, *, target: str, user_state: UserState) -> float:
    signals = dict(user_state.signals)
    if situation_type == 'insult':
        value = 0.55 + 0.2 * float(signals.get('contains_anger') or 0.0) + 0.1 * float(target == 'persona')
    elif situation_type == 'user_distress':
        value = 0.45 + 0.25 * float(signals.get('contains_distress') or 0.0) + 0.1 * float(signals.get('contains_help_request') or 0.0)
    elif situation_type == 'abnormal_behavior':
        value = 0.7 + 0.15 * float(signals.get('contains_moral_violation') or 0.0) + 0.1 * float(signals.get('contains_celebratory') or 0.0)
    elif situation_type == 'user_anger':
        value = 0.45 + 0.2 * float(signals.get('contains_anger') or 0.0) + 0.05 * float(signals.get('contains_insult') or 0.0)
    elif situation_type == 'neutral_query':
        value = 0.35 + 0.15 * float(signals.get('contains_question') or 0.0)
    else:
        value = 0.2 + 0.1 * float(signals.get('contains_question') or 0.0)
    return round(_clamp(value), 4)


def situation_summary(situation: Situation | dict[str, Any] | str) -> str:
    if isinstance(situation, str):
        return str(situation or '').strip()
    if isinstance(situation, Situation):
        situation_type = str(situation.type or 'neutral_statement').strip() or 'neutral_statement'
        target = str(situation.target or 'external').strip() or 'external'
        severity = float(situation.severity or 0.0)
        return f'type={situation_type}; target={target}; severity={severity:.2f}'
    if not isinstance(situation, dict):
        return ''
    situation_type = str(situation.get('type') or 'neutral_statement').strip() or 'neutral_statement'
    target = str(situation.get('target') or 'external').strip() or 'external'
    severity = float(situation.get('severity') or 0.0)
    return f'type={situation_type}; target={target}; severity={severity:.2f}'


def model_situation(
    *,
    message: str,
    primary_entity: str = '',
    selected_head: str = '',
    user_state: UserState | None = None,
) -> Situation:
    clean_message = _clean_message(message)
    state = user_state or UserState()
    target = _detect_target(
        clean_message,
        primary_entity=primary_entity,
        selected_head=selected_head,
        signals=dict(state.signals),
    )
    features = _build_situation_features(clean_message, target=target, user_state=state)
    decision = DEFAULT_SITUATION_CLASSIFIER.classify(features)
    situation_type = _rule_driven_type(classifier_decision=decision, user_state=state, target=target)
    situation = Situation(
        type=situation_type,
        target=target,
        severity=_severity_for(situation_type, target=target, user_state=state),
        classifier_hint={
            'type': decision.situation_type,
            'confidence': decision.confidence,
            'votes': dict(decision.votes),
        },
        evidence=list(decision.evidence),
    )
    return Situation(
        type=situation.type,
        target=situation.target,
        severity=situation.severity,
        classifier_hint=dict(situation.classifier_hint),
        evidence=list(situation.evidence),
        summary=situation_summary(situation),
    )
