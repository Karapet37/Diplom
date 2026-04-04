from __future__ import annotations

import os
import re
from typing import Any

from .duplicate_resolver import normalize_name
from .llm import call_json_model_for_role
from .models import InteractionFrame
from .prompt_builder import build_interaction_router_prompt
from .runtime_config import get_runtime_config
from .text_resources import capitalized_entity_phrases

_PERSONA_SWITCH_PATTERNS = (
    r'^\s*(?:ok|okay|fine|alright|well)?[\s,:\-]*(?:you are|you\'re|be|act as|speak as|pretend to be)\s+([A-Za-zА-Яа-я][\w\'-]*(?:\s+[A-Za-zА-Яа-я][\w\'-]*){0,3})(?:[.!?]\s*(.*))?$',
    r'^\s*(?:ладно|хорошо|ну ладно)?[\s,:\-]*(?:ты|ты теперь|будь|говори как|веди себя как)\s+([A-Za-zА-Яа-я][\w\'-]*(?:\s+[A-Za-zА-Яа-я][\w\'-]*){0,3})(?:[.!?]\s*(.*))?$',
)
_KNOWN_MESSAGE_KINDS = {'statement', 'question', 'command', 'mixed'}
_KNOWN_TOPIC_MODES = {'unknown', 'persona_self', 'external_topic', 'session_followup', 'persona_switch'}
_KNOWN_FOLLOWUP_MODES = {'none', 'followup_on_previous_topic', 'followup_on_persona'}
_QUESTION_PREFIXES = ('who ', 'what ', 'why ', 'how ', 'when ', 'where ', 'does ', 'do ', 'did ', 'is ', 'are ', 'can ', 'could ', 'would ', 'should ', 'will ', 'has ', 'have ', 'had ')
_REQUEST_PREFIXES = ('explain ', 'describe ', 'tell ', 'show ', 'outline ', 'list ', 'compare ', 'lay out ', 'walk me through ')
_REQUEST_ADVERBS = {'briefly', 'clearly', 'concretely', 'simply', 'just'}


def _enabled_model_stages() -> set[str]:
    raw = str(os.getenv('COGNITIVE_STAGE_MODEL_STEPS', 'interaction_router,response_shaper') or '').strip().lower()
    if not raw:
        return set()
    return {item.strip() for item in raw.split(',') if item.strip()}


def _router_stage_enabled() -> bool:
    return 'interaction_router' in _enabled_model_stages()


def _clean_message(message: str) -> str:
    return ' '.join(str(message or '').strip().split())


def _normalize_flag(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    raw = str(value or '').strip().lower()
    if not raw:
        return default
    if raw in {'1', 'true', 'yes', 'on'}:
        return True
    if raw in {'0', 'false', 'no', 'off'}:
        return False
    return default


def _resolve_known_name(candidate: str, known_entities: list[dict[str, Any]]) -> str:
    candidate_token = normalize_name(candidate)
    if not candidate_token:
        return ''
    for entity in list(known_entities or []):
        names = [str(entity.get('name') or '')] + [str(item) for item in list(entity.get('aliases') or [])]
        for name in names:
            if normalize_name(name) == candidate_token:
                return str(entity.get('name') or name).strip()
    return ''


def _clean_candidate_phrase(candidate: str, known_entities: list[dict[str, Any]], *, allow_unknown: bool = True) -> str:
    clean = _clean_message(candidate)
    if not clean:
        return ''
    for separator in ('.', '!', '?', ';'):
        if separator in clean:
            clean = clean.split(separator, 1)[0].strip()
    for separator in (', and ', ' and ', ', but ', ' but ', ', а ', ' а ', ', но ', ' но '):
        lowered = clean.lower()
        index = lowered.find(separator)
        if index > 0:
            clean = clean[:index].strip()
            break
    resolved = _resolve_known_name(clean, known_entities)
    if resolved:
        return resolved
    if not allow_unknown:
        return ''
    words = [part for part in clean.split() if part]
    if not words:
        return ''
    clipped = ' '.join(words[:4]).strip()
    return clipped.title() if clipped == clipped.lower() else clipped


def _extract_explicit_persona_switch(message: str, known_entities: list[dict[str, Any]]) -> tuple[str, str]:
    clean = _clean_message(message)
    if not clean:
        return '', ''
    for pattern in _PERSONA_SWITCH_PATTERNS:
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = _clean_candidate_phrase(match.group(1) or '', known_entities, allow_unknown=True)
        if candidate:
            remainder = _clean_message(match.group(2) or '')
            return candidate, remainder
    return '', clean


def _match_known_entities(message: str, known_entities: list[dict[str, Any]], *, exclude: set[str] | None = None) -> list[str]:
    lowered = normalize_name(message)
    excluded = set(exclude or set())
    matches: list[str] = []
    for entity in list(known_entities or []):
        names = [str(entity.get('name') or '')] + [str(item) for item in list(entity.get('aliases') or [])]
        canonical = str(entity.get('name') or '').strip()
        canonical_token = normalize_name(canonical)
        if not canonical or canonical_token in excluded:
            continue
        for name in names:
            token = normalize_name(name)
            if token and token not in excluded and token in lowered:
                matches.append(canonical)
                break
    return list(dict.fromkeys(item for item in matches if item))


def _capitalized_candidates(message: str, *, exclude: set[str] | None = None) -> list[str]:
    excluded = set(exclude or set())
    values: list[str] = []
    clean = _clean_message(message)
    for candidate in capitalized_entity_phrases(message):
        token = normalize_name(candidate)
        if not candidate or not token or token in excluded:
            continue
        if ' ' not in candidate and clean.startswith(candidate):
            # Sentence-leading singleton capitals are usually discourse starters
            # like "Who", "Why", "Does", "Briefly", not stable topic entities.
            continue
        if candidate and token and token not in excluded:
            values.append(candidate)
    return list(dict.fromkeys(values))


def _seed_topic_entity(
    *,
    routed_message: str,
    known_entities: list[dict[str, Any]],
    selected_persona: str,
    session_persona: str,
    current_entity: str,
) -> tuple[str, str, str, list[str]]:
    exclude = {
        normalize_name(value)
        for value in (
            selected_persona,
            session_persona,
        )
        if normalize_name(value)
    }
    evidence: list[str] = []
    matches = _match_known_entities(routed_message, known_entities, exclude=exclude)
    if matches:
        topic = matches[0]
        mode = 'external_topic'
        followup = 'followup_on_previous_topic' if normalize_name(topic) == normalize_name(current_entity) else 'none'
        evidence.append('matched graph entity in routed message')
        return topic, mode, followup, evidence
    capitalized = _capitalized_candidates(routed_message, exclude=exclude)
    if capitalized:
        topic = capitalized[0]
        mode = 'external_topic'
        followup = 'followup_on_previous_topic' if normalize_name(topic) == normalize_name(current_entity) else 'none'
        evidence.append('found capitalized topic candidate')
        return topic, mode, followup, evidence
    if current_entity:
        token_count = len([part for part in routed_message.split() if part])
        if token_count <= 10:
            evidence.append('short message with no new entity; keep previous topic')
            return current_entity, 'session_followup', 'followup_on_previous_topic', evidence
    if selected_persona:
        evidence.append('no external topic candidate; treat message as about active persona')
        return selected_persona, 'persona_self', 'followup_on_persona', evidence
    return '', 'unknown', 'none', evidence


def _seed_interaction_frame(
    *,
    message: str,
    selected_persona_hint: str,
    session_persona: str,
    current_entity: str,
    known_entities: list[dict[str, Any]],
) -> InteractionFrame:
    clean = _clean_message(message)
    requested_persona, routed_message = _extract_explicit_persona_switch(clean, known_entities)
    explicit_persona_switch = bool(requested_persona)
    active_persona = requested_persona or selected_persona_hint or session_persona
    topic_entity, topic_mode, followup_mode, evidence = _seed_topic_entity(
        routed_message=routed_message or clean,
        known_entities=known_entities,
        selected_persona=active_persona,
        session_persona=session_persona,
        current_entity=current_entity,
    )
    lowered = normalize_name(clean)
    lowered_for_shape = lowered
    lowered_tokens = [token for token in lowered.split() if token]
    if lowered_tokens and lowered_tokens[0] in _REQUEST_ADVERBS and len(lowered_tokens) > 1:
        lowered_for_shape = ' '.join(lowered_tokens[1:])
    if '?' in clean or any(lowered_for_shape.startswith(prefix) for prefix in _QUESTION_PREFIXES):
        message_kind = 'question'
    elif any(lowered_for_shape.startswith(prefix) for prefix in _REQUEST_PREFIXES):
        message_kind = 'command'
    else:
        message_kind = 'statement'
    question_present = message_kind == 'question' or '?' in clean
    if explicit_persona_switch:
        evidence.append('explicit persona switch scaffold detected')
    if session_persona and not explicit_persona_switch:
        evidence.append('kept session speaker persona')
    return InteractionFrame(
        message_kind=message_kind,
        question_present=question_present,
        explicit_persona_switch=explicit_persona_switch,
        requested_persona=requested_persona,
        keep_session_persona=bool((selected_persona_hint or session_persona) and not explicit_persona_switch),
        topic_entity=topic_entity,
        topic_mode='persona_switch' if explicit_persona_switch else topic_mode,
        followup_mode=followup_mode,
        routed_message=routed_message or clean,
        evidence=list(dict.fromkeys(item for item in evidence if item)),
        source='deterministic',
    )


def _normalized_known_tokens(known_entities: list[dict[str, Any]]) -> set[str]:
    tokens: set[str] = set()
    for entity in list(known_entities or []):
        names = [str(entity.get('name') or '')] + [str(item) for item in list(entity.get('aliases') or [])]
        for name in names:
            token = normalize_name(name)
            if token:
                tokens.add(token)
    return tokens


def _supports_candidate(candidate: str, *, message: str, current_entity: str, known_entities: list[dict[str, Any]]) -> bool:
    token = normalize_name(candidate)
    if not token:
        return False
    if token == normalize_name(current_entity):
        return True
    if token in _normalized_known_tokens(known_entities):
        return True
    return token in normalize_name(message)


def _validate_message_kind(value: Any, *, seed: InteractionFrame) -> str:
    candidate = str(value or '').strip().lower()
    return candidate if candidate in _KNOWN_MESSAGE_KINDS else seed.message_kind


def _validate_topic_mode(value: Any, *, seed: InteractionFrame) -> str:
    candidate = str(value or '').strip().lower()
    return candidate if candidate in _KNOWN_TOPIC_MODES else seed.topic_mode


def _validate_followup_mode(value: Any, *, seed: InteractionFrame) -> str:
    candidate = str(value or '').strip().lower()
    return candidate if candidate in _KNOWN_FOLLOWUP_MODES else seed.followup_mode


def _validate_interaction_frame(
    payload: dict[str, Any],
    *,
    message: str,
    session_persona: str,
    current_entity: str,
    known_entities: list[dict[str, Any]],
    seed: InteractionFrame,
) -> InteractionFrame | None:
    requested_persona = _clean_candidate_phrase(str(payload.get('requested_persona') or ''), known_entities, allow_unknown=True)
    explicit_persona_switch = _normalize_flag(payload.get('explicit_persona_switch'), default=seed.explicit_persona_switch)
    if explicit_persona_switch and not requested_persona:
        requested_persona = seed.requested_persona
    if explicit_persona_switch and not requested_persona:
        return None
    routed_message = _clean_message(str(payload.get('routed_message') or seed.routed_message or message))
    keep_session_persona = _normalize_flag(payload.get('keep_session_persona'), default=seed.keep_session_persona)
    if explicit_persona_switch:
        keep_session_persona = False
    raw_topic_entity = _clean_candidate_phrase(str(payload.get('topic_entity') or ''), known_entities, allow_unknown=True)
    if raw_topic_entity and not _supports_candidate(
        raw_topic_entity,
        message=routed_message or message,
        current_entity=current_entity,
        known_entities=known_entities,
    ):
        raw_topic_entity = ''
    topic_entity = raw_topic_entity or seed.topic_entity
    followup_mode = _validate_followup_mode(payload.get('followup_mode'), seed=seed)
    if not topic_entity and followup_mode == 'followup_on_previous_topic' and current_entity:
        topic_entity = current_entity
    topic_mode = _validate_topic_mode(payload.get('topic_mode'), seed=seed)
    message_kind = _validate_message_kind(payload.get('message_kind'), seed=seed)
    question_present = _normalize_flag(payload.get('question_present'), default=seed.question_present or message_kind == 'question')
    evidence = [str(item).strip() for item in list(payload.get('evidence') or []) if str(item).strip()][:6]
    if not routed_message:
        return None
    return InteractionFrame(
        message_kind=message_kind,
        question_present=question_present,
        explicit_persona_switch=explicit_persona_switch,
        requested_persona=requested_persona,
        keep_session_persona=keep_session_persona,
        topic_entity=topic_entity,
        topic_mode=topic_mode,
        followup_mode=followup_mode,
        routed_message=routed_message,
        evidence=evidence or list(seed.evidence),
        source='llm_guided',
    )


def route_interaction(
    *,
    message: str,
    selected_persona_hint: str = '',
    session_persona: str = '',
    current_entity: str = '',
    known_entities: list[dict[str, Any]] | None = None,
) -> InteractionFrame:
    known = list(known_entities or [])
    seed = _seed_interaction_frame(
        message=message,
        selected_persona_hint=selected_persona_hint,
        session_persona=session_persona,
        current_entity=current_entity,
        known_entities=known,
    )
    if not _router_stage_enabled():
        return seed
    payload = call_json_model_for_role(
        build_interaction_router_prompt(
            message=message,
            session_persona=selected_persona_hint or session_persona,
            current_entity=current_entity,
            known_entities=known,
            routed_seed=seed.to_dict(),
        ),
        role=get_runtime_config().roles.extraction,
    )
    if not isinstance(payload, dict):
        return seed
    frame = _validate_interaction_frame(
        payload,
        message=message,
        session_persona=session_persona,
        current_entity=current_entity,
        known_entities=known,
        seed=seed,
    )
    return frame or seed
