from __future__ import annotations

import re
from typing import Any

from .duplicate_resolver import normalize_name
from .language_tools import detect_language_code
from .models import MessageAnalysis, MessageEntity, UserState
from .situation_engine import model_situation

ENTITY_PATTERNS = (
    re.compile(r'"([^"]{2,80})"'),
    re.compile(r"'([^']{2,80})'"),
    re.compile(r'\b(?:about|like|as|with|to|for)\s+([A-ZА-Я][\w-]*(?:\s+[A-ZА-Я][\w-]*){0,3})'),
    re.compile(r'\b(?:about|regarding|around|про|о|как)\s+([A-Za-zА-Яа-я][\w-]*(?:\s+[A-Za-zА-Яа-я][\w-]*){0,3})', re.IGNORECASE),
)

ENTITY_STOPWORDS = {
    'a',
    'an',
    'the',
    'me',
    'you',
    'us',
    'them',
    'it',
    'this',
    'that',
    'someone',
    'something',
    'anything',
    'everything',
    'who',
    'what',
    'why',
    'how',
    'about',
    'like',
    'with',
    'to',
    'for',
    'про',
    'как',
    'что',
    'кто',
    'day',
    'night',
    'today',
    'tomorrow',
    'yesterday',
}

ENTITY_BREAKWORDS = {'the', 'a', 'an'}
ENTITY_DESCRIPTOR_WORDS = {
    'fictional',
    'character',
    'vampire',
    'nobleman',
    'scientist',
    'physicist',
    'wizard',
    'monster',
    'profession',
    'object',
    'phenomenon',
    'concept',
}

QUESTION_WORDS = ('who', 'what', 'why', 'how', 'when', 'where', 'кто', 'что', 'почему', 'как', 'когда', 'где')
INSULT_WORDS = (
    'stupid',
    'idiot',
    'pathetic',
    'disgusting',
    'hate',
    'trash',
    'ненавиж',
    'туп',
    'дурак',
    'мерзк',
    'жалк',
)
ANGER_WORDS = ('angry', 'furious', 'mad', 'rage', 'pissed', 'злой', 'ярость', 'бешен', 'сердит')
DISTRESS_WORDS = (
    'sad',
    'depressed',
    'hopeless',
    'hurt',
    'cry',
    'crying',
    'lost',
    'terrible',
    'afraid',
    'scared',
    'help me',
    'боюсь',
    'плохо',
    'груст',
    'тяжело',
    'одиноко',
)
HELP_WORDS = ('please', 'help', 'sorry', 'support', 'помоги', 'пожалуйста', 'извини', 'нужна помощь')
CELEBRATORY_WORDS = ('happy', 'glad', 'delighted', 'thrilled', 'excited', 'enjoyed', 'loved', 'рада', 'счастлив', 'доволен')
MORAL_VIOLATION_WORDS = (
    'kill',
    'killed',
    'hurt',
    'harm',
    'abuse',
    'torture',
    'stole',
    'steal',
    'murder',
    'violence',
    'pain',
    'убил',
    'навред',
    'избил',
    'мучил',
    'украл',
)
SECOND_PERSON_HINTS = {'you', 'your', 'yourself', 'ты', 'тебя', 'твой', 'вам', 'дու', 'քեզ', 'քո'}
SELF_REFERENCE_HINTS = {'i', 'me', 'my', 'myself', 'я', 'мне', 'мой', 'меня', 'ես', 'ինձ', 'իմ'}


def _clean_message(message: str) -> str:
    return ' '.join(str(message or '').strip().split())


def _capitalized_entities(message: str) -> list[str]:
    matches = re.findall(r'\b[A-ZА-Я][\w-]*(?:\s+[A-ZА-Я][\w-]*){0,3}\b', message)
    return [match.strip() for match in matches if len(match.strip()) > 1]


def _clean_candidate_phrase(value: str) -> str:
    parts = [part.strip(" .,!?;:'\"") for part in str(value or '').split()]
    lowered = [normalize_name(part) for part in parts]
    for index, token in enumerate(lowered[1:], start=1):
        if token in ENTITY_BREAKWORDS:
            parts = parts[:index]
            lowered = lowered[:index]
            break
    if len(parts) > 1 and any(token in ENTITY_DESCRIPTOR_WORDS for token in lowered[1:]):
        parts = parts[:1]
    filtered = [part for part in parts if normalize_name(part) and normalize_name(part) not in ENTITY_STOPWORDS]
    return ' '.join(filtered[:4]).strip()


def _match_known_entities(message: str, known_entities: list[dict[str, Any]]) -> list[str]:
    lowered = normalize_name(message)
    matches: list[str] = []
    for entity in known_entities:
        names = [str(entity.get('name') or '')] + [str(item) for item in list(entity.get('aliases') or [])]
        for name in names:
            token = normalize_name(name)
            if token and token in lowered:
                matches.append(str(entity.get('name') or name).strip())
                break
    return matches


def _extract_entities(message: str, known_entities: list[dict[str, Any]]) -> list[MessageEntity]:
    candidates: list[str] = []
    for pattern in ENTITY_PATTERNS:
        candidates.extend(_clean_candidate_phrase(match) for match in pattern.findall(message) if str(match).strip())
    candidates.extend(_capitalized_entities(message))
    candidates.extend(_match_known_entities(message, known_entities))

    ordered: list[MessageEntity] = []
    seen: set[str] = set()
    for candidate in candidates:
        clean = _clean_candidate_phrase(candidate)
        token = normalize_name(clean)
        if not clean or not token or token in seen:
            continue
        seen.add(token)
        ordered.append(MessageEntity(name=clean, source_text=clean))
    return ordered


def _detect_situation(message: str, primary_entity: str) -> str:
    stripped = _clean_message(message)
    if not stripped:
        return ''
    if primary_entity:
        return f'{primary_entity}: {stripped}'
    return stripped


def _contains_any(text: str, values: tuple[str, ...]) -> float:
    lowered = normalize_name(text)
    return float(any(normalize_name(value) and normalize_name(value) in lowered for value in values))


def _token_overlap(text: str, values: set[str]) -> float:
    tokens = {token for token in normalize_name(text).split() if token}
    return float(bool(tokens & values))


def _analyze_user_state(message: str, *, primary_entity: str = '', selected_head: str = '') -> UserState:
    clean = _clean_message(message)
    lowered = normalize_name(clean)
    persona_token = normalize_name(selected_head or primary_entity)
    signals = {
        'contains_question': float('?' in clean or any(word in lowered for word in QUESTION_WORDS)),
        'contains_insult': _contains_any(clean, INSULT_WORDS),
        'contains_anger': _contains_any(clean, ANGER_WORDS),
        'contains_distress': _contains_any(clean, DISTRESS_WORDS),
        'contains_help_request': _contains_any(clean, HELP_WORDS),
        'contains_celebratory': _contains_any(clean, CELEBRATORY_WORDS),
        'contains_moral_violation': _contains_any(clean, MORAL_VIOLATION_WORDS),
        'contains_persona_reference': float(bool(persona_token and persona_token in lowered) or _token_overlap(clean, SECOND_PERSON_HINTS)),
        'contains_self_reference': _token_overlap(clean, SELF_REFERENCE_HINTS),
    }

    if signals['contains_distress']:
        tone = 'distressed'
    elif signals['contains_anger'] or signals['contains_insult']:
        tone = 'angry'
    elif signals['contains_celebratory']:
        tone = 'celebratory'
    elif signals['contains_question']:
        tone = 'inquisitive'
    else:
        tone = 'neutral'

    if signals['contains_insult']:
        intent = 'insult'
    elif signals['contains_distress'] or signals['contains_help_request']:
        intent = 'seek_support'
    elif signals['contains_moral_violation']:
        intent = 'confession'
    elif signals['contains_question']:
        intent = 'question'
    else:
        intent = 'statement'

    return UserState(
        language=detect_language_code(clean, fallback='en'),
        tone=tone,
        intent=intent,
        signals=signals,
    )


def analyze_message_state(
    *,
    message: str,
    session_id: str,
    selected_head: str = '',
    current_entity: str = '',
    explicit_context: str = '',
    known_entities: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    clean = _clean_message(message)
    known = list(known_entities or [])
    entities = _extract_entities(clean, known)

    if selected_head.strip():
        selected = selected_head.strip()
        selected_token = normalize_name(selected)
        if all(normalize_name(entity.name) != selected_token for entity in entities):
            entities.insert(0, MessageEntity(name=selected, source_text=selected))

    primary = selected_head.strip()
    if not primary and entities:
        primary = entities[0].name
    if not primary:
        primary = current_entity.strip()

    user_state = _analyze_user_state(clean, primary_entity=primary, selected_head=selected_head.strip())
    return {
        'message': clean,
        'session_id': session_id,
        'selected_head': selected_head.strip(),
        'primary_entity': primary,
        'current_entity': current_entity.strip(),
        'explicit_context': explicit_context.strip(),
        'entities': entities,
        'user_state': user_state,
    }


def analyze_message(
    *,
    message: str,
    session_id: str,
    selected_head: str = '',
    current_entity: str = '',
    explicit_context: str = '',
    known_entities: list[dict[str, Any]] | None = None,
) -> MessageAnalysis:
    prepared = analyze_message_state(
        message=message,
        session_id=session_id,
        selected_head=selected_head,
        current_entity=current_entity,
        explicit_context=explicit_context,
        known_entities=known_entities,
    )
    situation = model_situation(
        message=str(prepared['message'] or ''),
        primary_entity=str(prepared['primary_entity'] or ''),
        selected_head=str(prepared['selected_head'] or ''),
        user_state=prepared['user_state'],
    )
    return MessageAnalysis(
        message=str(prepared['message'] or ''),
        session_id=session_id,
        selected_head=str(prepared['selected_head'] or ''),
        primary_entity=str(prepared['primary_entity'] or ''),
        current_entity=str(prepared['current_entity'] or ''),
        explicit_context=str(prepared['explicit_context'] or ''),
        entities=list(prepared['entities'] or []),
        user_state=prepared['user_state'],
        situation=situation,
    )
