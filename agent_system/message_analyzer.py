from __future__ import annotations

import re
from typing import Any

from .duplicate_resolver import normalize_name
from .models import MessageAnalysis, MessageEntity

ENTITY_PATTERNS = (
    re.compile(r'"([^"]{2,80})"'),
    re.compile(r"'([^']{2,80})'"),
    re.compile(r'\b(?:about|like|as|with|to|for)\s+([A-ZА-Я][\w-]*(?:\s+[A-ZА-Я][\w-]*){0,3})'),
    re.compile(r'\b(?:about|like|as|with|to|for|regarding|around|про|о|как)\s+([A-Za-zА-Яа-я][\w-]*(?:\s+[A-Za-zА-Яа-я][\w-]*){0,3})', re.IGNORECASE),
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


def analyze_message(
    *,
    message: str,
    session_id: str,
    selected_head: str = '',
    current_entity: str = '',
    explicit_context: str = '',
    known_entities: list[dict[str, Any]] | None = None,
) -> MessageAnalysis:
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

    cues = {
        'contains_question': float('?' in clean or any(word in clean.lower() for word in ('who', 'what', 'why', 'как', 'кто', 'что', 'почему'))),
        'contains_insult': float(any(word in clean.lower() for word in ('stupid', 'idiot', 'hate', 'ненавиж', 'туп', 'дурак'))),
        'contains_fear': float(any(word in clean.lower() for word in ('afraid', 'fear', 'scared', 'боюсь', 'страх'))),
        'contains_empathy': float(any(word in clean.lower() for word in ('please', 'help', 'sorry', 'помоги', 'пожалуйста', 'извини'))),
    }
    return MessageAnalysis(
        message=clean,
        session_id=session_id,
        selected_head=selected_head.strip(),
        primary_entity=primary,
        situation=_detect_situation(clean, primary),
        current_entity=current_entity.strip(),
        explicit_context=explicit_context.strip(),
        entities=entities,
        cues=cues,
    )
