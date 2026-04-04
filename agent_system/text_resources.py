from __future__ import annotations

import re
from functools import lru_cache

SECOND_PERSON_TOKENS = {'you', 'your', 'yourself', 'ты', 'тебя', 'твой', 'вам', 'дու', 'քեզ', 'քո'}
SELF_REFERENCE_TOKENS = {'i', 'me', 'my', 'myself', 'я', 'мне', 'мой', 'меня', 'ես', 'ինձ', 'իմ'}

ENTITY_PATTERN_SPECS: tuple[tuple[str, int], ...] = (
    (r'"([^"]{2,80})"', 0),
    (r"'([^']{2,80})'", 0),
    (r'\b(?:about|like|as|with|to|for)\s+([A-ZА-Я][\w-]*(?:\s+[A-ZА-Я][\w-]*){0,3})', 0),
    (r'\b(?:about|regarding|around|про|о|как)\s+([A-Za-zА-Яа-я][\w-]*(?:\s+[A-Za-zА-Яа-я][\w-]*){0,3})', re.IGNORECASE),
)
CAPITALIZED_ENTITY_PATTERN = re.compile(r'\b[A-ZА-Я][\w-]*(?:\s+[A-ZА-Я][\w-]*){0,3}\b')


@lru_cache(maxsize=1)
def compiled_entity_patterns() -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, flags) for pattern, flags in ENTITY_PATTERN_SPECS)


def capitalized_entity_phrases(text: str) -> list[str]:
    return [match.strip() for match in CAPITALIZED_ENTITY_PATTERN.findall(str(text or '')) if len(match.strip()) > 1]
