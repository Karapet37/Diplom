from __future__ import annotations

from typing import Iterable

from .duplicate_resolver import normalize_name
from .models import EntityFeatures, MessageAnalysis, MessageEntity

PERSON_HINTS = {'he', 'she', 'they', 'friend', 'person', 'human', 'man', 'woman', 'neighbor', 'scientist', 'actor', 'leader'}
FICTIONAL_HINTS = {'fictional', 'character', 'vampire', 'wizard', 'dragon', 'hero', 'villain', 'count', 'monster', 'myth'}
PROFESSION_HINTS = {'engineer', 'doctor', 'physicist', 'teacher', 'programmer', 'artist', 'scientist', 'profession', 'job', 'career'}
CONCEPT_HINTS = {'idea', 'concept', 'theory', 'justice', 'freedom', 'ethics', 'meaning', 'philosophy', 'algorithm'}
PHENOMENON_HINTS = {'storm', 'rain', 'hurricane', 'earthquake', 'fire', 'inflation', 'gravity', 'weather', 'phenomenon'}
OBJECT_HINTS = {'phone', 'book', 'castle', 'sword', 'artifact', 'chair', 'computer', 'machine', 'object'}


def _keyword_score(text: str, keywords: Iterable[str]) -> float:
    text_tokens = [token for token in normalize_name(text).split() if token]
    token_set = set(text_tokens)
    score = 0
    for keyword in keywords:
        normalized = normalize_name(keyword)
        if not normalized:
            continue
        keyword_tokens = [token for token in normalized.split() if token]
        if not keyword_tokens:
            continue
        if len(keyword_tokens) == 1:
            score += int(keyword_tokens[0] in token_set)
            continue
        width = len(keyword_tokens)
        for index in range(0, max(len(text_tokens) - width + 1, 0)):
            if text_tokens[index : index + width] == keyword_tokens:
                score += 1
                break
    return float(score)


def _title_case_score(value: str) -> float:
    words = [part for part in str(value or '').split() if part]
    if not words:
        return 0.0
    titled = sum(1 for word in words if word[:1].isupper())
    return titled / len(words)


def extract_features(entity: MessageEntity, analysis: MessageAnalysis) -> EntityFeatures:
    combined_text = ' '.join(
        part for part in (
            entity.name,
            entity.description,
            analysis.message,
            analysis.explicit_context,
        )
        if str(part).strip()
    )
    normalized_name = normalize_name(entity.name)
    feature_map = {
        'title_case_ratio': round(_title_case_score(entity.name), 4),
        'name_token_count': float(len(normalized_name.split())),
        'message_question_signal': analysis.user_state.signal('contains_question'),
        'person_hint_score': _keyword_score(combined_text, PERSON_HINTS),
        'fictional_hint_score': _keyword_score(combined_text, FICTIONAL_HINTS),
        'profession_hint_score': _keyword_score(combined_text, PROFESSION_HINTS),
        'concept_hint_score': _keyword_score(combined_text, CONCEPT_HINTS),
        'phenomenon_hint_score': _keyword_score(combined_text, PHENOMENON_HINTS),
        'object_hint_score': _keyword_score(combined_text, OBJECT_HINTS),
        'contains_of_title': float(' of ' in normalized_name),
        'contains_role_suffix': float(any(normalized_name.endswith(token) for token in ('er', 'ist', 'or'))),
        'is_single_token': float(len(normalized_name.split()) == 1),
    }
    evidence: list[str] = []
    for key, value in feature_map.items():
        if value > 0:
            evidence.append(f'{key}={value}')
    return EntityFeatures(
        entity_name=entity.name,
        normalized_name=normalized_name,
        description=entity.description,
        feature_map=feature_map,
        evidence=evidence,
    )
