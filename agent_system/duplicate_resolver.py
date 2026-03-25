from __future__ import annotations

import json
import math
from difflib import SequenceMatcher
from typing import Any

STOP_TOKENS = {'mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'madam', 'the', 'a', 'an'}
SEMANTIC_EQUIVALENTS = {
    'human': 'human',
    'humans': 'human',
    'human being': 'human',
    'human beings': 'human',
    'person': 'human',
    'people': 'human',
    'человек': 'human',
    'люди': 'human',
    'мարդ': 'human',
    'մարդիկ': 'human',
    'sunlight': 'sunlight',
    'daylight': 'sunlight',
    'solar light': 'sunlight',
    'солнечный свет': 'sunlight',
    'свет солнца': 'sunlight',
    'արևի լույս': 'sunlight',
}


def normalize_name(value: str) -> str:
    lowered = ''.join(char.lower() if char.isalnum() else ' ' for char in str(value or ''))
    tokens = [token for token in lowered.split() if token and token not in STOP_TOKENS]
    return ' '.join(tokens).strip()


def semantic_normalize_name(value: str) -> str:
    normalized = normalize_name(value)
    return SEMANTIC_EQUIVALENTS.get(normalized, normalized)


def token_similarity(left: str, right: str) -> float:
    left_tokens = set(semantic_normalize_name(left).split())
    right_tokens = set(semantic_normalize_name(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


def text_similarity(left: str, right: str) -> float:
    left_norm = semantic_normalize_name(left)
    right_norm = semantic_normalize_name(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def name_similarity(left: str, right: str) -> float:
    left_norm = semantic_normalize_name(left)
    right_norm = semantic_normalize_name(right)
    if left_norm and right_norm and (left_norm in right_norm or right_norm in left_norm):
        overlap = token_similarity(left, right)
        if overlap > 0:
            return round(max(0.92, overlap), 4)
    return round(max(token_similarity(left, right), text_similarity(left, right)), 4)


def _context_blob(payload: dict[str, Any]) -> str:
    context = payload.get('context') if isinstance(payload.get('context'), dict) else {}
    parts = [
        str(payload.get('description') or ''),
        json.dumps(context, ensure_ascii=False, sort_keys=True),
        json.dumps(payload.get('aliases') or [], ensure_ascii=False, sort_keys=True),
        str(payload.get('folder') or ''),
    ]
    return ' '.join(part for part in parts if part).strip()


def context_similarity(existing: dict[str, Any], candidate: dict[str, Any]) -> float:
    left = _context_blob(existing)
    right = _context_blob(candidate)
    if not left or not right:
        return 0.0
    left_tokens = {token for token in normalize_name(left).split() if token}
    right_tokens = {token for token in normalize_name(right).split() if token}
    overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    ratio = SequenceMatcher(None, left, right).ratio()
    return round(max(overlap, ratio), 4)


def combined_similarity(existing: dict[str, Any], candidate: dict[str, Any]) -> float:
    if semantic_normalize_name(str(existing.get('name') or '')) == semantic_normalize_name(str(candidate.get('name') or '')):
        return 1.0
    name_score = name_similarity(str(existing.get('name') or ''), str(candidate.get('name') or ''))
    context_score = context_similarity(existing, candidate)
    aliases_left = ' '.join(str(item) for item in list(existing.get('aliases') or []))
    aliases_right = ' '.join(str(item) for item in list(candidate.get('aliases') or []))
    alias_score = name_similarity(aliases_left, aliases_right) if aliases_left and aliases_right else 0.0
    weighted = (name_score * 0.6) + (context_score * 0.3) + (alias_score * 0.1)
    return round(weighted, 4)


def should_merge(existing: dict[str, Any], candidate: dict[str, Any], *, threshold: float = 0.88) -> bool:
    existing_type = str(existing.get('type') or '').upper()
    candidate_type = str(candidate.get('type') or '').upper()
    if existing_type and candidate_type and existing_type != candidate_type:
        return False
    return combined_similarity(existing, candidate) >= threshold


def merge_aliases(*values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for group in values:
        for item in group:
            clean = str(item or '').strip()
            token = normalize_name(clean)
            if not clean or not token or token in seen:
                continue
            seen.add(token)
            ordered.append(clean)
    return ordered


def canonical_name(*values: str) -> str:
    candidates = []
    seen: set[str] = set()
    for value in values:
        clean = str(value or '').strip()
        token = normalize_name(clean)
        if not clean or not token or token in seen:
            continue
        seen.add(token)
        candidates.append(clean)
    if not candidates:
        return ''

    def score(value: str) -> tuple[int, int, int, int, str]:
        normalized = normalize_name(value)
        semantic = semantic_normalize_name(value)
        is_semantic_canonical = int(bool(normalized and normalized == semantic))
        ascii_alpha = sum(1 for char in value if char.isascii() and char.isalpha())
        token_count = len(normalized.split()) if normalized else 99
        return (
            is_semantic_canonical,
            int(ascii_alpha > 0),
            -token_count,
            -len(value),
            value.lower(),
        )

    return max(candidates, key=score)


def similarity_bucket(score: float) -> str:
    if score >= 0.95:
        return 'exactish'
    if score >= 0.88:
        return 'merge'
    if score >= 0.72:
        return 'review'
    return 'distinct'


def relevance_decay(value: float, *, factor: float = 0.99) -> float:
    return round(max(0.0, float(value or 0.0) * factor), 6)


def score_node(node: dict[str, Any]) -> float:
    importance = float(node.get('importance') or 0.0)
    confidence = max(float(node.get('confidence') or 0.0), 0.0)
    frequency = max(int(node.get('frequency') or 0), 1)
    return round(importance * confidence * max(math.log(frequency), 0.1), 6)
