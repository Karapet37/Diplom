from __future__ import annotations

import re
from typing import Any

from .classifier_forest import DEFAULT_CLASSIFIER
from .concept_graphs import concept_graph_extraction
from .duplicate_resolver import normalize_name
from .feature_extractor import extract_features
from .message_analyzer import analyze_message
from .models import ENTITY_TYPES, MessageEntity
from .prompt_builder import build_entity_extraction_prompt
from .runtime_config import get_runtime_config
from . import llm

_CLAUSE_MARKERS = {
    'is',
    'are',
    'was',
    'were',
    'be',
    'being',
    'means',
    'because',
    'that',
    'which',
    'who',
    'where',
    'when',
    'does',
    'doing',
    'делает',
    'делают',
    'является',
    'который',
    'которые',
    'потому',
    'что',
    'как',
}

_NON_ENTITY_TOKENS = {
    'build',
    'show',
    'create',
    'make',
    'tell',
    'explain',
    'draw',
    'list',
    'построй',
    'сделай',
    'покажи',
    'объясни',
    'опиши',
    'нарисуй',
    'дай',
}


def _clean_entity_name(value: Any) -> str:
    return ' '.join(str(value or '').replace('\n', ' ').split()).strip(" \t\r\n'\".,;:!?")


def _is_fragment_like_name(value: str) -> bool:
    clean = _clean_entity_name(value)
    if not clean:
        return True
    if any(marker in clean for marker in (',', ';', ':', '!', '?', '\n')):
        return True
    normalized = normalize_name(clean)
    tokens = [token for token in normalized.split() if token]
    if not tokens:
        return True
    if len(tokens) == 1 and tokens[0] in _NON_ENTITY_TOKENS:
        return True
    if tokens[0] in _NON_ENTITY_TOKENS and len(tokens) > 1:
        return True
    if len(tokens) > 5 or len(clean) > 64:
        return True
    if len(tokens) >= 3 and any(token in _CLAUSE_MARKERS for token in tokens[1:]):
        return True
    return False


def _merge_extractions(*payloads: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    seen_entities: set[str] = set()
    seen_relations: set[tuple[str, str, str]] = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for entity in list(payload.get('entities') or []):
            if not isinstance(entity, dict):
                continue
            token = normalize_name(str(entity.get('name') or ''))
            if not token or token in seen_entities:
                continue
            seen_entities.add(token)
            entities.append(entity)
        for relation in list(payload.get('relations') or []):
            if not isinstance(relation, dict):
                continue
            key = (
                normalize_name(str(relation.get('from') or '')),
                str(relation.get('type') or '').strip().upper(),
                normalize_name(str(relation.get('to') or relation.get('target') or '')),
            )
            if not all(key) or key in seen_relations:
                continue
            seen_relations.add(key)
            relations.append(relation)
    return {'entities': entities, 'relations': relations}


def _normalize_relation(item: dict[str, Any]) -> dict[str, Any] | None:
    src = str(item.get('from') or '').strip()
    dst = str(item.get('to') or item.get('target') or '').strip()
    relation_type = str(item.get('type') or 'RELATED_TO').strip().upper()
    if not src or not dst or _is_fragment_like_name(src) or _is_fragment_like_name(dst):
        return None
    return {
        'from': _clean_entity_name(src),
        'to': _clean_entity_name(dst),
        'type': relation_type or 'RELATED_TO',
        'weight': float(item.get('weight') or item.get('strength') or 0.7),
        'confidence': float(item.get('confidence') or 0.75),
    }


def _heuristic_entities(text: str) -> list[dict[str, Any]]:
    candidates = re.findall(r'\b[A-ZА-Я][\w-]*(?:\s+[A-ZА-Я][\w-]*){0,3}\b', str(text or ''))
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()
    analysis = analyze_message(message=str(text or ''), session_id='heuristic')
    for candidate in candidates[:8]:
        token = _clean_entity_name(candidate)
        if _is_fragment_like_name(token):
            continue
        if token.lower() in seen:
            continue
        seen.add(token.lower())
        entity = MessageEntity(name=token, source_text=token)
        features = extract_features(entity, analysis)
        decision = DEFAULT_CLASSIFIER.classify(features)
        entities.append(
            {
                'name': token,
                'aliases': [],
                'type': decision.entity_type,
                'description': f'Extracted from text about {token}.',
                'facts': [],
                'confidence': decision.confidence,
                'context': {'source': 'heuristic'},
            }
        )
    return entities


def _resolved_entity_type(item: dict[str, Any], decision_type: str, feature_map: dict[str, float]) -> str:
    declared = str(item.get('type') or '').strip().upper()
    if decision_type == 'FICTIONAL_CHARACTER' and feature_map.get('fictional_hint_score', 0.0) > 0:
        return decision_type
    if decision_type == 'PROFESSION' and feature_map.get('profession_hint_score', 0.0) > 0:
        return decision_type
    if decision_type == 'PERSON' and (
        feature_map.get('person_hint_score', 0.0) > 0 or feature_map.get('title_case_ratio', 0.0) >= 0.75
    ):
        return decision_type
    if decision_type == 'PHENOMENON' and feature_map.get('phenomenon_hint_score', 0.0) > 0:
        return decision_type
    if decision_type == 'OBJECT' and feature_map.get('object_hint_score', 0.0) > 0:
        return decision_type
    if decision_type == 'CONCEPT' and feature_map.get('concept_hint_score', 0.0) > 0:
        return decision_type
    if declared in ENTITY_TYPES:
        return declared
    return 'CONCEPT'


def validate_extraction(payload: Any, *, source: str) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        return {'entities': [], 'relations': []}
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []

    for item in list(payload.get('entities') or []):
        if not isinstance(item, dict):
            continue
        name = _clean_entity_name(item.get('name') or '')
        if not name or _is_fragment_like_name(name):
            continue
        description = str(item.get('description') or '').strip()
        analysis = analyze_message(message=f'{name}. {description}', session_id='extract', explicit_context=str(item.get('facts') or ''))
        features = extract_features(MessageEntity(name=name, description=description, aliases=list(item.get('aliases') or [])), analysis)
        decision = DEFAULT_CLASSIFIER.classify(features)
        entity_type = _resolved_entity_type(item, decision.entity_type, features.feature_map)
        entities.append(
            {
                'id': f"{entity_type.lower()}:{re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_') or 'entity'}",
                'name': name,
                'aliases': [str(alias).strip() for alias in list(item.get('aliases') or []) if str(alias).strip()],
                'translation_line': str(item.get('translation_line') or '').strip(),
                'type': entity_type,
                'description': description or f'Knowledge extracted for {name}.',
                'facts': [str(fact).strip() for fact in list(item.get('facts') or []) if str(fact).strip()],
                'importance': float(item.get('importance') or 0.7),
                'confidence': max(decision.confidence, float(item.get('confidence') or 0.6)),
                'frequency': int(item.get('frequency') or 1),
                'context': {**dict(item.get('context') or {}), 'source': source},
            }
        )
    for item in list(payload.get('relations') or []):
        if not isinstance(item, dict):
            continue
        relation = _normalize_relation(item)
        if relation:
            relations.append(relation)
    return {'entities': entities, 'relations': relations}


def extract_knowledge(text: str, *, source: str = 'session') -> dict[str, list[dict[str, Any]]]:
    deterministic = validate_extraction(concept_graph_extraction(text, source=source), source=source)
    raw = llm._call_model_compat(
        build_entity_extraction_prompt(text, source=source),
        mode='knowledge',
        role=get_runtime_config().roles.extraction,
    )
    payload = llm.extract_json_block(raw)
    validated = validate_extraction(payload, source=source)
    combined = _merge_extractions(deterministic, validated)
    if combined['entities'] or combined['relations']:
        return combined
    if str(raw or '').strip():
        return deterministic if deterministic['entities'] or deterministic['relations'] else {'entities': [], 'relations': []}
    heuristic = {'entities': _heuristic_entities(text), 'relations': []}
    return _merge_extractions(deterministic, heuristic)
