from __future__ import annotations

import re
from typing import Any

from .classifier_forest import DEFAULT_CLASSIFIER
from .feature_extractor import extract_features
from .message_analyzer import analyze_message
from .models import ENTITY_TYPES, MessageEntity
from .prompt_builder import build_entity_extraction_prompt
from . import llm


def _normalize_relation(item: dict[str, Any]) -> dict[str, Any] | None:
    src = str(item.get('from') or '').strip()
    dst = str(item.get('to') or item.get('target') or '').strip()
    relation_type = str(item.get('type') or 'RELATED_TO').strip().upper()
    if not src or not dst:
        return None
    return {
        'from': src,
        'to': dst,
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
        token = candidate.strip()
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
    if not entities and str(text or '').strip():
        entity = MessageEntity(name=str(text or '').split()[0], source_text=str(text or ''))
        features = extract_features(entity, analysis)
        decision = DEFAULT_CLASSIFIER.classify(features)
        entities.append(
            {
                'name': entity.name,
                'aliases': [],
                'type': decision.entity_type,
                'description': f'Extracted from text about {entity.name}.',
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
        name = str(item.get('name') or '').strip()
        if not name:
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
    raw = llm._call_model(build_entity_extraction_prompt(text, source=source), mode='knowledge')
    payload = llm.extract_json_block(raw)
    validated = validate_extraction(payload, source=source)
    if validated['entities'] or validated['relations']:
        return validated
    if str(raw or '').strip():
        return {'entities': [], 'relations': []}
    return {'entities': _heuristic_entities(text), 'relations': []}
