from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .duplicate_resolver import merge_aliases, normalize_name
from .graph_store import (
    GraphStore,
    heads_dir,
    load_json,
    normalize_personality_name,
    personality_graph_path,
    personality_index_path,
    personality_proposal_path,
    personality_proposals_dir,
    write_json,
    write_text,
)
from .llm import call_json_model_for_role
from .memory_layers import append_persona_overflow_archive, list_persona_snapshots, record_persona_snapshot
from .models import (
    EMOTION_KEYS,
    HEAD_ENTITY_TYPES,
    HeadBundle,
    PersonaBaselineDefinition,
    PersonaDynamicState,
    PersonaIndicators,
    PersonaLearnedPatterns,
    PersonaResponseExplanation,
    PersonaSystemModel,
    ReactionOutcome,
    Situation,
)
from .prompt_builder import build_persona_profile_prompt
from .reliability import RecoveryFailure, StorageWriteFailure
from .runtime_config import get_runtime_config
from .situation_engine import situation_summary


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _head_dir(name: str) -> Path:
    return heads_dir() / normalize_personality_name(name)


def _head_file(name: str, filename: str) -> Path:
    return _head_dir(name) / filename


def _baseline_path(name: str) -> Path:
    return _head_file(name, 'baseline.json')


def _dynamic_state_path(name: str) -> Path:
    return _head_file(name, 'dynamic_state.json')


def _learned_patterns_path(name: str) -> Path:
    return _head_file(name, 'learned_patterns.json')


def _revisions_path(name: str) -> Path:
    return _head_file(name, 'revisions.json')


def _capture_persona_storage_state(name: str) -> dict[str, Any]:
    clean = normalize_personality_name(name)
    head_path = _head_dir(clean)
    files: dict[str, dict[str, Any]] = {}
    if head_path.exists():
        for path in sorted(head_path.rglob('*')):
            if not path.is_file():
                continue
            relative = str(path.relative_to(head_path))
            text = path.read_text(encoding='utf-8')
            if path.suffix == '.json':
                try:
                    files[relative] = {'kind': 'json', 'payload': json.loads(text)}
                    continue
                except json.JSONDecodeError:
                    pass
            files[relative] = {'kind': 'text', 'payload': text}
    return {
        'head_exists': head_path.exists(),
        'files': files,
        'index': _load_index(),
    }


def _restore_persona_storage_state(name: str, snapshot: dict[str, Any]) -> None:
    clean = normalize_personality_name(name)
    head_path = _head_dir(clean)
    head_exists = bool(dict(snapshot or {}).get('head_exists'))
    files = dict(dict(snapshot or {}).get('files') or {})
    if not head_exists:
        if head_path.exists():
            shutil.rmtree(head_path)
        _save_index(list(dict(snapshot or {}).get('index') or []))
        return
    head_path.mkdir(parents=True, exist_ok=True)
    current_files = [path for path in head_path.rglob('*') if path.is_file()]
    for path in current_files:
        relative = str(path.relative_to(head_path))
        if relative not in files:
            path.unlink(missing_ok=True)
    for relative, payload in files.items():
        target = head_path / relative
        if str(payload.get('kind') or '') == 'json':
            write_json(target, payload.get('payload'))
        else:
            write_text(target, str(payload.get('payload') or ''))
    for path in sorted([item for item in head_path.rglob('*') if item.is_dir()], reverse=True):
        try:
            path.rmdir()
        except OSError:
            continue
    _save_index(list(dict(snapshot or {}).get('index') or []))


def _execute_persona_mutation(
    name: str,
    *,
    reason: str,
    archive_snapshot: bool,
    graph_snapshot: bool,
    operation: Any,
) -> Any:
    clean = normalize_personality_name(name)
    storage_state = _capture_persona_storage_state(clean)
    persona_snapshot_path = ''
    graph_snapshot_path = ''
    if archive_snapshot:
        persona_snapshot_path = str(
            record_persona_snapshot(
                clean,
                {'storage_state': storage_state},
                reason=reason,
            )
        )
    if graph_snapshot:
        graph_snapshot_path = str(GraphStore().snapshot_graph(reason=f'pre-persona-{reason}:{clean}').get('path') or '')
    try:
        return operation()
    except Exception as exc:
        rollback_errors: dict[str, str] = {}
        graph_restore_result: dict[str, Any] = {}
        try:
            _restore_persona_storage_state(clean, storage_state)
        except Exception as restore_exc:
            rollback_errors['persona_restore_error'] = str(restore_exc)
        if graph_snapshot_path:
            try:
                graph_restore_result = GraphStore().restore_snapshot(graph_snapshot_path)
                if not graph_restore_result.get('ok'):
                    rollback_errors['graph_restore_error'] = str(graph_restore_result.get('reason') or 'graph_restore_failed')
            except Exception as restore_exc:
                rollback_errors['graph_restore_error'] = str(restore_exc)
        if rollback_errors:
            raise RecoveryFailure(
                f'Persona mutation failed for {clean} and rollback did not complete.',
                details={
                    'name': clean,
                    'reason': reason,
                    'error': str(exc),
                    'persona_snapshot_path': persona_snapshot_path,
                    'graph_snapshot_path': graph_snapshot_path,
                    **rollback_errors,
                },
            ) from exc
        raise StorageWriteFailure(
            f'Persona mutation failed for {clean}. Previous state was restored.',
            details={
                'name': clean,
                'reason': reason,
                'error': str(exc),
                'persona_snapshot_path': persona_snapshot_path,
                'graph_snapshot_path': graph_snapshot_path,
                'graph_restore_result': graph_restore_result,
            },
        ) from exc


def is_head_entity_type(entity_type: str) -> bool:
    token = str(entity_type or '').strip().upper()
    return token in HEAD_ENTITY_TYPES


def _normalized_head_entity_type(entity_type: str, *, explicit: bool = False) -> str:
    token = str(entity_type or '').strip().upper()
    if token in HEAD_ENTITY_TYPES:
        return token
    return 'PERSON' if explicit else 'CONCEPT'


def _default_emotion_vector() -> dict[str, float]:
    return {
        'anger': 0.1,
        'fear': 0.1,
        'curiosity': 0.55,
        'confidence': 0.55,
        'empathy': 0.45,
    }


def _memory_limits() -> Any:
    return get_runtime_config().memory


def _revision_limit() -> int:
    return max(8, int(_memory_limits().archive_index_limit or 16))


def _normalize_string_list(values: list[Any], *, limit: int) -> list[str]:
    return list(
        dict.fromkeys(str(item).strip() for item in list(values or []) if str(item).strip())
    )[:limit]


def _normalized_emotion_vector(payload: Any) -> dict[str, float]:
    vector = _default_emotion_vector()
    source = payload if isinstance(payload, dict) else {}
    fallback = source.get('trait_vector') if isinstance(source.get('trait_vector'), dict) else {}
    for key in EMOTION_KEYS:
        value = source.get(key, fallback.get(key))
        if value is None and key == 'confidence':
            value = fallback.get('confidence')
        if value is None and key == 'empathy':
            value = fallback.get('empathy')
        if value is None and key == 'curiosity':
            value = fallback.get('curiosity')
        if value is None and key == 'anger':
            value = fallback.get('aggression')
        if value is None and key == 'fear':
            value = fallback.get('fear')
        if value is None:
            continue
        try:
            vector[key] = round(min(1.0, max(0.0, float(value))), 4)
        except (TypeError, ValueError):
            continue
    return vector


def _validate_relations(values: list[Any]) -> list[dict[str, Any]]:
    limit = _memory_limits().persona_relation_limit
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(values or []):
        if not isinstance(item, dict):
            continue
        relation_type = str(item.get('type') or '').strip().upper()
        target = str(item.get('target') or item.get('to') or '').strip()
        if not relation_type or not target:
            continue
        key = f'{relation_type}::{normalize_name(target)}'
        if key in seen:
            continue
        seen.add(key)
        rows.append({'type': relation_type, 'target': target, 'weight': float(item.get('weight') or 0.8)})
    return rows[:limit]


def _default_revision_payload() -> dict[str, Any]:
    return {
        'current': {
            'revision': 1,
            'baseline_revision': 1,
            'dynamic_revision': 1,
            'learned_revision': 1,
        },
        'history': [],
    }


def _load_revision_payload(name: str) -> dict[str, Any]:
    payload = load_json(_revisions_path(name), _default_revision_payload())
    if not isinstance(payload, dict):
        return _default_revision_payload()
    current = payload.get('current') if isinstance(payload.get('current'), dict) else {}
    history = payload.get('history') if isinstance(payload.get('history'), list) else []
    merged = _default_revision_payload()
    merged['current'].update(current)
    merged['history'] = [dict(item) for item in history if isinstance(item, dict)]
    return merged


def _derive_baseline_definition(
    name: str,
    *,
    entity_type: str,
    traits: list[str],
    aliases: list[str],
    relations: list[dict[str, Any]],
    knowledge: str,
    meta: dict[str, Any],
) -> PersonaBaselineDefinition:
    return PersonaBaselineDefinition(
        name=str(meta.get('name') or name),
        slug=str(meta.get('slug') or normalize_personality_name(name)),
        entity_type=str(entity_type or 'CONCEPT'),
        traits=_normalize_string_list(traits, limit=_memory_limits().persona_trait_limit),
        aliases=merge_aliases(list(aliases or []), list(meta.get('aliases') or [])),
        relations=_validate_relations(relations),
        knowledge=str(knowledge or '').strip()[: _memory_limits().persona_knowledge_char_limit],
        revision=int(meta.get('baseline_revision') or 1),
        updated_at=str(meta.get('last_baseline_update_at') or meta.get('updated_at') or ''),
        source=str(meta.get('source') or 'legacy'),
    )


def _derive_dynamic_state(
    *,
    emotion_vector: dict[str, Any],
    meta: dict[str, Any],
) -> PersonaDynamicState:
    return PersonaDynamicState(
        emotion_vector=_normalized_emotion_vector(emotion_vector),
        last_situation=str(meta.get('last_situation') or '').strip(),
        last_response_style=str(meta.get('last_response_style') or '').strip(),
        revision=int(meta.get('dynamic_revision') or 1),
        updated_at=str(meta.get('last_dynamic_update_at') or meta.get('updated_at') or ''),
    )


def _derive_learned_patterns(
    *,
    examples: list[str],
    situation_reactions: list[dict[str, Any]],
    log_tuples: list[dict[str, Any]],
    persona_form: dict[str, Any],
    decision_explanation: str,
    traits: list[str],
    meta: dict[str, Any],
) -> PersonaLearnedPatterns:
    return PersonaLearnedPatterns(
        examples=_normalize_string_list(examples, limit=_memory_limits().persona_example_limit),
        situation_reactions=[dict(item) for item in list(situation_reactions or []) if isinstance(item, dict)][
            : _memory_limits().persona_reaction_limit
        ],
        log_tuples=[dict(item) for item in list(log_tuples or []) if isinstance(item, dict)][
            : _memory_limits().persona_log_tuple_limit
        ],
        persona_form=dict(persona_form or {}),
        decision_explanation=str(decision_explanation or '').strip()[:600],
        learned_traits=_normalize_string_list(list(traits or []), limit=_memory_limits().persona_trait_limit),
        revision=int(meta.get('learned_revision') or 1),
        updated_at=str(meta.get('last_learned_update_at') or meta.get('updated_at') or ''),
    )


def _compute_persona_indicators(
    baseline: PersonaBaselineDefinition,
    dynamic: PersonaDynamicState,
    learned: PersonaLearnedPatterns,
    revision_meta: dict[str, Any],
) -> PersonaIndicators:
    log_frequency = sum(max(1, int(item.get('frequency') or 1)) for item in learned.log_tuples if isinstance(item, dict))
    evidence_count = len(learned.examples) + len(learned.situation_reactions) + log_frequency
    learned_pattern_count = len(learned.situation_reactions) + len(learned.log_tuples) + len(learned.learned_traits)
    revision_count = int(revision_meta.get('revision') or 1)
    confidence_score = min(
        1.0,
        0.2 * min(1.0, len(baseline.traits) / 4.0)
        + 0.15 * min(1.0, len(baseline.relations) / 4.0)
        + 0.15 * (1.0 if baseline.knowledge else 0.0)
        + 0.25 * min(1.0, evidence_count / 10.0)
        + 0.1 * (1.0 if learned.persona_form else 0.0)
        + 0.1 * (1.0 if learned.decision_explanation else 0.0)
        + 0.05 * min(1.0, revision_count / 6.0),
    )
    maturity_score = min(
        1.0,
        0.3 * min(1.0, evidence_count / 12.0)
        + 0.2 * min(1.0, learned_pattern_count / 8.0)
        + 0.15 * min(1.0, revision_count / 8.0)
        + 0.15 * (1.0 if learned.persona_form else 0.0)
        + 0.1 * (1.0 if learned.decision_explanation else 0.0)
        + 0.1 * (1.0 if dynamic.last_response_style else 0.0),
    )
    if maturity_score >= 0.8:
        maturity_level = 'mature'
    elif maturity_score >= 0.6:
        maturity_level = 'stable'
    elif maturity_score >= 0.35:
        maturity_level = 'emerging'
    else:
        maturity_level = 'bootstrap'
    return PersonaIndicators(
        confidence_score=round(confidence_score, 4),
        maturity_score=round(maturity_score, 4),
        maturity_level=maturity_level,
        evidence_count=evidence_count,
        learned_pattern_count=learned_pattern_count,
        adaptation_locked=maturity_level in {'stable', 'mature'},
    )


def _build_revision_meta(meta: dict[str, Any], revision_payload: dict[str, Any]) -> dict[str, Any]:
    current = dict(revision_payload.get('current') or {})
    return {
        'schema_version': int(meta.get('schema_version') or 2),
        'revision': int(meta.get('revision') or current.get('revision') or 1),
        'baseline_revision': int(meta.get('baseline_revision') or current.get('baseline_revision') or 1),
        'dynamic_revision': int(meta.get('dynamic_revision') or current.get('dynamic_revision') or 1),
        'learned_revision': int(meta.get('learned_revision') or current.get('learned_revision') or 1),
        'last_baseline_update_at': str(meta.get('last_baseline_update_at') or ''),
        'last_dynamic_update_at': str(meta.get('last_dynamic_update_at') or ''),
        'last_learned_update_at': str(meta.get('last_learned_update_at') or ''),
        'history': [dict(item) for item in list(revision_payload.get('history') or [])],
    }


def _write_revision_state(name: str, *, layers: dict[str, dict[str, Any]], reason: str) -> dict[str, Any]:
    clean = normalize_personality_name(name)
    timestamp = _utc_now()
    meta = load_json(_head_file(clean, 'meta.json'), {})
    payload = _load_revision_payload(clean)
    current = dict(payload.get('current') or {})
    global_revision = int(meta.get('revision') or current.get('revision') or 1) + 1
    current['revision'] = global_revision
    meta['schema_version'] = 2
    meta['revision'] = global_revision
    meta['updated_at'] = timestamp
    history_entry = {
        'revision': global_revision,
        'timestamp': timestamp,
        'reason': str(reason or '').strip() or 'persona_update',
        'layers': {},
    }
    for layer_name, snapshot in layers.items():
        key = f'{layer_name}_revision'
        layer_revision = int(meta.get(key) or current.get(key) or 1) + 1
        current[key] = layer_revision
        meta[key] = layer_revision
        meta[f'last_{layer_name}_update_at'] = timestamp
        history_entry['layers'][layer_name] = {
            'revision': layer_revision,
            'snapshot': dict(snapshot),
        }
    payload['current'] = current
    history = [history_entry] + [dict(item) for item in list(payload.get('history') or []) if isinstance(item, dict)]
    payload['history'] = history[: _revision_limit()]
    write_json(_revisions_path(clean), payload)
    write_json(_head_file(clean, 'meta.json'), meta)
    return meta


def _write_layered_state(
    name: str,
    *,
    baseline: PersonaBaselineDefinition | None = None,
    dynamic: PersonaDynamicState | None = None,
    learned: PersonaLearnedPatterns | None = None,
    reason: str,
) -> dict[str, Any]:
    clean = normalize_personality_name(name)
    changed_layers: dict[str, dict[str, Any]] = {}
    if baseline is not None:
        write_json(_baseline_path(clean), baseline.to_dict())
        changed_layers['baseline'] = baseline.to_dict()
    if dynamic is not None:
        write_json(_dynamic_state_path(clean), dynamic.to_dict())
        changed_layers['dynamic'] = dynamic.to_dict()
    if learned is not None:
        write_json(_learned_patterns_path(clean), learned.to_dict())
        changed_layers['learned'] = learned.to_dict()
    if not changed_layers:
        return load_json(_head_file(clean, 'meta.json'), {})
    return _write_revision_state(clean, layers=changed_layers, reason=reason)


def _load_index() -> list[str]:
    payload = load_json(personality_index_path(), {'heads': []})
    rows = payload.get('heads') if isinstance(payload, dict) else []
    return [normalize_personality_name(item) for item in list(rows or []) if str(item).strip()]


def _save_index(names: list[str]) -> None:
    ordered: list[str] = []
    seen: set[str] = set()
    for name in names:
        clean = normalize_personality_name(name)
        if clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    write_json(personality_index_path(), {'heads': ordered})


def _ensure_head_files(name: str, *, entity_type: str = 'CONCEPT', aliases: list[str] | None = None, source: str = 'system') -> None:
    head_dir = _head_dir(name)
    head_dir.mkdir(parents=True, exist_ok=True)
    slug = normalize_personality_name(name)
    timestamp = _utc_now()
    defaults: dict[str, Any] = {
        'traits.json': {'traits': [], 'entity_type': entity_type},
        'relations.json': {'aliases': list(aliases or []), 'relations': []},
        'examples.json': {'examples': [], 'situation_reactions': []},
        'emotion_vector.json': _default_emotion_vector(),
        'log_tuples.json': {'items': []},
        'persona_form.json': {},
        'meta.json': {
            'name': name,
            'slug': slug,
            'entity_type': entity_type,
            'folder': str(head_dir),
            'importance': 0.8,
            'frequency': 1,
            'created_at': timestamp,
            'updated_at': timestamp,
            'source': source,
            'aliases': list(aliases or []),
            'schema_version': 2,
            'revision': 1,
            'baseline_revision': 1,
            'dynamic_revision': 1,
            'learned_revision': 1,
            'last_baseline_update_at': timestamp,
            'last_dynamic_update_at': timestamp,
            'last_learned_update_at': timestamp,
        },
        'local_graph.json': {'nodes': [], 'edges': []},
        'baseline.json': {
            'name': name,
            'slug': slug,
            'entity_type': entity_type,
            'traits': [],
            'aliases': list(aliases or []),
            'relations': [],
            'knowledge': '',
            'revision': 1,
            'updated_at': timestamp,
            'source': source,
        },
        'dynamic_state.json': {
            'emotion_vector': _default_emotion_vector(),
            'last_situation': '',
            'last_response_style': '',
            'revision': 1,
            'updated_at': timestamp,
        },
        'learned_patterns.json': {
            'examples': [],
            'situation_reactions': [],
            'log_tuples': [],
            'persona_form': {},
            'decision_explanation': '',
            'learned_traits': [],
            'revision': 1,
            'updated_at': timestamp,
        },
        'revisions.json': {
            'current': {
                'revision': 1,
                'baseline_revision': 1,
                'dynamic_revision': 1,
                'learned_revision': 1,
            },
            'history': [],
        },
    }
    for filename, payload in defaults.items():
        path = _head_file(name, filename)
        if not path.exists():
            write_json(path, payload)
    knowledge_path = _head_file(name, 'knowledge.txt')
    if not knowledge_path.exists():
        write_text(knowledge_path, '')
    decision_path = _head_file(name, 'decision_explanation.txt')
    if not decision_path.exists():
        write_text(decision_path, '')


def persona_exists(name: str) -> bool:
    return _head_dir(name).exists()


def load_persona(name: str) -> HeadBundle | None:
    clean = normalize_personality_name(name)
    if not persona_exists(clean):
        return None
    traits_payload = load_json(_head_file(clean, 'traits.json'), {'traits': []})
    relations_payload = load_json(_head_file(clean, 'relations.json'), {'aliases': [], 'relations': []})
    examples_payload = load_json(_head_file(clean, 'examples.json'), {'examples': [], 'situation_reactions': []})
    emotion_payload = load_json(_head_file(clean, 'emotion_vector.json'), _default_emotion_vector())
    log_payload = load_json(_head_file(clean, 'log_tuples.json'), {'items': []})
    persona_form_payload = load_json(_head_file(clean, 'persona_form.json'), {})
    meta_payload = load_json(_head_file(clean, 'meta.json'), {})
    knowledge = _head_file(clean, 'knowledge.txt').read_text(encoding='utf-8') if _head_file(clean, 'knowledge.txt').exists() else ''
    decision_explanation = (
        _head_file(clean, 'decision_explanation.txt').read_text(encoding='utf-8')
        if _head_file(clean, 'decision_explanation.txt').exists()
        else ''
    )
    baseline_payload = load_json(_baseline_path(clean), None)
    dynamic_payload = load_json(_dynamic_state_path(clean), None)
    learned_payload = load_json(_learned_patterns_path(clean), None)
    revision_payload = _load_revision_payload(clean)
    baseline = _derive_baseline_definition(
        clean,
        entity_type=str(meta_payload.get('entity_type') or traits_payload.get('entity_type') or 'CONCEPT'),
        traits=list(traits_payload.get('traits') or []),
        aliases=list(relations_payload.get('aliases') or []),
        relations=list(relations_payload.get('relations') or []),
        knowledge=knowledge,
        meta=meta_payload,
    )
    if isinstance(baseline_payload, dict):
        baseline = PersonaBaselineDefinition(
            name=str(baseline_payload.get('name') or baseline.name),
            slug=str(baseline_payload.get('slug') or baseline.slug),
            entity_type=str(baseline_payload.get('entity_type') or baseline.entity_type),
            traits=_normalize_string_list(list(baseline_payload.get('traits') or baseline.traits), limit=_memory_limits().persona_trait_limit),
            aliases=merge_aliases(list(baseline_payload.get('aliases') or baseline.aliases)),
            relations=_validate_relations(list(baseline_payload.get('relations') or baseline.relations)),
            knowledge=str(baseline_payload.get('knowledge') or baseline.knowledge).strip()[: _memory_limits().persona_knowledge_char_limit],
            revision=int(baseline_payload.get('revision') or baseline.revision or 1),
            updated_at=str(baseline_payload.get('updated_at') or baseline.updated_at),
            source=str(baseline_payload.get('source') or baseline.source),
        )
    dynamic = _derive_dynamic_state(emotion_vector=emotion_payload, meta=meta_payload)
    if isinstance(dynamic_payload, dict):
        dynamic = PersonaDynamicState(
            emotion_vector=_normalized_emotion_vector(dynamic_payload.get('emotion_vector') or dynamic.emotion_vector),
            last_situation=str(dynamic_payload.get('last_situation') or dynamic.last_situation).strip(),
            last_response_style=str(dynamic_payload.get('last_response_style') or dynamic.last_response_style).strip(),
            revision=int(dynamic_payload.get('revision') or dynamic.revision or 1),
            updated_at=str(dynamic_payload.get('updated_at') or dynamic.updated_at),
        )
    learned = _derive_learned_patterns(
        examples=list(examples_payload.get('examples') or []),
        situation_reactions=[dict(item) for item in list(examples_payload.get('situation_reactions') or []) if isinstance(item, dict)],
        log_tuples=[dict(item) for item in list(log_payload.get('items') or []) if isinstance(item, dict)],
        persona_form=dict(persona_form_payload) if isinstance(persona_form_payload, dict) else {},
        decision_explanation=decision_explanation,
        traits=[],
        meta=meta_payload,
    )
    if isinstance(learned_payload, dict):
        learned = PersonaLearnedPatterns(
            examples=_normalize_string_list(list(learned_payload.get('examples') or learned.examples), limit=_memory_limits().persona_example_limit),
            situation_reactions=[dict(item) for item in list(learned_payload.get('situation_reactions') or learned.situation_reactions) if isinstance(item, dict)],
            log_tuples=[dict(item) for item in list(learned_payload.get('log_tuples') or learned.log_tuples) if isinstance(item, dict)],
            persona_form=dict(learned_payload.get('persona_form') or learned.persona_form)
            if isinstance(learned_payload.get('persona_form') or learned.persona_form, dict)
            else {},
            decision_explanation=str(learned_payload.get('decision_explanation') or learned.decision_explanation).strip()[:600],
            learned_traits=_normalize_string_list(list(learned_payload.get('learned_traits') or learned.learned_traits), limit=_memory_limits().persona_trait_limit),
            revision=int(learned_payload.get('revision') or learned.revision or 1),
            updated_at=str(learned_payload.get('updated_at') or learned.updated_at),
        )
    indicators = _compute_persona_indicators(baseline, dynamic, learned, _build_revision_meta(meta_payload, revision_payload))
    return HeadBundle(
        name=str(meta_payload.get('name') or baseline.name or clean),
        folder=str(_head_dir(clean)),
        entity_type=str(baseline.entity_type or meta_payload.get('entity_type') or 'CONCEPT'),
        traits=list(baseline.traits),
        relations=[dict(item) for item in baseline.relations],
        examples=list(learned.examples),
        situation_reactions=[dict(item) for item in learned.situation_reactions],
        knowledge=str(baseline.knowledge or '').strip(),
        emotion_vector=_normalized_emotion_vector(dynamic.emotion_vector),
        meta=dict(meta_payload),
        log_tuples=[dict(item) for item in learned.log_tuples],
        persona_form=dict(learned.persona_form),
        decision_explanation=str(learned.decision_explanation or '').strip(),
        baseline_definition=baseline,
        dynamic_state=dynamic,
        learned_patterns=learned,
        indicators=indicators,
        revision_meta=_build_revision_meta(meta_payload, revision_payload),
    )


def load_persona_graph(name: str) -> dict[str, Any]:
    payload = load_json(personality_graph_path(name), {'nodes': [], 'edges': []})
    return payload if isinstance(payload, dict) else {'nodes': [], 'edges': []}


def list_personas() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in _load_index():
        bundle = load_persona(name)
        if bundle is None:
            continue
        rows.append(
            {
                'name': bundle.name,
                'slug': bundle.meta.get('slug') or normalize_personality_name(bundle.name),
                'entity_type': bundle.entity_type,
                'emotion_vector': bundle.emotion_vector,
                'folder': bundle.folder,
            }
        )
    return rows


def _detect_traits(examples: list[str]) -> list[str]:
    text = ' '.join(examples).lower()
    traits: list[str] = []
    if any(token in text for token in ('vampire', 'вампир', 'blood', 'кров')):
        traits.extend(['vampire', 'predatory'])
    if any(token in text for token in ('aristocrat', 'noble', 'аристократ')):
        traits.append('aristocratic')
    if any(token in text for token in ('science', 'physics', 'physicist', 'учен', 'физик')):
        traits.extend(['logical', 'analytical'])
    if any(token in text for token in ('kind', 'gentle', 'эмпат', 'добр')):
        traits.append('empathetic')
    return list(dict.fromkeys(traits))


def _merge_examples(existing: list[str], new_examples: list[str]) -> list[str]:
    return _normalize_string_list(existing + new_examples, limit=_memory_limits().persona_example_limit)


def _render_tuple_key(parts: tuple[str, ...]) -> str:
    return ' | '.join(str(part).strip() for part in parts if str(part).strip())


def _build_log_tuples(
    examples: list[str],
    situation_reactions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    limit = _memory_limits().persona_log_tuple_limit
    counter: Counter[tuple[str, ...]] = Counter()
    samples: dict[tuple[str, ...], str] = {}
    for example in list(examples or []):
        clean = str(example or '').strip()
        if not clean:
            continue
        key = ('utterance_pattern', normalize_name(clean))
        counter[key] += 1
        samples.setdefault(key, clean)
    for item in list(situation_reactions or []):
        if not isinstance(item, dict):
            continue
        situation = str(item.get('situation') or '').strip()
        reaction = str(item.get('reaction') or '').strip()
        if not situation or not reaction:
            continue
        key = ('situation_reaction', normalize_name(situation), normalize_name(reaction))
        counter[key] += 1
        samples.setdefault(key, f'{situation} -> {reaction}')
    rows: list[dict[str, Any]] = []
    for key, frequency in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        rows.append(
            {
                'tuple': list(key),
                'frequency': int(frequency),
                'sample': samples.get(key, _render_tuple_key(key)),
            }
        )
    return rows[:limit]


def _merge_log_tuples(existing: list[dict[str, Any]], new_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    limit = _memory_limits().persona_log_tuple_limit
    merged: dict[tuple[str, ...], dict[str, Any]] = {}
    for item in list(existing or []) + list(new_items or []):
        if not isinstance(item, dict):
            continue
        key = tuple(str(part).strip() for part in list(item.get('tuple') or []) if str(part).strip())
        if not key:
            continue
        frequency = max(int(item.get('frequency') or 1), 1)
        payload = merged.setdefault(key, {'tuple': list(key), 'frequency': 0, 'sample': ''})
        payload['frequency'] += frequency
        if not str(payload.get('sample') or '').strip():
            payload['sample'] = str(item.get('sample') or _render_tuple_key(key)).strip()
    ordered = sorted(merged.values(), key=lambda item: (-int(item.get('frequency') or 0), str(item.get('sample') or '')))
    return ordered[:limit]


def _identity_class(entity_type: str) -> str:
    token = str(entity_type or '').strip().upper()
    if token == 'PERSON':
        return 'human'
    if token == 'FICTIONAL_CHARACTER':
        return 'fictional_character'
    if token == 'PROFESSION':
        return 'professional_persona'
    return 'conceptual_persona'


def _sarcasm_profile(traits: list[str], examples: list[str]) -> str:
    trait_text = normalize_name(' '.join(traits))
    example_text = normalize_name(' '.join(examples[:12]))
    if any(marker in trait_text or marker in example_text for marker in ('sarcastic', 'sarcasm', 'ironic', 'ирони', 'сарказ')):
        return 'high'
    if 'witty' in trait_text or 'wit' in trait_text:
        return 'medium'
    if any(marker in trait_text for marker in ('aristocratic', 'analytical', 'logical')):
        return 'low'
    return 'none'


def _default_persona_form(
    name: str,
    *,
    entity_type: str,
    traits: list[str],
    relations: list[dict[str, Any]],
    examples: list[str],
    log_tuples: list[dict[str, Any]],
    existing_form: dict[str, Any] | None = None,
) -> dict[str, Any]:
    relation_targets = _normalize_string_list([str(item.get('target') or '') for item in list(relations or [])], limit=8)
    interaction_style = []
    decision_patterns = []
    risk_controls = ['do_not_mirror_user_emotion', 'stay_grounded_in_context']
    if any('logical' in normalize_name(item) or 'analytical' in normalize_name(item) for item in traits):
        interaction_style.append('analytical')
        decision_patterns.append('checks internal consistency before answering')
    if any('empathetic' in normalize_name(item) or 'warm' in normalize_name(item) for item in traits):
        interaction_style.append('supportive')
        decision_patterns.append('accounts for user distress before final wording')
    if any('aggressive' in normalize_name(item) or 'predatory' in normalize_name(item) for item in traits):
        interaction_style.append('firm')
        decision_patterns.append('sets boundaries quickly under insult or hostile pressure')
    if any('aristocratic' in normalize_name(item) for item in traits):
        interaction_style.append('formal')
    if not interaction_style:
        interaction_style.append('direct')
    if not decision_patterns:
        decision_patterns.append('matches reply style to the current situation and available evidence')
    form = {
        'identity_class': _identity_class(entity_type),
        'interaction_style': _normalize_string_list((existing_form or {}).get('interaction_style') or interaction_style, limit=8),
        'core_dispositions': _normalize_string_list((existing_form or {}).get('core_dispositions') or traits, limit=12),
        'decision_patterns': _normalize_string_list((existing_form or {}).get('decision_patterns') or decision_patterns, limit=8),
        'clarification_policy': str((existing_form or {}).get('clarification_policy') or 'Ask a clarifying question when the target, intent, or grounding is insufficient.').strip(),
        'sarcasm_profile': str((existing_form or {}).get('sarcasm_profile') or _sarcasm_profile(traits, examples)).strip() or 'none',
        'response_priorities': _normalize_string_list(
            (existing_form or {}).get('response_priorities') or ['answer_substance', 'clarify_if_underspecified', 'stay_in_character'],
            limit=8,
        ),
        'knowledge_domains': _normalize_string_list((existing_form or {}).get('knowledge_domains') or relation_targets, limit=10),
        'risk_controls': _normalize_string_list((existing_form or {}).get('risk_controls') or risk_controls, limit=8),
        'log_signature_count': len(log_tuples),
    }
    return form


def _validated_persona_form(value: Any, *, fallback: dict[str, Any]) -> dict[str, Any]:
    raw = dict(value or {}) if isinstance(value, dict) else {}
    result = dict(fallback)
    for key in ('identity_class', 'clarification_policy', 'sarcasm_profile'):
        token = str(raw.get(key) or result.get(key) or '').strip()
        if token:
            result[key] = token
    for key, limit in (
        ('interaction_style', 8),
        ('core_dispositions', 12),
        ('decision_patterns', 8),
        ('response_priorities', 8),
        ('knowledge_domains', 10),
        ('risk_controls', 8),
    ):
        result[key] = _normalize_string_list(list(raw.get(key) or result.get(key) or []), limit=limit)
    result['log_signature_count'] = int(raw.get('log_signature_count') or result.get('log_signature_count') or 0)
    return result


def _default_decision_explanation(name: str, persona_form: dict[str, Any]) -> str:
    identity = str(persona_form.get('identity_class') or 'persona').replace('_', ' ')
    patterns = list(persona_form.get('decision_patterns') or [])
    priorities = list(persona_form.get('response_priorities') or [])
    sarcasm = str(persona_form.get('sarcasm_profile') or 'none')
    clarification = str(persona_form.get('clarification_policy') or '').strip()
    pattern_text = patterns[0] if patterns else 'checks the current situation before choosing a reply'
    priority_text = priorities[0] if priorities else 'answer_substance'
    sarcasm_text = 'can use sarcasm if the situation allows it' if sarcasm in {'medium', 'high'} else 'does not rely on sarcasm as a primary strategy'
    parts = [
        f'{name} is treated as a {identity}.',
        f'It first {pattern_text}.',
        f'Then it prioritizes {priority_text.replace("_", " ")}.',
        sarcasm_text + '.',
    ]
    if clarification:
        parts.append(clarification)
    return ' '.join(part.strip() for part in parts if part.strip())


def _validated_decision_explanation(value: Any, *, fallback: str) -> str:
    text = str(value or '').strip()
    return text[:600] if text else fallback[:600]


def _validated_persona_payload(
    name: str,
    payload: Any,
    *,
    fallback_examples: list[str] | None = None,
    explicit: bool = False,
) -> dict[str, Any]:
    limits = _memory_limits()
    container = payload if isinstance(payload, dict) else {}
    raw = dict(container.get('persona_payload') or container) if isinstance(container, dict) else {}
    entity_type = str(raw.get('entity_type') or raw.get('type') or 'PERSON').strip().upper()
    entity_type = _normalized_head_entity_type(entity_type, explicit=explicit)
    raw_traits = [str(item).strip() for item in list(raw.get('traits') or []) if str(item).strip()]
    traits = _normalize_string_list(raw_traits, limit=limits.persona_trait_limit)
    aliases = merge_aliases(list(raw.get('aliases') or []))
    raw_example_rows = [str(item).strip() for item in list(fallback_examples or []) if str(item).strip()] + [
        str(item).strip() for item in list(raw.get('examples') or []) if str(item).strip()
    ]
    deduped_raw_examples = list(dict.fromkeys(raw_example_rows))
    examples = _normalize_string_list(deduped_raw_examples, limit=limits.persona_example_limit)
    example_overflow = deduped_raw_examples[limits.persona_example_limit :]
    raw_relations = [dict(item) for item in list(raw.get('relations') or []) if isinstance(item, dict)]
    relations = _validate_relations(raw_relations)
    relation_overflow = raw_relations[limits.persona_relation_limit :]
    raw_knowledge = str(raw.get('knowledge') or '').strip()
    knowledge = raw_knowledge[: limits.persona_knowledge_char_limit]
    if not knowledge and examples:
        knowledge = '\n'.join(f'- {item}' for item in examples[:8])
    raw_reactions = [dict(item) for item in list(raw.get('situation_reactions') or []) if isinstance(item, dict)]
    raw_log_tuples = [dict(item) for item in list(container.get('log_tuples') or raw.get('log_tuples') or []) if isinstance(item, dict)]
    overflow = {
        'traits': raw_traits[limits.persona_trait_limit :],
        'relations': relation_overflow,
        'examples': example_overflow,
        'situation_reactions': raw_reactions[limits.persona_reaction_limit :],
        'log_tuples': raw_log_tuples[limits.persona_log_tuple_limit :],
        'knowledge_overflow': raw_knowledge[limits.persona_knowledge_char_limit :],
    }
    return {
        'name': str(raw.get('name') or name).strip() or name,
        'entity_type': entity_type,
        'traits': traits or _detect_traits(examples),
        'learned_traits': _normalize_string_list(list(raw.get('learned_traits') or []), limit=limits.persona_trait_limit),
        'aliases': aliases,
        'examples': examples,
        'relations': relations,
        'situation_reactions': raw_reactions[: limits.persona_reaction_limit],
        'emotion_vector': _normalized_emotion_vector(raw.get('emotion_vector') or raw),
        'knowledge': knowledge,
        'log_tuples': raw_log_tuples[: limits.persona_log_tuple_limit],
        'persona_form': dict(container.get('persona_form') or raw.get('persona_form') or {})
        if isinstance(container.get('persona_form') or raw.get('persona_form') or {}, dict)
        else {},
        'decision_explanation': str(container.get('decision_explanation') or raw.get('decision_explanation') or '').strip()[:600],
        'archival_overflow': {
            key: value
            for key, value in overflow.items()
            if value not in (None, '', [], {})
        },
    }


def synthesize_persona_from_logs(
    name: str,
    excerpts: list[str],
    *,
    existing_bundle: HeadBundle | None = None,
    reason: str = '',
    explicit: bool = True,
    relations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cleaned_excerpts = [str(item).strip() for item in list(excerpts or []) if str(item).strip()]
    log_tuples = _build_log_tuples(cleaned_excerpts, list(existing_bundle.situation_reactions) if existing_bundle else [])
    current_form = dict(existing_bundle.persona_form) if existing_bundle is not None else {}
    current_summary = existing_bundle.decision_explanation if existing_bundle is not None else ''
    raw_payload = call_json_model_for_role(
        build_persona_profile_prompt(
            name,
            cleaned_excerpts,
            reason=reason,
            log_tuples=log_tuples,
            current_form=current_form,
            current_summary=current_summary,
        ),
        role=get_runtime_config().roles.persona_synthesis,
    )
    validated = _validated_persona_payload(
        name,
        raw_payload,
        fallback_examples=cleaned_excerpts,
        explicit=explicit,
    )
    existing_log_tuples = list(existing_bundle.log_tuples) if existing_bundle is not None else []
    validated['log_tuples'] = _merge_log_tuples(existing_log_tuples, list(validated.get('log_tuples') or []) + log_tuples)
    merged_relations = _validate_relations(
        list(existing_bundle.relations if existing_bundle is not None else [])
        + list(validated.get('relations') or [])
        + list(relations or [])
    )
    validated['relations'] = merged_relations
    fallback_form = _default_persona_form(
        name,
        entity_type=str(validated.get('entity_type') or (existing_bundle.entity_type if existing_bundle else 'PERSON')),
        traits=list(validated.get('traits') or (existing_bundle.traits if existing_bundle else [])),
        relations=merged_relations,
        examples=list(validated.get('examples') or cleaned_excerpts),
        log_tuples=list(validated.get('log_tuples') or []),
        existing_form=current_form,
    )
    validated['persona_form'] = _validated_persona_form(validated.get('persona_form'), fallback=fallback_form)
    validated['decision_explanation'] = _validated_decision_explanation(
        validated.get('decision_explanation'),
        fallback=_default_decision_explanation(name, validated['persona_form']),
    )
    return validated


def spawn_head(
    name: str,
    *,
    entity_type: str = 'CONCEPT',
    aliases: list[str] | None = None,
    source: str = 'system',
    sync_graph: bool = True,
) -> HeadBundle:
    clean = normalize_personality_name(name)
    _ensure_head_files(clean, entity_type=entity_type, aliases=aliases, source=source)
    bundle = load_persona(clean)
    assert bundle is not None
    meta = dict(bundle.meta)
    meta['name'] = name
    meta['entity_type'] = entity_type
    meta['aliases'] = merge_aliases(list(meta.get('aliases') or []), list(aliases or []))
    meta['frequency'] = max(int(meta.get('frequency') or 1), 1)
    meta['updated_at'] = _utc_now()
    write_json(_head_file(clean, 'meta.json'), meta)
    _save_index(_load_index() + [clean])
    if sync_graph:
        GraphStore().register_head(
            name=name,
            folder=str(_head_dir(clean)),
            entity_type=entity_type,
            aliases=list(meta.get('aliases') or []),
        )
    _sync_local_graph(clean)
    updated = load_persona(clean)
    assert updated is not None
    return updated


def request_persona_profile(name: str, reason: str, session_id: str, excerpt: str) -> dict[str, Any]:
    payload = {
        'name': normalize_personality_name(name),
        'reason': str(reason or '').strip() or 'Missing head referenced in dialogue.',
        'session_id': str(session_id or '').strip(),
        'excerpt': str(excerpt or '').strip(),
        'created_at': _utc_now(),
    }
    write_json(personality_proposal_path(name), payload)
    spawn_head(name, entity_type='PERSON', source='proposal')
    return payload


def list_persona_proposals() -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for path in sorted(personality_proposals_dir().glob('*.json')):
        payload = load_json(path, None)
        if isinstance(payload, dict):
            proposals.append(payload)
    return proposals


def list_persona_revisions(name: str) -> list[dict[str, Any]]:
    if not str(name or '').strip():
        return []
    return [dict(item) for item in list(_load_revision_payload(normalize_personality_name(name)).get('history') or [])]


def restore_persona_revision(name: str, revision: int) -> HeadBundle | None:
    clean = normalize_personality_name(name)
    if not clean:
        return None
    bundle = load_persona(clean) or spawn_head(clean, entity_type='PERSON')
    history = list_persona_revisions(clean)
    target = next((item for item in history if int(item.get('revision') or 0) == int(revision or 0)), None)
    if not isinstance(target, dict):
        return None
    layer_snapshots = dict(target.get('layers') or {})

    def _restore() -> HeadBundle | None:
        baseline = bundle.baseline_definition
        dynamic = bundle.dynamic_state
        learned = bundle.learned_patterns
        if baseline is not None and isinstance(layer_snapshots.get('baseline'), dict):
            snapshot = dict(layer_snapshots['baseline'].get('snapshot') or {})
            baseline = PersonaBaselineDefinition(
                name=str(snapshot.get('name') or baseline.name),
                slug=str(snapshot.get('slug') or baseline.slug),
                entity_type=str(snapshot.get('entity_type') or baseline.entity_type),
                traits=_normalize_string_list(list(snapshot.get('traits') or baseline.traits), limit=_memory_limits().persona_trait_limit),
                aliases=merge_aliases(list(snapshot.get('aliases') or baseline.aliases)),
                relations=_validate_relations(list(snapshot.get('relations') or baseline.relations)),
                knowledge=str(snapshot.get('knowledge') or baseline.knowledge).strip()[: _memory_limits().persona_knowledge_char_limit],
                revision=int(snapshot.get('revision') or baseline.revision or 1),
                updated_at=str(snapshot.get('updated_at') or baseline.updated_at),
                source=str(snapshot.get('source') or baseline.source or 'revision_restore'),
            )
            write_json(_head_file(clean, 'traits.json'), {'traits': list(baseline.traits), 'entity_type': baseline.entity_type})
            write_json(_head_file(clean, 'relations.json'), {'aliases': list(baseline.aliases), 'relations': list(baseline.relations)})
            write_text(_head_file(clean, 'knowledge.txt'), baseline.knowledge)
        if dynamic is not None and isinstance(layer_snapshots.get('dynamic'), dict):
            snapshot = dict(layer_snapshots['dynamic'].get('snapshot') or {})
            dynamic = PersonaDynamicState(
                emotion_vector=_normalized_emotion_vector(snapshot.get('emotion_vector') or dynamic.emotion_vector),
                last_situation=str(snapshot.get('last_situation') or dynamic.last_situation).strip(),
                last_response_style=str(snapshot.get('last_response_style') or dynamic.last_response_style).strip(),
                revision=int(snapshot.get('revision') or dynamic.revision or 1),
                updated_at=str(snapshot.get('updated_at') or dynamic.updated_at),
            )
            write_json(_head_file(clean, 'emotion_vector.json'), dict(dynamic.emotion_vector))
        if learned is not None and isinstance(layer_snapshots.get('learned'), dict):
            snapshot = dict(layer_snapshots['learned'].get('snapshot') or {})
            examples_source = snapshot['examples'] if 'examples' in snapshot else learned.examples
            reactions_source = snapshot['situation_reactions'] if 'situation_reactions' in snapshot else learned.situation_reactions
            log_source = snapshot['log_tuples'] if 'log_tuples' in snapshot else learned.log_tuples
            form_source = snapshot['persona_form'] if 'persona_form' in snapshot else learned.persona_form
            learned_traits_source = snapshot['learned_traits'] if 'learned_traits' in snapshot else learned.learned_traits
            learned = PersonaLearnedPatterns(
                examples=_normalize_string_list(list(examples_source or []), limit=_memory_limits().persona_example_limit),
                situation_reactions=[dict(item) for item in list(reactions_source or []) if isinstance(item, dict)],
                log_tuples=[dict(item) for item in list(log_source or []) if isinstance(item, dict)],
                persona_form=dict(form_source or {}) if isinstance(form_source or {}, dict) else {},
                decision_explanation=str(snapshot.get('decision_explanation') or learned.decision_explanation).strip()[:600],
                learned_traits=_normalize_string_list(list(learned_traits_source or []), limit=_memory_limits().persona_trait_limit),
                revision=int(snapshot.get('revision') or learned.revision or 1),
                updated_at=str(snapshot.get('updated_at') or learned.updated_at),
            )
            write_json(
                _head_file(clean, 'examples.json'),
                {
                    'examples': list(learned.examples),
                    'situation_reactions': [dict(item) for item in learned.situation_reactions],
                },
            )
            write_json(_head_file(clean, 'log_tuples.json'), {'items': list(learned.log_tuples)})
            write_json(_head_file(clean, 'persona_form.json'), dict(learned.persona_form))
            write_text(_head_file(clean, 'decision_explanation.txt'), learned.decision_explanation)
        _write_layered_state(clean, baseline=baseline, dynamic=dynamic, learned=learned, reason=f'revision_restore:{revision}')
        if baseline is not None:
            GraphStore().sync_head(
                name=baseline.name,
                folder=str(_head_dir(clean)),
                entity_type=baseline.entity_type,
                aliases=list(baseline.aliases),
                description=(baseline.knowledge.splitlines()[0] if baseline.knowledge else f'Persona head for {baseline.name}.'),
                facts=list((learned.examples if learned is not None else []))[:8],
                knowledge=baseline.knowledge,
                relations=list(baseline.relations),
            )
        _sync_local_graph(clean)
        _sync_meta_summary(clean)
        return load_persona(clean)

    return _execute_persona_mutation(
        clean,
        reason=f'restore_persona_revision:{revision}',
        archive_snapshot=True,
        graph_snapshot=True,
        operation=_restore,
    )


def infer_persona_name(message: str, *, selected_name: str = '', current_entity: str = '') -> str:
    if str(selected_name or '').strip():
        return normalize_personality_name(selected_name)
    lowered = normalize_name(message)
    for item in list_personas():
        names = [str(item.get('name') or ''), str(item.get('slug') or '')]
        if any(normalize_name(name) and normalize_name(name) in lowered for name in names):
            return str(item.get('slug') or '')
    return normalize_personality_name(current_entity) if str(current_entity or '').strip() else ''


DIRECT_USER_EMOTION_INHERITANCE_FORBIDDEN = True


def _trait_parameters(bundle: HeadBundle) -> dict[str, float]:
    profile = {
        'aggression': 0.3,
        'empathy': 0.45,
        'curiosity': 0.5,
        'confidence': 0.5,
        'restraint': 0.45,
        'fear_sensitivity': 0.3,
    }
    adjustments = {
        'aggressive': {'aggression': 0.35, 'restraint': -0.1},
        'defensive': {'aggression': 0.18},
        'predatory': {'aggression': 0.2, 'empathy': -0.08},
        'empathetic': {'empathy': 0.35, 'aggression': -0.08},
        'warm': {'empathy': 0.22},
        'logical': {'restraint': 0.28, 'curiosity': 0.08},
        'analytical': {'restraint': 0.22, 'curiosity': 0.14},
        'aristocratic': {'confidence': 0.18, 'restraint': 0.08},
        'precise': {'restraint': 0.18},
        'curious': {'curiosity': 0.2},
        'brave': {'confidence': 0.16, 'fear_sensitivity': -0.1},
        'cautious': {'fear_sensitivity': 0.2, 'restraint': 0.08},
    }
    for trait in bundle.traits:
        token = normalize_name(trait)
        for key, delta in adjustments.items():
            if key in token:
                for field, value in delta.items():
                    profile[field] = profile[field] + value
    return {key: round(min(1.0, max(0.0, value)), 4) for key, value in profile.items()}


def _drift_toward_baseline(vector: dict[str, float]) -> dict[str, float]:
    baseline = _default_emotion_vector()
    drifted: dict[str, float] = {}
    for key in EMOTION_KEYS:
        current = float(vector.get(key, baseline[key]))
        drifted[key] = current + 0.04 * (baseline[key] - current)
    return drifted


def formalize_persona(bundle: HeadBundle) -> PersonaSystemModel:
    return PersonaSystemModel(
        T={
            'entity_type': bundle.entity_type,
            'traits': list(bundle.traits),
            'parameters': _trait_parameters(bundle),
        },
        E=_normalized_emotion_vector(bundle.emotion_vector),
        R='deterministic_situation_reaction_policy',
        M={
            'examples': list(bundle.examples),
            'relations': [dict(item) for item in bundle.relations],
            'situation_reactions': [dict(item) for item in bundle.situation_reactions],
            'knowledge': bundle.knowledge,
            'folder': bundle.folder,
            'log_tuples': [dict(item) for item in bundle.log_tuples],
            'persona_form': dict(bundle.persona_form),
            'decision_explanation': bundle.decision_explanation,
            'baseline_definition': bundle.baseline_definition.to_dict() if bundle.baseline_definition is not None else {},
            'learned_patterns': bundle.learned_patterns.to_dict() if bundle.learned_patterns is not None else {},
            'indicators': bundle.indicators.to_dict() if bundle.indicators is not None else {},
            'revision_meta': {
                key: value
                for key, value in dict(bundle.revision_meta or {}).items()
                if key != 'history'
            },
        },
    )


def _response_style(bundle: HeadBundle, *, situation_type: str, target: str, severity: float, emotion_state: dict[str, float]) -> str:
    traits = _trait_parameters(bundle)
    if situation_type == 'insult' and target == 'persona':
        return 'firm_boundary' if traits['restraint'] >= 0.5 else 'defensive'
    if situation_type == 'user_distress':
        return 'supportive' if traits['empathy'] >= 0.5 else 'measured_support'
    if situation_type == 'abnormal_behavior':
        return 'corrective' if traits['restraint'] >= 0.45 else 'cold_disapproval'
    if situation_type == 'user_anger':
        return 'de_escalating' if traits['empathy'] >= traits['aggression'] else 'assertive'
    if situation_type == 'neutral_query':
        return 'inquisitive' if emotion_state.get('curiosity', 0.0) >= 0.45 else 'formal'
    return 'steady'


def reaction_policy(bundle: HeadBundle, situation: Situation | dict[str, Any] | None, *, emotion_state: dict[str, float] | None = None) -> ReactionOutcome:
    state = _normalized_emotion_vector(emotion_state or bundle.emotion_vector)
    profile = _trait_parameters(bundle)
    if isinstance(situation, Situation):
        situation_type = str(situation.type or 'neutral_statement').strip() or 'neutral_statement'
        target = str(situation.target or 'external').strip() or 'external'
        severity = min(1.0, max(0.0, float(situation.severity or 0.0)))
    else:
        situation_map = dict(situation or {})
        situation_type = str(situation_map.get('type') or 'neutral_statement').strip() or 'neutral_statement'
        target = str(situation_map.get('target') or 'external').strip() or 'external'
        severity = min(1.0, max(0.0, float(situation_map.get('severity') or 0.0)))

    delta = {key: 0.0 for key in EMOTION_KEYS}
    if DIRECT_USER_EMOTION_INHERITANCE_FORBIDDEN:
        if situation_type == 'insult' and target == 'persona':
            delta['anger'] += severity * (0.08 + 0.28 * profile['aggression'])
            delta['confidence'] -= severity * (0.02 + 0.04 * (1.0 - profile['confidence']))
            delta['curiosity'] -= severity * 0.03
        elif situation_type == 'user_distress':
            delta['empathy'] += severity * (0.08 + 0.26 * profile['empathy'])
            delta['anger'] -= severity * 0.05
            delta['fear'] += severity * (0.01 + 0.05 * profile['fear_sensitivity'])
        elif situation_type == 'abnormal_behavior':
            delta['anger'] += severity * (0.03 + 0.16 * profile['aggression'])
            delta['empathy'] += severity * (0.02 + 0.08 * profile['empathy'])
            delta['confidence'] -= severity * 0.03
            delta['curiosity'] -= severity * 0.04
        elif situation_type == 'user_anger':
            delta['anger'] += severity * (0.02 + 0.1 * profile['aggression'])
            delta['empathy'] += severity * (0.01 + 0.07 * profile['empathy'])
            delta['confidence'] += severity * (0.01 + 0.03 * profile['confidence'])
        elif situation_type == 'neutral_query':
            delta['curiosity'] += severity * (0.05 + 0.18 * profile['curiosity'])
            delta['confidence'] += severity * (0.01 + 0.04 * profile['confidence'])
            delta['fear'] -= severity * 0.02
        else:
            delta['curiosity'] += severity * 0.02 * profile['curiosity']

    return ReactionOutcome(
        delta_emotion={key: round(value, 4) for key, value in delta.items()},
        response_style=_response_style(bundle, situation_type=situation_type, target=target, severity=severity, emotion_state=state),
        situation_type=situation_type,
        target=target,
        severity=round(severity, 4),
    )


def evolve_emotion_state(bundle: HeadBundle, situation: Situation | dict[str, Any] | None) -> tuple[dict[str, float], ReactionOutcome]:
    current_state = _normalized_emotion_vector(bundle.emotion_vector)
    decayed = _drift_toward_baseline(current_state)
    outcome = reaction_policy(bundle, situation, emotion_state=current_state)
    next_state = {
        key: round(min(1.0, max(0.0, float(decayed.get(key, 0.0)) + float(outcome.delta_emotion.get(key, 0.0)))), 4)
        for key in EMOTION_KEYS
    }
    return next_state, outcome


def _sync_meta_summary(name: str) -> dict[str, Any]:
    bundle = load_persona(name)
    if bundle is None:
        return {}
    meta = dict(bundle.meta)
    meta['schema_version'] = 2
    meta['confidence_score'] = bundle.indicators.confidence_score if bundle.indicators is not None else 0.0
    meta['maturity_score'] = bundle.indicators.maturity_score if bundle.indicators is not None else 0.0
    meta['maturity_level'] = bundle.indicators.maturity_level if bundle.indicators is not None else 'bootstrap'
    meta['evidence_count'] = bundle.indicators.evidence_count if bundle.indicators is not None else 0
    meta['learned_pattern_count'] = bundle.indicators.learned_pattern_count if bundle.indicators is not None else 0
    meta['adaptation_locked'] = bundle.indicators.adaptation_locked if bundle.indicators is not None else False
    for key in (
        'revision',
        'baseline_revision',
        'dynamic_revision',
        'learned_revision',
        'last_baseline_update_at',
        'last_dynamic_update_at',
        'last_learned_update_at',
    ):
        if key in bundle.revision_meta:
            meta[key] = bundle.revision_meta[key]
    write_json(_head_file(name, 'meta.json'), meta)
    return meta


def _update_meta(name: str, *, entity_type: str | None = None, aliases: list[str] | None = None, importance_delta: float = 0.0) -> dict[str, Any]:
    bundle = load_persona(name) or spawn_head(name, entity_type=entity_type or 'CONCEPT')
    meta = dict(bundle.meta)
    if entity_type:
        meta['entity_type'] = entity_type
    meta['aliases'] = merge_aliases(list(meta.get('aliases') or []), list(aliases or []))
    meta['frequency'] = int(meta.get('frequency') or 1) + 1
    meta['importance'] = round(min(1.0, max(0.05, float(meta.get('importance') or 0.5) + importance_delta)), 4)
    meta['updated_at'] = _utc_now()
    meta['schema_version'] = 2
    write_json(_head_file(name, 'meta.json'), meta)
    return meta


def explain_response_style(
    bundle: HeadBundle,
    situation: Situation | dict[str, Any] | None,
    *,
    outcome: ReactionOutcome | None = None,
) -> PersonaResponseExplanation:
    actual = outcome or reaction_policy(bundle, situation, emotion_state=bundle.emotion_vector)
    state_source = bundle.dynamic_state.emotion_vector if bundle.dynamic_state is not None else bundle.emotion_vector
    sorted_state = sorted(
        ((key, float(value)) for key, value in dict(state_source or {}).items()),
        key=lambda item: (-item[1], item[0]),
    )
    state_influences = [f'{key}={round(value, 3)}' for key, value in sorted_state[:3] if value >= 0.15]
    trait_influences: list[str] = []
    if actual.response_style in {'firm_boundary', 'defensive', 'assertive'}:
        trait_influences.extend([trait for trait in bundle.traits if normalize_name(trait) in {'aggressive', 'predatory', 'aristocratic', 'logical'}][:3])
    elif actual.response_style in {'supportive', 'measured_support', 'de_escalating'}:
        trait_influences.extend([trait for trait in bundle.traits if normalize_name(trait) in {'empathetic', 'warm', 'logical'}][:3])
    elif actual.response_style in {'inquisitive', 'formal', 'steady'}:
        trait_influences.extend([trait for trait in bundle.traits if normalize_name(trait) in {'logical', 'analytical', 'curious', 'aristocratic'}][:3])
    learned_influences: list[str] = []
    if bundle.persona_form:
        sarcasm = str(bundle.persona_form.get('sarcasm_profile') or '').strip()
        if sarcasm:
            learned_influences.append(f'sarcasm_profile={sarcasm}')
        for pattern in list(bundle.persona_form.get('decision_patterns') or [])[:2]:
            clean = str(pattern).strip()
            if clean:
                learned_influences.append(clean)
    for item in relevant_reactions(bundle.name, situation, limit=2):
        rendered = str(item.get('reaction') or '').strip()
        if rendered:
            learned_influences.append(rendered)
    reason = (
        f"Response style '{actual.response_style}' was selected because situation "
        f"'{actual.situation_type}' targets '{actual.target}' with severity {actual.severity:.2f}, "
        'then the persona reaction policy applied trait-conditioned state shaping.'
    )
    return PersonaResponseExplanation(
        persona_name=bundle.name,
        response_style=actual.response_style,
        reason=reason,
        situation_summary=situation_summary(situation),
        state_influences=list(dict.fromkeys(state_influences))[:4],
        trait_influences=list(dict.fromkeys(trait_influences))[:4],
        learned_influences=list(dict.fromkeys(learned_influences))[:4],
    )


def adjust_emotion_vector(name: str, situation: Situation | dict[str, Any] | None) -> dict[str, float]:
    clean = normalize_personality_name(name)

    def _apply() -> dict[str, float]:
        bundle = load_persona(clean) or spawn_head(clean, entity_type='PERSON')
        next_state, outcome = evolve_emotion_state(bundle, situation)
        write_json(_head_file(clean, 'emotion_vector.json'), next_state)
        dynamic = PersonaDynamicState(
            emotion_vector=next_state,
            last_situation=situation_summary(situation),
            last_response_style=outcome.response_style,
            revision=int((bundle.dynamic_state.revision if bundle.dynamic_state is not None else bundle.meta.get('dynamic_revision') or 1)),
            updated_at=_utc_now(),
        )
        _write_layered_state(clean, dynamic=dynamic, reason='emotion_update')
        _sync_meta_summary(clean)
        return next_state

    return _execute_persona_mutation(
        clean,
        reason='emotion_update',
        archive_snapshot=False,
        graph_snapshot=False,
        operation=_apply,
    )


def record_situation_reaction(name: str, situation: str | Situation | dict[str, Any], reaction: str | int) -> None:
    if not str(name or '').strip():
        return
    clean = normalize_personality_name(name)

    def _apply() -> None:
        bundle = load_persona(clean) or spawn_head(clean, entity_type='PERSON')
        payload = load_json(_head_file(clean, 'examples.json'), {'examples': [], 'situation_reactions': []})
        situation_reactions = [dict(item) for item in list(payload.get('situation_reactions') or []) if isinstance(item, dict)]
        entry = {'situation': situation_summary(situation), 'reaction': reaction if reaction == 0 else str(reaction or '').strip()}
        if not entry['situation']:
            return
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in [entry] + situation_reactions:
            key = normalize_name(str(item.get('situation') or ''))
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        payload['situation_reactions'] = deduped[: _memory_limits().persona_reaction_limit]
        payload['examples'] = list(payload.get('examples') or bundle.examples)
        write_json(_head_file(clean, 'examples.json'), payload)
        learned = PersonaLearnedPatterns(
            examples=list(bundle.examples),
            situation_reactions=[dict(item) for item in payload['situation_reactions']],
            log_tuples=_merge_log_tuples(
                list(bundle.log_tuples),
                _build_log_tuples(list(bundle.examples), list(payload['situation_reactions'])),
            ),
            persona_form=dict(bundle.persona_form),
            decision_explanation=bundle.decision_explanation,
            learned_traits=list(bundle.learned_patterns.learned_traits) if bundle.learned_patterns is not None else [],
            revision=int((bundle.learned_patterns.revision if bundle.learned_patterns is not None else bundle.meta.get('learned_revision') or 1)),
            updated_at=_utc_now(),
        )
        write_json(_head_file(clean, 'log_tuples.json'), {'items': learned.log_tuples})
        _write_layered_state(clean, learned=learned, reason='record_situation_reaction')
        _sync_meta_summary(clean)

    _execute_persona_mutation(
        clean,
        reason='record_situation_reaction',
        archive_snapshot=False,
        graph_snapshot=False,
        operation=_apply,
    )


def build_persona_graph(profile: HeadBundle | dict[str, Any]) -> dict[str, Any]:
    if isinstance(profile, HeadBundle):
        name = profile.name
        entity_type = profile.entity_type
        traits = profile.traits
        relations = profile.relations
        examples = profile.examples
        emotion_vector = profile.emotion_vector
        folder = profile.folder
    else:
        name = str(profile.get('name') or 'unknown_head')
        entity_type = str(profile.get('entity_type') or 'CONCEPT')
        traits = _normalize_string_list(list(profile.get('traits') or []), limit=_memory_limits().persona_trait_limit)
        relations = _validate_relations(list(profile.get('relations') or []))
        examples = _normalize_string_list(list(profile.get('examples') or []), limit=_memory_limits().persona_example_limit)
        emotion_vector = _normalized_emotion_vector(profile.get('emotion_vector') or profile)
        folder = str(profile.get('folder') or _head_dir(name))
    slug = normalize_personality_name(name)
    root_id = f'head:{slug}'
    nodes = [
        {
            'id': root_id,
            'name': name,
            'type': entity_type,
            'folder': folder,
            'importance': 0.8,
            'confidence': 0.85,
            'frequency': max(len(examples), 1),
            'emotion_vector': emotion_vector,
            'facts': examples[:6],
        }
    ]
    edges: list[dict[str, Any]] = []
    for trait in traits:
        trait_id = f'trait:{slug}:{normalize_personality_name(trait)}'
        nodes.append({'id': trait_id, 'name': trait, 'type': 'CONCEPT', 'importance': 0.4, 'confidence': 0.8, 'frequency': 1})
        edges.append({'from': root_id, 'to': trait_id, 'type': 'HAS_TRAIT', 'weight': 0.9, 'confidence': 0.85})
    for relation in relations:
        target = str(relation.get('target') or relation.get('to') or '').strip()
        relation_type = str(relation.get('type') or 'RELATED_TO').strip().upper()
        if not target:
            continue
        target_id = f'concept:{normalize_personality_name(target)}'
        nodes.append({'id': target_id, 'name': target, 'type': 'CONCEPT', 'importance': 0.3, 'confidence': 0.7, 'frequency': 1})
        edges.append({'from': root_id, 'to': target_id, 'type': relation_type, 'weight': 0.8, 'confidence': 0.8})
    for example in examples[:6]:
        example_id = f'example:{slug}:{normalize_personality_name(example)[:32]}'
        nodes.append({'id': example_id, 'name': example, 'type': 'CONCEPT', 'importance': 0.2, 'confidence': 0.6, 'frequency': 1})
        edges.append({'from': root_id, 'to': example_id, 'type': 'HAS_EXAMPLE', 'weight': 0.5, 'confidence': 0.6})
    return {'nodes': nodes, 'edges': edges}


def _sync_local_graph(name: str) -> None:
    bundle = load_persona(name)
    if bundle is None:
        return
    write_json(personality_graph_path(name), build_persona_graph(bundle))


def materialize_persona(
    name: str,
    payload: dict[str, Any],
    *,
    explicit: bool = False,
    adaptation_mode: str = 'baseline_refresh',
) -> HeadBundle:
    normalized = _validated_persona_payload(name, payload, fallback_examples=list(payload.get('examples') or []), explicit=explicit)
    clean = normalize_personality_name(normalized['name'])

    def _apply() -> HeadBundle:
        existing_bundle = load_persona(clean)
        overflow = dict(normalized.get('archival_overflow') or {})
        if overflow:
            append_persona_overflow_archive(clean, overflow, reason='persona_active_bounds')
        bundle = spawn_head(
            normalized['name'],
            entity_type=normalized['entity_type'],
            aliases=list(normalized.get('aliases') or []),
            source='materialize',
            sync_graph=False,
        )
        use_learned_update = adaptation_mode == 'learned_update' and existing_bundle is not None
        baseline_traits = (
            list(existing_bundle.baseline_definition.traits)
            if use_learned_update and existing_bundle.baseline_definition is not None
            else list(normalized['traits'])
        )
        baseline_relations = (
            [dict(item) for item in existing_bundle.baseline_definition.relations]
            if use_learned_update and existing_bundle.baseline_definition is not None
            else list(normalized['relations'])
        )
        baseline_aliases = (
            list(existing_bundle.baseline_definition.aliases)
            if use_learned_update and existing_bundle.baseline_definition is not None
            else merge_aliases(list(normalized.get('aliases') or []), list(bundle.meta.get('aliases') or []))
        )
        baseline_entity_type = (
            str(existing_bundle.baseline_definition.entity_type)
            if use_learned_update and existing_bundle.baseline_definition is not None
            else str(normalized['entity_type'])
        )
        baseline_knowledge = (
            str(existing_bundle.baseline_definition.knowledge)
            if use_learned_update and existing_bundle.baseline_definition is not None
            else str(normalized['knowledge'])
        )
        learned_traits = _normalize_string_list(
            (
                list(existing_bundle.learned_patterns.learned_traits)
                if existing_bundle and existing_bundle.learned_patterns is not None
                else []
            )
            + list(normalized.get('learned_traits') or [])
            + [trait for trait in list(normalized.get('traits') or []) if trait not in baseline_traits],
            limit=_memory_limits().persona_trait_limit,
        )
        write_json(_head_file(clean, 'traits.json'), {'traits': list(baseline_traits), 'entity_type': baseline_entity_type})
        write_json(_head_file(clean, 'relations.json'), {'aliases': list(baseline_aliases), 'relations': list(baseline_relations)})
        write_json(
            _head_file(clean, 'examples.json'),
            {
                'examples': _merge_examples(bundle.examples, list(normalized['examples'])),
                'situation_reactions': (
                    [dict(item) for item in list(bundle.situation_reactions) if isinstance(item, dict)]
                    if use_learned_update
                    else [dict(item) for item in list(normalized['situation_reactions']) if isinstance(item, dict)]
                ),
            },
        )
        write_json(_head_file(clean, 'emotion_vector.json'), normalized['emotion_vector'])
        write_text(_head_file(clean, 'knowledge.txt'), baseline_knowledge)
        provided_log_tuples = list(normalized.get('log_tuples') or [])
        if provided_log_tuples:
            merged_log_tuples = _merge_log_tuples(list(bundle.log_tuples), provided_log_tuples)
        else:
            merged_log_tuples = _merge_log_tuples(
                list(bundle.log_tuples),
                _build_log_tuples(list(normalized['examples']), list(normalized['situation_reactions'])),
            )
        persona_form = _validated_persona_form(
            normalized.get('persona_form'),
            fallback=_default_persona_form(
                normalized['name'],
                entity_type=baseline_entity_type,
                traits=list(baseline_traits),
                relations=list(baseline_relations),
                examples=list(normalized['examples']),
                log_tuples=merged_log_tuples,
                existing_form=bundle.persona_form,
            ),
        )
        decision_explanation = _validated_decision_explanation(
            normalized.get('decision_explanation'),
            fallback=_default_decision_explanation(normalized['name'], persona_form),
        )
        write_json(_head_file(clean, 'log_tuples.json'), {'items': merged_log_tuples})
        write_json(_head_file(clean, 'persona_form.json'), persona_form)
        write_text(_head_file(clean, 'decision_explanation.txt'), decision_explanation)
        baseline = PersonaBaselineDefinition(
            name=normalized['name'],
            slug=clean,
            entity_type=baseline_entity_type,
            traits=list(baseline_traits),
            aliases=list(baseline_aliases),
            relations=[dict(item) for item in baseline_relations],
            knowledge=baseline_knowledge,
            revision=int((bundle.baseline_definition.revision if bundle.baseline_definition is not None else bundle.meta.get('baseline_revision') or 1)),
            updated_at=_utc_now(),
            source='materialize' if not use_learned_update else 'baseline_locked',
        )
        dynamic = PersonaDynamicState(
            emotion_vector=_normalized_emotion_vector(normalized['emotion_vector']),
            last_situation=str(bundle.dynamic_state.last_situation) if bundle.dynamic_state is not None else '',
            last_response_style=str(bundle.dynamic_state.last_response_style) if bundle.dynamic_state is not None else '',
            revision=int((bundle.dynamic_state.revision if bundle.dynamic_state is not None else bundle.meta.get('dynamic_revision') or 1)),
            updated_at=_utc_now(),
        )
        learned = PersonaLearnedPatterns(
            examples=_merge_examples(bundle.examples, list(normalized['examples'])),
            situation_reactions=(
                [dict(item) for item in list(bundle.situation_reactions) if isinstance(item, dict)]
                if use_learned_update
                else [dict(item) for item in list(normalized['situation_reactions']) if isinstance(item, dict)]
            ),
            log_tuples=list(merged_log_tuples),
            persona_form=dict(persona_form),
            decision_explanation=decision_explanation,
            learned_traits=list(learned_traits),
            revision=int((bundle.learned_patterns.revision if bundle.learned_patterns is not None else bundle.meta.get('learned_revision') or 1)),
            updated_at=_utc_now(),
        )
        if use_learned_update:
            _write_layered_state(clean, learned=learned, reason='learned_update')
        else:
            _write_layered_state(clean, baseline=baseline, dynamic=dynamic, learned=learned, reason='materialize_persona')
        meta = _update_meta(
            clean,
            entity_type=baseline_entity_type,
            aliases=list(baseline_aliases),
            importance_delta=0.04 if use_learned_update else 0.08,
        )
        write_json(_head_file(clean, 'meta.json'), meta)

        GraphStore().sync_head(
            name=normalized['name'],
            folder=str(_head_dir(clean)),
            entity_type=baseline_entity_type,
            aliases=list(meta.get('aliases') or []),
            description=(baseline_knowledge.splitlines()[0] if baseline_knowledge else f'Persona head for {normalized["name"]}.'),
            facts=list(learned.examples)[:8],
            knowledge=baseline_knowledge,
            relations=list(baseline_relations),
        )
        _sync_local_graph(clean)
        _sync_meta_summary(clean)
        updated = load_persona(clean)
        assert updated is not None
        return updated

    return _execute_persona_mutation(
        clean,
        reason='materialize_persona' if adaptation_mode != 'learned_update' else 'materialize_persona_learned_update',
        archive_snapshot=True,
        graph_snapshot=True,
        operation=_apply,
    )


def update_persona_from_examples(name: str, examples: list[str], relations: list[dict[str, Any]] | None = None) -> HeadBundle:
    clean = normalize_personality_name(name)
    bundle = load_persona(clean) or spawn_head(name, entity_type='PERSON')
    cleaned_examples = [str(item).strip() for item in list(examples or []) if str(item).strip()]
    synthesized = synthesize_persona_from_logs(
        bundle.name,
        cleaned_examples,
        existing_bundle=bundle,
        reason='Session examples and repeated behavior patterns.',
        explicit=True,
        relations=relations,
    )
    merged_examples = _merge_examples(bundle.examples, list(synthesized.get('examples') or cleaned_examples))
    suggested_traits = list(synthesized.get('traits') or [])
    synthesized['entity_type'] = str(bundle.entity_type or 'PERSON')
    synthesized['traits'] = list(bundle.traits)
    synthesized['learned_traits'] = suggested_traits
    synthesized['relations'] = _validate_relations(list(bundle.relations) + list(relations or []))
    synthesized['examples'] = merged_examples
    synthesized['situation_reactions'] = [dict(item) for item in list(bundle.situation_reactions) if isinstance(item, dict)]
    synthesized['emotion_vector'] = synthesized.get('emotion_vector') or bundle.emotion_vector
    synthesized['knowledge'] = str(bundle.knowledge or '').strip()
    synthesized['aliases'] = list(bundle.meta.get('aliases') or [])
    synthesized['log_tuples'] = _merge_log_tuples(list(bundle.log_tuples), list(synthesized.get('log_tuples') or []))
    return materialize_persona(bundle.name, synthesized, explicit=True, adaptation_mode='learned_update')


def process_persona_proposals() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for proposal in list_persona_proposals():
        name = str(proposal.get('name') or '').strip()
        excerpt = str(proposal.get('excerpt') or '').strip()
        if not name:
            continue
        payload = synthesize_persona_from_logs(
            name,
            [excerpt] if excerpt else [],
            reason=str(proposal.get('reason') or ''),
            explicit=True,
        )
        bundle = materialize_persona(name, payload, explicit=True)
        proposal_path = personality_proposal_path(name)
        if proposal_path.exists():
            proposal_path.unlink()
        results.append({'name': name, 'folder': bundle.folder, 'entity_type': bundle.entity_type})
    return results


def emotion_label(vector: dict[str, float]) -> str:
    dominant = max(vector.items(), key=lambda item: item[1])[0] if vector else 'curiosity'
    return {
        'anger': 'defensive',
        'fear': 'cautious',
        'curiosity': 'inquisitive',
        'confidence': 'assured',
        'empathy': 'warm',
    }.get(dominant, 'balanced')


def relevant_reactions(name: str, situation: str | Situation | dict[str, Any], *, limit: int = 3) -> list[dict[str, Any]]:
    bundle = load_persona(name)
    if bundle is None:
        return []
    target = normalize_name(situation_summary(situation))
    scored: list[tuple[int, dict[str, Any]]] = []
    for item in bundle.situation_reactions:
        candidate = normalize_name(str(item.get('situation') or ''))
        if not candidate:
            continue
        score = sum(1 for token in candidate.split() if token in target)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda row: (-row[0], str(row[1].get('situation') or '')))
    return [item for _, item in scored[:limit]]
