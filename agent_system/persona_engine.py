from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
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
    ChangeDirection,
    EMOTION_KEYS,
    HEAD_ENTITY_TYPES,
    HeadBundle,
    PersonaBaselineDefinition,
    PersonaBehavior,
    PersonaConflict,
    PersonaCore,
    PersonaDefense,
    PersonaDynamicState,
    PersonaDynamics,
    PersonaGraphExplanation,
    PersonaIdentity,
    PersonaIndicators,
    PersonaLearnedPatterns,
    PersonaMeta,
    PersonaResponseExplanation,
    PersonaSystemModel,
    ReactionOutcome,
    Situation,
    StructuredPersona,
)
from .prompt_builder import build_persona_profile_prompt
from .reliability import MutationRejectedFailure, RecoveryFailure, StorageWriteFailure
from .runtime_config import get_runtime_config
from .situation_engine import situation_summary

_REGISTRY_LOCK = Lock()
_PERSONA_REGISTRY_STATUS_ACTIVE = 'active'
_PERSONA_REGISTRY_STATUS_DRAFT = 'draft'
_PERSONA_REGISTRY_STATUS_REJECTED = 'rejected'
_GENERIC_PERSONA_NAMES = {
    'human',
    'person',
    'people',
    'file',
    'document',
    'pdf',
    'docx',
    'odt',
    'fb2',
    'csv',
    'json',
    'markdown',
    'metadata',
    'object',
    'concept',
    'topic',
    'label',
    'tag',
    'engine',
    'game_engine',
    'unity',
    'unity_engine',
}
_GENERIC_PERSONA_TOKENS = {
    'human',
    'person',
    'people',
    'file',
    'document',
    'pdf',
    'docx',
    'odt',
    'fb2',
    'csv',
    'json',
    'markdown',
    'metadata',
    'object',
    'concept',
    'topic',
    'label',
    'tag',
    'query',
    'prompt',
    'fragment',
    'engine',
    'game',
    'unity',
}
_PROMPT_FRAGMENT_TOKENS = {
    'you',
    'your',
    'yourself',
    'ты',
    'тебя',
    'тебе',
    'твой',
    'твоя',
    'твои',
    'я',
    'мне',
    'меня',
    'как',
    'what',
    'why',
    'who',
    'how',
    'будешь',
    'питаешься',
    'действовать',
    'using',
    'used',
}
_BEHAVIOR_RELATION_TYPES = {
    'FEARS',
    'FEEDS_ON',
    'TRUSTS',
    'MISTRUSTS',
    'LOVES',
    'HATES',
    'DESIRES',
    'AVOIDS',
    'PROTECTS',
    'RESENTS',
    'OBEYS',
    'LEADS',
    'DEPENDS_ON',
    'WORKS_AS',
    'VALUES',
}
_BEHAVIOR_TEXT_MARKERS = (
    'fears',
    'fear',
    'trusts',
    'mistrusts',
    'loves',
    'hates',
    'desires',
    'avoids',
    'protects',
    'resents',
    'obeys',
    'leads',
    'depends on',
    'depends',
    'values',
    'prefers',
    'speaks',
    'reacts',
    'withdraws',
    'defends',
    'ashamed',
    'jealous',
    'conflicted',
    'боится',
    'доверяет',
    'ненавидит',
    'любит',
    'избегает',
    'защищает',
    'зависит',
    'ценит',
    'предпочитает',
    'говорит',
    'реагирует',
    'стыдно',
    'ревнует',
    'конфликт',
    'робкий',
    'гордый',
)
_STRUCTURED_PERSONA_FORM_KEYS = (
    'core_self_image',
    'vulnerabilities',
    'defense_mechanisms',
    'triggers',
    'dependency_patterns',
    'communication_style',
    'internal_contradictions',
    'change_resistance',
    'growth_dynamics',
    'decision_patterns',
    'speech_style',
    'speech_tendencies',
    'emotional_tendencies',
    'conflict_behavior',
)
_REJECT_EXACT = {
    'file',
    'pdf',
    'human',
    'unity',
    'game engine',
    'engine',
}
_REJECT_SUBSTRINGS = (
    'ты пита',
    'unity:',
    '/мой коммент/',
)
_PERSONA_CREATION_PREFIXES = (
    'create persona',
    'create personality',
    'build persona',
    'make a persona',
    'создай личность',
    'создай персонажа',
    'сделай личность',
    'сформируй личность',
)

_HEURISTIC_TRAIT_MARKERS: tuple[tuple[str, str], ...] = (
    ('робк', 'shy'),
    ('застенчив', 'shy'),
    ('нерешител', 'hesitant'),
    ('неуверен', 'hesitant'),
    ('горд', 'proud'),
    ('горделив', 'proud'),
    ('осторож', 'cautious'),
    ('сдержан', 'restrained'),
    ('тревож', 'anxious'),
    ('ревнив', 'jealous'),
    ('влюб', 'attached'),
    ('стыд', 'ashamed'),
    ('logical', 'logical'),
    ('analytical', 'analytical'),
    ('empathetic', 'empathetic'),
    ('warm', 'warm'),
    ('sarcastic', 'sarcastic'),
    ('formal', 'formal'),
    ('proud', 'proud'),
    ('shy', 'shy'),
    ('hesitant', 'hesitant'),
    ('cautious', 'cautious'),
    ('anxious', 'anxious'),
    ('jealous', 'jealous'),
    ('in love', 'attached'),
    ('ashamed', 'ashamed'),
)
_HEURISTIC_VULNERABILITY_MARKERS: tuple[tuple[str, str], ...] = (
    ('стыд', 'shame around admitting vulnerability'),
    ('ashamed', 'shame around admitting vulnerability'),
    ('робк', 'fear of direct confrontation'),
    ('shy', 'fear of direct confrontation'),
    ('влюб', 'emotional dependency on the desired person'),
    ('in love', 'emotional dependency on the desired person'),
    ('использует', 'susceptibility to exploitation by desired people'),
    ('used', 'susceptibility to exploitation by desired people'),
    ('одиноч', 'fear of abandonment and loneliness'),
    ('alone', 'fear of abandonment and loneliness'),
)
_HEURISTIC_DEFENSE_MARKERS: tuple[tuple[str, str], ...] = (
    ('горд', 'hides pain behind dignity and pride'),
    ('proud', 'hides pain behind dignity and pride'),
    ('осторож', 'keeps distance before trusting'),
    ('cautious', 'keeps distance before trusting'),
    ('сдержан', 'suppresses direct confession'),
    ('restrained', 'suppresses direct confession'),
    ('ирони', 'uses irony to avoid exposing weakness'),
    ('sarcas', 'uses irony to avoid exposing weakness'),
)
_HEURISTIC_TRIGGER_MARKERS: tuple[tuple[str, str], ...] = (
    ('использует', 'being used after showing attachment'),
    ('used', 'being used after showing attachment'),
    ('стыд', 'being made to feel weak or exposed'),
    ('ashamed', 'being made to feel weak or exposed'),
    ('гоняет', 'commands that turn affection into servitude'),
    ('slave', 'commands that turn affection into servitude'),
    ('преда', 'betrayal after trust'),
    ('betray', 'betrayal after trust'),
)
_HEURISTIC_DEPENDENCY_MARKERS: tuple[tuple[str, str], ...] = (
    ('влюб', 'attachment makes the persona slow to cut ties'),
    ('in love', 'attachment makes the persona slow to cut ties'),
    ('завис', 'depends on emotional reciprocity'),
    ('depends', 'depends on emotional reciprocity'),
    ('привяз', 'attachment makes the persona slow to cut ties'),
    ('attached', 'attachment makes the persona slow to cut ties'),
    ('одобр', 'seeks signs of chosen closeness'),
    ('approval', 'seeks signs of chosen closeness'),
)
_HEURISTIC_COMMUNICATION_MARKERS: tuple[tuple[str, str], ...] = (
    ('робк', 'hesitant and self-editing'),
    ('shy', 'hesitant and self-editing'),
    ('нерешител', 'breaks statements before committing'),
    ('hesitant', 'breaks statements before committing'),
    ('горд', 'keeps dignity visible in the wording'),
    ('proud', 'keeps dignity visible in the wording'),
    ('осторож', 'answers carefully and avoids overexposure'),
    ('cautious', 'answers carefully and avoids overexposure'),
    ('с паузами', 'uses pauses instead of full emotional disclosure'),
    ('with pauses', 'uses pauses instead of full emotional disclosure'),
)
_HEURISTIC_CONTRADICTION_MARKERS: tuple[tuple[tuple[str, ...], str], ...] = (
    ((('робк', 'shy'), ('горд', 'proud')), 'wants closeness but refuses visible humiliation'),
    ((('влюб', 'in love'), ('использует', 'used')), 'craves attachment but resents dependence'),
    ((('осторож', 'cautious'), ('влюб', 'in love')), 'longs for intimacy but resists surrendering control'),
)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def rejected_candidates_dir() -> Path:
    path = heads_dir() / '_rejected'
    path.mkdir(parents=True, exist_ok=True)
    return path


def rejected_candidates_log_path() -> Path:
    path = heads_dir() / '_rejected_candidates.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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


def _structured_persona_path(name: str) -> Path:
    return _head_file(name, 'structured_persona.json')


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


def _append_unique_limited(values: list[str], item: str, *, limit: int) -> list[str]:
    clean = str(item or '').strip()
    if not clean:
        return _normalize_string_list(values, limit=limit)
    return _normalize_string_list(list(values or []) + [clean], limit=limit)


def _prepend_unique_limited(values: list[str], item: str, *, limit: int) -> list[str]:
    clean = str(item or '').strip()
    if not clean:
        return _normalize_string_list(values, limit=limit)
    return _normalize_string_list([clean] + list(values or []), limit=limit)


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


def _append_rejected_candidate_log(
    *,
    name: str,
    slug: str,
    reason_codes: list[str],
    source: str,
    evidence: dict[str, Any] | None = None,
) -> None:
    payload = {
        'name': str(name or '').strip(),
        'slug': str(slug or '').strip(),
        'reason_codes': [str(item).strip() for item in list(reason_codes or []) if str(item).strip()],
        'source': str(source or '').strip(),
        'evidence': dict(evidence or {}),
        'timestamp': _utc_now(),
    }
    path = rejected_candidates_log_path()
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + '\n')


def normalize_text(text: str) -> str:
    return ' '.join(str(text or '').lower().strip().split())


def _strip_persona_creation_prefix(text: str) -> str:
    source = str(text or '').strip()
    if not source:
        return ''
    lines = [str(line).strip() for line in source.splitlines() if str(line).strip()]
    if not lines:
        return ''
    first_line = lines[0]
    first_norm = normalize_text(first_line)
    for marker in _PERSONA_CREATION_PREFIXES:
        marker_norm = normalize_text(marker)
        if first_norm == marker_norm:
            return '\n'.join(lines[1:]).strip() or source
        if first_norm.startswith(marker_norm):
            suffix = first_line[len(marker) :].lstrip(' :;-—')
            remainder = [suffix] if suffix else []
            remainder.extend(lines[1:])
            cleaned = '\n'.join(item for item in remainder if str(item).strip()).strip()
            return cleaned or source
    return source


def _looks_like_explicit_persona_name(label: str) -> bool:
    clean = ' '.join(str(label or '').strip().split())
    if not clean or looks_like_garbage_label(clean):
        return False
    if any(char in clean for char in '.!?'):
        return False
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё'\\-]*", clean)
    if not tokens or len(tokens) > 4:
        return False
    if any(normalize_text(token) in _PROMPT_FRAGMENT_TOKENS for token in tokens):
        return False
    uppercase_tokens = [token for token in tokens if token[:1].isupper()]
    if len(tokens) >= 2:
        return len(uppercase_tokens) >= 2
    return len(uppercase_tokens) == 1 and len(tokens[0]) >= 3


def extract_explicit_persona_name(description: str) -> str:
    cleaned = _strip_persona_creation_prefix(description)
    if not cleaned:
        return ''
    lines = [str(line).strip(' \t-—') for line in cleaned.splitlines() if str(line).strip()]
    if not lines:
        return ''
    first_line = lines[0]
    candidates = [
        first_line,
        re.split(r'[,;]', first_line, maxsplit=1)[0].strip(),
        re.split(r'\s+[—–-]\s+', first_line, maxsplit=1)[0].strip(),
        re.split(r':', first_line, maxsplit=1)[0].strip(),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        candidate_clean = ' '.join(str(candidate or '').strip().split())
        if not candidate_clean:
            continue
        normalized = normalize_text(candidate_clean)
        if normalized in seen:
            continue
        seen.add(normalized)
        if _looks_like_explicit_persona_name(candidate_clean):
            return candidate_clean
    return ''


def looks_like_garbage_label(label: str) -> bool:
    norm = normalize_text(label)
    if not norm:
        return True
    if norm in _REJECT_EXACT:
        return True
    if len(norm.split()) > 8:
        return True
    if any(part in norm for part in _REJECT_SUBSTRINGS):
        return True
    return bool(_persona_name_rejection_reasons(label))


def _listify_text(value: Any, *, limit: int = 8) -> list[str]:
    if isinstance(value, list):
        source = value
    elif isinstance(value, tuple):
        source = list(value)
    elif isinstance(value, str):
        source = [value]
    else:
        source = []
    return _normalize_string_list([str(item).strip() for item in source if str(item).strip()], limit=limit)


def _first_non_empty_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return ' '.join(value.strip().split())
    return ''


def _joined_field(value: Any) -> str:
    if isinstance(value, str):
        return ' '.join(value.strip().split())
    if isinstance(value, list):
        return '; '.join(str(item).strip() for item in value if str(item).strip())[:280]
    return ''


def _coerce_persona_type(value: Any, *, entity_type: str = '', visible_traits: list[str] | None = None, hidden_traits: list[str] | None = None) -> str:
    token = normalize_text(value)
    if token in {'archetype', 'psychological', 'situational', 'hybrid'}:
        return token
    if str(entity_type or '').strip().upper() == 'FICTIONAL_CHARACTER':
        return 'archetype'
    if list(hidden_traits or []):
        return 'psychological'
    if list(visible_traits or []):
        return 'hybrid'
    return 'situational'


def _coerce_change_direction(value: Any) -> ChangeDirection:
    token = normalize_text(value)
    if token in {'lighter', 'darker', 'mixed', 'unstable'}:
        return token  # type: ignore[return-value]
    return 'unstable'


def _persona_signal_count(persona: StructuredPersona) -> int:
    sections = (
        persona.core.self_image,
        persona.core.visible_traits,
        persona.core.hidden_traits,
        persona.core.motivations,
        persona.core.fears,
        persona.core.needs,
        persona.core.vulnerabilities,
        persona.conflict.internal_contradictions,
        persona.conflict.shame_points,
        persona.conflict.dependency_patterns,
        persona.conflict.resentment_patterns,
        persona.defense.defense_mechanisms,
        persona.defense.self_justifications,
        persona.defense.avoidance_patterns,
        persona.defense.escalation_patterns,
        persona.behavior.communication_style,
        persona.behavior.triggers,
        persona.behavior.pressure_response,
        persona.dynamics.softening_conditions,
        persona.dynamics.darkening_conditions,
    )
    count = sum(1 for section in sections if list(section or []))
    if persona.behavior.attachment_style:
        count += 1
    if persona.behavior.refusal_style:
        count += 1
    if persona.dynamics.resistance_to_change:
        count += 1
    if persona.dynamics.growth_pattern:
        count += 1
    return count


def _infer_persona_readiness(persona: StructuredPersona) -> str:
    signal_count = _persona_signal_count(persona)
    if persona.is_minimally_valid() and signal_count >= 9:
        return 'full'
    if persona.is_minimally_valid() and signal_count >= 5:
        return 'draft'
    return 'seed'


def _suggest_persona_label(persona: StructuredPersona) -> str:
    trait_tokens = {
        'proud': 'Proud',
        'shy': 'Shy',
        'hesitant': 'Hesitant',
        'cautious': 'Guarded',
        'restrained': 'Restrained',
        'anxious': 'Anxious',
        'attached': 'Dependent',
        'in_love': 'Attached',
        'ashamed': 'Ashamed',
        'jealous': 'Jealous',
        'sarcastic': 'Bitter',
        'quiet': 'Quiet',
    }
    contradiction_tokens = (
        ('depends', 'Dependent'),
        ('стыд', 'Ashamed'),
        ('shame', 'Ashamed'),
        ('humili', 'Cornered'),
        ('reject', 'Rejected'),
        ('use', 'Used'),
    )
    parts: list[str] = []
    for trait in list(persona.core.visible_traits) + list(persona.core.hidden_traits):
        mapped = trait_tokens.get(normalize_name(trait))
        if mapped and mapped not in parts:
            parts.append(mapped)
        if len(parts) >= 2:
            break
    combined_conflict = normalize_name(
        ' '.join(
            list(persona.conflict.internal_contradictions)
            + list(persona.conflict.dependency_patterns)
            + list(persona.conflict.shame_points)
        )
    )
    for marker, token in contradiction_tokens:
        if marker in combined_conflict and token not in parts:
            parts.append(token)
        if len(parts) >= 3:
            break
    if not parts:
        fallback = normalize_text(persona.identity.label).split()
        parts = [token[:1].upper() + token[1:] for token in fallback[:3] if token]
    return ' '.join(parts[:4]).strip() or 'Structured Persona'


def _default_short_description(persona: StructuredPersona) -> str:
    visible = ', '.join(list(persona.core.visible_traits)[:3])
    hidden = ', '.join(list(persona.core.hidden_traits)[:2])
    contradiction = next((item for item in list(persona.conflict.internal_contradictions) if str(item).strip()), '')
    parts = []
    if visible:
        parts.append(visible)
    if hidden:
        parts.append(f'while hiding {hidden}')
    if contradiction:
        parts.append(f'and lives inside the contradiction: {contradiction}')
    return ' '.join(part.strip() for part in parts if part.strip())[:240]


def _default_hover_text(persona: StructuredPersona) -> str:
    short = _first_non_empty_text(persona.identity.short_description, _default_short_description(persona))
    conflict = next((item for item in persona.conflict.internal_contradictions if str(item).strip()), '')
    defense = next((item for item in persona.defense.defense_mechanisms if str(item).strip()), '')
    parts = [short]
    if defense:
        parts.append(f'Defends with {defense}.')
    if conflict:
        parts.append(f'Core contradiction: {conflict}.')
    return ' '.join(part.strip() for part in parts if part.strip())[:320]


def _default_hard_system_constraints() -> list[str]:
    return [
        'no threats',
        'no coercion',
        'no kidnapping',
        'no violent domination',
    ]


def _infer_core_goal(persona: StructuredPersona) -> str:
    if str(persona.core_goal or '').strip():
        return str(persona.core_goal or '').strip()
    for candidate in list(persona.core.motivations or []) + list(persona.core.needs or []):
        clean = str(candidate or '').strip()
        if clean:
            return clean
    if persona.identity.short_description.strip():
        return f"protect the self-image implied by: {persona.identity.short_description.strip()}"
    return ''


def _infer_secondary_goals(persona: StructuredPersona) -> list[str]:
    seed = list(persona.secondary_goals or []) + list(persona.core.motivations or []) + list(persona.core.needs or [])
    primary = normalize_name(_infer_core_goal(persona))
    values: list[str] = []
    seen: set[str] = set()
    for item in seed:
        clean = str(item or '').strip()
        key = normalize_name(clean)
        if not clean or not key or key == primary or key in seen:
            continue
        seen.add(key)
        values.append(clean)
    return values[:6]


def _infer_constraints_internal(persona: StructuredPersona) -> list[str]:
    seed = list(persona.constraints_internal or [])
    seed.extend(list(persona.conflict.internal_contradictions or []))
    seed.extend(list(persona.conflict.shame_points or []))
    seed.extend(list(persona.core.vulnerabilities or []))
    values = _normalize_string_list(seed, limit=8)
    if values:
        return values
    if any(normalize_name(item) in {'proud', 'shy', 'hesitant', 'attached', 'ashamed'} for item in list(persona.core.visible_traits or []) + list(persona.core.hidden_traits or [])):
        return _normalize_string_list(
            [
                'cannot admit vulnerability directly',
                'cannot ask plainly when shame is active',
            ],
            limit=8,
        )
    return []


def _infer_constraints_social(persona: StructuredPersona) -> list[str]:
    seed = list(persona.constraints_social or [])
    seed.extend(list(persona.conflict.dependency_patterns or []))
    seed.extend(list(persona.conflict.resentment_patterns or []))
    values = _normalize_string_list(seed, limit=8)
    if values:
        return values
    if persona.behavior.attachment_style:
        return _normalize_string_list(
            [
                'cannot openly destroy the important relationship',
                'cannot break the social frame too bluntly',
            ],
            limit=8,
        )
    return []


def _infer_allowed_methods(persona: StructuredPersona) -> list[str]:
    seed = list(persona.allowed_methods or [])
    seed.extend(list(persona.behavior.communication_style or []))
    seed.extend(list(persona.behavior.pressure_response or []))
    values = _normalize_string_list(seed, limit=8)
    if values:
        return values
    return _normalize_string_list(['answer cautiously', 'probe indirectly'], limit=8)


def _infer_maladaptive_methods(persona: StructuredPersona) -> list[str]:
    seed = list(persona.maladaptive_methods or [])
    seed.extend(list(persona.defense.avoidance_patterns or []))
    seed.extend(list(persona.defense.escalation_patterns or []))
    seed.extend(list(persona.defense.defense_mechanisms or []))
    return _normalize_string_list(seed, limit=8)


def validate_persona(persona: StructuredPersona) -> tuple[str, list[str]]:
    errors: list[str] = []
    invalid_label = looks_like_garbage_label(persona.identity.label)
    if invalid_label:
        errors.append('invalid_label')

    trait_count = len(persona.core.visible_traits) + len(persona.core.hidden_traits)
    if trait_count < 1:
        errors.append('missing_traits')
    elif trait_count < 2:
        errors.append('insufficient_traits')

    has_depth = any(
        (
            len(persona.conflict.internal_contradictions) > 0,
            len(persona.conflict.dependency_patterns) > 0,
            len(persona.defense.defense_mechanisms) > 0,
            len(persona.behavior.communication_style) > 0,
        )
    )
    if not has_depth:
        errors.append('insufficient_psychological_depth')

    if not persona.meta.hover_text.strip():
        errors.append('missing_hover_text')

    if len(persona.identity.short_description.strip()) < 20:
        errors.append('short_description_too_weak')

    core_goal = _infer_core_goal(persona)
    secondary_goals = _infer_secondary_goals(persona)
    constraints_internal = _infer_constraints_internal(persona)
    constraints_social = _infer_constraints_social(persona)
    allowed_methods = _infer_allowed_methods(persona)
    maladaptive_methods = _infer_maladaptive_methods(persona)
    if not core_goal and not secondary_goals:
        errors.append('missing_goal_structure')
    if not constraints_internal and not constraints_social:
        errors.append('missing_constraint_structure')
    if not allowed_methods and not maladaptive_methods:
        errors.append('missing_method_structure')

    seed_like = str(persona.identity.readiness or '').strip().lower() == 'seed' or str(persona.identity.persona_type or '').strip().lower() == 'archetype'
    seed_scaffold_ok = bool(
        str(persona.identity.label or '').strip()
        and len(str(persona.identity.short_description or '').strip()) >= 20
        and str(persona.meta.hover_text or '').strip()
    )
    if errors:
        if invalid_label:
            return 'rejected', errors
        if trait_count < 1 and not has_depth:
            if seed_like and seed_scaffold_ok:
                return 'partial', list(dict.fromkeys(errors + ['seed_persona_scaffold_only']))
            return 'rejected', errors
        return 'partial', errors
    return 'valid', []


def _persona_form_from_structured(persona: StructuredPersona, *, entity_type: str) -> dict[str, Any]:
    return _validated_persona_form(
        {
            'identity_class': _identity_class(entity_type),
            'core_goal': _infer_core_goal(persona),
            'secondary_goals': _infer_secondary_goals(persona),
            'core_self_image': _joined_field(persona.core.self_image),
            'constraints_internal': _infer_constraints_internal(persona),
            'constraints_social': _infer_constraints_social(persona),
            'constraints_hard_system': _normalize_string_list(list(persona.constraints_hard_system or []) + _default_hard_system_constraints(), limit=8),
            'allowed_methods': _infer_allowed_methods(persona),
            'maladaptive_methods': _infer_maladaptive_methods(persona),
            'vulnerabilities': list(persona.core.vulnerabilities),
            'defense_mechanisms': list(persona.defense.defense_mechanisms),
            'triggers': list(persona.behavior.triggers),
            'dependency_patterns': list(persona.conflict.dependency_patterns),
            'communication_style': list(persona.behavior.communication_style),
            'internal_contradictions': list(persona.conflict.internal_contradictions),
            'change_resistance': _listify_text(persona.dynamics.resistance_to_change, limit=8),
            'growth_dynamics': _listify_text(persona.dynamics.growth_pattern, limit=8),
            'speech_tendencies': list(persona.behavior.communication_style),
            'speech_style': list(persona.behavior.communication_style),
            'decision_patterns': _normalize_string_list(list(persona.core.motivations) + list(persona.core.needs), limit=8),
            'emotional_tendencies': _normalize_string_list(list(persona.core.hidden_traits) + list(persona.core.fears), limit=8),
            'conflict_behavior': list(persona.behavior.pressure_response),
            'values': list(persona.secondary_goals or persona.core.motivations),
            'response_priorities': _normalize_string_list(list(persona.needs or persona.core.needs) + list(persona.secondary_goals or persona.core.motivations), limit=8),
        },
        fallback={},
    )


def _structured_persona_from_payload(
    name: str,
    payload: Any,
    *,
    fallback_examples: list[str] | None = None,
    entity_type: str = 'PERSON',
) -> StructuredPersona:
    root = dict(payload or {}) if isinstance(payload, dict) else {}
    if 'structured_persona' in root and isinstance(root.get('structured_persona'), dict):
        root = dict(root.get('structured_persona') or {})
    persona_payload_source = root.get('persona_payload') if isinstance(root.get('persona_payload'), dict) else root
    persona_payload = dict(persona_payload_source or {}) if isinstance(persona_payload_source or {}, dict) else {}
    nested = root if any(key in root for key in ('identity', 'core', 'conflict', 'defense', 'behavior', 'dynamics', 'meta')) else {}
    persona_form = (
        dict(root.get('persona_form') or {})
        if isinstance(root.get('persona_form') or {}, dict)
        else dict(persona_payload.get('persona_form') or {})
        if isinstance(persona_payload.get('persona_form') or {}, dict)
        else {}
    )
    identity_raw = dict(nested.get('identity') or {})
    core_raw = dict(nested.get('core') or {})
    conflict_raw = dict(nested.get('conflict') or {})
    defense_raw = dict(nested.get('defense') or {})
    behavior_raw = dict(nested.get('behavior') or {})
    dynamics_raw = dict(nested.get('dynamics') or {})
    meta_raw = dict(nested.get('meta') or {})
    top_level_fears = _listify_text(root.get('fears'), limit=10)
    top_level_needs = _listify_text(root.get('needs'), limit=10)
    source_text = _first_non_empty_text(
        identity_raw.get('source_text'),
        '\n'.join(str(item).strip() for item in list(fallback_examples or []) if str(item).strip()),
    )
    visible_traits = _listify_text(core_raw.get('visible_traits') or persona_payload.get('traits'), limit=12)
    hidden_traits = _listify_text(core_raw.get('hidden_traits') or persona_form.get('emotional_tendencies'), limit=12)
    persona = StructuredPersona(
        identity=PersonaIdentity(
            label=_first_non_empty_text(identity_raw.get('label'), persona_payload.get('name'), name),
            short_description=_first_non_empty_text(
                identity_raw.get('short_description'),
                meta_raw.get('hover_text'),
                root.get('decision_explanation'),
                persona_payload.get('knowledge'),
            ),
            persona_type=_coerce_persona_type(
                identity_raw.get('persona_type'),
                entity_type=entity_type,
                visible_traits=visible_traits,
                hidden_traits=hidden_traits,
            ),
            source_text=source_text,
            readiness='draft',
        ),
        core_goal=_first_non_empty_text(
            root.get('core_goal'),
            persona_form.get('core_goal'),
            persona_form.get('primary_goal'),
        ),
        secondary_goals=_listify_text(
            root.get('secondary_goals') or persona_form.get('secondary_goals'),
            limit=8,
        ),
        fears=top_level_fears,
        needs=top_level_needs,
        constraints_internal=_listify_text(
            root.get('constraints_internal') or persona_form.get('constraints_internal'),
            limit=8,
        ),
        constraints_social=_listify_text(
            root.get('constraints_social') or persona_form.get('constraints_social'),
            limit=8,
        ),
        constraints_hard_system=_listify_text(
            root.get('constraints_hard_system') or persona_form.get('constraints_hard_system'),
            limit=8,
        ),
        allowed_methods=_listify_text(
            root.get('allowed_methods') or persona_form.get('allowed_methods'),
            limit=8,
        ),
        maladaptive_methods=_listify_text(
            root.get('maladaptive_methods') or persona_form.get('maladaptive_methods'),
            limit=8,
        ),
        core=PersonaCore(
            self_image=_listify_text(core_raw.get('self_image') or persona_form.get('core_self_image'), limit=8),
            visible_traits=visible_traits,
            hidden_traits=hidden_traits,
            motivations=_listify_text(core_raw.get('motivations') or persona_form.get('values') or persona_form.get('response_priorities'), limit=10),
            fears=_listify_text(core_raw.get('fears') or root.get('fears') or persona_form.get('triggers'), limit=10),
            needs=_listify_text(core_raw.get('needs') or root.get('needs') or persona_form.get('response_priorities'), limit=10),
            vulnerabilities=_listify_text(core_raw.get('vulnerabilities') or persona_form.get('vulnerabilities'), limit=10),
        ),
        conflict=PersonaConflict(
            internal_contradictions=_listify_text(conflict_raw.get('internal_contradictions') or persona_form.get('internal_contradictions') or persona_form.get('conflicts'), limit=10),
            shame_points=_listify_text(conflict_raw.get('shame_points'), limit=10),
            dependency_patterns=_listify_text(conflict_raw.get('dependency_patterns') or persona_form.get('dependency_patterns'), limit=10),
            resentment_patterns=_listify_text(conflict_raw.get('resentment_patterns') or persona_form.get('conflict_behavior'), limit=10),
        ),
        defense=PersonaDefense(
            defense_mechanisms=_listify_text(defense_raw.get('defense_mechanisms') or persona_form.get('defense_mechanisms'), limit=10),
            self_justifications=_listify_text(defense_raw.get('self_justifications'), limit=10),
            avoidance_patterns=_listify_text(defense_raw.get('avoidance_patterns') or persona_form.get('reaction_patterns'), limit=10),
            escalation_patterns=_listify_text(defense_raw.get('escalation_patterns') or persona_form.get('conflict_behavior'), limit=10),
        ),
        behavior=PersonaBehavior(
            communication_style=_listify_text(behavior_raw.get('communication_style') or persona_form.get('communication_style') or persona_form.get('speech_tendencies'), limit=10),
            triggers=_listify_text(behavior_raw.get('triggers') or persona_form.get('triggers'), limit=10),
            pressure_response=_listify_text(behavior_raw.get('pressure_response') or persona_form.get('conflict_behavior') or persona_form.get('reaction_patterns'), limit=10),
            attachment_style=_first_non_empty_text(behavior_raw.get('attachment_style')),
            refusal_style=_first_non_empty_text(behavior_raw.get('refusal_style')),
        ),
        dynamics=PersonaDynamics(
            resistance_to_change=_first_non_empty_text(dynamics_raw.get('resistance_to_change'), _joined_field(persona_form.get('change_resistance'))),
            growth_pattern=_first_non_empty_text(dynamics_raw.get('growth_pattern'), _joined_field(persona_form.get('growth_dynamics'))),
            likely_change_direction=_coerce_change_direction(dynamics_raw.get('likely_change_direction')),
            softening_conditions=_listify_text(dynamics_raw.get('softening_conditions'), limit=8),
            darkening_conditions=_listify_text(dynamics_raw.get('darkening_conditions'), limit=8),
        ),
        meta=PersonaMeta(
            tags=_listify_text(meta_raw.get('tags') or persona_payload.get('traits'), limit=12),
            hover_text=_first_non_empty_text(meta_raw.get('hover_text')),
            validation_status='partial',
            validation_notes=_listify_text(meta_raw.get('validation_notes'), limit=8),
            confidence=float(meta_raw.get('confidence') or 0.0),
        ),
    )
    if not persona.identity.label or looks_like_garbage_label(persona.identity.label):
        persona.identity.label = _suggest_persona_label(persona)
    if not persona.identity.short_description.strip():
        persona.identity.short_description = _default_short_description(persona)
    persona.core_goal = _infer_core_goal(persona)
    persona.secondary_goals = _infer_secondary_goals(persona)
    persona.fears = _normalize_string_list(list(persona.fears or []) + list(persona.core.fears or []), limit=10)
    persona.needs = _normalize_string_list(list(persona.needs or []) + list(persona.core.needs or []), limit=10)
    persona.constraints_internal = _infer_constraints_internal(persona)
    persona.constraints_social = _infer_constraints_social(persona)
    persona.constraints_hard_system = _normalize_string_list(
        list(persona.constraints_hard_system or []) + _default_hard_system_constraints(),
        limit=8,
    )
    persona.allowed_methods = _infer_allowed_methods(persona)
    persona.maladaptive_methods = _infer_maladaptive_methods(persona)
    if not persona.meta.hover_text.strip():
        persona.meta.hover_text = _default_hover_text(persona)
    persona.identity.readiness = _first_non_empty_text(identity_raw.get('readiness')) or _infer_persona_readiness(persona)
    if persona.identity.readiness not in {'seed', 'draft', 'full'}:
        persona.identity.readiness = _infer_persona_readiness(persona)
    status, errors = validate_persona(persona)
    persona.meta.validation_status = status  # type: ignore[assignment]
    persona.meta.validation_notes = _normalize_string_list(list(persona.meta.validation_notes) + list(errors), limit=10)
    if not persona.meta.confidence:
        persona.meta.confidence = 0.91 if status == 'valid' else 0.66 if status == 'partial' else 0.12
    return persona

def _meaningful_trait_count(traits: list[str]) -> int:
    count = 0
    for trait in list(traits or []):
        clean = normalize_name(trait)
        if not clean:
            continue
        tokens = [token for token in clean.split() if token]
        if len(tokens) == 1 and tokens[0] in _GENERIC_PERSONA_TOKENS:
            continue
        count += 1
    return count


def _behavior_relation_count(relations: list[dict[str, Any]]) -> int:
    count = 0
    for item in list(relations or []):
        relation_type = str(item.get('type') or '').strip().upper()
        target = normalize_name(str(item.get('target') or item.get('to') or ''))
        if not relation_type or not target:
            continue
        if relation_type in _BEHAVIOR_RELATION_TYPES:
            count += 1
    return count


def _text_has_behavioral_signals(*parts: str) -> bool:
    lowered = normalize_name(' '.join(str(part or '').strip() for part in parts if str(part or '').strip()))
    return any(marker in lowered for marker in _BEHAVIOR_TEXT_MARKERS)


def _structured_field_count(persona_form: dict[str, Any]) -> int:
    count = 0
    for key in _STRUCTURED_PERSONA_FORM_KEYS:
        value = persona_form.get(key)
        if isinstance(value, str) and str(value).strip():
            count += 1
        elif isinstance(value, list) and any(str(item).strip() for item in value):
            count += 1
    return count


def _persona_name_rejection_reasons(name: str) -> list[str]:
    clean = ' '.join(str(name or '').strip().split())
    normalized = normalize_name(clean)
    if not normalized:
        return ['empty_name']
    reasons: list[str] = []
    tokens = [token for token in re.split(r'[\s_]+', normalized) if token]
    if normalize_text(clean) in _REJECT_EXACT:
        reasons.append('generic_ontology_or_media_label')
    if any(part in normalize_text(clean) for part in _REJECT_SUBSTRINGS):
        reasons.append('prompt_fragment_name')
    if normalized in {'unknown_head', 'new_head'}:
        reasons.append('unknown_placeholder_name')
    if len(clean) < 3:
        reasons.append('name_too_short')
    if len(tokens) > 6:
        reasons.append('name_too_long_for_persona_label')
    if normalized in _GENERIC_PERSONA_NAMES:
        reasons.append('generic_ontology_or_media_label')
    if tokens and all(token in _GENERIC_PERSONA_TOKENS for token in tokens):
        reasons.append('generic_token_only_name')
    if any(token in _PROMPT_FRAGMENT_TOKENS for token in tokens):
        reasons.append('prompt_fragment_name')
    if re.search(r'[?!]', clean):
        reasons.append('query_like_name')
    if clean.count('/') >= 2 or clean.count('\\') >= 2:
        reasons.append('path_like_or_comment_fragment')
    if ':' in clean and len(tokens) <= 3:
        reasons.append('metadata_style_name')
    if any(token in normalized for token in ('питаешься', 'будешь', 'действовать', 'использует', 'clarify', 'context')):
        reasons.append('verb_heavy_prompt_leftover')
    return list(dict.fromkeys(reasons))


def validate_persona_candidate(
    name: str,
    *,
    entity_type: str,
    traits: list[str],
    relations: list[dict[str, Any]],
    examples: list[str],
    knowledge: str,
    persona_form: dict[str, Any],
    decision_explanation: str,
    structured_persona: StructuredPersona | None = None,
    explicit: bool,
    source: str,
) -> dict[str, Any]:
    persona = structured_persona or _structured_persona_from_payload(
        name,
        {
            'persona_payload': {
                'name': name,
                'entity_type': entity_type,
                'traits': list(traits or []),
                'knowledge': knowledge,
            },
            'persona_form': dict(persona_form or {}),
            'decision_explanation': decision_explanation,
        },
        fallback_examples=list(examples or []),
        entity_type=entity_type,
    )
    validation_status, validation_notes = validate_persona(persona)
    reasons = list(dict.fromkeys(_persona_name_rejection_reasons(name) + list(validation_notes)))
    trait_count = _meaningful_trait_count(traits)
    relation_count = _behavior_relation_count(relations)
    structured_count = _structured_field_count(persona_form)
    behavior_text = bool(examples) and _text_has_behavioral_signals(*examples)
    knowledge_behavior = _text_has_behavioral_signals(knowledge, decision_explanation)
    has_identity_frame = bool(str(persona_form.get('identity_class') or '').strip())
    has_communication = bool(
        list(persona_form.get('communication_style') or [])
        or list(persona_form.get('speech_style') or [])
        or list(persona_form.get('speech_tendencies') or [])
    )
    has_goal_structure = bool(
        str(persona_form.get('core_goal') or '').strip()
        or list(persona_form.get('secondary_goals') or [])
        or str(persona.core_goal or '').strip()
        or list(persona.secondary_goals or [])
    )
    has_constraint_structure = bool(
        list(persona_form.get('constraints_internal') or [])
        or list(persona_form.get('constraints_social') or [])
        or list(persona_form.get('constraints_hard_system') or [])
        or list(persona.constraints_internal or [])
        or list(persona.constraints_social or [])
        or list(persona.constraints_hard_system or [])
    )
    has_method_structure = bool(
        list(persona_form.get('allowed_methods') or [])
        or list(persona_form.get('maladaptive_methods') or [])
        or list(persona.allowed_methods or [])
        or list(persona.maladaptive_methods or [])
    )
    has_inner_conflict = bool(
        list(persona_form.get('vulnerabilities') or [])
        or list(persona_form.get('internal_contradictions') or [])
        or list(persona_form.get('conflicts') or [])
    )
    has_motivation = bool(
        list(persona_form.get('dependency_patterns') or [])
        or list(persona_form.get('growth_dynamics') or [])
        or list(persona_form.get('values') or [])
        or list(persona_form.get('response_priorities') or [])
    )
    evidence_units = sum(
        1
        for item in (
            trait_count >= 2,
            relation_count >= 1,
            behavior_text,
            knowledge_behavior,
            structured_count >= 3,
            has_goal_structure,
            has_constraint_structure,
            has_method_structure,
            has_communication,
            has_inner_conflict,
            has_motivation,
        )
        if item
    )
    if str(entity_type or '').strip().upper() not in HEAD_ENTITY_TYPES:
        reasons.append('non_persona_entity_type')
    if explicit:
        if not has_identity_frame and validation_status == 'rejected':
            reasons.append('missing_identity_framing')
        if structured_count < 3:
            reasons.append('insufficient_structured_persona_fields')
        if not has_goal_structure:
            reasons.append('missing_goal_structure')
        if not has_constraint_structure:
            reasons.append('missing_constraint_structure')
        if not has_method_structure:
            reasons.append('missing_method_structure')
        if not (has_communication or has_inner_conflict or has_motivation):
            reasons.append('missing_behavioral_personality_structure')
    else:
        if evidence_units < 2 or not (trait_count >= 2 or relation_count >= 1 or behavior_text or knowledge_behavior):
            reasons.append('insufficient_behavioral_evidence')
    reasons = list(dict.fromkeys(reasons))
    return {
        'ok': validation_status != 'rejected' and not _persona_name_rejection_reasons(name),
        'validation_status': validation_status,
        'reason_codes': reasons,
        'structured_persona': persona.to_dict(),
        'evidence': {
            'entity_type': str(entity_type or '').strip().upper(),
            'trait_count': int(trait_count),
            'relation_count': int(relation_count),
            'structured_field_count': int(structured_count),
            'behavior_text': bool(behavior_text),
            'knowledge_behavior': bool(knowledge_behavior),
            'has_identity_frame': bool(has_identity_frame),
            'has_communication': bool(has_communication),
            'has_goal_structure': bool(has_goal_structure),
            'has_constraint_structure': bool(has_constraint_structure),
            'has_method_structure': bool(has_method_structure),
            'has_inner_conflict': bool(has_inner_conflict),
            'has_motivation': bool(has_motivation),
            'evidence_units': int(evidence_units),
            'explicit': bool(explicit),
            'source': str(source or '').strip(),
            'readiness': persona.identity.readiness,
            'validation_status': validation_status,
        },
    }


def _quarantine_persona_directory(name: str, *, reason_codes: list[str], source: str, evidence: dict[str, Any] | None = None) -> str:
    clean = normalize_personality_name(name)
    source_path = _head_dir(clean)
    if source_path.exists():
        target = rejected_candidates_dir() / f'{clean}__{datetime.now(UTC).strftime("%Y%m%d%H%M%S")}'
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(source_path), str(target))
        archive_path = str(target)
    else:
        archive_path = ''
    _append_rejected_candidate_log(
        name=name,
        slug=clean,
        reason_codes=reason_codes,
        source=source,
        evidence=evidence,
    )
    return archive_path


def _registry_validation_from_bundle(bundle: HeadBundle) -> dict[str, Any]:
    return validate_persona_candidate(
        bundle.name,
        entity_type=bundle.entity_type,
        traits=list(bundle.traits),
        relations=[dict(item) for item in list(bundle.relations or [])],
        examples=list(bundle.examples),
        knowledge=str(bundle.knowledge or '').strip(),
        persona_form=dict(bundle.persona_form or {}),
        decision_explanation=str(bundle.decision_explanation or '').strip(),
        structured_persona=bundle.structured_persona,
        explicit=True,
        source=str(bundle.meta.get('source') or 'registry'),
    )


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


def ensure_persona_registry_hygiene() -> dict[str, Any]:
    with _REGISTRY_LOCK:
        original_index = list(_load_index())
        active_rows: list[str] = []
        rejected: list[dict[str, Any]] = []
        for name in original_index:
            bundle = load_persona(name)
            if bundle is None:
                rejected.append({'name': name, 'reason_codes': ['missing_persona_files']})
                continue
            validation = _registry_validation_from_bundle(bundle)
            if validation.get('ok'):
                active_rows.append(normalize_personality_name(bundle.name))
                continue
            archive_path = _quarantine_persona_directory(
                bundle.name,
                reason_codes=list(validation.get('reason_codes') or []),
                source='registry_hygiene',
                evidence=dict(validation.get('evidence') or {}),
            )
            rejected.append(
                {
                    'name': bundle.name,
                    'reason_codes': list(validation.get('reason_codes') or []),
                    'archive_path': archive_path,
                }
            )
        active_rows = list(dict.fromkeys(active_rows))
        if active_rows != original_index:
            _save_index(active_rows)
        return {
            'ok': True,
            'active_count': len(active_rows),
            'rejected_count': len(rejected),
            'rejected': rejected,
        }


def persona_is_registered(name: str) -> bool:
    clean = normalize_personality_name(name)
    ensure_persona_registry_hygiene()
    return clean in _load_index()


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
            'registry_status': _PERSONA_REGISTRY_STATUS_DRAFT,
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
        'structured_persona.json': {},
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
    structured_persona_payload = load_json(_structured_persona_path(clean), {})
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
    structured_persona = _structured_persona_from_payload(
        clean,
        structured_persona_payload if isinstance(structured_persona_payload, dict) and structured_persona_payload else {
            'identity': {
                'label': str(meta_payload.get('display_label') or meta_payload.get('name') or baseline.name or clean),
                'short_description': str(meta_payload.get('short_description') or ''),
                'persona_type': str(meta_payload.get('persona_type') or ''),
                'source_text': str(meta_payload.get('source_text') or ''),
                'readiness': str(meta_payload.get('readiness') or ''),
            },
            'persona_payload': {
                'name': str(meta_payload.get('name') or baseline.name or clean),
                'entity_type': str(baseline.entity_type or meta_payload.get('entity_type') or 'PERSON'),
                'traits': list(baseline.traits),
                'knowledge': str(baseline.knowledge or ''),
            },
            'persona_form': dict(learned.persona_form),
            'meta': {
                'hover_text': str(meta_payload.get('hover_text') or ''),
                'validation_status': str(meta_payload.get('validation_status') or ''),
                'validation_notes': list(meta_payload.get('validation_notes') or []),
                'confidence': float(meta_payload.get('persona_confidence') or 0.0),
                'tags': list(meta_payload.get('persona_tags') or []),
            },
        },
        fallback_examples=list(learned.examples),
        entity_type=str(baseline.entity_type or meta_payload.get('entity_type') or 'PERSON'),
    )
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
        structured_persona=structured_persona,
    )


def load_active_persona(name: str) -> HeadBundle | None:
    clean = normalize_personality_name(name)
    ensure_persona_registry_hygiene()
    if clean not in _load_index():
        return None
    return load_persona(clean)


def _load_mutable_active_persona(name: str) -> HeadBundle | None:
    clean = normalize_personality_name(name)
    if not clean:
        return None
    active = load_active_persona(clean)
    if active is not None:
        return active
    draft = load_persona(clean)
    if draft is None:
        return None
    if _persona_name_rejection_reasons(draft.name):
        return None
    return draft


def load_persona_graph(name: str) -> dict[str, Any]:
    payload = load_json(personality_graph_path(name), {'nodes': [], 'edges': []})
    return payload if isinstance(payload, dict) else {'nodes': [], 'edges': []}


def list_personas() -> list[dict[str, Any]]:
    ensure_persona_registry_hygiene()
    rows: list[dict[str, Any]] = []
    for name in _load_index():
        bundle = load_persona(name)
        if bundle is None:
            continue
        persona = bundle.structured_persona or _structured_persona_from_payload(
            bundle.name,
            {'persona_payload': {'name': bundle.name, 'entity_type': bundle.entity_type, 'traits': list(bundle.traits), 'knowledge': bundle.knowledge}, 'persona_form': dict(bundle.persona_form)},
            fallback_examples=list(bundle.examples),
            entity_type=bundle.entity_type,
        )
        rows.append(
            {
                'name': bundle.name,
                'slug': bundle.meta.get('slug') or normalize_personality_name(bundle.name),
                'label': persona.identity.label,
                'short_description': persona.identity.short_description,
                'readiness': persona.identity.readiness,
                'entity_type': bundle.entity_type,
                'emotion_vector': bundle.emotion_vector,
                'folder': bundle.folder,
                'validation_status': persona.meta.validation_status,
                'validation_notes': list(persona.meta.validation_notes),
                'hover_text': persona.meta.hover_text,
                'confidence': persona.meta.confidence,
                'tags': list(persona.meta.tags),
                'indicators': bundle.indicators.to_dict() if bundle.indicators is not None else {},
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


def _collect_marker_hits(text: str, markers: tuple[tuple[str, str], ...], *, limit: int) -> list[str]:
    lowered = normalize_name(text)
    hits: list[str] = []
    seen: set[str] = set()
    for marker, label in markers:
        normalized_marker = normalize_name(marker)
        if not normalized_marker or normalized_marker not in lowered:
            continue
        if label in seen:
            continue
        seen.add(label)
        hits.append(label)
        if len(hits) >= max(int(limit or 0), 1):
            break
    return hits


def _infer_persona_label_from_examples(name: str, excerpts: list[str]) -> str:
    clean_name = str(name or '').strip()
    if clean_name:
        return clean_name
    combined = normalize_name(' '.join(excerpts))
    for marker, label in (
        ('вампир', 'Vampire Persona'),
        ('vampire', 'Vampire Persona'),
        ('капитан', 'Captain Persona'),
        ('captain', 'Captain Persona'),
        ('врач', 'Doctor Persona'),
        ('doctor', 'Doctor Persona'),
        ('юрист', 'Lawyer Persona'),
        ('lawyer', 'Lawyer Persona'),
    ):
        if marker in combined:
            return label
    return 'Custom Persona'


def _heuristic_persona_payload(
    name: str,
    excerpts: list[str],
    *,
    existing_bundle: HeadBundle | None = None,
    relations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    combined = ' '.join(str(item or '').strip() for item in list(excerpts or []) if str(item or '').strip())
    inferred_name = _infer_persona_label_from_examples(name, excerpts)
    traits = _normalize_string_list(
        list(existing_bundle.traits if existing_bundle is not None else [])
        + _collect_marker_hits(combined, _HEURISTIC_TRAIT_MARKERS, limit=10)
        + _detect_traits(excerpts),
        limit=_memory_limits().persona_trait_limit,
    )
    vulnerabilities = _collect_marker_hits(combined, _HEURISTIC_VULNERABILITY_MARKERS, limit=8)
    defenses = _collect_marker_hits(combined, _HEURISTIC_DEFENSE_MARKERS, limit=8)
    triggers = _collect_marker_hits(combined, _HEURISTIC_TRIGGER_MARKERS, limit=8)
    dependency_patterns = _collect_marker_hits(combined, _HEURISTIC_DEPENDENCY_MARKERS, limit=8)
    communication_style = _collect_marker_hits(combined, _HEURISTIC_COMMUNICATION_MARKERS, limit=8)
    emotional_tendencies = _normalize_string_list(
        [trait for trait in traits if trait in {'shy', 'hesitant', 'anxious', 'proud', 'attached', 'jealous', 'ashamed'}],
        limit=8,
    )
    contradictions: list[str] = []
    normalized_text = normalize_name(combined)
    for marker_pair, description in _HEURISTIC_CONTRADICTION_MARKERS:
        if all(any(normalize_name(marker) in normalized_text for marker in group) for group in marker_pair):
            contradictions.append(description)
    decision_patterns: list[str] = []
    if any(item in traits for item in {'cautious', 'anxious', 'hesitant'}):
        decision_patterns.append('slows down before committing to an emotionally risky move')
    if any(item in traits for item in {'proud', 'restrained'}):
        decision_patterns.append('protects dignity before making a confession or request')
    if dependency_patterns:
        decision_patterns.append('tests whether attachment is being used before offering more access')
    if not decision_patterns:
        decision_patterns.append('tries to read the emotional cost before acting')
    speech_tendencies = list(communication_style)
    if 'hesitant and self-editing' in speech_tendencies:
        speech_tendencies.append('uses pauses and unfinished starts under emotional pressure')
    growth_dynamics = []
    if dependency_patterns or triggers:
        growth_dynamics.append('can grow by setting boundaries before attachment turns into self-erasure')
    if contradictions:
        growth_dynamics.append('improves when pride stops blocking honest self-protection')
    change_resistance = []
    if any(item in traits for item in {'proud', 'cautious', 'restrained'}):
        change_resistance.append('resists asking for help because it feels like visible weakness')
    if any(item in traits for item in {'attached', 'shy'}):
        change_resistance.append('hesitates to break attachment even when the pattern is harmful')
    core_self_image_parts: list[str] = []
    if 'proud' in traits:
        core_self_image_parts.append('someone who must preserve dignity')
    if 'shy' in traits or 'hesitant' in traits:
        core_self_image_parts.append('someone who exposes feelings reluctantly')
    if 'attached' in traits:
        core_self_image_parts.append('someone whose attachment can override caution')
    core_self_image = ', '.join(core_self_image_parts[:3]) or 'someone trying to protect self-respect under emotional pressure'
    core_goal = (
        'preserve dignity without losing the emotionally significant bond'
        if dependency_patterns or 'proud' in traits
        else 'protect a coherent sense of self under pressure'
    )
    secondary_goals = _normalize_string_list(
        ['avoid humiliation', 'keep emotional significance', 'avoid visible neediness'],
        limit=8,
    )
    constraints_internal = _normalize_string_list(
        [
            'cannot admit vulnerability directly' if any(item in traits for item in {'proud', 'shy', 'hesitant'}) else '',
            'cannot confess attachment cleanly' if dependency_patterns else '',
        ],
        limit=8,
    )
    constraints_social = _normalize_string_list(
        [
            'cannot fully destroy the relationship frame' if dependency_patterns else '',
            'cannot behave too openly desperate' if dependency_patterns or 'attached' in traits else '',
        ],
        limit=8,
    )
    allowed_methods = _normalize_string_list(
        decision_patterns + ['probe indirectly', 'withdraw before full collapse'],
        limit=8,
    )
    maladaptive_methods = _normalize_string_list(
        list(defenses) + ['endure too long', 'mask dependence with usefulness'] if dependency_patterns else list(defenses),
        limit=8,
    )
    heuristic_form = {
        'identity_class': str((existing_bundle.persona_form if existing_bundle is not None else {}).get('identity_class') or 'human'),
        'core_goal': core_goal,
        'secondary_goals': secondary_goals,
        'core_self_image': core_self_image,
        'constraints_internal': constraints_internal,
        'constraints_social': constraints_social,
        'constraints_hard_system': _default_hard_system_constraints(),
        'allowed_methods': allowed_methods,
        'maladaptive_methods': maladaptive_methods,
        'vulnerabilities': vulnerabilities,
        'defense_mechanisms': defenses,
        'triggers': triggers,
        'dependency_patterns': dependency_patterns,
        'communication_style': communication_style or ['guarded and direct'],
        'internal_contradictions': contradictions,
        'change_resistance': change_resistance,
        'growth_dynamics': growth_dynamics,
        'decision_patterns': decision_patterns,
        'speech_tendencies': speech_tendencies,
        'emotional_tendencies': emotional_tendencies,
        'conflict_behavior': _normalize_string_list(
            ['withdraws to regain dignity', 'sets a boundary when exploitation becomes explicit'] if triggers else ['tests intent before escalating conflict'],
            limit=8,
        ),
    }
    relation_rows = _validate_relations(list(existing_bundle.relations if existing_bundle is not None else []) + list(relations or []))
    if any(marker in normalized_text for marker in ('использует', 'used', 'manipulat', 'гоняет')):
        relation_rows = _validate_relations(relation_rows + [{'type': 'RESENTS', 'target': 'exploitative attachment', 'weight': 0.82}])
    if any(marker in normalized_text for marker in ('влюб', 'in love', 'likes him', 'likes her')):
        relation_rows = _validate_relations(relation_rows + [{'type': 'DESIRES', 'target': 'desired person', 'weight': 0.86}])
    knowledge = combined[: _memory_limits().persona_knowledge_char_limit]
    structured_persona = _structured_persona_from_payload(
        inferred_name,
        {
            'identity': {
                'label': inferred_name,
                'short_description': _first_non_empty_text(
                    '',
                    f'{", ".join(traits[:3])} persona shaped by pride, attachment, and defensive self-protection.' if traits else '',
                ),
                'persona_type': 'psychological' if contradictions or dependency_patterns or defenses else 'hybrid',
                'source_text': combined,
            },
            'core_goal': core_goal,
            'secondary_goals': secondary_goals,
            'fears': ['being used', 'being exposed as weak'] if triggers or vulnerabilities else [],
            'needs': ['respect', 'emotional significance'] if traits else [],
            'constraints_internal': constraints_internal,
            'constraints_social': constraints_social,
            'constraints_hard_system': _default_hard_system_constraints(),
            'allowed_methods': allowed_methods,
            'maladaptive_methods': maladaptive_methods,
            'core': {
                'self_image': _listify_text(core_self_image, limit=8),
                'visible_traits': list(traits[:8]),
                'hidden_traits': list(emotional_tendencies[:8]),
                'motivations': ['preserve dignity', 'avoid humiliation'] if any(item in traits for item in {'proud', 'restrained'}) else [],
                'fears': ['being used', 'being exposed as weak'] if triggers or vulnerabilities else [],
                'needs': ['respect', 'emotional significance'] if traits else [],
                'vulnerabilities': list(vulnerabilities[:8]),
            },
            'conflict': {
                'internal_contradictions': list(contradictions[:8]),
                'dependency_patterns': list(dependency_patterns[:8]),
                'resentment_patterns': ['resentment builds when attachment turns into exploitation'] if dependency_patterns else [],
            },
            'defense': {
                'defense_mechanisms': list(defenses[:8]),
                'avoidance_patterns': ['avoids direct confession under emotional pressure'] if any(item in traits for item in {'shy', 'hesitant', 'restrained'}) else [],
                'escalation_patterns': ['withdraws first, then hardens into biting self-protection'] if triggers else [],
            },
            'behavior': {
                'communication_style': list((communication_style or ['guarded and direct'])[:8]),
                'triggers': list(triggers[:8]),
                'pressure_response': list((heuristic_form.get('conflict_behavior') or [])[:8]),
                'attachment_style': 'anxious-dependent' if dependency_patterns else None,
                'refusal_style': 'struggles to refuse directly when attachment is active' if dependency_patterns else None,
            },
            'dynamics': {
                'resistance_to_change': _joined_field(change_resistance),
                'growth_pattern': _joined_field(growth_dynamics),
                'likely_change_direction': 'unstable',
            },
            'meta': {
                'tags': list(traits[:8]),
            },
        },
        fallback_examples=list(excerpts),
        entity_type=str(existing_bundle.entity_type if existing_bundle is not None else 'PERSON'),
    )
    return {
        'name': inferred_name,
        'entity_type': str(existing_bundle.entity_type if existing_bundle is not None else 'PERSON'),
        'traits': traits,
        'examples': list(excerpts),
        'relations': relation_rows,
        'knowledge': knowledge,
        'structured_persona': structured_persona.to_dict(),
        'persona_form': heuristic_form,
        'decision_explanation': _default_decision_explanation(inferred_name, _validated_persona_form(heuristic_form, fallback=_default_persona_form(
            inferred_name,
            entity_type=str(existing_bundle.entity_type if existing_bundle is not None else 'PERSON'),
            traits=traits,
            relations=relation_rows,
            examples=list(excerpts),
            log_tuples=list(existing_bundle.log_tuples if existing_bundle is not None else []),
            existing_form=dict(existing_bundle.persona_form) if existing_bundle is not None else {},
        ))),
    }


def _merge_persona_payloads(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary or {})
    for key, value in dict(secondary or {}).items():
        if key == 'traits':
            merged[key] = _normalize_string_list(
                list(primary.get(key) or []) + list(value or []),
                limit=_memory_limits().persona_trait_limit,
            )
        elif key == 'aliases':
            merged[key] = merge_aliases(list(primary.get(key) or []), list(value or []))
        elif key == 'examples':
            merged[key] = _normalize_string_list(
                list(primary.get(key) or []) + list(value or []),
                limit=_memory_limits().persona_example_limit,
            )
        elif key == 'relations':
            merged[key] = _validate_relations(list(primary.get(key) or []) + list(value or []))
        elif key == 'persona_form':
            base_form = dict(primary.get('persona_form') or {})
            extra_form = dict(value or {}) if isinstance(value, dict) else {}
            merged[key] = _validated_persona_form(extra_form, fallback=_validated_persona_form(base_form, fallback=base_form or {}))
        elif key == 'structured_persona':
            base_persona = dict(primary.get('structured_persona') or {})
            extra_persona = dict(value or {}) if isinstance(value, dict) else {}
            merged[key] = {
                **base_persona,
                **extra_persona,
                'identity': {
                    **dict(base_persona.get('identity') or {}),
                    **dict(extra_persona.get('identity') or {}),
                },
                'core': {
                    **dict(base_persona.get('core') or {}),
                    **dict(extra_persona.get('core') or {}),
                },
                'conflict': {
                    **dict(base_persona.get('conflict') or {}),
                    **dict(extra_persona.get('conflict') or {}),
                },
                'defense': {
                    **dict(base_persona.get('defense') or {}),
                    **dict(extra_persona.get('defense') or {}),
                },
                'behavior': {
                    **dict(base_persona.get('behavior') or {}),
                    **dict(extra_persona.get('behavior') or {}),
                },
                'dynamics': {
                    **dict(base_persona.get('dynamics') or {}),
                    **dict(extra_persona.get('dynamics') or {}),
                },
                'meta': {
                    **dict(base_persona.get('meta') or {}),
                    **dict(extra_persona.get('meta') or {}),
                },
            }
        elif key == 'decision_explanation':
            merged[key] = str(value or primary.get(key) or '').strip() or str(primary.get(key) or '').strip()
        elif key == 'knowledge':
            merged[key] = str(value or '').strip() or str(primary.get(key) or '').strip()
        else:
            merged[key] = value if value not in (None, '', [], {}) else merged.get(key)
    return merged


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
    normalized_traits = [normalize_name(item) for item in traits]
    inferred_self_image = str((existing_form or {}).get('core_self_image') or '').strip()
    if not inferred_self_image:
        if 'proud' in normalized_traits or 'aristocratic' in normalized_traits:
            inferred_self_image = 'someone who protects dignity before exposing weakness'
        elif 'logical' in normalized_traits or 'analytical' in normalized_traits:
            inferred_self_image = 'someone who trusts structure, consistency, and evidence'
        elif 'empathetic' in normalized_traits or 'warm' in normalized_traits:
            inferred_self_image = 'someone who tries to protect others without losing clarity'
        elif 'aggressive' in normalized_traits or 'predatory' in normalized_traits:
            inferred_self_image = 'someone who asserts control quickly when pressure rises'
        else:
            inferred_self_image = f'{name} is treated as a persona with a stable response pattern'
    inferred_growth = _normalize_string_list(
        list((existing_form or {}).get('growth_dynamics') or [])
        or ['can adapt when new evidence or repeated consequences force revision'],
        limit=8,
    )
    inferred_change_resistance = _normalize_string_list(
        list((existing_form or {}).get('change_resistance') or [])
        or (['resists changing the response style without a strong reason'] if normalized_traits else []),
        limit=8,
    )
    existing_conflicts = _normalize_string_list((existing_form or {}).get('conflicts') or (existing_form or {}).get('weaknesses') or [], limit=8)
    existing_speech = _normalize_string_list((existing_form or {}).get('speech_style') or [], limit=8)
    form = {
        'identity_class': _identity_class(entity_type),
        'core_goal': str((existing_form or {}).get('core_goal') or decision_patterns[0] or '').strip(),
        'secondary_goals': _normalize_string_list((existing_form or {}).get('secondary_goals') or [], limit=8),
        'interaction_style': _normalize_string_list((existing_form or {}).get('interaction_style') or interaction_style, limit=8),
        'core_dispositions': _normalize_string_list((existing_form or {}).get('core_dispositions') or traits, limit=12),
        'biography': str((existing_form or {}).get('biography') or '').strip()[:1200],
        'core_self_image': inferred_self_image[:280],
        'constraints_internal': _normalize_string_list((existing_form or {}).get('constraints_internal') or existing_conflicts, limit=8),
        'constraints_social': _normalize_string_list((existing_form or {}).get('constraints_social') or relation_targets, limit=8),
        'constraints_hard_system': _normalize_string_list((existing_form or {}).get('constraints_hard_system') or _default_hard_system_constraints(), limit=8),
        'allowed_methods': _normalize_string_list((existing_form or {}).get('allowed_methods') or decision_patterns, limit=8),
        'maladaptive_methods': _normalize_string_list((existing_form or {}).get('maladaptive_methods') or (existing_form or {}).get('reaction_patterns') or [], limit=8),
        'social_roles': _normalize_string_list((existing_form or {}).get('social_roles') or interaction_style, limit=8),
        'habits': _normalize_string_list((existing_form or {}).get('habits') or (existing_form or {}).get('work_habits') or [], limit=10),
        'values': _normalize_string_list((existing_form or {}).get('values') or [], limit=8),
        'conflicts': existing_conflicts,
        'vulnerabilities': _normalize_string_list((existing_form or {}).get('vulnerabilities') or existing_conflicts, limit=8),
        'defense_mechanisms': _normalize_string_list((existing_form or {}).get('defense_mechanisms') or [], limit=8),
        'triggers': _normalize_string_list((existing_form or {}).get('triggers') or [], limit=8),
        'dependency_patterns': _normalize_string_list((existing_form or {}).get('dependency_patterns') or [], limit=8),
        'topic_affinities': _normalize_string_list((existing_form or {}).get('topic_affinities') or relation_targets, limit=10),
        'speech_style': existing_speech,
        'speech_tendencies': _normalize_string_list((existing_form or {}).get('speech_tendencies') or (existing_form or {}).get('speech_style') or [], limit=8),
        'communication_style': _normalize_string_list((existing_form or {}).get('communication_style') or existing_speech or interaction_style, limit=8),
        'emotional_tendencies': _normalize_string_list((existing_form or {}).get('emotional_tendencies') or [], limit=8),
        'conflict_behavior': _normalize_string_list((existing_form or {}).get('conflict_behavior') or [], limit=8),
        'decision_patterns': _normalize_string_list((existing_form or {}).get('decision_patterns') or decision_patterns, limit=8),
        'reaction_patterns': _normalize_string_list((existing_form or {}).get('reaction_patterns') or (existing_form or {}).get('conflict_behavior') or decision_patterns, limit=10),
        'clarification_policy': str((existing_form or {}).get('clarification_policy') or 'Ask a clarifying question when the target, intent, or grounding is insufficient.').strip(),
        'sarcasm_profile': str((existing_form or {}).get('sarcasm_profile') or _sarcasm_profile(traits, examples)).strip() or 'none',
        'response_priorities': _normalize_string_list(
            (existing_form or {}).get('response_priorities') or ['answer_substance', 'clarify_if_underspecified', 'stay_in_character'],
            limit=8,
        ),
        'knowledge_domains': _normalize_string_list((existing_form or {}).get('knowledge_domains') or relation_targets, limit=10),
        'risk_controls': _normalize_string_list((existing_form or {}).get('risk_controls') or risk_controls, limit=8),
        'trust_model': _normalize_string_list((existing_form or {}).get('trust_model') or [], limit=8),
        'work_habits': _normalize_string_list((existing_form or {}).get('work_habits') or [], limit=8),
        'memory_anchors': _normalize_string_list((existing_form or {}).get('memory_anchors') or [], limit=10),
        'memories': _normalize_string_list((existing_form or {}).get('memories') or (existing_form or {}).get('memory_anchors') or [], limit=12),
        'recurring_style_markers': _normalize_string_list((existing_form or {}).get('recurring_style_markers') or [], limit=8),
        'strengths': _normalize_string_list((existing_form or {}).get('strengths') or [], limit=8),
        'weaknesses': _normalize_string_list((existing_form or {}).get('weaknesses') or [], limit=8),
        'internal_contradictions': _normalize_string_list((existing_form or {}).get('internal_contradictions') or existing_conflicts, limit=8),
        'change_resistance': inferred_change_resistance,
        'growth_dynamics': inferred_growth,
        'personal_history': _normalize_string_list((existing_form or {}).get('personal_history') or [], limit=10),
        'log_signature_count': len(log_tuples),
    }
    return form


def _validated_persona_form(value: Any, *, fallback: dict[str, Any]) -> dict[str, Any]:
    raw = dict(value or {}) if isinstance(value, dict) else {}
    result = dict(fallback)
    for key in ('identity_class', 'clarification_policy', 'sarcasm_profile', 'core_goal'):
        token = str(raw.get(key) or result.get(key) or '').strip()
        if token:
            result[key] = token
    result['biography'] = str(raw.get('biography') or result.get('biography') or '').strip()[:1200]
    for key, limit in (
        ('interaction_style', 8),
        ('secondary_goals', 8),
        ('core_dispositions', 12),
        ('constraints_internal', 8),
        ('constraints_social', 8),
        ('constraints_hard_system', 8),
        ('allowed_methods', 8),
        ('maladaptive_methods', 8),
        ('social_roles', 8),
        ('habits', 10),
        ('values', 8),
        ('conflicts', 8),
        ('vulnerabilities', 8),
        ('defense_mechanisms', 8),
        ('triggers', 8),
        ('dependency_patterns', 8),
        ('topic_affinities', 10),
        ('speech_style', 8),
        ('speech_tendencies', 8),
        ('communication_style', 8),
        ('emotional_tendencies', 8),
        ('conflict_behavior', 8),
        ('decision_patterns', 8),
        ('reaction_patterns', 10),
        ('response_priorities', 8),
        ('knowledge_domains', 10),
        ('risk_controls', 8),
        ('trust_model', 8),
        ('work_habits', 8),
        ('memory_anchors', 10),
        ('memories', 12),
        ('recurring_style_markers', 8),
        ('strengths', 8),
        ('weaknesses', 8),
        ('internal_contradictions', 8),
        ('change_resistance', 8),
        ('growth_dynamics', 8),
        ('personal_history', 10),
    ):
        result[key] = _normalize_string_list(list(raw.get(key) or result.get(key) or []), limit=limit)
    result['core_self_image'] = str(raw.get('core_self_image') or result.get('core_self_image') or '').strip()[:280]
    result['log_signature_count'] = int(raw.get('log_signature_count') or result.get('log_signature_count') or 0)
    return result


def _default_decision_explanation(name: str, persona_form: dict[str, Any]) -> str:
    identity = str(persona_form.get('identity_class') or 'persona').replace('_', ' ')
    patterns = list(persona_form.get('decision_patterns') or [])
    priorities = list(persona_form.get('response_priorities') or [])
    core_goal = str(persona_form.get('core_goal') or '').strip()
    internal_constraints = list(persona_form.get('constraints_internal') or [])
    sarcasm = str(persona_form.get('sarcasm_profile') or 'none')
    clarification = str(persona_form.get('clarification_policy') or '').strip()
    pattern_text = patterns[0] if patterns else 'checks the current situation before choosing a reply'
    priority_text = priorities[0] if priorities else 'answer_substance'
    sarcasm_text = 'can use sarcasm if the situation allows it' if sarcasm in {'medium', 'high'} else 'does not rely on sarcasm as a primary strategy'
    parts = [
        f'{name} is treated as a {identity}.',
        f'Its core goal is {core_goal}.' if core_goal else '',
        f'It first {pattern_text}.',
        f'Then it prioritizes {priority_text.replace("_", " ")}.',
        sarcasm_text + '.',
    ]
    if internal_constraints:
        parts.append(f"Internal constraint: {str(internal_constraints[0]).strip()}.")
    if clarification:
        parts.append(clarification)
    return ' '.join(part.strip() for part in parts if part.strip())


def _validated_decision_explanation(value: Any, *, fallback: str) -> str:
    text = str(value or '').strip()
    return text[:600] if text else fallback[:600]


def _persona_dossier_bucket(fact: str) -> str:
    lowered = normalize_name(fact)
    if any(token in lowered for token in ('role', 'friend', 'ally', 'critic', 'rival', 'mentor', 'comfort', 'provocat')):
        return 'social_roles'
    if any(token in lowered for token in ('habit', 'routine', 'always', 'usually', 'tend to', 'keep doing')):
        return 'habits'
    if any(token in lowered for token in ('work', 'shift', 'hospital', 'clinic', 'resident', 'triage', 'mentor', 'overnight')):
        return 'work_habits'
    if any(token in lowered for token in ('value', 'values', 'protect', 'prefer', 'principle', 'believe')):
        return 'values'
    if any(token in lowered for token in ('conflict', 'torn', 'struggle', 'resent', 'fear becoming', 'can not forgive')):
        return 'conflicts'
    if any(token in lowered for token in ('topic', 'obsessed', 'interested in', 'drawn to', 'care about')):
        return 'topic_affinities'
    if any(token in lowered for token in ('speak', 'voice', 'tone', 'dryly', 'blunt', 'sarcast', 'quietly')):
        return 'speech_tendencies'
    if any(token in lowered for token in ('trust', 'distrust', 'skeptical of', 'earns attention')):
        return 'trust_model'
    if any(token in lowered for token in ('watch', 'notebook', 'anchor', 'father', 'mother', 'sister', 'son', 'daughter', 'child', 'children', 'family')):
        return 'memory_anchors'
    if any(token in lowered for token in ('remember', 'once', 'after', 'during', 'lost', 'learned that')):
        return 'memories'
    return 'personal_history'


def _persona_dossier_bucket_updates(fact: str) -> dict[str, list[str]]:
    normalized_fact = ' '.join(str(fact or '').strip().split())
    if not normalized_fact:
        return {}
    lowered = normalize_name(normalized_fact)
    updates: dict[str, list[str]] = {
        _persona_dossier_bucket(normalized_fact): [normalized_fact],
    }
    family_terms = ('son', 'daughter', 'child', 'children', 'family', 'wife', 'husband', 'mother', 'father', 'sister', 'brother')
    softening_terms = ('softer', 'softer toward', 'more patient', 'gentler', 'less harsh', 'protective', 'more protective', 'tender', 'forgiving')
    vulnerability_terms = ('close people', 'close person', 'vulnerable', 'mistakes', 'family-related', 'family')

    if any(token in lowered for token in family_terms):
        updates.setdefault('memories', []).append(normalized_fact)
        updates.setdefault('personal_history', []).append(normalized_fact)
    if any(token in lowered for token in softening_terms):
        updates.setdefault('emotional_tendencies', []).append('softens in close and vulnerable contexts without losing directness')
    if any(token in lowered for token in family_terms) and any(token in lowered for token in softening_terms + vulnerability_terms):
        updates.setdefault('conflict_behavior', []).append('shows more patience toward close people after family life changed')
        updates.setdefault('reaction_patterns', []).append('becomes more protective and less cold when close people are vulnerable or imperfect')
        updates.setdefault('values', []).append('protect close people without abandoning responsibility or practical judgment')
    return {
        key: _normalize_string_list(value, limit=_memory_limits().persona_example_limit)
        for key, value in updates.items()
        if list(value or [])
    }


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
    structured_persona = _structured_persona_from_payload(
        name,
        container if isinstance(container, dict) else {},
        fallback_examples=examples,
        entity_type=entity_type,
    )
    structured_dict = structured_persona.to_dict()
    raw_persona_form = container.get('persona_form') or raw.get('persona_form') or {}
    persona_form = (
        dict(raw_persona_form)
        if isinstance(raw_persona_form, dict) and raw_persona_form
        else _persona_form_from_structured(structured_persona, entity_type=entity_type)
    )
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
        'structured_persona': structured_dict,
        'persona_form': persona_form,
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
    heuristic_payload = _heuristic_persona_payload(
        name,
        cleaned_excerpts,
        existing_bundle=existing_bundle,
        relations=relations,
    )
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
    merged_payload = _merge_persona_payloads(
        heuristic_payload,
        raw_payload if isinstance(raw_payload, dict) else {},
    )
    validated = _validated_persona_payload(
        name,
        merged_payload,
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
    register: bool = False,
) -> HeadBundle:
    clean = normalize_personality_name(name)
    if register:
        name_reasons = _persona_name_rejection_reasons(name)
        if name_reasons:
            _append_rejected_candidate_log(
                name=name,
                slug=clean,
                reason_codes=name_reasons,
                source=f'spawn_head:{source}',
                evidence={'register': True, 'entity_type': entity_type},
            )
            raise MutationRejectedFailure(
                f'Persona registration for "{name}" was rejected by registry validation.',
                details={'name': clean, 'reason_codes': name_reasons, 'source': source},
            )
    _ensure_head_files(clean, entity_type=entity_type, aliases=aliases, source=source)
    bundle = load_persona(clean)
    assert bundle is not None
    meta = dict(bundle.meta)
    meta['name'] = name
    meta['entity_type'] = entity_type
    meta['aliases'] = merge_aliases(list(meta.get('aliases') or []), list(aliases or []))
    meta['frequency'] = max(int(meta.get('frequency') or 1), 1)
    meta['updated_at'] = _utc_now()
    meta['registry_status'] = _PERSONA_REGISTRY_STATUS_ACTIVE if register else str(meta.get('registry_status') or _PERSONA_REGISTRY_STATUS_DRAFT)
    write_json(_head_file(clean, 'meta.json'), meta)
    if register:
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
    return payload


def derive_persona_label(
    description: str,
    *,
    selected_name: str = '',
    session_persona: str = '',
) -> str:
    explicit = str(selected_name or '').strip()
    if explicit and not looks_like_garbage_label(explicit):
        return explicit
    extracted = extract_explicit_persona_name(description)
    if extracted and not looks_like_garbage_label(extracted):
        return extracted
    fallback_session_persona = str(session_persona or '').strip()
    if fallback_session_persona and not looks_like_garbage_label(fallback_session_persona):
        return fallback_session_persona
    cleaned_description = _strip_persona_creation_prefix(description)
    heuristic = _heuristic_persona_payload('', [cleaned_description or description])
    structured = _structured_persona_from_payload('', heuristic, fallback_examples=[description], entity_type='PERSON')
    label = _suggest_persona_label(structured)
    return label if not looks_like_garbage_label(label) else 'Custom Persona'


def create_persona_from_description(
    description: str,
    *,
    name_hint: str = '',
    session_persona: str = '',
    activate: bool = True,
) -> dict[str, Any]:
    cleaned_description = _strip_persona_creation_prefix(description)
    label = derive_persona_label(cleaned_description or description, selected_name=name_hint, session_persona=session_persona)
    payload = synthesize_persona_from_logs(
        label,
        [cleaned_description or description],
        reason='Persona specification request from dialogue.',
        explicit=True,
    )
    bundle = materialize_persona(label, payload, explicit=True)
    structured = bundle.structured_persona or _structured_persona_from_payload(
        bundle.name,
        dict(payload),
        fallback_examples=[description],
        entity_type=bundle.entity_type,
    )
    return {
        'created': True,
        'activated': bool(activate),
        'persona_name': bundle.name,
        'persona_slug': normalize_personality_name(bundle.name),
        'persona_object': structured.to_dict(),
    }


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
    bundle = load_persona(clean) or spawn_head(clean, entity_type='PERSON', register=True)
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
            'structured_persona': bundle.structured_persona.to_dict() if bundle.structured_persona is not None else {},
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
        if target == 'persona' and (traits['restraint'] >= 0.5 or traits['confidence'] >= 0.5):
            return 'direct_explanatory'
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
    structured = bundle.structured_persona
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
    if structured is not None:
        meta['display_label'] = structured.identity.label
        meta['short_description'] = structured.identity.short_description
        meta['persona_type'] = structured.identity.persona_type
        meta['readiness'] = structured.identity.readiness
        meta['source_text'] = structured.identity.source_text
        meta['hover_text'] = structured.meta.hover_text
        meta['validation_status'] = structured.meta.validation_status
        meta['validation_notes'] = list(structured.meta.validation_notes)
        meta['persona_confidence'] = float(structured.meta.confidence or 0.0)
        meta['persona_tags'] = list(structured.meta.tags)
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
    elif actual.response_style in {'inquisitive', 'formal', 'steady', 'direct_explanatory'}:
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
        bundle = _load_mutable_active_persona(clean)
        if bundle is None:
            return _default_emotion_vector()
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
        bundle = _load_mutable_active_persona(clean)
        if bundle is None:
            return
        payload = load_json(_head_file(clean, 'examples.json'), {'examples': [], 'situation_reactions': []})
        situation_reactions = [dict(item) for item in list(payload.get('situation_reactions') or []) if isinstance(item, dict)]
        reaction_outcome = reaction_policy(bundle, situation)
        raw_reaction = ' '.join(str(reaction or '').split()) if reaction != 0 else ''
        reaction_excerpt = raw_reaction
        if len(reaction_excerpt) > 220:
            reaction_excerpt = reaction_excerpt[:217].rsplit(' ', 1)[0].strip() + '...'
        situation_type = ''
        if isinstance(situation, Situation):
            situation_type = str(situation.type or '').strip()
        elif isinstance(situation, dict):
            situation_type = str(situation.get('type') or '').strip()
        if situation_type in {'neutral_query', 'neutral_statement'}:
            if raw_reaction.endswith('?'):
                reaction_value = f'response_style={reaction_outcome.response_style}; asks_for_clarification_when_grounding_is_thin'
            else:
                reaction_value = f'response_style={reaction_outcome.response_style}; answers_substance_first_with_persona_grounding'
        elif reaction_excerpt:
            reaction_value = f'response_style={reaction_outcome.response_style}; {reaction_excerpt}'
        else:
            reaction_value = f'response_style={reaction_outcome.response_style}'
        entry = {'situation': situation_summary(situation), 'reaction': reaction_value}
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


def record_persona_dossier_fact(name: str, fact: str) -> HeadBundle | None:
    clean = normalize_personality_name(name)
    normalized_fact = ' '.join(str(fact or '').strip().split())
    if not clean or not normalized_fact:
        return load_persona(clean) if clean else None

    def _apply() -> HeadBundle | None:
        bundle = _load_mutable_active_persona(clean)
        if bundle is None:
            return None
        payload = load_json(_head_file(clean, 'examples.json'), {'examples': [], 'situation_reactions': []})
        updated_examples = _merge_examples(
            list(payload.get('examples') or bundle.examples),
            [f'User-provided fact: {normalized_fact}'],
        )
        updated_reactions = [
            dict(item)
            for item in list(payload.get('situation_reactions') or bundle.situation_reactions)
            if isinstance(item, dict)
        ]
        write_json(
            _head_file(clean, 'examples.json'),
            {
                'examples': list(updated_examples),
                'situation_reactions': updated_reactions,
            },
        )

        persona_form = dict(bundle.persona_form or {})
        for bucket, values in _persona_dossier_bucket_updates(normalized_fact).items():
            existing_values = [str(item).strip() for item in list(persona_form.get(bucket) or []) if str(item).strip()]
            merged_values = list(existing_values)
            for value in reversed(list(values or [])):
                merged_values = _prepend_unique_limited(
                    merged_values,
                    value,
                    limit=_memory_limits().persona_example_limit,
                )
            persona_form[bucket] = merged_values
        write_json(_head_file(clean, 'persona_form.json'), persona_form)

        learned = PersonaLearnedPatterns(
            examples=list(updated_examples),
            situation_reactions=list(updated_reactions),
            log_tuples=_merge_log_tuples(
                list(bundle.log_tuples),
                _build_log_tuples(list(updated_examples), list(updated_reactions)),
            ),
            persona_form=dict(persona_form),
            decision_explanation=bundle.decision_explanation,
            learned_traits=list(bundle.learned_patterns.learned_traits) if bundle.learned_patterns is not None else [],
            revision=int((bundle.learned_patterns.revision if bundle.learned_patterns is not None else bundle.meta.get('learned_revision') or 1)),
            updated_at=_utc_now(),
        )
        write_json(_head_file(clean, 'log_tuples.json'), {'items': list(learned.log_tuples)})
        _write_layered_state(clean, learned=learned, reason='record_persona_dossier_fact')
        baseline = bundle.baseline_definition
        if baseline is not None:
            GraphStore().sync_head(
                name=baseline.name,
                folder=str(_head_dir(clean)),
                entity_type=baseline.entity_type,
                aliases=list(baseline.aliases),
                description=(baseline.knowledge.splitlines()[0] if baseline.knowledge else f'Persona head for {baseline.name}.'),
                facts=list(updated_examples)[:8],
                knowledge=baseline.knowledge,
                relations=list(baseline.relations),
            )
        _sync_local_graph(clean)
        _sync_meta_summary(clean)
        return load_persona(clean)

    return _execute_persona_mutation(
        clean,
        reason='record_persona_dossier_fact',
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
        persona_form = dict(profile.persona_form or {})
    else:
        name = str(profile.get('name') or 'unknown_head')
        entity_type = str(profile.get('entity_type') or 'CONCEPT')
        traits = _normalize_string_list(list(profile.get('traits') or []), limit=_memory_limits().persona_trait_limit)
        relations = _validate_relations(list(profile.get('relations') or []))
        examples = _normalize_string_list(list(profile.get('examples') or []), limit=_memory_limits().persona_example_limit)
        emotion_vector = _normalized_emotion_vector(profile.get('emotion_vector') or profile)
        folder = str(profile.get('folder') or _head_dir(name))
        persona_form = dict(profile.get('persona_form') or {})
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
            'context': {'source': 'persona_graph', 'persona_name': name, 'category': 'identity'},
        }
    ]
    edges: list[dict[str, Any]] = []

    def _append_node(node_id: str, node_name: str, *, importance: float, confidence: float, category: str, facts: list[str] | None = None) -> None:
        nodes.append(
            {
                'id': node_id,
                'name': node_name,
                'type': 'CONCEPT',
                'importance': importance,
                'confidence': confidence,
                'frequency': 1,
                'facts': list(facts or [])[:6],
                'context': {'source': 'persona_graph', 'persona_name': name, 'category': category},
            }
        )

    def _append_edge(from_id: str, to_id: str, relation_type: str, *, weight: float = 0.78, confidence: float = 0.8) -> None:
        edges.append({'from': from_id, 'to': to_id, 'type': relation_type, 'weight': weight, 'confidence': confidence})

    for trait in traits:
        trait_id = f'trait:{slug}:{normalize_personality_name(trait)}'
        _append_node(trait_id, trait, importance=0.4, confidence=0.8, category='trait')
        _append_edge(root_id, trait_id, 'HAS_TRAIT', weight=0.9, confidence=0.85)
    for relation in relations:
        target = str(relation.get('target') or relation.get('to') or '').strip()
        relation_type = str(relation.get('type') or 'RELATED_TO').strip().upper()
        if not target:
            continue
        target_id = f'concept:{normalize_personality_name(target)}'
        _append_node(target_id, target, importance=0.3, confidence=0.7, category='relation_target')
        _append_edge(root_id, target_id, relation_type, weight=0.8, confidence=0.8)
    for example in examples[:6]:
        example_id = f'example:{slug}:{normalize_personality_name(example)[:32]}'
        _append_node(example_id, example, importance=0.2, confidence=0.6, category='example')
        _append_edge(root_id, example_id, 'HAS_EXAMPLE', weight=0.5, confidence=0.6)

    structural_categories = (
        ('social_roles', 'role', 'CAN_PLAY_ROLE', 0.46),
        ('habits', 'habit', 'HAS_HABIT', 0.42),
        ('values', 'value', 'VALUES', 0.5),
        ('conflicts', 'conflict', 'HAS_CONFLICT', 0.47),
        ('topic_affinities', 'topic', 'AFFINITY_FOR', 0.4),
        ('speech_tendencies', 'speech', 'SPEAKS_WITH', 0.4),
        ('memories', 'memory', 'REMEMBERS', 0.38),
        ('reaction_patterns', 'reaction', 'USES_REACTION_PATTERN', 0.44),
    )
    category_nodes: dict[str, list[str]] = {}
    for form_key, prefix, relation_type, importance in structural_categories:
        items = [str(item).strip() for item in list(persona_form.get(form_key) or []) if str(item).strip()]
        local_ids: list[str] = []
        for item in items[:12]:
            node_id = f'{prefix}:{slug}:{normalize_personality_name(item)[:36]}'
            _append_node(node_id, item, importance=importance, confidence=0.82, category=form_key)
            _append_edge(root_id, node_id, relation_type, weight=0.84, confidence=0.82)
            local_ids.append(node_id)
        category_nodes[form_key] = local_ids

    for role_id in category_nodes.get('social_roles', []):
        for habit_id in category_nodes.get('habits', [])[:4]:
            _append_edge(role_id, habit_id, 'SUPPORTED_BY_HABIT', weight=0.62, confidence=0.74)
        for speech_id in category_nodes.get('speech_tendencies', [])[:4]:
            _append_edge(role_id, speech_id, 'COLORED_BY_SPEECH', weight=0.58, confidence=0.72)
        for reaction_id in category_nodes.get('reaction_patterns', [])[:4]:
            _append_edge(role_id, reaction_id, 'EXPRESSED_AS', weight=0.61, confidence=0.74)
    for value_id in category_nodes.get('values', []):
        for conflict_id in category_nodes.get('conflicts', [])[:4]:
            _append_edge(value_id, conflict_id, 'IN_TENSION_WITH', weight=0.66, confidence=0.76)
    for topic_id in category_nodes.get('topic_affinities', []):
        for memory_id in category_nodes.get('memories', [])[:4]:
            _append_edge(memory_id, topic_id, 'SHAPES_INTEREST_IN', weight=0.57, confidence=0.72)
    return {'nodes': nodes, 'edges': edges}


def explain_persona_graph(name: str) -> PersonaGraphExplanation | None:
    bundle = load_persona(name)
    if bundle is None:
        return None
    graph = load_persona_graph(name)
    nodes = [dict(item) for item in list(graph.get('nodes') or []) if isinstance(item, dict)]
    edges = [dict(item) for item in list(graph.get('edges') or []) if isinstance(item, dict)]
    if not nodes:
        return PersonaGraphExplanation(persona_name=bundle.name, summary='No local persona graph is available yet.')
    degree: Counter[str] = Counter()
    for edge in edges:
        degree[str(edge.get('from') or '')] += 1
        degree[str(edge.get('to') or '')] += 1
    ranked_nodes = sorted(
        nodes,
        key=lambda item: (
            -degree.get(str(item.get('id') or ''), 0),
            -float(item.get('importance') or 0.0),
            str(item.get('name') or ''),
        ),
    )
    central_nodes = [str(item.get('name') or '') for item in ranked_nodes[:6] if str(item.get('name') or '').strip()]
    peripheral_nodes = [
        str(item.get('name') or '')
        for item in sorted(nodes, key=lambda item: (degree.get(str(item.get('id') or ''), 0), float(item.get('importance') or 0.0), str(item.get('name') or '')))[:6]
        if str(item.get('name') or '').strip() and str(item.get('name') or '') not in central_nodes
    ]
    conflict_nodes = [
        str(item.get('name') or '')
        for item in nodes
        if str(dict(item.get('context') or {}).get('category') or '') == 'conflicts'
    ][:6]
    node_map = {str(item.get('id') or ''): str(item.get('name') or item.get('id') or '') for item in nodes}
    causal_links = [
        f"{node_map.get(str(edge.get('from') or ''), str(edge.get('from') or ''))} {str(edge.get('type') or '').replace('_', ' ').lower()} {node_map.get(str(edge.get('to') or ''), str(edge.get('to') or ''))}"
        for edge in edges
        if str(edge.get('type') or '') in {'SUPPORTED_BY_HABIT', 'COLORED_BY_SPEECH', 'EXPRESSED_AS', 'IN_TENSION_WITH', 'SHAPES_INTEREST_IN', 'CAN_PLAY_ROLE', 'USES_REACTION_PATTERN'}
    ][:10]
    summary_parts = [
        f'{bundle.name} is represented as a social personality graph rather than a flat prompt.',
        f"Central structures are {', '.join(central_nodes[:4])}." if central_nodes else '',
        f"Persistent tensions appear around {', '.join(conflict_nodes[:4])}." if conflict_nodes else '',
        'Roles, habits, values, memories, speech tendencies, and reaction patterns are linked so that behavior can be grounded in structure.'
    ]
    return PersonaGraphExplanation(
        persona_name=bundle.name,
        summary=' '.join(part for part in summary_parts if part).strip(),
        central_nodes=central_nodes,
        peripheral_nodes=peripheral_nodes,
        conflict_nodes=conflict_nodes,
        causal_links=causal_links,
    )


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
    preview_bundle = load_persona(clean)
    preview_structured = _structured_persona_from_payload(
        normalized['name'],
        dict(normalized.get('structured_persona') or payload or {}),
        fallback_examples=list(normalized.get('examples') or []),
        entity_type=str(normalized.get('entity_type') or (preview_bundle.entity_type if preview_bundle is not None else 'PERSON')),
    )
    preview_form = _validated_persona_form(
        normalized.get('persona_form') or _persona_form_from_structured(preview_structured, entity_type=str(normalized.get('entity_type') or 'PERSON')),
        fallback=_default_persona_form(
            normalized['name'],
            entity_type=str(normalized.get('entity_type') or (preview_bundle.entity_type if preview_bundle is not None else 'PERSON')),
            traits=list(normalized.get('traits') or (preview_bundle.traits if preview_bundle is not None else [])),
            relations=list(normalized.get('relations') or (preview_bundle.relations if preview_bundle is not None else [])),
            examples=list(normalized.get('examples') or (preview_bundle.examples if preview_bundle is not None else [])),
            log_tuples=list(normalized.get('log_tuples') or (preview_bundle.log_tuples if preview_bundle is not None else [])),
            existing_form=dict(preview_bundle.persona_form) if preview_bundle is not None else {},
        ),
    )
    preview_decision = _validated_decision_explanation(
        normalized.get('decision_explanation'),
        fallback=_default_decision_explanation(normalized['name'], preview_form),
    )
    validation = validate_persona_candidate(
        normalized['name'],
        entity_type=str(normalized.get('entity_type') or 'PERSON'),
        traits=list(normalized.get('traits') or []),
        relations=list(normalized.get('relations') or []),
        examples=list(normalized.get('examples') or []),
        knowledge=str(normalized.get('knowledge') or '').strip(),
        persona_form=preview_form,
        decision_explanation=preview_decision,
        structured_persona=preview_structured,
        explicit=bool(explicit),
        source='materialize',
    )
    if not validation.get('ok'):
        archive_path = _quarantine_persona_directory(
            normalized['name'],
            reason_codes=list(validation.get('reason_codes') or []),
            source='materialize_rejected',
            evidence=dict(validation.get('evidence') or {}),
        )
        if clean in _load_index():
            _save_index([item for item in _load_index() if item != clean])
        raise MutationRejectedFailure(
            f'Persona candidate {normalized["name"]} was rejected by validation.',
            details={
                'name': normalized['name'],
                'slug': clean,
                'reason_codes': list(validation.get('reason_codes') or []),
                'evidence': dict(validation.get('evidence') or {}),
                'archive_path': archive_path,
            },
        )

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
            register=True,
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
        structured_persona = _structured_persona_from_payload(
            normalized['name'],
            dict(normalized.get('structured_persona') or {}),
            fallback_examples=list(normalized['examples']),
            entity_type=baseline_entity_type,
        )
        structured_persona.meta.validation_status = str(validation.get('validation_status') or structured_persona.meta.validation_status)  # type: ignore[assignment]
        structured_persona.meta.validation_notes = _normalize_string_list(
            list(structured_persona.meta.validation_notes) + list(validation.get('reason_codes') or []),
            limit=10,
        )
        if not structured_persona.identity.label.strip() or looks_like_garbage_label(structured_persona.identity.label):
            structured_persona.identity.label = normalized['name']
        if not structured_persona.identity.short_description.strip():
            structured_persona.identity.short_description = _default_short_description(structured_persona)
        if not structured_persona.meta.hover_text.strip():
            structured_persona.meta.hover_text = _default_hover_text(structured_persona)
        write_json(_structured_persona_path(clean), structured_persona.to_dict())
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
        meta['registry_status'] = _PERSONA_REGISTRY_STATUS_ACTIVE
        meta['display_label'] = structured_persona.identity.label
        meta['short_description'] = structured_persona.identity.short_description
        meta['persona_type'] = structured_persona.identity.persona_type
        meta['readiness'] = structured_persona.identity.readiness
        meta['hover_text'] = structured_persona.meta.hover_text
        meta['validation_status'] = structured_persona.meta.validation_status
        meta['validation_notes'] = list(structured_persona.meta.validation_notes)
        meta['persona_confidence'] = float(structured_persona.meta.confidence or 0.0)
        meta['persona_tags'] = list(structured_persona.meta.tags)
        meta['source_text'] = structured_persona.identity.source_text
        write_json(_head_file(clean, 'meta.json'), meta)
        _save_index(_load_index() + [clean])

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
    cleaned_examples = [str(item).strip() for item in list(examples or []) if str(item).strip()]
    bundle = _load_mutable_active_persona(clean)
    if bundle is None:
        synthesized = synthesize_persona_from_logs(
            name,
            cleaned_examples,
            reason='Session examples and repeated behavior patterns.',
            explicit=True,
            relations=relations,
        )
        return materialize_persona(name, synthesized, explicit=True)
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
        proposal_path = personality_proposal_path(name)
        try:
            payload = synthesize_persona_from_logs(
                name,
                [excerpt] if excerpt else [],
                reason=str(proposal.get('reason') or ''),
                explicit=True,
            )
            bundle = materialize_persona(name, payload, explicit=True)
        except MutationRejectedFailure as exc:
            if proposal_path.exists():
                proposal_path.unlink()
            continue
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
