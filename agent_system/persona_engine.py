from __future__ import annotations

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
from .models import EMOTION_KEYS, HEAD_ENTITY_TYPES, HeadBundle, PersonaSystemModel, ReactionOutcome, Situation
from .prompt_builder import build_persona_profile_prompt
from .runtime_config import get_runtime_config
from .situation_engine import situation_summary


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _head_dir(name: str) -> Path:
    return heads_dir() / normalize_personality_name(name)


def _head_file(name: str, filename: str) -> Path:
    return _head_dir(name) / filename


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
    return rows[:12]


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
            'created_at': _utc_now(),
            'updated_at': _utc_now(),
            'source': source,
            'aliases': list(aliases or []),
        },
        'local_graph.json': {'nodes': [], 'edges': []},
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
    return HeadBundle(
        name=str(meta_payload.get('name') or clean),
        folder=str(_head_dir(clean)),
        entity_type=str(meta_payload.get('entity_type') or traits_payload.get('entity_type') or 'CONCEPT'),
        traits=_normalize_string_list(list(traits_payload.get('traits') or []), limit=20),
        relations=_validate_relations(list(relations_payload.get('relations') or [])),
        examples=_normalize_string_list(list(examples_payload.get('examples') or []), limit=24),
        situation_reactions=[dict(item) for item in list(examples_payload.get('situation_reactions') or []) if isinstance(item, dict)],
        knowledge=str(knowledge or '').strip(),
        emotion_vector=_normalized_emotion_vector(emotion_payload),
        meta=dict(meta_payload),
        log_tuples=[dict(item) for item in list(log_payload.get('items') or []) if isinstance(item, dict)],
        persona_form=dict(persona_form_payload) if isinstance(persona_form_payload, dict) else {},
        decision_explanation=str(decision_explanation or '').strip(),
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
    return _normalize_string_list(existing + new_examples, limit=24)


def _render_tuple_key(parts: tuple[str, ...]) -> str:
    return ' | '.join(str(part).strip() for part in parts if str(part).strip())


def _build_log_tuples(
    examples: list[str],
    situation_reactions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
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
    return rows[:24]


def _merge_log_tuples(existing: list[dict[str, Any]], new_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    return ordered[:24]


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
    container = payload if isinstance(payload, dict) else {}
    raw = dict(container.get('persona_payload') or container) if isinstance(container, dict) else {}
    entity_type = str(raw.get('entity_type') or raw.get('type') or 'PERSON').strip().upper()
    entity_type = _normalized_head_entity_type(entity_type, explicit=explicit)
    traits = _normalize_string_list(list(raw.get('traits') or []), limit=20)
    aliases = merge_aliases(list(raw.get('aliases') or []))
    examples = _merge_examples(
        [str(item).strip() for item in list(fallback_examples or []) if str(item).strip()],
        [str(item).strip() for item in list(raw.get('examples') or []) if str(item).strip()],
    )
    relations = _validate_relations(list(raw.get('relations') or []))
    knowledge = str(raw.get('knowledge') or '').strip()[:4000]
    if not knowledge and examples:
        knowledge = '\n'.join(f'- {item}' for item in examples[:8])
    return {
        'name': str(raw.get('name') or name).strip() or name,
        'entity_type': entity_type,
        'traits': traits or _detect_traits(examples),
        'aliases': aliases,
        'examples': examples,
        'relations': relations,
        'situation_reactions': [dict(item) for item in list(raw.get('situation_reactions') or []) if isinstance(item, dict)][:20],
        'emotion_vector': _normalized_emotion_vector(raw.get('emotion_vector') or raw),
        'knowledge': knowledge,
        'log_tuples': [dict(item) for item in list(container.get('log_tuples') or raw.get('log_tuples') or []) if isinstance(item, dict)][:24],
        'persona_form': dict(container.get('persona_form') or raw.get('persona_form') or {})
        if isinstance(container.get('persona_form') or raw.get('persona_form') or {}, dict)
        else {},
        'decision_explanation': str(container.get('decision_explanation') or raw.get('decision_explanation') or '').strip()[:600],
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


def _update_meta(name: str, *, entity_type: str | None = None, aliases: list[str] | None = None, importance_delta: float = 0.0) -> dict[str, Any]:
    bundle = load_persona(name) or spawn_head(name, entity_type=entity_type or 'CONCEPT')
    meta = dict(bundle.meta)
    if entity_type:
        meta['entity_type'] = entity_type
    meta['aliases'] = merge_aliases(list(meta.get('aliases') or []), list(aliases or []))
    meta['frequency'] = int(meta.get('frequency') or 1) + 1
    meta['importance'] = round(min(1.0, max(0.05, float(meta.get('importance') or 0.5) + importance_delta)), 4)
    meta['updated_at'] = _utc_now()
    write_json(_head_file(name, 'meta.json'), meta)
    return meta


def adjust_emotion_vector(name: str, situation: Situation | dict[str, Any] | None) -> dict[str, float]:
    bundle = load_persona(name) or spawn_head(name, entity_type='PERSON')
    next_state, _ = evolve_emotion_state(bundle, situation)
    write_json(_head_file(name, 'emotion_vector.json'), next_state)
    return next_state


def record_situation_reaction(name: str, situation: str | Situation | dict[str, Any], reaction: str | int) -> None:
    if not str(name or '').strip():
        return
    bundle = load_persona(name) or spawn_head(name, entity_type='PERSON')
    payload = load_json(_head_file(name, 'examples.json'), {'examples': [], 'situation_reactions': []})
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
    payload['situation_reactions'] = deduped[:20]
    payload['examples'] = list(payload.get('examples') or bundle.examples)
    write_json(_head_file(name, 'examples.json'), payload)


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
        traits = _normalize_string_list(list(profile.get('traits') or []), limit=20)
        relations = _validate_relations(list(profile.get('relations') or []))
        examples = _normalize_string_list(list(profile.get('examples') or []), limit=24)
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


def materialize_persona(name: str, payload: dict[str, Any], *, explicit: bool = False) -> HeadBundle:
    normalized = _validated_persona_payload(name, payload, fallback_examples=list(payload.get('examples') or []), explicit=explicit)
    clean = normalize_personality_name(normalized['name'])
    bundle = spawn_head(
        normalized['name'],
        entity_type=normalized['entity_type'],
        aliases=list(normalized.get('aliases') or []),
        source='materialize',
        sync_graph=False,
    )
    write_json(_head_file(clean, 'traits.json'), {'traits': list(normalized['traits']), 'entity_type': normalized['entity_type']})
    write_json(_head_file(clean, 'relations.json'), {'aliases': merge_aliases(list(normalized.get('aliases') or []), list(bundle.meta.get('aliases') or [])), 'relations': list(normalized['relations'])})
    write_json(
        _head_file(clean, 'examples.json'),
        {
            'examples': _merge_examples(bundle.examples, list(normalized['examples'])),
            'situation_reactions': [dict(item) for item in list(normalized['situation_reactions']) if isinstance(item, dict)],
        },
    )
    write_json(_head_file(clean, 'emotion_vector.json'), normalized['emotion_vector'])
    write_text(_head_file(clean, 'knowledge.txt'), normalized['knowledge'])
    provided_log_tuples = list(normalized.get('log_tuples') or [])
    if provided_log_tuples:
        merged_log_tuples = _merge_log_tuples([], provided_log_tuples)
    else:
        merged_log_tuples = _merge_log_tuples(
            list(bundle.log_tuples),
            _build_log_tuples(list(normalized['examples']), list(normalized['situation_reactions'])),
        )
    persona_form = _validated_persona_form(
        normalized.get('persona_form'),
        fallback=_default_persona_form(
            normalized['name'],
            entity_type=normalized['entity_type'],
            traits=list(normalized['traits']),
            relations=list(normalized['relations']),
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
    meta = _update_meta(clean, entity_type=normalized['entity_type'], aliases=list(normalized.get('aliases') or []), importance_delta=0.08)
    write_json(_head_file(clean, 'meta.json'), meta)

    GraphStore().sync_head(
        name=normalized['name'],
        folder=str(_head_dir(clean)),
        entity_type=normalized['entity_type'],
        aliases=list(meta.get('aliases') or []),
        description=(normalized['knowledge'].splitlines()[0] if normalized['knowledge'] else f'Persona head for {normalized["name"]}.'),
        facts=list(normalized['examples'])[:8],
        knowledge=normalized['knowledge'],
        relations=list(normalized['relations']),
    )
    _sync_local_graph(clean)
    updated = load_persona(clean)
    assert updated is not None
    return updated


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
    synthesized['entity_type'] = str(synthesized.get('entity_type') or bundle.entity_type or 'PERSON')
    synthesized['traits'] = list(dict.fromkeys(bundle.traits + list(synthesized.get('traits') or [])))
    synthesized['relations'] = _validate_relations(list(bundle.relations) + list(synthesized.get('relations') or []) + list(relations or []))
    synthesized['examples'] = merged_examples
    synthesized['situation_reactions'] = [dict(item) for item in list(bundle.situation_reactions) if isinstance(item, dict)]
    synthesized['emotion_vector'] = synthesized.get('emotion_vector') or bundle.emotion_vector
    knowledge_lines = [str(bundle.knowledge or '').strip(), str(synthesized.get('knowledge') or '').strip()]
    knowledge_lines.extend(f'- {example}' for example in merged_examples[:8])
    synthesized['knowledge'] = '\n'.join(line for line in knowledge_lines if line).strip()
    synthesized['aliases'] = bundle.meta.get('aliases') or []
    synthesized['log_tuples'] = _merge_log_tuples(list(bundle.log_tuples), list(synthesized.get('log_tuples') or []))
    return materialize_persona(bundle.name, synthesized, explicit=True)


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
