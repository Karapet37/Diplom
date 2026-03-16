from __future__ import annotations

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
from .llm import call_json_model
from .models import EMOTION_KEYS, HEAD_ENTITY_TYPES, HeadBundle
from .prompt_builder import build_persona_profile_prompt


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
    meta_payload = load_json(_head_file(clean, 'meta.json'), {})
    knowledge = _head_file(clean, 'knowledge.txt').read_text(encoding='utf-8') if _head_file(clean, 'knowledge.txt').exists() else ''
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


def _validated_persona_payload(
    name: str,
    payload: Any,
    *,
    fallback_examples: list[str] | None = None,
    explicit: bool = False,
) -> dict[str, Any]:
    raw = payload if isinstance(payload, dict) else {}
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
    }


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


def adjust_emotion_vector(name: str, cues: dict[str, float]) -> dict[str, float]:
    bundle = load_persona(name) or spawn_head(name, entity_type='PERSON')
    vector = dict(bundle.emotion_vector)
    vector['anger'] = min(1.0, max(0.0, vector['anger'] + (0.18 if cues.get('contains_insult') else -0.02)))
    vector['fear'] = min(1.0, max(0.0, vector['fear'] + (0.12 if cues.get('contains_fear') else -0.01)))
    vector['curiosity'] = min(1.0, max(0.0, vector['curiosity'] + (0.08 if cues.get('contains_question') else -0.01)))
    vector['confidence'] = min(1.0, max(0.0, vector['confidence'] - (0.04 if cues.get('contains_fear') else 0.02)))
    vector['empathy'] = min(1.0, max(0.0, vector['empathy'] + (0.12 if cues.get('contains_empathy') else -0.01)))
    write_json(_head_file(name, 'emotion_vector.json'), {key: round(value, 4) for key, value in vector.items()})
    return vector


def record_situation_reaction(name: str, situation: str, reaction: str | int) -> None:
    if not str(name or '').strip():
        return
    bundle = load_persona(name) or spawn_head(name, entity_type='PERSON')
    payload = load_json(_head_file(name, 'examples.json'), {'examples': [], 'situation_reactions': []})
    situation_reactions = [dict(item) for item in list(payload.get('situation_reactions') or []) if isinstance(item, dict)]
    entry = {'situation': str(situation or '').strip(), 'reaction': reaction if reaction == 0 else str(reaction or '').strip()}
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
    merged_examples = _merge_examples(bundle.examples, [str(item).strip() for item in examples if str(item).strip()])
    merged_traits = list(dict.fromkeys(bundle.traits + _detect_traits(merged_examples)))
    merged_relations = _validate_relations(list(bundle.relations) + list(relations or []))
    knowledge_lines = [bundle.knowledge] if bundle.knowledge else []
    knowledge_lines.extend(f'- {example}' for example in merged_examples[:8])
    return materialize_persona(
        bundle.name,
        {
            'entity_type': bundle.entity_type,
            'traits': merged_traits,
            'relations': merged_relations,
            'examples': merged_examples,
            'situation_reactions': bundle.situation_reactions,
            'emotion_vector': bundle.emotion_vector,
            'knowledge': '\n'.join(line for line in knowledge_lines if line).strip(),
            'aliases': bundle.meta.get('aliases') or [],
        },
        explicit=True,
    )


def process_persona_proposals() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for proposal in list_persona_proposals():
        name = str(proposal.get('name') or '').strip()
        excerpt = str(proposal.get('excerpt') or '').strip()
        if not name:
            continue
        raw_payload = call_json_model(build_persona_profile_prompt(name, [excerpt] if excerpt else [], reason=str(proposal.get('reason') or '')))
        payload = _validated_persona_payload(
            name,
            raw_payload,
            fallback_examples=[excerpt] if excerpt else [],
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


def relevant_reactions(name: str, situation: str, *, limit: int = 3) -> list[dict[str, Any]]:
    bundle = load_persona(name)
    if bundle is None:
        return []
    target = normalize_name(situation)
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
