from __future__ import annotations

from typing import Any

from .classifier_forest import DEFAULT_CLASSIFIER
from .duplicate_resolver import normalize_name
from .feature_extractor import extract_features
from .graph_localizer import ensure_node_localization
from .graph_store import GraphStore
from .llm import call_json_model_for_role
from .message_analyzer import analyze_message
from .models import GraphModelToolPolicy, MessageEntity
from .prompt_builder import build_node_rethink_prompt, render_graph_context
from .runtime_config import get_runtime_config

DEFAULT_RETHINK_LIMIT = 6

_ABSTRACT_CONCEPT_NAMES = {
    'aristocracy',
    'cultural icon',
    'ecosystem',
    'fear',
    'hunting',
    'human',
    'identity',
    'immortal beings',
    'language',
    'life',
    'mythology',
    'politics',
    'power',
    'predation',
    'social class',
    'sociology',
    'supernatural',
    'survival',
    'vampire',
}

_DESCRIPTION_HINTS_FOR_CONCEPT = {
    'biological',
    'condition',
    'concept',
    'culture',
    'emotion',
    'energy',
    'environment',
    'myth',
    'phenomenon',
    'political',
    'process',
    'relation',
    'social',
    'system',
}

_ALLOWED_LINK_ROLES: dict[str, tuple[str, str]] = {
    'related_to': ('RELATED_TO', 'out'),
    'has_dimension': ('HAS_DIMENSION', 'out'),
    'supports': ('SUPPORTS', 'out'),
    'makes_possible': ('MAKES_POSSIBLE', 'out'),
    'depends_on': ('DEPENDS_ON', 'out'),
    'part_of': ('PART_OF', 'out'),
    'influences': ('INFLUENCES', 'out'),
    'warms': ('WARMS', 'out'),
    'regulates': ('REGULATES', 'out'),
    'is_a': ('IS_A', 'out'),
    'symbolizes': ('SYMBOLIZES', 'out'),
    'affects': ('AFFECTS', 'out'),
}

_ABSTRACT_PERSON_RETYPE_NAMES = {
    'aristocracy',
    'cultural icon',
    'ecosystem',
    'fear',
    'hunting',
    'mythology',
    'politics',
    'power',
    'social class',
    'supernatural',
    'survival',
}

_CANONICAL_REPAIRS: dict[str, dict[str, Any]] = {
    'vampire': {
        'description': 'Vampire is a mythological and literary concept describing a supernatural being that sustains itself by feeding on human life or blood.',
        'translation_line': 'Vampire: вампир, արնախում',
        'facts': [
            'The vampire figure belongs to mythology, folklore, and fiction.',
            'Vampire narratives often combine fear, immortality, and predatory dependence on humans.',
        ],
    },
    'predation': {
        'description': 'Predation is a biological and behavioral relation in which one organism survives by hunting, killing, or consuming another.',
        'translation_line': 'Predation: хищничество, գիշատչություն',
        'facts': [
            'Predation shapes food chains and ecosystem balance.',
            'Predation differs from general aggression because it is tied to survival and feeding.',
        ],
    },
    'immortal beings': {
        'description': 'Immortal beings is a mythological concept for creatures imagined as exempt from ordinary aging and death.',
        'translation_line': 'Immortal beings: бессмертные существа, անմահ էակներ',
        'facts': [
            'Immortality motifs usually belong to mythology, fantasy, or religion rather than ordinary biology.',
            'Immortal beings are often associated with supernatural power and symbolic fear.',
        ],
    },
}


def _clip_chars(text: str, max_chars: int) -> str:
    raw = str(text or '').strip()
    if not raw or max_chars <= 0:
        return ''
    return raw[:max_chars].strip() if len(raw) > max_chars else raw


def _default_rethink_context_budget() -> int:
    return get_runtime_config().settings.rethink_context_budget


def _tool_policy(policy: GraphModelToolPolicy | None, *, context_budget: int) -> GraphModelToolPolicy:
    if policy is not None:
        return policy
    runtime = get_runtime_config()
    return GraphModelToolPolicy(
        context_budget=int(context_budget or _default_rethink_context_budget()),
        reconstruction_role=runtime.roles.rethink,
        translation_role=runtime.roles.translation,
    )


def _string_list(value: Any, *, limit: int = 8) -> list[str]:
    items: list[str] = []
    for item in list(value or []):
        token = str(item or '').strip()
        if token:
            items.append(token)
    return list(dict.fromkeys(items))[:limit]


def _clean_candidate_name(value: Any) -> str:
    return ' '.join(str(value or '').replace('\n', ' ').split()).strip(" \t\r\n'\".,;:!?")


def _looks_fragment_like(name: str) -> bool:
    clean = _clean_candidate_name(name)
    if not clean:
        return True
    if any(marker in clean for marker in (',', ';', ':', '!', '?', '\n')):
        return True
    tokens = [token for token in normalize_name(clean).split() if token]
    if not tokens or len(tokens) > 5 or len(clean) > 64:
        return True
    clause_markers = {
        'is',
        'are',
        'was',
        'were',
        'because',
        'which',
        'who',
        'does',
        'делает',
        'является',
        'который',
        'как',
    }
    if len(tokens) >= 3 and any(token in clause_markers for token in tokens[1:]):
        return True
    return False


def _fallback_node_improvement(node: dict[str, Any], node_view: dict[str, Any]) -> dict[str, Any]:
    facts = _string_list(node.get('facts') or [])
    relation_lines = []
    for edge in list(node_view.get('how_it_acts') or [])[:6]:
        relation_lines.append(
            f"{edge.get('type')} -> {edge.get('to_name') if str(edge.get('from') or '') == str(node.get('id') or '') else edge.get('from_name') or edge.get('from')}"
        )
    plain_explanation = str(node.get('description') or '').strip() or f"{node.get('name')} is a graph concept node."
    return {
        'description': plain_explanation,
        'plain_explanation': plain_explanation,
        'facts': facts,
        'capabilities': _string_list(relation_lines[:4]),
        'mechanisms': [],
        'reinterpretation_form': {
            'who_or_what': f"{node.get('name')} ({node.get('type')})",
            'what_can_it_do': ' | '.join(facts[:4]) or plain_explanation,
            'how_does_it_work': ' | '.join(relation_lines[:4]) or 'Acts through the graph relations currently attached to this node.',
            'why_it_matters': plain_explanation,
            'suggested_links': [],
        },
    }


def _node_context_text(node: dict[str, Any], node_view: dict[str, Any], graph_context: str) -> str:
    description = str(node.get('description') or '').strip()
    translation = str(node.get('translation_line') or '').strip()
    facts = _string_list(node.get('facts') or [], limit=10)
    relations = []
    for edge in list(node_view.get('how_it_acts') or [])[:10]:
        direction = 'out' if str(edge.get('from') or '') == str(node.get('id') or '') else 'in'
        relations.append(f"{direction}: {edge.get('from_name') or edge.get('from')} {edge.get('type')} {edge.get('to_name') or edge.get('to')}")
    blocks = [
        f"Node: {node.get('name')} [{node.get('type')}]",
        f"Translation: {translation}" if translation else '',
        f"Description: {description}" if description else '',
        f"Facts: {' | '.join(facts)}" if facts else '',
        f"Relations: {' | '.join(relations)}" if relations else '',
        'Neighborhood graph:',
        graph_context,
    ]
    return '\n'.join(part for part in blocks if str(part).strip())


def _validated_node_improvement(node: dict[str, Any], payload: dict[str, Any], *, fallback: dict[str, Any]) -> dict[str, Any]:
    raw = dict(payload.get('node_improvement') or payload.get('node_update') or {})
    if not raw:
        return fallback
    description = str(raw.get('description') or fallback.get('description') or node.get('description') or '').strip()
    plain_explanation = str(raw.get('plain_explanation') or fallback.get('plain_explanation') or '').strip()
    facts = _string_list(list(node.get('facts') or []) + list(raw.get('facts') or []), limit=16)
    capabilities = _string_list(raw.get('capabilities') or fallback.get('capabilities') or [], limit=8)
    mechanisms = _string_list(raw.get('mechanisms') or fallback.get('mechanisms') or [], limit=8)
    form = dict(fallback.get('reinterpretation_form') or {})
    form.update({key: value for key, value in dict(raw.get('reinterpretation_form') or {}).items() if value})
    form['suggested_links'] = _string_list(form.get('suggested_links') or raw.get('links') or [], limit=10)
    return {
        'description': description,
        'plain_explanation': plain_explanation or description,
        'facts': facts,
        'capabilities': capabilities,
        'mechanisms': mechanisms,
        'reinterpretation_form': form,
    }


def _normalized_role(value: Any, *, allowed_roles: tuple[str, ...]) -> str:
    token = normalize_name(str(value or '')).replace(' ', '_')
    if token in _ALLOWED_LINK_ROLES and token in set(allowed_roles):
        return token
    return 'related_to'


def _validated_link_suggestions(
    payload: dict[str, Any],
    *,
    fallback: dict[str, Any],
    policy: GraphModelToolPolicy,
) -> list[dict[str, Any]]:
    raw_items = list(payload.get('link_suggestions') or [])
    if not raw_items:
        raw_items = list(dict(payload.get('node_improvement') or payload.get('node_update') or {}).get('link_suggestions') or [])
    if not raw_items:
        raw_items = list(fallback.get('reinterpretation_form', {}).get('suggested_links') or [])
    suggestions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_items[: max(int(policy.max_link_suggestions or 4), 1)]:
        if isinstance(item, dict):
            name = _clean_candidate_name(item.get('name') or item.get('label') or item.get('node') or '')
            if not name or _looks_fragment_like(name):
                continue
            key = normalize_name(name)
            if key in seen:
                continue
            seen.add(key)
            suggestions.append(
                {
                    'name': name,
                    'role': _normalized_role(item.get('role') or item.get('relation_role') or 'related_to', allowed_roles=policy.allowed_link_roles),
                    'why': _clip_chars(str(item.get('why') or '').strip(), 280),
                    'description': _clip_chars(str(item.get('description') or '').strip(), 420),
                    'facts': _string_list(item.get('facts') or [], limit=8),
                    'aliases': _string_list(item.get('aliases') or [], limit=6),
                    'translation_line': str(item.get('translation_line') or '').strip(),
                }
            )
            continue
        name = _clean_candidate_name(item)
        if not name or _looks_fragment_like(name):
            continue
        key = normalize_name(name)
        if key in seen:
            continue
        seen.add(key)
        suggestions.append(
            {
                'name': name,
                'role': 'related_to',
                'why': '',
                'description': '',
                'facts': [],
                'aliases': [],
                'translation_line': '',
            }
        )
    return suggestions


def _deterministic_entity_type(
    *,
    name: str,
    description: str = '',
    aliases: list[str] | None = None,
) -> str:
    normalized = normalize_name(name)
    desc_norm = normalize_name(description)
    if normalized in _ABSTRACT_CONCEPT_NAMES:
        return 'CONCEPT'
    if normalized in {'sunlight', 'daylight', 'solar light'}:
        return 'PHENOMENON'
    if any(marker in desc_norm for marker in _DESCRIPTION_HINTS_FOR_CONCEPT):
        return 'CONCEPT'
    analysis = analyze_message(message=f'{name}. {description}', session_id='rethink_type', explicit_context=' | '.join(list(aliases or [])))
    features = extract_features(MessageEntity(name=name, description=description, aliases=list(aliases or [])), analysis)
    decision = DEFAULT_CLASSIFIER.classify(features)
    entity_type = str(decision.entity_type or 'CONCEPT').upper()
    if entity_type == 'PERSON' and any(marker in desc_norm for marker in _DESCRIPTION_HINTS_FOR_CONCEPT):
        return 'CONCEPT'
    return entity_type if entity_type else 'CONCEPT'


def _relation_from_role(source_id: str, target_id: str, role: str) -> tuple[str, str, str]:
    relation_type, direction = _ALLOWED_LINK_ROLES.get(role, _ALLOWED_LINK_ROLES['related_to'])
    if direction == 'in':
        return target_id, source_id, relation_type
    return source_id, target_id, relation_type


def _context_patch_for_improvement(
    improvement: dict[str, Any],
    suggestions: list[dict[str, Any]],
    *,
    context_budget: int,
) -> dict[str, Any]:
    link_names = [item['name'] for item in suggestions]
    reinterpretation_form = dict(improvement.get('reinterpretation_form') or {})
    reinterpretation_form['suggested_links'] = link_names
    return {
        'source': 'reinterpretation',
        'plain_explanation': improvement.get('plain_explanation') or improvement.get('description') or '',
        'capabilities': list(improvement.get('capabilities') or []),
        'mechanisms': list(improvement.get('mechanisms') or []),
        'reinterpretation_form': reinterpretation_form,
        'link_suggestions': link_names,
        'reflection_context_budget': int(context_budget or _default_rethink_context_budget()),
        'rethink_mode': 'constrained_content_only',
    }


def _build_preview_payload(
    *,
    node: dict[str, Any],
    improvement: dict[str, Any],
    suggestions: list[dict[str, Any]],
    policy: GraphModelToolPolicy,
) -> dict[str, Any]:
    preview_links = []
    for suggestion in suggestions[: max(int(policy.max_link_suggestions or 4), 1)]:
        role = _normalized_role(suggestion.get('role') or 'related_to', allowed_roles=policy.allowed_link_roles)
        relation_type, direction = _ALLOWED_LINK_ROLES.get(role, _ALLOWED_LINK_ROLES['related_to'])
        preview_links.append(
            {
                'name': suggestion.get('name'),
                'role': role,
                'relation_type': relation_type,
                'direction': direction,
                'why': str(suggestion.get('why') or '').strip(),
                'description': str(suggestion.get('description') or '').strip(),
                'facts': list(suggestion.get('facts') or []),
            }
        )
    return {
        'node_id': str(node.get('id') or ''),
        'node_name': str(node.get('name') or ''),
        'description': str(improvement.get('description') or '').strip(),
        'plain_explanation': str(improvement.get('plain_explanation') or '').strip(),
        'facts': list(improvement.get('facts') or []),
        'capabilities': list(improvement.get('capabilities') or []),
        'mechanisms': list(improvement.get('mechanisms') or []),
        'reinterpretation_form': dict(improvement.get('reinterpretation_form') or {}),
        'links': preview_links,
    }


def _apply_node_improvement(
    graph_store: GraphStore,
    *,
    node_id: str,
    improvement: dict[str, Any],
    suggestions: list[dict[str, Any]],
    context_budget: int,
) -> dict[str, Any]:
    return graph_store.patch_node(
        node_id,
        description=str(improvement.get('description') or '').strip(),
        facts=list(improvement.get('facts') or []),
        context_patch=_context_patch_for_improvement(improvement, suggestions, context_budget=context_budget),
    )


def _apply_link_suggestions(
    graph_store: GraphStore,
    *,
    source_node: dict[str, Any],
    suggestions: list[dict[str, Any]],
    policy: GraphModelToolPolicy,
) -> dict[str, Any]:
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    source_id = str(source_node.get('id') or '')
    source_name_key = normalize_name(str(source_node.get('name') or ''))
    max_items = max(min(int(policy.max_new_entities or 4), int(policy.max_link_suggestions or 4)), 1)
    for suggestion in suggestions[:max_items]:
        name = _clean_candidate_name(suggestion.get('name') or '')
        if not name or _looks_fragment_like(name):
            skipped.append({'name': name, 'reason': 'invalid_name'})
            continue
        if normalize_name(name) == source_name_key:
            skipped.append({'name': name, 'reason': 'self_link'})
            continue
        role = _normalized_role(suggestion.get('role') or 'related_to', allowed_roles=policy.allowed_link_roles)
        description = str(suggestion.get('description') or suggestion.get('why') or '').strip()
        facts = _string_list(suggestion.get('facts') or [], limit=8)
        aliases = _string_list(suggestion.get('aliases') or [], limit=6)
        entity_type = _deterministic_entity_type(name=name, description=description, aliases=aliases)
        target = graph_store.upsert_entity(
            name=name,
            entity_type=entity_type,
            aliases=aliases,
            description=description or f'{name} is a concept linked to {source_node.get("name")}.',
            facts=facts,
            confidence=0.72,
            importance=0.66,
            translation_line=str(suggestion.get('translation_line') or '').strip(),
            source='reinterpretation',
            context={
                'source': 'reinterpretation',
                'rethink_origin': str(source_node.get('name') or ''),
                'why': str(suggestion.get('why') or '').strip(),
            },
        )
        target_id = str(target.get('id') or '')
        from_id, to_id, relation_type = _relation_from_role(source_id, target_id, role)
        graph_store.connect_nodes(
            from_id=from_id,
            to_id=to_id,
            relation_type=relation_type,
            weight=0.82,
            confidence=0.82,
            source='reinterpretation',
        )
        applied.append({'name': name, 'node_id': target_id, 'role': role, 'relation_type': relation_type})
    return {'applied': applied, 'skipped': skipped}


def _merge_node_result(report: dict[str, Any], graph_store: GraphStore, *, primary_id: str, secondary_id: str) -> None:
    primary = graph_store.get_node_by_id(primary_id)
    secondary = graph_store.get_node_by_id(secondary_id)
    if primary is None or secondary is None or primary_id == secondary_id:
        return
    result = graph_store.merge_nodes_manual(primary_id=primary_id, secondary_id=secondary_id)
    if result.get('ok'):
        report['merged'] += 1
        report['redirects'][secondary_id] = primary_id


def _patch_concept_node(graph_store: GraphStore, *, name: str, description: str, translation_line: str = '', facts: list[str] | None = None) -> bool:
    node = graph_store.get_node(name)
    if node is None:
        node = graph_store.create_node(
            name=name,
            node_type='CONCEPT',
            description=description,
            facts=list(facts or []),
            translation_line=translation_line,
        )
        return bool(node)
    merged_facts = _string_list(list(node.get('facts') or []) + list(facts or []), limit=16)
    result = graph_store.patch_node(
        str(node.get('id') or ''),
        description=description if not str(node.get('description') or '').strip() else str(node.get('description') or '').strip(),
        facts=merged_facts,
        translation_line=translation_line or str(node.get('translation_line') or ''),
    )
    return bool(result.get('ok'))


def repair_graph_semantics(store: GraphStore | None = None) -> dict[str, Any]:
    graph_store = store or GraphStore()
    report = {
        'ok': True,
        'merged': 0,
        'retyped': 0,
        'patched': 0,
        'redirects': {},
    }

    _merge_node_result(report, graph_store, primary_id='person:dracula', secondary_id='concept:dracula')
    _merge_node_result(report, graph_store, primary_id='concept:life', secondary_id='person:life')

    vampire_patch = _CANONICAL_REPAIRS['vampire']
    if _patch_concept_node(
        graph_store,
        name='Vampire',
        description=str(vampire_patch['description']),
        translation_line=str(vampire_patch['translation_line']),
        facts=list(vampire_patch['facts']),
    ):
        report['patched'] += 1
    _merge_node_result(report, graph_store, primary_id='concept:vampire', secondary_id='fictional_character:vampire')

    for repair_name in _ABSTRACT_PERSON_RETYPE_NAMES:
        node = graph_store.get_node_by_id(f'person:{repair_name.replace(" ", "_")}')
        if node is None or str(node.get('type') or '').upper() != 'PERSON':
            continue
        concept_node = graph_store.create_node(
            name=str(node.get('name') or ''),
            node_type='CONCEPT',
            aliases=list(node.get('aliases') or []),
            description=str(node.get('description') or '').strip(),
            facts=list(node.get('facts') or []),
            translation_line=str(node.get('translation_line') or '').strip(),
        )
        concept_id = str(concept_node.get('id') or '')
        old_id = str(node.get('id') or '')
        if concept_id and concept_id != old_id:
            _merge_node_result(report, graph_store, primary_id=concept_id, secondary_id=old_id)
            report['retyped'] += 1

    for token, payload in _CANONICAL_REPAIRS.items():
        if _patch_concept_node(
            graph_store,
            name=' '.join(part.capitalize() for part in token.split()),
            description=str(payload['description']),
            translation_line=str(payload['translation_line']),
            facts=list(payload['facts']),
        ):
            report['patched'] += 1

    return report


def rethink_node(
    node_id: str,
    *,
    language: str = 'en',
    store: GraphStore | None = None,
    context_budget: int | None = None,
    policy: GraphModelToolPolicy | None = None,
    preview_only: bool = False,
) -> dict[str, Any]:
    graph_store = store or GraphStore()
    resolved_context_budget = int(context_budget or _default_rethink_context_budget())
    tool_policy = _tool_policy(policy, context_budget=resolved_context_budget)
    node = graph_store.get_node_by_id(node_id)
    if node is None:
        return {'ok': False, 'reason': 'node_not_found', 'node_id': node_id}

    node_view = graph_store.answerable_node_view(node_id) or {}
    neighborhood = graph_store.subgraph(str(node.get('name') or ''), limit=8, depth=2)
    graph_context = render_graph_context(list(neighborhood.get('nodes') or []), list(neighborhood.get('edges') or []))
    context_text = _clip_chars(_node_context_text(node, node_view, graph_context), max(resolved_context_budget, 1000))
    prompt = build_node_rethink_prompt(
        node=node,
        node_view=node_view,
        graph_context=context_text,
        context_budget=resolved_context_budget,
        max_new_entities=tool_policy.max_new_entities,
        max_new_links=tool_policy.max_link_suggestions,
        allowed_roles=tool_policy.allowed_link_roles,
    )
    payload = call_json_model_for_role(prompt, role=tool_policy.reconstruction_role) if tool_policy.allow_reconstruction else {}

    fallback = _fallback_node_improvement(node, node_view)
    improvement = _validated_node_improvement(node, payload if isinstance(payload, dict) else {}, fallback=fallback)
    suggestions = _validated_link_suggestions(payload if isinstance(payload, dict) else {}, fallback=fallback, policy=tool_policy)
    preview = _build_preview_payload(node=node, improvement=improvement, suggestions=suggestions, policy=tool_policy)

    if preview_only:
        return {
            'ok': True,
            'preview_only': True,
            'node_id': str(node.get('id') or ''),
            'node_name': node.get('name'),
            'node_update': improvement,
            'preview': preview,
            'context_budget': resolved_context_budget,
        }

    _apply_node_improvement(
        graph_store,
        node_id=str(node.get('id') or ''),
        improvement=improvement,
        suggestions=suggestions,
        context_budget=resolved_context_budget,
    )
    refreshed_source = graph_store.get_node_by_id(str(node.get('id') or '')) or node
    link_report = _apply_link_suggestions(graph_store, source_node=refreshed_source, suggestions=suggestions, policy=tool_policy)
    localization = ensure_node_localization(
        str(node.get('id') or ''),
        language=language,
        store=graph_store,
        policy=tool_policy,
    ) if tool_policy.allow_translation else {'ok': False}

    return {
        'ok': True,
        'preview_only': False,
        'node_id': str(node.get('id') or ''),
        'node_name': node.get('name'),
        'node_update': improvement,
        'preview': preview,
        'applied_links': link_report['applied'],
        'skipped_suggestions': link_report['skipped'],
        'context_budget': resolved_context_budget,
        'localization': localization,
    }


def rethink_graph_nodes(
    *,
    node_ids: list[str] | None = None,
    query: str = '',
    limit: int = DEFAULT_RETHINK_LIMIT,
    context_budget: int | None = None,
    active_mode: bool = False,
    language: str = 'en',
    store: GraphStore | None = None,
    preview_only: bool = False,
) -> dict[str, Any]:
    graph_store = store or GraphStore()
    resolved_context_budget = int(context_budget or _default_rethink_context_budget())
    tool_policy = _tool_policy(None, context_budget=resolved_context_budget)
    repair_report = {'ok': True, 'merged': 0, 'retyped': 0, 'patched': 0, 'redirects': {}}
    if not preview_only:
        repair_report = repair_graph_semantics(graph_store)
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for raw_node_id in list(node_ids or []):
        node_id = str(repair_report.get('redirects', {}).get(raw_node_id, raw_node_id))
        node = graph_store.get_node_by_id(node_id)
        if node is None:
            continue
        token = str(node.get('id') or '')
        if token and token not in seen_ids and not token.startswith('summary:'):
            seen_ids.add(token)
            candidates.append(node)

    if not candidates and str(query or '').strip():
        for node in graph_store.search_nodes(query, limit=max(int(limit or DEFAULT_RETHINK_LIMIT), 1)):
            token = str(node.get('id') or '')
            if token and token not in seen_ids and not token.startswith('summary:'):
                seen_ids.add(token)
                candidates.append(node)

    if active_mode and len(candidates) < max(int(limit or DEFAULT_RETHINK_LIMIT), 1):
        ranked = sorted(graph_store.load_nodes(), key=lambda item: float(item.get('importance') or 0.0), reverse=True)
        for node in ranked:
            token = str(node.get('id') or '')
            if not token or token in seen_ids or token.startswith('summary:'):
                continue
            seen_ids.add(token)
            candidates.append(node)
            if len(candidates) >= max(int(limit or DEFAULT_RETHINK_LIMIT), 1):
                break

    results = [
        rethink_node(
            str(node.get('id') or ''),
            language=language,
            store=graph_store,
            context_budget=resolved_context_budget,
            policy=tool_policy,
            preview_only=preview_only,
        )
        for node in candidates[: max(int(limit or DEFAULT_RETHINK_LIMIT), 1)]
    ]
    return {
        'ok': True,
        'preview_only': bool(preview_only),
        'active_mode': bool(active_mode),
        'processed': len(results),
        'results': results,
        'repair': repair_report,
        'graph': graph_store.load_graph(),
    }
