from __future__ import annotations

import math
from collections import Counter
from typing import Any

from .duplicate_resolver import normalize_name, score_node
from .graph_store import GraphStore
from .history_store import infer_current_entity, parse_session, recent_dialogue
from .models import ContextCandidate, ContextPayload, ContextScoreBreakdown, HeadBundle, Situation
from .persona_engine import emotion_label, infer_persona_name, load_persona, load_persona_graph, persona_exists, reaction_policy, relevant_reactions
from .prompt_builder import render_graph_context
from .runtime_config import get_runtime_config
from .situation_engine import situation_summary

# Weighted factors stay explicit so the ranking remains explainable in debug output.
_CONTEXT_SCORE_WEIGHTS: dict[str, float] = {
    'relevance': 0.34,
    'recency': 0.12,
    'importance': 0.16,
    'confidence': 0.12,
    'persona_alignment': 0.16,
    'graph_connectivity': 0.10,
}

# Tie-break priorities are intentionally kept out of the score itself so that
# debugging can distinguish semantic score from deterministic ordering policy.
_SOURCE_PRIORITY: dict[str, int] = {
    'persona_memory': 0,
    'persona_triad': 1,
    'local_graph_neighborhood': 2,
    'global_graph_facts': 3,
    'file_ingested_knowledge': 4,
    'session_short_term_history': 5,
}

_PERSONA_RENDER_ORDER: dict[str, int] = {
    'persona_core': 0,
    'persona_state': 1,
    'persona_form': 2,
    'persona_decision_explanation': 3,
    'persona_reactions': 4,
    'persona_relations': 5,
    'persona_examples': 6,
    'persona_knowledge': 7,
    'persona_log_tuples': 8,
}


def _context_config() -> tuple[int, int, int, dict[str, int], dict[str, int]]:
    config = get_runtime_config().context
    return (
        config.max_context_tokens,
        config.question_tokens,
        config.prompt_overhead_tokens,
        dict(config.section_budgets),
        dict(config.section_minimums),
    )


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(str(text or '')) / 4)


def _clip(text: str, token_limit: int) -> str:
    raw = str(text or '').strip()
    if not raw:
        return ''
    max_chars = max(int(token_limit or 0), 0) * 4
    return raw[:max_chars].strip() if max_chars and len(raw) > max_chars else raw


def _fit_section_budget(sections: dict[str, str], *, token_limit: int) -> tuple[dict[str, str], int]:
    _, _, _, section_budgets, section_minimums = _context_config()
    fitted = {
        name: _clip(sections.get(name, ''), section_budgets.get(name, token_limit))
        for name in ('persona_block', 'graph_context', 'recent_dialogue')
    }
    total = sum(_estimate_tokens(value) for value in fitted.values())
    if total <= token_limit:
        return fitted, total
    trim_order = ('graph_context', 'recent_dialogue', 'persona_block')
    while total > token_limit:
        changed = False
        overflow = total - token_limit
        for name in trim_order:
            current = _estimate_tokens(fitted[name])
            minimum = section_minimums.get(name, 0)
            if current <= minimum:
                continue
            reduction = min(max(overflow, 1), max(current - minimum, 0), max(current // 4, 1))
            fitted[name] = _clip(fitted[name], current - reduction)
            total = sum(_estimate_tokens(value) for value in fitted.values())
            changed = True
            if total <= token_limit:
                break
        if not changed:
            break
    return fitted, total


def _normalized_tokens(text: str) -> set[str]:
    return {token for token in normalize_name(text).split() if token}


def _clamp01(value: float) -> float:
    return round(min(1.0, max(0.0, float(value or 0.0))), 6)


def _text_relevance(query_tokens: set[str], text: str) -> float:
    candidate_tokens = _normalized_tokens(text)
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = len(query_tokens & candidate_tokens) / max(len(query_tokens), 1)
    density = len(query_tokens & candidate_tokens) / max(len(candidate_tokens), 1)
    return _clamp01((overlap * 0.7) + (density * 0.3))


def _context_source_for_node(node: dict[str, Any], *, local_neighbor: bool = False) -> str:
    if local_neighbor:
        return 'local_graph_neighborhood'
    context = node.get('context') if isinstance(node.get('context'), dict) else {}
    source = str(context.get('source') or '').strip().lower()
    if source == 'file':
        return 'file_ingested_knowledge'
    return 'global_graph_facts'


def _candidate_recency(candidate: ContextCandidate) -> float:
    if candidate.source == 'session_short_term_history':
        return _clamp01(float(candidate.metadata.get('recency_score') or 0.0))
    return {
        'persona_memory': 0.62,
        'persona_triad': 0.58,
        'local_graph_neighborhood': 0.56,
        'global_graph_facts': 0.44,
        'file_ingested_knowledge': 0.4,
    }.get(candidate.source, 0.35)


def _candidate_importance(candidate: ContextCandidate) -> float:
    if candidate.section == 'graph_context':
        node = candidate.metadata.get('node') if isinstance(candidate.metadata.get('node'), dict) else {}
        return _clamp01(float(node.get('importance') or 0.0))
    if candidate.item_type in {'persona_core', 'persona_state'}:
        return 1.0
    if candidate.item_type in {'persona_form', 'persona_decision_explanation'}:
        return 0.9
    if candidate.item_type == 'persona_reactions':
        return 0.82
    if candidate.item_type == 'persona_knowledge':
        return 0.78
    if candidate.item_type in {'persona_relations', 'persona_examples', 'persona_log_tuples'}:
        return 0.72
    if candidate.section == 'recent_dialogue':
        return 0.55
    return 0.5


def _candidate_confidence(candidate: ContextCandidate) -> float:
    if candidate.section == 'graph_context':
        node = candidate.metadata.get('node') if isinstance(candidate.metadata.get('node'), dict) else {}
        return _clamp01(float(node.get('confidence') or 0.0))
    if candidate.section == 'recent_dialogue':
        return 0.92
    return 0.95


def _candidate_persona_alignment(
    candidate: ContextCandidate,
    *,
    resolved_persona: str,
    current_entity: str,
    persona_relation_targets: set[str],
) -> float:
    if not resolved_persona:
        return 0.0
    if candidate.source in {'persona_memory', 'persona_triad'}:
        return 1.0
    title_norm = normalize_name(candidate.title)
    resolved_norm = normalize_name(resolved_persona)
    current_norm = normalize_name(current_entity)
    if candidate.source == 'local_graph_neighborhood':
        return 0.96
    if title_norm and title_norm in persona_relation_targets:
        return 0.84
    if current_norm and title_norm and title_norm == current_norm:
        return 0.78
    if resolved_norm and resolved_norm in normalize_name(candidate.text):
        return 0.62
    if candidate.source == 'session_short_term_history':
        return 0.35 if resolved_norm and resolved_norm in normalize_name(candidate.text) else 0.18
    return 0.0


def _candidate_graph_connectivity(candidate: ContextCandidate, *, max_degree: int) -> float:
    if candidate.section == 'graph_context':
        degree = int(candidate.metadata.get('degree') or 0)
        if max_degree <= 0:
            return 0.0
        return _clamp01(degree / max(max_degree, 1))
    if candidate.item_type == 'persona_relations':
        return 0.68
    if candidate.item_type == 'persona_reactions':
        return 0.44
    if candidate.item_type == 'persona_log_tuples':
        return 0.3
    return 0.0


def _score_candidate(
    candidate: ContextCandidate,
    *,
    query_tokens: set[str],
    resolved_persona: str,
    current_entity: str,
    persona_relation_targets: set[str],
    max_degree: int,
) -> None:
    text_for_relevance = ' '.join(
        part
        for part in (
            candidate.title,
            candidate.text,
            ' '.join(str(item) for item in list(candidate.metadata.get('aliases') or []) if str(item).strip()),
        )
        if str(part).strip()
    )
    breakdown = ContextScoreBreakdown(
        relevance=_text_relevance(query_tokens, text_for_relevance),
        recency=_candidate_recency(candidate),
        importance=_candidate_importance(candidate),
        confidence=_candidate_confidence(candidate),
        persona_alignment=_candidate_persona_alignment(
            candidate,
            resolved_persona=resolved_persona,
            current_entity=current_entity,
            persona_relation_targets=persona_relation_targets,
        ),
        graph_connectivity=_candidate_graph_connectivity(candidate, max_degree=max_degree),
    )
    breakdown.total = _clamp01(
        sum(getattr(breakdown, name) * weight for name, weight in _CONTEXT_SCORE_WEIGHTS.items())
    )
    candidate.score = breakdown
    factor_map = {
        'relevance': breakdown.relevance,
        'recency': breakdown.recency,
        'importance': breakdown.importance,
        'confidence': breakdown.confidence,
        'persona_alignment': breakdown.persona_alignment,
        'graph_connectivity': breakdown.graph_connectivity,
    }
    ordered = sorted(factor_map.items(), key=lambda item: (-item[1], item[0]))
    reasons = [f'source={candidate.source}']
    for label, value in ordered[:3]:
        if value > 0:
            reasons.append(f'{label}={value:.2f}')
    candidate.reasons = reasons


def _rank_candidates(candidates: list[ContextCandidate]) -> list[ContextCandidate]:
    grouped: dict[str, list[ContextCandidate]] = {'persona_block': [], 'graph_context': [], 'recent_dialogue': []}
    for candidate in candidates:
        grouped.setdefault(candidate.section, []).append(candidate)
    ranked: list[ContextCandidate] = []
    for section, rows in grouped.items():
        rows.sort(
            key=lambda candidate: (
                -candidate.score.total,
                -candidate.score.persona_alignment,
                _SOURCE_PRIORITY.get(candidate.source, 99),
                str(candidate.title or '').lower(),
                candidate.candidate_id,
            )
        )
        for index, candidate in enumerate(rows, start=1):
            candidate.rank = index
        ranked.extend(rows)
    return ranked


def _candidate_token_limit(candidate: ContextCandidate) -> int:
    if candidate.section == 'recent_dialogue':
        return 90
    if candidate.item_type == 'persona_knowledge':
        return 320
    if candidate.item_type == 'persona_examples':
        return 180
    if candidate.item_type == 'persona_log_tuples':
        return 130
    if candidate.item_type == 'persona_relations':
        return 120
    if candidate.item_type == 'persona_reactions':
        return 180
    if candidate.item_type in {'persona_core', 'persona_state', 'persona_form', 'persona_decision_explanation'}:
        return 220
    if candidate.section == 'graph_context':
        return 170
    return 180


def _compress_candidates(candidates: list[ContextCandidate]) -> int:
    compressed = 0
    for candidate in candidates:
        token_limit = _candidate_token_limit(candidate)
        clipped = _clip(candidate.text, token_limit)
        if clipped != candidate.text:
            candidate.text = clipped
            candidate.token_estimate = _estimate_tokens(clipped)
            candidate.compressed = True
            candidate.reasons.append(f'compressed_to={token_limit}')
            compressed += 1
    return compressed


def _persona_query_hints(name: str) -> str:
    if not name:
        return ''
    bundle = load_persona(name)
    if bundle is None:
        return ''
    relation_targets = ' '.join(str(item.get('target') or '').strip() for item in bundle.relations[:6] if str(item.get('target') or '').strip())
    return ' '.join(
        part
        for part in (
            bundle.name,
            ' '.join(bundle.traits[:8]),
            relation_targets,
            _clip(bundle.knowledge, 250),
            ' '.join(bundle.examples[:4]),
        )
        if str(part).strip()
    )


def _build_persona_candidates(name: str, situation: Situation | dict[str, Any] | str) -> tuple[HeadBundle | None, list[ContextCandidate]]:
    bundle = load_persona(name)
    if bundle is None:
        return None, []
    summary = situation_summary(situation)
    fallback_situation = Situation(type='neutral_statement', target='external', severity=0.2, summary='type=neutral_statement; target=external; severity=0.20')
    policy = reaction_policy(bundle, situation if isinstance(situation, (dict, Situation)) else fallback_situation)
    reaction_lines = []
    for item in relevant_reactions(name, situation):
        learned_reaction = item.get('reaction')
        rendered = f"{item.get('situation')} -> {learned_reaction}"
        reaction_lines.append(rendered)

    candidates: list[ContextCandidate] = []

    core_lines = [
        f'You are {bundle.name}.',
        f'Entity type: {bundle.entity_type}.',
        f'Emotion profile: {emotion_label(bundle.emotion_vector)}.',
        'Answer in first person from this persona head.',
        'React from persona traits and the current situation, not from raw user emotion.',
    ]
    if summary:
        core_lines.append(f'Current situation: {summary}.')
    candidates.append(
        ContextCandidate(
            candidate_id=f'persona:{normalize_name(bundle.name)}:core',
            source='persona_memory',
            section='persona_block',
            item_type='persona_core',
            title='persona_core',
            text='\n'.join(core_lines).strip(),
            token_estimate=_estimate_tokens('\n'.join(core_lines)),
            metadata={'persona_name': bundle.name},
        )
    )

    state_lines: list[str] = []
    if bundle.traits:
        state_lines.append(f"Traits: {', '.join(bundle.traits[:8])}.")
    if bundle.emotion_vector:
        emotion_text = ', '.join(f'{key}={value}' for key, value in bundle.emotion_vector.items())
        state_lines.append(f'Emotion vector: {emotion_text}.')
    if bundle.indicators is not None:
        state_lines.append(
            'Persona maturity: '
            f"{bundle.indicators.maturity_level} "
            f"(confidence={bundle.indicators.confidence_score}, "
            f"maturity={bundle.indicators.maturity_score}, "
            f"locked={bundle.indicators.adaptation_locked})."
        )
    if bundle.revision_meta:
        state_lines.append(
            'Persona revisions: '
            f"overall={bundle.revision_meta.get('revision', 1)}, "
            f"baseline={bundle.revision_meta.get('baseline_revision', 1)}, "
            f"dynamic={bundle.revision_meta.get('dynamic_revision', 1)}, "
            f"learned={bundle.revision_meta.get('learned_revision', 1)}."
        )
    state_lines.append(f'Response style: {policy.response_style}.')
    candidates.append(
        ContextCandidate(
            candidate_id=f'persona:{normalize_name(bundle.name)}:state',
            source='persona_memory',
            section='persona_block',
            item_type='persona_state',
            title='persona_state',
            text='\n'.join(state_lines).strip(),
            token_estimate=_estimate_tokens('\n'.join(state_lines)),
            metadata={'persona_name': bundle.name},
        )
    )

    if bundle.persona_form:
        form_lines: list[str] = []
        identity_class = str(bundle.persona_form.get('identity_class') or '').strip()
        sarcasm_profile = str(bundle.persona_form.get('sarcasm_profile') or '').strip()
        clarification_policy = str(bundle.persona_form.get('clarification_policy') or '').strip()
        decision_patterns = [str(item).strip() for item in list(bundle.persona_form.get('decision_patterns') or []) if str(item).strip()]
        response_priorities = [str(item).strip() for item in list(bundle.persona_form.get('response_priorities') or []) if str(item).strip()]
        if identity_class:
            form_lines.append(f'Identity class: {identity_class}.')
        if sarcasm_profile:
            form_lines.append(f'Sarcasm profile: {sarcasm_profile}.')
        if clarification_policy:
            form_lines.append(f'Clarification policy: {clarification_policy}')
        if decision_patterns:
            form_lines.append(f"Decision patterns: {' | '.join(decision_patterns[:4])}.")
        if response_priorities:
            form_lines.append(f"Response priorities: {' | '.join(response_priorities[:4])}.")
        if form_lines:
            form_text = '\n'.join(form_lines).strip()
            candidates.append(
                ContextCandidate(
                    candidate_id=f'persona:{normalize_name(bundle.name)}:form',
                    source='persona_triad',
                    section='persona_block',
                    item_type='persona_form',
                    title='persona_form',
                    text=form_text,
                    token_estimate=_estimate_tokens(form_text),
                    metadata={'persona_name': bundle.name},
                )
            )

    if bundle.decision_explanation:
        candidates.append(
            ContextCandidate(
                candidate_id=f'persona:{normalize_name(bundle.name)}:decision_explanation',
                source='persona_triad',
                section='persona_block',
                item_type='persona_decision_explanation',
                title='persona_decision_explanation',
                text=f'Decision explanation: {bundle.decision_explanation}',
                token_estimate=_estimate_tokens(bundle.decision_explanation),
                metadata={'persona_name': bundle.name},
            )
        )

    if bundle.relations:
        relation_text = '; '.join(f"{item.get('type')} {item.get('target')}" for item in bundle.relations[:6])
        candidates.append(
            ContextCandidate(
                candidate_id=f'persona:{normalize_name(bundle.name)}:relations',
                source='persona_memory',
                section='persona_block',
                item_type='persona_relations',
                title='persona_relations',
                text=f'Relations: {relation_text}.',
                token_estimate=_estimate_tokens(relation_text),
                metadata={'persona_name': bundle.name},
            )
        )

    if bundle.examples:
        examples_text = f"Examples: {' | '.join(bundle.examples[:4])}."
        candidates.append(
            ContextCandidate(
                candidate_id=f'persona:{normalize_name(bundle.name)}:examples',
                source='persona_memory',
                section='persona_block',
                item_type='persona_examples',
                title='persona_examples',
                text=examples_text,
                token_estimate=_estimate_tokens(examples_text),
                metadata={'persona_name': bundle.name},
            )
        )

    if bundle.knowledge:
        knowledge_text = f'Knowledge: {bundle.knowledge}'
        candidates.append(
            ContextCandidate(
                candidate_id=f'persona:{normalize_name(bundle.name)}:knowledge',
                source='persona_memory',
                section='persona_block',
                item_type='persona_knowledge',
                title='persona_knowledge',
                text=knowledge_text,
                token_estimate=_estimate_tokens(knowledge_text),
                metadata={'persona_name': bundle.name},
            )
        )

    if reaction_lines:
        reactions_text = 'Learned situation reactions:\n' + '\n'.join(f'- {line}' for line in reaction_lines[:4])
        candidates.append(
            ContextCandidate(
                candidate_id=f'persona:{normalize_name(bundle.name)}:reactions',
                source='persona_memory',
                section='persona_block',
                item_type='persona_reactions',
                title='persona_reactions',
                text=reactions_text,
                token_estimate=_estimate_tokens(reactions_text),
                metadata={'persona_name': bundle.name},
            )
        )

    if bundle.log_tuples:
        rendered_tuples = [
            f"{tuple(item.get('tuple') or ())} freq={item.get('frequency')}"
            for item in list(bundle.log_tuples or [])[:4]
            if isinstance(item, dict)
        ]
        if rendered_tuples:
            tuple_text = f"Behavior log tuples: {' | '.join(rendered_tuples)}."
            candidates.append(
                ContextCandidate(
                    candidate_id=f'persona:{normalize_name(bundle.name)}:log_tuples',
                    source='persona_triad',
                    section='persona_block',
                    item_type='persona_log_tuples',
                    title='persona_log_tuples',
                    text=tuple_text,
                    token_estimate=_estimate_tokens(tuple_text),
                    metadata={'persona_name': bundle.name},
                )
            )

    return bundle, [candidate for candidate in candidates if candidate.text]


def _build_history_candidates(session_id: str) -> list[ContextCandidate]:
    parsed = parse_session(session_id)
    messages = list(parsed.get('messages') or [])[-6:] if parsed else []
    candidates: list[ContextCandidate] = []
    total = len(messages)
    for index, item in enumerate(messages):
        role = str(item.get('role') or '').strip()
        message = str(item.get('message') or '').strip()
        if not role or not message:
            continue
        recency_score = ((index + 1) / total) if total else 0.0
        text = f'{role}: {message}'
        candidates.append(
            ContextCandidate(
                candidate_id=f'history:{session_id}:{index}:{role}',
                source='session_short_term_history',
                section='recent_dialogue',
                item_type='message_line',
                title=f'{role}_{index}',
                text=text,
                token_estimate=_estimate_tokens(text),
                metadata={
                    'role': role,
                    'sequence': index,
                    'timestamp': str(item.get('timestamp') or ''),
                    'recency_score': recency_score,
                },
            )
        )
    if candidates:
        return candidates
    fallback = recent_dialogue(session_id)
    if not fallback:
        return []
    for index, line in enumerate(part.strip() for part in fallback.splitlines() if part.strip()):
        candidates.append(
            ContextCandidate(
                candidate_id=f'history:{session_id}:fallback:{index}',
                source='session_short_term_history',
                section='recent_dialogue',
                item_type='message_line',
                title=f'fallback_{index}',
                text=line,
                token_estimate=_estimate_tokens(line),
                metadata={'sequence': index, 'recency_score': 0.5},
            )
        )
    return candidates


def _merge_node_payload(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if key in {'aliases', 'facts'}:
            existing = [str(item).strip() for item in list(merged.get(key) or []) if str(item).strip()]
            extra = [str(item).strip() for item in list(value or []) if str(item).strip()]
            merged[key] = list(dict.fromkeys(existing + extra))
            continue
        if key == 'description':
            merged[key] = str(merged.get(key) or '').strip() or str(value or '').strip()
            continue
        if key == 'context':
            merged[key] = {**dict(merged.get(key) or {}), **dict(value or {})}
            continue
        if merged.get(key) in (None, '', [], {}) and value not in (None, '', [], {}):
            merged[key] = value
    return merged


def _build_graph_candidates(
    *,
    query: str,
    resolved_persona: str,
    store: GraphStore,
) -> tuple[list[ContextCandidate], list[dict[str, Any]], list[dict[str, Any]]]:
    subgraph = store.subgraph(query, limit=6, depth=1)
    local_graph = load_persona_graph(resolved_persona) if resolved_persona else {'nodes': [], 'edges': []}

    entries: dict[str, dict[str, Any]] = {}
    for node in list(subgraph.get('nodes') or []):
        node_id = str(node.get('id') or '')
        if not node_id:
            continue
        entries[node_id] = {'node': dict(node), 'local_neighbor': False}
    for node in list(local_graph.get('nodes') or []):
        node_id = str(node.get('id') or '')
        if not node_id:
            continue
        if node_id in entries:
            entries[node_id]['node'] = _merge_node_payload(entries[node_id]['node'], dict(node))
            entries[node_id]['local_neighbor'] = True
        else:
            entries[node_id] = {'node': dict(node), 'local_neighbor': True}

    all_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for edge in list(subgraph.get('edges') or []) + list(local_graph.get('edges') or []):
        key = (str(edge.get('from') or ''), str(edge.get('type') or ''), str(edge.get('to') or ''))
        if not all(key) or key in seen_edges:
            continue
        seen_edges.add(key)
        all_edges.append(dict(edge))

    degree_map: Counter[str] = Counter()
    for edge in all_edges:
        src = str(edge.get('from') or '')
        dst = str(edge.get('to') or '')
        if src:
            degree_map[src] += 1
        if dst:
            degree_map[dst] += 1

    candidates: list[ContextCandidate] = []
    for node_id, payload in entries.items():
        node = dict(payload['node'])
        aliases = [str(item).strip() for item in list(node.get('aliases') or []) if str(item).strip()]
        facts = [str(item).strip() for item in list(node.get('facts') or []) if str(item).strip()]
        preview_lines = [
            f"{node.get('name')} [{node.get('type')}] importance={node.get('importance')} confidence={node.get('confidence')} frequency={node.get('frequency')}.",
        ]
        if str(node.get('translation_line') or '').strip():
            preview_lines.append(f"translation: {node.get('translation_line')}")
        if str(node.get('description') or '').strip():
            preview_lines.append(str(node.get('description') or '').strip())
        if facts:
            preview_lines.append(f"facts: {' | '.join(facts[:4])}")
        preview = '\n'.join(preview_lines).strip()
        source = _context_source_for_node(node, local_neighbor=bool(payload.get('local_neighbor')))
        candidates.append(
            ContextCandidate(
                candidate_id=f'graph:{node_id}',
                source=source,
                section='graph_context',
                item_type='graph_node',
                title=str(node.get('name') or node_id),
                text=preview,
                token_estimate=_estimate_tokens(preview),
                metadata={
                    'node': node,
                    'node_id': node_id,
                    'aliases': aliases,
                    'degree': degree_map.get(node_id, 0),
                },
            )
        )
    return candidates, [dict(item['node']) for item in entries.values()], all_edges


def _collect_candidates(
    *,
    question: str,
    session_id: str,
    selected_persona: str,
    explicit_context: str,
    situation: Situation | dict[str, Any] | str | None,
    store: GraphStore,
) -> dict[str, Any]:
    clipped_question = _clip(question, _context_config()[1])
    current_entity = infer_current_entity(session_id)
    resolved = infer_persona_name(question, selected_name=selected_persona, current_entity=current_entity)
    if resolved and not persona_exists(resolved):
        resolved = ''
    rendered_situation = situation_summary(situation or {})
    query = ' '.join(
        part
        for part in (
            clipped_question,
            explicit_context,
            recent_dialogue(session_id),
            current_entity,
            resolved,
            rendered_situation,
            _persona_query_hints(resolved),
        )
        if str(part).strip()
    )
    persona_bundle, persona_candidates = _build_persona_candidates(
        resolved,
        situation or Situation(type='neutral_query', target='external', severity=0.35, summary='type=neutral_query; target=external; severity=0.35'),
    ) if resolved else (None, [])
    history_candidates = _build_history_candidates(session_id)
    graph_candidates, graph_nodes, graph_edges = _build_graph_candidates(query=query, resolved_persona=resolved, store=store)
    candidates = persona_candidates + history_candidates + graph_candidates
    return {
        'clipped_question': clipped_question,
        'current_entity': current_entity,
        'resolved_persona': resolved,
        'rendered_situation': rendered_situation,
        'query': query,
        'persona_bundle': persona_bundle,
        'candidates': candidates,
        'graph_nodes': graph_nodes,
        'graph_edges': graph_edges,
    }


def _score_rank_compress(
    *,
    candidates: list[ContextCandidate],
    question: str,
    explicit_context: str,
    current_entity: str,
    rendered_situation: str,
    persona_bundle: HeadBundle | None,
) -> tuple[list[ContextCandidate], dict[str, Any]]:
    query_tokens = _normalized_tokens(' '.join(part for part in (question, explicit_context, current_entity, rendered_situation) if str(part).strip()))
    persona_relation_targets = {
        normalize_name(str(item.get('target') or item.get('to') or ''))
        for item in list(persona_bundle.relations or [])
        if isinstance(item, dict)
    } if persona_bundle else set()
    max_degree = max((int(candidate.metadata.get('degree') or 0) for candidate in candidates if candidate.section == 'graph_context'), default=0)
    for candidate in candidates:
        _score_candidate(
            candidate,
            query_tokens=query_tokens,
            resolved_persona=persona_bundle.name if persona_bundle else '',
            current_entity=current_entity,
            persona_relation_targets=persona_relation_targets,
            max_degree=max_degree,
        )
    ranked = _rank_candidates(candidates)
    compressed = _compress_candidates(ranked)
    diagnostics = {
        'weights': dict(_CONTEXT_SCORE_WEIGHTS),
        'source_counts': dict(Counter(candidate.source for candidate in ranked)),
        'query_tokens': sorted(query_tokens),
        'compressed_candidates': compressed,
    }
    return ranked, diagnostics


def _select_edges_for_nodes(selected_nodes: list[dict[str, Any]], all_edges: list[dict[str, Any]], node_rank_map: dict[str, int]) -> list[dict[str, Any]]:
    selected_ids = {str(node.get('id') or '') for node in selected_nodes}
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in all_edges:
        key = (str(edge.get('from') or ''), str(edge.get('type') or ''), str(edge.get('to') or ''))
        if not all(key) or key in seen:
            continue
        if key[0] not in selected_ids or key[2] not in selected_ids:
            continue
        seen.add(key)
        edges.append(dict(edge))
    edges.sort(
        key=lambda edge: (
            min(node_rank_map.get(str(edge.get('from') or ''), 999), node_rank_map.get(str(edge.get('to') or ''), 999)),
            str(edge.get('type') or ''),
            str(edge.get('from') or ''),
            str(edge.get('to') or ''),
        )
    )
    return edges[:14]


def _pack_persona_block(candidates: list[ContextCandidate], *, token_budget: int) -> tuple[str, list[ContextCandidate]]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            _PERSONA_RENDER_ORDER.get(candidate.item_type, 99),
            candidate.rank,
            candidate.candidate_id,
        ),
    )
    selected: list[ContextCandidate] = []
    used = 0
    for candidate in ordered:
        if used + candidate.token_estimate > token_budget and selected:
            continue
        selected.append(candidate)
        used += candidate.token_estimate
        candidate.selected = True
    if not selected and ordered:
        selected = [ordered[0]]
        ordered[0].selected = True
    text = '\n'.join(candidate.text for candidate in selected if candidate.text).strip()
    return text, selected


def _pack_recent_dialogue(candidates: list[ContextCandidate], *, token_budget: int) -> tuple[str, list[ContextCandidate]]:
    ranked = sorted(candidates, key=lambda candidate: (candidate.rank, candidate.candidate_id))
    selected: list[ContextCandidate] = []
    used = 0
    for candidate in ranked:
        if used + candidate.token_estimate > token_budget and selected:
            continue
        selected.append(candidate)
        used += candidate.token_estimate
        candidate.selected = True
    if not selected and ranked:
        selected = [ranked[0]]
        ranked[0].selected = True
    selected.sort(key=lambda candidate: int(candidate.metadata.get('sequence') or 0))
    return '\n'.join(candidate.text for candidate in selected if candidate.text).strip(), selected


def _pack_graph_context(
    candidates: list[ContextCandidate],
    *,
    all_edges: list[dict[str, Any]],
    token_budget: int,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[ContextCandidate]]:
    ranked = sorted(candidates, key=lambda candidate: (candidate.rank, candidate.candidate_id))
    selected_candidates: list[ContextCandidate] = []
    selected_nodes: list[dict[str, Any]] = []
    node_rank_map: dict[str, int] = {}
    graph_context = ''
    selected_edges: list[dict[str, Any]] = []
    for candidate in ranked:
        node = dict(candidate.metadata.get('node') or {})
        node_id = str(node.get('id') or '')
        if not node_id or node_id in node_rank_map:
            continue
        trial_nodes = selected_nodes + [node]
        trial_rank_map = {**node_rank_map, node_id: candidate.rank}
        trial_edges = _select_edges_for_nodes(trial_nodes, all_edges, trial_rank_map)
        trial_context = render_graph_context(trial_nodes, trial_edges)
        if _estimate_tokens(trial_context) > token_budget and selected_nodes:
            continue
        selected_candidates.append(candidate)
        selected_nodes = trial_nodes
        node_rank_map = trial_rank_map
        selected_edges = trial_edges
        graph_context = trial_context
        candidate.selected = True
        if len(selected_nodes) >= 8:
            break
    if not selected_nodes and ranked:
        fallback = ranked[0]
        node = dict(fallback.metadata.get('node') or {})
        if node:
            selected_candidates = [fallback]
            selected_nodes = [node]
            node_rank_map = {str(node.get('id') or ''): fallback.rank}
            selected_edges = _select_edges_for_nodes(selected_nodes, all_edges, node_rank_map)
            graph_context = render_graph_context(selected_nodes, selected_edges)
            fallback.selected = True
    return _clip(graph_context, token_budget), selected_nodes, selected_edges, selected_candidates


def _pack_context(
    *,
    candidates: list[ContextCandidate],
    all_edges: list[dict[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    max_context_tokens, question_budget, prompt_overhead_tokens, section_budgets, _ = _context_config()
    persona_candidates = [candidate for candidate in candidates if candidate.section == 'persona_block']
    graph_candidates = [candidate for candidate in candidates if candidate.section == 'graph_context']
    history_candidates = [candidate for candidate in candidates if candidate.section == 'recent_dialogue']

    persona_block, selected_persona = _pack_persona_block(persona_candidates, token_budget=section_budgets['persona_block'])
    graph_context, selected_nodes, selected_edges, selected_graph = _pack_graph_context(
        graph_candidates,
        all_edges=all_edges,
        token_budget=section_budgets['graph_context'],
    )
    recent_text, selected_history = _pack_recent_dialogue(history_candidates, token_budget=section_budgets['recent_dialogue'])

    packed = {
        'persona_block': persona_block,
        'graph_context': graph_context,
        'recent_dialogue': recent_text,
    }
    diagnostics = {
        'packed_candidate_counts': {
            'persona_block': len(selected_persona),
            'graph_context': len(selected_graph),
            'recent_dialogue': len(selected_history),
        },
        'question_budget': question_budget,
        'prompt_overhead_tokens': prompt_overhead_tokens,
        'max_context_tokens': max_context_tokens,
    }
    return packed, selected_nodes, selected_edges, diagnostics


def build_context(
    *,
    question: str,
    session_id: str,
    selected_persona: str = '',
    explicit_context: str = '',
    situation: Situation | dict[str, Any] | str | None = None,
    store: GraphStore | None = None,
) -> dict[str, Any]:
    max_context_tokens, _, prompt_overhead_tokens, _, _ = _context_config()
    graph_store = store or GraphStore()
    collected = _collect_candidates(
        question=question,
        session_id=session_id,
        selected_persona=selected_persona,
        explicit_context=explicit_context,
        situation=situation,
        store=graph_store,
    )

    ranked, score_debug = _score_rank_compress(
        candidates=list(collected['candidates']),
        question=collected['clipped_question'],
        explicit_context=explicit_context,
        current_entity=collected['current_entity'],
        rendered_situation=collected['rendered_situation'],
        persona_bundle=collected['persona_bundle'],
    )
    packed_sections, selected_nodes, selected_edges, pack_debug = _pack_context(
        candidates=ranked,
        all_edges=list(collected['graph_edges']),
    )

    section_limit = max(max_context_tokens - _estimate_tokens(collected['clipped_question']) - prompt_overhead_tokens, 0)
    sections, section_tokens = _fit_section_budget(packed_sections, token_limit=section_limit)
    estimated = section_tokens + _estimate_tokens(collected['clipped_question']) + prompt_overhead_tokens

    payload = ContextPayload(
        persona_name=collected['resolved_persona'],
        persona_block=sections['persona_block'],
        graph_context=sections['graph_context'],
        recent_dialogue=sections['recent_dialogue'],
        current_entity=collected['current_entity'] or collected['resolved_persona'],
        selected_nodes=selected_nodes,
        selected_edges=selected_edges,
        estimated_tokens=estimated,
        context_debug={
            'stages': {
                'collect_candidates': len(collected['candidates']),
                'score_candidates': len(ranked),
                'rank_candidates': len(ranked),
                'compress_candidates': int(score_debug['compressed_candidates']),
                'pack_context': dict(pack_debug['packed_candidate_counts']),
            },
            'query': collected['query'],
            'situation': collected['rendered_situation'],
            'weights': dict(score_debug['weights']),
            'source_counts': dict(score_debug['source_counts']),
            'selected_items': [candidate.to_dict() for candidate in ranked if candidate.selected],
            'top_unselected_items': [candidate.to_dict() for candidate in ranked if not candidate.selected][:12],
            'selected_node_ids': [str(node.get('id') or '') for node in selected_nodes],
            'selected_edge_keys': [
                (str(edge.get('from') or ''), str(edge.get('type') or ''), str(edge.get('to') or ''))
                for edge in selected_edges
            ],
            'estimated_tokens': estimated,
        },
    )
    response = payload.to_dict()
    response['situation'] = collected['rendered_situation']
    return response


def answerable_node_view(node_id: str, *, store: GraphStore | None = None) -> dict[str, Any] | None:
    return (store or GraphStore()).answerable_node_view(node_id)
