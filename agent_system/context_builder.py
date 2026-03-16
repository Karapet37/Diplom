from __future__ import annotations

import math
from typing import Any

from .duplicate_resolver import score_node
from .graph_store import GraphStore
from .history_store import infer_current_entity, recent_dialogue
from .models import ContextPayload
from .persona_engine import emotion_label, infer_persona_name, load_persona, load_persona_graph, persona_exists, relevant_reactions
from .prompt_builder import render_graph_context

MAX_CONTEXT_TOKENS = 5000
QUESTION_TOKEN_BUDGET = 1200
PROMPT_OVERHEAD_TOKENS = 180
SECTION_BUDGETS = {
    'persona_block': 1600,
    'graph_context': 2200,
    'recent_dialogue': 900,
}
SECTION_MINIMUMS = {
    'persona_block': 500,
    'graph_context': 350,
    'recent_dialogue': 180,
}


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(str(text or '')) / 4)


def _clip(text: str, token_limit: int) -> str:
    raw = str(text or '').strip()
    if not raw:
        return ''
    max_chars = token_limit * 4
    return raw[:max_chars].strip() if len(raw) > max_chars else raw


def _fit_section_budget(sections: dict[str, str], *, token_limit: int) -> tuple[dict[str, str], int]:
    fitted = {
        name: _clip(sections.get(name, ''), SECTION_BUDGETS.get(name, token_limit))
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
            minimum = SECTION_MINIMUMS.get(name, 0)
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


def _build_persona_block(name: str, situation: str) -> str:
    bundle = load_persona(name)
    if bundle is None:
        return ''
    reaction_lines = []
    for item in relevant_reactions(name, situation):
        reaction = item.get('reaction')
        rendered = f"{item.get('situation')} -> {reaction}"
        reaction_lines.append(rendered)
    lines = [
        f'You are {bundle.name}.',
        f'Entity type: {bundle.entity_type}.',
        f'Emotion profile: {emotion_label(bundle.emotion_vector)}.',
        'Answer in first person from this persona head.',
    ]
    if bundle.traits:
        lines.append(f"Traits: {', '.join(bundle.traits[:8])}.")
    if bundle.emotion_vector:
        emotion_text = ', '.join(f'{key}={value}' for key, value in bundle.emotion_vector.items())
        lines.append(f'Emotion vector: {emotion_text}.')
    if bundle.relations:
        relation_text = '; '.join(f"{item.get('type')} {item.get('target')}" for item in bundle.relations[:6])
        lines.append(f'Relations: {relation_text}.')
    if bundle.examples:
        lines.append(f"Examples: {' | '.join(bundle.examples[:4])}.")
    if bundle.knowledge:
        lines.append(f'Knowledge: {_clip(bundle.knowledge, 700)}')
    if reaction_lines:
        lines.append('Learned situation reactions:')
        lines.extend(f'- {line}' for line in reaction_lines[:4])
    return '\n'.join(lines).strip()


def build_context(
    *,
    question: str,
    session_id: str,
    selected_persona: str = '',
    explicit_context: str = '',
    situation: str = '',
    store: GraphStore | None = None,
) -> dict[str, Any]:
    graph_store = store or GraphStore()
    clipped_question = _clip(question, QUESTION_TOKEN_BUDGET)
    recent = recent_dialogue(session_id)
    current_entity = infer_current_entity(session_id)
    resolved = infer_persona_name(question, selected_name=selected_persona, current_entity=current_entity)
    if resolved and not persona_exists(resolved):
        resolved = ''

    query = ' '.join(
        part
        for part in (
            clipped_question,
            explicit_context,
            recent,
            current_entity,
            resolved,
            _persona_query_hints(resolved),
        )
        if str(part).strip()
    )
    subgraph = graph_store.subgraph(query, limit=8, depth=1)
    selected_nodes = list(subgraph.get('nodes') or [])
    selected_nodes.sort(key=score_node, reverse=True)
    selected_nodes = selected_nodes[:8]
    selected_node_ids = {str(node.get('id') or '') for node in selected_nodes}
    selected_edges = [
        edge
        for edge in list(subgraph.get('edges') or [])
        if str(edge.get('from') or '') in selected_node_ids or str(edge.get('to') or '') in selected_node_ids
    ][:18]

    if resolved:
        local_graph = load_persona_graph(resolved)
        for node in list(local_graph.get('nodes') or []):
            if str(node.get('id') or '') not in selected_node_ids:
                selected_nodes.append(node)
                selected_node_ids.add(str(node.get('id') or ''))
        seen_edges = {
            (str(edge.get('from') or ''), str(edge.get('type') or ''), str(edge.get('to') or ''))
            for edge in selected_edges
        }
        for edge in list(local_graph.get('edges') or []):
            key = (str(edge.get('from') or ''), str(edge.get('type') or ''), str(edge.get('to') or ''))
            if all(key) and key not in seen_edges:
                seen_edges.add(key)
                selected_edges.append(edge)

    selected_nodes = selected_nodes[:10]
    selected_edges = selected_edges[:18]
    persona_block = _build_persona_block(resolved, situation or question) if resolved else ''
    graph_context = render_graph_context(selected_nodes, selected_edges)
    recent_text = _clip(recent, 900)
    section_limit = max(MAX_CONTEXT_TOKENS - _estimate_tokens(clipped_question) - PROMPT_OVERHEAD_TOKENS, 0)
    sections, section_tokens = _fit_section_budget(
        {
            'persona_block': persona_block,
            'graph_context': graph_context,
            'recent_dialogue': recent_text,
        },
        token_limit=section_limit,
    )
    persona_block = sections['persona_block']
    graph_context = sections['graph_context']
    recent_text = sections['recent_dialogue']
    estimated = section_tokens + _estimate_tokens(clipped_question) + PROMPT_OVERHEAD_TOKENS

    payload = ContextPayload(
        persona_name=resolved,
        persona_block=persona_block,
        graph_context=graph_context,
        recent_dialogue=recent_text,
        current_entity=current_entity or resolved,
        selected_nodes=selected_nodes,
        selected_edges=selected_edges,
        estimated_tokens=estimated,
    )
    return {
        'persona_name': payload.persona_name,
        'persona_block': payload.persona_block,
        'graph_context': payload.graph_context,
        'recent_dialogue': payload.recent_dialogue,
        'current_entity': payload.current_entity,
        'nodes': payload.selected_nodes,
        'edges': payload.selected_edges,
        'estimated_tokens': payload.estimated_tokens,
    }


def answerable_node_view(node_id: str, *, store: GraphStore | None = None) -> dict[str, Any] | None:
    return (store or GraphStore()).answerable_node_view(node_id)
