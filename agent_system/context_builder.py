from __future__ import annotations

import math
from typing import Any

from .duplicate_resolver import score_node
from .graph_store import GraphStore
from .history_store import infer_current_entity, recent_dialogue
from .models import ContextPayload, Situation
from .persona_engine import emotion_label, infer_persona_name, load_persona, load_persona_graph, persona_exists, reaction_policy, relevant_reactions
from .prompt_builder import render_graph_context
from .runtime_config import get_runtime_config
from .situation_engine import situation_summary


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
    max_chars = token_limit * 4
    return raw[:max_chars].strip() if len(raw) > max_chars else raw


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


def _build_persona_block(name: str, situation: Situation | dict[str, Any] | str) -> str:
    bundle = load_persona(name)
    if bundle is None:
        return ''
    summary = situation_summary(situation)
    fallback_situation = Situation(type='neutral_statement', target='external', severity=0.2, summary='type=neutral_statement; target=external; severity=0.20')
    policy = reaction_policy(bundle, situation if isinstance(situation, (dict, Situation)) else fallback_situation)
    reaction_lines = []
    for item in relevant_reactions(name, situation):
        learned_reaction = item.get('reaction')
        rendered = f"{item.get('situation')} -> {learned_reaction}"
        reaction_lines.append(rendered)
    lines = [
        f'You are {bundle.name}.',
        f'Entity type: {bundle.entity_type}.',
        f'Emotion profile: {emotion_label(bundle.emotion_vector)}.',
        'Answer in first person from this persona head.',
        'React from persona traits and the current situation, not from raw user emotion.',
    ]
    if summary:
        lines.append(f'Current situation: {summary}.')
    if bundle.traits:
        lines.append(f"Traits: {', '.join(bundle.traits[:8])}.")
    if bundle.emotion_vector:
        emotion_text = ', '.join(f'{key}={value}' for key, value in bundle.emotion_vector.items())
        lines.append(f'Emotion vector: {emotion_text}.')
    lines.append(f'Response style: {policy.response_style}.')
    if bundle.persona_form:
        identity_class = str(bundle.persona_form.get('identity_class') or '').strip()
        sarcasm_profile = str(bundle.persona_form.get('sarcasm_profile') or '').strip()
        clarification_policy = str(bundle.persona_form.get('clarification_policy') or '').strip()
        decision_patterns = [str(item).strip() for item in list(bundle.persona_form.get('decision_patterns') or []) if str(item).strip()]
        response_priorities = [str(item).strip() for item in list(bundle.persona_form.get('response_priorities') or []) if str(item).strip()]
        if identity_class:
            lines.append(f'Identity class: {identity_class}.')
        if sarcasm_profile:
            lines.append(f'Sarcasm profile: {sarcasm_profile}.')
        if clarification_policy:
            lines.append(f'Clarification policy: {clarification_policy}')
        if decision_patterns:
            lines.append(f"Decision patterns: {' | '.join(decision_patterns[:4])}.")
        if response_priorities:
            lines.append(f"Response priorities: {' | '.join(response_priorities[:4])}.")
    if bundle.decision_explanation:
        lines.append(f'Decision explanation: {_clip(bundle.decision_explanation, 220)}')
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
    if bundle.log_tuples:
        rendered_tuples = [
            f"{tuple(item.get('tuple') or ())} freq={item.get('frequency')}"
            for item in list(bundle.log_tuples or [])[:4]
            if isinstance(item, dict)
        ]
        if rendered_tuples:
            lines.append(f"Behavior log tuples: {' | '.join(rendered_tuples)}.")
    return '\n'.join(lines).strip()


def build_context(
    *,
    question: str,
    session_id: str,
    selected_persona: str = '',
    explicit_context: str = '',
    situation: Situation | dict[str, Any] | str | None = None,
    store: GraphStore | None = None,
) -> dict[str, Any]:
    max_context_tokens, question_token_budget, prompt_overhead_tokens, _, _ = _context_config()
    graph_store = store or GraphStore()
    clipped_question = _clip(question, question_token_budget)
    recent = recent_dialogue(session_id)
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
            recent,
            current_entity,
            resolved,
            rendered_situation,
            _persona_query_hints(resolved),
        )
        if str(part).strip()
    )
    subgraph = graph_store.subgraph(query, limit=6, depth=1)
    selected_nodes = list(subgraph.get('nodes') or [])
    selected_nodes.sort(key=score_node, reverse=True)
    selected_nodes = selected_nodes[:6]
    selected_node_ids = {str(node.get('id') or '') for node in selected_nodes}
    selected_edges = [
        edge
        for edge in list(subgraph.get('edges') or [])
        if str(edge.get('from') or '') in selected_node_ids or str(edge.get('to') or '') in selected_node_ids
    ][:12]

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

    selected_nodes = selected_nodes[:8]
    selected_edges = selected_edges[:14]
    persona_block = _build_persona_block(
        resolved,
        situation or Situation(type='neutral_query', target='external', severity=0.35, summary='type=neutral_query; target=external; severity=0.35'),
    ) if resolved else ''
    graph_context = render_graph_context(selected_nodes, selected_edges)
    recent_text = _clip(recent, 900)
    section_limit = max(max_context_tokens - _estimate_tokens(clipped_question) - prompt_overhead_tokens, 0)
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
    estimated = section_tokens + _estimate_tokens(clipped_question) + prompt_overhead_tokens

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
    response = payload.to_dict()
    response['situation'] = rendered_situation
    return response


def answerable_node_view(node_id: str, *, store: GraphStore | None = None) -> dict[str, Any] | None:
    return (store or GraphStore()).answerable_node_view(node_id)
