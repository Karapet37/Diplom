from __future__ import annotations

import math
from collections import Counter
from typing import Any

from .duplicate_resolver import normalize_name, score_node
from .graph_store import GraphStore
from .history_store import infer_current_entity, parse_session, recent_dialogue
from .models import ContextCandidate, ContextPayload, ContextScoreBreakdown, HeadBundle, MessageAnalysis, MoodResearchReport, Situation, SocialRoleDecision
from .persona_engine import infer_persona_name, load_active_persona, load_persona_graph, persona_is_registered, reaction_policy, relevant_reactions
from .prompt_builder import render_graph_context
from .runtime_config import get_runtime_config
from .semantic_routing import infer_semantic_focus
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
    'social_role': 2,
    'mood_research': 3,
    'session_graph_context': 4,
    'local_graph_neighborhood': 5,
    'global_graph_facts': 6,
    'file_ingested_knowledge': 7,
    'session_short_term_history': 8,
}

_PERSONA_RENDER_ORDER: dict[str, int] = {
    'persona_core': 0,
    'persona_identity': 1,
    'persona_work_profile': 2,
    'persona_social_role': 3,
    'persona_mood_dynamics': 4,
    'persona_control': 5,
    'persona_knowledge': 6,
    'persona_relations': 7,
    'persona_form': 8,
    'persona_decision_explanation': 9,
    'persona_state': 10,
    'persona_reactions': 11,
    'persona_examples': 12,
    'persona_log_tuples': 13,
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


def _render_emotional_stance(emotion_vector: dict[str, Any]) -> str:
    vector = {str(key): float(value or 0.0) for key, value in dict(emotion_vector or {}).items()}
    traits: list[str] = []
    if vector.get('confidence', 0.0) >= 0.6:
        traits.append('steady')
    if vector.get('curiosity', 0.0) >= 0.6:
        traits.append('curious')
    if vector.get('empathy', 0.0) >= 0.5:
        traits.append('attentive to people')
    if vector.get('anger', 0.0) >= 0.35:
        traits.append('firm')
    if vector.get('fear', 0.0) >= 0.35:
        traits.append('cautious')
    if not traits:
        traits.append('composed')
    return ', '.join(traits[:3])


def _render_situation_note(situation: Situation | dict[str, Any] | str) -> str:
    if isinstance(situation, Situation):
        situation_type = str(situation.type or '').strip()
        target = str(situation.target or '').strip()
    elif isinstance(situation, dict):
        situation_type = str(situation.get('type') or '').strip()
        target = str(situation.get('target') or '').strip()
    else:
        situation_type = ''
        target = ''
    if situation_type == 'neutral_query' and target == 'persona':
        return 'Situation note: the user is asking a direct question about your own life or work. Answer concretely from your lived routine, values, and known biography.'
    if situation_type == 'neutral_query':
        return 'Situation note: answer directly and stay grounded in what you know.'
    if situation_type == 'user_distress':
        return 'Situation note: the user appears distressed. Be warmer, practical, and clear without pretending certainty.'
    if situation_type == 'insult' and target == 'persona':
        return 'Situation note: the user is insulting you. Hold a calm boundary and return to the concrete issue without mirroring the hostility.'
    if situation_type == 'abnormal_behavior':
        return 'Situation note: the user describes harmful or morally abnormal behavior. Do not mirror approval; respond in a corrective and grounded way.'
    if situation_type == 'user_anger':
        return 'Situation note: the user is angry. Stay measured and do not let their emotion dictate your own tone.'
    return ''


def _render_learned_reaction_line(situation_label: str, reaction_label: str) -> str:
    situation_text = str(situation_label or '').strip()
    reaction_text = str(reaction_label or '').strip()
    lowered = reaction_text.lower()
    if 'answers_substance_first_with_persona_grounding' in lowered:
        return f"{situation_text} -> answer with substance first, using your own biography, work, and known routines."
    if 'asks_for_missing_fact_before_claiming_certainty' in lowered:
        return f"{situation_text} -> ask for the missing fact that would materially change the judgment before sounding certain."
    if 'firm_boundary' in lowered:
        return f"{situation_text} -> set a calm boundary, then return to facts and consequences."
    if 'measured_support' in lowered or 'supportive' in lowered:
        return f"{situation_text} -> respond with measured support, practical care, and honest limits."
    if 'corrective' in lowered:
        return f"{situation_text} -> respond in a corrective, non-approving tone."
    return f'{situation_text} -> {reaction_text}'


def _context_source_for_node(
    node: dict[str, Any],
    *,
    session_id: str = '',
    local_neighbor: bool = False,
    retrieval_source: str = '',
) -> str:
    context = node.get('context') if isinstance(node.get('context'), dict) else {}
    source = str(context.get('source') or '').strip().lower()
    retrieval = str(retrieval_source or '').strip().lower()
    session_ids = {str(item).strip() for item in list(context.get('session_ids') or []) if str(item).strip()}
    if str(context.get('session_id') or '').strip():
        session_ids.add(str(context.get('session_id') or '').strip())
    if str(session_id or '').strip() and str(session_id or '').strip() in session_ids:
        return 'session_graph_context'
    if retrieval in {'anchor_match', 'local_1hop', 'local_2hop'}:
        return 'local_graph_neighborhood'
    if source == 'file':
        return 'file_ingested_knowledge'
    if local_neighbor:
        return 'local_graph_neighborhood'
    return 'global_graph_facts'


def _candidate_recency(candidate: ContextCandidate) -> float:
    if candidate.source == 'session_short_term_history':
        return _clamp01(float(candidate.metadata.get('recency_score') or 0.0))
    return {
        'persona_memory': 0.62,
        'persona_triad': 0.58,
        'session_graph_context': 0.68,
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
    if candidate.item_type == 'persona_identity':
        return 0.98
    if candidate.item_type == 'persona_work_profile':
        return 0.95
    if candidate.item_type == 'persona_control':
        return 0.94
    if candidate.item_type in {'persona_form', 'persona_decision_explanation'}:
        return 0.9
    if candidate.item_type == 'persona_reactions':
        return 0.82
    if candidate.item_type == 'persona_knowledge':
        return 0.78
    if candidate.item_type == 'persona_relations':
        return 0.72
    if candidate.item_type == 'persona_examples':
        return 0.62
    if candidate.item_type == 'persona_log_tuples':
        return 0.58
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
        return 48
    if candidate.item_type == 'persona_identity':
        return 160
    if candidate.item_type == 'persona_work_profile':
        return 144
    if candidate.item_type == 'persona_knowledge':
        return 132
    if candidate.item_type == 'persona_examples':
        return 56
    if candidate.item_type == 'persona_log_tuples':
        return 48
    if candidate.item_type == 'persona_relations':
        return 72
    if candidate.item_type == 'persona_reactions':
        return 96
    if candidate.item_type in {'persona_core', 'persona_state', 'persona_form', 'persona_decision_explanation'}:
        return 112
    if candidate.section == 'graph_context':
        return 72
    return 120


def _recent_first_items(values: list[str]) -> list[str]:
    clean = [str(item).strip() for item in list(values or []) if str(item).strip()]
    if not clean:
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    for item in reversed(clean):
        key = normalize_name(item)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


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
    bundle = load_active_persona(name)
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
    bundle = load_active_persona(name)
    if bundle is None:
        return None, []
    fallback_situation = Situation(type='neutral_statement', target='external', severity=0.2, summary='type=neutral_statement; target=external; severity=0.20')
    policy = reaction_policy(bundle, situation if isinstance(situation, (dict, Situation)) else fallback_situation)
    reaction_lines = []
    for item in relevant_reactions(name, situation):
        learned_reaction = item.get('reaction')
        rendered = _render_learned_reaction_line(str(item.get('situation') or ''), str(learned_reaction or ''))
        reaction_lines.append(rendered)

    candidates: list[ContextCandidate] = []

    core_lines = [
        f'You are {bundle.name}.',
        f'Entity type: {bundle.entity_type}.',
        'Answer in first person from this persona head.',
        'React from persona traits and the current situation, not from raw user emotion.',
        'Identity lock: do not invent a different biography, job, worldview, or temperament.',
        'If the available facts are thin, stay within known persona evidence and ask or answer cautiously instead of improvising a new identity.',
    ]
    stance = _render_emotional_stance(bundle.emotion_vector)
    if stance:
        core_lines.append(f'Current stance: {stance}.')
    situation_note = _render_situation_note(situation)
    if situation_note:
        core_lines.append(situation_note)
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

    if bundle.persona_form:
        identity_lines: list[str] = []
        biography = str(bundle.persona_form.get('biography') or '').strip()
        values = [str(item).strip() for item in list(bundle.persona_form.get('values') or []) if str(item).strip()]
        speech_style = [str(item).strip() for item in list(bundle.persona_form.get('speech_style') or []) if str(item).strip()]
        emotional_tendencies = [
            str(item).strip()
            for item in list(bundle.persona_form.get('emotional_tendencies') or [])
            if str(item).strip()
        ]
        conflict_behavior = [
            str(item).strip()
            for item in list(bundle.persona_form.get('conflict_behavior') or [])
            if str(item).strip()
        ]
        trust_model = [str(item).strip() for item in list(bundle.persona_form.get('trust_model') or []) if str(item).strip()]
        work_habits = [str(item).strip() for item in list(bundle.persona_form.get('work_habits') or []) if str(item).strip()]
        prioritized_work_habits = _recent_first_items(work_habits)
        memory_anchors = [
            str(item).strip()
            for item in list(bundle.persona_form.get('memory_anchors') or [])
            if str(item).strip()
        ]
        memories = [
            str(item).strip()
            for item in list(bundle.persona_form.get('memories') or [])
            if str(item).strip()
        ]
        personal_history = [
            str(item).strip()
            for item in list(bundle.persona_form.get('personal_history') or [])
            if str(item).strip()
        ]
        style_markers = [
            str(item).strip()
            for item in list(bundle.persona_form.get('recurring_style_markers') or [])
            if str(item).strip()
        ]
        if biography:
            identity_lines.append(f'Biography: {biography}')
        if values:
            identity_lines.append(f"Values: {' | '.join(values[:5])}.")
        if speech_style:
            identity_lines.append(f"Voice and speech style: {' | '.join(speech_style[:5])}.")
        if emotional_tendencies:
            identity_lines.append(f"Emotional tendencies: {' | '.join(emotional_tendencies[:4])}.")
        if conflict_behavior:
            identity_lines.append(f"Conflict behavior: {' | '.join(conflict_behavior[:4])}.")
        if trust_model:
            identity_lines.append(f"Trust model: {' | '.join(trust_model[:4])}.")
        if prioritized_work_habits:
            identity_lines.append(f"Work habits: {' | '.join(prioritized_work_habits[:5])}.")
        if memory_anchors:
            identity_lines.append(f"Memory anchors: {' | '.join(memory_anchors[:5])}.")
        if memories:
            identity_lines.append(f"Memories: {' | '.join(_recent_first_items(memories)[:4])}.")
        if personal_history:
            identity_lines.append(f"Personal history: {' | '.join(_recent_first_items(personal_history)[:4])}.")
        if style_markers:
            identity_lines.append(f"Recurring style markers: {' | '.join(style_markers[:4])}.")
        if identity_lines:
            identity_text = '\n'.join(identity_lines).strip()
            candidates.append(
                ContextCandidate(
                    candidate_id=f'persona:{normalize_name(bundle.name)}:identity',
                    source='persona_triad',
                    section='persona_block',
                    item_type='persona_identity',
                    title='persona_identity',
                    text=identity_text,
                    token_estimate=_estimate_tokens(identity_text),
                    metadata={'persona_name': bundle.name},
                )
            )

        work_lines: list[str] = []
        work_relations = [
            f"{str(item.get('type') or '').strip()} {str(item.get('target') or '').strip()}"
            for item in bundle.relations
            if str(item.get('type') or '').strip().upper() in {'WORKS_IN', 'SPECIALIZES_IN', 'WORKS_WITH', 'LIVES_IN'}
            and str(item.get('target') or '').strip()
        ]
        decision_patterns = [
            str(item).strip()
            for item in list(bundle.persona_form.get('decision_patterns') or [])
            if str(item).strip()
        ]
        if biography:
            first_sentence = biography.split('. ')[0].strip()
            if first_sentence and not first_sentence.endswith('.'):
                first_sentence += '.'
            if first_sentence:
                work_lines.append(f'Professional role: {first_sentence}')
        if work_relations:
            work_lines.append(f"Work structure: {' | '.join(work_relations[:5])}.")
        if prioritized_work_habits:
            work_lines.append(f"Daily work habits: {' | '.join(prioritized_work_habits[:5])}.")
        if decision_patterns:
            work_lines.append(f"Operational decision pattern: {' | '.join(decision_patterns[:4])}.")
        if work_lines:
            work_text = '\n'.join(work_lines).strip()
            candidates.append(
                ContextCandidate(
                    candidate_id=f'persona:{normalize_name(bundle.name)}:work_profile',
                    source='persona_triad',
                    section='persona_block',
                    item_type='persona_work_profile',
                    title='persona_work_profile',
                    text=work_text,
                    token_estimate=_estimate_tokens(work_text),
                    metadata={'persona_name': bundle.name},
                )
            )

    state_lines: list[str] = []
    if bundle.traits:
        state_lines.append(f"Traits: {', '.join(bundle.traits[:8])}.")
    if bundle.emotion_vector:
        state_lines.append(f'Current stance qualities: {_render_emotional_stance(bundle.emotion_vector)}.')
    if bundle.indicators is not None:
        state_lines.append(
            'Adaptation state: '
            f"{bundle.indicators.maturity_level}, "
            f"reviewable, "
            f"drift_locked={bundle.indicators.adaptation_locked}."
        )
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

    structured = bundle.structured_persona
    if structured is not None:
        control_lines: list[str] = []
        if str(structured.core_goal or '').strip():
            control_lines.append(f'Core goal: {structured.core_goal}.')
        if list(structured.secondary_goals or []):
            control_lines.append(
                f"Secondary goals: {' | '.join(str(item).strip() for item in list(structured.secondary_goals or [])[:4] if str(item).strip())}."
            )
        if list(structured.constraints_internal or []):
            control_lines.append(
                f"Internal constraints: {' | '.join(str(item).strip() for item in list(structured.constraints_internal or [])[:4] if str(item).strip())}."
            )
        if list(structured.constraints_social or []):
            control_lines.append(
                f"Social constraints: {' | '.join(str(item).strip() for item in list(structured.constraints_social or [])[:4] if str(item).strip())}."
            )
        if list(structured.allowed_methods or []):
            control_lines.append(
                f"Allowed methods: {' | '.join(str(item).strip() for item in list(structured.allowed_methods or [])[:4] if str(item).strip())}."
            )
        if list(structured.maladaptive_methods or []):
            control_lines.append(
                f"Maladaptive methods: {' | '.join(str(item).strip() for item in list(structured.maladaptive_methods or [])[:4] if str(item).strip())}."
            )
        if control_lines:
            control_text = '\n'.join(control_lines).strip()
            candidates.append(
                ContextCandidate(
                    candidate_id=f'persona:{normalize_name(bundle.name)}:control',
                    source='persona_memory',
                    section='persona_block',
                    item_type='persona_control',
                    title='persona_control',
                    text=control_text,
                    token_estimate=_estimate_tokens(control_text),
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
            existing_context = dict(merged.get(key) or {})
            incoming_context = dict(value or {})
            merged_context = {**existing_context, **incoming_context}
            existing_source = str(existing_context.get('source') or '').strip().lower()
            incoming_source = str(incoming_context.get('source') or '').strip().lower()
            if existing_source in {'file', 'session'} and incoming_source in {'head', 'persona_graph'}:
                merged_context['source'] = existing_source
            merged[key] = merged_context
            continue
        if merged.get(key) in (None, '', [], {}) and value not in (None, '', [], {}):
            merged[key] = value
    return merged


def _build_graph_candidates(
    *,
    query: str,
    session_id: str,
    resolved_persona: str,
    current_entity: str,
    store: GraphStore,
) -> tuple[list[ContextCandidate], list[dict[str, Any]], list[dict[str, Any]]]:
    graph_native = store.graph_native_search(
        query,
        anchor_names=[item for item in (resolved_persona, current_entity) if str(item).strip()],
        session_id=session_id,
        limit=6,
    )
    local_graph = load_persona_graph(resolved_persona) if resolved_persona else {'nodes': [], 'edges': []}
    global_by_name = {
        normalize_name(str(node.get('name') or node.get('id') or '')): dict(node)
        for node in store.load_nodes()
        if normalize_name(str(node.get('name') or node.get('id') or ''))
    }

    entries: dict[str, dict[str, Any]] = {}
    for entry in list(graph_native.get('entries') or []):
        node = dict(entry.get('node') or {})
        retrieval_source = str(entry.get('retrieval_source') or '').strip()
        node_id = str(node.get('id') or '')
        if not node_id:
            continue
        entries[node_id] = {
            'node': dict(node),
            'local_neighbor': retrieval_source in {'anchor_match', 'local_1hop', 'local_2hop'},
            'retrieval_source': retrieval_source or 'semantic_dense',
        }
    for node in list(local_graph.get('nodes') or []):
        node_id = str(node.get('id') or '')
        if not node_id:
            continue
        global_match = global_by_name.get(normalize_name(str(node.get('name') or node_id)))
        if global_match is not None:
            node = _merge_node_payload(dict(global_match), dict(node))
        if node_id in entries:
            entries[node_id]['node'] = _merge_node_payload(entries[node_id]['node'], dict(node))
            entries[node_id]['local_neighbor'] = True
            entries[node_id]['retrieval_source'] = str(entries[node_id].get('retrieval_source') or 'persona_graph')
        else:
            entries[node_id] = {
                'node': dict(node),
                'local_neighbor': True,
                'retrieval_source': 'persona_graph',
            }

    collapsed_entries: dict[str, dict[str, Any]] = {}
    for payload in entries.values():
        node = dict(payload['node'])
        collapse_key = normalize_name(str(node.get('name') or node.get('id') or ''))
        if not collapse_key:
            continue
        if collapse_key in collapsed_entries:
            collapsed_entries[collapse_key]['node'] = _merge_node_payload(collapsed_entries[collapse_key]['node'], node)
            collapsed_entries[collapse_key]['local_neighbor'] = bool(collapsed_entries[collapse_key].get('local_neighbor')) or bool(
                payload.get('local_neighbor')
            )
            existing_retrieval = str(collapsed_entries[collapse_key].get('retrieval_source') or '').strip()
            incoming_retrieval = str(payload.get('retrieval_source') or '').strip()
            if existing_retrieval in {'semantic_dense', 'structural_salience'} and incoming_retrieval:
                collapsed_entries[collapse_key]['retrieval_source'] = incoming_retrieval
        else:
            collapsed_entries[collapse_key] = {
                'node': node,
                'local_neighbor': bool(payload.get('local_neighbor')),
                'retrieval_source': str(payload.get('retrieval_source') or '').strip(),
            }
    entries = {
        str(payload['node'].get('id') or key): payload
        for key, payload in collapsed_entries.items()
    }

    all_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for edge in list(graph_native.get('edges') or []) + list(local_graph.get('edges') or []):
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
        node_context = dict(node.get('context') or {})
        node_source = str(node_context.get('source') or '').strip().lower()
        if (
            node_source == 'chat'
            and not str(node.get('description') or '').strip()
            and not facts
            and int(node.get('frequency') or 0) <= 1
        ):
            continue
        retrieval_source = str(payload.get('retrieval_source') or '').strip()
        preview_lines = [
            f"{node.get('name')} [{node.get('type')}] importance={node.get('importance')} confidence={node.get('confidence')} frequency={node.get('frequency')}.",
        ]
        if retrieval_source:
            preview_lines.append(f"retrieval: {retrieval_source}")
        if str(node.get('translation_line') or '').strip():
            preview_lines.append(f"translation: {node.get('translation_line')}")
        if str(node.get('description') or '').strip():
            preview_lines.append(str(node.get('description') or '').strip())
        if facts:
            preview_lines.append(f"facts: {' | '.join(facts[:4])}")
        preview = '\n'.join(preview_lines).strip()
        source = _context_source_for_node(
            node,
            session_id=session_id,
            local_neighbor=bool(payload.get('local_neighbor')),
            retrieval_source=retrieval_source,
        )
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
                    'retrieval_source': retrieval_source,
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
    social_role: SocialRoleDecision | None,
    mood_report: MoodResearchReport | None,
    analysis: MessageAnalysis | None,
) -> dict[str, Any]:
    clipped_question = _clip(question, _context_config()[1])
    current_entity = str((analysis.primary_entity if analysis is not None else '') or infer_current_entity(session_id) or '').strip()
    resolved = infer_persona_name(question, selected_name=selected_persona, current_entity=current_entity)
    if resolved and not persona_is_registered(resolved):
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
    if social_role is not None:
        role_text = '\n'.join(
            item
            for item in (
                f'Selected social role: {social_role.role}.',
                f'Role reason: {social_role.reason}',
                f"Role evidence: {' | '.join(social_role.evidence[:4])}." if social_role.evidence else '',
                f"Persona graph anchors: {' | '.join(social_role.graph_anchors[:4])}." if social_role.graph_anchors else '',
            )
            if str(item).strip()
        )
        persona_candidates.append(
            ContextCandidate(
                candidate_id=f'persona:{normalize_name(resolved)}:social_role',
                source='social_role',
                section='persona_block',
                item_type='persona_social_role',
                title='persona_social_role',
                text=role_text,
                token_estimate=_estimate_tokens(role_text),
                metadata={'role': social_role.role, 'persona_name': resolved},
            )
        )
    if mood_report is not None and mood_report.snapshot_count > 0:
        transition_preview = ' | '.join(
            f"{item.get('from')} -> {item.get('to')}"
            for item in list(mood_report.transition_counts or [])[:3]
        )
        mood_text = '\n'.join(
            item
            for item in (
                f'Recent mood pattern: {mood_report.latest_cluster_label}.' if mood_report.latest_cluster_label else '',
                f"Observed role effects: {' | '.join(str(item.get('role') or '') for item in list(mood_report.role_effects or [])[:4] if str(item.get('role') or '').strip())}."
                if mood_report.role_effects
                else '',
                f'Key transition patterns: {transition_preview}.'
                if transition_preview
                else '',
            )
            if str(item).strip()
        )
        if mood_text:
            persona_candidates.append(
                ContextCandidate(
                    candidate_id=f'persona:{normalize_name(resolved)}:mood_dynamics',
                    source='mood_research',
                    section='persona_block',
                    item_type='persona_mood_dynamics',
                    title='persona_mood_dynamics',
                    text=mood_text,
                    token_estimate=_estimate_tokens(mood_text),
                    metadata={'persona_name': resolved},
                )
            )
    history_candidates = _build_history_candidates(session_id)
    graph_candidates, graph_nodes, graph_edges = _build_graph_candidates(
        query=query,
        session_id=session_id,
        resolved_persona=resolved,
        current_entity=current_entity,
        store=store,
    )
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
    source_counts = Counter(candidate.source for candidate in ranked)
    for source_name in _SOURCE_PRIORITY:
        source_counts.setdefault(source_name, 0)
    diagnostics = {
        'weights': dict(_CONTEXT_SCORE_WEIGHTS),
        'source_counts': dict(source_counts),
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
    return _pack_persona_block_for_focus(candidates, token_budget=token_budget, question_focus=set())


def _pack_persona_block_for_focus(
    candidates: list[ContextCandidate],
    *,
    token_budget: int,
    question_focus: set[str],
) -> tuple[str, list[ContextCandidate]]:
    focus = set(question_focus or set())
    focus_render_order = dict(_PERSONA_RENDER_ORDER)
    if 'work' in focus:
        focus_render_order.update(
            {
                'persona_core': 0,
                'persona_work_profile': 1,
                'persona_relations': 2,
                'persona_knowledge': 3,
                'persona_identity': 4,
                'persona_decision_explanation': 5,
                'persona_form': 6,
            }
        )
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            focus_render_order.get(candidate.item_type, 99),
            candidate.rank,
            candidate.candidate_id,
        ),
    )
    suppressed: set[str] = set()
    max_items: int | None = None
    if focus & {'identity', 'work', 'memory'}:
        suppressed.update({'persona_reactions', 'persona_log_tuples'})
    if focus & {'work', 'memory'}:
        max_items = 6
    if focus == {'work'}:
        suppressed.add('persona_state')
    if focus == {'decision'}:
        suppressed.update({'persona_examples', 'persona_log_tuples'})
    selected: list[ContextCandidate] = []
    used = 0
    for candidate in ordered:
        if candidate.item_type in suppressed:
            continue
        if max_items is not None and len(selected) >= max_items:
            continue
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
    question_focus: set[str],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[ContextCandidate]]:
    ranked = sorted(candidates, key=lambda candidate: (candidate.rank, candidate.candidate_id))
    has_persona_aligned_graph = any(candidate.score.persona_alignment >= 0.5 for candidate in ranked)
    selected_candidates: list[ContextCandidate] = []
    selected_nodes: list[dict[str, Any]] = []
    node_rank_map: dict[str, int] = {}
    graph_context = ''
    selected_edges: list[dict[str, Any]] = []
    bare_node_count = 0
    max_nodes = 3 if question_focus & {'identity', 'work', 'memory'} else 5
    for candidate in ranked:
        if has_persona_aligned_graph and candidate.score.persona_alignment < 0.2 and selected_candidates:
            continue
        node = dict(candidate.metadata.get('node') or {})
        node_id = str(node.get('id') or '')
        if not node_id or node_id in node_rank_map:
            continue
        if node_id.startswith('example:') or node_id.startswith('log:'):
            continue
        if (
            has_persona_aligned_graph
            and len(selected_nodes) >= 2
            and str(node.get('id') or '').startswith('trait:')
            and int(candidate.metadata.get('degree') or 0) <= 1
        ):
            continue
        if not str(node.get('description') or '').strip() and not list(node.get('facts') or []):
            if len(selected_nodes) >= 3 and candidate.score.relevance < 0.09:
                continue
            if bare_node_count >= 3:
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
        if not str(node.get('description') or '').strip() and not list(node.get('facts') or []):
            bare_node_count += 1
        if len(selected_nodes) >= max_nodes:
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
    question_focus: set[str],
) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    max_context_tokens, question_budget, prompt_overhead_tokens, section_budgets, _ = _context_config()
    persona_candidates = [candidate for candidate in candidates if candidate.section == 'persona_block']
    graph_candidates = [candidate for candidate in candidates if candidate.section == 'graph_context']
    history_candidates = [candidate for candidate in candidates if candidate.section == 'recent_dialogue']

    persona_block, selected_persona = _pack_persona_block_for_focus(
        persona_candidates,
        token_budget=section_budgets['persona_block'],
        question_focus=question_focus,
    )
    graph_context, selected_nodes, selected_edges, selected_graph = _pack_graph_context(
        graph_candidates,
        all_edges=all_edges,
        token_budget=section_budgets['graph_context'],
        question_focus=question_focus,
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
    social_role: SocialRoleDecision | None = None,
    mood_report: MoodResearchReport | None = None,
    analysis: MessageAnalysis | None = None,
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
        social_role=social_role,
        mood_report=mood_report,
        analysis=analysis,
    )

    ranked, score_debug = _score_rank_compress(
        candidates=list(collected['candidates']),
        question=collected['clipped_question'],
        explicit_context=explicit_context,
        current_entity=collected['current_entity'],
        rendered_situation=collected['rendered_situation'],
        persona_bundle=collected['persona_bundle'],
    )
    semantic_focus = infer_semantic_focus(
        question=collected['clipped_question'],
        persona_bundle=collected['persona_bundle'],
        analysis=analysis,
        situation=analysis.situation if analysis is not None else (situation if isinstance(situation, Situation) else None),
    )
    question_focus = set(str(item).strip() for item in list(semantic_focus.get('focus') or []) if str(item).strip())
    packed_sections, selected_nodes, selected_edges, pack_debug = _pack_context(
        candidates=ranked,
        all_edges=list(collected['graph_edges']),
        question_focus=question_focus,
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
            'question_focus': sorted(question_focus),
            'semantic_focus': dict(semantic_focus),
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
