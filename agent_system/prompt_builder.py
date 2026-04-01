from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .language_tools import language_label, normalize_language_code
from .semantic_routing import infer_semantic_focus, render_semantic_focus_guidance


_PROMPTS_DIR = Path(__file__).resolve().parent / 'prompts'


@lru_cache(maxsize=32)
def load_prompt_stage(name: str) -> str:
    path = _PROMPTS_DIR / f'{str(name or "").strip()}.md'
    if not path.exists():
        return ''
    return path.read_text(encoding='utf-8').strip()


def _staged_prompt_header(stage_name: str) -> str:
    shared = load_prompt_stage('shared_rules')
    stage = load_prompt_stage(stage_name)
    return '\n\n'.join(part for part in (shared, stage) if str(part).strip())


def _truncate_tokens_equivalent(text: str, max_tokens_equivalent: int) -> str:
    raw = str(text or '').strip()
    if not raw:
        return ''
    max_chars = max_tokens_equivalent * 4
    return raw[:max_chars].strip() if len(raw) > max_chars else raw


def _reply_shape_guidance(question: str) -> list[str]:
    normalized = ' '.join(str(question or '').strip().lower().split())
    if not normalized:
        return ['Keep the reply brief by default: 2 to 4 sentences unless the user explicitly asks for detail.']
    if any(marker in normalized for marker in ('one sentence', 'single sentence')):
        return ['Reply in exactly one sentence.']
    if any(marker in normalized for marker in ('two sentences', '2 sentences', 'two short sentences', '2 short sentences')):
        return ['Reply in exactly two short sentences.']
    if any(marker in normalized for marker in ('briefly', 'brief', 'short answer', 'concise', 'in short')):
        return ['Keep the reply concise: no more than 3 short sentences.']
    if any(marker in normalized for marker in ('in detail', 'detailed', 'walk me through', 'long answer', 'explain fully')):
        return ['The user explicitly wants depth. Give a fuller answer, but stay concrete and in character.']
    return ['Keep the reply brief by default: 2 to 4 sentences unless the user explicitly asks for detail.']


def build_chat_prompt(
    *,
    question: str,
    internal_question: str = '',
    persona_block: str = '',
    social_role_block: str = '',
    mood_research_block: str = '',
    graph_context: str = '',
    recent_dialogue: str = '',
    state_transition_block: str = '',
    reviewed_context_block: str = '',
    response_shaping_block: str = '',
    language: str = 'en',
    semantic_focus: dict[str, Any] | None = None,
) -> str:
    reply_language = language_label(normalize_language_code(language, fallback='en'))
    effective_focus = dict(semantic_focus or infer_semantic_focus(question=str(internal_question or question or '')))
    blocks = [
        _staged_prompt_header('final_generator'),
        f'Respond in {reply_language}.',
        'Generate the final reply from the reviewed current context, not from the raw user message alone.',
        'Answer as the persona in first person, not as an assistant or helpdesk.',
        'The persona is the speaker and the user is speaking to that person.',
        'Answer the user question directly and concretely before adding any extra explanation.',
        *_reply_shape_guidance(question),
        'Prefer lived biography, work habits, relations, values, and memory anchors over slogans.',
        'Use the selected social role and the current situation without mirroring the user emotion directly.',
        'Stay inside known persona facts; if a fact is uncertain, say so plainly instead of inventing a new identity.',
        'Use graph context only as support. If it conflicts with persona identity, prefer the persona block.',
        'Do not output analysis, hidden reasoning, `<think>` tags, system notes, prompt commentary, or JSON.',
        'Do not mention being an AI, assistant, language model, system, or following instructions.',
    ]
    blocks.extend(render_semantic_focus_guidance(effective_focus))
    blocks.extend(['User question:', _truncate_tokens_equivalent(str(question or '').strip(), 420)])
    if state_transition_block:
        blocks.extend(['State transition:', _truncate_tokens_equivalent(state_transition_block, 220)])
    if response_shaping_block:
        blocks.extend(['Response shaping:', _truncate_tokens_equivalent(response_shaping_block, 220)])
    if reviewed_context_block:
        blocks.extend(['Reviewed current context:', _truncate_tokens_equivalent(reviewed_context_block, 720)])
    if persona_block:
        blocks.extend(['Persona head:', _truncate_tokens_equivalent(persona_block, 760)])
    if social_role_block:
        blocks.extend(['Interaction role:', _truncate_tokens_equivalent(social_role_block, 120)])
    if mood_research_block:
        blocks.extend(['Mood research:', _truncate_tokens_equivalent(mood_research_block, 96)])
    if graph_context:
        blocks.extend(['Knowledge graph:', _truncate_tokens_equivalent(graph_context, 520)])
    if recent_dialogue:
        blocks.extend(['Recent dialogue:', _truncate_tokens_equivalent(recent_dialogue, 220)])
    return '\n\n'.join(block for block in blocks if str(block).strip())


def build_state_reader_prompt(*, state_payload: dict[str, Any]) -> str:
    return '\n\n'.join(
        [
            _staged_prompt_header('state_reader'),
            'Return valid JSON only.',
            'Schema:',
            '{"summary":"...","active_layers":["persona_baseline"],"graph_anchors":["triage"],"memory_anchors":["father\'s watch"],"priorities":["answer_substance"],"risks":["weak_grounding"],"constraints":["stay_in_first_person"]}',
            'Current active state payload:',
            json.dumps(dict(state_payload or {}), ensure_ascii=False, indent=2),
        ]
    )


def build_influence_interpreter_prompt(
    *,
    previous_state: dict[str, Any],
    message: str,
    analysis: dict[str, Any],
    situation: dict[str, Any],
) -> str:
    return '\n\n'.join(
        [
            _staged_prompt_header('influence_interpreter'),
            'Return valid JSON only.',
            'Schema:',
            '{"summary":"...","activation":["work"],"pressure_points":["uncertainty"],"themes":["decision"],"conflicts":["needs_precision_without_grounding"],"direction":"narrow_then_answer","tension":"moderate","role_pressure":"mentor","uncertainty_level":"moderate","risk_level":"low"}',
            'Previous state:',
            json.dumps(dict(previous_state or {}), ensure_ascii=False, indent=2),
            'User message:',
            str(message or '').strip(),
            'Analysis:',
            json.dumps(dict(analysis or {}), ensure_ascii=False, indent=2),
            'Situation:',
            json.dumps(dict(situation or {}), ensure_ascii=False, indent=2),
        ]
    )


def build_state_transition_prompt(
    *,
    previous_state: dict[str, Any],
    influence: dict[str, Any],
    updated_state_seed: dict[str, Any],
) -> str:
    return '\n\n'.join(
        [
            _staged_prompt_header('state_transition_guide'),
            'Return valid JSON only.',
            'Schema:',
            '{"summary":"...","changed":["dynamic_state"],"unchanged":["baseline_definition"],"priorities":["answer_substance"],"risks":["weak_grounding"],"constraints":["stay_in_first_person","preserve_identity_invariants"]}',
            'Previous state:',
            json.dumps(dict(previous_state or {}), ensure_ascii=False, indent=2),
            'Interpreted influence:',
            json.dumps(dict(influence or {}), ensure_ascii=False, indent=2),
            'Updated state seed:',
            json.dumps(dict(updated_state_seed or {}), ensure_ascii=False, indent=2),
        ]
    )


def build_context_curator_prompt(*, updated_state: dict[str, Any], working_context_seed: dict[str, Any]) -> str:
    return '\n\n'.join(
        [
            _staged_prompt_header('context_curator'),
            'Return valid JSON only.',
            'Schema:',
            '{"summary":"...","important_items":["persona_identity","work_profile"],"priorities":["answer_substance"],"constraints":["stay_in_first_person"],"risks":["weak_grounding"]}',
            'Updated state:',
            json.dumps(dict(updated_state or {}), ensure_ascii=False, indent=2),
            'Working context seed:',
            json.dumps(dict(working_context_seed or {}), ensure_ascii=False, indent=2),
        ]
    )


def build_context_reviewer_prompt(*, working_context: dict[str, Any]) -> str:
    return '\n\n'.join(
        [
            _staged_prompt_header('context_reviewer'),
            'Return valid JSON only.',
            'Schema:',
            '{"summary":"...","important_items":["persona_identity"],"weak_items":["bare_graph_node"],"contradictions":["none"],"risks":["weak_grounding"],"constraints":["answer_from_reviewed_context_only"]}',
            'Working context:',
            json.dumps(dict(working_context or {}), ensure_ascii=False, indent=2),
        ]
    )


def build_response_shaper_prompt(
    *,
    reviewed_context: dict[str, Any],
    influence: dict[str, Any],
    social_role: dict[str, Any],
    response_explanation: dict[str, Any],
) -> str:
    return '\n\n'.join(
        [
            _staged_prompt_header('response_shaper'),
            'Return valid JSON only.',
            'Schema:',
            '{"role":"mentor","style":"direct_explanatory","behavior_mode":"grounded_answer","summary":"...","priorities":["answer_substance"],"constraints":["stay_in_first_person","avoid_assistant_tone"],"risk_posture":"measured"}',
            'Reviewed context:',
            json.dumps(dict(reviewed_context or {}), ensure_ascii=False, indent=2),
            'Influence interpretation:',
            json.dumps(dict(influence or {}), ensure_ascii=False, indent=2),
            'Selected social role:',
            json.dumps(dict(social_role or {}), ensure_ascii=False, indent=2),
            'Persona response explanation:',
            json.dumps(dict(response_explanation or {}), ensure_ascii=False, indent=2),
        ]
    )


def build_entity_extraction_prompt(text: str, *, source: str = 'session') -> str:
    return '\n\n'.join(
        [
            'Return valid JSON only.',
            'Schema:',
            '{"entities":[{"name":"Dracula","aliases":["Count Dracula"],"description":"Fictional vampire nobleman.","facts":["Dracula feeds on humans."],"context":{"source":"file"}}],"relations":[{"from":"Dracula","to":"humans","type":"FEEDS_ON","weight":0.9}]}',
            'Rules:',
            '- entities must be canonical concepts or entities, not fragments of sentences.',
            '- entity names should usually be noun phrases of 1 to 4 words.',
            '- never use whole clauses, propositions, quoted sentences or verb-heavy phrases as entity names.',
            '- for structural topics prefer concept nodes such as Biology, Psychology, Sociology, Language, Culture.',
            '- Use concise fact-like descriptions.',
            '- descriptions may explain what the concept is and what it does.',
            '- facts should be short strings.',
            '- relations must contain from, to, type.',
            '- if unsure return {"entities":[],"relations":[]}.',
            f'Source: {source}',
            'Text:',
            str(text or '').strip(),
        ]
    )


def build_persona_profile_prompt(
    name: str,
    excerpts: list[str],
    *,
    reason: str = '',
    log_tuples: list[dict[str, Any]] | None = None,
    current_form: dict[str, Any] | None = None,
    current_summary: str = '',
) -> str:
    rendered = '\n'.join(f'- {item}' for item in excerpts[:8] if str(item).strip()) or '- none'
    rendered_logs = '\n'.join(
        f"- tuple={tuple(item.get('tuple') or ())}; frequency={item.get('frequency')}; sample={item.get('sample')}"
        for item in list(log_tuples or [])[:12]
        if isinstance(item, dict)
    ) or '- none'
    rendered_form = json.dumps(dict(current_form or {}), ensure_ascii=False, indent=2) if current_form else '{}'
    return '\n\n'.join(
        [
            'Return valid JSON only.',
            'Schema:',
            '{"persona_payload":{"name":"dracula","entity_type":"FICTIONAL_CHARACTER","traits":["vampire","aristocratic"],"aliases":["Count Dracula"],"emotion_vector":{"anger":0.2,"fear":0.1,"curiosity":0.5,"confidence":0.9,"empathy":0.2},"examples":["I prefer the night."],"relations":[{"type":"FEEDS_ON","target":"humans"}],"knowledge":"Short persona knowledge summary."},"persona_form":{"identity_class":"fictional_character","interaction_style":["formal","controlled"],"core_dispositions":["aristocratic","predatory"],"decision_patterns":["checks role and situation before replying"],"clarification_policy":"Ask a clarifying question when the request is underspecified.","sarcasm_profile":"low","response_priorities":["answer_substance","clarify_if_underspecified","stay_in_character"],"knowledge_domains":["night","predation"],"risk_controls":["do_not_mirror_user_emotion","stay_grounded_in_context"]},"decision_explanation":"Dracula first checks who is addressed and what situation is active, then answers in character using controlled confidence instead of mirroring the user emotion."}',
            'Rules:',
            '- persona_payload contains only content, never imperative commands.',
            '- persona_form must be explicit, structured, and concise.',
            '- decision_explanation must be short and plain enough for a non-expert reader.',
            '- use the log tuples as compressed evidence of repeated patterns.',
            f'Persona name: {name}',
            f'Reason: {reason}',
            'Current persona form:',
            rendered_form,
            f'Current summary: {current_summary}',
            'Log tuples:',
            rendered_logs,
            'Relevant excerpts:',
            rendered,
        ]
    )


def build_node_rethink_prompt(
    *,
    node: dict[str, Any],
    node_view: dict[str, Any],
    graph_context: str,
    context_budget: int = 4000,
    max_new_entities: int = 4,
    max_new_links: int = 4,
    allowed_roles: list[str] | tuple[str, ...] = (),
) -> str:
    who = dict(node_view.get('who_or_what') or {})
    what = dict(node_view.get('what_is_it_like') or {})
    links = list(node_view.get('how_it_acts') or [])
    rendered_links = '\n'.join(
        f"- {edge.get('from')} {edge.get('type')} {edge.get('to')} (weight={edge.get('weight')})"
        for edge in links[:10]
    ) or '- none'
    allowed_role_list = ', '.join(str(role).strip() for role in allowed_roles if str(role).strip()) or 'related_to'
    return '\n\n'.join(
        [
            'Return valid JSON only.',
            'Goal: review one graph node and explain what should be improved in its content and conceptual neighborhood.',
            'The model must not issue raw graph mutations, imperative commands, or arbitrary relation types.',
            'Suggest only content improvements and canonical candidate nodes. The system will apply changes deterministically.',
            'Prefer concise concept nodes such as Planet, Life, Human, Biology, Energy, Atmosphere.',
            'Never create sentence fragments as entity names.',
            'Write node_improvement.description and node_improvement.plain_explanation in English.',
            f'You may suggest at most {int(max_new_entities)} candidate nodes and {int(max_new_links)} link suggestions.',
            f'Allowed link roles: {allowed_role_list}.',
            'Schema:',
            '{"node_improvement":{"description":"Short academic description.","plain_explanation":"Longer plain explanation.","facts":["fact 1"],"capabilities":["what it can do"],"mechanisms":["how it works"],"reinterpretation_form":{"who_or_what":"...","what_can_it_do":"...","how_does_it_work":"...","why_it_matters":"...","suggested_links":["Life","Planet"]}},"link_suggestions":[{"name":"Life","role":"makes_possible","why":"Sunlight enables life.","description":"Life as an organized biological process.","facts":["Life depends on stable energy flows."],"aliases":["biological life"]}]}',
            f'Context budget: {int(context_budget or 4000)}',
            f"Node name: {node.get('name')}",
            f"Node type: {node.get('type')}",
            f"Translation line: {node.get('translation_line') or ''}",
            f"Current description: {node.get('description') or ''}",
            f"Current facts: {' | '.join(str(item).strip() for item in list(node.get('facts') or []) if str(item).strip())}",
            f"Identity block: {who}",
            f"Current node form: {what}",
            'Current relations:',
            rendered_links,
            'Neighborhood graph context:',
            _truncate_tokens_equivalent(graph_context, max(200, int(context_budget / 2.2))),
        ]
    )


def render_graph_context(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    node_map = {str(node.get('id') or ''): node for node in nodes}
    lines: list[str] = []
    for node in nodes[:10]:
        aliases = list(node.get('aliases') or [])
        alias_line = f" aliases={', '.join(str(item) for item in aliases[:4])}" if aliases else ''
        lines.append(
            f"- {node.get('name')} [{node.get('type')}] importance={node.get('importance')} confidence={node.get('confidence')} frequency={node.get('frequency')}.{alias_line}"
        )
        if str(node.get('translation_line') or '').strip():
            lines.append(f"  translation: {node.get('translation_line')}")
        if str(node.get('description') or '').strip():
            lines.append(f"  {node.get('description')}")
        facts = [str(item).strip() for item in list(node.get('facts') or []) if str(item).strip()]
        if facts:
            lines.append(f"  facts: {' | '.join(facts[:4])}")
    for edge in edges[:14]:
        src = node_map.get(str(edge.get('from') or ''), {}).get('name') or edge.get('from')
        dst = node_map.get(str(edge.get('to') or ''), {}).get('name') or edge.get('to')
        lines.append(f'- relation: {src} {edge.get("type")} {dst} (weight={edge.get("weight")})')
    return _truncate_tokens_equivalent('\n'.join(lines).strip(), 2200)
