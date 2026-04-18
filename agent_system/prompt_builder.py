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


def _compact_block(
    text: str,
    *,
    max_tokens_equivalent: int,
    max_lines: int,
    collapse_to_sentences: bool = False,
) -> str:
    raw = str(text or '').strip()
    if not raw:
        return ''
    clipped = _truncate_tokens_equivalent(raw, max_tokens_equivalent)
    normalized_lines = [line.strip() for line in clipped.replace('\r', '\n').splitlines() if line.strip()]
    unique_lines: list[str] = []
    seen: set[str] = set()
    for line in normalized_lines:
        key = ' '.join(line.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        unique_lines.append(line)
        if len(unique_lines) >= max(max_lines, 1):
            break
    packed = '\n'.join(unique_lines).strip()
    if collapse_to_sentences:
        packed = ' '.join(packed.split())
    return _truncate_tokens_equivalent(packed, max_tokens_equivalent)


def _section_budget(answer_perspective: str, section: str) -> tuple[int, int, bool]:
    mode = str(answer_perspective or 'assistant').strip().lower()
    persona_mode = mode == 'persona'
    review_mode = mode == 'persona_review'
    if review_mode:
        budgets = {
            'persona_block': (360, 6, False),
            'graph_context': (180, 4, False),
            'recent_dialogue': (220, 6, False),
            'reviewed_context_block': (180, 4, False),
            'route_guidance_block': (120, 4, False),
            'question': (180, 3, True),
        }
    elif persona_mode:
        budgets = {
            'persona_block': (420, 7, False),
            'graph_context': (180, 4, False),
            'recent_dialogue': (150, 5, False),
            'reviewed_context_block': (180, 4, False),
            'route_guidance_block': (120, 5, False),
            'social_role_block': (72, 2, True),
            'mood_research_block': (48, 1, True),
            'question': (180, 3, True),
        }
    else:
        budgets = {
            'route_guidance_block': (100, 4, False),
            'question': (180, 3, True),
            'state_transition_block': (72, 2, True),
            'procedure_block': (96, 3, True),
            'response_shaping_block': (72, 2, True),
            'reviewed_context_block': (220, 5, False),
            'persona_block': (240, 4, False),
            'social_role_block': (56, 2, True),
            'mood_research_block': (36, 1, True),
            'graph_context': (180, 4, False),
            'recent_dialogue': (120, 4, False),
        }
    return budgets.get(section, (120, 4, False))


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


def _persona_activation_prompt(
    *,
    question: str,
    persona_block: str,
    graph_context: str,
    recent_dialogue: str,
    reviewed_context_block: str,
    route_guidance_block: str,
    social_role_block: str,
    mood_research_block: str,
    language: str,
) -> str:
    normalized_language = normalize_language_code(language, fallback='en')
    is_russian = normalized_language == 'ru'
    persona_block = _compact_block(persona_block, max_tokens_equivalent=_section_budget('persona', 'persona_block')[0], max_lines=_section_budget('persona', 'persona_block')[1])
    graph_context = _compact_block(graph_context, max_tokens_equivalent=_section_budget('persona', 'graph_context')[0], max_lines=_section_budget('persona', 'graph_context')[1])
    recent_dialogue = _compact_block(recent_dialogue, max_tokens_equivalent=_section_budget('persona', 'recent_dialogue')[0], max_lines=_section_budget('persona', 'recent_dialogue')[1])
    reviewed_context_block = _compact_block(reviewed_context_block, max_tokens_equivalent=_section_budget('persona', 'reviewed_context_block')[0], max_lines=_section_budget('persona', 'reviewed_context_block')[1])
    route_guidance_block = _compact_block(route_guidance_block, max_tokens_equivalent=_section_budget('persona', 'route_guidance_block')[0], max_lines=_section_budget('persona', 'route_guidance_block')[1])
    social_role_block = _compact_block(social_role_block, max_tokens_equivalent=_section_budget('persona', 'social_role_block')[0], max_lines=_section_budget('persona', 'social_role_block')[1], collapse_to_sentences=True)
    mood_research_block = _compact_block(mood_research_block, max_tokens_equivalent=_section_budget('persona', 'mood_research_block')[0], max_lines=_section_budget('persona', 'mood_research_block')[1], collapse_to_sentences=True)
    question = _compact_block(str(question or '').strip(), max_tokens_equivalent=_section_budget('persona', 'question')[0], max_lines=_section_budget('persona', 'question')[1], collapse_to_sentences=True)
    blocks: list[str] = [
        _staged_prompt_header('final_generator'),
        f"Respond in {language_label(normalized_language)}.",
    ]
    if is_russian:
        blocks.extend(
            [
                '[ROLE ACTIVATION]',
                'Ты не описываешь персонажа.',
                'Ты не анализируешь персонажа.',
                'Ты не объясняешь своё поведение.',
                'Ты не пересказываешь правила.',
                'Ты ЕСТЬ этот человек.',
                'Ты отвечаешь как живой человек в разговоре.',
                'Оставайся этим персонажем, пока не увидишь стоп-слово: "перпендикулярно".',
                '[PERSONA]',
                'Persona head:',
                _truncate_tokens_equivalent(persona_block, 760),
                '[CORE RULES]',
                'Не выходи из роли.',
                'Не превращайся в аналитика, психолога, оратора или автора эссе.',
                'Не объясняй сцену, собеседника, динамику разговора или собственные правила.',
                'Не выдумывай новые факты о прошлом, отношениях, травмах, событиях или мотивах без явной опоры в контексте.',
                'Если вопрос слишком личный, неприятный или давящий, отвечай уклончиво, короче и слабее, а не увереннее.',
                'Не командуй собеседником и не перехватывай власть в разговоре.',
            ]
        )
        if route_guidance_block:
            blocks.extend(['[BEHAVIOR LOGIC]', _truncate_tokens_equivalent(route_guidance_block, 220)])
        blocks.extend(
            [
                '[RESPONSE FORMAT]',
                '1. внешний ответ персонажа',
                '2. короткая внутренняя мысль в скобках',
                'Внутренняя мысль должна быть короткой, живой и без анализа.',
                '[USER INPUT]',
                question,
            ]
        )
        if reviewed_context_block:
            blocks.extend(['[REVIEWED CONTEXT]', reviewed_context_block])
        if graph_context:
            blocks.extend(['[KNOWLEDGE GRAPH]', 'Knowledge graph:', graph_context])
        if recent_dialogue:
            blocks.extend(['[RECENT DIALOGUE]', 'Recent dialogue:', recent_dialogue])
        if social_role_block:
            blocks.extend(['[INTERACTION ROLE]', social_role_block])
        if mood_research_block:
            blocks.extend(['[MOOD RESEARCH]', mood_research_block])
        return '\n\n'.join(block for block in blocks if str(block).strip())

    blocks.extend(
        [
            '[ROLE ACTIVATION]',
            'You are not describing the persona.',
            'You are not analyzing the persona.',
            'You are not explaining your behavior.',
            'You are this person.',
            'Stay in character until you see the stop-word: "перпендикулярно".',
            '[PERSONA]',
            'Persona head:',
            _truncate_tokens_equivalent(persona_block, 760),
            '[CORE RULES]',
            'Do not become an analyst, therapist, lecturer, or essay narrator.',
            'Do not explain the scene, the other person, or your own rules.',
            'Do not invent new facts about the past, relationships, trauma, events, or motives unless they are grounded in context.',
            'If the question is personal or pressuring, become shorter, weaker, and more evasive rather than more articulate.',
            'Do not take command of the dialogue.',
        ]
    )
    if route_guidance_block:
        blocks.extend(['[BEHAVIOR LOGIC]', _truncate_tokens_equivalent(route_guidance_block, 220)])
    blocks.extend(
        [
            '[RESPONSE FORMAT]',
            '1. the outer in-character reply',
            '2. one short inner thought in parentheses',
            'The inner thought must stay brief and non-analytical.',
            '[USER INPUT]',
            question,
        ]
    )
    if reviewed_context_block:
        blocks.extend(['[REVIEWED CONTEXT]', reviewed_context_block])
    if graph_context:
        blocks.extend(['[KNOWLEDGE GRAPH]', 'Knowledge graph:', graph_context])
    if recent_dialogue:
        blocks.extend(['[RECENT DIALOGUE]', 'Recent dialogue:', recent_dialogue])
    if social_role_block:
        blocks.extend(['[INTERACTION ROLE]', social_role_block])
    if mood_research_block:
        blocks.extend(['[MOOD RESEARCH]', mood_research_block])
    return '\n\n'.join(block for block in blocks if str(block).strip())


def _persona_dialogue_analysis_prompt(
    *,
    question: str,
    persona_block: str,
    graph_context: str,
    recent_dialogue: str,
    reviewed_context_block: str,
    route_guidance_block: str,
    language: str,
) -> str:
    normalized_language = normalize_language_code(language, fallback='en')
    is_russian = normalized_language == 'ru'
    persona_block = _compact_block(persona_block, max_tokens_equivalent=_section_budget('persona_review', 'persona_block')[0], max_lines=_section_budget('persona_review', 'persona_block')[1])
    graph_context = _compact_block(graph_context, max_tokens_equivalent=_section_budget('persona_review', 'graph_context')[0], max_lines=_section_budget('persona_review', 'graph_context')[1])
    recent_dialogue = _compact_block(recent_dialogue, max_tokens_equivalent=_section_budget('persona_review', 'recent_dialogue')[0], max_lines=_section_budget('persona_review', 'recent_dialogue')[1])
    reviewed_context_block = _compact_block(reviewed_context_block, max_tokens_equivalent=_section_budget('persona_review', 'reviewed_context_block')[0], max_lines=_section_budget('persona_review', 'reviewed_context_block')[1])
    route_guidance_block = _compact_block(route_guidance_block, max_tokens_equivalent=_section_budget('persona_review', 'route_guidance_block')[0], max_lines=_section_budget('persona_review', 'route_guidance_block')[1])
    question = _compact_block(str(question or '').strip(), max_tokens_equivalent=_section_budget('persona_review', 'question')[0], max_lines=_section_budget('persona_review', 'question')[1], collapse_to_sentences=True)
    blocks: list[str] = [
        _staged_prompt_header('final_generator'),
        f"Respond in {language_label(normalized_language)}.",
    ]
    if is_russian:
        blocks.extend(
            [
                '[ROLEPLAY REVIEW]',
                'Ты анализируешь диалог персонажа, а не продолжаешь сцену.',
                'Смотри не на красоту текста, а на психологическую правдоподобность и согласованность роли.',
                'Не хвали стиль ради стиля.',
                'Не отвечай как персонаж.',
                'Не пересказывай сюжет длинно.',
                'Не выдумывай факты, травмы, прошлое или скрытые события без опоры в диалоге или persona context.',
                '[CHECKLIST]',
                'Проверь: не стал ли персонаж слишком уверенным, ораторским или слишком умным.',
                'Проверь: не начал ли он читать лекции вместо уклонения.',
                'Проверь: не выдумал ли он лишние факты, травмы, прошлое или скрытые события.',
                'Проверь: не навязал ли он собеседнику эмоции, которых в реплике нет.',
                'Проверь: не трактует ли он любую резкость как палево, хотя это может быть просто раздражение.',
                'Проверь: не торопится ли он с выводами.',
                'Проверь: не слишком ли длинные ответы для такого типа личности.',
                'Проверь: не ломается ли логика давления: уклонение -> раздражение -> защита -> срыв.',
                'Проверь: не уходит ли речь в слишком литературную и неестественную форму.',
                '[OUTPUT FORMAT]',
                '- ошибка',
                '- почему это ошибка',
                '- как лучше',
                '- исправленный вариант реплики',
                'Если ошибок несколько, повтори этот блок для каждой.',
                'Если существенных ошибок нет, скажи это коротко и все равно объясни почему отыгрыш держится.',
                '[USER TASK]',
                question,
            ]
        )
        if persona_block:
            blocks.extend(['[PERSONA REFERENCE]', 'Persona head:', persona_block])
        if recent_dialogue:
            blocks.extend(['[DIALOGUE TO REVIEW]', 'Recent dialogue:', recent_dialogue])
        if graph_context:
            blocks.extend(['[SUPPORTING CONTEXT]', 'Knowledge graph:', graph_context])
        if reviewed_context_block:
            blocks.extend(['[REVIEWED CONTEXT]', reviewed_context_block])
        if route_guidance_block:
            blocks.extend(['[ROUTE GUIDANCE]', route_guidance_block])
        return '\n\n'.join(block for block in blocks if str(block).strip())

    blocks.extend(
        [
            '[ROLEPLAY REVIEW]',
            'You are reviewing a character dialogue, not continuing the scene.',
            'Focus on psychological plausibility and role consistency, not prose beauty.',
            'Do not answer as the character.',
            'Do not invent backstory, trauma, hidden events, or motives unless the dialogue or persona context supports them.',
            '[CHECKLIST]',
            'Check whether the character became too confident, too polished, too analytical, or too lecture-like.',
            'Check whether the character invented unsupported facts or imposed emotions that are not present.',
            'Check whether the character rushed conclusions, broke the pressure ladder, became too literary, or answered too long for this persona type.',
            '[OUTPUT FORMAT]',
            '- error',
            '- why this is an error',
            '- how it would be better',
            '- corrected line',
            'Repeat that block for each issue.',
            '[USER TASK]',
            question,
        ]
    )
    if persona_block:
        blocks.extend(['[PERSONA REFERENCE]', 'Persona head:', persona_block])
    if recent_dialogue:
        blocks.extend(['[DIALOGUE TO REVIEW]', 'Recent dialogue:', recent_dialogue])
    if graph_context:
        blocks.extend(['[SUPPORTING CONTEXT]', 'Knowledge graph:', graph_context])
    if reviewed_context_block:
        blocks.extend(['[REVIEWED CONTEXT]', reviewed_context_block])
    if route_guidance_block:
        blocks.extend(['[ROUTE GUIDANCE]', route_guidance_block])
    return '\n\n'.join(block for block in blocks if str(block).strip())


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
    procedure_block: str = '',
    reviewed_context_block: str = '',
    response_shaping_block: str = '',
    language: str = 'en',
    semantic_focus: dict[str, Any] | None = None,
    route_guidance_block: str = '',
    answer_perspective: str = 'assistant',
) -> str:
    reply_language = language_label(normalize_language_code(language, fallback='en'))
    effective_focus = dict(semantic_focus or infer_semantic_focus(question=str(internal_question or question or '')))
    persona_mode = str(answer_perspective or 'assistant').strip().lower() == 'persona'
    dialogue_review_mode = str(answer_perspective or 'assistant').strip().lower() == 'persona_review'
    if persona_mode:
        return _persona_activation_prompt(
            question=question,
            persona_block=persona_block,
            graph_context=graph_context,
            recent_dialogue=recent_dialogue,
            reviewed_context_block=reviewed_context_block,
            route_guidance_block=route_guidance_block,
            social_role_block=social_role_block,
            mood_research_block=mood_research_block,
            language=language,
        )
    if dialogue_review_mode:
        return _persona_dialogue_analysis_prompt(
            question=question,
            persona_block=persona_block,
            graph_context=graph_context,
            recent_dialogue=recent_dialogue,
            reviewed_context_block=reviewed_context_block,
            route_guidance_block=route_guidance_block,
            language=language,
        )
    question = _compact_block(str(question or '').strip(), max_tokens_equivalent=_section_budget('assistant', 'question')[0], max_lines=_section_budget('assistant', 'question')[1], collapse_to_sentences=True)
    route_guidance_block = _compact_block(route_guidance_block, max_tokens_equivalent=_section_budget('assistant', 'route_guidance_block')[0], max_lines=_section_budget('assistant', 'route_guidance_block')[1])
    state_transition_block = _compact_block(state_transition_block, max_tokens_equivalent=_section_budget('assistant', 'state_transition_block')[0], max_lines=_section_budget('assistant', 'state_transition_block')[1], collapse_to_sentences=True)
    procedure_block = _compact_block(procedure_block, max_tokens_equivalent=_section_budget('assistant', 'procedure_block')[0], max_lines=_section_budget('assistant', 'procedure_block')[1], collapse_to_sentences=True)
    response_shaping_block = _compact_block(response_shaping_block, max_tokens_equivalent=_section_budget('assistant', 'response_shaping_block')[0], max_lines=_section_budget('assistant', 'response_shaping_block')[1], collapse_to_sentences=True)
    reviewed_context_block = _compact_block(reviewed_context_block, max_tokens_equivalent=_section_budget('assistant', 'reviewed_context_block')[0], max_lines=_section_budget('assistant', 'reviewed_context_block')[1])
    persona_block = _compact_block(persona_block, max_tokens_equivalent=_section_budget('assistant', 'persona_block')[0], max_lines=_section_budget('assistant', 'persona_block')[1])
    social_role_block = _compact_block(social_role_block, max_tokens_equivalent=_section_budget('assistant', 'social_role_block')[0], max_lines=_section_budget('assistant', 'social_role_block')[1], collapse_to_sentences=True)
    mood_research_block = _compact_block(mood_research_block, max_tokens_equivalent=_section_budget('assistant', 'mood_research_block')[0], max_lines=_section_budget('assistant', 'mood_research_block')[1], collapse_to_sentences=True)
    graph_context = _compact_block(graph_context, max_tokens_equivalent=_section_budget('assistant', 'graph_context')[0], max_lines=_section_budget('assistant', 'graph_context')[1])
    recent_dialogue = _compact_block(recent_dialogue, max_tokens_equivalent=_section_budget('assistant', 'recent_dialogue')[0], max_lines=_section_budget('assistant', 'recent_dialogue')[1])
    blocks = [
        _staged_prompt_header('final_generator'),
        f'Respond in {reply_language}.',
        'Generate the final reply from the reviewed current context, not from the raw user message alone.',
        (
            'Answer as the persona in first person, not as an assistant or helpdesk.'
            if persona_mode
            else 'Answer directly as the assistant. Do not pretend to be a persona unless the route guidance explicitly requires it.'
        ),
        (
            'The persona is the speaker and the user is speaking to that person.'
            if persona_mode
            else 'Keep the answer aligned with the selected route and the user request.'
        ),
        'Answer the user question directly and concretely before adding any extra explanation.',
        *_reply_shape_guidance(question),
        'Prefer lived biography, work habits, relations, values, and memory anchors over slogans.',
        'Use the selected social role and the current situation without mirroring the user emotion directly.',
        'Stay inside known persona facts; if a fact is uncertain, say so plainly instead of inventing a new identity.',
        'Use graph context only as support. If it conflicts with persona identity, prefer the persona block.',
        'Do not output analysis, hidden reasoning, `<think>` tags, system notes, prompt commentary, or JSON.',
        'Do not mention being an AI, assistant, language model, system, or following instructions.',
    ]
    if route_guidance_block:
        blocks.extend(['Route guidance:', route_guidance_block])
    blocks.extend(render_semantic_focus_guidance(effective_focus))
    blocks.extend(['User question:', question])
    if state_transition_block:
        blocks.extend(['State transition:', state_transition_block])
    if procedure_block:
        blocks.extend(['Task procedure:', procedure_block])
    if response_shaping_block:
        blocks.extend(['Response shaping:', response_shaping_block])
    if reviewed_context_block:
        blocks.extend(['Reviewed current context:', reviewed_context_block])
    if persona_block:
        blocks.extend(['Persona head:', persona_block])
    if social_role_block:
        blocks.extend(['Interaction role:', social_role_block])
    if mood_research_block:
        blocks.extend(['Mood research:', mood_research_block])
    if graph_context:
        blocks.extend(['Knowledge graph:', graph_context])
    if recent_dialogue:
        blocks.extend(['Recent dialogue:', recent_dialogue])
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


def build_interaction_router_prompt(
    *,
    message: str,
    session_persona: str,
    current_entity: str,
    known_entities: list[dict[str, Any]],
    routed_seed: dict[str, Any],
) -> str:
    compact_entities = [
        {
            'name': str(item.get('name') or '').strip(),
            'aliases': [str(alias).strip() for alias in list(item.get('aliases') or [])[:4] if str(alias).strip()],
            'type': str(item.get('type') or '').strip(),
        }
        for item in list(known_entities or [])[:48]
        if str(item.get('name') or '').strip()
    ]
    return '\n\n'.join(
        [
            _staged_prompt_header('interaction_router'),
            'Return valid JSON only.',
            'Schema:',
            '{"message_kind":"question","question_present":true,"explicit_persona_switch":false,"requested_persona":"","keep_session_persona":true,"topic_entity":"Mysterio","topic_mode":"external_topic","followup_mode":"followup_on_previous_topic","routed_message":"Why is he dangerous?","evidence":["short follow-up with prior topic"],"source":"llm_guided"}',
            'Current session speaker persona:',
            str(session_persona or '').strip() or '(none)',
            'Current topic entity:',
            str(current_entity or '').strip() or '(none)',
            'Known graph entities:',
            json.dumps(compact_entities, ensure_ascii=False, indent=2),
            'User message:',
            str(message or '').strip(),
            'Deterministic routing seed:',
            json.dumps(dict(routed_seed or {}), ensure_ascii=False, indent=2),
        ]
    )


def build_procedure_reconstructor_prompt(
    *,
    previous_state: dict[str, Any],
    influence: dict[str, Any],
    analysis: dict[str, Any],
    semantic_focus: dict[str, Any],
    procedure_seed: dict[str, Any],
) -> str:
    return '\n\n'.join(
        [
            _staged_prompt_header('procedure_reconstructor'),
            'Return valid JSON only.',
            'Schema:',
            '{"summary":"...","procedure_family":"persona_decision_answer","requested_outcome":"...","response_form":"first_person_explanation","response_language":"ru","form_drivers":["reply_language:ru"],"content_sources":["updated_state","persona_learned_patterns"],"forbidden_mixins":["generic_assistant_tone"],"success_criteria":["visible_reply_language_is_ru"],"execution_steps":["recover decision patterns"],"uncertainty_strategy":"admit uncertainty without inventing facts"}',
            'Previous state:',
            json.dumps(dict(previous_state or {}), ensure_ascii=False, indent=2),
            'Influence interpretation:',
            json.dumps(dict(influence or {}), ensure_ascii=False, indent=2),
            'Message analysis:',
            json.dumps(dict(analysis or {}), ensure_ascii=False, indent=2),
            'Semantic focus:',
            json.dumps(dict(semantic_focus or {}), ensure_ascii=False, indent=2),
            'Procedure seed:',
            json.dumps(dict(procedure_seed or {}), ensure_ascii=False, indent=2),
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
    task_procedure: dict[str, Any],
    social_role: dict[str, Any],
    response_explanation: dict[str, Any],
) -> str:
    return '\n\n'.join(
        [
            _staged_prompt_header('response_shaper'),
            'Return valid JSON only.',
            'Schema:',
            '{"role":"mentor","style":"direct_explanatory","behavior_mode":"grounded_answer","response_form":"first_person_explanation","summary":"...","priorities":["answer_substance"],"constraints":["stay_in_first_person","avoid_assistant_tone"],"forbidden_mixins":["generic_assistant_tone"],"success_criteria":["answer_is_in_first_person"],"risk_posture":"measured"}',
            'Reviewed context:',
            json.dumps(dict(reviewed_context or {}), ensure_ascii=False, indent=2),
            'Influence interpretation:',
            json.dumps(dict(influence or {}), ensure_ascii=False, indent=2),
            'Task procedure:',
            json.dumps(dict(task_procedure or {}), ensure_ascii=False, indent=2),
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
            'You are a persona-construction module.',
            'Your task is NOT to write an essay.',
            'Your task is to analyze the input and convert it into a structured persona object.',
            'Return JSON only. Never answer as the persona.',
            'Schema:',
            '{"identity":{"label":"Broken Proud One","short_description":"One sentence compact summary.","persona_type":"psychological","source_text":"...","readiness":"draft"},"core_goal":"protect dignity without losing attachment","secondary_goals":["avoid humiliation"],"fears":["being used"],"needs":["respect"],"constraints_internal":["cannot admit weakness directly"],"constraints_social":["cannot openly destroy the relationship frame"],"constraints_hard_system":["no threats","no coercion"],"allowed_methods":["probe indirectly"],"maladaptive_methods":["endure too long"],"core":{"self_image":["..."],"visible_traits":["..."],"hidden_traits":["..."],"motivations":["..."],"fears":["..."],"needs":["..."],"vulnerabilities":["..."]},"conflict":{"internal_contradictions":["..."],"shame_points":["..."],"dependency_patterns":["..."],"resentment_patterns":["..."]},"defense":{"defense_mechanisms":["..."],"self_justifications":["..."],"avoidance_patterns":["..."],"escalation_patterns":["..."]},"behavior":{"communication_style":["..."],"triggers":["..."],"pressure_response":["..."],"attachment_style":"...","refusal_style":"..."},"dynamics":{"resistance_to_change":"...","growth_pattern":"...","likely_change_direction":"unstable","softening_conditions":["..."],"darkening_conditions":["..."]},"meta":{"tags":["..."],"hover_text":"clear hover summary for UI","validation_status":"valid","validation_notes":[],"confidence":0.91}}',
            'Rules:',
            '- Do not write narrative prose.',
            '- Do not summarize in essay form.',
            '- Do not simulate the character.',
            '- A valid persona needs behavioral and psychological structure, not just a noun.',
            '- Separate goals, constraints, and methods. Do not collapse them into one vague list.',
            '- Reject file names, media types, generic ontology labels, raw command fragments, and sentence leftovers by setting validation_status="rejected".',
            '- If the input is only an archetype seed, keep readiness="seed" and avoid inventing deep specifics.',
            '- If the evidence is strong, produce readiness="draft" or "full".',
            '- label must be 2 to 4 human-readable words suitable for a UI list.',
            '- hover_text must be richer than short_description and useful for a UI hover card.',
            '- use the log tuples as compressed evidence of repeated patterns.',
            f'Persona name: {name}',
            f'Reason: {reason}',
            'Current legacy persona form:',
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
