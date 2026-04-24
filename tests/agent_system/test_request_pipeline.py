from __future__ import annotations

from agent_system.models import InteractionFrame, MessageAnalysis, MessageEntity, RouteDecision, RouteMemory, Situation, UserState
from agent_system.request_pipeline import (
    build_request_envelope,
    plan_capabilities,
    preprocess_request,
    render_route_guidance,
    select_route,
    validate_response,
)


def _analysis(
    message: str,
    *,
    language: str = 'en',
    intent: str = 'question',
    selected_head: str = '',
    current_entity: str = '',
    entities: list[str] | None = None,
) -> MessageAnalysis:
    return MessageAnalysis(
        message=message,
        session_id='session_test',
        selected_head=selected_head,
        current_entity=current_entity,
        entities=[MessageEntity(name=item) for item in list(entities or [])],
        user_state=UserState(language=language, intent=intent, tone='neutral', signals={}),
        situation=Situation(type='neutral_query', target='external', severity=0.1),
    )


def test_request_pipeline_routes_hypothetical_without_strict_grounding() -> None:
    message = 'Представь, что ты капитан корабля во время шторма.'
    envelope = build_request_envelope(request_id='r1', session_id='s1', raw_text=message)
    analysis = _analysis(message, language='ru')
    frame = InteractionFrame(message_kind='question', question_present=True, routed_message=message)

    preprocessing = preprocess_request(envelope=envelope, analysis=analysis, interaction_frame=frame)
    route = select_route(envelope=envelope, preprocessing=preprocessing, analysis=analysis, interaction_frame=frame)
    capability = plan_capabilities(route)

    assert route.selected_route == 'hypothetical_roleplay'
    assert route.strict_grounding is False
    assert capability.use_heavy_persona_pipeline is False
    assert capability.use_context_builder is False


def test_request_pipeline_extracts_persona_style_and_speech_hints() -> None:
    message = 'представь что ты влюблен в х, ты робкий но гордый человек, отвечай робко, с паузами и словами паразитами'
    envelope = build_request_envelope(request_id='r_style', session_id='s1', raw_text=message)
    analysis = _analysis(message, language='ru')
    frame = InteractionFrame(message_kind='question', question_present=True, routed_message=message)

    preprocessing = preprocess_request(envelope=envelope, analysis=analysis, interaction_frame=frame)
    route = select_route(envelope=envelope, preprocessing=preprocessing, analysis=analysis, interaction_frame=frame)
    guidance = render_route_guidance(route)

    assert route.selected_route == 'hypothetical_roleplay'
    assert 'shy' in route.persona_style_traits
    assert 'proud' in route.persona_style_traits
    assert any('hesitant' in item or 'filler words' in item for item in route.speech_style_hints)
    assert 'Requested persona disposition' in guidance
    assert 'Requested speaking manner' in guidance


def test_request_pipeline_continues_hypothetical_followup_and_inherits_style_hints() -> None:
    message = 'я не понял. конкретизирую вопрос. так как будешь действовать дальше?'
    envelope = build_request_envelope(request_id='r_followup', session_id='s1', raw_text=message)
    analysis = _analysis(message, language='ru', intent='question')
    frame = InteractionFrame(message_kind='question', question_present=True, routed_message=message)
    route_memory = RouteMemory(
        previous_route='hypothetical_roleplay',
        previous_interaction_mode='hypothetical',
        persona_style_traits=['shy', 'proud'],
        speech_style_hints=['sound hesitant and self-interrupting', 'keep dignity and self-respect visible in the wording'],
    )

    preprocessing = preprocess_request(
        envelope=envelope,
        analysis=analysis,
        interaction_frame=frame,
        route_memory=route_memory,
    )
    route = select_route(
        envelope=envelope,
        preprocessing=preprocessing,
        analysis=analysis,
        interaction_frame=frame,
        route_memory=route_memory,
    )
    guidance = render_route_guidance(route)

    assert preprocessing.hypothetical_continuation is True
    assert route.selected_route == 'hypothetical_roleplay'
    assert route.requires_history is True
    assert 'shy' in route.persona_style_traits
    assert 'proud' in route.persona_style_traits
    assert 'Requested speaking manner' in guidance


def test_request_pipeline_routes_meta_questions_to_history_mode() -> None:
    message = 'Почему ты так ответил в прошлом сообщении?'
    envelope = build_request_envelope(request_id='r2', session_id='s1', raw_text=message)
    analysis = _analysis(message, language='ru')
    frame = InteractionFrame(message_kind='question', question_present=True, routed_message=message, followup_mode='followup_on_previous_topic')

    preprocessing = preprocess_request(envelope=envelope, analysis=analysis, interaction_frame=frame)
    route = select_route(envelope=envelope, preprocessing=preprocessing, analysis=analysis, interaction_frame=frame)

    assert route.selected_route == 'meta_previous_answer'
    assert route.requires_history is True
    assert route.requires_graph is False
    assert route.strict_grounding is False


def test_request_pipeline_routes_persona_dialogue_analysis_to_review_mode() -> None:
    message = 'Проанализируй диалог персонажа и найди ошибки в отыгрыше.'
    envelope = build_request_envelope(request_id='r_review', session_id='s1', raw_text=message)
    analysis = _analysis(message, language='ru', selected_head='Аня Юсупова')
    frame = InteractionFrame(message_kind='question', question_present=True, routed_message=message)

    preprocessing = preprocess_request(envelope=envelope, analysis=analysis, interaction_frame=frame)
    route = select_route(
        envelope=envelope,
        preprocessing=preprocessing,
        analysis=analysis,
        interaction_frame=frame,
        selected_persona='Аня Юсупова',
    )
    capability = plan_capabilities(route)
    guidance = render_route_guidance(route)

    assert preprocessing.request_type == 'persona_analysis'
    assert route.selected_route == 'persona_dialogue_analysis'
    assert route.requires_history is True
    assert route.requires_persona is True
    assert route.strict_grounding is False
    assert capability.use_context_builder is True
    assert capability.use_heavy_persona_pipeline is False
    assert 'ошибка' in guidance.lower()
    assert 'исправленный вариант' in guidance.lower()


def test_request_pipeline_routes_lightweight_conversation_to_fast_path() -> None:
    message = 'Привет'
    envelope = build_request_envelope(request_id='r3', session_id='s1', raw_text=message)
    analysis = _analysis(message, language='ru', intent='statement')
    frame = InteractionFrame(message_kind='statement', question_present=False, routed_message=message)

    preprocessing = preprocess_request(envelope=envelope, analysis=analysis, interaction_frame=frame)
    route = select_route(envelope=envelope, preprocessing=preprocessing, analysis=analysis, interaction_frame=frame)
    capability = plan_capabilities(route)

    assert route.selected_route == 'lightweight_conversation'
    assert route.fast_path is True
    assert route.requires_graph is False
    assert route.requires_persona is False
    assert capability.use_context_builder is False


def test_request_pipeline_keeps_persona_voice_for_lightweight_social_intro() -> None:
    message = 'Привет, я Маша, а ты?'
    envelope = build_request_envelope(request_id='r_light_persona', session_id='s1', raw_text=message)
    analysis = _analysis(message, language='ru', intent='question', selected_head='Катерина')
    frame = InteractionFrame(message_kind='question', question_present=True, routed_message=message, topic_mode='persona_self')

    preprocessing = preprocess_request(envelope=envelope, analysis=analysis, interaction_frame=frame)
    route = select_route(
        envelope=envelope,
        preprocessing=preprocessing,
        analysis=analysis,
        interaction_frame=frame,
        selected_persona='Катерина',
    )
    capability = plan_capabilities(route)
    guidance = render_route_guidance(route)

    assert preprocessing.interaction_mode == 'lightweight_conversation'
    assert preprocessing.request_type == 'persona_chat'
    assert route.selected_route == 'persona_chat_fast_path'
    assert route.requires_persona is True
    assert route.requires_graph is False
    assert route.strict_grounding is False
    assert capability.use_context_builder is False
    assert capability.use_heavy_persona_pipeline is False
    assert route.fast_path is True
    assert 'persona chat fast path' in guidance.lower()


def test_request_pipeline_does_not_treat_direct_persona_self_query_as_lightweight_intro() -> None:
    message = 'Привет, кто ты, Дракула?'
    envelope = build_request_envelope(request_id='r_self_query', session_id='s1', raw_text=message)
    analysis = _analysis(message, language='ru', intent='question', selected_head='Dracula', current_entity='Дракула', entities=['Дракула'])
    frame = InteractionFrame(message_kind='question', question_present=True, routed_message=message, topic_mode='persona_self')

    preprocessing = preprocess_request(envelope=envelope, analysis=analysis, interaction_frame=frame)
    route = select_route(
        envelope=envelope,
        preprocessing=preprocessing,
        analysis=analysis,
        interaction_frame=frame,
        selected_persona='Dracula',
    )

    assert preprocessing.interaction_mode == 'persona_dialogue'
    assert route.selected_route == 'persona_graph_reasoning'
    assert route.requires_persona is True
    assert route.strict_grounding is True


def test_request_pipeline_routes_graph_persona_reasoning_when_persona_is_selected() -> None:
    message = 'Кто для тебя Леонард?'
    envelope = build_request_envelope(request_id='r4', session_id='s1', raw_text=message)
    analysis = _analysis(message, language='ru', selected_head='sheldon', current_entity='Leonard', entities=['Leonard'])
    frame = InteractionFrame(
        message_kind='question',
        question_present=True,
        requested_persona='sheldon',
        keep_session_persona=True,
        topic_entity='Leonard',
        topic_mode='persona_self',
        routed_message=message,
    )

    preprocessing = preprocess_request(envelope=envelope, analysis=analysis, interaction_frame=frame)
    route = select_route(
        envelope=envelope,
        preprocessing=preprocessing,
        analysis=analysis,
        interaction_frame=frame,
        selected_persona='sheldon',
    )

    assert route.selected_route == 'persona_graph_reasoning'
    assert route.requires_persona is True
    assert route.requires_graph is True
    assert route.strict_grounding is True


def test_request_pipeline_routes_short_persona_followup_to_fast_path() -> None:
    message = 'а ты что скажешь?'
    envelope = build_request_envelope(request_id='r_fast_persona_followup', session_id='s1', raw_text=message)
    analysis = _analysis(message, language='ru', intent='question', selected_head='Катерина')
    frame = InteractionFrame(
        message_kind='question',
        question_present=True,
        routed_message=message,
        topic_mode='persona_self',
        followup_mode='followup_on_persona',
    )

    preprocessing = preprocess_request(envelope=envelope, analysis=analysis, interaction_frame=frame)
    route = select_route(
        envelope=envelope,
        preprocessing=preprocessing,
        analysis=analysis,
        interaction_frame=frame,
        selected_persona='Катерина',
    )

    assert preprocessing.request_type == 'persona_chat'
    assert route.selected_route == 'persona_chat_fast_path'
    assert route.fast_path is True
    assert route.requires_history is True
    assert route.requires_graph is False


def test_request_pipeline_routes_rich_persona_profile_to_persona_specification_even_with_detected_name() -> None:
    message = (
        'Катерина — сильная, холодная, собранная женщина с острым языком и жёсткой внутренней дисциплиной. '
        'Она мало говорит, быстро считывает людей, не терпит слабость как позу и защищает своих без лишней нежности. '
        'Говорит коротко, сухо, уверенно, иногда колко. Внутри уязвимее, чем кажется, но почти никогда этого не показывает.'
    )
    envelope = build_request_envelope(request_id='r_profile', session_id='s1', raw_text=message)
    analysis = _analysis(message, language='ru', intent='statement', selected_head='Катерина', entities=['Катерина'])
    frame = InteractionFrame(message_kind='statement', question_present=False, routed_message=message)

    preprocessing = preprocess_request(envelope=envelope, analysis=analysis, interaction_frame=frame)
    route = select_route(
        envelope=envelope,
        preprocessing=preprocessing,
        analysis=analysis,
        interaction_frame=frame,
    )

    assert preprocessing.request_type == 'persona_specification'
    assert route.selected_route == 'persona_specification'
    assert 'persona_specification_detected' in preprocessing.evidence or 'persona_profile_signal_detected' in preprocessing.evidence


def test_request_pipeline_continues_persona_specification_after_intro_command() -> None:
    message = 'Катерина, 31, бухгалтер, скромная, робкая, терпеливая, стыдливая, говорит тихо и коротко, боится навязываться.'
    envelope = build_request_envelope(request_id='r_profile_followup', session_id='s1', raw_text=message)
    analysis = _analysis(message, language='ru', intent='statement')
    frame = InteractionFrame(message_kind='statement', question_present=False, routed_message=message)
    route_memory = RouteMemory(
        previous_route='persona_specification',
        previous_request_type='persona_specification',
        previous_interaction_mode='persona_specification',
    )

    preprocessing = preprocess_request(
        envelope=envelope,
        analysis=analysis,
        interaction_frame=frame,
        route_memory=route_memory,
    )
    route = select_route(
        envelope=envelope,
        preprocessing=preprocessing,
        analysis=analysis,
        interaction_frame=frame,
        route_memory=route_memory,
    )

    assert preprocessing.request_type == 'persona_specification'
    assert 'persona_specification_continuation_detected' in preprocessing.evidence
    assert route.selected_route == 'persona_specification'


def test_request_pipeline_validation_detects_hypothetical_route_mismatch() -> None:
    route = RouteDecision(
        selected_route='hypothetical_roleplay',
        detected_language='ru',
        intent_type='question',
        interaction_mode='hypothetical',
        strict_grounding=False,
        validation_mode='hypothetical_route',
    )

    validation = validate_response(
        route=route,
        reply='Не хватает надежного контекста. Уточни объект.',
        fallback_triggered=True,
        fallback_reason_code='no_grounding',
        history_used=False,
        graph_used=False,
        persona_used=False,
        history_available=False,
    )

    assert validation.ok is False
    assert validation.route_match is False
    assert validation.repair_strategy == 'regenerate_hypothetical'


def test_request_pipeline_validation_requires_history_for_meta_questions() -> None:
    route = RouteDecision(
        selected_route='meta_previous_answer',
        detected_language='en',
        intent_type='question',
        interaction_mode='meta_analysis',
        requires_history=True,
        strict_grounding=False,
        validation_mode='meta_history',
    )

    validation = validate_response(
        route=route,
        reply='I answered that way.',
        fallback_triggered=False,
        fallback_reason_code='',
        history_used=False,
        graph_used=False,
        persona_used=False,
        history_available=True,
    )

    assert validation.ok is False
    assert validation.route_match is False
    assert validation.mismatch_reason == 'meta_question_did_not_load_history'
    assert validation.repair_strategy == 'regenerate_meta'


def test_request_pipeline_validation_requires_dialogue_review_structure() -> None:
    route = RouteDecision(
        selected_route='persona_dialogue_analysis',
        detected_language='ru',
        intent_type='question',
        interaction_mode='persona_analysis',
        requires_history=True,
        validation_mode='dialogue_review',
    )

    validation = validate_response(
        route=route,
        reply='Реплика звучит слишком уверенно и не очень подходит персонажу.',
        fallback_triggered=False,
        fallback_reason_code='',
        history_used=True,
        graph_used=False,
        persona_used=True,
        history_available=True,
    )

    assert validation.ok is False
    assert validation.mismatch_reason == 'dialogue_review_missing_required_structure'
    assert validation.repair_strategy == 'regenerate_meta'


def test_request_pipeline_validation_rejects_generation_scaffold_leak() -> None:
    route = RouteDecision(
        selected_route='persona_graph_reasoning',
        detected_language='ru',
        intent_type='question',
        interaction_mode='persona_dialogue',
        requires_llm=True,
        validation_mode='persona_consistency',
    )

    validation = validate_response(
        route=route,
        reply='# Анализ ситуации\n**Ключевая проблема:** контекст неясен.\n\n**Внешний ответ персонажа:** Ладно, скажу проще.',
        fallback_triggered=False,
        fallback_reason_code='',
        history_used=True,
        graph_used=True,
        persona_used=True,
        history_available=True,
    )

    assert validation.ok is False
    assert validation.mismatch_reason == 'reply_leaked_generation_scaffold'
    assert validation.repair_strategy == 'regenerate_style_guard'


def test_request_pipeline_validation_rejects_review_notes_scaffold_leak() -> None:
    route = RouteDecision(
        selected_route='persona_chat_fast_path',
        detected_language='hy',
        intent_type='question',
        interaction_mode='persona_dialogue',
        requires_llm=True,
        validation_mode='persona_consistency',
    )

    validation = validate_response(
        route=route,
        reply='# Answer\nԲարև։\n\n**Review Notes:**\n- draft is in Armenian.\n**Issues Identified:**\n- persona inconsistency.',
        fallback_triggered=False,
        fallback_reason_code='',
        history_used=True,
        graph_used=False,
        persona_used=True,
        history_available=True,
    )

    assert validation.ok is False
    assert validation.mismatch_reason == 'reply_leaked_generation_scaffold'
    assert validation.repair_strategy == 'regenerate_style_guard'


def test_request_pipeline_validation_detects_output_budget_truncation() -> None:
    route = RouteDecision(
        selected_route='persona_graph_reasoning',
        detected_language='ru',
        intent_type='question',
        interaction_mode='persona_dialogue',
        requires_llm=True,
        validation_mode='persona_consistency',
    )

    validation = validate_response(
        route=route,
        reply='Я вампир, изгнанный из своей стаи. Я питаюсь тем, что могу, и',
        fallback_triggered=False,
        fallback_reason_code='',
        history_used=True,
        graph_used=True,
        persona_used=True,
        history_available=True,
        model_budget={'output_truncated': True, 'output_budget_too_small': True},
    )

    assert validation.ok is False
    assert validation.mismatch_reason == 'reply_was_truncated_by_output_budget'
    assert validation.repair_strategy == 'regenerate_with_budget'


def test_request_pipeline_validation_detects_repetition_loops() -> None:
    route = RouteDecision(
        selected_route='persona_graph_reasoning',
        detected_language='ru',
        intent_type='question',
        interaction_mode='persona_dialogue',
        requires_llm=True,
        validation_mode='persona_consistency',
    )

    validation = validate_response(
        route=route,
        reply='Я питаюсь осторожно и скрытно. Я питаюсь осторожно и скрытно. Я питаюсь осторожно и скрытно. Я питаюсь осторожно и скрытно.',
        fallback_triggered=False,
        fallback_reason_code='',
        history_used=True,
        graph_used=True,
        persona_used=True,
        history_available=True,
    )

    assert validation.ok is False
    assert validation.mismatch_reason == 'reply_contains_repetitive_looping_text'
    assert validation.repair_strategy == 'regenerate_with_budget'


def test_request_pipeline_validation_rejects_overly_analytic_fragile_persona_reply() -> None:
    route = RouteDecision(
        selected_route='hypothetical_roleplay',
        detected_language='ru',
        intent_type='question',
        interaction_mode='hypothetical',
        requires_llm=True,
        strict_grounding=False,
        validation_mode='hypothetical_route',
        persona_style_traits=['shy', 'proud'],
        speech_style_hints=['sound hesitant and self-interrupting'],
    )

    validation = validate_response(
        route=route,
        reply='В этой ситуации я бы сначала выделил главный риск, затем спокойно обозначил границы и объяснил, что такое использование мной больше невозможно.',
        fallback_triggered=False,
        fallback_reason_code='',
        history_used=False,
        graph_used=False,
        persona_used=False,
        history_available=False,
    )

    assert validation.ok is False
    assert validation.mismatch_reason == 'reply_failed_fragile_persona_style_guard'
    assert validation.repair_strategy == 'regenerate_style_guard'
    assert 'validation:too_analytic' in validation.reason_codes


def test_request_pipeline_validation_rejects_strong_direct_reply_under_pressure() -> None:
    route = RouteDecision(
        selected_route='persona_graph_reasoning',
        detected_language='ru',
        intent_type='question',
        interaction_mode='persona_dialogue',
        requires_llm=True,
        strict_grounding=True,
        validation_mode='persona_consistency',
        persona_style_traits=['shy', 'proud', 'dependent'],
        speech_style_hints=['sound hesitant and self-interrupting'],
    )

    validation = validate_response(
        route=route,
        reply='Я люблю его и поэтому сначала спокойно объясню, что он больше не может мной пользоваться.',
        fallback_triggered=False,
        fallback_reason_code='',
        history_used=True,
        graph_used=True,
        persona_used=True,
        history_available=True,
        request_text='ты влюблен в Y, а он использует тебя. как ты поступишь?',
    )

    assert validation.ok is False
    assert validation.mismatch_reason == 'reply_failed_fragile_persona_style_guard'
    assert validation.repair_strategy == 'regenerate_style_guard'
    assert 'validation:too_structured' in validation.reason_codes
    assert 'validation:too_strong_under_pressure' in validation.reason_codes
    assert 'validation:too_direct_under_pressure' in validation.reason_codes


def test_request_pipeline_truncated_fragile_reply_prefers_style_guard_repair() -> None:
    route = RouteDecision(
        selected_route='hypothetical_roleplay',
        detected_language='ru',
        intent_type='question',
        interaction_mode='hypothetical',
        requires_llm=True,
        strict_grounding=False,
        validation_mode='hypothetical_route',
        persona_style_traits=['shy', 'proud'],
        speech_style_hints=['sound hesitant and self-interrupting'],
    )

    validation = validate_response(
        route=route,
        reply='Я бы сначала спокойно разобрал ситуацию и обозначил границы',
        fallback_triggered=False,
        fallback_reason_code='',
        history_used=False,
        graph_used=False,
        persona_used=False,
        history_available=False,
        model_budget={'output_truncated': True, 'output_budget_too_small': True},
        request_text='представь что ты влюблен в х, а он использует тебя',
    )

    assert validation.ok is False
    assert validation.mismatch_reason == 'reply_was_truncated_by_output_budget'
    assert validation.repair_strategy == 'regenerate_style_guard'
