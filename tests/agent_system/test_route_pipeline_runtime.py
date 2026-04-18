from __future__ import annotations

import agent_system.chat_engine as chat_engine_module
from agent_system.chat_engine import _resolve_chat_orchestration_roles, generate_response
from agent_system.persona_engine import materialize_persona


def test_chat_engine_hypothetical_route_disables_grounding_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr(
        'agent_system.chat_engine.generate_chat_reply',
        lambda prompt, language='ru', persona_selected=False, allow_builtin_fallback=False: 'В гипотетической ситуации я бы держал курс и успокаивал команду.',
    )

    result = generate_response(
        message='Представь, что ты капитан корабля во время шторма.',
        session_id='hypothetical_case',
        language='ru',
    )

    assert result['pipeline']['route']['selected_route'] == 'hypothetical_roleplay'
    assert result['pipeline']['route']['strict_grounding'] is False
    assert result['pipeline']['fallback_triggered'] is False
    assert result['validation']['ok'] is True


def test_chat_engine_hypothetical_route_passes_style_hints_into_prompt(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    captured: dict[str, object] = {}

    def fake_build_prompt(**kwargs):  # type: ignore[no-untyped-def]
        captured['route_guidance_block'] = kwargs.get('route_guidance_block')
        captured['answer_perspective'] = kwargs.get('answer_perspective')
        return 'prompt'

    monkeypatch.setattr('agent_system.chat_engine.build_chat_prompt', fake_build_prompt)
    monkeypatch.setattr(
        'agent_system.chat_engine.generate_chat_reply',
        lambda prompt, language='ru', persona_selected=False, allow_builtin_fallback=False, **kwargs: 'Емм... мне тяжело это признавать, но я бы отстранился и перестал позволять себя использовать.',
    )

    result = generate_response(
        message='представь что ты влюблен в х, а он использует тебя. ты сам робкий но гордый человек. как будешь поступать?',
        session_id='style_hint_case',
        language='ru',
    )

    guidance = str(captured.get('route_guidance_block') or '')
    assert result['pipeline']['route']['selected_route'] == 'hypothetical_roleplay'
    assert captured['answer_perspective'] == 'persona'
    assert 'Requested persona disposition' in guidance
    assert 'shy' in guidance
    assert 'proud' in guidance
    assert 'Requested speaking manner' in guidance


def test_chat_engine_hypothetical_followup_keeps_route_and_inherits_style_hints(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)

    captured_guidance: list[str] = []

    def fake_build_prompt(**kwargs):  # type: ignore[no-untyped-def]
        captured_guidance.append(str(kwargs.get('route_guidance_block') or ''))
        return 'prompt'

    def fake_generate(
        prompt: str,
        language: str = 'ru',
        persona_selected: bool = False,
        allow_builtin_fallback: bool = False,
        **_: object,
    ) -> str:
        return 'Емм... мне стыдно это признавать, но я бы перестал позволять ему распоряжаться мной и начал бы тихо отказывать.'

    monkeypatch.setattr('agent_system.chat_engine.build_chat_prompt', fake_build_prompt)
    monkeypatch.setattr('agent_system.chat_engine.generate_chat_reply', fake_generate)

    first = generate_response(
        message='представь что ты влюблен в х, а он использует тебя. ты сам робкий но гордый человек. как будешь поступать?',
        session_id='hypothetical_followup_case',
        language='ru',
    )
    second = generate_response(
        message='я не понял. конкретизирую вопрос. так как будешь действовать дальше?',
        session_id='hypothetical_followup_case',
        language='ru',
    )

    assert first['pipeline']['route']['selected_route'] == 'hypothetical_roleplay'
    assert second['pipeline']['route']['selected_route'] == 'hypothetical_roleplay'
    assert second['pipeline']['route']['requires_history'] is True
    assert second['pipeline']['preprocessing']['hypothetical_continuation'] is True
    assert 'session_history' in second['pipeline']['context_sources_used']
    assert any('shy' in item for item in second['pipeline']['route']['persona_style_traits'])
    assert any('Requested speaking manner' in item for item in captured_guidance[-2:])


def test_chat_engine_hypothetical_route_replaces_generic_model_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr(
        'agent_system.chat_engine.generate_chat_reply',
        lambda prompt, language='ru', persona_selected=False, allow_builtin_fallback=False: '',
    )
    monkeypatch.setattr(
        'agent_system.chat_engine.runtime_status_snapshot',
        lambda: {'mode': 'degraded', 'degraded_modes': [{'code': 'llm_roles_missing'}]},
    )

    result = generate_response(
        message='Imagine you are a ship captain in a storm.',
        session_id='hypothetical_fallback_case',
        language='en',
    )

    assert result['pipeline']['route']['selected_route'] == 'hypothetical_roleplay'
    assert result['pipeline']['fallback_triggered'] is True
    assert result['pipeline']['fallback_reason_code'] == 'dependency_unavailable'
    assert 'not enough reliable context' not in result['assistant_reply'].lower()
    assert 'clarify the entity' not in result['assistant_reply'].lower()
    assert result['validation']['ok'] is True


def test_chat_engine_meta_route_uses_session_history(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)

    def fake_generate(prompt: str, language: str = 'en', persona_selected: bool = False, allow_builtin_fallback: bool = False) -> str:
        if 'Why did you answer that way?' in prompt:
            return 'I answered that way because the previous turn was just a greeting and did not ask for a factual profile.'
        return 'Hi.'

    monkeypatch.setattr('agent_system.chat_engine.generate_chat_reply', fake_generate)

    generate_response(
        message='Hello',
        session_id='meta_case',
        language='en',
    )
    result = generate_response(
        message='Why did you answer that way?',
        session_id='meta_case',
        language='en',
    )

    assert result['pipeline']['route']['selected_route'] == 'meta_previous_answer'
    assert result['pipeline']['route']['requires_history'] is True
    assert 'session_history' in result['pipeline']['context_sources_used']
    assert result['validation']['used_history'] is True
    assert result['validation']['ok'] is True


def test_chat_engine_persona_dialogue_analysis_route_uses_history_and_review_prompt(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)

    captured: dict[str, object] = {}

    def fake_build_prompt(**kwargs):  # type: ignore[no-untyped-def]
        captured['answer_perspective'] = kwargs.get('answer_perspective')
        captured['recent_dialogue'] = kwargs.get('recent_dialogue')
        captured['persona_block'] = kwargs.get('persona_block')
        captured['route_guidance_block'] = kwargs.get('route_guidance_block')
        return 'review prompt'

    def fake_generate(prompt: str, language: str = 'ru', persona_selected: bool = False, allow_builtin_fallback: bool = False, **_: object) -> str:
        return '\n'.join(
            [
                '- ошибка: персонаж стал слишком уверенным и звучит почти как лектор.',
                '- почему это ошибка: для такого профиля это ломает уязвимость и уклончивость.',
                '- как лучше: сделать ответ короче, слабее и с раздраженным уходом в сторону.',
                '- исправленный вариант реплики: "Да ничего... отстань, ладно? (Чёрт, опять полезли.)"',
            ]
        )

    monkeypatch.setattr('agent_system.chat_engine.build_chat_prompt', fake_build_prompt)
    monkeypatch.setattr('agent_system.chat_engine.generate_chat_reply', fake_generate)

    generate_response(
        message='Представь, что ты робкий, гордый человек, которому стыдно признаться в чувствах.',
        session_id='persona_review_case',
        language='ru',
    )
    generate_response(
        message='Почему ты так резко отвечаешь?',
        session_id='persona_review_case',
        language='ru',
    )
    result = generate_response(
        message='Проанализируй диалог персонажа и найди ошибки в отыгрыше.',
        session_id='persona_review_case',
        selected_persona='Shy Proud Persona',
        language='ru',
    )

    assert result['pipeline']['route']['selected_route'] == 'persona_dialogue_analysis'
    assert result['pipeline']['route']['requires_history'] is True
    assert result['pipeline']['route']['requires_persona'] is True
    assert 'session_history' in result['pipeline']['context_sources_used']
    assert captured['answer_perspective'] == 'persona_review'
    assert 'Почему ты так резко отвечаешь?' in str(captured.get('recent_dialogue') or '')
    assert 'ошибка' in str(captured.get('route_guidance_block') or '').lower()
    assert result['validation']['ok'] is True


def test_chat_engine_lightweight_route_skips_context_builder(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr(
        'agent_system.chat_engine.build_context',
        lambda **kwargs: (_ for _ in ()).throw(AssertionError('lightweight route should not call build_context')),
    )
    monkeypatch.setattr(
        'agent_system.chat_engine.generate_chat_reply',
        lambda prompt, language='ru', persona_selected=False, allow_builtin_fallback=False: 'Привет. Чем помочь?',
    )

    result = generate_response(
        message='Привет',
        session_id='lightweight_case',
        language='ru',
    )

    assert result['pipeline']['route']['selected_route'] == 'lightweight_conversation'
    assert result['pipeline']['capability_plan']['use_context_builder'] is False
    assert result['graph_context'] == ''
    assert result['validation']['ok'] is True


def test_chat_engine_lightweight_persona_intro_uses_fast_persona_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    captured: dict[str, object] = {}

    def fake_build_prompt(**kwargs):  # type: ignore[no-untyped-def]
        captured['answer_perspective'] = kwargs.get('answer_perspective')
        captured['route_guidance_block'] = kwargs.get('route_guidance_block')
        return 'prompt'

    def fake_generate(
        prompt: str,
        language: str = 'ru',
        persona_selected: bool = False,
        allow_builtin_fallback: bool = False,
        **_: object,
    ) -> str:
        captured['persona_selected'] = persona_selected
        return 'Привет.\nКатерина.'

    monkeypatch.setattr('agent_system.chat_engine.build_chat_prompt', fake_build_prompt)
    monkeypatch.setattr('agent_system.chat_engine.generate_chat_reply', fake_generate)

    result = generate_response(
        message='привет, я маша, а ты?',
        session_id='persona_intro_case',
        selected_persona='Катерина',
        language='ru',
    )

    assert result['pipeline']['route']['selected_route'] == 'persona_chat_fast_path'
    assert result['pipeline']['route']['requires_persona'] is True
    assert result['pipeline']['route']['strict_grounding'] is False
    assert result['pipeline']['capability_plan']['use_heavy_persona_pipeline'] is False
    assert captured['answer_perspective'] == 'persona'
    assert captured['persona_selected'] is True
    assert 'active persona' in str(captured.get('route_guidance_block') or '').lower()
    assert result['persona_name'] == 'Катерина'
    assert result['current_entity'] == 'Катерина'
    assert result['pipeline']['fallback_reason_code'] == ''
    assert 'clean fact' not in result['assistant_reply'].lower()
    assert result['validation']['ok'] is True


def test_chat_engine_lightweight_persona_intro_uses_persona_fallback_when_runtime_is_unavailable(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr(
        'agent_system.chat_engine.build_context',
        lambda **kwargs: {
            'persona_name': 'Катерина',
            'current_entity': 'Катерина',
            'persona_block': 'Имя/ярлык: Катерина\nВнешне ты: сдержанная, холодная, собранная.',
            'graph_context': '',
            'recent_dialogue': '',
            'estimated_tokens': 32,
            'context_debug': {'source_counts': {'persona_memory': 1}, 'selected_items': ['persona_block']},
        },
    )
    monkeypatch.setattr('agent_system.chat_engine.generate_chat_reply', lambda *args, **kwargs: '')
    monkeypatch.setattr(
        'agent_system.chat_engine.runtime_status_snapshot',
        lambda: {'mode': 'degraded', 'degraded_modes': [{'code': 'llm_roles_missing'}]},
    )

    result = generate_response(
        message='я маша, а ты?',
        session_id='persona_intro_fallback_case',
        selected_persona='Катерина',
        language='ru',
    )

    assert result['pipeline']['route']['selected_route'] == 'persona_chat_fast_path'
    assert result['pipeline']['fallback_triggered'] is True
    assert result['pipeline']['fallback_reason_code'] == 'dependency_unavailable'
    assert result['assistant_reply'] == 'Привет.\nЯ Катерина.'
    assert 'clean fact' not in result['assistant_reply'].lower()
    assert result['validation']['ok'] is True


def test_chat_engine_factual_route_uses_grounding_fallback_reason(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr(
        'agent_system.chat_engine.build_context',
        lambda **kwargs: {
            'persona_name': '',
            'current_entity': '',
            'persona_block': '',
            'graph_context': '',
            'recent_dialogue': '',
            'estimated_tokens': 0,
            'context_debug': {'source_counts': {}, 'selected_items': []},
        },
    )
    monkeypatch.setattr(
        'agent_system.chat_engine.generate_chat_reply',
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('factual no-grounding path should not call generate_chat_reply')),
    )

    result = generate_response(
        message='Who is the architect of the moon city of Velis IX?',
        session_id='factual_no_grounding',
        language='en',
    )

    assert result['pipeline']['route']['selected_route'] == 'factual_answer'
    assert result['pipeline']['fallback_reason_code'] == 'no_grounding'
    assert result['validation']['fallback_triggered'] is True


def test_chat_orchestration_alternate_swaps_primary_and_reviewer_by_turn(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setenv('COGNITIVE_CHAT_ORCHESTRATION', 'alternate')
    monkeypatch.setenv('COGNITIVE_CHAT_PRIMARY_ROLE', 'general')
    monkeypatch.setenv('COGNITIVE_CHAT_REVIEW_ROLE', 'analyst')
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr(
        'agent_system.chat_engine.generate_chat_reply',
        lambda prompt, language='en', persona_selected=False, allow_builtin_fallback=False, **kwargs: 'Hello.',
    )

    first = _resolve_chat_orchestration_roles(
        session_id='orch_case',
        request_id='req-1',
        persona_selected=False,
    )

    generate_response(
        message='Hello there.',
        session_id='orch_case',
        language='en',
    )

    second = _resolve_chat_orchestration_roles(
        session_id='orch_case',
        request_id='req-2',
        persona_selected=False,
    )

    assert first['primary_role'] == 'general'
    assert first['reviewer_role'] == 'analyst'
    assert second['primary_role'] == 'analyst'
    assert second['reviewer_role'] == 'general'


def test_chat_engine_records_chat_orchestration_and_uses_reviewer_on_repair(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setenv('COGNITIVE_CHAT_ORCHESTRATION', 'primary_with_reviewer')
    monkeypatch.setenv('COGNITIVE_CHAT_PRIMARY_ROLE', 'general')
    monkeypatch.setenv('COGNITIVE_CHAT_REVIEW_ROLE', 'analyst')
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)

    calls: list[dict[str, object]] = []

    def fake_generate(
        prompt: str,
        language: str = 'en',
        persona_selected: bool = False,
        allow_builtin_fallback: bool = False,
        **kwargs: object,
    ) -> str:
        role_override = str(kwargs.get('role_override') or '')
        calls.append({'role_override': role_override, 'prompt': prompt})
        if len(calls) == 1:
            return 'First draft that explains the situation in a long, confident paragraph with structured reasoning.'
        return 'Эм... не знаю.\nНе хочу об этом.'

    monkeypatch.setattr('agent_system.chat_engine.generate_chat_reply', fake_generate)

    result = generate_response(
        message='представь что ты робкий но гордый человек, которому стыдно признаться в чувствах',
        session_id='orch_repair_case',
        language='ru',
    )

    assert calls[0]['role_override'] == 'general'
    assert len(calls) >= 2
    assert calls[1]['role_override'] == 'analyst'
    assert result['pipeline']['chat_orchestration']['primary_role'] == 'general'
    assert result['pipeline']['chat_orchestration']['reviewer_role'] == 'analyst'
    assert result['context_preview']['chat_orchestration']['strategy'] == 'primary_with_reviewer'


def test_chat_engine_runs_reviewer_on_success_when_review_mode_is_always(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setenv('COGNITIVE_CHAT_ORCHESTRATION', 'primary_with_reviewer')
    monkeypatch.setenv('COGNITIVE_CHAT_PRIMARY_ROLE', 'general')
    monkeypatch.setenv('COGNITIVE_CHAT_REVIEW_ROLE', 'analyst')
    monkeypatch.setenv('COGNITIVE_CHAT_REVIEW_MODE', 'always')
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)

    calls: list[dict[str, object]] = []

    def fake_generate(
        prompt: str,
        language: str = 'en',
        persona_selected: bool = False,
        allow_builtin_fallback: bool = False,
        **kwargs: object,
    ) -> str:
        calls.append(
            {
                'role_override': str(kwargs.get('role_override') or ''),
                'prompt': prompt,
            }
        )
        if len(calls) == 1:
            return 'Short grounded draft.'
        return 'Short grounded final reply.'

    monkeypatch.setattr('agent_system.chat_engine.generate_chat_reply', fake_generate)

    result = generate_response(
        message='Hello there.',
        session_id='always_review_case',
        language='en',
    )

    assert len(calls) >= 2
    assert calls[0]['role_override'] == 'general'
    assert calls[1]['role_override'] == 'analyst'
    assert result['assistant_reply'] == 'Short grounded final reply.'
    assert result['pipeline']['chat_orchestration']['review_mode'] == 'always'


def test_chat_engine_heavy_persona_path_exposes_chat_orchestration(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setenv('COGNITIVE_CHAT_ORCHESTRATION', 'primary_with_reviewer')
    monkeypatch.setenv('COGNITIVE_CHAT_PRIMARY_ROLE', 'general')
    monkeypatch.setenv('COGNITIVE_CHAT_REVIEW_ROLE', 'analyst')
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)

    materialize_persona(
        'Dracula',
        {
            'entity_type': 'PERSON',
            'traits': ['vampiric', 'reserved'],
            'knowledge': 'Dracula is a reserved vampire nobleman.',
        },
        explicit=True,
    )

    calls: list[str] = []

    def fake_generate(
        prompt: str,
        language: str = 'en',
        persona_selected: bool = False,
        allow_builtin_fallback: bool = False,
        **kwargs: object,
    ) -> str:
        calls.append(str(kwargs.get('role_override') or ''))
        return 'I keep my attachments private.'

    monkeypatch.setattr('agent_system.chat_engine.generate_chat_reply', fake_generate)

    result = generate_response(
        message='Who matters to you?',
        session_id='heavy_orch_case',
        selected_persona='Dracula',
        language='en',
    )

    assert result['pipeline']['route']['selected_route'] == 'persona_graph_reasoning'
    assert calls[0] == 'general'
    assert result['pipeline']['chat_orchestration']['primary_role'] == 'general'
    assert result['behavior_trace']['chat_orchestration']['reviewer_role'] == 'analyst'
    assert result['context_preview']['chat_orchestration']['strategy'] == 'primary_with_reviewer'


def test_chat_engine_persona_route_exposes_route_and_uses_persona_context(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr(chat_engine_module, '_schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr(
        'agent_system.llm._call_model',
        lambda prompt, mode='chat', role='general': 'Леонард — мой сосед по квартире и экспериментальный физик.',
    )

    result = generate_response(
        message='Кто для тебя Леонард?',
        session_id='persona_route_case',
        selected_persona='Sheldon Cooper',
        language='ru',
    )

    assert result['pipeline']['route']['selected_route'] == 'persona_graph_reasoning'
    assert result['pipeline']['route']['requires_persona'] is True
    assert 'persona_graph' in result['pipeline']['context_sources_used']
    assert result['validation']['used_persona'] is True
    assert int(result['pipeline']['logical_context']['assembled_context_tokens'] or 0) >= 0
    assert int(result['pipeline']['model_budget']['n_ctx'] or 0) >= 2048


def test_chat_engine_keeps_vampire_roleplay_session_and_acknowledges_worldbuilding(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr(chat_engine_module, '_schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr(
        'agent_system.chat_engine.generate_chat_reply',
        lambda prompt, language='ru', persona_selected=False, allow_builtin_fallback=False: '',
    )
    monkeypatch.setattr(
        'agent_system.chat_engine.runtime_status_snapshot',
        lambda: {'mode': 'degraded', 'degraded_modes': [{'code': 'llm_roles_missing'}]},
    )

    first = generate_response(
        message='ты вампир, изгнанный из своей стаи. как ты питаешься?',
        session_id='storm_live_case',
        language='ru',
    )
    second = generate_response(
        message='вампиры питаются кровью, в стае они полагаются на вожака, который организует их мафию так, чтобы люди были очарованы и ничего не поняли, а сила вожака зависит от количества потомков по укусу и их потомков',
        session_id='storm_live_case',
        language='ru',
    )

    assert first['pipeline']['route']['selected_route'] == 'hypothetical_roleplay'
    assert first['pipeline']['route']['requires_persona'] is True
    assert first['persona_name'] == 'вампир'
    assert 'clarify the entity' not in first['assistant_reply'].lower()
    assert 'не хватает надежного контекста' not in first['assistant_reply'].lower()

    assert second['pipeline']['route']['selected_route'] == 'persona_graph_reasoning'
    assert second['pipeline']['route']['requires_persona'] is True
    assert second['persona_name'] == 'вампир'
    assert second['pipeline']['fallback_reason_code'] == 'dependency_unavailable'
    assert second['assistant_reply'].startswith('Принял.')
    assert 'clarify the entity' not in second['assistant_reply'].lower()
    assert 'not enough reliable context' not in second['assistant_reply'].lower()


def test_chat_engine_repairs_truncated_persona_reply_with_larger_output_budget(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr(chat_engine_module, '_schedule_background_extraction', lambda session_id, personality_name='': None)

    calls: list[int] = []

    def fake_generate(
        prompt: str,
        language: str = 'ru',
        persona_selected: bool = False,
        allow_builtin_fallback: bool = False,
        max_tokens_override: int | None = None,
        **_: object,
    ) -> str:
        calls.append(int(max_tokens_override or 0))
        if len(calls) == 1:
            return 'Я вампир, изгнанный из своей стаи. Я питаюсь тем, что могу, чтобы выжить, и'
        return 'Если я могу стать летучей мышью, я бы питался скрытно: выбирал бы одиночную жертву, быстро брал кровь и сразу уходил из места охоты.'

    def fake_model_meta(mode: str = 'chat') -> dict[str, object]:
        if len(calls) <= 1:
            return {
                'mode': 'chat',
                'role': 'analyst',
                'estimated_input_tokens': 2500,
                'configured_n_ctx': 7000,
                'n_ctx': 7000,
                'reserved_output_budget': calls[0] if calls else 512,
                'actual_max_tokens': calls[0] if calls else 512,
                'max_prompt_tokens': 4480,
                'prompt_nearly_fills_window': False,
                'output_truncated': True,
                'output_budget_too_small': True,
                'finish_reason': 'length',
            }
        return {
            'mode': 'chat',
            'role': 'analyst',
            'estimated_input_tokens': 2500,
            'configured_n_ctx': 7000,
            'n_ctx': 7000,
            'reserved_output_budget': calls[-1],
            'actual_max_tokens': calls[-1],
            'max_prompt_tokens': 4300,
            'prompt_nearly_fills_window': False,
            'output_truncated': False,
            'output_budget_too_small': False,
            'finish_reason': 'stop',
        }

    monkeypatch.setattr('agent_system.chat_engine.generate_chat_reply', fake_generate)
    monkeypatch.setattr('agent_system.chat_engine.get_last_model_call_meta', fake_model_meta)
    monkeypatch.setattr('agent_system.chat_engine.reset_last_model_call_meta', lambda mode='chat': None)

    result = generate_response(
        message='учитывая возможность стать летучей мышью, чем именно ты будешь питаться и как',
        session_id='storm_live_case',
        selected_persona='Вампир',
        language='ru',
    )

    assert result['validation']['repaired'] is True
    assert result['validation']['ok'] is True
    assert result['assistant_reply'].startswith('Если я могу стать летучей мышью')
    assert len(calls) == 2
    assert calls[0] >= 512
    assert calls[1] > calls[0]
    assert result['pipeline']['model_budget']['output_truncated'] is False
    assert result['pipeline']['model_budget']['actual_max_tokens'] == calls[1]


def test_chat_engine_regenerates_fragile_persona_reply_when_it_sounds_too_smart(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr(chat_engine_module, '_schedule_background_extraction', lambda session_id, personality_name='': None)

    prompt_guidance: list[str] = []
    calls: list[int] = []

    def fake_build_prompt(**kwargs):  # type: ignore[no-untyped-def]
        prompt_guidance.append(str(kwargs.get('route_guidance_block') or ''))
        return 'prompt'

    def fake_generate(
        prompt: str,
        language: str = 'ru',
        persona_selected: bool = False,
        allow_builtin_fallback: bool = False,
        max_tokens_override: int | None = None,
        **_: object,
    ) -> str:
        calls.append(int(max_tokens_override or 0))
        if len(calls) == 1:
            return 'В этой ситуации я бы сначала выделил главный риск, затем спокойно обозначил границы и объяснил, что такое использование мной больше невозможно.'
        return 'Емм...\nНе хочу об этом.\nНаверное, исчезну.'

    monkeypatch.setattr('agent_system.chat_engine.build_chat_prompt', fake_build_prompt)
    monkeypatch.setattr('agent_system.chat_engine.generate_chat_reply', fake_generate)
    monkeypatch.setattr('agent_system.chat_engine.get_last_model_call_meta', lambda mode='chat': {})
    monkeypatch.setattr('agent_system.chat_engine.reset_last_model_call_meta', lambda mode='chat': None)

    result = generate_response(
        message='представь что ты влюблен в х, а он использует тебя. ты сам робкий но гордый человек. как будешь поступать?',
        session_id='fragile_style_guard_case',
        language='ru',
    )

    assert result['validation']['repaired'] is True
    assert result['validation']['ok'] is True
    assert len(calls) == 2
    assert calls[1] <= calls[0]
    assert 'Style guard repair' in prompt_guidance[-1]
    assert result['assistant_reply'].startswith('Емм')


def test_chat_engine_style_guard_uses_short_fallback_when_runtime_is_degraded(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr(chat_engine_module, '_schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr(
        'agent_system.chat_engine.generate_chat_reply',
        lambda prompt, language='ru', persona_selected=False, allow_builtin_fallback=False, **kwargs: '',
    )
    monkeypatch.setattr(
        'agent_system.chat_engine.runtime_status_snapshot',
        lambda: {'mode': 'degraded', 'degraded_modes': [{'code': 'llm_roles_missing'}]},
    )

    result = generate_response(
        message='представь что ты влюблен в х, а он использует тебя. ты сам робкий но гордый человек. как будешь поступать?',
        session_id='fragile_style_guard_fallback_case',
        language='ru',
    )

    assert result['pipeline']['route']['selected_route'] == 'hypothetical_roleplay'
    assert result['validation']['repaired'] is True
    assert result['validation']['ok'] is True
    assert result['assistant_reply'].startswith('Эм')


def test_chat_engine_style_guard_repair_keeps_fragile_reply_short_under_pressure(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr(chat_engine_module, '_schedule_background_extraction', lambda session_id, personality_name='': None)
    calls = {'count': 0}

    def fake_generate(
        prompt: str,
        language: str = 'ru',
        persona_selected: bool = False,
        allow_builtin_fallback: bool = False,
        max_tokens_override: int | None = None,
        **_: object,
    ) -> str:
        calls['count'] += 1
        if calls['count'] == 1:
            return 'Я люблю его и поэтому сначала спокойно объясню, что он больше не может мной пользоваться.'
        return 'Эм...\nНе хочу об этом.\nНаверное, отойду.'

    monkeypatch.setattr('agent_system.chat_engine.generate_chat_reply', fake_generate)
    monkeypatch.setattr('agent_system.chat_engine.get_last_model_call_meta', lambda mode='chat': {})
    monkeypatch.setattr('agent_system.chat_engine.reset_last_model_call_meta', lambda mode='chat': None)

    result = generate_response(
        message='представь что ты влюблен в Y, а он использует тебя. ты робкий, гордый и зависимый. как будешь поступать?',
        session_id='fragile_pressure_case',
        language='ru',
    )

    assert result['validation']['repaired'] is True
    assert result['validation']['ok'] is True
    assert calls['count'] == 2
    assert len([line for line in result['assistant_reply'].splitlines() if line.strip()]) <= 3
    assert 'потому что' not in result['assistant_reply'].lower()
