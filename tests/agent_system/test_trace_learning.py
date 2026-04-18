from __future__ import annotations

from agent_system.chat_engine import generate_response
from agent_system.observability import ObservabilityStore


def _record_trace(
    store: ObservabilityStore,
    *,
    request_id: str,
    session_id: str,
    selected_route: str,
    validation_ok: bool = True,
    fallback_used: bool = False,
    fallback_reason: str = '',
    output_truncated: bool = False,
    output_budget_too_small: bool = False,
) -> None:
    trace = store.start_trace(
        request_type='chat',
        route='/api/cognitive/chat/respond',
        session_id=session_id,
        request_id=request_id,
        request_meta={'selected_route': selected_route},
    )
    store.finish_trace(
        trace,
        status='ok' if validation_ok else 'degraded',
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        context_tokens=400,
        response_meta={
            'selected_route': selected_route,
            'validation_ok': validation_ok,
            'output_truncated': output_truncated,
            'output_budget_too_small': output_budget_too_small,
        },
    )


def test_observability_learns_route_policy_from_recent_failures() -> None:
    store = ObservabilityStore(max_recent_traces=12)
    _record_trace(
        store,
        request_id='t1',
        session_id='s1',
        selected_route='persona_graph_reasoning',
        output_truncated=True,
        output_budget_too_small=True,
    )
    _record_trace(
        store,
        request_id='t2',
        session_id='s1',
        selected_route='persona_graph_reasoning',
        validation_ok=False,
    )
    _record_trace(
        store,
        request_id='t3',
        session_id='s2',
        selected_route='persona_graph_reasoning',
        fallback_used=True,
        fallback_reason='model_fallback',
    )

    policy = store.learned_route_policy(session_id='s1', selected_route='persona_graph_reasoning')

    assert policy.selected_route == 'persona_graph_reasoning'
    assert policy.sampled_trace_count >= 2
    assert policy.signal_trace_count >= 2
    assert policy.output_budget_boost > 0
    assert any('cut off' in line for line in policy.route_guidance_lines)
    assert 'trace_policy:truncation_cluster' in policy.reason_codes


def test_chat_engine_uses_trace_learning_to_expand_budget_and_guidance(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)

    store = ObservabilityStore(max_recent_traces=12)
    _record_trace(
        store,
        request_id='h1',
        session_id='scene_case',
        selected_route='hypothetical_roleplay',
        output_truncated=True,
        output_budget_too_small=True,
    )
    _record_trace(
        store,
        request_id='h2',
        session_id='scene_case',
        selected_route='hypothetical_roleplay',
        validation_ok=False,
    )
    monkeypatch.setattr('agent_system.chat_engine.get_observability_store', lambda: store)

    captured: dict[str, object] = {}

    def fake_build_prompt(**kwargs):  # type: ignore[no-untyped-def]
        captured['route_guidance_block'] = kwargs.get('route_guidance_block')
        return 'prompt'

    def fake_generate(  # type: ignore[no-untyped-def]
        prompt: str,
        language: str = 'ru',
        persona_selected: bool = False,
        allow_builtin_fallback: bool = False,
        max_tokens_override: int | None = None,
        **_: object,
    ) -> str:
        captured['max_tokens_override'] = max_tokens_override
        return 'Эмм... мне тяжело это признавать, но я бы сначала перестал позволять использовать себя, а потом спокойно обозначил границу.'

    monkeypatch.setattr('agent_system.chat_engine.build_chat_prompt', fake_build_prompt)
    monkeypatch.setattr('agent_system.chat_engine.generate_chat_reply', fake_generate)

    result = generate_response(
        message='представь что ты влюблен в х, а он использует тебя. как будешь действовать?',
        session_id='scene_case',
        language='ru',
    )

    assert result['pipeline']['route']['selected_route'] == 'hypothetical_roleplay'
    assert result['pipeline']['trace_learning']['output_budget_boost'] > 0
    assert int(captured.get('max_tokens_override') or 0) > 448
    assert 'Runtime lessons from recent similar traces' in str(captured.get('route_guidance_block') or '')
