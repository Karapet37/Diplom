from __future__ import annotations

from agent_system.controller_runtime import build_controller_state
from agent_system.models import RouteMemory


def test_controller_runtime_builds_persona_fast_path_state() -> None:
    state = build_controller_state(
        request_id='req1',
        session_id='session_fast',
        message='Привет, я Маша, а ты?',
        selected_persona='Катерина',
        current_entity='Катерина',
        session_persona='Катерина',
        route_memory=RouteMemory(previous_persona_name='Катерина'),
        known_entities=[],
    )

    assert state.preprocessing.request_type == 'persona_chat'
    assert state.route.selected_route == 'persona_chat_fast_path'
    assert state.route.fast_path is True
    assert state.capability_plan.use_heavy_persona_pipeline is False
    assert state.capability_plan.use_context_builder is False


def test_controller_runtime_builds_persona_specification_state() -> None:
    state = build_controller_state(
        request_id='req2',
        session_id='session_persona',
        message='Создай личность\nКатерина — сильная, холодная, собранная женщина с острым языком.',
        known_entities=[],
    )

    assert state.preprocessing.request_type == 'persona_specification'
    assert state.route.selected_route == 'persona_specification'
    assert state.capability_plan.preferred_generation_mode == 'structured_persona_action'
