from __future__ import annotations

import json

from agent_system.chat_engine import generate_response


def test_chat_runtime_persists_current_context_and_transition_log(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setenv('COGNITIVE_RUNTIME_DIR', str(tmp_path / 'runtime'))
    monkeypatch.setenv('COGNITIVE_STAGE_MODEL_STEPS', '')
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat', role='general': 'I am Dracula, and I answer from the reviewed state.')

    payload = generate_response(
        message='Tell me about Dracula and how you decide what matters.',
        session_id='stateful-turn',
        selected_persona='Dracula',
        language='en',
    )

    current_context_path = tmp_path / 'runtime' / 'current_context' / 'current_context.json'
    transition_log_path = tmp_path / 'runtime' / 'logs' / 'state_transitions.jsonl'

    assert current_context_path.exists()
    assert transition_log_path.exists()
    assert payload['side_effects']['current_context_path'] == str(current_context_path)
    assert payload['side_effects']['transition_log_path'] == str(transition_log_path)

    current_context = json.loads(current_context_path.read_text(encoding='utf-8'))
    transition_rows = [json.loads(line) for line in transition_log_path.read_text(encoding='utf-8').splitlines() if line.strip()]

    assert current_context['turn_id'] == payload['trace_id']
    assert current_context['reviewed_context']['summary']
    assert current_context['response_plan']['behavior_mode']
    assert transition_rows
    assert transition_rows[-1]['turn_id'] == payload['trace_id']
    assert transition_rows[-1]['previous_state']['summary']
    assert transition_rows[-1]['interpreted_influence']['summary']
    assert transition_rows[-1]['new_state']['summary']
    assert transition_rows[-1]['reviewed_context']['summary']
    assert transition_rows[-1]['selected_response_mode']['behavior_mode']
    assert transition_rows[-1]['final_response_summary']

    assert payload['state_transition']['previous_state']['summary']
    assert payload['state_transition']['influence']['summary']
    assert payload['state_transition']['updated_state']['summary']
    assert payload['state_transition']['working_context']['summary']
    assert payload['state_transition']['reviewed_context']['summary']
    assert payload['state_transition']['response_plan']['behavior_mode']
