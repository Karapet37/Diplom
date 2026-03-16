from __future__ import annotations

import json

from agent_system.chat_engine import generate_response
from agent_system.history_store import parse_session
from agent_system.persona_engine import load_persona


def test_chat_engine_spawns_head_updates_emotions_and_learns_reaction(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat': 'I answer in the first person as Dracula.')

    result = generate_response(
        message='Speak as Dracula the vampire count, and why should I trust you?',
        session_id='session_test',
        selected_persona='Dracula',
        language='en',
    )

    assert result['assistant_reply'] == 'I answer in the first person as Dracula.'
    assert result['persona_name'] == 'dracula'
    session_path = tmp_path / 'memory' / 'sessions' / 'session_test.txt'
    assert session_path.exists()
    parsed = parse_session('session_test')
    assert parsed is not None
    assert len(parsed['messages']) == 2

    bundle = load_persona('dracula')
    assert bundle is not None
    assert bundle.emotion_vector['curiosity'] > 0.55
    payload = json.loads((tmp_path / 'memory' / 'heads' / 'dracula' / 'examples.json').read_text(encoding='utf-8'))
    assert 'I answer in the first person as Dracula.' not in payload['examples']
    assert payload['situation_reactions']
    assert 'Speak as Dracula' in payload['situation_reactions'][0]['situation']


def test_chat_engine_routes_lowercase_entity_mentions(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat': 'I am Dracula, a vampire nobleman.')

    result = generate_response(
        message='tell me about dracula the vampire nobleman',
        session_id='session_lowercase',
        language='en',
    )

    assert result['persona_name'] == 'dracula'
    assert 'dracula' in [item.lower() for item in result['analysis']['entities']]
    assert load_persona('dracula') is not None


def test_chat_engine_surfaces_background_repair_failures(tmp_path, monkeypatch) -> None:
    class ImmediateExecutor:
        def submit(self, fn):
            fn()
            return None

    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._BACKGROUND_EXECUTOR', ImmediateExecutor())
    monkeypatch.setattr('agent_system.chat_engine.rebuild_artifacts', lambda session_id, personality_name='': (_ for _ in ()).throw(RuntimeError('repair exploded')))
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat': 'I answer in first person.')

    result = generate_response(
        message='Speak as Dracula the vampire count.',
        session_id='session_failure',
        selected_persona='Dracula',
        language='en',
    )

    assert result['repair_status']['status'] == 'error'
    assert 'repair exploded' in result['repair_status']['error']
