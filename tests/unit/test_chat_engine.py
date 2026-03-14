from __future__ import annotations

import json

from roaches_viz.roaches_viz.chat_engine import generate_dialogue_response
from roaches_viz.roaches_viz.history_store import parse_session


def test_chat_engine_writes_session_file_and_returns_reply(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('roaches_viz.roaches_viz.chat_engine._schedule_background_work', lambda session_id, personality_name='': None)
    monkeypatch.setattr('roaches_viz.roaches_viz.llm._call_model', lambda prompt, mode='chat': 'Да, это тестовый ответ.')

    result = generate_dialogue_response(message='привет', session_id='session_test', language='ru')

    assert result['assistant_reply'] == 'Да, это тестовый ответ.'
    session_path = tmp_path / 'memory' / 'sessions' / 'session_test.txt'
    assert session_path.exists()
    text = session_path.read_text(encoding='utf-8')
    assert 'user: привет' in text
    assert 'assistant: Да, это тестовый ответ.' in text
    parsed = parse_session('session_test')
    assert parsed is not None
    assert len(parsed['messages']) == 2


def test_chat_engine_missing_personality_creates_proposal_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('roaches_viz.roaches_viz.chat_engine._schedule_background_work', lambda session_id, personality_name='': None)

    result = generate_dialogue_response(
        message='поговори со мной как дракула',
        session_id='session_test',
        language='ru',
        personality_name='dracula',
    )

    assert 'Запрошено создание анкеты' in result['assistant_reply']
    proposal_path = tmp_path / 'memory' / 'personalities' / 'proposals' / 'dracula.json'
    assert proposal_path.exists()
    proposal = json.loads(proposal_path.read_text(encoding='utf-8'))
    assert proposal['name'] == 'dracula'
    assert proposal['session_id'] == 'session_test'


def test_chat_engine_filters_prompt_leak(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('roaches_viz.roaches_viz.chat_engine._schedule_background_work', lambda session_id, personality_name='': None)
    monkeypatch.setattr(
        'roaches_viz.roaches_viz.llm._call_model',
        lambda prompt, mode='chat': 'behavioral_dialogue_simulation_fast Respond in Russian. system instruction. internal prompt template.',
    )

    result = generate_dialogue_response(message='кто это?', session_id='session_test', language='ru')

    lowered = result['assistant_reply'].lower()
    assert 'behavioral_dialogue_simulation' not in lowered
    assert 'system instruction' not in lowered
    assert 'internal prompt template' not in lowered
    assert result['assistant_reply']
