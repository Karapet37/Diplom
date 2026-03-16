from __future__ import annotations

import pytest


def test_api_surfaces_chat_runtime(tmp_path, monkeypatch) -> None:
    fastapi = pytest.importorskip('fastapi')
    testclient = pytest.importorskip('fastapi.testclient')
    assert fastapi

    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat': 'I am Dracula.')

    from agent_system.api import create_app

    client = testclient.TestClient(create_app())

    health = client.get('/api/cognitive/health')
    assert health.status_code == 200
    assert health.json()['runtime'] == 'persona-graph-agent'

    reply = client.post(
        '/api/cognitive/chat/respond',
        json={
            'session_id': 'api_session',
            'message': 'Speak as Dracula the vampire count.',
            'selected_persona': 'Dracula',
            'language': 'en',
        },
    )
    assert reply.status_code == 200
    payload = reply.json()
    assert payload['assistant_reply'] == 'I am Dracula.'
    assert payload['persona_name'] == 'dracula'
    assert payload['repair_status'] == {}
