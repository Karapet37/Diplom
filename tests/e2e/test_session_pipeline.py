from __future__ import annotations

import os

from fastapi.testclient import TestClient

from src.web.combined_app import create_combined_app


def test_session_pipeline(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('roaches_viz.roaches_viz.chat_engine._schedule_background_work', lambda session_id, personality_name='': None)
    monkeypatch.setattr('roaches_viz.roaches_viz.llm._call_model', lambda prompt, mode='chat': 'Да, это персонаж-вампир.' if mode == 'chat' else '{"proposals":[{"entity":"Dracula","type":"PERSON","traits":["вампир"],"relations":[{"type":"FEEDS_ON","target":"люди"}]}]}')

    with TestClient(create_combined_app()) as client:
        client.post('/api/cognitive/sessions', json={'session_id': 'session_test', 'title': 'Dracula test'})
        first = client.post('/api/cognitive/chat/respond', json={'session_id': 'session_test', 'message': 'знаешь дракулу?', 'language': 'ru'})
        assert first.status_code == 200
        assert first.json()['assistant_reply']

        session_log_path = tmp_path / 'memory' / 'sessions' / 'session_test.txt'
        assert session_log_path.exists()
        session_log = session_log_path.read_text(encoding='utf-8').lower()
        assert 'user: знаешь дракулу?' in session_log
        assert 'assistant:' in session_log

        rebuilt = client.post('/api/cognitive/rebuild', json={'session_id': 'session_test'})
        assert rebuilt.status_code == 200

        assert os.path.exists(tmp_path / 'memory' / 'graphs' / 'nodes.json')
        assert 'dracula' in (tmp_path / 'memory' / 'graphs' / 'nodes.json').read_text(encoding='utf-8').lower()
