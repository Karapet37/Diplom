from __future__ import annotations

from fastapi.testclient import TestClient

from src.web.combined_app import create_combined_app


def _fake_model(prompt: str, mode: str = 'chat') -> str:
    lowered = prompt.lower()
    if 'schema:' in lowered and 'george' in lowered and 'anna' in lowered:
        return '{"proposals":[{"entity":"George","type":"PERSON","traits":["confident"],"relations":[{"type":"KNOWS","target":"Anna"}]}]}'
    if 'schema:' in lowered and 'session text:' in lowered:
        return '{"proposals":[]}'
    return 'ok'


def test_file_ingestion_pipeline(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('roaches_viz.roaches_viz.llm._call_model', _fake_model)

    with TestClient(create_combined_app()) as client:
        client.post('/api/cognitive/sessions', json={'session_id': 'session_file', 'title': 'File test'})
        response = client.post(
            '/api/cognitive/files/upload',
            data={'session_id': 'session_file'},
            files=[('files', ('george.txt', b'George knows Anna and sounds confident.', 'text/plain'))],
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload['stored_files']

        stored_path = tmp_path / 'memory' / 'files' / 'uploaded_documents' / 'session_file' / 'george.txt'
        assert stored_path.exists()

        graph = client.get('/api/cognitive/graph')
        assert graph.status_code == 200
        graph_text = str(graph.json()).lower()
        assert 'george' in graph_text
        assert 'anna' in graph_text
