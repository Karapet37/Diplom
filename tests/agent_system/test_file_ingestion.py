from __future__ import annotations

import json

from agent_system.file_ingestion import chunk_text, ingest_file, rebuild_artifacts, store_uploaded_file
from agent_system.graph_store import GraphStore
from agent_system.history_store import append_turn, create_session
from agent_system.persona_engine import load_persona


def test_file_ingestion_chunks_under_token_budget_and_updates_heads(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    def fake_model(prompt: str, mode: str = 'chat') -> str:
        if mode != 'knowledge':
            return ''
        return json.dumps(
            {
                'entities': [
                    {
                        'name': 'Dracula',
                        'aliases': ['Count Dracula'],
                        'description': 'Fictional vampire nobleman.',
                        'facts': ['Dracula feeds on humans.', 'Dracula fears sunlight.'],
                        'context': {'source': 'file'},
                    },
                    {'name': 'humans', 'aliases': [], 'description': 'People.', 'facts': [], 'context': {'source': 'file'}},
                    {'name': 'sunlight', 'aliases': [], 'description': 'Daylight.', 'facts': [], 'context': {'source': 'file'}},
                ],
                'relations': [
                    {'from': 'Dracula', 'to': 'humans', 'type': 'FEEDS_ON', 'weight': 0.9},
                    {'from': 'Dracula', 'to': 'sunlight', 'type': 'FEARS', 'weight': 0.8},
                ],
            }
        )

    monkeypatch.setattr('agent_system.llm._call_model', fake_model)

    large_text = ('Dracula is a vampire nobleman who fears sunlight.\n\n' * 600).strip()
    chunks = chunk_text(large_text)
    assert len(chunks) > 1
    assert all(len(chunk) <= 8000 for chunk in chunks)

    create_session('session_test', 'Session')
    path = store_uploaded_file('session_test', 'dracula.txt', large_text.encode('utf-8'))
    result = ingest_file(path)
    assert result['ok'] is True

    bundle = load_persona('dracula')
    assert bundle is not None
    assert any(relation['type'] == 'FEEDS_ON' for relation in bundle.relations)
    assert load_persona('humans') is None
    assert load_persona('sunlight') is None
    graph = GraphStore().load_graph()
    dracula = next(node for node in graph['nodes'] if str(node.get('name') or '').lower() == 'dracula')
    assert 'Count Dracula' in dracula['aliases']


def test_rebuild_artifacts_learns_from_session_and_uploaded_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    def fake_model(prompt: str, mode: str = 'chat') -> str:
        if mode != 'knowledge':
            return ''
        return json.dumps(
            {
                'entities': [{'name': 'Dracula', 'aliases': [], 'description': 'Fictional vampire nobleman.', 'facts': ['Dracula is a vampire.'], 'context': {'source': 'session'}}],
                'relations': [],
            }
        )

    monkeypatch.setattr('agent_system.llm._call_model', fake_model)

    create_session('session_test', 'Session')
    append_turn('session_test', 'Do you know Dracula?', 'Yes.')
    store_uploaded_file('session_test', 'notes.md', b'Dracula is a fictional vampire nobleman.')
    result = rebuild_artifacts('session_test', personality_name='Dracula')

    assert result['ok'] is True
    assert result['validation']['ok'] is True
    assert not result['errors']
    bundle = load_persona('dracula')
    assert bundle is not None
    assert 'Yes.' not in bundle.examples
