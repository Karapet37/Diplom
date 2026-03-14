from __future__ import annotations

import json

from roaches_viz.roaches_viz.graph_store import graph_nodes_path, personality_profile_path
from roaches_viz.roaches_viz.history_store import append_turn, create_session
from roaches_viz.roaches_viz.knowledge_extractor import _chunk_text, extract_file, extract_session, process_personality_proposals, request_missing_personality, store_uploaded_file


def _fake_model(prompt: str, mode: str = 'chat') -> str:
    lowered = prompt.lower()
    if 'schema:' in lowered and 'session text:' in lowered:
        return json.dumps({
            'proposals': [
                {
                    'entity': 'Dracula',
                    'type': 'PERSON',
                    'traits': ['вампир', 'аристократический', 'хищный'],
                    'relations': [
                        {'type': 'FEEDS_ON', 'target': 'люди'},
                        {'type': 'FEARS', 'target': 'солнечный свет'},
                    ],
                }
            ]
        }, ensure_ascii=False)
    if 'schema:' in lowered and 'name: dracula' in lowered:
        return json.dumps({
            'name': 'dracula',
            'traits': ['вампир', 'аристократический', 'хищный'],
            'patterns': ['пьет_кровь', 'избегает_солнца'],
            'examples': ['знаешь дракулу?'],
        }, ensure_ascii=False)
    return ''


def test_knowledge_extractor_applies_valid_json_and_discards_invalid(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('roaches_viz.roaches_viz.llm._call_model', _fake_model)
    create_session('session_test', 'Session')
    append_turn('session_test', 'знаешь дракулу?', 'Да.')

    result = extract_session('session_test')
    assert result['ok'] is True
    assert 'dracula' in graph_nodes_path().read_text(encoding='utf-8').lower()

    monkeypatch.setattr('roaches_viz.roaches_viz.llm._call_model', lambda prompt, mode='chat': 'not json')
    invalid = extract_session('session_test')
    assert invalid['reason'] == 'no_valid_proposals'


def test_knowledge_extractor_chunks_big_files_and_materializes_personality(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('roaches_viz.roaches_viz.llm._call_model', _fake_model)

    large_text = ('Dracula is a vampire.\n\n' * 800).encode('utf-8')
    path = store_uploaded_file('session_test', 'dracula.txt', large_text)
    file_result = extract_file(path)
    request_missing_personality('dracula', 'missing', 'session_test', 'знаешь дракулу?')
    personality_results = process_personality_proposals()

    assert len(_chunk_text(large_text.decode('utf-8'))) > 1
    assert file_result['ok'] is True
    assert personality_results
    assert personality_profile_path('dracula').exists()
