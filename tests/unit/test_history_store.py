from __future__ import annotations

from roaches_viz.roaches_viz.history_store import append_turn, create_session, infer_current_entity, list_sessions, parse_session, recent_dialogue, session_files_dir


def test_history_store_creates_parses_and_lists_sessions(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    created = create_session('session_test', 'Test title')
    append_turn('session_test', 'знаешь дракулу?', 'Да, знаю.')

    parsed = parse_session('session_test')
    sessions = list_sessions()

    assert created['session_id'] == 'session_test'
    assert parsed is not None
    assert parsed['title'] == 'Test title'
    assert len(parsed['messages']) == 2
    assert sessions[0]['session_id'] == 'session_test'
    assert infer_current_entity('session_test') == 'dracula'
    assert session_files_dir('session_test').exists()


def test_history_store_recent_dialogue_trims_to_recent_messages(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    create_session('session_test', 'Trim')
    for index in range(10):
        append_turn('session_test', f'user {index}', f'assistant {index}')

    context = recent_dialogue('session_test', max_messages=3, max_tokens_equivalent=20)

    assert 'assistant 9' in context
    assert 'assistant 0' not in context
