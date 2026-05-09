from __future__ import annotations

from pathlib import Path

from agent_system.chat_engine import generate_response
from agent_system.history_store import parse_session, session_messages_path
from agent_system.message_annotation_store import (
    build_annotation_workspace,
    build_runtime_message_vector_payload,
    save_message_annotation,
)


def test_chat_engine_preserves_raw_user_text_in_structured_session(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat', **kwargs: 'Ответ без изменений.')

    raw_message = 'привет катя, чё по чём?'
    result = generate_response(
        message=raw_message,
        session_id='raw_text_session',
        language='ru',
    )

    parsed = parse_session('raw_text_session')

    assert result['session']['messages'][0]['raw_text'] == raw_message
    assert parsed is not None
    assert parsed['messages'][0]['raw_text'] == raw_message
    assert parsed['messages'][0]['display_text'] == raw_message
    assert parsed['messages'][0]['message'] == raw_message
    assert session_messages_path('raw_text_session').exists()


def test_annotation_workspace_builds_registry_vector_and_context_matrix(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    from agent_system.history_store import append_turn, create_session

    create_session('annotation_workspace', 'Annotation Workspace')
    append_turn('annotation_workspace', 'Почему ты так сказал?', 'Потому что я видел риск.')
    append_turn('annotation_workspace', 'Ну молодец, конечно.', 'Я слышу сарказм.')

    workspace = build_annotation_workspace('annotation_workspace', window_size=4)

    assert workspace['message_count'] == 4
    assert len(workspace['registry']) == 49

    latest = workspace['messages'][-1]
    assert latest['message_id']
    assert len(latest['context_matrix']) == 3
    assert latest['context_window'] == [row['message_id'] for row in workspace['messages'][-4:-1]]
    assert set(latest['vector'].keys()) == {f'P{index}' for index in range(1, 50)}
    assert {'main', 'extra'} <= set(latest['vector']['F1'].keys())
    assert isinstance(latest['transition_interpretation'], dict)


def test_annotation_store_saves_correction_layer_separately(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    from agent_system.history_store import append_turn, create_session

    create_session('annotation_api', 'Annotation API')
    append_turn('annotation_api', 'Ну молодец, можешь когда хочешь.', 'Ладно.')

    workspace = build_annotation_workspace('annotation_api')
    latest = workspace['messages'][-1]

    save_message_annotation(
        session_id='annotation_api',
        message_payload=latest,
        coordinates={
            'F1': {'main': 'statement', 'extra': ['question']},
            'F24': {'main': 'false_praise', 'extra': ['sarcasm']},
            'F46': {'main': 'false_praise', 'extra': []},
            'F49': {'main': 'toward_masking', 'extra': []},
        },
        context_window=latest['context_window'],
        context_matrix=latest['context_matrix'],
        transition_interpretation={
            'from': ['sharpening'],
            'to': ['false_praise', 'toward_masking'],
            'type': 'masking',
        },
        notes='Исправлено через UI.',
    )

    saved_workspace = build_annotation_workspace('annotation_api')
    saved_latest = saved_workspace['messages'][-1]

    assert saved_latest['has_correction'] is True
    assert saved_latest['vector']['F24']['main'] == 'false_praise'
    assert saved_latest['vector']['F24']['extra'] == ['sarcasm']
    assert saved_latest['transition_interpretation']['type'] == 'masking'

    annotation_file = Path(tmp_path / 'memory' / 'message_annotations' / 'annotation_api.json')
    dataset_file = Path(tmp_path / 'memory' / 'message_annotations' / 'global.jsonl')
    assert annotation_file.exists()
    assert dataset_file.exists()


def test_runtime_message_vector_payload_uses_corrected_context_matrix(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    from agent_system.history_store import append_turn, create_session

    create_session('annotation_runtime', 'Annotation Runtime')
    append_turn('annotation_runtime', 'Ну молодец, можешь когда хочешь.', 'Это звучит грубо.')

    workspace = build_annotation_workspace('annotation_runtime')
    last_assistant = workspace['messages'][-1]
    save_message_annotation(
        session_id='annotation_runtime',
        message_payload=last_assistant,
        coordinates={
            'F13': {'main': 'attack', 'extra': []},
            'F35': {'main': 'escalation', 'extra': []},
            'F49': {'main': 'toward_escalation', 'extra': []},
        },
        context_window=last_assistant['context_window'],
        context_matrix=last_assistant['context_matrix'],
        transition_interpretation={
            'from': ['sharpening'],
            'to': ['escalation', 'toward_escalation'],
            'type': 'escalation',
        },
        notes='Исправление для runtime context matrix.',
    )

    runtime_payload = build_runtime_message_vector_payload(
        'annotation_runtime',
        message_text='Почему ты так резко ответила?',
        role='user',
        persona_name='Катерина',
    )

    assert runtime_payload['context_window']
    assert runtime_payload['context_matrix'][-1]['vector']['F35']['main'] == 'escalation'
    assert runtime_payload['context_matrix'][-1]['vector']['F49']['main'] == 'toward_escalation'
    assert runtime_payload['current_vector']['F1']['main'] in {'question', 'statement'}
