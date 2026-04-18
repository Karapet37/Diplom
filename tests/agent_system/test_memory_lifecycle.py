from __future__ import annotations

from agent_system.graph_store import GraphStore
from agent_system.history_store import append_turn, create_session, delete_session, infer_current_entity, parse_active_session, parse_session, recent_dialogue, save_session_route_state, session_files_dir, session_text_path
from agent_system.memory_layers import append_session_archive, describe_memory_layers, load_persona_overflow_archive, load_session_archive
from agent_system.persona_engine import load_persona, materialize_persona


def test_session_memory_archives_old_turns_but_preserves_full_session_view(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setenv('COGNITIVE_SESSION_ARCHIVE_AFTER_MESSAGES', '8')
    monkeypatch.setenv('COGNITIVE_SESSION_KEEP_RECENT_MESSAGES', '4')

    create_session('session_lifecycle', 'Lifecycle Session')
    for index in range(6):
        append_turn('session_lifecycle', f'user turn {index}', f'assistant turn {index}')

    active = parse_active_session('session_lifecycle')
    combined = parse_session('session_lifecycle')
    archive = load_session_archive('session_lifecycle')

    assert active is not None
    assert combined is not None
    assert len(active['messages']) == 4
    assert len(list(archive.get('archived_messages') or [])) == 8
    assert len(combined['messages']) == 12
    assert combined['archived_message_count'] == 8
    assert combined['active_message_count'] == 4
    assert 'user: user turn 5' in recent_dialogue('session_lifecycle')
    assert 'user turn 0' not in recent_dialogue('session_lifecycle')


def test_persona_memory_archives_overflow_while_active_bundle_stays_bounded(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setenv('COGNITIVE_PERSONA_EXAMPLE_LIMIT', '6')
    monkeypatch.setenv('COGNITIVE_PERSONA_REACTION_LIMIT', '4')
    monkeypatch.setenv('COGNITIVE_PERSONA_LOG_TUPLE_LIMIT', '5')
    monkeypatch.setenv('COGNITIVE_PERSONA_RELATION_LIMIT', '3')
    monkeypatch.setenv('COGNITIVE_PERSONA_TRAIT_LIMIT', '4')
    monkeypatch.setenv('COGNITIVE_PERSONA_KNOWLEDGE_CHAR_LIMIT', '120')

    materialize_persona(
        'Archive Persona',
        {
            'entity_type': 'PERSON',
            'traits': [f'trait_{index}' for index in range(8)],
            'examples': [f'Example {index}' for index in range(12)],
            'relations': [{'type': 'KNOWS', 'target': f'Target {index}'} for index in range(7)],
            'situation_reactions': [{'situation': f'situation {index}', 'reaction': f'reaction {index}'} for index in range(9)],
            'log_tuples': [{'tuple': ['utterance_pattern', f'pattern_{index}'], 'frequency': index + 1, 'sample': f'sample {index}'} for index in range(8)],
            'knowledge': 'K' * 240,
        },
        explicit=True,
    )

    bundle = load_persona('archive_persona')
    archive = load_persona_overflow_archive('archive_persona')

    assert bundle is not None
    assert len(bundle.traits) == 4
    assert len(bundle.examples) == 6
    assert len(bundle.relations) == 3
    assert len(bundle.situation_reactions) == 4
    assert len(bundle.log_tuples) == 5
    assert len(bundle.knowledge) == 120
    assert archive['entries']
    latest = archive['entries'][0]
    assert len(list(latest.get('traits') or [])) == 4
    assert len(list(latest.get('examples') or [])) == 6
    assert len(list(latest.get('relations') or [])) == 4
    assert len(list(latest.get('situation_reactions') or [])) == 5
    assert len(list(latest.get('log_tuples') or [])) == 3
    assert str(latest.get('knowledge_overflow') or '') == ('K' * 120)


def test_graph_snapshot_uses_cold_archive_without_touching_active_graph(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    store = GraphStore()
    store.merge_extraction(
        {
            'entities': [
                {
                    'name': 'Physics',
                    'type': 'CONCEPT',
                    'description': 'Physics studies matter and energy.',
                    'facts': ['Physics explains motion.'],
                    'confidence': 0.93,
                }
            ],
            'relations': [],
        },
        source='file',
    )

    snapshot = store.snapshot_graph(reason='test_snapshot')
    description = describe_memory_layers()

    assert snapshot['ok'] is True
    assert snapshot['node_count'] >= 1
    assert 'archive_graphs_dir' in description['paths']
    assert snapshot['path'].startswith(description['paths']['archive_graphs_dir'])


def test_infer_current_entity_ignores_assistant_generated_phrase_noise(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    create_session('entity_drift', 'Entity Drift')
    append_turn(
        'entity_drift',
        'Tell me about Dracula.',
        'I focus on stabilizing patients and staying calm under pressure.',
    )

    assert infer_current_entity('entity_drift') == 'Dracula'


def test_infer_current_entity_keeps_multiword_name_from_last_user_turn(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    create_session('entity_multiword', 'Entity Multiword')
    append_turn(
        'entity_multiword',
        'Who is Jack Sparrow?',
        'Jack Sparrow is a pirate. What about him?',
    )

    assert infer_current_entity('entity_multiword') == 'Jack Sparrow'


def test_delete_session_removes_text_route_state_uploads_and_archive(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    create_session('delete_me', 'Delete Me')
    append_turn('delete_me', 'user 0', 'assistant 0')
    append_session_archive(
        'delete_me',
        title='Delete Me',
        messages=[
            {'timestamp': '2026-01-01T00:00:00Z', 'role': 'user', 'message': 'older user'},
            {'timestamp': '2026-01-01T00:00:01Z', 'role': 'assistant', 'message': 'older assistant'},
        ],
        reason='test_setup',
    )

    save_session_route_state('delete_me', {'selected_route': 'factual_answer'})
    upload_path = session_files_dir('delete_me') / 'notes.txt'
    upload_path.write_text('hello', encoding='utf-8')
    archive = load_session_archive('delete_me')

    assert session_text_path('delete_me').exists()
    assert upload_path.exists()
    assert archive['archived_messages']

    result = delete_session('delete_me')

    assert result['ok'] is True
    assert parse_session('delete_me') is None
    assert not upload_path.exists()
    assert not any('delete_me.txt' in path for path in result['missing_paths'])
