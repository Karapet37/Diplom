from __future__ import annotations

import pytest

from agent_system.reliability import MutationRejectedFailure, StorageWriteFailure, runtime_status_snapshot


def test_graph_save_rolls_back_when_write_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    import agent_system.graph_store as graph_store_module
    from agent_system.graph_store import GraphStore

    store = GraphStore()
    store.create_node(name='Human', node_type='CONCEPT', description='Human is a concept node.')
    original_nodes = store.load_nodes()
    original_edges = store.load_edges()
    candidate = store._candidate_node(  # noqa: SLF001
        name='Biology',
        entity_type='CONCEPT',
        description='Biology is a concept node.',
        source='test',
    )
    next_nodes = [dict(item) for item in original_nodes] + [candidate]

    original_write_json = graph_store_module.write_json
    calls = {'count': 0}

    def flaky_write_json(path, payload) -> None:
        calls['count'] += 1
        if calls['count'] == 1:
            raise OSError('simulated write failure')
        original_write_json(path, payload)

    monkeypatch.setattr(graph_store_module, 'write_json', flaky_write_json)

    with pytest.raises(StorageWriteFailure) as exc_info:
        store.save_graph(next_nodes, list(original_edges), reason='patch_node', apply_hygiene=False, snapshot_before_write=False)

    assert exc_info.value.details['reason'] == 'patch_node'
    assert store.load_nodes() == original_nodes
    assert store.load_edges() == original_edges


def test_materialize_persona_rolls_back_when_sync_head_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    from agent_system.persona_engine import list_persona_snapshots, load_persona, materialize_persona

    original = materialize_persona(
        'Dracula',
        {
            'entity_type': 'PERSON',
            'traits': ['aristocratic', 'predatory'],
            'examples': ['I am Dracula.'],
            'knowledge': 'Dracula is a fictional vampire count.',
        },
        explicit=True,
    )
    assert 'fictional vampire count' in original.knowledge

    def broken_sync_head(self, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError('graph sync failed')

    monkeypatch.setattr('agent_system.persona_engine.GraphStore.sync_head', broken_sync_head)

    with pytest.raises(StorageWriteFailure):
        materialize_persona(
            'Dracula',
            {
                'entity_type': 'PERSON',
                'traits': ['aristocratic', 'predatory', 'hostile'],
                'examples': ['I am Dracula.'],
                'knowledge': 'BROKEN UPDATE SHOULD NOT STICK',
            },
            explicit=True,
        )

    restored = load_persona('dracula')
    assert restored is not None
    assert 'BROKEN UPDATE SHOULD NOT STICK' not in restored.knowledge
    assert 'fictional vampire count' in restored.knowledge
    assert list_persona_snapshots('dracula')


def test_runtime_status_snapshot_reports_degraded_local_llm(monkeypatch) -> None:
    from src.utils import local_llm_provider

    monkeypatch.setattr(local_llm_provider, 'Llama', None, raising=False)
    monkeypatch.setattr(
        local_llm_provider,
        'list_model_advisors',
        lambda: {
            'advisors': [
                {'role': 'analyst', 'available': False},
                {'role': 'general', 'available': False},
                {'role': 'translator', 'available': False},
            ]
        },
    )

    status = runtime_status_snapshot()
    codes = {item['code'] for item in status['degraded_modes']}

    assert status['mode'] == 'degraded'
    assert 'llama_cpp_missing' in codes or 'llm_roles_missing' in codes


def test_chat_degrades_to_behavioral_fallback_when_provider_is_unavailable(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr(
        'agent_system.chat_engine.runtime_status_snapshot',
        lambda: {
            'mode': 'degraded',
            'degraded_modes': [
                {
                    'code': 'llm_roles_missing',
                    'summary': 'Required chat role is unavailable.',
                    'detail': 'Missing role: analyst',
                }
            ],
        },
    )
    monkeypatch.setattr(
        'agent_system.chat_engine.generate_chat_reply',
        lambda prompt, language='en', persona_selected=False: (
            'I will answer in first person from the current persona graph and emotional state.'
            if persona_selected
            else 'I do not have enough reliable context yet. Clarify the entity or add one fact.'
        ),
    )

    from agent_system.chat_engine import generate_response

    payload = generate_response(message='Tell me about Dracula.', session_id='degraded-session', language='en')

    assert payload['assistant_reply']
    assert 'graph and emotional state' not in payload['assistant_reply']
    assert payload['runtime_status']['mode'] == 'degraded'
    assert payload['fallback_strategy']['strategy']
    assert payload['operator_messages']
    assert any('local chat provider is unavailable' in item.lower() for item in payload['operator_messages'])


def test_rethink_apply_rolls_back_graph_on_mutation_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    from agent_system.graph_store import GraphStore
    from agent_system.node_rethinker import rethink_node

    store = GraphStore()
    node = store.create_node(name='Sunlight', node_type='PHENOMENON', description='Old description.')
    node_id = str(node.get('id') or '')

    monkeypatch.setattr(
        'agent_system.node_rethinker.call_json_model_for_role',
        lambda prompt, role='general': {
            'node_improvement': {
                'description': 'Sunlight is radiant energy from the Sun.',
                'facts': ['Sunlight warms the planet.'],
            },
            'link_suggestions': [
                {
                    'name': 'Life',
                    'role': 'makes_possible',
                    'description': 'Life depends on sunlight.',
                }
            ],
        },
    )
    monkeypatch.setattr(
        store,
        'connect_nodes',
        lambda **kwargs: (_ for _ in ()).throw(OSError('simulated link failure')),
    )

    with pytest.raises(MutationRejectedFailure) as exc_info:
        rethink_node(node_id, store=store, preview_only=False, language='en')

    restored = store.get_node_by_id(node_id)
    assert restored is not None
    assert restored.get('description') == 'Old description.'
    assert store.get_node('Life') is None
    assert str(exc_info.value.details.get('rollback_snapshot_path') or '').strip()
