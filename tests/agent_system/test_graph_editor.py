from __future__ import annotations

from agent_system.graph_store import GraphStore


def test_graph_store_manual_edit_operations(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    store = GraphStore()

    human = store.create_node(
        name='Human',
        node_type='CONCEPT',
        description='Human as a structured concept.',
        translation_line='Human: человек, մարդ',
    )
    biology = store.create_node(
        name='Biology',
        node_type='CONCEPT',
        description='Biology branch.',
        translation_line='Biology: биология, կենսաբանություն',
    )
    human_duplicate = store.create_node(
        name='Mankind',
        node_type='CONCEPT',
        description='Alternative label for the human concept.',
        translation_line='Mankind: человечество, մարդկություն',
    )

    connect = store.connect_nodes(from_id=human['id'], to_id=biology['id'], relation_type='HAS_DIMENSION')
    assert connect['ok'] is True

    merged = store.merge_nodes_manual(primary_id=human['id'], secondary_id=human_duplicate['id'])
    assert merged['ok'] is True

    graph = store.load_graph()
    names = {str(node.get('name') or '') for node in graph['nodes']}
    assert 'Human' in names
    assert 'Human Being' not in names
    assert any(str(edge.get('type') or '') == 'HAS_DIMENSION' for edge in graph['edges'])

    deleted_edge = store.delete_edge(edge_id=connect['edge']['id'])
    assert deleted_edge['ok'] is True
    assert not any(str(edge.get('id') or '') == connect['edge']['id'] for edge in store.load_graph()['edges'])

    deleted_node = store.delete_node(biology['id'])
    assert deleted_node['ok'] is True
    assert store.get_node_by_id(biology['id']) is None
