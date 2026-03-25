from __future__ import annotations

from agent_system.graph_store import GraphStore


def test_low_confidence_extraction_is_quarantined_and_excluded_from_search(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    store = GraphStore()

    result = store.merge_extraction(
        {
            'entities': [
                {
                    'name': 'Shadow rumor',
                    'type': 'CONCEPT',
                    'description': 'Weakly supported concept.',
                    'confidence': 0.32,
                    'importance': 0.21,
                    'context': {'source': 'test'},
                }
            ],
            'relations': [],
        },
        source='test',
    )

    assert result['quarantined_nodes'] == 1
    graph = store.load_graph()
    suspect = next(node for node in graph['nodes'] if str(node.get('name') or '') == 'Shadow rumor')
    assert suspect['lifecycle_state'] == 'suspect'
    assert suspect['context']['review_status'] == 'quarantine'
    assert store.search_nodes('Shadow rumor', limit=4) == []
    assert graph['diagnostics']['suspect_node_count'] == 1


def test_review_similarity_duplicates_become_suspect_instead_of_auto_merge(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    store = GraphStore()

    store.upsert_entity(
        name='Cultural life',
        entity_type='CONCEPT',
        aliases=['layered system'],
        description='Structured concept with stable context.',
        context={'domain': 'human', 'source': 'test'},
        translation_line='Cultural life: культурная жизнь, մշակութային կյանք',
    )
    store.upsert_entity(
        name='Social life',
        entity_type='CONCEPT',
        aliases=['layered system'],
        description='Structured concept with stable context.',
        context={'domain': 'human', 'source': 'test'},
        translation_line='Social life: социальная жизнь, սոցիալական կյանք',
    )

    graph = store.load_graph()
    names = {str(node.get('name') or ''): node for node in graph['nodes']}
    assert 'Cultural life' in names
    assert 'Social life' in names
    suspect_names = {str(node.get('name') or '') for node in graph['nodes'] if str(node.get('lifecycle_state') or '') == 'suspect'}
    assert suspect_names
    assert graph['diagnostics']['duplicate_review_candidates'] >= 1


def test_low_value_nodes_are_archived_and_counted_in_diagnostics(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    store = GraphStore()

    store.upsert_entity(
        name='Dust',
        entity_type='OBJECT',
        description='Low value dust.',
        confidence=0.2,
        importance=0.01,
        source='test',
    )

    graph = store.load_graph()
    assert not any(str(node.get('name') or '') == 'Dust' for node in graph['nodes'])
    assert graph['diagnostics']['archived_node_count'] >= 1


def test_large_connected_graph_receives_cluster_labels(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setenv('COGNITIVE_GRAPH_CLUSTERING_MIN_NODES', '8')
    monkeypatch.setenv('COGNITIVE_GRAPH_CLUSTERING_COMPONENT_MIN', '3')
    store = GraphStore()

    nodes = [
        store.create_node(name='Human', node_type='CONCEPT', description='Human concept.'),
        store.create_node(name='Biology', node_type='CONCEPT', description='Biology concept.'),
        store.create_node(name='Psychology', node_type='CONCEPT', description='Psychology concept.'),
        store.create_node(name='Sociology', node_type='CONCEPT', description='Sociology concept.'),
        store.create_node(name='Culture', node_type='CONCEPT', description='Culture concept.'),
        store.create_node(name='Language', node_type='CONCEPT', description='Language concept.'),
        store.create_node(name='Ethics', node_type='CONCEPT', description='Ethics concept.'),
        store.create_node(name='Identity', node_type='CONCEPT', description='Identity concept.'),
    ]
    root = nodes[0]
    for node in nodes[1:]:
        store.connect_nodes(from_id=root['id'], to_id=node['id'], relation_type='HAS_DIMENSION')

    graph = store.load_graph()
    cluster_labels = {str(node.get('cluster_label') or '') for node in graph['nodes']}
    assert any(label for label in cluster_labels)
    assert graph['diagnostics']['cluster_count'] >= 1


def test_manual_review_can_promote_and_archive_nodes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    store = GraphStore()

    node = store.create_node(name='Speculative node', node_type='CONCEPT', description='Speculative concept.')
    suspect = store.review_node_state(node['id'], lifecycle_state='suspect')
    assert suspect['ok'] is True
    assert store.get_node_by_id(node['id'])['lifecycle_state'] == 'suspect'

    archived = store.review_node_state(node['id'], lifecycle_state='archived')
    assert archived['ok'] is True
    assert store.get_node_by_id(node['id']) is None
    assert store.load_graph()['diagnostics']['archived_node_count'] >= 1
