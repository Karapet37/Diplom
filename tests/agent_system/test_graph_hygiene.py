from __future__ import annotations

from agent_system.graph_store import GraphStore


def test_graph_hygiene_merges_aliases_and_collects_garbage_without_small_graph_summaries(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    store = GraphStore()

    store.upsert_entity(
        name='Dracula',
        entity_type='FICTIONAL_CHARACTER',
        aliases=['Count Dracula'],
        description='Fictional vampire nobleman.',
        confidence=0.95,
        source='test',
    )
    store.upsert_entity(
        name='Count Dracula',
        entity_type='FICTIONAL_CHARACTER',
        aliases=['Dracula'],
        description='Fictional vampire nobleman.',
        confidence=0.9,
        source='test',
    )
    store.upsert_entity(name='Dust', entity_type='OBJECT', aliases=[], description='Low value dust.', confidence=0.3, source='test', importance=0.01)
    store.merge_extraction(
        {
            'entities': [
                {'name': 'sunlight', 'type': 'PHENOMENON', 'description': 'Daylight.', 'aliases': [], 'facts': [], 'importance': 0.4, 'confidence': 0.7, 'context': {'source': 'test'}},
                {'name': 'garlic', 'type': 'OBJECT', 'description': 'Garlic bulbs.', 'aliases': [], 'facts': [], 'importance': 0.4, 'confidence': 0.7, 'context': {'source': 'test'}},
            ],
            'relations': [
                {'from': 'Dracula', 'to': 'sunlight', 'type': 'FEARS', 'weight': 0.8},
                {'from': 'Dracula', 'to': 'garlic', 'type': 'FEARS', 'weight': 0.75},
            ],
        },
        source='test',
    )

    graph = store.load_graph()
    dracula_nodes = [node for node in graph['nodes'] if str(node.get('type') or '') == 'FICTIONAL_CHARACTER']
    assert len(dracula_nodes) == 1
    assert 'Count Dracula' in dracula_nodes[0]['aliases']
    assert not any(str(node.get('name') or '') == 'Dust' for node in graph['nodes'])
    assert not any(str(node.get('id') or '').startswith('summary:') for node in graph['nodes'])


def test_graph_hygiene_merges_multilingual_human_duplicates(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    store = GraphStore()

    store.upsert_entity(
        name='Human',
        entity_type='CONCEPT',
        aliases=['человек', 'human being'],
        description='Human as a layered living and social system.',
        confidence=0.95,
        source='test',
    )
    store.upsert_entity(
        name='люди',
        entity_type='CONCEPT',
        aliases=['people'],
        description='Plural label for human beings.',
        confidence=0.8,
        source='test',
    )

    graph = store.load_graph()
    human_nodes = [node for node in graph['nodes'] if str(node.get('type') or '') == 'CONCEPT']
    assert len(human_nodes) == 1
    assert 'люди' in human_nodes[0]['aliases'] or human_nodes[0]['name'] == 'люди'


def test_graph_hygiene_optimizes_quality_objective(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    store = GraphStore()

    quality_before = store.graph_quality()
    store.upsert_entity(
        name='Dracula',
        entity_type='FICTIONAL_CHARACTER',
        aliases=['Count Dracula'],
        description='Fictional vampire nobleman.',
        confidence=0.95,
        source='test',
    )
    result = store.upsert_entity(
        name='Count Dracula',
        entity_type='FICTIONAL_CHARACTER',
        aliases=['Dracula'],
        description='Fictional vampire nobleman.',
        confidence=0.9,
        source='test',
    )
    store.merge_extraction(
        {
            'entities': [
                {'name': 'sunlight', 'type': 'PHENOMENON', 'description': 'Daylight.', 'aliases': [], 'facts': [], 'importance': 0.4, 'confidence': 0.7, 'context': {'source': 'test'}},
                {'name': 'garlic', 'type': 'OBJECT', 'description': 'Garlic bulbs.', 'aliases': [], 'facts': [], 'importance': 0.4, 'confidence': 0.7, 'context': {'source': 'test'}},
            ],
            'relations': [
                {'from': 'Dracula', 'to': 'sunlight', 'type': 'FEARS', 'weight': 0.8},
                {'from': 'Dracula', 'to': 'garlic', 'type': 'FEARS', 'weight': 0.75},
            ],
        },
        source='test',
    )

    quality_after = store.graph_quality()
    assert quality_after.score >= quality_before.score
    assert quality_after.redundancy <= 1.0
    assert quality_after.connectivity >= 0.0

    hygiene = store.apply_hygiene()
    assert 'quality_before' in hygiene and 'quality_after' in hygiene
    assert hygiene['quality_after']['score'] >= hygiene['quality_before']['score']
