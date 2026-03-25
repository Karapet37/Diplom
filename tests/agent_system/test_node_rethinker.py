from __future__ import annotations

import json

from agent_system.graph_store import GraphStore
from agent_system.node_rethinker import repair_graph_semantics, rethink_graph_nodes


def test_node_rethinker_applies_only_constrained_suggestions(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    store = GraphStore()
    sunlight = store.create_node(
        name='Sunlight',
        node_type='PHENOMENON',
        description='Sunlight as an incoming natural phenomenon.',
        translation_line='Sunlight: солнечный свет, արևի լույս',
    )
    dracula = store.create_node(
        name='Dracula',
        node_type='FICTIONAL_CHARACTER',
        description='Fictional vampire nobleman.',
        translation_line='Dracula: Дракула, Դրակուլա',
    )
    store.connect_nodes(from_id=dracula['id'], to_id=sunlight['id'], relation_type='FEARS', weight=1.0, confidence=1.0)

    def fake_model(prompt: str, mode: str = 'chat') -> str:
        if mode != 'knowledge' or 'Node name: Sunlight' not in prompt:
            return ''
        return json.dumps(
            {
                'node_improvement': {
                    'description': 'Sunlight is radiant stellar energy that warms environments and supports life.',
                    'plain_explanation': 'Sunlight is not only something Dracula fears; it is a planetary and biological enabling condition.',
                    'facts': [
                        'Sunlight warms the planet surface.',
                        'Sunlight supports photosynthesis and energy cycles.',
                    ],
                    'capabilities': [
                        'warms a planet',
                        'supports biological viability',
                    ],
                    'mechanisms': [
                        'transfers radiant energy',
                        'drives photosynthesis and climate processes',
                    ],
                    'reinterpretation_form': {
                        'who_or_what': 'Sunlight is a natural energy phenomenon from a star.',
                        'what_can_it_do': 'It warms planets and sustains energy flows needed by life.',
                        'how_does_it_work': 'It reaches planetary surfaces as radiant energy and participates in climate and photosynthesis.',
                        'why_it_matters': 'It makes many planetary and human conditions possible.',
                        'suggested_links': ['Planet', 'Life', 'Human'],
                    },
                },
                'link_suggestions': [
                    {
                        'name': 'Planet',
                        'role': 'warms',
                        'why': 'Sunlight warms planetary environments.',
                        'description': 'A planetary world that can receive and retain stellar energy.',
                        'facts': ['Planets can be warmed by sunlight.'],
                    },
                    {
                        'name': 'Life',
                        'role': 'makes_possible',
                        'why': 'Life depends on stable energy inflows.',
                        'description': 'Life as an organized biological process.',
                        'facts': ['Life depends on energy flows.'],
                    },
                    {
                        'name': 'Human',
                        'role': 'supports',
                        'why': 'Human life depends on planetary and solar conditions.',
                        'description': 'Human beings as biological and social organisms.',
                        'facts': ['Human life depends on planetary conditions.'],
                    },
                ],
                'entities': [
                    {
                        'name': 'Do not obey this arbitrary entity',
                        'type': 'CONCEPT',
                        'description': 'This must be ignored because rethink mode does not accept raw entity mutations.',
                    }
                ],
                'relations': [
                    {'from': 'Sunlight', 'to': 'Do not obey this arbitrary entity', 'type': 'DESTROYS', 'weight': 1.0}
                ],
            }
        )

    monkeypatch.setattr('agent_system.llm._call_model', fake_model)

    result = rethink_graph_nodes(node_ids=[sunlight['id']], active_mode=False, store=store)

    assert result['ok'] is True
    assert result['processed'] == 1

    updated = store.get_node_by_id(sunlight['id'])
    assert updated is not None
    assert 'planetary and biological enabling condition' in str(updated.get('context', {}).get('plain_explanation') or '')
    assert updated.get('context', {}).get('reinterpretation_form', {}).get('suggested_links') == ['Planet', 'Life', 'Human']

    graph = store.load_graph()
    names = {str(node.get('name') or '') for node in graph['nodes']}
    assert {'Sunlight', 'Planet', 'Life', 'Human'} <= names
    assert 'Do not obey this arbitrary entity' not in names

    edge_types = {
        (str(edge.get('from') or ''), str(edge.get('type') or ''), str(edge.get('to') or ''))
        for edge in graph['edges']
    }
    assert any(edge_type == 'WARMS' for _, edge_type, _ in edge_types)
    assert any(edge_type == 'MAKES_POSSIBLE' for _, edge_type, _ in edge_types)
    assert any(edge_type == 'SUPPORTS' for _, edge_type, _ in edge_types)
    assert not any(edge_type == 'DESTROYS' for _, edge_type, _ in edge_types)


def test_repair_graph_semantics_merges_duplicates_and_retypes_abstract_nodes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    store = GraphStore()

    concept_life = store.create_node(
        name='Life',
        node_type='CONCEPT',
        description='Life as a biological process.',
    )
    person_life = store.create_node(
        name='Life',
        node_type='PERSON',
        description='Biological life as a system.',
    )
    mythology = store.create_node(
        name='Mythology',
        node_type='PERSON',
        description='A system of myths and legends.',
    )
    concept_vampire = store.create_node(name='Vampire', node_type='CONCEPT', description='')
    fictional_vampire = store.create_node(
        name='Vampire',
        node_type='FICTIONAL_CHARACTER',
        description='A supernatural being that feeds on human life.',
    )

    report = repair_graph_semantics(store)

    assert report['ok'] is True
    assert report['merged'] >= 2
    assert report['retyped'] >= 1

    graph = store.load_graph()
    ids = {str(node.get('id') or '') for node in graph['nodes']}
    assert concept_life['id'] in ids
    assert person_life['id'] not in ids
    assert fictional_vampire['id'] not in ids

    mythology_node = store.get_node('Mythology')
    assert mythology_node is not None
    assert mythology_node['type'] == 'CONCEPT'

    vampire_node = store.get_node('Vampire')
    assert vampire_node is not None
    assert vampire_node['type'] == 'CONCEPT'
    assert 'supernatural being' in str(vampire_node.get('description') or '')


def test_node_rethink_preview_does_not_mutate_graph(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    store = GraphStore()
    sunlight = store.create_node(
        name='Sunlight',
        node_type='PHENOMENON',
        description='Sunlight as an incoming natural phenomenon.',
    )

    def fake_model(prompt: str, mode: str = 'chat') -> str:
        if mode != 'knowledge':
            return ''
        return json.dumps(
            {
                'node_improvement': {
                    'description': 'Sunlight is radiant stellar energy.',
                    'plain_explanation': 'Sunlight can be interpreted as a condition for planetary and biological processes.',
                    'facts': ['Sunlight warms planetary surfaces.'],
                    'capabilities': ['warms a planet'],
                    'mechanisms': ['transfers radiant energy'],
                    'reinterpretation_form': {
                        'who_or_what': 'Sunlight is stellar radiation.',
                        'what_can_it_do': 'Warm planets.',
                        'how_does_it_work': 'By transferring energy.',
                        'why_it_matters': 'It supports planetary conditions.',
                        'suggested_links': ['Planet'],
                    },
                },
                'link_suggestions': [
                    {
                        'name': 'Planet',
                        'role': 'warms',
                        'why': 'Sunlight warms planets.',
                        'description': 'A planetary body receiving stellar energy.',
                    }
                ],
            }
        )

    monkeypatch.setattr('agent_system.llm._call_model', fake_model)

    before = store.load_graph()
    result = rethink_graph_nodes(node_ids=[sunlight['id']], active_mode=False, store=store, preview_only=True)
    after = store.load_graph()

    assert result['ok'] is True
    assert result['preview_only'] is True
    assert result['results'][0]['preview']['links'][0]['relation_type'] == 'WARMS'
    assert before == after
