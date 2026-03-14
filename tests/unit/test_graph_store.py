from __future__ import annotations

import json
import os

from roaches_viz.roaches_viz.graph_store import GraphStore, graph_edges_path, graph_nodes_path


def test_graph_store_merges_proposals_and_blocks_empty_overwrite(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    store = GraphStore()

    merged = store.merge_proposals([
        {
            'entity': 'Dracula',
            'type': 'PERSON',
            'traits': ['вампир', 'аристократический'],
            'relations': [{'type': 'FEEDS_ON', 'target': 'люди'}],
        }
    ])
    skipped = store.save_graph([], [], reason='empty')

    assert merged['ok'] is True
    assert skipped['ok'] is False
    assert skipped['reason'] == 'skipped_empty_overwrite'
    assert os.path.exists(graph_nodes_path())
    assert 'dracula' in graph_nodes_path().read_text(encoding='utf-8').lower()
    assert os.path.exists(graph_edges_path())


def test_graph_store_duplicate_detection_keeps_distinct_people_with_qualifiers(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    store = GraphStore()

    store.merge_proposals([
        {'entity': 'George', 'type': 'PERSON', 'profession': 'mechanic', 'traits': ['calm'], 'relations': []},
        {'entity': 'George', 'type': 'PERSON', 'profession': 'professor', 'traits': ['curious'], 'relations': []},
        {'entity': 'George', 'type': 'PERSON', 'profession': 'mechanic', 'traits': ['steady'], 'relations': []},
    ])

    nodes = json.loads(graph_nodes_path().read_text(encoding='utf-8'))
    person_nodes = [node for node in nodes if node['type'] == 'PERSON' and node['name'] == 'George']
    assert len(person_nodes) == 2


def test_graph_store_answerable_node_view_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    store = GraphStore()
    store.merge_proposals([
        {'entity': 'Dracula', 'type': 'PERSON', 'traits': ['вампир'], 'relations': [{'type': 'FEEDS_ON', 'target': 'люди'}]}
    ])

    view = store.answerable_node_view('person:dracula')

    assert view is not None
    assert set(view.keys()) == {'who_or_what', 'what_is_it_like', 'how_it_acts'}
