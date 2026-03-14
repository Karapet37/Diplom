from __future__ import annotations

import json

from roaches_viz.roaches_viz.context_builder import build_chat_context, list_personalities
from roaches_viz.roaches_viz.graph_store import GraphStore, personality_graph_path, personality_index_path, personality_profile_path


def test_context_builder_injects_persona_and_bounds_graph_context(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    personality_index_path().write_text(json.dumps({'personalities': ['sheldon_cooper']}, ensure_ascii=False), encoding='utf-8')
    personality_profile_path('sheldon_cooper').write_text(json.dumps({
        'name': 'Sheldon Cooper',
        'traits': ['hyperlogical', 'literal', 'arrogant'],
        'patterns': ['correct_people', 'quote_science'],
        'examples': ['Леонард — мой сосед по квартире.'],
    }, ensure_ascii=False), encoding='utf-8')
    personality_graph_path('sheldon_cooper').write_text(json.dumps({'nodes': [], 'edges': [{'from': 'person:sheldon', 'type': 'KNOWS', 'to': 'person:leonard'}]}, ensure_ascii=False), encoding='utf-8')

    store = GraphStore()
    store.merge_proposals([
        {'entity': 'Leonard', 'type': 'PERSON', 'traits': ['experimental'], 'relations': [{'type': 'KNOWS', 'target': 'Sheldon Cooper'}]}
    ])

    context = build_chat_context(
        message='кто для тебя Леонард?',
        recent_dialogue='user: кто для тебя Леонард?',
        selected_personality='sheldon_cooper',
        current_entity='leonard',
        store=store,
    )

    assert 'You are Sheldon Cooper.' in context['personality_prompt']
    assert 'Traits: hyperlogical, literal, arrogant.' in context['personality_prompt']
    assert len(context['graph_context']) <= 1800 * 4
    assert list_personalities()[0]['name'] == 'sheldon_cooper'
