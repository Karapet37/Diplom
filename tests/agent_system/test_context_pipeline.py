from __future__ import annotations

from agent_system.context_builder import build_context
from agent_system.graph_store import GraphStore
from agent_system.history_store import create_session
from agent_system.persona_engine import materialize_persona


def test_context_pipeline_exposes_sources_scores_and_selected_items(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    materialize_persona(
        'Sheldon Cooper',
        {
            'entity_type': 'PERSON',
            'traits': ['logical', 'precise'],
            'examples': ['Leonard is my roommate.', 'Physics requires precision.'],
            'relations': [{'type': 'KNOWS', 'target': 'Leonard'}, {'type': 'STUDIES', 'target': 'Physics'}],
            'emotion_vector': {'confidence': 0.85, 'curiosity': 0.8, 'empathy': 0.2},
            'knowledge': 'I am a theoretical physicist who values rigor, precise language, and structured reasoning.',
        },
    )
    GraphStore().merge_extraction(
        {
            'entities': [
                {
                    'name': 'Leonard',
                    'type': 'PERSON',
                    'description': 'Experimental physicist and roommate.',
                    'aliases': [],
                    'facts': ['Leonard is a physicist.'],
                    'confidence': 0.92,
                    'context': {'source': 'session'},
                },
                {
                    'name': 'Physics',
                    'type': 'CONCEPT',
                    'description': 'Scientific study of matter, energy, motion, and physical law.',
                    'aliases': [],
                    'facts': ['Physics explains natural regularities.'],
                    'confidence': 0.94,
                    'context': {'source': 'file'},
                },
            ],
            'relations': [
                {'from': 'Sheldon Cooper', 'to': 'Leonard', 'type': 'KNOWS', 'weight': 0.9},
                {'from': 'Sheldon Cooper', 'to': 'Physics', 'type': 'STUDIES', 'weight': 0.92},
            ],
        },
        source='file',
    )
    session = create_session('context_debug', 'Context Debug')
    session_path = tmp_path / 'memory' / 'sessions' / f"{session['session_id']}.txt"
    session_path.write_text(
        '\n\n'.join(
            [
                '[2026-03-17T00:00:00Z]\nuser: Leonard and Sheldon are discussing precision in physics.',
                '[2026-03-17T00:00:10Z]\nassistant: Precision matters when facts and identities are linked carefully.',
                '[2026-03-17T00:00:20Z]\nuser: Explain Leonard and physics again, but stay precise.',
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    built = build_context(
        question='How does Leonard relate to your physics-oriented personality, and why does precision matter here?',
        session_id='context_debug',
        selected_persona='sheldon_cooper',
        situation={'type': 'neutral_query', 'target': 'persona', 'severity': 0.4},
    )

    debug = built['context_debug']
    assert debug['stages']['collect_candidates'] >= 6
    assert debug['weights']['relevance'] > 0
    assert debug['estimated_tokens'] <= 4000
    assert debug['source_counts']['persona_memory'] >= 1
    assert debug['source_counts']['persona_triad'] >= 1
    assert debug['source_counts']['session_short_term_history'] >= 1
    assert debug['source_counts']['local_graph_neighborhood'] >= 1
    assert debug['source_counts']['file_ingested_knowledge'] >= 1
    assert any(item['score']['total'] > 0 for item in debug['selected_items'])
    assert any(item['source'] == 'session_short_term_history' for item in debug['selected_items'])
    assert any(item['section'] == 'persona_block' for item in debug['selected_items'])
    visible_items = list(debug['selected_items']) + list(debug['top_unselected_items'])
    assert any(item['source'] == 'file_ingested_knowledge' for item in visible_items)


def test_context_pipeline_orders_equal_ranked_graph_candidates_deterministically(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    GraphStore().merge_extraction(
        {
            'entities': [
                {
                    'name': 'Alpha Archive',
                    'type': 'CONCEPT',
                    'description': 'Alpha archive is a concept node about alpha records.',
                    'aliases': ['Alpha'],
                    'facts': ['Alpha archive stores alpha records.'],
                    'confidence': 0.9,
                    'context': {'source': 'file'},
                },
                {
                    'name': 'Alpha Atlas',
                    'type': 'CONCEPT',
                    'description': 'Alpha atlas is a concept node about alpha mappings.',
                    'aliases': ['Alpha'],
                    'facts': ['Alpha atlas stores alpha mappings.'],
                    'confidence': 0.9,
                    'context': {'source': 'file'},
                },
            ],
            'relations': [],
        },
        source='file',
    )
    create_session('context_order', 'Context Order')

    first = build_context(
        question='Explain alpha.',
        session_id='context_order',
        situation={'type': 'neutral_query', 'target': 'external', 'severity': 0.3},
    )
    second = build_context(
        question='Explain alpha.',
        session_id='context_order',
        situation={'type': 'neutral_query', 'target': 'external', 'severity': 0.3},
    )

    first_graph = [
        item['title']
        for item in first['context_debug']['selected_items']
        if item['section'] == 'graph_context'
    ]
    second_graph = [
        item['title']
        for item in second['context_debug']['selected_items']
        if item['section'] == 'graph_context'
    ]

    assert first_graph == second_graph
    assert first_graph == sorted(first_graph)
    assert first['estimated_tokens'] <= 4000
    assert second['estimated_tokens'] <= 4000
