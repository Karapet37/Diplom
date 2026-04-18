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


def test_graph_native_search_combines_local_and_global_graph_evidence(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    store = GraphStore()
    store.merge_extraction(
        {
            'entities': [
                {
                    'name': 'Captain',
                    'type': 'PERSON',
                    'description': 'Ship captain responsible for navigation decisions.',
                    'aliases': [],
                    'facts': ['Captain gives orders during storms.'],
                    'confidence': 0.92,
                    'context': {'source': 'session'},
                },
            ],
            'relations': [],
        },
        source='session',
        session_id='storm_session',
    )
    store.merge_extraction(
        {
            'entities': [
                {
                    'name': 'Crew Discipline',
                    'type': 'CONCEPT',
                    'description': 'Discipline keeps the crew aligned under pressure.',
                    'aliases': [],
                    'facts': ['Orders must be clear during storms.'],
                    'confidence': 0.9,
                    'context': {'source': 'file'},
                },
                {
                    'name': 'Storm Systems',
                    'type': 'PHENOMENON',
                    'description': 'Storm fronts change navigation risk rapidly.',
                    'aliases': [],
                    'facts': ['Storm systems force course changes.'],
                    'confidence': 0.88,
                    'context': {'source': 'file'},
                },
                {
                    'name': 'Harbor Signals',
                    'type': 'CONCEPT',
                    'description': 'Harbor signals help ships coordinate entry and emergency docking.',
                    'aliases': [],
                    'facts': ['Signals matter even outside immediate storm commands.'],
                    'confidence': 0.86,
                    'context': {'source': 'file'},
                },
            ],
            'relations': [
                {'from': 'Captain', 'to': 'Crew Discipline', 'type': 'USES', 'weight': 0.91},
                {'from': 'Captain', 'to': 'Storm Systems', 'type': 'MONITORS', 'weight': 0.9},
                {'from': 'Crew Discipline', 'to': 'Harbor Signals', 'type': 'COORDINATES_WITH', 'weight': 0.74},
            ],
        },
        source='file',
    )

    result = store.graph_native_search('captain storm orders', anchor_names=['Captain'], session_id='storm_session', limit=6)
    retrieval_sources = {str(item.get('retrieval_source') or '') for item in result['entries']}
    names = {str(item['node'].get('name') or '') for item in result['entries']}

    assert 'Captain' in names
    assert 'Crew Discipline' in names
    assert 'anchor_match' in retrieval_sources or 'local_1hop' in retrieval_sources
    assert 'semantic_dense' in retrieval_sources
    assert result['diagnostics']['one_hop_count'] >= 1
    assert result['diagnostics']['session_scoped'] is True
    assert result['diagnostics']['session_dense_count'] >= 1
    assert result['diagnostics']['external_dense_count'] >= 1


def test_context_pipeline_exposes_graph_retrieval_sources_in_debug_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    GraphStore().merge_extraction(
        {
            'entities': [
                {
                    'name': 'Captain',
                    'type': 'PERSON',
                    'description': 'Captain commands the ship.',
                    'aliases': [],
                    'facts': ['The captain issues orders.'],
                    'confidence': 0.92,
                    'context': {'source': 'session'},
                },
                {
                    'name': 'Crew Discipline',
                    'type': 'CONCEPT',
                    'description': 'Crew discipline keeps orders clear and respected.',
                    'aliases': [],
                    'facts': ['Crew discipline matters during storms.'],
                    'confidence': 0.9,
                    'context': {'source': 'file'},
                },
            ],
            'relations': [
                {'from': 'Captain', 'to': 'Crew Discipline', 'type': 'USES', 'weight': 0.9},
            ],
        },
        source='file',
    )
    create_session('graph_native_context', 'Graph Native Context')

    built = build_context(
        question='How does the captain rely on crew discipline during a storm?',
        session_id='graph_native_context',
        situation={'type': 'neutral_query', 'target': 'external', 'severity': 0.4},
    )

    graph_items = [item for item in built['context_debug']['selected_items'] if item['section'] == 'graph_context']
    retrieval_sources = {str(item.get('metadata', {}).get('retrieval_source') or '') for item in graph_items}

    assert graph_items
    assert any(source in {'anchor_match', 'local_1hop', 'semantic_dense', 'structural_salience'} for source in retrieval_sources)


def test_context_pipeline_prefers_session_graph_context_over_weak_global_background(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    store = GraphStore()
    create_session('session_scoped_context', 'Session Scoped Context')
    store.merge_extraction(
        {
            'entities': [
                {
                    'name': 'Captain',
                    'type': 'PERSON',
                    'description': 'Captain of this session route who gives direct storm orders.',
                    'aliases': [],
                    'facts': ['The captain gives orders during storms in this session scenario.'],
                    'confidence': 0.95,
                    'context': {'source': 'session'},
                },
                {
                    'name': 'Storm Orders',
                    'type': 'CONCEPT',
                    'description': 'Orders issued during a storm to keep the ship under control.',
                    'aliases': [],
                    'facts': ['Storm orders must be precise and immediate.'],
                    'confidence': 0.92,
                    'context': {'source': 'session'},
                },
                {
                    'name': 'Office Orders',
                    'type': 'CONCEPT',
                    'description': 'Routine office orders unrelated to ships or storms.',
                    'aliases': [],
                    'facts': ['Office orders are about paperwork and reporting.'],
                    'confidence': 0.84,
                    'context': {'source': 'file'},
                },
            ],
            'relations': [
                {'from': 'Captain', 'to': 'Storm Orders', 'type': 'USES', 'weight': 0.93},
            ],
        },
        source='file',
        session_id='session_scoped_context',
    )

    built = build_context(
        question='How does the captain give orders in the storm?',
        session_id='session_scoped_context',
        situation={'type': 'neutral_query', 'target': 'external', 'severity': 0.4},
        store=store,
    )

    debug = built['context_debug']
    graph_items = [item for item in debug['selected_items'] if item['section'] == 'graph_context']
    titles = [str(item.get('title') or '') for item in graph_items]
    sources = [str(item.get('source') or '') for item in graph_items]

    assert graph_items
    assert debug['source_counts']['session_graph_context'] >= 1
    assert 'Captain' in titles
    assert 'Storm Orders' in titles
    assert 'session_graph_context' in sources
