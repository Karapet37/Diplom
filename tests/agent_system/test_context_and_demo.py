from __future__ import annotations

import json

from agent_system.context_builder import build_context
from agent_system.demo import run_demo
from agent_system.graph_store import GraphStore, graph_nodes_path
from agent_system.history_store import create_session
from agent_system.persona_engine import materialize_persona


def test_context_builder_limits_budget_and_uses_persona_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    long_knowledge = ' '.join(['precision matters deeply to Sheldon.'] * 1200)
    materialize_persona(
        'Sheldon Cooper',
        {
            'entity_type': 'PERSON',
            'traits': ['logical', 'precise'],
            'examples': ['Leonard is my roommate.'] + [f'Example {index}: Sheldon values precision.' for index in range(40)],
            'relations': [{'type': 'KNOWS', 'target': 'Leonard'}],
            'emotion_vector': {'confidence': 0.85, 'curiosity': 0.8, 'empathy': 0.2},
            'knowledge': long_knowledge,
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
                    'confidence': 0.9,
                    'context': {'source': 'session'},
                }
            ],
            'relations': [{'from': 'Sheldon Cooper', 'to': 'Leonard', 'type': 'KNOWS', 'weight': 0.9}],
        },
        source='session',
    )
    session = create_session('session_test', 'Session')
    session_path = tmp_path / 'memory' / 'sessions' / f"{session['session_id']}.txt"
    session_path.write_text(
        '\n\n'.join(
            [
                '[2026-03-17T00:00:00Z]\nuser: Leonard and Sheldon have a long debate about precision.'
                for _ in range(20)
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    built = build_context(
        question='Who is Leonard to you, and how does precision shape your view of him? ' * 80,
        session_id='session_test',
        selected_persona='sheldon_cooper',
        situation={'type': 'neutral_query', 'target': 'persona', 'severity': 0.45},
    )

    assert 'Emotion vector:' in built['persona_block']
    assert 'Response style:' in built['persona_block']
    assert 'Current situation: type=neutral_query; target=persona; severity=0.45.' in built['persona_block']
    assert built['situation'] == 'type=neutral_query; target=persona; severity=0.45'
    assert 'logical' in built['persona_block']
    assert 'Leonard' in built['graph_context']
    assert built['estimated_tokens'] <= 4000
    assert built['recent_dialogue']


def test_demo_runs_ingestion_then_persona_response(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    def fake_model(prompt: str, mode: str = 'chat') -> str:
        lowered = prompt.lower()
        if mode == 'knowledge' and 'text:' in lowered:
            return json.dumps(
                {
                    'entities': [
                        {
                            'name': 'Dracula',
                            'aliases': ['Count Dracula'],
                            'description': 'Fictional vampire nobleman.',
                            'facts': ['Dracula feeds on humans.', 'Dracula fears sunlight.'],
                            'context': {'source': 'file'},
                        },
                        {'name': 'humans', 'aliases': [], 'description': 'People.', 'facts': [], 'context': {'source': 'file'}},
                        {'name': 'sunlight', 'aliases': [], 'description': 'Daylight.', 'facts': [], 'context': {'source': 'file'}},
                    ],
                    'relations': [
                        {'from': 'Dracula', 'to': 'humans', 'type': 'FEEDS_ON', 'weight': 0.9},
                        {'from': 'Dracula', 'to': 'sunlight', 'type': 'FEARS', 'weight': 0.8},
                    ],
                }
            )
        return 'I am an immortal vampire nobleman.'

    monkeypatch.setattr('agent_system.llm._call_model', fake_model)
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)

    document = tmp_path / 'dracula.txt'
    document.write_text('Dracula is an immortal vampire nobleman who feeds on humans and fears sunlight.', encoding='utf-8')
    result = run_demo(document=str(document), question='Who are you?', persona='Dracula', session_id='demo_test', language='en')

    assert 'vampire' in result['assistant_reply'].lower()
    assert graph_nodes_path().exists()
    assert 'dracula' in graph_nodes_path().read_text(encoding='utf-8').lower()
