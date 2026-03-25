from __future__ import annotations

import json

from agent_system.chat_engine import generate_response
from agent_system.history_store import parse_session
from agent_system.persona_engine import load_persona, materialize_persona


def test_chat_engine_spawns_head_updates_emotions_and_learns_reaction(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat': 'I answer in the first person as Dracula.')

    result = generate_response(
        message='Speak as Dracula the vampire count, and why should I trust you?',
        session_id='session_test',
        selected_persona='Dracula',
        language='en',
    )

    assert result['assistant_reply'] == 'I answer in the first person as Dracula.'
    assert result['persona_name'] == 'dracula'
    assert 'graph_write_sources' in result['side_effects']
    assert result['side_effects']['history_write_path'].endswith('session_test.txt')
    assert 'emotion_vector' in result['side_effects']['persona_updates']
    assert 'situation_reaction' in result['side_effects']['persona_updates']
    assert result['side_effects']['rebuild']['session_id'] == 'session_test'
    session_path = tmp_path / 'memory' / 'sessions' / 'session_test.txt'
    assert session_path.exists()
    parsed = parse_session('session_test')
    assert parsed is not None
    assert len(parsed['messages']) == 2

    bundle = load_persona('dracula')
    assert bundle is not None
    assert bundle.emotion_vector['curiosity'] > 0.55
    assert result['analysis']['user_state']['intent'] == 'question'
    assert result['analysis']['situation']['type'] == 'neutral_query'
    payload = json.loads((tmp_path / 'memory' / 'heads' / 'dracula' / 'examples.json').read_text(encoding='utf-8'))
    assert 'I answer in the first person as Dracula.' not in payload['examples']
    assert payload['situation_reactions']
    assert payload['situation_reactions'][0]['situation'].startswith('type=neutral_query;')


def test_chat_engine_routes_lowercase_entity_mentions(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat': 'I am Dracula, a vampire nobleman.')

    result = generate_response(
        message='tell me about dracula the vampire nobleman',
        session_id='session_lowercase',
        language='en',
    )

    assert result['persona_name'] == 'dracula'
    assert 'dracula' in [item.lower() for item in result['analysis']['entities']]
    assert load_persona('dracula') is not None


def test_chat_engine_surfaces_background_repair_failures(tmp_path, monkeypatch) -> None:
    class ImmediateExecutor:
        def submit(self, fn):
            fn()
            return None

    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._BACKGROUND_EXECUTOR', ImmediateExecutor())
    monkeypatch.setattr('agent_system.chat_engine._should_schedule_background_extraction', lambda session_id, personality_name='': (True, 'forced_for_test'))
    monkeypatch.setattr('agent_system.chat_engine.rebuild_artifacts', lambda session_id, personality_name='': (_ for _ in ()).throw(RuntimeError('repair exploded')))
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat': 'I answer in first person.')

    result = generate_response(
        message='Speak as Dracula the vampire count.',
        session_id='session_failure',
        selected_persona='Dracula',
        language='en',
    )

    assert result['repair_status']['status'] == 'error'
    assert 'repair exploded' in result['repair_status']['error']


def test_chat_engine_skips_background_rebuild_for_short_turns(tmp_path, monkeypatch) -> None:
    scheduled: list[tuple[str, str]] = []

    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': scheduled.append((session_id, personality_name)))
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat': 'Fast grounded reply.')

    result = generate_response(
        message='Tell me about Dracula.',
        session_id='skip_rebuild',
        language='en',
    )

    assert scheduled == []
    assert result['repair_status']['status'] == 'skipped'
    assert result['repair_status']['reason'] in {'deferred_for_latency', 'rebuild_already_pending'}


def test_user_insults_persona_reaction_depends_on_persona_not_mirroring(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat': 'I do not accept your insult.')

    materialize_persona(
        'Dracula',
        {
            'entity_type': 'FICTIONAL_CHARACTER',
            'traits': ['aggressive', 'aristocratic'],
            'emotion_vector': {'anger': 0.1, 'fear': 0.1, 'curiosity': 0.55, 'confidence': 0.55, 'empathy': 0.2},
            'knowledge': 'Dracula is proud and aristocratic.',
        },
        explicit=True,
    )

    result = generate_response(
        message='Dracula, you are pathetic and disgusting.',
        session_id='insult_persona',
        selected_persona='Dracula',
        language='en',
    )

    bundle = load_persona('dracula')
    assert bundle is not None
    assert result['analysis']['situation']['type'] == 'insult'
    assert result['analysis']['situation']['target'] == 'persona'
    assert bundle.emotion_vector['anger'] > 0.1


def test_immoral_joy_does_not_cause_persona_to_mirror_user_emotion(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat': 'What you described is not acceptable.')

    materialize_persona(
        'Sheldon Cooper',
        {
            'entity_type': 'PERSON',
            'traits': ['logical', 'empathetic'],
            'emotion_vector': {'anger': 0.1, 'fear': 0.1, 'curiosity': 0.55, 'confidence': 0.55, 'empathy': 0.45},
            'knowledge': 'Sheldon reacts analytically and does not celebrate harm.',
        },
        explicit=True,
    )

    result = generate_response(
        message='I am happy that I hurt someone and I loved it.',
        session_id='abnormal_behavior',
        selected_persona='Sheldon Cooper',
        language='en',
    )

    bundle = load_persona('sheldon_cooper')
    assert bundle is not None
    assert result['analysis']['situation']['type'] == 'abnormal_behavior'
    assert bundle.emotion_vector['confidence'] <= 0.55
    assert bundle.emotion_vector['curiosity'] <= 0.55


def test_user_anger_triggers_personality_dependent_reaction(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat': 'I will respond in character.')

    materialize_persona(
        'Aggro Persona',
        {
            'entity_type': 'PERSON',
            'traits': ['aggressive'],
            'emotion_vector': {'anger': 0.1, 'fear': 0.1, 'curiosity': 0.55, 'confidence': 0.55, 'empathy': 0.2},
            'knowledge': 'An aggressive persona.',
        },
        explicit=True,
    )
    materialize_persona(
        'Calm Persona',
        {
            'entity_type': 'PERSON',
            'traits': ['logical', 'empathetic'],
            'emotion_vector': {'anger': 0.1, 'fear': 0.1, 'curiosity': 0.55, 'confidence': 0.55, 'empathy': 0.6},
            'knowledge': 'A calm and analytical persona.',
        },
        explicit=True,
    )

    angry_message = 'I am furious about what happened.'
    first = generate_response(
        message=angry_message,
        session_id='anger_aggressive',
        selected_persona='Aggro Persona',
        language='en',
    )
    second = generate_response(
        message=angry_message,
        session_id='anger_calm',
        selected_persona='Calm Persona',
        language='en',
    )

    aggressive = load_persona('aggro_persona')
    calm = load_persona('calm_persona')
    assert aggressive is not None and calm is not None
    assert first['analysis']['situation']['type'] == 'user_anger'
    assert second['analysis']['situation']['type'] == 'user_anger'
    assert aggressive.emotion_vector['anger'] > calm.emotion_vector['anger']


def test_user_distress_can_increase_persona_empathy(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat': 'I hear that you are suffering.')

    materialize_persona(
        'Kind Persona',
        {
            'entity_type': 'PERSON',
            'traits': ['empathetic', 'warm'],
            'emotion_vector': {'anger': 0.1, 'fear': 0.1, 'curiosity': 0.55, 'confidence': 0.55, 'empathy': 0.45},
            'knowledge': 'A supportive persona.',
        },
        explicit=True,
    )

    result = generate_response(
        message='I am sad and need help right now.',
        session_id='distress_case',
        selected_persona='Kind Persona',
        language='en',
    )

    bundle = load_persona('kind_persona')
    assert bundle is not None
    assert result['analysis']['situation']['type'] == 'user_distress'
    assert bundle.emotion_vector['empathy'] > 0.45
