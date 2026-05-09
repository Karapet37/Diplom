from __future__ import annotations

import json

import agent_system.chat_engine as chat_engine_module
from agent_system.chat_engine import generate_response
from agent_system.history_store import append_turn, create_session, parse_session
from agent_system.message_annotation_store import build_annotation_workspace, save_message_annotation
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
    assert result['persona_selection']['persona_name'] == 'Dracula'
    assert result['persona_selection']['source'] == 'explicit_selection'
    assert result['persona_response']['persona_name'] == 'Dracula'
    assert str(result['persona_response']['response_style']).strip()
    assert result['persona_response']['state_influences']
    payload = json.loads((tmp_path / 'memory' / 'heads' / 'dracula' / 'examples.json').read_text(encoding='utf-8'))
    assert 'I answer in the first person as Dracula.' not in payload['examples']
    assert payload['situation_reactions']
    assert payload['situation_reactions'][0]['situation'].startswith('type=neutral_query;')
    assert payload['situation_reactions'][0]['reaction'] != 'I answer in the first person as Dracula.'
    assert 'response_style=' in payload['situation_reactions'][0]['reaction']


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


def test_chat_engine_prefers_detected_message_language_for_visible_reply(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)

    def fake_model(prompt: str, mode: str = 'chat', *, role: str = 'general') -> str:
        if mode == 'translation':
            return 'Я Дракула.'
        return 'I am Dracula.'

    monkeypatch.setattr('agent_system.llm._call_model', fake_model)

    result = generate_response(
        message='Привет, кто ты, Дракула?',
        session_id='session_ru',
        selected_persona='Dracula',
        language='en',
    )

    assert result['analysis']['user_state']['language'] == 'ru'
    assert result['response_language'] == 'ru'
    assert result['assistant_reply'] == 'Я Дракула.'


def test_chat_engine_injects_message_vector_runtime_guidance_into_prompt(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr('agent_system.chat_engine.translate_text', lambda text, **kwargs: text)
    class FailingGenomeStore:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError('skip cognitive runtime for this test')

    monkeypatch.setattr('agent_system.chat_engine.GenomeStore', FailingGenomeStore)

    materialize_persona(
        'Катерина',
        {
            'entity_type': 'PERSON',
            'traits': ['sharp', 'guarded'],
            'knowledge': 'Катерина держит дистанцию и не любит липкий флирт.',
        },
        explicit=True,
    )
    create_session('vector_runtime_prompt', 'Vector Runtime Prompt')
    append_turn('vector_runtime_prompt', 'Ну молодец, конечно.', 'Я это запомнил.')
    workspace = build_annotation_workspace('vector_runtime_prompt')
    last_assistant = workspace['messages'][-1]
    save_message_annotation(
        session_id='vector_runtime_prompt',
        message_payload=last_assistant,
        coordinates={
            'F24': {'main': 'false_praise', 'extra': ['sarcasm']},
            'F35': {'main': 'escalation', 'extra': []},
            'F49': {'main': 'toward_escalation', 'extra': []},
        },
        context_window=last_assistant['context_window'],
        context_matrix=last_assistant['context_matrix'],
        transition_interpretation={
            'from': ['sharpening'],
            'to': ['false_praise', 'toward_escalation'],
            'type': 'masking',
        },
        notes='Runtime prompt guidance test.',
    )

    captured: dict[str, str] = {}

    def fake_build_chat_prompt(**kwargs):  # type: ignore[no-untyped-def]
        captured['route_guidance_block'] = str(kwargs.get('route_guidance_block') or '')
        return 'prompt'

    monkeypatch.setattr('agent_system.chat_engine.build_chat_prompt', fake_build_chat_prompt)
    monkeypatch.setattr('agent_system.chat_engine.generate_chat_reply', lambda **kwargs: 'Короткий ответ по делу.')

    result = generate_response(
        message='Why did you answer like that?',
        session_id='vector_runtime_prompt',
        selected_persona='Катерина',
        language='en',
    )

    vector_runtime = result['context_preview']['message_vector_runtime']
    assert '[P-COORDINATE LAYER]' in captured['route_guidance_block']
    assert 'Context matrix:' in captured['route_guidance_block']
    assert vector_runtime['context_matrix'][-1]['vector']['F35']['main'] == 'escalation'
    assert vector_runtime['transition_interpretation']['type'] in {'masking', 'escalation'}
    assert result['behavior_trace']['message_vector_runtime']['current_vector']


def test_chat_engine_uses_internal_translation_for_context_routing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)

    captured: dict[str, str] = {}

    def fake_translate(text: str, target_language: str, source_language: str = 'auto', role: str = 'translator') -> str:
        if target_language == 'en':
            return 'What else is part of your working routine?'
        if target_language == 'ru':
            return 'Я Дракула.'
        return text

    def fake_build_context(*, question: str, session_id: str, selected_persona: str = '', explicit_context: str = '', situation=None, store=None):
        captured['context_question'] = question
        return {
            'persona_name': 'dracula',
            'current_entity': 'dracula',
            'persona_block': 'You are Dracula.',
            'graph_context': '',
            'recent_dialogue': '',
            'estimated_tokens': 32,
            'context_debug': {},
        }

    def fake_build_prompt(*, question: str, internal_question: str = '', persona_block: str = '', graph_context: str = '', recent_dialogue: str = '', language: str = 'en') -> str:
        captured['prompt_question'] = question
        captured['prompt_internal_question'] = internal_question
        return 'prompt'

    monkeypatch.setattr('agent_system.chat_engine.translate_text', fake_translate)
    monkeypatch.setattr('agent_system.chat_engine.build_context', fake_build_context)
    monkeypatch.setattr('agent_system.chat_engine.build_chat_prompt', fake_build_prompt)
    monkeypatch.setattr('agent_system.chat_engine.generate_chat_reply', lambda prompt, language='en', persona_selected=False: 'I am Dracula.')

    result = generate_response(
        message='Что входит в твою рабочую рутину, Дракула?',
        session_id='session_ru_internal',
        selected_persona='Dracula',
        language='en',
    )

    assert captured['context_question'] == 'What else is part of your working routine?'
    assert captured['prompt_question'] == 'Что входит в твою рабочую рутину, Дракула?'
    assert captured['prompt_internal_question'] == 'What else is part of your working routine?'
    assert result['response_language'] == 'ru'


def test_chat_engine_adds_explicit_persona_fact_to_learned_dossier(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)

    materialize_persona(
        'Dr. Aram Petrosyan',
        {
            'entity_type': 'PERSON',
            'traits': ['skeptical', 'precise'],
            'knowledge': 'Aram is an emergency physician from Yerevan.',
        },
        explicit=True,
    )

    captured: dict[str, object] = {}

    def fake_update(name: str, fact: str):  # type: ignore[no-untyped-def]
        captured['name'] = name
        captured['fact'] = fact
        return load_persona(name)

    monkeypatch.setattr('agent_system.chat_engine.record_persona_dossier_fact', fake_update)

    result = generate_response(
        message='For the record, you work night toxicology shifts and keep handwritten shift cards.',
        session_id='persona_fact',
        selected_persona='Dr. Aram Petrosyan',
        language='en',
    )

    assert captured['name'] == 'dr_aram_petrosyan'
    assert 'night toxicology shifts' in captured['fact']
    assert str(captured['fact']).lower().startswith('you work')
    assert 'for the record' not in str(captured['fact']).lower()
    assert 'record' in result['assistant_reply'].lower()
    assert 'future' in result['assistant_reply'].lower() or 'later' in result['assistant_reply'].lower()
    assert 'learned_update' in result['side_effects']['persona_updates']
    assert any('added to the learned dossier' in item for item in result['operator_messages'])


def test_chat_engine_exposes_task_procedure_and_anti_mixing_contract(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat': 'I distrust vague promises because they hide risk, blur responsibility, and tempt people to skip verification.')

    materialize_persona(
        'Dr. Aram Petrosyan',
        {
            'entity_type': 'PERSON',
            'traits': ['skeptical', 'precise', 'empathetic'],
            'persona_form': {
                'social_roles': ['mentor', 'critic'],
                'decision_patterns': ['sorts facts before committing to a judgment'],
                'values': ['protect the vulnerable first'],
            },
        },
        explicit=True,
    )

    result = generate_response(
        message='Briefly explain why, when facts are incomplete, you distrust vague promises.',
        session_id='task_contract_live',
        selected_persona='Dr. Aram Petrosyan',
        language='en',
    )

    assert result['state_transition']['task_procedure']['procedure_family']
    assert result['state_transition']['task_procedure']['response_form'] in {
        'first_person_explanation',
        'direct_answer',
        'clarifying_answer',
    }
    assert 'generic_assistant_tone' in result['state_transition']['task_procedure']['forbidden_mixins']
    assert result['behavior_trace']['task_procedure']['success_criteria']
    assert result['context_preview']['task_procedure']['content_sources']
    assert result['state_transition']['response_plan']['forbidden_mixins']


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


def test_chat_engine_marks_background_rebuild_degraded_when_executor_is_closed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr(chat_engine_module, '_REPAIR_STATUS', {})
    monkeypatch.setattr(chat_engine_module, '_BACKGROUND_EXECUTOR_CLOSED', True)

    chat_engine_module._schedule_background_extraction('closed_executor_case', 'Dracula')

    repair_status = chat_engine_module._get_repair_status('closed_executor_case')
    assert repair_status['status'] == 'degraded'
    assert repair_status['error'] == 'background_executor_unavailable'


def test_chat_engine_bounds_background_repair_status_memory(monkeypatch) -> None:
    monkeypatch.setattr(chat_engine_module, '_REPAIR_STATUS', {})

    for index in range(chat_engine_module._REPAIR_STATUS_LIMIT + 7):
        chat_engine_module._set_repair_status(f'session-{index}', {'status': 'ok', 'index': index})

    stored = dict(chat_engine_module._REPAIR_STATUS)
    assert len(stored) == chat_engine_module._REPAIR_STATUS_LIMIT
    assert 'session-0' not in stored
    assert f"session-{chat_engine_module._REPAIR_STATUS_LIMIT + 6}" in stored


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


def test_chat_engine_uses_session_context_for_follow_up_entity_questions(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)

    prompts: list[str] = []

    def fake_reply(prompt: str, language: str = 'en', persona_selected: bool = False) -> str:
        prompts.append(prompt)
        lowered = prompt.lower()
        if 'does he have a brother?' in lowered:
            assert 'recent dialogue:' in lowered
            assert 'who is jack sparrow?' in lowered
            assert 'jack sparrow is a pirate. what about him?' in lowered
            assert 'jack sparrow' in lowered
            return "Yes. Jack Sparrow has a brother, and the family is strange. His brother carries around their mother's shrunken head."
        if 'who is jack sparrow?' in lowered:
            return 'Jack Sparrow is a pirate. What about him?'
        return 'Unexpected reply path.'

    monkeypatch.setattr('agent_system.chat_engine.generate_chat_reply', fake_reply)

    first = generate_response(
        message='Who is Jack Sparrow?',
        session_id='jack_context',
        language='en',
    )
    second = generate_response(
        message='Does he have a brother?',
        session_id='jack_context',
        language='en',
    )

    parsed = parse_session('jack_context')

    assert first['assistant_reply'] == 'Jack Sparrow is a pirate. What about him?'
    assert second['state_transition']['previous_state']['current_entity'] == 'Jack Sparrow'
    assert second['context_preview']['current_entity'] == 'Jack Sparrow'
    assert second['assistant_reply'].startswith('Yes. Jack Sparrow has a brother')
    assert 'whose brother' not in second['assistant_reply'].lower()
    assert parsed is not None
    assert len(parsed['messages']) == 4
    assert len(prompts) == 2


def test_chat_engine_persists_speaker_persona_across_session_while_topic_entity_changes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setenv('COGNITIVE_RUNTIME_DIR', str(tmp_path / 'runtime'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)

    prompts: list[str] = []

    def fake_reply(prompt: str, language: str = 'en', persona_selected: bool = False) -> str:
        prompts.append(prompt)
        lowered = prompt.lower()
        if 'why is he dangerous?' in lowered:
            assert 'persona head:' in lowered
            assert 'peter parker' in lowered
            assert 'recent dialogue:' in lowered
            assert 'mysterio' in lowered
            return 'Mysterio is dangerous because he weaponizes spectacle and lies, and I learned that the hard way.'
        if 'who is mysterio?' in lowered:
            assert 'peter parker' in lowered
            return 'I am Peter Parker. Mysterio is a manipulator who hides behind illusions.'
        return 'Unexpected reply path.'

    monkeypatch.setattr('agent_system.chat_engine.generate_chat_reply', fake_reply)

    first = generate_response(
        message="Okay, you're Peter Parker. Who is Mysterio?",
        session_id='speaker_topic_session',
        language='en',
    )
    second = generate_response(
        message='Why is he dangerous?',
        session_id='speaker_topic_session',
        language='en',
    )

    assert first['persona_name'] == 'peter_parker'
    assert second['persona_name'] == 'peter_parker'
    assert first['context_preview']['current_entity'] == 'Mysterio'
    assert second['context_preview']['current_entity'] == 'Mysterio'
    assert second['state_transition']['previous_state']['persona_name'] == 'Peter Parker'
    assert second['assistant_reply'].startswith('Mysterio is dangerous because')
    assert len(prompts) == 2


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
    assert first['persona_response']['response_style'] in {'assertive', 'defensive', 'firm_boundary', 'steady'}
    assert second['persona_response']['response_style'] in {'de_escalating', 'measured_support', 'steady', 'supportive'}


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
