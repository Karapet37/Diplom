from __future__ import annotations

import json

from agent_system.context_builder import build_context
from agent_system.demo import run_demo
from agent_system.graph_store import GraphStore, graph_nodes_path
from agent_system.history_store import create_session
from agent_system.persona_engine import materialize_persona
from agent_system.prompt_builder import build_chat_prompt


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

    assert 'Current stance:' in built['persona_block']
    assert 'Situation note:' in built['persona_block']
    assert 'Response style:' not in built['persona_block']
    assert 'Emotion vector:' not in built['persona_block']
    assert built['situation'] == 'type=neutral_query; target=persona; severity=0.45'
    assert 'logical' in built['persona_block']
    assert 'Leonard' in built['graph_context']
    assert built['estimated_tokens'] <= 4000
    assert built['recent_dialogue']


def test_context_builder_surfaces_persona_identity_biography_and_anchors(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    materialize_persona(
        'Dr. Aram Petrosyan',
        {
            'entity_type': 'PERSON',
            'traits': ['skeptical', 'precise', 'dryly humorous'],
            'examples': ['Let us separate signal from noise.'],
            'knowledge': 'Aram is an emergency physician and triage lead from Yerevan.',
            'persona_form': {
                'identity_class': 'human',
                'biography': 'Aram Petrosyan is an emergency physician and triage lead from Yerevan who also rotates through rural clinics in Lori.',
                'values': ['evidence over theater', 'protect the vulnerable first'],
                'speech_style': ['concise', 'dryly humorous when tension rises', 'fact-first'],
                'emotional_tendencies': ['steady under pressure', 'warmer with genuine distress'],
                'trust_model': ['trust is earned through clarity, follow-through, and willingness to revise'],
                'memory_anchors': ['steel watch from father', 'blue field notebook'],
                'recurring_style_markers': ['Let us separate signal from noise.', 'I do not fake certainty.'],
                'decision_patterns': ['stabilize risk first'],
                'response_priorities': ['protect_people'],
                'clarification_policy': 'Ask for the missing fact that would change the decision.',
                'sarcasm_profile': 'low_to_medium',
            },
            'decision_explanation': 'Aram stabilizes risk first, then strips away noise before answering.',
        },
        explicit=True,
    )
    create_session('aram_identity', 'Aram Identity')

    built = build_context(
        question='Who are you and how do you think under pressure?',
        session_id='aram_identity',
        selected_persona='dr_aram_petrosyan',
        situation={'type': 'neutral_query', 'target': 'persona', 'severity': 0.4},
    )

    assert built['context_debug']['question_focus'] == ['decision', 'identity']
    assert 'Biography: Aram Petrosyan is an emergency physician' in built['persona_block']
    assert 'Memory anchors: steel watch from father | blue field notebook.' in built['persona_block']
    assert 'Recurring style markers: Let us separate signal from noise.' in built['persona_block']
    assert 'Learned situation reactions:' not in built['persona_block']
    assert 'Preferred reply shape' not in built['persona_block']


def test_context_builder_prioritizes_work_profile_for_workday_questions(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    materialize_persona(
        'Dr. Aram Petrosyan',
        {
            'entity_type': 'PERSON',
            'traits': ['skeptical', 'precise', 'dryly humorous'],
            'knowledge': 'Aram is an emergency physician and triage lead from Yerevan.',
            'relations': [
                {'type': 'WORKS_IN', 'target': 'Emergency medicine'},
                {'type': 'SPECIALIZES_IN', 'target': 'Triage'},
                {'type': 'WORKS_WITH', 'target': 'Rural clinics'},
                {'type': 'LIVES_IN', 'target': 'Yerevan'},
            ],
            'persona_form': {
                'identity_class': 'human',
                'biography': 'Aram Petrosyan is an emergency physician and triage lead based in Yerevan. He rotates through rural clinics in Lori.',
                'work_habits': [
                    'keeps a blue field notebook',
                    'writes after hard shifts',
                    'tracks near-misses and outcomes',
                    'reviews what almost fooled him',
                    'mentors new emergency residents during overnight shifts',
                ],
                'decision_patterns': ['stabilize risk first', 'separate signal from noise'],
                'memory_anchors': ['steel watch from father', 'blue field notebook'],
            },
        },
        explicit=True,
    )
    create_session('aram_workday', 'Aram Workday')

    built = build_context(
        question='What else is part of your working routine?',
        session_id='aram_workday',
        selected_persona='dr_aram_petrosyan',
        situation={'type': 'neutral_query', 'target': 'persona', 'severity': 0.4},
    )

    assert built['context_debug']['question_focus'] == ['work']
    assert 'Professional role:' in built['persona_block']
    assert 'Daily work habits:' in built['persona_block']
    assert 'Work structure:' in built['persona_block']
    assert 'mentors new emergency residents during overnight shifts' in built['persona_block']


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


def test_build_chat_prompt_hardens_direct_persona_self_questions() -> None:
    prompt = build_chat_prompt(
        question='What does your normal working day look like?',
        persona_block='You are Dr. Aram Petrosyan.',
        graph_context='',
        recent_dialogue='',
        language='en',
    )

    assert 'This is a direct personal question about the persona.' in prompt
    assert 'Because the user is asking about a normal day or work routine' in prompt
    assert 'User question:' in prompt
    assert prompt.index('User question:') < prompt.index('Persona head:')
    assert 'Keep the reply brief by default: 2 to 4 sentences unless the user explicitly asks for detail.' in prompt
    assert 'Do not output analysis, hidden reasoning, `<think>` tags, system notes, prompt commentary, or JSON.' in prompt
    assert 'Do not mention being an AI, assistant, language model, system, or following instructions.' in prompt


def test_build_chat_prompt_persona_mode_uses_role_activation_scaffold() -> None:
    prompt = build_chat_prompt(
        question='кто тебе нравится?',
        persona_block='Имя/ярлык: Сломанный гордец.\nВнутри ты зависим и стыдишься своей привязанности.',
        graph_context='- relation: Y KNOWS Persona',
        recent_dialogue='user: ну и кто это?',
        reviewed_context_block='persona is under emotional pressure',
        route_guidance_block='Requested persona disposition: shy, proud, dependent.',
        answer_perspective='persona',
        language='ru',
    )

    assert '[ROLE ACTIVATION]' in prompt
    assert 'Ты ЕСТЬ этот человек.' in prompt
    assert 'перпендикулярно' in prompt
    assert '[PERSONA]' in prompt
    assert '[RESPONSE FORMAT]' in prompt
    assert '[USER INPUT]' in prompt
    assert 'Верни только внешнюю реплику персонажа обычным plain text.' in prompt
    assert 'короткая внутренняя мысль в скобках' not in prompt
    assert '[KNOWLEDGE GRAPH]' in prompt
    assert '[RECENT DIALOGUE]' in prompt


def test_build_chat_prompt_persona_review_mode_uses_dialogue_analysis_scaffold() -> None:
    prompt = build_chat_prompt(
        question='Проанализируй диалог персонажа и найди ошибки в отыгрыше.',
        persona_block='Имя/ярлык: Сломанный гордец.\nВнешне сдержанный, внутри зависимый и стыдливый.',
        graph_context='- relation: Y KNOWS Persona',
        recent_dialogue='assistant: Да отстань.\nuser: Почему ты злишься?',
        reviewed_context_block='persona drifts into confident speech under pressure',
        route_guidance_block='Для каждой ошибки: ошибка, почему это ошибка, как лучше, исправленный вариант реплики.',
        answer_perspective='persona_review',
        language='ru',
    )

    assert '[ROLEPLAY REVIEW]' in prompt
    assert 'Ты анализируешь диалог персонажа, а не продолжаешь сцену.' in prompt
    assert '[CHECKLIST]' in prompt
    assert '[OUTPUT FORMAT]' in prompt
    assert 'исправленный вариант реплики' in prompt
    assert '[DIALOGUE TO REVIEW]' in prompt
    assert 'Recent dialogue:' in prompt
    assert 'Persona head:' in prompt


def test_build_chat_prompt_compacts_large_redundant_sections() -> None:
    repeated_persona = '\n'.join(['Имя/ярлык: Катерина'] + ['Внешне ты: сдержанная, холодная, собранная.'] * 12)
    repeated_graph = '\n'.join(['- relation: Y KNOWS Persona'] * 20)
    repeated_dialogue = '\n'.join(['user: привет'] * 12 + ['assistant: ...'] * 12)

    prompt = build_chat_prompt(
        question='Расскажи коротко, как ты будешь отвечать дальше?',
        persona_block=repeated_persona,
        graph_context=repeated_graph,
        recent_dialogue=repeated_dialogue,
        reviewed_context_block='persona under pressure\npersona under pressure\npersona under pressure',
        route_guidance_block='Requested persona disposition: restrained, proud.\nRequested persona disposition: restrained, proud.',
        answer_perspective='persona',
        language='ru',
    )

    assert prompt.count('Внешне ты: сдержанная, холодная, собранная.') == 1
    assert prompt.count('- relation: Y KNOWS Persona') == 1
    assert prompt.count('user: привет') <= 2
    assert len(prompt) < 5000
