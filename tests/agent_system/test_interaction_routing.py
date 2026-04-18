from __future__ import annotations

from agent_system.interaction_routing import route_interaction
from agent_system.message_analyzer import analyze_message_state
from agent_system.models import InteractionFrame


def test_interaction_router_can_separate_speaker_persona_from_topic_entity_with_llm_guidance(monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_STAGE_MODEL_STEPS', 'interaction_router')

    def fake_model(prompt: str, mode: str = 'chat', *, role: str = 'general') -> str:
        if 'INTERACTION_ROUTER stage' not in prompt:
            return ''
        return (
            '{"message_kind":"question","question_present":true,'
            '"explicit_persona_switch":true,"requested_persona":"Peter Parker",'
            '"keep_session_persona":false,"topic_entity":"Mysterio",'
            '"topic_mode":"external_topic","followup_mode":"none",'
            '"routed_message":"Why is Mysterio dangerous?",'
            '"evidence":["explicit persona switch and external topic"],'
            '"source":"llm_guided"}'
        )

    monkeypatch.setattr('agent_system.llm._call_model', fake_model)

    frame = route_interaction(
        message='Why did he do that?',
        session_persona='',
        current_entity='',
        known_entities=[
            {'name': 'Peter Parker', 'aliases': ['Spider-Man'], 'type': 'FICTIONAL_CHARACTER'},
            {'name': 'Mysterio', 'aliases': ['Quentin Beck'], 'type': 'FICTIONAL_CHARACTER'},
        ],
    )

    assert frame.source == 'llm_guided'
    assert frame.explicit_persona_switch is True
    assert frame.requested_persona == 'Peter Parker'
    assert frame.topic_entity == 'Mysterio'
    assert frame.routed_message == 'Why is Mysterio dangerous?'


def test_interaction_router_detects_second_person_identity_frame_deterministically() -> None:
    frame = route_interaction(
        message='ты вампир, изгнанный из своей стаи. как ты питаешься?',
        session_persona='',
        current_entity='',
        known_entities=[],
    )

    assert frame.source == 'deterministic'
    assert frame.explicit_persona_switch is True
    assert frame.requested_persona == 'Вампир'
    assert frame.topic_mode in {'persona_switch', 'persona_self'}
    assert frame.question_present is True


def test_interaction_router_skips_llm_when_deterministic_seed_is_decisive(monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_STAGE_MODEL_STEPS', 'interaction_router')
    monkeypatch.setattr('agent_system.llm._call_model', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('llm router should be skipped')))

    frame = route_interaction(
        message="Okay, you're Peter Parker. Why is Mysterio dangerous?",
        session_persona='',
        current_entity='',
        known_entities=[
            {'name': 'Peter Parker', 'aliases': ['Spider-Man'], 'type': 'FICTIONAL_CHARACTER'},
            {'name': 'Mysterio', 'aliases': ['Quentin Beck'], 'type': 'FICTIONAL_CHARACTER'},
        ],
    )

    assert frame.source == 'deterministic'
    assert frame.explicit_persona_switch is True
    assert frame.requested_persona == 'Peter Parker'
    assert frame.topic_entity == 'Mysterio'


def test_message_analyzer_uses_interaction_frame_as_source_of_followup_topic() -> None:
    prepared = analyze_message_state(
        message='Does he have a brother',
        session_id='followup_frame',
        current_entity='Jack Sparrow',
        interaction_frame=InteractionFrame(
            message_kind='question',
            question_present=True,
            keep_session_persona=True,
            topic_entity='Jack Sparrow',
            topic_mode='session_followup',
            followup_mode='followup_on_previous_topic',
            routed_message='Does he have a brother',
            source='llm_guided',
        ),
        known_entities=[],
    )

    assert prepared['primary_entity'] == 'Jack Sparrow'
    assert prepared['user_state'].intent == 'question'
    assert prepared['user_state'].signal('contains_question') == 1.0
