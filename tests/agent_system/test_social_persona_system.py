from __future__ import annotations

from agent_system.chat_engine import generate_response
from agent_system.message_analyzer import analyze_message_state
from agent_system.mood_research import (
    analyze_mood_research,
    build_mood_snapshot,
    load_mood_report,
    record_mood_snapshot,
    refresh_mood_reports,
)
from agent_system.models import MessageAnalysis, SocialRoleDecision
from agent_system.persona_engine import explain_persona_graph, load_persona, materialize_persona
from agent_system.social_roles import choose_social_role
from agent_system.situation_engine import model_situation


def _materialize_aram() -> None:
    materialize_persona(
        'Dr. Aram Petrosyan',
        {
            'entity_type': 'PERSON',
            'traits': ['skeptical', 'precise', 'empathetic', 'disciplined'],
            'knowledge': 'Emergency physician from Yerevan who mentors younger clinicians and keeps hard-won field notes.',
            'examples': [
                'I do not waste words when someone is bleeding or lying to themselves.',
                'If there is uncertainty, I narrow it fast and explain what matters.',
            ],
            'persona_form': {
                'biography': 'Aram is an emergency physician from Yerevan who splits his time between a city hospital and rural outreach clinics.',
                'social_roles': ['mentor', 'critic', 'comforter'],
                'habits': ['keeps handwritten shift cards', 'reviews decisions after difficult nights'],
                'values': ['protect the vulnerable first', 'do not confuse panic with urgency'],
                'conflicts': ['resentful of bureaucracy that slows care'],
                'topic_affinities': ['triage', 'toxicology', 'medical ethics'],
                'speech_tendencies': ['dry under pressure', 'fact-first', 'short sentences when stakes are high'],
                'memories': ['a winter night when a delayed ambulance cost a patient precious minutes'],
                'reaction_patterns': ['sharpens under pressure', 'turns colder when someone hides critical facts'],
                'trust_model': ['trust is earned by honesty under strain'],
            },
        },
        explicit=True,
    )


def test_persona_graph_materializes_social_structure(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    _materialize_aram()

    explanation = explain_persona_graph('Dr. Aram Petrosyan')

    assert explanation is not None
    assert 'social personality graph' in explanation.summary.lower()
    assert explanation.central_nodes
    assert any('bureaucracy' in item.lower() for item in explanation.conflict_nodes)

    bundle = load_persona('Dr. Aram Petrosyan')
    assert bundle is not None
    graph = bundle.meta  # keep bundle alive for inspection
    local_graph = (tmp_path / 'memory' / 'heads' / 'dr_aram_petrosyan' / 'local_graph.json').read_text(encoding='utf-8')
    assert 'CAN_PLAY_ROLE' in local_graph
    assert 'HAS_HABIT' in local_graph
    assert 'AFFINITY_FOR' in local_graph
    assert graph is not None


def test_mood_research_persists_clusters_and_role_effects(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    _materialize_aram()
    bundle = load_persona('Dr. Aram Petrosyan')
    assert bundle is not None

    prepared = analyze_message_state(
        message='I am scared and I need your help deciding what to do next.',
        session_id='mood_session',
        selected_head='Dr. Aram Petrosyan',
        known_entities=[],
    )
    situation = model_situation(
        message=prepared['message'],
        primary_entity=prepared['primary_entity'],
        selected_head=prepared['selected_head'],
        user_state=prepared['user_state'],
    )
    analysis = MessageAnalysis(
        message=prepared['message'],
        session_id='mood_session',
        selected_head=prepared['selected_head'],
        primary_entity=prepared['primary_entity'],
        current_entity=prepared['current_entity'],
        explicit_context='',
        entities=list(prepared['entities']),
        user_state=prepared['user_state'],
        situation=situation,
    )
    decision = SocialRoleDecision(role='comforter', confidence=0.82, reason='distress-heavy message')
    snapshot = build_mood_snapshot(
        analysis=analysis,
        persona_bundle=bundle,
        social_role=decision,
        response_style='supportive',
        session_id='mood_session',
    )
    record_mood_snapshot(snapshot)
    refresh_mood_reports(persona_name='Dr. Aram Petrosyan', session_id='mood_session')

    report = analyze_mood_research(persona_name='Dr. Aram Petrosyan')
    saved = load_mood_report(persona_name='Dr. Aram Petrosyan')

    assert report.snapshot_count >= 1
    assert report.clusters
    assert report.role_effects
    assert any(str(item.get('role') or '') == 'comforter' for item in report.role_effects)
    assert saved is not None
    assert saved.latest_cluster_label


def test_social_role_selection_prefers_comfort_or_mentor_under_distress(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    _materialize_aram()
    bundle = load_persona('Dr. Aram Petrosyan')
    assert bundle is not None

    prepared = analyze_message_state(
        message='I do not know what to do and I am falling apart a little.',
        session_id='role_session',
        selected_head='Dr. Aram Petrosyan',
        known_entities=[],
    )
    situation = model_situation(
        message=prepared['message'],
        primary_entity=prepared['primary_entity'],
        selected_head=prepared['selected_head'],
        user_state=prepared['user_state'],
    )
    analysis = MessageAnalysis(
        message=prepared['message'],
        session_id='role_session',
        selected_head=prepared['selected_head'],
        primary_entity=prepared['primary_entity'],
        current_entity=prepared['current_entity'],
        explicit_context='',
        entities=list(prepared['entities']),
        user_state=prepared['user_state'],
        situation=situation,
    )

    decision = choose_social_role(bundle=bundle, analysis=analysis, situation=situation)

    assert decision.role in {'comforter', 'mentor', 'ally'}
    assert decision.reason
    assert decision.evidence


def test_chat_engine_records_social_role_and_mood_research(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr('agent_system.chat_engine.schedule_mood_research_refresh', lambda persona_name='', session_id='': None)
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat', role='general': 'I will answer as Aram, terse and human.')
    _materialize_aram()

    captured: dict[str, str] = {}

    def fake_prompt_builder(**kwargs):  # type: ignore[no-untyped-def]
        captured['social_role_block'] = str(kwargs.get('social_role_block') or '')
        captured['mood_research_block'] = str(kwargs.get('mood_research_block') or '')
        return 'prompt'

    monkeypatch.setattr('agent_system.chat_engine.build_chat_prompt', fake_prompt_builder)
    monkeypatch.setattr('agent_system.chat_engine.generate_chat_reply', lambda prompt, language='en', persona_selected=False: 'I will answer as Aram, terse and human.')

    result = generate_response(
        message='I need a hard answer, not comfort. What am I avoiding?',
        session_id='chat_social',
        selected_persona='Dr. Aram Petrosyan',
        language='en',
    )

    report = analyze_mood_research(session_id='chat_social')

    assert result['social_role']['role']
    assert result['context_preview']['social_role']['role'] == result['social_role']['role']
    assert 'Selected social role:' in captured['social_role_block']
    assert report.snapshot_count >= 1
    assert report.latest_cluster_label
    assert result['behavior_trace']['semantic_focus']['focus']
    assert result['behavior_trace']['social_role']['role'] == result['social_role']['role']
