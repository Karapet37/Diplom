from __future__ import annotations

from agent_system.message_analyzer import analyze_message_state
from agent_system.models import InfluenceInterpretation, MessageAnalysis
from agent_system.persona_engine import load_persona, materialize_persona
from agent_system.situation_engine import model_situation
from agent_system.task_procedures import seed_task_procedure


def _analysis_for(message: str, *, session_id: str, selected_head: str = '') -> MessageAnalysis:
    prepared = analyze_message_state(
        message=message,
        session_id=session_id,
        selected_head=selected_head,
        known_entities=[],
    )
    situation = model_situation(
        message=prepared['message'],
        primary_entity=prepared['primary_entity'],
        selected_head=prepared['selected_head'],
        user_state=prepared['user_state'],
    )
    return MessageAnalysis(
        message=prepared['message'],
        session_id=session_id,
        selected_head=prepared['selected_head'],
        primary_entity=prepared['primary_entity'],
        current_entity=prepared['current_entity'],
        explicit_context='',
        entities=list(prepared['entities']),
        user_state=prepared['user_state'],
        situation=situation,
    )


def test_task_procedure_reconstructs_form_content_and_success_contract(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    materialize_persona(
        'Dr. Aram Petrosyan',
        {
            'entity_type': 'PERSON',
            'traits': ['skeptical', 'precise', 'empathetic'],
            'persona_form': {
                'social_roles': ['mentor', 'critic'],
                'decision_patterns': ['sorts facts before committing to a judgment'],
                'values': ['protect the vulnerable first'],
                'memory_anchors': ['a winter night when delayed care cost precious minutes'],
                'personal_history': ['worked emergency and rural rotations for years'],
            },
        },
        explicit=True,
    )
    bundle = load_persona('Dr. Aram Petrosyan')
    assert bundle is not None
    analysis = _analysis_for(
        'Lay out three concrete reasons why, when facts are incomplete, you still distrust vague promises.',
        session_id='task_contract',
        selected_head='Dr. Aram Petrosyan',
    )
    influence = InfluenceInterpretation(
        summary='The message asks for a concrete structured answer about judgment under uncertainty.',
        themes=['decision', 'values'],
        pressure_points=['needs_answer'],
        direction='answer_from_updated_state',
        tension='moderate',
        role_pressure='mentor',
        uncertainty_level='moderate',
        risk_level='low',
    )

    plan, semantic_focus = seed_task_procedure(
        message=analysis.message,
        reply_language='en',
        analysis=analysis,
        situation=analysis.situation,
        previous_state=None,
        influence=influence,
        persona_bundle=bundle,
    )

    assert semantic_focus['focus']
    assert plan.procedure_family in {'persona_decision_answer', 'principle_grounded_answer'}
    assert plan.response_form == 'structured_list'
    assert 'reviewed_context' in plan.content_sources
    assert 'persona_learned_patterns' in plan.content_sources or 'persona_form' in plan.content_sources
    assert 'generic_assistant_tone' in plan.forbidden_mixins
    assert any(item.startswith('visible_reply_language_is_en') for item in plan.success_criteria)
    assert any('identify the real task' in item for item in plan.execution_steps)


def test_task_procedure_marks_dossier_update_as_state_change_not_generic_answer(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    materialize_persona(
        'Dr. Aram Petrosyan',
        {
            'entity_type': 'PERSON',
            'traits': ['skeptical', 'precise'],
        },
        explicit=True,
    )
    bundle = load_persona('Dr. Aram Petrosyan')
    assert bundle is not None
    analysis = _analysis_for(
        'For the record, you have a son now and you have become more patient with the mistakes of close people.',
        session_id='task_dossier',
        selected_head='Dr. Aram Petrosyan',
    )
    influence = InfluenceInterpretation(
        summary='The message updates biography and should alter later behavior without rewriting identity.',
        themes=['statement', 'memory'],
        pressure_points=[],
        direction='answer_from_updated_state',
        tension='moderate',
        role_pressure='ally',
        uncertainty_level='low',
        risk_level='low',
    )

    plan, _semantic_focus = seed_task_procedure(
        message='For the record, you have a son now and you have become more patient with the mistakes of close people.',
        reply_language='en',
        analysis=analysis,
        situation=analysis.situation,
        previous_state=None,
        influence=influence,
        persona_bundle=bundle,
        dossier_update_statement=True,
    )

    assert plan.procedure_family == 'persona_dossier_update'
    assert plan.response_form == 'acknowledgement'
    assert 'user_message' in plan.content_sources
    assert 'new_fact_is_acknowledged' in plan.success_criteria
    assert 'future_influence_is_signaled' in plan.success_criteria
