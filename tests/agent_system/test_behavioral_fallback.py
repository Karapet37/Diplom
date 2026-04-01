from __future__ import annotations

from agent_system.chat_engine import generate_response
from agent_system.persona_engine import load_persona, materialize_persona, record_persona_dossier_fact


def test_behavioral_fallback_uses_principle_based_strategy_for_mentor_persona(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat', role='general': '')
    monkeypatch.setattr('agent_system.chat_engine.runtime_status_snapshot', lambda: {'mode': 'full', 'degraded_modes': []})

    materialize_persona(
        'Dr. Aram Petrosyan',
        {
            'entity_type': 'PERSON',
            'traits': ['skeptical', 'precise', 'direct'],
            'knowledge': 'Aram is an emergency physician from Yerevan who teaches residents during difficult triage shifts.',
            'persona_form': {
                'social_roles': ['mentor'],
                'decision_patterns': ['separates reversible risk from irreversible harm before acting'],
                'values': ['protect people without lying to them'],
                'response_priorities': ['answer_substance', 'clarify_if_underspecified', 'stay_in_character'],
                'speech_tendencies': ['dry', 'precise'],
            },
        },
        explicit=True,
    )

    result = generate_response(
        message='What exactly should I do in a situation you cannot fully verify yet?',
        session_id='fallback_mentor',
        selected_persona='Dr. Aram Petrosyan',
        language='en',
    )

    assert result['fallback_strategy']['strategy'] == 'principle_based_fallback'
    assert result['fallback_strategy']['style_mode'] in {'direct', 'instructive', 'grounded'}
    assert 'rule' in result['assistant_reply'].lower()
    assert 'reversible risk' in result['assistant_reply'].lower() or 'protect people' in result['assistant_reply'].lower()
    assert 'assistant' not in result['assistant_reply'].lower()
    assert result['behavior_trace']['fallback_strategy']['strategy'] == 'principle_based_fallback'


def test_behavioral_fallback_uses_protective_narrowing_for_high_risk_uncertainty(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat', role='general': '')

    materialize_persona(
        'Mariam Vardanyan',
        {
            'entity_type': 'PERSON',
            'traits': ['empathetic', 'protective', 'steady'],
            'knowledge': 'Mariam keeps calm under pressure and protects vulnerable people without theatrical language.',
            'persona_form': {
                'social_roles': ['comforter', 'ally'],
                'conflict_behavior': ['stays present and narrows danger before talking big'],
                'speech_tendencies': ['steady', 'human'],
                'values': ['protect the vulnerable first'],
            },
        },
        explicit=True,
    )

    monkeypatch.setattr(
        'agent_system.chat_engine.build_context',
        lambda **kwargs: {
            'persona_name': 'mariam_vardanyan',
            'current_entity': 'mariam_vardanyan',
            'persona_block': '',
            'graph_context': '',
            'recent_dialogue': '',
            'estimated_tokens': 12,
            'context_debug': {'source_counts': {}, 'selected_items': [], 'semantic_focus': {'kind': 'decision'}},
        },
    )

    result = generate_response(
        message='My friend may have overdosed and I am panicking. Tell me exactly what is happening.',
        session_id='fallback_risk',
        selected_persona='Mariam Vardanyan',
        language='en',
    )

    assert result['fallback_strategy']['strategy'] == 'protective_narrowing'
    assert result['fallback_strategy']['risk_level'] == 'high'
    assert 'narrow' in result['assistant_reply'].lower()
    assert 'improvise' in result['assistant_reply'].lower() or 'consequential' in result['assistant_reply'].lower()
    assert result['behavior_trace']['fallback_strategy']['strategy'] == 'protective_narrowing'


def test_record_persona_dossier_fact_enriches_behavior_after_family_softening_update(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    materialize_persona(
        'Aram Petrosyan',
        {
            'entity_type': 'PERSON',
            'traits': ['responsible', 'direct', 'practical'],
            'knowledge': 'Aram is responsible, direct, and practical.',
            'persona_form': {
                'values': ['responsibility', 'practical thinking'],
                'conflict_behavior': ['sets boundaries quickly under stress'],
                'decision_patterns': ['checks what can fail before he commits'],
            },
        },
        explicit=True,
    )

    record_persona_dossier_fact(
        'Aram Petrosyan',
        "Aram has a son and became softer toward close people's mistakes.",
    )

    bundle = load_persona('aram_petrosyan')
    assert bundle is not None
    assert 'responsible' in [item.lower() for item in bundle.traits]
    assert 'direct' in [item.lower() for item in bundle.traits]
    assert 'practical' in [item.lower() for item in bundle.traits]
    assert any('has a son' in item.lower() for item in list(bundle.persona_form.get('memories') or []))
    assert any('patience toward close people' in item.lower() for item in list(bundle.persona_form.get('conflict_behavior') or []))
    assert any('more protective and less cold' in item.lower() for item in list(bundle.persona_form.get('reaction_patterns') or []))
    assert any('without abandoning responsibility' in item.lower() for item in list(bundle.persona_form.get('values') or []))
