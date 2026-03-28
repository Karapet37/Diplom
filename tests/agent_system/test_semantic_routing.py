from __future__ import annotations

from agent_system.persona_engine import materialize_persona, load_persona
from agent_system.semantic_routing import infer_semantic_focus
from agent_system.message_analyzer import analyze_message_state
from agent_system.models import MessageAnalysis
from agent_system.situation_engine import model_situation


def test_semantic_focus_generalizes_beyond_fixed_prompt_wording(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    materialize_persona(
        'Dr. Aram Petrosyan',
        {
            'entity_type': 'PERSON',
            'traits': ['skeptical', 'precise', 'empathetic'],
            'persona_form': {
                'social_roles': ['mentor', 'critic'],
                'work_habits': ['reviews decisions after difficult nights', 'keeps handwritten shift cards'],
                'decision_patterns': ['sorts facts before committing to a judgment'],
                'memory_anchors': ['a winter night when delayed care cost precious minutes'],
                'values': ['protect the vulnerable first'],
            },
        },
        explicit=True,
    )
    bundle = load_persona('Dr. Aram Petrosyan')
    assert bundle is not None
    prepared = analyze_message_state(
        message='When facts are incomplete, how do you usually sort yourself out and decide what matters first?',
        session_id='semantic_focus',
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
        session_id='semantic_focus',
        selected_head=prepared['selected_head'],
        primary_entity=prepared['primary_entity'],
        current_entity=prepared['current_entity'],
        explicit_context='',
        entities=list(prepared['entities']),
        user_state=prepared['user_state'],
        situation=situation,
    )

    payload = infer_semantic_focus(
        question=prepared['message'],
        persona_bundle=bundle,
        analysis=analysis,
        situation=situation,
    )

    assert 'decision' in payload['focus']
    assert payload['evidence']['decision']
