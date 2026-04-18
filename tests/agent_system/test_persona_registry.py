from __future__ import annotations

import json

import pytest

from agent_system.chat_engine import generate_response
from agent_system.graph_store import load_json, normalize_personality_name, personality_index_path, write_json
from agent_system.history_store import load_session_route_state
from agent_system.persona_engine import (
    create_persona_from_description,
    extract_explicit_persona_name,
    list_personas,
    load_active_persona,
    load_persona,
    materialize_persona,
    rejected_candidates_log_path,
    spawn_head,
)
from agent_system.reliability import MutationRejectedFailure


def _valid_persona_payload() -> dict[str, object]:
    return {
        'entity_type': 'PERSON',
        'traits': ['guarded', 'proud', 'cautious'],
        'examples': ['He hides pain behind dignity and avoids asking for help.'],
        'relations': [{'type': 'AVOIDS', 'target': 'humiliation', 'weight': 0.82}],
        'knowledge': 'A proud and cautious person who defends dignity under pressure.',
        'persona_form': {
            'identity_class': 'human',
            'core_self_image': 'someone who must preserve dignity under pressure',
            'vulnerabilities': ['fear of humiliation'],
            'defense_mechanisms': ['keeps distance before trusting'],
            'triggers': ['being used after showing attachment'],
            'dependency_patterns': ['attachment makes detachment difficult'],
            'communication_style': ['guarded and direct'],
            'internal_contradictions': ['wants closeness but resists visible weakness'],
            'change_resistance': ['resists asking for help'],
            'growth_dynamics': ['learns to set boundaries earlier'],
        },
        'decision_explanation': 'This persona protects dignity first, then decides how much emotional exposure is safe.',
        'structured_persona': {
            'identity': {
                'label': 'Guarded Pride',
                'short_description': 'A proud and cautious person who hides dependence behind guarded self-control.',
                'persona_type': 'psychological',
                'source_text': 'He hides pain behind dignity and avoids asking for help.',
                'readiness': 'full',
            },
            'core': {
                'self_image': ['must preserve dignity under pressure'],
                'visible_traits': ['guarded', 'proud', 'cautious'],
                'hidden_traits': ['dependent', 'ashamed of weakness'],
                'motivations': ['stay respected', 'avoid humiliation'],
                'fears': ['being used', 'being exposed as weak'],
                'needs': ['respect', 'emotional safety'],
                'vulnerabilities': ['fear of humiliation'],
            },
            'conflict': {
                'internal_contradictions': ['wants closeness but resists visible weakness'],
                'shame_points': ['visible dependence'],
                'dependency_patterns': ['attachment makes detachment difficult'],
                'resentment_patterns': ['resentment builds when attachment turns exploitative'],
            },
            'defense': {
                'defense_mechanisms': ['keeps distance before trusting'],
                'self_justifications': ['distance keeps dignity intact'],
                'avoidance_patterns': ['avoids asking for help directly'],
                'escalation_patterns': ['first withdraws, then hardens into cold anger'],
            },
            'behavior': {
                'communication_style': ['guarded and direct'],
                'triggers': ['being used after showing attachment'],
                'pressure_response': ['withdraws to regain dignity'],
                'attachment_style': 'anxious-guarded',
                'refusal_style': 'reluctant but sharper under repeated pressure',
            },
            'dynamics': {
                'resistance_to_change': 'high',
                'growth_pattern': 'learns to set boundaries earlier',
                'likely_change_direction': 'unstable',
                'softening_conditions': ['respectful confrontation'],
                'darkening_conditions': ['humiliation'],
            },
            'meta': {
                'tags': ['pride', 'guardedness', 'dependency'],
                'hover_text': 'Proud and guarded on the surface, dependent and ashamed underneath.',
                'validation_status': 'valid',
                'validation_notes': [],
                'confidence': 0.92,
            },
        },
    }


def test_persona_registry_hygiene_removes_garbage_from_active_pool(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    materialize_persona('Dr. Aram Petrosyan', _valid_persona_payload(), explicit=True)
    spawn_head('PDF', entity_type='PERSON', source='test', register=False)
    spawn_head('Unity', entity_type='PERSON', source='test', register=False)
    spawn_head('ты питаешься', entity_type='PERSON', source='test', register=False)

    write_json(
        personality_index_path(),
        {
            'heads': [
                normalize_personality_name('Dr. Aram Petrosyan'),
                normalize_personality_name('PDF'),
                normalize_personality_name('Unity'),
                normalize_personality_name('ты питаешься'),
            ]
        },
    )

    rows = list_personas()

    assert [item['slug'] for item in rows] == ['dr_aram_petrosyan']
    assert load_active_persona('PDF') is None
    assert load_active_persona('Unity') is None
    assert load_active_persona('ты питаешься') is None

    rejected_log = rejected_candidates_log_path()
    assert rejected_log.exists()
    rejected_rows = [json.loads(line) for line in rejected_log.read_text(encoding='utf-8').splitlines() if line.strip()]
    rejected_names = {str(item.get('name') or '') for item in rejected_rows}
    assert {'PDF', 'Unity', 'ты питаешься'} <= rejected_names


@pytest.mark.parametrize('bad_name', ['PDF', 'Human', 'File', 'Unity', 'ты питаешься'])
def test_persona_materialization_rejects_garbage_names(tmp_path, monkeypatch, bad_name: str) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    with pytest.raises(MutationRejectedFailure):
        materialize_persona(bad_name, _valid_persona_payload(), explicit=True)

    assert load_active_persona(bad_name) is None


def test_create_persona_from_description_builds_structured_object_when_llm_returns_essay(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr(
        'agent_system.persona_engine.call_json_model_for_role',
        lambda *args, **kwargs: 'This is a long narrative essay about the person instead of structured JSON.',
    )

    result = create_persona_from_description(
        'Робкий, но гордый человек. Ему стыдно просить помощи, он боится быть использованным, '
        'но все равно слишком привязывается к тем, кто ему дорог.',
        name_hint='Hidden Lover',
    )

    assert result['created'] is True
    assert result['activated'] is True
    assert result['persona_name'] == 'Hidden Lover'
    persona_object = dict(result['persona_object'])
    assert persona_object['identity']['label'] == 'Hidden Lover'
    assert persona_object['identity']['readiness'] in {'draft', 'full'}
    assert persona_object['core_goal']
    assert persona_object['constraints_internal']
    assert persona_object['constraints_hard_system']
    assert persona_object['allowed_methods']
    assert persona_object['core']['self_image']
    assert persona_object['core']['vulnerabilities']
    assert persona_object['defense']['defense_mechanisms']
    assert persona_object['conflict']['dependency_patterns']
    assert persona_object['behavior']['communication_style']
    assert persona_object['conflict']['internal_contradictions']
    bundle = load_active_persona('Hidden Lover')
    assert bundle is not None
    assert bundle.structured_persona is not None
    assert bundle.structured_persona.meta.validation_status in {'valid', 'partial'}


def test_extract_explicit_persona_name_prefers_human_name_from_description() -> None:
    assert (
        extract_explicit_persona_name(
            'Создай личность\nАня Юсупова, 27, бухгалтер, скромная, робкая и терпеливая.'
        )
        == 'Аня Юсупова'
    )


def test_create_persona_from_description_uses_explicit_name_from_first_profile_line(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr(
        'agent_system.persona_engine.call_json_model_for_role',
        lambda *args, **kwargs: 'Narrative prose instead of structured JSON.',
    )

    result = create_persona_from_description(
        'Создай личность\nАня Юсупова, 27, бухгалтер, скромная, робкая, трудолюбивая, стыдливая и терпеливая.',
        activate=False,
    )

    assert result['persona_name'] == 'Аня Юсупова'
    assert result['persona_slug'] == normalize_personality_name('Аня Юсупова')
    assert result['persona_object']['identity']['label'] == 'Аня Юсупова'


def test_persona_specification_route_creates_object_instead_of_essay(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr(
        'agent_system.persona_engine.call_json_model_for_role',
        lambda *args, **kwargs: 'He is a tragic and complicated person with many conflicting traits.',
    )
    monkeypatch.setattr('agent_system.llm._call_model', lambda *args, **kwargs: 'This essay should not be returned.')

    result = generate_response(
        message='Создай личность: робкий, но гордый человек, которому стыдно просить помощи и который боится быть использованным.',
        session_id='persona_spec_session',
        language='ru',
    )

    assert result['pipeline']['route']['selected_route'] == 'persona_specification'
    assert result['persona_action']['created'] is True
    assert result['persona_action']['persona_object']['identity']['label']
    assert result['persona_action']['persona_object']['core_goal']
    assert result['persona_action']['persona_object']['constraints_internal']
    assert result['persona_action']['persona_object']['allowed_methods']
    assert result['persona_action']['persona_object']['core']['self_image']
    assert result['assistant_reply'].startswith('Создал личность')
    assert 'tragic and complicated person' not in result['assistant_reply']


def test_persona_specification_route_uses_explicit_name_from_description(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr(
        'agent_system.persona_engine.call_json_model_for_role',
        lambda *args, **kwargs: 'Narrative prose instead of structured JSON.',
    )

    result = generate_response(
        message='Создай личность\nАня Юсупова, 27, бухгалтер, скромная, робкая и терпеливая.',
        session_id='persona_named_spec_session',
        language='ru',
    )

    assert result['pipeline']['route']['selected_route'] == 'persona_specification'
    assert result['persona_action']['persona_name'] == 'Аня Юсупова'
    assert result['persona_action']['persona_object']['identity']['label'] == 'Аня Юсупова'
    assert 'Аня Юсупова' in result['assistant_reply']


def test_rich_persona_description_without_explicit_command_creates_persona_object(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr(
        'agent_system.persona_engine.call_json_model_for_role',
        lambda *args, **kwargs: 'Narrative prose instead of structured JSON.',
    )

    result = generate_response(
        message=(
            'Катерина — сильная, холодная, собранная женщина с острым языком и жёсткой внутренней дисциплиной. '
            'Она мало говорит, быстро считывает людей, не терпит слабость как позу и защищает своих без лишней нежности. '
            'Говорит коротко, сухо, уверенно, иногда колко. Внутри уязвимее, чем кажется, но почти никогда этого не показывает.'
        ),
        session_id='persona_rich_profile_session',
        language='ru',
    )

    assert result['pipeline']['route']['selected_route'] == 'persona_specification'
    assert result['persona_action']['created'] is True
    assert result['persona_action']['persona_name'] == 'Катерина'
    assert 'Катерина' in result['assistant_reply']


def test_persona_specification_continues_after_create_command_with_followup_description(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr(
        'agent_system.persona_engine.call_json_model_for_role',
        lambda *args, **kwargs: 'Narrative prose instead of structured JSON.',
    )

    first = generate_response(
        message='создай личность',
        session_id='persona_followup_spec_session',
        language='ru',
    )
    second = generate_response(
        message='Катерина, 31, бухгалтер, скромная, робкая, терпеливая, стыдливая, говорит тихо и коротко, боится навязываться.',
        session_id='persona_followup_spec_session',
        language='ru',
    )

    assert first['pipeline']['route']['selected_route'] == 'persona_specification'
    assert second['pipeline']['route']['selected_route'] == 'persona_specification'
    assert second['persona_action']['created'] is True
    assert second['persona_action']['persona_name'] == 'Катерина'


def test_persona_assignment_uses_validated_registry_after_creation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr(
        'agent_system.persona_engine.call_json_model_for_role',
        lambda *args, **kwargs: 'Narrative prose instead of JSON.',
    )

    created = create_persona_from_description(
        'Осторожный и гордый человек, который не любит быть зависимым от чужой воли.',
        name_hint='Guarded Pride',
    )
    assert created['created'] is True

    result = generate_response(
        message='сделай текущей личностью',
        session_id='persona_assign_session',
        selected_persona='Guarded Pride',
        language='ru',
    )

    assert result['pipeline']['route']['selected_route'] == 'persona_assignment'
    assert result['persona_action']['assigned'] is True
    route_state = load_session_route_state('persona_assign_session')
    assert route_state['persona_name'] == 'Guarded Pride'


def test_personality_endpoint_lists_only_validated_entries(tmp_path, monkeypatch) -> None:
    fastapi = pytest.importorskip('fastapi')
    testclient = pytest.importorskip('fastapi.testclient')
    assert fastapi

    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    materialize_persona('Dr. Aram Petrosyan', _valid_persona_payload(), explicit=True)
    spawn_head('PDF', entity_type='PERSON', source='test', register=False)
    write_json(
        personality_index_path(),
        {
            'heads': [
                normalize_personality_name('Dr. Aram Petrosyan'),
                normalize_personality_name('PDF'),
            ]
        },
    )

    from agent_system.api import create_app

    client = testclient.TestClient(create_app())
    response = client.get('/api/cognitive/personalities')

    assert response.status_code == 200
    payload = response.json()
    assert [item['slug'] for item in payload['personalities']] == ['dr_aram_petrosyan']
    assert payload['personalities'][0]['hover_text']
    assert payload['personalities'][0]['validation_status'] in {'valid', 'partial'}
    assert load_json(personality_index_path(), {'heads': []})['heads'] == ['dr_aram_petrosyan']


@pytest.mark.parametrize('bad_name', ['Unity: Engine', 'Game Engine', 'ты питаешься', '/мой коммент/'])
def test_prompt_fragments_and_ontology_labels_are_rejected(tmp_path, monkeypatch, bad_name: str) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    with pytest.raises(MutationRejectedFailure):
        materialize_persona(bad_name, _valid_persona_payload(), explicit=True)

    assert load_active_persona(bad_name) is None


def test_personality_detail_exposes_structured_persona_object(tmp_path, monkeypatch) -> None:
    fastapi = pytest.importorskip('fastapi')
    testclient = pytest.importorskip('fastapi.testclient')
    assert fastapi

    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    materialize_persona('Dr. Aram Petrosyan', _valid_persona_payload(), explicit=True)

    from agent_system.api import create_app

    client = testclient.TestClient(create_app())
    response = client.get('/api/cognitive/personalities/Dr.%20Aram%20Petrosyan')

    assert response.status_code == 200
    payload = response.json()
    assert payload['structured_persona']['identity']['label'] == 'Guarded Pride'
    assert payload['structured_persona']['meta']['hover_text']
    assert payload['short_description']


def test_materialized_persona_persists_structured_schema(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    materialize_persona('Dr. Aram Petrosyan', _valid_persona_payload(), explicit=True)
    bundle = load_persona('Dr. Aram Petrosyan')

    assert bundle is not None
    assert bundle.structured_persona is not None
    assert bundle.structured_persona.identity.label == 'Guarded Pride'
    assert bundle.structured_persona.meta.validation_status == 'valid'
