from __future__ import annotations

import json

from agent_system.classifier_forest import DEFAULT_CLASSIFIER
from agent_system.feature_extractor import extract_features
from agent_system.graph_store import GraphStore
from agent_system.head_caller import prepare_heads
from agent_system.message_analyzer import analyze_message
from agent_system.models import MessageAnalysis, MessageEntity, UserState
from agent_system.persona_engine import (
    evolve_emotion_state,
    formalize_persona,
    load_persona,
    reaction_policy,
    restore_persona_revision,
    update_persona_from_examples,
)


def test_classifier_forest_routes_dracula_to_fictional_character() -> None:
    analysis = analyze_message(message='Talk about Dracula the fictional vampire nobleman.', session_id='classifier')
    features = extract_features(
        MessageEntity(name='Dracula', description='Fictional vampire nobleman and literary character.'),
        analysis,
    )
    decision = DEFAULT_CLASSIFIER.classify(features)

    assert decision.entity_type == 'FICTIONAL_CHARACTER'
    assert decision.votes['FICTIONAL_CHARACTER'] >= 4


def test_message_analyzer_does_not_promote_day_to_entity() -> None:
    analysis = analyze_message(
        message='Dr. Aram Petrosyan, what do you do day to day?',
        session_id='classifier',
        selected_head='Dr. Aram Petrosyan',
    )

    entity_names = [entity.name for entity in analysis.entities]
    assert any('Aram Petrosyan' in item for item in entity_names)
    assert 'day' not in [item.lower() for item in entity_names]


def test_materialized_head_uses_folder_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    from agent_system.persona_engine import materialize_persona

    bundle = materialize_persona(
        'Dracula',
        {
            'entity_type': 'FICTIONAL_CHARACTER',
            'aliases': ['Count Dracula'],
            'traits': ['vampire', 'aristocratic'],
            'examples': ['I prefer the night.'],
            'relations': [{'type': 'FEEDS_ON', 'target': 'humans'}],
            'emotion_vector': {'confidence': 0.9, 'curiosity': 0.4, 'empathy': 0.2},
            'knowledge': 'Dracula is an immortal vampire nobleman.',
        },
    )

    assert load_persona('dracula') is not None
    head_dir = tmp_path / 'memory' / 'heads' / 'dracula'
    for filename in (
        'traits.json',
        'relations.json',
        'examples.json',
        'knowledge.txt',
        'emotion_vector.json',
        'meta.json',
        'baseline.json',
        'dynamic_state.json',
        'learned_patterns.json',
        'revisions.json',
    ):
        assert (head_dir / filename).exists()
    meta = json.loads((head_dir / 'meta.json').read_text(encoding='utf-8'))
    assert meta['entity_type'] == 'FICTIONAL_CHARACTER'
    assert meta['schema_version'] == 2
    assert meta['revision'] >= 2
    assert bundle.emotion_vector['confidence'] == 0.9
    assert bundle.baseline_definition is not None
    assert bundle.dynamic_state is not None
    assert bundle.learned_patterns is not None
    assert bundle.indicators is not None


def test_persona_is_formalized_as_stateful_model_and_reaction_policy(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    from agent_system.persona_engine import materialize_persona

    bundle = materialize_persona(
        'Dracula',
        {
            'entity_type': 'FICTIONAL_CHARACTER',
            'aliases': ['Count Dracula'],
            'traits': ['vampire', 'aggressive', 'aristocratic'],
            'examples': ['I prefer the night.'],
            'relations': [{'type': 'FEEDS_ON', 'target': 'humans'}],
            'emotion_vector': {'anger': 0.2, 'fear': 0.1, 'curiosity': 0.4, 'confidence': 0.9, 'empathy': 0.2},
            'knowledge': 'Dracula is an immortal vampire nobleman.',
        },
        explicit=True,
    )

    model = formalize_persona(bundle)
    assert sorted(model.T.keys()) == ['entity_type', 'parameters', 'traits']
    assert sorted(model.E.keys()) == ['anger', 'confidence', 'curiosity', 'empathy', 'fear']
    assert model.R == 'deterministic_situation_reaction_policy'
    assert 'examples' in model.M and 'relations' in model.M and 'situation_reactions' in model.M
    assert 'log_tuples' in model.M and 'persona_form' in model.M and 'decision_explanation' in model.M

    outcome = reaction_policy(bundle, {'type': 'insult', 'target': 'persona', 'severity': 0.8})
    assert outcome.situation_type == 'insult'
    assert outcome.target == 'persona'
    assert outcome.delta_emotion['anger'] > 0.0
    assert outcome.response_style in {'defensive', 'firm_boundary'}

    neutral_outcome = reaction_policy(bundle, {'type': 'neutral_query', 'target': 'persona', 'severity': 0.4})
    assert neutral_outcome.response_style in {'direct_explanatory', 'inquisitive', 'formal'}

    evolved, _ = evolve_emotion_state(bundle, {'type': 'neutral_query', 'target': 'persona', 'severity': 0.4})
    assert all(0.0 <= value <= 1.0 for value in evolved.values())


def test_update_persona_from_examples_builds_persona_triad(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    def fake_model(prompt: str, mode: str = 'chat', *, role: str = 'general') -> str:
        assert role == 'analyst'
        return json.dumps(
            {
                'persona_payload': {
                    'name': 'Greg House',
                    'entity_type': 'PERSON',
                    'traits': ['logical', 'sarcastic'],
                    'examples': ['Answer directly.', 'Use sarcasm lightly.'],
                    'knowledge': 'Greg House is a human persona who prefers diagnostic reasoning and controlled sarcasm.',
                },
                'persona_form': {
                    'identity_class': 'human',
                    'interaction_style': ['analytical', 'dry'],
                    'core_dispositions': ['logical', 'sarcastic'],
                    'decision_patterns': ['checks factual consistency before answering'],
                    'clarification_policy': 'Ask a clarifying question when the target or evidence is weak.',
                    'sarcasm_profile': 'medium',
                    'response_priorities': ['answer_substance', 'clarify_if_underspecified'],
                    'knowledge_domains': ['diagnostics'],
                    'risk_controls': ['do_not_mirror_user_emotion'],
                },
                'decision_explanation': 'Greg House first checks the case structure, then answers directly and may use sarcasm if it does not block the substance of the reply.',
            }
        )

    monkeypatch.setattr('agent_system.llm._call_model', fake_model)

    bundle = update_persona_from_examples(
        'Greg House',
        ['Answer directly.', 'Answer directly.', 'Use sarcasm lightly.'],
    )

    assert bundle.persona_form['identity_class'] == 'human'
    assert bundle.persona_form['sarcasm_profile'] == 'medium'
    assert 'first checks the case structure' in bundle.decision_explanation
    assert any(tuple(item.get('tuple') or ())[:1] == ('utterance_pattern',) and int(item.get('frequency') or 0) >= 2 for item in bundle.log_tuples)
    assert bundle.learned_patterns is not None
    assert bundle.learned_patterns.decision_explanation.startswith('Greg House first checks')
    assert bundle.indicators is not None
    assert bundle.indicators.evidence_count >= 3
    assert bundle.revision_meta['learned_revision'] >= 2
    head_dir = tmp_path / 'memory' / 'heads' / 'greg_house'
    assert (head_dir / 'log_tuples.json').exists()
    assert (head_dir / 'persona_form.json').exists()
    assert (head_dir / 'decision_explanation.txt').exists()


def test_learned_updates_do_not_drift_persona_baseline(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    from agent_system.persona_engine import materialize_persona

    materialize_persona(
        'Stable Persona',
        {
            'entity_type': 'PERSON',
            'traits': ['logical', 'measured'],
            'examples': ['Answer with structure.'],
            'knowledge': 'Stable Persona is a measured human persona.',
        },
        explicit=True,
    )

    def fake_model(prompt: str, mode: str = 'chat', *, role: str = 'general') -> str:
        assert role == 'analyst'
        return json.dumps(
            {
                'persona_payload': {
                    'name': 'Stable Persona',
                    'entity_type': 'PERSON',
                    'traits': ['chaotic', 'impulsive'],
                    'examples': ['Shout nonsense.'],
                    'knowledge': 'A wildly unstable persona.',
                },
                'persona_form': {
                    'identity_class': 'human',
                    'interaction_style': ['direct'],
                    'core_dispositions': ['logical', 'measured'],
                    'decision_patterns': ['checks context before answering'],
                    'clarification_policy': 'Clarify when context is weak.',
                    'sarcasm_profile': 'none',
                    'response_priorities': ['answer_substance'],
                    'knowledge_domains': ['reasoning'],
                    'risk_controls': ['do_not_mirror_user_emotion'],
                },
                'decision_explanation': 'Stable Persona checks context first, then answers directly.',
            }
        )

    monkeypatch.setattr('agent_system.llm._call_model', fake_model)

    updated = update_persona_from_examples(
        'Stable Persona',
        ['Random chaotic anecdote.', 'Another unstable fragment.'],
    )

    assert updated.baseline_definition is not None
    assert updated.learned_patterns is not None
    assert updated.baseline_definition.traits == ['logical', 'measured']
    assert updated.baseline_definition.knowledge == 'Stable Persona is a measured human persona.'
    assert 'chaotic' in updated.learned_patterns.learned_traits
    assert updated.revision_meta['baseline_revision'] == 2
    assert updated.revision_meta['learned_revision'] > updated.revision_meta['baseline_revision']


def test_persona_revision_restore_recovers_previous_learned_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    from agent_system.persona_engine import materialize_persona

    initial = materialize_persona(
        'Restore Persona',
        {
            'entity_type': 'PERSON',
            'traits': ['logical'],
            'examples': ['Answer clearly.'],
            'knowledge': 'Restore Persona is a precise human persona.',
        },
        explicit=True,
    )
    initial_revision = int(initial.revision_meta['revision'])

    def fake_model(prompt: str, mode: str = 'chat', *, role: str = 'general') -> str:
        return json.dumps(
            {
                'persona_payload': {
                    'name': 'Restore Persona',
                    'entity_type': 'PERSON',
                    'traits': ['sarcastic'],
                    'examples': ['Use dry wit.'],
                },
                'persona_form': {
                    'identity_class': 'human',
                    'interaction_style': ['direct'],
                    'core_dispositions': ['logical'],
                    'decision_patterns': ['answers directly'],
                    'clarification_policy': 'Clarify when needed.',
                    'sarcasm_profile': 'medium',
                    'response_priorities': ['answer_substance'],
                    'knowledge_domains': ['reasoning'],
                    'risk_controls': ['do_not_mirror_user_emotion'],
                },
                'decision_explanation': 'Restore Persona answers directly and may use dry wit.',
            }
        )

    monkeypatch.setattr('agent_system.llm._call_model', fake_model)
    updated = update_persona_from_examples('Restore Persona', ['Use dry wit.'])
    assert updated.learned_patterns is not None
    assert 'sarcastic' in updated.learned_patterns.learned_traits

    restored = restore_persona_revision('Restore Persona', initial_revision)
    assert restored is not None
    assert restored.learned_patterns is not None
    assert 'sarcastic' not in restored.learned_patterns.learned_traits


def test_prepare_heads_does_not_materialize_non_head_noise_entities(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    analysis = MessageAnalysis(
        message='What do you do day to day?',
        session_id='head_noise',
        selected_head='Dr. Aram Petrosyan',
        primary_entity='Dr. Aram Petrosyan',
        current_entity='',
        explicit_context='',
        entities=[MessageEntity(name='Dr. Aram Petrosyan'), MessageEntity(name='day')],
        user_state=UserState(language='en', tone='inquisitive', intent='question', signals={}),
    )
    dr_features = extract_features(MessageEntity(name='Dr. Aram Petrosyan', description='Emergency physician.'), analysis)
    day_features = extract_features(MessageEntity(name='day', description='Time period.'), analysis)
    decisions = [
        DEFAULT_CLASSIFIER.classify(dr_features),
        DEFAULT_CLASSIFIER.classify(day_features),
    ]
    store = GraphStore()

    prepared = prepare_heads(analysis=analysis, classifications=decisions, graph_store=store)

    assert any(item.get('head') is not None for item in prepared if item['decision'].entity_name == 'Dr. Aram Petrosyan')
    assert not any(item.get('head') is not None for item in prepared if item['decision'].entity_name == 'day')
    assert store.get_node('day') is None
