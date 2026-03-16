from __future__ import annotations

import json

from agent_system.classifier_forest import DEFAULT_CLASSIFIER
from agent_system.feature_extractor import extract_features
from agent_system.message_analyzer import analyze_message
from agent_system.models import MessageEntity
from agent_system.persona_engine import load_persona


def test_classifier_forest_routes_dracula_to_fictional_character() -> None:
    analysis = analyze_message(message='Talk about Dracula the fictional vampire nobleman.', session_id='classifier')
    features = extract_features(
        MessageEntity(name='Dracula', description='Fictional vampire nobleman and literary character.'),
        analysis,
    )
    decision = DEFAULT_CLASSIFIER.classify(features)

    assert decision.entity_type == 'FICTIONAL_CHARACTER'
    assert decision.votes['FICTIONAL_CHARACTER'] >= 4


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
    for filename in ('traits.json', 'relations.json', 'examples.json', 'knowledge.txt', 'emotion_vector.json', 'meta.json'):
        assert (head_dir / filename).exists()
    meta = json.loads((head_dir / 'meta.json').read_text(encoding='utf-8'))
    assert meta['entity_type'] == 'FICTIONAL_CHARACTER'
    assert bundle.emotion_vector['confidence'] == 0.9
