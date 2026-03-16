from __future__ import annotations

import random
from dataclasses import dataclass

from .models import ClassificationDecision, ENTITY_TYPES, EntityFeatures


@dataclass(slots=True)
class DecisionTree:
    seed: int
    feature_names: tuple[str, ...]

    def vote(self, features: EntityFeatures) -> str:
        data = features.feature_map
        rng = random.Random(self.seed)
        weights = {entity_type: 0.0 for entity_type in ENTITY_TYPES}

        if data.get('profession_hint_score', 0.0) >= 1.0:
            weights['PROFESSION'] += 2.4
        if data.get('fictional_hint_score', 0.0) >= 1.0:
            weights['FICTIONAL_CHARACTER'] += 2.3
        if data.get('phenomenon_hint_score', 0.0) >= 1.0:
            weights['PHENOMENON'] += 2.2
        if data.get('concept_hint_score', 0.0) >= 1.0:
            weights['CONCEPT'] += 2.0
        if data.get('object_hint_score', 0.0) >= 1.0:
            weights['OBJECT'] += 2.0
        if data.get('person_hint_score', 0.0) >= 1.0:
            weights['PERSON'] += 1.9

        if data.get('title_case_ratio', 0.0) >= 0.75 and data.get('fictional_hint_score', 0.0) <= 0.0:
            weights['PERSON'] += 1.2
        if data.get('title_case_ratio', 0.0) >= 0.75 and data.get('fictional_hint_score', 0.0) >= 1.0:
            weights['FICTIONAL_CHARACTER'] += 1.3
        if data.get('contains_role_suffix', 0.0) >= 1.0 and data.get('profession_hint_score', 0.0) >= 0.5:
            weights['PROFESSION'] += 1.1
        if data.get('is_single_token', 0.0) >= 1.0 and data.get('concept_hint_score', 0.0) >= 1.0:
            weights['CONCEPT'] += 0.8
        if data.get('contains_of_title', 0.0) >= 1.0 and data.get('fictional_hint_score', 0.0) >= 0.5:
            weights['FICTIONAL_CHARACTER'] += 0.7

        for feature_name in self.feature_names:
            value = data.get(feature_name, 0.0)
            if value <= 0:
                continue
            if feature_name.endswith('person_hint_score'):
                weights['PERSON'] += value * (1.0 + rng.random() * 0.15)
            elif feature_name.endswith('fictional_hint_score'):
                weights['FICTIONAL_CHARACTER'] += value * (1.0 + rng.random() * 0.15)
            elif feature_name.endswith('profession_hint_score'):
                weights['PROFESSION'] += value * (1.0 + rng.random() * 0.15)
            elif feature_name.endswith('concept_hint_score'):
                weights['CONCEPT'] += value * (1.0 + rng.random() * 0.15)
            elif feature_name.endswith('phenomenon_hint_score'):
                weights['PHENOMENON'] += value * (1.0 + rng.random() * 0.15)
            elif feature_name.endswith('object_hint_score'):
                weights['OBJECT'] += value * (1.0 + rng.random() * 0.15)
        winner = max(weights.items(), key=lambda item: (item[1], item[0]))[0]
        return winner


class RandomForestEntityClassifier:
    def __init__(self, tree_count: int = 7) -> None:
        available = (
            'person_hint_score',
            'fictional_hint_score',
            'profession_hint_score',
            'concept_hint_score',
            'phenomenon_hint_score',
            'object_hint_score',
            'title_case_ratio',
            'contains_role_suffix',
            'contains_of_title',
            'is_single_token',
        )
        self.trees = [
            DecisionTree(seed=index + 11, feature_names=tuple(random.Random(index + 11).sample(available, k=5)))
            for index in range(max(tree_count, 3))
        ]

    def classify(self, features: EntityFeatures) -> ClassificationDecision:
        votes = {entity_type: 0 for entity_type in ENTITY_TYPES}
        for tree in self.trees:
            votes[tree.vote(features)] += 1
        winner = max(votes.items(), key=lambda item: (item[1], item[0]))[0]
        confidence = round(votes[winner] / max(len(self.trees), 1), 4)
        return ClassificationDecision(
            entity_name=features.entity_name,
            entity_type=winner,
            votes=votes,
            confidence=confidence,
            features=dict(features.feature_map),
            evidence=list(features.evidence),
        )


DEFAULT_CLASSIFIER = RandomForestEntityClassifier()
