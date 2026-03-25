from __future__ import annotations

import random
from dataclasses import dataclass

from .models import ClassificationDecision, ENTITY_TYPES, EntityFeatures, SITUATION_TYPES, SituationDecision, SituationFeatures


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


@dataclass(slots=True)
class SituationDecisionTree:
    seed: int
    feature_names: tuple[str, ...]

    def vote(self, features: SituationFeatures) -> str:
        data = features.feature_map
        rng = random.Random(self.seed)
        weights = {situation_type: 0.0 for situation_type in SITUATION_TYPES}

        if data.get('moral_violation_signal', 0.0) >= 1.0 and data.get('celebratory_signal', 0.0) >= 1.0:
            weights['abnormal_behavior'] += 2.9
        if data.get('insult_signal', 0.0) >= 1.0 and data.get('persona_target_signal', 0.0) >= 1.0:
            weights['insult'] += 2.7
        if data.get('distress_signal', 0.0) >= 1.0:
            weights['user_distress'] += 2.5
        if data.get('anger_signal', 0.0) >= 1.0:
            weights['user_anger'] += 2.1
        if data.get('question_signal', 0.0) >= 1.0 and data.get('insult_signal', 0.0) <= 0.0:
            weights['neutral_query'] += 1.9
        if all(value <= 0.0 for value in data.values()):
            weights['neutral_statement'] += 1.5

        for feature_name in self.feature_names:
            value = data.get(feature_name, 0.0)
            if value <= 0.0:
                continue
            noise = 1.0 + rng.random() * 0.1
            if feature_name == 'insult_signal':
                weights['insult'] += value * noise
            elif feature_name == 'distress_signal':
                weights['user_distress'] += value * noise
            elif feature_name == 'moral_violation_signal':
                weights['abnormal_behavior'] += value * noise
            elif feature_name == 'celebratory_signal':
                weights['abnormal_behavior'] += value * (0.8 + rng.random() * 0.1)
            elif feature_name == 'anger_signal':
                weights['user_anger'] += value * noise
            elif feature_name == 'question_signal':
                weights['neutral_query'] += value * noise
            elif feature_name.endswith('target_signal') and value > 0.0:
                weights['neutral_statement'] += value * (0.2 + rng.random() * 0.05)

        return max(weights.items(), key=lambda item: (item[1], item[0]))[0]


class RandomForestSituationClassifier:
    def __init__(self, tree_count: int = 5) -> None:
        available = (
            'insult_signal',
            'anger_signal',
            'distress_signal',
            'question_signal',
            'help_signal',
            'moral_violation_signal',
            'celebratory_signal',
            'persona_target_signal',
            'self_target_signal',
            'external_target_signal',
        )
        self.trees = [
            SituationDecisionTree(seed=index + 101, feature_names=tuple(random.Random(index + 101).sample(available, k=5)))
            for index in range(max(tree_count, 3))
        ]

    def classify(self, features: SituationFeatures) -> SituationDecision:
        votes = {situation_type: 0 for situation_type in SITUATION_TYPES}
        for tree in self.trees:
            votes[tree.vote(features)] += 1
        winner = max(votes.items(), key=lambda item: (item[1], item[0]))[0]
        confidence = round(votes[winner] / max(len(self.trees), 1), 4)
        return SituationDecision(
            situation_type=winner,
            votes=votes,
            confidence=confidence,
            features=dict(features.feature_map),
            evidence=list(features.evidence),
        )


DEFAULT_CLASSIFIER = RandomForestEntityClassifier()
DEFAULT_SITUATION_CLASSIFIER = RandomForestSituationClassifier()
