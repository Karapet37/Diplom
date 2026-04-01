from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ENTITY_TYPES = (
    'PERSON',
    'CONCEPT',
    'PHENOMENON',
    'OBJECT',
    'FICTIONAL_CHARACTER',
    'PROFESSION',
)

GRAPH_NODE_LIFECYCLE_STATES = (
    'active',
    'weak',
    'suspect',
    'archived',
    'merged',
)

HEAD_ENTITY_TYPES = (
    'PERSON',
    'FICTIONAL_CHARACTER',
    'PROFESSION',
)

EMOTION_KEYS = ('anger', 'fear', 'curiosity', 'confidence', 'empathy')
SITUATION_TYPES = (
    'insult',
    'user_distress',
    'abnormal_behavior',
    'user_anger',
    'neutral_query',
    'neutral_statement',
)


@dataclass(slots=True)
class UserState:
    language: str = 'en'
    tone: str = 'neutral'
    intent: str = 'statement'
    signals: dict[str, float] = field(default_factory=dict)

    def signal(self, name: str, default: float = 0.0) -> float:
        try:
            return float(self.signals.get(name, default))
        except (TypeError, ValueError):
            return float(default)

    def to_dict(self) -> dict[str, Any]:
        return {
            'language': self.language,
            'tone': self.tone,
            'intent': self.intent,
            'signals': dict(self.signals),
        }


@dataclass(slots=True)
class Situation:
    type: str = 'neutral_statement'
    target: str = 'external'
    severity: float = 0.0
    classifier_hint: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    summary: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'type': self.type,
            'target': self.target,
            'severity': self.severity,
            'classifier_hint': dict(self.classifier_hint),
            'evidence': list(self.evidence),
            'summary': self.summary,
        }


@dataclass(slots=True)
class MessageEntity:
    name: str
    source_text: str = ''
    description: str = ''
    aliases: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MessageAnalysis:
    message: str
    session_id: str
    selected_head: str = ''
    primary_entity: str = ''
    current_entity: str = ''
    explicit_context: str = ''
    entities: list[MessageEntity] = field(default_factory=list)
    user_state: UserState = field(default_factory=UserState)
    situation: Situation = field(default_factory=Situation)

    @property
    def cues(self) -> dict[str, Any]:
        return dict(self.user_state.signals)

    def to_dict(self) -> dict[str, Any]:
        return {
            'primary_entity': self.primary_entity,
            'user_state': self.user_state.to_dict(),
            'situation': self.situation.to_dict(),
            'entities': [entity.name for entity in self.entities],
        }


@dataclass(slots=True)
class EntityFeatures:
    entity_name: str
    normalized_name: str
    description: str
    feature_map: dict[str, float]
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ClassificationDecision:
    entity_name: str
    entity_type: str
    votes: dict[str, int]
    confidence: float
    features: dict[str, float]
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SituationFeatures:
    feature_map: dict[str, float]
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SituationDecision:
    situation_type: str
    votes: dict[str, int]
    confidence: float
    features: dict[str, float]
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReactionOutcome:
    delta_emotion: dict[str, float]
    response_style: str
    situation_type: str
    target: str
    severity: float


@dataclass(slots=True)
class PersonaBaselineDefinition:
    name: str
    slug: str
    entity_type: str
    traits: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    knowledge: str = ''
    revision: int = 0
    updated_at: str = ''
    source: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'slug': self.slug,
            'entity_type': self.entity_type,
            'traits': list(self.traits),
            'aliases': list(self.aliases),
            'relations': [dict(item) for item in self.relations],
            'knowledge': self.knowledge,
            'revision': int(self.revision or 0),
            'updated_at': self.updated_at,
            'source': self.source,
        }


@dataclass(slots=True)
class PersonaDynamicState:
    emotion_vector: dict[str, float] = field(default_factory=dict)
    last_situation: str = ''
    last_response_style: str = ''
    revision: int = 0
    updated_at: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'emotion_vector': dict(self.emotion_vector),
            'last_situation': self.last_situation,
            'last_response_style': self.last_response_style,
            'revision': int(self.revision or 0),
            'updated_at': self.updated_at,
        }


@dataclass(slots=True)
class PersonaLearnedPatterns:
    examples: list[str] = field(default_factory=list)
    situation_reactions: list[dict[str, Any]] = field(default_factory=list)
    log_tuples: list[dict[str, Any]] = field(default_factory=list)
    persona_form: dict[str, Any] = field(default_factory=dict)
    decision_explanation: str = ''
    learned_traits: list[str] = field(default_factory=list)
    revision: int = 0
    updated_at: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'examples': list(self.examples),
            'situation_reactions': [dict(item) for item in self.situation_reactions],
            'log_tuples': [dict(item) for item in self.log_tuples],
            'persona_form': dict(self.persona_form),
            'decision_explanation': self.decision_explanation,
            'learned_traits': list(self.learned_traits),
            'revision': int(self.revision or 0),
            'updated_at': self.updated_at,
        }


@dataclass(slots=True)
class PersonaIndicators:
    confidence_score: float = 0.0
    maturity_score: float = 0.0
    maturity_level: str = 'bootstrap'
    evidence_count: int = 0
    learned_pattern_count: int = 0
    adaptation_locked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            'confidence_score': round(float(self.confidence_score or 0.0), 6),
            'maturity_score': round(float(self.maturity_score or 0.0), 6),
            'maturity_level': self.maturity_level,
            'evidence_count': int(self.evidence_count or 0),
            'learned_pattern_count': int(self.learned_pattern_count or 0),
            'adaptation_locked': bool(self.adaptation_locked),
        }


@dataclass(slots=True)
class PersonaSelectionExplanation:
    persona_name: str = ''
    source: str = ''
    reason: str = ''
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'persona_name': self.persona_name,
            'source': self.source,
            'reason': self.reason,
            'evidence': list(self.evidence),
        }


@dataclass(slots=True)
class PersonaResponseExplanation:
    persona_name: str = ''
    response_style: str = ''
    reason: str = ''
    situation_summary: str = ''
    state_influences: list[str] = field(default_factory=list)
    trait_influences: list[str] = field(default_factory=list)
    learned_influences: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'persona_name': self.persona_name,
            'response_style': self.response_style,
            'reason': self.reason,
            'situation_summary': self.situation_summary,
            'state_influences': list(self.state_influences),
            'trait_influences': list(self.trait_influences),
            'learned_influences': list(self.learned_influences),
        }


@dataclass(slots=True)
class SocialRoleDecision:
    role: str = 'ally'
    confidence: float = 0.0
    reason: str = ''
    evidence: list[str] = field(default_factory=list)
    graph_anchors: list[str] = field(default_factory=list)
    mood_signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'role': self.role,
            'confidence': round(float(self.confidence or 0.0), 6),
            'reason': self.reason,
            'evidence': list(self.evidence),
            'graph_anchors': list(self.graph_anchors),
            'mood_signals': {
                str(key): round(float(value or 0.0), 6)
                for key, value in dict(self.mood_signals or {}).items()
            },
        }


@dataclass(slots=True)
class BehavioralFallbackDecision:
    strategy: str = 'honest_limitation'
    confidence: float = 0.0
    uncertainty_level: str = 'moderate'
    risk_level: str = 'low'
    style_mode: str = 'grounded'
    reason: str = ''
    evidence: list[str] = field(default_factory=list)
    source_grounding: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'strategy': self.strategy,
            'confidence': round(float(self.confidence or 0.0), 6),
            'uncertainty_level': self.uncertainty_level,
            'risk_level': self.risk_level,
            'style_mode': self.style_mode,
            'reason': self.reason,
            'evidence': list(self.evidence),
            'source_grounding': self.source_grounding,
        }


@dataclass(slots=True)
class MoodFeatureSnapshot:
    snapshot_id: str
    session_id: str
    persona_name: str
    source: str
    language: str
    user_features: dict[str, float] = field(default_factory=dict)
    persona_features: dict[str, float] = field(default_factory=dict)
    combined_features: dict[str, float] = field(default_factory=dict)
    selected_role: str = ''
    response_style: str = ''
    situation_type: str = ''
    summary: str = ''
    created_at: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'snapshot_id': self.snapshot_id,
            'session_id': self.session_id,
            'persona_name': self.persona_name,
            'source': self.source,
            'language': self.language,
            'user_features': {
                str(key): round(float(value or 0.0), 6)
                for key, value in dict(self.user_features or {}).items()
            },
            'persona_features': {
                str(key): round(float(value or 0.0), 6)
                for key, value in dict(self.persona_features or {}).items()
            },
            'combined_features': {
                str(key): round(float(value or 0.0), 6)
                for key, value in dict(self.combined_features or {}).items()
            },
            'selected_role': self.selected_role,
            'response_style': self.response_style,
            'situation_type': self.situation_type,
            'summary': self.summary,
            'created_at': self.created_at,
        }


@dataclass(slots=True)
class MoodCluster:
    cluster_id: str
    label: str
    centroid: dict[str, float] = field(default_factory=dict)
    feature_peaks: list[str] = field(default_factory=list)
    size: int = 0
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'cluster_id': self.cluster_id,
            'label': self.label,
            'centroid': {
                str(key): round(float(value or 0.0), 6)
                for key, value in dict(self.centroid or {}).items()
            },
            'feature_peaks': list(self.feature_peaks),
            'size': int(self.size or 0),
            'examples': list(self.examples),
        }


@dataclass(slots=True)
class MoodResearchReport:
    scope: str
    snapshot_count: int = 0
    latest_language: str = 'en'
    latest_cluster_id: str = ''
    latest_cluster_label: str = ''
    clusters: list[MoodCluster] = field(default_factory=list)
    transition_counts: list[dict[str, Any]] = field(default_factory=list)
    classification: dict[str, Any] = field(default_factory=dict)
    regressions: list[dict[str, Any]] = field(default_factory=list)
    role_effects: list[dict[str, Any]] = field(default_factory=list)
    updated_at: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'scope': self.scope,
            'snapshot_count': int(self.snapshot_count or 0),
            'latest_language': self.latest_language,
            'latest_cluster_id': self.latest_cluster_id,
            'latest_cluster_label': self.latest_cluster_label,
            'clusters': [item.to_dict() for item in list(self.clusters or [])],
            'transition_counts': [dict(item) for item in list(self.transition_counts or [])],
            'classification': dict(self.classification or {}),
            'regressions': [dict(item) for item in list(self.regressions or [])],
            'role_effects': [dict(item) for item in list(self.role_effects or [])],
            'updated_at': self.updated_at,
        }


@dataclass(slots=True)
class PersonaGraphExplanation:
    persona_name: str
    summary: str
    central_nodes: list[str] = field(default_factory=list)
    peripheral_nodes: list[str] = field(default_factory=list)
    conflict_nodes: list[str] = field(default_factory=list)
    causal_links: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'persona_name': self.persona_name,
            'summary': self.summary,
            'central_nodes': list(self.central_nodes),
            'peripheral_nodes': list(self.peripheral_nodes),
            'conflict_nodes': list(self.conflict_nodes),
            'causal_links': list(self.causal_links),
        }


@dataclass(slots=True)
class PersonaSystemModel:
    T: dict[str, Any]
    E: dict[str, float]
    R: str
    M: dict[str, Any]


@dataclass(slots=True)
class GraphQuality:
    relevance: float
    redundancy: float
    connectivity: float
    score: float


@dataclass(slots=True)
class GraphHealthMetrics:
    node_count: int = 0
    edge_count: int = 0
    active_node_count: int = 0
    weak_node_count: int = 0
    suspect_node_count: int = 0
    archived_node_count: int = 0
    merged_node_count: int = 0
    duplicate_candidates: int = 0
    duplicate_review_candidates: int = 0
    duplicate_rate: float = 0.0
    orphan_nodes: int = 0
    orphan_rate: float = 0.0
    average_relation_density: float = 0.0
    low_value_nodes: int = 0
    summary_nodes: int = 0
    cluster_count: int = 0
    quality: GraphQuality | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'node_count': int(self.node_count or 0),
            'edge_count': int(self.edge_count or 0),
            'active_node_count': int(self.active_node_count or 0),
            'weak_node_count': int(self.weak_node_count or 0),
            'suspect_node_count': int(self.suspect_node_count or 0),
            'archived_node_count': int(self.archived_node_count or 0),
            'merged_node_count': int(self.merged_node_count or 0),
            'duplicate_candidates': int(self.duplicate_candidates or 0),
            'duplicate_review_candidates': int(self.duplicate_review_candidates or 0),
            'duplicate_rate': round(float(self.duplicate_rate or 0.0), 6),
            'orphan_nodes': int(self.orphan_nodes or 0),
            'orphan_rate': round(float(self.orphan_rate or 0.0), 6),
            'average_relation_density': round(float(self.average_relation_density or 0.0), 6),
            'low_value_nodes': int(self.low_value_nodes or 0),
            'summary_nodes': int(self.summary_nodes or 0),
            'cluster_count': int(self.cluster_count or 0),
            'quality': {
                'relevance': round(float((self.quality.relevance if self.quality else 0.0) or 0.0), 6),
                'redundancy': round(float((self.quality.redundancy if self.quality else 0.0) or 0.0), 6),
                'connectivity': round(float((self.quality.connectivity if self.quality else 0.0) or 0.0), 6),
                'score': round(float((self.quality.score if self.quality else 0.0) or 0.0), 6),
            },
        }


@dataclass(slots=True)
class GraphModelToolPolicy:
    canonical_language: str = 'en'
    context_budget: int = 4000
    max_new_entities: int = 4
    max_new_relations: int = 8
    max_link_suggestions: int = 4
    allowed_link_roles: tuple[str, ...] = (
        'related_to',
        'has_dimension',
        'supports',
        'makes_possible',
        'depends_on',
        'part_of',
        'influences',
        'warms',
        'regulates',
        'is_a',
        'symbolizes',
        'affects',
    )
    allow_reconstruction: bool = True
    allow_translation: bool = True
    reconstruction_role: str = 'analyst'
    translation_role: str = 'translator'


@dataclass(slots=True)
class HeadBundle:
    name: str
    folder: str
    entity_type: str
    traits: list[str]
    relations: list[dict[str, Any]]
    examples: list[str]
    situation_reactions: list[dict[str, Any]]
    knowledge: str
    emotion_vector: dict[str, float]
    meta: dict[str, Any]
    log_tuples: list[dict[str, Any]] = field(default_factory=list)
    persona_form: dict[str, Any] = field(default_factory=dict)
    decision_explanation: str = ''
    baseline_definition: PersonaBaselineDefinition | None = None
    dynamic_state: PersonaDynamicState | None = None
    learned_patterns: PersonaLearnedPatterns | None = None
    indicators: PersonaIndicators | None = None
    revision_meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ContextPayload:
    persona_name: str
    persona_block: str
    graph_context: str
    recent_dialogue: str
    current_entity: str
    selected_nodes: list[dict[str, Any]]
    selected_edges: list[dict[str, Any]]
    estimated_tokens: int
    context_debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'persona_name': self.persona_name,
            'persona_block': self.persona_block,
            'graph_context': self.graph_context,
            'recent_dialogue': self.recent_dialogue,
            'current_entity': self.current_entity,
            'nodes': list(self.selected_nodes),
            'edges': list(self.selected_edges),
            'estimated_tokens': self.estimated_tokens,
            'context_debug': dict(self.context_debug),
        }


@dataclass(slots=True)
class StateSnapshot:
    turn_id: str
    session_id: str
    persona_name: str = ''
    current_entity: str = ''
    active_role: str = ''
    mood_cluster: str = ''
    summary: str = ''
    active_layers: list[str] = field(default_factory=list)
    graph_anchors: list[str] = field(default_factory=list)
    memory_anchors: list[str] = field(default_factory=list)
    priorities: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    source: str = 'deterministic'

    def to_dict(self) -> dict[str, Any]:
        return {
            'turn_id': self.turn_id,
            'session_id': self.session_id,
            'persona_name': self.persona_name,
            'current_entity': self.current_entity,
            'active_role': self.active_role,
            'mood_cluster': self.mood_cluster,
            'summary': self.summary,
            'active_layers': list(self.active_layers),
            'graph_anchors': list(self.graph_anchors),
            'memory_anchors': list(self.memory_anchors),
            'priorities': list(self.priorities),
            'risks': list(self.risks),
            'constraints': list(self.constraints),
            'changed': list(self.changed),
            'unchanged': list(self.unchanged),
            'source': self.source,
        }


@dataclass(slots=True)
class InfluenceInterpretation:
    summary: str = ''
    activation: list[str] = field(default_factory=list)
    pressure_points: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    direction: str = ''
    tension: str = ''
    role_pressure: str = ''
    uncertainty_level: str = 'moderate'
    risk_level: str = 'low'
    source: str = 'deterministic'

    def to_dict(self) -> dict[str, Any]:
        return {
            'summary': self.summary,
            'activation': list(self.activation),
            'pressure_points': list(self.pressure_points),
            'themes': list(self.themes),
            'conflicts': list(self.conflicts),
            'direction': self.direction,
            'tension': self.tension,
            'role_pressure': self.role_pressure,
            'uncertainty_level': self.uncertainty_level,
            'risk_level': self.risk_level,
            'source': self.source,
        }


@dataclass(slots=True)
class WorkingContextLayer:
    context_id: str
    turn_id: str
    session_id: str
    persona_name: str = ''
    current_entity: str = ''
    summary: str = ''
    sections: dict[str, str] = field(default_factory=dict)
    important_items: list[dict[str, Any]] = field(default_factory=list)
    weak_items: list[dict[str, Any]] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    priorities: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    source_counts: dict[str, int] = field(default_factory=dict)
    estimated_tokens: int = 0
    source: str = 'deterministic'

    def to_dict(self) -> dict[str, Any]:
        return {
            'context_id': self.context_id,
            'turn_id': self.turn_id,
            'session_id': self.session_id,
            'persona_name': self.persona_name,
            'current_entity': self.current_entity,
            'summary': self.summary,
            'sections': dict(self.sections),
            'important_items': [dict(item) for item in list(self.important_items)],
            'weak_items': [dict(item) for item in list(self.weak_items)],
            'contradictions': list(self.contradictions),
            'priorities': list(self.priorities),
            'constraints': list(self.constraints),
            'risks': list(self.risks),
            'source_counts': {str(key): int(value or 0) for key, value in dict(self.source_counts).items()},
            'estimated_tokens': int(self.estimated_tokens or 0),
            'source': self.source,
        }


@dataclass(slots=True)
class ResponseShapingPlan:
    role: str = ''
    style: str = ''
    behavior_mode: str = ''
    summary: str = ''
    priorities: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    risk_posture: str = ''
    source: str = 'deterministic'

    def to_dict(self) -> dict[str, Any]:
        return {
            'role': self.role,
            'style': self.style,
            'behavior_mode': self.behavior_mode,
            'summary': self.summary,
            'priorities': list(self.priorities),
            'constraints': list(self.constraints),
            'risk_posture': self.risk_posture,
            'source': self.source,
        }


@dataclass(slots=True)
class ContextScoreBreakdown:
    relevance: float = 0.0
    recency: float = 0.0
    importance: float = 0.0
    confidence: float = 0.0
    persona_alignment: float = 0.0
    graph_connectivity: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            'relevance': round(float(self.relevance or 0.0), 6),
            'recency': round(float(self.recency or 0.0), 6),
            'importance': round(float(self.importance or 0.0), 6),
            'confidence': round(float(self.confidence or 0.0), 6),
            'persona_alignment': round(float(self.persona_alignment or 0.0), 6),
            'graph_connectivity': round(float(self.graph_connectivity or 0.0), 6),
            'total': round(float(self.total or 0.0), 6),
        }


@dataclass(slots=True)
class ContextCandidate:
    candidate_id: str
    source: str
    section: str
    item_type: str
    title: str
    text: str
    token_estimate: int
    score: ContextScoreBreakdown = field(default_factory=ContextScoreBreakdown)
    metadata: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    rank: int = 0
    selected: bool = False
    compressed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            'candidate_id': self.candidate_id,
            'source': self.source,
            'section': self.section,
            'item_type': self.item_type,
            'title': self.title,
            'text': self.text,
            'token_estimate': int(self.token_estimate or 0),
            'score': self.score.to_dict(),
            'metadata': dict(self.metadata),
            'reasons': list(self.reasons),
            'rank': int(self.rank or 0),
            'selected': bool(self.selected),
            'compressed': bool(self.compressed),
        }


@dataclass(slots=True)
class ChatTurnRequest:
    message: str
    session_id: str = ''
    selected_persona: str = ''
    explicit_context: str = ''
    language: str = 'en'


@dataclass(slots=True)
class BackgroundRebuildDecision:
    session_id: str
    personality_name: str = ''
    should_schedule: bool = False
    reason: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'session_id': self.session_id,
            'personality_name': self.personality_name,
            'should_schedule': self.should_schedule,
            'reason': self.reason,
        }


@dataclass(slots=True)
class ChatSideEffects:
    graph_write_sources: list[str] = field(default_factory=list)
    history_write_path: str = ''
    current_context_path: str = ''
    transition_log_path: str = ''
    persona_updates: list[str] = field(default_factory=list)
    rebuild: BackgroundRebuildDecision | None = None

    def add_graph_write(self, source: str) -> None:
        clean = str(source or '').strip()
        if clean and clean not in self.graph_write_sources:
            self.graph_write_sources.append(clean)

    def add_persona_update(self, update_type: str) -> None:
        clean = str(update_type or '').strip()
        if clean and clean not in self.persona_updates:
            self.persona_updates.append(clean)

    def to_dict(self) -> dict[str, Any]:
        return {
            'graph_write_sources': list(self.graph_write_sources),
            'history_write_path': self.history_write_path,
            'current_context_path': self.current_context_path,
            'transition_log_path': self.transition_log_path,
            'persona_updates': list(self.persona_updates),
            'rebuild': self.rebuild.to_dict() if self.rebuild is not None else {},
        }


@dataclass(slots=True)
class ChatTurnResult:
    assistant_reply: str
    response_language: str
    session_id: str
    session: dict[str, Any]
    persona_name: str
    graph_context: str
    current_entity: str
    analysis: MessageAnalysis
    classifications: list[ClassificationDecision]
    repair_status: dict[str, Any]
    proposal_requested: bool = False
    trace_id: str = ''
    side_effects: ChatSideEffects = field(default_factory=ChatSideEffects)
    persona_selection: PersonaSelectionExplanation = field(default_factory=PersonaSelectionExplanation)
    persona_response: PersonaResponseExplanation = field(default_factory=PersonaResponseExplanation)
    social_role: SocialRoleDecision = field(default_factory=SocialRoleDecision)
    fallback_strategy: BehavioralFallbackDecision | None = None
    mood_research: dict[str, Any] = field(default_factory=dict)
    state_transition: dict[str, Any] = field(default_factory=dict)
    behavior_trace: dict[str, Any] = field(default_factory=dict)
    context_preview: dict[str, Any] = field(default_factory=dict)
    runtime_status: dict[str, Any] = field(default_factory=dict)
    operator_messages: list[str] = field(default_factory=list)

    def to_dict(self, *, include_side_effects: bool = True) -> dict[str, Any]:
        payload = {
            'assistant_reply': self.assistant_reply,
            'response_language': self.response_language,
            'session_id': self.session_id,
            'trace_id': self.trace_id,
            'session': dict(self.session),
            'persona_name': self.persona_name,
            'graph_context': self.graph_context,
            'current_entity': self.current_entity,
            'analysis': self.analysis.to_dict(),
            'classifications': [
                {
                    'entity_name': decision.entity_name,
                    'entity_type': decision.entity_type,
                    'votes': dict(decision.votes),
                    'confidence': decision.confidence,
                }
                for decision in self.classifications
            ],
            'repair_status': dict(self.repair_status),
            'proposal_requested': self.proposal_requested,
            'persona_selection': self.persona_selection.to_dict(),
            'persona_response': self.persona_response.to_dict(),
            'social_role': self.social_role.to_dict(),
            'fallback_strategy': self.fallback_strategy.to_dict() if self.fallback_strategy is not None else {},
            'mood_research': dict(self.mood_research),
            'state_transition': dict(self.state_transition),
            'behavior_trace': dict(self.behavior_trace),
            'context_preview': dict(self.context_preview),
            'runtime_status': dict(self.runtime_status),
            'operator_messages': list(self.operator_messages),
        }
        if include_side_effects:
            payload['side_effects'] = self.side_effects.to_dict()
        return payload
