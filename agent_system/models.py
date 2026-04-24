from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

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
PersonaType = Literal['archetype', 'psychological', 'situational', 'hybrid']
ChangeDirection = Literal['lighter', 'darker', 'mixed', 'unstable']
ValidationStatus = Literal['valid', 'partial', 'rejected']
PersonaReadiness = Literal['seed', 'draft', 'full']
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
    session_persona: str = ''
    explicit_context: str = ''
    entities: list[MessageEntity] = field(default_factory=list)
    user_state: UserState = field(default_factory=UserState)
    situation: Situation = field(default_factory=Situation)

    @property
    def cues(self) -> dict[str, Any]:
        return dict(self.user_state.signals)

    def to_dict(self) -> dict[str, Any]:
        return {
            'selected_head': self.selected_head,
            'primary_entity': self.primary_entity,
            'current_entity': self.current_entity,
            'session_persona': self.session_persona,
            'user_state': self.user_state.to_dict(),
            'situation': self.situation.to_dict(),
            'entities': [entity.name for entity in self.entities],
        }


@dataclass(slots=True)
class InteractionFrame:
    message_kind: str = 'statement'
    question_present: bool = False
    explicit_persona_switch: bool = False
    requested_persona: str = ''
    keep_session_persona: bool = False
    topic_entity: str = ''
    topic_mode: str = 'unknown'
    followup_mode: str = 'none'
    routed_message: str = ''
    evidence: list[str] = field(default_factory=list)
    source: str = 'deterministic'

    def to_dict(self) -> dict[str, Any]:
        return {
            'message_kind': self.message_kind,
            'question_present': bool(self.question_present),
            'explicit_persona_switch': bool(self.explicit_persona_switch),
            'requested_persona': self.requested_persona,
            'keep_session_persona': bool(self.keep_session_persona),
            'topic_entity': self.topic_entity,
            'topic_mode': self.topic_mode,
            'followup_mode': self.followup_mode,
            'routed_message': self.routed_message,
            'evidence': list(self.evidence),
            'source': self.source,
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
class PersonaIdentity:
    label: str = ''
    short_description: str = ''
    persona_type: PersonaType = 'hybrid'
    source_text: str = ''
    readiness: PersonaReadiness = 'draft'

    def to_dict(self) -> dict[str, Any]:
        return {
            'label': self.label,
            'short_description': self.short_description,
            'persona_type': self.persona_type,
            'source_text': self.source_text,
            'readiness': self.readiness,
        }


@dataclass(slots=True)
class PersonaCore:
    self_image: list[str] = field(default_factory=list)
    visible_traits: list[str] = field(default_factory=list)
    hidden_traits: list[str] = field(default_factory=list)
    motivations: list[str] = field(default_factory=list)
    fears: list[str] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)
    vulnerabilities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'self_image': list(self.self_image),
            'visible_traits': list(self.visible_traits),
            'hidden_traits': list(self.hidden_traits),
            'motivations': list(self.motivations),
            'fears': list(self.fears),
            'needs': list(self.needs),
            'vulnerabilities': list(self.vulnerabilities),
        }


@dataclass(slots=True)
class PersonaConflict:
    internal_contradictions: list[str] = field(default_factory=list)
    shame_points: list[str] = field(default_factory=list)
    dependency_patterns: list[str] = field(default_factory=list)
    resentment_patterns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'internal_contradictions': list(self.internal_contradictions),
            'shame_points': list(self.shame_points),
            'dependency_patterns': list(self.dependency_patterns),
            'resentment_patterns': list(self.resentment_patterns),
        }


@dataclass(slots=True)
class PersonaDefense:
    defense_mechanisms: list[str] = field(default_factory=list)
    self_justifications: list[str] = field(default_factory=list)
    avoidance_patterns: list[str] = field(default_factory=list)
    escalation_patterns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'defense_mechanisms': list(self.defense_mechanisms),
            'self_justifications': list(self.self_justifications),
            'avoidance_patterns': list(self.avoidance_patterns),
            'escalation_patterns': list(self.escalation_patterns),
        }


@dataclass(slots=True)
class PersonaBehavior:
    communication_style: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    pressure_response: list[str] = field(default_factory=list)
    attachment_style: Optional[str] = None
    refusal_style: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'communication_style': list(self.communication_style),
            'triggers': list(self.triggers),
            'pressure_response': list(self.pressure_response),
            'attachment_style': self.attachment_style,
            'refusal_style': self.refusal_style,
        }


@dataclass(slots=True)
class PersonaDynamics:
    resistance_to_change: str = ''
    growth_pattern: str = ''
    likely_change_direction: ChangeDirection = 'unstable'
    softening_conditions: list[str] = field(default_factory=list)
    darkening_conditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'resistance_to_change': self.resistance_to_change,
            'growth_pattern': self.growth_pattern,
            'likely_change_direction': self.likely_change_direction,
            'softening_conditions': list(self.softening_conditions),
            'darkening_conditions': list(self.darkening_conditions),
        }


@dataclass(slots=True)
class PersonaMeta:
    tags: list[str] = field(default_factory=list)
    hover_text: str = ''
    validation_status: ValidationStatus = 'partial'
    validation_notes: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            'tags': list(self.tags),
            'hover_text': self.hover_text,
            'validation_status': self.validation_status,
            'validation_notes': list(self.validation_notes),
            'confidence': round(float(self.confidence or 0.0), 6),
        }


@dataclass(slots=True)
class StructuredPersona:
    identity: PersonaIdentity = field(default_factory=PersonaIdentity)
    core_goal: str = ''
    secondary_goals: list[str] = field(default_factory=list)
    fears: list[str] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)
    constraints_internal: list[str] = field(default_factory=list)
    constraints_social: list[str] = field(default_factory=list)
    constraints_hard_system: list[str] = field(default_factory=list)
    allowed_methods: list[str] = field(default_factory=list)
    maladaptive_methods: list[str] = field(default_factory=list)
    core: PersonaCore = field(default_factory=PersonaCore)
    conflict: PersonaConflict = field(default_factory=PersonaConflict)
    defense: PersonaDefense = field(default_factory=PersonaDefense)
    behavior: PersonaBehavior = field(default_factory=PersonaBehavior)
    dynamics: PersonaDynamics = field(default_factory=PersonaDynamics)
    meta: PersonaMeta = field(default_factory=PersonaMeta)

    def is_minimally_valid(self) -> bool:
        return (
            bool(self.identity.label.strip())
            and len(self.core.visible_traits) + len(self.core.hidden_traits) >= 2
            and (
                bool(str(self.core_goal or '').strip())
                or len(self.secondary_goals) > 0
                or len(self.needs) > 0
            )
            and (
                len(self.conflict.internal_contradictions) > 0
                or len(self.conflict.dependency_patterns) > 0
                or len(self.defense.defense_mechanisms) > 0
                or len(self.constraints_internal) > 0
                or len(self.constraints_social) > 0
            )
            and bool(self.meta.hover_text.strip())
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'identity': self.identity.to_dict(),
            'core_goal': self.core_goal,
            'secondary_goals': list(self.secondary_goals),
            'fears': list(self.fears),
            'needs': list(self.needs),
            'constraints_internal': list(self.constraints_internal),
            'constraints_social': list(self.constraints_social),
            'constraints_hard_system': list(self.constraints_hard_system),
            'allowed_methods': list(self.allowed_methods),
            'maladaptive_methods': list(self.maladaptive_methods),
            'core': self.core.to_dict(),
            'conflict': self.conflict.to_dict(),
            'defense': self.defense.to_dict(),
            'behavior': self.behavior.to_dict(),
            'dynamics': self.dynamics.to_dict(),
            'meta': self.meta.to_dict(),
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
    structured_persona: StructuredPersona | None = None


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
class TaskProcedurePlan:
    procedure_family: str = ''
    requested_outcome: str = ''
    response_form: str = ''
    response_language: str = 'en'
    form_drivers: list[str] = field(default_factory=list)
    content_sources: list[str] = field(default_factory=list)
    forbidden_mixins: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    execution_steps: list[str] = field(default_factory=list)
    uncertainty_strategy: str = ''
    summary: str = ''
    source: str = 'deterministic'

    def to_dict(self) -> dict[str, Any]:
        return {
            'procedure_family': self.procedure_family,
            'requested_outcome': self.requested_outcome,
            'response_form': self.response_form,
            'response_language': self.response_language,
            'form_drivers': list(self.form_drivers),
            'content_sources': list(self.content_sources),
            'forbidden_mixins': list(self.forbidden_mixins),
            'success_criteria': list(self.success_criteria),
            'execution_steps': list(self.execution_steps),
            'uncertainty_strategy': self.uncertainty_strategy,
            'summary': self.summary,
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
    response_form: str = ''
    summary: str = ''
    priorities: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    forbidden_mixins: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    risk_posture: str = ''
    source: str = 'deterministic'

    def to_dict(self) -> dict[str, Any]:
        return {
            'role': self.role,
            'style': self.style,
            'behavior_mode': self.behavior_mode,
            'response_form': self.response_form,
            'summary': self.summary,
            'priorities': list(self.priorities),
            'constraints': list(self.constraints),
            'forbidden_mixins': list(self.forbidden_mixins),
            'success_criteria': list(self.success_criteria),
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
    user_persona_name: str = ''  # who the user is speaking as ("Незнакомец" = anonymous)


@dataclass(slots=True)
class RequestEnvelope:
    request_id: str
    session_id: str
    raw_text: str
    normalized_text: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'request_id': self.request_id,
            'session_id': self.session_id,
            'raw_text': self.raw_text,
            'normalized_text': self.normalized_text,
            'timestamp': self.timestamp,
        }


@dataclass(slots=True)
class ControllerState:
    envelope: RequestEnvelope
    interaction_frame: InteractionFrame
    analysis: MessageAnalysis
    preprocessing: RequestPreprocessing
    route: RouteDecision
    capability_plan: CapabilityPlan
    selected_persona: str = ''
    session_persona: str = ''
    current_entity: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'envelope': self.envelope.to_dict(),
            'interaction_frame': self.interaction_frame.to_dict(),
            'analysis': self.analysis.to_dict(),
            'preprocessing': self.preprocessing.to_dict(),
            'route': self.route.to_dict(),
            'capability_plan': self.capability_plan.to_dict(),
            'selected_persona': self.selected_persona,
            'session_persona': self.session_persona,
            'current_entity': self.current_entity,
        }


@dataclass(slots=True)
class RequestPreprocessing:
    request_type: str = 'general_chat'
    detected_language: str = 'en'
    intent_type: str = 'statement'
    interaction_mode: str = 'conversation'
    request_kind: str = 'statement'
    hypothetical: bool = False
    hypothetical_continuation: bool = False
    meta_request: bool = False
    file_related: bool = False
    system_debug: bool = False
    clarification_needed: bool = False
    persona_style_traits: list[str] = field(default_factory=list)
    speech_style_hints: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'request_type': self.request_type,
            'detected_language': self.detected_language,
            'intent_type': self.intent_type,
            'interaction_mode': self.interaction_mode,
            'request_kind': self.request_kind,
            'hypothetical': bool(self.hypothetical),
            'hypothetical_continuation': bool(self.hypothetical_continuation),
            'meta_request': bool(self.meta_request),
            'file_related': bool(self.file_related),
            'system_debug': bool(self.system_debug),
            'clarification_needed': bool(self.clarification_needed),
            'persona_style_traits': list(self.persona_style_traits),
            'speech_style_hints': list(self.speech_style_hints),
            'evidence': list(self.evidence),
        }


@dataclass(slots=True)
class RouteDecision:
    selected_route: str = 'factual_answer'
    request_type: str = 'general_chat'
    detected_language: str = 'en'
    intent_type: str = 'statement'
    interaction_mode: str = 'conversation'
    requires_history: bool = False
    requires_graph: bool = False
    requires_persona: bool = False
    requires_llm: bool = True
    strict_grounding: bool = True
    response_style: str = 'direct_answer'
    validation_mode: str = 'grounded_factual'
    fast_path: bool = False
    persona_style_traits: list[str] = field(default_factory=list)
    speech_style_hints: list[str] = field(default_factory=list)
    secondary_flags: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'selected_route': self.selected_route,
            'request_type': self.request_type,
            'detected_language': self.detected_language,
            'intent_type': self.intent_type,
            'interaction_mode': self.interaction_mode,
            'requires_history': bool(self.requires_history),
            'requires_graph': bool(self.requires_graph),
            'requires_persona': bool(self.requires_persona),
            'requires_llm': bool(self.requires_llm),
            'strict_grounding': bool(self.strict_grounding),
            'response_style': self.response_style,
            'validation_mode': self.validation_mode,
            'fast_path': bool(self.fast_path),
            'persona_style_traits': list(self.persona_style_traits),
            'speech_style_hints': list(self.speech_style_hints),
            'secondary_flags': list(self.secondary_flags),
            'reason_codes': list(self.reason_codes),
            'evidence': list(self.evidence),
        }


@dataclass(slots=True)
class RouteMemory:
    previous_route: str = ''
    previous_request_type: str = ''
    previous_interaction_mode: str = ''
    previous_response_style: str = ''
    previous_persona_name: str = ''
    previous_current_entity: str = ''
    persona_style_traits: list[str] = field(default_factory=list)
    speech_style_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'previous_route': self.previous_route,
            'previous_request_type': self.previous_request_type,
            'previous_interaction_mode': self.previous_interaction_mode,
            'previous_response_style': self.previous_response_style,
            'previous_persona_name': self.previous_persona_name,
            'previous_current_entity': self.previous_current_entity,
            'persona_style_traits': list(self.persona_style_traits),
            'speech_style_hints': list(self.speech_style_hints),
        }


@dataclass(slots=True)
class TraceLearningPolicy:
    selected_route: str = ''
    batch_size: int = 0
    sampled_trace_count: int = 0
    session_trace_count: int = 0
    global_trace_count: int = 0
    signal_trace_count: int = 0
    failure_density: float = 0.0
    output_budget_boost: int = 0
    route_guidance_lines: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'selected_route': self.selected_route,
            'batch_size': int(self.batch_size or 0),
            'sampled_trace_count': int(self.sampled_trace_count or 0),
            'session_trace_count': int(self.session_trace_count or 0),
            'global_trace_count': int(self.global_trace_count or 0),
            'signal_trace_count': int(self.signal_trace_count or 0),
            'failure_density': float(self.failure_density or 0.0),
            'output_budget_boost': int(self.output_budget_boost or 0),
            'route_guidance_lines': list(self.route_guidance_lines),
            'reason_codes': list(self.reason_codes),
        }


@dataclass(slots=True)
class CapabilityPlan:
    requires_history: bool = False
    requires_graph: bool = False
    requires_persona: bool = False
    requires_llm: bool = True
    use_context_builder: bool = False
    use_heavy_persona_pipeline: bool = False
    strict_grounding: bool = True
    preferred_generation_mode: str = 'chat_llm'
    context_sources: list[str] = field(default_factory=list)
    skipped_components: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'requires_history': bool(self.requires_history),
            'requires_graph': bool(self.requires_graph),
            'requires_persona': bool(self.requires_persona),
            'requires_llm': bool(self.requires_llm),
            'use_context_builder': bool(self.use_context_builder),
            'use_heavy_persona_pipeline': bool(self.use_heavy_persona_pipeline),
            'strict_grounding': bool(self.strict_grounding),
            'preferred_generation_mode': self.preferred_generation_mode,
            'context_sources': list(self.context_sources),
            'skipped_components': list(self.skipped_components),
            'reason_codes': list(self.reason_codes),
        }


@dataclass(slots=True)
class ResponseValidation:
    ok: bool = True
    validation_mode: str = 'grounded_factual'
    route_match: bool = True
    relevance_match: bool = True
    used_history: bool = False
    used_graph: bool = False
    used_persona: bool = False
    fallback_triggered: bool = False
    fallback_reason_code: str = ''
    mismatch_reason: str = ''
    repair_strategy: str = ''
    repaired: bool = False
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'ok': bool(self.ok),
            'validation_mode': self.validation_mode,
            'route_match': bool(self.route_match),
            'relevance_match': bool(self.relevance_match),
            'used_history': bool(self.used_history),
            'used_graph': bool(self.used_graph),
            'used_persona': bool(self.used_persona),
            'fallback_triggered': bool(self.fallback_triggered),
            'fallback_reason_code': self.fallback_reason_code,
            'mismatch_reason': self.mismatch_reason,
            'repair_strategy': self.repair_strategy,
            'repaired': bool(self.repaired),
            'reason_codes': list(self.reason_codes),
        }


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
    route_state_path: str = ''
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
            'route_state_path': self.route_state_path,
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
    pipeline: dict[str, Any] = field(default_factory=dict)
    validation: ResponseValidation = field(default_factory=ResponseValidation)
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
    persona_action: dict[str, Any] = field(default_factory=dict)

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
            'pipeline': dict(self.pipeline),
            'validation': self.validation.to_dict(),
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
            'persona_action': dict(self.persona_action),
        }
        if include_side_effects:
            payload['side_effects'] = self.side_effects.to_dict()
        return payload
