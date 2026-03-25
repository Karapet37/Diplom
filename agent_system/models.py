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
            'persona_updates': list(self.persona_updates),
            'rebuild': self.rebuild.to_dict() if self.rebuild is not None else {},
        }


@dataclass(slots=True)
class ChatTurnResult:
    assistant_reply: str
    session_id: str
    session: dict[str, Any]
    persona_name: str
    graph_context: str
    current_entity: str
    analysis: MessageAnalysis
    classifications: list[ClassificationDecision]
    repair_status: dict[str, Any]
    proposal_requested: bool = False
    side_effects: ChatSideEffects = field(default_factory=ChatSideEffects)

    def to_dict(self, *, include_side_effects: bool = True) -> dict[str, Any]:
        payload = {
            'assistant_reply': self.assistant_reply,
            'session_id': self.session_id,
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
        }
        if include_side_effects:
            payload['side_effects'] = self.side_effects.to_dict()
        return payload
