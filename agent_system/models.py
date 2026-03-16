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
    situation: str = ''
    current_entity: str = ''
    explicit_context: str = ''
    entities: list[MessageEntity] = field(default_factory=list)
    cues: dict[str, Any] = field(default_factory=dict)


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
