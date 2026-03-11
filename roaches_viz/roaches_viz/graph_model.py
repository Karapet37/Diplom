from __future__ import annotations

from dataclasses import dataclass
from typing import Any


NODE_TYPES = {
    "DOMAIN",
    "CONCEPT",
    "PATTERN",
    "EXAMPLE",
    "PERSON",
    "TRAIT",
    "SIGNAL",
    "AGENT",
    "PROFESSION",
}

EDGE_TYPES = {
    "IS_PART_OF",
    "HAS_TRAIT",
    "USES_PATTERN",
    "SAID_EXAMPLE",
    "WORKS_IN_DOMAIN",
    "RELATED_TO",
    "RESPONDS_WITH",
    "INSTANCE_OF",
    "EXAMPLE_OF",
}


def normalize_node_type(value: str) -> str:
    cleaned = str(value or "PATTERN").strip().upper()
    if cleaned in NODE_TYPES:
        return cleaned
    legacy_map = {
        "THOUGHT": "EXAMPLE",
        "TERM": "PATTERN",
        "CHARACTER": "PERSON",
        "STYLE": "TRAIT",
        "EPISODE": "EXAMPLE",
        "CONCEPT": "PATTERN",
        "TRIGGER": "SIGNAL",
        "SERIES": "DOMAIN",
    }
    return legacy_map.get(cleaned, "PATTERN")


@dataclass
class Node:
    id: str
    type: str
    name: str = ""
    description: str = ""
    what_it_is: str = ""
    how_it_works: str = ""
    how_to_recognize: str = ""
    examples_json: str = "[]"
    tags_json: str = "[]"
    speech_patterns_json: str = "[]"
    behavior_patterns_json: str = "[]"
    triggers_json: str = "[]"
    values_json: str = "[]"
    preferences_json: str = "[]"
    reaction_logic_json: str = "[]"
    tolerance_thresholds_json: str = "{}"
    conflict_patterns_json: str = "[]"
    background: str = ""
    profession: str = ""
    speech_style_json: str = "{}"
    temperament: str = ""
    tolerance_threshold: float = 0.5
    formality: float = 0.5
    slang_level: float = 0.3
    directness: float = 0.5
    profanity_tolerance: float = 0.1
    possible_intents_json: str = "[]"
    emotion_signals_json: str = "[]"
    conflict_level: float = 0.0
    irony_probability: float = 0.0
    logic_weight: float = 0.5
    emotion_weight: float = 0.5
    risk_weight: float = 0.5
    relevance_weight: float = 0.5
    confidence: float = 0.7
    label: str = ""
    short_gloss: str = ""
    plain_explanation: str = ""

    def __post_init__(self) -> None:
        self.type = normalize_node_type(self.type)
        if not self.name and self.label:
            self.name = self.label
        if not self.label and self.name:
            self.label = self.name
        if not self.description and self.short_gloss:
            self.description = self.short_gloss
        if not self.short_gloss and self.description:
            self.short_gloss = self.description
        if not self.what_it_is and self.plain_explanation:
            self.what_it_is = self.plain_explanation
        if not self.plain_explanation:
            self.plain_explanation = self.what_it_is or self.how_it_works or self.description


@dataclass
class Edge:
    src_id: str
    dst_id: str
    type: str
    weight: float = 1.0
    confidence: float = 0.7
    metadata_json: str = "{}"


@dataclass(frozen=True)
class Evidence:
    edge_key: str
    source_id: str
    snippet_text: str
    offset_start: int = 0
    offset_end: int = 0


@dataclass(frozen=True)
class Source:
    source_id: str
    raw_text: str


@dataclass(frozen=True)
class GraphSnapshot:
    nodes: dict[str, dict[str, Any]]
    edges: list[dict[str, Any]]
    adjacency: dict[str, tuple[str, ...]]


def edge_key(src_id: str, edge_type: str, dst_id: str) -> str:
    return f"{src_id}|{edge_type}|{dst_id}"
