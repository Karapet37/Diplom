"""Reasoning layer: semantic extraction, entity linking and confidence tracking."""

from __future__ import annotations

import re
from typing import Any

from src.living_system.embedding import HashEmbeddingService
from src.living_system.knowledge_sql import KnowledgeSQLStore


_ENTITY_RE = re.compile(r"(?:[A-ZА-Я][a-zа-я0-9_-]{1,40})|(?:[a-zа-я0-9_-]{3,40})", flags=re.UNICODE)

_RELATION_HINTS: tuple[tuple[str, str], ...] = (
    ("likes", "likes"),
    ("prefers", "likes"),
    ("loves", "likes"),
    ("fears", "fears"),
    ("avoids", "fears"),
    ("works at", "works_at"),
    ("работаю в", "works_at"),
    ("работает в", "works_at"),
    ("любит", "likes"),
    ("боится", "fears"),
)


class SemanticReasoningService:
    """Rule-based reasoning pipeline with explicit confidence and uncertainty flags."""

    def __init__(self, store: KnowledgeSQLStore, embeddings: HashEmbeddingService):
        self.store = store
        self.embeddings = embeddings

    @staticmethod
    def _extract_entities(text: str) -> list[str]:
        raw = [token.strip(" ,.;:!?\n\t") for token in _ENTITY_RE.findall(text or "")]
        deduped: list[str] = []
        seen: set[str] = set()
        for token in raw:
            normalized = token.casefold()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(token)
        return deduped[:30]

    @staticmethod
    def _infer_relation(text: str) -> str:
        source = str(text or "").lower()
        for hint, relation in _RELATION_HINTS:
            if hint in source:
                return relation
        return "related_to"

    @staticmethod
    def _entity_confidence(entity: str) -> float:
        token = str(entity or "").strip()
        if not token:
            return 0.0
        if len(token) <= 2:
            return 0.2
        if token.isnumeric():
            return 0.35
        if token[0].isupper():
            return 0.8
        return 0.6

    def process(self, text: str, *, user_id: str, language: str) -> dict[str, Any]:
        relation = self._infer_relation(text)
        entities = self._extract_entities(text)
        actions: list[dict[str, Any]] = []
        node_ids: list[str] = []
        uncertainties: list[dict[str, Any]] = []

        for entity in entities:
            confidence = self._entity_confidence(entity)
            existing = self.store.find_node(user_id=user_id, node_type="concept", display_name=entity)
            if existing is not None:
                node_id = str(existing["node_id"])
                node_ids.append(node_id)
                actions.append(
                    {
                        "action": "reuse_node",
                        "node_id": node_id,
                        "display_name": entity,
                        "confidence": max(confidence, float(existing.get("confidence", 0.7))),
                    }
                )
                continue

            if confidence < 0.55:
                uncertainties.append(
                    {
                        "entity": entity,
                        "reason": "low_confidence_entity",
                        "confidence": confidence,
                    }
                )
                continue

            node_id = self.store.upsert_node(
                {
                    "user_id": user_id,
                    "node_type": "concept",
                    "display_name": entity,
                    "confidence": confidence,
                    "language_code": language,
                    "properties": {"name": entity, "source": "semantic_extraction"},
                    "metadata": {"relation_hint": relation},
                }
            )
            node_ids.append(node_id)
            vector = self.embeddings.embed_and_store(
                owner_type="node",
                owner_id=node_id,
                text=entity,
            )
            actions.append(
                {
                    "action": "create_node",
                    "node_id": node_id,
                    "display_name": entity,
                    "confidence": confidence,
                    "embedding_dimensions": len(vector),
                }
            )

        edge_actions: list[dict[str, Any]] = []
        if len(node_ids) >= 2:
            base_conf = 0.7 if not uncertainties else 0.5
            for index in range(len(node_ids) - 1):
                edge_id = self.store.upsert_edge(
                    {
                        "user_id": user_id,
                        "from_node": node_ids[index],
                        "to_node": node_ids[index + 1],
                        "relation_type": relation,
                        "confidence": base_conf,
                        "weight": base_conf,
                        "metadata": {
                            "source": "semantic_linking",
                            "input_text": text,
                        },
                    }
                )
                edge_actions.append(
                    {
                        "action": "upsert_edge",
                        "edge_id": edge_id,
                        "from_node": node_ids[index],
                        "to_node": node_ids[index + 1],
                        "relation_type": relation,
                        "confidence": base_conf,
                    }
                )

        if not entities:
            uncertainties.append(
                {
                    "entity": "",
                    "reason": "no_entities_extracted",
                    "confidence": 0.0,
                }
            )

        confidence_values = [float(row.get("confidence", 0.0)) for row in actions + edge_actions]
        overall_conf = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        requires_confirmation = bool(uncertainties) or overall_conf < 0.55

        explanation = (
            "Entities were semantically extracted, linked to existing nodes when present, "
            "and new nodes were created only for confident unmatched entities."
        )

        return {
            "relation_type": relation,
            "actions": actions,
            "edge_actions": edge_actions,
            "uncertainties": uncertainties,
            "requires_confirmation": requires_confirmation,
            "confidence": round(float(overall_conf), 4),
            "explanation": explanation,
        }
