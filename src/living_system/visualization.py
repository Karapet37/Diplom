"""Visualization layer for graph structure and behavior views."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.living_system.knowledge_sql import KnowledgeSQLStore


class GraphVisualizationService:
    """Builds textual and Mermaid graph visualizations from SQL state."""

    def __init__(self, store: KnowledgeSQLStore):
        self.store = store

    def graph_view(self, *, user_id: str = "") -> dict[str, Any]:
        state = self.store.graph_state(user_id=user_id)
        nodes = list(state.get("nodes", []) or [])
        edges = list(state.get("edges", []) or [])
        node_type_counts = Counter(str(row.get("node_type", "generic")) for row in nodes)
        relation_counts = Counter(str(row.get("relation_type", "related_to")) for row in edges)

        lines = ["graph TD"]
        for row in nodes[:120]:
            node_id = str(row.get("node_id", ""))
            name = str(row.get("display_name", node_id)).replace("\"", "'")
            lines.append(f"    {node_id}[\"{name}\"]")
        for row in edges[:240]:
            src = str(row.get("from_node", ""))
            dst = str(row.get("to_node", ""))
            rel = str(row.get("relation_type", "related_to")).replace("\"", "'")
            lines.append(f"    {src} -->|{rel}| {dst}")

        return {
            "metrics": {
                "nodes": len(nodes),
                "edges": len(edges),
                "node_type_counts": dict(sorted(node_type_counts.items())),
                "relation_counts": dict(sorted(relation_counts.items())),
            },
            "mermaid": "\n".join(lines),
            "state": state,
        }
