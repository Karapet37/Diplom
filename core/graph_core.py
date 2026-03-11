from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ImportanceVector:
    logic_weight: float = 0.5
    emotion_weight: float = 0.5
    risk_weight: float = 0.5
    relevance_weight: float = 0.5

    def clamped(self) -> "ImportanceVector":
        return ImportanceVector(
            logic_weight=min(max(float(self.logic_weight), 0.0), 1.0),
            emotion_weight=min(max(float(self.emotion_weight), 0.0), 1.0),
            risk_weight=min(max(float(self.risk_weight), 0.0), 1.0),
            relevance_weight=min(max(float(self.relevance_weight), 0.0), 1.0),
        )

    def as_dict(self) -> dict[str, float]:
        c = self.clamped()
        return {
            "logic_weight": c.logic_weight,
            "emotion_weight": c.emotion_weight,
            "risk_weight": c.risk_weight,
            "relevance_weight": c.relevance_weight,
        }


@dataclass(frozen=True)
class NodeCore:
    node_id: str
    node_type: str
    name: str
    description: str
    importance_vector: ImportanceVector = field(default_factory=ImportanceVector)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeBranches:
    patterns: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    signals: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()

    def merged(self, *, patterns: tuple[str, ...] | None = None, examples: tuple[str, ...] | None = None, signals: tuple[str, ...] | None = None, relations: tuple[str, ...] | None = None) -> "NodeBranches":
        return NodeBranches(
            patterns=self.patterns if patterns is None else tuple(patterns),
            examples=self.examples if examples is None else tuple(examples),
            signals=self.signals if signals is None else tuple(signals),
            relations=self.relations if relations is None else tuple(relations),
        )


@dataclass(frozen=True)
class NodeContext:
    domains: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    recent_references: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    node_core: NodeCore
    node_branches: NodeBranches = field(default_factory=NodeBranches)
    node_context: NodeContext = field(default_factory=NodeContext)

    def with_branches(self, node_branches: NodeBranches) -> "GraphNode":
        return GraphNode(
            node_id=self.node_id,
            node_core=self.node_core,
            node_branches=node_branches,
            node_context=self.node_context,
        )

    def with_context(self, node_context: NodeContext) -> "GraphNode":
        return GraphNode(
            node_id=self.node_id,
            node_core=self.node_core,
            node_branches=self.node_branches,
            node_context=node_context,
        )


@dataclass(frozen=True)
class GraphEdge:
    src_id: str
    dst_id: str
    edge_type: str
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankedContextNode:
    node: GraphNode
    score: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RAMContextGraph:
    query: str
    signals: Mapping[str, Any]
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    ranked_nodes: tuple[RankedContextNode, ...]
    recent_history: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "signals": dict(self.signals),
            "nodes": [serialize_graph_node(node) for node in self.nodes],
            "edges": [serialize_graph_edge(edge) for edge in self.edges],
            "ranked_nodes": [
                {
                    "node_id": item.node.node_id,
                    "name": item.node.node_core.name,
                    "type": item.node.node_core.node_type,
                    "description": item.node.node_core.description,
                    "score": item.score,
                    "reasons": list(item.reasons),
                    "importance_vector": item.node.node_core.importance_vector.as_dict(),
                }
                for item in self.ranked_nodes
            ],
            "recent_history": list(self.recent_history),
        }


@dataclass(frozen=True)
class GraphMemory:
    nodes: Mapping[str, GraphNode]
    edges: tuple[GraphEdge, ...]

    def get_node(self, node_id: str) -> GraphNode | None:
        return self.nodes.get(node_id)

    def neighbors(self, node_id: str) -> tuple[GraphNode, ...]:
        related: list[GraphNode] = []
        seen: set[str] = set()
        for edge in self.edges:
            other_id = ""
            if edge.src_id == node_id:
                other_id = edge.dst_id
            elif edge.dst_id == node_id:
                other_id = edge.src_id
            if not other_id or other_id in seen:
                continue
            node = self.nodes.get(other_id)
            if node is None:
                continue
            seen.add(other_id)
            related.append(node)
        return tuple(related)

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": {node_id: serialize_graph_node(node) for node_id, node in self.nodes.items()},
            "edges": [serialize_graph_edge(edge) for edge in self.edges],
        }


def serialize_graph_node(node: GraphNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "node_core": {
            "node_id": node.node_core.node_id,
            "node_type": node.node_core.node_type,
            "name": node.node_core.name,
            "description": node.node_core.description,
            "importance_vector": node.node_core.importance_vector.as_dict(),
            "metadata": dict(node.node_core.metadata),
        },
        "node_branches": {
            "patterns": list(node.node_branches.patterns),
            "examples": list(node.node_branches.examples),
            "signals": list(node.node_branches.signals),
            "relations": list(node.node_branches.relations),
        },
        "node_context": {
            "domains": list(node.node_context.domains),
            "tags": list(node.node_context.tags),
            "recent_references": list(node.node_context.recent_references),
            "note": node.node_context.note,
        },
    }


def serialize_graph_edge(edge: GraphEdge) -> dict[str, Any]:
    return {
        "src_id": edge.src_id,
        "dst_id": edge.dst_id,
        "edge_type": edge.edge_type,
        "weight": edge.weight,
        "metadata": dict(edge.metadata),
    }
