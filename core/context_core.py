from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .graph_core import GraphEdge, GraphNode, RAMContextGraph, RankedContextNode


@dataclass(frozen=True)
class ContextSignals:
    entities: tuple[str, ...] = ()
    emotion: tuple[str, ...] = ()
    logic: tuple[str, ...] = ()
    intent: tuple[str, ...] = ()
    social: tuple[str, ...] = ()
    references: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "entities": list(self.entities),
            "emotion": list(self.emotion),
            "logic": list(self.logic),
            "intent": list(self.intent),
            "social": list(self.social),
            "references": list(self.references),
        }


@dataclass(frozen=True)
class ContextQuery:
    query: str
    recent_history: tuple[str, ...] = ()
    person_id: str | None = None
    max_nodes: int = 16


@dataclass(frozen=True)
class SearchHit:
    node_id: str
    score: float
    reasons: tuple[str, ...] = ()


class SignalExtractorPort(Protocol):
    def extract(self, request: ContextQuery) -> ContextSignals: ...


class GraphSearchPort(Protocol):
    def search(self, request: ContextQuery, signals: ContextSignals) -> Sequence[SearchHit]: ...


class GraphHydrationPort(Protocol):
    def hydrate(self, node_ids: Sequence[str]) -> Mapping[str, GraphNode]: ...


class BranchExpansionPort(Protocol):
    def expand(self, node_ids: Sequence[str], hydrated_nodes: Mapping[str, GraphNode], max_nodes: int) -> tuple[Sequence[GraphNode], Sequence[GraphEdge]]: ...


@dataclass(frozen=True)
class ContextCore:
    name: str = "default_context_core"

    def build_ram_context(
        self,
        request: ContextQuery,
        *,
        signal_extractor: SignalExtractorPort,
        graph_search: GraphSearchPort,
        graph_hydration: GraphHydrationPort,
        branch_expander: BranchExpansionPort,
    ) -> RAMContextGraph:
        signals = signal_extractor.extract(request)
        hits = list(graph_search.search(request, signals))
        top_hits = hits[: max(1, request.max_nodes)]
        hydrated_nodes = graph_hydration.hydrate([item.node_id for item in top_hits])
        expanded_nodes, expanded_edges = branch_expander.expand([item.node_id for item in top_hits], hydrated_nodes, request.max_nodes)
        node_index = {node.node_id: node for node in expanded_nodes}
        ranked_nodes = tuple(
            RankedContextNode(
                node=node_index[item.node_id],
                score=float(item.score),
                reasons=tuple(item.reasons),
            )
            for item in top_hits
            if item.node_id in node_index
        )
        return RAMContextGraph(
            query=request.query,
            signals={"query_signals": signals.as_dict()},
            nodes=tuple(expanded_nodes),
            edges=tuple(expanded_edges),
            ranked_nodes=ranked_nodes,
            recent_history=request.recent_history,
        )
