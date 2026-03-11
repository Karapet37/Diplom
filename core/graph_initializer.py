from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .graph_core import GraphMemory


@dataclass(frozen=True)
class GraphInitializationResult:
    graph_memory: GraphMemory
    layer_index: Mapping[str, tuple[str, ...]]
    stats: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": {key: list(value) for key, value in self.layer_index.items()},
            "stats": dict(self.stats),
            "graph_memory": self.graph_memory.as_dict(),
        }


class GraphInitializer:
    def initialize(self, graph_payload: dict[str, Any]) -> GraphInitializationResult:
        from roaches_viz.roaches_viz.core_bridge import graph_payload_to_memory

        graph_memory = graph_payload_to_memory(graph_payload)
        layers: dict[str, list[str]] = {
            "root_core": [],
            "domain_core": [],
            "concept_core": [],
            "pattern_core": [],
            "example_nodes": [],
        }
        for node_id, node in graph_memory.nodes.items():
            tags = set(node.node_context.tags)
            node_type = node.node_core.node_type.upper()
            if "root_core" in tags:
                layers["root_core"].append(node_id)
            if node_type == "DOMAIN" or "domain_core" in tags:
                layers["domain_core"].append(node_id)
            if node_type == "CONCEPT" or "concept_core" in tags:
                layers["concept_core"].append(node_id)
            if node_type == "PATTERN" or "pattern_core" in tags:
                layers["pattern_core"].append(node_id)
            if node_type == "EXAMPLE" or "example_nodes" in tags:
                layers["example_nodes"].append(node_id)
        frozen_layers = {key: tuple(sorted(values)) for key, values in layers.items()}
        stats = {
            "node_count": len(graph_memory.nodes),
            "edge_count": len(graph_memory.edges),
            "root_count": len(frozen_layers["root_core"]),
            "domain_count": len(frozen_layers["domain_core"]),
            "concept_count": len(frozen_layers["concept_core"]),
            "pattern_count": len(frozen_layers["pattern_core"]),
            "example_count": len(frozen_layers["example_nodes"]),
        }
        return GraphInitializationResult(graph_memory=graph_memory, layer_index=frozen_layers, stats=stats)
