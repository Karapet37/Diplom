from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .agent_core import AgentCore
from .context_core import ContextCore
from .graph_core import GraphMemory
from .personality_core import PersonalityCore


@dataclass(frozen=True)
class SystemCore:
    graph_memory: GraphMemory
    personalities: Mapping[str, PersonalityCore] = field(default_factory=dict)
    agents: Mapping[str, AgentCore] = field(default_factory=dict)
    context_builder: ContextCore = field(default_factory=ContextCore)

    def describe(self) -> dict[str, object]:
        return {
            "core": "system",
            "graph_nodes": len(self.graph_memory.nodes),
            "graph_edges": len(self.graph_memory.edges),
            "personalities": sorted(self.personalities.keys()),
            "agents": sorted(self.agents.keys()),
            "context_builder": self.context_builder.name,
            "principles": [
                "node_core is immutable",
                "agents cannot modify node_core directly",
                "llm receives RAM context graph only",
                "personality changes must be gradual",
            ],
        }
