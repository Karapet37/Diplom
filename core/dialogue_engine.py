from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agent_roles import AgentRole
from .graph_core import RAMContextGraph
from .personality_core import PersonalityCore
from .speech_dna import SpeechDNA


@dataclass(frozen=True)
class DialogueContract:
    system_instructions: tuple[str, ...]
    personality_summary: dict[str, Any]
    style_summary: dict[str, Any]
    graph_summary: dict[str, Any]
    agent_roles: tuple[dict[str, object], ...]
    scenario: dict[str, object]

    def as_dict(self) -> dict[str, Any]:
        return {
            "system_instructions": list(self.system_instructions),
            "personality_summary": dict(self.personality_summary),
            "style_summary": dict(self.style_summary),
            "graph_summary": dict(self.graph_summary),
            "agent_roles": [dict(item) for item in self.agent_roles],
            "scenario": dict(self.scenario),
        }


class DialogueEngine:
    def build_contract(
        self,
        *,
        query: str,
        personality_core: PersonalityCore | None,
        speech_dna: SpeechDNA | None,
        ram_graph: RAMContextGraph,
        agent_roles: tuple[AgentRole, ...] = (),
        scenario: dict[str, object] | None = None,
    ) -> DialogueContract:
        personality_summary = personality_core.as_dict() if personality_core else {}
        style_summary = speech_dna.as_dict() if speech_dna else {}
        graph_summary = {
            "query": query,
            "ranked_nodes": [
                {
                    "node_id": item.node.node_id,
                    "name": item.node.node_core.name,
                    "type": item.node.node_core.node_type,
                    "description": item.node.node_core.description,
                    "score": item.score,
                    "reasons": list(item.reasons),
                }
                for item in ram_graph.ranked_nodes[:8]
            ],
            "edge_count": len(ram_graph.edges),
            "node_count": len(ram_graph.nodes),
            "signals": dict(ram_graph.signals),
        }
        instructions = [
            "Use only RAM graph context and explicit personality data.",
            "Do not answer like a polite assistant.",
            "Match stable personality logic before style decoration.",
        ]
        if speech_dna and speech_dna.style_embedding:
            instructions.append("Apply speech DNA to pacing, punctuation, vocabulary bias, and directness without copying user content.")
        if personality_core:
            instructions.append("Respect temperament, values, reasoning style, risk tolerance, aggression, and humor levels.")
        return DialogueContract(
            system_instructions=tuple(instructions),
            personality_summary=personality_summary,
            style_summary=style_summary,
            graph_summary=graph_summary,
            agent_roles=tuple(role.as_dict() for role in agent_roles),
            scenario=dict(scenario or {}),
        )
