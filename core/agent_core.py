from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class AgentMutationProposal:
    target_scope: str
    operation: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentCore:
    agent_id: str
    name: str
    purpose: str
    readable_scopes: tuple[str, ...] = ("graph", "personality", "ram_graph")
    writable_scopes: tuple[str, ...] = ("node_branches", "node_context")
    forbidden_scopes: tuple[str, ...] = ("node_core", "personality_core")

    def can_read(self, scope: str) -> bool:
        return scope in self.readable_scopes

    def can_write(self, scope: str) -> bool:
        return scope in self.writable_scopes and scope not in self.forbidden_scopes

    def validate_proposal(self, proposal: AgentMutationProposal) -> None:
        if proposal.target_scope in self.forbidden_scopes:
            raise PermissionError(f"agent '{self.agent_id}' cannot modify forbidden scope '{proposal.target_scope}'")
        if proposal.target_scope not in self.writable_scopes:
            raise PermissionError(f"agent '{self.agent_id}' cannot modify scope '{proposal.target_scope}'")
