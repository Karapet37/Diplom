from __future__ import annotations

from dataclasses import dataclass

from .agent_core import AgentCore


@dataclass(frozen=True)
class AgentRole:
    role_id: str
    title: str
    reasoning_focus: tuple[str, ...]
    agent_core: AgentCore

    def as_dict(self) -> dict[str, object]:
        return {
            "role_id": self.role_id,
            "title": self.title,
            "reasoning_focus": list(self.reasoning_focus),
            "agent_core": {
                "agent_id": self.agent_core.agent_id,
                "name": self.agent_core.name,
                "purpose": self.agent_core.purpose,
                "readable_scopes": list(self.agent_core.readable_scopes),
                "writable_scopes": list(self.agent_core.writable_scopes),
                "forbidden_scopes": list(self.agent_core.forbidden_scopes),
            },
        }


def default_agent_roles() -> dict[str, AgentRole]:
    return {
        "law": AgentRole(
            role_id="law",
            title="Law Agent",
            reasoning_focus=("rights", "duties", "evidence", "liability", "escalation_risk"),
            agent_core=AgentCore(agent_id="agent:law", name="Law Agent", purpose="Reads graph evidence through legal obligations and escalation risk."),
        ),
        "business": AgentRole(
            role_id="business",
            title="Business Agent",
            reasoning_focus=("incentives", "negotiation", "status", "pressure", "tradeoffs"),
            agent_core=AgentCore(agent_id="agent:business", name="Business Agent", purpose="Reads graph evidence through incentives, leverage, and tradeoffs."),
        ),
        "strategy": AgentRole(
            role_id="strategy",
            title="Strategy Agent",
            reasoning_focus=("sequencing", "alternatives", "timing", "constraints", "next_moves"),
            agent_core=AgentCore(agent_id="agent:strategy", name="Strategy Agent", purpose="Turns graph evidence into ordered next-step reasoning."),
        ),
    }
