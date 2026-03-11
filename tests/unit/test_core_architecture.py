from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from core import AgentCore, AgentMutationProposal, ContextCore, ContextQuery, ContextSignals, GraphEdge, GraphNode, ImportanceVector, NodeBranches, NodeContext, NodeCore, PersonalityCore, SearchHit, SpeechStyleCore, SystemCore
from core.graph_core import GraphMemory


class DummySignalExtractor:
    def extract(self, request: ContextQuery) -> ContextSignals:
        return ContextSignals(entities=("psychology",), emotion=("tension",), logic=("pressure",), intent=("control",))


class DummyGraphSearch:
    def search(self, request: ContextQuery, signals: ContextSignals):
        return [SearchHit(node_id="pattern:guilt_pressure", score=2.4, reasons=("signal:control", "entity:psychology"))]


class DummyHydration:
    def __init__(self, node: GraphNode) -> None:
        self.node = node

    def hydrate(self, node_ids):
        return {self.node.node_id: self.node}


class DummyExpansion:
    def __init__(self, node: GraphNode) -> None:
        self.node = node

    def expand(self, node_ids, hydrated_nodes, max_nodes):
        return [self.node], [GraphEdge(src_id=self.node.node_id, dst_id="domain:psychology", edge_type="IS_PART_OF", weight=0.9)]


def test_node_core_is_immutable_and_branches_are_replaced_not_mutated() -> None:
    node = GraphNode(
        node_id="concept:psychology",
        node_core=NodeCore(
            node_id="concept:psychology",
            node_type="CONCEPT",
            name="Psychology",
            description="Practical study of human behavior and mental processes.",
            importance_vector=ImportanceVector(0.8, 0.7, 0.4, 0.9),
        ),
        node_branches=NodeBranches(patterns=("pattern:projection",), examples=("example:1",), signals=("signal:tension",), relations=("domain:psychology",)),
        node_context=NodeContext(domains=("domain:psychology",), tags=("behavior",)),
    )
    with pytest.raises(FrozenInstanceError):
        node.node_core.name = "Broken"  # type: ignore[misc]
    updated = node.with_branches(node.node_branches.merged(examples=("example:1", "example:2")))
    assert node.node_branches.examples == ("example:1",)
    assert updated.node_branches.examples == ("example:1", "example:2")


def test_personality_core_changes_slowly() -> None:
    base = PersonalityCore(
        temperament="guarded",
        values=("clarity",),
        speech_style=SpeechStyleCore(formality=0.4, slang_level=0.2, directness=0.6, profanity_tolerance=0.1),
        reasoning_style="grounded",
        risk_tolerance=0.3,
        aggression_level=0.2,
        humor_level=0.1,
    )
    updated = base.gradual_update(risk_tolerance=1.0, aggression_level=0.9, humor_level=0.8, max_step=0.05)
    assert round(updated.risk_tolerance - base.risk_tolerance, 6) == 0.05
    assert round(updated.aggression_level - base.aggression_level, 6) == 0.05
    assert round(updated.humor_level - base.humor_level, 6) == 0.05


def test_context_core_builds_ram_graph_from_ports() -> None:
    node = GraphNode(
        node_id="pattern:guilt_pressure",
        node_core=NodeCore(
            node_id="pattern:guilt_pressure",
            node_type="PATTERN",
            name="Guilt pressure",
            description="Pressure framed as proof of love or loyalty.",
        ),
    )
    context_core = ContextCore()
    ram = context_core.build_ram_context(
        ContextQuery(query="Is this guilt pressure?", recent_history=("Family conflict came first.",), max_nodes=6),
        signal_extractor=DummySignalExtractor(),
        graph_search=DummyGraphSearch(),
        graph_hydration=DummyHydration(node),
        branch_expander=DummyExpansion(node),
    )
    assert ram.query == "Is this guilt pressure?"
    assert ram.ranked_nodes[0].node.node_id == "pattern:guilt_pressure"
    assert ram.edges[0].edge_type == "IS_PART_OF"
    assert ram.signals["query_signals"]["intent"] == ["control"]


def test_agent_core_cannot_mutate_core_scope() -> None:
    agent = AgentCore(agent_id="agent:analyst", name="Analyst", purpose="Reads graph evidence.")
    with pytest.raises(PermissionError):
        agent.validate_proposal(AgentMutationProposal(target_scope="node_core", operation="rewrite"))
    agent.validate_proposal(AgentMutationProposal(target_scope="node_branches", operation="append_example"))


def test_system_core_describes_single_brain_contract() -> None:
    node = GraphNode(
        node_id="domain:psychology",
        node_core=NodeCore(
            node_id="domain:psychology",
            node_type="DOMAIN",
            name="Psychology",
            description="Behavior-focused knowledge domain.",
        ),
    )
    system = SystemCore(
        graph_memory=GraphMemory(nodes={node.node_id: node}, edges=()),
        personalities={"person:observer": PersonalityCore(temperament="calm")},
        agents={"agent:analyst": AgentCore(agent_id="agent:analyst", name="Analyst", purpose="Reads graph evidence.")},
    )
    summary = system.describe()
    assert summary["graph_nodes"] == 1
    assert summary["personalities"] == ["person:observer"]
    assert "node_core is immutable" in summary["principles"]
