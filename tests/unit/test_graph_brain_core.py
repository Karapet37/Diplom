from __future__ import annotations

from core import GraphInitializer, GraphTraversalEngine
from roaches_viz.foundations import load_builtin_foundation
from roaches_viz.roaches_viz.core_bridge import graph_payload_to_memory


def test_graph_brain_demo_initializer_has_expected_layer_counts() -> None:
    dataset = load_builtin_foundation("graph_brain_demo")
    result = GraphInitializer().initialize({"nodes": dataset["nodes"], "edges": dataset["edges"]})
    assert result.stats["root_count"] == 1
    assert result.stats["concept_count"] == 10
    assert result.stats["pattern_count"] == 20
    assert result.stats["example_count"] == 50
    assert "root:graph_brain" in result.layer_index["root_core"]


def test_graph_traversal_engine_builds_ram_graph_from_small_query() -> None:
    dataset = load_builtin_foundation("graph_brain_demo")
    memory = graph_payload_to_memory({"nodes": dataset["nodes"], "edges": dataset["edges"]})
    traversal = GraphTraversalEngine().traverse(
        memory,
        query="Is this guilt pressure after criticism in family conflict?",
        recent_history=("The pressure came after a refusal.",),
        max_nodes=12,
    )
    top_ids = {item.node.node_id for item in traversal.ranked_nodes[:6]}
    assert "pattern:guilt_pressure" in top_ids
    assert any(node_id.startswith("concept:") for node_id in top_ids)
    assert traversal.ram_graph.nodes
    assert traversal.ram_graph.edges
