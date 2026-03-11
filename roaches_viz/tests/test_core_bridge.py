from __future__ import annotations

from pathlib import Path

from roaches_viz.foundations import load_builtin_foundation, seed_foundation
from roaches_viz.graph_rag import build_ram_graph, generate_behavioral_dialogue
from roaches_viz.store import GraphStore


def test_ram_graph_and_dialogue_expose_core_contract(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "graph.sqlite3")
    try:
        result = seed_foundation(store, load_builtin_foundation("psychology_foundations"), replace_graph=True)
        assert result["ok"] is True
        payload = store.export_graph()
        ram = build_ram_graph(
            payload,
            query="Is this guilt pressure in family conflict?",
            recent_history=["Criticism came first."],
            person_id="person:psychologist",
        )
        assert ram["ram_core"]["query"]
        assert ram["ram_core"]["nodes"]
        dialogue = generate_behavioral_dialogue(
            payload,
            query="How would this person react to repeated guilt pressure?",
            recent_history=["The pressure appeared after criticism."],
            person_id="person:psychologist",
            llm_role="general",
        )
        assert dialogue["core"]["system"]["graph_nodes"] >= 1
        assert dialogue["core"]["ram"]["ranked_nodes"]
    finally:
        store.close()
