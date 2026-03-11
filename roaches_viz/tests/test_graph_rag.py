from __future__ import annotations

from pathlib import Path

from roaches_viz.concurrency.graph_actor import GraphActor
from roaches_viz.graph_rag import build_ram_graph
from roaches_viz.ingest import ingest_text
from roaches_viz.store import GraphStore


def test_build_ram_graph_ranks_behavioral_context(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "graph.sqlite3")
    try:
        ingest_text(
            store,
            source_id="src:ram",
            text='Psychology note. Maria said: "If you loved me, you would do this." Family conflict keeps returning after criticism.',
            top_tokens_per_sentence=3,
        )
        payload = store.export_graph()
        ram = build_ram_graph(payload, query="Is this guilt pressure in family conflict?", recent_history=["criticism came first"])
        assert ram["ranked_context"]
        top_ids = {item["node_id"] for item in ram["ranked_context"][:6]}
        assert any(node_id.startswith("pattern:psychology:guilt_pressure") for node_id in top_ids)
        assert any(node_id.startswith("signal:psychology:") for node_id in top_ids)
    finally:
        store.close()


def test_graph_actor_keeps_system_agents_after_replace_graph(tmp_path: Path) -> None:
    actor = GraphActor(db_path=tmp_path / "graph.sqlite3", top_tokens_per_sentence=3)
    actor.start()
    try:
        seeded = actor.ask("seed_series", {"dataset_id": "psychology_foundations", "replace_graph": True})
        assert seeded["ok"] is True
        graph = actor.ask("snapshot", {})
        assert any(node["type"] == "AGENT" for node in graph["nodes"])
    finally:
        actor.shutdown()
