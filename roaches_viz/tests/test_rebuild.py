from __future__ import annotations

from pathlib import Path

from roaches_viz.ingest import ingest_text
from roaches_viz.rebuild import rebuild_graph_from_sources
from roaches_viz.store import GraphStore


def _canonical_export(store: GraphStore) -> tuple[tuple[tuple[str, str, str], ...], tuple[tuple[str, str, str, float], ...]]:
    payload = store.export_graph()
    nodes = tuple(
        (node["id"], node["type"], node["name"])
        for node in payload["nodes"]
    )
    edges = tuple(
        (edge["src_id"], edge["dst_id"], edge["type"], float(edge["weight"]))
        for edge in payload["edges"]
    )
    return nodes, edges


def test_rebuild_full_is_deterministic_for_behavioral_graph(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "graph.sqlite3")
    try:
        ingest_text(store, "src:a", "Psychology note. John said if you loved me you would do this.", top_tokens_per_sentence=3)
        ingest_text(store, "src:b", "Family conflict often triggers guilt pressure after criticism.", top_tokens_per_sentence=3)
        first = rebuild_graph_from_sources(store, mode="full", top_tokens_per_sentence=3)
        second = rebuild_graph_from_sources(store, mode="full", top_tokens_per_sentence=3)
        assert first["ok"] is True
        assert second["ok"] is True
        assert _canonical_export(store) == _canonical_export(store)
    finally:
        store.close()


def test_rebuild_scoped_requires_source_ids(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "graph.sqlite3")
    try:
        ingest_text(store, "src:a", "A short note about boundaries and pressure.", top_tokens_per_sentence=3)
        result = rebuild_graph_from_sources(store, mode="scoped", source_ids=[], top_tokens_per_sentence=3)
        assert result["ok"] is False
        assert "source_ids" in str(result.get("error"))
    finally:
        store.close()
