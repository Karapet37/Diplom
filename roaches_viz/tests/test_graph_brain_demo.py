from __future__ import annotations

from pathlib import Path

from core import GraphInitializer
from roaches_viz.foundations import list_builtin_foundations, load_builtin_foundation, seed_foundation
from roaches_viz.store import GraphStore


def test_graph_brain_demo_dataset_is_listed_and_seedable(tmp_path: Path) -> None:
    assert "graph_brain_demo" in list_builtin_foundations()
    store = GraphStore(tmp_path / "graph.sqlite3")
    try:
        seeded = seed_foundation(store, load_builtin_foundation("graph_brain_demo"), replace_graph=True)
        assert seeded["ok"] is True
        assert seeded["graph_brain"]["stats"]["concept_count"] == 10
        assert seeded["graph_brain"]["stats"]["pattern_count"] == 20
        assert seeded["graph_brain"]["stats"]["example_count"] == 50
        payload = store.export_graph()
        init = GraphInitializer().initialize(payload)
        assert init.stats["concept_count"] == 10
        assert init.stats["pattern_count"] == 20
        assert init.stats["example_count"] == 50
    finally:
        store.close()
