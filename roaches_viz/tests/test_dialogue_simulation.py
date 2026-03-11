from __future__ import annotations

from pathlib import Path

import roaches_viz.graph_rag as graph_rag
from roaches_viz.foundations import load_builtin_foundation, seed_foundation
from roaches_viz.store import GraphStore


def test_generate_behavioral_dialogue_writes_person_log(tmp_path: Path, monkeypatch) -> None:
    store = GraphStore(tmp_path / "graph.sqlite3")
    try:
        seeded = seed_foundation(store, load_builtin_foundation("psychology_foundations"), replace_graph=True)
        assert seeded["ok"] is True
        payload = store.export_graph()
        monkeypatch.setattr(graph_rag, "build_reasoning_llm", lambda *_args, **_kwargs: None)

        result = graph_rag.generate_behavioral_dialogue(
            payload,
            query="How would this person react to repeated guilt pressure?",
            recent_history=["The pressure appeared after criticism."],
            person_id="person:psychologist",
            llm_role="general",
        )

        assert result["assistant_reply"]
        assert result["personality"]["enabled"] is True
        log_path = Path(str(result["personality"]["log_path"]))
        assert log_path.exists()
        assert "person_psychologist" in log_path.name
    finally:
        store.close()
