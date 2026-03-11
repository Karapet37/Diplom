from __future__ import annotations

from pathlib import Path

from roaches_viz.concurrency.graph_actor import GraphActor
from roaches_viz.graph_model import Node


def test_graph_hygiene_tick_makes_node_text_human_readable(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.sqlite3"
    actor = GraphActor(db_path=db_path, top_tokens_per_sentence=4)
    actor._store.upsert_node(
        Node(
            id="concept:graph_db",
            type="CONCEPT",
            label="Графовая база данных описание",
            short_gloss="",
            plain_explanation=(
                "разновидность баз данных с реализацией сетевой модели в виде графа и его обобщений. "
                "Графовая СУБД — система управления графовыми базами данных. "
                "Модель хранения информации в виде графов сложилась в 1990—2000 годах[1]."
            ),
            examples_json="[]",
            tags_json='["database","graph"]',
        )
    )
    actor._store.conn.commit()
    actor._swap_snapshot()
    actor.start()
    try:
        result = actor.ask("graph_hygiene_tick", {"reason": "test"})
        assert result["ok"] is True
        assert result["maintenance"]["last_hygiene_reason"] == "test"
        assert result["maintenance"]["last_hygiene_updates"]["updated_nodes"] == 1
        health = actor.ask("health")
        assert health["maintenance"]["last_hygiene_reason"] == "test"
        node = next(item for item in actor.ask("snapshot", {})["nodes"] if item["id"] == "concept:graph_db")
        assert node["label"] == "Графовая база данных"
        assert "[1]" not in node["plain_explanation"]
        assert node["short_gloss"].startswith("разновидность баз данных")
    finally:
        actor.shutdown()
