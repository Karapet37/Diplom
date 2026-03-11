from __future__ import annotations

import sqlite3
from pathlib import Path

from roaches_viz.controller import GraphMutationController
from roaches_viz.ingest import ingest_text
from roaches_viz.store import GraphStore


def test_behavioral_ingest_creates_domain_pattern_example_person_nodes(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "graph.sqlite3")
    try:
        result = ingest_text(
            store,
            source_id="src:test",
            text='Psychology note. John said: "If you loved me you would do this." This pressure appears after criticism in family conflict.',
            top_tokens_per_sentence=3,
        )
        assert result["nodes"] > 0
        graph = store.export_graph()
        node_types = {node["type"] for node in graph["nodes"]}
        assert "DOMAIN" in node_types
        assert "PATTERN" in node_types
        assert "EXAMPLE" in node_types
        assert "PERSON" in node_types
        assert "SIGNAL" in node_types
        assert "TRAIT" in node_types
        assert any(node["id"] == "domain:psychology" for node in graph["nodes"])
        assert any(edge["type"] == "SAID_EXAMPLE" for edge in graph["edges"])
        assert any(edge["type"] == "EXAMPLE_OF" for edge in graph["edges"])
        assert any(edge["type"] == "RELATED_TO" for edge in graph["edges"])
        assert any(edge["type"] == "USES_PATTERN" for edge in graph["edges"])
    finally:
        store.close()


def test_store_creates_domain_and_tag_indexes(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "graph.sqlite3")
    try:
        ingest_text(
            store,
            source_id="src:psych",
            text="Psychology and family systems help explain repeated guilt pressure.",
            top_tokens_per_sentence=3,
        )
        domains = store.list_domains()
        assert any(item["domain_id"] == "domain:psychology" for item in domains)
        search = store.search_nodes("guilt")
        assert search
        assert any("guilt" in (node["name"] + " " + node["description"]).lower() for node in search)
    finally:
        store.close()


def test_schema_migrations_add_behavioral_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE nodes (
          id TEXT PRIMARY KEY,
          type TEXT NOT NULL,
          label TEXT NOT NULL,
          short_gloss TEXT NOT NULL DEFAULT '',
          plain_explanation TEXT NOT NULL DEFAULT '',
          examples_json TEXT NOT NULL DEFAULT '[]',
          tags_json TEXT NOT NULL DEFAULT '[]',
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE edges (
          src_id TEXT NOT NULL,
          dst_id TEXT NOT NULL,
          type TEXT NOT NULL,
          weight REAL NOT NULL DEFAULT 1.0,
          confidence REAL NOT NULL DEFAULT 0.7,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (src_id, dst_id, type)
        )
        """
    )
    conn.commit()
    conn.close()

    store = GraphStore(db_path)
    try:
        node_columns = {row["name"] for row in store.conn.execute("PRAGMA table_info('nodes')").fetchall()}
        edge_columns = {row["name"] for row in store.conn.execute("PRAGMA table_info('edges')").fetchall()}
        assert {"name", "description", "what_it_is", "how_it_works", "how_to_recognize", "confidence"} <= node_columns
        assert "metadata_json" in edge_columns
    finally:
        store.close()


def test_controller_blocks_radical_mutation() -> None:
    controller = GraphMutationController(max_nodes_per_request=4, max_edges_per_request=6)
    allowed = controller.validate_ai_change(nodes_to_add=3, edges_to_add=4)
    blocked = controller.validate_ai_change(nodes_to_add=8, edges_to_add=4)
    assert allowed.ok is True
    assert blocked.ok is False
    assert blocked.reason == "too_many_nodes"
