from __future__ import annotations

import sqlite3
from pathlib import Path


BASE_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS schema_migrations (
  version TEXT PRIMARY KEY,
  applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS nodes (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  name TEXT NOT NULL DEFAULT '',
  label TEXT NOT NULL DEFAULT '',
  description TEXT NOT NULL DEFAULT '',
  short_gloss TEXT NOT NULL DEFAULT '',
  plain_explanation TEXT NOT NULL DEFAULT '',
  what_it_is TEXT NOT NULL DEFAULT '',
  how_it_works TEXT NOT NULL DEFAULT '',
  how_to_recognize TEXT NOT NULL DEFAULT '',
  examples_json TEXT NOT NULL DEFAULT '[]',
  tags_json TEXT NOT NULL DEFAULT '[]',
  speech_patterns_json TEXT NOT NULL DEFAULT '[]',
  behavior_patterns_json TEXT NOT NULL DEFAULT '[]',
  triggers_json TEXT NOT NULL DEFAULT '[]',
  values_json TEXT NOT NULL DEFAULT '[]',
  preferences_json TEXT NOT NULL DEFAULT '[]',
  reaction_logic_json TEXT NOT NULL DEFAULT '[]',
  tolerance_thresholds_json TEXT NOT NULL DEFAULT '{}',
  conflict_patterns_json TEXT NOT NULL DEFAULT '[]',
  background TEXT NOT NULL DEFAULT '',
  profession TEXT NOT NULL DEFAULT '',
  speech_style_json TEXT NOT NULL DEFAULT '{}',
  temperament TEXT NOT NULL DEFAULT '',
  tolerance_threshold REAL NOT NULL DEFAULT 0.5,
  formality REAL NOT NULL DEFAULT 0.5,
  slang_level REAL NOT NULL DEFAULT 0.3,
  directness REAL NOT NULL DEFAULT 0.5,
  profanity_tolerance REAL NOT NULL DEFAULT 0.1,
  possible_intents_json TEXT NOT NULL DEFAULT '[]',
  emotion_signals_json TEXT NOT NULL DEFAULT '[]',
  conflict_level REAL NOT NULL DEFAULT 0.0,
  irony_probability REAL NOT NULL DEFAULT 0.0,
  logic_weight REAL NOT NULL DEFAULT 0.5,
  emotion_weight REAL NOT NULL DEFAULT 0.5,
  risk_weight REAL NOT NULL DEFAULT 0.5,
  relevance_weight REAL NOT NULL DEFAULT 0.5,
  confidence REAL NOT NULL DEFAULT 0.7,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS edges (
  src_id TEXT NOT NULL,
  dst_id TEXT NOT NULL,
  type TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 1.0,
  confidence REAL NOT NULL DEFAULT 0.7,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (src_id, dst_id, type),
  FOREIGN KEY (src_id) REFERENCES nodes(id) ON DELETE CASCADE,
  FOREIGN KEY (dst_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS examples (
  example_id TEXT PRIMARY KEY,
  node_id TEXT NOT NULL,
  text TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS example_sources (
  example_id TEXT NOT NULL,
  source_id TEXT NOT NULL,
  snippet_text TEXT NOT NULL,
  offset_start INTEGER NOT NULL DEFAULT 0,
  offset_end INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (example_id, source_id, snippet_text),
  FOREIGN KEY (example_id) REFERENCES examples(example_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tags (
  node_id TEXT NOT NULL,
  tag TEXT NOT NULL,
  PRIMARY KEY (node_id, tag),
  FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS domains (
  domain_id TEXT PRIMARY KEY,
  node_id TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sources (
  source_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  raw_text TEXT NOT NULL
);


CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_examples_node_id ON examples(node_id);
CREATE INDEX IF NOT EXISTS idx_example_sources_source_id ON example_sources(source_id);
CREATE INDEX IF NOT EXISTS idx_tags_node_id ON tags(node_id);
"""

MIGRATIONS: list[tuple[str, list[str]]] = [
    (
        "0001_behavioral_schema",
        [
            BASE_SCHEMA,
        ],
    ),
    (
        "0002_nodes_new_fields",
        [
            "ALTER TABLE nodes ADD COLUMN name TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE nodes ADD COLUMN description TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE nodes ADD COLUMN what_it_is TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE nodes ADD COLUMN how_it_works TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE nodes ADD COLUMN how_to_recognize TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE nodes ADD COLUMN speech_patterns_json TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE nodes ADD COLUMN behavior_patterns_json TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE nodes ADD COLUMN triggers_json TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE nodes ADD COLUMN values_json TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE nodes ADD COLUMN preferences_json TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE nodes ADD COLUMN confidence REAL NOT NULL DEFAULT 0.7",
        ],
    ),
    (
        "0003_edges_metadata",
        [
            "ALTER TABLE edges ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'",
        ],
    ),
    (
        "0004_runtime_support_tables",
        [
            BASE_SCHEMA,
        ],
    ),
    (
        "0005_node_importance_and_personality_fields",
        [
            "ALTER TABLE nodes ADD COLUMN reaction_logic_json TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE nodes ADD COLUMN tolerance_thresholds_json TEXT NOT NULL DEFAULT '{}'",
            "ALTER TABLE nodes ADD COLUMN conflict_patterns_json TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE nodes ADD COLUMN logic_weight REAL NOT NULL DEFAULT 0.5",
            "ALTER TABLE nodes ADD COLUMN emotion_weight REAL NOT NULL DEFAULT 0.5",
            "ALTER TABLE nodes ADD COLUMN risk_weight REAL NOT NULL DEFAULT 0.5",
            "ALTER TABLE nodes ADD COLUMN relevance_weight REAL NOT NULL DEFAULT 0.5",
        ],
    ),
    (
        "0006_behavior_profiles_and_phrase_fields",
        [
            "ALTER TABLE nodes ADD COLUMN background TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE nodes ADD COLUMN profession TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE nodes ADD COLUMN speech_style_json TEXT NOT NULL DEFAULT '{}'",
            "ALTER TABLE nodes ADD COLUMN temperament TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE nodes ADD COLUMN tolerance_threshold REAL NOT NULL DEFAULT 0.5",
            "ALTER TABLE nodes ADD COLUMN formality REAL NOT NULL DEFAULT 0.5",
            "ALTER TABLE nodes ADD COLUMN slang_level REAL NOT NULL DEFAULT 0.3",
            "ALTER TABLE nodes ADD COLUMN directness REAL NOT NULL DEFAULT 0.5",
            "ALTER TABLE nodes ADD COLUMN profanity_tolerance REAL NOT NULL DEFAULT 0.1",
            "ALTER TABLE nodes ADD COLUMN possible_intents_json TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE nodes ADD COLUMN emotion_signals_json TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE nodes ADD COLUMN conflict_level REAL NOT NULL DEFAULT 0.0",
            "ALTER TABLE nodes ADD COLUMN irony_probability REAL NOT NULL DEFAULT 0.0",
        ],
    ),
]


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _column_names(conn: sqlite3.Connection, table_name: str) -> set[str]:
    if not _table_exists(conn, table_name):
        return set()
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()}


def _run_statement_if_possible(conn: sqlite3.Connection, statement: str) -> None:
    lowered = statement.strip().lower()
    if lowered.startswith("alter table nodes add column"):
        column_name = statement.split("ADD COLUMN", 1)[1].strip().split()[0]
        if column_name in _column_names(conn, "nodes"):
            return
    if lowered.startswith("alter table edges add column"):
        column_name = statement.split("ADD COLUMN", 1)[1].strip().split()[0]
        if column_name in _column_names(conn, "edges"):
            return
    conn.execute(statement)


def _backfill_behavioral_fields(conn: sqlite3.Connection) -> None:
    node_columns = _column_names(conn, "nodes")
    if not node_columns:
        return
    if {"name", "label"} <= node_columns:
        conn.execute("UPDATE nodes SET name = COALESCE(NULLIF(name, ''), label, id)")
        conn.execute("UPDATE nodes SET label = COALESCE(NULLIF(label, ''), name, id)")
    if {"description", "short_gloss"} <= node_columns:
        conn.execute("UPDATE nodes SET description = COALESCE(NULLIF(description, ''), short_gloss, plain_explanation, label)")
        conn.execute("UPDATE nodes SET short_gloss = COALESCE(NULLIF(short_gloss, ''), description, label)")
    if {"what_it_is", "plain_explanation"} <= node_columns:
        conn.execute("UPDATE nodes SET what_it_is = COALESCE(NULLIF(what_it_is, ''), plain_explanation, description)")
        conn.execute("UPDATE nodes SET plain_explanation = COALESCE(NULLIF(plain_explanation, ''), what_it_is, how_it_works, description)")
    if {"logic_weight", "emotion_weight", "risk_weight", "relevance_weight"} <= node_columns:
        conn.execute(
            """
            UPDATE nodes
            SET
              logic_weight = CASE
                WHEN type IN ('CONCEPT', 'DOMAIN', 'AGENT') AND logic_weight = 0.5 THEN 0.7
                ELSE logic_weight
              END,
              emotion_weight = CASE
                WHEN type IN ('PERSON', 'TRAIT', 'SIGNAL') AND emotion_weight = 0.5 THEN 0.7
                ELSE emotion_weight
              END,
              risk_weight = CASE
                WHEN type IN ('PATTERN', 'TRIGGER', 'SIGNAL') AND risk_weight = 0.5 THEN 0.68
                ELSE risk_weight
              END,
              relevance_weight = CASE
                WHEN relevance_weight = 0.5 THEN 0.6
                ELSE relevance_weight
              END
            """
        )
    if {"tolerance_threshold", "formality", "slang_level", "directness", "profanity_tolerance"} <= node_columns:
        conn.execute(
            """
            UPDATE nodes
            SET
              tolerance_threshold = COALESCE(tolerance_threshold, 0.5),
              formality = COALESCE(formality, 0.5),
              slang_level = COALESCE(slang_level, 0.3),
              directness = COALESCE(directness, 0.5),
              profanity_tolerance = COALESCE(profanity_tolerance, 0.1)
            """
        )


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(BASE_SCHEMA)
    applied = {
        str(row["version"])
        for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
    }
    for version, statements in MIGRATIONS:
        if version in applied:
            continue
        for statement in statements:
            if statement.strip():
                if "CREATE TABLE IF NOT EXISTS" in statement or "CREATE INDEX IF NOT EXISTS" in statement or "PRAGMA " in statement:
                    conn.executescript(statement)
                else:
                    _run_statement_if_possible(conn, statement)
        conn.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES(?)", (version,))
    _backfill_behavioral_fields(conn)
    conn.commit()


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn
