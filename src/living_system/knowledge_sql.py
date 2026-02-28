"""SQLite knowledge layer for long-living autonomous agent state."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time
from typing import Any
import zlib

from src.living_system.models import FailureRecord, PromptRun, ReasoningTrace


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_load(value: Any, *, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


class KnowledgeSQLStore:
    """Durable SQL backbone for graph memory, diagnostics and evolution."""

    BASELINE_SCHEMA_VERSION = "2026.02.17"
    SCHEMA_VERSION = "2026.02.28"

    def __init__(self, db_path: str | Path = "data/living_system.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    @staticmethod
    def _build_profile_summary(
        *,
        display_name: str,
        primary_language: str,
        secondary_languages: list[str],
        preferences: dict[str, Any],
        behavior_model: dict[str, Any],
    ) -> str:
        parts = [display_name.strip() or "user", f"primary={primary_language.strip() or 'hy'}"]
        if secondary_languages:
            parts.append(f"secondary={','.join(secondary_languages[:4])}")
        if preferences:
            parts.append(f"preferences={len(preferences)}")
        if behavior_model:
            parts.append(f"behavior={len(behavior_model)}")
        return " | ".join(parts)[:280]

    @staticmethod
    def _build_trace_summary(input_text: str, output_text: str) -> str:
        input_head = " ".join(str(input_text or "").split())[:120].strip()
        output_head = " ".join(str(output_text or "").split())[:120].strip()
        parts = []
        if input_head:
            parts.append(f"in={input_head}")
        if output_head:
            parts.append(f"out={output_head}")
        return " | ".join(parts)[:280]

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        return {str(row["name"]) for row in rows}

    @classmethod
    def _has_column(cls, conn: sqlite3.Connection, table: str, column: str) -> bool:
        return column in cls._table_columns(conn, table)

    @staticmethod
    def _applied_schema_versions(conn: sqlite3.Connection) -> set[str]:
        rows = conn.execute("SELECT version FROM schema_versions ORDER BY id ASC").fetchall()
        return {str(row["version"]) for row in rows}

    @staticmethod
    def _record_schema_version(conn: sqlite3.Connection, version: str) -> None:
        token = str(version or "").strip()
        if not token:
            return
        exists = conn.execute(
            "SELECT 1 FROM schema_versions WHERE version = ? LIMIT 1",
            (token,),
        ).fetchone()
        if exists:
            return
        conn.execute(
            "INSERT INTO schema_versions(version, applied_at) VALUES(?, ?)",
            (token, time.time()),
        )

    def _apply_pending_migrations(self, conn: sqlite3.Connection) -> None:
        applied = self._applied_schema_versions(conn)
        if not applied:
            self._record_schema_version(conn, self.BASELINE_SCHEMA_VERSION)
            applied = self._applied_schema_versions(conn)

        migrations: tuple[tuple[str, Any], ...] = (
            (self.SCHEMA_VERSION, self._migration_2026_02_28),
        )
        for version, migration_fn in migrations:
            if version in applied:
                continue
            migration_fn(conn)
            self._record_schema_version(conn, version)
            applied.add(version)

    def _migration_2026_02_28(self, conn: sqlite3.Connection) -> None:
        if not self._has_column(conn, "users", "last_active_at"):
            conn.execute("ALTER TABLE users ADD COLUMN last_active_at REAL NOT NULL DEFAULT 0")
        conn.execute(
            "UPDATE users SET last_active_at = updated_at WHERE COALESCE(last_active_at, 0) <= 0"
        )

        if not self._has_column(conn, "user_profiles", "profile_summary"):
            conn.execute("ALTER TABLE user_profiles ADD COLUMN profile_summary TEXT NOT NULL DEFAULT ''")
        conn.execute(
            """
            UPDATE user_profiles
            SET profile_summary = printf(
                '%s | primary=%s',
                CASE WHEN trim(user_id) = '' THEN 'user' ELSE user_id END,
                CASE WHEN trim(primary_language) = '' THEN 'hy' ELSE primary_language END
            )
            WHERE trim(COALESCE(profile_summary, '')) = ''
            """
        )

        if not self._has_column(conn, "nodes", "source_tag"):
            conn.execute("ALTER TABLE nodes ADD COLUMN source_tag TEXT NOT NULL DEFAULT ''")
        conn.execute(
            """
            UPDATE nodes
            SET source_tag = CASE
                WHEN trim(COALESCE(node_type, '')) = '' THEN 'generic'
                ELSE node_type
            END
            WHERE trim(COALESCE(source_tag, '')) = ''
            """
        )

        if not self._has_column(conn, "edges", "status"):
            conn.execute("ALTER TABLE edges ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
        conn.execute(
            """
            UPDATE edges
            SET status = 'active'
            WHERE trim(COALESCE(status, '')) = ''
            """
        )

        if not self._has_column(conn, "reasoning_traces", "trace_summary"):
            conn.execute("ALTER TABLE reasoning_traces ADD COLUMN trace_summary TEXT NOT NULL DEFAULT ''")
        trace_rows = conn.execute(
            "SELECT trace_id, input_text, output_text, trace_summary FROM reasoning_traces"
        ).fetchall()
        for row in trace_rows:
            existing_summary = str(row["trace_summary"] or "").strip()
            if existing_summary:
                continue
            summary = self._build_trace_summary(
                str(row["input_text"] or ""),
                str(row["output_text"] or ""),
            )
            conn.execute(
                "UPDATE reasoning_traces SET trace_summary = ? WHERE trace_id = ?",
                (summary, int(row["trace_id"])),
            )

    def initialize(self) -> None:
        schema_statements = [
            """
            CREATE TABLE IF NOT EXISTS schema_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                applied_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL DEFAULT '',
                last_active_at REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                primary_language TEXT NOT NULL DEFAULT 'hy',
                secondary_languages_json TEXT NOT NULL DEFAULT '[]',
                preferences_json TEXT NOT NULL DEFAULT '{}',
                behavior_model_json TEXT NOT NULL DEFAULT '{}',
                timeline_json TEXT NOT NULL DEFAULT '[]',
                profile_summary TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT '',
                node_type TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0,
                version INTEGER NOT NULL DEFAULT 1,
                source_tag TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS node_properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                prop_key TEXT NOT NULL,
                prop_value_json TEXT NOT NULL,
                language_code TEXT NOT NULL DEFAULT '',
                version INTEGER NOT NULL DEFAULT 1,
                updated_at REAL NOT NULL,
                FOREIGN KEY (node_id) REFERENCES nodes(node_id) ON DELETE CASCADE
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_node_props_lookup
            ON node_properties(node_id, prop_key, language_code, version)
            """,
            """
            CREATE TABLE IF NOT EXISTS edges (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT '',
                from_node TEXT NOT NULL,
                to_node TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                weight REAL NOT NULL DEFAULT 0.0,
                version INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'active',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (from_node) REFERENCES nodes(node_id) ON DELETE CASCADE,
                FOREIGN KEY (to_node) REFERENCES nodes(node_id) ON DELETE CASCADE
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_edges_lookup
            ON edges(user_id, from_node, to_node, relation_type, version)
            """,
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_type TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component TEXT NOT NULL,
                version TEXT NOT NULL,
                checksum TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT NOT NULL,
                component TEXT NOT NULL,
                message TEXT NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}',
                user_id TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_logs_component_created
            ON logs(component, created_at)
            """,
            """
            CREATE TABLE IF NOT EXISTS errors (
                error_id INTEGER PRIMARY KEY AUTOINCREMENT,
                signature TEXT NOT NULL UNIQUE,
                error_type TEXT NOT NULL,
                message TEXT NOT NULL,
                traceback TEXT NOT NULL,
                component TEXT NOT NULL,
                severity TEXT NOT NULL,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 1,
                last_snapshot_id INTEGER,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_type TEXT NOT NULL,
                user_id TEXT NOT NULL DEFAULT '',
                state_json TEXT NOT NULL,
                checksum TEXT NOT NULL,
                parent_snapshot_id INTEGER,
                created_at REAL NOT NULL,
                FOREIGN KEY (parent_snapshot_id) REFERENCES snapshots(snapshot_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS reasoning_traces (
                trace_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT '',
                session_id TEXT NOT NULL,
                input_text TEXT NOT NULL,
                output_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                trace_summary TEXT NOT NULL DEFAULT '',
                trace_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS localization_texts (
                language_code TEXT NOT NULL,
                text_key TEXT NOT NULL,
                text_value TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (language_code, text_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS prompts (
                prompt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version INTEGER NOT NULL,
                language_code TEXT NOT NULL DEFAULT 'en',
                template_text TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(name, version, language_code)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS prompt_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id INTEGER NOT NULL,
                user_id TEXT NOT NULL DEFAULT '',
                input_json TEXT NOT NULL,
                output_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (prompt_id) REFERENCES prompts(prompt_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS recovery_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS health_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                layer_name TEXT NOT NULL,
                status TEXT NOT NULL,
                score REAL NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS audit_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT '',
                action_type TEXT NOT NULL,
                target_path TEXT NOT NULL,
                content_checksum TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            )
            """,
        ]

        with self._connect() as conn:
            for statement in schema_statements:
                conn.execute(statement)
            self._apply_pending_migrations(conn)
            conn.commit()

    def ensure_localized_text(self, language_code: str, text_key: str, text_value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO localization_texts(language_code, text_key, text_value, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(language_code, text_key)
                DO UPDATE SET text_value=excluded.text_value, updated_at=excluded.updated_at
                """,
                (language_code, text_key, text_value, time.time()),
            )
            conn.commit()

    def get_localized_text(self, text_key: str, *, language_code: str, fallback: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT text_value FROM localization_texts WHERE language_code = ? AND text_key = ?",
                (language_code, text_key),
            ).fetchone()
            if row:
                return str(row["text_value"])
            row = conn.execute(
                "SELECT text_value FROM localization_texts WHERE language_code = 'hy' AND text_key = ?",
                (text_key,),
            ).fetchone()
            if row:
                return str(row["text_value"])
            return fallback

    def upsert_user_profile(self, payload: dict[str, Any]) -> None:
        user_id = str(payload.get("user_id", "")).strip()
        if not user_id:
            raise ValueError("user_id is required")
        display_name = str(payload.get("display_name", user_id)).strip()
        primary_language = str(payload.get("primary_language", "hy") or "hy").strip() or "hy"
        secondary = payload.get("secondary_languages", ["ru", "en", "pt", "ar", "zh"])
        if not isinstance(secondary, (list, tuple, set)):
            secondary = []
        secondary_values = [str(item) for item in secondary]
        preferences = dict(payload.get("preferences", {}) or {})
        behavior_model = dict(payload.get("behavior_model", {}) or {})
        profile_summary = str(payload.get("profile_summary", "") or "").strip() or self._build_profile_summary(
            display_name=display_name,
            primary_language=primary_language,
            secondary_languages=secondary_values,
            preferences=preferences,
            behavior_model=behavior_model,
        )
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO users(user_id, display_name, last_active_at, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    display_name=excluded.display_name,
                    last_active_at=excluded.last_active_at,
                    updated_at=excluded.updated_at
                """,
                (user_id, display_name, now, now, now),
            )
            conn.execute(
                """
                INSERT INTO user_profiles(
                    user_id,
                    primary_language,
                    secondary_languages_json,
                    preferences_json,
                    behavior_model_json,
                    timeline_json,
                    profile_summary,
                    updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    primary_language=excluded.primary_language,
                    secondary_languages_json=excluded.secondary_languages_json,
                    preferences_json=excluded.preferences_json,
                    behavior_model_json=excluded.behavior_model_json,
                    timeline_json=excluded.timeline_json,
                    profile_summary=excluded.profile_summary,
                    updated_at=excluded.updated_at
                """,
                (
                    user_id,
                    primary_language,
                    _json_dump(secondary_values),
                    _json_dump(preferences),
                    _json_dump(behavior_model),
                    _json_dump(list(payload.get("timeline", []) or [])),
                    profile_summary,
                    now,
                ),
            )
            conn.commit()

    @staticmethod
    def _stable_node_id(user_id: str, node_type: str, display_name: str) -> str:
        token = f"{user_id}|{node_type}|{display_name}".strip().casefold()
        digest = int(zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF)
        return f"n_{digest:08x}"

    def upsert_node(self, payload: dict[str, Any]) -> str:
        user_id = str(payload.get("user_id", "")).strip()
        node_type = str(payload.get("node_type", "generic") or "generic").strip() or "generic"
        display_name = str(payload.get("display_name", "")).strip()
        node_id = str(payload.get("node_id", "")).strip() or self._stable_node_id(user_id, node_type, display_name)
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.5) or 0.5)))
        metadata = dict(payload.get("metadata", {}) or {})
        properties = dict(payload.get("properties", {}) or {})
        language_code = str(payload.get("language_code", "") or "").strip()
        source_tag = str(payload.get("source_tag", metadata.get("source", node_type)) or "").strip() or node_type
        now = time.time()

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS max_version FROM nodes WHERE node_id = ?",
                (node_id,),
            ).fetchone()
            version = int(existing["max_version"] or 0) + 1
            conn.execute(
                """
                INSERT INTO nodes(
                    node_id, user_id, node_type, display_name, confidence, version, source_tag, metadata_json, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    node_type=excluded.node_type,
                    display_name=excluded.display_name,
                    confidence=excluded.confidence,
                    version=excluded.version,
                    source_tag=excluded.source_tag,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (node_id, user_id, node_type, display_name, confidence, version, source_tag, _json_dump(metadata), now, now),
            )

            for key, value in properties.items():
                row = conn.execute(
                    """
                    SELECT COALESCE(MAX(version), 0) AS max_version
                    FROM node_properties
                    WHERE node_id = ? AND prop_key = ? AND language_code = ?
                    """,
                    (node_id, str(key), language_code),
                ).fetchone()
                prop_version = int(row["max_version"] or 0) + 1
                conn.execute(
                    """
                    INSERT INTO node_properties(node_id, prop_key, prop_value_json, language_code, version, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (node_id, str(key), _json_dump(value), language_code, prop_version, now),
                )
            conn.commit()
        return node_id

    def find_node(self, *, user_id: str, node_type: str, display_name: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT node_id, node_type, display_name, confidence, metadata_json
                , source_tag
                FROM nodes
                WHERE user_id = ? AND node_type = ? AND lower(display_name) = lower(?)
                ORDER BY version DESC
                LIMIT 1
                """,
                (user_id, node_type, display_name),
            ).fetchone()
            if not row:
                return None
            return {
                "node_id": str(row["node_id"]),
                "node_type": str(row["node_type"]),
                "display_name": str(row["display_name"]),
                "confidence": float(row["confidence"]),
                "source_tag": str(row["source_tag"] or ""),
                "metadata": _json_load(row["metadata_json"], default={}),
            }

    def upsert_edge(self, payload: dict[str, Any]) -> int:
        user_id = str(payload.get("user_id", "")).strip()
        from_node = str(payload.get("from_node", "")).strip()
        to_node = str(payload.get("to_node", "")).strip()
        relation_type = str(payload.get("relation_type", "related_to") or "related_to").strip() or "related_to"
        if not from_node or not to_node:
            raise ValueError("from_node and to_node are required")
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.5) or 0.5)))
        weight = max(0.0, min(1.0, float(payload.get("weight", confidence) or confidence)))
        metadata = dict(payload.get("metadata", {}) or {})
        status = str(payload.get("status", metadata.get("status", "active")) or "").strip() or "active"
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT edge_id, version
                FROM edges
                WHERE user_id = ? AND from_node = ? AND to_node = ? AND relation_type = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (user_id, from_node, to_node, relation_type),
            ).fetchone()
            version = 1
            if row:
                version = int(row["version"] or 0) + 1
            cursor = conn.execute(
                """
                INSERT INTO edges(
                    user_id, from_node, to_node, relation_type, confidence, weight, version, status, metadata_json, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, from_node, to_node, relation_type, confidence, weight, version, status, _json_dump(metadata), now, now),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def store_embedding(
        self,
        *,
        owner_type: str,
        owner_id: str,
        vector: list[float],
        model_name: str,
        version: int = 1,
    ) -> int:
        dims = len(vector)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO embeddings(owner_type, owner_id, model_name, vector_json, dimensions, version, created_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (owner_type, owner_id, model_name, _json_dump(vector), dims, version, time.time()),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def append_log(
        self,
        *,
        level: str,
        component: str,
        message: str,
        details: dict[str, Any] | None = None,
        user_id: str = "",
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO logs(level, component, message, details_json, user_id, created_at)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (level, component, message, _json_dump(dict(details or {})), user_id, time.time()),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def capture_failure(self, failure: FailureRecord) -> int:
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT error_id, hit_count FROM errors WHERE signature = ?",
                (failure.signature,),
            ).fetchone()
            if row:
                hit_count = int(row["hit_count"] or 0) + 1
                conn.execute(
                    """
                    UPDATE errors
                    SET message = ?, traceback = ?, severity = ?, component = ?, last_seen = ?, hit_count = ?
                    WHERE signature = ?
                    """,
                    (
                        failure.message,
                        failure.traceback,
                        failure.severity,
                        failure.component,
                        now,
                        hit_count,
                        failure.signature,
                    ),
                )
                conn.commit()
                return int(row["error_id"])
            cursor = conn.execute(
                """
                INSERT INTO errors(
                    signature, error_type, message, traceback, component, severity,
                    first_seen, last_seen, hit_count, metadata_json
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, 1, '{}')
                """,
                (
                    failure.signature,
                    failure.error_type,
                    failure.message,
                    failure.traceback,
                    failure.component,
                    failure.severity,
                    now,
                    now,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def save_snapshot(
        self,
        snapshot_type: str,
        state: dict[str, Any],
        *,
        user_id: str = "",
        parent_snapshot_id: int | None = None,
    ) -> int:
        payload = _json_dump(state)
        checksum = f"{zlib.crc32(payload.encode('utf-8')) & 0xFFFFFFFF:08x}"
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO snapshots(snapshot_type, user_id, state_json, checksum, parent_snapshot_id, created_at)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (snapshot_type, user_id, payload, checksum, parent_snapshot_id, time.time()),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def load_snapshot(self, snapshot_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM snapshots WHERE snapshot_id = ?",
                (int(snapshot_id),),
            ).fetchone()
            if not row:
                return None
            return _json_load(row["state_json"], default={})

    def latest_snapshot_id(self, *, user_id: str = "") -> int | None:
        with self._connect() as conn:
            if user_id:
                row = conn.execute(
                    "SELECT snapshot_id FROM snapshots WHERE user_id = ? ORDER BY snapshot_id DESC LIMIT 1",
                    (user_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT snapshot_id FROM snapshots ORDER BY snapshot_id DESC LIMIT 1"
                ).fetchone()
            if not row:
                return None
            return int(row["snapshot_id"])

    def store_reasoning_trace(self, trace: ReasoningTrace) -> int:
        trace_summary = self._build_trace_summary(trace.input_text, trace.output_text)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO reasoning_traces(
                    user_id, session_id, input_text, output_text, confidence, trace_summary, trace_json, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.user_id,
                    trace.session_id,
                    trace.input_text,
                    trace.output_text,
                    max(0.0, min(1.0, float(trace.confidence))),
                    trace_summary,
                    _json_dump(trace.trace),
                    time.time(),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def recent_reasoning_confidences(self, *, limit: int = 50) -> list[float]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT confidence
                FROM reasoning_traces
                ORDER BY trace_id DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
            return [float(row["confidence"]) for row in rows]

    def upsert_prompt(
        self,
        *,
        name: str,
        template_text: str,
        version: int = 1,
        language_code: str = "en",
        description: str = "",
        is_active: bool = True,
    ) -> int:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO prompts(name, version, language_code, template_text, description, is_active, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name, version, language_code)
                DO UPDATE SET
                    template_text=excluded.template_text,
                    description=excluded.description,
                    is_active=excluded.is_active,
                    updated_at=excluded.updated_at
                """,
                (
                    name,
                    int(version),
                    language_code,
                    template_text,
                    description,
                    1 if is_active else 0,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                """
                SELECT prompt_id FROM prompts
                WHERE name = ? AND version = ? AND language_code = ?
                """,
                (name, int(version), language_code),
            ).fetchone()
            conn.commit()
            return int(row["prompt_id"]) if row else 0

    def get_prompt(self, *, name: str, language_code: str = "en") -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT prompt_id, name, version, language_code, template_text, description
                FROM prompts
                WHERE name = ? AND language_code = ? AND is_active = 1
                ORDER BY version DESC
                LIMIT 1
                """,
                (name, language_code),
            ).fetchone()
            if row:
                return {
                    "prompt_id": int(row["prompt_id"]),
                    "name": str(row["name"]),
                    "version": int(row["version"]),
                    "language_code": str(row["language_code"]),
                    "template_text": str(row["template_text"]),
                    "description": str(row["description"]),
                }
            row = conn.execute(
                """
                SELECT prompt_id, name, version, language_code, template_text, description
                FROM prompts
                WHERE name = ? AND language_code = 'en' AND is_active = 1
                ORDER BY version DESC
                LIMIT 1
                """,
                (name,),
            ).fetchone()
            if row:
                return {
                    "prompt_id": int(row["prompt_id"]),
                    "name": str(row["name"]),
                    "version": int(row["version"]),
                    "language_code": str(row["language_code"]),
                    "template_text": str(row["template_text"]),
                    "description": str(row["description"]),
                }

            row = conn.execute(
                """
                SELECT prompt_id, name, version, language_code, template_text, description
                FROM prompts
                WHERE name = ? AND is_active = 1
                ORDER BY version DESC
                LIMIT 1
                """,
                (name,),
            ).fetchone()
            if not row:
                return None
            return {
                "prompt_id": int(row["prompt_id"]),
                "name": str(row["name"]),
                "version": int(row["version"]),
                "language_code": str(row["language_code"]),
                "template_text": str(row["template_text"]),
                "description": str(row["description"]),
            }

    def list_active_prompts(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT prompt_id, name, version, language_code, description
                FROM prompts
                WHERE is_active = 1
                ORDER BY name ASC, version DESC
                """
            ).fetchall()
            return [
                {
                    "prompt_id": int(row["prompt_id"]),
                    "name": str(row["name"]),
                    "version": int(row["version"]),
                    "language_code": str(row["language_code"]),
                    "description": str(row["description"]),
                }
                for row in rows
            ]

    def record_prompt_run(self, run: PromptRun) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO prompt_runs(prompt_id, user_id, input_json, output_text, confidence, created_at)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    int(run.prompt_id),
                    run.user_id,
                    _json_dump(run.input_payload),
                    run.output_text,
                    max(0.0, min(1.0, float(run.confidence))),
                    time.time(),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def record_recovery_action(self, *, action: str, status: str, details: dict[str, Any] | None = None) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO recovery_actions(action, status, details_json, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (action, status, _json_dump(dict(details or {})), time.time()),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def record_health(self, *, layer_name: str, status: str, score: float, details: dict[str, Any] | None = None) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO health_checks(layer_name, status, score, details_json, created_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (layer_name, status, max(0.0, min(1.0, float(score))), _json_dump(dict(details or {})), time.time()),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def record_audit_action(
        self,
        *,
        user_id: str,
        action_type: str,
        target_path: str,
        content: str,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> int:
        checksum = f"{zlib.crc32(str(content).encode('utf-8')) & 0xFFFFFFFF:08x}" if content else ""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO audit_actions(user_id, action_type, target_path, content_checksum, status, details_json, created_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, action_type, target_path, checksum, status, _json_dump(dict(details or {})), time.time()),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def record_component_version(
        self,
        *,
        component: str,
        version: str,
        checksum: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO versions(component, version, checksum, metadata_json, created_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (component, version, checksum, _json_dump(dict(metadata or {})), time.time()),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def table_counts(self) -> dict[str, int]:
        names = [
            "users",
            "user_profiles",
            "nodes",
            "node_properties",
            "edges",
            "embeddings",
            "versions",
            "logs",
            "errors",
            "snapshots",
            "reasoning_traces",
            "prompts",
            "prompt_runs",
            "recovery_actions",
            "health_checks",
            "audit_actions",
        ]
        out: dict[str, int] = {}
        with self._connect() as conn:
            for table in names:
                row = conn.execute(f"SELECT COUNT(*) AS cnt FROM {table}").fetchone()
                out[table] = int(row["cnt"] or 0)
        return out

    def describe_schema(self) -> dict[str, Any]:
        """Return SQL schema in structured JSON format."""
        with self._connect() as conn:
            version_rows = conn.execute(
                "SELECT version, applied_at FROM schema_versions ORDER BY id ASC"
            ).fetchall()
            table_rows = conn.execute(
                """
                SELECT name, sql
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name ASC
                """
            ).fetchall()

            tables: list[dict[str, Any]] = []
            for row in table_rows:
                table = str(row["name"])
                create_sql = str(row["sql"] or "")

                col_rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
                columns = [
                    {
                        "cid": int(col["cid"]),
                        "name": str(col["name"]),
                        "type": str(col["type"] or ""),
                        "not_null": bool(int(col["notnull"] or 0)),
                        "default": None if col["dflt_value"] is None else str(col["dflt_value"]),
                        "primary_key_order": int(col["pk"] or 0),
                    }
                    for col in col_rows
                ]

                idx_rows = conn.execute(f"PRAGMA index_list('{table}')").fetchall()
                indexes: list[dict[str, Any]] = []
                for idx in idx_rows:
                    index_name = str(idx["name"])
                    idx_cols = conn.execute(f"PRAGMA index_info('{index_name}')").fetchall()
                    indexes.append(
                        {
                            "name": index_name,
                            "unique": bool(int(idx["unique"] or 0)),
                            "origin": str(idx["origin"] or ""),
                            "partial": bool(int(idx["partial"] or 0)),
                            "columns": [str(item["name"]) for item in idx_cols],
                        }
                    )

                tables.append(
                    {
                        "name": table,
                        "create_sql": create_sql,
                        "columns": columns,
                        "indexes": indexes,
                    }
                )

        return {
            "database_path": str(self.db_path),
            "generated_at": time.time(),
            "schema_version": str(version_rows[-1]["version"]) if version_rows else self.BASELINE_SCHEMA_VERSION,
            "target_schema_version": self.SCHEMA_VERSION,
            "applied_versions": [
                {
                    "version": str(row["version"]),
                    "applied_at": float(row["applied_at"]),
                }
                for row in version_rows
            ],
            "tables": tables,
        }

    def required_tables_present(self) -> dict[str, bool]:
        required = {
            "schema_versions",
            "users",
            "user_profiles",
            "nodes",
            "node_properties",
            "edges",
            "embeddings",
            "versions",
            "logs",
            "errors",
            "snapshots",
            "reasoning_traces",
            "localization_texts",
            "prompts",
            "prompt_runs",
            "recovery_actions",
            "health_checks",
            "audit_actions",
        }
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            existing = {str(row["name"]) for row in rows}
        return {name: (name in existing) for name in sorted(required)}

    def latest_errors(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT signature, error_type, message, component, severity, first_seen, last_seen, hit_count
                FROM errors
                ORDER BY last_seen DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
            return [
                {
                    "signature": str(row["signature"]),
                    "error_type": str(row["error_type"]),
                    "message": str(row["message"]),
                    "component": str(row["component"]),
                    "severity": str(row["severity"]),
                    "first_seen": float(row["first_seen"]),
                    "last_seen": float(row["last_seen"]),
                    "hit_count": int(row["hit_count"]),
                }
                for row in rows
            ]

    def restore_graph_from_snapshot(self, snapshot_id: int) -> dict[str, Any] | None:
        payload = self.load_snapshot(snapshot_id)
        if not isinstance(payload, dict):
            return None
        nodes = list(payload.get("nodes", []) or [])
        edges = list(payload.get("edges", []) or [])
        with self._connect() as conn:
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM node_properties")
            conn.execute("DELETE FROM nodes")
            now = time.time()
            for row in nodes:
                if not isinstance(row, dict):
                    continue
                node_id = str(row.get("node_id", "")).strip()
                if not node_id:
                    continue
                conn.execute(
                    """
                    INSERT INTO nodes(
                        node_id, user_id, node_type, display_name, confidence, version,
                        source_tag, metadata_json, created_at, updated_at
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node_id,
                        str(row.get("user_id", "")),
                        str(row.get("node_type", "generic")),
                        str(row.get("display_name", "")),
                        max(0.0, min(1.0, float(row.get("confidence", 0.5) or 0.5))),
                        int(row.get("version", 1) or 1),
                        str(row.get("source_tag", row.get("node_type", "generic")) or "generic"),
                        _json_dump(dict(row.get("metadata", {}) or {})),
                        now,
                        now,
                    ),
                )
                for key, value in dict(row.get("properties", {}) or {}).items():
                    conn.execute(
                        """
                        INSERT INTO node_properties(node_id, prop_key, prop_value_json, language_code, version, updated_at)
                        VALUES(?, ?, ?, '', 1, ?)
                        """,
                        (node_id, str(key), _json_dump(value), now),
                    )
            for row in edges:
                if not isinstance(row, dict):
                    continue
                if not str(row.get("from_node", "")).strip() or not str(row.get("to_node", "")).strip():
                    continue
                conn.execute(
                    """
                    INSERT INTO edges(
                        user_id, from_node, to_node, relation_type, confidence, weight, version,
                        status, metadata_json, created_at, updated_at
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(row.get("user_id", "")),
                        str(row.get("from_node")),
                        str(row.get("to_node")),
                        str(row.get("relation_type", "related_to")),
                        max(0.0, min(1.0, float(row.get("confidence", 0.5) or 0.5))),
                        max(0.0, min(1.0, float(row.get("weight", 0.5) or 0.5))),
                        int(row.get("version", 1) or 1),
                        str(row.get("status", "active") or "active"),
                        _json_dump(dict(row.get("metadata", {}) or {})),
                        now,
                        now,
                    ),
                )
            conn.commit()
        return payload

    def graph_state(self, *, user_id: str = "") -> dict[str, Any]:
        query_user = ""
        params: tuple[Any, ...] = ()
        if user_id:
            query_user = "WHERE user_id = ?"
            params = (user_id,)

        with self._connect() as conn:
            node_rows = conn.execute(
                f"""
                SELECT node_id, user_id, node_type, display_name, confidence, version, source_tag, metadata_json
                FROM nodes
                {query_user}
                """,
                params,
            ).fetchall()
            nodes = []
            for row in node_rows:
                props = conn.execute(
                    """
                    SELECT prop_key, prop_value_json, language_code, version
                    FROM node_properties
                    WHERE node_id = ?
                    ORDER BY id DESC
                    """,
                    (str(row["node_id"]),),
                ).fetchall()
                latest_props: dict[str, Any] = {}
                for prop in props:
                    key = str(prop["prop_key"])
                    if key in latest_props:
                        continue
                    latest_props[key] = _json_load(prop["prop_value_json"], default=None)
                nodes.append(
                    {
                        "node_id": str(row["node_id"]),
                        "user_id": str(row["user_id"]),
                        "node_type": str(row["node_type"]),
                        "display_name": str(row["display_name"]),
                        "confidence": float(row["confidence"]),
                        "version": int(row["version"]),
                        "source_tag": str(row["source_tag"] or ""),
                        "metadata": _json_load(row["metadata_json"], default={}),
                        "properties": latest_props,
                    }
                )

            edge_rows = conn.execute(
                f"""
                SELECT user_id, from_node, to_node, relation_type, confidence, weight, version, status, metadata_json
                FROM edges
                {query_user}
                ORDER BY edge_id ASC
                """,
                params,
            ).fetchall()
            edges = [
                {
                    "user_id": str(row["user_id"]),
                    "from_node": str(row["from_node"]),
                    "to_node": str(row["to_node"]),
                    "relation_type": str(row["relation_type"]),
                    "confidence": float(row["confidence"]),
                    "weight": float(row["weight"]),
                    "version": int(row["version"]),
                    "status": str(row["status"] or "active"),
                    "metadata": _json_load(row["metadata_json"], default={}),
                }
                for row in edge_rows
            ]

        return {"nodes": nodes, "edges": edges}
