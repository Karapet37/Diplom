from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .db import connect
from .graph_model import Edge, Evidence, GraphSnapshot, Node, Source, edge_key, normalize_node_type


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [item.strip() for item in raw.splitlines() if item.strip()]
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return []


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


class GraphStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = connect(db_path)

    def close(self) -> None:
        self.conn.close()

    def _sync_tags(self, node_id: str, tags: list[str]) -> None:
        self.conn.execute("DELETE FROM tags WHERE node_id = ?", (node_id,))
        for tag in sorted({tag for tag in tags if tag}):
            self.conn.execute(
                "INSERT OR IGNORE INTO tags(node_id, tag) VALUES(?, ?)",
                (node_id, tag),
            )

    def _sync_domain_row(self, node: Node) -> None:
        self.conn.execute("DELETE FROM domains WHERE node_id = ?", (node.id,))
        if normalize_node_type(node.type) == "DOMAIN":
            self.conn.execute(
                "INSERT OR REPLACE INTO domains(domain_id, node_id, name) VALUES(?, ?, ?)",
                (node.id, node.id, node.name or node.label or node.id),
            )

    def _sync_examples(self, node: Node, *, source_id: str | None = None) -> None:
        examples = _json_list(node.examples_json)
        self.conn.execute("DELETE FROM examples WHERE node_id = ?", (node.id,))
        for index, text in enumerate(examples):
            example_id = f"example:{node.id}:{index}"
            self.conn.execute(
                """
                INSERT OR REPLACE INTO examples(example_id, node_id, text, description, updated_at)
                VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (example_id, node.id, text, node.description or node.short_gloss or node.label),
            )
            if source_id:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO example_sources(example_id, source_id, snippet_text, offset_start, offset_end)
                    VALUES(?, ?, ?, 0, ?)
                    """,
                    (example_id, source_id, text, len(text)),
                )

    def _node_row_to_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        payload["type"] = normalize_node_type(payload.get("type") or "PATTERN")
        payload["name"] = str(payload.get("name") or payload.get("label") or payload.get("id") or "")
        payload["label"] = payload["name"]
        payload["description"] = str(payload.get("description") or payload.get("short_gloss") or payload["name"])
        payload["short_gloss"] = payload["description"]
        payload["what_it_is"] = str(payload.get("what_it_is") or payload.get("plain_explanation") or payload["description"])
        payload["how_it_works"] = str(payload.get("how_it_works") or "")
        payload["how_to_recognize"] = str(payload.get("how_to_recognize") or "")
        payload["plain_explanation"] = str(payload.get("plain_explanation") or payload["what_it_is"] or payload["description"])
        payload["examples"] = _json_list(payload.get("examples_json"))
        payload["tags"] = _json_list(payload.get("tags_json"))
        payload["speech_patterns"] = _json_list(payload.get("speech_patterns_json"))
        payload["behavior_patterns"] = _json_list(payload.get("behavior_patterns_json"))
        payload["triggers"] = _json_list(payload.get("triggers_json"))
        payload["values"] = _json_list(payload.get("values_json"))
        payload["preferences"] = _json_list(payload.get("preferences_json"))
        payload["reaction_logic"] = _json_list(payload.get("reaction_logic_json"))
        payload["tolerance_thresholds"] = _json_object(payload.get("tolerance_thresholds_json"))
        payload["conflict_patterns"] = _json_list(payload.get("conflict_patterns_json"))
        payload["background"] = str(payload.get("background") or "")
        payload["profession"] = str(payload.get("profession") or "")
        payload["speech_style"] = _json_object(payload.get("speech_style_json"))
        payload["temperament"] = str(payload.get("temperament") or "")
        payload["tolerance_threshold"] = float(payload.get("tolerance_threshold") or 0.5)
        payload["speech_profile"] = {
            "formality": float(payload.get("formality") or 0.5),
            "slang_level": float(payload.get("slang_level") or 0.3),
            "directness": float(payload.get("directness") or 0.5),
            "profanity_tolerance": float(payload.get("profanity_tolerance") or 0.1),
        }
        payload["possible_intents"] = _json_list(payload.get("possible_intents_json"))
        payload["emotion_signals"] = _json_list(payload.get("emotion_signals_json"))
        payload["conflict_level"] = float(payload.get("conflict_level") or 0.0)
        payload["irony_probability"] = float(payload.get("irony_probability") or 0.0)
        payload["importance_vector"] = {
            "logic_weight": float(payload.get("logic_weight") or 0.5),
            "emotion_weight": float(payload.get("emotion_weight") or 0.5),
            "risk_weight": float(payload.get("risk_weight") or 0.5),
            "relevance_weight": float(payload.get("relevance_weight") or 0.5),
        }
        return payload

    def upsert_source(self, source: Source) -> None:
        self.conn.execute(
            """
            INSERT INTO sources(source_id, raw_text)
            VALUES(?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
              raw_text=excluded.raw_text,
              created_at=CURRENT_TIMESTAMP
            """,
            (source.source_id, source.raw_text),
        )
        self.conn.commit()

    def list_sources(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT source_id, created_at, raw_text FROM sources ORDER BY created_at").fetchall()
        return [dict(row) for row in rows]

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return self._node_row_to_payload(dict(row)) if row else None

    def upsert_node(self, node: Node) -> None:
        self.conn.execute(
            """
            INSERT INTO nodes(
              id, type, name, label, description, short_gloss, plain_explanation,
              what_it_is, how_it_works, how_to_recognize, examples_json, tags_json,
              speech_patterns_json, behavior_patterns_json, triggers_json, values_json,
              preferences_json, reaction_logic_json, tolerance_thresholds_json,
              conflict_patterns_json, background, profession, speech_style_json,
              temperament, tolerance_threshold, formality, slang_level, directness,
              profanity_tolerance, possible_intents_json, emotion_signals_json,
              conflict_level, irony_probability, logic_weight, emotion_weight,
              risk_weight, relevance_weight, confidence
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              type=excluded.type,
              name=excluded.name,
              label=excluded.label,
              description=excluded.description,
              short_gloss=excluded.short_gloss,
              plain_explanation=excluded.plain_explanation,
              what_it_is=excluded.what_it_is,
              how_it_works=excluded.how_it_works,
              how_to_recognize=excluded.how_to_recognize,
              examples_json=excluded.examples_json,
              tags_json=excluded.tags_json,
              speech_patterns_json=excluded.speech_patterns_json,
              behavior_patterns_json=excluded.behavior_patterns_json,
              triggers_json=excluded.triggers_json,
              values_json=excluded.values_json,
              preferences_json=excluded.preferences_json,
              reaction_logic_json=excluded.reaction_logic_json,
              tolerance_thresholds_json=excluded.tolerance_thresholds_json,
              conflict_patterns_json=excluded.conflict_patterns_json,
              background=excluded.background,
              profession=excluded.profession,
              speech_style_json=excluded.speech_style_json,
              temperament=excluded.temperament,
              tolerance_threshold=excluded.tolerance_threshold,
              formality=excluded.formality,
              slang_level=excluded.slang_level,
              directness=excluded.directness,
              profanity_tolerance=excluded.profanity_tolerance,
              possible_intents_json=excluded.possible_intents_json,
              emotion_signals_json=excluded.emotion_signals_json,
              conflict_level=excluded.conflict_level,
              irony_probability=excluded.irony_probability,
              logic_weight=excluded.logic_weight,
              emotion_weight=excluded.emotion_weight,
              risk_weight=excluded.risk_weight,
              relevance_weight=excluded.relevance_weight,
              confidence=excluded.confidence,
              updated_at=CURRENT_TIMESTAMP
            """,
            (
                node.id,
                normalize_node_type(node.type),
                node.name or node.label or node.id,
                node.label or node.name or node.id,
                node.description or node.short_gloss or node.name or node.label or node.id,
                node.short_gloss or node.description or node.name or node.label or node.id,
                node.plain_explanation or node.what_it_is or node.description or node.short_gloss or "",
                node.what_it_is or node.plain_explanation or node.description or "",
                node.how_it_works or "",
                node.how_to_recognize or "",
                node.examples_json or "[]",
                node.tags_json or "[]",
                node.speech_patterns_json or "[]",
                node.behavior_patterns_json or "[]",
                node.triggers_json or "[]",
                node.values_json or "[]",
                node.preferences_json or "[]",
                node.reaction_logic_json or "[]",
                node.tolerance_thresholds_json or "{}",
                node.conflict_patterns_json or "[]",
                node.background or "",
                node.profession or "",
                node.speech_style_json or "{}",
                node.temperament or "",
                float(node.tolerance_threshold),
                float(node.formality),
                float(node.slang_level),
                float(node.directness),
                float(node.profanity_tolerance),
                node.possible_intents_json or "[]",
                node.emotion_signals_json or "[]",
                float(node.conflict_level),
                float(node.irony_probability),
                float(node.logic_weight),
                float(node.emotion_weight),
                float(node.risk_weight),
                float(node.relevance_weight),
                float(node.confidence),
            ),
        )
        self._sync_tags(node.id, _json_list(node.tags_json))
        self._sync_domain_row(node)
        self._sync_examples(node)

    def list_nodes(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM nodes ORDER BY updated_at DESC, id ASC").fetchall()
        return [self._node_row_to_payload(dict(row)) for row in rows]

    def list_nodes_by_type(self, node_type: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM nodes WHERE type = ? ORDER BY updated_at DESC, id ASC",
            (normalize_node_type(node_type),),
        ).fetchall()
        return [self._node_row_to_payload(dict(row)) for row in rows]

    def get_edge(self, src_id: str, dst_id: str, edge_type: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM edges WHERE src_id = ? AND dst_id = ? AND type = ?",
            (src_id, dst_id, edge_type),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["edge_key"] = edge_key(item["src_id"], item["type"], item["dst_id"])
        item["metadata"] = _json_object(item.get("metadata_json"))
        example_rows = self.conn.execute(
            """
            SELECT example_sources.source_id, example_sources.snippet_text, example_sources.offset_start, example_sources.offset_end, example_sources.created_at
            FROM example_sources
            JOIN examples ON examples.example_id = example_sources.example_id
            WHERE examples.node_id = ?
            ORDER BY example_sources.created_at
            LIMIT 12
            """,
            (item["src_id"],),
        ).fetchall()
        item["evidence"] = [dict(evidence) for evidence in example_rows]
        return item

    def replace_edge(self, edge: Edge) -> None:
        self.conn.execute(
            """
            INSERT INTO edges(src_id, dst_id, type, weight, confidence, metadata_json)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(src_id, dst_id, type) DO UPDATE SET
              weight=excluded.weight,
              confidence=excluded.confidence,
              metadata_json=excluded.metadata_json,
              updated_at=CURRENT_TIMESTAMP
            """,
            (edge.src_id, edge.dst_id, edge.type, edge.weight, edge.confidence, edge.metadata_json),
        )
        self.conn.commit()

    def update_edge(
        self,
        src_id: str,
        dst_id: str,
        edge_type: str,
        *,
        weight: float | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        current = self.get_edge(src_id, dst_id, edge_type)
        if not current:
            return None
        self.replace_edge(
            Edge(
                src_id=src_id,
                dst_id=dst_id,
                type=edge_type,
                weight=float(weight if weight is not None else current.get("weight", 1.0)),
                confidence=float(confidence if confidence is not None else current.get("confidence", 0.7)),
                metadata_json=json.dumps(metadata if metadata is not None else current.get("metadata") or {}, ensure_ascii=False, sort_keys=True),
            )
        )
        return self.get_edge(src_id, dst_id, edge_type)

    def list_edges(self, *, edge_type: str | None = None, min_weight: float = 0.0) -> list[dict[str, Any]]:
        if edge_type:
            rows = self.conn.execute(
                "SELECT * FROM edges WHERE type = ? AND weight >= ? ORDER BY src_id, dst_id, type",
                (edge_type, float(min_weight)),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM edges WHERE weight >= ? ORDER BY src_id, dst_id, type",
                (float(min_weight),),
            ).fetchall()
        return [self.get_edge(str(row["src_id"]), str(row["dst_id"]), str(row["type"])) for row in rows]

    def add_evidence(self, evidence: Evidence) -> None:
        linked_node_id = None
        if evidence.edge_key:
            parts = evidence.edge_key.split("|", 2)
            if len(parts) == 3:
                linked_node_id = parts[0]
        if not linked_node_id:
            linked_node_id = evidence.source_id
        example_hash = hashlib.sha1(
            f"{evidence.edge_key}|{evidence.source_id}|{evidence.snippet_text}|{evidence.offset_start}|{evidence.offset_end}".encode("utf-8")
        ).hexdigest()[:16]
        example_id = f"evidence:{example_hash}"
        self.conn.execute(
            """
            INSERT OR IGNORE INTO examples(example_id, node_id, text, description)
            VALUES(?, ?, ?, ?)
            """,
            (
                example_id,
                linked_node_id,
                evidence.snippet_text,
                f"Evidence for {evidence.edge_key}",
            ),
        )
        self.conn.execute(
            """
            INSERT OR REPLACE INTO example_sources(example_id, source_id, snippet_text, offset_start, offset_end)
            VALUES(?, ?, ?, ?, ?)
            """,
            (
                example_id,
                evidence.source_id,
                evidence.snippet_text,
                int(evidence.offset_start),
                int(evidence.offset_end),
            ),
        )

    def update_node(self, node_id: str, **changes: Any) -> dict[str, Any] | None:
        current = self.get_node(node_id)
        if not current:
            return None
        node = Node(
            id=node_id,
            type=str(changes.get("type") or current.get("type") or "PATTERN"),
            name=str(changes.get("name") or changes.get("label") or current.get("name") or current.get("label") or node_id),
            label=str(changes.get("label") or changes.get("name") or current.get("label") or current.get("name") or node_id),
            description=str(changes.get("description") or changes.get("short_gloss") or current.get("description") or current.get("short_gloss") or ""),
            short_gloss=str(changes.get("short_gloss") or changes.get("description") or current.get("short_gloss") or current.get("description") or ""),
            plain_explanation=str(changes.get("plain_explanation") or changes.get("what_it_is") or current.get("plain_explanation") or current.get("what_it_is") or ""),
            what_it_is=str(changes.get("what_it_is") or changes.get("plain_explanation") or current.get("what_it_is") or current.get("plain_explanation") or ""),
            how_it_works=str(changes.get("how_it_works") or current.get("how_it_works") or ""),
            how_to_recognize=str(changes.get("how_to_recognize") or current.get("how_to_recognize") or ""),
            examples_json=json.dumps(changes.get("examples") if isinstance(changes.get("examples"), list) else current.get("examples") or [], ensure_ascii=False),
            tags_json=json.dumps(changes.get("tags") if isinstance(changes.get("tags"), list) else current.get("tags") or [], ensure_ascii=False),
            speech_patterns_json=json.dumps(changes.get("speech_patterns") if isinstance(changes.get("speech_patterns"), list) else current.get("speech_patterns") or [], ensure_ascii=False),
            behavior_patterns_json=json.dumps(changes.get("behavior_patterns") if isinstance(changes.get("behavior_patterns"), list) else current.get("behavior_patterns") or [], ensure_ascii=False),
            triggers_json=json.dumps(changes.get("triggers") if isinstance(changes.get("triggers"), list) else current.get("triggers") or [], ensure_ascii=False),
            values_json=json.dumps(changes.get("values") if isinstance(changes.get("values"), list) else current.get("values") or [], ensure_ascii=False),
            preferences_json=json.dumps(changes.get("preferences") if isinstance(changes.get("preferences"), list) else current.get("preferences") or [], ensure_ascii=False),
            reaction_logic_json=json.dumps(changes.get("reaction_logic") if isinstance(changes.get("reaction_logic"), list) else current.get("reaction_logic") or [], ensure_ascii=False),
            tolerance_thresholds_json=json.dumps(changes.get("tolerance_thresholds") if isinstance(changes.get("tolerance_thresholds"), dict) else current.get("tolerance_thresholds") or {}, ensure_ascii=False),
            conflict_patterns_json=json.dumps(changes.get("conflict_patterns") if isinstance(changes.get("conflict_patterns"), list) else current.get("conflict_patterns") or [], ensure_ascii=False),
            background=str(changes.get("background") if changes.get("background") is not None else current.get("background") or ""),
            profession=str(changes.get("profession") if changes.get("profession") is not None else current.get("profession") or ""),
            speech_style_json=json.dumps(changes.get("speech_style") if isinstance(changes.get("speech_style"), dict) else current.get("speech_style") or {}, ensure_ascii=False),
            temperament=str(changes.get("temperament") if changes.get("temperament") is not None else current.get("temperament") or ""),
            tolerance_threshold=float(changes.get("tolerance_threshold") if changes.get("tolerance_threshold") is not None else current.get("tolerance_threshold", 0.5)),
            formality=float(changes.get("formality") if changes.get("formality") is not None else (current.get("speech_profile") or {}).get("formality", 0.5)),
            slang_level=float(changes.get("slang_level") if changes.get("slang_level") is not None else (current.get("speech_profile") or {}).get("slang_level", 0.3)),
            directness=float(changes.get("directness") if changes.get("directness") is not None else (current.get("speech_profile") or {}).get("directness", 0.5)),
            profanity_tolerance=float(changes.get("profanity_tolerance") if changes.get("profanity_tolerance") is not None else (current.get("speech_profile") or {}).get("profanity_tolerance", 0.1)),
            possible_intents_json=json.dumps(changes.get("possible_intents") if isinstance(changes.get("possible_intents"), list) else current.get("possible_intents") or [], ensure_ascii=False),
            emotion_signals_json=json.dumps(changes.get("emotion_signals") if isinstance(changes.get("emotion_signals"), list) else current.get("emotion_signals") or [], ensure_ascii=False),
            conflict_level=float(changes.get("conflict_level") if changes.get("conflict_level") is not None else current.get("conflict_level", 0.0)),
            irony_probability=float(changes.get("irony_probability") if changes.get("irony_probability") is not None else current.get("irony_probability", 0.0)),
            logic_weight=float(changes.get("logic_weight") if changes.get("logic_weight") is not None else (current.get("importance_vector") or {}).get("logic_weight", 0.5)),
            emotion_weight=float(changes.get("emotion_weight") if changes.get("emotion_weight") is not None else (current.get("importance_vector") or {}).get("emotion_weight", 0.5)),
            risk_weight=float(changes.get("risk_weight") if changes.get("risk_weight") is not None else (current.get("importance_vector") or {}).get("risk_weight", 0.5)),
            relevance_weight=float(changes.get("relevance_weight") if changes.get("relevance_weight") is not None else (current.get("importance_vector") or {}).get("relevance_weight", 0.5)),
            confidence=float(changes.get("confidence") if changes.get("confidence") is not None else current.get("confidence", 0.7)),
        )
        self.upsert_node(node)
        self.conn.commit()
        return self.get_node(node_id)

    def export_graph(self, *, edge_type: str | None = None, min_weight: float = 0.0) -> dict[str, Any]:
        node_rows = self.conn.execute("SELECT * FROM nodes ORDER BY id").fetchall()
        if edge_type:
            edge_rows = self.conn.execute(
                "SELECT * FROM edges WHERE type = ? AND weight >= ? ORDER BY src_id, dst_id, type",
                (edge_type, min_weight),
            ).fetchall()
        else:
            edge_rows = self.conn.execute(
                "SELECT * FROM edges WHERE weight >= ? ORDER BY src_id, dst_id, type",
                (min_weight,),
            ).fetchall()

        nodes = [self._node_row_to_payload(dict(r)) for r in node_rows]
        edges = [dict(r) for r in edge_rows]

        node_examples: dict[str, list[dict[str, Any]]] = {}
        for example in self.conn.execute(
            "SELECT example_id, node_id, text, description, created_at FROM examples ORDER BY example_id"
        ).fetchall():
            node_examples.setdefault(str(example["node_id"]), []).append(dict(example))

        for edge in edges:
            ek = edge_key(edge["src_id"], edge["type"], edge["dst_id"])
            rows = self.conn.execute(
                """
                SELECT example_sources.source_id, example_sources.snippet_text, example_sources.offset_start, example_sources.offset_end, example_sources.created_at
                FROM example_sources
                JOIN examples ON examples.example_id = example_sources.example_id
                WHERE examples.node_id IN (?, ?)
                ORDER BY example_sources.created_at
                LIMIT 12
                """,
                (edge["src_id"], edge["dst_id"]),
            ).fetchall()
            edge["edge_key"] = ek
            edge["metadata"] = _json_object(edge.get("metadata_json"))
            edge["evidence"] = [dict(r) for r in rows]

        for node in nodes:
            node["examples"] = [row["text"] for row in node_examples.get(str(node["id"]), [])]
            node["examples_json"] = json.dumps(node["examples"], ensure_ascii=False)

        return {"nodes": nodes, "edges": edges}

    def snapshot(self, *, edge_type: str | None = None, min_weight: float = 0.0) -> GraphSnapshot:
        export = self.export_graph(edge_type=edge_type, min_weight=min_weight)
        nodes = {node["id"]: node for node in export["nodes"]}
        adjacency: dict[str, set[str]] = {node_id: set() for node_id in nodes}
        for edge in export["edges"]:
            adjacency.setdefault(edge["src_id"], set()).add(edge["dst_id"])
            adjacency.setdefault(edge["dst_id"], set())
        frozen_adj = {k: tuple(sorted(v)) for k, v in adjacency.items()}
        return GraphSnapshot(nodes=nodes, edges=export["edges"], adjacency=frozen_adj)

    def export_json(self, path: Path, *, edge_type: str | None = None, min_weight: float = 0.0) -> None:
        payload = self.export_graph(edge_type=edge_type, min_weight=min_weight)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear_graph(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM example_sources")
            self.conn.execute("DELETE FROM examples")
            self.conn.execute("DELETE FROM tags")
            self.conn.execute("DELETE FROM domains")
            self.conn.execute("DELETE FROM edges")
            self.conn.execute("DELETE FROM nodes")

    def replace_graph(
        self,
        *,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> dict[str, int]:
        ordered_nodes = sorted(nodes, key=lambda node: str(node.get("id") or ""))
        ordered_edges = sorted(
            edges,
            key=lambda edge: (
                str(edge.get("src_id") or ""),
                str(edge.get("dst_id") or ""),
                str(edge.get("type") or ""),
            ),
        )
        ordered_evidence = sorted(
            evidence,
            key=lambda ev: (
                str(ev.get("edge_key") or ""),
                str(ev.get("source_id") or ""),
                str(ev.get("snippet_text") or ""),
                int(ev.get("offset_start") or 0),
                int(ev.get("offset_end") or 0),
            ),
        )
        with self.conn:
            self.conn.execute("DELETE FROM example_sources")
            self.conn.execute("DELETE FROM examples")
            self.conn.execute("DELETE FROM tags")
            self.conn.execute("DELETE FROM domains")
            self.conn.execute("DELETE FROM edges")
            self.conn.execute("DELETE FROM nodes")
            for node in ordered_nodes:
                self.upsert_node(
                    Node(
                        id=str(node["id"]),
                        type=str(node.get("type") or "CONCEPT"),
                        name=str(node.get("name") or node.get("label") or ""),
                        label=str(node.get("label") or node.get("name") or ""),
                        description=str(node.get("description") or node.get("short_gloss") or ""),
                        short_gloss=str(node.get("short_gloss") or node.get("description") or ""),
                        plain_explanation=str(node.get("plain_explanation") or node.get("what_it_is") or ""),
                        what_it_is=str(node.get("what_it_is") or node.get("plain_explanation") or ""),
                        how_it_works=str(node.get("how_it_works") or ""),
                        how_to_recognize=str(node.get("how_to_recognize") or ""),
                        examples_json=str(node.get("examples_json") or "[]"),
                        tags_json=str(node.get("tags_json") or "[]"),
                        speech_patterns_json=str(node.get("speech_patterns_json") or "[]"),
                        behavior_patterns_json=str(node.get("behavior_patterns_json") or "[]"),
                        triggers_json=str(node.get("triggers_json") or "[]"),
                        values_json=str(node.get("values_json") or "[]"),
                        preferences_json=str(node.get("preferences_json") or "[]"),
                        reaction_logic_json=str(node.get("reaction_logic_json") or "[]"),
                        tolerance_thresholds_json=str(node.get("tolerance_thresholds_json") or "{}"),
                        conflict_patterns_json=str(node.get("conflict_patterns_json") or "[]"),
                        background=str(node.get("background") or ""),
                        profession=str(node.get("profession") or ""),
                        speech_style_json=str(node.get("speech_style_json") or json.dumps(node.get("speech_style") or {}, ensure_ascii=False)),
                        temperament=str(node.get("temperament") or ""),
                        tolerance_threshold=float(node.get("tolerance_threshold") or 0.5),
                        formality=float(node.get("formality") or (node.get("speech_profile") or {}).get("formality") or 0.5),
                        slang_level=float(node.get("slang_level") or (node.get("speech_profile") or {}).get("slang_level") or 0.3),
                        directness=float(node.get("directness") or (node.get("speech_profile") or {}).get("directness") or 0.5),
                        profanity_tolerance=float(node.get("profanity_tolerance") or (node.get("speech_profile") or {}).get("profanity_tolerance") or 0.1),
                        possible_intents_json=str(node.get("possible_intents_json") or json.dumps(node.get("possible_intents") or [], ensure_ascii=False)),
                        emotion_signals_json=str(node.get("emotion_signals_json") or json.dumps(node.get("emotion_signals") or [], ensure_ascii=False)),
                        conflict_level=float(node.get("conflict_level") or 0.0),
                        irony_probability=float(node.get("irony_probability") or 0.0),
                        logic_weight=float(node.get("logic_weight") or (node.get("importance_vector") or {}).get("logic_weight") or 0.5),
                        emotion_weight=float(node.get("emotion_weight") or (node.get("importance_vector") or {}).get("emotion_weight") or 0.5),
                        risk_weight=float(node.get("risk_weight") or (node.get("importance_vector") or {}).get("risk_weight") or 0.5),
                        relevance_weight=float(node.get("relevance_weight") or (node.get("importance_vector") or {}).get("relevance_weight") or 0.5),
                        confidence=float(node.get("confidence") or 0.7),
                    )
                )
            for edge in ordered_edges:
                self.replace_edge(
                    Edge(
                        src_id=str(edge["src_id"]),
                        dst_id=str(edge["dst_id"]),
                        type=str(edge["type"]),
                        weight=float(edge.get("weight", 1.0)),
                        confidence=float(edge.get("confidence", 0.7)),
                        metadata_json=json.dumps(edge.get("metadata") or _json_object(edge.get("metadata_json")), ensure_ascii=False, sort_keys=True),
                    )
                )
            for ev in ordered_evidence:
                self.add_evidence(
                    Evidence(
                        edge_key=str(ev["edge_key"]),
                        source_id=str(ev["source_id"]),
                        snippet_text=str(ev["snippet_text"]),
                        offset_start=int(ev.get("offset_start", 0)),
                        offset_end=int(ev.get("offset_end", 0)),
                    )
                )
        return {"nodes": len(ordered_nodes), "edges": len(ordered_edges), "evidence": len(ordered_evidence)}

    def search_nodes(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        like = f"%{str(query or '').strip()}%"
        rows = self.conn.execute(
            """
            SELECT * FROM nodes
            WHERE name LIKE ? OR label LIKE ? OR description LIKE ? OR what_it_is LIKE ? OR tags_json LIKE ?
            ORDER BY updated_at DESC, id ASC
            LIMIT ?
            """,
            (like, like, like, like, like, int(limit)),
        ).fetchall()
        return [self._node_row_to_payload(dict(row)) for row in rows]

    def list_domains(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT domains.domain_id, domains.name, nodes.description, nodes.what_it_is
            FROM domains
            JOIN nodes ON nodes.id = domains.node_id
            ORDER BY domains.name
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def apply_batch(
        self,
        *,
        source: Source,
        nodes: list[Node],
        edges: list[Edge],
        evidence: list[Evidence],
    ) -> dict[str, int]:
        with self.conn:
            self.upsert_source(source)
            for node in nodes:
                self.upsert_node(node)
            for edge in edges:
                self.conn.execute(
                    """
                    INSERT INTO edges(src_id, dst_id, type, weight, confidence, metadata_json)
                    VALUES(?, ?, ?, ?, ?, ?)
                    ON CONFLICT(src_id, dst_id, type) DO UPDATE SET
                      weight=excluded.weight,
                      confidence=excluded.confidence,
                      metadata_json=excluded.metadata_json,
                      updated_at=CURRENT_TIMESTAMP
                    """,
                    (
                        edge.src_id,
                        edge.dst_id,
                        edge.type,
                        float(edge.weight),
                        float(edge.confidence),
                        edge.metadata_json or "{}",
                    ),
                )
            for item in evidence:
                self.add_evidence(item)
        return {"nodes": len(nodes), "edges": len(edges), "evidence": len(evidence)}
