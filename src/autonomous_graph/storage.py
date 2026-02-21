"""Persistence adapters for autonomous graph snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class JsonGraphDBAdapter:
    """Local JSON snapshot adapter."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def persist_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        self.path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def load_snapshot(self) -> Mapping[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload


class Neo4jGraphDBAdapter:
    """
    Graph DB adapter for Neo4j.

    Requires `neo4j` package and reachable Neo4j instance.
    Stores snapshot in labels:
    - (:AGNode {id, type, attributes_json, state_json})
    - (:AGNode)-[:AG_EDGE {relation_type, weight, direction, logic_rule}]->(:AGNode)
    """

    def __init__(
        self,
        *,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ):
        try:
            from neo4j import GraphDatabase
        except Exception as exc:
            raise RuntimeError("neo4j package is not installed") from exc
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:
            pass

    def persist_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        nodes = dict(snapshot.get("nodes", {}) or {})
        edges = list(snapshot.get("edges", []) or [])
        with self._driver.session(database=self.database) as session:
            session.run("MATCH (n:AGNode) DETACH DELETE n")
            for raw_id, data in nodes.items():
                try:
                    node_id = int(raw_id)
                except Exception:
                    continue
                node_type = str((data or {}).get("type", "generic") or "generic")
                attributes = json.dumps((data or {}).get("attributes", {}) or {}, ensure_ascii=False)
                state = json.dumps((data or {}).get("state", {}) or {}, ensure_ascii=False)
                session.run(
                    (
                        "MERGE (n:AGNode {id: $id}) "
                        "SET n.type = $type, n.attributes_json = $attributes, n.state_json = $state"
                    ),
                    id=node_id,
                    type=node_type,
                    attributes=attributes,
                    state=state,
                )
            for item in edges:
                if not isinstance(item, Mapping):
                    continue
                try:
                    from_id = int(item.get("from"))
                    to_id = int(item.get("to"))
                except Exception:
                    continue
                relation_type = str(item.get("relation_type", "") or "")
                if not relation_type:
                    continue
                session.run(
                    (
                        "MATCH (a:AGNode {id: $from_id}), (b:AGNode {id: $to_id}) "
                        "MERGE (a)-[r:AG_EDGE {relation_type: $relation_type, direction: $direction}]->(b) "
                        "SET r.weight = $weight, r.logic_rule = $logic_rule"
                    ),
                    from_id=from_id,
                    to_id=to_id,
                    relation_type=relation_type,
                    direction=str(item.get("direction", "directed") or "directed"),
                    weight=float(item.get("weight", 1.0) or 1.0),
                    logic_rule=str(item.get("logic_rule", "explicit") or "explicit"),
                )

    def load_snapshot(self) -> Mapping[str, Any]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        with self._driver.session(database=self.database) as session:
            node_rows = session.run(
                "MATCH (n:AGNode) RETURN n.id AS id, n.type AS type, n.attributes_json AS attributes, n.state_json AS state"
            )
            for row in node_rows:
                raw_id = str(row.get("id"))
                try:
                    attrs = json.loads(row.get("attributes") or "{}")
                except Exception:
                    attrs = {}
                try:
                    state = json.loads(row.get("state") or "{}")
                except Exception:
                    state = {}
                nodes[raw_id] = {
                    "type": str(row.get("type") or "generic"),
                    "attributes": attrs if isinstance(attrs, dict) else {},
                    "state": state if isinstance(state, dict) else {},
                }

            edge_rows = session.run(
                (
                    "MATCH (a:AGNode)-[r:AG_EDGE]->(b:AGNode) "
                    "RETURN a.id AS from_id, b.id AS to_id, "
                    "r.relation_type AS relation_type, r.weight AS weight, "
                    "r.direction AS direction, r.logic_rule AS logic_rule"
                )
            )
            for row in edge_rows:
                edges.append(
                    {
                        "from": int(row.get("from_id")),
                        "to": int(row.get("to_id")),
                        "relation_type": str(row.get("relation_type") or ""),
                        "weight": float(row.get("weight") or 1.0),
                        "direction": str(row.get("direction") or "directed"),
                        "logic_rule": str(row.get("logic_rule") or "explicit"),
                    }
                )
        return {"nodes": nodes, "edges": edges}
