from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

ALLOWED_NODE_TYPES = {"ENTITY", "PERSON", "TRAIT", "PATTERN", "SIGNAL", "RELATION", "STYLE", "CONCEPT", "SKILL", "CONTEXT", "EXAMPLE"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def memory_root() -> Path:
    env_root = str(os.environ.get("COGNITIVE_MEMORY_ROOT", "")).strip()
    root = Path(env_root).resolve() if env_root else repo_root() / "memory"
    root.mkdir(parents=True, exist_ok=True)
    return root


def graphs_dir() -> Path:
    path = memory_root() / "graphs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def personalities_dir() -> Path:
    path = memory_root() / "personalities"
    path.mkdir(parents=True, exist_ok=True)
    return path


def personality_proposals_dir() -> Path:
    path = personalities_dir() / "proposals"
    path.mkdir(parents=True, exist_ok=True)
    return path


def graph_nodes_path() -> Path:
    return graphs_dir() / "nodes.json"


def graph_edges_path() -> Path:
    return graphs_dir() / "edges.json"


def personality_index_path() -> Path:
    path = personalities_dir() / "index.json"
    if not path.exists():
        write_json(path, {"personalities": []})
    return path


def normalize_personality_name(value: str) -> str:
    token = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value or ""))
    token = token.strip("_").lower()
    return token or "unknown_personality"


def personality_profile_path(name: str) -> Path:
    return personalities_dir() / f"{normalize_personality_name(name)}.json"


def personality_graph_path(name: str) -> Path:
    return personalities_dir() / f"{normalize_personality_name(name)}_graph.json"


def personality_proposal_path(name: str) -> Path:
    return personality_proposals_dir() / f"{normalize_personality_name(name)}.json"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _slug(value: str) -> str:
    token = "".join(char.lower() if char.isalnum() else "_" for char in str(value or ""))
    token = "_".join(part for part in token.split("_") if part)
    return token or "item"


def _normalize_node_type(value: str) -> str:
    token = str(value or "CONCEPT").strip().upper()
    return token if token in ALLOWED_NODE_TYPES else "CONCEPT"


def _read_nodes() -> list[dict[str, Any]]:
    payload = load_json(graph_nodes_path(), [])
    if isinstance(payload, dict):
        payload = payload.get("nodes") or []
    return [dict(item) for item in payload if isinstance(item, dict)]


def _read_edges() -> list[dict[str, Any]]:
    payload = load_json(graph_edges_path(), [])
    if isinstance(payload, dict):
        payload = payload.get("edges") or []
    return [dict(item) for item in payload if isinstance(item, dict)]


class GraphStore:
    def load_nodes(self) -> list[dict[str, Any]]:
        return _read_nodes()

    def load_edges(self) -> list[dict[str, Any]]:
        return _read_edges()

    def load_graph(self) -> dict[str, list[dict[str, Any]]]:
        return {"nodes": self.load_nodes(), "edges": self.load_edges()}

    def save_graph(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]], *, reason: str = "sync") -> dict[str, Any]:
        existing_nodes = self.load_nodes()
        existing_edges = self.load_edges()
        normalized_nodes = [self._normalize_node_payload(item) for item in nodes if isinstance(item, dict)]
        normalized_edges = [self._normalize_edge_payload(item) for item in edges if isinstance(item, dict)]
        if not normalized_nodes and existing_nodes:
            return {
                "ok": False,
                "reason": "skipped_empty_overwrite",
                "node_count": len(existing_nodes),
                "edge_count": len(existing_edges),
            }
        write_json(graph_nodes_path(), normalized_nodes)
        write_json(graph_edges_path(), normalized_edges)
        return {"ok": True, "reason": reason, "node_count": len(normalized_nodes), "edge_count": len(normalized_edges)}

    def sync_graph_files(self, *, nodes: list[dict[str, Any]], edges: list[dict[str, Any]], reason: str = "sync") -> dict[str, Any]:
        return self.save_graph(nodes, edges, reason=reason)

    def entity_exists(self, name: str) -> bool:
        target = _slug(name)
        for node in self.load_nodes():
            if _slug(node.get("name") or node.get("id") or "") == target:
                return True
        return False

    def merge_proposals(self, proposals: list[dict[str, Any]]) -> dict[str, Any]:
        nodes = self.load_nodes()
        edges = self.load_edges()
        nodes_by_id = {str(node.get("id") or ""): self._normalize_node_payload(node) for node in nodes}
        edge_keys = {
            (str(edge.get("from") or ""), str(edge.get("type") or ""), str(edge.get("to") or "")): self._normalize_edge_payload(edge)
            for edge in edges
        }
        touched_nodes = 0
        touched_edges = 0
        for item in proposals:
            if not isinstance(item, dict):
                continue
            entity = str(item.get("entity") or "").strip()
            if not entity:
                continue
            entity_type = _normalize_node_type(str(item.get("type") or "ENTITY"))
            qualifiers = self._context_from_proposal(item)
            entity_id = self._node_id_for(entity, entity_type, qualifiers)
            description = str(item.get("description") or "").strip()
            node = nodes_by_id.get(entity_id) or self._new_node(entity_id, entity, entity_type, qualifiers)
            node["type"] = entity_type
            node["name"] = entity
            node["description"] = description or node.get("description") or f"{entity} is a {entity_type.lower()} node in the knowledge graph."
            node["short_gloss"] = node["description"]
            node["attributes"] = self._merge_attributes(node.get("attributes"), item)
            node["context"] = self._merge_context(node.get("context"), qualifiers)
            node["frequency"] = int(node.get("frequency") or 0) + 1
            node["importance"] = round(max(0.1, float(node.get("importance") or 1.0) + math.log1p(node["frequency"])), 4)
            node["confidence"] = round(max(float(node.get("confidence") or 0.6), 0.8), 4)
            nodes_by_id[entity_id] = self._normalize_node_payload(node)
            touched_nodes += 1

            for trait in [str(v).strip() for v in list(item.get("traits") or []) if str(v).strip()]:
                trait_id = f"trait:{_slug(entity)}:{_slug(trait)}"
                trait_node = nodes_by_id.get(trait_id) or self._new_node(trait_id, trait, "TRAIT", {"source": "session"})
                trait_node["description"] = trait_node.get("description") or f"{trait} is a trait associated with {entity}."
                trait_node["short_gloss"] = trait_node["description"]
                trait_node["attributes"] = self._merge_attributes(trait_node.get("attributes"), {"traits": [trait]})
                trait_node["frequency"] = int(trait_node.get("frequency") or 0) + 1
                trait_node["importance"] = round(max(0.1, float(trait_node.get("importance") or 1.0) + math.log1p(trait_node["frequency"])), 4)
                trait_node["confidence"] = round(max(float(trait_node.get("confidence") or 0.6), 0.78), 4)
                nodes_by_id[trait_id] = self._normalize_node_payload(trait_node)
                edge_keys[(entity_id, "HAS_TRAIT", trait_id)] = self._normalize_edge_payload({"from": entity_id, "to": trait_id, "type": "HAS_TRAIT", "weight": 1.0})
                touched_nodes += 1
                touched_edges += 1

            for relation in list(item.get("relations") or []):
                if not isinstance(relation, dict):
                    continue
                relation_type = str(relation.get("type") or "RELATED_TO").strip().upper() or "RELATED_TO"
                target = str(relation.get("target") or "").strip()
                if not target:
                    continue
                target_id = f"entity:{_slug(target)}"
                target_node = nodes_by_id.get(target_id) or self._new_node(target_id, target, "ENTITY", {"source": "session"})
                target_node["description"] = target_node.get("description") or f"{target} is a related graph entity."
                target_node["short_gloss"] = target_node["description"]
                nodes_by_id[target_id] = self._normalize_node_payload(target_node)
                edge_keys[(entity_id, relation_type, target_id)] = self._normalize_edge_payload({
                    "from": entity_id,
                    "to": target_id,
                    "type": relation_type,
                    "weight": float(relation.get("strength") or relation.get("weight") or 1.0),
                })
                touched_nodes += 1
                touched_edges += 1

        result = self.save_graph(list(nodes_by_id.values()), list(edge_keys.values()), reason="merge_proposals")
        result.update({"touched_nodes": touched_nodes, "touched_edges": touched_edges})
        return result

    def search_nodes(self, query: str, *, limit: int = 12) -> list[dict[str, Any]]:
        query_tokens = {_slug(part) for part in str(query or "").split() if _slug(part)}
        if not query_tokens:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for node in self.load_nodes():
            haystack = " ".join(
                [
                    str(node.get("name") or ""),
                    str(node.get("description") or ""),
                    str(node.get("attributes") or ""),
                    str(node.get("context") or ""),
                ]
            ).lower()
            hay_tokens = {_slug(part) for part in haystack.split() if _slug(part)}
            overlap = len(query_tokens & hay_tokens)
            if overlap <= 0:
                continue
            score = overlap * max(float(node.get("importance") or 1.0), 0.1) * max(float(node.get("confidence") or 0.5), 0.1) * max(math.log1p(int(node.get("frequency") or 1)), 0.1)
            scored.append((score, deepcopy(node)))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("name") or item[1].get("id") or "")))
        return [item[1] for item in scored[:limit]]

    def subgraph(self, query: str, *, limit: int = 12) -> dict[str, Any]:
        seeds = self.search_nodes(query, limit=limit)
        if not seeds:
            return {"query": str(query or ""), "nodes": [], "edges": [], "seed_node_ids": []}
        seed_ids = {str(item.get("id") or "") for item in seeds}
        selected_ids = set(seed_ids)
        edges = []
        for edge in self.load_edges():
            src = str(edge.get("from") or "")
            dst = str(edge.get("to") or "")
            if src in seed_ids or dst in seed_ids:
                edges.append(deepcopy(edge))
                if src:
                    selected_ids.add(src)
                if dst:
                    selected_ids.add(dst)
        nodes = [node for node in self.load_nodes() if str(node.get("id") or "") in selected_ids]
        return {"query": str(query or ""), "nodes": nodes, "edges": edges[: max(limit, 12)], "seed_node_ids": sorted(seed_ids)}

    def answerable_node_view(self, node_id: str) -> dict[str, Any] | None:
        nodes = {str(node.get("id") or ""): node for node in self.load_nodes()}
        node = nodes.get(str(node_id or ""))
        if not node:
            return None
        related_edges = [
            edge for edge in self.load_edges()
            if str(edge.get("from") or "") == str(node_id) or str(edge.get("to") or "") == str(node_id)
        ]
        return {
            "who_or_what": {
                "id": node.get("id"),
                "type": node.get("type"),
                "name": node.get("name"),
            },
            "what_is_it_like": {
                "description": node.get("description"),
                "attributes": node.get("attributes") or {},
                "context": node.get("context") or {},
                "importance": node.get("importance"),
                "confidence": node.get("confidence"),
                "frequency": node.get("frequency"),
            },
            "how_it_acts": related_edges,
        }

    def _node_id_for(self, entity: str, node_type: str, context: dict[str, Any]) -> str:
        base = f"{node_type.lower()}:{_slug(entity)}"
        if node_type != "PERSON":
            return base
        qualifiers = [
            _slug(context.get("profession") or ""),
            _slug(context.get("location") or ""),
            _slug(context.get("context") or ""),
        ]
        qualifiers = [item for item in qualifiers if item and item != "item"]
        return f"{base}:{':'.join(qualifiers)}" if qualifiers else base

    def _context_from_proposal(self, proposal: dict[str, Any]) -> dict[str, Any]:
        context = proposal.get("context") if isinstance(proposal.get("context"), dict) else {}
        return {
            "profession": str(proposal.get("profession") or context.get("profession") or "").strip(),
            "location": str(proposal.get("location") or context.get("location") or "").strip(),
            "context": str(proposal.get("context_text") or context.get("context") or "").strip(),
            "source": str(context.get("source") or proposal.get("source") or "session").strip() or "session",
        }

    def _merge_attributes(self, existing: Any, proposal: dict[str, Any]) -> dict[str, Any]:
        payload = dict(existing) if isinstance(existing, dict) else {}
        existing_traits = [str(item).strip() for item in list(payload.get("traits") or []) if str(item).strip()]
        proposal_traits = [str(item).strip() for item in list(proposal.get("traits") or []) if str(item).strip()]
        payload["traits"] = list(dict.fromkeys(existing_traits + proposal_traits))
        for key in ("friendliness", "fear", "confidence", "aggression", "empathy", "curiosity"):
            if key in proposal:
                try:
                    payload[key] = max(0.0, min(1.0, float(proposal.get(key))))
                except (TypeError, ValueError):
                    continue
        return payload

    def _merge_context(self, existing: Any, proposal_context: dict[str, Any]) -> dict[str, Any]:
        payload = dict(existing) if isinstance(existing, dict) else {}
        for key in ("profession", "location", "context", "source"):
            value = str(proposal_context.get(key) or payload.get(key) or "").strip()
            payload[key] = value
        return payload

    def _new_node(self, node_id: str, name: str, node_type: str, context: dict[str, Any]) -> dict[str, Any]:
        return self._normalize_node_payload(
            {
                "id": node_id,
                "type": node_type,
                "name": name,
                "description": f"{name} is a {node_type.lower()} node.",
                "short_gloss": f"{name} is a {node_type.lower()} node.",
                "attributes": {"traits": []},
                "context": {
                    "profession": str(context.get("profession") or "").strip(),
                    "location": str(context.get("location") or "").strip(),
                    "context": str(context.get("context") or "").strip(),
                    "source": str(context.get("source") or "session").strip() or "session",
                },
                "importance": 1.0,
                "confidence": 0.7,
                "frequency": 1,
            }
        )

    def _normalize_node_payload(self, node: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(node)
        normalized["id"] = str(normalized.get("id") or f"concept:{_slug(normalized.get('name') or 'item')}")
        normalized["type"] = _normalize_node_type(str(normalized.get("type") or "CONCEPT"))
        normalized["name"] = str(normalized.get("name") or normalized.get("label") or normalized["id"])
        normalized["description"] = str(normalized.get("description") or normalized.get("short_gloss") or normalized["name"])
        normalized["short_gloss"] = str(normalized.get("short_gloss") or normalized["description"])
        attributes = normalized.get("attributes") if isinstance(normalized.get("attributes"), dict) else {}
        context = normalized.get("context") if isinstance(normalized.get("context"), dict) else {}
        normalized["attributes"] = {**attributes, "traits": [str(item).strip() for item in list(attributes.get("traits") or []) if str(item).strip()]}
        normalized["context"] = {
            "profession": str(context.get("profession") or "").strip(),
            "location": str(context.get("location") or "").strip(),
            "context": str(context.get("context") or "").strip(),
            "source": str(context.get("source") or "session").strip() or "session",
        }
        normalized["importance"] = float(normalized.get("importance") or 1.0)
        normalized["confidence"] = float(normalized.get("confidence") or 0.7)
        normalized["frequency"] = int(normalized.get("frequency") or 1)
        return normalized

    def _normalize_edge_payload(self, edge: dict[str, Any]) -> dict[str, Any]:
        return {
            "from": str(edge.get("from") or edge.get("src_id") or ""),
            "to": str(edge.get("to") or edge.get("dst_id") or ""),
            "type": str(edge.get("type") or "RELATED_TO").strip().upper() or "RELATED_TO",
            "weight": float(edge.get("weight") or 1.0),
        }
