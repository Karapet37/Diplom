from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from core import GraphInitializer

from .graph_model import Edge, Evidence, Node, Source, edge_key, normalize_node_type
from .store import GraphStore


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "series"
SUPPORTED_KINDS = {"graph_foundation", "narrative_series"}


def _json_array(values: list[str] | tuple[str, ...]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def list_builtin_foundations() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted(path.stem for path in DATA_DIR.glob("*.json"))


def load_builtin_foundation(dataset_id: str) -> dict[str, Any]:
    path = DATA_DIR / f"{dataset_id}.json"
    if not path.exists():
        raise ValueError(f"unknown foundation dataset '{dataset_id}'")
    payload = _load_json(path)
    kind = str(payload.get("kind") or "").strip()
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"dataset '{dataset_id}' is not a supported foundation payload")
    return payload


def load_foundation_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    dataset = dict(payload or {})
    kind = str(dataset.get("kind") or "").strip()
    if kind not in SUPPORTED_KINDS:
        raise ValueError("payload kind must be graph_foundation or narrative_series")
    dataset_id = str(dataset.get("id") or "").strip()
    if not dataset_id:
        raise ValueError("payload id is required")
    if kind == "graph_foundation":
        if not isinstance(dataset.get("nodes"), list):
            raise ValueError("graph_foundation payload must include nodes")
        if not isinstance(dataset.get("edges"), list):
            raise ValueError("graph_foundation payload must include edges")
    else:
        if not isinstance(dataset.get("characters"), list) or not dataset.get("characters"):
            raise ValueError("narrative_series payload must include characters")
    return dataset


def is_foundation_source(raw_text: str) -> bool:
    try:
        payload = json.loads(str(raw_text or ""))
    except json.JSONDecodeError:
        return False
    return str(payload.get("kind") or "") in SUPPORTED_KINDS


def foundation_from_source(raw_text: str) -> dict[str, Any]:
    payload = json.loads(str(raw_text or ""))
    if str(payload.get("kind") or "") not in SUPPORTED_KINDS:
        raise ValueError("raw_text is not a supported foundation payload")
    return payload


def _node_from_item(item: dict[str, Any], *, default_type: str) -> dict[str, Any]:
    return {
        "id": str(item["id"]),
        "type": normalize_node_type(str(item.get("type") or default_type)),
        "label": str(item.get("label") or item.get("name") or item["id"]),
        "name": str(item.get("name") or item.get("label") or item["id"]),
        "description": str(item.get("description") or item.get("short_gloss") or ""),
        "short_gloss": str(item.get("short_gloss") or item.get("description") or ""),
        "plain_explanation": str(item.get("plain_explanation") or item.get("what_it_is") or ""),
        "what_it_is": str(item.get("what_it_is") or item.get("plain_explanation") or ""),
        "how_it_works": str(item.get("how_it_works") or ""),
        "how_to_recognize": str(item.get("how_to_recognize") or ""),
        "examples_json": json.dumps(list(item.get("examples") or []), ensure_ascii=False),
        "tags_json": json.dumps(list(item.get("tags") or []), ensure_ascii=False),
        "speech_patterns_json": json.dumps(list(item.get("speech_patterns") or []), ensure_ascii=False),
        "behavior_patterns_json": json.dumps(list(item.get("behavior_patterns") or []), ensure_ascii=False),
        "triggers_json": json.dumps(list(item.get("triggers") or []), ensure_ascii=False),
        "values_json": json.dumps(list(item.get("values") or []), ensure_ascii=False),
        "preferences_json": json.dumps(list(item.get("preferences") or []), ensure_ascii=False),
        "reaction_logic_json": json.dumps(list(item.get("reaction_logic") or []), ensure_ascii=False),
        "tolerance_thresholds_json": json.dumps(dict(item.get("tolerance_thresholds") or {}), ensure_ascii=False),
        "conflict_patterns_json": json.dumps(list(item.get("conflict_patterns") or []), ensure_ascii=False),
        "background": str(item.get("background") or ""),
        "profession": str(item.get("profession") or ""),
        "speech_style_json": json.dumps(dict(item.get("speech_style") or {}), ensure_ascii=False),
        "temperament": str(item.get("temperament") or ""),
        "tolerance_threshold": float(item.get("tolerance_threshold") or 0.5),
        "formality": float(item.get("formality") or (item.get("speech_profile") or {}).get("formality") or 0.5),
        "slang_level": float(item.get("slang_level") or (item.get("speech_profile") or {}).get("slang_level") or 0.3),
        "directness": float(item.get("directness") or (item.get("speech_profile") or {}).get("directness") or 0.5),
        "profanity_tolerance": float(item.get("profanity_tolerance") or (item.get("speech_profile") or {}).get("profanity_tolerance") or 0.1),
        "possible_intents_json": json.dumps(list(item.get("possible_intents") or []), ensure_ascii=False),
        "emotion_signals_json": json.dumps(list(item.get("emotion_signals") or []), ensure_ascii=False),
        "conflict_level": float(item.get("conflict_level") or 0.0),
        "irony_probability": float(item.get("irony_probability") or 0.0),
        "logic_weight": float(item.get("logic_weight") or (item.get("importance_vector") or {}).get("logic_weight") or 0.5),
        "emotion_weight": float(item.get("emotion_weight") or (item.get("importance_vector") or {}).get("emotion_weight") or 0.5),
        "risk_weight": float(item.get("risk_weight") or (item.get("importance_vector") or {}).get("risk_weight") or 0.5),
        "relevance_weight": float(item.get("relevance_weight") or (item.get("importance_vector") or {}).get("relevance_weight") or 0.5),
        "confidence": float(item.get("confidence") or 0.7),
    }


def _build_legacy_series_payload(dataset: dict[str, Any]) -> dict[str, Any]:
    series = dict(dataset.get("series") or {})
    dataset_id = str(dataset.get("id") or series.get("id") or "").strip()
    if not dataset_id:
        raise ValueError("dataset id is required")
    source_payload = json.dumps(dataset, ensure_ascii=False, sort_keys=True, indent=2)
    source = Source(source_id=f"seed:foundation:{dataset_id}", raw_text=source_payload)

    nodes: list[dict[str, Any]] = [
        _node_from_item(
            {
                "id": str(series.get("id") or f"domain:{dataset_id}"),
                "type": str(series.get("type") or "DOMAIN"),
                "name": str(series.get("label") or dataset_id.replace("_", " ").title()),
                "label": str(series.get("label") or dataset_id.replace("_", " ").title()),
                "description": str(series.get("short_gloss") or ""),
                "short_gloss": str(series.get("short_gloss") or ""),
                "plain_explanation": str(series.get("plain_explanation") or ""),
                "examples": list(series.get("examples") or []),
                "tags": list(series.get("tags") or []),
            },
            default_type="DOMAIN",
        )
    ]
    root_id = nodes[0]["id"]

    for item in dataset.get("characters") or []:
        nodes.append(_node_from_item(dict(item), default_type="PERSON"))
    for item in dataset.get("concepts") or []:
        nodes.append(_node_from_item(dict(item), default_type="PATTERN"))
    for item in dataset.get("styles") or []:
        nodes.append(_node_from_item(dict(item), default_type="TRAIT"))
    for item in dataset.get("episodes") or []:
        episode = dict(item)
        nodes.append(
            _node_from_item(
                {
                    "id": str(episode["id"]),
                    "type": "EXAMPLE",
                    "name": str(episode.get("label") or episode.get("title") or episode.get("code") or episode["id"]),
                    "label": str(episode.get("label") or episode.get("title") or episode.get("code") or episode["id"]),
                    "description": str(episode.get("short_gloss") or episode.get("code") or ""),
                    "short_gloss": str(episode.get("short_gloss") or episode.get("code") or ""),
                    "plain_explanation": str(episode.get("summary") or episode.get("plain_explanation") or ""),
                    "examples": list(episode.get("examples") or []),
                    "tags": list(episode.get("tags") or ["example"]),
                },
                default_type="EXAMPLE",
            )
        )

    edges_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    evidence_by_edge: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add_edge(src_id: str, dst_id: str, edge_type: str, weight: float, confidence: float, snippet: str) -> None:
        key = (str(src_id), str(dst_id), str(edge_type))
        if key in edges_by_key:
            current = edges_by_key[key]
            current["weight"] = round(float(current["weight"]) + float(weight), 6)
            current["confidence"] = max(float(current["confidence"]), float(confidence))
        else:
            edges_by_key[key] = {
                "src_id": str(src_id),
                "dst_id": str(dst_id),
                "type": str(edge_type),
                "weight": float(weight),
                "confidence": float(confidence),
            }
        ek = edge_key(str(src_id), str(edge_type), str(dst_id))
        evidence_by_edge[ek].append(
            {
                "edge_key": ek,
                "source_id": source.source_id,
                "snippet_text": str(snippet),
                "offset_start": 0,
                "offset_end": len(str(snippet)),
            }
        )

    for item in dataset.get("characters") or []:
        add_edge(root_id, str(item["id"]), "WORKS_IN_DOMAIN", 1.0, 0.98, f"{item['id']} appears in {root_id}.")
    for item in dataset.get("concepts") or []:
        add_edge(root_id, str(item["id"]), "RELATED_TO", 0.82, 0.9, f"{item['id']} is a tracked pattern in {root_id}.")
    for item in dataset.get("styles") or []:
        add_edge(root_id, str(item["id"]), "HAS_TRAIT", 0.78, 0.88, f"{item['id']} is a recurring trait style in {root_id}.")
    for item in dataset.get("style_links") or []:
        add_edge(
            str(item["src_id"]),
            str(item["dst_id"]),
            "HAS_TRAIT",
            float(item.get("weight", 1.0)),
            float(item.get("confidence", 0.9)),
            str(item.get("evidence") or f"{item['src_id']} expresses {item['dst_id']}."),
        )
    for item in dataset.get("links") or []:
        edge_type = str(item.get("type") or "RELATED_TO")
        if edge_type in {"SHAPED_BY", "EMBODIES_STYLE"}:
            edge_type = "RELATED_TO" if edge_type == "SHAPED_BY" else "HAS_TRAIT"
        add_edge(
            str(item["src_id"]),
            str(item["dst_id"]),
            edge_type,
            float(item.get("weight", 1.0)),
            float(item.get("confidence", 0.85)),
            str(item.get("evidence") or ""),
        )

    for episode in dataset.get("episodes") or []:
        episode_id = str(episode["id"])
        participants = set(str(item) for item in episode.get("participants") or [])
        for interaction in episode.get("interactions") or []:
            participants.add(str(interaction["src_id"]))
            participants.add(str(interaction["dst_id"]))
        for participant in sorted(participants):
            add_edge(episode_id, participant, "RELATED_TO", 0.6, 0.9, f"{episode_id} features {participant}.")
        for interaction in episode.get("interactions") or []:
            snippet = str(interaction.get("evidence") or interaction.get("summary") or interaction.get("type") or "")
            add_edge(
                str(interaction["src_id"]),
                str(interaction["dst_id"]),
                "RESPONDS_WITH",
                float(interaction.get("weight", 1.0)),
                float(interaction.get("confidence", 0.85)),
                snippet,
            )
            add_edge(episode_id, str(interaction["src_id"]), "SAID_EXAMPLE", 0.5, 0.8, snippet)
            add_edge(episode_id, str(interaction["dst_id"]), "RELATED_TO", 0.4, 0.8, snippet)

    nodes_sorted = sorted(nodes, key=lambda item: str(item["id"]))
    edges_sorted = [edges_by_key[key] for key in sorted(edges_by_key)]
    evidence_sorted: list[dict[str, Any]] = []
    for ek in sorted(evidence_by_edge):
        evidence_sorted.extend(evidence_by_edge[ek])

    return {
        "dataset_id": dataset_id,
        "source": source,
        "nodes": nodes_sorted,
        "edges": edges_sorted,
        "evidence": evidence_sorted,
    }


def build_foundation_payload(dataset: dict[str, Any]) -> dict[str, Any]:
    kind = str(dataset.get("kind") or "").strip()
    dataset_id = str(dataset.get("id") or "").strip()
    if not dataset_id:
        raise ValueError("dataset id is required")
    if kind == "graph_foundation":
        source = Source(
            source_id=f"seed:foundation:{dataset_id}",
            raw_text=json.dumps(dataset, ensure_ascii=False, sort_keys=True, indent=2),
        )
        nodes = [_node_from_item(dict(item), default_type="PATTERN") for item in list(dataset.get("nodes") or [])]
        edges = [
            {
                "src_id": str(item["src_id"]),
                "dst_id": str(item["dst_id"]),
                "type": str(item["type"]),
                "weight": float(item.get("weight", 1.0)),
                "confidence": float(item.get("confidence", 0.7)),
                "metadata_json": json.dumps(item.get("metadata") or {}, ensure_ascii=False, sort_keys=True),
            }
            for item in list(dataset.get("edges") or [])
        ]
        evidence = [
            {
                "edge_key": str(item["edge_key"]),
                "source_id": str(item["source_id"]),
                "snippet_text": str(item["snippet_text"]),
                "offset_start": int(item.get("offset_start", 0)),
                "offset_end": int(item.get("offset_end", 0)),
            }
            for item in list(dataset.get("evidence") or [])
        ]
        return {
            "dataset_id": dataset_id,
            "source": source,
            "nodes": sorted(nodes, key=lambda item: str(item["id"])),
            "edges": sorted(edges, key=lambda item: (str(item["src_id"]), str(item["dst_id"]), str(item["type"]))),
            "evidence": sorted(evidence, key=lambda item: (str(item["edge_key"]), str(item["source_id"]), str(item["snippet_text"]))),
        }
    if kind == "narrative_series":
        return _build_legacy_series_payload(dataset)
    raise ValueError(f"unsupported foundation kind '{kind}'")


def seed_foundation(store: GraphStore, dataset: dict[str, Any], *, replace_graph: bool = True) -> dict[str, Any]:
    payload = build_foundation_payload(dataset)
    graph_brain = GraphInitializer().initialize({"nodes": payload["nodes"], "edges": payload["edges"]})
    store.upsert_source(payload["source"])
    if replace_graph:
        stats = store.replace_graph(
            nodes=list(payload["nodes"]),
            edges=list(payload["edges"]),
            evidence=list(payload["evidence"]),
        )
    else:
        stats = store.apply_batch(
            source=payload["source"],
            nodes=[Node(**item) for item in payload["nodes"]],
            edges=[Edge(**item) for item in payload["edges"]],
            evidence=[Evidence(**item) for item in payload["evidence"]],
        )
    return {
        "ok": True,
        "seed": payload["dataset_id"],
        "replace_graph": bool(replace_graph),
        "graph_brain": {
            "layer_index": {key: list(value) for key, value in graph_brain.layer_index.items()},
            "stats": dict(graph_brain.stats),
        },
        **stats,
    }
