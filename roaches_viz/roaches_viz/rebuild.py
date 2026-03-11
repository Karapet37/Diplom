from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import default_settings
from .graph_model import Edge, Evidence, Node, Source
from .ingest import build_ingest_payload
from .foundations import build_foundation_payload, foundation_from_source, is_foundation_source
from .store import GraphStore


def _flatten_evidence(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for edge in edges:
        for ev in edge.get("evidence", []):
            rows.append(
                {
                    "edge_key": edge["edge_key"],
                    "source_id": ev["source_id"],
                    "snippet_text": ev["snippet_text"],
                    "offset_start": int(ev.get("offset_start", 0)),
                    "offset_end": int(ev.get("offset_end", 0)),
                }
            )
    return rows


def rebuild_graph_from_sources(
    store: GraphStore,
    *,
    mode: str = "full",
    source_ids: list[str] | None = None,
    top_tokens_per_sentence: int = 5,
) -> dict[str, Any]:
    known_sources = store.list_sources()
    ordered = sorted(known_sources, key=lambda row: str(row.get("source_id") or ""))
    if mode == "scoped":
        wanted = {sid for sid in (source_ids or []) if sid}
        if not wanted:
            return {"ok": False, "error": "scoped rebuild requires source_ids"}
        selected = [row for row in ordered if row["source_id"] in wanted]
    else:
        selected = ordered

    temp_store = GraphStore(Path(":memory:"))
    try:
        for row in selected:
            if is_foundation_source(str(row["raw_text"])):
                seed_payload = build_foundation_payload(foundation_from_source(str(row["raw_text"])))
                temp_store.apply_batch(
                    source=seed_payload["source"],
                    nodes=[Node(**item) for item in seed_payload["nodes"]],
                    edges=[Edge(**item) for item in seed_payload["edges"]],
                    evidence=[Evidence(**item) for item in seed_payload["evidence"]],
                )
            else:
                nodes, edges, evidence = build_ingest_payload(
                    row["source_id"],
                    row["raw_text"],
                    top_tokens_per_sentence=top_tokens_per_sentence,
                )
                temp_store.apply_batch(
                    source=Source(source_id=row["source_id"], raw_text=row["raw_text"]),
                    nodes=nodes,
                    edges=edges,
                    evidence=evidence,
                )
        payload = temp_store.export_graph()
    finally:
        temp_store.close()

    evidence_rows = _flatten_evidence(payload["edges"])
    replaced = store.replace_graph(nodes=payload["nodes"], edges=payload["edges"], evidence=evidence_rows)
    return {
        "ok": True,
        "mode": mode,
        "sources_processed": len(selected),
        "selected_source_ids": [row["source_id"] for row in selected],
        **replaced,
    }


def rebuild_all() -> dict[str, object]:
    settings = default_settings(Path(__file__).resolve().parents[1])
    store = GraphStore(settings.db_path)
    try:
        return rebuild_graph_from_sources(
            store,
            mode="full",
            source_ids=None,
            top_tokens_per_sentence=settings.top_tokens_per_sentence,
        )
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild graph from source corpus")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    settings = default_settings(Path(__file__).resolve().parents[1])
    store = GraphStore(settings.db_path)
    try:
        if args.all:
            result = rebuild_graph_from_sources(store, mode="full", top_tokens_per_sentence=args.top_k)
        else:
            result = rebuild_graph_from_sources(
                store,
                mode="scoped",
                source_ids=list(args.source_id or []),
                top_tokens_per_sentence=args.top_k,
            )
    finally:
        store.close()
    print(result)


if __name__ == "__main__":
    main()
