from __future__ import annotations

from typing import Any


def detect_loops(graph_payload: dict[str, Any]) -> dict[str, Any]:
    edges = graph_payload.get("edges", [])
    self_loops = [edge for edge in edges if edge.get("src_id") == edge.get("dst_id")]
    pair_set: set[tuple[str, str]] = set()
    two_cycles: list[dict[str, Any]] = []
    for edge in edges:
        src = str(edge.get("src_id") or "")
        dst = str(edge.get("dst_id") or "")
        if not src or not dst or src == dst:
            continue
        if (dst, src) in pair_set:
            two_cycles.append({"a": min(src, dst), "b": max(src, dst), "type": "two_cycle"})
        pair_set.add((src, dst))
    unique_cycles = sorted({(c["a"], c["b"]) for c in two_cycles})
    normalized_two_cycles = [{"a": a, "b": b, "type": "two_cycle"} for a, b in unique_cycles]
    return {
        "count": len(self_loops) + len(normalized_two_cycles),
        "self_loops": self_loops,
        "two_cycles": normalized_two_cycles,
    }
