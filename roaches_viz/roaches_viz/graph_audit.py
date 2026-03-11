from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .analysis_loops import detect_loops


_PRACTICAL_TYPES = {"CONCEPT", "PATTERN", "TRAIT", "TRIGGER", "DOMAIN", "SIGNAL"}


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _compact_key(value: str) -> str:
    return re.sub(r"[^a-zа-яё0-9]+", "", value.lower(), flags=re.IGNORECASE)


def _node_display(node: dict[str, Any]) -> str:
    return _norm_text(node.get("name") or node.get("label") or node.get("id") or "node")


def _practical_field_findings(node: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    node_type = _norm_text(node.get("type")).upper()
    if node_type not in _PRACTICAL_TYPES:
        return findings
    description = _norm_text(node.get("description") or node.get("short_gloss"))
    what_it_is = _norm_text(node.get("what_it_is") or node.get("plain_explanation"))
    how_it_works = _norm_text(node.get("how_it_works"))
    how_to_recognize = _norm_text(node.get("how_to_recognize"))

    if len(description) < 24:
        findings.append("description_too_short")
    if len(what_it_is) < 24:
        findings.append("missing_practical_what_it_is")
    if node_type in {"CONCEPT", "PATTERN", "TRAIT", "TRIGGER", "DOMAIN"} and len(how_it_works) < 18:
        findings.append("missing_how_it_works")
    if node_type in {"CONCEPT", "PATTERN", "TRAIT", "TRIGGER", "SIGNAL"} and len(how_to_recognize) < 18:
        findings.append("missing_how_to_recognize")
    return findings


def _is_weak_description(node: dict[str, Any]) -> bool:
    name = _node_display(node)
    description = _norm_text(node.get("description") or node.get("short_gloss"))
    if len(description) < 24:
        return True
    normalized_name = _compact_key(name)
    normalized_description = _compact_key(description)
    if normalized_name and normalized_name == normalized_description:
        return True
    return False


def build_graph_audit(graph_payload: dict[str, Any]) -> dict[str, Any]:
    nodes = list(graph_payload.get("nodes") or [])
    edges = list(graph_payload.get("edges") or [])
    node_ids = {str(node.get("id") or "") for node in nodes}
    incident_counts: dict[str, int] = {node_id: 0 for node_id in node_ids}
    low_confidence_edges: list[dict[str, Any]] = []
    for edge in edges:
        src_id = str(edge.get("src_id") or edge.get("from") or "")
        dst_id = str(edge.get("dst_id") or edge.get("to") or "")
        if src_id in incident_counts:
            incident_counts[src_id] += 1
        if dst_id in incident_counts:
            incident_counts[dst_id] += 1
        if float(edge.get("confidence") or 0.0) < 0.45:
            low_confidence_edges.append(
                {
                    "edge_key": edge.get("edge_key") or f"{src_id}|{edge.get('type') or edge.get('relation_type')}|{dst_id}",
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "type": edge.get("type") or edge.get("relation_type") or "",
                    "confidence": float(edge.get("confidence") or 0.0),
                }
            )

    orphan_nodes: list[dict[str, Any]] = []
    weak_nodes: list[dict[str, Any]] = []
    missing_fields: list[dict[str, Any]] = []
    node_findings_by_id: dict[str, list[str]] = {}
    duplicate_index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for node in nodes:
        node_id = str(node.get("id") or "")
        node_type = _norm_text(node.get("type")).upper()
        display = _node_display(node)
        duplicate_index[(node_type, _compact_key(display))].append(node)

        findings = _practical_field_findings(node)
        if findings:
            node_findings_by_id[node_id] = findings
            missing_fields.append(
                {
                    "id": node_id,
                    "name": display,
                    "type": node_type,
                    "findings": findings,
                }
            )

        if _is_weak_description(node):
            weak_nodes.append(
                {
                    "id": node_id,
                    "name": display,
                    "type": node_type,
                    "description": _norm_text(node.get("description") or node.get("short_gloss")),
                }
            )

        if incident_counts.get(node_id, 0) == 0 and node_type not in {"DOMAIN"}:
            orphan_nodes.append(
                {
                    "id": node_id,
                    "name": display,
                    "type": node_type,
                }
            )

    duplicate_candidates: list[dict[str, Any]] = []
    for (node_type, compact_name), group in sorted(duplicate_index.items()):
        if not compact_name or len(group) < 2:
            continue
        duplicate_candidates.append(
            {
                "type": node_type,
                "name": _node_display(group[0]),
                "node_ids": sorted(str(node.get("id") or "") for node in group),
                "count": len(group),
            }
        )

    loops = detect_loops({"edges": edges})
    checks = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "orphan_nodes": len(orphan_nodes),
        "weak_descriptions": len(weak_nodes),
        "missing_practical_fields": len(missing_fields),
        "duplicate_candidates": len(duplicate_candidates),
        "low_confidence_edges": len(low_confidence_edges),
        "contradictions": int(loops.get("count") or 0),
        "signal_nodes": sum(1 for node in nodes if _norm_text(node.get("type")).upper() == "SIGNAL"),
        "trait_nodes": sum(1 for node in nodes if _norm_text(node.get("type")).upper() == "TRAIT"),
    }

    penalty = (
        checks["orphan_nodes"] * 2.5
        + checks["weak_descriptions"] * 1.8
        + checks["missing_practical_fields"] * 1.3
        + checks["duplicate_candidates"] * 4.0
        + checks["low_confidence_edges"] * 0.8
        + checks["contradictions"] * 1.5
    )
    score = max(0, round(100 - penalty))

    recommendations: list[str] = []
    if checks["weak_descriptions"]:
        recommendations.append("Rewrite weak node descriptions so a tired reader can still understand the node quickly.")
    if checks["missing_practical_fields"]:
        recommendations.append("Fill what_it_is, how_it_works, and how_to_recognize for concept-like nodes.")
    if checks["duplicate_candidates"]:
        recommendations.append("Review duplicate-looking nodes before the graph drifts into parallel copies of the same concept.")
    if checks["orphan_nodes"]:
        recommendations.append("Connect orphan nodes or delete them if they do not support any pattern, concept, or example.")
    if checks["low_confidence_edges"]:
        recommendations.append("Review low-confidence edges and either strengthen them with better evidence or prune them.")
    if not recommendations:
        recommendations.append("Graph quality is stable. Keep enriching examples and practical node explanations.")

    return {
        "ok": True,
        "score": score,
        "checks": checks,
        "recommendations": recommendations,
        "orphan_nodes": orphan_nodes[:24],
        "weak_nodes": weak_nodes[:24],
        "missing_fields": missing_fields[:24],
        "duplicate_candidates": duplicate_candidates[:24],
        "low_confidence_edges": low_confidence_edges[:24],
        "node_findings_by_id": node_findings_by_id,
        "loops": loops,
    }
