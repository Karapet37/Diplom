from __future__ import annotations

import re
from typing import Any

from roaches_viz.roaches_viz.micro_signals import extract_micro_signals


_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_'-]{3,}", flags=re.UNICODE)
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about", "what", "when", "like", "just", "have",
    "your", "their", "would", "could", "should", "если", "когда", "потому", "что", "это", "как", "для", "или",
    "про", "надо", "есть", "был", "была", "были", "через", "после", "очень",
}


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(_normalize_text(text)) if token.lower() not in _STOPWORDS]


def _node_text(node: dict[str, Any]) -> str:
    parts = [
        node.get("name"),
        node.get("label"),
        node.get("description"),
        node.get("what_it_is"),
        node.get("how_it_works"),
        node.get("how_to_recognize"),
        " ".join(str(item) for item in list(node.get("examples") or [])),
        " ".join(str(item) for item in list(node.get("tags") or [])),
        " ".join(str(item) for item in list(node.get("speech_patterns") or [])),
        " ".join(str(item) for item in list(node.get("behavior_patterns") or [])),
    ]
    return " ".join(_normalize_text(item) for item in parts if _normalize_text(item))


def _query_signal_hints(tokens: set[str]) -> tuple[list[str], list[str], list[str]]:
    concepts = sorted(tokens)[:6]
    patterns: list[str] = []
    triggers: list[str] = []
    if "guilt" in tokens and ("pressure" in tokens or "trip" in tokens):
        patterns.append("guilt_pressure")
    if "blame" in tokens or ("fault" in tokens and "your" in tokens):
        patterns.append("blame_shifting")
    if "boundary" in tokens or "limit" in tokens:
        patterns.append("boundary_testing")
    if "avoid" in tokens or "peace" in tokens:
        patterns.append("conflict_avoidance")
    if "criticism" in tokens or "criticize" in tokens:
        triggers.append("criticism")
    if "rejection" in tokens or "ignored" in tokens:
        triggers.append("rejection")
    return concepts, patterns, triggers


class GraphEngine:
    def __init__(self, actor):
        self.actor = actor

    def snapshot_payload(self) -> dict[str, Any]:
        snap = self.actor.ask("snapshot", {"edge_type": None, "min_weight": 0.0})
        return {"nodes": list(snap.get("nodes") or []), "edges": list(snap.get("edges") or [])}

    def fast_lookup(self, *, query: str, context: str = "", limit: int = 12) -> dict[str, Any]:
        payload = self.snapshot_payload()
        effective = "\n\n".join(part for part in [_normalize_text(query), _normalize_text(context)] if part).strip()
        query_tokens = set(_tokenize(effective))
        concepts, patterns, triggers = _query_signal_hints(query_tokens)
        micro = extract_micro_signals(effective, concepts=concepts, patterns=patterns, triggers=triggers)
        signal_tokens = {
            str(item.get("key") or "").lower()
            for values in micro.values()
            for item in list(values or [])
            if str(item.get("key") or "").strip()
        }
        ranked: list[dict[str, Any]] = []
        for node in list(payload.get("nodes") or []):
            haystack_tokens = set(_tokenize(_node_text(node)))
            overlap = sorted(query_tokens & haystack_tokens)
            signal_overlap = sorted(signal_tokens & haystack_tokens)
            if not overlap and not signal_overlap:
                continue
            importance = dict(node.get("importance_vector") or {})
            score = (
                len(overlap) * 1.5
                + len(signal_overlap) * 1.2
                + float(importance.get("logic_weight") or 0.0) * 0.4
                + float(importance.get("emotion_weight") or 0.0) * 0.2
                + float(importance.get("risk_weight") or 0.0) * 0.2
                + float(importance.get("relevance_weight") or 0.0) * 0.8
            )
            ranked.append(
                {
                    "node_id": str(node.get("id") or ""),
                    "name": str(node.get("label") or node.get("name") or node.get("id") or ""),
                    "type": str(node.get("type") or ""),
                    "score": round(score, 6),
                    "overlap_tokens": overlap,
                    "signal_overlap": signal_overlap,
                    "description": str(node.get("description") or node.get("short_gloss") or ""),
                }
            )
        ranked.sort(key=lambda item: (-float(item.get("score") or 0.0), str(item.get("node_id") or "")))
        matched_tokens = sorted({token for item in ranked[:limit] for token in list(item.get("overlap_tokens") or [])})
        missing_tokens = sorted(query_tokens - set(matched_tokens))
        return {
            "query": effective,
            "signals": micro,
            "ranked_nodes": ranked[:limit],
            "matched_tokens": matched_tokens,
            "missing_tokens": missing_tokens[:12],
        }

    def apply_materialized_graph(
        self,
        *,
        memory_text: str,
        source_id: str,
        query: str,
        assistant_reply: str,
        person_id: str | None = None,
    ) -> dict[str, Any]:
        return self.actor.ask(
            "apply_materialized_graph",
            {
                "memory_text": memory_text,
                "source_id": source_id,
                "query": query,
                "assistant_reply": assistant_reply,
                "person_id": person_id or "",
            },
            timeout=90.0,
        )
