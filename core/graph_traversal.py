from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .graph_core import GraphEdge, GraphMemory, GraphNode, RAMContextGraph, RankedContextNode


_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_'-]{3,}", flags=re.UNICODE)
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about", "what", "when", "like", "just", "have",
    "your", "their", "would", "could", "should", "если", "когда", "потому", "что", "это", "как", "для", "или",
    "про", "надо", "есть", "был", "была", "были", "через", "после", "очень",
}


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(str(text or "")) if token.lower() not in _STOPWORDS]


@dataclass(frozen=True)
class TraversalResult:
    query_tokens: tuple[str, ...]
    signal_tokens: tuple[str, ...]
    ranked_nodes: tuple[RankedContextNode, ...]
    ram_graph: RAMContextGraph


class GraphTraversalEngine:
    def detect_signals(self, query: str) -> dict[str, list[str]]:
        lowered = str(query or "").lower()
        tokens = _tokenize(query)
        emotion = [token for token in tokens if token in {"fear", "anger", "shame", "guilt", "stress", "стыд", "страх", "гнев", "вина"}]
        logic = [token for token in tokens if token in {"because", "therefore", "if", "unless", "если", "потому", "следовательно"}]
        intent = [token for token in tokens if token in {"control", "push", "pressure", "манипуля", "давлен"}]
        social = [token for token in tokens if token in {"family", "boss", "friend", "client", "семья", "друг", "клиент", "началь"}]
        references = [token for token in tokens if token in {"history", "again", "after", "before", "история", "снова", "после", "до"}]
        if "?" in lowered:
            logic.append("question")
        return {
            "entities": sorted(set(tokens[:8])),
            "emotion": sorted(set(emotion)),
            "logic": sorted(set(logic)),
            "intent": sorted(set(intent)),
            "social": sorted(set(social)),
            "references": sorted(set(references)),
        }

    def _node_text(self, node: GraphNode) -> str:
        parts = [
            node.node_core.name,
            node.node_core.description,
            str(node.node_core.metadata.get("plain_explanation") or ""),
            str(node.node_core.metadata.get("what_it_is") or ""),
            str(node.node_core.metadata.get("how_it_works") or ""),
            str(node.node_core.metadata.get("how_to_recognize") or ""),
            " ".join(node.node_branches.patterns),
            " ".join(node.node_branches.examples),
            " ".join(node.node_branches.signals),
            " ".join(node.node_context.tags),
        ]
        return " ".join(part for part in parts if part).lower()

    def _relation_bonus(self, left_type: str, right_type: str, weight: float) -> float:
        base = max(weight, 0.15)
        pair = (left_type.upper(), right_type.upper())
        if pair in {("SIGNAL", "PATTERN"), ("PATTERN", "SIGNAL")}:
            return 0.95 * base
        if pair in {("PERSON", "TRAIT"), ("TRAIT", "PERSON")}:
            return 0.74 * base
        if pair in {("PERSON", "PROFESSION"), ("PROFESSION", "PERSON")}:
            return 0.68 * base
        if pair in {("EXAMPLE", "SIGNAL"), ("SIGNAL", "EXAMPLE"), ("EXAMPLE", "PATTERN"), ("PATTERN", "EXAMPLE")}:
            return 0.46 * base
        if pair in {("AGENT", "PROFESSION"), ("PROFESSION", "AGENT")}:
            return 0.52 * base
        if pair in {("DOMAIN", "CONCEPT"), ("CONCEPT", "DOMAIN"), ("CONCEPT", "PATTERN"), ("PATTERN", "CONCEPT")}:
            return 0.95 * base
        return 0.16 * base

    def traverse(
        self,
        graph_memory: GraphMemory,
        *,
        query: str,
        recent_history: tuple[str, ...] = (),
        focus_node_id: str | None = None,
        max_nodes: int = 16,
    ) -> TraversalResult:
        query_tokens = tuple(sorted(set(_tokenize(query))))
        history_tokens = tuple(sorted(set(_tokenize(" ".join(recent_history)))))
        signals = self.detect_signals(query)
        signal_tokens = tuple(sorted({token for values in signals.values() for token in values}))
        score_map: dict[str, dict[str, Any]] = {}

        for node_id, node in graph_memory.nodes.items():
            text = self._node_text(node)
            node_tokens = set(_tokenize(text))
            overlap = len(node_tokens & set(query_tokens))
            history_overlap = len(node_tokens & set(history_tokens))
            signal_overlap = sum(1 for token in signal_tokens if token and token in text)
            iv = node.node_core.importance_vector.as_dict()
            type_name = node.node_core.node_type.upper()
            type_bonus = 0.0
            if type_name in {"PATTERN", "AGENT", "DOMAIN", "PROFESSION", "CONCEPT"}:
                type_bonus += iv["logic_weight"] * 0.35
            if type_name in {"PERSON", "TRAIT", "SIGNAL"}:
                type_bonus += iv["emotion_weight"] * 0.35
            if type_name in {"PATTERN", "SIGNAL", "AGENT", "CONCEPT"}:
                type_bonus += iv["risk_weight"] * 0.25
            score = (
                overlap * (1.45 + iv["relevance_weight"])
                + history_overlap * 0.55
                + signal_overlap * (0.85 + iv["emotion_weight"] * 0.5 + iv["risk_weight"] * 0.45)
                + type_bonus
            )
            if type_name == "PATTERN" and overlap:
                score += overlap * (0.8 + iv["risk_weight"] * 0.6)
            if type_name == "SIGNAL" and (signal_overlap or overlap):
                score += signal_overlap * 0.6 + overlap * 0.35
            if focus_node_id and node_id == focus_node_id:
                score += 4.0
            score_map[node_id] = {
                "score": round(max(score, 0.0), 6),
                "overlap": overlap,
                "history_overlap": history_overlap,
                "signal_overlap": signal_overlap,
            }

        for edge in graph_memory.edges:
            src = graph_memory.get_node(edge.src_id)
            dst = graph_memory.get_node(edge.dst_id)
            if src is None or dst is None:
                continue
            src_score = float(score_map.get(edge.src_id, {}).get("score") or 0.0)
            dst_score = float(score_map.get(edge.dst_id, {}).get("score") or 0.0)
            if src_score > 0:
                propagated = min(src_score, 6.0) * self._relation_bonus(src.node_core.node_type, dst.node_core.node_type, edge.weight)
                score_map.setdefault(edge.dst_id, {"score": 0.0, "overlap": 0, "history_overlap": 0, "signal_overlap": 0})
                score_map[edge.dst_id]["score"] = round(float(score_map[edge.dst_id]["score"]) + propagated, 6)
            if dst_score > 0:
                propagated = min(dst_score, 6.0) * self._relation_bonus(dst.node_core.node_type, src.node_core.node_type, edge.weight)
                score_map.setdefault(edge.src_id, {"score": 0.0, "overlap": 0, "history_overlap": 0, "signal_overlap": 0})
                score_map[edge.src_id]["score"] = round(float(score_map[edge.src_id]["score"]) + propagated, 6)

        ranked_ids = [
            node_id
            for node_id, item in sorted(score_map.items(), key=lambda row: (-float(row[1]["score"]), row[0]))
            if float(item["score"]) > 0.0
        ]
        seed_ids = ranked_ids[: max(5, max_nodes // 2)]
        if focus_node_id and focus_node_id in graph_memory.nodes and focus_node_id not in seed_ids:
            seed_ids.insert(0, focus_node_id)

        selected_ids = list(seed_ids)
        for edge in graph_memory.edges:
            if edge.src_id in seed_ids and edge.dst_id not in selected_ids:
                selected_ids.append(edge.dst_id)
            if edge.dst_id in seed_ids and edge.src_id not in selected_ids:
                selected_ids.append(edge.src_id)
            if len(selected_ids) >= max_nodes:
                break
        selected_ids = selected_ids[:max_nodes]

        selected_nodes = tuple(graph_memory.nodes[node_id] for node_id in selected_ids if node_id in graph_memory.nodes)
        selected_edges = tuple(
            edge for edge in graph_memory.edges if edge.src_id in selected_ids and edge.dst_id in selected_ids
        )
        ranked_nodes = tuple(
            RankedContextNode(
                node=graph_memory.nodes[node_id],
                score=float(score_map[node_id]["score"]),
                reasons=tuple(
                    reason
                    for reason in [
                        f"token_overlap={int(score_map[node_id]['overlap'])}",
                        f"signal_overlap={int(score_map[node_id]['signal_overlap'])}",
                        f"history_overlap={int(score_map[node_id]['history_overlap'])}",
                    ]
                    if not reason.endswith("=0")
                ),
            )
            for node_id in selected_ids
            if node_id in graph_memory.nodes and float(score_map.get(node_id, {}).get("score") or 0.0) > 0.0
        )
        ram_graph = RAMContextGraph(
            query=query,
            signals={"query_signals": signals},
            nodes=selected_nodes,
            edges=selected_edges,
            ranked_nodes=ranked_nodes,
            recent_history=recent_history,
        )
        return TraversalResult(
            query_tokens=query_tokens,
            signal_tokens=signal_tokens,
            ranked_nodes=ranked_nodes,
            ram_graph=ram_graph,
        )
