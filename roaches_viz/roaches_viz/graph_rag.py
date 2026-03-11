from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core import DialogueEngine, GraphTraversalEngine, RAMContextGraph, RankedContextNode, ScenarioEngine, SpeechDNA, default_agent_roles

from .core_bridge import graph_payload_to_memory, graph_payload_to_system_core, payload_edge_to_core, payload_node_to_core, person_node_to_personality
from .llm_runtime import build_fast_llm, build_reasoning_llm
from .micro_signals import extract_micro_signals
from .style_response import apply_style_to_fallback, build_style_prompt_block


_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_'-]{3,}", flags=re.UNICODE)
_REQUEST_RE = re.compile(r"REQUEST-(?P<index>\\d+)\\s*:\\s*(?P<body>.+)", flags=re.IGNORECASE)
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about", "what", "when", "like", "just", "have",
    "your", "their", "would", "could", "should", "если", "когда", "потому", "что", "это", "как", "для", "или",
    "про", "надо", "есть", "был", "была", "были", "через", "после", "очень",
}


def _extract_request_lines(raw: str) -> list[str]:
    requests: list[tuple[int, str]] = []
    for line in str(raw or "").splitlines():
        match = _REQUEST_RE.match(line.strip())
        if not match:
            continue
        body = _norm_text(match.group("body"))
        if not body:
            continue
        requests.append((int(match.group("index") or "0"), body))
    requests.sort(key=lambda item: item[0])
    return [body for _, body in requests]


def _ranked_context_token_set(ram_graph: dict[str, Any]) -> set[str]:
    token_set: set[str] = set()
    for item in list(ram_graph.get("ranked_context") or [])[:8]:
        token_set.update(_tokenize(" ".join([
            str(item.get("name") or ""),
            str(item.get("description") or ""),
            " ".join(str(reason) for reason in list(item.get("reasons") or [])),
        ])))
    return token_set


def build_graph_update_scenario(
    graph_payload: dict[str, Any],
    *,
    query: str,
    assistant_reply: str,
    person_id: str | None = None,
) -> dict[str, Any]:
    effective_query = "\n\n".join(part for part in [_norm_text(query), _norm_text(assistant_reply)] if part).strip()
    ram_graph = build_ram_graph(graph_payload, query=effective_query, recent_history=[_norm_text(query), _norm_text(assistant_reply)], person_id=person_id)
    query_tokens = set(_tokenize(query))
    reply_tokens = set(_tokenize(assistant_reply))
    context_tokens = _ranked_context_token_set(ram_graph)
    missing_tokens = sorted((query_tokens | reply_tokens) - context_tokens)
    ranked_context = list(ram_graph.get("ranked_context") or [])
    context_is_sufficient = len(ranked_context) >= 6 and len(missing_tokens) <= 2
    if context_is_sufficient:
        return {
            "should_update": False,
            "reason": "graph_context_sufficient",
            "missing_tokens": missing_tokens[:6],
            "requests": [],
            "ram_graph": ram_graph,
        }

    request_lines: list[str] = []
    fast_llm = build_fast_llm()
    if fast_llm is not None:
        prompt = json.dumps(
            {
                "task": "graph_update_scenario",
                "instructions": [
                    "Read the query and current reply.",
                    "Decide what graph-building work is still missing.",
                    "Return only lines in the form REQUEST-1: ..., REQUEST-2: ...",
                    "Each request must describe what to add and how to ground it.",
                    "Do not exceed 4 requests.",
                ],
                "query": query,
                "assistant_reply": assistant_reply,
                "ranked_context": ranked_context[:6],
                "missing_tokens": missing_tokens[:8],
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            request_lines = _extract_request_lines(fast_llm(prompt))
        except Exception:
            request_lines = []

    if not request_lines:
        request_lines = [
            f"Capture the practical concept around '{token}' and connect it to the current dialogue as an example."
            for token in missing_tokens[:3]
        ]
    request_lines = [item for item in request_lines if _norm_text(item)][:4]
    return {
        "should_update": bool(request_lines),
        "reason": "missing_context_detected" if request_lines else "no_missing_context_requests",
        "missing_tokens": missing_tokens[:8],
        "requests": request_lines,
        "ram_graph": ram_graph,
    }


def materialize_graph_update_requests(
    graph_payload: dict[str, Any],
    *,
    query: str,
    assistant_reply: str,
    requests: list[str],
    person_id: str | None = None,
    llm_role: str = "general",
) -> dict[str, Any]:
    effective_query = "\n\n".join(part for part in [_norm_text(query), _norm_text(assistant_reply)] if part).strip()
    ram_graph = build_ram_graph(graph_payload, query=effective_query, recent_history=[_norm_text(query), _norm_text(assistant_reply)], person_id=person_id)
    context_text = _context_text(ram_graph, personality={}, agent_plan=[])
    reasoning_llm = build_reasoning_llm(llm_role)
    chunks: list[str] = []

    for index, request_text in enumerate([item for item in requests if _norm_text(item)][:4], start=1):
        if reasoning_llm is not None:
            prompt = json.dumps(
                {
                    "task": "graph_update_request_materialization",
                    "instructions": [
                        "Use the request, the original query, the assistant reply, and the RAM context.",
                        "Produce short graph-ready knowledge notes.",
                        "Prefer practical definitions, behavior patterns, and concrete examples.",
                        "Do not produce historical background or assistant commentary.",
                        "Return plain text, 2 to 5 lines.",
                    ],
                    "request": request_text,
                    "query": query,
                    "assistant_reply": assistant_reply,
                    "ram_context": context_text,
                },
                ensure_ascii=False,
                indent=2,
            )
            try:
                material = _norm_text(reasoning_llm(prompt))
            except Exception:
                material = ""
        else:
            material = ""
        if not material:
            material = "\n".join(
                [
                    f"Request {index}: {request_text}",
                    f"Observed query: {query[:260]}",
                    f"Observed reply: {assistant_reply[:260]}",
                ]
            )
        chunks.append(material)

    memory_text = "\n\n".join(chunks).strip()[:2200]
    return {
        "requests": [item for item in requests if _norm_text(item)][:4],
        "memory_text": memory_text,
        "source_preview": memory_text[:320],
        "ram_graph": ram_graph,
    }
_AGENT_BLUEPRINTS = {
    "agent:signal_analyst": {
        "name": "Signal Analyst",
        "description": "Reads micro-signals, emotional cues, hidden pressure, and ambiguity before any conclusion.",
        "profession": "profession:research",
        "hints": ["tone", "signal", "emotion", "intent", "cue", "тон", "эмоци", "сигнал", "намек"],
    },
    "agent:strategy_planner": {
        "name": "Strategy Planner",
        "description": "Builds next-step plans from grounded graph context, constraints, and pattern chains.",
        "profession": "profession:strategy",
        "hints": ["plan", "strategy", "next", "how", "should", "стратег", "план", "как", "дальше"],
    },
    "agent:legal_reasoner": {
        "name": "Legal Reasoner",
        "description": "Frames rights, duties, contracts, liability, and escalation risk from graph evidence.",
        "profession": "profession:law",
        "hints": ["law", "legal", "right", "duty", "contract", "закон", "прав", "обязан", "суд"],
    },
    "agent:business_operator": {
        "name": "Business Operator",
        "description": "Reads incentives, power, negotiation pressure, and operational friction.",
        "profession": "profession:business",
        "hints": ["deal", "client", "boss", "manager", "money", "клиент", "менеджер", "босс", "деньги"],
    },
    "agent:research_grounder": {
        "name": "Research Grounder",
        "description": "Prioritizes examples, source quality, and evidence-backed grounding before response generation.",
        "profession": "profession:research",
        "hints": ["source", "evidence", "proof", "study", "wiki", "источник", "доказ", "пример", "вики"],
    },
}
_PROFESSION_BLUEPRINTS = {
    "profession:law": "Legal practice; formal language, evidence discipline, and risk awareness.",
    "profession:business": "Business operations; incentive awareness, negotiation pressure, and pragmatic tradeoffs.",
    "profession:strategy": "Strategy work; structured planning, scenario comparison, and sequencing of moves.",
    "profession:research": "Research work; source quality, explicit uncertainty, and evidence-first interpretation.",
    "profession:therapy": "Therapeutic work; emotional pacing, relational patterns, and safer intervention language.",
}


def _serialize_core_system(graph_payload: dict[str, Any]) -> dict[str, Any]:
    return graph_payload_to_system_core(graph_payload).describe()


def _build_ram_core_object(*, query: str, recent_history: list[str] | None, micro: dict[str, Any], nodes: list[dict[str, Any]], edges: list[dict[str, Any]], ranked_context: list[dict[str, Any]]) -> RAMContextGraph:
    node_index = {str(node.get("id") or ""): payload_node_to_core(node) for node in nodes}
    ranked_nodes: list[RankedContextNode] = []
    for item in ranked_context:
        node_id = str(item.get("node_id") or "")
        core_node = node_index.get(node_id)
        if core_node is None:
            continue
        ranked_nodes.append(
            RankedContextNode(
                node=core_node,
                score=float(item.get("score") or 0.0),
                reasons=tuple(
                    reason
                    for reason in [
                        f"token_overlap={int(item.get('overlap') or 0)}" if item.get("overlap") is not None else "",
                        f"signal_overlap={int(item.get('signal_overlap') or 0)}" if item.get("signal_overlap") is not None else "",
                        f"history_overlap={int(item.get('history_overlap') or 0)}" if item.get("history_overlap") is not None else "",
                    ]
                    if reason
                ),
            )
        )
    return RAMContextGraph(
        query=query,
        signals={"micro_signals": micro},
        nodes=tuple(node_index.values()),
        edges=tuple(payload_edge_to_core(edge) for edge in edges),
        ranked_nodes=tuple(ranked_nodes),
        recent_history=tuple(recent_history or []),
    )


def _serialize_ram_core(*, query: str, recent_history: list[str] | None, micro: dict[str, Any], nodes: list[dict[str, Any]], edges: list[dict[str, Any]], ranked_context: list[dict[str, Any]]) -> dict[str, Any]:
    ram_core = _build_ram_core_object(
        query=query,
        recent_history=recent_history,
        micro=micro,
        nodes=nodes,
        edges=edges,
        ranked_context=ranked_context,
    )
    payload = ram_core.as_dict()
    return payload


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raw = _norm_text(value)
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return []


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    raw = _norm_text(value)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(_norm_text(text)) if token.lower() not in _STOPWORDS]


def _node_text(node: dict[str, Any]) -> str:
    parts = [
        node.get("name"),
        node.get("label"),
        node.get("description"),
        node.get("what_it_is"),
        node.get("how_it_works"),
        node.get("how_to_recognize"),
        node.get("background"),
        node.get("profession"),
        node.get("temperament"),
        " ".join(_json_list(node.get("examples_json") or node.get("examples"))),
        " ".join(_json_list(node.get("tags_json") or node.get("tags"))),
        " ".join(_json_list(node.get("speech_patterns_json") or node.get("speech_patterns"))),
        " ".join(_json_list(node.get("behavior_patterns_json") or node.get("behavior_patterns"))),
        " ".join(_json_list(node.get("reaction_logic_json") or node.get("reaction_logic"))),
        " ".join(_json_list(node.get("conflict_patterns_json") or node.get("conflict_patterns"))),
        " ".join(_json_list(node.get("values_json") or node.get("values"))),
        " ".join(_json_list(node.get("preferences_json") or node.get("preferences"))),
        " ".join(_json_list(node.get("possible_intents_json") or node.get("possible_intents"))),
        " ".join(_json_list(node.get("emotion_signals_json") or node.get("emotion_signals"))),
    ]
    return " ".join(_norm_text(item) for item in parts if _norm_text(item))


def _node_importance(node: dict[str, Any]) -> dict[str, float]:
    iv = dict((node.get("importance_vector") or {}))
    return {
        "logic_weight": float(iv.get("logic_weight") or node.get("logic_weight") or 0.5),
        "emotion_weight": float(iv.get("emotion_weight") or node.get("emotion_weight") or 0.5),
        "risk_weight": float(iv.get("risk_weight") or node.get("risk_weight") or 0.5),
        "relevance_weight": float(iv.get("relevance_weight") or node.get("relevance_weight") or 0.5),
    }


def list_person_nodes(graph_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [node for node in list(graph_payload.get("nodes") or []) if _norm_text(node.get("type")).upper() == "PERSON"],
        key=lambda item: _norm_text(item.get("name") or item.get("label") or item.get("id")).lower(),
    )


def list_agent_nodes(graph_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [node for node in list(graph_payload.get("nodes") or []) if _norm_text(node.get("type")).upper() == "AGENT"],
        key=lambda item: _norm_text(item.get("name") or item.get("label") or item.get("id")).lower(),
    )


def list_profession_nodes(graph_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [node for node in list(graph_payload.get("nodes") or []) if _norm_text(node.get("type")).upper() == "PROFESSION"],
        key=lambda item: _norm_text(item.get("name") or item.get("label") or item.get("id")).lower(),
    )


def build_ram_graph(
    graph_payload: dict[str, Any],
    *,
    query: str,
    recent_history: list[str] | None = None,
    person_id: str | None = None,
    max_nodes: int = 16,
) -> dict[str, Any]:
    graph_memory = graph_payload_to_memory(graph_payload)
    traversal = GraphTraversalEngine().traverse(
        graph_memory,
        query=query,
        recent_history=tuple(recent_history or []),
        focus_node_id=person_id,
        max_nodes=max_nodes,
    )
    nodes_by_id = {str(node.get("id") or ""): node for node in list(graph_payload.get("nodes") or [])}
    edges = list(graph_payload.get("edges") or [])
    selected_ids = [node.node_id for node in traversal.ram_graph.nodes]
    ram_nodes = [nodes_by_id[node_id] for node_id in selected_ids if node_id in nodes_by_id]
    ram_edges = [edge for edge in edges if str(edge.get("src_id") or "") in selected_ids and str(edge.get("dst_id") or "") in selected_ids]
    micro = extract_micro_signals(query, concepts=list(traversal.query_tokens)[:4], patterns=[], triggers=[])
    ranked_context = [
        {
            "node_id": item.node.node_id,
            "name": item.node.node_core.name,
            "type": item.node.node_core.node_type,
            "description": item.node.node_core.description,
            "importance_vector": item.node.node_core.importance_vector.as_dict(),
            "score": float(item.score),
            "reasons": list(item.reasons),
        }
        for item in traversal.ranked_nodes
    ]
    result = {
        "query_tokens": list(traversal.query_tokens),
        "signal_tokens": list(traversal.signal_tokens),
        "micro_signals": micro,
        "nodes": ram_nodes,
        "edges": ram_edges,
        "ranked_context": ranked_context,
    }
    result["ram_core"] = _serialize_ram_core(
        query=query,
        recent_history=recent_history,
        micro=micro,
        nodes=ram_nodes,
        edges=ram_edges,
        ranked_context=ranked_context,
    )
    return result


def _linked_nodes(graph_payload: dict[str, Any], node_id: str, *, allowed_types: set[str] | None = None, limit: int = 8) -> list[dict[str, Any]]:
    nodes = {str(node.get("id") or ""): node for node in list(graph_payload.get("nodes") or [])}
    edges = list(graph_payload.get("edges") or [])
    linked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for edge in edges:
        src_id = str(edge.get("src_id") or "")
        dst_id = str(edge.get("dst_id") or "")
        other_id = ""
        if src_id == node_id:
            other_id = dst_id
        elif dst_id == node_id:
            other_id = src_id
        if not other_id or other_id in seen or other_id not in nodes:
            continue
        other = nodes[other_id]
        other_type = _norm_text(other.get("type")).upper()
        if allowed_types and other_type not in allowed_types:
            continue
        linked.append(other)
        seen.add(other_id)
        if len(linked) >= limit:
            break
    return linked


def build_agent_plan(graph_payload: dict[str, Any], query: str, *, person_id: str | None = None) -> list[dict[str, Any]]:
    nodes = list(graph_payload.get("nodes") or [])
    query_text = " ".join(_tokenize(query))
    person_node = {str(node.get("id") or ""): node for node in nodes}.get(str(person_id or "")) if person_id else None
    profession_name = _norm_text(person_node.get("profession") if person_node else "")
    plan: list[dict[str, Any]] = []
    for node in nodes:
        if _norm_text(node.get("type")).upper() != "AGENT":
            continue
        node_id = str(node.get("id") or "")
        blueprint = _AGENT_BLUEPRINTS.get(node_id, {})
        hints = list(blueprint.get("hints") or [])
        score = float(_node_importance(node)["logic_weight"])
        score += sum(1 for hint in hints if hint.lower() in query_text)
        if profession_name and profession_name in node_id:
            score += 0.8
        if person_node and profession_name and blueprint.get("profession") == f"profession:{profession_name}":
            score += 0.9
        if score <= 0.45:
            continue
        plan.append(
            {
                "agent_id": node_id,
                "name": _norm_text(node.get("name") or node.get("label") or node_id),
                "description": _norm_text(node.get("description") or blueprint.get("description") or ""),
                "profession": _norm_text(blueprint.get("profession") or ""),
                "score": round(score, 6),
            }
        )
    if not plan:
        plan = [
            {"agent_id": "agent:signal_analyst", "name": "Signal Analyst", "description": "Extracts tone, intent, and micro-signals.", "profession": "profession:research", "score": 1.0},
            {"agent_id": "agent:strategy_planner", "name": "Strategy Planner", "description": "Builds a grounded reasoning plan from graph context.", "profession": "profession:strategy", "score": 0.9},
        ]
    return sorted(plan, key=lambda item: (-float(item["score"]), item["agent_id"]))[:4]


def _build_personality_profile(graph_payload: dict[str, Any], person_node: dict[str, Any] | None) -> dict[str, Any]:
    if not person_node:
        return {}
    node_id = str(person_node.get("id") or "")
    traits = _linked_nodes(graph_payload, node_id, allowed_types={"TRAIT"}, limit=6)
    professions = _linked_nodes(graph_payload, node_id, allowed_types={"PROFESSION"}, limit=3)
    patterns = _linked_nodes(graph_payload, node_id, allowed_types={"PATTERN"}, limit=6)
    examples = _linked_nodes(graph_payload, node_id, allowed_types={"EXAMPLE"}, limit=4)
    speech_style = _json_object(person_node.get("speech_style_json") or person_node.get("speech_style"))
    if not speech_style:
        speech_style = dict(person_node.get("speech_profile") or {})
    profession_name = _norm_text(person_node.get("profession"))
    if not profession_name and professions:
        profession_name = _norm_text(professions[0].get("name"))
    profile = {
        "id": node_id,
        "name": _norm_text(person_node.get("name") or person_node.get("label") or node_id),
        "background": _norm_text(person_node.get("background")),
        "profession": profession_name,
        "temperament": _norm_text(person_node.get("temperament")),
        "speech_style": speech_style,
        "tolerance_threshold": float(person_node.get("tolerance_threshold") or 0.5),
        "values": _json_list(person_node.get("values_json") or person_node.get("values")),
        "speech_patterns": _json_list(person_node.get("speech_patterns_json") or person_node.get("speech_patterns")),
        "behavior_patterns": _json_list(person_node.get("behavior_patterns_json") or person_node.get("behavior_patterns")),
        "reaction_logic": _json_list(person_node.get("reaction_logic_json") or person_node.get("reaction_logic")),
        "conflict_patterns": _json_list(person_node.get("conflict_patterns_json") or person_node.get("conflict_patterns")),
        "traits": [_norm_text(item.get("name") or item.get("label") or item.get("id")) for item in traits],
        "pattern_nodes": [_norm_text(item.get("name") or item.get("label") or item.get("id")) for item in patterns],
        "example_lines": [_norm_text(item.get("plain_explanation") or item.get("description") or item.get("name")) for item in examples],
    }
    return profile


def _style_directive(profile: dict[str, Any]) -> str:
    if not profile:
        return "Respond as a grounded human analyst, not as a polite assistant."
    speech = dict(profile.get("speech_style") or {})
    directness = float(speech.get("directness") or 0.5)
    formality = float(speech.get("formality") or 0.5)
    slang = float(speech.get("slang_level") or 0.0)
    temperament = _norm_text(profile.get("temperament")) or "guarded"
    tone_bits: list[str] = []
    tone_bits.append("high directness" if directness >= 0.68 else "measured directness" if directness >= 0.45 else "indirect cautious phrasing")
    tone_bits.append("formal wording" if formality >= 0.68 else "mixed formal/casual wording" if formality >= 0.42 else "casual wording")
    if slang >= 0.45:
        tone_bits.append("some slang is natural")
    if temperament:
        tone_bits.append(f"temperament={temperament}")
    return "Use this human profile: " + ", ".join(tone_bits) + ". Do not sound like a helpful assistant."


def _context_text(ram_graph: dict[str, Any], *, personality: dict[str, Any], agent_plan: list[dict[str, Any]]) -> str:
    lines = [
        "Use only the RAM graph below. If the graph is insufficient, say what is missing.",
        "Do not rely on unstated internal knowledge.",
    ]
    if personality:
        lines.extend(
            [
                f"Person: {personality.get('name')}",
                f"Background: {personality.get('background')}",
                f"Profession: {personality.get('profession')}",
                f"Temperament: {personality.get('temperament')}",
                f"Values: {', '.join(personality.get('values') or [])}",
                f"Traits: {', '.join(personality.get('traits') or [])}",
                f"Behavior patterns: {', '.join(personality.get('behavior_patterns') or [])}",
                f"Reaction logic: {', '.join(personality.get('reaction_logic') or [])}",
                f"Conflict patterns: {', '.join(personality.get('conflict_patterns') or [])}",
                f"Speech patterns: {', '.join(personality.get('speech_patterns') or [])}",
                _style_directive(personality),
            ]
        )
    lines.append("Reasoning agents:")
    for item in agent_plan[:4]:
        lines.append(f"- {item['name']}: {item['description']} [{item.get('profession') or 'general'}]")
    lines.append("RAM graph nodes:")
    for item in ram_graph.get("ranked_context", [])[:10]:
        lines.append(
            f"- [{item['type']}] {item['name']}: {item['description']} | "
            f"logic={item['importance_vector']['logic_weight']:.2f} emotion={item['importance_vector']['emotion_weight']:.2f} "
            f"risk={item['importance_vector']['risk_weight']:.2f} relevance={item['importance_vector']['relevance_weight']:.2f}"
        )
    lines.append("RAM graph relations:")
    for edge in list(ram_graph.get("edges") or [])[:14]:
        lines.append(
            f"- {edge.get('src_id')} --{edge.get('type')}--> {edge.get('dst_id')} "
            f"(w={float(edge.get('weight') or 0.0):.2f}, c={float(edge.get('confidence') or 0.0):.2f})"
        )
    return "\n".join(line for line in lines if line.strip())


def _fallback_grounded_reply(query: str, ram_graph: dict[str, Any], *, personality: dict[str, Any], agent_plan: list[dict[str, Any]]) -> str:
    ranked = list(ram_graph.get("ranked_context") or [])
    if not ranked:
        return "I do not have enough graph-grounded context to answer that like a real person. I need more examples, patterns, or a named person first."
    lead = ranked[0]
    if personality:
        name = _norm_text(personality.get("name")) or "This person"
        profession = _norm_text(personality.get("profession"))
        temperament = _norm_text(personality.get("temperament")) or "guarded"
        values = list(personality.get("values") or [])
        reaction_logic = list(personality.get("reaction_logic") or [])
        traits = list(personality.get("traits") or [])
        opener = "Look,"
        speech = dict(personality.get("speech_style") or {})
        if float(speech.get("formality") or 0.0) >= 0.7:
            opener = "From my point of view,"
        elif float(speech.get("slang_level") or 0.0) >= 0.45:
            opener = "Honestly,"
        pieces = [
            opener,
            f"{name} would read this through {lead['name']} first, not as random chat noise.",
            f"The strongest grounded context is {', '.join(item['name'] for item in ranked[:3])}.",
        ]
        if profession:
            pieces.append(f"Professionally this leans toward {profession} logic.")
        if traits:
            pieces.append(f"The social style here looks {', '.join(traits[:2])}.")
        if reaction_logic:
            pieces.append(f"Typical reaction rule: {reaction_logic[0]}.")
        else:
            pieces.append(f"Temperament baseline: {temperament}.")
        if values:
            pieces.append(f"The values under pressure look like {', '.join(values[:2])}.")
        return " ".join(pieces)
    lead_agents = ", ".join(item["name"] for item in agent_plan[:2])
    return (
        f"The graph says this runs first through {lead['name']}. "
        f"The strongest grounded context nodes are {', '.join(item['name'] for item in ranked[:3])}. "
        f"The active reasoning agents are {lead_agents}."
    )


def _write_personality_log(person_id: str, *, query: str, ram_graph: dict[str, Any], response: str, evaluation: str) -> str:
    logs_dir = Path(__file__).resolve().parents[1] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"{person_id.replace(':', '_')}.txt"
    context_nodes = [item["node_id"] for item in ram_graph.get("ranked_context", [])[:8]]
    signals = ram_graph.get("signal_tokens", [])
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "context_nodes": context_nodes,
        "signals": signals,
        "response": response,
        "evaluation": evaluation,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return str(path)


def ensure_system_agents(graph_store) -> list[str]:
    from .graph_model import Node, Edge

    created: list[str] = []
    for profession_id, description in _PROFESSION_BLUEPRINTS.items():
        if not graph_store.get_node(profession_id):
            graph_store.upsert_node(
                Node(
                    id=profession_id,
                    type="PROFESSION",
                    name=profession_id.split(":", 1)[1].replace("_", " ").title(),
                    description=description,
                    what_it_is="A profession logic node used to constrain reasoning style and attention.",
                    how_it_works="Profession nodes influence which examples matter and how answers are structured.",
                    how_to_recognize="Use profession nodes when work logic shapes dialogue and decision-making.",
                    tags_json=json.dumps(["system", "profession"], ensure_ascii=False),
                    logic_weight=0.82,
                    risk_weight=0.66,
                    relevance_weight=0.72,
                )
            )
            created.append(profession_id)
    for agent_id, spec in _AGENT_BLUEPRINTS.items():
        if not graph_store.get_node(agent_id):
            graph_store.upsert_node(
                Node(
                    id=agent_id,
                    type="AGENT",
                    name=str(spec["name"]),
                    description=str(spec["description"]),
                    what_it_is="A specialized reasoning agent that shapes graph-grounded analysis before dialogue generation.",
                    how_it_works="The agent reads the graph, prioritizes evidence, and contributes a profession-consistent reasoning step.",
                    how_to_recognize="Use agent nodes when you need stable specialized reasoning behavior rather than generic assistant tone.",
                    tags_json=json.dumps(["system", "agent"], ensure_ascii=False),
                    logic_weight=0.86,
                    emotion_weight=0.42,
                    risk_weight=0.72,
                    relevance_weight=0.82,
                )
            )
            created.append(agent_id)
        profession_id = str(spec.get("profession") or "")
        if profession_id:
            graph_store.replace_edge(
                Edge(
                    src_id=agent_id,
                    dst_id=profession_id,
                    type="RESPONDS_WITH",
                    weight=0.92,
                    confidence=0.88,
                    metadata_json=json.dumps({"basis": "system_agent_profession"}, ensure_ascii=False, sort_keys=True),
                )
            )
    if created:
        graph_store.conn.commit()
    return created


def generate_behavioral_dialogue(
    graph_payload: dict[str, Any],
    *,
    query: str,
    recent_history: list[str] | None = None,
    person_id: str | None = None,
    llm_role: str = "general",
    user_id: str | None = None,
    style_profile: dict[str, Any] | None = None,
    fast_only: bool = False,
) -> dict[str, Any]:
    nodes = {str(node.get("id") or ""): node for node in list(graph_payload.get("nodes") or [])}
    person_node = nodes.get(str(person_id)) if person_id else None
    system_core = _serialize_core_system(graph_payload)
    ram_graph = build_ram_graph(graph_payload, query=query, recent_history=recent_history, person_id=person_id)
    personality = _build_personality_profile(graph_payload, person_node)
    personality_core = person_node_to_personality(person_node or {}) if person_node else None
    agent_plan = build_agent_plan(graph_payload, query, person_id=person_id)
    context_text = _context_text(ram_graph, personality=personality, agent_plan=agent_plan)
    style_prompt = build_style_prompt_block(style_profile)
    style_applied = bool(style_profile and list(style_profile.get("style_embedding") or []))
    speech_dna = SpeechDNA.from_style_profile(style_profile)
    ram_core = _build_ram_core_object(
        query=query,
        recent_history=recent_history,
        micro=dict(ram_graph.get("micro_signals") or {}),
        nodes=list(ram_graph.get("nodes") or []),
        edges=list(ram_graph.get("edges") or []),
        ranked_context=list(ram_graph.get("ranked_context") or []),
    )
    scenario = ScenarioEngine().pick(query, profession=str(personality.get("profession") or ""))
    role_registry = default_agent_roles()
    selected_roles = tuple(role_registry[role_id] for role_id in scenario.recommended_agents if role_id in role_registry)
    dialogue_contract = DialogueEngine().build_contract(
        query=query,
        personality_core=personality_core,
        speech_dna=speech_dna,
        ram_graph=ram_core,
        agent_roles=selected_roles,
        scenario=scenario.as_dict(),
    )

    fast_llm = build_fast_llm()
    reasoning_llm = None if fast_only else build_reasoning_llm(llm_role)
    reply = ""
    response_mode = "behavioral_fallback"
    if fast_llm is not None:
        prompt = json.dumps(
            {
                "task": "behavioral_dialogue_simulation_fast",
                "instructions": [
                    "Use only the RAM graph and personality data below.",
                    "Answer directly and practically.",
                    "Do not answer like a polite assistant.",
                    "If evidence is weak, say what is missing inside the character's logic.",
                    *( [style_prompt] if style_prompt else [] ),
                ],
                "query": query,
                "personality": personality,
                "agent_plan": agent_plan,
                "ram_context": context_text,
                "dialogue_contract": dialogue_contract.as_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            reply = _norm_text(fast_llm(prompt))
            if reply:
                response_mode = "behavioral_fast_llm"
        except Exception:
            reply = ""
    if not reply and not fast_only and reasoning_llm is not None:
        prompt = json.dumps(
            {
                "task": "behavioral_dialogue_simulation",
                "instructions": [
                    "Use only the RAM graph and personality data below.",
                    "Do not answer like a polite assistant.",
                    "Do not invent biography, facts, or motives outside the graph.",
                    "If evidence is weak, say what is missing inside the character's logic.",
                    "Return JSON with keys: reply, reasoning_style, missing_context.",
                    *( [style_prompt] if style_prompt else [] ),
                ],
                "query": query,
                "personality": personality,
                "agent_plan": agent_plan,
                "ram_context": context_text,
                "dialogue_contract": dialogue_contract.as_dict(),
                "style_profile": {
                    "style_embedding": list(style_profile.get("style_embedding") or []),
                    "speech_dna": dict(style_profile.get("speech_dna") or {}),
                    "style_examples": list(style_profile.get("style_examples") or []),
                } if style_profile else {},
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            raw = reasoning_llm(prompt)
            parsed = json.loads(raw) if isinstance(raw, str) and raw.strip().startswith("{") else {}
            if isinstance(parsed, dict):
                reply = _norm_text(parsed.get("reply"))
            if not reply:
                reply = _norm_text(raw)
            if reply:
                response_mode = "behavioral_reasoning_llm"
        except Exception:
            reply = ""
    if not reply:
        reply = _fallback_grounded_reply(query, ram_graph, personality=personality, agent_plan=agent_plan)
    if style_applied:
        reply = apply_style_to_fallback(reply, style_profile)

    log_path = ""
    if person_id:
        log_path = _write_personality_log(str(person_id), query=query, ram_graph=ram_graph, response=reply, evaluation="graph_grounded")

    return {
        "assistant_reply": reply,
        "assistant_reply_en": reply,
        "response_mode": response_mode,
        "ram_graph": ram_graph,
        "context_nodes": ram_graph.get("ranked_context", [])[:10],
        "agent_plan": agent_plan,
        "personality": {
            "person_id": person_id or "",
            "enabled": bool(person_node),
            "profile": personality,
            "log_path": log_path,
        },
        "runtime": {
            "provider": "behavioral_graph_rag",
            "response_mode": response_mode,
        },
        "core": {
            "system": system_core,
            "ram": dict(ram_graph.get("ram_core") or {}),
            "speech_dna": speech_dna.as_dict(),
            "scenario": scenario.as_dict(),
            "agent_roles": [role.as_dict() for role in selected_roles],
            "dialogue_contract": dialogue_contract.as_dict(),
        },
        "style": {
            "applied": style_applied,
            "user_id": str(user_id or ""),
            "sample_count": int(style_profile.get("sample_count") or 0) if style_profile else 0,
            "last_updated": str(style_profile.get("last_updated") or "") if style_profile else "",
            "style_examples": list(style_profile.get("style_examples") or [])[:3] if style_profile else [],
            "style_embedding": list(style_profile.get("style_embedding") or []) if style_profile else [],
        },
    }


def generate_grounded_graph_response(
    graph_payload: dict[str, Any],
    *,
    query: str,
    recent_history: list[str] | None = None,
    persona_id: str | None = None,
    llm_role: str = "general",
) -> dict[str, Any]:
    return generate_behavioral_dialogue(
        graph_payload,
        query=query,
        recent_history=recent_history,
        person_id=persona_id,
        llm_role=llm_role,
    )
