from __future__ import annotations

from typing import Any

from core import AgentCore, ContextCore, GraphEdge, GraphMemory, GraphNode, ImportanceVector, NodeBranches, NodeContext, NodeCore, PersonalityCore, SpeechStyleCore, SystemCore


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _to_tuple(values: Any) -> tuple[str, ...]:
    if isinstance(values, list):
        return tuple(str(item).strip() for item in values if str(item).strip())
    if isinstance(values, tuple):
        return tuple(str(item).strip() for item in values if str(item).strip())
    return ()


def importance_vector_from_node(node: dict[str, Any]) -> ImportanceVector:
    iv = dict(node.get("importance_vector") or {})
    return ImportanceVector(
        logic_weight=float(iv.get("logic_weight") or node.get("logic_weight") or 0.5),
        emotion_weight=float(iv.get("emotion_weight") or node.get("emotion_weight") or 0.5),
        risk_weight=float(iv.get("risk_weight") or node.get("risk_weight") or 0.5),
        relevance_weight=float(iv.get("relevance_weight") or node.get("relevance_weight") or 0.5),
    ).clamped()


def payload_node_to_core(node: dict[str, Any]) -> GraphNode:
    node_id = _norm(node.get("id") or node.get("node_id"))
    tags = _to_tuple(node.get("tags") or node.get("tags_json"))
    examples = _to_tuple(node.get("examples") or node.get("examples_json"))
    patterns = _to_tuple(node.get("behavior_patterns") or node.get("behavior_patterns_json"))
    signals = _to_tuple(node.get("emotion_signals") or node.get("emotion_signals_json"))
    relations = _to_tuple(node.get("pattern_nodes") or node.get("relations"))
    metadata = {
        "plain_explanation": _norm(node.get("plain_explanation")),
        "what_it_is": _norm(node.get("what_it_is")),
        "how_it_works": _norm(node.get("how_it_works")),
        "how_to_recognize": _norm(node.get("how_to_recognize")),
        "background": _norm(node.get("background")),
        "profession": _norm(node.get("profession")),
    }
    return GraphNode(
        node_id=node_id,
        node_core=NodeCore(
            node_id=node_id,
            node_type=_norm(node.get("type") or "PATTERN").upper(),
            name=_norm(node.get("name") or node.get("label") or node_id),
            description=_norm(node.get("description") or node.get("short_gloss") or node.get("plain_explanation") or node_id),
            importance_vector=importance_vector_from_node(node),
            metadata=metadata,
        ),
        node_branches=NodeBranches(
            patterns=patterns,
            examples=examples,
            signals=signals,
            relations=relations,
        ),
        node_context=NodeContext(
            domains=_to_tuple(node.get("domains") or node.get("domain_ids")),
            tags=tags,
            recent_references=_to_tuple(node.get("recent_references")),
            note=_norm(node.get("note")),
        ),
    )


def payload_edge_to_core(edge: dict[str, Any]) -> GraphEdge:
    metadata = dict(edge.get("metadata") or {})
    if "confidence" not in metadata and edge.get("confidence") is not None:
        metadata["confidence"] = float(edge.get("confidence") or 0.0)
    return GraphEdge(
        src_id=_norm(edge.get("src_id")),
        dst_id=_norm(edge.get("dst_id")),
        edge_type=_norm(edge.get("type") or edge.get("edge_type")),
        weight=float(edge.get("weight") or 0.0),
        metadata=metadata,
    )


def person_node_to_personality(node: dict[str, Any]) -> PersonalityCore | None:
    if _norm(node.get("type")).upper() != "PERSON":
        return None
    speech_profile = dict(node.get("speech_profile") or node.get("speech_style") or {})
    return PersonalityCore(
        temperament=_norm(node.get("temperament")) or "grounded",
        values=_to_tuple(node.get("values") or node.get("values_json")),
        speech_style=SpeechStyleCore(
            formality=float(speech_profile.get("formality") or node.get("formality") or 0.5),
            slang_level=float(speech_profile.get("slang_level") or node.get("slang_level") or 0.3),
            directness=float(speech_profile.get("directness") or node.get("directness") or 0.5),
            profanity_tolerance=float(speech_profile.get("profanity_tolerance") or node.get("profanity_tolerance") or 0.1),
        ).clamp(),
        reasoning_style=_norm(node.get("reasoning_style")) or "grounded",
        risk_tolerance=float(node.get("tolerance_threshold") or node.get("risk_tolerance") or 0.5),
        aggression_level=float(node.get("aggression_level") or 0.2),
        humor_level=float(node.get("humor_level") or 0.2),
    )


def style_profile_to_speech_dna_payload(style_profile: dict[str, Any] | None) -> dict[str, Any]:
    from core import SpeechDNA

    return SpeechDNA.from_style_profile(style_profile).as_dict()


def agent_node_to_core(node: dict[str, Any]) -> AgentCore | None:
    if _norm(node.get("type")).upper() != "AGENT":
        return None
    agent_id = _norm(node.get("id") or node.get("node_id"))
    return AgentCore(
        agent_id=agent_id,
        name=_norm(node.get("name") or node.get("label") or agent_id),
        purpose=_norm(node.get("description") or node.get("plain_explanation") or "grounded reasoning"),
    )


def graph_payload_to_memory(graph_payload: dict[str, Any]) -> GraphMemory:
    nodes = {node.node_id: node for node in (payload_node_to_core(node) for node in list(graph_payload.get("nodes") or []))}
    edges = tuple(payload_edge_to_core(edge) for edge in list(graph_payload.get("edges") or []))
    return GraphMemory(nodes=nodes, edges=edges)


def graph_payload_to_system_core(graph_payload: dict[str, Any]) -> SystemCore:
    nodes = list(graph_payload.get("nodes") or [])
    personalities = {
        _norm(node.get("id") or node.get("node_id")): personality
        for node in nodes
        for personality in [person_node_to_personality(node)]
        if personality is not None
    }
    agents = {
        _norm(node.get("id") or node.get("node_id")): agent
        for node in nodes
        for agent in [agent_node_to_core(node)]
        if agent is not None
    }
    return SystemCore(
        graph_memory=graph_payload_to_memory(graph_payload),
        personalities=personalities,
        agents=agents,
        context_builder=ContextCore(),
    )
