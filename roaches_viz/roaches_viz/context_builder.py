from __future__ import annotations

from typing import Any

from .graph_store import GraphStore, load_json, normalize_personality_name, personality_graph_path, personality_index_path, personality_profile_path


def _truncate_tokens_equivalent(text: str, max_tokens_equivalent: int) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    max_chars = max_tokens_equivalent * 4
    return raw[:max_chars].strip() if len(raw) > max_chars else raw


def _load_personality_index() -> list[str]:
    payload = load_json(personality_index_path(), {"personalities": []})
    names = list(payload.get("personalities") or []) if isinstance(payload, dict) else []
    return [normalize_personality_name(name) for name in names if str(name).strip()]


def list_personalities() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in _load_personality_index():
        rows.append({"name": name, "profile": load_personality(name) or {"name": name, "traits": [], "patterns": [], "examples": []}})
    return rows


def load_personality(name: str) -> dict[str, Any] | None:
    clean = normalize_personality_name(name)
    payload = load_json(personality_profile_path(clean), None)
    return dict(payload) if isinstance(payload, dict) else None


def load_personality_graph(name: str) -> dict[str, Any]:
    clean = normalize_personality_name(name)
    payload = load_json(personality_graph_path(clean), {"nodes": [], "edges": []})
    return payload if isinstance(payload, dict) else {"nodes": [], "edges": []}


def personality_exists(name: str) -> bool:
    return personality_profile_path(name).exists()


def infer_personality_name(message: str, selected_name: str = "", current_entity: str = "") -> str:
    if str(selected_name or "").strip():
        return normalize_personality_name(selected_name)
    lower = str(message or "").lower()
    candidates = {
        "сшелдон": "sheldon_cooper",
        "шелдон": "sheldon_cooper",
        "sheldon": "sheldon_cooper",
        "драку": "dracula",
        "dracula": "dracula",
        "ураган": "hurricane",
        "hurricane": "hurricane",
    }
    for needle, value in candidates.items():
        if needle in lower:
            return value
    if str(current_entity or "").strip():
        return normalize_personality_name(current_entity)
    return ""


def current_entity_hint(message: str, session_context: str) -> str:
    lower = f"{message}\n{session_context}".lower()
    for token, name in (
        ("драку", "dracula"),
        ("dracula", "dracula"),
        ("шелдон", "sheldon_cooper"),
        ("sheldon", "sheldon_cooper"),
        ("леонард", "leonard"),
        ("leonard", "leonard"),
        ("ураган", "hurricane"),
        ("hurricane", "hurricane"),
    ):
        if token in lower:
            return name
    return ""


def build_personality_prompt(name: str) -> str:
    profile = load_personality(name)
    if not profile:
        return ""
    graph = load_personality_graph(name)
    traits = ", ".join(str(item).strip() for item in list(profile.get("traits") or []) if str(item).strip())
    patterns = ", ".join(str(item).strip() for item in list(profile.get("patterns") or []) if str(item).strip())
    examples = [str(item).strip() for item in list(profile.get("examples") or []) if str(item).strip()][:2]
    graph_facts = []
    for edge in list(graph.get("edges") or [])[:6]:
        src = str(edge.get("from") or "")
        dst = str(edge.get("to") or "")
        etype = str(edge.get("type") or "")
        if src and dst and etype:
            graph_facts.append(f"{src} {etype} {dst}")
    blocks = [
        f"You are {profile.get('name') or name}.",
        "Answer in first person from this personality perspective.",
    ]
    if traits:
        blocks.append(f"Traits: {traits}.")
    if patterns:
        blocks.append(f"Patterns: {patterns}.")
    if examples:
        blocks.append(f"Example cues: {' | '.join(examples)}.")
    if graph_facts:
        blocks.append("Personality graph facts:")
        blocks.extend(graph_facts)
    return "\n".join(blocks).strip()


def build_chat_context(
    *,
    message: str,
    recent_dialogue: str,
    selected_personality: str = "",
    current_entity: str = "",
    explicit_context: str = "",
    store: GraphStore | None = None,
) -> dict[str, Any]:
    graph_store = store or GraphStore()
    resolved_personality = infer_personality_name(message, selected_name=selected_personality, current_entity=current_entity)
    graph_query = " ".join(part for part in [resolved_personality, current_entity, explicit_context, message] if str(part or "").strip())
    subgraph = graph_store.subgraph(graph_query, limit=8)
    nodes = list(subgraph.get("nodes") or [])[:8]
    edges = list(subgraph.get("edges") or [])[:12]

    graph_lines: list[str] = []
    for node in nodes:
        name = str(node.get("name") or node.get("id") or "").strip()
        description = str(node.get("description") or node.get("short_gloss") or "").strip()
        if name and description:
            graph_lines.append(f"- {name} [{node.get('type')}]: {description}")
        elif name:
            graph_lines.append(f"- {name} [{node.get('type')}]")
    for edge in edges:
        graph_lines.append(f"- relation: {edge.get('from')} {edge.get('type')} {edge.get('to')} (weight={edge.get('weight')})")

    graph_context = _truncate_tokens_equivalent("\n".join(graph_lines).strip(), 1800)
    personality_prompt = build_personality_prompt(resolved_personality) if resolved_personality and personality_exists(resolved_personality) else ""
    session_context = _truncate_tokens_equivalent(recent_dialogue, 1200)
    return {
        "personality_name": resolved_personality,
        "personality_prompt": personality_prompt,
        "graph_context": graph_context,
        "session_context": session_context,
        "current_entity": current_entity or resolved_personality,
        "nodes": nodes,
        "edges": edges,
    }


def answerable_node_view(node_id: str, *, store: GraphStore | None = None) -> dict[str, Any] | None:
    graph_store = store or GraphStore()
    return graph_store.answerable_node_view(node_id)
