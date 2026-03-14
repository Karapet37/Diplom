from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any

from .graph_store import (
    GraphStore,
    load_json,
    normalize_personality_name,
    personality_graph_path,
    personality_index_path,
    personality_profile_path,
    personality_proposal_path,
    personality_proposals_dir,
    write_json,
)
from .history_store import list_sessions, parse_session, session_files_dir
from .llm import generate_graph_proposals, generate_personality_profile_proposal

SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".csv"}


def _load_personality_index() -> list[str]:
    payload = load_json(personality_index_path(), {"personalities": []})
    names = list(payload.get("personalities") or []) if isinstance(payload, dict) else []
    return [normalize_personality_name(name) for name in names if str(name).strip()]


def _save_personality_index(names: list[str]) -> None:
    unique: list[str] = []
    seen: set[str] = set()
    for name in names:
        clean = normalize_personality_name(name)
        if clean in seen:
            continue
        seen.add(clean)
        unique.append(clean)
    write_json(personality_index_path(), {"personalities": unique})


def _session_text(parsed: dict[str, Any]) -> str:
    lines: list[str] = []
    for item in list(parsed.get("messages") or []):
        role = str(item.get("role") or "assistant")
        message = str(item.get("message") or "").strip()
        if message:
            lines.append(f"{role}: {message}")
    return "\n".join(lines).strip()


def _chunk_text(text: str, *, max_chars: int = 7000, overlap: int = 500) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    paragraphs = [part.strip() for part in raw.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:].strip()
            current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
        else:
            while len(paragraph) > max_chars:
                chunks.append(paragraph[:max_chars].strip())
                paragraph = paragraph[max_chars - overlap :].strip()
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def validate_graph_proposals(proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for item in proposals:
        if not isinstance(item, dict):
            continue
        entity = str(item.get("entity") or "").strip()
        entity_type = str(item.get("type") or "").strip().upper()
        if not entity or not entity_type:
            continue
        relations = list(item.get("relations") or [])
        if any(not isinstance(rel, dict) or not str(rel.get("type") or "").strip() or not str(rel.get("target") or "").strip() for rel in relations):
            continue
        valid.append(item)
    return valid


def request_missing_personality(name: str, reason: str, session_id: str, excerpt: str) -> dict[str, Any]:
    clean = normalize_personality_name(name)
    payload = {
        "name": clean,
        "reason": str(reason or "User selected or mentioned a personality missing from graph and personality files.").strip(),
        "session_id": str(session_id or "").strip(),
        "excerpt": str(excerpt or "").strip(),
    }
    write_json(personality_proposal_path(clean), payload)
    return payload


def list_personality_proposals() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(personality_proposals_dir().glob("*.json")):
        payload = load_json(path, None)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _extract_examples(parsed: dict[str, Any]) -> list[str]:
    examples: list[str] = []
    for item in list(parsed.get("messages") or []):
        role = str(item.get("role") or "")
        message = str(item.get("message") or "").strip()
        if role in {"user", "assistant"} and message:
            examples.append(message)
    return list(dict.fromkeys(examples))


def _detect_signals(example: str) -> list[str]:
    text = str(example or "").lower()
    signals: list[str] = []
    if any(token in text for token in ("должен", "обязан", "if you loved me", "ты должен")):
        signals.append("pressure_language")
    if any(token in text for token in ("неправ", "actually", "вообще-то", "ошибаешься")):
        signals.append("corrective_tone")
    if any(token in text for token in ("кров", "вампир", "vampire", "blood")):
        signals.append("predatory_imagery")
    if any(token in text for token in ("логик", "physics", "физик", "экспериментальный")):
        signals.append("hyperlogical_frame")
    return list(dict.fromkeys(signals))


def _detect_patterns(example: str, signals: list[str]) -> list[str]:
    patterns: list[str] = []
    if "corrective_tone" in signals:
        patterns.append("correct_people")
    if "pressure_language" in signals:
        patterns.append("dominance_warning")
    if "predatory_imagery" in signals:
        patterns.append("feeds_on_blood")
    if "hyperlogical_frame" in signals:
        patterns.append("quote_science")
    return list(dict.fromkeys(patterns))


def _detect_traits(example: str, signals: list[str], patterns: list[str]) -> list[str]:
    traits: list[str] = []
    text = str(example or "").lower()
    if "hyperlogical_frame" in signals or "quote_science" in patterns:
        traits.append("logical")
    if "corrective_tone" in signals:
        traits.append("literal")
    if any(token in text for token in ("вампир", "vampire", "бессмерт", "immortal")):
        traits.extend(["vampire", "immortal"])
    if any(token in text for token in ("аристократ", "noble", "aristocrat")):
        traits.append("aristocratic")
    if "predatory_imagery" in signals:
        traits.append("predatory")
    return list(dict.fromkeys(traits))


def _trait_weight(example_count: int) -> float:
    return round(max(0.1, math.log(max(2, int(example_count) + 1))), 4)


def _build_personality_graph(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    examples = [str(item).strip() for item in list(payload.get("examples") or []) if str(item).strip()]
    traits = [str(item).strip() for item in list(payload.get("traits") or []) if str(item).strip()]
    patterns = [str(item).strip() for item in list(payload.get("patterns") or []) if str(item).strip()]
    nodes: list[dict[str, Any]] = [{"id": f"person:{name}", "type": "PERSON", "name": name, "weight": 1.0}]
    edges: list[dict[str, Any]] = []
    for example in examples:
        example_id = f"example:{name}:{abs(hash(example))}"
        nodes.append({"id": example_id, "type": "EXAMPLE", "name": example, "weight": 1.0})
        edges.append({"from": f"person:{name}", "to": example_id, "type": "HAS_EXAMPLE", "weight": 1.0})
        signals = _detect_signals(example)
        for signal in signals:
            signal_id = f"signal:{name}:{signal}"
            nodes.append({"id": signal_id, "type": "SIGNAL", "name": signal, "weight": 1.0})
            edges.append({"from": example_id, "to": signal_id, "type": "HAS_SIGNAL", "weight": 1.0})
    for pattern in patterns:
        pattern_id = f"pattern:{name}:{pattern}"
        nodes.append({"id": pattern_id, "type": "PATTERN", "name": pattern, "weight": 1.0})
        edges.append({"from": f"person:{name}", "to": pattern_id, "type": "USES_PATTERN", "weight": 1.0})
    for trait in traits:
        trait_id = f"trait:{name}:{trait}"
        trait_example_count = max(1, sum(1 for example in examples if trait in _detect_traits(example, _detect_signals(example), _detect_patterns(example, _detect_signals(example)))))
        nodes.append({"id": trait_id, "type": "TRAIT", "name": trait, "weight": _trait_weight(trait_example_count), "example_count": trait_example_count})
        edges.append({"from": f"person:{name}", "to": trait_id, "type": "HAS_TRAIT", "weight": _trait_weight(trait_example_count)})
    unique_nodes: dict[str, dict[str, Any]] = {}
    for node in nodes:
        unique_nodes[str(node.get("id") or "")] = node
    unique_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for edge in edges:
        key = (str(edge.get("from") or ""), str(edge.get("type") or ""), str(edge.get("to") or ""))
        if key in seen_edges or not all(key):
            continue
        seen_edges.add(key)
        unique_edges.append(edge)
    return {"nodes": list(unique_nodes.values()), "edges": unique_edges}


def _save_personality_profile(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    clean = normalize_personality_name(name)
    materialized = {
        "name": clean,
        "traits": [str(item).strip() for item in list(payload.get("traits") or []) if str(item).strip()],
        "patterns": [str(item).strip() for item in list(payload.get("patterns") or []) if str(item).strip()],
        "examples": [str(item).strip() for item in list(payload.get("examples") or []) if str(item).strip()],
    }
    write_json(personality_profile_path(clean), materialized)
    _save_personality_index(_load_personality_index() + [clean])
    return materialized


def _save_personality_graph(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    clean = normalize_personality_name(name)
    graph = {"nodes": list(payload.get("nodes") or []), "edges": list(payload.get("edges") or [])}
    write_json(personality_graph_path(clean), graph)
    _save_personality_index(_load_personality_index() + [clean])
    return graph


def process_personality_proposals() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for proposal in list_personality_proposals():
        name = str(proposal.get("name") or "").strip()
        if not name:
            continue
        materialized = generate_personality_profile_proposal(
            name=name,
            excerpt=str(proposal.get("excerpt") or ""),
            reason=str(proposal.get("reason") or ""),
        )
        if not materialized.get("traits") and not materialized.get("patterns") and not materialized.get("examples"):
            continue
        profile = _save_personality_profile(name, materialized)
        graph = _save_personality_graph(name, _build_personality_graph(name, materialized))
        proposal_path = personality_proposal_path(name)
        if proposal_path.exists():
            proposal_path.unlink()
        results.append({"name": name, "profile": profile, "graph": graph})
    return results


def update_personality_from_session(session_id: str, personality_name: str) -> dict[str, Any]:
    parsed = parse_session(session_id)
    if not parsed:
        return {"ok": False, "reason": "missing_session", "session_id": session_id}
    examples = _extract_examples(parsed)
    if not examples:
        return {"ok": False, "reason": "no_examples", "session_id": session_id}
    clean = normalize_personality_name(personality_name)
    existing_profile = load_json(personality_profile_path(clean), {"name": clean, "traits": [], "patterns": [], "examples": []})
    all_examples = list(dict.fromkeys(list(existing_profile.get("examples") or []) + examples))
    all_patterns = list(dict.fromkeys(list(existing_profile.get("patterns") or [])))
    all_traits = list(dict.fromkeys(list(existing_profile.get("traits") or [])))
    for example in examples:
        signals = _detect_signals(example)
        patterns = _detect_patterns(example, signals)
        traits = _detect_traits(example, signals, patterns)
        all_patterns = list(dict.fromkeys(all_patterns + patterns))
        all_traits = list(dict.fromkeys(all_traits + traits))
    payload = {"name": clean, "traits": all_traits, "patterns": all_patterns, "examples": all_examples}
    profile = _save_personality_profile(clean, payload)
    graph = _save_personality_graph(clean, _build_personality_graph(clean, payload))
    return {"ok": True, "profile": profile, "graph": graph, "example_count": len(all_examples)}


def _render_json_text(raw: bytes) -> str:
    try:
        payload = json.loads(raw.decode("utf-8"))
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        return raw.decode("utf-8", errors="ignore")


def _render_csv_text(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    lines: list[str] = []
    for row in reader:
        if not isinstance(row, dict):
            continue
        lines.append(", ".join(f"{key}={value}" for key, value in row.items()))
    return "\n".join(lines).strip() or text


def _file_to_text(path: Path) -> str:
    suffix = path.suffix.lower()
    raw = path.read_bytes()
    if suffix in {".txt", ".md"}:
        return raw.decode("utf-8", errors="ignore")
    if suffix == ".json":
        return _render_json_text(raw)
    if suffix == ".csv":
        return _render_csv_text(raw)
    return ""


def store_uploaded_file(session_id: str, filename: str, content: bytes) -> Path:
    safe_name = Path(filename or "upload.txt").name
    path = session_files_dir(session_id) / safe_name
    path.write_bytes(content)
    return path


def extract_session(session_id: str, *, store: GraphStore | None = None) -> dict[str, Any]:
    parsed = parse_session(session_id)
    if not parsed:
        return {"ok": False, "reason": "missing_session", "session_id": session_id}
    text = _session_text(parsed)
    proposals = validate_graph_proposals(generate_graph_proposals(text))
    if not proposals:
        return {"ok": True, "reason": "no_valid_proposals", "session_id": session_id, "proposal_count": 0}
    graph_store = store or GraphStore()
    result = graph_store.merge_proposals(proposals)
    result.update({"session_id": session_id, "proposal_count": len(proposals), "proposals": proposals})
    return result


def extract_file(path: Path, *, store: GraphStore | None = None) -> dict[str, Any]:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return {"ok": False, "reason": "unsupported_type", "path": str(path)}
    text = _file_to_text(path)
    if not text.strip():
        return {"ok": False, "reason": "empty_text", "path": str(path)}
    proposals: list[dict[str, Any]] = []
    for chunk in _chunk_text(text):
        proposals.extend(validate_graph_proposals(generate_graph_proposals(chunk)))
    if not proposals:
        return {"ok": True, "reason": "no_valid_proposals", "path": str(path), "proposal_count": 0}
    graph_store = store or GraphStore()
    result = graph_store.merge_proposals(proposals)
    result.update({"path": str(path), "proposal_count": len(proposals), "proposals": proposals})
    return result


def extract_session_files(session_id: str, *, store: GraphStore | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(session_files_dir(session_id).glob("*")):
        if path.is_file():
            results.append(extract_file(path, store=store))
    return results


def process_session_artifacts(session_id: str, *, personality_name: str = "", store: GraphStore | None = None) -> dict[str, Any]:
    graph_store = store or GraphStore()
    session_result = extract_session(session_id, store=graph_store)
    file_results = extract_session_files(session_id, store=graph_store)
    proposal_results = process_personality_proposals()
    personality_result = update_personality_from_session(session_id, personality_name) if str(personality_name or "").strip() and personality_profile_path(personality_name).exists() else None
    return {
        "ok": True,
        "session_result": session_result,
        "file_results": file_results,
        "proposal_results": proposal_results,
        "personality_result": personality_result,
    }


def rebuild(session_id: str = "", personality_name: str = "") -> dict[str, Any]:
    graph_store = GraphStore()
    sessions = [session_id] if str(session_id or "").strip() else [row["session_id"] for row in list_sessions()]
    results = [process_session_artifacts(item, personality_name=personality_name, store=graph_store) for item in sessions if str(item).strip()]
    return {"ok": True, "results": results}
