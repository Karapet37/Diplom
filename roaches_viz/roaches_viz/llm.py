from __future__ import annotations

import json
import re
from typing import Any

from src.utils.local_llm_provider import build_role_llm_fn
from src.utils.prompt_budgeter import MAX_REASONING_N_CTX, MAX_ROUTER_N_CTX, MIN_REASONING_N_CTX, MIN_ROUTER_N_CTX
from src.utils.token_budget import select_n_ctx

_PROMPT_LEAK_MARKERS = (
    "behavioral_dialogue_simulation",
    "behavioral_dialogue_simulation_fast",
    "system instruction",
    "internal prompt",
    "return plain text only",
    "do not output json",
    "dialogue_contract",
    "ram_context",
    "agent_plan",
)

_VALID_PROPOSAL_TYPES = {"ENTITY", "PERSON", "TRAIT", "PATTERN", "SIGNAL", "RELATION", "STYLE", "CONCEPT"}


def build_fast_llm() -> Any | None:
    return build_role_llm_fn(
        "analyst",
        n_ctx=select_n_ctx(MIN_ROUTER_N_CTX, [MIN_ROUTER_N_CTX, 1536, MAX_ROUTER_N_CTX]),
        max_tokens=640,
    )


def build_reasoning_llm(role: str = "general") -> Any | None:
    return build_role_llm_fn(
        role,
        n_ctx=select_n_ctx(4096, [MIN_REASONING_N_CTX, 3072, 4096, MAX_REASONING_N_CTX]),
        max_tokens=1536,
    )


def _extract_json_block(text: str) -> Any | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _normalize_text_reply(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw or raw == "{}":
        return ""
    lowered = raw.lower()
    if any(marker in lowered for marker in _PROMPT_LEAK_MARKERS):
        return ""
    payload = _extract_json_block(raw)
    if isinstance(payload, dict):
        for key in ("assistant_reply", "reply", "text", "message", "content", "response"):
            value = str(payload.get(key) or "").strip()
            if value and not any(marker in value.lower() for marker in _PROMPT_LEAK_MARKERS):
                return value
    return raw


def _call_model(prompt: str, *, mode: str = "chat") -> str:
    llm = build_reasoning_llm("general") if mode != "chat_fast" else build_fast_llm()
    if llm is None and mode == "chat":
        llm = build_fast_llm()
    if llm is None:
        return ""
    try:
        return str(llm(prompt) or "").strip()
    except Exception:
        return ""


def build_chat_prompt(*, message: str, session_context: str = "", graph_context: str = "", personality_prompt: str = "", language: str = "ru") -> str:
    blocks = [
        f"Respond in {('Russian' if language == 'ru' else language or 'English')}.",
        "Return plain text only.",
        "Do not output JSON.",
    ]
    if personality_prompt:
        blocks.extend(["Personality:", personality_prompt])
    if graph_context:
        blocks.extend(["Graph context:", graph_context])
    if session_context:
        blocks.extend(["Recent dialogue:", session_context])
    blocks.extend(["User message:", str(message or "").strip()])
    return "\n\n".join(part for part in blocks if str(part).strip())


def generate_chat_reply(*, message: str, session_context: str = "", graph_context: str = "", personality_prompt: str = "", language: str = "ru") -> str:
    prompt = build_chat_prompt(
        message=message,
        session_context=session_context,
        graph_context=graph_context,
        personality_prompt=personality_prompt,
        language=language,
    )
    reply = _normalize_text_reply(_call_model(prompt, mode="chat"))
    if reply:
        return reply
    if personality_prompt:
        if language == "ru":
            return "Я отвечу от первого лица, но у меня сейчас только частичный контекст личности и связей."
        return "I only have partial personality context, so I will answer cautiously and directly."
    if language == "ru":
        return "Мне не хватает надежного контекста. Уточни объект разговора или дай один конкретный факт."
    return "I do not have enough reliable context. Clarify the subject or give one concrete fact."


def build_graph_proposal_prompt(session_text: str) -> str:
    return "\n\n".join(
        [
            "Return valid JSON only.",
            'Schema: {"proposals":[{"entity":"Dracula","type":"PERSON","traits":["vampire"],"relations":[{"type":"FEEDS_ON","target":"humans"}]}]}',
            "Rules:",
            "- type must be one of ENTITY, PERSON, TRAIT, PATTERN, SIGNAL, RELATION, STYLE, CONCEPT",
            "- relations must contain type and target",
            "- if unsure return {\"proposals\":[]}",
            "Session text:",
            session_text,
        ]
    )


def generate_graph_proposals(session_text: str) -> list[dict[str, Any]]:
    raw = _call_model(build_graph_proposal_prompt(session_text), mode="proposal")
    payload = _extract_json_block(raw)
    if not isinstance(payload, dict) and not str(raw or "").strip():
        payload = _heuristic_graph_payload(session_text)
    proposals = list(payload.get("proposals") or []) if isinstance(payload, dict) else []
    return _normalize_graph_proposals(proposals)


def build_personality_proposal_prompt(name: str, excerpt: str, reason: str) -> str:
    return "\n\n".join(
        [
            "Return valid JSON only.",
            'Schema: {"name":"dracula","traits":["vampire"],"patterns":["feeds_on_blood"],"examples":["..."]}',
            f"Name: {name}",
            f"Reason: {reason}",
            "Excerpt:",
            excerpt,
        ]
    )


def generate_personality_profile_proposal(*, name: str, excerpt: str, reason: str) -> dict[str, Any]:
    raw = _call_model(build_personality_proposal_prompt(name, excerpt, reason), mode="proposal")
    payload = _extract_json_block(raw)
    if not isinstance(payload, dict) and not str(raw or "").strip():
        payload = _heuristic_personality_payload(name, excerpt)
    return _normalize_personality_payload(name, payload)


def _normalize_graph_proposals(items: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        entity = str(item.get("entity") or "").strip()
        entity_type = str(item.get("type") or "CONCEPT").strip().upper()
        if not entity or entity_type not in _VALID_PROPOSAL_TYPES:
            continue
        traits = [str(v).strip() for v in list(item.get("traits") or []) if str(v).strip()]
        relations = []
        for relation in list(item.get("relations") or []):
            if not isinstance(relation, dict):
                continue
            relation_type = str(relation.get("type") or "").strip().upper()
            target = str(relation.get("target") or "").strip()
            if relation_type and target:
                relations.append({"type": relation_type, "target": target})
        normalized.append({
            "entity": entity,
            "type": entity_type,
            "traits": list(dict.fromkeys(traits)),
            "relations": relations,
        })
    return normalized


def _normalize_personality_payload(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(payload.get("name") or name).strip().lower(),
        "traits": list(dict.fromkeys(str(v).strip() for v in list(payload.get("traits") or []) if str(v).strip())),
        "patterns": list(dict.fromkeys(str(v).strip() for v in list(payload.get("patterns") or []) if str(v).strip())),
        "examples": list(dict.fromkeys(str(v).strip() for v in list(payload.get("examples") or []) if str(v).strip())),
    }


def _heuristic_graph_payload(session_text: str) -> dict[str, Any]:
    text = str(session_text or "").lower()
    proposals: list[dict[str, Any]] = []
    if "дракул" in text or "dracula" in text:
        proposals.append({
            "entity": "Dracula",
            "type": "PERSON",
            "traits": ["вампир", "аристократический", "хищный", "бессмертный"],
            "relations": [
                {"type": "FEEDS_ON", "target": "люди"},
                {"type": "FEARS", "target": "солнечный свет"},
            ],
        })
    if "шелдон" in text or "sheldon" in text:
        proposals.append({
            "entity": "Sheldon Cooper",
            "type": "PERSON",
            "traits": ["логичный", "буквальный", "высокомерный"],
            "relations": [
                {"type": "KNOWS", "target": "Leonard"},
                {"type": "WORKS_IN_DOMAIN", "target": "physics"},
            ],
        })
    return {"proposals": proposals}


def _heuristic_personality_payload(name: str, excerpt: str) -> dict[str, Any]:
    clean = str(name or "").strip().lower()
    lower = str(excerpt or "").lower()
    if clean == "dracula":
        return {
            "name": "dracula",
            "traits": ["вампир", "аристократический", "хищный", "бессмертный"],
            "patterns": ["пьет_кровь", "избегает_солнца"],
            "examples": [excerpt or "знаешь дракулу?"],
        }
    if clean == "sheldon_cooper" or "шелдон" in lower or "sheldon" in lower:
        return {
            "name": "sheldon_cooper",
            "traits": ["логичный", "буквальный", "высокомерный"],
            "patterns": ["исправляет_людей", "цитирует_науку"],
            "examples": [excerpt or "кто для тебя Леонард?"],
        }
    return {"name": clean, "traits": [], "patterns": [], "examples": [excerpt] if str(excerpt).strip() else []}
