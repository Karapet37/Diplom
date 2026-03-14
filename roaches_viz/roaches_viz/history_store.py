from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .graph_store import memory_root


def sessions_dir() -> Path:
    path = memory_root() / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def files_root() -> Path:
    path = memory_root() / "files" / "uploaded_documents"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_session_id(session_id: str) -> str:
    token = str(session_id or "").strip()
    return token or f"session-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"


def session_text_path(session_id: str) -> Path:
    return sessions_dir() / f"{_safe_session_id(session_id)}.txt"


def session_files_dir(session_id: str) -> Path:
    path = files_root() / _safe_session_id(session_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_session(session_id: str = "", title: str = "") -> dict[str, Any]:
    clean_session_id = _safe_session_id(session_id)
    path = session_text_path(clean_session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if title.strip():
            path.write_text(f"# {title.strip()}\n", encoding="utf-8")
        else:
            path.touch()
    parsed = parse_session(clean_session_id)
    return parsed or {"session_id": clean_session_id, "title": title.strip() or "New session", "messages": [], "updated_at": _utc_now(), "path": str(path)}


def append_turn(session_id: str, user_message: str, assistant_message: str) -> Path:
    clean_session_id = _safe_session_id(session_id)
    path = session_text_path(clean_session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    blocks: list[str] = []
    if str(user_message or "").strip():
        blocks.append(f"[{_utc_now()}]\nuser: {str(user_message).strip()}")
    if str(assistant_message or "").strip():
        blocks.append(f"[{_utc_now()}]\nassistant: {str(assistant_message).strip()}")
    if not blocks:
        return path
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    payload = (existing.rstrip() + "\n\n" if existing.strip() else "") + "\n\n".join(blocks).strip() + "\n"
    path.write_text(payload, encoding="utf-8")
    return path


def parse_session(session_id: str) -> dict[str, Any] | None:
    path = session_text_path(session_id)
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8")
    messages: list[dict[str, str]] = []
    current_timestamp = ""
    title = ""
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            current_timestamp = stripped[1:-1].strip()
            continue
        if stripped.startswith("user:"):
            messages.append({"role": "user", "message": stripped.split(":", 1)[1].strip(), "timestamp": current_timestamp})
            continue
        if stripped.startswith("assistant:"):
            messages.append({"role": "assistant", "message": stripped.split(":", 1)[1].strip(), "timestamp": current_timestamp})
    resolved_title = title or next((item["message"][:60] for item in messages if item["role"] == "user" and item["message"].strip()), "New session")
    updated_at = current_timestamp or datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
    return {
        "session_id": _safe_session_id(session_id),
        "title": resolved_title,
        "messages": messages,
        "updated_at": updated_at,
        "path": str(path),
    }


def list_sessions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(sessions_dir().glob("*.txt"), key=lambda item: item.stat().st_mtime, reverse=True):
        parsed = parse_session(path.stem)
        if parsed is not None:
            rows.append(parsed)
    return rows


def recent_dialogue(session_id: str, *, max_messages: int = 6, max_tokens_equivalent: int = 1200) -> str:
    parsed = parse_session(session_id)
    if not parsed:
        return ""
    selected = list(parsed.get("messages") or [])[-max_messages:]
    lines = []
    for item in selected:
        role = str(item.get("role") or "assistant")
        message = str(item.get("message") or "").strip()
        if message:
            lines.append(f"{role}: {message}")
    context = "\n".join(lines).strip()
    max_chars = max_tokens_equivalent * 4
    return context[-max_chars:].strip() if len(context) > max_chars else context


def infer_current_entity(session_id: str) -> str:
    parsed = parse_session(session_id)
    if not parsed:
        return ""
    messages = list(parsed.get("messages") or [])
    for item in reversed(messages):
        text = str(item.get("message") or "").lower()
        for marker, value in (
            ("драку", "dracula"),
            ("dracula", "dracula"),
            ("шелдон", "sheldon_cooper"),
            ("sheldon", "sheldon_cooper"),
            ("леонард", "leonard"),
            ("leonard", "leonard"),
            ("ураган", "hurricane"),
            ("hurricane", "hurricane"),
        ):
            if marker in text:
                return value
    return ""
