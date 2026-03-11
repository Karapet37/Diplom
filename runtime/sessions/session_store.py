from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionRecord:
    session_id: str
    title: str
    created_at: str
    updated_at: str
    last_query: str
    tools: dict[str, Any]
    messages: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_query": self.last_query,
            "tools": dict(self.tools),
            "messages": list(self.messages),
        }


class SessionStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def create(self, *, title: str = "", tools: dict[str, Any] | None = None) -> dict[str, Any]:
        session_id = f"session-{uuid4().hex[:12]}"
        now = _utc_now()
        record = SessionRecord(
            session_id=session_id,
            title=(title or "New session").strip()[:120] or "New session",
            created_at=now,
            updated_at=now,
            last_query="",
            tools=dict(tools or {}),
            messages=[],
        )
        self.save(record.as_dict())
        return record.as_dict()

    def list(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.json")):
            payload = self.load(path.stem)
            if not payload:
                continue
            items.append(
                {
                    "session_id": payload["session_id"],
                    "title": payload.get("title") or "Untitled session",
                    "created_at": payload.get("created_at") or "",
                    "updated_at": payload.get("updated_at") or "",
                    "last_query": payload.get("last_query") or "",
                    "message_count": len(payload.get("messages") or []),
                }
            )
        items.sort(key=lambda item: (item.get("updated_at") or "", item.get("created_at") or ""), reverse=True)
        return items

    def load(self, session_id: str) -> dict[str, Any] | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        payload.setdefault("session_id", session_id)
        payload.setdefault("title", "Untitled session")
        payload.setdefault("created_at", _utc_now())
        payload.setdefault("updated_at", payload["created_at"])
        payload.setdefault("last_query", "")
        payload.setdefault("tools", {})
        payload.setdefault("messages", [])
        return payload

    def save(self, session: dict[str, Any]) -> dict[str, Any]:
        session_id = str(session.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        previous = self.load(session_id)
        now = _utc_now()
        payload = {
            "session_id": session_id,
            "title": str(session.get("title") or (previous or {}).get("title") or "Untitled session").strip()[:120] or "Untitled session",
            "created_at": str(session.get("created_at") or (previous or {}).get("created_at") or now),
            "updated_at": now,
            "last_query": str(session.get("last_query") or ""),
            "tools": dict(session.get("tools") or (previous or {}).get("tools") or {}),
            "messages": list(session.get("messages") or []),
        }
        self._path(session_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
