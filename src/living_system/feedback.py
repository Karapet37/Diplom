"""Feedback layer for human-in-the-loop reinforcement and auditing."""

from __future__ import annotations

from typing import Any

from src.living_system.knowledge_sql import KnowledgeSQLStore


class FeedbackService:
    """Captures user/system feedback and persists explainable traces."""

    def __init__(self, store: KnowledgeSQLStore):
        self.store = store

    def capture(self, payload: dict[str, Any]) -> dict[str, Any]:
        user_id = str(payload.get("user_id", "")).strip()
        event_type = str(payload.get("event_type", "feedback") or "feedback")
        score = max(-1.0, min(1.0, float(payload.get("score", 0.0) or 0.0)))
        details = dict(payload.get("details", {}) or {})
        message = str(payload.get("message", "") or "")

        log_id = self.store.append_log(
            level="info",
            component="feedback",
            message=f"{event_type}:{message}",
            details={"score": score, **details},
            user_id=user_id,
        )

        return {
            "ok": True,
            "log_id": log_id,
            "event_type": event_type,
            "score": score,
            "message": message,
            "details": details,
        }
