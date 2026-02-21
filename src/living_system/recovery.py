"""Recovery layer: snapshot, rollback, safe mode and operator override."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.living_system.knowledge_sql import KnowledgeSQLStore
from src.living_system.models import RecoveryAction


@dataclass
class RecoveryState:
    safe_mode: bool = False
    reason: str = ""
    human_override: bool = False


class RecoveryService:
    """Failure-tolerant recovery orchestration over SQL snapshots."""

    def __init__(self, store: KnowledgeSQLStore):
        self.store = store
        self.state = RecoveryState()

    def create_snapshot(self, *, reason: str, user_id: str = "") -> int:
        current = self.store.graph_state(user_id=user_id)
        latest = self.store.latest_snapshot_id(user_id=user_id)
        snapshot_id = self.store.save_snapshot(
            "graph_state",
            current,
            user_id=user_id,
            parent_snapshot_id=latest,
        )
        self.store.record_recovery_action(
            action="snapshot_create",
            status="ok",
            details={"reason": reason, "snapshot_id": snapshot_id, "user_id": user_id},
        )
        return snapshot_id

    def rollback(self, snapshot_id: int) -> RecoveryAction:
        restored = self.store.restore_graph_from_snapshot(int(snapshot_id))
        if restored is None:
            action = RecoveryAction(
                action="rollback",
                status="failed",
                details={"snapshot_id": int(snapshot_id), "reason": "snapshot_not_found"},
            )
            self.store.record_recovery_action(
                action=action.action,
                status=action.status,
                details=action.details,
            )
            return action

        action = RecoveryAction(
            action="rollback",
            status="ok",
            details={
                "snapshot_id": int(snapshot_id),
                "nodes": len(list(restored.get("nodes", []) or [])),
                "edges": len(list(restored.get("edges", []) or [])),
            },
        )
        self.store.record_recovery_action(
            action=action.action,
            status=action.status,
            details=action.details,
        )
        return action

    def set_safe_mode(self, enabled: bool, *, reason: str = "") -> RecoveryAction:
        self.state.safe_mode = bool(enabled)
        self.state.reason = str(reason or "")
        action = RecoveryAction(
            action="safe_mode",
            status="enabled" if enabled else "disabled",
            details={"reason": self.state.reason},
        )
        self.store.record_recovery_action(
            action=action.action,
            status=action.status,
            details=action.details,
        )
        return action

    def set_human_override(self, enabled: bool, *, reason: str = "") -> RecoveryAction:
        self.state.human_override = bool(enabled)
        action = RecoveryAction(
            action="human_override",
            status="enabled" if enabled else "disabled",
            details={"reason": str(reason or "")},
        )
        self.store.record_recovery_action(
            action=action.action,
            status=action.status,
            details=action.details,
        )
        return action

    def auto_recover(self, *, diagnostics: dict[str, Any]) -> RecoveryAction:
        if diagnostics.get("ok", True):
            return RecoveryAction(action="auto_recover", status="noop", details={"reason": "health_ok"})

        if diagnostics.get("message") == "confidence_drop":
            latest = self.store.latest_snapshot_id()
            if latest is not None:
                return self.rollback(latest)
        return self.set_safe_mode(True, reason="automatic safeguard")
