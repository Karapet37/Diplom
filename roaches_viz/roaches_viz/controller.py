from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ControllerDecision:
    ok: bool
    reason: str
    details: dict[str, Any]


class GraphMutationController:
    def __init__(
        self,
        *,
        max_nodes_per_request: int = 12,
        max_edges_per_request: int = 24,
        allow_deletes: bool = False,
    ) -> None:
        self.max_nodes_per_request = max_nodes_per_request
        self.max_edges_per_request = max_edges_per_request
        self.allow_deletes = allow_deletes

    def validate_ai_change(self, *, nodes_to_add: int, edges_to_add: int, deletes_requested: int = 0) -> ControllerDecision:
        if deletes_requested and not self.allow_deletes:
            return ControllerDecision(
                ok=False,
                reason="deletes_blocked",
                details={"deletes_requested": deletes_requested},
            )
        if nodes_to_add > self.max_nodes_per_request:
            return ControllerDecision(
                ok=False,
                reason="too_many_nodes",
                details={"nodes_to_add": nodes_to_add, "max_nodes_per_request": self.max_nodes_per_request},
            )
        if edges_to_add > self.max_edges_per_request:
            return ControllerDecision(
                ok=False,
                reason="too_many_edges",
                details={"edges_to_add": edges_to_add, "max_edges_per_request": self.max_edges_per_request},
            )
        return ControllerDecision(
            ok=True,
            reason="accepted",
            details={"nodes_to_add": nodes_to_add, "edges_to_add": edges_to_add, "deletes_requested": deletes_requested},
        )

    def validate_node_patch(self, patch: dict[str, Any]) -> ControllerDecision:
        forbidden = {"id", "created_at", "updated_at"}
        touched = sorted(set(patch) - {"node_id"})
        if any(key in forbidden for key in touched):
            return ControllerDecision(
                ok=False,
                reason="forbidden_node_fields",
                details={"touched": touched, "forbidden": sorted(forbidden)},
            )
        return ControllerDecision(ok=True, reason="accepted", details={"touched": touched})
