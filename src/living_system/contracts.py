"""Layer contracts for replaceable long-living agent components."""

from __future__ import annotations

from typing import Any, Protocol

from src.living_system.models import FailureRecord, LayerHealth, ReasoningTrace, RecoveryAction, TelemetryEvent


class MonitoringLayer(Protocol):
    def emit(self, event: TelemetryEvent) -> None:
        ...


class KnowledgeLayer(Protocol):
    def initialize(self) -> None:
        ...

    def upsert_user_profile(self, payload: dict[str, Any]) -> None:
        ...

    def upsert_node(self, payload: dict[str, Any]) -> str:
        ...

    def upsert_edge(self, payload: dict[str, Any]) -> int:
        ...

    def store_reasoning_trace(self, trace: ReasoningTrace) -> int:
        ...

    def save_snapshot(self, snapshot_type: str, state: dict[str, Any], *, user_id: str = "") -> int:
        ...


class EmbeddingLayer(Protocol):
    def embed_text(self, text: str) -> list[float]:
        ...


class ReasoningLayer(Protocol):
    def process(self, text: str, *, user_id: str, language: str) -> dict[str, Any]:
        ...


class VisualizationLayer(Protocol):
    def graph_view(self, *, user_id: str = "") -> dict[str, Any]:
        ...


class FeedbackLayer(Protocol):
    def capture(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


class DiagnosticsLayer(Protocol):
    def health_check(self) -> list[LayerHealth]:
        ...

    def integrity_check(self) -> dict[str, Any]:
        ...

    def drift_check(self) -> dict[str, Any]:
        ...


class RecoveryLayer(Protocol):
    def create_snapshot(self, *, reason: str, user_id: str = "") -> int:
        ...

    def rollback(self, snapshot_id: int) -> RecoveryAction:
        ...

    def set_safe_mode(self, enabled: bool, *, reason: str = "") -> RecoveryAction:
        ...


class EvolutionLayer(Protocol):
    def record_release(self, component: str, version: str, metadata: dict[str, Any] | None = None) -> None:
        ...

    def plan(self) -> dict[str, Any]:
        ...


class PromptBrainLayer(Protocol):
    def run_prompt(
        self,
        *,
        prompt_name: str,
        variables: dict[str, Any],
        user_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        ...

    def create_file(self, *, relative_path: str, content: str, user_id: str = "") -> dict[str, Any]:
        ...

    def update_file(self, *, relative_path: str, content: str, user_id: str = "") -> dict[str, Any]:
        ...

    def delete_file(self, *, relative_path: str, user_id: str = "") -> dict[str, Any]:
        ...


class ErrorSink(Protocol):
    def capture_failure(self, failure: FailureRecord) -> int:
        ...
