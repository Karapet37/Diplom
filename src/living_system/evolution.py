"""Evolution layer: release tracking, migration hooks and improvement planning."""

from __future__ import annotations

from typing import Any

from src.living_system.knowledge_sql import KnowledgeSQLStore


class EvolutionService:
    """Maintains long-term adaptability and upgrade continuity."""

    def __init__(self, store: KnowledgeSQLStore):
        self.store = store

    def record_release(self, component: str, version: str, metadata: dict[str, Any] | None = None) -> None:
        checksum = ""
        if metadata:
            checksum = str(metadata.get("checksum", "") or "")
        self.store.record_component_version(
            component=component,
            version=version,
            checksum=checksum,
            metadata=dict(metadata or {}),
        )

    def plan(self) -> dict[str, Any]:
        counts = self.store.table_counts()
        backlog: list[str] = []

        if counts.get("reasoning_traces", 0) < 25:
            backlog.append("collect_more_reasoning_traces")
        if counts.get("health_checks", 0) < 10:
            backlog.append("increase_health_check_frequency")
        if counts.get("snapshots", 0) < 5:
            backlog.append("enable_periodic_snapshots")
        if counts.get("errors", 0) > 20:
            backlog.append("allocate_error_burn_down_cycle")
        if counts.get("prompt_runs", 0) < 30:
            backlog.append("expand_prompt_brain_usage")

        return {
            "status": "active",
            "backlog": backlog,
            "counts": counts,
            "migration_hooks": [
                "schema_versioning",
                "snapshot_compatibility",
                "prompt_template_versioning",
            ],
        }
