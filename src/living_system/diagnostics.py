"""Self-diagnostics layer: health, integrity, drift and uncertainty checks."""

from __future__ import annotations

from statistics import mean
from typing import Any

from src.living_system.knowledge_sql import KnowledgeSQLStore
from src.living_system.models import LayerHealth


class DiagnosticsService:
    """Runs periodic diagnostics and records machine-readable health reports."""

    def __init__(self, store: KnowledgeSQLStore):
        self.store = store

    def health_check(self) -> list[LayerHealth]:
        table_presence = self.store.required_tables_present()
        integrity_score = sum(1 for ok in table_presence.values() if ok) / max(1, len(table_presence))
        integrity_status = "ok" if integrity_score >= 1.0 else "degraded"

        confidences = self.store.recent_reasoning_confidences(limit=50)
        avg_conf = mean(confidences) if confidences else 1.0
        confidence_status = "ok" if avg_conf >= 0.6 else "warning"

        errors = self.store.latest_errors(limit=20)
        critical_count = sum(1 for row in errors if row.get("severity") == "critical")
        error_status = "ok" if critical_count == 0 else "critical"
        error_score = 1.0 if critical_count == 0 else max(0.0, 1.0 - (critical_count / 10.0))

        rows = [
            LayerHealth(
                layer="knowledge_sql",
                status=integrity_status,
                score=round(integrity_score, 4),
                details={"tables": table_presence},
            ),
            LayerHealth(
                layer="reasoning_confidence",
                status=confidence_status,
                score=round(float(avg_conf), 4),
                details={"sample_size": len(confidences)},
            ),
            LayerHealth(
                layer="error_budget",
                status=error_status,
                score=round(float(error_score), 4),
                details={"critical_errors": critical_count},
            ),
        ]

        for item in rows:
            self.store.record_health(
                layer_name=item.layer,
                status=item.status,
                score=item.score,
                details=item.details,
            )
        return rows

    def integrity_check(self) -> dict[str, Any]:
        presence = self.store.required_tables_present()
        missing = [name for name, ok in presence.items() if not ok]
        return {
            "ok": not missing,
            "missing_tables": missing,
            "tables": presence,
        }

    def drift_check(self) -> dict[str, Any]:
        confidences = self.store.recent_reasoning_confidences(limit=100)
        if not confidences:
            return {
                "ok": True,
                "confidence_drift": 0.0,
                "sample_size": 0,
                "message": "not_enough_data",
            }

        midpoint = max(1, len(confidences) // 2)
        recent = confidences[:midpoint]
        older = confidences[midpoint:]
        recent_avg = mean(recent) if recent else 1.0
        older_avg = mean(older) if older else recent_avg
        drift = recent_avg - older_avg

        return {
            "ok": drift > -0.2,
            "confidence_drift": round(float(drift), 4),
            "recent_avg": round(float(recent_avg), 4),
            "older_avg": round(float(older_avg), 4),
            "sample_size": len(confidences),
            "message": "confidence_drop" if drift <= -0.2 else "stable",
        }

    def uncertainty_monitor(self) -> dict[str, Any]:
        confidences = self.store.recent_reasoning_confidences(limit=100)
        uncertain = [value for value in confidences if value < 0.55]
        ratio = (len(uncertain) / len(confidences)) if confidences else 0.0
        return {
            "ok": ratio < 0.4,
            "uncertain_ratio": round(float(ratio), 4),
            "threshold": 0.4,
            "sample_size": len(confidences),
        }
