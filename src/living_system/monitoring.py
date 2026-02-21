"""Monitoring layer for telemetry, process health and behavior traces."""

from __future__ import annotations

import os
import platform
import resource
import time
from typing import Any

from src.living_system.knowledge_sql import KnowledgeSQLStore
from src.living_system.models import TelemetryEvent


class MonitoringService:
    """Captures observability signals and stores them in SQL logs."""

    def __init__(self, store: KnowledgeSQLStore):
        self.store = store

    def emit(self, event: TelemetryEvent) -> None:
        self.store.append_log(
            level=event.level,
            component=event.component,
            message=f"{event.event_type}: {event.message}",
            details=event.details,
            user_id=event.user_id,
        )

    def process_snapshot(self) -> dict[str, Any]:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
        return {
            "timestamp": time.time(),
            "platform": f"{platform.system()} {platform.release()}",
            "pid": os.getpid(),
            "memory_mb": round(float(usage.ru_maxrss) / 1024.0, 3),
            "cpu_user_s": float(usage.ru_utime),
            "cpu_system_s": float(usage.ru_stime),
            "load_avg": [float(load[0]), float(load[1]), float(load[2])],
        }

    def heartbeat(self, *, user_id: str = "") -> dict[str, Any]:
        snapshot = self.process_snapshot()
        self.store.append_log(
            level="info",
            component="monitoring",
            message="heartbeat",
            details=snapshot,
            user_id=user_id,
        )
        return snapshot
