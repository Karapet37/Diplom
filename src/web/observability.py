"""Runtime metrics collection and Prometheus exposition helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os
import resource
import time
from threading import Lock
from typing import Any


def _escape_label(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


@dataclass
class _PathLatency:
    count: int = 0
    total: float = 0.0
    max_latency: float = 0.0


class RuntimeMetrics:
    """In-memory metrics registry exposed in Prometheus text format."""

    def __init__(self):
        self.started_at = time.time()
        self._lock = Lock()
        self._inflight = 0
        self._request_total: dict[tuple[str, str, int], int] = defaultdict(int)
        self._request_errors: dict[tuple[str, str], int] = defaultdict(int)
        self._request_latency: dict[tuple[str, str], _PathLatency] = defaultdict(_PathLatency)
        self._inference_total: dict[str, int] = defaultdict(int)
        self._inference_latency: dict[str, _PathLatency] = defaultdict(_PathLatency)

    def mark_inflight(self, delta: int) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight + int(delta))

    def record_request(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        latency_seconds: float,
        is_inference: bool,
    ) -> None:
        method_key = str(method or "GET").upper()
        path_key = str(path or "/")
        status_key = int(status_code)
        latency = max(0.0, float(latency_seconds))
        with self._lock:
            self._request_total[(method_key, path_key, status_key)] += 1
            if status_key >= 400:
                self._request_errors[(method_key, path_key)] += 1

            row = self._request_latency[(method_key, path_key)]
            row.count += 1
            row.total += latency
            row.max_latency = max(row.max_latency, latency)

            if is_inference:
                self._inference_total[path_key] += 1
                inf_row = self._inference_latency[path_key]
                inf_row.count += 1
                inf_row.total += latency
                inf_row.max_latency = max(inf_row.max_latency, latency)

    @staticmethod
    def _process_metrics() -> dict[str, float]:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux ru_maxrss is in kilobytes.
        memory_bytes = float(usage.ru_maxrss) * 1024.0
        cpu_seconds = float(usage.ru_utime) + float(usage.ru_stime)
        load_avg = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
        return {
            "memory_bytes": memory_bytes,
            "cpu_seconds": cpu_seconds,
            "load1": float(load_avg[0]),
            "load5": float(load_avg[1]),
            "load15": float(load_avg[2]),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "started_at": self.started_at,
                "inflight": self._inflight,
                "request_total": dict(self._request_total),
                "request_errors": dict(self._request_errors),
                "request_latency": {
                    key: _PathLatency(value.count, value.total, value.max_latency)
                    for key, value in self._request_latency.items()
                },
                "inference_total": dict(self._inference_total),
                "inference_latency": {
                    key: _PathLatency(value.count, value.total, value.max_latency)
                    for key, value in self._inference_latency.items()
                },
            }

    def render_prometheus(self, *, extra_metrics: dict[str, float] | None = None) -> str:
        snap = self.snapshot()
        proc = self._process_metrics()
        uptime = max(0.0, time.time() - float(snap["started_at"]))
        lines: list[str] = []

        lines.append("# HELP autograph_uptime_seconds Process uptime in seconds")
        lines.append("# TYPE autograph_uptime_seconds gauge")
        lines.append(f"autograph_uptime_seconds {uptime:.6f}")

        lines.append("# HELP autograph_active_requests Current in-flight requests")
        lines.append("# TYPE autograph_active_requests gauge")
        lines.append(f"autograph_active_requests {int(snap['inflight'])}")

        lines.append("# HELP autograph_process_memory_bytes Process max RSS memory")
        lines.append("# TYPE autograph_process_memory_bytes gauge")
        lines.append(f"autograph_process_memory_bytes {proc['memory_bytes']:.3f}")

        lines.append("# HELP autograph_process_cpu_seconds_total CPU seconds consumed")
        lines.append("# TYPE autograph_process_cpu_seconds_total counter")
        lines.append(f"autograph_process_cpu_seconds_total {proc['cpu_seconds']:.6f}")

        lines.append("# HELP autograph_process_load_avg System load average")
        lines.append("# TYPE autograph_process_load_avg gauge")
        lines.append(f"autograph_process_load_avg{{window=\"1m\"}} {proc['load1']:.6f}")
        lines.append(f"autograph_process_load_avg{{window=\"5m\"}} {proc['load5']:.6f}")
        lines.append(f"autograph_process_load_avg{{window=\"15m\"}} {proc['load15']:.6f}")

        lines.append("# HELP autograph_requests_total Total HTTP requests")
        lines.append("# TYPE autograph_requests_total counter")
        for (method, path, status), value in sorted(snap["request_total"].items()):
            labels = (
                f'method="{_escape_label(method)}",'
                f'path="{_escape_label(path)}",'
                f'status="{int(status)}"'
            )
            lines.append(f"autograph_requests_total{{{labels}}} {int(value)}")

        lines.append("# HELP autograph_request_errors_total Total HTTP errors")
        lines.append("# TYPE autograph_request_errors_total counter")
        for (method, path), value in sorted(snap["request_errors"].items()):
            labels = f'method="{_escape_label(method)}",path="{_escape_label(path)}"'
            lines.append(f"autograph_request_errors_total{{{labels}}} {int(value)}")

        lines.append("# HELP autograph_request_latency_seconds Request latency stats")
        lines.append("# TYPE autograph_request_latency_seconds summary")
        for (method, path), row in sorted(snap["request_latency"].items()):
            labels = f'method="{_escape_label(method)}",path="{_escape_label(path)}"'
            lines.append(f"autograph_request_latency_seconds_sum{{{labels}}} {row.total:.6f}")
            lines.append(f"autograph_request_latency_seconds_count{{{labels}}} {int(row.count)}")
            lines.append(f"autograph_request_latency_seconds_max{{{labels}}} {row.max_latency:.6f}")

        lines.append("# HELP autograph_inference_requests_total AI inference calls")
        lines.append("# TYPE autograph_inference_requests_total counter")
        for path, value in sorted(snap["inference_total"].items()):
            labels = f'path="{_escape_label(path)}"'
            lines.append(f"autograph_inference_requests_total{{{labels}}} {int(value)}")

        lines.append("# HELP autograph_inference_latency_seconds AI inference latency")
        lines.append("# TYPE autograph_inference_latency_seconds summary")
        for path, row in sorted(snap["inference_latency"].items()):
            labels = f'path="{_escape_label(path)}"'
            lines.append(f"autograph_inference_latency_seconds_sum{{{labels}}} {row.total:.6f}")
            lines.append(f"autograph_inference_latency_seconds_count{{{labels}}} {int(row.count)}")
            lines.append(f"autograph_inference_latency_seconds_max{{{labels}}} {row.max_latency:.6f}")

        if extra_metrics:
            for name, value in sorted(extra_metrics.items()):
                metric_name = str(name).strip()
                if not metric_name:
                    continue
                lines.append(f"{metric_name} {float(value):.6f}")

        return "\n".join(lines) + "\n"


INFERENCE_PATHS: tuple[str, ...] = (
    "/api/graph/foundation/create",
    "/api/graph/profile/infer",
    "/api/graph/simulate",
    "/api/living/process",
    "/api/living/prompt/run",
    "/api/project/archive/chat",
    "/api/integration/layer/invoke",
)


def is_inference_path(path: str) -> bool:
    return str(path or "/") in INFERENCE_PATHS
