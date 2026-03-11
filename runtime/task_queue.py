from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import queue
import threading
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class TaskEnvelope:
    job_id: str
    kind: str
    payload: dict[str, Any]


class TaskQueue:
    def __init__(self, *, name: str = "task-queue", maxsize: int = 2048):
        self.name = name
        self._queue: queue.Queue[TaskEnvelope] = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._counter = 0

    def submit(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._counter += 1
            job_id = f"{kind}-{self._counter}"
            record = {
                "job_id": job_id,
                "kind": kind,
                "status": "queued",
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "payload": dict(payload),
            }
            self._jobs[job_id] = record
        self._queue.put(TaskEnvelope(job_id=job_id, kind=kind, payload=dict(payload)))
        return self.get(job_id) or {}

    def claim(self, timeout: float = 0.5) -> TaskEnvelope | None:
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        self.update(item.job_id, status="running")
        return item

    def complete(self, job_id: str, **changes: Any) -> None:
        self.update(job_id, status="done", **changes)

    def fail(self, job_id: str, reason: str, **changes: Any) -> None:
        self.update(job_id, status="failed", reason=reason, **changes)

    def skip(self, job_id: str, reason: str, **changes: Any) -> None:
        self.update(job_id, status="skipped", reason=reason, **changes)

    def update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            current = dict(self._jobs.get(job_id) or {})
            current.update(changes)
            current["updated_at"] = _now_iso()
            self._jobs[job_id] = current

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._jobs.get(job_id)
            return dict(item) if item else None

    def summary(self) -> dict[str, Any]:
        with self._lock:
            jobs = [dict(item) for item in self._jobs.values()]
        return {
            "queued": sum(1 for item in jobs if item.get("status") == "queued"),
            "running": sum(1 for item in jobs if item.get("status") == "running"),
            "done": sum(1 for item in jobs if item.get("status") == "done"),
            "failed": sum(1 for item in jobs if item.get("status") == "failed"),
            "skipped": sum(1 for item in jobs if item.get("status") == "skipped"),
            "recent": jobs[-8:],
        }

