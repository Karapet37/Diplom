from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class GraphMemoryZones:
    """Filesystem zones for pending and verified graph knowledge artifacts."""

    def __init__(self, root: Path | None = None):
        base = root or (_repo_root() / "graph")
        self.root = base
        self.pending_dir = self.root / "pending"
        self.verified_dir = self.root / "verified"
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.verified_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def pending_path(self, job_id: str) -> Path:
        return self.pending_dir / f"{job_id}.json"

    def verified_path(self, job_id: str) -> Path:
        return self.verified_dir / f"{job_id}.json"

    def write_pending(self, job_id: str, payload: dict[str, Any]) -> Path:
        path = self.pending_path(job_id)
        with self._lock:
            self._write_json(path, payload)
        return path

    def update_pending(self, job_id: str, changes: dict[str, Any]) -> Path:
        path = self.pending_path(job_id)
        with self._lock:
            payload = self._load_json(path)
            payload.update(changes)
            self._write_json(path, payload)
        return path

    def move_to_verified(self, job_id: str, payload: dict[str, Any]) -> Path:
        pending = self.pending_path(job_id)
        verified = self.verified_path(job_id)
        with self._lock:
            self._write_json(verified, payload)
            if pending.exists():
                pending.unlink()
        return verified

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)
