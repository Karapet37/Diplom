"""Runtime control-plane for centrally gating API behaviors."""

from __future__ import annotations

from dataclasses import dataclass
import os
from threading import Lock
import time
from typing import Any, Mapping

_WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

_CONTROL_WRITE_EXEMPT_PATHS = {
    "/api/auth/token",
    "/api/control/update",
    "/api/control/reload",
}

_GRAPH_MUTATION_PATHS = {
    "/api/graph/node",
    "/api/graph/node/update",
    "/api/graph/node/delete",
    "/api/graph/edge",
    "/api/graph/edge/update",
    "/api/graph/edge/delete",
    "/api/graph/simulate",
    "/api/graph/event/reward",
    "/api/graph/relation/reinforce",
    "/api/graph/persist",
    "/api/graph/load",
    "/api/graph/clear",
    "/api/graph/seed-demo",
    "/api/graph/profile/infer",
    "/api/project/user-graph/update",
    "/api/project/llm/debate",
    "/api/project/hallucination/report",
    "/api/project/archive/chat",
    "/api/project/archive/review",
}

_KNOWLEDGE_MUTATION_PATHS = {
    "/api/living/knowledge/initialize",
    "/api/living/knowledge/branch",
    "/api/living/knowledge/merge",
    "/api/project/bootstrap",
}


def _bool_env(name: str, default: bool) -> bool:
    value = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
    return value not in {"0", "false", "no", "off", ""}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value or "").strip().lower()
    return token in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ControlPlaneFlags:
    enabled: bool = True
    read_only: bool = False
    allow_graph_writes: bool = True
    allow_project_demo: bool = True
    allow_project_daily: bool = True
    allow_autoruns_import: bool = True
    allow_client_introspection: bool = True
    allow_living_file_ops: bool = True
    allow_knowledge_mutations: bool = True
    allow_prompt_execution: bool = True

    def as_dict(self) -> dict[str, bool]:
        return {
            "enabled": bool(self.enabled),
            "read_only": bool(self.read_only),
            "allow_graph_writes": bool(self.allow_graph_writes),
            "allow_project_demo": bool(self.allow_project_demo),
            "allow_project_daily": bool(self.allow_project_daily),
            "allow_autoruns_import": bool(self.allow_autoruns_import),
            "allow_client_introspection": bool(self.allow_client_introspection),
            "allow_living_file_ops": bool(self.allow_living_file_ops),
            "allow_knowledge_mutations": bool(self.allow_knowledge_mutations),
            "allow_prompt_execution": bool(self.allow_prompt_execution),
        }


class RuntimeControlPlane:
    """Process-local control plane with safe defaults and explicit gate reasons."""

    def __init__(self, *, flags: ControlPlaneFlags, admin_key: str = ""):
        self._flags = flags
        self._admin_key = str(admin_key or "").strip()
        self._updated_at = time.time()
        self._lock = Lock()

    @classmethod
    def from_env(cls) -> "RuntimeControlPlane":
        flags = ControlPlaneFlags(
            enabled=_bool_env("CONTROL_PLANE_ENABLE", True),
            read_only=_bool_env("CONTROL_READ_ONLY", False),
            allow_graph_writes=_bool_env("CONTROL_ALLOW_GRAPH_WRITES", True),
            allow_project_demo=_bool_env("CONTROL_ALLOW_PROJECT_DEMO", True),
            allow_project_daily=_bool_env("CONTROL_ALLOW_PROJECT_DAILY", True),
            allow_autoruns_import=_bool_env("CONTROL_ALLOW_AUTORUNS_IMPORT", True),
            allow_client_introspection=_bool_env("CONTROL_ALLOW_CLIENT_INTROSPECTION", True),
            allow_living_file_ops=_bool_env("CONTROL_ALLOW_LIVING_FILE_OPS", True),
            allow_knowledge_mutations=_bool_env("CONTROL_ALLOW_KNOWLEDGE_MUTATIONS", True),
            allow_prompt_execution=_bool_env("CONTROL_ALLOW_PROMPT_EXECUTION", True),
        )
        return cls(flags=flags, admin_key=str(os.getenv("CONTROL_ADMIN_KEY", "") or ""))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "flags": self._flags.as_dict(),
                "updated_at": float(self._updated_at),
                "admin_key_configured": bool(self._admin_key),
            }

    def admin_key_configured(self) -> bool:
        with self._lock:
            return bool(self._admin_key)

    def admin_key(self) -> str:
        with self._lock:
            return str(self._admin_key)

    def apply_patch(self, patch: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(patch, Mapping):
            raise ValueError("patch must be a mapping")
        with self._lock:
            current = self._flags.as_dict()
            changed: dict[str, bool] = {}
            ignored: list[str] = []
            for key, value in patch.items():
                token = str(key or "").strip()
                if not token:
                    continue
                if token not in current:
                    ignored.append(token)
                    continue
                parsed = _to_bool(value)
                if current[token] != parsed:
                    current[token] = parsed
                    changed[token] = parsed
            self._flags = ControlPlaneFlags(**current)
            self._updated_at = time.time()
            return {
                "ok": True,
                "changed": changed,
                "ignored": ignored,
                "flags": self._flags.as_dict(),
                "updated_at": float(self._updated_at),
            }

    def reload_from_env(self) -> dict[str, Any]:
        fresh = RuntimeControlPlane.from_env()
        with self._lock:
            self._flags = fresh._flags
            self._admin_key = fresh._admin_key
            self._updated_at = time.time()
            return self.snapshot()

    def allow_request(self, *, method: str, path: str) -> tuple[bool, str]:
        req_path = str(path or "")
        req_method = str(method or "GET").upper()
        if not req_path.startswith("/api/"):
            return True, ""
        with self._lock:
            flags = self._flags
            if not flags.enabled:
                return True, ""

            if flags.read_only and req_method in _WRITE_METHODS and req_path not in _CONTROL_WRITE_EXEMPT_PATHS:
                return False, "control_read_only"

            if not flags.allow_graph_writes and req_method in _WRITE_METHODS and req_path in _GRAPH_MUTATION_PATHS:
                return False, "graph_writes_disabled"

            if not flags.allow_project_demo and req_path == "/api/project/demo/watch":
                return False, "project_demo_disabled"

            if not flags.allow_project_daily and req_path == "/api/project/daily-mode":
                return False, "project_daily_disabled"

            if not flags.allow_autoruns_import and req_path == "/api/project/autoruns/import":
                return False, "autoruns_import_disabled"

            if not flags.allow_client_introspection and req_path == "/api/client/introspect":
                return False, "client_introspection_disabled"

            if not flags.allow_living_file_ops and req_path.startswith("/api/living/file/"):
                return False, "living_file_ops_disabled"

            if not flags.allow_knowledge_mutations and req_path in _KNOWLEDGE_MUTATION_PATHS:
                return False, "knowledge_mutations_disabled"

            if not flags.allow_prompt_execution and req_path in {
                "/api/living/prompt/run",
                "/api/project/llm/debate",
                "/api/project/archive/chat",
            }:
                return False, "prompt_execution_disabled"

        return True, ""
