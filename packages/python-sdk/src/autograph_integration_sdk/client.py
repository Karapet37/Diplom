"""Packaged integration layer SDK."""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


JsonMapping = dict[str, Any]
ManifestFn = Callable[[Mapping[str, Any]], JsonMapping]
InvokeFn = Callable[[Mapping[str, Any]], JsonMapping]
HttpRequester = Callable[[str, str, Mapping[str, Any] | None, Mapping[str, str] | None, float], JsonMapping]


def _as_dict(payload: Mapping[str, Any] | None) -> JsonMapping:
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, Mapping):
        return {str(k): v for k, v in payload.items()}
    return {}


def _slug(value: Any, default: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return default
    out = "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in raw).strip("_.-")
    return out[:64] or default


def _require_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _default_http_requester(
    method: str,
    url: str,
    payload: Mapping[str, Any] | None,
    headers: Mapping[str, str] | None,
    timeout_seconds: float,
) -> JsonMapping:
    upper_method = str(method or "GET").strip().upper() or "GET"
    request_headers: dict[str, str] = {"Accept": "application/json"}
    if isinstance(headers, Mapping):
        for key, value in headers.items():
            request_headers[str(key)] = str(value)

    body: bytes | None = None
    if upper_method != "GET":
        request_headers.setdefault("Content-Type", "application/json")
        body = json.dumps(_as_dict(payload), ensure_ascii=False).encode("utf-8")

    req = Request(url=url, data=body, method=upper_method, headers=request_headers)
    try:
        with urlopen(req, timeout=float(timeout_seconds)) as resp:
            raw = resp.read().decode("utf-8") if resp else ""
        if not raw.strip():
            return {}
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"data": parsed}
    except HTTPError as exc:
        detail = ""
        try:
            raw_error = exc.read().decode("utf-8")
            payload_error = json.loads(raw_error) if raw_error.strip() else {}
            if isinstance(payload_error, dict):
                detail = str(payload_error.get("detail") or payload_error)
            else:
                detail = str(payload_error)
        except Exception:
            detail = str(exc.reason)
        raise RuntimeError(f"integration layer http error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"integration layer connection failed: {exc.reason}") from exc


class IntegrationLayerClient:
    """Standalone or HTTP client for the integration layer."""

    def __init__(
        self,
        *,
        manifest_fn: ManifestFn,
        invoke_fn: InvokeFn,
        default_host: str = "generic",
        default_app_id: str = "external_app",
    ) -> None:
        self._manifest_fn = manifest_fn
        self._invoke_fn = invoke_fn
        self.default_host = _slug(default_host, "generic")
        self.default_app_id = _slug(default_app_id, "external_app")

    @classmethod
    def from_workspace(
        cls,
        workspace: Any,
        *,
        host: str = "generic",
        app_id: str = "external_app",
    ) -> "IntegrationLayerClient":
        if workspace is None:
            raise ValueError("workspace is required")
        if not hasattr(workspace, "project_integration_layer_manifest"):
            raise ValueError("workspace must provide project_integration_layer_manifest")
        if not hasattr(workspace, "project_integration_layer_invoke"):
            raise ValueError("workspace must provide project_integration_layer_invoke")
        return cls(
            manifest_fn=lambda payload: _as_dict(workspace.project_integration_layer_manifest(payload)),
            invoke_fn=lambda payload: _as_dict(workspace.project_integration_layer_invoke(payload)),
            default_host=host,
            default_app_id=app_id,
        )

    @classmethod
    def from_http(
        cls,
        base_url: str,
        *,
        host: str = "generic",
        app_id: str = "external_app",
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 20.0,
        requester: HttpRequester | None = None,
    ) -> "IntegrationLayerClient":
        base = str(base_url or "").strip().rstrip("/")
        if not base:
            raise ValueError("base_url is required for HTTP integration mode")
        http_request = requester or _default_http_requester

        def _manifest(payload: Mapping[str, Any]) -> JsonMapping:
            root = _as_dict(payload)
            query = urlencode(
                {
                    "host": _slug(root.get("host"), _slug(host, "generic")),
                    "app_id": _slug(root.get("app_id"), _slug(app_id, "external_app")),
                }
            )
            url = f"{base}/api/integration/layer/manifest?{query}"
            return _as_dict(http_request("GET", url, None, headers, timeout_seconds))

        def _invoke(payload: Mapping[str, Any]) -> JsonMapping:
            url = f"{base}/api/integration/layer/invoke"
            return _as_dict(http_request("POST", url, _as_dict(payload), headers, timeout_seconds))

        return cls(
            manifest_fn=_manifest,
            invoke_fn=_invoke,
            default_host=host,
            default_app_id=app_id,
        )

    def manifest(self, *, host: str | None = None, app_id: str | None = None) -> JsonMapping:
        payload = {
            "host": _slug(host, self.default_host),
            "app_id": _slug(app_id, self.default_app_id),
        }
        return _as_dict(self._manifest_fn(payload))

    def invoke(self, payload: Mapping[str, Any]) -> JsonMapping:
        root = _as_dict(payload)
        root["host"] = _slug(root.get("host"), self.default_host)
        root["app_id"] = _slug(root.get("app_id"), self.default_app_id)
        return _as_dict(self._invoke_fn(root))

    def invoke_action(
        self,
        action: str,
        *,
        input_payload: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
        user_id: str = "default_user",
        session_id: str = "",
        host: str | None = None,
        app_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> JsonMapping:
        payload: JsonMapping = {
            "action": _require_text(action, "action"),
            "host": _slug(host, self.default_host),
            "app_id": _slug(app_id, self.default_app_id),
            "user_id": str(user_id or "default_user").strip() or "default_user",
            "session_id": str(session_id or "").strip(),
            "input": _as_dict(input_payload),
            "options": _as_dict(options),
        }
        extra_payload = _as_dict(extra)
        for key, value in extra_payload.items():
            if key in {"action", "host", "app_id", "user_id", "session_id", "input", "options"}:
                continue
            payload[key] = value
        return self.invoke(payload)

    def respond(
        self,
        message: str,
        *,
        user_id: str = "default_user",
        session_id: str = "",
        role: str = "general",
        model_path: str = "",
        use_memory: bool = True,
        auto_triage: bool = True,
        triage_with_llm: bool = True,
    ) -> JsonMapping:
        return self.invoke_action(
            "wrapper.respond",
            user_id=user_id,
            session_id=session_id,
            input_payload={"message": _require_text(message, "message")},
            options={
                "role": str(role or "general").strip() or "general",
                "model_path": str(model_path or "").strip(),
                "use_memory": bool(use_memory),
                "auto_triage": bool(auto_triage),
                "triage_with_llm": bool(triage_with_llm),
            },
        )

    def archive_chat(
        self,
        message: str,
        *,
        user_id: str = "default_user",
        session_id: str = "",
        context: str = "",
        model_path: str = "",
        model_role: str = "general",
        apply_to_graph: bool = True,
        auto_triage: bool = True,
        triage_with_llm: bool = True,
    ) -> JsonMapping:
        return self.invoke_action(
            "archive.chat",
            user_id=user_id,
            session_id=session_id,
            input_payload={
                "message": _require_text(message, "message"),
                "context": str(context or "").strip(),
                "model_path": str(model_path or "").strip(),
                "model_role": str(model_role or "general").strip() or "general",
            },
            options={
                "model_path": str(model_path or "").strip(),
                "model_role": str(model_role or "general").strip() or "general",
                "apply_to_graph": bool(apply_to_graph),
                "auto_triage": bool(auto_triage),
                "triage_with_llm": bool(triage_with_llm),
            },
        )

    def update_user_graph(
        self,
        *,
        text: str,
        user_id: str = "default_user",
        session_id: str = "",
        display_name: str = "",
        language: str = "en",
        use_llm_profile: bool = True,
    ) -> JsonMapping:
        return self.invoke_action(
            "user_graph.update",
            user_id=user_id,
            session_id=session_id,
            input_payload={
                "message": _require_text(text, "text"),
                "display_name": str(display_name or user_id).strip() or str(user_id),
                "language": str(language or "en").strip() or "en",
            },
            options={
                "use_llm_profile": bool(use_llm_profile),
            },
        )

    def ingest_personal_tree(
        self,
        *,
        text: str,
        user_id: str = "default_user",
        session_id: str = "",
        title: str = "",
        topic: str = "",
        source_url: str = "",
        source_title: str = "",
        max_points: int = 6,
    ) -> JsonMapping:
        return self.invoke_action(
            "personal_tree.ingest",
            user_id=user_id,
            session_id=session_id,
            input_payload={
                "message": _require_text(text, "text"),
                "title": str(title or "").strip(),
                "topic": str(topic or "").strip(),
                "source_url": str(source_url or "").strip(),
                "source_title": str(source_title or "").strip(),
                "max_points": int(max(2, min(12, _to_int(max_points, 6)))),
            },
        )


__all__ = ["IntegrationLayerClient"]
