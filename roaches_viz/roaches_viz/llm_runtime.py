from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

_PROVIDER_MODULE: Any | None = None
_PROVIDER_ATTEMPTED = False


def _load_provider() -> Any | None:
    global _PROVIDER_MODULE, _PROVIDER_ATTEMPTED
    if _PROVIDER_ATTEMPTED:
        return _PROVIDER_MODULE
    _PROVIDER_ATTEMPTED = True
    provider_path = Path(__file__).resolve().parents[2] / "src" / "utils" / "local_llm_provider.py"
    if not provider_path.exists():
        _PROVIDER_MODULE = None
        return None
    spec = importlib.util.spec_from_file_location("roaches_viz_legacy_local_llm_provider", provider_path)
    if spec is None or spec.loader is None:
        _PROVIDER_MODULE = None
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _PROVIDER_MODULE = module
    return module


def list_runtime_models() -> dict[str, Any]:
    provider = _load_provider()
    if provider is None:
        return {
            "ok": False,
            "provider": "missing",
            "error": "local GGUF provider is not available in this workspace",
            "advisors": [],
        }
    try:
        payload = provider.list_model_advisors()
    except Exception as exc:
        return {"ok": False, "provider": "legacy_local_llm_provider", "error": str(exc), "advisors": []}
    return {"ok": True, "provider": "legacy_local_llm_provider", **payload}


def build_role_llm(
    role: str,
    *,
    n_ctx: int | None = None,
    max_tokens: int | None = None,
) -> Callable[[str], str] | None:
    provider = _load_provider()
    if provider is None:
        return None
    return provider.build_role_llm_fn(role, n_ctx=n_ctx, max_tokens=max_tokens)


def build_fast_llm() -> Callable[[str], str] | None:
    return build_role_llm("analyst", n_ctx=1024, max_tokens=768)


def build_reasoning_llm(role: str = "general") -> Callable[[str], str] | None:
    return build_role_llm(role, n_ctx=2048, max_tokens=2048)


def build_uncensored_llm() -> Callable[[str], str] | None:
    return build_role_llm("uncensored", n_ctx=2048, max_tokens=2048)


def build_translator_llm() -> Callable[[str], str] | None:
    provider = _load_provider()
    if provider is None:
        return None
    try:
        advisors_payload = provider.list_model_advisors()
        advisors = list(advisors_payload.get("advisors") or [])
        translator_advisor = next((item for item in advisors if str(item.get("role") or "") == "translator"), None)
        if not translator_advisor or not translator_advisor.get("available"):
            return None
    except Exception:
        return None
    translator_role = getattr(provider, "ROLE_TRANSLATOR", "translator")
    return provider.build_role_llm_fn(translator_role, n_ctx=2048, max_tokens=512)
