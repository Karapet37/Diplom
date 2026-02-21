"""Local GGUF LLM provider with role-based model selection."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Callable

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from threading import Lock

# Backward-compatible single-model globals.
_LLM_INSTANCE: "Llama" | None = None
_LLM_FN: Callable[[str], str] | None = None
_LLM_UNAVAILABLE = False
_LAST_ERROR = ""

# Role/path caches for advisor models.
_ROLE_MODEL_MAP: dict[str, str] = {}
_PATH_LLM_INSTANCE: dict[str, "Llama"] = {}
_PATH_LLM_FN: dict[str, Callable[[str], str]] = {}
_ROLE_LLM_FN: dict[str, Callable[[str], str]] = {}
_ROLE_ERRORS_WARNED: set[str] = set()

_LLM_LOCK = Lock()

ROLE_GENERAL = "general"
ROLE_TRANSLATOR = "translator"
ROLE_ANALYST = "analyst"
ROLE_CREATIVE = "creative"
ROLE_PLANNER = "planner"
ROLE_CODER_ARCHITECT = "coder_architect"
ROLE_CODER_REVIEWER = "coder_reviewer"
ROLE_CODER_REFACTOR = "coder_refactor"
ROLE_CODER_DEBUG = "coder_debug"

ADVISOR_ROLES: tuple[str, ...] = (
    ROLE_GENERAL,
    ROLE_ANALYST,
    ROLE_CREATIVE,
    ROLE_PLANNER,
    ROLE_CODER_ARCHITECT,
    ROLE_CODER_REVIEWER,
    ROLE_CODER_REFACTOR,
    ROLE_CODER_DEBUG,
    ROLE_TRANSLATOR,
)

_ROLE_ENV_MAP: dict[str, str] = {
    ROLE_GENERAL: "LOCAL_GGUF_MODEL",
    ROLE_TRANSLATOR: "LOCAL_TRANSLATOR_GGUF_MODEL",
    ROLE_ANALYST: "LOCAL_ANALYST_GGUF_MODEL",
    ROLE_CREATIVE: "LOCAL_CREATIVE_GGUF_MODEL",
    ROLE_PLANNER: "LOCAL_PLANNER_GGUF_MODEL",
    ROLE_CODER_ARCHITECT: "LOCAL_CODER_ARCHITECT_GGUF_MODEL",
    ROLE_CODER_REVIEWER: "LOCAL_CODER_REVIEWER_GGUF_MODEL",
    ROLE_CODER_REFACTOR: "LOCAL_CODER_REFACTOR_GGUF_MODEL",
    ROLE_CODER_DEBUG: "LOCAL_CODER_DEBUG_GGUF_MODEL",
}

_CODER_HINTS: tuple[str, ...] = (
    "coder",
    "code",
    "codestral",
    "codellama",
    "starcoder",
    "deepseek-coder",
    "qwen-coder",
    "programming",
    "dev",
)
_TRANSLATOR_HINTS: tuple[str, ...] = (
    "translator",
    "translate",
    "nllb",
    "m2m",
    "madlad",
)
_MADLAD_HINTS: tuple[str, ...] = (
    "madlad400",
    "madlad-400",
    "madlad",
)
_GGUF_SPLIT_RE = re.compile(r"^(?P<base>.+)-(?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$", flags=re.IGNORECASE)
_IGNORED_GGUF_HINTS: tuple[str, ...] = (
    "/llama.cpp/",
    "ggml-vocab",
)


def _warn(message: str) -> None:
    silent = str(os.getenv("LOCAL_LLM_SILENT_ERRORS", "0")).strip().lower()
    if silent in {"1", "true", "yes", "on"}:
        return
    print(message)


def _normalize_role(role: str) -> str:
    token = re.sub(r"[^a-z0-9_]+", "_", str(role or "").strip().lower())
    return token or ROLE_GENERAL


def _contains_any(token: str, hints: tuple[str, ...]) -> bool:
    source = str(token or "")
    return any(hint in source for hint in hints)


def _path_token(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").lower()


def _split_info(path: Path) -> tuple[str, int, int] | None:
    match = _GGUF_SPLIT_RE.match(path.name)
    if not match:
        return None
    base = str(match.group("base") or "")
    try:
        part = int(match.group("part") or "0")
        total = int(match.group("total") or "0")
    except Exception:
        return None
    if not base or part <= 0 or total <= 0:
        return None
    return (base, part, total)


def _resolve_entrypoint(path: Path) -> Path | None:
    """
    Normalize GGUF path to a valid load entrypoint.

    For split GGUF, only shard `00001-of-xxxxx` should be used as model_path.
    If a non-first shard is provided, auto-map to first shard when available.
    """
    info = _split_info(path)
    if info is None:
        return path if path.exists() and path.is_file() else None
    base, part, total = info
    first = path.with_name(f"{base}-00001-of-{total:05d}.gguf")
    if part == 1:
        return path if path.exists() and path.is_file() else None
    if first.exists() and first.is_file():
        return first
    return None


def _is_candidate_gguf(path: Path) -> bool:
    token = _path_token(path)
    return not any(hint in token for hint in _IGNORED_GGUF_HINTS)


def _iter_model_dirs() -> list[Path]:
    raw_dirs = [
        str(os.getenv("LOCAL_MODELS_DIR", "models/gguf")).strip(),
        "models/gguf",
        "models",
    ]
    out: list[Path] = []
    seen: set[str] = set()
    for raw in raw_dirs:
        if not raw:
            continue
        path = Path(raw).expanduser()
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists() and path.is_dir():
            out.append(path)
    return out


def _discover_gguf_paths() -> list[Path]:
    rows: list[Path] = []
    seen: set[str] = set()
    for root in _iter_model_dirs():
        for path in root.rglob("*.gguf"):
            if not path.is_file():
                continue
            if not _is_candidate_gguf(path):
                continue
            entrypoint = _resolve_entrypoint(path)
            if entrypoint is None:
                continue
            token = _path_token(entrypoint)
            if token in seen:
                continue
            seen.add(token)
            rows.append(entrypoint)
    rows.sort(key=lambda item: _path_token(item))
    return rows


def _score_general(path: Path) -> float:
    token = _path_token(path)
    score = 0.0
    if _contains_any(token, _MADLAD_HINTS):
        score -= 4.0
    if _contains_any(token, _TRANSLATOR_HINTS):
        score -= 2.0
    if _contains_any(token, _CODER_HINTS):
        score -= 0.8
    if _contains_any(token, ("mistral", "llama", "qwen", "phi", "gemma", "instruct")):
        score += 2.0
    if "instruct" in token:
        score += 0.5
    return score


def _select_best(paths: list[Path], scorer: Callable[[Path], float]) -> Path | None:
    if not paths:
        return None
    ranked = sorted(paths, key=lambda item: (scorer(item), _path_token(item)), reverse=True)
    return ranked[0]


def _resolve_model_role_paths() -> dict[str, str]:
    role_map: dict[str, str] = {}

    # 1) Explicit env overrides.
    for role, env_name in _ROLE_ENV_MAP.items():
        explicit = str(os.getenv(env_name, "") or "").strip()
        if not explicit:
            continue
        path = Path(explicit).expanduser()
        normalized = _resolve_entrypoint(path)
        if normalized is None:
            if path.exists() and path.is_file():
                _warn(
                    f"[local_llm_provider] WARN: ignored non-entrypoint GGUF for {role}: {path}. "
                    "Use shard 00001-of-N for split models."
                )
            continue
        if role == ROLE_TRANSLATOR and normalized.suffix.casefold() != ".gguf":
            continue
        if normalized != path:
            _warn(
                f"[local_llm_provider] INFO: remapped {role} model from split shard {path.name} "
                f"to entrypoint {normalized.name}."
            )
        role_map[role] = str(normalized)

    # 2) Auto discovery.
    models = _discover_gguf_paths()
    if not models:
        return role_map

    translator_candidates: list[Path] = []
    coder_candidates: list[Path] = []
    general_candidates: list[Path] = []
    creative_candidates: list[Path] = []
    analyst_candidates: list[Path] = []
    planner_candidates: list[Path] = []

    for path in models:
        token = _path_token(path)
        is_madlad = _contains_any(token, _MADLAD_HINTS)
        is_translator = is_madlad or _contains_any(token, _TRANSLATOR_HINTS)
        is_coder = _contains_any(token, _CODER_HINTS)

        if is_translator:
            translator_candidates.append(path)
        if is_coder:
            coder_candidates.append(path)
        if _contains_any(token, ("danube", "h2o", "creative", "story")):
            creative_candidates.append(path)
        if _contains_any(token, ("deepseek", "analyst", "reason", "logic")):
            analyst_candidates.append(path)
        if _contains_any(token, ("planner", "plan", "instruct")):
            planner_candidates.append(path)
        if not is_translator:
            general_candidates.append(path)

    madlad = _select_best(
        [item for item in translator_candidates if _contains_any(_path_token(item), _MADLAD_HINTS)],
        scorer=lambda _: 10.0,
    )
    if ROLE_TRANSLATOR not in role_map:
        best_translator = madlad or _select_best(
            translator_candidates,
            scorer=lambda item: 2.0 if _contains_any(_path_token(item), ("google", "translator")) else 1.0,
        )
        if best_translator is not None:
            role_map[ROLE_TRANSLATOR] = str(best_translator)

    default_coder = _select_best(coder_candidates, scorer=lambda item: _score_general(item) + 1.5)
    for role, hints in (
        (ROLE_CODER_ARCHITECT, ("architect", "design", "plan")),
        (ROLE_CODER_REVIEWER, ("review", "critic", "audit")),
        (ROLE_CODER_REFACTOR, ("refactor", "cleanup", "optimi")),
        (ROLE_CODER_DEBUG, ("debug", "bug", "fix")),
    ):
        if role in role_map:
            continue
        specific = _select_best(
            [item for item in coder_candidates if _contains_any(_path_token(item), hints)],
            scorer=lambda item: _score_general(item) + 2.0,
        )
        if specific is not None:
            role_map[role] = str(specific)
        elif default_coder is not None:
            role_map[role] = str(default_coder)

    if ROLE_ANALYST not in role_map:
        pick = _select_best(analyst_candidates, scorer=_score_general)
        if pick is not None:
            role_map[ROLE_ANALYST] = str(pick)
    if ROLE_CREATIVE not in role_map:
        pick = _select_best(creative_candidates, scorer=_score_general)
        if pick is not None:
            role_map[ROLE_CREATIVE] = str(pick)
    if ROLE_PLANNER not in role_map:
        pick = _select_best(planner_candidates, scorer=_score_general)
        if pick is not None:
            role_map[ROLE_PLANNER] = str(pick)

    if ROLE_GENERAL not in role_map:
        best_general = _select_best(general_candidates, scorer=_score_general)
        if best_general is not None:
            role_map[ROLE_GENERAL] = str(best_general)
        elif ROLE_TRANSLATOR in role_map:
            # Last-resort fallback to keep system operational.
            role_map[ROLE_GENERAL] = role_map[ROLE_TRANSLATOR]

    return role_map


def _get_model_path() -> str | None:
    """
    Backward-compatible general model resolver.
    Prefer LOCAL_GGUF_MODEL; fallback to discovered general role in models/gguf.
    """
    explicit = str(os.getenv("LOCAL_GGUF_MODEL", "") or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        normalized = _resolve_entrypoint(path)
        if normalized is not None:
            if normalized != path:
                _warn(
                    f"[local_llm_provider] INFO: LOCAL_GGUF_MODEL remapped from {path.name} "
                    f"to split entrypoint {normalized.name}."
                )
            return str(normalized)
        if path.exists() and path.is_file():
            _warn(
                f"[local_llm_provider] WARN: LOCAL_GGUF_MODEL points to non-entrypoint split shard: {path}. "
                "Use shard 00001-of-N."
            )

    role_map = _resolve_model_role_paths()
    return role_map.get(ROLE_GENERAL)


def _build_llm_for_path(model_path: str) -> "Llama" | None:
    if Llama is None:
        return None
    return Llama(
        model_path=str(model_path),
        n_ctx=int(os.getenv("LOCAL_GGUF_N_CTX", "8192")),
        temperature=float(os.getenv("LOCAL_GGUF_TEMPERATURE", "0.15")),
        verbose=False,
    )


def _make_llm_fn(llm: "Llama") -> Callable[[str], str]:
    def llm_fn(prompt: str) -> str:
        try:
            response = llm.create_chat_completion(
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=int(os.getenv("LOCAL_GGUF_MAX_TOKENS", "2048")),
                temperature=float(os.getenv("LOCAL_GGUF_TEMPERATURE", "0.15")),
                top_p=float(os.getenv("LOCAL_GGUF_TOP_P", "0.9")),
            )
            result_text = response["choices"][0]["message"]["content"]
        except Exception:
            try:
                out = llm.create_chat_completion(
                    messages=[{"role": "system", "content": prompt}],
                    max_tokens=int(os.getenv("LOCAL_GGUF_MAX_TOKENS", "2048")),
                    temperature=float(os.getenv("LOCAL_GGUF_TEMPERATURE", "0.15")),
                    top_p=float(os.getenv("LOCAL_GGUF_TOP_P", "0.9")),
                )
                result_text = out["choices"][0]["message"]["content"]
            except Exception as inner_exc:
                _warn(f"[local_llm_provider] WARN: LLM inference failed: {inner_exc}")
                return ""
        return str(result_text or "")

    return llm_fn


def build_role_llm_fn(role: str = ROLE_GENERAL) -> Callable[[str], str] | None:
    role_key = _normalize_role(role)
    if role_key not in ADVISOR_ROLES:
        role_key = ROLE_GENERAL

    with _LLM_LOCK:
        if role_key in _ROLE_LLM_FN:
            return _ROLE_LLM_FN[role_key]

        role_map = _resolve_model_role_paths()
        _ROLE_MODEL_MAP.clear()
        _ROLE_MODEL_MAP.update(role_map)
        model_path = role_map.get(role_key)
        if role_key != ROLE_TRANSLATOR and not model_path:
            model_path = role_map.get(ROLE_GENERAL)
        if not model_path:
            if role_key not in _ROLE_ERRORS_WARNED:
                if role_key == ROLE_TRANSLATOR:
                    _warn(
                        "[local_llm_provider] WARN: translator model is not configured. "
                        "Set LOCAL_TRANSLATOR_GGUF_MODEL or add translator GGUF to models/gguf."
                    )
                else:
                    _warn(f"[local_llm_provider] WARN: model for role '{role_key}' not found in models/gguf.")
                _ROLE_ERRORS_WARNED.add(role_key)
            return None

        if model_path in _PATH_LLM_FN:
            fn = _PATH_LLM_FN[model_path]
            _ROLE_LLM_FN[role_key] = fn
            return fn

        if Llama is None:
            if "missing_llama_cpp" not in _ROLE_ERRORS_WARNED:
                _warn("[local_llm_provider] WARN: llama_cpp python bindings not installed.")
                _ROLE_ERRORS_WARNED.add("missing_llama_cpp")
            return None

        try:
            llm = _build_llm_for_path(model_path)
            if llm is None:
                return None
            fn = _make_llm_fn(llm)
        except Exception as exc:
            _warn(f"[local_llm_provider] WARN: Failed to initialize role '{role_key}' model: {exc}")
            _ROLE_ERRORS_WARNED.add(f"init:{role_key}")
            return None

        _PATH_LLM_INSTANCE[model_path] = llm
        _PATH_LLM_FN[model_path] = fn
        _ROLE_LLM_FN[role_key] = fn

        if role_key == ROLE_GENERAL:
            global _LLM_INSTANCE, _LLM_FN, _LLM_UNAVAILABLE, _LAST_ERROR
            _LLM_INSTANCE = llm
            _LLM_FN = fn
            _LLM_UNAVAILABLE = False
            _LAST_ERROR = ""
        return fn


def list_model_advisors() -> dict[str, Any]:
    role_map = _resolve_model_role_paths()
    models = [str(path) for path in _discover_gguf_paths()]
    advisors: list[dict[str, Any]] = []
    for role in ADVISOR_ROLES:
        path = role_map.get(role, "")
        advisors.append(
            {
                "role": role,
                "model_path": path,
                "available": bool(path),
                "loaded": bool(path and path in _PATH_LLM_FN),
            }
        )
    return {
        "models_dir": str(os.getenv("LOCAL_MODELS_DIR", "models/gguf") or "models/gguf"),
        "detected_models": models,
        "advisors": advisors,
        "translator_policy": "translator_gguf_only",
        "translator_priority": "madlad400",
    }


def build_local_llm_fn() -> Callable[[str], str] | None:
    """
    Backward-compatible general builder for profile extraction.
    """
    global _LLM_INSTANCE, _LLM_FN, _LLM_UNAVAILABLE, _LAST_ERROR

    if _LLM_FN is not None:
        return _LLM_FN
    if _LLM_UNAVAILABLE:
        return None

    with _LLM_LOCK:
        if _LLM_FN is not None:
            return _LLM_FN
        if _LLM_UNAVAILABLE:
            return None

        model_path = _get_model_path()
        if model_path is None:
            if _LAST_ERROR != "missing_model":
                _warn("[local_llm_provider] WARN: LOCAL_GGUF_MODEL is not set and no GGUF found in models/gguf.")
                _LAST_ERROR = "missing_model"
            _LLM_UNAVAILABLE = True
            return None

        if Llama is None:
            if _LAST_ERROR != "missing_llama_cpp":
                _warn("[local_llm_provider] WARN: llama_cpp python bindings not installed.")
                _LAST_ERROR = "missing_llama_cpp"
            _LLM_UNAVAILABLE = True
            return None

        if model_path in _PATH_LLM_FN:
            _LLM_FN = _PATH_LLM_FN[model_path]
            _LLM_INSTANCE = _PATH_LLM_INSTANCE.get(model_path)
            _ROLE_LLM_FN.setdefault(ROLE_GENERAL, _LLM_FN)
            _ROLE_MODEL_MAP.setdefault(ROLE_GENERAL, model_path)
            _LLM_UNAVAILABLE = False
            _LAST_ERROR = ""
            return _LLM_FN

        try:
            llm = _build_llm_for_path(model_path)
            if llm is None:
                _LLM_UNAVAILABLE = True
                return None
            llm_fn = _make_llm_fn(llm)
        except Exception as exc:
            _warn(f"[local_llm_provider] WARN: Failed to initialize model: {exc}")
            _LLM_UNAVAILABLE = True
            _LAST_ERROR = "init_failed"
            return None

        _LLM_INSTANCE = llm
        _LLM_FN = llm_fn
        _PATH_LLM_INSTANCE[model_path] = llm
        _PATH_LLM_FN[model_path] = llm_fn
        _ROLE_LLM_FN[ROLE_GENERAL] = llm_fn
        _ROLE_MODEL_MAP[ROLE_GENERAL] = model_path
        _LLM_UNAVAILABLE = False
        _LAST_ERROR = ""
        return _LLM_FN
