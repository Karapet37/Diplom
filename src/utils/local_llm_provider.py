"""Local GGUF LLM provider with role-based model selection."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import os
from pathlib import Path
import re
import struct
import sys
from typing import Any, Callable

from agent_system.runtime_config import get_runtime_config

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from threading import Lock

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE  = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE_TAG = "</think>"


_THINKING_PROCESS_RE = re.compile(
    r'^(thinking process:|process of thinking:|размышление:|ход мыслей:)',
    re.IGNORECASE | re.MULTILINE,
)


def _strip_think_blocks(text: str) -> str:
    """
    Strip internal reasoning from thinking models.
    Handles:
    1. <think>reasoning</think>answer  — full tags
    2. reasoning</think>answer         — chat-template hides opening tag
    3. <think>reasoning...             — truncated (no closing tag)
    4. "Thinking Process:\n..." header without tags — model narrates reasoning
    """
    # Case 1: full <think>...</think> blocks
    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    # Case 2: chat template hides <think>, output is: reasoning\n</think>\n\nAnswer
    if _THINK_CLOSE_TAG.lower() in cleaned.lower():
        idx = cleaned.lower().rfind(_THINK_CLOSE_TAG.lower())
        after = cleaned[idx + len(_THINK_CLOSE_TAG):].strip()
        if after:
            return after
        cleaned = ""
    # Case 3: truncated — strip everything from opening <think>
    if not cleaned:
        cleaned = _THINK_OPEN_RE.sub("", text).strip()
    # Case 4: model wrote "Thinking Process: ..." header — find first non-header line
    if cleaned and _THINKING_PROCESS_RE.search(cleaned):
        lines = cleaned.splitlines()
        in_header = False
        result_lines: list[str] = []
        for line in lines:
            if _THINKING_PROCESS_RE.match(line.strip()):
                in_header = True
                continue
            if in_header and not line.strip():
                continue
            in_header = False
            result_lines.append(line)
        candidate = "\n".join(result_lines).strip()
        if candidate:
            cleaned = candidate
    return cleaned


from src.utils.prompt_budgeter import (
    MAX_REASONING_N_CTX,
    MAX_ROUTER_N_CTX,
    MIN_REASONING_N_CTX,
    MIN_ROUTER_N_CTX,
    SAFE_ERROR_REPLY,
    retry_infer,
)
from src.utils.token_budget import select_n_ctx

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
_GGUF_ARCH_CACHE: dict[str, str] = {}

_LLM_LOCK = Lock()

ROLE_GENERAL = "general"
ROLE_UNCENSORED = "uncensored"
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
    ROLE_UNCENSORED,
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
    ROLE_UNCENSORED: "LOCAL_UNCENSORED_GGUF_MODEL",
    ROLE_TRANSLATOR: "LOCAL_TRANSLATOR_GGUF_MODEL",
    ROLE_ANALYST: "LOCAL_ANALYST_GGUF_MODEL",
    ROLE_CREATIVE: "LOCAL_CREATIVE_GGUF_MODEL",
    ROLE_PLANNER: "LOCAL_PLANNER_GGUF_MODEL",
    ROLE_CODER_ARCHITECT: "LOCAL_CODER_ARCHITECT_GGUF_MODEL",
    ROLE_CODER_REVIEWER: "LOCAL_CODER_REVIEWER_GGUF_MODEL",
    ROLE_CODER_REFACTOR: "LOCAL_CODER_REFACTOR_GGUF_MODEL",
    ROLE_CODER_DEBUG: "LOCAL_CODER_DEBUG_GGUF_MODEL",
}

_ROLE_N_CTX_ENV_MAP: dict[str, str] = {
    ROLE_GENERAL: "LOCAL_GGUF_N_CTX",
    ROLE_UNCENSORED: "LOCAL_UNCENSORED_N_CTX",
    ROLE_TRANSLATOR: "LOCAL_TRANSLATOR_N_CTX",
    ROLE_ANALYST: "LOCAL_ANALYST_N_CTX",
    ROLE_CREATIVE: "LOCAL_CREATIVE_N_CTX",
    ROLE_PLANNER: "LOCAL_PLANNER_N_CTX",
    ROLE_CODER_ARCHITECT: "LOCAL_CODER_ARCHITECT_N_CTX",
    ROLE_CODER_REVIEWER: "LOCAL_CODER_REVIEWER_N_CTX",
    ROLE_CODER_REFACTOR: "LOCAL_CODER_REFACTOR_N_CTX",
    ROLE_CODER_DEBUG: "LOCAL_CODER_DEBUG_N_CTX",
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
_FAST_MODEL_HINTS: tuple[str, ...] = (
    "nanbeige4.1-3b.q3_k_m.gguf",
    "nanbeige4.1-3b",
)
_UNCENSORED_MODEL_HINTS: tuple[str, ...] = (
    "mistral-nemo-2407-12b-thinking-claude-gemini-gpt5.2-uncensored-heretic_q3_k_m.gguf",
    "uncensored-heretic",
    "mistral-nemo-2407-12b",
)
_JSON_PROMPT_HINTS: tuple[str, ...] = (
    "return valid json only",
    "schema:",
    '"entities":[',
    '"persona_payload":{',
    '"node_improvement":{',
)
_CHAT_QUESTION_LABEL = "User question:"
_CHAT_SECTION_LABELS: tuple[str, ...] = (
    "Persona head:",
    "Interaction role:",
    "Mood research:",
    "Knowledge graph:",
    "Recent dialogue:",
)


def _allow_uncensored_autodiscovery() -> bool:
    token = str(os.getenv("LOCAL_ALLOW_UNCENSORED_AUTODISCOVERY", "") or "").strip().lower()
    return token in {"1", "true", "yes", "on"}


def _warn(message: str) -> None:
    silent = str(os.getenv("LOCAL_LLM_SILENT_ERRORS", "0")).strip().lower()
    if silent in {"1", "true", "yes", "on"}:
        return
    print(message, file=sys.stderr)


def _version_tuple(raw: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in re.findall(r"\d+", str(raw or "")):
        try:
            parts.append(int(token))
        except Exception:
            continue
    return tuple(parts)


def _llama_cpp_version() -> str:
    try:
        return str(importlib_metadata.version("llama-cpp-python") or "").strip()
    except Exception:
        return ""


def _llama_cpp_version_tuple() -> tuple[int, ...]:
    return _version_tuple(_llama_cpp_version())


_GGUF_SCALAR_TYPE_SIZES: dict[int, int] = {
    0: 1,   # uint8
    1: 1,   # int8
    2: 2,   # uint16
    3: 2,   # int16
    4: 4,   # uint32
    5: 4,   # int32
    6: 4,   # float32
    7: 1,   # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}


def _read_gguf_string(handle: Any) -> str:
    size_raw = handle.read(8)
    if len(size_raw) != 8:
        raise ValueError("invalid gguf string length")
    size = struct.unpack("<Q", size_raw)[0]
    payload = handle.read(size)
    if len(payload) != size:
        raise ValueError("invalid gguf string payload")
    return payload.decode("utf-8", errors="replace")


def _skip_gguf_value(handle: Any, value_type: int) -> None:
    if value_type == 8:  # string
        _read_gguf_string(handle)
        return
    if value_type == 9:  # array
        element_type_raw = handle.read(4)
        count_raw = handle.read(8)
        if len(element_type_raw) != 4 or len(count_raw) != 8:
            raise ValueError("invalid gguf array header")
        element_type = struct.unpack("<I", element_type_raw)[0]
        count = struct.unpack("<Q", count_raw)[0]
        for _ in range(count):
            _skip_gguf_value(handle, element_type)
        return
    size = _GGUF_SCALAR_TYPE_SIZES.get(int(value_type))
    if size is None:
        raise ValueError(f"unsupported gguf value type: {value_type}")
    payload = handle.read(size)
    if len(payload) != size:
        raise ValueError("invalid gguf scalar payload")


def _read_gguf_architecture(model_path: str) -> str:
    normalized_path = str(model_path or "").strip()
    if not normalized_path:
        return ""
    cached = _GGUF_ARCH_CACHE.get(normalized_path)
    if cached is not None:
        return cached
    architecture = ""
    try:
        with Path(normalized_path).open("rb") as handle:
            if handle.read(4) != b"GGUF":
                _GGUF_ARCH_CACHE[normalized_path] = ""
                return ""
            version_raw = handle.read(4)
            if len(version_raw) != 4:
                _GGUF_ARCH_CACHE[normalized_path] = ""
                return ""
            version = struct.unpack("<I", version_raw)[0]
            if version not in {2, 3}:
                _GGUF_ARCH_CACHE[normalized_path] = ""
                return ""
            tensor_count_raw = handle.read(8)
            kv_count_raw = handle.read(8)
            if len(tensor_count_raw) != 8 or len(kv_count_raw) != 8:
                _GGUF_ARCH_CACHE[normalized_path] = ""
                return ""
            kv_count = struct.unpack("<Q", kv_count_raw)[0]
            for _ in range(kv_count):
                key = _read_gguf_string(handle)
                value_type_raw = handle.read(4)
                if len(value_type_raw) != 4:
                    break
                value_type = struct.unpack("<I", value_type_raw)[0]
                if key == "general.architecture" and value_type == 8:
                    architecture = _read_gguf_string(handle).strip()
                    break
                _skip_gguf_value(handle, value_type)
    except Exception:
        architecture = ""
    _GGUF_ARCH_CACHE[normalized_path] = architecture
    return architecture


def _preflight_model_block_reason(model_path: str) -> str:
    normalized_path = str(model_path or "").strip()
    if not normalized_path:
        return "empty_model_path"
    architecture = _read_gguf_architecture(normalized_path)
    llama_version = _llama_cpp_version_tuple()
    if architecture == "qwen35" and llama_version and llama_version < (0, 3, 17):
        version_text = _llama_cpp_version() or "unknown"
        return (
            f"GGUF architecture '{architecture}' is not supported by llama-cpp-python {version_text}. "
            "Upgrade llama-cpp-python / llama.cpp to a build that supports qwen35."
        )
    return ""


def _normalize_role(role: str) -> str:
    token = re.sub(r"[^a-z0-9_]+", "_", str(role or "").strip().lower())
    aliases = {
        "analysis": ROLE_ANALYST,
        "reasoning": ROLE_GENERAL,
    }
    return aliases.get(token, token or ROLE_GENERAL)


def _contains_any(token: str, hints: tuple[str, ...]) -> bool:
    source = str(token or "")
    return any(hint in source for hint in hints)


def _path_token(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").lower()


def _resolve_runtime_path(raw: str) -> Path:
    path = Path(str(raw or '').strip()).expanduser()
    if not path.is_absolute():
        path = get_runtime_config().paths.repo_root / path
    return path.resolve()


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
        path = _resolve_runtime_path(raw)
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


def _score_analyst(path: Path) -> float:
    token = _path_token(path)
    score = _score_general(path)
    if _contains_any(token, _FAST_MODEL_HINTS):
        score += 8.0
    if _contains_any(token, ("nanbeige", "3b", "fast")):
        score += 2.5
    return score


def _score_uncensored(path: Path) -> float:
    token = _path_token(path)
    score = _score_general(path)
    if _contains_any(token, _UNCENSORED_MODEL_HINTS):
        score += 9.0
    if _contains_any(token, ("uncensored", "nemo", "heretic")):
        score += 3.0
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
        path = _resolve_runtime_path(explicit)
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
        pick = _select_best(analyst_candidates or general_candidates, scorer=_score_analyst)
        if pick is not None:
            role_map[ROLE_ANALYST] = str(pick)
    if ROLE_UNCENSORED not in role_map and _allow_uncensored_autodiscovery():
        pick = _select_best(general_candidates, scorer=_score_uncensored)
        if pick is not None:
            role_map[ROLE_UNCENSORED] = str(pick)
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
        path = _resolve_runtime_path(explicit)
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


def _llm_cache_key(model_path: str, *, n_ctx: int, max_tokens: int) -> str:
    return f"{model_path}::ctx={int(n_ctx)}::max={int(max_tokens)}"


def _resolve_n_ctx(role: str | None = None, explicit_n_ctx: int | None = None) -> int:
    role_key = _normalize_role(role or ROLE_GENERAL)
    allowed = _allowed_n_ctx_list_for_role(role_key)
    if explicit_n_ctx is not None:
        return select_n_ctx(int(explicit_n_ctx), allowed)
    role_env = _ROLE_N_CTX_ENV_MAP.get(role_key, "LOCAL_GGUF_N_CTX")
    value = os.getenv(role_env)
    if value is None or str(value).strip() == "":
        fallback = os.getenv("LOCAL_GGUF_N_CTX")
        value = fallback if fallback is not None and str(fallback).strip() != "" else "2048"
    return select_n_ctx(int(value), allowed)


def _allowed_n_ctx_list_for_role(role: str | None) -> list[int]:
    role_key = _normalize_role(role or ROLE_GENERAL)
    if role_key == ROLE_ANALYST:
        return [MIN_ROUTER_N_CTX, 1536, MAX_ROUTER_N_CTX, 3072, 4096, MAX_REASONING_N_CTX]
    if role_key in {ROLE_GENERAL, ROLE_CREATIVE, ROLE_TRANSLATOR}:
        return [MIN_ROUTER_N_CTX, 1536, 2048, 3072, 4096, MAX_REASONING_N_CTX]
    return [MIN_REASONING_N_CTX, 3072, 4096, MAX_REASONING_N_CTX]


def _is_model_loaded(model_path: str) -> bool:
    prefix = f"{model_path}::"
    return any(key == model_path or key.startswith(prefix) for key in _PATH_LLM_FN) or any(
        key == model_path or key.startswith(prefix) for key in _ROLE_LLM_FN if "::" in key
    )


def _build_llm_for_path(model_path: str, *, n_ctx: int | None = None) -> "Llama" | None:
    if Llama is None:
        return None
    resolved_n_ctx = max(MIN_ROUTER_N_CTX, min(MAX_REASONING_N_CTX, int(n_ctx))) if n_ctx is not None else _resolve_n_ctx()
    cpu_threads = max(1, int(os.cpu_count() or 4))
    configured_threads = max(1, int(os.getenv("LOCAL_GGUF_THREADS", str(cpu_threads))))
    configured_batch_threads = max(1, int(os.getenv("LOCAL_GGUF_THREADS_BATCH", str(configured_threads))))
    return Llama(
        model_path=str(model_path),
        n_ctx=resolved_n_ctx,
        temperature=float(os.getenv("LOCAL_GGUF_TEMPERATURE", "0.15")),
        n_batch=int(os.getenv("LOCAL_GGUF_N_BATCH", "256")),
        n_threads=configured_threads,
        n_threads_batch=configured_batch_threads,
        verbose=False,
    )


def _prompt_prefers_json_output(prompt: str) -> bool:
    lowered = str(prompt or "").strip().lower()
    return any(hint in lowered for hint in _JSON_PROMPT_HINTS)


def _split_chat_prompt_messages(prompt: str) -> list[dict[str, str]]:
    raw = str(prompt or "").strip()
    if not raw:
        return [{"role": "user", "content": ""}]
    if _CHAT_QUESTION_LABEL not in raw:
        return [{"role": "user", "content": raw}]
    prefix, remainder = raw.split(_CHAT_QUESTION_LABEL, 1)
    question_block = remainder.strip()
    system_parts = [prefix.strip()]
    for label in _CHAT_SECTION_LABELS:
        if label in question_block:
            question_text, tail = question_block.split(label, 1)
            question_block = question_text.strip()
            system_parts.append(f"{label}\n\n{tail.strip()}".strip())
    user_content = question_block.strip() or raw
    system_content = "\n\n".join(part for part in system_parts if part).strip()
    if system_content:
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
    return [{"role": "user", "content": user_content}]


def _make_raw_llm_fn(llm: "Llama", *, max_tokens: int | None = None) -> Callable[[str], str]:
    resolved_max_tokens = int(max_tokens if max_tokens is not None else int(os.getenv("LOCAL_GGUF_MAX_TOKENS", "2048")))

    def llm_fn(prompt: str) -> str:
        metadata: dict[str, Any] = {
            "finish_reason": "",
            "completion_tokens": 0,
            "prompt_eval_tokens": 0,
            "total_tokens": 0,
        }
        messages = _split_chat_prompt_messages(prompt)
        # Qwen3 no-think mode: if system prompt starts with /no_think, prefill the
        # assistant turn with </think> so the model skips its reasoning block entirely.
        _first_msg_content = str((messages[0] if messages else {}).get("content") or "")
        if _first_msg_content.lstrip().startswith("/no_think"):
            messages = messages + [{"role": "assistant", "content": "</think>\n"}]
        kwargs = {
            "max_tokens": resolved_max_tokens,
            "temperature": float(os.getenv("LOCAL_GGUF_TEMPERATURE", "0.15")),
            "top_p": float(os.getenv("LOCAL_GGUF_TOP_P", "0.9")),
            "stop": ["\nUser question:"],
        }
        if _prompt_prefers_json_output(prompt):
            try:
                response = llm.create_chat_completion(
                    messages=messages,
                    response_format={"type": "json_object"},
                    **kwargs,
                )
                choice = dict((response.get("choices") or [{}])[0] or {})
                usage = dict(response.get("usage") or {})
                metadata.update(
                    {
                        "finish_reason": str(choice.get("finish_reason") or "").strip(),
                        "completion_tokens": int(usage.get("completion_tokens") or 0),
                        "prompt_eval_tokens": int(usage.get("prompt_tokens") or 0),
                        "total_tokens": int(usage.get("total_tokens") or 0),
                    }
                )
                result_text = choice.get("message", {}).get("content")
                setattr(llm_fn, "_last_infer_meta", dict(metadata))
                return _strip_think_blocks(str(result_text or ""))
            except Exception:
                pass
        try:
            out = llm.create_chat_completion(messages=messages, **kwargs)
            choice = dict((out.get("choices") or [{}])[0] or {})
            usage = dict(out.get("usage") or {})
            metadata.update(
                {
                    "finish_reason": str(choice.get("finish_reason") or "").strip(),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "prompt_eval_tokens": int(usage.get("prompt_tokens") or 0),
                    "total_tokens": int(usage.get("total_tokens") or 0),
                }
            )
            result_text = choice.get("message", {}).get("content")
        except Exception as inner_exc:
            if len(messages) > 1 and 'system role not supported' in str(inner_exc).lower():
                fallback_messages = [
                    {
                        "role": "user",
                        "content": "\n\n".join(str(item.get("content") or "").strip() for item in messages if str(item.get("content") or "").strip()),
                    }
                ]
                try:
                    out = llm.create_chat_completion(messages=fallback_messages, **kwargs)
                    choice = dict((out.get("choices") or [{}])[0] or {})
                    usage = dict(out.get("usage") or {})
                    metadata.update(
                        {
                            "finish_reason": str(choice.get("finish_reason") or "").strip(),
                            "completion_tokens": int(usage.get("completion_tokens") or 0),
                            "prompt_eval_tokens": int(usage.get("prompt_tokens") or 0),
                            "total_tokens": int(usage.get("total_tokens") or 0),
                        }
                    )
                    result_text = choice.get("message", {}).get("content")
                    setattr(llm_fn, "_last_infer_meta", dict(metadata))
                    return _strip_think_blocks(str(result_text or ""))
                except Exception as fallback_exc:
                    setattr(llm_fn, "_last_infer_meta", dict(metadata))
                    raise RuntimeError(str(fallback_exc)) from fallback_exc
            setattr(llm_fn, "_last_infer_meta", dict(metadata))
            raise RuntimeError(str(inner_exc)) from inner_exc
        setattr(llm_fn, "_last_infer_meta", dict(metadata))
        return _strip_think_blocks(str(result_text or ""))

    setattr(llm_fn, "_llm", llm)
    setattr(llm_fn, "_max_tokens", resolved_max_tokens)
    setattr(llm_fn, "_last_infer_meta", {})
    return llm_fn


def _resolve_max_tokens(explicit_max_tokens: int | None = None) -> int:
    return int(explicit_max_tokens if explicit_max_tokens is not None else int(os.getenv("LOCAL_GGUF_MAX_TOKENS", "2048")))


def _ensure_raw_llm_fn(
    *,
    model_path: str,
    role: str,
    requested_n_ctx: int,
    requested_max_tokens: int,
) -> Callable[[str], str] | None:
    cache_key = _llm_cache_key(model_path, n_ctx=requested_n_ctx, max_tokens=requested_max_tokens)
    with _LLM_LOCK:
        if cache_key in _PATH_LLM_FN:
            return _PATH_LLM_FN[cache_key]
        preflight_block_reason = _preflight_model_block_reason(model_path)
        if preflight_block_reason:
            warned_key = f"incompatible:{role}:{model_path}"
            if warned_key not in _ROLE_ERRORS_WARNED:
                _warn(
                    f"[local_llm_provider] WARN: skipped incompatible model '{model_path}' "
                    f"for role '{role}': {preflight_block_reason}"
                )
                _ROLE_ERRORS_WARNED.add(warned_key)
            return None
        try:
            llm = _build_llm_for_path(model_path, n_ctx=requested_n_ctx)
        except Exception as exc:
            _warn(
                f"[local_llm_provider] WARN: Failed to initialize model '{model_path}' "
                f"for role '{role}' at n_ctx={requested_n_ctx}: {exc}"
            )
            return None
        if llm is None:
            return None
        raw_fn = _make_raw_llm_fn(llm, max_tokens=requested_max_tokens)
        _PATH_LLM_INSTANCE[cache_key] = llm
        _PATH_LLM_FN[cache_key] = raw_fn
        return raw_fn


def _candidate_model_paths_for_role(role_key: str, role_map: dict[str, str]) -> list[str]:
    candidates: list[str] = []

    def _push(path: str) -> None:
        token = str(path or "").strip()
        if token and token not in candidates:
            candidates.append(token)

    _push(role_map.get(role_key, ""))
    if role_key != ROLE_TRANSLATOR:
        if role_key != ROLE_GENERAL:
            _push(role_map.get(ROLE_GENERAL, ""))
        if role_key != ROLE_ANALYST:
            _push(role_map.get(ROLE_ANALYST, ""))
        _push(role_map.get(ROLE_CREATIVE, ""))
        _push(role_map.get(ROLE_PLANNER, ""))
    return candidates


def _select_usable_model_path(role_key: str, role_map: dict[str, str]) -> tuple[str | None, str]:
    candidates = _candidate_model_paths_for_role(role_key, role_map)
    if not candidates:
        return None, ""
    blocked_reason = ""
    for candidate in candidates:
        reason = _preflight_model_block_reason(candidate)
        if not reason:
            return candidate, blocked_reason
        if not blocked_reason:
            blocked_reason = reason
    return None, blocked_reason


def _make_budgeted_llm_fn(
    *,
    role: str,
    model_path: str,
    n_ctx: int,
    max_tokens: int,
) -> Callable[[str], str]:
    allowed_contexts = _allowed_n_ctx_list_for_role(role)

    def _builder(_role: str, requested_n_ctx: int, requested_max_tokens: int) -> Callable[[str], str] | None:
        return _ensure_raw_llm_fn(
            model_path=model_path,
            role=role,
            requested_n_ctx=requested_n_ctx,
            requested_max_tokens=requested_max_tokens,
        )

    def llm_fn(prompt: str) -> str:
        outcome = retry_infer(
            _builder,
            role,
            str(prompt or ""),
            token_budget=n_ctx,
            allowed_n_ctx_range=allowed_contexts,
            max_tokens=max_tokens,
        )
        if not outcome.get("ok"):
            _warn(
                f"[local_llm_provider] WARN: inference fallback used for role '{role}' "
                f"after attempts={outcome.get('attempts')!r}"
            )
        setattr(
            llm_fn,
            "_last_budget_meta",
            {
                "role": role,
                "model_path": model_path,
                **dict(outcome or {}),
            },
        )
        return str(outcome.get("text") or SAFE_ERROR_REPLY)

    setattr(llm_fn, "_model_path", model_path)
    setattr(llm_fn, "_n_ctx", n_ctx)
    setattr(llm_fn, "_max_tokens", max_tokens)
    setattr(llm_fn, "_last_budget_meta", {})
    return llm_fn


def build_role_llm_fn(
    role: str = ROLE_GENERAL,
    *,
    n_ctx: int | None = None,
    max_tokens: int | None = None,
) -> Callable[[str], str] | None:
    role_key = _normalize_role(role)
    if role_key not in ADVISOR_ROLES:
        role_key = ROLE_GENERAL

    with _LLM_LOCK:
        if role_key in _ROLE_LLM_FN and n_ctx is None and max_tokens is None:
            return _ROLE_LLM_FN[role_key]

        role_map = _resolve_model_role_paths()
        _ROLE_MODEL_MAP.clear()
        _ROLE_MODEL_MAP.update(role_map)
        requested_model_path = role_map.get(role_key, "")
        model_path, blocked_reason = _select_usable_model_path(role_key, role_map)
        if not model_path:
            if role_key not in _ROLE_ERRORS_WARNED:
                if blocked_reason:
                    _warn(
                        f"[local_llm_provider] WARN: no compatible model available for role '{role_key}'. "
                        f"{blocked_reason}"
                    )
                elif role_key == ROLE_TRANSLATOR:
                    _warn(
                        "[local_llm_provider] WARN: translator model is not configured. "
                        "Set LOCAL_TRANSLATOR_GGUF_MODEL or add translator GGUF to models/gguf."
                    )
                else:
                    _warn(f"[local_llm_provider] WARN: model for role '{role_key}' not found in models/gguf.")
                _ROLE_ERRORS_WARNED.add(role_key)
            return None
        if requested_model_path and model_path != requested_model_path:
            warned_key = f"fallback_path:{role_key}:{requested_model_path}->{model_path}"
            if warned_key not in _ROLE_ERRORS_WARNED:
                _warn(
                    f"[local_llm_provider] WARN: role '{role_key}' falling back from '{requested_model_path}' "
                    f"to compatible model '{model_path}'."
                )
                _ROLE_ERRORS_WARNED.add(warned_key)
            _ROLE_MODEL_MAP[role_key] = model_path
        if not Path(model_path).exists():
            warned_key = f"missing_path:{role_key}"
            if warned_key not in _ROLE_ERRORS_WARNED:
                _warn(f"[local_llm_provider] WARN: configured model for role '{role_key}' does not exist: {model_path}")
                _ROLE_ERRORS_WARNED.add(warned_key)
            return None

        resolved_n_ctx = _resolve_n_ctx(role_key, n_ctx)
        resolved_max_tokens = _resolve_max_tokens(max_tokens)
        cache_key = _llm_cache_key(model_path, n_ctx=resolved_n_ctx, max_tokens=resolved_max_tokens)

        if cache_key in _ROLE_LLM_FN:
            fn = _ROLE_LLM_FN[cache_key]
            if n_ctx is None and max_tokens is None:
                _ROLE_LLM_FN[role_key] = fn
            return fn

        if Llama is None:
            if "missing_llama_cpp" not in _ROLE_ERRORS_WARNED:
                _warn("[local_llm_provider] WARN: llama_cpp python bindings not installed.")
                _ROLE_ERRORS_WARNED.add("missing_llama_cpp")
            return None

        fn = _make_budgeted_llm_fn(
            role=role_key,
            model_path=model_path,
            n_ctx=resolved_n_ctx,
            max_tokens=resolved_max_tokens,
        )

        _ROLE_LLM_FN[cache_key] = fn
        if n_ctx is None and max_tokens is None:
            _ROLE_LLM_FN[role_key] = fn

        if role_key == ROLE_GENERAL:
            global _LLM_INSTANCE, _LLM_FN, _LLM_UNAVAILABLE, _LAST_ERROR
            _LLM_INSTANCE = None
            _LLM_FN = fn
            _LLM_UNAVAILABLE = False
        _LAST_ERROR = ""
        return fn


def build_model_llm_fn(
    model_path: str,
    *,
    n_ctx: int | None = None,
    max_tokens: int | None = None,
) -> Callable[[str], str] | None:
    """
    Build or reuse an LLM callable for an explicit GGUF model path.

    The path is normalized to split entrypoint when needed, and must resolve to a
    valid candidate model file.
    """
    raw = str(model_path or "").strip()
    if not raw:
        return None
    path = _resolve_runtime_path(raw)
    normalized = _resolve_entrypoint(path)
    if normalized is None:
        if path.exists() and path.is_file():
            _warn(
                f"[local_llm_provider] WARN: explicit model path is not a valid split entrypoint: {path}. "
                "Use shard 00001-of-N for split models."
            )
        return None
    if not _is_candidate_gguf(normalized):
        _warn(f"[local_llm_provider] WARN: explicit model path is not an allowed GGUF candidate: {normalized}")
        return None

    normalized_path = str(normalized)
    with _LLM_LOCK:
        blocked_reason = _preflight_model_block_reason(normalized_path)
        if blocked_reason:
            warned_key = f"explicit_incompatible:{normalized_path}"
            if warned_key not in _ROLE_ERRORS_WARNED:
                _warn(
                    f"[local_llm_provider] WARN: explicit model path is incompatible: {normalized_path}. "
                    f"{blocked_reason}"
                )
                _ROLE_ERRORS_WARNED.add(warned_key)
            return None
        resolved_n_ctx = _resolve_n_ctx(explicit_n_ctx=n_ctx)
        resolved_max_tokens = _resolve_max_tokens(max_tokens)
        cache_key = _llm_cache_key(normalized_path, n_ctx=resolved_n_ctx, max_tokens=resolved_max_tokens)
        if cache_key in _ROLE_LLM_FN:
            return _ROLE_LLM_FN[cache_key]

        if Llama is None:
            if "missing_llama_cpp" not in _ROLE_ERRORS_WARNED:
                _warn("[local_llm_provider] WARN: llama_cpp python bindings not installed.")
                _ROLE_ERRORS_WARNED.add("missing_llama_cpp")
            return None

        fn = _make_budgeted_llm_fn(
            role=ROLE_GENERAL,
            model_path=normalized_path,
            n_ctx=resolved_n_ctx,
            max_tokens=resolved_max_tokens,
        )

        _ROLE_LLM_FN[cache_key] = fn
        return fn


def get_role_llm_instance(role: str = ROLE_GENERAL) -> "Llama | None":
    """Return the raw Llama instance for *role*, loading it if necessary.

    Used by qwen_fast_respond() to call create_completion() directly.
    Returns None if the model is unavailable or llama_cpp is not installed.
    """
    fn = build_role_llm_fn(role)
    if fn is None:
        return None
    model_path = getattr(fn, "_model_path", None)
    if not model_path:
        return None
    prefix = f"{model_path}::"
    with _LLM_LOCK:
        for key, instance in _PATH_LLM_INSTANCE.items():
            if key == model_path or key.startswith(prefix):
                return instance
    # Instance not yet loaded — force-initialize it now (lazy model load
    # only fires when llm_fn is called; we trigger it here explicitly).
    n_ctx = int(getattr(fn, "_n_ctx", None) or 2048)
    max_tokens = int(getattr(fn, "_max_tokens", None) or 320)
    _ensure_raw_llm_fn(
        model_path=model_path,
        role=_normalize_role(role),
        requested_n_ctx=n_ctx,
        requested_max_tokens=max_tokens,
    )
    with _LLM_LOCK:
        for key, instance in _PATH_LLM_INSTANCE.items():
            if key == model_path or key.startswith(prefix):
                return instance
    return None


def list_model_advisors() -> dict[str, Any]:
    role_map = _resolve_model_role_paths()
    models = [str(path) for path in _discover_gguf_paths()]
    advisors: list[dict[str, Any]] = []
    for role in ADVISOR_ROLES:
        path = role_map.get(role, "")
        blocked_reason = _preflight_model_block_reason(path) if path else ""
        advisors.append(
            {
                "role": role,
                "model_path": path,
                "available": bool(path),
                "loaded": bool(path and _is_model_loaded(path)),
                "architecture": _read_gguf_architecture(path) if path else "",
                "compatible": not bool(blocked_reason),
                "compatibility_note": blocked_reason,
            }
        )
    return {
        "models_dir": str(os.getenv("LOCAL_MODELS_DIR", "models/gguf") or "models/gguf"),
        "detected_models": models,
        "advisors": advisors,
        "uncensored_autodiscovery_enabled": _allow_uncensored_autodiscovery(),
        "translator_policy": "optional_if_available",
        "translator_priority": "madlad400",
        "fast_model_priority": "nanbeige4.1-3b.q3_k_m.gguf",
        "uncensored_model_priority": "mistral-nemo-2407-12b-thinking-claude-gemini-gpt5.2-uncensored-heretic_q3_k_m.gguf",
    }


def prewarm_role_model(
    role: str = ROLE_GENERAL,
    *,
    n_ctx: int | None = None,
    max_tokens: int | None = None,
) -> bool:
    role_key = _normalize_role(role)
    if role_key not in ADVISOR_ROLES:
        role_key = ROLE_GENERAL
    role_map = _resolve_model_role_paths()
    requested_model_path = role_map.get(role_key, "")
    model_path, _ = _select_usable_model_path(role_key, role_map)
    if not model_path or not Path(model_path).exists() or Llama is None:
        return False
    if requested_model_path and model_path != requested_model_path:
        warned_key = f"prewarm_fallback:{role_key}:{requested_model_path}->{model_path}"
        if warned_key not in _ROLE_ERRORS_WARNED:
            _warn(
                f"[local_llm_provider] WARN: prewarm for role '{role_key}' is using fallback model '{model_path}' "
                f"instead of incompatible '{requested_model_path}'."
            )
            _ROLE_ERRORS_WARNED.add(warned_key)
    resolved_n_ctx = _resolve_n_ctx(role_key, n_ctx)
    resolved_max_tokens = _resolve_max_tokens(max_tokens)
    raw_fn = _ensure_raw_llm_fn(
        model_path=model_path,
        role=role_key,
        requested_n_ctx=resolved_n_ctx,
        requested_max_tokens=resolved_max_tokens,
    )
    if raw_fn is None:
        return False
    cache_key = _llm_cache_key(model_path, n_ctx=resolved_n_ctx, max_tokens=resolved_max_tokens)
    with _LLM_LOCK:
        if cache_key not in _ROLE_LLM_FN:
            _ROLE_LLM_FN[cache_key] = _make_budgeted_llm_fn(
                role=role_key,
                model_path=model_path,
                n_ctx=resolved_n_ctx,
                max_tokens=resolved_max_tokens,
            )
        if n_ctx is None and max_tokens is None:
            _ROLE_LLM_FN[role_key] = _ROLE_LLM_FN[cache_key]
    return True


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

        resolved_n_ctx = _resolve_n_ctx()
        resolved_max_tokens = _resolve_max_tokens()
        default_cache_key = _llm_cache_key(model_path, n_ctx=resolved_n_ctx, max_tokens=resolved_max_tokens)
        if default_cache_key in _ROLE_LLM_FN:
            _LLM_FN = _ROLE_LLM_FN[default_cache_key]
            _LLM_INSTANCE = None
            _ROLE_LLM_FN.setdefault(ROLE_GENERAL, _LLM_FN)
            _ROLE_MODEL_MAP.setdefault(ROLE_GENERAL, model_path)
            _LLM_UNAVAILABLE = False
            _LAST_ERROR = ""
            return _LLM_FN

        llm_fn = _make_budgeted_llm_fn(
            role=ROLE_GENERAL,
            model_path=model_path,
            n_ctx=resolved_n_ctx,
            max_tokens=resolved_max_tokens,
        )

        _LLM_INSTANCE = None
        _LLM_FN = llm_fn
        _ROLE_LLM_FN[default_cache_key] = llm_fn
        _ROLE_LLM_FN[ROLE_GENERAL] = llm_fn
        _ROLE_MODEL_MAP[ROLE_GENERAL] = model_path
        _LLM_UNAVAILABLE = False
        _LAST_ERROR = ""
        return _LLM_FN
