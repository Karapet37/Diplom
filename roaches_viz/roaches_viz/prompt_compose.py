from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROMPT_DIR = Path(__file__).resolve().parent / "prompt_blocks"


@lru_cache(maxsize=1)
def load_prompt_core() -> dict[str, Any]:
    return yaml.safe_load((PROMPT_DIR / "core.yaml").read_text(encoding="utf-8")) or {}


@lru_cache(maxsize=1)
def load_prompt_modes() -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load((PROMPT_DIR / "modes.yaml").read_text(encoding="utf-8")) or {}
    return {str(key): dict(value or {}) for key, value in raw.items()}


def list_prompt_modes() -> list[str]:
    return sorted(load_prompt_modes())


def _compose_payload(*, mode: str, task: str) -> dict[str, Any]:
    task_text = str(task or "").strip()
    if not task_text:
        raise ValueError("task must not be empty")
    mode_name = str(mode or "").strip() or "build"
    modes = load_prompt_modes()
    if mode_name not in modes:
        raise ValueError(f"unknown mode '{mode_name}', expected one of: {', '.join(list_prompt_modes())}")
    core = load_prompt_core()
    mode_payload = dict(modes[mode_name] or {})
    system_lines = [str(item).strip() for item in list(core.get("system") or []) if str(item).strip()]
    mode_lines = [str(item).strip() for item in list(mode_payload.get("instructions") or []) if str(item).strip()]
    sections = [
        "SYSTEM:",
        *system_lines,
        "",
        f"MODE: {mode_name}",
        *mode_lines,
        "",
        f"TASK: {task_text}",
    ]
    return {
        "mode": mode_name,
        "task": task_text,
        "available_modes": list_prompt_modes(),
        "prompt": "\n".join(sections).strip(),
        "applied_overlays": [],
    }


def compose_prompt(*, mode: str, task: str, db_path: Path | None = None) -> str:
    _ = db_path
    return str(_compose_payload(mode=mode, task=task)["prompt"])


def compose_prompt_payload(*, mode: str, task: str, db_path: Path | None = None) -> dict[str, Any]:
    _ = db_path
    return _compose_payload(mode=mode, task=task)
