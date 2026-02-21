"""Lightweight .env loader used by local CLI/UI entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
import re

ENV_ASSIGN_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")


def _strip_unquoted_comment(value: str) -> str:
    in_quote = ""
    escaped = False
    out: list[str] = []
    for ch in value:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_quote:
            out.append(ch)
            escaped = True
            continue
        if ch in {"'", '"'}:
            if not in_quote:
                in_quote = ch
            elif in_quote == ch:
                in_quote = ""
            out.append(ch)
            continue
        if ch == "#" and not in_quote:
            break
        out.append(ch)
    return "".join(out).strip()


def _normalize_value(raw: str) -> str:
    value = _strip_unquoted_comment(raw)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.strip()


def load_local_env(path: str | Path = ".env", *, override: bool = False) -> int:
    """
    Load KEY=VALUE pairs from local .env file into process env.
    Returns number of variables set in current process.
    """
    env_path = Path(path)
    if not env_path.exists():
        return 0

    loaded = 0
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = ENV_ASSIGN_RE.match(line)
        if not match:
            continue
        key, raw_value = match.groups()
        if not override and key in os.environ:
            continue
        os.environ[key] = _normalize_value(raw_value)
        loaded += 1
    return loaded
