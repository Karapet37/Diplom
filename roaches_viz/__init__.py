from __future__ import annotations

from pathlib import Path

_inner_dir = Path(__file__).resolve().parent / "roaches_viz"
if _inner_dir.is_dir():
    __path__.append(str(_inner_dir))  # type: ignore[name-defined]

