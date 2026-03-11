from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    db_path: Path
    top_tokens_per_sentence: int = 5
    hypothesis_k: int = 3


def default_settings(base_dir: Path | None = None) -> Settings:
    root = (base_dir or Path(__file__).resolve().parents[1]).resolve()
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(db_path=data_dir / "graph.sqlite3")
