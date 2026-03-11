from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .style_extractor import build_style_profile


def default_style_profiles_dir(base_dir: Path | None = None) -> Path:
    env_root = os.environ.get("STYLE_PROFILES_DIR", "").strip()
    root = Path(env_root).resolve() if env_root else (base_dir or Path(__file__).resolve().parents[2]).resolve()
    path = root / "style_profiles"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_user_id(user_id: str) -> str:
    token = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(user_id or "default_user"))
    return token.strip("_") or "default_user"


def style_profile_path(user_id: str, *, base_dir: Path | None = None) -> Path:
    return default_style_profiles_dir(base_dir) / f"{_safe_user_id(user_id)}.json"


def load_style_profile(user_id: str, *, base_dir: Path | None = None) -> dict[str, Any] | None:
    path = style_profile_path(user_id, base_dir=base_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def save_style_profile(user_id: str, profile: dict[str, Any], *, base_dir: Path | None = None) -> dict[str, Any]:
    path = style_profile_path(user_id, base_dir=base_dir)
    payload = dict(profile)
    payload["user_id"] = _safe_user_id(user_id)
    payload["last_updated"] = datetime.now(timezone.utc).isoformat()
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)
    return payload


def learn_style_profile(
    user_id: str,
    messages: list[dict[str, Any]] | list[str],
    *,
    learn_style_button: bool,
    max_messages: int = 12,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    existing = load_style_profile(user_id, base_dir=base_dir) or {}
    if not learn_style_button:
        return {
            "ok": True,
            "learned": False,
            "user_id": _safe_user_id(user_id),
            "profile": existing or None,
            "reason": "trigger_not_pressed",
        }

    learned = build_style_profile(messages, max_messages=max_messages)
    merged_examples: list[str] = []
    for item in list(existing.get("style_examples") or []) + list(learned.get("style_examples") or []):
        text = str(item or "").strip()
        if text and text not in merged_examples:
            merged_examples.append(text)

    profile = {
        "style_embedding": learned.get("style_embedding", []),
        "speech_dna": learned.get("speech_dna", {}),
        "style_examples": merged_examples[:6],
        "features": learned.get("features", {}),
        "sample_count": int(learned.get("sample_count") or 0),
    }
    saved = save_style_profile(user_id, profile, base_dir=base_dir)
    return {"ok": True, "learned": True, "user_id": saved["user_id"], "profile": saved}
