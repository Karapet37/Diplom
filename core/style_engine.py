from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .speech_dna import SpeechDNA
from roaches_viz.roaches_viz.style_extractor import build_style_profile
from roaches_viz.roaches_viz.style_profiles import load_style_profile, save_style_profile


@dataclass(frozen=True)
class StyleLearningResult:
    ok: bool
    learned: bool
    user_id: str
    profile: Mapping[str, Any] | None
    reason: str = ""


class StyleEngine:
    def __init__(self, *, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir

    def load(self, user_id: str) -> Mapping[str, Any] | None:
        return load_style_profile(user_id, base_dir=self.base_dir)

    def learn(
        self,
        *,
        user_id: str,
        messages: list[dict[str, Any]] | list[str],
        learn_style_button: bool,
        max_messages: int = 12,
    ) -> StyleLearningResult:
        existing = self.load(user_id)
        if not learn_style_button:
            return StyleLearningResult(ok=True, learned=False, user_id=user_id, profile=existing, reason="trigger_not_pressed")

        learned = build_style_profile(messages, max_messages=max_messages)
        merged_examples: list[str] = []
        for item in list((existing or {}).get("style_examples") or []) + list(learned.get("style_examples") or []):
            text = str(item or "").strip()
            if text and text not in merged_examples:
                merged_examples.append(text)

        profile = {
            "style_embedding": learned.get("style_embedding", []),
            "speech_dna": learned.get("speech_dna", {}),
            "style_examples": merged_examples[:6],
            "features": learned.get("features", {}),
            "sample_count": int(learned.get("sample_count") or 0),
            "speech_dna_core": SpeechDNA.from_style_profile(learned).as_dict(),
        }
        saved = save_style_profile(user_id, profile, base_dir=self.base_dir)
        return StyleLearningResult(ok=True, learned=True, user_id=str(saved.get("user_id") or user_id), profile=saved)
