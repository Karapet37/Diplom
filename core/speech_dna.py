from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SpeechDNA:
    style_embedding: tuple[float, ...] = ()
    typical_phrases: tuple[str, ...] = ()
    vocabulary_patterns: tuple[str, ...] = ()
    punctuation_profile: Mapping[str, float] = field(default_factory=dict)
    sentence_rhythm: tuple[float, ...] = ()

    @classmethod
    def from_style_profile(cls, style_profile: Mapping[str, Any] | None) -> "SpeechDNA":
        if not style_profile:
            return cls()
        speech_dna = dict(style_profile.get("speech_dna") or {})
        examples = list(style_profile.get("style_examples") or [])
        return cls(
            style_embedding=tuple(float(item) for item in list(style_profile.get("style_embedding") or [])),
            typical_phrases=tuple(str(item).strip() for item in (speech_dna.get("example_phrases") or examples) if str(item).strip())[:6],
            vocabulary_patterns=tuple(str(item).strip() for item in list(speech_dna.get("vocabulary_bias") or []) if str(item).strip())[:12],
            punctuation_profile={str(key): float(value) for key, value in dict(speech_dna.get("punctuation_profile") or {}).items()},
            sentence_rhythm=tuple(float(item) for item in list(speech_dna.get("sentence_rhythm") or [])[:8]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "style_embedding": list(self.style_embedding),
            "typical_phrases": list(self.typical_phrases),
            "vocabulary_patterns": list(self.vocabulary_patterns),
            "punctuation_profile": dict(self.punctuation_profile),
            "sentence_rhythm": list(self.sentence_rhythm),
        }
