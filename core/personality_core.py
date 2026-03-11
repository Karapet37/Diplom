from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpeechStyleCore:
    formality: float = 0.5
    slang_level: float = 0.3
    directness: float = 0.5
    profanity_tolerance: float = 0.1

    def clamp(self) -> "SpeechStyleCore":
        return SpeechStyleCore(
            formality=min(max(float(self.formality), 0.0), 1.0),
            slang_level=min(max(float(self.slang_level), 0.0), 1.0),
            directness=min(max(float(self.directness), 0.0), 1.0),
            profanity_tolerance=min(max(float(self.profanity_tolerance), 0.0), 1.0),
        )


@dataclass(frozen=True)
class PersonalityCore:
    temperament: str
    values: tuple[str, ...] = ()
    speech_style: SpeechStyleCore = field(default_factory=SpeechStyleCore)
    reasoning_style: str = "grounded"
    risk_tolerance: float = 0.5
    aggression_level: float = 0.2
    humor_level: float = 0.2

    def gradual_update(
        self,
        *,
        temperament: str | None = None,
        values: tuple[str, ...] | None = None,
        speech_style: SpeechStyleCore | None = None,
        reasoning_style: str | None = None,
        risk_tolerance: float | None = None,
        aggression_level: float | None = None,
        humor_level: float | None = None,
        max_step: float = 0.08,
    ) -> "PersonalityCore":
        def _slow(current: float, target: float | None) -> float:
            if target is None:
                return current
            target = min(max(float(target), 0.0), 1.0)
            if abs(target - current) <= max_step:
                return target
            return current + max_step if target > current else current - max_step

        return PersonalityCore(
            temperament=self.temperament if temperament is None else str(temperament).strip() or self.temperament,
            values=self.values if values is None else tuple(values),
            speech_style=self.speech_style if speech_style is None else speech_style.clamp(),
            reasoning_style=self.reasoning_style if reasoning_style is None else str(reasoning_style).strip() or self.reasoning_style,
            risk_tolerance=_slow(float(self.risk_tolerance), risk_tolerance),
            aggression_level=_slow(float(self.aggression_level), aggression_level),
            humor_level=_slow(float(self.humor_level), humor_level),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "temperament": self.temperament,
            "values": list(self.values),
            "speech_style": {
                "formality": self.speech_style.formality,
                "slang_level": self.speech_style.slang_level,
                "directness": self.speech_style.directness,
                "profanity_tolerance": self.speech_style.profanity_tolerance,
            },
            "reasoning_style": self.reasoning_style,
            "risk_tolerance": self.risk_tolerance,
            "aggression_level": self.aggression_level,
            "humor_level": self.humor_level,
        }
