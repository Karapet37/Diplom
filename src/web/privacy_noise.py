"""Experimental synthetic telemetry generator.

Disabled by default and isolated from core request flow.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import time
from typing import Any


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "1" if default else "0").strip().lower()
    return value not in {"0", "false", "no", "off", ""}


@dataclass(frozen=True)
class PrivacyNoiseConfig:
    enabled: bool = False
    intensity: float = 0.05
    seed: int = 20260217

    @classmethod
    def from_env(cls) -> "PrivacyNoiseConfig":
        intensity_raw = os.getenv("PRIVACY_NOISE_INTENSITY", "0.05").strip()
        try:
            intensity = float(intensity_raw)
        except Exception:
            intensity = 0.05
        intensity = max(0.0, min(1.0, intensity))
        seed_raw = os.getenv("PRIVACY_NOISE_SEED", "20260217").strip()
        try:
            seed = int(seed_raw)
        except Exception:
            seed = 20260217
        return cls(
            enabled=_bool_env("PRIVACY_NOISE_ENABLE", False),
            intensity=intensity,
            seed=seed,
        )


class PrivacyNoisePlugin:
    """Generates synthetic metrics/reports for privacy masking experiments."""

    def __init__(self, config: PrivacyNoiseConfig):
        self.config = config
        self._random = random.Random(int(config.seed))

    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def synthetic_metrics(self) -> dict[str, float]:
        if not self.enabled():
            return {}
        # Experimental synthetic counters to blend telemetry patterns.
        scale = max(0.0001, float(self.config.intensity))
        return {
            "autograph_privacy_synthetic_events_total": round(self._random.uniform(10.0, 250.0) * scale, 6),
            "autograph_privacy_synthetic_errors_total": round(self._random.uniform(0.0, 25.0) * scale, 6),
            "autograph_privacy_synthetic_latency_seconds": round(self._random.uniform(0.05, 1.5) * scale, 6),
        }

    def report(self) -> dict[str, Any]:
        if not self.enabled():
            return {
                "enabled": False,
                "message": "privacy noise plugin is disabled",
                "intensity": float(self.config.intensity),
            }
        metrics = self.synthetic_metrics()
        return {
            "enabled": True,
            "intensity": float(self.config.intensity),
            "generated_at": time.time(),
            "metrics": metrics,
        }
