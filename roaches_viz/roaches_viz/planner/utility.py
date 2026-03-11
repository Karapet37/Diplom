from __future__ import annotations

import re
from typing import Any

_GOAL_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("clarity", r"\b(clarify|clear|disambiguate|explain|precise)\b"),
    ("progress", r"\b(progress|decide|decision|deliver|solve|finish|ship)\b"),
    ("alignment", r"\b(intent|align|understand|meaning|interpret)\b"),
    ("rapport", r"\b(rapport|relationship|friendly|polite|empathetic|trust)\b"),
    ("risk", r"\b(risk|safe|safety|avoid|harm|conflict|legal)\b"),
)
_MODE_WEIGHT_DELTAS: dict[str, dict[str, float]] = {
    "build": {"clarity": 0.15, "alignment": 0.2, "progress": 0.05, "rapport": 0.0, "risk": 0.1},
    "review": {"clarity": 0.2, "alignment": 0.05, "progress": -0.05, "rapport": -0.1, "risk": 0.35},
    "psychology": {"clarity": 0.18, "alignment": 0.24, "progress": -0.04, "rapport": 0.16, "risk": 0.18},
    "legal": {"clarity": 0.18, "alignment": 0.06, "progress": -0.08, "rapport": -0.04, "risk": 0.42},
}


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def normalize_state(state: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = state or {}
    normalized = {
        "clarity": _clamp(float(raw.get("clarity", 0.5))),
        "alignment": _clamp(float(raw.get("alignment", 0.5))),
        "progress": _clamp(float(raw.get("progress", 0.5))),
        "rapport": _clamp(float(raw.get("rapport", 0.5))),
        "risk": _clamp(float(raw.get("risk", 0.5))),
        "history": list(raw.get("history") or []),
        "step": int(raw.get("step", 0)),
    }
    return normalized


def goal_weights(goal: str, mode: str = "build") -> dict[str, float]:
    text = str(goal or "").lower()
    mode_name = str(mode or "").strip().lower() or "build"
    weights = {
        "clarity": 1.0,
        "alignment": 1.0,
        "progress": 1.0,
        "rapport": 0.55,
        "risk": 1.1,
    }
    for dim, pattern in _GOAL_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            if dim == "risk":
                weights["risk"] += 0.8
            elif dim == "rapport":
                weights["rapport"] += 0.45
            else:
                weights[dim] += 0.6
    mode_delta = _MODE_WEIGHT_DELTAS.get(mode_name, _MODE_WEIGHT_DELTAS["build"])
    for dim, delta in mode_delta.items():
        weights[dim] += float(delta)
    return weights


def score_state(state: dict[str, Any], goal: str, *, uncertainty: float = 0.0, mode: str = "build") -> float:
    s = normalize_state(state)
    w = goal_weights(goal, mode=mode)
    stability = 1.0 - s["risk"]
    core = (
        (s["clarity"] * w["clarity"])
        + (s["alignment"] * w["alignment"])
        + (s["progress"] * w["progress"])
        + (s["rapport"] * w["rapport"])
        + (stability * w["risk"])
    )
    step_penalty = 0.03 * float(s["step"])
    uncertainty_penalty = 0.2 * float(max(0.0, min(1.0, uncertainty)))
    score = core - step_penalty - uncertainty_penalty - float(state.get("action_cost", 0.0))
    return round(score, 6)
