from __future__ import annotations

from typing import Any


_DEFAULT_ACTIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "ask_clarifying_question",
        "label": "Ask clarifying question",
        "effects": {"clarity": 0.28, "alignment": 0.18, "progress": 0.08, "rapport": 0.05, "risk": -0.06},
        "cost": 0.01,
    },
    {
        "id": "provide_direct_answer",
        "label": "Provide direct answer",
        "effects": {"clarity": 0.08, "alignment": 0.04, "progress": 0.26, "rapport": -0.03, "risk": 0.14},
        "cost": 0.02,
    },
    {
        "id": "gather_more_context",
        "label": "Gather more context",
        "effects": {"clarity": 0.18, "alignment": 0.22, "progress": 0.1, "rapport": 0.03, "risk": -0.04},
        "cost": 0.01,
    },
    {
        "id": "summarize_options",
        "label": "Summarize options",
        "effects": {"clarity": 0.21, "alignment": 0.16, "progress": 0.14, "rapport": 0.02, "risk": -0.05},
        "cost": 0.01,
    },
    {
        "id": "challenge_assumption",
        "label": "Challenge assumption",
        "effects": {"clarity": 0.2, "alignment": 0.08, "progress": 0.11, "rapport": -0.14, "risk": 0.2},
        "cost": 0.03,
    },
    {
        "id": "separate_fact_from_inference",
        "label": "Separate fact from inference",
        "effects": {"clarity": 0.24, "alignment": 0.18, "progress": 0.09, "rapport": 0.01, "risk": -0.14},
        "cost": 0.01,
    },
    {
        "id": "ask_for_primary_source",
        "label": "Ask for primary source",
        "effects": {"clarity": 0.22, "alignment": 0.14, "progress": 0.06, "rapport": -0.01, "risk": -0.18},
        "cost": 0.01,
    },
    {
        "id": "map_situation_constraints",
        "label": "Map situation constraints",
        "effects": {"clarity": 0.16, "alignment": 0.24, "progress": 0.06, "rapport": 0.04, "risk": -0.12},
        "cost": 0.01,
    },
    {
        "id": "hedge_response",
        "label": "Hedge response",
        "effects": {"clarity": -0.06, "alignment": 0.05, "progress": 0.04, "rapport": 0.1, "risk": -0.13},
        "cost": 0.01,
    },
)

_DIM_MULTIPLIERS: dict[str, dict[str, float]] = {
    "literal": {"clarity": 1.0, "alignment": 1.0, "progress": 1.08, "rapport": 1.0, "risk": 0.9},
    "sarcastic": {"clarity": 0.66, "alignment": 0.72, "progress": 0.82, "rapport": 0.8, "risk": 1.28},
    "ambiguous": {"clarity": 0.86, "alignment": 0.94, "progress": 0.82, "rapport": 0.95, "risk": 1.06},
    "conflicted": {"clarity": 0.76, "alignment": 0.78, "progress": 0.72, "rapport": 0.84, "risk": 1.34},
}


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _normalize_action(raw: dict[str, Any]) -> dict[str, Any] | None:
    action_id = str(raw.get("id") or "").strip().lower().replace(" ", "_")
    if not action_id:
        return None
    label = str(raw.get("label") or action_id).strip()
    effects_raw = raw.get("effects") if isinstance(raw.get("effects"), dict) else {}
    effects = {
        "clarity": float(effects_raw.get("clarity", 0.0)),
        "alignment": float(effects_raw.get("alignment", 0.0)),
        "progress": float(effects_raw.get("progress", 0.0)),
        "rapport": float(effects_raw.get("rapport", 0.0)),
        "risk": float(effects_raw.get("risk", 0.0)),
    }
    return {"id": action_id, "label": label, "effects": effects, "cost": float(raw.get("cost", 0.0))}


def available_actions(
    state: dict[str, Any],
    custom_actions: list[dict[str, Any]] | None = None,
    *,
    mode: str = "build",
) -> list[dict[str, Any]]:
    _ = state
    base = custom_actions if custom_actions else list(_DEFAULT_ACTIONS)
    normalized = [_normalize_action(item) for item in base if isinstance(item, dict)]
    cleaned = [item for item in normalized if item]
    mode_name = str(mode or "").strip().lower() or "build"
    if mode_name == "review":
        for item in cleaned:
            if item["id"] == "challenge_assumption":
                item["effects"]["clarity"] += 0.08
                item["effects"]["alignment"] += 0.04
                item["effects"]["risk"] += 0.03
            if item["id"] == "provide_direct_answer":
                item["effects"]["progress"] -= 0.06
            if item["id"] == "ask_clarifying_question":
                item["effects"]["clarity"] += 0.04
    elif mode_name == "legal":
        for item in cleaned:
            if item["id"] == "ask_for_primary_source":
                item["effects"]["clarity"] += 0.08
                item["effects"]["risk"] -= 0.04
            if item["id"] == "separate_fact_from_inference":
                item["effects"]["clarity"] += 0.06
                item["effects"]["alignment"] += 0.03
            if item["id"] == "provide_direct_answer":
                item["effects"]["progress"] -= 0.08
                item["effects"]["risk"] += 0.02
    elif mode_name == "psychology":
        for item in cleaned:
            if item["id"] == "map_situation_constraints":
                item["effects"]["alignment"] += 0.07
                item["effects"]["rapport"] += 0.04
            if item["id"] == "gather_more_context":
                item["effects"]["alignment"] += 0.05
            if item["id"] == "challenge_assumption":
                item["effects"]["rapport"] += 0.03
                item["effects"]["risk"] -= 0.02
    else:
        for item in cleaned:
            if item["id"] == "summarize_options":
                item["effects"]["clarity"] += 0.04
                item["effects"]["alignment"] += 0.03
    return sorted(cleaned, key=lambda action: str(action["id"]))


def apply_action(
    state: dict[str, Any],
    action: dict[str, Any],
    *,
    hypothesis_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    multipliers = _DIM_MULTIPLIERS.get(hypothesis_id, _DIM_MULTIPLIERS["ambiguous"])
    features = {
        "clarity": float(state.get("clarity", 0.5)),
        "alignment": float(state.get("alignment", 0.5)),
        "progress": float(state.get("progress", 0.5)),
        "rapport": float(state.get("rapport", 0.5)),
        "risk": float(state.get("risk", 0.5)),
    }
    deltas: dict[str, float] = {}
    for dim in ("clarity", "alignment", "progress", "rapport", "risk"):
        raw_delta = float(action["effects"].get(dim, 0.0))
        delta = raw_delta * float(multipliers.get(dim, 1.0))
        deltas[dim] = round(delta, 6)

    action_id = str(action["id"])
    if action_id == "ask_clarifying_question" and hypothesis_id in {"sarcastic", "ambiguous", "conflicted"}:
        deltas["clarity"] = round(deltas["clarity"] + 0.1, 6)
        deltas["risk"] = round(deltas["risk"] - 0.04, 6)
    if action_id == "provide_direct_answer" and hypothesis_id in {"sarcastic", "conflicted"}:
        deltas["risk"] = round(deltas["risk"] + 0.1, 6)
        deltas["progress"] = round(deltas["progress"] - 0.1, 6)
    if action_id == "hedge_response" and hypothesis_id == "literal":
        deltas["progress"] = round(deltas["progress"] - 0.08, 6)
        deltas["clarity"] = round(deltas["clarity"] - 0.05, 6)
    if action_id == "challenge_assumption" and hypothesis_id == "literal":
        deltas["clarity"] = round(deltas["clarity"] + 0.05, 6)
    if action_id == "challenge_assumption" and hypothesis_id in {"sarcastic", "conflicted"}:
        deltas["risk"] = round(deltas["risk"] + 0.08, 6)
        deltas["rapport"] = round(deltas["rapport"] - 0.07, 6)
    if action_id == "ask_for_primary_source" and hypothesis_id in {"ambiguous", "conflicted"}:
        deltas["clarity"] = round(deltas["clarity"] + 0.05, 6)
        deltas["risk"] = round(deltas["risk"] - 0.05, 6)
    if action_id == "separate_fact_from_inference" and hypothesis_id in {"ambiguous", "conflicted", "sarcastic"}:
        deltas["clarity"] = round(deltas["clarity"] + 0.05, 6)
        deltas["alignment"] = round(deltas["alignment"] + 0.04, 6)
    if action_id == "map_situation_constraints" and hypothesis_id in {"conflicted", "sarcastic"}:
        deltas["alignment"] = round(deltas["alignment"] + 0.05, 6)
        deltas["rapport"] = round(deltas["rapport"] + 0.04, 6)
        deltas["risk"] = round(deltas["risk"] - 0.04, 6)

    next_state = dict(state)
    for dim in ("clarity", "alignment", "progress", "rapport", "risk"):
        next_state[dim] = round(_clamp(features[dim] + deltas[dim]), 6)
    history = list(state.get("history") or [])
    history.append(action_id)
    next_state["history"] = history
    next_state["step"] = int(state.get("step", 0)) + 1
    next_state["last_action"] = action_id
    next_state["action_cost"] = float(action.get("cost", 0.0))
    transition = {
        "action_id": action_id,
        "hypothesis_id": hypothesis_id,
        "deltas": deltas,
        "result_state": {k: next_state[k] for k in ("clarity", "alignment", "progress", "rapport", "risk")},
    }
    return next_state, transition
