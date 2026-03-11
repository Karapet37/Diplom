from __future__ import annotations

from typing import Any

from ..interpret.hypotheses import normalized_entropy
from .actions import apply_action, available_actions
from .utility import goal_weights, normalize_state, score_state


def _normalize_hypotheses(hypotheses: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not hypotheses:
        return [{"id": "literal", "probability": 1.0}]
    cleaned: list[dict[str, Any]] = []
    for raw in hypotheses:
        if not isinstance(raw, dict):
            continue
        hid = str(raw.get("id") or "").strip().lower()
        if not hid:
            continue
        p = float(raw.get("probability", raw.get("prob", 0.0)))
        if p < 0:
            p = 0.0
        cleaned.append({"id": hid, "probability": p})
    if not cleaned:
        return [{"id": "literal", "probability": 1.0}]
    total = sum(item["probability"] for item in cleaned)
    if total <= 0:
        uniform = 1.0 / len(cleaned)
        return [{"id": item["id"], "probability": uniform} for item in cleaned]
    normalized = [{"id": item["id"], "probability": item["probability"] / total} for item in cleaned]
    return sorted(normalized, key=lambda item: (-item["probability"], item["id"]))


def _state_view(state: dict[str, Any]) -> dict[str, float]:
    return {
        "clarity": round(float(state.get("clarity", 0.5)), 6),
        "alignment": round(float(state.get("alignment", 0.5)), 6),
        "progress": round(float(state.get("progress", 0.5)), 6),
        "rapport": round(float(state.get("rapport", 0.5)), 6),
        "risk": round(float(state.get("risk", 0.5)), 6),
    }


def _select_actions_for_depth(actions: list[dict[str, Any]], beam_width: int) -> list[dict[str, Any]]:
    if beam_width <= 0:
        return actions
    return actions[: min(len(actions), beam_width)]


def _decision_node(
    *,
    state: dict[str, Any],
    goal: str,
    mode: str,
    depth_remaining: int,
    hypotheses: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    beam_width: int,
    uncertainty: float,
    counter: dict[str, int],
) -> dict[str, Any]:
    counter["visited"] = int(counter.get("visited", 0)) + 1
    if depth_remaining <= 0:
        leaf_score = score_state(state, goal, uncertainty=uncertainty, mode=mode)
        return {
            "score": leaf_score,
            "best_line": [],
            "trace": {
                "type": "leaf",
                "score": leaf_score,
                "state": _state_view(state),
                "depth_remaining": depth_remaining,
            },
            "ranked": [],
        }

    selected_actions = _select_actions_for_depth(actions, beam_width)
    ranked: list[dict[str, Any]] = []
    trace_children: list[dict[str, Any]] = []

    for action in selected_actions:
        expected_score = 0.0
        chance_nodes: list[dict[str, Any]] = []
        representative: dict[str, Any] | None = None

        for hypothesis in hypotheses:
            prob = float(hypothesis["probability"])
            hyp_id = str(hypothesis["id"])
            next_state, transition = apply_action(state, action, hypothesis_id=hyp_id)
            child = _decision_node(
                state=next_state,
                goal=goal,
                mode=mode,
                depth_remaining=depth_remaining - 1,
                hypotheses=hypotheses,
                actions=actions,
                beam_width=beam_width,
                uncertainty=uncertainty,
                counter=counter,
            )
            child_score = float(child["score"])
            expected_score += prob * child_score
            chance_node = {
                "type": "chance",
                "hypothesis_id": hyp_id,
                "probability": round(prob, 6),
                "transition": transition,
                "child_score": round(child_score, 6),
                "child": child["trace"],
            }
            chance_nodes.append(chance_node)
            if representative is None or prob > float(representative["probability"]):
                representative = {
                    "probability": prob,
                    "line": child["best_line"],
                }

        action_score = round(expected_score, 6)
        candidate_line = [str(action["id"])] + list((representative or {}).get("line", []))
        ranked.append(
            {
                "action_id": str(action["id"]),
                "action_label": str(action.get("label") or action["id"]),
                "score": action_score,
                "line": candidate_line,
                "chance": chance_nodes,
            }
        )
        trace_children.append(
            {
                "type": "action",
                "action_id": str(action["id"]),
                "label": str(action.get("label") or action["id"]),
                "expected_score": action_score,
                "chance": chance_nodes,
            }
        )

    ranked = sorted(ranked, key=lambda item: (-float(item["score"]), str(item["action_id"])))
    best = ranked[0]
    trace = {
        "type": "decision",
        "depth_remaining": depth_remaining,
        "state": _state_view(state),
        "score": best["score"],
        "best_action": best["action_id"],
        "children": trace_children,
    }
    return {
        "score": best["score"],
        "best_line": best["line"],
        "trace": trace,
        "ranked": ranked,
    }


def plan_expectimax(
    state: dict[str, Any],
    goal: str,
    depth: int = 2,
    *,
    mode: str = "build",
    hypotheses: list[dict[str, Any]] | None = None,
    beam_width: int = 4,
    actions_override: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved_depth = max(1, int(depth))
    resolved_beam = max(1, int(beam_width))
    mode_name = str(mode or "").strip().lower() or "build"
    normalized_state = normalize_state(state)
    normalized_hypotheses = _normalize_hypotheses(hypotheses)
    uncertainty = normalized_entropy([float(h["probability"]) for h in normalized_hypotheses])
    action_space = available_actions(normalized_state, custom_actions=actions_override, mode=mode_name)
    if not action_space:
        return {
            "best_line": [],
            "alternatives": [],
            "trace": {"type": "empty", "children": []},
            "assumptions": {
                "goal": goal,
                "mode": mode_name,
                "depth": resolved_depth,
                "beam_width": resolved_beam,
                "hypotheses": normalized_hypotheses,
                "uncertainty": round(uncertainty, 6),
                "goal_weights": goal_weights(goal, mode=mode_name),
            },
            "nodes_visited": 0,
        }

    counter: dict[str, int] = {"visited": 0}
    root = _decision_node(
        state=normalized_state,
        goal=goal,
        mode=mode_name,
        depth_remaining=resolved_depth,
        hypotheses=normalized_hypotheses,
        actions=action_space,
        beam_width=resolved_beam,
        uncertainty=uncertainty,
        counter=counter,
    )

    ranked = root["ranked"]
    alt_count = min(3, max(0, len(ranked) - 1))
    alternatives = [
        {
            "line": item["line"],
            "expected_score": item["score"],
            "first_action": item["action_id"],
        }
        for item in ranked[1 : 1 + alt_count]
    ]

    return {
        "best_line": root["best_line"],
        "best_score": root["score"],
        "alternatives": alternatives,
        "trace": root["trace"],
        "assumptions": {
            "goal": goal,
            "mode": mode_name,
            "depth": resolved_depth,
            "beam_width": resolved_beam,
            "hypotheses": normalized_hypotheses,
            "uncertainty": round(uncertainty, 6),
            "goal_weights": goal_weights(goal, mode=mode_name),
        },
        "nodes_visited": int(counter["visited"]),
    }
