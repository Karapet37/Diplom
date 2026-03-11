from __future__ import annotations

from roaches_viz.planner.expectimax import plan_expectimax


def test_planner_is_deterministic_for_same_input() -> None:
    state = {"clarity": 0.4, "alignment": 0.45, "progress": 0.35, "rapport": 0.5, "risk": 0.55}
    hypotheses = [
        {"id": "sarcastic", "probability": 0.5},
        {"id": "ambiguous", "probability": 0.3},
        {"id": "literal", "probability": 0.2},
    ]
    first = plan_expectimax(state, "clarify intent and reduce risk", depth=3, hypotheses=hypotheses, beam_width=4)
    second = plan_expectimax(state, "clarify intent and reduce risk", depth=3, hypotheses=hypotheses, beam_width=4)
    assert first == second


def test_planner_returns_best_and_alternatives_with_trace() -> None:
    result = plan_expectimax(
        {"clarity": 0.45, "alignment": 0.4, "progress": 0.32, "rapport": 0.55, "risk": 0.58},
        "clarify and decide safely",
        depth=3,
        hypotheses=[
            {"id": "conflicted", "probability": 0.45},
            {"id": "ambiguous", "probability": 0.35},
            {"id": "literal", "probability": 0.2},
        ],
        beam_width=5,
    )
    assert result["best_line"]
    assert len(result["alternatives"]) >= 2
    assert len(result["alternatives"]) <= 3
    assert result["trace"]["type"] == "decision"
    assert result["nodes_visited"] > 0
    assert "goal_weights" in result["assumptions"]


def test_planner_mode_changes_assumptions_and_score_profile() -> None:
    state = {"clarity": 0.45, "alignment": 0.4, "progress": 0.32, "rapport": 0.55, "risk": 0.58}
    hypotheses = [
        {"id": "conflicted", "probability": 0.45},
        {"id": "ambiguous", "probability": 0.35},
        {"id": "literal", "probability": 0.2},
    ]
    build = plan_expectimax(state, "clarify and decide safely", depth=2, mode="build", hypotheses=hypotheses, beam_width=4)
    review = plan_expectimax(state, "clarify and decide safely", depth=2, mode="review", hypotheses=hypotheses, beam_width=4)
    assert build["assumptions"]["mode"] == "build"
    assert review["assumptions"]["mode"] == "review"
    assert build["assumptions"]["goal_weights"] != review["assumptions"]["goal_weights"]
