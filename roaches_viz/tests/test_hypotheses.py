from __future__ import annotations

from roaches_viz.interpret.updater import interpret_text


def test_hypothesis_update_deterministic_and_normalized() -> None:
    text = 'Sure, "great" plan 🙄 ... maybe?'
    first = interpret_text(text, k=3)
    second = interpret_text(text, k=3)
    assert first == second

    top = first["top_hypotheses"]
    assert len(top) == 3
    probs = [float(item["probability"]) for item in first["all_hypotheses"]]
    assert abs(sum(probs) - 1.0) < 1e-5
    assert 0.0 <= float(first["uncertainty"]) <= 1.0
    assert first["best_clarifying_question"]["question"]
    assert first["best_clarifying_question"]["expected_information_gain"] > 0


def test_hypothesis_responds_to_literal_cues() -> None:
    text = "Data shows 42 results because we measured every case."
    result = interpret_text(text, k=3)
    top = result["top_hypotheses"][0]["id"]
    assert top == "literal"


def test_mode_changes_interpretation_bias() -> None:
    text = 'Sure, "great" plan 🙄 but maybe.'
    build = interpret_text(text, k=3, mode="build")
    review = interpret_text(text, k=3, mode="review")
    build_conflicted = next(item for item in build["all_hypotheses"] if item["id"] == "conflicted")
    review_conflicted = next(item for item in review["all_hypotheses"] if item["id"] == "conflicted")
    assert build["mode"] == "build"
    assert review["mode"] == "review"
    assert review_conflicted["probability"] > build_conflicted["probability"]


def test_interpretation_returns_human_and_legal_analysis() -> None:
    text = (
        'My boss said I am "unreliable," but yesterday he changed the deadline after the meeting, '
        "and a coworker told me HR already heard a different version."
    )
    result = interpret_text(text, k=3, mode="legal")
    assert result["mode"] == "legal"
    assert "analysis" in result
    assert result["analysis"]["legal_analysis"]["not_legal_advice"] is True
    assert result["analysis"]["person_analysis"]["attribution_risk"]["score"] >= 0
