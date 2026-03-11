from __future__ import annotations

from roaches_viz.human_analysis import analyze_human_context
from roaches_viz.interpret.cues import extract_cues


def test_human_analysis_flags_attribution_and_hearsay_risk() -> None:
    text = (
        'My boss said I am "unreliable," but yesterday he moved the deadline after the meeting, '
        "and a coworker told me HR already heard a different version."
    )
    cues = extract_cues(text)
    analysis = analyze_human_context(
        text,
        cues=cues["cues"],
        top_hypotheses=[{"id": "ambiguous", "label": "Ambiguous", "probability": 0.5}],
        mode="legal",
    )
    assert analysis["person_analysis"]["attribution_risk"]["score"] > 0
    assert analysis["legal_analysis"]["reported_speech"]
    assert "employment" in analysis["legal_analysis"]["issue_spots"]
    assert "money_or_contract" not in analysis["legal_analysis"]["issue_spots"]
    assert analysis["analysis_adjustments"]["planning_state_delta"]["risk"] >= 0


def test_human_analysis_is_deterministic() -> None:
    text = "Maybe he is rude, or maybe I am missing context because I only heard this second-hand."
    cues = extract_cues(text)
    first = analyze_human_context(text, cues=cues["cues"], top_hypotheses=[], mode="psychology")
    second = analyze_human_context(text, cues=cues["cues"], top_hypotheses=[], mode="psychology")
    assert first == second
