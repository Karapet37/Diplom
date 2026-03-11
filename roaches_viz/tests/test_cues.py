from __future__ import annotations

from roaches_viz.interpret.cues import extract_cues


def test_extract_cues_detects_small_detail_markers() -> None:
    text = 'Yeah right, this is "great"... sure 🙄?? but maybe'
    result = extract_cues(text)
    kinds = [cue["kind"] for cue in result["cues"]]
    assert "sarcasm_marker" in kinds
    assert "irony_quote" in kinds
    assert "ellipsis" in kinds
    assert "emoji_sarcasm" in kinds
    assert "repeated_question" in kinds
    assert "hedge_marker" in kinds
    assert result["summary"]["cue_count"] == len(result["cues"])


def test_extract_cues_is_deterministic() -> None:
    text = "TOTALLY fine?! maybe... no, actually yes."
    first = extract_cues(text)
    second = extract_cues(text)
    assert first == second
