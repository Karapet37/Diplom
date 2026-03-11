from __future__ import annotations

from roaches_viz.micro_signals import extract_micro_signals


def test_extract_micro_signals_captures_fine_grained_layers() -> None:
    result = extract_micro_signals(
        'If you loved me, you would do this, right?',
        concepts=["love", "family", "pressure"],
        patterns=["guilt_pressure"],
        triggers=["criticism"],
    )

    assert result["semantic_meaning"]
    assert result["emotional_signals"]
    assert result["logical_structure"]
    assert result["hidden_intent"]
    assert result["behavior_pattern_candidates"]

    assert any(item["key"] == "guilt_loaded" for item in result["emotional_signals"])
    assert any(item["key"] == "conditional_leverage" for item in result["logical_structure"])
    assert any(item["key"] == "compliance_pressure" for item in result["hidden_intent"])
    assert any(item["key"] == "guilt_pressure" for item in result["behavior_pattern_candidates"])


def test_extract_micro_signals_is_deterministic() -> None:
    sentence = "Maybe you should explain why you ignored me."
    first = extract_micro_signals(sentence, concepts=["explain", "ignored"], patterns=["blame_shifting"], triggers=["rejection"])
    second = extract_micro_signals(sentence, concepts=["explain", "ignored"], patterns=["blame_shifting"], triggers=["rejection"])
    assert first == second
