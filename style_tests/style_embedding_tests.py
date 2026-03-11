from __future__ import annotations

from style_tests.style_metrics import SYNTHETIC_USERS, build_synthetic_profiles, extract_style_features


def test_style_feature_extraction_matches_expected_patterns() -> None:
    formal = extract_style_features(SYNTHETIC_USERS["formal_analyst"]["messages"])
    blunt = extract_style_features(SYNTHETIC_USERS["street_blunt"]["messages"])
    playful = extract_style_features(SYNTHETIC_USERS["playful_emoji"]["messages"])
    soft = extract_style_features(SYNTHETIC_USERS["soft_indirect"]["messages"])
    aggressive = extract_style_features(SYNTHETIC_USERS["aggressive_profane"]["messages"])

    assert float(formal["formality"]) > float(blunt["formality"])
    assert float(blunt["directness"]) > float(soft["directness"])
    assert float(playful["humor_level"]) > float(formal["humor_level"])
    assert float(playful["emoji_usage"]) > 0.0
    assert float(aggressive["profanity_tolerance"]) > float(soft["profanity_tolerance"])


def test_style_embedding_is_numeric_stable_and_separable() -> None:
    profiles = build_synthetic_profiles()
    formal = profiles["formal_analyst"]
    blunt = profiles["street_blunt"]
    playful = profiles["playful_emoji"]

    assert len(formal["style_embedding"]) == 8
    assert all(isinstance(value, float) for value in formal["style_embedding"])
    assert formal["style_embedding"] == build_synthetic_profiles()["formal_analyst"]["style_embedding"]

    formal_vs_blunt = sum(abs(a - b) for a, b in zip(formal["style_embedding"], blunt["style_embedding"]))
    formal_vs_playful = sum(abs(a - b) for a, b in zip(formal["style_embedding"], playful["style_embedding"]))
    assert formal_vs_blunt > 0.15
    assert formal_vs_playful > 0.15
