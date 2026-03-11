from __future__ import annotations

from style_tests.style_metrics import (
    build_synthetic_profiles,
    generate_style_response,
    style_similarity_score,
)


def test_multiple_user_profiles_remain_isolated() -> None:
    profiles = build_synthetic_profiles()
    prompt = "Explain the pressure pattern in practical terms. Focus on family conflict."

    response_a = generate_style_response(prompt=prompt, user_id="formal_analyst", style_profile=profiles["formal_analyst"])["assistant_reply"]
    response_b = generate_style_response(prompt=prompt, user_id="playful_emoji", style_profile=profiles["playful_emoji"])["assistant_reply"]

    own_a = style_similarity_score(profiles["formal_analyst"], response_a)
    own_b = style_similarity_score(profiles["playful_emoji"], response_b)

    assert own_a >= 0.70
    assert own_b >= 0.60
    assert response_a != response_b
    assert response_a.startswith("From my point of view,")
    assert "haha" in response_b.lower() or ":)" in response_b


def test_user_a_style_does_not_pollute_user_b_profile() -> None:
    profiles = build_synthetic_profiles()
    assert profiles["formal_analyst"]["style_embedding"] != profiles["playful_emoji"]["style_embedding"]
    assert profiles["street_blunt"]["style_examples"] != profiles["soft_indirect"]["style_examples"]
