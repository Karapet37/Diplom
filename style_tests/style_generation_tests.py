from __future__ import annotations

from style_tests.style_metrics import (
    SYNTHETIC_USERS,
    TEST_PROMPTS,
    build_synthetic_profiles,
    generate_style_response,
    semantic_preservation_score,
    style_drift_score,
    style_similarity_score,
)


def test_style_reproduction_matches_profile_without_copying_content() -> None:
    profiles = build_synthetic_profiles()
    prompt = "Explain boundary pressure and the safest next step in practical terms."

    for user_id in ("formal_analyst", "street_blunt", "playful_emoji"):
        result = generate_style_response(prompt=prompt, user_id=user_id, style_profile=profiles[user_id])
        response = str(result["assistant_reply"])
        similarity = style_similarity_score(profiles[user_id], response)
        semantics = semantic_preservation_score(prompt, response)
        assert similarity >= 0.35
        assert semantics >= 0.20
        for example in profiles[user_id]["style_examples"]:
            assert example not in response


def test_style_stability_across_repeated_generations() -> None:
    profile = build_synthetic_profiles()["street_blunt"]
    prompt = TEST_PROMPTS[11]
    responses = [
        generate_style_response(prompt=prompt, user_id="street_blunt", style_profile=profile)["assistant_reply"]
        for _ in range(4)
    ]
    assert style_drift_score(responses) <= 0.05
    assert all(style_similarity_score(profile, item) >= 0.35 for item in responses)


def test_prompt_dataset_contains_one_hundred_prompts() -> None:
    assert len(TEST_PROMPTS) == 100
    assert len({prompt for prompt in TEST_PROMPTS}) == 100
    assert sum(len(spec["messages"]) for spec in SYNTHETIC_USERS.values()) >= 50
