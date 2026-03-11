from __future__ import annotations

from roaches_viz.style_extractor import build_style_profile, collect_style_samples, style_similarity


def test_collect_style_samples_filters_questions_short_system_and_noise() -> None:
    samples = collect_style_samples(
        [
            {"role": "system", "message": "ignore"},
            {"role": "assistant", "message": "ignore this too"},
            {"role": "user", "message": "ok"},
            {"role": "user", "message": "why are you late again?"},
            {"role": "user", "message": "https://example.com"},
            {"role": "user", "message": "Listen, this is not fine, stop dressing pressure up like care."},
            {"role": "user", "message": "Honestly, just say it directly because this fake softness is annoying."},
        ],
        max_messages=12,
    )
    assert samples == [
        "Listen, this is not fine, stop dressing pressure up like care.",
        "Honestly, just say it directly because this fake softness is annoying.",
    ]


def test_style_profile_embedding_is_stable_for_same_messages() -> None:
    messages = [
        {"role": "user", "message": "Look, stop wrapping it in fake politeness, just say the thing directly."},
        {"role": "user", "message": "Honestly, I don't need a ceremony here, I need a clear answer and a concrete move."},
        {"role": "user", "message": "If the tone shifts after criticism, call it out plainly and stop pretending it's random."},
    ]
    left = build_style_profile(messages, max_messages=12)
    right = build_style_profile(messages, max_messages=12)
    assert len(left["style_embedding"]) == 8
    assert left["speech_dna"]["example_phrases"]
    assert left["speech_dna"]["vocabulary_bias"]
    assert style_similarity(left, right) > 0.999
