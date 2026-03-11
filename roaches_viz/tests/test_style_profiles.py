from __future__ import annotations

from pathlib import Path

from roaches_viz.style_profiles import learn_style_profile, load_style_profile, style_profile_path


def test_learn_style_requires_explicit_trigger(tmp_path: Path) -> None:
    result = learn_style_profile(
        "user_one",
        [{"role": "user", "message": "Look, this is not a drill, be direct and stop hedging."}],
        learn_style_button=False,
        base_dir=tmp_path,
    )
    assert result["ok"] is True
    assert result["learned"] is False
    assert result["reason"] == "trigger_not_pressed"
    assert load_style_profile("user_one", base_dir=tmp_path) is None
    assert not style_profile_path("user_one", base_dir=tmp_path).exists()


def test_learn_style_updates_profile_and_appends_examples(tmp_path: Path) -> None:
    first = learn_style_profile(
        "user_two",
        [
            {"role": "user", "message": "Honestly, stop softening it, just say what's wrong and what changes now."},
            {"role": "user", "message": "Look, I want the short version first, then the details if they matter."},
        ],
        learn_style_button=True,
        base_dir=tmp_path,
    )
    second = learn_style_profile(
        "user_two",
        [
            {"role": "user", "message": "If the pattern repeats after criticism, call it what it is instead of dancing around it."},
            {"role": "user", "message": "Keep it direct, practical, and a little sharp if needed."},
        ],
        learn_style_button=True,
        base_dir=tmp_path,
    )
    assert first["learned"] is True
    assert second["learned"] is True
    profile = load_style_profile("user_two", base_dir=tmp_path)
    assert profile is not None
    assert profile["sample_count"] >= 1
    assert len(profile["style_embedding"]) == 8
    assert len(profile["style_examples"]) >= 3
    assert profile["last_updated"]
