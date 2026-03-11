from __future__ import annotations

from pathlib import Path

from roaches_viz.style_profiles import learn_style_profile, load_style_profile

from style_tests.style_metrics import SYNTHETIC_USERS


def test_style_learning_occurs_only_when_button_pressed(tmp_path: Path) -> None:
    messages = [{"role": "user", "message": item} for item in SYNTHETIC_USERS["street_blunt"]["messages"]]
    ignored = learn_style_profile("user_a", messages, learn_style_button=False, base_dir=tmp_path)
    assert ignored["learned"] is False
    assert load_style_profile("user_a", base_dir=tmp_path) is None

    learned = learn_style_profile("user_a", messages, learn_style_button=True, base_dir=tmp_path)
    assert learned["learned"] is True
    assert learned["profile"]["style_embedding"]
    assert load_style_profile("user_a", base_dir=tmp_path) is not None


def test_style_profile_updates_when_learning_retriggered(tmp_path: Path) -> None:
    first_messages = [{"role": "user", "message": item} for item in SYNTHETIC_USERS["formal_analyst"]["messages"][:5]]
    second_messages = [{"role": "user", "message": item} for item in SYNTHETIC_USERS["formal_analyst"]["messages"][5:]]
    first = learn_style_profile("user_b", first_messages, learn_style_button=True, base_dir=tmp_path)
    second = learn_style_profile("user_b", second_messages, learn_style_button=True, base_dir=tmp_path)
    assert first["profile"]["last_updated"] != second["profile"]["last_updated"]
    assert len(second["profile"]["style_examples"]) >= len(first["profile"]["style_examples"])


def test_synthetic_dataset_meets_minimum_size() -> None:
    assert len(SYNTHETIC_USERS) >= 5
    assert sum(len(spec["messages"]) for spec in SYNTHETIC_USERS.values()) >= 50
