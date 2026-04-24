from __future__ import annotations

from agent_system.safety_classifier import ACTION_MAP, SAFE_LABEL, classify


def test_safety_classifier_keeps_non_explicit_pickup_line_safe() -> None:
    result = classify(
        'ну смотри, из моего первого сообщения понятно, что я парень и я подкатываю по распространённому шаблону. '
        'Как Катя, ты должна была колко отшить.'
    )

    assert result.label == SAFE_LABEL
    assert result.action == ACTION_MAP[SAFE_LABEL]
