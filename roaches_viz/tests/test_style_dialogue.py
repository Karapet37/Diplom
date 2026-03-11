from __future__ import annotations

from pathlib import Path

import roaches_viz.graph_rag as graph_rag
from roaches_viz.foundations import load_builtin_foundation, seed_foundation
from roaches_viz.store import GraphStore


def test_dialogue_applies_style_profile_without_copying_content(tmp_path: Path, monkeypatch) -> None:
    store = GraphStore(tmp_path / "graph.sqlite3")
    try:
        seed_foundation(store, load_builtin_foundation("human_foundations"), replace_graph=True)
        payload = store.export_graph()
    finally:
        store.close()

    monkeypatch.setattr(graph_rag, "build_reasoning_llm", lambda *_args, **_kwargs: None)
    style_profile = {
        "sample_count": 3,
        "style_embedding": [0.55, 0.64, 0.22, 0.48, 0.31, 0.44, 0.81, 0.25],
        "style_examples": [
            "Look, stop dressing pressure up like care.",
            "Honestly, say the thing directly and stop circling it.",
        ],
        "features": {
            "sentence_length": 11.0,
            "slang_level": 0.52,
            "formality": 0.24,
            "aggressiveness": 0.34,
            "humor_level": 0.12,
            "punctuation_style": 0.31,
            "directness": 0.78,
            "profanity_tolerance": 0.18,
        },
        "speech_dna": {
            "vocabulary_bias": ["look", "honestly", "directly"],
            "punctuation_profile": {"!": 0.22, ".": 0.12},
            "sentence_rhythm": [1.0, 0.9, 1.1],
        },
    }

    result = graph_rag.generate_behavioral_dialogue(
        payload,
        query="Explain the difference between action and inaction using the graph.",
        recent_history=["Keep it practical."],
        llm_role="general",
        user_id="user_style",
        style_profile=style_profile,
    )
    assert result["style"]["applied"] is True
    assert result["style"]["user_id"] == "user_style"
    assert result["assistant_reply"]
    assert result["assistant_reply"].startswith(("Look,", "Honestly,"))
    assert "example message 1" not in result["assistant_reply"].lower()
