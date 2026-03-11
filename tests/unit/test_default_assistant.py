from __future__ import annotations

from runtime.assistant.default_assistant import DefaultAssistant


def test_default_assistant_requests_clarification_for_short_query() -> None:
    assistant = DefaultAssistant()
    decision = assistant.assess(
        query="why this?",
        context="",
        lookup={"ranked_nodes": [], "missing_tokens": ["this", "why"]},
        apply_to_graph=True,
    )
    assert decision.clarification_needed is True
    assert decision.requires_graph_update is False
    assert decision.reason == "clarification_needed"


def test_default_assistant_skips_graph_update_when_verified_context_is_sufficient() -> None:
    assistant = DefaultAssistant()
    decision = assistant.assess(
        query="Is this guilt pressure after criticism?",
        context="family conflict",
        lookup={
            "ranked_nodes": [{"node_id": "pattern:pressure", "score": 3.2}],
            "missing_tokens": [],
        },
        apply_to_graph=True,
    )
    assert decision.clarification_needed is False
    assert decision.requires_graph_update is False
    assert decision.reason == "verified_graph_sufficient"

