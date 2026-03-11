from __future__ import annotations

from typing import Any

from roaches_viz.roaches_viz.graph_rag import build_graph_update_scenario, generate_behavioral_dialogue


def generate_fast_response(
    graph_payload: dict[str, Any],
    *,
    query: str,
    context: str = "",
    person_id: str | None = None,
    user_id: str | None = None,
    style_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return generate_behavioral_dialogue(
        graph_payload,
        query="\n\n".join(part for part in [query, context] if str(part or "").strip()).strip(),
        recent_history=[query, context] if str(context or "").strip() else [query],
        person_id=person_id,
        llm_role="analyst",
        user_id=user_id,
        style_profile=style_profile,
        fast_only=True,
    )


def plan_graph_scenario(
    graph_payload: dict[str, Any],
    *,
    query: str,
    assistant_reply: str,
    person_id: str | None = None,
) -> dict[str, Any]:
    return build_graph_update_scenario(
        graph_payload,
        query=query,
        assistant_reply=assistant_reply,
        person_id=person_id,
    )

