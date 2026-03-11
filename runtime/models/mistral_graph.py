from __future__ import annotations

from typing import Any

from roaches_viz.roaches_viz.graph_rag import materialize_graph_update_requests


def build_graph_material(
    graph_payload: dict[str, Any],
    *,
    query: str,
    assistant_reply: str,
    requests: list[str],
    person_id: str | None = None,
) -> dict[str, Any]:
    return materialize_graph_update_requests(
        graph_payload,
        query=query,
        assistant_reply=assistant_reply,
        requests=requests,
        person_id=person_id,
        llm_role="general",
    )

