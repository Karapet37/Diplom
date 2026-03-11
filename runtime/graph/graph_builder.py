from __future__ import annotations

from typing import Any

from runtime.models.mistral_graph import build_graph_material
from runtime.models.nanbeige_router import plan_graph_scenario


def prepare_graph_build(
    graph_payload: dict[str, Any],
    *,
    query: str,
    assistant_reply: str,
    person_id: str | None = None,
) -> dict[str, Any]:
    return plan_graph_scenario(
        graph_payload,
        query=query,
        assistant_reply=assistant_reply,
        person_id=person_id,
    )


def materialize_graph_build(
    graph_payload: dict[str, Any],
    *,
    query: str,
    assistant_reply: str,
    requests: list[str],
    person_id: str | None = None,
) -> dict[str, Any]:
    return build_graph_material(
        graph_payload,
        query=query,
        assistant_reply=assistant_reply,
        requests=requests,
        person_id=person_id,
    )

