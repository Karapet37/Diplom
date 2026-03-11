from __future__ import annotations

from typing import Any

from roaches_viz.roaches_viz.style_profiles import load_style_profile
from runtime.assistant.default_assistant import DefaultAssistant
from runtime.graph.graph_engine import GraphEngine
from runtime.graph.memory_zones import GraphMemoryZones
from runtime.models.nanbeige_router import generate_fast_response


def execute_fast_chat_turn(
    *,
    actor,
    task_queue,
    payload: dict[str, Any],
) -> dict[str, Any]:
    graph_engine = GraphEngine(actor)
    graph_payload = graph_engine.snapshot_payload()
    assistant = DefaultAssistant()
    memory_zones = GraphMemoryZones()
    person_id = str(payload.get("person_id") or payload.get("persona_id") or "").strip() or None
    user_id = str(payload.get("user_id") or "").strip() or None
    style_profile = load_style_profile(user_id) if user_id else None
    lookup = graph_engine.fast_lookup(
        query=str(payload.get("message") or ""),
        context=str(payload.get("context") or ""),
        limit=10,
    )
    decision = assistant.assess(
        query=str(payload.get("message") or ""),
        context=str(payload.get("context") or ""),
        lookup=lookup,
        apply_to_graph=bool(payload.get("apply_to_graph", payload.get("save_to_graph", True))),
    )
    if decision.clarification_needed:
        return {
            "ok": True,
            "assistant_reply": decision.clarification_question,
            "assistant_reply_en": decision.clarification_question,
            "response_mode": "assistant_clarification",
            "ram_graph": {"ranked_context": list(lookup.get("ranked_nodes") or []), "nodes": [], "edges": [], "micro_signals": dict(lookup.get("signals") or {})},
            "context_nodes": list(lookup.get("ranked_nodes") or [])[:10],
            "agent_plan": [],
            "core": {"speech_dna": {}, "scenario": {}, "dialogue_contract": {}, "assistant": decision.as_dict()},
            "style": {"applied": False, "user_id": str(user_id or ""), "sample_count": 0, "last_updated": "", "style_examples": [], "style_embedding": []},
            "graph_job": None,
            "graph_scenario": {
                "status": "clarification_needed",
                "should_update": False,
                "requests": [],
                "missing_tokens": list(lookup.get("missing_tokens") or []),
                "reason": decision.reason,
            },
            "graph_guard": {
                "mode": "no_graph_write_before_clarification",
                "controller_ok": True,
                "controller_reason": "",
                "summary_preview": decision.clarification_question[:220],
            },
            "archive_updates": [],
            "web_context": {"snippets": []},
            "graph_binding": {"attached": False, "source_id": ""},
            "graph_diff": {"attached": False, "nodes": [], "edges": [], "node_count": 0, "edge_count": 0},
        }
    result = generate_fast_response(
        graph_payload,
        query=str(payload.get("message") or ""),
        context=str(payload.get("context") or ""),
        person_id=person_id,
        user_id=user_id,
        style_profile=style_profile,
    )
    job = None
    if decision.requires_graph_update:
        job = task_queue.submit(
            "graph_build",
            {
                "query": str(payload.get("message") or ""),
                "context": str(payload.get("context") or ""),
                "assistant_reply": str(result.get("assistant_reply") or ""),
                "person_id": person_id or "",
                "source_id": str(payload.get("source_id") or ""),
            },
        )
        memory_zones.write_pending(
            str(job.get("job_id") or ""),
            {
                "status": str(job.get("status") or "queued"),
                "query": str(payload.get("message") or ""),
                "context": str(payload.get("context") or ""),
                "assistant_reply": str(result.get("assistant_reply") or ""),
                "assistant_plan": decision.as_dict(),
                "lookup": lookup,
            },
        )
    result["ok"] = True
    result["graph_job"] = job
    result["graph_scenario"] = {
        "status": "queued" if job else ("disabled" if not bool(payload.get("apply_to_graph", payload.get("save_to_graph", True))) else "verified_graph_sufficient"),
        "should_update": bool(job),
        "requests": [],
        "missing_tokens": list(lookup.get("missing_tokens") or []),
        "reason": "queued_for_background_build" if job else decision.reason,
    }
    result["graph_guard"] = {
        "mode": "async_background_graph_build",
        "controller_ok": True,
        "controller_reason": "",
        "summary_preview": str(result.get("assistant_reply") or "")[:220],
    }
    result["archive_updates"] = []
    result["web_context"] = {"snippets": []}
    result["graph_binding"] = {"attached": False, "source_id": ""}
    result["graph_diff"] = {"attached": False, "nodes": [], "edges": [], "node_count": 0, "edge_count": 0}
    result.setdefault("core", {})
    result["core"]["assistant"] = decision.as_dict()
    return result
