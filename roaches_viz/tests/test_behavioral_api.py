from __future__ import annotations

import asyncio
from pathlib import Path
import time

from roaches_viz.api import (
    CognitiveEdgeCreateRequest,
    CognitiveNodeCreateRequest,
    DialogueRequest,
    IngestRequest,
    RamPreviewRequest,
    ScenarioPreviewRequest,
    SessionCreateRequest,
    SessionSaveRequest,
    create_app,
)
from roaches_viz.concurrency.graph_actor import GraphActor
from runtime.task_queue import TaskQueue
from runtime.workers.graph_worker import GraphWorker


def _get_endpoint(app, path: str, method: str | None = None):
    for route in app.routes:
        if getattr(route, "path", None) == path and (method is None or method in getattr(route, "methods", set())):
            return route.endpoint
    raise AssertionError(f"endpoint not found: {path}")


def test_behavioral_api_ingest_dialogue_and_runtime_indexes(tmp_path: Path) -> None:
    previous_cwd = Path.cwd()
    try:
        import os
        os.chdir(tmp_path)
        _run_behavioral_api_test(tmp_path)
    finally:
        os.chdir(previous_cwd)


def _run_behavioral_api_test(tmp_path: Path) -> None:
    actor = GraphActor(db_path=tmp_path / "graph.sqlite3", top_tokens_per_sentence=3)
    actor.start()
    task_queue = TaskQueue(name="behavioral-api-queue")
    worker = GraphWorker(task_queue=task_queue, actor=actor, name="behavioral-api-worker")
    worker.start()
    try:
        app = create_app()
        app.state.graph_actor = actor
        app.state.graph_task_queue = task_queue
        ingest_endpoint = _get_endpoint(app, "/ingest")
        graph_endpoint = _get_endpoint(app, "/graph")
        search_endpoint = _get_endpoint(app, "/graph/search")
        domains_endpoint = _get_endpoint(app, "/domains")
        persons_endpoint = _get_endpoint(app, "/persons")
        agents_endpoint = _get_endpoint(app, "/agents")
        professions_endpoint = _get_endpoint(app, "/professions")
        agent_roles_endpoint = _get_endpoint(app, "/agent-roles")
        scenario_endpoint = _get_endpoint(app, "/scenario/preview")
        ram_preview_endpoint = _get_endpoint(app, "/ram/preview")
        dialogue_endpoint = _get_endpoint(app, "/dialogue/respond")
        graph_job_endpoint = _get_endpoint(app, "/graph/jobs/{job_id}")
        audit_endpoint = _get_endpoint(app, "/graph/audit")
        sessions_endpoint = _get_endpoint(app, "/sessions", "GET")
        create_session_endpoint = _get_endpoint(app, "/sessions", "POST")
        get_session_endpoint = _get_endpoint(app, "/sessions/{session_id}")
        save_session_endpoint = _get_endpoint(app, "/sessions/{session_id}", "PUT")
        subgraph_endpoint = _get_endpoint(app, "/graph/subgraph")
        create_node_endpoint = _get_endpoint(app, "/nodes", "POST")
        create_edge_endpoint = _get_endpoint(app, "/edges", "POST")

        ingest = ingest_endpoint(
            IngestRequest(
                source_id="src:api",
                text='Psychology note. Maria said "If you loved me you would do this." Family conflict triggers guilt pressure.',
            )
        )
        assert ingest["ok"] is True

        graph = graph_endpoint(edge_type=None, min_weight=0.0)
        assert graph["ok"] is True
        assert any(node["type"] == "PATTERN" for node in graph["nodes"])
        assert any(node["type"] == "PERSON" for node in graph["nodes"])
        assert any(node["type"] == "SIGNAL" for node in graph["nodes"])
        assert any(node["type"] == "TRAIT" for node in graph["nodes"])
        assert any(edge["type"] == "USES_PATTERN" for edge in graph["edges"])

        domains = domains_endpoint()
        assert domains["ok"] is True
        assert any(item["domain_id"] == "domain:psychology" for item in domains["domains"])

        persons = persons_endpoint()
        assert persons["ok"] is True
        assert any(item["id"] == "person:maria" for item in persons["persons"])

        agents = agents_endpoint()
        assert agents["ok"] is True
        assert any(item["type"] == "AGENT" for item in agents["agents"])

        professions = professions_endpoint()
        assert professions["ok"] is True
        assert any(item["type"] == "PROFESSION" for item in professions["professions"])

        roles = agent_roles_endpoint()
        assert roles["ok"] is True
        assert any(item["role_id"] == "law" for item in roles["roles"])

        scenario = scenario_endpoint(ScenarioPreviewRequest(query="There is contract risk here.", profession=""))
        assert scenario["ok"] is True
        assert scenario["scenario"]["scenario_type"] == "professional_consultation"

        search = search_endpoint(query="guilt", limit=10)
        assert search["ok"] is True
        assert search["nodes"]

        ram = ram_preview_endpoint(
            RamPreviewRequest(
                message="Is this guilt pressure?",
                context="The phrase came right after criticism.",
                person_id="person:maria",
            )
        )
        assert ram["ok"] is True
        assert ram["ram_graph"]["ranked_context"]
        assert ram["agent_plan"]

        dialogue = asyncio.run(
            dialogue_endpoint(
                DialogueRequest(
                    message="Is this guilt pressure?",
                    context="The phrase came right after criticism.",
                    person_id="person:maria",
                    apply_to_graph=False,
                )
            )
        )
        assert dialogue["ok"] is True
        assert dialogue["assistant_reply"]
        assert dialogue["agent_plan"]
        assert dialogue["ram_graph"]["ranked_context"]
        assert dialogue["core"]["speech_dna"] is not None
        assert dialogue["core"]["scenario"]["scenario_type"]
        assert "dialogue_contract" in dialogue["core"]
        assert "assistant" in dialogue["core"]
        assert dialogue["graph_job"] is None

        audit = audit_endpoint()
        assert audit["ok"] is True
        assert audit["checks"]["node_count"] >= 1
        assert "recommendations" in audit

        session_create = create_session_endpoint(SessionCreateRequest(title="API session", tools={"user_id": "u1"}))
        assert session_create["ok"] is True
        session_id = session_create["session"]["session_id"]
        assert session_id

        session_saved = save_session_endpoint(
            session_id,
            SessionSaveRequest(
                title="API session",
                last_query="guilt pressure",
                tools={"user_id": "u1", "extra_context": "family"},
                messages=[{"role": "user", "message": "hello"}],
            ),
        )
        assert session_saved["ok"] is True
        assert session_saved["session"]["last_query"] == "guilt pressure"

        session_loaded = get_session_endpoint(session_id)
        assert session_loaded["ok"] is True
        assert session_loaded["session"]["tools"]["extra_context"] == "family"

        session_list = sessions_endpoint()
        assert session_list["ok"] is True
        assert any(item["session_id"] == session_id for item in session_list["sessions"])

        created_node = create_node_endpoint(
            CognitiveNodeCreateRequest(
                node_id="concept:test-session",
                type="CONCEPT",
                label="Test session concept",
                short_gloss="A concise test concept",
                what_it_is="A concept created by the API test.",
            )
        )
        assert created_node["ok"] is True

        created_edge = create_edge_endpoint(
            CognitiveEdgeCreateRequest(
                src_id="domain:psychology",
                dst_id="concept:test-session",
                type="RELATED_TO",
                weight=0.8,
                confidence=0.9,
            )
        )
        assert created_edge["ok"] is True

        subgraph = subgraph_endpoint(query="psychology", limit=12, hops=1)
        assert subgraph["ok"] is True
        assert subgraph["nodes"]
        assert subgraph["seed_node_ids"]

        queued = asyncio.run(
            dialogue_endpoint(
                DialogueRequest(
                    message="Map the hidden pressure pattern in this family exchange.",
                    context="Repeated guilt and criticism are both present.",
                    person_id="person:maria",
                    apply_to_graph=True,
                )
            )
        )
        assert queued["ok"] is True
        assert queued["graph_job"] is not None
        job_id = str(queued["graph_job"]["job_id"])
        status = {}
        for _ in range(40):
            status = graph_job_endpoint(job_id)
            if (status.get("job") or {}).get("status") in {"done", "skipped", "failed"}:
                break
            time.sleep(0.05)
        assert (status.get("job") or {}).get("status") in {"done", "skipped", "failed"}
    finally:
        worker.stop()
        worker.join(timeout=2.0)
        actor.shutdown()
