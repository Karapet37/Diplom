from __future__ import annotations

import asyncio
from pathlib import Path
import time

from roaches_viz.api import (
    DialogueRequest,
    CognitiveEdgeUpdateRequest,
    CognitiveNodeUpdateRequest,
    ComposeRequest,
    FoundationLoadRequest,
    GraphImportRequest,
    IngestRequest,
    InterpretRequest,
    PlanRequest,
    RamPreviewRequest,
    RebuildRequest,
    StyleLearnRequest,
    create_app,
)
from roaches_viz.concurrency.graph_actor import GraphActor
from runtime.graph.memory_zones import GraphMemoryZones
from runtime.task_queue import TaskQueue
from runtime.workers.graph_worker import GraphWorker


def _get_endpoint(app, path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint
    raise AssertionError(f"endpoint not found: {path}")


def test_behavioral_runtime_endpoints_work_together(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("STYLE_PROFILES_DIR", str(tmp_path))
    actor = GraphActor(db_path=tmp_path / "graph.sqlite3", top_tokens_per_sentence=4)
    actor.start()
    task_queue = TaskQueue(name="test-graph-build-queue")
    worker = GraphWorker(task_queue=task_queue, actor=actor, name="test-graph-build-worker")
    worker.start()
    memory_zones = GraphMemoryZones(root=tmp_path / "graph")
    worker.memory_zones = memory_zones
    try:
        app = create_app()
        app.state.graph_actor = actor
        app.state.graph_task_queue = task_queue

        health = _get_endpoint(app, "/health")
        foundations = _get_endpoint(app, "/foundations")
        foundations_load = _get_endpoint(app, "/foundations/load")
        graph = _get_endpoint(app, "/graph")
        search = _get_endpoint(app, "/graph/search")
        audit = _get_endpoint(app, "/graph/audit")
        domains = _get_endpoint(app, "/domains")
        sources = _get_endpoint(app, "/sources")
        ingest = _get_endpoint(app, "/ingest")
        interpret = _get_endpoint(app, "/interpret")
        compose = _get_endpoint(app, "/compose")
        plan = _get_endpoint(app, "/plan")
        rebuild = _get_endpoint(app, "/rebuild")
        persons = _get_endpoint(app, "/persons")
        agents = _get_endpoint(app, "/agents")
        professions = _get_endpoint(app, "/professions")
        ram_preview = _get_endpoint(app, "/ram/preview")
        dialogue = _get_endpoint(app, "/dialogue/respond")
        graph_job_status = _get_endpoint(app, "/graph/jobs/{job_id}")
        style_learn = _get_endpoint(app, "/style/learn")
        style_profile = _get_endpoint(app, "/style/profile")
        import_graph = _get_endpoint(app, "/graph/import")
        update_node = _get_endpoint(app, "/nodes/{node_id}")
        update_edge = _get_endpoint(app, "/edges")

        health_result = health()
        assert health_result["ok"] is True

        foundation_result = foundations()
        assert foundation_result["ok"] is True
        assert "human_foundations" in foundation_result["datasets"]
        assert "psychology_foundations" in foundation_result["datasets"]
        assert "big_bang_theory" not in foundation_result["datasets"]

        loaded = foundations_load(FoundationLoadRequest(dataset_id="psychology_foundations", replace_graph=True))
        assert loaded["ok"] is True
        assert loaded["seed"] == "psychology_foundations"

        graph_result = graph(edge_type=None, min_weight=0.0)
        assert graph_result["ok"] is True
        assert any(node["id"] == "domain:psychology" for node in graph_result["nodes"])
        assert any(node["type"] == "AGENT" for node in graph_result["nodes"])
        assert any(node["type"] == "PROFESSION" for node in graph_result["nodes"])

        search_result = search(query="psychology", limit=10)
        assert search_result["ok"] is True
        assert search_result["nodes"]

        domain_result = domains()
        assert domain_result["ok"] is True
        assert any(item["domain_id"] == "domain:psychology" for item in domain_result["domains"])

        persons_result = persons()
        assert persons_result["ok"] is True
        assert isinstance(persons_result["persons"], list)

        agents_result = agents()
        assert agents_result["ok"] is True
        assert agents_result["agents"]

        professions_result = professions()
        assert professions_result["ok"] is True
        assert professions_result["professions"]

        ingest_result = ingest(
            IngestRequest(
                source_id="src:integration",
                text='Family note. John said: "If you loved me you would do this." The pressure appears after criticism.',
            )
        )
        assert ingest_result["ok"] is True
        assert ingest_result["nodes"] > 0

        interpret_result = interpret(InterpretRequest(text="If you loved me you would do this.", k=3, mode="build"))
        assert interpret_result["ok"] is True
        assert interpret_result["top_hypotheses"]
        assert interpret_result["analysis"]

        plan_result = plan(
            PlanRequest(
                text="If you loved me you would do this.",
                goal="understand the pressure pattern and choose a safe next step",
                mode="build",
                depth=3,
                beam_width=4,
                state={"clarity": 0.4, "alignment": 0.4, "progress": 0.3, "rapport": 0.4, "risk": 0.6},
            )
        )
        assert plan_result["ok"] is True
        assert plan_result["best_line"]
        assert plan_result["trace"]

        compose_result = compose(ComposeRequest(mode="build", task="analyze manipulation pattern from grounded graph context"))
        assert compose_result["ok"] is True
        assert "grounded" in compose_result["prompt"].lower()

        ram_result = ram_preview(
            RamPreviewRequest(
                message="Is this guilt pressure or normal care?",
                context="The phrase came right after criticism in a family conflict.",
                person_id="person:john",
            )
        )
        assert ram_result["ok"] is True
        assert ram_result["ram_graph"]["ranked_context"]
        assert ram_result["agent_plan"]

        style_not_triggered = style_learn(
            StyleLearnRequest(
                user_id="integration_user",
                messages=[
                    {"role": "user", "message": "Look, stop dressing pressure up like care and say it directly."},
                    {"role": "user", "message": "Honestly, if the pattern repeats after criticism, call it out."},
                ],
                learn_style_button=False,
            )
        )
        assert style_not_triggered["learned"] is False
        assert style_profile(user_id="integration_user")["profile"] is None

        style_learned = style_learn(
            StyleLearnRequest(
                user_id="integration_user",
                messages=[
                    {"role": "user", "message": "Look, stop dressing pressure up like care and say it directly."},
                    {"role": "user", "message": "Honestly, if the pattern repeats after criticism, call it out."},
                    {"role": "user", "message": "Keep it practical and stop circling the point."},
                ],
                learn_style_button=True,
            )
        )
        assert style_learned["learned"] is True
        assert style_learned["profile"]["style_examples"]
        assert style_profile(user_id="integration_user")["profile"]["style_embedding"]

        dialogue_result = asyncio.run(
            dialogue(
                DialogueRequest(
                    message="Is this guilt pressure or normal care?",
                    context="The phrase came right after criticism in a family conflict.",
                    person_id="person:john",
                    user_id="integration_user",
                    apply_to_graph=True,
                    save_to_graph=True,
                    use_internet=False,
                )
            )
        )
        assert dialogue_result["ok"] is True
        assert dialogue_result["assistant_reply"]
        assert "ram_graph" in dialogue_result
        assert dialogue_result["graph_guard"]["controller_ok"] is True
        assert dialogue_result["style"]["applied"] is True
        assert dialogue_result["graph_job"]["status"] in {"queued", "running"}

        job_id = str(dialogue_result["graph_job"]["job_id"])
        job_status = {}
        for _ in range(40):
            job_status = graph_job_status(job_id)
            if (job_status.get("job") or {}).get("status") in {"done", "skipped", "failed"}:
                break
            time.sleep(0.05)
        assert (job_status.get("job") or {}).get("status") in {"done", "skipped"}
        if (job_status.get("job") or {}).get("status") == "done":
            assert memory_zones.verified_path(job_id).exists()
        else:
            assert memory_zones.pending_path(job_id).exists()

        graph_after_chat = graph(edge_type=None, min_weight=0.0)
        first_node = next(node for node in graph_after_chat["nodes"] if node["type"] in {"PATTERN", "SIGNAL", "PERSON"})
        node_update = update_node(
            first_node["id"],
            CognitiveNodeUpdateRequest(
                label=f"{first_node['label']} refined",
                short_gloss="Short practical meaning.",
                plain_explanation="Plain explanation for a human reader.",
                what_it_is="A practical concept node.",
                how_it_works="It links examples, signals, and patterns.",
                how_to_recognize="Look for repeated pressure after criticism.",
                examples=["If you loved me, you would do this."],
                tags=["family", "pressure"],
            ),
        )
        assert node_update["ok"] is True
        assert node_update["node"]["label"].endswith("refined")

        current_edge = graph_after_chat["edges"][0]
        edge_update = update_edge(
            CognitiveEdgeUpdateRequest(
                src_id=current_edge["src_id"],
                dst_id=current_edge["dst_id"],
                type=current_edge["type"],
                weight=0.77,
                confidence=0.66,
            )
        )
        assert edge_update["ok"] is True
        assert edge_update["edge"]["weight"] == 0.77

        imported = import_graph(
            GraphImportRequest(
                replace_graph=True,
                payload={
                    "id": "behavior_import",
                    "nodes": [
                        {
                            "id": "domain:test",
                            "type": "DOMAIN",
                            "label": "Test domain",
                            "name": "Test domain",
                            "description": "A small imported domain.",
                            "what_it_is": "Imported domain root.",
                            "how_it_works": "Groups imported nodes.",
                            "how_to_recognize": "Exists only in the import test.",
                            "examples": ["Imported example"],
                            "tags": ["import"],
                        },
                        {
                            "id": "pattern:test_pattern",
                            "type": "PATTERN",
                            "label": "Test pattern",
                            "name": "Test pattern",
                            "description": "Imported pattern node.",
                            "what_it_is": "A behavior pattern used to test import.",
                            "how_it_works": "Connects to the test domain.",
                            "how_to_recognize": "Appears after graph import.",
                            "examples": ["Imported concept example"],
                            "tags": ["import", "pattern"],
                        },
                    ],
                    "edges": [
                        {
                            "src_id": "domain:test",
                            "dst_id": "pattern:test_pattern",
                            "type": "RELATED_TO",
                            "weight": 0.9,
                            "confidence": 0.8,
                            "evidence": [
                                {
                                    "source_id": "import:graph",
                                    "snippet_text": "Imported concept belongs to imported domain.",
                                    "offset_start": 0,
                                    "offset_end": 42,
                                }
                            ],
                        }
                    ],
                },
            )
        )
        assert imported["ok"] is True
        assert imported["seed"] == "behavior_import"

        imported_graph = graph(edge_type=None, min_weight=0.0)
        assert any(node["id"] == "domain:test" for node in imported_graph["nodes"])
        assert any(edge["type"] == "RELATED_TO" for edge in imported_graph["edges"])

        rebuild_result = rebuild(RebuildRequest(mode="full", source_ids=[]))
        assert rebuild_result["ok"] is True

        sources_result = sources()
        assert sources_result["ok"] is True
        assert sources_result["count"] >= 1

        audit_result = audit()
        assert audit_result["ok"] is True
        assert audit_result["checks"]["node_count"] >= 1
    finally:
        worker.stop()
        worker.join(timeout=2.0)
        actor.shutdown()
