from __future__ import annotations

import threading
from typing import Any

from runtime.graph.graph_builder import materialize_graph_build, prepare_graph_build
from runtime.graph.graph_engine import GraphEngine
from runtime.graph.memory_zones import GraphMemoryZones


class GraphWorker(threading.Thread):
    def __init__(self, *, task_queue, actor, name: str = "graph-worker"):
        super().__init__(name=name, daemon=True)
        self.task_queue = task_queue
        self.graph_engine = GraphEngine(actor)
        self.memory_zones = GraphMemoryZones()
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            task = self.task_queue.claim(timeout=0.5)
            if task is None:
                continue
            if task.kind != "graph_build":
                self.task_queue.fail(task.job_id, "unknown_task_kind")
                continue
            payload = dict(task.payload or {})
            try:
                self.memory_zones.update_pending(
                    task.job_id,
                    {"status": "running", "query": str(payload.get("query") or ""), "assistant_reply": str(payload.get("assistant_reply") or "")},
                )
                graph_payload = self.graph_engine.snapshot_payload()
                lookup = self.graph_engine.fast_lookup(
                    query=str(payload.get("query") or ""),
                    context=str(payload.get("assistant_reply") or ""),
                    limit=10,
                )
                scenario = prepare_graph_build(
                    graph_payload,
                    query=str(payload.get("query") or ""),
                    assistant_reply=str(payload.get("assistant_reply") or ""),
                    person_id=str(payload.get("person_id") or "") or None,
                )
                if not scenario.get("should_update"):
                    self.memory_zones.update_pending(
                        task.job_id,
                        {
                            "status": "skipped",
                            "lookup": lookup,
                            "scenario": scenario,
                            "reason": str(scenario.get("reason") or "graph_context_sufficient"),
                        },
                    )
                    self.task_queue.skip(
                        task.job_id,
                        str(scenario.get("reason") or "graph_context_sufficient"),
                        scenario=scenario,
                        lookup=lookup,
                    )
                    continue
                materialized = materialize_graph_build(
                    graph_payload,
                    query=str(payload.get("query") or ""),
                    assistant_reply=str(payload.get("assistant_reply") or ""),
                    requests=list(scenario.get("requests") or []),
                    person_id=str(payload.get("person_id") or "") or None,
                )
                self.memory_zones.update_pending(
                    task.job_id,
                    {
                        "status": "materialized",
                        "lookup": lookup,
                        "scenario": scenario,
                        "materialized": {
                            "requests": list(materialized.get("requests") or []),
                            "source_preview": str(materialized.get("source_preview") or ""),
                        },
                    },
                )
                applied = self.graph_engine.apply_materialized_graph(
                    memory_text=str(materialized.get("memory_text") or ""),
                    source_id=str(payload.get("source_id") or ""),
                    query=str(payload.get("query") or ""),
                    assistant_reply=str(payload.get("assistant_reply") or ""),
                    person_id=str(payload.get("person_id") or "") or None,
                )
                final_payload = {
                    "status": "done",
                    "query": str(payload.get("query") or ""),
                    "assistant_reply": str(payload.get("assistant_reply") or ""),
                    "lookup": lookup,
                    "scenario": scenario,
                    "materialized": {
                        "requests": list(materialized.get("requests") or []),
                        "source_preview": str(materialized.get("source_preview") or ""),
                    },
                    "result": applied,
                }
                self.memory_zones.move_to_verified(task.job_id, final_payload)
                self.task_queue.complete(
                    task.job_id,
                    lookup=lookup,
                    scenario=scenario,
                    materialized={
                        "requests": list(materialized.get("requests") or []),
                        "source_preview": str(materialized.get("source_preview") or ""),
                    },
                    result=applied,
                )
            except Exception as exc:
                self.memory_zones.update_pending(task.job_id, {"status": "failed", "reason": str(exc)})
                self.task_queue.fail(task.job_id, str(exc))
