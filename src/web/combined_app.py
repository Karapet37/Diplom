from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import Request
from runtime.task_queue import TaskQueue
from runtime.workers.graph_worker import GraphWorker

from src.web.api import attach_frontend_routes, create_app as create_workspace_app


_PASSIVE_IDLE_PATHS = {
    "/api/health",
    "/api/status",
    "/api/control/state",
    "/api/modules",
    "/api/project/overview",
    "/api/project/llm-policy",
    "/api/project/model-advisors",
    "/api/graph/events",
    "/api/merged/health",
    "/api/cognitive/health",
}


def _load_cognitive_modules():
    repo_root = Path(__file__).resolve().parents[2]
    roaches_root = repo_root / "roaches_viz"
    if str(roaches_root) not in sys.path:
        sys.path.insert(0, str(roaches_root))
    from roaches_viz.api import create_router as create_cognitive_router
    from roaches_viz.config import default_settings as cognitive_default_settings
    from roaches_viz.concurrency.graph_actor import GraphActor

    return roaches_root, create_cognitive_router, cognitive_default_settings, GraphActor


def _is_busy_request(request: Request) -> bool:
    if request.method.upper() == "OPTIONS":
        return False
    if request.method.upper() == "GET" and request.url.path in _PASSIVE_IDLE_PATHS:
        return False
    return True


def _run_idle_maintenance(app, actor, stop_event: threading.Event) -> None:
    while not stop_event.wait(10.0):
        if not getattr(app.state, "_cognitive_actor_started", False):
            continue
        busy_requests = int(getattr(app.state, "_busy_requests", 0) or 0)
        last_busy_finished_at = float(getattr(app.state, "_last_busy_finished_at", 0.0) or 0.0)
        if busy_requests > 0:
            continue
        if time.monotonic() - last_busy_finished_at < float(getattr(app.state, "_maintenance_idle_window_seconds", 4.0)):
            continue
        actor.submit("graph_hygiene_tick", {"reason": "idle_after_requests"})


def create_combined_app():
    app = create_workspace_app(include_frontend_routes=False)
    cognitive_root, create_cognitive_router, cognitive_default_settings, GraphActor = _load_cognitive_modules()
    settings = cognitive_default_settings(cognitive_root)
    actor = GraphActor(db_path=settings.db_path, top_tokens_per_sentence=settings.top_tokens_per_sentence)
    task_queue = TaskQueue(name="combined-graph-build-queue")
    worker = GraphWorker(task_queue=task_queue, actor=actor, name="combined-graph-worker")
    app.state.cognitive_actor = actor
    app.state.graph_task_queue = task_queue
    app.state.graph_worker = worker
    app.state._busy_requests = 0
    app.state._busy_requests_lock = threading.Lock()
    app.state._last_busy_finished_at = time.monotonic()
    app.state._maintenance_idle_window_seconds = 4.0
    app.state._maintenance_stop_event = threading.Event()
    app.state._maintenance_thread = None
    app.include_router(
        create_cognitive_router(lambda: app.state.cognitive_actor, lambda: getattr(app.state, "graph_task_queue", None)),
        prefix="/api/cognitive",
    )

    @app.middleware("http")
    async def track_busy_requests(request: Request, call_next):
        busy = _is_busy_request(request)
        lock = app.state._busy_requests_lock
        if busy:
            with lock:
                app.state._busy_requests += 1
        try:
            return await call_next(request)
        finally:
            if busy:
                with lock:
                    app.state._busy_requests = max(0, int(app.state._busy_requests) - 1)
                    app.state._last_busy_finished_at = time.monotonic()

    @app.on_event("startup")
    async def _startup_cognitive_engine() -> None:
        if not getattr(app.state, "_cognitive_actor_started", False):
            actor.start()
            setattr(actor, "_runtime_started", True)
            app.state._cognitive_actor_started = True
        if not worker.is_alive():
            worker.start()
        maintenance_thread = getattr(app.state, "_maintenance_thread", None)
        if maintenance_thread is None or not maintenance_thread.is_alive():
            app.state._maintenance_stop_event.clear()
            app.state._maintenance_thread = threading.Thread(
                target=_run_idle_maintenance,
                args=(app, actor, app.state._maintenance_stop_event),
                name="combined-idle-maintenance",
                daemon=True,
            )
            app.state._maintenance_thread.start()

    @app.on_event("shutdown")
    async def _shutdown_cognitive_engine() -> None:
        stop_event = getattr(app.state, "_maintenance_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        maintenance_thread = getattr(app.state, "_maintenance_thread", None)
        if maintenance_thread is not None:
            maintenance_thread.join(timeout=2.0)
            app.state._maintenance_thread = None
        if getattr(app.state, "_cognitive_actor_started", False):
            worker.stop()
            worker.join(timeout=2.0)
            actor.shutdown()
            app.state._cognitive_actor_started = False

    @app.get("/api/merged/health")
    def merged_health() -> dict[str, Any]:
        workspace_ok = True
        cognitive_health = actor.ask("health")
        return {
            "ok": True,
            "workspace": {"ok": workspace_ok},
            "cognitive_engine": cognitive_health,
            "runtime": {
                "busy_requests": int(getattr(app.state, "_busy_requests", 0) or 0),
                "maintenance_idle_window_seconds": float(getattr(app.state, "_maintenance_idle_window_seconds", 4.0)),
                "graph_jobs": task_queue.summary(),
            },
        }

    attach_frontend_routes(app)

    return app


app = create_combined_app()
