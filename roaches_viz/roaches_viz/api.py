from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
from pathlib import Path
import queue
import threading
import time

from fastapi import APIRouter, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from core import ScenarioEngine, StyleEngine, default_agent_roles
from runtime.api.chat_api import execute_fast_chat_turn
from runtime.sessions.session_store import SessionStore
from runtime.task_queue import TaskQueue
from runtime.workers.graph_worker import GraphWorker

from .config import default_settings
from .concurrency.graph_actor import GraphActor
from .human_analysis import analyze_human_context
from .interpret.updater import interpret_text
from .planner.expectimax import plan_expectimax
from .planner.utility import normalize_state
from .prompt_compose import compose_prompt_payload, list_prompt_modes
from .foundations import list_builtin_foundations

INFERENCE_TIMEOUT_SECONDS = 120.0


async def _await_actor_command(actor: GraphActor, command: str, payload: dict[str, object], *, timeout: float) -> dict[str, object]:
    result_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

    def _runner() -> None:
        try:
            result = actor.ask(command, payload, timeout=timeout)
        except Exception as exc:
            result_queue.put(("err", exc))
            return
        result_queue.put(("ok", result))

    threading.Thread(target=_runner, name=f"api-{command}", daemon=True).start()
    deadline = time.monotonic() + timeout + 1.0
    while time.monotonic() < deadline:
        try:
            status, value = result_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.05)
            continue
        if status == "err":
            raise value  # type: ignore[misc]
        return value  # type: ignore[return-value]
    raise asyncio.TimeoutError(f"Actor '{actor.name}' timed out waiting for command '{command}' after {timeout:.1f}s")


class IngestRequest(BaseModel):
    text: str = Field(min_length=1)
    source_id: str | None = None


class DialogueRequest(BaseModel):
    message: str = Field(min_length=1)
    context: str = Field(default="")
    language: str = Field(default="en")
    llm_role: str = Field(default="general")
    chat_model_role: str = Field(default="general")
    person_id: str = Field(default="")
    persona_id: str = Field(default="")
    source_id: str | None = None
    save_to_graph: bool = Field(default=True)
    apply_to_graph: bool = Field(default=True)
    use_internet: bool = Field(default=False)
    user_id: str = Field(default="")


ChatRequest = DialogueRequest


class RebuildRequest(BaseModel):
    mode: str = Field(default="full")
    source_ids: list[str] = Field(default_factory=list)


class InterpretRequest(BaseModel):
    text: str = Field(min_length=1)
    k: int = Field(default=3, ge=2, le=6)
    mode: str = Field(default="build")


class ComposeRequest(BaseModel):
    mode: str = Field(default="build")
    task: str = Field(min_length=1)


class PlanRequest(BaseModel):
    state: dict[str, float] = Field(default_factory=dict)
    goal: str = Field(min_length=1)
    mode: str = Field(default="build")
    depth: int = Field(default=3, ge=1, le=6)
    beam_width: int = Field(default=4, ge=1, le=8)
    text: str | None = None
    hypotheses: list[dict[str, object]] = Field(default_factory=list)
    k_hypotheses: int = Field(default=3, ge=2, le=6)
    actions: list[dict[str, object]] = Field(default_factory=list)


class FoundationLoadRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    replace_graph: bool = Field(default=True)


class GraphImportRequest(BaseModel):
    payload: dict[str, object]
    replace_graph: bool = Field(default=True)


class RamPreviewRequest(BaseModel):
    message: str = Field(min_length=1)
    context: str = Field(default="")
    person_id: str = Field(default="")
    persona_id: str = Field(default="")


class CognitiveNodeUpdateRequest(BaseModel):
    name: str | None = None
    label: str | None = None
    type: str | None = None
    description: str | None = None
    short_gloss: str | None = None
    plain_explanation: str | None = None
    what_it_is: str | None = None
    how_it_works: str | None = None
    how_to_recognize: str | None = None
    examples: list[str] | None = None
    tags: list[str] | None = None
    speech_patterns: list[str] | None = None
    behavior_patterns: list[str] | None = None
    triggers: list[str] | None = None
    values: list[str] | None = None
    preferences: list[str] | None = None
    reaction_logic: list[str] | None = None
    tolerance_thresholds: dict[str, object] | None = None
    conflict_patterns: list[str] | None = None
    logic_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    emotion_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    relevance_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class CognitiveEdgeUpdateRequest(BaseModel):
    src_id: str = Field(min_length=1)
    dst_id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    weight: float | None = Field(default=None, ge=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class StyleLearnRequest(BaseModel):
    user_id: str = Field(min_length=1)
    messages: list[object] = Field(default_factory=list)
    learn_style_button: bool = Field(default=False)
    max_messages: int = Field(default=12, ge=3, le=48)


class ScenarioPreviewRequest(BaseModel):
    query: str = Field(min_length=1)
    profession: str = Field(default="")


class SessionCreateRequest(BaseModel):
    title: str = Field(default="New session")
    tools: dict[str, object] = Field(default_factory=dict)


class SessionSaveRequest(BaseModel):
    title: str | None = None
    last_query: str = Field(default="")
    tools: dict[str, object] = Field(default_factory=dict)
    messages: list[dict[str, object]] = Field(default_factory=list)


class CognitiveNodeCreateRequest(CognitiveNodeUpdateRequest):
    node_id: str = Field(min_length=1)
    type: str = Field(default="CONCEPT")
    name: str = Field(default="")
    label: str = Field(default="")


class CognitiveEdgeCreateRequest(CognitiveEdgeUpdateRequest):
    confidence: float | None = Field(default=0.7, ge=0.0, le=1.0)


class GraphSubgraphRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=24, ge=1, le=100)
    hops: int = Field(default=1, ge=0, le=3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = default_settings(Path(__file__).resolve().parents[1])
    actor = GraphActor(db_path=settings.db_path, top_tokens_per_sentence=settings.top_tokens_per_sentence)
    task_queue = TaskQueue(name="cognitive-graph-build-queue")
    worker = GraphWorker(task_queue=task_queue, actor=actor)
    actor.start()
    worker.start()
    setattr(actor, "_runtime_started", True)
    app.state.graph_actor = actor
    app.state.graph_task_queue = task_queue
    app.state.graph_worker = worker
    try:
        yield
    finally:
        worker.stop()
        worker.join(timeout=2.0)
        actor.shutdown()


def create_router(actor_provider, task_queue_provider=None) -> APIRouter:
    router = APIRouter()
    style_engine = StyleEngine()
    scenario_engine = ScenarioEngine()
    session_store = SessionStore(Path("data/sessions"))

    def get_actor() -> GraphActor:
        actor: GraphActor = actor_provider()
        if not actor.is_alive() and not getattr(actor, "_runtime_started", False):
            actor.start()
            setattr(actor, "_runtime_started", True)
        return actor

    def get_task_queue():
        if task_queue_provider is None:
            return None
        return task_queue_provider()

    @router.get("/health")
    def health() -> dict[str, object]:
        result = get_actor().ask("health")
        task_queue = get_task_queue()
        if task_queue is not None:
            result["task_queue"] = task_queue.summary()
        return result

    @router.post("/ingest")
    def ingest(payload: IngestRequest) -> dict[str, object]:
        result = get_actor().ask("ingest", {"text": payload.text, "source_id": payload.source_id})
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error") or "ingest failed")
        return result

    @router.get("/persons")
    def persons() -> dict[str, object]:
        return get_actor().ask("list_persons", {})

    @router.get("/agents")
    def agents() -> dict[str, object]:
        return get_actor().ask("list_agents", {})

    @router.get("/professions")
    def professions() -> dict[str, object]:
        return get_actor().ask("list_professions", {})

    async def _dialogue_respond(payload: DialogueRequest) -> dict[str, object]:
        task_queue = get_task_queue()
        if task_queue is not None:
            try:
                result = await asyncio.to_thread(
                    execute_fast_chat_turn,
                    actor=get_actor(),
                    task_queue=task_queue,
                    payload=payload.model_dump(),
                )
            except TimeoutError as exc:
                raise HTTPException(status_code=504, detail=str(exc)) from exc
            if not result.get("ok"):
                raise HTTPException(status_code=400, detail=result.get("error") or "dialogue failed")
            return result
        actor = get_actor()
        try:
            result = await _await_actor_command(actor, "dialogue_turn", payload.model_dump(), timeout=INFERENCE_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail=f"Actor '{actor.name}' timed out waiting for command 'dialogue_turn' after {INFERENCE_TIMEOUT_SECONDS:.1f}s") from exc
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error") or "dialogue failed")
        return result

    @router.post("/dialogue/respond")
    async def dialogue_respond(payload: DialogueRequest) -> dict[str, object]:
        return await _dialogue_respond(payload)

    @router.post("/chat/respond")
    async def chat_respond(payload: ChatRequest) -> dict[str, object]:
        return await _dialogue_respond(payload)

    @router.get("/graph")
    def graph(edge_type: str | None = Query(default=None), min_weight: float = Query(default=0.0, ge=0.0)) -> dict[str, object]:
        return get_actor().ask("snapshot", {"edge_type": edge_type, "min_weight": min_weight})

    @router.get("/graph/search")
    def graph_search(query: str = Query(min_length=1), limit: int = Query(default=20, ge=1, le=100)) -> dict[str, object]:
        return get_actor().ask("search_nodes", {"query": query, "limit": limit})

    @router.get("/graph/audit")
    def graph_audit() -> dict[str, object]:
        return get_actor().ask("graph_audit")

    @router.post("/ram/preview")
    def ram_preview(payload: RamPreviewRequest) -> dict[str, object]:
        result = get_actor().ask("ram_preview", payload.model_dump())
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error") or "ram preview failed")
        return result

    @router.get("/analysis/loops")
    def analysis_loops() -> dict[str, object]:
        return get_actor().ask("analysis_loops")

    @router.get("/domains")
    def domains() -> dict[str, object]:
        return get_actor().ask("list_domains", {})

    @router.get("/sources")
    def sources() -> dict[str, object]:
        return get_actor().ask("sources", {})

    @router.get("/sessions")
    def sessions() -> dict[str, object]:
        return {"ok": True, "sessions": session_store.list()}

    @router.post("/sessions")
    def session_create(payload: SessionCreateRequest) -> dict[str, object]:
        return {"ok": True, "session": session_store.create(title=payload.title, tools=payload.tools)}

    @router.get("/sessions/{session_id}")
    def session_get(session_id: str) -> dict[str, object]:
        session = session_store.load(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="session not found")
        return {"ok": True, "session": session}

    @router.put("/sessions/{session_id}")
    def session_save(session_id: str, payload: SessionSaveRequest) -> dict[str, object]:
        previous = session_store.load(session_id)
        if not previous:
            raise HTTPException(status_code=404, detail="session not found")
        session = session_store.save(
            {
                **previous,
                "session_id": session_id,
                "title": payload.title or previous.get("title") or "Untitled session",
                "last_query": payload.last_query,
                "tools": payload.tools or previous.get("tools") or {},
                "messages": payload.messages,
            }
        )
        return {"ok": True, "session": session}

    @router.get("/graph/jobs/{job_id}")
    def graph_job_status(job_id: str) -> dict[str, object]:
        task_queue = get_task_queue()
        if task_queue is not None:
            return {"ok": True, "job": task_queue.get(job_id)}
        return get_actor().ask("graph_job_status", {"job_id": job_id})

    @router.get("/graph/subgraph")
    def graph_subgraph(
        query: str = Query(min_length=1),
        limit: int = Query(default=24, ge=1, le=100),
        hops: int = Query(default=1, ge=0, le=3),
    ) -> dict[str, object]:
        return get_actor().ask("query_subgraph", {"query": query, "limit": limit, "hops": hops})

    @router.get("/style/profile")
    def style_profile(user_id: str = Query(min_length=1)) -> dict[str, object]:
        return {"ok": True, "user_id": user_id, "profile": style_engine.load(user_id)}

    @router.post("/style/learn")
    def style_learn(payload: StyleLearnRequest) -> dict[str, object]:
        result = style_engine.learn(
            user_id=payload.user_id,
            messages=list(payload.messages or []),
            learn_style_button=payload.learn_style_button,
            max_messages=payload.max_messages,
        )
        return {
            "ok": result.ok,
            "learned": result.learned,
            "user_id": result.user_id,
            "profile": result.profile,
            "reason": result.reason,
        }

    @router.get("/agent-roles")
    def agent_roles() -> dict[str, object]:
        return {"ok": True, "roles": [role.as_dict() for role in default_agent_roles().values()]}

    @router.post("/scenario/preview")
    def scenario_preview(payload: ScenarioPreviewRequest) -> dict[str, object]:
        scenario = scenario_engine.pick(payload.query, profession=payload.profession)
        return {"ok": True, "scenario": scenario.as_dict()}

    @router.get("/foundations")
    def foundations() -> dict[str, object]:
        return {"ok": True, "datasets": list_builtin_foundations()}

    @router.post("/foundations/load")
    def foundations_load(payload: FoundationLoadRequest) -> dict[str, object]:
        result = get_actor().ask("seed_series", {"dataset_id": payload.dataset_id, "replace_graph": payload.replace_graph})
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error") or "foundation load failed")
        return result

    @router.post("/graph/import")
    def graph_import(payload: GraphImportRequest) -> dict[str, object]:
        result = get_actor().ask("import_series_payload", {"payload": dict(payload.payload), "replace_graph": payload.replace_graph})
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error") or "graph import failed")
        return result

    @router.patch("/nodes/{node_id}")
    def update_node(node_id: str, payload: CognitiveNodeUpdateRequest) -> dict[str, object]:
        result = get_actor().ask("update_node", {"node_id": node_id, **payload.model_dump(exclude_none=True)})
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error") or "node update failed")
        return result

    @router.post("/nodes")
    def create_node(payload: CognitiveNodeCreateRequest) -> dict[str, object]:
        result = get_actor().ask("create_node", payload.model_dump(exclude_none=True))
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error") or "node create failed")
        return result

    @router.patch("/edges")
    def update_edge(payload: CognitiveEdgeUpdateRequest) -> dict[str, object]:
        result = get_actor().ask("update_edge", payload.model_dump(exclude_none=True))
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error") or "edge update failed")
        return result

    @router.post("/edges")
    def create_edge(payload: CognitiveEdgeCreateRequest) -> dict[str, object]:
        result = get_actor().ask("create_edge", payload.model_dump(exclude_none=True))
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error") or "edge create failed")
        return result

    @router.post("/rebuild")
    def rebuild(payload: RebuildRequest) -> dict[str, object]:
        mode = str(payload.mode or "full").lower()
        if mode not in {"full", "scoped"}:
            raise HTTPException(status_code=400, detail="mode must be full or scoped")
        result = get_actor().ask("rebuild", {"mode": mode, "source_ids": payload.source_ids})
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error") or "rebuild failed")
        return result

    @router.post("/interpret")
    def interpret(payload: InterpretRequest) -> dict[str, object]:
        return {"ok": True, **interpret_text(payload.text, k=payload.k, mode=payload.mode)}

    @router.post("/compose")
    def compose(payload: ComposeRequest) -> dict[str, object]:
        try:
            actor = get_actor()
            return {"ok": True, **compose_prompt_payload(mode=payload.mode, task=payload.task, db_path=actor.db_path)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/plan")
    def plan(payload: PlanRequest) -> dict[str, object]:
        hypotheses = list(payload.hypotheses or [])
        interpretation: dict[str, object] | None = None
        analysis: dict[str, object] | None = None
        if not hypotheses and payload.text:
            interpretation = interpret_text(payload.text, k=payload.k_hypotheses, mode=payload.mode)
            hypotheses = list(interpretation.get("top_hypotheses", []))
            analysis = dict(interpretation.get("analysis") or {})
        elif payload.text:
            analysis = analyze_human_context(payload.text, top_hypotheses=hypotheses, mode=payload.mode)
        normalized_state = normalize_state(payload.state)
        effective_state = dict(normalized_state)
        analysis_delta = dict((analysis or {}).get("analysis_adjustments", {}).get("planning_state_delta", {})) if analysis else {}
        for dim in ("clarity", "alignment", "progress", "rapport", "risk"):
            effective_state[dim] = max(0.0, min(1.0, float(effective_state.get(dim, 0.5)) + float(analysis_delta.get(dim, 0.0))))
        result = plan_expectimax(
            effective_state,
            payload.goal,
            depth=payload.depth,
            mode=payload.mode,
            hypotheses=hypotheses,
            beam_width=payload.beam_width,
            actions_override=list(payload.actions or []),
        )
        if interpretation is not None:
            result["interpretation"] = {
                "top_hypotheses": interpretation.get("top_hypotheses", []),
                "uncertainty": interpretation.get("uncertainty"),
                "best_clarifying_question": interpretation.get("best_clarifying_question"),
            }
        if analysis is not None:
            result["analysis"] = analysis
            result["effective_state"] = {
                "before": {k: round(float(normalized_state.get(k, 0.0)), 6) for k in ("clarity", "alignment", "progress", "rapport", "risk")},
                "delta": {k: round(float(analysis_delta.get(k, 0.0)), 6) for k in ("clarity", "alignment", "progress", "rapport", "risk")},
                "after": {k: round(float(effective_state.get(k, 0.0)), 6) for k in ("clarity", "alignment", "progress", "rapport", "risk")},
            }
        return {"ok": True, **result}

    return router


def create_app() -> FastAPI:
    app = FastAPI(title="RoachesViz", version="0.1.0", lifespan=lifespan)
    app.include_router(create_router(lambda: app.state.graph_actor, lambda: getattr(app.state, "graph_task_queue", None)))
    return app


app = create_app()
