"""FastAPI endpoints for graph-first autonomous-system workspace."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import platform
from threading import Lock
from time import perf_counter
from typing import Any
import hmac

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from src.web.control_plane import RuntimeControlPlane
from src.web.observability import RuntimeMetrics, is_inference_path
from src.web.privacy_noise import PrivacyNoiseConfig, PrivacyNoisePlugin
from src.web.security import (
    InMemoryRateLimiter,
    JWTError,
    SecuritySettings,
    auth_error_response,
    create_access_token,
    extract_bearer_token,
    extract_client_ip,
    is_strong_password,
    is_strong_secret,
    requires_auth,
    should_rate_limit,
    try_enable_slowapi,
    verify_request_token,
)
from src.web.graph_workspace import GraphWorkspaceService
from src.web.web_context import fetch_random_wikipedia_article, search_wikipedia_articles


class NodeCreateRequest(BaseModel):
    node_type: str = Field(default="generic")
    attributes: dict[str, Any] | str | None = None
    state: dict[str, Any] | str | None = None

    # Human shortcuts
    first_name: str | None = None
    last_name: str | None = None
    bio: str | None = None
    profile_text: str | None = None
    employment: list[dict[str, Any]] | str | None = None
    employment_text: str | None = None

    # Company shortcuts
    name: str | None = None
    industry: str | None = None
    description: str | None = None


class EdgeCreateRequest(BaseModel):
    from_node: int
    to_node: int
    relation_type: str
    weight: float = 1.0
    direction: str = "directed"
    logic_rule: str = "explicit"


class NodeUpdateRequest(BaseModel):
    node_id: int
    attributes: dict[str, Any] | str | None = None
    state: dict[str, Any] | str | None = None
    first_name: str | None = None
    last_name: str | None = None
    bio: str | None = None
    name: str | None = None
    industry: str | None = None
    description: str | None = None


class GraphNodeAssistRequest(BaseModel):
    node_id: int
    action: str = "improve"
    message: str = ""
    context: str = ""
    user_id: str = "default_user"
    session_id: str = ""
    model_path: str = ""
    model_role: str = "general"
    apply_to_graph: bool = True
    verification_mode: str = "balanced"
    top_k: int = 5
    capture_dialect: bool = True
    auto_triage: bool = True
    triage_with_llm: bool = True


class GraphEdgeAssistRequest(BaseModel):
    from_node: int
    to_node: int
    relation_type: str
    direction: str = "directed"
    action: str = "improve"
    message: str = ""
    context: str = ""
    user_id: str = "default_user"
    session_id: str = ""
    model_path: str = ""
    model_role: str = "general"
    apply_to_graph: bool = True
    verification_mode: str = "balanced"
    top_k: int = 5
    capture_dialect: bool = True
    auto_triage: bool = True
    triage_with_llm: bool = True


class GraphFoundationCreateRequest(BaseModel):
    topic: str = ""
    context: str = ""
    target_node_id: int = 0
    depth: int = Field(default=2, ge=1, le=3)
    concept_limit: int = Field(default=4, ge=2, le=6)
    user_id: str = "default_user"
    session_id: str = ""
    model_path: str = ""
    model_role: str = "planner"


class WikipediaSearchRequest(BaseModel):
    query: str
    language: str = "ru"
    limit: int = Field(default=5, ge=1, le=8)


class NodeDeleteRequest(BaseModel):
    node_id: int


class EdgeUpdateRequest(BaseModel):
    from_node: int
    to_node: int
    relation_type: str
    direction: str = "directed"
    weight: float | None = None
    logic_rule: str | None = None
    metadata: dict[str, Any] | str | None = None


class EdgeDeleteRequest(BaseModel):
    from_node: int
    to_node: int
    relation_type: str
    direction: str = "directed"


class SimulateRequest(BaseModel):
    seed_node_ids: list[int] | str | None = None
    recursive_depth: int = 2
    propagation_steps: int = 3
    damping: float = 0.15
    activation: str = "tanh"
    infer_rounds: int = 1


class EventRewardRequest(BaseModel):
    event_id: int
    reward: float
    learning_rate: float = 0.15


class RelationReinforceRequest(BaseModel):
    relation_type: str
    reward: float
    learning_rate: float = 0.15


class ProfileInferRequest(BaseModel):
    text: str
    entity_type_hint: str = "human"
    create_graph: bool = True
    save_json: bool = True


class LivingProcessRequest(BaseModel):
    text: str
    user_id: str = "default_user"
    display_name: str = "Default User"
    language: str = "hy"
    session_id: str = "default"
    auto_snapshot: bool = True


class LivingSnapshotRequest(BaseModel):
    reason: str = "manual"
    user_id: str = ""


class LivingRollbackRequest(BaseModel):
    snapshot_id: int


class LivingSafeModeRequest(BaseModel):
    enabled: bool = True
    reason: str = ""


class LivingFeedbackRequest(BaseModel):
    user_id: str = ""
    event_type: str = "feedback"
    score: float = 0.0
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class LivingPromptRunRequest(BaseModel):
    prompt_name: str
    variables: dict[str, Any] = Field(default_factory=dict)
    user_id: str = ""
    session_id: str = ""
    security_decision: str = ""
    force_execute: bool = False


class LivingProjectMapRequest(BaseModel):
    max_files: int = 600


class LivingFileRequest(BaseModel):
    relative_path: str
    content: str = ""
    user_id: str = ""


class AuthTokenRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=512)


class KnowledgeAnalyzeRequest(BaseModel):
    text: str
    user_id: str = "default_user"
    display_name: str = "Default User"
    language: str = "en"
    branch_id: str = "main"
    apply_changes: bool = False
    sources: list[dict[str, Any] | str] = Field(default_factory=list)


class KnowledgeInitializeRequest(BaseModel):
    user_id: str = "foundation_user"
    display_name: str = "Foundation User"
    language: str = "en"
    branch_id: str = "foundation"
    apply_changes: bool = True


class KnowledgeBranchRequest(BaseModel):
    user_id: str = "default_user"
    branch_name: str


class KnowledgeMergeRequest(BaseModel):
    user_id: str = "default_user"
    base_snapshot_id: int
    target_snapshot_id: int
    apply_changes: bool = False


class ProjectPipelineRequest(BaseModel):
    text: str
    user_id: str = "default_user"
    display_name: str = "Default User"
    language: str = "en"
    session_id: str = "default"
    auto_snapshot: bool = True
    branch_id: str = "main"
    apply_knowledge_changes: bool = False
    sources: list[dict[str, Any] | str] = Field(default_factory=list)


class ProjectBootstrapRequest(BaseModel):
    user_id: str = "foundation_user"
    display_name: str = "Foundation User"
    language: str = "en"
    branch_id: str = "foundation"
    apply_changes: bool = True
    seed_graph_demo: bool = True


class ProjectDemoWatchRequest(BaseModel):
    persona_name: str = "You"
    narrative: str = ""
    language: str = "ru"
    reset_graph: bool = True
    use_llm: bool = True


class ProjectDailyModeRequest(BaseModel):
    text: str
    user_id: str = "default_user"
    display_name: str = "Default User"
    language: str = "en"
    session_id: str = "daily"
    auto_snapshot: bool = True
    recommendation_count: int = 4
    run_knowledge_analysis: bool = True
    apply_profile_update: bool = True
    use_llm_profile: bool = True
    include_client_profile: bool = False
    client: dict[str, Any] = Field(default_factory=dict)


class ProjectLLMDebateRequest(BaseModel):
    topic: str
    user_id: str = "default_user"
    session_id: str = ""
    hypothesis_count: int = 3
    attach_to_graph: bool = True
    proposer_role: str = "creative"
    critic_role: str = "analyst"
    judge_role: str = "planner"
    personalization: dict[str, Any] = Field(default_factory=dict)
    feedback_items: list[dict[str, Any] | str] = Field(default_factory=list)


class ProjectUserGraphUpdateRequest(BaseModel):
    user_id: str = "default_user"
    display_name: str = "Default User"
    text: str = ""
    language: str = "en"
    session_id: str = ""
    use_llm_profile: bool = True
    include_client_profile: bool = True
    client: dict[str, Any] = Field(default_factory=dict)
    profile_text: str = ""
    profile: dict[str, Any] = Field(default_factory=dict)
    personality: dict[str, Any] = Field(default_factory=dict)
    personalization: dict[str, Any] = Field(default_factory=dict)
    feedback_items: list[dict[str, Any] | str] = Field(default_factory=list)
    fears: list[str] | str | None = None
    desires: list[str] | str | None = None
    goals: list[str] | str | None = None
    principles: list[str] | str | None = None
    opportunities: list[str] | str | None = None
    abilities: list[str] | str | None = None
    access: list[str] | str | None = None
    knowledge: list[str] | str | None = None
    assets: list[str] | str | None = None


class ProjectPersonalTreeIngestRequest(BaseModel):
    user_id: str = "default_user"
    session_id: str = ""
    title: str = ""
    topic: str = ""
    text: str
    source_type: str = "text"
    source_url: str = ""
    source_title: str = ""
    max_points: int = 6
    parent_node_id: int = 0
    max_nodes: int = 200


class ProjectPersonalTreeNoteRequest(BaseModel):
    user_id: str = "default_user"
    session_id: str = ""
    note_id: int = 0
    parent_node_id: int = 0
    title: str = ""
    note: str = ""
    tags: list[str] | str | None = None
    links: list[str] | str | None = None
    source_type: str = "note"
    source_url: str = ""
    source_title: str = ""
    max_nodes: int = 180


class ProjectPersonalTreeViewRequest(BaseModel):
    user_id: str = "default_user"
    focus_node_id: int = 0
    max_nodes: int = 180


class ProjectPackagesManageRequest(BaseModel):
    user_id: str = "default_user"
    session_id: str = ""
    package_name: str = "inbox"
    action: str = "list"
    items: list[str] | str | None = None
    item_node_ids: list[int] | str | None = None
    model_role: str = "coder_reviewer"
    model_path: str = ""
    classify_with_llm: bool = True
    apply_changes: bool = False
    confirmation: str = ""


class ProjectMemoryNamespaceApplyRequest(BaseModel):
    user_id: str = "default_user"
    session_id: str = ""
    namespace: str = "personal"
    source_namespace: str = ""
    scope: str = "owned"
    node_ids: list[int] | str | None = None
    query: str = ""
    min_score: float = 0.2
    apply_changes: bool = True
    confirmation: str = ""


class ProjectMemoryNamespaceViewRequest(BaseModel):
    user_id: str = ""
    scope: str = "all"
    max_nodes: int = 220


class ProjectGraphRagQueryRequest(BaseModel):
    query: str
    user_id: str = ""
    scope: str = "all"
    namespace: str = ""
    top_k: int = 6
    use_llm: bool = True
    model_role: str = "analyst"
    model_path: str = ""


class ProjectContradictionScanRequest(BaseModel):
    user_id: str = ""
    session_id: str = ""
    scope: str = "all"
    namespace: str = ""
    max_nodes: int = 120
    top_k: int = 20
    min_overlap: float = 0.32
    apply_to_graph: bool = False
    confirmation: str = ""


class ProjectTaskRiskBoardRequest(BaseModel):
    user_id: str = "default_user"
    session_id: str = ""
    tasks: list[dict[str, Any] | str] = Field(default_factory=list)
    text: str = ""
    apply_to_graph: bool = True
    confirmation: str = ""


class ProjectTimelineReplayRequest(BaseModel):
    user_id: str = ""
    session_id: str = ""
    event_type: str = ""
    limit: int = 600
    from_ts: float = 0.0
    to_ts: float = 0.0


class ProjectLLMPolicyUpdateRequest(BaseModel):
    mode: str = "confirm_required"
    trusted_sessions: list[str] | str | None = None
    trusted_users: list[str] | str | None = None
    allow_apply_for_actions: list[str] | str | None = None
    merge_lists: bool = True


class ProjectQualityHarnessRequest(BaseModel):
    user_id: str = ""
    sample_queries: list[str] | str | None = None


class ProjectBackupCreateRequest(BaseModel):
    label: str = "manual"
    user_id: str = ""
    include_events: bool = True
    event_limit: int = 2000


class ProjectBackupRestoreRequest(BaseModel):
    path: str = ""
    latest: bool = True
    user_id: str = ""
    session_id: str = ""
    apply_changes: bool = False
    confirmation: str = ""
    restore_policy: bool = True


class ProjectAuditLogsRequest(BaseModel):
    limit: int = 200
    include_backups: bool = True


class ProjectWrapperRespondRequest(BaseModel):
    user_id: str = "default_user"
    session_id: str = ""
    message: str
    role: str = "general"
    model_path: str = ""
    context: str = ""
    mode_node_id: int = 0
    use_mode_policy: bool = True
    use_memory: bool = True
    memory_scope: str = "owned"
    memory_namespace: str = ""
    memory_top_k: int = 6
    personalization: dict[str, Any] = Field(default_factory=dict)
    feedback_items: list[dict[str, Any] | str] = Field(default_factory=list)
    apply_profile_update: bool = True
    store_interaction: bool = False
    subject_name: str = ""
    gossip_mode: str = "auto"
    allow_subject_branch_write: bool = True
    capture_dialect: bool = True
    auto_triage: bool = True
    triage_with_llm: bool = True


class ProjectWrapperProfileUpdateRequest(BaseModel):
    user_id: str = "default_user"
    preferred_role: str = "general"
    preferred_model_path: str = ""
    memory_scope: str = "owned"
    personalization: dict[str, Any] = Field(default_factory=dict)


class ProjectWrapperFeedbackRequest(BaseModel):
    user_id: str = "default_user"
    session_id: str = ""
    feedback_items: list[dict[str, Any] | str] = Field(default_factory=list)
    message: str = ""
    decision: str = ""
    score: float = 0.0
    target: str = ""
    attach_to_graph: bool = False


class ProjectHallucinationReportRequest(BaseModel):
    user_id: str = "default_user"
    session_id: str = ""
    prompt: str = ""
    llm_answer: str = ""
    correct_answer: str = ""
    source: str = ""
    tags: list[str] | str | None = None
    severity: str = "medium"
    confidence: float = 0.8
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectHallucinationCheckRequest(BaseModel):
    user_id: str = "default_user"
    prompt: str = ""
    llm_answer: str = ""
    top_k: int = 3


class ProjectArchiveVerifiedChatRequest(BaseModel):
    user_id: str = "default_user"
    session_id: str = ""
    message: str = ""
    context: str = ""
    model_path: str = ""
    model_role: str = "general"
    mode_node_id: int = 0
    use_mode_policy: bool = True
    apply_to_graph: bool = True
    verification_mode: str = "strict"
    top_k: int = 3
    subject_name: str = ""
    gossip_mode: str = "auto"
    allow_subject_branch_write: bool = True
    capture_dialect: bool = True
    auto_triage: bool = True
    triage_with_llm: bool = True


class ProjectChatGraphRequest(BaseModel):
    user_id: str = "default_user"
    session_id: str = ""
    message: str = ""
    context: str = ""
    chat_model_path: str = ""
    chat_model_role: str = "general"
    parser_backend: str = "local"
    parser_model_path: str = ""
    parser_model_role: str = "analyst"
    parser_ollama_model: str = ""
    use_internet: bool = True
    apply_to_graph: bool = True
    verification_mode: str = "balanced"
    top_k: int = 4


class ProjectModePolicyResolveRequest(BaseModel):
    user_id: str = "default_user"
    session_id: str = ""
    message: str = ""
    context: str = ""
    model_role: str = "general"
    mode_node_id: int = 0
    use_mode_policy: bool = True


class ProjectContextModeUpsertRequest(BaseModel):
    user_id: str = "default_user"
    session_id: str = ""
    mode_node_id: int = 0
    name: str = ""
    domain: str = "general"
    prompt_guardrails: str = ""
    protected_memory: str = ""
    llm_role: str = "analyst"
    summary: str = ""
    context_weight: float | None = None


class ProjectContextFocusCaptureRequest(BaseModel):
    user_id: str = "default_user"
    session_id: str = ""
    mode_node_id: int
    name: str = ""
    summary: str = ""
    details: str = ""
    subject: str = ""
    source: str = "context_mode"
    manual_capture: bool = True
    weight: float | None = None


class ProjectContextModeFeedbackRequest(BaseModel):
    user_id: str = "default_user"
    session_id: str = ""
    mode_node_id: int
    target_focus_node_id: int = 0
    decision: str = "good"
    summary: str = ""
    details: str = ""
    message: str = ""


class IntegrationLayerInvokeRequest(BaseModel):
    action: str = "wrapper.respond"
    host: str = "generic"
    app_id: str = "external_app"
    user_id: str = "default_user"
    session_id: str = ""
    input: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    context: str = ""
    model_path: str = ""
    model_role: str = "general"
    auto_triage: bool = True
    triage_with_llm: bool = True


class ProjectArchiveReviewApplyRequest(BaseModel):
    user_id: str = "default_user"
    session_id: str = ""
    message: str = ""
    context: str = ""
    summary: str = ""
    archive_updates: list[dict[str, Any]] | dict[str, Any] | str = Field(default_factory=list)
    verification_mode: str = "strict"
    apply_to_graph: bool = True
    top_k: int = 3


class ClientIntrospectionRequest(BaseModel):
    session_id: str = ""
    user_id: str = ""
    client: dict[str, Any] = Field(default_factory=dict)


class ControlPlaneUpdateRequest(BaseModel):
    enabled: bool | None = None
    read_only: bool | None = None
    allow_graph_writes: bool | None = None
    allow_project_demo: bool | None = None
    allow_project_daily: bool | None = None
    allow_client_introspection: bool | None = None
    allow_living_file_ops: bool | None = None
    allow_knowledge_mutations: bool | None = None
    allow_prompt_execution: bool | None = None


def _scan_modules() -> list[dict[str, Any]]:
    root = Path("src")
    buckets: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root)
        if rel.name == "__init__.py":
            continue
        namespace = rel.parts[0] if rel.parts else "root"
        buckets.setdefault(namespace, []).append(str(rel))

    descriptions = {
        "autonomous_graph": "Graph engine: node types, relations, inference, propagation.",
        "web": "FastAPI orchestration and React delivery layer.",
        "utils": "Environment and local LLM utilities.",
    }

    modules: list[dict[str, Any]] = []
    for namespace, files in sorted(buckets.items()):
        modules.append(
            {
                "name": namespace,
                "description": descriptions.get(namespace, "Project module."),
                "files": files,
                "count": len(files),
            }
        )
    return modules


class GraphEventHub:
    """Process-local websocket broadcaster for graph runtime events."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = Lock()

    def bind_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        with self._lock:
            self._connections.discard(websocket)

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        with self._lock:
            connections = list(self._connections)
        if not connections:
            return
        stale: list[WebSocket] = []
        for websocket in connections:
            try:
                await websocket.send_json(payload)
            except Exception:
                stale.append(websocket)
        if stale:
            with self._lock:
                for websocket in stale:
                    self._connections.discard(websocket)

    def publish_graph_event(self, event: Any) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        payload = {
            "type": "graph_event",
            "event": {
                "id": int(getattr(event, "id", 0)),
                "event_type": str(getattr(event, "event_type", "") or ""),
                "timestamp": float(getattr(event, "timestamp", 0.0) or 0.0),
                "payload": dict(getattr(event, "payload", {}) or {}),
            },
        }
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(payload), loop)
        except Exception:
            return


def attach_frontend_routes(app: FastAPI) -> None:
    dist_dir = Path("webapp/dist")
    index_path = dist_dir / "index.html"

    def _frontend_not_built() -> HTMLResponse:
        return HTMLResponse(
            content=(
                "<html><body style='font-family: sans-serif; padding: 18px;'>"
                "<h3>React frontend is not built yet.</h3>"
                "<p>Run <code>cd webapp && npm install && npm run build</code> "
                "then restart backend.</p>"
                "</body></html>"
            ),
            status_code=503,
        )

    def _serve_index() -> FileResponse:
        return FileResponse(
            index_path,
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/")
    def root() -> Any:
        if not index_path.exists():
            return _frontend_not_built()
        return _serve_index()

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> Any:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        if not dist_dir.exists() or not index_path.exists():
            return _frontend_not_built()

        target = dist_dir / full_path
        if target.exists() and target.is_file():
            return FileResponse(target)

        path_obj = Path(full_path)
        if full_path.startswith("assets/") or path_obj.suffix:
            raise HTTPException(status_code=404, detail="Asset not found")

        return _serve_index()


def create_app(*, include_frontend_routes: bool = True) -> FastAPI:
    graph = GraphWorkspaceService(use_env_adapter=True)
    security = SecuritySettings.from_env()
    control_plane = RuntimeControlPlane.from_env()
    metrics = RuntimeMetrics()
    privacy_noise = PrivacyNoisePlugin(PrivacyNoiseConfig.from_env())
    in_memory_limiter = InMemoryRateLimiter(security.rate_limit_per_minute)
    event_hub = GraphEventHub()
    graph.add_graph_event_listener(event_hub.publish_graph_event)

    app = FastAPI(
        title="Autonomous Graph Workspace API",
        version="2.0.0",
        description="Graph-first API for autonomous logical system simulation.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    rate_limit_backend = "disabled"
    if security.rate_limit_enable:
        if try_enable_slowapi(app, security):
            rate_limit_backend = "slowapi"
        else:
            rate_limit_backend = "in_memory"

    @app.on_event("startup")
    async def register_event_loop() -> None:
        event_hub.bind_event_loop(asyncio.get_running_loop())

    @app.on_event("shutdown")
    async def detach_graph_listener() -> None:
        graph.remove_graph_event_listener(event_hub.publish_graph_event)

    def _assert_control_admin(request: Request) -> None:
        admin_key = control_plane.admin_key()
        if admin_key:
            presented = str(request.headers.get("X-Control-Key", "") or "")
            if not presented or not hmac.compare_digest(presented, admin_key):
                raise HTTPException(status_code=403, detail="missing or invalid X-Control-Key")
            return
        if not security.auth_enable:
            raise HTTPException(
                status_code=503,
                detail="control updates require AUTH_ENABLE=1 or CONTROL_ADMIN_KEY",
            )

    @app.middleware("http")
    async def security_and_metrics_pipeline(request: Request, call_next):  # type: ignore[no-untyped-def]
        method = str(request.method or "GET").upper()
        path = str(request.url.path or "/")
        start = perf_counter()
        status_code = 500
        metrics.mark_inflight(+1)
        try:
            if rate_limit_backend == "in_memory" and should_rate_limit(
                settings=security,
                path=path,
            ):
                ip = extract_client_ip(request, settings=security)
                if not in_memory_limiter.allow(ip):
                    status_code = 429
                    return JSONResponse(
                        status_code=429,
                        content={
                            "detail": "rate limit exceeded",
                            "limit_per_minute": security.rate_limit_per_minute,
                        },
                    )

            if requires_auth(settings=security, method=method, path=path):
                if not security.jwt_secret:
                    status_code = 503
                    return auth_error_response(
                        "authentication is enabled but AUTH_JWT_SECRET is not configured",
                        status_code=503,
                    )
                if not is_strong_secret(security.jwt_secret):
                    status_code = 503
                    return auth_error_response(
                        "authentication is enabled but AUTH_JWT_SECRET is weak; set a strong secret (>=24 chars)",
                        status_code=503,
                    )
                token = extract_bearer_token(str(request.headers.get("Authorization", "") or ""))
                if not token:
                    status_code = 401
                    return auth_error_response("missing bearer token", status_code=401)
                try:
                    claims = verify_request_token(token, settings=security)
                except JWTError as exc:
                    status_code = 401
                    return auth_error_response(f"invalid token: {exc}", status_code=401)
                request.state.auth_claims = claims

            allowed, reason = control_plane.allow_request(method=method, path=path)
            if not allowed:
                status_code = 423
                return JSONResponse(
                    status_code=423,
                    content={
                        "detail": "request blocked by control plane",
                        "reason": reason,
                        "path": path,
                    },
                )

            response = await call_next(request)
            status_code = int(getattr(response, "status_code", 200))
            return response
        finally:
            metrics.mark_inflight(-1)
            metrics.record_request(
                method=method,
                path=path,
                status_code=status_code,
                latency_seconds=(perf_counter() - start),
                is_inference=is_inference_path(path),
            )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "autonomous-graph-workspace",
            "version": "2.0.0",
            "platform": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "security": {
                "auth_enabled": security.auth_enable,
                "rate_limit_enabled": security.rate_limit_enable,
                "rate_limit_backend": rate_limit_backend,
            },
            "control_plane": control_plane.snapshot(),
            "privacy_noise_enabled": privacy_noise.enabled(),
        }

    @app.get("/api/control/state")
    def control_state() -> dict[str, Any]:
        return control_plane.snapshot()

    @app.post("/api/control/update")
    def control_update(payload: ControlPlaneUpdateRequest, request: Request) -> dict[str, Any]:
        _assert_control_admin(request)
        data = payload.model_dump(exclude_none=True)
        if not data:
            return {
                "ok": True,
                "changed": {},
                "flags": control_plane.snapshot()["flags"],
                "updated_at": control_plane.snapshot()["updated_at"],
                "note": "no changes requested",
            }
        return control_plane.apply_patch(data)

    @app.post("/api/control/reload")
    def control_reload(request: Request) -> dict[str, Any]:
        _assert_control_admin(request)
        return control_plane.reload_from_env()

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics_endpoint() -> str:
        return metrics.render_prometheus(extra_metrics=privacy_noise.synthetic_metrics())

    @app.post("/api/auth/token")
    def auth_token(payload: AuthTokenRequest) -> dict[str, Any]:
        if not security.jwt_secret:
            raise HTTPException(
                status_code=503,
                detail="AUTH_JWT_SECRET is not configured",
            )
        if not is_strong_secret(security.jwt_secret):
            raise HTTPException(
                status_code=503,
                detail="AUTH_JWT_SECRET is weak; set a strong secret (>=24 chars)",
            )
        expected_user = str(os.getenv("AUTH_USER", "admin")).strip()
        expected_pass = str(os.getenv("AUTH_PASSWORD", "")).strip()
        if not expected_pass:
            raise HTTPException(
                status_code=503,
                detail="AUTH_PASSWORD is not configured",
            )
        if security.auth_enable and not is_strong_password(expected_pass):
            raise HTTPException(
                status_code=503,
                detail="AUTH_PASSWORD is weak; set a strong password (>=12 chars)",
            )
        if payload.username != expected_user or payload.password != expected_pass:
            raise HTTPException(status_code=401, detail="invalid credentials")
        token = create_access_token(
            subject=payload.username,
            secret=security.jwt_secret,
            issuer=security.jwt_issuer,
            audience=security.jwt_audience,
            expires_minutes=security.jwt_exp_minutes,
        )
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in_minutes": security.jwt_exp_minutes,
        }

    @app.get("/api/privacy/noise/report")
    def privacy_noise_report() -> dict[str, Any]:
        return privacy_noise.report()

    @app.get("/api/modules")
    def modules() -> dict[str, Any]:
        return {"modules": _scan_modules()}

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        payload = graph.snapshot_payload()
        return {
            "node_types": graph.list_node_types(),
            "metrics": payload["metrics"],
            "storage_adapter": type(graph.api.engine.graph_adapter).__name__
            if graph.api.engine.graph_adapter is not None
            else "in_memory",
        }

    @app.get("/api/graph/node-types")
    def graph_node_types() -> dict[str, Any]:
        return {"node_types": graph.list_node_types()}

    @app.get("/api/graph/snapshot")
    def graph_snapshot() -> dict[str, Any]:
        return graph.snapshot_payload()

    @app.get("/api/graph/events")
    def graph_events(
        limit: int = Query(default=200, ge=1, le=2000),
        event_type: str = Query(default=""),
    ) -> dict[str, Any]:
        return {"events": graph.list_events(limit=limit, event_type=event_type)}

    @app.websocket("/api/graph/ws")
    async def graph_events_ws(websocket: WebSocket) -> None:
        await event_hub.connect(websocket)
        try:
            snapshot_payload = graph.snapshot_payload()
            await websocket.send_json(
                {
                    "type": "hello",
                    "snapshot": snapshot_payload.get("snapshot", {}),
                    "metrics": snapshot_payload.get("metrics", {}),
                    "events": graph.list_events(limit=100),
                }
            )
            while True:
                message = await websocket.receive_text()
                if str(message or "").strip().lower() == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            return
        finally:
            await event_hub.disconnect(websocket)

    @app.post("/api/graph/node")
    def graph_create_node(payload: NodeCreateRequest) -> dict[str, Any]:
        try:
            return graph.create_node(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/graph/edge")
    def graph_create_edge(payload: EdgeCreateRequest) -> dict[str, Any]:
        try:
            return graph.create_edge(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/graph/node/update")
    def graph_update_node(payload: NodeUpdateRequest) -> dict[str, Any]:
        try:
            return graph.update_node(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/graph/node/delete")
    def graph_delete_node(payload: NodeDeleteRequest) -> dict[str, Any]:
        try:
            return graph.delete_node(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/graph/node/assist")
    def graph_node_assist(payload: GraphNodeAssistRequest) -> dict[str, Any]:
        try:
            return graph.project_graph_node_assist(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/graph/foundation/create")
    def graph_foundation_create(payload: GraphFoundationCreateRequest) -> dict[str, Any]:
        try:
            return graph.project_graph_foundation_create(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/graph/edge/assist")
    def graph_edge_assist(payload: GraphEdgeAssistRequest) -> dict[str, Any]:
        try:
            return graph.project_graph_edge_assist(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/graph/edge/update")
    def graph_update_edge(payload: EdgeUpdateRequest) -> dict[str, Any]:
        try:
            return graph.update_edge(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/graph/edge/delete")
    def graph_delete_edge(payload: EdgeDeleteRequest) -> dict[str, Any]:
        try:
            return graph.delete_edge(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/graph/simulate")
    def graph_simulate(payload: SimulateRequest) -> dict[str, Any]:
        try:
            return graph.simulate(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/graph/event/reward")
    def graph_reward_event(payload: EventRewardRequest) -> dict[str, Any]:
        try:
            return graph.reward_event(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/graph/relation/reinforce")
    def graph_reinforce_relation(payload: RelationReinforceRequest) -> dict[str, Any]:
        try:
            return graph.reinforce_relation(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/graph/persist")
    def graph_persist() -> dict[str, Any]:
        try:
            return graph.persist()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/graph/load")
    def graph_load() -> dict[str, Any]:
        try:
            return graph.load()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/graph/clear")
    def graph_clear() -> dict[str, Any]:
        try:
            return graph.clear()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/graph/seed-demo")
    def graph_seed_demo() -> dict[str, Any]:
        try:
            return graph.seed_demo()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/demo/watch")
    def project_demo_watch(payload: ProjectDemoWatchRequest) -> dict[str, Any]:
        try:
            return graph.watch_demo(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/daily-mode")
    def project_daily_mode(payload: ProjectDailyModeRequest, request: Request) -> dict[str, Any]:
        try:
            return graph.project_daily_mode(
                payload.model_dump(),
                request_headers=request.headers,
                request_ip=extract_client_ip(request, settings=security),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/llm/debate")
    def project_llm_debate(payload: ProjectLLMDebateRequest) -> dict[str, Any]:
        try:
            return graph.project_llm_debate(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/hallucination/report")
    def project_hallucination_report(payload: ProjectHallucinationReportRequest) -> dict[str, Any]:
        try:
            return graph.project_hallucination_report(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/hallucination/check")
    def project_hallucination_check(payload: ProjectHallucinationCheckRequest) -> dict[str, Any]:
        try:
            return graph.project_hallucination_check(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/archive/chat")
    def project_archive_chat(payload: ProjectArchiveVerifiedChatRequest) -> dict[str, Any]:
        try:
            return graph.project_archive_verified_chat(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/chat-graph")
    def project_chat_graph(payload: ProjectChatGraphRequest) -> dict[str, Any]:
        try:
            cognitive_actor = getattr(app.state, "cognitive_actor", None)
            graph_task_queue = getattr(app.state, "graph_task_queue", None)
            if cognitive_actor is not None and graph_task_queue is not None:
                from runtime.api.chat_api import execute_fast_chat_turn

                return execute_fast_chat_turn(
                    actor=cognitive_actor,
                    task_queue=graph_task_queue,
                    payload={
                        "message": payload.message,
                        "context": payload.context,
                        "language": "en",
                        "chat_model_role": payload.chat_model_role,
                        "llm_role": payload.chat_model_role,
                        "save_to_graph": payload.apply_to_graph,
                        "apply_to_graph": payload.apply_to_graph,
                        "use_internet": payload.use_internet,
                        "source_id": "",
                    },
                )
            if cognitive_actor is not None:
                try:
                    return cognitive_actor.ask(
                        "chat_graph",
                        {
                            "message": payload.message,
                            "context": payload.context,
                            "language": "en",
                            "chat_model_role": payload.chat_model_role,
                            "llm_role": payload.chat_model_role,
                            "save_to_graph": payload.apply_to_graph,
                            "apply_to_graph": payload.apply_to_graph,
                            "use_internet": payload.use_internet,
                            "source_id": "",
                        },
                        timeout=120.0,
                    )
                except TimeoutError as exc:
                    raise HTTPException(status_code=504, detail=str(exc)) from exc
            return graph.project_chat_graph(payload.model_dump())
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/project/wiki/random")
    def project_wikipedia_random(language: str = Query(default="ru")) -> dict[str, Any]:
        try:
            return fetch_random_wikipedia_article(language=language)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/project/wiki/search")
    def project_wikipedia_search(
        query: str = Query(default=""),
        language: str = Query(default="ru"),
        limit: int = Query(default=5, ge=1, le=8),
    ) -> dict[str, Any]:
        try:
            return search_wikipedia_articles(query, language=language, limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/mode-policy/resolve")
    def project_mode_policy_resolve(payload: ProjectModePolicyResolveRequest) -> dict[str, Any]:
        try:
            return graph.project_mode_policy_resolve(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/mode/save")
    def project_context_mode_save(payload: ProjectContextModeUpsertRequest) -> dict[str, Any]:
        try:
            return graph.project_context_mode_upsert(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/mode/focus")
    def project_context_mode_focus(payload: ProjectContextFocusCaptureRequest) -> dict[str, Any]:
        try:
            return graph.project_context_mode_capture_focus(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/mode/feedback")
    def project_context_mode_feedback(payload: ProjectContextModeFeedbackRequest) -> dict[str, Any]:
        try:
            return graph.project_context_mode_feedback(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/archive/review")
    def project_archive_review(payload: ProjectArchiveReviewApplyRequest) -> dict[str, Any]:
        try:
            return graph.project_archive_review_apply(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/user-graph/update")
    def project_user_graph_update(payload: ProjectUserGraphUpdateRequest, request: Request) -> dict[str, Any]:
        try:
            return graph.project_user_graph_update(
                payload.model_dump(),
                request_headers=request.headers,
                request_ip=extract_client_ip(request, settings=security),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/personal-tree/ingest")
    def project_personal_tree_ingest(payload: ProjectPersonalTreeIngestRequest) -> dict[str, Any]:
        try:
            return graph.project_personal_tree_ingest(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/personal-tree/note")
    def project_personal_tree_note(payload: ProjectPersonalTreeNoteRequest) -> dict[str, Any]:
        try:
            return graph.project_personal_tree_note(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/personal-tree/view")
    def project_personal_tree_view(payload: ProjectPersonalTreeViewRequest) -> dict[str, Any]:
        try:
            return graph.project_personal_tree_view(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/packages/manage")
    def project_packages_manage(payload: ProjectPackagesManageRequest) -> dict[str, Any]:
        try:
            return graph.project_packages_manage(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/memory/namespace/apply")
    def project_memory_namespace_apply(payload: ProjectMemoryNamespaceApplyRequest) -> dict[str, Any]:
        try:
            return graph.project_memory_namespace_apply(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/memory/namespace/view")
    def project_memory_namespace_view(payload: ProjectMemoryNamespaceViewRequest) -> dict[str, Any]:
        try:
            return graph.project_memory_namespace_view(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/graph-rag/query")
    def project_graph_rag_query(payload: ProjectGraphRagQueryRequest) -> dict[str, Any]:
        try:
            return graph.project_graph_rag_query(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/contradiction/scan")
    def project_contradiction_scan(payload: ProjectContradictionScanRequest) -> dict[str, Any]:
        try:
            return graph.project_contradiction_scan(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/task-risk/board")
    def project_task_risk_board(payload: ProjectTaskRiskBoardRequest) -> dict[str, Any]:
        try:
            return graph.project_task_risk_board(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/timeline/replay")
    def project_timeline_replay(payload: ProjectTimelineReplayRequest) -> dict[str, Any]:
        try:
            return graph.project_timeline_replay(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/project/llm-policy")
    def project_llm_policy() -> dict[str, Any]:
        try:
            return graph.project_llm_policy_get({})
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/llm-policy")
    def project_llm_policy_update(payload: ProjectLLMPolicyUpdateRequest) -> dict[str, Any]:
        try:
            return graph.project_llm_policy_update(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/quality/harness")
    def project_quality_harness(payload: ProjectQualityHarnessRequest) -> dict[str, Any]:
        try:
            return graph.project_quality_harness(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/backup/create")
    def project_backup_create(payload: ProjectBackupCreateRequest) -> dict[str, Any]:
        try:
            return graph.project_backup_create(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/backup/restore")
    def project_backup_restore(payload: ProjectBackupRestoreRequest) -> dict[str, Any]:
        try:
            return graph.project_backup_restore(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/audit/logs")
    def project_audit_logs(payload: ProjectAuditLogsRequest) -> dict[str, Any]:
        try:
            return graph.project_audit_logs(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/wrapper/respond")
    def project_wrapper_respond(payload: ProjectWrapperRespondRequest) -> dict[str, Any]:
        try:
            return graph.project_wrapper_respond(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/project/wrapper/profile")
    def project_wrapper_profile(user_id: str = Query(default="default_user")) -> dict[str, Any]:
        try:
            return graph.project_wrapper_profile_get({"user_id": user_id})
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/wrapper/profile")
    def project_wrapper_profile_update(payload: ProjectWrapperProfileUpdateRequest) -> dict[str, Any]:
        try:
            return graph.project_wrapper_profile_update(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/wrapper/feedback")
    def project_wrapper_feedback(payload: ProjectWrapperFeedbackRequest) -> dict[str, Any]:
        try:
            return graph.project_wrapper_feedback(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/integration/layer/manifest")
    def integration_layer_manifest(
        host: str = Query(default="generic"),
        app_id: str = Query(default="external_app"),
    ) -> dict[str, Any]:
        try:
            return graph.project_integration_layer_manifest({"host": host, "app_id": app_id})
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/integration/layer/invoke")
    def integration_layer_invoke(payload: IntegrationLayerInvokeRequest) -> dict[str, Any]:
        try:
            return graph.project_integration_layer_invoke(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/client/introspect")
    def client_introspect(payload: ClientIntrospectionRequest, request: Request) -> dict[str, Any]:
        try:
            return graph.capture_client_profile(
                payload.model_dump(),
                request_headers=request.headers,
                request_ip=extract_client_ip(request, settings=security),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/project/db/schema")
    def project_db_schema() -> dict[str, Any]:
        try:
            return graph.project_db_schema()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/graph/profile/prompt")
    def graph_profile_prompt(entity_type_hint: str = Query(default="human")) -> dict[str, Any]:
        return {
            "prompt_template": graph.profile_prompt_template(
                text="<put user narrative text here>",
                entity_type_hint=entity_type_hint,
            )
        }

    @app.post("/api/graph/profile/infer")
    def graph_profile_infer(payload: ProfileInferRequest) -> dict[str, Any]:
        try:
            return graph.infer_profile_from_text(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/living/architecture")
    def living_architecture() -> dict[str, Any]:
        try:
            return graph.living_architecture()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/living/health")
    def living_health() -> dict[str, Any]:
        try:
            return graph.living_health()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/living/process")
    def living_process(payload: LivingProcessRequest) -> dict[str, Any]:
        try:
            return graph.living_process(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/living/graph-view")
    def living_graph_view(user_id: str = Query(default="")) -> dict[str, Any]:
        try:
            return graph.living_graph_view({"user_id": user_id})
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/living/snapshot")
    def living_snapshot(payload: LivingSnapshotRequest) -> dict[str, Any]:
        try:
            return graph.living_snapshot(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/living/rollback")
    def living_rollback(payload: LivingRollbackRequest) -> dict[str, Any]:
        try:
            return graph.living_rollback(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/living/safe-mode")
    def living_safe_mode(payload: LivingSafeModeRequest) -> dict[str, Any]:
        try:
            return graph.living_safe_mode(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/living/human-override")
    def living_human_override(payload: LivingSafeModeRequest) -> dict[str, Any]:
        try:
            return graph.living_human_override(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/living/feedback")
    def living_feedback(payload: LivingFeedbackRequest) -> dict[str, Any]:
        try:
            return graph.living_feedback(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/living/evolution")
    def living_evolution() -> dict[str, Any]:
        try:
            return graph.living_evolution_plan()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/living/prompts")
    def living_prompts() -> dict[str, Any]:
        try:
            return graph.living_prompt_catalog()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/living/prompt/run")
    def living_prompt_run(payload: LivingPromptRunRequest) -> dict[str, Any]:
        try:
            return graph.living_prompt_run(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/living/project-map")
    def living_project_map(payload: LivingProjectMapRequest) -> dict[str, Any]:
        try:
            return graph.living_project_map(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/living/file/create")
    def living_file_create(payload: LivingFileRequest) -> dict[str, Any]:
        try:
            return graph.living_file_create(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/living/file/update")
    def living_file_update(payload: LivingFileRequest) -> dict[str, Any]:
        try:
            return graph.living_file_update(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/living/file/delete")
    def living_file_delete(payload: LivingFileRequest) -> dict[str, Any]:
        try:
            return graph.living_file_delete(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/living/knowledge/analyze")
    def living_knowledge_analyze(payload: KnowledgeAnalyzeRequest) -> dict[str, Any]:
        try:
            return graph.living_knowledge_analyze(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/living/knowledge/initialize")
    def living_knowledge_initialize(payload: KnowledgeInitializeRequest) -> dict[str, Any]:
        try:
            return graph.living_knowledge_initialize(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/living/knowledge/evaluate")
    def living_knowledge_evaluate(user_id: str = Query(default="")) -> dict[str, Any]:
        try:
            return graph.living_knowledge_evaluate({"user_id": user_id})
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/living/knowledge/branch")
    def living_knowledge_branch(payload: KnowledgeBranchRequest) -> dict[str, Any]:
        try:
            return graph.living_knowledge_branch(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/living/knowledge/merge")
    def living_knowledge_merge(payload: KnowledgeMergeRequest) -> dict[str, Any]:
        try:
            return graph.living_knowledge_merge(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/project/overview")
    def project_overview() -> dict[str, Any]:
        try:
            return graph.project_overview()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/project/model-advisors")
    def project_model_advisors() -> dict[str, Any]:
        try:
            return graph.project_model_advisors()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/pipeline")
    def project_pipeline(payload: ProjectPipelineRequest) -> dict[str, Any]:
        try:
            return graph.project_pipeline(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/bootstrap")
    def project_bootstrap(payload: ProjectBootstrapRequest) -> dict[str, Any]:
        try:
            return graph.project_bootstrap(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/project/evaluate")
    def project_evaluate(user_id: str = Query(default="")) -> dict[str, Any]:
        try:
            return graph.project_evaluate({"user_id": user_id})
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if include_frontend_routes:
        attach_frontend_routes(app)

    return app


app = create_app()
