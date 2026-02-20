"""FastAPI endpoints for graph-first autonomous-system workspace."""

from __future__ import annotations

import os
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

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
    persona_name: str = "Alexa"
    narrative: str = ""
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


class ProjectUserGraphUpdateRequest(BaseModel):
    user_id: str = "default_user"
    display_name: str = "Default User"
    profile_text: str = ""
    profile: dict[str, Any] = Field(default_factory=dict)
    personality: dict[str, Any] = Field(default_factory=dict)
    fears: list[str] | str | None = None
    desires: list[str] | str | None = None
    goals: list[str] | str | None = None
    principles: list[str] | str | None = None
    opportunities: list[str] | str | None = None
    abilities: list[str] | str | None = None
    access: list[str] | str | None = None
    knowledge: list[str] | str | None = None
    assets: list[str] | str | None = None


class ProjectAutorunsImportRequest(BaseModel):
    text: str
    delimiter: str = ""
    user_id: str = "default_user"
    session_id: str = "autoruns_session"
    host_label: str = ""
    max_rows: int = 1000


class ClientIntrospectionRequest(BaseModel):
    session_id: str = ""
    user_id: str = ""
    client: dict[str, Any] = Field(default_factory=dict)


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


def create_app() -> FastAPI:
    graph = GraphWorkspaceService(use_env_adapter=True)
    security = SecuritySettings.from_env()
    metrics = RuntimeMetrics()
    privacy_noise = PrivacyNoisePlugin(PrivacyNoiseConfig.from_env())
    in_memory_limiter = InMemoryRateLimiter(security.rate_limit_per_minute)

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
            "privacy_noise_enabled": privacy_noise.enabled(),
        }

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
    def project_daily_mode(payload: ProjectDailyModeRequest) -> dict[str, Any]:
        try:
            return graph.project_daily_mode(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/user-graph/update")
    def project_user_graph_update(payload: ProjectUserGraphUpdateRequest) -> dict[str, Any]:
        try:
            return graph.project_user_graph_update(payload.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/autoruns/import")
    def project_autoruns_import(payload: ProjectAutorunsImportRequest) -> dict[str, Any]:
        try:
            return graph.project_autoruns_import(payload.model_dump())
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

        # Never return index.html for static asset requests.
        path_obj = Path(full_path)
        if full_path.startswith("assets/") or path_obj.suffix:
            raise HTTPException(status_code=404, detail="Asset not found")

        return _serve_index()

    return app


app = create_app()
