from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .chat_engine import generate_response
from .file_ingestion import ingest_file, rebuild_artifacts, store_uploaded_file
from .graph_localizer import localized_node_view
from .observability import get_observability_store
from .graph_store import GraphStore
from .history_store import create_session, list_sessions, parse_session
from .reliability import RuntimeFailure, operator_messages_from_status, runtime_status_snapshot
from .runtime_config import get_runtime_config
from .node_rethinker import rethink_graph_nodes
from .persona_engine import (
    formalize_persona,
    list_persona_revisions,
    list_personas,
    load_persona,
    restore_persona_revision,
)
from .memory_layers import list_persona_snapshots


class SessionRequest(BaseModel):
    session_id: str = ''
    title: str = ''


class ChatRequest(BaseModel):
    session_id: str = ''
    message: str
    selected_persona: str = ''
    personality_name: str = ''
    explicit_context: str = ''
    language: str = 'en'


class UploadRequest(BaseModel):
    session_id: str
    filename: str
    content_base64: str = Field(description='Base64 encoded file bytes.')


class RebuildRequest(BaseModel):
    session_id: str
    personality_name: str = ''


class GraphNodeRequest(BaseModel):
    name: str
    node_type: str = 'CONCEPT'
    aliases: list[str] = Field(default_factory=list)
    description: str = ''
    facts: list[str] = Field(default_factory=list)
    translation_line: str = ''


class GraphEdgeRequest(BaseModel):
    from_id: str
    to_id: str
    relation_type: str = 'RELATED_TO'
    weight: float = 0.78
    confidence: float = 0.82


class GraphMergeRequest(BaseModel):
    primary_id: str
    secondary_id: str


class GraphNodeStateRequest(BaseModel):
    lifecycle_state: str
    reason: str = 'manual_review'


class GraphRethinkRequest(BaseModel):
    node_ids: list[str] = Field(default_factory=list)
    query: str = ''
    limit: int = 6
    context_budget: int = 4000
    active_mode: bool = False
    language: str = 'en'
    preview_only: bool = False


class GraphRestoreRequest(BaseModel):
    snapshot_path: str = ''


def _frontend_index_candidates() -> tuple[Path, Path, Path]:
    paths = get_runtime_config().paths
    return paths.webapp_assets_dir, paths.webapp_dist_index, paths.webapp_fallback_index


def _parse_upload_request(payload: Any) -> UploadRequest:
    if hasattr(UploadRequest, 'model_validate'):
        return UploadRequest.model_validate(payload)
    return UploadRequest.parse_obj(payload)


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get('/health')
    def health() -> dict[str, Any]:
        status = runtime_status_snapshot()
        return {
            'ok': True,
            'runtime': 'persona-graph-agent',
            'runtime_status': status,
            'operator_messages': operator_messages_from_status(status),
        }

    @router.get('/debug/metrics')
    def debug_metrics_endpoint() -> dict[str, Any]:
        return {'ok': True, 'metrics': get_observability_store().snapshot()}

    @router.get('/debug/traces')
    def debug_traces_endpoint(limit: int = 20, session_id: str = '') -> dict[str, Any]:
        return {
            'ok': True,
            'traces': get_observability_store().recent_traces(limit=max(1, min(int(limit or 20), 100)), session_id=session_id),
        }

    @router.get('/debug/graph-health')
    def debug_graph_health_endpoint() -> dict[str, Any]:
        return {'ok': True, 'graph_health': GraphStore().graph_diagnostics()}

    @router.get('/debug/runtime-status')
    def debug_runtime_status_endpoint() -> dict[str, Any]:
        status = runtime_status_snapshot()
        return {
            'ok': True,
            'runtime_status': status,
            'operator_messages': operator_messages_from_status(status),
        }

    @router.get('/sessions')
    def list_sessions_endpoint() -> dict[str, Any]:
        return {'sessions': list_sessions()}

    @router.post('/sessions')
    def create_session_endpoint(request: SessionRequest) -> dict[str, Any]:
        session = create_session(request.session_id, request.title)
        return {'session': session}

    @router.get('/sessions/{session_id}')
    def session_endpoint(session_id: str) -> dict[str, Any]:
        session = parse_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail='Session not found')
        return session

    @router.post('/chat/respond')
    def chat_endpoint(request: ChatRequest) -> dict[str, Any]:
        selected_persona = str(request.selected_persona or request.personality_name or '').strip()
        return generate_response(
            message=request.message,
            session_id=request.session_id,
            selected_persona=selected_persona,
            explicit_context=request.explicit_context,
            language=request.language,
        )

    @router.post('/files/upload')
    async def upload_endpoint(request: Request) -> dict[str, Any]:
        content_type = str(request.headers.get('content-type') or '').lower()
        if 'multipart/form-data' in content_type:
            form = await request.form()
            raw_session_id = str(form.get('session_id') or '').strip()
            session = create_session(raw_session_id)
            uploaded: list[dict[str, Any]] = []
            for item in form.getlist('files'):
                filename = str(getattr(item, 'filename', '') or '').strip()
                if not filename or not hasattr(item, 'read'):
                    continue
                content = await item.read()
                path = store_uploaded_file(session['session_id'], filename, content)
                uploaded.append({'path': str(path), 'result': ingest_file(path)})
            if not uploaded:
                raise HTTPException(status_code=400, detail='No files were uploaded')
            return {
                'session_id': session['session_id'],
                'files': uploaded,
            }

        try:
            payload = _parse_upload_request(await request.json())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f'Invalid upload payload: {exc}') from exc
        try:
            content = base64.b64decode(payload.content_base64.encode('utf-8'))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f'Invalid base64 payload: {exc}') from exc
        session = create_session(payload.session_id)
        path = store_uploaded_file(session['session_id'], payload.filename, content)
        result = ingest_file(path)
        return {
            'session_id': session['session_id'],
            'path': str(path),
            'result': result,
            'files': [{'path': str(path), 'result': result}],
        }

    @router.get('/graph')
    def graph_endpoint() -> dict[str, Any]:
        return GraphStore().load_graph()

    @router.get('/graph/snapshots')
    def graph_snapshots_endpoint(limit: int = 16) -> dict[str, Any]:
        return {'ok': True, 'snapshots': GraphStore().list_snapshots(limit=max(1, min(int(limit or 16), 64)))}

    @router.post('/graph/restore')
    def graph_restore_endpoint(request: GraphRestoreRequest) -> dict[str, Any]:
        result = GraphStore().restore_snapshot(request.snapshot_path)
        if not result.get('ok'):
            raise HTTPException(status_code=404, detail=result.get('reason') or 'snapshot_not_found')
        return {'ok': True, 'result': result, 'graph': GraphStore().load_graph()}

    @router.get('/graph/subgraph')
    def subgraph_endpoint(query: str = '', limit: int = 8) -> dict[str, Any]:
        default_limit = get_runtime_config().settings.graph_subgraph_limit
        return GraphStore().subgraph(query, limit=max(1, min(int(limit or default_limit), 64)), depth=1)

    @router.get('/graph/nodes/{node_id}/view')
    def graph_node_view_endpoint(node_id: str, language: str = 'en') -> dict[str, Any]:
        view = localized_node_view(node_id, language=language, store=GraphStore())
        if view is None:
            raise HTTPException(status_code=404, detail='node_not_found')
        return view

    @router.post('/graph/nodes')
    def create_graph_node_endpoint(request: GraphNodeRequest) -> dict[str, Any]:
        store = GraphStore()
        node = store.create_node(
            name=request.name,
            node_type=request.node_type,
            aliases=list(request.aliases or []),
            description=request.description,
            facts=list(request.facts or []),
            translation_line=request.translation_line,
        )
        return {'ok': True, 'node': node, 'graph': store.load_graph()}

    @router.delete('/graph/nodes/{node_id}')
    def delete_graph_node_endpoint(node_id: str) -> dict[str, Any]:
        store = GraphStore()
        result = store.delete_node(node_id)
        if not result.get('ok'):
            raise HTTPException(status_code=404, detail=result.get('reason') or 'node_not_found')
        return {'ok': True, 'result': result, 'graph': store.load_graph()}

    @router.post('/graph/edges')
    def create_graph_edge_endpoint(request: GraphEdgeRequest) -> dict[str, Any]:
        store = GraphStore()
        result = store.connect_nodes(
            from_id=request.from_id,
            to_id=request.to_id,
            relation_type=request.relation_type,
            weight=request.weight,
            confidence=request.confidence,
        )
        if not result.get('ok'):
            raise HTTPException(status_code=400, detail=result.get('reason') or 'connect_failed')
        return {'ok': True, 'result': result, 'graph': store.load_graph()}

    @router.delete('/graph/edges/{edge_id}')
    def delete_graph_edge_endpoint(edge_id: str) -> dict[str, Any]:
        store = GraphStore()
        result = store.delete_edge(edge_id=edge_id)
        if not result.get('ok'):
            raise HTTPException(status_code=404, detail=result.get('reason') or 'edge_not_found')
        return {'ok': True, 'result': result, 'graph': store.load_graph()}

    @router.post('/graph/nodes/merge')
    def merge_graph_nodes_endpoint(request: GraphMergeRequest) -> dict[str, Any]:
        store = GraphStore()
        result = store.merge_nodes_manual(primary_id=request.primary_id, secondary_id=request.secondary_id)
        if not result.get('ok'):
            raise HTTPException(status_code=400, detail=result.get('reason') or 'merge_failed')
        return {'ok': True, 'result': result, 'graph': store.load_graph()}

    @router.post('/graph/nodes/{node_id}/state')
    def review_graph_node_endpoint(node_id: str, request: GraphNodeStateRequest) -> dict[str, Any]:
        store = GraphStore()
        result = store.review_node_state(
            node_id,
            lifecycle_state=request.lifecycle_state,
            reason=request.reason,
        )
        if not result.get('ok'):
            raise HTTPException(status_code=400, detail=result.get('reason') or 'review_failed')
        return {'ok': True, 'result': result, 'graph': store.load_graph()}

    @router.post('/graph/rethink')
    def rethink_graph_endpoint(request: GraphRethinkRequest) -> dict[str, Any]:
        runtime = get_runtime_config().settings
        return rethink_graph_nodes(
            node_ids=list(request.node_ids or []),
            query=request.query,
            limit=max(1, min(int(request.limit or 6), 16)),
            context_budget=max(
                runtime.rethink_context_budget_min,
                min(int(request.context_budget or runtime.rethink_context_budget), runtime.rethink_context_budget_max),
            ),
            active_mode=bool(request.active_mode),
            language=request.language,
            preview_only=bool(request.preview_only),
        )

    @router.get('/personalities')
    def personalities_endpoint() -> dict[str, Any]:
        return {'personalities': list_personas()}

    @router.get('/personalities/{name}')
    def personality_endpoint(name: str) -> dict[str, Any]:
        bundle = load_persona(name)
        if bundle is None:
            raise HTTPException(status_code=404, detail='Head not found')
        model = formalize_persona(bundle)
        return {
            'name': bundle.name,
            'entity_type': bundle.entity_type,
            'traits': bundle.traits,
            'relations': bundle.relations,
            'examples': bundle.examples,
            'situation_reactions': bundle.situation_reactions,
            'knowledge': bundle.knowledge,
            'emotion_vector': bundle.emotion_vector,
            'formal_model': {
                'T': model.T,
                'E': model.E,
                'R': model.R,
                'M': model.M,
            },
            'triad': {
                'log_tuples': bundle.log_tuples,
                'persona_form': bundle.persona_form,
                'decision_explanation': bundle.decision_explanation,
            },
            'baseline': bundle.baseline_definition.to_dict() if bundle.baseline_definition is not None else {},
            'dynamic_state': bundle.dynamic_state.to_dict() if bundle.dynamic_state is not None else {},
            'learned_patterns': bundle.learned_patterns.to_dict() if bundle.learned_patterns is not None else {},
            'indicators': bundle.indicators.to_dict() if bundle.indicators is not None else {},
            'revisions': dict(bundle.revision_meta),
            'meta': bundle.meta,
        }

    @router.get('/personalities/{name}/revisions')
    def personality_revisions_endpoint(name: str) -> dict[str, Any]:
        bundle = load_persona(name)
        if bundle is None:
            raise HTTPException(status_code=404, detail='Head not found')
        return {'ok': True, 'revisions': list_persona_revisions(name)}

    @router.get('/personalities/{name}/snapshots')
    def personality_snapshots_endpoint(name: str, limit: int = 12) -> dict[str, Any]:
        bundle = load_persona(name)
        if bundle is None:
            raise HTTPException(status_code=404, detail='Head not found')
        return {'ok': True, 'snapshots': list_persona_snapshots(name, limit=max(1, min(int(limit or 12), 64)))}

    @router.post('/personalities/{name}/restore/{revision}')
    def personality_restore_endpoint(name: str, revision: int) -> dict[str, Any]:
        bundle = restore_persona_revision(name, revision)
        if bundle is None:
            raise HTTPException(status_code=404, detail='revision_not_found')
        model = formalize_persona(bundle)
        return {
            'ok': True,
            'name': bundle.name,
            'entity_type': bundle.entity_type,
            'emotion_vector': bundle.emotion_vector,
            'formal_model': {
                'T': model.T,
                'E': model.E,
                'R': model.R,
                'M': model.M,
            },
            'revisions': dict(bundle.revision_meta),
            'meta': bundle.meta,
        }

    @router.post('/rebuild')
    def rebuild_endpoint(request: RebuildRequest) -> dict[str, Any]:
        return rebuild_artifacts(request.session_id, personality_name=request.personality_name)

    return router


def create_app() -> FastAPI:
    app = FastAPI(title='Persona Graph Agent')
    app.include_router(create_router(), prefix='/api/cognitive')
    assets_dir, dist_index, fallback_index = _frontend_index_candidates()
    feature_flags = get_runtime_config().features

    @app.exception_handler(RuntimeFailure)
    async def runtime_failure_handler(_request: Request, exc: RuntimeFailure) -> JSONResponse:
        payload = exc.to_dict()
        return JSONResponse(
            status_code=exc.status_code,
            content={
                'ok': False,
                'error': payload,
            },
        )

    if feature_flags.enable_frontend_assets and assets_dir.exists() and not any(getattr(route, 'path', None) == '/assets' for route in app.routes):
        app.mount('/assets', StaticFiles(directory=assets_dir), name='assets')

    @app.get('/api/health')
    def api_health() -> dict[str, Any]:
        status = runtime_status_snapshot()
        return {
            'ok': True,
            'surface': 'combined',
            'runtime_status': status,
            'operator_messages': operator_messages_from_status(status),
        }

    @app.get('/')
    def root() -> Any:
        if feature_flags.enable_frontend_root and dist_index.exists():
            return FileResponse(dist_index)
        if feature_flags.enable_frontend_root and fallback_index.exists():
            return FileResponse(fallback_index)
        return {'ok': True, 'runtime': 'persona-graph-agent'}

    return app
