from __future__ import annotations

import base64
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .chat_engine import generate_response
from .file_ingestion import ingest_file, rebuild_artifacts, store_uploaded_file
from .graph_store import GraphStore
from .history_store import create_session
from .persona_engine import list_personas, load_persona


class SessionRequest(BaseModel):
    session_id: str = ''
    title: str = ''


class ChatRequest(BaseModel):
    session_id: str = ''
    message: str
    selected_persona: str = ''
    explicit_context: str = ''
    language: str = 'en'


class UploadRequest(BaseModel):
    session_id: str
    filename: str
    content_base64: str = Field(description='Base64 encoded file bytes.')


class RebuildRequest(BaseModel):
    session_id: str
    personality_name: str = ''


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get('/health')
    def health() -> dict[str, Any]:
        return {'ok': True, 'runtime': 'persona-graph-agent'}

    @router.post('/sessions')
    def create_session_endpoint(request: SessionRequest) -> dict[str, Any]:
        return create_session(request.session_id, request.title)

    @router.post('/chat/respond')
    def chat_endpoint(request: ChatRequest) -> dict[str, Any]:
        return generate_response(
            message=request.message,
            session_id=request.session_id,
            selected_persona=request.selected_persona,
            explicit_context=request.explicit_context,
            language=request.language,
        )

    @router.post('/files/upload')
    def upload_endpoint(request: UploadRequest) -> dict[str, Any]:
        try:
            content = base64.b64decode(request.content_base64.encode('utf-8'))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f'Invalid base64 payload: {exc}') from exc
        path = store_uploaded_file(request.session_id, request.filename, content)
        result = ingest_file(path)
        return {'path': str(path), 'result': result}

    @router.get('/graph')
    def graph_endpoint() -> dict[str, Any]:
        return GraphStore().load_graph()

    @router.get('/graph/subgraph')
    def subgraph_endpoint(query: str = '') -> dict[str, Any]:
        return GraphStore().subgraph(query, limit=8, depth=1)

    @router.get('/personalities')
    def personalities_endpoint() -> list[dict[str, Any]]:
        return list_personas()

    @router.get('/personalities/{name}')
    def personality_endpoint(name: str) -> dict[str, Any]:
        bundle = load_persona(name)
        if bundle is None:
            raise HTTPException(status_code=404, detail='Head not found')
        return {
            'name': bundle.name,
            'entity_type': bundle.entity_type,
            'traits': bundle.traits,
            'relations': bundle.relations,
            'examples': bundle.examples,
            'situation_reactions': bundle.situation_reactions,
            'knowledge': bundle.knowledge,
            'emotion_vector': bundle.emotion_vector,
            'meta': bundle.meta,
        }

    @router.post('/rebuild')
    def rebuild_endpoint(request: RebuildRequest) -> dict[str, Any]:
        return rebuild_artifacts(request.session_id, personality_name=request.personality_name)

    return router


def create_app() -> FastAPI:
    app = FastAPI(title='Persona Graph Agent')
    app.include_router(create_router(), prefix='/api/cognitive')

    @app.get('/api/health')
    def api_health() -> dict[str, Any]:
        return {'ok': True, 'surface': 'combined'}

    @app.get('/')
    def root() -> dict[str, Any]:
        return {'ok': True, 'runtime': 'persona-graph-agent'}

    return app
