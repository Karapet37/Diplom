from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .chat_engine import generate_dialogue_response
from .context_builder import list_personalities, load_personality, load_personality_graph
from .graph_store import GraphStore
from .history_store import create_session, list_sessions, parse_session
from .knowledge_extractor import extract_file, rebuild, store_uploaded_file


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str = Field(default='')
    context: str = Field(default='')
    language: str = Field(default='ru')
    personality_name: str = Field(default='')
    user_id: str = Field(default='')


class SessionCreateRequest(BaseModel):
    session_id: str = Field(default='')
    title: str = Field(default='New session')


class RebuildRequest(BaseModel):
    session_id: str = Field(default='')
    personality_name: str = Field(default='')


router = APIRouter()


@router.get('/health')
def health() -> dict[str, Any]:
    return {'ok': True, 'runtime': 'mvp-file-first'}


@router.get('/sessions')
def get_sessions() -> dict[str, Any]:
    return {'sessions': list_sessions()}


@router.post('/sessions')
def post_sessions(payload: SessionCreateRequest) -> dict[str, Any]:
    session = create_session(payload.session_id, payload.title)
    return {'ok': True, 'session': session}


@router.get('/sessions/{session_id}')
def get_session(session_id: str) -> dict[str, Any]:
    session = parse_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail='Session not found')
    return session


@router.post('/chat/respond')
def post_chat_respond(payload: ChatRequest) -> dict[str, Any]:
    result = generate_dialogue_response(
        message=payload.message,
        session_id=payload.session_id,
        context=payload.context,
        language=payload.language,
        personality_name=payload.personality_name,
        user_id=payload.user_id,
    )
    return {
        'ok': True,
        'assistant_reply': result['assistant_reply'],
        'session_id': result['session_id'],
        'session': result['session'],
        'personality_name': result.get('personality_name') or '',
        'graph_context': result.get('graph_context') or '',
        'current_entity': result.get('current_entity') or '',
        'proposal_requested': bool(result.get('proposal_requested')),
    }


@router.post('/files/upload')
async def post_files_upload(session_id: str = Form(default=''), files: list[UploadFile] = File(default_factory=list)) -> dict[str, Any]:
    session = create_session(session_id or '')
    clean_session_id = str(session.get('session_id') or '')
    if not files:
        raise HTTPException(status_code=400, detail='No files supplied')
    stored_files: list[dict[str, Any]] = []
    extraction_results: list[dict[str, Any]] = []
    for upload in files:
        content = await upload.read()
        path = store_uploaded_file(clean_session_id, upload.filename or 'upload.txt', content)
        stored_files.append({'name': upload.filename or path.name, 'path': str(path), 'size': len(content)})
        extraction_results.append(extract_file(path))
    return {
        'ok': True,
        'session_id': clean_session_id,
        'stored_files': stored_files,
        'extraction_results': extraction_results,
    }


@router.post('/rebuild')
def post_rebuild(payload: RebuildRequest) -> dict[str, Any]:
    return rebuild(session_id=payload.session_id, personality_name=payload.personality_name)


@router.get('/graph')
def get_graph() -> dict[str, Any]:
    return GraphStore().load_graph()


@router.get('/graph/subgraph')
def get_graph_subgraph(query: str = '', limit: int = 12) -> dict[str, Any]:
    store = GraphStore()
    if not str(query or '').strip():
        graph = store.load_graph()
        graph['query'] = ''
        graph['seed_node_ids'] = []
        return graph
    return store.subgraph(query, limit=limit)


@router.get('/personalities')
def get_personalities() -> dict[str, Any]:
    return {'personalities': list_personalities()}


@router.get('/personalities/{name}')
def get_personality(name: str) -> dict[str, Any]:
    profile = load_personality(name)
    if not profile:
        raise HTTPException(status_code=404, detail='Personality not found')
    return {'name': name, 'profile': profile, 'graph': load_personality_graph(name)}


def create_router() -> APIRouter:
    return router


def create_app() -> FastAPI:
    app = FastAPI(title='Cognitive MVP API')
    app.include_router(router)
    return app


app = create_app()
