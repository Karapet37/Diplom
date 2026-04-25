from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .chat_engine import forget_session_runtime_state, generate_response
from .file_ingestion import ingest_file, rebuild_artifacts, store_uploaded_file
from .graph_localizer import localized_node_view
from .observability import get_observability_store
from .graph_store import GraphStore
from .history_store import create_session, delete_session, list_sessions, parse_session
from .llm import prewarm_runtime_models_async
from .message_annotation_store import build_annotation_workspace, save_message_annotation
from .mood_research import analyze_mood_research, load_mood_report
from .reliability import RuntimeFailure, operator_messages_from_status, runtime_status_snapshot
from .runtime_config import get_runtime_config
from .node_rethinker import rethink_graph_nodes
from .persona_engine import (
    delete_persona_head,
    ensure_persona_registry_hygiene,
    explain_persona_graph,
    formalize_persona,
    list_persona_revisions,
    list_personas,
    load_active_persona,
    materialize_persona,
    restore_persona_revision,
    synthesize_persona_from_logs,
)
from .graph_store import normalize_personality_name
from .memory_layers import list_persona_snapshots
from .personality_schema import PersonalityObject
from .personality_store import (
    delete_personality,
    get_or_create_personality,
    list_biography_facts,
    list_conflicts,
    list_personalities,
    list_provenance,
    load_personality,
    load_personality_for_persona,
    save_personality,
)
from .personality_update_parser import (
    build_direct_biography_update,
    build_direct_profile_update,
    parse_document_candidates,
    parse_update_candidates,
)
from .personality_merge_engine import (
    apply_update_candidates,
    override_profile_field_entry,
    resolve_conflict,
)
from .personality_summarizer import (
    generate_compact_summary,
    generate_operator_label,
    update_hover_summary,
)
from .behavioral_action_engine import (
    ActionSelectionResult,
    CurrentStateModifier,
    PersonalityBehaviorModel,
    build_behavior_model,
    select_action,
    select_action_for_persona,
)
from .notes_store import (
    clear_notes,
    delete_note,
    execute_note_command,
    format_notes_for_context,
    load_notes,
    save_note,
)
from .planning_engine import (
    build_planning_output,
    classify_feedback,
    detect_planning_intent,
)
from .situation_regulator import (
    RegulatorState,
    classify_event,
    classify_event_multi,
    run_regulator_pipeline,
)
from .importance_learner import (
    build_importance_profile,
    load_examples,
    score_importance,
    suggest_important_turns,
)
from .training_examples_store import (
    add_example as add_training_example,
    count_examples as count_training_examples,
    delete_example as delete_training_example,
    export_jsonl as export_training_jsonl,
    get_example as get_training_example,
    list_examples as list_training_examples,
)


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
    user_persona_id: str = ''    # internal identifier (persona name slug)
    user_persona_name: str = ''  # display label sent to LLM prompt


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


class SafetyClassifyRequest(BaseModel):
    text: str
    language: str = 'en'
    k: int = 7


class SafetyExampleAddRequest(BaseModel):
    text: str
    label: str
    notes: str = ''


class TrainingExampleCreateRequest(BaseModel):
    input_text: str
    correct_output: str
    session_id: str = ''
    persona_name: str = ''
    notes: str = ''
    tags: list[str] = Field(default_factory=list)


class PersonaQuestionnaireRequest(BaseModel):
    # Identity — persona_name is the only required field
    persona_name: str
    display_name: str = ''
    age: str = ''
    gender: str = ''
    nationality: str = ''
    occupation: str = ''
    education: str = ''
    location: str = ''
    # Personality
    self_description: str = ''
    how_others_see_you: str = ''
    # Values
    core_values: str = ''
    life_philosophy: str = ''
    worldview: str = ''
    # Goals
    main_life_goals: str = ''
    dreams: str = ''
    what_drives_you: str = ''
    # Fears
    main_fears: str = ''
    insecurities: str = ''
    shame_topics: str = ''
    triggers: str = ''
    # Life story
    childhood: str = ''
    key_life_events: str = ''
    biggest_challenge: str = ''
    proudest_achievement: str = ''
    # Relationships
    family_description: str = ''
    friendship_style: str = ''
    romantic_patterns: str = ''
    # Communication
    communication_style: str = ''
    humor_style: str = ''
    conflict_approach: str = ''
    emotional_expression: str = ''
    # Habits
    hobbies: str = ''
    pet_peeves: str = ''
    guilty_pleasures: str = ''
    # Free form
    additional_context: str = ''
    fictional_basis: str = ''


class MessageAnnotationCreateRequest(BaseModel):
    message_id: str
    role: str
    raw_text: str = ''
    analysis_text: str = ''
    display_text: str = ''
    timestamp: str = ''
    persona_name: str = ''
    coordinates: dict[str, Any] = Field(default_factory=dict)
    context_window: list[str] = Field(default_factory=list)
    context_matrix: list[dict[str, Any]] = Field(default_factory=list)
    transition_interpretation: dict[str, Any] = Field(default_factory=dict)
    notes: str = ''


def _frontend_index_candidates() -> tuple[Path, Path, Path]:
    paths = get_runtime_config().paths
    return paths.webapp_assets_dir, paths.webapp_dist_index, paths.webapp_fallback_index


def _parse_upload_request(payload: Any) -> UploadRequest:
    validator = getattr(UploadRequest, 'model_validate', None)
    if callable(validator):
        return validator(payload)
    parser = getattr(UploadRequest, 'parse_obj', None)
    if callable(parser):
        return parser(payload)
    try:
        return UploadRequest(**dict(payload or {}))
    except Exception as exc:  # pragma: no cover - defensive compatibility path
        raise ValueError('UploadRequest validation is unavailable for the installed Pydantic version.') from exc


def _upload_response_payload(*, session_id: str, uploads: list[dict[str, Any]]) -> dict[str, Any]:
    primary = dict(uploads[0] if uploads else {})
    return {
        'session_id': session_id,
        'path': str(primary.get('path') or ''),
        'result': primary.get('result'),
        'files': [dict(item) for item in list(uploads or []) if isinstance(item, dict)],
    }


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

    @router.get('/sessions/{session_id}/annotation-workspace')
    def session_annotation_workspace_endpoint(session_id: str, window_size: int = 4) -> dict[str, Any]:
        session = parse_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail='Session not found')
        workspace = build_annotation_workspace(session_id, window_size=max(1, min(int(window_size or 4), 12)))
        return {'ok': True, 'workspace': workspace}

    @router.post('/sessions/{session_id}/annotations')
    def save_message_annotation_endpoint(session_id: str, body: MessageAnnotationCreateRequest) -> dict[str, Any]:
        session = parse_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail='Session not found')
        try:
            annotation = save_message_annotation(
                session_id=session_id,
                message_payload={
                    'message_id': body.message_id,
                    'role': body.role,
                    'raw_text': body.raw_text,
                    'analysis_text': body.analysis_text,
                    'display_text': body.display_text,
                    'timestamp': body.timestamp,
                    'persona_name': body.persona_name,
                },
                coordinates=body.coordinates,
                context_window=list(body.context_window or []),
                context_matrix=list(body.context_matrix or []),
                transition_interpretation=dict(body.transition_interpretation or {}),
                notes=body.notes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        workspace = build_annotation_workspace(session_id, window_size=max(len(body.context_window or []), 4))
        return {'ok': True, 'annotation': annotation, 'workspace': workspace}

    @router.delete('/sessions/{session_id}')
    def delete_session_endpoint(session_id: str) -> dict[str, Any]:
        result = delete_session(session_id)
        if not result.get('ok'):
            raise HTTPException(status_code=404, detail='Session not found')
        forget_session_runtime_state(session_id)
        return result

    @router.post('/chat/respond')
    def chat_endpoint(request: ChatRequest) -> dict[str, Any]:
        selected_persona = str(request.selected_persona or request.personality_name or '').strip()
        # Prefer display label; fall back to id if label not provided
        _user_persona_display = str(request.user_persona_name or request.user_persona_id or '').strip()
        return generate_response(
            message=request.message,
            session_id=request.session_id,
            selected_persona=selected_persona,
            explicit_context=request.explicit_context,
            language=request.language,
            user_persona_name=_user_persona_display,
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
            return _upload_response_payload(session_id=session['session_id'], uploads=uploaded)

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
        return _upload_response_payload(
            session_id=session['session_id'],
            uploads=[{'path': str(path), 'result': result}],
        )

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
        bundle = load_active_persona(name)
        if bundle is None:
            raise HTTPException(status_code=404, detail='Head not found')
        model = formalize_persona(bundle)
        graph_explanation = explain_persona_graph(name)
        mood_report = load_mood_report(persona_name=name) or analyze_mood_research(persona_name=name)
        return {
            'name': bundle.name,
            'label': bundle.structured_persona.identity.label if bundle.structured_persona is not None else bundle.name,
            'short_description': bundle.structured_persona.identity.short_description if bundle.structured_persona is not None else '',
            'readiness': bundle.structured_persona.identity.readiness if bundle.structured_persona is not None else '',
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
            'structured_persona': bundle.structured_persona.to_dict() if bundle.structured_persona is not None else {},
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
            'graph_explanation': graph_explanation.to_dict() if graph_explanation is not None else {},
            'mood_research': mood_report.to_dict(),
        }

    @router.get('/personalities/{name}/graph-explanation')
    def personality_graph_explanation_endpoint(name: str) -> dict[str, Any]:
        explanation = explain_persona_graph(name)
        if explanation is None:
            raise HTTPException(status_code=404, detail='Head not found')
        return {'ok': True, 'graph_explanation': explanation.to_dict()}

    @router.get('/personalities/{name}/revisions')
    def personality_revisions_endpoint(name: str) -> dict[str, Any]:
        bundle = load_active_persona(name)
        if bundle is None:
            raise HTTPException(status_code=404, detail='Head not found')
        return {'ok': True, 'revisions': list_persona_revisions(name)}

    @router.get('/personalities/{name}/snapshots')
    def personality_snapshots_endpoint(name: str, limit: int = 12) -> dict[str, Any]:
        bundle = load_active_persona(name)
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

    @router.get('/mood/research')
    def mood_research_endpoint(persona_name: str = '', session_id: str = '') -> dict[str, Any]:
        report = load_mood_report(persona_name=persona_name, session_id=session_id) or analyze_mood_research(
            persona_name=persona_name,
            session_id=session_id,
        )
        return {'ok': True, 'report': report.to_dict()}

    # -----------------------------------------------------------------------
    # Personality construction endpoints
    # -----------------------------------------------------------------------

    class PersonalityCreateRequest(BaseModel):
        persona_name: str = ''
        label: str = ''
        source_texts: list[str] = Field(default_factory=list)

    class PersonalityUpdateFromTextRequest(BaseModel):
        personality_id: str = ''
        persona_name: str = ''
        text: str
        source_channel: str = 'chat_user'
        session_id: str = ''

    class PersonalityUpdateFromDocRequest(BaseModel):
        personality_id: str = ''
        persona_name: str = ''
        text: str
        session_id: str = ''

    class PersonalityDirectFieldUpdateRequest(BaseModel):
        field_name: str
        value: str
        confidence: float = 0.90
        source_channel: str = 'manual_edit'
        session_id: str = ''
        evidence_excerpt: str = ''

    class PersonalityDirectBioUpdateRequest(BaseModel):
        fact_type: str = 'factual_statement'
        value: str
        confidence: float = 0.90
        source_channel: str = 'manual_edit'
        session_id: str = ''
        evidence_excerpt: str = ''

    class PersonalityFieldOverrideRequest(BaseModel):
        field_name: str
        entry_id: str
        new_value: str
        confidence: float = 0.90
        source_channel: str = 'manual_edit'
        session_id: str = ''

    class PersonalityConflictResolveRequest(BaseModel):
        conflict_id: str
        resolution: str
        keep_entry_id: str = ''

    class ActionScoreRequest(BaseModel):
        persona_name: str = ''
        personality_id: str = ''
        route: str = 'persona_chat'
        situation_type: str = 'neutral_statement'
        pressure: float = 0.0
        shame_active: bool = False
        trust_level: float = 0.5

    def _load_personality_or_404(personality_id: str = '', persona_name: str = '') -> PersonalityObject:
        if personality_id:
            # Try direct ID lookup first, then treat the value as a persona name (index lookup)
            p = load_personality(personality_id) or load_personality_for_persona(personality_id)
        elif persona_name:
            p = load_personality_for_persona(persona_name)
        else:
            p = None
        if p is None:
            raise HTTPException(status_code=404, detail='Personality not found')
        return p

    @router.get('/personality-construction/personalities')
    def list_personalities_endpoint() -> dict[str, Any]:
        return {'ok': True, 'personalities': list_personalities()}

    @router.post('/personality-construction/personalities')
    def create_personality_endpoint(request: PersonalityCreateRequest) -> dict[str, Any]:
        personality = get_or_create_personality(
            request.persona_name,
            label=request.label,
            source_texts=list(request.source_texts or []),
        )
        update_hover_summary(personality)
        save_personality(personality)
        return {'ok': True, 'personality': generate_compact_summary(personality)}

    @router.post('/personality-construction/personalities/from-questionnaire')
    def create_persona_from_questionnaire_endpoint(body: PersonaQuestionnaireRequest) -> dict[str, Any]:
        """
        Create a new persona from a structured questionnaire.
        Questionnaire answers are formatted as text, analyzed by LLM,
        and merged into a full PersonalityObject + runtime head.
        """
        from .prompt_builder import format_questionnaire_as_text
        from .reliability import MutationRejectedFailure

        name = str(body.display_name or body.persona_name).strip()
        if not name:
            raise HTTPException(status_code=400, detail='persona_name is required')

        excerpt = format_questionnaire_as_text(body.model_dump())
        if not excerpt.strip():
            raise HTTPException(status_code=400, detail='Questionnaire has no content to synthesize')

        # LLM synthesis → runtime persona head
        try:
            payload = synthesize_persona_from_logs(
                name, [excerpt],
                reason='Created from identity questionnaire.',
            )
            bundle = materialize_persona(name, payload, explicit=True)
        except MutationRejectedFailure as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Store structured biography facts in the personality store
        slug = normalize_personality_name(bundle.name)
        personality = get_or_create_personality(slug, label=name, source_texts=[excerpt])

        bio_fields: list[tuple[str, str]] = [
            ('occupation', 'occupation'),
            ('education', 'education'),
            ('location', 'location'),
            ('family_description', 'family'),
            ('childhood', 'life_event'),
            ('key_life_events', 'life_event'),
            ('biggest_challenge', 'life_event'),
            ('proudest_achievement', 'achievement'),
        ]
        bio_candidates = [
            build_direct_biography_update(fact_type, val, source_channel='questionnaire')
            for attr, fact_type in bio_fields
            for val in [str(getattr(body, attr, '') or '').strip()]
            if val
        ]
        if bio_candidates:
            apply_update_candidates(personality, bio_candidates)

        update_hover_summary(personality)
        save_personality(personality)

        # Resolve persona summary from runtime list
        persona_summary = next(
            (p for p in list_personas() if p.get('name') == slug or str(p.get('label') or '').lower() == name.lower()),
            {'name': slug, 'label': name},
        )
        return {
            'ok': True,
            'persona_name': bundle.name,
            'persona_slug': slug,
            'persona_summary': persona_summary,
            'personality': generate_compact_summary(personality),
        }

    @router.get('/personality-construction/personalities/{personality_id}')
    def get_personality_endpoint(personality_id: str) -> dict[str, Any]:
        personality = _load_personality_or_404(personality_id=personality_id)
        return {
            'ok': True,
            'personality': personality.to_dict(),
            'summary': generate_compact_summary(personality),
            'label': generate_operator_label(personality),
        }

    @router.delete('/personality-construction/personalities/{personality_id}')
    def delete_personality_endpoint(personality_id: str) -> dict[str, Any]:
        # Try PersonalityObject store first (biography/traits system)
        ok = delete_personality(personality_id)
        # Also try persona head store (runtime heads system) — this is where most personas live
        ok = delete_persona_head(personality_id) or ok
        if not ok:
            raise HTTPException(status_code=404, detail='Personality not found')
        return {'ok': True}

    @router.post('/personality-construction/personalities/{personality_id}/update-from-text')
    def update_personality_from_text_endpoint(
        personality_id: str, request: PersonalityUpdateFromTextRequest
    ) -> dict[str, Any]:
        personality = _load_personality_or_404(personality_id=personality_id)
        candidates = parse_update_candidates(
            request.text,
            source_channel=request.source_channel,
            session_id=request.session_id,
        )
        result = apply_update_candidates(personality, candidates)
        update_hover_summary(personality)
        save_personality(personality)
        return {'ok': True, 'update_result': result.to_dict(), 'summary': generate_compact_summary(personality)}

    @router.post('/personality-construction/personalities/{personality_id}/update-from-document')
    def update_personality_from_doc_endpoint(
        personality_id: str, request: PersonalityUpdateFromDocRequest
    ) -> dict[str, Any]:
        personality = _load_personality_or_404(personality_id=personality_id)
        candidates = parse_document_candidates(request.text, session_id=request.session_id)
        result = apply_update_candidates(personality, candidates)
        update_hover_summary(personality)
        save_personality(personality)
        return {'ok': True, 'update_result': result.to_dict(), 'summary': generate_compact_summary(personality)}

    @router.post('/personality-construction/personalities/{personality_id}/update-field')
    def update_personality_field_endpoint(
        personality_id: str, request: PersonalityDirectFieldUpdateRequest
    ) -> dict[str, Any]:
        personality = _load_personality_or_404(personality_id=personality_id)
        candidate = build_direct_profile_update(
            request.field_name,
            request.value,
            source_channel=request.source_channel,
            session_id=request.session_id,
            confidence=request.confidence,
            evidence_excerpt=request.evidence_excerpt,
        )
        result = apply_update_candidates(personality, [candidate])
        update_hover_summary(personality)
        save_personality(personality)
        return {'ok': True, 'update_result': result.to_dict()}

    @router.post('/personality-construction/personalities/{personality_id}/update-biography')
    def update_personality_biography_endpoint(
        personality_id: str, request: PersonalityDirectBioUpdateRequest
    ) -> dict[str, Any]:
        personality = _load_personality_or_404(personality_id=personality_id)
        candidate = build_direct_biography_update(
            request.fact_type,
            request.value,
            source_channel=request.source_channel,
            session_id=request.session_id,
            confidence=request.confidence,
            evidence_excerpt=request.evidence_excerpt,
        )
        result = apply_update_candidates(personality, [candidate])
        update_hover_summary(personality)
        save_personality(personality)
        return {'ok': True, 'update_result': result.to_dict()}

    @router.post('/personality-construction/personalities/{personality_id}/override-entry')
    def override_personality_entry_endpoint(
        personality_id: str, request: PersonalityFieldOverrideRequest
    ) -> dict[str, Any]:
        personality = _load_personality_or_404(personality_id=personality_id)
        ok = override_profile_field_entry(
            personality,
            request.field_name,
            request.entry_id,
            request.new_value,
            source_channel=request.source_channel,
            session_id=request.session_id,
            confidence=request.confidence,
        )
        if not ok:
            raise HTTPException(status_code=404, detail='Entry not found')
        update_hover_summary(personality)
        save_personality(personality)
        return {'ok': True}

    @router.get('/personality-construction/personalities/{personality_id}/biography')
    def personality_biography_endpoint(personality_id: str, fact_type: str = '') -> dict[str, Any]:
        facts = list_biography_facts(personality_id, fact_type=fact_type)
        return {'ok': True, 'facts': facts}

    @router.get('/personality-construction/personalities/{personality_id}/provenance')
    def personality_provenance_endpoint(personality_id: str) -> dict[str, Any]:
        return {'ok': True, 'provenance': list_provenance(personality_id)}

    @router.get('/personality-construction/personalities/{personality_id}/conflicts')
    def personality_conflicts_endpoint(personality_id: str) -> dict[str, Any]:
        return {'ok': True, 'conflicts': list_conflicts(personality_id)}

    @router.post('/personality-construction/personalities/{personality_id}/resolve-conflict')
    def personality_resolve_conflict_endpoint(
        personality_id: str, request: PersonalityConflictResolveRequest
    ) -> dict[str, Any]:
        personality = _load_personality_or_404(personality_id=personality_id)
        ok = resolve_conflict(
            personality,
            request.conflict_id,
            request.resolution,
            keep_entry_id=request.keep_entry_id,
        )
        if not ok:
            raise HTTPException(status_code=404, detail='Conflict not found')
        save_personality(personality)
        return {'ok': True}

    # -----------------------------------------------------------------------
    # Behavioral action scoring endpoint
    # -----------------------------------------------------------------------

    @router.post('/personality-construction/score-actions')
    def score_actions_endpoint(request: ActionScoreRequest) -> dict[str, Any]:
        """
        Score behavioral actions for a persona given current context.

        Returns the selected action and full scoring breakdown — letting the
        operator see exactly why the system chose a particular behavior.
        """
        personality = _load_personality_or_404(
            personality_id=request.personality_id,
            persona_name=request.persona_name,
        )
        modifiers: list[CurrentStateModifier] = []
        if request.pressure > 0.0:
            from .behavioral_action_engine import pressure_modifier
            modifiers.append(pressure_modifier(request.pressure))
        if request.shame_active:
            from .behavioral_action_engine import shame_activation_modifier
            modifiers.append(shame_activation_modifier())
        if request.trust_level != 0.5:
            from .behavioral_action_engine import trust_modifier
            modifiers.append(trust_modifier(request.trust_level))

        result = select_action_for_persona(
            personality,
            request.route,
            situation_type=request.situation_type,
            current_state_modifiers=modifiers,
        )
        return {'ok': True, 'action_selection': result.to_dict()}

    @router.get('/personality-construction/personalities/{personality_id}/behavior-model')
    def personality_behavior_model_endpoint(personality_id: str) -> dict[str, Any]:
        """Inspect the behavioral model derived from a personality."""
        personality = _load_personality_or_404(personality_id=personality_id)
        model = build_behavior_model(personality)
        return {'ok': True, 'behavior_model': model.to_dict()}

    # -----------------------------------------------------------------------
    # Notes API — user-controlled memory layer
    # -----------------------------------------------------------------------

    @router.get('/sessions/{session_id}/notes')
    def get_notes_endpoint(session_id: str) -> dict[str, Any]:
        """List all saved notes for a session."""
        notes = load_notes(session_id)
        return {
            'ok': True,
            'session_id': session_id,
            'notes': notes,
            'count': len(notes),
        }

    class SaveNoteRequest(BaseModel):
        text: str
        source: str = 'manual'

    @router.post('/sessions/{session_id}/notes')
    def save_note_endpoint(session_id: str, body: SaveNoteRequest) -> dict[str, Any]:
        """Save a new note for a session."""
        note = save_note(session_id, body.text, source=body.source)
        if not note:
            raise HTTPException(status_code=400, detail='Note text is required.')
        return {'ok': True, 'note': note}

    @router.delete('/sessions/{session_id}/notes/{note_ref}')
    def delete_note_endpoint(session_id: str, note_ref: str) -> dict[str, Any]:
        """Delete a note by note_id or 1-based index."""
        deleted = delete_note(session_id, note_ref)
        return {'ok': deleted, 'session_id': session_id, 'deleted': deleted}

    @router.delete('/sessions/{session_id}/notes')
    def clear_notes_endpoint(session_id: str) -> dict[str, Any]:
        """Delete all notes for a session."""
        count = clear_notes(session_id)
        return {'ok': True, 'session_id': session_id, 'deleted_count': count}

    @router.get('/sessions/{session_id}/notes/context')
    def notes_context_endpoint(session_id: str, max_notes: int = 20) -> dict[str, Any]:
        """Get saved notes formatted for context injection."""
        context = format_notes_for_context(session_id, max_notes=max_notes)
        return {'ok': True, 'session_id': session_id, 'context': context}

    # -----------------------------------------------------------------------
    # Planning API — planning mode output
    # -----------------------------------------------------------------------

    class PlanningRequest(BaseModel):
        session_id: str
        message: str
        personality_name: str = ''
        language: str = 'en'
        recent_feedback: list[str] = Field(default_factory=list)

    @router.post('/planning/analyze')
    def planning_analyze_endpoint(body: PlanningRequest) -> dict[str, Any]:
        """
        Build a planning output for a message.
        Returns structured planning data (what_matters, forces, steps, follow-up).
        Does NOT call the LLM — returns raw structure only.
        """
        output = build_planning_output(
            message=body.message,
            session_id=body.session_id,
            personality_name=body.personality_name,
            recent_feedback=list(body.recent_feedback or []),
            language=body.language,
        )
        return {'ok': True, 'planning': output.to_dict()}

    @router.post('/planning/classify-feedback')
    def classify_feedback_endpoint(body: dict[str, Any]) -> dict[str, Any]:
        """Classify a user message as feedback on a previous step."""
        message = str(body.get('message') or '')
        feedback_class = classify_feedback(message)
        return {'ok': True, 'message': message, 'feedback_class': feedback_class}

    # -----------------------------------------------------------------------
    # Situation Regulator API — event/regulator/action pipeline
    # -----------------------------------------------------------------------

    class RegulatorRequest(BaseModel):
        message: str
        personality_name: str = ''
        overload_level: float = 0.0
        feedback_class: str = 'neutral'

    @router.post('/regulator/analyze')
    def regulator_analyze_endpoint(body: RegulatorRequest) -> dict[str, Any]:
        """
        Run the event → regulator → action pipeline on a message.
        Returns event classification, regulator state, and selected action.
        """
        pressure_response: list[str] = []
        vulnerabilities: list[str] = []
        defense_mechanisms: list[str] = []
        if body.personality_name:
            personality = load_personality_for_persona(body.personality_name)
            if personality is not None:
                pressure_response = [
                    e.value for e in personality.profile.get_field('pressure_response')
                    if not e.is_temporary
                ][:5]
                vulnerabilities = [
                    e.value for e in personality.profile.get_field('vulnerabilities')
                    if not e.is_temporary
                ][:5]
                defense_mechanisms = [
                    e.value for e in personality.profile.get_field('defense_mechanisms')
                    if not e.is_temporary
                ][:5]
        output = run_regulator_pipeline(
            message=body.message,
            overload_level=body.overload_level,
            feedback_class=body.feedback_class,
            personality_pressure_response=pressure_response,
            personality_vulnerabilities=vulnerabilities,
            personality_defense_mechanisms=defense_mechanisms,
        )
        return {'ok': True, 'regulator': output.to_dict()}

    @router.post('/regulator/classify-event')
    def classify_event_endpoint(body: dict[str, Any]) -> dict[str, Any]:
        """Classify the primary event type from a message."""
        message = str(body.get('message') or '')
        primary = classify_event(message)
        multi = classify_event_multi(message)
        return {'ok': True, 'message': message, 'event_type': primary, 'event_multi': multi}

    # -----------------------------------------------------------------------
    # Importance Learner API
    # -----------------------------------------------------------------------

    @router.get('/sessions/{session_id}/importance-profile')
    def importance_profile_endpoint(session_id: str) -> dict[str, Any]:
        """Get the importance profile for a session (what the user tends to save)."""
        profile = build_importance_profile(session_id)
        return {'ok': True, 'session_id': session_id, 'importance_profile': profile}

    @router.post('/sessions/{session_id}/importance/score')
    def score_importance_endpoint(session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Score the likely importance of a text snippet."""
        text = str(body.get('text') or '')
        score = score_importance(text)
        return {'ok': True, 'text': text[:200], 'score': score}

    @router.post('/sessions/{session_id}/importance/suggest')
    def suggest_important_endpoint(session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """
        Given a list of recent turn texts, suggest which ones are most worth saving.
        """
        turns = [str(t) for t in list(body.get('turns') or []) if t]
        threshold = float(body.get('threshold') or 0.45)
        top_n = int(body.get('top_n') or 3)
        suggestions = suggest_important_turns(turns, threshold=threshold, top_n=top_n)
        return {'ok': True, 'suggestions': suggestions}

    # -----------------------------------------------------------------------
    # Training Examples API — fine-tuning dataset curation
    # -----------------------------------------------------------------------

    @router.post('/training-examples')
    def create_training_example_endpoint(body: TrainingExampleCreateRequest) -> dict[str, Any]:
        """
        Save a (input, correct_output) pair as a training example.

        Use this to curate preferred responses for fine-tuning.
        """
        try:
            example = add_training_example(
                input_text=body.input_text,
                correct_output=body.correct_output,
                session_id=body.session_id,
                persona_name=body.persona_name,
                notes=body.notes,
                tags=list(body.tags or []),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Train per-module response evaluators (P1-P49) when a correction is saved
        if body.persona_name and body.input_text and body.correct_output:
            try:
                original = ''
                for note in (body.notes or '').splitlines():
                    if note.startswith('original:'):
                        original = note[len('original:'):].strip()
                        break
                # Run input analysis to get module signals, then train response evaluator
                from agent_system.cognitive_modules_v2 import get_persona_response_evaluator
                from agent_system.cognitive_modules_v2 import CognitiveRuntimeV2 as CognitiveModuleSystem
                cms = CognitiveModuleSystem()
                signals, _ = cms.forward(body.input_text)
                get_persona_response_evaluator().record_correction(
                    persona_name=body.persona_name,
                    signals=signals,
                    user_input=body.input_text,
                    original_reply=original,
                    corrected_reply=body.correct_output,
                )
                # Also update the monolithic fallback classifier
                from agent_system.response_coherence_classifier import get_coherence_checker
                get_coherence_checker().record_correction(
                    persona_name=body.persona_name,
                    user_input=body.input_text,
                    original_reply=original,
                    corrected_reply=body.correct_output,
                )
            except Exception:
                pass
        return {'ok': True, 'example': example}

    @router.get('/training-examples')
    def list_training_examples_endpoint(
        session_id: str = '',
        persona_name: str = '',
    ) -> dict[str, Any]:
        """List training examples, optionally filtered by session or persona."""
        examples = list_training_examples(session_id=session_id, persona_name=persona_name)
        return {
            'ok': True,
            'examples': examples,
            'count': len(examples),
        }

    @router.get('/training-examples/{example_id}')
    def get_training_example_endpoint(example_id: str) -> dict[str, Any]:
        """Get a single training example by ID."""
        example = get_training_example(example_id)
        if example is None:
            raise HTTPException(status_code=404, detail='Training example not found')
        return {'ok': True, 'example': example}

    @router.delete('/training-examples/{example_id}')
    def delete_training_example_endpoint(example_id: str) -> dict[str, Any]:
        """Delete a training example by ID."""
        deleted = delete_training_example(example_id)
        if not deleted:
            raise HTTPException(status_code=404, detail='Training example not found')
        return {'ok': True, 'deleted': True}

    @router.get('/training-examples/export/jsonl')
    def export_training_examples_endpoint(
        session_id: str = '',
        persona_name: str = '',
        fmt: str = 'openai',
    ) -> dict[str, Any]:
        """
        Export training examples as JSONL string.

        fmt: 'openai' (messages format) or 'simple' (prompt/completion format).
        Returns the JSONL content and example count.
        """
        if fmt not in ('openai', 'simple'):
            raise HTTPException(status_code=400, detail="fmt must be 'openai' or 'simple'")
        content, count = export_training_jsonl(
            session_id=session_id,
            persona_name=persona_name,
            fmt=fmt,
        )
        return {'ok': True, 'jsonl': content, 'count': count, 'fmt': fmt}

    # ---------------------------------------------------------------------------
    # Safety classifier
    # ---------------------------------------------------------------------------

    @router.post('/safety/classify')
    def safety_classify_endpoint(body: SafetyClassifyRequest) -> dict[str, Any]:
        """
        Classify a message using the deterministic KNN safety classifier.
        Returns label (safe/suggestive/explicit/illegal), confidence, action,
        matched features, and top-3 nearest neighbors.
        """
        from .safety_classifier import classify as safety_classify
        result = safety_classify(str(body.text or ''), k=max(1, min(20, body.k)))
        return {
            'ok': True,
            'label': result.label,
            'confidence': result.confidence,
            'action': result.action,
            'matched_features': result.matched_features,
            'top_neighbors': result.top_neighbors,
            'fast_path': result.fast_path,
        }

    @router.post('/safety/examples')
    def safety_add_example_endpoint(body: SafetyExampleAddRequest) -> dict[str, Any]:
        """Add a labelled example to the safety classifier's runtime database."""
        from .safety_classifier import add_example as safety_add_example
        try:
            safety_add_example(
                text=str(body.text or ''),
                label=str(body.label or ''),
                notes=str(body.notes or ''),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {'ok': True}

    # ---------------------------------------------------------------------------
    # Cognitive genome endpoints
    # ---------------------------------------------------------------------------

    class GenomeFeedbackRequest(BaseModel):
        session_id:    str
        turn_id:       str
        feedback_type: str   # too_cold / too_warm / underestimated_fear / etc.
        intensity:     float = 0.7
        feedback:      float = -1.0   # +1 success -1 failure

    class GenomeCalibrateRequest(BaseModel):
        persona_id: str = ''

    @router.get('/genome/{persona_id}')
    def genome_get_endpoint(persona_id: str) -> dict[str, Any]:
        """Return current genome params for a persona."""
        from .genome_store import GenomeStore
        cfg = get_runtime_config()
        store = GenomeStore(Path(cfg.memory_root))
        genome = store.load_genome(str(persona_id or '').strip() or 'default')
        return {'ok': True, 'persona_id': genome.persona_id, 'version': genome.version,
                'params': {p.name: {'value': round(p.value, 4), 'prior': round(p.prior, 4),
                                    'stability': p.stability, 'updates': p.updates}
                           for p in genome.all_params()}}

    @router.post('/genome/feedback')
    def genome_feedback_endpoint(body: GenomeFeedbackRequest) -> dict[str, Any]:
        """Apply explicit feedback to genome and update training store."""
        import numpy as _np
        from .genome_store import GenomeStore
        from .training_store import TrainingStore
        from .perceptron_trainer import apply_explicit_feedback
        cfg = get_runtime_config()
        store = GenomeStore(Path(cfg.memory_root))
        genome = store.load_genome('default')
        updated = apply_explicit_feedback(
            str(body.feedback_type or ''),
            float(_np.clip(body.intensity, 0.0, 1.0)),
            genome,
        )
        store.save_genome(genome)
        ts = TrainingStore(Path(cfg.memory_root) / 'training_tuples.jsonl')
        ts.apply_feedback(str(body.turn_id or ''), float(body.feedback), str(body.feedback_type or ''))
        return {'ok': True, 'updated_params': updated, 'genome_version': genome.version}

    @router.post('/genome/calibrate')
    def genome_calibrate_endpoint(body: GenomeCalibrateRequest) -> dict[str, Any]:
        """Run offline perceptron calibration on stored training data."""
        from .genome_store import GenomeStore
        from .training_store import TrainingStore
        from .perceptron_trainer import run_batch_calibration
        cfg = get_runtime_config()
        gs = GenomeStore(Path(cfg.memory_root))
        rt = gs.load_runtime()
        ts = TrainingStore(Path(cfg.memory_root) / 'training_tuples.jsonl')
        persona_ids = gs.list_genomes() or ['default']
        genome_map = {pid: gs.load_genome(pid) for pid in persona_ids}
        result = run_batch_calibration(rt, ts, genome_map)
        gs.save_runtime(rt)
        return {'ok': True, 'result': result}

    @router.get('/genome/training-stats')
    def genome_training_stats_endpoint() -> dict[str, Any]:
        from .training_store import TrainingStore
        cfg = get_runtime_config()
        ts = TrainingStore(Path(cfg.memory_root) / 'training_tuples.jsonl')
        return {'ok': True, 'total_tuples': ts.count()}

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

    @app.on_event('startup')
    async def _startup_prewarm() -> None:
        ensure_persona_registry_hygiene()
        prewarm_runtime_models_async()

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
