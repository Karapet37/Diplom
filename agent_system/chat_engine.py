from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

from .classifier_forest import DEFAULT_CLASSIFIER
from .context_builder import build_context
from .feature_extractor import extract_features
from .file_ingestion import rebuild_artifacts
from .graph_store import GraphStore, normalize_personality_name
from .head_caller import prepare_heads, select_primary_head
from .history_store import append_turn, create_session, infer_current_entity, parse_session
from .llm import generate_chat_reply
from .message_analyzer import analyze_message
from .persona_engine import adjust_emotion_vector, record_situation_reaction
from .prompt_builder import build_chat_prompt

_BACKGROUND_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='agent-system-rebuild')
_REPAIR_STATUS_LOCK = Lock()
_REPAIR_STATUS: dict[str, dict[str, Any]] = {}


def _set_repair_status(session_id: str, payload: dict[str, Any]) -> None:
    with _REPAIR_STATUS_LOCK:
        _REPAIR_STATUS[session_id] = dict(payload)


def _get_repair_status(session_id: str) -> dict[str, Any]:
    with _REPAIR_STATUS_LOCK:
        return dict(_REPAIR_STATUS.get(session_id, {}))


def _schedule_background_extraction(session_id: str, personality_name: str = '') -> None:
    _set_repair_status(
        session_id,
        {
            'status': 'pending',
            'session_id': session_id,
            'personality_name': personality_name,
        },
    )

    def _runner() -> None:
        try:
            result = rebuild_artifacts(session_id, personality_name=personality_name)
        except Exception as exc:
            _set_repair_status(
                session_id,
                {
                    'status': 'error',
                    'session_id': session_id,
                    'personality_name': personality_name,
                    'error': str(exc),
                },
            )
            return
        _set_repair_status(
            session_id,
            {
                'status': 'ok' if result.get('ok') else 'degraded',
                'session_id': session_id,
                'personality_name': personality_name,
                'errors': list(result.get('errors') or []),
                'validation': result.get('validation') or {},
            },
        )

    _BACKGROUND_EXECUTOR.submit(_runner)


def generate_response(
    *,
    message: str,
    session_id: str = '',
    selected_persona: str = '',
    explicit_context: str = '',
    language: str = 'en',
) -> dict[str, Any]:
    clean_message = str(message or '').strip()
    session = create_session(session_id or '')
    clean_session_id = str(session.get('session_id') or session_id or '')
    graph_store = GraphStore()
    current_entity = infer_current_entity(clean_session_id)
    analysis = analyze_message(
        message=clean_message,
        session_id=clean_session_id,
        selected_head=selected_persona,
        current_entity=current_entity,
        explicit_context=explicit_context,
        known_entities=graph_store.load_nodes(),
    )
    classifications = [DEFAULT_CLASSIFIER.classify(extract_features(entity, analysis)) for entity in analysis.entities]
    prepared_heads = prepare_heads(analysis=analysis, classifications=classifications, graph_store=graph_store)
    primary = select_primary_head(analysis=analysis, prepared_heads=prepared_heads)
    primary_name = ''
    if primary and primary.get('head') is not None:
        primary_name = normalize_personality_name(primary['head'].name)
        adjust_emotion_vector(primary_name, analysis.cues)

    built = build_context(
        question=clean_message,
        session_id=clean_session_id,
        selected_persona=primary_name,
        explicit_context=explicit_context,
        situation=analysis.situation,
        store=graph_store,
    )
    prompt = build_chat_prompt(
        question=clean_message,
        persona_block=built.get('persona_block') or '',
        graph_context=built.get('graph_context') or '',
        recent_dialogue=built.get('recent_dialogue') or '',
        language=language,
    )
    assistant_reply = generate_chat_reply(prompt, language=language, persona_selected=bool(primary_name))
    if not str(assistant_reply or '').strip():
        assistant_reply = (
            'I will answer in first person from the current persona graph and emotional state.'
            if primary_name
            else 'I do not have enough reliable context yet. Clarify the entity or add one fact.'
        )
    append_turn(clean_session_id, clean_message, assistant_reply)

    if primary_name:
        record_situation_reaction(primary_name, analysis.situation, assistant_reply or 0)

    _schedule_background_extraction(clean_session_id, personality_name=primary_name)
    repair_status = _get_repair_status(clean_session_id)
    return {
        'assistant_reply': assistant_reply,
        'session_id': clean_session_id,
        'session': parse_session(clean_session_id) or session,
        'persona_name': built.get('persona_name') or primary_name,
        'graph_context': built.get('graph_context') or '',
        'current_entity': built.get('current_entity') or primary_name,
        'analysis': {
            'primary_entity': analysis.primary_entity,
            'situation': analysis.situation,
            'entities': [entity.name for entity in analysis.entities],
        },
        'classifications': [
            {
                'entity_name': decision.entity_name,
                'entity_type': decision.entity_type,
                'votes': decision.votes,
                'confidence': decision.confidence,
            }
            for decision in classifications
        ],
        'repair_status': repair_status,
        'proposal_requested': False,
    }
