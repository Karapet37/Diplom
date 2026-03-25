from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

from .classifier_forest import DEFAULT_CLASSIFIER
from .concept_graphs import concept_graph_extraction
from .context_builder import build_context
from .feature_extractor import extract_features
from .file_ingestion import rebuild_artifacts
from .graph_store import GraphStore, normalize_personality_name, personality_proposals_dir
from .head_caller import prepare_heads, select_primary_head
from .history_store import append_turn, create_session, infer_current_entity, parse_session
from .llm import fallback_chat_reply, generate_chat_reply
from .message_analyzer import analyze_message
from .models import BackgroundRebuildDecision, ChatSideEffects, ChatTurnRequest, ChatTurnResult, Situation
from .persona_engine import adjust_emotion_vector, record_situation_reaction
from .prompt_builder import build_chat_prompt
from .runtime_config import get_runtime_config

_BACKGROUND_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='agent-system-rebuild')
_REPAIR_STATUS_LOCK = Lock()
_REPAIR_STATUS: dict[str, dict[str, Any]] = {}


def _set_repair_status(session_id: str, payload: dict[str, Any]) -> None:
    with _REPAIR_STATUS_LOCK:
        _REPAIR_STATUS[session_id] = dict(payload)


def _get_repair_status(session_id: str) -> dict[str, Any]:
    with _REPAIR_STATUS_LOCK:
        return dict(_REPAIR_STATUS.get(session_id, {}))


def _user_turn_count(session_id: str) -> int:
    parsed = parse_session(session_id) or {}
    return sum(1 for item in list(parsed.get('messages') or []) if str(item.get('role') or '').strip() == 'user')


def _has_pending_persona_proposals() -> bool:
    return any(personality_proposals_dir().glob('*.json'))


def _should_schedule_background_extraction(session_id: str, *, personality_name: str = '') -> tuple[bool, str]:
    runtime = get_runtime_config()
    if not runtime.features.enable_background_rebuild:
        return False, 'background_rebuild_disabled'
    current_status = _get_repair_status(session_id)
    if str(current_status.get('status') or '').strip() == 'pending':
        return False, 'rebuild_already_pending'
    if _has_pending_persona_proposals():
        return True, 'pending_persona_proposals'
    user_turns = _user_turn_count(session_id)
    if user_turns and user_turns % runtime.settings.background_rebuild_interval == 0:
        return True, 'periodic_hygiene'
    if personality_name and user_turns >= 3 and user_turns % 3 == 0:
        return True, 'periodic_persona_sync'
    return False, 'deferred_for_latency'


def _schedule_background_extraction(session_id: str, personality_name: str = '') -> None:
    if str(_get_repair_status(session_id).get('status') or '').strip() == 'pending':
        return
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


def _write_concept_graph_from_message(message: str, *, graph_store: GraphStore, side_effects: ChatSideEffects) -> None:
    if not get_runtime_config().features.enable_concept_graph_premerge:
        return
    deterministic_concept_graph = concept_graph_extraction(message, source='chat')
    if deterministic_concept_graph.get('entities') or deterministic_concept_graph.get('relations'):
        graph_store.merge_extraction(deterministic_concept_graph, source='chat')
        side_effects.add_graph_write('chat:concept_graph_extraction')


def _write_session_history(session_id: str, user_message: str, assistant_reply: str, *, side_effects: ChatSideEffects) -> None:
    history_path = append_turn(session_id, user_message, assistant_reply)
    side_effects.history_write_path = str(history_path)


def _apply_emotion_update(persona_name: str, situation: Situation, *, side_effects: ChatSideEffects) -> None:
    if not persona_name:
        return
    adjust_emotion_vector(persona_name, situation)
    side_effects.add_persona_update('emotion_vector')


def _record_persona_reaction(persona_name: str, situation: Situation, assistant_reply: str, *, side_effects: ChatSideEffects) -> None:
    if not persona_name:
        return
    record_situation_reaction(persona_name, situation, assistant_reply or 0)
    side_effects.add_persona_update('situation_reaction')


def _apply_rebuild_schedule(session_id: str, *, personality_name: str, side_effects: ChatSideEffects) -> dict[str, Any]:
    should_schedule, reason = _should_schedule_background_extraction(session_id, personality_name=personality_name)
    side_effects.rebuild = BackgroundRebuildDecision(
        session_id=session_id,
        personality_name=personality_name,
        should_schedule=should_schedule,
        reason=reason,
    )
    if should_schedule:
        _schedule_background_extraction(session_id, personality_name=personality_name)
    else:
        _set_repair_status(
            session_id,
            {
                'status': 'skipped',
                'reason': reason,
                'session_id': session_id,
                'personality_name': personality_name,
            },
        )
    return _get_repair_status(session_id)


def run_chat_turn(request: ChatTurnRequest) -> ChatTurnResult:
    clean_message = str(request.message or '').strip()
    session = create_session(request.session_id or '')
    clean_session_id = str(session.get('session_id') or request.session_id or '')
    graph_store = GraphStore()
    side_effects = ChatSideEffects()
    _write_concept_graph_from_message(clean_message, graph_store=graph_store, side_effects=side_effects)
    current_entity = infer_current_entity(clean_session_id)
    analysis = analyze_message(
        message=clean_message,
        session_id=clean_session_id,
        selected_head=request.selected_persona,
        current_entity=current_entity,
        explicit_context=request.explicit_context,
        known_entities=graph_store.load_nodes(),
    )
    classifications = [DEFAULT_CLASSIFIER.classify(extract_features(entity, analysis)) for entity in analysis.entities]
    prepared_heads = prepare_heads(analysis=analysis, classifications=classifications, graph_store=graph_store)
    primary = select_primary_head(analysis=analysis, prepared_heads=prepared_heads)
    primary_name = ''
    if primary and primary.get('head') is not None:
        primary_name = normalize_personality_name(primary['head'].name)
        _apply_emotion_update(primary_name, analysis.situation, side_effects=side_effects)

    built = build_context(
        question=clean_message,
        session_id=clean_session_id,
        selected_persona=primary_name,
        explicit_context=request.explicit_context,
        situation=analysis.situation,
        store=graph_store,
    )
    prompt = build_chat_prompt(
        question=clean_message,
        persona_block=built.get('persona_block') or '',
        graph_context=built.get('graph_context') or '',
        recent_dialogue=built.get('recent_dialogue') or '',
        language=request.language,
    )
    if not any(str(part or '').strip() for part in (built.get('persona_block'), built.get('graph_context'), built.get('recent_dialogue'))):
        assistant_reply = fallback_chat_reply(language=request.language, persona_selected=bool(primary_name))
    else:
        assistant_reply = generate_chat_reply(prompt, language=request.language, persona_selected=bool(primary_name))

    _write_session_history(clean_session_id, clean_message, assistant_reply, side_effects=side_effects)
    _record_persona_reaction(primary_name, analysis.situation, assistant_reply, side_effects=side_effects)
    repair_status = _apply_rebuild_schedule(clean_session_id, personality_name=primary_name, side_effects=side_effects)

    return ChatTurnResult(
        assistant_reply=assistant_reply,
        session_id=clean_session_id,
        session=parse_session(clean_session_id) or session,
        persona_name=built.get('persona_name') or primary_name,
        graph_context=built.get('graph_context') or '',
        current_entity=built.get('current_entity') or primary_name,
        analysis=analysis,
        classifications=classifications,
        repair_status=repair_status,
        proposal_requested=False,
        side_effects=side_effects,
    )


def generate_response(
    *,
    message: str,
    session_id: str = '',
    selected_persona: str = '',
    explicit_context: str = '',
    language: str = 'en',
) -> dict[str, Any]:
    result = run_chat_turn(
        ChatTurnRequest(
            message=message,
            session_id=session_id,
            selected_persona=selected_persona,
            explicit_context=explicit_context,
            language=language,
        )
    )
    return result.to_dict(include_side_effects=get_runtime_config().features.include_side_effects_in_response)
