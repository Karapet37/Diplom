from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
from threading import Lock
from typing import Any

from .classifier_forest import DEFAULT_CLASSIFIER
from .concept_graphs import concept_graph_extraction
from .context_builder import build_context
from .duplicate_resolver import normalize_name
from .feature_extractor import extract_features
from .file_ingestion import rebuild_artifacts
from .graph_store import GraphStore, normalize_personality_name, personality_proposals_dir
from .head_caller import prepare_heads, select_primary_head
from .history_store import append_turn, create_session, infer_current_entity, parse_session
from .language_tools import normalize_language_code
from .llm import fallback_chat_reply, generate_chat_reply, translate_text
from .message_analyzer import analyze_message_state
from .mood_research import build_mood_snapshot, load_mood_report, record_mood_snapshot, schedule_mood_research_refresh
from .models import (
    BackgroundRebuildDecision,
    ChatSideEffects,
    ChatTurnRequest,
    ChatTurnResult,
    MessageAnalysis,
    PersonaResponseExplanation,
    PersonaSelectionExplanation,
    Situation,
)
from .observability import get_observability_store
from .persona_engine import adjust_emotion_vector, explain_response_style, load_persona, record_persona_dossier_fact, record_situation_reaction
from .prompt_builder import build_chat_prompt
from .reliability import operator_messages_from_status, runtime_status_snapshot
from .runtime_config import get_runtime_config
from .social_roles import choose_social_role, render_social_role_block
from .situation_engine import model_situation

_BACKGROUND_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='agent-system-rebuild')
_REPAIR_STATUS_LOCK = Lock()
_REPAIR_STATUS: dict[str, dict[str, Any]] = {}
_PERSONA_DOSSIER_FACT_MARKERS = (
    'you are',
    'you work',
    'you live',
    'you keep',
    'you value',
    'you trust',
    'you trained',
    'you rotate',
    'you still wear',
    'your sister',
    'your father',
    'your mother',
    'you usually',
    'you tend to',
)
_PERSONA_DOSSIER_UPDATE_MARKERS = (
    'for the record',
    'just so you know',
    'remember that',
    'note that',
    'keep in mind',
)
_SECOND_PERSON_TOKENS = {
    'you',
    'your',
    'yourself',
}


def _normalized_marker_hits(text: str, markers: tuple[str, ...]) -> list[str]:
    normalized_text = normalize_name(text)
    hits: list[str] = []
    for marker in markers:
        normalized_marker = normalize_name(marker)
        if normalized_marker and normalized_marker in normalized_text:
            hits.append(marker)
    return hits


def _strip_leading_persona_update_scaffolding(text: str) -> str:
    clean = ' '.join(str(text or '').strip().split())
    if not clean:
        return ''
    lowered = clean.casefold()
    for marker in sorted(_PERSONA_DOSSIER_UPDATE_MARKERS, key=len, reverse=True):
        marker_clean = ' '.join(str(marker or '').strip().split())
        if not marker_clean:
            continue
        marker_lowered = marker_clean.casefold()
        if lowered.startswith(marker_lowered):
            trimmed = clean[len(marker_clean) :].lstrip(' ,:;-')
            if trimmed:
                return trimmed
    return clean


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
            get_observability_store().record_rebuild_status('error')
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
        get_observability_store().record_rebuild_status('ok' if result.get('ok') else 'degraded')
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


def _effective_response_language(*, requested_language: str, detected_language: str) -> str:
    requested = normalize_language_code(requested_language, fallback='')
    detected = normalize_language_code(detected_language, fallback='en')
    if detected in {'ru', 'hy', 'zh'}:
        return detected
    if requested:
        return requested
    return detected or 'en'


def _internal_reasoning_message(message: str, *, detected_language: str) -> str:
    clean = ' '.join(str(message or '').strip().split())
    source_language = normalize_language_code(detected_language, fallback='en')
    if not clean or source_language == 'en':
        return clean
    translated = translate_text(
        clean,
        target_language='en',
        source_language=source_language,
        role=get_runtime_config().roles.translation,
    )
    return ' '.join(str(translated or clean).strip().split()) or clean


def _looks_like_persona_dossier_update(message: str, *, persona_name: str, analysis: MessageAnalysis) -> bool:
    clean = ' '.join(str(message or '').strip().split())
    if not clean or not persona_name:
        return False
    if analysis.user_state.intent in {'insult', 'seek_support', 'confession'}:
        return False
    lowered = normalize_name(clean)
    persona_token = normalize_name(persona_name)
    has_persona_reference = bool(persona_token and persona_token in lowered)
    has_second_person = any(token in lowered.split() for token in _SECOND_PERSON_TOKENS)
    if not has_persona_reference and not has_second_person:
        return False
    if len(clean.split()) < 5:
        return False
    update_hits = _normalized_marker_hits(clean, _PERSONA_DOSSIER_UPDATE_MARKERS)
    fact_hits = _normalized_marker_hits(clean, _PERSONA_DOSSIER_FACT_MARKERS)
    if '?' in clean and not update_hits:
        return False
    return bool(fact_hits or update_hits)


def _capture_persona_dossier_update(
    persona_name: str,
    message: str,
    *,
    analysis: MessageAnalysis,
    response_language: str,
    side_effects: ChatSideEffects,
    detection_message: str = '',
) -> bool:
    detection_input = str(detection_message or message or '').strip()
    if not _looks_like_persona_dossier_update(detection_input, persona_name=persona_name, analysis=analysis):
        return False
    try:
        clean = ' '.join(str(message or '').strip().split())
        canonical_excerpt = _strip_leading_persona_update_scaffolding(clean)
        source_language = normalize_language_code(response_language or analysis.user_state.language, fallback='en')
        if source_language != 'en':
            canonical_excerpt = translate_text(
                canonical_excerpt,
                target_language='en',
                source_language=source_language,
                role=get_runtime_config().roles.translation,
            ) or canonical_excerpt
        canonical_excerpt = _strip_leading_persona_update_scaffolding(canonical_excerpt)
        record_persona_dossier_fact(
            persona_name,
            canonical_excerpt,
        )
        side_effects.add_persona_update('learned_update')
        return True
    except Exception:
        return False


def _persona_dossier_acknowledgement(language: str) -> str:
    target = normalize_language_code(language, fallback='en')
    base = 'Noted. I will add that to my personal record and use it in later answers.'
    if target == 'en':
        return base
    return translate_text(
        base,
        target_language=target,
        source_language='en',
        role=get_runtime_config().roles.translation,
    ) or base


def _normalized_persona_dossier_situation() -> Situation:
    normalized = Situation(
        type='neutral_statement',
        target='persona',
        severity=0.22,
    )
    return Situation(
        type=normalized.type,
        target=normalized.target,
        severity=normalized.severity,
        summary=f'type={normalized.type}; target={normalized.target}; severity={normalized.severity:.2f}',
    )


def _record_persona_reaction(persona_name: str, situation: Situation, assistant_reply: str, *, side_effects: ChatSideEffects) -> None:
    if not persona_name:
        return
    record_situation_reaction(persona_name, situation, assistant_reply or 0)
    side_effects.add_persona_update('situation_reaction')


def _apply_rebuild_schedule(session_id: str, *, personality_name: str, side_effects: ChatSideEffects) -> dict[str, Any]:
    should_schedule, reason = _should_schedule_background_extraction(session_id, personality_name=personality_name)
    observability = get_observability_store()
    side_effects.rebuild = BackgroundRebuildDecision(
        session_id=session_id,
        personality_name=personality_name,
        should_schedule=should_schedule,
        reason=reason,
    )
    if should_schedule:
        observability.record_rebuild_schedule(scheduled=True, reason=reason, status='pending')
        _schedule_background_extraction(session_id, personality_name=personality_name)
    else:
        observability.record_rebuild_schedule(scheduled=False, reason=reason, status='skipped')
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


def _call_build_context_compat(**kwargs: Any) -> dict[str, Any]:
    try:
        signature = inspect.signature(build_context)
    except (TypeError, ValueError):
        return build_context(**kwargs)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return build_context(**kwargs)
    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return build_context(**accepted)


def _call_build_chat_prompt_compat(**kwargs: Any) -> str:
    try:
        signature = inspect.signature(build_chat_prompt)
    except (TypeError, ValueError):
        return build_chat_prompt(**kwargs)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return build_chat_prompt(**kwargs)
    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return build_chat_prompt(**accepted)


def run_chat_turn(request: ChatTurnRequest) -> ChatTurnResult:
    clean_message = str(request.message or '').strip()
    session = create_session(request.session_id or '')
    clean_session_id = str(session.get('session_id') or request.session_id or '')
    graph_store = GraphStore()
    side_effects = ChatSideEffects()
    observability = get_observability_store()
    trace = observability.start_trace(
        request_type='chat',
        route='/api/cognitive/chat/respond',
        session_id=clean_session_id,
        request_meta={
            'language': request.language,
            'message_chars': len(clean_message),
            'selected_persona': str(request.selected_persona or '').strip(),
        },
    )
    try:
        observability.time_stage(
            trace,
            'graph_prewrite',
            lambda: _write_concept_graph_from_message(clean_message, graph_store=graph_store, side_effects=side_effects),
            meta_builder=lambda _result: {'graph_write_sources': list(side_effects.graph_write_sources)},
        )
        current_entity = infer_current_entity(clean_session_id)
        known_nodes = graph_store.load_nodes()
        prepared = observability.time_stage(
            trace,
            'analysis',
            lambda: analyze_message_state(
                message=clean_message,
                session_id=clean_session_id,
                selected_head=request.selected_persona,
                current_entity=current_entity,
                explicit_context=request.explicit_context,
                known_entities=known_nodes,
            ),
            meta_builder=lambda payload: {
                'entity_count': len(list(payload.get('entities') or [])),
                'tone': payload['user_state'].tone,
                'intent': payload['user_state'].intent,
            },
        )
        situation = observability.time_stage(
            trace,
            'situation_building',
            lambda: model_situation(
                message=str(prepared['message'] or ''),
                primary_entity=str(prepared['primary_entity'] or ''),
                selected_head=str(prepared['selected_head'] or ''),
                user_state=prepared['user_state'],
            ),
            meta_builder=lambda item: {
                'type': item.type,
                'target': item.target,
                'severity': item.severity,
            },
        )
        analysis = MessageAnalysis(
            message=str(prepared['message'] or ''),
            session_id=clean_session_id,
            selected_head=str(prepared['selected_head'] or ''),
            primary_entity=str(prepared['primary_entity'] or ''),
            current_entity=str(prepared['current_entity'] or ''),
            explicit_context=str(prepared['explicit_context'] or ''),
            entities=list(prepared['entities'] or []),
            user_state=prepared['user_state'],
            situation=situation,
        )
        response_language = _effective_response_language(
            requested_language=request.language,
            detected_language=analysis.user_state.language,
        )
        reasoning_message = _internal_reasoning_message(
            clean_message,
            detected_language=analysis.user_state.language,
        )
        dossier_update_candidate = _looks_like_persona_dossier_update(
            reasoning_message,
            persona_name=str(prepared['selected_head'] or request.selected_persona or ''),
            analysis=analysis,
        )
        dossier_update_statement = bool(
            dossier_update_candidate
            and analysis.user_state.intent == 'statement'
            and not bool(analysis.user_state.signals.get('contains_question'))
        )
        if dossier_update_statement:
            analysis = MessageAnalysis(
                message=analysis.message,
                session_id=analysis.session_id,
                selected_head=analysis.selected_head,
                primary_entity=analysis.primary_entity,
                current_entity=analysis.current_entity,
                explicit_context=analysis.explicit_context,
                entities=list(analysis.entities),
                user_state=analysis.user_state,
                situation=_normalized_persona_dossier_situation(),
            )
        features = observability.time_stage(
            trace,
            'feature_extraction',
            lambda: [extract_features(entity, analysis) for entity in analysis.entities],
            meta_builder=lambda rows: {'entity_count': len(rows)},
        )
        classifications = observability.time_stage(
            trace,
            'classification',
            lambda: [DEFAULT_CLASSIFIER.classify(feature_row) for feature_row in features],
            meta_builder=lambda rows: {'classified_entities': len(rows)},
        )

        def _resolve_heads() -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
            prepared_heads = prepare_heads(analysis=analysis, classifications=classifications, graph_store=graph_store)
            primary = select_primary_head(analysis=analysis, prepared_heads=prepared_heads)
            primary_name = ''
            if primary and primary.get('head') is not None:
                primary_name = normalize_personality_name(primary['head'].name)
            return prepared_heads, primary, primary_name

        prepared_heads, primary, primary_name = observability.time_stage(
            trace,
            'head_selection',
            _resolve_heads,
            meta_builder=lambda rows: {
                'prepared_heads': len(rows[0]),
                'primary_name': rows[2],
            },
        )
        observability.time_stage(
            trace,
            'persona_update',
            lambda: _apply_emotion_update(primary_name, analysis.situation, side_effects=side_effects),
            meta_builder=lambda _result: {
                'updated': bool(primary_name),
                'persona_updates': list(side_effects.persona_updates),
            },
        )
        active_persona_bundle = load_persona(primary_name) if primary_name else None
        session_mood_report = load_mood_report(session_id=clean_session_id) if clean_session_id else None
        persona_mood_report = load_mood_report(persona_name=primary_name) if primary_name else None
        active_mood_report = session_mood_report or persona_mood_report
        selection_explanation = (
            primary.get('selection_explanation')
            if primary and isinstance(primary.get('selection_explanation'), PersonaSelectionExplanation)
            else PersonaSelectionExplanation(persona_name=primary_name)
        )
        response_explanation = (
            explain_response_style(active_persona_bundle, analysis.situation)
            if active_persona_bundle is not None
            else PersonaResponseExplanation(persona_name=primary_name)
        )
        social_role = choose_social_role(
            bundle=active_persona_bundle,
            analysis=analysis,
            situation=analysis.situation,
            mood_report=active_mood_report,
        )
        built = observability.time_stage(
            trace,
            'context_building',
            lambda: _call_build_context_compat(
                question=reasoning_message,
                session_id=clean_session_id,
                selected_persona=primary_name,
                explicit_context=request.explicit_context,
                situation=analysis.situation,
                store=graph_store,
                social_role=social_role,
                mood_report=active_mood_report,
                analysis=analysis,
            ),
            meta_builder=lambda payload: {
                'estimated_tokens': int(payload.get('estimated_tokens') or 0),
                'selected_items': len(list(dict(payload.get('context_debug') or {}).get('selected_items') or [])),
                'source_counts': dict(dict(payload.get('context_debug') or {}).get('source_counts') or {}),
                'packed_candidate_counts': dict(dict(dict(payload.get('context_debug') or {}).get('stages') or {}).get('pack_context') or {}),
            },
        )
        prompt = _call_build_chat_prompt_compat(
            question=clean_message,
            internal_question=reasoning_message,
            persona_block=built.get('persona_block') or '',
            social_role_block=render_social_role_block(social_role, mood_report=active_mood_report),
            mood_research_block=(
                f"Latest mood cluster: {active_mood_report.latest_cluster_label}. "
                f"Observed role trend: {', '.join(str(item.get('role') or '') for item in list(active_mood_report.role_effects or [])[:3] if str(item.get('role') or '').strip())}."
                if active_mood_report is not None and active_mood_report.snapshot_count > 0
                else ''
            ),
            graph_context=built.get('graph_context') or '',
            recent_dialogue=built.get('recent_dialogue') or '',
            language=response_language,
            semantic_focus=dict(dict(built.get('context_debug') or {}).get('semantic_focus') or {}),
        )

        def _generate_reply() -> tuple[str, bool, str, dict[str, Any]]:
            if dossier_update_statement:
                return _persona_dossier_acknowledgement(response_language), False, '', {}
            fallback_text = fallback_chat_reply(language=response_language, persona_selected=bool(primary_name))
            grounded = any(str(part or '').strip() for part in (built.get('persona_block'), built.get('graph_context'), built.get('recent_dialogue')))
            if not grounded:
                return fallback_text, True, 'no_grounding', {}
            reply = generate_chat_reply(prompt, language=response_language, persona_selected=bool(primary_name))
            used_fallback = str(reply or '').strip() == fallback_text
            status = runtime_status_snapshot() if used_fallback else {}
            fallback_reason = ''
            if used_fallback:
                fallback_reason = 'dependency_unavailable' if str(dict(status or {}).get('mode') or '') == 'degraded' else 'model_fallback'
            return reply, used_fallback, fallback_reason, status

        assistant_reply, fallback_used, fallback_reason, runtime_status = observability.time_stage(
            trace,
            'llm_call',
            _generate_reply,
            meta_builder=lambda result: {
                'fallback_used': bool(result[1]),
                'fallback_reason': str(result[2] or ''),
                'runtime_mode': str(dict(result[3] or {}).get('mode') or 'full'),
            },
        )
        operator_messages = operator_messages_from_status(runtime_status)
        if fallback_reason == 'no_grounding':
            operator_messages = [
                'Fallback reply was returned because the bounded context contained no grounded items.',
                *operator_messages,
            ]
        elif fallback_reason == 'model_fallback':
            operator_messages = [
                'Fallback reply was returned because the model produced no safe usable text.',
                *operator_messages,
            ]
        elif fallback_reason == 'dependency_unavailable':
            operator_messages = [
                'Fallback reply was returned because the local chat provider is unavailable.',
                *operator_messages,
            ]
        repair_status = observability.time_stage(
            trace,
            'storage_writes',
            lambda: (
                _write_session_history(clean_session_id, clean_message, assistant_reply, side_effects=side_effects),
                _record_persona_reaction(primary_name, analysis.situation, assistant_reply, side_effects=side_effects),
                _capture_persona_dossier_update(
                    primary_name,
                    clean_message,
                    analysis=analysis,
                    response_language=response_language,
                    side_effects=side_effects,
                    detection_message=reasoning_message,
                ),
                record_mood_snapshot(
                    build_mood_snapshot(
                        analysis=analysis,
                        persona_bundle=active_persona_bundle,
                        social_role=social_role,
                        response_style=response_explanation.response_style,
                        session_id=clean_session_id,
                    )
                ),
                schedule_mood_research_refresh(persona_name=primary_name, session_id=clean_session_id),
                _apply_rebuild_schedule(clean_session_id, personality_name=primary_name, side_effects=side_effects),
            )[-1],
            meta_builder=lambda payload: {
                'history_write_path': side_effects.history_write_path,
                'persona_updates': list(side_effects.persona_updates) + ['mood_research'],
                'repair_status': str(payload.get('status') or ''),
            },
        )
        side_effects.add_persona_update('mood_research')
        observability.finish_trace(
            trace,
            status='ok',
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            context_tokens=int(built.get('estimated_tokens') or 0),
            persona_name=built.get('persona_name') or primary_name,
            current_entity=built.get('current_entity') or primary_name,
            response_meta={
                'classification_count': len(classifications),
                'repair_status': str(repair_status.get('status') or ''),
                'runtime_mode': str(dict(runtime_status or {}).get('mode') or 'full'),
                'response_language': response_language,
                'social_role': social_role.role,
            },
        )
        if 'learned_update' in side_effects.persona_updates:
            operator_messages = [
                *operator_messages,
                'A new user-provided fact about the persona was added to the learned dossier and will influence future replies.',
            ]
        operator_messages = [
            *operator_messages,
            f"The current interaction role was selected as '{social_role.role}' based on persona structure, mood signals, and situation context.",
        ]
        return ChatTurnResult(
            assistant_reply=assistant_reply,
            response_language=response_language,
            session_id=clean_session_id,
            trace_id=trace.request_id,
            session=parse_session(clean_session_id) or session,
            persona_name=built.get('persona_name') or primary_name,
            graph_context=built.get('graph_context') or '',
            current_entity=built.get('current_entity') or primary_name,
            analysis=analysis,
            classifications=classifications,
            repair_status=repair_status,
            proposal_requested=False,
            side_effects=side_effects,
            persona_selection=selection_explanation,
            persona_response=response_explanation,
            social_role=social_role,
            mood_research={
                'active_report_scope': active_mood_report.scope if active_mood_report is not None else '',
                'latest_cluster': active_mood_report.latest_cluster_label if active_mood_report is not None else '',
                'snapshot_count': active_mood_report.snapshot_count if active_mood_report is not None else 0,
            },
            behavior_trace={
                'semantic_focus': dict(dict(built.get('context_debug') or {}).get('semantic_focus') or {}),
                'social_role': social_role.to_dict(),
                'mood_cluster': active_mood_report.latest_cluster_label if active_mood_report is not None else '',
                'selected_context_sources': dict(dict(built.get('context_debug') or {}).get('source_counts') or {}),
                'selected_context_items': [
                    {
                        'source': str(item.get('source') or ''),
                        'item_type': str(item.get('item_type') or ''),
                        'title': str(item.get('title') or ''),
                        'reasons': list(item.get('reasons') or []),
                    }
                    for item in list(dict(built.get('context_debug') or {}).get('selected_items') or [])[:12]
                    if isinstance(item, dict)
                ],
                'dossier_update_candidate': bool(dossier_update_candidate),
                'response_style': response_explanation.response_style,
            },
            context_preview={
                'estimated_tokens': int(built.get('estimated_tokens') or 0),
                'graph_context': built.get('graph_context') or '',
                'current_entity': built.get('current_entity') or primary_name,
                'persona_name': built.get('persona_name') or primary_name,
                'social_role': social_role.to_dict(),
                'source_counts': dict(dict(built.get('context_debug') or {}).get('source_counts') or {}),
                'selected_items': list(dict(built.get('context_debug') or {}).get('selected_items') or []),
            },
            runtime_status=dict(runtime_status or {}),
            operator_messages=list(dict.fromkeys(operator_messages)),
        )
    except Exception as exc:
        observability.finish_trace(
            trace,
            status='error',
            fallback_used=False,
            fallback_reason='',
            context_tokens=0,
            persona_name='',
            current_entity='',
            response_meta={'error': str(exc)},
        )
        raise


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
