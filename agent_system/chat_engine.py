from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from pathlib import Path
from threading import Lock
from typing import Any

from .classifier_forest import DEFAULT_CLASSIFIER
from .behavioral_fallback import choose_behavioral_fallback, render_behavioral_fallback, render_memory_update_acknowledgement
from .concept_graphs import concept_graph_extraction
from .controller_runtime import build_controller_state
from .context_builder import build_context
from .duplicate_resolver import normalize_name
from .feature_extractor import extract_features
from .file_ingestion import rebuild_artifacts
from .graph_store import GraphStore, normalize_personality_name, personality_proposals_dir
from .head_caller import prepare_heads, select_primary_head
from .history_store import append_turn, create_session, infer_current_entity, load_session_route_state, parse_session, recent_dialogue, save_session_route_state
from .language_tools import normalize_language_code
from .llm import (
    fallback_chat_reply,
    generate_chat_reply,
    get_last_model_call_meta,
    plan_model_budget,
    reset_last_model_call_meta,
    translate_text,
)
from .message_annotation_store import build_runtime_message_vector_payload
from .mood_research import build_mood_snapshot, load_mood_report, record_mood_snapshot, schedule_mood_research_refresh
from .models import (
    BackgroundRebuildDecision,
    BehavioralFallbackDecision,
    ChatSideEffects,
    ChatTurnRequest,
    ChatTurnResult,
    MessageAnalysis,
    PersonaResponseExplanation,
    PersonaSelectionExplanation,
    RequestEnvelope,
    RequestPreprocessing,
    ResponseValidation,
    RouteDecision,
    RouteMemory,
    Situation,
    TraceLearningPolicy,
)
from .observability import get_observability_store
from .persona_engine import adjust_emotion_vector, explain_response_style, load_persona, record_persona_dossier_fact, record_situation_reaction
from .persona_engine import create_persona_from_description, load_active_persona
from .prompt_builder import build_chat_prompt
from .request_pipeline import (
    build_request_envelope,
    deterministic_reply,
    render_route_guidance,
    validate_response,
)
from .reliability import operator_messages_from_status, runtime_status_snapshot
from .runtime_config import get_runtime_config
from .social_roles import choose_social_role, render_social_role_block
from .state_transition_runtime import (
    append_transition_log,
    build_bounded_next_state,
    build_working_context_layer,
    infer_session_active_persona,
    interpret_user_influence,
    persist_current_context,
    reconstruct_task_procedure,
    render_response_shaping_block,
    render_reviewed_context_block,
    render_state_transition_block,
    render_task_procedure_block,
    review_working_context,
    sample_state_snapshot,
    shape_response_plan,
)
from .text_resources import SECOND_PERSON_TOKENS
from .notes_store import execute_note_command, format_notes_for_context, parse_note_command, save_note
from .planning_engine import (
    build_planning_output,
    build_planning_prompt,
    detect_planning_intent,
    classify_feedback,
)
from .situation_regulator import run_regulator_pipeline, RegulatorState
from .importance_learner import record_positive_example, record_negative_example, suggest_important_turns
from .safety_classifier import run_safety_check, SafetyBlockError, get_block_response
from .localization_engine import build_localization_prompt, detect_language as detect_text_language
from .genome_store import GenomeStore
from .cognitive_pipeline import CognitiveRuntime, RegulatorState as CogRegulatorState
from .training_store import TrainingStore, TrainingTuple

_BACKGROUND_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='agent-system-rebuild')
_BACKGROUND_EXECUTOR_LOCK = Lock()
_BACKGROUND_EXECUTOR_CLOSED = False
_REPAIR_STATUS_LOCK = Lock()
_REPAIR_STATUS: dict[str, dict[str, Any]] = {}
_REPAIR_STATUS_LIMIT = 128
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
        clean_session_id = str(session_id or '').strip()
        if clean_session_id in _REPAIR_STATUS:
            _REPAIR_STATUS.pop(clean_session_id, None)
        _REPAIR_STATUS[clean_session_id] = dict(payload)
        while len(_REPAIR_STATUS) > _REPAIR_STATUS_LIMIT:
            oldest_key = next(iter(_REPAIR_STATUS), '')
            if not oldest_key:
                break
            _REPAIR_STATUS.pop(oldest_key, None)


def _get_repair_status(session_id: str) -> dict[str, Any]:
    with _REPAIR_STATUS_LOCK:
        return dict(_REPAIR_STATUS.get(session_id, {}))


def forget_session_runtime_state(session_id: str) -> None:
    clean_session_id = str(session_id or '').strip()
    if not clean_session_id:
        return
    with _REPAIR_STATUS_LOCK:
        _REPAIR_STATUS.pop(clean_session_id, None)


def _shutdown_background_executor() -> None:
    global _BACKGROUND_EXECUTOR_CLOSED
    with _BACKGROUND_EXECUTOR_LOCK:
        if _BACKGROUND_EXECUTOR_CLOSED:
            return
        executor = _BACKGROUND_EXECUTOR
        shutdown = getattr(executor, 'shutdown', None)
        if callable(shutdown):
            shutdown(wait=False, cancel_futures=True)
        _BACKGROUND_EXECUTOR_CLOSED = True


atexit.register(_shutdown_background_executor)


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
    global _BACKGROUND_EXECUTOR_CLOSED
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

    with _BACKGROUND_EXECUTOR_LOCK:
        if _BACKGROUND_EXECUTOR_CLOSED:
            _set_repair_status(
                session_id,
                {
                    'status': 'degraded',
                    'session_id': session_id,
                    'personality_name': personality_name,
                    'error': 'background_executor_unavailable',
                },
            )
            return
        try:
            _BACKGROUND_EXECUTOR.submit(_runner)
        except RuntimeError:
            _BACKGROUND_EXECUTOR_CLOSED = True
            _set_repair_status(
                session_id,
                {
                    'status': 'degraded',
                    'session_id': session_id,
                    'personality_name': personality_name,
                    'error': 'background_executor_unavailable',
                },
            )


def _write_concept_graph_from_message(message: str, *, graph_store: GraphStore, side_effects: ChatSideEffects, session_id: str = '') -> None:
    if not get_runtime_config().features.enable_concept_graph_premerge:
        return
    deterministic_concept_graph = concept_graph_extraction(message, source='chat')
    if deterministic_concept_graph.get('entities') or deterministic_concept_graph.get('relations'):
        graph_store.merge_extraction(deterministic_concept_graph, source='chat', session_id=session_id)
        side_effects.add_graph_write('chat:concept_graph_extraction')


def _write_session_history(session_id: str, user_message: str, assistant_reply: str, *, side_effects: ChatSideEffects, persona_name: str = '', user_persona_name: str = '') -> None:
    history_path = append_turn(session_id, user_message, assistant_reply, persona_name=persona_name, user_persona_name=user_persona_name)
    side_effects.history_write_path = str(history_path)


def _apply_emotion_update(persona_name: str, situation: Situation, *, side_effects: ChatSideEffects) -> None:
    if not persona_name:
        return
    adjust_emotion_vector(persona_name, situation)
    side_effects.add_persona_update('emotion_vector')


def _effective_response_language(*, requested_language: str, detected_language: str) -> str:
    requested = normalize_language_code(requested_language, fallback='')
    detected = normalize_language_code(detected_language, fallback='en')
    if detected and detected != 'en':
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
    has_second_person = any(token in lowered.split() for token in SECOND_PERSON_TOKENS)
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


def _persona_dossier_acknowledgement(
    language: str,
    *,
    persona_bundle: Any | None = None,
    social_role: Any | None = None,
    analysis: MessageAnalysis | None = None,
) -> str:
    target = normalize_language_code(language, fallback='en')
    base = render_memory_update_acknowledgement(
        bundle=persona_bundle,
        social_role=social_role,
        analysis=analysis,
    )
    if target == 'en':
        return base
    if target == 'ru':
        return 'Принял. Это теперь часть моего контекста, и дальше я буду отвечать с учётом этого.'
    return translate_text(
        base,
        target_language=target,
        source_language='en',
        role=get_runtime_config().roles.translation,
    ) or base


def _persona_creation_acknowledgement(language: str, *, persona_name: str, activated: bool) -> str:
    target = normalize_language_code(language, fallback='en')
    if target == 'ru':
        if activated:
            return f'Создал личность "{persona_name}" как структурированный профиль и сделал её текущей.'
        return f'Создал личность "{persona_name}" как структурированный профиль.'
    if activated:
        return f'I created the persona "{persona_name}" as a structured profile and made it the current persona.'
    return f'I created the persona "{persona_name}" as a structured profile.'


def _persona_assignment_acknowledgement(language: str, *, persona_name: str) -> str:
    target = normalize_language_code(language, fallback='en')
    if target == 'ru':
        return f'Сделал "{persona_name}" текущей личностью.'
    return f'"{persona_name}" is now the current persona.'


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


def _call_generate_chat_reply_compat(**kwargs: Any) -> str:
    try:
        signature = inspect.signature(generate_chat_reply)
    except (TypeError, ValueError):
        return generate_chat_reply(**kwargs)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return generate_chat_reply(**kwargs)
    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return generate_chat_reply(**accepted)


def _session_has_previous_assistant_turn(session_id: str) -> bool:
    parsed = parse_session(session_id) or {}
    return any(str(item.get('role') or '').strip() == 'assistant' for item in list(parsed.get('messages') or []))



def _append_persona_examples(persona_name: str, persona_block: str, max_examples: int = 3) -> str:
    """Append few-shot examples from user corrections — the ML signal for persona behavior."""
    try:
        examples = _list_training_examples(persona_name=persona_name)
        if not examples:
            return persona_block
        shots = []
        for ex in reversed(examples[-max_examples * 2:]):
            inp = str(ex.get('input_text') or '').strip()
            out = str(ex.get('correct_output') or '').strip()
            if inp and out:
                shots.append(f'User: {inp}\n{persona_name}: {out}')
            if len(shots) >= max_examples:
                break
        if not shots:
            return persona_block
        return persona_block + '\n\nExamples of correct responses:\n' + '\n\n'.join(shots)
    except Exception:
        return persona_block


def _simple_context_payload(
    *,
    session_id: str,
    current_entity: str,
    route: RouteDecision,
    selected_persona: str = '',
) -> dict[str, Any]:
    recent = recent_dialogue(session_id) if route.requires_history else ''
    recent_turns: list[dict[str, str]] = []
    parsed = parse_session(session_id) or {}
    for item in list(parsed.get('messages') or [])[-8:]:
        role = str(item.get('role') or '').strip()
        content = str(item.get('display_text') or item.get('raw_text') or item.get('message') or '').strip()
        if role and content:
            recent_turns.append(
                {
                    'role': role,
                    'content': content,
                    'persona_name': str(item.get('persona_name') or '').strip(),
                    'timestamp': str(item.get('timestamp') or '').strip(),
                }
            )
    persona_name = str(selected_persona or '').strip()
    persona_block = ''
    if route.requires_persona and persona_name:
        bundle = load_active_persona(persona_name)
        structured = bundle.structured_persona if bundle is not None else None
        lines: list[str] = []
        if structured is not None:
            lines.append(f'You are {persona_name}.')
            if str(structured.identity.short_description or '').strip():
                lines.append(f"Summary: {structured.identity.short_description}")
            if str(structured.core_goal or '').strip():
                lines.append(f"Core goal: {structured.core_goal}")
            if list(structured.constraints_internal or []):
                lines.append(
                    f"Internal constraints: {' | '.join(str(item).strip() for item in list(structured.constraints_internal or [])[:4] if str(item).strip())}."
                )
            if list(structured.constraints_social or []):
                lines.append(
                    f"Social constraints: {' | '.join(str(item).strip() for item in list(structured.constraints_social or [])[:4] if str(item).strip())}."
                )
            if list(structured.allowed_methods or []):
                lines.append(
                    f"Allowed methods: {' | '.join(str(item).strip() for item in list(structured.allowed_methods or [])[:4] if str(item).strip())}."
                )
            if list(structured.maladaptive_methods or []):
                lines.append(
                    f"Maladaptive methods: {' | '.join(str(item).strip() for item in list(structured.maladaptive_methods or [])[:4] if str(item).strip())}."
                )
            if list(structured.behavior.communication_style or []):
                lines.append(
                    f"Communication style: {' | '.join(str(item).strip() for item in list(structured.behavior.communication_style or [])[:4] if str(item).strip())}."
                )
        elif bundle is not None:
            lines.append(f'You are {persona_name}.')
            if bundle.traits:
                lines.append(f"Traits: {', '.join(bundle.traits[:6])}.")
            if bundle.knowledge:
                lines.append(f"Knowledge: {bundle.knowledge[:700]}")
        persona_block = '\n'.join(line for line in lines if str(line).strip()).strip()
        # Inject few-shot training examples — user corrections become in-context examples
        if persona_name and persona_block:
            persona_block = _append_persona_examples(persona_name, persona_block)
    estimated_tokens = max((len(recent) + len(persona_block)) // 4, 0)
    return {
        'persona_name': persona_name,
        'current_entity': str(current_entity or '').strip(),
        'persona_block': persona_block,
        'graph_context': '',
        'recent_dialogue': recent,
        'recent_dialogue_turns': recent_turns,
        'estimated_tokens': estimated_tokens,
        'context_debug': {
            'source_counts': {
                'recent_dialogue': 1 if recent else 0,
                'knowledge_graph': 0,
                'persona_graph': 1 if persona_block else 0,
            },
            'selected_items': [],
            'selected_route': route.selected_route,
        },
    }


def _recent_exchange_pairs(turns: list[dict[str, Any]], *, pair_limit: int = 3) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pending_user = ''
    for turn in list(turns or []):
        role = str(turn.get('role') or '').strip()
        content = str(turn.get('content') or '').strip()
        if not role or not content:
            continue
        if role == 'user':
            pending_user = content
            continue
        if role == 'assistant' and pending_user:
            pairs.append((pending_user, content))
            pending_user = ''
    limit = max(int(pair_limit or 0), 0)
    return pairs[-limit:] if limit else pairs


def _route_aware_fallback_decision(
    route: RouteDecision,
    *,
    language: str,
    history_available: bool = False,
) -> BehavioralFallbackDecision | None:
    if route.selected_route != 'hypothetical_roleplay':
        return None
    reply = deterministic_reply(route, language=language, history_available=history_available)
    if not str(reply or '').strip():
        return None
    return BehavioralFallbackDecision(
        strategy='route_preserving_hypothetical',
        confidence=0.82,
        uncertainty_level='moderate',
        risk_level='low',
        style_mode='grounded',
        reason='The runtime preserved the hypothetical route instead of degrading into a grounding limitation.',
        evidence=[
            'selected route: hypothetical_roleplay',
            'fallback stayed inside the requested hypothetical frame',
        ],
        source_grounding='route_policy',
    )


def _pipeline_payload(
    *,
    envelope: RequestEnvelope,
    preprocessing: RequestPreprocessing,
    route: RouteDecision,
    capability_plan: Any,
    context_sources_used: list[str],
    fallback_used: bool,
    fallback_reason: str,
    validation: ResponseValidation,
    model_budget: dict[str, Any] | None = None,
    logical_context: dict[str, Any] | None = None,
    trace_learning: TraceLearningPolicy | None = None,
) -> dict[str, Any]:
    return {
        'request': envelope.to_dict(),
        'preprocessing': preprocessing.to_dict(),
        'route': route.to_dict(),
        'capability_plan': capability_plan.to_dict(),
        'context_sources_used': list(context_sources_used),
        'fallback_triggered': bool(fallback_used),
        'fallback_reason_code': str(fallback_reason or '').strip(),
        'validation': validation.to_dict(),
        'model_budget': dict(model_budget or {}),
        'logical_context': dict(logical_context or {}),
        'trace_learning': trace_learning.to_dict() if trace_learning is not None else {},
    }


def _load_route_memory(session_id: str) -> RouteMemory:
    payload = load_session_route_state(session_id)
    if not isinstance(payload, dict):
        return RouteMemory()
    return RouteMemory(
        previous_route=str(payload.get('selected_route') or '').strip(),
        previous_request_type=str(payload.get('request_type') or '').strip(),
        previous_interaction_mode=str(payload.get('interaction_mode') or '').strip(),
        previous_response_style=str(payload.get('response_style') or '').strip(),
        previous_persona_name=str(payload.get('persona_name') or '').strip(),
        previous_current_entity=str(payload.get('current_entity') or '').strip(),
        persona_style_traits=[str(item).strip() for item in list(payload.get('persona_style_traits') or []) if str(item).strip()][:6],
        speech_style_hints=[str(item).strip() for item in list(payload.get('speech_style_hints') or []) if str(item).strip()][:6],
    )


_SYSTEM_PERSONA_NAMES_ROUTE = frozenset({
    'generate_persona', 'generate', 'create_persona', 'new_persona',
    'unnamed', 'unknown', 'default', 'assistant', 'persona',
})
_VECTOR_GUIDANCE_INTERPRETERS = (
    'F1', 'F4', 'F8', 'F11', 'F13', 'F16', 'F18', 'F20', 'F24', 'F29',
    'F30', 'F34', 'F35', 'F36', 'F38', 'F39', 'F40', 'F41', 'F48', 'F49',
)
_VECTOR_GUIDANCE_HIDDEN = frozenset({
    'absent', 'neutral', 'unclear', 'continuation', 'defined', 'direct',
    'literal', 'statement', 'non_answer', 'independent',
    'opening_new_direction', 'calm',
})


def _is_system_persona_name(name: str) -> bool:
    n = str(name or '').strip().lower()
    return not n or n in _SYSTEM_PERSONA_NAMES_ROUTE or n.startswith('generate_')


def _format_vector_choice(choice: dict[str, Any] | None) -> str:
    item = dict(choice or {})
    main = str(item.get('main') or '').strip()
    if not main or main in _VECTOR_GUIDANCE_HIDDEN:
        return ''
    extras = [
        str(extra or '').strip()
        for extra in list(item.get('extra') or [])[:2]
        if str(extra or '').strip() and str(extra or '').strip() != main
    ]
    return f'{main} (+{", ".join(extras)})' if extras else main


def _summarize_vector_labels(vector: dict[str, Any] | None, *, limit: int = 6) -> list[str]:
    labels: list[str] = []
    payload = dict(vector or {})
    for interpreter_id in _VECTOR_GUIDANCE_INTERPRETERS:
        choice = _format_vector_choice(payload.get(interpreter_id))
        if choice:
            labels.append(f'{interpreter_id}={choice}')
        if len(labels) >= max(int(limit or 0), 0):
            break
    return labels


def _render_message_vector_guidance(payload: dict[str, Any] | None) -> str:
    data = dict(payload or {})
    current_labels = _summarize_vector_labels(data.get('current_vector'), limit=8)
    context_bits: list[str] = []
    for item in list(data.get('context_matrix') or [])[-2:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get('role') or '').strip() or 'unknown'
        persona = str(item.get('persona_name') or '').strip()
        role_label = f'{role}[{persona}]' if persona else role
        labels = _summarize_vector_labels(item.get('vector'), limit=4)
        if labels:
            context_bits.append(f'{role_label}: {", ".join(labels)}')
    transition = dict(data.get('transition_interpretation') or {})
    flow = [str(item).strip() for item in list(data.get('history_flow') or []) if str(item).strip()]
    lines = ['[P-COORDINATE LAYER]', 'Use the learned P1..P49 vector layer as behavioral guidance. Do not mention the labels explicitly.']
    if current_labels:
        lines.append('Current user vector: ' + '; '.join(current_labels) + '.')
    if context_bits:
        lines.append('Context matrix: ' + ' | '.join(context_bits) + '.')
    if flow:
        lines.append('History flow: ' + ' -> '.join(flow[:6]) + '.')
    if transition:
        from_labels = [str(item).strip() for item in list(transition.get('from') or []) if str(item).strip()]
        to_labels = [str(item).strip() for item in list(transition.get('to') or []) if str(item).strip()]
        transition_type = str(transition.get('type') or '').strip()
        transition_parts: list[str] = []
        if from_labels:
            transition_parts.append('from ' + ', '.join(from_labels[:4]))
        if to_labels:
            transition_parts.append('to ' + ', '.join(to_labels[:4]))
        if transition_type:
            transition_parts.append(f'type={transition_type}')
        if transition_parts:
            lines.append('Transition: ' + '; '.join(transition_parts) + '.')
    return '\n'.join(line for line in lines if str(line).strip())


def _save_route_memory(
    session_id: str,
    *,
    route: RouteDecision,
    preprocessing: RequestPreprocessing,
    persona_name: str,
    current_entity: str,
) -> str:
    clean_persona = str(persona_name or '').strip()
    if _is_system_persona_name(clean_persona):
        clean_persona = ''
    path = save_session_route_state(
        session_id,
        {
            'selected_route': route.selected_route,
            'request_type': route.request_type,
            'interaction_mode': route.interaction_mode,
            'response_style': route.response_style,
            'persona_name': clean_persona,
            'current_entity': str(current_entity or '').strip(),
            'persona_style_traits': list(preprocessing.persona_style_traits),
            'speech_style_hints': list(preprocessing.speech_style_hints),
        },
    )
    return str(path)


def _compact_model_budget(mode: str = 'chat', *, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = dict(meta or get_last_model_call_meta(mode) or {})
    if not meta:
        return {}
    return {
        'mode': str(meta.get('mode') or mode),
        'role': str(meta.get('role') or ''),
        'estimated_input_tokens': int(meta.get('estimated_input_tokens') or meta.get('prompt_tokens') or 0),
        'prompt_tokens': int(meta.get('prompt_tokens') or meta.get('estimated_input_tokens') or 0),
        'configured_n_ctx': int(meta.get('configured_n_ctx') or 0),
        'n_ctx': int(meta.get('n_ctx') or 0),
        'reserved_output_budget': int(meta.get('reserved_output_budget') or 0),
        'actual_max_tokens': int(meta.get('actual_max_tokens') or 0),
        'max_prompt_tokens': int(meta.get('max_prompt_tokens') or 0),
        'prompt_nearly_fills_window': bool(meta.get('prompt_nearly_fills_window')),
        'model_context_exhausted': bool(meta.get('model_context_exhausted') or meta.get('prompt_exceeds_window_budget') or meta.get('logical_context_exhausted')),
        'prompt_exceeds_window_budget': bool(meta.get('prompt_exceeds_window_budget') or meta.get('model_context_exhausted') or meta.get('logical_context_exhausted')),
        'input_truncated': bool(meta.get('truncated') or meta.get('input_truncated')),
        'output_truncated': bool(meta.get('output_truncated')),
        'output_budget_too_small': bool(meta.get('output_budget_too_small')),
        'finish_reason': str(meta.get('finish_reason') or '').strip(),
        'window_shortfall_tokens': int(meta.get('window_shortfall_tokens') or 0),
        'budget_pressure_ratio': float(meta.get('budget_pressure_ratio') or 0.0),
        'n_ctx_auto_expanded': bool(meta.get('n_ctx_auto_expanded')),
    }


def _chat_role_for_request(*, persona_selected: bool) -> str:
    roles = get_runtime_config().roles
    if persona_selected and roles.use_general_for_persona_chat:
        return 'general'
    return str(roles.chat or 'general').strip() or 'general'


def _session_turn_index(session_id: str) -> int:
    parsed = parse_session(session_id) or {}
    messages = list(parsed.get('messages') or [])
    user_turns = [item for item in messages if str(item.get('role') or '').strip() == 'user']
    return max(len(user_turns), 0)


def _resolve_chat_orchestration_roles(
    *,
    session_id: str,
    request_id: str,
    persona_selected: bool,
) -> dict[str, Any]:
    runtime = get_runtime_config()
    config = runtime.chat_orchestration
    default_primary = _chat_role_for_request(persona_selected=persona_selected)
    primary = str(config.primary_role or default_primary).strip().lower() or default_primary
    reviewer = str(config.reviewer_role or runtime.roles.rethink or 'analyst').strip().lower() or 'analyst'
    strategy = str(config.strategy or 'primary_with_reviewer').strip().lower() or 'primary_with_reviewer'
    review_mode = str(config.review_mode or 'on_failure').strip().lower() or 'on_failure'
    turn_index = _session_turn_index(session_id)
    if strategy == 'single':
        reviewer = primary
    elif strategy == 'alternate':
        if turn_index % 2 == 1:
            primary, reviewer = reviewer, primary
    elif strategy == 'randomized':
        digest = hashlib.sha256(f'{session_id}:{request_id}'.encode('utf-8')).hexdigest()
        if int(digest[:2], 16) % 2 == 1:
            primary, reviewer = reviewer, primary
    else:
        strategy = 'primary_with_reviewer'
    return {
        'strategy': strategy,
        'review_mode': review_mode,
        'primary_role': primary,
        'reviewer_role': reviewer,
        'turn_index': turn_index,
        'reviewer_enabled': bool(reviewer and reviewer != primary and strategy != 'single'),
    }


def _build_reviewer_prompt(
    *,
    question: str,
    draft_reply: str,
    route: RouteDecision,
    route_guidance_block: str,
    persona_block: str,
    graph_context: str,
    recent_dialogue: str,
    language: str,
    validation_hint: str = '',
) -> str:
    def _compact(value: str, *, max_chars: int) -> str:
        raw = str(value or '').strip()
        if not raw:
            return ''
        lines = [line.strip() for line in raw.replace('\r', '\n').splitlines() if line.strip()]
        unique: list[str] = []
        seen: set[str] = set()
        for line in lines:
            key = ' '.join(line.lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(line)
            if len('\n'.join(unique)) >= max_chars:
                break
        packed = '\n'.join(unique).strip()
        return packed[:max_chars].strip() if len(packed) > max_chars else packed

    route_guidance_block = _compact(route_guidance_block, max_chars=420)
    persona_block = _compact(persona_block, max_chars=900)
    graph_context = _compact(graph_context, max_chars=700)
    recent_dialogue = _compact(recent_dialogue, max_chars=520)
    draft_reply = _compact(draft_reply, max_chars=700)
    question = _compact(question, max_chars=260)
    validation_hint = _compact(validation_hint, max_chars=160)
    target = normalize_language_code(language, fallback='en')
    if target == 'ru':
        parts = [
            '/no_think',
            'Верни только финальный ответ пользователю plain text.',
            'Ты второй проверяющий model.',
            'Первый model уже написал черновик. Твоя задача: проверить его по выбранному route и, если нужно, переписать лучше.',
            'Не объясняй проверку. Не пиши JSON. Не добавляй комментарии о правилах.',
            f'Route: {route.selected_route}.',
            'Сохрани задачу пользователя и язык ответа.',
            'Если черновик слабый, исправь его. Если он хороший, верни улучшенную, но короткую финальную версию.',
            'Особенно следи за route guidance, persona consistency, unsupported facts, truncation, repetition, and overconfident drift.',
        ]
        if validation_hint:
            parts.extend(['Validation hint:', validation_hint])
        if route_guidance_block:
            parts.extend(['Route guidance:', route_guidance_block])
        if persona_block:
            parts.extend(['Persona head:', persona_block])
        if graph_context:
            parts.extend(['Knowledge graph:', graph_context])
        if recent_dialogue:
            parts.extend(['Recent dialogue:', recent_dialogue])
        parts.extend(
            [
                'User question:',
                str(question or '').strip(),
                'Draft reply:',
                str(draft_reply or '').strip(),
            ]
        )
        return '\n\n'.join(part for part in parts if str(part).strip())
    parts = [
        '/no_think',
        'Return only the final user-facing reply as plain text.',
        'You are the second reviewing model.',
        'The first model already produced a draft. Check it against the selected route and rewrite it if needed.',
        'Do not explain the review. Do not output JSON. Do not add commentary about instructions.',
        f'Route: {route.selected_route}.',
        'Preserve the user task and the reply language.',
        'If the draft is weak, fix it. If it is already good, return a tighter final version.',
        'Pay special attention to route guidance, persona consistency, unsupported facts, truncation, repetition, and overconfident drift.',
    ]
    if validation_hint:
        parts.extend(['Validation hint:', validation_hint])
    if route_guidance_block:
        parts.extend(['Route guidance:', route_guidance_block])
    if persona_block:
        parts.extend(['Persona head:', persona_block])
    if graph_context:
        parts.extend(['Knowledge graph:', graph_context])
    if recent_dialogue:
        parts.extend(['Recent dialogue:', recent_dialogue])
    parts.extend(
        [
            'User question:',
            str(question or '').strip(),
            'Draft reply:',
            str(draft_reply or '').strip(),
        ]
    )
    return '\n\n'.join(part for part in parts if str(part).strip())


def _logical_context_snapshot(built: dict[str, Any]) -> dict[str, Any]:
    estimated_tokens = max(int(built.get('estimated_tokens') or 0), 0)
    logical_budget = max(int(get_runtime_config().context.max_context_tokens or 0), 0)
    context_debug = dict(built.get('context_debug') or {})
    selected_items = list(context_debug.get('selected_items') or [])
    pack_stage = dict(dict(context_debug.get('stages') or {}).get('pack_context') or {})
    return {
        'assembled_context_tokens': estimated_tokens,
        'logical_token_budget': logical_budget,
        'nearly_fills_logical_budget': bool(logical_budget and estimated_tokens >= max(int(logical_budget * 0.9), 1)),
        'selected_item_count': len(selected_items),
        'source_counts': dict(context_debug.get('source_counts') or {}),
        'packed_candidate_counts': pack_stage,
    }


def _finalize_model_budget(
    *,
    mode: str,
    planned_budget: dict[str, Any],
    generation_invoked: bool,
    skip_reason: str = '',
) -> dict[str, Any]:
    resolved = dict(planned_budget or {})
    actual = _compact_model_budget(mode=mode)
    if actual:
        resolved.update(actual)
        resolved['budget_source'] = 'actual'
    else:
        resolved['budget_source'] = 'planned'
    resolved['generation_invoked'] = bool(generation_invoked)
    resolved['generation_skipped'] = not bool(generation_invoked)
    resolved['skip_reason'] = str(skip_reason or '').strip() if not generation_invoked else ''
    resolved['prompt_exceeds_window_budget'] = bool(
        resolved.get('prompt_exceeds_window_budget')
        or resolved.get('model_context_exhausted')
        or resolved.get('input_truncated')
    )
    resolved['model_context_exhausted'] = bool(
        resolved.get('model_context_exhausted')
        or resolved.get('prompt_exceeds_window_budget')
        or resolved.get('input_truncated')
    )
    return resolved


def _model_budget_trace_meta(model_budget: dict[str, Any]) -> dict[str, Any]:
    return {
        'estimated_input_tokens': int(model_budget.get('estimated_input_tokens') or 0),
        'configured_n_ctx': int(model_budget.get('configured_n_ctx') or 0),
        'n_ctx': int(model_budget.get('n_ctx') or 0),
        'reserved_output_budget': int(model_budget.get('reserved_output_budget') or 0),
        'actual_max_tokens': int(model_budget.get('actual_max_tokens') or 0),
        'prompt_nearly_fills_window': bool(model_budget.get('prompt_nearly_fills_window')),
        'prompt_exceeds_window_budget': bool(model_budget.get('prompt_exceeds_window_budget')),
        'output_truncated': bool(model_budget.get('output_truncated')),
        'output_budget_too_small': bool(model_budget.get('output_budget_too_small')),
        'input_truncated': bool(model_budget.get('input_truncated')),
        'budget_source': str(model_budget.get('budget_source') or ''),
        'generation_invoked': bool(model_budget.get('generation_invoked')),
        'generation_skipped': bool(model_budget.get('generation_skipped')),
    }


def _route_output_budget(route: RouteDecision, *, repair: bool = False) -> int:
    selected = str(route.selected_route or '').strip()
    if selected == 'persona_graph_reasoning':
        return 512 if repair else 320
    if selected == 'persona_chat_fast_path':
        return 448 if repair else 280
    if selected == 'persona_dialogue_analysis':
        return 512 if repair else 320
    if selected in {'hypothetical_roleplay', 'project_document_analysis'}:
        return 448 if repair else 280
    if selected == 'meta_previous_answer':
        return 704 if repair else 512
    if selected == 'factual_answer':
        return 512 if repair else 320
    return 448 if repair else 256


def _answer_perspective_for_route(route: RouteDecision) -> str:
    # Any route with a persona selected → speak AS the persona, not as an assistant
    if route.requires_persona:
        return 'persona'
    if route.selected_route in {'persona_graph_reasoning', 'persona_chat_fast_path', 'hypothetical_roleplay'}:
        return 'persona'
    if route.selected_route == 'persona_dialogue_analysis':
        return 'persona_review'
    return 'assistant'


def _repair_route_guidance(route_guidance: str) -> str:
    parts = [
        str(route_guidance or '').strip(),
        'Repair guidance: the previous draft was truncated or became repetitive.',
        'Answer the same task again in a tighter form.',
        'Do not repeat clauses or restate the persona introduction more than once.',
        'Finish the answer cleanly in 2 to 5 concrete sentences.',
    ]
    return '\n'.join(part for part in parts if part)


def _style_guard_route_guidance(route_guidance: str, route: RouteDecision) -> str:
    traits = ', '.join(str(item).strip() for item in list(route.persona_style_traits or []) if str(item).strip())
    hints = [str(item).strip() for item in list(route.speech_style_hints or []) if str(item).strip()]
    parts = [
        str(route_guidance or '').strip(),
        'Style guard repair: the previous draft sounded too analytical, too confident, too dominant, or too polished for the selected persona.',
        'Do not analyze the situation, the other person, or the dialogue.',
        'Do not lecture, explain, command, or structure the answer as steps.',
        'Reply in 1 to 3 short lines with hesitation and awkward incompleteness when pressure is present.',
        'Keep the voice defensive, irritated, or evasive rather than articulate and composed.',
        'If the topic is embarrassing, dependent, or tied to a significant person, weaken externally instead of becoming clearer or stronger.',
        'Do not state feelings too directly; dodge, deflect, or trail off when that pressure is part of the persona.',
        'Do not invent new facts or background events.',
    ]
    if traits:
        parts.append(f'Preserve these persona tensions in the wording: {traits}.')
    if hints:
        parts.append('Speaking manner to preserve:')
        parts.extend(f'- {item}.' for item in hints[:6])
    return '\n'.join(part for part in parts if part)


def _style_guard_fallback_reply(route: RouteDecision, *, language: str) -> str:
    target = normalize_language_code(language, fallback='en')
    if target == 'ru':
        if route.selected_route == 'hypothetical_roleplay':
            return 'Эм...\nНе хочу об этом.\nНаверное, исчезну.'
        return 'Эм...\nНе хочу это раскладывать.\nОтойду.'
    if route.selected_route == 'hypothetical_roleplay':
        return "Um...\nI do not want to say it.\nI would probably just pull away."
    return "Um...\nI do not want to spell it out.\nI would step back."


def _resolve_explicit_persona_name(*candidates: Any) -> str:
    for candidate in candidates:
        clean_candidate = str(candidate or '').strip()
        if not clean_candidate:
            continue
        normalized = normalize_personality_name(clean_candidate)
        if normalized and not _is_system_persona_name(normalized):
            return normalized
    return ''


def _apply_trace_learning_to_guidance(route_guidance: str, trace_learning: TraceLearningPolicy) -> str:
    learned_lines = [str(item).strip() for item in list(trace_learning.route_guidance_lines or []) if str(item).strip()]
    if not learned_lines:
        return route_guidance
    parts = [
        str(route_guidance or '').strip(),
        'Runtime lessons from recent similar traces:',
        *learned_lines,
    ]
    return '\n'.join(part for part in parts if part)


def _route_output_budget_with_learning(route: RouteDecision, trace_learning: TraceLearningPolicy, *, repair: bool = False) -> int:
    base = _route_output_budget(route, repair=repair)
    boost = int(trace_learning.output_budget_boost or 0)
    if repair and boost:
        boost = max(boost, 128)
    return base + max(boost, 0)


def _style_guard_output_budget(route: RouteDecision, trace_learning: TraceLearningPolicy) -> int:
    base = _route_output_budget_with_learning(route, trace_learning, repair=False)
    if route.selected_route == 'persona_graph_reasoning':
        return min(base, 320)
    if route.selected_route == 'hypothetical_roleplay':
        return min(base, 288)
    return min(base, 320)


def _operator_messages_from_model_budget(model_budget: dict[str, Any]) -> list[str]:
    if not model_budget:
        return []
    messages: list[str] = []
    if model_budget.get('prompt_exceeds_window_budget'):
        configured_n_ctx = int(model_budget.get('configured_n_ctx') or 0)
        n_ctx = int(model_budget.get('n_ctx') or 0)
        if configured_n_ctx and n_ctx <= configured_n_ctx:
            messages.append(
                f'The prompt exceeded the configured model window budget at n_ctx={configured_n_ctx}; the current architecture needs a larger n_ctx or tighter context selection.'
            )
        else:
            messages.append(
                'The prompt exceeded the model prompt budget and had to be reduced before generation.'
            )
    elif model_budget.get('prompt_nearly_fills_window'):
        messages.append(
            'The prompt used most of the available model window; a longer reply may still be clipped unless output space is reserved carefully.'
        )
    if model_budget.get('n_ctx_auto_expanded'):
        messages.append(
            f"The runtime expanded n_ctx from {int(model_budget.get('configured_n_ctx') or 0)} to {int(model_budget.get('n_ctx') or 0)} to preserve reply space."
        )
    if model_budget.get('output_truncated'):
        messages.append(
            f"The model stopped at the output budget (max_tokens={int(model_budget.get('actual_max_tokens') or 0)}); increase reserved output space if replies are being cut off."
        )
    elif model_budget.get('output_budget_too_small'):
        messages.append(
            'The output budget was too small for the generated reply; the next request should reserve more completion space.'
        )
    return messages


# ---------------------------------------------------------------------------
# Personality construction integration helpers
# ---------------------------------------------------------------------------

def _build_behavioral_action_block(
    persona_name: str,
    route_name: str,
    situation_type: str = 'neutral_statement',
) -> str:
    """
    Build an action directive block for the LLM prompt.

    Returns a compact directive string if a personality exists for the persona
    and the route is persona_chat.  Returns empty string otherwise.

    The directive instructs the LLM WHAT to do (the selected action) without
    letting the LLM override that choice.
    """
    if not persona_name:
        return ''
    if route_name not in ('persona_chat', 'persona_specification'):
        return ''
    try:
        from .personality_store import load_personality_for_persona
        from .behavioral_action_engine import select_action_for_persona
        personality = load_personality_for_persona(persona_name)
        if personality is None:
            return ''
        result = select_action_for_persona(
            personality,
            route_name,
            situation_type=situation_type,
        )
        if result.verbalization_directive:
            return result.verbalization_directive
    except Exception:
        pass
    return ''


def _maybe_update_personality_from_chat(
    persona_name: str,
    user_message: str,
    session_id: str,
) -> None:
    """
    Parse the user message for structured personality updates and apply them.

    This is a non-blocking, best-effort operation. Failures are silently swallowed
    so that a personality parse error never breaks the chat turn.

    Only runs when a persona is selected.  Only messages with meaningful content
    (more than 8 words) are parsed to avoid noise.
    """
    if not persona_name or not user_message:
        return
    clean = ' '.join(str(user_message or '').strip().split())
    if len(clean.split()) < 5:
        return
    try:
        from .personality_store import load_personality_for_persona, save_personality
        from .personality_update_parser import parse_update_candidates
        from .personality_merge_engine import apply_update_candidates
        from .personality_summarizer import update_hover_summary
        personality = load_personality_for_persona(persona_name)
        if personality is None:
            return
        candidates = parse_update_candidates(
            clean,
            source_channel='chat_user',
            session_id=session_id,
        )
        if not candidates:
            return
        apply_update_candidates(personality, candidates)
        update_hover_summary(personality)
        save_personality(personality)
    except Exception:
        pass


def _make_note_command_result(
    *,
    reply: str,
    session_id: str,
    session: dict[str, Any],
    side_effects: ChatSideEffects,
    language: str = 'en',
) -> ChatTurnResult:
    """Build a minimal ChatTurnResult for note-command fast paths (no LLM call)."""
    return ChatTurnResult(
        assistant_reply=reply,
        response_language=language,
        session_id=session_id,
        trace_id='',
        session=session,
        persona_name='',
        graph_context='',
        current_entity='',
        analysis=MessageAnalysis(message='', session_id=session_id),
        classifications=[],
        repair_status={'status': 'ok', 'reason': 'note_command_fast_path'},
        pipeline={'fast_path': True, 'route': 'note_command'},
        side_effects=side_effects,
        operator_messages=['Note command handled via fast path — no LLM call.'],
    )


def _build_regulator_directive(
    message: str,
    *,
    personality_name: str = '',
    overload_level: float = 0.0,
    feedback_class: str = 'neutral',
) -> tuple[str, dict[str, Any]]:
    """
    Run the regulator pipeline and return (action_directive, regulator_trace).

    Used to inject a behavioral directive into the planning/chat prompt.
    """
    pressure_response: list[str] = []
    vulnerabilities: list[str] = []
    defense_mechanisms: list[str] = []

    if personality_name:
        try:
            from .personality_store import load_personality_for_persona
            personality = load_personality_for_persona(personality_name)
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
        except Exception:
            pass

    reg_output = run_regulator_pipeline(
        message=message,
        overload_level=overload_level,
        feedback_class=feedback_class,
        personality_pressure_response=pressure_response,
        personality_vulnerabilities=vulnerabilities,
        personality_defense_mechanisms=defense_mechanisms,
    )
    directive = reg_output.render_action_directive()
    return directive, reg_output.to_dict()


def _build_planning_context_block(
    message: str,
    *,
    session_id: str,
    personality_name: str = '',
    language: str = 'en',
    recent_feedback: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Build the planning context block for injection into the LLM prompt.

    Returns (planning_prompt_section, planning_trace_dict).
    """
    planning_output = build_planning_output(
        message=message,
        session_id=session_id,
        personality_name=personality_name,
        recent_feedback=list(recent_feedback or []),
        language=language,
    )

    if planning_output.is_crisis:
        # Return the canned crisis response directly — no LLM needed
        return planning_output.crisis_response, planning_output.to_dict()

    context_section = planning_output.render_planning_prompt_section()
    return context_section, planning_output.to_dict()


def _record_importance_learning(
    session_id: str,
    message: str,
    *,
    was_saved: bool = False,
) -> None:
    """Record a turn for the importance learner (non-blocking)."""
    try:
        if was_saved:
            record_positive_example(session_id, message)
        else:
            record_negative_example(session_id, message)
    except Exception:
        pass


def run_chat_turn(request: ChatTurnRequest) -> ChatTurnResult:
    raw_message = '' if request.message is None else str(request.message)
    clean_message = raw_message.strip()
    session = create_session(request.session_id or '')
    clean_session_id = str(session.get('session_id') or request.session_id or '')

    # -----------------------------------------------------------------------
    # NOTE COMMAND FAST PATH — handle /save, /notes, /del_note, /clear_notes
    # before any routing, graph writes, or LLM calls.
    # -----------------------------------------------------------------------
    _note_cmd = parse_note_command(clean_message)
    if _note_cmd is not None:
        _side_effects_nc = ChatSideEffects()
        _reply_nc = execute_note_command(clean_session_id, _note_cmd)
        # If the user saved a note, record a positive importance example
        if _note_cmd.get('cmd') == 'save':
            _record_importance_learning(clean_session_id, str(_note_cmd.get('text') or ''), was_saved=True)
        # Persist the command and reply in session history (important for context continuity)
        append_turn(clean_session_id, raw_message, _reply_nc)
        return _make_note_command_result(
            reply=_reply_nc,
            session_id=clean_session_id,
            session=session,
            side_effects=_side_effects_nc,
            language=request.language or 'en',
        )

    graph_store = GraphStore()
    side_effects = ChatSideEffects()
    observability = get_observability_store()
    trace = observability.start_trace(
        request_type='chat',
        route='/api/cognitive/chat/respond',
        session_id=clean_session_id,
        request_meta={
            'language': request.language,
            'message_chars': len(raw_message),
            'selected_persona': str(request.selected_persona or '').strip(),
        },
    )
    try:
        envelope = observability.time_stage(
            trace,
            'request_intake',
            lambda: build_request_envelope(
                request_id=trace.request_id,
                session_id=clean_session_id,
                raw_text=raw_message,
            ),
            meta_builder=lambda item: {
                'request_id': item.request_id,
                'session_id': item.session_id,
                'timestamp': item.timestamp,
                'raw_chars': len(item.raw_text),
            },
        )
        trace.request_meta.update(
            {
                'request_id': envelope.request_id,
                'raw_text': envelope.raw_text,
                'normalized_text': envelope.normalized_text,
                'timestamp': envelope.timestamp,
            }
        )
        observability.time_stage(
            trace,
            'graph_prewrite',
            lambda: _write_concept_graph_from_message(
                clean_message,
                graph_store=graph_store,
                side_effects=side_effects,
                session_id=clean_session_id,
            ),
            meta_builder=lambda _result: {'graph_write_sources': list(side_effects.graph_write_sources)},
        )
        current_entity = infer_current_entity(clean_session_id)
        session_persona = infer_session_active_persona(clean_session_id)
        route_memory = _load_route_memory(clean_session_id)
        known_nodes = graph_store.load_nodes()
        controller = observability.time_stage(
            trace,
            'controller_interpretation',
            lambda: build_controller_state(
                request_id=trace.request_id,
                session_id=clean_session_id,
                message=clean_message,
                selected_persona=str(request.selected_persona or '').strip(),
                explicit_context=request.explicit_context,
                current_entity=current_entity,
                session_persona=session_persona,
                route_memory=route_memory,
                known_entities=known_nodes,
            ),
            meta_builder=lambda item: {
                'request_type': item.preprocessing.request_type,
                'selected_route': item.route.selected_route,
                'interaction_mode': item.preprocessing.interaction_mode,
                'message_kind': item.interaction_frame.message_kind,
                'topic_entity': item.interaction_frame.topic_entity,
                'requires_history': bool(item.route.requires_history),
                'requires_graph': bool(item.route.requires_graph),
                'requires_persona': bool(item.route.requires_persona),
                'fast_path': bool(item.route.fast_path),
            },
        )
        interaction_frame = controller.interaction_frame
        analysis = controller.analysis
        preprocessing = controller.preprocessing
        route = controller.route
        capability_plan = controller.capability_plan
        effective_selected_persona = controller.selected_persona
        trace_learning = observability.time_stage(
            trace,
            'trace_learning',
            lambda: (
                observability.learned_route_policy(
                    session_id=clean_session_id,
                    selected_route=route.selected_route,
                )
                if route.requires_llm
                else TraceLearningPolicy(selected_route=route.selected_route)
            ),
            meta_builder=lambda item: {
                'selected_route': item.selected_route,
                'batch_size': int(item.batch_size or 0),
                'sampled_trace_count': int(item.sampled_trace_count or 0),
                'signal_trace_count': int(item.signal_trace_count or 0),
                'failure_density': float(item.failure_density or 0.0),
                'output_budget_boost': int(item.output_budget_boost or 0),
                'reason_codes': list(item.reason_codes),
            },
        )
        route_guidance = _apply_trace_learning_to_guidance(render_route_guidance(route), trace_learning)

        # ── FUTURE INTEGRATION POINT ──────────────────────────────────────────
        # When SpeechPlanner replaces staged prompts, insert here:
        #
        #   from .speech_planner import SpeechPlanner, verbalizer_prompt
        #   _cog_out, _new_reg = cognitive_runtime.forward(
        #       clean_message, genome, regulator_state
        #   )
        #   _speech_plan = SpeechPlanner().build(
        #       _cog_out,
        #       persona_context={'speech_style': ..., 'language': response_language},
        #       user_text=clean_message,
        #   )
        #   prompt = verbalizer_prompt(
        #       _speech_plan,
        #       persona_voice=built.get('persona_voice_summary', ''),
        #       recent_exchange=built.get('recent_dialogue', ''),
        #   )
        #   # Then call LLM once with this prompt — all staged prompts are replaced.
        #
        # Until then: action directive is injected into the existing prompt as a hint.
        # ─────────────────────────────────────────────────────────────────────────

        # Personality-driven behavioral action directive.
        # Computed from the weighted PersonalityObject (not from LLM imagination).
        # Appended to route_guidance so the LLM verbalizes the pre-selected action.
        _effective_persona_for_action = str(
            request.selected_persona
            or analysis.selected_head
            or analysis.session_persona
            or route_memory.previous_persona_name
            or ''
        ).strip()
        _action_directive = _build_behavioral_action_block(
            _effective_persona_for_action,
            route.selected_route,
            situation_type=str(analysis.situation.type or 'neutral_statement'),
        )
        if _action_directive:
            route_guidance = (route_guidance + '\n\n' + _action_directive).strip()

        # Capture any personality updates from the user message (non-blocking).
        _maybe_update_personality_from_chat(
            _effective_persona_for_action,
            clean_message,
            clean_session_id,
        )

        # -----------------------------------------------------------------------
        # PLANNING MODE & SITUATION REGULATOR INJECTION
        #
        # If the user is asking for planning guidance, inject:
        #   1. Planning context (notes, personality summary, overload, forces)
        #   2. Regulator directive (selected behavioral action)
        #
        # The regulator pipeline runs on ALL turns (not just planning).
        # The planning context is only injected when planning intent is detected.
        # GUARD: coaching/regulator directives must NOT fire during persona roleplay.
        # When a persona is active on a persona route, the LLM must stay in character.
        # -----------------------------------------------------------------------
        _is_persona_route = route.selected_route in (
            'persona_chat',
            'persona_chat_fast_path',
            'persona_dialogue_analysis',
            'persona_specification',
            'persona_assignment',
        )
        _has_active_persona = bool(_effective_persona_for_action)
        _coaching_mode = not (_is_persona_route and _has_active_persona)

        _feedback_class = classify_feedback(clean_message)
        _regulator_directive, _regulator_trace = _build_regulator_directive(
            clean_message,
            personality_name=_effective_persona_for_action,
            overload_level=0.0,
            feedback_class=_feedback_class,
        )
        if _coaching_mode and _regulator_directive and not _action_directive:
            # Only inject regulator directive when personality action engine didn't fire
            # and we are NOT in an active persona roleplay session
            route_guidance = (route_guidance + '\n\n' + _regulator_directive).strip()

        _planning_context_block = ''
        _planning_trace: dict[str, Any] = {}
        if _coaching_mode and (
            detect_planning_intent(clean_message) or route.selected_route in ('planning_mode',)
        ):
            _planning_context_block, _planning_trace = _build_planning_context_block(
                clean_message,
                session_id=clean_session_id,
                personality_name=_effective_persona_for_action,
                language=request.language or 'en',
            )
            # If crisis detected, override with canned response immediately
            if _planning_trace.get('is_crisis'):
                _crisis_reply = str(_planning_trace.get('crisis_response') or '')
                if _crisis_reply:
                    append_turn(clean_session_id, raw_message, _crisis_reply)
                    return _make_note_command_result(
                        reply=_crisis_reply,
                        session_id=clean_session_id,
                        session=session,
                        side_effects=side_effects,
                        language=request.language or 'en',
                    )
            if _planning_context_block:
                route_guidance = (_planning_context_block + '\n\n' + route_guidance).strip()

        # -----------------------------------------------------------------------
        # SAFETY CLASSIFICATION
        # Deterministic KNN classifier — no LLM, runs before verbalization.
        # illegal → immediate block response, LLM is never called.
        # explicit/suggestive → inject behavior modifier into route_guidance.
        # -----------------------------------------------------------------------
        _safety_language = str(request.language or analysis.user_state.language or 'en')
        try:
            _safety_result, _safety_instruction = run_safety_check(
                clean_message,
                language=_safety_language,
            )
            if _safety_instruction:
                route_guidance = (route_guidance + '\n\n' + _safety_instruction).strip()
        except SafetyBlockError:
            _block_reply = get_block_response(_safety_language)
            append_turn(clean_session_id, raw_message, _block_reply)
            return _make_note_command_result(
                reply=_block_reply,
                session_id=clean_session_id,
                session=session,
                side_effects=side_effects,
                language=_safety_language,
            )

        # -----------------------------------------------------------------------
        # LOCALIZATION CONTEXT
        # Build language-specific voice guide injected into verbalization prompt.
        # Guides the LLM to produce natively-sounding output in the target language.
        # -----------------------------------------------------------------------
        _localization_block = build_localization_prompt(
            target_language=_safety_language,
            persona_context=_effective_persona_for_action or None,
            user_message=clean_message,
            system_intent=route_guidance[:300] if route_guidance else '',
        )
        if _localization_block:
            route_guidance = (route_guidance + '\n\n' + _localization_block).strip()

        # -----------------------------------------------------------------------
        # COGNITIVE PIPELINE  (P1–P6 perceptrons)
        # Runs deterministically. Result injects action context into verbalization.
        # Training tuple is recorded for offline calibration.
        # -----------------------------------------------------------------------
        _cog_output = None
        _cog_authority_score = 0.0
        _cog_mode = 'pure_llm'
        try:
            from .cognitive_authority import (
                cognitive_authority_score, authority_mode,
                build_cognitive_hint_block, _count_training_tuples,
            )
            _cfg = get_runtime_config()
            _cog_store = GenomeStore(Path(_cfg.memory_root))
            _cog_rt = _cog_store.load_runtime()
            _cog_genome = _cog_store.load_genome(
                str(getattr(request, 'personality_name', '') or selected_persona or 'default')
            )
            _cog_reg = _cog_store.load_regulator(clean_session_id, _cog_genome)
            _cog_ctx = {
                'prior_failure': False,
                'repeated': False,
                'escalating': False,
                'resolved': False,
            }
            _cog_output, _new_cog_reg = _cog_rt.forward(
                clean_message, _cog_genome, _cog_reg, ctx=_cog_ctx
            )
            _cog_store.save_regulator(clean_session_id, _new_cog_reg)

            # ── Compute authority score ──────────────────────────────────────
            _session_turns = len(list(parse_session(clean_session_id))) if clean_session_id else 0
            _n_nodes = len(list(graph_store.load_nodes())) if graph_store else 0
            _n_tuples = _count_training_tuples(Path(_cfg.memory_root))
            _reg_vals = list(_new_cog_reg.values.tolist())
            _cog_authority_score = cognitive_authority_score(
                session_turns   = _session_turns,
                graph_nodes     = _n_nodes,
                genome_version  = int(_cog_genome.version),
                training_tuples = _n_tuples,
                regulator_values = _reg_vals,
            )
            _cog_mode = authority_mode(_cog_authority_score)

            # ── Inject cognitive signal according to mode ────────────────────
            _cog_vc = _cog_output.verbalization_context()
            if _cog_mode == 'hint':
                _hint = build_cognitive_hint_block(_cog_vc)
                if _hint:
                    route_guidance = (route_guidance + '\n\n' + _hint).strip()
            elif _cog_mode == 'planner':
                # Full planner takes over at prompt-build time below;
                # still record the verbalization context for logging.
                _cog_block = (
                    f'[Internal state] action={_cog_vc["action"]} '
                    f'risk={_cog_vc["perceived_risk"]} event={_cog_vc["primary_event"]}'
                )
                route_guidance = (route_guidance + '\n\n' + _cog_block).strip()
            # pure_llm: inject nothing, let LLM answer freely

            # ── Record training tuple (always) ───────────────────────────────
            import uuid as _uuid
            _turn_id = str(_uuid.uuid4())[:8]
            _ts = TrainingStore(Path(_cfg.memory_root) / 'training_tuples.jsonl')
            _ts.append(TrainingTuple(
                session_id=clean_session_id,
                turn_id=_turn_id,
                timestamp=__import__('time').time(),
                text=clean_message,
                genome_id=_cog_genome.persona_id,
                action_id=_cog_output.action_id,
                event_probs=_cog_output.event_probs,
                action_probs=_cog_output.action_probs,
                thought_vec=_cog_output.thought_vec,
                conflict_vec=_cog_output.conflict_vec,
                regulator_snapshot=_cog_output.regulator_state,
            ))
        except Exception as _cog_err:
            import logging as _log
            _log.getLogger(__name__).debug('cognitive pipeline skipped: %s', _cog_err)

        # Record this turn for the importance learner (non-blocking)
        _record_importance_learning(clean_session_id, clean_message, was_saved=False)

        history_available = _session_has_previous_assistant_turn(clean_session_id)
        trace.request_meta.update(
            {
                'request_type': preprocessing.request_type,
                'detected_language': preprocessing.detected_language,
                'intent_type': preprocessing.intent_type,
                'interaction_mode': preprocessing.interaction_mode,
                'selected_route': route.selected_route,
                'requires_history': route.requires_history,
                'requires_graph': route.requires_graph,
                'requires_persona': route.requires_persona,
                'requires_llm': route.requires_llm,
                'strict_grounding': route.strict_grounding,
                'response_style': route.response_style,
                'validation_mode': route.validation_mode,
                'previous_route': route_memory.previous_route,
                'trace_learning_batch_size': int(trace_learning.batch_size or 0),
                'trace_learning_output_budget_boost': int(trace_learning.output_budget_boost or 0),
            }
        )
        response_language = _effective_response_language(
            requested_language=request.language,
            detected_language=analysis.user_state.language,
        )
        analysis_message = str(interaction_frame.routed_message or clean_message).strip()
        chat_orchestration = _resolve_chat_orchestration_roles(
            session_id=clean_session_id,
            request_id=trace.request_id,
            persona_selected=bool(request.selected_persona or analysis.selected_head or analysis.session_persona),
        )
        reasoning_message = _internal_reasoning_message(
            analysis_message,
            detected_language=analysis.user_state.language,
        )
        trace.request_meta.update(
            {
                'chat_orchestration_strategy': str(chat_orchestration.get('strategy') or ''),
                'chat_primary_role': str(chat_orchestration.get('primary_role') or ''),
                'chat_reviewer_role': str(chat_orchestration.get('reviewer_role') or ''),
                'chat_review_mode': str(chat_orchestration.get('review_mode') or ''),
            }
        )
        if route.selected_route in {'persona_specification', 'persona_assignment'}:
            context_sources_used: list[str] = []
            fallback_used = False
            fallback_reason = ''
            logical_context: dict[str, Any] = {}
            model_budget: dict[str, Any] = {}
            runtime_status: dict[str, Any] = {}
            persona_action: dict[str, Any] = {}
            persona_name = ''
            current_entity_for_state = ''
            operator_messages: list[str] = []

            if route.selected_route == 'persona_specification':
                try:
                    persona_action = observability.time_stage(
                        trace,
                        'persona_object_creation',
                        lambda: create_persona_from_description(
                            clean_message,
                            name_hint=str(request.selected_persona or '').strip(),
                            session_persona=session_persona,
                            activate=True,
                        ),
                        meta_builder=lambda item: {
                            'created': bool(item.get('created')),
                            'activated': bool(item.get('activated')),
                            'persona_name': str(item.get('persona_name') or ''),
                        },
                    )
                    persona_name = str(persona_action.get('persona_slug') or persona_action.get('persona_name') or '').strip()
                    current_entity_for_state = str(persona_action.get('persona_name') or '').strip()
                    assistant_reply = _persona_creation_acknowledgement(
                        response_language,
                        persona_name=str(persona_action.get('persona_name') or 'persona').strip(),
                        activated=bool(persona_action.get('activated')),
                    )
                    operator_messages.append('The request was interpreted as persona specification, so the system created a structured persona object instead of generating a narrative answer.')
                except Exception as exc:
                    assistant_reply = (
                        'Не удалось создать личность как структурированный профиль.'
                        if response_language == 'ru'
                        else 'I could not create that persona as a structured profile.'
                    )
                    persona_action = {
                        'created': False,
                        'activated': False,
                        'error': str(exc),
                    }
                    operator_messages.append('Persona creation was attempted but rejected by persona validation or storage safeguards.')
            else:
                assignment_target = str(
                    request.selected_persona
                    or interaction_frame.requested_persona
                    or analysis.selected_head
                    or route_memory.previous_persona_name
                    or session_persona
                    or ''
                ).strip()
                active_bundle = load_active_persona(assignment_target) if assignment_target else None
                if active_bundle is None:
                    assistant_reply = (
                        'Не удалось активировать личность: в реестре нет валидной persona.'
                        if response_language == 'ru'
                        else 'I could not activate that persona because it is not present in the validated registry.'
                    )
                    persona_action = {
                        'assigned': False,
                        'persona_name': assignment_target,
                        'error': 'persona_not_found_in_validated_registry',
                    }
                    operator_messages.append('Persona assignment was blocked because the requested persona was not present in the validated registry.')
                else:
                    persona_name = normalize_personality_name(active_bundle.name)
                    current_entity_for_state = active_bundle.name
                    persona_action = {
                        'assigned': True,
                        'activated': True,
                        'persona_name': active_bundle.name,
                        'persona_slug': str(active_bundle.meta.get('slug') or normalize_personality_name(active_bundle.name)),
                    }
                    assistant_reply = _persona_assignment_acknowledgement(
                        response_language,
                        persona_name=active_bundle.name,
                    )
                    operator_messages.append('The request was interpreted as persona assignment, so the current persona was switched without running freeform chat generation.')

            validation = ResponseValidation(
                ok=bool(persona_action.get('created') or persona_action.get('assigned')),
                validation_mode=route.validation_mode,
                route_match=True,
                relevance_match=True,
                fallback_triggered=False,
                reason_codes=list(route.reason_codes),
            )
            repair_status = observability.time_stage(
                trace,
                'storage_writes',
                lambda: (
                    _write_session_history(clean_session_id, raw_message, assistant_reply, side_effects=side_effects, persona_name=str(persona_action.get('persona_name') or current_entity_for_state or '').strip(), user_persona_name=str(request.user_persona_name or '').strip()),
                    setattr(
                        side_effects,
                        'route_state_path',
                        _save_route_memory(
                            clean_session_id,
                            route=route,
                            preprocessing=preprocessing,
                            persona_name=str(persona_action.get('persona_name') or current_entity_for_state or '').strip(),
                            current_entity=str(current_entity_for_state or analysis.current_entity or '').strip(),
                        ),
                    ),
                    _apply_rebuild_schedule(
                        clean_session_id,
                        personality_name=str(persona_action.get('persona_name') or '').strip(),
                        side_effects=side_effects,
                    ),
                )[-1],
                meta_builder=lambda payload: {
                    'history_write_path': side_effects.history_write_path,
                    'route_state_path': side_effects.route_state_path,
                    'repair_status': str(payload.get('status') or ''),
                },
            )
            pipeline = _pipeline_payload(
                envelope=envelope,
                preprocessing=preprocessing,
                route=route,
                capability_plan=capability_plan,
                context_sources_used=context_sources_used,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                validation=validation,
                model_budget=model_budget,
                logical_context=logical_context,
                trace_learning=trace_learning,
            )
            trace.request_meta['context_sources_used'] = list(context_sources_used)
            observability.finish_trace(
                trace,
                status='ok' if validation.ok else 'degraded',
                fallback_used=False,
                fallback_reason='',
                context_tokens=0,
                persona_name=str(persona_action.get('persona_name') or '').strip(),
                current_entity=str(current_entity_for_state or '').strip(),
                response_meta={
                    'selected_route': route.selected_route,
                    'request_type': route.request_type,
                    'validation_result': validation.validation_mode,
                    'validation_ok': bool(validation.ok),
                    'repair_status': str(repair_status.get('status') or ''),
                    'response_language': response_language,
                    'persona_action': dict(persona_action),
                },
            )
            return ChatTurnResult(
                assistant_reply=assistant_reply,
                response_language=response_language,
                session_id=clean_session_id,
                trace_id=trace.request_id,
                session=parse_session(clean_session_id) or session,
                persona_name=str(persona_action.get('persona_name') or '').strip(),
                graph_context='',
                current_entity=str(current_entity_for_state or '').strip(),
                analysis=analysis,
                classifications=[],
                repair_status=repair_status,
                pipeline=pipeline,
                validation=validation,
                proposal_requested=False,
                side_effects=side_effects,
                behavior_trace={
                    'route': route.to_dict(),
                    'capability_plan': capability_plan.to_dict(),
                    'persona_action': dict(persona_action),
                },
                context_preview={
                    'selected_route': route.selected_route,
                    'persona_action': dict(persona_action),
                },
                runtime_status=runtime_status,
                operator_messages=list(dict.fromkeys(operator_messages)),
                persona_action=dict(persona_action),
            )
        if not capability_plan.use_heavy_persona_pipeline:
            simple_selected_persona = _resolve_explicit_persona_name(
                effective_selected_persona,
                request.selected_persona,
                analysis.selected_head,
                analysis.session_persona,
                route_memory.previous_persona_name,
            )
            built = observability.time_stage(
                trace,
                'context_building',
                lambda: (
                    _call_build_context_compat(
                        question=reasoning_message,
                        session_id=clean_session_id,
                        selected_persona=simple_selected_persona,
                        explicit_context=request.explicit_context,
                        situation=analysis.situation,
                        store=graph_store,
                        analysis=analysis,
                    )
                    if capability_plan.use_context_builder
                    else _simple_context_payload(
                        session_id=clean_session_id,
                        current_entity=analysis.current_entity,
                        route=route,
                        selected_persona=simple_selected_persona,
                    )
                ),
                meta_builder=lambda payload: {
                    'estimated_tokens': int(payload.get('estimated_tokens') or 0),
                    'selected_items': len(list(dict(payload.get('context_debug') or {}).get('selected_items') or [])),
                    'source_counts': dict(dict(payload.get('context_debug') or {}).get('source_counts') or {}),
                },
            )
            message_vector_runtime_payload = observability.time_stage(
                trace,
                'message_vector_runtime',
                lambda: build_runtime_message_vector_payload(
                    clean_session_id,
                    message_text=clean_message,
                    role='user',
                    persona_name=str(built.get('persona_name') or simple_selected_persona or '').strip(),
                ),
                meta_builder=lambda payload: {
                    'context_window': len(list(payload.get('context_window') or [])),
                    'history_flow': list(payload.get('history_flow') or []),
                    'transition_type': str(dict(payload.get('transition_interpretation') or {}).get('type') or ''),
                },
            )
            built['message_vector_runtime'] = dict(message_vector_runtime_payload)
            vector_guidance = _render_message_vector_guidance(message_vector_runtime_payload)
            if vector_guidance:
                route_guidance = (route_guidance + '\n\n' + vector_guidance).strip()

            # ── DIALOG CONTEXT MATRIX ─────────────────────────────────────
            # Записываем ход в контекстную матрицу сессии, получаем logic_ctx.
            try:
                from .dialog_tracker import record_turn as _record_turn
                _dialog_ctx = _record_turn(
                    clean_session_id,
                    text=clean_message,
                    speaker='user',
                )
                built['dialog_context'] = _dialog_ctx

                # F7 прагматический подтекст → route_guidance
                _f7_sub = str(_dialog_ctx.get('f7_subtext') or '').strip()
                _f6_dir = str(_dialog_ctx.get('f6_directive') or '').strip()
                if _f7_sub:
                    _subtext_block = f'[Прагматический подтекст: {_f7_sub[:300]}]'
                    route_guidance = (route_guidance + '\n\n' + _subtext_block).strip()
                if _f6_dir and _f6_dir != 'respond_normal':
                    _dir_block = f'[F6-директива: {_f6_dir}]'
                    route_guidance = (route_guidance + '\n' + _dir_block).strip()
            except Exception:
                built['dialog_context'] = {}

            context_sources_used = [
                source
                for source, enabled in (
                    ('session_history', bool(str(built.get('recent_dialogue') or '').strip())),
                    ('knowledge_graph', bool(str(built.get('graph_context') or '').strip())),
                    ('persona_graph', bool(str(built.get('persona_block') or '').strip())),
                    ('message_vector_runtime', bool(dict(built.get('message_vector_runtime') or {}))),
                )
                if enabled
            ]
            if _cog_output is not None and route.requires_llm:
                # ── SPEECH PLANNER PATH (mandatory) ─────────────────────────
                # P1–P6 pipeline is the decision-maker; LLM only verbalizes.
                # _cog_mode is still recorded for logging but no longer gates
                # this path — SpeechPlan is always used when cog output exists.
                from .speech_planner import SpeechPlanner, verbalizer_prompt as _vp
                _speech_plan = SpeechPlanner().build(
                    _cog_output,
                    built=built,
                    user_text=clean_message,
                    language=response_language,
                )
                prompt = _vp(
                    _speech_plan,
                    persona_voice=str(built.get('persona_block') or '')[:500],
                    recent_exchange=str(built.get('recent_dialogue') or '')[:600],
                )
            else:
                _speech_plan = None
                # ── LEGACY STAGED PROMPT PATH (no cognitive output) ──────────
                _mvr = dict(built.get('message_vector_runtime') or {})
                prompt = _call_build_chat_prompt_compat(
                    question=clean_message,
                    internal_question=reasoning_message,
                    persona_block=built.get('persona_block') or '',
                    graph_context=built.get('graph_context') or '',
                    recent_dialogue=built.get('recent_dialogue') or '',
                    language=response_language,
                    semantic_focus=dict(dict(built.get('context_debug') or {}).get('semantic_focus') or {}),
                    route_guidance_block=route_guidance,
                    answer_perspective=_answer_perspective_for_route(route),
                    user_persona_name=str(request.user_persona_name or '').strip(),
                    persona_name=str(built.get('persona_name') or simple_selected_persona or '').strip(),
                    context_matrix=list(_mvr.get('context_matrix') or []),
                    current_vector=dict(_mvr.get('current_vector') or {}),
                    user_affect=dict(built.get('dialog_context', {}).get('user_affect') or {}),
                )
            logical_context = _logical_context_snapshot(built)
            simple_persona_selected = bool(simple_selected_persona)
            simple_chat_role = str(chat_orchestration.get('primary_role') or _chat_role_for_request(persona_selected=simple_persona_selected)).strip() or _chat_role_for_request(persona_selected=simple_persona_selected)
            planned_model_budget = _compact_model_budget(
                mode='chat',
                meta=plan_model_budget(
                    prompt,
                    mode='chat',
                    role=simple_chat_role,
                    max_tokens_override=_route_output_budget_with_learning(route, trace_learning),
                ),
            )
            reset_last_model_call_meta('chat')
            generation_state = {'invoked': False}

            def _generate_simple_reply() -> tuple[str, bool, str, dict[str, Any], BehavioralFallbackDecision | None]:
                deterministic = deterministic_reply(route, language=response_language, history_available=history_available)
                if deterministic and route.selected_route in {'clarification_request'}:
                    return deterministic, False, '', {}, None
                if deterministic and route.selected_route == 'meta_previous_answer' and not history_available:
                    return deterministic, False, '', {}, None
                if not route.requires_llm:
                    return deterministic or '', False, '', {}, None
                grounded = any(
                    str(part or '').strip()
                    for part in (built.get('persona_block'), built.get('graph_context'), built.get('recent_dialogue'))
                )
                if route.strict_grounding and not grounded and (_cog_mode != 'pure_llm' or route.requires_graph):
                    base = deterministic or fallback_chat_reply(language=response_language, persona_selected=simple_persona_selected)
                    return base, True, 'no_grounding', {}, None
                status = runtime_status_snapshot()
                if str(dict(status or {}).get('mode') or '') == 'degraded':
                    if route.selected_route in {'lightweight_conversation', 'persona_chat_fast_path'} and simple_persona_selected:
                        reply = deterministic or fallback_chat_reply(language=response_language, persona_selected=simple_persona_selected)
                    else:
                        reply = deterministic or fallback_chat_reply(language=response_language, persona_selected=simple_persona_selected)
                    return reply, True, 'dependency_unavailable', status, None
                generation_state['invoked'] = True
                # ── FAST PATH: model-aware raw completion (no chat API) ──────
                # Qwen3.x: pre-filled <think> skips reasoning (1.5–4s).
                # Mistral/LLaMA-family: direct [INST] prompt, no thinking tokens.
                # Falls through to legacy generate_chat_reply if unavailable.
                reply = ''
                if _speech_plan is not None:
                    try:
                        from src.utils.local_llm_provider import get_role_llm_instance
                        from .speech_planner import (
                            mistral_fast_respond as _mfr,
                            qwen_fast_respond as _qfr,
                        )
                        _llm_instance = get_role_llm_instance(simple_chat_role)
                        if _llm_instance is None:
                            _llm_instance = get_role_llm_instance('general')
                        if _llm_instance is not None:
                            _hist = _recent_exchange_pairs(
                                list(list(built.get('recent_dialogue_turns') or []) or []),
                                pair_limit=3,
                            ) if built.get('recent_dialogue_turns') else []
                            _model_path = str(getattr(_llm_instance, 'model_path', '') or '').lower()
                            _is_qwen = 'qwen' in _model_path or 'nanbeige' in _model_path
                            if _is_qwen:
                                reply = _qfr(
                                    _speech_plan,
                                    clean_message,
                                    _llm_instance,
                                    persona_system=str(built.get('persona_block') or '')[:300],
                                    history=_hist or None,
                                )
                            else:
                                reply = _mfr(
                                    _speech_plan,
                                    clean_message,
                                    _llm_instance,
                                    persona_system=str(built.get('persona_block') or '')[:300],
                                    history=_hist or None,
                                )
                    except Exception:
                        reply = ''
                if not str(reply or '').strip():
                    reply = _call_generate_chat_reply_compat(
                        prompt=prompt,
                        language=response_language,
                        persona_selected=simple_persona_selected,
                        allow_builtin_fallback=False,
                        max_tokens_override=_route_output_budget_with_learning(route, trace_learning),
                        role_override=simple_chat_role,
                    )
                used_fallback = not str(reply or '').strip()
                fallback_reason = ''
                if used_fallback:
                    fallback_reason = 'model_fallback'
                    if route.selected_route in {'lightweight_conversation', 'persona_chat_fast_path'} and simple_persona_selected:
                        reply = deterministic or fallback_chat_reply(language=response_language, persona_selected=simple_persona_selected)
                    else:
                        reply = deterministic or fallback_chat_reply(language=response_language, persona_selected=simple_persona_selected)
                return reply, used_fallback, fallback_reason, {}, None

            assistant_reply, fallback_used, fallback_reason, runtime_status, fallback_decision = observability.time_stage(
                trace,
                'response_generation',
                _generate_simple_reply,
                meta_builder=lambda result: {
                    'fallback_used': bool(result[1]),
                    'fallback_reason': str(result[2] or ''),
                    'runtime_mode': str(dict(result[3] or {}).get('mode') or 'full'),
                    'primary_role': simple_chat_role,
                    'reviewer_role': str(chat_orchestration.get('reviewer_role') or ''),
                    'orchestration_strategy': str(chat_orchestration.get('strategy') or ''),
                    **_model_budget_trace_meta(
                        _finalize_model_budget(
                            mode='chat',
                            planned_budget=planned_model_budget,
                            generation_invoked=bool(generation_state.get('invoked')),
                            skip_reason=(
                                'deterministic_route'
                                if not route.requires_llm
                                else ('no_grounding' if str(result[2] or '').strip() == 'no_grounding' else 'generation_bypassed')
                            ),
                        )
                    ),
                },
            )
            model_budget = _finalize_model_budget(
                mode='chat',
                planned_budget=planned_model_budget,
                generation_invoked=bool(generation_state.get('invoked')),
                skip_reason=(
                    'deterministic_route'
                    if not route.requires_llm
                    else ('no_grounding' if fallback_reason == 'no_grounding' else 'generation_bypassed')
                ),
            )
            # Genome fit check: P1-51 семьи проверяют ответ против 53 генов персоны
            _genome_repair_block = ''
            if str(assistant_reply or '').strip() and route.requires_persona:
                try:
                    from .genome_validator import (
                        load_or_init_persona_genome,
                        p51_gate,
                    )
                    _g_name = str(built.get('persona_name') or simple_selected_persona or '').strip()
                    if _g_name:
                        _g = load_or_init_persona_genome(_g_name)
                        _p51, _genome_repair_block = p51_gate(
                            assistant_reply, _g, language=response_language
                        )
                except Exception:
                    pass

            validation = observability.time_stage(
            trace,
            'response_validation',
            lambda: validate_response(
                route=route,
                reply=assistant_reply,
                fallback_triggered=fallback_used,
                fallback_reason_code=fallback_reason,
                history_used='session_history' in context_sources_used,
                graph_used='knowledge_graph' in context_sources_used,
                persona_used='persona_graph' in context_sources_used,
                history_available=history_available,
                model_budget=model_budget,
                persona_reference_text=built.get('persona_block') or '',
                request_text=clean_message,
            ),
            meta_builder=lambda item: {
                'ok': bool(item.ok),
                'route_match': bool(item.route_match),
                'fallback_triggered': bool(item.fallback_triggered),
                'repair_strategy': item.repair_strategy,
            },
            )
            # Если genome check провалился но validation была OK → форсируем repair
            if validation.ok and _genome_repair_block:
                from .models import ResponseValidation as _RV
                validation = _RV(
                    ok=False,
                    validation_mode=validation.validation_mode,
                    route_match=validation.route_match,
                    fallback_triggered=validation.fallback_triggered,
                    fallback_reason_code=validation.fallback_reason_code,
                    mismatch_reason='genome_character_mismatch',
                    repair_strategy='regenerate_with_budget',
                    used_persona=bool(getattr(validation, 'used_persona', False)),
                )

            if not validation.ok:
                repaired_reply = ''
                if validation.repair_strategy == 'deterministic_clarification':
                    repaired_reply = deterministic_reply(route, language=response_language, history_available=history_available)
                elif validation.repair_strategy == 'regenerate_with_budget':
                    if _cog_output is not None:
                        from .speech_planner import SpeechPlanner, verbalizer_prompt as _vp
                        _rplan = SpeechPlanner().build(_cog_output, built=built, user_text=clean_message, language=response_language)
                        _base_prompt = _vp(_rplan, persona_voice=str(built.get('persona_block') or '')[:500], recent_exchange=str(built.get('recent_dialogue') or '')[:600])
                        repaired_prompt = (_base_prompt + '\n\n' + _genome_repair_block).strip() if _genome_repair_block else _base_prompt
                    else:
                        repaired_prompt = _call_build_chat_prompt_compat(
                            question=clean_message,
                            internal_question=reasoning_message,
                            persona_block=built.get('persona_block') or '',
                            graph_context=built.get('graph_context') or '',
                            recent_dialogue=built.get('recent_dialogue') or '',
                            language=response_language,
                            semantic_focus=dict(dict(built.get('context_debug') or {}).get('semantic_focus') or {}),
                            route_guidance_block=_repair_route_guidance(route_guidance),
                            answer_perspective=_answer_perspective_for_route(route),
                            user_persona_name=str(request.user_persona_name or '').strip(),
                        )
                    reset_last_model_call_meta('chat')
                    repaired_reply = _call_generate_chat_reply_compat(
                        prompt=repaired_prompt,
                        language=response_language,
                        persona_selected=simple_persona_selected,
                        allow_builtin_fallback=False,
                        max_tokens_override=_route_output_budget_with_learning(route, trace_learning, repair=True),
                        role_override=str(chat_orchestration.get('reviewer_role') or simple_chat_role).strip() or simple_chat_role,
                    )
                elif validation.repair_strategy == 'regenerate_style_guard':
                    if _cog_output is not None:
                        from .speech_planner import SpeechPlanner, verbalizer_prompt as _vp
                        _rplan = SpeechPlanner().build(_cog_output, built=built, user_text=clean_message, language=response_language)
                        repaired_prompt = _vp(_rplan, persona_voice=str(built.get('persona_block') or '')[:500], recent_exchange=str(built.get('recent_dialogue') or '')[:600])
                    else:
                        repaired_prompt = _call_build_chat_prompt_compat(
                            question=clean_message,
                            internal_question=reasoning_message,
                            persona_block=built.get('persona_block') or '',
                            graph_context=built.get('graph_context') or '',
                            recent_dialogue=built.get('recent_dialogue') or '',
                            language=response_language,
                            semantic_focus=dict(dict(built.get('context_debug') or {}).get('semantic_focus') or {}),
                            route_guidance_block=_style_guard_route_guidance(route_guidance, route),
                            answer_perspective=_answer_perspective_for_route(route),
                            user_persona_name=str(request.user_persona_name or '').strip(),
                        )
                    reset_last_model_call_meta('chat')
                    repaired_reply = _call_generate_chat_reply_compat(
                        prompt=repaired_prompt,
                        language=response_language,
                        persona_selected=simple_persona_selected,
                        allow_builtin_fallback=False,
                        max_tokens_override=_style_guard_output_budget(route, trace_learning),
                        role_override=str(chat_orchestration.get('reviewer_role') or simple_chat_role).strip() or simple_chat_role,
                    )
                elif route.requires_llm:
                    repaired_prompt = _build_reviewer_prompt(
                        question=clean_message,
                        draft_reply=assistant_reply,
                        route=route,
                        route_guidance_block=route_guidance,
                        persona_block=built.get('persona_block') or '',
                        graph_context=built.get('graph_context') or '',
                        recent_dialogue=built.get('recent_dialogue') or '',
                        language=response_language,
                        validation_hint=str(validation.mismatch_reason or 'route mismatch').strip(),
                    )
                    repaired_reply = _call_generate_chat_reply_compat(
                        prompt=repaired_prompt,
                        language=response_language,
                        persona_selected=simple_persona_selected,
                        allow_builtin_fallback=False,
                        max_tokens_override=_style_guard_output_budget(route, trace_learning),
                        role_override=str(chat_orchestration.get('reviewer_role') or simple_chat_role).strip() or simple_chat_role,
                    )
                if str(repaired_reply or '').strip():
                    assistant_reply = repaired_reply
                    repaired_budget_override = (
                        _style_guard_output_budget(route, trace_learning)
                        if validation.repair_strategy == 'regenerate_style_guard'
                        else _route_output_budget_with_learning(route, trace_learning, repair=True)
                    )
                    repaired_budget_prompt = (
                        repaired_prompt
                        if validation.repair_strategy in {'regenerate_with_budget', 'regenerate_style_guard'}
                        else prompt
                    )
                    model_budget = _finalize_model_budget(
                        mode='chat',
                        planned_budget=_compact_model_budget(
                            mode='chat',
                            meta=plan_model_budget(
                                repaired_budget_prompt,
                                mode='chat',
                                role=simple_chat_role,
                                max_tokens_override=repaired_budget_override,
                            ),
                        ),
                        generation_invoked=True,
                    )
                    validation = validate_response(
                        route=route,
                        reply=assistant_reply,
                        fallback_triggered=fallback_used,
                        fallback_reason_code=fallback_reason,
                        history_used='session_history' in context_sources_used,
                        graph_used='knowledge_graph' in context_sources_used,
                        persona_used='persona_graph' in context_sources_used,
                        history_available=history_available,
                        model_budget=model_budget,
                        persona_reference_text=built.get('persona_block') or '',
                        request_text=clean_message,
                    )
                    validation.repaired = True
                elif validation.repair_strategy == 'regenerate_style_guard':
                    assistant_reply = _style_guard_fallback_reply(route, language=response_language)
                    validation = validate_response(
                        route=route,
                        reply=assistant_reply,
                        fallback_triggered=fallback_used,
                        fallback_reason_code=fallback_reason,
                        history_used='session_history' in context_sources_used,
                        graph_used='knowledge_graph' in context_sources_used,
                        persona_used='persona_graph' in context_sources_used,
                        history_available=history_available,
                        model_budget=model_budget,
                        persona_reference_text=built.get('persona_block') or '',
                        request_text=clean_message,
                    )
                    validation.repaired = True

            if (
                validation.ok
                and route.requires_llm
                and bool(chat_orchestration.get('reviewer_enabled'))
                and str(chat_orchestration.get('review_mode') or '').strip().lower() == 'always'
            ):
                review_prompt = _build_reviewer_prompt(
                    question=clean_message,
                    draft_reply=assistant_reply,
                    route=route,
                    route_guidance_block=route_guidance,
                    persona_block=built.get('persona_block') or '',
                    graph_context=built.get('graph_context') or '',
                    recent_dialogue=built.get('recent_dialogue') or '',
                    language=response_language,
                    validation_hint='secondary reviewer pass',
                )
                reviewed_reply = _call_generate_chat_reply_compat(
                    prompt=review_prompt,
                    language=response_language,
                    persona_selected=simple_persona_selected,
                    allow_builtin_fallback=False,
                    max_tokens_override=_route_output_budget_with_learning(route, trace_learning),
                    role_override=str(chat_orchestration.get('reviewer_role') or '').strip() or simple_chat_role,
                )
                if str(reviewed_reply or '').strip():
                    assistant_reply = reviewed_reply

            operator_messages = operator_messages_from_status(runtime_status)
            if validation.repaired:
                operator_messages = [
                    'The initial draft mismatched the selected route and was regenerated before returning.',
                    *operator_messages,
                ]
            if fallback_reason == 'no_grounding':
                operator_messages = [
                    'The selected route required grounded context, but no grounded context was available.',
                    *operator_messages,
                ]
            if trace_learning.route_guidance_lines:
                operator_messages = [
                    f"Recent similar traces contributed {len(trace_learning.route_guidance_lines)} runtime lesson(s) to this route.",
                    *operator_messages,
                ]
            operator_messages = [
                *_operator_messages_from_model_budget(model_budget),
                *operator_messages,
            ]
            if dict(built.get('message_vector_runtime') or {}):
                operator_messages = [
                    'The reply was shaped with the P1..P49 message vector layer built from the corrected context matrix.',
                    *operator_messages,
                ]
            repair_status = observability.time_stage(
                trace,
                'storage_writes',
                lambda: (
                    _write_session_history(clean_session_id, raw_message, assistant_reply, side_effects=side_effects, persona_name=str(built.get('persona_name') or simple_selected_persona or '').strip(), user_persona_name=str(request.user_persona_name or '').strip()),
                    setattr(
                        side_effects,
                        'route_state_path',
                        _save_route_memory(
                            clean_session_id,
                            route=route,
                            preprocessing=preprocessing,
                            persona_name=str(built.get('persona_name') or simple_selected_persona or '').strip(),
                            current_entity=str(built.get('current_entity') or analysis.current_entity or '').strip(),
                        ),
                    ),
                    _apply_rebuild_schedule(clean_session_id, personality_name='', side_effects=side_effects),
                )[-1],
                meta_builder=lambda payload: {
                    'history_write_path': side_effects.history_write_path,
                    'route_state_path': side_effects.route_state_path,
                    'repair_status': str(payload.get('status') or ''),
                },
            )
            pipeline = _pipeline_payload(
                envelope=envelope,
                preprocessing=preprocessing,
                route=route,
                capability_plan=capability_plan,
                context_sources_used=context_sources_used,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                validation=validation,
                model_budget=model_budget,
                logical_context=logical_context,
                trace_learning=trace_learning,
            )
            pipeline['chat_orchestration'] = dict(chat_orchestration)
            trace.request_meta['context_sources_used'] = list(context_sources_used)
            observability.finish_trace(
                trace,
                status='ok' if validation.ok else 'degraded',
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                context_tokens=int(built.get('estimated_tokens') or 0),
                persona_name=str(built.get('persona_name') or simple_selected_persona or '').strip(),
                current_entity=built.get('current_entity') or analysis.current_entity,
                response_meta={
                    'selected_route': route.selected_route,
                    'validation_result': validation.validation_mode,
                    'validation_ok': bool(validation.ok),
                    'repair_status': str(repair_status.get('status') or ''),
                    'runtime_mode': str(dict(runtime_status or {}).get('mode') or 'full'),
                    'response_language': response_language,
                    'context_sources_used': list(context_sources_used),
                    'logical_context_tokens': int(logical_context.get('assembled_context_tokens') or 0),
                    'logical_context_budget': int(logical_context.get('logical_token_budget') or 0),
                    'estimated_input_tokens': int(model_budget.get('estimated_input_tokens') or 0),
                    'configured_n_ctx': int(model_budget.get('configured_n_ctx') or 0),
                    'n_ctx': int(model_budget.get('n_ctx') or 0),
                    'reserved_output_budget': int(model_budget.get('reserved_output_budget') or 0),
                    'actual_max_tokens': int(model_budget.get('actual_max_tokens') or 0),
                    'prompt_nearly_fills_window': bool(model_budget.get('prompt_nearly_fills_window')),
                    'prompt_exceeds_window_budget': bool(model_budget.get('prompt_exceeds_window_budget')),
                    'output_truncated': bool(model_budget.get('output_truncated')),
                    'output_budget_too_small': bool(model_budget.get('output_budget_too_small')),
                    'trace_learning_batch_size': int(trace_learning.batch_size or 0),
                    'trace_learning_output_budget_boost': int(trace_learning.output_budget_boost or 0),
                    'chat_orchestration': dict(chat_orchestration),
                },
            )
            return ChatTurnResult(
                assistant_reply=assistant_reply,
                response_language=response_language,
                session_id=clean_session_id,
                trace_id=trace.request_id,
                session=parse_session(clean_session_id) or session,
                persona_name=str(built.get('persona_name') or simple_selected_persona or '').strip(),
                graph_context=built.get('graph_context') or '',
                current_entity=built.get('current_entity') or analysis.current_entity,
                analysis=analysis,
                classifications=[],
                repair_status=repair_status,
                pipeline=pipeline,
                validation=validation,
                proposal_requested=False,
                side_effects=side_effects,
                behavior_trace={
                    'route': route.to_dict(),
                    'capability_plan': capability_plan.to_dict(),
                    'context_sources_used': list(context_sources_used),
                    'fallback_strategy': {},
                    'chat_orchestration': dict(chat_orchestration),
                    'message_vector_runtime': dict(built.get('message_vector_runtime') or {}),
                },
                context_preview={
                    'estimated_tokens': int(built.get('estimated_tokens') or 0),
                    'graph_context': built.get('graph_context') or '',
                    'current_entity': built.get('current_entity') or analysis.current_entity,
                    'selected_route': route.selected_route,
                    'logical_context': dict(logical_context),
                    'model_budget': dict(model_budget),
                    'trace_learning': trace_learning.to_dict(),
                    'chat_orchestration': dict(chat_orchestration),
                    'message_vector_runtime': dict(built.get('message_vector_runtime') or {}),
                    'source_counts': dict(dict(built.get('context_debug') or {}).get('source_counts') or {}),
                    'selected_items': list(dict(built.get('context_debug') or {}).get('selected_items') or []),
                },
                runtime_status=dict(runtime_status or {}),
                operator_messages=list(dict.fromkeys(operator_messages)),
            )
        dossier_update_candidate = _looks_like_persona_dossier_update(
            reasoning_message,
            persona_name=str(analysis.selected_head or effective_selected_persona or request.selected_persona or ''),
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
        explicit_primary_name = _resolve_explicit_persona_name(
            effective_selected_persona,
            request.selected_persona,
            analysis.selected_head,
            analysis.session_persona,
            route_memory.previous_persona_name,
        )

        def _resolve_heads() -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
            prepared_heads = prepare_heads(analysis=analysis, classifications=classifications, graph_store=graph_store)
            primary = select_primary_head(analysis=analysis, prepared_heads=prepared_heads)
            primary_name = ''
            if primary and primary.get('head') is not None:
                candidate = normalize_personality_name(primary['head'].name)
                primary_name = '' if _is_system_persona_name(candidate) else candidate
            if not primary_name and explicit_primary_name:
                primary_name = explicit_primary_name
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
        pre_update_persona_bundle = load_persona(primary_name) if primary_name else None
        pre_update_mood_report = load_mood_report(persona_name=primary_name) if primary_name else None
        previous_state = observability.time_stage(
            trace,
            'current_state_sampling',
            lambda: sample_state_snapshot(
                turn_id=trace.request_id,
                session_id=clean_session_id,
                bundle=pre_update_persona_bundle,
                current_entity=analysis.current_entity or primary_name,
                active_role='',
                mood_report=pre_update_mood_report,
            ),
            meta_builder=lambda item: {
                'persona_name': item.persona_name,
                'active_role': item.active_role,
                'layers': list(item.active_layers),
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
        influence = observability.time_stage(
            trace,
            'influence_interpretation',
            lambda: interpret_user_influence(
                previous_state=previous_state,
                message=clean_message,
                analysis=analysis,
                situation=analysis.situation,
                active_role=social_role.role,
            ),
            meta_builder=lambda item: {
                'themes': list(item.themes),
                'risk_level': item.risk_level,
                'uncertainty_level': item.uncertainty_level,
            },
        )
        task_procedure, semantic_focus = observability.time_stage(
            trace,
            'task_procedure_reconstruction',
            lambda: reconstruct_task_procedure(
                message=reasoning_message,
                reply_language=response_language,
                analysis=analysis,
                situation=analysis.situation,
                previous_state=previous_state,
                influence=influence,
                persona_bundle=active_persona_bundle,
                dossier_update_statement=dossier_update_statement,
            ),
            meta_builder=lambda item: {
                'procedure_family': item[0].procedure_family,
                'response_form': item[0].response_form,
                'content_sources': list(item[0].content_sources),
            },
        )
        next_state = observability.time_stage(
            trace,
            'bounded_state_transition',
            lambda: build_bounded_next_state(
                turn_id=trace.request_id,
                session_id=clean_session_id,
                previous_state=previous_state,
                influence=influence,
                task_procedure=task_procedure,
                active_bundle=active_persona_bundle,
                current_entity=analysis.current_entity or primary_name,
                social_role=social_role,
                mood_report=active_mood_report,
            ),
            meta_builder=lambda item: {
                'active_role': item.active_role,
                'changed': list(item.changed),
                'risks': list(item.risks),
            },
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
        message_vector_runtime_payload = observability.time_stage(
            trace,
            'message_vector_runtime',
            lambda: build_runtime_message_vector_payload(
                clean_session_id,
                message_text=clean_message,
                role='user',
                persona_name=str(built.get('persona_name') or primary_name or '').strip(),
            ),
            meta_builder=lambda payload: {
                'context_window': len(list(payload.get('context_window') or [])),
                'history_flow': list(payload.get('history_flow') or []),
                'transition_type': str(dict(payload.get('transition_interpretation') or {}).get('type') or ''),
            },
        )
        built['message_vector_runtime'] = dict(message_vector_runtime_payload)
        vector_guidance = _render_message_vector_guidance(message_vector_runtime_payload)
        if vector_guidance:
            route_guidance = (route_guidance + '\n\n' + vector_guidance).strip()

        # ── DIALOG CONTEXT MATRIX ─────────────────────────────────────────
        try:
            from .dialog_tracker import record_turn as _record_turn
            _dialog_ctx = _record_turn(
                clean_session_id,
                text=clean_message,
                speaker='user',
            )
            built['dialog_context'] = _dialog_ctx
        except Exception:
            built['dialog_context'] = {}

        working_context = observability.time_stage(
            trace,
            'working_context_construction',
            lambda: build_working_context_layer(
                turn_id=trace.request_id,
                session_id=clean_session_id,
                current_entity=built.get('current_entity') or primary_name,
                persona_name=built.get('persona_name') or primary_name,
                updated_state=next_state,
                influence=influence,
                task_procedure=task_procedure,
                built_context=built,
            ),
            meta_builder=lambda item: {
                'estimated_tokens': item.estimated_tokens,
                'important_items': len(list(item.important_items)),
                'source_counts': dict(item.source_counts),
            },
        )
        reviewed_context = observability.time_stage(
            trace,
            'context_review',
            lambda: review_working_context(
                layer=working_context,
                updated_state=next_state,
                task_procedure=task_procedure,
            ),
            meta_builder=lambda item: {
                'important_items': len(list(item.important_items)),
                'weak_items': len(list(item.weak_items)),
                'contradictions': len(list(item.contradictions)),
            },
        )
        response_plan = observability.time_stage(
            trace,
            'response_shaping',
            lambda: shape_response_plan(
                reviewed_context=reviewed_context,
                influence=influence,
                task_procedure=task_procedure,
                social_role=social_role,
                response_explanation=response_explanation,
            ),
            meta_builder=lambda item: {
                'role': item.role,
                'style': item.style,
                'behavior_mode': item.behavior_mode,
            },
        )
        current_context_json_path, _current_context_txt_path = observability.time_stage(
            trace,
            'current_context_persistence',
            lambda: persist_current_context(
                reviewed_context=reviewed_context,
                response_plan=response_plan,
                task_procedure=task_procedure,
                previous_state=previous_state,
                influence=influence,
                next_state=next_state,
            ),
            meta_builder=lambda item: {
                'current_context_json': str(item[0] or ''),
            },
        )
        side_effects.current_context_path = str(current_context_json_path or '')
        if _cog_output is not None:
            # ── SPEECH PLANNER PATH (mandatory for persona route) ────────────
            # Pipeline controls content, LLM only verbalizes.
            # social_role, mood, reviewed_context inform the SpeechPlan style.
            from .speech_planner import SpeechPlanner, verbalizer_prompt as _vp
            _persona_built = dict(built)
            # Enrich built with mood/role signals for style hints
            if active_mood_report is not None and active_mood_report.snapshot_count > 0:
                _persona_built['mood_cluster'] = active_mood_report.latest_cluster_label
            if social_role:
                _persona_built['social_role'] = str(social_role)
            _speech_plan = SpeechPlanner().build(
                _cog_output,
                built=_persona_built,
                user_text=clean_message,
                language=response_language,
            )
            prompt = _vp(
                _speech_plan,
                persona_voice=str(built.get('persona_block') or '')[:500],
                recent_exchange=str(built.get('recent_dialogue') or '')[:600],
            )
        else:
            # ── LEGACY FULL PIPELINE PATH (no cognitive output) ──────────────
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
                state_transition_block=render_state_transition_block(
                    previous_state=previous_state,
                    influence=influence,
                    next_state=next_state,
                ),
                procedure_block=render_task_procedure_block(task_procedure),
                reviewed_context_block=render_reviewed_context_block(reviewed_context),
                response_shaping_block=render_response_shaping_block(response_plan),
                language=response_language,
                semantic_focus=dict(semantic_focus or dict(dict(built.get('context_debug') or {}).get('semantic_focus') or {})),
                route_guidance_block=route_guidance,
                answer_perspective='persona',
                user_persona_name=str(request.user_persona_name or '').strip(),
                context_matrix=list(dict(built.get('message_vector_runtime') or {}).get('context_matrix') or []),
                current_vector=dict(dict(built.get('message_vector_runtime') or {}).get('current_vector') or {}),
            )
        logical_context = _logical_context_snapshot(built)
        heavy_chat_role = str(chat_orchestration.get('primary_role') or _chat_role_for_request(persona_selected=bool(primary_name))).strip() or _chat_role_for_request(persona_selected=bool(primary_name))
        planned_model_budget = _compact_model_budget(
            mode='chat',
            meta=plan_model_budget(
                prompt,
                mode='chat',
                role=heavy_chat_role,
                max_tokens_override=_route_output_budget_with_learning(route, trace_learning),
            ),
        )
        reset_last_model_call_meta('chat')
        generation_state = {'invoked': False}

        def _behavioral_fallback(fallback_reason: str, runtime_status: dict[str, Any] | None) -> tuple[str, BehavioralFallbackDecision]:
            route_decision = _route_aware_fallback_decision(
                route,
                language=response_language,
                history_available=history_available,
            )
            if route_decision is not None:
                return deterministic_reply(route, language=response_language, history_available=history_available), route_decision
            if (
                route.selected_route == 'persona_graph_reasoning'
                and analysis.user_state.intent == 'statement'
                and not bool(analysis.user_state.signals.get('contains_question'))
                and str(analysis.situation.target or '').strip() == 'persona'
                and str(analysis.situation.type or '').strip() == 'neutral_statement'
                and not bool(analysis.user_state.signals.get('contains_help_request'))
                and not bool(analysis.user_state.signals.get('contains_distress'))
                and not bool(analysis.user_state.signals.get('contains_anger'))
                and not bool(analysis.user_state.signals.get('contains_moral_violation'))
            ):
                return (
                    _persona_dossier_acknowledgement(
                        response_language,
                        persona_bundle=active_persona_bundle,
                        social_role=social_role,
                        analysis=analysis,
                    ),
                    BehavioralFallbackDecision(
                        strategy='persona_record_acknowledgement',
                        confidence=0.84,
                        uncertainty_level='moderate',
                        risk_level='low',
                        style_mode='grounded',
                        reason='The route stayed inside persona memory handling and acknowledged new world or biography facts instead of narrowing into an unrelated limitation.',
                        evidence=[
                            'selected route: persona_graph_reasoning',
                            'user turn is a factual statement that adds persona/world context',
                            'fallback preserved session continuity by acknowledging the update',
                        ],
                        source_grounding='route_policy',
                    ),
                )
            decision = choose_behavioral_fallback(
                bundle=active_persona_bundle,
                analysis=analysis,
                situation=analysis.situation,
                social_role=social_role,
                mood_report=active_mood_report,
                question=clean_message,
                grounded=grounded,
                fallback_reason=fallback_reason,
                runtime_status=runtime_status,
            )
            reply = render_behavioral_fallback(
                decision,
                bundle=active_persona_bundle,
                question=clean_message,
            )
            if response_language != 'en':
                translated_reply = translate_text(
                    reply,
                    target_language=response_language,
                    source_language='en',
                    role=get_runtime_config().roles.translation,
                )
                if translated_reply:
                    reply = translated_reply
            return reply, decision

        grounded = any(str(part or '').strip() for part in (built.get('persona_block'), built.get('graph_context'), built.get('recent_dialogue')))

        def _generate_reply() -> tuple[str, bool, str, dict[str, Any], BehavioralFallbackDecision | None]:
            if dossier_update_statement:
                return (
                    _persona_dossier_acknowledgement(
                        response_language,
                        persona_bundle=active_persona_bundle,
                        social_role=social_role,
                        analysis=analysis,
                    ),
                    False,
                    '',
                    {},
                    None,
                )
            if route.strict_grounding and not grounded and (_cog_mode != 'pure_llm' or route.requires_graph):
                fallback_text, decision = _behavioral_fallback('no_grounding', {})
                return fallback_text, True, 'no_grounding', {}, decision
            status = runtime_status_snapshot()
            if str(dict(status or {}).get('mode') or '') == 'degraded':
                fallback_reason = 'dependency_unavailable'
                reply, fallback_decision = _behavioral_fallback(fallback_reason, status)
                return reply, True, fallback_reason, status, fallback_decision
            generation_state['invoked'] = True
            reply = _call_generate_chat_reply_compat(
                prompt=prompt,
                language=response_language,
                persona_selected=bool(primary_name),
                allow_builtin_fallback=False,
                max_tokens_override=_route_output_budget_with_learning(route, trace_learning),
                role_override=heavy_chat_role,
            )
            used_fallback = not str(reply or '').strip()
            fallback_reason = ''
            fallback_decision: BehavioralFallbackDecision | None = None
            if used_fallback:
                fallback_reason = 'model_fallback'
                reply, fallback_decision = _behavioral_fallback(fallback_reason, {})
            return reply, used_fallback, fallback_reason, {}, fallback_decision

        assistant_reply, fallback_used, fallback_reason, runtime_status, fallback_decision = observability.time_stage(
            trace,
            'llm_call',
            _generate_reply,
            meta_builder=lambda result: {
                'fallback_used': bool(result[1]),
                'fallback_reason': str(result[2] or ''),
                'runtime_mode': str(dict(result[3] or {}).get('mode') or 'full'),
                'fallback_strategy': str((result[4].strategy if result[4] is not None else '') or ''),
                'primary_role': heavy_chat_role,
                'reviewer_role': str(chat_orchestration.get('reviewer_role') or ''),
                'orchestration_strategy': str(chat_orchestration.get('strategy') or ''),
                **_model_budget_trace_meta(
                    _finalize_model_budget(
                        mode='chat',
                        planned_budget=planned_model_budget,
                        generation_invoked=bool(generation_state.get('invoked')),
                        skip_reason=('no_grounding' if str(result[2] or '').strip() == 'no_grounding' else 'generation_bypassed'),
                    )
                ),
            },
        )
        model_budget = _finalize_model_budget(
            mode='chat',
            planned_budget=planned_model_budget,
            generation_invoked=bool(generation_state.get('invoked')),
            skip_reason=('no_grounding' if fallback_reason == 'no_grounding' else 'generation_bypassed'),
        )
        context_sources_used = [
            source
            for source, enabled in (
                ('session_history', bool(str(built.get('recent_dialogue') or '').strip())),
                ('knowledge_graph', bool(str(built.get('graph_context') or '').strip())),
                ('persona_graph', bool(str(built.get('persona_block') or '').strip())),
                ('message_vector_runtime', bool(dict(built.get('message_vector_runtime') or {}))),
            )
            if enabled
        ]
        validation = observability.time_stage(
            trace,
            'response_validation',
            lambda: validate_response(
                route=route,
                reply=assistant_reply,
                fallback_triggered=fallback_used,
                fallback_reason_code=fallback_reason,
                history_used='session_history' in context_sources_used,
                graph_used='knowledge_graph' in context_sources_used,
                persona_used='persona_graph' in context_sources_used,
                history_available=history_available,
                model_budget=model_budget,
                persona_reference_text=built.get('persona_block') or '',
                request_text=clean_message,
            ),
            meta_builder=lambda item: {
                'ok': bool(item.ok),
                'route_match': bool(item.route_match),
                'fallback_triggered': bool(item.fallback_triggered),
                'repair_strategy': item.repair_strategy,
            },
        )
        if not validation.ok:
            style_guard_repair = validation.repair_strategy == 'regenerate_style_guard'
            repaired_prompt = _call_build_chat_prompt_compat(
                question=clean_message,
                internal_question=reasoning_message,
                persona_block=built.get('persona_block') or '',
                social_role_block='' if style_guard_repair else render_social_role_block(social_role, mood_report=active_mood_report),
                mood_research_block=(
                    f"Latest mood cluster: {active_mood_report.latest_cluster_label}. "
                    f"Observed role trend: {', '.join(str(item.get('role') or '') for item in list(active_mood_report.role_effects or [])[:3] if str(item.get('role') or '').strip())}."
                    if active_mood_report is not None and active_mood_report.snapshot_count > 0
                    else ''
                ),
                graph_context=built.get('graph_context') or '',
                recent_dialogue=built.get('recent_dialogue') or '',
                state_transition_block=render_state_transition_block(
                    previous_state=previous_state,
                    influence=influence,
                    next_state=next_state,
                ),
                procedure_block='' if style_guard_repair else render_task_procedure_block(task_procedure),
                reviewed_context_block=render_reviewed_context_block(reviewed_context),
                response_shaping_block='' if style_guard_repair else render_response_shaping_block(response_plan),
                language=response_language,
                semantic_focus=dict(semantic_focus or dict(dict(built.get('context_debug') or {}).get('semantic_focus') or {})),
                route_guidance_block=(
                    _repair_route_guidance(route_guidance)
                    if validation.repair_strategy == 'regenerate_with_budget'
                    else (
                        _style_guard_route_guidance(route_guidance, route)
                        if validation.repair_strategy == 'regenerate_style_guard'
                        else '\n'.join(
                            [
                                route_guidance,
                                'Validation repair: the previous draft mismatched the selected route.',
                                'Follow the selected route strictly and repair the route mismatch without changing the task.',
                            ]
                        ).strip()
                    )
                ),
                answer_perspective='persona',
                user_persona_name=str(request.user_persona_name or '').strip(),
            )
            reset_last_model_call_meta('chat')
            repaired_reply = _call_generate_chat_reply_compat(
                prompt=repaired_prompt,
                language=response_language,
                persona_selected=bool(primary_name),
                allow_builtin_fallback=False,
                max_tokens_override=(
                    _style_guard_output_budget(route, trace_learning)
                    if validation.repair_strategy == 'regenerate_style_guard'
                    else _route_output_budget_with_learning(route, trace_learning, repair=True)
                ),
                role_override=(
                    str(chat_orchestration.get('reviewer_role') or heavy_chat_role).strip()
                    if validation.repair_strategy in {'regenerate_with_budget', 'regenerate_style_guard'}
                    else heavy_chat_role
                ),
            )
            if str(repaired_reply or '').strip():
                assistant_reply = repaired_reply
                model_budget = _finalize_model_budget(
                    mode='chat',
                    planned_budget=_compact_model_budget(
                        mode='chat',
                        meta=plan_model_budget(
                            repaired_prompt,
                            mode='chat',
                            role=heavy_chat_role,
                            max_tokens_override=(
                                _style_guard_output_budget(route, trace_learning)
                                if validation.repair_strategy == 'regenerate_style_guard'
                                else _route_output_budget_with_learning(route, trace_learning, repair=True)
                            ),
                        ),
                    ),
                    generation_invoked=True,
                )
                validation = validate_response(
                    route=route,
                    reply=assistant_reply,
                    fallback_triggered=fallback_used,
                    fallback_reason_code=fallback_reason,
                    history_used='session_history' in context_sources_used,
                    graph_used='knowledge_graph' in context_sources_used,
                    persona_used='persona_graph' in context_sources_used,
                    history_available=history_available,
                    model_budget=model_budget,
                    persona_reference_text=built.get('persona_block') or '',
                    request_text=clean_message,
                )
                validation.repaired = True
            elif validation.repair_strategy == 'regenerate_style_guard':
                assistant_reply = _style_guard_fallback_reply(route, language=response_language)
                validation = validate_response(
                    route=route,
                    reply=assistant_reply,
                    fallback_triggered=fallback_used,
                    fallback_reason_code=fallback_reason,
                    history_used='session_history' in context_sources_used,
                    graph_used='knowledge_graph' in context_sources_used,
                    persona_used='persona_graph' in context_sources_used,
                    history_available=history_available,
                    model_budget=model_budget,
                    persona_reference_text=built.get('persona_block') or '',
                    request_text=clean_message,
                )
                validation.repaired = True

        if (
            validation.ok
            and route.requires_llm
            and bool(chat_orchestration.get('reviewer_enabled'))
            and str(chat_orchestration.get('review_mode') or '').strip().lower() == 'always'
        ):
            review_prompt = _build_reviewer_prompt(
                question=clean_message,
                draft_reply=assistant_reply,
                route=route,
                route_guidance_block=route_guidance,
                persona_block=built.get('persona_block') or '',
                graph_context=built.get('graph_context') or '',
                recent_dialogue=built.get('recent_dialogue') or '',
                language=response_language,
                validation_hint='secondary reviewer pass',
            )
            reviewed_reply = _call_generate_chat_reply_compat(
                prompt=review_prompt,
                language=response_language,
                persona_selected=bool(primary_name),
                allow_builtin_fallback=False,
                max_tokens_override=_route_output_budget_with_learning(route, trace_learning),
                role_override=str(chat_orchestration.get('reviewer_role') or '').strip() or heavy_chat_role,
            )
            if str(reviewed_reply or '').strip():
                assistant_reply = reviewed_reply
        operator_messages = operator_messages_from_status(runtime_status)
        if validation.repaired:
            operator_messages = [
                'The initial draft mismatched the selected route and was regenerated before returning.',
                *operator_messages,
            ]
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
        operator_messages = [
            *_operator_messages_from_model_budget(model_budget),
            *operator_messages,
        ]
        if trace_learning.route_guidance_lines:
            operator_messages = [
                f"Recent similar traces contributed {len(trace_learning.route_guidance_lines)} runtime lesson(s) to this route.",
                *operator_messages,
            ]
        if fallback_decision is not None:
            operator_messages = [
                f"Under uncertainty the system selected behavioral fallback '{fallback_decision.strategy}' "
                f"with {fallback_decision.risk_level} risk and {fallback_decision.uncertainty_level} uncertainty.",
                *operator_messages,
            ]
        repair_status = observability.time_stage(
            trace,
            'storage_writes',
            lambda: (
                _write_session_history(clean_session_id, raw_message, assistant_reply, side_effects=side_effects, persona_name=primary_name, user_persona_name=str(request.user_persona_name or '').strip()),
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
                setattr(
                    side_effects,
                    'transition_log_path',
                    append_transition_log(
                        turn_id=trace.request_id,
                        session_id=clean_session_id,
                        user_message=clean_message,
                        previous_state=previous_state,
                        influence=influence,
                        task_procedure=task_procedure,
                        next_state=next_state,
                        working_context=working_context,
                        reviewed_context=reviewed_context,
                        response_plan=response_plan,
                        assistant_reply=assistant_reply,
                    ),
                ),
                setattr(
                    side_effects,
                    'route_state_path',
                    _save_route_memory(
                        clean_session_id,
                        route=route,
                        preprocessing=preprocessing,
                        persona_name=str(
                            (active_persona_bundle.name if active_persona_bundle is not None else '')
                            or built.get('persona_name')
                            or primary_name
                            or ''
                        ).strip(),
                        current_entity=str(built.get('current_entity') or primary_name or '').strip(),
                    ),
                ),
                schedule_mood_research_refresh(persona_name=primary_name, session_id=clean_session_id),
                _apply_rebuild_schedule(clean_session_id, personality_name=primary_name, side_effects=side_effects),
            )[-1],
            meta_builder=lambda payload: {
                'history_write_path': side_effects.history_write_path,
                'route_state_path': side_effects.route_state_path,
                'current_context_path': side_effects.current_context_path,
                'transition_log_path': side_effects.transition_log_path,
                'persona_updates': list(side_effects.persona_updates) + ['mood_research'],
                'repair_status': str(payload.get('status') or ''),
            },
        )
        side_effects.add_persona_update('mood_research')
        pipeline = _pipeline_payload(
            envelope=envelope,
            preprocessing=preprocessing,
            route=route,
            capability_plan=capability_plan,
            context_sources_used=context_sources_used,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            validation=validation,
            model_budget=model_budget,
            logical_context=logical_context,
            trace_learning=trace_learning,
        )
        pipeline['chat_orchestration'] = dict(chat_orchestration)
        trace.request_meta['context_sources_used'] = list(context_sources_used)
        observability.finish_trace(
            trace,
            status='ok' if validation.ok else 'degraded',
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            context_tokens=int(built.get('estimated_tokens') or 0),
            persona_name=built.get('persona_name') or primary_name,
            current_entity=built.get('current_entity') or primary_name,
            response_meta={
                'selected_route': route.selected_route,
                'validation_result': validation.validation_mode,
                'validation_ok': bool(validation.ok),
                'classification_count': len(classifications),
                'repair_status': str(repair_status.get('status') or ''),
                'runtime_mode': str(dict(runtime_status or {}).get('mode') or 'full'),
                'response_language': response_language,
                'social_role': social_role.role,
                'context_sources_used': list(context_sources_used),
                'logical_context_tokens': int(logical_context.get('assembled_context_tokens') or 0),
                'logical_context_budget': int(logical_context.get('logical_token_budget') or 0),
                'estimated_input_tokens': int(model_budget.get('estimated_input_tokens') or 0),
                'configured_n_ctx': int(model_budget.get('configured_n_ctx') or 0),
                'n_ctx': int(model_budget.get('n_ctx') or 0),
                'reserved_output_budget': int(model_budget.get('reserved_output_budget') or 0),
                'actual_max_tokens': int(model_budget.get('actual_max_tokens') or 0),
                'prompt_nearly_fills_window': bool(model_budget.get('prompt_nearly_fills_window')),
                'prompt_exceeds_window_budget': bool(model_budget.get('prompt_exceeds_window_budget')),
                'output_truncated': bool(model_budget.get('output_truncated')),
                'output_budget_too_small': bool(model_budget.get('output_budget_too_small')),
                'trace_learning_batch_size': int(trace_learning.batch_size or 0),
                'trace_learning_output_budget_boost': int(trace_learning.output_budget_boost or 0),
                'chat_orchestration': dict(chat_orchestration),
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
        if dict(built.get('message_vector_runtime') or {}):
            operator_messages = [
                'The reply was shaped with the P1..P49 message vector layer built from the corrected context matrix.',
                *operator_messages,
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
            pipeline=pipeline,
            validation=validation,
            proposal_requested=False,
            side_effects=side_effects,
            persona_selection=selection_explanation,
            persona_response=response_explanation,
            social_role=social_role,
            fallback_strategy=fallback_decision,
            mood_research={
                'active_report_scope': active_mood_report.scope if active_mood_report is not None else '',
                'latest_cluster': active_mood_report.latest_cluster_label if active_mood_report is not None else '',
                'snapshot_count': active_mood_report.snapshot_count if active_mood_report is not None else 0,
            },
            state_transition={
                'interaction_frame': interaction_frame.to_dict(),
                'previous_state': previous_state.to_dict(),
                'influence': influence.to_dict(),
                'task_procedure': task_procedure.to_dict(),
                'updated_state': next_state.to_dict(),
                'working_context': working_context.to_dict(),
                'reviewed_context': reviewed_context.to_dict(),
                'response_plan': response_plan.to_dict(),
            },
            behavior_trace={
                'route': route.to_dict(),
                'capability_plan': capability_plan.to_dict(),
                'interaction_frame': interaction_frame.to_dict(),
                'semantic_focus': dict(semantic_focus or dict(dict(built.get('context_debug') or {}).get('semantic_focus') or {})),
                'social_role': social_role.to_dict(),
                'mood_cluster': active_mood_report.latest_cluster_label if active_mood_report is not None else '',
                'previous_state': previous_state.to_dict(),
                'influence': influence.to_dict(),
                'task_procedure': task_procedure.to_dict(),
                'updated_state': next_state.to_dict(),
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
                'response_plan': response_plan.to_dict(),
                'logical_context': dict(logical_context),
                'model_budget': dict(model_budget),
                'trace_learning': trace_learning.to_dict(),
                'chat_orchestration': dict(chat_orchestration),
                'fallback_strategy': fallback_decision.to_dict() if fallback_decision is not None else {},
                'message_vector_runtime': dict(built.get('message_vector_runtime') or {}),
            },
            context_preview={
                'estimated_tokens': int(built.get('estimated_tokens') or 0),
                'graph_context': built.get('graph_context') or '',
                'current_entity': built.get('current_entity') or primary_name,
                'persona_name': built.get('persona_name') or primary_name,
                'selected_route': route.selected_route,
                'logical_context': dict(logical_context),
                'model_budget': dict(model_budget),
                'trace_learning': trace_learning.to_dict(),
                'interaction_frame': interaction_frame.to_dict(),
                'social_role': social_role.to_dict(),
                'task_procedure': task_procedure.to_dict(),
                'working_context': working_context.to_dict(),
                'reviewed_context': reviewed_context.to_dict(),
                'response_plan': response_plan.to_dict(),
                'chat_orchestration': dict(chat_orchestration),
                'fallback_strategy': fallback_decision.to_dict() if fallback_decision is not None else {},
                'message_vector_runtime': dict(built.get('message_vector_runtime') or {}),
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
    user_persona_name: str = '',
) -> dict[str, Any]:
    result = run_chat_turn(
        ChatTurnRequest(
            message=message,
            session_id=session_id,
            selected_persona=selected_persona,
            explicit_context=explicit_context,
            language=language,
            user_persona_name=user_persona_name,
        )
    )
    return result.to_dict(include_side_effects=get_runtime_config().features.include_side_effects_in_response)
