from __future__ import annotations

from .interaction_routing import route_interaction
from .message_analyzer import analyze_message
from .models import ControllerState, RouteMemory
from .request_pipeline import build_request_envelope, plan_capabilities, preprocess_request, select_route


def build_controller_state(
    *,
    request_id: str,
    session_id: str,
    message: str,
    selected_persona: str = '',
    explicit_context: str = '',
    current_entity: str = '',
    session_persona: str = '',
    route_memory: RouteMemory | None = None,
    known_entities: list[dict[str, object]] | None = None,
) -> ControllerState:
    envelope = build_request_envelope(
        request_id=request_id,
        session_id=session_id,
        raw_text=message,
    )
    interaction_frame = route_interaction(
        message=message,
        selected_persona_hint=str(selected_persona or '').strip(),
        session_persona=str(session_persona or '').strip(),
        current_entity=str(current_entity or '').strip(),
        known_entities=list(known_entities or []),
    )
    effective_selected_persona = str(
        selected_persona
        or interaction_frame.requested_persona
        or (session_persona if interaction_frame.keep_session_persona else '')
        or ''
    ).strip()
    analysis_message = str(interaction_frame.routed_message or message).strip()
    analysis = analyze_message(
        message=analysis_message,
        session_id=session_id,
        selected_head=effective_selected_persona,
        current_entity=str(interaction_frame.topic_entity or current_entity or '').strip(),
        session_persona=str(session_persona or '').strip(),
        explicit_context=explicit_context,
        interaction_frame=interaction_frame,
        known_entities=list(known_entities or []),
    )
    preprocessing = preprocess_request(
        envelope=envelope,
        analysis=analysis,
        interaction_frame=interaction_frame,
        explicit_context=explicit_context,
        route_memory=route_memory,
    )
    route = select_route(
        envelope=envelope,
        preprocessing=preprocessing,
        analysis=analysis,
        interaction_frame=interaction_frame,
        selected_persona=effective_selected_persona,
        route_memory=route_memory,
    )
    capability_plan = plan_capabilities(route)
    return ControllerState(
        envelope=envelope,
        interaction_frame=interaction_frame,
        analysis=analysis,
        preprocessing=preprocessing,
        route=route,
        capability_plan=capability_plan,
        selected_persona=effective_selected_persona,
        session_persona=str(session_persona or '').strip(),
        current_entity=str(current_entity or '').strip(),
    )
