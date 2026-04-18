from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from threading import Lock
from typing import Any
from uuid import uuid4

from .duplicate_resolver import normalize_name
from .llm import call_json_model_for_role
from .history_store import load_session_route_state
from .models import (
    HeadBundle,
    InfluenceInterpretation,
    MessageAnalysis,
    MoodResearchReport,
    PersonaResponseExplanation,
    ResponseShapingPlan,
    Situation,
    SocialRoleDecision,
    StateSnapshot,
    TaskProcedurePlan,
    WorkingContextLayer,
)
from .prompt_builder import (
    build_context_curator_prompt,
    build_context_reviewer_prompt,
    build_influence_interpreter_prompt,
    build_procedure_reconstructor_prompt,
    build_response_shaper_prompt,
    build_state_reader_prompt,
    build_state_transition_prompt,
)
from .runtime_config import get_runtime_config
from .task_procedures import seed_task_procedure

_TRANSITION_LOG_LOCK = Lock()
_CURRENT_CONTEXT_LOCK = Lock()


def _enabled_model_stages() -> set[str]:
    raw = str(os.getenv('COGNITIVE_STAGE_MODEL_STEPS', 'response_shaper') or '').strip().lower()
    if not raw:
        return set()
    return {item.strip() for item in raw.split(',') if item.strip()}


def _stage_model_enabled(stage_name: str) -> bool:
    return str(stage_name or '').strip().lower() in _enabled_model_stages()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def infer_session_active_persona(session_id: str) -> str:
    clean_session_id = str(session_id or '').strip()
    if not clean_session_id:
        return ''
    route_state = load_session_route_state(clean_session_id)
    route_persona = str(route_state.get('persona_name') or '').strip()
    if route_persona:
        return route_persona
    path = get_runtime_config().paths.state_transitions_log_path
    if not path.exists():
        return ''
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return ''
    for line in reversed(lines[-256:]):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if str(payload.get('session_id') or '').strip() != clean_session_id:
            continue
        for section_name in ('new_state', 'reviewed_context', 'previous_state'):
            section = payload.get(section_name)
            if not isinstance(section, dict):
                continue
            persona_name = str(section.get('persona_name') or '').strip()
            if persona_name:
                return persona_name
    return ''


def _normalize_items(values: list[Any], *, limit: int = 8) -> list[str]:
    return list(dict.fromkeys(str(item).strip() for item in list(values or []) if str(item).strip()))[:limit]


def _clip_text(text: str, *, limit: int = 1200) -> str:
    raw = str(text or '').strip()
    if not raw:
        return ''
    return raw[:limit].strip()


def _emotion_posture(bundle: HeadBundle | None) -> list[str]:
    if bundle is None:
        return []
    vector = dict(bundle.emotion_vector or {})
    posture: list[str] = []
    if float(vector.get('confidence') or 0.0) >= 0.6:
        posture.append('steady')
    if float(vector.get('curiosity') or 0.0) >= 0.6:
        posture.append('curious')
    if float(vector.get('empathy') or 0.0) >= 0.5:
        posture.append('attentive')
    if float(vector.get('anger') or 0.0) >= 0.35:
        posture.append('firm')
    if float(vector.get('fear') or 0.0) >= 0.35:
        posture.append('cautious')
    return posture[:4]


def _bundle_graph_anchors(bundle: HeadBundle | None) -> list[str]:
    if bundle is None:
        return []
    relation_targets = [str(item.get('target') or '').strip() for item in list(bundle.relations or []) if str(item.get('target') or '').strip()]
    affinities = [str(item).strip() for item in list((bundle.persona_form or {}).get('topic_affinities') or []) if str(item).strip()]
    return _normalize_items(relation_targets + affinities, limit=8)


def _bundle_memory_anchors(bundle: HeadBundle | None) -> list[str]:
    if bundle is None:
        return []
    persona_form = dict(bundle.persona_form or {})
    anchors = list(persona_form.get('memory_anchors') or []) + list(persona_form.get('memories') or []) + list(persona_form.get('personal_history') or [])
    return _normalize_items(anchors, limit=8)


def _bundle_priorities(bundle: HeadBundle | None) -> list[str]:
    if bundle is None:
        return ['answer_substance']
    persona_form = dict(bundle.persona_form or {})
    priorities = list(persona_form.get('response_priorities') or [])
    if not priorities:
        priorities = ['answer_substance', 'clarify_if_underspecified', 'stay_in_character']
    return _normalize_items(priorities, limit=6)


def _bundle_constraints(bundle: HeadBundle | None) -> list[str]:
    if bundle is None:
        return ['stay_in_first_person', 'avoid_assistant_tone']
    persona_form = dict(bundle.persona_form or {})
    constraints = list(persona_form.get('risk_controls') or [])
    constraints.extend(['stay_in_first_person', 'avoid_assistant_tone'])
    return _normalize_items(constraints, limit=8)


def _risk_signals_from_analysis(analysis: MessageAnalysis, situation: Situation) -> list[str]:
    signals = dict(analysis.user_state.signals or {})
    risks: list[str] = []
    if float(signals.get('contains_distress') or 0.0) >= 0.35 or situation.type == 'user_distress':
        risks.append('user_distress')
    if float(signals.get('contains_help_request') or 0.0) >= 0.4:
        risks.append('help_request')
    if float(signals.get('contains_insult') or 0.0) >= 0.3 or situation.type == 'insult':
        risks.append('antagonism')
    if situation.type == 'abnormal_behavior':
        risks.append('behavioral_abnormality')
    if float(situation.severity or 0.0) >= 0.45:
        risks.append('elevated_severity')
    return _normalize_items(risks, limit=6)


def _default_state_summary(*, bundle: HeadBundle | None, current_entity: str, active_role: str, mood_cluster: str) -> str:
    if bundle is None:
        return _clip_text(f'No active persona bundle is loaded. Current entity={current_entity or "unknown"}.')
    posture = ', '.join(_emotion_posture(bundle)) or 'composed'
    role_part = active_role or 'ally'
    mood_part = mood_cluster or 'no explicit mood cluster'
    return _clip_text(
        f"{bundle.name} is the active persona. Current role pressure is {role_part}. "
        f"Dynamic posture is {posture}. Mood pattern: {mood_part}."
    )


def sample_state_snapshot(
    *,
    turn_id: str,
    session_id: str,
    bundle: HeadBundle | None,
    current_entity: str,
    active_role: str = '',
    mood_report: MoodResearchReport | None = None,
) -> StateSnapshot:
    mood_cluster = str(mood_report.latest_cluster_label or '').strip() if mood_report is not None else ''
    persona_name = str(bundle.name if bundle is not None else '').strip()
    inferred_role = ''
    if bundle is not None:
        available_roles = _normalize_items(list((bundle.persona_form or {}).get('social_roles') or []), limit=1)
        if available_roles:
            inferred_role = available_roles[0]
    base = StateSnapshot(
        turn_id=turn_id,
        session_id=session_id,
        persona_name=persona_name,
        current_entity=str(current_entity or persona_name or '').strip(),
        active_role=active_role or inferred_role,
        mood_cluster=mood_cluster,
        summary='',
        active_layers=_normalize_items(
            [
                'persona_baseline' if bundle and bundle.baseline_definition is not None else '',
                'persona_dynamic_state' if bundle and bundle.dynamic_state is not None else '',
                'persona_learned_patterns' if bundle and bundle.learned_patterns is not None else '',
                'mood_research' if mood_cluster else '',
                'graph_context',
            ],
            limit=8,
        ),
        graph_anchors=_bundle_graph_anchors(bundle),
        memory_anchors=_bundle_memory_anchors(bundle),
        priorities=_bundle_priorities(bundle),
        risks=[],
        constraints=_bundle_constraints(bundle),
        changed=[],
        unchanged=[],
        source='deterministic',
    )
    base.summary = _default_state_summary(bundle=bundle, current_entity=base.current_entity, active_role=base.active_role, mood_cluster=base.mood_cluster)
    payload = None
    if _stage_model_enabled('state_reader'):
        prompt = build_state_reader_prompt(state_payload=base.to_dict())
        payload = call_json_model_for_role(prompt, role=get_runtime_config().roles.extraction)
    if isinstance(payload, dict):
        summary = _clip_text(str(payload.get('summary') or ''), limit=800)
        if summary:
            base.summary = summary
            base.source = 'llm_guided'
        base.active_layers = _normalize_items(list(payload.get('active_layers') or base.active_layers), limit=8)
        base.graph_anchors = _normalize_items(list(payload.get('graph_anchors') or base.graph_anchors), limit=8)
        base.memory_anchors = _normalize_items(list(payload.get('memory_anchors') or base.memory_anchors), limit=8)
        base.priorities = _normalize_items(list(payload.get('priorities') or base.priorities), limit=8)
        base.risks = _normalize_items(list(payload.get('risks') or base.risks), limit=8)
        base.constraints = _normalize_items(list(payload.get('constraints') or base.constraints), limit=8)
    return base


def interpret_user_influence(
    *,
    previous_state: StateSnapshot,
    message: str,
    analysis: MessageAnalysis,
    situation: Situation,
    active_role: str,
) -> InfluenceInterpretation:
    normalized_message = normalize_name(message)
    activation = list(dict.fromkeys(str(item).strip() for item in list(analysis.entities or []) if getattr(item, 'name', '').strip()))
    themes: list[str] = []
    if situation.type == 'neutral_query':
        themes.append('query')
    if situation.type == 'neutral_statement':
        themes.append('statement')
    if situation.type == 'user_distress':
        themes.append('distress')
    if 'work' in normalized_message or 'routine' in normalized_message:
        themes.append('work')
    if 'who' in normalized_message or 'introduce' in normalized_message or 'about yourself' in normalized_message:
        themes.append('identity')
    if 'remember' in normalized_message or 'before' in normalized_message:
        themes.append('memory')
    pressure_points = _risk_signals_from_analysis(analysis, situation)
    if not pressure_points and '?' in str(message or ''):
        pressure_points.append('needs_answer')
    conflicts: list[str] = []
    if situation.type in {'insult', 'user_anger'}:
        conflicts.append('tone_pressure')
    if not previous_state.graph_anchors:
        conflicts.append('thin_graph_grounding')
    direction = 'clarify_then_answer' if situation.type == 'neutral_query' and float(situation.severity or 0.0) >= 0.45 else 'answer_from_updated_state'
    tension = 'elevated' if pressure_points else 'moderate'
    uncertainty_level = 'high' if 'thin_graph_grounding' in conflicts else ('moderate' if situation.type == 'neutral_query' else 'low')
    risk_level = 'high' if any(item in pressure_points for item in ('user_distress', 'behavioral_abnormality')) else ('medium' if pressure_points else 'low')
    base = InfluenceInterpretation(
        summary=_clip_text(
            f"The message activates {', '.join(themes or ['general interaction'])} and pressures {', '.join(pressure_points or ['no acute pressure'])}.",
            limit=800,
        ),
        activation=_normalize_items([getattr(item, 'name', '') for item in list(analysis.entities or [])], limit=8),
        pressure_points=_normalize_items(pressure_points, limit=8),
        themes=_normalize_items(themes, limit=8),
        conflicts=_normalize_items(conflicts, limit=8),
        direction=direction,
        tension=tension,
        role_pressure=active_role or previous_state.active_role or 'ally',
        uncertainty_level=uncertainty_level,
        risk_level=risk_level,
        source='deterministic',
    )
    payload = None
    if _stage_model_enabled('influence_interpreter'):
        prompt = build_influence_interpreter_prompt(
            previous_state=previous_state.to_dict(),
            message=message,
            analysis=analysis.to_dict(),
            situation=situation.to_dict(),
        )
        payload = call_json_model_for_role(prompt, role=get_runtime_config().roles.extraction)
    if isinstance(payload, dict):
        summary = _clip_text(str(payload.get('summary') or ''), limit=800)
        if summary:
            base.summary = summary
            base.source = 'llm_guided'
        base.activation = _normalize_items(list(payload.get('activation') or base.activation), limit=8)
        base.pressure_points = _normalize_items(list(payload.get('pressure_points') or base.pressure_points), limit=8)
        base.themes = _normalize_items(list(payload.get('themes') or base.themes), limit=8)
        base.conflicts = _normalize_items(list(payload.get('conflicts') or base.conflicts), limit=8)
        base.direction = str(payload.get('direction') or base.direction).strip() or base.direction
        base.tension = str(payload.get('tension') or base.tension).strip() or base.tension
        base.role_pressure = str(payload.get('role_pressure') or base.role_pressure).strip() or base.role_pressure
        base.uncertainty_level = str(payload.get('uncertainty_level') or base.uncertainty_level).strip() or base.uncertainty_level
        base.risk_level = str(payload.get('risk_level') or base.risk_level).strip() or base.risk_level
    return base


def reconstruct_task_procedure(
    *,
    message: str,
    reply_language: str,
    analysis: MessageAnalysis,
    situation: Situation,
    previous_state: StateSnapshot,
    influence: InfluenceInterpretation,
    persona_bundle: HeadBundle | None,
    dossier_update_statement: bool = False,
) -> tuple[TaskProcedurePlan, dict[str, Any]]:
    plan, semantic_focus = seed_task_procedure(
        message=message,
        reply_language=reply_language,
        analysis=analysis,
        situation=situation,
        previous_state=previous_state,
        influence=influence,
        persona_bundle=persona_bundle,
        dossier_update_statement=dossier_update_statement,
    )
    payload = None
    if _stage_model_enabled('procedure_reconstructor'):
        prompt = build_procedure_reconstructor_prompt(
            previous_state=previous_state.to_dict(),
            influence=influence.to_dict(),
            analysis=analysis.to_dict(),
            semantic_focus=semantic_focus,
            procedure_seed=plan.to_dict(),
        )
        payload = call_json_model_for_role(prompt, role=get_runtime_config().roles.extraction)
    if isinstance(payload, dict):
        summary = _clip_text(str(payload.get('summary') or ''), limit=900)
        if summary:
            plan.summary = summary
            plan.source = 'llm_guided'
        plan.procedure_family = str(payload.get('procedure_family') or plan.procedure_family).strip() or plan.procedure_family
        plan.requested_outcome = _clip_text(str(payload.get('requested_outcome') or plan.requested_outcome), limit=220)
        plan.response_form = str(payload.get('response_form') or plan.response_form).strip() or plan.response_form
        plan.response_language = str(payload.get('response_language') or plan.response_language).strip() or plan.response_language
        plan.form_drivers = _normalize_items(list(payload.get('form_drivers') or plan.form_drivers), limit=8)
        plan.content_sources = _normalize_items(list(payload.get('content_sources') or plan.content_sources), limit=8)
        plan.forbidden_mixins = _normalize_items(list(payload.get('forbidden_mixins') or plan.forbidden_mixins), limit=10)
        plan.success_criteria = _normalize_items(list(payload.get('success_criteria') or plan.success_criteria), limit=8)
        plan.execution_steps = _normalize_items(list(payload.get('execution_steps') or plan.execution_steps), limit=8)
        plan.uncertainty_strategy = _clip_text(str(payload.get('uncertainty_strategy') or plan.uncertainty_strategy), limit=220)
    return plan, semantic_focus


def build_bounded_next_state(
    *,
    turn_id: str,
    session_id: str,
    previous_state: StateSnapshot,
    influence: InfluenceInterpretation,
    task_procedure: TaskProcedurePlan,
    active_bundle: HeadBundle | None,
    current_entity: str,
    social_role: SocialRoleDecision,
    mood_report: MoodResearchReport | None,
) -> StateSnapshot:
    next_state = StateSnapshot(
        turn_id=turn_id,
        session_id=session_id,
        persona_name=str(active_bundle.name if active_bundle is not None else previous_state.persona_name),
        current_entity=str(current_entity or previous_state.current_entity),
        active_role=social_role.role,
        mood_cluster=str(mood_report.latest_cluster_label or '').strip() if mood_report is not None else previous_state.mood_cluster,
        summary='',
        active_layers=_normalize_items(previous_state.active_layers + ['working_context', 'task_procedure'], limit=8),
        graph_anchors=_bundle_graph_anchors(active_bundle) or list(previous_state.graph_anchors),
        memory_anchors=_bundle_memory_anchors(active_bundle) or list(previous_state.memory_anchors),
        priorities=_normalize_items((list(previous_state.priorities) + list(influence.themes) + _bundle_priorities(active_bundle) + list(task_procedure.success_criteria[:3])), limit=8),
        risks=_normalize_items(list(previous_state.risks) + list(influence.pressure_points) + [influence.risk_level], limit=8),
        constraints=_normalize_items(list(previous_state.constraints) + ['respond_from_reviewed_context_only', f"match_form:{task_procedure.response_form}"] + list(task_procedure.forbidden_mixins[:3]), limit=8),
        changed=[],
        unchanged=[],
        source='deterministic',
    )
    if active_bundle is None or previous_state.persona_name != next_state.persona_name:
        next_state.changed.append('active_persona_selection')
    if previous_state.active_role != next_state.active_role:
        next_state.changed.append('active_role')
    if previous_state.mood_cluster != next_state.mood_cluster:
        next_state.changed.append('mood_cluster')
    if previous_state.memory_anchors != next_state.memory_anchors:
        next_state.changed.append('memory_activation')
    if previous_state.graph_anchors != next_state.graph_anchors:
        next_state.changed.append('graph_activation')
    for item in ('baseline_identity', 'responsibility_invariants'):
        if item not in next_state.changed:
            next_state.unchanged.append(item)
    posture = ', '.join(_emotion_posture(active_bundle)) or 'composed'
    next_state.summary = _clip_text(
        f"Active role is {next_state.active_role}. The message pushes {', '.join(influence.themes or ['general interaction'])}. "
        f"Task procedure is {task_procedure.procedure_family} in form {task_procedure.response_form}. "
        f"Current dynamic posture is {posture}. Priorities now are {', '.join(next_state.priorities[:4])}.",
        limit=900,
    )
    payload = None
    if _stage_model_enabled('state_transition_guide'):
        prompt = build_state_transition_prompt(
            previous_state=previous_state.to_dict(),
            influence=influence.to_dict(),
            updated_state_seed=next_state.to_dict(),
        )
        payload = call_json_model_for_role(prompt, role=get_runtime_config().roles.extraction)
    if isinstance(payload, dict):
        summary = _clip_text(str(payload.get('summary') or ''), limit=900)
        if summary:
            next_state.summary = summary
            next_state.source = 'llm_guided'
        next_state.changed = _normalize_items(list(payload.get('changed') or next_state.changed), limit=8)
        next_state.unchanged = _normalize_items(list(payload.get('unchanged') or next_state.unchanged), limit=8)
        next_state.priorities = _normalize_items(list(payload.get('priorities') or next_state.priorities), limit=8)
        next_state.risks = _normalize_items(list(payload.get('risks') or next_state.risks), limit=8)
        next_state.constraints = _normalize_items(list(payload.get('constraints') or next_state.constraints), limit=8)
    return next_state


def build_working_context_layer(
    *,
    turn_id: str,
    session_id: str,
    current_entity: str,
    persona_name: str,
    updated_state: StateSnapshot,
    influence: InfluenceInterpretation,
    task_procedure: TaskProcedurePlan,
    built_context: dict[str, Any],
) -> WorkingContextLayer:
    source_counts = {str(key): int(value or 0) for key, value in dict(dict(built_context.get('context_debug') or {}).get('source_counts') or {}).items()}
    selected_items = [
        {
            'source': str(item.get('source') or ''),
            'item_type': str(item.get('item_type') or ''),
            'title': str(item.get('title') or ''),
            'score': dict(item.get('score') or {}),
            'reasons': list(item.get('reasons') or []),
        }
        for item in list(dict(built_context.get('context_debug') or {}).get('selected_items') or [])[:12]
        if isinstance(item, dict)
    ]
    summary = _clip_text(
        f"Working context is built from the updated state, task procedure {task_procedure.procedure_family}, "
        f"role {updated_state.active_role}, themes {', '.join(influence.themes or ['general'])}, "
        f"and sources {', '.join(key for key, value in source_counts.items() if value)}.",
        limit=900,
    )
    layer = WorkingContextLayer(
        context_id=f'context:{uuid4().hex[:12]}',
        turn_id=turn_id,
        session_id=session_id,
        persona_name=persona_name,
        current_entity=current_entity,
        summary=summary,
        sections={
            'updated_state_summary': updated_state.summary,
            'influence_summary': influence.summary,
            'task_procedure_summary': task_procedure.summary,
            'persona_block': str(built_context.get('persona_block') or '').strip(),
            'graph_context': str(built_context.get('graph_context') or '').strip(),
            'recent_dialogue': str(built_context.get('recent_dialogue') or '').strip(),
        },
        important_items=[
            {
                'source': 'task_procedure',
                'item_type': 'task_contract',
                'title': task_procedure.procedure_family,
                'score': {'confidence': 1.0, 'relevance': 1.0},
                'reasons': list(task_procedure.success_criteria[:4]),
            },
            *selected_items[:7],
        ],
        weak_items=[],
        contradictions=[],
        priorities=_normalize_items(list(updated_state.priorities) + list(task_procedure.execution_steps[:2]), limit=8),
        constraints=_normalize_items(list(updated_state.constraints) + list(task_procedure.forbidden_mixins[:3]), limit=8),
        risks=list(updated_state.risks),
        source_counts=source_counts,
        estimated_tokens=int(built_context.get('estimated_tokens') or 0),
        source='deterministic',
    )
    payload = None
    if _stage_model_enabled('context_curator'):
        prompt = build_context_curator_prompt(updated_state=updated_state.to_dict(), working_context_seed=layer.to_dict())
        payload = call_json_model_for_role(prompt, role=get_runtime_config().roles.extraction)
    if isinstance(payload, dict):
        text = _clip_text(str(payload.get('summary') or ''), limit=900)
        if text:
            layer.summary = text
            layer.source = 'llm_guided'
        layer.priorities = _normalize_items(list(payload.get('priorities') or layer.priorities), limit=8)
        layer.constraints = _normalize_items(list(payload.get('constraints') or layer.constraints), limit=8)
        layer.risks = _normalize_items(list(payload.get('risks') or layer.risks), limit=8)
    return layer


def review_working_context(
    *,
    layer: WorkingContextLayer,
    updated_state: StateSnapshot,
    task_procedure: TaskProcedurePlan,
) -> WorkingContextLayer:
    contradictions: list[str] = []
    important_items = list(layer.important_items)
    weak_items: list[dict[str, Any]] = []
    for item in important_items:
        score = dict(item.get('score') or {})
        confidence = float(score.get('confidence') or 0.0)
        relevance = float(score.get('relevance') or 0.0)
        if confidence and confidence < 0.55:
            weak_items.append({**item, 'issue': 'low_confidence'})
        elif relevance and relevance < 0.08:
            weak_items.append({**item, 'issue': 'low_relevance'})
    persona_block = str(layer.sections.get('persona_block') or '').strip()
    graph_context = str(layer.sections.get('graph_context') or '').strip()
    if persona_block and graph_context and 'Identity lock' in persona_block and 'relation:' not in graph_context and not updated_state.graph_anchors:
        contradictions.append('graph_context_is_thin_relative_to_persona_state')
    reviewed = WorkingContextLayer(
        context_id=layer.context_id,
        turn_id=layer.turn_id,
        session_id=layer.session_id,
        persona_name=layer.persona_name,
        current_entity=layer.current_entity,
        summary=_clip_text(
            f"Reviewed context keeps {len(important_items[:8])} important items, flags {len(weak_items)} weak items, and enforces {', '.join(layer.constraints[:4])}.",
            limit=900,
        ),
        sections={
            key: _clip_text(value, limit=1800 if key == 'persona_block' else 1200)
            for key, value in dict(layer.sections or {}).items()
            if str(value or '').strip()
        },
        important_items=important_items[:8],
        weak_items=weak_items[:8],
        contradictions=contradictions[:6],
        priorities=_normalize_items(list(layer.priorities) + list(task_procedure.success_criteria[:2]), limit=8),
        constraints=_normalize_items(list(layer.constraints) + ['answer_from_reviewed_context_only'] + list(task_procedure.forbidden_mixins[:3]), limit=8),
        risks=list(layer.risks),
        source_counts=dict(layer.source_counts),
        estimated_tokens=int(layer.estimated_tokens or 0),
        source='deterministic',
    )
    payload = None
    if _stage_model_enabled('context_reviewer'):
        prompt = build_context_reviewer_prompt(working_context=reviewed.to_dict())
        payload = call_json_model_for_role(prompt, role=get_runtime_config().roles.extraction)
    if isinstance(payload, dict):
        summary = _clip_text(str(payload.get('summary') or ''), limit=900)
        if summary:
            reviewed.summary = summary
            reviewed.source = 'llm_guided'
        reviewed.contradictions = _normalize_items(list(payload.get('contradictions') or reviewed.contradictions), limit=8)
        reviewed.risks = _normalize_items(list(payload.get('risks') or reviewed.risks), limit=8)
        reviewed.constraints = _normalize_items(list(payload.get('constraints') or reviewed.constraints), limit=8)
    return reviewed


def shape_response_plan(
    *,
    reviewed_context: WorkingContextLayer,
    influence: InfluenceInterpretation,
    task_procedure: TaskProcedurePlan,
    social_role: SocialRoleDecision,
    response_explanation: PersonaResponseExplanation,
) -> ResponseShapingPlan:
    if influence.risk_level == 'high':
        behavior_mode = 'protective_narrowing'
        risk_posture = 'protective'
    elif influence.uncertainty_level in {'high', 'severe'}:
        behavior_mode = 'bounded_answer'
        risk_posture = 'measured'
    elif influence.direction == 'clarify_then_answer':
        behavior_mode = 'clarifying_answer'
        risk_posture = 'measured'
    else:
        behavior_mode = 'grounded_answer'
        risk_posture = 'open'
    plan = ResponseShapingPlan(
        role=social_role.role,
        style=response_explanation.response_style or 'direct_explanatory',
        behavior_mode=behavior_mode,
        response_form=task_procedure.response_form,
        summary=_clip_text(
            f"Respond as {social_role.role} in a {response_explanation.response_style or 'grounded'} style, using reviewed context only. "
            f"Treat the task as {task_procedure.procedure_family} with form {task_procedure.response_form} and prioritize {', '.join(reviewed_context.priorities[:3])}.",
            limit=900,
        ),
        priorities=_normalize_items(list(reviewed_context.priorities), limit=8),
        constraints=_normalize_items(list(reviewed_context.constraints) + ['stay_in_first_person', 'avoid_assistant_tone'], limit=8),
        forbidden_mixins=_normalize_items(list(task_procedure.forbidden_mixins), limit=10),
        success_criteria=_normalize_items(list(task_procedure.success_criteria), limit=8),
        risk_posture=risk_posture,
        source='deterministic',
    )
    payload = None
    if _stage_model_enabled('response_shaper'):
        prompt = build_response_shaper_prompt(
            reviewed_context=reviewed_context.to_dict(),
            influence=influence.to_dict(),
            task_procedure=task_procedure.to_dict(),
            social_role=social_role.to_dict(),
            response_explanation=response_explanation.to_dict(),
        )
        payload = call_json_model_for_role(prompt, role=get_runtime_config().roles.extraction)
    if isinstance(payload, dict):
        role = str(payload.get('role') or '').strip()
        style = str(payload.get('style') or '').strip()
        behavior = str(payload.get('behavior_mode') or '').strip()
        response_form = str(payload.get('response_form') or '').strip()
        summary = _clip_text(str(payload.get('summary') or ''), limit=900)
        if role:
            plan.role = role
        if style:
            plan.style = style
        if behavior:
            plan.behavior_mode = behavior
        if response_form:
            plan.response_form = response_form
        if summary:
            plan.summary = summary
            plan.source = 'llm_guided'
        plan.priorities = _normalize_items(list(payload.get('priorities') or plan.priorities), limit=8)
        plan.constraints = _normalize_items(list(payload.get('constraints') or plan.constraints), limit=8)
        plan.forbidden_mixins = _normalize_items(list(payload.get('forbidden_mixins') or plan.forbidden_mixins), limit=10)
        plan.success_criteria = _normalize_items(list(payload.get('success_criteria') or plan.success_criteria), limit=8)
        plan.risk_posture = str(payload.get('risk_posture') or plan.risk_posture).strip() or plan.risk_posture
    return plan


def render_state_transition_block(*, previous_state: StateSnapshot, influence: InfluenceInterpretation, next_state: StateSnapshot) -> str:
    lines = [
        f'Previous state: {previous_state.summary}',
        f'Influence: {influence.summary}',
        f'Updated state: {next_state.summary}',
    ]
    if next_state.changed:
        lines.append(f"Changed: {' | '.join(next_state.changed[:6])}.")
    if next_state.unchanged:
        lines.append(f"Unchanged invariants: {' | '.join(next_state.unchanged[:4])}.")
    return '\n'.join(lines).strip()


def render_reviewed_context_block(reviewed_context: WorkingContextLayer) -> str:
    parts = [reviewed_context.summary]
    for key in ('updated_state_summary', 'influence_summary', 'task_procedure_summary', 'persona_block', 'graph_context', 'recent_dialogue'):
        text = str(reviewed_context.sections.get(key) or '').strip()
        if text:
            parts.append(f'{key}: {text}')
    if reviewed_context.contradictions:
        parts.append(f"Contradictions: {' | '.join(reviewed_context.contradictions[:4])}.")
    if reviewed_context.risks:
        parts.append(f"Risks: {' | '.join(reviewed_context.risks[:4])}.")
    return '\n\n'.join(part for part in parts if part)


def render_response_shaping_block(plan: ResponseShapingPlan) -> str:
    lines = [
        f'Selected response role: {plan.role}.',
        f'Style: {plan.style}.',
        f'Behavior mode: {plan.behavior_mode}.',
        f'Response form: {plan.response_form}.',
        f'Risk posture: {plan.risk_posture}.',
        f'Summary: {plan.summary}',
    ]
    if plan.priorities:
        lines.append(f"Priorities: {' | '.join(plan.priorities[:4])}.")
    if plan.constraints:
        lines.append(f"Constraints: {' | '.join(plan.constraints[:5])}.")
    if plan.forbidden_mixins:
        lines.append(f"Forbidden mixins: {' | '.join(plan.forbidden_mixins[:5])}.")
    if plan.success_criteria:
        lines.append(f"Success criteria: {' | '.join(plan.success_criteria[:5])}.")
    return '\n'.join(lines).strip()


def render_task_procedure_block(plan: TaskProcedurePlan) -> str:
    lines = [
        f'Task family: {plan.procedure_family}.',
        f'Requested outcome: {plan.requested_outcome}',
        f'Visible form: {plan.response_form}.',
        f'Visible language: {plan.response_language}.',
        f'Summary: {plan.summary}',
    ]
    if plan.form_drivers:
        lines.append(f"Form drivers: {' | '.join(plan.form_drivers[:5])}.")
    if plan.content_sources:
        lines.append(f"Content sources: {' | '.join(plan.content_sources[:5])}.")
    if plan.forbidden_mixins:
        lines.append(f"Do not mix in: {' | '.join(plan.forbidden_mixins[:5])}.")
    if plan.success_criteria:
        lines.append(f"Success looks like: {' | '.join(plan.success_criteria[:5])}.")
    if plan.execution_steps:
        lines.append(f"Execution procedure: {' | '.join(plan.execution_steps[:6])}.")
    if plan.uncertainty_strategy:
        lines.append(f"Uncertainty strategy: {plan.uncertainty_strategy}")
    return '\n'.join(lines).strip()


def persist_current_context(*, reviewed_context: WorkingContextLayer, response_plan: ResponseShapingPlan, task_procedure: TaskProcedurePlan, previous_state: StateSnapshot, influence: InfluenceInterpretation, next_state: StateSnapshot) -> tuple[str, str]:
    config = get_runtime_config()
    json_path = config.paths.current_context_json
    txt_path = config.paths.current_context_txt
    payload = {
        'generated_at': _utc_now(),
        'turn_id': reviewed_context.turn_id,
        'session_id': reviewed_context.session_id,
        'persona_name': reviewed_context.persona_name,
        'current_entity': reviewed_context.current_entity,
        'previous_state': previous_state.to_dict(),
        'influence': influence.to_dict(),
        'task_procedure': task_procedure.to_dict(),
        'updated_state': next_state.to_dict(),
        'reviewed_context': reviewed_context.to_dict(),
        'response_plan': response_plan.to_dict(),
    }
    text = '\n\n'.join(
        [
            f"Turn: {reviewed_context.turn_id}",
            f"Persona: {reviewed_context.persona_name or 'unknown'}",
            f"Current entity: {reviewed_context.current_entity or 'unknown'}",
            f"Previous state: {previous_state.summary}",
            f"Influence: {influence.summary}",
            f"Task procedure: {task_procedure.summary}",
            f"Updated state: {next_state.summary}",
            f"Reviewed context: {reviewed_context.summary}",
            f"Response plan: {response_plan.summary}",
        ]
    ).strip()
    with _CURRENT_CONTEXT_LOCK:
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        txt_path.write_text(text, encoding='utf-8')
    return str(json_path), str(txt_path)


def append_transition_log(
    *,
    turn_id: str,
    session_id: str,
    user_message: str,
    previous_state: StateSnapshot,
    influence: InfluenceInterpretation,
    task_procedure: TaskProcedurePlan,
    next_state: StateSnapshot,
    working_context: WorkingContextLayer,
    reviewed_context: WorkingContextLayer,
    response_plan: ResponseShapingPlan,
    assistant_reply: str,
) -> str:
    path = get_runtime_config().paths.state_transitions_log_path
    response_summary = _clip_text(str(assistant_reply or '').strip(), limit=280)
    payload = {
        'turn_id': turn_id,
        'session_id': session_id,
        'timestamp': _utc_now(),
        'user_message': str(user_message or '').strip(),
        'previous_state': previous_state.to_dict(),
        'interpreted_influence': influence.to_dict(),
        'task_procedure': task_procedure.to_dict(),
        'new_state': next_state.to_dict(),
        'working_context': working_context.to_dict(),
        'reviewed_context': reviewed_context.to_dict(),
        'selected_response_mode': response_plan.to_dict(),
        'final_response_summary': response_summary,
    }
    with _TRANSITION_LOG_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
    return str(path)
