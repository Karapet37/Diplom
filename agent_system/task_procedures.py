from __future__ import annotations

from typing import Any

from .duplicate_resolver import normalize_name
from .language_tools import normalize_language_code
from .models import HeadBundle, InfluenceInterpretation, MessageAnalysis, Situation, TaskProcedurePlan
from .semantic_routing import infer_semantic_focus


_LIST_CUES = ('list', 'bullet', 'bullets', 'reasons', 'points', 'items', 'several')
_STEP_CUES = ('step', 'steps', 'plan', 'roadmap', 'procedure', 'workflow', 'how to')
_COMPARISON_CUES = ('compare', 'comparison', 'difference', 'different', 'versus', 'vs', 'contrast')
_SUMMARY_CUES = ('summary', 'summarize', 'gist', 'brief recap', 'short recap')
_REWRITE_CUES = ('rewrite', 'rephrase', 'polish', 'edit', 'improve wording')
_TRANSLATION_CUES = ('translate', 'translation')


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    clean = normalize_name(text)
    return any(marker in clean for marker in markers)


def _normalize_items(values: list[str], *, limit: int = 8) -> list[str]:
    out: list[str] = []
    for item in list(values or []):
        clean = str(item or '').strip()
        if clean and clean not in out:
            out.append(clean)
        if len(out) >= limit:
            break
    return out


def _clip_text(text: str, *, limit: int = 240) -> str:
    clean = ' '.join(str(text or '').strip().split())
    if not clean:
        return ''
    return clean[:limit].strip()


def _procedure_family(
    *,
    message: str,
    analysis: MessageAnalysis,
    influence: InfluenceInterpretation,
    semantic_focus: dict[str, Any],
    dossier_update_statement: bool,
) -> str:
    focus = [str(item).strip() for item in list(dict(semantic_focus or {}).get('focus') or []) if str(item).strip()]
    normalized = normalize_name(message)
    if dossier_update_statement:
        return 'persona_dossier_update'
    if _contains_any(normalized, _TRANSLATION_CUES):
        return 'translation'
    if _contains_any(normalized, _REWRITE_CUES):
        return 'rewrite'
    if _contains_any(normalized, _SUMMARY_CUES):
        return 'summary'
    if _contains_any(normalized, _COMPARISON_CUES):
        return 'comparison'
    if _contains_any(normalized, _STEP_CUES):
        return 'procedural_guidance'
    if 'identity' in focus:
        return 'persona_identity_answer'
    if 'work' in focus:
        return 'persona_work_answer'
    if 'decision' in focus:
        return 'persona_decision_answer'
    if 'memory' in focus:
        return 'memory_continuity_answer'
    if 'values' in focus or 'relations' in focus or 'conflict' in focus:
        return 'principle_grounded_answer'
    if influence.direction == 'clarify_then_answer':
        return 'clarifying_answer'
    if analysis.user_state.intent == 'statement':
        return 'stateful_social_reaction'
    return 'grounded_direct_answer'


def _response_form(*, family: str, message: str) -> str:
    normalized = normalize_name(message)
    if family == 'translation':
        return 'translation'
    if family == 'rewrite':
        return 'rewrite'
    if family == 'summary':
        return 'summary'
    if family == 'comparison':
        return 'comparison'
    if family == 'persona_dossier_update':
        return 'acknowledgement'
    if _contains_any(normalized, _LIST_CUES):
        return 'structured_list'
    if family == 'procedural_guidance':
        return 'numbered_steps'
    if family in {'persona_identity_answer', 'persona_work_answer', 'persona_decision_answer', 'memory_continuity_answer'}:
        return 'first_person_explanation'
    if family == 'clarifying_answer':
        return 'clarifying_answer'
    if family == 'stateful_social_reaction':
        return 'social_reaction'
    return 'direct_answer'


def _requested_outcome(*, family: str, focus: list[str], message: str) -> str:
    base_map = {
        'persona_identity_answer': 'Explain who the persona is in a concrete first-person way.',
        'persona_work_answer': 'Explain the persona’s work, routine, or responsibilities in first person.',
        'persona_decision_answer': 'Explain how the persona makes judgments under pressure.',
        'memory_continuity_answer': 'Answer using continuity between past turns, dossier memory, and current state.',
        'principle_grounded_answer': 'Answer from values, trust model, and conflicts instead of generic advice.',
        'procedural_guidance': 'Provide a usable procedure rather than vague commentary.',
        'comparison': 'Contrast the requested options along concrete dimensions.',
        'summary': 'Compress the relevant material without dropping core meaning.',
        'rewrite': 'Transform the given wording while preserving intended meaning.',
        'translation': 'Translate faithfully into the target language or requested register.',
        'clarifying_answer': 'Narrow the ambiguity before committing to a full answer.',
        'stateful_social_reaction': 'Respond socially from the persona state instead of producing a detached explanation.',
        'persona_dossier_update': 'Acknowledge the new persona fact and signal that it will affect later replies.',
        'grounded_direct_answer': 'Answer directly from the grounded state and selected sources.',
    }
    if family in base_map:
        return base_map[family]
    return f"Resolve the request using these active focuses: {', '.join(focus or ['general'])}. Message: {_clip_text(message, limit=120)}"


def _form_drivers(*, family: str, response_form: str, reply_language: str, influence: InfluenceInterpretation) -> list[str]:
    drivers = [
        f'reply_language:{reply_language}',
        f'response_form:{response_form}',
        'selected_social_role',
        'persona_first_person_contract',
    ]
    if family in {'persona_identity_answer', 'persona_work_answer', 'persona_decision_answer', 'memory_continuity_answer'}:
        drivers.append('persona-directed_question')
    if family in {'comparison', 'summary', 'rewrite', 'translation'}:
        drivers.append('user-requested_transformation')
    if influence.direction == 'clarify_then_answer':
        drivers.append('ambiguity_requires_narrowing')
    return _normalize_items(drivers, limit=8)


def _content_sources(*, family: str, focus: list[str], persona_bundle: HeadBundle | None) -> list[str]:
    sources = ['updated_state', 'reviewed_context']
    if family in {'persona_identity_answer', 'persona_work_answer', 'persona_decision_answer', 'memory_continuity_answer', 'principle_grounded_answer', 'stateful_social_reaction'}:
        sources.extend(['persona_baseline', 'persona_learned_patterns'])
    if family in {'persona_work_answer', 'grounded_direct_answer', 'comparison', 'procedural_guidance', 'summary'}:
        sources.append('graph_context')
    if family in {'memory_continuity_answer', 'stateful_social_reaction', 'persona_dossier_update'}:
        sources.extend(['recent_dialogue', 'memory_anchors'])
    if family in {'persona_identity_answer', 'persona_work_answer', 'persona_decision_answer', 'principle_grounded_answer'}:
        sources.append('persona_form')
    if family in {'persona_dossier_update', 'rewrite', 'translation'}:
        sources.append('user_message')
    if persona_bundle is not None and list((persona_bundle.persona_form or {}).get('personal_history') or []):
        sources.append('personal_history')
    if 'memory' in focus:
        sources.append('session_memory')
    return _normalize_items(sources, limit=8)


def _forbidden_mixins(*, family: str, reply_language: str) -> list[str]:
    items = [
        'generic_assistant_tone',
        'hidden_reasoning',
        'system_prompt_leakage',
        'unsupported_identity_facts',
        'irrelevant_graph_noise',
        'user_emotion_mirroring',
        f'mixed_output_language_except_{reply_language}',
    ]
    if family in {'persona_identity_answer', 'persona_work_answer', 'persona_decision_answer', 'memory_continuity_answer', 'stateful_social_reaction'}:
        items.extend(['third_person_detachment', 'service_politeness'])
    if family in {'rewrite', 'translation'}:
        items.append('answering_instead_of_transforming')
    if family in {'memory_continuity_answer', 'persona_dossier_update'}:
        items.append('stale_or_deleted_memory')
    if family == 'procedural_guidance':
        items.append('vague_motivational_filler')
    return _normalize_items(items, limit=10)


def _success_criteria(*, family: str, response_form: str, reply_language: str) -> list[str]:
    criteria = [
        f'visible_reply_language_is_{reply_language}',
        f'output_matches_{response_form}',
        'answer_comes_from_selected_sources',
        'persona_identity_is_preserved',
        'no_forbidden_mixins_appear',
    ]
    if family in {'persona_identity_answer', 'persona_work_answer', 'persona_decision_answer', 'memory_continuity_answer'}:
        criteria.append('answer_is_in_first_person')
    if family == 'persona_dossier_update':
        criteria.extend(['new_fact_is_acknowledged', 'future_influence_is_signaled'])
    if family == 'clarifying_answer':
        criteria.append('ambiguity_is_narrowed_before_claiming_specific_facts')
    if family == 'procedural_guidance':
        criteria.append('procedure_is_actionable')
    return _normalize_items(criteria, limit=8)


def _execution_steps(*, family: str, response_form: str, focus: list[str]) -> list[str]:
    general = [
        'identify the real task behind the user wording',
        'lock the visible output form and language',
        'pull only the content sources relevant to that task',
        'filter out forbidden mixins and weak grounding',
        'generate the answer only after the task contract is satisfied',
    ]
    family_steps = {
        'persona_identity_answer': [
            'recover identity-relevant persona layers',
            'select biography, values, and stable traits',
            'answer in first person before adding any framing',
        ],
        'persona_work_answer': [
            'recover work profile, habits, and memory anchors',
            'describe routine and responsibilities concretely',
            'connect routine to decision style under pressure',
        ],
        'persona_decision_answer': [
            'recover decision patterns, values, and conflict behavior',
            'show how facts, trust, and risk are weighed',
            'keep the explanation practical rather than abstract',
        ],
        'memory_continuity_answer': [
            'recover recent dialogue and stored memory anchors',
            'use the new memory causally instead of parroting it',
            'avoid references to deleted or unsupported memory',
        ],
        'principle_grounded_answer': [
            'recover values, trust model, and conflicts',
            'answer from principles instead of generic advice',
        ],
        'procedural_guidance': [
            'recover the generic way this kind of task is done',
            'turn it into ordered steps',
            'keep each step concrete and usable',
        ],
        'comparison': [
            'identify the compared options',
            'line them up on shared dimensions',
            'end with the main trade-off or choice pressure',
        ],
        'summary': [
            'recover the core claims',
            'compress without flattening meaning',
        ],
        'rewrite': [
            'preserve intended meaning',
            'change wording and shape only as requested',
        ],
        'translation': [
            'preserve meaning',
            'move it into the requested language and register',
        ],
        'persona_dossier_update': [
            'identify the new biographical fact',
            'confirm that it enters the learned dossier',
            'signal that future answers will reflect it',
        ],
        'clarifying_answer': [
            'state what is still underspecified',
            'narrow the ambiguity without inventing details',
        ],
        'stateful_social_reaction': [
            'recover the active role and mood pressure',
            'respond socially from state rather than from a template',
        ],
    }
    return _normalize_items(list(family_steps.get(family, [])) + general + [f'focus_on:{item}' for item in list(focus or [])[:2]], limit=8)


def _uncertainty_strategy(*, influence: InfluenceInterpretation, family: str) -> str:
    if family == 'persona_dossier_update':
        return 'acknowledge the update without inventing downstream consequences'
    if influence.risk_level == 'high':
        return 'narrow the claim set and keep only high-confidence grounded content'
    if influence.uncertainty_level in {'high', 'severe'}:
        return 'admit uncertainty, keep the answer bounded, and avoid unsupported specifics'
    if family == 'procedural_guidance':
        return 'fall back to general procedure only when concrete grounded details are thin'
    return 'answer directly while staying inside grounded persona and graph evidence'


def seed_task_procedure(
    *,
    message: str,
    reply_language: str,
    analysis: MessageAnalysis,
    situation: Situation,
    previous_state: Any,
    influence: InfluenceInterpretation,
    persona_bundle: HeadBundle | None,
    dossier_update_statement: bool = False,
) -> tuple[TaskProcedurePlan, dict[str, Any]]:
    focus_payload = infer_semantic_focus(
        question=message,
        persona_bundle=persona_bundle,
        analysis=analysis,
        situation=situation,
    )
    focus = [str(item).strip() for item in list(dict(focus_payload or {}).get('focus') or []) if str(item).strip()]
    family = _procedure_family(
        message=message,
        analysis=analysis,
        influence=influence,
        semantic_focus=focus_payload,
        dossier_update_statement=dossier_update_statement,
    )
    response_form = _response_form(family=family, message=message)
    normalized_language = normalize_language_code(reply_language, fallback='en')
    plan = TaskProcedurePlan(
        procedure_family=family,
        requested_outcome=_requested_outcome(family=family, focus=focus, message=message),
        response_form=response_form,
        response_language=normalized_language,
        form_drivers=_form_drivers(
            family=family,
            response_form=response_form,
            reply_language=normalized_language,
            influence=influence,
        ),
        content_sources=_content_sources(family=family, focus=focus, persona_bundle=persona_bundle),
        forbidden_mixins=_forbidden_mixins(family=family, reply_language=normalized_language),
        success_criteria=_success_criteria(
            family=family,
            response_form=response_form,
            reply_language=normalized_language,
        ),
        execution_steps=_execution_steps(family=family, response_form=response_form, focus=focus),
        uncertainty_strategy=_uncertainty_strategy(influence=influence, family=family),
        summary=_clip_text(
            f"Task family={family}. Produce a {response_form} in {normalized_language} using {', '.join(_content_sources(family=family, focus=focus, persona_bundle=persona_bundle)[:4])} and excluding forbidden mixins.",
            limit=260,
        ),
        source='deterministic',
    )
    return plan, focus_payload
