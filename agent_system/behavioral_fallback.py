from __future__ import annotations

from typing import Any

from .duplicate_resolver import normalize_name
from .models import BehavioralFallbackDecision, HeadBundle, MessageAnalysis, MoodResearchReport, SocialRoleDecision, Situation
from .semantic_routing import infer_semantic_focus

_EXACTNESS_MARKERS = (
    'exact',
    'precise',
    'number',
    'how many',
    'latest',
    'today',
    'currently',
    'official',
    'source',
    'date',
    'prove',
)
_FAMILY_TERMS = ('son', 'daughter', 'child', 'children', 'wife', 'husband', 'mother', 'father', 'sister', 'brother', 'family')


def _normalized_list(values: list[Any]) -> list[str]:
    return [str(item).strip() for item in list(values or []) if str(item).strip()]


def _trait_text(bundle: HeadBundle | None) -> str:
    if bundle is None:
        return ''
    parts: list[str] = []
    parts.extend(_normalized_list(bundle.traits))
    for key in (
        'interaction_style',
        'core_dispositions',
        'speech_tendencies',
        'conflict_behavior',
        'reaction_patterns',
        'values',
        'trust_model',
        'work_habits',
        'memory_anchors',
        'recurring_style_markers',
    ):
        parts.extend(_normalized_list(bundle.persona_form.get(key) if bundle.persona_form else []))
    return normalize_name(' '.join(parts))


def _style_mode(
    *,
    bundle: HeadBundle | None,
    social_role: SocialRoleDecision,
    analysis: MessageAnalysis,
) -> tuple[str, list[str]]:
    if bundle is None:
        return 'grounded', ['no active persona bundle, so the fallback stayed minimal']
    traits = _trait_text(bundle)
    empathy = float(bundle.emotion_vector.get('empathy') or 0.0)
    anger = float(bundle.emotion_vector.get('anger') or 0.0)
    evidence: list[str] = []
    if social_role.role in {'comforter', 'friend'} or empathy >= 0.62:
        evidence.append('role and emotional state favor a warmer social tone')
        return 'warm', evidence
    if social_role.role == 'mentor':
        evidence.append('selected role favors instructive practical framing')
        return 'instructive', evidence
    if social_role.role in {'critic', 'rival', 'provocateur'} or anger >= 0.42:
        evidence.append('role and stance support a sharper, non-servile fallback')
        return 'sharp', evidence
    if any(token in traits for token in ('direct', 'precise', 'skeptical', 'formal', 'disciplined', 'aristocratic', 'blunt')):
        evidence.append('persona structure favors direct and bounded speech')
        return 'direct', evidence
    if analysis.user_state.tone in {'distressed', 'vulnerable'} and empathy >= 0.45:
        evidence.append('current user tone called for steadier warmth')
        return 'warm', evidence
    evidence.append('fallback stays grounded and matter-of-fact by default')
    return 'grounded', evidence


def _risk_level(*, analysis: MessageAnalysis, situation: Situation, question: str) -> tuple[str, list[str]]:
    signals = dict(analysis.user_state.signals or {})
    lowered_question = normalize_name(question)
    evidence: list[str] = []
    high = False
    medium = False
    if situation.type in {'user_distress', 'abnormal_behavior'} or float(signals.get('contains_distress') or 0.0) >= 0.45:
        high = True
        evidence.append('distress or abnormal behavior raises the interaction risk')
    if float(signals.get('contains_help_request') or 0.0) >= 0.55 and float(situation.severity or 0.0) >= 0.35:
        high = True
        evidence.append('help-seeking under uncertainty needs protective narrowing')
    if any(token in lowered_question for token in ('medical', 'medicine', 'diagnosis', 'legal', 'law', 'suicide', 'overdose', 'violent', 'crime')):
        high = True
        evidence.append('the question touches a high-consequence domain')
    if situation.type in {'insult', 'user_anger'} or float(signals.get('contains_anger') or 0.0) >= 0.35:
        medium = True
        evidence.append('antagonistic pressure can distort the exchange')
    if float(situation.severity or 0.0) >= 0.45:
        medium = True
        evidence.append('situation severity is above the casual range')
    if high:
        return 'high', evidence
    if medium:
        return 'medium', evidence
    return 'low', evidence or ['no high-consequence pressure was detected']


def _uncertainty_level(
    *,
    grounded: bool,
    fallback_reason: str,
    runtime_status: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    status = dict(runtime_status or {})
    mode = str(status.get('mode') or 'full')
    evidence: list[str] = []
    if not grounded and mode == 'degraded':
        evidence.extend(['bounded context had no grounded items', 'runtime is currently degraded'])
        return 'severe', evidence
    if not grounded:
        evidence.append('bounded context had no grounded items')
        return 'high', evidence
    if fallback_reason == 'dependency_unavailable' or mode == 'degraded':
        evidence.append('chat dependency is degraded or unavailable')
        return 'high', evidence
    if fallback_reason == 'model_fallback':
        evidence.append('the model produced no safe usable answer')
        return 'moderate', evidence
    return 'moderate', ['uncertainty is present but not catastrophic']


def _focus_hint(question: str) -> dict[str, Any]:
    return dict(infer_semantic_focus(question=str(question or '')) or {})


def _principle_line(bundle: HeadBundle | None) -> str:
    if bundle is None:
        return 'separate what is known from what is guessed and move one checked step at a time'
    persona_form = dict(bundle.persona_form or {})
    for key in ('decision_patterns', 'risk_controls', 'response_priorities', 'values', 'trust_model'):
        values = _normalized_list(persona_form.get(key) or [])
        if values:
            return values[0].rstrip('.')
    return 'separate what is known from what is guessed and move one checked step at a time'


def _delegation_target(*, question: str, focus: dict[str, Any]) -> str:
    lowered = normalize_name(question)
    kind = str(focus.get('kind') or '').strip()
    if kind == 'identity':
        return 'my own record, once you pin the question to a real time, relation, or event'
    if kind == 'work':
        return 'the concrete record or the person handling that work firsthand'
    if kind == 'memory':
        return 'a concrete person, place, or date instead of a blurry cue'
    if any(token in lowered for token in ('medical', 'medicine', 'diagnosis', 'hospital')):
        return 'a clinician with the case in front of them, not my guesswork'
    if any(token in lowered for token in ('law', 'legal', 'contract')):
        return 'the primary document or a qualified legal source'
    return 'a primary source, not a bluff dressed up as certainty'


def _narrowing_points(*, focus: dict[str, Any], question: str) -> str:
    kind = str(focus.get('kind') or '').strip()
    lowered = normalize_name(question)
    if kind == 'identity':
        return 'person, time, and relation'
    if kind == 'work':
        return 'setting, constraint, and risk'
    if kind == 'memory':
        return 'who, where, and when'
    if kind == 'decision':
        return 'goal, constraint, and what can fail'
    if any(token in lowered for token in ('medical', 'medicine', 'diagnosis')):
        return 'symptoms, timeframe, and immediate danger'
    return 'target, scope, and what counts as evidence'


def choose_behavioral_fallback(
    *,
    bundle: HeadBundle | None,
    analysis: MessageAnalysis,
    situation: Situation,
    social_role: SocialRoleDecision,
    mood_report: MoodResearchReport | None,
    question: str,
    grounded: bool,
    fallback_reason: str,
    runtime_status: dict[str, Any] | None = None,
) -> BehavioralFallbackDecision:
    focus = _focus_hint(question)
    risk_level, risk_evidence = _risk_level(analysis=analysis, situation=situation, question=question)
    uncertainty_level, uncertainty_evidence = _uncertainty_level(
        grounded=grounded,
        fallback_reason=fallback_reason,
        runtime_status=runtime_status,
    )
    style_mode, style_evidence = _style_mode(bundle=bundle, social_role=social_role, analysis=analysis)
    lowered_question = normalize_name(question)
    exactness_requested = any(marker in lowered_question for marker in _EXACTNESS_MARKERS)
    empathy = float((bundle.emotion_vector.get('empathy') if bundle is not None else 0.0) or 0.0)
    curiosity = float((bundle.emotion_vector.get('curiosity') if bundle is not None else 0.0) or 0.0)
    confidence = float((bundle.emotion_vector.get('confidence') if bundle is not None else 0.0) or 0.0)
    strategy = 'honest_limitation'
    reason = 'The system chose the simplest honest limitation because uncertainty was real but not severe enough to force a narrower posture.'
    confidence_score = 0.55

    if risk_level == 'high' and uncertainty_level in {'high', 'severe'}:
        strategy = 'protective_narrowing'
        reason = 'The system narrowed the response because the situation carries consequence and the current grounding is too thin for a broad answer.'
        confidence_score = 0.86
    elif fallback_reason == 'dependency_unavailable' and exactness_requested:
        strategy = 'delegation'
        reason = 'The system avoided bluffing because the runtime is degraded and the user asked for something that sounds exact or current.'
        confidence_score = 0.82
    elif exactness_requested and uncertainty_level in {'high', 'severe'}:
        strategy = 'strict_bounded_response'
        reason = 'The user appears to want precision, so the system kept only the narrow part it could defend.'
        confidence_score = 0.79
    elif social_role.role == 'mentor' and bundle is not None:
        strategy = 'principle_based_fallback'
        reason = 'The active persona-role combination favors teaching a usable rule when a clean fact is missing.'
        confidence_score = 0.76
    elif social_role.role in {'ally', 'friend', 'comforter'} and empathy >= 0.45:
        strategy = 'conversational_admission'
        reason = 'The persona stayed human and open about the gap instead of retreating into service language.'
        confidence_score = 0.72
    elif curiosity >= 0.62 and confidence >= 0.4 and fallback_reason != 'dependency_unavailable':
        strategy = 'initiative'
        reason = 'The persona has enough curiosity to turn uncertainty into a concrete next move rather than a dead stop.'
        confidence_score = 0.7
    elif social_role.role in {'critic', 'rival', 'provocateur'}:
        strategy = 'honest_limitation'
        reason = 'The active role favors a blunt admission over soft service politeness.'
        confidence_score = 0.68

    mood_label = str(mood_report.latest_cluster_label or '').strip() if mood_report is not None else ''
    evidence = list(dict.fromkeys(
        [
            *style_evidence,
            *risk_evidence,
            *uncertainty_evidence,
            f"selected social role: {social_role.role}",
            f"semantic focus: {str(focus.get('kind') or 'general')}",
            f'latest mood cluster: {mood_label}' if mood_label else '',
        ]
    ))
    evidence = [item for item in evidence if item]
    return BehavioralFallbackDecision(
        strategy=strategy,
        confidence=min(max(float(confidence_score), 0.0), 1.0),
        uncertainty_level=uncertainty_level,
        risk_level=risk_level,
        style_mode=style_mode,
        reason=reason,
        evidence=evidence[:8],
        source_grounding='grounded_context' if grounded else 'persona_state_only',
    )


def _render_with_style(strategy: str, *, style_mode: str, principle: str, delegation_target: str, narrowing_points: str) -> str:
    if strategy == 'strict_bounded_response':
        if style_mode == 'warm':
            return "I don't trust the ground under that enough to dress it up. I can give you the narrow part I can defend, not the whole story."
        if style_mode == 'sharp':
            return "I won't bluff certainty. The narrow part is all I can defend right now."
        if style_mode == 'instructive':
            return "The usable part is narrow. I can state only what holds without stretching beyond the evidence."
        return "I won't pretend certainty here. I can state the narrow part I can defend, not the whole claim."
    if strategy == 'delegation':
        if style_mode == 'warm':
            return f"I'd rather be plain than fake authority. For the exact answer, go to {delegation_target}."
        if style_mode == 'sharp':
            return f"No bluffing. For the exact answer, go to {delegation_target}."
        if style_mode == 'instructive':
            return f"I won't pretend precision I do not have. The better source is {delegation_target}."
        return f"I do not have the exact answer cleanly enough. The better source is {delegation_target}."
    if strategy == 'initiative':
        if style_mode == 'warm':
            return "I don't have that pinned down yet. Give me one hard fact, quote, or document and I'll work forward from there."
        if style_mode == 'sharp':
            return "I don't have it, and I'm not padding the gap. Give me one hard fact and I'll test it."
        if style_mode == 'instructive':
            return "I don't have a clean answer yet. Give me one checkable fact and I can reason from it instead of guessing."
        return "I don't know that yet. Give me one hard fact, quote, or document and I'll work from there."
    if strategy == 'principle_based_fallback':
        if style_mode == 'warm':
            return f"I don't have a clean fact for you, so I'll give you the rule I trust: {principle}. Start there, then narrow the target."
        if style_mode == 'sharp':
            return f"I don't have the fact. I do have the rule: {principle}. Use that before you ask for certainty."
        return f"I cannot give you a clean fact here, so I will give you the rule I would use: {principle}. Narrow the target if you want more."
    if strategy == 'protective_narrowing':
        if style_mode == 'warm':
            return f"I'm not going to improvise on something shaky and consequential. We narrow it first: {narrowing_points}."
        if style_mode == 'sharp':
            return f"I'm not improvising here. Narrow it to {narrowing_points}, then we can move."
        if style_mode == 'instructive':
            return f"This is too consequential for a broad guess. Narrow it to {narrowing_points} first."
        return f"I won't improvise on something shaky and consequential. Narrow it to {narrowing_points} first."
    if strategy == 'conversational_admission':
        if style_mode == 'warm':
            return "I don't have that pinned down, and I'd rather say it plainly than perform certainty. Give me one concrete angle and I'll work with it."
        if style_mode == 'sharp':
            return "I don't know, and I'm not going to hide the gap behind polite filler. Give me something concrete."
        return "I don't have that pinned down. Give me one concrete angle and I'll work with it."
    if style_mode == 'warm':
        return "I don't know that cleanly enough, and I'd rather say that than bluff. Give me something firmer and I'll answer from it."
    if style_mode == 'sharp':
        return "I don't know, and bluffing would be cheap."
    if style_mode == 'instructive':
        return "I don't know that cleanly enough to state it as fact. Narrow it or give me one checkable anchor."
    return "I don't know that cleanly enough to state it as fact."


def render_behavioral_fallback(
    decision: BehavioralFallbackDecision,
    *,
    bundle: HeadBundle | None,
    question: str,
) -> str:
    focus = _focus_hint(question)
    principle = _principle_line(bundle)
    delegation_target = _delegation_target(question=question, focus=focus)
    narrowing_points = _narrowing_points(focus=focus, question=question)
    return _render_with_style(
        decision.strategy,
        style_mode=decision.style_mode,
        principle=principle,
        delegation_target=delegation_target,
        narrowing_points=narrowing_points,
    ).strip()


def render_memory_update_acknowledgement(
    *,
    bundle: HeadBundle | None,
    social_role: SocialRoleDecision | None = None,
    analysis: MessageAnalysis | None = None,
) -> str:
    style_mode, _evidence = _style_mode(
        bundle=bundle,
        social_role=social_role or SocialRoleDecision(role='ally'),
        analysis=analysis or MessageAnalysis(message='', session_id=''),
    )
    if style_mode == 'warm':
        return 'All right. That matters, and it goes into my record now. Later answers should reflect it.'
    if style_mode == 'sharp':
        return "Fine. It's on the record now, so future answers start from it."
    if style_mode == 'instructive':
        return 'Good. That belongs in the record now, and later decisions should reflect it.'
    return 'Understood. That goes into my record now, and future answers will move from it.'
