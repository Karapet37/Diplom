from __future__ import annotations

from typing import Any

from .duplicate_resolver import normalize_name
from .models import HeadBundle, MessageAnalysis, MoodResearchReport, SocialRoleDecision, Situation

_ROLE_GUIDANCE = {
    'ally': 'Back the user in a frank, grounded way. Stay on their side without becoming servile.',
    'friend': 'Be warm and recognizably human. Use ease and familiarity instead of service politeness.',
    'critic': 'Interrogate weak reasoning, vagueness, or self-deception directly. Stay concrete, not bureaucratic.',
    'rival': 'Answer competitively and challenge the user to sharpen their thinking or resolve.',
    'provocateur': 'Use strategic friction to expose contradictions, avoidance, or complacency.',
    'comforter': 'Stay emotionally present, steady, and protective without drifting into therapy-speak.',
    'mentor': 'Be practical, experience-shaped, and instructive. Teach without sounding like a helpdesk.',
}


def _tokenized(values: list[str]) -> str:
    return normalize_name(' '.join(str(item).strip() for item in list(values or []) if str(item).strip()))


def _list_from_form(bundle: HeadBundle, key: str) -> list[str]:
    return [str(item).strip() for item in list(bundle.persona_form.get(key) or []) if str(item).strip()]


def _trait_text(bundle: HeadBundle) -> str:
    return normalize_name(
        ' '.join(
            list(bundle.traits)
            + _list_from_form(bundle, 'interaction_style')
            + _list_from_form(bundle, 'core_dispositions')
            + _list_from_form(bundle, 'speech_tendencies')
            + _list_from_form(bundle, 'conflict_behavior')
            + _list_from_form(bundle, 'reaction_patterns')
        )
    )


def _preferred_roles(bundle: HeadBundle) -> list[str]:
    direct = _list_from_form(bundle, 'social_roles')
    if direct:
        return direct[:8]
    traits = _trait_text(bundle)
    inferred: list[str] = ['ally']
    if any(token in traits for token in ('empathetic', 'warm', 'protective', 'steady')):
        inferred.extend(['comforter', 'friend', 'mentor'])
    if any(token in traits for token in ('logical', 'analytical', 'skeptical', 'precise')):
        inferred.extend(['critic', 'mentor'])
    if any(token in traits for token in ('aggressive', 'predatory', 'provocative', 'ironic', 'sarcastic')):
        inferred.extend(['rival', 'provocateur'])
    if any(token in traits for token in ('formal', 'aristocratic', 'disciplined')):
        inferred.append('critic')
    return list(dict.fromkeys(inferred))


def _bundle_anchor_phrases(bundle: HeadBundle) -> list[str]:
    anchors = []
    for key in (
        'values',
        'conflicts',
        'topic_affinities',
        'speech_tendencies',
        'reaction_patterns',
        'memory_anchors',
        'memories',
    ):
        anchors.extend(_list_from_form(bundle, key))
    return list(dict.fromkeys(anchors))


def _score_role(
    role: str,
    *,
    bundle: HeadBundle,
    analysis: MessageAnalysis,
    situation: Situation,
    mood_report: MoodResearchReport | None,
) -> tuple[float, list[str], dict[str, float]]:
    signals = dict(analysis.user_state.signals)
    empathy = float(bundle.emotion_vector.get('empathy') or 0.0)
    confidence = float(bundle.emotion_vector.get('confidence') or 0.0)
    aggression = 1.0 - empathy + float(bundle.emotion_vector.get('anger') or 0.0)
    curiosity = float(bundle.emotion_vector.get('curiosity') or 0.0)
    fear = float(bundle.emotion_vector.get('fear') or 0.0)
    traits = _trait_text(bundle)
    reasons: list[str] = []
    score = 0.0

    if role in _preferred_roles(bundle):
        score += 0.25
        reasons.append('listed or inferred as a plausible social mode for this persona')

    if role == 'comforter':
        score += 0.9 * float(signals.get('contains_distress') or 0.0)
        score += 0.6 * float(signals.get('contains_help_request') or 0.0)
        score += 0.4 * empathy
        score -= 0.4 * float(signals.get('contains_insult') or 0.0)
        if empathy >= 0.45:
            reasons.append('persona carries enough empathy to stay emotionally present')
    elif role == 'mentor':
        score += 0.55 * float(signals.get('contains_question') or 0.0)
        score += 0.35 * curiosity
        score += 0.2 * confidence
        if any(token in traits for token in ('logical', 'analytical', 'teacher', 'mentor', 'precise', 'disciplined')):
            score += 0.25
            reasons.append('persona leans toward instructive or disciplined explanation')
    elif role == 'ally':
        score += 0.35 * float(signals.get('contains_question') or 0.0)
        score += 0.25 * float(signals.get('contains_celebratory') or 0.0)
        score += 0.35 * empathy
        if any(token in traits for token in ('loyal', 'supportive', 'empathetic', 'warm')):
            score += 0.2
            reasons.append('persona disposition favors cooperative alignment')
    elif role == 'friend':
        score += 0.45 * float(signals.get('contains_celebratory') or 0.0)
        score += 0.25 * empathy
        score += 0.15 * float(signals.get('contains_self_reference') or 0.0)
        if any(token in traits for token in ('warm', 'friendly', 'witty', 'humorous')):
            score += 0.2
            reasons.append('persona can sustain informal warmth without dissolving into service talk')
    elif role == 'critic':
        score += 0.55 * float(signals.get('contains_moral_violation') or 0.0)
        score += 0.45 * float(signals.get('contains_question') or 0.0)
        score += 0.25 * confidence
        if any(token in traits for token in ('logical', 'skeptical', 'formal', 'exacting', 'aristocratic')):
            score += 0.3
            reasons.append('persona has a corrective or exacting streak')
    elif role == 'rival':
        score += 0.75 * float(signals.get('contains_insult') or 0.0)
        score += 0.4 * float(signals.get('contains_anger') or 0.0)
        score += 0.3 * aggression
        score += 0.15 * confidence
        if any(token in traits for token in ('aggressive', 'competitive', 'predatory', 'sarcastic')):
            score += 0.25
            reasons.append('persona can meet hostility with open challenge')
    elif role == 'provocateur':
        score += 0.4 * float(signals.get('contains_question') or 0.0)
        score += 0.3 * float(signals.get('contains_insult') or 0.0)
        score += 0.25 * aggression
        score += 0.2 * curiosity
        if any(token in traits for token in ('sarcastic', 'ironic', 'provocative', 'chaotic', 'predatory')):
            score += 0.3
            reasons.append('persona can use friction to force clarity')

    if situation.type == 'user_distress' and role in {'comforter', 'ally', 'mentor'}:
        score += 0.18
        reasons.append('current situation contains user distress')
    if situation.type in {'insult', 'user_anger'} and role in {'rival', 'critic', 'provocateur'}:
        score += 0.18
        reasons.append('current situation contains antagonism')
    if situation.type == 'neutral_query' and role in {'mentor', 'ally', 'critic'}:
        score += 0.12
        reasons.append('current situation is a grounded query rather than a plea for service')
    if fear >= 0.55 and role in {'provocateur', 'rival'}:
        score -= 0.12
    if mood_report is not None:
        latest_label = normalize_name(mood_report.latest_cluster_label)
        if role in {'comforter', 'ally'} and 'distress' in latest_label:
            score += 0.08
            reasons.append('recent mood research shows a distress-heavy interaction pattern')
        if role in {'critic', 'rival', 'provocateur'} and 'antagon' in latest_label:
            score += 0.08
            reasons.append('recent mood research shows antagonistic pressure')
        for item in list(mood_report.role_effects or [])[:8]:
            if str(item.get('role') or '') == role:
                score += min(max(float(item.get('fit_score') or 0.0), 0.0), 1.0) * 0.1
                break

    signal_map = {
        'distress': float(signals.get('contains_distress') or 0.0),
        'help_request': float(signals.get('contains_help_request') or 0.0),
        'anger': float(signals.get('contains_anger') or 0.0),
        'insult': float(signals.get('contains_insult') or 0.0),
        'curiosity': float(signals.get('contains_question') or 0.0),
        'celebratory': float(signals.get('contains_celebratory') or 0.0),
        'persona_empathy': empathy,
        'persona_confidence': confidence,
    }
    return score, list(dict.fromkeys(reasons))[:5], signal_map


def choose_social_role(
    *,
    bundle: HeadBundle | None,
    analysis: MessageAnalysis,
    situation: Situation,
    mood_report: MoodResearchReport | None = None,
) -> SocialRoleDecision:
    if bundle is None:
        return SocialRoleDecision(
            role='ally',
            confidence=0.25,
            reason='No active persona head was available, so the system stayed with a minimal cooperative role.',
            evidence=['no persona bundle'],
        )

    roles = ('ally', 'friend', 'critic', 'rival', 'provocateur', 'comforter', 'mentor')
    ranked: list[tuple[str, float, list[str], dict[str, float]]] = []
    for role in roles:
        score, reasons, signals = _score_role(role, bundle=bundle, analysis=analysis, situation=situation, mood_report=mood_report)
        ranked.append((role, score, reasons, signals))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    role, score, reasons, signal_map = ranked[0]
    anchors = [anchor for anchor in _bundle_anchor_phrases(bundle) if anchor][:4]
    if not reasons:
        reasons = ['the persona graph did not strongly force another role, so the system kept the least assistant-like cooperative stance']
    return SocialRoleDecision(
        role=role,
        confidence=min(max(score, 0.0), 1.0),
        reason=reasons[0],
        evidence=reasons,
        graph_anchors=anchors,
        mood_signals=signal_map,
    )


def role_prompt_guidance(decision: SocialRoleDecision) -> str:
    return _ROLE_GUIDANCE.get(decision.role, _ROLE_GUIDANCE['ally'])


def render_social_role_block(decision: SocialRoleDecision, *, mood_report: MoodResearchReport | None = None) -> str:
    lines = [
        f'Selected social role: {decision.role}.',
        f'Role reason: {decision.reason}',
    ]
    if decision.evidence:
        lines.append(f"Role evidence: {' | '.join(decision.evidence[:4])}.")
    if decision.graph_anchors:
        lines.append(f"Persona graph anchors: {' | '.join(decision.graph_anchors[:4])}.")
    if mood_report is not None and mood_report.latest_cluster_label:
        lines.append(f'Recent mood research pattern: {mood_report.latest_cluster_label}.')
    guidance = role_prompt_guidance(decision)
    if guidance:
        lines.append(f'Behavior guidance: {guidance}')
    return '\n'.join(item for item in lines if str(item).strip()).strip()
