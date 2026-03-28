from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Protocol

from .dialogue_cases import FALLBACK_PATTERNS
from .evolution_metrics import evaluate_evolution_layers
from .models import DialogueObservation, StartupDiagnosis
from .persona_fixture import CanonicalTestPersona


class RealismJudge(Protocol):
    def evaluate(
        self,
        *,
        persona: CanonicalTestPersona,
        startup: StartupDiagnosis,
        reachability: dict[str, Any],
        dialogue_results: list[dict[str, Any]],
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]: ...

GENERIC_LEAK_MARKERS = [
    'as an ai',
    'as an llm',
    'as a language model',
    'how can i assist',
    'i am here to help',
    'i do not have enough reliable context yet',
    'i will answer in first person from the current persona graph and emotional state',
    'i cannot browse',
    'i cannot provide',
    'i do not have a body',
    'i do not have personal experiences',
    'i cannot have children',
    'helpful assistant',
]

STOPWORDS = {
    'a',
    'an',
    'and',
    'are',
    'as',
    'at',
    'be',
    'but',
    'by',
    'do',
    'for',
    'from',
    'i',
    'if',
    'in',
    'is',
    'it',
    'me',
    'my',
    'of',
    'on',
    'or',
    'the',
    'to',
    'we',
    'you',
}


def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text or '').strip().lower())


def _tokens(text: str) -> set[str]:
    normalized = re.sub(r'[^a-z0-9\s-]+', ' ', _normalize(text))
    return {token for token in normalized.split() if token and token not in STOPWORDS}


def _contains_phrase(text: str, phrase: str) -> bool:
    return _normalize(phrase) in _normalize(text)


def _hit_count(text: str, clues: list[str]) -> tuple[int, list[str]]:
    hits: list[str] = []
    normalized = _normalize(text)
    token_set = _tokens(text)
    for clue in clues:
        clean = _normalize(clue)
        if not clean:
            continue
        if ' ' in clean:
            if clean in normalized:
                hits.append(clue)
        elif clean in token_set:
            hits.append(clue)
    return len(hits), hits


def _word_count(text: str) -> int:
    return len(re.findall(r'\w+', str(text or '')))


def _extract_numbered_item(text: str, number: int) -> str:
    pattern = re.compile(rf'(?m)^\s*{number}\s*[\.\):-]\s*(.+)$')
    match = pattern.search(str(text or ''))
    return match.group(1).strip() if match else ''


def _lexical_overlap(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), 1)


def _style_shape_score(text: str) -> float:
    words = _word_count(text)
    if words == 0:
        return 0.0
    if 18 <= words <= 120:
        return 1.0
    if 10 <= words <= 160:
        return 0.7
    return 0.35


def _fallback_detected(text: str) -> bool:
    return any(_contains_phrase(text, pattern) for pattern in FALLBACK_PATTERNS)


def _safe_mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _reply_text_from_observation(observation: DialogueObservation) -> str:
    response = observation.response
    if isinstance(response.json_body, dict):
        candidate = str(response.json_body.get('assistant_reply') or '')
        if candidate.strip():
            return candidate
    return str(response.text or '')


def _generic_leakage_badness(text: str) -> tuple[float, list[str]]:
    hits = [marker for marker in GENERIC_LEAK_MARKERS if _contains_phrase(text, marker)]
    if not text.strip():
        return 1.0, ['empty_reply']
    badness = min(1.0, 0.25 * len(hits))
    if _word_count(text) > 220:
        hits.append('overexplaining')
        badness = min(1.0, badness + 0.2)
    return badness, hits


def _metric_band(score: float, *, reverse: bool = False) -> str:
    value = max(0.0, min(1.0, float(score or 0.0)))
    if reverse:
        if value <= 0.2:
            return 'low'
        if value <= 0.45:
            return 'moderate'
        if value <= 0.7:
            return 'elevated'
        return 'high'
    if value >= 0.75:
        return 'strong'
    if value >= 0.5:
        return 'acceptable'
    if value >= 0.3:
        return 'weak'
    return 'poor'


def _infrastructure_explanation(*, startup_success: bool, api_reachable: bool, frontend_reachable: bool) -> str:
    if not startup_success:
        return 'Infrastructure score is failed because the backend did not complete startup through the real runtime entrypoint.'
    if not api_reachable:
        return 'Infrastructure score is degraded because the process started but the live chat API did not complete a valid request.'
    if not frontend_reachable:
        return 'Infrastructure score is partially degraded because the backend is alive but the root operator surface is not serving HTML.'
    return 'Infrastructure score is healthy because startup, live chat reachability, and root surface reachability all succeeded.'


def _latency_explanation(*, average_latency_ms: float, max_latency_ms: float, timeout_count: int) -> str:
    if timeout_count:
        return (
            f'Latency score is degraded because {timeout_count} request(s) timed out; '
            f'average latency was {average_latency_ms:.1f} ms and max latency was {max_latency_ms:.1f} ms.'
        )
    return f'Latency summary is based on live response timings: average {average_latency_ms:.1f} ms, max {max_latency_ms:.1f} ms.'


def _case_note(text: str, hits: list[str], misses: list[str]) -> str:
    parts: list[str] = [text]
    if hits:
        parts.append('hits=' + ', '.join(hits[:5]))
    if misses:
        parts.append('issues=' + ', '.join(misses[:5]))
    return '; '.join(parts)


def _run_optional_judge(
    judge: RealismJudge | None,
    *,
    persona: CanonicalTestPersona,
    startup: StartupDiagnosis,
    reachability: dict[str, Any],
    dialogue_results: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        'enabled': judge is not None,
        'used': False,
        'result': {},
        'error': '',
    }
    if judge is None:
        return payload
    try:
        payload['result'] = dict(
            judge.evaluate(
                persona=persona,
                startup=startup,
                reachability=reachability,
                dialogue_results=dialogue_results,
                diagnostics=diagnostics,
            )
            or {}
        )
        payload['used'] = True
    except Exception as exc:  # noqa: BLE001
        payload['error'] = str(exc)
    return payload


def _trait_expectation_clues(persona: CanonicalTestPersona, traits: list[str]) -> list[str]:
    mapping = {
        'identity': ['emergency', 'physician', 'triage', 'Yerevan'],
        'biography': ['Lori', 'rural', 'clinic'],
        'work_habits': ['notebook', 'shift', 'write', 'near-miss'],
        'trust_model': ['evidence', 'facts', 'signal', 'noise'],
        'skepticism': ['skeptical', 'facts', 'evidence'],
        'decision_process': ['risk', 'clarify', 'reversible', 'triage', 'stabilize'],
        'boundary_behavior': ['do not', 'won’t', 'clarify', 'lie', 'certainty'],
        'speech_style': ['dry', 'concise', 'boring truth'],
        'irritants': ['flattery', 'theater', 'noise'],
        'emotional_control': ['calm', 'firm', 'steady'],
        'practical_reasoning': ['first', 'then', 'because'],
        'memory_anchor': ['watch', 'father', 'notebook', 'Anahit'],
        'personal_history': ['father', 'watch', 'sister', 'clinic'],
        'memory_continuity': ['earlier', 'rule', 'same'],
        'values': ['truth', 'evidence', 'protect', 'lie'],
    }
    clues = list(persona.style_markers())
    for trait in traits:
        clues.extend(mapping.get(trait, []))
    return list(dict.fromkeys(clues))


def evaluate_realism(
    *,
    startup: StartupDiagnosis,
    reachability: dict[str, Any],
    persona: CanonicalTestPersona,
    persona_materialization: dict[str, Any],
    persona_endpoint: dict[str, Any] | None,
    dialogue_observations: list[DialogueObservation],
    diagnostics: dict[str, Any],
    advanced_results: dict[str, Any] | None = None,
    judge: RealismJudge | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    latency_values: list[float] = []
    timeout_count = 0
    contradiction_count = 0
    exploratory_failure_count = 0
    suspicious_patterns: list[str] = []
    major_failures: list[str] = []

    observations_by_case = {item.case.case_id: item for item in dialogue_observations}
    consistency_buckets: dict[str, list[str]] = defaultdict(list)

    for observation in dialogue_observations:
        case = observation.case
        response = observation.response
        reply_text = _reply_text_from_observation(observation)

        latency_values.append(float(response.latency_ms or 0.0))
        if response.status_code == 0 or 'timed out' in str(response.error or '').lower():
            timeout_count += 1

        trait_clues = _trait_expectation_clues(persona, case.expected_traits)
        trait_hit_count, trait_hits = _hit_count(reply_text, trait_clues)
        style_hit_count, style_hits = _hit_count(reply_text, case.target_style_clues + persona.style_markers())
        forbidden_hits = [pattern for pattern in case.forbidden_failure_patterns if _contains_phrase(reply_text, pattern)]
        leakage_badness, leakage_hits = _generic_leakage_badness(reply_text)
        anchor_hits_count, anchor_hits = _hit_count(
            reply_text,
            persona.anchor_keywords()['identity'] + persona.anchor_keywords()['memory'] + persona.anchor_keywords()['values'],
        )
        persona_signal_hits = [signal for signal in case.persona_success_signals if _contains_phrase(reply_text, signal)]
        generic_failure_hits = [
            signal for signal in case.generic_llm_failure_signals if _contains_phrase(reply_text, signal)
        ]

        fidelity_anchor_component = 0.4 * min(anchor_hits_count / 3.0, 1.0)
        fidelity_trait_component = 0.35 * min(trait_hit_count / max(len(case.expected_traits), 1), 1.0)
        fidelity_style_component = 0.15 * min(style_hit_count / max(len(case.target_style_clues) or 1, 1), 1.0)
        fidelity_boundary_component = 0.1 * (0.0 if forbidden_hits else 1.0)
        fidelity = min(
            1.0,
            fidelity_anchor_component + fidelity_trait_component + fidelity_style_component + fidelity_boundary_component,
        )
        style_shape_component = 0.55 * _style_shape_score(reply_text)
        style_marker_component = 0.45 * min(style_hit_count / 3.0, 1.0)
        style_score = max(0.0, min(1.0, style_shape_component + style_marker_component))
        decision_marker_hits, _ = _hit_count(reply_text, persona.anchor_keywords()['values'] + persona.anchor_keywords()['boundaries'])
        decision_values_component = 0.65 * min(decision_marker_hits / 3.0, 1.0)
        decision_scenario_component = 0.35 * (
            1.0 if any(marker in case.category for marker in ('decision', 'adversarial')) and trait_hit_count else 0.6
        )
        decision_score = min(1.0, decision_values_component + decision_scenario_component)
        memory_score = 0.0
        memory_expected_fragment = ''
        if case.expects_memory_from:
            prior = observations_by_case.get(case.expects_memory_from)
            prior_reply = _reply_text_from_observation(prior) if prior is not None else ''
            memory_expected_fragment = _extract_numbered_item(prior_reply, 2)
            memory_score = _lexical_overlap(memory_expected_fragment, reply_text) if memory_expected_fragment else 0.0
        elif case.category == 'memory_continuity':
            memory_score = min(anchor_hits_count / 2.0, 1.0)

        if case.consistency_group:
            consistency_buckets[case.consistency_group].append(reply_text)
        if forbidden_hits:
            suspicious_patterns.append(f"{case.case_id}: forbidden pattern -> {', '.join(forbidden_hits)}")
        if leakage_hits:
            suspicious_patterns.append(f"{case.case_id}: leakage -> {', '.join(leakage_hits)}")
        if case.category == 'exploratory' and (forbidden_hits or leakage_hits or fidelity < 0.3):
            exploratory_failure_count += 1

        results.append(
            {
                'case_id': case.case_id,
                'category': case.category,
                'prompt': case.prompt,
                'trait_probe': case.trait_probe,
                'status_code': response.status_code,
                'latency_ms': round(float(response.latency_ms or 0.0), 3),
                'reply': reply_text,
                'trait_hits': trait_hits,
                'style_hits': style_hits,
                'anchor_hits': anchor_hits,
                'forbidden_hits': forbidden_hits,
                'leakage_hits': leakage_hits,
                'generic_llm_failure_hits': generic_failure_hits,
                'persona_success_signal_hits': persona_signal_hits,
                'persona_fidelity': round(fidelity, 4),
                'style_consistency': round(style_score, 4),
                'memory_continuity': round(memory_score, 4),
                'decision_authenticity': round(decision_score, 4),
                'generic_leakage_badness': round(leakage_badness, 4),
                'score_breakdown': {
                    'persona_fidelity': {
                        'anchor_component': round(fidelity_anchor_component, 4),
                        'trait_component': round(fidelity_trait_component, 4),
                        'style_component': round(fidelity_style_component, 4),
                        'forbidden_component': round(fidelity_boundary_component, 4),
                    },
                    'style_consistency': {
                        'shape_component': round(style_shape_component, 4),
                        'marker_component': round(style_marker_component, 4),
                    },
                    'memory_continuity': {
                        'expected_fragment': memory_expected_fragment,
                        'score': round(memory_score, 4),
                    },
                    'decision_authenticity': {
                        'values_component': round(decision_values_component, 4),
                        'scenario_component': round(decision_scenario_component, 4),
                    },
                    'generic_llm_leakage': {
                        'marker_hits': leakage_hits,
                        'badness': round(leakage_badness, 4),
                    },
                },
                'evaluation_notes': [
                    _case_note('Trait/anchor alignment for this probe.', trait_hits + anchor_hits, forbidden_hits),
                    _case_note('Style signals observed in reply.', style_hits + persona_signal_hits, generic_failure_hits + leakage_hits),
                ],
            }
        )

    contradiction_details: list[dict[str, Any]] = []
    for group, replies in consistency_buckets.items():
        if len(replies) < 2:
            continue
        overlaps: list[float] = []
        for left, right in zip(replies, replies[1:]):
            overlaps.append(_lexical_overlap(left, right))
        average_overlap = _safe_mean(overlaps)
        contradiction_details.append(
            {
                'group': group,
                'reply_count': len(replies),
                'average_overlap': round(average_overlap, 4),
                'band': 'stable' if average_overlap >= 0.12 else 'drifting',
            }
        )
        if average_overlap < 0.12:
            contradiction_count += 1
            suspicious_patterns.append(f'low consistency overlap in group={group} ({average_overlap:.2f})')

    persona_fidelity_score = _safe_mean([item['persona_fidelity'] for item in results])
    style_consistency_score = _safe_mean([item['style_consistency'] for item in results])
    memory_continuity_cases = [item['memory_continuity'] for item in results if item['memory_continuity'] > 0.0]
    memory_continuity_score = _safe_mean(memory_continuity_cases)
    decision_authenticity_score = _safe_mean([item['decision_authenticity'] for item in results])
    generic_llm_leakage_score = _safe_mean([item['generic_leakage_badness'] for item in results])
    exploratory_results = [item for item in results if item['category'] == 'exploratory']

    average_latency_ms = _safe_mean(latency_values)
    max_latency_ms = max(latency_values) if latency_values else 0.0
    frontend_reachable = bool(reachability.get('root_html'))
    api_reachable = bool(reachability.get('chat_alive')) and startup.startup_success

    infrastructure_status = 'alive'
    if not startup.startup_success:
        infrastructure_status = 'startup_failed'
        major_failures.append('Backend did not start through the real runtime entrypoint.')
    elif not api_reachable:
        infrastructure_status = 'api_unreachable'
        major_failures.append('Chat API did not complete a valid live request.')
    elif not frontend_reachable:
        infrastructure_status = 'frontend_degraded'
        suspicious_patterns.append('Root page did not return an HTML operator surface.')

    if not persona_materialization.get('ok'):
        major_failures.append('Canonical test persona was not fully materialized in expected storage format.')
    if results and persona_fidelity_score < 0.35:
        major_failures.append('Persona fidelity stayed too low: the runtime sounds unlike the intended persona.')
    if results and generic_llm_leakage_score > 0.65:
        major_failures.append('Generic assistant leakage is too high.')
    if results and memory_continuity_score < 0.25:
        suspicious_patterns.append('Session continuity is weak or missing on explicit follow-up prompts.')
    if exploratory_results and exploratory_failure_count:
        suspicious_patterns.append(
            f'Exploratory prompts exposed {exploratory_failure_count} persona-collapse case(s) outside the fixed benchmark.'
        )
    if exploratory_results and exploratory_failure_count >= max(1, len(exploratory_results) // 2):
        major_failures.append('Persona collapses too often on exploratory prompts outside the scripted benchmark.')
    if average_latency_ms > 5000:
        suspicious_patterns.append(f'Average latency is high ({average_latency_ms:.1f} ms).')
    if diagnostics.get('runtime_status', {}).get('mode') == 'degraded':
        suspicious_patterns.append('Runtime reports degraded mode during the realism run.')

    if infrastructure_status == 'startup_failed':
        overall_verdict = 'startup_failed'
    elif not api_reachable:
        overall_verdict = 'api_unreachable'
    elif persona_fidelity_score >= 0.72 and style_consistency_score >= 0.68 and memory_continuity_score >= 0.45 and generic_llm_leakage_score <= 0.25:
        overall_verdict = 'persona_alive_and_believable'
    elif persona_fidelity_score >= 0.48 and generic_llm_leakage_score <= 0.5:
        overall_verdict = 'persona_partially_alive_but_uneven'
    else:
        overall_verdict = 'generic_assistant_drift_or_degraded_runtime'

    recommendations: list[str] = []
    if not startup.startup_success:
        recommendations.append('Inspect startup logs first. The realism harness could not reach a live backend through start.py.')
    if diagnostics.get('runtime_status', {}).get('mode') == 'degraded':
        recommendations.append('Resolve degraded runtime status before tuning prompts. Missing local roles or inference bindings invalidate persona realism.')
    if results and persona_fidelity_score < 0.5:
        recommendations.append('Increase persona-specific anchors in active context and verify that persona block ranking stays above graph-only generic facts.')
    if results and generic_llm_leakage_score > 0.5:
        recommendations.append('Tighten fallback triggers and persona-response shaping. The system still leaks generic assistant phrasing.')
    if results and memory_continuity_score < 0.4:
        recommendations.append('Inspect session-memory selection and recency scoring. Cross-turn recall is not consistently visible in replies.')
    if exploratory_results and exploratory_failure_count:
        recommendations.append('Expand persona-shaping checks with less-scripted prompts. The persona still collapses on exploratory or counterfactual questions.')
    if results and average_latency_ms > 4000:
        recommendations.append('Reduce hot-path context load or model startup overhead. Latency is too high for an operator-facing local workflow.')
    if startup.startup_success and not frontend_reachable:
        recommendations.append('Rebuild or remount the frontend so the operator gets a full product surface instead of a degraded root response.')
    if not recommendations:
        recommendations.append('Keep watching degraded-mode diagnostics and rerun realism after graph/persona changes. The system currently behaves like a living product.')

    dialogue_summary = {
        'case_count': len(results),
        'successful_response_count': sum(1 for item in results if item['status_code'] == 200),
        'fallback_like_count': sum(1 for item in results if _fallback_detected(item['reply'])),
        'exploratory_case_count': len(exploratory_results),
        'exploratory_failure_count': exploratory_failure_count,
    }
    score_explanations = {
        'infrastructure_status': _infrastructure_explanation(
            startup_success=startup.startup_success,
            api_reachable=api_reachable,
            frontend_reachable=frontend_reachable,
        ),
        'latency': _latency_explanation(
            average_latency_ms=average_latency_ms,
            max_latency_ms=max_latency_ms,
            timeout_count=timeout_count,
        ),
        'persona_fidelity': (
            f"Persona fidelity is `{_metric_band(persona_fidelity_score)}` because the evaluator averages biography anchors, "
            f"trait clues, style markers, and forbidden-pattern avoidance across {len(results)} live replies."
        ),
        'style_consistency': (
            f"Style consistency is `{_metric_band(style_consistency_score)}` because it rewards compact response shape and "
            'presence of expected persona-specific verbal markers.'
        ),
        'memory_continuity': (
            f"Memory continuity is `{_metric_band(memory_continuity_score)}` because explicit recall cases are scored by overlap "
            'with prior answers rather than by vague similarity.'
        ),
        'contradiction_detection': (
            f'Contradiction detection counted {contradiction_count} low-overlap consistency group(s); lower contradiction count means replies stayed semantically aligned.'
        ),
        'generic_llm_leakage': (
            f"Generic leakage is `{_metric_band(generic_llm_leakage_score, reverse=True)}` because the evaluator looks for assistant phrases, "
            'empty replies, and over-explaining without persona anchors.'
        ),
        'decision_authenticity': (
            f"Decision authenticity is `{_metric_band(decision_authenticity_score)}` because the evaluator checks for values, boundaries, "
            'and scenario-specific practical reasoning instead of generic moralizing.'
        ),
        'exploratory_resilience': (
            f'Exploratory resilience recorded {exploratory_failure_count} failure case(s) across {len(exploratory_results)} less-scripted prompts.'
        ),
    }
    metric_breakdowns = {
        'infrastructure_status': {
            'startup_success': bool(startup.startup_success),
            'api_reachable': bool(api_reachable),
            'frontend_reachable': bool(frontend_reachable),
            'status': infrastructure_status,
        },
        'latency': {
            'average_latency_ms': round(average_latency_ms, 3),
            'max_latency_ms': round(max_latency_ms, 3),
            'timeout_count': timeout_count,
            'sample_count': len(latency_values),
        },
        'persona_fidelity': {
            'score': round(persona_fidelity_score, 4),
            'band': _metric_band(persona_fidelity_score),
            'average_anchor_hits': round(_safe_mean([len(item['anchor_hits']) for item in results]), 4),
            'average_trait_hits': round(_safe_mean([len(item['trait_hits']) for item in results]), 4),
            'average_style_hits': round(_safe_mean([len(item['style_hits']) for item in results]), 4),
        },
        'style_consistency': {
            'score': round(style_consistency_score, 4),
            'band': _metric_band(style_consistency_score),
            'average_style_hits': round(_safe_mean([len(item['style_hits']) for item in results]), 4),
        },
        'memory_continuity': {
            'score': round(memory_continuity_score, 4),
            'band': _metric_band(memory_continuity_score),
            'case_count': len(memory_continuity_cases),
        },
        'contradiction_detection': {
            'contradiction_count': contradiction_count,
            'consistency_groups': contradiction_details,
        },
        'generic_llm_leakage': {
            'score': round(generic_llm_leakage_score, 4),
            'band': _metric_band(generic_llm_leakage_score, reverse=True),
            'average_leak_markers': round(_safe_mean([len(item['leakage_hits']) for item in results]), 4),
            'fallback_like_count': dialogue_summary['fallback_like_count'],
        },
        'decision_authenticity': {
            'score': round(decision_authenticity_score, 4),
            'band': _metric_band(decision_authenticity_score),
            'average_value_hits': round(_safe_mean([len(item['trait_hits']) for item in results]), 4),
        },
        'exploratory_resilience': {
            'case_count': len(exploratory_results),
            'failure_count': exploratory_failure_count,
            'failure_rate': round(exploratory_failure_count / max(len(exploratory_results), 1), 4) if exploratory_results else 0.0,
        },
    }
    judge_evaluation = _run_optional_judge(
        judge,
        persona=persona,
        startup=startup,
        reachability=reachability,
        dialogue_results=results,
        diagnostics=diagnostics,
    )
    advanced_enabled = bool(advanced_results) or any(
        str(item.get('category') or '') in {
            'unexpected_rare',
            'unseen_generalization',
            'memory_injection',
            'persona_evolution',
            'memory_deletion',
            'graph_editor',
            'contradiction_resistance',
            'identity_continuity',
            'chaos',
        }
        for item in results
    )
    evolution_metrics = {'enabled': False}
    if advanced_enabled:
        evolution_metrics = evaluate_evolution_layers(
            advanced_results=advanced_results,
            dialogue_results=results,
        )
        evolution_metrics['enabled'] = True
        suspicious_patterns.extend(list(evolution_metrics.get('suspicious_patterns') or []))
        recommendations.extend(list(evolution_metrics.get('engineering_recommendations') or []))
        metric_breakdowns['adaptation_quality'] = dict(dict(evolution_metrics.get('metric_breakdowns') or {}).get('adaptation_quality') or {})
        metric_breakdowns['memory_usage'] = dict(dict(evolution_metrics.get('metric_breakdowns') or {}).get('memory_usage') or {})
        metric_breakdowns['system_stability'] = dict(dict(evolution_metrics.get('metric_breakdowns') or {}).get('system_stability') or {})
        score_explanations['adaptation_quality'] = (
            f"Adaptation quality is `{_metric_band(float(evolution_metrics.get('adaptation_quality_score') or 0.0))}` because it averages how well the persona absorbs unseen prompts, memory injections, local evolution, and graph edits."
        )
        score_explanations['memory_usage'] = (
            f"Memory usage is `{_metric_band(float(evolution_metrics.get('memory_usage_score') or 0.0))}` because injected facts must later appear in reasoning and disappear again after deletion or restore."
        )
        score_explanations['identity_continuity'] = (
            f"Identity continuity is `{_metric_band(float(evolution_metrics.get('identity_continuity_score') or 0.0))}` because the persona must stay recognizably the same person after mutations."
        )
        score_explanations['system_stability'] = (
            f"System stability is `{_metric_band(float(evolution_metrics.get('system_stability_score') or 0.0))}` because mutation paths, cleanup, and post-suite health checks all stay in the loop."
        )
        if float(evolution_metrics.get('adaptation_quality_score') or 0.0) < 0.35:
            major_failures.append('Adaptive scenarios show weak transfer: the persona does not generalize well after live mutations.')
        if float(evolution_metrics.get('memory_usage_score') or 0.0) < 0.35:
            major_failures.append('Memory lifecycle is weak: injected or deleted persona state is not reflected cleanly in later replies.')
        if float(evolution_metrics.get('identity_continuity_score') or 0.0) < 0.35:
            major_failures.append('Identity continuity degraded after local mutations.')
        if float(evolution_metrics.get('system_stability_score') or 0.0) < 0.55:
            major_failures.append('Mutation paths or post-mutation runtime health are too unstable.')

        if overall_verdict == 'persona_alive_and_believable':
            if (
                float(evolution_metrics.get('adaptation_quality_score') or 0.0) < 0.6
                or float(evolution_metrics.get('identity_continuity_score') or 0.0) < 0.6
                or float(evolution_metrics.get('memory_usage_score') or 0.0) < 0.5
            ):
                overall_verdict = 'persona_partially_alive_but_uneven'
        elif overall_verdict == 'persona_partially_alive_but_uneven':
            if (
                float(evolution_metrics.get('adaptation_quality_score') or 0.0) < 0.35
                or float(evolution_metrics.get('system_stability_score') or 0.0) < 0.5
            ):
                overall_verdict = 'generic_assistant_drift_or_degraded_runtime'

    return {
        'infrastructure_status': infrastructure_status,
        'startup_success': startup.startup_success,
        'frontend_reachable': frontend_reachable,
        'api_reachable': api_reachable,
        'average_latency_ms': round(average_latency_ms, 3),
        'max_latency_ms': round(max_latency_ms, 3),
        'timeout_count': timeout_count,
        'persona_fidelity_score': round(persona_fidelity_score, 4),
        'style_consistency_score': round(style_consistency_score, 4),
        'memory_continuity_score': round(memory_continuity_score, 4),
        'decision_authenticity_score': round(decision_authenticity_score, 4),
        'generic_llm_leakage_score': round(generic_llm_leakage_score, 4),
        'contradiction_count': contradiction_count,
        'overall_verdict': overall_verdict,
        'major_failures': list(dict.fromkeys(major_failures)),
        'suspicious_patterns': list(dict.fromkeys(suspicious_patterns)),
        'engineering_recommendations': list(dict.fromkeys(recommendations)),
        'dialogue_results': results,
        'dialogue_summary': dialogue_summary,
        'advanced_metrics': evolution_metrics,
        'persona_endpoint_visible': bool(persona_endpoint),
        'persona_materialization_ok': bool(persona_materialization.get('ok')),
        'diagnostic_counters': dict(diagnostics.get('metrics', {}).get('counters') or {}),
        'graph_health': dict(diagnostics.get('graph_health') or {}),
        'runtime_status': dict(diagnostics.get('runtime_status') or {}),
        'score_explanations': score_explanations,
        'metric_breakdowns': metric_breakdowns,
        'judge_evaluation': judge_evaluation,
    }
