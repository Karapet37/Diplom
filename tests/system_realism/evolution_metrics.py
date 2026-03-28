from __future__ import annotations

from typing import Any


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _category_rows(dialogue_results: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    return [item for item in dialogue_results if str(item.get('category') or '') == category]


def _contains_any(text: str, phrases: list[str]) -> list[str]:
    lowered = str(text or '').lower()
    return [phrase for phrase in phrases if str(phrase or '').lower() in lowered]


def evaluate_evolution_layers(
    *,
    advanced_results: dict[str, Any] | None,
    dialogue_results: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = dict(advanced_results or {})
    scenario_observations = list(payload.get('scenario_observations') or [])
    mutation_summary = dict(payload.get('mutation_summary') or {})
    post_suite_health = dict(payload.get('post_suite_health') or {})

    memory_injection_rows = _category_rows(dialogue_results, 'memory_injection')
    persona_evolution_rows = _category_rows(dialogue_results, 'persona_evolution')
    unexpected_rows = _category_rows(dialogue_results, 'unexpected_rare')
    generalization_rows = _category_rows(dialogue_results, 'unseen_generalization')
    graph_rows = _category_rows(dialogue_results, 'graph_editor')
    contradiction_rows = _category_rows(dialogue_results, 'contradiction_resistance')
    identity_rows = _category_rows(dialogue_results, 'identity_continuity')
    deletion_rows = _category_rows(dialogue_results, 'memory_deletion')
    chaos_rows = _category_rows(dialogue_results, 'chaos')

    adaptation_components: list[float] = []
    if memory_injection_rows:
        adaptation_components.append(_safe_mean([float(item.get('persona_fidelity') or 0.0) for item in memory_injection_rows]))
    if persona_evolution_rows:
        adaptation_components.append(_safe_mean([float(item.get('persona_fidelity') or 0.0) for item in persona_evolution_rows]))
    if graph_rows:
        adaptation_components.append(_safe_mean([float(item.get('decision_authenticity') or 0.0) for item in graph_rows]))
    if unexpected_rows or generalization_rows:
        adaptation_components.append(_safe_mean([float(item.get('style_consistency') or 0.0) for item in unexpected_rows + generalization_rows]))
    adaptation_quality = _safe_mean(adaptation_components)

    memory_components: list[float] = []
    if memory_injection_rows:
        memory_components.append(_safe_mean([float(item.get('persona_fidelity') or 0.0) for item in memory_injection_rows]))
        memory_components.append(_safe_mean([float(item.get('memory_continuity') or 0.0) for item in memory_injection_rows]))
    injected_memory_usage = _safe_mean(memory_components)
    deletion_stale_hits = 0
    deletion_stale_markers: list[str] = []
    for item in deletion_rows:
        hits = _contains_any(
            str(item.get('reply') or ''),
            ['orange trauma shears', 'orange shears', 'mariam', 'daughter mariam'],
        )
        if hits:
            deletion_stale_hits += len(hits)
            deletion_stale_markers.extend(hits)
    graph_stale_hits = 0
    for item in graph_rows:
        if str(item.get('case_id') or '') == 'graph_editor_deleted_tool_probe':
            graph_stale_hits += len(
                _contains_any(
                    str(item.get('reply') or ''),
                    ['portable ultrasound', 'battery', 'cracked probe'],
                )
            )
    memory_usage = max(0.0, min(1.0, injected_memory_usage - min(0.8, 0.25 * deletion_stale_hits)))
    contradiction_components: list[float] = []
    if contradiction_rows:
        contradiction_components.append(_safe_mean([float(item.get('decision_authenticity') or 0.0) for item in contradiction_rows]))
        contradiction_components.append(_safe_mean([1.0 - float(item.get('generic_leakage_badness') or 0.0) for item in contradiction_rows]))
    contradiction_handling = _safe_mean(contradiction_components)
    identity_components: list[float] = []
    if identity_rows or persona_evolution_rows:
        identity_components.append(_safe_mean([float(item.get('persona_fidelity') or 0.0) for item in identity_rows + persona_evolution_rows]))
        identity_components.append(_safe_mean([float(item.get('style_consistency') or 0.0) for item in identity_rows + persona_evolution_rows]))
    identity_continuity = _safe_mean(identity_components)
    mutation_failures = int(mutation_summary.get('setup_failures') or 0) + int(mutation_summary.get('cleanup_failures') or 0)
    mutation_action_count = int(mutation_summary.get('setup_action_count') or 0) + int(mutation_summary.get('cleanup_action_count') or 0)
    mutation_success_rate = 1.0 if mutation_action_count == 0 else max(0.0, 1.0 - (mutation_failures / mutation_action_count))
    system_stability = _safe_mean(
        [
            mutation_success_rate,
            1.0 if post_suite_health.get('ok') else 0.0,
            _safe_mean([1.0 - float(item.get('generic_leakage_badness') or 0.0) for item in chaos_rows]) if chaos_rows else 1.0,
        ]
    )

    scenario_findings: list[dict[str, Any]] = []
    for item in scenario_observations:
        scenario = dict(item.get('scenario') or {})
        setup_records = list(item.get('setup_records') or [])
        cleanup_records = list(item.get('cleanup_records') or [])
        probes = list(item.get('probe_observations') or [])
        flat_replies: list[str] = []
        for probe in probes:
            response = dict(dict(probe).get('response') or {})
            json_body = response.get('json_body')
            assistant_reply = ''
            if isinstance(json_body, dict):
                assistant_reply = str(json_body.get('assistant_reply') or '').strip()
            if not assistant_reply:
                assistant_reply = str(response.get('text') or '').strip()
            flat_replies.append(assistant_reply)
        scenario_findings.append(
            {
                'scenario_id': str(scenario.get('scenario_id') or ''),
                'category': str(scenario.get('category') or ''),
                'setup_ok': all(bool(dict(record).get('ok')) for record in setup_records) if setup_records else True,
                'cleanup_ok': all(bool(dict(record).get('ok')) for record in cleanup_records) if cleanup_records else True,
                'probe_count': len(probes),
                'reply_lengths': [len(reply.split()) for reply in flat_replies],
                'stale_markers': _contains_any(' '.join(flat_replies), ['as an ai', 'portable ultrasound', 'mariam', 'orange trauma shears']),
            }
        )

    suspicious_patterns: list[str] = []
    if deletion_stale_hits:
        suspicious_patterns.append(
            f'Deleted persona memories still leaked into replies ({", ".join(sorted(set(deletion_stale_markers)))}) after revision restore.'
        )
    if graph_stale_hits:
        suspicious_patterns.append('Deleted graph tool details still leaked into later replies after node removal.')
    if mutation_failures:
        suspicious_patterns.append(f'{mutation_failures} mutation action(s) failed during the evolution suite.')
    if chaos_rows and not post_suite_health.get('ok'):
        suspicious_patterns.append('Chaos run completed but runtime health probe failed afterward.')

    recommendations: list[str] = []
    if adaptation_quality < 0.45:
        recommendations.append('Strengthen how new persona facts and local graph edits are surfaced into the bounded prompt after mutations.')
    if memory_usage < 0.45:
        recommendations.append('Audit learned dossier integration and revision restore semantics. The persona is not using or forgetting state changes cleanly.')
    if contradiction_handling < 0.5:
        recommendations.append('Tighten contradiction handling so persona rules survive direct requests to do the opposite.')
    if identity_continuity < 0.5:
        recommendations.append('Preserve baseline anchors more aggressively after learned updates so local life changes do not erase the original persona.')
    if system_stability < 0.7:
        recommendations.append('Reduce mutation-path fragility. Graph/persona edits are leaving the runtime less stable than they should.')

    return {
        'adaptation_quality_score': round(adaptation_quality, 4),
        'memory_usage_score': round(memory_usage, 4),
        'contradiction_handling_score': round(contradiction_handling, 4),
        'identity_continuity_score': round(identity_continuity, 4),
        'system_stability_score': round(system_stability, 4),
        'mutation_success_rate': round(mutation_success_rate, 4),
        'scenario_findings': scenario_findings,
        'mutation_summary': mutation_summary,
        'post_suite_health': post_suite_health,
        'suspicious_patterns': suspicious_patterns,
        'engineering_recommendations': recommendations,
        'metric_breakdowns': {
            'adaptation_quality': {
                'unexpected_case_count': len(unexpected_rows),
                'generalization_case_count': len(generalization_rows),
                'persona_evolution_case_count': len(persona_evolution_rows),
                'graph_editor_case_count': len(graph_rows),
            },
            'memory_usage': {
                'memory_injection_case_count': len(memory_injection_rows),
                'memory_deletion_case_count': len(deletion_rows),
                'deletion_stale_hits': deletion_stale_hits,
                'graph_stale_hits': graph_stale_hits,
            },
            'system_stability': {
                'mutation_action_count': mutation_action_count,
                'mutation_failures': mutation_failures,
                'chaos_case_count': len(chaos_rows),
                'post_suite_health_ok': bool(post_suite_health.get('ok')),
            },
        },
    }
