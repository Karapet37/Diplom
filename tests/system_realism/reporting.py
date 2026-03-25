from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')


def _iso_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _bool_icon(value: Any) -> str:
    return 'yes' if bool(value) else 'no'


def _shorten(text: str, *, limit: int = 220) -> str:
    raw = ' '.join(str(text or '').split())
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)].rstrip() + '...'


def _payload_with_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(payload or {})
    metadata = dict(enriched.get('report_metadata') or {})
    metadata.setdefault('generated_at_utc', _iso_now())
    metadata.setdefault('format_version', 'realism-report.v1')
    enriched['report_metadata'] = metadata
    return enriched


def create_report_run_dir(output_root: Path, *, tag: str) -> Path:
    run_dir = Path(output_root) / f'{_utc_stamp()}-{tag}'
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json_report(run_dir: Path, payload: dict[str, Any]) -> Path:
    path = Path(run_dir) / 'realism_report.json'
    path.write_text(json.dumps(_payload_with_metadata(payload), ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def write_markdown_report(run_dir: Path, payload: dict[str, Any]) -> Path:
    path = Path(run_dir) / 'realism_report.md'
    path.write_text(render_markdown_report(payload), encoding='utf-8')
    return path


def write_text_artifact(run_dir: Path, filename: str, text: str) -> Path:
    path = Path(run_dir) / filename
    path.write_text(str(text or ''), encoding='utf-8')
    return path


def _prioritized_recommendations(payload: dict[str, Any]) -> list[tuple[str, str]]:
    startup = dict(payload.get('startup') or {})
    evaluation = dict(payload.get('evaluation') or {})
    diagnostics = dict(payload.get('diagnostics') or {})
    recommendations = [str(item).strip() for item in list(evaluation.get('engineering_recommendations') or []) if str(item).strip()]

    prioritized: list[tuple[str, str]] = []
    for item in recommendations:
        priority = 'P2'
        lowered = item.lower()
        if (not startup.get('startup_success')) or 'startup' in lowered or 'live backend' in lowered:
            priority = 'P0'
        elif (
            diagnostics.get('runtime_status', {}).get('mode') == 'degraded'
            or 'leaks generic assistant' in lowered
            or 'latency' in lowered
            or 'cross-turn recall' in lowered
        ):
            priority = 'P1'
        prioritized.append((priority, item))
    return prioritized


def _dialogue_findings_section(dialogue_results: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    lines.append('## Dialogue-by-Dialogue Findings')
    lines.append('')
    if not dialogue_results:
        lines.append('No live dialogue findings were recorded because the runtime never reached a usable chat stage.')
        lines.append('')
        return lines

    for item in dialogue_results:
        lines.append(f"### `{item.get('case_id', '')}`")
        lines.append('')
        lines.append(f"- Category: `{item.get('category', '')}`")
        if item.get('trait_probe'):
            lines.append(f"- Probe: {item.get('trait_probe')}")
        lines.append(f"- Latency: `{item.get('latency_ms', 0)} ms`")
        lines.append(
            f"- Scores: fidelity `{item.get('persona_fidelity')}`, style `{item.get('style_consistency')}`, "
            f"memory `{item.get('memory_continuity')}`, decision `{item.get('decision_authenticity')}`, "
            f"leakage `{item.get('generic_leakage_badness')}`"
        )
        lines.append(f"- Prompt: {_shorten(str(item.get('prompt') or ''), limit=180)}")
        lines.append(f"- Reply excerpt: {_shorten(str(item.get('reply') or ''), limit=260)}")
        if item.get('trait_hits'):
            lines.append("- Trait hits: " + ', '.join(list(item.get('trait_hits') or [])[:6]))
        if item.get('anchor_hits'):
            lines.append("- Anchor hits: " + ', '.join(list(item.get('anchor_hits') or [])[:6]))
        if item.get('persona_success_signal_hits'):
            lines.append("- Persona success signals: " + ', '.join(list(item.get('persona_success_signal_hits') or [])[:6]))
        issues = list(item.get('forbidden_hits') or []) + list(item.get('leakage_hits') or []) + list(item.get('generic_llm_failure_hits') or [])
        if issues:
            lines.append("- Issues: " + ', '.join(issues[:8]))
        for note in list(item.get('evaluation_notes') or [])[:3]:
            lines.append(f"- Note: {note}")
        lines.append('')
    return lines


def _leakage_section(evaluation: dict[str, Any]) -> list[str]:
    dialogue_results = list(evaluation.get('dialogue_results') or [])
    lines: list[str] = []
    lines.append('## Generic Assistant Leakage Analysis')
    lines.append('')
    lines.append(
        f"- Leakage score: `{evaluation.get('generic_llm_leakage_score', 0)}`"
    )
    breakdown = dict(dict(evaluation.get('metric_breakdowns') or {}).get('generic_llm_leakage') or {})
    lines.append(f"- Fallback-like replies: `{breakdown.get('fallback_like_count', 0)}`")
    lines.append(f"- Average leak markers per reply: `{breakdown.get('average_leak_markers', 0)}`")
    leaking_cases = [item for item in dialogue_results if list(item.get('leakage_hits') or []) or list(item.get('generic_llm_failure_hits') or [])]
    if leaking_cases:
        lines.append('- Cases that showed leakage or generic-assistant drift:')
        for item in leaking_cases:
            markers = list(item.get('leakage_hits') or []) + list(item.get('generic_llm_failure_hits') or [])
            lines.append(f"  - `{item.get('case_id', '')}` -> {', '.join(markers[:8])}")
    else:
        lines.append('- No explicit generic-assistant leakage markers were observed in recorded replies.')
    lines.append('')
    return lines


def _memory_section(evaluation: dict[str, Any]) -> list[str]:
    dialogue_results = list(evaluation.get('dialogue_results') or [])
    lines: list[str] = []
    lines.append('## Memory Continuity Analysis')
    lines.append('')
    lines.append(f"- Memory continuity score: `{evaluation.get('memory_continuity_score', 0)}`")
    memory_cases = [item for item in dialogue_results if float(item.get('memory_continuity') or 0.0) > 0.0]
    if memory_cases:
        for item in memory_cases:
            expected_fragment = str(
                dict(dict(item.get('score_breakdown') or {}).get('memory_continuity') or {}).get('expected_fragment') or ''
            ).strip()
            lines.append(
                f"- `{item.get('case_id', '')}` -> score `{item.get('memory_continuity')}`"
                + (f"; expected fragment: {_shorten(expected_fragment, limit=120)}" if expected_fragment else '')
            )
    else:
        lines.append('- No explicit memory-continuity cases produced a measurable recall signal in this run.')
    lines.append('')
    return lines


def _contradictions_section(evaluation: dict[str, Any]) -> list[str]:
    breakdown = dict(dict(evaluation.get('metric_breakdowns') or {}).get('contradiction_detection') or {})
    lines: list[str] = []
    lines.append('## Contradictions')
    lines.append('')
    lines.append(f"- Contradiction count: `{evaluation.get('contradiction_count', 0)}`")
    groups = list(breakdown.get('consistency_groups') or [])
    if groups:
        for group in groups:
            lines.append(
                f"- Group `{group.get('group', '')}` -> overlap `{group.get('average_overlap', 0)}`, band `{group.get('band', '')}`, replies `{group.get('reply_count', 0)}`"
            )
    else:
        lines.append('- No repeated-topic consistency groups were available to compare in this run.')
    lines.append('')
    return lines


def render_markdown_report(payload: dict[str, Any]) -> str:
    payload = _payload_with_metadata(payload)
    run = dict(payload.get('run') or {})
    startup = dict(payload.get('startup') or {})
    reachability = dict(payload.get('reachability') or {})
    evaluation = dict(payload.get('evaluation') or {})
    persona = dict(payload.get('persona_materialization') or {})
    diagnostics = dict(payload.get('diagnostics') or {})
    dialogue_results = list(evaluation.get('dialogue_results') or [])
    prioritized_recommendations = _prioritized_recommendations(payload)

    lines: list[str] = []
    lines.append('# Runtime Realism Report')
    lines.append('')
    lines.append('This report is written as an engineering diagnosis of a live AI product path, not as a unit-test transcript.')
    lines.append('')
    lines.append('## Final Verdict')
    lines.append('')
    lines.append(f"- Verdict: `{evaluation.get('overall_verdict', 'unknown')}`")
    lines.append(f"- Infrastructure status: `{evaluation.get('infrastructure_status', 'unknown')}`")
    lines.append(f"- Startup success: `{startup.get('startup_success', False)}`")
    lines.append(f"- Frontend reachable: `{evaluation.get('frontend_reachable', False)}`")
    lines.append(f"- API reachable: `{evaluation.get('api_reachable', False)}`")
    lines.append(f"- Report generated at: `{payload.get('report_metadata', {}).get('generated_at_utc', '')}`")
    lines.append('')

    lines.append('## Startup Results')
    lines.append('')
    lines.append(f"- Command: `{ ' '.join(startup.get('command') or []) }`")
    lines.append(f"- Profile: `{run.get('profile', '')}`")
    lines.append(f"- Host/port: `{run.get('host', '')}:{run.get('port', '')}`")
    lines.append(f"- Startup time: `{startup.get('startup_time_ms', 0)} ms`")
    lines.append(f"- Probable failure reason: `{startup.get('probable_failure_reason', '')}`")
    if startup.get('process_exited_early'):
        lines.append("- Process exited before readiness was reached.")
    if startup.get('log_tail'):
        lines.append('- Log tail:')
        lines.append('```text')
        lines.extend(str(item) for item in list(startup.get('log_tail') or [])[-25:])
        lines.append('```')
    lines.append('')

    lines.append('## Endpoint Reachability')
    lines.append('')
    lines.append(f"- Root reachable: `{_bool_icon(reachability.get('root_reachable'))}`")
    lines.append(f"- Root looks like HTML: `{_bool_icon(reachability.get('root_html'))}`")
    lines.append(f"- Runtime health reachable: `{_bool_icon(reachability.get('health_reachable'))}`")
    lines.append(f"- Chat endpoint reachable: `{_bool_icon(reachability.get('chat_alive'))}`")
    lines.append(f"- API JSON valid: `{_bool_icon(reachability.get('api_json_valid'))}`")
    lines.append('')

    lines.append('## Latency Summary')
    lines.append('')
    lines.append(f"- Average latency: `{evaluation.get('average_latency_ms', 0)} ms`")
    lines.append(f"- Max latency: `{evaluation.get('max_latency_ms', 0)} ms`")
    lines.append(f"- Timeout count: `{evaluation.get('timeout_count', 0)}`")
    if evaluation.get('score_explanations'):
        lines.append(f"- Interpretation: {dict(evaluation.get('score_explanations') or {}).get('latency', '')}")
    lines.append('')

    lines.append('## Persona Materialization Status')
    lines.append('')
    lines.append(f"- Persona: `{persona.get('name', '')}`")
    lines.append(f"- Slug: `{persona.get('slug', '')}`")
    lines.append(f"- Materialization ok: `{persona.get('ok', False)}`")
    lines.append(f"- Graph sync visible: `{persona.get('graph_sync_visible', False)}`")
    if persona.get('summary'):
        summary = dict(persona.get('summary') or {})
        lines.append(
            f"- Summary: entity `{summary.get('entity_type', '')}`, traits `{summary.get('trait_count', 0)}`, "
            f"examples `{summary.get('example_count', 0)}`, relations `{summary.get('relation_count', 0)}`, "
            f"knowledge chars `{summary.get('knowledge_chars', 0)}`"
        )
    if persona.get('required_files'):
        missing = [name for name, present in sorted(dict(persona.get('required_files') or {}).items()) if not present]
        if missing:
            lines.append("- Missing required files: " + ', '.join(missing))
        else:
            lines.append('- All required persona files were present.')
    lines.append('')

    lines.append('## Scorecard')
    lines.append('')
    for key in (
        'persona_fidelity_score',
        'style_consistency_score',
        'memory_continuity_score',
        'decision_authenticity_score',
        'generic_llm_leakage_score',
        'contradiction_count',
    ):
        lines.append(f"- `{key}`: `{evaluation.get(key)}`")
    lines.append('')

    if evaluation.get('score_explanations'):
        lines.append('## Score Explanations')
        lines.append('')
        for key, value in dict(evaluation.get('score_explanations') or {}).items():
            lines.append(f"- `{key}`: {value}")
        lines.append('')

    lines.extend(_dialogue_findings_section(dialogue_results))
    lines.extend(_leakage_section(evaluation))
    lines.extend(_memory_section(evaluation))
    lines.extend(_contradictions_section(evaluation))

    if evaluation.get('major_failures'):
        lines.append('## Major Failures')
        lines.append('')
        for item in list(evaluation.get('major_failures') or []):
            lines.append(f'- {item}')
        lines.append('')

    if evaluation.get('suspicious_patterns'):
        lines.append('## Suspicious Patterns')
        lines.append('')
        for item in list(evaluation.get('suspicious_patterns') or []):
            lines.append(f'- {item}')
        lines.append('')

    lines.append('## Diagnostics')
    lines.append('')
    lines.append(f"- Runtime mode: `{dict(diagnostics.get('runtime_status') or {}).get('mode', '')}`")
    lines.append(f"- Trace count observed: `{len(list(diagnostics.get('traces') or []))}`")
    lines.append(f"- Graph health node count: `{dict(diagnostics.get('graph_health') or {}).get('node_count', 0)}`")
    lines.append(f"- Graph duplicate rate: `{dict(diagnostics.get('graph_health') or {}).get('duplicate_rate', 0)}`")
    lines.append(f"- Graph orphan rate: `{dict(diagnostics.get('graph_health') or {}).get('orphan_rate', 0)}`")
    judge = dict(evaluation.get('judge_evaluation') or {})
    lines.append(f"- Judge enabled: `{judge.get('enabled', False)}`")
    lines.append(f"- Judge used: `{judge.get('used', False)}`")
    if judge.get('error'):
        lines.append(f"- Judge error: `{judge.get('error')}`")
    if diagnostics.get('runtime_operator_messages'):
        lines.append('- Runtime operator messages:')
        for item in list(diagnostics.get('runtime_operator_messages') or [])[:10]:
            lines.append(f'  - {item}')
    lines.append('')

    lines.append('## Prioritized Recommendations')
    lines.append('')
    if prioritized_recommendations:
        for priority, item in prioritized_recommendations:
            lines.append(f'- `{priority}` {item}')
    else:
        lines.append('- No recommendations were generated.')
    lines.append('')

    return '\n'.join(lines).strip() + '\n'
