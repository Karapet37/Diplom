from __future__ import annotations

import json

from tests.system_realism.reporting import render_markdown_report, write_json_report


def _sample_payload(startup_success: bool) -> dict:
    return {
        'run': {
            'profile': 'local-demo',
            'host': '127.0.0.1',
            'port': 8123,
        },
        'startup': {
            'startup_success': startup_success,
            'startup_time_ms': 321.5,
            'probable_failure_reason': '' if startup_success else 'missing_uvicorn_dependency',
            'command': ['python', 'start.py', '--profile', 'local-demo'],
            'process_exited_early': not startup_success,
            'log_tail': ["Runtime startup error: No module named 'uvicorn'."] if not startup_success else [],
        },
        'reachability': {
            'root_reachable': startup_success,
            'root_html': startup_success,
            'health_reachable': startup_success,
            'chat_alive': startup_success,
            'api_json_valid': startup_success,
        },
        'persona_materialization': {
            'name': 'Dr. Aram Petrosyan',
            'slug': 'dr_aram_petrosyan',
            'ok': True,
            'graph_sync_visible': True,
            'summary': {'entity_type': 'PERSON', 'trait_count': 9, 'example_count': 12, 'relation_count': 9, 'knowledge_chars': 3317},
            'required_files': {'traits.json': True, 'persona_form.json': True},
        },
        'diagnostics': {
            'runtime_status': {'mode': 'normal' if startup_success else 'degraded'},
            'traces': [{'trace_id': 't1'}] if startup_success else [],
            'graph_health': {'node_count': 12, 'duplicate_rate': 0.0, 'orphan_rate': 0.08},
            'runtime_operator_messages': ['local role missing'] if not startup_success else [],
        },
        'evaluation': {
            'overall_verdict': 'persona_alive_and_believable' if startup_success else 'startup_failed',
            'infrastructure_status': 'alive' if startup_success else 'startup_failed',
            'frontend_reachable': startup_success,
            'api_reachable': startup_success,
            'average_latency_ms': 812.1 if startup_success else 0.0,
            'max_latency_ms': 1099.4 if startup_success else 0.0,
            'timeout_count': 0,
            'persona_fidelity_score': 0.82 if startup_success else 0.0,
            'style_consistency_score': 0.78 if startup_success else 0.0,
            'memory_continuity_score': 0.63 if startup_success else 0.0,
            'decision_authenticity_score': 0.81 if startup_success else 0.0,
            'generic_llm_leakage_score': 0.11 if startup_success else 0.0,
            'contradiction_count': 0,
            'major_failures': ['Backend did not start through the real runtime entrypoint.'] if not startup_success else [],
            'suspicious_patterns': ['Runtime reports degraded mode during the realism run.'] if not startup_success else [],
            'engineering_recommendations': ['Inspect startup logs first. The realism harness could not reach a live backend through start.py.'] if not startup_success else ['Keep watching degraded-mode diagnostics and rerun realism after graph/persona changes. The system currently behaves like a living product.'],
            'score_explanations': {
                'latency': 'Latency summary is based on live response timings.',
                'persona_fidelity': 'Persona fidelity is strong because anchors and style markers are present.',
            },
            'dialogue_results': [
                {
                    'case_id': 'identity_work',
                    'category': 'identity',
                    'prompt': 'Who are you?',
                    'trait_probe': 'Core identity.',
                    'latency_ms': 550.0,
                    'reply': 'I am Aram Petrosyan, an emergency physician and triage lead in Yerevan.',
                    'persona_fidelity': 0.84,
                    'style_consistency': 0.75,
                    'memory_continuity': 0.0,
                    'decision_authenticity': 0.72,
                    'generic_leakage_badness': 0.0,
                    'trait_hits': ['emergency', 'triage'],
                    'anchor_hits': ['Yerevan'],
                    'persona_success_signal_hits': ['answers in first person as Aram rather than as a generic assistant'],
                    'forbidden_hits': [],
                    'leakage_hits': [],
                    'generic_llm_failure_hits': [],
                    'evaluation_notes': ['Trait/anchor alignment for this probe.'],
                    'score_breakdown': {'memory_continuity': {'expected_fragment': ''}},
                }
            ]
            if startup_success
            else [],
            'metric_breakdowns': {
                'generic_llm_leakage': {'fallback_like_count': 0, 'average_leak_markers': 0.0},
                'contradiction_detection': {'consistency_groups': []},
            },
            'judge_evaluation': {'enabled': False, 'used': False, 'error': ''},
        },
    }


def test_markdown_report_contains_engineering_sections_for_live_like_run() -> None:
    markdown = render_markdown_report(_sample_payload(True))
    assert '# Runtime Realism Report' in markdown
    assert '## Startup Results' in markdown
    assert '## Endpoint Reachability' in markdown
    assert '## Latency Summary' in markdown
    assert '## Persona Materialization Status' in markdown
    assert '## Dialogue-by-Dialogue Findings' in markdown
    assert '## Generic Assistant Leakage Analysis' in markdown
    assert '## Memory Continuity Analysis' in markdown
    assert '## Contradictions' in markdown
    assert '## Final Verdict' in markdown
    assert '## Prioritized Recommendations' in markdown


def test_markdown_report_still_useful_when_startup_fails() -> None:
    markdown = render_markdown_report(_sample_payload(False))
    assert 'startup_failed' in markdown
    assert 'missing_uvicorn_dependency' in markdown
    assert 'No live dialogue findings were recorded because the runtime never reached a usable chat stage.' in markdown
    assert 'Inspect startup logs first.' in markdown


def test_json_report_adds_machine_readable_metadata(tmp_path) -> None:
    path = write_json_report(tmp_path, _sample_payload(False))
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert 'report_metadata' in payload
    assert payload['report_metadata']['format_version'] == 'realism-report.v1'
    assert payload['report_metadata']['generated_at_utc']
