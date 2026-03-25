from __future__ import annotations

import os
from pathlib import Path

from .harness import run_realism_suite
from .models import RealismRunConfig


def test_runtime_realism_harness_generates_operator_report() -> None:
    strict = str(os.getenv('COGNITIVE_REALISM_STRICT', '')).strip() in {'1', 'true', 'yes'}
    profile = str(os.getenv('COGNITIVE_REALISM_PROFILE', 'local-demo')).strip() or 'local-demo'
    repo_root = Path(__file__).resolve().parents[2]
    output_root = Path(
        os.getenv(
            'COGNITIVE_REALISM_OUTPUT_ROOT',
            str(repo_root / 'runtime' / 'system_realism_reports'),
        )
    )
    memory_root = Path(
        os.getenv(
            'COGNITIVE_REALISM_MEMORY_ROOT',
            str(repo_root / 'runtime' / 'system_realism_memory_pytest'),
        )
    )
    report = run_realism_suite(
        RealismRunConfig(
            profile=profile,
            memory_root=memory_root,
            output_root=output_root,
            strict=strict,
            report_tag='pytest',
        )
    )

    artifacts = dict(report.get('artifacts') or {})
    assert artifacts.get('json_report')
    assert artifacts.get('markdown_report')
    assert Path(str(artifacts.get('json_report'))).exists()
    assert Path(str(artifacts.get('markdown_report'))).exists()
    assert report['startup']['startup_attempted'] is True
    assert report['evaluation']['overall_verdict']

    if strict:
        assert report['startup']['startup_success'] is True
        assert report['evaluation']['overall_verdict'] not in {
            'startup_failed',
            'api_unreachable',
            'generic_assistant_drift_or_degraded_runtime',
        }
