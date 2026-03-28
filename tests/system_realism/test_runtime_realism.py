from __future__ import annotations

import os
from pathlib import Path

from .harness import run_realism_suite
from .models import RealismRunConfig


def test_runtime_realism_harness_generates_operator_report() -> None:
    strict = str(os.getenv('COGNITIVE_REALISM_STRICT', '')).strip() in {'1', 'true', 'yes'}
    profile = str(os.getenv('COGNITIVE_REALISM_PROFILE', 'local-demo')).strip() or 'local-demo'
    suite = str(os.getenv('COGNITIVE_REALISM_SUITE', 'advanced')).strip() or 'advanced'
    request_timeout_s = float(str(os.getenv('COGNITIVE_REALISM_REQUEST_TIMEOUT', '10')).strip() or '10')
    exploratory_case_count = int(str(os.getenv('COGNITIVE_REALISM_EXPLORATORY_CASES', '6')).strip() or '6')
    exploratory_seed = int(str(os.getenv('COGNITIVE_REALISM_EXPLORATORY_SEED', '17')).strip() or '17')
    unexpected_case_count = int(str(os.getenv('COGNITIVE_REALISM_UNEXPECTED_CASES', '1')).strip() or '1')
    generalization_case_count = int(str(os.getenv('COGNITIVE_REALISM_GENERALIZATION_CASES', '1')).strip() or '1')
    mutation_subset = str(os.getenv('COGNITIVE_REALISM_MUTATION_SUBSET', 'smoke')).strip() or 'smoke'
    include_chaos = str(os.getenv('COGNITIVE_REALISM_INCLUDE_CHAOS', '')).strip() in {'1', 'true', 'yes'}
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
            suite=suite,
            memory_root=memory_root,
            output_root=output_root,
            strict=strict,
            request_timeout_s=request_timeout_s,
            exploratory_case_count=exploratory_case_count,
            exploratory_seed=exploratory_seed,
            unexpected_case_count=unexpected_case_count,
            generalization_case_count=generalization_case_count,
            mutation_subset=mutation_subset,
            include_chaos=include_chaos,
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
