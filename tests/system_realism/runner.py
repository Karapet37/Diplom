from __future__ import annotations

import argparse
import json
from pathlib import Path

from .harness import run_realism_suite
from .models import RealismRunConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description='Run the persona-graph system realism harness.')
    parser.add_argument('--profile', default='local-demo', help='Runtime profile to launch via start.py')
    parser.add_argument(
        '--suite',
        default='core',
        choices=('baseline', 'core', 'advanced', 'full'),
        help='baseline=only baseline dialogues, core=baseline+mutation realism, advanced=mutation realism only, full=all baseline dialogues plus advanced realism',
    )
    parser.add_argument('--host', default='127.0.0.1', help='Host for the launched runtime')
    parser.add_argument('--port', type=int, default=0, help='Explicit port. Default: choose a free localhost port')
    parser.add_argument('--api-only', action='store_true', help='Launch backend API only without combined frontend routes')
    parser.add_argument('--strict', action='store_true', help='Fail with non-zero exit status on bad realism verdicts')
    parser.add_argument('--request-timeout', type=float, default=90.0, help='Per-request timeout for live dialogue probes')
    parser.add_argument('--startup-timeout', type=float, default=40.0, help='Startup readiness timeout')
    parser.add_argument('--exploratory-cases', type=int, default=6, help='How many exploratory prompts to sample beyond the fixed benchmark')
    parser.add_argument('--exploratory-seed', type=int, default=17, help='Seed for deterministic exploratory prompt sampling')
    parser.add_argument('--unexpected-cases', type=int, default=3, help='How many generated unexpected/rare prompts to run in the advanced suite')
    parser.add_argument('--generalization-cases', type=int, default=3, help='How many unseen paraphrased prompts to run in the advanced suite')
    parser.add_argument(
        '--mutation-subset',
        default='all',
        choices=('smoke', 'all'),
        help='smoke=lighter mutation path, all=full evolution and graph-editor path',
    )
    parser.add_argument('--include-chaos', action='store_true', help='Enable low-frequency chaos mutation scenarios')
    parser.add_argument(
        '--memory-root',
        default=str(repo_root / 'runtime' / 'system_realism_memory_runner'),
        help='Memory root used for the isolated realism run',
    )
    parser.add_argument(
        '--output-root',
        default=str(repo_root / 'runtime' / 'system_realism_reports'),
        help='Directory where the realism reports will be written',
    )
    parser.add_argument('--tag', default='manual', help='Tag suffix for the report directory name')
    parser.add_argument('--json', action='store_true', help='Print compact JSON summary instead of prose')
    args = parser.parse_args()

    report = run_realism_suite(
        RealismRunConfig(
            profile=args.profile,
            suite=args.suite,
            host=args.host,
            port=int(args.port or 0),
            api_only=bool(args.api_only),
            strict=bool(args.strict),
            request_timeout_s=float(args.request_timeout),
            startup_timeout_s=float(args.startup_timeout),
            exploratory_case_count=int(args.exploratory_cases),
            exploratory_seed=int(args.exploratory_seed),
            unexpected_case_count=int(args.unexpected_cases),
            generalization_case_count=int(args.generalization_cases),
            mutation_subset=str(args.mutation_subset or 'all'),
            include_chaos=bool(args.include_chaos),
            memory_root=Path(args.memory_root),
            output_root=Path(args.output_root),
            report_tag=str(args.tag or 'manual'),
        )
    )

    summary = {
        'overall_verdict': report.get('evaluation', {}).get('overall_verdict'),
        'startup_success': report.get('startup', {}).get('startup_success'),
        'json_report': report.get('artifacts', {}).get('json_report'),
        'markdown_report': report.get('artifacts', {}).get('markdown_report'),
        'server_log': report.get('artifacts', {}).get('server_log'),
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Verdict: {summary['overall_verdict']}")
        print(f"Startup success: {summary['startup_success']}")
        print(f"Markdown report: {summary['markdown_report']}")
        print(f"JSON report: {summary['json_report']}")
        print(f"Server log: {summary['server_log']}")

    if args.strict and report.get('evaluation', {}).get('overall_verdict') in {
        'startup_failed',
        'api_unreachable',
        'generic_assistant_drift_or_degraded_runtime',
    }:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
