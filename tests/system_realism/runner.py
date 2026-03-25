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
    parser.add_argument('--host', default='127.0.0.1', help='Host for the launched runtime')
    parser.add_argument('--port', type=int, default=0, help='Explicit port. Default: choose a free localhost port')
    parser.add_argument('--api-only', action='store_true', help='Launch backend API only without combined frontend routes')
    parser.add_argument('--strict', action='store_true', help='Fail with non-zero exit status on bad realism verdicts')
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
            host=args.host,
            port=int(args.port or 0),
            api_only=bool(args.api_only),
            strict=bool(args.strict),
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
