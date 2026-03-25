# System Realism Harness

This subsystem launches the real runtime, seeds a canonical persona, runs dialogue probes through the live API, evaluates behavioral realism, and writes engineering reports.

## What it checks

- real startup through `start.py`
- root page reachability
- health reachability
- chat API reachability
- persona materialization in actual storage format
- persona fidelity vs generic assistant drift
- memory continuity across turns
- latency and timeout behavior
- degraded runtime signals and graph diagnostics

## How to run

Default one-command pytest entry:

```bash
python3 -m pytest tests/system_realism -q
```

This runs the real subprocess harness and writes reports under `runtime/system_realism_reports/` by default.

Strict mode, which fails the test on a bad realism verdict:

```bash
COGNITIVE_REALISM_STRICT=1 python3 -m pytest tests/system_realism -q
```

Choose a runtime profile:

```bash
COGNITIVE_REALISM_PROFILE=local-demo python3 -m pytest tests/system_realism -q
```

Direct runner, which is better when you want a report without pytest framing:

```bash
python3 -m tests.system_realism --profile local-demo --tag manual
```

## Ports

The harness chooses a free localhost port automatically unless `RealismRunConfig.port` is set explicitly.

## Runtime launcher

The launcher lives in `runtime_launcher.py` and uses the real project entrypoint:

```text
python start.py --profile <profile> --host <host> --port <port>
```

It captures merged stdout/stderr logs, waits on `/api/cognitive/health`, diagnoses startup failure reasons from real log output, and exposes helpers for:

- root reachability
- runtime health reachability
- combined surface health reachability
- live chat API reachability

## Reports

The harness writes:

- `realism_report.json`
- `realism_report.md`
- `server.log`
- `log_tail.txt`

The default output root is `runtime/system_realism_reports/`, but the pytest entrypoint overrides it with an isolated temporary directory for test isolation.

## Persona fixture

The canonical persona is defined in `persona_fixture.py` and materialized via the system’s own `materialize_persona(...)` path, so the test uses the real storage format instead of a fake fixture layout.

## Adding new personas or dialogue sets

1. Add or extend a fixture in `persona_fixture.py`.
2. Add dialogue probes in `dialogue_cases.py`.
3. Extend heuristic scoring in `evaluator.py` if the new persona has important style or memory signals that are not yet covered.

## Notes

- The markdown report is the primary human-readable artifact; the JSON report is intended for automation or later aggregation.
- `test_runtime_realism.py` is the top-level integration entry that wires together launcher, persona materialization, dialogue benchmark, evaluator, and report generation.
- The direct runner in `runner.py` uses the same harness but is friendlier for manual operator use.
