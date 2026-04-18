# System Realism Harness

This subsystem launches the real runtime, seeds personas, drives live API requests, and evaluates whether the system behaves like the controller-first architecture it claims to implement.

## What It Checks

The realism harness is aimed at behavioral correctness, not only endpoint reachability.

Main checks include:

- real startup through `start.py`
- health and chat API reachability
- persona materialization through the real storage path
- persona creation from structured descriptions
- persona fast-path behavior for lightweight turns
- persona graph reasoning for heavier turns
- memory continuity across turns
- graph-backed retrieval behavior
- degraded runtime signals
- latency and timeout behavior
- runtime logs and report generation

This harness complements, rather than replaces, the denser unit coverage in `tests/agent_system/` for planning mode, notes, behavior regulation, personality construction, observability, and training-example storage.

## Why It Exists

The project intentionally avoids “single prompt survival” as a success metric.

This harness helps verify that:

- request routing is correct,
- persona behavior remains stable,
- memory layers are used in the right order,
- the real app still behaves correctly under live conditions.

## Run

Default pytest entry:

```bash
.venv/bin/python -m pytest tests/system_realism -q
```

Strict mode:

```bash
COGNITIVE_REALISM_STRICT=1 .venv/bin/python -m pytest tests/system_realism -q
```

Choose a runtime profile:

```bash
COGNITIVE_REALISM_PROFILE=local-demo .venv/bin/python -m pytest tests/system_realism -q
```

Direct runner:

```bash
.venv/bin/python -m tests.system_realism --profile local-demo --request-timeout 120 --tag manual
```

## Suites

Available suites:

- `baseline`
- `core`
- `advanced`
- `full`

The lighter path is useful for frequent local regression checks. The heavier path is better when validating long-running persona and graph behavior under real subprocess conditions.

## Reports

The harness writes artifacts such as:

- `realism_report.json`
- `realism_report.md`
- `server.log`
- `log_tail.txt`

Default output root:

- `runtime/system_realism_reports/`

## Notes

- The harness uses the real project entrypoint instead of a fake in-process mock.
- It is intended to catch route drift, persona drift, and live-runtime regressions that unit tests may miss.
- It is strongest for startup, subprocess orchestration, reporting, and end-to-end realism signals; planning/note/regulator correctness is covered more exhaustively in the unit suite.
- In restricted sandboxes that disallow opening TCP sockets, runtime-launcher tests can fail with `PermissionError` during free-port allocation even when the harness code is otherwise correct.
