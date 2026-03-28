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
- exploratory prompts that probe less-scripted persona behavior
- generated unexpected and unseen-generalization prompts
- live persona evolution through dossier updates
- memory deletion through real persona revision restore
- graph editor simulation through graph create/connect/patch/delete actions
- contradiction resistance and identity continuity after mutations
- optional low-frequency chaos runs
- latency and timeout behavior
- degraded runtime signals and graph diagnostics

## How to run

Default one-command pytest entry:

```bash
python3 -m pytest tests/system_realism -q
```

This runs the real subprocess harness and writes reports under `runtime/system_realism_reports/` by default.

By default the pytest entry uses the lighter `advanced` suite:

- generated unexpected prompts
- unseen paraphrase/generalization prompts
- persona memory injection and evolution
- graph editor simulation
- deletion / restore checks
- contradiction and identity continuity probes

Use `COGNITIVE_REALISM_SUITE=full` or the direct runner when you want the heavier baseline-plus-advanced pass.
The pytest entry also defaults to `COGNITIVE_REALISM_MUTATION_SUBSET=smoke`, which keeps the live mutation pass short enough for local GGUF runtimes. Use `all` for the full mutation graph/persona path.

Strict mode, which fails the test on a bad realism verdict:

```bash
COGNITIVE_REALISM_STRICT=1 python3 -m pytest tests/system_realism -q
```

Choose a runtime profile:

```bash
COGNITIVE_REALISM_PROFILE=local-demo python3 -m pytest tests/system_realism -q
```

Choose the suite:

```bash
COGNITIVE_REALISM_SUITE=full python3 -m pytest tests/system_realism -q
```

Available suites:

- `baseline`: only baseline dialogue realism
- `core`: baseline + mutation/evolution realism
- `advanced`: mutation/evolution realism without the baseline pass
- `full`: all baseline dialogues, exploratory prompts, and mutation/evolution realism

Increase request timeout for slower local models:

```bash
COGNITIVE_REALISM_REQUEST_TIMEOUT=120 python3 -m pytest tests/system_realism -q
```

Change exploratory prompt coverage:

```bash
COGNITIVE_REALISM_EXPLORATORY_CASES=8 COGNITIVE_REALISM_EXPLORATORY_SEED=23 python3 -m pytest tests/system_realism -q
```

Change generated unexpected/generalization coverage:

```bash
COGNITIVE_REALISM_UNEXPECTED_CASES=4 COGNITIVE_REALISM_GENERALIZATION_CASES=4 python3 -m pytest tests/system_realism -q
```

Enable low-frequency chaos:

```bash
COGNITIVE_REALISM_INCLUDE_CHAOS=1 python3 -m pytest tests/system_realism -q
```

Choose mutation coverage:

```bash
COGNITIVE_REALISM_MUTATION_SUBSET=all python3 -m pytest tests/system_realism -q
```

Direct runner, which is better when you want a report without pytest framing:

```bash
python3 -m tests.system_realism --profile local-demo --request-timeout 120 --tag manual
```

Or with the full evolution suite:

```bash
python3 -m tests.system_realism --profile local-demo --suite full --unexpected-cases 4 --generalization-cases 4 --tag manual
```

Or run the lighter mutation smoke path explicitly:

```bash
python3 -m tests.system_realism --profile local-demo --suite advanced --mutation-subset smoke --unexpected-cases 1 --generalization-cases 1 --tag manual
```

## Ports

The harness chooses a free localhost port automatically unless `RealismRunConfig.port` is set explicitly.

## Runtime launcher

The launcher lives in `runtime_launcher.py` and uses the real project entrypoint:

```text
python start.py --profile <profile> --host <host> --port <port>
```

It prefers:

1. `COGNITIVE_REALISM_PYTHON`
2. repo-local `.venv/bin/python`
3. current `sys.executable`

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

The realism harness now has two dialogue layers:

1. the fixed canonical benchmark
2. a larger exploratory prompt pool sampled by seed
3. generated unexpected / unseen-generalization prompts
4. live mutation scenarios that change persona or graph state and then probe adaptation

The fixed benchmark makes regressions reproducible. The exploratory pool reduces overfitting to one narrow prompt set.
The mutation scenarios test whether the system behaves from state rather than from memorized prompts.

## Notes

- The markdown report is the primary human-readable artifact; the JSON report is intended for automation or later aggregation.
- `test_runtime_realism.py` is the top-level integration entry that wires together launcher, persona materialization, dialogue benchmark, evaluator, and report generation.
- The direct runner in `runner.py` uses the same harness but is friendlier for manual operator use.
