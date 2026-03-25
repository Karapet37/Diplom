# Runtime Profiles and Startup

This runtime is local-first and profile-driven.

The single startup command is still `python start.py`, but startup now layers configuration in a reproducible order:

```text
CLI flags
  -> shell environment
  -> env file
  -> runtime profile template
  -> code defaults
```

## Available profiles

Profile templates live in `config/runtime-profiles/`.

- `development`
  - default local development profile
  - combined backend + frontend
  - moderate context budgets
  - background rebuild enabled
- `local-demo`
  - fast local demo profile
  - smaller budgets
  - background rebuild disabled by default for responsiveness
- `local-heavy`
  - larger context and graph limits
  - intended for stronger local retrieval when latency is acceptable
- `server`
  - network-bound profile for local server or container use
  - persistent runtime paths under `runtime/server/`

## Recommended setup

1. Bootstrap once:

```bash
./scripts/bootstrap_local.sh
```

2. Edit `.env.local` if you need explicit GGUF model paths.

3. Build the frontend when you need the combined app surface:

```bash
cd webapp
npm run build
```

4. Start a profile:

```bash
./scripts/run_profile.sh development
./scripts/run_profile.sh local-demo
./scripts/run_profile.sh local-heavy
./scripts/run_profile.sh server
```

## Direct startup commands

List profiles:

```bash
python start.py --list-profiles
```

Inspect startup and exit:

```bash
python start.py --profile development --check
```

Print the resolved runtime config:

```bash
python start.py --profile development --print-config
```

Run API only:

```bash
python start.py --profile development --api-only
```

Run with explicit env/config layering:

```bash
python start.py --profile local-demo --env-file .env.local --config config/runtime-profiles/local-demo.yaml
```

## Paths and configurability

The runtime now treats these paths as configurable and repo-relative unless made absolute:

- `COGNITIVE_MEMORY_ROOT`
- `COGNITIVE_WEBAPP_DIR`
- `COGNITIVE_WEBAPP_DIST_DIR`
- `LOCAL_MODELS_DIR`
- explicit `LOCAL_*_GGUF_MODEL` role paths

This avoids machine-specific assumptions about the current working directory.

## Operator-friendly failure modes

`start.py --check` prints a readable startup summary and warnings for common deployment issues:

- missing frontend build
- missing fallback frontend file
- missing GGUF models for required runtime roles
- missing env/config files

The runtime does not silently switch into a different architecture when these are missing. It reports degradation explicitly.
