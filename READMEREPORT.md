# Project Report

Repository: `https://github.com/Karapet37/Diplom`

## 1. Current State

The repository was converged into one active application instead of several parallel experiments.

Current active system:

- one backend runtime
- one frontend
- one root Python environment
- one cognitive graph subsystem

Main active directories:

- `core/`
- `runtime/`
- `roaches_viz/roaches_viz/`
- `src/web/`
- `webapp/`

## 2. Active Architecture

### Core layer

The repository now has a core-first contract:

- `system_core`
- `personality_core`
- `graph_core`
- `context_core`
- `agent_core`
- `style_engine`
- `speech_dna`
- `dialogue_engine`
- `scenario_engine`
- `agent_roles`
- `graph_initializer`
- `graph_traversal`

This layer defines the stable data model and orchestration boundaries.

### Runtime layer

The active runtime is asynchronous:

1. user query
2. fast assistant/router reply
3. response returned immediately
4. background graph job
5. pending -> verified graph update

This removed the old synchronous path that caused long delays and `504` errors.

### Storage layer

The system stores:

- graph state
- sessions
- style profiles
- personality logs

Runtime storage zones:

- `graph/verified/`
- `graph/pending/`
- `data/sessions/`
- `style_profiles/`
- `roaches_viz/logs/`

## 3. What Was Cleaned Up

The repository was prepared for normal git use.

### Kept under version control

- source code
- tests
- documentation
- runtime directory placeholders (`.gitkeep`)

### Removed from git tracking / ignored

- runtime graph artifacts
- project backups
- session JSON files
- style profile JSON files
- runtime logs
- DB sidecar files
- local models
- local environments

This reduces noise and makes the repository copyable without dragging local runtime state into git.

## 4. Documentation Updated

### `README.md`

Updated to reflect:

- current unified architecture
- active runtime surfaces
- workspaces
- run commands
- test commands
- git hygiene rules

### `READMEREPORT.md`

Rewritten to reflect the current system instead of older experimental stages.

## 5. Current User-Facing System

### Chat workspace

- session sidebar
- message history
- fixed input bar
- async response path

### Graph workspace

- relevant subgraph rendering
- node editing
- relation editing
- graph search and subgraph loading

## 6. Current Run Instructions

Backend:

```bash
cd /home/karapet/agent_project
pip install -e .[dev]
python start.py --host 127.0.0.1 --port 8008
```

Frontend:

```bash
cd /home/karapet/agent_project
VITE_API_BASE_URL=http://127.0.0.1:8008 npm --prefix webapp run dev
```

## 7. Current Test Instructions

Backend:

```bash
cd /home/karapet/agent_project
PYTHONPATH=. roaches_viz/.venv/bin/python -m pytest -q tests/unit roaches_viz/tests style_tests
```

Frontend:

```bash
cd /home/karapet/agent_project
npm --prefix webapp run build
```

## 8. Practical Result

The repository is now closer to a normal working product:

- code and docs stay tracked
- runtime state stays local
- generated files no longer pollute git history
- architecture is documented as one active system, not a stack of experiments
