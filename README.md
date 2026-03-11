# Agent Project

Unified graph-first AI runtime with one backend, one frontend, one root environment.

## Active Architecture

The repository now has one active system boundary:

- `core/`
  - foundational contracts for graph, personality, context, agents, style, dialogue, scenarios, and traversal
- `runtime/`
  - async request orchestration
  - fast response pipeline
  - background graph build queue and workers
- `roaches_viz/roaches_viz/`
  - behavioral graph engine, storage, ingestion, Graph-RAG, API layer, actor integration
- `src/web/`
  - combined FastAPI application
  - workspace/control-plane endpoints
  - mounts the cognitive runtime under the same server
- `webapp/`
  - the only runtime frontend

There is no separate production frontend or second backend runtime.

## Runtime Model

Current request path:

1. user query
2. fast assistant/router
3. immediate response
4. background graph job
5. pending/verified graph update

Graph storage zones:

- `graph/verified/`
- `graph/pending/`

Session storage:

- `data/sessions/{session_id}.json`

Style profiles:

- `style_profiles/{user_id}.json`

Personality logs:

- `roaches_viz/logs/{person_id}.txt`

These runtime artifacts are intentionally ignored by git.

## Core-First Modules

Main core modules:

- `core/system_core.py`
- `core/personality_core.py`
- `core/graph_core.py`
- `core/context_core.py`
- `core/agent_core.py`
- `core/style_engine.py`
- `core/speech_dna.py`
- `core/dialogue_engine.py`
- `core/scenario_engine.py`
- `core/agent_roles.py`
- `core/graph_initializer.py`
- `core/graph_traversal.py`

Design notes:

- `docs/core_system_design.md`
- `docs/ui_workspaces.md`

## Main API Surfaces

Workspace and controller:

- `/api/...`

Cognitive runtime:

- `/api/cognitive/...`

Key endpoints:

- `GET /api/cognitive/health`
- `POST /api/cognitive/chat/respond`
- `GET /api/cognitive/graph`
- `GET /api/cognitive/graph/subgraph`
- `GET /api/cognitive/graph/search`
- `POST /api/cognitive/ingest`
- `POST /api/cognitive/foundations/load`
- `GET /api/cognitive/sessions`
- `POST /api/cognitive/sessions`
- `POST /api/cognitive/style/learn`
- `GET /api/cognitive/style/profile`
- `GET /api/cognitive/agent-roles`
- `POST /api/cognitive/scenario/preview`

## Frontend Workspaces

The frontend is split into two workspaces:

1. `Chat`
   - session sidebar
   - scrollable message thread
   - fixed composer at the bottom

2. `Graph`
   - graph tools sidebar
   - relevant subgraph canvas
   - node editor / relation editor

The graph workspace renders only relevant subgraphs, not the entire graph.

## Run

Backend:

```bash
cd /home/karapet/agent_project
pip install -e .[dev]
python start.py --host 127.0.0.1 --port 8008
```

Frontend dev:

```bash
cd /home/karapet/agent_project
VITE_API_BASE_URL=http://127.0.0.1:8008 npm --prefix webapp run dev
```

Open:

- `http://127.0.0.1:5173`

## Tests

Backend:

```bash
cd /home/karapet/agent_project
PYTHONPATH=. roaches_viz/.venv/bin/python -m pytest -q tests/unit roaches_viz/tests style_tests
```

Frontend build:

```bash
cd /home/karapet/agent_project
npm --prefix webapp run build
```

## Git Hygiene

The repository is prepared so that git tracks source code and docs, but ignores runtime state:

- ignored:
  - local envs
  - model files
  - DB files
  - session files
  - style profiles
  - graph pending/verified JSON artifacts
  - runtime logs
  - project backups
- tracked:
  - source code
  - tests
  - docs
  - `.gitkeep` placeholders for runtime directories
