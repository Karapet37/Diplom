# Agent Project

Stable persona graph agent runtime.

## Canonical Architecture

```text
chat (neck)
-> message analyzer
-> feature extractor
-> classifier forest
-> head caller
-> persona head
-> context builder
-> LLM
-> response
```

The LLM is only used for:

- knowledge extraction
- response generation

System logic, routing, head spawning, graph hygiene, and context ranking are deterministic Python code.

## Active Runtime

The repository now treats `agent_system/` as the single canonical backend. The packaged runtime stays below 25 Python modules and centers on:

- `agent_system/chat_engine.py`
- `agent_system/message_analyzer.py`
- `agent_system/feature_extractor.py`
- `agent_system/classifier_forest.py`
- `agent_system/head_caller.py`
- `agent_system/persona_engine.py`
- `agent_system/graph_store.py`
- `agent_system/context_builder.py`
- `agent_system/entity_extractor.py`
- `agent_system/file_ingestion.py`
- `agent_system/llm.py`
- `agent_system/api.py`

Legacy directories such as `roaches_viz/`, `src/living_system/`, and `src/autonomous_graph/` are retained only as archival reference material. They are not packaged and are not part of the active validation path.

## Head Storage

Each persona head is stored as a folder:

```text
memory/
  heads/
    dracula/
      traits.json
      relations.json
      examples.json
      knowledge.txt
      emotion_vector.json
      meta.json
```

`examples.json` contains both example utterances and `situation_reactions`.

## Knowledge Graph

Global graph memory is file-first:

```text
memory/
  graphs/
    nodes.json
    edges.json
  sessions/
    {session_id}.txt
  files/
    uploaded_documents/
      {session_id}/
        {filename}
  proposals/
    {head}.json
```

Graph hygiene applies on persistence:

- importance decay: `importance *= 0.99`
- duplicate merge using name similarity plus context similarity
- garbage collection when `importance < 0.05` and `frequency < 2`
- summary-node compression for repeated relation clusters

## Document Learning

Supported ingestion formats:

- `txt`
- `md`
- `json`
- `csv`

Chunks stay below 2000 estimated tokens before LLM extraction proposals are validated and merged.

## Runtime Profiles

The runtime is profile-driven and local-first.

Available profiles:

- `development`
- `local-demo`
- `local-heavy`
- `server`

Profile templates live in [config/runtime-profiles/development.yaml](/home/karapet/agent_project/config/runtime-profiles/development.yaml), [config/runtime-profiles/local-demo.yaml](/home/karapet/agent_project/config/runtime-profiles/local-demo.yaml), [config/runtime-profiles/local-heavy.yaml](/home/karapet/agent_project/config/runtime-profiles/local-heavy.yaml), and [config/runtime-profiles/server.yaml](/home/karapet/agent_project/config/runtime-profiles/server.yaml).

## Setup

```bash
cd <project_root>
./scripts/bootstrap_local.sh
```

This creates `.env.local` from [/.env.example](/home/karapet/agent_project/.env.example) when needed and installs backend/frontend dependencies.

If you build the combined UI locally:

```bash
cd <project_root>/webapp
npm run build
```

## Run

Recommended:

```bash
cd <project_root>
./scripts/run_profile.sh development
```

Other profiles:

```bash
./scripts/run_profile.sh local-demo
./scripts/run_profile.sh local-heavy
./scripts/run_profile.sh server
```

Direct startup:

```bash
python start.py --profile development
python start.py --profile server
python start.py --profile development --api-only
```

Useful diagnostics:

```bash
python start.py --list-profiles
python start.py --profile development --check
python start.py --profile development --print-config
```

## Configuration

The main runtime bootstrap order is:

```text
CLI flags -> shell environment -> env file -> runtime profile template -> code defaults
```

Common configurable paths:

- `COGNITIVE_MEMORY_ROOT`
- `COGNITIVE_WEBAPP_DIR`
- `COGNITIVE_WEBAPP_DIST_DIR`
- `LOCAL_MODELS_DIR`
- `LOCAL_*_GGUF_MODEL`

Paths are repo-relative unless made absolute, which keeps local deployment reproducible even when the process is launched from another directory.

## Documentation

Deployment and startup details are documented in [runtime_profiles.md](/home/karapet/agent_project/docs/runtime_profiles.md) and [runtime_flow.md](/home/karapet/agent_project/docs/runtime_flow.md).

## Tests

```bash
cd <project_root>
python3 -m pytest -q
```
