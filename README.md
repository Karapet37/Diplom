# Agent Project

Minimal graph-grounded dialogue MVP.

The repository now converges to one active runtime:

`chat_engine -> llm -> session file -> knowledge_extractor -> memory/graphs -> context_builder -> next answer`

## Active Modules

Canonical business modules:

- `roaches_viz/roaches_viz/chat_engine.py`
- `roaches_viz/roaches_viz/llm.py`
- `roaches_viz/roaches_viz/history_store.py`
- `roaches_viz/roaches_viz/knowledge_extractor.py`
- `roaches_viz/roaches_viz/graph_store.py`
- `roaches_viz/roaches_viz/context_builder.py`
- `roaches_viz/roaches_viz/api.py`

Thin bootstrap:

- `start.py`
- `src/web/combined_app.py`
- `src/web/api.py`

Frontend:

- `webapp/`

## File-First Memory

```text
memory/
  sessions/
    {session_id}.txt
  files/
    uploaded_documents/
      {session_id}/
        {safe_filename}
  personalities/
    index.json
    {name}.json
    {name}_graph.json
    proposals/
      {name}.json
  graphs/
    nodes.json
    edges.json
```

## Runtime Rules

1. User sends a message.
2. `chat_engine.py` loads:
   - recent session text
   - selected personality file, if any
   - graph context through `context_builder.py`
3. `llm.py` builds the prompt and returns:
   - plain text reply
   - optional structured proposal JSON
4. `history_store.py` appends the exchange to `memory/sessions/{session_id}.txt`
5. Background extraction reads session files and uploaded files.
6. `knowledge_extractor.py` generates validated graph proposals and merges them into:
   - `memory/graphs/nodes.json`
   - `memory/graphs/edges.json`
7. If a personality/entity is referenced but not found, the system writes:
   - `memory/personalities/proposals/{name}.json`
8. `knowledge_extractor.py` validates and materializes:
   - `memory/personalities/{name}.json`
   - `memory/personalities/{name}_graph.json`
9. Next replies may use graph and personality context.

## Critical Contract

- The LLM never writes graph files directly.
- The LLM only returns proposals.
- The extractor validates proposals before merge.
- Graph files are never overwritten with empty data.

## Graph UI Contract

For each node, the graph workspace answers:

1. who / what is this node
2. what it is like
3. how it acts through relations

## Run

Backend:

```bash
cd <project_root>
pip install -e .[dev]
python start.py --host 127.0.0.1 --port 8008
```

Frontend:

```bash
cd <project_root>
VITE_API_BASE_URL=http://127.0.0.1:8008 npm --prefix webapp run dev
```

Open:

- `http://127.0.0.1:5173`

## Tests

Backend:

```bash
cd <project_root>
PYTHONPATH=. roaches_viz/.venv/bin/python -m pytest -q
```

Frontend build:

```bash
cd <project_root>
npm --prefix webapp run build
```

## Personality Proposal Behavior

If the system detects a personality and cannot find it in graph or files, it must do this:

`вижу личность, ищу в графе, не нашел, запрашиваю создание файла анкеты под его описание`

The user-facing fallback is:

- `Контекст личности не найден. Запрошено создание анкеты по описанию.`
