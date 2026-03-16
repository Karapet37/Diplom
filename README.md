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

## Run

```bash
cd <project_root>
pip install -e .[dev]
python start.py --host 127.0.0.1 --port 8008
```

## Tests

```bash
cd <project_root>
python3 -m pytest -q
```
