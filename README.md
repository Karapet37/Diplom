# Agent Project

Controller-first `Persona-Graph-Agent` runtime with persona memory, graph-backed context, a deterministic cognitive pipeline, and an operator-facing annotation layer for message-vector correction.

## What This System Is

This project is not a thin `message -> LLM -> answer` wrapper.

The current system combines:

- controller-driven request interpretation and route selection,
- graph and history-backed persona runtime,
- deterministic `P1–P6` cognitive behavior pipeline for heavy persona turns,
- trainable message-vector interpretation with `main + extra` coordinates,
- operator-side correction capture through the web UI,
- local GGUF model orchestration with visible degradation handling,
- validation, repair, persistence, and trace logging.

Canonical flow:

```text
request
-> controller interpretation
-> route selection
-> capability plan
-> cognitive pipeline
-> context assembly
-> response shaping
-> generation
-> validation
-> repair
-> persistence
```

The `LLM` is used inside a bounded runtime. It does not decide the whole system behavior by itself.

## Runtime Overview

```
run_chat_turn()
  -> request envelope
  -> controller_runtime
  -> route decision + capability plan
  -> simple path OR persona path
  -> runtime pre-flight status check
  -> generation or deterministic fallback
  -> validation / repair
  -> history + trace persistence
```

Main high-level branches:

- `simple path`: lightweight conversation, factual turns, and direct bounded prompting
- `persona path`: persona selection, cognitive runtime, state-transition shaping, and structured verbalization

## Core Backend Modules

The canonical backend lives in `agent_system/`.

Primary orchestration and routing:

- `agent_system/chat_engine.py`
- `agent_system/controller_runtime.py`
- `agent_system/request_pipeline.py`
- `agent_system/context_builder.py`
- `agent_system/prompt_builder.py`
- `agent_system/llm.py`
- `agent_system/api.py`

Persona, graph, and persistence:

- `agent_system/head_caller.py`
- `agent_system/persona_engine.py`
- `agent_system/graph_store.py`
- `agent_system/history_store.py`
- `agent_system/personality_store.py`
- `agent_system/memory_layers.py`
- `agent_system/node_rethinker.py`
- `agent_system/reliability.py`

Cognitive and behavioral runtime:

- `agent_system/cognitive_pipeline.py`
- `agent_system/cognitive_authority.py`
- `agent_system/speech_planner.py`
- `agent_system/state_transition_runtime.py`
- `agent_system/behavioral_fallback.py`
- `agent_system/safety_classifier.py`

Message-vector and correction layer:

- `agent_system/message_vector_registry.py`
- `agent_system/message_vector_runtime.py`
- `agent_system/message_annotation_store.py`
- `agent_system/dataset_layer.py`
- `agent_system/cognitive_modules_v2.py`
- `agent_system/response_coherence_classifier.py`

## Cognitive Pipeline

Heavy persona turns run through a deterministic six-stage cognitive runtime:

- `P1 EventEncoder`
- `P2 TriggerNetwork`
- `P3 RegulatorCell`
- `P4 ThoughtMLP`
- `P5 ConflictScorer`
- `P6 ActionPolicy`

This pipeline produces structured internal state such as:

- `action_name`
- `dominant_resolution`
- `perceived_risk`
- `intensity`
- `thought_vec`
- `conflict_vec`
- `blocked_actions`

`CognitiveAuthority` then selects the generation mode:

- `pure_llm`
- `hint`
- `planner`

In `planner` mode, `SpeechPlanner` builds a structured `SpeechPlan`, and the model mainly verbalizes that plan instead of inventing behavior from scratch.

## Message-Vector Interpretation Layer

The project now also contains a separate message-vector layer used for annotation, correction, and learned interpretation.

Conceptually:

```text
message_t + context_matrix_t -> P-interpreters -> vector_t
```

Key properties:

- coordinates are not a single confidence score,
- each coordinate stores `main` plus `extra`,
- context is a matrix of previous message vectors, not a scalar flag,
- operator corrections are saved separately from ordinary chat history,
- those corrections can feed future retraining.

Current coordinate shape:

```json
{
  "P1": {
    "main": "statement",
    "extra": ["question"]
  }
}
```

The original target scheme was `P1..P49`. The current codebase extends that registry with additional structural coordinates in the live runtime, so the effective registry is slightly broader than the original 49-interpreter brief.

## Message and Annotation Data Model

User text and analysis text are now kept separate.

Important fields:

- `raw_text`: original user input, preserved without destructive rewriting
- `display_text`: text shown in the chat UI
- `analysis_text`: optional normalized copy for analysis-only paths

Typical annotation workspace row:

```json
{
  "message_id": "m17",
  "role": "assistant",
  "raw_text": "Well done, of course.",
  "display_text": "Well done, of course.",
  "analysis_text": "Well done, of course.",
  "context_window": ["m13", "m14", "m15", "m16"],
  "context_matrix_ref": "ctx_m17",
  "vector": {
    "P1": {"main": "statement", "extra": []},
    "P24": {"main": "false_praise", "extra": ["sarcasm"]}
  }
}
```

Correction storage is separate from ordinary session history:

- session-level annotations are stored under `memory/message_annotations/`
- global correction rows are appended to `memory/message_annotations/global.jsonl`
- runtime message-vector state is stored under `memory/message_vector_models/`

## Web Operator UI

The frontend in `webapp/` is an operator workspace, not just a chat shell.

Important current chat behavior:

- the selected persona in the dropdown is the primary speaker label source for assistant replies,
- the message composer clears immediately on submit,
- the user message appears in the thread optimistically before the backend reply returns,
- common prompt-leak scaffolding is stripped from displayed assistant replies,
- the chat surface can open an annotation workspace for message-vector correction.

The UI also exposes:

- session management,
- persona inspection,
- graph tools,
- diagnostics,
- file upload,
- training example curation,
- message annotation and context-matrix correction.

See [webapp/README.md](/home/karapet/agent_project/webapp/README.md:1) for the frontend surface map.

## API Highlights

Core chat and session endpoints:

- `GET /api/cognitive/sessions`
- `POST /api/cognitive/sessions`
- `GET /api/cognitive/sessions/{session_id}`
- `DELETE /api/cognitive/sessions/{session_id}`
- `POST /api/cognitive/chat/respond`

Annotation and correction endpoints:

- `GET /api/cognitive/sessions/{session_id}/annotation-workspace`
- `POST /api/cognitive/sessions/{session_id}/annotations`

Learning and diagnostics:

- `POST /api/cognitive/training-examples`
- `GET /api/cognitive/training-examples`
- `GET /api/cognitive/training-examples/export/jsonl`
- `POST /api/cognitive/safety/classify`
- `GET /api/cognitive/debug/metrics`
- `GET /api/cognitive/debug/traces`
- `GET /api/cognitive/debug/graph-health`

## Local Development

Backend:

```bash
.venv/bin/python start.py --profile development
```

API-only mode:

```bash
.venv/bin/python start.py --profile development --api-only
```

Frontend:

```bash
cd webapp
npm install
npm run dev
```

Production build:

```bash
cd webapp
npm run build
```

## Training and Dataset Utilities

Offline helpers:

- `scripts/train_pipeline.py`
  - calibrates `P1` and `P6` using dataset tuples
- `scripts/build_coordinate_datasets.py`
  - builds coordinate-vector datasets into `DataSets/coordinate_vectors/`

Supporting datasets and references:

- [DataSets/idea_attractors/README.md](/home/karapet/agent_project/DataSets/idea_attractors/README.md:1)
- [tests/system_realism/README.md](/home/karapet/agent_project/tests/system_realism/README.md:1)
- [models/README.md](/home/karapet/agent_project/models/README.md:1)

## Documentation Map

- [READMEREPORT.md](/home/karapet/agent_project/READMEREPORT.md:1) — detailed Armenian technical report
- [webapp/README.md](/home/karapet/agent_project/webapp/README.md:1) — operator UI and chat surface
- [models/README.md](/home/karapet/agent_project/models/README.md:1) — local GGUF model runtime
- [packages/python-sdk/README.md](/home/karapet/agent_project/packages/python-sdk/README.md:1) — optional Python integration-layer client
- [packages/integration-layer-sdk/README.md](/home/karapet/agent_project/packages/integration-layer-sdk/README.md:1) — optional JavaScript integration-layer client

## Notes

- Vendor documentation under `node_modules/` is not part of the project documentation set.
- The runtime is intentionally designed so that degraded model availability is visible, not silently hidden behind changing controller behavior.
- The correction layer is meant to improve the system through UI-mediated annotation, not by rewriting ordinary chat history in place.
