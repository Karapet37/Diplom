# Agent Project

Controller-first `Persona-Graph-Agent` runtime with structured persona memory, file-first graph storage, cognitive behavioral pipeline, and bounded local-LLM prompting.

## Core Idea

The project does not follow a direct `message -> answer` path.

Canonical flow:

```text
request
-> controller interpretation
-> route selection
-> capability plan
-> cognitive pipeline (P1–P6)
-> bounded context assembly
-> state transition / response shaping
-> generation
-> validation
-> repair
-> persistence
```

The `LLM` is not treated as the system itself. It is used only inside a deterministic runtime that decides:

- what kind of request this is,
- which route should handle it,
- what emotional/behavioral state the persona is in,
- which memory layers are needed,
- how much context may be packed,
- how the answer is validated,
- when regeneration or fallback is allowed.

## Active Backend

The canonical backend lives in `agent_system/`.

Core runtime modules:

- `agent_system/chat_engine.py`
- `agent_system/controller_runtime.py`
- `agent_system/request_pipeline.py`
- `agent_system/cognitive_pipeline.py`
- `agent_system/cognitive_authority.py`
- `agent_system/genome.py`
- `agent_system/genome_store.py`
- `agent_system/speech_planner.py`
- `agent_system/state_transition_runtime.py`
- `agent_system/head_caller.py`
- `agent_system/persona_engine.py`
- `agent_system/graph_store.py`
- `agent_system/context_builder.py`
- `agent_system/history_store.py`
- `agent_system/reliability.py`
- `agent_system/node_rethinker.py`
- `agent_system/behavioral_action_engine.py`
- `agent_system/behavioral_fallback.py`
- `agent_system/clarification_engine.py`
- `agent_system/llm.py`
- `agent_system/prompt_builder.py`
- `agent_system/observability.py`
- `agent_system/safety_classifier.py`
- `agent_system/localization_engine.py`
- `agent_system/social_roles.py`
- `agent_system/mood_research.py`
- `agent_system/situation_engine.py`
- `agent_system/situation_regulator.py`
- `agent_system/planning_engine.py`
- `agent_system/notes_store.py`
- `agent_system/importance_learner.py`
- `agent_system/file_ingestion.py`
- `agent_system/personality_schema.py`
- `agent_system/personality_store.py`
- `agent_system/training_examples_store.py`
- `agent_system/api.py`

## Cognitive Pipeline

The runtime includes a deterministic 6-stage cognitive pipeline that runs on every turn inside the heavy persona path.

**P1 — EventEncoder**: Encodes the incoming text into an event probability vector and intensity signal. Event types: `neutral`, `threat`, `reward`, `overload`, `shame_trigger`, `loss_of_control`, `failure`, `criticism`, `rejection`, `intimacy`, `opportunity`, `boredom`, `novelty`, `uncertainty`.

**P2 — TriggerNetwork**: Activates genome-derived trigger weights against the event vector and intensity. The `PersonalityGenome` encodes stable traits as float fields (fears, defense mechanisms, vulnerabilities, drive profile).

**P3 — RegulatorCell**: GRU-like cell that updates a 10-dimensional regulator state (anxiety, motivation, fatigue, shame, frustration, guilt, closeness, hope, emptiness) from trigger activations and the current genome.

**P4 — ThoughtMLP**: Produces a 16-dimensional thought vector from triggers and regulator state. Contains perceived risk, confidence, need dimensions (connection / achievement / safety), and frame dimensions (approach / hold / retreat).

**P5 — ConflictScorer**: Scores 8 resolution strategies (avoidance, overcompensation, attack, freeze, planning, support-seeking, self-deception) from the thought vector and genome defense profile.

**P6 — ActionPolicy**: MLP selecting one of 14 action families (approach, avoid, freeze, attack, placate, analyze, ask_for_help, seek_control, reduce_exposure, reframe, self_protect, connect, withdraw, plan_small_step) from thought, conflict, regulator state, and defense vector.

`CognitiveRuntime.__init__` uses a fixed seed for weight initialization, isolated from global numpy state, so results are deterministic regardless of test order.

### CognitiveAuthority

After the pipeline runs, `CognitiveAuthority` scores the output and selects a generation mode:

- `pure_llm` (score < 0.20): pipeline not active enough to guide generation; use legacy prompt
- `hint` (score < 0.55): inject a cognitive hint into `route_guidance`; LLM still drives the reply
- `planner` (score ≥ 0.55): full `SpeechPlanner` path

### SpeechPlanner

In `planner` mode, `SpeechPlanner.build()` converts `CognitiveTurnOutput` + built context into a structured `SpeechPlan`:

- `action_name`, `speech_goal`, `tone`
- `perceived_risk`, `confidence`, `intensity`
- `key_points` (ordered content directives derived from pipeline signals)
- `blocked_topics` (actions blocked by dominant resolution)
- `style_hints` (language, register, warmth)
- `max_tokens` (tighter under high risk or intensity)

`verbalizer_prompt(plan)` builds the final LLM prompt. PERSONA and RECENT EXCHANGE go to the system role; the SPEECH DIRECTIVE goes to the user role via the `"User question:"` separator.

## Request Model

The controller explicitly classifies requests before generation.

Supported request classes include:

- `factual_query`
- `roleplay_prompt`
- `persona_specification`
- `persona_assignment`
- `persona_analysis`
- `persona_chat`
- `document_request`
- `meta_previous_answer`
- `general_chat`
- `project_architecture_request`
- `clarification_request`

Main runtime routes:

- `factual_answer`
- `lightweight_conversation`
- `hypothetical_roleplay`
- `persona_chat_fast_path`
- `persona_specification`
- `persona_assignment`
- `persona_dialogue_analysis`
- `persona_graph_reasoning`
- `project_document_analysis`
- `meta_previous_answer`
- `clarification_request`

## Persona Architecture

Persona is stored as a structured object, not as prompt debris or freeform narrative.

High-level shape:

- `identity`
- `core_goal`
- `secondary_goals`
- `fears`
- `needs`
- `constraints_internal`
- `constraints_social`
- `constraints_hard_system`
- `allowed_methods`
- `maladaptive_methods`
- `core`
- `conflict`
- `defense`
- `behavior`
- `dynamics`
- `meta`

This separation is important because believable behavior is derived from:

```text
goal + fears + constraints + methods + current trigger
```

and not from loose style adjectives alone.

Persona registry hygiene is enforced. Junk entries such as file labels, ontology labels, prompt fragments, or behaviorless nouns are rejected and quarantined instead of entering the active persona pool.

Internal storage uses `memory/heads/`, while the operator-facing API exposes these objects under `/api/cognitive/personalities/...`.

## Personality Construction Layer

The project distinguishes between:

- validated runtime personas stored under `memory/heads/`,
- richer `PersonalityObject` records stored under `memory/personalities/`.

Those construction records keep:

- psychological profile fields separate from biography facts,
- per-entry provenance and confidence,
- stable vs temporary entries,
- uncertain hypotheses and conflict records,
- operator-facing update and override paths.

The API surface for this layer lives under `/api/cognitive/personality-construction/...` and supports creation, deletion, text/document updates, biography inspection, provenance review, conflict resolution, and behavior-model scoring.

## State Transition Runtime

`state_transition_runtime.py` orchestrates a series of LLM-guided enrichment stages that run between context building and generation:

- **state_reader**: reads the current persona state snapshot
- **persona_update**: updates emotion vector and situation reactions
- **bounded_state_transition**: selects next active role, risk posture, and mood signals
- **context_curator**: curates the working context layer
- **context_reviewer**: reviews context for contradictions and priorities
- **response_shaping**: shapes response style, behavior mode, and constraints

Each stage calls `call_json_model_for_role` and is individually gated by `COGNITIVE_STAGE_MODEL_STEPS`. All stages fall back gracefully to deterministic output when the LLM returns nothing useful.

## Reliability

`reliability.py` provides atomic rollback guarantees and runtime health reporting.

`StorageWriteFailure` — raised when a graph or persona write fails partway through. The caller must restore the previous snapshot. Both `graph_store.save_graph()` and `persona_engine.materialize_persona()` snapshot state before writing and roll back on failure.

`MutationRejectedFailure` — raised when a node rethink applies a description update but a subsequent graph mutation (e.g. connecting a new link) fails. The node description is rolled back and a snapshot path is included in `details`.

`runtime_status_snapshot()` — inspects the local LLM provider and returns a status dict with `mode` (`full` or `degraded`) and a list of `degraded_modes` with codes and summaries. Used as a pre-flight check before generation: if degraded, the system skips the LLM call entirely and routes to behavioral fallback.

## Thinking Model Support

The runtime supports thinking models (Qwen3.5-2B, Nanbeige4.1-3B) that emit internal reasoning blocks before their answer.

`_strip_think_blocks(text)` in `local_llm_provider.py` handles three output formats:

1. Full `<think>...</think>` block in output
2. Template-hidden format: `reasoning\n</think>\n\nAnswer` (chat template hides the opening tag)
3. Truncated unclosed block (context window exhausted during thinking)

In all cases the function returns only the answer portion. Stop token lists do not include `<think>` or `</think>` to avoid 1-token completions.

llama-cpp-python ≥ 0.3.17 is required for Qwen3.5 GGUF support (`qwen35` architecture).

## Degradation Detection

The generation functions in `chat_engine.py` check `runtime_status_snapshot()` **before** calling the LLM. If the runtime is degraded:

- the LLM call is skipped entirely,
- behavioral fallback is selected directly,
- `fallback_reason = 'dependency_unavailable'` is set,
- operator messages include `'local chat provider is unavailable'`.

This removes the fragile post-generation `reply == generic_fallback` string comparison that was previously used to detect degraded responses.

## Memory Layout

```text
memory/
  sessions/
    {session_id}.txt
    _route_state/
  notes/
    {session_id}.jsonl
  personalities/
    personalities_index.json
    {personality_id}.json
  training_examples/
    global.jsonl
    {session_id}.jsonl
  importance_learner/
    global_examples.jsonl
    {session_id}_examples.jsonl
  files/
    uploaded_documents/
      {session_id}/
  graphs/
    nodes.json
    edges.json
  heads/
    index.json
    {persona_slug}/
  archive/
```

Key principles:

- session history is persistent,
- manual notes are separate from graph memory and session logs,
- graph updates are merge-only,
- persona registry is validated before activation,
- personality construction keeps biography separate from profile inference,
- importance learning records save/skip signals but does not autosave turns,
- training examples are curated explicitly as `(input, correct_output)` pairs,
- session-first graph retrieval is preferred over unrelated global graph mass.

## Planning, Notes, and Behavioral Regulation

The chat runtime includes deterministic fast paths before the heavy persona/graph pipeline.

- `/save`, `/notes`, `/del_note`, and `/clear_notes` are executed directly at the top of `run_chat_turn` with no LLM call.
- planning mode builds a bounded structure from notes, personality summary, feedback class, and overload estimate before asking the model to verbalize it.
- the regulator pipeline classifies turn events and selects an action family before prompt construction, so the LLM phrases behavior instead of inventing it.
- crisis signals short-circuit to a canned supportive response, and blocked safety content never reaches the model.
- the importance learner observes saved vs unsaved turns, while `training_examples_store.py` keeps explicit fine-tuning examples and JSONL exports.

## File Ingestion

Supported formats:

- `txt`
- `md`
- `json`
- `csv`
- `pdf`
- `docx`
- `odt`
- `fb2`

Pipeline:

```text
file
-> ingestion
-> chunking
-> structured extraction
-> graph merge
-> session/global retrieval availability
```

`pdf` handling includes structural extraction so section-aware context can be used instead of only flat text dumps when possible.

## Context and Prompting

Context selection is density-first, not mass-first.

Packing order is effectively:

1. current request needs
2. current session evidence
3. active persona block
4. local graph evidence
5. only then relevant global graph evidence

Prompt construction uses compact packing with section budgets and reserved answer space. The system explicitly distinguishes:

- logical context assembly,
- model context window,
- output token reserve.

## Generation and Review

The runtime supports:

- `single`
- `primary_with_reviewer`
- `alternate`
- `randomized`

The reviewer pass is used for:

- route mismatch detection,
- truncation repair,
- persona style repair,
- invalid draft rewriting.

Fallbacks are reason-coded and not treated as silent success.

## Observability and Runtime Signals

Every chat request receives a trace with:

- request and route metadata,
- per-stage timings,
- context-token estimates,
- fallback usage and reason,
- model-budget metadata such as `n_ctx`, reserved output budget, and near-window warnings.

The runtime also records counters for graph writes, rebuild scheduling, rethink outcomes, and route-learning signals. These are exposed through debug endpoints and surfaced in the operator UI.

## Content Safety

A deterministic KNN-based safety classifier runs on every incoming message before LLM generation.

- No LLM involved — fully deterministic and explainable.
- Labels: `safe`, `suggestive`, `explicit`, `illegal`.
- Actions: `normal_response`, `soft_filter`, `blur_or_generalize`, `block`.
- Illegal content (e.g. child + sexual co-occurrence) is caught by a fast-path rule before KNN.
- Blocked messages never reach the LLM. The pipeline returns a canned refusal response.
- Examples can be added at runtime via `POST /api/cognitive/safety/examples` and persisted to `memory/safety_examples.jsonl`.

## Spirit-Based Localization

The localization engine does not translate. It shapes LLM verbalization behavior for a target language.

- Detects language from character set (Cyrillic → `ru`, Armenian → `hy`, etc.).
- Extracts voice axes from persona context: formality, warmth, edge, pace.
- Builds a voice guide injected into the prompt: register rules, avoid rules, rhythm notes.
- The model writes as a native speaker of the target language — it does not translate from English.
- Profiles are available for `ru`, `hy`, `en`, `es`, `fr` with sensible fallback to English.

## API Surface

Health and diagnostics:

- `GET /api/health`
- `GET /api/cognitive/health`
- `GET /api/cognitive/debug/metrics`
- `GET /api/cognitive/debug/traces`
- `GET /api/cognitive/debug/graph-health`
- `GET /api/cognitive/debug/runtime-status`

Sessions, chat, and files:

- `GET /api/cognitive/sessions`
- `POST /api/cognitive/sessions`
- `GET /api/cognitive/sessions/{session_id}`
- `DELETE /api/cognitive/sessions/{session_id}`
- `POST /api/cognitive/chat/respond`
- `POST /api/cognitive/files/upload`
- `POST /api/cognitive/rebuild`

Graph inspection and maintenance:

- `GET /api/cognitive/graph`
- `GET /api/cognitive/graph/snapshots`
- `POST /api/cognitive/graph/restore`
- `GET /api/cognitive/graph/subgraph`
- `GET /api/cognitive/graph/nodes/{node_id}/view`
- `POST /api/cognitive/graph/nodes`
- `DELETE /api/cognitive/graph/nodes/{node_id}`
- `POST /api/cognitive/graph/edges`
- `DELETE /api/cognitive/graph/edges/{edge_id}`
- `POST /api/cognitive/graph/nodes/merge`
- `POST /api/cognitive/graph/nodes/{node_id}/state`
- `POST /api/cognitive/graph/rethink`

Persona and personality construction:

- `GET /api/cognitive/personalities`
- `GET /api/cognitive/personalities/{name}`
- `GET /api/cognitive/personalities/{name}/graph-explanation`
- `GET /api/cognitive/personalities/{name}/revisions`
- `GET /api/cognitive/personalities/{name}/snapshots`
- `POST /api/cognitive/personalities/{name}/restore/{revision}`
- `GET /api/cognitive/personality-construction/personalities`
- `POST /api/cognitive/personality-construction/personalities`
- `GET /api/cognitive/personality-construction/personalities/{personality_id}`
- `DELETE /api/cognitive/personality-construction/personalities/{personality_id}`
- `POST /api/cognitive/personality-construction/personalities/{personality_id}/update-from-text`
- `POST /api/cognitive/personality-construction/personalities/{personality_id}/update-from-document`
- `POST /api/cognitive/personality-construction/personalities/{personality_id}/update-field`
- `POST /api/cognitive/personality-construction/personalities/{personality_id}/update-biography`
- `POST /api/cognitive/personality-construction/personalities/{personality_id}/override-entry`
- `GET /api/cognitive/personality-construction/personalities/{personality_id}/biography`
- `GET /api/cognitive/personality-construction/personalities/{personality_id}/provenance`
- `GET /api/cognitive/personality-construction/personalities/{personality_id}/conflicts`
- `POST /api/cognitive/personality-construction/personalities/{personality_id}/resolve-conflict`
- `POST /api/cognitive/personality-construction/score-actions`
- `GET /api/cognitive/personality-construction/personalities/{personality_id}/behavior-model`

Notes, planning, learning, and safety:

- `GET /api/cognitive/sessions/{session_id}/notes`
- `POST /api/cognitive/sessions/{session_id}/notes`
- `DELETE /api/cognitive/sessions/{session_id}/notes/{note_ref}`
- `DELETE /api/cognitive/sessions/{session_id}/notes`
- `GET /api/cognitive/sessions/{session_id}/notes/context`
- `POST /api/cognitive/planning/analyze`
- `POST /api/cognitive/planning/classify-feedback`
- `POST /api/cognitive/regulator/analyze`
- `POST /api/cognitive/regulator/classify-event`
- `GET /api/cognitive/sessions/{session_id}/importance-profile`
- `POST /api/cognitive/sessions/{session_id}/importance/score`
- `POST /api/cognitive/sessions/{session_id}/importance/suggest`
- `POST /api/cognitive/training-examples`
- `GET /api/cognitive/training-examples`
- `GET /api/cognitive/training-examples/{example_id}`
- `DELETE /api/cognitive/training-examples/{example_id}`
- `GET /api/cognitive/training-examples/export/jsonl`
- `POST /api/cognitive/safety/classify`
- `POST /api/cognitive/safety/examples`

## Web UI

The frontend in `webapp/` is an operator workspace focused on:

- session management,
- persona selection, inspection, and deletion,
- training-example curation and export,
- file upload,
- chat,
- graph inspection and maintenance,
- multilingual operator chrome (`en`, `ru`, `hy`, `zh`),
- debug and trace visibility.

See `webapp/README.md`.

## Models

Model discovery is local-first and prefers `models/gguf/`.

The runtime supports role-based model resolution, compatibility preflight, token budgeting, thinking-block stripping, and reviewer orchestration. See `models/README.md`.

Tested local models:

- `Qwen3.5-2B.Q4_K_M.gguf` — general and creative roles; thinking model
- `Nanbeige4.1-3B.Q3_K_M.gguf` — analyst and planner roles; thinking model
- Requires llama-cpp-python ≥ 0.3.17 for Qwen3.5 (`qwen35` architecture)

## Setup

```bash
cd <project_root>
./scripts/bootstrap_local.sh
```

Frontend build:

```bash
cd <project_root>/webapp
npm install
npm run build
```

## Run

Recommended:

```bash
cd <project_root>
./scripts/run_profile.sh development
```

Direct startup:

```bash
.venv/bin/python start.py --profile development
.venv/bin/python start.py --profile development --api-only
.venv/bin/python start.py --profile server
```

Diagnostics:

```bash
.venv/bin/python start.py --list-profiles
.venv/bin/python start.py --profile development --check
.venv/bin/python start.py --profile development --print-config
```

## Tests

Backend:

```bash
cd <project_root>
.venv/bin/python -m pytest -q tests/agent_system
```

System realism harness:

```bash
.venv/bin/python -m pytest -q tests/system_realism
```

Frontend build check:

```bash
npm --prefix webapp run build
```

Current test suite state:

- `tests/agent_system`: `382 tests collected, 382 passed` — full suite green
- Test files cover: routing, cognitive pipeline attractors, persona registry, graph lifecycle, graph hygiene, graph localizer, node rethinker, reliability and rollback, state transition runtime, behavioral fallback, social persona system, task procedures, trace learning, LLM runtime, local LLM provider policy, semantic routing, interaction routing, request pipeline, controller runtime, context pipeline, memory lifecycle, personality construction, planning and notes, behavior quality, file ingestion, API and failures, chat engine, and more.
- `tests/system_realism`: behavior depends on sandbox socket permissions

## Additional Documentation

- `READMEREPORT.md`
- `READMECONCLUSION.md`
- `models/README.md`
- `webapp/README.md`
- `packages/python-sdk/README.md`
- `packages/integration-layer-sdk/README.md`
- `tests/system_realism/README.md`
