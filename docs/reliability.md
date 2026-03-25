# Reliability and Degraded Modes

This document describes the reliability layer for the Persona-Graph Agent System. It is intentionally operational and focused on failure handling, storage safety, and recovery.

## 1. Failure classes

Runtime failures are represented by explicit classes in `agent_system/reliability.py`:

- `DependencyUnavailableFailure`
- `StorageWriteFailure`
- `MutationRejectedFailure`
- `RecoveryFailure`

Each failure is paired with a `FailurePolicy` containing:

- machine-readable `code`
- HTTP `status_code`
- `severity`
- expected `degraded_mode`
- suggested `operator_action`
- `retryable` flag

These failures are surfaced by `agent_system/api.py` through a structured JSON error payload instead of opaque 500 responses.

## 2. Degraded runtime modes

`runtime_status_snapshot()` checks the current runtime and returns:

- `mode = full | degraded`
- `degraded_modes[]`

Current degraded conditions include:

- frontend unavailable
- `llama_cpp` bindings unavailable
- required local LLM roles unavailable
- LLM diagnostics unavailable

This status is exposed through:

- `GET /api/cognitive/health`
- `GET /api/cognitive/debug/runtime-status`
- `GET /api/health`

In the chat path, if the chat provider is unavailable, the system returns a safe fallback reply immediately instead of waiting for a failing local model call.

## 3. Safe graph writes

Graph writes remain deterministic and validated. Before risky mutations, `GraphStore.save_graph()` can create a cold snapshot in `memory/archive/graphs/`.

Risky graph mutations include:

- create node
- patch node
- connect nodes
- delete node
- delete edge
- manual merge
- review state
- snapshot restore

Write behavior:

1. normalize and validate target state
2. optionally snapshot current graph
3. write `nodes.json`
4. write `edges.json`
5. if write fails, restore previous graph files
6. if rollback also fails, raise `RecoveryFailure`

Operator recovery:

- `GET /api/cognitive/graph/snapshots`
- `POST /api/cognitive/graph/restore`

## 4. Safe persona writes

Persona writes remain file-first, but risky persona mutations now use a guarded execution path.

Before risky persona mutations, the system captures:

- current persona file state
- cold persona snapshot in `memory/archive/heads/{head}/snapshots/`
- optional pre-mutation graph snapshot when the persona mutation also syncs graph state

If the mutation fails:

1. persona files are restored from the captured state
2. graph is restored from the pre-mutation snapshot when needed
3. a structured `StorageWriteFailure` or `RecoveryFailure` is raised

This guarded path is applied to:

- `materialize_persona(...)`
- `restore_persona_revision(...)`

Hot-path persona writes such as emotion updates and situation-reaction logging also use rollback protection, but without cold snapshot overhead.

Operator inspection and recovery:

- `GET /api/cognitive/personalities/{name}/revisions`
- `GET /api/cognitive/personalities/{name}/snapshots`
- `POST /api/cognitive/personalities/{name}/restore/{revision}`

## 5. Rethink apply safety

`node_rethinker.py` remains constrained: the LLM only suggests content improvements and link suggestions.

Before `preview_only = false` apply:

1. the current graph is snapshotted
2. deterministic node patch is applied
3. deterministic link creation is attempted
4. if mutation fails, the graph is restored from the snapshot

Translation/localization failures do not roll back graph mutations. They are downgraded to a non-fatal localization error because they do not threaten graph integrity.

## 6. Operator-visible fallback behavior

Fallback replies are explicit and structured.

Current fallback reasons include:

- `no_grounding`
- `dependency_unavailable`
- `model_fallback`

Chat responses can include:

- `runtime_status`
- `operator_messages`

This gives an operator enough context to understand whether the reply came from grounded generation or from a safe degraded path, without exposing hidden chain-of-thought.
