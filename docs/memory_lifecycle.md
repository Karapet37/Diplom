# Layered Memory Lifecycle

This document describes the active layered-memory model of the Persona-Graph Agent System.

The system is not append-only memory. It uses layered memory with explicit lifecycle rules so that long-lived storage does not leak uncontrolled noise into the chat hot path.

## 1. Memory Layers

The active layers are:

```text
working memory
session memory
persona memory
graph knowledge memory
archive / cold memory
```

### Working Memory

- Role:
  ephemeral turn-local state used only during the current runtime pass.
- Storage:
  in-process transient state, typed request/analysis/context objects.
- Hot path:
  read/write allowed.
- Rule:
  never treated as durable knowledge.

### Session Memory

- Role:
  conversational continuity for one session.
- Storage:
  active tail in `memory/sessions/{session_id}.txt`.
- Hot path:
  read/write allowed.
- Rule:
  hot path reads only the active recent tail.

### Persona Memory

- Role:
  bounded stateful persona bundle.
- Storage:
  `memory/heads/{head_slug}/`.
- Hot path:
  read/write allowed.
- Rule:
  active persona files stay bounded; overflow is archived.

### Graph Knowledge Memory

- Role:
  validated long-lived structured knowledge.
- Storage:
  `memory/graphs/nodes.json` and `memory/graphs/edges.json`.
- Hot path:
  read/write allowed through deterministic validation only.
- Rule:
  active graph stays normalized by hygiene, merge, decay, and compression.

### Archive / Cold Memory

- Role:
  aged session turns, persona overflow, and graph snapshots.
- Storage:
  `memory/archive/`.
- Hot path:
  never read directly during ordinary chat context assembly.
- Rule:
  archive exists for auditability, migration safety, and offline maintenance.

## 2. Read / Write Rules

### Working Memory

- Read:
  current request lifecycle only.
- Write:
  transient turn state only.
- Forbidden:
  promoting content into durable memory without deterministic validation.

### Session Memory

- Read:
  `recent_dialogue()` and entity inference use active session memory only.
- Write:
  `append_turn()` writes to the active session file.
- Lifecycle:
  once active session length exceeds policy, older messages are archived and the active file is trimmed to the configured hot window.

### Persona Memory

- Read:
  active traits, examples, reactions, triad, and local graph only.
- Write:
  materialization, emotion updates, and reaction recording.
- Lifecycle:
  active persona memory is bounded by configured limits for traits, relations, examples, reactions, log tuples, and knowledge size.

### Graph Knowledge Memory

- Read:
  retrieval, context building, graph views, and local graph maintenance.
- Write:
  validated merge, edit, and hygiene paths only.
- Lifecycle:
  active graph memory uses decay, duplicate merge, garbage collection, and compression.

### Archive / Cold Memory

- Read:
  only by explicit maintenance, migration, or diagnostic workflows.
- Write:
  through session archival, persona overflow archival, and graph snapshotting.
- Forbidden:
  direct prompt packing from archive artifacts.

## 3. Retention and Compression Policies

### Session Retention

- Active session memory is hot and bounded.
- Older turns are moved into `memory/archive/sessions/{session_id}.json`.
- `parse_session(...)` remains backward-compatible by reconstructing the full session from archive + active tail.
- `recent_dialogue(...)` stays hot-path safe by reading only the active tail.

### Persona Retention

- Active persona files remain bounded by runtime-configured limits.
- Overflow beyond those limits is recorded in `memory/archive/heads/{head_slug}/overflow.json`.
- Repeated behavior is compressed into `log_tuples.json`.
- Active persona memory remains suitable for prompt grounding; archive memory is not injected automatically.

### Graph Retention

- Active graph memory remains the source of truth for retrieval.
- Graph hygiene provides decay, duplicate merge, garbage collection, and compression.
- Optional cold snapshots are written into `memory/archive/graphs/` through explicit maintenance calls such as `GraphStore.snapshot_graph(...)`.

## 4. Backward Compatibility

The migration is incremental:

- existing active paths stay unchanged:
  `memory/sessions/`, `memory/heads/`, `memory/graphs/`;
- archive storage is additive rather than destructive;
- full session reads remain available through `parse_session(...)`;
- hot-path reads remain bounded because `recent_dialogue(...)` and `infer_current_entity(...)` read only active session memory.

## 5. Maintenance / Migration Hooks

Minimal utility hooks currently available:

- `history_store.apply_session_memory_policy(session_id)`
  applies session archival policy to one session.
- `GraphStore.snapshot_graph(reason="...")`
  writes a cold graph snapshot without changing active graph state.
- persona overflow archival is automatic during active persona materialization when incoming payload exceeds configured bounds.

## 6. Design Constraint

The most important rule is:

```text
Archive memory is not part of normal chat grounding.
Only active bounded layers may contribute directly to context.
```

This prevents old noise, oversized persona payloads, and stale graph snapshots from contaminating the hot runtime path.
