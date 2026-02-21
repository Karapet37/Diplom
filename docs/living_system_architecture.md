# Living System Architecture

## 1) Architecture Overview

The platform is restructured as a long-living modular organism under `src/living_system`.
Each layer is isolated, replaceable, and stateful through SQL.

Layers:

1. `core_engine` (`LivingSystemEngine`): orchestration and lifecycle control.
2. `monitoring` (`MonitoringService`): telemetry and heartbeat snapshots.
3. `knowledge_sql` (`KnowledgeSQLStore`): authoritative SQL state.
4. `embedding` (`HashEmbeddingService`): deterministic local embeddings.
5. `reasoning` (`SemanticReasoningService`): semantic extraction + entity linking.
6. `visualization` (`GraphVisualizationService`): state-to-view serialization.
7. `feedback` (`FeedbackService`): reinforcement and operator input.
8. `diagnostics` (`DiagnosticsService`): health/integrity/drift/uncertainty checks.
9. `recovery` (`RecoveryService`): snapshots, rollback, safe mode, override.
10. `evolution` (`EvolutionService`): versioning and improvement planning.
11. `prompt_brain` (`PromptBrain`): prompt memory + local GGUF coding assistant.
12. `universal_knowledge` (`UniversalKnowledgeAgent`): ontology extraction, contradiction tracking, branch/merge simulation.

## 2) Module Diagram (Textual)

```text
User/API
  -> GraphWorkspaceService (facade)
    -> LivingSystemEngine
      -> MonitoringService
      -> SemanticReasoningService
        -> HashEmbeddingService
        -> KnowledgeSQLStore
      -> PromptBrain
        -> KnowledgeSQLStore(prompts, prompt_runs, audit_actions)
        -> Local GGUF LLM (optional)
      -> DiagnosticsService
      -> RecoveryService
      -> EvolutionService
      -> GraphVisualizationService
      -> FeedbackService
      -> KnowledgeSQLStore (single source of truth)
```

## 3) Database Schema

Authoritative SQL schema is implemented in `src/living_system/knowledge_sql.py`.

Core tables:

- Graph: `nodes`, `node_properties`, `edges`
- Semantics: `embeddings`, `reasoning_traces`
- Runtime state: `snapshots`, `logs`, `health_checks`, `errors`
- User personalization: `users`, `user_profiles`, `localization_texts`
- Prompt brain: `prompts`, `prompt_runs`, `audit_actions`
- Recovery and evolution: `recovery_actions`, `versions`, `schema_versions`

Design principle: full graph + operations + diagnostics are reconstructable from SQL snapshots and history.

## 4) Core Algorithms

1. **Semantic Graph Write Path**
   1. Extract candidate entities from input text.
   2. Infer relation hint.
   3. Search existing SQL nodes for same user/type/name.
   4. Reuse existing node when found.
   5. Create new node only if confidence threshold passed.
   6. Mark low-confidence entities as `requires_confirmation`.
   7. Persist reasoning trace with confidence score.

2. **Failure Handling Path**
   1. Capture exception as `FailureRecord`.
   2. Upsert into `errors` (deduplicated by signature).
   3. Record telemetry log.
   4. Enter safe mode.
   5. Allow manual override and rollback.

3. **Recovery Path**
   1. Snapshot current SQL graph state.
   2. Persist parent-child snapshot lineage.
   3. On rollback: clear graph tables and restore from snapshot payload.
   4. Record action in `recovery_actions`.

## 5) Generated Code

New package:

- `src/living_system/models.py`
- `src/living_system/contracts.py`
- `src/living_system/knowledge_sql.py`
- `src/living_system/monitoring.py`
- `src/living_system/embedding.py`
- `src/living_system/reasoning.py`
- `src/living_system/visualization.py`
- `src/living_system/feedback.py`
- `src/living_system/diagnostics.py`
- `src/living_system/recovery.py`
- `src/living_system/evolution.py`
- `src/living_system/prompt_brain.py`
- `src/living_system/universal_knowledge.py`
- `src/living_system/core_engine.py`
- `src/living_system/__init__.py`

Integration points:

- `src/web/graph_workspace.py` (living-system facade methods)
- `src/web/api.py` (`/api/living/*` endpoints)

## 6) Test Suite

New tests are in `tests/unit/test_living_system.py` covering:

- SQL schema initialization and required tables.
- End-to-end input processing and reasoning persistence.
- Prompt brain create/update/delete file operations with policy and audit.
- Recovery snapshot + rollback graph restoration.

## 7) Monitoring System

`MonitoringService` captures:

- process memory and CPU usage,
- load average,
- heartbeat logs,
- component events with structured details.

All signals are persisted in `logs` and can be replayed for long-term diagnostics.

## 8) Recovery Strategy

`RecoveryService` implements:

- periodic/manual snapshots,
- rollback from snapshot id,
- safe mode gate,
- human override gate,
- recovery action history in SQL.

## 9) Evolution Plan

`EvolutionService` stores releases in `versions` and computes backlog from live metrics:

- low trace volume,
- insufficient health checks,
- snapshot frequency,
- error budget pressure,
- prompt brain usage volume.

## 10) Documentation and Inheritance Readiness

The design supports 10+ year maintainability by:

- strict layer boundaries,
- SQL-backed durable state,
- explainable reasoning traces,
- deterministic fallback embeddings,
- prompt template versioning,
- explicit recovery + diagnostics paths.

Inheritance checklist for future engineers:

1. Review `docs/living_system_architecture.md`.
2. Inspect schema in `src/living_system/knowledge_sql.py`.
3. Validate `tests/unit/test_living_system.py`.
4. Add new modules by implementing contracts in `src/living_system/contracts.py`.
5. Keep outputs localized through `localization_texts`, not hardcoded UI strings.
