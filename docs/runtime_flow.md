# Persona-Graph Runtime Flow

## 1. Active runtime path

The active product runtime starts here:

```text
start.py --profile <name> [--env-file ...] [--config ...]
  -> bootstrap_runtime_environment()
  -> get_runtime_config()
  -> src.web.combined_app.create_combined_app()
    -> agent_system.api.create_app()
      -> /api/cognitive/chat/respond
        -> agent_system.chat_engine.run_chat_turn()
```

This is the current hot path. The main runtime modules on that path are:

- `agent_system/runtime_config.py`
- `agent_system/api.py`
- `agent_system/chat_engine.py`
- `agent_system/message_analyzer.py`
- `agent_system/situation_engine.py`
- `agent_system/feature_extractor.py`
- `agent_system/classifier_forest.py`
- `agent_system/head_caller.py`
- `agent_system/persona_engine.py`
- `agent_system/context_builder.py`
- `agent_system/prompt_builder.py`
- `agent_system/llm.py`
- `src/utils/local_llm_provider.py`
- `src/utils/prompt_budgeter.py`
- `agent_system/history_store.py`
- `agent_system/graph_store.py`

Legacy or non-hot-path modules remain in the repo but are not part of the default request lifecycle:

- `src/living_system/`
- `src/autonomous_graph/`
- `roaches_viz/`
- historical storages outside `memory/heads/` and `memory/graphs/`

## 2. Centralized runtime configuration

`agent_system/runtime_config.py` is the centralized source for:

- runtime profile bootstrapping
- runtime paths
- context budgets
- model roles
- feature flags
- rebuild cadence
- rethink limits
- retry behavior for local inference

The goal is not to hide behavior, but to make runtime assumptions explicit in one place.

## 3. Typed runtime contracts

The core chat path now uses typed contracts from `agent_system/models.py`:

- `ChatTurnRequest`
- `UserState`
- `Situation`
- `ChatSideEffects`
- `ChatTurnResult`

This replaces implicit dict-passing at the major runtime boundaries while preserving the external API payload shape.

## 4. Exact request lifecycle

### 4.1 API entry

`agent_system/api.py` receives `POST /api/cognitive/chat/respond` and forwards the request to `chat_engine.generate_response()`.

`generate_response()` is the compatibility layer.
`run_chat_turn()` is the typed internal runtime path.

### 4.2 Chat orchestration

`chat_engine.run_chat_turn()` performs the following steps in order:

1. Create or load the session.
2. Optionally run deterministic concept-graph extraction on the raw user message.
3. Analyze the message into `UserState` and entity candidates.
4. Convert user analysis into a typed `Situation`.
5. Extract deterministic entity features.
6. Classify entity types with the classifier forest.
7. Upsert graph nodes and decide whether a persona head should exist.
8. Select the primary persona head.
9. Update persona emotion from `Situation`, not from raw user emotion.
10. Build bounded context from persona, graph, recent dialogue, and situation.
11. Build the final prompt.
12. Call the local LLM through the narrow adapter.
13. Write the dialogue turn to session history.
14. Write persona `situation_reaction`.
15. Decide whether to schedule background rebuild work.
16. Return a typed `ChatTurnResult`, then serialize it for the API response.

## 5. Explicit side effects

The runtime now treats write-side effects as explicit steps.

### 5.1 Graph writes

Graph writes happen only through deterministic code:

- concept graph premerge in `chat_engine`
- extraction merge in `entity_extractor` / `file_ingestion`
- validated graph editing in `graph_store`
- constrained rethink apply path in `node_rethinker`

`LLM` may suggest content, but it does not mutate the graph directly.

### 5.2 History writes

`history_store.append_turn()` writes `memory/sessions/{session_id}.txt` after the assistant reply is produced.

### 5.3 Persona writes

`persona_engine` is responsible for:

- emotion vector updates
- situation-reaction memory updates
- persona materialization
- triad updates (`log_tuples`, `persona_form`, `decision_explanation`)

Persona emotion remains situation-based:

```text
persona_emotion = f(persona_traits, situation)
NOT f(user_emotion)
```

### 5.4 Rebuild scheduling

Background rebuild scheduling is explicit and policy-driven.
It is not triggered blindly on every turn.

The scheduler decides based on:

- pending persona proposals
- periodic hygiene interval
- periodic persona synchronization interval
- runtime feature flags
- already-pending rebuild state

## 6. Stability boundary

The architectural boundary is unchanged:

```text
LLM is a subordinate inference module.
Deterministic code owns routing, validation, storage writes, graph mutations, and persona state transitions.
```

This refactor stabilizes that boundary without changing the product into an LLM-controlled system.
