# Local GGUF Models

This directory documents the local model runtime used by the project.

Preferred discovery root:

- `models/gguf/`

The resolver can also scan `models/`, but `models/gguf/` is the canonical location.

## Runtime Principles

The model layer is role-based and local-first.

The runtime resolves models for roles such as:

- `general`
- `uncensored`
- `analyst`
- `creative`
- `planner`
- `translator`
- `coder_architect`
- `coder_reviewer`
- `coder_refactor`
- `coder_debug`

These roles are consumed by the controller runtime, chat orchestration, extraction flow, and reviewer loop.

## Selection Rules

1. Explicit environment variable overrides have highest priority.
2. Otherwise the resolver auto-discovers `*.gguf` files under the configured model root.
3. The resolver performs compatibility preflight before loading.
4. If a configured model is incompatible with the current local runtime, the provider logs the reason and falls back to a compatible discovered model when possible.
5. Translator role remains strict and should use a dedicated translation-capable model.
6. Uncensored autodiscovery is opt-in through `LOCAL_ALLOW_UNCENSORED_AUTODISCOVERY`.
7. Split GGUF files are supported through the entry shard (`...-00001-of-0000N.gguf`), and non-entry shards are remapped or rejected explicitly.

## Important Runtime Behavior

The local provider now distinguishes between:

- configured model path,
- active compatible model path,
- runtime context budget,
- output budget,
- compatibility failures.

This matters because a model file may exist but still be unusable with the current `llama-cpp-python` build or model architecture support.

## Useful Environment Variables

Model-path overrides:

- `LOCAL_MODELS_DIR`
- `LOCAL_GGUF_MODEL`
- `LOCAL_UNCENSORED_GGUF_MODEL`
- `LOCAL_ANALYST_GGUF_MODEL`
- `LOCAL_TRANSLATOR_GGUF_MODEL`
- `LOCAL_CREATIVE_GGUF_MODEL`
- `LOCAL_PLANNER_GGUF_MODEL`
- `LOCAL_CODER_ARCHITECT_GGUF_MODEL`
- `LOCAL_CODER_REVIEWER_GGUF_MODEL`
- `LOCAL_CODER_REFACTOR_GGUF_MODEL`
- `LOCAL_CODER_DEBUG_GGUF_MODEL`

Role-specific context windows:

- `LOCAL_GGUF_N_CTX`
- `LOCAL_UNCENSORED_N_CTX`
- `LOCAL_ANALYST_N_CTX`
- `LOCAL_TRANSLATOR_N_CTX`
- `LOCAL_CREATIVE_N_CTX`
- `LOCAL_PLANNER_N_CTX`
- `LOCAL_CODER_ARCHITECT_N_CTX`
- `LOCAL_CODER_REVIEWER_N_CTX`
- `LOCAL_CODER_REFACTOR_N_CTX`
- `LOCAL_CODER_DEBUG_N_CTX`

Budget and runtime tuning:

- `LOCAL_GGUF_MAX_TOKENS`
- `LOCAL_GGUF_THREADS`
- `LOCAL_GGUF_THREADS_BATCH`
- `LOCAL_GGUF_TEMPERATURE`
- `LOCAL_GGUF_TOP_P`
- `LOCAL_ALLOW_UNCENSORED_AUTODISCOVERY`

Chat orchestration controls:

- `COGNITIVE_CHAT_ORCHESTRATION`
- `COGNITIVE_CHAT_PRIMARY_ROLE`
- `COGNITIVE_CHAT_REVIEW_ROLE`
- `COGNITIVE_CHAT_REVIEW_MODE`

## Context and Budgeting

The runtime explicitly tracks:

- estimated input tokens,
- configured `n_ctx`,
- reserved output budget,
- actual `max_tokens`,
- prompt-near-limit conditions,
- truncation.

This makes it possible to distinguish:

1. logical context overload,
2. model context window exhaustion,
3. output budget that is too small.

The same budget metadata is surfaced through request traces, so operators can inspect `n_ctx`, reserved output budget, and near-window warnings after live turns.

## Translator Role

Translation uses a dedicated translator role.

If translator configuration is missing, the system reports that state instead of silently routing translation through an unrelated general model.

## Notes

- Ollama is not required for the current path; the project uses direct local `GGUF` loading through `llama_cpp`.
- A missing or incompatible model should degrade the runtime in a visible way, not silently change the controller logic.
- `models/PersonaAgentwGraphRAG-DE6F/README.md` is kept as an archival research reference, not the canonical description of the current runtime.
- Model configuration details are runtime concerns and are documented here rather than in the formal report/conclusion documents.
