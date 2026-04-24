# Local GGUF Models

This directory documents the local model runtime used by the current controller-first system.

Preferred discovery root:

- `models/gguf/`

The resolver can also scan `models/`, but `models/gguf/` is the canonical location.

## Runtime Role Model

The local provider is role-based and local-first.

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

These roles are consumed by:

- controller-guided chat generation,
- persona heavy-path verbalization,
- reviewer and rewrite loops,
- translation,
- coding and analysis helper paths.

## Selection Rules

1. Explicit environment-variable overrides win.
2. Otherwise the resolver auto-discovers compatible `*.gguf` files under the configured model root.
3. Compatibility is checked before activation.
4. If a configured model is incompatible, the provider records the reason and falls back only to a compatible discovered model when possible.
5. Translator remains a dedicated role rather than silently reusing the general model.
6. Uncensored autodiscovery is opt-in through `LOCAL_ALLOW_UNCENSORED_AUTODISCOVERY`.
7. Split GGUF files are supported through the entry shard such as `...-00001-of-0000N.gguf`.

## Important Runtime Behavior

The current provider distinguishes between:

- configured model path,
- active compatible model path,
- runtime context budget,
- reserved output budget,
- compatibility failures,
- prompt-near-limit conditions.

This matters because a file may exist on disk and still be unusable for the current `llama-cpp-python` build or architecture support.

## Fast Verbalization Path

When the runtime already has a structured speech directive, it can bypass the higher-level chat wrapper and call raw `llama_cpp` completion directly.

Supported fast paths include:

- `qwen_fast_respond()`
- `mistral_fast_respond()`

These paths are used to:

- keep persona verbalization short and bounded,
- avoid visible reasoning blocks in the final reply,
- preserve output budget when the planner path already decided the content.

If the raw completion path fails or yields nothing useful, the runtime falls back to the regular `generate_chat_reply()` path.

## Thinking-Model Handling

Some supported local models emit visible reasoning scaffolds such as `<think>...</think>`.

`_strip_think_blocks()` removes:

- full `<think>...</think>` blocks,
- hidden-template leftovers,
- truncated unclosed think prefixes,
- plain `Thinking Process:` headers.

This cleanup happens before visible reply normalization so the operator chat thread does not display raw reasoning artifacts.

## Runtime Degradation

Missing or incompatible models should degrade the runtime visibly rather than silently changing controller behavior.

The runtime uses:

- compatibility preflight,
- `runtime_status_snapshot()`,
- explicit degraded-mode reporting,
- deterministic fallbacks when generation is unavailable.

This is important for operator trust: the system should show that it is degraded instead of pretending that nothing changed.

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
- `LOCAL_ANALYST_MAX_TOKENS`
- `LOCAL_TRANSLATOR_MAX_TOKENS`
- `LOCAL_CREATIVE_MAX_TOKENS`
- `LOCAL_PLANNER_MAX_TOKENS`
- `LOCAL_GGUF_THREADS`
- `LOCAL_GGUF_THREADS_BATCH`
- `LOCAL_GGUF_TEMPERATURE`
- `LOCAL_GGUF_TOP_P`
- `LOCAL_ALLOW_UNCENSORED_AUTODISCOVERY`

Chat/runtime orchestration:

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
- near-window warnings,
- truncation.

This helps distinguish:

1. logical context overload,
2. true model context-window exhaustion,
3. output budgets that are too small for the requested reply.

These signals are also surfaced through runtime traces so operators can inspect context and budget pressure after live turns.

## Notes

- The project uses direct local `GGUF` loading through `llama_cpp`; Ollama is not required for the main path.
- Explicit persona selection, planner verbalization, and response cleanup all rely on this same role-based provider layer.
- This directory documents the active runtime model layer, not a legacy research archive.
