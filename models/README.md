# Local GGUF Models

Repository: `https://github.com/Karapet37/Diplom`

Primary discovery path:

- `models/gguf`

Role resolver can also scan `models/`, but `models/gguf` is preferred.

## Selection rules

1. Explicit env override has priority.
2. Otherwise resolver auto-discovers `*.gguf` files and assigns advisor roles.
3. Translator role is strict: translation uses only `translator` role model (no fallback to `general`).
4. If translator model is missing, translation prompt returns configuration error instead of using non-translator LLM.
5. Split GGUF is supported by entrypoint only: use `...-00001-of-0000N.gguf`.
6. Non-entry shards like `...-00002-of-0000N.gguf` are auto-remapped to shard `00001` when possible.

## Advisor roles and filename hints

Current auto-mapping by filename/path tokens:

- `translator`: `translator`, `translate`, `nllb`, `m2m`, `madlad`
- `coder_*`: `coder`, `code`, `codestral`, `codellama`, `starcoder`, `deepseek-coder`, `qwen-coder`, `programming`, `dev`
- `analyst`: `deepseek`, `analyst`, `reason`, `logic`
- `creative`: `danube`, `h2o`, `creative`, `story`
- `planner`: `planner`, `plan`, `instruct`
- `general`: best non-translator model (prefers instruct/general-purpose families)

These advisor roles are now used directly by the UI debate/personalization flow:

- `proposer` -> usually `creative`
- `critic` -> usually `analyst`
- `judge` -> usually `planner`

Personalization profile can override these defaults per request via `personalization.llm_roles`.

## Direct model-path usage (archive chat)

Project now supports explicit GGUF selection per request:

- `POST /api/project/archive/chat` can receive `model_path` (for example:
  - `models/gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`
  - `models/gguf/textGen/h2o-danube3-4b-chat-Q5_K_M.gguf`)
- user-facing answer is conversational (`assistant_reply`), not raw JSON.
- structured archive updates are reviewed separately via `POST /api/project/archive/review`.

Recommended explicit role assignment in current workspace:

- `LOCAL_GGUF_MODEL` -> `models/gguf/textGen/mistral-7b-instruct-v0.3-q4_k_m.gguf`
- `LOCAL_ANALYST_GGUF_MODEL` -> `models/gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`
- `LOCAL_CREATIVE_GGUF_MODEL` -> `models/gguf/textGen/h2o-danube3-4b-chat-Q5_K_M.gguf`
- `LOCAL_PLANNER_GGUF_MODEL` -> `models/gguf/textGen/mistral-7b-instruct-v0.3-q4_k_m.gguf`
- `LOCAL_CODER_*_GGUF_MODEL` -> `models/gguf/coder/qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf`
- `LOCAL_TRANSLATOR_GGUF_MODEL` -> `models/translator/model-q4k.gguf`

## Explicit env overrides

- `LOCAL_GGUF_MODEL=/absolute/path/to/general.gguf`
- `LOCAL_TRANSLATOR_GGUF_MODEL=/absolute/path/to/translator.gguf`
- `LOCAL_ANALYST_GGUF_MODEL=...`
- `LOCAL_CREATIVE_GGUF_MODEL=...`
- `LOCAL_PLANNER_GGUF_MODEL=...`
- `LOCAL_CODER_ARCHITECT_GGUF_MODEL=...`
- `LOCAL_CODER_REVIEWER_GGUF_MODEL=...`
- `LOCAL_CODER_REFACTOR_GGUF_MODEL=...`
- `LOCAL_CODER_DEBUG_GGUF_MODEL=...`

## Translator GGUF

Recommended translator model reference:

- https://huggingface.co/google/madlad400-3b-mt/blob/main/model-q4k.gguf

Recommended local placement:

- `models/gguf/madlad400-3b-mt/model-q4k.gguf`

Resolver treats `madlad400` as translator-priority inside translator role.

## Optional `.env` tuning

- `LOCAL_MODELS_DIR=models/gguf`
- `LOCAL_GGUF_N_CTX=2048`
- `LOCAL_GGUF_TEMPERATURE=0.25`
- `LOCAL_GGUF_MAX_TOKENS=220`
- `LOCAL_GGUF_MAX_LOADED=2`

## Split GGUF notes

- If your model files are split:
  - `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`
  - `qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf`
- Set env path to shard `00001` (or let resolver auto-remap from shard `00002`).
- Keep all shards in the same folder with unchanged names.

## Ollama

- Current local provider uses `llama_cpp` directly (`.gguf`) and does not require Ollama.
- Ollama can be added as a separate backend, but it is not required for this path.

## Notes

- System keeps running in deterministic fallback mode for non-translation flows even if GGUF models are missing.
- Translation flow requires translator role model when `translate_text` prompt is used.
