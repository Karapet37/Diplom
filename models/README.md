# Local GGUF Models

Primary discovery path:

- `models/gguf`

Role resolver can also scan `models/`, but `models/gguf` is preferred.

## Selection rules

1. Explicit env override has priority.
2. Otherwise resolver auto-discovers `*.gguf` files and assigns advisor roles.
3. Translator role is strict: translation uses only `translator` role model (no fallback to `general`).
4. If translator model is missing, translation prompt returns configuration error instead of using non-translator LLM.

## Advisor roles and filename hints

Current auto-mapping by filename/path tokens:

- `translator`: `translator`, `translate`, `nllb`, `m2m`, `madlad`
- `coder_*`: `coder`, `code`, `codestral`, `codellama`, `starcoder`, `deepseek-coder`, `qwen-coder`, `programming`, `dev`
- `analyst`: `deepseek`, `analyst`, `reason`, `logic`
- `creative`: `danube`, `h2o`, `creative`, `story`
- `planner`: `planner`, `plan`, `instruct`
- `general`: best non-translator model (prefers instruct/general-purpose families)

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
- `LOCAL_GGUF_N_CTX=8192`
- `LOCAL_GGUF_TEMPERATURE=0.25`
- `LOCAL_GGUF_MAX_TOKENS=220`
- `LOCAL_GGUF_MAX_LOADED=2`

## Notes

- System keeps running in deterministic fallback mode for non-translation flows even if GGUF models are missing.
- Translation flow requires translator role model when `translate_text` prompt is used.
