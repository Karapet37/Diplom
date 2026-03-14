from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .token_budget import select_n_ctx, token_count, truncate_to_fit


MIN_ROUTER_N_CTX = 1024
MAX_ROUTER_N_CTX = 2048
MIN_REASONING_N_CTX = 2048
MAX_REASONING_N_CTX = 5000
SAFE_ERROR_REPLY = "I couldn't produce a safe grounded response. Please retry with a shorter prompt."


@dataclass(frozen=True, slots=True)
class BoundedPrompt:
    prompt: str
    prompt_tokens: int
    token_budget: int
    reserve_tokens_for_output: int
    truncated: bool
    included_sections: int


def build_bounded_prompt(
    prompt_sections: list[str],
    token_budget: int,
    llm: Any,
    reserve_tokens_for_output: int,
) -> BoundedPrompt:
    normalized_budget = max(MIN_ROUTER_N_CTX, min(MAX_REASONING_N_CTX, int(token_budget or MIN_REASONING_N_CTX)))
    reserve = max(128, min(int(reserve_tokens_for_output or 256), normalized_budget - 128))
    max_prompt_tokens = max(256, normalized_budget - reserve)
    clean_sections = [str(section or "").strip() for section in prompt_sections if str(section or "").strip()]
    if not clean_sections:
        return BoundedPrompt(
            prompt="",
            prompt_tokens=0,
            token_budget=normalized_budget,
            reserve_tokens_for_output=reserve,
            truncated=False,
            included_sections=0,
        )

    prompt_parts: list[str] = []
    truncated = False
    included = 0
    for section in clean_sections:
        candidate = "\n\n".join([*prompt_parts, section]) if prompt_parts else section
        if token_count(llm, candidate) <= max_prompt_tokens:
            prompt_parts.append(section)
            included += 1
            continue
        if not prompt_parts:
            prompt_parts.append(truncate_to_fit(llm, section, max_prompt_tokens))
            truncated = True
            included = 1 if prompt_parts[0] else 0
        else:
            allowed = max_prompt_tokens - token_count(llm, "\n\n".join(prompt_parts))
            if allowed > 32:
                prompt_parts.append(truncate_to_fit(llm, section, allowed))
                if prompt_parts[-1]:
                    included += 1
            truncated = True
        break
    prompt = "\n\n".join(part for part in prompt_parts if part).strip()
    prompt_tokens = token_count(llm, prompt)
    if prompt_tokens > max_prompt_tokens:
        prompt = truncate_to_fit(llm, prompt, max_prompt_tokens)
        prompt_tokens = token_count(llm, prompt)
        truncated = True
    return BoundedPrompt(
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        token_budget=normalized_budget,
        reserve_tokens_for_output=reserve,
        truncated=truncated or included < len(clean_sections),
        included_sections=included,
    )


def retry_infer(
    llm_fn_builder: Callable[[str, int, int], Callable[[str], str] | None],
    role: str,
    prompt: str,
    token_budget: int,
    allowed_n_ctx_range: list[int] | tuple[int, ...],
    max_tokens: int,
) -> dict[str, Any]:
    allowed_contexts = sorted(
        {
            max(MIN_ROUTER_N_CTX, min(MAX_REASONING_N_CTX, int(value)))
            for value in allowed_n_ctx_range
            if int(value) > 0
        }
    )
    if not allowed_contexts:
        allowed_contexts = [select_n_ctx(token_budget)]
    requested_budget = select_n_ctx(token_budget, allowed_contexts)
    start_index = allowed_contexts.index(requested_budget)
    attempts: list[dict[str, Any]] = []
    current_prompt = str(prompt or "")
    for retry_round in range(2):
        for n_ctx in allowed_contexts[start_index:]:
            llm_fn = llm_fn_builder(role, n_ctx, max_tokens)
            if llm_fn is None:
                attempts.append({"n_ctx": n_ctx, "status": "builder_unavailable"})
                continue
            llm = getattr(llm_fn, "_llm", None)
            bounded = build_bounded_prompt([current_prompt], n_ctx, llm, max_tokens)
            attempts.append(
                {
                    "n_ctx": n_ctx,
                    "status": "ready",
                    "prompt_tokens": bounded.prompt_tokens,
                    "truncated": bounded.truncated,
                }
            )
            try:
                result = str(llm_fn(bounded.prompt) or "").strip()
            except Exception as exc:  # pragma: no cover - provider wrapper should shield this
                attempts[-1]["status"] = "error"
                attempts[-1]["error"] = str(exc)
                result = ""
            if result and result != SAFE_ERROR_REPLY:
                return {
                    "ok": True,
                    "text": result,
                    "n_ctx": n_ctx,
                    "prompt_tokens": bounded.prompt_tokens,
                    "truncated": bounded.truncated,
                    "attempts": attempts,
                }
            attempts[-1]["status"] = "retry"
        if retry_round == 0:
            current_prompt = truncate_to_fit(None, current_prompt, max(256, int(token_count(None, current_prompt) * 0.72)))
            start_index = 0
    return {
        "ok": False,
        "text": SAFE_ERROR_REPLY,
        "n_ctx": allowed_contexts[-1],
        "prompt_tokens": token_count(None, current_prompt),
        "truncated": True,
        "attempts": attempts,
    }
