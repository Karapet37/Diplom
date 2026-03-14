from __future__ import annotations

import math
import re
from typing import Any


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_HEURISTIC_FACTOR = 1.35
_MIN_ALLOWED_N_CTX = 1024
_MAX_ALLOWED_N_CTX = 5000


def token_count(llm: Any, text: str) -> int:
    raw = str(text or "")
    if not raw.strip():
        return 0
    tokenizer = getattr(llm, "tokenize", None)
    if callable(tokenizer):
        encoded = raw.encode("utf-8", errors="ignore")
        for kwargs in ({"add_bos": False, "special": False}, {"add_bos": False}, {}):
            try:
                tokens = tokenizer(encoded, **kwargs)
                return len(tokens)
            except TypeError:
                continue
            except Exception:
                break
    return int(math.ceil(len(_TOKEN_RE.findall(raw)) * _HEURISTIC_FACTOR))


def truncate_to_fit(llm: Any, text: str, max_prompt_tokens: int) -> str:
    raw = str(text or "").strip()
    if not raw or max_prompt_tokens <= 0:
        return ""
    if token_count(llm, raw) <= max_prompt_tokens:
        return raw

    low = 0
    high = len(raw)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = raw[:middle].rstrip()
        current = token_count(llm, candidate)
        if current <= max_prompt_tokens:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best.rstrip()


def select_n_ctx(
    target_tokens: int,
    allowed_n_ctx_list: list[int] | tuple[int, ...] = (1024, 1536, 2048, 3072, 4096, 5000),
) -> int:
    allowed = sorted(
        {
            max(_MIN_ALLOWED_N_CTX, min(_MAX_ALLOWED_N_CTX, int(value)))
            for value in allowed_n_ctx_list
            if int(value) > 0
        }
    )
    if not allowed:
        allowed = [_MIN_ALLOWED_N_CTX, 1536, 2048, 3072, 4096, _MAX_ALLOWED_N_CTX]
    target = max(_MIN_ALLOWED_N_CTX, min(_MAX_ALLOWED_N_CTX, int(target_tokens or _MIN_ALLOWED_N_CTX)))
    for value in allowed:
        if value >= target:
            return value
    return allowed[-1]
