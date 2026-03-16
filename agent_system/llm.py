from __future__ import annotations

import json
from typing import Any, Callable

PROMPT_LEAK_MARKERS = (
    'system instruction',
    'internal prompt',
    'return valid json only',
    'return plain text only',
    'do not output json',
)


def _provider() -> Callable[[str], str] | None:
    try:
        from src.utils.local_llm_provider import build_role_llm_fn

        return build_role_llm_fn('general', n_ctx=4096, max_tokens=1400)
    except Exception:
        return None


def _call_model(prompt: str, mode: str = 'chat') -> str:
    provider = _provider()
    if provider is None:
        return ''
    try:
        return str(provider(prompt) or '').strip()
    except Exception:
        return ''


def extract_json_block(text: str) -> Any | None:
    raw = str(text or '').strip()
    if not raw:
        return None
    candidates = [raw]
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])
    start = raw.find('[')
    end = raw.rfind(']')
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def normalize_text_reply(value: Any) -> str:
    raw = str(value or '').strip()
    if not raw or raw in {'{}', '[]'}:
        return ''
    lowered = raw.lower()
    if any(marker in lowered for marker in PROMPT_LEAK_MARKERS):
        return ''
    payload = extract_json_block(raw)
    if isinstance(payload, dict):
        for key in ('assistant_reply', 'reply', 'text', 'message', 'content', 'response'):
            candidate = str(payload.get(key) or '').strip()
            if candidate and not any(marker in candidate.lower() for marker in PROMPT_LEAK_MARKERS):
                return candidate
    return raw


def call_json_model(prompt: str) -> Any | None:
    return extract_json_block(_call_model(prompt, mode='knowledge'))


def generate_chat_reply(prompt: str, *, language: str = 'en', persona_selected: bool = False) -> str:
    reply = normalize_text_reply(_call_model(prompt, mode='chat'))
    if reply:
        return reply
    if persona_selected:
        if str(language or 'en').lower().startswith('ru'):
            return 'Я отвечу от первого лица на основе текущего графа личности и эмоционального состояния.'
        return 'I will answer in first person from the current persona graph and emotional state.'
    if str(language or 'en').lower().startswith('ru'):
        return 'Мне не хватает надежного контекста. Уточни сущность или добавь один факт.'
    return 'I do not have enough reliable context yet. Clarify the entity or add one fact.'
