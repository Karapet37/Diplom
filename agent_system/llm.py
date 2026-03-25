from __future__ import annotations

import json
from typing import Any, Callable

from .language_tools import language_label, normalize_language_code
from .runtime_config import get_runtime_config

PROMPT_LEAK_MARKERS = (
    'system instruction',
    'internal prompt',
    'return valid json only',
    'return plain text only',
    'do not output json',
)


def _provider(role: str = 'general', *, n_ctx: int = 4096, max_tokens: int = 1400) -> Callable[[str], str] | None:
    try:
        from src.utils.local_llm_provider import build_role_llm_fn

        return build_role_llm_fn(role, n_ctx=n_ctx, max_tokens=max_tokens)
    except Exception:
        return None


def _mode_defaults(mode: str, *, role: str = 'general') -> tuple[int, int]:
    return get_runtime_config().llm_window(mode, role=role)


def _call_model(prompt: str, mode: str = 'chat', *, role: str = 'general') -> str:
    n_ctx, max_tokens = _mode_defaults(mode, role=role)
    provider = _provider(role=role, n_ctx=n_ctx, max_tokens=max_tokens)
    if provider is None:
        return ''
    try:
        return str(provider(prompt) or '').strip()
    except Exception:
        return ''


def _call_model_compat(prompt: str, mode: str = 'chat', *, role: str = 'general') -> str:
    try:
        return _call_model(prompt, mode=mode, role=role)
    except TypeError:
        return _call_model(prompt, mode=mode)


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
    return extract_json_block(_call_model_compat(prompt, mode='knowledge'))


def call_json_model_for_role(prompt: str, *, role: str = 'general') -> Any | None:
    return extract_json_block(_call_model_compat(prompt, mode='knowledge', role=role))


def translate_text(text: str, *, target_language: str, source_language: str = 'en', role: str = 'translator') -> str:
    raw = str(text or '').strip()
    if not raw:
        return ''
    source = normalize_language_code(source_language, fallback='en')
    target = normalize_language_code(target_language, fallback='en')
    if source == target:
        return raw
    prompt = '\n\n'.join(
        [
            'Return plain text only.',
            'Translate the text faithfully and naturally.',
            'Do not summarize, explain, or add commentary.',
            f'Source language: {language_label(source)}.',
            f'Target language: {language_label(target)}.',
            'Text:',
            raw,
        ]
    )
    translated = normalize_text_reply(_call_model_compat(prompt, mode='translation', role=role))
    return translated or raw


def _chat_role(*, persona_selected: bool = False) -> str:
    config = get_runtime_config().roles
    if persona_selected and config.use_general_for_persona_chat:
        return 'general'
    return config.chat


def fallback_chat_reply(*, language: str = 'en', persona_selected: bool = False) -> str:
    if persona_selected:
        if str(language or 'en').lower().startswith('ru'):
            return 'Я отвечу от первого лица на основе текущего графа личности и эмоционального состояния.'
        if str(language or 'en').lower().startswith('hy'):
            return 'Ես կպատասխանեմ առաջին դեմքով՝ հենվելով ընթացիկ persona գրաֆի և հուզական վիճակի վրա։'
        return 'I will answer in first person from the current persona graph and emotional state.'
    if str(language or 'en').lower().startswith('ru'):
        return 'Мне не хватает надежного контекста. Уточни сущность или добавь один факт.'
    if str(language or 'en').lower().startswith('hy'):
        return 'Ինձ դեռ չի բավականացնում վստահելի համատեքստը։ Հստակեցրու էությունը կամ ավելացրու մեկ փաստ։'
    return 'I do not have enough reliable context yet. Clarify the entity or add one fact.'


def generate_chat_reply(prompt: str, *, language: str = 'en', persona_selected: bool = False) -> str:
    reply = normalize_text_reply(_call_model_compat(prompt, mode='chat', role=_chat_role(persona_selected=persona_selected)))
    if reply:
        return reply
    return fallback_chat_reply(language=language, persona_selected=persona_selected)
