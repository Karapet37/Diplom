from __future__ import annotations

import json
import os
import re
from threading import Lock, Thread
from typing import Any, Callable

from .language_tools import detect_language_code, language_label, normalize_language_code
from .runtime_config import get_runtime_config
from src.utils.prompt_budgeter import SAFE_ERROR_REPLY

PROMPT_LEAK_MARKERS = (
    'system instruction',
    'internal prompt',
    'return valid json only',
    'return plain text only',
    'do not output json',
)
VISIBLE_REPLY_LEAK_MARKERS = (
    '<think>',
    '</think>',
    'system note',
    'the user has instructed me',
    'i must adhere strictly',
    'identity lock',
    'core principles',
    'below is my response structured',
    'respond in english',
    'persona head:',
    'user question:',
)
_MODEL_CALL_LOCK = Lock()
_PREWARM_LOCK = Lock()
_PREWARM_STARTED = False
_THINK_BLOCK_RE = re.compile(r'<think>.*?</think>', flags=re.IGNORECASE | re.DOTALL)
_THINK_PREFIX_RE = re.compile(r'^\s*<think>.*?(?:</think>|$)', flags=re.IGNORECASE | re.DOTALL)
_CODE_FENCE_RE = re.compile(r'^```[a-zA-Z0-9_-]*\s*|\s*```$', flags=re.MULTILINE)


def _provider(role: str = 'general', *, n_ctx: int = 4096, max_tokens: int = 1400) -> Callable[[str], str] | None:
    try:
        from src.utils.local_llm_provider import build_role_llm_fn

        return build_role_llm_fn(role, n_ctx=n_ctx, max_tokens=max_tokens)
    except Exception:
        return None


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, '') or '').strip().lower()
    if not raw:
        return default
    return raw in {'1', 'true', 'yes', 'on'}


def _mode_defaults(mode: str, *, role: str = 'general') -> tuple[int, int]:
    return get_runtime_config().llm_window(mode, role=role)


def _call_model(prompt: str, mode: str = 'chat', *, role: str = 'general') -> str:
    n_ctx, max_tokens = _mode_defaults(mode, role=role)
    provider = _provider(role=role, n_ctx=n_ctx, max_tokens=max_tokens)
    if provider is None:
        return ''
    try:
        with _MODEL_CALL_LOCK:
            return str(provider(prompt) or '').strip()
    except Exception:
        return ''


def _call_model_compat(prompt: str, mode: str = 'chat', *, role: str = 'general') -> str:
    try:
        return _call_model(prompt, mode=mode, role=role)
    except TypeError:
        return _call_model(prompt, mode=mode)


def _preferred_role_for_mode(mode: str, *, preferred_role: str) -> str:
    n_ctx, max_tokens = _mode_defaults(mode, role=preferred_role)
    if _provider(role=preferred_role, n_ctx=n_ctx, max_tokens=max_tokens) is not None:
        return preferred_role
    config = get_runtime_config().roles
    fallbacks = [
        config.chat,
        'analyst',
        'general',
    ]
    for fallback_role in fallbacks:
        n_ctx, max_tokens = _mode_defaults(mode, role=fallback_role)
        if _provider(role=fallback_role, n_ctx=n_ctx, max_tokens=max_tokens) is not None:
            return fallback_role
    return preferred_role


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
    if not raw or raw in {'{}', '[]', SAFE_ERROR_REPLY}:
        return ''
    payload = extract_json_block(raw)
    if isinstance(payload, dict):
        for key in ('assistant_reply', 'reply', 'text', 'message', 'content', 'response'):
            candidate = normalize_text_reply(payload.get(key))
            if candidate:
                return candidate
    cleaned = _sanitize_visible_reply(raw)
    lowered = cleaned.lower()
    if not cleaned or any(marker in lowered for marker in PROMPT_LEAK_MARKERS):
        return ''
    return cleaned


def _strip_leading_leak_lines(text: str) -> str:
    lines = [line.rstrip() for line in str(text or '').splitlines()]
    kept: list[str] = []
    dropping = True
    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()
        if dropping and (
            not stripped
            or lowered in {'---', '###', '##'}
            or any(marker in lowered for marker in VISIBLE_REPLY_LEAK_MARKERS)
            or lowered.startswith('note:')
            or lowered.startswith('persona identity:')
            or lowered.startswith('core directive:')
        ):
            continue
        dropping = False
        kept.append(line)
    return '\n'.join(kept).strip()


def _sanitize_visible_reply(text: str) -> str:
    raw = str(text or '').strip()
    if not raw:
        return ''
    cleaned = _THINK_BLOCK_RE.sub(' ', raw)
    cleaned = _THINK_PREFIX_RE.sub(' ', cleaned)
    cleaned = _CODE_FENCE_RE.sub('', cleaned)
    cleaned = cleaned.replace('\r', '\n')
    cleaned = _strip_leading_leak_lines(cleaned)
    paragraphs = [part.strip() for part in re.split(r'\n\s*\n', cleaned) if part.strip()]
    filtered: list[str] = []
    for paragraph in paragraphs:
        lowered = paragraph.lower()
        if any(marker in lowered for marker in VISIBLE_REPLY_LEAK_MARKERS):
            continue
        if lowered.startswith('⚠️') and 'system' in lowered:
            continue
        filtered.append(paragraph)
    normalized = '\n\n'.join(filtered).strip()
    normalized = re.sub(r'\n{3,}', '\n\n', normalized)
    normalized = re.sub(r'[ \t]{2,}', ' ', normalized)
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {'"', "'"}:
        normalized = normalized[1:-1].strip()
    return normalized.strip()


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
            'Do not leave source-language fragments in the answer unless they are proper names or unavoidable technical terms.',
            f'Source language: {language_label(source)}.',
            f'Target language: {language_label(target)}.',
            'Text:',
            raw,
        ]
    )
    translated = normalize_text_reply(
        _call_model_compat(
            prompt,
            mode='translation',
            role=_preferred_role_for_mode('translation', preferred_role=role),
        )
    )
    return translated or raw


def _chat_role(*, persona_selected: bool = False) -> str:
    config = get_runtime_config().roles
    if persona_selected and config.use_general_for_persona_chat:
        return 'general'
    return config.chat


def chat_runtime_available(*, persona_selected: bool = False) -> bool:
    role = _chat_role(persona_selected=persona_selected)
    n_ctx, max_tokens = _mode_defaults('chat', role=role)
    return _provider(role=role, n_ctx=n_ctx, max_tokens=max_tokens) is not None


def fallback_chat_reply(*, language: str = 'en', persona_selected: bool = False) -> str:
    target_language = normalize_language_code(language, fallback='en')
    if persona_selected:
        base = 'I will answer in first person from the current persona graph and emotional state.'
    else:
        base = 'I do not have enough reliable context yet. Clarify the entity or add one fact.'
    if target_language == 'en':
        return base
    return translate_text(base, target_language=target_language, source_language='en') or base


def generate_chat_reply(
    prompt: str,
    *,
    language: str = 'en',
    persona_selected: bool = False,
    allow_builtin_fallback: bool = True,
) -> str:
    target_language = normalize_language_code(language, fallback='en')
    reply = normalize_text_reply(_call_model_compat(prompt, mode='chat', role=_chat_role(persona_selected=persona_selected)))
    if reply:
        detected_reply_language = normalize_language_code(detect_language_code(reply, fallback='en'), fallback='en')
        if target_language and detected_reply_language != target_language:
            translated = translate_text(
                reply,
                target_language=target_language,
                source_language=detected_reply_language or 'en',
            )
            return translated or reply
        return reply
    if allow_builtin_fallback:
        return fallback_chat_reply(language=target_language, persona_selected=persona_selected)
    return ''


def prewarm_runtime_models_async() -> None:
    global _PREWARM_STARTED
    if not _env_flag('COGNITIVE_PREWARM_ACTIVE_MODELS', False):
        return
    with _PREWARM_LOCK:
        if _PREWARM_STARTED:
            return
        _PREWARM_STARTED = True

    def _runner() -> None:
        runtime = get_runtime_config()
        roles = [
            (runtime.roles.chat, 'chat'),
            (runtime.roles.translation, 'translation'),
        ]
        seen: set[tuple[str, str]] = set()
        for role, mode in roles:
            key = (str(role or '').strip(), mode)
            if not key[0] or key in seen:
                continue
            seen.add(key)
            n_ctx, max_tokens = _mode_defaults(mode, role=role)
            try:
                from src.utils.local_llm_provider import prewarm_role_model

                prewarm_role_model(role=role, n_ctx=n_ctx, max_tokens=max_tokens)
            except Exception:
                continue

    Thread(target=_runner, name='agent-system-llm-prewarm', daemon=True).start()
