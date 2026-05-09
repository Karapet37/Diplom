from __future__ import annotations

import json
import os
import re
from threading import Lock, Thread
from typing import Any, Callable

from .language_tools import detect_language_code, language_label, normalize_language_code
from .runtime_config import get_runtime_config
from src.utils.prompt_budgeter import SAFE_ERROR_REPLY
from src.utils.token_budget import select_n_ctx, token_count

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
    'analyze the request',
    '**analyze the request',
    'analysis of the situation',
    '# анализ ситуации',
    'ключевая проблема',
    'safety/policy',
    'constraint:',
    'output format:',
    'what i can say',
    'what i cannot',
    'что я могу сказать',
    'что я не могу',
    'что происходит',
    'мои ограничения',
    'что делать',
    'моя реакция',
    '**что',
    '**мои',
    'внешний ответ персонажа',
    'outer response',
    'the user has instructed me',
    'i must adhere strictly',
    'identity lock',
    'core principles',
    'below is my response structured',
    'respond in english',
    'persona head:',
    'user question:',
    'review notes',
    'issues identified',
    'мой ответ:',
    'my response:',
    'internal reasoning',
    'внутренние рассуждения',
)
_MODEL_CALL_LOCK = Lock()
_MODEL_META_LOCK = Lock()
_PREWARM_LOCK = Lock()
_PREWARM_STARTED = False
_THINK_BLOCK_RE = re.compile(r'<think>.*?</think>', flags=re.IGNORECASE | re.DOTALL)
_THINK_PREFIX_RE = re.compile(r'^\s*<think>.*?(?:</think>|$)', flags=re.IGNORECASE | re.DOTALL)
_CODE_FENCE_RE = re.compile(r'^```[a-zA-Z0-9_-]*\s*|\s*```$', flags=re.MULTILINE)
_LABELED_VISIBLE_REPLY_RE = re.compile(
    r'(?:внешний ответ персонажа|ответ персонажа|outer response(?: of the persona)?|corrected line|исправленный вариант реплики)\s*[:：]\s*(.+)',
    flags=re.IGNORECASE | re.DOTALL,
)
_HEADED_VISIBLE_REPLY_RE = re.compile(
    r'^\s*#+\s*answer\b\s*(.+?)(?=\n\s*(?:---|\*\*?\s*review notes?:|\*\*?\s*issues identified:|review notes?:|issues identified:)|$)',
    flags=re.IGNORECASE | re.DOTALL,
)
_LABELED_VISIBLE_REPLY_STOP_RE = re.compile(
    r'\n\s*(?:[#>*-]+\s*)?(?:внутренняя мысль|inner thought|safety/policy|constraint|output format|what i can say|what i cannot|что я могу сказать|что я не могу|review notes?|issues identified|role:|task:)\b',
    flags=re.IGNORECASE,
)
_LAST_MODEL_CALL_META: dict[str, dict[str, Any]] = {}


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


def _allowed_n_ctx_for_role(role: str) -> list[int]:
    role_key = str(role or 'general').strip().lower()
    if role_key == 'translator':
        return [1024, 1536, 2048]
    if role_key == 'analyst':
        return [1024, 1536, 2048, 3072, 4096, 7000]
    return [1024, 1536, 2048, 3072, 4096, 7000]


def _minimum_output_budget(mode: str, *, role: str) -> int:
    mode_key = str(mode or 'chat').strip().lower()
    role_key = str(role or 'general').strip().lower()
    if mode_key == 'translation':
        return 192
    if mode_key == 'knowledge':
        return 384 if role_key == 'analyst' else 448
    if mode_key == 'chat':
        return 200
    return 200


def plan_model_budget(
    prompt: str,
    *,
    mode: str = 'chat',
    role: str = 'general',
    n_ctx_override: int | None = None,
    max_tokens_override: int | None = None,
) -> dict[str, Any]:
    estimated_input_tokens = token_count(None, prompt)
    configured_n_ctx, configured_max_tokens = _mode_defaults(mode, role=role)
    if n_ctx_override is not None:
        configured_n_ctx = max(int(n_ctx_override or 0), 1024)
    if max_tokens_override is not None:
        configured_max_tokens = max(int(max_tokens_override or 0), 96)
    reserved_output_budget = max(int(configured_max_tokens or 0), _minimum_output_budget(mode, role=role))
    allowed_n_ctx = _allowed_n_ctx_for_role(role)
    target_total = estimated_input_tokens + reserved_output_budget
    n_ctx = select_n_ctx(max(int(configured_n_ctx or 0), target_total), allowed_n_ctx)
    max_prompt_tokens = max(int(n_ctx - reserved_output_budget), 0)
    window_shortfall_tokens = max(target_total - n_ctx, 0)
    budget_pressure_ratio = round((estimated_input_tokens + reserved_output_budget) / max(n_ctx, 1), 4)
    prompt_exceeds_window_budget = bool(estimated_input_tokens > max_prompt_tokens)
    return {
        'mode': str(mode or 'chat').strip().lower() or 'chat',
        'role': str(role or 'general').strip() or 'general',
        'estimated_input_tokens': int(estimated_input_tokens or 0),
        'configured_n_ctx': int(configured_n_ctx or 0),
        'n_ctx': int(n_ctx or 0),
        'reserved_output_budget': int(reserved_output_budget or 0),
        'actual_max_tokens': int(reserved_output_budget or 0),
        'max_prompt_tokens': int(max_prompt_tokens or 0),
        'prompt_nearly_fills_window': bool((estimated_input_tokens + reserved_output_budget) >= max(int(n_ctx * 0.9), reserved_output_budget)),
        'model_context_exhausted': prompt_exceeds_window_budget,
        'prompt_exceeds_window_budget': prompt_exceeds_window_budget,
        'logical_context_exhausted': prompt_exceeds_window_budget,
        'window_shortfall_tokens': int(window_shortfall_tokens or 0),
        'budget_pressure_ratio': budget_pressure_ratio,
        'n_ctx_auto_expanded': bool(int(n_ctx or 0) > int(configured_n_ctx or 0)),
    }


def _record_model_call_meta(mode: str, payload: dict[str, Any]) -> None:
    clean_mode = str(mode or 'chat').strip().lower() or 'chat'
    with _MODEL_META_LOCK:
        _LAST_MODEL_CALL_META[clean_mode] = dict(payload)


def get_last_model_call_meta(mode: str = 'chat') -> dict[str, Any]:
    clean_mode = str(mode or 'chat').strip().lower() or 'chat'
    with _MODEL_META_LOCK:
        return dict(_LAST_MODEL_CALL_META.get(clean_mode, {}))


def reset_last_model_call_meta(mode: str = 'chat') -> None:
    clean_mode = str(mode or 'chat').strip().lower() or 'chat'
    with _MODEL_META_LOCK:
        _LAST_MODEL_CALL_META.pop(clean_mode, None)


def _call_model(
    prompt: str,
    mode: str = 'chat',
    *,
    role: str = 'general',
    n_ctx_override: int | None = None,
    max_tokens_override: int | None = None,
) -> str:
    budget = plan_model_budget(
        prompt,
        mode=mode,
        role=role,
        n_ctx_override=n_ctx_override,
        max_tokens_override=max_tokens_override,
    )
    provider = _provider(role=role, n_ctx=int(budget.get('n_ctx') or 0), max_tokens=int(budget.get('actual_max_tokens') or 0))
    if provider is None:
        _record_model_call_meta(
            mode,
            {
                **budget,
                'provider_available': False,
                'prompt_tokens': int(budget.get('estimated_input_tokens') or 0),
                'input_truncated': False,
                'output_truncated': False,
                'output_budget_too_small': False,
                'finish_reason': '',
            },
        )
        return ''
    try:
        with _MODEL_CALL_LOCK:
            text = str(provider(prompt) or '').strip()
    except Exception:
        provider_meta = dict(getattr(provider, '_last_budget_meta', {}) or {})
        _record_model_call_meta(
            mode,
            {
                **budget,
                **provider_meta,
                'provider_available': True,
            },
        )
        return ''
    provider_meta = dict(getattr(provider, '_last_budget_meta', {}) or {})
    _record_model_call_meta(
        mode,
        {
            **budget,
            **provider_meta,
            'provider_available': True,
            'input_truncated': bool(provider_meta.get('truncated')),
            'output_truncated': bool(provider_meta.get('output_truncated')),
            'output_budget_too_small': bool(provider_meta.get('output_budget_too_small')),
            'finish_reason': str(provider_meta.get('finish_reason') or '').strip(),
        },
    )
    return text


def _call_model_compat(
    prompt: str,
    mode: str = 'chat',
    *,
    role: str = 'general',
    n_ctx_override: int | None = None,
    max_tokens_override: int | None = None,
) -> str:
    try:
        return _call_model(
            prompt,
            mode=mode,
            role=role,
            n_ctx_override=n_ctx_override,
            max_tokens_override=max_tokens_override,
        )
    except TypeError:
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


_JSON_RESP_KEY_RE = re.compile(
    r'"(?:response|reply|text|message|content|assistant_reply)"\s*:\s*"((?:[^"\\]|\\.)*)',
    re.DOTALL,
)

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?…])\s+')


def _dedup_repeated_sentences(text: str) -> str:
    """Truncate text when a sentence repeats 3+ times — classic small-LLM loop fix."""
    sentences = _SENT_SPLIT_RE.split(text.strip())
    seen: dict[str, int] = {}
    result: list[str] = []
    for s in sentences:
        key = s.strip().lower()
        if not key:
            result.append(s)
            continue
        count = seen.get(key, 0) + 1
        seen[key] = count
        if count <= 2:
            result.append(s)
        else:
            break
    return ' '.join(result).strip()


def normalize_text_reply(value: Any) -> str:
    raw = str(value or '').strip()
    if not raw or raw in {'{}', '[]', SAFE_ERROR_REPLY}:
        return ''
    payload = extract_json_block(raw)
    if isinstance(payload, dict):
        for key in ('assistant_reply', 'reply', 'text', 'message', 'content', 'response'):
            candidate = normalize_text_reply(payload.get(key))
            if candidate:
                return _dedup_repeated_sentences(candidate)
    # Fallback: regex extraction for truncated JSON — when the LLM ran out of tokens
    # mid-JSON and json.loads() fails, pull text from the "response" key directly.
    if raw.lstrip().startswith('{'):
        m = _JSON_RESP_KEY_RE.search(raw)
        if m:
            extracted = m.group(1).replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\').strip()
            if extracted:
                return extracted
    cleaned = _sanitize_visible_reply(raw)
    lowered = cleaned.lower()
    if not cleaned or any(marker in lowered for marker in PROMPT_LEAK_MARKERS):
        return ''
    return _dedup_repeated_sentences(cleaned)


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


def _extract_labeled_visible_reply(text: str) -> str:
    raw = str(text or '').strip()
    if not raw:
        return ''
    match = _LABELED_VISIBLE_REPLY_RE.search(raw)
    candidate = str(match.group(1) or '').strip() if match else ''
    if not candidate:
        headed = _HEADED_VISIBLE_REPLY_RE.search(raw)
        candidate = str(headed.group(1) or '').strip() if headed else ''
    if not candidate:
        return ''
    stop = _LABELED_VISIBLE_REPLY_STOP_RE.search(candidate)
    if stop:
        candidate = candidate[:stop.start()].strip()
    candidate = re.sub(r'\s*\([^()\n]{1,180}\)\s*$', '', candidate).strip()
    candidate = re.sub(r'^[\-\*\d\.\)\s]+', '', candidate).strip()
    return candidate


def _looks_like_visible_reply_candidate(paragraph: str) -> bool:
    lowered = str(paragraph or '').strip().lower()
    if not lowered:
        return False
    if any(marker in lowered for marker in VISIBLE_REPLY_LEAK_MARKERS):
        return False
    if lowered.startswith(('role:', 'task:', 'constraint:', 'output format:', 'safety/policy:')):
        return False
    return bool(re.search(r'[.!?…]|[А-ЯЁа-яё]{3,}|[A-Za-z]{3,}', paragraph))


def _sanitize_visible_reply(text: str) -> str:
    raw = str(text or '').strip()
    if not raw:
        return ''
    cleaned = _THINK_BLOCK_RE.sub(' ', raw)
    cleaned = _THINK_PREFIX_RE.sub(' ', cleaned)
    cleaned = _CODE_FENCE_RE.sub('', cleaned)
    cleaned = cleaned.replace('\r', '\n')
    labeled = _extract_labeled_visible_reply(cleaned)
    if labeled:
        cleaned = labeled
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
    if filtered and any(any(marker in paragraph.lower() for marker in VISIBLE_REPLY_LEAK_MARKERS) for paragraph in paragraphs[:2]):
        for paragraph in reversed(filtered):
            if _looks_like_visible_reply_candidate(paragraph):
                return paragraph.strip()
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
    # Never complain about missing context — just answer or stay present.
    target_language = normalize_language_code(language, fallback='en')
    base = 'I\'m here.' if not persona_selected else 'Go ahead.'
    if target_language == 'en':
        return base
    return translate_text(base, target_language=target_language, source_language='en') or base


def generate_chat_reply(
    prompt: str,
    *,
    language: str = 'en',
    persona_selected: bool = False,
    allow_builtin_fallback: bool = True,
    max_tokens_override: int | None = None,
    n_ctx_override: int | None = None,
    role_override: str | None = None,
) -> str:
    target_language = normalize_language_code(language, fallback='en')
    resolved_role = str(role_override or _chat_role(persona_selected=persona_selected)).strip() or _chat_role(persona_selected=persona_selected)
    reply = normalize_text_reply(
        _call_model_compat(
            prompt,
            mode='chat',
            role=resolved_role,
            max_tokens_override=max_tokens_override,
            n_ctx_override=n_ctx_override,
        )
    )
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
