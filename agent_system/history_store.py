from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .duplicate_resolver import normalize_name
from .memory_layers import append_session_archive, load_session_archive
from .runtime_config import get_runtime_config

_ENTITY_PATTERNS = (
    r'"([^"]{2,80})"',
    r"'([^']{2,80})'",
    r'\b(?:about|like|as|with|to|for|regarding|around|про|о|как)\s+([A-Za-zА-Яа-я][\w-]*(?:\s+[A-Za-zА-Яа-я][\w-]*){0,3})',
)

_ENTITY_STOPWORDS = {
    'a',
    'an',
    'the',
    'me',
    'you',
    'it',
    'this',
    'that',
    'someone',
    'something',
    'who',
    'what',
    'why',
    'how',
    'about',
    'like',
    'with',
    'to',
    'for',
    'про',
    'как',
    'что',
    'кто',
}


def sessions_dir() -> Path:
    return get_runtime_config().paths.sessions_dir


def files_dir() -> Path:
    return get_runtime_config().paths.uploaded_documents_dir


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _safe_session_id(session_id: str) -> str:
    clean = ''.join(char if char.isalnum() or char in {'-', '_'} else '-' for char in str(session_id or '').strip())
    return clean or f"session-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"


def session_text_path(session_id: str) -> Path:
    return sessions_dir() / f'{_safe_session_id(session_id)}.txt'


def session_files_dir(session_id: str) -> Path:
    path = files_dir() / _safe_session_id(session_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_session(session_id: str = '', title: str = '') -> dict[str, Any]:
    clean = _safe_session_id(session_id)
    path = session_text_path(clean)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if title.strip():
            path.write_text(f'# {title.strip()}\n', encoding='utf-8')
        else:
            path.touch()
    parsed = parse_session(clean)
    if parsed is not None:
        return parsed
    return {
        'session_id': clean,
        'title': title.strip() or 'New session',
        'messages': [],
        'updated_at': _utc_now(),
        'path': str(path),
    }


def _render_session_text(title: str, messages: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    if str(title or '').strip():
        blocks.append(f'# {str(title).strip()}')
    for item in list(messages or []):
        role = str(item.get('role') or '').strip()
        message = str(item.get('message') or '').strip()
        timestamp = str(item.get('timestamp') or '').strip()
        if not role or not message:
            continue
        if timestamp:
            blocks.append(f'[{timestamp}]')
        blocks.append(f'{role}: {message}')
        blocks.append('')
    return '\n'.join(blocks).strip() + ('\n' if blocks else '')


def _parse_session_path(path: Path, *, session_id: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding='utf-8')
    title = ''
    timestamp = ''
    messages: list[dict[str, str]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('# '):
            title = stripped[2:].strip()
            continue
        if stripped.startswith('[') and stripped.endswith(']'):
            timestamp = stripped[1:-1]
            continue
        if ':' not in stripped:
            continue
        role, message = stripped.split(':', 1)
        if role in {'user', 'assistant'}:
            messages.append({'role': role, 'message': message.strip(), 'timestamp': timestamp})
    resolved_title = title or next((item['message'][:60] for item in messages if item['role'] == 'user'), 'New session')
    updated_at = timestamp or datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
    return {'session_id': _safe_session_id(session_id), 'title': resolved_title, 'messages': messages, 'updated_at': updated_at, 'path': str(path)}


def parse_active_session(session_id: str) -> dict[str, Any] | None:
    return _parse_session_path(session_text_path(session_id), session_id=session_id)


def _apply_session_retention(session_id: str) -> None:
    config = get_runtime_config().memory
    parsed = parse_active_session(session_id)
    if not parsed:
        return
    messages = list(parsed.get('messages') or [])
    archive = load_session_archive(session_id)
    has_archive = bool(list(archive.get('archived_messages') or []))
    threshold = config.session_keep_recent_messages if has_archive else config.session_archive_after_messages
    if len(messages) <= threshold:
        return
    keep_count = min(max(config.session_keep_recent_messages, 0), len(messages))
    cutoff = max(len(messages) - keep_count, 0)
    archived_messages = messages[:cutoff]
    active_messages = messages[cutoff:]
    if not archived_messages or len(active_messages) >= len(messages):
        return
    append_session_archive(
        session_id,
        title=str(parsed.get('title') or ''),
        messages=archived_messages,
        reason='session_hot_window_trim',
    )
    session_text_path(session_id).write_text(
        _render_session_text(str(parsed.get('title') or ''), active_messages),
        encoding='utf-8',
    )


def apply_session_memory_policy(session_id: str) -> dict[str, Any]:
    _apply_session_retention(session_id)
    parsed = parse_session(session_id)
    archive = load_session_archive(session_id)
    return {
        'session_id': _safe_session_id(session_id),
        'active_message_count': int((parsed or {}).get('active_message_count') or 0),
        'archived_message_count': len(list(archive.get('archived_messages') or [])),
        'archive_events': list(archive.get('archive_events') or []),
    }


def append_turn(session_id: str, user_message: str, assistant_message: str) -> Path:
    path = session_text_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    blocks: list[str] = []
    if str(user_message or '').strip():
        blocks.append(f'[{_utc_now()}]\nuser: {str(user_message).strip()}')
    if str(assistant_message or '').strip():
        blocks.append(f'[{_utc_now()}]\nassistant: {str(assistant_message).strip()}')
    if not blocks:
        return path
    current = path.read_text(encoding='utf-8') if path.exists() else ''
    payload = (current.rstrip() + '\n\n' if current.strip() else '') + '\n\n'.join(blocks) + '\n'
    path.write_text(payload, encoding='utf-8')
    _apply_session_retention(session_id)
    return path


def parse_session(session_id: str, *, include_archived: bool = True) -> dict[str, Any] | None:
    active = parse_active_session(session_id)
    archive = load_session_archive(session_id) if include_archived else {'archived_messages': [], 'title': '', 'updated_at': ''}
    if active is None and not list(archive.get('archived_messages') or []):
        return None
    active_messages = list(active.get('messages') or []) if active else []
    archived_messages = [dict(item) for item in list(archive.get('archived_messages') or []) if isinstance(item, dict)]
    messages = archived_messages + active_messages
    title = str((active or {}).get('title') or archive.get('title') or '').strip()
    if not title:
        title = next((str(item.get('message') or '')[:60] for item in messages if str(item.get('role') or '').strip() == 'user'), 'New session')
    updated_at = str((active or {}).get('updated_at') or archive.get('updated_at') or _utc_now()).strip()
    return {
        'session_id': _safe_session_id(session_id),
        'title': title,
        'messages': messages,
        'updated_at': updated_at,
        'path': str(session_text_path(session_id)),
        'active_message_count': len(active_messages),
        'archived_message_count': len(archived_messages),
    }


def list_sessions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(sessions_dir().glob('*.txt'), key=lambda item: item.stat().st_mtime, reverse=True):
        parsed = parse_session(path.stem)
        if parsed is not None:
            rows.append(parsed)
    return rows


def recent_dialogue(session_id: str, *, max_messages: int = 6, max_tokens_equivalent: int = 1200) -> str:
    parsed = parse_active_session(session_id)
    if not parsed:
        return ''
    lines: list[str] = []
    for item in list(parsed.get('messages') or [])[-max_messages:]:
        role = str(item.get('role') or '').strip()
        message = str(item.get('message') or '').strip()
        if role and message:
            lines.append(f'{role}: {message}')
    text = '\n'.join(lines).strip()
    max_chars = max_tokens_equivalent * 4
    return text[-max_chars:].strip() if len(text) > max_chars else text


def _clean_entity_candidate(value: str) -> str:
    parts = [part.strip(" .,!?;:'\"") for part in str(value or '').split()]
    filtered = [part for part in parts if normalize_name(part) and normalize_name(part) not in _ENTITY_STOPWORDS]
    return ' '.join(filtered[:4]).strip()


def infer_current_entity(session_id: str) -> str:
    parsed = parse_active_session(session_id)
    if not parsed:
        return ''
    for item in reversed(list(parsed.get('messages') or [])):
        message = str(item.get('message') or '').strip()
        for pattern in _ENTITY_PATTERNS:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                candidate = _clean_entity_candidate(match.group(1) or '')
                if candidate:
                    return candidate
        tokens = [_clean_entity_candidate(token) for token in message.split() if token[:1].isupper()]
        tokens = [token for token in tokens if token]
        if tokens:
            return tokens[-1]
    return ''
