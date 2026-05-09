from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from .duplicate_resolver import normalize_name
from .memory_layers import append_session_archive, load_session_archive, session_archive_path
from .runtime_config import get_runtime_config
from .text_resources import capitalized_entity_phrases, compiled_entity_patterns

_ENTITY_PATTERNS = compiled_entity_patterns()

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
    'i',
    'im',
    'm',
    'my',
    'we',
    'our',
    'glad',
    'про',
    'как',
    'что',
    'кто',
    'do',
    'does',
    'did',
    'is',
    'are',
    'was',
    'were',
    'has',
    'have',
    'had',
    'can',
    'could',
    'will',
    'would',
    'should',
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


def session_messages_dir() -> Path:
    path = sessions_dir() / '_messages'
    path.mkdir(parents=True, exist_ok=True)
    return path


def session_messages_path(session_id: str) -> Path:
    return session_messages_dir() / f'{_safe_session_id(session_id)}.jsonl'


def session_files_dir(session_id: str) -> Path:
    path = files_dir() / _safe_session_id(session_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def session_route_state_path(session_id: str) -> Path:
    path = sessions_dir() / '_route_state'
    path.mkdir(parents=True, exist_ok=True)
    return path / f'{_safe_session_id(session_id)}.json'


def load_session_route_state(session_id: str) -> dict[str, Any]:
    path = session_route_state_path(session_id)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def save_session_route_state(session_id: str, payload: dict[str, Any]) -> Path:
    path = session_route_state_path(session_id)
    clean_payload = dict(payload or {})
    clean_payload['session_id'] = _safe_session_id(session_id)
    path.write_text(json.dumps(clean_payload, ensure_ascii=False, indent=2), encoding='utf-8')
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


def _session_uploaded_files_path(session_id: str) -> Path:
    return files_dir() / _safe_session_id(session_id)


def _session_route_state_file(session_id: str) -> Path:
    return sessions_dir() / '_route_state' / f'{_safe_session_id(session_id)}.json'


def _session_mood_dataset_path(session_id: str) -> Path:
    token = normalize_name(_safe_session_id(session_id)) or 'session'
    return get_runtime_config().paths.mood_sessions_dir / f'{token}.jsonl'


def _session_mood_report_path(session_id: str) -> Path:
    token = normalize_name(_safe_session_id(session_id)) or 'session'
    return get_runtime_config().paths.mood_reports_dir / f'session__{token}.json'


def _session_annotation_path(session_id: str) -> Path:
    return get_runtime_config().paths.memory_root / 'message_annotations' / f'{_safe_session_id(session_id)}.json'


def _render_session_text(title: str, messages: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    if str(title or '').strip():
        blocks.append(f'# {str(title).strip()}')
    for item in list(messages or []):
        role = str(item.get('role') or '').strip()
        message = _message_display_text(item)
        timestamp = str(item.get('timestamp') or '').strip()
        persona_name = str(item.get('persona_name') or '').strip()
        if not role or not message:
            continue
        if timestamp:
            blocks.append(f'[{timestamp}]')
        if role == 'assistant' and persona_name:
            blocks.append(f'assistant[{persona_name}]: {message}')
        else:
            blocks.append(f'{role}: {message}')
        blocks.append('')
    return '\n'.join(blocks).strip() + ('\n' if blocks else '')


def _message_text(value: Any) -> str:
    if value is None:
        return ''
    return str(value)


def _message_display_text(item: dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ''
    return _message_text(
        item.get('display_text')
        or item.get('raw_text')
        or item.get('message')
    )


def _message_analysis_text(item: dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ''
    return _message_text(
        item.get('analysis_text')
        or item.get('display_text')
        or item.get('raw_text')
        or item.get('message')
    )


def _message_id_for(
    *,
    session_id: str,
    role: str,
    timestamp: str,
    message: str,
    index: int,
) -> str:
    token = '||'.join(
        (
            _safe_session_id(session_id),
            str(role or ''),
            str(timestamp or ''),
            str(index),
            str(message or ''),
        )
    )
    digest = hashlib.blake2b(token.encode('utf-8'), digest_size=8).hexdigest()
    return f'm_{digest}'


def _normalize_message_item(
    item: dict[str, Any],
    *,
    session_id: str,
    index: int,
) -> dict[str, Any]:
    role = str(item.get('role') or '').strip()
    raw_text = _message_text(item.get('raw_text') if 'raw_text' in item else item.get('message'))
    display_text = _message_text(item.get('display_text') if 'display_text' in item else raw_text)
    analysis_text = _message_text(item.get('analysis_text') if 'analysis_text' in item else display_text)
    timestamp = str(item.get('timestamp') or '').strip()
    persona_name = str(item.get('persona_name') or '').strip()
    message_id = str(item.get('message_id') or item.get('id') or '').strip()
    if not message_id:
        message_id = _message_id_for(
            session_id=session_id,
            role=role,
            timestamp=timestamp,
            message=raw_text or display_text,
            index=index,
        )
    return {
        'id': message_id,
        'message_id': message_id,
        'role': role,
        'message': display_text,
        'raw_text': raw_text,
        'analysis_text': analysis_text,
        'display_text': display_text,
        'timestamp': timestamp,
        'persona_name': persona_name,
        'user_persona_name': str(item.get('user_persona_name') or '').strip(),
        'speaker_name': str(item.get('speaker_name') or '').strip(),
    }


def _load_structured_messages(session_id: str) -> list[dict[str, Any]]:
    path = session_messages_path(session_id)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        normalized = _normalize_message_item(payload, session_id=session_id, index=index)
        if normalized.get('role') and normalized.get('display_text'):
            rows.append(normalized)
    return rows


def _append_structured_messages(session_id: str, messages: list[dict[str, Any]]) -> None:
    path = session_messages_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        for item in messages:
            normalized = _normalize_message_item(item, session_id=session_id, index=0)
            handle.write(json.dumps(normalized, ensure_ascii=False) + '\n')


def _rewrite_structured_messages(session_id: str, messages: list[dict[str, Any]]) -> None:
    path = session_messages_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps(
            _normalize_message_item(item, session_id=session_id, index=index),
            ensure_ascii=False,
        )
        for index, item in enumerate(messages)
    ]
    path.write_text('\n'.join(rows) + ('\n' if rows else ''), encoding='utf-8')


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
        # assistant[Персона]: message or just assistant: message
        import re as _re
        persona_tag_m = _re.match(r'^(assistant)\[(.+?)\]$', role)
        if role == 'user' or role == 'assistant' or persona_tag_m:
            actual_role = 'assistant' if persona_tag_m else role
            pname = persona_tag_m.group(2).strip() if persona_tag_m else ''
            messages.append(
                _normalize_message_item(
                    {
                        'role': actual_role,
                        'message': message.strip(),
                        'raw_text': message.strip(),
                        'analysis_text': message.strip(),
                        'display_text': message.strip(),
                        'timestamp': timestamp,
                        'persona_name': pname,
                    },
                    session_id=session_id,
                    index=len(messages),
                )
            )
    resolved_title = title or next((item['display_text'][:60] for item in messages if item['role'] == 'user'), 'New session')
    updated_at = timestamp or datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
    return {'session_id': _safe_session_id(session_id), 'title': resolved_title, 'messages': messages, 'updated_at': updated_at, 'path': str(path)}


def _compose_session_payload(
    *,
    session_id: str,
    active: dict[str, Any] | None,
    archive: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    archive_payload = dict(archive or {})
    archived_messages = [
        _normalize_message_item(item, session_id=session_id, index=index)
        for index, item in enumerate(list(archive_payload.get('archived_messages') or []))
        if isinstance(item, dict)
    ]
    active_messages = [
        _normalize_message_item(item, session_id=session_id, index=len(archived_messages) + index)
        for index, item in enumerate(list((active or {}).get('messages') or []))
        if isinstance(item, dict)
    ]
    if active is None and not archived_messages:
        return None
    messages = archived_messages + active_messages
    title = str((active or {}).get('title') or archive_payload.get('title') or '').strip()
    if not title:
        title = next((str(item.get('display_text') or '')[:60] for item in messages if str(item.get('role') or '').strip() == 'user'), 'New session')
    updated_at = str((active or {}).get('updated_at') or archive_payload.get('updated_at') or _utc_now()).strip()
    return {
        'session_id': _safe_session_id(session_id),
        'title': title,
        'messages': messages,
        'updated_at': updated_at,
        'path': str(session_text_path(session_id)),
        'active_message_count': len(active_messages),
        'archived_message_count': len(archived_messages),
    }


def parse_active_session(session_id: str) -> dict[str, Any] | None:
    legacy = _parse_session_path(session_text_path(session_id), session_id=session_id)
    structured_messages = _load_structured_messages(session_id)
    if structured_messages:
        title = str((legacy or {}).get('title') or '').strip()
        if not title:
            title = next((item['display_text'][:60] for item in structured_messages if item['role'] == 'user'), 'New session')
        updated_at = str(structured_messages[-1].get('timestamp') or (legacy or {}).get('updated_at') or _utc_now()).strip()
        active = {
            'session_id': _safe_session_id(session_id),
            'title': title,
            'messages': structured_messages,
            'updated_at': updated_at,
            'path': str(session_text_path(session_id)),
        }
    else:
        active = legacy
    return _compose_session_payload(session_id=session_id, active=active, archive=None)


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
    _rewrite_structured_messages(session_id, active_messages)
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


def append_turn(session_id: str, user_message: str, assistant_message: str, *, persona_name: str = '', user_persona_name: str = '') -> Path:
    path = session_text_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    blocks: list[str] = []
    structured_messages: list[dict[str, Any]] = []
    raw_user_message = _message_text(user_message)
    raw_assistant_message = _message_text(assistant_message)
    if raw_user_message.strip():
        user_timestamp = _utc_now()
        structured_messages.append(
            {
                'message_id': _message_id_for(
                    session_id=session_id,
                    role='user',
                    timestamp=user_timestamp,
                    message=raw_user_message,
                    index=0,
                ),
                'role': 'user',
                'message': raw_user_message,
                'raw_text': raw_user_message,
                'analysis_text': raw_user_message.strip(),
                'display_text': raw_user_message,
                'timestamp': user_timestamp,
                'persona_name': '',
                'user_persona_name': str(user_persona_name or '').strip(),
            }
        )
        blocks.append(f'[{user_timestamp}]\nuser: {raw_user_message}')
    if raw_assistant_message.strip():
        assistant_timestamp = _utc_now()
        clean_persona_name = str(persona_name or '').strip()
        structured_messages.append(
            {
                'message_id': _message_id_for(
                    session_id=session_id,
                    role='assistant',
                    timestamp=assistant_timestamp,
                    message=raw_assistant_message,
                    index=len(structured_messages),
                ),
                'role': 'assistant',
                'message': raw_assistant_message,
                'raw_text': raw_assistant_message,
                'analysis_text': raw_assistant_message.strip(),
                'display_text': raw_assistant_message,
                'timestamp': assistant_timestamp,
                'persona_name': clean_persona_name,
            }
        )
        persona_tag = f'[{clean_persona_name}]' if clean_persona_name else ''
        blocks.append(f'[{assistant_timestamp}]\nassistant{persona_tag}: {raw_assistant_message}')
    if not blocks:
        return path
    current = path.read_text(encoding='utf-8') if path.exists() else ''
    payload = (current.rstrip() + '\n\n' if current.strip() else '') + '\n\n'.join(blocks) + '\n'
    path.write_text(payload, encoding='utf-8')
    _append_structured_messages(session_id, structured_messages)
    _apply_session_retention(session_id)
    return path


def parse_session(session_id: str, *, include_archived: bool = True) -> dict[str, Any] | None:
    active = parse_active_session(session_id)
    archive = load_session_archive(session_id) if include_archived else {'archived_messages': [], 'title': '', 'updated_at': ''}
    return _compose_session_payload(session_id=session_id, active=active, archive=archive)


def list_sessions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_ids = {path.stem for path in sessions_dir().glob('*.txt')}
    candidate_ids.update(path.stem for path in session_messages_dir().glob('*.jsonl'))
    def _mtime(session_id: str) -> float:
        timestamps = [
            session_text_path(session_id).stat().st_mtime if session_text_path(session_id).exists() else 0.0,
            session_messages_path(session_id).stat().st_mtime if session_messages_path(session_id).exists() else 0.0,
        ]
        return max(timestamps)
    for session_id in sorted(candidate_ids, key=_mtime, reverse=True):
        parsed = parse_session(session_id)
        if parsed is not None:
            rows.append(parsed)
    return rows


def delete_session(session_id: str) -> dict[str, Any]:
    clean_session_id = _safe_session_id(session_id)
    file_paths = [
        session_text_path(clean_session_id),
        session_messages_path(clean_session_id),
        _session_annotation_path(clean_session_id),
        _session_route_state_file(clean_session_id),
        session_archive_path(clean_session_id),
        _session_mood_dataset_path(clean_session_id),
        _session_mood_report_path(clean_session_id),
    ]
    dir_paths = [
        _session_uploaded_files_path(clean_session_id),
    ]
    deleted_paths: list[str] = []
    missing_paths: list[str] = []

    for path in file_paths:
        try:
            if path.exists():
                path.unlink()
                deleted_paths.append(str(path))
            else:
                missing_paths.append(str(path))
        except OSError:
            missing_paths.append(str(path))

    for path in dir_paths:
        try:
            if path.exists():
                shutil.rmtree(path)
                deleted_paths.append(str(path))
            else:
                missing_paths.append(str(path))
        except OSError:
            missing_paths.append(str(path))

    return {
        'ok': bool(deleted_paths),
        'session_id': clean_session_id,
        'deleted_paths': deleted_paths,
        'missing_paths': missing_paths,
    }


def recent_dialogue(session_id: str, *, max_messages: int = 6, max_tokens_equivalent: int = 1200) -> str:
    parsed = parse_active_session(session_id)
    if not parsed:
        return ''
    lines: list[str] = []
    for item in list(parsed.get('messages') or [])[-max_messages:]:
        role = str(item.get('role') or '').strip()
        message = _message_display_text(item).strip()
        if not (role and message):
            continue
        if role == 'user':
            # Показываем имя пользователя-персоны если оно задано
            # Это позволяет ассистенту различать нескольких собеседников в одной сессии
            user_persona = str(item.get('user_persona_name') or '').strip()
            display_role = user_persona if user_persona else 'user'
        elif role == 'assistant':
            persona = str(item.get('persona_name') or '').strip()
            display_role = persona if persona else 'assistant'
        else:
            display_role = role
        lines.append(f'{display_role}: {message}')
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
    messages = list(parsed.get('messages') or [])

    # Find the most recent user turn.
    last_user_item: dict | None = None
    for item in reversed(messages):
        if str(item.get('role') or '').strip() == 'user':
            last_user_item = item
            break
    if last_user_item is None:
        return ''

    # Modern history items carry user_persona_name (written by append_turn).
    # If the field is present it's authoritative for the speaker identity:
    #   - non-empty → that persona IS the current entity (return directly)
    #   - empty    → anonymous sender; scan ONLY this turn's text so stale
    #               persona names from previous turns don't bleed through
    if 'user_persona_name' in last_user_item:
        stored_persona = str(last_user_item['user_persona_name'] or '').strip()
        if stored_persona:
            return stored_persona
        # Anonymous sender: look for topic entities in the current message only.
        message = _message_display_text(last_user_item).strip()
        for pattern in _ENTITY_PATTERNS:
            match = pattern.search(message)
            if match:
                candidate = _clean_entity_candidate(match.group(1) or '')
                if candidate:
                    return candidate
        phrases = [_clean_entity_candidate(token) for token in capitalized_entity_phrases(message)]
        phrases = [token for token in phrases if token]
        return phrases[-1] if phrases else ''

    # Legacy history (no user_persona_name field): scan all user turns.
    for item in reversed(messages):
        if str(item.get('role') or '').strip() != 'user':
            continue
        message = _message_display_text(item).strip()
        for pattern in _ENTITY_PATTERNS:
            match = pattern.search(message)
            if match:
                candidate = _clean_entity_candidate(match.group(1) or '')
                if candidate:
                    return candidate
        phrases = [_clean_entity_candidate(token) for token in capitalized_entity_phrases(message)]
        phrases = [token for token in phrases if token]
        if phrases:
            return phrases[-1]
    return ''
