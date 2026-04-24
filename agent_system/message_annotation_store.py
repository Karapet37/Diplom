from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .history_store import parse_session
from .message_vector_registry import coordinate_registry_payload
from .message_vector_runtime import get_message_vector_runtime, normalize_coordinate_vector
from .runtime_config import get_runtime_config

_STORE_LOCK = Lock()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _annotations_dir() -> Path:
    path = get_runtime_config().paths.memory_root / 'message_annotations'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _session_annotations_path(session_id: str) -> Path:
    safe = ''.join(char if char.isalnum() or char in {'-', '_'} else '_' for char in str(session_id or '').strip()) or 'default'
    return _annotations_dir() / f'{safe}.json'


def _global_annotations_path() -> Path:
    return _annotations_dir() / 'global.jsonl'


def _load_session_annotations(session_id: str) -> dict[str, dict[str, Any]]:
    path = _session_annotations_path(session_id)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        payload = {}
    annotations = payload.get('annotations') if isinstance(payload, dict) else {}
    if not isinstance(annotations, dict):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for message_id, item in annotations.items():
        if isinstance(item, dict):
            rows[str(message_id)] = dict(item)
    return rows


def _write_session_annotations(session_id: str, annotations: dict[str, dict[str, Any]]) -> None:
    path = _session_annotations_path(session_id)
    payload = {
        'session_id': str(session_id or ''),
        'updated_at': _utc_now(),
        'annotations': annotations,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _append_global_annotation(row: dict[str, Any]) -> None:
    path = _global_annotations_path()
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def list_session_annotations(session_id: str) -> list[dict[str, Any]]:
    with _STORE_LOCK:
        rows = list(_load_session_annotations(session_id).values())
    rows.sort(key=lambda item: str(item.get('updated_at') or item.get('created_at') or ''), reverse=True)
    return rows


def _build_effective_session_rows(session_id: str, *, window_size: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    session = parse_session(session_id)
    if session is None:
        return (
            {
                'session_id': session_id,
                'title': '',
                'messages': [],
            },
            [],
        )
    annotations_by_message = {
        str(item.get('message_id') or ''): dict(item)
        for item in list_session_annotations(session_id)
        if isinstance(item, dict)
    }
    runtime = get_message_vector_runtime()
    clean_window_size = max(1, int(window_size or 4))
    message_rows: list[dict[str, Any]] = []
    for item in list(session.get('messages') or []):
        message_id = str(item.get('message_id') or item.get('id') or '').strip()
        context_rows = message_rows[-clean_window_size:]
        context_matrix = [
            {
                'message_id': str(entry.get('message_id') or ''),
                'role': str(entry.get('role') or ''),
                'display_text': str(entry.get('display_text') or ''),
                'vector': normalize_coordinate_vector(entry.get('vector')),
            }
            for entry in context_rows
        ]
        predicted_vector = runtime.predict_vector(
            text=str(item.get('analysis_text') or item.get('display_text') or item.get('raw_text') or ''),
            role=str(item.get('role') or ''),
            context_matrix=context_matrix,
            persona_name=str(item.get('persona_name') or ''),
        )
        correction = dict(annotations_by_message.get(message_id) or {})
        effective_vector = normalize_coordinate_vector(correction.get('coordinates') or predicted_vector)
        transition_interpretation = (
            dict(correction.get('transition_interpretation') or {})
            if correction
            else runtime.predict_transition_interpretation(context_matrix=context_matrix, current_vector=effective_vector)
        )
        message_rows.append(
            {
                'message_id': message_id,
                'id': message_id,
                'role': str(item.get('role') or ''),
                'timestamp': str(item.get('timestamp') or ''),
                'persona_name': str(item.get('persona_name') or ''),
                'raw_text': str(item.get('raw_text') or item.get('message') or ''),
                'analysis_text': str(item.get('analysis_text') or item.get('display_text') or item.get('message') or ''),
                'display_text': str(item.get('display_text') or item.get('raw_text') or item.get('message') or ''),
                'predicted_vector': predicted_vector,
                'vector': effective_vector,
                'has_correction': bool(correction),
                'annotation_id': str(correction.get('annotation_id') or ''),
                'annotation_notes': str(correction.get('notes') or ''),
                'context_window': [str(entry.get('message_id') or '') for entry in context_rows],
                'context_matrix_ref': f'ctx_{message_id}',
                'context_matrix': context_matrix,
                'transition_interpretation': transition_interpretation,
                'history_flow': runtime.build_history_flow(context_matrix=context_matrix, current_vector=effective_vector),
            }
        )
    return session, message_rows


def save_message_annotation(
    *,
    session_id: str,
    message_payload: dict[str, Any],
    coordinates: dict[str, Any],
    context_window: list[str] | None = None,
    context_matrix: list[dict[str, Any]] | None = None,
    transition_interpretation: dict[str, Any] | None = None,
    notes: str = '',
) -> dict[str, Any]:
    message_id = str(message_payload.get('message_id') or message_payload.get('id') or '').strip()
    if not message_id:
        raise ValueError('message_id is required')
    role = str(message_payload.get('role') or '').strip()
    display_text = str(message_payload.get('display_text') or message_payload.get('raw_text') or message_payload.get('message') or '')
    analysis_text = str(message_payload.get('analysis_text') or display_text)
    raw_text = str(message_payload.get('raw_text') or display_text)
    if not role or not display_text:
        raise ValueError('message role and text are required')

    clean_context_matrix = [dict(item) for item in list(context_matrix or []) if isinstance(item, dict)]
    record = {
        'annotation_id': str(message_payload.get('annotation_id') or uuid.uuid4().hex[:12]),
        'session_id': str(session_id or ''),
        'message_id': message_id,
        'role': role,
        'raw_text': raw_text,
        'analysis_text': analysis_text,
        'display_text': display_text,
        'timestamp': str(message_payload.get('timestamp') or ''),
        'persona_name': str(message_payload.get('persona_name') or ''),
        'coordinates': normalize_coordinate_vector(coordinates),
        'context_window': [str(item) for item in list(context_window or []) if str(item)],
        'context_matrix_ref': f'ctx_{message_id}',
        'context_matrix': clean_context_matrix,
        'transition_interpretation': dict(transition_interpretation or {}),
        'notes': str(notes or ''),
        'created_at': str(message_payload.get('created_at') or _utc_now()),
        'updated_at': _utc_now(),
        'source': 'ui_correction',
    }

    with _STORE_LOCK:
        annotations = _load_session_annotations(session_id)
        existing = dict(annotations.get(message_id) or {})
        if existing.get('annotation_id'):
            record['annotation_id'] = str(existing.get('annotation_id'))
            record['created_at'] = str(existing.get('created_at') or record['created_at'])
        annotations[message_id] = record
        _write_session_annotations(session_id, annotations)
        _append_global_annotation(record)

    get_message_vector_runtime().record_annotation(
        text=analysis_text,
        role=role,
        context_matrix=clean_context_matrix,
        coordinates=record['coordinates'],
        persona_name=record['persona_name'],
    )
    return record


def build_annotation_workspace(session_id: str, *, window_size: int = 4) -> dict[str, Any]:
    clean_window_size = max(1, int(window_size or 4))
    session, message_rows = _build_effective_session_rows(session_id, window_size=clean_window_size)
    if not list(session.get('messages') or []):
        return {
            'session_id': session_id,
            'registry': coordinate_registry_payload(),
            'messages': [],
            'message_count': 0,
            'window_size': clean_window_size,
        }
    return {
        'session_id': str(session.get('session_id') or session_id),
        'session_title': str(session.get('title') or ''),
        'registry': coordinate_registry_payload(),
        'messages': message_rows,
        'message_count': len(message_rows),
        'window_size': clean_window_size,
    }


def build_runtime_message_vector_payload(
    session_id: str,
    *,
    message_text: str,
    role: str = 'user',
    persona_name: str = '',
    window_size: int = 4,
) -> dict[str, Any]:
    clean_window_size = max(1, int(window_size or 4))
    _session, message_rows = _build_effective_session_rows(session_id, window_size=clean_window_size)
    runtime = get_message_vector_runtime()
    context_rows = message_rows[-clean_window_size:]
    context_matrix = [
        {
            'message_id': str(entry.get('message_id') or ''),
            'role': str(entry.get('role') or ''),
            'persona_name': str(entry.get('persona_name') or ''),
            'display_text': str(entry.get('display_text') or ''),
            'vector': normalize_coordinate_vector(entry.get('vector')),
        }
        for entry in context_rows
    ]
    current_vector = runtime.predict_vector(
        text=str(message_text or ''),
        role=str(role or ''),
        context_matrix=context_matrix,
        persona_name=str(persona_name or ''),
    )
    return {
        'window_size': clean_window_size,
        'context_window': [str(entry.get('message_id') or '') for entry in context_rows],
        'context_matrix_ref': f'ctx_runtime_{str(session_id or "").strip() or "session"}',
        'context_matrix': context_matrix,
        'current_vector': current_vector,
        'history_flow': runtime.build_history_flow(context_matrix=context_matrix, current_vector=current_vector),
        'transition_interpretation': runtime.predict_transition_interpretation(
            context_matrix=context_matrix,
            current_vector=current_vector,
        ),
    }
