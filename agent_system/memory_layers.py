from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .runtime_config import get_runtime_config


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _safe_session_id(session_id: str) -> str:
    clean = ''.join(char if char.isalnum() or char in {'-', '_'} else '-' for char in str(session_id or '').strip())
    return clean or f"session-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"


def _slug(value: str) -> str:
    token = ''.join(char.lower() if char.isalnum() else '_' for char in str(value or '').strip())
    return '_'.join(part for part in token.split('_') if part) or 'unknown'


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp_path.replace(path)


@dataclass(frozen=True, slots=True)
class MemoryLayerPolicy:
    name: str
    storage_kind: str
    hot_path_read: bool
    hot_path_write: bool
    bounded: bool
    read_rule: str
    write_rule: str
    retention_rule: str
    compression_rule: str
    archival_rule: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'storage_kind': self.storage_kind,
            'hot_path_read': self.hot_path_read,
            'hot_path_write': self.hot_path_write,
            'bounded': self.bounded,
            'read_rule': self.read_rule,
            'write_rule': self.write_rule,
            'retention_rule': self.retention_rule,
            'compression_rule': self.compression_rule,
            'archival_rule': self.archival_rule,
        }


def working_memory_policy() -> MemoryLayerPolicy:
    return MemoryLayerPolicy(
        name='working_memory',
        storage_kind='ephemeral_in_process',
        hot_path_read=True,
        hot_path_write=True,
        bounded=True,
        read_rule='Read only during the active turn and never as a substitute for durable memory.',
        write_rule='Write transient turn state only; do not persist as canonical knowledge.',
        retention_rule='Retain only the current turn and immediately adjacent transient state.',
        compression_rule='No archival compression; overwritten by the next active turn.',
        archival_rule='Never archived directly; durable facts must be promoted to session/persona/graph layers by deterministic code.',
    )


def session_memory_policy() -> MemoryLayerPolicy:
    config = get_runtime_config().memory
    return MemoryLayerPolicy(
        name='session_memory',
        storage_kind='file_first_text',
        hot_path_read=True,
        hot_path_write=True,
        bounded=True,
        read_rule='Hot path reads only the active recent tail for dialogue context and entity inference.',
        write_rule='Append new turns to the active session file, then apply deterministic archival policy.',
        retention_rule=f'Keep {config.session_keep_recent_messages} recent messages in hot session storage; archive older messages once active session exceeds {config.session_archive_after_messages}.',
        compression_rule='Archive in chronological chunks without rewriting semantic content.',
        archival_rule='Move older turns to cold session archives; full session reconstruction remains available by combining archive + active tail.',
    )


def persona_memory_policy() -> MemoryLayerPolicy:
    config = get_runtime_config().memory
    return MemoryLayerPolicy(
        name='persona_memory',
        storage_kind='file_first_folder_bundle',
        hot_path_read=True,
        hot_path_write=True,
        bounded=True,
        read_rule='Read active traits, emotion state, relations, examples, triad, and local graph only from bounded persona files.',
        write_rule='Write state updates and synthesized persona content to active head files, then archive overflow beyond policy limits.',
        retention_rule=(
            f'Bound traits to {config.persona_trait_limit}, relations to {config.persona_relation_limit}, examples to '
            f'{config.persona_example_limit}, reactions to {config.persona_reaction_limit}, log tuples to {config.persona_log_tuple_limit}, '
            f'knowledge to {config.persona_knowledge_char_limit} chars.'
        ),
        compression_rule='Summarize repeated behavior into log tuples and keep only bounded active persona memory.',
        archival_rule='Overflow and snapshots are stored in cold persona archives and never read directly into hot chat context.',
    )


def graph_memory_policy() -> MemoryLayerPolicy:
    return MemoryLayerPolicy(
        name='graph_knowledge_memory',
        storage_kind='file_first_structured_graph',
        hot_path_read=True,
        hot_path_write=True,
        bounded=True,
        read_rule='Read only normalized active graph state for retrieval, scoring, and context assembly.',
        write_rule='Write through validated merge/edit/hygiene paths only.',
        retention_rule='Retain active graph through deterministic hygiene, duplicate resolution, decay, and garbage collection.',
        compression_rule='Use graph compression and hygiene, not raw append-only growth.',
        archival_rule='Cold graph snapshots are optional maintenance artifacts and are excluded from hot-path retrieval.',
    )


def archive_memory_policy() -> MemoryLayerPolicy:
    return MemoryLayerPolicy(
        name='archive_cold_memory',
        storage_kind='file_first_json_archives',
        hot_path_read=False,
        hot_path_write=False,
        bounded=True,
        read_rule='Never consulted on the chat hot path unless an explicit maintenance or migration workflow asks for it.',
        write_rule='Write only through explicit lifecycle policies from session/persona/graph layers.',
        retention_rule='Keep bounded archive indexes and snapshots for auditability and migration safety.',
        compression_rule='Store cold summaries, overflow payloads, and snapshots in compact JSON artifacts.',
        archival_rule='Acts as sink for aged session turns, persona overflow, and graph snapshots.',
    )


def layered_memory_policies() -> dict[str, MemoryLayerPolicy]:
    return {
        'working_memory': working_memory_policy(),
        'session_memory': session_memory_policy(),
        'persona_memory': persona_memory_policy(),
        'graph_knowledge_memory': graph_memory_policy(),
        'archive_cold_memory': archive_memory_policy(),
    }


def describe_memory_layers() -> dict[str, Any]:
    config = get_runtime_config()
    return {
        'paths': {
            'memory_root': str(config.paths.memory_root),
            'working_dir': str(config.paths.working_dir),
            'sessions_dir': str(config.paths.sessions_dir),
            'heads_dir': str(config.paths.heads_dir),
            'graphs_dir': str(config.paths.graphs_dir),
            'archive_dir': str(config.paths.archive_dir),
            'archive_sessions_dir': str(config.paths.archive_sessions_dir),
            'archive_heads_dir': str(config.paths.archive_heads_dir),
            'archive_graphs_dir': str(config.paths.archive_graphs_dir),
        },
        'layers': {name: policy.to_dict() for name, policy in layered_memory_policies().items()},
    }


def session_archive_path(session_id: str) -> Path:
    return get_runtime_config().paths.archive_sessions_dir / f'{_safe_session_id(session_id)}.json'


def load_session_archive(session_id: str) -> dict[str, Any]:
    path = session_archive_path(session_id)
    if not path.exists():
        return {
            'session_id': _safe_session_id(session_id),
            'title': '',
            'archived_messages': [],
            'archive_events': [],
            'updated_at': '',
        }
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        payload = {}
    return {
        'session_id': _safe_session_id(session_id),
        'title': str(payload.get('title') or '').strip(),
        'archived_messages': [dict(item) for item in list(payload.get('archived_messages') or []) if isinstance(item, dict)],
        'archive_events': [dict(item) for item in list(payload.get('archive_events') or []) if isinstance(item, dict)],
        'updated_at': str(payload.get('updated_at') or '').strip(),
    }


def append_session_archive(session_id: str, *, title: str, messages: list[dict[str, Any]], reason: str = 'retention') -> dict[str, Any]:
    if not messages:
        return load_session_archive(session_id)
    payload = load_session_archive(session_id)
    seen = {
        (
            str(item.get('message_id') or '').strip(),
            str(item.get('timestamp') or '').strip(),
            str(item.get('role') or '').strip(),
            str(item.get('raw_text') or item.get('display_text') or item.get('message') or ''),
        )
        for item in list(payload.get('archived_messages') or [])
        if isinstance(item, dict)
    }
    for item in messages:
        raw_text = str(item.get('raw_text') or item.get('display_text') or item.get('message') or '')
        display_text = str(item.get('display_text') or raw_text)
        key = (
            str(item.get('message_id') or '').strip(),
            str(item.get('timestamp') or '').strip(),
            str(item.get('role') or '').strip(),
            raw_text,
        )
        if not key[2] or not key[3] or key in seen:
            continue
        seen.add(key)
        payload['archived_messages'].append(
            {
                'message_id': key[0],
                'timestamp': key[1],
                'role': key[2],
                'message': display_text,
                'raw_text': raw_text,
                'analysis_text': str(item.get('analysis_text') or display_text),
                'display_text': display_text,
                'persona_name': str(item.get('persona_name') or '').strip(),
            }
        )
    payload['title'] = str(title or payload.get('title') or '').strip()
    payload['archive_events'] = (
        [
            {
                'recorded_at': _utc_now(),
                'reason': str(reason or 'retention'),
                'moved_messages': len(messages),
            }
        ]
        + list(payload.get('archive_events') or [])
    )[: get_runtime_config().memory.archive_index_limit]
    payload['updated_at'] = _utc_now()
    _write_json(session_archive_path(session_id), payload)
    return payload


def persona_archive_dir(name: str) -> Path:
    return get_runtime_config().paths.archive_heads_dir / _slug(name)


def persona_snapshots_dir(name: str) -> Path:
    return persona_archive_dir(name) / 'snapshots'


def persona_overflow_archive_path(name: str) -> Path:
    return persona_archive_dir(name) / 'overflow.json'


def load_persona_overflow_archive(name: str) -> dict[str, Any]:
    path = persona_overflow_archive_path(name)
    if not path.exists():
        return {'head': _slug(name), 'entries': [], 'updated_at': ''}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        payload = {}
    return {
        'head': str(payload.get('head') or _slug(name)),
        'entries': [dict(item) for item in list(payload.get('entries') or []) if isinstance(item, dict)],
        'updated_at': str(payload.get('updated_at') or '').strip(),
    }


def append_persona_overflow_archive(name: str, overflow: dict[str, Any], *, reason: str = 'payload_trim') -> dict[str, Any]:
    cleaned = {
        key: value
        for key, value in dict(overflow or {}).items()
        if value not in (None, '', [], {})
    }
    if not cleaned:
        return load_persona_overflow_archive(name)
    payload = load_persona_overflow_archive(name)
    entry = {
        'recorded_at': _utc_now(),
        'reason': str(reason or 'payload_trim'),
        **cleaned,
    }
    payload['entries'] = [entry] + list(payload.get('entries') or [])
    payload['entries'] = payload['entries'][: get_runtime_config().memory.archive_index_limit]
    payload['updated_at'] = _utc_now()
    _write_json(persona_overflow_archive_path(name), payload)
    return payload


def persona_snapshot_path(name: str, reason: str = 'manual') -> Path:
    stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    slug = _slug(reason or 'manual')
    return persona_snapshots_dir(name) / f'{stamp}-{slug}.json'


def record_persona_snapshot(name: str, payload: dict[str, Any], *, reason: str = 'manual') -> Path:
    path = persona_snapshot_path(name, reason)
    body = {
        'recorded_at': _utc_now(),
        'reason': str(reason or 'manual'),
        **dict(payload or {}),
    }
    _write_json(path, body)
    snapshots = sorted(persona_snapshots_dir(name).glob('*.json'))
    limit = get_runtime_config().memory.archive_index_limit
    for stale in snapshots[:-limit]:
        stale.unlink(missing_ok=True)
    return path


def list_persona_snapshots(name: str, *, limit: int = 12) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(persona_snapshots_dir(name).glob('*.json'), reverse=True)[: max(int(limit or 12), 1)]:
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                'path': str(path),
                'recorded_at': str(payload.get('recorded_at') or '').strip(),
                'reason': str(payload.get('reason') or '').strip(),
            }
        )
    return rows


def graph_snapshot_path(reason: str = 'manual') -> Path:
    stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    slug = _slug(reason or 'manual')
    return get_runtime_config().paths.archive_graphs_dir / f'{stamp}-{slug}.json'


def record_graph_snapshot(*, nodes: list[dict[str, Any]], edges: list[dict[str, Any]], reason: str = 'manual', diagnostics: dict[str, Any] | None = None) -> Path:
    path = graph_snapshot_path(reason)
    payload = {
        'recorded_at': _utc_now(),
        'reason': str(reason or 'manual'),
        'nodes': list(nodes),
        'edges': list(edges),
        'diagnostics': dict(diagnostics or {}),
    }
    _write_json(path, payload)
    snapshots = sorted(get_runtime_config().paths.archive_graphs_dir.glob('*.json'))
    limit = get_runtime_config().memory.graph_snapshot_limit
    for stale in snapshots[:-limit]:
        stale.unlink(missing_ok=True)
    return path


def list_graph_snapshots(*, limit: int = 16) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(get_runtime_config().paths.archive_graphs_dir.glob('*.json'), reverse=True)[: max(int(limit or 16), 1)]:
        if path.name == 'lifecycle.json':
            continue
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                'path': str(path),
                'recorded_at': str(payload.get('recorded_at') or '').strip(),
                'reason': str(payload.get('reason') or '').strip(),
                'node_count': len(list(payload.get('nodes') or [])),
                'edge_count': len(list(payload.get('edges') or [])),
            }
        )
    return rows


def load_graph_snapshot(path_value: str) -> dict[str, Any] | None:
    clean = str(path_value or '').strip()
    if not clean:
        return None
    path = Path(clean).expanduser()
    if not path.is_absolute():
        path = get_runtime_config().paths.archive_graphs_dir / path
    path = path.resolve()
    try:
        archive_root = get_runtime_config().paths.archive_graphs_dir.resolve()
        path.relative_to(archive_root)
    except Exception:
        return None
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def graph_lifecycle_log_path() -> Path:
    return get_runtime_config().paths.archive_graphs_dir / 'lifecycle.json'


def load_graph_lifecycle_log() -> dict[str, Any]:
    path = graph_lifecycle_log_path()
    if not path.exists():
        return {'entries': [], 'updated_at': ''}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        payload = {}
    return {
        'entries': [dict(item) for item in list(payload.get('entries') or []) if isinstance(item, dict)],
        'updated_at': str(payload.get('updated_at') or '').strip(),
    }


def append_graph_lifecycle_events(events: list[dict[str, Any]], *, reason: str = 'maintenance') -> dict[str, Any]:
    cleaned_events = [
        {
            'node_id': str(item.get('node_id') or '').strip(),
            'name': str(item.get('name') or '').strip(),
            'state': str(item.get('state') or '').strip(),
            'target_id': str(item.get('target_id') or '').strip(),
            'source': str(item.get('source') or '').strip(),
            'detail': str(item.get('detail') or '').strip(),
        }
        for item in list(events or [])
        if isinstance(item, dict) and str(item.get('state') or '').strip()
    ]
    cleaned_events = [item for item in cleaned_events if item['node_id'] or item['name']]
    if not cleaned_events:
        return load_graph_lifecycle_log()
    payload = load_graph_lifecycle_log()
    entry = {
        'recorded_at': _utc_now(),
        'reason': str(reason or 'maintenance'),
        'events': cleaned_events,
    }
    payload['entries'] = [entry] + list(payload.get('entries') or [])
    payload['entries'] = payload['entries'][: get_runtime_config().memory.archive_index_limit]
    payload['updated_at'] = _utc_now()
    _write_json(graph_lifecycle_log_path(), payload)
    return payload
