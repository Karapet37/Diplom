"""
Notes store.

A lightweight, user-controlled memory layer.

The operator/user saves important moments explicitly with commands:
    /save <text>      — save a note
    /notes            — list saved notes
    /del_note <id>    — delete a note by 1-based index or note_id
    /clear_notes      — delete all notes for this session

Notes are stored per-session as a JSONL file under memory/notes/.

They are separate from the graph, session log, and personality.
They represent what the USER explicitly considers important.

Notes are injected into planning context in compact form.

This is also the training data source for the importance learner:
every turn that was explicitly saved becomes a positive example.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .runtime_config import get_runtime_config

_STORE_LOCK = Lock()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _notes_dir() -> Path:
    return get_runtime_config().paths.memory_root / 'notes'


def _notes_path(session_id: str) -> Path:
    safe = re.sub(r'[^\w\-]', '_', str(session_id or 'default').strip())
    return _notes_dir() / f'{safe}.jsonl'


# ---------------------------------------------------------------------------
# Note model
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _new_note(text: str, session_id: str = '', source: str = 'manual') -> dict[str, Any]:
    return {
        'note_id': uuid.uuid4().hex[:10],
        'session_id': session_id,
        'text': str(text or '').strip(),
        'timestamp': _utc_now(),
        'source': source,          # 'manual' | 'suggested' | 'auto'
        'confirmed': True,         # manually saved notes are always confirmed
    }


# ---------------------------------------------------------------------------
# Low-level I/O
# ---------------------------------------------------------------------------

def _load_notes_raw(session_id: str) -> list[dict[str, Any]]:
    path = _notes_path(session_id)
    if not path.exists():
        return []
    notes: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and obj.get('text'):
                notes.append(obj)
        except json.JSONDecodeError:
            continue
    return notes


def _write_notes_raw(session_id: str, notes: list[dict[str, Any]]) -> None:
    path = _notes_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(n, ensure_ascii=False) for n in notes]
    path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_note(session_id: str, text: str, *, source: str = 'manual') -> dict[str, Any]:
    """
    Save a note for this session. Returns the saved note dict (including note_id).
    """
    text = str(text or '').strip()
    if not text:
        return {}
    with _STORE_LOCK:
        notes = _load_notes_raw(session_id)
        note = _new_note(text, session_id=session_id, source=source)
        notes.append(note)
        _write_notes_raw(session_id, notes)
    return note


def load_notes(session_id: str) -> list[dict[str, Any]]:
    """Return all saved notes for a session, newest last."""
    with _STORE_LOCK:
        return list(_load_notes_raw(session_id))


def delete_note(session_id: str, note_ref: str) -> bool:
    """
    Delete a note by note_id or 1-based index string.
    Returns True if something was deleted.
    """
    note_ref = str(note_ref or '').strip()
    if not note_ref:
        return False
    with _STORE_LOCK:
        notes = _load_notes_raw(session_id)
        original_len = len(notes)
        # Try by note_id first
        filtered = [n for n in notes if n.get('note_id') != note_ref]
        if len(filtered) < original_len:
            _write_notes_raw(session_id, filtered)
            return True
        # Try by 1-based index
        try:
            idx = int(note_ref) - 1
            if 0 <= idx < len(notes):
                notes.pop(idx)
                _write_notes_raw(session_id, notes)
                return True
        except (ValueError, TypeError):
            pass
        return False


def clear_notes(session_id: str) -> int:
    """Delete all notes for a session. Returns count deleted."""
    with _STORE_LOCK:
        notes = _load_notes_raw(session_id)
        count = len(notes)
        if count:
            _write_notes_raw(session_id, [])
        return count


def format_notes_for_context(session_id: str, *, max_notes: int = 20) -> str:
    """
    Return a compact string representation of saved notes
    suitable for injection into a planning prompt.
    """
    notes = load_notes(session_id)
    if not notes:
        return ''
    recent = notes[-max_notes:]
    lines = []
    for i, note in enumerate(recent, 1):
        ts = str(note.get('timestamp') or '')[:10]
        text = str(note.get('text') or '').strip()
        lines.append(f'[{i}] ({ts}) {text}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Command parser
# ---------------------------------------------------------------------------

# Matches: /save ..., /notes, /del_note ..., /clear_notes
_COMMAND_RE = re.compile(
    r'^\s*/(save|notes|del_note|clear_notes)\s*(.*)',
    re.IGNORECASE | re.DOTALL,
)


def parse_note_command(message: str) -> dict[str, Any] | None:
    """
    If the message is a note command, return a dict describing it.
    Returns None if not a note command.

    Return shapes:
        {'cmd': 'save',        'text': '...'}
        {'cmd': 'notes'}
        {'cmd': 'del_note',    'ref': '...'}
        {'cmd': 'clear_notes'}
    """
    m = _COMMAND_RE.match(str(message or ''))
    if not m:
        return None
    cmd = m.group(1).lower()
    arg = m.group(2).strip()
    if cmd == 'save':
        return {'cmd': 'save', 'text': arg}
    if cmd == 'notes':
        return {'cmd': 'notes'}
    if cmd == 'del_note':
        return {'cmd': 'del_note', 'ref': arg}
    if cmd == 'clear_notes':
        return {'cmd': 'clear_notes'}
    return None


def execute_note_command(session_id: str, command: dict[str, Any]) -> str:
    """
    Execute a parsed note command and return a human-readable reply string.
    """
    cmd = str(command.get('cmd') or '')

    if cmd == 'save':
        text = str(command.get('text') or '').strip()
        if not text:
            return 'Nothing to save. Usage: /save <your note text>'
        note = save_note(session_id, text)
        note_id = note.get('note_id', '?')
        return f'Saved note [{note_id}]: {text}'

    if cmd == 'notes':
        notes = load_notes(session_id)
        if not notes:
            return 'No saved notes.'
        lines = ['Saved notes:']
        for i, note in enumerate(notes, 1):
            ts = str(note.get('timestamp') or '')[:10]
            text = str(note.get('text') or '').strip()
            nid = str(note.get('note_id') or '')
            lines.append(f'  [{i}] ({ts}) [{nid}] {text}')
        return '\n'.join(lines)

    if cmd == 'del_note':
        ref = str(command.get('ref') or '').strip()
        if not ref:
            return 'Usage: /del_note <index or note_id>'
        deleted = delete_note(session_id, ref)
        if deleted:
            return f'Deleted note: {ref}'
        return f'Note not found: {ref}'

    if cmd == 'clear_notes':
        count = clear_notes(session_id)
        if count:
            return f'Cleared {count} note(s).'
        return 'No notes to clear.'

    return f'Unknown note command: {cmd}'
