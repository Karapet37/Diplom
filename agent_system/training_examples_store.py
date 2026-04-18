"""
Training examples store.

Lets users save (input, correct_output) pairs as positive fine-tuning signal.

Use cases:
    - "That response was good, save it as a training example"
    - "Here's a better response for this input"
    - "Mark this turn as a preferred answer for this persona"

Storage:
    JSONL files under memory/training_examples/
    - global.jsonl  — all examples across all sessions/personas
    - {session_id}.jsonl  — per-session examples

Export:
    Exports as JSONL in OpenAI fine-tuning format (messages: [{role, content}])
    or in a simple {prompt, completion} format for other frameworks.
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

def _examples_dir() -> Path:
    return get_runtime_config().paths.memory_root / 'training_examples'


def _global_path() -> Path:
    return _examples_dir() / 'global.jsonl'


def _session_path(session_id: str) -> Path:
    safe = re.sub(r'[^\w\-]', '_', str(session_id or 'default').strip())
    return _examples_dir() / f'{safe}.jsonl'


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Low-level I/O
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except json.JSONDecodeError:
            continue
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def _rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + ('\n' if rows else ''),
        encoding='utf-8',
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_example(
    *,
    input_text: str,
    correct_output: str,
    session_id: str = '',
    persona_name: str = '',
    notes: str = '',
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Save a (input, correct_output) training pair.

    Returns the saved example dict (with example_id and timestamp).
    """
    input_text = str(input_text or '').strip()
    correct_output = str(correct_output or '').strip()
    if not input_text or not correct_output:
        raise ValueError('input_text and correct_output must not be empty')

    example: dict[str, Any] = {
        'example_id': _new_id(),
        'input_text': input_text,
        'correct_output': correct_output,
        'session_id': str(session_id or ''),
        'persona_name': str(persona_name or ''),
        'notes': str(notes or ''),
        'tags': list(tags or []),
        'created_at': _utc_now(),
    }

    with _STORE_LOCK:
        _append_jsonl(_global_path(), example)
        if session_id:
            _append_jsonl(_session_path(session_id), example)

    return example


def list_examples(
    *,
    session_id: str = '',
    persona_name: str = '',
) -> list[dict[str, Any]]:
    """
    List training examples, optionally filtered by session or persona.

    If session_id is given, loads from the session file; otherwise loads
    from the global file and optionally filters by persona_name.
    """
    with _STORE_LOCK:
        if session_id:
            rows = _load_jsonl(_session_path(session_id))
        else:
            rows = _load_jsonl(_global_path())

    if persona_name:
        rows = [r for r in rows if r.get('persona_name') == persona_name]

    # Newest first
    rows.sort(key=lambda r: r.get('created_at', ''), reverse=True)
    return rows


def get_example(example_id: str) -> dict[str, Any] | None:
    """Look up a single example by its ID (searches global store)."""
    with _STORE_LOCK:
        rows = _load_jsonl(_global_path())
    for row in rows:
        if row.get('example_id') == example_id:
            return row
    return None


def delete_example(example_id: str) -> bool:
    """
    Delete an example by ID from both the global store and any session file.

    Returns True if found and deleted, False if not found.
    """
    with _STORE_LOCK:
        global_rows = _load_jsonl(_global_path())
        original_count = len(global_rows)
        global_rows = [r for r in global_rows if r.get('example_id') != example_id]

        if len(global_rows) == original_count:
            return False  # not found

        _rewrite_jsonl(_global_path(), global_rows)

        # Also remove from any session files that may contain this example
        examples_dir = _examples_dir()
        if examples_dir.exists():
            for session_file in examples_dir.glob('*.jsonl'):
                if session_file == _global_path():
                    continue
                session_rows = _load_jsonl(session_file)
                cleaned = [r for r in session_rows if r.get('example_id') != example_id]
                if len(cleaned) != len(session_rows):
                    _rewrite_jsonl(session_file, cleaned)

    return True


def export_jsonl(
    *,
    output_path: Path | str | None = None,
    persona_name: str = '',
    session_id: str = '',
    fmt: str = 'openai',  # 'openai' | 'simple'
) -> tuple[str, int]:
    """
    Export training examples as a JSONL string (and optionally write to file).

    fmt='openai'  → {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    fmt='simple'  → {"prompt": "...", "completion": "..."}

    Returns (jsonl_string, count).
    """
    rows = list_examples(session_id=session_id, persona_name=persona_name)

    lines: list[str] = []
    for row in rows:
        inp = str(row.get('input_text') or '')
        out = str(row.get('correct_output') or '')
        if not inp or not out:
            continue
        if fmt == 'openai':
            record: dict[str, Any] = {
                'messages': [
                    {'role': 'user', 'content': inp},
                    {'role': 'assistant', 'content': out},
                ]
            }
            if row.get('persona_name'):
                # Include system message with persona context when available
                record['messages'].insert(0, {
                    'role': 'system',
                    'content': f'You are {row["persona_name"]}.',
                })
        else:
            record = {'prompt': inp, 'completion': out}

        lines.append(json.dumps(record, ensure_ascii=False))

    content = '\n'.join(lines) + ('\n' if lines else '')

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')

    return content, len(lines)


def count_examples(*, session_id: str = '', persona_name: str = '') -> int:
    """Return number of stored training examples."""
    return len(list_examples(session_id=session_id, persona_name=persona_name))
