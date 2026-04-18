"""
Personality store.

Persists PersonalityObject instances to disk under memory/personalities/.
Each personality is stored as a single JSON file named by personality_id.

An index file (personalities_index.json) maps:
    persona_name → personality_id

This allows:
    - fast lookup by persona slug
    - listing all stored personalities
    - multiple personalities over time (revisions are not yet kept here,
      but the merge engine provides in-memory history via provenance)
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .personality_schema import PersonalityObject, PersonalityStateMeta
from .runtime_config import get_runtime_config

_STORE_LOCK = Lock()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _personalities_dir() -> Path:
    return get_runtime_config().paths.memory_root / 'personalities'


def _personalities_index_path() -> Path:
    return _personalities_dir() / 'personalities_index.json'


def _personality_path(personality_id: str) -> Path:
    safe = re.sub(r'[^\w\-]', '_', str(personality_id or ''))
    return _personalities_dir() / f'{safe}.json'


# ---------------------------------------------------------------------------
# Low-level I/O
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def _load_index() -> dict[str, str]:
    raw = _load_json(_personalities_index_path())
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if k and v}


def _save_index(index: dict[str, str]) -> None:
    _write_json(_personalities_index_path(), index)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_personality_id(persona_name: str = '') -> str:
    """Generate a stable personality_id from a persona_name or UUID."""
    if persona_name:
        base = re.sub(r'[^\w]', '_', str(persona_name).lower().strip())
        ts = datetime.now(UTC).strftime('%Y%m%dT%H%M%S')
        return f'personality_{base}_{ts}'
    return f'personality_{uuid.uuid4().hex[:12]}'


def save_personality(personality: PersonalityObject) -> str:
    """
    Persist a PersonalityObject to disk.

    If personality_id is empty, a new one is generated.
    Returns the personality_id.
    """
    with _STORE_LOCK:
        if not personality.personality_id:
            personality.personality_id = generate_personality_id(personality.persona_name)

        pid = personality.personality_id
        personality.state_meta.updated_at = datetime.now(UTC).isoformat()

        path = _personality_path(pid)
        _write_json(path, personality.to_dict())

        # Update index
        index = _load_index()
        if personality.persona_name:
            index[str(personality.persona_name)] = pid
        index[pid] = pid
        _save_index(index)

        return pid


def load_personality(personality_id: str) -> PersonalityObject | None:
    """Load a PersonalityObject by its personality_id. Returns None if not found."""
    path = _personality_path(str(personality_id or ''))
    raw = _load_json(path)
    if not isinstance(raw, dict):
        return None
    try:
        return PersonalityObject.from_dict(raw)
    except Exception:
        return None


def load_personality_for_persona(persona_name: str) -> PersonalityObject | None:
    """
    Load the PersonalityObject associated with a persona slug.

    Returns None if no personality has been registered for this persona.
    """
    index = _load_index()
    pid = index.get(str(persona_name or ''))
    if not pid:
        return None
    return load_personality(pid)


def get_or_create_personality(
    persona_name: str,
    *,
    label: str = '',
    source_texts: list[str] | None = None,
) -> PersonalityObject:
    """
    Load existing personality for a persona, or create a fresh one.

    This is the preferred entry point when the chat engine or API needs a
    personality for a named persona.
    """
    existing = load_personality_for_persona(persona_name)
    if existing is not None:
        return existing

    from .personality_schema import PersonalityIdentity
    pid = generate_personality_id(persona_name)
    personality = PersonalityObject(
        personality_id=pid,
        persona_name=str(persona_name or ''),
        identity=PersonalityIdentity(
            label=str(label or persona_name or ''),
            source_texts=list(source_texts or []),
        ),
    )
    save_personality(personality)
    return personality


def list_personalities() -> list[dict[str, Any]]:
    """Return a list of summary dicts for all stored personalities."""
    index = _load_index()
    seen_ids: set[str] = set()
    result: list[dict[str, Any]] = []

    # Deduplicate: the index has both persona_name→id and id→id entries
    for pid in index.values():
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        personality = load_personality(pid)
        if personality is None:
            continue
        result.append({
            'personality_id': personality.personality_id,
            'persona_name': personality.persona_name,
            'label': personality.identity.label,
            'readiness': personality.state_meta.readiness,
            'confidence': personality.state_meta.confidence,
            'completeness': personality.state_meta.completeness,
            'validation_status': personality.state_meta.validation_status,
            'updated_at': personality.state_meta.updated_at,
            'hover_summary': personality.state_meta.hover_summary,
        })
    return result


def delete_personality(personality_id: str) -> bool:
    """Delete a personality from disk and remove from index. Returns True if found."""
    with _STORE_LOCK:
        path = _personality_path(str(personality_id or ''))
        if not path.exists():
            return False
        path.unlink(missing_ok=True)
        index = _load_index()
        keys_to_remove = [k for k, v in index.items() if v == personality_id]
        for k in keys_to_remove:
            index.pop(k, None)
        _save_index(index)
        return True


def list_biography_facts(
    personality_id: str,
    *,
    fact_type: str = '',
) -> list[dict[str, Any]]:
    """
    Return biography facts for a personality, optionally filtered by fact_type.
    """
    personality = load_personality(personality_id)
    if personality is None:
        return []
    bio = personality.biography
    all_facts = list(bio.facts) + list(bio.timeline) + list(bio.relationships)
    if fact_type:
        all_facts = [f for f in all_facts if f.fact_type == fact_type]
    return [f.to_dict() for f in all_facts]


def list_provenance(personality_id: str) -> list[dict[str, Any]]:
    """
    Return a flat list of all provenance records across all profile fields.

    Useful for operator inspection: "where did each fact come from?"
    """
    personality = load_personality(personality_id)
    if personality is None:
        return []

    records: list[dict[str, Any]] = []
    for field_name in personality.profile.all_field_names():
        for entry in personality.profile.get_field(field_name):
            for prov in entry.provenance:
                records.append({
                    'field_name': field_name,
                    'entry_id': entry.entry_id,
                    'value': entry.value,
                    'confidence': entry.confidence,
                    **prov.to_dict(),
                })
    # Biography provenance
    bio = personality.biography
    for fact in list(bio.facts) + list(bio.timeline) + list(bio.relationships):
        records.append({
            'field_name': f'biography.{fact.fact_type}',
            'entry_id': fact.fact_id,
            'value': fact.value,
            'confidence': fact.confidence,
            'source': fact.source,
            'source_channel': fact.source_channel,
            'session_id': fact.session_id,
            'timestamp': fact.timestamp,
            'evidence_excerpt': fact.evidence_excerpt,
        })
    return records


def list_conflicts(personality_id: str) -> list[dict[str, Any]]:
    """Return all unresolved conflict records for a personality."""
    personality = load_personality(personality_id)
    if personality is None:
        return []

    conflicts: list[dict[str, Any]] = []
    for field_name in personality.profile.all_field_names():
        for entry in personality.profile.get_field(field_name):
            for conflict in entry.conflicts:
                if not conflict.resolved:
                    conflicts.append({
                        'location': f'profile.{field_name}',
                        'entry_id': entry.entry_id,
                        'entry_value': entry.value,
                        **conflict.to_dict(),
                    })
    bio = personality.biography
    for fact in list(bio.facts) + list(bio.timeline) + list(bio.relationships):
        for conflict in fact.conflicts:
            if not conflict.resolved:
                conflicts.append({
                    'location': f'biography.{fact.fact_type}',
                    'entry_id': fact.fact_id,
                    'entry_value': fact.value,
                    **conflict.to_dict(),
                })
    return conflicts
