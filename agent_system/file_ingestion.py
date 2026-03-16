from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from .classifier_forest import DEFAULT_CLASSIFIER
from .entity_extractor import extract_knowledge
from .feature_extractor import extract_features
from .graph_store import GraphStore
from .history_store import parse_session, session_files_dir
from .models import HEAD_ENTITY_TYPES, MessageAnalysis, MessageEntity
from .persona_engine import materialize_persona, process_persona_proposals, spawn_head, update_persona_from_examples

SUPPORTED_EXTENSIONS = {'.txt', '.md', '.json', '.csv'}


def session_text(parsed: dict[str, Any]) -> str:
    lines: list[str] = []
    for item in list(parsed.get('messages') or []):
        role = str(item.get('role') or 'assistant')
        message = str(item.get('message') or '').strip()
        if role and message:
            lines.append(f'{role}: {message}')
    return '\n'.join(lines).strip()


def chunk_text(text: str, *, max_tokens: int = 2000, overlap_tokens: int = 200) -> list[str]:
    raw = str(text or '').strip()
    if not raw:
        return []
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4
    paragraphs = [part.strip() for part in raw.split('\n\n') if part.strip()]
    chunks: list[str] = []
    current = ''
    for paragraph in paragraphs:
        candidate = f'{current}\n\n{paragraph}'.strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap_chars:].strip()
            current = f'{tail}\n\n{paragraph}'.strip() if tail else paragraph
            continue
        while len(paragraph) > max_chars:
            chunks.append(paragraph[:max_chars].strip())
            paragraph = paragraph[max_chars - overlap_chars :].strip()
        current = paragraph
    if current:
        chunks.append(current)
    return chunks


def _render_json_text(raw: bytes) -> str:
    try:
        payload = json.loads(raw.decode('utf-8'))
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        return raw.decode('utf-8', errors='ignore')


def _render_csv_text(raw: bytes) -> str:
    text = raw.decode('utf-8', errors='ignore')
    reader = csv.DictReader(io.StringIO(text))
    lines: list[str] = []
    for row in reader:
        if not isinstance(row, dict):
            continue
        lines.append(', '.join(f'{key}={value}' for key, value in row.items()))
    return '\n'.join(lines).strip() or text


def file_to_text(path: Path) -> str:
    suffix = path.suffix.lower()
    raw = path.read_bytes()
    if suffix in {'.txt', '.md'}:
        return raw.decode('utf-8', errors='ignore')
    if suffix == '.json':
        return _render_json_text(raw)
    if suffix == '.csv':
        return _render_csv_text(raw)
    return ''


def store_uploaded_file(session_id: str, filename: str, content: bytes) -> Path:
    safe_name = Path(filename or 'upload.txt').name
    path = session_files_dir(session_id) / safe_name
    path.write_bytes(content)
    return path


def _entity_knowledge(entity: dict[str, Any]) -> str:
    facts = [str(item).strip() for item in list(entity.get('facts') or []) if str(item).strip()]
    parts = [str(entity.get('description') or '').strip()] + facts
    return '\n'.join(part for part in parts if part).strip()


def _relations_for(name: str, extraction: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relation in list(extraction.get('relations') or []):
        if not isinstance(relation, dict):
            continue
        if str(relation.get('from') or '').strip() != str(name or '').strip():
            continue
        rows.append({'type': str(relation.get('type') or 'RELATED_TO').strip().upper(), 'target': str(relation.get('to') or '').strip()})
    return rows


def _classify_entity(entity: dict[str, Any]) -> str:
    declared = str(entity.get('type') or '').strip().upper()
    if declared in HEAD_ENTITY_TYPES:
        return declared
    description = str(entity.get('description') or '').strip()
    facts = ' '.join(str(item).strip() for item in list(entity.get('facts') or []) if str(item).strip())
    analysis = MessageAnalysis(
        message=' '.join(part for part in (description, facts) if part),
        session_id='ingestion',
        explicit_context=json.dumps(entity.get('context') or {}, ensure_ascii=False),
        primary_entity=str(entity.get('name') or '').strip(),
    )
    features = extract_features(
        MessageEntity(
            name=str(entity.get('name') or '').strip(),
            description=' '.join(part for part in (description, facts) if part),
            aliases=[str(item).strip() for item in list(entity.get('aliases') or []) if str(item).strip()],
        ),
        analysis,
    )
    decision = DEFAULT_CLASSIFIER.classify(features)
    feature_map = features.feature_map
    if decision.entity_type == 'FICTIONAL_CHARACTER' and feature_map.get('fictional_hint_score', 0.0) > 0:
        return decision.entity_type
    if decision.entity_type == 'PROFESSION' and feature_map.get('profession_hint_score', 0.0) > 0:
        return decision.entity_type
    if decision.entity_type == 'PERSON' and (
        feature_map.get('person_hint_score', 0.0) > 0 or feature_map.get('title_case_ratio', 0.0) >= 0.75
    ):
        return decision.entity_type
    return 'CONCEPT'


def _update_heads(extraction: dict[str, Any]) -> list[str]:
    touched: list[str] = []
    for entity in list(extraction.get('entities') or []):
        if not isinstance(entity, dict):
            continue
        name = str(entity.get('name') or '').strip()
        if not name:
            continue
        entity_type = _classify_entity(entity)
        if entity_type not in HEAD_ENTITY_TYPES:
            continue
        bundle = materialize_persona(
            name,
            {
                'entity_type': entity_type,
                'aliases': list(entity.get('aliases') or []),
                'traits': list(entity.get('traits') or []),
                'examples': [fact for fact in list(entity.get('facts') or []) if str(fact).strip()],
                'relations': _relations_for(name, extraction),
                'knowledge': _entity_knowledge(entity),
            },
        )
        touched.append(bundle.name)
    return touched


def extract_text_knowledge(text: str, *, source: str, store: GraphStore | None = None) -> dict[str, Any]:
    graph_store = store or GraphStore()
    chunks = chunk_text(text)
    if not chunks:
        return {'ok': False, 'reason': 'empty_text', 'chunk_count': 0, 'merged': []}
    merged: list[dict[str, Any]] = []
    touched_heads: set[str] = set()
    for chunk in chunks:
        extraction = extract_knowledge(chunk, source=source)
        if not extraction['entities'] and not extraction['relations']:
            continue
        merged.append(graph_store.merge_extraction(extraction, source=source))
        touched_heads.update(_update_heads(extraction))
    if not merged:
        return {'ok': False, 'reason': 'no_valid_proposals', 'chunk_count': len(chunks), 'merged': []}
    return {'ok': True, 'chunk_count': len(chunks), 'merged': merged, 'heads': sorted(touched_heads)}


def extract_session(session_id: str, *, store: GraphStore | None = None) -> dict[str, Any]:
    parsed = parse_session(session_id)
    if not parsed:
        return {'ok': False, 'reason': 'missing_session', 'session_id': session_id}
    result = extract_text_knowledge(session_text(parsed), source='session', store=store)
    result['session_id'] = session_id
    return result


def ingest_file(path: Path, *, store: GraphStore | None = None) -> dict[str, Any]:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return {'ok': False, 'reason': 'unsupported_extension', 'path': str(path)}
    text = file_to_text(path)
    if not text.strip():
        return {'ok': False, 'reason': 'empty_file', 'path': str(path)}
    result = extract_text_knowledge(text, source='file', store=store)
    result['path'] = str(path)
    return result


def rebuild_artifacts(session_id: str, *, personality_name: str = '', store: GraphStore | None = None) -> dict[str, Any]:
    graph_store = store or GraphStore()
    errors: list[str] = []
    session_result = extract_session(session_id, store=graph_store)
    if not session_result.get('ok'):
        errors.append(f"session:{session_result.get('reason', 'unknown_error')}")
    file_results: list[dict[str, Any]] = []
    for path in sorted(session_files_dir(session_id).glob('*')):
        if path.is_file():
            result = ingest_file(path, store=graph_store)
            file_results.append(result)
            if not result.get('ok'):
                errors.append(f"file:{Path(result.get('path') or path).name}:{result.get('reason', 'unknown_error')}")
    if personality_name:
        parsed = parse_session(session_id)
        if parsed:
            examples = [
                str(item.get('message') or '').strip()
                for item in list(parsed.get('messages') or [])
                if str(item.get('role') or '').strip() == 'user' and str(item.get('message') or '').strip()
            ]
            if examples:
                spawn_head(personality_name, entity_type='PERSON', source='rebuild')
                update_persona_from_examples(personality_name, examples)
    proposal_results = process_persona_proposals()
    hygiene = graph_store.apply_hygiene()
    if not hygiene.get('ok'):
        errors.append(f"hygiene:{hygiene.get('reason', 'failed')}")
    validation = graph_store.validate_graph()
    if not validation.get('ok'):
        errors.extend(str(item) for item in list(validation.get('errors') or []))
    return {
        'ok': (
            (bool(session_result.get('ok')) or any(item.get('ok') for item in file_results) or bool(proposal_results))
            and bool(validation.get('ok'))
        ),
        'session': session_result,
        'files': file_results,
        'personality_updates': proposal_results,
        'hygiene': hygiene,
        'validation': validation,
        'errors': errors,
        'graph': graph_store.load_graph(),
    }
