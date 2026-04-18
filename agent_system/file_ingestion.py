from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile
from typing import Any
import xml.etree.ElementTree as ET

from .classifier_forest import DEFAULT_CLASSIFIER
from .duplicate_resolver import normalize_name
from .entity_extractor import extract_knowledge
from .feature_extractor import extract_features
from .graph_store import GraphStore
from .history_store import parse_session, session_files_dir
from .models import HEAD_ENTITY_TYPES, MessageAnalysis, MessageEntity
from .persona_engine import load_active_persona, materialize_persona, process_persona_proposals, update_persona_from_examples
from .reliability import MutationRejectedFailure

SUPPORTED_EXTENSIONS = {'.txt', '.md', '.json', '.csv', '.pdf', '.docx', '.odt', '.fb2', '.djvu'}
_PDF_OUTLINE_STOPWORDS = {
    'contents',
    'table of contents',
    'оглавление',
    'содержание',
}
_PDF_TOC_LINE_RE = re.compile(
    r'^(?:(?P<number>\d+(?:\.\d+)*)\s+)?(?P<title>.+?)(?:\s+\.{2,}\s*|\s{2,}|\s+)(?P<page>\d{1,4})$'
)
_PDF_HEADING_RE = re.compile(r'^(?P<number>\d+(?:\.\d+)*[.)]?)\s+(?P<title>.+)$')


def _collapse_text(text: str) -> str:
    return ' '.join(str(text or '').split()).strip()


def _display_name(value: str) -> str:
    return ' '.join(str(value or '').replace('_', ' ').replace('-', ' ').split()).strip()


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


def _local_name(tag: str) -> str:
    raw = str(tag or '')
    return raw.split('}', 1)[-1] if '}' in raw else raw


def _paragraph_texts_from_xml(raw: bytes, *, paragraph_tags: set[str]) -> str:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return raw.decode('utf-8', errors='ignore')
    paragraphs: list[str] = []
    for element in root.iter():
        if _local_name(element.tag) not in paragraph_tags:
            continue
        text = _collapse_text(''.join(element.itertext()))
        if text:
            paragraphs.append(text)
    return '\n\n'.join(paragraphs).strip()


def _render_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            raw = archive.read('word/document.xml')
    except Exception:
        return ''
    return _paragraph_texts_from_xml(raw, paragraph_tags={'p'})


def _render_odt_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            raw = archive.read('content.xml')
    except Exception:
        return ''
    return _paragraph_texts_from_xml(raw, paragraph_tags={'p', 'h'})


def _render_fb2_text(raw: bytes) -> str:
    return _paragraph_texts_from_xml(raw, paragraph_tags={'p', 'subtitle', 'text-author'})


def _render_djvu_text(path: Path) -> str:
    # Try djvutxt (standard CLI tool shipped with djvulibre)
    try:
        result = subprocess.run(
            ['djvutxt', str(path)],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0:
            text = result.stdout.decode('utf-8', errors='ignore').strip()
            if text:
                return text
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: DJVU stores hidden text in XML annotations ("<HIDDENTEXT>" chunks).
    # Extract printable ASCII/UTF-8 runs from the raw binary.
    raw = path.read_bytes()
    # Look for DjVu hidden text annotation blocks (BZZ-compressed XML)
    # Simple fallback: extract printable sequences ≥ 4 chars
    printable = re.findall(rb'[\x20-\x7e\xc0-\xff]{4,}', raw)
    lines = []
    for chunk in printable:
        try:
            decoded = chunk.decode('utf-8', errors='ignore').strip()
        except Exception:
            decoded = chunk.decode('ascii', errors='ignore').strip()
        # Filter out binary noise (high ratio of special chars)
        alnum = sum(c.isalnum() or c.isspace() for c in decoded)
        if decoded and alnum / (len(decoded) + 1) > 0.5:
            lines.append(decoded)
    return '\n'.join(lines)


def _decode_pdf_literal(match: bytes) -> str:
    token = match[1:-1]
    token = re.sub(rb'\\([\\()])', rb'\1', token)
    token = token.replace(rb'\n', b'\n').replace(rb'\r', b' ').replace(rb'\t', b' ')
    return token.decode('utf-8', errors='ignore')


def _run_pdftotext(path: Path, *, layout: bool = False) -> str:
    args = ['pdftotext', '-enc', 'UTF-8']
    if layout:
        args.append('-layout')
    args.extend([str(path), '-'])
    try:
        completed = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.decode('utf-8', errors='ignore')
    except Exception:
        return ''


def _split_pdf_pages(text: str) -> list[str]:
    return [page.strip() for page in str(text or '').replace('\x0c', '\f').split('\f') if page.strip()]


def _parse_pdf_toc_line(line: str) -> dict[str, Any] | None:
    clean = ' '.join(str(line or '').strip().split())
    if len(clean) < 6 or len(clean) > 120:
        return None
    if normalize_name(clean) in _PDF_OUTLINE_STOPWORDS:
        return None
    match = _PDF_TOC_LINE_RE.match(clean)
    if not match:
        return None
    title = _collapse_text(match.group('title'))
    if not title:
        return None
    if normalize_name(title) in _PDF_OUTLINE_STOPWORDS:
        return None
    alpha_tokens = [token for token in title.split() if any(char.isalpha() for char in token)]
    if not alpha_tokens or len(alpha_tokens) > 14:
        return None
    try:
        page = max(int(match.group('page') or 0), 1)
    except ValueError:
        return None
    return {
        'title': title,
        'page': page,
        'number': str(match.group('number') or '').strip(),
    }


def _looks_like_pdf_heading(line: str) -> dict[str, Any] | None:
    clean = ' '.join(str(line or '').strip().split())
    if len(clean) < 4 or len(clean) > 90:
        return None
    if normalize_name(clean) in _PDF_OUTLINE_STOPWORDS:
        return None
    match = _PDF_HEADING_RE.match(clean)
    if match:
        return {
            'title': _collapse_text(match.group('title')),
            'page': 0,
            'number': str(match.group('number') or '').strip(),
        }
    tokens = clean.split()
    alpha_tokens = [token for token in tokens if any(char.isalpha() for char in token)]
    if not alpha_tokens or len(alpha_tokens) > 8:
        return None
    title_like_ratio = sum(1 for token in alpha_tokens if token[:1].isupper() or token.isupper()) / max(len(alpha_tokens), 1)
    if title_like_ratio < 0.8 and not clean.isupper():
        return None
    if clean.endswith(('.', ';', '?', '!')):
        return None
    return {'title': clean, 'page': 0, 'number': ''}


def _extract_pdf_outline(pages: list[str]) -> list[dict[str, Any]]:
    outline: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for page_index, page in enumerate(pages[:12], start=1):
        lines = [line for line in str(page or '').splitlines() if line.strip()]
        hits = [item for item in (_parse_pdf_toc_line(line) for line in lines) if item is not None]
        if len(hits) < 2:
            continue
        for hit in hits:
            title_key = normalize_name(str(hit.get('title') or ''))
            if not title_key or title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            outline.append(
                {
                    'title': str(hit.get('title') or '').strip(),
                    'page': max(int(hit.get('page') or page_index), 1),
                    'number': str(hit.get('number') or '').strip(),
                    'source': 'toc',
                }
            )
    if outline:
        return outline
    for page_index, page in enumerate(pages[:18], start=1):
        for line in str(page or '').splitlines():
            heading = _looks_like_pdf_heading(line)
            if heading is None:
                continue
            title_key = normalize_name(str(heading.get('title') or ''))
            if not title_key or title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            outline.append(
                {
                    'title': str(heading.get('title') or '').strip(),
                    'page': page_index,
                    'number': str(heading.get('number') or '').strip(),
                    'source': 'heading',
                }
            )
            if len(outline) >= 12:
                return outline
    return outline


def _section_facts(text: str, *, title: str) -> list[str]:
    facts: list[str] = []
    title_norm = normalize_name(title)
    for raw_line in str(text or '').splitlines():
        line = _collapse_text(raw_line)
        if not line:
            continue
        if normalize_name(line) in _PDF_OUTLINE_STOPWORDS:
            continue
        if normalize_name(line) == title_norm:
            continue
        if len(line) < 12:
            continue
        facts.append(line[:220])
        if len(facts) >= 4:
            break
    return facts


def _pdf_document_payload(path: Path) -> dict[str, Any]:
    layout_text = _run_pdftotext(path, layout=True)
    plain_text = _run_pdftotext(path, layout=False)
    pages = _split_pdf_pages(layout_text or plain_text)
    outline = _extract_pdf_outline(pages)
    section_records: list[dict[str, Any]] = []
    page_count = len(pages)
    for index, item in enumerate(outline):
        start_page = max(min(int(item.get('page') or 1), max(page_count, 1)), 1)
        next_page = int(outline[index + 1].get('page') or start_page + 1) if index + 1 < len(outline) else page_count
        end_page = max(start_page, min(max(next_page - 1, start_page), page_count or start_page))
        section_pages = pages[start_page - 1 : end_page]
        section_text = '\n\n'.join(page for page in section_pages if page).strip()
        section_records.append(
            {
                'title': str(item.get('title') or '').strip(),
                'number': str(item.get('number') or '').strip(),
                'start_page': start_page,
                'end_page': end_page,
                'facts': _section_facts(section_text, title=str(item.get('title') or '')),
                'text': section_text,
                'source': str(item.get('source') or 'toc'),
            }
        )
    if plain_text.strip():
        text = plain_text.strip()
    elif pages:
        text = '\n\n'.join(pages).strip()
    else:
        raw = path.read_bytes()
        literals = [_collapse_text(_decode_pdf_literal(match)) for match in re.findall(rb'\((?:\\.|[^\\()])+\)', raw)]
        text = '\n\n'.join(item for item in literals if item).strip()
    return {
        'text': text,
        'pages': pages,
        'page_count': page_count,
        'outline': outline,
        'sections': section_records,
    }


def _render_pdf_text(path: Path) -> str:
    payload = _pdf_document_payload(path)
    return str(payload.get('text') or '').strip()


def _build_pdf_skeleton_extraction(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    document_title = _display_name(path.stem) or path.stem or path.name
    file_name = path.name
    outline = [str(item.get('title') or '').strip() for item in list(payload.get('outline') or []) if str(item.get('title') or '').strip()]
    page_count = int(payload.get('page_count') or 0)
    entities: list[dict[str, Any]] = [
        {
            'name': document_title,
            'aliases': [file_name],
            'type': 'OBJECT',
            'description': f'PDF document "{file_name}" ingested for contextual retrieval.',
            'facts': [
                *( [f'Document type: PDF.'] ),
                *( [f'Page count: {page_count}.'] if page_count else [] ),
                *( [f'Sections: {", ".join(outline[:8])}.'] if outline else [] ),
            ],
            'importance': 0.72,
            'confidence': 0.92,
            'context': {
                'source': 'file',
                'document_type': 'pdf',
                'file_name': file_name,
                'file_path': str(path),
                'document_role': 'document_root',
            },
        }
    ]
    relations: list[dict[str, Any]] = []
    previous_section_name = ''
    for section in list(payload.get('sections') or [])[:24]:
        title = str(section.get('title') or '').strip()
        if not title:
            continue
        section_name = f'{document_title}: {title}'
        entities.append(
            {
                'name': section_name,
                'aliases': [title, f'{document_title} {title}'],
                'type': 'CONCEPT',
                'description': f'Section "{title}" from PDF "{file_name}".',
                'facts': list(section.get('facts') or [])[:4],
                'importance': 0.68,
                'confidence': 0.88,
                'context': {
                    'source': 'file',
                    'document_type': 'pdf',
                    'file_name': file_name,
                    'file_path': str(path),
                    'document_name': document_title,
                    'section_title': title,
                    'section_number': str(section.get('number') or '').strip(),
                    'start_page': int(section.get('start_page') or 0),
                    'end_page': int(section.get('end_page') or 0),
                    'document_role': 'document_section',
                },
            }
        )
        relations.append({'from': document_title, 'to': section_name, 'type': 'HAS_SECTION', 'weight': 0.88, 'confidence': 0.9})
        if previous_section_name:
            relations.append(
                {'from': previous_section_name, 'to': section_name, 'type': 'NEXT_SECTION', 'weight': 0.74, 'confidence': 0.8}
            )
        previous_section_name = section_name
    return {'entities': entities, 'relations': relations}


def _pdf_section_chunks(path: Path, payload: dict[str, Any]) -> list[str]:
    document_title = _display_name(path.stem) or path.stem or path.name
    chunks: list[str] = []
    for section in list(payload.get('sections') or []):
        section_text = str(section.get('text') or '').strip()
        title = str(section.get('title') or '').strip()
        if not section_text or not title:
            continue
        wrapped = '\n'.join(
            [
                f'Document: {document_title}',
                f'File: {path.name}',
                f'Section: {title}',
                f'Pages: {int(section.get("start_page") or 0)}-{int(section.get("end_page") or 0)}',
                '',
                section_text,
            ]
        ).strip()
        chunks.extend(chunk_text(wrapped))
    return chunks
    raw = path.read_bytes()
    literals = [_collapse_text(_decode_pdf_literal(match)) for match in re.findall(rb'\((?:\\.|[^\\()])+\)', raw)]
    literals = [item for item in literals if item]
    return '\n\n'.join(literals).strip()


def file_to_text(path: Path) -> str:
    suffix = path.suffix.lower()
    raw = path.read_bytes()
    if suffix in {'.txt', '.md'}:
        return raw.decode('utf-8', errors='ignore')
    if suffix == '.json':
        return _render_json_text(raw)
    if suffix == '.csv':
        return _render_csv_text(raw)
    if suffix == '.pdf':
        return _render_pdf_text(path)
    if suffix == '.docx':
        return _render_docx_text(path)
    if suffix == '.odt':
        return _render_odt_text(path)
    if suffix == '.fb2':
        return _render_fb2_text(raw)
    if suffix == '.djvu':
        return _render_djvu_text(path)
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
        try:
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
        except MutationRejectedFailure:
            continue
        touched.append(bundle.name)
    return touched


def extract_text_knowledge(
    text: str,
    *,
    source: str,
    session_id: str = '',
    store: GraphStore | None = None,
    chunks: list[str] | None = None,
    prelude_extractions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    graph_store = store or GraphStore()
    extraction_chunks = [str(item).strip() for item in list(chunks or chunk_text(text)) if str(item).strip()]
    prelude = [dict(item) for item in list(prelude_extractions or []) if isinstance(item, dict)]
    if not extraction_chunks and not prelude:
        return {'ok': False, 'reason': 'empty_text', 'chunk_count': 0, 'merged': []}
    merged: list[dict[str, Any]] = []
    touched_heads: set[str] = set()
    for extraction in prelude:
        if not extraction.get('entities') and not extraction.get('relations'):
            continue
        merged.append(graph_store.merge_extraction(extraction, source=source, session_id=session_id))
        touched_heads.update(_update_heads(extraction))
    for chunk in extraction_chunks:
        extraction = extract_knowledge(chunk, source=source)
        if not extraction['entities'] and not extraction['relations']:
            continue
        merged.append(graph_store.merge_extraction(extraction, source=source, session_id=session_id))
        touched_heads.update(_update_heads(extraction))
    if not merged:
        return {'ok': False, 'reason': 'no_valid_proposals', 'chunk_count': len(extraction_chunks), 'merged': []}
    return {
        'ok': True,
        'chunk_count': len(extraction_chunks),
        'merged': merged,
        'heads': sorted(touched_heads),
        'prelude_count': len(prelude),
    }


def extract_session(session_id: str, *, store: GraphStore | None = None) -> dict[str, Any]:
    parsed = parse_session(session_id)
    if not parsed:
        return {'ok': False, 'reason': 'missing_session', 'session_id': session_id}
    result = extract_text_knowledge(session_text(parsed), source='session', session_id=session_id, store=store)
    result['session_id'] = session_id
    return result


def ingest_file(path: Path, *, session_id: str = '', store: GraphStore | None = None) -> dict[str, Any]:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return {'ok': False, 'reason': 'unsupported_extension', 'path': str(path)}
    text = ''
    chunks: list[str] | None = None
    prelude_extractions: list[dict[str, Any]] = []
    extra_meta: dict[str, Any] = {}
    if path.suffix.lower() == '.pdf':
        payload = _pdf_document_payload(path)
        text = str(payload.get('text') or '').strip()
        outline = list(payload.get('outline') or [])
        sections = list(payload.get('sections') or [])
        chunks = _pdf_section_chunks(path, payload) or None
        if outline or sections:
            prelude_extractions.append(_build_pdf_skeleton_extraction(path, payload))
        extra_meta = {
            'document_outline_count': len(outline),
            'document_section_count': len(sections),
            'document_page_count': int(payload.get('page_count') or 0),
        }
    else:
        text = file_to_text(path)
    if not text.strip():
        return {'ok': False, 'reason': 'empty_file', 'path': str(path)}
    result = extract_text_knowledge(
        text,
        source='file',
        session_id=session_id,
        store=store,
        chunks=chunks,
        prelude_extractions=prelude_extractions,
    )
    result['path'] = str(path)
    result.update(extra_meta)
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
            result = ingest_file(path, session_id=session_id, store=graph_store)
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
                active_persona = load_active_persona(personality_name)
                if active_persona is not None:
                    update_persona_from_examples(active_persona.name, examples)
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
