from __future__ import annotations

import json
import zipfile

from agent_system.file_ingestion import chunk_text, file_to_text, ingest_file, rebuild_artifacts, store_uploaded_file
from agent_system.graph_store import GraphStore
from agent_system.history_store import append_turn, create_session
from agent_system.persona_engine import load_persona


def test_file_ingestion_chunks_under_token_budget_and_updates_heads(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    def fake_model(prompt: str, mode: str = 'chat') -> str:
        if mode != 'knowledge':
            return ''
        return json.dumps(
            {
                'entities': [
                    {
                        'name': 'Dracula',
                        'aliases': ['Count Dracula'],
                        'description': 'Fictional vampire nobleman.',
                        'facts': ['Dracula feeds on humans.', 'Dracula fears sunlight.'],
                        'context': {'source': 'file'},
                    },
                    {'name': 'humans', 'aliases': [], 'description': 'People.', 'facts': [], 'context': {'source': 'file'}},
                    {'name': 'sunlight', 'aliases': [], 'description': 'Daylight.', 'facts': [], 'context': {'source': 'file'}},
                ],
                'relations': [
                    {'from': 'Dracula', 'to': 'humans', 'type': 'FEEDS_ON', 'weight': 0.9},
                    {'from': 'Dracula', 'to': 'sunlight', 'type': 'FEARS', 'weight': 0.8},
                ],
            }
        )

    monkeypatch.setattr('agent_system.llm._call_model', fake_model)

    large_text = ('Dracula is a vampire nobleman who fears sunlight.\n\n' * 600).strip()
    chunks = chunk_text(large_text)
    assert len(chunks) > 1
    assert all(len(chunk) <= 8000 for chunk in chunks)

    create_session('session_test', 'Session')
    path = store_uploaded_file('session_test', 'dracula.txt', large_text.encode('utf-8'))
    result = ingest_file(path)
    assert result['ok'] is True

    bundle = load_persona('dracula')
    assert bundle is not None
    assert any(relation['type'] == 'FEEDS_ON' for relation in bundle.relations)
    assert load_persona('humans') is None
    assert load_persona('sunlight') is None
    graph = GraphStore().load_graph()
    dracula = next(node for node in graph['nodes'] if str(node.get('name') or '').lower() == 'dracula')
    assert 'Count Dracula' in dracula['aliases']


def test_rebuild_artifacts_learns_from_session_and_uploaded_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    def fake_model(prompt: str, mode: str = 'chat') -> str:
        if mode != 'knowledge':
            return ''
        return json.dumps(
            {
                'entities': [{'name': 'Dracula', 'aliases': [], 'description': 'Fictional vampire nobleman.', 'facts': ['Dracula is a vampire.'], 'context': {'source': 'session'}}],
                'relations': [],
            }
        )

    monkeypatch.setattr('agent_system.llm._call_model', fake_model)

    create_session('session_test', 'Session')
    append_turn('session_test', 'Do you know Dracula?', 'Yes.')
    store_uploaded_file('session_test', 'notes.md', b'Dracula is a fictional vampire nobleman.')
    result = rebuild_artifacts('session_test', personality_name='Dracula')

    assert result['ok'] is True
    assert result['validation']['ok'] is True
    assert not result['errors']
    bundle = load_persona('dracula')
    assert bundle is not None
    assert 'Yes.' not in bundle.examples
    graph = GraphStore().load_graph()
    dracula = next(node for node in graph['nodes'] if str(node.get('name') or '').lower() == 'dracula')
    session_ids = list(dict(dracula.get('context') or {}).get('session_ids') or [])
    assert 'session_test' in session_ids


def test_file_to_text_supports_docx_odt_and_fb2(tmp_path) -> None:
    docx_path = tmp_path / 'sample.docx'
    with zipfile.ZipFile(docx_path, 'w') as archive:
        archive.writestr(
            'word/document.xml',
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                '<w:body><w:p><w:r><w:t>Dracula fears sunlight.</w:t></w:r></w:p></w:body>'
                '</w:document>'
            ),
        )

    odt_path = tmp_path / 'sample.odt'
    with zipfile.ZipFile(odt_path, 'w') as archive:
        archive.writestr(
            'content.xml',
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<office:document-content '
                'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
                'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
                '<office:body><office:text><text:p>Dracula feeds on humans.</text:p></office:text></office:body>'
                '</office:document-content>'
            ),
        )

    fb2_path = tmp_path / 'sample.fb2'
    fb2_path.write_text(
        (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">'
            '<body><section><title><p>Dracula</p></title><p>Immortal vampire nobleman.</p></section></body>'
            '</FictionBook>'
        ),
        encoding='utf-8',
    )

    assert 'Dracula fears sunlight.' in file_to_text(docx_path)
    assert 'Dracula feeds on humans.' in file_to_text(odt_path)
    fb2_text = file_to_text(fb2_path)
    assert 'Dracula' in fb2_text
    assert 'Immortal vampire nobleman.' in fb2_text


def test_ingest_file_supports_pdf_via_pdftotext(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    def fake_model(prompt: str, mode: str = 'chat') -> str:
        if mode != 'knowledge':
            return ''
        return json.dumps(
            {
                'entities': [
                    {
                        'name': 'Dracula',
                        'aliases': [],
                        'description': 'Immortal vampire nobleman.',
                        'facts': ['Dracula fears sunlight.'],
                        'context': {'source': 'file'},
                    }
                ],
                'relations': [],
            }
        )

    class Completed:
        def __init__(self, stdout: bytes) -> None:
            self.stdout = stdout
            self.stderr = b''
            self.returncode = 0

    monkeypatch.setattr('agent_system.llm._call_model', fake_model)
    monkeypatch.setattr(
        'agent_system.file_ingestion.subprocess.run',
        lambda *args, **kwargs: Completed(b'Dracula fears sunlight.'),
    )

    pdf_path = tmp_path / 'sample.pdf'
    pdf_path.write_bytes(b'%PDF-1.4\n1 0 obj\n<<>>\nendobj\n')
    result = ingest_file(pdf_path)

    assert result['ok'] is True
    graph = GraphStore().load_graph()
    assert any(str(node.get('name') or '').lower() == 'dracula' for node in graph['nodes'])


def test_ingest_pdf_builds_outline_backed_section_graph(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    class Completed:
        def __init__(self, stdout: bytes) -> None:
            self.stdout = stdout
            self.stderr = b''
            self.returncode = 0

    pdf_text = (
        'Contents\n'
        '1 Storm Systems ........ 2\n'
        '2 Crew Discipline ........ 4\n'
        '\f'
        'Storm Systems\n'
        'Wind shear and wave impact determine immediate ship handling.\n'
        '\f'
        'Crew Discipline\n'
        'Clear orders keep the crew coordinated under pressure.\n'
    )

    monkeypatch.setattr(
        'agent_system.file_ingestion.subprocess.run',
        lambda *args, **kwargs: Completed(pdf_text.encode('utf-8')),
    )
    monkeypatch.setattr(
        'agent_system.llm._call_model',
        lambda *args, **kwargs: json.dumps({'entities': [], 'relations': []}) if kwargs.get('mode') == 'knowledge' else '',
    )

    pdf_path = tmp_path / 'captain_manual.pdf'
    pdf_path.write_bytes(b'%PDF-1.4\n1 0 obj\n<<>>\nendobj\n')
    result = ingest_file(pdf_path)

    assert result['ok'] is True
    assert result['document_outline_count'] >= 2
    assert result['document_section_count'] >= 2

    store = GraphStore()
    graph = store.load_graph()
    names = {str(node.get('name') or '') for node in graph['nodes']}
    assert 'captain manual' in {name.lower() for name in names}
    assert any('Storm Systems' in name for name in names)
    assert any('Crew Discipline' in name for name in names)
    assert any(str(edge.get('type') or '') == 'HAS_SECTION' for edge in graph['edges'])

    ranked = store.search_nodes('crew pressure orders', limit=4)
    assert ranked
    assert any('Crew Discipline' in str(node.get('name') or '') for node in ranked)
