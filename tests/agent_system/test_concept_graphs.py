from __future__ import annotations

from agent_system.chat_engine import generate_response
from agent_system.duplicate_resolver import normalize_name
from agent_system.entity_extractor import extract_knowledge, validate_extraction
from agent_system.graph_store import GraphStore


def test_extract_knowledge_builds_human_concept_graph_from_structural_request(monkeypatch) -> None:
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat': '')

    extraction = extract_knowledge(
        'Построй граф человека как системы понятий: биология, психология, социология, язык, культура и этика.',
        source='session',
    )

    names = {normalize_name(str(entity.get('name') or '')) for entity in extraction['entities']}
    assert {'human', 'biology', 'psychology', 'sociology', 'language', 'culture', 'ethics'} <= names

    human = next(entity for entity in extraction['entities'] if normalize_name(str(entity.get('name') or '')) == 'human')
    assert human['translation_line'] == 'Human: человек, մարդ'
    assert 'multi-layered living and social system' in str(human.get('description') or '')
    assert any(
        normalize_name(str(relation.get('from') or '')) == 'human'
        and normalize_name(str(relation.get('to') or '')) == 'biology'
        and str(relation.get('type') or '') == 'HAS_DIMENSION'
        for relation in extraction['relations']
    )


def test_extract_knowledge_uses_fast_structured_role(monkeypatch) -> None:
    captured: list[str] = []

    def fake_model(prompt: str, mode: str = 'chat', *, role: str = 'general') -> str:
        captured.append(role)
        return '{"entities":[],"relations":[]}'

    monkeypatch.setattr('agent_system.llm._call_model', fake_model)

    extract_knowledge('Explain sunlight as a concept node.', source='session')

    assert captured
    assert captured[-1] == 'analyst'


def test_validate_extraction_rejects_sentence_fragment_nodes() -> None:
    validated = validate_extraction(
        {
            'entities': [
                {
                    'name': 'human is a biological and social being',
                    'description': 'A sentence fragment that should not become a node.',
                    'facts': [],
                },
                {
                    'name': 'Psychology',
                    'description': 'Mental life and regulation of action.',
                    'facts': [],
                },
            ],
            'relations': [
                {'from': 'human is a biological and social being', 'to': 'Psychology', 'type': 'RELATED_TO'},
                {'from': 'Human', 'to': 'Psychology', 'type': 'HAS_DIMENSION'},
            ],
        },
        source='test',
    )

    names = {normalize_name(str(entity.get('name') or '')) for entity in validated['entities']}
    assert 'human is a biological and social being' not in names
    assert 'psychology' in names
    assert not any(
        normalize_name(str(relation.get('from') or '')) == 'human is a biological and social being'
        for relation in validated['relations']
    )


def test_chat_request_can_materialize_human_concept_graph_immediately(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat': 'Вот концептуальный граф человека.')

    generate_response(
        message='Сделай граф человека: биология, психология, социология, культура, язык и мышление.',
        session_id='human_graph_session',
        language='ru',
    )

    graph = GraphStore().subgraph('человек', limit=12)
    names = {normalize_name(str(node.get('name') or '')) for node in graph['nodes']}
    assert {'human', 'biology', 'psychology', 'sociology', 'language', 'culture', 'cognition'} <= names
    human = next(node for node in graph['nodes'] if normalize_name(str(node.get('name') or '')) == 'human')
    assert human['translation_line'] == 'Human: человек, մարդ'
    assert not any(str(node.get('id') or '').startswith('summary:') for node in graph['nodes'])
