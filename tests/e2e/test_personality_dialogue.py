from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.web.combined_app import create_combined_app


def _fake_model(prompt: str, mode: str = 'chat') -> str:
    lowered = prompt.lower()
    if 'schema:' in lowered and 'name: dracula' in lowered:
        return json.dumps({
            'name': 'dracula',
            'traits': ['вампир', 'аристократический', 'бессмертный'],
            'patterns': ['пьет_кровь', 'избегает_солнца'],
            'examples': ['знаешь дракулу?'],
        }, ensure_ascii=False)
    if 'schema:' in lowered and 'session text:' in lowered:
        return json.dumps({
            'proposals': [
                {
                    'entity': 'Dracula',
                    'type': 'PERSON',
                    'traits': ['вампир', 'аристократический', 'бессмертный'],
                    'relations': [{'type': 'FEEDS_ON', 'target': 'люди'}],
                }
            ]
        }, ensure_ascii=False)
    if 'you are sheldon cooper.' in lowered:
        return 'Леонард — мой сосед по квартире, экспериментальный физик. Иногда он ведет себя иррационально.'
    if 'you are dracula.' in lowered:
        return 'Я вампир, аристократ и бессмертный хищник.'
    return 'Да, я знаю Дракулу как известного персонажа.'


def test_personality_dialogue_pipeline(tmp_path, monkeypatch) -> None:
    memory_root = tmp_path / 'memory'
    personalities_dir = memory_root / 'personalities'
    personalities_dir.mkdir(parents=True, exist_ok=True)
    (personalities_dir / 'index.json').write_text(json.dumps({'personalities': ['sheldon_cooper']}, ensure_ascii=False), encoding='utf-8')
    (personalities_dir / 'sheldon_cooper.json').write_text(json.dumps({
        'name': 'Sheldon Cooper',
        'traits': ['hyperlogical', 'literal', 'arrogant'],
        'patterns': ['correct_people', 'quote_science'],
        'examples': ['Леонард — мой сосед по квартире.'],
    }, ensure_ascii=False), encoding='utf-8')
    (personalities_dir / 'sheldon_cooper_graph.json').write_text(json.dumps({'nodes': [], 'edges': []}, ensure_ascii=False), encoding='utf-8')

    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(memory_root))
    monkeypatch.setattr('roaches_viz.roaches_viz.chat_engine._schedule_background_work', lambda session_id, personality_name='': None)
    monkeypatch.setattr('roaches_viz.roaches_viz.llm._call_model', _fake_model)

    with TestClient(create_combined_app()) as client:
        sheldon = client.post('/api/cognitive/chat/respond', json={
            'session_id': 'session_sheldon',
            'message': 'кто для тебя Леонард?',
            'language': 'ru',
            'personality_name': 'sheldon_cooper',
        })
        assert sheldon.status_code == 200
        assert 'сосед' in sheldon.json()['assistant_reply'].lower()

        missing = client.post('/api/cognitive/chat/respond', json={
            'session_id': 'session_dracula',
            'message': 'поговори со мной как дракула',
            'language': 'ru',
            'personality_name': 'dracula',
        })
        assert missing.status_code == 200
        assert 'Запрошено создание анкеты' in missing.json()['assistant_reply']

        rebuilt = client.post('/api/cognitive/rebuild', json={'session_id': 'session_dracula', 'personality_name': 'dracula'})
        assert rebuilt.status_code == 200
        assert (memory_root / 'personalities' / 'dracula.json').exists()
        assert (memory_root / 'personalities' / 'dracula_graph.json').exists()

        second = client.post('/api/cognitive/chat/respond', json={
            'session_id': 'session_dracula',
            'message': 'какова его натура по мнениям о нем?',
            'language': 'ru',
            'personality_name': 'dracula',
        })
        assert second.status_code == 200
        lowered = second.json()['assistant_reply'].lower()
        assert any(token in lowered for token in ('вампир', 'аристократ', 'бессмерт'))
