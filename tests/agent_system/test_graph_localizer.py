from __future__ import annotations

from agent_system.graph_localizer import localized_node_view
from agent_system.graph_store import GraphStore


def test_localized_node_view_keeps_canonical_english_and_adds_target_translation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    store = GraphStore()
    node = store.create_node(
        name='солнечный свет',
        node_type='CONCEPT',
        description='Солнечный свет — это поток энергии, делающий жизнь возможной.',
        translation_line='солнечный свет: sunlight, արևի լույս',
    )
    before = store.get_node_by_id(node['id'])
    before_frequency = int(before.get('frequency') or 0)

    def fake_model(prompt: str, mode: str = 'chat', role: str = 'general') -> str:
        if mode != 'translation':
            return ''
        if 'Target language: English.' in prompt:
            return 'Sunlight is a stream of energy that helps make life possible.'
        if 'Target language: Armenian.' in prompt:
            return 'Արեւի լույսը էներգիայի հոսք է, որը օգնում է հնարավոր դարձնել կյանքը։'
        return ''

    monkeypatch.setattr('agent_system.llm._call_model', fake_model)

    view = localized_node_view(node['id'], language='hy', store=store)

    assert view is not None
    context = view['what_is_it_like']['context']
    assert context['canonical_english_explanation'] == 'Sunlight is a stream of energy that helps make life possible.'
    assert context['localized_explanation'] == 'Արեւի լույսը էներգիայի հոսք է, որը օգնում է հնարավոր դարձնել կյանքը։'

    after = store.get_node_by_id(node['id'])
    assert int(after.get('frequency') or 0) == before_frequency
    assert after.get('context', {}).get('canonical_english_explanation') == context['canonical_english_explanation']
