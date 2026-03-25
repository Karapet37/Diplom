from __future__ import annotations

from agent_system.llm import _mode_defaults, fallback_chat_reply, generate_chat_reply


def test_generate_chat_reply_prefers_fast_chat_role(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setenv('COGNITIVE_CHAT_ROLE', 'analyst')
    monkeypatch.delenv('AGENT_CHAT_ROLE', raising=False)

    def fake_model(prompt: str, mode: str = 'chat', *, role: str = 'general') -> str:
        calls.append(role)
        return 'Short grounded reply.'

    monkeypatch.setattr('agent_system.llm._call_model', fake_model)

    reply = generate_chat_reply('Grounded prompt.', language='en', persona_selected=False)

    assert reply == 'Short grounded reply.'
    assert calls == ['analyst']


def test_fallback_chat_reply_is_used_when_no_grounding() -> None:
    assert 'reliable context' in fallback_chat_reply(language='en', persona_selected=False)


def test_mode_defaults_use_fast_budgets() -> None:
    assert _mode_defaults('chat', role='analyst') == (1536, 640)
    assert _mode_defaults('knowledge', role='analyst') == (1536, 700)
    assert _mode_defaults('translation', role='translator') == (1024, 384)
