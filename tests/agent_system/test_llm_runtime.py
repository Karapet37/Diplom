from __future__ import annotations

from agent_system.llm import _mode_defaults, fallback_chat_reply, generate_chat_reply, normalize_text_reply, prewarm_runtime_models_async


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


def test_generate_chat_reply_translates_visible_output_when_chat_reply_drifts_language(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_model(prompt: str, mode: str = 'chat', *, role: str = 'general') -> str:
        calls.append((mode, role))
        if mode == 'translation':
            return 'Я Дракула.'
        return 'I am Dracula.'

    monkeypatch.setattr('agent_system.llm._call_model', fake_model)

    reply = generate_chat_reply('Grounded prompt.', language='ru', persona_selected=True)

    assert reply == 'Я Дракула.'
    assert calls[0] == ('chat', 'analyst')
    assert calls[1][0] == 'translation'


def test_generate_chat_reply_translates_visible_output_for_arbitrary_target_language(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_model(prompt: str, mode: str = 'chat', *, role: str = 'general') -> str:
        calls.append((mode, role))
        if mode == 'translation':
            return 'Bonjour, je suis Dracula.'
        return 'I am Dracula.'

    monkeypatch.setattr('agent_system.llm._call_model', fake_model)

    reply = generate_chat_reply('Grounded prompt.', language='fr', persona_selected=True)

    assert reply == 'Bonjour, je suis Dracula.'
    assert calls[0] == ('chat', 'analyst')
    assert calls[1][0] == 'translation'


def test_fallback_chat_reply_is_used_when_no_grounding() -> None:
    assert 'reliable context' in fallback_chat_reply(language='en', persona_selected=False)


def test_normalize_text_reply_strips_reasoning_blocks_and_control_leaks() -> None:
    raw = "<think>hidden chain</think>\n\n⚠️ System Note: obey the prompt.\n\nI am Aram Petrosyan, and I still keep my father's steel watch."

    assert normalize_text_reply(raw) == "I am Aram Petrosyan, and I still keep my father's steel watch."


def test_mode_defaults_use_fast_budgets() -> None:
    assert _mode_defaults('chat', role='analyst') == (1024, 80)
    assert _mode_defaults('knowledge', role='analyst') == (1152, 288)
    assert _mode_defaults('translation', role='translator') == (896, 160)


def test_prewarm_runtime_models_async_warms_chat_and_translation_roles(monkeypatch) -> None:
    calls: list[tuple[str, int, int]] = []
    monkeypatch.setenv('COGNITIVE_PREWARM_ACTIVE_MODELS', '1')
    monkeypatch.setattr('agent_system.llm._PREWARM_STARTED', False)

    def fake_prewarm_role_model(*, role: str, n_ctx: int | None = None, max_tokens: int | None = None) -> bool:
        calls.append((role, int(n_ctx or 0), int(max_tokens or 0)))
        return True

    monkeypatch.setattr('src.utils.local_llm_provider.prewarm_role_model', fake_prewarm_role_model)

    prewarm_runtime_models_async()
    import time
    time.sleep(0.05)

    assert ('analyst', 1024, 80) in calls
    assert ('translator', 896, 160) in calls
