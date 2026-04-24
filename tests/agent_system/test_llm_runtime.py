from __future__ import annotations

from agent_system.llm import (
    _mode_defaults,
    fallback_chat_reply,
    generate_chat_reply,
    normalize_text_reply,
    plan_model_budget,
    prewarm_runtime_models_async,
)
import src.utils.local_llm_provider as local_llm_provider


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


def test_generate_chat_reply_respects_role_override(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setenv('COGNITIVE_CHAT_ROLE', 'general')

    def fake_model(prompt: str, mode: str = 'chat', *, role: str = 'general') -> str:
        calls.append(role)
        return 'Short grounded reply.'

    monkeypatch.setattr('agent_system.llm._call_model', fake_model)

    reply = generate_chat_reply(
        'Grounded prompt.',
        language='en',
        persona_selected=False,
        role_override='analyst',
    )

    assert reply == 'Short grounded reply.'
    assert calls == ['analyst']


def test_generate_chat_reply_translates_visible_output_when_chat_reply_drifts_language(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setenv('COGNITIVE_CHAT_ROLE', 'analyst')
    monkeypatch.setenv('COGNITIVE_CHAT_USE_GENERAL_FOR_PERSONA', '0')

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
    monkeypatch.setenv('COGNITIVE_CHAT_ROLE', 'analyst')
    monkeypatch.setenv('COGNITIVE_CHAT_USE_GENERAL_FOR_PERSONA', '0')

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
    assert fallback_chat_reply(language='en', persona_selected=False) == "I'm here."
    assert fallback_chat_reply(language='en', persona_selected=True) == 'Go ahead.'


def test_normalize_text_reply_strips_reasoning_blocks_and_control_leaks() -> None:
    raw = "<think>hidden chain</think>\n\n⚠️ System Note: obey the prompt.\n\nI am Aram Petrosyan, and I still keep my father's steel watch."

    assert normalize_text_reply(raw) == "I am Aram Petrosyan, and I still keep my father's steel watch."


def test_normalize_text_reply_extracts_visible_reply_from_analysis_scaffold() -> None:
    raw = """
1. **Analyze the Request:**
* **Role:** Final Generator stage.
* **Safety/Policy:** The user is making a sexually explicit/NSFW statement.

# Ответ
**Внешний ответ персонажа:** Не занята. Но корону тебе за такой заход не выдам.
(мысленно: ну хоть не скучно)
"""

    assert normalize_text_reply(raw) == "Не занята. Но корону тебе за такой заход не выдам."


def test_normalize_text_reply_extracts_answer_before_review_notes() -> None:
    raw = """
# Answer
Բարև։ Առանց նավի էլ ես մնում եմ Ջեք, պարզապես ծովի հոտը քիչ է։

---
**Review Notes:**
- The draft is in Armenian.
**Issues Identified:**
- Persona inconsistency.
"""

    assert normalize_text_reply(raw) == "Բարև։ Առանց նավի էլ ես մնում եմ Ջեք, պարզապես ծովի հոտը քիչ է։"


def test_mode_defaults_use_reserved_output_budgets(monkeypatch) -> None:
    monkeypatch.delenv('LOCAL_GGUF_N_CTX', raising=False)
    monkeypatch.delenv('LOCAL_GGUF_MAX_TOKENS', raising=False)
    monkeypatch.delenv('LOCAL_ANALYST_N_CTX', raising=False)
    monkeypatch.delenv('LOCAL_ANALYST_MAX_TOKENS', raising=False)
    monkeypatch.delenv('LOCAL_TRANSLATOR_N_CTX', raising=False)
    monkeypatch.delenv('LOCAL_TRANSLATOR_MAX_TOKENS', raising=False)

    assert _mode_defaults('chat', role='analyst') == (2048, 256)
    assert _mode_defaults('knowledge', role='analyst') == (2048, 384)
    assert _mode_defaults('translation', role='translator') == (1024, 192)


def test_plan_model_budget_auto_expands_n_ctx_when_needed(monkeypatch) -> None:
    monkeypatch.delenv('LOCAL_GGUF_N_CTX', raising=False)
    monkeypatch.delenv('LOCAL_GGUF_MAX_TOKENS', raising=False)
    monkeypatch.delenv('LOCAL_ANALYST_N_CTX', raising=False)
    monkeypatch.delenv('LOCAL_ANALYST_MAX_TOKENS', raising=False)

    prompt = 'storm heading ' * 1400
    budget = plan_model_budget(prompt, mode='chat', role='analyst')

    assert budget['configured_n_ctx'] == 2048
    assert budget['n_ctx'] > budget['configured_n_ctx']
    assert budget['reserved_output_budget'] >= 256
    assert budget['actual_max_tokens'] >= 256
    assert budget['n_ctx_auto_expanded'] is True


def test_plan_model_budget_reports_when_prompt_exceeds_window_budget(monkeypatch) -> None:
    monkeypatch.delenv('LOCAL_GGUF_N_CTX', raising=False)
    monkeypatch.delenv('LOCAL_GGUF_MAX_TOKENS', raising=False)
    monkeypatch.delenv('LOCAL_ANALYST_N_CTX', raising=False)
    monkeypatch.delenv('LOCAL_ANALYST_MAX_TOKENS', raising=False)

    prompt = 'abcd ' * 5000
    budget = plan_model_budget(prompt, mode='chat', role='analyst')

    assert budget['n_ctx'] == 7000
    assert budget['prompt_exceeds_window_budget'] is True
    assert budget['model_context_exhausted'] is True
    assert budget['window_shortfall_tokens'] > 0


def test_prewarm_runtime_models_async_warms_chat_and_translation_roles(monkeypatch) -> None:
    calls: list[tuple[str, int, int]] = []
    monkeypatch.setenv('COGNITIVE_PREWARM_ACTIVE_MODELS', '1')
    monkeypatch.setenv('COGNITIVE_CHAT_ROLE', 'analyst')
    monkeypatch.setenv('COGNITIVE_CHAT_USE_GENERAL_FOR_PERSONA', '0')
    monkeypatch.setattr('agent_system.llm._PREWARM_STARTED', False)

    def fake_prewarm_role_model(*, role: str, n_ctx: int | None = None, max_tokens: int | None = None) -> bool:
        calls.append((role, int(n_ctx or 0), int(max_tokens or 0)))
        return True

    monkeypatch.setattr('src.utils.local_llm_provider.prewarm_role_model', fake_prewarm_role_model)

    prewarm_runtime_models_async()
    import time
    time.sleep(0.05)

    assert ('analyst', 2048, 256) in calls
    assert ('translator', 1024, 192) in calls


def test_local_llm_provider_falls_back_when_general_model_is_incompatible(monkeypatch) -> None:
    monkeypatch.setenv('LOCAL_GGUF_MODEL', 'models/Qwen3.5-2B.Q4_K_M.gguf')
    monkeypatch.setenv('LOCAL_ANALYST_GGUF_MODEL', 'models/Nanbeige4.1-3B.Q3_K_M.gguf')
    monkeypatch.setattr(local_llm_provider, 'Llama', object())
    monkeypatch.setattr(
        local_llm_provider,
        '_preflight_model_block_reason',
        lambda path: 'qwen35 unsupported' if 'Qwen3.5-2B' in str(path) else '',
    )

    built_paths: list[str] = []

    def fake_make_budgeted_llm_fn(*, role: str, model_path: str, n_ctx: int, max_tokens: int):  # type: ignore[no-untyped-def]
        built_paths.append(model_path)
        def _fn(prompt: str) -> str:
            return 'ok'
        setattr(_fn, '_model_path', model_path)
        return _fn

    local_llm_provider._ROLE_LLM_FN.clear()
    local_llm_provider._ROLE_MODEL_MAP.clear()
    monkeypatch.setattr(local_llm_provider, '_make_budgeted_llm_fn', fake_make_budgeted_llm_fn)

    fn = local_llm_provider.build_role_llm_fn('general')

    assert fn is not None
    assert built_paths
    assert built_paths[0].endswith('models/Nanbeige4.1-3B.Q3_K_M.gguf')


def test_local_llm_provider_detects_known_incompatible_qwen35_architecture(monkeypatch) -> None:
    monkeypatch.setattr(local_llm_provider, '_read_gguf_architecture', lambda path: 'qwen35')
    monkeypatch.setattr(local_llm_provider, '_llama_cpp_version', lambda: '0.3.16')
    monkeypatch.setattr(local_llm_provider, '_llama_cpp_version_tuple', lambda: (0, 3, 16))

    reason = local_llm_provider._preflight_model_block_reason('models/Qwen3.5-2B.Q4_K_M.gguf')

    assert 'qwen35' in reason
    assert '0.3.16' in reason
