from __future__ import annotations

from pathlib import Path

from src.utils import local_llm_provider


def test_uncensored_autodiscovery_is_disabled_by_default(monkeypatch, tmp_path) -> None:
    general = tmp_path / "mistral-7b-instruct.gguf"
    uncensored = tmp_path / "Mistral-Nemo-2407-12B-Thinking-Claude-Gemini-GPT5.2-Uncensored-HERETIC_Q3_k_m.gguf"
    general.write_text("stub", encoding="utf-8")
    uncensored.write_text("stub", encoding="utf-8")

    monkeypatch.delenv("LOCAL_UNCENSORED_GGUF_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_ALLOW_UNCENSORED_AUTODISCOVERY", raising=False)
    monkeypatch.setattr(local_llm_provider, "_discover_gguf_paths", lambda: [general, uncensored])

    role_map = local_llm_provider._resolve_model_role_paths()

    assert role_map.get(local_llm_provider.ROLE_GENERAL)
    assert local_llm_provider.ROLE_UNCENSORED not in role_map


def test_uncensored_role_requires_explicit_opt_in_or_path(monkeypatch, tmp_path) -> None:
    uncensored = tmp_path / "Mistral-Nemo-2407-12B-Thinking-Claude-Gemini-GPT5.2-Uncensored-HERETIC_Q3_k_m.gguf"
    uncensored.write_text("stub", encoding="utf-8")

    monkeypatch.setenv("LOCAL_ALLOW_UNCENSORED_AUTODISCOVERY", "1")
    monkeypatch.delenv("LOCAL_UNCENSORED_GGUF_MODEL", raising=False)
    monkeypatch.setattr(local_llm_provider, "_discover_gguf_paths", lambda: [uncensored])

    role_map = local_llm_provider._resolve_model_role_paths()

    assert role_map.get(local_llm_provider.ROLE_UNCENSORED) == str(uncensored)


def test_prompt_prefers_json_only_for_structured_prompts() -> None:
    assert local_llm_provider._prompt_prefers_json_output('Return valid JSON only.\nSchema:\n{"entities":[]}')
    assert not local_llm_provider._prompt_prefers_json_output('Respond in Armenian.\nDo not output JSON.\nUser question: ...')


def test_raw_llm_fn_uses_plain_text_path_for_chat_prompts() -> None:
    calls: list[dict[str, object]] = []

    class FakeLlama:
        def create_chat_completion(self, **kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "plain text reply"}}]}

    llm_fn = local_llm_provider._make_raw_llm_fn(FakeLlama(), max_tokens=128)
    result = llm_fn('Respond in Armenian.\nDo not output JSON.\nUser question: Who are you?')

    assert result == 'plain text reply'
    assert calls
    assert 'response_format' not in calls[0]
