from __future__ import annotations

from agent_system.runtime_config import get_runtime_config


def test_runtime_config_resolves_memory_paths_and_roles(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setenv('COGNITIVE_CHAT_ROLE', 'analyst')
    monkeypatch.setenv('COGNITIVE_EXTRACTION_ROLE', 'analyst')
    monkeypatch.setenv('COGNITIVE_RETHINK_ROLE', 'analyst')
    monkeypatch.setenv('COGNITIVE_TRANSLATION_ROLE', 'translator')
    monkeypatch.setenv('COGNITIVE_BACKGROUND_REBUILD_INTERVAL', '9')

    config = get_runtime_config()

    assert config.paths.memory_root == (tmp_path / 'memory').resolve()
    assert config.paths.graphs_dir.exists()
    assert config.paths.heads_dir.exists()
    assert config.roles.chat == 'analyst'
    assert config.roles.extraction == 'analyst'
    assert config.roles.rethink == 'analyst'
    assert config.roles.translation == 'translator'
    assert config.settings.background_rebuild_interval == 9
    assert config.context.max_context_tokens == 4000


def test_runtime_config_llm_windows_and_context_attempts() -> None:
    config = get_runtime_config()

    assert config.llm_window('chat', role='analyst') == (1536, 640)
    assert config.llm_window('translation', role='translator') == (1024, 384)
    assert config.max_context_attempts_for_role('analyst') >= 1
    assert config.max_context_attempts_for_role('general') >= 1
