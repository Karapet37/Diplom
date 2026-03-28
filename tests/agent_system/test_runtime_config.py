from __future__ import annotations

from agent_system.runtime_config import (
    bootstrap_runtime_environment,
    get_runtime_config,
    list_runtime_profiles,
)


def test_runtime_config_resolves_memory_paths_and_roles(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setenv('COGNITIVE_CHAT_ROLE', 'analyst')
    monkeypatch.setenv('COGNITIVE_EXTRACTION_ROLE', 'analyst')
    monkeypatch.setenv('COGNITIVE_RETHINK_ROLE', 'analyst')
    monkeypatch.setenv('COGNITIVE_TRANSLATION_ROLE', 'translator')
    monkeypatch.setenv('COGNITIVE_BACKGROUND_REBUILD_INTERVAL', '9')

    config = get_runtime_config()

    assert config.paths.memory_root == (tmp_path / 'memory').resolve()
    assert config.paths.working_dir.exists()
    assert config.paths.graphs_dir.exists()
    assert config.paths.heads_dir.exists()
    assert config.paths.archive_sessions_dir.exists()
    assert config.paths.archive_heads_dir.exists()
    assert config.paths.archive_graphs_dir.exists()
    assert config.roles.chat == 'analyst'
    assert config.roles.extraction == 'analyst'
    assert config.roles.rethink == 'analyst'
    assert config.roles.translation == 'translator'
    assert config.settings.background_rebuild_interval == 9
    assert config.context.max_context_tokens == 4000
    assert config.memory.session_archive_after_messages >= config.memory.session_keep_recent_messages
    assert config.graph.extraction_quarantine_confidence > 0
    assert config.graph.clustering_min_nodes >= config.graph.clustering_min_component_size


def test_runtime_config_resolves_relative_paths_against_repo_root(monkeypatch) -> None:
    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', 'runtime/test-profile/memory')
    monkeypatch.delenv('COGNITIVE_WEBAPP_DIR', raising=False)
    monkeypatch.delenv('COGNITIVE_WEBAPP_DIST_DIR', raising=False)

    config = get_runtime_config()

    assert config.paths.memory_root == (config.paths.repo_root / 'runtime' / 'test-profile' / 'memory').resolve()
    assert config.paths.webapp_dir == (config.paths.repo_root / 'webapp').resolve()
    assert config.paths.webapp_dist_dir == (config.paths.repo_root / 'webapp' / 'dist').resolve()


def test_runtime_config_llm_windows_and_context_attempts() -> None:
    config = get_runtime_config()

    assert config.llm_window('chat', role='analyst') == (1024, 80)
    assert config.llm_window('knowledge', role='analyst') == (1152, 288)
    assert config.llm_window('translation', role='translator') == (896, 160)
    assert config.max_context_attempts_for_role('analyst') >= 1
    assert config.max_context_attempts_for_role('general') >= 1


def test_list_runtime_profiles_contains_standard_profiles() -> None:
    profiles = {item.name: item for item in list_runtime_profiles()}

    assert {'development', 'local-demo', 'local-heavy', 'server'}.issubset(profiles)
    assert profiles['development'].path.exists()
    assert profiles['server'].description


def test_bootstrap_runtime_environment_layers_profile_and_env_file(tmp_path, monkeypatch) -> None:
    profile_path = tmp_path / 'custom-profile.yaml'
    profile_path.write_text(
        '\n'.join(
            [
                'profile: custom-demo',
                'description: Custom demo profile',
                'env:',
                '  WEB_HOST: 0.0.0.0',
                '  WEB_PORT: "9111"',
                '  COGNITIVE_MEMORY_ROOT: runtime/custom-profile/memory',
                '  COGNITIVE_GRAPH_SUBGRAPH_LIMIT: "14"',
            ]
        ),
        encoding='utf-8',
    )
    env_path = tmp_path / 'custom.env'
    env_path.write_text('WEB_PORT=9222\nCOGNITIVE_CHAT_ROLE=general\n', encoding='utf-8')

    monkeypatch.delenv('COGNITIVE_RUNTIME_PROFILE', raising=False)
    monkeypatch.delenv('COGNITIVE_RUNTIME_PROFILE_DESCRIPTION', raising=False)
    monkeypatch.delenv('COGNITIVE_MEMORY_ROOT', raising=False)
    monkeypatch.delenv('COGNITIVE_GRAPH_SUBGRAPH_LIMIT', raising=False)
    monkeypatch.delenv('COGNITIVE_CHAT_ROLE', raising=False)
    monkeypatch.delenv('WEB_HOST', raising=False)
    monkeypatch.delenv('WEB_PORT', raising=False)

    report = bootstrap_runtime_environment(
        profile='custom-demo',
        env_file=str(env_path),
        config_file=str(profile_path),
    )
    try:
        config = get_runtime_config()

        assert report.profile == 'custom-demo'
        assert report.env_file == env_path.resolve()
        assert report.config_file == profile_path.resolve()
        assert config.settings.profile_name == 'custom-demo'
        assert config.settings.profile_description == 'Custom demo profile'
        assert config.settings.host == '0.0.0.0'
        assert config.settings.port == 9222
        assert config.roles.chat == 'general'
        assert config.paths.memory_root == (config.paths.repo_root / 'runtime' / 'custom-profile' / 'memory').resolve()
        assert config.settings.graph_subgraph_limit == 14
    finally:
        for key in report.applied_defaults:
            monkeypatch.delenv(key, raising=False)
