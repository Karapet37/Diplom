from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, '') or '').strip().lower()
    if not raw:
        return default
    return raw in {'1', 'true', 'yes', 'on'}


def _env_int(name: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = str(os.getenv(name, '') or '').strip()
    value = default
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _env_text(name: str, default: str) -> str:
    raw = str(os.getenv(name, '') or '').strip()
    return raw or default


def _env_float(name: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    raw = str(os.getenv(name, '') or '').strip()
    value = default
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _resolve_repo_path(raw: str, *, repo_root: Path, default: str | Path) -> Path:
    token = str(raw or '').strip()
    path = Path(token).expanduser() if token else Path(default)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _normalize_profile_name(raw: str) -> str:
    token = str(raw or '').strip().lower().replace('_', '-').replace(' ', '-')
    return token or 'development'


def _normalize_env_mapping(payload: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in payload.items():
        key_text = str(key or '').strip()
        if not key_text:
            continue
        if isinstance(value, bool):
            out[key_text] = '1' if value else '0'
        elif value is None:
            out[key_text] = ''
        else:
            out[key_text] = str(value)
    return out


def _read_yaml_file(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(payload, dict):
        raise ValueError(f'Expected a mapping in runtime config file: {path}')
    return payload


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            continue
        key, raw_value = stripped.split('=', 1)
        key = key.strip()
        value = raw_value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        values[key] = value
    return values


@dataclass(frozen=True, slots=True)
class RuntimeProfileInfo:
    name: str
    description: str
    path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'path': str(self.path),
        }


@dataclass(frozen=True, slots=True)
class RuntimeBootstrapReport:
    profile: str
    description: str
    profile_path: Path | None
    env_file: Path | None
    config_file: Path | None
    applied_defaults: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            'profile': self.profile,
            'description': self.description,
            'profile_path': str(self.profile_path) if self.profile_path else '',
            'env_file': str(self.env_file) if self.env_file else '',
            'config_file': str(self.config_file) if self.config_file else '',
            'applied_defaults': dict(self.applied_defaults),
            'warnings': list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    repo_root: Path
    config_dir: Path
    runtime_profiles_dir: Path
    runtime_dir: Path
    runtime_logs_dir: Path
    state_transitions_log_path: Path
    current_context_dir: Path
    current_context_json: Path
    current_context_txt: Path
    memory_root: Path
    working_dir: Path
    graphs_dir: Path
    heads_dir: Path
    proposals_dir: Path
    sessions_dir: Path
    uploaded_documents_dir: Path
    archive_dir: Path
    archive_sessions_dir: Path
    archive_heads_dir: Path
    archive_graphs_dir: Path
    archive_files_dir: Path
    mood_research_dir: Path
    mood_datasets_dir: Path
    mood_personas_dir: Path
    mood_sessions_dir: Path
    mood_reports_dir: Path
    webapp_dir: Path
    webapp_dist_dir: Path
    webapp_assets_dir: Path
    webapp_dist_index: Path
    webapp_fallback_index: Path


@dataclass(frozen=True, slots=True)
class ContextBudgetConfig:
    max_context_tokens: int
    question_tokens: int
    prompt_overhead_tokens: int
    section_budgets: dict[str, int] = field(default_factory=dict)
    section_minimums: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LlmRoleConfig:
    chat: str
    extraction: str
    persona_synthesis: str
    rethink: str
    translation: str
    use_general_for_persona_chat: bool


@dataclass(frozen=True, slots=True)
class LlmRetryConfig:
    retry_rounds: int
    fast_role_context_attempts: int
    general_context_attempts: int


@dataclass(frozen=True, slots=True)
class ChatOrchestrationConfig:
    strategy: str
    primary_role: str
    reviewer_role: str
    review_mode: str


@dataclass(frozen=True, slots=True)
class FeatureFlagConfig:
    enable_frontend_root: bool
    enable_frontend_assets: bool
    enable_background_rebuild: bool
    enable_concept_graph_premerge: bool
    include_side_effects_in_response: bool


@dataclass(frozen=True, slots=True)
class MemoryLifecycleConfig:
    session_archive_after_messages: int
    session_keep_recent_messages: int
    session_archive_chunk_size: int
    persona_trait_limit: int
    persona_relation_limit: int
    persona_example_limit: int
    persona_reaction_limit: int
    persona_log_tuple_limit: int
    persona_knowledge_char_limit: int
    archive_index_limit: int
    graph_snapshot_limit: int


@dataclass(frozen=True, slots=True)
class GraphLifecycleConfig:
    weak_importance_threshold: float
    weak_confidence_threshold: float
    suspect_confidence_threshold: float
    archive_importance_threshold: float
    archive_confidence_threshold: float
    extraction_quarantine_confidence: float
    clustering_min_nodes: int
    clustering_min_component_size: int


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    profile_name: str
    profile_description: str
    host: str
    port: int
    background_rebuild_interval: int
    graph_subgraph_limit: int
    rethink_context_budget: int
    rethink_context_budget_min: int
    rethink_context_budget_max: int


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    paths: RuntimePaths
    context: ContextBudgetConfig
    roles: LlmRoleConfig
    retries: LlmRetryConfig
    chat_orchestration: ChatOrchestrationConfig
    features: FeatureFlagConfig
    memory: MemoryLifecycleConfig
    graph: GraphLifecycleConfig
    settings: RuntimeSettings

    def llm_window(self, mode: str, *, role: str) -> tuple[int, int]:
        mode_key = str(mode or 'chat').strip().lower()
        role_key = str(role or self.roles.chat).strip().lower()
        role_env_prefix = {
            'general': 'LOCAL_GGUF',
            'analyst': 'LOCAL_ANALYST',
            'translator': 'LOCAL_TRANSLATOR',
            'creative': 'LOCAL_CREATIVE',
            'planner': 'LOCAL_PLANNER',
            'coder_architect': 'LOCAL_CODER_ARCHITECT',
            'coder_reviewer': 'LOCAL_CODER_REVIEWER',
            'coder_refactor': 'LOCAL_CODER_REFACTOR',
            'coder_debug': 'LOCAL_CODER_DEBUG',
            'uncensored': 'LOCAL_UNCENSORED_GGUF',
        }.get(role_key, 'LOCAL_GGUF')

        if mode_key == 'translation':
            default_n_ctx, default_max_tokens = 1024, 192
        elif mode_key == 'knowledge':
            default_n_ctx, default_max_tokens = ((2048, 384) if role_key == 'analyst' else (3072, 448))
        elif mode_key == 'chat':
            default_n_ctx, default_max_tokens = ((2048, 256) if role_key == 'analyst' else (3072, 320))
        else:
            default_n_ctx, default_max_tokens = 2048, 320

        resolved_n_ctx = _env_int(
            f'{role_env_prefix}_N_CTX',
            _env_int('LOCAL_GGUF_N_CTX', default_n_ctx, minimum=1024, maximum=7000),
            minimum=1024,
            maximum=7000,
        )
        resolved_max_tokens = _env_int(
            f'{role_env_prefix}_MAX_TOKENS',
            _env_int('LOCAL_GGUF_MAX_TOKENS', default_max_tokens, minimum=96, maximum=2048),
            minimum=96,
            maximum=2048,
        )
        return resolved_n_ctx, resolved_max_tokens

    def max_context_attempts_for_role(self, role: str) -> int:
        role_key = str(role or '').strip().lower()
        if role_key in {'analyst', 'translator'}:
            return self.retries.fast_role_context_attempts
        return self.retries.general_context_attempts

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['paths'] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in payload['paths'].items()
        }
        return payload


def list_runtime_profiles(repo_root: Path | None = None) -> tuple[RuntimeProfileInfo, ...]:
    root = repo_root or _repo_root()
    profiles_dir = root / 'config' / 'runtime-profiles'
    if not profiles_dir.exists():
        return ()
    rows: list[RuntimeProfileInfo] = []
    for path in sorted(profiles_dir.glob('*.yaml')):
        try:
            payload = _read_yaml_file(path)
        except Exception:
            payload = {}
        name = _normalize_profile_name(str(payload.get('profile') or path.stem))
        description = str(payload.get('description') or '').strip()
        rows.append(RuntimeProfileInfo(name=name, description=description, path=path))
    return tuple(rows)


def bootstrap_runtime_environment(
    *,
    profile: str = '',
    env_file: str = '',
    config_file: str = '',
) -> RuntimeBootstrapReport:
    repo_root = _repo_root()
    warnings: list[str] = []
    selected_profile = _normalize_profile_name(profile or os.getenv('COGNITIVE_RUNTIME_PROFILE', 'development'))

    profile_path: Path | None = None
    config_path: Path | None = None
    profile_payload: dict[str, Any] = {}
    if config_file:
        config_path = _resolve_repo_path(config_file, repo_root=repo_root, default=config_file)
        if config_path.exists():
            profile_payload = _read_yaml_file(config_path)
            profile_path = config_path
        else:
            warnings.append(f'Runtime config file was not found: {config_path}')
    else:
        profile_index = {item.name: item for item in list_runtime_profiles(repo_root)}
        profile_info = profile_index.get(selected_profile)
        if profile_info:
            profile_path = profile_info.path
            config_path = profile_info.path
            profile_payload = _read_yaml_file(profile_info.path)
        else:
            warnings.append(
                f"Runtime profile '{selected_profile}' was not found in {repo_root / 'config' / 'runtime-profiles'}. "
                'Using built-in defaults.'
            )

    description = str(profile_payload.get('description') or '').strip() or selected_profile
    profile_env = _normalize_env_mapping(profile_payload.get('env') or {})

    env_path: Path | None = None
    candidate_env = str(env_file or os.getenv('COGNITIVE_ENV_FILE', '') or '').strip()
    if candidate_env:
        env_path = _resolve_repo_path(candidate_env, repo_root=repo_root, default=candidate_env)
    else:
        local_env = repo_root / '.env.local'
        root_env = repo_root / '.env'
        if local_env.exists():
            env_path = local_env
        elif root_env.exists():
            env_path = root_env

    env_values: dict[str, str] = {}
    if env_path:
        if env_path.exists():
            env_values = _read_env_file(env_path)
        else:
            warnings.append(f'Environment file was not found: {env_path}')
            env_path = None

    merged_defaults = dict(profile_env)
    merged_defaults.update(env_values)
    merged_defaults.setdefault('COGNITIVE_RUNTIME_PROFILE', selected_profile)
    merged_defaults.setdefault('COGNITIVE_RUNTIME_PROFILE_DESCRIPTION', description)
    if env_path is not None:
        merged_defaults.setdefault('COGNITIVE_ENV_FILE', str(env_path))
    if config_path is not None:
        merged_defaults.setdefault('COGNITIVE_CONFIG_FILE', str(config_path))

    applied_defaults: dict[str, str] = {}
    for key, value in merged_defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            applied_defaults[key] = value

    return RuntimeBootstrapReport(
        profile=selected_profile,
        description=description,
        profile_path=profile_path,
        env_file=env_path,
        config_file=config_path,
        applied_defaults=applied_defaults,
        warnings=tuple(warnings),
    )


def get_runtime_config() -> RuntimeConfig:
    repo_root = _repo_root()
    config_dir = _resolve_repo_path(_env_text('COGNITIVE_CONFIG_DIR', 'config'), repo_root=repo_root, default='config')
    runtime_profiles_dir = _resolve_repo_path(
        _env_text('COGNITIVE_RUNTIME_PROFILES_DIR', 'config/runtime-profiles'),
        repo_root=repo_root,
        default='config/runtime-profiles',
    )
    runtime_dir = _resolve_repo_path(_env_text('COGNITIVE_RUNTIME_DIR', 'runtime'), repo_root=repo_root, default='runtime')
    memory_root = _resolve_repo_path(_env_text('COGNITIVE_MEMORY_ROOT', 'memory'), repo_root=repo_root, default='memory')
    webapp_dir = _resolve_repo_path(_env_text('COGNITIVE_WEBAPP_DIR', 'webapp'), repo_root=repo_root, default='webapp')
    webapp_dist_dir = _resolve_repo_path(
        _env_text('COGNITIVE_WEBAPP_DIST_DIR', str(Path('webapp') / 'dist')),
        repo_root=repo_root,
        default=Path('webapp') / 'dist',
    )

    paths = RuntimePaths(
        repo_root=repo_root,
        config_dir=config_dir,
        runtime_profiles_dir=runtime_profiles_dir,
        runtime_dir=runtime_dir,
        runtime_logs_dir=runtime_dir / 'logs',
        state_transitions_log_path=runtime_dir / 'logs' / 'state_transitions.jsonl',
        current_context_dir=runtime_dir / 'current_context',
        current_context_json=runtime_dir / 'current_context' / 'current_context.json',
        current_context_txt=runtime_dir / 'current_context' / 'current_context.txt',
        memory_root=memory_root,
        working_dir=memory_root / 'working',
        graphs_dir=memory_root / 'graphs',
        heads_dir=memory_root / 'heads',
        proposals_dir=memory_root / 'proposals',
        sessions_dir=memory_root / 'sessions',
        uploaded_documents_dir=memory_root / 'files' / 'uploaded_documents',
        archive_dir=memory_root / 'archive',
        archive_sessions_dir=memory_root / 'archive' / 'sessions',
        archive_heads_dir=memory_root / 'archive' / 'heads',
        archive_graphs_dir=memory_root / 'archive' / 'graphs',
        archive_files_dir=memory_root / 'archive' / 'files',
        mood_research_dir=memory_root / 'mood_research',
        mood_datasets_dir=memory_root / 'mood_research' / 'datasets',
        mood_personas_dir=memory_root / 'mood_research' / 'personas',
        mood_sessions_dir=memory_root / 'mood_research' / 'sessions',
        mood_reports_dir=memory_root / 'mood_research' / 'reports',
        webapp_dir=webapp_dir,
        webapp_dist_dir=webapp_dist_dir,
        webapp_assets_dir=webapp_dist_dir / 'assets',
        webapp_dist_index=webapp_dist_dir / 'index.html',
        webapp_fallback_index=webapp_dir / 'index.html',
    )
    context = ContextBudgetConfig(
        max_context_tokens=_env_int('COGNITIVE_MAX_CONTEXT_TOKENS', 4000, minimum=2048, maximum=6000),
        question_tokens=_env_int('COGNITIVE_QUESTION_TOKEN_BUDGET', 900, minimum=256, maximum=1800),
        prompt_overhead_tokens=_env_int('COGNITIVE_PROMPT_OVERHEAD_TOKENS', 180, minimum=64, maximum=512),
        section_budgets={
            'persona_block': _env_int('COGNITIVE_PERSONA_BLOCK_BUDGET', 1250, minimum=256, maximum=2200),
            'graph_context': _env_int('COGNITIVE_GRAPH_CONTEXT_BUDGET', 1750, minimum=256, maximum=2600),
            'recent_dialogue': _env_int('COGNITIVE_RECENT_DIALOGUE_BUDGET', 650, minimum=128, maximum=1400),
        },
        section_minimums={
            'persona_block': _env_int('COGNITIVE_PERSONA_BLOCK_MIN', 400, minimum=128, maximum=1200),
            'graph_context': _env_int('COGNITIVE_GRAPH_CONTEXT_MIN', 280, minimum=96, maximum=1000),
            'recent_dialogue': _env_int('COGNITIVE_RECENT_DIALOGUE_MIN', 120, minimum=64, maximum=480),
        },
    )
    roles = LlmRoleConfig(
        chat=_env_text('COGNITIVE_CHAT_ROLE', _env_text('AGENT_CHAT_ROLE', 'analyst')).lower(),
        extraction=_env_text('COGNITIVE_EXTRACTION_ROLE', 'analyst').lower(),
        persona_synthesis=_env_text('COGNITIVE_PERSONA_ROLE', 'analyst').lower(),
        rethink=_env_text('COGNITIVE_RETHINK_ROLE', 'analyst').lower(),
        translation=_env_text('COGNITIVE_TRANSLATION_ROLE', 'translator').lower(),
        use_general_for_persona_chat=_env_flag('COGNITIVE_CHAT_USE_GENERAL_FOR_PERSONA', False),
    )
    retries = LlmRetryConfig(
        retry_rounds=_env_int('LOCAL_LLM_RETRY_ROUNDS', 1, minimum=1, maximum=4),
        fast_role_context_attempts=_env_int('LOCAL_LLM_FAST_CONTEXT_ATTEMPTS', 1, minimum=1, maximum=3),
        general_context_attempts=_env_int('LOCAL_LLM_MAX_CONTEXT_ATTEMPTS', 2, minimum=1, maximum=4),
    )
    chat_orchestration = ChatOrchestrationConfig(
        strategy=_env_text('COGNITIVE_CHAT_ORCHESTRATION', 'primary_with_reviewer').lower(),
        primary_role=_env_text('COGNITIVE_CHAT_PRIMARY_ROLE', roles.chat).lower(),
        reviewer_role=_env_text('COGNITIVE_CHAT_REVIEW_ROLE', roles.rethink or 'analyst').lower(),
        review_mode=_env_text('COGNITIVE_CHAT_REVIEW_MODE', 'on_failure').lower(),
    )
    features = FeatureFlagConfig(
        enable_frontend_root=_env_flag('COGNITIVE_ENABLE_FRONTEND_ROOT', True),
        enable_frontend_assets=_env_flag('COGNITIVE_ENABLE_FRONTEND_ASSETS', True),
        enable_background_rebuild=_env_flag('COGNITIVE_ENABLE_BACKGROUND_REBUILD', True),
        enable_concept_graph_premerge=_env_flag('COGNITIVE_ENABLE_CONCEPT_GRAPH_PREMERGE', True),
        include_side_effects_in_response=_env_flag('COGNITIVE_INCLUDE_SIDE_EFFECTS', True),
    )
    memory = MemoryLifecycleConfig(
        session_archive_after_messages=_env_int('COGNITIVE_SESSION_ARCHIVE_AFTER_MESSAGES', 40, minimum=8, maximum=400),
        session_keep_recent_messages=_env_int('COGNITIVE_SESSION_KEEP_RECENT_MESSAGES', 16, minimum=4, maximum=120),
        session_archive_chunk_size=_env_int('COGNITIVE_SESSION_ARCHIVE_CHUNK_SIZE', 16, minimum=4, maximum=120),
        persona_trait_limit=_env_int('COGNITIVE_PERSONA_TRAIT_LIMIT', 20, minimum=4, maximum=64),
        persona_relation_limit=_env_int('COGNITIVE_PERSONA_RELATION_LIMIT', 12, minimum=1, maximum=64),
        persona_example_limit=_env_int('COGNITIVE_PERSONA_EXAMPLE_LIMIT', 24, minimum=4, maximum=128),
        persona_reaction_limit=_env_int('COGNITIVE_PERSONA_REACTION_LIMIT', 20, minimum=4, maximum=64),
        persona_log_tuple_limit=_env_int('COGNITIVE_PERSONA_LOG_TUPLE_LIMIT', 24, minimum=4, maximum=128),
        persona_knowledge_char_limit=_env_int('COGNITIVE_PERSONA_KNOWLEDGE_CHAR_LIMIT', 4000, minimum=64, maximum=12000),
        archive_index_limit=_env_int('COGNITIVE_ARCHIVE_INDEX_LIMIT', 48, minimum=8, maximum=256),
        graph_snapshot_limit=_env_int('COGNITIVE_GRAPH_SNAPSHOT_LIMIT', 16, minimum=4, maximum=128),
    )
    graph = GraphLifecycleConfig(
        weak_importance_threshold=_env_float('COGNITIVE_GRAPH_WEAK_IMPORTANCE', 0.18, minimum=0.01, maximum=1.0),
        weak_confidence_threshold=_env_float('COGNITIVE_GRAPH_WEAK_CONFIDENCE', 0.58, minimum=0.01, maximum=1.0),
        suspect_confidence_threshold=_env_float('COGNITIVE_GRAPH_SUSPECT_CONFIDENCE', 0.5, minimum=0.01, maximum=1.0),
        archive_importance_threshold=_env_float('COGNITIVE_GRAPH_ARCHIVE_IMPORTANCE', 0.05, minimum=0.0, maximum=1.0),
        archive_confidence_threshold=_env_float('COGNITIVE_GRAPH_ARCHIVE_CONFIDENCE', 0.42, minimum=0.0, maximum=1.0),
        extraction_quarantine_confidence=_env_float('COGNITIVE_GRAPH_EXTRACTION_QUARANTINE_CONFIDENCE', 0.55, minimum=0.01, maximum=1.0),
        clustering_min_nodes=_env_int('COGNITIVE_GRAPH_CLUSTERING_MIN_NODES', 24, minimum=8, maximum=256),
        clustering_min_component_size=_env_int('COGNITIVE_GRAPH_CLUSTERING_COMPONENT_MIN', 3, minimum=2, maximum=32),
    )
    settings = RuntimeSettings(
        profile_name=_normalize_profile_name(_env_text('COGNITIVE_RUNTIME_PROFILE', 'development')),
        profile_description=_env_text('COGNITIVE_RUNTIME_PROFILE_DESCRIPTION', 'development'),
        host=_env_text('WEB_HOST', '127.0.0.1'),
        port=_env_int('WEB_PORT', 8008, minimum=1, maximum=65535),
        background_rebuild_interval=_env_int('COGNITIVE_BACKGROUND_REBUILD_INTERVAL', 6, minimum=1, maximum=100),
        graph_subgraph_limit=_env_int('COGNITIVE_GRAPH_SUBGRAPH_LIMIT', 8, minimum=1, maximum=64),
        rethink_context_budget=_env_int('COGNITIVE_RETHINK_CONTEXT_BUDGET', 4000, minimum=1200, maximum=4096),
        rethink_context_budget_min=_env_int('COGNITIVE_RETHINK_CONTEXT_MIN', 1200, minimum=800, maximum=4096),
        rethink_context_budget_max=_env_int('COGNITIVE_RETHINK_CONTEXT_MAX', 4096, minimum=1200, maximum=8192),
    )
    for path in (
        paths.config_dir,
        paths.runtime_profiles_dir,
        paths.runtime_dir,
        paths.runtime_logs_dir,
        paths.current_context_dir,
        paths.memory_root,
        paths.working_dir,
        paths.graphs_dir,
        paths.heads_dir,
        paths.proposals_dir,
        paths.sessions_dir,
        paths.uploaded_documents_dir,
        paths.archive_dir,
        paths.archive_sessions_dir,
        paths.archive_heads_dir,
        paths.archive_graphs_dir,
        paths.archive_files_dir,
        paths.mood_research_dir,
        paths.mood_datasets_dir,
        paths.mood_personas_dir,
        paths.mood_sessions_dir,
        paths.mood_reports_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return RuntimeConfig(
        paths=paths,
        context=context,
        roles=roles,
        retries=retries,
        chat_orchestration=chat_orchestration,
        features=features,
        memory=memory,
        graph=graph,
        settings=settings,
    )
