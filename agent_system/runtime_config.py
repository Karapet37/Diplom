from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


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


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    repo_root: Path
    memory_root: Path
    graphs_dir: Path
    heads_dir: Path
    proposals_dir: Path
    sessions_dir: Path
    uploaded_documents_dir: Path
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
class FeatureFlagConfig:
    enable_frontend_root: bool
    enable_frontend_assets: bool
    enable_background_rebuild: bool
    enable_concept_graph_premerge: bool
    include_side_effects_in_response: bool


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
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
    features: FeatureFlagConfig
    settings: RuntimeSettings

    def llm_window(self, mode: str, *, role: str) -> tuple[int, int]:
        mode_key = str(mode or 'chat').strip().lower()
        role_key = str(role or self.roles.chat).strip().lower()
        if mode_key == 'translation':
            return 1024, 384
        if mode_key == 'knowledge':
            return (1536, 700) if role_key == 'analyst' else (2048, 800)
        if mode_key == 'chat':
            return (1536, 640) if role_key == 'analyst' else (2048, 768)
        return 1536, 700

    def max_context_attempts_for_role(self, role: str) -> int:
        role_key = str(role or '').strip().lower()
        if role_key in {'analyst', 'translator'}:
            return self.retries.fast_role_context_attempts
        return self.retries.general_context_attempts


def get_runtime_config() -> RuntimeConfig:
    repo_root = Path(__file__).resolve().parents[1]
    env_memory_root = str(os.getenv('COGNITIVE_MEMORY_ROOT', '') or '').strip()
    memory_root = Path(env_memory_root).expanduser().resolve() if env_memory_root else repo_root / 'memory'
    paths = RuntimePaths(
        repo_root=repo_root,
        memory_root=memory_root,
        graphs_dir=memory_root / 'graphs',
        heads_dir=memory_root / 'heads',
        proposals_dir=memory_root / 'proposals',
        sessions_dir=memory_root / 'sessions',
        uploaded_documents_dir=memory_root / 'files' / 'uploaded_documents',
        webapp_dir=repo_root / 'webapp',
        webapp_dist_dir=repo_root / 'webapp' / 'dist',
        webapp_assets_dir=repo_root / 'webapp' / 'dist' / 'assets',
        webapp_dist_index=repo_root / 'webapp' / 'dist' / 'index.html',
        webapp_fallback_index=repo_root / 'webapp' / 'index.html',
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
    features = FeatureFlagConfig(
        enable_frontend_root=_env_flag('COGNITIVE_ENABLE_FRONTEND_ROOT', True),
        enable_frontend_assets=_env_flag('COGNITIVE_ENABLE_FRONTEND_ASSETS', True),
        enable_background_rebuild=_env_flag('COGNITIVE_ENABLE_BACKGROUND_REBUILD', True),
        enable_concept_graph_premerge=_env_flag('COGNITIVE_ENABLE_CONCEPT_GRAPH_PREMERGE', True),
        include_side_effects_in_response=_env_flag('COGNITIVE_INCLUDE_SIDE_EFFECTS', True),
    )
    settings = RuntimeSettings(
        host=_env_text('WEB_HOST', '127.0.0.1'),
        port=_env_int('WEB_PORT', 8008, minimum=1, maximum=65535),
        background_rebuild_interval=_env_int('COGNITIVE_BACKGROUND_REBUILD_INTERVAL', 6, minimum=1, maximum=100),
        graph_subgraph_limit=_env_int('COGNITIVE_GRAPH_SUBGRAPH_LIMIT', 8, minimum=1, maximum=64),
        rethink_context_budget=_env_int('COGNITIVE_RETHINK_CONTEXT_BUDGET', 4000, minimum=1200, maximum=4096),
        rethink_context_budget_min=_env_int('COGNITIVE_RETHINK_CONTEXT_MIN', 1200, minimum=800, maximum=4096),
        rethink_context_budget_max=_env_int('COGNITIVE_RETHINK_CONTEXT_MAX', 4096, minimum=1200, maximum=8192),
    )
    for path in (
        paths.memory_root,
        paths.graphs_dir,
        paths.heads_dir,
        paths.proposals_dir,
        paths.sessions_dir,
        paths.uploaded_documents_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return RuntimeConfig(
        paths=paths,
        context=context,
        roles=roles,
        retries=retries,
        features=features,
        settings=settings,
    )
