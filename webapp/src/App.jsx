import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  applyProjectArchiveReview,
  clearGraph,
  checkProjectHallucination,
  createEdge,
  createNode,
  deleteEdge,
  deleteNode,
  getEvents,
  getModules,
  getNodeTypes,
  getProjectDbSchema,
  getProjectModelAdvisors,
  getProfilePrompt,
  getSnapshot,
  inferProfileGraph,
  importProjectAutoruns,
  introspectClient,
  loadGraph,
  persistGraph,
  reinforceRelation,
  reportProjectHallucination,
  rewardEvent,
  runProjectLLMDebate,
  runProjectGraphRagQuery,
  runProjectTaskRiskBoard,
  runProjectTimelineReplay,
  runProjectPersonalTreeIngest,
  runProjectArchiveChat,
  runProjectDailyMode,
  runProjectQualityHarness,
  saveProjectPersonalTreeNote,
  scanProjectContradictions,
  manageProjectPackages,
  applyProjectMemoryNamespace,
  viewProjectMemoryNamespace,
  getProjectLLMPolicy,
  updateProjectLLMPolicy,
  createProjectBackup,
  restoreProjectBackup,
  getProjectAuditLogs,
  simulateGraph,
  subscribeGraphEvents,
  updateEdge,
  updateNode,
  updateProjectUserGraph,
  viewProjectPersonalTree,
  watchProjectDemo,
} from "./api";

const RELATION_OPTIONS = [
  "works_at",
  "owns",
  "influences",
  "competes_with",
  "parent_of",
  "part_of",
];

const ACTIVATION_OPTIONS = ["tanh", "identity", "relu", "sigmoid"];
const HALLUCINATION_SEVERITY_OPTIONS = ["low", "medium", "high", "critical"];
const TASK_PRIORITY_OPTIONS = ["low", "medium", "high", "critical"];
const TASK_STATUS_OPTIONS = ["backlog", "active", "blocked", "done"];
const RISK_PROBABILITY_OPTIONS = ["very_low", "low", "medium", "high", "very_high"];
const RISK_IMPACT_OPTIONS = ["low", "medium", "high", "critical"];
const REQUEST_STATUS_OPTIONS = ["backlog", "active", "review", "done"];
const MULTITOOL_DOMAIN_OPTIONS = [
  "general",
  "psychology",
  "jurisprudence",
  "legislation",
  "education",
  "career",
  "finance",
  "health",
  "operations",
];
const LEGISLATION_COUNTRY_OPTIONS = [
  "global",
  "armenia",
  "russia",
  "usa",
  "eu",
  "india",
  "china",
  "japan",
  "uae",
];
const LLM_ROLE_OPTIONS = [
  "creative",
  "analyst",
  "planner",
  "general",
  "coder_architect",
  "coder_reviewer",
  "coder_refactor",
  "coder_debug",
];
const MEMORY_NAMESPACE_OPTIONS = ["global", "personal", "session", "experiment", "trash"];
const MEMORY_SCOPE_OPTIONS = ["all", "owned"];
const LLM_POLICY_MODE_OPTIONS = ["propose_only", "confirm_required", "assisted_autonomy"];
const PAGE_KEYS = ["overview", "builder", "simulation", "data"];
const OVERVIEW_SECTION_KEYS = [
  "demo",
  "daily",
  "user_graph",
  "autoruns",
  "multitool",
  "graph",
  "client",
  "advisors",
  "hallucination_hunter",
];

const UI_LANG_OPTIONS = [
  { code: "hy", label: "Հայերեն" },
  { code: "ru", label: "Русский" },
  { code: "en", label: "English" },
  { code: "fr", label: "Français" },
  { code: "es", label: "Español" },
  { code: "pt", label: "Português" },
  { code: "ar", label: "العربية" },
  { code: "hi", label: "हिन्दी" },
  { code: "zh", label: "中文" },
  { code: "ja", label: "日本語" },
];

const PERSONALIZATION_STORAGE_KEY = "workspace_personalization_profile_v1";
const STYLE_NODE_STORAGE_KEY = "workspace_style_node_index_v1";

const PERSONALIZATION_STYLE_OPTIONS = ["adaptive", "concise", "balanced", "deep"];
const PERSONALIZATION_DEPTH_OPTIONS = ["quick", "balanced", "deep"];
const PERSONALIZATION_RISK_OPTIONS = ["low", "medium", "high"];
const PERSONALIZATION_TONE_OPTIONS = ["neutral", "direct", "empathetic", "challenging"];
const BRANCH_INSIGHT_VIEW_OPTIONS = ["cards", "charts", "lists", "tips"];

const STYLE_NODE_PRESETS = [
  {
    id: "base_control",
    name: "Base Control",
    description: "Default balanced workspace look.",
    vars: {
      "--bg": "#f4f7fb",
      "--card": "#ffffff",
      "--line": "#c9d8e9",
      "--text": "#0f2334",
      "--muted": "#4f6274",
      "--accent": "#0f7a66",
      "--danger": "#8d2f2f",
      "--accent-soft": "#eaf7f3",
    },
  },
  {
    id: "signal_focus",
    name: "Signal Focus",
    description: "Higher contrast for analysis-heavy sessions.",
    vars: {
      "--bg": "#e8f0f9",
      "--card": "#ffffff",
      "--line": "#8fb2d3",
      "--text": "#0b1f33",
      "--muted": "#334e66",
      "--accent": "#125da5",
      "--danger": "#9c2d1d",
      "--accent-soft": "#eaf3ff",
    },
  },
  {
    id: "strategy_board",
    name: "Strategy Board",
    description: "Warmer planning style with stronger hierarchy.",
    vars: {
      "--bg": "#f6f1e8",
      "--card": "#fffdf9",
      "--line": "#d8bfa0",
      "--text": "#2f2217",
      "--muted": "#70533b",
      "--accent": "#8f4f17",
      "--danger": "#9a2f2f",
      "--accent-soft": "#fff2df",
    },
  },
  {
    id: "calm_matrix",
    name: "Calm Matrix",
    description: "Soft neutral palette for long reading sessions.",
    vars: {
      "--bg": "#edf2ef",
      "--card": "#fbfefc",
      "--line": "#b9cbc1",
      "--text": "#1f332b",
      "--muted": "#4b6158",
      "--accent": "#276c5a",
      "--danger": "#8f3a3a",
      "--accent-soft": "#eaf7f2",
    },
  },
];

const GRAPH_NODE_TYPE_UI_STYLE = "ui_style_node";
const GRAPH_NODE_TYPE_BRANCH_REPORT = "branch_report_node";
const GRAPH_NODE_TYPE_MULTITOOL_REQUEST = "multitool_request_node";
const GRAPH_NODE_TYPE_PREFERENCE_PROFILE = "preference_profile_node";
const GRAPH_NODE_TYPE_TASK_ITEM = "task_item_node";
const GRAPH_NODE_TYPE_RISK_ITEM = "risk_item_node";
const GRAPH_NODE_TYPE_DOMAIN_BRANCH = "domain_branch_node";
const GRAPH_NODE_TYPE_LEGISLATION_COUNTRY = "country_law_node";
const GRAPH_NODE_TYPE_PERSONAL_TREE_ROOT = "personal_info_tree_root";
const GRAPH_NODE_TYPE_THOUGHT_SESSION = "thought_tree_session";
const GRAPH_NODE_TYPE_THOUGHT_SUMMARY = "thought_summary_node";
const GRAPH_NODE_TYPE_THOUGHT_POINT = "thought_point_node";
const GRAPH_NODE_TYPE_PERSONAL_NOTE = "personal_note_node";
const GRAPH_NODE_TYPE_SOURCE_REFERENCE = "source_reference";
const STYLE_VAR_ALLOWLIST = Object.freeze([
  "--bg",
  "--card",
  "--line",
  "--text",
  "--muted",
  "--accent",
  "--danger",
  "--accent-soft",
]);

const DEFAULT_PERSONALIZATION_DRAFT = {
  response_style: "adaptive",
  reasoning_depth: "balanced",
  risk_tolerance: "medium",
  tone: "neutral",
  focus_goals_text: "",
  domain_focus_text: "",
  avoid_topics_text: "",
  memory_notes: "",
  role_proposer: "creative",
  role_critic: "analyst",
  role_judge: "planner",
  auto_apply_user_graph: true,
  auto_apply_daily: true,
  auto_apply_debate: true,
};

const DEFAULT_MULTITOOL_REQUEST_DRAFT = {
  title: "",
  details: "",
  desired_output: "",
  layout_mode: "graph",
  status: "backlog",
  priority: "medium",
  domain: "general",
  country: "global",
};

const DEFAULT_MULTITOOL_PREFERENCE_DRAFT = {
  profile_name: "default",
  likes_text: "",
  dislikes_text: "",
  style_examples_text: "",
  tool_examples_text: "",
  notes: "",
  domain: "general",
  country: "global",
};

const DEFAULT_MULTITOOL_TASK_DRAFT = {
  title: "",
  description: "",
  status: "backlog",
  priority: "medium",
  due_at: "",
  domain: "general",
  country: "global",
};

const DEFAULT_MULTITOOL_RISK_DRAFT = {
  title: "",
  description: "",
  probability: "medium",
  impact: "medium",
  mitigation_text: "",
  domain: "general",
  country: "global",
};

const DEFAULT_PERSONAL_TREE_INGEST_DRAFT = {
  title: "",
  topic: "",
  text: "",
  source_type: "text",
  source_url: "",
  source_title: "",
  max_points: 6,
};

const DEFAULT_PERSONAL_TREE_NOTE_DRAFT = {
  title: "",
  note: "",
  tags_text: "",
  links_text: "",
  source_type: "note",
  source_url: "",
  source_title: "",
};

const DEFAULT_PACKAGES_DRAFT = {
  package_name: "inbox",
  items_text: "",
  restore_ids_text: "",
  model_role: "coder_reviewer",
  model_path: "",
  classify_with_llm: true,
  apply_changes: true,
  confirmation: "confirm",
};

const DEFAULT_MEMORY_NAMESPACE_DRAFT = {
  namespace: "personal",
  source_namespace: "",
  scope: "owned",
  query: "",
  node_ids_text: "",
  min_score: "0.2",
  apply_changes: true,
  confirmation: "confirm",
};

const DEFAULT_GRAPH_RAG_DRAFT = {
  query: "",
  top_k: 6,
  scope: "owned",
  namespace: "",
  use_llm: true,
  model_role: "analyst",
  model_path: "",
};

const DEFAULT_CONTRADICTION_SCAN_DRAFT = {
  scope: "owned",
  namespace: "",
  max_nodes: 120,
  top_k: 20,
  min_overlap: "0.32",
  apply_to_graph: true,
  confirmation: "confirm",
};

const DEFAULT_TASK_RISK_DRAFT = {
  tasks_text: "",
  apply_to_graph: true,
  confirmation: "confirm",
};

const DEFAULT_TIMELINE_REPLAY_DRAFT = {
  event_type: "",
  limit: 300,
  from_ts: "",
  to_ts: "",
};

const DEFAULT_LLM_POLICY_DRAFT = {
  mode: "confirm_required",
  trusted_sessions_text: "",
  trusted_users_text: "",
  allow_actions_text: "",
  merge_lists: true,
};

const DEFAULT_QUALITY_DRAFT = {
  sample_queries_text: "",
};

const DEFAULT_BACKUP_DRAFT = {
  label: "manual",
  include_events: true,
  event_limit: 1000,
  path: "",
  latest: true,
  apply_changes: true,
  confirmation: "confirm",
  restore_policy: true,
};

const DEFAULT_AUDIT_DRAFT = {
  limit: 200,
  include_backups: true,
};

const TRANSLATIONS = {
  en: {
    app_title: "Autonomous Graph Workspace",
    app_subtitle: "Graph-first workspace: build nodes, infer relations, simulate and inspect.",
    ui_language: "UI Language",
    action_refresh: "Refresh",
    action_seed_demo: "Watch Demo",
    action_clear: "Clear",
    runtime_error_title: "Frontend Runtime Error",
    runtime_error_hint: "Open browser devtools console and fix the error below.",
    action_try_continue: "Try Continue",
    action_reload_data: "Reload Data",
    page_overview: "Overview",
    page_builder: "Builder",
    page_simulation: "Simulation",
    page_data: "Data",
    llm_profile_builder: "LLM Profile Builder",
    profile_input_text: "Narrative Text",
    entity_type_hint: "Entity Type Hint",
    entity_type_human: "Human",
    entity_type_company: "Company",
    entity_type_technology: "Technology",
    entity_type_generic: "Generic",
    extract_profile_graph: "Extract Profile -> Build Graph",
    profile_result: "LLM Profile Result",
    profile_prompt_preview: "Prompt Template",
    model_missing_hint: "Put a GGUF model into ./models and retry.",
    workspace_status: "Workspace Status",
    execution_log: "Execution Log",
    graph_visualization: "Graph Visualization",
    graph_empty: "Graph is empty. Add nodes and relations, then run simulation.",
    create_node: "Create Node",
    node_type: "Node Type",
    first_name: "First Name",
    last_name: "Last Name",
    company_name: "Company Name",
    company_industry: "Industry",
    company_description: "Description",
    bio: "Bio",
    profile_text: "Profile Text",
    profile_placeholder:
      "preferences: jazz, history films\\nvalues: freedom, business\\nfears: market crash\\ngoals: launch product",
    employment_text: "Employment Text",
    employment_placeholder: "founder @ Vector Dynamics; engineer at North Capital",
    employment_json: "Employment JSON",
    attributes_json: "Attributes JSON",
    state_json: "State JSON",
    create_node_btn: "Create Node",
    create_edge: "Create Edge",
    from_node: "From Node",
    to_node: "To Node",
    relation: "Relation",
    weight: "Weight",
    direction: "Direction",
    logic_rule: "Logic Rule",
    directed: "directed",
    undirected: "undirected",
    create_edge_btn: "Create Edge",
    persistence: "Persistence",
    persist_snapshot: "Persist Snapshot",
    load_snapshot: "Load Snapshot",
    simulation: "Simulation",
    seed_ids: "Seed IDs",
    depth: "Depth",
    steps: "Steps",
    damping: "Damping",
    activation: "Activation",
    infer_rounds: "Infer Rounds",
    run_simulation: "Run Simulation",
    last_simulation_output: "Last Simulation Output",
    event_feedback: "Event Feedback",
    event_id: "Event ID",
    reward: "Reward",
    learning_rate: "Learning Rate",
    apply_reward: "Apply Reward to Event",
    batch_relation_reinforcement: "Batch Relation Reinforcement",
    reinforce_relation: "Reinforce Relation",
    event_stream: "Event Stream",
    snapshot_nodes: "Snapshot Nodes",
    snapshot_edges: "Snapshot Edges",
    project_modules: "Project Modules",
    files: "Files",
    show_files: "Show files",
    log_system: "SYSTEM",
    log_error: "ERROR",
    log_workspace_refreshed: "workspace refreshed",
    log_action_complete: "complete",
    log_node_created: "node created",
    log_profile_imported: "profile imported from LLM",
    error_invalid_json_payload: "Invalid JSON payload",
  },
  ru: {
    app_title: "Рабочее пространство автономного графа",
    app_subtitle: "Графовый режим: узлы, связи, симуляция и анализ.",
    ui_language: "Язык интерфейса",
    action_refresh: "Обновить",
    action_seed_demo: "Смотреть демо",
    action_clear: "Очистить",
    runtime_error_title: "Ошибка фронтенда",
    runtime_error_hint: "Открой консоль браузера и исправь ошибку ниже.",
    action_try_continue: "Попробовать продолжить",
    action_reload_data: "Перезагрузить данные",
    page_overview: "Обзор",
    page_builder: "Конструктор",
    page_simulation: "Симуляция",
    page_data: "Данные",
    llm_profile_builder: "LLM-построение профиля",
    profile_input_text: "Текст-описание",
    entity_type_hint: "Подсказка типа сущности",
    entity_type_human: "Человек",
    entity_type_company: "Компания",
    entity_type_technology: "Технология",
    entity_type_generic: "Общее",
    extract_profile_graph: "Извлечь профиль -> Построить граф",
    profile_result: "Результат профиля от LLM",
    profile_prompt_preview: "Шаблон промпта",
    model_missing_hint: "Положи GGUF-модель в ./models и повтори.",
    workspace_status: "Статус пространства",
    execution_log: "Журнал выполнения",
    graph_visualization: "Визуализация графа",
    graph_empty: "Граф пуст. Добавь узлы и связи, затем запусти симуляцию.",
    create_node: "Создать узел",
    node_type: "Тип узла",
    first_name: "Имя",
    last_name: "Фамилия",
    company_name: "Название компании",
    company_industry: "Отрасль",
    company_description: "Описание",
    bio: "Биография",
    profile_text: "Профильный текст",
    profile_placeholder:
      "предпочтения: джаз, исторические фильмы\\nценности: свобода, бизнес\\nстрахи: крах рынка\\nцели: запустить продукт",
    employment_text: "Текст занятости",
    employment_placeholder: "founder @ Vector Dynamics; engineer at North Capital",
    employment_json: "JSON занятости",
    attributes_json: "JSON атрибутов",
    state_json: "JSON состояния",
    create_node_btn: "Создать узел",
    create_edge: "Создать связь",
    from_node: "Из узла",
    to_node: "В узел",
    relation: "Тип связи",
    weight: "Вес",
    direction: "Направление",
    logic_rule: "Логическое правило",
    directed: "направленная",
    undirected: "ненаправленная",
    create_edge_btn: "Создать связь",
    persistence: "Сохранение",
    persist_snapshot: "Сохранить снимок",
    load_snapshot: "Загрузить снимок",
    simulation: "Симуляция",
    seed_ids: "Seed ID",
    depth: "Глубина",
    steps: "Шаги",
    damping: "Затухание",
    activation: "Активация",
    infer_rounds: "Раунды вывода",
    run_simulation: "Запустить симуляцию",
    last_simulation_output: "Последний результат симуляции",
    event_feedback: "Обратная связь по событиям",
    event_id: "ID события",
    reward: "Награда",
    learning_rate: "Скорость обучения",
    apply_reward: "Применить награду",
    batch_relation_reinforcement: "Пакетное усиление связей",
    reinforce_relation: "Усилить связь",
    event_stream: "Поток событий",
    snapshot_nodes: "Узлы снимка",
    snapshot_edges: "Связи снимка",
    project_modules: "Модули проекта",
    files: "Файлы",
    show_files: "Показать файлы",
    log_system: "СИСТЕМА",
    log_error: "ОШИБКА",
    log_workspace_refreshed: "пространство обновлено",
    log_action_complete: "завершено",
    log_node_created: "узел создан",
    log_profile_imported: "профиль импортирован через LLM",
    error_invalid_json_payload: "Некорректный JSON",
  },
  hy: {
    app_title: "Ավտոնոմ գրաֆի աշխատանքային միջավայր",
    app_subtitle: "Գրաֆային ռեժիմ՝ հանգույցներ, կապեր, սիմուլյացիա, վերլուծություն:",
    ui_language: "Ինտերֆեյսի լեզու",
    action_refresh: "Թարմացնել",
    action_seed_demo: "Դիտել դեմո",
    action_clear: "Մաքրել",
    runtime_error_title: "Ֆրոնթենդի սխալ",
    runtime_error_hint: "Բացիր browser console-ն ու ուղղիր ստորև սխալը:",
    action_try_continue: "Փորձել շարունակել",
    action_reload_data: "Վերաբեռնել տվյալները",
    page_overview: "Ընդհանուր",
    page_builder: "Կառուցող",
    page_simulation: "Սիմուլյացիա",
    page_data: "Տվյալներ",
    llm_profile_builder: "LLM պրոֆիլ կոնստրուկտոր",
    profile_input_text: "Պատմողական տեքստ",
    entity_type_hint: "Entity տեսակի հուշում",
    entity_type_human: "Մարդ",
    entity_type_company: "Ընկերություն",
    entity_type_technology: "Տեխնոլոգիա",
    entity_type_generic: "Ընդհանուր",
    extract_profile_graph: "Քաղել պրոֆիլ -> Կառուցել գրաֆ",
    profile_result: "LLM պրոֆիլի արդյունք",
    profile_prompt_preview: "Պրոմպտի ձևանմուշ",
    model_missing_hint: "Տեղադրիր GGUF մոդել ./models թղթապանակում և նորից փորձիր:",
    workspace_status: "Միջավայրի վիճակ",
    execution_log: "Կատարումների log",
    graph_visualization: "Գրաֆի վիզուալիզացիա",
    graph_empty: "Գրաֆը դատարկ է։ Ավելացրու հանգույցներ և կապեր, հետո գործարկիր սիմուլյացիան։",
    create_node: "Ստեղծել հանգույց",
    node_type: "Հանգույցի տեսակ",
    first_name: "Անուն",
    last_name: "Ազգանուն",
    company_name: "Ընկերության անուն",
    company_industry: "Ոլորտ",
    company_description: "Նկարագրություն",
    bio: "Կենսագրություն",
    profile_text: "Պրոֆիլի տեքստ",
    profile_placeholder:
      "նախընտրություններ՝ ջազ, պատմական ֆիլմեր\\nարժեքներ՝ ազատություն, բիզնես\\nվախեր՝ շուկայի անկում\\ննպատակներ՝ թողարկել պրոդուկտ",
    employment_text: "Աշխատանքի տեքստ",
    employment_placeholder: "founder @ Vector Dynamics; engineer at North Capital",
    employment_json: "Աշխատանքի JSON",
    attributes_json: "Ատրիբուտների JSON",
    state_json: "Վիճակի JSON",
    create_node_btn: "Ստեղծել հանգույց",
    create_edge: "Ստեղծել կապ",
    from_node: "Որ հանգույցից",
    to_node: "Որ հանգույցին",
    relation: "Կապ",
    weight: "Քաշ",
    direction: "Ուղղություն",
    logic_rule: "Տրամաբանական կանոն",
    directed: "ուղղված",
    undirected: "չուղղված",
    create_edge_btn: "Ստեղծել կապ",
    persistence: "Պահպանում",
    persist_snapshot: "Պահպանել snapshot",
    load_snapshot: "Բեռնել snapshot",
    simulation: "Սիմուլյացիա",
    seed_ids: "Seed ID-ներ",
    depth: "Խորություն",
    steps: "Քայլեր",
    damping: "Թուլացում",
    activation: "Ակտիվացում",
    infer_rounds: "Եզրահանգման փուլեր",
    run_simulation: "Գործարկել սիմուլյացիա",
    last_simulation_output: "Վերջին արդյունք",
    event_feedback: "Իվենթների ֆիդբեք",
    event_id: "Իվենթի ID",
    reward: "Պարգև",
    learning_rate: "Ուսուցման արագություն",
    apply_reward: "Կիրառել պարգև",
    batch_relation_reinforcement: "Կապերի փաթեթային ուժեղացում",
    reinforce_relation: "Ուժեղացնել կապը",
    event_stream: "Իվենթների հոսք",
    snapshot_nodes: "Snapshot հանգույցներ",
    snapshot_edges: "Snapshot կապեր",
    project_modules: "Նախագծի մոդուլներ",
    files: "Ֆայլեր",
    show_files: "Ցույց տալ ֆայլերը",
    log_system: "ՀԱՄԱԿԱՐԳ",
    log_error: "ՍԽԱԼ",
    log_workspace_refreshed: "միջավայրը թարմացվեց",
    log_action_complete: "ավարտված",
    log_node_created: "հանգույցը ստեղծվեց",
    log_profile_imported: "պրոֆիլը ներմուծվեց LLM-ից",
    error_invalid_json_payload: "Սխալ JSON",
  },
  zh: {
    app_title: "自治图工作台",
    app_subtitle: "图优先模式：构建节点、推理关系、仿真与分析。",
    ui_language: "界面语言",
    action_refresh: "刷新",
    action_seed_demo: "观看演示",
    action_clear: "清空",
    runtime_error_title: "前端运行错误",
    runtime_error_hint: "请打开浏览器控制台并修复以下错误。",
    action_try_continue: "继续尝试",
    action_reload_data: "重新加载数据",
    page_overview: "概览",
    page_builder: "构建",
    page_simulation: "仿真",
    page_data: "数据",
    llm_profile_builder: "LLM 画像构建",
    profile_input_text: "叙述文本",
    entity_type_hint: "实体类型提示",
    entity_type_human: "人",
    entity_type_company: "公司",
    entity_type_technology: "技术",
    entity_type_generic: "通用",
    extract_profile_graph: "提取画像 -> 构建图",
    profile_result: "LLM 画像结果",
    profile_prompt_preview: "提示词模板",
    model_missing_hint: "请把 GGUF 模型放到 ./models 后重试。",
    workspace_status: "工作台状态",
    execution_log: "执行日志",
    graph_visualization: "图可视化",
    graph_empty: "图为空。请先添加节点和关系，再运行仿真。",
    create_node: "创建节点",
    node_type: "节点类型",
    first_name: "名",
    last_name: "姓",
    company_name: "公司名称",
    company_industry: "行业",
    company_description: "描述",
    bio: "简介",
    profile_text: "画像文本",
    profile_placeholder: "偏好: 爵士, 历史电影\\n价值观: 自由, 商业\\n担忧: 市场崩盘\\n目标: 发布产品",
    employment_text: "工作文本",
    employment_placeholder: "founder @ Vector Dynamics; engineer at North Capital",
    employment_json: "工作 JSON",
    attributes_json: "属性 JSON",
    state_json: "状态 JSON",
    create_node_btn: "创建节点",
    create_edge: "创建关系",
    from_node: "起点节点",
    to_node: "终点节点",
    relation: "关系",
    weight: "权重",
    direction: "方向",
    logic_rule: "逻辑规则",
    directed: "有向",
    undirected: "无向",
    create_edge_btn: "创建关系",
    persistence: "持久化",
    persist_snapshot: "保存快照",
    load_snapshot: "加载快照",
    simulation: "仿真",
    seed_ids: "种子 ID",
    depth: "深度",
    steps: "步数",
    damping: "阻尼",
    activation: "激活函数",
    infer_rounds: "推理轮数",
    run_simulation: "运行仿真",
    last_simulation_output: "最近仿真输出",
    event_feedback: "事件反馈",
    event_id: "事件 ID",
    reward: "奖励",
    learning_rate: "学习率",
    apply_reward: "应用奖励",
    batch_relation_reinforcement: "批量关系强化",
    reinforce_relation: "强化关系",
    event_stream: "事件流",
    snapshot_nodes: "快照节点",
    snapshot_edges: "快照边",
    project_modules: "项目模块",
    files: "文件",
    show_files: "显示文件",
    log_system: "系统",
    log_error: "错误",
    log_workspace_refreshed: "工作台已刷新",
    log_action_complete: "完成",
    log_node_created: "节点已创建",
    log_profile_imported: "已通过 LLM 导入画像",
    error_invalid_json_payload: "JSON 无效",
  },
  es: {
    app_title: "Espacio de trabajo de grafo autónomo",
    app_subtitle: "Modo centrado en grafo: nodos, relaciones, simulación y análisis.",
    ui_language: "Idioma de la interfaz",
    action_refresh: "Actualizar",
    action_seed_demo: "Ver demo",
    action_clear: "Limpiar",
    runtime_error_title: "Error de ejecución del frontend",
    runtime_error_hint: "Abre la consola del navegador y corrige el error mostrado.",
    action_try_continue: "Intentar continuar",
    action_reload_data: "Recargar datos",
    page_overview: "Resumen",
    page_builder: "Constructor",
    page_simulation: "Simulación",
    page_data: "Datos",
    llm_profile_builder: "Constructor de perfil LLM",
    profile_input_text: "Texto narrativo",
    entity_type_hint: "Tipo de entidad",
    entity_type_human: "Humano",
    entity_type_company: "Compañía",
    entity_type_technology: "Tecnología",
    entity_type_generic: "Genérico",
    extract_profile_graph: "Extraer perfil -> Construir grafo",
    profile_result: "Resultado de perfil LLM",
    profile_prompt_preview: "Plantilla de prompt",
    model_missing_hint: "Coloca un modelo GGUF en ./models e inténtalo de nuevo.",
    workspace_status: "Estado del espacio",
    execution_log: "Registro de ejecución",
    graph_visualization: "Visualización del grafo",
    graph_empty: "El grafo está vacío. Agrega nodos y relaciones y ejecuta la simulación.",
    create_node: "Crear nodo",
    node_type: "Tipo de nodo",
    first_name: "Nombre",
    last_name: "Apellido",
    company_name: "Nombre de empresa",
    company_industry: "Industria",
    company_description: "Descripción",
    bio: "Biografía",
    profile_text: "Texto de perfil",
    profile_placeholder:
      "preferencias: jazz, películas históricas\\nvalores: libertad, negocio\\nmiedos: caída del mercado\\nobjetivos: lanzar producto",
    employment_text: "Texto laboral",
    employment_placeholder: "founder @ Vector Dynamics; engineer at North Capital",
    employment_json: "JSON laboral",
    attributes_json: "JSON de atributos",
    state_json: "JSON de estado",
    create_node_btn: "Crear nodo",
    create_edge: "Crear relación",
    from_node: "Nodo origen",
    to_node: "Nodo destino",
    relation: "Relación",
    weight: "Peso",
    direction: "Dirección",
    logic_rule: "Regla lógica",
    directed: "dirigida",
    undirected: "no dirigida",
    create_edge_btn: "Crear relación",
    persistence: "Persistencia",
    persist_snapshot: "Guardar snapshot",
    load_snapshot: "Cargar snapshot",
    simulation: "Simulación",
    seed_ids: "IDs semilla",
    depth: "Profundidad",
    steps: "Pasos",
    damping: "Amortiguación",
    activation: "Activación",
    infer_rounds: "Rondas de inferencia",
    run_simulation: "Ejecutar simulación",
    last_simulation_output: "Última salida de simulación",
    event_feedback: "Feedback de eventos",
    event_id: "ID de evento",
    reward: "Recompensa",
    learning_rate: "Tasa de aprendizaje",
    apply_reward: "Aplicar recompensa",
    batch_relation_reinforcement: "Refuerzo masivo de relaciones",
    reinforce_relation: "Reforzar relación",
    event_stream: "Flujo de eventos",
    snapshot_nodes: "Nodos del snapshot",
    snapshot_edges: "Relaciones del snapshot",
    project_modules: "Módulos del proyecto",
    files: "Archivos",
    show_files: "Mostrar archivos",
    log_system: "SISTEMA",
    log_error: "ERROR",
    log_workspace_refreshed: "espacio actualizado",
    log_action_complete: "completado",
    log_node_created: "nodo creado",
    log_profile_imported: "perfil importado desde LLM",
    error_invalid_json_payload: "JSON inválido",
  },
  pt: {
    app_title: "Workspace de grafo autônomo",
    app_subtitle: "Modo orientado a grafo: nós, relações, simulação e análise.",
    ui_language: "Idioma da interface",
    action_refresh: "Atualizar",
    action_seed_demo: "Ver demo",
    action_clear: "Limpar",
    runtime_error_title: "Erro de execução do frontend",
    runtime_error_hint: "Abra o console do navegador e corrija o erro abaixo.",
    action_try_continue: "Tentar continuar",
    action_reload_data: "Recarregar dados",
    page_overview: "Visão geral",
    page_builder: "Construtor",
    page_simulation: "Simulação",
    page_data: "Dados",
    llm_profile_builder: "Construtor de perfil LLM",
    profile_input_text: "Texto narrativo",
    entity_type_hint: "Tipo de entidade",
    entity_type_human: "Humano",
    entity_type_company: "Empresa",
    entity_type_technology: "Tecnologia",
    entity_type_generic: "Genérico",
    extract_profile_graph: "Extrair perfil -> Construir grafo",
    profile_result: "Resultado do perfil LLM",
    profile_prompt_preview: "Template de prompt",
    model_missing_hint: "Coloque um modelo GGUF em ./models e tente novamente.",
    workspace_status: "Status do workspace",
    execution_log: "Log de execução",
    graph_visualization: "Visualização do grafo",
    graph_empty: "O grafo está vazio. Adicione nós e relações e rode a simulação.",
    create_node: "Criar nó",
    node_type: "Tipo de nó",
    first_name: "Nome",
    last_name: "Sobrenome",
    company_name: "Nome da empresa",
    company_industry: "Setor",
    company_description: "Descrição",
    bio: "Bio",
    profile_text: "Texto de perfil",
    profile_placeholder:
      "preferências: jazz, filmes históricos\\nvalores: liberdade, negócios\\nmedos: queda do mercado\\nobjetivos: lançar produto",
    employment_text: "Texto de trabalho",
    employment_placeholder: "founder @ Vector Dynamics; engineer at North Capital",
    employment_json: "JSON de trabalho",
    attributes_json: "JSON de atributos",
    state_json: "JSON de estado",
    create_node_btn: "Criar nó",
    create_edge: "Criar relação",
    from_node: "Nó de origem",
    to_node: "Nó de destino",
    relation: "Relação",
    weight: "Peso",
    direction: "Direção",
    logic_rule: "Regra lógica",
    directed: "direcionada",
    undirected: "não direcionada",
    create_edge_btn: "Criar relação",
    persistence: "Persistência",
    persist_snapshot: "Salvar snapshot",
    load_snapshot: "Carregar snapshot",
    simulation: "Simulação",
    seed_ids: "IDs semente",
    depth: "Profundidade",
    steps: "Passos",
    damping: "Amortecimento",
    activation: "Ativação",
    infer_rounds: "Rodadas de inferência",
    run_simulation: "Executar simulação",
    last_simulation_output: "Último resultado da simulação",
    event_feedback: "Feedback de eventos",
    event_id: "ID do evento",
    reward: "Recompensa",
    learning_rate: "Taxa de aprendizado",
    apply_reward: "Aplicar recompensa",
    batch_relation_reinforcement: "Reforço em lote de relações",
    reinforce_relation: "Reforçar relação",
    event_stream: "Fluxo de eventos",
    snapshot_nodes: "Nós do snapshot",
    snapshot_edges: "Arestas do snapshot",
    project_modules: "Módulos do projeto",
    files: "Arquivos",
    show_files: "Mostrar arquivos",
    log_system: "SISTEMA",
    log_error: "ERRO",
    log_workspace_refreshed: "workspace atualizado",
    log_action_complete: "concluído",
    log_node_created: "nó criado",
    log_profile_imported: "perfil importado via LLM",
    error_invalid_json_payload: "JSON inválido",
  },
};

const EXTRA_TRANSLATIONS = {
  en: {
    overview_sections: "Overview Sections",
    overview_section_status: "Status",
    overview_section_demo: "Demo",
    overview_section_daily: "Daily",
    overview_section_user_graph: "User Graph",
    overview_section_autoruns: "Autoruns",
    overview_section_graph: "Graph",
    overview_section_editor: "Editor",
    overview_section_client: "Client",
    overview_section_advisors: "Advisors",
    overview_section_hallucination_hunter: "Hallucination Hunter",
    pager_prev: "Prev",
    pager_next: "Next",
    pager_events: "Events",
    pager_nodes: "Nodes",
    pager_edges: "Edges",
    pager_modules: "Modules",
    demo_narrative: "Demo Narrative",
    scenario: "Scenario",
    demo_narrative_placeholder: "Alexa: who is this person, how childhood went, what they are happy with, and what they want...",
    refresh_client_profile: "Refresh My Client Profile",
    daily_mode: "Daily Mode",
    daily_journal: "Journal",
    daily_journal_placeholder: "Describe your day: goals, problems, wins, and what you want to improve...",
    run_daily_analysis: "Run Daily Analysis",
    daily_recommendations_scores: "Daily Recommendations + Scores",
    user_semantic_graph: "User Semantic Graph",
    user_graph_narrative: "Narrative Profile Text",
    user_graph_narrative_placeholder:
      "My name is ..., I grew up ..., now I do ..., I am satisfied with ..., I want ...",
    user_fears: "Fears",
    user_desires: "Desires",
    user_goals: "Goals",
    user_principles: "Principles",
    user_opportunities: "Opportunities",
    user_abilities: "Abilities",
    user_access: "Access",
    user_knowledge: "Knowledge",
    user_assets: "Assets",
    user_assets_placeholder: "physical, digital, spiritual...",
    apply_user_graph: "Apply User Graph",
    user_graph_update_result: "User Graph Update Result",
    autoruns_import_title: "Sysinternals Autoruns Import",
    autoruns_import_help:
      "Paste Autoruns CSV/TSV or type a query; if no table is provided the system infers startup/process profile from client telemetry.",
    autoruns_input_or_query: "Autoruns CSV/TSV or Query",
    autoruns_placeholder:
      "Entry,Entry Location,Enabled,Category,Profile,Description,Publisher,Image Path,Launch String,Signer,Verified,VirusTotal\n...",
    import_autoruns: "Import Autoruns",
    autoruns_import_result: "Autoruns Import Result",
    selected_node_editor: "Selected Node Editor",
    selected_edge_editor: "Selected Edge Editor",
    metadata_json: "Metadata JSON",
    update_node: "Update Node",
    delete_node: "Delete Node",
    update_edge: "Update Edge",
    delete_edge: "Delete Edge",
    client_profile_semantic_input: "My Client Profile (Semantic Input)",
    mini_coders_advisors: "Mini Coders / Advisors",
    prompt_catalog: "Prompt Catalog",
    sql_table_schema_json: "SQL Table Schema (JSON)",
    error_daily_journal_empty: "Daily journal text is empty",
    error_autoruns_text_empty: "Autoruns text is empty",
    error_profile_input_empty: "Narrative text is empty",
    action_daily_mode: "daily mode",
    action_user_graph_update: "user graph update",
    action_autoruns_import: "autoruns import",
    action_update_node: "update node",
    action_delete_node: "delete node",
    action_update_edge: "update edge",
    action_delete_edge: "delete edge",
    action_client_introspection: "client introspection",
    employment_json_placeholder: '[{"status":"engineer","importance_score":0.9,"company_name":"Vector Dynamics"}]',
    simulation_timeline: "Simulation Timeline",
    timeline_idle: "No simulation in progress yet",
    timeline_in_progress: "In Progress",
    timeline_completed: "Completed",
    timeline_progress: "Progress",
    llm_role_debate: "LLM Role Debate",
    debate_prompt: "Debate Topic",
    debate_variants: "Hypotheses",
    debate_attach_graph: "Attach Branches To Graph",
    debate_run: "Run Debate",
    debate_result: "Debate Result",
    debate_prompt_placeholder: "Propose strategy alternatives for this system goal...",
    error_debate_prompt_empty: "Debate topic is empty",
    action_llm_debate: "llm role debate",
    reasoning_path: "Reasoning Path",
    reasoning_path_empty: "Select a node to inspect its reasoning path and dependency closure.",
    reasoning_roots: "Roots",
    reasoning_chain: "Chain",
    reasoning_prerequisites: "Prerequisites",
    reasoning_dependents: "Dependents",
    graph_hover_dependencies_hint: "Hover any node to highlight its dependency closure.",
    reasoning_trace_options: "Trace Variants",
    reasoning_trace_empty: "No alternative traces found.",
    reasoning_trace_score: "score",
    edge_reasoning: "Edge Reasoning",
    edge_reasoning_empty: "Select an edge to inspect why it exists.",
    edge_reasoning_relation: "Relation",
    edge_reasoning_logic: "Logic",
    edge_reasoning_direction: "Direction",
    edge_reasoning_strength: "Strength",
    edge_reasoning_facts: "Key Facts",
    edge_history_timeline: "Edge Change Timeline",
    edge_history_empty: "No change events for this edge in current stream window.",
    edge_history_reason: "Reason",
    edge_history_weight: "Weight",
    edge_history_logic: "Logic",
    edge_history_event: "Event",
    personalization_studio: "Personalization Studio",
    personalization_summary: "Active Personalization",
    personalization_response_style: "Response Style",
    personalization_reasoning_depth: "Reasoning Depth",
    personalization_risk_tolerance: "Risk Tolerance",
    personalization_tone: "Tone",
    personalization_roles: "Default LLM Roles",
    personalization_focus_goals: "Focus Goals",
    personalization_domain_focus: "Domain Focus",
    personalization_avoid_topics: "Avoid Topics",
    personalization_memory_notes: "Memory Notes",
    personalization_auto_apply_user_graph: "Auto-apply in User Graph update",
    personalization_auto_apply_daily: "Auto-apply in Daily analysis",
    personalization_auto_apply_debate: "Auto-apply in LLM debate",
    personalization_save: "Save Personalization",
    personalization_reset: "Reset",
    personalization_sync_roles: "Sync Roles To Debate",
    personalization_goals_placeholder: "launch product, improve focus, reduce context switching",
    personalization_domains_placeholder: "architecture, growth, security",
    personalization_avoid_placeholder: "generic motivation, long theory",
    personalization_memory_placeholder: "Prefer practical steps, examples from previous runs, and concise action items.",
    log_personalization_saved: "personalization profile saved",
    log_personalization_reset: "personalization profile reset",
    log_personalization_roles_synced: "personalization roles synced to debate controls",
    hallucination_hunter_title: "Hallucination Hunter",
    hallucination_report_title: "Report Hallucination Case",
    hallucination_check_title: "Check For Known Hallucinations",
    hallucination_prompt: "Question / Prompt",
    hallucination_prompt_placeholder: "What was the user question where LLM hallucinated?",
    hallucination_wrong_answer: "Hallucinated Answer",
    hallucination_wrong_answer_placeholder: "Paste incorrect model answer",
    hallucination_correct_answer: "Correct Answer",
    hallucination_correct_answer_placeholder: "Paste verified correct answer",
    hallucination_source: "Verification Source",
    hallucination_source_placeholder: "Link, document, or trusted source",
    hallucination_tags: "Tags",
    hallucination_tags_placeholder: "facts, geography, medical...",
    hallucination_severity: "Severity",
    hallucination_report_action: "Save Hallucination",
    hallucination_report_result: "Saved Case Result",
    hallucination_check_action: "Check Repetition Risk",
    hallucination_check_result: "Hunter Check Result",
    hallucination_llm_answer_hint: "Current LLM Answer (optional)",
    hallucination_llm_answer_hint_placeholder: "Paste candidate answer for overlap check",
    error_hallucination_prompt_empty: "Prompt is empty",
    error_hallucination_wrong_empty: "Hallucinated answer is empty",
    error_hallucination_correct_empty: "Correct answer is empty",
    action_hallucination_report: "hallucination report",
    action_hallucination_check: "hallucination check",
    archive_chat_title: "Verified Archive Chat",
    archive_chat_message: "Message",
    archive_chat_message_placeholder: "Describe what should be added/changed in your archive...",
    archive_chat_context: "Context (optional)",
    archive_chat_context_placeholder: "Constraints, domain rules, evidence policy...",
    archive_chat_model_path: "Model Path",
    archive_chat_model_role: "Fallback Role",
    archive_chat_verification_mode: "Verification Mode",
    archive_chat_attach_graph: "Attach updates to graph branch",
    archive_chat_run: "Run Verified Chat",
    archive_chat_result: "Verified Archive Result",
    archive_chat_suggestions: "Detected Local Models",
    archive_chat_history: "Dialogue",
    archive_chat_no_messages: "No messages yet.",
    archive_review_title: "Review And Edit Conclusions",
    archive_review_editor: "Archive Updates JSON",
    archive_review_editor_placeholder: "[{\"entity\":\"...\",\"field\":\"...\",\"operation\":\"upsert\",\"value\":\"...\"}]",
    archive_review_apply: "Re-check And Apply Edited Draft",
    archive_review_result: "Review Check Result",
    error_archive_chat_message_empty: "Archive chat message is empty",
    error_archive_review_json_invalid: "Invalid archive updates JSON",
    action_archive_chat: "archive verified chat",
    action_archive_review_apply: "archive review apply",
    branch_visual_toolkit: "Branch Visual Toolkit",
    branch_visual_mode_cards: "Cards",
    branch_visual_mode_charts: "Charts",
    branch_visual_mode_lists: "Lists",
    branch_visual_mode_tips: "Lifehacks",
    branch_scope: "Scope",
    branch_scope_selected: "Selected Branch",
    branch_scope_global: "Global Graph",
    branch_metric_nodes: "Nodes",
    branch_metric_edges: "Edges",
    branch_metric_avg_weight: "Avg Edge Weight",
    branch_metric_hints: "Action Hints",
    branch_top_relations: "Top Relations",
    branch_top_node_types: "Top Node Types",
    branch_top_nodes: "Top Nodes",
    branch_top_edges: "Top Edges",
    branch_lifehacks: "Actionable Lifehacks",
    style_nodes_title: "Style Nodes (Safe UI Sandbox)",
    style_nodes_safe_hint: "Only visual tokens can change. Reset always restores baseline.",
    style_nodes_slider_label: "Style Node Slider",
    style_nodes_reset: "Reset To Base",
    style_nodes_activate: "Activate",
    style_nodes_active: "Active",
    style_nodes_name: "Style Name",
    style_nodes_description: "Style Description",
    style_nodes_vars_json: "Style Vars JSON",
    style_nodes_save: "Save Style Node",
    style_nodes_saved_result: "Saved Style Node Result",
    branch_report_editor_title: "Branch Report Node Editor",
    branch_report_summary: "Report Summary",
    branch_report_tips: "Report Tips",
    branch_report_save: "Save Branch Report Node",
    branch_report_saved_result: "Saved Branch Report Result",
    action_style_node_save: "save style node",
    action_branch_report_save: "save branch report",
    error_style_vars_json_invalid: "Invalid style vars JSON",
  },
  ru: {
    overview_sections: "Разделы обзора",
    overview_section_status: "Статус",
    overview_section_demo: "Демо",
    overview_section_daily: "Дневник",
    overview_section_user_graph: "Граф пользователя",
    overview_section_autoruns: "Autoruns",
    overview_section_graph: "Граф",
    overview_section_editor: "Редактор",
    overview_section_client: "Клиент",
    overview_section_advisors: "Советчики",
    overview_section_hallucination_hunter: "Охотник на галлюцинации",
    pager_prev: "Назад",
    pager_next: "Вперед",
    pager_events: "События",
    pager_nodes: "Узлы",
    pager_edges: "Связи",
    pager_modules: "Модули",
    demo_narrative: "Демо-сценарий",
    scenario: "Сценарий",
    demo_narrative_placeholder: "Алекса: кто он, как прошло детство, чем доволен и чего хочет...",
    refresh_client_profile: "Обновить мой профиль клиента",
    daily_mode: "Режим дневника",
    daily_journal: "Запись",
    daily_journal_placeholder: "Опиши день: цели, проблемы, что получилось, что хочешь улучшить...",
    run_daily_analysis: "Запустить анализ дня",
    daily_recommendations_scores: "Рекомендации и оценки за день",
    user_semantic_graph: "Семантический граф пользователя",
    user_graph_narrative: "Свободный текст профиля",
    user_graph_narrative_placeholder:
      "Меня зовут ..., я вырос ..., сейчас занимаюсь ..., доволен ..., хочу ...",
    user_fears: "Страхи",
    user_desires: "Желания",
    user_goals: "Цели",
    user_principles: "Принципы",
    user_opportunities: "Возможности",
    user_abilities: "Умения",
    user_access: "Доступ",
    user_knowledge: "Знания",
    user_assets: "Имущество",
    user_assets_placeholder: "физическое, цифровое, духовное...",
    apply_user_graph: "Применить граф пользователя",
    user_graph_update_result: "Результат обновления графа пользователя",
    autoruns_import_title: "Импорт Sysinternals Autoruns",
    autoruns_import_help:
      "Вставь CSV/TSV Autoruns или текстовый запрос; без таблицы система сама построит профиль автозапуска/процессов из клиентской телеметрии.",
    autoruns_input_or_query: "Autoruns CSV/TSV или запрос",
    autoruns_placeholder:
      "Entry,Entry Location,Enabled,Category,Profile,Description,Publisher,Image Path,Launch String,Signer,Verified,VirusTotal\n...",
    import_autoruns: "Импортировать Autoruns",
    autoruns_import_result: "Результат импорта Autoruns",
    selected_node_editor: "Редактор выбранного узла",
    selected_edge_editor: "Редактор выбранной связи",
    metadata_json: "JSON метаданных",
    update_node: "Обновить узел",
    delete_node: "Удалить узел",
    update_edge: "Обновить связь",
    delete_edge: "Удалить связь",
    client_profile_semantic_input: "Мой профиль клиента (семантический ввод)",
    mini_coders_advisors: "Мини-кодеры / советчики",
    prompt_catalog: "Каталог промптов",
    sql_table_schema_json: "Схема SQL-таблиц (JSON)",
    error_daily_journal_empty: "Текст дневника пуст",
    error_autoruns_text_empty: "Текст Autoruns пуст",
    error_profile_input_empty: "Текст-описание пуст",
    action_daily_mode: "режим дневника",
    action_user_graph_update: "обновление графа пользователя",
    action_autoruns_import: "импорт autoruns",
    action_update_node: "обновление узла",
    action_delete_node: "удаление узла",
    action_update_edge: "обновление связи",
    action_delete_edge: "удаление связи",
    action_client_introspection: "интроспекция клиента",
    employment_json_placeholder: '[{"status":"инженер","importance_score":0.9,"company_name":"Vector Dynamics"}]',
    simulation_timeline: "Таймлайн симуляции",
    timeline_idle: "Симуляция еще не запускалась",
    timeline_in_progress: "В процессе",
    timeline_completed: "Завершено",
    timeline_progress: "Прогресс",
    llm_role_debate: "LLM-дебаты ролей",
    debate_prompt: "Тема дебатов",
    debate_variants: "Гипотезы",
    debate_attach_graph: "Сохранять ветки в граф",
    debate_run: "Запустить дебаты",
    debate_result: "Результат дебатов",
    debate_prompt_placeholder: "Предложи альтернативные стратегии для этой цели системы...",
    error_debate_prompt_empty: "Тема дебатов пуста",
    action_llm_debate: "llm дебаты ролей",
    reasoning_path: "Цепочка рассуждений",
    reasoning_path_empty: "Выбери узел, чтобы увидеть цепочку и замыкание зависимостей.",
    reasoning_roots: "Корни",
    reasoning_chain: "Цепочка",
    reasoning_prerequisites: "Предпосылки",
    reasoning_dependents: "Зависимые",
    graph_hover_dependencies_hint: "Наведи курсор на узел, чтобы подсветить его зависимости.",
    reasoning_trace_options: "Варианты трассы",
    reasoning_trace_empty: "Альтернативные трассы не найдены.",
    reasoning_trace_score: "оценка",
    edge_reasoning: "Объяснение связи",
    edge_reasoning_empty: "Выбери связь, чтобы увидеть причину её существования.",
    edge_reasoning_relation: "Тип связи",
    edge_reasoning_logic: "Логика",
    edge_reasoning_direction: "Направление",
    edge_reasoning_strength: "Сила",
    edge_reasoning_facts: "Ключевые факты",
    edge_history_timeline: "Таймлайн изменений связи",
    edge_history_empty: "В текущем окне stream нет событий изменения этой связи.",
    edge_history_reason: "Причина",
    edge_history_weight: "Вес",
    edge_history_logic: "Логика",
    edge_history_event: "Событие",
    personalization_studio: "Студия персонализации",
    personalization_summary: "Активная персонализация",
    personalization_response_style: "Стиль ответа",
    personalization_reasoning_depth: "Глубина рассуждения",
    personalization_risk_tolerance: "Профиль риска",
    personalization_tone: "Тон",
    personalization_roles: "Роли LLM по умолчанию",
    personalization_focus_goals: "Фокус-цели",
    personalization_domain_focus: "Фокус-домены",
    personalization_avoid_topics: "Чего избегать",
    personalization_memory_notes: "Память и контекст",
    personalization_auto_apply_user_graph: "Автоприменение в обновлении графа пользователя",
    personalization_auto_apply_daily: "Автоприменение в дневном анализе",
    personalization_auto_apply_debate: "Автоприменение в LLM-дебатах",
    personalization_save: "Сохранить персонализацию",
    personalization_reset: "Сбросить",
    personalization_sync_roles: "Синхронизировать роли в дебатах",
    personalization_goals_placeholder: "запуск продукта, фокус, снижение переключения контекста",
    personalization_domains_placeholder: "архитектура, рост, безопасность",
    personalization_avoid_placeholder: "общая мотивация, длинная теория",
    personalization_memory_placeholder:
      "Предпочитаю практичные шаги, примеры из прошлых запусков и короткие action items.",
    log_personalization_saved: "профиль персонализации сохранен",
    log_personalization_reset: "профиль персонализации сброшен",
    log_personalization_roles_synced: "роли персонализации синхронизированы с дебатами",
    hallucination_hunter_title: "Охотник на галлюцинации",
    hallucination_report_title: "Сообщить о галлюцинации",
    hallucination_check_title: "Проверка на повтор известной галлюцинации",
    hallucination_prompt: "Вопрос / Промпт",
    hallucination_prompt_placeholder: "Какой был исходный вопрос, где LLM ошиблась?",
    hallucination_wrong_answer: "Ошибочный ответ LLM",
    hallucination_wrong_answer_placeholder: "Вставь неверный ответ модели",
    hallucination_correct_answer: "Правильный ответ",
    hallucination_correct_answer_placeholder: "Вставь проверенный корректный ответ",
    hallucination_source: "Источник проверки",
    hallucination_source_placeholder: "Ссылка, документ или надежный источник",
    hallucination_tags: "Теги",
    hallucination_tags_placeholder: "факты, география, медицина...",
    hallucination_severity: "Серьезность",
    hallucination_report_action: "Сохранить галлюцинацию",
    hallucination_report_result: "Результат сохранения",
    hallucination_check_action: "Проверить риск повтора",
    hallucination_check_result: "Результат проверки охотника",
    hallucination_llm_answer_hint: "Текущий ответ LLM (необязательно)",
    hallucination_llm_answer_hint_placeholder: "Вставь ответ для проверки пересечения",
    error_hallucination_prompt_empty: "Пустой вопрос/промпт",
    error_hallucination_wrong_empty: "Пустой ошибочный ответ LLM",
    error_hallucination_correct_empty: "Пустой правильный ответ",
    action_hallucination_report: "запись галлюцинации",
    action_hallucination_check: "проверка галлюцинации",
    archive_chat_title: "Проверенный архивный чат",
    archive_chat_message: "Сообщение",
    archive_chat_message_placeholder: "Опиши, что нужно добавить/изменить в архиве...",
    archive_chat_context: "Контекст (необязательно)",
    archive_chat_context_placeholder: "Ограничения, правила домена, политика источников...",
    archive_chat_model_path: "Путь к модели",
    archive_chat_model_role: "Роль по умолчанию",
    archive_chat_verification_mode: "Режим проверки",
    archive_chat_attach_graph: "Сохранять обновления в ветку графа",
    archive_chat_run: "Запустить проверенный чат",
    archive_chat_result: "Результат проверенного архива",
    archive_chat_suggestions: "Найденные локальные модели",
    archive_chat_history: "Диалог",
    archive_chat_no_messages: "Сообщений пока нет.",
    archive_review_title: "Проверка и редактирование выводов",
    archive_review_editor: "JSON обновлений архива",
    archive_review_editor_placeholder: "[{\"entity\":\"...\",\"field\":\"...\",\"operation\":\"upsert\",\"value\":\"...\"}]",
    archive_review_apply: "Перепроверить и применить правки",
    archive_review_result: "Результат проверки правок",
    error_archive_chat_message_empty: "Пустое сообщение архивного чата",
    error_archive_review_json_invalid: "Некорректный JSON обновлений архива",
    action_archive_chat: "проверенный архивный чат",
    action_archive_review_apply: "применение review архива",
    branch_visual_toolkit: "Визуальный набор ветки",
    branch_visual_mode_cards: "Карточки",
    branch_visual_mode_charts: "Графики",
    branch_visual_mode_lists: "Списки",
    branch_visual_mode_tips: "Лайфхаки",
    branch_scope: "Охват",
    branch_scope_selected: "Выбранная ветка",
    branch_scope_global: "Весь граф",
    branch_metric_nodes: "Узлы",
    branch_metric_edges: "Связи",
    branch_metric_avg_weight: "Средний вес связи",
    branch_metric_hints: "Подсказки действий",
    branch_top_relations: "Топ отношений",
    branch_top_node_types: "Топ типов узлов",
    branch_top_nodes: "Топ узлов",
    branch_top_edges: "Топ связей",
    branch_lifehacks: "Практические лайфхаки",
    style_nodes_title: "Узлы стиля (безопасная песочница UI)",
    style_nodes_safe_hint: "Меняются только визуальные токены. Сброс всегда возвращает базовый стиль.",
    style_nodes_slider_label: "Слайдер узла стиля",
    style_nodes_reset: "Сбросить к базовому",
    style_nodes_activate: "Активировать",
    style_nodes_active: "Активен",
    style_nodes_name: "Название стиля",
    style_nodes_description: "Описание стиля",
    style_nodes_vars_json: "JSON переменных стиля",
    style_nodes_save: "Сохранить узел стиля",
    style_nodes_saved_result: "Результат сохранения узла стиля",
    branch_report_editor_title: "Редактор узла отчета ветки",
    branch_report_summary: "Сводка отчета",
    branch_report_tips: "Подсказки отчета",
    branch_report_save: "Сохранить узел отчета ветки",
    branch_report_saved_result: "Результат сохранения отчета ветки",
    action_style_node_save: "сохранение узла стиля",
    action_branch_report_save: "сохранение отчета ветки",
    error_style_vars_json_invalid: "Некорректный JSON переменных стиля",
  },
  hy: {
    overview_sections: "Ընդհանուր բաժիններ",
    overview_section_status: "Վիճակ",
    overview_section_demo: "Դեմո",
    overview_section_daily: "Օրագիր",
    overview_section_user_graph: "Օգտատիրոջ գրաֆ",
    overview_section_autoruns: "Autoruns",
    overview_section_graph: "Գրաֆ",
    overview_section_editor: "Խմբագրիչ",
    overview_section_client: "Հաճախորդ",
    overview_section_advisors: "Խորհրդատուներ",
    pager_prev: "Նախորդ",
    pager_next: "Հաջորդ",
    pager_events: "Իվենթներ",
    pager_nodes: "Հանգույցներ",
    pager_edges: "Կապեր",
    pager_modules: "Մոդուլներ",
    demo_narrative: "Դեմո սցենար",
    scenario: "Սցենար",
    demo_narrative_placeholder: "Ալեքսա․ ով է նա, ինչպես է անցել մանկությունը, ինչով է գոհ, ինչ է ուզում...",
    refresh_client_profile: "Թարմացնել իմ հաճախորդի պրոֆիլը",
    daily_mode: "Օրագրի ռեժիմ",
    daily_journal: "Գրառում",
    daily_journal_placeholder: "Նկարագրիր օրը՝ նպատակներ, խնդիրներ, հաջողություններ, ինչն ես ուզում բարելավել...",
    run_daily_analysis: "Գործարկել օրվա վերլուծություն",
    daily_recommendations_scores: "Օրվա առաջարկներ և գնահատականներ",
    user_semantic_graph: "Օգտատիրոջ սեմանտիկ գրաֆ",
    user_graph_narrative: "Պրոֆիլի ազատ տեքստ",
    user_graph_narrative_placeholder:
      "Իմ անունը ..., մեծացել եմ ..., հիմա զբաղվում եմ ..., գոհ եմ ..., ուզում եմ ...",
    user_fears: "Վախեր",
    user_desires: "Ցանկություններ",
    user_goals: "Նպատակներ",
    user_principles: "Սկզբունքներ",
    user_opportunities: "Հնարավորություններ",
    user_abilities: "Հմտություններ",
    user_access: "Մուտք",
    user_knowledge: "Գիտելիք",
    user_assets: "Գույք",
    user_assets_placeholder: "ֆիզիկական, թվային, հոգևոր...",
    apply_user_graph: "Կիրառել օգտատիրոջ գրաֆը",
    user_graph_update_result: "Օգտատիրոջ գրաֆի թարմացման արդյունք",
    autoruns_import_title: "Sysinternals Autoruns ներմուծում",
    autoruns_import_help:
      "Տեղադրիր Autoruns CSV/TSV կամ գրիր հարցում․ եթե աղյուսակ չկա, համակարգը կկազմի startup/process պրոֆիլը հաճախորդի telemetry-ից։",
    autoruns_input_or_query: "Autoruns CSV/TSV կամ հարցում",
    autoruns_placeholder:
      "Entry,Entry Location,Enabled,Category,Profile,Description,Publisher,Image Path,Launch String,Signer,Verified,VirusTotal\n...",
    import_autoruns: "Ներմուծել Autoruns",
    autoruns_import_result: "Autoruns ներմուծման արդյունք",
    selected_node_editor: "Ընտրված հանգույցի խմբագրիչ",
    selected_edge_editor: "Ընտրված կապի խմբագրիչ",
    metadata_json: "Մետատվյալների JSON",
    update_node: "Թարմացնել հանգույցը",
    delete_node: "Ջնջել հանգույցը",
    update_edge: "Թարմացնել կապը",
    delete_edge: "Ջնջել կապը",
    client_profile_semantic_input: "Իմ հաճախորդի պրոֆիլը (սեմանտիկ մուտք)",
    mini_coders_advisors: "Մինի կոդերներ / խորհրդատուներ",
    prompt_catalog: "Պրոմպտների կատալոգ",
    sql_table_schema_json: "SQL աղյուսակների սխեմա (JSON)",
    error_daily_journal_empty: "Օրագրի տեքստը դատարկ է",
    error_autoruns_text_empty: "Autoruns տեքստը դատարկ է",
    error_profile_input_empty: "Պատմողական տեքստը դատարկ է",
    action_daily_mode: "օրագրի ռեժիմ",
    action_user_graph_update: "օգտատիրոջ գրաֆի թարմացում",
    action_autoruns_import: "autoruns ներմուծում",
    action_update_node: "հանգույցի թարմացում",
    action_delete_node: "հանգույցի ջնջում",
    action_update_edge: "կապի թարմացում",
    action_delete_edge: "կապի ջնջում",
    action_client_introspection: "հաճախորդի ինտրոսպեկցիա",
    employment_json_placeholder: '[{"status":"engineer","importance_score":0.9,"company_name":"Vector Dynamics"}]',
    simulation_timeline: "Սիմուլյացիայի թայմլայն",
    timeline_idle: "Սիմուլյացիա դեռ չի գործարկվել",
    timeline_in_progress: "Ընթացքի մեջ",
    timeline_completed: "Ավարտված",
    timeline_progress: "Պրոգրես",
    llm_role_debate: "LLM role debate",
    debate_prompt: "Debate թեմա",
    debate_variants: "Հիպոթեզներ",
    debate_attach_graph: "Պահել branch-երը գրաֆում",
    debate_run: "Գործարկել debate",
    debate_result: "Debate արդյունք",
    debate_prompt_placeholder: "Առաջարկիր տարբեր ռազմավարություններ այս նպատակի համար...",
    error_debate_prompt_empty: "Debate թեման դատարկ է",
    action_llm_debate: "llm role debate",
  },
  zh: {
    overview_sections: "概览分区",
    overview_section_status: "状态",
    overview_section_demo: "演示",
    overview_section_daily: "日记",
    overview_section_user_graph: "用户图谱",
    overview_section_autoruns: "Autoruns",
    overview_section_graph: "图谱",
    overview_section_editor: "编辑器",
    overview_section_client: "客户端",
    overview_section_advisors: "顾问",
    pager_prev: "上一页",
    pager_next: "下一页",
    pager_events: "事件",
    pager_nodes: "节点",
    pager_edges: "边",
    pager_modules: "模块",
    demo_narrative: "演示叙述",
    scenario: "场景",
    demo_narrative_placeholder: "Alexa：这个人是谁，童年如何，满意什么，想要什么……",
    refresh_client_profile: "刷新我的客户端画像",
    daily_mode: "日记模式",
    daily_journal: "记录",
    daily_journal_placeholder: "描述你的一天：目标、问题、收获，以及希望改进的内容……",
    run_daily_analysis: "运行每日分析",
    daily_recommendations_scores: "每日建议与评分",
    user_semantic_graph: "用户语义图谱",
    user_graph_narrative: "自由叙述文本",
    user_graph_narrative_placeholder: "我叫..., 我成长于..., 现在从事..., 我满意..., 我想要...",
    user_fears: "恐惧",
    user_desires: "愿望",
    user_goals: "目标",
    user_principles: "原则",
    user_opportunities: "机会",
    user_abilities: "能力",
    user_access: "访问",
    user_knowledge: "知识",
    user_assets: "资产",
    user_assets_placeholder: "实体、数字、精神……",
    apply_user_graph: "应用用户图谱",
    user_graph_update_result: "用户图谱更新结果",
    autoruns_import_title: "Sysinternals Autoruns 导入",
    autoruns_import_help: "粘贴 Autoruns CSV/TSV 或输入查询；若未提供表格，系统将基于客户端遥测推断启动/进程画像。",
    autoruns_input_or_query: "Autoruns CSV/TSV 或查询",
    autoruns_placeholder:
      "Entry,Entry Location,Enabled,Category,Profile,Description,Publisher,Image Path,Launch String,Signer,Verified,VirusTotal\n...",
    import_autoruns: "导入 Autoruns",
    autoruns_import_result: "Autoruns 导入结果",
    selected_node_editor: "所选节点编辑器",
    selected_edge_editor: "所选边编辑器",
    metadata_json: "元数据 JSON",
    update_node: "更新节点",
    delete_node: "删除节点",
    update_edge: "更新边",
    delete_edge: "删除边",
    client_profile_semantic_input: "我的客户端画像（语义输入）",
    mini_coders_advisors: "迷你编码顾问",
    prompt_catalog: "提示词目录",
    sql_table_schema_json: "SQL 表结构（JSON）",
    error_daily_journal_empty: "日记文本为空",
    error_autoruns_text_empty: "Autoruns 文本为空",
    error_profile_input_empty: "叙述文本为空",
    action_daily_mode: "日记模式",
    action_user_graph_update: "更新用户图谱",
    action_autoruns_import: "导入 autoruns",
    action_update_node: "更新节点",
    action_delete_node: "删除节点",
    action_update_edge: "更新边",
    action_delete_edge: "删除边",
    action_client_introspection: "客户端探查",
    employment_json_placeholder: '[{"status":"工程师","importance_score":0.9,"company_name":"Vector Dynamics"}]',
    simulation_timeline: "仿真时间线",
    timeline_idle: "尚未运行仿真",
    timeline_in_progress: "进行中",
    timeline_completed: "已完成",
    timeline_progress: "进度",
    llm_role_debate: "LLM 角色辩论",
    debate_prompt: "辩论主题",
    debate_variants: "假设数量",
    debate_attach_graph: "将分支写入图谱",
    debate_run: "运行辩论",
    debate_result: "辩论结果",
    debate_prompt_placeholder: "为这个系统目标提出多个策略假设...",
    error_debate_prompt_empty: "辩论主题为空",
    action_llm_debate: "llm 角色辩论",
  },
  es: {
    overview_sections: "Secciones de resumen",
    overview_section_status: "Estado",
    overview_section_demo: "Demo",
    overview_section_daily: "Diario",
    overview_section_user_graph: "Grafo del usuario",
    overview_section_autoruns: "Autoruns",
    overview_section_graph: "Grafo",
    overview_section_editor: "Editor",
    overview_section_client: "Cliente",
    overview_section_advisors: "Asesores",
    pager_prev: "Anterior",
    pager_next: "Siguiente",
    pager_events: "Eventos",
    pager_nodes: "Nodos",
    pager_edges: "Relaciones",
    pager_modules: "Módulos",
    demo_narrative: "Narrativa demo",
    scenario: "Escenario",
    demo_narrative_placeholder: "Alexa: quién es, cómo fue su infancia, con qué está satisfecho y qué quiere...",
    refresh_client_profile: "Actualizar mi perfil de cliente",
    daily_mode: "Modo diario",
    daily_journal: "Diario",
    daily_journal_placeholder: "Describe tu día: metas, problemas, logros y qué quieres mejorar...",
    run_daily_analysis: "Ejecutar análisis diario",
    daily_recommendations_scores: "Recomendaciones y puntuaciones del día",
    user_semantic_graph: "Grafo semántico del usuario",
    user_graph_narrative: "Texto narrativo del perfil",
    user_graph_narrative_placeholder:
      "Me llamo..., crecí..., ahora me dedico a..., estoy satisfecho con..., quiero...",
    user_fears: "Miedos",
    user_desires: "Deseos",
    user_goals: "Metas",
    user_principles: "Principios",
    user_opportunities: "Oportunidades",
    user_abilities: "Habilidades",
    user_access: "Acceso",
    user_knowledge: "Conocimiento",
    user_assets: "Activos",
    user_assets_placeholder: "físico, digital, espiritual...",
    apply_user_graph: "Aplicar grafo del usuario",
    user_graph_update_result: "Resultado de actualización del grafo del usuario",
    autoruns_import_title: "Importación de Sysinternals Autoruns",
    autoruns_import_help:
      "Pega CSV/TSV de Autoruns o escribe una consulta; sin tabla, el sistema infiere el perfil de inicio/procesos desde la telemetría del cliente.",
    autoruns_input_or_query: "Autoruns CSV/TSV o consulta",
    autoruns_placeholder:
      "Entry,Entry Location,Enabled,Category,Profile,Description,Publisher,Image Path,Launch String,Signer,Verified,VirusTotal\n...",
    import_autoruns: "Importar Autoruns",
    autoruns_import_result: "Resultado de importación de Autoruns",
    selected_node_editor: "Editor de nodo seleccionado",
    selected_edge_editor: "Editor de relación seleccionada",
    metadata_json: "JSON de metadatos",
    update_node: "Actualizar nodo",
    delete_node: "Eliminar nodo",
    update_edge: "Actualizar relación",
    delete_edge: "Eliminar relación",
    client_profile_semantic_input: "Mi perfil de cliente (entrada semántica)",
    mini_coders_advisors: "Mini coders / asesores",
    prompt_catalog: "Catálogo de prompts",
    sql_table_schema_json: "Esquema de tablas SQL (JSON)",
    error_daily_journal_empty: "El texto del diario está vacío",
    error_autoruns_text_empty: "El texto de Autoruns está vacío",
    error_profile_input_empty: "El texto narrativo está vacío",
    action_daily_mode: "modo diario",
    action_user_graph_update: "actualización del grafo del usuario",
    action_autoruns_import: "importación de autoruns",
    action_update_node: "actualización de nodo",
    action_delete_node: "eliminación de nodo",
    action_update_edge: "actualización de relación",
    action_delete_edge: "eliminación de relación",
    action_client_introspection: "introspección de cliente",
    employment_json_placeholder: '[{"status":"ingeniero","importance_score":0.9,"company_name":"Vector Dynamics"}]',
    simulation_timeline: "Timeline de simulación",
    timeline_idle: "Aún no se ejecutó simulación",
    timeline_in_progress: "En progreso",
    timeline_completed: "Completado",
    timeline_progress: "Progreso",
    llm_role_debate: "Debate de roles LLM",
    debate_prompt: "Tema del debate",
    debate_variants: "Hipótesis",
    debate_attach_graph: "Guardar ramas en el grafo",
    debate_run: "Ejecutar debate",
    debate_result: "Resultado del debate",
    debate_prompt_placeholder: "Propón estrategias alternativas para este objetivo del sistema...",
    error_debate_prompt_empty: "El tema del debate está vacío",
    action_llm_debate: "debate de roles llm",
  },
  pt: {
    overview_sections: "Seções da visão geral",
    overview_section_status: "Status",
    overview_section_demo: "Demo",
    overview_section_daily: "Diário",
    overview_section_user_graph: "Grafo do usuário",
    overview_section_autoruns: "Autoruns",
    overview_section_graph: "Grafo",
    overview_section_editor: "Editor",
    overview_section_client: "Cliente",
    overview_section_advisors: "Conselheiros",
    pager_prev: "Anterior",
    pager_next: "Próximo",
    pager_events: "Eventos",
    pager_nodes: "Nós",
    pager_edges: "Arestas",
    pager_modules: "Módulos",
    demo_narrative: "Narrativa de demo",
    scenario: "Cenário",
    demo_narrative_placeholder: "Alexa: quem é, como foi a infância, com o que está satisfeito e o que deseja...",
    refresh_client_profile: "Atualizar meu perfil de cliente",
    daily_mode: "Modo diário",
    daily_journal: "Diário",
    daily_journal_placeholder: "Descreva seu dia: metas, problemas, conquistas e o que quer melhorar...",
    run_daily_analysis: "Executar análise diária",
    daily_recommendations_scores: "Recomendações e pontuações do dia",
    user_semantic_graph: "Grafo semântico do usuário",
    user_graph_narrative: "Texto narrativo do perfil",
    user_graph_narrative_placeholder:
      "Meu nome é..., cresci..., hoje faço..., estou satisfeito com..., quero...",
    user_fears: "Medos",
    user_desires: "Desejos",
    user_goals: "Metas",
    user_principles: "Princípios",
    user_opportunities: "Oportunidades",
    user_abilities: "Habilidades",
    user_access: "Acesso",
    user_knowledge: "Conhecimento",
    user_assets: "Ativos",
    user_assets_placeholder: "físico, digital, espiritual...",
    apply_user_graph: "Aplicar grafo do usuário",
    user_graph_update_result: "Resultado da atualização do grafo do usuário",
    autoruns_import_title: "Importação do Sysinternals Autoruns",
    autoruns_import_help:
      "Cole CSV/TSV do Autoruns ou escreva uma consulta; sem tabela, o sistema infere perfil de inicialização/processos a partir da telemetria do cliente.",
    autoruns_input_or_query: "Autoruns CSV/TSV ou consulta",
    autoruns_placeholder:
      "Entry,Entry Location,Enabled,Category,Profile,Description,Publisher,Image Path,Launch String,Signer,Verified,VirusTotal\n...",
    import_autoruns: "Importar Autoruns",
    autoruns_import_result: "Resultado da importação do Autoruns",
    selected_node_editor: "Editor do nó selecionado",
    selected_edge_editor: "Editor da aresta selecionada",
    metadata_json: "JSON de metadados",
    update_node: "Atualizar nó",
    delete_node: "Excluir nó",
    update_edge: "Atualizar aresta",
    delete_edge: "Excluir aresta",
    client_profile_semantic_input: "Meu perfil de cliente (entrada semântica)",
    mini_coders_advisors: "Mini coders / conselheiros",
    prompt_catalog: "Catálogo de prompts",
    sql_table_schema_json: "Esquema das tabelas SQL (JSON)",
    error_daily_journal_empty: "O texto do diário está vazio",
    error_autoruns_text_empty: "O texto do Autoruns está vazio",
    error_profile_input_empty: "O texto narrativo está vazio",
    action_daily_mode: "modo diário",
    action_user_graph_update: "atualização do grafo do usuário",
    action_autoruns_import: "importação do autoruns",
    action_update_node: "atualização de nó",
    action_delete_node: "exclusão de nó",
    action_update_edge: "atualização de aresta",
    action_delete_edge: "exclusão de aresta",
    action_client_introspection: "introspecção do cliente",
    employment_json_placeholder: '[{"status":"engenheiro","importance_score":0.9,"company_name":"Vector Dynamics"}]',
    simulation_timeline: "Timeline da simulação",
    timeline_idle: "A simulação ainda não foi executada",
    timeline_in_progress: "Em progresso",
    timeline_completed: "Concluída",
    timeline_progress: "Progresso",
    llm_role_debate: "Debate de papéis LLM",
    debate_prompt: "Tema do debate",
    debate_variants: "Hipóteses",
    debate_attach_graph: "Salvar branches no grafo",
    debate_run: "Executar debate",
    debate_result: "Resultado do debate",
    debate_prompt_placeholder: "Proponha estratégias alternativas para este objetivo do sistema...",
    error_debate_prompt_empty: "O tema do debate está vazio",
    action_llm_debate: "debate de papéis llm",
  },
};

const TRANSLATION_EXTENSIONS = {
  fr: {
    app_title: "Espace de travail du graphe autonome",
    app_subtitle: "Mode orienté graphe : nœuds, relations, simulation et analyse.",
    ui_language: "Langue de l'interface",
    action_refresh: "Actualiser",
    action_seed_demo: "Voir la démo",
    action_clear: "Effacer",
    runtime_error_title: "Erreur d'exécution frontend",
    runtime_error_hint: "Ouvrez la console du navigateur et corrigez l'erreur ci-dessous.",
    action_try_continue: "Essayer de continuer",
    action_reload_data: "Recharger les données",
    page_overview: "Vue d'ensemble",
    page_builder: "Constructeur",
    page_simulation: "Simulation",
    page_data: "Données",
    llm_profile_builder: "Constructeur de profil LLM",
    profile_input_text: "Texte narratif",
    entity_type_hint: "Type d'entité",
    entity_type_human: "Humain",
    entity_type_company: "Entreprise",
    entity_type_technology: "Technologie",
    entity_type_generic: "Générique",
    extract_profile_graph: "Extraire le profil -> Construire le graphe",
    profile_result: "Résultat du profil LLM",
    profile_prompt_preview: "Modèle de prompt",
    model_missing_hint: "Placez un modèle GGUF dans ./models puis réessayez.",
    workspace_status: "État de l'espace",
    execution_log: "Journal d'exécution",
    graph_visualization: "Visualisation du graphe",
    graph_empty: "Le graphe est vide. Ajoutez des nœuds et des relations, puis lancez la simulation.",
    create_node: "Créer un nœud",
    node_type: "Type de nœud",
    first_name: "Prénom",
    last_name: "Nom",
    company_name: "Nom de l'entreprise",
    company_industry: "Secteur",
    company_description: "Description",
    bio: "Bio",
    profile_text: "Texte de profil",
    employment_text: "Texte d'emploi",
    employment_json: "JSON d'emploi",
    attributes_json: "JSON des attributs",
    state_json: "JSON d'état",
    create_node_btn: "Créer un nœud",
    create_edge: "Créer une relation",
    from_node: "Nœud source",
    to_node: "Nœud cible",
    relation: "Relation",
    weight: "Poids",
    direction: "Direction",
    logic_rule: "Règle logique",
    directed: "orientée",
    undirected: "non orientée",
    create_edge_btn: "Créer une relation",
    persistence: "Persistance",
    persist_snapshot: "Enregistrer le snapshot",
    load_snapshot: "Charger le snapshot",
    simulation: "Simulation",
    seed_ids: "ID de départ",
    depth: "Profondeur",
    steps: "Étapes",
    damping: "Amortissement",
    activation: "Activation",
    infer_rounds: "Tours d'inférence",
    run_simulation: "Lancer la simulation",
    last_simulation_output: "Dernier résultat de simulation",
    event_feedback: "Feedback d'événement",
    event_id: "ID d'événement",
    reward: "Récompense",
    learning_rate: "Taux d'apprentissage",
    apply_reward: "Appliquer la récompense",
    batch_relation_reinforcement: "Renforcement par lot des relations",
    reinforce_relation: "Renforcer la relation",
    event_stream: "Flux d'événements",
    snapshot_nodes: "Nœuds du snapshot",
    snapshot_edges: "Arêtes du snapshot",
    project_modules: "Modules du projet",
    files: "Fichiers",
    show_files: "Afficher les fichiers",
    log_system: "SYSTÈME",
    log_error: "ERREUR",
    log_workspace_refreshed: "espace rafraîchi",
    log_action_complete: "terminé",
    log_node_created: "nœud créé",
    log_profile_imported: "profil importé via LLM",
    error_invalid_json_payload: "JSON invalide",
  },
  ar: {
    app_title: "مساحة عمل الرسم البياني الذاتي",
    app_subtitle: "وضع يعتمد على الرسم البياني: عقد، علاقات، محاكاة وتحليل.",
    ui_language: "لغة الواجهة",
    action_refresh: "تحديث",
    action_seed_demo: "عرض التجربة",
    action_clear: "مسح",
    runtime_error_title: "خطأ وقت تشغيل الواجهة",
    runtime_error_hint: "افتح وحدة تحكم المتصفح وأصلح الخطأ أدناه.",
    action_try_continue: "محاولة المتابعة",
    action_reload_data: "إعادة تحميل البيانات",
    page_overview: "نظرة عامة",
    page_builder: "المنشئ",
    page_simulation: "المحاكاة",
    page_data: "البيانات",
    llm_profile_builder: "منشئ ملف LLM",
    profile_input_text: "نص سردي",
    entity_type_hint: "نوع الكيان",
    entity_type_human: "إنسان",
    entity_type_company: "شركة",
    entity_type_technology: "تقنية",
    entity_type_generic: "عام",
    extract_profile_graph: "استخراج الملف -> بناء الرسم",
    profile_result: "نتيجة ملف LLM",
    profile_prompt_preview: "قالب البرومبت",
    model_missing_hint: "ضع نموذج GGUF في ./models ثم أعد المحاولة.",
    workspace_status: "حالة المساحة",
    execution_log: "سجل التنفيذ",
    graph_visualization: "تصور الرسم البياني",
    graph_empty: "الرسم البياني فارغ. أضف عقدًا وعلاقات ثم شغّل المحاكاة.",
    create_node: "إنشاء عقدة",
    node_type: "نوع العقدة",
    first_name: "الاسم الأول",
    last_name: "اسم العائلة",
    company_name: "اسم الشركة",
    company_industry: "القطاع",
    company_description: "الوصف",
    bio: "نبذة",
    profile_text: "نص الملف",
    employment_text: "نص الوظائف",
    employment_json: "JSON الوظائف",
    attributes_json: "JSON السمات",
    state_json: "JSON الحالة",
    create_node_btn: "إنشاء عقدة",
    create_edge: "إنشاء علاقة",
    from_node: "من عقدة",
    to_node: "إلى عقدة",
    relation: "العلاقة",
    weight: "الوزن",
    direction: "الاتجاه",
    logic_rule: "قاعدة منطقية",
    directed: "موجّه",
    undirected: "غير موجّه",
    create_edge_btn: "إنشاء علاقة",
    persistence: "الحفظ",
    persist_snapshot: "حفظ اللقطة",
    load_snapshot: "تحميل اللقطة",
    simulation: "المحاكاة",
    seed_ids: "معرّفات البدء",
    depth: "العمق",
    steps: "الخطوات",
    damping: "التخميد",
    activation: "التفعيل",
    infer_rounds: "جولات الاستدلال",
    run_simulation: "تشغيل المحاكاة",
    last_simulation_output: "آخر خرج للمحاكاة",
    event_feedback: "تغذية راجعة للحدث",
    event_id: "معرّف الحدث",
    reward: "المكافأة",
    learning_rate: "معدل التعلم",
    apply_reward: "تطبيق المكافأة",
    batch_relation_reinforcement: "تعزيز العلاقات بشكل دفعي",
    reinforce_relation: "تعزيز العلاقة",
    event_stream: "دفق الأحداث",
    snapshot_nodes: "عقد اللقطة",
    snapshot_edges: "حواف اللقطة",
    project_modules: "وحدات المشروع",
    files: "الملفات",
    show_files: "عرض الملفات",
    log_system: "النظام",
    log_error: "خطأ",
    log_workspace_refreshed: "تم تحديث المساحة",
    log_action_complete: "اكتمل",
    log_node_created: "تم إنشاء عقدة",
    log_profile_imported: "تم استيراد الملف عبر LLM",
    error_invalid_json_payload: "JSON غير صالح",
  },
  hi: {
    app_title: "स्वायत्त ग्राफ वर्कस्पेस",
    app_subtitle: "ग्राफ-आधारित मोड: नोड, संबंध, सिमुलेशन और विश्लेषण।",
    ui_language: "इंटरफ़ेस भाषा",
    action_refresh: "रिफ्रेश",
    action_seed_demo: "डेमो देखें",
    action_clear: "साफ़ करें",
    runtime_error_title: "फ़्रंटएंड रनटाइम त्रुटि",
    runtime_error_hint: "ब्राउज़र कंसोल खोलें और नीचे की त्रुटि ठीक करें।",
    action_try_continue: "जारी रखने की कोशिश करें",
    action_reload_data: "डेटा पुनः लोड करें",
    page_overview: "ओवरव्यू",
    page_builder: "बिल्डर",
    page_simulation: "सिमुलेशन",
    page_data: "डेटा",
    llm_profile_builder: "LLM प्रोफ़ाइल बिल्डर",
    profile_input_text: "वर्णनात्मक टेक्स्ट",
    entity_type_hint: "एंटिटी प्रकार संकेत",
    entity_type_human: "मानव",
    entity_type_company: "कंपनी",
    entity_type_technology: "तकनीक",
    entity_type_generic: "सामान्य",
    extract_profile_graph: "प्रोफ़ाइल निकालें -> ग्राफ बनाएं",
    profile_result: "LLM प्रोफ़ाइल परिणाम",
    profile_prompt_preview: "प्रॉम्प्ट टेम्पलेट",
    model_missing_hint: "GGUF मॉडल को ./models में रखें और पुनः प्रयास करें।",
    workspace_status: "वर्कस्पेस स्थिति",
    execution_log: "एक्ज़ीक्यूशन लॉग",
    graph_visualization: "ग्राफ विज़ुअलाइज़ेशन",
    graph_empty: "ग्राफ खाली है। नोड और संबंध जोड़ें, फिर सिमुलेशन चलाएँ।",
    create_node: "नोड बनाएं",
    node_type: "नोड प्रकार",
    first_name: "पहला नाम",
    last_name: "अंतिम नाम",
    company_name: "कंपनी का नाम",
    company_industry: "उद्योग",
    company_description: "विवरण",
    bio: "बायो",
    profile_text: "प्रोफ़ाइल टेक्स्ट",
    employment_text: "रोज़गार टेक्स्ट",
    employment_json: "रोज़गार JSON",
    attributes_json: "एट्रिब्यूट्स JSON",
    state_json: "स्टेट JSON",
    create_node_btn: "नोड बनाएं",
    create_edge: "एज बनाएं",
    from_node: "से नोड",
    to_node: "तक नोड",
    relation: "संबंध",
    weight: "वज़न",
    direction: "दिशा",
    logic_rule: "लॉजिक नियम",
    directed: "दिशात्मक",
    undirected: "अदिशात्मक",
    create_edge_btn: "एज बनाएं",
    persistence: "पर्सिस्टेंस",
    persist_snapshot: "स्नैपशॉट सेव करें",
    load_snapshot: "स्नैपशॉट लोड करें",
    simulation: "सिमुलेशन",
    seed_ids: "सीड आईडी",
    depth: "गहराई",
    steps: "स्टेप्स",
    damping: "डैम्पिंग",
    activation: "एक्टिवेशन",
    infer_rounds: "इन्फर राउंड्स",
    run_simulation: "सिमुलेशन चलाएँ",
    last_simulation_output: "आख़िरी सिमुलेशन आउटपुट",
    event_feedback: "इवेंट फीडबैक",
    event_id: "इवेंट आईडी",
    reward: "रिवार्ड",
    learning_rate: "लर्निंग रेट",
    apply_reward: "रिवार्ड लागू करें",
    batch_relation_reinforcement: "बैच संबंध सुदृढ़ीकरण",
    reinforce_relation: "संबंध मजबूत करें",
    event_stream: "इवेंट स्ट्रीम",
    snapshot_nodes: "स्नैपशॉट नोड्स",
    snapshot_edges: "स्नैपशॉट एजेस",
    project_modules: "प्रोजेक्ट मॉड्यूल्स",
    files: "फाइलें",
    show_files: "फाइलें दिखाएँ",
    log_system: "सिस्टम",
    log_error: "त्रुटि",
    log_workspace_refreshed: "वर्कस्पेस रिफ्रेश हुआ",
    log_action_complete: "पूरा",
    log_node_created: "नोड बनाया गया",
    log_profile_imported: "प्रोफ़ाइल LLM से आयात हुई",
    error_invalid_json_payload: "अमान्य JSON",
  },
  ja: {
    app_title: "自律グラフワークスペース",
    app_subtitle: "グラフ中心モード: ノード、関係、シミュレーション、分析。",
    ui_language: "UI言語",
    action_refresh: "更新",
    action_seed_demo: "デモを見る",
    action_clear: "クリア",
    runtime_error_title: "フロントエンド実行エラー",
    runtime_error_hint: "ブラウザのコンソールを開き、下のエラーを修正してください。",
    action_try_continue: "続行を試す",
    action_reload_data: "データ再読み込み",
    page_overview: "概要",
    page_builder: "ビルダー",
    page_simulation: "シミュレーション",
    page_data: "データ",
    llm_profile_builder: "LLMプロファイルビルダー",
    profile_input_text: "記述テキスト",
    entity_type_hint: "エンティティ種別",
    entity_type_human: "人",
    entity_type_company: "会社",
    entity_type_technology: "技術",
    entity_type_generic: "汎用",
    extract_profile_graph: "プロファイル抽出 -> グラフ構築",
    profile_result: "LLMプロファイル結果",
    profile_prompt_preview: "プロンプトテンプレート",
    model_missing_hint: "GGUFモデルを ./models に置いて再試行してください。",
    workspace_status: "ワークスペース状態",
    execution_log: "実行ログ",
    graph_visualization: "グラフ可視化",
    graph_empty: "グラフが空です。ノードと関係を追加してからシミュレーションを実行してください。",
    create_node: "ノード作成",
    node_type: "ノード種別",
    first_name: "名",
    last_name: "姓",
    company_name: "会社名",
    company_industry: "業界",
    company_description: "説明",
    bio: "紹介",
    profile_text: "プロファイルテキスト",
    employment_text: "職歴テキスト",
    employment_json: "職歴 JSON",
    attributes_json: "属性 JSON",
    state_json: "状態 JSON",
    create_node_btn: "ノード作成",
    create_edge: "エッジ作成",
    from_node: "開始ノード",
    to_node: "終了ノード",
    relation: "関係",
    weight: "重み",
    direction: "方向",
    logic_rule: "論理ルール",
    directed: "有向",
    undirected: "無向",
    create_edge_btn: "エッジ作成",
    persistence: "永続化",
    persist_snapshot: "スナップショット保存",
    load_snapshot: "スナップショット読み込み",
    simulation: "シミュレーション",
    seed_ids: "シードID",
    depth: "深さ",
    steps: "ステップ",
    damping: "減衰",
    activation: "活性化",
    infer_rounds: "推論ラウンド",
    run_simulation: "シミュレーション実行",
    last_simulation_output: "直近シミュレーション出力",
    event_feedback: "イベントフィードバック",
    event_id: "イベントID",
    reward: "報酬",
    learning_rate: "学習率",
    apply_reward: "報酬を適用",
    batch_relation_reinforcement: "関係のバッチ強化",
    reinforce_relation: "関係を強化",
    event_stream: "イベントストリーム",
    snapshot_nodes: "スナップショットノード",
    snapshot_edges: "スナップショットエッジ",
    project_modules: "プロジェクトモジュール",
    files: "ファイル",
    show_files: "ファイルを表示",
    log_system: "SYSTEM",
    log_error: "ERROR",
    log_workspace_refreshed: "ワークスペースを更新しました",
    log_action_complete: "完了",
    log_node_created: "ノードを作成しました",
    log_profile_imported: "LLMからプロファイルを取り込みました",
    error_invalid_json_payload: "不正なJSON",
  },
};

const EXTRA_TRANSLATION_EXTENSIONS = {
  hy: {
    branch_visual_toolkit: "Ճյուղի վիզուալ գործիքակազմ",
    branch_visual_mode_cards: "Քարտեր",
    branch_visual_mode_charts: "Գրաֆիկներ",
    branch_visual_mode_lists: "Ցանկեր",
    branch_visual_mode_tips: "Լայֆհաքեր",
    branch_scope: "Շրջանակ",
    branch_scope_selected: "Ընտրված ճյուղ",
    branch_scope_global: "Գլոբալ գրաֆ",
    branch_metric_nodes: "Հանգույցներ",
    branch_metric_edges: "Կապեր",
    branch_metric_avg_weight: "Կապի միջին քաշ",
    branch_metric_hints: "Գործողության հուշումներ",
    branch_top_relations: "Թոփ կապեր",
    branch_top_node_types: "Թոփ հանգույցների տեսակներ",
    branch_top_nodes: "Թոփ հանգույցներ",
    branch_top_edges: "Թոփ կապեր",
    branch_lifehacks: "Գործնական լայֆհաքեր",
    style_nodes_title: "Style Nodes (անվտանգ UI sandbox)",
    style_nodes_safe_hint: "Փոխվում են միայն տեսողական token-ները։ Reset-ը միշտ վերադարձնում է base տեսքը։",
    style_nodes_slider_label: "Style node սլայդեր",
    style_nodes_reset: "Վերականգնել base-ին",
    style_nodes_activate: "Ակտիվացնել",
    style_nodes_active: "Ակտիվ",
    reasoning_path: "Մտածողության շղթա",
    reasoning_path_empty: "Ընտրիր հանգույց՝ տեսնելու reasoning path-ը և dependency փակումը։",
    reasoning_roots: "Արմատներ",
    reasoning_chain: "Շղթա",
    reasoning_prerequisites: "Նախապայմաններ",
    reasoning_dependents: "Կախյալներ",
    graph_hover_dependencies_hint: "Տար հանգույցի վրա՝ dependency-ները ընդգծելու համար։",
    reasoning_trace_options: "Trace տարբերակներ",
    reasoning_trace_empty: "Այլընտրանքային trace չի գտնվել։",
    reasoning_trace_score: "գնահատական",
    edge_reasoning: "Կապի բացատրություն",
    edge_reasoning_empty: "Ընտրիր կապ՝ հասկանալու համար ինչու է այն գոյություն ունի։",
    edge_reasoning_relation: "Կապի տեսակ",
    edge_reasoning_logic: "Տրամաբանություն",
    edge_reasoning_direction: "Ուղղություն",
    edge_reasoning_strength: "Ուժ",
    edge_reasoning_facts: "Հիմնական փաստեր",
    edge_history_timeline: "Կապի փոփոխությունների թայմլայն",
    edge_history_empty: "Ընթացիկ stream պատուհանում այս կապի փոփոխության իրադարձություններ չկան։",
    edge_history_reason: "Պատճառ",
    edge_history_weight: "Քաշ",
    edge_history_logic: "Տրամաբանություն",
    edge_history_event: "Իրադարձություն",
    style_nodes_name: "Ստիլի անուն",
    style_nodes_description: "Ստիլի նկարագրություն",
    style_nodes_vars_json: "Ստիլի փոփոխականների JSON",
    style_nodes_save: "Պահպանել style node-ը",
    style_nodes_saved_result: "Style node-ի պահպանման արդյունք",
    branch_report_editor_title: "Branch report node խմբագիր",
    branch_report_summary: "Հաշվետվության ամփոփում",
    branch_report_tips: "Հաշվետվության հուշումներ",
    branch_report_save: "Պահպանել branch report node-ը",
    branch_report_saved_result: "Branch report-ի պահպանման արդյունք",
    action_style_node_save: "style node պահպանում",
    action_branch_report_save: "branch report պահպանում",
    error_style_vars_json_invalid: "Ստիլի փոփոխականների JSON-ը սխալ է",
  },
  zh: {
    branch_visual_toolkit: "分支可视化工具包",
    branch_visual_mode_cards: "卡片",
    branch_visual_mode_charts: "图表",
    branch_visual_mode_lists: "列表",
    branch_visual_mode_tips: "技巧",
    branch_scope: "范围",
    branch_scope_selected: "选中分支",
    branch_scope_global: "全局图谱",
    branch_metric_nodes: "节点",
    branch_metric_edges: "边",
    branch_metric_avg_weight: "平均边权",
    branch_metric_hints: "行动提示",
    branch_top_relations: "关系排行",
    branch_top_node_types: "节点类型排行",
    branch_top_nodes: "节点排行",
    branch_top_edges: "边排行",
    branch_lifehacks: "实用建议",
    style_nodes_title: "样式节点（安全 UI 沙盒）",
    style_nodes_safe_hint: "仅允许修改视觉变量。重置可恢复默认样式。",
    style_nodes_slider_label: "样式节点滑块",
    style_nodes_reset: "重置为默认",
    style_nodes_activate: "启用",
    style_nodes_active: "已启用",
    reasoning_path: "推理路径",
    reasoning_path_empty: "选择一个节点以查看其推理路径与依赖闭包。",
    reasoning_roots: "根节点",
    reasoning_chain: "链路",
    reasoning_prerequisites: "前置依赖",
    reasoning_dependents: "后续依赖",
    graph_hover_dependencies_hint: "将鼠标悬停在节点上可高亮其依赖闭包。",
    reasoning_trace_options: "路径候选",
    reasoning_trace_empty: "未找到可选路径。",
    reasoning_trace_score: "评分",
    edge_reasoning: "边解释",
    edge_reasoning_empty: "选择一条边以查看其存在原因。",
    edge_reasoning_relation: "关系",
    edge_reasoning_logic: "逻辑",
    edge_reasoning_direction: "方向",
    edge_reasoning_strength: "强度",
    edge_reasoning_facts: "关键事实",
    edge_history_timeline: "边变更时间线",
    edge_history_empty: "当前事件窗口中没有该边的变更事件。",
    edge_history_reason: "原因",
    edge_history_weight: "权重",
    edge_history_logic: "逻辑",
    edge_history_event: "事件",
    style_nodes_name: "样式名称",
    style_nodes_description: "样式说明",
    style_nodes_vars_json: "样式变量 JSON",
    style_nodes_save: "保存样式节点",
    style_nodes_saved_result: "样式节点保存结果",
    branch_report_editor_title: "分支报告节点编辑器",
    branch_report_summary: "报告摘要",
    branch_report_tips: "报告提示",
    branch_report_save: "保存分支报告节点",
    branch_report_saved_result: "分支报告保存结果",
    action_style_node_save: "保存样式节点",
    action_branch_report_save: "保存分支报告",
    error_style_vars_json_invalid: "样式变量 JSON 无效",
  },
  es: {
    branch_visual_toolkit: "Kit visual de rama",
    branch_visual_mode_cards: "Tarjetas",
    branch_visual_mode_charts: "Gráficos",
    branch_visual_mode_lists: "Listas",
    branch_visual_mode_tips: "Trucos",
    branch_scope: "Ámbito",
    branch_scope_selected: "Rama seleccionada",
    branch_scope_global: "Grafo global",
    branch_metric_nodes: "Nodos",
    branch_metric_edges: "Relaciones",
    branch_metric_avg_weight: "Peso medio de relación",
    branch_metric_hints: "Sugerencias de acción",
    branch_top_relations: "Top de relaciones",
    branch_top_node_types: "Top de tipos de nodo",
    branch_top_nodes: "Top de nodos",
    branch_top_edges: "Top de relaciones",
    branch_lifehacks: "Consejos prácticos",
    style_nodes_title: "Nodos de estilo (sandbox UI seguro)",
    style_nodes_safe_hint: "Solo cambian tokens visuales. Restablecer siempre vuelve al estilo base.",
    style_nodes_slider_label: "Slider de nodo de estilo",
    style_nodes_reset: "Restablecer base",
    style_nodes_activate: "Activar",
    style_nodes_active: "Activo",
    reasoning_path: "Ruta de razonamiento",
    reasoning_path_empty: "Selecciona un nodo para ver su ruta de razonamiento y cierre de dependencias.",
    reasoning_roots: "Raíces",
    reasoning_chain: "Cadena",
    reasoning_prerequisites: "Prerequisitos",
    reasoning_dependents: "Dependientes",
    graph_hover_dependencies_hint: "Pasa el cursor sobre un nodo para resaltar su cierre de dependencias.",
    reasoning_trace_options: "Variantes de traza",
    reasoning_trace_empty: "No se encontraron trazas alternativas.",
    reasoning_trace_score: "puntuación",
    edge_reasoning: "Explicación de relación",
    edge_reasoning_empty: "Selecciona una relación para inspeccionar por qué existe.",
    edge_reasoning_relation: "Relación",
    edge_reasoning_logic: "Lógica",
    edge_reasoning_direction: "Dirección",
    edge_reasoning_strength: "Fuerza",
    edge_reasoning_facts: "Hechos clave",
    edge_history_timeline: "Timeline de cambios de la relación",
    edge_history_empty: "No hay eventos de cambio para esta relación en la ventana actual.",
    edge_history_reason: "Motivo",
    edge_history_weight: "Peso",
    edge_history_logic: "Lógica",
    edge_history_event: "Evento",
    style_nodes_name: "Nombre del estilo",
    style_nodes_description: "Descripción del estilo",
    style_nodes_vars_json: "JSON de variables de estilo",
    style_nodes_save: "Guardar nodo de estilo",
    style_nodes_saved_result: "Resultado de guardado del nodo de estilo",
    branch_report_editor_title: "Editor de nodo de informe de rama",
    branch_report_summary: "Resumen del informe",
    branch_report_tips: "Consejos del informe",
    branch_report_save: "Guardar nodo de informe de rama",
    branch_report_saved_result: "Resultado de guardado del informe de rama",
    action_style_node_save: "guardar nodo de estilo",
    action_branch_report_save: "guardar informe de rama",
    error_style_vars_json_invalid: "JSON de variables de estilo inválido",
  },
  pt: {
    branch_visual_toolkit: "Kit visual da branch",
    branch_visual_mode_cards: "Cartões",
    branch_visual_mode_charts: "Gráficos",
    branch_visual_mode_lists: "Listas",
    branch_visual_mode_tips: "Dicas",
    branch_scope: "Escopo",
    branch_scope_selected: "Branch selecionada",
    branch_scope_global: "Grafo global",
    branch_metric_nodes: "Nós",
    branch_metric_edges: "Arestas",
    branch_metric_avg_weight: "Peso médio da aresta",
    branch_metric_hints: "Dicas de ação",
    branch_top_relations: "Top relações",
    branch_top_node_types: "Top tipos de nó",
    branch_top_nodes: "Top nós",
    branch_top_edges: "Top arestas",
    branch_lifehacks: "Dicas práticas",
    style_nodes_title: "Nós de estilo (sandbox UI seguro)",
    style_nodes_safe_hint: "Apenas tokens visuais podem mudar. Reset sempre restaura o padrão.",
    style_nodes_slider_label: "Slider de nó de estilo",
    style_nodes_reset: "Resetar para base",
    style_nodes_activate: "Ativar",
    style_nodes_active: "Ativo",
    reasoning_path: "Trilha de raciocínio",
    reasoning_path_empty: "Selecione um nó para inspecionar sua trilha de raciocínio e fechamento de dependências.",
    reasoning_roots: "Raízes",
    reasoning_chain: "Cadeia",
    reasoning_prerequisites: "Pré-requisitos",
    reasoning_dependents: "Dependentes",
    graph_hover_dependencies_hint: "Passe o mouse em um nó para destacar seu fechamento de dependências.",
    reasoning_trace_options: "Variantes de trilha",
    reasoning_trace_empty: "Nenhuma trilha alternativa encontrada.",
    reasoning_trace_score: "pontuação",
    edge_reasoning: "Explicação da aresta",
    edge_reasoning_empty: "Selecione uma aresta para inspecionar por que ela existe.",
    edge_reasoning_relation: "Relação",
    edge_reasoning_logic: "Lógica",
    edge_reasoning_direction: "Direção",
    edge_reasoning_strength: "Força",
    edge_reasoning_facts: "Fatos-chave",
    edge_history_timeline: "Timeline de mudanças da aresta",
    edge_history_empty: "Não há eventos de mudança para esta aresta na janela atual.",
    edge_history_reason: "Motivo",
    edge_history_weight: "Peso",
    edge_history_logic: "Lógica",
    edge_history_event: "Evento",
    style_nodes_name: "Nome do estilo",
    style_nodes_description: "Descrição do estilo",
    style_nodes_vars_json: "JSON de variáveis de estilo",
    style_nodes_save: "Salvar nó de estilo",
    style_nodes_saved_result: "Resultado do salvamento do nó de estilo",
    branch_report_editor_title: "Editor de nó de relatório de branch",
    branch_report_summary: "Resumo do relatório",
    branch_report_tips: "Dicas do relatório",
    branch_report_save: "Salvar nó de relatório de branch",
    branch_report_saved_result: "Resultado do salvamento do relatório de branch",
    action_style_node_save: "salvar nó de estilo",
    action_branch_report_save: "salvar relatório de branch",
    error_style_vars_json_invalid: "JSON de variáveis de estilo inválido",
  },
  fr: {
    branch_visual_toolkit: "Boîte visuelle de branche",
    branch_visual_mode_cards: "Cartes",
    branch_visual_mode_charts: "Graphiques",
    branch_visual_mode_lists: "Listes",
    branch_visual_mode_tips: "Astuces",
    branch_scope: "Portée",
    branch_scope_selected: "Branche sélectionnée",
    branch_scope_global: "Graphe global",
    branch_metric_nodes: "Nœuds",
    branch_metric_edges: "Arêtes",
    branch_metric_avg_weight: "Poids moyen des arêtes",
    branch_metric_hints: "Indications d'action",
    branch_top_relations: "Relations principales",
    branch_top_node_types: "Types de nœuds principaux",
    branch_top_nodes: "Nœuds principaux",
    branch_top_edges: "Arêtes principales",
    branch_lifehacks: "Astuces pratiques",
    style_nodes_title: "Nœuds de style (sandbox UI sécurisé)",
    style_nodes_safe_hint: "Seuls les tokens visuels changent. Reset restaure toujours le style de base.",
    style_nodes_slider_label: "Curseur de nœud de style",
    style_nodes_reset: "Réinitialiser au style de base",
    style_nodes_activate: "Activer",
    style_nodes_active: "Actif",
    overview_sections: "Sections d'ensemble",
    overview_section_status: "Statut",
    overview_section_demo: "Démo",
    overview_section_daily: "Journal",
    overview_section_user_graph: "Graphe utilisateur",
    overview_section_autoruns: "Autoruns",
    overview_section_graph: "Graphe",
    overview_section_editor: "Éditeur",
    overview_section_client: "Client",
    overview_section_advisors: "Conseillers",
    pager_prev: "Préc.",
    pager_next: "Suiv.",
    pager_events: "Événements",
    pager_nodes: "Nœuds",
    pager_edges: "Arêtes",
    pager_modules: "Modules",
    demo_narrative: "Narratif démo",
    scenario: "Scénario",
    refresh_client_profile: "Rafraîchir mon profil client",
    daily_mode: "Mode journal",
    daily_journal: "Journal",
    run_daily_analysis: "Lancer l'analyse du jour",
    daily_recommendations_scores: "Recommandations du jour + scores",
    user_semantic_graph: "Graphe sémantique utilisateur",
    user_graph_narrative: "Texte narratif du profil",
    user_fears: "Peurs",
    user_desires: "Désirs",
    user_goals: "Objectifs",
    user_principles: "Principes",
    user_opportunities: "Opportunités",
    user_abilities: "Capacités",
    user_access: "Accès",
    user_knowledge: "Connaissances",
    user_assets: "Actifs",
    apply_user_graph: "Appliquer le graphe utilisateur",
    user_graph_update_result: "Résultat de mise à jour du graphe utilisateur",
    autoruns_import_title: "Import Sysinternals Autoruns",
    autoruns_input_or_query: "Autoruns CSV/TSV ou requête",
    import_autoruns: "Importer Autoruns",
    autoruns_import_result: "Résultat d'import Autoruns",
    selected_node_editor: "Éditeur du nœud sélectionné",
    selected_edge_editor: "Éditeur de l'arête sélectionnée",
    metadata_json: "JSON des métadonnées",
    update_node: "Mettre à jour le nœud",
    delete_node: "Supprimer le nœud",
    update_edge: "Mettre à jour l'arête",
    delete_edge: "Supprimer l'arête",
    client_profile_semantic_input: "Mon profil client (entrée sémantique)",
    mini_coders_advisors: "Mini codeurs / conseillers",
    prompt_catalog: "Catalogue de prompts",
    sql_table_schema_json: "Schéma SQL des tables (JSON)",
    simulation_timeline: "Chronologie de simulation",
    timeline_idle: "Aucune simulation lancée pour l'instant",
    timeline_in_progress: "En cours",
    timeline_completed: "Terminé",
    timeline_progress: "Progression",
    llm_role_debate: "Débat de rôles LLM",
    debate_prompt: "Sujet du débat",
    debate_variants: "Hypothèses",
    debate_attach_graph: "Attacher les branches au graphe",
    debate_run: "Lancer le débat",
    debate_result: "Résultat du débat",
    error_debate_prompt_empty: "Le sujet du débat est vide",
    action_llm_debate: "débat de rôles llm",
    reasoning_path: "Chemin de raisonnement",
    reasoning_path_empty: "Sélectionnez un nœud pour inspecter son chemin de raisonnement et sa clôture de dépendances.",
    reasoning_roots: "Racines",
    reasoning_chain: "Chaîne",
    reasoning_prerequisites: "Prérequis",
    reasoning_dependents: "Dépendants",
    graph_hover_dependencies_hint: "Survolez un nœud pour mettre en évidence sa clôture de dépendances.",
    reasoning_trace_options: "Variantes de trace",
    reasoning_trace_empty: "Aucune trace alternative trouvée.",
    reasoning_trace_score: "score",
    edge_reasoning: "Raisonnement de l'arête",
    edge_reasoning_empty: "Sélectionnez une arête pour voir pourquoi elle existe.",
    edge_reasoning_relation: "Relation",
    edge_reasoning_logic: "Logique",
    edge_reasoning_direction: "Direction",
    edge_reasoning_strength: "Force",
    edge_reasoning_facts: "Faits clés",
    edge_history_timeline: "Chronologie des changements d'arête",
    edge_history_empty: "Aucun événement de changement pour cette arête dans la fenêtre actuelle.",
    edge_history_reason: "Raison",
    edge_history_weight: "Poids",
    edge_history_logic: "Logique",
    edge_history_event: "Événement",
    style_nodes_name: "Nom du style",
    style_nodes_description: "Description du style",
    style_nodes_vars_json: "JSON des variables de style",
    style_nodes_save: "Enregistrer le nœud de style",
    style_nodes_saved_result: "Résultat d'enregistrement du nœud de style",
    branch_report_editor_title: "Éditeur de nœud de rapport de branche",
    branch_report_summary: "Résumé du rapport",
    branch_report_tips: "Conseils du rapport",
    branch_report_save: "Enregistrer le nœud de rapport de branche",
    branch_report_saved_result: "Résultat d'enregistrement du rapport de branche",
    action_style_node_save: "enregistrement du nœud de style",
    action_branch_report_save: "enregistrement du rapport de branche",
    error_style_vars_json_invalid: "JSON des variables de style invalide",
  },
  ar: {
    branch_visual_toolkit: "مجموعة التصور للفرع",
    branch_visual_mode_cards: "بطاقات",
    branch_visual_mode_charts: "مخططات",
    branch_visual_mode_lists: "قوائم",
    branch_visual_mode_tips: "نصائح",
    branch_scope: "النطاق",
    branch_scope_selected: "الفرع المحدد",
    branch_scope_global: "الرسم العام",
    branch_metric_nodes: "العُقد",
    branch_metric_edges: "الحواف",
    branch_metric_avg_weight: "متوسط وزن الحواف",
    branch_metric_hints: "تلميحات إجراء",
    branch_top_relations: "أهم العلاقات",
    branch_top_node_types: "أهم أنواع العُقد",
    branch_top_nodes: "أهم العُقد",
    branch_top_edges: "أهم الحواف",
    branch_lifehacks: "نصائح عملية",
    style_nodes_title: "عقد الأنماط (بيئة UI آمنة)",
    style_nodes_safe_hint: "يمكن تغيير المتغيرات البصرية فقط. إعادة الضبط تعيد النمط الأساسي دائمًا.",
    style_nodes_slider_label: "شريط عقدة النمط",
    style_nodes_reset: "إعادة إلى الأساسي",
    style_nodes_activate: "تفعيل",
    style_nodes_active: "نشط",
    overview_sections: "أقسام النظرة العامة",
    overview_section_status: "الحالة",
    overview_section_demo: "عرض",
    overview_section_daily: "اليومي",
    overview_section_user_graph: "رسم المستخدم",
    overview_section_autoruns: "Autoruns",
    overview_section_graph: "الرسم",
    overview_section_editor: "المحرر",
    overview_section_client: "العميل",
    overview_section_advisors: "المستشارون",
    pager_prev: "السابق",
    pager_next: "التالي",
    pager_events: "الأحداث",
    pager_nodes: "العُقد",
    pager_edges: "الحواف",
    pager_modules: "الوحدات",
    demo_narrative: "سرد تجريبي",
    scenario: "السيناريو",
    refresh_client_profile: "تحديث ملف العميل الخاص بي",
    daily_mode: "الوضع اليومي",
    daily_journal: "اليومية",
    run_daily_analysis: "تشغيل التحليل اليومي",
    daily_recommendations_scores: "توصيات يومية + درجات",
    user_semantic_graph: "الرسم الدلالي للمستخدم",
    user_graph_narrative: "نص الملف السردي",
    user_fears: "المخاوف",
    user_desires: "الرغبات",
    user_goals: "الأهداف",
    user_principles: "المبادئ",
    user_opportunities: "الفرص",
    user_abilities: "القدرات",
    user_access: "الوصول",
    user_knowledge: "المعرفة",
    user_assets: "الأصول",
    apply_user_graph: "تطبيق رسم المستخدم",
    user_graph_update_result: "نتيجة تحديث رسم المستخدم",
    autoruns_import_title: "استيراد Sysinternals Autoruns",
    autoruns_input_or_query: "Autoruns CSV/TSV أو استعلام",
    import_autoruns: "استيراد Autoruns",
    autoruns_import_result: "نتيجة استيراد Autoruns",
    selected_node_editor: "محرر العقدة المحددة",
    selected_edge_editor: "محرر الحافة المحددة",
    metadata_json: "JSON البيانات الوصفية",
    update_node: "تحديث العقدة",
    delete_node: "حذف العقدة",
    update_edge: "تحديث الحافة",
    delete_edge: "حذف الحافة",
    client_profile_semantic_input: "ملفي كعميل (إدخال دلالي)",
    mini_coders_advisors: "مطورون مصغّرون / مستشارون",
    prompt_catalog: "كتالوج البرومبتات",
    sql_table_schema_json: "مخطط جداول SQL (JSON)",
    simulation_timeline: "الخط الزمني للمحاكاة",
    timeline_idle: "لا توجد محاكاة جارية بعد",
    timeline_in_progress: "قيد التنفيذ",
    timeline_completed: "مكتمل",
    timeline_progress: "التقدم",
    llm_role_debate: "مناظرة أدوار LLM",
    debate_prompt: "موضوع المناظرة",
    debate_variants: "الفرضيات",
    debate_attach_graph: "إرفاق الفروع بالرسم",
    debate_run: "تشغيل المناظرة",
    debate_result: "نتيجة المناظرة",
    error_debate_prompt_empty: "موضوع المناظرة فارغ",
    action_llm_debate: "مناظرة أدوار llm",
    reasoning_path: "مسار الاستدلال",
    reasoning_path_empty: "اختر عقدة لفحص مسار الاستدلال وإغلاق التبعيات.",
    reasoning_roots: "الجذور",
    reasoning_chain: "السلسلة",
    reasoning_prerequisites: "المتطلبات",
    reasoning_dependents: "التوابع",
    graph_hover_dependencies_hint: "مرّر فوق أي عقدة لإبراز إغلاق تبعياتها.",
    reasoning_trace_options: "خيارات التتبع",
    reasoning_trace_empty: "لم يتم العثور على مسارات بديلة.",
    reasoning_trace_score: "الدرجة",
    edge_reasoning: "تفسير الحافة",
    edge_reasoning_empty: "اختر حافة لمعرفة سبب وجودها.",
    edge_reasoning_relation: "العلاقة",
    edge_reasoning_logic: "المنطق",
    edge_reasoning_direction: "الاتجاه",
    edge_reasoning_strength: "القوة",
    edge_reasoning_facts: "حقائق أساسية",
    edge_history_timeline: "الخط الزمني لتغييرات الحافة",
    edge_history_empty: "لا توجد أحداث تغيير لهذه الحافة في نافذة البث الحالية.",
    edge_history_reason: "السبب",
    edge_history_weight: "الوزن",
    edge_history_logic: "المنطق",
    edge_history_event: "الحدث",
    style_nodes_name: "اسم النمط",
    style_nodes_description: "وصف النمط",
    style_nodes_vars_json: "JSON متغيرات النمط",
    style_nodes_save: "حفظ عقدة النمط",
    style_nodes_saved_result: "نتيجة حفظ عقدة النمط",
    branch_report_editor_title: "محرر عقدة تقرير الفرع",
    branch_report_summary: "ملخص التقرير",
    branch_report_tips: "نصائح التقرير",
    branch_report_save: "حفظ عقدة تقرير الفرع",
    branch_report_saved_result: "نتيجة حفظ تقرير الفرع",
    action_style_node_save: "حفظ عقدة النمط",
    action_branch_report_save: "حفظ تقرير الفرع",
    error_style_vars_json_invalid: "JSON متغيرات النمط غير صالح",
  },
  hi: {
    branch_visual_toolkit: "ब्रांच विज़ुअल टूलकिट",
    branch_visual_mode_cards: "कार्ड्स",
    branch_visual_mode_charts: "चार्ट्स",
    branch_visual_mode_lists: "सूचियाँ",
    branch_visual_mode_tips: "लाइफहैक्स",
    branch_scope: "स्कोप",
    branch_scope_selected: "चयनित ब्रांच",
    branch_scope_global: "ग्लोबल ग्राफ",
    branch_metric_nodes: "नोड्स",
    branch_metric_edges: "एजेस",
    branch_metric_avg_weight: "औसत एज वज़न",
    branch_metric_hints: "एक्शन हिंट्स",
    branch_top_relations: "शीर्ष संबंध",
    branch_top_node_types: "शीर्ष नोड प्रकार",
    branch_top_nodes: "शीर्ष नोड्स",
    branch_top_edges: "शीर्ष एजेस",
    branch_lifehacks: "व्यावहारिक लाइफहैक्स",
    style_nodes_title: "स्टाइल नोड्स (सुरक्षित UI सैंडबॉक्स)",
    style_nodes_safe_hint: "सिर्फ विज़ुअल टोकन बदलेंगे। रीसेट हमेशा बेस स्टाइल लौटाता है।",
    style_nodes_slider_label: "स्टाइल नोड स्लाइडर",
    style_nodes_reset: "बेस पर रीसेट",
    style_nodes_activate: "सक्रिय करें",
    style_nodes_active: "सक्रिय",
    overview_sections: "ओवरव्यू सेक्शन",
    overview_section_status: "स्थिति",
    overview_section_demo: "डेमो",
    overview_section_daily: "दैनिक",
    overview_section_user_graph: "यूज़र ग्राफ",
    overview_section_autoruns: "Autoruns",
    overview_section_graph: "ग्राफ",
    overview_section_editor: "एडिटर",
    overview_section_client: "क्लाइंट",
    overview_section_advisors: "सलाहकार",
    pager_prev: "पिछला",
    pager_next: "अगला",
    pager_events: "इवेंट्स",
    pager_nodes: "नोड्स",
    pager_edges: "एजेस",
    pager_modules: "मॉड्यूल्स",
    demo_narrative: "डेमो नैरेटिव",
    scenario: "परिदृश्य",
    refresh_client_profile: "मेरा क्लाइंट प्रोफ़ाइल रीफ़्रेश करें",
    daily_mode: "दैनिक मोड",
    daily_journal: "जर्नल",
    run_daily_analysis: "दैनिक विश्लेषण चलाएँ",
    daily_recommendations_scores: "दैनिक सिफारिशें + स्कोर",
    user_semantic_graph: "यूज़र सिमेंटिक ग्राफ",
    user_graph_narrative: "प्रोफ़ाइल नैरेटिव टेक्स्ट",
    user_fears: "डर",
    user_desires: "इच्छाएँ",
    user_goals: "लक्ष्य",
    user_principles: "सिद्धांत",
    user_opportunities: "अवसर",
    user_abilities: "क्षमताएँ",
    user_access: "एक्सेस",
    user_knowledge: "ज्ञान",
    user_assets: "संसाधन",
    apply_user_graph: "यूज़र ग्राफ लागू करें",
    user_graph_update_result: "यूज़र ग्राफ अपडेट परिणाम",
    autoruns_import_title: "Sysinternals Autoruns इम्पोर्ट",
    autoruns_input_or_query: "Autoruns CSV/TSV या क्वेरी",
    import_autoruns: "Autoruns इम्पोर्ट करें",
    autoruns_import_result: "Autoruns इम्पोर्ट परिणाम",
    selected_node_editor: "चयनित नोड संपादक",
    selected_edge_editor: "चयनित एज संपादक",
    metadata_json: "मेटाडेटा JSON",
    update_node: "नोड अपडेट करें",
    delete_node: "नोड हटाएँ",
    update_edge: "एज अपडेट करें",
    delete_edge: "एज हटाएँ",
    client_profile_semantic_input: "मेरा क्लाइंट प्रोफ़ाइल (सिमेंटिक इनपुट)",
    mini_coders_advisors: "मिनी कोडर्स / सलाहकार",
    prompt_catalog: "प्रॉम्प्ट कैटलॉग",
    sql_table_schema_json: "SQL टेबल स्कीमा (JSON)",
    simulation_timeline: "सिमुलेशन टाइमलाइन",
    timeline_idle: "अभी तक कोई सिमुलेशन नहीं चला",
    timeline_in_progress: "प्रगति पर",
    timeline_completed: "पूर्ण",
    timeline_progress: "प्रगति",
    llm_role_debate: "LLM रोल डिबेट",
    debate_prompt: "डिबेट विषय",
    debate_variants: "हाइपोथीसिस",
    debate_attach_graph: "ब्रांच को ग्राफ से जोड़ें",
    debate_run: "डिबेट चलाएँ",
    debate_result: "डिबेट परिणाम",
    error_debate_prompt_empty: "डिबेट विषय खाली है",
    action_llm_debate: "llm रोल डिबेट",
    reasoning_path: "रीज़निंग पाथ",
    reasoning_path_empty: "किसी नोड का चयन करें ताकि उसका reasoning path और dependency closure देखा जा सके।",
    reasoning_roots: "रूट्स",
    reasoning_chain: "चेन",
    reasoning_prerequisites: "पूर्वापेक्षाएँ",
    reasoning_dependents: "निर्भर",
    graph_hover_dependencies_hint: "किसी नोड पर होवर करें ताकि उसकी dependency closure हाइलाइट हो।",
    reasoning_trace_options: "ट्रेस विकल्प",
    reasoning_trace_empty: "कोई वैकल्पिक ट्रेस नहीं मिला।",
    reasoning_trace_score: "स्कोर",
    edge_reasoning: "एज रीज़निंग",
    edge_reasoning_empty: "कोई एज चुनें ताकि समझ सकें वह क्यों मौजूद है।",
    edge_reasoning_relation: "संबंध",
    edge_reasoning_logic: "लॉजिक",
    edge_reasoning_direction: "दिशा",
    edge_reasoning_strength: "ताकत",
    edge_reasoning_facts: "मुख्य तथ्य",
    edge_history_timeline: "एज परिवर्तन टाइमलाइन",
    edge_history_empty: "वर्तमान स्ट्रीम विंडो में इस एज के लिए कोई परिवर्तन इवेंट नहीं हैं।",
    edge_history_reason: "कारण",
    edge_history_weight: "वज़न",
    edge_history_logic: "लॉजिक",
    edge_history_event: "इवेंट",
    style_nodes_name: "स्टाइल नाम",
    style_nodes_description: "स्टाइल विवरण",
    style_nodes_vars_json: "स्टाइल वेरिएबल्स JSON",
    style_nodes_save: "स्टाइल नोड सहेजें",
    style_nodes_saved_result: "स्टाइल नोड सेव परिणाम",
    branch_report_editor_title: "ब्रांच रिपोर्ट नोड संपादक",
    branch_report_summary: "रिपोर्ट सारांश",
    branch_report_tips: "रिपोर्ट टिप्स",
    branch_report_save: "ब्रांच रिपोर्ट नोड सहेजें",
    branch_report_saved_result: "ब्रांच रिपोर्ट सेव परिणाम",
    action_style_node_save: "स्टाइल नोड सहेजना",
    action_branch_report_save: "ब्रांच रिपोर्ट सहेजना",
    error_style_vars_json_invalid: "स्टाइल वेरिएबल्स JSON अमान्य है",
  },
  ja: {
    branch_visual_toolkit: "ブランチ可視化ツールキット",
    branch_visual_mode_cards: "カード",
    branch_visual_mode_charts: "チャート",
    branch_visual_mode_lists: "リスト",
    branch_visual_mode_tips: "ライフハック",
    branch_scope: "スコープ",
    branch_scope_selected: "選択中のブランチ",
    branch_scope_global: "グローバルグラフ",
    branch_metric_nodes: "ノード",
    branch_metric_edges: "エッジ",
    branch_metric_avg_weight: "平均エッジ重み",
    branch_metric_hints: "アクションヒント",
    branch_top_relations: "主要関係",
    branch_top_node_types: "主要ノードタイプ",
    branch_top_nodes: "主要ノード",
    branch_top_edges: "主要エッジ",
    branch_lifehacks: "実践ライフハック",
    style_nodes_title: "スタイルノード（安全UIサンドボックス）",
    style_nodes_safe_hint: "変更できるのは視覚トークンのみ。リセットで常に基本スタイルへ戻せます。",
    style_nodes_slider_label: "スタイルノードスライダー",
    style_nodes_reset: "基本にリセット",
    style_nodes_activate: "有効化",
    style_nodes_active: "有効",
    overview_sections: "概要セクション",
    overview_section_status: "状態",
    overview_section_demo: "デモ",
    overview_section_daily: "日次",
    overview_section_user_graph: "ユーザーグラフ",
    overview_section_autoruns: "Autoruns",
    overview_section_graph: "グラフ",
    overview_section_editor: "エディタ",
    overview_section_client: "クライアント",
    overview_section_advisors: "アドバイザー",
    pager_prev: "前へ",
    pager_next: "次へ",
    pager_events: "イベント",
    pager_nodes: "ノード",
    pager_edges: "エッジ",
    pager_modules: "モジュール",
    demo_narrative: "デモの物語",
    scenario: "シナリオ",
    refresh_client_profile: "クライアントプロファイルを更新",
    daily_mode: "日次モード",
    daily_journal: "ジャーナル",
    run_daily_analysis: "日次分析を実行",
    daily_recommendations_scores: "日次提案 + スコア",
    user_semantic_graph: "ユーザー意味グラフ",
    user_graph_narrative: "プロファイル記述テキスト",
    user_fears: "不安",
    user_desires: "望み",
    user_goals: "目標",
    user_principles: "原則",
    user_opportunities: "機会",
    user_abilities: "能力",
    user_access: "アクセス",
    user_knowledge: "知識",
    user_assets: "資産",
    apply_user_graph: "ユーザーグラフを適用",
    user_graph_update_result: "ユーザーグラフ更新結果",
    autoruns_import_title: "Sysinternals Autoruns 取込",
    autoruns_input_or_query: "Autoruns CSV/TSV またはクエリ",
    import_autoruns: "Autoruns を取り込む",
    autoruns_import_result: "Autoruns 取込結果",
    selected_node_editor: "選択ノード編集",
    selected_edge_editor: "選択エッジ編集",
    metadata_json: "メタデータ JSON",
    update_node: "ノード更新",
    delete_node: "ノード削除",
    update_edge: "エッジ更新",
    delete_edge: "エッジ削除",
    client_profile_semantic_input: "私のクライアントプロファイル（意味入力）",
    mini_coders_advisors: "ミニコーダー / アドバイザー",
    prompt_catalog: "プロンプトカタログ",
    sql_table_schema_json: "SQLテーブルスキーマ (JSON)",
    simulation_timeline: "シミュレーションタイムライン",
    timeline_idle: "まだシミュレーションは実行されていません",
    timeline_in_progress: "進行中",
    timeline_completed: "完了",
    timeline_progress: "進捗",
    llm_role_debate: "LLMロール討論",
    debate_prompt: "討論テーマ",
    debate_variants: "仮説",
    debate_attach_graph: "分岐をグラフに保存",
    debate_run: "討論を実行",
    debate_result: "討論結果",
    error_debate_prompt_empty: "討論テーマが空です",
    action_llm_debate: "llm ロール討論",
    reasoning_path: "推論パス",
    reasoning_path_empty: "ノードを選択すると、その推論パスと依存関係クロージャを表示します。",
    reasoning_roots: "ルート",
    reasoning_chain: "チェーン",
    reasoning_prerequisites: "前提",
    reasoning_dependents: "依存先",
    graph_hover_dependencies_hint: "ノードにホバーすると依存関係クロージャを強調表示します。",
    reasoning_trace_options: "トレース候補",
    reasoning_trace_empty: "代替トレースは見つかりませんでした。",
    reasoning_trace_score: "スコア",
    edge_reasoning: "エッジの説明",
    edge_reasoning_empty: "エッジを選択すると存在理由を確認できます。",
    edge_reasoning_relation: "関係",
    edge_reasoning_logic: "ロジック",
    edge_reasoning_direction: "方向",
    edge_reasoning_strength: "強さ",
    edge_reasoning_facts: "主要な事実",
    edge_history_timeline: "エッジ変更タイムライン",
    edge_history_empty: "現在のストリーム範囲では、このエッジの変更イベントはありません。",
    edge_history_reason: "理由",
    edge_history_weight: "重み",
    edge_history_logic: "ロジック",
    edge_history_event: "イベント",
    style_nodes_name: "スタイル名",
    style_nodes_description: "スタイル説明",
    style_nodes_vars_json: "スタイル変数 JSON",
    style_nodes_save: "スタイルノードを保存",
    style_nodes_saved_result: "スタイルノード保存結果",
    branch_report_editor_title: "ブランチレポートノード編集",
    branch_report_summary: "レポート要約",
    branch_report_tips: "レポートのヒント",
    branch_report_save: "ブランチレポートノードを保存",
    branch_report_saved_result: "ブランチレポート保存結果",
    action_style_node_save: "スタイルノード保存",
    action_branch_report_save: "ブランチレポート保存",
    error_style_vars_json_invalid: "スタイル変数 JSON が不正です",
  },
};

const MULTITOOL_UI_TRANSLATIONS = {
  en: {
    overview_section_multitool: "Multitool",
    multitool_title: "Personal Multitool Workspace",
    multitool_subtitle:
      "Capture improvements, preferences, tasks, and risks with domain + legislation branching in your graph memory.",
    multitool_section_requests: "Improvement Requests",
    multitool_section_preferences: "Preferences And Anti-Preferences",
    multitool_section_tasks: "Task Board",
    multitool_section_risks: "Risk Analysis",
    multitool_choose_existing: "Existing Node",
    multitool_request_title: "Request Title",
    multitool_request_details: "Request Details",
    multitool_request_output: "Desired Output",
    multitool_request_layout: "Preferred View",
    multitool_request_status: "Status",
    multitool_request_priority: "Priority",
    multitool_request_result: "Request Save Result",
    multitool_preferences_profile_name: "Profile Name",
    multitool_preferences_likes: "Likes",
    multitool_preferences_dislikes: "Dislikes",
    multitool_preferences_style: "Style Preferences",
    multitool_preferences_tools: "Preferred Tools",
    multitool_preferences_notes: "Memory Notes",
    multitool_preferences_result: "Preference Save Result",
    multitool_task_title: "Task Title",
    multitool_task_description: "Task Description",
    multitool_task_status: "Task Status",
    multitool_task_priority: "Task Priority",
    multitool_task_due: "Due Date",
    multitool_task_result: "Task Save Result",
    multitool_risk_title: "Risk Title",
    multitool_risk_description: "Risk Description",
    multitool_risk_probability: "Probability",
    multitool_risk_impact: "Impact",
    multitool_risk_mitigation: "Mitigation Steps",
    multitool_risk_result: "Risk Save Result",
    multitool_domain: "Domain Branch",
    multitool_country: "Legislation Country",
    multitool_save: "Save",
    multitool_new_item: "New",
    multitool_dashboard: "Multitool Dashboard",
    multitool_chart_task_status: "Task Status Distribution",
    multitool_chart_task_priority: "Task Priority Distribution",
    multitool_chart_risk_probability: "Risk Probability Distribution",
    multitool_chart_risk_impact: "Risk Impact Distribution",
    multitool_chart_domain_coverage: "Domain Coverage",
    multitool_open_tasks: "Open Tasks",
    multitool_top_risks: "Top Risks",
    multitool_widget_contradictions: "Top Contradictions",
    multitool_widget_quality_trend: "Quality Score Trend",
    multitool_widget_backup_history: "Backup History",
    multitool_no_items: "No data yet.",
    action_multitool_request_save: "save improvement request",
    action_multitool_preference_save: "save preferences profile",
    action_multitool_task_save: "save task item",
    action_multitool_risk_save: "save risk item",
    error_multitool_request_title_empty: "Request title is empty",
    error_multitool_task_title_empty: "Task title is empty",
    error_multitool_risk_title_empty: "Risk title is empty",
    personal_tree_title: "Personal Informative Tree",
    personal_tree_subtitle:
      "Build a thought tree from articles, laws and notes. Keep sources, links and concise takeaways in one branch.",
    personal_tree_topic: "Thought Topic",
    personal_tree_title_field: "Session Title",
    personal_tree_text: "Source Text",
    personal_tree_source_type: "Source Type",
    personal_tree_source_title: "Source Title",
    personal_tree_source_url: "Source URL",
    personal_tree_max_points: "Key Points",
    personal_tree_ingest_action: "Build Thought Tree",
    personal_tree_extraction_result: "Extraction Result",
    personal_tree_note_title: "Quick Notes",
    personal_tree_note_title_field: "Note Title",
    personal_tree_note_text: "Note Text",
    personal_tree_note_tags: "Tags",
    personal_tree_note_links: "Links",
    personal_tree_note_save_action: "Save Note To Tree",
    personal_tree_note_result: "Note Save Result",
    personal_tree_small_window_title: "Mini Tree Window",
    personal_tree_refresh_tree: "Refresh Tree",
    personal_tree_view_result: "Tree View Result",
    personal_tree_no_tree: "No personal tree yet.",
    personal_tree_sources: "Sources",
    personal_tree_notes: "Notes",
    action_personal_tree_ingest: "personal tree ingest",
    action_personal_tree_note_save: "personal tree note save",
    action_personal_tree_view: "personal tree view",
    error_personal_tree_text_empty: "Source text is empty",
    error_personal_tree_note_empty: "Note title and text are empty",
    multitool_ops_title: "Automation And Analysis Hub",
    multitool_ops_subtitle:
      "Run package hygiene, namespace memory routing, graph-RAG, contradiction scans, risk boards, timeline replay, policy control, quality checks and backups.",
    packages_title: "Packages / Trash Manager",
    packages_name: "Package Name",
    packages_items: "Items (one per line)",
    packages_restore_ids: "Restore Node IDs",
    packages_model_role: "Manager Role",
    packages_model_path: "Manager Model Path",
    packages_apply_changes: "Apply Changes",
    packages_confirmation: "Confirmation Token",
    packages_list: "List",
    packages_store: "Store",
    packages_purge: "Purge",
    packages_restore: "Restore",
    packages_result: "Packages Result",
    memory_namespace_title: "Memory Namespace Router",
    memory_namespace: "Target Namespace",
    memory_source_namespace: "Source Namespace",
    memory_scope: "Scope",
    memory_query: "Filter Query",
    memory_node_ids: "Node IDs",
    memory_min_score: "Min Match Score",
    memory_apply: "Apply Namespace",
    memory_view: "View Namespaces",
    memory_result: "Memory Result",
    rag_title: "Graph RAG",
    rag_query: "RAG Query",
    rag_top_k: "Top K",
    rag_use_llm: "Use LLM Answer Synthesis",
    rag_run: "Run RAG",
    rag_result: "RAG Result",
    contradiction_title: "Contradiction Scanner",
    contradiction_run: "Run Scan",
    contradiction_apply_graph: "Attach Scan To Graph",
    contradiction_result: "Contradiction Result",
    task_risk_title: "Task-Risk Board Generator",
    task_risk_tasks: "Tasks (line format: title | details)",
    task_risk_run: "Build Task-Risk Board",
    task_risk_result: "Task-Risk Result",
    timeline_replay_title: "Timeline Replay",
    timeline_event_type: "Event Type Filter",
    timeline_limit: "Timeline Limit",
    timeline_from_ts: "From Timestamp",
    timeline_to_ts: "To Timestamp",
    timeline_run: "Replay Timeline",
    timeline_result: "Timeline Result",
    policy_title: "LLM Policy Layer",
    policy_mode: "Policy Mode",
    policy_trusted_sessions: "Trusted Sessions",
    policy_trusted_users: "Trusted Users",
    policy_allowed_actions: "Allowlisted Actions",
    policy_merge_lists: "Merge Lists",
    policy_load: "Load Policy",
    policy_save: "Save Policy",
    policy_result: "Policy Result",
    quality_title: "Quality Harness",
    quality_queries: "Sample Queries",
    quality_run: "Run Quality Harness",
    quality_result: "Quality Result",
    backup_title: "Backup / Restore / Audit",
    backup_label: "Backup Label",
    backup_include_events: "Include Events",
    backup_event_limit: "Event Limit",
    backup_create: "Create Backup",
    backup_latest: "Use Latest Backup",
    backup_path: "Backup Path",
    backup_restore: "Restore Backup",
    backup_restore_policy: "Restore Saved Policy",
    backup_result: "Backup Result",
    audit_load: "Load Audit",
    audit_limit: "Audit Limit",
    audit_include_backups: "Include Backup Files",
    audit_result: "Audit Result",
    action_packages_list: "packages list",
    action_packages_store: "packages store",
    action_packages_purge: "packages purge",
    action_packages_restore: "packages restore",
    action_memory_apply: "memory namespace apply",
    action_memory_view: "memory namespace view",
    action_graph_rag: "graph rag query",
    action_contradiction_scan: "contradiction scan",
    action_task_risk_board: "task risk board",
    action_timeline_replay: "timeline replay",
    action_policy_load: "llm policy load",
    action_policy_save: "llm policy save",
    action_quality_harness: "quality harness",
    action_backup_create: "backup create",
    action_backup_restore: "backup restore",
    action_audit_load: "audit logs load",
    error_packages_items_empty: "Items are empty",
    error_rag_query_empty: "RAG query is empty",
    error_task_risk_tasks_empty: "Task list is empty",
  },
  ru: {
    overview_section_multitool: "Мультитул",
    multitool_title: "Персональный мультитул",
    multitool_subtitle:
      "Фиксируйте улучшения, предпочтения, задачи и риски с ветвлением по доменам и странам законодательства.",
    multitool_section_requests: "Запросы на улучшения",
    multitool_section_preferences: "Предпочтения и антипредпочтения",
    multitool_section_tasks: "Задачник",
    multitool_section_risks: "Анализ рисков",
    multitool_choose_existing: "Существующий узел",
    multitool_request_title: "Название запроса",
    multitool_request_details: "Описание запроса",
    multitool_request_output: "Желаемый формат результата",
    multitool_request_layout: "Предпочтительный вид",
    multitool_request_status: "Статус",
    multitool_request_priority: "Приоритет",
    multitool_request_result: "Результат сохранения запроса",
    multitool_preferences_profile_name: "Имя профиля",
    multitool_preferences_likes: "Нравится",
    multitool_preferences_dislikes: "Не нравится",
    multitool_preferences_style: "Предпочтения по стилю",
    multitool_preferences_tools: "Предпочтительные инструменты",
    multitool_preferences_notes: "Заметки памяти",
    multitool_preferences_result: "Результат сохранения предпочтений",
    multitool_task_title: "Название задачи",
    multitool_task_description: "Описание задачи",
    multitool_task_status: "Статус задачи",
    multitool_task_priority: "Приоритет задачи",
    multitool_task_due: "Срок",
    multitool_task_result: "Результат сохранения задачи",
    multitool_risk_title: "Название риска",
    multitool_risk_description: "Описание риска",
    multitool_risk_probability: "Вероятность",
    multitool_risk_impact: "Влияние",
    multitool_risk_mitigation: "Шаги снижения",
    multitool_risk_result: "Результат сохранения риска",
    multitool_domain: "Домен",
    multitool_country: "Страна законодательства",
    multitool_save: "Сохранить",
    multitool_new_item: "Новый",
    multitool_dashboard: "Панель мультитула",
    multitool_chart_task_status: "Распределение статусов задач",
    multitool_chart_task_priority: "Распределение приоритетов задач",
    multitool_chart_risk_probability: "Распределение вероятностей рисков",
    multitool_chart_risk_impact: "Распределение влияния рисков",
    multitool_chart_domain_coverage: "Покрытие доменов",
    multitool_open_tasks: "Открытые задачи",
    multitool_top_risks: "Ключевые риски",
    multitool_no_items: "Пока нет данных.",
    action_multitool_request_save: "сохранение запроса на улучшение",
    action_multitool_preference_save: "сохранение профиля предпочтений",
    action_multitool_task_save: "сохранение задачи",
    action_multitool_risk_save: "сохранение риска",
    error_multitool_request_title_empty: "Пустое название запроса",
    error_multitool_task_title_empty: "Пустое название задачи",
    error_multitool_risk_title_empty: "Пустое название риска",
    personal_tree_title: "Персональное инфо-дерево",
    personal_tree_subtitle:
      "Строй дерево мыслей по статьям, законам и заметкам. Сохраняй источники, ссылки и выжимки в одной ветке.",
    personal_tree_topic: "Тема размышления",
    personal_tree_title_field: "Название сессии",
    personal_tree_text: "Текст источника",
    personal_tree_source_type: "Тип источника",
    personal_tree_source_title: "Название источника",
    personal_tree_source_url: "Ссылка на источник",
    personal_tree_max_points: "Ключевые пункты",
    personal_tree_ingest_action: "Построить дерево мыслей",
    personal_tree_extraction_result: "Результат выжимки",
    personal_tree_note_title: "Быстрые заметки",
    personal_tree_note_title_field: "Название заметки",
    personal_tree_note_text: "Текст заметки",
    personal_tree_note_tags: "Теги",
    personal_tree_note_links: "Ссылки",
    personal_tree_note_save_action: "Сохранить заметку в дерево",
    personal_tree_note_result: "Результат сохранения заметки",
    personal_tree_small_window_title: "Мини-окно дерева",
    personal_tree_refresh_tree: "Обновить дерево",
    personal_tree_view_result: "Результат просмотра дерева",
    personal_tree_no_tree: "Персональное дерево пока пусто.",
    personal_tree_sources: "Источники",
    personal_tree_notes: "Заметки",
    action_personal_tree_ingest: "ингест в персональное дерево",
    action_personal_tree_note_save: "сохранение заметки в дерево",
    action_personal_tree_view: "просмотр персонального дерева",
    error_personal_tree_text_empty: "Пустой текст источника",
    error_personal_tree_note_empty: "Пустые название и текст заметки",
  },
  hy: {
    overview_section_multitool: "Մուլտիթուլ",
    multitool_title: "Անհատական մուլտիթուլ",
    multitool_subtitle:
      "Պահպանիր բարելավման հարցումներ, նախասիրություններ, խնդիրներ և ռիսկեր՝ դոմենային ու օրենսդրական ճյուղավորմամբ։",
    multitool_section_requests: "Բարելավման հարցումներ",
    multitool_section_preferences: "Սիրելի և չսիրելի ձևաչափեր",
    multitool_section_tasks: "Խնդրագիր",
    multitool_section_risks: "Ռիսկերի վերլուծություն",
    multitool_choose_existing: "Գոյություն ունեցող հանգույց",
    multitool_request_title: "Հարցման վերնագիր",
    multitool_request_details: "Հարցման մանրամասներ",
    multitool_request_output: "Ցանկալի արդյունք",
    multitool_request_layout: "Նախընտրելի տեսք",
    multitool_request_status: "Կարգավիճակ",
    multitool_request_priority: "Առաջնահերթություն",
    multitool_request_result: "Հարցման պահպանման արդյունք",
    multitool_preferences_profile_name: "Պրոֆիլի անուն",
    multitool_preferences_likes: "Սիրում եմ",
    multitool_preferences_dislikes: "Չեմ սիրում",
    multitool_preferences_style: "Սթայլի նախասիրություններ",
    multitool_preferences_tools: "Նախընտրելի գործիքներ",
    multitool_preferences_notes: "Հիշողության նշումներ",
    multitool_preferences_result: "Նախասիրությունների պահպանման արդյունք",
    multitool_task_title: "Խնդրի վերնագիր",
    multitool_task_description: "Խնդրի նկարագրություն",
    multitool_task_status: "Խնդրի կարգավիճակ",
    multitool_task_priority: "Խնդրի առաջնահերթություն",
    multitool_task_due: "Ժամկետ",
    multitool_task_result: "Խնդրի պահպանման արդյունք",
    multitool_risk_title: "Ռիսկի վերնագիր",
    multitool_risk_description: "Ռիսկի նկարագրություն",
    multitool_risk_probability: "Հավանականություն",
    multitool_risk_impact: "Ազդեցություն",
    multitool_risk_mitigation: "Մեղմման քայլեր",
    multitool_risk_result: "Ռիսկի պահպանման արդյունք",
    multitool_domain: "Դոմենային ճյուղ",
    multitool_country: "Օրենսդրության երկիր",
    multitool_save: "Պահպանել",
    multitool_new_item: "Նոր",
    multitool_dashboard: "Մուլտիթուլի վահանակ",
    multitool_chart_task_status: "Խնդիրների կարգավիճակների բաշխում",
    multitool_chart_task_priority: "Խնդիրների առաջնահերթությունների բաշխում",
    multitool_chart_risk_probability: "Ռիսկերի հավանականությունների բաշխում",
    multitool_chart_risk_impact: "Ռիսկերի ազդեցությունների բաշխում",
    multitool_chart_domain_coverage: "Դոմենների ծածկույթ",
    multitool_open_tasks: "Բաց խնդիրներ",
    multitool_top_risks: "Գլխավոր ռիսկեր",
    multitool_no_items: "Տվյալներ դեռ չկան։",
    action_multitool_request_save: "բարելավման հարցման պահպանում",
    action_multitool_preference_save: "նախասիրությունների պրոֆիլի պահպանում",
    action_multitool_task_save: "խնդրի պահպանում",
    action_multitool_risk_save: "ռիսկի պահպանում",
    error_multitool_request_title_empty: "Հարցման վերնագիրը դատարկ է",
    error_multitool_task_title_empty: "Խնդրի վերնագիրը դատարկ է",
    error_multitool_risk_title_empty: "Ռիսկի վերնագիրը դատարկ է",
  },
  fr: {
    overview_section_multitool: "Outil perso",
    multitool_title: "Espace multitool personnel",
    multitool_subtitle:
      "Conservez demandes d'amélioration, préférences, tâches et risques avec branches métier + pays législatifs.",
    multitool_section_requests: "Demandes d'amélioration",
    multitool_section_preferences: "Préférences et anti-préférences",
    multitool_section_tasks: "Gestion des tâches",
    multitool_section_risks: "Analyse des risques",
    multitool_choose_existing: "Nœud existant",
    multitool_request_title: "Titre de la demande",
    multitool_request_details: "Détails",
    multitool_request_output: "Résultat souhaité",
    multitool_request_layout: "Vue préférée",
    multitool_request_status: "Statut",
    multitool_request_priority: "Priorité",
    multitool_request_result: "Résultat de sauvegarde",
    multitool_preferences_profile_name: "Nom du profil",
    multitool_preferences_likes: "J'aime",
    multitool_preferences_dislikes: "Je n'aime pas",
    multitool_preferences_style: "Préférences de style",
    multitool_preferences_tools: "Outils préférés",
    multitool_preferences_notes: "Notes mémoire",
    multitool_preferences_result: "Résultat des préférences",
    multitool_task_title: "Titre de la tâche",
    multitool_task_description: "Description",
    multitool_task_status: "Statut de tâche",
    multitool_task_priority: "Priorité de tâche",
    multitool_task_due: "Échéance",
    multitool_task_result: "Résultat de tâche",
    multitool_risk_title: "Titre du risque",
    multitool_risk_description: "Description du risque",
    multitool_risk_probability: "Probabilité",
    multitool_risk_impact: "Impact",
    multitool_risk_mitigation: "Étapes de mitigation",
    multitool_risk_result: "Résultat du risque",
    multitool_domain: "Domaine",
    multitool_country: "Pays de législation",
    multitool_save: "Enregistrer",
    multitool_new_item: "Nouveau",
    multitool_dashboard: "Tableau multitool",
    multitool_chart_task_status: "Distribution des statuts de tâches",
    multitool_chart_task_priority: "Distribution des priorités de tâches",
    multitool_chart_risk_probability: "Distribution des probabilités de risque",
    multitool_chart_risk_impact: "Distribution des impacts de risque",
    multitool_chart_domain_coverage: "Couverture des domaines",
    multitool_open_tasks: "Tâches ouvertes",
    multitool_top_risks: "Risques principaux",
    multitool_no_items: "Pas encore de données.",
    action_multitool_request_save: "sauvegarde de demande d'amélioration",
    action_multitool_preference_save: "sauvegarde du profil de préférences",
    action_multitool_task_save: "sauvegarde de tâche",
    action_multitool_risk_save: "sauvegarde de risque",
    error_multitool_request_title_empty: "Le titre de la demande est vide",
    error_multitool_task_title_empty: "Le titre de la tâche est vide",
    error_multitool_risk_title_empty: "Le titre du risque est vide",
  },
  es: {
    overview_section_multitool: "Multiherramienta",
    multitool_title: "Espacio personal multiherramienta",
    multitool_subtitle:
      "Guarda mejoras, preferencias, tareas y riesgos con ramas por dominio y legislación por país.",
    multitool_section_requests: "Solicitudes de mejora",
    multitool_section_preferences: "Preferencias y no-preferencias",
    multitool_section_tasks: "Gestor de tareas",
    multitool_section_risks: "Análisis de riesgos",
    multitool_choose_existing: "Nodo existente",
    multitool_request_title: "Título de solicitud",
    multitool_request_details: "Detalles",
    multitool_request_output: "Resultado deseado",
    multitool_request_layout: "Vista preferida",
    multitool_request_status: "Estado",
    multitool_request_priority: "Prioridad",
    multitool_request_result: "Resultado de guardado",
    multitool_preferences_profile_name: "Nombre del perfil",
    multitool_preferences_likes: "Me gusta",
    multitool_preferences_dislikes: "No me gusta",
    multitool_preferences_style: "Preferencias de estilo",
    multitool_preferences_tools: "Herramientas preferidas",
    multitool_preferences_notes: "Notas de memoria",
    multitool_preferences_result: "Resultado de preferencias",
    multitool_task_title: "Título de tarea",
    multitool_task_description: "Descripción de tarea",
    multitool_task_status: "Estado de tarea",
    multitool_task_priority: "Prioridad de tarea",
    multitool_task_due: "Fecha límite",
    multitool_task_result: "Resultado de tarea",
    multitool_risk_title: "Título de riesgo",
    multitool_risk_description: "Descripción de riesgo",
    multitool_risk_probability: "Probabilidad",
    multitool_risk_impact: "Impacto",
    multitool_risk_mitigation: "Pasos de mitigación",
    multitool_risk_result: "Resultado de riesgo",
    multitool_domain: "Dominio",
    multitool_country: "País de legislación",
    multitool_save: "Guardar",
    multitool_new_item: "Nuevo",
    multitool_dashboard: "Panel multiherramienta",
    multitool_chart_task_status: "Distribución de estados de tareas",
    multitool_chart_task_priority: "Distribución de prioridades de tareas",
    multitool_chart_risk_probability: "Distribución de probabilidad de riesgos",
    multitool_chart_risk_impact: "Distribución de impacto de riesgos",
    multitool_chart_domain_coverage: "Cobertura por dominio",
    multitool_open_tasks: "Tareas abiertas",
    multitool_top_risks: "Riesgos principales",
    multitool_no_items: "Aún no hay datos.",
    action_multitool_request_save: "guardar solicitud de mejora",
    action_multitool_preference_save: "guardar perfil de preferencias",
    action_multitool_task_save: "guardar tarea",
    action_multitool_risk_save: "guardar riesgo",
    error_multitool_request_title_empty: "El título de solicitud está vacío",
    error_multitool_task_title_empty: "El título de tarea está vacío",
    error_multitool_risk_title_empty: "El título de riesgo está vacío",
  },
  pt: {
    overview_section_multitool: "Multiferramenta",
    multitool_title: "Workspace pessoal multiferramenta",
    multitool_subtitle:
      "Salve melhorias, preferências, tarefas e riscos com ramificação por domínio e legislação por país.",
    multitool_section_requests: "Pedidos de melhoria",
    multitool_section_preferences: "Preferências e antipreferências",
    multitool_section_tasks: "Gestor de tarefas",
    multitool_section_risks: "Análise de riscos",
    multitool_choose_existing: "Nó existente",
    multitool_request_title: "Título do pedido",
    multitool_request_details: "Detalhes",
    multitool_request_output: "Saída desejada",
    multitool_request_layout: "Visual preferido",
    multitool_request_status: "Status",
    multitool_request_priority: "Prioridade",
    multitool_request_result: "Resultado de salvamento",
    multitool_preferences_profile_name: "Nome do perfil",
    multitool_preferences_likes: "Curtidas",
    multitool_preferences_dislikes: "Não gosto",
    multitool_preferences_style: "Preferências de estilo",
    multitool_preferences_tools: "Ferramentas preferidas",
    multitool_preferences_notes: "Notas de memória",
    multitool_preferences_result: "Resultado de preferências",
    multitool_task_title: "Título da tarefa",
    multitool_task_description: "Descrição da tarefa",
    multitool_task_status: "Status da tarefa",
    multitool_task_priority: "Prioridade da tarefa",
    multitool_task_due: "Prazo",
    multitool_task_result: "Resultado da tarefa",
    multitool_risk_title: "Título do risco",
    multitool_risk_description: "Descrição do risco",
    multitool_risk_probability: "Probabilidade",
    multitool_risk_impact: "Impacto",
    multitool_risk_mitigation: "Passos de mitigação",
    multitool_risk_result: "Resultado do risco",
    multitool_domain: "Domínio",
    multitool_country: "País de legislação",
    multitool_save: "Salvar",
    multitool_new_item: "Novo",
    multitool_dashboard: "Painel multiferramenta",
    multitool_chart_task_status: "Distribuição de status de tarefas",
    multitool_chart_task_priority: "Distribuição de prioridade de tarefas",
    multitool_chart_risk_probability: "Distribuição de probabilidade de riscos",
    multitool_chart_risk_impact: "Distribuição de impacto de riscos",
    multitool_chart_domain_coverage: "Cobertura de domínios",
    multitool_open_tasks: "Tarefas abertas",
    multitool_top_risks: "Principais riscos",
    multitool_no_items: "Ainda sem dados.",
    action_multitool_request_save: "salvar pedido de melhoria",
    action_multitool_preference_save: "salvar perfil de preferências",
    action_multitool_task_save: "salvar tarefa",
    action_multitool_risk_save: "salvar risco",
    error_multitool_request_title_empty: "O título do pedido está vazio",
    error_multitool_task_title_empty: "O título da tarefa está vazio",
    error_multitool_risk_title_empty: "O título do risco está vazio",
  },
  ar: {
    overview_section_multitool: "أداة متعددة",
    multitool_title: "مساحة أداة شخصية متعددة",
    multitool_subtitle:
      "احفظ طلبات التحسين والتفضيلات والمهام والمخاطر مع تفرعات المجال والتشريعات حسب الدولة.",
    multitool_section_requests: "طلبات التحسين",
    multitool_section_preferences: "التفضيلات وما لا تفضله",
    multitool_section_tasks: "إدارة المهام",
    multitool_section_risks: "تحليل المخاطر",
    multitool_choose_existing: "عقدة موجودة",
    multitool_request_title: "عنوان الطلب",
    multitool_request_details: "تفاصيل الطلب",
    multitool_request_output: "المخرجات المطلوبة",
    multitool_request_layout: "العرض المفضل",
    multitool_request_status: "الحالة",
    multitool_request_priority: "الأولوية",
    multitool_request_result: "نتيجة حفظ الطلب",
    multitool_preferences_profile_name: "اسم الملف الشخصي",
    multitool_preferences_likes: "ما تفضله",
    multitool_preferences_dislikes: "ما لا تفضله",
    multitool_preferences_style: "تفضيلات النمط",
    multitool_preferences_tools: "الأدوات المفضلة",
    multitool_preferences_notes: "ملاحظات الذاكرة",
    multitool_preferences_result: "نتيجة حفظ التفضيلات",
    multitool_task_title: "عنوان المهمة",
    multitool_task_description: "وصف المهمة",
    multitool_task_status: "حالة المهمة",
    multitool_task_priority: "أولوية المهمة",
    multitool_task_due: "تاريخ الاستحقاق",
    multitool_task_result: "نتيجة حفظ المهمة",
    multitool_risk_title: "عنوان الخطر",
    multitool_risk_description: "وصف الخطر",
    multitool_risk_probability: "الاحتمال",
    multitool_risk_impact: "التأثير",
    multitool_risk_mitigation: "خطوات التخفيف",
    multitool_risk_result: "نتيجة حفظ الخطر",
    multitool_domain: "فرع المجال",
    multitool_country: "دولة التشريع",
    multitool_save: "حفظ",
    multitool_new_item: "جديد",
    multitool_dashboard: "لوحة الأداة المتعددة",
    multitool_chart_task_status: "توزيع حالات المهام",
    multitool_chart_task_priority: "توزيع أولويات المهام",
    multitool_chart_risk_probability: "توزيع احتمال المخاطر",
    multitool_chart_risk_impact: "توزيع تأثير المخاطر",
    multitool_chart_domain_coverage: "تغطية المجالات",
    multitool_open_tasks: "المهام المفتوحة",
    multitool_top_risks: "أعلى المخاطر",
    multitool_no_items: "لا توجد بيانات بعد.",
    action_multitool_request_save: "حفظ طلب التحسين",
    action_multitool_preference_save: "حفظ ملف التفضيلات",
    action_multitool_task_save: "حفظ المهمة",
    action_multitool_risk_save: "حفظ الخطر",
    error_multitool_request_title_empty: "عنوان الطلب فارغ",
    error_multitool_task_title_empty: "عنوان المهمة فارغ",
    error_multitool_risk_title_empty: "عنوان الخطر فارغ",
  },
  hi: {
    overview_section_multitool: "मल्टीटूल",
    multitool_title: "पर्सनल मल्टीटूल वर्कस्पेस",
    multitool_subtitle:
      "सुधार अनुरोध, पसंद/नापसंद, कार्य और जोखिम को डोमेन व देश-आधारित विधिक शाखाओं के साथ सहेजें।",
    multitool_section_requests: "सुधार अनुरोध",
    multitool_section_preferences: "पसंद और नापसंद",
    multitool_section_tasks: "टास्क बोर्ड",
    multitool_section_risks: "जोखिम विश्लेषण",
    multitool_choose_existing: "मौजूदा नोड",
    multitool_request_title: "अनुरोध शीर्षक",
    multitool_request_details: "अनुरोध विवरण",
    multitool_request_output: "वांछित आउटपुट",
    multitool_request_layout: "पसंदीदा दृश्य",
    multitool_request_status: "स्थिति",
    multitool_request_priority: "प्राथमिकता",
    multitool_request_result: "अनुरोध सेव परिणाम",
    multitool_preferences_profile_name: "प्रोफ़ाइल नाम",
    multitool_preferences_likes: "पसंद",
    multitool_preferences_dislikes: "नापसंद",
    multitool_preferences_style: "स्टाइल प्राथमिकताएँ",
    multitool_preferences_tools: "पसंदीदा टूल्स",
    multitool_preferences_notes: "मेमोरी नोट्स",
    multitool_preferences_result: "प्राथमिकता सेव परिणाम",
    multitool_task_title: "टास्क शीर्षक",
    multitool_task_description: "टास्क विवरण",
    multitool_task_status: "टास्क स्थिति",
    multitool_task_priority: "टास्क प्राथमिकता",
    multitool_task_due: "अंतिम तिथि",
    multitool_task_result: "टास्क सेव परिणाम",
    multitool_risk_title: "जोखिम शीर्षक",
    multitool_risk_description: "जोखिम विवरण",
    multitool_risk_probability: "संभाव्यता",
    multitool_risk_impact: "प्रभाव",
    multitool_risk_mitigation: "निवारण कदम",
    multitool_risk_result: "जोखिम सेव परिणाम",
    multitool_domain: "डोमेन शाखा",
    multitool_country: "विधिक देश",
    multitool_save: "सहेजें",
    multitool_new_item: "नया",
    multitool_dashboard: "मल्टीटूल डैशबोर्ड",
    multitool_chart_task_status: "टास्क स्थिति वितरण",
    multitool_chart_task_priority: "टास्क प्राथमिकता वितरण",
    multitool_chart_risk_probability: "जोखिम संभाव्यता वितरण",
    multitool_chart_risk_impact: "जोखिम प्रभाव वितरण",
    multitool_chart_domain_coverage: "डोमेन कवरेज",
    multitool_open_tasks: "खुले कार्य",
    multitool_top_risks: "शीर्ष जोखिम",
    multitool_no_items: "अभी कोई डेटा नहीं।",
    action_multitool_request_save: "सुधार अनुरोध सहेजना",
    action_multitool_preference_save: "प्राथमिकता प्रोफ़ाइल सहेजना",
    action_multitool_task_save: "टास्क सहेजना",
    action_multitool_risk_save: "जोखिम सहेजना",
    error_multitool_request_title_empty: "अनुरोध शीर्षक खाली है",
    error_multitool_task_title_empty: "टास्क शीर्षक खाली है",
    error_multitool_risk_title_empty: "जोखिम शीर्षक खाली है",
  },
  zh: {
    overview_section_multitool: "多工具",
    multitool_title: "个人多工具工作台",
    multitool_subtitle: "将改进请求、偏好、任务和风险写入图谱记忆，并按领域与国家法域分支管理。",
    multitool_section_requests: "改进请求",
    multitool_section_preferences: "偏好与反偏好",
    multitool_section_tasks: "任务看板",
    multitool_section_risks: "风险分析",
    multitool_choose_existing: "已有节点",
    multitool_request_title: "请求标题",
    multitool_request_details: "请求详情",
    multitool_request_output: "期望输出",
    multitool_request_layout: "偏好视图",
    multitool_request_status: "状态",
    multitool_request_priority: "优先级",
    multitool_request_result: "请求保存结果",
    multitool_preferences_profile_name: "画像名称",
    multitool_preferences_likes: "喜欢",
    multitool_preferences_dislikes: "不喜欢",
    multitool_preferences_style: "风格偏好",
    multitool_preferences_tools: "常用工具",
    multitool_preferences_notes: "记忆备注",
    multitool_preferences_result: "偏好保存结果",
    multitool_task_title: "任务标题",
    multitool_task_description: "任务描述",
    multitool_task_status: "任务状态",
    multitool_task_priority: "任务优先级",
    multitool_task_due: "截止日期",
    multitool_task_result: "任务保存结果",
    multitool_risk_title: "风险标题",
    multitool_risk_description: "风险描述",
    multitool_risk_probability: "概率",
    multitool_risk_impact: "影响",
    multitool_risk_mitigation: "缓解步骤",
    multitool_risk_result: "风险保存结果",
    multitool_domain: "领域分支",
    multitool_country: "法域国家",
    multitool_save: "保存",
    multitool_new_item: "新建",
    multitool_dashboard: "多工具看板",
    multitool_chart_task_status: "任务状态分布",
    multitool_chart_task_priority: "任务优先级分布",
    multitool_chart_risk_probability: "风险概率分布",
    multitool_chart_risk_impact: "风险影响分布",
    multitool_chart_domain_coverage: "领域覆盖",
    multitool_open_tasks: "未完成任务",
    multitool_top_risks: "重点风险",
    multitool_no_items: "暂无数据。",
    action_multitool_request_save: "保存改进请求",
    action_multitool_preference_save: "保存偏好画像",
    action_multitool_task_save: "保存任务",
    action_multitool_risk_save: "保存风险",
    error_multitool_request_title_empty: "请求标题为空",
    error_multitool_task_title_empty: "任务标题为空",
    error_multitool_risk_title_empty: "风险标题为空",
  },
  ja: {
    overview_section_multitool: "マルチツール",
    multitool_title: "個人マルチツールワークスペース",
    multitool_subtitle:
      "改善要望、好み、タスク、リスクを保存し、ドメインと国別法制度でグラフ分岐管理します。",
    multitool_section_requests: "改善リクエスト",
    multitool_section_preferences: "好み・苦手",
    multitool_section_tasks: "タスクボード",
    multitool_section_risks: "リスク分析",
    multitool_choose_existing: "既存ノード",
    multitool_request_title: "リクエスト名",
    multitool_request_details: "詳細",
    multitool_request_output: "期待する出力",
    multitool_request_layout: "希望ビュー",
    multitool_request_status: "ステータス",
    multitool_request_priority: "優先度",
    multitool_request_result: "リクエスト保存結果",
    multitool_preferences_profile_name: "プロファイル名",
    multitool_preferences_likes: "好き",
    multitool_preferences_dislikes: "苦手",
    multitool_preferences_style: "スタイル嗜好",
    multitool_preferences_tools: "好みのツール",
    multitool_preferences_notes: "メモ",
    multitool_preferences_result: "嗜好保存結果",
    multitool_task_title: "タスク名",
    multitool_task_description: "タスク説明",
    multitool_task_status: "タスク状態",
    multitool_task_priority: "タスク優先度",
    multitool_task_due: "期限",
    multitool_task_result: "タスク保存結果",
    multitool_risk_title: "リスク名",
    multitool_risk_description: "リスク説明",
    multitool_risk_probability: "確率",
    multitool_risk_impact: "影響度",
    multitool_risk_mitigation: "緩和ステップ",
    multitool_risk_result: "リスク保存結果",
    multitool_domain: "ドメイン分岐",
    multitool_country: "法制度の国",
    multitool_save: "保存",
    multitool_new_item: "新規",
    multitool_dashboard: "マルチツールダッシュボード",
    multitool_chart_task_status: "タスク状態分布",
    multitool_chart_task_priority: "タスク優先度分布",
    multitool_chart_risk_probability: "リスク確率分布",
    multitool_chart_risk_impact: "リスク影響分布",
    multitool_chart_domain_coverage: "ドメインカバレッジ",
    multitool_open_tasks: "未完了タスク",
    multitool_top_risks: "主要リスク",
    multitool_no_items: "まだデータがありません。",
    action_multitool_request_save: "改善リクエスト保存",
    action_multitool_preference_save: "嗜好プロファイル保存",
    action_multitool_task_save: "タスク保存",
    action_multitool_risk_save: "リスク保存",
    error_multitool_request_title_empty: "リクエスト名が空です",
    error_multitool_task_title_empty: "タスク名が空です",
    error_multitool_risk_title_empty: "リスク名が空です",
  },
};

for (const [code, pack] of Object.entries(TRANSLATION_EXTENSIONS)) {
  TRANSLATIONS[code] = {
    ...(TRANSLATIONS.en || {}),
    ...(pack || {}),
  };
}

for (const [code, pack] of Object.entries(EXTRA_TRANSLATION_EXTENSIONS)) {
  EXTRA_TRANSLATIONS[code] = {
    ...(EXTRA_TRANSLATIONS.en || {}),
    ...(pack || {}),
  };
}

for (const [code, pack] of Object.entries(MULTITOOL_UI_TRANSLATIONS)) {
  EXTRA_TRANSLATIONS[code] = {
    ...(EXTRA_TRANSLATIONS[code] || EXTRA_TRANSLATIONS.en || {}),
    ...(pack || {}),
  };
}

const OVERVIEW_SECTION_TRANSLATION_KEYS = {
  demo: "overview_section_demo",
  daily: "overview_section_daily",
  user_graph: "overview_section_user_graph",
  autoruns: "overview_section_autoruns",
  multitool: "overview_section_multitool",
  graph: "overview_section_graph",
  client: "overview_section_client",
  advisors: "overview_section_advisors",
  hallucination_hunter: "overview_section_hallucination_hunter",
};

function detectInitialLanguage() {
  try {
    const saved = localStorage.getItem("ui_language");
    if (saved && UI_LANG_OPTIONS.some((item) => item.code === saved)) {
      return saved;
    }
  } catch (_error) {
    // ignore
  }

  const raw = String(navigator.language || "en").toLowerCase();
  if (raw.startsWith("hy")) return "hy";
  if (raw.startsWith("ru")) return "ru";
  if (raw.startsWith("fr")) return "fr";
  if (raw.startsWith("es")) return "es";
  if (raw.startsWith("pt")) return "pt";
  if (raw.startsWith("ar")) return "ar";
  if (raw.startsWith("hi")) return "hi";
  if (raw.startsWith("zh")) return "zh";
  if (raw.startsWith("ja")) return "ja";
  return "en";
}

function getInitialPage() {
  const raw = String(window.location.hash || "").replace(/^#\/?/, "").trim();
  if (PAGE_KEYS.includes(raw)) {
    return raw;
  }
  return "overview";
}

function stringifySafe(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch (_error) {
    return String(value);
  }
}

function parseJsonSafe(text, fallbackValue, invalidJsonMessage) {
  const raw = String(text || "").trim();
  if (!raw) {
    return fallbackValue;
  }
  try {
    return JSON.parse(raw);
  } catch (_error) {
    throw new Error(invalidJsonMessage);
  }
}

function parseListText(text) {
  return String(text || "")
    .split(/[\n,;|]+/g)
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseNumericListText(text) {
  return String(text || "")
    .split(/[\n,;| ]+/g)
    .map((item) => Number(item))
    .filter((value) => Number.isFinite(value) && value > 0)
    .map((value) => Math.trunc(value));
}

function countRowsFromObject(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return [];
  }
  return Object.entries(value)
    .map(([name, count]) => ({
      name: String(name || "").trim(),
      count: Number(count || 0),
    }))
    .filter((row) => row.name && Number.isFinite(row.count) && row.count > 0)
    .sort((a, b) => b.count - a.count);
}

function normalizeRole(value, fallback = "general") {
  const raw = String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, "_");
  if (LLM_ROLE_OPTIONS.includes(raw)) {
    return raw;
  }
  return fallback;
}

function normalizePersonalizationDraft(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  return {
    response_style: PERSONALIZATION_STYLE_OPTIONS.includes(String(source.response_style || ""))
      ? String(source.response_style)
      : DEFAULT_PERSONALIZATION_DRAFT.response_style,
    reasoning_depth: PERSONALIZATION_DEPTH_OPTIONS.includes(String(source.reasoning_depth || ""))
      ? String(source.reasoning_depth)
      : DEFAULT_PERSONALIZATION_DRAFT.reasoning_depth,
    risk_tolerance: PERSONALIZATION_RISK_OPTIONS.includes(String(source.risk_tolerance || ""))
      ? String(source.risk_tolerance)
      : DEFAULT_PERSONALIZATION_DRAFT.risk_tolerance,
    tone: PERSONALIZATION_TONE_OPTIONS.includes(String(source.tone || ""))
      ? String(source.tone)
      : DEFAULT_PERSONALIZATION_DRAFT.tone,
    focus_goals_text: String(source.focus_goals_text || ""),
    domain_focus_text: String(source.domain_focus_text || ""),
    avoid_topics_text: String(source.avoid_topics_text || ""),
    memory_notes: String(source.memory_notes || ""),
    role_proposer: normalizeRole(source.role_proposer, DEFAULT_PERSONALIZATION_DRAFT.role_proposer),
    role_critic: normalizeRole(source.role_critic, DEFAULT_PERSONALIZATION_DRAFT.role_critic),
    role_judge: normalizeRole(source.role_judge, DEFAULT_PERSONALIZATION_DRAFT.role_judge),
    auto_apply_user_graph:
      typeof source.auto_apply_user_graph === "boolean"
        ? source.auto_apply_user_graph
        : DEFAULT_PERSONALIZATION_DRAFT.auto_apply_user_graph,
    auto_apply_daily:
      typeof source.auto_apply_daily === "boolean"
        ? source.auto_apply_daily
        : DEFAULT_PERSONALIZATION_DRAFT.auto_apply_daily,
    auto_apply_debate:
      typeof source.auto_apply_debate === "boolean"
        ? source.auto_apply_debate
        : DEFAULT_PERSONALIZATION_DRAFT.auto_apply_debate,
  };
}

function normalizeOption(value, options, fallback) {
  const token = String(value || "").trim();
  if (options.includes(token)) {
    return token;
  }
  return fallback;
}

function normalizeMultitoolDomain(value) {
  return normalizeOption(value, MULTITOOL_DOMAIN_OPTIONS, "general");
}

function normalizeLegislationCountry(value) {
  return normalizeOption(value, LEGISLATION_COUNTRY_OPTIONS, "global");
}

function normalizeMultitoolRequestDraft(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  return {
    title: String(source.title || ""),
    details: String(source.details || ""),
    desired_output: String(source.desired_output || ""),
    layout_mode: String(source.layout_mode || "graph"),
    status: normalizeOption(source.status, REQUEST_STATUS_OPTIONS, "backlog"),
    priority: normalizeOption(source.priority, TASK_PRIORITY_OPTIONS, "medium"),
    domain: normalizeMultitoolDomain(source.domain),
    country: normalizeLegislationCountry(source.country),
  };
}

function normalizeMultitoolPreferenceDraft(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  return {
    profile_name: String(source.profile_name || "default"),
    likes_text: String(source.likes_text || ""),
    dislikes_text: String(source.dislikes_text || ""),
    style_examples_text: String(source.style_examples_text || ""),
    tool_examples_text: String(source.tool_examples_text || ""),
    notes: String(source.notes || ""),
    domain: normalizeMultitoolDomain(source.domain),
    country: normalizeLegislationCountry(source.country),
  };
}

function normalizeMultitoolTaskDraft(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  return {
    title: String(source.title || ""),
    description: String(source.description || ""),
    status: normalizeOption(source.status, TASK_STATUS_OPTIONS, "backlog"),
    priority: normalizeOption(source.priority, TASK_PRIORITY_OPTIONS, "medium"),
    due_at: String(source.due_at || ""),
    domain: normalizeMultitoolDomain(source.domain),
    country: normalizeLegislationCountry(source.country),
  };
}

function normalizeMultitoolRiskDraft(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  return {
    title: String(source.title || ""),
    description: String(source.description || ""),
    probability: normalizeOption(source.probability, RISK_PROBABILITY_OPTIONS, "medium"),
    impact: normalizeOption(source.impact, RISK_IMPACT_OPTIONS, "medium"),
    mitigation_text: String(source.mitigation_text || ""),
    domain: normalizeMultitoolDomain(source.domain),
    country: normalizeLegislationCountry(source.country),
  };
}

function normalizePersonalTreeIngestDraft(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  return {
    title: String(source.title || ""),
    topic: String(source.topic || ""),
    text: String(source.text || ""),
    source_type: String(source.source_type || "text"),
    source_url: String(source.source_url || ""),
    source_title: String(source.source_title || ""),
    max_points: Math.max(2, Math.min(12, Number(source.max_points || 6))),
  };
}

function normalizePersonalTreeNoteDraft(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  return {
    title: String(source.title || ""),
    note: String(source.note || ""),
    tags_text: String(source.tags_text || ""),
    links_text: String(source.links_text || ""),
    source_type: String(source.source_type || "note"),
    source_url: String(source.source_url || ""),
    source_title: String(source.source_title || ""),
  };
}

function normalizePackagesDraft(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  return {
    package_name: String(source.package_name || "inbox")
      .trim()
      .replace(/[^a-z0-9_]+/gi, "_")
      .replace(/^_+|_+$/g, "")
      .toLowerCase() || "inbox",
    items_text: String(source.items_text || ""),
    restore_ids_text: String(source.restore_ids_text || ""),
    model_role: normalizeRole(source.model_role, "coder_reviewer"),
    model_path: String(source.model_path || "").trim(),
    classify_with_llm: Boolean(source.classify_with_llm),
    apply_changes: Boolean(source.apply_changes),
    confirmation: String(source.confirmation || "confirm"),
  };
}

function normalizeMemoryNamespaceDraft(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  return {
    namespace: normalizeOption(source.namespace, MEMORY_NAMESPACE_OPTIONS, "personal"),
    source_namespace: normalizeOption(source.source_namespace, MEMORY_NAMESPACE_OPTIONS, ""),
    scope: normalizeOption(source.scope, MEMORY_SCOPE_OPTIONS, "owned"),
    query: String(source.query || ""),
    node_ids_text: String(source.node_ids_text || ""),
    min_score: String(source.min_score || "0.2"),
    apply_changes: Boolean(source.apply_changes),
    confirmation: String(source.confirmation || "confirm"),
  };
}

function normalizeGraphRagDraft(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  return {
    query: String(source.query || ""),
    top_k: Math.max(1, Math.min(20, Number(source.top_k || 6))),
    scope: normalizeOption(source.scope, MEMORY_SCOPE_OPTIONS, "owned"),
    namespace: normalizeOption(source.namespace, MEMORY_NAMESPACE_OPTIONS, ""),
    use_llm: Boolean(source.use_llm),
    model_role: normalizeRole(source.model_role, "analyst"),
    model_path: String(source.model_path || "").trim(),
  };
}

function normalizeContradictionScanDraft(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  return {
    scope: normalizeOption(source.scope, MEMORY_SCOPE_OPTIONS, "owned"),
    namespace: normalizeOption(source.namespace, MEMORY_NAMESPACE_OPTIONS, ""),
    max_nodes: Math.max(10, Math.min(240, Number(source.max_nodes || 120))),
    top_k: Math.max(1, Math.min(120, Number(source.top_k || 20))),
    min_overlap: String(source.min_overlap || "0.32"),
    apply_to_graph: Boolean(source.apply_to_graph),
    confirmation: String(source.confirmation || "confirm"),
  };
}

function normalizeTaskRiskDraft(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  return {
    tasks_text: String(source.tasks_text || ""),
    apply_to_graph: Boolean(source.apply_to_graph),
    confirmation: String(source.confirmation || "confirm"),
  };
}

function normalizeTimelineReplayDraft(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  return {
    event_type: String(source.event_type || "").trim(),
    limit: Math.max(1, Math.min(3000, Number(source.limit || 300))),
    from_ts: String(source.from_ts || ""),
    to_ts: String(source.to_ts || ""),
  };
}

function normalizeLlmPolicyDraft(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  return {
    mode: normalizeOption(source.mode, LLM_POLICY_MODE_OPTIONS, "confirm_required"),
    trusted_sessions_text: String(source.trusted_sessions_text || ""),
    trusted_users_text: String(source.trusted_users_text || ""),
    allow_actions_text: String(source.allow_actions_text || ""),
    merge_lists: Boolean(source.merge_lists),
  };
}

function normalizeQualityDraft(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  return {
    sample_queries_text: String(source.sample_queries_text || ""),
  };
}

function normalizeBackupDraft(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  return {
    label: String(source.label || "manual"),
    include_events: Boolean(source.include_events),
    event_limit: Math.max(0, Math.min(10000, Number(source.event_limit || 1000))),
    path: String(source.path || "").trim(),
    latest: Boolean(source.latest),
    apply_changes: Boolean(source.apply_changes),
    confirmation: String(source.confirmation || "confirm"),
    restore_policy: Boolean(source.restore_policy),
  };
}

function normalizeAuditDraft(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  return {
    limit: Math.max(1, Math.min(2000, Number(source.limit || 200))),
    include_backups: Boolean(source.include_backups),
  };
}

function countRowsFromTokens(tokens, ordered = []) {
  const counts = new Map();
  for (const raw of tokens || []) {
    const key = String(raw || "").trim();
    if (!key) continue;
    counts.set(key, Number(counts.get(key) || 0) + 1);
  }
  const out = [];
  for (const key of ordered) {
    const count = Number(counts.get(key) || 0);
    if (count > 0) {
      out.push({ name: key, count });
      counts.delete(key);
    }
  }
  for (const [name, count] of counts.entries()) {
    out.push({ name, count });
  }
  out.sort((a, b) => Number(b.count || 0) - Number(a.count || 0));
  return out;
}

function parseNodeUpdatedTimestamp(node) {
  const attrs = node?.attributes && typeof node.attributes === "object" ? node.attributes : {};
  const raw = String(attrs.updated_at || attrs.created_at || "");
  if (!raw) return 0;
  const ts = Date.parse(raw);
  return Number.isFinite(ts) ? ts : 0;
}

function loadPersonalizationDraft() {
  try {
    const raw = localStorage.getItem(PERSONALIZATION_STORAGE_KEY);
    if (!raw) {
      return { ...DEFAULT_PERSONALIZATION_DRAFT };
    }
    return normalizePersonalizationDraft(JSON.parse(raw));
  } catch (_error) {
    return { ...DEFAULT_PERSONALIZATION_DRAFT };
  }
}

function loadStyleNodeIndex() {
  try {
    const raw = localStorage.getItem(STYLE_NODE_STORAGE_KEY);
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) return 0;
    const max = Math.max(0, STYLE_NODE_PRESETS.length - 1);
    return Math.max(0, Math.min(max, Math.trunc(parsed)));
  } catch (_error) {
    return 0;
  }
}

function stylePresetByIndex(index) {
  const max = Math.max(0, STYLE_NODE_PRESETS.length - 1);
  const safe = Math.max(0, Math.min(max, Number(index || 0)));
  return STYLE_NODE_PRESETS[safe] || STYLE_NODE_PRESETS[0];
}

function applyStylePreset(preset) {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  if (!root || !preset?.vars || typeof preset.vars !== "object") return;
  for (const [name, value] of Object.entries(preset.vars)) {
    root.style.setProperty(String(name), String(value));
  }
}

function coerceStyleVars(rawVars, fallbackVars = {}) {
  const source = rawVars && typeof rawVars === "object" && !Array.isArray(rawVars) ? rawVars : {};
  const next = { ...(fallbackVars || {}) };
  for (const key of STYLE_VAR_ALLOWLIST) {
    const candidate = source[key];
    if (typeof candidate === "string" && candidate.trim()) {
      next[key] = candidate.trim();
    }
  }
  return next;
}

function styleNodeAttributes(node) {
  const attrs = node?.attributes && typeof node.attributes === "object" ? node.attributes : {};
  const styleId = String(attrs.style_id || "").trim();
  if (!styleId) return null;
  return {
    styleId,
    styleName: String(attrs.style_name || attrs.name || "").trim(),
    styleDescription: String(attrs.style_description || attrs.description || "").trim(),
    styleVars: attrs.style_vars && typeof attrs.style_vars === "object" ? attrs.style_vars : {},
    isActive: Boolean(attrs.is_active),
    updatedAt: String(attrs.updated_at || ""),
  };
}

function branchScopeKeyForInsights(insights) {
  if (insights?.hasTarget && Number.isFinite(Number(insights?.targetNode))) {
    return `node:${Number(insights.targetNode)}`;
  }
  return "global";
}

function buildPersonalizationPayload(draft, uiLanguage) {
  const safe = normalizePersonalizationDraft(draft);
  const focusGoals = parseListText(safe.focus_goals_text);
  const domainFocus = parseListText(safe.domain_focus_text);
  const avoidTopics = parseListText(safe.avoid_topics_text);
  return {
    response_style: safe.response_style,
    reasoning_depth: safe.reasoning_depth,
    risk_tolerance: safe.risk_tolerance,
    tone: safe.tone,
    focus_goals: focusGoals,
    domain_focus: domainFocus,
    avoid_topics: avoidTopics,
    memory_notes: String(safe.memory_notes || "").trim(),
    language: String(uiLanguage || "en"),
    llm_roles: {
      proposer: normalizeRole(safe.role_proposer, "creative"),
      critic: normalizeRole(safe.role_critic, "analyst"),
      judge: normalizeRole(safe.role_judge, "planner"),
    },
  };
}

function summarizePersonalization(payload) {
  const p = payload && typeof payload === "object" ? payload : {};
  const goalsCount = Array.isArray(p.focus_goals) ? p.focus_goals.length : 0;
  const domainsCount = Array.isArray(p.domain_focus) ? p.domain_focus.length : 0;
  return [
    `style=${p.response_style || "adaptive"}`,
    `depth=${p.reasoning_depth || "balanced"}`,
    `risk=${p.risk_tolerance || "medium"}`,
    `tone=${p.tone || "neutral"}`,
    `goals=${goalsCount}`,
    `domains=${domainsCount}`,
  ].join(" | ");
}

function personalizationPromptBlock(payload) {
  const p = payload && typeof payload === "object" ? payload : {};
  const rows = [
    `response_style=${p.response_style || "adaptive"}`,
    `reasoning_depth=${p.reasoning_depth || "balanced"}`,
    `risk_tolerance=${p.risk_tolerance || "medium"}`,
    `tone=${p.tone || "neutral"}`,
  ];
  if (Array.isArray(p.focus_goals) && p.focus_goals.length) {
    rows.push(`focus_goals=${p.focus_goals.join(", ")}`);
  }
  if (Array.isArray(p.domain_focus) && p.domain_focus.length) {
    rows.push(`domain_focus=${p.domain_focus.join(", ")}`);
  }
  if (Array.isArray(p.avoid_topics) && p.avoid_topics.length) {
    rows.push(`avoid_topics=${p.avoid_topics.join(", ")}`);
  }
  if (typeof p.memory_notes === "string" && p.memory_notes.trim()) {
    rows.push(`memory_notes=${p.memory_notes.trim()}`);
  }
  return rows.join("\n");
}

function pagedSlice(items, page, pageSize) {
  const list = Array.isArray(items) ? items : [];
  const safeSize = Math.max(1, Number(pageSize || 1));
  const totalPages = Math.max(1, Math.ceil(list.length / safeSize));
  const safePage = Math.max(0, Math.min(totalPages - 1, Number(page || 0)));
  const start = safePage * safeSize;
  return {
    totalPages,
    page: safePage,
    items: list.slice(start, start + safeSize),
  };
}

function overviewSectionLabel(key, t) {
  const textKey = OVERVIEW_SECTION_TRANSLATION_KEYS[key];
  if (!textKey) {
    return key;
  }
  return t(textKey);
}

function getClientSessionId() {
  const key = "client_session_id";
  try {
    const current = localStorage.getItem(key);
    if (current) return current;
  } catch (_error) {
    // ignore
  }

  const generated = `sess_${Math.random().toString(36).slice(2, 10)}_${Date.now().toString(36)}`;
  try {
    localStorage.setItem(key, generated);
  } catch (_error) {
    // ignore
  }
  return generated;
}

function collectClientContext() {
  const nav = typeof navigator !== "undefined" ? navigator : {};
  const screenObj = typeof screen !== "undefined" ? screen : {};
  const connection = nav.connection || nav.mozConnection || nav.webkitConnection || {};
  const viewport = {
    width: typeof window !== "undefined" ? Number(window.innerWidth || 0) : 0,
    height: typeof window !== "undefined" ? Number(window.innerHeight || 0) : 0,
  };
  return {
    user_agent: String(nav.userAgent || ""),
    platform: String(nav.platform || ""),
    vendor: String(nav.vendor || ""),
    language: String(nav.language || ""),
    languages: Array.isArray(nav.languages) ? nav.languages : [],
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || "",
    hardware_concurrency: Number(nav.hardwareConcurrency || 0),
    device_memory_gb: Number(nav.deviceMemory || 0),
    max_touch_points: Number(nav.maxTouchPoints || 0),
    webdriver: Boolean(nav.webdriver),
    online: Boolean(nav.onLine),
    cookies_enabled: typeof nav.cookieEnabled === "boolean" ? nav.cookieEnabled : true,
    do_not_track: String(nav.doNotTrack || ""),
    screen: {
      width: Number(screenObj.width || 0),
      height: Number(screenObj.height || 0),
      color_depth: Number(screenObj.colorDepth || 0),
      pixel_ratio: typeof window !== "undefined" ? Number(window.devicePixelRatio || 1) : 1,
    },
    viewport,
    connection: {
      effective_type: String(connection.effectiveType || ""),
      downlink_mbps: Number(connection.downlink || 0),
      rtt_ms: Number(connection.rtt || 0),
      save_data: Boolean(connection.saveData),
    },
  };
}

function nodeLabel(node) {
  const attrs = node?.attributes || {};
  const name =
    attrs.name ||
    [attrs.first_name || "", attrs.last_name || ""].join(" ").trim() ||
    attrs.identifier ||
    attrs.code ||
    "";
  return String(name || `#${node?.id || "?"}`);
}

function edgeSignature(edge) {
  return `${edge.from}|${edge.to}|${edge.relation_type}|${edge.direction || "directed"}`;
}

function edgeSignatureFromParts(fromNode, toNode, relationType, direction = "directed") {
  return `${Number(fromNode)}|${Number(toNode)}|${String(relationType || "")}|${String(direction || "directed")}`;
}

function humanizeToken(value) {
  return String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function classifyEdgeStrength(weight) {
  const score = Number(weight || 0);
  if (score >= 0.85) return "very strong";
  if (score >= 0.65) return "strong";
  if (score >= 0.4) return "moderate";
  if (score > 0) return "weak";
  return "neutral";
}

function summarizeEdgeReasoning(edge, nodeNameById) {
  if (!edge) {
    return {
      summary: "",
      facts: [],
      direction: "",
      relation: "",
      logic: "",
      strength: "",
    };
  }
  const from = Number(edge.from);
  const to = Number(edge.to);
  const relation = humanizeToken(edge.relation_type) || "relation";
  const direction = String(edge.direction || "directed") === "undirected" ? "undirected" : "directed";
  const logic = humanizeToken(edge.logic_rule) || "explicit";
  const weight = Number(edge.weight || 0);
  const strength = `${classifyEdgeStrength(weight)} (${weight.toFixed(2)})`;
  const fromName = nodeNameById.get(from) || `#${from}`;
  const toName = nodeNameById.get(to) || `#${to}`;
  const metadata = edge.metadata && typeof edge.metadata === "object" ? edge.metadata : {};
  const facts = [];

  const relationHints = {
    debate_branch: "Created as a debate branch from session to hypothesis.",
    criticized_by: "Represents critique attached to a hypothesis branch.",
    judged_by: "Represents final judgement node connected to debate session.",
    selects: "Represents selected hypothesis chosen by judge.",
    influences: "Represents inferred influence between entities.",
    works_at: "Represents employment or affiliation relation.",
    owns: "Represents ownership or control relation.",
    part_of: "Represents composition or membership relation.",
    parent_of: "Represents hierarchical parent-child relation.",
  };
  const logicHints = {
    llm_debate_proposer: "Generated by proposer role in LLM debate flow.",
    llm_debate_critic: "Generated by critic role evaluating risks/contradictions.",
    llm_debate_judge: "Generated by judge role consolidating final decision.",
    llm_debate_selection: "Generated by judge selection step in debate flow.",
    explicit: "Manually created or explicitly updated relation.",
  };

  const relationToken = String(edge.relation_type || "").trim().toLowerCase();
  const logicToken = String(edge.logic_rule || "").trim().toLowerCase();
  if (relationHints[relationToken]) {
    facts.push(relationHints[relationToken]);
  }
  if (logicHints[logicToken]) {
    facts.push(logicHints[logicToken]);
  }
  if (metadata.branch_id) {
    facts.push(`Branch context: ${String(metadata.branch_id)}`);
  }
  if (metadata.selected_index != null) {
    facts.push(`Selected hypothesis index: ${Number(metadata.selected_index) || metadata.selected_index}`);
  }
  if (metadata.index != null) {
    facts.push(`Branch index: ${Number(metadata.index) || metadata.index}`);
  }
  if (metadata.reasoning) {
    facts.push(`Reasoning note: ${String(metadata.reasoning)}`);
  }
  if (metadata.explanation) {
    facts.push(`Explanation: ${String(metadata.explanation)}`);
  }
  if (metadata.rationale) {
    facts.push(`Rationale: ${String(metadata.rationale)}`);
  }
  const metadataKeys = Object.keys(metadata);
  if (metadataKeys.length) {
    facts.push(`Metadata keys: ${metadataKeys.join(", ")}`);
  } else {
    facts.push("No metadata attached to this edge.");
  }

  return {
    summary: `${fromName} -> ${toName}`,
    facts: facts.slice(0, 7),
    direction,
    relation,
    logic,
    strength,
  };
}

function edgeSignatureFromEvent(event) {
  const payload = event?.payload && typeof event.payload === "object" ? event.payload : {};
  const relationType = String(payload.relation_type || "").trim();
  if (!relationType) {
    return "";
  }
  const from = Number(payload.from);
  const to = Number(payload.to);
  if (!Number.isFinite(from) || !Number.isFinite(to)) {
    return "";
  }
  return edgeSignatureFromParts(from, to, relationType, String(payload.direction || "directed"));
}

function formatEventTimeLabel(event) {
  const timestamp = Number(event?.timestamp || 0);
  if (!Number.isFinite(timestamp) || timestamp <= 0) {
    return "";
  }
  return new Date(timestamp * 1000).toLocaleTimeString();
}

function formatEpochLabel(value) {
  const raw = Number(value || 0);
  if (!Number.isFinite(raw) || raw <= 0) {
    return "-";
  }
  const ms = raw > 1000000000000 ? raw : raw * 1000;
  return new Date(ms).toLocaleString();
}

function deriveEdgeHistory(events, edgeSig) {
  if (!edgeSig) return [];
  const rows = Array.isArray(events) ? events : [];
  const typeLabel = {
    edge_added: "edge_added",
    edge_updated: "edge_updated",
    edge_updated_manual: "edge_updated_manual",
    edge_weight_feedback: "edge_weight_feedback",
    edge_deleted: "edge_deleted",
  };

  const out = [];
  for (const row of rows) {
    const sig = edgeSignatureFromEvent(row);
    if (!sig || sig !== edgeSig) {
      continue;
    }
    const payload = row?.payload && typeof row.payload === "object" ? row.payload : {};
    const eventType = String(row?.event_type || "");
    const metadata = payload.metadata && typeof payload.metadata === "object" ? payload.metadata : {};
    const reason =
      String(
        metadata.reasoning ||
          metadata.explanation ||
          metadata.rationale ||
          payload.reason ||
          payload.source_event ||
          ""
      ).trim() || "n/a";
    const weightNow =
      eventType === "edge_weight_feedback"
        ? Number(payload.new_weight)
        : Number(payload.weight);
    const prevWeight = Number(payload.previous_weight ?? payload.old_weight);
    const weightLabel = Number.isFinite(weightNow)
      ? Number.isFinite(prevWeight)
        ? `${prevWeight.toFixed(2)} -> ${weightNow.toFixed(2)}`
        : weightNow.toFixed(2)
      : "n/a";
    const logic = String(payload.logic_rule || payload.new_logic_rule || "").trim() || "n/a";
    const metadataKeys = Object.keys(metadata);
    out.push({
      id: Number(row?.id || 0),
      timestamp: Number(row?.timestamp || 0),
      timeLabel: formatEventTimeLabel(row),
      eventType: typeLabel[eventType] || eventType || "event",
      weightLabel,
      logic,
      reason,
      metadataKeys,
    });
  }

  return out
    .sort((a, b) => {
      if (a.timestamp !== b.timestamp) return b.timestamp - a.timestamp;
      return b.id - a.id;
    })
    .slice(0, 18);
}

function computeSnapshotMetrics(snapshot) {
  const nodes = Array.isArray(snapshot?.nodes) ? snapshot.nodes : [];
  const edges = Array.isArray(snapshot?.edges) ? snapshot.edges : [];
  const relationCounts = {};
  const nodeTypeCounts = {};

  for (const edge of edges) {
    const key = String(edge?.relation_type || "");
    relationCounts[key] = Number(relationCounts[key] || 0) + 1;
  }
  for (const node of nodes) {
    const key = String(node?.type || "generic");
    nodeTypeCounts[key] = Number(nodeTypeCounts[key] || 0) + 1;
  }

  return {
    node_count: nodes.length,
    edge_count: edges.length,
    relation_counts: relationCounts,
    node_type_counts: nodeTypeCounts,
  };
}

function buildSnapshotPayload(snapshot) {
  return {
    snapshot,
    metrics: computeSnapshotMetrics(snapshot),
  };
}

function applyGraphEventToSnapshot(currentPayload, event) {
  const currentSnapshot = currentPayload?.snapshot || { nodes: [], edges: [] };
  const nodes = Array.isArray(currentSnapshot.nodes) ? [...currentSnapshot.nodes] : [];
  const edges = Array.isArray(currentSnapshot.edges) ? [...currentSnapshot.edges] : [];
  const payload = event?.payload || {};
  const type = String(event?.event_type || "");

  let changed = false;
  let needsSync = false;

  const upsertNode = (node) => {
    if (!node || typeof node !== "object") return false;
    const nodeId = Number(node.id);
    if (Number.isNaN(nodeId) || nodeId <= 0) return false;
    const normalized = {
      id: nodeId,
      type: String(node.type || "generic"),
      attributes: node.attributes && typeof node.attributes === "object" ? node.attributes : {},
      state: node.state && typeof node.state === "object" ? node.state : {},
    };
    const idx = nodes.findIndex((row) => Number(row?.id) === nodeId);
    if (idx >= 0) {
      nodes[idx] = normalized;
    } else {
      nodes.push(normalized);
    }
    return true;
  };

  const upsertEdge = (edgeLike) => {
    if (!edgeLike || typeof edgeLike !== "object") return false;
    const normalized = {
      from: Number(edgeLike.from),
      to: Number(edgeLike.to),
      relation_type: String(edgeLike.relation_type || ""),
      weight: Number(edgeLike.weight || 0),
      direction: String(edgeLike.direction || "directed"),
      logic_rule: String(edgeLike.logic_rule || "explicit"),
      metadata: edgeLike.metadata && typeof edgeLike.metadata === "object" ? edgeLike.metadata : {},
    };
    if (Number.isNaN(normalized.from) || Number.isNaN(normalized.to) || !normalized.relation_type) {
      return false;
    }
    const signature = edgeSignatureFromParts(
      normalized.from,
      normalized.to,
      normalized.relation_type,
      normalized.direction
    );
    const idx = edges.findIndex((row) => edgeSignature(row) === signature);
    if (idx >= 0) {
      edges[idx] = normalized;
    } else {
      edges.push(normalized);
    }
    return true;
  };

  if (type === "node_added") {
    changed = upsertNode(payload.node);
    if (!changed) needsSync = true;
  } else if (type === "node_updated") {
    changed = upsertNode(payload.node);
    if (!changed) needsSync = true;
  } else if (type === "node_deleted") {
    const nodeId = Number(payload.node_id);
    if (!Number.isNaN(nodeId) && nodeId > 0) {
      const beforeNodes = nodes.length;
      const beforeEdges = edges.length;
      const nextNodes = nodes.filter((node) => Number(node?.id) !== nodeId);
      const nextEdges = edges.filter(
        (edge) => Number(edge?.from) !== nodeId && Number(edge?.to) !== nodeId
      );
      changed = beforeNodes !== nextNodes.length || beforeEdges !== nextEdges.length;
      if (changed) {
        nodes.splice(0, nodes.length, ...nextNodes);
        edges.splice(0, edges.length, ...nextEdges);
      }
    } else {
      needsSync = true;
    }
  } else if (type === "edge_added" || type === "edge_updated" || type === "edge_updated_manual") {
    changed = upsertEdge({
      from: payload.from,
      to: payload.to,
      relation_type: payload.relation_type,
      weight: payload.weight,
      direction: payload.direction,
      logic_rule: payload.logic_rule,
      metadata: payload.metadata || {},
    });
    if (!changed) needsSync = true;
  } else if (type === "edge_deleted") {
    const signature = edgeSignatureFromParts(
      payload.from,
      payload.to,
      payload.relation_type,
      payload.direction || "directed"
    );
    const before = edges.length;
    const nextEdges = edges.filter((edge) => edgeSignature(edge) !== signature);
    changed = before !== nextEdges.length;
    if (changed) {
      edges.splice(0, edges.length, ...nextEdges);
    }
  } else if (type === "edge_weight_feedback") {
    const signature = edgeSignatureFromParts(
      payload.from,
      payload.to,
      payload.relation_type,
      payload.direction || "directed"
    );
    const idx = edges.findIndex((edge) => edgeSignature(edge) === signature);
    if (idx >= 0) {
      const next = { ...edges[idx], weight: Number(payload.new_weight || edges[idx].weight || 0) };
      edges[idx] = next;
      changed = true;
    } else {
      needsSync = true;
    }
  } else if (
    type === "state_propagation_step" ||
    type === "simulation_started" ||
    type === "simulation_phase" ||
    type === "simulation_infer_round" ||
    type === "simulation_completed" ||
    type === "load_snapshot" ||
    type === "clear_graph"
  ) {
    needsSync = true;
  }

  if (!changed) {
    return { nextPayload: null, needsSync };
  }

  nodes.sort((a, b) => Number(a?.id || 0) - Number(b?.id || 0));
  const nextSnapshot = { nodes, edges };
  return {
    nextPayload: buildSnapshotPayload(nextSnapshot),
    needsSync,
  };
}

function timelineStep(event, label, state = "done") {
  const timestamp = Number(event?.timestamp || 0);
  const timeLabel = Number.isFinite(timestamp) && timestamp > 0
    ? new Date(timestamp * 1000).toLocaleTimeString()
    : "";
  return {
    id: Number(event?.id || 0),
    label,
    state,
    timeLabel,
  };
}

function deriveSimulationTimeline(events) {
  const rows = Array.isArray(events) ? events : [];
  let startIndex = -1;
  for (let idx = rows.length - 1; idx >= 0; idx -= 1) {
    if (String(rows[idx]?.event_type || "") === "simulation_started") {
      startIndex = idx;
      break;
    }
  }
  if (startIndex < 0) {
    return { status: "idle", progress: 0, steps: [] };
  }

  const runRows = rows.slice(startIndex);
  const startEvent = runRows[0] || {};
  const steps = [timelineStep(startEvent, "start")];
  const inferSteps = [];
  const propagationSteps = [];
  let recursiveStep = null;
  let completedStep = null;

  for (const event of runRows.slice(1)) {
    const type = String(event?.event_type || "");
    if (type === "simulation_phase") {
      const phaseName = String(event?.payload?.phase || "phase").replace(/_/g, " ");
      recursiveStep = timelineStep(event, phaseName);
    } else if (type === "simulation_infer_round") {
      const round = Number(event?.payload?.round || 0);
      const roundsTotal = Number(event?.payload?.rounds_total || 0);
      inferSteps.push(timelineStep(event, `infer ${round}/${Math.max(1, roundsTotal)}`));
    } else if (type === "state_propagation_step") {
      const step = Number(event?.payload?.step || 0);
      const total = Number(event?.payload?.steps_total || 0);
      propagationSteps.push(timelineStep(event, `propagation ${step}/${Math.max(1, total)}`));
    } else if (type === "simulation_completed") {
      completedStep = timelineStep(event, "completed");
      break;
    }
  }

  if (recursiveStep) {
    steps.push(recursiveStep);
  }
  steps.push(...inferSteps);
  steps.push(...propagationSteps);
  if (completedStep) {
    steps.push(completedStep);
  }

  const expectedInfer = Math.max(1, Number(startEvent?.payload?.infer_rounds || 1));
  const expectedPropagation = Math.max(1, Number(startEvent?.payload?.propagation_steps || 1));
  const expectedSteps = 2 + expectedInfer + expectedPropagation + 1;
  const completed = Boolean(completedStep);
  const status = completed ? "completed" : "running";
  const progress = completed
    ? 100
    : Math.max(8, Math.min(96, Math.round((steps.length / Math.max(1, expectedSteps)) * 100)));

  if (!completed && steps.length) {
    steps[steps.length - 1] = { ...steps[steps.length - 1], state: "active" };
  }

  return {
    status,
    progress,
    steps,
  };
}

function deriveGraphExploration(snapshot, targetNodeId) {
  const nodes = Array.isArray(snapshot?.nodes) ? snapshot.nodes : [];
  const edges = Array.isArray(snapshot?.edges) ? snapshot.edges : [];
  const target = Number(targetNodeId);
  const nodeIds = nodes
    .map((node) => Number(node?.id))
    .filter((id) => Number.isFinite(id));
  const nodeIdSet = new Set(nodeIds);
  if (!Number.isFinite(target) || !nodeIdSet.has(target)) {
    return {
      hasTarget: false,
      targetNodeId: null,
      rootNodeIds: [],
      pathNodeIds: [],
      pathEdgeSigs: [],
      ancestorNodeIds: [],
      ancestorEdgeSigs: [],
      descendantNodeIds: [],
      descendantEdgeSigs: [],
      dependencyNodeIds: [],
      dependencyEdgeSigs: [],
      focusNodeIds: [],
      focusEdgeSigs: [],
    };
  }

  const outgoing = new Map();
  const incoming = new Map();
  const incomingCount = new Map();
  for (const nodeId of nodeIds) {
    incomingCount.set(nodeId, 0);
  }

  const pushLink = (graph, key, entry) => {
    const list = graph.get(key);
    if (list) {
      list.push(entry);
    } else {
      graph.set(key, [entry]);
    }
  };

  for (const edge of edges) {
    const from = Number(edge?.from);
    const to = Number(edge?.to);
    if (!Number.isFinite(from) || !Number.isFinite(to)) continue;
    if (!nodeIdSet.has(from) || !nodeIdSet.has(to)) continue;
    const sig = edgeSignature(edge);
    const direction = String(edge?.direction || "directed");
    const direct = { from, to, sig };
    pushLink(outgoing, from, direct);
    pushLink(incoming, to, direct);
    incomingCount.set(to, Number(incomingCount.get(to) || 0) + 1);
    if (direction === "undirected") {
      const reverse = { from: to, to: from, sig };
      pushLink(outgoing, to, reverse);
      pushLink(incoming, from, reverse);
      incomingCount.set(from, Number(incomingCount.get(from) || 0) + 1);
    }
  }

  let rootNodeIds = nodeIds.filter((id) => Number(incomingCount.get(id) || 0) === 0);
  if (!rootNodeIds.length) {
    let minIncoming = Number.POSITIVE_INFINITY;
    for (const nodeId of nodeIds) {
      minIncoming = Math.min(minIncoming, Number(incomingCount.get(nodeId) || 0));
    }
    rootNodeIds = nodeIds.filter((id) => Number(incomingCount.get(id) || 0) === minIncoming);
  }

  const queue = [];
  const visited = new Set();
  const previousNode = new Map();
  const previousEdge = new Map();
  for (const rootId of rootNodeIds) {
    if (visited.has(rootId)) continue;
    visited.add(rootId);
    queue.push(rootId);
  }

  while (queue.length) {
    const current = Number(queue.shift());
    if (current === target) break;
    for (const link of outgoing.get(current) || []) {
      const next = Number(link.to);
      if (visited.has(next)) continue;
      visited.add(next);
      previousNode.set(next, current);
      previousEdge.set(next, String(link.sig));
      queue.push(next);
    }
  }

  const pathNodeIds = [];
  const pathEdgeSigs = [];
  if (visited.has(target)) {
    let cursor = target;
    pathNodeIds.push(cursor);
    while (previousNode.has(cursor)) {
      pathEdgeSigs.push(String(previousEdge.get(cursor) || ""));
      cursor = Number(previousNode.get(cursor));
      pathNodeIds.push(cursor);
    }
    pathNodeIds.reverse();
    pathEdgeSigs.reverse();
  } else {
    pathNodeIds.push(target);
  }

  const ancestorNodeIds = [];
  const ancestorNodeSet = new Set();
  const ancestorEdgeSet = new Set();
  const incomingQueue = [target];
  const incomingSeen = new Set([target]);
  while (incomingQueue.length) {
    const current = Number(incomingQueue.shift());
    for (const link of incoming.get(current) || []) {
      const parent = Number(link.from);
      ancestorEdgeSet.add(String(link.sig));
      if (incomingSeen.has(parent)) continue;
      incomingSeen.add(parent);
      if (parent !== target) {
        ancestorNodeSet.add(parent);
        ancestorNodeIds.push(parent);
      }
      incomingQueue.push(parent);
    }
  }

  const descendantNodeIds = [];
  const descendantNodeSet = new Set();
  const descendantEdgeSet = new Set();
  const outgoingQueue = [target];
  const outgoingSeen = new Set([target]);
  while (outgoingQueue.length) {
    const current = Number(outgoingQueue.shift());
    for (const link of outgoing.get(current) || []) {
      const child = Number(link.to);
      descendantEdgeSet.add(String(link.sig));
      if (outgoingSeen.has(child)) continue;
      outgoingSeen.add(child);
      if (child !== target) {
        descendantNodeSet.add(child);
        descendantNodeIds.push(child);
      }
      outgoingQueue.push(child);
    }
  }

  const dependencyNodeSet = new Set([...ancestorNodeSet, ...descendantNodeSet]);
  const dependencyEdgeSet = new Set([...ancestorEdgeSet, ...descendantEdgeSet]);
  const focusNodeSet = new Set([target, ...pathNodeIds, ...dependencyNodeSet]);
  const focusEdgeSet = new Set([...pathEdgeSigs, ...dependencyEdgeSet]);

  return {
    hasTarget: true,
    targetNodeId: target,
    rootNodeIds,
    pathNodeIds,
    pathEdgeSigs,
    ancestorNodeIds,
    ancestorEdgeSigs: Array.from(ancestorEdgeSet),
    descendantNodeIds,
    descendantEdgeSigs: Array.from(descendantEdgeSet),
    dependencyNodeIds: Array.from(dependencyNodeSet),
    dependencyEdgeSigs: Array.from(dependencyEdgeSet),
    focusNodeIds: Array.from(focusNodeSet),
    focusEdgeSigs: Array.from(focusEdgeSet),
  };
}

function deriveTraceVariants(snapshot, targetNodeId, limit = 5) {
  const nodes = Array.isArray(snapshot?.nodes) ? snapshot.nodes : [];
  const edges = Array.isArray(snapshot?.edges) ? snapshot.edges : [];
  const target = Number(targetNodeId);
  if (!Number.isFinite(target)) return [];

  const nodeIds = nodes
    .map((node) => Number(node?.id))
    .filter((id) => Number.isFinite(id));
  const nodeIdSet = new Set(nodeIds);
  if (!nodeIdSet.has(target)) return [];

  const incoming = new Map();
  const incomingCount = new Map();
  const edgeWeightBySig = new Map();
  for (const id of nodeIds) {
    incomingCount.set(id, 0);
  }

  const pushIncoming = (to, row) => {
    const list = incoming.get(to);
    if (list) {
      list.push(row);
    } else {
      incoming.set(to, [row]);
    }
  };

  for (const edge of edges) {
    const from = Number(edge?.from);
    const to = Number(edge?.to);
    if (!Number.isFinite(from) || !Number.isFinite(to)) continue;
    if (!nodeIdSet.has(from) || !nodeIdSet.has(to)) continue;
    const sig = edgeSignature(edge);
    const weight = Number(edge?.weight || 0);
    edgeWeightBySig.set(sig, weight);
    const direction = String(edge?.direction || "directed");

    pushIncoming(to, { from, to, sig });
    incomingCount.set(to, Number(incomingCount.get(to) || 0) + 1);
    if (direction === "undirected") {
      pushIncoming(from, { from: to, to: from, sig });
      incomingCount.set(from, Number(incomingCount.get(from) || 0) + 1);
    }
  }

  let rootNodeIds = nodeIds.filter((id) => Number(incomingCount.get(id) || 0) === 0);
  if (!rootNodeIds.length) {
    let minIncoming = Number.POSITIVE_INFINITY;
    for (const id of nodeIds) {
      minIncoming = Math.min(minIncoming, Number(incomingCount.get(id) || 0));
    }
    rootNodeIds = nodeIds.filter((id) => Number(incomingCount.get(id) || 0) === minIncoming);
  }
  const rootSet = new Set(rootNodeIds);

  const maxDepth = Math.max(4, Math.min(15, Math.ceil(Math.sqrt(nodeIds.length)) + 5));
  const maxCandidates = Math.max(16, Math.min(120, limit * 14));
  const stack = [
    {
      node: target,
      nodeTrail: [target],
      edgeTrail: [],
      visited: new Set([target]),
    },
  ];
  const raw = [];
  const dedupe = new Set();

  while (stack.length && raw.length < maxCandidates) {
    const state = stack.pop();
    if (!state) break;
    const node = Number(state.node);
    if (rootSet.has(node)) {
      const nodeTrail = [...state.nodeTrail].reverse();
      const edgeTrail = [...state.edgeTrail].reverse();
      const key = `${nodeTrail.join(">")}|${edgeTrail.join(">")}`;
      if (!dedupe.has(key)) {
        dedupe.add(key);
        const avgWeight =
          edgeTrail.reduce((sum, sig) => sum + Number(edgeWeightBySig.get(sig) || 0), 0) /
          Math.max(1, edgeTrail.length);
        const scoreRaw = Math.max(0, Math.min(1, avgWeight - (nodeTrail.length - 1) * 0.03 + 0.3));
        raw.push({
          nodeIds: nodeTrail,
          edgeSigs: edgeTrail,
          rootNodeId: nodeTrail[0],
          depth: Math.max(0, nodeTrail.length - 1),
          avgWeight,
          score: Number(scoreRaw.toFixed(3)),
        });
      }
      continue;
    }
    if (state.nodeTrail.length >= maxDepth) {
      continue;
    }
    for (const link of incoming.get(node) || []) {
      const parent = Number(link.from);
      if (state.visited.has(parent)) continue;
      const nextVisited = new Set(state.visited);
      nextVisited.add(parent);
      stack.push({
        node: parent,
        nodeTrail: [...state.nodeTrail, parent],
        edgeTrail: [...state.edgeTrail, String(link.sig)],
        visited: nextVisited,
      });
    }
  }

  if (!raw.length) {
    return [
      {
        id: "trace_1",
        nodeIds: [target],
        edgeSigs: [],
        rootNodeId: target,
        depth: 0,
        avgWeight: 0,
        score: 0,
      },
    ];
  }

  const sorted = raw
    .sort((a, b) => {
      if (a.depth !== b.depth) return a.depth - b.depth;
      if (a.score !== b.score) return b.score - a.score;
      return b.avgWeight - a.avgWeight;
    })
    .slice(0, Math.max(1, limit))
    .map((row, index) => ({
      ...row,
      id: `trace_${index + 1}`,
    }));

  return sorted;
}

function deriveBranchInsights(snapshot, exploration, trace, nodeNameById) {
  const nodes = Array.isArray(snapshot?.nodes) ? snapshot.nodes : [];
  const edges = Array.isArray(snapshot?.edges) ? snapshot.edges : [];
  const hasTarget = Boolean(exploration?.hasTarget);
  const targetNode = Number(exploration?.targetNodeId);

  let focusIds = [];
  if (hasTarget) {
    const base = Array.isArray(exploration?.focusNodeIds) ? exploration.focusNodeIds : [];
    focusIds = base.map((id) => Number(id)).filter((id) => Number.isFinite(id));
  } else {
    focusIds = nodes.map((node) => Number(node?.id)).filter((id) => Number.isFinite(id));
  }
  if (trace && Array.isArray(trace.nodeIds)) {
    for (const id of trace.nodeIds) {
      const n = Number(id);
      if (Number.isFinite(n)) {
        focusIds.push(n);
      }
    }
  }
  const focusSet = new Set(focusIds);
  const branchNodes = nodes.filter((node) => focusSet.has(Number(node?.id)));
  const branchNodeSet = new Set(branchNodes.map((node) => Number(node?.id)));
  const branchEdges = edges.filter(
    (edge) => branchNodeSet.has(Number(edge?.from)) && branchNodeSet.has(Number(edge?.to))
  );

  const relationCounts = {};
  const nodeTypeCounts = {};
  const degreeByNodeId = {};
  let totalEdgeWeight = 0;

  for (const node of branchNodes) {
    const typeKey = String(node?.type || "generic");
    nodeTypeCounts[typeKey] = Number(nodeTypeCounts[typeKey] || 0) + 1;
    degreeByNodeId[Number(node?.id)] = 0;
  }
  for (const edge of branchEdges) {
    const relKey = String(edge?.relation_type || "unknown");
    relationCounts[relKey] = Number(relationCounts[relKey] || 0) + 1;
    const from = Number(edge?.from);
    const to = Number(edge?.to);
    degreeByNodeId[from] = Number(degreeByNodeId[from] || 0) + 1;
    degreeByNodeId[to] = Number(degreeByNodeId[to] || 0) + 1;
    totalEdgeWeight += Number(edge?.weight || 0);
  }

  const avgWeight = branchEdges.length ? totalEdgeWeight / branchEdges.length : 0;
  const topRelations = Object.entries(relationCounts)
    .map(([name, count]) => ({ name, count: Number(count || 0) }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 6);
  const topNodeTypes = Object.entries(nodeTypeCounts)
    .map(([name, count]) => ({ name, count: Number(count || 0) }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 6);
  const topNodes = branchNodes
    .map((node) => {
      const id = Number(node?.id);
      return {
        id,
        label: nodeNameById.get(id) || `#${id}`,
        type: String(node?.type || "generic"),
        degree: Number(degreeByNodeId[id] || 0),
      };
    })
    .sort((a, b) => {
      if (a.degree !== b.degree) return b.degree - a.degree;
      return a.id - b.id;
    })
    .slice(0, 8);
  const topEdges = branchEdges
    .map((edge) => ({
      signature: edgeSignature(edge),
      label: `${nodeNameById.get(Number(edge?.from)) || `#${Number(edge?.from)}`} -> ${
        nodeNameById.get(Number(edge?.to)) || `#${Number(edge?.to)}`
      }`,
      relation: String(edge?.relation_type || "relation"),
      weight: Number(edge?.weight || 0),
    }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 8);

  const lifehacks = [];
  if (!hasTarget) {
    lifehacks.push("Pick a node to see branch-level visual analytics instead of whole-graph aggregate.");
  } else {
    lifehacks.push(`Target branch node: ${nodeNameById.get(targetNode) || `#${targetNode}`}.`);
  }
  if (Number(exploration?.ancestorNodeIds?.length || 0) > 8) {
    lifehacks.push("Too many prerequisites: split this branch into milestone sub-branches.");
  }
  if (Number(exploration?.descendantNodeIds?.length || 0) === 0 && hasTarget) {
    lifehacks.push("No downstream actions found: add at least one execution node linked from this target.");
  }
  if (avgWeight < 0.45 && branchEdges.length) {
    lifehacks.push("Branch confidence is weak: reinforce critical edges with evidence and higher weights.");
  }
  if (branchNodes.some((node) => String(node?.type || "").includes("hallucination"))) {
    lifehacks.push("Hallucination artifacts detected: run hunter check before merging this branch into plan.");
  }
  if (branchEdges.length && !branchEdges.some((edge) => String(edge?.logic_rule || "").trim())) {
    lifehacks.push("Edge logic is missing: annotate edges with explicit logic_rule for clearer traceability.");
  }
  if (lifehacks.length < 3) {
    lifehacks.push("Use the list view to prioritize top nodes by degree, then refine those nodes first.");
  }

  return {
    hasTarget,
    targetNode,
    nodeCount: branchNodes.length,
    edgeCount: branchEdges.length,
    avgWeight,
    topRelations,
    topNodeTypes,
    topNodes,
    topEdges,
    lifehacks: lifehacks.slice(0, 8),
  };
}

function colorForNodeType(type) {
  const key = String(type || "generic").toLowerCase();
  if (key === "human" || key === "client_session") return "#1f6feb";
  if (key === "company" || key === "domain") return "#1f8a5f";
  if (key === "concept") return "#935f1f";
  if (key === "ip_address" || key === "network_profile") return "#7f4f97";
  if (key === "browser" || key === "operating_system") return "#495057";
  if (key === GRAPH_NODE_TYPE_PERSONAL_TREE_ROOT) return "#0f766e";
  if (key === GRAPH_NODE_TYPE_THOUGHT_SESSION) return "#b45309";
  if (key === GRAPH_NODE_TYPE_THOUGHT_SUMMARY) return "#1d4ed8";
  if (key === GRAPH_NODE_TYPE_THOUGHT_POINT) return "#15803d";
  if (key === GRAPH_NODE_TYPE_PERSONAL_NOTE) return "#b91c1c";
  if (key === GRAPH_NODE_TYPE_SOURCE_REFERENCE) return "#334155";
  return "#5f6f85";
}

function GraphCanvas({
  snapshot,
  t,
  selectedNodeId,
  selectedEdgeSig,
  tracePath,
  edgeEffectsBySig,
  onSelectNode,
  onSelectEdge,
}) {
  const nodes = Array.isArray(snapshot?.nodes) ? snapshot.nodes : [];
  const edges = Array.isArray(snapshot?.edges) ? snapshot.edges : [];
  const [hoveredNodeId, setHoveredNodeId] = useState(null);

  useEffect(() => {
    if (hoveredNodeId == null) return;
    const hasNode = nodes.some((node) => Number(node?.id) === Number(hoveredNodeId));
    if (!hasNode) {
      setHoveredNodeId(null);
    }
  }, [nodes, hoveredNodeId]);

  const focusNodeId = useMemo(() => {
    if (hoveredNodeId != null && nodes.some((node) => Number(node?.id) === Number(hoveredNodeId))) {
      return Number(hoveredNodeId);
    }
    const selected = Number(selectedNodeId);
    return Number.isFinite(selected) ? selected : null;
  }, [hoveredNodeId, nodes, selectedNodeId]);

  const exploration = useMemo(
    () => deriveGraphExploration({ nodes, edges }, focusNodeId),
    [nodes, edges, focusNodeId]
  );
  const useTraceOverride = useMemo(() => {
    if (hoveredNodeId != null) return false;
    const targetNodeId = Number(tracePath?.targetNodeId);
    if (!Number.isFinite(targetNodeId) || Number(focusNodeId) !== targetNodeId) return false;
    return Array.isArray(tracePath?.nodeIds) && tracePath.nodeIds.length > 1;
  }, [hoveredNodeId, focusNodeId, tracePath]);
  const effectivePathNodeIds = useMemo(() => {
    if (useTraceOverride) {
      return (tracePath?.nodeIds || []).map((id) => Number(id)).filter((id) => Number.isFinite(id));
    }
    return exploration.pathNodeIds || [];
  }, [useTraceOverride, tracePath, exploration.pathNodeIds]);
  const effectivePathEdgeSigs = useMemo(() => {
    if (useTraceOverride) {
      return (tracePath?.edgeSigs || []).map((sig) => String(sig));
    }
    return exploration.pathEdgeSigs || [];
  }, [useTraceOverride, tracePath, exploration.pathEdgeSigs]);
  const pathNodeSet = useMemo(() => new Set(effectivePathNodeIds || []), [effectivePathNodeIds]);
  const pathEdgeSet = useMemo(() => new Set(effectivePathEdgeSigs || []), [effectivePathEdgeSigs]);
  const ancestorNodeSet = useMemo(
    () => new Set(exploration.ancestorNodeIds || []),
    [exploration.ancestorNodeIds]
  );
  const ancestorEdgeSet = useMemo(
    () => new Set(exploration.ancestorEdgeSigs || []),
    [exploration.ancestorEdgeSigs]
  );
  const descendantNodeSet = useMemo(
    () => new Set(exploration.descendantNodeIds || []),
    [exploration.descendantNodeIds]
  );
  const descendantEdgeSet = useMemo(
    () => new Set(exploration.descendantEdgeSigs || []),
    [exploration.descendantEdgeSigs]
  );

  const layout = useMemo(() => {
    const width = 1080;
    const height = 620;
    const byId = new Map();

    if (!nodes.length) {
      return { width, height, byId };
    }

    const centerX = width / 2;
    const centerY = height / 2;
    const baseRadius = Math.max(180, Math.min(260, 120 + nodes.length * 6));
    const pointMap = new Map();

    nodes.forEach((node, index) => {
      const angle = (2 * Math.PI * index) / Math.max(1, nodes.length);
      pointMap.set(node.id, {
        x: centerX + baseRadius * Math.cos(angle),
        y: centerY + baseRadius * Math.sin(angle),
        vx: 0,
        vy: 0,
      });
    });

    const edgeList = edges.filter(
      (edge) => pointMap.has(edge.from) && pointMap.has(edge.to)
    );

    for (let iter = 0; iter < 240; iter += 1) {
      for (let i = 0; i < nodes.length; i += 1) {
        const a = pointMap.get(nodes[i].id);
        if (!a) continue;
        for (let j = i + 1; j < nodes.length; j += 1) {
          const b = pointMap.get(nodes[j].id);
          if (!b) continue;
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const dist2 = Math.max(90, dx * dx + dy * dy);
          const force = 5200 / dist2;
          const dist = Math.sqrt(dist2);
          const ux = dx / dist;
          const uy = dy / dist;
          a.vx += ux * force;
          a.vy += uy * force;
          b.vx -= ux * force;
          b.vy -= uy * force;
        }
      }

      for (const edge of edgeList) {
        const a = pointMap.get(edge.from);
        const b = pointMap.get(edge.to);
        if (!a || !b) continue;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy));
        const desired = 120;
        const k = 0.015 * (Number(edge.weight || 0.4) + 0.4);
        const force = k * (dist - desired);
        const ux = dx / dist;
        const uy = dy / dist;
        a.vx += ux * force;
        a.vy += uy * force;
        b.vx -= ux * force;
        b.vy -= uy * force;
      }

      for (const node of nodes) {
        const p = pointMap.get(node.id);
        if (!p) continue;
        p.vx += (centerX - p.x) * 0.0009;
        p.vy += (centerY - p.y) * 0.0009;
        p.vx *= 0.83;
        p.vy *= 0.83;
        p.x += p.vx * 0.9;
        p.y += p.vy * 0.9;
        p.x = Math.max(54, Math.min(width - 54, p.x));
        p.y = Math.max(54, Math.min(height - 54, p.y));
      }
    }

    nodes.forEach((node) => {
      const p = pointMap.get(node.id);
      if (!p) return;
      byId.set(node.id, { x: p.x, y: p.y });
    });

    return { width, height, byId };
  }, [nodes, edges]);

  if (!nodes.length) {
    return <div className="empty-canvas">{t("graph_empty")}</div>;
  }

  const hasFocus = Boolean(exploration.hasTarget);

  return (
    <svg
      viewBox={`0 0 ${layout.width} ${layout.height}`}
      className="graph-canvas"
      onClick={() => {
        setHoveredNodeId(null);
        onSelectNode(null);
        onSelectEdge(null);
      }}
    >
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
          <polygon points="0 0, 10 5, 0 10" fill="#285989" />
        </marker>
        <marker id="arrow-active" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
          <polygon points="0 0, 10 5, 0 10" fill="#c44a11" />
        </marker>
        <marker id="arrow-path" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
          <polygon points="0 0, 10 5, 0 10" fill="#cb5418" />
        </marker>
        <marker id="arrow-ancestor" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
          <polygon points="0 0, 10 5, 0 10" fill="#0d765f" />
        </marker>
        <marker id="arrow-descendant" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
          <polygon points="0 0, 10 5, 0 10" fill="#2b74b0" />
        </marker>
        <filter id="node-glow" x="-40%" y="-40%" width="180%" height="180%">
          <feDropShadow dx="0" dy="1.5" stdDeviation="2.2" floodColor="#12395e" floodOpacity="0.2" />
        </filter>
        <pattern id="grid" width="26" height="26" patternUnits="userSpaceOnUse">
          <path d="M 26 0 L 0 0 0 26" fill="none" stroke="#d7e6f5" strokeWidth="1" />
        </pattern>
      </defs>

      <rect x="0" y="0" width={layout.width} height={layout.height} fill="url(#grid)" />

      {edges.map((edge, idx) => {
        const a = layout.byId.get(edge.from);
        const b = layout.byId.get(edge.to);
        if (!a || !b) {
          return null;
        }

        const sig = edgeSignature(edge);
        const liveEffect = edgeEffectsBySig && typeof edgeEffectsBySig === "object" ? edgeEffectsBySig[sig] : null;
        const active = selectedEdgeSig === sig;
        const inPath = pathEdgeSet.has(sig);
        const inAncestors = ancestorEdgeSet.has(sig);
        const inDescendants = descendantEdgeSet.has(sig);
        const related = inPath || inAncestors || inDescendants;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const length = Math.max(1, Math.sqrt(dx * dx + dy * dy));
        const nx = -dy / length;
        const ny = dx / length;
        const curve = active ? 18 : 10;
        const cx = (a.x + b.x) / 2 + nx * curve;
        const cy = (a.y + b.y) / 2 + ny * curve;
        const path = `M ${a.x} ${a.y} Q ${cx} ${cy} ${b.x} ${b.y}`;
        const midX = (a.x + 2 * cx + b.x) / 4;
        const midY = (a.y + 2 * cy + b.y) / 4;
        let stroke = "#2e6ca8";
        let strokeOpacity = hasFocus ? 0.22 : 0.78;
        if (inDescendants) {
          stroke = "#2b74b0";
          strokeOpacity = hasFocus ? 0.62 : 0.78;
        }
        if (inAncestors) {
          stroke = "#0d765f";
          strokeOpacity = hasFocus ? 0.74 : 0.82;
        }
        if (inPath) {
          stroke = "#cb5418";
          strokeOpacity = 0.93;
        }
        if (active) {
          stroke = "#c44a11";
          strokeOpacity = 0.98;
        }
        const markerEnd =
          edge.direction === "undirected"
            ? undefined
            : active
              ? "url(#arrow-active)"
              : inPath
                ? "url(#arrow-path)"
                : inAncestors
                  ? "url(#arrow-ancestor)"
                  : inDescendants
                    ? "url(#arrow-descendant)"
                    : "url(#arrow)";
        const edgeLabelColor = active || inPath ? "#8b380f" : inAncestors ? "#0d6250" : inDescendants ? "#215f90" : "#204e78";
        const edgeLabelOpacity = hasFocus && !related && !active ? 0.42 : 1;
        const edgeClassName = [
          "edge-line",
          inPath ? "edge-line-path" : "",
          inAncestors ? "edge-line-ancestor" : "",
          inDescendants ? "edge-line-descendant" : "",
          active ? "edge-line-active" : "",
          liveEffect?.kind === "added" ? "edge-live-added" : "",
          liveEffect?.kind === "updated" ? "edge-live-updated" : "",
          hasFocus && !related && !active ? "edge-line-muted" : "",
        ]
          .filter(Boolean)
          .join(" ");

        return (
          <g
            key={`edge-${sig}-${Number(liveEffect?.token || 0)}-${idx}`}
            className={`edge-group ${active ? "active" : ""}`}
            onClick={(event) => {
              event.stopPropagation();
              onSelectEdge(sig);
              onSelectNode(null);
            }}
          >
            <path
              d={path}
              fill="none"
              className={edgeClassName}
              stroke={stroke}
              strokeOpacity={strokeOpacity}
              strokeWidth={1.25 + Math.max(0, Number(edge.weight || 0)) * (active ? 3.1 : 2.2)}
              markerEnd={markerEnd}
            />
            {liveEffect && (
              <path
                d={path}
                fill="none"
                pointerEvents="none"
                className={`edge-live-overlay edge-live-overlay-${liveEffect.kind || "updated"}`}
                stroke={
                  liveEffect.kind === "added"
                    ? "#1faa74"
                    : "#da6418"
                }
                strokeWidth={2.8 + Math.max(0, Number(edge.weight || 0)) * 3.4}
                markerEnd={markerEnd}
              />
            )}
            <text x={midX} y={midY - 5} className="edge-label" fill={edgeLabelColor} opacity={edgeLabelOpacity}>
              {edge.relation_type} · {Number(edge.weight || 0).toFixed(2)}
            </text>
          </g>
        );
      })}

      {nodes.map((node) => {
        const p = layout.byId.get(node.id);
        if (!p) {
          return null;
        }

        const nodeId = Number(node.id);
        const selected = Number(selectedNodeId) === nodeId;
        const focused = Number(focusNodeId) === nodeId;
        const onPath = pathNodeSet.has(nodeId);
        const ancestor = ancestorNodeSet.has(nodeId);
        const descendant = descendantNodeSet.has(nodeId);
        const related = focused || onPath || ancestor || descendant;
        const muted = hasFocus && !related && !selected;
        let fill = "#f7fbff";
        let stroke = colorForNodeType(node.type);
        let strokeWidth = 2;
        let textColor = "#113f69";
        if (descendant) {
          fill = "#eaf3ff";
          stroke = "#2b74b0";
          textColor = "#1f5681";
        }
        if (ancestor) {
          fill = "#e8f7f1";
          stroke = "#0d765f";
          textColor = "#0d5948";
        }
        if (onPath) {
          fill = "#fff2e7";
          stroke = "#cb5418";
          textColor = "#7f340e";
        }
        if (focused) {
          fill = "#ffe7d5";
          stroke = "#b44712";
          strokeWidth = 3;
          textColor = "#7f310e";
        }
        if (selected) {
          fill = "#ffe2cc";
          stroke = "#9f3f12";
          strokeWidth = 3.2;
          textColor = "#6e2c0c";
        }
        if (muted) {
          fill = "#f2f7fc";
          stroke = "#a4b6c8";
          textColor = "#5f7487";
        }
        const title = nodeLabel(node);

        return (
          <g
            key={`node-${nodeId}`}
            className={`node-group ${selected ? "active" : ""}`}
            onClick={(event) => {
              event.stopPropagation();
              onSelectNode(nodeId);
              onSelectEdge(null);
              setHoveredNodeId(null);
            }}
            onMouseEnter={() => setHoveredNodeId(nodeId)}
            onMouseLeave={() => setHoveredNodeId((current) => (Number(current) === nodeId ? null : current))}
            opacity={muted ? 0.46 : 1}
            style={{ cursor: "pointer" }}
          >
            {(selected || focused || onPath) && (
              <circle
                cx={p.x}
                cy={p.y}
                r={selected ? 41 : focused ? 38 : 35}
                fill="none"
                className={`node-aura ${selected ? "node-aura-selected" : focused ? "node-aura-focus" : "node-aura-path"}`}
              />
            )}
            <circle
              cx={p.x}
              cy={p.y}
              r={selected || focused ? 33 : 29}
              fill={fill}
              stroke={stroke}
              strokeWidth={strokeWidth}
              filter="url(#node-glow)"
            />
            <text x={p.x} y={p.y - 5} textAnchor="middle" className="node-id" fill={textColor}>
              {title.slice(0, 18)}
            </text>
            <text x={p.x} y={p.y + 12} textAnchor="middle" className="node-type" fill={textColor}>
              {node.type}
            </text>
            <title>{`${title} (#${node.id})`}</title>
          </g>
        );
      })}
    </svg>
  );
}

export default function App() {
  const [busy, setBusy] = useState(false);
  const [fatalError, setFatalError] = useState("");
  const [uiLanguage, setUiLanguage] = useState(() => detectInitialLanguage());
  const [currentPage, setCurrentPage] = useState(() => getInitialPage());

  const [modules, setModules] = useState([]);
  const [modelAdvisors, setModelAdvisors] = useState(null);
  const [snapshotPayload, setSnapshotPayload] = useState({
    snapshot: { nodes: [], edges: [] },
    metrics: {},
  });
  const [events, setEvents] = useState([]);
  const [nodeTypes, setNodeTypes] = useState(["generic", "human", "company"]);
  const [logLines, setLogLines] = useState([]);

  const [nodeType, setNodeType] = useState("human");
  const [nodeAttributesText, setNodeAttributesText] = useState("{}");
  const [nodeStateText, setNodeStateText] = useState('{"influence": 0.5}');
  const [humanFirstName, setHumanFirstName] = useState("");
  const [humanLastName, setHumanLastName] = useState("");
  const [humanBio, setHumanBio] = useState("I build autonomous systems and optimize analytical workflows.");
  const [humanProfileText, setHumanProfileText] = useState("");
  const [humanEmploymentText, setHumanEmploymentText] = useState("");
  const [humanEmploymentJsonText, setHumanEmploymentJsonText] = useState("[]");

  const [companyName, setCompanyName] = useState("Vector Dynamics");
  const [companyIndustry, setCompanyIndustry] = useState("AI Infrastructure");
  const [companyDescription, setCompanyDescription] = useState("Autonomous graph platform.");

  const [edgeFrom, setEdgeFrom] = useState("");
  const [edgeTo, setEdgeTo] = useState("");
  const [edgeRelation, setEdgeRelation] = useState("influences");
  const [edgeWeight, setEdgeWeight] = useState("0.7");
  const [edgeDirection, setEdgeDirection] = useState("directed");
  const [edgeLogicRule, setEdgeLogicRule] = useState("explicit");

  const [simSeedIds, setSimSeedIds] = useState("");
  const [simDepth, setSimDepth] = useState("2");
  const [simSteps, setSimSteps] = useState("3");
  const [simDamping, setSimDamping] = useState("0.15");
  const [simActivation, setSimActivation] = useState("tanh");
  const [simInferRounds, setSimInferRounds] = useState("1");
  const [lastSimulation, setLastSimulation] = useState(null);

  const [rewardEventId, setRewardEventId] = useState("");
  const [rewardValue, setRewardValue] = useState("0.8");
  const [rewardLr, setRewardLr] = useState("0.15");

  const [reinforceRelationType, setReinforceRelationType] = useState("influences");
  const [reinforceReward, setReinforceReward] = useState("0.3");
  const [reinforceLr, setReinforceLr] = useState("0.12");
  const [profileInputText, setProfileInputText] = useState("");
  const [profileEntityHint, setProfileEntityHint] = useState("human");
  const [profilePromptPreview, setProfilePromptPreview] = useState("");
  const [profileResult, setProfileResult] = useState(null);
  const [dbSchema, setDbSchema] = useState(null);
  const [clientProfile, setClientProfile] = useState(null);
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [selectedEdgeSig, setSelectedEdgeSig] = useState(null);
  const [selectedNodeAttributesText, setSelectedNodeAttributesText] = useState("{}");
  const [selectedNodeStateText, setSelectedNodeStateText] = useState("{}");
  const [selectedEdgeWeight, setSelectedEdgeWeight] = useState("0.7");
  const [selectedEdgeLogicRule, setSelectedEdgeLogicRule] = useState("explicit");
  const [selectedEdgeMetadataText, setSelectedEdgeMetadataText] = useState("{}");
  const [demoNarrative, setDemoNarrative] = useState("");
  const [dailyJournalText, setDailyJournalText] = useState("");
  const [dailyModeResult, setDailyModeResult] = useState(null);
  const [userNarrativeText, setUserNarrativeText] = useState("");
  const [userFearsText, setUserFearsText] = useState("");
  const [userDesiresText, setUserDesiresText] = useState("");
  const [userGoalsText, setUserGoalsText] = useState("");
  const [userPrinciplesText, setUserPrinciplesText] = useState("");
  const [userOpportunitiesText, setUserOpportunitiesText] = useState("");
  const [userAbilitiesText, setUserAbilitiesText] = useState("");
  const [userAccessText, setUserAccessText] = useState("");
  const [userKnowledgeText, setUserKnowledgeText] = useState("");
  const [userAssetsText, setUserAssetsText] = useState("");
  const [userGraphResult, setUserGraphResult] = useState(null);
  const [autorunsImportText, setAutorunsImportText] = useState("");
  const [autorunsImportResult, setAutorunsImportResult] = useState(null);
  const [debatePromptText, setDebatePromptText] = useState("");
  const [debateHypothesisCount, setDebateHypothesisCount] = useState("3");
  const [debateAttachGraph, setDebateAttachGraph] = useState(true);
  const [debateProposerRole, setDebateProposerRole] = useState("creative");
  const [debateCriticRole, setDebateCriticRole] = useState("analyst");
  const [debateJudgeRole, setDebateJudgeRole] = useState("planner");
  const [debateResult, setDebateResult] = useState(null);
  const [archiveChatMessageText, setArchiveChatMessageText] = useState("");
  const [archiveChatContextText, setArchiveChatContextText] = useState("");
  const [archiveChatModelPath, setArchiveChatModelPath] = useState("");
  const [archiveChatModelRole, setArchiveChatModelRole] = useState("general");
  const [archiveChatVerificationMode, setArchiveChatVerificationMode] = useState("strict");
  const [archiveChatAttachGraph, setArchiveChatAttachGraph] = useState(true);
  const [archiveChatResult, setArchiveChatResult] = useState(null);
  const [archiveChatMessages, setArchiveChatMessages] = useState([]);
  const [archiveReviewUpdatesText, setArchiveReviewUpdatesText] = useState("[]");
  const [archiveReviewResult, setArchiveReviewResult] = useState(null);
  const [hallucinationPromptText, setHallucinationPromptText] = useState("");
  const [hallucinationWrongAnswerText, setHallucinationWrongAnswerText] = useState("");
  const [hallucinationCorrectAnswerText, setHallucinationCorrectAnswerText] = useState("");
  const [hallucinationSourceText, setHallucinationSourceText] = useState("");
  const [hallucinationTagsText, setHallucinationTagsText] = useState("");
  const [hallucinationSeverity, setHallucinationSeverity] = useState("medium");
  const [hallucinationLlmAnswerText, setHallucinationLlmAnswerText] = useState("");
  const [hallucinationReportResult, setHallucinationReportResult] = useState(null);
  const [hallucinationCheckResult, setHallucinationCheckResult] = useState(null);
  const [personalizationDraft, setPersonalizationDraft] = useState(() => loadPersonalizationDraft());
  const [styleNodeIndex, setStyleNodeIndex] = useState(() => loadStyleNodeIndex());
  const [branchInsightView, setBranchInsightView] = useState("cards");
  const [styleNodeDraftName, setStyleNodeDraftName] = useState("");
  const [styleNodeDraftDescription, setStyleNodeDraftDescription] = useState("");
  const [styleNodeDraftVarsText, setStyleNodeDraftVarsText] = useState("{}");
  const [styleNodeSaveResult, setStyleNodeSaveResult] = useState(null);
  const [branchReportSummaryText, setBranchReportSummaryText] = useState("");
  const [branchReportTipsText, setBranchReportTipsText] = useState("");
  const [branchReportSaveResult, setBranchReportSaveResult] = useState(null);
  const [selectedRequestNodeId, setSelectedRequestNodeId] = useState(0);
  const [selectedPreferenceNodeId, setSelectedPreferenceNodeId] = useState(0);
  const [selectedTaskNodeId, setSelectedTaskNodeId] = useState(0);
  const [selectedRiskNodeId, setSelectedRiskNodeId] = useState(0);
  const [multitoolRequestDraft, setMultitoolRequestDraft] = useState({
    ...DEFAULT_MULTITOOL_REQUEST_DRAFT,
  });
  const [multitoolPreferenceDraft, setMultitoolPreferenceDraft] = useState({
    ...DEFAULT_MULTITOOL_PREFERENCE_DRAFT,
  });
  const [multitoolTaskDraft, setMultitoolTaskDraft] = useState({
    ...DEFAULT_MULTITOOL_TASK_DRAFT,
  });
  const [multitoolRiskDraft, setMultitoolRiskDraft] = useState({
    ...DEFAULT_MULTITOOL_RISK_DRAFT,
  });
  const [multitoolRequestSaveResult, setMultitoolRequestSaveResult] = useState(null);
  const [multitoolPreferenceSaveResult, setMultitoolPreferenceSaveResult] = useState(null);
  const [multitoolTaskSaveResult, setMultitoolTaskSaveResult] = useState(null);
  const [multitoolRiskSaveResult, setMultitoolRiskSaveResult] = useState(null);
  const [personalTreeIngestDraft, setPersonalTreeIngestDraft] = useState({
    ...DEFAULT_PERSONAL_TREE_INGEST_DRAFT,
  });
  const [personalTreeNoteDraft, setPersonalTreeNoteDraft] = useState({
    ...DEFAULT_PERSONAL_TREE_NOTE_DRAFT,
  });
  const [personalTreeFocusNodeId, setPersonalTreeFocusNodeId] = useState(0);
  const [personalTreeIngestResult, setPersonalTreeIngestResult] = useState(null);
  const [personalTreeNoteResult, setPersonalTreeNoteResult] = useState(null);
  const [personalTreeViewResult, setPersonalTreeViewResult] = useState(null);
  const [packagesDraft, setPackagesDraft] = useState({
    ...DEFAULT_PACKAGES_DRAFT,
  });
  const [packagesResult, setPackagesResult] = useState(null);
  const [memoryNamespaceDraft, setMemoryNamespaceDraft] = useState({
    ...DEFAULT_MEMORY_NAMESPACE_DRAFT,
  });
  const [memoryNamespaceApplyResult, setMemoryNamespaceApplyResult] = useState(null);
  const [memoryNamespaceViewResult, setMemoryNamespaceViewResult] = useState(null);
  const [graphRagDraft, setGraphRagDraft] = useState({
    ...DEFAULT_GRAPH_RAG_DRAFT,
  });
  const [graphRagResult, setGraphRagResult] = useState(null);
  const [contradictionScanDraft, setContradictionScanDraft] = useState({
    ...DEFAULT_CONTRADICTION_SCAN_DRAFT,
  });
  const [contradictionScanResult, setContradictionScanResult] = useState(null);
  const [taskRiskDraft, setTaskRiskDraft] = useState({
    ...DEFAULT_TASK_RISK_DRAFT,
  });
  const [taskRiskResult, setTaskRiskResult] = useState(null);
  const [timelineReplayDraft, setTimelineReplayDraft] = useState({
    ...DEFAULT_TIMELINE_REPLAY_DRAFT,
  });
  const [timelineReplayResult, setTimelineReplayResult] = useState(null);
  const [llmPolicyDraft, setLlmPolicyDraft] = useState({
    ...DEFAULT_LLM_POLICY_DRAFT,
  });
  const [llmPolicyResult, setLlmPolicyResult] = useState(null);
  const [qualityDraft, setQualityDraft] = useState({
    ...DEFAULT_QUALITY_DRAFT,
  });
  const [qualityResult, setQualityResult] = useState(null);
  const [backupDraft, setBackupDraft] = useState({
    ...DEFAULT_BACKUP_DRAFT,
  });
  const [backupCreateResult, setBackupCreateResult] = useState(null);
  const [backupRestoreResult, setBackupRestoreResult] = useState(null);
  const [auditDraft, setAuditDraft] = useState({
    ...DEFAULT_AUDIT_DRAFT,
  });
  const [auditResult, setAuditResult] = useState(null);
  const [selectedTraceIndex, setSelectedTraceIndex] = useState(0);
  const [edgeEffectsBySig, setEdgeEffectsBySig] = useState({});
  const [overviewSectionIndex, setOverviewSectionIndex] = useState(0);
  const [eventsPage, setEventsPage] = useState(0);
  const [nodesPage, setNodesPage] = useState(0);
  const [edgesPage, setEdgesPage] = useState(0);
  const [modulesPage, setModulesPage] = useState(0);
  const liveSyncTimerRef = useRef(null);
  const liveSyncInFlightRef = useRef(false);
  const liveReconnectRef = useRef(null);
  const edgeEffectTimersRef = useRef(new Map());
  const styleDraftSyncTokenRef = useRef("");
  const branchDraftSyncTokenRef = useRef("");
  const snapshotPayloadRef = useRef(snapshotPayload);

  const snapshot = snapshotPayload?.snapshot || { nodes: [], edges: [] };
  const safeModules = Array.isArray(modules) ? modules : [];
  const safeEvents = Array.isArray(events) ? events : [];
  const eventsView = useMemo(() => pagedSlice(safeEvents, eventsPage, 30), [safeEvents, eventsPage]);
  const nodesView = useMemo(() => pagedSlice(snapshot.nodes || [], nodesPage, 50), [snapshot.nodes, nodesPage]);
  const edgesView = useMemo(() => pagedSlice(snapshot.edges || [], edgesPage, 50), [snapshot.edges, edgesPage]);
  const modulesView = useMemo(() => pagedSlice(safeModules, modulesPage, 8), [safeModules, modulesPage]);
  const simulationTimeline = useMemo(() => deriveSimulationTimeline(safeEvents), [safeEvents]);
  const selectedNode = useMemo(
    () => (snapshot.nodes || []).find((node) => node.id === selectedNodeId) || null,
    [snapshot.nodes, selectedNodeId]
  );
  const selectedEdge = useMemo(
    () => (snapshot.edges || []).find((edge) => edgeSignature(edge) === selectedEdgeSig) || null,
    [snapshot.edges, selectedEdgeSig]
  );
  const selectedNodeExploration = useMemo(
    () => deriveGraphExploration(snapshot, selectedNodeId),
    [snapshot, selectedNodeId]
  );
  const selectedNodeNameById = useMemo(() => {
    const map = new Map();
    for (const node of snapshot.nodes || []) {
      map.set(Number(node?.id), nodeLabel(node));
    }
    return map;
  }, [snapshot.nodes]);
  const selectedTraceOptions = useMemo(
    () => deriveTraceVariants(snapshot, selectedNodeId, 6),
    [snapshot, selectedNodeId]
  );
  const safeTraceIndex = Math.max(0, Math.min(selectedTraceOptions.length - 1, Number(selectedTraceIndex || 0)));
  const selectedTrace = selectedTraceOptions[safeTraceIndex] || null;
  const styleGraphNodes = useMemo(() => {
    const rows = Array.isArray(snapshot.nodes) ? snapshot.nodes : [];
    return rows
      .filter((node) => String(node?.type || "") === GRAPH_NODE_TYPE_UI_STYLE)
      .map((node) => {
        const info = styleNodeAttributes(node);
        if (!info) return null;
        return {
          id: Number(node?.id || 0),
          node,
          ...info,
        };
      })
      .filter(Boolean);
  }, [snapshot.nodes]);
  const styleGraphByStyleId = useMemo(() => {
    const map = new Map();
    for (const row of styleGraphNodes) {
      if (!row?.styleId) continue;
      map.set(String(row.styleId), row);
    }
    return map;
  }, [styleGraphNodes]);
  const baseStylePreset = useMemo(
    () => stylePresetByIndex(styleNodeIndex),
    [styleNodeIndex]
  );
  const graphStyleOverride = useMemo(
    () => styleGraphByStyleId.get(String(baseStylePreset?.id || "")) || null,
    [styleGraphByStyleId, baseStylePreset]
  );
  const activeStylePreset = useMemo(
    () => {
      const base = baseStylePreset || STYLE_NODE_PRESETS[0];
      const overrideName = String(graphStyleOverride?.styleName || "").trim();
      const overrideDescription = String(graphStyleOverride?.styleDescription || "").trim();
      return {
        ...base,
        name: overrideName || base.name,
        description: overrideDescription || base.description,
        vars: coerceStyleVars(graphStyleOverride?.styleVars, base.vars),
      };
    },
    [baseStylePreset, graphStyleOverride]
  );
  const branchInsights = useMemo(
    () => deriveBranchInsights(snapshot, selectedNodeExploration, selectedTrace, selectedNodeNameById),
    [snapshot, selectedNodeExploration, selectedTrace, selectedNodeNameById]
  );
  const branchReportScopeKey = useMemo(
    () => branchScopeKeyForInsights(branchInsights),
    [branchInsights]
  );
  const branchReportNodes = useMemo(() => {
    const rows = Array.isArray(snapshot.nodes) ? snapshot.nodes : [];
    return rows
      .filter((node) => String(node?.type || "") === GRAPH_NODE_TYPE_BRANCH_REPORT)
      .map((node) => {
        const attrs = node?.attributes && typeof node.attributes === "object" ? node.attributes : {};
        const branchKey = String(attrs.branch_key || "").trim();
        if (!branchKey) return null;
        return {
          id: Number(node?.id || 0),
          node,
          branchKey,
          attributes: attrs,
        };
      })
      .filter(Boolean);
  }, [snapshot.nodes]);
  const branchReportByKey = useMemo(() => {
    const map = new Map();
    for (const row of branchReportNodes) {
      if (!row?.branchKey) continue;
      map.set(row.branchKey, row);
    }
    return map;
  }, [branchReportNodes]);
  const activeBranchReportNode = useMemo(
    () => branchReportByKey.get(branchReportScopeKey) || null,
    [branchReportByKey, branchReportScopeKey]
  );
  const multitoolRequestNodes = useMemo(() => {
    const rows = Array.isArray(snapshot.nodes) ? snapshot.nodes : [];
    return rows
      .filter((node) => String(node?.type || "") === GRAPH_NODE_TYPE_MULTITOOL_REQUEST)
      .slice()
      .sort((a, b) => parseNodeUpdatedTimestamp(b) - parseNodeUpdatedTimestamp(a));
  }, [snapshot.nodes]);
  const multitoolPreferenceNodes = useMemo(() => {
    const rows = Array.isArray(snapshot.nodes) ? snapshot.nodes : [];
    return rows
      .filter((node) => String(node?.type || "") === GRAPH_NODE_TYPE_PREFERENCE_PROFILE)
      .slice()
      .sort((a, b) => parseNodeUpdatedTimestamp(b) - parseNodeUpdatedTimestamp(a));
  }, [snapshot.nodes]);
  const multitoolTaskNodes = useMemo(() => {
    const rows = Array.isArray(snapshot.nodes) ? snapshot.nodes : [];
    return rows
      .filter((node) => String(node?.type || "") === GRAPH_NODE_TYPE_TASK_ITEM)
      .slice()
      .sort((a, b) => parseNodeUpdatedTimestamp(b) - parseNodeUpdatedTimestamp(a));
  }, [snapshot.nodes]);
  const multitoolRiskNodes = useMemo(() => {
    const rows = Array.isArray(snapshot.nodes) ? snapshot.nodes : [];
    return rows
      .filter((node) => String(node?.type || "") === GRAPH_NODE_TYPE_RISK_ITEM)
      .slice()
      .sort((a, b) => parseNodeUpdatedTimestamp(b) - parseNodeUpdatedTimestamp(a));
  }, [snapshot.nodes]);
  const multitoolDomainNodes = useMemo(() => {
    const rows = Array.isArray(snapshot.nodes) ? snapshot.nodes : [];
    return rows.filter((node) => String(node?.type || "") === GRAPH_NODE_TYPE_DOMAIN_BRANCH);
  }, [snapshot.nodes]);
  const multitoolCountryNodes = useMemo(() => {
    const rows = Array.isArray(snapshot.nodes) ? snapshot.nodes : [];
    return rows.filter((node) => String(node?.type || "") === GRAPH_NODE_TYPE_LEGISLATION_COUNTRY);
  }, [snapshot.nodes]);
  const multitoolDomainByKey = useMemo(() => {
    const map = new Map();
    for (const node of multitoolDomainNodes) {
      const attrs = node?.attributes && typeof node.attributes === "object" ? node.attributes : {};
      const key = normalizeMultitoolDomain(attrs.domain_key || attrs.domain || attrs.name || "");
      if (!key) continue;
      map.set(key, node);
    }
    return map;
  }, [multitoolDomainNodes]);
  const multitoolCountryByKey = useMemo(() => {
    const map = new Map();
    for (const node of multitoolCountryNodes) {
      const attrs = node?.attributes && typeof node.attributes === "object" ? node.attributes : {};
      const key = normalizeLegislationCountry(attrs.country_key || attrs.country || attrs.name || "");
      if (!key || key === "global") continue;
      map.set(key, node);
    }
    return map;
  }, [multitoolCountryNodes]);
  const multitoolTaskStatusRows = useMemo(
    () =>
      countRowsFromTokens(
        multitoolTaskNodes.map((node) => {
          const attrs = node?.attributes && typeof node.attributes === "object" ? node.attributes : {};
          return attrs.status;
        }),
        TASK_STATUS_OPTIONS
      ),
    [multitoolTaskNodes]
  );
  const multitoolTaskPriorityRows = useMemo(
    () =>
      countRowsFromTokens(
        multitoolTaskNodes.map((node) => {
          const attrs = node?.attributes && typeof node.attributes === "object" ? node.attributes : {};
          return attrs.priority;
        }),
        TASK_PRIORITY_OPTIONS
      ),
    [multitoolTaskNodes]
  );
  const multitoolRiskProbabilityRows = useMemo(
    () =>
      countRowsFromTokens(
        multitoolRiskNodes.map((node) => {
          const attrs = node?.attributes && typeof node.attributes === "object" ? node.attributes : {};
          return attrs.probability;
        }),
        RISK_PROBABILITY_OPTIONS
      ),
    [multitoolRiskNodes]
  );
  const multitoolRiskImpactRows = useMemo(
    () =>
      countRowsFromTokens(
        multitoolRiskNodes.map((node) => {
          const attrs = node?.attributes && typeof node.attributes === "object" ? node.attributes : {};
          return attrs.impact;
        }),
        RISK_IMPACT_OPTIONS
      ),
    [multitoolRiskNodes]
  );
  const multitoolDomainCoverageRows = useMemo(() => {
    const tokens = [];
    for (const node of [
      ...multitoolRequestNodes,
      ...multitoolPreferenceNodes,
      ...multitoolTaskNodes,
      ...multitoolRiskNodes,
    ]) {
      const attrs = node?.attributes && typeof node.attributes === "object" ? node.attributes : {};
      tokens.push(normalizeMultitoolDomain(attrs.domain || attrs.domain_key || "general"));
    }
    return countRowsFromTokens(tokens, MULTITOOL_DOMAIN_OPTIONS);
  }, [multitoolRequestNodes, multitoolPreferenceNodes, multitoolTaskNodes, multitoolRiskNodes]);
  const multitoolOpenTasks = useMemo(() => {
    return multitoolTaskNodes.filter((node) => {
      const attrs = node?.attributes && typeof node.attributes === "object" ? node.attributes : {};
      return String(attrs.status || "") !== "done";
    });
  }, [multitoolTaskNodes]);
  const multitoolTopRisks = useMemo(() => {
    const rank = { critical: 4, high: 3, medium: 2, low: 1 };
    return multitoolRiskNodes
      .slice()
      .sort((a, b) => {
        const attrsA = a?.attributes && typeof a.attributes === "object" ? a.attributes : {};
        const attrsB = b?.attributes && typeof b.attributes === "object" ? b.attributes : {};
        const scoreA =
          Number(rank[String(attrsA.impact || "medium")] || 0) * 10 +
          Number(rank[String(attrsA.probability || "medium")] || 0);
        const scoreB =
          Number(rank[String(attrsB.impact || "medium")] || 0) * 10 +
          Number(rank[String(attrsB.probability || "medium")] || 0);
        return scoreB - scoreA;
      })
      .slice(0, 6);
  }, [multitoolRiskNodes]);
  const personalTreePayload = useMemo(() => {
    if (personalTreeViewResult?.tree) return personalTreeViewResult.tree;
    if (personalTreeIngestResult?.tree) return personalTreeIngestResult.tree;
    if (personalTreeNoteResult?.tree) return personalTreeNoteResult.tree;
    return null;
  }, [personalTreeViewResult, personalTreeIngestResult, personalTreeNoteResult]);
  const personalTreeSnapshot = useMemo(
    () => ({
      nodes: Array.isArray(personalTreePayload?.nodes) ? personalTreePayload.nodes : [],
      edges: Array.isArray(personalTreePayload?.edges) ? personalTreePayload.edges : [],
    }),
    [personalTreePayload]
  );
  const personalTreeSelectedNodeId = useMemo(() => {
    const focus = Number(personalTreeFocusNodeId || personalTreePayload?.focus_node_id || 0);
    if (!Number.isFinite(focus) || focus <= 0) return null;
    const exists = personalTreeSnapshot.nodes.some((node) => Number(node?.id || 0) === focus);
    return exists ? focus : null;
  }, [personalTreeFocusNodeId, personalTreePayload, personalTreeSnapshot.nodes]);
  const packageNamespaceRows = useMemo(
    () => countRowsFromObject(packagesResult?.stats?.namespace_counts),
    [packagesResult]
  );
  const packageStatusRows = useMemo(
    () => countRowsFromObject(packagesResult?.stats?.status_counts),
    [packagesResult]
  );
  const memoryNamespaceRows = useMemo(
    () => countRowsFromObject(memoryNamespaceViewResult?.namespace_counts),
    [memoryNamespaceViewResult]
  );
  const contradictionIssueRows = useMemo(
    () => countRowsFromObject({ issues: Number(contradictionScanResult?.issue_count || 0) }),
    [contradictionScanResult]
  );
  const taskRiskLevelRows = useMemo(
    () => countRowsFromObject(taskRiskResult?.risk_level_counts),
    [taskRiskResult]
  );
  const timelineEventRows = useMemo(() => {
    const rows = Array.isArray(timelineReplayResult?.timeline) ? timelineReplayResult.timeline : [];
    return countRowsFromTokens(
      rows.map((row) => String(row?.event_type || "").trim()).filter(Boolean),
      []
    );
  }, [timelineReplayResult]);
  const qualityNamespaceRows = useMemo(
    () => countRowsFromObject(qualityResult?.checks?.namespace_counts),
    [qualityResult]
  );
  const contradictionTopIssues = useMemo(() => {
    const rows = Array.isArray(contradictionScanResult?.issues) ? contradictionScanResult.issues : [];
    return rows
      .slice()
      .sort((a, b) => Number(b?.score || 0) - Number(a?.score || 0))
      .slice(0, 6);
  }, [contradictionScanResult]);
  const qualityTrendRows = useMemo(() => {
    const events = Array.isArray(auditResult?.events) ? auditResult.events : [];
    const rows = events
      .filter((row) => String(row?.event_type || "") === "quality_harness_ran")
      .map((row) => {
        const payload = row?.payload && typeof row.payload === "object" ? row.payload : {};
        const score = Number(payload.score || 0);
        if (!Number.isFinite(score) || score <= 0) return null;
        return {
          name: String(formatEventTimeLabel(row) || `run-${Number(row?.id || 0)}`).trim(),
          count: Number(score.toFixed(2)),
          timestamp: Number(row?.timestamp || 0),
        };
      })
      .filter(Boolean)
      .sort((a, b) => Number(a.timestamp || 0) - Number(b.timestamp || 0));
    if (Number.isFinite(Number(qualityResult?.score))) {
      rows.push({
        name: "current",
        count: Number(Number(qualityResult.score || 0).toFixed(2)),
        timestamp: Date.now() / 1000,
      });
    }
    return rows.slice(-10).map((row) => ({ name: row.name, count: row.count }));
  }, [auditResult, qualityResult]);
  const backupHistoryRows = useMemo(() => {
    const rows = Array.isArray(auditResult?.backups) ? auditResult.backups : [];
    return rows
      .slice()
      .sort((a, b) => Number(b?.modified_at || 0) - Number(a?.modified_at || 0))
      .slice(0, 8);
  }, [auditResult]);
  const selectedReasoningChainText = useMemo(() => {
    if (!selectedNodeExploration.hasTarget) return "";
    const pathIds = (selectedTrace?.nodeIds || selectedNodeExploration.pathNodeIds || []).map((id) => Number(id));
    return pathIds
      .map((id) => `${selectedNodeNameById.get(Number(id)) || `#${id}`}`)
      .join(" -> ");
  }, [selectedNodeExploration, selectedNodeNameById, selectedTrace]);
  const selectedReasoningRootsText = useMemo(() => {
    if (!selectedNodeExploration.hasTarget) return "";
    const names = selectedNodeExploration.rootNodeIds
      .slice(0, 6)
      .map((id) => selectedNodeNameById.get(Number(id)) || `#${id}`);
    const suffix = selectedNodeExploration.rootNodeIds.length > names.length ? "..." : "";
    return `${names.join(", ")}${suffix}`;
  }, [selectedNodeExploration, selectedNodeNameById]);
  const selectedReasoningPrerequisitesText = useMemo(() => {
    if (!selectedNodeExploration.hasTarget) return "";
    const names = selectedNodeExploration.ancestorNodeIds
      .slice(0, 12)
      .map((id) => selectedNodeNameById.get(Number(id)) || `#${id}`);
    const suffix = selectedNodeExploration.ancestorNodeIds.length > names.length ? "..." : "";
    return `${names.join(", ")}${suffix}`;
  }, [selectedNodeExploration, selectedNodeNameById]);
  const selectedReasoningDependentsText = useMemo(() => {
    if (!selectedNodeExploration.hasTarget) return "";
    const names = selectedNodeExploration.descendantNodeIds
      .slice(0, 12)
      .map((id) => selectedNodeNameById.get(Number(id)) || `#${id}`);
    const suffix = selectedNodeExploration.descendantNodeIds.length > names.length ? "..." : "";
    return `${names.join(", ")}${suffix}`;
  }, [selectedNodeExploration, selectedNodeNameById]);
  const selectedEdgeReasoning = useMemo(
    () => summarizeEdgeReasoning(selectedEdge, selectedNodeNameById),
    [selectedEdge, selectedNodeNameById]
  );
  const selectedEdgeHistory = useMemo(
    () => deriveEdgeHistory(safeEvents, selectedEdgeSig),
    [safeEvents, selectedEdgeSig]
  );
  const personalizationPayload = useMemo(
    () => buildPersonalizationPayload(personalizationDraft, uiLanguage),
    [personalizationDraft, uiLanguage]
  );
  const personalizationSummaryText = useMemo(
    () => summarizePersonalization(personalizationPayload),
    [personalizationPayload]
  );
  const advisorDetectedModels = useMemo(() => {
    const rows = modelAdvisors?.advisors?.detected_models;
    return Array.isArray(rows) ? rows.map((item) => String(item || "").trim()).filter(Boolean) : [];
  }, [modelAdvisors]);
  const advisorRoleModels = useMemo(() => {
    const rows = modelAdvisors?.advisors?.advisors;
    return Array.isArray(rows) ? rows : [];
  }, [modelAdvisors]);
  const archiveChatModelOptions = useMemo(() => {
    const out = [...advisorDetectedModels];
    const current = String(archiveChatModelPath || "").trim();
    if (current && !out.includes(current)) {
      out.unshift(current);
    }
    return out;
  }, [advisorDetectedModels, archiveChatModelPath]);

  const t = useMemo(() => {
    const selected = {
      ...(TRANSLATIONS.en || {}),
      ...(EXTRA_TRANSLATIONS.en || {}),
      ...(TRANSLATIONS[uiLanguage] || {}),
      ...(EXTRA_TRANSLATIONS[uiLanguage] || {}),
    };
    return (key) => selected[key] || key;
  }, [uiLanguage]);

  const boundedOverviewIndex = Math.max(
    0,
    Math.min(OVERVIEW_SECTION_KEYS.length - 1, Number(overviewSectionIndex || 0))
  );

  useEffect(() => {
    snapshotPayloadRef.current = snapshotPayload;
  }, [snapshotPayload]);

  useEffect(() => {
    try {
      localStorage.setItem("ui_language", uiLanguage);
    } catch (_error) {
      // ignore
    }
  }, [uiLanguage]);

  useEffect(() => {
    try {
      localStorage.setItem(
        PERSONALIZATION_STORAGE_KEY,
        JSON.stringify(normalizePersonalizationDraft(personalizationDraft))
      );
    } catch (_error) {
      // ignore
    }
  }, [personalizationDraft]);

  useEffect(() => {
    applyStylePreset(activeStylePreset);
    try {
      const max = Math.max(0, STYLE_NODE_PRESETS.length - 1);
      const safe = Math.max(0, Math.min(max, Number(styleNodeIndex || 0)));
      localStorage.setItem(STYLE_NODE_STORAGE_KEY, String(Math.trunc(safe)));
    } catch (_error) {
      // ignore
    }
  }, [styleNodeIndex, activeStylePreset]);

  useEffect(() => {
    const base = stylePresetByIndex(styleNodeIndex);
    const override = styleGraphByStyleId.get(String(base?.id || ""));
    const syncToken = `${styleNodeIndex}:${Number(override?.id || 0)}:${String(override?.updatedAt || "")}`;
    if (styleDraftSyncTokenRef.current === syncToken) {
      return;
    }
    styleDraftSyncTokenRef.current = syncToken;
    const styleName = String(override?.styleName || base?.name || "").trim();
    const styleDescription = String(override?.styleDescription || base?.description || "").trim();
    const styleVars = coerceStyleVars(override?.styleVars, base?.vars || {});
    setStyleNodeDraftName(styleName);
    setStyleNodeDraftDescription(styleDescription);
    setStyleNodeDraftVarsText(stringifySafe(styleVars));
  }, [styleNodeIndex, styleGraphByStyleId]);

  useEffect(() => {
    if (!BRANCH_INSIGHT_VIEW_OPTIONS.includes(String(branchInsightView || ""))) {
      setBranchInsightView("cards");
    }
  }, [branchInsightView]);

  useEffect(() => {
    const attrs = activeBranchReportNode?.attributes || {};
    const syncToken = `${branchReportScopeKey}:${Number(activeBranchReportNode?.id || 0)}:${String(attrs.updated_at || "")}`;
    if (branchDraftSyncTokenRef.current === syncToken) {
      return;
    }
    branchDraftSyncTokenRef.current = syncToken;
    const defaultSummary = branchInsights.hasTarget
      ? `Branch ${branchReportScopeKey}: ${branchInsights.nodeCount} nodes / ${branchInsights.edgeCount} edges`
      : `Global graph overview: ${branchInsights.nodeCount} nodes / ${branchInsights.edgeCount} edges`;
    const summaryText = String(attrs.summary_text || defaultSummary).trim();
    const tips = Array.isArray(attrs.tips) ? attrs.tips : branchInsights.lifehacks;
    setBranchReportSummaryText(summaryText);
    setBranchReportTipsText((Array.isArray(tips) ? tips : []).map((item) => String(item || "").trim()).filter(Boolean).join("\n"));
  }, [activeBranchReportNode, branchInsights, branchReportScopeKey]);

  useEffect(() => {
    if (String(archiveChatModelPath || "").trim()) {
      return;
    }
    const generalRow = advisorRoleModels.find((row) => String(row?.role || "") === "general");
    const generalPath = String(generalRow?.model_path || "").trim();
    const fallbackPath = advisorDetectedModels[0] || "";
    const nextPath = generalPath || fallbackPath;
    if (nextPath) {
      setArchiveChatModelPath(nextPath);
    }
  }, [archiveChatModelPath, advisorDetectedModels, advisorRoleModels]);

  useEffect(() => {
    const root = document?.documentElement;
    if (!root) return;
    root.setAttribute("dir", uiLanguage === "ar" ? "rtl" : "ltr");
  }, [uiLanguage]);

  useEffect(() => {
    if (!selectedNode) {
      return;
    }
    setSelectedNodeAttributesText(stringifySafe(selectedNode.attributes || {}));
    setSelectedNodeStateText(stringifySafe(selectedNode.state || {}));
  }, [selectedNodeId, selectedNode]);

  useEffect(() => {
    if (!selectedEdge) {
      return;
    }
    setSelectedEdgeWeight(String(Number(selectedEdge.weight || 0.7)));
    setSelectedEdgeLogicRule(String(selectedEdge.logic_rule || "explicit"));
    setSelectedEdgeMetadataText(stringifySafe(selectedEdge.metadata || {}));
  }, [selectedEdgeSig, selectedEdge]);

  useEffect(() => {
    const suggested = Number(personalTreePayload?.focus_node_id || 0);
    if (!Number.isFinite(suggested) || suggested <= 0) {
      return;
    }
    if (Number(personalTreeFocusNodeId || 0) > 0) {
      return;
    }
    setPersonalTreeFocusNodeId(suggested);
  }, [personalTreePayload, personalTreeFocusNodeId]);

  useEffect(() => {
    if (!selectedRequestNodeId) return;
    const target = multitoolRequestNodes.find((node) => Number(node?.id || 0) === Number(selectedRequestNodeId));
    if (!target) {
      setSelectedRequestNodeId(0);
      return;
    }
    const attrs = target?.attributes && typeof target.attributes === "object" ? target.attributes : {};
    setMultitoolRequestDraft(
      normalizeMultitoolRequestDraft({
        title: attrs.title || attrs.name || "",
        details: attrs.details || attrs.description || "",
        desired_output: attrs.desired_output || "",
        layout_mode: attrs.layout_mode || "graph",
        status: attrs.status || "backlog",
        priority: attrs.priority || "medium",
        domain: attrs.domain || "general",
        country: attrs.country || "global",
      })
    );
  }, [selectedRequestNodeId, multitoolRequestNodes]);

  useEffect(() => {
    if (!selectedPreferenceNodeId) return;
    const target = multitoolPreferenceNodes.find((node) => Number(node?.id || 0) === Number(selectedPreferenceNodeId));
    if (!target) {
      setSelectedPreferenceNodeId(0);
      return;
    }
    const attrs = target?.attributes && typeof target.attributes === "object" ? target.attributes : {};
    setMultitoolPreferenceDraft(
      normalizeMultitoolPreferenceDraft({
        profile_name: attrs.profile_name || attrs.name || "default",
        likes_text: Array.isArray(attrs.likes) ? attrs.likes.join("\n") : attrs.likes_text || "",
        dislikes_text: Array.isArray(attrs.dislikes) ? attrs.dislikes.join("\n") : attrs.dislikes_text || "",
        style_examples_text: Array.isArray(attrs.style_examples)
          ? attrs.style_examples.join("\n")
          : attrs.style_examples_text || "",
        tool_examples_text: Array.isArray(attrs.tool_examples)
          ? attrs.tool_examples.join("\n")
          : attrs.tool_examples_text || "",
        notes: attrs.notes || "",
        domain: attrs.domain || "general",
        country: attrs.country || "global",
      })
    );
  }, [selectedPreferenceNodeId, multitoolPreferenceNodes]);

  useEffect(() => {
    if (!selectedTaskNodeId) return;
    const target = multitoolTaskNodes.find((node) => Number(node?.id || 0) === Number(selectedTaskNodeId));
    if (!target) {
      setSelectedTaskNodeId(0);
      return;
    }
    const attrs = target?.attributes && typeof target.attributes === "object" ? target.attributes : {};
    setMultitoolTaskDraft(
      normalizeMultitoolTaskDraft({
        title: attrs.title || attrs.name || "",
        description: attrs.description || attrs.details || "",
        status: attrs.status || "backlog",
        priority: attrs.priority || "medium",
        due_at: attrs.due_at || "",
        domain: attrs.domain || "general",
        country: attrs.country || "global",
      })
    );
  }, [selectedTaskNodeId, multitoolTaskNodes]);

  useEffect(() => {
    if (!selectedRiskNodeId) return;
    const target = multitoolRiskNodes.find((node) => Number(node?.id || 0) === Number(selectedRiskNodeId));
    if (!target) {
      setSelectedRiskNodeId(0);
      return;
    }
    const attrs = target?.attributes && typeof target.attributes === "object" ? target.attributes : {};
    setMultitoolRiskDraft(
      normalizeMultitoolRiskDraft({
        title: attrs.title || attrs.name || "",
        description: attrs.description || attrs.details || "",
        probability: attrs.probability || "medium",
        impact: attrs.impact || "medium",
        mitigation_text: Array.isArray(attrs.mitigation_steps)
          ? attrs.mitigation_steps.join("\n")
          : attrs.mitigation_text || "",
        domain: attrs.domain || "general",
        country: attrs.country || "global",
      })
    );
  }, [selectedRiskNodeId, multitoolRiskNodes]);

  useEffect(() => {
    setSelectedTraceIndex(0);
  }, [selectedNodeId]);

  useEffect(() => {
    if (selectedTraceOptions.length <= 0 && selectedTraceIndex !== 0) {
      setSelectedTraceIndex(0);
      return;
    }
    if (selectedTraceIndex > selectedTraceOptions.length - 1) {
      setSelectedTraceIndex(0);
    }
  }, [selectedTraceOptions, selectedTraceIndex]);

  useEffect(() => {
    if (selectedNodeId != null && !selectedNode) {
      setSelectedNodeId(null);
    }
    if (selectedEdgeSig && !selectedEdge) {
      setSelectedEdgeSig(null);
    }
    if (
      selectedRequestNodeId &&
      !multitoolRequestNodes.some((node) => Number(node?.id || 0) === Number(selectedRequestNodeId))
    ) {
      setSelectedRequestNodeId(0);
    }
    if (
      selectedPreferenceNodeId &&
      !multitoolPreferenceNodes.some((node) => Number(node?.id || 0) === Number(selectedPreferenceNodeId))
    ) {
      setSelectedPreferenceNodeId(0);
    }
    if (
      selectedTaskNodeId &&
      !multitoolTaskNodes.some((node) => Number(node?.id || 0) === Number(selectedTaskNodeId))
    ) {
      setSelectedTaskNodeId(0);
    }
    if (
      selectedRiskNodeId &&
      !multitoolRiskNodes.some((node) => Number(node?.id || 0) === Number(selectedRiskNodeId))
    ) {
      setSelectedRiskNodeId(0);
    }
  }, [
    snapshot,
    selectedNodeId,
    selectedNode,
    selectedEdgeSig,
    selectedEdge,
    selectedRequestNodeId,
    selectedPreferenceNodeId,
    selectedTaskNodeId,
    selectedRiskNodeId,
    multitoolRequestNodes,
    multitoolPreferenceNodes,
    multitoolTaskNodes,
    multitoolRiskNodes,
  ]);

  useEffect(() => {
    setEventsPage((prev) => Math.max(0, Math.min(prev, eventsView.totalPages - 1)));
  }, [eventsView.totalPages]);

  useEffect(() => {
    setNodesPage((prev) => Math.max(0, Math.min(prev, nodesView.totalPages - 1)));
  }, [nodesView.totalPages]);

  useEffect(() => {
    setEdgesPage((prev) => Math.max(0, Math.min(prev, edgesView.totalPages - 1)));
  }, [edgesView.totalPages]);

  useEffect(() => {
    setModulesPage((prev) => Math.max(0, Math.min(prev, modulesView.totalPages - 1)));
  }, [modulesView.totalPages]);

  useEffect(() => {
    function onHashChange() {
      const hashPage = getInitialPage();
      if (PAGE_KEYS.includes(hashPage)) {
        setCurrentPage(hashPage);
      }
    }
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  function goToPage(pageKey) {
    if (!PAGE_KEYS.includes(pageKey)) {
      return;
    }
    setCurrentPage(pageKey);
    window.location.hash = `#${pageKey}`;
  }

  function appendLog(line) {
    setLogLines((prev) => [...prev, line]);
  }

  function appendArchiveChatMessage(role, text) {
    const body = String(text || "").trim();
    if (!body) return;
    setArchiveChatMessages((prev) => [
      ...(Array.isArray(prev) ? prev : []),
      {
        role: String(role || "assistant"),
        text: body,
        ts: Date.now(),
      },
    ]);
  }

  function patchPersonalizationDraft(patch) {
    setPersonalizationDraft((prev) => normalizePersonalizationDraft({ ...(prev || {}), ...(patch || {}) }));
  }

  function patchMultitoolRequestDraft(patch) {
    setMultitoolRequestDraft((prev) => normalizeMultitoolRequestDraft({ ...(prev || {}), ...(patch || {}) }));
  }

  function patchMultitoolPreferenceDraft(patch) {
    setMultitoolPreferenceDraft((prev) => normalizeMultitoolPreferenceDraft({ ...(prev || {}), ...(patch || {}) }));
  }

  function patchMultitoolTaskDraft(patch) {
    setMultitoolTaskDraft((prev) => normalizeMultitoolTaskDraft({ ...(prev || {}), ...(patch || {}) }));
  }

  function patchMultitoolRiskDraft(patch) {
    setMultitoolRiskDraft((prev) => normalizeMultitoolRiskDraft({ ...(prev || {}), ...(patch || {}) }));
  }

  function patchPersonalTreeIngestDraft(patch) {
    setPersonalTreeIngestDraft((prev) => normalizePersonalTreeIngestDraft({ ...(prev || {}), ...(patch || {}) }));
  }

  function patchPersonalTreeNoteDraft(patch) {
    setPersonalTreeNoteDraft((prev) => normalizePersonalTreeNoteDraft({ ...(prev || {}), ...(patch || {}) }));
  }

  function patchPackagesDraft(patch) {
    setPackagesDraft((prev) => normalizePackagesDraft({ ...(prev || {}), ...(patch || {}) }));
  }

  function patchMemoryNamespaceDraft(patch) {
    setMemoryNamespaceDraft((prev) => normalizeMemoryNamespaceDraft({ ...(prev || {}), ...(patch || {}) }));
  }

  function patchGraphRagDraft(patch) {
    setGraphRagDraft((prev) => normalizeGraphRagDraft({ ...(prev || {}), ...(patch || {}) }));
  }

  function patchContradictionScanDraft(patch) {
    setContradictionScanDraft((prev) => normalizeContradictionScanDraft({ ...(prev || {}), ...(patch || {}) }));
  }

  function patchTaskRiskDraft(patch) {
    setTaskRiskDraft((prev) => normalizeTaskRiskDraft({ ...(prev || {}), ...(patch || {}) }));
  }

  function patchTimelineReplayDraft(patch) {
    setTimelineReplayDraft((prev) => normalizeTimelineReplayDraft({ ...(prev || {}), ...(patch || {}) }));
  }

  function patchLlmPolicyDraft(patch) {
    setLlmPolicyDraft((prev) => normalizeLlmPolicyDraft({ ...(prev || {}), ...(patch || {}) }));
  }

  function patchQualityDraft(patch) {
    setQualityDraft((prev) => normalizeQualityDraft({ ...(prev || {}), ...(patch || {}) }));
  }

  function patchBackupDraft(patch) {
    setBackupDraft((prev) => normalizeBackupDraft({ ...(prev || {}), ...(patch || {}) }));
  }

  function patchAuditDraft(patch) {
    setAuditDraft((prev) => normalizeAuditDraft({ ...(prev || {}), ...(patch || {}) }));
  }

  function onSavePersonalization() {
    patchPersonalizationDraft({});
    appendLog(`${t("log_system")}: ${t("log_personalization_saved")}`);
  }

  function onResetPersonalization() {
    setPersonalizationDraft({ ...DEFAULT_PERSONALIZATION_DRAFT });
    appendLog(`${t("log_system")}: ${t("log_personalization_reset")}`);
  }

  function onSyncPersonalizationRoles() {
    const roles = personalizationPayload?.llm_roles || {};
    setDebateProposerRole(normalizeRole(roles.proposer, "creative"));
    setDebateCriticRole(normalizeRole(roles.critic, "analyst"));
    setDebateJudgeRole(normalizeRole(roles.judge, "planner"));
    appendLog(`${t("log_system")}: ${t("log_personalization_roles_synced")}`);
  }

  async function upsertEdgeRelation(payload) {
    try {
      return await updateEdge(payload);
    } catch (_error) {
      try {
        return await createEdge(payload);
      } catch (__error) {
        return null;
      }
    }
  }

  async function ensureMultitoolBranchNodes(domainValue, countryValue) {
    const domainKey = normalizeMultitoolDomain(domainValue);
    const countryKey = normalizeLegislationCountry(countryValue);
    let domainNodeId = Number(multitoolDomainByKey.get(domainKey)?.id || 0);
    if (!domainNodeId) {
      const createdDomain = await createNode({
        node_type: GRAPH_NODE_TYPE_DOMAIN_BRANCH,
        attributes: {
          name: domainKey,
          domain_key: domainKey,
          updated_at: new Date().toISOString(),
        },
        state: { weight: 1.0 },
      });
      domainNodeId = Number(createdDomain?.node?.id || 0);
    }

    let countryNodeId = 0;
    if (countryKey !== "global") {
      countryNodeId = Number(multitoolCountryByKey.get(countryKey)?.id || 0);
      if (!countryNodeId) {
        const createdCountry = await createNode({
          node_type: GRAPH_NODE_TYPE_LEGISLATION_COUNTRY,
          attributes: {
            name: countryKey,
            country_key: countryKey,
            updated_at: new Date().toISOString(),
          },
          state: { weight: 1.0 },
        });
        countryNodeId = Number(createdCountry?.node?.id || 0);
      }
      if (domainNodeId && countryNodeId) {
        await upsertEdgeRelation({
          from_node: domainNodeId,
          to_node: countryNodeId,
          relation_type: "branches_to_country",
          direction: "directed",
          weight: 0.82,
          logic_rule: "multitool_branch",
          metadata: {
            source: "multitool_workspace",
            domain: domainKey,
            country: countryKey,
          },
        });
      }
    }
    return { domainKey, countryKey, domainNodeId, countryNodeId };
  }

  async function onSaveMultitoolRequest() {
    const draft = normalizeMultitoolRequestDraft(multitoolRequestDraft);
    if (!String(draft.title || "").trim()) {
      appendLog(`${t("log_error")}: ${t("error_multitool_request_title_empty")}`);
      return;
    }
    try {
      const out = await runAction(t("action_multitool_request_save"), async () => {
        const branch = await ensureMultitoolBranchNodes(draft.domain, draft.country);
        const activeNode = multitoolRequestNodes.find(
          (node) => Number(node?.id || 0) === Number(selectedRequestNodeId || 0)
        );
        const nextAttrs = {
          ...(activeNode?.attributes && typeof activeNode.attributes === "object" ? activeNode.attributes : {}),
          title: String(draft.title || "").trim(),
          details: String(draft.details || "").trim(),
          desired_output: String(draft.desired_output || "").trim(),
          layout_mode: String(draft.layout_mode || "graph"),
          status: String(draft.status || "backlog"),
          priority: String(draft.priority || "medium"),
          domain: branch.domainKey,
          country: branch.countryKey,
          updated_at: new Date().toISOString(),
        };
        let nodeId = Number(activeNode?.id || 0);
        let response = null;
        if (nodeId > 0) {
          response = await updateNode({
            node_id: nodeId,
            attributes: nextAttrs,
            state: activeNode?.state || { urgency: 0.5 },
          });
        } else {
          response = await createNode({
            node_type: GRAPH_NODE_TYPE_MULTITOOL_REQUEST,
            attributes: nextAttrs,
            state: { urgency: 0.6 },
          });
          nodeId = Number(response?.node?.id || 0);
        }
        if (nodeId > 0 && branch.domainNodeId > 0) {
          await upsertEdgeRelation({
            from_node: nodeId,
            to_node: branch.domainNodeId,
            relation_type: "requests_in_domain",
            direction: "directed",
            weight: 0.86,
            logic_rule: "multitool_request_link",
            metadata: { source: "multitool_workspace" },
          });
        }
        if (nodeId > 0 && branch.countryNodeId > 0) {
          await upsertEdgeRelation({
            from_node: nodeId,
            to_node: branch.countryNodeId,
            relation_type: "applies_in_country",
            direction: "directed",
            weight: 0.8,
            logic_rule: "multitool_request_country_link",
            metadata: { source: "multitool_workspace" },
          });
        }
        return response;
      });
      setMultitoolRequestSaveResult(out || null);
      setSelectedRequestNodeId(Number(out?.node?.id || selectedRequestNodeId || 0));
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onSaveMultitoolPreference() {
    const draft = normalizeMultitoolPreferenceDraft(multitoolPreferenceDraft);
    try {
      const out = await runAction(t("action_multitool_preference_save"), async () => {
        const branch = await ensureMultitoolBranchNodes(draft.domain, draft.country);
        const activeNode = multitoolPreferenceNodes.find(
          (node) => Number(node?.id || 0) === Number(selectedPreferenceNodeId || 0)
        );
        const nextAttrs = {
          ...(activeNode?.attributes && typeof activeNode.attributes === "object" ? activeNode.attributes : {}),
          profile_name: String(draft.profile_name || "default").trim() || "default",
          likes: parseListText(draft.likes_text),
          dislikes: parseListText(draft.dislikes_text),
          style_examples: parseListText(draft.style_examples_text),
          tool_examples: parseListText(draft.tool_examples_text),
          notes: String(draft.notes || "").trim(),
          domain: branch.domainKey,
          country: branch.countryKey,
          updated_at: new Date().toISOString(),
        };
        let nodeId = Number(activeNode?.id || 0);
        let response = null;
        if (nodeId > 0) {
          response = await updateNode({
            node_id: nodeId,
            attributes: nextAttrs,
            state: activeNode?.state || { confidence: 0.8 },
          });
        } else {
          response = await createNode({
            node_type: GRAPH_NODE_TYPE_PREFERENCE_PROFILE,
            attributes: nextAttrs,
            state: { confidence: 0.8 },
          });
          nodeId = Number(response?.node?.id || 0);
        }
        if (nodeId > 0 && branch.domainNodeId > 0) {
          await upsertEdgeRelation({
            from_node: nodeId,
            to_node: branch.domainNodeId,
            relation_type: "preference_in_domain",
            direction: "directed",
            weight: 0.84,
            logic_rule: "multitool_preference_link",
            metadata: { source: "multitool_workspace" },
          });
        }
        if (nodeId > 0 && branch.countryNodeId > 0) {
          await upsertEdgeRelation({
            from_node: nodeId,
            to_node: branch.countryNodeId,
            relation_type: "preference_in_country",
            direction: "directed",
            weight: 0.78,
            logic_rule: "multitool_preference_country_link",
            metadata: { source: "multitool_workspace" },
          });
        }
        return response;
      });
      setMultitoolPreferenceSaveResult(out || null);
      setSelectedPreferenceNodeId(Number(out?.node?.id || selectedPreferenceNodeId || 0));
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onSaveMultitoolTask() {
    const draft = normalizeMultitoolTaskDraft(multitoolTaskDraft);
    if (!String(draft.title || "").trim()) {
      appendLog(`${t("log_error")}: ${t("error_multitool_task_title_empty")}`);
      return;
    }
    try {
      const out = await runAction(t("action_multitool_task_save"), async () => {
        const branch = await ensureMultitoolBranchNodes(draft.domain, draft.country);
        const activeNode = multitoolTaskNodes.find(
          (node) => Number(node?.id || 0) === Number(selectedTaskNodeId || 0)
        );
        const nextAttrs = {
          ...(activeNode?.attributes && typeof activeNode.attributes === "object" ? activeNode.attributes : {}),
          title: String(draft.title || "").trim(),
          description: String(draft.description || "").trim(),
          status: String(draft.status || "backlog"),
          priority: String(draft.priority || "medium"),
          due_at: String(draft.due_at || "").trim(),
          domain: branch.domainKey,
          country: branch.countryKey,
          updated_at: new Date().toISOString(),
        };
        let nodeId = Number(activeNode?.id || 0);
        let response = null;
        if (nodeId > 0) {
          response = await updateNode({
            node_id: nodeId,
            attributes: nextAttrs,
            state: activeNode?.state || { progress: 0.2 },
          });
        } else {
          response = await createNode({
            node_type: GRAPH_NODE_TYPE_TASK_ITEM,
            attributes: nextAttrs,
            state: { progress: 0.2 },
          });
          nodeId = Number(response?.node?.id || 0);
        }
        if (nodeId > 0 && branch.domainNodeId > 0) {
          await upsertEdgeRelation({
            from_node: nodeId,
            to_node: branch.domainNodeId,
            relation_type: "task_in_domain",
            direction: "directed",
            weight: 0.85,
            logic_rule: "multitool_task_link",
            metadata: { source: "multitool_workspace" },
          });
        }
        if (nodeId > 0 && branch.countryNodeId > 0) {
          await upsertEdgeRelation({
            from_node: nodeId,
            to_node: branch.countryNodeId,
            relation_type: "task_in_country",
            direction: "directed",
            weight: 0.8,
            logic_rule: "multitool_task_country_link",
            metadata: { source: "multitool_workspace" },
          });
        }
        return response;
      });
      setMultitoolTaskSaveResult(out || null);
      setSelectedTaskNodeId(Number(out?.node?.id || selectedTaskNodeId || 0));
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onSaveMultitoolRisk() {
    const draft = normalizeMultitoolRiskDraft(multitoolRiskDraft);
    if (!String(draft.title || "").trim()) {
      appendLog(`${t("log_error")}: ${t("error_multitool_risk_title_empty")}`);
      return;
    }
    try {
      const out = await runAction(t("action_multitool_risk_save"), async () => {
        const branch = await ensureMultitoolBranchNodes(draft.domain, draft.country);
        const activeNode = multitoolRiskNodes.find(
          (node) => Number(node?.id || 0) === Number(selectedRiskNodeId || 0)
        );
        const nextAttrs = {
          ...(activeNode?.attributes && typeof activeNode.attributes === "object" ? activeNode.attributes : {}),
          title: String(draft.title || "").trim(),
          description: String(draft.description || "").trim(),
          probability: String(draft.probability || "medium"),
          impact: String(draft.impact || "medium"),
          mitigation_steps: parseListText(draft.mitigation_text),
          domain: branch.domainKey,
          country: branch.countryKey,
          updated_at: new Date().toISOString(),
        };
        let nodeId = Number(activeNode?.id || 0);
        let response = null;
        if (nodeId > 0) {
          response = await updateNode({
            node_id: nodeId,
            attributes: nextAttrs,
            state: activeNode?.state || { severity: 0.6 },
          });
        } else {
          response = await createNode({
            node_type: GRAPH_NODE_TYPE_RISK_ITEM,
            attributes: nextAttrs,
            state: { severity: 0.6 },
          });
          nodeId = Number(response?.node?.id || 0);
        }
        if (nodeId > 0 && branch.domainNodeId > 0) {
          await upsertEdgeRelation({
            from_node: nodeId,
            to_node: branch.domainNodeId,
            relation_type: "risk_in_domain",
            direction: "directed",
            weight: 0.87,
            logic_rule: "multitool_risk_link",
            metadata: { source: "multitool_workspace" },
          });
        }
        if (nodeId > 0 && branch.countryNodeId > 0) {
          await upsertEdgeRelation({
            from_node: nodeId,
            to_node: branch.countryNodeId,
            relation_type: "risk_in_country",
            direction: "directed",
            weight: 0.83,
            logic_rule: "multitool_risk_country_link",
            metadata: { source: "multitool_workspace" },
          });
        }
        return response;
      });
      setMultitoolRiskSaveResult(out || null);
      setSelectedRiskNodeId(Number(out?.node?.id || selectedRiskNodeId || 0));
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onRunPersonalTreeIngest() {
    const draft = normalizePersonalTreeIngestDraft(personalTreeIngestDraft);
    const text = String(draft.text || "").trim();
    if (!text) {
      appendLog(`${t("log_error")}: ${t("error_personal_tree_text_empty")}`);
      return;
    }
    const sessionId = getClientSessionId();
    const userId = `web_${sessionId}`;
    try {
      const out = await runAction(t("action_personal_tree_ingest"), () =>
        runProjectPersonalTreeIngest({
          user_id: userId,
          session_id: sessionId,
          title: String(draft.title || "").trim(),
          topic: String(draft.topic || "").trim(),
          text,
          source_type: String(draft.source_type || "text"),
          source_url: String(draft.source_url || "").trim(),
          source_title: String(draft.source_title || "").trim(),
          max_points: Number(draft.max_points || 6),
          parent_node_id: Number(personalTreeFocusNodeId || 0),
          max_nodes: 220,
        })
      );
      setPersonalTreeIngestResult(out || null);
      setPersonalTreeViewResult(out || null);
      const summaryNodeId = Number(out?.semantic_binding?.summary_node_id || 0);
      if (summaryNodeId > 0) {
        setPersonalTreeFocusNodeId(summaryNodeId);
      }
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onSavePersonalTreeNote() {
    const draft = normalizePersonalTreeNoteDraft(personalTreeNoteDraft);
    const title = String(draft.title || "").trim();
    const note = String(draft.note || "").trim();
    if (!title && !note) {
      appendLog(`${t("log_error")}: ${t("error_personal_tree_note_empty")}`);
      return;
    }
    const sessionId = getClientSessionId();
    const userId = `web_${sessionId}`;
    try {
      const out = await runAction(t("action_personal_tree_note_save"), () =>
        saveProjectPersonalTreeNote({
          user_id: userId,
          session_id: sessionId,
          parent_node_id: Number(personalTreeFocusNodeId || 0),
          title,
          note,
          tags: parseListText(draft.tags_text),
          links: parseListText(draft.links_text),
          source_type: String(draft.source_type || "note"),
          source_url: String(draft.source_url || "").trim(),
          source_title: String(draft.source_title || "").trim(),
          max_nodes: 220,
        })
      );
      setPersonalTreeNoteResult(out || null);
      setPersonalTreeViewResult(out || null);
      const noteNodeId = Number(out?.note?.node_id || 0);
      if (noteNodeId > 0) {
        setPersonalTreeFocusNodeId(noteNodeId);
      }
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onRefreshPersonalTreeView(nextFocusNodeId = null) {
    const sessionId = getClientSessionId();
    const userId = `web_${sessionId}`;
    const focusNodeId =
      nextFocusNodeId == null ? Number(personalTreeFocusNodeId || 0) : Number(nextFocusNodeId || 0);
    try {
      const out = await runAction(t("action_personal_tree_view"), () =>
        viewProjectPersonalTree({
          user_id: userId,
          focus_node_id: focusNodeId > 0 ? focusNodeId : 0,
          max_nodes: 220,
        })
      );
      setPersonalTreeViewResult(out || null);
      if (focusNodeId > 0) {
        setPersonalTreeFocusNodeId(focusNodeId);
      }
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onRunPackagesAction(action) {
    const draft = normalizePackagesDraft(packagesDraft);
    if (action === "store" && !parseListText(draft.items_text).length) {
      appendLog(`${t("log_error")}: ${t("error_packages_items_empty")}`);
      return;
    }
    const sessionId = getClientSessionId();
    const userId = `web_${sessionId}`;
    try {
      const out = await runAction(t(`action_packages_${action}`), () =>
        manageProjectPackages({
          user_id: userId,
          session_id: sessionId,
          package_name: draft.package_name,
          action,
          items: action === "store" ? parseListText(draft.items_text) : undefined,
          item_node_ids: action === "restore" ? parseNumericListText(draft.restore_ids_text) : undefined,
          model_role: draft.model_role,
          model_path: draft.model_path,
          classify_with_llm: Boolean(draft.classify_with_llm),
          apply_changes: Boolean(draft.apply_changes),
          confirmation: String(draft.confirmation || "").trim(),
        })
      );
      setPackagesResult(out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onApplyMemoryNamespace() {
    const draft = normalizeMemoryNamespaceDraft(memoryNamespaceDraft);
    const sessionId = getClientSessionId();
    const userId = `web_${sessionId}`;
    try {
      const out = await runAction(t("action_memory_apply"), () =>
        applyProjectMemoryNamespace({
          user_id: userId,
          session_id: sessionId,
          namespace: draft.namespace,
          source_namespace: draft.source_namespace || "",
          scope: draft.scope,
          query: String(draft.query || "").trim(),
          node_ids: parseNumericListText(draft.node_ids_text),
          min_score: Math.max(0, Math.min(1, Number(draft.min_score || 0.2))),
          apply_changes: Boolean(draft.apply_changes),
          confirmation: String(draft.confirmation || "").trim(),
        })
      );
      setMemoryNamespaceApplyResult(out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onViewMemoryNamespace() {
    const draft = normalizeMemoryNamespaceDraft(memoryNamespaceDraft);
    const sessionId = getClientSessionId();
    const userId = `web_${sessionId}`;
    try {
      const out = await runAction(t("action_memory_view"), () =>
        viewProjectMemoryNamespace({
          user_id: userId,
          scope: draft.scope,
          max_nodes: 240,
        })
      );
      setMemoryNamespaceViewResult(out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onRunGraphRagQuery() {
    const draft = normalizeGraphRagDraft(graphRagDraft);
    const query = String(draft.query || "").trim();
    if (!query) {
      appendLog(`${t("log_error")}: ${t("error_rag_query_empty")}`);
      return;
    }
    const sessionId = getClientSessionId();
    const userId = `web_${sessionId}`;
    try {
      const out = await runAction(t("action_graph_rag"), () =>
        runProjectGraphRagQuery({
          query,
          user_id: userId,
          scope: draft.scope,
          namespace: draft.namespace || "",
          top_k: Number(draft.top_k || 6),
          use_llm: Boolean(draft.use_llm),
          model_role: draft.model_role,
          model_path: draft.model_path,
        })
      );
      setGraphRagResult(out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onRunContradictionScan() {
    const draft = normalizeContradictionScanDraft(contradictionScanDraft);
    const sessionId = getClientSessionId();
    const userId = `web_${sessionId}`;
    try {
      const out = await runAction(t("action_contradiction_scan"), () =>
        scanProjectContradictions({
          user_id: userId,
          session_id: sessionId,
          scope: draft.scope,
          namespace: draft.namespace || "",
          max_nodes: Number(draft.max_nodes || 120),
          top_k: Number(draft.top_k || 20),
          min_overlap: Math.max(0.1, Math.min(0.95, Number(draft.min_overlap || 0.32))),
          apply_to_graph: Boolean(draft.apply_to_graph),
          confirmation: String(draft.confirmation || "").trim(),
        })
      );
      setContradictionScanResult(out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onRunTaskRiskBoard() {
    const draft = normalizeTaskRiskDraft(taskRiskDraft);
    const tasks = String(draft.tasks_text || "")
      .split(/\n+/g)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        const [title, ...rest] = line.split("|");
        return {
          title: String(title || "").trim(),
          description: String(rest.join("|") || "").trim(),
        };
      })
      .filter((row) => String(row.title || "").trim());
    if (!tasks.length) {
      appendLog(`${t("log_error")}: ${t("error_task_risk_tasks_empty")}`);
      return;
    }
    const sessionId = getClientSessionId();
    const userId = `web_${sessionId}`;
    try {
      const out = await runAction(t("action_task_risk_board"), () =>
        runProjectTaskRiskBoard({
          user_id: userId,
          session_id: sessionId,
          tasks,
          apply_to_graph: Boolean(draft.apply_to_graph),
          confirmation: String(draft.confirmation || "").trim(),
        })
      );
      setTaskRiskResult(out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onRunTimelineReplay() {
    const draft = normalizeTimelineReplayDraft(timelineReplayDraft);
    const sessionId = getClientSessionId();
    const userId = `web_${sessionId}`;
    try {
      const out = await runAction(t("action_timeline_replay"), () =>
        runProjectTimelineReplay({
          user_id: userId,
          session_id: sessionId,
          event_type: draft.event_type,
          limit: Number(draft.limit || 300),
          from_ts: Number(draft.from_ts || 0) || 0,
          to_ts: Number(draft.to_ts || 0) || 0,
        })
      );
      setTimelineReplayResult(out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onLoadLlmPolicy() {
    try {
      const out = await runAction(t("action_policy_load"), () => getProjectLLMPolicy());
      setLlmPolicyResult(out || null);
      const policy = out?.policy || {};
      patchLlmPolicyDraft({
        mode: String(policy.mode || "confirm_required"),
        trusted_sessions_text: Array.isArray(policy.trusted_sessions) ? policy.trusted_sessions.join("\n") : "",
        trusted_users_text: Array.isArray(policy.trusted_users) ? policy.trusted_users.join("\n") : "",
        allow_actions_text: Array.isArray(policy.allow_apply_for_actions)
          ? policy.allow_apply_for_actions.join("\n")
          : "",
      });
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onSaveLlmPolicy() {
    const draft = normalizeLlmPolicyDraft(llmPolicyDraft);
    try {
      const out = await runAction(t("action_policy_save"), () =>
        updateProjectLLMPolicy({
          mode: draft.mode,
          trusted_sessions: parseListText(draft.trusted_sessions_text),
          trusted_users: parseListText(draft.trusted_users_text),
          allow_apply_for_actions: parseListText(draft.allow_actions_text),
          merge_lists: Boolean(draft.merge_lists),
        })
      );
      setLlmPolicyResult(out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onRunQualityHarness() {
    const draft = normalizeQualityDraft(qualityDraft);
    const sessionId = getClientSessionId();
    const userId = `web_${sessionId}`;
    try {
      const out = await runAction(t("action_quality_harness"), () =>
        runProjectQualityHarness({
          user_id: userId,
          sample_queries: parseListText(draft.sample_queries_text),
        })
      );
      setQualityResult(out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onCreateBackup() {
    const draft = normalizeBackupDraft(backupDraft);
    const sessionId = getClientSessionId();
    const userId = `web_${sessionId}`;
    try {
      const out = await runAction(t("action_backup_create"), () =>
        createProjectBackup({
          label: String(draft.label || "manual").trim() || "manual",
          user_id: userId,
          include_events: Boolean(draft.include_events),
          event_limit: Number(draft.event_limit || 1000),
        })
      );
      setBackupCreateResult(out || null);
      if (String(out?.path || "").trim()) {
        patchBackupDraft({ path: String(out.path) });
      }
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onRestoreBackup() {
    const draft = normalizeBackupDraft(backupDraft);
    const sessionId = getClientSessionId();
    const userId = `web_${sessionId}`;
    try {
      const out = await runAction(t("action_backup_restore"), () =>
        restoreProjectBackup({
          path: String(draft.path || "").trim(),
          latest: Boolean(draft.latest),
          user_id: userId,
          session_id: sessionId,
          apply_changes: Boolean(draft.apply_changes),
          confirmation: String(draft.confirmation || "").trim(),
          restore_policy: Boolean(draft.restore_policy),
        })
      );
      setBackupRestoreResult(out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onLoadAuditLogs() {
    const draft = normalizeAuditDraft(auditDraft);
    try {
      const out = await runAction(t("action_audit_load"), () =>
        getProjectAuditLogs({
          limit: Number(draft.limit || 200),
          include_backups: Boolean(draft.include_backups),
        })
      );
      setAuditResult(out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onSaveStyleNodeGraph() {
    const base = stylePresetByIndex(styleNodeIndex);
    const styleName = String(styleNodeDraftName || "").trim() || String(base?.name || "");
    const styleDescription = String(styleNodeDraftDescription || "").trim() || String(base?.description || "");
    let parsedVars = {};
    try {
      parsedVars = parseJsonSafe(styleNodeDraftVarsText, {}, t("error_style_vars_json_invalid"));
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
      return;
    }
    if (!parsedVars || typeof parsedVars !== "object" || Array.isArray(parsedVars)) {
      appendLog(`${t("log_error")}: ${t("error_style_vars_json_invalid")}`);
      return;
    }
    const styleVars = coerceStyleVars(parsedVars, base?.vars || {});
    try {
      const out = await runAction(t("action_style_node_save"), async () => {
        const existing = styleGraphByStyleId.get(String(base?.id || ""));
        const nextAttrs = {
          ...(existing?.node?.attributes && typeof existing.node.attributes === "object" ? existing.node.attributes : {}),
          style_id: String(base?.id || ""),
          style_name: styleName,
          style_description: styleDescription,
          style_vars: styleVars,
          is_active: true,
          updated_at: new Date().toISOString(),
        };
        let targetNodeId = 0;
        let response = null;
        if (existing?.id) {
          targetNodeId = Number(existing.id);
          response = await updateNode({
            node_id: targetNodeId,
            attributes: nextAttrs,
            state: existing?.node?.state || {},
          });
        } else {
          response = await createNode({
            node_type: GRAPH_NODE_TYPE_UI_STYLE,
            attributes: nextAttrs,
            state: { visual_weight: 1.0 },
          });
          targetNodeId = Number(response?.node?.id || 0);
        }
        for (const row of styleGraphNodes) {
          const rowId = Number(row?.id || 0);
          if (!rowId || rowId === targetNodeId) continue;
          const attrs = row?.node?.attributes && typeof row.node.attributes === "object" ? row.node.attributes : {};
          if (!attrs.is_active) continue;
          await updateNode({
            node_id: rowId,
            attributes: {
              ...attrs,
              is_active: false,
              updated_at: new Date().toISOString(),
            },
            state: row?.node?.state || {},
          });
        }
        return response;
      });
      setStyleNodeSaveResult(out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onSaveBranchReportGraph() {
    const scopeKey = branchReportScopeKey;
    const summaryText = String(branchReportSummaryText || "").trim();
    const tips = parseListText(branchReportTipsText);
    const activeNode = activeBranchReportNode;
    const reportAttrs = {
      ...(activeNode?.attributes && typeof activeNode.attributes === "object" ? activeNode.attributes : {}),
      branch_key: scopeKey,
      summary_text: summaryText,
      tips,
      scope: branchInsights.hasTarget ? "selected" : "global",
      target_node_id: branchInsights.hasTarget ? Number(branchInsights.targetNode || 0) : 0,
      node_count: Number(branchInsights.nodeCount || 0),
      edge_count: Number(branchInsights.edgeCount || 0),
      avg_weight: Number(branchInsights.avgWeight || 0),
      top_relations: branchInsights.topRelations || [],
      top_node_types: branchInsights.topNodeTypes || [],
      top_nodes: branchInsights.topNodes || [],
      top_edges: branchInsights.topEdges || [],
      updated_at: new Date().toISOString(),
    };
    try {
      const out = await runAction(t("action_branch_report_save"), async () => {
        let nodeId = Number(activeNode?.id || 0);
        let response = null;
        if (nodeId > 0) {
          response = await updateNode({
            node_id: nodeId,
            attributes: reportAttrs,
            state: activeNode?.node?.state || {},
          });
        } else {
          response = await createNode({
            node_type: GRAPH_NODE_TYPE_BRANCH_REPORT,
            attributes: reportAttrs,
            state: { confidence: 0.85 },
          });
          nodeId = Number(response?.node?.id || 0);
        }
        if (nodeId > 0 && branchInsights.hasTarget && Number.isFinite(Number(branchInsights.targetNode))) {
          await upsertEdgeRelation({
            from_node: nodeId,
            to_node: Number(branchInsights.targetNode),
            relation_type: "reports_on",
            direction: "directed",
            weight: 0.88,
            logic_rule: "ui_branch_report",
            metadata: {
              branch_key: scopeKey,
              source: "branch_visual_toolkit",
            },
          });
        }
        return response;
      });
      setBranchReportSaveResult(out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  function triggerEdgeLiveEffect(event) {
    const type = String(event?.event_type || "");
    if (!["edge_added", "edge_updated", "edge_updated_manual", "edge_weight_feedback"].includes(type)) {
      return;
    }
    const sig = edgeSignatureFromEvent(event);
    if (!sig) {
      return;
    }
    const kind = type === "edge_added" ? "added" : "updated";
    const token = `${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    setEdgeEffectsBySig((prev) => ({
      ...(prev || {}),
      [sig]: { kind, token },
    }));

    const timers = edgeEffectTimersRef.current;
    const existing = timers.get(sig);
    if (existing != null) {
      window.clearTimeout(existing);
    }
    const timeoutId = window.setTimeout(() => {
      setEdgeEffectsBySig((prev) => {
        const current = prev && typeof prev === "object" ? prev : {};
        if (!current[sig]) return current;
        const next = { ...current };
        delete next[sig];
        return next;
      });
      timers.delete(sig);
    }, 1500);
    timers.set(sig, timeoutId);
  }

  async function syncGraphFromServer() {
    if (liveSyncInFlightRef.current) {
      return;
    }
    liveSyncInFlightRef.current = true;
    try {
      const [snapshotResp, eventsResp] = await Promise.all([getSnapshot(), getEvents(200)]);
      if (snapshotResp && typeof snapshotResp === "object") {
        setSnapshotPayload(snapshotResp);
      }
      setEvents(Array.isArray(eventsResp?.events) ? eventsResp.events : []);
    } catch (_error) {
      // stream sync should be best-effort and never block UI interactions
    } finally {
      liveSyncInFlightRef.current = false;
    }
  }

  function queueLiveSync(delayMs = 160) {
    if (liveSyncTimerRef.current != null) {
      return;
    }
    liveSyncTimerRef.current = window.setTimeout(() => {
      liveSyncTimerRef.current = null;
      void syncGraphFromServer();
    }, delayMs);
  }

  async function refreshAll() {
    setBusy(true);
    try {
      const sessionId = getClientSessionId();
      const [modulesPayload, snapshotResp, eventsResp, typesResp] = await Promise.all([
        getModules(),
        getSnapshot(),
        getEvents(200),
        getNodeTypes(),
      ]);
      setModules(Array.isArray(modulesPayload?.modules) ? modulesPayload.modules : []);
      if (snapshotResp && typeof snapshotResp === "object") {
        setSnapshotPayload(snapshotResp);
      } else {
        setSnapshotPayload({ snapshot: { nodes: [], edges: [] }, metrics: {} });
      }
      setEvents(Array.isArray(eventsResp?.events) ? eventsResp.events : []);
      setNodeTypes(Array.isArray(typesResp?.node_types) ? typesResp.node_types : ["generic", "human", "company"]);

      const optional = await Promise.allSettled([
        getProjectDbSchema(),
        getProjectModelAdvisors(),
        introspectClient({
          session_id: sessionId,
          user_id: `web_${sessionId}`,
          client: collectClientContext(),
        }),
      ]);
      if (optional[0].status === "fulfilled") {
        setDbSchema(optional[0].value || null);
      }
      if (optional[1].status === "fulfilled") {
        setModelAdvisors(optional[1].value || null);
      }
      if (optional[2].status === "fulfilled") {
        setClientProfile(optional[2].value?.profile || null);
      }
      appendLog(`${t("log_system")}: ${t("log_workspace_refreshed")}`);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    refreshAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let disposed = false;
    let unsubscribe = () => {};
    let reconnectDelayMs = 1000;

    const connect = () => {
      if (disposed) {
        return;
      }
      unsubscribe = subscribeGraphEvents({
        onOpen: () => {
          reconnectDelayMs = 1000;
          appendLog("SYSTEM: live stream connected");
        },
        onMessage: (payload) => {
          if (!payload || typeof payload !== "object") {
            return;
          }
          if (payload.type === "hello") {
            if (payload.snapshot && payload.metrics) {
              setSnapshotPayload({
                snapshot: payload.snapshot,
                metrics: payload.metrics,
              });
            }
            if (Array.isArray(payload.events)) {
              setEvents(payload.events);
            }
            return;
          }
          if (payload.type !== "graph_event" || !payload.event) {
            return;
          }
          const event = payload.event;
          triggerEdgeLiveEffect(event);
          setEvents((prev) => {
            const current = Array.isArray(prev) ? prev : [];
            if (current.some((row) => Number(row?.id) === Number(event.id))) {
              return current;
            }
            const next = [...current, event];
            return next.slice(-300);
          });
          const patch = applyGraphEventToSnapshot(snapshotPayloadRef.current, event);
          if (patch.nextPayload) {
            snapshotPayloadRef.current = patch.nextPayload;
            setSnapshotPayload(patch.nextPayload);
          }
          if (patch.needsSync) {
            queueLiveSync(120);
          }
        },
        onClose: () => {
          appendLog("SYSTEM: live stream disconnected");
          if (disposed || liveReconnectRef.current != null) {
            return;
          }
          liveReconnectRef.current = window.setTimeout(() => {
            liveReconnectRef.current = null;
            connect();
          }, reconnectDelayMs);
          reconnectDelayMs = Math.min(12000, Math.round(reconnectDelayMs * 1.8));
        },
        onError: () => {
          appendLog("ERROR: live stream error");
        },
      });
    };

    connect();
    return () => {
      disposed = true;
      unsubscribe();
      if (liveReconnectRef.current != null) {
        window.clearTimeout(liveReconnectRef.current);
        liveReconnectRef.current = null;
      }
      if (liveSyncTimerRef.current != null) {
        window.clearTimeout(liveSyncTimerRef.current);
        liveSyncTimerRef.current = null;
      }
      for (const timeoutId of edgeEffectTimersRef.current.values()) {
        window.clearTimeout(timeoutId);
      }
      edgeEffectTimersRef.current.clear();
      setEdgeEffectsBySig({});
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    function onError(event) {
      const message = event?.error?.stack || event?.message || "unknown runtime error";
      setFatalError(String(message));
    }
    function onUnhandledRejection(event) {
      const reason = event?.reason;
      const message =
        (reason && (reason.stack || reason.message || String(reason))) ||
        "unhandled promise rejection";
      setFatalError(String(message));
    }
    window.addEventListener("error", onError);
    window.addEventListener("unhandledrejection", onUnhandledRejection);
    return () => {
      window.removeEventListener("error", onError);
      window.removeEventListener("unhandledrejection", onUnhandledRejection);
    };
  }, []);

  async function runAction(actionName, fn) {
    try {
      setBusy(true);
      const payload = await fn();
      const sessionId = getClientSessionId();
      if (payload.snapshot && payload.metrics) {
        setSnapshotPayload({ snapshot: payload.snapshot, metrics: payload.metrics });
      } else {
        const snap = await getSnapshot();
        setSnapshotPayload(snap);
      }
      const eventsResp = await getEvents(200);
      const optional = await Promise.allSettled([
        getProjectDbSchema(),
        getProjectModelAdvisors(),
        introspectClient({
          session_id: sessionId,
          user_id: `web_${sessionId}`,
          client: collectClientContext(),
        }),
      ]);
      setEvents(eventsResp.events || []);
      if (optional[0].status === "fulfilled") {
        setDbSchema(optional[0].value || null);
      }
      if (optional[1].status === "fulfilled") {
        setModelAdvisors(optional[1].value || null);
      }
      if (optional[2].status === "fulfilled") {
        setClientProfile(optional[2].value?.profile || null);
      }
      appendLog(`${t("log_system")}: ${actionName} ${t("log_action_complete")}`);
      return payload;
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
      throw error;
    } finally {
      setBusy(false);
    }
  }

  async function onCreateNode() {
    try {
      const payload = {
        node_type: nodeType,
        attributes: parseJsonSafe(nodeAttributesText, {}, t("error_invalid_json_payload")),
        state: parseJsonSafe(nodeStateText, {}, t("error_invalid_json_payload")),
      };

      if (nodeType === "human") {
        payload.first_name = humanFirstName;
        payload.last_name = humanLastName;
        payload.bio = humanBio;
        payload.profile_text = humanProfileText;
        payload.employment_text = humanEmploymentText;
        payload.employment = parseJsonSafe(humanEmploymentJsonText, [], t("error_invalid_json_payload"));
      }
      if (nodeType === "company") {
        payload.name = companyName;
        payload.industry = companyIndustry;
        payload.description = companyDescription;
      }

      const out = await runAction(t("create_node_btn"), () => createNode(payload));
      if (out?.node?.id) {
        appendLog(`${t("log_system")}: ${t("log_node_created")} #${out.node.id} (${out.node.type})`);
      }
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onCreateEdge() {
    await runAction(t("create_edge_btn"), () =>
      createEdge({
        from_node: Number(edgeFrom),
        to_node: Number(edgeTo),
        relation_type: edgeRelation,
        weight: Number(edgeWeight),
        direction: edgeDirection,
        logic_rule: edgeLogicRule,
      })
    );
  }

  async function onSimulate() {
    const payload = {
      seed_node_ids: simSeedIds,
      recursive_depth: Number(simDepth),
      propagation_steps: Number(simSteps),
      damping: Number(simDamping),
      activation: simActivation,
      infer_rounds: Number(simInferRounds),
    };
    const out = await runAction(t("run_simulation"), () => simulateGraph(payload));
    setLastSimulation(out.result || null);
  }

  async function onRewardEvent() {
    await runAction(t("apply_reward"), () =>
      rewardEvent({
        event_id: Number(rewardEventId),
        reward: Number(rewardValue),
        learning_rate: Number(rewardLr),
      })
    );
  }

  async function onReinforceRelation() {
    await runAction(t("reinforce_relation"), () =>
      reinforceRelation({
        relation_type: reinforceRelationType,
        reward: Number(reinforceReward),
        learning_rate: Number(reinforceLr),
      })
    );
  }

  async function onPersist() {
    await runAction(t("persist_snapshot"), () => persistGraph());
  }

  async function onLoad() {
    await runAction(t("load_snapshot"), () => loadGraph());
  }

  async function onClear() {
    await runAction(t("action_clear"), () => clearGraph());
    setLastSimulation(null);
    setDailyModeResult(null);
    setUserGraphResult(null);
    setAutorunsImportResult(null);
    setDebateResult(null);
    setArchiveChatResult(null);
    setArchiveChatMessages([]);
    setArchiveReviewUpdatesText("[]");
    setArchiveReviewResult(null);
  }

  async function onSeedDemo() {
    const narrativeText = String(demoNarrative || "").trim();
    await runAction(t("action_seed_demo"), () =>
      watchProjectDemo({
        persona_name: "Alexa",
        narrative: narrativeText,
        language: String(uiLanguage || "en"),
        reset_graph: true,
        use_llm: true,
      })
    );
  }

  async function onRunDailyMode() {
    const text = String(dailyJournalText || "").trim();
    if (!text) {
      appendLog(`${t("log_error")}: ${t("error_daily_journal_empty")}`);
      return;
    }
    const sessionId = getClientSessionId();
    const clientContext = collectClientContext();
    const applyPersonalization = Boolean(personalizationDraft?.auto_apply_daily);
    const personalization = applyPersonalization ? personalizationPayload : {};
    const effectiveText = applyPersonalization
      ? `${text}\n\n[personalization]\n${personalizationPromptBlock(personalization)}`
      : text;
    try {
      const out = await runAction(t("action_daily_mode"), () =>
        runProjectDailyMode({
          text: effectiveText,
          user_id: `web_${sessionId}`,
          display_name: "Web User",
          language: String(uiLanguage || "en"),
          session_id: `daily_${sessionId}`,
          auto_snapshot: true,
          recommendation_count: 4,
          run_knowledge_analysis: true,
          apply_profile_update: true,
          use_llm_profile: true,
          include_client_profile: true,
          client: clientContext,
          personalization,
        })
      );
      setDailyModeResult(out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onApplyUserGraph() {
    const sessionId = getClientSessionId();
    const text = String(userNarrativeText || "").trim();
    const clientContext = collectClientContext();
    const personalization = personalizationDraft?.auto_apply_user_graph ? personalizationPayload : {};
    try {
      const out = await runAction(t("action_user_graph_update"), () =>
        updateProjectUserGraph({
          user_id: `web_${sessionId}`,
          display_name: "Web User",
          text,
          language: String(uiLanguage || "en"),
          session_id: `profile_${sessionId}`,
          use_llm_profile: true,
          include_client_profile: true,
          client: clientContext,
          profile_text: text,
          personalization,
          fears: parseListText(userFearsText),
          desires: parseListText(userDesiresText),
          goals: parseListText(userGoalsText),
          principles: parseListText(userPrinciplesText),
          opportunities: parseListText(userOpportunitiesText),
          abilities: parseListText(userAbilitiesText),
          access: parseListText(userAccessText),
          knowledge: parseListText(userKnowledgeText),
          assets: parseListText(userAssetsText),
        })
      );
      setUserGraphResult(out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onImportAutoruns() {
    const text = String(autorunsImportText || "").trim();
    const sessionId = getClientSessionId();
    const clientContext = collectClientContext();
    try {
      const out = await runAction(t("action_autoruns_import"), () =>
        importProjectAutoruns({
          text,
          auto_detect: true,
          query: text,
          language: String(uiLanguage || "en"),
          client: clientContext,
          user_id: `web_${sessionId}`,
          session_id: `autoruns_${sessionId}`,
          host_label: "Web User",
          max_rows: 1200,
        })
      );
      setAutorunsImportResult(out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onRunLLMDebate() {
    const topic = String(debatePromptText || "").trim();
    if (!topic) {
      appendLog(`${t("log_error")}: ${t("error_debate_prompt_empty")}`);
      return;
    }
    const sessionId = getClientSessionId();
    const applyPersonalization = Boolean(personalizationDraft?.auto_apply_debate);
    const personalization = applyPersonalization ? personalizationPayload : {};
    const profileRoles = personalization?.llm_roles || {};
    const proposerRole = applyPersonalization
      ? normalizeRole(profileRoles.proposer, "creative")
      : normalizeRole(debateProposerRole, "creative");
    const criticRole = applyPersonalization
      ? normalizeRole(profileRoles.critic, "analyst")
      : normalizeRole(debateCriticRole, "analyst");
    const judgeRole = applyPersonalization
      ? normalizeRole(profileRoles.judge, "planner")
      : normalizeRole(debateJudgeRole, "planner");
    try {
      const out = await runAction(t("action_llm_debate"), () =>
        runProjectLLMDebate({
          topic,
          user_id: `web_${sessionId}`,
          session_id: `debate_${sessionId}`,
          hypothesis_count: Number(debateHypothesisCount || 3),
          attach_to_graph: Boolean(debateAttachGraph),
          proposer_role: proposerRole,
          critic_role: criticRole,
          judge_role: judgeRole,
          personalization,
        })
      );
      setDebateResult(out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onRunArchiveChat() {
    const message = String(archiveChatMessageText || "").trim();
    if (!message) {
      appendLog(`${t("log_error")}: ${t("error_archive_chat_message_empty")}`);
      return;
    }
    const sessionId = getClientSessionId();
    appendArchiveChatMessage("user", message);
    try {
      const out = await runAction(t("action_archive_chat"), () =>
        runProjectArchiveChat({
          user_id: `web_${sessionId}`,
          session_id: `archive_${sessionId}`,
          message,
          context: String(archiveChatContextText || "").trim(),
          model_path: String(archiveChatModelPath || "").trim(),
          model_role: normalizeRole(archiveChatModelRole, "general"),
          apply_to_graph: Boolean(archiveChatAttachGraph),
          verification_mode: String(archiveChatVerificationMode || "strict").trim() || "strict",
          top_k: 5,
        })
      );
      setArchiveChatResult(out || null);
      const assistantReply = String(out?.assistant_reply || out?.summary || "").trim();
      appendArchiveChatMessage("assistant", assistantReply || "Processed. Review conclusions in the review panel.");
      setArchiveReviewUpdatesText(stringifySafe(out?.archive_updates || []));
      setArchiveReviewResult(out?.review || out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onApplyArchiveReview() {
    let updatesDraft = [];
    try {
      updatesDraft = parseJsonSafe(
        archiveReviewUpdatesText,
        [],
        t("error_archive_review_json_invalid")
      );
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
      return;
    }
    const message = String(archiveChatMessageText || "").trim() || "Manual archive review draft";
    const sessionId = getClientSessionId();
    try {
      const out = await runAction(t("action_archive_review_apply"), () =>
        applyProjectArchiveReview({
          user_id: `web_${sessionId}`,
          session_id: `archive_review_${sessionId}`,
          message,
          context: String(archiveChatContextText || "").trim(),
          summary: String(archiveChatResult?.summary || "").trim(),
          archive_updates: updatesDraft,
          verification_mode: String(archiveChatVerificationMode || "strict").trim() || "strict",
          apply_to_graph: Boolean(archiveChatAttachGraph),
          top_k: 5,
        })
      );
      setArchiveReviewResult(out?.review || out || null);
      setArchiveChatResult(out || null);
      setArchiveReviewUpdatesText(stringifySafe(out?.archive_updates || updatesDraft || []));
      appendArchiveChatMessage(
        "assistant",
        String(out?.assistant_reply || "Reviewed edited draft. Check verification result in the review panel.")
      );
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onReportHallucination() {
    const prompt = String(hallucinationPromptText || "").trim();
    const llmAnswer = String(hallucinationWrongAnswerText || "").trim();
    const correctAnswer = String(hallucinationCorrectAnswerText || "").trim();
    if (!prompt) {
      appendLog(`${t("log_error")}: ${t("error_hallucination_prompt_empty")}`);
      return;
    }
    if (!llmAnswer) {
      appendLog(`${t("log_error")}: ${t("error_hallucination_wrong_empty")}`);
      return;
    }
    if (!correctAnswer) {
      appendLog(`${t("log_error")}: ${t("error_hallucination_correct_empty")}`);
      return;
    }
    const sessionId = getClientSessionId();
    try {
      const out = await runAction(t("action_hallucination_report"), () =>
        reportProjectHallucination({
          user_id: `web_${sessionId}`,
          session_id: `hall_${sessionId}`,
          prompt,
          llm_answer: llmAnswer,
          correct_answer: correctAnswer,
          source: String(hallucinationSourceText || "").trim(),
          tags: parseListText(hallucinationTagsText),
          severity: String(hallucinationSeverity || "medium"),
          confidence: 0.9,
        })
      );
      setHallucinationReportResult(out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onCheckHallucinationHunter() {
    const prompt = String(hallucinationPromptText || "").trim();
    const llmAnswer = String(hallucinationLlmAnswerText || "").trim();
    if (!prompt && !llmAnswer) {
      appendLog(`${t("log_error")}: ${t("error_hallucination_prompt_empty")}`);
      return;
    }
    const sessionId = getClientSessionId();
    try {
      const out = await runAction(t("action_hallucination_check"), () =>
        checkProjectHallucination({
          user_id: `web_${sessionId}`,
          prompt,
          llm_answer: llmAnswer,
          top_k: 5,
        })
      );
      setHallucinationCheckResult(out || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function refreshProfilePromptPreview() {
    try {
      const payload = await getProfilePrompt(profileEntityHint);
      setProfilePromptPreview(String(payload?.prompt_template || ""));
    } catch (_error) {
      setProfilePromptPreview("");
    }
  }

  useEffect(() => {
    refreshProfilePromptPreview();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [profileEntityHint]);

  async function onInferProfileGraph() {
    const text = String(profileInputText || "").trim();
    if (!text) {
      appendLog(`${t("log_error")}: ${t("error_profile_input_empty")}`);
      return;
    }
    try {
      const out = await runAction(t("extract_profile_graph"), () =>
        inferProfileGraph({
          text,
          entity_type_hint: profileEntityHint,
          create_graph: true,
          save_json: true,
        })
      );
      setProfileResult(
        out?.profile_json
          ? { profile_json_file: out?.profile_json_file || "", profile_json: out.profile_json }
          : null
      );
      appendLog(`${t("log_system")}: ${t("log_profile_imported")}`);
      if (out?.profile_json_file) {
        appendLog(`${t("log_system")}: JSON -> ${out.profile_json_file}`);
      }
    } catch (error) {
      const details = String(error?.message || "");
      if (details.toLowerCase().includes("local profile llm is unavailable")) {
        appendLog(`${t("log_error")}: ${t("model_missing_hint")}`);
      } else {
        appendLog(`${t("log_error")}: ${details}`);
      }
    }
  }

  async function onUpdateSelectedNode() {
    if (!selectedNode) return;
    try {
      await runAction(t("action_update_node"), () =>
        updateNode({
          node_id: Number(selectedNode.id),
          attributes: parseJsonSafe(selectedNodeAttributesText, {}, t("error_invalid_json_payload")),
          state: parseJsonSafe(selectedNodeStateText, {}, t("error_invalid_json_payload")),
        })
      );
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onDeleteSelectedNode() {
    if (!selectedNode) return;
    try {
      await runAction(t("action_delete_node"), () =>
        deleteNode({
          node_id: Number(selectedNode.id),
        })
      );
      setSelectedNodeId(null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onUpdateSelectedEdge() {
    if (!selectedEdge) return;
    try {
      await runAction(t("action_update_edge"), () =>
        updateEdge({
          from_node: Number(selectedEdge.from),
          to_node: Number(selectedEdge.to),
          relation_type: String(selectedEdge.relation_type || ""),
          direction: String(selectedEdge.direction || "directed"),
          weight: Number(selectedEdgeWeight),
          logic_rule: String(selectedEdgeLogicRule || "explicit"),
          metadata: parseJsonSafe(selectedEdgeMetadataText, {}, t("error_invalid_json_payload")),
        })
      );
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onDeleteSelectedEdge() {
    if (!selectedEdge) return;
    try {
      await runAction(t("action_delete_edge"), () =>
        deleteEdge({
          from_node: Number(selectedEdge.from),
          to_node: Number(selectedEdge.to),
          relation_type: String(selectedEdge.relation_type || ""),
          direction: String(selectedEdge.direction || "directed"),
        })
      );
      setSelectedEdgeSig(null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  async function onRefreshClientProfile() {
    try {
      const sessionId = getClientSessionId();
      const out = await runAction(t("action_client_introspection"), () =>
        introspectClient({
          session_id: sessionId,
          user_id: `web_${sessionId}`,
          client: collectClientContext(),
        })
      );
      setClientProfile(out?.profile || null);
    } catch (error) {
      appendLog(`${t("log_error")}: ${error.message}`);
    }
  }

  function renderPager({ page, totalPages, setPage, label }) {
    return (
      <div className="pager">
        <button type="button" disabled={page <= 0} onClick={() => setPage(Math.max(0, page - 1))}>
          {t("pager_prev")}
        </button>
        <span>
          {label}: {page + 1}/{Math.max(1, totalPages)}
        </span>
        <button
          type="button"
          disabled={page >= totalPages - 1}
          onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
        >
          {t("pager_next")}
        </button>
      </div>
    );
  }

  function renderMiniBars(rows, prefix) {
    const items = Array.isArray(rows) ? rows : [];
    const maxCount = Math.max(1, ...items.map((item) => Number(item?.count || 0)));
    return (
      <div className="mini-bar-list">
        {items.map((item, index) => {
          const count = Number(item?.count || 0);
          const widthPercent = Math.max(8, Math.round((count / maxCount) * 100));
          return (
            <div key={`${prefix}-${item.name}-${index}`} className="mini-bar-row">
              <span className="mini-bar-label">{String(item.name || "n/a")}</span>
              <div className="mini-bar-track">
                <div className="mini-bar-fill" style={{ width: `${widthPercent}%` }} />
              </div>
              <span className="mini-bar-value">{count}</span>
            </div>
          );
        })}
      </div>
    );
  }

  function renderOverviewPage() {
    const sectionKey = OVERVIEW_SECTION_KEYS[boundedOverviewIndex] || OVERVIEW_SECTION_KEYS[0];
    const sectionTitle = overviewSectionLabel(sectionKey, t);

    return (
      <>
        <section className="card">
          <div className="overview-pager-header">
            <h2>{t("overview_sections")}</h2>
            {renderPager({
              page: boundedOverviewIndex,
              totalPages: OVERVIEW_SECTION_KEYS.length,
              setPage: setOverviewSectionIndex,
              label: sectionTitle,
            })}
          </div>
          <div className="page-nav subpage-nav">
            {OVERVIEW_SECTION_KEYS.map((key, index) => (
              <button
                key={key}
                type="button"
                className={`page-tab ${boundedOverviewIndex === index ? "active" : ""}`}
                onClick={() => setOverviewSectionIndex(index)}
              >
                {overviewSectionLabel(key, t)}
              </button>
            ))}
          </div>
        </section>

        {sectionKey === "demo" && (
          <section className="card">
            <h2>{t("demo_narrative")}</h2>
            <div className="row">
              <label>{t("scenario")}</label>
              <textarea
                value={demoNarrative}
                onChange={(e) => setDemoNarrative(e.target.value)}
                rows={4}
                placeholder={t("demo_narrative_placeholder")}
              />
            </div>
            <div className="row-actions">
              <button disabled={busy} onClick={onSeedDemo}>
                {t("action_seed_demo")}
              </button>
              <button disabled={busy} onClick={onRefreshClientProfile}>
                {t("refresh_client_profile")}
              </button>
            </div>
          </section>
        )}

        {sectionKey === "daily" && (
          <section className="card grid-2">
            <div>
              <h2>{t("daily_mode")}</h2>
              <div className="row">
                <label>{t("daily_journal")}</label>
                <textarea
                  value={dailyJournalText}
                  onChange={(e) => setDailyJournalText(e.target.value)}
                  rows={8}
                  placeholder={t("daily_journal_placeholder")}
                />
              </div>
              <div className="row-actions">
                <button disabled={busy} onClick={onRunDailyMode}>
                  {t("run_daily_analysis")}
                </button>
              </div>
            </div>
            <div>
              <h2>{t("daily_recommendations_scores")}</h2>
              <pre>{stringifySafe(dailyModeResult)}</pre>
            </div>
          </section>
        )}

        {sectionKey === "user_graph" && (
          <section className="card grid-2">
            <div>
              <div className="personalization-studio">
                <div className="personalization-studio-head">
                  <h3>{t("personalization_studio")}</h3>
                  <span className="personalization-summary-chip">{t("personalization_summary")}</span>
                </div>
                <p className="personalization-summary-text">{personalizationSummaryText}</p>
                <div className="grid-3">
                  <div className="row">
                    <label>{t("personalization_response_style")}</label>
                    <select
                      value={personalizationDraft.response_style}
                      onChange={(e) => patchPersonalizationDraft({ response_style: e.target.value })}
                    >
                      {PERSONALIZATION_STYLE_OPTIONS.map((item) => (
                        <option key={`p-style-${item}`} value={item}>
                          {item}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="row">
                    <label>{t("personalization_reasoning_depth")}</label>
                    <select
                      value={personalizationDraft.reasoning_depth}
                      onChange={(e) => patchPersonalizationDraft({ reasoning_depth: e.target.value })}
                    >
                      {PERSONALIZATION_DEPTH_OPTIONS.map((item) => (
                        <option key={`p-depth-${item}`} value={item}>
                          {item}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="row">
                    <label>{t("personalization_risk_tolerance")}</label>
                    <select
                      value={personalizationDraft.risk_tolerance}
                      onChange={(e) => patchPersonalizationDraft({ risk_tolerance: e.target.value })}
                    >
                      {PERSONALIZATION_RISK_OPTIONS.map((item) => (
                        <option key={`p-risk-${item}`} value={item}>
                          {item}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <div className="row">
                  <label>{t("personalization_tone")}</label>
                  <select
                    value={personalizationDraft.tone}
                    onChange={(e) => patchPersonalizationDraft({ tone: e.target.value })}
                  >
                    {PERSONALIZATION_TONE_OPTIONS.map((item) => (
                      <option key={`p-tone-${item}`} value={item}>
                        {item}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="grid-3">
                  <div className="row">
                    <label>{`${t("personalization_roles")} · proposer`}</label>
                    <select
                      value={personalizationDraft.role_proposer}
                      onChange={(e) => patchPersonalizationDraft({ role_proposer: e.target.value })}
                    >
                      {LLM_ROLE_OPTIONS.map((role) => (
                        <option key={`p-role-proposer-${role}`} value={role}>
                          {role}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="row">
                    <label>{`${t("personalization_roles")} · critic`}</label>
                    <select
                      value={personalizationDraft.role_critic}
                      onChange={(e) => patchPersonalizationDraft({ role_critic: e.target.value })}
                    >
                      {LLM_ROLE_OPTIONS.map((role) => (
                        <option key={`p-role-critic-${role}`} value={role}>
                          {role}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="row">
                    <label>{`${t("personalization_roles")} · judge`}</label>
                    <select
                      value={personalizationDraft.role_judge}
                      onChange={(e) => patchPersonalizationDraft({ role_judge: e.target.value })}
                    >
                      {LLM_ROLE_OPTIONS.map((role) => (
                        <option key={`p-role-judge-${role}`} value={role}>
                          {role}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <div className="row">
                  <label>{t("personalization_focus_goals")}</label>
                  <textarea
                    value={personalizationDraft.focus_goals_text}
                    onChange={(e) => patchPersonalizationDraft({ focus_goals_text: e.target.value })}
                    rows={2}
                    placeholder={t("personalization_goals_placeholder")}
                  />
                </div>
                <div className="row">
                  <label>{t("personalization_domain_focus")}</label>
                  <textarea
                    value={personalizationDraft.domain_focus_text}
                    onChange={(e) => patchPersonalizationDraft({ domain_focus_text: e.target.value })}
                    rows={2}
                    placeholder={t("personalization_domains_placeholder")}
                  />
                </div>
                <div className="row">
                  <label>{t("personalization_avoid_topics")}</label>
                  <textarea
                    value={personalizationDraft.avoid_topics_text}
                    onChange={(e) => patchPersonalizationDraft({ avoid_topics_text: e.target.value })}
                    rows={2}
                    placeholder={t("personalization_avoid_placeholder")}
                  />
                </div>
                <div className="row">
                  <label>{t("personalization_memory_notes")}</label>
                  <textarea
                    value={personalizationDraft.memory_notes}
                    onChange={(e) => patchPersonalizationDraft({ memory_notes: e.target.value })}
                    rows={3}
                    placeholder={t("personalization_memory_placeholder")}
                  />
                </div>
                <div className="row row-checkbox">
                  <label>{t("personalization_auto_apply_user_graph")}</label>
                  <input
                    type="checkbox"
                    checked={Boolean(personalizationDraft.auto_apply_user_graph)}
                    onChange={(e) => patchPersonalizationDraft({ auto_apply_user_graph: Boolean(e.target.checked) })}
                  />
                </div>
                <div className="row row-checkbox">
                  <label>{t("personalization_auto_apply_daily")}</label>
                  <input
                    type="checkbox"
                    checked={Boolean(personalizationDraft.auto_apply_daily)}
                    onChange={(e) => patchPersonalizationDraft({ auto_apply_daily: Boolean(e.target.checked) })}
                  />
                </div>
                <div className="row row-checkbox">
                  <label>{t("personalization_auto_apply_debate")}</label>
                  <input
                    type="checkbox"
                    checked={Boolean(personalizationDraft.auto_apply_debate)}
                    onChange={(e) => patchPersonalizationDraft({ auto_apply_debate: Boolean(e.target.checked) })}
                  />
                </div>
                <div className="row-actions">
                  <button type="button" disabled={busy} onClick={onSavePersonalization}>
                    {t("personalization_save")}
                  </button>
                  <button type="button" disabled={busy} onClick={onSyncPersonalizationRoles}>
                    {t("personalization_sync_roles")}
                  </button>
                  <button type="button" disabled={busy} onClick={onResetPersonalization}>
                    {t("personalization_reset")}
                  </button>
                </div>
              </div>
              <h2>{t("user_semantic_graph")}</h2>
              <div className="row">
                <label>{t("user_graph_narrative")}</label>
                <textarea
                  value={userNarrativeText}
                  onChange={(e) => setUserNarrativeText(e.target.value)}
                  rows={5}
                  placeholder={t("user_graph_narrative_placeholder")}
                />
              </div>
              <div className="row">
                <label>{t("user_fears")}</label>
                <textarea value={userFearsText} onChange={(e) => setUserFearsText(e.target.value)} rows={2} />
              </div>
              <div className="row">
                <label>{t("user_desires")}</label>
                <textarea value={userDesiresText} onChange={(e) => setUserDesiresText(e.target.value)} rows={2} />
              </div>
              <div className="row">
                <label>{t("user_goals")}</label>
                <textarea value={userGoalsText} onChange={(e) => setUserGoalsText(e.target.value)} rows={2} />
              </div>
              <div className="row">
                <label>{t("user_principles")}</label>
                <textarea value={userPrinciplesText} onChange={(e) => setUserPrinciplesText(e.target.value)} rows={2} />
              </div>
              <div className="row">
                <label>{t("user_opportunities")}</label>
                <textarea value={userOpportunitiesText} onChange={(e) => setUserOpportunitiesText(e.target.value)} rows={2} />
              </div>
              <div className="row">
                <label>{t("user_abilities")}</label>
                <textarea value={userAbilitiesText} onChange={(e) => setUserAbilitiesText(e.target.value)} rows={2} />
              </div>
              <div className="row">
                <label>{t("user_access")}</label>
                <textarea value={userAccessText} onChange={(e) => setUserAccessText(e.target.value)} rows={2} />
              </div>
              <div className="row">
                <label>{t("user_knowledge")}</label>
                <textarea value={userKnowledgeText} onChange={(e) => setUserKnowledgeText(e.target.value)} rows={2} />
              </div>
              <div className="row">
                <label>{t("user_assets")}</label>
                <textarea
                  value={userAssetsText}
                  onChange={(e) => setUserAssetsText(e.target.value)}
                  rows={2}
                  placeholder={t("user_assets_placeholder")}
                />
              </div>
              <div className="row-actions">
                <button disabled={busy} onClick={onApplyUserGraph}>
                  {t("apply_user_graph")}
                </button>
              </div>
            </div>
            <div>
              <h2>{t("user_graph_update_result")}</h2>
              <pre>{stringifySafe(userGraphResult)}</pre>
            </div>
          </section>
        )}

        {sectionKey === "multitool" && (
          <section className="card multitool-layout">
            <div className="multitool-head">
              <h2>{t("multitool_title")}</h2>
              <p>{t("multitool_subtitle")}</p>
            </div>
            <div className="multitool-grid">
              <article className="multitool-card">
                <h3>{t("multitool_section_requests")}</h3>
                <div className="row">
                  <label>{t("multitool_choose_existing")}</label>
                  <select
                    value={String(selectedRequestNodeId || 0)}
                    onChange={(e) => setSelectedRequestNodeId(Number(e.target.value || 0))}
                  >
                    <option value="0">-</option>
                    {multitoolRequestNodes.map((node) => {
                      const attrs = node?.attributes && typeof node.attributes === "object" ? node.attributes : {};
                      const title = String(attrs.title || attrs.name || `#${node.id}`);
                      return (
                        <option key={`request-node-${node.id}`} value={String(node.id)}>
                          {`${node.id} · ${title}`}
                        </option>
                      );
                    })}
                  </select>
                </div>
                <div className="row">
                  <label>{t("multitool_request_title")}</label>
                  <input
                    value={multitoolRequestDraft.title}
                    onChange={(e) => patchMultitoolRequestDraft({ title: e.target.value })}
                  />
                </div>
                <div className="row">
                  <label>{t("multitool_request_details")}</label>
                  <textarea
                    value={multitoolRequestDraft.details}
                    onChange={(e) => patchMultitoolRequestDraft({ details: e.target.value })}
                    rows={3}
                  />
                </div>
                <div className="row">
                  <label>{t("multitool_request_output")}</label>
                  <textarea
                    value={multitoolRequestDraft.desired_output}
                    onChange={(e) => patchMultitoolRequestDraft({ desired_output: e.target.value })}
                    rows={2}
                  />
                </div>
                <div className="grid-3">
                  <div className="row">
                    <label>{t("multitool_request_status")}</label>
                    <select
                      value={multitoolRequestDraft.status}
                      onChange={(e) => patchMultitoolRequestDraft({ status: e.target.value })}
                    >
                      {REQUEST_STATUS_OPTIONS.map((item) => (
                        <option key={`request-status-${item}`} value={item}>
                          {humanizeToken(item)}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="row">
                    <label>{t("multitool_request_priority")}</label>
                    <select
                      value={multitoolRequestDraft.priority}
                      onChange={(e) => patchMultitoolRequestDraft({ priority: e.target.value })}
                    >
                      {TASK_PRIORITY_OPTIONS.map((item) => (
                        <option key={`request-priority-${item}`} value={item}>
                          {humanizeToken(item)}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="row">
                    <label>{t("multitool_request_layout")}</label>
                    <select
                      value={multitoolRequestDraft.layout_mode}
                      onChange={(e) => patchMultitoolRequestDraft({ layout_mode: e.target.value })}
                    >
                      <option value="graph">graph</option>
                      <option value="cards">cards</option>
                      <option value="charts">charts</option>
                      <option value="lists">lists</option>
                      <option value="taskboard">taskboard</option>
                    </select>
                  </div>
                </div>
                <div className="grid-2">
                  <div className="row">
                    <label>{t("multitool_domain")}</label>
                    <select
                      value={multitoolRequestDraft.domain}
                      onChange={(e) => patchMultitoolRequestDraft({ domain: e.target.value })}
                    >
                      {MULTITOOL_DOMAIN_OPTIONS.map((item) => (
                        <option key={`request-domain-${item}`} value={item}>
                          {humanizeToken(item)}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="row">
                    <label>{t("multitool_country")}</label>
                    <select
                      value={multitoolRequestDraft.country}
                      onChange={(e) => patchMultitoolRequestDraft({ country: e.target.value })}
                    >
                      {LEGISLATION_COUNTRY_OPTIONS.map((item) => (
                        <option key={`request-country-${item}`} value={item}>
                          {humanizeToken(item)}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <div className="row-actions">
                  <button type="button" disabled={busy} onClick={onSaveMultitoolRequest}>
                    {t("multitool_save")}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedRequestNodeId(0);
                      setMultitoolRequestDraft({ ...DEFAULT_MULTITOOL_REQUEST_DRAFT });
                    }}
                  >
                    {t("multitool_new_item")}
                  </button>
                </div>
                <h4>{t("multitool_request_result")}</h4>
                <pre>{stringifySafe(multitoolRequestSaveResult)}</pre>
              </article>

              <article className="multitool-card">
                <h3>{t("multitool_section_preferences")}</h3>
                <div className="row">
                  <label>{t("multitool_choose_existing")}</label>
                  <select
                    value={String(selectedPreferenceNodeId || 0)}
                    onChange={(e) => setSelectedPreferenceNodeId(Number(e.target.value || 0))}
                  >
                    <option value="0">-</option>
                    {multitoolPreferenceNodes.map((node) => {
                      const attrs = node?.attributes && typeof node.attributes === "object" ? node.attributes : {};
                      const profileName = String(attrs.profile_name || attrs.name || `#${node.id}`);
                      return (
                        <option key={`preference-node-${node.id}`} value={String(node.id)}>
                          {`${node.id} · ${profileName}`}
                        </option>
                      );
                    })}
                  </select>
                </div>
                <div className="row">
                  <label>{t("multitool_preferences_profile_name")}</label>
                  <input
                    value={multitoolPreferenceDraft.profile_name}
                    onChange={(e) => patchMultitoolPreferenceDraft({ profile_name: e.target.value })}
                  />
                </div>
                <div className="row">
                  <label>{t("multitool_preferences_likes")}</label>
                  <textarea
                    value={multitoolPreferenceDraft.likes_text}
                    onChange={(e) => patchMultitoolPreferenceDraft({ likes_text: e.target.value })}
                    rows={2}
                  />
                </div>
                <div className="row">
                  <label>{t("multitool_preferences_dislikes")}</label>
                  <textarea
                    value={multitoolPreferenceDraft.dislikes_text}
                    onChange={(e) => patchMultitoolPreferenceDraft({ dislikes_text: e.target.value })}
                    rows={2}
                  />
                </div>
                <div className="row">
                  <label>{t("multitool_preferences_style")}</label>
                  <textarea
                    value={multitoolPreferenceDraft.style_examples_text}
                    onChange={(e) => patchMultitoolPreferenceDraft({ style_examples_text: e.target.value })}
                    rows={2}
                  />
                </div>
                <div className="row">
                  <label>{t("multitool_preferences_tools")}</label>
                  <textarea
                    value={multitoolPreferenceDraft.tool_examples_text}
                    onChange={(e) => patchMultitoolPreferenceDraft({ tool_examples_text: e.target.value })}
                    rows={2}
                  />
                </div>
                <div className="row">
                  <label>{t("multitool_preferences_notes")}</label>
                  <textarea
                    value={multitoolPreferenceDraft.notes}
                    onChange={(e) => patchMultitoolPreferenceDraft({ notes: e.target.value })}
                    rows={3}
                  />
                </div>
                <div className="grid-2">
                  <div className="row">
                    <label>{t("multitool_domain")}</label>
                    <select
                      value={multitoolPreferenceDraft.domain}
                      onChange={(e) => patchMultitoolPreferenceDraft({ domain: e.target.value })}
                    >
                      {MULTITOOL_DOMAIN_OPTIONS.map((item) => (
                        <option key={`pref-domain-${item}`} value={item}>
                          {humanizeToken(item)}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="row">
                    <label>{t("multitool_country")}</label>
                    <select
                      value={multitoolPreferenceDraft.country}
                      onChange={(e) => patchMultitoolPreferenceDraft({ country: e.target.value })}
                    >
                      {LEGISLATION_COUNTRY_OPTIONS.map((item) => (
                        <option key={`pref-country-${item}`} value={item}>
                          {humanizeToken(item)}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <div className="row-actions">
                  <button type="button" disabled={busy} onClick={onSaveMultitoolPreference}>
                    {t("multitool_save")}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedPreferenceNodeId(0);
                      setMultitoolPreferenceDraft({ ...DEFAULT_MULTITOOL_PREFERENCE_DRAFT });
                    }}
                  >
                    {t("multitool_new_item")}
                  </button>
                </div>
                <h4>{t("multitool_preferences_result")}</h4>
                <pre>{stringifySafe(multitoolPreferenceSaveResult)}</pre>
              </article>

              <article className="multitool-card">
                <h3>{t("multitool_section_tasks")}</h3>
                <div className="row">
                  <label>{t("multitool_choose_existing")}</label>
                  <select
                    value={String(selectedTaskNodeId || 0)}
                    onChange={(e) => setSelectedTaskNodeId(Number(e.target.value || 0))}
                  >
                    <option value="0">-</option>
                    {multitoolTaskNodes.map((node) => {
                      const attrs = node?.attributes && typeof node.attributes === "object" ? node.attributes : {};
                      const title = String(attrs.title || attrs.name || `#${node.id}`);
                      return (
                        <option key={`task-node-${node.id}`} value={String(node.id)}>
                          {`${node.id} · ${title}`}
                        </option>
                      );
                    })}
                  </select>
                </div>
                <div className="row">
                  <label>{t("multitool_task_title")}</label>
                  <input
                    value={multitoolTaskDraft.title}
                    onChange={(e) => patchMultitoolTaskDraft({ title: e.target.value })}
                  />
                </div>
                <div className="row">
                  <label>{t("multitool_task_description")}</label>
                  <textarea
                    value={multitoolTaskDraft.description}
                    onChange={(e) => patchMultitoolTaskDraft({ description: e.target.value })}
                    rows={3}
                  />
                </div>
                <div className="grid-2">
                  <div className="row">
                    <label>{t("multitool_task_status")}</label>
                    <select
                      value={multitoolTaskDraft.status}
                      onChange={(e) => patchMultitoolTaskDraft({ status: e.target.value })}
                    >
                      {TASK_STATUS_OPTIONS.map((item) => (
                        <option key={`task-status-${item}`} value={item}>
                          {humanizeToken(item)}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="row">
                    <label>{t("multitool_task_priority")}</label>
                    <select
                      value={multitoolTaskDraft.priority}
                      onChange={(e) => patchMultitoolTaskDraft({ priority: e.target.value })}
                    >
                      {TASK_PRIORITY_OPTIONS.map((item) => (
                        <option key={`task-priority-${item}`} value={item}>
                          {humanizeToken(item)}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <div className="row">
                  <label>{t("multitool_task_due")}</label>
                  <input
                    value={multitoolTaskDraft.due_at}
                    onChange={(e) => patchMultitoolTaskDraft({ due_at: e.target.value })}
                    placeholder="YYYY-MM-DD"
                  />
                </div>
                <div className="grid-2">
                  <div className="row">
                    <label>{t("multitool_domain")}</label>
                    <select
                      value={multitoolTaskDraft.domain}
                      onChange={(e) => patchMultitoolTaskDraft({ domain: e.target.value })}
                    >
                      {MULTITOOL_DOMAIN_OPTIONS.map((item) => (
                        <option key={`task-domain-${item}`} value={item}>
                          {humanizeToken(item)}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="row">
                    <label>{t("multitool_country")}</label>
                    <select
                      value={multitoolTaskDraft.country}
                      onChange={(e) => patchMultitoolTaskDraft({ country: e.target.value })}
                    >
                      {LEGISLATION_COUNTRY_OPTIONS.map((item) => (
                        <option key={`task-country-${item}`} value={item}>
                          {humanizeToken(item)}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <div className="row-actions">
                  <button type="button" disabled={busy} onClick={onSaveMultitoolTask}>
                    {t("multitool_save")}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedTaskNodeId(0);
                      setMultitoolTaskDraft({ ...DEFAULT_MULTITOOL_TASK_DRAFT });
                    }}
                  >
                    {t("multitool_new_item")}
                  </button>
                </div>
                <h4>{t("multitool_task_result")}</h4>
                <pre>{stringifySafe(multitoolTaskSaveResult)}</pre>
              </article>

              <article className="multitool-card">
                <h3>{t("multitool_section_risks")}</h3>
                <div className="row">
                  <label>{t("multitool_choose_existing")}</label>
                  <select
                    value={String(selectedRiskNodeId || 0)}
                    onChange={(e) => setSelectedRiskNodeId(Number(e.target.value || 0))}
                  >
                    <option value="0">-</option>
                    {multitoolRiskNodes.map((node) => {
                      const attrs = node?.attributes && typeof node.attributes === "object" ? node.attributes : {};
                      const title = String(attrs.title || attrs.name || `#${node.id}`);
                      return (
                        <option key={`risk-node-${node.id}`} value={String(node.id)}>
                          {`${node.id} · ${title}`}
                        </option>
                      );
                    })}
                  </select>
                </div>
                <div className="row">
                  <label>{t("multitool_risk_title")}</label>
                  <input
                    value={multitoolRiskDraft.title}
                    onChange={(e) => patchMultitoolRiskDraft({ title: e.target.value })}
                  />
                </div>
                <div className="row">
                  <label>{t("multitool_risk_description")}</label>
                  <textarea
                    value={multitoolRiskDraft.description}
                    onChange={(e) => patchMultitoolRiskDraft({ description: e.target.value })}
                    rows={3}
                  />
                </div>
                <div className="grid-2">
                  <div className="row">
                    <label>{t("multitool_risk_probability")}</label>
                    <select
                      value={multitoolRiskDraft.probability}
                      onChange={(e) => patchMultitoolRiskDraft({ probability: e.target.value })}
                    >
                      {RISK_PROBABILITY_OPTIONS.map((item) => (
                        <option key={`risk-prob-${item}`} value={item}>
                          {humanizeToken(item)}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="row">
                    <label>{t("multitool_risk_impact")}</label>
                    <select
                      value={multitoolRiskDraft.impact}
                      onChange={(e) => patchMultitoolRiskDraft({ impact: e.target.value })}
                    >
                      {RISK_IMPACT_OPTIONS.map((item) => (
                        <option key={`risk-impact-${item}`} value={item}>
                          {humanizeToken(item)}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <div className="row">
                  <label>{t("multitool_risk_mitigation")}</label>
                  <textarea
                    value={multitoolRiskDraft.mitigation_text}
                    onChange={(e) => patchMultitoolRiskDraft({ mitigation_text: e.target.value })}
                    rows={4}
                  />
                </div>
                <div className="grid-2">
                  <div className="row">
                    <label>{t("multitool_domain")}</label>
                    <select
                      value={multitoolRiskDraft.domain}
                      onChange={(e) => patchMultitoolRiskDraft({ domain: e.target.value })}
                    >
                      {MULTITOOL_DOMAIN_OPTIONS.map((item) => (
                        <option key={`risk-domain-${item}`} value={item}>
                          {humanizeToken(item)}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="row">
                    <label>{t("multitool_country")}</label>
                    <select
                      value={multitoolRiskDraft.country}
                      onChange={(e) => patchMultitoolRiskDraft({ country: e.target.value })}
                    >
                      {LEGISLATION_COUNTRY_OPTIONS.map((item) => (
                        <option key={`risk-country-${item}`} value={item}>
                          {humanizeToken(item)}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <div className="row-actions">
                  <button type="button" disabled={busy} onClick={onSaveMultitoolRisk}>
                    {t("multitool_save")}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedRiskNodeId(0);
                      setMultitoolRiskDraft({ ...DEFAULT_MULTITOOL_RISK_DRAFT });
                    }}
                  >
                    {t("multitool_new_item")}
                  </button>
                </div>
                <h4>{t("multitool_risk_result")}</h4>
                <pre>{stringifySafe(multitoolRiskSaveResult)}</pre>
              </article>

              <article className="multitool-card multitool-card-wide">
                <h3>{t("personal_tree_title")}</h3>
                <p className="multitool-empty">{t("personal_tree_subtitle")}</p>
                <div className="grid-3">
                  <div className="row">
                    <label>{t("personal_tree_topic")}</label>
                    <input
                      value={personalTreeIngestDraft.topic}
                      onChange={(e) => patchPersonalTreeIngestDraft({ topic: e.target.value })}
                    />
                  </div>
                  <div className="row">
                    <label>{t("personal_tree_title_field")}</label>
                    <input
                      value={personalTreeIngestDraft.title}
                      onChange={(e) => patchPersonalTreeIngestDraft({ title: e.target.value })}
                    />
                  </div>
                  <div className="row">
                    <label>{t("personal_tree_source_type")}</label>
                    <select
                      value={personalTreeIngestDraft.source_type}
                      onChange={(e) => patchPersonalTreeIngestDraft({ source_type: e.target.value })}
                    >
                      <option value="text">text</option>
                      <option value="article">article</option>
                      <option value="law">law</option>
                      <option value="note">note</option>
                      <option value="other">other</option>
                    </select>
                  </div>
                </div>
                <div className="grid-2">
                  <div className="row">
                    <label>{t("personal_tree_source_title")}</label>
                    <input
                      value={personalTreeIngestDraft.source_title}
                      onChange={(e) => patchPersonalTreeIngestDraft({ source_title: e.target.value })}
                    />
                  </div>
                  <div className="row">
                    <label>{t("personal_tree_source_url")}</label>
                    <input
                      value={personalTreeIngestDraft.source_url}
                      onChange={(e) => patchPersonalTreeIngestDraft({ source_url: e.target.value })}
                    />
                  </div>
                </div>
                <div className="row">
                  <label>{t("personal_tree_text")}</label>
                  <textarea
                    value={personalTreeIngestDraft.text}
                    onChange={(e) => patchPersonalTreeIngestDraft({ text: e.target.value })}
                    rows={5}
                  />
                </div>
                <div className="row">
                  <label>{t("personal_tree_max_points")}</label>
                  <input
                    type="number"
                    min={2}
                    max={12}
                    value={Number(personalTreeIngestDraft.max_points || 6)}
                    onChange={(e) => patchPersonalTreeIngestDraft({ max_points: Number(e.target.value || 6) })}
                  />
                </div>
                <div className="row-actions">
                  <button type="button" disabled={busy} onClick={onRunPersonalTreeIngest}>
                    {t("personal_tree_ingest_action")}
                  </button>
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => onRefreshPersonalTreeView(personalTreeSelectedNodeId || personalTreeFocusNodeId || 0)}
                  >
                    {t("personal_tree_refresh_tree")}
                  </button>
                </div>
                <h4>{t("personal_tree_extraction_result")}</h4>
                <pre>{stringifySafe(personalTreeIngestResult?.extraction || personalTreeIngestResult)}</pre>

                <h4>{t("personal_tree_note_title")}</h4>
                <div className="grid-2">
                  <div className="row">
                    <label>{t("personal_tree_note_title_field")}</label>
                    <input
                      value={personalTreeNoteDraft.title}
                      onChange={(e) => patchPersonalTreeNoteDraft({ title: e.target.value })}
                    />
                  </div>
                  <div className="row">
                    <label>{t("personal_tree_source_type")}</label>
                    <select
                      value={personalTreeNoteDraft.source_type}
                      onChange={(e) => patchPersonalTreeNoteDraft({ source_type: e.target.value })}
                    >
                      <option value="note">note</option>
                      <option value="article">article</option>
                      <option value="law">law</option>
                      <option value="text">text</option>
                      <option value="other">other</option>
                    </select>
                  </div>
                </div>
                <div className="row">
                  <label>{t("personal_tree_note_text")}</label>
                  <textarea
                    value={personalTreeNoteDraft.note}
                    onChange={(e) => patchPersonalTreeNoteDraft({ note: e.target.value })}
                    rows={4}
                  />
                </div>
                <div className="grid-2">
                  <div className="row">
                    <label>{t("personal_tree_note_tags")}</label>
                    <textarea
                      value={personalTreeNoteDraft.tags_text}
                      onChange={(e) => patchPersonalTreeNoteDraft({ tags_text: e.target.value })}
                      rows={2}
                    />
                  </div>
                  <div className="row">
                    <label>{t("personal_tree_note_links")}</label>
                    <textarea
                      value={personalTreeNoteDraft.links_text}
                      onChange={(e) => patchPersonalTreeNoteDraft({ links_text: e.target.value })}
                      rows={2}
                    />
                  </div>
                </div>
                <div className="grid-2">
                  <div className="row">
                    <label>{t("personal_tree_source_title")}</label>
                    <input
                      value={personalTreeNoteDraft.source_title}
                      onChange={(e) => patchPersonalTreeNoteDraft({ source_title: e.target.value })}
                    />
                  </div>
                  <div className="row">
                    <label>{t("personal_tree_source_url")}</label>
                    <input
                      value={personalTreeNoteDraft.source_url}
                      onChange={(e) => patchPersonalTreeNoteDraft({ source_url: e.target.value })}
                    />
                  </div>
                </div>
                <div className="row-actions">
                  <button type="button" disabled={busy} onClick={onSavePersonalTreeNote}>
                    {t("personal_tree_note_save_action")}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setPersonalTreeNoteDraft({ ...DEFAULT_PERSONAL_TREE_NOTE_DRAFT });
                      setPersonalTreeIngestDraft({ ...DEFAULT_PERSONAL_TREE_INGEST_DRAFT });
                    }}
                  >
                    {t("multitool_new_item")}
                  </button>
                </div>
                <h4>{t("personal_tree_note_result")}</h4>
                <pre>{stringifySafe(personalTreeNoteResult?.note || personalTreeNoteResult)}</pre>

                <h4>{t("personal_tree_small_window_title")}</h4>
                <div className="personal-tree-mini-window">
                  {personalTreeSnapshot.nodes.length ? (
                    <GraphCanvas
                      snapshot={personalTreeSnapshot}
                      t={t}
                      selectedNodeId={personalTreeSelectedNodeId}
                      selectedEdgeSig={null}
                      tracePath={null}
                      edgeEffectsBySig={{}}
                      onSelectNode={(nodeId) => {
                        setPersonalTreeFocusNodeId(Number(nodeId || 0));
                      }}
                      onSelectEdge={() => {}}
                    />
                  ) : (
                    <p className="multitool-empty">{t("personal_tree_no_tree")}</p>
                  )}
                </div>
                <div className="grid-2">
                  <div>
                    <h4>{t("personal_tree_sources")}</h4>
                    <ul className="insight-list">
                      {(personalTreePayload?.sources || []).length ? (
                        (personalTreePayload?.sources || []).slice(0, 8).map((row) => (
                          <li key={`tree-source-${row.node_id}`}>
                            <strong>{row.title || `#${row.node_id}`}</strong>
                            <span>{row.source_type || "text"}</span>
                            <span>{row.url || "-"}</span>
                          </li>
                        ))
                      ) : (
                        <li>
                          <span>{t("multitool_no_items")}</span>
                        </li>
                      )}
                    </ul>
                  </div>
                  <div>
                    <h4>{t("personal_tree_notes")}</h4>
                    <ul className="insight-list">
                      {(personalTreePayload?.notes || []).length ? (
                        (personalTreePayload?.notes || []).slice(0, 8).map((row) => (
                          <li key={`tree-note-${row.node_id}`}>
                            <strong>{row.title || `#${row.node_id}`}</strong>
                            <span>{(row.tags || []).join(", ") || "-"}</span>
                          </li>
                        ))
                      ) : (
                        <li>
                          <span>{t("multitool_no_items")}</span>
                        </li>
                      )}
                    </ul>
                  </div>
                </div>
                <div className="row-actions">
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => onRefreshPersonalTreeView(personalTreeSelectedNodeId || personalTreeFocusNodeId || 0)}
                  >
                    {t("personal_tree_refresh_tree")}
                  </button>
                </div>
                <h4>{t("personal_tree_view_result")}</h4>
                <pre>{stringifySafe(personalTreePayload?.stats || personalTreeViewResult)}</pre>
              </article>

              <article className="multitool-card multitool-card-wide">
                <h3>{t("multitool_ops_title")}</h3>
                <p className="multitool-empty">{t("multitool_ops_subtitle")}</p>
                <div className="multitool-ops-grid">
                  <section className="multitool-ops-panel">
                    <h4>{t("packages_title")}</h4>
                    <div className="grid-2">
                      <div className="row">
                        <label>{t("packages_name")}</label>
                        <input
                          value={packagesDraft.package_name}
                          onChange={(e) => patchPackagesDraft({ package_name: e.target.value })}
                        />
                      </div>
                      <div className="row">
                        <label>{t("packages_model_role")}</label>
                        <select
                          value={packagesDraft.model_role}
                          onChange={(e) => patchPackagesDraft({ model_role: e.target.value })}
                        >
                          {LLM_ROLE_OPTIONS.map((role) => (
                            <option key={`pkg-role-${role}`} value={role}>
                              {role}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                    <div className="row">
                      <label>{t("packages_model_path")}</label>
                      <input
                        value={packagesDraft.model_path}
                        onChange={(e) => patchPackagesDraft({ model_path: e.target.value })}
                      />
                    </div>
                    <div className="row">
                      <label>{t("packages_items")}</label>
                      <textarea
                        rows={4}
                        value={packagesDraft.items_text}
                        onChange={(e) => patchPackagesDraft({ items_text: e.target.value })}
                      />
                    </div>
                    <div className="grid-2">
                      <div className="row">
                        <label>{t("packages_restore_ids")}</label>
                        <input
                          value={packagesDraft.restore_ids_text}
                          onChange={(e) => patchPackagesDraft({ restore_ids_text: e.target.value })}
                        />
                      </div>
                      <div className="row">
                        <label>{t("packages_confirmation")}</label>
                        <input
                          value={packagesDraft.confirmation}
                          onChange={(e) => patchPackagesDraft({ confirmation: e.target.value })}
                        />
                      </div>
                    </div>
                    <div className="row row-checkbox">
                      <label>
                        <input
                          type="checkbox"
                          checked={Boolean(packagesDraft.classify_with_llm)}
                          onChange={(e) => patchPackagesDraft({ classify_with_llm: e.target.checked })}
                        />
                        <span>{t("rag_use_llm")}</span>
                      </label>
                      <label>
                        <input
                          type="checkbox"
                          checked={Boolean(packagesDraft.apply_changes)}
                          onChange={(e) => patchPackagesDraft({ apply_changes: e.target.checked })}
                        />
                        <span>{t("packages_apply_changes")}</span>
                      </label>
                    </div>
                    <div className="row-actions">
                      <button type="button" disabled={busy} onClick={() => onRunPackagesAction("list")}>
                        {t("packages_list")}
                      </button>
                      <button type="button" disabled={busy} onClick={() => onRunPackagesAction("store")}>
                        {t("packages_store")}
                      </button>
                      <button type="button" disabled={busy} onClick={() => onRunPackagesAction("purge")}>
                        {t("packages_purge")}
                      </button>
                      <button type="button" disabled={busy} onClick={() => onRunPackagesAction("restore")}>
                        {t("packages_restore")}
                      </button>
                    </div>
                    <div className="grid-2">
                      <div>
                        <h4>{t("memory_namespace")}</h4>
                        {packageNamespaceRows.length ? (
                          renderMiniBars(packageNamespaceRows, "pkg-namespace")
                        ) : (
                          <p className="multitool-empty">{t("multitool_no_items")}</p>
                        )}
                      </div>
                      <div>
                        <h4>{t("multitool_task_status")}</h4>
                        {packageStatusRows.length ? (
                          renderMiniBars(packageStatusRows, "pkg-status")
                        ) : (
                          <p className="multitool-empty">{t("multitool_no_items")}</p>
                        )}
                      </div>
                    </div>
                    <h4>{t("packages_result")}</h4>
                    <pre>{stringifySafe(packagesResult)}</pre>
                  </section>

                  <section className="multitool-ops-panel">
                    <h4>{t("memory_namespace_title")}</h4>
                    <div className="grid-3">
                      <div className="row">
                        <label>{t("memory_namespace")}</label>
                        <select
                          value={memoryNamespaceDraft.namespace}
                          onChange={(e) => patchMemoryNamespaceDraft({ namespace: e.target.value })}
                        >
                          {MEMORY_NAMESPACE_OPTIONS.map((name) => (
                            <option key={`mem-ns-${name}`} value={name}>
                              {name}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div className="row">
                        <label>{t("memory_scope")}</label>
                        <select
                          value={memoryNamespaceDraft.scope}
                          onChange={(e) => patchMemoryNamespaceDraft({ scope: e.target.value })}
                        >
                          {MEMORY_SCOPE_OPTIONS.map((name) => (
                            <option key={`mem-scope-${name}`} value={name}>
                              {name}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div className="row">
                        <label>{t("memory_min_score")}</label>
                        <input
                          value={memoryNamespaceDraft.min_score}
                          onChange={(e) => patchMemoryNamespaceDraft({ min_score: e.target.value })}
                        />
                      </div>
                    </div>
                    <div className="grid-2">
                      <div className="row">
                        <label>{t("memory_source_namespace")}</label>
                        <select
                          value={memoryNamespaceDraft.source_namespace}
                          onChange={(e) => patchMemoryNamespaceDraft({ source_namespace: e.target.value })}
                        >
                          <option value="">any</option>
                          {MEMORY_NAMESPACE_OPTIONS.map((name) => (
                            <option key={`mem-src-${name}`} value={name}>
                              {name}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div className="row">
                        <label>{t("memory_node_ids")}</label>
                        <input
                          value={memoryNamespaceDraft.node_ids_text}
                          onChange={(e) => patchMemoryNamespaceDraft({ node_ids_text: e.target.value })}
                        />
                      </div>
                    </div>
                    <div className="row">
                      <label>{t("memory_query")}</label>
                      <textarea
                        rows={3}
                        value={memoryNamespaceDraft.query}
                        onChange={(e) => patchMemoryNamespaceDraft({ query: e.target.value })}
                      />
                    </div>
                    <div className="row row-checkbox">
                      <label>
                        <input
                          type="checkbox"
                          checked={Boolean(memoryNamespaceDraft.apply_changes)}
                          onChange={(e) => patchMemoryNamespaceDraft({ apply_changes: e.target.checked })}
                        />
                        <span>{t("packages_apply_changes")}</span>
                      </label>
                    </div>
                    <div className="row-actions">
                      <button type="button" disabled={busy} onClick={onApplyMemoryNamespace}>
                        {t("memory_apply")}
                      </button>
                      <button type="button" disabled={busy} onClick={onViewMemoryNamespace}>
                        {t("memory_view")}
                      </button>
                    </div>
                    <div>
                      <h4>{t("memory_view")}</h4>
                      {memoryNamespaceRows.length ? (
                        renderMiniBars(memoryNamespaceRows, "memory-view")
                      ) : (
                        <p className="multitool-empty">{t("multitool_no_items")}</p>
                      )}
                    </div>
                    <h4>{t("memory_result")}</h4>
                    <pre>{stringifySafe(memoryNamespaceApplyResult || memoryNamespaceViewResult)}</pre>
                  </section>

                  <section className="multitool-ops-panel">
                    <h4>{t("rag_title")}</h4>
                    <div className="row">
                      <label>{t("rag_query")}</label>
                      <textarea
                        rows={3}
                        value={graphRagDraft.query}
                        onChange={(e) => patchGraphRagDraft({ query: e.target.value })}
                      />
                    </div>
                    <div className="grid-3">
                      <div className="row">
                        <label>{t("rag_top_k")}</label>
                        <input
                          type="number"
                          min={1}
                          max={20}
                          value={Number(graphRagDraft.top_k || 6)}
                          onChange={(e) => patchGraphRagDraft({ top_k: Number(e.target.value || 6) })}
                        />
                      </div>
                      <div className="row">
                        <label>{t("memory_scope")}</label>
                        <select
                          value={graphRagDraft.scope}
                          onChange={(e) => patchGraphRagDraft({ scope: e.target.value })}
                        >
                          {MEMORY_SCOPE_OPTIONS.map((name) => (
                            <option key={`rag-scope-${name}`} value={name}>
                              {name}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div className="row">
                        <label>{t("memory_namespace")}</label>
                        <select
                          value={graphRagDraft.namespace}
                          onChange={(e) => patchGraphRagDraft({ namespace: e.target.value })}
                        >
                          <option value="">any</option>
                          {MEMORY_NAMESPACE_OPTIONS.map((name) => (
                            <option key={`rag-ns-${name}`} value={name}>
                              {name}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                    <div className="grid-2">
                      <div className="row">
                        <label>{t("packages_model_role")}</label>
                        <select
                          value={graphRagDraft.model_role}
                          onChange={(e) => patchGraphRagDraft({ model_role: e.target.value })}
                        >
                          {LLM_ROLE_OPTIONS.map((role) => (
                            <option key={`rag-role-${role}`} value={role}>
                              {role}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div className="row">
                        <label>{t("packages_model_path")}</label>
                        <input
                          value={graphRagDraft.model_path}
                          onChange={(e) => patchGraphRagDraft({ model_path: e.target.value })}
                        />
                      </div>
                    </div>
                    <div className="row row-checkbox">
                      <label>
                        <input
                          type="checkbox"
                          checked={Boolean(graphRagDraft.use_llm)}
                          onChange={(e) => patchGraphRagDraft({ use_llm: e.target.checked })}
                        />
                        <span>{t("rag_use_llm")}</span>
                      </label>
                    </div>
                    <div className="row-actions">
                      <button type="button" disabled={busy} onClick={onRunGraphRagQuery}>
                        {t("rag_run")}
                      </button>
                    </div>
                    <h4>{t("rag_result")}</h4>
                    <pre>{stringifySafe(graphRagResult)}</pre>
                  </section>

                  <section className="multitool-ops-panel">
                    <h4>{t("contradiction_title")}</h4>
                    <div className="grid-3">
                      <div className="row">
                        <label>{t("memory_scope")}</label>
                        <select
                          value={contradictionScanDraft.scope}
                          onChange={(e) => patchContradictionScanDraft({ scope: e.target.value })}
                        >
                          {MEMORY_SCOPE_OPTIONS.map((name) => (
                            <option key={`contr-scope-${name}`} value={name}>
                              {name}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div className="row">
                        <label>{t("memory_namespace")}</label>
                        <select
                          value={contradictionScanDraft.namespace}
                          onChange={(e) => patchContradictionScanDraft({ namespace: e.target.value })}
                        >
                          <option value="">any</option>
                          {MEMORY_NAMESPACE_OPTIONS.map((name) => (
                            <option key={`contr-ns-${name}`} value={name}>
                              {name}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div className="row">
                        <label>{t("contradiction_apply_graph")}</label>
                        <select
                          value={String(Boolean(contradictionScanDraft.apply_to_graph))}
                          onChange={(e) =>
                            patchContradictionScanDraft({ apply_to_graph: String(e.target.value) === "true" })
                          }
                        >
                          <option value="true">true</option>
                          <option value="false">false</option>
                        </select>
                      </div>
                    </div>
                    <div className="grid-3">
                      <div className="row">
                        <label>{t("personal_tree_max_points")}</label>
                        <input
                          type="number"
                          min={10}
                          max={240}
                          value={Number(contradictionScanDraft.max_nodes || 120)}
                          onChange={(e) => patchContradictionScanDraft({ max_nodes: Number(e.target.value || 120) })}
                        />
                      </div>
                      <div className="row">
                        <label>{t("rag_top_k")}</label>
                        <input
                          type="number"
                          min={1}
                          max={120}
                          value={Number(contradictionScanDraft.top_k || 20)}
                          onChange={(e) => patchContradictionScanDraft({ top_k: Number(e.target.value || 20) })}
                        />
                      </div>
                      <div className="row">
                        <label>{t("memory_min_score")}</label>
                        <input
                          value={contradictionScanDraft.min_overlap}
                          onChange={(e) => patchContradictionScanDraft({ min_overlap: e.target.value })}
                        />
                      </div>
                    </div>
                    <div className="row-actions">
                      <button type="button" disabled={busy} onClick={onRunContradictionScan}>
                        {t("contradiction_run")}
                      </button>
                    </div>
                    <div>
                      {contradictionIssueRows.length ? (
                        renderMiniBars(contradictionIssueRows, "contradiction")
                      ) : (
                        <p className="multitool-empty">{t("multitool_no_items")}</p>
                      )}
                    </div>
                    <h4>{t("contradiction_result")}</h4>
                    <pre>{stringifySafe(contradictionScanResult)}</pre>

                    <h4>{t("task_risk_title")}</h4>
                    <div className="row">
                      <label>{t("task_risk_tasks")}</label>
                      <textarea
                        rows={4}
                        value={taskRiskDraft.tasks_text}
                        onChange={(e) => patchTaskRiskDraft({ tasks_text: e.target.value })}
                      />
                    </div>
                    <div className="row row-checkbox">
                      <label>
                        <input
                          type="checkbox"
                          checked={Boolean(taskRiskDraft.apply_to_graph)}
                          onChange={(e) => patchTaskRiskDraft({ apply_to_graph: e.target.checked })}
                        />
                        <span>{t("contradiction_apply_graph")}</span>
                      </label>
                    </div>
                    <div className="row-actions">
                      <button type="button" disabled={busy} onClick={onRunTaskRiskBoard}>
                        {t("task_risk_run")}
                      </button>
                    </div>
                    <div>
                      {taskRiskLevelRows.length ? (
                        renderMiniBars(taskRiskLevelRows, "task-risk-level")
                      ) : (
                        <p className="multitool-empty">{t("multitool_no_items")}</p>
                      )}
                    </div>
                    <h4>{t("task_risk_result")}</h4>
                    <pre>{stringifySafe(taskRiskResult)}</pre>
                  </section>

                  <section className="multitool-ops-panel">
                    <h4>{t("timeline_replay_title")}</h4>
                    <div className="grid-2">
                      <div className="row">
                        <label>{t("timeline_event_type")}</label>
                        <input
                          value={timelineReplayDraft.event_type}
                          onChange={(e) => patchTimelineReplayDraft({ event_type: e.target.value })}
                        />
                      </div>
                      <div className="row">
                        <label>{t("timeline_limit")}</label>
                        <input
                          type="number"
                          min={1}
                          max={3000}
                          value={Number(timelineReplayDraft.limit || 300)}
                          onChange={(e) => patchTimelineReplayDraft({ limit: Number(e.target.value || 300) })}
                        />
                      </div>
                    </div>
                    <div className="grid-2">
                      <div className="row">
                        <label>{t("timeline_from_ts")}</label>
                        <input
                          value={timelineReplayDraft.from_ts}
                          onChange={(e) => patchTimelineReplayDraft({ from_ts: e.target.value })}
                        />
                      </div>
                      <div className="row">
                        <label>{t("timeline_to_ts")}</label>
                        <input
                          value={timelineReplayDraft.to_ts}
                          onChange={(e) => patchTimelineReplayDraft({ to_ts: e.target.value })}
                        />
                      </div>
                    </div>
                    <div className="row-actions">
                      <button type="button" disabled={busy} onClick={onRunTimelineReplay}>
                        {t("timeline_run")}
                      </button>
                    </div>
                    <div>
                      {timelineEventRows.length ? (
                        renderMiniBars(timelineEventRows, "timeline-events")
                      ) : (
                        <p className="multitool-empty">{t("multitool_no_items")}</p>
                      )}
                    </div>
                    <h4>{t("timeline_result")}</h4>
                    <pre>{stringifySafe(timelineReplayResult)}</pre>

                    <h4>{t("policy_title")}</h4>
                    <div className="grid-2">
                      <div className="row">
                        <label>{t("policy_mode")}</label>
                        <select
                          value={llmPolicyDraft.mode}
                          onChange={(e) => patchLlmPolicyDraft({ mode: e.target.value })}
                        >
                          {LLM_POLICY_MODE_OPTIONS.map((mode) => (
                            <option key={`policy-mode-${mode}`} value={mode}>
                              {mode}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div className="row">
                        <label>{t("policy_merge_lists")}</label>
                        <select
                          value={String(Boolean(llmPolicyDraft.merge_lists))}
                          onChange={(e) => patchLlmPolicyDraft({ merge_lists: String(e.target.value) === "true" })}
                        >
                          <option value="true">true</option>
                          <option value="false">false</option>
                        </select>
                      </div>
                    </div>
                    <div className="grid-3">
                      <div className="row">
                        <label>{t("policy_trusted_sessions")}</label>
                        <textarea
                          rows={3}
                          value={llmPolicyDraft.trusted_sessions_text}
                          onChange={(e) => patchLlmPolicyDraft({ trusted_sessions_text: e.target.value })}
                        />
                      </div>
                      <div className="row">
                        <label>{t("policy_trusted_users")}</label>
                        <textarea
                          rows={3}
                          value={llmPolicyDraft.trusted_users_text}
                          onChange={(e) => patchLlmPolicyDraft({ trusted_users_text: e.target.value })}
                        />
                      </div>
                      <div className="row">
                        <label>{t("policy_allowed_actions")}</label>
                        <textarea
                          rows={3}
                          value={llmPolicyDraft.allow_actions_text}
                          onChange={(e) => patchLlmPolicyDraft({ allow_actions_text: e.target.value })}
                        />
                      </div>
                    </div>
                    <div className="row-actions">
                      <button type="button" disabled={busy} onClick={onLoadLlmPolicy}>
                        {t("policy_load")}
                      </button>
                      <button type="button" disabled={busy} onClick={onSaveLlmPolicy}>
                        {t("policy_save")}
                      </button>
                    </div>
                    <h4>{t("policy_result")}</h4>
                    <pre>{stringifySafe(llmPolicyResult)}</pre>
                  </section>

                  <section className="multitool-ops-panel">
                    <h4>{t("quality_title")}</h4>
                    <div className="row">
                      <label>{t("quality_queries")}</label>
                      <textarea
                        rows={3}
                        value={qualityDraft.sample_queries_text}
                        onChange={(e) => patchQualityDraft({ sample_queries_text: e.target.value })}
                      />
                    </div>
                    <div className="row-actions">
                      <button type="button" disabled={busy} onClick={onRunQualityHarness}>
                        {t("quality_run")}
                      </button>
                    </div>
                    <div>
                      {qualityNamespaceRows.length ? (
                        renderMiniBars(qualityNamespaceRows, "quality-namespace")
                      ) : (
                        <p className="multitool-empty">{t("multitool_no_items")}</p>
                      )}
                    </div>
                    <h4>{t("quality_result")}</h4>
                    <pre>{stringifySafe(qualityResult)}</pre>

                    <h4>{t("backup_title")}</h4>
                    <div className="grid-2">
                      <div className="row">
                        <label>{t("backup_label")}</label>
                        <input
                          value={backupDraft.label}
                          onChange={(e) => patchBackupDraft({ label: e.target.value })}
                        />
                      </div>
                      <div className="row">
                        <label>{t("backup_event_limit")}</label>
                        <input
                          type="number"
                          min={0}
                          max={10000}
                          value={Number(backupDraft.event_limit || 1000)}
                          onChange={(e) => patchBackupDraft({ event_limit: Number(e.target.value || 1000) })}
                        />
                      </div>
                    </div>
                    <div className="row row-checkbox">
                      <label>
                        <input
                          type="checkbox"
                          checked={Boolean(backupDraft.include_events)}
                          onChange={(e) => patchBackupDraft({ include_events: e.target.checked })}
                        />
                        <span>{t("backup_include_events")}</span>
                      </label>
                    </div>
                    <div className="row-actions">
                      <button type="button" disabled={busy} onClick={onCreateBackup}>
                        {t("backup_create")}
                      </button>
                    </div>
                    <div className="grid-2">
                      <div className="row">
                        <label>{t("backup_path")}</label>
                        <input
                          value={backupDraft.path}
                          onChange={(e) => patchBackupDraft({ path: e.target.value })}
                        />
                      </div>
                      <div className="row">
                        <label>{t("packages_confirmation")}</label>
                        <input
                          value={backupDraft.confirmation}
                          onChange={(e) => patchBackupDraft({ confirmation: e.target.value })}
                        />
                      </div>
                    </div>
                    <div className="row row-checkbox">
                      <label>
                        <input
                          type="checkbox"
                          checked={Boolean(backupDraft.latest)}
                          onChange={(e) => patchBackupDraft({ latest: e.target.checked })}
                        />
                        <span>{t("backup_latest")}</span>
                      </label>
                      <label>
                        <input
                          type="checkbox"
                          checked={Boolean(backupDraft.apply_changes)}
                          onChange={(e) => patchBackupDraft({ apply_changes: e.target.checked })}
                        />
                        <span>{t("packages_apply_changes")}</span>
                      </label>
                      <label>
                        <input
                          type="checkbox"
                          checked={Boolean(backupDraft.restore_policy)}
                          onChange={(e) => patchBackupDraft({ restore_policy: e.target.checked })}
                        />
                        <span>{t("backup_restore_policy")}</span>
                      </label>
                    </div>
                    <div className="row-actions">
                      <button type="button" disabled={busy} onClick={onRestoreBackup}>
                        {t("backup_restore")}
                      </button>
                    </div>
                    <h4>{t("backup_result")}</h4>
                    <pre>{stringifySafe({ create: backupCreateResult, restore: backupRestoreResult })}</pre>

                    <div className="grid-2">
                      <div className="row">
                        <label>{t("audit_limit")}</label>
                        <input
                          type="number"
                          min={1}
                          max={2000}
                          value={Number(auditDraft.limit || 200)}
                          onChange={(e) => patchAuditDraft({ limit: Number(e.target.value || 200) })}
                        />
                      </div>
                      <div className="row row-checkbox">
                        <label>
                          <input
                            type="checkbox"
                            checked={Boolean(auditDraft.include_backups)}
                            onChange={(e) => patchAuditDraft({ include_backups: e.target.checked })}
                          />
                          <span>{t("audit_include_backups")}</span>
                        </label>
                      </div>
                    </div>
                    <div className="row-actions">
                      <button type="button" disabled={busy} onClick={onLoadAuditLogs}>
                        {t("audit_load")}
                      </button>
                    </div>
                    <h4>{t("audit_result")}</h4>
                    <pre>{stringifySafe(auditResult)}</pre>
                  </section>
                </div>
              </article>
            </div>
            <div className="multitool-dashboard">
              <h3>{t("multitool_dashboard")}</h3>
              <div className="multitool-stats-grid">
                <div>
                  <h4>{t("multitool_chart_task_status")}</h4>
                  {multitoolTaskStatusRows.length ? (
                    renderMiniBars(multitoolTaskStatusRows, "multitool-task-status")
                  ) : (
                    <p className="multitool-empty">{t("multitool_no_items")}</p>
                  )}
                </div>
                <div>
                  <h4>{t("multitool_chart_task_priority")}</h4>
                  {multitoolTaskPriorityRows.length ? (
                    renderMiniBars(multitoolTaskPriorityRows, "multitool-task-priority")
                  ) : (
                    <p className="multitool-empty">{t("multitool_no_items")}</p>
                  )}
                </div>
                <div>
                  <h4>{t("multitool_chart_risk_probability")}</h4>
                  {multitoolRiskProbabilityRows.length ? (
                    renderMiniBars(multitoolRiskProbabilityRows, "multitool-risk-prob")
                  ) : (
                    <p className="multitool-empty">{t("multitool_no_items")}</p>
                  )}
                </div>
                <div>
                  <h4>{t("multitool_chart_risk_impact")}</h4>
                  {multitoolRiskImpactRows.length ? (
                    renderMiniBars(multitoolRiskImpactRows, "multitool-risk-impact")
                  ) : (
                    <p className="multitool-empty">{t("multitool_no_items")}</p>
                  )}
                </div>
              </div>
              <div className="multitool-stats-grid">
                <div>
                  <h4>{t("multitool_chart_domain_coverage")}</h4>
                  {multitoolDomainCoverageRows.length ? (
                    renderMiniBars(multitoolDomainCoverageRows, "multitool-domain")
                  ) : (
                    <p className="multitool-empty">{t("multitool_no_items")}</p>
                  )}
                </div>
                <div>
                  <h4>{t("multitool_open_tasks")}</h4>
                  <ul className="insight-list">
                    {multitoolOpenTasks.length ? (
                      multitoolOpenTasks.slice(0, 6).map((node) => {
                        const attrs = node?.attributes && typeof node.attributes === "object" ? node.attributes : {};
                        return (
                          <li key={`open-task-${node.id}`}>
                            <strong>{String(attrs.title || attrs.name || `#${node.id}`)}</strong>
                            <span>{`${attrs.status || "backlog"} · ${attrs.priority || "medium"}`}</span>
                          </li>
                        );
                      })
                    ) : (
                      <li>
                        <span>{t("multitool_no_items")}</span>
                      </li>
                    )}
                  </ul>
                </div>
                <div>
                  <h4>{t("multitool_top_risks")}</h4>
                  <ul className="insight-list">
                    {multitoolTopRisks.length ? (
                      multitoolTopRisks.map((node) => {
                        const attrs = node?.attributes && typeof node.attributes === "object" ? node.attributes : {};
                        return (
                          <li key={`top-risk-${node.id}`}>
                            <strong>{String(attrs.title || attrs.name || `#${node.id}`)}</strong>
                            <span>{`${attrs.probability || "medium"} · ${attrs.impact || "medium"}`}</span>
                          </li>
                        );
                      })
                    ) : (
                      <li>
                        <span>{t("multitool_no_items")}</span>
                      </li>
                    )}
                  </ul>
                </div>
              </div>
              <div className="multitool-stats-grid">
                <div>
                  <h4>{t("multitool_widget_contradictions")}</h4>
                  <ul className="insight-list">
                    {contradictionTopIssues.length ? (
                      contradictionTopIssues.map((row, idx) => (
                        <li key={`contr-top-${idx}`}>
                          <strong>{String(row?.left_preview || `#${row?.left_node_id || idx + 1}`)}</strong>
                          <span>{String(row?.right_preview || `#${row?.right_node_id || idx + 1}`)}</span>
                          <span>{`score ${(Number(row?.score || 0) || 0).toFixed(2)}`}</span>
                        </li>
                      ))
                    ) : (
                      <li>
                        <span>{t("multitool_no_items")}</span>
                      </li>
                    )}
                  </ul>
                </div>
                <div>
                  <h4>{t("multitool_widget_quality_trend")}</h4>
                  {qualityTrendRows.length ? (
                    renderMiniBars(qualityTrendRows, "quality-trend")
                  ) : (
                    <p className="multitool-empty">{t("multitool_no_items")}</p>
                  )}
                </div>
                <div>
                  <h4>{t("multitool_widget_backup_history")}</h4>
                  <ul className="insight-list">
                    {backupHistoryRows.length ? (
                      backupHistoryRows.map((row, idx) => {
                        const path = String(row?.path || "");
                        const name = path ? path.split("/").pop() : `backup-${idx + 1}`;
                        return (
                          <li key={`backup-history-${idx}`}>
                            <strong>{name}</strong>
                            <span>{formatEpochLabel(row?.modified_at)}</span>
                            <span>{`${Number(row?.size_bytes || 0)} B`}</span>
                          </li>
                        );
                      })
                    ) : (
                      <li>
                        <span>{t("multitool_no_items")}</span>
                      </li>
                    )}
                  </ul>
                </div>
              </div>
            </div>
          </section>
        )}

        {sectionKey === "autoruns" && (
          <section className="card grid-2">
            <div>
              <h2>{t("autoruns_import_title")}</h2>
              <p>{t("autoruns_import_help")}</p>
              <div className="row">
                <label>{t("autoruns_input_or_query")}</label>
                <textarea
                  value={autorunsImportText}
                  onChange={(e) => setAutorunsImportText(e.target.value)}
                  rows={10}
                  placeholder={t("autoruns_placeholder")}
                />
              </div>
              <div className="row-actions">
                <button disabled={busy} onClick={onImportAutoruns}>
                  {t("import_autoruns")}
                </button>
              </div>
            </div>
            <div>
              <h2>{t("autoruns_import_result")}</h2>
              <pre>{stringifySafe(autorunsImportResult)}</pre>
            </div>
          </section>
        )}

        {sectionKey === "graph" && (
          <section className="card">
            <h2>{t("graph_visualization")}</h2>
            <GraphCanvas
              snapshot={snapshot}
              t={t}
              selectedNodeId={selectedNodeId}
              selectedEdgeSig={selectedEdgeSig}
              tracePath={
                selectedTrace
                  ? {
                      targetNodeId: selectedNodeId,
                      nodeIds: selectedTrace.nodeIds,
                      edgeSigs: selectedTrace.edgeSigs,
                    }
                  : null
              }
              edgeEffectsBySig={edgeEffectsBySig}
              onSelectNode={setSelectedNodeId}
              onSelectEdge={setSelectedEdgeSig}
            />
            <p className="graph-hint">{t("graph_hover_dependencies_hint")}</p>
            <div className="reasoning-panel">
              <h3>{t("reasoning_path")}</h3>
              {!selectedNodeExploration.hasTarget ? (
                <p className="reasoning-empty">{t("reasoning_path_empty")}</p>
              ) : (
                <>
                  <div className="trace-variants-header">{t("reasoning_trace_options")}</div>
                  <div className="trace-variants">
                    {selectedTraceOptions.length ? (
                      selectedTraceOptions.map((variant, index) => (
                        <button
                          key={variant.id}
                          type="button"
                          className={`trace-chip ${safeTraceIndex === index ? "active" : ""}`}
                          onClick={() => setSelectedTraceIndex(index)}
                        >
                          {`#${index + 1} · ${variant.depth}h · ${t("reasoning_trace_score")} ${variant.score.toFixed(2)}`}
                        </button>
                      ))
                    ) : (
                      <span className="trace-empty">{t("reasoning_trace_empty")}</span>
                    )}
                  </div>
                  <div className="reasoning-stats">
                    <span>{`${t("reasoning_roots")}: ${selectedNodeExploration.rootNodeIds.length}`}</span>
                    <span>{`${t("reasoning_prerequisites")}: ${selectedNodeExploration.ancestorNodeIds.length}`}</span>
                    <span>{`${t("reasoning_dependents")}: ${selectedNodeExploration.descendantNodeIds.length}`}</span>
                  </div>
                  <div className="reasoning-row">
                    <strong>{t("reasoning_roots")}:</strong>
                    <span>{selectedReasoningRootsText || "-"}</span>
                  </div>
                  <div className="reasoning-row">
                    <strong>{t("reasoning_chain")}:</strong>
                    <span>{selectedReasoningChainText || "-"}</span>
                  </div>
                  <div className="reasoning-row">
                    <strong>{t("reasoning_prerequisites")}:</strong>
                    <span>{selectedReasoningPrerequisitesText || "-"}</span>
                  </div>
                  <div className="reasoning-row">
                    <strong>{t("reasoning_dependents")}:</strong>
                    <span>{selectedReasoningDependentsText || "-"}</span>
                  </div>
                </>
              )}
            </div>
            <div className="graph-toolkit-grid">
              <section className="branch-visual-panel">
                <div className="branch-visual-head">
                  <h3>{t("branch_visual_toolkit")}</h3>
                  <span className="branch-scope-chip">
                    {`${t("branch_scope")}: ${
                      branchInsights.hasTarget ? t("branch_scope_selected") : t("branch_scope_global")
                    }`}
                  </span>
                </div>
                <div className="insight-tabs">
                  {BRANCH_INSIGHT_VIEW_OPTIONS.map((mode) => (
                    <button
                      key={`insight-mode-${mode}`}
                      type="button"
                      className={`insight-tab ${branchInsightView === mode ? "active" : ""}`}
                      onClick={() => setBranchInsightView(mode)}
                    >
                      {t(`branch_visual_mode_${mode}`)}
                    </button>
                  ))}
                </div>
                {branchInsightView === "cards" && (
                  <div className="insight-metrics-grid">
                    <article className="insight-metric-card">
                      <strong>{t("branch_metric_nodes")}</strong>
                      <span>{branchInsights.nodeCount}</span>
                    </article>
                    <article className="insight-metric-card">
                      <strong>{t("branch_metric_edges")}</strong>
                      <span>{branchInsights.edgeCount}</span>
                    </article>
                    <article className="insight-metric-card">
                      <strong>{t("branch_metric_avg_weight")}</strong>
                      <span>{branchInsights.avgWeight.toFixed(2)}</span>
                    </article>
                    <article className="insight-metric-card">
                      <strong>{t("branch_metric_hints")}</strong>
                      <span>{branchInsights.lifehacks.length}</span>
                    </article>
                  </div>
                )}
                {branchInsightView === "charts" && (
                  <div className="insight-chart-grid">
                    <div>
                      <h4>{t("branch_top_relations")}</h4>
                      {renderMiniBars(branchInsights.topRelations, "relation")}
                    </div>
                    <div>
                      <h4>{t("branch_top_node_types")}</h4>
                      {renderMiniBars(branchInsights.topNodeTypes, "node-type")}
                    </div>
                  </div>
                )}
                {branchInsightView === "lists" && (
                  <div className="insight-list-grid">
                    <div>
                      <h4>{t("branch_top_nodes")}</h4>
                      <ul className="insight-list">
                        {(branchInsights.topNodes || []).map((node) => (
                          <li key={`insight-node-${node.id}`}>
                            <strong>{node.label}</strong>
                            <span>{`${node.type} · degree ${node.degree}`}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <h4>{t("branch_top_edges")}</h4>
                      <ul className="insight-list">
                        {(branchInsights.topEdges || []).map((edge) => (
                          <li key={`insight-edge-${edge.signature}`}>
                            <strong>{edge.label}</strong>
                            <span>{`${edge.relation} · weight ${edge.weight.toFixed(2)}`}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}
                {branchInsightView === "tips" && (
                  <div>
                    <h4>{t("branch_lifehacks")}</h4>
                    <ul className="insight-list">
                      {(branchInsights.lifehacks || []).map((hint, index) => (
                        <li key={`insight-hint-${index}`}>
                          <span>{hint}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                <div className="insight-editor">
                  <h4>{t("branch_report_editor_title")}</h4>
                  <div className="row">
                    <label>{t("branch_report_summary")}</label>
                    <textarea
                      value={branchReportSummaryText}
                      onChange={(e) => setBranchReportSummaryText(e.target.value)}
                      rows={3}
                    />
                  </div>
                  <div className="row">
                    <label>{t("branch_report_tips")}</label>
                    <textarea
                      value={branchReportTipsText}
                      onChange={(e) => setBranchReportTipsText(e.target.value)}
                      rows={5}
                    />
                  </div>
                  <div className="row-actions">
                    <button type="button" disabled={busy} onClick={onSaveBranchReportGraph}>
                      {t("branch_report_save")}
                    </button>
                  </div>
                  <h4>{t("branch_report_saved_result")}</h4>
                  <pre>{stringifySafe(branchReportSaveResult)}</pre>
                </div>
              </section>
              <section className="style-node-panel">
                <h3>{t("style_nodes_title")}</h3>
                <p className="style-node-hint">{t("style_nodes_safe_hint")}</p>
                <div className="row">
                  <label>{t("style_nodes_slider_label")}</label>
                  <input
                    type="range"
                    min={0}
                    max={Math.max(0, STYLE_NODE_PRESETS.length - 1)}
                    step={1}
                    value={Math.max(0, Math.min(STYLE_NODE_PRESETS.length - 1, Number(styleNodeIndex || 0)))}
                    onChange={(e) => setStyleNodeIndex(Number(e.target.value || 0))}
                  />
                </div>
                <div className="style-node-current">
                  <strong>{activeStylePreset?.name || "n/a"}</strong>
                  <span>{activeStylePreset?.description || ""}</span>
                </div>
                <div className="row">
                  <label>{t("style_nodes_name")}</label>
                  <input
                    value={styleNodeDraftName}
                    onChange={(e) => setStyleNodeDraftName(e.target.value)}
                  />
                </div>
                <div className="row">
                  <label>{t("style_nodes_description")}</label>
                  <textarea
                    value={styleNodeDraftDescription}
                    onChange={(e) => setStyleNodeDraftDescription(e.target.value)}
                    rows={2}
                  />
                </div>
                <div className="row">
                  <label>{t("style_nodes_vars_json")}</label>
                  <textarea
                    value={styleNodeDraftVarsText}
                    onChange={(e) => setStyleNodeDraftVarsText(e.target.value)}
                    rows={6}
                  />
                </div>
                <div className="style-node-list">
                  {STYLE_NODE_PRESETS.map((preset, index) => {
                    const isActive = index === Number(styleNodeIndex || 0);
                    return (
                      <article key={preset.id} className={`style-node-card ${isActive ? "active" : ""}`}>
                        <div className="style-node-card-head">
                          <strong>{preset.name}</strong>
                          {isActive && <span className="style-node-badge">{t("style_nodes_active")}</span>}
                        </div>
                        <p>{preset.description}</p>
                        <button type="button" onClick={() => setStyleNodeIndex(index)}>
                          {t("style_nodes_activate")}
                        </button>
                      </article>
                    );
                  })}
                </div>
                <div className="row-actions">
                  <button type="button" disabled={busy} onClick={onSaveStyleNodeGraph}>
                    {t("style_nodes_save")}
                  </button>
                  <button type="button" onClick={() => setStyleNodeIndex(0)}>
                    {t("style_nodes_reset")}
                  </button>
                </div>
                <h4>{t("style_nodes_saved_result")}</h4>
                <pre>{stringifySafe(styleNodeSaveResult)}</pre>
              </section>
            </div>
            <div className="grid-2" style={{ marginTop: "14px" }}>
              <div>
                <h2>{t("selected_node_editor")}</h2>
                <pre>{stringifySafe(selectedNode)}</pre>
                <div className="row">
                  <label>{t("attributes_json")}</label>
                  <textarea
                    value={selectedNodeAttributesText}
                    onChange={(e) => setSelectedNodeAttributesText(e.target.value)}
                    rows={8}
                    disabled={!selectedNode}
                  />
                </div>
                <div className="row">
                  <label>{t("state_json")}</label>
                  <textarea
                    value={selectedNodeStateText}
                    onChange={(e) => setSelectedNodeStateText(e.target.value)}
                    rows={5}
                    disabled={!selectedNode}
                  />
                </div>
                <div className="row-actions">
                  <button disabled={busy || !selectedNode} onClick={onUpdateSelectedNode}>
                    {t("update_node")}
                  </button>
                  <button disabled={busy || !selectedNode} onClick={onDeleteSelectedNode}>
                    {t("delete_node")}
                  </button>
                </div>
              </div>
              <div>
                <h2>{t("selected_edge_editor")}</h2>
                <div className="edge-reasoning-panel">
                  <div className="edge-reasoning-title">{t("edge_reasoning")}</div>
                  {!selectedEdge ? (
                    <p className="edge-reasoning-empty">{t("edge_reasoning_empty")}</p>
                  ) : (
                    <>
                      <div className="edge-reasoning-summary">{selectedEdgeReasoning.summary}</div>
                      <div className="edge-reasoning-grid">
                        <div>
                          <strong>{t("edge_reasoning_relation")}</strong>
                          <span>{selectedEdgeReasoning.relation}</span>
                        </div>
                        <div>
                          <strong>{t("edge_reasoning_logic")}</strong>
                          <span>{selectedEdgeReasoning.logic}</span>
                        </div>
                        <div>
                          <strong>{t("edge_reasoning_direction")}</strong>
                          <span>{selectedEdgeReasoning.direction}</span>
                        </div>
                        <div>
                          <strong>{t("edge_reasoning_strength")}</strong>
                          <span>{selectedEdgeReasoning.strength}</span>
                        </div>
                      </div>
                      <div className="edge-reasoning-facts">
                        <strong>{t("edge_reasoning_facts")}</strong>
                        <ul>
                          {(selectedEdgeReasoning.facts || []).map((fact, idx) => (
                            <li key={`edge-fact-${idx}`}>{fact}</li>
                          ))}
                        </ul>
                      </div>
                    </>
                  )}
                </div>
                <div className="edge-history-panel">
                  <div className="edge-history-title">{t("edge_history_timeline")}</div>
                  {!selectedEdge || !selectedEdgeHistory.length ? (
                    <p className="edge-history-empty">{t("edge_history_empty")}</p>
                  ) : (
                    <div className="edge-history-list">
                      {selectedEdgeHistory.map((item) => (
                        <article key={`edge-history-${item.id}-${item.eventType}`} className="edge-history-item">
                          <div className="edge-history-dot" />
                          <div className="edge-history-content">
                            <div className="edge-history-head">
                              <span className="edge-history-event">
                                {`${t("edge_history_event")}: ${humanizeToken(item.eventType)}`}
                              </span>
                              <time className="edge-history-time">{item.timeLabel || "-"}</time>
                            </div>
                            <div className="edge-history-meta">
                              <span>{`${t("edge_history_weight")}: ${item.weightLabel}`}</span>
                              <span>{`${t("edge_history_logic")}: ${item.logic}`}</span>
                            </div>
                            <div className="edge-history-reason">
                              <strong>{`${t("edge_history_reason")}:`}</strong>
                              <span>{item.reason}</span>
                            </div>
                          </div>
                        </article>
                      ))}
                    </div>
                  )}
                </div>
                <pre>{stringifySafe(selectedEdge)}</pre>
                <div className="row">
                  <label>{t("weight")}</label>
                  <input
                    value={selectedEdgeWeight}
                    onChange={(e) => setSelectedEdgeWeight(e.target.value)}
                    disabled={!selectedEdge}
                  />
                </div>
                <div className="row">
                  <label>{t("logic_rule")}</label>
                  <input
                    value={selectedEdgeLogicRule}
                    onChange={(e) => setSelectedEdgeLogicRule(e.target.value)}
                    disabled={!selectedEdge}
                  />
                </div>
                <div className="row">
                  <label>{t("metadata_json")}</label>
                  <textarea
                    value={selectedEdgeMetadataText}
                    onChange={(e) => setSelectedEdgeMetadataText(e.target.value)}
                    rows={6}
                    disabled={!selectedEdge}
                  />
                </div>
                <div className="row-actions">
                  <button disabled={busy || !selectedEdge} onClick={onUpdateSelectedEdge}>
                    {t("update_edge")}
                  </button>
                  <button disabled={busy || !selectedEdge} onClick={onDeleteSelectedEdge}>
                    {t("delete_edge")}
                  </button>
                </div>
              </div>
            </div>
          </section>
        )}

        {sectionKey === "client" && (
          <section className="card">
            <h2>{t("client_profile_semantic_input")}</h2>
            <pre>{stringifySafe(clientProfile)}</pre>
          </section>
        )}

        {sectionKey === "advisors" && (
          <>
            <section className="card">
              <h2>{t("llm_role_debate")}</h2>
              <div className="row">
                <label>{t("debate_prompt")}</label>
                <textarea
                  value={debatePromptText}
                  onChange={(e) => setDebatePromptText(e.target.value)}
                  rows={4}
                  placeholder={t("debate_prompt_placeholder")}
                />
              </div>
              <div className="grid-2">
                <div className="row">
                  <label>{t("debate_variants")}</label>
                  <input
                    value={debateHypothesisCount}
                    onChange={(e) => setDebateHypothesisCount(e.target.value)}
                  />
                </div>
                <div className="row row-checkbox">
                  <label>{t("debate_attach_graph")}</label>
                  <input
                    type="checkbox"
                    checked={debateAttachGraph}
                    onChange={(e) => setDebateAttachGraph(Boolean(e.target.checked))}
                  />
                </div>
              </div>
              <div className="grid-3">
                <div className="row">
                  <label>proposer</label>
                  <select value={debateProposerRole} onChange={(e) => setDebateProposerRole(e.target.value)}>
                    {LLM_ROLE_OPTIONS.map((role) => (
                      <option key={`proposer-${role}`} value={role}>
                        {role}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="row">
                  <label>critic</label>
                  <select value={debateCriticRole} onChange={(e) => setDebateCriticRole(e.target.value)}>
                    {LLM_ROLE_OPTIONS.map((role) => (
                      <option key={`critic-${role}`} value={role}>
                        {role}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="row">
                  <label>judge</label>
                  <select value={debateJudgeRole} onChange={(e) => setDebateJudgeRole(e.target.value)}>
                    {LLM_ROLE_OPTIONS.map((role) => (
                      <option key={`judge-${role}`} value={role}>
                        {role}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <div className="row-actions">
                <button disabled={busy} onClick={onRunLLMDebate}>
                  {t("debate_run")}
                </button>
              </div>
            </section>

            <section className="card grid-2">
              <div>
                <h2>{t("debate_result")}</h2>
                <pre>{stringifySafe(debateResult)}</pre>
              </div>
              <div>
                <h2>{t("mini_coders_advisors")}</h2>
                <pre>{stringifySafe(modelAdvisors?.advisors || modelAdvisors || {})}</pre>
              </div>
            </section>

            <section className="card grid-2">
              <div>
                <h2>{t("archive_chat_title")}</h2>
                <h3>{t("archive_chat_history")}</h3>
                <div className="chat-log">
                  {(archiveChatMessages || []).length ? (
                    (archiveChatMessages || []).map((item) => (
                      <div key={`archive-chat-msg-${item.ts}-${item.role}`}>
                        <strong>{item.role === "user" ? "You" : "Assistant"}:</strong> {String(item.text || "")}
                      </div>
                    ))
                  ) : (
                    <div>{t("archive_chat_no_messages")}</div>
                  )}
                </div>
                <div className="row">
                  <label>{t("archive_chat_message")}</label>
                  <textarea
                    value={archiveChatMessageText}
                    onChange={(e) => setArchiveChatMessageText(e.target.value)}
                    rows={4}
                    placeholder={t("archive_chat_message_placeholder")}
                  />
                </div>
                <div className="row">
                  <label>{t("archive_chat_context")}</label>
                  <textarea
                    value={archiveChatContextText}
                    onChange={(e) => setArchiveChatContextText(e.target.value)}
                    rows={3}
                    placeholder={t("archive_chat_context_placeholder")}
                  />
                </div>
                <div className="grid-2">
                  <div className="row">
                    <label>{t("archive_chat_model_path")}</label>
                    <select value={archiveChatModelPath} onChange={(e) => setArchiveChatModelPath(e.target.value)}>
                      <option value="">auto</option>
                      {archiveChatModelOptions.map((path) => (
                        <option key={`archive-model-${path}`} value={path}>
                          {path}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="row">
                    <label>{t("archive_chat_model_role")}</label>
                    <select value={archiveChatModelRole} onChange={(e) => setArchiveChatModelRole(e.target.value)}>
                      {LLM_ROLE_OPTIONS.map((role) => (
                        <option key={`archive-role-${role}`} value={role}>
                          {role}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <div className="grid-2">
                  <div className="row">
                    <label>{t("archive_chat_verification_mode")}</label>
                    <select
                      value={archiveChatVerificationMode}
                      onChange={(e) => setArchiveChatVerificationMode(e.target.value)}
                    >
                      <option value="strict">strict</option>
                      <option value="balanced">balanced</option>
                    </select>
                  </div>
                  <div className="row row-checkbox">
                    <label>{t("archive_chat_attach_graph")}</label>
                    <input
                      type="checkbox"
                      checked={archiveChatAttachGraph}
                      onChange={(e) => setArchiveChatAttachGraph(Boolean(e.target.checked))}
                    />
                  </div>
                </div>
                <div className="row-actions">
                  <button disabled={busy} onClick={onRunArchiveChat}>
                    {t("archive_chat_run")}
                  </button>
                </div>
                <h3>{t("archive_chat_suggestions")}</h3>
                <pre>{stringifySafe(advisorDetectedModels)}</pre>
              </div>
              <div>
                <h2>{t("archive_review_title")}</h2>
                <div className="row">
                  <label>{t("archive_review_editor")}</label>
                  <textarea
                    value={archiveReviewUpdatesText}
                    onChange={(e) => setArchiveReviewUpdatesText(e.target.value)}
                    rows={14}
                    placeholder={t("archive_review_editor_placeholder")}
                  />
                </div>
                <div className="row-actions">
                  <button disabled={busy} onClick={onApplyArchiveReview}>
                    {t("archive_review_apply")}
                  </button>
                </div>
                <h3>{t("archive_review_result")}</h3>
                <pre>{stringifySafe(archiveReviewResult)}</pre>
                <h3>{t("archive_chat_result")}</h3>
                <pre>{stringifySafe(archiveChatResult?.verification || archiveChatResult || {})}</pre>
              </div>
            </section>

            <section className="card">
              <h2>{t("prompt_catalog")}</h2>
              <pre>{stringifySafe(modelAdvisors?.prompts || [])}</pre>
            </section>
          </>
        )}

        {sectionKey === "hallucination_hunter" && (
          <section className="card grid-2">
            <div>
              <h2>{t("hallucination_report_title")}</h2>
              <div className="row">
                <label>{t("hallucination_prompt")}</label>
                <textarea
                  value={hallucinationPromptText}
                  onChange={(e) => setHallucinationPromptText(e.target.value)}
                  rows={3}
                  placeholder={t("hallucination_prompt_placeholder")}
                />
              </div>
              <div className="row">
                <label>{t("hallucination_wrong_answer")}</label>
                <textarea
                  value={hallucinationWrongAnswerText}
                  onChange={(e) => setHallucinationWrongAnswerText(e.target.value)}
                  rows={3}
                  placeholder={t("hallucination_wrong_answer_placeholder")}
                />
              </div>
              <div className="row">
                <label>{t("hallucination_correct_answer")}</label>
                <textarea
                  value={hallucinationCorrectAnswerText}
                  onChange={(e) => setHallucinationCorrectAnswerText(e.target.value)}
                  rows={3}
                  placeholder={t("hallucination_correct_answer_placeholder")}
                />
              </div>
              <div className="row">
                <label>{t("hallucination_source")}</label>
                <input
                  value={hallucinationSourceText}
                  onChange={(e) => setHallucinationSourceText(e.target.value)}
                  placeholder={t("hallucination_source_placeholder")}
                />
              </div>
              <div className="row">
                <label>{t("hallucination_tags")}</label>
                <input
                  value={hallucinationTagsText}
                  onChange={(e) => setHallucinationTagsText(e.target.value)}
                  placeholder={t("hallucination_tags_placeholder")}
                />
              </div>
              <div className="row">
                <label>{t("hallucination_severity")}</label>
                <select value={hallucinationSeverity} onChange={(e) => setHallucinationSeverity(e.target.value)}>
                  {HALLUCINATION_SEVERITY_OPTIONS.map((item) => (
                    <option key={`hall-severity-${item}`} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              </div>
              <div className="row-actions">
                <button disabled={busy} onClick={onReportHallucination}>
                  {t("hallucination_report_action")}
                </button>
              </div>
              <h3>{t("hallucination_report_result")}</h3>
              <pre>{stringifySafe(hallucinationReportResult)}</pre>
            </div>

            <div>
              <h2>{t("hallucination_check_title")}</h2>
              <div className="row">
                <label>{t("hallucination_prompt")}</label>
                <textarea
                  value={hallucinationPromptText}
                  onChange={(e) => setHallucinationPromptText(e.target.value)}
                  rows={3}
                  placeholder={t("hallucination_prompt_placeholder")}
                />
              </div>
              <div className="row">
                <label>{t("hallucination_llm_answer_hint")}</label>
                <textarea
                  value={hallucinationLlmAnswerText}
                  onChange={(e) => setHallucinationLlmAnswerText(e.target.value)}
                  rows={3}
                  placeholder={t("hallucination_llm_answer_hint_placeholder")}
                />
              </div>
              <div className="row-actions">
                <button disabled={busy} onClick={onCheckHallucinationHunter}>
                  {t("hallucination_check_action")}
                </button>
              </div>
              <h3>{t("hallucination_check_result")}</h3>
              <pre>{stringifySafe(hallucinationCheckResult)}</pre>
            </div>
          </section>
        )}
      </>
    );
  }

  function renderBuilderPage() {
    return (
      <>
        <section className="card grid-2">
          <div>
            <h2>{t("create_node")}</h2>
            <div className="row">
              <label>{t("node_type")}</label>
              <select value={nodeType} onChange={(e) => setNodeType(e.target.value)}>
                {nodeTypes.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </div>

            {nodeType === "human" && (
              <>
                <div className="row">
                  <label>{t("first_name")}</label>
                  <input value={humanFirstName} onChange={(e) => setHumanFirstName(e.target.value)} />
                </div>
                <div className="row">
                  <label>{t("last_name")}</label>
                  <input value={humanLastName} onChange={(e) => setHumanLastName(e.target.value)} />
                </div>
                <div className="row">
                  <label>{t("bio")}</label>
                  <textarea value={humanBio} onChange={(e) => setHumanBio(e.target.value)} rows={3} />
                </div>
                <div className="row">
                  <label>{t("profile_text")}</label>
                  <textarea
                    value={humanProfileText}
                    onChange={(e) => setHumanProfileText(e.target.value)}
                    rows={5}
                    placeholder={t("profile_placeholder")}
                  />
                </div>
                <div className="row">
                  <label>{t("employment_text")}</label>
                  <textarea
                    value={humanEmploymentText}
                    onChange={(e) => setHumanEmploymentText(e.target.value)}
                    rows={3}
                    placeholder={t("employment_placeholder")}
                  />
                </div>
                <div className="row">
                  <label>{t("employment_json")}</label>
                  <textarea
                    value={humanEmploymentJsonText}
                    onChange={(e) => setHumanEmploymentJsonText(e.target.value)}
                    rows={3}
                    placeholder={t("employment_json_placeholder")}
                  />
                </div>
              </>
            )}

            {nodeType === "company" && (
              <>
                <div className="row">
                  <label>{t("company_name")}</label>
                  <input value={companyName} onChange={(e) => setCompanyName(e.target.value)} />
                </div>
                <div className="row">
                  <label>{t("company_industry")}</label>
                  <input value={companyIndustry} onChange={(e) => setCompanyIndustry(e.target.value)} />
                </div>
                <div className="row">
                  <label>{t("company_description")}</label>
                  <textarea value={companyDescription} onChange={(e) => setCompanyDescription(e.target.value)} rows={3} />
                </div>
              </>
            )}

            <div className="row">
              <label>{t("attributes_json")}</label>
              <textarea value={nodeAttributesText} onChange={(e) => setNodeAttributesText(e.target.value)} rows={4} />
            </div>
            <div className="row">
              <label>{t("state_json")}</label>
              <textarea value={nodeStateText} onChange={(e) => setNodeStateText(e.target.value)} rows={3} />
            </div>
            <div className="row-actions">
              <button disabled={busy} onClick={onCreateNode}>
                {t("create_node_btn")}
              </button>
            </div>
          </div>

          <div>
            <h2>{t("create_edge")}</h2>
            <div className="row">
              <label>{t("from_node")}</label>
              <input value={edgeFrom} onChange={(e) => setEdgeFrom(e.target.value)} placeholder="1" />
            </div>
            <div className="row">
              <label>{t("to_node")}</label>
              <input value={edgeTo} onChange={(e) => setEdgeTo(e.target.value)} placeholder="2" />
            </div>
            <div className="row">
              <label>{t("relation")}</label>
              <select value={edgeRelation} onChange={(e) => setEdgeRelation(e.target.value)}>
                {RELATION_OPTIONS.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </div>
            <div className="row">
              <label>{t("weight")}</label>
              <input value={edgeWeight} onChange={(e) => setEdgeWeight(e.target.value)} />
            </div>
            <div className="row">
              <label>{t("direction")}</label>
              <select value={edgeDirection} onChange={(e) => setEdgeDirection(e.target.value)}>
                <option value="directed">{t("directed")}</option>
                <option value="undirected">{t("undirected")}</option>
              </select>
            </div>
            <div className="row">
              <label>{t("logic_rule")}</label>
              <input value={edgeLogicRule} onChange={(e) => setEdgeLogicRule(e.target.value)} />
            </div>
            <div className="row-actions">
              <button disabled={busy} onClick={onCreateEdge}>
                {t("create_edge_btn")}
              </button>
            </div>

            <h3>{t("persistence")}</h3>
            <div className="row-actions">
              <button disabled={busy} onClick={onPersist}>
                {t("persist_snapshot")}
              </button>
              <button disabled={busy} onClick={onLoad}>
                {t("load_snapshot")}
              </button>
            </div>
          </div>
        </section>

        <section className="card grid-2">
          <div>
            <h2>{t("llm_profile_builder")}</h2>
            <div className="row">
              <label>{t("entity_type_hint")}</label>
              <select value={profileEntityHint} onChange={(e) => setProfileEntityHint(e.target.value)}>
                <option value="human">{t("entity_type_human")}</option>
                <option value="company">{t("entity_type_company")}</option>
                <option value="technology">{t("entity_type_technology")}</option>
                <option value="generic">{t("entity_type_generic")}</option>
              </select>
            </div>
            <div className="row">
              <label>{t("profile_input_text")}</label>
              <textarea
                value={profileInputText}
                onChange={(e) => setProfileInputText(e.target.value)}
                rows={10}
                placeholder={t("profile_placeholder")}
              />
            </div>
            <div className="row-actions">
              <button disabled={busy} onClick={onInferProfileGraph}>
                {t("extract_profile_graph")}
              </button>
            </div>
          </div>

          <div>
            <h2>{t("profile_prompt_preview")}</h2>
            <pre>{stringifySafe(profilePromptPreview)}</pre>
            <h3>{t("profile_result")}</h3>
            <pre>{stringifySafe(profileResult)}</pre>
          </div>
        </section>
      </>
    );
  }

  function renderSimulationPage() {
    const timelineStatusText =
      simulationTimeline.status === "completed"
        ? t("timeline_completed")
        : simulationTimeline.status === "running"
          ? t("timeline_in_progress")
          : t("timeline_idle");

    return (
      <>
        <section className="card">
          <div className="timeline-header">
            <h2>{t("simulation_timeline")}</h2>
            <div className={`timeline-status timeline-status-${simulationTimeline.status}`}>
              {timelineStatusText}
            </div>
          </div>
          <div className="timeline-progress-row">
            <span>{t("timeline_progress")}</span>
            <span>{simulationTimeline.progress}%</span>
          </div>
          <div className="timeline-progress-track">
            <div className="timeline-progress-fill" style={{ width: `${simulationTimeline.progress}%` }} />
          </div>
          {!!simulationTimeline.steps.length && (
            <div className="timeline-steps">
              {simulationTimeline.steps.map((step) => (
                <div key={`${step.id}:${step.label}`} className={`timeline-step timeline-step-${step.state}`}>
                  <div className="timeline-dot" />
                  <div className="timeline-step-content">
                    <span className="timeline-label">{step.label}</span>
                    <span className="timeline-time">{step.timeLabel}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="card grid-2">
          <div>
            <h2>{t("simulation")}</h2>
            <div className="row">
              <label>{t("seed_ids")}</label>
              <input value={simSeedIds} onChange={(e) => setSimSeedIds(e.target.value)} placeholder="1,2" />
            </div>
            <div className="row">
              <label>{t("depth")}</label>
              <input value={simDepth} onChange={(e) => setSimDepth(e.target.value)} />
            </div>
            <div className="row">
              <label>{t("steps")}</label>
              <input value={simSteps} onChange={(e) => setSimSteps(e.target.value)} />
            </div>
            <div className="row">
              <label>{t("damping")}</label>
              <input value={simDamping} onChange={(e) => setSimDamping(e.target.value)} />
            </div>
            <div className="row">
              <label>{t("activation")}</label>
              <select value={simActivation} onChange={(e) => setSimActivation(e.target.value)}>
                {ACTIVATION_OPTIONS.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </div>
            <div className="row">
              <label>{t("infer_rounds")}</label>
              <input value={simInferRounds} onChange={(e) => setSimInferRounds(e.target.value)} />
            </div>
            <div className="row-actions">
              <button disabled={busy} onClick={onSimulate}>
                {t("run_simulation")}
              </button>
            </div>
          </div>

          <div>
            <h2>{t("last_simulation_output")}</h2>
            <pre>{stringifySafe(lastSimulation)}</pre>
          </div>
        </section>

        <section className="card grid-2">
          <div>
            <h2>{t("event_feedback")}</h2>
            <div className="row">
              <label>{t("event_id")}</label>
              <input value={rewardEventId} onChange={(e) => setRewardEventId(e.target.value)} />
            </div>
            <div className="row">
              <label>{t("reward")}</label>
              <input value={rewardValue} onChange={(e) => setRewardValue(e.target.value)} />
            </div>
            <div className="row">
              <label>{t("learning_rate")}</label>
              <input value={rewardLr} onChange={(e) => setRewardLr(e.target.value)} />
            </div>
            <div className="row-actions">
              <button disabled={busy} onClick={onRewardEvent}>
                {t("apply_reward")}
              </button>
            </div>

            <h3>{t("batch_relation_reinforcement")}</h3>
            <div className="row">
              <label>{t("relation")}</label>
              <select value={reinforceRelationType} onChange={(e) => setReinforceRelationType(e.target.value)}>
                {RELATION_OPTIONS.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </div>
            <div className="row">
              <label>{t("reward")}</label>
              <input value={reinforceReward} onChange={(e) => setReinforceReward(e.target.value)} />
            </div>
            <div className="row">
              <label>{t("learning_rate")}</label>
              <input value={reinforceLr} onChange={(e) => setReinforceLr(e.target.value)} />
            </div>
            <div className="row-actions">
              <button disabled={busy} onClick={onReinforceRelation}>
                {t("reinforce_relation")}
              </button>
            </div>
          </div>

          <div>
            <h2>{t("event_stream")}</h2>
            {renderPager({
              page: eventsView.page,
              totalPages: eventsView.totalPages,
              setPage: setEventsPage,
              label: t("pager_events"),
            })}
            <pre>{stringifySafe(eventsView.items)}</pre>
          </div>
        </section>
      </>
    );
  }

  function renderDataPage() {
    return (
      <>
        <section className="card grid-2">
          <div>
            <h2>{t("snapshot_nodes")}</h2>
            {renderPager({
              page: nodesView.page,
              totalPages: nodesView.totalPages,
              setPage: setNodesPage,
              label: t("pager_nodes"),
            })}
            <pre>{stringifySafe(nodesView.items)}</pre>
          </div>
          <div>
            <h2>{t("snapshot_edges")}</h2>
            {renderPager({
              page: edgesView.page,
              totalPages: edgesView.totalPages,
              setPage: setEdgesPage,
              label: t("pager_edges"),
            })}
            <pre>{stringifySafe(edgesView.items)}</pre>
          </div>
        </section>

        <section className="card">
          <h2>{t("sql_table_schema_json")}</h2>
          <pre>{stringifySafe(dbSchema)}</pre>
        </section>

        <section className="card">
          <h2>{t("project_modules")}</h2>
          {renderPager({
            page: modulesView.page,
            totalPages: modulesView.totalPages,
            setPage: setModulesPage,
            label: t("pager_modules"),
          })}
          <div className="modules-grid">
            {modulesView.items.map((mod) => (
              <article key={mod.name} className="module-card">
                <h3>{mod.name}</h3>
                <p>{mod.description}</p>
                <p>
                  {t("files")}: {mod.count}
                </p>
                <details>
                  <summary>{t("show_files")}</summary>
                  <pre>{(mod.files || []).join("\n")}</pre>
                </details>
              </article>
            ))}
          </div>
        </section>
      </>
    );
  }

  if (fatalError) {
    return (
      <div className="app-shell">
        <section className="card">
          <h2>{t("runtime_error_title")}</h2>
          <p>{t("runtime_error_hint")}</p>
          <pre>{fatalError}</pre>
          <div className="row-actions">
            <button onClick={() => setFatalError("")}>{t("action_try_continue")}</button>
            <button onClick={refreshAll}>{t("action_reload_data")}</button>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="top-header">
        <div>
          <h1>{t("app_title")}</h1>
          <p>{t("app_subtitle")}</p>
        </div>
        <div className="header-right">
          <div className="language-switcher">
            <label htmlFor="ui-language">{t("ui_language")}</label>
            <select id="ui-language" value={uiLanguage} onChange={(e) => setUiLanguage(e.target.value)}>
              {UI_LANG_OPTIONS.map((item) => (
                <option key={item.code} value={item.code}>
                  {item.label}
                </option>
              ))}
            </select>
          </div>
          <div className="row-actions">
            <button disabled={busy} onClick={refreshAll}>
              {t("action_refresh")}
            </button>
            <button disabled={busy} onClick={onSeedDemo}>
              {t("action_seed_demo")}
            </button>
            <button disabled={busy} onClick={onClear}>
              {t("action_clear")}
            </button>
          </div>
        </div>
      </header>

      <section className="card page-nav-card">
        <div className="page-nav">
          {PAGE_KEYS.map((pageKey) => (
            <button
              key={pageKey}
              type="button"
              className={`page-tab ${currentPage === pageKey ? "active" : ""}`}
              onClick={() => goToPage(pageKey)}
            >
              {t(`page_${pageKey}`)}
            </button>
          ))}
        </div>
      </section>

      {currentPage === "overview" && renderOverviewPage()}
      {currentPage === "builder" && renderBuilderPage()}
      {currentPage === "simulation" && renderSimulationPage()}
      {currentPage === "data" && renderDataPage()}
    </div>
  );
}
