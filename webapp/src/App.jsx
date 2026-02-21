import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  clearGraph,
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
  rewardEvent,
  runProjectLLMDebate,
  runProjectDailyMode,
  simulateGraph,
  subscribeGraphEvents,
  updateEdge,
  updateNode,
  updateProjectUserGraph,
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
const PAGE_KEYS = ["overview", "builder", "simulation", "data"];
const OVERVIEW_SECTION_KEYS = [
  "demo",
  "daily",
  "user_graph",
  "autoruns",
  "graph",
  "client",
  "advisors",
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

const PERSONALIZATION_STYLE_OPTIONS = ["adaptive", "concise", "balanced", "deep"];
const PERSONALIZATION_DEPTH_OPTIONS = ["quick", "balanced", "deep"];
const PERSONALIZATION_RISK_OPTIONS = ["low", "medium", "high"];
const PERSONALIZATION_TONE_OPTIONS = ["neutral", "direct", "empathetic", "challenging"];

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
  },
  zh: {
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
  },
  es: {
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
  },
  pt: {
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
  },
  fr: {
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
  },
  ar: {
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
  },
  hi: {
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
  },
  ja: {
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

const OVERVIEW_SECTION_TRANSLATION_KEYS = {
  demo: "overview_section_demo",
  daily: "overview_section_daily",
  user_graph: "overview_section_user_graph",
  autoruns: "overview_section_autoruns",
  graph: "overview_section_graph",
  client: "overview_section_client",
  advisors: "overview_section_advisors",
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

function colorForNodeType(type) {
  const key = String(type || "generic").toLowerCase();
  if (key === "human" || key === "client_session") return "#1f6feb";
  if (key === "company" || key === "domain") return "#1f8a5f";
  if (key === "concept") return "#935f1f";
  if (key === "ip_address" || key === "network_profile") return "#7f4f97";
  if (key === "browser" || key === "operating_system") return "#495057";
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
  const [personalizationDraft, setPersonalizationDraft] = useState(() => loadPersonalizationDraft());
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
  }, [snapshot, selectedNodeId, selectedNode, selectedEdgeSig, selectedEdge]);

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

  function patchPersonalizationDraft(patch) {
    setPersonalizationDraft((prev) => normalizePersonalizationDraft({ ...(prev || {}), ...(patch || {}) }));
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

            <section className="card">
              <h2>{t("prompt_catalog")}</h2>
              <pre>{stringifySafe(modelAdvisors?.prompts || [])}</pre>
            </section>
          </>
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
