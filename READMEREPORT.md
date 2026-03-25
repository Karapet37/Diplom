# Persona-Graph Agent System: AI Report

Этот файл больше не является 3-й главой диплома. Теперь это рабочий системный отчёт для AI-агента или нового разработчика, который должен быстро понять:

1. где что лежит в проекте;
2. какие части реально участвуют в активном runtime;
3. как проходит запрос от пользователя до ответа;
4. где хранятся память, persona, graph и файлы;
5. какие зоны являются полезными, а какие вторичными, историческими или шумовыми.

Ниже приоритет отдан не “красивому описанию”, а практической ориентации по проекту.

---

## 1. Что в проекте главное

Текущий активный продукт в этом репозитории — это `persona-graph agent system`, который:

- принимает чат-сообщение;
- детерминированно выделяет сущности и ситуацию;
- выбирает или материализует persona-head;
- строит ограниченный context;
- вызывает локальную LLM через строгий provider path;
- отвечает;
- хранит структурированную память в графе и файловом storage;
- даёт UI для просмотра и редактирования графа.

Ключевая архитектурная идея:

```text
LLM не управляет системой.
LLM только заполняет строгие формы или генерирует ответ.
Все изменения graph/persona делает deterministic code.
```

---

## 2. Структура репозитория

Ниже перечислены важные директории и их роль.

### 2.1 Корневая структура

| Путь | Роль | Статус |
| --- | --- | --- |
| `start.py` | Главный entrypoint локального запуска | Активно используется |
| `agent_system/` | Основная backend-логика persona-graph runtime | Основная активная зона |
| `src/utils/` | Local LLM provider, token budgeting, env utils | Основная активная зона |
| `src/web/` | Склейка backend + frontend routes | Активно используется |
| `webapp/` | React frontend, Graph Workspace, chat UI | Активно используется |
| `config/runtime-profiles/` | Профили запуска: development, local-demo, local-heavy, server | Активно используется |
| `scripts/` | Bootstrap и profile-aware run scripts | Активно используется |
| `tests/agent_system/` | Основные регрессионные тесты текущей системы | Активно используется |
| `memory/` | Runtime storage: graph, heads, sessions, files | Активно используется |
| `docs/` | Документы, схемы, заметки | Вспомогательно |
| `models/` | GGUF-модели и внешние reference artifacts | Частично активно |
| `data/` | Исследовательские/служебные данные, не hot path чата | Вторично |
| `src/living_system/` | Старая/параллельная исследовательская линия | Не в hot path текущего runtime |
| `src/autonomous_graph/` | Отдельный экспериментальный слой | Не в hot path текущего runtime |
| `roaches_viz/` | Отдельный side-project | Не относится к текущему runtime |
| `packages/` | SDK и интеграционные пакеты | Вспомогательно |
| `infra/` | Nginx, Prometheus, Grafana | Инфраструктурно |

### 2.2 Что запускать

Основной запуск сейчас идёт через:

```text
start.py --profile <name> [--env-file ...] [--config ...]
  -> bootstrap_runtime_environment()
  -> get_runtime_config()
  -> src.web.combined_app.create_combined_app()
    -> agent_system.api.create_app()
    -> src.web.api.attach_frontend_routes()
```

То есть локальный runtime — это не просто API, а combined app:

- backend API из `agent_system/api.py`;
- frontend routes из `src/web/api.py`;
- UI из `webapp/dist` или `webapp/index.html`.

Для productionized локального запуска теперь существуют явные профили:

- `development`
- `local-demo`
- `local-heavy`
- `server`

А операторские команды запуска стандартизированы через:

- `python start.py --list-profiles`
- `python start.py --profile development --check`
- `./scripts/run_profile.sh development`
- `./scripts/bootstrap_local.sh`

### 2.3 Активные backend-модули

| Путь | Для чего нужен |
| --- | --- |
| `agent_system/chat_engine.py` | Главная orchestration-функция chat path |
| `agent_system/message_analyzer.py` | Анализ user message, выделение `user_state` и сущностей |
| `agent_system/situation_engine.py` | Преобразует user analysis в structured situation |
| `agent_system/feature_extractor.py` | Детерминированные признаки для classifier |
| `agent_system/classifier_forest.py` | Vote-based классификация entity type |
| `agent_system/head_caller.py` | Решает, нужен ли persona head и какой head главный |
| `agent_system/persona_engine.py` | Persona storage, layered baseline/dynamic/learned state, emotion update, triad, revisions, indicators, explainability |
| `agent_system/context_builder.py` | Собирает bounded context через deterministic stages: collect, score, rank, compress, pack |
| `agent_system/graph_store.py` | Global graph storage, merge, hygiene, editing |
| `agent_system/duplicate_resolver.py` | Duplicate resolution и semantic normalization |
| `agent_system/entity_extractor.py` | Structured extraction из текста в graph proposals |
| `agent_system/file_ingestion.py` | File learning pipeline и rebuild |
| `agent_system/node_rethinker.py` | Ограниченный rethink mode для узлов graph |
| `agent_system/graph_localizer.py` | Canonical English + localized node explanations |
| `agent_system/language_tools.py` | Language detection/normalization |
| `agent_system/llm.py` | Узкий адаптер от runtime к local LLM provider |
| `agent_system/memory_layers.py` | Layered memory policies, archive paths, cold-storage lifecycle rules |
| `agent_system/observability.py` | Request tracing, stage timings, counters, debug diagnostics |
| `agent_system/reliability.py` | Failure policies, degraded runtime modes, safe mutation rollback and recovery signaling |
| `agent_system/prompt_builder.py` | Жёсткие prompt-формы для chat, extraction, rethink, persona synthesis |
| `agent_system/runtime_config.py` | Единая runtime-конфигурация: paths, budgets, model roles, feature flags |
| `agent_system/history_store.py` | Session history и recent dialogue |
| `agent_system/api.py` | FastAPI endpoints |
| `agent_system/models.py` | Typed contracts и dataclasses для core runtime |

### 2.4 Активные frontend-модули

| Путь | Для чего нужен |
| --- | --- |
| `webapp/src/App.jsx` | Главная orchestration-логика UI |
| `webapp/src/api.js` | Вызовы backend endpoints |
| `webapp/src/components/Operator/ChatSurface.jsx` | Operator surface для chat, persona selection inspection, context preview и request trace inspection |
| `webapp/src/components/Operator/GraphOperatorSurface.jsx` | Operator surface для graph navigation, node health, hygiene actions и rethink preview/apply |
| `webapp/src/components/Operator/PersonaInspectionSurface.jsx` | Operator surface для layered persona inspection, revisions и maturity/confidence indicators |
| `webapp/src/components/Operator/FilesIngestionSurface.jsx` | Operator surface для file upload, ingestion inspection и last-run visibility |
| `webapp/src/components/Operator/DiagnosticsSurface.jsx` | Operator surface для runtime metrics, graph health и trace inspection |
| `webapp/src/components/Graph/GraphWorkspace.jsx` | Основная рабочая зона graph |
| `webapp/src/components/Chat/ChatGraphPanel.jsx` | Chat + graph panel |
| `webapp/src/lib/operatorFormatters.js` | Безопасное форматирование node/persona/context previews для operator UI |
| `webapp/src/lib/graphView.js` | Layout, neighbourhood extraction, path logic |
| `webapp/src/lib/i18n.js` | UI localization |
| `webapp/src/styles.css` | Основные стили |

### 2.5 Runtime storage

Текущая активная память системы лежит в `memory/`.

#### `memory/graphs/`

- `nodes.json` — глобальные узлы графа;
- `edges.json` — глобальные связи графа.

#### `memory/heads/{head_slug}/`

Для каждой persona head:

- `traits.json` — traits и `entity_type`;
- `relations.json` — aliases и relations;
- `examples.json` — примеры и `situation_reactions`;
- `knowledge.txt` — краткое текстовое знание;
- `emotion_vector.json` — текущее состояние эмоций;
- `baseline.json` — baseline definition persona;
- `dynamic_state.json` — dynamic emotional state и последний style/situation;
- `learned_patterns.json` — bounded learned interaction patterns;
- `log_tuples.json` — сжатые поведенческие сигнатуры без дублей, с `frequency`;
- `persona_form.json` — подробная анкета persona;
- `decision_explanation.txt` — простое объяснение decision-patterns;
- `revisions.json` — revision counters и bounded snapshots изменённых слоёв;
- `meta.json` — служебные метаданные;
- `local_graph.json` — локальный persona subgraph.

#### `memory/sessions/`

- текстовые session histories.

#### `memory/files/uploaded_documents/`

- загруженные файлы по session id.

#### `memory/proposals/`

- persona proposals для materialization.

#### `memory/archive/`

- cold session archives;
- persona overflow archives;
- graph snapshots;
- persona snapshots before risky persona mutations;
- lifecycle and recovery artifacts that are intentionally excluded from hot-path context reads.

### 2.6 Что в репозитории вторично или шумно

AI-агенту не стоит тратить основной фокус на эти зоны, если задача не относится к ним напрямую:

- `src/living_system/` — отдельная исследовательская линия, не в активном chat-path;
- `src/autonomous_graph/` — отдельный экспериментальный контур;
- `roaches_viz/` — другой проект;
- `memory/personalities/` — исторический storage, не основной для текущего `agent_system`;
- `models/PersonaAgentwGraphRAG-DE6F/` — reference-материалы и внешняя база сравнения, не hot path;
- `node_modules/` на корне — шум для основного Python runtime;
- `data/` — исследовательские и служебные данные, не основной current runtime path.

Если задача касается именно текущей системы, основной фокус должен быть на:

```text
agent_system/
src/utils/
src/web/
webapp/src/
memory/
tests/agent_system/
start.py
```

---

## 3. Как система работает

### 3.1 Общий pipeline чата

Текущий pipeline:

```text
chat request
  -> ChatTurnRequest
  -> message_analyzer
  -> situation_engine
  -> feature_extractor
  -> classifier_forest
  -> head_caller
  -> explicit persona emotion update
  -> persona_engine
  -> context_builder
  -> llm.generate_chat_reply
  -> explicit storage writes
  -> ChatTurnResult
  -> response
  -> optional background rebuild
```

`generate_response()` остаётся совместимым внешним API и по-прежнему возвращает словарь, но внутренняя hot-path orchestration теперь строится вокруг typed contracts:

- `ChatTurnRequest` — вход в chat runtime;
- `UserState` — нормализованное состояние пользователя;
- `Situation` — структурированная интерпретация ситуации;
- `ChatSideEffects` — явное описание write-side effects;
- `ChatTurnResult` — итог runtime-прохода до сериализации в API response.

`context_builder.py` внутри этого lifecycle теперь выполняет не неявную склейку секций, а явный pipeline:

```text
collect candidates
  -> score candidates
  -> rank candidates
  -> compress candidates
  -> pack bounded final context
```

### 3.2 Что делает каждый модуль в chat-path

#### `agent_system/chat_engine.py`

Это главный orchestration-слой.

Он делает:

1. создаёт или грузит session;
2. запускает deterministic concept graph extraction по самому сообщению;
3. получает `analysis` из analyzer;
4. классифицирует найденные сущности;
5. подготавливает heads;
6. выбирает primary persona;
7. обновляет её emotion vector;
8. строит context;
9. вызывает LLM;
10. сохраняет turn в history;
11. записывает situation-reaction;
12. решает, нужен ли background rebuild.

Теперь эти side effects выделены явно, а не “растворены” в одной длинной процедуре. На chat-path отдельно видны:

- graph prewrite;
- history write;
- persona emotion update;
- persona reaction memory write;
- rebuild scheduling decision.
- degraded fallback decision, если local chat provider недоступен или вернул пустой unusable output.

Ключевой момент:

```text
background rebuild не запускается безусловно после каждого сообщения.
Он теперь отложен и периодичен, чтобы не душить latency.
```

#### `agent_system/message_analyzer.py`

Analyzer не решает, как persona должна чувствовать себя.

Он возвращает типизированный `UserState`:

```python
UserState(
    tone=...,
    intent=...,
    signals=...,
    language=...,
)
```

и также список сущностей.

#### `agent_system/situation_engine.py`

`situation_engine` переводит user analysis в типизированный `Situation`:

```python
Situation(
    type=...,
    target=...,
    severity=...,
)
```

Это принципиально важно, потому что:

```text
persona реагирует не на raw user emotion,
а на интерпретированную ситуацию.
```

#### `agent_system/feature_extractor.py` + `classifier_forest.py`

Здесь идут deterministic features и vote-based classification.

Classifier:

- классифицирует entity types;
- помогает в routing;
- не управляет эмоциями persona.

#### `agent_system/head_caller.py`

Решает:

- оставлять entity как graph node;
- materialize как persona head;
- какой head считать primary в текущем turn.

#### `agent_system/persona_engine.py`

Это один из центральных модулей.

Он отвечает за:

- загрузку и сохранение heads;
- baseline definition;
- dynamic emotional state;
- learned interaction patterns;
- materialization;
- emotion evolution;
- reaction policy;
- triad persona storage;
- revision metadata и bounded snapshots;
- confidence/maturity indicators;
- persona selection and response explainability;
- formal persona model;
- persona graph sync.

### 3.3 Формальная модель persona

В системе persona понимается как stateful system:

```text
Persona = (T, E, R, M)
```

где:

- `T` — traits и static parameters;
- `E` — emotion vector;
- `R` — deterministic reaction policy;
- `M` — memory: examples, relations, situation-reactions, graph links, triad.

Формальная реакция:

```text
R: (situation, T, E) -> (ΔE, response_style)
```

Emotion update:

```text
E(t+1) = clamp( D(E(t)) + R(situation, T) )
```

То есть:

- есть drift к baseline;
- есть boundedness `[0,1]`;
- есть trait-conditioned reaction;
- нет прямого наследования user emotion.

Жёсткое правило:

```text
persona_emotion = f(persona_traits, situation)
NOT f(user_emotion)
```

### 3.4 Persona triad

В текущем состоянии persona хранит не только traits и knowledge, но и triad:

#### 1. `log_tuples`

Сжатые logs как tuple signatures без дублей.

Повторы не дублируются строками, а увеличивают `frequency`.

Пример логики:

```text
("utterance_pattern", "answer directly") -> frequency = 3
```

или

```text
("situation_reaction", "type=insult;target=persona", "firm boundary") -> frequency = 2
```

#### 2. `persona_form`

Подробная анкета личности.

Типичные поля:

- `identity_class`
- `interaction_style`
- `core_dispositions`
- `decision_patterns`
- `clarification_policy`
- `sarcasm_profile`
- `response_priorities`
- `knowledge_domains`
- `risk_controls`

#### 3. `decision_explanation`

Короткое простое объяснение для человека:

```text
как эта persona принимает решения,
в каком порядке что проверяет,
когда отвечает по сути,
когда уточняет,
когда допускает сарказм.
```

Это нужно не только для объяснимости, но и для более стабильного prompt grounding.

### 3.5 Как persona triad используется в ответе

Поверх triad persona head теперь имеет явное layered-state разделение:

- `baseline_definition`
  - устойчивые traits, relations, aliases, knowledge;
- `dynamic_state`
  - текущий `emotion_vector`, `last_situation`, `last_response_style`;
- `learned_patterns`
  - examples, `situation_reactions`, `log_tuples`, `persona_form`, `decision_explanation`, `learned_traits`.

Это разделение нужно для того, чтобы random interaction events не превращались в uncontrolled persona drift. Baseline должен оставаться устойчивым, dynamic state должен быстро обновляться на hot path, а learned patterns — адаптироваться ограниченно и reviewable way.

У persona head также есть versioned operational meta:

- `revision`
- `baseline_revision`
- `dynamic_revision`
- `learned_revision`
- `confidence_score`
- `maturity_score`
- `maturity_level`
- `adaptation_locked`

Ключевое bounded-adaptation правило такое:

```text
случайные chat-события могут обновлять learned patterns,
но не должны тихо переписывать baseline traits/knowledge
```

Поэтому `update_persona_from_examples()` теперь прежде всего обновляет learned layer, а не baseline. Если синтез дал новые рискованные trait-сигналы, они попадают в `learned_traits` и revision trail, а не незаметно заменяют baseline definition.

`context_builder.py` сначала раздельно собирает candidates из:

- short-term session history;
- persona memory;
- persona triad;
- global graph facts;
- local graph neighborhood;
- file-ingested knowledge.

После этого каждый candidate получает explainable score по факторам:

- relevance;
- recency;
- importance;
- confidence;
- persona alignment;
- graph connectivity.

Затем выполняются deterministic ranking, compression и bounded packing. Это важно, потому что теперь выбор элементов `persona_block`, `graph_context` и `recent_dialogue` можно разбирать по `context_debug`, а не только смотреть на итоговый prompt.

В `persona_block` попадает:

- identity class;
- sarcasm profile;
- clarification policy;
- decision patterns;
- response priorities;
- короткое decision explanation;
- часть log tuples;
- emotion vector;
- maturity/confidence summary;
- revision summary;
- relevant reactions.

То есть LLM получает не абстрактную “личность в вакууме”, а уже структурированную operational persona.

---

## 4. Как работает graph memory

### 4.1 Где главный graph

Глобальный graph лежит в:

- `memory/graphs/nodes.json`
- `memory/graphs/edges.json`

Узлы содержат:

- `id`
- `name`
- `type`
- `aliases`
- `description`
- `facts`
- `translation_line`
- `importance`
- `confidence`
- `frequency`
- `context`

### 4.2 Что делает `graph_store.py`

`agent_system/graph_store.py` отвечает за:

- загрузку и сохранение graph state;
- merge extraction results;
- lifecycle classification узлов;
- node/edge editing;
- validation;
- graph hygiene;
- quarantine/review flow для низкоконфидентных и review-range узлов;
- archival и merged event logging;
- deterministic cluster labeling для больших graph states;
- manual merge/delete/connect operations;
- subgraph retrieval;
- node views.

### 4.3 Graph hygiene

Graph hygiene — это не набор случайных хаков, а controlled optimization process.

Формально:

```text
Quality(G) = α * relevance - β * redundancy + γ * connectivity
```

Где:

- `relevance` — насколько graph полезен для retrieval;
- `redundancy` — сколько там дублей и low-value мусора;
- `connectivity` — насколько хорошо сохранены meaningful relations.

В активной реализации это выражается через:

- decay;
- duplicate merge;
- duplicate review marking;
- garbage collection;
- compression;
- semantic repair.

Также graph layer теперь использует явные lifecycle states:

- `active`
- `weak`
- `suspect`
- `archived`
- `merged`

Из них только `active` и `weak` попадают в hot-path retrieval. `suspect` остаётся видимым для оператора в полном graph view, но исключается из `search_nodes()` и `subgraph()`. `archived` и `merged` фиксируются в lifecycle archive log и не возвращаются как обычные active nodes.

### 4.4 Что именно считалось мусором на практике

Во время реальной работы уже были замечены следующие плохие артефакты:

- sentence-fragment nodes;
- дубли вроде `human` и `люди` в роли разных узлов;
- abstract concepts, ошибочно типизированные как `PERSON`;
- пустые placeholder descriptions;
- reinterpretation drift после слишком свободного rethink mode;
- избыточные summary nodes на маленьких графах.
- low-confidence extractions, которые не должны сразу становиться полноценным knowledge слоем;
- review-range duplicate pairs, которые требуют ручной проверки, а не мгновенного auto-merge.

Эти проблемы не считаются “мелкими UI дефектами”. Для системы они критичны, потому что ухудшают:

- retrieval quality;
- persona grounding;
- subgraph usefulness;
- explainability.

Именно поэтому:

- `duplicate_resolver.py`
- `graph_store.py`
- `node_rethinker.py`
- `memory_layers.py`

являются таким же ядром системы, как и chat path.

### 4.5 Graph lifecycle и review flow

Graph layer больше не считается просто “append-only knowledge map”. Для узлов действуют explicit lifecycle rules.

- low-confidence extraction -> `suspect` + `review_status=quarantine`
- review-range duplicate -> более слабый node маркируется как `suspect` с `review_reason=duplicate_candidate`
- aged low-value node -> выводится из active graph и записывается как `archived`
- deterministic merge -> secondary node фиксируется как `merged`

Это позволяет не пропускать шум в context, но при этом не терять audit trail и оставлять manual review возможным.

---

## 4.6 Layered memory lifecycle

Текущая память больше не рассматривается как один плоский storage layer. Она разделена на:

- working memory;
- session memory;
- persona memory;
- graph knowledge memory;
- archive / cold memory.

Ключевое правило такое:

```text
archive memory не читается напрямую в обычный chat context.
В prompt могут попадать только bounded active layers.
```

Практически это означает следующее.

- `working memory` — это ephemeral turn-local state в typed runtime objects; он не является долгосрочным knowledge store.
- `session memory` хранится в `memory/sessions/` как active hot tail, а старые turns уходят в `memory/archive/sessions/`.
- `persona memory` хранится в `memory/heads/{head}/`, но overflow по examples, reactions, log tuples и oversized knowledge уходит в `memory/archive/heads/`.
- `graph knowledge memory` остаётся активным source of truth в `memory/graphs/`, а cold snapshots попадают в `memory/archive/graphs/` только по явной maintenance-команде.
- `archive / cold memory` существует для auditability, migration safety и offline maintenance, а не для прямого prompt grounding.

Отдельная техническая записка по этим правилам находится в `docs/memory_lifecycle.md`.

---

## 5. Как документ learning добавляет знания

### 5.1 Горячий путь

`agent_system/file_ingestion.py` делает:

```text
file
  -> text conversion
  -> chunking
  -> extract_knowledge()
  -> validate_extraction()
  -> graph_store.merge_extraction()
  -> optional head updates
```

### 5.2 Роль `entity_extractor.py`

`entity_extractor.py` объединяет:

- deterministic concept extraction;
- LLM structured extraction;
- validation;
- fallback heuristics.

Важно:

```text
LLM здесь не пишет graph напрямую.
Он только предлагает structured content.
```

Все `entities` и `relations` проходят validation:

- отбрасываются sentence fragments;
- нормализуются relation fields;
- повторно определяется тип сущности;
- мусор не materialize-ится как persona.

### 5.3 Какая модель используется

Structured extraction и persona synthesis сейчас идут через fast role path, а не через тяжёлый “общий” режим по умолчанию.

Это нужно для двух причин:

1. уменьшить latency;
2. уменьшить нестабильность JSON output.

---

## 6. Как работает rethink mode

`agent_system/node_rethinker.py` — это ограниченный слой переосмысления нод.

Критический принцип:

```text
LLM не возвращает готовые graph mutations как приказы.
LLM возвращает только content improvements и link suggestions.
```

Дальше deterministic code делает:

- валидацию имён;
- отсев fragment-like названий;
- нормализацию role;
- маппинг `role -> relation_type`;
- типизацию узлов;
- `patch_node`;
- `create_node` / `upsert`;
- `connect_nodes`.

### 6.1 Режимы rethink

Есть два режима:

- `preview` — показывает план без записи в graph;
- `apply` — применяет только разрешённые изменения.

### 6.2 Почему это важно

Старый более свободный rethink pipeline уже успел создать semantic defects.

Поэтому текущий rethink mode deliberately constrained:

- content-only suggestions;
- no raw graph control;
- whitelist link roles;
- preview first;
- apply second.

---

## 7. Graph Workspace и UI

Frontend в `webapp/` — это не декоративный слой, а реальный операторский интерфейс системы.

### 7.1 Что умеет Graph Workspace

Через `webapp/src/components/Graph/GraphWorkspace.jsx` и соседние модули доступны:

- просмотр графа;
- zoom/pan;
- локализованный скролл;
- branch windows;
- drill-down по узлам двойным кликом;
- manual graph editing;
- node rethink preview/apply;
- node cards с canonical English explanation;
- localized explanation block;
- translation line для узлов.
- lifecycle state labels для `active / weak / suspect`;
- graph diagnostics summary;
- cluster labels для больших graph views.

Но operator-grade UI теперь организован не как одна смешанная страница, а как набор отдельных рабочих поверхностей:

- `chat`
  - conversation flow, persona selection inspection, response shaping, safe context preview, request trace inspection;
- `graph workspace`
  - graph navigation, node health/state inspection, merge/quarantine/delete/connect actions, rethink preview/apply;
- `persona inspection`
  - layered persona state, baseline/dynamic/learned split, revisions, maturity/confidence indicators;
- `files / ingestion`
  - upload flow, last ingestion result, per-file ingestion inspection;
- `diagnostics`
  - runtime metrics, graph health, recent traces, selected trace details.

### 7.2 Зачем нужен frontend с точки зрения AI-системы

Он важен не только для человека, но и как operational surface:

- позволяет визуально замечать мусор;
- быстрее ловит semantic drift;
- показывает, что retrieval useful, а что нет;
- даёт человеку право финального контроля;
- делает видимыми reasoning inputs без раскрытия hidden chain-of-thought;
- позволяет оператору вручную применять hygiene actions и inspect/debug runtime без прямого доступа к storage-файлам.

### 7.3 Что именно оператор может проверить через UI

Через новый layout оператор может видеть:

- почему была выбрана именно эта persona;
- каким response style shaped текущий ответ;
- какие context items реально попали в bounded prompt;
- какой trace сопровождал запрос и какие stage timings он набрал;
- в каком lifecycle state находится выбранный graph node;
- какая у узла `importance`, `confidence`, `frequency`;
- какие persona layers и revisions влияли на текущее поведение.

Важно, что UI показывает не hidden chain-of-thought, а безопасные структурированные explanations и runtime artifacts, которые система уже считает допустимыми для operator inspection.

---

## 8. Как подключена LLM

### 8.1 Активный путь

Основной local provider:

- `src/utils/local_llm_provider.py`

Тонкий runtime adapter:

- `agent_system/llm.py`

Budgeting и retries:

- `src/utils/prompt_budgeter.py`
- `src/utils/token_budget.py`

### 8.2 Главный принцип доступа к LLM

Система не “является LLM”.
Система подключает LLM как подчинённый вычислительный модуль.

Использование делится на режимы:

- `chat`
- `knowledge`
- `translation`

и на role paths:

- `analyst`
- `general`
- `translator`
- другие role advisors при наличии.

### 8.3 Что было проблемой по скорости

Главные источники лагов раньше были такими:

- слишком тяжёлый chat role;
- слишком раздутый context;
- rebuild после каждого turn;
- structured-output forcing для обычного chat prompt;
- лишние retry rounds.

### 8.4 Что сейчас исправлено

Сейчас runtime уже зажат:

- chat path идёт через fast role по умолчанию;
- context budget урезан до `4000`;
- rebuild не крутится безусловно после каждого turn;
- JSON forcing включается только для действительно структурных prompt-ов;
- retry path укорочен по умолчанию;
- fallback reply возвращается сразу, если grounding недостаточен.

То есть задержка больше не должна появляться из-за того, что обычный chat prompt сначала принудительно гоняется через JSON-only path.

### 8.5 Наблюдаемость и диагностика

Для текущего runtime добавлен отдельный слой наблюдаемости:

- `agent_system/observability.py`
- `agent_system/reliability.py`
- `docs/observability.md`
- `docs/context_pipeline.md`
- `docs/memory_lifecycle.md`
- `docs/graph_lifecycle.md`
- `docs/reliability.md`

Он не меняет продуктовую логику и не превращает систему в “дашборд ради дашборда”. Его задача — сделать измеримыми:

- latency по стадиям;
- fallback rate;
- usage context budget;
- rebuild frequency;
- rethink preview/apply outcomes;
- graph health.

Основные debug endpoints:

- `GET /api/cognitive/debug/metrics`
- `GET /api/cognitive/debug/traces`
- `GET /api/cognitive/debug/graph-health`
- `GET /api/cognitive/debug/runtime-status`
- `POST /api/cognitive/graph/nodes/{node_id}/state`

Именно через них удобно проверять, где висит chat-path, не выросла ли redundancy графа и не начал ли rethink mode деградировать качество graph state.

### 8.6 Надёжность, degraded mode и recovery

Отдельный reliability-слой нужен потому, что для долгоживущей системы важнее предсказуемая деградация, чем попытка “любой ценой продолжать работать”.

Failure-классы живут в `agent_system/reliability.py`:

- `DependencyUnavailableFailure`
- `StorageWriteFailure`
- `MutationRejectedFailure`
- `RecoveryFailure`

Практический смысл такой:

- слабые зависимости переводят runtime в degraded mode;
- рискованные graph/persona мутации либо завершаются валидно, либо откатываются;
- silent corruption считается недопустимой.

Что уже защищено:

- risky graph writes делают pre-mutation snapshot;
- risky persona mutations сохраняют persona snapshot и, при необходимости, graph snapshot;
- `rethink/apply` восстанавливает graph snapshot, если mutation stage падает;
- chat-path при недоступности local provider сразу возвращает safe fallback reply, а не остаётся в подвешенном состоянии.

Operator recovery endpoints:

- `GET /api/cognitive/graph/snapshots`
- `POST /api/cognitive/graph/restore`
- `GET /api/cognitive/personalities/{name}/revisions`
- `GET /api/cognitive/personalities/{name}/snapshots`
- `POST /api/cognitive/personalities/{name}/restore/{revision}`

---

## 9. Какой модуль где участвует

### 9.1 Если приходит обычное чат-сообщение

Участвуют:

- `start.py`
- `src/web/combined_app.py`
- `agent_system/api.py`
- `agent_system/runtime_config.py`
- `agent_system/chat_engine.py`
- `agent_system/message_analyzer.py`
- `agent_system/situation_engine.py`
- `agent_system/feature_extractor.py`
- `agent_system/classifier_forest.py`
- `agent_system/head_caller.py`
- `agent_system/persona_engine.py`
- `agent_system/context_builder.py`
- `agent_system/prompt_builder.py`
- `agent_system/llm.py`
- `src/utils/local_llm_provider.py`
- `src/utils/prompt_budgeter.py`
- `agent_system/history_store.py`

Точный lifecycle здесь такой:

1. `start.py` поднимает combined runtime через `src/web/combined_app.py`.
2. `agent_system/api.py` принимает `POST /api/cognitive/chat/respond`.
3. Внешний request конвертируется в `ChatTurnRequest`.
4. `chat_engine.py` создаёт или грузит session.
5. При необходимости выполняется deterministic graph prewrite по самому сообщению.
6. `message_analyzer.py` строит `UserState` и entity list.
7. `situation_engine.py` строит `Situation`.
8. `feature_extractor.py` и `classifier_forest.py` классифицируют сущности.
9. `head_caller.py` выбирает primary persona-head.
10. `persona_engine.py` обновляет emotion vector до вызова LLM.
11. `context_builder.py` строит bounded context.
12. `llm.py` вызывает local provider.
13. `history_store.py` записывает turn в session storage.
14. `persona_engine.py` записывает `situation_reaction`.
15. `chat_engine.py` формирует explainability-блоки `persona_selection` и `persona_response`.
16. `chat_engine.py` принимает явное решение о background rebuild и только затем возвращает `ChatTurnResult`.

Storage writes на этом пути происходят в строго фиксированных местах:

- `memory/graphs/*` — только если был deterministic graph prewrite;
- `memory/sessions/{session_id}.txt` — после получения assistant reply;
- `memory/heads/{head}/emotion_vector.json` — при emotion update;
- `memory/heads/{head}/examples.json` — при записи `situation_reaction`;
- `memory/proposals/*` и rebuild artifacts — только по явному решению scheduler-а.

### 9.2 Если загружается файл

Участвуют:

- `agent_system/api.py`
- `agent_system/file_ingestion.py`
- `agent_system/entity_extractor.py`
- `agent_system/prompt_builder.py`
- `agent_system/llm.py`
- `agent_system/graph_store.py`
- `agent_system/persona_engine.py`

### 9.3 Если пользователь работает с графом

Участвуют:

- `webapp/src/App.jsx`
- `webapp/src/components/Graph/GraphWorkspace.jsx`
- `webapp/src/api.js`
- `agent_system/api.py`
- `agent_system/graph_store.py`
- `agent_system/graph_localizer.py`
- `agent_system/node_rethinker.py`
- `agent_system/memory_layers.py`

### 9.4 Если материализуется или обновляется persona

Участвуют:

- `agent_system/persona_engine.py`
- `agent_system/prompt_builder.py`
- `agent_system/llm.py`
- `agent_system/graph_store.py`

Именно здесь создаются:

- traits;
- relations;
- examples;
- emotion vector;
- baseline definition;
- dynamic state;
- learned patterns;
- revision trail;
- maturity/confidence indicators;
- log tuples;
- persona form;
- decision explanation.

---

## 10. Ограничения и правила системы

### 10.1 Что нельзя отдавать на волю LLM

Нельзя:

- прямое управление graph mutations;
- direct emotion inheritance from user;
- свободную типизацию сущностей без validation;
- raw relation injection;
- неограниченный rethink mode.

### 10.2 Что разрешено LLM

Разрешено:

- structured extraction;
- structured persona synthesis;
- translation;
- final reply generation;
- content suggestions для rethink mode.

### 10.3 Что должно делаться кодом

Кодом должны делаться:

- routing;
- validation;
- graph merge;
- graph hygiene;
- duplicate resolution;
- persona materialization;
- emotion update;
- preview/apply logic;
- storage writes.

---

## 11. Текущие тесты

Главная живая зона тестов:

- `tests/agent_system/`

Она покрывает:

- chat engine;
- classifier and heads;
- context builder;
- concept graphs;
- graph editor;
- graph hygiene;
- graph lifecycle;
- graph localizer;
- local llm provider policy;
- node rethinker;
- API failures.

Базовая команда:

```bash
python3 -m pytest tests/agent_system -q
```

---

## 12. Как читать проект AI-агенту

Если AI-агенту нужно быстро войти в проект, лучший порядок чтения такой:

1. `start.py`
2. `agent_system/runtime_config.py`
3. `agent_system/api.py`
4. `agent_system/chat_engine.py`
5. `agent_system/message_analyzer.py`
6. `agent_system/situation_engine.py`
7. `agent_system/persona_engine.py`
8. `agent_system/context_builder.py`
9. `agent_system/graph_store.py`
10. `agent_system/node_rethinker.py`
11. `agent_system/llm.py`
12. `src/utils/local_llm_provider.py`
13. `agent_system/observability.py`
14. `agent_system/memory_layers.py`
15. `docs/runtime_flow.md`
16. `docs/observability.md`
17. `docs/context_pipeline.md`
18. `docs/persona_lifecycle.md`
19. `docs/memory_lifecycle.md`
20. `docs/graph_lifecycle.md`
21. `webapp/src/App.jsx`
22. `webapp/src/components/Graph/GraphWorkspace.jsx`
23. `tests/agent_system/`

Если задача только по runtime-speed, то фокус должен быть на:

- `agent_system/llm.py`
- `src/utils/local_llm_provider.py`
- `src/utils/prompt_budgeter.py`
- `agent_system/context_builder.py`
- `agent_system/chat_engine.py`
- `agent_system/reliability.py`

Если задача только по graph quality, то фокус должен быть на:

- `agent_system/graph_store.py`
- `agent_system/duplicate_resolver.py`
- `agent_system/node_rethinker.py`
- `agent_system/entity_extractor.py`

Если задача только по persona behavior, то фокус должен быть на:

- `agent_system/persona_engine.py`
- `agent_system/message_analyzer.py`
- `agent_system/situation_engine.py`
- `agent_system/context_builder.py`

---

## 13. Краткий вывод

Этот проект — не “обёртка вокруг LLM”.
Это детерминированная AI-система с:

- file-first memory;
- graph-grounded storage;
- persona heads;
- situation-based emotion model;
- constrained rethink pipeline;
- role-based local LLM access;
- UI для ручного контроля и graph editing.

Главная практическая ошибка при работе с этим репозиторием — воспринимать его как pure prompt system. Это неверно.

Правильная модель такая:

```text
Это orchestration system,
в которой LLM — лишь один из подчинённых модулей.
```
