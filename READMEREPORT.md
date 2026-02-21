# Autonomous Graph Workspace - Project Report

Ստեղծվել է: 2026-02-20 -> report generated with codex

Repository: `https://github.com/Karapet37/Diplom`

## 1. Ինչ է այս նախագիծը

Autonomous Graph Workspace-ը `graph-first AI platform` է, որը միավորում է՝

- `semantic graph engine` (`nodes/edges`, `inference`, `simulation`),
- կայուն և երկարաժամկետ `runtime` (`living_system` modules),
- `FastAPI backend`՝ operations և control-ի համար,
- `React frontend`՝ graph interaction-ի համար,
- local `GGUF` model orchestration՝ `llama_cpp`-ի միջոցով։

Հիմնական նպատակը՝ թափանցիկ `API`-ներով պահել և զարգացնել օգտատերերի, sessions-ի, concepts-ի և operational events-ի շարունակաբար թարմացվող semantic model։

## 2. Ինչպես է աշխատում (End-to-End)

### 2.1 Core Runtime Layers

- `src/autonomous_graph`: `graph model`, relation management, propagation/inference, storage adapters։
- `src/web`: `API routing`, security middleware, monitoring endpoints, client introspection, workspace orchestration։
- `src/living_system`: diagnostics, feedback loops, recovery/evolution primitives, `SQL-backed` knowledge representation։
- `src/utils/local_llm_provider.py`: local model role resolution և `GGUF loading policy`։
- `webapp`: UI՝ demo execution, daily mode, user graph updates, autoruns analysis և graph editing-ի համար։

### 2.2 Main Data Flows

1. `demo/watch`:
- Ստեղծում/թարմացնում է seeded graph context։
- Փորձում է `LLM-first persona extraction`։
- Եթե model-ը հասանելի չէ, անցնում է deterministic persona synthesis-ի։

2. `daily-mode`:
- Ընդունում է journal text։
- Քաղում է goals/problems/wins և recommendation set։
- Ակտիվացնում է profile update (`free-text -> structured JSON update`)։

3. `user-graph/update`:
- Ընդունում է և structured lists (`fears`, `goals`, `assets` և այլն), և free text։
- `LLM/heuristics`-ից կազմում է profile patch (`name`, `history`, `description`, dimensions)։
- Ըստ անհրաժեշտության client introspection session nodes-ը կապում է user graph-ի հետ։

4. `autoruns/import`:
- Normal mode՝ parse է անում `CSV/TSV Sysinternals export`։
- Auto mode՝ եթե table չկա, client telemetry-ից infer է անում startup/process profile։
- Հաշվում է risk score և արդյունքները կապում semantic graph-ի մեջ։

### 2.3 Persistence

- Graph state-ը պահպանվում է storage adapter-ի միջոցով։
- Living system knowledge և diagnostics-ը `SQL-backed` են։
- Profile exports-ը պահվում են `data/profile_exports/`-ում։

## 3. Security և Operations

Իրականացված controls՝

- `JWT auth` (configurable, write-only protection option),
- `Rate limiting` (`slowapi` + in-memory fallback),
- `Proxy trust gating`՝ client IP extraction-ի համար,
- `Runtime control plane`՝ global feature gates և read-only mode-ի համար,
- `Prometheus metrics endpoint` (`/metrics`),
- `Privacy noise plugin` (isolated, optional, disabled by default)։

Operational deployment՝

- local run (`python3 start.py` + `webapp`),
- containerized stack (`docker compose`)՝ optional external `HTTPS` և reverse proxy path-ով։

## 4. Ուժեղ կողմեր

- Լավ modularity graph engine-ի, web layer-ի և long-living runtime-ի միջև։
- Backward-compatible `API extension` pattern (նոր fields՝ առանց հին inputs-ը ջնջելու)։
- Գործնական local `LLM` integration role-based routing-ով և translator isolation-ով։
- Լավ operational features՝ health, metrics, control plane, feature gates։
- Աջակցում է mixed structured/unstructured user input semantic graph updates-ի համար։

## 5. Ներկայիս սահմանափակումներ և ռիսկեր

1. `Repository hygiene` risk:
- Workspace-ում հիմա կան շատ untracked/new files։
- Սա բարձրացնում է release predictability և review complexity ռիսկը։

2. `Documentation drift` risk:
- `README`-ն լայն է և endpoints-ի փոփոխությունների դեպքում կարող է արագ հնանալ։
- UI-level behavior-ի որոշ մանրամասներ դեռ implicit են։

3. `Model-dependency variance`:
- Վարքագծի որակը կախված է local `GGUF models`-ից և hardware-ից։
- Fallback paths-ը deterministic են, բայց semantic առումով ավելի պարզ են, քան `LLM mode`-ը։

4. `Mixed concern density` `graph_workspace.py`-ում:
- Հարուստ orchestration logic-ը մեկ module-ում մեծացնում է maintenance overhead-ը։

5. `Privacy/compliance` caution:
- Client introspection-ը օգտակար է, բայց պետք է հստակ deployment policy և user notice։

## 6. Առաջարկվող բարելավումներ

### P0 (Immediate)

- Կայունացնել `git baseline`-ը՝
  - որոշել tracked scope-ը և մաքրել untracked noise-ը,
  - կիրառել branch/PR workflow փոքր, review-ված commits-ներով։
- Ավելացնել `CI checks`՝
  - unit tests,
  - API smoke tests,
  - web build verification։
- Ավելացնել `API contract snapshots` critical endpoints-ի համար։

### P1 (Near-Term)

- Բաժանել `graph_workspace.py`-ը focused service modules-ի՝
  - profile update service,
  - autoruns service,
  - daily-mode orchestration service։
- Ավելացնել request/response `JSON schema docs` project endpoints-ի համար։
- Ավելացնել migration/version նշումներ `DB schema evolution`-ի համար։

### P2 (Mid-Term)

- Ընդլայնել observability-ը՝
  - latency/error budgets ըստ endpoint group-ի,
  - dashboard templates semantic update quality-ի համար։
- Ավելացնել policy-driven client telemetry retention/expiration։
- Ավելացնել conflict/contradiction tracking competing knowledge claims-ի համար։

## 7. Առաջարկվող Quality Gates

Մինչև release՝

- `PYTHONPATH=. pytest -q tests/unit/test_graph_workspace.py`
- `PYTHONPATH=. pytest -q tests/unit/test_project_unified.py tests/unit/test_control_plane.py tests/unit/test_autoruns_import.py`
- `npm --prefix webapp run build`
- manual API checks՝
  - `/api/project/daily-mode`,
  - `/api/project/user-graph/update`,
  - `/api/project/autoruns/import`։

## 8. Գործնական եզրակացություն

Նախագիծը կիրառելի է որպես՝

- personal `semantic knowledge graph engine`,
- local-`LLM`-assisted context system,
- monitoring-aware operational platform։

Ամենաբարձր ազդեցությամբ հաջորդ քայլը rewrite-ը չէ, այլ controlled stabilization-ը՝

- մաքուր repository state,
- codified integration contracts,
- repeatable `CI` և release routine։
