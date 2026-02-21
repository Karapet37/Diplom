# Autonomous Graph Workspace - Project Report

Ստեղծման ամսաթիվ՝ 2026-02-20 (հաշվետվությունը գեներացվել է Codex-ով)

Repository: `https://github.com/Karapet37/Diplom`

## Թեմայի ձևակերպում

- Մեքենայական ուսուցման միջոցով անհատականացված համակարգի մշակում
- Разработка персонализированной системы с использованием машинного обучения
- Development of a Personalized System Using Machine Learning

## 1. Նախագծի նպատակը

`Autonomous Graph Workspace`-ը `graph-first AI platform` է, որը միավորում է՝

- `semantic graph engine` (`nodes/edges`, `inference`, `simulation`),
- կայուն և երկարաժամկետ `runtime` (`living_system` modules),
- `FastAPI backend`՝ operations և control-ի համար,
- `React frontend`՝ graph interaction-ի համար,
- local `GGUF` model orchestration՝ `llama_cpp`-ի միջոցով։

Հիմնական նպատակը՝ թափանցիկ `API`-ներով պահել և զարգացնել օգտատերերի, sessions-ի, concepts-ի և operational events-ի շարունակաբար թարմացվող semantic model։

## 2. Ինչպես է աշխատում (End-to-End)

### 2.1 Հիմնական runtime շերտերը

- `src/autonomous_graph`՝ `graph model`, relation management, propagation/inference, storage adapters։
- `src/web`՝ `API routing`, security middleware, monitoring endpoints, client introspection, workspace orchestration։
- `src/living_system`՝ diagnostics, feedback loops, recovery/evolution primitives, `SQL-backed` knowledge representation։
- `src/utils/local_llm_provider.py`՝ local model role resolution և `GGUF loading policy`։
- `webapp`՝ UI demo execution, daily mode, user graph updates, autoruns analysis և graph editing-ի համար։

### 2.2 Հիմնական data flow-ներ

1. `demo/watch`
- Ստեղծում կամ թարմացնում է seeded graph context։
- Փորձում է `LLM-first persona extraction`։
- Եթե model-ը հասանելի չէ, անցնում է deterministic persona synthesis-ի։

2. `daily-mode`
- Ընդունում է journal text։
- Քաղում է goals/problems/wins և recommendation set։
- Ակտիվացնում է profile update (`free-text -> structured JSON update`)։

3. `user-graph/update`
- Ընդունում է և structured lists (`fears`, `goals`, `assets` և այլն), և free text։
- `LLM/heuristics`-ի միջոցով կազմում է profile patch (`name`, `history`, `description`, dimensions)։
- Ըստ անհրաժեշտության client introspection session nodes-ը կապում է user graph-ի հետ։

4. `autoruns/import`
- Normal mode-ում parse է անում `CSV/TSV Sysinternals export`։
- Auto mode-ում, եթե table չկա, client telemetry-ից infer է անում startup/process profile։
- Հաշվում է risk score և արդյունքները կապում semantic graph-ի մեջ։

### 2.3 Persistence

- Graph state-ը պահպանվում է storage adapter-ի միջոցով։
- Living system knowledge և diagnostics-ը `SQL-backed` են։
- Profile exports-ը պահվում են `data/profile_exports/`-ում։

## 3. Անվտանգություն և Operations

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

## 4. Ինչ է արված (կատարված աշխատանքներ)

- Իրականացվել է personal semantic graph-ի աշխատանքային հոսք։
- Ավելացվել է `LLM role debate` մեխանիզմ՝ proposer/critic/judge դերերով։
- Ավելացվել է `Hallucination Hunter` verification branch-ը՝
  - սխալ պատասխանների գրանցմամբ,
  - ճիշտ պատասխանների պահպանմամբ,
  - կրկնվող սխալների կանխման guard check-երով։
- Ավելացվել է `Verified Archive Chat` հոսքը՝
  - chat-ում վերջնական պատասխանը conversational ձևով է (ոչ թե raw JSON),
  - JSON update-ները գնում են առանձին review/edit փուլ՝ նորից verify անելու համար։
- `project_user_graph_update` և `project_llm_debate` endpoint-ներում ավելացվել են `personalization` և `feedback_items`։
- Debate prompt pipeline-ը հիմա ներառում է personalization context (`style/depth/risk/tone` + role defaults)։
- Frontend-ում ավելացվել է `Personalization Studio` և persistent local profile (`localStorage`)։
- UI language matrix-ը ընդլայնվել է՝ `fr`, `ar`, `hi`, `ja` լեզուներով և հստակ ordering-ով։
- Graph explainability UI-ը ուժեղացվել է՝
  - reasoning trace variants,
  - edge reasoning panel,
  - edge history timeline,
  - live edge stream visual effects։

## 5. Ինչ է պլանավորվում

### P0 (Immediate)

- Կայունացնել `git baseline`-ը՝
  - ճշտել tracked scope-ը,
  - մաքրել untracked noise-ը,
  - կիրառել փոքր և review-ված branch/PR workflow։
- Ավելացնել `CI checks`՝ unit tests, API smoke tests, web build verification։
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

## 6. Ինչ կարելի է բարելավել

1. `Repository hygiene`
- Workspace-ում կան շատ untracked/new files։
- Սա բարձրացնում է release predictability և review complexity ռիսկը։

2. `Documentation drift`
- `README`-ները պետք է պարբերաբար համաժամեցվեն endpoint-ների և UI behavior-ի հետ։

3. `Model-dependency variance`
- Արդյունքի որակը կախված է local `GGUF models`-ից և hardware-ից։
- Fallback paths-ը կայուն են, բայց semantic խորությամբ կարող են զիջել `LLM mode`-ին։

4. `Mixed concern density`
- `graph_workspace.py`-ում orchestration logic-ի բարձր խտությունը մեծացնում է maintenance overhead-ը։

5. `Privacy/compliance`
- Client introspection-ի համար անհրաժեշտ է հստակ deployment policy և user notice։

## 7. Ինչպես է հաճախորդը օգտագործում պատրաստի համակարգը

1. Անձնական graph-ի ստեղծում

- Օգտատերը գրում է իր կոնտեքստը (նպատակներ, խնդիրներ, ռեսուրսներ, սահմանափակումներ)։
- Համակարգը `POST /api/project/user-graph/update`-ով կառուցում է personal semantic graph։
- Արդյունքում տեսանելի են nodes/edges-ը և reasoning path-երը։

2. Սցենարի մուտքագրում և գործողությունների մոդելավորում

- Օգտատերը տալիս է կոնկրետ իրավիճակ (օր.` «ինչպես կատարել անցում նոր կարիերայի ուղղության»)։
- `POST /api/project/llm/debate`-ը ստեղծում է proposer/critic/judge տարբերակներ։
- Graph-ում ձևավորվում են վարկածներ, հակափաստարկներ և եզրակացություն՝ առանձին ճյուղերով։

3. Վերլուծություն և պլանի բարելավում

- `POST /api/project/daily-mode`-ով համակարգը գնահատում է ընթացիկ վիճակը, ռիսկերը և առաջընթացը։
- Օգտատերը համեմատում է տարբեր պլաններ, տեսնում dependency-ները և ընտրում հաջորդ քայլերը։
- Feedback-ը պահվում է, որպեսզի հաջորդ խորհուրդները լինեն ավելի անհատական։

4. «Hallucination Hunter» ցիկլ

- Եթե պատասխանում հայտնվում է սխալ պնդում, այն գրանցվում է `POST /api/project/hallucination/report`-ով։
- Համակարգը պահում է սխալ տարբերակը, ճշգրիտ տարբերակը և աղբյուրը graph-ի առանձին verification branch-ում։
- Նույնատիպ հարցի ժամանակ `POST /api/project/hallucination/check`-ը տալիս է guard hints՝ կրկնվող սխալը կանխելու համար։

5. Archive chat + review ցիկլ

- `POST /api/project/archive/chat`-ով օգտատերը գրում է բնական լեզվով հարցը և ընտրում model path/role։
- Համակարգը վերադարձնում է conversational assistant reply, իսկ structural update-ները տալիս է review բլոկով։
- `POST /api/project/archive/review`-ով օգտատերը խմբագրում/վերավերացնում է update-ները և հետո կիրառում graph branch-ում։

6. Live դիտարկում

- `GET /api/graph/snapshot` և `WS /api/graph/ws`-ով օգտատերը real-time տեսնում է graph-ի փոփոխությունները։
- Սա թույլ է տալիս անմիջապես հասկանալ, թե որ reasoning chain-ն է աշխատել, որը՝ ոչ։

### 7.1 Ընթացիկ GGUF model-ները (այս կոնֆիգուրացիայում)

- `general` (`LOCAL_GGUF_MODEL`):
  - `models/gguf/textGen/mistral-7b-instruct-v0.3-q4_k_m.gguf` (recommended default)
- `coder_architect`, `coder_reviewer`, `coder_refactor`, `coder_debug`:
  - `models/gguf/coder/qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf`
- `translator` (`LOCAL_TRANSLATOR_GGUF_MODEL`):
  - `models/translator/model-q4k.gguf` (`MADLAD` class translator profile)
- `textGen` պանակում հասանելի լրացուցիչ model-ներ:
  - `models/gguf/textGen/GLM-4.7-Flash-Q5_K_M.gguf`
  - `models/gguf/textGen/h2o-danube3-4b-chat-Q5_K_M.gguf`
  - `models/gguf/textGen/llama-2-7b-chat.Q4_K_M.gguf`
- Context tuning:
  - `LOCAL_GGUF_N_CTX=8192`
  - `LOCAL_TRANSLATOR_N_CTX=8192`
  - `LOCAL_GGUF_MAX_LOADED=1`

Առաջարկվող role strategy՝
- `mistral`՝ general/planning reasoning-ի համար,
- `qwen2.5-coder`՝ coding role-ների համար,
- `MADLAD`՝ translator role-ի համար։

Ակտիվ role mapping-ը runtime-ում ստուգվում է `GET /api/project/model-advisors` endpoint-ով։

## 8. Առաջարկվող Quality Gates

Մինչև release՝

- `PYTHONPATH=. pytest -q tests/unit/test_graph_workspace.py`
- `PYTHONPATH=. pytest -q tests/unit/test_project_unified.py tests/unit/test_control_plane.py tests/unit/test_autoruns_import.py`
- `npm --prefix webapp run build`
- manual API checks՝
  - `/api/project/daily-mode`,
  - `/api/project/user-graph/update`,
  - `/api/project/llm/debate`,
  - `/api/project/autoruns/import`։

## 9. Գործնական եզրակացություն

Նախագիծը գործնականում կիրառելի է որպես՝

- personal `semantic knowledge graph engine`,
- local-`LLM`-assisted context system,
- monitoring-aware operational platform։

Ամենաբարձր ազդեցությամբ հաջորդ քայլը rewrite-ը չէ, այլ controlled stabilization-ը՝

- մաքուր repository state,
- codified integration contracts,
- repeatable `CI` և release routine։
