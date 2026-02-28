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
- `src/web/integration_sdk.py` և `packages/`՝ standalone/integration SDK-ներ արտաքին գործիքների մեջ շերտով միացնելու համար։
- `webapp`՝ UI demo execution, daily mode, user graph updates, graph node assist և graph editing-ի համար։

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
- `living_system/prompt_brain`-ում backend մակարդակով միացվել է prompt -> role routing-ը՝
  - `code_architect` -> `coder_architect`,
  - `code_patch` -> `coder_refactor`,
  - `translate_text` -> `translator`,
  - մնացած prompt-ները -> `general`։
- Ավելացվել է `Prompt Security Scanner` (`POST /api/living/prompt/run`)՝
  - հայտնաբերում է վտանգավոր command/process pattern-ներ,
  - կանգնեցնում է inference-ը մինչև user decision,
  - վերադարձնում է explanation + match snippet-ներ + `proceed/cancel` ընտրանքներ։
- `project_user_graph_update` և `project_llm_debate` endpoint-ներում ավելացվել են `personalization` և `feedback_items`։
- Debate prompt pipeline-ը հիմա ներառում է personalization context (`style/depth/risk/tone` + role defaults)։
- Frontend-ում ավելացվել է `Personalization Studio` և persistent local profile (`localStorage`)։
- UI language matrix-ը ընդլայնվել է՝ `fr`, `ar`, `hi`, `ja` լեզուներով և հստակ ordering-ով։
- Graph explainability UI-ը ուժեղացվել է՝
  - reasoning trace variants,
  - edge reasoning panel,
  - edge history timeline,
  - live edge stream visual effects։
- Ավելացվել են առանձին installable SDK փաթեթներ՝
  - `packages/python-sdk` (`autograph-integration-sdk`),
  - `packages/integration-layer-sdk` (`@autograph/integration-layer-sdk`)։
- Նույն integration contract-ը հիմա օգտագործվում է և UI-ում, և արտաքին հավելվածների embedding-ի համար։

### 4.1 Վերջին փոփոխությունների հաշվետվություն

- UI-ից ամբողջությամբ հանվել է `autoruns` բաժինը, քանի որ այն այլևս արժեքավոր user flow չէր տալիս։
- `graph` էկրանն այժմ կենտրոնացած է իրական edit/assist գործողությունների վրա, ոչ թե raw debug output-ի։
- `Graph node assist` և `Graph edge assist`-ի համար ավելացվել են առանձին interactive popover-ներ՝
  - node click -> explanation/improve/risks/tasks/memory,
  - edge click -> explain/improve/risks/merge/split,
  - localized labels/hints՝ բոլոր աջակցվող UI լեզուներով։
- `node/edge popover` UI-ը դուրս է բերվել առանձին component-ների մեջ (`GraphPopovers.jsx`), որպեսզի `App.jsx`-ը այլևս ամբողջությամբ monolith render չլինի։
- Ավելացվել է առանձին backend endpoint՝ `POST /api/graph/edge/assist`։
  - Այն իրականում վերլուծում է կապը,
  - գրում է փոփոխության metadata-ն edge-ի վրա,
  - ստեղծում է session/update links graph-ում,
  - վերադարձնում է թարմ snapshot։
- Ավելացվել է նոր backend/UI հոսք՝ `Create Foundation` (`POST /api/graph/foundation/create`)։
  - Օգտագործում է `Mistral`-ը որպես default planner model,
  - topic/selected node-ի հիման վրա կառուցում է տեսանելի foundation branch,
  - ստեղծում է `foundation_branch` root node,
  - ավելացնում է nested `concept` nodes,
  - կապում է դրանք `expands_concept`, `contains_concept`, `deepens_concept` edge-երով,
  - հետո գործարկում է `graph monitor`՝ փոքր, անվտանգ patch-երի համար։
- `Create Foundation`-ը control-plane-ում gated է որպես
  - `graph write`,
  - `prompt execution`,
  ուստի այն չի շրջանցում runtime policy-ն։
- Observability շերտում `Create Foundation`-ը արդեն հաշվառվում է inference endpoint-ների թվում։
- Վերջին iteration-ից հետո նախագծի հիմնական quality check-երը կրկին անցել են (`web build` + backend unit tests)։

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
  - graph node assist service,
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

2. Հիմքային ճյուղի կառուցում (`Create Foundation`)

- Եթե օգտատերը ցանկանում է թեման «բացել» և մանրամասնել, graph բաժնում օգտագործում է `Create Foundation` կոճակը։
- Կարելի է կամ ընտրել արդեն գոյություն ունեցող node, կամ պարզապես գրել նոր topic։
- Backend-ը `POST /api/graph/foundation/create`-ով կանչում է `Mistral` planner model-ը և ստեղծում է նոր foundation branch։
- Ստացված concepts-ը անմիջապես երևում են graph-ում որպես նոր nodes/edges, ոչ թե մնում են միայն text output-ի մակարդակում։

3. Սցենարի մուտքագրում և գործողությունների մոդելավորում

- Օգտատերը տալիս է կոնկրետ իրավիճակ (օր.` «ինչպես կատարել անցում նոր կարիերայի ուղղության»)։
- `POST /api/project/llm/debate`-ը ստեղծում է proposer/critic/judge տարբերակներ։
- Graph-ում ձևավորվում են վարկածներ, հակափաստարկներ և եզրակացություն՝ առանձին ճյուղերով։

4. Վերլուծություն և պլանի բարելավում

- `POST /api/project/daily-mode`-ով համակարգը գնահատում է ընթացիկ վիճակը, ռիսկերը և առաջընթացը։
- Օգտատերը համեմատում է տարբեր պլաններ, տեսնում dependency-ները և ընտրում հաջորդ քայլերը։
- Feedback-ը պահվում է, որպեսզի հաջորդ խորհուրդները լինեն ավելի անհատական։

5. Graph assist ըստ click-ի

- Օգտատերը սեղմում է node-ի վրա և ստանում է contextual popover։
- `POST /api/graph/node/assist`-ը տալիս է explanation/improve/risks/tasks/memory արդյունք։
- Օգտատերը կարող է նույն պատուհանից արդյունքը ուղարկել `multitool`-ի request/task/risk/note հոսքերի մեջ։
- Օգտատերը կարող է նաև սեղմել edge-ի վրա և գործարկել `POST /api/graph/edge/assist`։
- Սա օգնում է վերանայել կապի իմաստը, ռիսկը, merge/split հնարավորությունը և փոփոխել graph semantics-ը։

6. «Hallucination Hunter» ցիկլ

- Եթե պատասխանում հայտնվում է սխալ պնդում, այն գրանցվում է `POST /api/project/hallucination/report`-ով։
- Համակարգը պահում է սխալ տարբերակը, ճշգրիտ տարբերակը և աղբյուրը graph-ի առանձին verification branch-ում։
- Նույնատիպ հարցի ժամանակ `POST /api/project/hallucination/check`-ը տալիս է guard hints՝ կրկնվող սխալը կանխելու համար։

7. Archive chat + review ցիկլ

- `POST /api/project/archive/chat`-ով օգտատերը գրում է բնական լեզվով հարցը և ընտրում model path/role։
- Համակարգը վերադարձնում է conversational assistant reply, իսկ structural update-ները տալիս է review բլոկով։
- `POST /api/project/archive/review`-ով օգտատերը խմբագրում/վերավերացնում է update-ները և հետո կիրառում graph branch-ում։

8. Living prompt-ների անվտանգ գործարկում

- Եթե օգտագործվում է `POST /api/living/prompt/run`, backend scanner-ը նախ ստուգում է վտանգավոր հրամանները։
- Եթե ռիսկ է հայտնաբերվում, պատասխանում ստացվում է `blocked_for_confirmation` կարգավիճակ։
- Օգտատերը ընտրում է՝
  - `security_decision: "proceed"` (`force_execute: true`)՝ «ամեն դեպքում կատարել»,
  - `security_decision: "cancel"`՝ «չկատարել»։
- Պատասխանում միշտ վերադառնում է `security` բլոկ (`risk_level`, `matches`, `explanation`)՝ audit/review-ի համար։

9. Live դիտարկում

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
- `mistral`՝ general/planner reasoning-ի համար,
- `qwen2.5`՝ analyst reasoning-ի համար,
- `h2o-danube3`՝ creative ideation-ի համար,
- `qwen2.5-coder`՝ coding role-ների համար,
- `MADLAD`՝ translator role-ի համար։

Ակտիվ role mapping-ը runtime-ում ստուգվում է `GET /api/project/model-advisors` endpoint-ով։

### 7.2 Prompt Security Scanner API նշումներ

- Endpoint՝ `POST /api/living/prompt/run`
- Նոր request fields՝
  - `security_decision` (`proceed|cancel`)
  - `force_execute` (`true|false`, shortcut for `proceed`)
- Հիմնական response statuses՝
  - `ok`
  - `blocked_for_confirmation`
  - `cancelled_by_user`

### 7.3 Efficiency-First Wrapper (առավել արդյունավետ աշխատանքային ռեժիմ)

Եթե պետք է հենց `LLM GGUF`-ի վրա թեթև, արագ և user-adaptive շերտ (առանց UI «էֆեկտների»), օգտագործվում է wrapper API-ը՝

- `POST /api/project/wrapper/respond`
- `GET /api/project/wrapper/profile`
- `POST /api/project/wrapper/profile`
- `POST /api/project/wrapper/feedback`

Հիմնական նպատակը՝
- role/model selection (`role` կամ explicit `model_path`),
- graph-memory retrieval (`owned/all` scope),
- personalization կիրառություն (`style/depth/risk/tone/goals`),
- feedback loop-ով ավտոմատ հարմարեցում օգտատիրոջ նախասիրություններին։

Այս ռեժիմը նախագծի «արտադրողական» միջուկն է, երբ պետք է առավելագույն արդյունավետություն և նվազագույն ավելորդ ֆունկցիոնալ բեռ։

### 7.4 Installable SDK-ներ (առանձին օգտագործման և embedding-ի համար)

Համակարգի integration layer-ը հիմա հասանելի է ոչ միայն backend endpoint-ներով, այլ նաև առանձին installable փաթեթներով։

Փաթեթներ՝
- Python՝ `packages/python-sdk` (`autograph-integration-sdk`)
- JavaScript՝ `packages/integration-layer-sdk` (`@autograph/integration-layer-sdk`)

Տեղադրում՝

```bash
pip install ./packages/python-sdk
npm install ./packages/integration-layer-sdk
```

Աջակցվող ռեժիմներ՝
- `standalone`՝ անմիջապես աշխատում է local workspace/service-ի հետ, առանց HTTP-ի։
- `integration`՝ աշխատում է `/api/integration/layer/manifest` և `/api/integration/layer/invoke` endpoint-ներով։

Python standalone օրինակ՝

```python
from autograph_integration_sdk import IntegrationLayerClient
from src.web.graph_workspace import GraphWorkspaceService

workspace = GraphWorkspaceService(use_env_adapter=False, enable_living_system=False)
client = IntegrationLayerClient.from_workspace(
    workspace,
    host="vscode",
    app_id="workspace_plugin",
)

manifest = client.manifest()
result = client.respond(
    "Կազմիր հաջորդ կարճ գործողությունների պլանը",
    user_id="demo_user",
    session_id="sess_1",
)
```

Python integration (HTTP) օրինակ՝

```python
from autograph_integration_sdk import IntegrationLayerClient

client = IntegrationLayerClient.from_http(
    "http://127.0.0.1:8008",
    host="chat_agent",
    app_id="bridge_tool",
)

manifest = client.manifest()
result = client.invoke_action(
    "archive.chat",
    user_id="demo_user",
    session_id="sess_2",
    input_payload={"message": "ստուգիր այս update-ը"},
)
```

JS փաթեթի օրինակ՝

```javascript
import {
  createHttpIntegrationLayerClient,
  createStandaloneIntegrationLayerClient,
} from "@autograph/integration-layer-sdk";

const httpClient = createHttpIntegrationLayerClient({
  baseUrl: "http://127.0.0.1:8008",
  host: "vscode",
  appId: "workspace_plugin",
});

const standaloneClient = createStandaloneIntegrationLayerClient({
  host: "generic",
  appId: "local_tool",
  standaloneManifest: async (payload) => ({ ok: true, payload }),
  standaloneInvoke: async (payload) => ({ ok: true, result: payload }),
});
```

## 8. Առաջարկվող Quality Gates

Մինչև release՝

- `PYTHONPATH=. pytest -q tests/unit/test_graph_workspace.py`
- `PYTHONPATH=. pytest -q tests/unit/test_project_unified.py tests/unit/test_control_plane.py`
- `PYTHONPATH=. pytest -q tests/unit/test_living_system.py`
- `npm --prefix webapp run build`
- manual API checks՝
  - `/api/project/daily-mode`,
  - `/api/project/user-graph/update`,
  - `/api/project/llm/debate`,
  - `/api/graph/node/assist`,
  - `/api/graph/edge/assist`,
  - `/api/graph/foundation/create`,
  - `/api/living/prompt/run` (scanner decision flow)։

## 9. Գործնական եզրակացություն

Նախագիծը գործնականում կիրառելի է որպես՝

- personal `semantic knowledge graph engine`,
- local-`LLM`-assisted context system,
- monitoring-aware operational platform։

Ամենաբարձր ազդեցությամբ հաջորդ քայլը rewrite-ը չէ, այլ controlled stabilization-ը՝

- մաքուր repository state,
- codified integration contracts,
- repeatable `CI` և release routine։
