# Autonomous Graph Workspace

Graph-first platform for modeling autonomous logical systems.

## Repository

- GitHub: `https://github.com/Karapet37/Diplom`
- Main branch remote: `origin` -> `Karapet37/Diplom`

Project scope is now intentionally minimal:
- graph engine (`src/autonomous_graph`),
- web API (`src/web`),
- React UI (`webapp`),
- local env + local GGUF provider (`src/utils`),
- long-living modular runtime (`src/living_system`).

Legacy agent/speech/personalizer modules were removed.

## What Is New In This Revision

- Personalization profile is now first-class:
  - response style, reasoning depth, risk tolerance, tone,
  - focus goals/domain focus/avoid topics,
  - memory notes and default debate roles.
- `LLM Role Debate` now accepts personalization context and uses it in proposer/critic/judge prompts.
- `Hallucination Hunter` is added as a dedicated verification branch:
  - report hallucinations with wrong/correct answers,
  - reuse known corrections during future debates/checks.
- `Verified Archive Chat` is added:
  - user gets conversational assistant reply (not raw JSON),
  - archive JSON conclusions are reviewed/edited in a separate verification flow.
- User profile updates accept personalization and explicit feedback items.
- UI now has `Personalization Studio` with local persistence and auto-apply toggles.
- Graph UX improvements:
  - reasoning path variants,
  - edge reasoning panel,
  - edge change timeline,
  - live edge update effects via WebSocket stream.
- UI language set expanded and ordered:
  - Armenian, Russian, English, French, Spanish, Portuguese, Arabic, Hindi, Chinese, Japanese.

## Architecture

- `src/autonomous_graph/core.py`: node/edge model, operators (recursive generation, inference, propagation).
- `src/autonomous_graph/api.py`: high-level graph facade.
- `src/autonomous_graph/storage.py`: JSON / Neo4j adapters.
- `src/web/graph_workspace.py`: workspace service + LLM profile import into graph.
- `src/web/api.py`: FastAPI routes.
- `src/web/integration_sdk.py`: shared Python SDK contract for standalone + HTTP integration.
- `src/living_system/`: 10-layer resilient runtime (SQL knowledge, diagnostics, recovery, evolution, prompt brain).
- `src/utils/env_loader.py`: `.env` loader.
- `src/utils/local_llm_provider.py`: local GGUF orchestration.
- `packages/`: installable Python and JS SDK wrappers for external integrations.
- `webapp/`: React UI.

## Install

```bash
pip install -r requirements.txt
```

## Run Backend

```bash
python3 start.py --host 127.0.0.1 --port 8008
```

Health check:

```bash
curl http://127.0.0.1:8008/api/health
```

Metrics endpoint:

```bash
curl http://127.0.0.1:8008/metrics
```

## Run Frontend

```bash
cd webapp
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Product Usage Flow (Client)

1. Build your personal graph:
- provide goals, constraints, abilities, and context via UI or `POST /api/project/user-graph/update`.
- system creates a personal semantic graph (nodes, edges, reasoning context).

2. Simulate a real situation:
- send a scenario to `POST /api/project/llm/debate`.
- proposer/critic/judge roles generate alternatives and a final decision path.

3. Analyze and improve your plan:
- use `POST /api/project/daily-mode` to evaluate progress, risks, and next actions.
- apply feedback so the next iterations become more personalized.

4. Prevent repeated hallucinations:
- save wrong answers with `POST /api/project/hallucination/report`.
- check new answers against known issues with `POST /api/project/hallucination/check`.

5. Chat and archive updates:
- ask in natural language via `POST /api/project/archive/chat` with selected GGUF model path or role.
- review/edit generated archive updates separately via `POST /api/project/archive/review`.

## Efficiency-First Wrapper (Recommended)

If you want a minimal productive layer over GGUF models (without visual extras), use the wrapper endpoints:

- `POST /api/project/wrapper/respond`
- `GET /api/project/wrapper/profile`
- `POST /api/project/wrapper/profile`
- `POST /api/project/wrapper/feedback`

What it does:
- resolves GGUF by `role` or explicit `model_path`,
- retrieves top relevant graph-memory context (owned/all scope),
- applies user personalization (`style/depth/risk/tone/goals`),
- stores profile adaptation and feedback loop.

Minimal response call:

```bash
curl -s -X POST "http://127.0.0.1:8008/api/project/wrapper/respond" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id":"demo_user",
    "session_id":"demo_session",
    "message":"Собери короткий план безопасного релиза",
    "role":"analyst",
    "use_memory":true,
    "memory_scope":"owned",
    "memory_top_k":6
  }'
```

Update profile preferences:

```bash
curl -s -X POST "http://127.0.0.1:8008/api/project/wrapper/profile" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id":"demo_user",
    "preferred_role":"analyst",
    "memory_scope":"owned",
    "personalization":{
      "response_style":"concise",
      "reasoning_depth":"balanced",
      "risk_tolerance":"low",
      "tone":"direct"
    }
  }'
```

Send feedback for adaptation:

```bash
curl -s -X POST "http://127.0.0.1:8008/api/project/wrapper/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id":"demo_user",
    "feedback_items":[
      {"message":"короче и по шагам", "decision":"accept", "score":0.9},
      {"message":"без рискованных действий", "decision":"accept", "score":0.8}
    ]
  }'
```

## Integration SDK Packages

The integration layer is now available both inside this repository and as installable packages.

Available packages:
- Python: `packages/python-sdk` (`autograph-integration-sdk`)
- JS: `packages/integration-layer-sdk` (`@autograph/integration-layer-sdk`)

Install the Python package:

```bash
pip install ./packages/python-sdk
```

Install the JS package:

```bash
npm install ./packages/integration-layer-sdk
```

Two usage modes are supported:
- `standalone`: direct in-process calls to the workspace service, no HTTP required.
- `integration`: remote calls to `/api/integration/layer/manifest` and `/api/integration/layer/invoke`.

Python standalone example:

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
    "Build a concise next-step plan",
    user_id="demo_user",
    session_id="sess_1",
)
```

Python HTTP integration example:

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
    input_payload={"message": "verify this update"},
)
```

JS package example:

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

## Local GGUF Models

Put local models in:

```text
models/
```

or set explicit path in `.env`:

```env
LOCAL_GGUF_MODEL=/absolute/path/to/model.gguf
```

Optional tuning:

```env
LOCAL_MODELS_DIR=models/gguf
LOCAL_GGUF_N_CTX=8192
LOCAL_GGUF_TEMPERATURE=0.25
LOCAL_GGUF_MAX_LOADED=1
LOCAL_GGUF_MAX_TOKENS=220
```

Split GGUF note:
- Use shard entrypoint only: `...-00001-of-0000N.gguf`.
- If `LOCAL_GGUF_MODEL` points to `...-00002-of-...`, provider auto-remaps to shard `00001` when present.
- Keep all shards in the same directory.

Role-based auto-discovery also works from `models/gguf`:
- `general`
- `coder_architect`, `coder_reviewer`, `coder_refactor`, `coder_debug`
- `analyst`, `creative`, `planner`
- `translator` (MADLAD priority, translator-only policy)

Current workspace configuration (`.env` in this repository):
- `general` -> `models/gguf/textGen/mistral-7b-instruct-v0.3-q4_k_m.gguf` (recommended default)
- `analyst` -> `models/gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`
- `creative` -> `models/gguf/textGen/h2o-danube3-4b-chat-Q5_K_M.gguf`
- `planner` -> `models/gguf/textGen/mistral-7b-instruct-v0.3-q4_k_m.gguf`
- `coder_architect` -> `models/gguf/coder/qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf`
- `coder_reviewer` -> `models/gguf/coder/qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf`
- `coder_refactor` -> `models/gguf/coder/qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf`
- `coder_debug` -> `models/gguf/coder/qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf`
- `translator` -> `models/translator/model-q4k.gguf`

Additional local `textGen` models available in this repository:
- `models/gguf/textGen/mistral-7b-instruct-v0.3-q4_k_m.gguf`
- `models/gguf/textGen/GLM-4.7-Flash-Q5_K_M.gguf`
- `models/gguf/textGen/h2o-danube3-4b-chat-Q5_K_M.gguf`
- `models/gguf/textGen/llama-2-7b-chat.Q4_K_M.gguf`

Suggested usage profile:
- `mistral-7b-instruct-v0.3` for `general` and `planner`.
- `qwen2.5-7b-instruct` for `analyst`.
- `h2o-danube3-4b-chat` for `creative`.
- `qwen2.5-coder-7b-instruct` for all `coder_*` roles.

Important:
- `translate_text` prompt resolves only `translator` role GGUF.
- Translation does not fallback to `general` model.
- This local provider uses `llama_cpp` directly; Ollama is optional and not required.

## Profile -> Graph API

Get prompt template:

```bash
curl -s "http://127.0.0.1:8008/api/graph/profile/prompt?entity_type_hint=human"
```

Infer profile JSON from narrative text and build graph:

```bash
curl -s -X POST "http://127.0.0.1:8008/api/graph/profile/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "text":"Меня зовут Арам. Играю на гитаре, знаю английский, программирую.",
    "entity_type_hint":"human",
    "create_graph":true,
    "save_json":true
  }'
```

Saved JSON exports are written to:

```text
data/profile_exports/
```

## Core API

- `GET /api/graph/snapshot`
- `GET /api/graph/node-types`
- `GET /api/graph/events`
- `WS /api/graph/ws`
- `POST /api/graph/node`
- `POST /api/graph/node/update`
- `POST /api/graph/node/delete`
- `POST /api/graph/edge`
- `POST /api/graph/edge/update`
- `POST /api/graph/edge/delete`
- `POST /api/graph/simulate`
- `POST /api/graph/event/reward`
- `POST /api/graph/relation/reinforce`
- `POST /api/graph/persist`
- `POST /api/graph/load`
- `POST /api/graph/clear`
- `POST /api/graph/seed-demo`
- `GET /api/graph/profile/prompt`
- `POST /api/graph/profile/infer`
- `POST /api/project/demo/watch`
- `POST /api/project/daily-mode`
- `POST /api/project/llm/debate`
- `POST /api/project/hallucination/report`
- `POST /api/project/hallucination/check`
- `POST /api/project/archive/chat`
- `POST /api/project/archive/review`
- `POST /api/project/wrapper/respond`
- `GET /api/project/wrapper/profile`
- `POST /api/project/wrapper/profile`
- `POST /api/project/wrapper/feedback`
- `POST /api/project/user-graph/update`
- `POST /api/project/autoruns/import`
- `GET /api/project/model-advisors`
- `POST /api/client/introspect`
- `GET /api/project/db/schema`
- `GET /api/control/state`
- `POST /api/control/update`
- `POST /api/control/reload`

## Living System API

- `GET /api/living/architecture`
- `GET /api/living/health`
- `POST /api/living/process`
- `GET /api/living/graph-view`
- `POST /api/living/snapshot`
- `POST /api/living/rollback`
- `POST /api/living/safe-mode`
- `POST /api/living/human-override`
- `POST /api/living/feedback`
- `GET /api/living/evolution`
- `GET /api/living/prompts`
- `POST /api/living/prompt/run`
- `POST /api/living/project-map`
- `POST /api/living/file/create`
- `POST /api/living/file/update`
- `POST /api/living/file/delete`
- `POST /api/living/knowledge/analyze`
- `POST /api/living/knowledge/initialize`
- `GET /api/living/knowledge/evaluate`
- `POST /api/living/knowledge/branch`
- `POST /api/living/knowledge/merge`

## Unified Project API

- `GET /api/project/overview`
- `POST /api/project/pipeline`
- `POST /api/project/bootstrap`
- `GET /api/project/evaluate`

## Demo Flow

- Demo button now runs `watch_demo` scenario (`Alexa`) with LLM-first extraction and deterministic fallback.
- Daily Mode (`/api/project/daily-mode`) creates AI diary entries, extracts goals/problems/wins, returns 3-5 recommendations and improvement scores, and can project journal text into `profile_update_json`.
- User semantic dimensions can be updated via `/api/project/user-graph/update` using both structured lists and free-text narrative:
  - `fears`, `desires`, `goals`, `principles`, `opportunities`,
  - `abilities`, `access`, `knowledge`, `assets`.
- Client introspection can be linked to user profile graph nodes (`observed_in_session`) during profile updates.
- Foundation concepts are auto-seeded on startup when graph is empty:

```env
AUTOGRAPH_BOOTSTRAP_FOUNDATION=1
AUTOGRAPH_BOOTSTRAP_LIVING_FOUNDATION=1
```

If local `.gguf` is unavailable, demo still works via fallback persona synthesis.

WebSocket stream notes:
- First frame: `{"type":"hello","snapshot":...,"metrics":...,"events":[...]}`
- Runtime frames: `{"type":"graph_event","event":{"event_type":"...", ...}}`
- Simulation now emits progress milestones:
  - `simulation_started`
  - `simulation_phase` (e.g. recursive generation)
  - `simulation_infer_round`
  - `simulation_completed`

## Personalization Payloads

`POST /api/project/user-graph/update` supports:

```json
{
  "user_id": "web_user",
  "display_name": "Web User",
  "text": "Free-text profile narrative",
  "personalization": {
    "response_style": "balanced",
    "reasoning_depth": "deep",
    "risk_tolerance": "medium",
    "tone": "direct",
    "focus_goals": ["ship product", "improve focus"],
    "domain_focus": ["architecture", "security"],
    "avoid_topics": ["generic motivation"],
    "memory_notes": "Prefer short actionable plans",
    "llm_roles": {
      "proposer": "creative",
      "critic": "analyst",
      "judge": "planner"
    }
  },
  "feedback_items": [
    {"message": "Concrete plans work best", "score": 0.9, "decision": "accept"},
    {"message": "Avoid long theory", "score": 0.2, "decision": "reject"}
  ]
}
```

`POST /api/project/llm/debate` supports:

```json
{
  "topic": "Improve runtime reasoning transparency",
  "hypothesis_count": 3,
  "personalization": {
    "response_style": "concise",
    "reasoning_depth": "balanced",
    "risk_tolerance": "low",
    "tone": "neutral",
    "llm_roles": {
      "proposer": "creative",
      "critic": "analyst",
      "judge": "planner"
    }
  },
  "feedback_items": [
    {"message": "Prefer practical solutions", "score": 0.8, "decision": "accept"}
  ]
}
```

## Verified Archive Chat + Review

`POST /api/project/archive/chat`:
- accepts `message`, optional `context`, and either explicit `model_path` or `model_role`.
- returns:
  - `assistant_reply` (conversational answer for chat UI),
  - `archive_updates` (structured conclusions),
  - `verification` (issues/warnings/score/verified flag),
  - `review` block for separate edit-and-apply step.

`POST /api/project/archive/review`:
- accepts edited `archive_updates`,
- re-runs verification checks,
- writes verified/candidate updates into dedicated archive graph branch.

## Sysinternals Autoruns Integration

Project supports importing Sysinternals Autoruns exports into the semantic graph:
- source reference: https://learn.microsoft.com/en-us/sysinternals/downloads/autoruns
- endpoint: `POST /api/project/autoruns/import`
- parser supports CSV/TSV (`autoruns` GUI export or `autorunsc` text export)
- fallback mode supports semantic auto-detection from client telemetry when CSV/TSV is not provided (`auto_detect=true`)

Example:

```bash
curl -s -X POST "http://127.0.0.1:8008/api/project/autoruns/import" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Entry,Entry Location,Enabled,Category,Profile,Description,Publisher,Image Path,Launch String,Signer,Verified,VirusTotal\nOneDrive,HKCU\\\\Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Run,Enabled,Logon,user,OneDrive startup,Microsoft Corporation,C:\\\\Program Files\\\\Microsoft OneDrive\\\\OneDrive.exe,\\\"C:\\\\Program Files\\\\Microsoft OneDrive\\\\OneDrive.exe\\\",Microsoft Corporation,Signed,0/74",
    "user_id": "web_user",
    "session_id": "autoruns_session_1",
    "host_label": "Workstation"
  }'
```

Response includes:
- scan metadata,
- risk summary (`low|medium|high`),
- semantic binding node IDs,
- updated snapshot/metrics.

Auto-detect example (no CSV text):

```bash
curl -s -X POST "http://127.0.0.1:8008/api/project/autoruns/import" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "",
    "auto_detect": true,
    "query": "show startup risks for browser updates",
    "user_id": "web_user",
    "session_id": "autoruns_session_2",
    "host_label": "Web User",
    "client": {
      "platform": "Linux x86_64",
      "user_agent": "Mozilla/5.0",
      "language": "en-US"
    }
  }'
```

## User Graph Update API

```bash
curl -s -X POST "http://127.0.0.1:8008/api/project/user-graph/update" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id":"web_user",
    "display_name":"Web User",
    "text":"My name is Alex. I grew up with mathematics and music, now I work as backend engineer, and I want to build resilient AI systems.",
    "language":"en",
    "session_id":"profile_web_user_1",
    "use_llm_profile":true,
    "include_client_profile":true,
    "client":{"platform":"Linux x86_64","user_agent":"Mozilla/5.0","language":"en-US"},
    "fears":["failure","public speaking"],
    "desires":["freedom"],
    "goals":["ship product"],
    "principles":["honesty"],
    "opportunities":["new market"],
    "abilities":["teaching"],
    "access":["api","internal docs"],
    "knowledge":["python","math"],
    "assets":["laptop","digital library"]
  }'
```

Response contains `profile_update_json` for traceable `text -> structured graph update` behavior.

## Role-Based Prompt Brain

- Advisor catalog endpoint:
  - `GET /api/project/model-advisors`
- Living prompts endpoint:
  - `GET /api/living/prompts`
- Included mini advisors:
  - `coder_architect_advisor`
  - `coder_reviewer_advisor`
  - `coder_refactor_advisor`
  - `coder_debug_advisor`
  - `translate_text`
- Backend role routing is active for living prompts:
  - `code_architect` -> `coder_architect`
  - `code_patch` -> `coder_refactor`
  - `translate_text` -> `translator` (strict, no fallback)
  - non-mapped prompts -> `general`

Translator special case:
- MADLAD GGUF is used as translator-priority when detected.
- Reference model:
  - https://huggingface.co/google/madlad400-3b-mt/blob/main/model-q4k.gguf
- Recommended local path:
  - `models/gguf/madlad400-3b-mt/model-q4k.gguf`
- Or explicit env override:
  - `LOCAL_TRANSLATOR_GGUF_MODEL=/absolute/path/model-q4k.gguf`
- If translator model is missing, translation returns configuration error instead of using non-translator model.

## Prompt Security Scanner (`/api/living/prompt/run`)

Living prompt execution now includes a backend scanner for harmful command/process patterns before LLM inference.

Detected examples:
- destructive shell commands (`rm -rf`, `mkfs`, raw-disk `dd`)
- remote execution pipes (`curl|bash`, `wget|sh`)
- process abuse/termination (`fork bomb`, force `killall/pkill/taskkill`)
- disruptive system commands (`shutdown`, `reboot`, `poweroff`)
- obfuscated PowerShell encoded command invocations

API request fields:
- `security_decision`: `proceed` or `cancel`
- `force_execute`: boolean alias for `proceed` (when `security_decision` is empty)

Scanner response statuses:
- `blocked_for_confirmation`: risky input detected, waiting for explicit decision
- `cancelled_by_user`: user rejected execution
- `ok`: execution allowed (clear scan or explicit override)

Response includes:
- `security.status`, `security.risk_level`, `security.matches[]`, `security.explanation`
- options list with `proceed` / `cancel` for UI confirmation dialog

Example (blocked until confirmation):

```bash
curl -s -X POST "http://127.0.0.1:8008/api/living/prompt/run" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_name": "code_patch",
    "variables": {
      "language": "en",
      "task": "run rm -rf /tmp/cache and restart workers",
      "target_file": "ops.sh",
      "constraints": "none"
    },
    "user_id": "web_user",
    "session_id": "scan_demo"
  }'
```

Example (explicit override):

```bash
curl -s -X POST "http://127.0.0.1:8008/api/living/prompt/run" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_name": "code_patch",
    "variables": {"language":"en","task":"maintenance command","target_file":"ops.sh","constraints":"none"},
    "security_decision": "proceed"
  }'
```

## Security and Rate Limiting

All security features are backward-compatible and disabled by default.
Enable in `.env`:

```env
AUTH_ENABLE=1
AUTH_PROTECT_WRITE_ONLY=1
AUTH_JWT_SECRET=replace_with_long_random_secret_at_least_24_chars
AUTH_JWT_ISSUER=autograph
AUTH_USER=admin
AUTH_PASSWORD=replace_with_strong_password_at_least_12_chars

RATE_LIMIT_ENABLE=1
RATE_LIMIT_BACKEND=slowapi
RATE_LIMIT_PER_MINUTE=240

# Trust proxy headers only behind trusted reverse proxy
TRUST_PROXY_HEADERS=0
TRUSTED_PROXY_IPS=127.0.0.1,::1
```

Request token:

```bash
curl -X POST http://127.0.0.1:8008/api/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"<your-strong-password>"}'
```

Use token for write API:

```bash
curl -X POST http://127.0.0.1:8008/api/graph/seed-demo \
  -H "Authorization: Bearer <TOKEN>"
```

Auth hardening behavior:
- weak `AUTH_JWT_SECRET` is rejected when auth is enabled.
- weak `AUTH_PASSWORD` is rejected when auth is enabled.
- client IP for rate limiting/auth logs uses direct peer IP by default.
- `X-Forwarded-For` is used only when `TRUST_PROXY_HEADERS=1` and source is trusted.

## Runtime Control Plane

Centralized runtime feature gates are available for whole-system control:

- `GET /api/control/state`
- `POST /api/control/update`
- `POST /api/control/reload`

Control env flags:

```env
CONTROL_PLANE_ENABLE=1
CONTROL_READ_ONLY=0
CONTROL_ALLOW_GRAPH_WRITES=1
CONTROL_ALLOW_PROJECT_DEMO=1
CONTROL_ALLOW_PROJECT_DAILY=1
CONTROL_ALLOW_AUTORUNS_IMPORT=1
CONTROL_ALLOW_CLIENT_INTROSPECTION=1
CONTROL_ALLOW_LIVING_FILE_OPS=1
CONTROL_ALLOW_KNOWLEDGE_MUTATIONS=1
CONTROL_ALLOW_PROMPT_EXECUTION=1

# Optional extra guard for control update/reload endpoints:
CONTROL_ADMIN_KEY=replace_with_random_control_key
```

Behavior:
- `CONTROL_READ_ONLY=1` blocks API write operations (`POST/PUT/PATCH/DELETE`) except auth token and control endpoints.
- Per-feature flags block corresponding API groups with explicit reason.
- If `AUTH_ENABLE=0`, control update/reload requires `CONTROL_ADMIN_KEY`.

## Security Hardening Report (Latest)

Applied fixes:
- Removed write endpoints from default auth-exempt list (`/api/project/demo/watch`, `/api/project/daily-mode`).
- Added strong-secret/strong-password validation for auth mode.
- Added proxy-header trust gating to prevent `X-Forwarded-For` spoofing.
- Hardened prompt-brain file operations:
  - blocked null-byte payloads,
  - added write size limit (`OperationPolicy.max_file_bytes`, default `1_000_000`),
  - blocked writes to symlink targets.
- Added prompt security scanner for `/api/living/prompt/run`:
  - blocks risky command/process patterns pending explicit user decision,
  - supports explicit `security_decision` (`proceed`/`cancel`) and `force_execute`,
  - returns structured explanations and match snippets for UI review.
- Enforced translator GGUF responsibility:
  - `translator` role has no fallback to `general`,
  - translation prompt fails closed when translator model is missing.
- Added runtime control-plane with read-only and subsystem gates (`/api/control/*`).

Control update example:

```bash
curl -X POST http://127.0.0.1:8008/api/control/update \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "X-Control-Key: <CONTROL_ADMIN_KEY>" \
  -d '{"read_only": true, "allow_project_demo": false, "allow_autoruns_import": false}'
```

## Privacy Noise Plugin (Experimental)

Disabled by default, isolated from core graph logic.

```env
PRIVACY_NOISE_ENABLE=1
PRIVACY_NOISE_INTENSITY=0.05
PRIVACY_NOISE_SEED=20260217
```

Endpoint:

- `GET /api/privacy/noise/report`

## Docker Monitoring Stack

Includes backend + nginx + prometheus + grafana:

```bash
docker compose up -d --build
```

Services:

- API via Nginx: `http://127.0.0.1/api/health`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`

### External HTTPS (Non-Local) with Docker

Default compose startup is backward-compatible (HTTP mode).
To enable public HTTPS:

1. Point DNS:
- `A` record: `<your-domain>` -> your server public IP

2. Place certificates on host:
- `infra/nginx/certs/fullchain.pem`
- `infra/nginx/certs/privkey.pem`

3. Set TLS env in `.env`:

```env
NGINX_TLS_ENABLE=1
NGINX_SERVER_NAME=your-domain.com
NGINX_TLS_CERT_FILE=/etc/nginx/certs/fullchain.pem
NGINX_TLS_KEY_FILE=/etc/nginx/certs/privkey.pem
```

4. Start stack:

```bash
docker compose up -d --build
```

5. Verify:

```bash
curl -I https://your-domain.com/api/health
curl -I http://your-domain.com/api/health
```

Expected:
- HTTPS returns `200`
- HTTP returns `301` redirect to HTTPS

Notes:
- If `NGINX_TLS_ENABLE=1` but cert files are missing, container falls back to HTTP mode and logs warning.
- Port `443` is published in `docker-compose.yml`.
- Keep Prometheus/Grafana ports firewalled if they should not be public.

### Temporary Public URL (For Demo)

For quick external access from another computer without owning a domain:

1. Start stack:

```bash
docker compose up -d --build
```

2. Set safe demo mode before sharing:

```bash
curl -X POST http://127.0.0.1:8008/api/control/update \
  -H "Content-Type: application/json" \
  -H "X-Control-Key: <CONTROL_ADMIN_KEY>" \
  -d '{"read_only": true}'
```

3. Run a temporary tunnel:

Option A (Cloudflare quick tunnel):
```bash
cloudflared tunnel --url http://127.0.0.1:8080
```

Option B (ngrok):
```bash
ngrok http 8080
```

Share the generated `https://...` URL.  
Use strong auth secrets and disable demo URL when session ends.

## Tests

```bash
python3 -m unittest discover -s tests/unit -p 'test_*.py'
```

## Git / GitHub Runbook

```bash
git status
git add README.md .gitignore
git commit -m "docs: refresh README and gitignore for current project behavior"
git push origin <your-branch>
```

### Git Refresh (safe sync)

```bash
git status
git fetch origin --prune
git checkout <your-branch>
git pull --rebase origin <your-branch>
```

If branch is behind `main`:

```bash
git fetch origin --prune
git rebase origin/main
```

Rollback last local commit (keep changes):

```bash
git reset --soft HEAD~1
```

Suggested PR checklist:
- `docker compose config` is valid
- backend starts: `python3 start.py`
- tests pass: `python3 -m unittest discover -s tests/unit -p 'test_*.py'`
- `https://<domain>/api/health` works when TLS env is enabled
