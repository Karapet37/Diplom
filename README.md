# Autonomous Graph Workspace

Graph-first platform for modeling autonomous logical systems.

Project scope is now intentionally minimal:
- graph engine (`src/autonomous_graph`),
- web API (`src/web`),
- React UI (`webapp`),
- local env + local GGUF provider (`src/utils`),
- long-living modular runtime (`src/living_system`).

Legacy agent/speech/personalizer modules were removed.

## Architecture

- `src/autonomous_graph/core.py`: node/edge model, operators (recursive generation, inference, propagation).
- `src/autonomous_graph/api.py`: high-level graph facade.
- `src/autonomous_graph/storage.py`: JSON / Neo4j adapters.
- `src/web/graph_workspace.py`: workspace service + LLM profile import into graph.
- `src/web/api.py`: FastAPI routes.
- `src/living_system/`: 10-layer resilient runtime (SQL knowledge, diagnostics, recovery, evolution, prompt brain).
- `src/utils/env_loader.py`: `.env` loader.
- `src/utils/local_llm_provider.py`: local GGUF orchestration.
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

Role-based auto-discovery also works from `models/gguf`:
- `general`
- `coder_architect`, `coder_reviewer`, `coder_refactor`, `coder_debug`
- `analyst`, `creative`, `planner`
- `translator` (MADLAD priority, translator-only policy)

Important:
- `translate_text` prompt resolves only `translator` role GGUF.
- Translation does not fallback to `general` model.

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
- Daily Mode (`/api/project/daily-mode`) creates AI diary entries, extracts goals/problems/wins, returns 3-5 recommendations and improvement scores.
- User semantic dimensions can be updated via `/api/project/user-graph/update`:
  - `fears`, `desires`, `goals`, `principles`, `opportunities`,
  - `abilities`, `access`, `knowledge`, `assets`.
- Foundation concepts are auto-seeded on startup when graph is empty:

```env
AUTOGRAPH_BOOTSTRAP_FOUNDATION=1
AUTOGRAPH_BOOTSTRAP_LIVING_FOUNDATION=1
```

If local `.gguf` is unavailable, demo still works via fallback persona synthesis.

## Sysinternals Autoruns Integration

Project supports importing Sysinternals Autoruns exports into the semantic graph:
- source reference: https://learn.microsoft.com/en-us/sysinternals/downloads/autoruns
- endpoint: `POST /api/project/autoruns/import`
- parser supports CSV/TSV (`autoruns` GUI export or `autorunsc` text export)

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

## User Graph Update API

```bash
curl -s -X POST "http://127.0.0.1:8008/api/project/user-graph/update" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id":"web_user",
    "display_name":"Web User",
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

## Mini Coder Advisors + Translator

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

Translator special case:
- MADLAD GGUF is used as translator-priority when detected.
- Reference model:
  - https://huggingface.co/google/madlad400-3b-mt/blob/main/model-q4k.gguf
- Recommended local path:
  - `models/gguf/madlad400-3b-mt/model-q4k.gguf`
- Or explicit env override:
  - `LOCAL_TRANSLATOR_GGUF_MODEL=/absolute/path/model-q4k.gguf`
- If translator model is missing, translation returns configuration error instead of using non-translator model.

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
git add docker-compose.yml infra/nginx/nginx-http.conf infra/nginx/nginx-https.conf infra/nginx/start-nginx.sh infra/nginx/certs/.gitkeep README.md .gitignore
git commit -m "Add optional external HTTPS nginx mode for docker deployment"
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
