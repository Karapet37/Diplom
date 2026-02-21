# Autonomous Graph React UI

Graph-first frontend for the Autonomous Graph Workspace.

Repository: `https://github.com/Karapet37/Diplom`

## New Overview Tools

- `User Semantic Graph`: update personal dimensions in graph
  - fears, desires, goals, principles, opportunities
  - abilities, access, knowledge, assets
- `Personalization Studio`:
  - response style, reasoning depth, risk tolerance, tone,
  - default LLM roles (proposer/critic/judge),
  - focus goals/domain focus/avoid topics/memory notes,
  - localStorage persistence + auto-apply toggles for Daily/UserGraph/Debate actions.
- `Sysinternals Autoruns Import`: paste Autoruns CSV/TSV export and bind startup/process telemetry into semantic graph
- `Daily Mode`: diary -> recommendations + scores + graph binding
- `Mini Coders / Advisors`: shows detected GGUF advisor roles and prompt catalog
- `Graph Explainability UX`:
  - reasoning path variants per selected node,
  - dependency closure highlight on hover,
  - edge reasoning panel,
  - edge change timeline from stream events,
  - animated live edge updates.
- `Translator advisor` is strict GGUF role (no fallback to general model)
- Overview now uses section pages (pagination) instead of one long vertical scroll

## UI Languages

Current UI language order:

1. Armenian (`hy`)
2. Russian (`ru`)
3. English (`en`)
4. French (`fr`)
5. Spanish (`es`)
6. Portuguese (`pt`)
7. Arabic (`ar`)
8. Hindi (`hi`)
9. Chinese (`zh`)
10. Japanese (`ja`)

## Run

```bash
cd webapp
npm install
npm run dev
```

Default API base is `/api` and Vite proxy forwards to `http://127.0.0.1:8008`.
For production Docker+Nginx HTTPS, frontend should call:
- `https://<your-domain>/api/...`
- no direct backend port exposure is required.

Start backend in another terminal:

```bash
python3 start.py --web-api --host 127.0.0.1 --port 8008
```

or simply:

```bash
python3 start.py
```

## Build

```bash
npm run build
```

Build output: `webapp/dist/`.
If present, FastAPI serves it at `/`.

## Backend Endpoints Used by UI

- `POST /api/project/daily-mode`
- `POST /api/project/user-graph/update`
- `POST /api/project/autoruns/import`
- `GET /api/project/model-advisors`
- `POST /api/client/introspect`
- `GET /api/graph/snapshot`
