# Persona Graph Agent Web UI

Operator-facing frontend for the controller-first `Persona-Graph-Agent` runtime.

## Purpose

The web app is not a generic chatbot shell. It is an operator workspace for:

- session management,
- persona selection and inspection,
- personality deletion and training-example curation,
- file upload,
- chat,
- graph inspection and maintenance,
- runtime diagnostics.

The frontend is designed to expose route and memory behavior, not only final text replies.

## Main Surfaces

- `Sessions` create, delete, and switch between runtime sessions.
- `Chat` sends requests through `/api/cognitive/chat/respond` and shows safety, persona-selection, response-shaping, context-preview, and trace-inspection panels.
- `Persona workspace` inspects validated personas, deletes personality-construction records, curates training examples, and exports JSONL.
- `Files workspace` uploads supported documents into the active session.
- `Graph workspace` searches the graph, inspects localized node views, creates/connects/merges/deletes nodes, reviews node state, and runs rethink preview/apply flows.
- `Diagnostics` shows runtime metrics, recent traces, and graph-health summaries.
- `Language shell` localizes the operator chrome in `en`, `ru`, `hy`, and `zh`.

## API Used by the UI

Core runtime:

- `GET /api/health`
- `GET /api/cognitive/health`
- `GET /api/cognitive/sessions`
- `POST /api/cognitive/sessions`
- `GET /api/cognitive/sessions/{session_id}`
- `DELETE /api/cognitive/sessions/{session_id}`
- `POST /api/cognitive/chat/respond`
- `POST /api/cognitive/files/upload`
- `POST /api/cognitive/rebuild`
- `GET /api/cognitive/personalities`
- `GET /api/cognitive/personalities/{name}`

Graph workspace:

- `GET /api/cognitive/graph`
- `GET /api/cognitive/graph/subgraph`
- `GET /api/cognitive/graph/nodes/{node_id}/view`
- `POST /api/cognitive/graph/nodes`
- `DELETE /api/cognitive/graph/nodes/{node_id}`
- `POST /api/cognitive/graph/edges`
- `DELETE /api/cognitive/graph/edges/{edge_id}`
- `POST /api/cognitive/graph/nodes/merge`
- `POST /api/cognitive/graph/nodes/{node_id}/state`
- `POST /api/cognitive/graph/rethink`

Diagnostics and learning surfaces:

- `GET /api/cognitive/debug/metrics`
- `GET /api/cognitive/debug/traces`
- `GET /api/cognitive/debug/graph-health`
- `DELETE /api/cognitive/personality-construction/personalities/{personality_id}`
- `POST /api/cognitive/training-examples`
- `GET /api/cognitive/training-examples`
- `DELETE /api/cognitive/training-examples/{example_id}`
- `GET /api/cognitive/training-examples/export/jsonl`
- `POST /api/cognitive/safety/classify`

The client API module also exposes `POST /api/cognitive/safety/examples` for future operator tooling, but the current UI only calls `safety/classify` automatically on message send.

## Local Development

```bash
cd <project_root>/webapp
npm install
npm run dev
```

Default Vite API base is `/api`, proxied to the backend.

Run the backend in another terminal:

```bash
cd <project_root>
.venv/bin/python start.py --profile development
```

or API-only:

```bash
.venv/bin/python start.py --profile development --api-only
```

## Build

```bash
cd <project_root>/webapp
npm run build
```

Build output:

- `webapp/dist/`

When present, the combined backend can serve the built frontend at `/`.

## Notes

- The frontend follows the runtime’s controller-first design.
- Lightweight turns, persona fast-path turns, persona specification, and graph-backed reasoning are handled by the backend route system, not by frontend heuristics.
- UI strings and controls are aligned with the active operator workflow rather than the earlier experimental workspace layout.
- Safety classification runs client-side in parallel with the chat request — it does not block the send flow.
- The operator panel for chat shows per-message safety context alongside persona selection, context preview, response shaping, and trace inspection.
- Graph explanations and rethink requests follow the currently selected UI language where the backend supports localization.
- Dedicated widgets for `/notes` and planning-mode output are still not present in the frontend even though the backend exposes those APIs.
