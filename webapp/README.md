# Persona Graph Agent Web UI

Operator-facing frontend for the controller-first `Persona-Graph-Agent` runtime.

## Purpose

The web app is not a generic chatbot shell. It is the main operator workspace for:

- session lifecycle,
- persona selection and inspection,
- live chat with controller metadata,
- message-vector annotation and correction,
- graph inspection and rethink actions,
- file upload,
- runtime diagnostics,
- training-example curation.

The frontend is designed to expose how the runtime decided, not only what text it produced.

## Main Surfaces

- `Sessions`
  - create, switch, reload, and delete runtime sessions
- `Chat`
  - sends requests through `/api/cognitive/chat/respond`
  - shows persona selection, context preview, response shaping, safety context, analysis, and traces
  - opens the annotation workspace for message-vector correction
- `Persona workspace`
  - inspects validated personas, curates training examples, and exports JSONL
- `Files workspace`
  - uploads documents into the active session
- `Graph workspace`
  - searches the graph, inspects localized node views, edits nodes and edges, and runs rethink flows
- `Diagnostics`
  - shows runtime metrics, traces, and graph health
- `Language shell`
  - localizes operator chrome in `en`, `ru`, `hy`, and `zh`

## Current Chat Behavior

The current chat surface intentionally behaves like an operator tool rather than a blocking form.

### Immediate send flow

On submit:

1. the composer is cleared immediately,
2. the user message is inserted into the thread optimistically,
3. the backend request continues in the background,
4. the assistant reply is appended when it returns.

This avoids the old behavior where a sent message stayed inside the textarea until the model responded.

### Persona speaker labeling

The selected persona dropdown is the primary source of the assistant speaker label in the thread.

Priority order is:

1. currently selected persona in the UI,
2. persona name stored on the message/result,
3. non-generic stored speaker name,
4. fallback label when no persona is known.

The UI should therefore prefer a persona name such as `Катерина` or `Капитан Джек Воробей`, not a generic `Assistant` label.

### Prompt-leak cleanup

The thread strips common scaffolding and leaked reviewer blocks before rendering assistant messages, including patterns such as:

- `Review Notes`
- `Issues Identified`
- `Analyze the Request`
- raw reasoning boilerplate

The goal is to keep the thread readable even if the backend receives noisy model output.

## Annotation Workspace

The chat surface now supports a dedicated annotation/correction layer over ordinary session history.

Per message, the operator can inspect:

- `message_id`
- role
- raw/display text
- predicted coordinate vector
- context window
- context matrix
- transition interpretation

Per coordinate, the operator can edit:

- `main`
- `extra`

The operator can also open previous context messages and correct their vectors, not only the latest reply.

This matters because the current turn may be wrong due to a misread history, not only due to a misread current message.

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

Annotation and learning:

- `GET /api/cognitive/sessions/{session_id}/annotation-workspace`
- `POST /api/cognitive/sessions/{session_id}/annotations`
- `POST /api/cognitive/training-examples`
- `GET /api/cognitive/training-examples`
- `DELETE /api/cognitive/training-examples/{example_id}`
- `GET /api/cognitive/training-examples/export/jsonl`
- `POST /api/cognitive/safety/classify`

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

Diagnostics:

- `GET /api/cognitive/debug/metrics`
- `GET /api/cognitive/debug/traces`
- `GET /api/cognitive/debug/graph-health`

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

- The frontend follows the runtime’s controller-first design and does not implement route heuristics on its own.
- Safety classification runs client-side in parallel with chat send and does not block the request path.
- The annotation workspace is a correction layer separate from normal chat history.
- The UI keeps the selected persona sticky across chat use so the thread and operator surfaces reflect the intended responding identity.
- Dedicated rich widgets for notes/planning remain less developed than the chat and graph surfaces, even though the backend exposes those capabilities.
