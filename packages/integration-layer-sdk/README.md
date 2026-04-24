# @autograph/integration-layer-sdk

JavaScript client for the project’s optional integration-layer surface.

This package is auxiliary. The canonical runtime is still the controller-first backend started through `start.py`.

## What It Is For

Use this package when another JS process needs to call the optional integration-layer wrapper around the project.

It is not the main operator client for:

- `/api/cognitive/chat/respond`
- session annotation workspace
- message-vector correction saves
- graph operator tooling in the web UI

Those runtime surfaces are exposed directly through the backend REST API.

## Install

```bash
npm install ./packages/integration-layer-sdk
```

## Modes

- `createHttpIntegrationLayerClient(...)`
  - the standard path for browser or Node clients talking to a running backend
- `createStandaloneIntegrationLayerClient(...)`
  - wraps local async handlers for tests or embedded tools
- `createIntegrationLayerClient(...)`
  - generic constructor when you want to choose the mode manually

## HTTP Example

```javascript
import { createHttpIntegrationLayerClient } from "@autograph/integration-layer-sdk";

const client = createHttpIntegrationLayerClient({
  baseUrl: "http://127.0.0.1:8008",
  host: "web_client",
  appId: "workspace_plugin",
});

const manifest = await client.manifest();
const result = await client.respond("Build the next action plan", {
  user_id: "web_user",
  session_id: "sess_web_1",
});

const graphResult = await client.updateUserGraph("The user prefers concise replies.", {
  user_id: "web_user",
  session_id: "sess_web_1",
  input: {
    display_name: "web_user",
    language: "en",
  },
});
```

## Standalone Example

```javascript
import { createStandaloneIntegrationLayerClient } from "@autograph/integration-layer-sdk";

const client = createStandaloneIntegrationLayerClient({
  host: "local_tool",
  appId: "demo_client",
  standaloneManifest: async () => ({ ok: true, name: "demo" }),
  standaloneInvoke: async (payload) => ({ ok: true, result: payload }),
});

const manifest = await client.manifest();
const archived = await client.archiveChat("capture this summary", {
  user_id: "local_user",
  session_id: "sess_local",
});
```

## Helper Methods

- `manifest()` loads the integration-layer manifest
- `invoke()` sends a raw invoke payload
- `invokeAction()` wraps a named action with `input` and `options`
- `respond()` calls `wrapper.respond`
- `archiveChat()` calls `archive.chat`
- `updateUserGraph()` calls `user_graph.update`
- `ingestPersonalTree()` calls `personal_tree.ingest`

## Notes

- This SDK is useful only when another JavaScript process needs the integration-layer surface.
- It is not part of the canonical operator chat, annotation, or graph-editing path.
- HTTP mode uses the browser or `globalThis.fetch`; for older Node runtimes provide `fetchImpl` in the client config.
- The client normalizes `host` and `appId`/`app_id` into connector-safe tokens before sending requests.
