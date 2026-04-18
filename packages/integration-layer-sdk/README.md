# @autograph/integration-layer-sdk

JavaScript client for the optional integration-layer surface around the project.

This package is auxiliary. The main runtime is still the controller-first backend started through `start.py`.

## Install

```bash
npm install ./packages/integration-layer-sdk
```

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

- `manifest()` loads the integration-layer manifest.
- `invoke()` sends a raw invoke payload.
- `invokeAction()` wraps a named action with `input` and `options`.
- `respond()` calls `wrapper.respond`.
- `archiveChat()` calls `archive.chat`.
- `updateUserGraph()` calls `user_graph.update`.
- `ingestPersonalTree()` calls `personal_tree.ingest`.

## Notes

- This SDK is useful only when another JS process needs to call the integration layer.
- It is not part of the canonical persona-graph runtime path.
- HTTP mode uses the browser or `globalThis.fetch`; for older Node runtimes provide `fetchImpl` in the client config.
