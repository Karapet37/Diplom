# @autograph/integration-layer-sdk

Standalone and HTTP integration client for the Autograph integration layer.

## Install

```bash
npm install ./packages/integration-layer-sdk
```

## HTTP mode

```javascript
import { createHttpIntegrationLayerClient } from "@autograph/integration-layer-sdk";

const client = createHttpIntegrationLayerClient({
  baseUrl: "http://127.0.0.1:8008",
  host: "vscode",
  appId: "workspace_plugin",
});

const manifest = await client.manifest();
const result = await client.respond("Build my next action plan", {
  user_id: "web_user",
  session_id: "sess_web_1",
});
```

## Standalone mode

```javascript
import { createStandaloneIntegrationLayerClient } from "@autograph/integration-layer-sdk";

const client = createStandaloneIntegrationLayerClient({
  host: "generic",
  appId: "local_tool",
  standaloneManifest: async (payload) => ({ ok: true, payload }),
  standaloneInvoke: async (payload) => ({ ok: true, result: payload }),
});

const manifest = await client.manifest();
```
