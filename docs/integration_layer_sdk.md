# Integration Layer SDK

This SDK supports two operation modes:

1. `standalone`: in-process calls into `GraphWorkspaceService` (no HTTP required).
2. `integration`: HTTP calls to `/api/integration/layer/manifest` and `/api/integration/layer/invoke`.

Packaged wrappers are also available:

1. Python package: `pip install ./packages/python-sdk`
2. JS package: `npm install ./packages/integration-layer-sdk`

## Python

### Standalone mode

```python
from src.web.graph_workspace import GraphWorkspaceService
from src.web.integration_sdk import IntegrationLayerClient

workspace = GraphWorkspaceService(use_env_adapter=False, enable_living_system=False)
client = IntegrationLayerClient.from_workspace(
    workspace,
    host="vscode",
    app_id="workspace_plugin",
)

manifest = client.manifest()
reply = client.respond(
    "Build a concise next-step plan",
    user_id="demo_user",
    session_id="sess_1",
)
```

### Integration mode (HTTP)

```python
from src.web.integration_sdk import IntegrationLayerClient

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
    input_payload={"message": "verify this archive patch"},
    options={"verification_mode": "strict"},
)
```

## Web/JS

```javascript
import { createIntegrationLayerClient } from "./integrationLayerSdk";

const client = createIntegrationLayerClient({
  mode: "integration", // or "standalone"
  host: "vscode",
  appId: "workspace_plugin",
});

const manifest = await client.manifest();
const out = await client.respond("Prepare my next action plan", {
  user_id: "web_user",
  session_id: "sess_web_1",
});
```

### JS standalone handlers

```javascript
const client = createIntegrationLayerClient({
  mode: "standalone",
  host: "generic",
  appId: "local_tool",
  standaloneManifest: async (payload) => ({ ok: true, payload }),
  standaloneInvoke: async (payload) => ({ ok: true, result: payload }),
});
```
