# autograph-integration-sdk

Python client for the project’s optional integration-layer surface.

This package is auxiliary. The canonical runtime remains the controller-first `agent_system/` backend started through `start.py`.

## What It Is For

Use this SDK when another Python process needs to call the optional integration-layer wrapper around the project.

It is not the main operator API for:

- `/api/cognitive/chat/respond`
- annotation workspace and message-vector corrections
- graph operator actions in the web UI

Those canonical runtime surfaces are exposed directly by the backend REST API, not through this SDK.

## Install

```bash
pip install ./packages/python-sdk
```

## Modes

- `IntegrationLayerClient.from_http(...)`
  - talks to `/api/integration/layer/manifest` and `/api/integration/layer/invoke`
- `IntegrationLayerClient.from_workspace(...)`
  - binds directly to a Python workspace object that already exposes integration-layer handlers
- `IntegrationLayerClient(...)`
  - can be constructed manually with manifest/invoke callables for tests or embedded tools

## HTTP Client Example

```python
from autograph_integration_sdk import IntegrationLayerClient

client = IntegrationLayerClient.from_http(
    "http://127.0.0.1:8008",
    host="chat_agent",
    app_id="bridge_tool",
)

manifest = client.manifest()
reply = client.respond(
    "verify this update",
    user_id="demo_user",
    session_id="sess_1",
)

graph_result = client.update_user_graph(
    text="The user works as an architect and prefers concise plans.",
    user_id="demo_user",
    session_id="sess_1",
    display_name="demo_user",
    language="en",
)
```

## Standalone Client Example

```python
from autograph_integration_sdk import IntegrationLayerClient

client = IntegrationLayerClient(
    manifest_fn=lambda payload: {"ok": True, "manifest": payload},
    invoke_fn=lambda payload: {"ok": True, "payload": payload},
    default_host="local_host",
    default_app_id="local_tool",
)
```

## Workspace Client Example

```python
from autograph_integration_sdk import IntegrationLayerClient

client = IntegrationLayerClient.from_workspace(
    workspace,
    host="controller_runtime",
    app_id="ops_tool",
)

manifest = client.manifest()
```

## Helper Methods

- `manifest()` fetches the integration-layer manifest
- `invoke()` sends a raw payload to the invoke surface
- `invoke_action()` wraps a named action with `input` and `options`
- `respond()` calls `wrapper.respond`
- `archive_chat()` calls `archive.chat`
- `update_user_graph()` calls `user_graph.update`
- `ingest_personal_tree()` calls `personal_tree.ingest`

## Notes

- This SDK is optional and not required to run the main persona-graph runtime.
- If you need the current chat, session, annotation, or training-example APIs, call the main backend REST endpoints directly.
- `manifest()` and `invoke()` normalize `host` and `app_id`, so callers can pass human-readable values and let the client sanitize them.
