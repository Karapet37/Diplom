# autograph-integration-sdk

Python client for the project’s optional integration-layer surface.

This package is auxiliary. The canonical runtime remains the controller-first `agent_system/` backend started through `start.py`.

## Install

```bash
pip install ./packages/python-sdk
```

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

- `manifest()` fetches the integration-layer manifest.
- `invoke()` sends a raw payload to the integration-layer invoke surface.
- `invoke_action()` wraps a named action with `input` and `options`.
- `respond()` calls `wrapper.respond`.
- `archive_chat()` calls `archive.chat`.
- `update_user_graph()` calls `user_graph.update`.
- `ingest_personal_tree()` calls `personal_tree.ingest`.

## Notes

- This SDK is not required to run the main persona-graph agent.
- Use it only when you need to call the integration-layer surface from another Python process or tool.
- HTTP mode targets `/api/integration/layer/manifest` and `/api/integration/layer/invoke`.
