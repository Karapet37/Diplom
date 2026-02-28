# autograph-integration-sdk

Python SDK for the Autograph integration layer.

Supports:

1. Standalone in-process mode.
2. HTTP integration mode.

## Install

```bash
pip install ./packages/python-sdk
```

## Standalone

```python
from autograph_integration_sdk import IntegrationLayerClient
from src.web.graph_workspace import GraphWorkspaceService

workspace = GraphWorkspaceService(use_env_adapter=False, enable_living_system=False)
client = IntegrationLayerClient.from_workspace(workspace, host="vscode", app_id="workspace_plugin")

manifest = client.manifest()
result = client.respond("Give me the next action plan", user_id="demo_user", session_id="sess_1")
```

## HTTP

```python
from autograph_integration_sdk import IntegrationLayerClient

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
    input_payload={"message": "verify this update"},
)
```
