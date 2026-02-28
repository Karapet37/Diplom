import unittest

from src.web.integration_sdk import IntegrationLayerClient


class _WorkspaceStub:
    def __init__(self) -> None:
        self.manifest_payloads: list[dict] = []
        self.invoke_payloads: list[dict] = []

    def project_integration_layer_manifest(self, payload):
        root = dict(payload or {})
        self.manifest_payloads.append(root)
        return {"ok": True, "manifest_payload": root}

    def project_integration_layer_invoke(self, payload):
        root = dict(payload or {})
        self.invoke_payloads.append(root)
        return {
            "ok": True,
            "action": root.get("action"),
            "chat_response": "stub ok",
            "result": {"echo": root},
        }


class IntegrationLayerSdkTests(unittest.TestCase):
    def test_standalone_workspace_client_manifest_and_respond(self):
        workspace = _WorkspaceStub()
        client = IntegrationLayerClient.from_workspace(
            workspace,
            host="vscode",
            app_id="my_plugin",
        )

        manifest = client.manifest()
        self.assertTrue(manifest["ok"])
        self.assertEqual(workspace.manifest_payloads[-1]["host"], "vscode")
        self.assertEqual(workspace.manifest_payloads[-1]["app_id"], "my_plugin")

        response = client.respond(
            "build next action plan",
            user_id="u1",
            session_id="s1",
            role="planner",
            use_memory=False,
        )
        self.assertTrue(response["ok"])
        sent = workspace.invoke_payloads[-1]
        self.assertEqual(sent["action"], "wrapper.respond")
        self.assertEqual(sent["host"], "vscode")
        self.assertEqual(sent["app_id"], "my_plugin")
        self.assertEqual(sent["input"]["message"], "build next action plan")
        self.assertFalse(sent["options"]["use_memory"])
        self.assertEqual(sent["options"]["role"], "planner")

    def test_standalone_convenience_methods(self):
        workspace = _WorkspaceStub()
        client = IntegrationLayerClient.from_workspace(workspace, host="chat_agent", app_id="agent_bridge")

        client.archive_chat(
            "check this draft",
            user_id="u2",
            session_id="s2",
            context="strict facts only",
            model_role="analyst",
            apply_to_graph=False,
        )
        archive_payload = workspace.invoke_payloads[-1]
        self.assertEqual(archive_payload["action"], "archive.chat")
        self.assertEqual(archive_payload["input"]["context"], "strict facts only")
        self.assertFalse(archive_payload["options"]["apply_to_graph"])
        self.assertEqual(archive_payload["options"]["model_role"], "analyst")

        client.update_user_graph(
            text="I prefer concise plans and risk analysis",
            user_id="u2",
            session_id="s3",
            language="en",
            use_llm_profile=False,
        )
        profile_payload = workspace.invoke_payloads[-1]
        self.assertEqual(profile_payload["action"], "user_graph.update")
        self.assertEqual(profile_payload["input"]["language"], "en")
        self.assertFalse(profile_payload["options"]["use_llm_profile"])

        client.ingest_personal_tree(
            text="Article summary about project architecture",
            user_id="u2",
            session_id="s4",
            title="Architecture Notes",
            max_points=9,
        )
        ingest_payload = workspace.invoke_payloads[-1]
        self.assertEqual(ingest_payload["action"], "personal_tree.ingest")
        self.assertEqual(ingest_payload["input"]["title"], "Architecture Notes")
        self.assertEqual(ingest_payload["input"]["max_points"], 9)

    def test_http_client_uses_requester_contract(self):
        calls: list[dict] = []

        def fake_requester(method, url, payload, headers, timeout_seconds):
            calls.append(
                {
                    "method": method,
                    "url": url,
                    "payload": payload,
                    "headers": dict(headers or {}),
                    "timeout_seconds": timeout_seconds,
                }
            )
            if method == "GET":
                return {"ok": True, "mode": "http", "url": url}
            return {"ok": True, "mode": "http", "echo": payload}

        client = IntegrationLayerClient.from_http(
            "http://127.0.0.1:8008",
            host="image_creator",
            app_id="image_tool",
            headers={"X-App": "demo"},
            timeout_seconds=11.0,
            requester=fake_requester,
        )

        manifest = client.manifest()
        self.assertTrue(manifest["ok"])
        self.assertIn("/api/integration/layer/manifest", calls[0]["url"])
        self.assertIn("host=image_creator", calls[0]["url"])
        self.assertEqual(calls[0]["headers"]["X-App"], "demo")
        self.assertEqual(calls[0]["timeout_seconds"], 11.0)

        out = client.invoke_action(
            "archive.chat",
            user_id="u3",
            session_id="s3",
            input_payload={"message": "verify this patch"},
            options={"verification_mode": "strict"},
        )
        self.assertTrue(out["ok"])
        self.assertEqual(calls[-1]["method"], "POST")
        self.assertIn("/api/integration/layer/invoke", calls[-1]["url"])
        self.assertEqual(calls[-1]["payload"]["action"], "archive.chat")
        self.assertEqual(calls[-1]["payload"]["host"], "image_creator")
        self.assertEqual(calls[-1]["payload"]["app_id"], "image_tool")


if __name__ == "__main__":
    unittest.main()
