import sys
import unittest
from pathlib import Path


class PackagedIntegrationSdkTests(unittest.TestCase):
    def test_packaged_python_sdk_import_and_use(self):
        package_src = Path("packages/python-sdk/src").resolve()
        self.assertTrue(package_src.exists())
        sys.path.insert(0, str(package_src))
        try:
            from autograph_integration_sdk import IntegrationLayerClient
        finally:
            try:
                sys.path.remove(str(package_src))
            except ValueError:
                pass

        class _WorkspaceStub:
            def project_integration_layer_manifest(self, payload):
                return {"ok": True, "payload": dict(payload or {})}

            def project_integration_layer_invoke(self, payload):
                root = dict(payload or {})
                return {"ok": True, "action": root.get("action"), "payload": root}

        client = IntegrationLayerClient.from_workspace(
            _WorkspaceStub(),
            host="vscode",
            app_id="packaged_plugin",
        )
        manifest = client.manifest()
        self.assertTrue(manifest["ok"])
        self.assertEqual(manifest["payload"]["host"], "vscode")

        out = client.respond("next step", user_id="u1", session_id="s1")
        self.assertTrue(out["ok"])
        self.assertEqual(out["action"], "wrapper.respond")
        self.assertEqual(out["payload"]["app_id"], "packaged_plugin")


if __name__ == "__main__":
    unittest.main()
