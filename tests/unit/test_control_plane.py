import unittest

from src.web.control_plane import ControlPlaneFlags, RuntimeControlPlane


class ControlPlaneTests(unittest.TestCase):
    def test_read_only_blocks_write_requests(self):
        plane = RuntimeControlPlane(flags=ControlPlaneFlags(read_only=True))
        allowed, reason = plane.allow_request(method="POST", path="/api/graph/node")
        self.assertFalse(allowed)
        self.assertEqual(reason, "control_read_only")

    def test_read_only_keeps_control_update_available(self):
        plane = RuntimeControlPlane(flags=ControlPlaneFlags(read_only=True))
        allowed, reason = plane.allow_request(method="POST", path="/api/control/update")
        self.assertTrue(allowed)
        self.assertEqual(reason, "")

    def test_feature_gate_blocks_specific_path(self):
        plane = RuntimeControlPlane(
            flags=ControlPlaneFlags(
                allow_autoruns_import=False,
                allow_client_introspection=False,
            )
        )
        allowed_autoruns, reason_autoruns = plane.allow_request(
            method="POST",
            path="/api/project/autoruns/import",
        )
        allowed_introspect, reason_introspect = plane.allow_request(
            method="POST",
            path="/api/client/introspect",
        )
        self.assertFalse(allowed_autoruns)
        self.assertEqual(reason_autoruns, "autoruns_import_disabled")
        self.assertFalse(allowed_introspect)
        self.assertEqual(reason_introspect, "client_introspection_disabled")

    def test_prompt_execution_gate_blocks_project_llm_debate(self):
        plane = RuntimeControlPlane(
            flags=ControlPlaneFlags(
                allow_prompt_execution=False,
            )
        )
        allowed, reason = plane.allow_request(
            method="POST",
            path="/api/project/llm/debate",
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "prompt_execution_disabled")

    def test_apply_patch_updates_known_flags_and_ignores_unknown(self):
        plane = RuntimeControlPlane(flags=ControlPlaneFlags())
        out = plane.apply_patch(
            {
                "read_only": True,
                "allow_project_demo": False,
                "unknown_flag": True,
            }
        )
        self.assertTrue(out["ok"])
        self.assertIn("read_only", out["changed"])
        self.assertIn("allow_project_demo", out["changed"])
        self.assertIn("unknown_flag", out["ignored"])
        snapshot = plane.snapshot()
        self.assertTrue(snapshot["flags"]["read_only"])
        self.assertFalse(snapshot["flags"]["allow_project_demo"])


if __name__ == "__main__":
    unittest.main()
