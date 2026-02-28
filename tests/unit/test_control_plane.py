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
                allow_client_introspection=False,
            )
        )
        allowed_introspect, reason_introspect = plane.allow_request(
            method="POST",
            path="/api/client/introspect",
        )
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

    def test_graph_writes_gate_blocks_hallucination_report(self):
        plane = RuntimeControlPlane(
            flags=ControlPlaneFlags(
                allow_graph_writes=False,
            )
        )
        allowed, reason = plane.allow_request(
            method="POST",
            path="/api/project/hallucination/report",
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "graph_writes_disabled")

    def test_prompt_execution_gate_blocks_archive_chat(self):
        plane = RuntimeControlPlane(
            flags=ControlPlaneFlags(
                allow_prompt_execution=False,
            )
        )
        allowed, reason = plane.allow_request(
            method="POST",
            path="/api/project/archive/chat",
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "prompt_execution_disabled")

    def test_prompt_execution_gate_blocks_graph_node_assist(self):
        plane = RuntimeControlPlane(
            flags=ControlPlaneFlags(
                allow_prompt_execution=False,
            )
        )
        allowed, reason = plane.allow_request(
            method="POST",
            path="/api/graph/node/assist",
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "prompt_execution_disabled")

    def test_prompt_execution_gate_blocks_graph_foundation_create(self):
        plane = RuntimeControlPlane(
            flags=ControlPlaneFlags(
                allow_prompt_execution=False,
            )
        )
        allowed, reason = plane.allow_request(
            method="POST",
            path="/api/graph/foundation/create",
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "prompt_execution_disabled")

    def test_prompt_execution_gate_blocks_graph_edge_assist(self):
        plane = RuntimeControlPlane(
            flags=ControlPlaneFlags(
                allow_prompt_execution=False,
            )
        )
        allowed, reason = plane.allow_request(
            method="POST",
            path="/api/graph/edge/assist",
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "prompt_execution_disabled")

    def test_graph_writes_gate_blocks_archive_review(self):
        plane = RuntimeControlPlane(
            flags=ControlPlaneFlags(
                allow_graph_writes=False,
            )
        )
        allowed, reason = plane.allow_request(
            method="POST",
            path="/api/project/archive/review",
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "graph_writes_disabled")

    def test_graph_writes_gate_blocks_personal_tree_ingest(self):
        plane = RuntimeControlPlane(
            flags=ControlPlaneFlags(
                allow_graph_writes=False,
            )
        )
        allowed, reason = plane.allow_request(
            method="POST",
            path="/api/project/personal-tree/ingest",
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "graph_writes_disabled")

    def test_graph_writes_gate_blocks_memory_namespace_apply(self):
        plane = RuntimeControlPlane(
            flags=ControlPlaneFlags(
                allow_graph_writes=False,
            )
        )
        allowed, reason = plane.allow_request(
            method="POST",
            path="/api/project/memory/namespace/apply",
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "graph_writes_disabled")

    def test_prompt_execution_gate_blocks_graph_rag_query(self):
        plane = RuntimeControlPlane(
            flags=ControlPlaneFlags(
                allow_prompt_execution=False,
            )
        )
        allowed, reason = plane.allow_request(
            method="POST",
            path="/api/project/graph-rag/query",
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "prompt_execution_disabled")

    def test_graph_writes_gate_blocks_wrapper_profile_update(self):
        plane = RuntimeControlPlane(
            flags=ControlPlaneFlags(
                allow_graph_writes=False,
            )
        )
        allowed, reason = plane.allow_request(
            method="POST",
            path="/api/project/wrapper/profile",
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "graph_writes_disabled")

    def test_prompt_execution_gate_blocks_wrapper_respond(self):
        plane = RuntimeControlPlane(
            flags=ControlPlaneFlags(
                allow_prompt_execution=False,
            )
        )
        allowed, reason = plane.allow_request(
            method="POST",
            path="/api/project/wrapper/respond",
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "prompt_execution_disabled")

    def test_graph_writes_gate_blocks_integration_invoke(self):
        plane = RuntimeControlPlane(
            flags=ControlPlaneFlags(
                allow_graph_writes=False,
            )
        )
        allowed, reason = plane.allow_request(
            method="POST",
            path="/api/integration/layer/invoke",
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "graph_writes_disabled")

    def test_prompt_execution_gate_blocks_integration_invoke(self):
        plane = RuntimeControlPlane(
            flags=ControlPlaneFlags(
                allow_prompt_execution=False,
            )
        )
        allowed, reason = plane.allow_request(
            method="POST",
            path="/api/integration/layer/invoke",
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
