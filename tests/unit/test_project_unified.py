import unittest

from src.web.graph_workspace import GraphWorkspaceService


class UnifiedProjectServiceTests(unittest.TestCase):
    def setUp(self):
        self.svc = GraphWorkspaceService(use_env_adapter=False)

    def test_project_overview(self):
        out = self.svc.project_overview()
        self.assertIn("project", out)
        self.assertIn("graph", out)
        self.assertIn("features", out)
        self.assertTrue(out["features"].get("graph_workspace"))

    def test_project_pipeline(self):
        out = self.svc.project_pipeline(
            {
                "text": "Knowledge graphs connect concepts via trusted relations.",
                "user_id": "u_pipeline",
                "display_name": "Pipeline User",
                "language": "en",
                "session_id": "s1",
                "branch_id": "main",
                "apply_knowledge_changes": False,
                "sources": [
                    {"url": "https://s1.example", "domain": "s1.example"},
                    {"url": "https://s2.example", "domain": "s2.example"},
                ],
            }
        )
        self.assertIn("living", out)
        self.assertIn("knowledge", out)
        self.assertIn("project_status", out)
        self.assertIn("concepts", out["knowledge"])
        self.assertIn("relations", out["knowledge"])

    def test_project_bootstrap_and_evaluate(self):
        boot = self.svc.project_bootstrap(
            {
                "user_id": "u_boot",
                "display_name": "Boot User",
                "language": "en",
                "branch_id": "foundation",
                "apply_changes": False,
                "seed_graph_demo": True,
            }
        )
        self.assertIn("foundation", boot)
        self.assertIn("project_status", boot)

        eval_out = self.svc.project_evaluate({"user_id": "u_boot"})
        self.assertIn("graph", eval_out)
        self.assertIn("living_health", eval_out)
        self.assertIn("knowledge_evaluation", eval_out)


if __name__ == "__main__":
    unittest.main()
