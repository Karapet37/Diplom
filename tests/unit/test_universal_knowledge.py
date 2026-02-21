import tempfile
import unittest
from pathlib import Path

from src.living_system.core_engine import LivingSystemEngine


class UniversalKnowledgeTests(unittest.TestCase):
    def _engine(self) -> tuple[LivingSystemEngine, Path]:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        db_path = Path(tmpdir.name) / "living.db"
        workspace = Path(tmpdir.name) / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        engine = LivingSystemEngine(
            db_path=str(db_path),
            workspace_root=str(workspace),
            prompt_llm_fn=lambda _: "ok",
        )
        return engine, workspace

    def test_analyze_returns_structured_json(self):
        engine, _ = self._engine()
        out = engine.analyze_knowledge(
            text=(
                "A semantic graph requires concepts, relations, confidence scoring, "
                "and mechanisms for causal reasoning."
            ),
            user_id="u1",
            sources=[
                {"url": "https://source-a.example", "domain": "source-a.example"},
                {"url": "https://source-b.example", "domain": "source-b.example"},
            ],
            branch_id="main",
            apply_changes=False,
        )

        self.assertIn("concepts", out)
        self.assertIn("mechanisms", out)
        self.assertIn("relations", out)
        self.assertIn("confidence", out)
        self.assertIn("sql", out)
        self.assertIn("cypher", out)
        self.assertTrue(out["concepts"])
        self.assertTrue(out["sql"])
        self.assertTrue(out["cypher"])

    def test_apply_changes_persists_versions(self):
        engine, _ = self._engine()
        out = engine.analyze_knowledge(
            text="Knowledge graphs use nodes and edges. Edges depend on trust.",
            user_id="u2",
            sources=[
                {"url": "https://x.example", "domain": "x.example"},
                {"url": "https://y.example", "domain": "y.example"},
            ],
            branch_id="branch_u2",
            apply_changes=True,
        )

        self.assertTrue(out["new_concepts"])
        self.assertIn("versions", out)
        self.assertTrue(out["versions"])

        state = engine.store.graph_state(user_id="u2")
        self.assertGreater(len(state.get("nodes", [])), 0)

    def test_initialize_foundational_domains(self):
        engine, _ = self._engine()
        out = engine.initialize_foundational_knowledge(
            user_id="foundation_user",
            branch_id="foundation",
            apply_changes=False,
        )
        domain_nodes = [row for row in out["concepts"] if row.get("type") == "domain"]
        domain_names = {row.get("name") for row in domain_nodes}
        expected = {
            "Mathematics",
            "Physics",
            "Biology",
            "Computer Science",
            "Philosophy",
            "Psychology",
            "Sociology",
            "Theology",
            "Economics",
            "Linguistics",
        }
        self.assertEqual(domain_names, expected)

    def test_evaluate_detects_conflicts(self):
        engine, _ = self._engine()
        a = engine.store.upsert_node(
            {
                "node_id": "n-a",
                "user_id": "u3",
                "node_type": "concept",
                "display_name": "Model A",
                "confidence": 0.4,
                "properties": {"name": "Model A"},
                "metadata": {},
            }
        )
        b = engine.store.upsert_node(
            {
                "node_id": "n-b",
                "user_id": "u3",
                "node_type": "concept",
                "display_name": "Model B",
                "confidence": 0.42,
                "properties": {"name": "Model B"},
                "metadata": {},
            }
        )
        engine.store.upsert_edge(
            {
                "user_id": "u3",
                "from_node": a,
                "to_node": b,
                "relation_type": "causes",
                "confidence": 0.45,
                "weight": 0.44,
                "metadata": {},
            }
        )
        engine.store.upsert_edge(
            {
                "user_id": "u3",
                "from_node": a,
                "to_node": b,
                "relation_type": "prevents",
                "confidence": 0.43,
                "weight": 0.41,
                "metadata": {},
            }
        )

        report = engine.evaluate_knowledge_graph(user_id="u3")
        self.assertTrue(report["weak_nodes"])
        self.assertTrue(report["low_confidence_edges"])
        self.assertTrue(report["logical_conflicts"])

    def test_branch_and_merge_simulation(self):
        engine, _ = self._engine()
        engine.analyze_knowledge(
            text="Graph mechanisms support reasoning.",
            user_id="u4",
            sources=[{"url": "https://a.example"}, {"url": "https://b.example"}],
            branch_id="main",
            apply_changes=True,
        )
        base = engine.create_knowledge_branch(user_id="u4", branch_name="base")

        engine.analyze_knowledge(
            text="Counterfactual simulation enables hypothesis testing.",
            user_id="u4",
            sources=[{"url": "https://c.example"}, {"url": "https://d.example"}],
            branch_id="exp",
            apply_changes=True,
        )
        target = engine.create_knowledge_branch(user_id="u4", branch_name="target")

        merge = engine.merge_knowledge_branches(
            user_id="u4",
            base_snapshot_id=int(base["snapshot_id"]),
            target_snapshot_id=int(target["snapshot_id"]),
            apply_changes=False,
        )
        self.assertIn("simulation", merge)
        self.assertIn("nodes_to_add", merge)
        self.assertGreaterEqual(merge["simulation"]["predicted_nodes"], merge["simulation"]["base_nodes"])


if __name__ == "__main__":
    unittest.main()
