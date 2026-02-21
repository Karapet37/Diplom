import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from src.autonomous_graph.api import GraphAPI, build_graph_engine_from_env, run_demo_simulation
from src.autonomous_graph.core import Company, Edge, GraphEngine, Human
from src.autonomous_graph.storage import JsonGraphDBAdapter


class _Startup(Company):
    pass


class AutonomousGraphTests(unittest.TestCase):
    def test_dynamic_node_type_registration(self):
        engine = GraphEngine()
        engine.register_node_type("startup", _Startup)
        node = engine.create_node(
            "startup",
            name="Nova Labs",
            industry="AI",
            description="Prototype builder.",
        )
        self.assertIsInstance(node, _Startup)
        self.assertEqual(node.type, "startup")
        self.assertEqual(node.attributes.get("name"), "Nova Labs")

    def test_recursive_generation_creates_company_and_works_at_edge(self):
        engine = GraphEngine()
        human = engine.create_node(
            "human",
            first_name=None,
            last_name=None,
            bio="I build reliable pipelines and care about analysis quality.",
            employment_status=[
                {
                    "status": "data engineer",
                    "importance_score": 0.9,
                    "company_name": "Vector Dynamics",
                    "company_attributes": {"industry": "data infra"},
                }
            ],
            attributes={"first_name": "Aram"},
            state={"influence": 0.4},
        )
        self.assertIsInstance(human, Human)

        logs = engine.recursive_generation_operator(seed_node_ids=[human.id], max_depth=2)
        self.assertTrue(logs)
        companies = [
            node
            for node in engine.store.nodes.values()
            if node.type == "company" and node.attributes.get("name") == "Vector Dynamics"
        ]
        self.assertEqual(len(companies), 1)
        company = companies[0]
        works_edges = [
            edge
            for edge in engine.store.edges
            if edge.from_node == human.id
            and edge.to_node == company.id
            and edge.relation_type == "works_at"
        ]
        self.assertEqual(len(works_edges), 1)

    def test_inference_and_state_propagation(self):
        engine = GraphEngine()
        human = engine.create_node(
            "human",
            bio="Systemic thinker with high collaboration and product focus.",
            attributes={"first_name": "Lena"},
            state={"influence": 0.7, "trust": 0.5},
        )
        company = engine.create_node(
            "company",
            name="Core Vector",
            industry="AI tools",
            description="Decision support platform.",
            state={"influence": 0.1},
        )
        holding = engine.create_node(
            "company",
            name="Parent Group",
            industry="Holding",
            description="Portfolio operator.",
            state={"influence": 0.4},
        )
        engine.add_edge(
            Edge(
                from_node=human.id,
                to_node=company.id,
                relation_type="works_at",
                weight=0.9,
            )
        )
        engine.add_edge(
            Edge(
                from_node=company.id,
                to_node=holding.id,
                relation_type="part_of",
                weight=0.8,
            )
        )
        inferred = engine.logical_inference_operator()
        self.assertTrue(
            any(
                edge.from_node == human.id
                and edge.to_node == holding.id
                and edge.relation_type == "part_of"
                for edge in inferred
            )
        )

        state = engine.state_propagation_operator(steps=1, damping=0.0, activation="identity")
        self.assertIn(company.id, state)
        self.assertGreater(state[company.id].get("influence", 0.0), 0.0)

    def test_demo_simulation_returns_snapshot(self):
        result = run_demo_simulation()
        self.assertIn("snapshot", result)
        self.assertIn("nodes", result["snapshot"])
        self.assertIn("edges", result["snapshot"])
        self.assertTrue(result["snapshot"]["nodes"])

    def test_graph_event_listener_subscription(self):
        engine = GraphEngine()
        received: list[str] = []

        def on_event(event):
            received.append(str(event.event_type))

        engine.add_event_listener(on_event)
        engine.create_node("company", name="Listener Co")
        self.assertIn("node_added", received)

        count_before = len(received)
        engine.remove_event_listener(on_event)
        engine.create_node("company", name="Detached Listener Co")
        self.assertEqual(count_before, len(received))


class GraphApiTests(unittest.TestCase):
    def test_api_simulate_pipeline(self):
        api = GraphAPI()
        human = api.create_human(
            first_name="Nora",
            bio="I optimize systems and lead cross-functional projects.",
            employment=[
                {
                    "status": "founder",
                    "importance_score": 1.0,
                    "company_name": "Blue Orbit",
                }
            ],
            state={"influence": 0.6},
        )
        out = api.simulate(seed_node_ids=[human.id], recursive_depth=2, propagation_steps=2)
        self.assertIn("snapshot", out)
        self.assertIn("state", out)
        self.assertTrue(out["snapshot"]["edges"])
        self.assertTrue(api.get_events(event_type="simulation_started"))
        self.assertTrue(api.get_events(event_type="simulation_completed"))

    def test_event_feedback_updates_edge_weight(self):
        api = GraphAPI()
        a = api.create_company(name="A", description="alpha")
        b = api.create_company(name="B", description="beta")
        api.connect(a.id, b.id, relation_type="influences", weight=0.4)
        edge_events = api.get_events(event_type="edge_added")
        self.assertTrue(edge_events)
        event_id = edge_events[-1].id

        changed = api.reward_event(event_id, reward=0.8, learning_rate=0.5)
        self.assertTrue(changed)
        edge = api.engine._find_edge(  # noqa: SLF001
            from_node=a.id,
            to_node=b.id,
            relation_type="influences",
        )
        self.assertIsNotNone(edge)
        self.assertGreater(edge.weight, 0.4)

    def test_json_adapter_persist_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snapshot.json"
            adapter = JsonGraphDBAdapter(path)
            engine_a = GraphEngine(graph_adapter=adapter)
            api_a = GraphAPI(engine_a)
            h = api_a.create_human(
                first_name="Mila",
                bio="I build analytical systems.",
                employment=[
                    {
                        "status": "engineer",
                        "importance_score": 0.9,
                        "company_name": "Graph Labs",
                    }
                ],
            )
            api_a.simulate(seed_node_ids=[h.id], recursive_depth=2, propagation_steps=1)
            self.assertTrue(api_a.persist())
            self.assertTrue(path.exists())

            engine_b = GraphEngine(graph_adapter=adapter)
            api_b = GraphAPI(engine_b)
            loaded = api_b.load()
            self.assertTrue(loaded)
            snap = api_b.engine.snapshot()
            self.assertTrue(snap["nodes"])
            self.assertTrue(snap["edges"])

    def test_build_graph_engine_from_env_uses_json_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = str(Path(tmpdir) / "graph.json")
            with patch.dict(
                "os.environ",
                {
                    "AUTOGRAPH_STORAGE_ADAPTER": "json",
                    "AUTOGRAPH_JSON_ENABLE": "1",
                    "AUTOGRAPH_JSON_PATH": snapshot_path,
                    "AUTOGRAPH_NEO4J_ENABLE": "0",
                },
                clear=False,
            ):
                engine = build_graph_engine_from_env()
            self.assertIsNotNone(engine.graph_adapter)
            self.assertIsInstance(engine.graph_adapter, JsonGraphDBAdapter)


if __name__ == "__main__":
    unittest.main()
