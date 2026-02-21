import importlib.util
import os
import unittest

from src.autonomous_graph.storage import Neo4jGraphDBAdapter


class Neo4jIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("RUN_NEO4J_INTEGRATION", "0").strip() == "1",
        "set RUN_NEO4J_INTEGRATION=1 to run Neo4j integration tests",
    )
    def test_neo4j_adapter_roundtrip(self):
        if importlib.util.find_spec("neo4j") is None:
            self.skipTest("neo4j package is not installed")

        uri = os.getenv("NEO4J_URI", "").strip()
        user = os.getenv("NEO4J_USER", "").strip()
        password = os.getenv("NEO4J_PASSWORD", "").strip()
        database = os.getenv("NEO4J_DATABASE", "neo4j").strip() or "neo4j"
        if not uri or not user or not password:
            self.skipTest("NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD are required")

        adapter = Neo4jGraphDBAdapter(
            uri=uri,
            user=user,
            password=password,
            database=database,
        )
        try:
            snapshot = {
                "nodes": {
                    "1": {
                        "type": "company",
                        "attributes": {"name": "Alpha"},
                        "state": {"influence": 0.7},
                    },
                    "2": {
                        "type": "company",
                        "attributes": {"name": "Beta"},
                        "state": {"influence": 0.5},
                    },
                },
                "edges": [
                    {
                        "from": 1,
                        "to": 2,
                        "relation_type": "influences",
                        "weight": 0.6,
                        "direction": "directed",
                        "logic_rule": "integration_test",
                    }
                ],
            }
            adapter.persist_snapshot(snapshot)
            loaded = adapter.load_snapshot()
            self.assertTrue(loaded.get("nodes"))
            self.assertTrue(loaded.get("edges"))
        finally:
            adapter.close()


if __name__ == "__main__":
    unittest.main()

