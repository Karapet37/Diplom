import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from src.web.graph_workspace import GraphWorkspaceService


class GraphWorkspaceServiceTests(unittest.TestCase):
    def setUp(self):
        self.svc = GraphWorkspaceService(use_env_adapter=False, enable_living_system=False)

    def test_create_node_and_edge_and_simulate(self):
        human = self.svc.create_node(
            {
                "node_type": "human",
                "first_name": "Aram",
                "bio": "I build analytical graph systems.",
                "employment": [
                    {
                        "status": "engineer",
                        "importance_score": 0.9,
                        "company_name": "Vector Dynamics",
                    }
                ],
                "state": {"influence": 0.6},
            }
        )
        self.assertIn("node", human)
        self.assertGreaterEqual(human["metrics"]["node_count"], 1)

        company = self.svc.create_node(
            {
                "node_type": "company",
                "name": "North Capital",
                "industry": "Investment",
                "description": "Investor",
                "state": {"influence": 0.7},
            }
        )
        from_id = company["node"]["id"]

        target_id = human["node"]["id"]

        edge = self.svc.create_edge(
            {
                "from_node": from_id,
                "to_node": target_id,
                "relation_type": "influences",
                "weight": 0.75,
            }
        )
        self.assertIn("edge", edge)
        self.assertGreaterEqual(edge["metrics"]["edge_count"], 1)

        result = self.svc.simulate(
            {
                "recursive_depth": 2,
                "propagation_steps": 2,
                "damping": 0.1,
                "activation": "tanh",
                "infer_rounds": 1,
            }
        )
        self.assertIn("result", result)
        self.assertIn("snapshot", result)
        self.assertIn("metrics", result)

    def test_event_feedback_and_relation_reinforcement(self):
        a = self.svc.create_node({"node_type": "company", "name": "A"})
        b = self.svc.create_node({"node_type": "company", "name": "B"})

        self.svc.create_edge(
            {
                "from_node": a["node"]["id"],
                "to_node": b["node"]["id"],
                "relation_type": "influences",
                "weight": 0.4,
            }
        )
        events = self.svc.list_events(limit=100)
        edge_events = [item for item in events if item["event_type"] == "edge_added"]
        self.assertTrue(edge_events)

        event_id = edge_events[-1]["id"]
        feedback = self.svc.reward_event(
            {
                "event_id": event_id,
                "reward": 0.8,
                "learning_rate": 0.2,
            }
        )
        self.assertIn("changed", feedback)

        batch = self.svc.reinforce_relation(
            {
                "relation_type": "influences",
                "reward": 0.2,
                "learning_rate": 0.1,
            }
        )
        self.assertGreaterEqual(batch["updated"], 1)

    def test_seed_demo_and_clear(self):
        seeded = self.svc.seed_demo()
        self.assertGreaterEqual(seeded["metrics"]["node_count"], 2)
        self.assertGreaterEqual(seeded["metrics"]["edge_count"], 1)

        cleared = self.svc.clear()
        self.assertEqual(cleared["metrics"]["node_count"], 0)
        self.assertEqual(cleared["metrics"]["edge_count"], 0)

    def test_foundation_concepts_available(self):
        snap = self.svc.snapshot_payload()["snapshot"]
        names = {
            str(node.get("attributes", {}).get("name", "")).strip().casefold()
            for node in snap.get("nodes", [])
        }
        self.assertIn("mathematics".casefold(), names)
        self.assertIn("music".casefold(), names)
        self.assertIn("computer science".casefold(), names)

        linguistics_nodes = [
            row
            for row in snap.get("nodes", [])
            if str(row.get("type")) == "domain"
            and str(row.get("attributes", {}).get("name", "")).strip().casefold() == "linguistics"
        ]
        self.assertTrue(linguistics_nodes)
        linguistics = linguistics_nodes[0].get("attributes", {})
        self.assertIn("history", str(linguistics.get("description", "")).casefold())
        self.assertTrue(linguistics.get("history_intro"))
        self.assertTrue(linguistics.get("connections"))
        self.assertTrue(linguistics.get("usage"))

        philosophy_nodes = [
            row
            for row in snap.get("nodes", [])
            if str(row.get("type")) == "domain"
            and str(row.get("attributes", {}).get("name", "")).strip().casefold() == "philosophy"
        ]
        self.assertTrue(philosophy_nodes)
        philosophy = philosophy_nodes[0].get("attributes", {})
        self.assertIn("knowledge", str(philosophy.get("summary", "")).casefold())
        self.assertTrue(str(philosophy.get("details", "")).strip())
        self.assertIn("Философия", list(philosophy.get("aliases", []) or []))
        localized = philosophy.get("localized_labels", {}) if isinstance(philosophy.get("localized_labels"), dict) else {}
        self.assertEqual(str(localized.get("ru", "")), "Философия")

        ethics_nodes = [
            row
            for row in snap.get("nodes", [])
            if str(row.get("type")) == "concept"
            and str(row.get("attributes", {}).get("name", "")).strip().casefold() == "ethics"
        ]
        self.assertTrue(ethics_nodes)
        ethics = ethics_nodes[0].get("attributes", {})
        self.assertIn("foundational concept", str(ethics.get("summary", "")).casefold())
        self.assertTrue(str(ethics.get("details", "")).strip())
        self.assertIn("Этика", list(ethics.get("aliases", []) or []))
        localized_ethics = ethics.get("localized_labels", {}) if isinstance(ethics.get("localized_labels"), dict) else {}
        self.assertEqual(str(localized_ethics.get("hy", "")), "Էթիկա")

        domain_nodes = [
            row
            for row in snap.get("nodes", [])
            if str(row.get("type", "")).strip().casefold() == "domain"
        ]
        self.assertGreaterEqual(len(domain_nodes), 10)
        for row in domain_nodes:
            attrs = row.get("attributes", {}) or {}
            self.assertTrue(str(attrs.get("description", "")).strip())
            self.assertTrue(str(attrs.get("history_intro", "")).strip())
            self.assertTrue(list(attrs.get("connections", []) or []))
            self.assertTrue(list(attrs.get("usage", []) or []))

    def test_graph_edit_crud(self):
        left = self.svc.create_node({"node_type": "company", "name": "Edit Left"})
        right = self.svc.create_node({"node_type": "company", "name": "Edit Right"})
        self.svc.create_edge(
            {
                "from_node": left["node"]["id"],
                "to_node": right["node"]["id"],
                "relation_type": "influences",
                "weight": 0.31,
            }
        )
        updated = self.svc.update_node(
            {
                "node_id": left["node"]["id"],
                "attributes": {"name": "Edit Left Updated", "risk_note": "changed"},
                "state": {"influence": 0.93},
            }
        )
        self.assertEqual(updated["node"]["attributes"].get("name"), "Edit Left Updated")

        edge_updated = self.svc.update_edge(
            {
                "from_node": left["node"]["id"],
                "to_node": right["node"]["id"],
                "relation_type": "influences",
                "direction": "directed",
                "weight": 0.77,
                "logic_rule": "manual_edit",
                "metadata": {"note": "updated in test"},
            }
        )
        self.assertAlmostEqual(edge_updated["edge"]["weight"], 0.77, places=3)
        self.assertEqual(edge_updated["edge"]["logic_rule"], "manual_edit")

        deleted_edge = self.svc.delete_edge(
            {
                "from_node": left["node"]["id"],
                "to_node": right["node"]["id"],
                "relation_type": "influences",
                "direction": "directed",
            }
        )
        self.assertTrue(deleted_edge["deleted"])

        deleted_node = self.svc.delete_node({"node_id": left["node"]["id"]})
        self.assertTrue(deleted_node["deleted"])

    def test_graph_workspace_autoloads_and_autopersists_with_json_adapter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = str(Path(tmpdir) / "workspace_graph.json")
            with patch.dict(
                "os.environ",
                {
                    "AUTOGRAPH_STORAGE_ADAPTER": "json",
                    "AUTOGRAPH_JSON_ENABLE": "1",
                    "AUTOGRAPH_JSON_PATH": snapshot_path,
                    "AUTOGRAPH_NEO4J_ENABLE": "0",
                    "AUTOGRAPH_AUTO_PERSIST_ON_WRITE": "1",
                    "AUTOGRAPH_AUTO_LOAD_ON_START": "1",
                },
                clear=False,
            ):
                writer = GraphWorkspaceService(use_env_adapter=True, enable_living_system=False)
                created = writer.create_node(
                    {
                        "node_type": "generic",
                        "attributes": {
                            "user_id": "persist_user",
                            "name": "Persisted node",
                            "summary": "This node must survive a fresh service instance.",
                        },
                    }
                )
                self.assertTrue(created["persisted"])

                reader = GraphWorkspaceService(use_env_adapter=True, enable_living_system=False)
                snapshot = reader.snapshot_payload()["snapshot"]

        node_names = {
            str((row.get("attributes") or {}).get("name", "") or "")
            for row in snapshot.get("nodes", [])
        }
        self.assertIn("Persisted node", node_names)

    def test_watch_demo_and_client_introspection(self):
        demo = self.svc.watch_demo({"persona_name": "Alexa", "use_llm": False, "reset_graph": True})
        self.assertIn("demo", demo)
        self.assertIn("snapshot", demo)
        self.assertGreaterEqual(demo["metrics"]["node_count"], 5)
        self.assertIn("Меня зовут Alexa", str(demo["demo"]["narrative"]))

        demo_en = self.svc.watch_demo(
            {"persona_name": "Alexa", "use_llm": False, "reset_graph": True, "language": "en"}
        )
        self.assertIn("My name is Alexa", str(demo_en["demo"]["narrative"]))

        out = self.svc.capture_client_profile(
            {
                "session_id": "sess_test",
                "user_id": "u_test",
                "client": {
                    "user_agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0",
                    "platform": "Linux x86_64",
                    "language": "ru-RU",
                    "timezone": "Asia/Yerevan",
                    "screen": {"width": 1920, "height": 1080},
                    "viewport": {"width": 1280, "height": 860},
                },
            },
            request_headers={
                "user-agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0",
                "x-forwarded-for": "198.51.100.10, 10.0.0.1",
                "x-real-ip": "198.51.100.10",
            },
            request_ip="198.51.100.10",
        )
        self.assertIn("profile", out)
        self.assertIn("semantic_binding", out)
        self.assertEqual(out["profile"]["network"]["ip"]["ip"], "198.51.100.10")
        self.assertTrue(out["profile"]["network"]["vpn_proxy_suspected"])

        with_living = GraphWorkspaceService(use_env_adapter=False, enable_living_system=True)
        schema = with_living.project_db_schema()
        self.assertTrue(schema.get("tables"))
        table_names = {row.get("name") for row in schema["tables"]}
        self.assertIn("nodes", table_names)
        self.assertIn("edges", table_names)

    def test_project_daily_mode(self):
        out = self.svc.project_daily_mode(
            {
                "text": (
                    "Сегодня цель: закрыть задачу по мониторингу и улучшить документацию. "
                    "Проблема: сильно устал и отвлекался. "
                    "Сделал: завершил API и тесты."
                ),
                "user_id": "daily_user",
                "display_name": "Daily User",
                "language": "ru",
                "recommendation_count": 4,
                "run_knowledge_analysis": False,
            }
        )
        self.assertIn("journal_entry", out)
        self.assertIn("signals", out)
        self.assertIn("recommendations", out)
        self.assertIn("improvement_scores", out)
        self.assertTrue(out["signals"]["goals"])
        self.assertTrue(out["signals"]["problems"])
        self.assertGreaterEqual(len(out["recommendations"]), 3)
        self.assertLessEqual(len(out["recommendations"]), 5)
        self.assertIn("metrics", out["improvement_scores"])
        self.assertIn("overall", out["improvement_scores"])
        self.assertIn("profile_update_json", out)
        self.assertIn("dimensions", out["profile_update_json"])

    def test_project_model_advisors_payload(self):
        out = self.svc.project_model_advisors()
        self.assertIn("advisors", out)
        self.assertIn("prompts", out)
        advisors = out["advisors"]
        self.assertIn("advisors", advisors)
        roles = {str(item.get("role", "")) for item in advisors.get("advisors", [])}
        self.assertIn("translator", roles)
        self.assertIn("coder_architect", roles)

    def test_human_profile_autofill_from_structured_text(self):
        human = self.svc.create_node(
            {
                "node_type": "human",
                "bio": "имя: Логан; фамилия: Манделла; возраст: 55; рост: 180см; вес: 69кг",
                "profile_text": "предпочтения: джаз, исторические фильмы\nценности: свобода, бизнес",
                "employment_text": "founder @ Vector Dynamics",
            }
        )
        attrs = human["node"]["attributes"]
        self.assertEqual(attrs.get("first_name"), "Логан")
        self.assertEqual(attrs.get("last_name"), "Манделла")
        self.assertEqual(attrs.get("age"), 55)
        self.assertEqual(attrs.get("height_cm"), 180.0)
        self.assertEqual(attrs.get("weight_kg"), 69.0)

        preferences = [str(item).lower() for item in attrs.get("preferences", [])]
        self.assertIn("джаз", preferences)
        self.assertTrue(attrs.get("employment_status"))
        self.assertEqual(attrs["employment_status"][0].get("company_name"), "Vector Dynamics")

    def test_human_profile_autofill_from_natural_text(self):
        human = self.svc.create_node(
            {
                "node_type": "human",
                "bio": "Меня зовут Армен. Я работаю как инженер в Data Forge и веду проекты по аналитике.",
            }
        )
        attrs = human["node"]["attributes"]
        self.assertEqual(attrs.get("first_name"), "Армен")
        self.assertTrue(attrs.get("employment_status"))
        self.assertEqual(attrs["employment_status"][0].get("company_name"), "Data Forge")

    def test_infer_profile_from_llm_and_reuse_language_nodes(self):
        def fake_llm(prompt: str) -> str:
            if "Second user" in prompt:
                name = "Lena"
                skill = "guitar"
            else:
                name = "Aram"
                skill = "programming"
            payload = {
                "entity": {
                    "type": "human",
                    "first_name": name,
                    "last_name": "Sargsyan",
                    "description": "Engineer and builder",
                },
                "personality": {
                    "traits": ["curious"],
                    "values": ["freedom"],
                    "fears": ["market crash"],
                    "goals": ["ship product"],
                    "preferences": ["jazz"],
                },
                "employment": [
                    {
                        "status": "engineer",
                        "company_name": "Vector Dynamics",
                        "importance_score": 0.8,
                    }
                ],
                "skills": [
                    {
                        "name": skill,
                        "category": "tech",
                        "level": "advanced",
                        "description": "core skill",
                    }
                ],
                "languages": [
                    {
                        "name": "English",
                        "code": "en",
                        "proficiency": "B2",
                        "family": "Indo-European",
                    }
                ],
                "capabilities": [],
                "links": [],
            }
            return "analysis section\n```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"

        svc = GraphWorkspaceService(
            use_env_adapter=False,
            profile_llm_fn=fake_llm,
        )
        out_first = svc.infer_profile_from_text(
            {
                "text": "First user narrative",
                "entity_type_hint": "human",
                "create_graph": True,
            }
        )
        out_second = svc.infer_profile_from_text(
            {
                "text": "Second user narrative",
                "entity_type_hint": "human",
                "create_graph": True,
            }
        )

        self.assertIn("profile_json", out_first)
        self.assertTrue(str(out_first.get("profile_json_file", "")).strip())
        self.assertTrue(Path(out_first["profile_json_file"]).exists())
        self.assertIn("snapshot", out_second)
        nodes = out_second["snapshot"]["nodes"]
        language_nodes = [
            row for row in nodes
            if row.get("type") == "language" and str(row.get("attributes", {}).get("code")) == "en"
        ]
        self.assertEqual(len(language_nodes), 1)
        skill_edges = [
            row for row in out_second["snapshot"]["edges"]
            if row.get("relation_type") == "has_skill"
        ]
        self.assertGreaterEqual(len(skill_edges), 2)

    def test_infer_profile_person_concepts_schema(self):
        def fake_llm(_: str) -> str:
            payload = {
                "person": {
                    "id": 101,
                    "first_name": "Aram",
                    "last_name": "Sargsyan",
                    "bio": "Engineer, loves jazz and fears market crash.",
                    "birth_date": "1990-01-01",
                },
                "concept_relations": [
                    {
                        "concept_id": 202,
                        "concept_name": "музыка",
                        "relation_type": "likes",
                        "embedding_vector": [0.11, 0.33, -0.02, 0.57],
                        "confidence": 0.91,
                        "details": {"additional_info": "Listens daily"},
                    },
                    {
                        "concept_id": 0,
                        "concept_name": "market crash",
                        "new_concept": True,
                        "relation_type": "fears",
                        "embedding_vector": [0.72, -0.41, 0.08, 0.19],
                        "confidence": 0.88,
                        "details": {"additional_info": "High anxiety trigger"},
                    },
                ],
            }
            return json.dumps(payload, ensure_ascii=False)

        svc = GraphWorkspaceService(use_env_adapter=False, profile_llm_fn=fake_llm)
        out = svc.infer_profile_from_text(
            {
                "text": "Narrative text",
                "entity_type_hint": "human",
                "create_graph": True,
                "save_json": False,
            }
        )

        profile_json = out["profile_json"]
        self.assertIn("person", profile_json)
        self.assertEqual(profile_json["person"]["first_name"], "Aram")
        self.assertEqual(len(profile_json.get("concept_relations", [])), 2)

        nodes = out["snapshot"]["nodes"]
        concept_nodes = [row for row in nodes if row.get("type") == "concept"]
        self.assertGreaterEqual(len(concept_nodes), 2)

        edges = out["snapshot"]["edges"]
        concept_edges = [row for row in edges if row.get("relation_type") in {"likes", "fears"}]
        self.assertGreaterEqual(len(concept_edges), 2)
        self.assertTrue(any("metadata" in row for row in concept_edges))

    def test_project_user_graph_update_dimensions(self):
        out = self.svc.project_user_graph_update(
            {
                "user_id": "u_dims",
                "display_name": "User Dims",
                "fears": ["failure", "public speaking"],
                "desires": ["freedom"],
                "goals": ["build product"],
                "principles": ["honesty"],
                "opportunities": ["new market"],
                "abilities": ["teaching"],
                "access": ["api", "internal docs"],
                "knowledge": ["python", "graph theory"],
                "assets": ["laptop", "domain knowledge"],
                "personalization": {
                    "response_style": "concise",
                    "reasoning_depth": "deep",
                    "risk_tolerance": "low",
                    "tone": "direct",
                    "focus_goals": ["ship product faster"],
                    "domain_focus": ["architecture"],
                    "llm_roles": {
                        "proposer": "creative",
                        "critic": "analyst",
                        "judge": "planner",
                    },
                },
                "feedback_items": [
                    {"message": "Short practical plans work best", "score": 0.9, "decision": "accept"},
                    {"message": "Avoid generic long explanations", "score": 0.2, "decision": "reject"},
                ],
            }
        )
        self.assertIn("user_profile", out)
        self.assertIn("binding", out)
        dims = out["user_profile"]["dimensions"]
        self.assertIn("fears", dims)
        self.assertIn("assets", dims)
        snapshot = out["snapshot"]
        relation_types = {row.get("relation_type") for row in snapshot.get("edges", [])}
        self.assertIn("has_fear", relation_types)
        self.assertIn("owns_asset", relation_types)
        self.assertIn("profile_update_json", out)
        self.assertTrue(out.get("personalization_applied"))
        self.assertEqual(int(out.get("feedback_summary", {}).get("new_items", 0)), 2)
        self.assertIn("personalization", out["profile_update_json"])
        self.assertIn("summary", out)
        self.assertGreaterEqual(int(out["summary"].get("dimension_count", 0)), 2)
        self.assertIn("execution", out)
        self.assertEqual(str(out["execution"].get("action", "")), "user_graph_update")
        self.assertIn("input_extraction", out["execution"])

    def test_project_user_graph_update_from_text_and_client_profile(self):
        out = self.svc.project_user_graph_update(
            {
                "user_id": "u_text",
                "display_name": "User Text",
                "text": (
                    "Меня зовут Алекс. В детстве занимался математикой и музыкой. "
                    "Сейчас я backend инженер и хочу построить устойчивую платформу."
                ),
                "language": "ru",
                "session_id": "sess_text",
                "include_client_profile": True,
                "client": {
                    "user_agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0",
                    "platform": "Linux x86_64",
                    "language": "ru-RU",
                },
            },
            request_headers={
                "user-agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0",
                "x-forwarded-for": "198.51.100.20, 10.0.0.1",
            },
            request_ip="198.51.100.20",
        )
        self.assertIn("profile_update_json", out)
        profile_json = out["profile_update_json"]
        self.assertIn("profile", profile_json)
        self.assertTrue(str(profile_json["profile"].get("history", "")).strip())
        self.assertIn("client_semantic_binding", out)
        self.assertGreater(int(out["client_semantic_binding"].get("session_node_id", 0)), 0)

    def test_project_user_graph_update_records_qwen_input_capture_and_mistral_monitor(self):
        def fake_model_resolver(model_path: str):
            token = str(model_path or "").strip()
            if "qwen2.5-7b-instruct" in token:
                def _run(_: str) -> str:
                    return json.dumps(
                        {
                            "summary": "Captured structured profile facts.",
                            "archive_updates": [
                                {
                                    "entity": "user_graph_update",
                                    "field": "goal",
                                    "operation": "append",
                                    "value": "Build a cleaner graph UX.",
                                    "reason": "Explicit intent from user text.",
                                    "source": "input_capture_test",
                                    "confidence": 0.83,
                                    "tags": ["profile", "goal"],
                                }
                            ],
                        },
                        ensure_ascii=False,
                    )

                return _run
            if "mistral-7b-instruct-v0.3" in token:
                def _run(prompt: str) -> str:
                    match = re.search(r'"node_id":\s*(\d+)', prompt)
                    node_id = int(match.group(1)) if match else 0
                    return json.dumps(
                        {
                            "node_patches": (
                                [
                                    {
                                        "node_id": node_id,
                                        "summary": "Monitor refreshed profile summary.",
                                        "confidence": 0.74,
                                        "reason": "Keep root profile node normalized.",
                                    }
                                ]
                                if node_id > 0
                                else []
                            ),
                            "edge_patches": [],
                        },
                        ensure_ascii=False,
                    )

                return _run
            return None

        svc = GraphWorkspaceService(
            use_env_adapter=False,
            enable_living_system=False,
            model_llm_resolver=fake_model_resolver,
        )
        out = svc.project_user_graph_update(
            {
                "user_id": "u_input_chain",
                "display_name": "Input Chain",
                "text": "I need a cleaner graph UI and a more reliable workflow.",
                "session_id": "sess_input_chain",
                "use_llm_profile": False,
                "include_client_profile": False,
            }
        )
        self.assertTrue(out["input_extraction"]["enabled"])
        self.assertGreaterEqual(int(out["input_extraction"]["updates_count"]), 1)
        self.assertTrue(out["input_extraction"]["graph_binding"]["attached"])
        self.assertTrue(out["graph_monitor"]["attached"])
        self.assertGreaterEqual(len(out["graph_monitor"].get("node_patches", [])), 1)
        self.assertIn("edge_patches", out["graph_monitor"])
        node_types = {row.get("type") for row in out["snapshot"]["nodes"]}
        self.assertIn("llm_archive_update_session", node_types)
        self.assertIn("graph_monitor_session", node_types)

    def test_project_llm_debate_with_role_outputs(self):
        def fake_role_resolver(role: str):
            role_key = str(role or "").strip().lower()

            def _run(prompt: str) -> str:
                if role_key == "creative":
                    return json.dumps(
                        {
                            "hypotheses": [
                                {
                                    "title": "Graph-first loop",
                                    "claim": "Build a loop that writes reasoning traces directly to graph edges.",
                                    "rationale": "Improves explainability for each decision.",
                                    "confidence": 0.79,
                                },
                                {
                                    "title": "UI stream-first",
                                    "claim": "Prioritize live graph streaming before expanding analytics breadth.",
                                    "rationale": "Faster feedback for users.",
                                    "confidence": 0.74,
                                },
                            ]
                        },
                        ensure_ascii=False,
                    )
                if role_key == "analyst":
                    return json.dumps(
                        {
                            "issues": ["Potential complexity in edge metadata growth."],
                            "contradictions": [],
                            "risk_score": 0.33,
                            "confidence": 0.77,
                            "recommendation": "accept_with_checks",
                        },
                        ensure_ascii=False,
                    )
                if role_key == "planner":
                    return json.dumps(
                        {
                            "selected_index": 1,
                            "decision": "Build graph-first loop with strict metadata schema.",
                            "consensus": "Hypothesis 1 gives better long-term observability with manageable risk.",
                            "confidence": 0.81,
                            "ranking": [{"index": 1, "score": 0.81}, {"index": 2, "score": 0.74}],
                        },
                        ensure_ascii=False,
                    )
                return ""

            return _run

        svc = GraphWorkspaceService(use_env_adapter=False, role_llm_resolver=fake_role_resolver)
        out = svc.project_llm_debate(
            {
                "topic": "Improve reasoning transparency for runtime decisions",
                "hypothesis_count": 2,
                "attach_to_graph": True,
                "proposer_role": "creative",
                "critic_role": "analyst",
                "judge_role": "planner",
                "personalization": {
                    "response_style": "balanced",
                    "reasoning_depth": "deep",
                    "risk_tolerance": "low",
                    "focus_goals": ["observable decisions", "stable quality"],
                    "llm_roles": {
                        "proposer": "creative",
                        "critic": "analyst",
                        "judge": "planner",
                    },
                },
                "feedback_items": [{"message": "Prefer concrete actions", "score": 0.8, "decision": "accept"}],
            }
        )
        self.assertIn("hypotheses", out)
        self.assertIn("critiques", out)
        self.assertIn("verdict", out)
        self.assertEqual(len(out["hypotheses"]), 2)
        self.assertIn("personalization", out)
        self.assertEqual(out["feedback_summary"]["items"], 1)
        self.assertTrue(out["graph_binding"]["attached"])
        node_types = {row.get("type") for row in out["snapshot"]["nodes"]}
        self.assertIn("llm_debate_session", node_types)
        self.assertIn("llm_hypothesis", node_types)
        self.assertIn("llm_judgement", node_types)

    def test_hallucination_hunter_report_check_and_debate_guard(self):
        report = self.svc.project_hallucination_report(
            {
                "user_id": "u_hall",
                "session_id": "s_hall_1",
                "prompt": "What is the capital of Armenia?",
                "llm_answer": "The capital is Tbilisi.",
                "correct_answer": "The capital of Armenia is Yerevan.",
                "source": "trusted_geo_dataset_v1",
                "tags": ["geography", "capital"],
                "severity": "high",
                "confidence": 0.93,
            }
        )
        self.assertIn("hallucination_branch", report)
        self.assertIn("case", report)
        self.assertTrue(report["case"]["created"])
        self.assertGreaterEqual(report["case"]["occurrence_count"], 1)

        check = self.svc.project_hallucination_check(
            {
                "user_id": "u_hall",
                "prompt": "Tell me the capital of Armenia",
                "llm_answer": "It is Tbilisi",
                "top_k": 3,
            }
        )
        self.assertTrue(check["has_known_hallucination_risk"])
        self.assertGreaterEqual(check["match_count"], 1)
        top = check["matches"][0]
        self.assertIn("Yerevan", top.get("correct_answer", ""))

        debate = self.svc.project_llm_debate(
            {
                "user_id": "u_hall",
                "topic": "Prepare a short answer about the capital of Armenia",
                "hypothesis_count": 2,
                "attach_to_graph": False,
            }
        )
        self.assertIn("hallucination_guard", debate)
        self.assertGreaterEqual(int(debate["hallucination_guard"]["hits"]), 1)

    def test_project_archive_verified_chat_with_explicit_model_and_verification(self):
        def fake_model_resolver(model_path: str):
            token = str(model_path or "").strip()
            if "h2o-danube3-4b-chat" not in token:
                return None

            def _run(_: str) -> str:
                return json.dumps(
                    {
                        "summary": "Add validated planning note and budget status.",
                        "archive_updates": [
                            {
                                "entity": "project_plan",
                                "field": "next_step",
                                "operation": "upsert",
                                "value": "Run benchmark on 10k nodes before release.",
                                "reason": "Keeps performance risks measurable.",
                                "source": "internal_test_protocol_v2",
                                "confidence": 0.83,
                                "tags": ["performance", "release"],
                            },
                            {
                                "entity": "project_budget",
                                "field": "status",
                                "operation": "upsert",
                                "value": "stable",
                                "reason": "Current costs are within plan.",
                                "source": "finance_snapshot_2026_02",
                                "confidence": 0.79,
                                "tags": ["budget"],
                            },
                        ],
                    },
                    ensure_ascii=False,
                )

            return _run

        svc = GraphWorkspaceService(
            use_env_adapter=False,
            enable_living_system=False,
            model_llm_resolver=fake_model_resolver,
        )
        out = svc.project_archive_verified_chat(
            {
                "user_id": "u_archive",
                "session_id": "s_archive_1",
                "message": "Update archive with validated next step and budget status.",
                "context": "Use practical, low-risk updates only.",
                "model_path": "models/gguf/textGen/h2o-danube3-4b-chat-Q5_K_M.gguf",
                "model_role": "general",
                "apply_to_graph": True,
                "verification_mode": "strict",
                "top_k": 3,
            }
        )
        self.assertIn("archive_updates", out)
        self.assertEqual(len(out["archive_updates"]), 2)
        self.assertTrue(out["verification"]["verified"])
        self.assertEqual(out["verification"]["issue_count"], 0)
        self.assertTrue(str(out.get("assistant_reply", "")).strip())
        self.assertEqual(out["model"]["resolution_mode"], "explicit_model_path")
        self.assertTrue(out["graph_binding"]["attached"])
        self.assertIn("triage", out)
        self.assertTrue(out["triage"]["enabled"])
        node_types = {row.get("type") for row in out["snapshot"]["nodes"]}
        self.assertIn("llm_archive_update_branch", node_types)
        self.assertIn("llm_archive_update_session", node_types)
        self.assertIn("llm_archive_update_record", node_types)

    @patch("src.web.graph_workspace.collect_web_context")
    def test_project_chat_graph_uses_dual_models_and_attaches_updates(self, mock_collect_web_context):
        mock_collect_web_context.return_value = {
            "enabled": True,
            "terms": ["coffee"],
            "snippets": [
                {
                    "source": "wikipedia",
                    "title": "Coffee",
                    "url": "https://en.wikipedia.org/wiki/Coffee",
                    "snippet": "Coffee is a brewed drink prepared from roasted coffee beans.",
                }
            ],
            "warning": "",
        }
        prompts: dict[str, str] = {}

        def fake_model_resolver(model_path: str):
            token = str(model_path or "").strip()

            def _run(prompt: str) -> str:
                prompts[token] = prompt
                if "qwen2.5-7b-instruct" in token:
                    return json.dumps(
                        {
                            "summary": "Useful coffee facts captured.",
                            "archive_updates": [
                                {
                                    "entity": "coffee",
                                    "field": "definition",
                                    "operation": "upsert",
                                    "value": "A brewed drink made from roasted coffee beans.",
                                    "reason": "Stable factual definition extracted from the conversation.",
                                    "source": "wikipedia:Coffee",
                                    "confidence": 0.86,
                                    "tags": ["coffee", "definition"],
                                }
                            ],
                        },
                        ensure_ascii=False,
                    )
                return "Coffee is best introduced as a practical topic with clear definitions first."

            return _run

        svc = GraphWorkspaceService(
            use_env_adapter=False,
            enable_living_system=False,
            model_llm_resolver=fake_model_resolver,
        )
        out = svc.project_chat_graph(
            {
                "user_id": "u_chat_graph",
                "session_id": "s_chat_graph_1",
                "message": "Explain coffee clearly and store only useful facts.",
                "context": "Keep it concise.",
                "chat_model_path": "models/gguf/textGen/mistral-7b-instruct-v0.3-q4_k_m.gguf",
                "parser_backend": "local",
                "parser_model_path": "models/gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                "use_internet": True,
                "apply_to_graph": True,
            }
        )
        self.assertTrue(out["ok"])
        self.assertIn("Coffee is best introduced", out["assistant_reply"])
        self.assertEqual(len(out["archive_updates"]), 1)
        self.assertEqual(out["archive_updates"][0]["entity"], "coffee")
        self.assertTrue(out["graph_binding"]["attached"])
        self.assertTrue(out["graph_diff"]["attached"])
        self.assertGreaterEqual(out["graph_diff"]["node_count"], 2)
        self.assertGreaterEqual(out["graph_diff"]["edge_count"], 1)
        edge_types = {row.get("type") for row in out["graph_diff"]["edges"]}
        self.assertIn("proposes_archive_update", edge_types)
        self.assertEqual(out["parser_model"]["backend"], "local")
        self.assertEqual(len(out["web_context"]["snippets"]), 1)
        self.assertIn("Internet snippets", prompts["models/gguf/textGen/mistral-7b-instruct-v0.3-q4_k_m.gguf"])
        self.assertIn("Assistant reply", prompts["models/gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"])

    def test_project_mode_policy_resolve_builds_weighted_context(self):
        mode_node = self.svc.api.engine.create_node(
            "context_mode",
            attributes={
                "user_id": "u_mode",
                "name": "Coffee Planner",
                "domain": "coffee",
                "prompt_guardrails": "Stay practical and compare roast and brewing choices.",
                "protected_memory": "Prefer balanced acidity over burnt bitterness.",
                "llm_role": "planner",
                "updated_at": 123.0,
            },
            state={"context_weight": 1.34},
        )
        focus_node = self.svc.api.engine.create_node(
            "context_focus",
            attributes={
                "user_id": "u_mode",
                "name": "Pour Over",
                "summary": "Clean cup, clear acidity, and high control over extraction.",
            },
            state={"importance": 0.9},
        )
        self.svc._connect_nodes(  # noqa: SLF001
            from_node=int(mode_node.id),
            to_node=int(focus_node.id),
            relation_type="mode_focus",
            weight=0.91,
            logic_rule="test_mode_policy",
        )

        out = self.svc.project_mode_policy_resolve(
            {
                "user_id": "u_mode",
                "session_id": "s_mode_1",
                "message": "Help me choose a brew for sweet balanced coffee.",
                "context": "Keep it short.",
                "model_role": "general",
            }
        )
        policy = out["mode_policy"]
        self.assertTrue(policy["selected"])
        self.assertEqual(policy["mode_node_id"], int(mode_node.id))
        self.assertEqual(policy["resolved_role"], "planner")
        self.assertEqual(policy["selection_reason"], "user_weighted")
        self.assertEqual(len(policy["memory_items"]), 1)
        self.assertIn("Protected memory", policy["effective_context"])
        self.assertIn("Weighted important context", policy["effective_context"])

    def test_context_mode_endpoints_save_focus_and_feedback_reorder_memory(self):
        created = self.svc.project_context_mode_upsert(
            {
                "user_id": "u_mode_mgr",
                "session_id": "s_mode_mgr_1",
                "name": "Star Profile",
                "domain": "persona",
                "prompt_guardrails": "Stay structured across body, psyche, duties, and rights.",
                "protected_memory": "Do not let unverified gossip overwrite stable profile facts.",
                "llm_role": "analyst",
            }
        )
        mode_id = int(created["mode"]["mode_node_id"])
        self.assertGreater(mode_id, 0)
        self.assertTrue(created["mode_policy"]["selected"])

        focus_a = self.svc.project_context_mode_capture_focus(
            {
                "user_id": "u_mode_mgr",
                "session_id": "s_mode_mgr_1",
                "mode_node_id": mode_id,
                "name": "Physical Structure",
                "summary": "Stable physical traits and visible presentation.",
                "manual_capture": True,
                "weight": 0.9,
            }
        )
        focus_b = self.svc.project_context_mode_capture_focus(
            {
                "user_id": "u_mode_mgr",
                "session_id": "s_mode_mgr_1",
                "mode_node_id": mode_id,
                "name": "Rights And Duties",
                "summary": "Rights, obligations, and legal constraints.",
                "manual_capture": True,
                "weight": 0.82,
            }
        )
        focus_a_id = int(focus_a["focus"]["focus_node_id"])
        focus_b_id = int(focus_b["focus"]["focus_node_id"])
        self.assertGreater(focus_a_id, 0)
        self.assertGreater(focus_b_id, 0)

        before = self.svc.project_mode_policy_resolve(
            {
                "user_id": "u_mode_mgr",
                "session_id": "s_mode_mgr_1",
                "message": "Build context for a person profile.",
                "context": "",
                "mode_node_id": mode_id,
            }
        )
        self.assertEqual(before["mode_policy"]["memory_items"][0]["name"], "Physical Structure")

        self.svc.project_context_mode_feedback(
            {
                "user_id": "u_mode_mgr",
                "session_id": "s_mode_mgr_1",
                "mode_node_id": mode_id,
                "target_focus_node_id": focus_a_id,
                "decision": "bad",
                "summary": "This branch was overused.",
            }
        )
        self.svc.project_context_mode_feedback(
            {
                "user_id": "u_mode_mgr",
                "session_id": "s_mode_mgr_1",
                "mode_node_id": mode_id,
                "target_focus_node_id": focus_b_id,
                "decision": "good",
                "summary": "This branch should lead context assembly.",
            }
        )

        after = self.svc.project_mode_policy_resolve(
            {
                "user_id": "u_mode_mgr",
                "session_id": "s_mode_mgr_1",
                "message": "Build context for a person profile.",
                "context": "",
                "mode_node_id": mode_id,
            }
        )
        self.assertEqual(after["mode_policy"]["memory_items"][0]["name"], "Rights And Duties")
        self.assertGreater(
            float(after["mode_policy"]["memory_items"][0]["learned_score"]),
            float(after["mode_policy"]["memory_items"][1]["learned_score"]),
        )
        self.assertGreaterEqual(float(after["mode_policy"]["mode_state"]["learned_score"]), 0.5)

    def test_project_archive_chat_uses_backend_mode_policy(self):
        captured: dict[str, str] = {}

        def fake_model_resolver(model_path: str):
            token = str(model_path or "").strip()
            if "qwen2.5-7b-instruct" not in token:
                return None

            def _run(prompt: str) -> str:
                captured.setdefault("prompt", str(prompt))
                return json.dumps(
                    {
                        "summary": "Saved one coffee choice note.",
                        "archive_updates": [
                            {
                                "entity": "coffee_preference",
                                "field": "next_brew",
                                "operation": "upsert",
                                "value": "Use pour over for a cleaner cup.",
                                "reason": "Matches the weighted saved context.",
                                "source": "mode_policy_test",
                                "confidence": 0.82,
                                "tags": ["coffee"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )

            return _run

        svc = GraphWorkspaceService(
            use_env_adapter=False,
            enable_living_system=False,
            model_llm_resolver=fake_model_resolver,
        )
        mode_node = svc.api.engine.create_node(
            "context_mode",
            attributes={
                "user_id": "u_mode_chat",
                "name": "Coffee Planner",
                "domain": "coffee",
                "prompt_guardrails": "Prefer practical brewing advice.",
                "protected_memory": "Avoid burnt or overly bitter recommendations.",
                "llm_role": "planner",
                "updated_at": 456.0,
            },
            state={"context_weight": 1.22},
        )
        focus_node = svc.api.engine.create_node(
            "context_focus",
            attributes={
                "user_id": "u_mode_chat",
                "name": "Pour Over",
                "summary": "Clean cup and better clarity for balanced sweetness.",
            },
            state={"importance": 0.88},
        )
        svc._connect_nodes(  # noqa: SLF001
            from_node=int(mode_node.id),
            to_node=int(focus_node.id),
            relation_type="mode_focus",
            weight=0.93,
            logic_rule="test_mode_policy",
        )

        out = svc.project_archive_verified_chat(
            {
                "user_id": "u_mode_chat",
                "session_id": "s_mode_chat_1",
                "message": "Recommend a coffee brewing direction.",
                "context": "Need one concise next step.",
                "model_path": "models/gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                "model_role": "general",
                "mode_node_id": int(mode_node.id),
                "use_mode_policy": True,
                "apply_to_graph": False,
                "verification_mode": "balanced",
                "top_k": 3,
                "auto_triage": False,
            }
        )
        self.assertIn("Mode anchor: Coffee Planner", captured.get("prompt", ""))
        self.assertIn("Weighted important context:", captured.get("prompt", ""))
        self.assertEqual(out["mode_policy"]["resolved_role"], "planner")
        self.assertEqual(out["model"]["requested_model_role"], "general")
        self.assertEqual(out["model"]["effective_model_role"], "planner")
        self.assertIn("effective_context", out["input"])
        self.assertIn("Protected memory", out["input"]["effective_context"])

    def test_project_graph_node_assist_links_to_selected_node(self):
        def fake_model_resolver(model_path: str):
            token = str(model_path or "").strip()
            if "qwen2.5-7b-instruct" not in token:
                return None

            def _run(_: str) -> str:
                return json.dumps(
                    {
                        "summary": "Clarify the node and attach one concrete next step.",
                        "archive_updates": [
                            {
                                "entity": "graph_node",
                                "field": "summary",
                                "operation": "upsert",
                                "value": "Clarified summary with an actionable next step.",
                                "reason": "Keeps the node useful to the user.",
                                "source": "node_assist_test",
                                "confidence": 0.84,
                                "tags": ["clarity", "planning"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )

            return _run

        svc = GraphWorkspaceService(
            use_env_adapter=False,
            enable_living_system=False,
            model_llm_resolver=fake_model_resolver,
        )
        created = svc.create_node(
            {
                "node_type": "concept",
                "name": "Mission Plan",
                "description": "Needs a tighter plan and clearer next action.",
            }
        )
        node_id = int(created["node"]["id"])

        out = svc.project_graph_node_assist(
            {
                "node_id": node_id,
                "action": "improve",
                "message": "Tighten this node and suggest the next action.",
                "model_path": "models/gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                "apply_to_graph": True,
                "verification_mode": "balanced",
                "top_k": 3,
            }
        )

        self.assertEqual(out["node_assist"]["node_id"], node_id)
        self.assertEqual(out["node_assist"]["action"], "improve")
        self.assertGreater(int(out["node_assist"]["session_node_id"] or 0), 0)
        self.assertGreaterEqual(len(out["archive_updates"]), 1)
        self.assertTrue(str(out.get("assistant_reply", "")).strip())
        edges = out["snapshot"]["edges"]
        session_node_id = int(out["node_assist"]["session_node_id"])
        self.assertTrue(
            any(
                int(edge.get("from", 0)) == node_id
                and int(edge.get("to", 0)) == session_node_id
                and str(edge.get("relation_type", "")) == "requested_node_assist"
                for edge in edges
            )
        )
        relation_types = {str(edge.get("relation_type", "")) for edge in edges}
        self.assertIn("targets_graph_node", relation_types)
        self.assertIn("suggests_change_for", relation_types)

    def test_project_graph_edge_assist_updates_edge_and_links_session(self):
        def fake_model_resolver(model_path: str):
            token = str(model_path or "").strip()
            if "qwen2.5-7b-instruct" not in token:
                return None

            def _run(_: str) -> str:
                return json.dumps(
                    {
                        "summary": "The relation needs tighter semantics and a clearer justification.",
                        "archive_updates": [
                            {
                                "entity": "graph_edge",
                                "field": "relation_type",
                                "operation": "review",
                                "value": "supports_with_evidence",
                                "reason": "The current link is too generic.",
                                "source": "edge_assist_test",
                                "confidence": 0.82,
                                "tags": ["edge", "semantics"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )

            return _run

        svc = GraphWorkspaceService(
            use_env_adapter=False,
            enable_living_system=False,
            model_llm_resolver=fake_model_resolver,
        )
        from_node = svc.create_node({"node_type": "concept", "name": "Source Concept"})
        to_node = svc.create_node({"node_type": "concept", "name": "Target Concept"})
        svc.create_edge(
            {
                "from_node": int(from_node["node"]["id"]),
                "to_node": int(to_node["node"]["id"]),
                "relation_type": "related_to",
                "direction": "directed",
                "weight": 0.42,
            }
        )

        out = svc.project_graph_edge_assist(
            {
                "from_node": int(from_node["node"]["id"]),
                "to_node": int(to_node["node"]["id"]),
                "relation_type": "related_to",
                "direction": "directed",
                "action": "improve",
                "message": "Tighten this relation and explain the better semantic.",
                "model_path": "models/gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                "apply_to_graph": True,
                "verification_mode": "balanced",
                "top_k": 3,
            }
        )

        self.assertEqual(out["edge_assist"]["relation_type"], "related_to")
        self.assertEqual(out["edge_assist"]["action"], "improve")
        self.assertGreater(int(out["edge_assist"]["session_node_id"] or 0), 0)
        self.assertGreaterEqual(len(out["archive_updates"]), 1)
        self.assertTrue(str(out.get("assistant_reply", "")).strip())
        refreshed_edge = next(
            (
                edge
                for edge in out["snapshot"]["edges"]
                if int(edge.get("from", 0)) == int(from_node["node"]["id"])
                and int(edge.get("to", 0)) == int(to_node["node"]["id"])
                and str(edge.get("relation_type", "")) == "related_to"
            ),
            None,
        )
        self.assertIsNotNone(refreshed_edge)
        metadata = refreshed_edge.get("metadata", {}) if isinstance(refreshed_edge, dict) else {}
        self.assertEqual(str(metadata.get("last_edge_assist_action", "")), "improve")
        relation_types = {str(edge.get("relation_type", "")) for edge in out["snapshot"]["edges"]}
        self.assertIn("requested_edge_assist", relation_types)
        self.assertIn("targets_graph_edge", relation_types)
        self.assertIn("suggests_edge_change_from", relation_types)
        self.assertIn("suggests_edge_change_to", relation_types)

    def test_project_graph_foundation_create_builds_visible_branch_with_mistral(self):
        def fake_model_resolver(model_path: str):
            token = str(model_path or "").strip()
            if "mistral-7b-instruct-v0.3" not in token:
                return None

            def _run(prompt: str) -> str:
                text = str(prompt or "")
                if "You are a graph monitor." in text:
                    return json.dumps(
                        {
                            "node_patches": [
                                {
                                    "node_id": 0,
                                    "summary": "ignored invalid patch",
                                    "confidence": 0.4,
                                    "reason": "invalid test row",
                                }
                            ],
                            "edge_patches": [],
                        },
                        ensure_ascii=False,
                    )
                return json.dumps(
                    {
                        "title": "Career Pivot Foundation",
                        "summary": "A compact branch that breaks the career pivot into practical concepts.",
                        "concepts": [
                            {
                                "name": "Skills Audit",
                                "summary": "Map current capabilities and missing gaps.",
                                "reason": "Clarifies the starting point.",
                                "confidence": 0.88,
                                "children": [
                                    {
                                        "name": "Transferable Skills",
                                        "summary": "List durable strengths that move across roles.",
                                        "reason": "These carry over immediately.",
                                        "confidence": 0.81,
                                    }
                                ],
                            },
                            {
                                "name": "Target Roles",
                                "summary": "Define role families worth testing first.",
                                "reason": "Keeps the search focused.",
                                "confidence": 0.84,
                                "children": [
                                    {
                                        "name": "Fast Validation",
                                        "summary": "Choose low-cost interviews and small experiments.",
                                        "reason": "Reduces risk before large moves.",
                                        "confidence": 0.79,
                                    }
                                ],
                            },
                        ],
                    },
                    ensure_ascii=False,
                )

            return _run

        svc = GraphWorkspaceService(
            use_env_adapter=False,
            enable_living_system=False,
            model_llm_resolver=fake_model_resolver,
        )
        anchor = svc.create_node(
            {
                "node_type": "concept",
                "name": "Career Pivot",
                "description": "Needs a structured branch with deeper concepts.",
            }
        )

        out = svc.project_graph_foundation_create(
            {
                "topic": "",
                "context": "Focus on practical planning, role discovery, and proof before large changes.",
                "target_node_id": int(anchor["node"]["id"]),
                "depth": 2,
                "concept_limit": 4,
            }
        )

        foundation = out["foundation"]
        self.assertTrue(out["ok"])
        self.assertEqual(foundation["target_node_id"], int(anchor["node"]["id"]))
        self.assertGreater(int(foundation["root_node_id"]), 0)
        self.assertGreaterEqual(int(foundation["created_nodes"]), 5)
        self.assertGreaterEqual(int(foundation["created_edges"]), 5)
        self.assertEqual(foundation["foundation_kind"], "planning")
        self.assertEqual(out["model"]["selected_model_path"], "models/gguf/textGen/mistral-7b-instruct-v0.3-q4_k_m.gguf")
        self.assertFalse(out["model"]["used_fallback"])
        relation_types = {str(edge.get("relation_type", "")) for edge in out["snapshot"]["edges"]}
        self.assertIn("expands_concept", relation_types)
        self.assertIn("contains_concept", relation_types)
        self.assertIn("deepens_concept", relation_types)
        self.assertIn("enables", relation_types)
        node_types = {str(node.get("type", "")) for node in out["snapshot"]["nodes"]}
        self.assertIn("foundation_branch", node_types)
        created_names = {
            str(node.get("attributes", {}).get("name", node.get("attributes", {}).get("title", ""))).strip()
            for node in out["snapshot"]["nodes"]
        }
        self.assertIn("Skills Audit", created_names)
        self.assertIn("Transferable Skills", created_names)
        self.assertEqual(out["blueprint"]["foundation_kind"], "planning")
        self.assertGreaterEqual(len(out["blueprint"].get("frame_plan", [])), 4)
        self.assertGreaterEqual(len(out["blueprint"].get("links", [])), 1)

    def test_project_graph_foundation_create_uses_logical_fallback_for_philosophy(self):
        svc = GraphWorkspaceService(
            use_env_adapter=False,
            enable_living_system=False,
            model_llm_resolver=lambda _path: None,
        )

        out = svc.project_graph_foundation_create(
            {
                "topic": "Philosophy",
                "context": "Treat philosophy as a field of inquiry, with branches, questions, methods, and practical use.",
                "depth": 2,
                "concept_limit": 5,
            }
        )

        self.assertTrue(out["ok"])
        self.assertTrue(out["model"]["used_fallback"])
        self.assertEqual(out["foundation"]["foundation_kind"], "philosophy")
        self.assertEqual(out["blueprint"]["foundation_kind"], "philosophy")
        frame_plan = out["blueprint"].get("frame_plan", [])
        self.assertGreaterEqual(len(frame_plan), 5)
        self.assertEqual(str(frame_plan[0].get("key", "")), "definition")
        created_names = {
            str(node.get("attributes", {}).get("name", node.get("attributes", {}).get("title", ""))).strip()
            for node in out["snapshot"]["nodes"]
        }
        self.assertIn("Definition of Philosophy", created_names)
        self.assertIn("Main Branches of Philosophy", created_names)
        relation_types = {str(edge.get("relation_type", "")) for edge in out["snapshot"]["edges"]}
        self.assertIn("organizes", relation_types)
        self.assertIn("raises_questions", relation_types)

    def test_project_graph_foundation_create_detects_coffee_domain_frames(self):
        svc = GraphWorkspaceService(
            use_env_adapter=False,
            enable_living_system=False,
            model_llm_resolver=lambda _path: None,
        )

        out = svc.project_graph_foundation_create(
            {
                "topic": "Coffee",
                "context": "Include history, varieties, processing, brewing methods, and taste preferences.",
                "depth": 2,
                "concept_limit": 6,
            }
        )

        self.assertTrue(out["ok"])
        self.assertEqual(out["foundation"]["foundation_kind"], "coffee")
        created_names = {
            str(node.get("attributes", {}).get("name", node.get("attributes", {}).get("title", ""))).strip()
            for node in out["snapshot"]["nodes"]
        }
        self.assertIn("What Coffee Is", created_names)
        self.assertIn("Brewing Methods", created_names)
        relation_types = {str(edge.get("relation_type", "")) for edge in out["snapshot"]["edges"]}
        self.assertIn("guides_action", relation_types)

    def test_project_archive_review_apply_allows_edit_and_recheck(self):
        edited_updates = [
            {
                "entity": "project_plan",
                "field": "next_step",
                "operation": "upsert",
                "value": "Run controlled benchmark before rollout.",
                "reason": "Keeps rollout risk measurable.",
                "source": "qa_runbook_v3",
                "confidence": 0.86,
                "tags": ["qa", "performance"],
            }
        ]
        out = self.svc.project_archive_review_apply(
            {
                "user_id": "u_archive_review",
                "session_id": "s_archive_review_1",
                "message": "Apply reviewed archive updates",
                "summary": "Reviewed by user",
                "archive_updates": edited_updates,
                "verification_mode": "strict",
                "apply_to_graph": True,
                "top_k": 3,
            }
        )
        self.assertIn("archive_updates", out)
        self.assertEqual(len(out["archive_updates"]), 1)
        self.assertTrue(out["verification"]["verified"])
        self.assertIn("assistant_reply", out)
        self.assertTrue(out["graph_binding"]["attached"])
        node_types = {row.get("type") for row in out["snapshot"]["nodes"]}
        self.assertIn("llm_archive_update_record", node_types)

    def test_project_personal_tree_ingest_creates_summary_points_and_source(self):
        out = self.svc.project_personal_tree_ingest(
            {
                "user_id": "u_tree",
                "session_id": "sess_tree_1",
                "title": "Law review",
                "topic": "Contract obligations",
                "text": (
                    "Article 12 defines core obligations for both parties. "
                    "The law emphasizes written notice and delivery evidence. "
                    "A breach triggers compensation and corrective actions."
                ),
                "source_type": "law",
                "source_url": "https://example.org/law/contract",
                "source_title": "Contract Law Extract",
                "max_points": 4,
            }
        )
        self.assertIn("extraction", out)
        self.assertTrue(str(out["extraction"].get("summary", "")).strip())
        self.assertTrue(out["extraction"].get("points"))
        self.assertIn("tree", out)
        tree = out["tree"]
        self.assertGreaterEqual(int(tree["stats"]["node_count"]), 3)
        node_types = {row.get("type") for row in tree.get("nodes", [])}
        self.assertIn("thought_summary_node", node_types)
        self.assertIn("thought_point_node", node_types)
        self.assertIn("source_reference", node_types)

    def test_project_personal_tree_note_and_view(self):
        saved = self.svc.project_personal_tree_note(
            {
                "user_id": "u_tree_note",
                "session_id": "sess_note_1",
                "title": "Idea",
                "note": "Use a focused mini-tree for legal risk analysis.",
                "tags": ["idea", "risk", "law"],
                "links": ["https://example.org/article/1"],
                "source_type": "article",
                "source_url": "https://example.org/article/1",
                "source_title": "Article One",
            }
        )
        self.assertIn("note", saved)
        note_node_id = int(saved["note"]["node_id"])
        self.assertGreater(note_node_id, 0)
        self.assertIn("persisted", saved)

        viewed = self.svc.project_personal_tree_view(
            {
                "user_id": "u_tree_note",
                "focus_node_id": note_node_id,
                "max_nodes": 80,
            }
        )
        self.assertIn("tree", viewed)
        tree = viewed["tree"]
        self.assertGreaterEqual(int(tree["stats"]["node_count"]), 2)
        self.assertTrue(tree.get("notes"))
        self.assertTrue(tree.get("sources"))

    def test_project_packages_manage_store_purge_restore(self):
        stored = self.svc.project_packages_manage(
            {
                "user_id": "u_pkg",
                "session_id": "s_pkg",
                "package_name": "inbox",
                "action": "store",
                "items": [
                    "Temporary tmp draft for old migration",
                    "Validated release checklist for deployment",
                ],
                "classify_with_llm": False,
            }
        )
        self.assertIn("items", stored)
        self.assertGreaterEqual(len(stored["items"]), 2)

        purged = self.svc.project_packages_manage(
            {
                "user_id": "u_pkg",
                "session_id": "s_pkg",
                "package_name": "inbox",
                "action": "purge",
                "apply_changes": True,
                "confirmation": "confirm",
            }
        )
        self.assertIn("changed_item_ids", purged)
        self.assertTrue(purged["changed_item_ids"])

        restore_target = int(purged["changed_item_ids"][0])
        restored = self.svc.project_packages_manage(
            {
                "user_id": "u_pkg",
                "session_id": "s_pkg",
                "package_name": "inbox",
                "action": "restore",
                "item_node_ids": [restore_target],
                "apply_changes": True,
                "confirmation": "confirm",
            }
        )
        restored_rows = [row for row in restored["items"] if int(row.get("node_id", 0)) == restore_target]
        self.assertTrue(restored_rows)
        self.assertEqual(str(restored_rows[0].get("status", "")), "active")

    def test_project_memory_namespace_apply_and_view(self):
        node_out = self.svc.create_node(
            {
                "node_type": "generic",
                "attributes": {
                    "user_id": "u_ns",
                    "name": "Risk checklist node",
                    "summary": "Risk checklist and mitigation tasks",
                    "namespace": "personal",
                },
            }
        )
        node_id = int(node_out["node"]["id"])
        applied = self.svc.project_memory_namespace_apply(
            {
                "user_id": "u_ns",
                "session_id": "s_ns",
                "namespace": "experiment",
                "scope": "owned",
                "node_ids": [node_id],
                "apply_changes": True,
                "confirmation": "confirm",
            }
        )
        self.assertEqual(int(applied["affected_count"]), 1)
        self.assertTrue(applied["policy"]["apply_allowed"])

        viewed = self.svc.project_memory_namespace_view(
            {
                "user_id": "u_ns",
                "scope": "owned",
                "max_nodes": 50,
            }
        )
        self.assertIn("namespace_counts", viewed)
        self.assertGreaterEqual(int(viewed["namespace_counts"].get("experiment", 0)), 1)

    def test_project_graph_rag_contradiction_task_risk(self):
        self.svc.create_node(
            {
                "node_type": "generic",
                "attributes": {
                    "user_id": "u_ai",
                    "name": "Release policy",
                    "summary": "System must deploy after legal review is complete.",
                    "namespace": "personal",
                },
            }
        )
        self.svc.create_node(
            {
                "node_type": "generic",
                "attributes": {
                    "user_id": "u_ai",
                    "name": "Emergency policy",
                    "summary": "System must not deploy before legal review.",
                    "namespace": "personal",
                },
            }
        )
        rag = self.svc.project_graph_rag_query(
            {
                "query": "legal review before deploy",
                "user_id": "u_ai",
                "scope": "owned",
                "use_llm": False,
                "top_k": 4,
            }
        )
        self.assertIn("hits", rag)
        self.assertTrue(rag["hits"])
        self.assertTrue(str(rag.get("answer", "")).strip())

        contradictions = self.svc.project_contradiction_scan(
            {
                "user_id": "u_ai",
                "scope": "owned",
                "max_nodes": 80,
                "top_k": 10,
                "apply_to_graph": True,
                "confirmation": "confirm",
            }
        )
        self.assertGreaterEqual(int(contradictions["issue_count"]), 1)
        self.assertTrue(contradictions["graph_binding"]["attached"])

        board = self.svc.project_task_risk_board(
            {
                "user_id": "u_ai",
                "session_id": "s_board",
                "tasks": [
                    {"title": "Finalize legal approval for release", "description": "deadline this week"},
                    {"title": "Prepare optional UI polish draft"},
                ],
                "apply_to_graph": True,
                "confirmation": "confirm",
            }
        )
        self.assertEqual(int(board["task_count"]), 2)
        self.assertTrue(board["graph_binding"]["attached"])

    def test_project_timeline_policy_quality_backup_and_audit(self):
        policy = self.svc.project_llm_policy_update(
            {
                "mode": "assisted_autonomy",
                "trusted_users": ["u_backup"],
                "allow_apply_for_actions": ["backup_restore"],
                "merge_lists": True,
            }
        )
        self.assertTrue(policy["ok"])
        self.assertEqual(str(policy["policy"]["mode"]), "assisted_autonomy")

        self.svc.create_node(
            {
                "node_type": "generic",
                "attributes": {
                    "user_id": "u_backup",
                    "name": "Backup anchor node",
                    "summary": "Anchor for backup restore test",
                    "namespace": "personal",
                },
            }
        )

        quality = self.svc.project_quality_harness(
            {
                "user_id": "u_backup",
                "sample_queries": ["backup anchor node"],
            }
        )
        self.assertIn("score", quality)
        self.assertGreaterEqual(float(quality["score"]), 0.0)

        created = self.svc.project_backup_create(
            {
                "label": "unit_test",
                "user_id": "u_backup",
                "include_events": True,
                "event_limit": 200,
            }
        )
        self.assertTrue(created["ok"])
        backup_path = str(created["path"])
        self.assertTrue(Path(backup_path).exists())

        restored = self.svc.project_backup_restore(
            {
                "path": backup_path,
                "user_id": "u_backup",
                "session_id": "s_restore",
                "apply_changes": True,
                "confirmation": "confirm",
            }
        )
        self.assertTrue(restored["applied"])
        self.assertGreaterEqual(int(restored["result"]["created_nodes"]), 1)

        timeline = self.svc.project_timeline_replay({"limit": 200})
        self.assertIn("timeline", timeline)
        self.assertTrue(timeline["timeline"])

        audit = self.svc.project_audit_logs({"limit": 200, "include_backups": True})
        self.assertIn("events", audit)
        self.assertGreaterEqual(int(audit["backup_count"]), 1)

    def test_project_wrapper_profile_update_feedback_and_get(self):
        got = self.svc.project_wrapper_profile_get({"user_id": "u_wrap"})
        self.assertIn("profile", got)
        self.assertEqual(str(got["profile"]["preferred_role"]), "general")

        updated = self.svc.project_wrapper_profile_update(
            {
                "user_id": "u_wrap",
                "preferred_role": "analyst",
                "memory_scope": "owned",
                "personalization": {
                    "response_style": "concise",
                    "reasoning_depth": "balanced",
                    "risk_tolerance": "low",
                    "tone": "direct",
                    "focus_goals": ["reduce noise", "actionable steps"],
                },
            }
        )
        self.assertTrue(updated["ok"])
        self.assertEqual(str(updated["profile"]["preferred_role"]), "analyst")
        self.assertEqual(str(updated["profile"]["personalization"]["response_style"]), "concise")

        feedback = self.svc.project_wrapper_feedback(
            {
                "user_id": "u_wrap",
                "feedback_items": [
                    {"message": "Shorter answers please", "decision": "accept", "score": 0.9},
                    {"message": "Too risky suggestions", "decision": "reject", "score": 0.2},
                ],
                "attach_to_graph": True,
            }
        )
        self.assertTrue(feedback["ok"])
        self.assertGreaterEqual(int(feedback["summary"]["new_items"]), 2)
        self.assertGreaterEqual(int(feedback["feedback_node_id"]), 1)

    def test_project_wrapper_respond_with_memory_context_fallback(self):
        self.svc.create_node(
            {
                "node_type": "generic",
                "attributes": {
                    "user_id": "u_wrap_2",
                    "name": "Release constraints",
                    "summary": "Do not deploy before legal and security checks are complete.",
                    "namespace": "personal",
                },
            }
        )
        out = self.svc.project_wrapper_respond(
            {
                "user_id": "u_wrap_2",
                "session_id": "s_wrap_2",
                "message": "How should I plan deployment safely?",
                "use_memory": True,
                "memory_scope": "owned",
                "memory_top_k": 5,
                "apply_profile_update": True,
            }
        )
        self.assertIn("reply", out)
        self.assertTrue(str(out["reply"]).strip())
        self.assertIn("memory", out)
        self.assertTrue(out["memory"]["context"])
        self.assertTrue(out["model"]["used_fallback"])
        self.assertIn("triage", out)
        self.assertTrue(out["triage"]["enabled"])

    def test_project_wrapper_respond_with_explicit_model_path(self):
        def fake_model_resolver(model_path: str):
            token = str(model_path or "").strip()
            if "h2o-danube3-4b-chat" not in token:
                return None

            def _run(_prompt: str) -> str:
                return "Plan: first collect constraints, then execute one low-risk step."

            return _run

        svc = GraphWorkspaceService(
            use_env_adapter=False,
            enable_living_system=False,
            model_llm_resolver=fake_model_resolver,
        )
        out = svc.project_wrapper_respond(
            {
                "user_id": "u_wrap_model",
                "message": "Need concise action plan",
                "model_path": "models/gguf/textGen/h2o-danube3-4b-chat-Q5_K_M.gguf",
                "role": "general",
                "use_memory": False,
            }
        )
        self.assertFalse(out["model"]["used_fallback"])
        self.assertEqual(str(out["model"]["resolution"]["mode"]), "explicit_model_path")
        self.assertIn("collect constraints", str(out["reply"]).lower())
        self.assertIn("triage", out)
        self.assertTrue(out["triage"]["enabled"])

    def test_project_wrapper_respond_includes_input_capture_and_monitor(self):
        def fake_model_resolver(model_path: str):
            token = str(model_path or "").strip()
            if "qwen2.5-7b-instruct" in token:
                def _run(_: str) -> str:
                    return json.dumps(
                        {
                            "summary": "Structured wrapper input captured.",
                            "archive_updates": [
                                {
                                    "entity": "wrapper_input",
                                    "field": "intent",
                                    "operation": "append",
                                    "value": "Need a concise action plan.",
                                    "reason": "Primary request extracted from chat input.",
                                    "source": "wrapper_test",
                                    "confidence": 0.81,
                                    "tags": ["wrapper", "intent"],
                                }
                            ],
                        },
                        ensure_ascii=False,
                    )

                return _run
            if "mistral-7b-instruct-v0.3" in token:
                def _run(prompt: str) -> str:
                    match = re.search(r'"node_id":\s*(\d+)', prompt)
                    node_id = int(match.group(1)) if match else 0
                    return json.dumps(
                        {
                            "node_patches": (
                                [
                                    {
                                        "node_id": node_id,
                                        "summary": "Wrapper monitor note.",
                                        "confidence": 0.71,
                                        "reason": "Keep wrapper context concise.",
                                    }
                                ]
                                if node_id > 0
                                else []
                            ),
                            "edge_patches": [],
                        },
                        ensure_ascii=False,
                    )

                return _run
            return None

        svc = GraphWorkspaceService(
            use_env_adapter=False,
            enable_living_system=False,
            model_llm_resolver=fake_model_resolver,
        )
        out = svc.project_wrapper_respond(
            {
                "user_id": "u_wrap_chain",
                "session_id": "s_wrap_chain",
                "message": "Give me one concise next step.",
                "use_memory": False,
                "store_interaction": True,
            }
        )
        self.assertTrue(out["input_extraction"]["enabled"])
        self.assertTrue(out["input_extraction"]["graph_binding"]["attached"])
        self.assertTrue(out["graph_monitor"]["attached"])
        self.assertGreater(int(out["graph_monitor"].get("session_node_id", 0)), 0)

    def test_project_wrapper_respond_creates_subject_branch_and_dialect_dictionary(self):
        out = self.svc.project_wrapper_respond(
            {
                "user_id": "u_wrap_gossip",
                "session_id": "s_wrap_gossip",
                "message": "Давай перемоем кости Ивана, говорят он сорвал дедлайн и скрыл правки.",
                "use_memory": False,
                "store_interaction": True,
                "subject_name": "Иван",
                "gossip_mode": "auto",
                "allow_subject_branch_write": True,
                "capture_dialect": True,
            }
        )
        self.assertTrue(out["gossip_detected"])
        self.assertTrue(out["subject_binding"]["attached"])
        self.assertGreaterEqual(len(out["subject_binding"]["subject_branch_node_ids"]), 1)
        self.assertGreaterEqual(len(out["subject_binding"]["claim_node_ids"]), 1)
        self.assertGreaterEqual(int(out["dialect"]["captured_terms"]), 1)
        self.assertGreaterEqual(int(out["dialect"]["dictionary_size"]), 1)
        node_types = {row.get("type") for row in out["snapshot"]["nodes"]}
        self.assertIn("subject_profile_branch", node_types)
        self.assertIn("subject_claim_node", node_types)
        self.assertIn("user_dialect_dictionary", node_types)

    def test_project_archive_chat_binds_subject_branch_when_gossip_detected(self):
        def fake_model_resolver(model_path: str):
            if "qwen2.5-7b-instruct" not in str(model_path or ""):
                return None

            def _run(_: str) -> str:
                return json.dumps(
                    {
                        "summary": "Captured unverified discussion claims about subject profile.",
                        "archive_updates": [
                            {
                                "entity": "Иван",
                                "field": "rumor_note",
                                "operation": "append",
                                "value": "Claim about missed deadlines.",
                                "reason": "Conversation mentions repeated delay concerns.",
                                "source": "chat_user_report",
                                "confidence": 0.66,
                                "tags": ["gossip", "rumor"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )

            return _run

        svc = GraphWorkspaceService(
            use_env_adapter=False,
            enable_living_system=False,
            model_llm_resolver=fake_model_resolver,
        )
        out = svc.project_archive_verified_chat(
            {
                "user_id": "u_archive_gossip",
                "session_id": "s_archive_gossip",
                "message": "Сплетни про Ивана: говорят, что он постоянно срывает дедлайны.",
                "model_path": "models/gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                "apply_to_graph": True,
                "verification_mode": "balanced",
                "subject_name": "Иван",
                "gossip_mode": "auto",
                "allow_subject_branch_write": True,
                "capture_dialect": True,
            }
        )
        self.assertTrue(out["graph_binding"]["attached"])
        self.assertTrue(out["gossip_detected"])
        self.assertTrue(out["subject_binding"]["attached"])
        self.assertGreaterEqual(len(out["subject_binding"]["subject_branch_node_ids"]), 1)
        self.assertGreaterEqual(len(out["subject_binding"]["claim_node_ids"]), 1)
        self.assertGreaterEqual(int(out["dialect"]["captured_terms"]), 1)
        node_types = {row.get("type") for row in out["snapshot"]["nodes"]}
        self.assertIn("subject_profile_branch", node_types)
        self.assertIn("subject_claim_node", node_types)

    def test_project_integration_layer_manifest_and_invoke_wrapper(self):
        manifest = self.svc.project_integration_layer_manifest(
            {
                "host": "vscode",
                "app_id": "workspace_plugin",
            }
        )
        self.assertEqual(str(manifest.get("layer", "")), "autograph_integration_layer")
        self.assertIn("capabilities", manifest)
        self.assertGreaterEqual(len(manifest["capabilities"]), 1)
        self.assertIn("actions_allowed", manifest)
        self.assertIn("actions", manifest)
        self.assertGreaterEqual(len(manifest["actions"]), 1)
        self.assertIn("summary", manifest)
        self.assertGreaterEqual(int(manifest["summary"].get("action_count", 0)), 1)

        out = self.svc.project_integration_layer_invoke(
            {
                "action": "wrapper.respond",
                "host": "vscode",
                "app_id": "workspace_plugin",
                "user_id": "u_integration",
                "session_id": "s_integration",
                "input": {
                    "message": "Give me one practical next step for release planning.",
                },
                "options": {
                    "use_memory": False,
                    "store_interaction": True,
                    "auto_triage": True,
                },
            }
        )
        self.assertTrue(out["ok"])
        self.assertEqual(str(out["action"]), "wrapper.respond")
        self.assertTrue(str(out.get("chat_response", "")).strip())
        self.assertIn("structured_result", out)
        triage = (out.get("structured_result", {}) or {}).get("triage", {})
        self.assertTrue(triage.get("enabled"))
        self.assertIn("summary", out)
        self.assertEqual(str(out["summary"].get("action", "")), "wrapper.respond")
        self.assertIn("execution", out)
        self.assertEqual(str(out["execution"].get("action", "")), "wrapper.respond")


if __name__ == "__main__":
    unittest.main()
