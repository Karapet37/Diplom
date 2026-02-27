import json
from pathlib import Path
import unittest

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

    def test_project_autoruns_import(self):
        sample = (
            "Entry,Entry Location,Enabled,Category,Profile,Description,Publisher,Image Path,Launch String,Signer,Verified,VirusTotal\n"
            "OneDrive,HKCU\\\\Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Run,Enabled,Logon,user,OneDrive startup,Microsoft Corporation,C:\\\\Program Files\\\\Microsoft OneDrive\\\\OneDrive.exe,\\\"C:\\\\Program Files\\\\Microsoft OneDrive\\\\OneDrive.exe\\\",Microsoft Corporation,Signed,0/74\n"
            "UnknownTask,Task Scheduler,Enabled,Scheduled Tasks,user,Unknown task,Unknown,C:\\\\Temp\\\\unknown.exe,C:\\\\Temp\\\\unknown.exe,,Not verified,5/74\n"
        )
        out = self.svc.project_autoruns_import(
            {
                "text": sample,
                "user_id": "u_autoruns",
                "session_id": "sess_autoruns",
                "host_label": "Test Host",
            }
        )
        self.assertIn("scan", out)
        self.assertIn("summary", out)
        self.assertEqual(out["scan"]["rows_processed"], 2)
        self.assertGreaterEqual(out["summary"]["virus_total_positive_entries"], 1)
        high_or_medium = (
            int(out["summary"]["risk_counts"].get("high", 0))
            + int(out["summary"]["risk_counts"].get("medium", 0))
        )
        self.assertGreaterEqual(high_or_medium, 1)
        node_types = {row.get("type") for row in out["snapshot"]["nodes"]}
        self.assertIn("autoruns_scan", node_types)
        self.assertIn("autorun_entry", node_types)

    def test_project_autoruns_import_auto_detect(self):
        out = self.svc.project_autoruns_import(
            {
                "text": "",
                "auto_detect": True,
                "query": "show startup risks for browser updates",
                "user_id": "u_auto",
                "session_id": "sess_auto",
                "host_label": "Auto Host",
                "client": {
                    "user_agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0",
                    "platform": "Linux x86_64",
                    "language": "en-US",
                },
            },
            request_headers={
                "user-agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0",
                "x-forwarded-for": "198.51.100.30, 10.0.0.2",
            },
            request_ip="198.51.100.30",
        )
        self.assertIn("scan", out)
        self.assertEqual(out["scan"]["mode"], "auto_detected")
        self.assertGreaterEqual(int(out["scan"]["rows_processed"]), 1)
        self.assertIn("client_profile", out)

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
        node_types = {row.get("type") for row in out["snapshot"]["nodes"]}
        self.assertIn("llm_archive_update_branch", node_types)
        self.assertIn("llm_archive_update_session", node_types)
        self.assertIn("llm_archive_update_record", node_types)

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


if __name__ == "__main__":
    unittest.main()
