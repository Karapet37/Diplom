import unittest
from unittest.mock import Mock

from fastapi import HTTPException

from src.web.api import ProjectChatGraphRequest, create_app as create_workspace_app
from src.web.combined_app import create_combined_app


class CombinedAppTests(unittest.TestCase):
    def test_combined_app_mounts_cognitive_api_surface(self):
        app = create_combined_app()
        mount_paths = [getattr(route, "path", "") for route in app.routes]
        self.assertIn("/api/cognitive/health", mount_paths)
        self.assertNotIn("/roaches-viz", mount_paths)

    def test_combined_app_exposes_merged_health_route(self):
        app = create_combined_app()
        route_paths = [getattr(route, "path", "") for route in app.routes]
        self.assertIn("/api/merged/health", route_paths)

    def test_combined_app_places_cognitive_mount_before_spa_fallback(self):
        app = create_combined_app()
        route_paths = [getattr(route, "path", "") for route in app.routes]
        self.assertIn("/api/cognitive/health", route_paths)
        self.assertIn("/{full_path:path}", route_paths)
        self.assertLess(route_paths.index("/api/cognitive/health"), route_paths.index("/{full_path:path}"))

    def test_combined_app_registers_cognitive_endpoints_directly(self):
        app = create_combined_app()
        route_map = {getattr(route, "path", ""): route for route in app.routes}
        self.assertIn("/api/cognitive/health", route_map)
        self.assertIn("/api/cognitive/graph", route_map)
        self.assertIn("/api/cognitive/dialogue/respond", route_map)
        self.assertIn("/api/cognitive/persons", route_map)
        self.assertIn("/api/cognitive/agents", route_map)
        self.assertIn("/api/cognitive/professions", route_map)
        self.assertIn("/api/cognitive/style/learn", route_map)
        self.assertIn("/api/cognitive/style/profile", route_map)
        self.assertIn("/api/cognitive/sessions", route_map)
        self.assertIn("/api/cognitive/sessions/{session_id}", route_map)
        self.assertIn("/api/cognitive/graph/subgraph", route_map)
        self.assertIn("/api/cognitive/nodes", route_map)
        self.assertIn("/api/cognitive/edges", route_map)

    def test_combined_app_initializes_idle_maintenance_state(self):
        app = create_combined_app()
        self.assertEqual(app.state._busy_requests, 0)
        self.assertGreater(app.state._maintenance_idle_window_seconds, 0.0)
        self.assertIsNotNone(app.state._maintenance_stop_event)
        self.assertIsNotNone(app.state.graph_task_queue)
        self.assertIsNotNone(app.state.graph_worker)

    def test_combined_app_exposes_project_wikipedia_routes(self):
        app = create_combined_app()
        route_paths = [getattr(route, "path", "") for route in app.routes]
        self.assertIn("/api/project/wiki/random", route_paths)
        self.assertIn("/api/project/wiki/search", route_paths)

    def test_workspace_chat_graph_uses_cognitive_actor_when_present(self):
        app = create_workspace_app(include_frontend_routes=False)
        actor = Mock()
        actor.ask.return_value = {"ok": True, "assistant_reply": "ok", "graph_binding": {"attached": True}}
        app.state.cognitive_actor = actor
        endpoint = None
        for route in app.routes:
            if getattr(route, "path", None) == "/api/project/chat-graph":
                endpoint = route.endpoint
                break
        self.assertIsNotNone(endpoint)
        payload = ProjectChatGraphRequest(
            message="Amy introduces a new routine to Sheldon carefully.",
            context="The change is small and structured.",
            apply_to_graph=True,
            chat_model_role="general",
        )
        result = endpoint(payload)
        self.assertTrue(result["ok"])
        actor.ask.assert_called_once()
        command, forwarded = actor.ask.call_args.args
        self.assertEqual(command, "chat_graph")
        self.assertEqual(forwarded["message"], payload.message)
        self.assertEqual(forwarded["context"], payload.context)
        self.assertTrue(forwarded["apply_to_graph"])

    def test_workspace_chat_graph_returns_504_when_cognitive_actor_times_out(self):
        app = create_workspace_app(include_frontend_routes=False)
        actor = Mock()
        actor.ask.side_effect = TimeoutError("Actor 'graph-actor' timed out waiting for command 'chat_graph' after 120.0s")
        app.state.cognitive_actor = actor
        endpoint = None
        for route in app.routes:
            if getattr(route, "path", None) == "/api/project/chat-graph":
                endpoint = route.endpoint
                break
        self.assertIsNotNone(endpoint)
        payload = ProjectChatGraphRequest(message="slow request", context="", apply_to_graph=True, chat_model_role="general")
        with self.assertRaises(HTTPException) as ctx:
            endpoint(payload)
        self.assertEqual(ctx.exception.status_code, 504)
