from __future__ import annotations

from fastapi.testclient import TestClient

from src.web.combined_app import create_combined_app


def test_combined_app_exposes_minimal_health_routes() -> None:
    with TestClient(create_combined_app()) as client:
        health = client.get('/api/health')
        cognitive = client.get('/api/cognitive/health')

    assert health.status_code == 200
    assert health.json()['ok'] is True
    assert cognitive.status_code == 200
    assert cognitive.json()['runtime'] == 'mvp-file-first'


def test_combined_app_mounts_minimal_cognitive_routes() -> None:
    app = create_combined_app()
    route_paths = {getattr(route, 'path', '') for route in app.routes}
    assert '/api/cognitive/chat/respond' in route_paths
    assert '/api/cognitive/files/upload' in route_paths
    assert '/api/cognitive/graph' in route_paths
    assert '/api/cognitive/graph/subgraph' in route_paths
    assert '/api/cognitive/personalities' in route_paths
    assert '/api/cognitive/rebuild' in route_paths
    assert '/api/health' in route_paths
    assert '/api/cognitive/dialogue/respond' not in route_paths
    assert '/api/merged/health' not in route_paths
    assert '/' in route_paths
