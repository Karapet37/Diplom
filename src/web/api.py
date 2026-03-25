from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agent_system.runtime_config import get_runtime_config


def attach_frontend_routes(app: FastAPI) -> None:
    paths = get_runtime_config().paths
    repo_root = paths.repo_root
    dist_dir = paths.webapp_dist_dir
    assets_dir = paths.webapp_assets_dir
    existing_paths = {getattr(route, 'path', None) for route in app.routes}

    if assets_dir.exists() and '/assets' not in existing_paths:
        app.mount('/assets', StaticFiles(directory=assets_dir), name='assets')

    if '/' in existing_paths:
        return

    @app.get('/', include_in_schema=False)
    def index() -> FileResponse:
        index_path = paths.webapp_dist_index
        if index_path.exists():
            return FileResponse(index_path)
        fallback = paths.webapp_fallback_index
        return FileResponse(fallback)
