from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def attach_frontend_routes(app: FastAPI) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dist_dir = repo_root / 'webapp' / 'dist'
    assets_dir = dist_dir / 'assets'

    if assets_dir.exists():
        app.mount('/assets', StaticFiles(directory=assets_dir), name='assets')

    @app.get('/', include_in_schema=False)
    def index() -> FileResponse:
        index_path = dist_dir / 'index.html'
        if index_path.exists():
            return FileResponse(index_path)
        fallback = repo_root / 'webapp' / 'index.html'
        return FileResponse(fallback)
