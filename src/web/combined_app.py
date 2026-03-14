from __future__ import annotations

from fastapi import FastAPI

from roaches_viz.roaches_viz.api import create_router
from src.web.api import attach_frontend_routes


def create_combined_app() -> FastAPI:
    app = FastAPI(title='Unified Graph Runtime')
    app.include_router(create_router(), prefix='/api/cognitive')

    @app.get('/api/health')
    def health() -> dict[str, object]:
        return {'ok': True, 'surface': 'combined'}

    attach_frontend_routes(app)
    return app


app = create_combined_app()
