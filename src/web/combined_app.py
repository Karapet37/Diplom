from __future__ import annotations

from agent_system.api import create_app
from src.web.api import attach_frontend_routes


def create_combined_app():
    app = create_app()
    attach_frontend_routes(app)
    return app


app = create_combined_app()
