# start.py

import argparse
import os

from src.utils.env_loader import load_local_env


def main() -> None:
    # Load local .env before constructing graph/LLM components.
    load_local_env()

    parser = argparse.ArgumentParser(
        description="Autonomous Graph Workspace launcher"
    )
    parser.add_argument(
        "--web-api",
        action="store_true",
        help="Run FastAPI backend for React web UI (default behavior)",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("WEB_HOST", "127.0.0.1"),
        help="Host for --web-api (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("WEB_PORT", "8008")),
        help="Port for --web-api (default: 8008)",
    )
    args = parser.parse_args()

    _ = args.web_api  # retained for backward-compatible CLI flag
    try:
        import uvicorn
        from src.web.api import app as web_app
    except Exception as exc:
        print(f"Web API startup error: {exc}. Install: pip install fastapi uvicorn python-multipart")
        return

    uvicorn.run(web_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
