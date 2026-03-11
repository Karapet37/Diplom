# start.py

import argparse
import os

from src.utils.env_loader import load_local_env


def main() -> None:
    load_local_env()

    parser = argparse.ArgumentParser(description="Unified Cognitive Graph Workspace launcher")
    parser.add_argument(
        "--web-api",
        action="store_true",
        help="Run FastAPI backend for the unified web UI (default behavior)",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("WEB_HOST", "127.0.0.1"),
        help="Host for the unified backend (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("WEB_PORT", "8008")),
        help="Port for the unified backend (default: 8008)",
    )
    args = parser.parse_args()

    _ = args.web_api
    try:
        import uvicorn
        from src.web.combined_app import app as web_app
    except Exception as exc:
        print(f"Web API startup error: {exc}. Install: pip install -e .[dev]")
        return

    uvicorn.run(web_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
