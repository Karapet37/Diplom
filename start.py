# start.py

import argparse

from agent_system.runtime_config import get_runtime_config


def main() -> None:
    runtime = get_runtime_config()
    parser = argparse.ArgumentParser(description='Persona Graph Agent launcher')
    parser.add_argument('--host', default=runtime.settings.host, help='Host for the backend')
    parser.add_argument('--port', type=int, default=runtime.settings.port, help='Port for the backend')
    args = parser.parse_args()

    try:
        import uvicorn

        try:
            from src.web.combined_app import create_combined_app as create_runtime_app
        except Exception:
            from agent_system.api import create_app as create_runtime_app
    except Exception as exc:
        print(f"Web API startup error: {exc}. Install: pip install -e .[dev]")
        return

    uvicorn.run(create_runtime_app(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
