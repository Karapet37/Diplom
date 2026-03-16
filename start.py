# start.py

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description='Persona Graph Agent launcher')
    parser.add_argument('--host', default=os.getenv('WEB_HOST', '127.0.0.1'), help='Host for the backend')
    parser.add_argument('--port', type=int, default=int(os.getenv('WEB_PORT', '8008')), help='Port for the backend')
    args = parser.parse_args()

    try:
        import uvicorn

        from agent_system.api import create_app
    except Exception as exc:
        print(f"Web API startup error: {exc}. Install: pip install -e .[dev]")
        return

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
