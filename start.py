# start.py

from __future__ import annotations

import argparse
import json
from typing import Any

from agent_system.runtime_config import (
    RuntimeBootstrapReport,
    bootstrap_runtime_environment,
    get_runtime_config,
    list_runtime_profiles,
)


def _runtime_warnings(config, *, api_only: bool) -> list[str]:
    warnings: list[str] = []
    if not api_only and config.features.enable_frontend_root and not config.paths.webapp_dist_index.exists():
        warnings.append(
            f"Frontend build was not found at {config.paths.webapp_dist_index}. "
            "Run `cd webapp && npm install && npm run build`, or start with `--api-only`."
        )
    if not config.paths.webapp_fallback_index.exists() and not api_only:
        warnings.append(
            f"Frontend fallback file is missing at {config.paths.webapp_fallback_index}. "
            "Frontend routes will be degraded."
        )

    try:
        from src.utils.local_llm_provider import list_model_advisors

        diagnostics = list_model_advisors()
        advisor_map = {str(item.get('role') or ''): item for item in diagnostics.get('advisors', [])}
        required_roles = {
            config.roles.chat,
            config.roles.extraction,
            config.roles.persona_synthesis,
            config.roles.rethink,
        }
        missing_roles = sorted(role for role in required_roles if role and not advisor_map.get(role, {}).get('available'))
        if missing_roles:
            warnings.append(
                "No configured GGUF model is available for runtime roles: "
                + ', '.join(missing_roles)
                + ". Set LOCAL_*_GGUF_MODEL paths or LOCAL_MODELS_DIR."
            )
    except Exception as exc:  # pragma: no cover - diagnostic-only branch
        warnings.append(f'Local model diagnostics unavailable: {exc}')

    return warnings


def _print_bootstrap_summary(report: RuntimeBootstrapReport, config, *, api_only: bool) -> None:
    print(f"[runtime] profile={report.profile} host={config.settings.host} port={config.settings.port}")
    print(f"[runtime] memory_root={config.paths.memory_root}")
    print(f"[runtime] frontend_mode={'api-only' if api_only else 'combined'}")
    if report.config_file:
        print(f"[runtime] profile_config={report.config_file}")
    if report.env_file:
        print(f"[runtime] env_file={report.env_file}")

    warnings = list(report.warnings) + _runtime_warnings(config, api_only=api_only)
    if warnings:
        print('[runtime] warnings:')
        for item in warnings:
            print(f'  - {item}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Persona Graph Agent launcher')
    parser.add_argument('--profile', default='', help='Runtime profile name: development, local-demo, local-heavy, server')
    parser.add_argument('--env-file', default='', help='Optional env file layered on top of the selected profile')
    parser.add_argument('--config', default='', help='Optional YAML config template file for runtime defaults')
    parser.add_argument('--host', default='', help='Override host for the backend')
    parser.add_argument('--port', type=int, default=0, help='Override port for the backend')
    parser.add_argument('--reload', action='store_true', help='Enable uvicorn reload for development')
    parser.add_argument('--api-only', action='store_true', help='Serve API only without mounting frontend routes')
    parser.add_argument('--check', action='store_true', help='Print startup diagnostics and exit')
    parser.add_argument('--print-config', action='store_true', help='Print resolved runtime config as JSON and exit')
    parser.add_argument('--list-profiles', action='store_true', help='List available runtime profiles and exit')
    parser.add_argument('--web-api', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.list_profiles:
        profiles = list_runtime_profiles()
        if not profiles:
            print('No runtime profiles found.')
            return
        for item in profiles:
            print(f"{item.name}\t{item.description}\t{item.path}")
        return

    bootstrap = bootstrap_runtime_environment(
        profile=args.profile,
        env_file=args.env_file,
        config_file=args.config,
    )
    runtime = get_runtime_config()

    host = args.host or runtime.settings.host
    port = args.port or runtime.settings.port
    api_only = bool(args.api_only)

    _print_bootstrap_summary(bootstrap, runtime, api_only=api_only)

    if args.print_config:
        payload: dict[str, Any] = {
            'bootstrap': bootstrap.to_dict(),
            'runtime': runtime.to_dict(),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if args.check:
        return

    try:
        import uvicorn
    except Exception as exc:
        print(f"Runtime startup error: {exc}. Install backend deps with `pip install -e .[dev]`.")
        return

    try:
        if api_only:
            from agent_system.api import create_app as create_runtime_app
        else:
            try:
                from src.web.combined_app import create_combined_app as create_runtime_app
            except Exception:
                from agent_system.api import create_app as create_runtime_app
    except Exception as exc:
        print(f"Application bootstrap error: {exc}. Try `python start.py --check` to inspect the runtime setup.")
        return

    uvicorn.run(create_runtime_app(), host=host, port=port, reload=bool(args.reload))


if __name__ == '__main__':
    main()
