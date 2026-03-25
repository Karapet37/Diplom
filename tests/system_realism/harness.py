from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .dialogue_cases import canonical_dialogue_cases
from .evaluator import evaluate_realism
from .models import DialogueObservation, RealismRunConfig
from .persona_fixture import canonical_test_persona, materialize_canonical_persona
from .reporting import create_report_run_dir, write_json_report, write_markdown_report, write_text_artifact
from .runtime_launcher import RuntimeLauncher, choose_free_port


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@contextmanager
def _patched_environ(overrides: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _looks_like_html(text: str) -> bool:
    lowered = str(text or '').lower()
    return '<html' in lowered or '<!doctype html' in lowered


def _collect_runtime_diagnostics(launcher: RuntimeLauncher) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for key, path in (
        ('metrics', '/api/cognitive/debug/metrics'),
        ('traces', '/api/cognitive/debug/traces?limit=50'),
        ('graph_health', '/api/cognitive/debug/graph-health'),
        ('runtime_status', '/api/cognitive/debug/runtime-status'),
    ):
        record = launcher.request('GET', path, timeout_s=8.0)
        diagnostics[f'{key}_probe'] = record.to_dict()
        if isinstance(record.json_body, dict):
            if key == 'metrics':
                diagnostics[key] = dict(record.json_body.get('metrics') or {})
            elif key == 'traces':
                diagnostics[key] = list(record.json_body.get('traces') or [])
            elif key == 'graph_health':
                diagnostics[key] = dict(record.json_body.get('graph_health') or {})
            elif key == 'runtime_status':
                diagnostics[key] = dict(record.json_body.get('runtime_status') or {})
                diagnostics['runtime_operator_messages'] = list(record.json_body.get('operator_messages') or [])
    return diagnostics


def _collect_persona_endpoint(launcher: RuntimeLauncher, persona_slug: str) -> dict[str, Any] | None:
    record = launcher.request('GET', f'/api/cognitive/personalities/{persona_slug}', timeout_s=8.0)
    if record.ok and isinstance(record.json_body, dict):
        return dict(record.json_body)
    return None


def run_realism_suite(config: RealismRunConfig) -> dict[str, Any]:
    repo_root = _repo_root()
    memory_root = Path(config.memory_root or (repo_root / 'runtime' / 'system_realism_memory')).resolve()
    output_root = Path(config.output_root or (repo_root / 'runtime' / 'system_realism_reports')).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = create_report_run_dir(output_root, tag=config.report_tag)
    host = config.host
    port = int(config.port or choose_free_port(host))
    env_overrides = {
        'COGNITIVE_MEMORY_ROOT': str(memory_root),
        'WEB_HOST': host,
        'WEB_PORT': str(port),
        'COGNITIVE_ENABLE_BACKGROUND_REBUILD': '0',
        'COGNITIVE_INCLUDE_SIDE_EFFECTS': '1',
    }

    with _patched_environ(env_overrides):
        persona_record = materialize_canonical_persona(memory_root)
        persona = canonical_test_persona()

    launcher = RuntimeLauncher(
        repo_root=repo_root,
        profile=config.profile,
        host=host,
        port=port,
        env_overrides=env_overrides,
        api_only=config.api_only,
        log_path=run_dir / 'server.log',
    )

    startup = None
    observations: list[DialogueObservation] = []
    diagnostics: dict[str, Any] = {}
    reachability: dict[str, Any] = {}
    persona_endpoint: dict[str, Any] | None = None
    session_id = f'system-realism-{config.report_tag}'

    try:
        launcher.start()
        startup = launcher.wait_until_ready(
            timeout_s=config.startup_timeout_s,
            poll_interval_s=config.readiness_poll_interval_s,
        )
        reachability = {
            'root_reachable': bool(startup.root_probe and startup.root_probe.ok),
            'root_html': bool(startup.root_probe and _looks_like_html(startup.root_probe.text)),
            'health_reachable': bool(startup.ready_probe and startup.ready_probe.ok),
            'api_json_valid': bool(startup.ready_probe and isinstance(startup.ready_probe.json_body, dict)),
            'chat_alive': False,
        }

        if startup.startup_success:
            cases = canonical_dialogue_cases()
            for case in cases:
                payload = {
                    'session_id': session_id,
                    'message': case.prompt,
                    'selected_persona': persona.name,
                    'language': 'en',
                }
                response = launcher.request('POST', '/api/cognitive/chat/respond', json_payload=payload, timeout_s=config.request_timeout_s)
                if response.ok and isinstance(response.json_body, dict):
                    reachability['chat_alive'] = True
                observations.append(DialogueObservation(case=case, request_payload=payload, response=response))

            diagnostics = _collect_runtime_diagnostics(launcher)
            persona_endpoint = _collect_persona_endpoint(launcher, persona.slug)
    finally:
        launcher.stop()

    if startup is None:
        raise RuntimeError('Startup diagnosis was not initialized.')

    evaluation = evaluate_realism(
        startup=startup,
        reachability=reachability,
        persona=canonical_test_persona(),
        persona_materialization=persona_record.to_dict(),
        persona_endpoint=persona_endpoint,
        dialogue_observations=observations,
        diagnostics=diagnostics,
    )
    report_payload = {
        'run': {
            'profile': config.profile,
            'host': host,
            'port': port,
            'api_only': bool(config.api_only),
            'strict': bool(config.strict),
            'memory_root': str(memory_root),
            'output_dir': str(run_dir),
        },
        'startup': startup.to_dict(),
        'reachability': dict(reachability),
        'persona_materialization': persona_record.to_dict(),
        'persona_endpoint': persona_endpoint or {},
        'dialogues': [item.to_dict() for item in observations],
        'diagnostics': diagnostics,
        'evaluation': evaluation,
    }
    json_path = write_json_report(run_dir, report_payload)
    markdown_path = write_markdown_report(run_dir, report_payload)
    write_text_artifact(run_dir, 'log_tail.txt', '\n'.join(launcher.logs[-120:]))
    report_payload['artifacts'] = {
        'json_report': str(json_path),
        'markdown_report': str(markdown_path),
        'server_log': str(run_dir / 'server.log'),
        'log_tail': str(run_dir / 'log_tail.txt'),
    }
    write_json_report(run_dir, report_payload)
    return report_payload
