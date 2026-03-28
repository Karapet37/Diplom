from __future__ import annotations

from pathlib import Path

from tests.system_realism.models import HttpCallRecord
from tests.system_realism.runtime_launcher import RuntimeLauncher, _probable_failure_reason, choose_free_port


def test_choose_free_port_returns_tcp_port() -> None:
    port = choose_free_port()
    assert isinstance(port, int)
    assert port > 0


def test_runtime_launcher_uses_real_project_entrypoint() -> None:
    launcher = RuntimeLauncher(
        repo_root=Path('/tmp/repo'),
        profile='local-demo',
        host='127.0.0.1',
        port=8123,
        api_only=True,
    )
    assert launcher.command[1] == 'start.py'
    assert '--profile' in launcher.command
    assert '--api-only' in launcher.command
    assert launcher.command[0]


def test_runtime_launcher_reachability_helpers_hit_expected_paths() -> None:
    launcher = RuntimeLauncher(
        repo_root=Path('/tmp/repo'),
        profile='local-demo',
        host='127.0.0.1',
        port=8124,
    )
    seen: list[tuple[str, str]] = []

    def fake_request(method: str, path: str, *, json_payload=None, timeout_s: float = 0.0):  # type: ignore[no-untyped-def]
        seen.append((method, path))
        return HttpCallRecord(method=method, path=path, url=f'http://x{path}', status_code=200, ok=True)

    launcher.request = fake_request  # type: ignore[method-assign]
    launcher.probe_root()
    launcher.probe_runtime_health()
    launcher.probe_surface_health()
    launcher.probe_chat(selected_persona='Dr. Aram Petrosyan')

    assert seen == [
        ('GET', '/'),
        ('GET', '/api/cognitive/health'),
        ('GET', '/api/health'),
        ('POST', '/api/cognitive/chat/respond'),
    ]


def test_probable_failure_reason_detects_missing_uvicorn() -> None:
    reason = _probable_failure_reason(["Runtime startup error: No module named 'uvicorn'."])
    assert reason == 'missing_uvicorn_dependency'
