from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from .models import HttpCallRecord, StartupDiagnosis


def choose_free_port(host: str = '127.0.0.1') -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _resolve_runtime_python(repo_root: Path) -> str:
    explicit = str(os.getenv('COGNITIVE_REALISM_PYTHON', '') or '').strip()
    if explicit:
        return explicit
    candidates = (
        Path(repo_root) / '.venv' / 'bin' / 'python',
        Path(repo_root) / '.venv' / 'Scripts' / 'python.exe',
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    return sys.executable


def _maybe_json(text: str) -> dict[str, Any] | list[Any] | None:
    raw = str(text or '').strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _probable_failure_reason(log_tail: list[str]) -> str:
    joined = '\n'.join(log_tail).lower()
    if "no module named 'uvicorn'" in joined:
        return 'missing_uvicorn_dependency'
    if "no module named 'fastapi'" in joined:
        return 'missing_fastapi_dependency'
    if 'runtime startup error:' in joined:
        return 'runtime_bootstrap_error'
    if 'application bootstrap error:' in joined:
        return 'application_bootstrap_error'
    if 'address already in use' in joined:
        return 'port_conflict'
    if 'no configured gguf model is available' in joined:
        return 'local_model_roles_missing'
    if 'module not found' in joined or 'importerror' in joined:
        return 'missing_dependency'
    if 'traceback' in joined:
        return 'startup_traceback'
    if 'timed out' in joined:
        return 'startup_timeout'
    return 'unknown_startup_failure'


class RuntimeLauncher:
    def __init__(
        self,
        *,
        repo_root: Path,
        profile: str,
        host: str,
        port: int,
        env_overrides: dict[str, str] | None = None,
        api_only: bool = False,
        log_path: Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.profile = profile
        self.host = host
        self.port = int(port)
        self.env_overrides = dict(env_overrides or {})
        self.api_only = bool(api_only)
        self.log_path = log_path
        self.process: subprocess.Popen[str] | None = None
        self._logs: list[str] = []
        self._pump_thread: threading.Thread | None = None
        self.python_executable = _resolve_runtime_python(self.repo_root)
        self.command = [
            self.python_executable,
            'start.py',
            '--profile',
            self.profile,
            '--host',
            self.host,
            '--port',
            str(self.port),
        ]
        if self.api_only:
            self.command.append('--api-only')

    @property
    def base_url(self) -> str:
        return f'http://{self.host}:{self.port}'

    @property
    def logs(self) -> list[str]:
        return list(self._logs)

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def __enter__(self) -> RuntimeLauncher:
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.stop()

    def last_logs(self, limit: int = 80) -> list[str]:
        return self.logs[-max(1, int(limit or 80)) :]

    def _pump_logs(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        log_handle = self.log_path.open('a', encoding='utf-8') if self.log_path is not None else None
        try:
            for line in self.process.stdout:
                clean = line.rstrip('\n')
                self._logs.append(clean)
                if len(self._logs) > 400:
                    self._logs = self._logs[-400:]
                if log_handle is not None:
                    log_handle.write(clean + '\n')
                    log_handle.flush()
        finally:
            if log_handle is not None:
                log_handle.close()

    def start(self) -> None:
        base_env = dict(os.environ)
        base_env.update(self.env_overrides)
        base_env.setdefault('PYTHONUNBUFFERED', '1')
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            self.command,
            cwd=self.repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=base_env,
        )
        self._pump_thread = threading.Thread(target=self._pump_logs, daemon=True)
        self._pump_thread.start()

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self._pump_thread is not None:
            self._pump_thread.join(timeout=2)

    def probe_root(self, *, timeout_s: float = 5.0) -> HttpCallRecord:
        return self.request('GET', '/', timeout_s=timeout_s)

    def probe_runtime_health(self, *, timeout_s: float = 5.0) -> HttpCallRecord:
        return self.request('GET', '/api/cognitive/health', timeout_s=timeout_s)

    def probe_surface_health(self, *, timeout_s: float = 5.0) -> HttpCallRecord:
        return self.request('GET', '/api/health', timeout_s=timeout_s)

    def probe_chat(
        self,
        *,
        session_id: str = 'runtime-launcher-probe',
        message: str = 'State your runtime name in one short sentence.',
        selected_persona: str = '',
        language: str = 'en',
        timeout_s: float = 10.0,
    ) -> HttpCallRecord:
        return self.request(
            'POST',
            '/api/cognitive/chat/respond',
            json_payload={
                'session_id': session_id,
                'message': message,
                'selected_persona': selected_persona,
                'language': language,
            },
            timeout_s=timeout_s,
        )

    def reachability_snapshot(
        self,
        *,
        selected_persona: str = '',
        timeout_s: float = 8.0,
    ) -> dict[str, HttpCallRecord]:
        return {
            'root': self.probe_root(timeout_s=timeout_s),
            'runtime_health': self.probe_runtime_health(timeout_s=timeout_s),
            'surface_health': self.probe_surface_health(timeout_s=timeout_s),
            'chat': self.probe_chat(selected_persona=selected_persona, timeout_s=timeout_s),
        }

    def request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        timeout_s: float = 10.0,
    ) -> HttpCallRecord:
        url = self.base_url + path
        payload_bytes: bytes | None = None
        headers = {'Accept': 'application/json, text/html;q=0.9, */*;q=0.8'}
        if json_payload is not None:
            payload_bytes = json.dumps(json_payload, ensure_ascii=False).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        req = request.Request(url, data=payload_bytes, method=method.upper(), headers=headers)
        started = time.perf_counter()
        try:
            with request.urlopen(req, timeout=timeout_s) as response:
                raw = response.read().decode('utf-8', errors='replace')
                latency_ms = (time.perf_counter() - started) * 1000.0
                return HttpCallRecord(
                    method=method.upper(),
                    path=path,
                    url=url,
                    status_code=int(response.status),
                    ok=200 <= int(response.status) < 300,
                    latency_ms=latency_ms,
                    text=raw,
                    json_body=_maybe_json(raw),
                    headers={key: value for key, value in response.headers.items()},
                )
        except error.HTTPError as exc:
            raw = exc.read().decode('utf-8', errors='replace')
            latency_ms = (time.perf_counter() - started) * 1000.0
            return HttpCallRecord(
                method=method.upper(),
                path=path,
                url=url,
                status_code=int(exc.code),
                ok=False,
                latency_ms=latency_ms,
                text=raw,
                json_body=_maybe_json(raw),
                error=str(exc),
                headers={key: value for key, value in exc.headers.items()},
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - started) * 1000.0
            return HttpCallRecord(
                method=method.upper(),
                path=path,
                url=url,
                status_code=0,
                ok=False,
                latency_ms=latency_ms,
                text='',
                json_body=None,
                error=str(exc),
            )

    def wait_until_ready(self, *, timeout_s: float, poll_interval_s: float) -> StartupDiagnosis:
        diagnosis = StartupDiagnosis(
            startup_attempted=True,
            startup_success=False,
            command=list(self.command),
        )
        started = time.perf_counter()
        while (time.perf_counter() - started) < timeout_s:
            if self.process is not None and self.process.poll() is not None:
                diagnosis.process_exited_early = True
                break
            ready_probe = self.probe_runtime_health(timeout_s=min(3.0, poll_interval_s + 2.0))
            diagnosis.ready_probe = ready_probe
            if ready_probe.ok and isinstance(ready_probe.json_body, dict) and bool(ready_probe.json_body.get('ok')):
                diagnosis.startup_success = True
                diagnosis.startup_time_ms = (time.perf_counter() - started) * 1000.0
                diagnosis.root_probe = self.probe_root(timeout_s=5.0)
                diagnosis.api_health_probe = self.probe_surface_health(timeout_s=5.0)
                return diagnosis
            time.sleep(poll_interval_s)
        diagnosis.startup_time_ms = (time.perf_counter() - started) * 1000.0
        diagnosis.log_tail = self.last_logs(80)
        diagnosis.probable_failure_reason = _probable_failure_reason(diagnosis.log_tail + ([diagnosis.ready_probe.error] if diagnosis.ready_probe and diagnosis.ready_probe.error else []))
        diagnosis.root_probe = self.probe_root(timeout_s=3.0)
        diagnosis.api_health_probe = self.probe_surface_health(timeout_s=3.0)
        return diagnosis
