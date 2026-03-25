from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RealismRunConfig:
    profile: str = 'local-demo'
    host: str = '127.0.0.1'
    port: int = 0
    startup_timeout_s: float = 40.0
    request_timeout_s: float = 20.0
    readiness_poll_interval_s: float = 0.4
    api_only: bool = False
    strict: bool = False
    memory_root: Path | None = None
    output_root: Path | None = None
    report_tag: str = 'canonical'


@dataclass(slots=True)
class DialogueCase:
    case_id: str
    category: str
    prompt: str
    trait_probe: str = ''
    expected_traits: list[str] = field(default_factory=list)
    forbidden_failure_patterns: list[str] = field(default_factory=list)
    target_style_clues: list[str] = field(default_factory=list)
    generic_llm_failure_signals: list[str] = field(default_factory=list)
    persona_success_signals: list[str] = field(default_factory=list)
    expects_memory_from: str = ''
    consistency_group: str = ''
    notes: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'case_id': self.case_id,
            'category': self.category,
            'prompt': self.prompt,
            'trait_probe': self.trait_probe,
            'expected_traits': list(self.expected_traits),
            'forbidden_failure_patterns': list(self.forbidden_failure_patterns),
            'target_style_clues': list(self.target_style_clues),
            'generic_llm_failure_signals': list(self.generic_llm_failure_signals),
            'persona_success_signals': list(self.persona_success_signals),
            'expects_memory_from': self.expects_memory_from,
            'consistency_group': self.consistency_group,
            'notes': self.notes,
        }


@dataclass(slots=True)
class HttpCallRecord:
    method: str
    path: str
    url: str
    status_code: int = 0
    ok: bool = False
    latency_ms: float = 0.0
    text: str = ''
    json_body: dict[str, Any] | list[Any] | None = None
    error: str = ''
    headers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'method': self.method,
            'path': self.path,
            'url': self.url,
            'status_code': int(self.status_code or 0),
            'ok': bool(self.ok),
            'latency_ms': round(float(self.latency_ms or 0.0), 3),
            'text': self.text,
            'json_body': self.json_body,
            'error': self.error,
            'headers': dict(self.headers),
        }


@dataclass(slots=True)
class StartupDiagnosis:
    startup_attempted: bool = False
    startup_success: bool = False
    process_exited_early: bool = False
    ready_probe: HttpCallRecord | None = None
    root_probe: HttpCallRecord | None = None
    api_health_probe: HttpCallRecord | None = None
    startup_time_ms: float = 0.0
    probable_failure_reason: str = ''
    log_tail: list[str] = field(default_factory=list)
    command: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'startup_attempted': bool(self.startup_attempted),
            'startup_success': bool(self.startup_success),
            'process_exited_early': bool(self.process_exited_early),
            'ready_probe': self.ready_probe.to_dict() if self.ready_probe is not None else {},
            'root_probe': self.root_probe.to_dict() if self.root_probe is not None else {},
            'api_health_probe': self.api_health_probe.to_dict() if self.api_health_probe is not None else {},
            'startup_time_ms': round(float(self.startup_time_ms or 0.0), 3),
            'probable_failure_reason': self.probable_failure_reason,
            'log_tail': list(self.log_tail),
            'command': list(self.command),
        }


@dataclass(slots=True)
class PersonaMaterializationRecord:
    ok: bool
    name: str
    slug: str
    head_dir: str
    required_files: dict[str, bool]
    graph_sync_visible: bool
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'ok': bool(self.ok),
            'name': self.name,
            'slug': self.slug,
            'head_dir': self.head_dir,
            'required_files': dict(self.required_files),
            'graph_sync_visible': bool(self.graph_sync_visible),
            'summary': dict(self.summary),
        }


@dataclass(slots=True)
class DialogueObservation:
    case: DialogueCase
    request_payload: dict[str, Any]
    response: HttpCallRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            'case': self.case.to_dict(),
            'request_payload': dict(self.request_payload),
            'response': self.response.to_dict(),
        }
