from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .runtime_config import get_runtime_config


@dataclass(slots=True)
class FailurePolicy:
    code: str
    status_code: int
    severity: str
    degraded_mode: str
    operator_action: str
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            'code': self.code,
            'status_code': int(self.status_code),
            'severity': self.severity,
            'degraded_mode': self.degraded_mode,
            'operator_action': self.operator_action,
            'retryable': bool(self.retryable),
        }


FAILURE_POLICIES: dict[str, FailurePolicy] = {
    'dependency_unavailable': FailurePolicy(
        code='dependency_unavailable',
        status_code=503,
        severity='error',
        degraded_mode='dependency_degraded',
        operator_action='Check local dependencies, model paths, and runtime profile configuration.',
        retryable=False,
    ),
    'storage_write_failed': FailurePolicy(
        code='storage_write_failed',
        status_code=503,
        severity='critical',
        degraded_mode='storage_protected',
        operator_action='Inspect storage permissions and archive snapshots before retrying mutations.',
        retryable=False,
    ),
    'mutation_rejected': FailurePolicy(
        code='mutation_rejected',
        status_code=409,
        severity='error',
        degraded_mode='mutation_blocked',
        operator_action='Review the proposed mutation, validation result, and available rollback snapshot.',
        retryable=False,
    ),
    'recovery_failed': FailurePolicy(
        code='recovery_failed',
        status_code=500,
        severity='critical',
        degraded_mode='recovery_failed',
        operator_action='Use stored snapshots or revisions to recover state before further writes.',
        retryable=False,
    ),
}


class RuntimeFailure(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        policy: FailurePolicy | None = None,
    ) -> None:
        self.code = str(code or 'dependency_unavailable').strip() or 'dependency_unavailable'
        self.message = str(message or 'Runtime failure').strip() or 'Runtime failure'
        self.details = dict(details or {})
        self.policy = policy or FAILURE_POLICIES.get(self.code) or FAILURE_POLICIES['dependency_unavailable']
        super().__init__(self.message)

    @property
    def status_code(self) -> int:
        return int(self.policy.status_code)

    def to_dict(self) -> dict[str, Any]:
        return {
            'code': self.code,
            'message': self.message,
            'severity': self.policy.severity,
            'degraded_mode': self.policy.degraded_mode,
            'operator_action': self.policy.operator_action,
            'retryable': bool(self.policy.retryable),
            'details': dict(self.details),
        }


class DependencyUnavailableFailure(RuntimeFailure):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(code='dependency_unavailable', message=message, details=details)


class StorageWriteFailure(RuntimeFailure):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(code='storage_write_failed', message=message, details=details)


class MutationRejectedFailure(RuntimeFailure):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(code='mutation_rejected', message=message, details=details)


class RecoveryFailure(RuntimeFailure):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(code='recovery_failed', message=message, details=details)


@dataclass(slots=True)
class DegradedMode:
    code: str
    summary: str
    detail: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'code': self.code,
            'summary': self.summary,
            'detail': self.detail,
        }


def runtime_status_snapshot() -> dict[str, Any]:
    config = get_runtime_config()
    modes: list[DegradedMode] = []

    if config.features.enable_frontend_root and not config.paths.webapp_dist_index.exists() and not config.paths.webapp_fallback_index.exists():
        modes.append(
            DegradedMode(
                code='frontend_unavailable',
                summary='Frontend surface is unavailable.',
                detail='Build webapp/dist or serve the API in api-only mode.',
            )
        )

    try:
        from src.utils import local_llm_provider

        diagnostics = local_llm_provider.list_model_advisors()
        advisor_map = {str(item.get('role') or ''): item for item in diagnostics.get('advisors', [])}
        required_roles = {
            str(config.roles.chat or '').strip(),
            str(config.roles.extraction or '').strip(),
            str(config.roles.persona_synthesis or '').strip(),
            str(config.roles.rethink or '').strip(),
        }
        missing_roles = sorted(role for role in required_roles if role and not advisor_map.get(role, {}).get('available'))
        if getattr(local_llm_provider, 'Llama', None) is None:
            modes.append(
                DegradedMode(
                    code='llama_cpp_missing',
                    summary='Local inference bindings are unavailable.',
                    detail='Install llama-cpp-python for local-first chat and extraction.',
                )
            )
        if missing_roles:
            modes.append(
                DegradedMode(
                    code='llm_roles_missing',
                    summary='One or more required model roles are unavailable.',
                    detail='Missing roles: ' + ', '.join(missing_roles),
                )
            )
    except Exception as exc:
        modes.append(
            DegradedMode(
                code='llm_diagnostics_unavailable',
                summary='LLM diagnostics could not be loaded.',
                detail=str(exc),
            )
        )

    mode = 'degraded' if modes else 'full'
    return {
        'mode': mode,
        'degraded_modes': [item.to_dict() for item in modes],
    }


def operator_messages_from_status(status: dict[str, Any] | None) -> list[str]:
    payload = dict(status or {})
    messages: list[str] = []
    for item in list(payload.get('degraded_modes') or []):
        if not isinstance(item, dict):
            continue
        summary = str(item.get('summary') or '').strip()
        detail = str(item.get('detail') or '').strip()
        if summary and detail:
            messages.append(f'{summary} {detail}')
        elif summary:
            messages.append(summary)
    return messages
