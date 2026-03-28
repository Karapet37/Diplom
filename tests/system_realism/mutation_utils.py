from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from agent_system.graph_store import GraphStore
from agent_system.runtime_config import get_runtime_config

from .models import DialogueCase, DialogueObservation, EvolutionScenario, MutationAction, MutationRecord
from .runtime_launcher import RuntimeLauncher


@contextmanager
def _patched_memory_root(memory_root: Path) -> Iterator[None]:
    previous = os.environ.get('COGNITIVE_MEMORY_ROOT')
    try:
        os.environ['COGNITIVE_MEMORY_ROOT'] = str(Path(memory_root).resolve())
        get_runtime_config()
        yield
    finally:
        if previous is None:
            os.environ.pop('COGNITIVE_MEMORY_ROOT', None)
        else:
            os.environ['COGNITIVE_MEMORY_ROOT'] = previous


class LiveSystemOperator:
    def __init__(
        self,
        *,
        launcher: RuntimeLauncher,
        memory_root: Path,
        persona_name: str,
        persona_slug: str,
        base_session_id: str,
        language: str = 'en',
        request_timeout_s: float = 90.0,
    ) -> None:
        self.launcher = launcher
        self.memory_root = Path(memory_root).resolve()
        self.persona_name = persona_name
        self.persona_slug = persona_slug
        self.base_session_id = base_session_id
        self.language = language
        self.request_timeout_s = float(request_timeout_s or 90.0)
        self.state: dict[str, Any] = {
            'persona_name': persona_name,
            'persona_slug': persona_slug,
            'base_session_id': base_session_id,
            'language': language,
        }
        self.refresh_persona_node_state()

    def refresh_persona_node_state(self) -> None:
        with _patched_memory_root(self.memory_root):
            node = GraphStore().get_node(self.persona_name)
        if isinstance(node, dict):
            self.state['persona_node_id'] = str(node.get('id') or '')
            self.state['persona_node_name'] = str(node.get('name') or '')

    def _resolve(self, payload: dict[str, Any], key: str, *, state_key_field: str | None = None, default: Any = '') -> Any:
        if state_key_field:
            state_key = str(payload.get(state_key_field) or '').strip()
            if state_key:
                return self.state.get(state_key, default)
        return payload.get(key, default)

    def _latest_persona_revision(self) -> int:
        response = self.launcher.request('GET', f'/api/cognitive/personalities/{self.persona_slug}/revisions', timeout_s=8.0)
        if not response.ok or not isinstance(response.json_body, dict):
            return 0
        revisions = list(response.json_body.get('revisions') or [])
        latest = 0
        for item in revisions:
            try:
                latest = max(latest, int(dict(item).get('revision') or 0))
            except Exception:  # noqa: BLE001
                continue
        self.state['latest_persona_revision'] = latest
        return latest

    def probe_case(self, case: DialogueCase, *, session_id: str = '') -> DialogueObservation:
        actual_session = str(session_id or self.base_session_id).strip() or self.base_session_id
        payload = {
            'session_id': actual_session,
            'message': case.prompt,
            'selected_persona': self.persona_name,
            'language': self.language,
        }
        response = self.launcher.request(
            'POST',
            '/api/cognitive/chat/respond',
            json_payload=payload,
            timeout_s=self.request_timeout_s,
        )
        return DialogueObservation(case=case, request_payload=payload, response=response)

    def execute(self, action: MutationAction, *, session_id: str = '') -> MutationRecord:
        started = time.perf_counter()
        try:
            result = self._execute_inner(action, session_id=session_id)
            latency_ms = (time.perf_counter() - started) * 1000.0
            return MutationRecord(
                action=action,
                ok=bool(result.get('ok')),
                latency_ms=latency_ms,
                details=dict(result),
                error=str(result.get('error') or ''),
            )
        except Exception as exc:  # noqa: BLE001
            return MutationRecord(
                action=action,
                ok=False,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                details={},
                error=str(exc),
            )

    def _execute_inner(self, action: MutationAction, *, session_id: str = '') -> dict[str, Any]:
        payload = dict(action.payload or {})
        action_type = str(action.action_type or '').strip()
        if action_type == 'capture_persona_revision':
            revision = self._latest_persona_revision()
            state_key = str(payload.get('state_key') or '').strip()
            if state_key:
                self.state[state_key] = revision
            return {'ok': revision >= 0, 'revision': revision, 'state_key': state_key}

        if action_type == 'chat_fact_injection':
            actual_session = str(session_id or self.base_session_id).strip() or self.base_session_id
            message = str(payload.get('message') or '').strip()
            response = self.launcher.request(
                'POST',
                '/api/cognitive/chat/respond',
                json_payload={
                    'session_id': actual_session,
                    'message': message,
                    'selected_persona': self.persona_name,
                    'language': self.language,
                },
                timeout_s=self.request_timeout_s,
            )
            latest = self._latest_persona_revision()
            assistant_reply = ''
            if isinstance(response.json_body, dict):
                assistant_reply = str(response.json_body.get('assistant_reply') or '').strip()
            if not assistant_reply:
                assistant_reply = str(response.text or '').strip()
            return {
                'ok': response.ok,
                'status_code': response.status_code,
                'assistant_reply': assistant_reply,
                'latest_revision': latest,
                'response': response.to_dict(),
            }

        if action_type == 'graph_create_node':
            response = self.launcher.request(
                'POST',
                '/api/cognitive/graph/nodes',
                json_payload={
                    'name': str(payload.get('name') or '').strip(),
                    'node_type': str(payload.get('node_type') or 'CONCEPT').strip(),
                    'aliases': list(payload.get('aliases') or []),
                    'description': str(payload.get('description') or '').strip(),
                    'facts': list(payload.get('facts') or []),
                    'translation_line': str(payload.get('translation_line') or '').strip(),
                },
                timeout_s=20.0,
            )
            node = dict((response.json_body or {}).get('node') or {}) if isinstance(response.json_body, dict) else {}
            state_key = str(payload.get('state_key') or '').strip()
            if state_key and node.get('id'):
                self.state[state_key] = str(node.get('id'))
            return {'ok': response.ok, 'response': response.to_dict(), 'node': node, 'state_key': state_key}

        if action_type == 'graph_connect':
            from_id = str(self._resolve(payload, 'from_id', state_key_field='from_state_key') or '').strip()
            to_id = str(self._resolve(payload, 'to_id', state_key_field='to_state_key') or '').strip()
            response = self.launcher.request(
                'POST',
                '/api/cognitive/graph/edges',
                json_payload={
                    'from_id': from_id,
                    'to_id': to_id,
                    'relation_type': str(payload.get('relation_type') or 'RELATED_TO').strip(),
                    'weight': float(payload.get('weight') or 0.78),
                    'confidence': float(payload.get('confidence') or 0.82),
                },
                timeout_s=20.0,
            )
            edge = dict((response.json_body or {}).get('result', {}).get('edge') or {}) if isinstance(response.json_body, dict) else {}
            if edge.get('id'):
                edge_state_key = str(payload.get('state_key') or '').strip()
                if edge_state_key:
                    self.state[edge_state_key] = str(edge.get('id'))
            return {'ok': response.ok, 'response': response.to_dict(), 'edge': edge}

        if action_type == 'graph_patch_node':
            node_id = str(self._resolve(payload, 'node_id', state_key_field='node_id_state') or '').strip()
            with _patched_memory_root(self.memory_root):
                result = GraphStore().patch_node(
                    node_id,
                    description=payload.get('description'),
                    facts=list(payload.get('facts') or []) if 'facts' in payload else None,
                    translation_line=payload.get('translation_line'),
                    context_patch=dict(payload.get('context_patch') or {}) if 'context_patch' in payload else None,
                    lifecycle_state=payload.get('lifecycle_state'),
                )
            return {'ok': bool(result.get('ok')), 'result': result, 'node_id': node_id}

        if action_type == 'graph_delete_node':
            node_id = str(self._resolve(payload, 'node_id', state_key_field='node_id_state') or '').strip()
            response = self.launcher.request('DELETE', f'/api/cognitive/graph/nodes/{node_id}', timeout_s=20.0)
            return {'ok': response.ok, 'response': response.to_dict(), 'node_id': node_id}

        if action_type == 'graph_delete_edge':
            edge_id = str(self._resolve(payload, 'edge_id', state_key_field='edge_id_state') or '').strip()
            response = self.launcher.request('DELETE', f'/api/cognitive/graph/edges/{edge_id}', timeout_s=20.0)
            return {'ok': response.ok, 'response': response.to_dict(), 'edge_id': edge_id}

        if action_type == 'graph_merge_nodes':
            primary_id = str(self._resolve(payload, 'primary_id', state_key_field='primary_state_key') or '').strip()
            secondary_id = str(self._resolve(payload, 'secondary_id', state_key_field='secondary_state_key') or '').strip()
            response = self.launcher.request(
                'POST',
                '/api/cognitive/graph/nodes/merge',
                json_payload={'primary_id': primary_id, 'secondary_id': secondary_id},
                timeout_s=20.0,
            )
            return {'ok': response.ok, 'response': response.to_dict(), 'primary_id': primary_id, 'secondary_id': secondary_id}

        if action_type == 'graph_quarantine_node':
            node_id = str(self._resolve(payload, 'node_id', state_key_field='node_id_state') or '').strip()
            response = self.launcher.request(
                'POST',
                f'/api/cognitive/graph/nodes/{node_id}/state',
                json_payload={
                    'lifecycle_state': str(payload.get('lifecycle_state') or 'suspect').strip(),
                    'reason': str(payload.get('reason') or 'system_realism_quarantine').strip(),
                },
                timeout_s=20.0,
            )
            return {'ok': response.ok, 'response': response.to_dict(), 'node_id': node_id}

        if action_type == 'persona_restore_revision':
            revision = self._resolve(payload, 'revision', state_key_field='revision_state', default=0)
            revision_number = int(revision or 0)
            response = self.launcher.request(
                'POST',
                f'/api/cognitive/personalities/{self.persona_slug}/restore/{revision_number}',
                timeout_s=20.0,
            )
            latest = self._latest_persona_revision()
            return {'ok': response.ok, 'response': response.to_dict(), 'restored_revision': revision_number, 'latest_revision': latest}

        if action_type == 'graph_snapshot':
            with _patched_memory_root(self.memory_root):
                result = GraphStore().snapshot_graph(reason=str(payload.get('reason') or 'system_realism'))
            state_key = str(payload.get('state_key') or '').strip()
            if state_key and result.get('snapshot_path'):
                self.state[state_key] = str(result.get('snapshot_path'))
            return {'ok': bool(result.get('ok')), 'result': result}

        if action_type == 'graph_restore_snapshot':
            snapshot_path = str(self._resolve(payload, 'snapshot_path', state_key_field='snapshot_state') or '').strip()
            with _patched_memory_root(self.memory_root):
                result = GraphStore().restore_snapshot(snapshot_path)
            return {'ok': bool(result.get('ok')), 'result': result}

        if action_type == 'probe_runtime_health':
            response = self.launcher.probe_runtime_health(timeout_s=8.0)
            return {'ok': response.ok, 'response': response.to_dict()}

        raise ValueError(f'Unsupported mutation action type: {action_type}')

    def run_scenario(self, scenario: EvolutionScenario, *, session_id: str = '') -> dict[str, Any]:
        actual_session = str(session_id or self.base_session_id).strip() or self.base_session_id
        setup = [self.execute(action, session_id=actual_session) for action in scenario.setup_actions]
        self.refresh_persona_node_state()
        probes = [self.probe_case(case, session_id=actual_session) for case in scenario.probe_cases]
        cleanup = [self.execute(action, session_id=actual_session) for action in scenario.cleanup_actions]
        self.refresh_persona_node_state()
        return {
            'scenario': scenario,
            'setup_records': setup,
            'probe_observations': probes,
            'cleanup_records': cleanup,
            'state_snapshot': dict(self.state),
        }
