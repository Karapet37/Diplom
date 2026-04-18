from __future__ import annotations

import json
import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from .duplicate_resolver import normalize_name
from .models import TraceLearningPolicy

_TRACE_POLICY_BATCH_CANDIDATES = (4, 8, 12, 16)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _json_logger() -> logging.Logger:
    logger = logging.getLogger('agent_system.observability')
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


@dataclass(slots=True)
class StageTiming:
    name: str
    duration_ms: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'duration_ms': round(float(self.duration_ms), 3),
            'meta': dict(self.meta),
        }


@dataclass(slots=True)
class RequestTrace:
    request_id: str
    request_type: str
    route: str
    session_id: str = ''
    started_at: str = field(default_factory=_utc_now)
    stages: list[StageTiming] = field(default_factory=list)
    status: str = 'ok'
    fallback_used: bool = False
    fallback_reason: str = ''
    context_tokens: int = 0
    persona_name: str = ''
    current_entity: str = ''
    request_meta: dict[str, Any] = field(default_factory=dict)
    response_meta: dict[str, Any] = field(default_factory=dict)
    finished_at: str = ''
    total_ms: float = 0.0

    def add_stage(self, name: str, duration_ms: float, *, meta: dict[str, Any] | None = None) -> None:
        self.stages.append(StageTiming(name=name, duration_ms=duration_ms, meta=dict(meta or {})))

    def to_dict(self) -> dict[str, Any]:
        stage_timings_ms = {
            stage.name: round(float(stage.duration_ms), 3)
            for stage in self.stages
        }
        return {
            'request_id': self.request_id,
            'request_type': self.request_type,
            'route': self.route,
            'session_id': self.session_id,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'status': self.status,
            'fallback_used': self.fallback_used,
            'fallback_reason': self.fallback_reason,
            'context_tokens': self.context_tokens,
            'persona_name': self.persona_name,
            'current_entity': self.current_entity,
            'request_meta': dict(self.request_meta),
            'response_meta': dict(self.response_meta),
            'total_ms': round(float(self.total_ms), 3),
            'stages': [stage.to_dict() for stage in self.stages],
            'stage_timings_ms': stage_timings_ms,
        }


class ObservabilityStore:
    def __init__(self, *, max_recent_traces: int = 200) -> None:
        self._lock = Lock()
        self._recent_traces: deque[dict[str, Any]] = deque(maxlen=max_recent_traces)
        self._counters: Counter[str] = Counter()
        self._stage_total_ms: Counter[str] = Counter()
        self._stage_counts: Counter[str] = Counter()
        self._stage_max_ms: dict[str, float] = {}
        self._context_token_samples: deque[int] = deque(maxlen=max_recent_traces)
        self._logger = _json_logger()

    def start_trace(
        self,
        *,
        request_type: str,
        route: str,
        session_id: str = '',
        request_id: str = '',
        request_meta: dict[str, Any] | None = None,
    ) -> RequestTrace:
        trace = RequestTrace(
            request_id=str(request_id or uuid4().hex),
            request_type=str(request_type or 'runtime').strip() or 'runtime',
            route=str(route or '/').strip() or '/',
            session_id=str(session_id or '').strip(),
            request_meta=dict(request_meta or {}),
        )
        self._logger.info(
            json.dumps(
                {
                    'event': 'request_started',
                    'request_id': trace.request_id,
                    'request_type': trace.request_type,
                    'route': trace.route,
                    'session_id': trace.session_id,
                    'request_meta': trace.request_meta,
                    'started_at': trace.started_at,
                },
                ensure_ascii=False,
            )
        )
        return trace

    def finish_trace(
        self,
        trace: RequestTrace,
        *,
        status: str = 'ok',
        fallback_used: bool = False,
        fallback_reason: str = '',
        context_tokens: int = 0,
        persona_name: str = '',
        current_entity: str = '',
        response_meta: dict[str, Any] | None = None,
    ) -> RequestTrace:
        trace.status = str(status or 'ok').strip() or 'ok'
        trace.fallback_used = bool(fallback_used)
        trace.fallback_reason = str(fallback_reason or '').strip()
        trace.context_tokens = max(int(context_tokens or 0), 0)
        trace.persona_name = str(persona_name or '').strip()
        trace.current_entity = str(current_entity or '').strip()
        trace.response_meta = dict(response_meta or {})
        trace.finished_at = _utc_now()
        trace.total_ms = sum(stage.duration_ms for stage in trace.stages)
        trace_payload = trace.to_dict()
        with self._lock:
            self._recent_traces.appendleft(trace_payload)
            self._counters[f'{trace.request_type}_requests_total'] += 1
            if trace.fallback_used:
                self._counters[f'{trace.request_type}_fallback_total'] += 1
            if trace.context_tokens > 0:
                self._context_token_samples.append(trace.context_tokens)
            for stage in trace.stages:
                self._stage_total_ms[stage.name] += float(stage.duration_ms)
                self._stage_counts[stage.name] += 1
                previous = float(self._stage_max_ms.get(stage.name, 0.0))
                self._stage_max_ms[stage.name] = max(previous, float(stage.duration_ms))
        self._logger.info(json.dumps({'event': 'request_finished', **trace_payload}, ensure_ascii=False))
        return trace

    def time_stage(
        self,
        trace: RequestTrace,
        stage_name: str,
        fn: Callable[[], Any],
        *,
        meta_builder: Callable[[Any], dict[str, Any]] | None = None,
    ) -> Any:
        started = perf_counter()
        result = fn()
        duration_ms = (perf_counter() - started) * 1000.0
        trace.add_stage(
            stage_name,
            duration_ms,
            meta=meta_builder(result) if meta_builder is not None else {},
        )
        return result

    def record_graph_write(self, *, reason: str, ok: bool, node_count: int, edge_count: int) -> None:
        with self._lock:
            self._counters['graph_writes_total'] += 1
            self._counters[f'graph_write_reason::{str(reason or "unknown").strip() or "unknown"}'] += 1
            if not ok:
                self._counters['graph_write_errors_total'] += 1
            self._counters['graph_nodes_last'] = int(node_count or 0)
            self._counters['graph_edges_last'] = int(edge_count or 0)

    def record_rebuild_schedule(self, *, scheduled: bool, reason: str, status: str = '') -> None:
        with self._lock:
            if scheduled:
                self._counters['rebuild_scheduled_total'] += 1
            else:
                self._counters['rebuild_skipped_total'] += 1
            if reason:
                self._counters[f'rebuild_reason::{reason}'] += 1
            if status:
                self._counters[f'rebuild_schedule_status::{status}'] += 1

    def record_rebuild_status(self, status: str) -> None:
        clean = str(status or '').strip()
        if not clean:
            return
        with self._lock:
            self._counters[f'rebuild_status::{clean}'] += 1

    def record_rethink_outcome(
        self,
        *,
        preview_only: bool,
        processed: int,
        results: list[dict[str, Any]] | None = None,
    ) -> None:
        rows = list(results or [])
        ok_count = sum(1 for item in rows if bool(item.get('ok')))
        fail_count = max(len(rows) - ok_count, 0)
        with self._lock:
            if preview_only:
                self._counters['rethink_preview_total'] += 1
            else:
                self._counters['rethink_apply_total'] += 1
            self._counters['rethink_nodes_processed_total'] += max(int(processed or 0), 0)
            self._counters['rethink_success_total'] += ok_count
            self._counters['rethink_failure_total'] += fail_count

    def recent_traces(self, *, limit: int = 20, session_id: str = '') -> list[dict[str, Any]]:
        clean_session = str(session_id or '').strip()
        with self._lock:
            rows = list(self._recent_traces)
        if clean_session:
            rows = [row for row in rows if str(row.get('session_id') or '').strip() == clean_session]
        return rows[: max(int(limit or 20), 1)]

    def learned_route_policy(
        self,
        *,
        session_id: str = '',
        selected_route: str = '',
        max_batch_size: int = 16,
    ) -> TraceLearningPolicy:
        clean_route = str(selected_route or '').strip()
        clean_session = str(session_id or '').strip()
        if not clean_route:
            return TraceLearningPolicy()
        with self._lock:
            rows = list(self._recent_traces)
        matching = [
            row for row in rows
            if str(self._trace_selected_route(row) or '').strip() == clean_route
        ]
        if not matching:
            return TraceLearningPolicy(selected_route=clean_route)

        session_rows = [
            row for row in matching
            if clean_session and str(row.get('session_id') or '').strip() == clean_session
        ]
        global_rows = [
            row for row in matching
            if not clean_session or str(row.get('session_id') or '').strip() != clean_session
        ]
        available_rows = session_rows + global_rows
        batch_size = self._select_trace_batch_size(available_rows, max_batch_size=max_batch_size)
        sampled_rows = self._sample_trace_rows(session_rows, global_rows, batch_size=batch_size)
        if not sampled_rows:
            return TraceLearningPolicy(selected_route=clean_route)

        signal_rows = [row for row in sampled_rows if self._trace_has_learning_signal(row)]
        weighted_rows = [
            *(row for row in sampled_rows if self._trace_has_learning_signal(row)),
            *sampled_rows,
        ]
        truncation_hits = sum(1 for row in weighted_rows if self._trace_output_problem(row))
        fallback_hits = sum(1 for row in weighted_rows if bool(row.get('fallback_used')))
        route_mismatch_hits = sum(1 for row in weighted_rows if not bool(dict(row.get('response_meta') or {}).get('validation_ok', True)))
        no_grounding_hits = sum(1 for row in weighted_rows if str(row.get('fallback_reason') or '').strip() == 'no_grounding')
        repeated_intro_hits = sum(1 for row in weighted_rows if self._trace_reply_looks_intro_loop(row))
        failure_density = round(len(signal_rows) / max(len(sampled_rows), 1), 3)

        guidance_lines: list[str] = []
        reason_codes: list[str] = []
        output_budget_boost = 0

        if truncation_hits:
            guidance_lines.append(
                'Recent similar traces were cut off. Answer directly, keep the persona introduction brief, and end on a complete sentence.'
            )
            reason_codes.append('trace_policy:truncation_cluster')
            output_budget_boost = max(output_budget_boost, 160 if truncation_hits >= 3 else 96)
        if repeated_intro_hits:
            guidance_lines.append(
                'Do not restart from the same self-introduction. Continue the scene and move to the concrete action quickly.'
            )
            reason_codes.append('trace_policy:persona_intro_loop')
            output_budget_boost = max(output_budget_boost, 96)
        if route_mismatch_hits or (clean_route == 'hypothetical_roleplay' and no_grounding_hits):
            guidance_lines.append(
                'Preserve the selected route strictly. Do not degrade into a generic limitation or clarification when the user is still inside the same scenario.'
            )
            reason_codes.append('trace_policy:route_drift')
        if fallback_hits and clean_route in {'persona_graph_reasoning', 'hypothetical_roleplay'}:
            guidance_lines.append(
                'When certainty is limited, stay inside the persona or hypothetical frame instead of dropping into assistant-style hedging.'
            )
            reason_codes.append('trace_policy:persona_frame_preservation')

        return TraceLearningPolicy(
            selected_route=clean_route,
            batch_size=batch_size,
            sampled_trace_count=len(sampled_rows),
            session_trace_count=len(session_rows),
            global_trace_count=len(global_rows),
            signal_trace_count=len(signal_rows),
            failure_density=failure_density,
            output_budget_boost=output_budget_boost,
            route_guidance_lines=guidance_lines[:4],
            reason_codes=reason_codes[:6],
        )

    def _trace_selected_route(self, row: dict[str, Any]) -> str:
        response_meta = dict(row.get('response_meta') or {})
        request_meta = dict(row.get('request_meta') or {})
        return str(
            response_meta.get('selected_route')
            or request_meta.get('selected_route')
            or ''
        ).strip()

    def _trace_output_problem(self, row: dict[str, Any]) -> bool:
        response_meta = dict(row.get('response_meta') or {})
        return bool(response_meta.get('output_truncated') or response_meta.get('output_budget_too_small'))

    def _trace_reply_looks_intro_loop(self, row: dict[str, Any]) -> bool:
        response_meta = dict(row.get('response_meta') or {})
        request_meta = dict(row.get('request_meta') or {})
        if not self._trace_output_problem(row):
            return False
        raw_text = str(request_meta.get('raw_text') or '').strip()
        if not raw_text:
            return False
        lowered = normalize_name(raw_text)
        return '...тупой' in lowered or 'как ты' in lowered or 'как будешь' in lowered

    def _trace_has_learning_signal(self, row: dict[str, Any]) -> bool:
        response_meta = dict(row.get('response_meta') or {})
        if bool(row.get('fallback_used')):
            return True
        if not bool(response_meta.get('validation_ok', True)):
            return True
        if bool(response_meta.get('output_truncated') or response_meta.get('output_budget_too_small')):
            return True
        return False

    def _select_trace_batch_size(self, rows: list[dict[str, Any]], *, max_batch_size: int) -> int:
        if not rows:
            return 0
        capped_max = max(1, min(int(max_batch_size or 16), max(_TRACE_POLICY_BATCH_CANDIDATES)))
        available = min(len(rows), capped_max)
        signal_count = sum(1 for row in rows[:available] if self._trace_has_learning_signal(row))
        failure_density = signal_count / max(available, 1)
        if failure_density >= 0.5:
            target = 4
        elif failure_density >= 0.25:
            target = 8
        elif failure_density >= 0.12:
            target = 12
        else:
            target = 16
        target = min(target, capped_max, len(rows))
        viable = [size for size in _TRACE_POLICY_BATCH_CANDIDATES if size <= target and size <= len(rows)]
        return viable[-1] if viable else max(1, min(len(rows), target))

    def _sample_trace_rows(
        self,
        session_rows: list[dict[str, Any]],
        global_rows: list[dict[str, Any]],
        *,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        if batch_size <= 0:
            return []
        session_signal_rows = [row for row in session_rows if self._trace_has_learning_signal(row)]
        global_signal_rows = [row for row in global_rows if self._trace_has_learning_signal(row)]

        sampled: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        def _append_rows(rows: list[dict[str, Any]]) -> None:
            for row in rows:
                trace_id = str(row.get('request_id') or '').strip()
                if trace_id and trace_id in seen_ids:
                    continue
                if trace_id:
                    seen_ids.add(trace_id)
                sampled.append(row)
                if len(sampled) >= batch_size:
                    return

        signal_budget = max(1, batch_size // 2)
        interleaved_signals: list[dict[str, Any]] = []
        for index in range(max(len(session_signal_rows), len(global_signal_rows))):
            if index < len(session_signal_rows):
                interleaved_signals.append(session_signal_rows[index])
            if index < len(global_signal_rows):
                interleaved_signals.append(global_signal_rows[index])
        _append_rows(interleaved_signals[:signal_budget])

        interleaved_all: list[dict[str, Any]] = []
        for index in range(max(len(session_rows), len(global_rows))):
            if index < len(session_rows):
                interleaved_all.append(session_rows[index])
            if index < len(global_rows):
                interleaved_all.append(global_rows[index])
        _append_rows(interleaved_all)
        return sampled[:batch_size]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            stage_totals = dict(self._stage_total_ms)
            stage_counts = dict(self._stage_counts)
            stage_max = dict(self._stage_max_ms)
            context_samples = list(self._context_token_samples)
            recent_trace_count = len(self._recent_traces)
        chat_requests = max(int(counters.get('chat_requests_total') or 0), 0)
        chat_fallbacks = max(int(counters.get('chat_fallback_total') or 0), 0)
        stage_metrics: dict[str, Any] = {}
        for stage_name, total_ms in stage_totals.items():
            count = max(int(stage_counts.get(stage_name) or 0), 1)
            stage_metrics[stage_name] = {
                'count': count,
                'avg_ms': round(float(total_ms) / count, 3),
                'max_ms': round(float(stage_max.get(stage_name, 0.0)), 3),
            }
        context_avg = round(sum(context_samples) / len(context_samples), 3) if context_samples else 0.0
        return {
            'counters': counters,
            'rates': {
                'fallback_rate': round(chat_fallbacks / chat_requests, 6) if chat_requests else 0.0,
            },
            'context_tokens': {
                'last': context_samples[-1] if context_samples else 0,
                'avg': context_avg,
                'max': max(context_samples) if context_samples else 0,
                'samples': len(context_samples),
            },
            'stage_timings_ms': stage_metrics,
            'recent_trace_count': recent_trace_count,
        }


_STORE = ObservabilityStore()


def get_observability_store() -> ObservabilityStore:
    return _STORE
