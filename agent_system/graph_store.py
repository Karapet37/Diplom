from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from .duplicate_resolver import (
    canonical_name,
    combined_similarity,
    merge_aliases,
    normalize_name,
    relevance_decay,
    score_node,
    should_merge,
    similarity_bucket,
)
from .memory_layers import (
    append_graph_lifecycle_events,
    list_graph_snapshots,
    load_graph_lifecycle_log,
    load_graph_snapshot,
    record_graph_snapshot,
)
from .models import ENTITY_TYPES, GRAPH_NODE_LIFECYCLE_STATES, GraphHealthMetrics, GraphQuality
from .observability import get_observability_store
from .reliability import MutationRejectedFailure, RecoveryFailure, StorageWriteFailure
from .runtime_config import get_runtime_config

GRAPH_NODE_TYPES = set(ENTITY_TYPES)
GRAPH_LIFECYCLE_STATES = set(GRAPH_NODE_LIFECYCLE_STATES)
GRAPH_VISIBLE_SEARCH_STATES = {'active', 'weak'}
QUALITY_ALPHA = 0.45
QUALITY_BETA = 0.35
QUALITY_GAMMA = 0.2
RISKY_GRAPH_MUTATION_REASONS = {
    'create_node',
    'patch_node',
    'review_node_state',
    'delete_node',
    'delete_edge',
    'merge_nodes_manual',
    'restore_graph_snapshot',
    'connect_nodes',
}


def repo_root() -> Path:
    return get_runtime_config().paths.repo_root


def memory_root() -> Path:
    return get_runtime_config().paths.memory_root


def graphs_dir() -> Path:
    return get_runtime_config().paths.graphs_dir


def heads_dir() -> Path:
    return get_runtime_config().paths.heads_dir


def personality_proposals_dir() -> Path:
    return get_runtime_config().paths.proposals_dir


def graph_nodes_path() -> Path:
    return graphs_dir() / 'nodes.json'


def graph_edges_path() -> Path:
    return graphs_dir() / 'edges.json'


def normalize_personality_name(value: str) -> str:
    token = ''.join(char.lower() if char.isalnum() else '_' for char in str(value or '').strip())
    token = '_'.join(part for part in token.split('_') if part)
    return token or 'unknown_head'


def personality_profile_path(name: str) -> Path:
    return heads_dir() / normalize_personality_name(name) / 'meta.json'


def personality_graph_path(name: str) -> Path:
    return heads_dir() / normalize_personality_name(name) / 'local_graph.json'


def personality_index_path() -> Path:
    path = heads_dir() / 'index.json'
    if not path.exists():
        write_json(path, {'heads': []})
    return path


def personality_proposal_path(name: str) -> Path:
    return personality_proposals_dir() / f'{normalize_personality_name(name)}.json'


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp_path.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    tmp_path.write_text(str(text or ''), encoding='utf-8')
    tmp_path.replace(path)


def _slug(value: str) -> str:
    return normalize_personality_name(value)


def _edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(edge.get('from') or '').strip(),
        str(edge.get('type') or '').strip(),
        str(edge.get('to') or '').strip(),
    )


class GraphStore:
    def load_nodes(self) -> list[dict[str, Any]]:
        payload = load_json(graph_nodes_path(), [])
        rows = payload.get('nodes') if isinstance(payload, dict) else payload
        return [self._normalize_node(dict(item)) for item in list(rows or []) if isinstance(item, dict)]

    def load_edges(self) -> list[dict[str, Any]]:
        payload = load_json(graph_edges_path(), [])
        rows = payload.get('edges') if isinstance(payload, dict) else payload
        return [self._normalize_edge(dict(item)) for item in list(rows or []) if isinstance(item, dict)]

    def load_graph(self) -> dict[str, Any]:
        nodes = self.load_nodes()
        edges = self.load_edges()
        diagnostics = self.graph_diagnostics(nodes, edges)
        return {
            'nodes': nodes,
            'edges': edges,
            'diagnostics': diagnostics,
            'clusters': diagnostics.get('clusters') or [],
        }

    def validate_graph(
        self,
        nodes: list[dict[str, Any]] | None = None,
        edges: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        state = self._prepare_graph_state(
            self.load_nodes() if nodes is None else nodes,
            self.load_edges() if edges is None else edges,
            apply_hygiene=False,
        )
        return {
            'ok': not state['errors'],
            'errors': state['errors'],
            'warnings': state['warnings'],
            'node_count': len(state['nodes']),
            'edge_count': len(state['edges']),
            'quality': asdict(self.graph_quality(state['nodes'], state['edges'])),
        }

    def graph_quality(
        self,
        nodes: list[dict[str, Any]] | None = None,
        edges: list[dict[str, Any]] | None = None,
    ) -> GraphQuality:
        node_rows = self.load_nodes() if nodes is None else [self._normalize_node(node) for node in nodes if isinstance(node, dict)]
        edge_rows = self.load_edges() if edges is None else [self._normalize_edge(edge) for edge in edges if isinstance(edge, dict)]
        active_rows = self._visible_nodes(node_rows, states=GRAPH_VISIBLE_SEARCH_STATES)
        if not active_rows:
            return GraphQuality(relevance=0.0, redundancy=0.0, connectivity=0.0, score=0.0)

        relevance = round(
            min(sum(min(score_node(node), 1.0) for node in active_rows) / max(len(active_rows), 1), 1.0),
            6,
        )
        low_value_ratio = sum(
            1 for node in active_rows if float(node.get('importance') or 0.0) < 0.05 and int(node.get('frequency') or 0) < 2
        ) / max(len(active_rows), 1)
        duplicate_metrics = self._duplicate_metrics(active_rows)
        duplicate_ratio = min(
            (duplicate_metrics['merge_count'] + duplicate_metrics['review_count']) / max(len(active_rows), 1),
            1.0,
        )
        redundancy = round(min((0.7 * duplicate_ratio) + (0.3 * low_value_ratio), 1.0), 6)

        node_ids = {str(node.get('id') or '') for node in active_rows}
        valid_edges = [
            edge for edge in edge_rows if str(edge.get('from') or '') in node_ids and str(edge.get('to') or '') in node_ids
        ]
        connected_nodes = {
            endpoint
            for edge in valid_edges
            for endpoint in (str(edge.get('from') or ''), str(edge.get('to') or ''))
            if endpoint
        }
        connected_ratio = len(connected_nodes) / max(len(active_rows), 1)
        edge_density = min(len(valid_edges) / max(len(active_rows) - 1, 1), 1.0)
        connectivity = round(min((0.5 * connected_ratio) + (0.5 * edge_density), 1.0), 6)

        score = round((QUALITY_ALPHA * relevance) - (QUALITY_BETA * redundancy) + (QUALITY_GAMMA * connectivity), 6)
        return GraphQuality(
            relevance=relevance,
            redundancy=redundancy,
            connectivity=connectivity,
            score=score,
        )

    def graph_diagnostics(
        self,
        nodes: list[dict[str, Any]] | None = None,
        edges: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        node_rows = self.load_nodes() if nodes is None else [self._normalize_node(node) for node in nodes if isinstance(node, dict)]
        edge_rows = self.load_edges() if edges is None else [self._normalize_edge(edge) for edge in edges if isinstance(edge, dict)]
        visible_nodes = self._visible_nodes(node_rows, states=GRAPH_VISIBLE_SEARCH_STATES)
        review_pool = [
            node
            for node in node_rows
            if self._normalize_lifecycle_state(node.get('lifecycle_state') or 'active') not in {'archived', 'merged'}
        ]
        node_ids = {str(node.get('id') or '') for node in visible_nodes}
        valid_edges = [
            edge for edge in edge_rows if str(edge.get('from') or '') in node_ids and str(edge.get('to') or '') in node_ids
        ]
        connected_nodes = {
            endpoint
            for edge in valid_edges
            for endpoint in (str(edge.get('from') or ''), str(edge.get('to') or ''))
            if endpoint
        }
        orphan_nodes = [node for node in visible_nodes if str(node.get('id') or '') and str(node.get('id') or '') not in connected_nodes]
        duplicate_metrics = self._duplicate_metrics(review_pool)
        review_marked_nodes = [
            node
            for node in review_pool
            if str(dict(node.get('context') or {}).get('review_reason') or '').strip().lower() == 'duplicate_candidate'
        ]
        low_value_nodes = [
            node
            for node in visible_nodes
            if float(node.get('importance') or 0.0) < 0.05 and int(node.get('frequency') or 0) < 2
        ]
        lifecycle_counts = self._lifecycle_counts(node_rows)
        archive_counts = self._archived_lifecycle_counts()
        cluster_summaries = self._cluster_summaries(node_rows)
        metrics = GraphHealthMetrics(
            node_count=len(node_rows),
            edge_count=len(edge_rows),
            active_node_count=lifecycle_counts['active'],
            weak_node_count=lifecycle_counts['weak'],
            suspect_node_count=lifecycle_counts['suspect'],
            archived_node_count=archive_counts['archived'],
            merged_node_count=archive_counts['merged'],
            duplicate_candidates=duplicate_metrics['merge_count'],
            duplicate_review_candidates=max(duplicate_metrics['review_count'], len(review_marked_nodes)),
            duplicate_rate=min(
                (duplicate_metrics['merge_count'] + max(duplicate_metrics['review_count'], len(review_marked_nodes))) / max(len(review_pool), 1),
                1.0,
            ),
            orphan_nodes=len(orphan_nodes),
            orphan_rate=min(len(orphan_nodes) / max(len(visible_nodes), 1), 1.0),
            average_relation_density=min(len(valid_edges) / max(len(visible_nodes), 1), 1.0) if visible_nodes else 0.0,
            low_value_nodes=len(low_value_nodes),
            summary_nodes=sum(1 for node in node_rows if str(node.get('id') or '').startswith('summary:')),
            cluster_count=len(cluster_summaries),
            quality=self.graph_quality(node_rows, edge_rows),
        )
        payload = metrics.to_dict()
        payload.update({
            'duplicate_candidate_samples': duplicate_metrics['merge_samples'][:10],
            'duplicate_review_samples': (
                duplicate_metrics['review_samples'][:10]
                or [
                    {
                        'left_id': str(node.get('id') or ''),
                        'left_name': str(node.get('name') or ''),
                        'right_id': str(dict(node.get('context') or {}).get('review_target_id') or ''),
                        'right_name': '',
                        'similarity': float(dict(node.get('context') or {}).get('review_similarity') or 0.0),
                        'bucket': 'review',
                    }
                    for node in review_marked_nodes[:10]
                ]
            ),
            'orphan_node_samples': [
                {
                    'id': str(node.get('id') or ''),
                    'name': str(node.get('name') or ''),
                    'type': str(node.get('type') or ''),
                }
                for node in orphan_nodes[:10]
            ],
            'clusters': cluster_summaries,
        })
        return payload

    def snapshot_graph(self, *, reason: str = 'manual') -> dict[str, Any]:
        nodes = self.load_nodes()
        edges = self.load_edges()
        diagnostics = self.graph_diagnostics(nodes, edges)
        path = record_graph_snapshot(nodes=nodes, edges=edges, reason=reason, diagnostics=diagnostics)
        return {
            'ok': True,
            'path': str(path),
            'node_count': len(nodes),
            'edge_count': len(edges),
            'reason': reason,
        }

    def list_snapshots(self, *, limit: int = 16) -> list[dict[str, Any]]:
        return list_graph_snapshots(limit=limit)

    def restore_snapshot(self, snapshot_path: str = '') -> dict[str, Any]:
        candidates = self.list_snapshots(limit=1) if not str(snapshot_path or '').strip() else []
        target_path = str(snapshot_path or (candidates[0]['path'] if candidates else '')).strip()
        payload = load_graph_snapshot(target_path)
        if not isinstance(payload, dict):
            return {'ok': False, 'reason': 'snapshot_not_found', 'snapshot_path': target_path}
        current_snapshot = self.snapshot_graph(reason='pre-restore_graph_snapshot')
        nodes = [self._normalize_node(dict(item)) for item in list(payload.get('nodes') or []) if isinstance(item, dict)]
        edges = [self._normalize_edge(dict(item)) for item in list(payload.get('edges') or []) if isinstance(item, dict)]
        result = self.save_graph(nodes, edges, reason='restore_graph_snapshot', apply_hygiene=False, snapshot_before_write=False)
        result['restored_snapshot_path'] = target_path
        result['pre_restore_snapshot_path'] = current_snapshot.get('path') or ''
        return result

    def _should_snapshot_before_save(self, reason: str, *, apply_hygiene: bool) -> bool:
        clean = str(reason or '').strip()
        return clean in RISKY_GRAPH_MUTATION_REASONS or (apply_hygiene and clean == 'graph_hygiene')

    def _write_graph_bundle(
        self,
        *,
        old_nodes: list[dict[str, Any]],
        old_edges: list[dict[str, Any]],
        new_nodes: list[dict[str, Any]],
        new_edges: list[dict[str, Any]],
        reason: str,
    ) -> None:
        try:
            write_json(graph_nodes_path(), new_nodes)
            write_json(graph_edges_path(), new_edges)
        except Exception as exc:
            try:
                write_json(graph_nodes_path(), old_nodes)
                write_json(graph_edges_path(), old_edges)
            except Exception as rollback_exc:
                raise RecoveryFailure(
                    f'Graph write failed and rollback did not complete for reason={reason}.',
                    details={
                        'reason': reason,
                        'write_error': str(exc),
                        'rollback_error': str(rollback_exc),
                    },
                ) from rollback_exc
            raise StorageWriteFailure(
                f'Graph write failed for reason={reason}. Previous graph state was restored.',
                details={'reason': reason, 'write_error': str(exc)},
            ) from exc

    def save_graph(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        *,
        reason: str = 'sync',
        apply_hygiene: bool = False,
        snapshot_before_write: bool | None = None,
    ) -> dict[str, Any]:
        existing_nodes = self.load_nodes()
        existing_edges = self.load_edges()
        quality_before = self.graph_quality(existing_nodes, existing_edges)
        state = self._prepare_graph_state(nodes, edges, apply_hygiene=apply_hygiene)
        quality_after = self.graph_quality(state['nodes'], state['edges'])
        if not state['nodes'] and existing_nodes and reason != 'clear':
            get_observability_store().record_graph_write(
                reason=reason,
                ok=False,
                node_count=len(existing_nodes),
                edge_count=len(existing_edges),
            )
            return {
                'ok': False,
                'reason': 'skipped_empty_overwrite',
                'node_count': len(existing_nodes),
                'edge_count': len(existing_edges),
                'validation_errors': state['errors'],
                'validation_warnings': state['warnings'],
                'quality_before': asdict(quality_before),
                'quality_after': asdict(quality_after),
            }
        do_snapshot = self._should_snapshot_before_save(reason, apply_hygiene=apply_hygiene) if snapshot_before_write is None else bool(snapshot_before_write)
        snapshot_info: dict[str, Any] = {}
        if do_snapshot and (existing_nodes or existing_edges):
            snapshot_info = self.snapshot_graph(reason=f'pre-{reason}')
        self._write_graph_bundle(
            old_nodes=existing_nodes,
            old_edges=existing_edges,
            new_nodes=state['nodes'],
            new_edges=state['edges'],
            reason=reason,
        )
        if state.get('lifecycle_events'):
            append_graph_lifecycle_events(list(state.get('lifecycle_events') or []), reason=reason)
        get_observability_store().record_graph_write(
            reason=reason,
            ok=True,
            node_count=len(state['nodes']),
            edge_count=len(state['edges']),
        )
        return {
            'ok': True,
            'reason': reason,
            'node_count': len(state['nodes']),
            'edge_count': len(state['edges']),
            'validation_errors': state['errors'],
            'validation_warnings': state['warnings'],
            'merged_nodes': state['merged_nodes'],
            'removed_nodes': state['removed_nodes'],
            'summary_nodes': state['summary_nodes'],
            'lifecycle_events': list(state.get('lifecycle_events') or []),
            'quality_before': asdict(quality_before),
            'quality_after': asdict(quality_after),
            'diagnostics': self.graph_diagnostics(state['nodes'], state['edges']),
            'snapshot_path': str(snapshot_info.get('path') or ''),
            'rollback_available': bool(snapshot_info.get('path')),
        }

    def entity_exists(self, name: str) -> bool:
        return self.get_node(name) is not None

    def get_node(self, name: str) -> dict[str, Any] | None:
        token = normalize_name(name)
        for node in self.load_nodes():
            names = [str(node.get('name') or '')] + [str(item) for item in list(node.get('aliases') or [])]
            if any(normalize_name(candidate) == token for candidate in names):
                return node
        return None

    def get_node_by_id(self, node_id: str) -> dict[str, Any] | None:
        token = str(node_id or '').strip()
        if not token:
            return None
        return next((node for node in self.load_nodes() if str(node.get('id') or '') == token), None)

    def upsert_entity(
        self,
        *,
        name: str,
        entity_type: str,
        aliases: list[str] | None = None,
        description: str = '',
        facts: list[str] | None = None,
        confidence: float = 0.75,
        source: str = 'chat',
        folder: str = '',
        importance: float = 0.7,
        translation_line: str = '',
        context: dict[str, Any] | None = None,
        commit_reason: str = 'upsert_entity',
    ) -> dict[str, Any]:
        nodes = self.load_nodes()
        edges = self.load_edges()
        candidate = self._candidate_node(
            name=name,
            entity_type=entity_type,
            aliases=aliases,
            description=description,
            facts=facts,
            confidence=confidence,
            source=source,
            folder=folder,
            importance=importance,
            translation_line=translation_line,
            context=context,
        )
        node = self._upsert_node(nodes, candidate)
        self._commit(nodes, edges, reason=commit_reason)
        return node

    def create_node(
        self,
        *,
        name: str,
        node_type: str = 'CONCEPT',
        aliases: list[str] | None = None,
        description: str = '',
        facts: list[str] | None = None,
        translation_line: str = '',
        confidence: float = 0.78,
        importance: float = 0.72,
        source: str = 'manual_edit',
    ) -> dict[str, Any]:
        return self.upsert_entity(
            name=name,
            entity_type=node_type,
            aliases=aliases,
            description=description,
            facts=facts,
            confidence=confidence,
            source=source,
            importance=importance,
            translation_line=translation_line,
            context={'source': source, 'edited': True},
            commit_reason='create_node',
        )

    def patch_node(
        self,
        node_id: str,
        *,
        description: str | None = None,
        facts: list[str] | None = None,
        translation_line: str | None = None,
        context_patch: dict[str, Any] | None = None,
        lifecycle_state: str | None = None,
    ) -> dict[str, Any]:
        target = str(node_id or '').strip()
        if not target:
            return {'ok': False, 'reason': 'missing_node_id'}
        nodes = self.load_nodes()
        found = False
        for node in nodes:
            if str(node.get('id') or '') != target:
                continue
            found = True
            if description is not None:
                node['description'] = str(description or '').strip()
            if facts is not None:
                node['facts'] = [str(item).strip() for item in list(facts or []) if str(item).strip()][:16]
            if translation_line is not None:
                node['translation_line'] = str(translation_line or '').strip()
            if context_patch:
                node['context'] = {**dict(node.get('context') or {}), **dict(context_patch or {})}
            if lifecycle_state is not None:
                node['lifecycle_state'] = self._normalize_lifecycle_state(lifecycle_state)
            break
        if not found:
            return {'ok': False, 'reason': 'node_not_found'}
        result = self.save_graph(nodes, self.load_edges(), reason='patch_node', apply_hygiene=False)
        result['node_id'] = target
        return result

    def review_node_state(self, node_id: str, *, lifecycle_state: str, reason: str = 'manual_review') -> dict[str, Any]:
        target = str(node_id or '').strip()
        next_state = self._normalize_lifecycle_state(lifecycle_state)
        if not target:
            return {'ok': False, 'reason': 'missing_node_id'}
        nodes = self.load_nodes()
        node = next((item for item in nodes if str(item.get('id') or '') == target), None)
        if node is None:
            return {'ok': False, 'reason': 'node_not_found'}
        if next_state == 'archived':
            next_nodes = [item for item in nodes if str(item.get('id') or '') != target]
            next_edges = [
                edge
                for edge in self.load_edges()
                if str(edge.get('from') or '') != target and str(edge.get('to') or '') != target
            ]
            append_graph_lifecycle_events(
                [
                    {
                        'node_id': target,
                        'name': str(node.get('name') or ''),
                        'state': 'archived',
                        'source': 'manual_review',
                        'detail': str(reason or 'manual_review'),
                    }
                ],
                reason='review_node_state',
            )
            result = self.save_graph(next_nodes, next_edges, reason='clear' if not next_nodes else 'review_node_state', apply_hygiene=False)
            result['node_id'] = target
            result['lifecycle_state'] = 'archived'
            return result
        return self.patch_node(
            target,
            context_patch={'review_reason': str(reason or 'manual_review')},
            lifecycle_state=next_state,
        )

    def sync_head(
        self,
        *,
        name: str,
        folder: str,
        entity_type: str,
        aliases: list[str] | None = None,
        description: str = '',
        facts: list[str] | None = None,
        knowledge: str = '',
        relations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        nodes = self.load_nodes()
        edges = self.load_edges()
        node = self._upsert_node(
            nodes,
            self._candidate_node(
                name=name,
                entity_type=entity_type,
                aliases=aliases,
                description=description,
                facts=facts,
                confidence=0.82,
                source='head',
                folder=folder,
                importance=0.82,
                context={'source': 'head', 'knowledge': str(knowledge or '').strip()},
            ),
        )
        src_id = str(node.get('id') or '')
        edge_keys = {_edge_key(edge) for edge in edges}
        for relation in list(relations or []):
            if not isinstance(relation, dict):
                continue
            target = str(relation.get('target') or relation.get('to') or '').strip()
            relation_type = str(relation.get('type') or 'RELATED_TO').strip().upper()
            if not target:
                continue
            dst_id = self._ensure_entity(nodes, target, source='head')
            edge = self._normalize_edge(
                {
                    'from': src_id,
                    'to': dst_id,
                    'type': relation_type,
                    'weight': float(relation.get('weight') or 0.8),
                    'confidence': float(relation.get('confidence') or 0.82),
                    'source': 'head',
                }
            )
            key = _edge_key(edge)
            if key not in edge_keys:
                edge_keys.add(key)
                edges.append(edge)
        return self._commit(nodes, edges, reason='sync_head')

    def register_head(
        self,
        *,
        name: str,
        folder: str,
        entity_type: str,
        aliases: list[str] | None = None,
        description: str = '',
        facts: list[str] | None = None,
        knowledge: str = '',
    ) -> dict[str, Any]:
        return self.sync_head(
            name=name,
            folder=folder,
            entity_type=entity_type,
            aliases=aliases,
            description=description or f'Persona head for {name}.',
            facts=facts,
            knowledge=knowledge,
            relations=[],
        )

    def merge_extraction(self, extraction: dict[str, Any], *, source: str = 'session') -> dict[str, Any]:
        nodes = self.load_nodes()
        edges = self.load_edges()
        touched_nodes = 0
        touched_edges = 0
        quarantined_nodes = 0
        name_to_id: dict[str, str] = {}
        lifecycle_events: list[dict[str, Any]] = []
        graph_policy = get_runtime_config().graph

        for entity in list(extraction.get('entities') or []):
            if not isinstance(entity, dict):
                continue
            confidence = float(entity.get('confidence') or 0.0)
            context = {**dict(entity.get('context') or {}), 'source': source}
            if confidence < graph_policy.extraction_quarantine_confidence:
                context = {
                    **context,
                    'review_status': 'quarantine',
                    'review_reason': 'low_confidence_extraction',
                }
                quarantined_nodes += 1
            candidate = self._normalize_node({**entity, 'context': context})
            current = self._upsert_node(nodes, candidate)
            name_to_id[normalize_name(str(entity.get('name') or ''))] = str(current.get('id') or '')
            if str(current.get('lifecycle_state') or '') == 'suspect':
                lifecycle_events.append(
                    {
                        'node_id': str(current.get('id') or ''),
                        'name': str(current.get('name') or ''),
                        'state': 'suspect',
                        'source': source,
                        'detail': str(dict(current.get('context') or {}).get('review_reason') or 'review'),
                    }
                )
            touched_nodes += 1

        edge_keys = {_edge_key(edge) for edge in edges}
        for relation in list(extraction.get('relations') or []):
            if not isinstance(relation, dict):
                continue
            src = str(relation.get('from') or '').strip()
            dst = str(relation.get('to') or '').strip()
            if not src or not dst:
                continue
            src_id = name_to_id.get(normalize_name(src)) or self._ensure_entity(nodes, src, source=source)
            dst_id = name_to_id.get(normalize_name(dst)) or self._ensure_entity(nodes, dst, source=source)
            edge = self._normalize_edge(
                {
                    'from': src_id,
                    'to': dst_id,
                    'type': str(relation.get('type') or 'RELATED_TO').strip().upper(),
                    'weight': float(relation.get('weight') or relation.get('strength') or 0.7),
                    'confidence': float(relation.get('confidence') or 0.7),
                    'source': source,
                }
            )
            key = _edge_key(edge)
            if key in edge_keys:
                continue
            edge_keys.add(key)
            edges.append(edge)
            touched_edges += 1

        result = self._commit(nodes, edges, reason='merge_extraction')
        if lifecycle_events:
            append_graph_lifecycle_events(lifecycle_events, reason='merge_extraction')
        result.update(
            {
                'touched_nodes': touched_nodes,
                'touched_edges': touched_edges,
                'quarantined_nodes': quarantined_nodes,
            }
        )
        return result

    def link_head_relation(
        self,
        *,
        head_name: str,
        relation_type: str,
        target_name: str,
        weight: float = 0.75,
    ) -> dict[str, Any]:
        nodes = self.load_nodes()
        edges = self.load_edges()
        src_id = self._ensure_entity(nodes, head_name, source='head')
        dst_id = self._ensure_entity(nodes, target_name, source='head')
        edge = self._normalize_edge({'from': src_id, 'to': dst_id, 'type': relation_type.upper(), 'weight': weight, 'confidence': 0.8, 'source': 'head'})
        if _edge_key(edge) not in {_edge_key(item) for item in edges}:
            edges.append(edge)
        return self._commit(nodes, edges, reason='link_head_relation')

    def connect_nodes(
        self,
        *,
        from_id: str,
        to_id: str,
        relation_type: str = 'RELATED_TO',
        weight: float = 0.78,
        confidence: float = 0.82,
        source: str = 'manual_edit',
    ) -> dict[str, Any]:
        src = str(from_id or '').strip()
        dst = str(to_id or '').strip()
        if not src or not dst:
            return {'ok': False, 'reason': 'missing_endpoint'}
        nodes = self.load_nodes()
        node_ids = {str(node.get('id') or '') for node in nodes}
        if src not in node_ids or dst not in node_ids:
            return {'ok': False, 'reason': 'missing_node'}
        edges = self.load_edges()
        edge = self._normalize_edge(
            {
                'from': src,
                'to': dst,
                'type': str(relation_type or 'RELATED_TO').strip().upper(),
                'weight': weight,
                'confidence': confidence,
                'source': source,
            }
        )
        if _edge_key(edge) not in {_edge_key(item) for item in edges}:
            edges.append(edge)
        result = self._commit(nodes, edges, reason='connect_nodes')
        result['edge'] = edge
        return result

    def delete_node(self, node_id: str) -> dict[str, Any]:
        target = str(node_id or '').strip()
        if not target:
            return {'ok': False, 'reason': 'missing_node_id'}
        nodes = self.load_nodes()
        node = next((item for item in nodes if str(item.get('id') or '') == target), None)
        if node is None:
            return {'ok': False, 'reason': 'node_not_found'}
        next_nodes = [node for node in nodes if str(node.get('id') or '') != target]
        next_edges = [
            edge
            for edge in self.load_edges()
            if str(edge.get('from') or '') != target and str(edge.get('to') or '') != target
        ]
        append_graph_lifecycle_events(
            [
                {
                    'node_id': target,
                    'name': str(node.get('name') or ''),
                    'state': 'archived',
                    'source': 'manual_delete',
                    'detail': 'delete_node',
                }
            ],
            reason='delete_node',
        )
        reason = 'clear' if not next_nodes else 'delete_node'
        result = self.save_graph(next_nodes, next_edges, reason=reason, apply_hygiene=False)
        result['deleted_node_id'] = target
        return result

    def delete_edge(self, *, edge_id: str = '', from_id: str = '', relation_type: str = '', to_id: str = '') -> dict[str, Any]:
        target_id = str(edge_id or '').strip()
        target_key = (str(from_id or '').strip(), str(relation_type or '').strip().upper(), str(to_id or '').strip())
        edges = self.load_edges()
        next_edges = [
            edge
            for edge in edges
            if not (
                (target_id and str(edge.get('id') or '') == target_id)
                or (all(target_key) and _edge_key(edge) == target_key)
            )
        ]
        if len(next_edges) == len(edges):
            return {'ok': False, 'reason': 'edge_not_found'}
        result = self.save_graph(self.load_nodes(), next_edges, reason='delete_edge', apply_hygiene=False)
        result['deleted_edge_id'] = target_id
        return result

    def merge_nodes_manual(self, *, primary_id: str, secondary_id: str) -> dict[str, Any]:
        left = str(primary_id or '').strip()
        right = str(secondary_id or '').strip()
        if not left or not right or left == right:
            return {'ok': False, 'reason': 'invalid_merge_pair'}
        nodes = self.load_nodes()
        primary = next((node for node in nodes if str(node.get('id') or '') == left), None)
        secondary = next((node for node in nodes if str(node.get('id') or '') == right), None)
        if primary is None or secondary is None:
            return {'ok': False, 'reason': 'node_not_found'}
        self._merge_node(primary, secondary)
        next_nodes = [node for node in nodes if str(node.get('id') or '') != right]
        next_edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for edge in self.load_edges():
            updated = self._normalize_edge(
                {
                    **edge,
                    'from': left if str(edge.get('from') or '') == right else str(edge.get('from') or ''),
                    'to': left if str(edge.get('to') or '') == right else str(edge.get('to') or ''),
                }
            )
            key = _edge_key(updated)
            if not all(key) or key[0] == key[2] or key in seen_edges:
                continue
            seen_edges.add(key)
            next_edges.append(updated)
        append_graph_lifecycle_events(
            [
                {
                    'node_id': right,
                    'name': str(secondary.get('name') or ''),
                    'target_id': left,
                    'state': 'merged',
                    'source': 'manual_merge',
                    'detail': str(primary.get('name') or ''),
                }
            ],
            reason='merge_nodes_manual',
        )
        result = self.save_graph(next_nodes, next_edges, reason='merge_nodes_manual', apply_hygiene=False)
        result['merged_into'] = left
        result['merged_from'] = right
        return result

    def search_nodes(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        tokens = {normalize_name(token) for token in str(query or '').split() if normalize_name(token)}
        nodes = self._visible_nodes(self.load_nodes(), states=GRAPH_VISIBLE_SEARCH_STATES)
        if not tokens:
            ranked = sorted(nodes, key=score_node, reverse=True)
            return ranked[:limit]
        node_map = {str(node.get('id') or ''): node for node in nodes}
        related_names = self._related_name_map(node_map, self.load_edges())
        scored: list[tuple[float, dict[str, Any]]] = []
        for node in nodes:
            lexical = self._lexical_score(node, tokens, related_names.get(str(node.get('id') or ''), []))
            if lexical <= 0:
                continue
            base = max(score_node(node), 0.05)
            scored.append((round(base * lexical, 6), node))
        scored.sort(key=lambda item: (-item[0], str(item[1].get('name') or '')))
        return [item[1] for item in scored[:limit]]

    def top_ranked_nodes(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        return self.search_nodes(query, limit=limit)

    def subgraph(self, query: str, *, limit: int = 8, depth: int = 1) -> dict[str, Any]:
        seeds = self.search_nodes(query, limit=limit)
        if not seeds:
            return {'query': query, 'nodes': [], 'edges': [], 'seed_node_ids': [], 'diagnostics': self.graph_diagnostics([], [])}
        visible_nodes = self._visible_nodes(self.load_nodes(), states=GRAPH_VISIBLE_SEARCH_STATES)
        node_map = {str(node.get('id') or ''): node for node in visible_nodes}
        seed_ids = {str(node.get('id') or '') for node in seeds}
        selected_ids = set(seed_ids)
        selected_edges: list[dict[str, Any]] = []
        frontier = set(seed_ids)
        for _ in range(max(depth, 1)):
            next_frontier: set[str] = set()
            for edge in self.load_edges():
                src = str(edge.get('from') or '')
                dst = str(edge.get('to') or '')
                if (src in frontier or dst in frontier) and src in node_map and dst in node_map:
                    selected_edges.append(edge)
                    if src:
                        next_frontier.add(src)
                    if dst:
                        next_frontier.add(dst)
            selected_ids.update(next_frontier)
            frontier = next_frontier
        selected_nodes = [node_map[node_id] for node_id in selected_ids if node_id in node_map]
        non_summary_nodes = [node for node in selected_nodes if not str(node.get('id') or '').startswith('summary:')]
        if non_summary_nodes:
            selected_nodes = non_summary_nodes
            selected_node_ids = {str(node.get('id') or '') for node in selected_nodes}
            selected_edges = [
                edge
                for edge in selected_edges
                if str(edge.get('type') or '') != 'SUMMARIZES'
                and str(edge.get('from') or '') in selected_node_ids
                and str(edge.get('to') or '') in selected_node_ids
            ]
        selected_nodes.sort(key=score_node, reverse=True)
        return {
            'query': query,
            'nodes': selected_nodes[: limit * 2],
            'edges': selected_edges[: limit * 3],
            'seed_node_ids': sorted(seed_ids),
            'diagnostics': self.graph_diagnostics(selected_nodes[: limit * 2], selected_edges[: limit * 3]),
        }

    def answerable_node_view(self, node_id: str) -> dict[str, Any] | None:
        node = next((item for item in self.load_nodes() if str(item.get('id') or '') == str(node_id)), None)
        if node is None:
            return None
        node_map = {str(item.get('id') or ''): item for item in self.load_nodes()}
        edges = []
        for edge in self.load_edges():
            if node_id not in {str(edge.get('from') or ''), str(edge.get('to') or '')}:
                continue
            edges.append(
                {
                    **edge,
                    'from_name': str(node_map.get(str(edge.get('from') or ''), {}).get('name') or edge.get('from') or ''),
                    'to_name': str(node_map.get(str(edge.get('to') or ''), {}).get('name') or edge.get('to') or ''),
                }
            )
        return {
            'who_or_what': {
                'id': node.get('id'),
                'name': node.get('name'),
                'type': node.get('type'),
                'aliases': node.get('aliases') or [],
                'lifecycle_state': node.get('lifecycle_state'),
                'cluster_label': node.get('cluster_label') or '',
            },
            'what_is_it_like': {
                'description': node.get('description'),
                'facts': node.get('facts') or [],
                'importance': node.get('importance'),
                'confidence': node.get('confidence'),
                'frequency': node.get('frequency'),
                'folder': node.get('folder'),
                'translation_line': node.get('translation_line') or '',
                'lifecycle_state': node.get('lifecycle_state'),
                'cluster_key': node.get('cluster_key') or '',
                'cluster_label': node.get('cluster_label') or '',
                'context': node.get('context') or {},
            },
            'how_it_acts': edges,
        }

    def apply_hygiene(self) -> dict[str, Any]:
        return self.save_graph(self.load_nodes(), self.load_edges(), reason='graph_hygiene', apply_hygiene=True)

    def _commit(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]], *, reason: str) -> dict[str, Any]:
        return self.save_graph(nodes, edges, reason=reason, apply_hygiene=True)

    def _candidate_node(
        self,
        *,
        name: str,
        entity_type: str,
        aliases: list[str] | None = None,
        description: str = '',
        facts: list[str] | None = None,
        confidence: float = 0.75,
        source: str = 'chat',
        folder: str = '',
        importance: float = 0.7,
        translation_line: str = '',
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._normalize_node(
            {
                'id': f"{str(entity_type or 'CONCEPT').lower()}:{_slug(name)}",
                'name': str(name or '').strip(),
                'type': str(entity_type or 'CONCEPT').strip().upper(),
                'aliases': list(aliases or []),
                'translation_line': str(translation_line or '').strip(),
                'description': str(description or '').strip(),
                'facts': list(facts or []),
                'folder': str(folder or '').strip(),
                'importance': float(importance or 0.7),
                'confidence': float(confidence or 0.75),
                'frequency': 1,
                'context': dict(context or {'source': source}),
            }
        )

    def _prepare_graph_state(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]], *, apply_hygiene: bool) -> dict[str, Any]:
        state = {
            'nodes': [self._normalize_node(node) for node in nodes if isinstance(node, dict)],
            'edges': [self._normalize_edge(edge) for edge in edges if isinstance(edge, dict)],
            'errors': [],
            'warnings': [],
            'merged_nodes': 0,
            'removed_nodes': 0,
            'summary_nodes': 0,
            'lifecycle_events': [],
            'quality_before': None,
            'quality_after': None,
        }
        state['nodes'], state['edges'], errors, warnings = self._validate_graph_state(state['nodes'], state['edges'])
        state['errors'].extend(errors)
        state['warnings'].extend(warnings)
        if apply_hygiene:
            state['quality_before'] = asdict(self.graph_quality(state['nodes'], state['edges']))
            state['nodes'], state['edges'], merged, removed, summaries, lifecycle_events, chosen_quality = self._apply_hygiene_in_memory(
                state['nodes'],
                state['edges'],
            )
            state['merged_nodes'] = merged
            state['removed_nodes'] = removed
            state['summary_nodes'] = summaries
            state['lifecycle_events'] = lifecycle_events
            state['quality_after'] = asdict(chosen_quality)
            state['nodes'], state['edges'], errors, warnings = self._validate_graph_state(state['nodes'], state['edges'])
            state['errors'].extend(errors)
            state['warnings'].extend(warnings)
        return state

    def _validate_graph_state(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []
        node_by_id: dict[str, dict[str, Any]] = {}
        for node in nodes:
            normalized = self._normalize_node(node)
            node_id = str(normalized.get('id') or '')
            if not node_id:
                errors.append('node_missing_id')
                continue
            if node_id in node_by_id:
                warnings.append(f'duplicate_node_id:{node_id}')
                self._merge_node(node_by_id[node_id], normalized)
                continue
            node_by_id[node_id] = normalized

        filtered_edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for edge in edges:
            normalized = self._normalize_edge(edge)
            key = _edge_key(normalized)
            if not all(key):
                errors.append('edge_missing_endpoint')
                continue
            if normalized['from'] not in node_by_id or normalized['to'] not in node_by_id:
                warnings.append(f'orphan_edge:{normalized["from"]}:{normalized["type"]}:{normalized["to"]}')
                continue
            if key in seen_edges:
                warnings.append(f'duplicate_edge:{normalized["from"]}:{normalized["type"]}:{normalized["to"]}')
                continue
            seen_edges.add(key)
            filtered_edges.append(normalized)
        return list(node_by_id.values()), filtered_edges, errors, warnings

    def _apply_hygiene_in_memory(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int, int, list[dict[str, Any]], GraphQuality]:
        base_quality = self.graph_quality(nodes, edges)
        decayed = [self._decay_node(node) for node in nodes]
        merged_nodes, merged_edges, merged, merged_events = self._merge_duplicates(decayed, edges)
        reviewed_nodes = self._mark_duplicate_review_candidates(merged_nodes)
        clustered_nodes = self._assign_cluster_labels(reviewed_nodes, merged_edges)
        survivor_nodes, survivor_edges, removed, archived_events = self._garbage_collect(clustered_nodes, merged_edges)
        summarized_nodes, summarized_edges, summaries = self._create_summary_nodes(survivor_nodes, survivor_edges)
        candidate_quality = self.graph_quality(summarized_nodes, summarized_edges)
        structural_changes = bool(merged or removed or summaries or merged_events or archived_events) or any(
            (
                str(before.get('lifecycle_state') or 'active') != str(after.get('lifecycle_state') or 'active')
                or str(before.get('cluster_key') or '') != str(after.get('cluster_key') or '')
                or str(before.get('cluster_label') or '') != str(after.get('cluster_label') or '')
            )
            for before, after in zip(
                sorted(nodes, key=lambda item: str(item.get('id') or '')),
                sorted(summarized_nodes, key=lambda item: str(item.get('id') or '')),
                strict=False,
            )
        )
        if candidate_quality.score >= base_quality.score or structural_changes:
            return summarized_nodes, summarized_edges, merged, removed, summaries, merged_events + archived_events, candidate_quality
        return list(nodes), list(edges), 0, 0, 0, [], base_quality

    def _normalize_lifecycle_state(self, value: str) -> str:
        token = str(value or '').strip().lower()
        return token if token in GRAPH_LIFECYCLE_STATES else 'active'

    def _visible_nodes(self, nodes: list[dict[str, Any]], *, states: set[str] | None = None) -> list[dict[str, Any]]:
        allowed = states or GRAPH_VISIBLE_SEARCH_STATES
        return [node for node in nodes if self._normalize_lifecycle_state(node.get('lifecycle_state') or 'active') in allowed]

    def _decay_node(self, node: dict[str, Any]) -> dict[str, Any]:
        frequency = max(int(node.get('frequency') or 1), 1)
        factor = 0.992 if frequency >= 3 else 0.975
        next_importance = relevance_decay(float(node.get('importance') or 0.0), factor=factor)
        next_confidence = float(node.get('confidence') or 0.0)
        if frequency <= 2 and next_confidence > 0.0:
            next_confidence = round(max(next_confidence - 0.015, 0.0), 6)
        return self._normalize_node(
            {
                **node,
                'importance': next_importance,
                'confidence': next_confidence,
            }
        )

    def _determine_lifecycle_state(self, node: dict[str, Any]) -> str:
        graph_policy = get_runtime_config().graph
        context = dict(node.get('context') or {})
        requested = str(node.get('lifecycle_state') or '').strip().lower()
        manual_override = str(context.get('review_reason') or '').strip().lower() == 'manual_review'
        importance = float(node.get('importance') or 0.0)
        confidence = float(node.get('confidence') or 0.0)
        frequency = max(int(node.get('frequency') or 1), 1)
        if requested == 'merged':
            return 'merged'
        if requested == 'archived':
            return 'archived'
        if requested == 'suspect' and manual_override:
            return 'suspect'
        if str(context.get('review_status') or '').strip().lower() in {'quarantine', 'review'}:
            return 'suspect'
        if requested == 'weak' and manual_override:
            return 'weak'
        if requested == 'active' and manual_override:
            return 'active'
        source = str(context.get('source') or '').strip().lower()
        if confidence < graph_policy.suspect_confidence_threshold and source in {'session', 'extraction', 'rebuild', 'file', 'upload', 'document'}:
            return 'suspect'
        if frequency <= 2 and (
            importance < graph_policy.weak_importance_threshold or confidence < graph_policy.weak_confidence_threshold
        ):
            return 'weak'
        return 'active'

    def _duplicate_metrics(self, nodes: list[dict[str, Any]]) -> dict[str, Any]:
        merge_samples: list[dict[str, Any]] = []
        review_samples: list[dict[str, Any]] = []
        merge_count = 0
        review_count = 0
        for index, left in enumerate(nodes):
            for right in nodes[index + 1 :]:
                left_tokens = {token for token in normalize_name(str(left.get('name') or '')).split() if token}
                right_tokens = {token for token in normalize_name(str(right.get('name') or '')).split() if token}
                if not (left_tokens & right_tokens):
                    continue
                score = combined_similarity(left, right)
                bucket = similarity_bucket(score)
                if bucket not in {'merge', 'review', 'exactish'}:
                    continue
                sample = {
                    'left_id': str(left.get('id') or ''),
                    'left_name': str(left.get('name') or ''),
                    'right_id': str(right.get('id') or ''),
                    'right_name': str(right.get('name') or ''),
                    'similarity': round(float(score or 0.0), 4),
                    'bucket': bucket,
                }
                if bucket in {'merge', 'exactish'} and should_merge(left, right):
                    merge_count += 1
                    merge_samples.append(sample)
                elif bucket == 'review':
                    review_count += 1
                    review_samples.append(sample)
        return {
            'merge_count': merge_count,
            'review_count': review_count,
            'merge_samples': merge_samples,
            'review_samples': review_samples,
        }

    def _lifecycle_counts(self, nodes: list[dict[str, Any]]) -> dict[str, int]:
        counts = {state: 0 for state in GRAPH_NODE_LIFECYCLE_STATES}
        for node in nodes:
            counts[self._normalize_lifecycle_state(node.get('lifecycle_state') or 'active')] += 1
        return counts

    def _archived_lifecycle_counts(self) -> dict[str, int]:
        payload = load_graph_lifecycle_log()
        archived_ids: set[str] = set()
        merged_ids: set[str] = set()
        for entry in list(payload.get('entries') or []):
            for event in list(entry.get('events') or []):
                state = str(event.get('state') or '').strip().lower()
                node_id = str(event.get('node_id') or event.get('name') or '').strip()
                if not node_id:
                    continue
                if state == 'archived':
                    archived_ids.add(node_id)
                elif state == 'merged':
                    merged_ids.add(node_id)
        return {
            'archived': len(archived_ids),
            'merged': len(merged_ids),
        }

    def _mark_duplicate_review_candidates(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reviewed = [self._normalize_node(node) for node in nodes]
        for index, left in enumerate(reviewed):
            for right in reviewed[index + 1 :]:
                left_tokens = {token for token in normalize_name(str(left.get('name') or '')).split() if token}
                right_tokens = {token for token in normalize_name(str(right.get('name') or '')).split() if token}
                if not (left_tokens & right_tokens):
                    continue
                score = combined_similarity(left, right)
                if similarity_bucket(score) != 'review':
                    continue
                weaker, stronger = sorted(
                    [left, right],
                    key=lambda item: (
                        float(item.get('confidence') or 0.0),
                        float(item.get('importance') or 0.0),
                        int(item.get('frequency') or 0),
                    ),
                )
                weaker['context'] = {
                    **dict(weaker.get('context') or {}),
                    'review_status': 'review',
                    'review_reason': 'duplicate_candidate',
                    'review_target_id': str(stronger.get('id') or ''),
                    'review_similarity': round(float(score or 0.0), 4),
                }
                weaker['lifecycle_state'] = 'suspect'
        return [self._normalize_node(node) for node in reviewed]

    def _assign_cluster_labels(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(nodes) < get_runtime_config().graph.clustering_min_nodes:
            return [self._normalize_node({**node, 'cluster_key': '', 'cluster_label': ''}) for node in nodes]
        min_component_size = get_runtime_config().graph.clustering_min_component_size
        node_map = {str(node.get('id') or ''): self._normalize_node(node) for node in nodes}
        adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_map}
        for edge in edges:
            src = str(edge.get('from') or '')
            dst = str(edge.get('to') or '')
            if src in adjacency and dst in adjacency:
                adjacency[src].add(dst)
                adjacency[dst].add(src)
        visited: set[str] = set()
        for node_id in sorted(node_map):
            if node_id in visited:
                continue
            stack = [node_id]
            component: list[str] = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                stack.extend(sorted(adjacency.get(current) or []))
            ranked_component = sorted(
                component,
                key=lambda item: (
                    score_node(node_map[item]),
                    str(node_map[item].get('name') or ''),
                ),
                reverse=True,
            )
            if len(component) < min_component_size:
                for item in component:
                    node_map[item] = self._normalize_node({**node_map[item], 'cluster_key': '', 'cluster_label': ''})
                continue
            anchor = node_map[ranked_component[0]]
            cluster_key = f"cluster:{_slug(anchor.get('name') or anchor.get('id') or 'group')}"
            cluster_label = str(anchor.get('name') or anchor.get('id') or '').strip()
            for item in component:
                node_map[item] = self._normalize_node(
                    {
                        **node_map[item],
                        'cluster_key': cluster_key,
                        'cluster_label': cluster_label,
                    }
                )
        return [node_map[str(node.get('id') or '')] for node in nodes if str(node.get('id') or '') in node_map]

    def _cluster_summaries(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for node in nodes:
            cluster_key = str(node.get('cluster_key') or '').strip()
            cluster_label = str(node.get('cluster_label') or '').strip()
            if not cluster_key or not cluster_label:
                continue
            group = groups.setdefault(cluster_key, {'cluster_key': cluster_key, 'cluster_label': cluster_label, 'size': 0, 'states': set()})
            group['size'] += 1
            group['states'].add(str(node.get('lifecycle_state') or 'active'))
        return [
            {
                'cluster_key': key,
                'cluster_label': value['cluster_label'],
                'size': value['size'],
                'states': sorted(value['states']),
            }
            for key, value in sorted(groups.items(), key=lambda item: (-item[1]['size'], item[1]['cluster_label']))
        ]

    def _lexical_score(self, node: dict[str, Any], tokens: set[str], related_names: list[str]) -> float:
        fields = [
            (str(node.get('name') or ''), 3.2),
            (' '.join(str(item) for item in list(node.get('aliases') or [])), 2.6),
            (str(node.get('description') or ''), 1.6),
            (' '.join(str(item) for item in list(node.get('facts') or [])), 2.2),
            (json.dumps(node.get('context') or {}, ensure_ascii=False), 1.1),
            (' '.join(related_names), 1.5),
        ]
        score = 0.0
        for field, weight in fields:
            field_tokens = {normalize_name(token) for token in str(field or '').split() if normalize_name(token)}
            overlap = len(tokens & field_tokens)
            if overlap:
                score += overlap * weight
        return score

    def _related_name_map(self, node_map: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, list[str]]:
        related: dict[str, list[str]] = {}
        for edge in edges:
            src = str(edge.get('from') or '')
            dst = str(edge.get('to') or '')
            if src and dst:
                related.setdefault(src, []).append(str(node_map.get(dst, {}).get('name') or dst))
                related.setdefault(dst, []).append(str(node_map.get(src, {}).get('name') or src))
        return related

    def _find_duplicate(self, nodes: list[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any] | None:
        for node in nodes:
            if should_merge(node, candidate):
                return node
        return None

    def _upsert_node(self, nodes: list[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any]:
        existing = self._find_duplicate(nodes, candidate)
        if existing is not None:
            self._merge_node(existing, candidate)
            return existing
        nodes.append(candidate)
        return candidate

    def _ensure_entity(self, nodes: list[dict[str, Any]], name: str, *, source: str) -> str:
        candidate = self._candidate_node(
            name=name,
            entity_type='CONCEPT',
            aliases=[],
            description='',
            facts=[],
            confidence=0.55,
            source=source,
            importance=0.4,
        )
        return str(self._upsert_node(nodes, candidate).get('id') or '')

    def _merge_node(self, existing: dict[str, Any], candidate: dict[str, Any]) -> None:
        previous_existing_name = str(existing.get('name') or '')
        candidate_name = str(candidate.get('name') or '')
        canonical = canonical_name(previous_existing_name, candidate_name)
        if canonical:
            existing['name'] = canonical
        existing['aliases'] = merge_aliases(
            list(existing.get('aliases') or []),
            [previous_existing_name],
            list(candidate.get('aliases') or []),
            [candidate_name],
        )
        existing['facts'] = list(
            dict.fromkeys(
                [
                    str(item).strip()
                    for item in list(existing.get('facts') or []) + list(candidate.get('facts') or [])
                    if str(item).strip()
                ]
            )
        )[:16]
        if candidate.get('description') and (
            not existing.get('description') or len(str(candidate.get('description') or '')) > len(str(existing.get('description') or ''))
        ):
            existing['description'] = candidate['description']
        if candidate.get('translation_line') and (
            not existing.get('translation_line')
            or len(str(candidate.get('translation_line') or '')) > len(str(existing.get('translation_line') or ''))
        ):
            existing['translation_line'] = str(candidate.get('translation_line') or '').strip()
        if not existing.get('folder') and candidate.get('folder'):
            existing['folder'] = candidate['folder']
        existing['importance'] = round(max(float(existing.get('importance') or 0.0), float(candidate.get('importance') or 0.0)) + 0.05, 6)
        existing['confidence'] = round(max(float(existing.get('confidence') or 0.0), float(candidate.get('confidence') or 0.0)), 6)
        existing['frequency'] = int(existing.get('frequency') or 0) + max(int(candidate.get('frequency') or 1), 1)
        existing['context'] = {**dict(existing.get('context') or {}), **dict(candidate.get('context') or {})}
        existing['cluster_key'] = str(existing.get('cluster_key') or candidate.get('cluster_key') or '').strip()
        existing['cluster_label'] = str(existing.get('cluster_label') or candidate.get('cluster_label') or '').strip()
        existing['lifecycle_state'] = self._determine_lifecycle_state(existing)

    def _merge_duplicates(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, list[dict[str, Any]]]:
        merged = 0
        remaining: list[dict[str, Any]] = []
        replacements: dict[str, str] = {}
        lifecycle_events: list[dict[str, Any]] = []
        for node in nodes:
            target = next((item for item in remaining if should_merge(item, node)), None)
            if target is None:
                remaining.append(node)
                continue
            merged += 1
            replacements[str(node.get('id') or '')] = str(target.get('id') or '')
            lifecycle_events.append(
                {
                    'node_id': str(node.get('id') or ''),
                    'name': str(node.get('name') or ''),
                    'target_id': str(target.get('id') or ''),
                    'state': 'merged',
                    'source': str(dict(node.get('context') or {}).get('source') or 'hygiene'),
                    'detail': str(target.get('name') or ''),
                }
            )
            self._merge_node(target, node)

        normalized_edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for edge in edges:
            src = replacements.get(str(edge.get('from') or ''), str(edge.get('from') or ''))
            dst = replacements.get(str(edge.get('to') or ''), str(edge.get('to') or ''))
            updated = self._normalize_edge({**edge, 'from': src, 'to': dst})
            key = _edge_key(updated)
            if not all(key) or key in seen_edges:
                continue
            seen_edges.add(key)
            normalized_edges.append(updated)
        return remaining, normalized_edges, merged, lifecycle_events

    def _garbage_collect(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, list[dict[str, Any]]]:
        graph_policy = get_runtime_config().graph
        survivors: list[dict[str, Any]] = []
        archived_events: list[dict[str, Any]] = []
        for node in nodes:
            importance = float(node.get('importance') or 0.0)
            confidence = float(node.get('confidence') or 0.0)
            frequency = int(node.get('frequency') or 0)
            if (
                str(node.get('lifecycle_state') or '') != 'suspect'
                and importance <= graph_policy.archive_importance_threshold
                and confidence <= graph_policy.archive_confidence_threshold
                and frequency < 2
            ):
                archived_events.append(
                    {
                        'node_id': str(node.get('id') or ''),
                        'name': str(node.get('name') or ''),
                        'state': 'archived',
                        'source': str(dict(node.get('context') or {}).get('source') or 'hygiene'),
                        'detail': 'low_value_decay',
                    }
                )
                continue
            survivors.append(self._normalize_node(node))
        removed_ids = {str(node.get('id') or '') for node in nodes if all(str(node.get('id') or '') != str(item.get('id') or '') for item in survivors)}
        filtered_edges = [edge for edge in edges if str(edge.get('from') or '') not in removed_ids and str(edge.get('to') or '') not in removed_ids]
        return survivors, filtered_edges, len(removed_ids), archived_events

    def _create_summary_nodes(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        if len(nodes) < 32:
            return nodes, edges, 0
        node_map = {str(node.get('id') or ''): node for node in nodes}
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for edge in edges:
            if str(edge.get('type') or '') in {'HAS_ALIAS', 'SUMMARIZES'}:
                continue
            groups.setdefault((str(edge.get('from') or ''), str(edge.get('type') or '')), []).append(edge)
        created = 0
        for (src_id, relation_type), group in groups.items():
            if len(group) < 2 or not src_id:
                continue
            summary_id = f'summary:{src_id}:{relation_type.lower()}'
            if summary_id in node_map:
                continue
            src_name = str(node_map.get(src_id, {}).get('name') or src_id)
            target_names = [str(node_map.get(str(edge.get('to') or ''), {}).get('name') or edge.get('to') or '') for edge in group[:4]]
            summary_node = self._normalize_node(
                {
                    'id': summary_id,
                    'name': f'{src_name} {relation_type.lower()} summary',
                    'type': 'CONCEPT',
                    'translation_line': f'{src_name} {relation_type.lower()} summary',
                    'description': f'{src_name} {relation_type.lower()} {", ".join(name for name in target_names if name)}.',
                    'aliases': target_names,
                    'facts': [f'{src_name} {relation_type.lower()} {name}.' for name in target_names if name],
                    'importance': 0.2 + (0.05 * len(group)),
                    'confidence': 0.7,
                    'frequency': len(group),
                    'context': {'source': 'compression'},
                }
            )
            nodes.append(summary_node)
            node_map[summary_id] = summary_node
            edges.append(self._normalize_edge({'from': src_id, 'to': summary_id, 'type': 'SUMMARIZES', 'weight': 0.6, 'confidence': 0.7, 'source': 'compression'}))
            created += 1
        deduped_edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for edge in edges:
            key = _edge_key(edge)
            if not all(key) or key in seen_edges:
                continue
            seen_edges.add(key)
            deduped_edges.append(edge)
        return nodes, deduped_edges, created

    def _normalize_node(self, node: dict[str, Any]) -> dict[str, Any]:
        node_type = str(node.get('type') or 'CONCEPT').strip().upper()
        if node_type not in GRAPH_NODE_TYPES:
            node_type = 'CONCEPT'
        facts = [
            str(item).strip()
            for item in list(node.get('facts') or [])
            if str(item).strip()
        ]
        normalized = {
            'id': str(node.get('id') or f'{node_type.lower()}:{_slug(node.get("name") or "item")}').strip(),
            'name': str(node.get('name') or node.get('id') or '').strip(),
            'type': node_type,
            'aliases': merge_aliases(list(node.get('aliases') or [])),
            'translation_line': str(node.get('translation_line') or '').strip(),
            'description': str(node.get('description') or '').strip(),
            'facts': list(dict.fromkeys(facts))[:16],
            'folder': str(node.get('folder') or '').strip(),
            'importance': round(max(float(node.get('importance') or 0.1), 0.0), 6),
            'confidence': round(min(max(float(node.get('confidence') or 0.5), 0.0), 1.0), 6),
            'frequency': max(int(node.get('frequency') or 1), 1),
            'context': dict(node.get('context') or {}),
            'cluster_key': str(node.get('cluster_key') or '').strip(),
            'cluster_label': str(node.get('cluster_label') or '').strip(),
        }
        normalized['lifecycle_state'] = self._determine_lifecycle_state({**normalized, 'lifecycle_state': node.get('lifecycle_state')})
        return normalized

    def _normalize_edge(self, edge: dict[str, Any]) -> dict[str, Any]:
        return {
            'id': str(edge.get('id') or f"edge:{uuid4().hex[:12]}").strip(),
            'from': str(edge.get('from') or '').strip(),
            'to': str(edge.get('to') or '').strip(),
            'type': str(edge.get('type') or 'RELATED_TO').strip().upper(),
            'weight': round(min(max(float(edge.get('weight') or 0.5), 0.0), 1.0), 6),
            'confidence': round(min(max(float(edge.get('confidence') or 0.5), 0.0), 1.0), 6),
            'source': str(edge.get('source') or '').strip(),
        }
