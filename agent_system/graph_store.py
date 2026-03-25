from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from .duplicate_resolver import merge_aliases, normalize_name, relevance_decay, score_node, should_merge
from .models import ENTITY_TYPES, GraphQuality
from .runtime_config import get_runtime_config

GRAPH_NODE_TYPES = set(ENTITY_TYPES)
QUALITY_ALPHA = 0.45
QUALITY_BETA = 0.35
QUALITY_GAMMA = 0.2


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
        return {'nodes': self.load_nodes(), 'edges': self.load_edges()}

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
        if not node_rows:
            return GraphQuality(relevance=0.0, redundancy=0.0, connectivity=0.0, score=0.0)

        relevance = round(
            min(sum(min(score_node(node), 1.0) for node in node_rows) / max(len(node_rows), 1), 1.0),
            6,
        )
        low_value_ratio = sum(
            1 for node in node_rows if float(node.get('importance') or 0.0) < 0.05 and int(node.get('frequency') or 0) < 2
        ) / max(len(node_rows), 1)
        merge_candidates = 0
        for index, left in enumerate(node_rows):
            for right in node_rows[index + 1 :]:
                if should_merge(left, right):
                    merge_candidates += 1
        duplicate_ratio = min(merge_candidates / max(len(node_rows), 1), 1.0)
        redundancy = round(min((0.7 * duplicate_ratio) + (0.3 * low_value_ratio), 1.0), 6)

        node_ids = {str(node.get('id') or '') for node in node_rows}
        valid_edges = [
            edge for edge in edge_rows if str(edge.get('from') or '') in node_ids and str(edge.get('to') or '') in node_ids
        ]
        connected_nodes = {
            endpoint
            for edge in valid_edges
            for endpoint in (str(edge.get('from') or ''), str(edge.get('to') or ''))
            if endpoint
        }
        connected_ratio = len(connected_nodes) / max(len(node_rows), 1)
        edge_density = min(len(valid_edges) / max(len(node_rows) - 1, 1), 1.0)
        connectivity = round(min((0.5 * connected_ratio) + (0.5 * edge_density), 1.0), 6)

        score = round((QUALITY_ALPHA * relevance) - (QUALITY_BETA * redundancy) + (QUALITY_GAMMA * connectivity), 6)
        return GraphQuality(
            relevance=relevance,
            redundancy=redundancy,
            connectivity=connectivity,
            score=score,
        )

    def save_graph(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        *,
        reason: str = 'sync',
        apply_hygiene: bool = False,
    ) -> dict[str, Any]:
        existing_nodes = self.load_nodes()
        existing_edges = self.load_edges()
        quality_before = self.graph_quality(existing_nodes, existing_edges)
        state = self._prepare_graph_state(nodes, edges, apply_hygiene=apply_hygiene)
        quality_after = self.graph_quality(state['nodes'], state['edges'])
        if not state['nodes'] and existing_nodes and reason != 'clear':
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
        write_json(graph_nodes_path(), state['nodes'])
        write_json(graph_edges_path(), state['edges'])
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
            'quality_before': asdict(quality_before),
            'quality_after': asdict(quality_after),
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
        self._commit(nodes, edges, reason='upsert_entity')
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
        )

    def patch_node(
        self,
        node_id: str,
        *,
        description: str | None = None,
        facts: list[str] | None = None,
        translation_line: str | None = None,
        context_patch: dict[str, Any] | None = None,
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
            break
        if not found:
            return {'ok': False, 'reason': 'node_not_found'}
        result = self.save_graph(nodes, self.load_edges(), reason='patch_node', apply_hygiene=False)
        result['node_id'] = target
        return result

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
        name_to_id: dict[str, str] = {}

        for entity in list(extraction.get('entities') or []):
            if not isinstance(entity, dict):
                continue
            candidate = self._normalize_node({**entity, 'context': {**dict(entity.get('context') or {}), 'source': source}})
            current = self._upsert_node(nodes, candidate)
            name_to_id[normalize_name(str(entity.get('name') or ''))] = str(current.get('id') or '')
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
        result.update({'touched_nodes': touched_nodes, 'touched_edges': touched_edges})
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
        if not any(str(node.get('id') or '') == target for node in nodes):
            return {'ok': False, 'reason': 'node_not_found'}
        next_nodes = [node for node in nodes if str(node.get('id') or '') != target]
        next_edges = [
            edge
            for edge in self.load_edges()
            if str(edge.get('from') or '') != target and str(edge.get('to') or '') != target
        ]
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
        result = self.save_graph(next_nodes, next_edges, reason='merge_nodes_manual', apply_hygiene=False)
        result['merged_into'] = left
        result['merged_from'] = right
        return result

    def search_nodes(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        tokens = {normalize_name(token) for token in str(query or '').split() if normalize_name(token)}
        nodes = self.load_nodes()
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
            return {'query': query, 'nodes': [], 'edges': [], 'seed_node_ids': []}
        node_map = {str(node.get('id') or ''): node for node in self.load_nodes()}
        seed_ids = {str(node.get('id') or '') for node in seeds}
        selected_ids = set(seed_ids)
        selected_edges: list[dict[str, Any]] = []
        frontier = set(seed_ids)
        for _ in range(max(depth, 1)):
            next_frontier: set[str] = set()
            for edge in self.load_edges():
                src = str(edge.get('from') or '')
                dst = str(edge.get('to') or '')
                if src in frontier or dst in frontier:
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
            },
            'what_is_it_like': {
                'description': node.get('description'),
                'facts': node.get('facts') or [],
                'importance': node.get('importance'),
                'confidence': node.get('confidence'),
                'frequency': node.get('frequency'),
                'folder': node.get('folder'),
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
            'quality_before': None,
            'quality_after': None,
        }
        state['nodes'], state['edges'], errors, warnings = self._validate_graph_state(state['nodes'], state['edges'])
        state['errors'].extend(errors)
        state['warnings'].extend(warnings)
        if apply_hygiene:
            state['quality_before'] = asdict(self.graph_quality(state['nodes'], state['edges']))
            state['nodes'], state['edges'], merged, removed, summaries, chosen_quality = self._apply_hygiene_in_memory(state['nodes'], state['edges'])
            state['merged_nodes'] = merged
            state['removed_nodes'] = removed
            state['summary_nodes'] = summaries
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
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int, int, GraphQuality]:
        base_quality = self.graph_quality(nodes, edges)
        decayed = [self._normalize_node({**node, 'importance': relevance_decay(float(node.get('importance') or 0.0))}) for node in nodes]
        merged_nodes, merged_edges, merged = self._merge_duplicates(decayed, edges)
        survivor_nodes, survivor_edges, removed = self._garbage_collect(merged_nodes, merged_edges)
        summarized_nodes, summarized_edges, summaries = self._create_summary_nodes(survivor_nodes, survivor_edges)
        candidate_quality = self.graph_quality(summarized_nodes, summarized_edges)
        if candidate_quality.score >= base_quality.score:
            return summarized_nodes, summarized_edges, merged, removed, summaries, candidate_quality
        return list(nodes), list(edges), 0, 0, 0, base_quality

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
        existing['aliases'] = merge_aliases(
            list(existing.get('aliases') or []),
            [str(existing.get('name') or '')],
            list(candidate.get('aliases') or []),
            [str(candidate.get('name') or '')],
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

    def _merge_duplicates(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        merged = 0
        remaining: list[dict[str, Any]] = []
        replacements: dict[str, str] = {}
        for node in nodes:
            target = next((item for item in remaining if should_merge(item, node)), None)
            if target is None:
                remaining.append(node)
                continue
            merged += 1
            replacements[str(node.get('id') or '')] = str(target.get('id') or '')
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
        return remaining, normalized_edges, merged

    def _garbage_collect(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        survivors = [node for node in nodes if not (float(node.get('importance') or 0.0) < 0.05 and int(node.get('frequency') or 0) < 2)]
        removed_ids = {str(node.get('id') or '') for node in nodes if node not in survivors}
        filtered_edges = [edge for edge in edges if str(edge.get('from') or '') not in removed_ids and str(edge.get('to') or '') not in removed_ids]
        return survivors, filtered_edges, len(removed_ids)

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
        return {
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
        }

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
