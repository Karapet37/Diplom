# Graph Lifecycle

This document describes the deterministic graph lifecycle used by the active `persona-graph agent system`.

## Lifecycle States

Nodes may move through the following states:

- `active` — normal graph node, visible to retrieval and subgraph search.
- `weak` — low-usage or lower-confidence node that remains visible but is visually marked as weaker.
- `suspect` — quarantined or review-required node. It remains visible in the full graph view but is excluded from hot-path retrieval and context search.
- `archived` — node removed from the active graph and recorded in the graph lifecycle archive.
- `merged` — node collapsed into a canonical node and recorded in the graph lifecycle archive.

The active graph file stores only live nodes. `archived` and `merged` counts are reconstructed from the archive lifecycle log in `memory/archive/graphs/lifecycle.json`.

## Read / Write Rules

Hot-path graph retrieval reads only:

- `active`
- `weak`

`suspect` nodes are kept out of `search_nodes()` and `subgraph()` so low-confidence or review-range noise does not leak into chat context.

The full graph workspace can still surface `suspect` nodes for manual inspection.

## Quarantine and Review

Low-confidence extractions are not silently promoted to normal graph knowledge.

- extraction confidence below `runtime_config.graph.extraction_quarantine_confidence`
  -> node gets `context.review_status = "quarantine"`
  -> lifecycle becomes `suspect`

Review-range duplicate pairs are not auto-merged.

- high-confidence duplicate bucket
  -> deterministic merge
- review bucket
  -> lower-quality node marked `suspect`
  -> `context.review_reason = "duplicate_candidate"`
  -> optional manual review can later promote or archive it

## Aging and Decay

Graph hygiene applies bounded decay during maintenance:

- low-frequency nodes decay faster in `importance`
- very low-usage nodes also lose a small amount of `confidence`
- nodes that fall below archival thresholds are removed from the active graph and recorded as `archived`

This keeps the active graph smaller and more retrieval-oriented without deleting the audit trail.

## Clustering

Large connected graphs can receive deterministic cluster labels.

- clustering activates only above `runtime_config.graph.clustering_min_nodes`
- connected components below `runtime_config.graph.clustering_min_component_size` are ignored
- the highest-scoring node in a component becomes the cluster anchor
- nodes receive `cluster_key` and `cluster_label`

Clusters are a structural aid for visualization and diagnostics. They do not change graph semantics.

## Diagnostics

`GraphStore.graph_diagnostics()` now reports:

- `duplicate_rate`
- `orphan_rate`
- `average_relation_density`
- `suspect_node_count`
- `archived_node_count`
- `merged_node_count`
- `duplicate_candidates`
- `duplicate_review_candidates`
- `cluster_count`

These metrics are surfaced via:

- `GET /api/cognitive/debug/graph-health`
- `GET /api/cognitive/graph`
- `GET /api/cognitive/graph/subgraph`

## Manual Review

Manual lifecycle review is available through:

- `POST /api/cognitive/graph/nodes/{node_id}/state`

Allowed transitions currently handled in code:

- `suspect`
- `weak`
- `active`
- `archived`

`merged` remains a merge-only state driven by deterministic merge workflows.
