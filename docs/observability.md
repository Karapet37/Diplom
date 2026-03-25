# Persona-Graph Observability

## Purpose

The observability layer is designed to answer three operational questions:

1. Where is request latency spent?
2. Is graph quality drifting?
3. Is runtime behavior falling back, rebuilding too often, or producing suspicious rethink outcomes?

The implementation is intentionally lightweight:

- in-memory recent traces
- structured JSON logs
- aggregated counters
- on-demand graph diagnostics

It does not redesign product behavior and does not introduce dashboard-only abstractions.

## Main runtime tracing

The main chat runtime produces a request trace for each `POST /api/cognitive/chat/respond`.

Each trace contains:

- `request_id`
- `session_id`
- route
- stage timings
- total time
- fallback flag and fallback reason
- context token usage
- persona name and current entity

Measured chat stages:

- `graph_prewrite`
- `analysis`
- `situation_building`
- `feature_extraction`
- `classification`
- `head_selection`
- `persona_update`
- `context_building`
- `llm_call`
- `storage_writes`

## Structured logs

Structured logs are emitted as JSON lines for:

- `request_started`
- `request_finished`

The log payloads include `request_id` and `session_id`, which makes them suitable for machine parsing and correlation.

## Graph health diagnostics

`GraphStore.graph_diagnostics()` exposes:

- `node_count`
- `edge_count`
- `duplicate_candidates`
- `orphan_nodes`
- `low_value_nodes`
- `summary_nodes`
- `quality`

This is computed on demand and is intended for diagnostics, not for every hot-path request.

## Aggregate metrics

The observability snapshot aggregates:

- request counts
- fallback rate
- recent context token usage
- per-stage timing averages and maxima
- rebuild schedule counts
- rethink preview/apply counts
- graph write counters

## Debug endpoints

Available endpoints:

- `GET /api/cognitive/debug/metrics`
- `GET /api/cognitive/debug/traces`
- `GET /api/cognitive/debug/graph-health`

Typical use:

```bash
curl http://127.0.0.1:8008/api/cognitive/debug/metrics
curl "http://127.0.0.1:8008/api/cognitive/debug/traces?limit=10"
curl http://127.0.0.1:8008/api/cognitive/debug/graph-health
```

## How to inspect a slow request

1. Call the chat endpoint.
2. Take the returned `trace_id`.
3. Read recent traces from `/api/cognitive/debug/traces`.
4. Compare:
   - `llm_call`
   - `context_building`
   - `storage_writes`
   - total context token usage

If `fallback_used=true`, inspect:

- `fallback_reason`
- graph context sparsity
- persona availability
- rebuild activity

## How to inspect graph drift

Use `/api/cognitive/debug/graph-health` and look at:

- `duplicate_candidates`
- `orphan_nodes`
- `low_value_nodes`
- `quality.redundancy`

If rethink mode is suspected, compare:

- `rethink_preview_total`
- `rethink_apply_total`
- `rethink_failure_total`

from `/api/cognitive/debug/metrics`.
