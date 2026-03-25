# Deterministic Context Pipeline

This document describes the active context assembly path used by `agent_system/context_builder.py`.

The context builder does not ask the LLM what context should exist. It performs a deterministic five-stage pipeline:

```text
collect candidates
  -> score candidates
  -> rank candidates
  -> compress candidates
  -> pack bounded final context
```

## 1. Candidate Sources

The active candidate pools are separated by source:

- `session_short_term_history`
  Short-term session lines from the latest parsed chat turns.
- `persona_memory`
  Persona identity, traits, emotion state, relations, examples, knowledge, and learned reactions.
- `persona_triad`
  Persona form, compressed behavioral log tuples, and decision explanation.
- `global_graph_facts`
  Nodes from the global graph subgraph search.
- `local_graph_neighborhood`
  Persona-local graph nodes and relations.
- `file_ingested_knowledge`
  Global graph nodes whose `context.source == "file"`.

These are collected independently and then normalized into typed `ContextCandidate` rows.

## 2. Scoring Model

Each candidate receives an explicit weighted score:

```text
score = 0.34 * relevance
      + 0.12 * recency
      + 0.16 * importance
      + 0.12 * confidence
      + 0.16 * persona_alignment
      + 0.10 * graph_connectivity
```

Factor meanings:

- `relevance`
  Lexical overlap with the active query.
- `recency`
  Freshness of the evidence source. Session lines get explicit recency by order; other sources use explicit source baselines.
- `importance`
  Node importance for graph facts or deterministic priority for persona items.
- `confidence`
  Node confidence for graph facts or deterministic confidence for storage-backed persona/session items.
- `persona_alignment`
  How strongly the item is aligned to the active persona or current entity.
- `graph_connectivity`
  Degree-based structural value for graph nodes, or explicit lower structural priors for relation/reaction summaries.

The score is explainable because each factor is recorded per candidate in `context_debug`.

## 3. Ranking and Ordering

Ranking happens per section:

- `persona_block`
- `graph_context`
- `recent_dialogue`

Within a section, candidates are ordered by:

1. descending total score;
2. descending persona alignment;
3. deterministic source priority;
4. title;
5. candidate id.

This keeps ordering stable across repeated runs.

## 4. Compression

Compression happens before final packing.

It is deterministic and item-specific:

- large persona knowledge blocks are clipped more aggressively than core persona identity;
- history lines are clipped to compact message-sized rows;
- graph node previews are compacted before graph rendering.

Compressed candidates are marked with `compressed=true` in `context_debug`.

## 5. Packing

Packing is bounded by runtime configuration from `agent_system/runtime_config.py`.

Section packing rules:

- `persona_block`
  Packs persona core/state first, then triad and auxiliary memory in fixed render order.
- `graph_context`
  Selects ranked graph nodes, then materializes edges between selected nodes, then renders with `render_graph_context(...)`.
- `recent_dialogue`
  Selects ranked dialogue rows but renders them back in chronological order.

After section packing, the final prompt sections still pass through global section fitting so the full context remains within the configured prompt budget.

## 6. Debug Visibility

`build_context(...)` returns `context_debug` with:

- stage counts;
- scoring weights;
- source counts;
- selected items with factor breakdowns and reasons;
- top unselected items;
- selected node ids and edge keys;
- final estimated token count.

The chat runtime also exposes a compact context summary in observability traces under the `context_building` stage.

## 7. Regression Coverage

Relevant tests:

- `tests/agent_system/test_context_and_demo.py`
- `tests/agent_system/test_context_pipeline.py`

These cover:

- bounded packing;
- persona-state preservation;
- source separation visibility;
- deterministic ordering for equal-ranked graph candidates.
