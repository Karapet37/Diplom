# Core System Design

## Goal
The repository now has a core-first contract that the active behavioral runtime can depend on.
The core defines stable interfaces for graph memory, personality, context building, and agent control.

## Core Modules
- `core/graph_core.py`
  - immutable `NodeCore`
  - `NodeBranches` and `NodeContext`
  - `GraphNode`, `GraphEdge`, `GraphMemory`
  - `RAMContextGraph`
- `core/personality_core.py`
  - stable `PersonalityCore`
  - gradual update policy for slow personality change
- `core/context_core.py`
  - orchestration contract for:
    - signal extraction
    - graph search
    - node hydration
    - branch expansion
  - output is always a `RAMContextGraph`
- `core/agent_core.py`
  - agent permissions
  - explicit ban on direct `node_core` and `personality_core` mutation
- `core/system_core.py`
  - composition root for the core system

## System Modules Built On Top Of Core
- `core/style_engine.py`
  - explicit trigger-only style learning
  - converts user samples into persisted style profiles
- `core/speech_dna.py`
  - stable speech-shaping structure:
    - style embedding
    - typical phrases
    - vocabulary patterns
    - punctuation profile
    - sentence rhythm
- `core/dialogue_engine.py`
  - builds a dialogue contract from:
    - `PersonalityCore`
    - `SpeechDNA`
    - `RAMContextGraph`
    - agent roles
    - scenario framing
- `core/scenario_engine.py`
  - classifies a dialogue situation into reusable scenario frames
- `core/agent_roles.py`
  - defines professional reasoning roles such as law, business, and strategy
- `core/graph_initializer.py`
  - turns a graph payload into a layered graph brain with:
    - `root_core`
    - `domain_core`
    - `concept_core`
    - `pattern_core`
    - `example_nodes`
- `core/graph_traversal.py`
  - implements the query pipeline:
    - signal detection
    - graph search
    - node core ranking
    - branch expansion
    - RAM graph construction

## Invariants
1. `node_core` is immutable.
2. Agents may read graph memory and personality data but may not mutate `node_core`.
3. Personality is stable and only changes through gradual updates.
4. Context is built as:
   - query
   - signal extraction
   - graph search
   - node hydration
   - branch expansion
   - RAM context graph
5. The LLM should receive RAM context, not the raw long-term graph.

## Runtime Bridge
The active behavioral runtime in `roaches_viz/roaches_viz/graph_rag.py` now serializes RAM context through the core model.
This keeps the current system alive while moving the architecture under one contract.
The same runtime now also builds:
- `SpeechDNA`
- scenario frames
- professional agent role context
- a dialogue contract used by the reasoning path

## Next Steps
1. Move active graph search and branch expansion into explicit `ContextCore` ports.
2. Move personality serialization in the runtime to `PersonalityCore` everywhere.
3. Make controller validation operate on `AgentMutationProposal` instead of ad hoc dicts.
