# Roadmap

## Phase 1: Stabilize core runtime

- [x] Fix broken intra-package imports in `src/*`.
- [x] Add lazy loading and offline fallbacks for heavy NLP modules.
- [x] Harden code sandbox against empty-result process exits.
- [x] Add baseline unit tests for critical code paths.
- [x] Add speech-to-text correction layer (sound anchors + context + optional LLM).
- [x] Add PyQt UI for parallel recording/transcription and live correction preview.
- [x] Add confidence scoring and top-3 STT alternatives.

## Phase 2: Standardize architecture

- [ ] Introduce shared config module and environment schema.
- [ ] Align all entrypoints (`start.py`, PyQt mode) to one command-line interface.
- [ ] Reduce duplicated DB schemas across offline agent modules.
- [ ] Add package initialization and explicit public APIs per module.

## Phase 3: Quality and observability

- [ ] Add structured logging and request correlation IDs.
- [ ] Add metrics for latency and memory footprint in long sessions.
- [ ] Expand tests to include integration coverage for SQLite and tool execution.
- [ ] Add CI pipeline for lint + tests.

## Phase 4: Product readiness

- [ ] Add model profile presets (CPU-only, GPU, low-RAM).
- [ ] Add safety policy and tool execution permissions matrix.
- [ ] Publish user-facing usage guide and troubleshooting docs.
