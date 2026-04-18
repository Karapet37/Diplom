"""
CognitiveAuthority — measures how much the cognitive pipeline should control
the response vs. letting the LLM answer freely.

Score: 0.0 (fresh start, pure LLM) → 1.0 (rich history, pipeline controls)

Factors
-------
- session_turns      : more dialogue history → more context for the pipeline
- graph_nodes        : more known entities → pipeline has grounding to work with
- genome_version     : more calibration iterations → pipeline decisions are reliable
- training_tuples    : more labelled feedback → P1/P6 are better calibrated
- regulator_drift    : non-zero regulators mean genuine state has accumulated

At score < PURE_LLM_THRESHOLD  : LLM answers directly (no planner, no hints)
At score < HINT_THRESHOLD       : LLM answers with cognitive hints injected
At score ≥ HINT_THRESHOLD       : SpeechPlanner controls, LLM only verbalizes
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

PURE_LLM_THRESHOLD = 0.20   # below this: plain LLM, zero pipeline interference
HINT_THRESHOLD     = 0.55   # below this: LLM + cognitive hints; above: full planner


def cognitive_authority_score(
    session_turns:   int,
    graph_nodes:     int,
    genome_version:  int,
    training_tuples: int,
    regulator_values: list[float] | None = None,
) -> float:
    """
    Returns a score in [0, 1] representing how much the cognitive pipeline
    should control the response.

    All signals are soft-capped with log so a single large value doesn't dominate.
    """

    def _soft(x: float, scale: float) -> float:
        """Map [0,∞) → [0,1) with log scaling."""
        return min(1.0, math.log1p(max(0.0, x) / scale) / math.log1p(1.0))

    # Weights: turns and nodes matter most for grounding
    w_turns   = 0.35 * _soft(session_turns,   scale=8.0)
    w_nodes   = 0.30 * _soft(graph_nodes,     scale=12.0)
    w_genome  = 0.15 * _soft(genome_version,  scale=3.0)
    w_tuples  = 0.15 * _soft(training_tuples, scale=15.0)

    # Regulator drift: if regulators have moved away from zero, state is real
    w_reg = 0.0
    if regulator_values:
        drift = sum(abs(v) for v in regulator_values) / (len(regulator_values) + 1e-9)
        w_reg = 0.05 * min(1.0, drift * 5.0)

    return round(w_turns + w_nodes + w_genome + w_tuples + w_reg, 3)


def authority_mode(score: float) -> str:
    """
    Returns one of:
      'pure_llm'   — no pipeline involvement, LLM answers freely
      'hint'       — LLM answers, cognitive hints injected as soft guidance
      'planner'    — SpeechPlanner controls content, LLM only verbalizes
    """
    if score < PURE_LLM_THRESHOLD:
        return 'pure_llm'
    if score < HINT_THRESHOLD:
        return 'hint'
    return 'planner'


def _count_training_tuples(memory_root: Path) -> int:
    p = memory_root / 'training_tuples.jsonl'
    if not p.exists():
        return 0
    try:
        return sum(1 for _ in p.open())
    except Exception:
        return 0


def build_cognitive_hint_block(cog_vc: dict[str, Any]) -> str:
    """
    Lightweight guidance injected into the LLM prompt at 'hint' authority level.
    Does not replace the prompt — just adds a brief internal-state note.
    Keeps the LLM in control but nudges it toward the pipeline's decision.
    """
    action   = cog_vc.get('action', '')
    event    = cog_vc.get('primary_event', 'neutral')
    risk     = float(cog_vc.get('perceived_risk', 0.5))
    conf     = float(cog_vc.get('confidence', 0.5))
    res      = cog_vc.get('dominant_resolution', '')

    tone_hint = ''
    if risk > 0.65:
        tone_hint = 'Be careful and measured in your reply.'
    elif conf < 0.35:
        tone_hint = 'It is okay to be uncertain — hedge where appropriate.'

    action_map = {
        'connect':       'The persona is inclined to connect.',
        'withdraw':      'The persona needs some distance.',
        'analyze':       'The persona wants to understand before responding.',
        'placate':       'The persona wants to reduce tension.',
        'freeze':        'The persona is taking time to process.',
        'seek_control':  'The persona wants to establish a clear frame.',
        'reframe':       'The persona sees a different angle on this.',
    }
    action_hint = action_map.get(action, '')

    parts = [p for p in [action_hint, tone_hint] if p]
    if not parts:
        return ''
    return '[Guidance: ' + ' '.join(parts) + ']'
