"""
Offline calibration: trains P1 and P6 on collected TrainingTuples.
Pure numpy, no frameworks. Greedy layer-wise.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from .cognitive_pipeline import (
    ACTION_FAMILIES, EVENT_TYPES, N_ACTION, N_EVENT,
    CognitiveRuntime, RegulatorState,
)
from .genome import PersonalityGenome
from .training_store import TrainingStore, TrainingTuple

log = logging.getLogger(__name__)

LR_P1     = 0.01
LR_P6     = 0.01
EPOCHS_P1 = 30
EPOCHS_P6 = 30
MIN_SAMPLES = 20   # don't train with less


def _cross_entropy(probs: np.ndarray, target: int) -> float:
    return -math.log(float(probs[target]) + 1e-9)


def _ce_grad_softmax(probs: np.ndarray, target: int) -> np.ndarray:
    """Gradient of cross-entropy loss w.r.t. pre-softmax logits."""
    g = probs.copy()
    g[target] -= 1.0
    return g


# ── P1 trainer ────────────────────────────────────────────────────────────────

def train_p1(rt: CognitiveRuntime, tuples: list[TrainingTuple]) -> dict[str, Any]:
    """Train EventEncoder on labelled (text, event_label) pairs."""
    labelled = [t for t in tuples if t.event_label is not None]
    if len(labelled) < MIN_SAMPLES:
        return {'skipped': True, 'reason': f'only {len(labelled)} labelled samples'}

    losses = []
    for epoch in range(EPOCHS_P1):
        epoch_loss = 0.0
        np.random.shuffle(labelled)
        for tup in labelled:
            x = rt.p1._features(tup.text)
            logits = rt.p1.W @ x + rt.p1.b
            # manual softmax
            e = np.exp(logits - logits.max())
            probs = e / e.sum()

            loss = _cross_entropy(probs, tup.event_label)
            epoch_loss += loss

            # weight loss by |feedback|
            scale = LR_P1 * (1.0 + abs(tup.feedback))
            grad_logits = _ce_grad_softmax(probs, tup.event_label)
            rt.p1.W -= scale * np.outer(grad_logits, x)
            rt.p1.b -= scale * grad_logits

        losses.append(epoch_loss / len(labelled))

    return {
        'samples': len(labelled),
        'epochs':  EPOCHS_P1,
        'final_loss': round(losses[-1], 5),
        'initial_loss': round(losses[0], 5),
    }


# ── P6 trainer ────────────────────────────────────────────────────────────────

def train_p6(
    rt: CognitiveRuntime,
    tuples: list[TrainingTuple],
    genome_map: dict[str, PersonalityGenome],
) -> dict[str, Any]:
    """Train ActionPolicy on (thought, conflict, regulators → action_target) pairs."""
    labelled = [t for t in tuples if t.action_target is not None and t.thought_vec]
    if len(labelled) < MIN_SAMPLES:
        return {'skipped': True, 'reason': f'only {len(labelled)} labelled samples'}

    losses = []
    for epoch in range(EPOCHS_P6):
        epoch_loss = 0.0
        np.random.shuffle(labelled)
        for tup in labelled:
            genome = genome_map.get(tup.genome_id)
            if genome is None:
                continue

            thought   = np.array(tup.thought_vec,  dtype=np.float32)
            conflict  = np.array(tup.conflict_vec, dtype=np.float32)
            reg       = np.array(list(tup.regulator_snapshot.values()), dtype=np.float32)[:10]
            if len(reg) < 10:
                reg = np.pad(reg, (0, 10 - len(reg)))

            defenses = rt.p6._defense_vec(genome)
            x = np.concatenate([thought, conflict, reg, defenses]).astype(np.float32)

            # forward
            h      = np.maximum(0, rt.p6.W1 @ x + rt.p6.b1)
            logits = rt.p6.W2 @ h + rt.p6.b2
            e      = np.exp(logits - logits.max())
            probs  = e / e.sum()

            loss = _cross_entropy(probs, tup.action_target)
            epoch_loss += loss

            scale = LR_P6 * (1.0 + abs(tup.feedback))
            dl = _ce_grad_softmax(probs, tup.action_target)

            # backprop through W2
            dW2 = np.outer(dl, h)
            dh  = rt.p6.W2.T @ dl

            # backprop through relu + W1
            dh_pre = dh * (h > 0).astype(np.float32)
            dW1 = np.outer(dh_pre, x)

            rt.p6.W2 -= scale * dW2
            rt.p6.b2 -= scale * dl
            rt.p6.W1 -= scale * dW1
            rt.p6.b1 -= scale * dh_pre

        losses.append(epoch_loss / max(1, len(labelled)))

    return {
        'samples': len(labelled),
        'epochs': EPOCHS_P6,
        'final_loss': round(losses[-1], 5),
        'initial_loss': round(losses[0], 5),
    }


# ── Online genome update from feedback ────────────────────────────────────────

FEEDBACK_TYPE_MAP: dict[str, list[tuple[str, float]]] = {
    'too_cold':           [('social_distance_default', -0.04), ('drive_closeness', +0.03)],
    'too_warm':           [('social_distance_default', +0.04), ('drive_closeness', -0.03)],
    'too_rational':       [('feel_first', +0.06), ('analysis_bias', -0.04)],
    'too_emotional':      [('feel_first', -0.06), ('analysis_bias', +0.04)],
    'underestimated_fear':   [('fear_shame', +0.05), ('fear_rejection', +0.04)],
    'overestimated_control': [('fear_loss_of_control', -0.04), ('drive_control', -0.03)],
    'too_perfect':        [('impulsivity', +0.05), ('analysis_bias', -0.04)],
    'wrong_action_avoid': [('defense_avoidance', +0.05), ('baseline_anxiety', +0.03)],
    'wrong_action_attack':   [('defense_aggression', -0.06)],
    'spirit_mismatch':    [('feel_first', +0.03), ('vulnerability_concealment', -0.03)],
}


def apply_explicit_feedback(
    feedback_type: str,
    intensity: float,
    genome: PersonalityGenome,
) -> list[str]:
    """Apply named feedback to genome. Returns list of updated param names."""
    adjustments = FEEDBACK_TYPE_MAP.get(feedback_type, [])
    updated = []
    for param_name, base_delta in adjustments:
        param = getattr(genome, param_name, None)
        if param is not None:
            param.step(base_delta * intensity, source=f'explicit:{feedback_type}')
            updated.append(param_name)
    if updated:
        genome.version += 1
    return updated


def run_batch_calibration(
    rt: CognitiveRuntime,
    store: TrainingStore,
    genome_map: dict[str, PersonalityGenome],
) -> dict[str, Any]:
    """Full calibration pass over all stored training tuples."""
    tuples = store.load_all()
    if not tuples:
        return {'skipped': True, 'reason': 'no training data'}

    p1_result = train_p1(rt, tuples)
    p6_result = train_p6(rt, tuples, genome_map)

    return {
        'total_tuples': len(tuples),
        'p1': p1_result,
        'p6': p6_result,
    }
