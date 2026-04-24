#!/usr/bin/env python3
"""
Offline training script: DataSets → calibrated P1 + P6 pipeline weights.

Usage:
    python scripts/train_pipeline.py [--max-train N] [--epochs N] [--out-dir PATH]

Steps:
  1. Load train split from all dataset sources.
  2. Evaluate P1 and P6 accuracy BEFORE training (baseline).
  3. Train P1 (EventEncoder) on event-labelled examples.
  4. Re-run P1-P5 to produce thought/conflict/regulator vectors.
  5. Train P6 (ActionPolicy) on action-labelled examples.
  6. Evaluate P1 and P6 accuracy AFTER training.
  7. Save calibrated weights to --out-dir.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from agent_system.cognitive_pipeline import (
    ACTION_FAMILIES, EVENT_TYPES, CognitiveRuntime, RegulatorState,
)
from agent_system.dataset_layer import (
    build_dataset, evaluate_p1, evaluate_p6, label_distribution,
)
from agent_system.genome import PersonalityGenome
from agent_system.perceptron_trainer import (
    LR_P1, LR_P6, _ce_grad_softmax, _cross_entropy,
)
from agent_system.training_store import TrainingTuple

log = logging.getLogger('train_pipeline')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# ─── Trainer implementations ──────────────────────────────────────────────────

def _train_p1_on_tuples(
    rt: CognitiveRuntime,
    tuples: list[TrainingTuple],
    epochs: int = 30,
    lr: float = LR_P1,
) -> dict:
    labelled = [t for t in tuples if t.event_label is not None and t.text]
    if not labelled:
        return {'skipped': True, 'reason': f'no labelled examples'}

    rng = np.random.default_rng(0)
    losses: list[float] = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        idxs = rng.permutation(len(labelled))
        for i in idxs:
            t = labelled[i]
            x = rt.p1._features(t.text)
            logits = rt.p1.W @ x + rt.p1.b
            e = np.exp(logits - logits.max())
            probs = e / e.sum()

            loss = _cross_entropy(probs, t.event_label)
            epoch_loss += loss

            scale = lr * (1.0 + abs(t.feedback))
            grad  = _ce_grad_softmax(probs, t.event_label)
            rt.p1.W -= scale * np.outer(grad, x)
            rt.p1.b -= scale * grad

        avg = epoch_loss / len(labelled)
        losses.append(avg)
        if (epoch + 1) % 5 == 0:
            log.info('P1 epoch %d/%d  loss=%.4f', epoch + 1, epochs, avg)

    return {
        'samples':      len(labelled),
        'epochs':       epochs,
        'initial_loss': round(losses[0],  5),
        'final_loss':   round(losses[-1], 5),
    }


def _fill_pipeline_vectors(
    rt: CognitiveRuntime,
    tuples: list[TrainingTuple],
    genome: PersonalityGenome,
) -> int:
    """Run P1-P5 on each tuple, fill thought_vec/conflict_vec/regulator_snapshot."""
    reg = RegulatorState.from_genome(genome)
    filled = 0
    for t in tuples:
        if not t.text or t.action_target is None:
            continue
        try:
            cog_out, new_reg = rt.forward(t.text, genome, reg)
            t.thought_vec         = list(cog_out.thought_vec)
            t.conflict_vec        = list(cog_out.conflict_vec)
            t.regulator_snapshot  = dict(new_reg.to_dict())
            t.action_id           = cog_out.action_id
            filled += 1
        except Exception:
            pass
    return filled


def _train_p6_on_tuples(
    rt: CognitiveRuntime,
    tuples: list[TrainingTuple],
    genome: PersonalityGenome,
    epochs: int = 30,
    lr: float = LR_P6,
) -> dict:
    labelled = [
        t for t in tuples
        if t.action_target is not None and t.thought_vec and t.conflict_vec
    ]
    if not labelled:
        return {'skipped': True, 'reason': 'no labelled examples with vectors'}

    rng = np.random.default_rng(1)
    losses: list[float] = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        idxs = rng.permutation(len(labelled))
        for i in idxs:
            t = labelled[i]
            thought  = np.array(t.thought_vec,  dtype=np.float32)
            conflict = np.array(t.conflict_vec, dtype=np.float32)
            reg_vals = np.array(list(t.regulator_snapshot.values()), dtype=np.float32)[:10]
            if len(reg_vals) < 10:
                reg_vals = np.pad(reg_vals, (0, 10 - len(reg_vals)))

            defenses = rt.p6._defense_vec(genome)
            x = np.concatenate([thought, conflict, reg_vals, defenses]).astype(np.float32)

            h      = np.maximum(0.0, rt.p6.W1 @ x + rt.p6.b1)
            logits = rt.p6.W2 @ h + rt.p6.b2
            e      = np.exp(logits - logits.max())
            probs  = e / e.sum()

            loss = _cross_entropy(probs, t.action_target)
            epoch_loss += loss

            scale = lr * (1.0 + abs(t.feedback))
            dl    = _ce_grad_softmax(probs, t.action_target)

            dW2   = np.outer(dl, h)
            dh    = rt.p6.W2.T @ dl
            dh_pre = dh * (h > 0).astype(np.float32)
            dW1   = np.outer(dh_pre, x)

            rt.p6.W2 -= scale * dW2
            rt.p6.b2 -= scale * dl
            rt.p6.W1 -= scale * dW1
            rt.p6.b1 -= scale * dh_pre

        avg = epoch_loss / len(labelled)
        losses.append(avg)
        if (epoch + 1) % 5 == 0:
            log.info('P6 epoch %d/%d  loss=%.4f', epoch + 1, epochs, avg)

    return {
        'samples':      len(labelled),
        'epochs':       epochs,
        'initial_loss': round(losses[0],  5),
        'final_loss':   round(losses[-1], 5),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Train P1+P6 cognitive pipeline on DataSets/*')
    parser.add_argument('--max-train', type=int, default=5000)
    parser.add_argument('--max-test',  type=int, default=1000)
    parser.add_argument('--epochs',    type=int, default=30)
    parser.add_argument('--lr-p1',     type=float, default=LR_P1)
    parser.add_argument('--lr-p6',     type=float, default=LR_P6)
    parser.add_argument('--out-dir',   type=str,   default='runtime/pipeline_weights')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────
    log.info('Loading training data (max=%d)...', args.max_train)
    train_data = build_dataset(split='train', max_empathetic=args.max_train, max_archive=2000)
    log.info('Loading test data (max=%d)...', args.max_test)
    test_data = build_dataset(split='test', max_empathetic=args.max_test, max_archive=300)

    log.info('Train: %s', label_distribution(train_data))
    log.info('Test:  %s', label_distribution(test_data))

    # ── 2. Baseline evaluation ────────────────────────────────────────────
    log.info('=== BASELINE ===')
    rt = CognitiveRuntime(seed=2)
    genome = PersonalityGenome('default')

    baseline_p1 = evaluate_p1(rt, test_data)
    baseline_p6 = evaluate_p6(rt, test_data, genome)
    log.info('P1 accuracy: %.1f%%  (n=%d)', baseline_p1['accuracy'] * 100, baseline_p1['n'])
    log.info('P6 accuracy: %.1f%%  (n=%d)', baseline_p6['accuracy'] * 100, baseline_p6['n'])

    # ── 3. Train P1 ───────────────────────────────────────────────────────
    log.info('=== TRAINING P1 ===')
    p1_result = _train_p1_on_tuples(rt, train_data, epochs=args.epochs, lr=args.lr_p1)
    log.info('P1 training: %s', p1_result)

    # ── 4. Fill pipeline vectors for P6 training ─────────────────────────
    log.info('Filling P1-P5 vectors for P6 training...')
    filled = _fill_pipeline_vectors(rt, train_data, genome)
    log.info('Filled %d vectors', filled)

    # ── 5. Train P6 ───────────────────────────────────────────────────────
    log.info('=== TRAINING P6 ===')
    p6_result = _train_p6_on_tuples(rt, train_data, genome, epochs=args.epochs, lr=args.lr_p6)
    log.info('P6 training: %s', p6_result)

    # ── 6. Post-training evaluation ───────────────────────────────────────
    log.info('=== POST-TRAINING ===')
    post_p1 = evaluate_p1(rt, test_data)
    post_p6 = evaluate_p6(rt, test_data, genome)
    log.info('P1 accuracy: %.1f%%  (n=%d)', post_p1['accuracy'] * 100, post_p1['n'])
    log.info('P6 accuracy: %.1f%%  (n=%d)', post_p6['accuracy'] * 100, post_p6['n'])
    log.info('P1 gain: %+.1f pp', (post_p1['accuracy'] - baseline_p1['accuracy']) * 100)
    log.info('P6 gain: %+.1f pp', (post_p6['accuracy'] - baseline_p6['accuracy']) * 100)

    # ── 7. Save weights ───────────────────────────────────────────────────
    rt.p1.save(out_dir / 'p1_event_encoder.npz')
    rt.p6.save(out_dir / 'p6_action_policy.npz')

    report = {
        'baseline': {'p1': baseline_p1, 'p6': baseline_p6},
        'post':     {'p1': post_p1,     'p6': post_p6},
        'training': {'p1': p1_result,   'p6': p6_result},
        'weights_saved_to': str(out_dir),
    }
    report_path = out_dir / 'training_report.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info('Report saved to %s', report_path)

    # ── 8. Success check ──────────────────────────────────────────────────
    p1_ok = post_p1['accuracy'] > 0.30
    p6_ok = post_p6['accuracy'] > 0.20
    log.info('P1 target >30%%: %s', 'PASS' if p1_ok else 'FAIL')
    log.info('P6 target >20%%: %s', 'PASS' if p6_ok else 'FAIL')

    if p1_ok and p6_ok:
        log.info('✓ Average success achieved')
    else:
        log.warning('⚠ Below target — consider more epochs or data')


if __name__ == '__main__':
    main()
