"""Personality genome: learnable parameters that define a persona's inner dynamics."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_DRIFT_STEP   = 0.08
NOISE_FLOOR      = 0.001
ELASTIC_COEFF    = 0.015   # pull toward prior per update cycle

# ── Learnable parameter ───────────────────────────────────────────────────────

@dataclass
class LearnableParam:
    name:       str
    value:      float
    prior:      float
    lr:         float   = 0.03
    stability:  float   = 0.5    # 0=free  1=frozen
    confidence: float   = 0.3
    lo:         float   = 0.0
    hi:         float   = 1.0
    updates:    int     = 0
    history:    list    = field(default_factory=list)  # [(ts, delta, source)]

    def step(self, delta: float, source: str = '') -> None:
        effective = delta * self.lr * (1.0 - self.stability)
        effective = float(np.clip(effective, -MAX_DRIFT_STEP, MAX_DRIFT_STEP))
        if abs(effective) < NOISE_FLOOR:
            return
        old = self.value
        self.value = float(np.clip(self.value + effective, self.lo, self.hi))
        self.updates += 1
        self.confidence = min(0.95, self.confidence + 0.005)
        self.history.append((round(time.time(), 2), round(effective, 5), source or ''))
        if len(self.history) > 200:
            self.history = self.history[-200:]
        _ = old  # suppress unused warning

    def elastic_pull(self) -> float:
        """Pull value back toward prior — regularisation."""
        dev = self.value - self.prior
        return -ELASTIC_COEFF * dev * (1.0 - self.stability)

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name, 'value': round(self.value, 6),
            'prior': round(self.prior, 6), 'lr': self.lr,
            'stability': self.stability, 'confidence': round(self.confidence, 4),
            'updates': self.updates,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'LearnableParam':
        p = cls(
            name=d['name'], value=float(d['value']), prior=float(d['prior']),
            lr=float(d.get('lr', 0.03)), stability=float(d.get('stability', 0.5)),
            confidence=float(d.get('confidence', 0.3)),
        )
        p.updates = int(d.get('updates', 0))
        return p


def _p(name: str, v: float, lr: float = 0.03, stability: float = 0.5) -> LearnableParam:
    return LearnableParam(name=name, value=v, prior=v, lr=lr, stability=stability)


# ── Genome ────────────────────────────────────────────────────────────────────

@dataclass
class PersonalityGenome:
    persona_id: str
    version:    int = 0

    # A. Goal drives
    drive_recognition:  LearnableParam = field(default_factory=lambda: _p('drive_recognition',  0.5))
    drive_security:     LearnableParam = field(default_factory=lambda: _p('drive_security',     0.5))
    drive_control:      LearnableParam = field(default_factory=lambda: _p('drive_control',      0.5))
    drive_closeness:    LearnableParam = field(default_factory=lambda: _p('drive_closeness',    0.5))
    drive_autonomy:     LearnableParam = field(default_factory=lambda: _p('drive_autonomy',     0.5))
    drive_superiority:  LearnableParam = field(default_factory=lambda: _p('drive_superiority',  0.3))
    drive_stability:    LearnableParam = field(default_factory=lambda: _p('drive_stability',    0.5))
    drive_meaning:      LearnableParam = field(default_factory=lambda: _p('drive_meaning',      0.5))

    # B. Fear drives
    fear_shame:             LearnableParam = field(default_factory=lambda: _p('fear_shame',             0.5, stability=0.7))
    fear_rejection:         LearnableParam = field(default_factory=lambda: _p('fear_rejection',         0.5, stability=0.7))
    fear_loss_of_control:   LearnableParam = field(default_factory=lambda: _p('fear_loss_of_control',   0.5, stability=0.6))
    fear_helplessness:      LearnableParam = field(default_factory=lambda: _p('fear_helplessness',      0.4, stability=0.6))
    fear_abandonment:       LearnableParam = field(default_factory=lambda: _p('fear_abandonment',       0.4, stability=0.7))
    fear_judgment:          LearnableParam = field(default_factory=lambda: _p('fear_judgment',          0.5, stability=0.6))
    fear_chaos:             LearnableParam = field(default_factory=lambda: _p('fear_chaos',             0.4, stability=0.6))
    fear_failure:           LearnableParam = field(default_factory=lambda: _p('fear_failure',           0.5, stability=0.6))

    # C. Cognitive style
    planning_depth:         LearnableParam = field(default_factory=lambda: _p('planning_depth',         0.5))
    analysis_bias:          LearnableParam = field(default_factory=lambda: _p('analysis_bias',          0.5))
    impulsivity:            LearnableParam = field(default_factory=lambda: _p('impulsivity',            0.3))
    ambiguity_tolerance:    LearnableParam = field(default_factory=lambda: _p('ambiguity_tolerance',    0.5))
    category_rigidity:      LearnableParam = field(default_factory=lambda: _p('category_rigidity',      0.4))
    generalization_bias:    LearnableParam = field(default_factory=lambda: _p('generalization_bias',    0.3))
    suspicion_bias:         LearnableParam = field(default_factory=lambda: _p('suspicion_bias',         0.3))
    threat_first:           LearnableParam = field(default_factory=lambda: _p('threat_first',           0.4))
    hypothesis_switch_speed:LearnableParam = field(default_factory=lambda: _p('hypothesis_switch_speed',0.4))

    # D. Social regulation
    hierarchy_sensitivity:  LearnableParam = field(default_factory=lambda: _p('hierarchy_sensitivity',  0.4))
    approval_seeking:       LearnableParam = field(default_factory=lambda: _p('approval_seeking',       0.4))
    dominance_tendency:     LearnableParam = field(default_factory=lambda: _p('dominance_tendency',     0.3))
    vulnerability_concealment: LearnableParam = field(default_factory=lambda: _p('vulnerability_concealment', 0.4))
    social_distance_default:LearnableParam = field(default_factory=lambda: _p('social_distance_default',0.5))
    trust_baseline:         LearnableParam = field(default_factory=lambda: _p('trust_baseline',         0.5))
    mirror_tendency:        LearnableParam = field(default_factory=lambda: _p('mirror_tendency',        0.4))

    # E. Defense mechanisms (weights, not booleans)
    defense_avoidance:      LearnableParam = field(default_factory=lambda: _p('defense_avoidance',      0.4))
    defense_rationalization:LearnableParam = field(default_factory=lambda: _p('defense_rationalization',0.4))
    defense_aggression:     LearnableParam = field(default_factory=lambda: _p('defense_aggression',     0.2))
    defense_freeze:         LearnableParam = field(default_factory=lambda: _p('defense_freeze',         0.3))
    defense_humor:          LearnableParam = field(default_factory=lambda: _p('defense_humor',          0.3))
    defense_hypercontrol:   LearnableParam = field(default_factory=lambda: _p('defense_hypercontrol',   0.3))

    # F. Energy / regulation baseline
    baseline_anxiety:       LearnableParam = field(default_factory=lambda: _p('baseline_anxiety',       0.3, stability=0.65))
    baseline_drive:         LearnableParam = field(default_factory=lambda: _p('baseline_drive',         0.5, stability=0.6))
    fatigue_sensitivity:    LearnableParam = field(default_factory=lambda: _p('fatigue_sensitivity',    0.4))
    stress_recovery_speed:  LearnableParam = field(default_factory=lambda: _p('stress_recovery_speed',  0.4))
    novelty_reward:         LearnableParam = field(default_factory=lambda: _p('novelty_reward',         0.4))
    shame_recovery_speed:   LearnableParam = field(default_factory=lambda: _p('shame_recovery_speed',   0.4))

    # G. Memory anchoring
    pain_memory_weight:     LearnableParam = field(default_factory=lambda: _p('pain_memory_weight',     0.6))
    shame_memory_weight:    LearnableParam = field(default_factory=lambda: _p('shame_memory_weight',    0.6))
    success_memory_weight:  LearnableParam = field(default_factory=lambda: _p('success_memory_weight',  0.5))
    past_influence_decay:   LearnableParam = field(default_factory=lambda: _p('past_influence_decay',   0.3))

    # H. Thought assembly biases  (range -1..1 mapped to 0..1 internally)
    threat_scan_first:      LearnableParam = field(default_factory=lambda: _p('threat_scan_first',      0.4))
    blame_self_vs_other:    LearnableParam = field(default_factory=lambda: _p('blame_self_vs_other',    0.5))  # 0=self 1=other
    feel_first:             LearnableParam = field(default_factory=lambda: _p('feel_first',             0.5))  # 0=think 1=feel
    justify_vs_plan:        LearnableParam = field(default_factory=lambda: _p('justify_vs_plan',        0.4))  # 0=plan 1=justify
    pessimism_bias:         LearnableParam = field(default_factory=lambda: _p('pessimism_bias',         0.4))  # 0=opt 1=pess

    def all_params(self) -> list[LearnableParam]:
        return [
            v for v in vars(self).values()
            if isinstance(v, LearnableParam)
        ]

    def to_vector(self) -> np.ndarray:
        return np.array([p.value for p in self.all_params()], dtype=np.float32)

    def apply_elastic_regularization(self) -> None:
        for p in self.all_params():
            p.step(p.elastic_pull(), source='elastic')
        self.version += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            'persona_id': self.persona_id,
            'version': self.version,
            'params': {p.name: p.to_dict() for p in self.all_params()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'PersonalityGenome':
        g = cls(persona_id=d['persona_id'], version=int(d.get('version', 0)))
        params_data = d.get('params', {})
        for p in g.all_params():
            if p.name in params_data:
                saved = LearnableParam.from_dict(params_data[p.name])
                p.value      = saved.value
                p.confidence = saved.confidence
                p.updates    = saved.updates
                p.history    = saved.history
        return g

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path: Path) -> 'PersonalityGenome':
        return cls.from_dict(json.loads(path.read_text(encoding='utf-8')))

    @classmethod
    def default_for(cls, persona_id: str) -> 'PersonalityGenome':
        return cls(persona_id=persona_id)
