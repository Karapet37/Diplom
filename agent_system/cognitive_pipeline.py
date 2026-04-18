"""
Cognitive pipeline: P1–P6 perceptron modules + regulator state.
No LLM inside. LLM is called only by the verbalizer (chat_engine.py).
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .genome import PersonalityGenome

# ── Helpers ───────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float32)))

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

# ── Event types ───────────────────────────────────────────────────────────────

EVENT_TYPES = [
    'criticism', 'praise', 'rejection', 'failure',
    'overload', 'uncertainty', 'intimacy', 'opportunity',
    'danger', 'shame_trigger', 'boredom', 'novelty',
    'loss_of_control', 'neutral',
]
N_EVENT = len(EVENT_TYPES)           # 14
EVENT_IDX = {e: i for i, e in enumerate(EVENT_TYPES)}

ACTION_FAMILIES = [
    'approach', 'avoid', 'freeze', 'attack', 'placate',
    'analyze', 'ask_for_help', 'seek_control', 'reduce_exposure',
    'reframe', 'self_protect', 'connect', 'withdraw', 'plan_small_step',
]
N_ACTION = len(ACTION_FAMILIES)      # 14
ACTION_IDX = {a: i for i, a in enumerate(ACTION_FAMILIES)}

N_REG    = 10   # regulators
N_GENOME = 53   # genome param count (matches PersonalityGenome.all_params())

# ── Keyword patterns for P1 ───────────────────────────────────────────────────

_EVENT_PATTERNS: dict[str, list[tuple[str, float]]] = {
    'criticism': [
        (r'\b(wrong|bad|awful|terrible|stupid|idiot|failure|weak|useless|incompetent)\b', 1.2),
        (r'\b(you (always|never)|your fault|blame you)\b', 1.0),
        (r'\b(criticize|judge|attack|mock|ridicule)\b', 0.9),
        (r'\bкрити[кч]\w*|\bглупо|неправильно|неверно|плохо\b', 0.9),
    ],
    'praise': [
        (r'\b(great|amazing|excellent|perfect|wonderful|brilliant|love|proud|well done)\b', 1.2),
        (r'\b(thank|appreciate|grateful|respect)\b', 0.9),
        (r'\bмолодец|отлично|замечательно|прекрасно|горжусь\b', 0.9),
    ],
    'rejection': [
        (r'\b(reject|refuse|deny|dismiss|ignore|leave|abandon|go away|don.t want)\b', 1.1),
        (r'\b(не нужен|уходи|оставь меня|не хочу)\b', 1.0),
    ],
    'failure': [
        (r'\b(fail|failed|mistake|error|broke|lost|miss|couldn.t|didn.t work)\b', 1.0),
        (r'\b(провал|ошибка|не получилось|не вышло)\b', 1.0),
    ],
    'overload': [
        (r'\b(too much|overwhelm|can.t handle|exhausted|burned out|too many)\b', 1.1),
        (r'\b(перегруз|слишком много|не справляюсь|устал)\b', 1.0),
    ],
    'uncertainty': [
        (r'\b(don.t know|unsure|unclear|confused|maybe|perhaps|not sure|what if)\b', 0.9),
        (r'\b(не знаю|непонятно|неясно|может быть|не уверен)\b', 0.9),
    ],
    'intimacy': [
        (r'\b(close to|trust (you|me|us|him|her|them)|love|care for|together|confide|vulnerable)\b', 1.0),
        (r'\b(trust (you|me|us)\b)', 1.1),
        (r'\b(близость|доверяю|люблю|вместе|открыться)\b', 1.0),
    ],
    'opportunity': [
        (r'\b(chance|opportunity|offer|possibility|could achieve|potential|open)\b', 1.0),
        (r'\b(шанс|возможность|предложение|могу достичь)\b', 1.0),
    ],
    'danger': [
        (r'\b(danger|threat|attack|hurt|harm|destroy|kill|violence|fear)\b', 1.3),
        (r'\b(опасность|угроза|нападение|вред|страх)\b', 1.2),
    ],
    'shame_trigger': [
        (r'\b(embarrass|shame|humiliat|pathetic|loser|worthless|disgrace)\b', 1.3),
        (r'\b(позор|стыд|унижение|ничтожество|жалкий)\b', 1.2),
    ],
    'boredom': [
        (r'\b(boring|bored|dull|tedious|same|again|repeat|whatever)\b', 0.8),
        (r'\b(скучно|надоело|опять|снова|одно и то же)\b', 0.8),
    ],
    'novelty': [
        (r'\b(new|never|first time|surprise|unexpected|discover|interesting)\b', 0.9),
        (r'\b(новое|первый раз|сюрприз|неожиданно|интересно)\b', 0.9),
    ],
    'loss_of_control': [
        (r'\b(control|order|chaos|mess|collapse|force|must|no choice)\b', 1.0),
        (r'\btrust the (plan|system|process|government|science)\b', 1.2),
        (r'\b(stop the steal|deep state|they.re hiding|wake up sheeple)\b', 1.3),
        (r'\b(контроль|порядок|хаос|нет выбора|заставил)\b', 1.0),
    ],
}

# ── Regulator state ───────────────────────────────────────────────────────────

REG_NAMES = [
    'anxiety', 'drive', 'fatigue', 'motivation', 'shame',
    'confidence', 'threat_sense', 'control_sense', 'closeness', 'frustration',
]

@dataclass
class RegulatorState:
    values: np.ndarray = field(default_factory=lambda: np.full(N_REG, 0.4, dtype=np.float32))
    updated_at: float = field(default_factory=time.time)

    def get(self, name: str) -> float:
        return float(self.values[REG_NAMES.index(name)])

    def set(self, name: str, v: float) -> None:
        self.values[REG_NAMES.index(name)] = float(np.clip(v, 0.0, 1.0))

    def to_dict(self) -> dict[str, float]:
        return {n: round(float(self.values[i]), 4) for i, n in enumerate(REG_NAMES)}

    def copy(self) -> 'RegulatorState':
        return RegulatorState(values=self.values.copy(), updated_at=self.updated_at)

    @classmethod
    def from_genome(cls, g: PersonalityGenome) -> 'RegulatorState':
        s = cls()
        s.set('anxiety',    g.baseline_anxiety.value)
        s.set('drive',      g.baseline_drive.value)
        s.set('motivation', g.baseline_drive.value)
        s.set('confidence', 1.0 - g.baseline_anxiety.value)
        return s

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding='utf-8')

    @classmethod
    def load(cls, path: Path) -> 'RegulatorState':
        d = json.loads(path.read_text(encoding='utf-8'))
        s = cls()
        for n, v in d.items():
            if n in REG_NAMES:
                s.set(n, float(v))
        return s


# ═══════════════════════════════════════════════════════════════════════════════
# P1 — Event Encoder
# ═══════════════════════════════════════════════════════════════════════════════

N_FEAT = 48   # feature vector dimension

class EventEncoder:
    """
    Learnable linear classifier: float[48] → float[14] event probabilities.
    Initialised from keyword patterns (rules → weights).
    """

    def __init__(self) -> None:
        self.W = np.zeros((N_EVENT, N_FEAT), dtype=np.float32)
        self.b = np.zeros(N_EVENT, dtype=np.float32)
        self._init_from_patterns()

    def _init_from_patterns(self) -> None:
        """Seed weights from keyword hit positions [0:14] in the feature vector."""
        for i, etype in enumerate(EVENT_TYPES):
            self.W[i, i] = 2.0   # diagonal: feature[i] → event[i]

    # ── Feature extraction (deterministic, not learnable) ──────────────────

    def _features(self, text: str, ctx: dict | None = None) -> np.ndarray:
        t = text.lower()
        x = np.zeros(N_FEAT, dtype=np.float32)

        # [0:14] keyword scores per event type
        for i, etype in enumerate(EVENT_TYPES):
            for pattern, weight in _EVENT_PATTERNS.get(etype, []):
                hits = len(re.findall(pattern, t, re.IGNORECASE))
                x[i] += hits * weight

        # [14:20] structural
        words = t.split()
        x[14] = min(1.0, len(words) / 50.0)
        x[15] = min(1.0, t.count('?') / 3.0)
        x[16] = min(1.0, t.count('!') / 3.0)
        x[17] = sum(1 for c in text if c.isupper()) / max(1, len(text))
        x[18] = min(1.0, len(re.findall(r'\b(not|no|never|нет|не|никогда)\b', t)) / 3.0)
        x[19] = min(1.0, len(re.findall(r'\b(do|tell|give|stop|show|помоги|скажи|стоп)\b', t)) / 2.0)

        # [20:26] person reference
        x[20] = min(1.0, len(re.findall(r'\b(i|me|my|myself|я|меня|мне|мой)\b', t)) / 5.0)
        x[21] = min(1.0, len(re.findall(r'\b(you|your|yourself|ты|тебя|твой)\b', t)) / 5.0)
        x[22] = min(1.0, len(re.findall(r'\b(he|she|they|он|она|они)\b', t)) / 3.0)
        x[23] = min(1.0, len(re.findall(r'\b(we|us|our|мы|нас|наш)\b', t)) / 3.0)
        x[24] = x[20]   # self-ref
        x[25] = x[21]   # other-ref

        # [26:32] temporal
        x[26] = min(1.0, len(re.findall(r'\b(was|were|had|did|был|была|сделал)\b', t)) / 3.0)
        x[27] = min(1.0, len(re.findall(r'\b(is|are|am|есть|является)\b', t)) / 3.0)
        x[28] = min(1.0, len(re.findall(r'\b(will|going to|shall|буду|хочу)\b', t)) / 3.0)
        x[29] = min(1.0, len(re.findall(r'\b(urgent|now|immediately|срочно|сейчас)\b', t)) / 2.0)
        x[30] = min(1.0, len(re.findall(r'\b(always|often|usually|всегда|часто)\b', t)) / 2.0)
        x[31] = min(1.0, len(re.findall(r'\b(long|years|forever|давно|долго|всю жизнь)\b', t)) / 2.0)

        # [32:38] intensity markers
        x[32] = min(1.0, len(re.findall(r'\b(very|extremely|absolutely|очень|крайне|совсем)\b', t)) / 3.0)
        x[33] = min(1.0, len(re.findall(r'\b(always|every|all|всегда|каждый|все)\b', t)) / 3.0)
        x[34] = min(1.0, len(re.findall(r'\b(never|nothing|none|никогда|ничего|никто)\b', t)) / 2.0)
        x[35] = min(1.0, len(re.findall(r'\b(most|best|worst|больше всего|лучший|худший)\b', t)) / 2.0)
        x[36] = min(1.0, len(re.findall(r'\b(still|keep|continue|всё ещё|продолжаю)\b', t)) / 2.0)
        x[37] = min(1.0, len(re.findall(r'\b(completely|totally|absolutely|полностью|совершенно)\b', t)) / 2.0)

        # [38:44] social context
        x[38] = min(1.0, len(re.findall(r'\b(everyone|public|people|все|люди|публично)\b', t)) / 2.0)
        x[39] = min(1.0, len(re.findall(r'\b(alone|private|secret|один|тайно|секрет)\b', t)) / 2.0)
        x[40] = min(1.0, len(re.findall(r'\b(better|worse|more|less|лучше|хуже|больше|меньше)\b', t)) / 3.0)
        x[41] = min(1.0, len(re.findall(r'\b(boss|leader|authority|начальник|власть|авторитет)\b', t)) / 2.0)
        x[42] = min(1.0, len(re.findall(r'\b(group|team|together|группа|команда|вместе)\b', t)) / 2.0)
        x[43] = min(1.0, len(re.findall(r'\b(someone|person|individual|кто-то|человек|личность)\b', t)) / 2.0)

        # [44:48] session context flags (passed via ctx dict)
        ctx = ctx or {}
        x[44] = float(bool(ctx.get('prior_failure')))
        x[45] = float(bool(ctx.get('repeated')))
        x[46] = float(bool(ctx.get('escalating')))
        x[47] = float(bool(ctx.get('resolved')))

        return x

    def encode(self, text: str, ctx: dict | None = None) -> tuple[np.ndarray, float]:
        """Returns (event_probs[14], intensity)."""
        x = self._features(text, ctx)
        logits = self.W @ x + self.b
        probs = _softmax(logits)
        intensity = float(np.clip(x[:N_EVENT].sum() / 5.0, 0.05, 1.0))
        return probs, intensity

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W=self.W, b=self.b)

    def load(self, path: Path) -> None:
        data = np.load(path)
        self.W = data['W']
        self.b = data['b']


# ═══════════════════════════════════════════════════════════════════════════════
# P2 — Trigger Network
# ═══════════════════════════════════════════════════════════════════════════════

class TriggerNetwork:
    """
    S[N_GENOME × (N_EVENT+6)] @ event_features  *  genome_vec  → trigger_activations
    Initialised from SENSITIVITY_MAP priors.
    """

    N_IN = N_EVENT + 6   # event probs + context flags

    def __init__(self, n_genes: int = N_GENOME) -> None:
        self.n_genes = n_genes
        self.S = np.zeros((n_genes, self.N_IN), dtype=np.float32)
        self._init_priors()

    def _init_priors(self) -> None:
        """Seed sensitivity from domain knowledge."""
        # Format: (gene_index, event_index, weight)
        priors = [
            (8,  0, 1.2),  # fear_shame     ← criticism
            (9,  0, 1.0),  # fear_rejection ← criticism
            (9,  2, 1.3),  # fear_rejection ← rejection
            (12, 2, 1.2),  # fear_abandonment ← rejection
            (10, 12, 1.4), # fear_loss_of_control ← loss_of_control
            (2,  12, -0.8),# drive_control ← loss_of_control (blocked)
            (8,  9, 1.5),  # fear_shame ← shame_trigger
            (11, 3, 1.0),  # fear_helplessness ← failure
            (7,  3, -0.6), # fear_failure ← failure
            (0,  1, 0.8),  # drive_recognition ← praise
            (3,  6, 0.9),  # drive_closeness ← intimacy
            (16, 5, 0.7),  # impulsivity ← uncertainty (amplifies)
            (18, 5, -0.5), # hypothesis_switch_speed ← uncertainty
            (13, 8, 0.8),  # fear_judgment ← danger
        ]
        for gi, ei, w in priors:
            if gi < self.n_genes and ei < self.N_IN:
                self.S[gi, ei] = w

    def activate(
        self,
        event_probs: np.ndarray,   # float[14]
        intensity: float,
        genome_vec: np.ndarray,    # float[N_GENOME]
        ctx_flags: np.ndarray,     # float[6]
    ) -> np.ndarray:
        ev = np.concatenate([event_probs, ctx_flags]).astype(np.float32)
        raw = self.S @ ev                           # float[N_GENOME]
        triggered = np.clip(raw * genome_vec * intensity, 0.0, 1.0)
        return triggered.astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.S)

    def load(self, path: Path) -> None:
        self.S = np.load(path)


# ═══════════════════════════════════════════════════════════════════════════════
# P3 — Regulator Cell (simplified GRU)
# ═══════════════════════════════════════════════════════════════════════════════

class RegulatorCell:
    """
    Simplified GRU over 10 regulators.
    Input: [triggers(N_GENOME), h_prev(10), energy_genes(6)]
    """

    N_ENERGY = 6
    N_IN     = N_GENOME + N_REG + N_ENERGY

    def __init__(self) -> None:
        scale = 0.1
        self.Wz = np.random.randn(N_REG, self.N_IN).astype(np.float32) * scale
        self.Wr = np.random.randn(N_REG, self.N_IN).astype(np.float32) * scale
        self.Wh = np.random.randn(N_REG, self.N_IN).astype(np.float32) * scale
        self.bz = np.zeros(N_REG, dtype=np.float32)
        self.br = np.zeros(N_REG, dtype=np.float32)
        self.bh = np.zeros(N_REG, dtype=np.float32)
        self._init_priors()

    def _init_priors(self) -> None:
        # fear_shame(8) → shame(4)
        self.Wh[4, 8] = 0.9
        # fear_rejection(9) → anxiety(0)
        self.Wh[0, 9] = 0.7
        # fear_loss_of_control(10) → anxiety(0), control_sense(7)
        self.Wh[0, 10] = 0.8
        self.Wh[7, 10] = -0.6
        # drive_recognition(0) → motivation(3)
        self.Wh[3, 0] = 0.5
        # drive_closeness(3) → closeness(8)
        self.Wh[8, 3] = 0.6
        # baseline_anxiety at energy index 0
        self.bh[0] = 0.1

    def _energy_vec(self, g: PersonalityGenome) -> np.ndarray:
        return np.array([
            g.baseline_anxiety.value,
            g.baseline_drive.value,
            g.fatigue_sensitivity.value,
            g.stress_recovery_speed.value,
            g.novelty_reward.value,
            g.shame_recovery_speed.value,
        ], dtype=np.float32)

    def step(
        self,
        triggers: np.ndarray,    # float[N_GENOME]
        h: np.ndarray,           # float[N_REG]
        genome: PersonalityGenome,
    ) -> np.ndarray:
        energy = self._energy_vec(genome)
        x = np.concatenate([triggers, h, energy]).astype(np.float32)
        z  = _sigmoid(self.Wz @ x + self.bz)
        r  = _sigmoid(self.Wr @ x + self.br)
        xr = np.concatenate([triggers, r * h, energy]).astype(np.float32)
        hc = np.tanh(self.Wh @ xr + self.bh)
        h_next = (1.0 - z) * h + z * hc
        return np.clip(h_next, 0.0, 1.0).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, Wz=self.Wz, Wr=self.Wr, Wh=self.Wh,
                 bz=self.bz, br=self.br, bh=self.bh)

    def load(self, path: Path) -> None:
        d = np.load(path)
        self.Wz, self.Wr, self.Wh = d['Wz'], d['Wr'], d['Wh']
        self.bz, self.br, self.bh = d['bz'], d['br'], d['bh']


# ═══════════════════════════════════════════════════════════════════════════════
# P4 — Thought MLP
# ═══════════════════════════════════════════════════════════════════════════════

N_COGNITIVE = 9   # cognitive genes subset
N_THOUGHT   = 16  # thought vector dimension

THOUGHT_NAMES = [
    'perceived_risk', 'confidence_in_frame',
    'dominant_need_goal',    # 0=security 1=recognition 2=control
    'dominant_need_social',  # 0=closeness 1=autonomy 2=stability
    'dominant_need_other',
    'frame_approach', 'frame_avoid', 'frame_freeze',
    'self_deception', 'action_urgency', 'social_concern',
    'planning_horizon', 'blame_direction', 'emotional_intensity',
    'control_assessment', 'novelty_seeking',
]

class ThoughtMLP:
    """
    3-layer MLP: [triggers, regulators, cognitive_genes] → thought_vector[16]
    """

    N_IN = N_GENOME + N_REG + N_COGNITIVE

    def __init__(self) -> None:
        scale = 0.05
        self.W1 = np.random.randn(32, self.N_IN).astype(np.float32) * scale
        self.b1 = np.zeros(32, dtype=np.float32)
        self.W2 = np.random.randn(N_THOUGHT, 32).astype(np.float32) * scale
        self.b2 = np.zeros(N_THOUGHT, dtype=np.float32)
        self._init_priors()

    def _init_priors(self) -> None:
        # threat_sense(reg 6) → perceived_risk(0)
        self.W1[0, N_GENOME + 6] = 1.2
        # shame(reg 4) → self_deception(8)
        self.W1[8, N_GENOME + 4] = 0.9
        # anxiety(reg 0) → frame_avoid(6)
        self.W1[6, N_GENOME + 0] = 1.0
        # motivation(reg 3) → action_urgency(9)
        self.W1[9, N_GENOME + 3] = 0.8
        # fatigue(reg 2) → planning_horizon(11) negative
        self.W1[11, N_GENOME + 2] = -0.7
        # confidence(reg 5) → confidence_in_frame(1)
        self.W1[1, N_GENOME + 5] = 0.9

    def _cognitive_genes(self, g: PersonalityGenome) -> np.ndarray:
        return np.array([
            g.planning_depth.value,
            g.analysis_bias.value,
            g.impulsivity.value,
            g.ambiguity_tolerance.value,
            g.threat_first.value,
            g.feel_first.value,
            g.justify_vs_plan.value,
            g.pessimism_bias.value,
            g.blame_self_vs_other.value,
        ], dtype=np.float32)

    def forward(
        self,
        triggers: np.ndarray,
        regulators: np.ndarray,
        genome: PersonalityGenome,
    ) -> np.ndarray:
        cog = self._cognitive_genes(genome)
        x   = np.concatenate([triggers, regulators, cog]).astype(np.float32)
        h   = _relu(self.W1 @ x + self.b1)
        raw = self.W2 @ h + self.b2

        out = np.zeros(N_THOUGHT, dtype=np.float32)
        out[0]   = float(_sigmoid(raw[0]))    # perceived_risk
        out[1]   = float(_sigmoid(raw[1]))    # confidence_in_frame
        out[2:5] = _softmax(raw[2:5])         # dominant_need group
        out[5:8] = _softmax(raw[5:8])         # frame direction
        out[8:]  = _sigmoid(raw[8:])          # rest as scalars
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path: Path) -> None:
        d = np.load(path)
        self.W1, self.b1, self.W2, self.b2 = d['W1'], d['b1'], d['W2'], d['b2']


# ═══════════════════════════════════════════════════════════════════════════════
# P5 — Conflict Scorer
# ═══════════════════════════════════════════════════════════════════════════════

N_CONFLICT = 8   # [intensity, 7 resolutions]
RESOLUTION_NAMES = [
    'avoidance', 'overcompensation', 'attack', 'freeze',
    'planning', 'support_seeking', 'self_deception',
]

# Which action families are blocked by each resolution
RESOLUTION_BLOCKS: dict[str, list[str]] = {
    'freeze':           ['approach', 'attack', 'connect'],
    'avoidance':        ['approach', 'connect', 'attack'],
    'overcompensation': ['placate', 'ask_for_help', 'withdraw'],
    'attack':           ['placate', 'ask_for_help', 'connect'],
    'self_deception':   ['analyze', 'reframe'],
}

class ConflictScorer:
    """
    Perceptron: [thought(16), goals(8), fears(8), regulators(10)] → conflict_vector[8]
    """

    N_IN = N_THOUGHT + 8 + 8 + N_REG

    def __init__(self) -> None:
        scale = 0.05
        self.W = np.random.randn(N_CONFLICT, self.N_IN).astype(np.float32) * scale
        self.b = np.zeros(N_CONFLICT, dtype=np.float32)
        self._init_priors()

    def _init_priors(self) -> None:
        off = N_THOUGHT
        # fear_shame(fears[0]) → freeze
        self.W[4, off + 8 + 0] = 1.0   # fears start at off+8
        # fear_rejection → avoidance
        self.W[1, off + 8 + 1] = 0.9
        # fatigue(reg 2) → freeze
        self.W[4, off + 8 + 8 + 2] = 0.8
        # motivation(reg 3) → planning
        self.W[5, off + 8 + 8 + 3] = 0.9
        # frustration(reg 9) → attack
        self.W[3, off + 8 + 8 + 9] = 0.9
        # approval_seeking gene (social[1]) → support_seeking
        # mapped roughly through goals
        self.W[6, off + 3] = 0.7   # drive_closeness in goals

    def _goals_vec(self, g: PersonalityGenome) -> np.ndarray:
        return np.array([
            g.drive_recognition.value, g.drive_security.value,
            g.drive_control.value, g.drive_closeness.value,
            g.drive_autonomy.value, g.drive_superiority.value,
            g.drive_stability.value, g.drive_meaning.value,
        ], dtype=np.float32)

    def _fears_vec(self, g: PersonalityGenome) -> np.ndarray:
        return np.array([
            g.fear_shame.value, g.fear_rejection.value,
            g.fear_loss_of_control.value, g.fear_helplessness.value,
            g.fear_abandonment.value, g.fear_judgment.value,
            g.fear_chaos.value, g.fear_failure.value,
        ], dtype=np.float32)

    def score(
        self,
        thought: np.ndarray,
        genome: PersonalityGenome,
        regulators: np.ndarray,
    ) -> np.ndarray:
        goals = self._goals_vec(genome)
        fears = self._fears_vec(genome)
        x = np.concatenate([thought, goals, fears, regulators]).astype(np.float32)
        raw = self.W @ x + self.b
        out = np.zeros(N_CONFLICT, dtype=np.float32)
        out[0]  = float(_sigmoid(raw[0]))    # intensity
        out[1:] = _softmax(raw[1:])          # resolution probs
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W=self.W, b=self.b)

    def load(self, path: Path) -> None:
        d = np.load(path); self.W, self.b = d['W'], d['b']


# ═══════════════════════════════════════════════════════════════════════════════
# P6 — Action Policy
# ═══════════════════════════════════════════════════════════════════════════════

class ActionPolicy:
    """
    MLP: [thought(16), conflict(8), regulators(10), defenses(6)] → action_logits[14]
    """

    N_IN = N_THOUGHT + N_CONFLICT + N_REG + 6

    def __init__(self) -> None:
        scale = 0.05
        self.W1 = np.random.randn(24, self.N_IN).astype(np.float32) * scale
        self.b1 = np.zeros(24, dtype=np.float32)
        self.W2 = np.random.randn(N_ACTION, 24).astype(np.float32) * scale
        self.b2 = np.zeros(N_ACTION, dtype=np.float32)
        self._init_priors()

    def _init_priors(self) -> None:
        off_c = N_THOUGHT
        off_r = off_c + N_CONFLICT
        # anxiety(reg 0) → avoid(1)
        self.W1[1, off_r + 0] = 1.0
        # motivation(reg 3) → approach(0)
        self.W1[0, off_r + 3] = 1.0
        # fatigue(reg 2) → freeze(2)
        self.W1[2, off_r + 2] = 0.9
        # frustration(reg 9) → attack(3)
        self.W1[3, off_r + 9] = 0.9
        # closeness(reg 8) → connect(11)
        self.W1[11, off_r + 8] = 0.9
        # shame(reg 4) → withdraw(12)
        self.W1[12, off_r + 4] = 0.8
        # planning from conflict(5) → plan_small_step(13)
        self.W1[13, off_c + 5] = 0.9
        # support_seeking(6) → ask_for_help(6)
        self.W1[6, off_c + 6] = 0.9

    def _defense_vec(self, g: PersonalityGenome) -> np.ndarray:
        return np.array([
            g.defense_avoidance.value,
            g.defense_rationalization.value,
            g.defense_aggression.value,
            g.defense_freeze.value,
            g.defense_humor.value,
            g.defense_hypercontrol.value,
        ], dtype=np.float32)

    def select(
        self,
        thought: np.ndarray,
        conflict: np.ndarray,
        regulators: np.ndarray,
        genome: PersonalityGenome,
        deterministic: bool = True,
    ) -> tuple[int, np.ndarray]:
        defenses = self._defense_vec(genome)
        x = np.concatenate([thought, conflict, regulators, defenses]).astype(np.float32)
        h = _relu(self.W1 @ x + self.b1)
        logits = self.W2 @ h + self.b2

        # Apply conflict blocking
        dominant_resolution = int(np.argmax(conflict[1:]))
        blocked_names = RESOLUTION_BLOCKS.get(RESOLUTION_NAMES[dominant_resolution], [])
        for name in blocked_names:
            if name in ACTION_IDX:
                logits[ACTION_IDX[name]] = -1e9

        probs = _softmax(logits)
        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(N_ACTION, p=probs))
        return action, probs

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path: Path) -> None:
        d = np.load(path)
        self.W1, self.b1, self.W2, self.b2 = d['W1'], d['b1'], d['W2'], d['b2']


# ═══════════════════════════════════════════════════════════════════════════════
# CognitiveRuntime — full forward pass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CognitiveTurnOutput:
    action_id:         int
    action_name:       str
    action_probs:      list[float]
    thought_vec:       list[float]
    conflict_vec:      list[float]
    regulator_state:   dict[str, float]
    event_probs:       list[float]
    primary_event:     str
    intensity:         float
    perceived_risk:    float
    dominant_resolution: str
    blocked_actions:   list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            'action_id':            self.action_id,
            'action_name':          self.action_name,
            'action_probs':         [round(float(p), 4) for p in self.action_probs],
            'thought_vec':          [round(float(v), 4) for v in self.thought_vec],
            'conflict_vec':         [round(float(v), 4) for v in self.conflict_vec],
            'regulator_state':      self.regulator_state,
            'event_probs':          [round(float(p), 4) for p in self.event_probs],
            'primary_event':        self.primary_event,
            'intensity':            round(self.intensity, 4),
            'perceived_risk':       round(self.perceived_risk, 4),
            'dominant_resolution':  self.dominant_resolution,
            'blocked_actions':      self.blocked_actions,
        }

    def verbalization_context(self) -> dict[str, Any]:
        """Minimal dict passed to the LLM verbalizer."""
        return {
            'action':              self.action_name,
            'perceived_risk':      round(self.perceived_risk, 2),
            'urgency':             round(float(self.thought_vec[9]), 2),
            'shame':               round(self.regulator_state.get('shame', 0.0), 2),
            'confidence':          round(self.regulator_state.get('confidence', 0.0), 2),
            'closeness':           round(self.regulator_state.get('closeness', 0.0), 2),
            'control':             round(self.regulator_state.get('control_sense', 0.0), 2),
            'primary_event':       self.primary_event,
            'dominant_resolution': self.dominant_resolution,
        }


class CognitiveRuntime:
    """
    Assembles P1–P6 and runs the full cognitive forward pass.
    Regulator state is session-scoped and updated in-place.
    """

    def __init__(self, seed: int = 2) -> None:
        _rng_state = np.random.get_state()
        np.random.seed(seed)
        try:
            self.p1 = EventEncoder()
            self.p2 = TriggerNetwork()
            self.p3 = RegulatorCell()
            self.p4 = ThoughtMLP()
            self.p5 = ConflictScorer()
            self.p6 = ActionPolicy()
        finally:
            np.random.set_state(_rng_state)

    def forward(
        self,
        text: str,
        genome: PersonalityGenome,
        regulator_state: RegulatorState,
        ctx: dict | None = None,
        deterministic: bool = True,
    ) -> tuple[CognitiveTurnOutput, RegulatorState]:

        # P1 — encode event
        event_probs, intensity = self.p1.encode(text, ctx)
        primary_event = EVENT_TYPES[int(np.argmax(event_probs))]
        ctx_flags = np.array([
            float(bool((ctx or {}).get('prior_failure'))),
            float(bool((ctx or {}).get('repeated'))),
            float(bool((ctx or {}).get('escalating'))),
            float(bool((ctx or {}).get('resolved'))),
            intensity,
            float(np.max(event_probs)),
        ], dtype=np.float32)

        # P2 — trigger genes
        genome_vec = genome.to_vector()
        triggers = self.p2.activate(event_probs, intensity, genome_vec, ctx_flags)

        # P3 — update regulators
        new_reg_values = self.p3.step(triggers, regulator_state.values, genome)
        new_reg = RegulatorState(values=new_reg_values, updated_at=time.time())

        # P4 — form thought
        thought = self.p4.forward(triggers, new_reg_values, genome)

        # P5 — score conflict
        conflict = self.p5.score(thought, genome, new_reg_values)
        dominant_res_idx = int(np.argmax(conflict[1:]))
        dominant_resolution = RESOLUTION_NAMES[dominant_res_idx]
        blocked = RESOLUTION_BLOCKS.get(dominant_resolution, [])

        # P6 — select action
        action_id, action_probs = self.p6.select(thought, conflict, new_reg_values, genome, deterministic)

        out = CognitiveTurnOutput(
            action_id=action_id,
            action_name=ACTION_FAMILIES[action_id],
            action_probs=action_probs.tolist(),
            thought_vec=thought.tolist(),
            conflict_vec=conflict.tolist(),
            regulator_state=new_reg.to_dict(),
            event_probs=event_probs.tolist(),
            primary_event=primary_event,
            intensity=float(intensity),
            perceived_risk=float(thought[0]),
            dominant_resolution=dominant_resolution,
            blocked_actions=blocked,
        )
        return out, new_reg

    def save(self, base_dir: Path) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        self.p1.save(base_dir / 'p1_event.npz')
        self.p2.save(base_dir / 'p2_trigger.npy')
        self.p3.save(base_dir / 'p3_regulator.npz')
        self.p4.save(base_dir / 'p4_thought.npz')
        self.p5.save(base_dir / 'p5_conflict.npz')
        self.p6.save(base_dir / 'p6_action.npz')

    def load(self, base_dir: Path) -> None:
        def _try(fn: Any, path: Path) -> None:
            if path.exists():
                fn(path)
        _try(self.p1.load, base_dir / 'p1_event.npz')
        _try(self.p2.load, base_dir / 'p2_trigger.npy')
        _try(self.p3.load, base_dir / 'p3_regulator.npz')
        _try(self.p4.load, base_dir / 'p4_thought.npz')
        _try(self.p5.load, base_dir / 'p5_conflict.npz')
        _try(self.p6.load, base_dir / 'p6_action.npz')
