"""
p4_genome_updater.py — P4: авто-обновление генома из P-сигналов.

Новая задача: после каждого хода читает P1-P47 сигналы
и инкрементально обновляет параметры генома персонажа.

Логика:
  P13 (attack) стреляет → suspicion_bias ↑, trust_baseline ↓
  P32 (approach) стреляет → drive_closeness ↑
  P41 (masking) стреляет → vulnerability_concealment ↑
  P47:hidden_plea стреляет → approval_seeking ↑
  P15 (tension) стреляет → baseline_anxiety ↑
  ... и т.д.

Варианты (output):
  genome_updated    — геном изменился
  genome_stable     — сигналы нейтральны, геном не менялся
  genome_conflict   — сигналы противоречат текущему геному (напр. persona=cold но P16:care=high)

Вызывается из chat_engine после каждого ассистентского хода.
"""
from __future__ import annotations

from typing import Any

from .p_subsystem import PVariant, PVariantOutput, PFamilyOutput

# ── Mapping: P-сигнал → genome параметр → delta ───────────────────────────────
# Формат: (p_id, variant_label или None для dominant_score) → (genome_param, delta_if_fire)
# delta: положительный = увеличить, отрицательный = уменьшить
# Применяется только когда score > THRESHOLD

_THRESHOLD = 0.60

_SIGNAL_TO_GENOME: list[tuple[str, str | None, str, float]] = [
    # p_id,       variant,              genome_param,           delta
    ('F13', 'attack',                'suspicion_bias',         +0.04),
    ('F13', 'attack',                'trust_baseline',         -0.04),
    ('F13', 'attack',                'threat_first',           +0.03),
    ('F32', 'approach',              'drive_closeness',        +0.03),
    ('F32', 'emotional_approach',    'drive_closeness',        +0.04),
    ('F33', 'distancing',            'social_distance_default',+0.04),
    ('F33', 'protective_distancing', 'vulnerability_concealment', +0.03),
    ('F41', 'emotional_masking',     'vulnerability_concealment', +0.03),
    ('F41', 'masking',               'defense_avoidance',      +0.03),
    ('F47', 'hidden_plea',           'approval_seeking',       +0.03),
    ('F47', 'hidden_reproach',       'defense_aggression',     +0.02),
    ('F47', 'hidden_contempt',       'social_distance_default',+0.03),
    ('F47', 'hidden_fear',           'baseline_anxiety',       +0.03),
    ('F47', 'hidden_fear',           'vulnerability_concealment', +0.02),
    ('F15', 'tense',                 'baseline_anxiety',       +0.02),
    ('F15', 'overloaded',            'baseline_anxiety',       +0.04),
    ('F15', 'overloaded',            'impulsivity',            +0.03),
    ('F9',  'overconfidence',        'dominance_tendency',     +0.03),
    ('F9',  'quiet_certainty',       'dominance_tendency',     +0.02),
    ('F22', 'dominance',             'dominance_tendency',     +0.03),
    ('F22', 'dominance',             'drive_superiority',      +0.03),
    ('F12', 'vulnerability',         'vulnerability_concealment', -0.03),  # showing vulner.
    ('F12', 'controlled_vulnerability', 'vulnerability_concealment', -0.01),
    ('F14', 'self_restraint',        'impulsivity',            -0.03),
    ('F29', 'guilt_manipulation',    'defense_aggression',     +0.03),
    ('F29', 'guilt_manipulation',    'blame_self_vs_other',    +0.04),
]


def update_genome_from_signals(
    persona_name: str,
    p_outputs: dict[str, PFamilyOutput],
    lr_scale: float = 1.0,
) -> tuple[str, dict[str, float]]:
    """
    Читает P-сигналы текущего хода, обновляет genome.
    Возвращает: (variant_label, {param: new_value})
    """
    try:
        from .genome_validator import load_or_init_persona_genome, save_persona_genome
    except ImportError:
        return 'genome_stable', {}

    if not persona_name:
        return 'genome_stable', {}

    genome = load_or_init_persona_genome(persona_name)
    changes: dict[str, float] = {}

    for p_id, variant, param, delta in _SIGNAL_TO_GENOME:
        out = p_outputs.get(p_id)
        if out is None:
            continue

        # Get score for this specific variant
        score = out.scores.get(variant, 0.0)
        if score < _THRESHOLD:
            continue

        # Apply delta scaled by score and learning rate
        param_obj = getattr(genome, param, None)
        if param_obj is None or not hasattr(param_obj, 'value'):
            continue

        old_val = float(param_obj.value)
        effective_delta = delta * score * lr_scale
        new_val = max(param_obj.lo, min(param_obj.hi, old_val + effective_delta))

        if abs(new_val - old_val) > 0.001:
            param_obj.value = new_val
            # Record update in history
            param_obj.updates = int(getattr(param_obj, 'updates', 0)) + 1
            if hasattr(param_obj, 'history'):
                param_obj.history = (param_obj.history or [])[-9:] + [round(old_val, 3)]
            changes[param] = round(new_val, 3)

    if changes:
        try:
            save_persona_genome(persona_name, genome)
        except Exception:
            pass
        return 'genome_updated', changes

    # Check for conflict: persona has high social_distance but P32 (approach) is high
    conflict = False
    p32_score = p_outputs.get('F32', None)
    if p32_score and p32_score.dominant_score > 0.6:
        sd = getattr(getattr(genome, 'social_distance_default', None), 'value', 0.5)
        if sd > 0.75:
            conflict = True

    return ('genome_conflict' if conflict else 'genome_stable'), {}


# ── PVariant ───────────────────────────────────────────────────────────────────

class P4GenomeVariant(PVariant):
    """
    P4 variant: запускает genome updater и возвращает результат как score.
    Вызывается когда persona_name доступен в context.
    """

    def __init__(self, variant_id: str, label: str):
        super().__init__(variant_id, label)
        self._trained = True

    def forward(self, text: str, context: Any) -> PVariantOutput:
        ctx = context if isinstance(context, dict) else {}

        # Cached result — все три варианта используют один вызов
        cached = ctx.get('_f4_genome_result')
        if cached is None:
            persona_name = str(ctx.get('persona_name') or ctx.get('selected_persona') or '').strip()
            p_outputs: dict[str, PFamilyOutput] = ctx.get('_f_outputs_snapshot', {})

            if persona_name and p_outputs:
                label_result, changes = update_genome_from_signals(persona_name, p_outputs)
                cached = (label_result, changes)
            else:
                cached = ('genome_stable', {})
            ctx['_f4_genome_result'] = cached

        result_label, changes = cached
        score = 1.0 if result_label == self.label else 0.0
        if self.label == 'genome_updated' and result_label == 'genome_updated':
            score = min(1.0, 0.70 + len(changes) * 0.03)

        return PVariantOutput(
            variant_id=self.variant_id,
            label=self.label,
            score=round(score, 3),
            confidence=0.80 if score > 0 else 0.0,
        )

    def train(self, p, n, s=None): pass
    def save(self, path): pass
    def load(self, path): pass

    @property
    def is_trained(self) -> bool:
        return True
