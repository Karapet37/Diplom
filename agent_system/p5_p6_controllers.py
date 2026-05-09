"""
p5_p6_controllers.py — P5: фаза разговора, P6: директива ответа.

P5 — Conversation Phase (state machine по P32-P39):
    opening       — начало, нет истории
    rapport       — контакт, нейтральные/тёплые сигналы
    conflict      — P13/P35/P36 высокие
    resolution    — P34/P38 после конфликта
    closing       — P37 или P6:closing в истории

P6 — Response Directive (вычисляется из P2+P5+P47):
    respond_normal  — обычный ответ
    hold_ground     — стой на своём (P2=hostile + persona proud)
    de_escalate     — смягчи (P2=hostile + persona cares)
    redirect        — смени тему (P50=topic_shift)
    call_out        — назови манипуляцию прямо (P3=manipulative)
"""
from __future__ import annotations

from typing import Any

from .p_subsystem import PVariant, PVariantOutput

# ── P5: Conversation Phase ─────────────────────────────────────────────────────

_CONFLICT_P = {'F13', 'F15', 'F35', 'F36'}
_CONFLICT_LABELS = {'attack', 'status_attack', 'escalation', 'sharpening',
                    'overloaded', 'tense', 'toward_escalation'}
_RESOLUTION_P = {'F34', 'F38', 'F32'}
_RESOLUTION_LABELS = {'reconciliation', 'genuine_reconciliation', 'softening',
                      'genuine_softening', 'approach', 'emotional_approach'}
_CLOSING_LABELS = {'hard_rupture', 'soft_rupture', 'rupture', 'closing', 'toward_rupture'}


def _compute_phase(p_outputs: dict, history_phases: list[str]) -> str:
    if not p_outputs:
        return 'opening'

    # Check conflict signals
    conflict_score = 0.0
    for pid in _CONFLICT_P:
        out = p_outputs.get(pid)
        if out and out.dominant in _CONFLICT_LABELS:
            conflict_score += out.dominant_score

    # Check resolution/warm signals
    resolution_score = 0.0
    for pid in _RESOLUTION_P:
        out = p_outputs.get(pid)
        if out and out.dominant in _RESOLUTION_LABELS:
            resolution_score += out.dominant_score

    # Check closing
    closing = any(
        p_outputs.get(pid) and p_outputs[pid].dominant in _CLOSING_LABELS
        for pid in ('F37', 'F6')
    )

    if closing:
        return 'closing'

    prev_phase = history_phases[-1] if history_phases else 'opening'

    if conflict_score >= 0.55:
        return 'conflict'

    if prev_phase == 'conflict' and resolution_score >= 0.40:
        return 'resolution'

    if not history_phases or prev_phase == 'opening':
        if resolution_score >= 0.30 or conflict_score < 0.20:
            return 'rapport' if history_phases else 'opening'

    return prev_phase or 'rapport'


class P5PhaseVariant(PVariant):
    """P5: фаза разговора."""

    PHASES = ['opening', 'rapport', 'conflict', 'resolution', 'closing']

    def __init__(self, variant_id: str, label: str):
        super().__init__(variant_id, label)
        self._trained = True

    def forward(self, text: str, context: Any) -> PVariantOutput:
        ctx = context if isinstance(context, dict) else {}

        cached = ctx.get('_f5_phase')
        if cached is None:
            p_outputs = ctx.get('_f_outputs_snapshot', {})
            history   = ctx.get('_phase_history', [])
            cached = _compute_phase(p_outputs, history)
            ctx['_f5_phase'] = cached

        score = 1.0 if cached == self.label else 0.0
        conf  = 0.75 if score > 0 else 0.0
        return PVariantOutput(self.variant_id, self.label, round(score, 3), conf)

    def train(self, p, n, s=None): pass
    def save(self, path): pass
    def load(self, path): pass

    @property
    def is_trained(self) -> bool:
        return True


# ── P6: Response Directive ─────────────────────────────────────────────────────

def _compute_directive(
    p_outputs: dict,
    phase: str,
    persona_name: str,
    context: dict | None = None,
) -> str:
    """
    Директива для ответа:
    - call_out:     F3=manipulation | F29 высокая | F7=implied_debt/false_rapport/catastrophizing
    - hold_ground:  F2=hostile + persona proud/distant
    - de_escalate:  F2=hostile + persona caring
    - redirect:     F50=topic_shift вне конфликта
    - respond_normal: всё остальное
    """
    ctx = context or {}

    # ── F7 прагматический подтекст → приоритетные директивы ──────────────────
    f7 = p_outputs.get('F7')
    f7_variant = f7.dominant if f7 else 'absent'
    f7_score   = f7.dominant_score if f7 else 0.0

    # Скрытая угроза → call_out если есть реакция персонажа
    if f7_variant == 'implicit_threat' and f7_score > 0.75:
        return 'call_out'

    # Ложная близость → hold_ground (не поддаваться)
    if f7_variant == 'false_rapport' and f7_score > 0.70:
        return 'hold_ground'

    # Катастрофизация → de_escalate (не паниковать вместе)
    if f7_variant == 'catastrophizing' and f7_score > 0.75:
        return 'de_escalate'

    # ── F7 boosts F3 detection ────────────────────────────────────────────────
    f7_p3_boosts: dict[str, float] = ctx.get('_f7_p3_boosts', {})

    # F3 manipulation → call_out (порог снижен если F7 тоже сигналит)
    f3 = p_outputs.get('F3')
    if f3 and f3.dominant in ('guilt', 'flattery_hook', 'pressure', 'victimhood'):
        f3_threshold = 0.60 if f3.dominant in f7_p3_boosts else 0.75
        if f3.dominant_score > f3_threshold:
            return 'call_out'

    # F29 manipulation (более точный — порог ниже)
    f29 = p_outputs.get('F29')
    if f29 and f29.dominant != 'absent' and f29.dominant_score > 0.70:
        return 'call_out'

    # ── F2 тон → hold/de-escalate ─────────────────────────────────────────────
    f2_label = None
    f2_ctx = p_outputs.get('F2')
    if f2_ctx:
        f2_label = f2_ctx.dominant

    if f2_label == 'manipulative':
        return 'call_out'

    if f2_label in ('hostile_aggressive', 'hostile_heated', 'hostile_cold'):
        sd = 0.5
        dc = 0.5
        if persona_name:
            try:
                from .genome_validator import load_or_init_persona_genome
                g = load_or_init_persona_genome(persona_name)
                sd = getattr(getattr(g, 'social_distance_default', None), 'value', 0.5)
                dc = getattr(getattr(g, 'drive_closeness', None), 'value', 0.5)
            except Exception:
                pass

        if sd > 0.65:
            return 'hold_ground'
        if dc > 0.65:
            return 'de_escalate'
        return 'hold_ground'

    # ── F50 смена темы → redirect ─────────────────────────────────────────────
    f50 = p_outputs.get('F50')
    if f50 and f50.dominant == 'topic_shift' and phase != 'conflict':
        return 'redirect'

    return 'respond_normal'


class P6DirectiveVariant(PVariant):
    """P6: директива ответа для пайплайна."""

    DIRECTIVES = ['respond_normal', 'hold_ground', 'de_escalate', 'redirect', 'call_out']

    def __init__(self, variant_id: str, label: str):
        super().__init__(variant_id, label)
        self._trained = True

    def forward(self, text: str, context: Any) -> PVariantOutput:
        ctx = context if isinstance(context, dict) else {}

        cached = ctx.get('_f6_directive')
        if cached is None:
            p_outputs = ctx.get('_f_outputs_snapshot', {})
            phase     = ctx.get('_f5_phase', 'rapport')
            persona   = str(ctx.get('persona_name') or ctx.get('selected_persona') or '').strip()
            cached = _compute_directive(p_outputs, phase, persona, context=ctx)
            ctx['_f6_directive'] = cached

        score = 1.0 if cached == self.label else 0.0
        return PVariantOutput(
            self.variant_id, self.label,
            round(score, 3),
            0.80 if score > 0 else 0.0,
        )

    def train(self, p, n, s=None): pass
    def save(self, path): pass
    def load(self, path): pass

    @property
    def is_trained(self) -> bool:
        return True
