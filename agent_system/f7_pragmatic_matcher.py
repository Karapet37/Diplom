"""
f7_pragmatic_matcher.py — F7: детектор прагматического подтекста.

Загружает pragmatic_kb.jsonl и матчит паттерны против входного текста.
Возвращает доминантный вариант + список активированных импликатур.

F7 variants:
    absent           — нет подтекста
    implied_debt     — создаётся долг/обязательство
    social_pressure  — апелляция к нормам/сравнению
    false_rapport    — ложная близость
    implicit_threat  — угроза без прямого слова
    passive_aggression — пассивная агрессия
    face_saving      — сохранение лица / косвенная просьба
    cultural_code    — культурный код
    catastrophizing  — гиперболизация последствий
    moral_erosion    — постепенная легитимизация неправильного

F7 вычисляется в фазе 1 (параллельно с F1-F3, F8-F48).
F6 читает F7 из snapshot → корректирует директиву.
ContextBuilder читает F7 → добавляет [Субтекст: ...] в промпт.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .p_subsystem import PVariant, PVariantOutput

_KB_PATH = Path(__file__).parent / 'pragmatic_kb.jsonl'

VARIANTS = [
    'absent', 'implied_debt', 'social_pressure', 'false_rapport',
    'implicit_threat', 'passive_aggression', 'face_saving',
    'cultural_code', 'catastrophizing', 'moral_erosion',
]


# ── KB entry ───────────────────────────────────────────────────────────────────

@dataclass
class KBEntry:
    id: str
    patterns: list[str]
    lang: str
    implied: str
    subtext: str
    p7_variant: str
    p3_boost: str | None
    probability: float
    category: str
    _compiled: list[re.Pattern] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._compiled = []
        for pat in self.patterns:
            try:
                self._compiled.append(re.compile(pat, re.I | re.UNICODE))
            except re.error:
                self._compiled.append(re.compile(re.escape(pat), re.I | re.UNICODE))

    def match(self, text: str) -> float:
        """Returns probability if any pattern matches, else 0.0."""
        t = text.lower()
        for rx in self._compiled:
            if rx.search(t):
                return self.probability
        return 0.0


# ── KB loader ──────────────────────────────────────────────────────────────────

_KB: list[KBEntry] | None = None


def _load_kb() -> list[KBEntry]:
    global _KB
    if _KB is not None:
        return _KB
    if not _KB_PATH.exists():
        _KB = []
        return _KB
    entries = []
    for line in _KB_PATH.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            entries.append(KBEntry(
                id=d['id'],
                patterns=d['patterns'],
                lang=d.get('lang', 'any'),
                implied=d['implied'],
                subtext=d['subtext'],
                p7_variant=d['p7_variant'],
                p3_boost=d.get('p3_boost'),
                probability=float(d.get('probability', 0.7)),
                category=d.get('category', 'unknown'),
            ))
        except Exception:
            pass
    _KB = entries
    return _KB


def reset_kb() -> None:
    global _KB
    _KB = None


# ── Matcher ────────────────────────────────────────────────────────────────────

@dataclass
class ImplicatureMatch:
    entry_id: str
    variant: str
    implied: str
    subtext: str
    probability: float
    p3_boost: str | None
    category: str


@dataclass
class F7Output:
    dominant_variant: str          # winning F7 variant label
    dominant_score: float
    matches: list[ImplicatureMatch]
    combined_subtext: str          # ready-to-inject string for LLM prompt
    p3_boosts: dict[str, float]    # {manipulation_label: max_probability}


def match_pragmatics(text: str) -> F7Output:
    kb = _load_kb()
    hits: list[ImplicatureMatch] = []

    for entry in kb:
        score = entry.match(text)
        if score > 0.0:
            hits.append(ImplicatureMatch(
                entry_id=entry.id,
                variant=entry.p7_variant,
                implied=entry.implied,
                subtext=entry.subtext,
                probability=score,
                p3_boost=entry.p3_boost,
                category=entry.category,
            ))

    if not hits:
        return F7Output('absent', 1.0, [], '', {})

    # Aggregate variant scores
    variant_scores: dict[str, float] = {}
    for h in hits:
        variant_scores[h.variant] = max(variant_scores.get(h.variant, 0.0), h.probability)

    dominant_variant = max(variant_scores, key=lambda v: variant_scores[v])
    dominant_score = variant_scores[dominant_variant]

    # Build combined subtext (top 3 by probability)
    top = sorted(hits, key=lambda h: h.probability, reverse=True)[:3]
    subtext_parts = [h.subtext for h in top]
    combined_subtext = ' '.join(subtext_parts)

    # Aggregate P3 boost signals
    p3_boosts: dict[str, float] = {}
    for h in hits:
        if h.p3_boost:
            p3_boosts[h.p3_boost] = max(p3_boosts.get(h.p3_boost, 0.0), h.probability)

    return F7Output(
        dominant_variant=dominant_variant,
        dominant_score=round(dominant_score, 3),
        matches=hits,
        combined_subtext=combined_subtext,
        p3_boosts=p3_boosts,
    )


# ── PVariant wrapper ───────────────────────────────────────────────────────────

class F7ImplicatureVariant(PVariant):
    """
    F7 в PFamily framework.
    Все варианты читают один cached F7Output из context['_f7_output'].
    Дополнительно записывает p3_boosts в context['_f7_p3_boosts']
    и combined_subtext в context['_f7_subtext'] для ContextBuilder.
    """

    def __init__(self, variant_id: str, label: str):
        super().__init__(variant_id, label)
        self._trained = True

    def forward(self, text: str, context: Any) -> PVariantOutput:
        ctx = context if isinstance(context, dict) else {}

        cached: F7Output | None = ctx.get('_f7_output')
        if cached is None:
            cached = match_pragmatics(text)
            ctx['_f7_output'] = cached
            ctx['_f7_p3_boosts'] = cached.p3_boosts
            ctx['_f7_subtext'] = cached.combined_subtext

        if cached.dominant_variant == self.label:
            score = max(cached.dominant_score, 0.70)
        else:
            score = 0.0

        return PVariantOutput(
            variant_id=self.variant_id,
            label=self.label,
            score=round(score, 3),
            confidence=round(cached.dominant_score if score > 0 else 0.0, 3),
        )

    def train(self, p, n, s=None): pass
    def save(self, path): pass
    def load(self, path): pass

    @property
    def is_trained(self) -> bool:
        return True
