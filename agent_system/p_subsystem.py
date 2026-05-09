"""
p_subsystem.py — базовые типы и интерфейс P-подсистемы.

Каждый из 51 P-координат — это семья (PFamily) специализированных детекторов.
Каждый детектор (PVariant) — отдельная обученная модель, возвращает float 0..1.

Вместо:
    P47 → "hidden_reproach" (одна метка)

Теперь:
    P47 → PFamilyOutput {
        dominant: "hidden_reproach",
        scores: {reproach: 0.82, plea: 0.31, threat: 0.11, affection: 0.03}
    }

Это значит:
  - один текст может нести несколько подтекстов одновременно
  - каждый вариант обучается независимо на своих примерах
  - affect_bridge видит полный спектр, а не одну метку
"""
from __future__ import annotations

import os
import pickle
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ─── Выходные типы ────────────────────────────────────────────────────────────

@dataclass
class PVariantOutput:
    """Результат одного детектора внутри семьи."""
    variant_id: str      # "P47.reproach"
    label: str           # "hidden_reproach"
    score: float         # 0..1, выраженность признака
    confidence: float    # уверенность модели в оценке


@dataclass
class PFamilyOutput:
    """Результат всей семьи P-координаты."""
    p_id: str                      # "F47"
    dominant: str                  # метка с наибольшим score
    dominant_score: float
    scores: dict[str, float]       # {variant_label: score}
    confidences: dict[str, float]  # {variant_label: confidence}
    raw: list[PVariantOutput] = field(default_factory=list)

    def to_legacy_label(self) -> str:
        """Обратная совместимость: возвращает одну метку как раньше."""
        return self.dominant if self.dominant_score > 0.4 else 'absent'

    def above(self, threshold: float = 0.5) -> list[str]:
        """Все варианты выше порога."""
        return [label for label, score in self.scores.items() if score >= threshold]

    def to_dict(self) -> dict[str, Any]:
        return {
            'p_id': self.p_id,
            'dominant': self.dominant,
            'dominant_score': round(self.dominant_score, 3),
            'scores': {k: round(v, 3) for k, v in self.scores.items()},
        }


# ─── Определение варианта ─────────────────────────────────────────────────────

@dataclass
class PVariantDef:
    """
    Описание одного варианта внутри P-семьи.
    Содержит только метаданные — модель хранится отдельно в PVariant.
    """
    label: str           # машинный ключ: "hidden_reproach"
    ru: str              # русский: "скрытый упрёк"
    description: str     # что именно детектирует
    examples: list[str] = field(default_factory=list)   # затравочные примеры
    counter_examples: list[str] = field(default_factory=list)  # явно не это


# ─── Определение семьи ────────────────────────────────────────────────────────

@dataclass
class PFamilyDef:
    """
    Полное определение P-семьи: что она измеряет, какие варианты ищет.
    """
    p_id: str              # "F47"
    group: str             # A..G
    title_ru: str          # "Скрытый подтекст"
    description: str       # что измеряет вся семья
    default_absent: str = 'absent'   # метка когда признак отсутствует
    variants: list[PVariantDef] = field(default_factory=list)

    @property
    def all_labels(self) -> list[str]:
        return [self.default_absent] + [v.label for v in self.variants]


# ─── Модель варианта (обучаемая часть) ────────────────────────────────────────

class PVariant(ABC):
    """Один детектор: текст → (score, confidence)."""

    def __init__(self, variant_id: str, label: str):
        self.variant_id = variant_id
        self.label = label
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    @abstractmethod
    def forward(self, text: str, context: dict[str, Any]) -> PVariantOutput:
        ...

    @abstractmethod
    def train(self, positives: list[str], negatives: list[str],
              scores: list[float] | None = None) -> None:
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        ...


class RuleVariant(PVariant):
    """
    Fallback: правило из keywords.
    Используется пока нет обученной модели.
    """
    def __init__(self, variant_id: str, label: str, keywords: list[str]):
        super().__init__(variant_id, label)
        self.keywords = [kw.lower() for kw in keywords]
        self._trained = True  # правила всегда "готовы"

    def forward(self, text: str, context: dict[str, Any]) -> PVariantOutput:
        import re
        low = text.lower()
        # Токены для точного совпадения одиночных коротких слов
        tokens = set(re.findall(r'\b\w+\b', low))
        hits = 0
        for kw in self.keywords:
            kw_low = kw.lower()
            if ' ' in kw_low:
                # многословная фраза — подстрочный поиск
                if kw_low in low:
                    hits += 1
            else:
                # одно слово — только точное совпадение токена
                if kw_low in tokens:
                    hits += 1
        # 1 совпадение → 0.65, 2+ → 1.0
        score = min(hits * 0.65, 1.0) if hits else 0.0
        return PVariantOutput(
            variant_id=self.variant_id,
            label=self.label,
            score=round(score, 3),
            confidence=0.60,
        )

    def train(self, positives, negatives, scores=None): pass
    def save(self, path): pass
    def load(self, path): pass


# ─── Метрический вариант (P39: цель, не паттерн) ─────────────────────────────

_RU_FILLER = frozenset([
    'ну', 'вот', 'это', 'как', 'бы', 'общем', 'короче', 'да', 'нет', 'ага',
    'угу', 'мм', 'так', 'значит', 'вообще', 'типа', 'просто', 'ладно', 'окей',
    'ок', 'ясно', 'понятно', 'хорошо', 'хм', 'слушай', 'кстати', 'собственно',
])
_EN_FILLER = frozenset([
    'ok', 'okay', 'sure', 'yeah', 'yes', 'no', 'right', 'got', 'it', 'i', 'see',
    'mm', 'hmm', 'well', 'so', 'um', 'uh', 'basically', 'actually',
])
_CONTACT_ACTIVE = [
    'расскажи', 'продолжай', 'и что', 'а что дальше', 'что ещё', 'интересно',
    'tell me more', 'go on', 'and then', 'what happened', 'continue',
    'կshel el', 'shchave el',
]
_CONTACT_MINIMAL = [
    'угу', 'ага', 'понятно', 'ясно', 'хорошо', 'ладно', 'ок', 'окей',
    'ok', 'okay', 'i see', 'right', 'sure', 'yeah',
]


def _water_ratio(text: str) -> float:
    """Доля слов-заполнителей в тексте [0..1]."""
    import re
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 1.0
    fillers = sum(1 for w in words if w in _RU_FILLER or w in _EN_FILLER)
    return fillers / len(words)


def _stem(word: str) -> str:
    """Грубый стем: первые 5 символов для русского, 4 для английского."""
    n = 5 if any('Ѐ' <= c <= 'ӿ' for c in word) else 4
    return word[:n]


def _repetition_score(text: str, history: list[str]) -> float:
    """
    Насколько текст повторяет предыдущие ходы [0..1].
    Использует стем-приближение для русских флексий.
    """
    if not history:
        return 0.0
    import re
    cur_words = set(re.findall(r'\b\w{4,}\b', text.lower()))
    if not cur_words:
        return 0.0
    cur_stems = {_stem(w) for w in cur_words}

    max_overlap = 0.0
    for prev in history[-3:]:
        prev_words = set(re.findall(r'\b\w{4,}\b', prev.lower()))
        if not prev_words:
            continue
        prev_stems = {_stem(w) for w in prev_words}
        overlap = len(cur_stems & prev_stems) / max(len(cur_stems), 1)
        max_overlap = max(max_overlap, overlap)
    return min(max_overlap, 1.0)


class MetricVariant(PVariant):
    """
    P39 — Удержание контакта: детектирует ЦЕЛЬ реплики, не паттерн.

    active_maintenance: усилие остаться — вопрос, follow-up, новый угол
    minimal_contact:    едва держит — "ага", "понятно", вода без содержания
    contact_maintenance: повтор предыдущего своими словами (высокий overlap)
    """

    def __init__(self, variant_id: str, label: str):
        super().__init__(variant_id, label)
        self._trained = True

    def forward(self, text: str, context: dict[str, Any]) -> PVariantOutput:
        history: list[str] = list(context.get('history') or [])
        t = text.strip()
        low = t.lower()
        water = _water_ratio(t)
        repeat = _repetition_score(t, history)
        short = len(t) < 20
        has_q = '?' in t

        if self.label == 'minimal_contact':
            # Явные маркеры минимального контакта — word-boundary
            import re as _re
            _toks = set(_re.findall(r'\b\w+\b', low))
            _multi = [m for m in _CONTACT_MINIMAL if ' ' in m]
            _single = [m for m in _CONTACT_MINIMAL if ' ' not in m]
            _match_min = any(m in _toks for m in _single) or any(m in low for m in _multi)
            if _match_min:
                score = 0.90
            # Короткий + высокая вода (не команда — команды обычно включают глагол-действие)
            elif short and water >= 0.55 and not any(
                imp in low for imp in ['купи', 'принеси', 'сделай', 'возьми', 'позвони', 'напиши', 'скажи', 'иди']
            ):
                score = 0.70
            else:
                score = 0.0

        elif self.label == 'active_maintenance':
            # Вопрос или явный follow-up + не повторяет
            if any(kw in low for kw in _CONTACT_ACTIVE):
                score = 0.88
            elif has_q and water < 0.5 and repeat < 0.4:
                score = 0.70
            elif has_q and not short:
                score = 0.55
            else:
                score = 0.0

        elif self.label == 'contact_maintenance':
            # Повторяет предыдущее другими словами (стем-совпадение)
            if repeat >= 0.45 and not short:
                score = 0.78
            elif repeat >= 0.28:
                score = 0.68  # выше порога детекции 0.65
            else:
                score = 0.0

        else:
            score = 0.0

        return PVariantOutput(self.variant_id, self.label, round(score, 3), 0.65)

    @property
    def is_trained(self) -> bool:
        return True

    def train(self, positives, negatives, scores=None): pass
    def save(self, path): pass
    def load(self, path): pass


# ─── P51 Режим 1: осмысленность одиночной реплики ───────────────────────────

_RU_NOISE_RE = re.compile(
    r'^[а-яёa-z\s\?\!\.…,;:]+$',
    re.IGNORECASE | re.UNICODE,
)
_MEANINGFUL_MIN_CHARS = 8
_MEANINGFUL_MIN_WORDS = 2


class CoherenceVariant(PVariant):
    """
    P51 sentence_coherent / sentence_noise — одиночная реплика.

    sentence_coherent: реплика осмысленна (реальное содержание, не белиберда)
    sentence_noise:    реплика пустая, повторяющаяся или бессмысленная

    Не требует истории. Всегда работает с первой реплики.
    """

    def __init__(self, variant_id: str, label: str):
        super().__init__(variant_id, label)
        self._trained = True

    def forward(self, text: str, context: dict[str, Any]) -> PVariantOutput:
        t = text.strip()
        low = t.lower()
        import re as _re

        words = _re.findall(r'\b\w+\b', low)
        n_words = len(words)
        n_chars = len(t)

        # Уникальность слов (повторение = низкое содержание)
        unique_ratio = len(set(words)) / max(n_words, 1)

        # Соотношение значимых слов к заполнителям
        filler_ratio = _water_ratio(t)

        # Доля буквенных символов (набор символов?)
        alpha_chars = sum(1 for c in t if c.isalpha())
        alpha_ratio = alpha_chars / max(n_chars, 1)

        if self.label == 'sentence_coherent':
            # Осмысленная: достаточно слов, хорошая уникальность, много букв
            if n_chars < _MEANINGFUL_MIN_CHARS or n_words < _MEANINGFUL_MIN_WORDS:
                score = 0.0
            elif alpha_ratio < 0.4:
                score = 0.0  # мало букв — скорее шум
            elif filler_ratio > 0.80:
                score = 0.30  # почти всё вода
            elif unique_ratio > 0.6 and filler_ratio < 0.5 and n_words >= 3:
                score = 0.85
            elif n_words >= 2 and alpha_ratio >= 0.6:
                score = 0.70
            else:
                score = 0.55

        else:  # sentence_noise
            # Шум: очень коротко, повторение, набор символов
            if n_chars < 3:
                score = 0.95
            elif alpha_ratio < 0.3:
                score = 0.88  # набор символов
            elif n_words >= 1 and unique_ratio < 0.3 and n_words > 3:
                score = 0.80  # слова повторяются
            elif n_words <= 1 and n_chars < _MEANINGFUL_MIN_CHARS:
                score = 0.72  # одно слово, слишком короткое
            elif filler_ratio >= 0.95 and n_words <= 2:
                score = 0.70  # только вода
            else:
                score = 0.0

        return PVariantOutput(self.variant_id, self.label, round(score, 3), 0.65)

    @property
    def is_trained(self) -> bool:
        return True

    def train(self, pos, neg, scores=None): pass
    def save(self, path): pass
    def load(self, path): pass


# ─── P51 Режим 2: логика диалога из P-матрицы (≥2 реплик) ───────────────────
#
# "казнить нельзя помиловать" — запятая определяется:
#   - ЧТО было сказано (P1-P49 скоры)
#   - КОГДА отправлено (timestamps — для интерпретации, не для смены темы)
#   - КАКАЯ КОМБИНАЦИЯ паттернов образовалась (cluster_size)
#
# Три зоны:
#   logic_coherent (>0.65): логика прослеживается
#   logic_partial  (0.30-0.65): паттерны P1-P49 что-то показывают, нечётко
#   logic_chaos    (<0.30): Шляпник, чай, Алиса

_DISRUPT_P  = frozenset({'F13', 'F15', 'F35', 'F37'})   # разрывы логики
_SUBTEXT_P  = frozenset({'F47', 'F41', 'F43', 'F29'})   # скрытая связность


class DialogBoundaryVariant(PVariant):
    """
    P51 dialog logic — оценка логичности цепочки реплик из P-матрицы.

    Источник:
        context["history"]     — тексты (≥2)
        context["timestamps"]  — когда отправлено (для интерпретации запятой, не смены темы)
        context["p_history"]   — ((сообщение)(когда)(49 метрик))
        context["cluster_id"]  / context["cluster_size"] — из DialogPatternClusterer

    При len(history) < 2 → score 0.0.
    """

    def __init__(self, variant_id: str, label: str):
        super().__init__(variant_id, label)
        self._trained = True

    def forward(self, text: str, context: dict[str, Any]) -> PVariantOutput:
        history    = list(context.get("history")     or [])
        p_history  = list(context.get("p_history")   or [])
        cluster_size = int(context.get("cluster_size") or -1)
        cluster_id   = int(context.get("cluster_id")   or -1)

        if len(history) < 2:
            return PVariantOutput(self.variant_id, self.label, 0.0, 0.0)

        logic = self._logic_score(history, p_history, cluster_size, cluster_id)
        score = self._map(logic)
        return PVariantOutput(self.variant_id, self.label, round(score, 3), 0.70)

    def _logic_score(self, history, p_history, cluster_size, cluster_id):
        """
        Агрегированный балл логики диалога [0..1].

        1. Кластерный сигнал: большой кластер = известный паттерн = логика
        2. P-стабильность: плавные переходы = логика
        3. Разрывы (P13/P35/P37 спайки): снижают
        4. Скрытая связность (P47/P41): есть подтекст — есть своя логика
        5. data_confidence: мало реплик → ближе к нейтральному 0.5
        """
        n = len(history)

        # 1. Кластерный сигнал
        if cluster_id == -1 or cluster_size < 0:
            cluster_signal = 0.20   # нет кластера — слабый сигнал
        elif cluster_size >= 30:
            cluster_signal = 0.92   # крупный кластер — сильная логика
        elif cluster_size >= 10:
            cluster_signal = 0.50
        else:
            cluster_signal = 0.12   # малый кластер — редкий паттерн

        # 2. P-стабильность
        p_stability = 1.0
        if len(p_history) >= 2:
            drifts = []
            for i in range(1, len(p_history)):
                prev, cur = p_history[i-1], p_history[i]
                common = set(prev) & set(cur)
                if common:
                    d = sum(abs(float(prev.get(k,0)) - float(cur.get(k,0))) for k in common)
                    drifts.append(d / len(common))
            if drifts:
                avg_drift = sum(drifts) / len(drifts)
                p_stability = max(0.0, 1.0 - avg_drift * 2.0)

        # 3. Разрывные события (порог 0.6, полная амплитуда)
        disruption = 0.0
        for ph in p_history:
            for pid in _DISRUPT_P:
                val = float(ph.get(pid, 0))
                if val > 0.6:
                    disruption = max(disruption, val)
        logic_penalty = min(disruption * 0.65, 0.55)  # сильный штраф, но не >0.55

        # 4. Скрытая связность
        subtext_bonus = 0.0
        for ph in p_history:
            for pid in _SUBTEXT_P:
                if float(ph.get(pid, 0)) > 0.6:
                    subtext_bonus = min(subtext_bonus + 0.05, 0.15)

        # 5. Data confidence
        data_confidence = min((n - 1) / 3.0, 1.0)  # полная уверенность при 4+ репликах

        raw = (cluster_signal * 0.45 + p_stability * 0.35 + subtext_bonus) - logic_penalty
        # При явном хаосе (disruption ≥ 0.8) — обходим нейтральный притяжитель
        if disruption >= 0.8:
            logic = float(max(0.0, raw))
        else:
            logic = raw * data_confidence + 0.5 * (1.0 - data_confidence)
        return float(max(0.0, min(1.0, logic)))

    def _map(self, logic: float) -> float:
        """
        Маппинг непрерывного балла логики → активность варианта.
        Каждый вариант активен в своей зоне; вне зоны = 0.
        Порог детекции PFamily = 0.65, поэтому minimum активного скора = 0.65.
        """
        if self.label == "logic_coherent":
            if logic > 0.65:
                # 0.65→0.65, 1.0→0.90
                return 0.65 + (logic - 0.65) * 0.71
            return 0.0
        elif self.label == "logic_partial":
            if 0.28 < logic <= 0.65:
                return 0.65  # просто сигнал "серая зона"
            return 0.0
        elif self.label == "logic_chaos":
            if logic <= 0.28:
                # 0.28→0.65, 0.0→0.90
                return 0.65 + (0.28 - logic) * (0.25 / 0.28)
            return 0.0
        return 0.0

    @property
    def is_trained(self) -> bool:
        return True

    def train(self, pos, neg, scores=None): pass
    def save(self, path): pass
    def load(self, path): pass



# ─── Кластерный вариант (P50/P51: логика диалога) ────────────────────────────

class ClusterVariant(PVariant):
    """
    P50 / P51 — читает размер кластера из context['cluster_size'].

    P50 (смена темы):   cluster_size в диапазоне (10, 30) → нелогичная но редкая комбинация
    P51 (граница):      cluster_size >= 30 → известный паттерн, граница чёткая
    noise (-1 или <10): аномалия → не P50 и не P51

    Ключи context:
        'cluster_size'  : int — размер кластера текущего окна
        'cluster_id'    : int — id кластера (-1 = шум)
        'noise_ratio'   : float — доля шумовых окон в матрице
    """
    P50_RANGE = (10, 30)
    P51_MIN   = 30

    def __init__(self, variant_id: str, label: str, mode: str):
        """mode: 'p50' | 'p51'"""
        super().__init__(variant_id, label)
        self._mode = mode
        self._trained = True

    def forward(self, text: str, context: dict[str, Any]) -> PVariantOutput:
        cluster_size = int(context.get('cluster_size') or -1)
        cluster_id   = int(context.get('cluster_id')   or -1)
        noise_ratio  = float(context.get('noise_ratio') or 0.0)

        if self._mode == 'p50':
            score = self._p50_score(cluster_size, cluster_id, noise_ratio, text)
        else:
            score = self._p51_score(cluster_size, cluster_id, noise_ratio, text)

        return PVariantOutput(self.variant_id, self.label, round(score, 3), 0.70)

    def _p50_score(self, size: int, cid: int, noise: float, text: str) -> float:
        low = text.lower()
        # Явные маркеры смены темы — всегда P50 независимо от кластера
        # (включая фразы которые P51 исключает)
        explicit = {
            'soft_shift':      ['кстати', 'это напомнило', 'by the way', 'speaking of',
                                 'к слову', 'сменим тему', 'поменяем тему'],
            'hard_shift':      ['ладно совсем другое', 'хватит об этом',
                                 "let's change the subject", 'давай о другом', 'другая тема'],
            'return_to_topic': ['так вот о чём', 'вернёмся к', 'back to',
                                 'кстати о другом'],
            'hijack':          ['это напоминает мою', 'кстати я', 'actually i'],
            'spiral_return':   ['хотя знаешь всё это связано', 'although this relates'],
        }
        if self.label in explicit:
            if any(kw in low for kw in explicit[self.label]):
                return 0.88

        # Кластерная логика
        lo, hi = self.P50_RANGE
        if cid == -1 or size < 0:
            return 0.0
        if lo < size <= hi:
            # В диапазоне P50 — масштабируем от 0.65 до 0.85
            return 0.65 + 0.20 * (1 - (size - lo) / (hi - lo))
        return 0.0

    # Явные фразы смены темы — P50 даже если кластер большой
    _P50_EXPLICIT_PHRASES = [
        'сменим тему', 'поменяем тему', 'давай о другом', 'другая тема',
        'кстати о другом', 'вернёмся к', 'кстати', 'к слову',
        "let's change the subject", 'changing topics', 'on another note',
        'actually', 'speaking of which', 'by the way',
    ]

    def _p51_score(self, size: int, cid: int, noise: float, text: str) -> float:
        low = text.lower()

        # Явные маркеры смены темы → P50 забирает приоритет, P51=0
        if any(kw in low for kw in self._P50_EXPLICIT_PHRASES):
            return 0.0

        # Явные маркеры границы контекста
        explicit = {
            'full_window': ['ты всегда так', 'мы уже сто раз', 'you always', "we've talked about"],
            'since_topic_shift': ['нет я о том что мы только что', 'i mean what we just'],
            'last_turn_only': ['погоди ты сказал', 'wait you said', 'не то ты сказал'],
            'anchor_only': ['помнишь ты тогда сказал', 'remember when you said'],
            'since_emotion_peak': ['после того как ты', 'с тех пор как', 'since you said'],
        }
        if self.label in explicit:
            if any(kw in low for kw in explicit[self.label]):
                return 0.88

        # Кластерная логика: крупный кластер + не явная смена темы
        if cid == -1 or size < 0:
            return 0.0
        if size >= self.P51_MIN:
            conf = min(0.65 + (size - self.P51_MIN) / 200, 0.95)
            return conf
        return 0.0

    @property
    def is_trained(self) -> bool:
        return True

    def train(self, positives, negatives, scores=None): pass
    def save(self, path): pass
    def load(self, path): pass


class SklearnVariant(PVariant):
    """
    sklearn LogisticRegression на TF-IDF.
    Быстро обучается, хорошо работает с малыми данными (50+ примеров).
    """

    def __init__(self, variant_id: str, label: str):
        super().__init__(variant_id, label)
        self._pipeline: Any = None
        self._fallback: RuleVariant | None = None

    def set_fallback(self, rule_variant: RuleVariant) -> None:
        self._fallback = rule_variant

    def forward(self, text: str, context: dict[str, Any]) -> PVariantOutput:
        if not self._trained or self._pipeline is None:
            if self._fallback:
                return self._fallback.forward(text, context)
            return PVariantOutput(self.variant_id, self.label, 0.0, 0.0)

        try:
            proba = self._pipeline.predict_proba([text])[0]
            # классы: [absent=0, present=1]
            ml_score = float(proba[1]) if len(proba) > 1 else float(proba[0])
            confidence = float(max(proba))

            # keyword-фоллбэк всегда добавляет сверху: явное совпадение → boost
            if self._fallback:
                rule_out = self._fallback.forward(text, context)
                score = max(ml_score, rule_out.score)
            else:
                score = ml_score

            return PVariantOutput(self.variant_id, self.label,
                                  round(score, 3), round(confidence, 3))
        except Exception:
            if self._fallback:
                return self._fallback.forward(text, context)
            return PVariantOutput(self.variant_id, self.label, 0.0, 0.0)

    def train(self, positives: list[str], negatives: list[str],
              scores: list[float] | None = None) -> None:
        if len(positives) < 3:
            return
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            X = positives + negatives
            y = [1] * len(positives) + [0] * len(negatives)

            self._pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=8000,
                    analyzer='char_wb',  # работает лучше для русского
                    min_df=1,
                )),
                ('clf', LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    class_weight='balanced',
                )),
            ])
            self._pipeline.fit(X, y)
            self._trained = True
        except ImportError:
            pass  # sklearn не установлен — остаёмся на правилах

    def save(self, path: Path) -> None:
        if self._trained and self._pipeline is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self._pipeline, f)

    def load(self, path: Path) -> None:
        if path.exists():
            with open(path, 'rb') as f:
                self._pipeline = pickle.load(f)
            self._trained = True


# ─── P-семья (объединяет варианты, запускает их) ──────────────────────────────

class PFamily:
    """
    Одна P-координата как семья детекторов.
    Запускает все варианты и возвращает PFamilyOutput.
    """

    def __init__(self, definition: PFamilyDef, models_dir: Path):
        self.definition = definition
        self.p_id = definition.p_id
        self.models_dir = models_dir
        self._variants: list[PVariant] = []

    def add_variant(self, variant: PVariant) -> None:
        self._variants.append(variant)

    def forward(self, text: str, context: dict[str, Any] | None = None) -> PFamilyOutput:
        context = context if context is not None else {}
        if not self._variants:
            return PFamilyOutput(
                p_id=self.p_id,
                dominant=self.definition.default_absent,
                dominant_score=0.0,
                scores={self.definition.default_absent: 1.0},
                confidences={self.definition.default_absent: 1.0},
            )

        outputs = [v.forward(text, context) for v in self._variants]
        scores = {o.label: o.score for o in outputs}
        confidences = {o.label: o.confidence for o in outputs}

        max_variant_score = max(scores.values())

        # Для обученных sklearn-моделей требуем порог 0.62 чтобы побить absent.
        # Keyword-правила выдают фиксированные 0.65/1.0 — они точнее, порог для них не нужен.
        # Если все варианты — keyword-fallback (confidence=0.60), снижаем порог до 0.60.
        has_trained = any(
            o.confidence > 0.60 and o.score > 0 for o in outputs
        )
        detection_threshold = 0.65 if has_trained else 0.60

        if max_variant_score < detection_threshold:
            absent_score = 1.0
        else:
            absent_score = max(0.0, 1.0 - max_variant_score)

        scores[self.definition.default_absent] = round(absent_score, 3)
        confidences[self.definition.default_absent] = 0.7

        dominant = max(scores, key=lambda k: scores[k])
        return PFamilyOutput(
            p_id=self.p_id,
            dominant=dominant,
            dominant_score=scores[dominant],
            scores=scores,
            confidences=confidences,
            raw=outputs,
        )

    def train_variant(self, label: str, positives: list[str],
                      negatives: list[str], scores: list[float] | None = None) -> bool:
        for v in self._variants:
            if v.label == label and isinstance(v, SklearnVariant):
                v.train(positives, negatives, scores)
                if v.is_trained:
                    model_path = self.models_dir / self.p_id / f"{label}.pkl"
                    v.save(model_path)
                    return True
        return False

    def load_models(self) -> int:
        loaded = 0
        for v in self._variants:
            if isinstance(v, SklearnVariant):
                model_path = self.models_dir / self.p_id / f"{v.label}.pkl"
                if model_path.exists():
                    v.load(model_path)
                    loaded += 1
        return loaded

    @property
    def trained_count(self) -> int:
        return sum(1 for v in self._variants if v.is_trained)

    @property
    def variant_labels(self) -> list[str]:
        return [v.label for v in self._variants]
