"""
p49_matrix_builder.py — P49: построитель матрицы релевантного контекста.

Принцип (как эмбеддинги):
  Для каждой новой реплики берётся P1-P48 вектор (48 флоатов).
  По истории сессии находятся top-K ходов с максимальным косинусным
  сходством + учётом временно́й актуальности.
  Результат — упорядоченная по релевантности матрица исторических состояний.

P49 НЕ входит в DialogContextMatrix (та хранит только P1-P48).
P49 работает ПОВЕРХ накопленной матрицы как retrieval-операция.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .p_subsystem import PVariant, PVariantOutput

P_DIM = 48  # только P1-P48

# ─── Данные истории ───────────────────────────────────────────────────────────

@dataclass
class HistoryEntry:
    vec: np.ndarray      # P1-P48 вектор (48 dims, float32)
    text: str
    speaker: str         # 'user' | 'persona'
    timestamp: float
    turn_idx: int


@dataclass
class RelevanceRow:
    """Одна строка в P49-матрице — историческое состояние с весами релевантности."""
    turn_idx: int
    similarity: float    # косинусное сходство с текущим вектором
    recency: float       # 0..1, 1 = самый недавний
    combined: float      # итоговый вес
    vec: np.ndarray      # P1-P48 вектор записи
    text: str
    speaker: str


# ─── P49MatrixBuilder ─────────────────────────────────────────────────────────

class P49MatrixBuilder:
    """
    Накапливает историю P1-P48 векторов в рамках сессии.
    По запросу строит матрицу релевантных контекстов для текущего хода.

    Параметры:
        max_history     — сколько ходов хранить (старые вытесняются)
        recency_weight  — доля временно́й актуальности в итоговом весе
                          (0 = только сходство, 1 = только новизна)
        top_k           — сколько строк возвращать в матрице
    """

    def __init__(
        self,
        max_history: int = 64,
        recency_weight: float = 0.30,
        top_k: int = 8,
    ):
        self._history: list[HistoryEntry] = []
        self._max_history = max_history
        self._recency_weight = recency_weight
        self._top_k = top_k
        self._prev_aggregate: np.ndarray | None = None

    def add(
        self,
        vec: np.ndarray,
        text: str,
        speaker: str,
        timestamp: float | None = None,
    ) -> None:
        """Добавить реплику в историю."""
        if vec.shape[0] < P_DIM:
            padded = np.zeros(P_DIM, dtype=np.float32)
            padded[:vec.shape[0]] = vec
            vec = padded
        else:
            vec = vec[:P_DIM].astype(np.float32)

        entry = HistoryEntry(
            vec=vec.copy(),
            text=text,
            speaker=speaker,
            timestamp=timestamp if timestamp is not None else time.time(),
            turn_idx=len(self._history),
        )
        self._history.append(entry)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def build_matrix(
        self,
        current_vec: np.ndarray,
        top_k: int | None = None,
    ) -> list[RelevanceRow]:
        """
        Строит матрицу релевантных исторических состояний.

        Score = (1 - recency_weight) * cosine_sim + recency_weight * recency
        Результат отсортирован по combined desc, ограничен top_k.
        """
        if not self._history:
            return []

        k = top_k if top_k is not None else self._top_k
        cur = current_vec[:P_DIM].astype(np.float32)
        cur_norm = float(np.linalg.norm(cur))

        rows: list[RelevanceRow] = []
        n = len(self._history)

        for i, entry in enumerate(self._history):
            h_norm = float(np.linalg.norm(entry.vec))
            if cur_norm < 1e-8 or h_norm < 1e-8:
                sim = 0.0
            else:
                sim = float(np.dot(cur, entry.vec) / (cur_norm * h_norm))
                sim = max(0.0, sim)

            recency = i / max(n - 1, 1)  # 0=oldest, 1=most recent
            w = self._recency_weight
            combined = (1.0 - w) * sim + w * recency

            rows.append(RelevanceRow(
                turn_idx=entry.turn_idx,
                similarity=round(sim, 4),
                recency=round(recency, 4),
                combined=round(combined, 4),
                vec=entry.vec,
                text=entry.text,
                speaker=entry.speaker,
            ))

        rows.sort(key=lambda r: r.combined, reverse=True)
        return rows[:k]

    def aggregate(self, matrix: list[RelevanceRow]) -> np.ndarray | None:
        """
        Свёртка матрицы в один P1-P48 вектор (взвешенное среднее по combined).
        Используется как вход для P50.
        """
        if not matrix:
            return None
        weights = np.array([r.combined for r in matrix], dtype=np.float32)
        w_sum = weights.sum()
        if w_sum < 1e-8:
            return None
        weights /= w_sum
        agg = np.zeros(P_DIM, dtype=np.float32)
        for i, row in enumerate(matrix):
            agg += weights[i] * row.vec[:P_DIM]
        return agg

    def best_similarity(self, matrix: list[RelevanceRow]) -> float:
        """Наибольшее косинусное сходство в матрице (0 если пустая)."""
        if not matrix:
            return 0.0
        return matrix[0].similarity

    def swap_prev_aggregate(self, new_agg: np.ndarray | None) -> np.ndarray | None:
        """Обновляет prev_aggregate и возвращает старое значение."""
        old = self._prev_aggregate
        self._prev_aggregate = new_agg.copy() if new_agg is not None else None
        return old

    @property
    def prev_aggregate(self) -> np.ndarray | None:
        return self._prev_aggregate

    def reset(self) -> None:
        self._history.clear()
        self._prev_aggregate = None

    def __len__(self) -> int:
        return len(self._history)


# ─── Singleton per session ────────────────────────────────────────────────────

_GLOBAL_BUILDER: P49MatrixBuilder | None = None


def get_global_builder() -> P49MatrixBuilder:
    global _GLOBAL_BUILDER
    if _GLOBAL_BUILDER is None:
        _GLOBAL_BUILDER = P49MatrixBuilder()
    return _GLOBAL_BUILDER


def reset_global_builder() -> None:
    global _GLOBAL_BUILDER
    _GLOBAL_BUILDER = None


# ─── P49Variant — интеграция в PFamily framework ──────────────────────────────

class P49Variant(PVariant):
    """
    P49 вариант для PFamily.

    Читает из context:
        'f1_48_vec'    : np.ndarray (48,) — текущий P1-P48 вектор
        'p49_builder'  : P49MatrixBuilder (необяз., fallback → global)

    Пишет в context (для P50):
        '_f49_matrix'     : list[RelevanceRow]
        '_f49_aggregate'  : np.ndarray | None

    Возвращает scalar score: best_similarity из матрицы.
    """

    # Пороги для маппинга score → label
    _HIGH = 0.60
    _MID = 0.30

    def __init__(self, variant_id: str, label: str):
        super().__init__(variant_id, label)
        self._trained = True

    def forward(self, text: str, context: dict[str, Any]) -> PVariantOutput:
        builder: P49MatrixBuilder = (
            context.get('p49_builder') or get_global_builder()
        )
        vec = context.get('f1_48_vec')
        if vec is None:
            # Нет вектора — нечего искать
            return PVariantOutput(self.variant_id, self.label, 0.0, 0.0)

        vec = np.asarray(vec, dtype=np.float32)
        matrix = builder.build_matrix(vec)
        best_sim = builder.best_similarity(matrix)
        agg = builder.aggregate(matrix)

        # Пишем в context чтобы P50 мог читать
        context['_f49_matrix'] = matrix
        context['_f49_aggregate'] = agg

        score = self._label_score(best_sim)
        return PVariantOutput(self.variant_id, self.label, round(score, 3), 0.75)

    def _label_score(self, best_sim: float) -> float:
        if self.label == 'context_found':
            return best_sim if best_sim >= self._HIGH else 0.0
        if self.label == 'partial_match':
            return best_sim if self._MID <= best_sim < self._HIGH else 0.0
        if self.label == 'no_context':
            return 1.0 - best_sim if best_sim < self._MID else 0.0
        if self.label == 'recurring':
            return best_sim if best_sim >= 0.85 else 0.0
        return 0.0

    def train(self, positives, negatives, scores=None): pass
    def save(self, path): pass
    def load(self, path): pass

    @property
    def is_trained(self) -> bool:
        return True
