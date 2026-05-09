"""
dialog_context_matrix.py — контекстная матрица диалога.

Каждая реплика → строка вектора (P-скоры + временны́е признаки + спикер).
Последовательность строк = матрица диалога.
Матрица служит входом для DialogPatternClusterer.

Структура строки (UtteranceRow):
    [P1_score, P2_score, ..., P48_score,  ← 48 P-скоров (P1-P48, состояния диалога)
     rel_time,                             ← нормированное время [0..1] в сессии
     speaker,                              ← 0=пользователь, 1=персона
     turn_idx_norm]                        ← нормированный номер хода [0..1]

P49/P50 работают поверх матрицы (не входят в неё).
P51 — бинарный гейт (genome_validator.p51_gate) — полностью вне матрицы.

Итого: FEATURE_DIM = 51 (при 48 P-семьях + 3 метапризнака).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

P_COUNT = 48                  # P1-P48 — состояния диалога; P49/P50/P51 — вне матрицы
_BASE_FEATURES = 3            # rel_time + speaker + turn_idx_norm
FEATURE_DIM = P_COUNT + _BASE_FEATURES           # 51 — одна строка
DELTA_DIM = P_COUNT                               # дельта P-скоров между ходами
WINDOW_ROW_DIM = FEATURE_DIM + DELTA_DIM         # 99 — строка окна (P + meta + delta)
_P_IDS = [f'F{i}' for i in range(1, 49)]


# ─── UtteranceRow ─────────────────────────────────────────────────────────────

@dataclass
class UtteranceRow:
    text: str
    speaker: str                        # 'user' | 'persona'
    timestamp: float = field(default_factory=time.time)
    p_scores: dict[str, float] = field(default_factory=dict)
    p_dominant: dict[str, str] = field(default_factory=dict)

    def to_feature_vec(
        self,
        rel_time: float = 0.0,          # [0..1] внутри сессии
        turn_idx_norm: float = 0.0,     # [0..1] порядковый номер
    ) -> np.ndarray:
        vec = np.zeros(FEATURE_DIM, dtype=np.float32)
        for i, pid in enumerate(_P_IDS):
            vec[i] = float(self.p_scores.get(pid, 0.0))
        vec[P_COUNT]     = rel_time
        vec[P_COUNT + 1] = 0.0 if self.speaker == 'user' else 1.0
        vec[P_COUNT + 2] = turn_idx_norm
        return vec


# ─── DialogContextMatrix ──────────────────────────────────────────────────────

class DialogContextMatrix:
    """
    Матрица диалога: список UtteranceRow → numpy matrix (N × FEATURE_DIM).

    Использование:
        mat = DialogContextMatrix(session_id='sess_001')
        mat.add(UtteranceRow(text='...', speaker='user', p_scores={...}))
        mat.add(UtteranceRow(text='...', speaker='persona', p_scores={...}))
        arr = mat.to_array()          # (N, 54) float32
        windows = mat.sliding_windows(w=4)  # список (4, 54) массивов
    """

    def __init__(self, session_id: str = '', metadata: dict[str, Any] | None = None):
        self.session_id = session_id
        self.metadata: dict[str, Any] = metadata or {}
        self._rows: list[UtteranceRow] = []

    def add(self, row: UtteranceRow) -> None:
        self._rows.append(row)

    def __len__(self) -> int:
        return len(self._rows)

    # ── Построение массива ────────────────────────────────────────────────────

    def to_array(self) -> np.ndarray:
        """Возвращает (N, FEATURE_DIM) float32."""
        if not self._rows:
            return np.zeros((0, FEATURE_DIM), dtype=np.float32)

        t_min = self._rows[0].timestamp
        t_max = self._rows[-1].timestamp
        t_span = max(t_max - t_min, 1e-6)
        n = len(self._rows)

        rows = []
        for idx, row in enumerate(self._rows):
            rel_t = (row.timestamp - t_min) / t_span
            turn_norm = idx / max(n - 1, 1)
            rows.append(row.to_feature_vec(rel_t, turn_norm))

        return np.stack(rows, axis=0)

    # ── Скользящее окно ───────────────────────────────────────────────────────

    def sliding_windows(self, w: int = 4) -> list[np.ndarray]:
        """
        Разбивает диалог на окна по w реплик.
        Каждая строка окна = базовый вектор (54) + дельта P-скоров от пред. хода (51).
        Итого размер строки = WINDOW_ROW_DIM = 105.
        Окно flatten → (w * WINDOW_ROW_DIM,).
        """
        arr = self.to_array()          # (N, 54)
        n = len(arr)
        if n == 0:
            return []

        # дельты: delta[i] = arr[i] - arr[i-1]; delta[0] = zeros
        deltas = np.zeros((n, DELTA_DIM), dtype=np.float32)
        if n > 1:
            deltas[1:] = arr[1:, :P_COUNT] - arr[:-1, :P_COUNT]

        # объединяем: (N, WINDOW_ROW_DIM=105)
        augmented = np.concatenate([arr, deltas], axis=1)

        if n < w:
            padded = np.zeros((w, WINDOW_ROW_DIM), dtype=np.float32)
            padded[:n] = augmented
            return [padded.flatten()]

        windows = []
        for start in range(0, n - w + 1):
            windows.append(augmented[start:start + w].flatten())
        return windows

    # ── Сводный вектор диалога ────────────────────────────────────────────────

    def summary_vector(self) -> np.ndarray:
        """
        (mean ‖ std) → (2 * FEATURE_DIM,) — быстрое представление диалога.
        Используется для кластеризации на уровне диалога (не окна).
        """
        arr = self.to_array()
        if len(arr) == 0:
            return np.zeros(2 * FEATURE_DIM, dtype=np.float32)
        return np.concatenate([arr.mean(axis=0), arr.std(axis=0)])

    # ── P-последовательность (читаемая) ──────────────────────────────────────

    def p_sequence(self) -> list[dict[str, str]]:
        """Список {P47: 'hidden_reproach', ...} для каждой реплики."""
        return [row.p_dominant for row in self._rows]

    def speaker_sequence(self) -> list[str]:
        return [row.speaker for row in self._rows]

    # ── Построение из реестра ─────────────────────────────────────────────────

    @classmethod
    def from_turns(
        cls,
        turns: list[dict[str, Any]],
        registry: Any,             # PSubsystemRegistry
        session_id: str = '',
        p49_builder: Any | None = None,  # P49MatrixBuilder — если передан, питаем историю
    ) -> 'DialogContextMatrix':
        """
        turns: [{'text': '...', 'speaker': 'user'|'persona', 'ts': float}, ...]
        registry: PSubsystemRegistry
        p49_builder: P49MatrixBuilder — накапливает P1-P48 историю для P49
        """
        mat = cls(session_id=session_id)
        for t in turns:
            text = str(t.get('text', ''))
            speaker = str(t.get('speaker', 'user'))
            ts = float(t.get('ts', time.time()))

            context: dict = {}

            # Если есть P49MatrixBuilder — передаём в context и предыдущий агрегат
            if p49_builder is not None:
                context['p49_builder'] = p49_builder
                context['_p49_prev'] = p49_builder.prev_aggregate

            outputs = registry.compute(text, context=context, parallel=False)
            # P1-P48 скоры только (не P49/P50/P51)
            p_scores = {
                pid: out.dominant_score
                for pid, out in outputs.items()
                if pid not in ('F49', 'F50', 'F51')
            }
            p_dominant = registry.to_legacy_vector(
                {pid: out for pid, out in outputs.items()
                 if pid not in ('F49', 'F50', 'F51')}
            )

            row = UtteranceRow(
                text=text,
                speaker=speaker,
                timestamp=ts,
                p_scores=p_scores,
                p_dominant=p_dominant,
            )
            mat.add(row)

            # После добавления реплики — обновляем P49 историю
            if p49_builder is not None:
                f1_48_vec = row.to_feature_vec()[:P_COUNT]  # первые 48 dims
                p49_builder.add(f1_48_vec, text, speaker, ts)
                # Обновляем prev_aggregate для следующего хода
                new_agg = context.get('_f49_aggregate')
                p49_builder.swap_prev_aggregate(new_agg)

        return mat
