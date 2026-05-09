"""
dialog_pattern_clusterer.py — кластеризация паттернов диалога без учителя.

Принцип:
    Каждое скользящее окно (W реплик) → вектор признаков.
    DBSCAN или HDBSCAN кластеризует эти векторы.
    Малые кластеры (< min_cluster_size) = "нелогичные" паттерны.
    Крупные кластеры = регулярные сценарии (накапливают смысл).

Иерархия уровней:
    window_level    — кластеры по W-репличным окнам
    dialog_level    — кластеры по summary-векторам диалога
    session_level   — кластеры по нескольким диалогам (накопленный опыт)

P50/P51-флаги используются как граница при сегментации на темы.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .dialog_context_matrix import DialogContextMatrix, WINDOW_ROW_DIM

# P50 варианты, сигнализирующие о смене темы
_TOPIC_END_VARIANTS = {
    'hard_shift', 'return_to_topic', 'soft_shift',
}
# P51 варианты, сужающие контекстное окно
_CONTEXT_RESET_VARIANTS = {
    'hard_shift', 'last_turn_only',
}


# ─── Кластерный паттерн ───────────────────────────────────────────────────────

@dataclass
class DialogPattern:
    """Один обнаруженный кластер."""
    cluster_id: int
    size: int                          # количество примеров
    centroid: np.ndarray               # среднее вектора
    is_noise: bool = False             # кластер-шум = нелогично
    label: str = ''                    # человекочитаемое имя (заполняется вручную)
    examples: list[str] = field(default_factory=list)   # текстовые примеры

    def to_dict(self) -> dict[str, Any]:
        return {
            'cluster_id': self.cluster_id,
            'size': self.size,
            'is_noise': self.is_noise,
            'label': self.label,
            'examples': self.examples[:3],
        }


# ─── Основной кластеризатор ───────────────────────────────────────────────────

class DialogPatternClusterer:
    """
    Накопительный кластеризатор паттернов диалога.

    Алгоритм:
      1. Из каждого диалога извлекаем скользящие окна → векторы
      2. Добавляем в буфер
      3. При достижении min_samples_to_fit — запускаем DBSCAN
      4. При добавлении нового диалога — предсказываем его окнам кластеры
         или помечаем как noise (нелогично)

    Параметры:
      window_size        — сколько реплик в одном окне (4-6 хорошо)
      min_cluster_size   — меньше этого = шум / нелогично
      min_samples_to_fit — сколько окон накопить перед первым обучением
      eps                — радиус DBSCAN (None = автокалибровка)
    """

    def __init__(
        self,
        window_size: int = 4,
        min_cluster_size: int = 5,
        min_samples_to_fit: int = 50,
        eps: float | None = None,
        use_hdbscan: bool = False,
    ):
        self.window_size = window_size
        self.min_cluster_size = min_cluster_size
        self.min_samples_to_fit = min_samples_to_fit
        self.eps = eps
        self.use_hdbscan = use_hdbscan

        self._buffer: list[np.ndarray] = []        # накопленные векторы окон
        self._buffer_meta: list[dict] = []         # session_id + window_start
        self._model: Any = None                    # sklearn DBSCAN/HDBSCAN
        self._is_fitted: bool = False
        self._patterns: dict[int, DialogPattern] = {}

    # ── Добавление диалога ────────────────────────────────────────────────────

    def add_dialog(self, mat: DialogContextMatrix) -> None:
        """
        Добавляет диалог в буфер (и переобучает если накоплено достаточно).
        """
        windows = mat.sliding_windows(w=self.window_size)
        for i, win in enumerate(windows):
            self._buffer.append(win)
            self._buffer_meta.append({'session_id': mat.session_id, 'window_start': i})

        if len(self._buffer) >= self.min_samples_to_fit:
            self._fit()

    # ── Оценка логичности диалога ─────────────────────────────────────────────

    def logic_score(self, mat: DialogContextMatrix) -> dict[str, Any]:
        """
        Возвращает оценку логичности диалога.

        {
            'is_logical': bool,
            'confidence': float,            # [0..1]
            'noise_window_ratio': float,    # доля нелогичных окон
            'cluster_ids': list[int],       # кластер для каждого окна (-1=шум)
            'patterns': list[dict],         # совпавшие паттерны
        }
        """
        windows = mat.sliding_windows(w=self.window_size)
        if not windows or not self._is_fitted:
            return {
                'is_logical': True,
                'confidence': 0.0,
                'noise_window_ratio': 0.0,
                'cluster_ids': [],
                'patterns': [],
                'note': 'not_enough_data',
            }

        cluster_ids = self._predict(windows)
        noise_count = sum(1 for c in cluster_ids if c == -1)
        noise_ratio = noise_count / len(cluster_ids)

        # логично если меньше 30% окон — шум
        is_logical = noise_ratio < 0.3
        confidence = self._model_confidence if self._is_fitted else 0.0

        matched_patterns = []
        seen = set()
        for cid in cluster_ids:
            if cid != -1 and cid not in seen:
                seen.add(cid)
                if cid in self._patterns:
                    matched_patterns.append(self._patterns[cid].to_dict())

        return {
            'is_logical': is_logical,
            'confidence': round(confidence, 3),
            'noise_window_ratio': round(noise_ratio, 3),
            'cluster_ids': cluster_ids,
            'patterns': matched_patterns,
        }

    # ── Сводка паттернов ──────────────────────────────────────────────────────

    def pattern_summary(self, top_n: int = 10) -> list[dict[str, Any]]:
        """Топ паттернов по размеру кластера."""
        sorted_pats = sorted(
            self._patterns.values(),
            key=lambda p: p.size,
            reverse=True,
        )
        return [p.to_dict() for p in sorted_pats[:top_n] if not p.is_noise]

    def noise_examples(self, limit: int = 5) -> list[str]:
        """Примеры нелогичных окон (шумовой кластер -1)."""
        if -1 in self._patterns:
            return self._patterns[-1].examples[:limit]
        return []

    def status(self) -> dict[str, Any]:
        return {
            'fitted': self._is_fitted,
            'buffer_size': len(self._buffer),
            'n_patterns': len([p for p in self._patterns.values() if not p.is_noise]),
            'noise_size': self._patterns.get(-1, DialogPattern(-1, 0, np.array([]))).size,
            'window_size': self.window_size,
        }

    # ── Сегментация по теме (P50/P51) ─────────────────────────────────────────

    @staticmethod
    def split_by_topic(
        mat: DialogContextMatrix,
        p50_outputs: dict[str, Any] | None = None,
    ) -> list[DialogContextMatrix]:
        """
        Разбивает диалог на под-диалоги по P50 (смена темы).

        p50_outputs: {session_id: [{turn_idx: int, variant: str}]}
        Если p50_outputs не передан — возвращает исходный диалог одним куском.
        """
        if not p50_outputs:
            return [mat]

        breakpoints = set()
        for item in p50_outputs.get(mat.session_id, []):
            if item.get('variant') in _TOPIC_END_VARIANTS:
                breakpoints.add(item['turn_idx'])

        if not breakpoints:
            return [mat]

        segments: list[DialogContextMatrix] = []
        start = 0
        sorted_bp = sorted(breakpoints)
        for bp in sorted_bp:
            seg = DialogContextMatrix(
                session_id=f'{mat.session_id}_{start}',
                metadata={'parent': mat.session_id, 'start_turn': start},
            )
            seg._rows = mat._rows[start:bp + 1]
            segments.append(seg)
            start = bp + 1

        if start < len(mat):
            seg = DialogContextMatrix(
                session_id=f'{mat.session_id}_{start}',
                metadata={'parent': mat.session_id, 'start_turn': start},
            )
            seg._rows = mat._rows[start:]
            segments.append(seg)

        return segments

    # ── Сохранение / загрузка ─────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self._model,
                'buffer': self._buffer,
                'buffer_meta': self._buffer_meta,
                'patterns': self._patterns,
                'is_fitted': self._is_fitted,
                'config': {
                    'window_size': self.window_size,
                    'min_cluster_size': self.min_cluster_size,
                    'eps': self.eps,
                    'use_hdbscan': self.use_hdbscan,
                },
            }, f)

    @classmethod
    def load(cls, path: Path) -> 'DialogPatternClusterer':
        path = Path(path)
        with open(path, 'rb') as f:
            state = pickle.load(f)
        cfg = state['config']
        inst = cls(
            window_size=cfg['window_size'],
            min_cluster_size=cfg['min_cluster_size'],
            eps=cfg['eps'],
            use_hdbscan=cfg['use_hdbscan'],
        )
        inst._model = state['model']
        inst._buffer = state['buffer']
        inst._buffer_meta = state['buffer_meta']
        inst._patterns = state['patterns']
        inst._is_fitted = state['is_fitted']
        return inst

    # ── Внутреннее: обучение ──────────────────────────────────────────────────

    def _fit(self) -> None:
        X = np.stack(self._buffer, axis=0)

        # Нормализация — сохраняем параметры для предсказания
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        X_norm = (X - self._mean) / self._std

        labels = self._dbscan_fit(X_norm)

        # Храним нормированные обучающие данные для k-NN предсказания
        self._X_norm_train = X_norm
        self._train_labels = labels.astype(int)

        self._is_fitted = True
        self._model_confidence = min(1.0, len(self._buffer) / 500)
        self._build_patterns(labels, X)

    def _dbscan_fit(self, X_norm: np.ndarray) -> np.ndarray:
        from sklearn.cluster import DBSCAN
        from sklearn.metrics.pairwise import euclidean_distances

        eps = self.eps
        if eps is None:
            # Автокалибровка: 20-й перцентиль ближайших расстояний
            sample_size = min(200, len(X_norm))
            rng = np.random.default_rng(0)
            idx = rng.choice(len(X_norm), sample_size, replace=False)
            dists = euclidean_distances(X_norm[idx])
            np.fill_diagonal(dists, np.inf)
            nn_dists = dists.min(axis=1)
            # игнорируем нулевые расстояния (дубликаты)
            nn_nonzero = nn_dists[nn_dists > 0]
            if len(nn_nonzero) > 0:
                eps = float(np.percentile(nn_nonzero, 20))
            else:
                eps = 1.0  # все дубликаты — берём широкий eps
            eps = max(eps, 0.3)
        self._fitted_eps = eps

        self._model = DBSCAN(
            eps=eps,
            min_samples=max(2, self.min_cluster_size // 2),
            metric='euclidean',
        )
        return self._model.fit_predict(X_norm)

    def _predict(self, windows: list[np.ndarray]) -> list[int]:
        """
        k-NN по нормированным обучающим данным.
        Если большинство среди k ближайших соседей — кластер C, назначаем C.
        Если расстояние до ближайшего соседа > fitted_eps*2 — noise (-1).
        """
        if not self._is_fitted:
            return [-1] * len(windows)

        X = np.stack(windows, axis=0)
        X_norm = (X - self._mean) / self._std

        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(X_norm, self._X_norm_train)  # (test, train)

        k = min(5, len(self._X_norm_train))
        threshold = self._fitted_eps * 2.0
        result = []

        for i in range(len(windows)):
            row = dists[i]
            nearest_dist = row.min()
            if nearest_dist > threshold:
                result.append(-1)
                continue
            # топ-k соседей
            top_k_idx = np.argpartition(row, k)[:k]
            top_k_labels = self._train_labels[top_k_idx]
            # голосование (исключаем noise=-1 если есть нормальные соседи)
            non_noise = top_k_labels[top_k_labels != -1]
            if len(non_noise) == 0:
                result.append(-1)
            else:
                from collections import Counter
                result.append(int(Counter(non_noise.tolist()).most_common(1)[0][0]))

        return result

    def _build_patterns(self, labels: np.ndarray, X: np.ndarray) -> None:
        from collections import defaultdict
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            buckets[int(lbl)].append(i)

        self._patterns = {}
        for cluster_id, indices in buckets.items():
            centroid = X[indices].mean(axis=0)
            is_noise = cluster_id == -1
            examples = []
            for i in indices[:5]:
                meta = self._buffer_meta[i]
                examples.append(f"session={meta['session_id']} w={meta['window_start']}")
            self._patterns[cluster_id] = DialogPattern(
                cluster_id=cluster_id,
                size=len(indices),
                centroid=centroid,
                is_noise=is_noise,
                examples=examples,
            )

    # ── Свойство доступа к конфигурации ──────────────────────────────────────

    @property
    def _model_confidence(self) -> float:
        return getattr(self, '_conf', 0.0)

    @_model_confidence.setter
    def _model_confidence(self, v: float) -> None:
        self._conf = v
