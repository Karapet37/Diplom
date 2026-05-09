"""
p50_topic_mlp.py — P50: выбор актуальной темы через MLP.

Принцип (перцептрон на сочетаниях текстовых паттернов):
  Входные данные:
    - текст текущей реплики (TF-IDF n-gram представление)
    - агрегированный P1-P48 вектор из P49 (текущий контекст)
    - prev_aggregate из P49 (предыдущий контекст)

  Задача 1: классификация темы (topic_id → название)
  Задача 2: обнаружение смены темы (тематический сдвиг)

Обучение: DataSets/everyday_conversations.jsonl
  Каждый диалог аннотирован полем 'full_topic' (иерархическая тема).
  Обучаем на парах (текст реплики → тема).

P50 НЕ входит в DialogContextMatrix.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .p_subsystem import PVariant, PVariantOutput

P_DIM = 48

# ─── Выходные данные P50 ──────────────────────────────────────────────────────

@dataclass
class P50Output:
    topic_id: int
    topic_label: str
    topic_shift: bool
    shift_confidence: float   # 0..1 — насколько уверен в смене темы
    stability_score: float    # 0..1 — насколько стабильна текущая тема


# ─── P50TopicSelector ─────────────────────────────────────────────────────────

class P50TopicSelector:
    """
    MLP-классификатор тем диалога.

    Обучается на `everyday_conversations.jsonl` (поле full_topic как метка).
    TF-IDF на биграммах + MLP (2 скрытых слоя) → тема.

    Для обнаружения смены темы: косинусное расстояние между
    текущим и предыдущим P49-агрегатом.
    """

    # Порог смены темы по косинусному расстоянию P49 агрегатов
    _SHIFT_THRESHOLD = 0.42

    def __init__(self, model_path: Path | None = None):
        self._pipeline: Any = None
        self._label_map: dict[int, str] = {}
        self._trained = False
        if model_path and Path(model_path).exists():
            self.load(Path(model_path))

    @property
    def is_trained(self) -> bool:
        return self._trained

    def select(
        self,
        text: str,
        p49_aggregate: np.ndarray | None,
        prev_aggregate: np.ndarray | None,
        prev_topic_id: int = -1,
    ) -> P50Output:
        """
        Классифицирует тему и определяет тематический сдвиг.

        Два канала сдвига:
          1. P49 канал: косинусное расстояние между агрегатами (работает когда P1-P48 обучены)
          2. MLP канал: смена топ-уровня темы (работает сразу после обучения MLP)
        Итоговый сдвиг = max(оба канала).
        """
        # Канал 1: P49 агрегаты
        p49_shift_conf = self._shift_confidence(p49_aggregate, prev_aggregate)

        topic_id = -1
        topic_label = 'unknown'
        stability = 1.0 - p49_shift_conf
        mlp_shift_conf = 0.0

        if self._trained and self._pipeline is not None:
            try:
                idx = int(self._pipeline.predict([text])[0])
                label = self._label_map.get(idx, str(idx))
                proba = self._pipeline.predict_proba([text])[0]
                conf = float(max(proba))

                topic_id = idx
                topic_label = label
                stability = conf

                # Канал 2: если топ-уровень темы изменился по MLP — это сдвиг
                if prev_topic_id >= 0 and prev_topic_id != idx:
                    # Теперь label = top-level тема (63 класса), сравниваем напрямую
                    mlp_shift_conf = min(0.85, conf * 1.2)  # взвешиваем уверенностью модели
            except Exception:
                pass

        shift_conf = max(p49_shift_conf, mlp_shift_conf)
        topic_shift = shift_conf > self._SHIFT_THRESHOLD

        return P50Output(
            topic_id=topic_id,
            topic_label=topic_label,
            topic_shift=topic_shift,
            shift_confidence=round(shift_conf, 3),
            stability_score=round(stability, 3),
        )

    def _shift_confidence(
        self,
        cur: np.ndarray | None,
        prev: np.ndarray | None,
    ) -> float:
        """Чем ниже косинусное сходство P49-агрегатов — тем выше вероятность сдвига."""
        if cur is None or prev is None:
            return 0.0
        d = float(np.linalg.norm(cur) * np.linalg.norm(prev))
        if d < 1e-8:
            return 0.0
        cos = float(np.dot(cur, prev) / d)
        return max(0.0, 1.0 - cos)

    def train(self, dataset_path: Path, model_path: Path | None = None) -> bool:
        """
        Обучает MLP на everyday_conversations.jsonl.

        Каждая реплика из `messages` получает метку `full_topic` своего диалога.
        Минимум 50 примеров, иначе False.
        """
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            return False

        texts: list[str] = []
        topics: list[str] = []

        with open(dataset_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    continue
                topic = str(ex.get('topic') or '').strip()  # top-level: 63 classes
                if not topic:
                    continue
                for msg in (ex.get('messages') or []):
                    content = str(msg.get('content') or '').strip()
                    if len(content) > 5:
                        texts.append(content)
                        topics.append(topic)

        if len(texts) < 50:
            return False

        le = LabelEncoder()
        y = le.fit_transform(topics)
        self._label_map = {int(i): str(label) for i, label in enumerate(le.classes_)}

        self._pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=6000,
                analyzer='word',
                min_df=1,
                sublinear_tf=True,
            )),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
            )),
        ])
        self._pipeline.fit(texts, y)
        self._trained = True

        if model_path is not None:
            self.save(Path(model_path))

        return True

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(
                {'pipeline': self._pipeline, 'label_map': self._label_map}, f
            )

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self._pipeline = data['pipeline']
            self._label_map = data['label_map']
            self._trained = True
            return True
        except Exception:
            return False


# ─── Singleton ────────────────────────────────────────────────────────────────

_GLOBAL_SELECTOR: P50TopicSelector | None = None
_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / 'models' / 'p50_topic_mlp.pkl'


def get_global_selector() -> P50TopicSelector:
    global _GLOBAL_SELECTOR
    if _GLOBAL_SELECTOR is None:
        _GLOBAL_SELECTOR = P50TopicSelector(
            model_path=_DEFAULT_MODEL_PATH if _DEFAULT_MODEL_PATH.exists() else None
        )
    return _GLOBAL_SELECTOR


def reset_global_selector() -> None:
    global _GLOBAL_SELECTOR
    _GLOBAL_SELECTOR = None


# ─── P50Variant — интеграция в PFamily framework ──────────────────────────────

class P50Variant(PVariant):
    """
    P50 вариант для PFamily.

    Читает из context:
        '_f49_aggregate'  : np.ndarray | None — выход P49
        '_p49_prev'       : np.ndarray | None — предыдущий P49 агрегат
        'p50_selector'    : P50TopicSelector (необяз., fallback → global)

    Возвращает scalar score:
        topic_stable   → stability_score
        topic_shift    → shift_confidence
        topic_return   → если prev shift потом похожесть вернулась
    """

    def __init__(self, variant_id: str, label: str):
        super().__init__(variant_id, label)
        self._trained = True

    def forward(self, text: str, context: dict[str, Any]) -> PVariantOutput:
        # Вычисляем P50Output один раз за ход — кешируем в context
        out: P50Output | None = context.get('_f50_output')
        if out is None:
            selector: P50TopicSelector = (
                context.get('p50_selector') or get_global_selector()
            )
            p49_agg: np.ndarray | None = context.get('_f49_aggregate')
            prev_agg: np.ndarray | None = context.get('_p49_prev')
            prev_topic_id: int = int(context.get('_f50_prev_topic_id') or -1)

            out = selector.select(text, p49_agg, prev_agg, prev_topic_id)
            context['_f50_output'] = out
            # Сохраняем id для следующего хода (вызывающий код должен передать в новый ctx)
            context['_f50_new_topic_id'] = out.topic_id

        score = self._label_score(out)
        return PVariantOutput(self.variant_id, self.label, round(score, 3), 0.70)

    def _label_score(self, out: P50Output) -> float:
        if self.label == 'topic_stable':
            return out.stability_score if not out.topic_shift else 0.0
        if self.label == 'topic_shift':
            return out.shift_confidence if out.topic_shift else 0.0
        if self.label == 'topic_return':
            # Смена с последующим возвратом к похожей теме — сложный сигнал
            # Пока простая эвристика: shift + stability оба умеренные
            if out.topic_shift and 0.3 <= out.stability_score <= 0.7:
                return 0.65
            return 0.0
        return 0.0

    def train(self, positives, negatives, scores=None): pass
    def save(self, path): pass
    def load(self, path): pass

    @property
    def is_trained(self) -> bool:
        return True


# ─── CLI-обучение ─────────────────────────────────────────────────────────────

def train_p50_from_dataset(
    dataset_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> bool:
    """
    Обучить P50 MLP из командной строки.

    python -m agent_system.p50_topic_mlp
    """
    ds = Path(dataset_path) if dataset_path else (
        Path(__file__).parent.parent / 'DataSets' / 'everyday_conversations.jsonl'
    )
    mp = Path(model_path) if model_path else _DEFAULT_MODEL_PATH

    if not ds.exists():
        print(f'Dataset not found: {ds}')
        return False

    selector = P50TopicSelector()
    ok = selector.train(ds, mp)
    if ok:
        n = len(selector._label_map)
        print(f'P50 MLP trained: {n} topic classes → {mp}')
    else:
        print('Training failed (too few samples or sklearn missing)')
    return ok


if __name__ == '__main__':
    train_p50_from_dataset()
