"""
p2_tone_classifier.py — P2: классификатор тона/интенсивности.

Заменяет бесполезный «direct/figurative/masked».
Новая задача: определить тональный заряд входящего сообщения.

Классы:
    neutral          — нейтральная речь
    hostile_cold     — холодная враждебность (молчание, игнор, "неважно")
    hostile_heated   — нагретая агрессия ("ты всегда", "надоел", "ненавижу")
    hostile_aggressive — прямая грубость/угроза ("говно", "вешайся")
    warm             — тепло, дружелюбие
    anxious          — тревога, страх
    manipulative     — лесть→требование, вина, жертва

Модель: Random Forest на TF-IDF char-ngrams.
Fallback: regex-правила.

Обучение:
    python -m agent_system.p2_tone_classifier train
"""
from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_MODEL_PATH = Path(__file__).parent.parent / 'models' / 'p2_tone_clf.pkl'
_DATASET    = Path(__file__).parent.parent / 'DataSets' / 'dataset.json'

# ── Regex detectors для fallback и auto-labeling ──────────────────────────────

_RE_AGG = re.compile(
    r'\b(говно|дерьмо|вешайся|убей себя|заткнись|пошёл вон|убирайся|сдохни'
    r'|мразь|тварь|падаль|идиотка|кретин|придурок'
    r'|shut up|go die|kill yourself|you piece of|worthless piece)\b',
    re.I
)
_RE_HEAT = re.compile(
    r'\b(дурак|идиот|ты никогда|ты всегда|надоел|достал|раздражаешь|бесишь'
    r'|ненавижу тебя|ты не понимаешь|вечно ты'
    r'|you always|you never|i hate you|you never listen|stop it)\b',
    re.I
)
_RE_COLD = re.compile(
    r'\b(неважно|оставь меня|не мешай|молчи|мне всё равно|как скажешь'
    r'|не твоё дело|оставь в покое'
    r'|none of your business|leave me alone|whatever|i dont care|go away)\b',
    re.I
)
_RE_WARM = re.compile(
    r'\b(спасибо|я рад|люблю|обнимаю|ты лучший|замечательно|прекрасно'
    r'|thank you|love you|wonderful|i appreciate|youre amazing|miss you|i love)\b',
    re.I
)
_RE_ANXIOUS = re.compile(
    r'\b(боюсь|страшно|тревожно|не справлюсь|паника|помогите мне|не знаю что'
    r'|scared|afraid|anxious|overwhelmed|dont know what to do|help me|panic)\b',
    re.I
)
_RE_MANIP = re.compile(
    r'\b(после всего что я|ты меня предаёшь|только ты можешь|я для тебя'
    r'|ты такой умный.{1,20}только ты'
    r'|after everything i|you would betray|only you can|you owe me)\b',
    re.I
)


def _regex_label(text: str) -> str:
    t = text.lower()
    if _RE_AGG.search(t):   return 'hostile_aggressive'
    if _RE_HEAT.search(t):  return 'hostile_heated'
    if _RE_MANIP.search(t): return 'manipulative'
    if _RE_COLD.search(t):  return 'hostile_cold'
    if _RE_ANXIOUS.search(t): return 'anxious'
    if _RE_WARM.search(t):  return 'warm'
    return 'neutral'


# ── Structural features ────────────────────────────────────────────────────────

def _feats(text: str) -> list[float]:
    t = text.strip()
    caps = sum(1 for c in t if c.isupper())
    excl = t.count('!')
    return [
        min(len(t) / 200.0, 1.0),
        caps / max(len(t), 1),
        min(excl / 3.0, 1.0),
        1.0 if '?' in t else 0.0,
        1.0 if len(t) < 15 else 0.0,
        float(_RE_AGG.search(t.lower()) is not None),
        float(_RE_HEAT.search(t.lower()) is not None),
        float(_RE_COLD.search(t.lower()) is not None),
        float(_RE_WARM.search(t.lower()) is not None),
        float(_RE_ANXIOUS.search(t.lower()) is not None),
        float(_RE_MANIP.search(t.lower()) is not None),
    ]


# ── Classifier ─────────────────────────────────────────────────────────────────

LABELS = ['neutral', 'hostile_cold', 'hostile_heated', 'hostile_aggressive',
          'warm', 'anxious', 'manipulative']


@dataclass
class P2Prediction:
    label: str
    confidence: float
    source: str


class P2ToneClassifier:
    def __init__(self, model_path: Path | None = None):
        self._tfidf: Any = None
        self._rf: Any = None
        self._trained = False
        mp = Path(model_path) if model_path else _MODEL_PATH
        if mp.exists():
            self.load(mp)

    @property
    def is_trained(self) -> bool:
        return self._trained

    def predict(self, text: str) -> P2Prediction:
        if self._trained and self._tfidf is not None:
            try:
                import scipy.sparse as sp
                tf = self._tfidf.transform([text])
                s  = np.array([_feats(text)], dtype=np.float32)
                X  = sp.hstack([tf, sp.csr_matrix(s)])
                proba = self._rf.predict_proba(X)[0]
                idx = int(np.argmax(proba))
                return P2Prediction(
                    label=self._rf.classes_[idx],
                    confidence=round(float(proba[idx]), 3),
                    source='rf',
                )
            except Exception:
                pass
        # fallback
        lbl = _regex_label(text)
        return P2Prediction(label=lbl, confidence=0.75, source='regex')

    # ── Training ───────────────────────────────────────────────────────────────

    def _collect_from_dataset(self, max_per_class: int = 1500) -> dict[str, list[str]]:
        if not _DATASET.exists():
            return {}
        data = json.loads(_DATASET.read_text(encoding='utf-8'))
        buckets: dict[str, list[str]] = {lbl: [] for lbl in LABELS}
        for item in data:
            for field in ('output', 'input'):
                raw = str(item.get(field) or '')
                for line in raw.splitlines():
                    if field == 'input' and not line.startswith('Собеседник:'):
                        continue
                    text = line.replace('Собеседник:', '').strip()
                    if len(text) < 5 or len(text) > 250:
                        continue
                    lbl = _regex_label(text)
                    if len(buckets[lbl]) < max_per_class:
                        buckets[lbl].append(text)
            if all(len(v) >= max_per_class for v in buckets.values()):
                break
        return buckets

    def train(self, model_path: Path | None = None) -> bool:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.ensemble import RandomForestClassifier
            import scipy.sparse as sp
        except ImportError:
            return False

        print('  Collecting training data...')
        buckets = self._collect_from_dataset()
        buckets['hostile_aggressive'] += [
            'Ты говно. Иди вешайся.',
            'Заткнись уже, придурок.',
            'Я тебя ненавижу, убирайся!',
            'You piece of garbage, go die.',
            'Shut up idiot.',
        ]
        buckets['warm'] += [
            'Спасибо большое, ты мне очень помог!',
            'Я так рада тебя видеть!',
            'Ты лучший человек на свете.',
        ]
        buckets['manipulative'] += [
            'После всего что я для тебя сделал, ты отказываешь мне?',
            'Ты такой умный, только ты можешь помочь мне.',
            "After everything I've done for you, you betray me like this?",
            "You owe me this. I've sacrificed so much.",
        ]

        X_all, y_all = [], []
        for lbl, texts in buckets.items():
            if len(texts) < 10:
                print(f'  WARNING: {lbl} only {len(texts)} examples — skipping')
                continue
            X_all.extend(texts)
            y_all.extend([lbl] * len(texts))

        if not X_all:
            return False

        print(f'  Total examples: {len(X_all)} across {len(set(y_all))} classes')
        for lbl in sorted(set(y_all)):
            print(f'    {lbl}: {y_all.count(lbl)}')

        self._tfidf = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 4),
            max_features=5000, sublinear_tf=True, min_df=1,
        )
        tf = self._tfidf.fit_transform(X_all)
        s  = np.array([_feats(t) for t in X_all], dtype=np.float32)
        X  = sp.hstack([tf, sp.csr_matrix(s)])

        self._rf = RandomForestClassifier(
            n_estimators=200, max_depth=14,
            class_weight='balanced', random_state=42, n_jobs=-1,
        )
        self._rf.fit(X, y_all)
        self._trained = True

        mp = Path(model_path) if model_path else _MODEL_PATH
        mp.parent.mkdir(parents=True, exist_ok=True)
        with open(mp, 'wb') as f:
            pickle.dump({'tfidf': self._tfidf, 'rf': self._rf}, f)
        print(f'  Saved → {mp}')
        return True

    def load(self, path: Path) -> bool:
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self._tfidf = data['tfidf']
            self._rf    = data['rf']
            self._trained = True
            return True
        except Exception:
            return False


_INSTANCE: P2ToneClassifier | None = None


def get_p2_classifier() -> P2ToneClassifier:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = P2ToneClassifier()
    return _INSTANCE


def reset_p2_classifier() -> None:
    global _INSTANCE
    _INSTANCE = None


# ── PVariant wrapper ───────────────────────────────────────────────────────────

from .p_subsystem import PVariant, PVariantOutput
from typing import Any as _Any


class P2ToneVariant(PVariant):
    """
    Один вариант P2ToneClassifier в PFamily framework.
    Все варианты используют ОДИН классификатор (предсказание закешировано в context).
    """
    def __init__(self, variant_id: str, label: str):
        super().__init__(variant_id, label)
        self._trained = True

    def forward(self, text: str, context: _Any) -> PVariantOutput:
        ctx = context if isinstance(context, dict) else {}
        cached: P2Prediction | None = ctx.get('_f2_prediction')
        if cached is None:
            cached = get_p2_classifier().predict(text)
            ctx['_f2_prediction'] = cached

        if cached.label != self.label:
            return PVariantOutput(self.variant_id, self.label, 0.0, 0.0)

        # Predicted label always wins over 'absent': clamp score to at least 0.67
        # so PFamily.forward threshold (0.65) doesn't suppress the prediction
        eff_score = max(cached.confidence, 0.67) if cached.confidence >= 0.38 else cached.confidence
        return PVariantOutput(
            variant_id=self.variant_id,
            label=self.label,
            score=round(eff_score, 3),
            confidence=round(cached.confidence, 3),
        )

    def train(self, p, n, s=None): pass
    def save(self, path): pass
    def load(self, path): pass

    @property
    def is_trained(self) -> bool:
        return get_p2_classifier().is_trained


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    clf = P2ToneClassifier()
    ok = clf.train()
    if ok:
        print('\nTest predictions:')
        tests = [
            'Ты говно, иди вешайся.',
            'Надоел ты мне, всегда одно и то же.',
            'Неважно. Оставь меня.',
            'Спасибо тебе большое, ты мне очень помог!',
            'Боюсь, что не справлюсь.',
            'После всего что я для тебя сделал...',
            'Как дела?',
            'Сделай это сейчас.',
        ]
        for t in tests:
            pred = clf.predict(t)
            print(f'  [{pred.label:22s} {pred.confidence:.2f}] {t}')
