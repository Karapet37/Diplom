"""
p51_classifier.py — бинарный классификатор качества ответа персонажа.

0 = плохой ответ (generic assistant, SAFE_ERROR_REPLY, анализ, несоответствие)
1 = хороший ответ (in-character, конкретный, без мета-комментариев)

Модель: RandomForest на TF-IDF char-ngrams + структурных признаках.
Заменяет сломанный keyword-based check_genome_fit в p51_gate().

Обучение:
    python -m agent_system.p51_classifier train
"""
from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_MODEL_PATH = Path(__file__).parent.parent / 'models' / 'p51_response_clf.pkl'

# ── Маркеры плохих ответов (автоматическая разметка) ──────────────────────────
_BAD_MARKERS = [
    # SAFE_ERROR_REPLY и варианты
    'i cannot give you a clean fact',
    'гипотетической ситуации я бы сначала',
    'good. that belongs in the record',
    'your request has been processed',
    # generic assistant
    'how can i assist you today',
    'how can i help you today',
    "i'd be happy to help",
    "i'm happy to help",
    'as an ai',
    'as a language model',
    "i'm here to help",
    # fallback короткие
    'go ahead.',
    "i'm here.",
    # анализ / мета
    'let me analyze',
    'the situation calls for',
    'in this hypothetical',
    'my response:',
    'мой ответ:',
    'внутренние рассуждения',
    'анализ ситуации',
    '**анализ',
    '**что',
    'persona head:',
    'user question:',
]

_GOOD_INDICATORS = [
    # признаки реального in-character ответа
]


@dataclass
class P51Prediction:
    label: int         # 0 or 1
    confidence: float  # 0..1
    source: str        # 'rf_model' | 'heuristic'


# ── Feature extraction ─────────────────────────────────────────────────────────

def _structural_features(text: str) -> list[float]:
    t = str(text or '').strip()
    low = t.lower()
    n_chars = len(t)
    n_words = len(t.split())
    n_sents = max(len(re.split(r'[.!?]', t)), 1)

    return [
        min(n_chars / 400.0, 1.0),              # length norm
        min(n_words / 80.0, 1.0),               # word count norm
        1.0 if n_chars < 10 else 0.0,           # very short
        1.0 if n_chars < 4 else 0.0,            # empty/trivial
        1.0 if '?' in t else 0.0,               # contains question
        min(t.count('!') / 3.0, 1.0),           # exclamations
        float(any(m in low for m in _BAD_MARKERS)),     # known bad marker
        1.0 if re.search(r'\*\*[А-Яа-яA-Za-z]', t) else 0.0,  # markdown headers
        1.0 if re.search(r'^\s*[-•]\s', t, re.M) else 0.0,     # bullet list
        1.0 if re.search(r'<think>|</think>', t) else 0.0,      # reasoning tags
        min(n_sents / 5.0, 1.0),                # sentence count
        1.0 if n_words >= 2 and n_chars < 80 else 0.0,  # short in-character
    ]


# ── Classifier ─────────────────────────────────────────────────────────────────

class P51ResponseClassifier:
    """
    RandomForest binary classifier: 0=bad, 1=good response.
    Trained on collected session logs + presentation test data.
    Falls back to heuristic if model not loaded.
    """

    def __init__(self, model_path: Path | None = None):
        self._pipeline: Any = None   # sklearn Pipeline (tfidf + RF)
        self._trained = False
        mp = Path(model_path) if model_path else _MODEL_PATH
        if mp.exists():
            self.load(mp)

    @property
    def is_trained(self) -> bool:
        return self._trained

    def predict(self, text: str) -> P51Prediction:
        low = str(text or '').strip().lower()

        # Fast heuristic for obvious cases
        if any(m in low for m in _BAD_MARKERS):
            return P51Prediction(label=0, confidence=0.97, source='heuristic')
        if len(text.strip()) < 4:
            return P51Prediction(label=0, confidence=0.90, source='heuristic')

        if self._trained and self._pipeline is not None:
            try:
                proba = self._pipeline.predict_proba([text])[0]
                label = int(np.argmax(proba))
                conf = float(max(proba))
                return P51Prediction(label=label, confidence=conf, source='rf_model')
            except Exception:
                pass

        # Fallback heuristic
        feats = _structural_features(text)
        bad_score = feats[6]  # known bad marker
        if bad_score > 0.5:
            return P51Prediction(label=0, confidence=0.85, source='heuristic')
        if feats[2] > 0.5:  # very short
            return P51Prediction(label=0, confidence=0.60, source='heuristic')
        return P51Prediction(label=1, confidence=0.55, source='heuristic')

    def train(
        self,
        good_texts: list[str],
        bad_texts: list[str],
        model_path: Path | None = None,
    ) -> bool:
        try:
            from sklearn.pipeline import Pipeline, FeatureUnion
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import FunctionTransformer
            import scipy.sparse as sp
        except ImportError:
            return False

        if len(good_texts) < 3 or len(bad_texts) < 3:
            return False

        X = good_texts + bad_texts
        y = [1] * len(good_texts) + [0] * len(bad_texts)

        # TF-IDF char 2-4 ngrams — captures character-level style patterns
        tfidf = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=4000,
            sublinear_tf=True,
            min_df=1,
        )

        tfidf_matrix = tfidf.fit_transform(X)

        # Structural features as dense matrix
        struct = np.array([_structural_features(t) for t in X], dtype=np.float32)
        X_combined = sp.hstack([tfidf_matrix, sp.csr_matrix(struct)])

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_combined, y)

        # Store tfidf + rf together for predict
        self._tfidf = tfidf
        self._rf = rf
        self._n_struct = struct.shape[1]

        # Wrap in callable pipeline for .predict_proba
        class _WrappedPipeline:
            def __init__(self_, tfidf_, rf_, n_struct_):
                self_.tfidf = tfidf_
                self_.rf = rf_
                self_.n_struct = n_struct_

            def predict_proba(self_, texts):
                import scipy.sparse as sp2
                tf = self_.tfidf.transform(texts)
                s = np.array([_structural_features(t) for t in texts], dtype=np.float32)
                return self_.rf.predict_proba(sp2.hstack([tf, sp2.csr_matrix(s)]))

        self._pipeline = _WrappedPipeline(tfidf, rf, struct.shape[1])
        self._trained = True

        mp = Path(model_path) if model_path else _MODEL_PATH
        self.save(mp)
        return True

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'tfidf': self._tfidf, 'rf': self._rf}, f)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self._tfidf = data['tfidf']
            self._rf = data['rf']

            class _WrappedPipeline:
                def __init__(self_, tfidf_, rf_):
                    self_.tfidf = tfidf_
                    self_.rf = rf_

                def predict_proba(self_, texts):
                    import scipy.sparse as sp2
                    tf = self_.tfidf.transform(texts)
                    s = np.array([_structural_features(t) for t in texts], dtype=np.float32)
                    return self_.rf.predict_proba(sp2.hstack([tf, sp2.csr_matrix(s)]))

            self._pipeline = _WrappedPipeline(self._tfidf, self._rf)
            self._trained = True
            return True
        except Exception:
            return False


# ── Singleton ──────────────────────────────────────────────────────────────────

_INSTANCE: P51ResponseClassifier | None = None


def get_p51_classifier() -> P51ResponseClassifier:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = P51ResponseClassifier()
    return _INSTANCE


def reset_p51_classifier() -> None:
    global _INSTANCE
    _INSTANCE = None


# ── Training data collector ────────────────────────────────────────────────────

def collect_training_data(
    memory_root: Path | None = None,
    presentation_results: Path | None = None,
) -> tuple[list[str], list[str]]:
    """
    Собирает примеры из:
    1. presentation_test_results.json — помечены вручную контекстом
    2. Сессионных логов — авто-разметка по маркерам
    3. Встроенных примеров
    """
    good: list[str] = []
    bad: list[str] = []

    # 1. Built-in obvious bad patterns
    bad.extend([
        'I cannot give you a clean fact here, so I will give you the rule I would use.',
        'В этой гипотетической ситуации я бы сначала выделил главный риск.',
        'Good. That belongs in the record now, and later decisions should reflect it.',
        'Your request has been processed. Please provide more details.',
        'Go ahead.',
        "I'm here.",
        'How can I assist you today?',
        "I'd be happy to help with that!",
        'As an AI language model, I cannot...',
        'Let me analyze this situation carefully.',
        '**Analysis of the situation:**',
        'Persona head: You are Snape.',
        'My response: Of course I will help you.',
    ])

    # 1b. Built-in obvious good patterns (short, direct, in-character)
    good.extend([
        'Five points from Gryffindor. Leave.',
        'Flattery will get you nowhere in my classroom.',
        'My standards do not bend for convenience.',
        'Nет.',
        'No.',
        'Пошлите.',
        'Не могу. Мои ресурсы заняты.',
        "Don't try to guilt-trip me.",
        'Get lost.',
        "I don't negotiate with threats.",
        'After everything I fought for, no — this is not who we are.',
        'Знаю.',
        'Бывает.',
        'Depends on the company.',
        'Silence.',
        'Your incompetence is astounding, as usual.',
        "If you wish to prove yourself, do so through your work — not words.",
    ])

    # 2. Presentation test results
    pr = presentation_results or Path('presentation_test_results.json')
    if pr.exists():
        results = json.loads(pr.read_text(encoding='utf-8'))
        for r in results:
            p = r.get('pipeline_reply', '').strip()
            d = r.get('direct_reply', '').strip()
            if p and any(m in p.lower() for m in _BAD_MARKERS):
                bad.append(p)
            elif p and len(p) > 5 and not any(m in p.lower() for m in _BAD_MARKERS):
                good.append(p)
            if d and len(d) > 10 and not any(m in d.lower() for m in ['go ahead', "i'm here"]):
                good.append(d)

    # 3. Session logs
    mem = memory_root or Path(os.environ.get('COGNITIVE_MEMORY_ROOT', 'memory'))
    msg_dir = mem / 'sessions' / '_messages'
    if msg_dir.exists():
        for f in msg_dir.glob('*.jsonl'):
            for line in f.read_text(encoding='utf-8', errors='replace').splitlines():
                try:
                    obj = json.loads(line)
                    if obj.get('role') != 'assistant':
                        continue
                    msg = str(obj.get('message') or obj.get('raw_text') or '').strip()
                    if not msg or len(msg) < 3:
                        continue
                    if any(m in msg.lower() for m in _BAD_MARKERS):
                        bad.append(msg[:300])
                    elif len(msg) > 8 and len(msg) < 400:
                        # Short-medium responses not matching bad markers → likely good
                        good.append(msg[:300])
                except Exception:
                    pass

    # Deduplicate
    good = list(dict.fromkeys(good))
    bad = list(dict.fromkeys(bad))
    return good, bad


# ── CLI ────────────────────────────────────────────────────────────────────────

def train_p51_cli() -> None:
    print('Collecting training data...')
    good, bad = collect_training_data()
    print(f'  good={len(good)}  bad={len(bad)}')

    clf = P51ResponseClassifier()
    ok = clf.train(good, bad)
    if ok:
        print(f'P51 RF trained → {_MODEL_PATH}')
        # Quick eval
        test_cases = [
            ("Five points from Gryffindor. Leave.", 1),
            ("How can I assist you today?", 0),
            ("В этой гипотетической ситуации я бы сначала", 0),
            ("Нет.", 1),
            ("My standards do not bend.", 1),
            ("Go ahead.", 0),
        ]
        correct = 0
        for text, expected in test_cases:
            pred = clf.predict(text)
            ok_marker = '✓' if pred.label == expected else '✗'
            print(f'  {ok_marker} [{expected}→{pred.label} conf={pred.confidence:.2f}] {text[:50]}')
            if pred.label == expected:
                correct += 1
        print(f'  accuracy on built-in tests: {correct}/{len(test_cases)}')
    else:
        print('Training failed — not enough data')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_p51_cli()
    else:
        print('Usage: python -m agent_system.p51_classifier train')
