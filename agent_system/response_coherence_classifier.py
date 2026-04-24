"""
Response Coherence Classifier — Pn+1 per-persona quality gate.

Architecture
------------
user_input + response
    │
    ▼
FeatureExtractor  →  feat[N]   (deterministic, language-aware)
    │
    ├── RuleEngine             (hard-coded veto rules, persona-agnostic)
    │       → veto bool + reason
    ├── PersonaPerceptron      (simple numpy perceptron, per-persona weights)
    │       → score float [0..1]
    ▼
CoherenceResult
    { score, veto, reason, features }

Training
--------
When a user saves a correction (original → corrected), call `record_correction()`.
The original reply is a negative example (label=0), the corrected is a positive (label=1).
Weights are updated online via gradient descent (learning_rate=0.05).
Weights persist in {heads_dir}/{persona}/coherence_weights.json.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ─── Feature names ────────────────────────────────────────────────────────────

_FEATURE_NAMES = [
    'has_analysis_opener',       # "Let me analyze", "**Analysis", "Let me think through"
    'has_hesitation_start',      # starts with Эм..., Well..., Hmm..., Um...
    'language_mismatch',         # user in Russian, response in English (or vice versa)
    'is_too_long',               # >60 words (bad for terse personas)
    'has_markdown_analysis',     # contains **bold headers** or numbered analysis lists
    'speaks_third_person_self',  # response talks about persona name in 3rd person
    'has_refusal_marker',        # "I can't", "I'm unable to", "Я не могу"
    'has_meta_commentary',       # "I'd like to note", "It's worth pointing out"
    'starts_with_punctuation',   # starts with "..." or "—" (sometimes persona-correct, sometimes not)
    'is_empty',                  # empty or whitespace-only response
]

_N_FEATURES = len(_FEATURE_NAMES)

# ─── Hard veto patterns ───────────────────────────────────────────────────────

_ANALYSIS_OPENERS = [
    r'let me analyze',
    r'let me think through',
    r'let me carefully',
    r'\*\*analysis',
    r'analysis of the situation',
    r'let me break this down',
    r'i\'ll analyze',
    r'to analyze this',
    r'\*\*let me',
]

_HESITATION_STARTS = [
    r'^эм[\.\.\.\s]',
    r'^эм\b',
    r'^well[\.,\s]',
    r'^hmm[\.,\s]',
    r'^um[\.,\s]',
    r'^uh[\.,\s]',
]

_REFUSAL_MARKERS = [
    "i can't",
    "i cannot",
    "i'm unable",
    "i am unable",
    "i don't feel comfortable",
    "я не могу ответить",
    "не могу это",
]

_META_COMMENTARY = [
    "i'd like to note",
    "it's worth noting",
    "it's important to",
    "it is worth",
    "i should mention",
    "i want to clarify",
]


# ─── Feature extraction ───────────────────────────────────────────────────────

def _is_cyrillic_dominant(text: str) -> bool:
    cyrillic = sum(1 for c in text if 'Ѐ' <= c <= 'ӿ')
    latin = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    return cyrillic > latin


def extract_features(user_input: str, response: str) -> np.ndarray:
    feat = np.zeros(_N_FEATURES, dtype=np.float32)
    resp_lower = response.lower().strip()
    resp_stripped = response.strip()
    words = resp_stripped.split()

    # 0: analysis opener
    feat[0] = float(any(re.search(p, resp_lower) for p in _ANALYSIS_OPENERS))

    # 1: hesitation start
    feat[1] = float(any(re.search(p, resp_lower) for p in _HESITATION_STARTS))

    # 2: language mismatch (user Russian → response English)
    user_cyrillic = _is_cyrillic_dominant(user_input)
    resp_cyrillic = _is_cyrillic_dominant(response)
    feat[2] = float(user_cyrillic and not resp_cyrillic and len(response.strip()) > 10)

    # 3: too long
    feat[3] = float(len(words) > 60)

    # 4: markdown analysis (headers or numbered analysis)
    feat[4] = float(bool(re.search(r'\*\*[A-ZА-Я].{3,}\*\*', response) or re.search(r'^\d+\.\s', response, re.MULTILINE)))

    # 5: speaks third person about self — rough heuristic (persona name in 3rd person)
    # We don't know the persona name here; checked in RuleEngine with persona context
    feat[5] = 0.0

    # 6: refusal
    feat[6] = float(any(m in resp_lower for m in _REFUSAL_MARKERS))

    # 7: meta commentary
    feat[7] = float(any(m in resp_lower for m in _META_COMMENTARY))

    # 8: starts with punctuation (neutral signal)
    feat[8] = float(bool(resp_stripped) and resp_stripped[0] in '…—–')

    # 9: empty
    feat[9] = float(not resp_stripped)

    return feat


# ─── Perceptron ───────────────────────────────────────────────────────────────

class PersonaPerceptron:
    """Single-layer perceptron for response coherence. Weights persist to disk."""

    LR = 0.05

    def __init__(self, persona_name: str, heads_dir: Path) -> None:
        self.persona_name = persona_name
        self._path = heads_dir / persona_name / 'coherence_weights.json'
        self._weights, self._bias, self._n_samples = self._load()

    def _load(self) -> tuple[np.ndarray, float, int]:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding='utf-8'))
                w = np.array(data.get('weights', [0.0] * _N_FEATURES), dtype=np.float32)
                b = float(data.get('bias', 0.0))
                n = int(data.get('n_samples', 0))
                if len(w) == _N_FEATURES:
                    return w, b, n
            except Exception:
                pass
        return np.zeros(_N_FEATURES, dtype=np.float32), 0.0, 0

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps({
                    'weights': self._weights.tolist(),
                    'bias': self._bias,
                    'n_samples': self._n_samples,
                    'feature_names': _FEATURE_NAMES,
                    'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                }, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
        except Exception:
            pass

    def predict(self, features: np.ndarray) -> float:
        logit = float(np.dot(self._weights, features) + self._bias)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def train(self, features: np.ndarray, label: float) -> None:
        pred = self.predict(features)
        error = label - pred
        self._weights += self.LR * error * features
        self._bias += self.LR * error
        self._n_samples += 1
        self._save()

    @property
    def n_samples(self) -> int:
        return self._n_samples


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class CoherenceResult:
    score: float          # 0.0 (very bad) to 1.0 (good)
    veto: bool            # hard rule triggered
    reason: str           # human-readable reason for low score / veto
    features: dict[str, float] = field(default_factory=dict)
    n_training_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─── Main checker ─────────────────────────────────────────────────────────────

class ResponseCoherenceChecker:
    """
    Usage:
        checker = ResponseCoherenceChecker(heads_dir)
        result = checker.score(persona_name, user_input, response)
        if result.veto:
            # block or retry
        checker.record_correction(persona_name, user_input, original, corrected)
    """

    # Score threshold below which we flag the response
    VETO_SCORE = 0.35

    def __init__(self, heads_dir: Path | str) -> None:
        self._heads_dir = Path(heads_dir)
        self._perceptrons: dict[str, PersonaPerceptron] = {}

    def _get_perceptron(self, persona_name: str) -> PersonaPerceptron:
        if persona_name not in self._perceptrons:
            self._perceptrons[persona_name] = PersonaPerceptron(persona_name, self._heads_dir)
        return self._perceptrons[persona_name]

    def score(self, persona_name: str, user_input: str, response: str) -> CoherenceResult:
        feat = extract_features(user_input, response)
        feat_dict = {name: float(v) for name, v in zip(_FEATURE_NAMES, feat)}

        # Hard veto rules (deterministic, immediate)
        veto, veto_reason = self._check_veto_rules(feat, response)
        if veto:
            return CoherenceResult(
                score=0.0,
                veto=True,
                reason=veto_reason,
                features=feat_dict,
            )

        # Perceptron score (if trained)
        perc = self._get_perceptron(persona_name)
        if perc.n_samples >= 4:
            perc_score = perc.predict(feat)
        else:
            # Not enough training data — use rule-based heuristic
            bad_signals = float(np.sum(feat[:8]))  # first 8 are "bad" features
            perc_score = max(0.0, 1.0 - bad_signals * 0.3)

        veto = perc_score < self.VETO_SCORE
        reason = '' if not veto else f'low coherence score {perc_score:.2f}'
        return CoherenceResult(
            score=perc_score,
            veto=veto,
            reason=reason,
            features=feat_dict,
            n_training_samples=perc.n_samples,
        )

    def record_correction(
        self,
        persona_name: str,
        user_input: str,
        original_reply: str,
        corrected_reply: str,
    ) -> None:
        """Train: original=negative(0), corrected=positive(1)."""
        if not persona_name:
            return
        perc = self._get_perceptron(persona_name)
        if original_reply and original_reply.strip() != corrected_reply.strip():
            bad_feat = extract_features(user_input, original_reply)
            perc.train(bad_feat, 0.0)
        if corrected_reply:
            good_feat = extract_features(user_input, corrected_reply)
            perc.train(good_feat, 1.0)

    def _check_veto_rules(self, feat: np.ndarray, response: str) -> tuple[bool, str]:
        if feat[9]:  # empty
            return True, 'empty_response'
        if feat[0]:  # analysis opener
            return True, 'analysis_opener_detected'
        if feat[1]:  # hesitation start
            return True, 'hesitation_start_detected'
        if feat[2]:  # language mismatch
            return True, 'language_mismatch'
        if feat[4]:  # markdown analysis headers
            return True, 'markdown_analysis_detected'
        return False, ''


# ─── Module-level singleton (lazy) ───────────────────────────────────────────

_checker: ResponseCoherenceChecker | None = None


def get_coherence_checker(heads_dir: Path | str | None = None) -> ResponseCoherenceChecker:
    global _checker
    if _checker is None:
        if heads_dir is None:
            from agent_system.persona_engine import _head_dir as _get_head_dir  # noqa: PLC0415
            heads_dir = _get_head_dir('').parent
        _checker = ResponseCoherenceChecker(Path(heads_dir))
    return _checker
