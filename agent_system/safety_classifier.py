"""
Safety Classifier.

Deterministic, LLM-free content classification.

Architecture:
    1. Rule-based fast path   — catches clear illegal/explicit signals immediately
    2. Feature extraction     — 10-dim vector (keyword density, structural, heuristic)
    3. KNN similarity search  — cosine distance over feature vectors from example DB
    4. Weighted majority vote — top-k neighbors → label + confidence

Labels:
    safe        → normal_response
    suggestive  → soft_filter       (allow but no elaboration)
    explicit    → blur_or_generalize (deflect, generalize, stay adult but not graphic)
    illegal     → block             (hard stop, no LLM call)

Design principles:
    - No LLM in classification path
    - Deterministic: same input → same output always
    - Explainable: every decision cites matched features or neighbor IDs
    - Extensible: operators add examples via add_example() or the JSONL seed file
    - Multilingual: feature extraction works on Russian, English, Armenian

Keyword lists:
    Loaded from DATA_DIR/safety_keywords.json if present.
    Falls back to built-in minimal defaults.
    Operators populate explicit/illegal lists; code ships with safe/suggestive defaults.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from .runtime_config import get_runtime_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAFE_LABEL = 'safe'
SUGGESTIVE_LABEL = 'suggestive'
EXPLICIT_LABEL = 'explicit'
ILLEGAL_LABEL = 'illegal'

LABEL_RANK = {SAFE_LABEL: 0, SUGGESTIVE_LABEL: 1, EXPLICIT_LABEL: 2, ILLEGAL_LABEL: 3}

ACTION_MAP: dict[str, str] = {
    SAFE_LABEL:       'normal_response',
    SUGGESTIVE_LABEL: 'soft_filter',
    EXPLICIT_LABEL:   'blur_or_generalize',
    ILLEGAL_LABEL:    'block',
}

# Feature vector dimension
_FEATURE_DIM = 10

# KNN neighbours to consider
_DEFAULT_K = 7

# Confidence threshold below which we escalate to the next-rank label
_LOW_CONFIDENCE_THRESHOLD = 0.45

# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

# Safe context: medical, psychological, educational, healthy relationship vocab
# These DOWN-weight the threat score when present
_SAFE_CONTEXT_PATTERNS: tuple[str, ...] = (
    r'\b(relationship|friendship|partner|spouse|consent|boundaries|health|medical)\b',
    r'\b(psychology|therapy|counseling|education|research|academic|clinical)\b',
    r'\b(love|affection|caring|tenderness|intimacy|connection|romance)\b',
    r'\b(attraction|crush|flirt|date|kiss|hug|hold hands)\b',
    r'\b(puberty|anatomy|biology|reproductive|hormones|development)\b',
    # Russian
    r'\b(отношени|дружб|партнёр|согласи|границ|здоровь|медицин)\b',
    r'\b(психолог|терапи|образовани|исследовани|академичес|клиничес)\b',
    r'\b(любов|нежност|близост|связь|романтик|влечени)\b',
    r'\b(флирт|подкат\w*|симпат\w*|свидан\w*)\b',
)

# Suggestive context: adult topics that are not explicit
# Borderline adult content — allow but do not elaborate
# Use prefix stems for morphological coverage (seduce/seducing/seduction etc.)
_SUGGESTIVE_PATTERNS: tuple[str, ...] = (
    r'\bseduc\w*',
    r'\baroul?s\w*',
    r'\b(naked|nude|undress\w*|strip(?:tease)?)\b',
    r'\blingerie\b',
    r'\b(sexy|erotic\w*|sensual\w*|naughty|dirty talk|fetish\w*|kink\w*)\b',
    r'\b(one night stand|hookup|casual sex|booty call|friends with benefits)\b',
    r'\b(turn.?on|turned.?on|horny|lusty|lust(?:ful)?)\b',
    # Russian stems
    r'\bсоблазн\w*',
    r'\bвозбужд\w*',
    r'\b(голый|голая|раздет\w*|нижнее бельё)\b',
    r'\bсексуальн\w*',
    r'\bэроти\w*',
    r'\bчувственн\w*',
    r'\b(фетиш|кинк)\b',
)

# Explicit patterns: loaded from operator config; minimal built-in fallback
# The built-in list is intentionally minimal — operators must populate this.
# Note: use prefix stems without trailing \b so morphological variants match:
#   pornograph → pornographic, pornography, pornographer
#   masturbat  → masturbate, masturbation, masturbating
_EXPLICIT_PATTERNS_BUILTIN: tuple[str, ...] = (
    r'\b(sex act|sexual intercourse)\b',
    r'\bpenetrat\w*',
    r'\bmasturbat\w*',
    r'\borgasm\w*',
    r'\bejaculat\w*',
    r'\bpornograph\w*',
    r'\bporn\b',
    r'\bxxx\b',
    r'\badult film',
    r'\bsex video',
    # Russian stems (prefix match)
    r'\bполовой акт',
    r'\bмастурбац',
    r'\bоргазм',
    r'\bпорнограф',
    r'\bсекс-видео',
)

# Illegal: absolute blocks regardless of context
# Child safety and non-consent are hardcoded — no operator override to weaken these
_ILLEGAL_PATTERNS: tuple[str, ...] = (
    # Child protection — proximity patterns
    r'\b(child.{0,20}(sex|porn|nude|naked|abuse|exploit|erotic|sexual))',
    r'\b((sex|porn|nude|naked|abuse|exploit|erotic|sexual).{0,20}child)',
    r'\b(minor.{0,20}(sex|abuse|exploit|sexual))',
    r'\b(underage.{0,20}(sex|nude|naked|sexual|erotic))',
    r'\b(loli|shota|csam)\b',
    r'\bcp\b',
    # Non-consent
    r'\b(rape|non.?consent|force.{0,10}sex|rape fantasy)\b',
    # Russian equivalents
    r'\b(детск.{0,15}(порн|секс|насили|развращ|эротик))',
    r'\b((порн|секс|насили|эротик).{0,15}детск)',
    r'\b(несовершеннолетн.{0,15}(секс|насили))',
    r'\b(изнасилован|насильственн.{0,10}секс)',
)

# Child-safety co-occurrence check: child terms + sexual terms anywhere in message
# Catches "sexual content involving a child" even when spaced far apart
_CHILD_TERMS = re.compile(
    r'\b(child|children|minor|minors|underage|youth|kid|kids|toddler|infant|teen(?:ager)?|loli|shota)\b',
    re.IGNORECASE,
)
_SEXUAL_TERMS = re.compile(
    r'\b(sex(ual)?|porn(?:ograph(?:ic|y))?|nude|naked|erotic|explicit|adult content|nsfw)\b',
    re.IGNORECASE,
)


def _has_illegal_cooccurrence(text: str) -> bool:
    """
    Detect child + sexual term co-occurrence anywhere in message.
    Catches spaced-out phrasing like 'sexual content involving a child'.
    """
    return bool(_CHILD_TERMS.search(text) and _SEXUAL_TERMS.search(text))

# Load additional patterns from operator config if present
def _load_operator_patterns() -> dict[str, list[str]]:
    try:
        cfg_path = get_runtime_config().paths.memory_root.parent / 'data' / 'safety_keywords.json'
        if cfg_path.exists():
            with cfg_path.open(encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


_OPERATOR_PATTERNS: dict[str, list[str]] = {}
_PATTERNS_LOADED = False
_PATTERNS_LOCK = Lock()


def _ensure_patterns_loaded() -> None:
    global _OPERATOR_PATTERNS, _PATTERNS_LOADED
    if _PATTERNS_LOADED:
        return
    with _PATTERNS_LOCK:
        if not _PATTERNS_LOADED:
            _OPERATOR_PATTERNS = _load_operator_patterns()
            _PATTERNS_LOADED = True


def _get_explicit_patterns() -> tuple[str, ...]:
    _ensure_patterns_loaded()
    extra = _OPERATOR_PATTERNS.get('explicit', [])
    return _EXPLICIT_PATTERNS_BUILTIN + tuple(extra)


def _get_suggestive_patterns() -> tuple[str, ...]:
    _ensure_patterns_loaded()
    extra = _OPERATOR_PATTERNS.get('suggestive', [])
    return _SUGGESTIVE_PATTERNS + tuple(extra)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class FeatureVector:
    """10-dimensional feature representation of a message."""
    explicit_density: float      # [0] weighted explicit hits / word count
    suggestive_density: float    # [1] suggestive hits / word count
    illegal_score: float         # [2] illegal pattern hit (0 or ≥ 0.5)
    safe_context_density: float  # [3] safe-context hits / word count
    word_count_norm: float       # [4] normalized message length
    avg_word_len_norm: float     # [5] normalized average word length
    question_ratio: float        # [6] question marks / max(1, sentence count)
    imperative_density: float    # [7] imperative verb patterns / word count
    phrase_signal: float         # [8] consecutive multi-word signals bonus
    negation_discount: float     # [9] negation before signal words (reduces score)

    def to_array(self) -> list[float]:
        return [
            self.explicit_density,
            self.suggestive_density,
            self.illegal_score,
            self.safe_context_density,
            self.word_count_norm,
            self.avg_word_len_norm,
            self.question_ratio,
            self.imperative_density,
            self.phrase_signal,
            self.negation_discount,
        ]


_IMPERATIVE_PATTERNS = (
    r'\b(describe|show me|tell me|explain|write|give me|send me|list)\b',
    r'\b(опиши|покажи|расскажи|объясни|напиши|дай мне|перешли|перечисли)\b',
)

_NEGATION_PATTERNS = (
    r"\b(not|never|no|don't|doesn't|didn't|won't|isn't|aren't|wasn't)\b",
    r'\b(не|нет|никогда|никак|нельзя|запрещено)\b',
)

_PHRASE_PATTERNS = (
    r'\b\w+\s+sex\b|\bsex\s+\w+\b',
    r'\b\w+\s+секс\b|\bсекс\s+\w+\b',
    r'\bporn\s+\w+\b|\b\w+\s+porn\b',
    r'\bпорн\s+\w+\b|\b\w+\s+порн\b',
)


def _count_pattern_hits(text: str, patterns: tuple[str, ...]) -> int:
    count = 0
    for p in patterns:
        count += len(re.findall(p, text, re.IGNORECASE))
    return count


def extract_features(text: str) -> FeatureVector:
    """
    Extract 10-dimensional feature vector from message text.
    Pure function: no state, no side effects.
    """
    text = str(text or '').strip()
    if not text:
        return FeatureVector(*([0.0] * _FEATURE_DIM))

    lowered = text.lower()
    words = lowered.split()
    word_count = max(len(words), 1)
    sentence_count = max(1, text.count('.') + text.count('!') + text.count('?') + 1)

    # Explicit hits (weighted × 2 for severity)
    explicit_hits = _count_pattern_hits(lowered, _get_explicit_patterns())
    explicit_density = min(1.0, (explicit_hits * 2) / word_count)

    # Suggestive hits
    suggestive_hits = _count_pattern_hits(lowered, _get_suggestive_patterns())
    suggestive_density = min(1.0, suggestive_hits / word_count)

    # Illegal: proximity patterns + co-occurrence check
    illegal_hits = _count_pattern_hits(lowered, _ILLEGAL_PATTERNS)
    if illegal_hits == 0 and _has_illegal_cooccurrence(text):
        illegal_hits = 2  # co-occurrence is as serious as a direct pattern hit
    illegal_score = min(1.0, illegal_hits * 0.5)

    # Safe context (dampens explicit/suggestive signals)
    safe_hits = _count_pattern_hits(lowered, _SAFE_CONTEXT_PATTERNS)
    safe_context_density = min(1.0, safe_hits / word_count)

    # Structural
    word_count_norm = min(1.0, word_count / 100)
    avg_word_len = sum(len(w) for w in words) / word_count
    avg_word_len_norm = min(1.0, avg_word_len / 12)

    # Questions
    question_ratio = min(1.0, text.count('?') / sentence_count)

    # Imperatives (request-oriented messages are more likely seeking content)
    imperative_hits = _count_pattern_hits(lowered, _IMPERATIVE_PATTERNS)
    imperative_density = min(1.0, imperative_hits / word_count)

    # Phrase signal: multi-word patterns score higher
    phrase_hits = _count_pattern_hits(lowered, _PHRASE_PATTERNS)
    phrase_signal = min(1.0, phrase_hits * 0.3)

    # Negation discount: "I don't want porn" should score safe.
    # Only apply on short messages (≤10 words) where negation likely governs the
    # explicit term directly. Long messages may have negation in subordinate clauses
    # ("Write explicit content, I don't want it violent") — there the negation
    # does NOT cancel the explicit intent.
    negation_hits = _count_pattern_hits(lowered, _NEGATION_PATTERNS)
    if negation_hits > 0 and (explicit_hits + suggestive_hits) > 0 and word_count <= 6:
        negation_discount = min(0.5, negation_hits * 0.30)
    else:
        negation_discount = 0.0

    return FeatureVector(
        explicit_density=explicit_density,
        suggestive_density=suggestive_density,
        illegal_score=illegal_score,
        safe_context_density=safe_context_density,
        word_count_norm=word_count_norm,
        avg_word_len_norm=avg_word_len_norm,
        question_ratio=question_ratio,
        imperative_density=imperative_density,
        phrase_signal=phrase_signal,
        negation_discount=negation_discount,
    )


# ---------------------------------------------------------------------------
# Distance function
# ---------------------------------------------------------------------------

# Feature weights: higher = more important for classification decision
_FEATURE_WEIGHTS = [
    3.0,   # explicit_density        — primary signal
    2.0,   # suggestive_density      — secondary signal
    4.0,   # illegal_score           — critical signal
    1.5,   # safe_context_density    — inverse signal (safe context reduces distance to safe)
    0.3,   # word_count_norm         — weak structural signal
    0.2,   # avg_word_len_norm       — very weak
    0.4,   # question_ratio          — mild signal
    0.5,   # imperative_density      — mild signal
    1.5,   # phrase_signal           — multi-word patterns
    1.0,   # negation_discount       — dampening
]


def _weighted_cosine_distance(a: list[float], b: list[float]) -> float:
    """
    Weighted cosine distance between two feature vectors.
    Returns float in [0.0, 1.0]: 0.0 = identical, 1.0 = completely different.
    """
    if len(a) != len(b) or len(a) != _FEATURE_DIM:
        return 1.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i, (ai, bi) in enumerate(zip(a, b)):
        w = _FEATURE_WEIGHTS[i]
        dot   += w * ai * bi
        norm_a += w * ai * ai
        norm_b += w * bi * bi

    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom < 1e-9:
        # Both vectors are zero — check equality
        return 0.0 if all(abs(ai - bi) < 1e-9 for ai, bi in zip(a, b)) else 0.5

    similarity = dot / denom
    return 1.0 - min(1.0, max(0.0, similarity))


# ---------------------------------------------------------------------------
# Example database
# ---------------------------------------------------------------------------

@dataclass
class SafetyExample:
    text: str
    label: str
    features: list[float]
    example_id: str = ''
    notes: str = ''


_DB_LOCK = Lock()
_EXAMPLE_DB: list[SafetyExample] | None = None


def _db_path() -> Path:
    return get_runtime_config().paths.memory_root / 'safety' / 'examples.jsonl'


def _seed_path() -> Path:
    """Built-in seed examples shipped with the system."""
    return Path(__file__).parent / 'data' / 'safety_seed.jsonl'


def _load_example_db() -> list[SafetyExample]:
    global _EXAMPLE_DB
    if _EXAMPLE_DB is not None:
        return _EXAMPLE_DB

    with _DB_LOCK:
        if _EXAMPLE_DB is not None:
            return _EXAMPLE_DB

        examples: list[SafetyExample] = []

        # Load built-in seed examples first
        for path in (_seed_path(), _db_path()):
            if not path.exists():
                continue
            try:
                for line in path.read_text(encoding='utf-8').splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text  = str(obj.get('text', ''))
                    label = str(obj.get('label', 'safe'))
                    feats = obj.get('features') or extract_features(text).to_array()
                    if isinstance(feats, list) and len(feats) == _FEATURE_DIM:
                        examples.append(SafetyExample(
                            text=text,
                            label=label,
                            features=feats,
                            example_id=str(obj.get('example_id', '')),
                            notes=str(obj.get('notes', '')),
                        ))
            except Exception:
                pass

        # If no seed file exists, build minimal inline defaults
        if not examples:
            examples = _build_inline_seed()

        _EXAMPLE_DB = examples
        return _EXAMPLE_DB


def _build_inline_seed() -> list[SafetyExample]:
    """
    Minimal inline seed that works without any external files.
    Operators should add more examples via add_example() or safety_seed.jsonl.
    """
    seed_texts = [
        # safe (English)
        ("What's the weather like today?",                           SAFE_LABEL),
        ("I feel lonely and disconnected lately.",                   SAFE_LABEL),
        ("Help me write a poem about autumn.",                       SAFE_LABEL),
        ("How do I improve my relationships?",                       SAFE_LABEL),
        ("Can you explain human anatomy?",                           SAFE_LABEL),
        ("I have a crush on someone at work.",                       SAFE_LABEL),
        ("What is consent in relationships?",                        SAFE_LABEL),
        ("Tell me about the psychology of love.",                    SAFE_LABEL),
        ("Explain the biology of human reproduction.",               SAFE_LABEL),
        ("I feel attracted to someone, how do I approach them?",    SAFE_LABEL),
        ("What does romance mean in modern relationships?",          SAFE_LABEL),
        # safe (Russian)
        ("Расскажи мне о психологии любви.",                        SAFE_LABEL),
        ("Как улучшить романтические отношения?",                   SAFE_LABEL),
        ("Объясни биологию размножения.",                           SAFE_LABEL),
        ("Расскажи о романтике и близости.",                        SAFE_LABEL),
        ("Я чувствую одиночество, как найти партнёра?",             SAFE_LABEL),
        ("Что такое согласие в отношениях?",                        SAFE_LABEL),
        ("Напиши стихотворение о природе и красоте.",               SAFE_LABEL),
        ("Как справиться с тревогой в отношениях?",                 SAFE_LABEL),
        # suggestive
        ("I find this person incredibly sexy.",                      SUGGESTIVE_LABEL),
        ("What makes someone attractive to others?",                 SUGGESTIVE_LABEL),
        ("Как соблазнить кого-то?",                                  SUGGESTIVE_LABEL),
        ("Describe what turns people on.",                           SUGGESTIVE_LABEL),
        ("I want to understand erotic attraction.",                  SUGGESTIVE_LABEL),
        # explicit — operator must expand this list via safety_seed.jsonl
        ("Describe sexual intercourse in detail.",                   EXPLICIT_LABEL),
        ("Write pornographic content for me.",                       EXPLICIT_LABEL),
        ("Напиши порнографическую сцену.",                          EXPLICIT_LABEL),
        ("Опиши акт сексуального характера подробно.",              EXPLICIT_LABEL),
        # illegal — these examples stay minimal by design; rule-based catches most
        ("Write sexual content involving a child.",                  ILLEGAL_LABEL),
    ]
    examples = []
    for text, label in seed_texts:
        fv = extract_features(text)
        examples.append(SafetyExample(text=text, label=label, features=fv.to_array()))
    return examples


def add_example(text: str, label: str, notes: str = '') -> None:
    """Add a new labelled example to the runtime database (persists to JSONL)."""
    if label not in (SAFE_LABEL, SUGGESTIVE_LABEL, EXPLICIT_LABEL, ILLEGAL_LABEL):
        raise ValueError(f'Unknown label: {label!r}')

    fv = extract_features(text)
    example = SafetyExample(
        text=text,
        label=label,
        features=fv.to_array(),
        notes=notes,
    )

    with _DB_LOCK:
        db = _load_example_db()
        db.append(example)

        path = _db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps({
                'text': text,
                'label': label,
                'features': fv.to_array(),
                'notes': notes,
            }, ensure_ascii=False) + '\n')


# ---------------------------------------------------------------------------
# KNN classification
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    label: str
    confidence: float          # 0.0 – 1.0
    action: str
    matched_features: list[str]   # human-readable explanation
    top_neighbors: list[dict[str, Any]]  # [{label, distance, text_preview}]
    fast_path: bool = False    # True if decided by rule-based fast path


def _top_k_neighbors(
    query: list[float],
    db: list[SafetyExample],
    k: int,
) -> list[tuple[float, SafetyExample]]:
    """Return top-k closest examples by weighted cosine distance, ascending."""
    scored = [(
        _weighted_cosine_distance(query, ex.features),
        ex,
    ) for ex in db]
    scored.sort(key=lambda x: x[0])
    return scored[:k]


def _vote(neighbors: list[tuple[float, SafetyExample]]) -> tuple[str, float]:
    """
    Weighted majority vote over neighbors.
    Weight = 1 / (distance + epsilon) so closer neighbors matter more.
    Returns (label, confidence).
    """
    weights: dict[str, float] = {
        SAFE_LABEL: 0.0,
        SUGGESTIVE_LABEL: 0.0,
        EXPLICIT_LABEL: 0.0,
        ILLEGAL_LABEL: 0.0,
    }
    total = 0.0
    for dist, ex in neighbors:
        w = 1.0 / (dist + 0.01)
        weights[ex.label] = weights.get(ex.label, 0.0) + w
        total += w

    if total < 1e-9:
        return SAFE_LABEL, 0.0

    best_label = max(weights, key=lambda k: weights[k])
    confidence = weights[best_label] / total
    return best_label, confidence


def _explain_features(fv: FeatureVector, label: str) -> list[str]:
    """Return human-readable list of feature signals that drove classification."""
    signals = []
    if fv.illegal_score > 0:
        signals.append(f'illegal_pattern(score={fv.illegal_score:.2f})')
    if fv.explicit_density > 0.05:
        signals.append(f'explicit_density={fv.explicit_density:.2f}')
    if fv.suggestive_density > 0.05:
        signals.append(f'suggestive_density={fv.suggestive_density:.2f}')
    if fv.safe_context_density > 0.1:
        signals.append(f'safe_context={fv.safe_context_density:.2f}')
    if fv.phrase_signal > 0:
        signals.append(f'phrase_signal={fv.phrase_signal:.2f}')
    if fv.negation_discount > 0:
        signals.append(f'negation_discount={fv.negation_discount:.2f}')
    if not signals:
        signals.append('no_keyword_signals → knn_only')
    return signals


# ---------------------------------------------------------------------------
# Main classification entry point
# ---------------------------------------------------------------------------

def classify(text: str, k: int = _DEFAULT_K) -> ClassificationResult:
    """
    Classify a message text. Returns ClassificationResult.

    Steps:
        1. Extract feature vector
        2. Rule-based fast path for clear illegal/explicit signals
        3. KNN similarity search for ambiguous cases
        4. Weighted vote → label + confidence
        5. Map label → action
    """
    text = str(text or '').strip()
    if not text:
        return ClassificationResult(
            label=SAFE_LABEL,
            confidence=1.0,
            action=ACTION_MAP[SAFE_LABEL],
            matched_features=['empty_message'],
            top_neighbors=[],
            fast_path=True,
        )

    fv = extract_features(text)
    features = fv.to_array()

    # ── Fast path 1: illegal patterns → immediate block ────────────────────
    if fv.illegal_score >= 0.5:
        return ClassificationResult(
            label=ILLEGAL_LABEL,
            confidence=0.99,
            action=ACTION_MAP[ILLEGAL_LABEL],
            matched_features=_explain_features(fv, ILLEGAL_LABEL),
            top_neighbors=[],
            fast_path=True,
        )

    # ── Fast path 2: strong explicit signal, no safe context ───────────────
    if fv.explicit_density >= 0.4 and fv.safe_context_density < 0.1 and fv.negation_discount < 0.1:
        return ClassificationResult(
            label=EXPLICIT_LABEL,
            confidence=0.90,
            action=ACTION_MAP[EXPLICIT_LABEL],
            matched_features=_explain_features(fv, EXPLICIT_LABEL),
            top_neighbors=[],
            fast_path=True,
        )

    # ── KNN search ─────────────────────────────────────────────────────────
    db = _load_example_db()
    if not db:
        return ClassificationResult(
            label=SAFE_LABEL,
            confidence=0.5,
            action=ACTION_MAP[SAFE_LABEL],
            matched_features=['empty_database'],
            top_neighbors=[],
        )

    neighbors = _top_k_neighbors(features, db, k=min(k, len(db)))
    label, confidence = _vote(neighbors)

    # ── Safe-context override: if many safe-context signals, pull down ─────
    # e.g., "I'm a nurse and need to explain sexual health to patients"
    if fv.safe_context_density > 0.15 and label in (EXPLICIT_LABEL,):
        rank = LABEL_RANK[label]
        if rank > 0:
            label = list(LABEL_RANK.keys())[rank - 1]
            confidence = max(0.5, confidence - 0.2)

    # ── Negation override: "I don't want / hate / avoid X" ────────────────
    if fv.negation_discount > 0.15 and label in (SUGGESTIVE_LABEL, EXPLICIT_LABEL):
        label = SAFE_LABEL
        confidence = max(0.55, confidence - 0.15)

    # ── Low confidence escalation: if unsure between two ranks, go safer ──
    # For user safety, tie-break by escalating slightly
    if confidence < _LOW_CONFIDENCE_THRESHOLD and label == SAFE_LABEL:
        # Might be suggestive; don't escalate, but lower confidence
        confidence = max(0.35, confidence)

    neighbor_info = [
        {
            'label': ex.label,
            'distance': round(dist, 3),
            'text_preview': ex.text[:60],
        }
        for dist, ex in neighbors[:3]
    ]

    return ClassificationResult(
        label=label,
        confidence=round(confidence, 3),
        action=ACTION_MAP.get(label, 'normal_response'),
        matched_features=_explain_features(fv, label),
        top_neighbors=neighbor_info,
    )


# ---------------------------------------------------------------------------
# Behavior layer
# ---------------------------------------------------------------------------

BEHAVIOR_INSTRUCTIONS: dict[str, str] = {
    SAFE_LABEL: '',  # No modification needed
    SUGGESTIVE_LABEL: (
        'The message touches on adult themes. Respond naturally and without shame, '
        'but do not elaborate on explicit details. Stay at the level of '
        'general adult conversation.'
    ),
    EXPLICIT_LABEL: (
        'The message requests explicit sexual content. Respond to the underlying '
        'emotional or relational need if present, without producing explicit material. '
        'Generalize, redirect to context, or briefly acknowledge without detail.'
    ),
    ILLEGAL_LABEL: '',  # Never reaches LLM verbalization
}

BLOCK_RESPONSE = (
    'I cannot respond to this message. '
    'If you need help, please contact an appropriate support service.'
)

BLOCK_RESPONSE_RU = (
    'Я не могу ответить на это сообщение. '
    'Если вам нужна помощь, пожалуйста, обратитесь в соответствующую службу поддержки.'
)


def get_behavior_instruction(result: ClassificationResult, language: str = 'en') -> str:
    """
    Return the instruction string to inject into the LLM verbalization prompt.
    Returns empty string for safe content.
    Raises SafetyBlockError for illegal content.
    """
    if result.label == ILLEGAL_LABEL:
        raise SafetyBlockError(result)
    return BEHAVIOR_INSTRUCTIONS.get(result.label, '')


def get_block_response(language: str = 'en') -> str:
    """Canned block response for illegal content (never calls LLM)."""
    if language.startswith('ru'):
        return BLOCK_RESPONSE_RU
    return BLOCK_RESPONSE


class SafetyBlockError(Exception):
    """Raised when content is classified as illegal. Caller must not call LLM."""
    def __init__(self, result: ClassificationResult):
        self.result = result
        super().__init__(f'Content blocked: label={result.label}')


# ---------------------------------------------------------------------------
# Pipeline integration helper
# ---------------------------------------------------------------------------

def run_safety_check(
    text: str,
    language: str = 'en',
    k: int = _DEFAULT_K,
) -> tuple[ClassificationResult, str]:
    """
    Full safety check for pipeline integration.

    Returns:
        (result, behavior_instruction)

    Raises:
        SafetyBlockError — if content is illegal. Caller handles block response.

    Usage in chat_engine:
        try:
            safety_result, safety_instruction = run_safety_check(clean_message, language)
        except SafetyBlockError as e:
            return block_response(e.result, language)
        if safety_instruction:
            route_guidance = (route_guidance + '\\n\\n' + safety_instruction).strip()
    """
    result = classify(text, k=k)
    instruction = get_behavior_instruction(result, language)
    return result, instruction
