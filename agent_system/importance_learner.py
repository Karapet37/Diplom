"""
Importance learner.

Learns what this user considers worth saving.

Stage 1 (current):
    - /save → positive examples
    - Each turn that passes without /save → weak negative evidence

Stage 2 (future):
    - system suggests likely-important moments for user confirmation

Architecture:
    ExampleStore: stores (text, label, timestamp) as JSONL
    Scorer:       simple keyword + length + recency heuristic scorer
    Suggester:    ranks unsaved turns and suggests top candidates

The system must NOT autosave everything.
Start with suggestion mode, not blind autosave.

What this user considers important:
    - personal facts (biography, relationships)
    - stated goals or intentions
    - emotional turning points
    - failures and successes they reported
    - commitments they made
    - things they explicitly asked to remember

What is probably noise:
    - short acknowledgments ("ok", "thanks", "yes")
    - pure questions with no personal content
    - greetings / farewells
    - /notes, /plan, /save commands themselves

Suggestion threshold:
    Score >= SUGGESTION_THRESHOLD → suggest to user
    Score < SUGGESTION_THRESHOLD → treat as noise
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .runtime_config import get_runtime_config

_STORE_LOCK = Lock()

SUGGESTION_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _learner_dir() -> Path:
    return get_runtime_config().paths.memory_root / 'importance_learner'


def _examples_path(session_id: str) -> Path:
    safe = re.sub(r'[^\w\-]', '_', str(session_id or 'default').strip())
    return _learner_dir() / f'{safe}_examples.jsonl'


def _global_examples_path() -> Path:
    return _learner_dir() / 'global_examples.jsonl'


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


# ---------------------------------------------------------------------------
# Example store
# ---------------------------------------------------------------------------

def _load_examples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    examples: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                examples.append(obj)
        except json.JSONDecodeError:
            continue
    return examples


def _append_example(path: Path, example: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(example, ensure_ascii=False) + '\n')


def record_positive_example(session_id: str, text: str) -> None:
    """Record a turn that the user explicitly saved (positive importance signal)."""
    text = str(text or '').strip()
    if not text:
        return
    example = {
        'text': text,
        'label': 1,
        'session_id': session_id,
        'timestamp': _utc_now(),
        'source': 'manual_save',
    }
    with _STORE_LOCK:
        _append_example(_examples_path(session_id), example)
        _append_example(_global_examples_path(), example)


def record_negative_example(session_id: str, text: str) -> None:
    """Record a turn that passed without being saved (weak negative signal)."""
    text = str(text or '').strip()
    if not text or len(text) < 20:
        return
    example = {
        'text': text,
        'label': 0,
        'session_id': session_id,
        'timestamp': _utc_now(),
        'source': 'implicit_skip',
    }
    with _STORE_LOCK:
        _append_example(_examples_path(session_id), example)


def load_examples(session_id: str = '', *, global_examples: bool = False) -> list[dict[str, Any]]:
    """Load training examples for a session or globally."""
    with _STORE_LOCK:
        if global_examples:
            return _load_examples(_global_examples_path())
        return _load_examples(_examples_path(session_id))


# ---------------------------------------------------------------------------
# Importance scorer
# ---------------------------------------------------------------------------

# Keywords that signal personal importance
_HIGH_IMPORTANCE_KEYWORDS = (
    # Personal facts — identity and state
    'i am', 'i\'m', 'я', 'my name', 'меня зовут',
    'i work', 'i live', 'i study', 'я работаю', 'я живу', 'я учусь',
    # Personal facts — possession, ability, history (ConvAI2-validated)
    'i have', 'i had', 'i can', 'i could', 'i was', 'i used to', 'i\'ve been',
    'i like', 'i love', 'i enjoy', 'i hate', 'i don\'t like',
    'i go to', 'i went to', 'i grew up', 'i grew',
    'у меня', 'я имею', 'я могу', 'я был', 'я была', 'мне нравится',
    # Goals and intentions
    'i want', 'i need', 'i plan', 'i intend', 'my goal', 'my plan',
    'я хочу', 'мне нужно', 'я планирую', 'моя цель',
    # Relationships
    'my wife', 'my husband', 'my partner', 'my friend', 'my boss', 'my mother', 'my father',
    'my sister', 'my brother', 'my son', 'my daughter', 'my child', 'my children',
    'my family', 'my dog', 'my cat', 'my pet',
    'моя жена', 'мой муж', 'мой партнёр', 'мой друг', 'мой начальник', 'моя мама', 'мой папа',
    # Emotional turning points
    'i realized', 'i discovered', 'i understood', 'changed my mind',
    'я понял', 'я осознал', 'я понял', 'изменил мнение',
    # Commitments
    'i decided', 'i will', 'i commit', 'i promise',
    'я решил', 'я буду', 'я обязуюсь', 'обещаю',
    # Failures and successes
    'i failed', 'i succeeded', 'didn\'t work', 'worked out',
    'я провалился', 'у меня получилось', 'не сработало', 'получилось',
    # Important events
    'important', 'remember', 'note that', 'don\'t forget',
    'важно', 'запомни', 'отметь', 'не забудь',
)

_LOW_IMPORTANCE_KEYWORDS = (
    'ok', 'okay', 'thanks', 'thank you', 'bye', 'hello', 'hi',
    'ок', 'хорошо', 'спасибо', 'привет', 'пока', 'да', 'нет', 'yes', 'no',
    '/notes', '/save', '/plan', '/del_note', '/clear_notes',
)

_FACTUAL_PATTERNS = (
    r'\b(born|рождён|родился)\b',
    r'\b(years old|лет)\b',
    r'\b(live in|живу в)\b',
    r'\b(work at|работаю в)\b',
    r'\b(studied at|учился в)\b',
    r'\b(married|женат|замужем)\b',
    r'\b(divorced|разведён)\b',
    r'\b(have \d+|есть \d+)\b',
)


def score_importance(text: str) -> float:
    """
    Score the likely importance of a piece of text.
    Returns float in [0.0, 1.0].

    0.0 = almost certainly noise
    1.0 = almost certainly important
    """
    text = str(text or '').strip()
    if not text:
        return 0.0

    lowered = text.lower()
    score = 0.0

    # Low importance indicators
    if len(text) < 15:
        return 0.05
    if any(kw in lowered for kw in _LOW_IMPORTANCE_KEYWORDS) and len(text) < 30:
        return 0.05

    # Length bonus (longer = more likely to contain personal content)
    length_score = min(0.2, len(text) / 500)
    score += length_score

    # High importance keyword hits
    hi_hits = sum(1 for kw in _HIGH_IMPORTANCE_KEYWORDS if kw in lowered)
    score += min(0.4, hi_hits * 0.08)

    # Factual pattern hits
    for pattern in _FACTUAL_PATTERNS:
        if re.search(pattern, lowered):
            score += 0.12

    # First-person singular presence (personal content)
    # Use word-boundary aware counting (handles start-of-string "I" and "My")
    import re as _re
    first_person = (
        len(_re.findall(r'\bi\b', lowered))
        + lowered.count('я ')
        + len(_re.findall(r'\bmy\b', lowered))
        + lowered.count('мой ')
        + lowered.count('моя ')
        + lowered.count('моё ')
    )
    score += min(0.2, first_person * 0.04)

    # Penalize pure questions (less likely to be personal fact statements)
    question_markers = ('?', 'what is', 'how do', 'why is', 'что такое', 'как работает')
    if any(m in lowered for m in question_markers):
        score *= 0.7

    return min(1.0, max(0.0, score))


# ---------------------------------------------------------------------------
# Suggestion engine
# ---------------------------------------------------------------------------

def suggest_important_turns(
    turns: list[str],
    *,
    threshold: float = SUGGESTION_THRESHOLD,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """
    Given a list of recent turn texts, suggest the top-N most likely important ones.

    Returns a list of dicts with keys: text, score, rank
    Only returns turns with score >= threshold.
    """
    scored: list[dict[str, Any]] = []
    for text in turns:
        s = score_importance(text)
        if s >= threshold:
            scored.append({'text': text, 'score': round(s, 3)})

    scored.sort(key=lambda x: x['score'], reverse=True)

    result = []
    for i, item in enumerate(scored[:top_n], 1):
        result.append({'text': item['text'], 'score': item['score'], 'rank': i})
    return result


def render_suggestion_message(suggestions: list[dict[str, Any]]) -> str:
    """
    Render a suggestion message for the user.

    The system is suggesting that certain moments may be worth saving.
    """
    if not suggestions:
        return ''
    lines = ['(Suggestion: these moments may be worth saving. Use /save <text> to save any of them.)']
    for s in suggestions:
        lines.append(f'  Score {s["score"]:.2f}: {s["text"][:120]}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Personal importance profile
# ---------------------------------------------------------------------------

def build_importance_profile(session_id: str) -> dict[str, Any]:
    """
    Build a profile of what this user tends to save.

    Returns a dict with:
    - positive_count: how many things they've saved
    - avg_length: average length of saved notes
    - top_keywords: most common keywords in saved texts
    - categories: rough categorization of saved content
    """
    examples = load_examples(session_id)
    positive = [e for e in examples if e.get('label') == 1]

    if not positive:
        return {
            'positive_count': 0,
            'avg_length': 0,
            'top_keywords': [],
            'summary': 'No saved notes yet.',
        }

    texts = [str(e.get('text') or '') for e in positive]
    avg_len = sum(len(t) for t in texts) / len(texts)

    # Count keyword frequency
    keyword_counts: dict[str, int] = {}
    for text in texts:
        lowered = text.lower()
        for kw in _HIGH_IMPORTANCE_KEYWORDS:
            if kw in lowered:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

    top_kw = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        'positive_count': len(positive),
        'avg_length': round(avg_len, 1),
        'top_keywords': [k for k, _ in top_kw],
        'summary': (
            f'{len(positive)} note(s) saved. '
            f'Average length: {round(avg_len)} chars. '
            f'Common themes: {", ".join(k for k, _ in top_kw[:3]) or "not yet determined"}.'
        ),
    }
