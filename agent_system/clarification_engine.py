"""
Clarification Engine.

Deterministic question selection for the 'ask_clarifying_question' action.

Data source: ClariQ dataset (3941 human-validated clarifying questions)
+ coaching-domain extensions.

Architecture:
    1. score_clarification_need(message) → float 0–1
       Rule-based scoring derived from ClariQ clarification_need label patterns.
       1.0 = definitely needs clarification.
       0.0 = clear intent, no clarification needed.

    2. select_clarifying_question(message, context) → str
       Keyword-overlap ranking over the question bank.
       Deterministic. No LLM.

    3. ClarificationResult — full output for the action layer.

Design principle:
    The LLM must NOT generate clarifying questions.
    The system selects from a validated bank using keyword matching.
    This makes clarification predictable, fast, and testable.

Dataset coverage:
    ClariQ train.tsv patterns show:
        - High need (score 1–2): action verbs ("give", "what", "tell me"),
          ambiguous scope, missing facets.
        - Low need (score 3–4): specific entity queries, "interested in",
          "find information about" patterns.

    The coaching extension adds 60+ questions for emotional/planning domains
    that are not covered by the information-retrieval-focused ClariQ bank.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATASETS_DIR = Path(__file__).parent.parent / 'DataSets' / 'ClariQ'
_QUESTION_BANK_PATH = _DATASETS_DIR / 'question_bank.tsv'
_CLARIQ_TRAIN_PATH = _DATASETS_DIR / 'train.tsv'

# ---------------------------------------------------------------------------
# Stopwords (English + Russian basics)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'it', 'its', 'this', 'that',
    'these', 'those', 'i', 'me', 'my', 'we', 'you', 'your', 'he', 'she',
    'they', 'them', 'their', 'what', 'when', 'where', 'which', 'who',
    'how', 'all', 'some', 'any', 'not', 'no', 'tell', 'about', 'find',
    'get', 'give', 'want', 'like', 'need', 'know', 'make', 'use', 'go',
    'see', 'more', 'than', 'from', 'just', 'also', 'very', 'so',
    # Russian basics
    'я', 'мне', 'мой', 'моя', 'моё', 'ты', 'тебе', 'твой', 'он', 'она',
    'это', 'то', 'как', 'что', 'где', 'когда', 'если', 'не', 'и', 'или',
    'но', 'а', 'в', 'на', 'за', 'по', 'с', 'к', 'от',
})

# ---------------------------------------------------------------------------
# High-clarification-need markers (derived from ClariQ analysis)
# ---------------------------------------------------------------------------

# These patterns appear significantly more often in clarification_need=1,2 rows
_HIGH_NEED_PATTERNS = (
    # Action-oriented vague requests
    r'\bgive me\b', r'\btell me about\b', r'\bwhat are\b', r'\bwhat is\b',
    r'\bhow do i\b', r'\bhow to\b', r'\bfind me\b', r'\bhelp me\b',
    r'\bi need\b', r'\bcan you help\b',
    # Vague scope markers
    r'\bsomething\b', r'\bstuff\b', r'\bthings\b', r'\binfo\b',
    r'\binformation about\b',
    # Coaching-domain: emotional ambiguity markers
    r'\bi feel\b', r'\bi don\'t know\b', r'\bi\'m not sure\b',
    r'\bне знаю\b', r'\bне уверен\b', r'\bне понимаю\b',
    r'\bчто-то\b', r'\bкак-то\b', r'\bнепонятно\b',
    # Goal ambiguity
    r'\bwhat should i\b', r'\bwhat next\b', r'\bwhat do i do\b',
    r'\bi don\'t understand\b', r'\bconfused\b', r'\bstuck\b',
)

# These patterns appear more in clarification_need=3,4 (specific/clear intent)
_LOW_NEED_PATTERNS = (
    r'\bi\'m interested in\b', r'\bi want to learn\b', r'\bi want to find\b',
    r'\bi decided\b', r'\bi will\b', r'\bmy goal is\b', r'\bi plan to\b',
    r'\bspecifically\b', r'\bin particular\b', r'\bexactly\b',
    r'\bя решил\b', r'\bмоя цель\b', r'\bя планирую\b', r'\bконкретно\b',
)

# ---------------------------------------------------------------------------
# Coaching-specific question bank extension
# Not in ClariQ (designed for information retrieval, not coaching).
# These cover the emotional/planning domain of this system.
# ---------------------------------------------------------------------------

_COACHING_QUESTIONS: list[dict[str, Any]] = [
    # Emotional state clarification
    {'question_id': 'CQ001', 'question': 'Are you describing how you feel right now, or reflecting on something that happened before?', 'domain': 'emotion', 'keywords': ['feel', 'feeling', 'felt', 'чувствую', 'чувствую']},
    {'question_id': 'CQ002', 'question': 'Is this something that has been weighing on you for a long time, or did it come up recently?', 'domain': 'emotion', 'keywords': ['long', 'time', 'recently', 'давно', 'недавно']},
    {'question_id': 'CQ003', 'question': 'When you say you feel stuck — is it more about not knowing what to do, or knowing but not being able to start?', 'domain': 'emotion', 'keywords': ['stuck', 'blocked', 'зашёл', 'застрял', 'не могу']},
    {'question_id': 'CQ004', 'question': 'Are you looking for a practical next step, or do you need to talk through what is happening first?', 'domain': 'emotion', 'keywords': ['step', 'talk', 'шаг', 'поговорить', 'обсудить']},
    {'question_id': 'CQ005', 'question': 'Is there a specific situation that brought this up, or is it more of a general feeling?', 'domain': 'emotion', 'keywords': ['specific', 'situation', 'general', 'конкретная', 'ситуация', 'общее']},
    # Goal clarification
    {'question_id': 'CQ010', 'question': 'What would a good outcome look like to you?', 'domain': 'goal', 'keywords': ['outcome', 'goal', 'result', 'цель', 'результат', 'итог']},
    {'question_id': 'CQ011', 'question': 'Are you trying to solve a specific problem, or are you exploring what you want more broadly?', 'domain': 'goal', 'keywords': ['problem', 'exploring', 'want', 'проблема', 'исследую', 'хочу']},
    {'question_id': 'CQ012', 'question': 'Is this goal something you chose yourself, or is it something you feel you should want?', 'domain': 'goal', 'keywords': ['goal', 'chose', 'should', 'цель', 'должен', 'хочу']},
    {'question_id': 'CQ013', 'question': 'When you imagine succeeding at this — what is different about your life?', 'domain': 'goal', 'keywords': ['succeed', 'success', 'life', 'different', 'успех', 'жизнь', 'изменится']},
    {'question_id': 'CQ014', 'question': 'Are you looking for a short-term fix or a lasting change?', 'domain': 'goal', 'keywords': ['short', 'lasting', 'change', 'краткосрочное', 'долгосрочное', 'изменение']},
    # Planning clarification
    {'question_id': 'CQ020', 'question': 'Do you have a sense of what is blocking you right now?', 'domain': 'planning', 'keywords': ['blocking', 'obstacle', 'barrier', 'блокирует', 'мешает', 'препятствие']},
    {'question_id': 'CQ021', 'question': 'Have you tried anything so far, and if so, what happened?', 'domain': 'planning', 'keywords': ['tried', 'attempt', 'before', 'пробовал', 'пытался', 'раньше']},
    {'question_id': 'CQ022', 'question': 'Is there a deadline or timeline you are working with?', 'domain': 'planning', 'keywords': ['deadline', 'timeline', 'when', 'срок', 'дедлайн', 'когда']},
    {'question_id': 'CQ023', 'question': 'What resources do you have available right now — time, energy, support?', 'domain': 'planning', 'keywords': ['resources', 'time', 'energy', 'support', 'ресурсы', 'время', 'энергия', 'поддержка']},
    {'question_id': 'CQ024', 'question': 'What is the smallest version of this step that would still feel meaningful?', 'domain': 'planning', 'keywords': ['small', 'step', 'meaningful', 'маленький', 'шаг', 'значимый']},
    # Feedback clarification
    {'question_id': 'CQ030', 'question': 'When you say it did not work — what exactly went wrong?', 'domain': 'feedback', 'keywords': ['wrong', 'failed', 'work', 'не получилось', 'не сработало', 'провалилось']},
    {'question_id': 'CQ031', 'question': 'Did anything about the last step feel right, even if the outcome was not what you hoped?', 'domain': 'feedback', 'keywords': ['right', 'outcome', 'hoped', 'почти', 'получилось', 'немного']},
    {'question_id': 'CQ032', 'question': 'How are you feeling about trying again — ready, hesitant, or somewhere in between?', 'domain': 'feedback', 'keywords': ['try', 'again', 'ready', 'снова', 'попробовать', 'готов']},
    # Constraint clarification
    {'question_id': 'CQ040', 'question': 'Are there things that are completely off the table for you — things you cannot or will not do?', 'domain': 'constraint', 'keywords': ['cannot', 'will not', 'off limits', 'нельзя', 'не могу', 'запрещено']},
    {'question_id': 'CQ041', 'question': 'Is there anyone else whose opinion matters to you in this decision?', 'domain': 'constraint', 'keywords': ['opinion', 'others', 'decision', 'мнение', 'другие', 'решение']},
    {'question_id': 'CQ042', 'question': 'What are you most afraid of getting wrong here?', 'domain': 'constraint', 'keywords': ['afraid', 'wrong', 'fear', 'боишься', 'страх', 'ошибка']},
    # Situation context
    {'question_id': 'CQ050', 'question': 'Can you tell me more about the specific situation you are in right now?', 'domain': 'context', 'keywords': ['situation', 'context', 'about', 'ситуация', 'контекст', 'расскажи']},
    {'question_id': 'CQ051', 'question': 'How long has this been going on?', 'domain': 'context', 'keywords': ['long', 'time', 'going on', 'давно', 'продолжается', 'сколько']},
    {'question_id': 'CQ052', 'question': 'Is this happening in your work life, personal life, or both?', 'domain': 'context', 'keywords': ['work', 'personal', 'life', 'работа', 'личная', 'жизнь']},
]

# ---------------------------------------------------------------------------
# Internal cache
# ---------------------------------------------------------------------------

_QUESTION_BANK: list[dict[str, Any]] | None = None


def _load_question_bank() -> list[dict[str, Any]]:
    """Load ClariQ question bank + coaching extensions. Cached after first load."""
    global _QUESTION_BANK
    if _QUESTION_BANK is not None:
        return _QUESTION_BANK

    questions: list[dict[str, Any]] = []

    # Load ClariQ question bank
    if _QUESTION_BANK_PATH.exists():
        try:
            with _QUESTION_BANK_PATH.open(encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    qid = str(row.get('question_id') or '').strip()
                    qtxt = str(row.get('question') or '').strip()
                    if qid and qtxt:
                        words = _tokenize(qtxt)
                        questions.append({
                            'question_id': qid,
                            'question': qtxt,
                            'domain': 'general',
                            'keywords': words,
                        })
        except Exception:
            pass  # Fall through to coaching-only bank

    # Always add coaching-specific questions
    for cq in _COACHING_QUESTIONS:
        entry = dict(cq)
        entry['keywords'] = _tokenize(cq['question']) + list(cq.get('keywords', []))
        questions.append(entry)

    _QUESTION_BANK = questions
    return _QUESTION_BANK


def _tokenize(text: str) -> list[str]:
    """Lowercase, split, remove stopwords, keep words ≥ 3 chars."""
    words = re.findall(r'\b[a-zа-яё]{3,}\b', text.lower())
    return [w for w in words if w not in _STOPWORDS]


def _overlap_score(query_tokens: set[str], question_tokens: list[str]) -> float:
    """Jaccard-like overlap between query terms and question terms."""
    if not query_tokens or not question_tokens:
        return 0.0
    q_set = set(question_tokens)
    intersection = len(query_tokens & q_set)
    union = len(query_tokens | q_set)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ClarificationResult:
    needs_clarification: bool
    score: float                      # 0.0 = clear, 1.0 = very ambiguous
    selected_question: str
    question_id: str
    question_domain: str              # 'general' | 'emotion' | 'goal' | 'planning' | ...
    fallback_question: str = ''


def score_clarification_need(message: str) -> float:
    """
    Score how much a message needs clarification.

    Returns float in [0.0, 1.0]:
        0.0 = intent is clear, no clarification needed
        1.0 = very ambiguous, clarification is essential

    Rule-based. Derived from ClariQ clarification_need label analysis.
    No LLM.
    """
    text = str(message or '').strip()
    if not text:
        return 0.0

    lowered = text.lower()
    score = 0.0

    # Very short messages — careful: single-word entity queries (ClariQ label=4)
    # are actually clear intent ("figs", "iron", "UNC"), not vague.
    # Only 2-3 word messages with no specific structure are genuinely ambiguous.
    word_count = len(text.split())
    if word_count == 1:
        score += 0.05  # single word = specific entity lookup, not vague
    elif word_count <= 3:
        score += 0.20  # 2-3 words = somewhat ambiguous, but less than before
    elif word_count <= 6:
        score += 0.15
    elif word_count >= 20:
        score -= 0.10  # longer = usually more context

    # High-need markers (action-oriented, vague)
    for pattern in _HIGH_NEED_PATTERNS:
        if re.search(pattern, lowered):
            score += 0.12

    # Low-need markers (specific intent)
    for pattern in _LOW_NEED_PATTERNS:
        if re.search(pattern, lowered):
            score -= 0.15

    # Ellipsis / trailing off (common in vague messages)
    if text.endswith('...') or text.endswith('…'):
        score += 0.10

    # Personal pronouns without clear referent ("it", "this", "that")
    vague_refs = len(re.findall(r'\b(it|this|that|these|those|something|somehow|это|то|что-то)\b', lowered))
    score += min(0.20, vague_refs * 0.07)

    # Questions without subject specificity
    if re.search(r'\?', text):
        if not any(re.search(p, lowered) for p in _LOW_NEED_PATTERNS):
            score += 0.08

    # Coaching-specific: emotional ambiguity without concrete situation
    emotion_words = re.findall(r'\b(feel|feeling|felt|afraid|scared|confused|lost|overwhelmed|anxious|чувствую|боюсь|потерян|тревожно)\b', lowered)
    situation_words = re.findall(r'\b(because|since|when|after|before|yesterday|today|week|month|year|потому что|так как|когда|после|вчера|сегодня)\b', lowered)
    if emotion_words and not situation_words:
        score += 0.18  # Emotion without situational context → clarify

    return round(min(1.0, max(0.0, score)), 3)


def select_clarifying_question(
    message: str,
    context: str = '',
    *,
    preferred_domain: str = '',
    top_n: int = 1,
    exclude_ids: set[str] | None = None,
) -> list[ClarificationResult]:
    """
    Select the most relevant clarifying question(s) from the question bank.

    Uses keyword overlap between the user message + context and each question.
    Deterministic. No LLM.

    Args:
        message: The user's current message.
        context: Optional additional context (recent turns, topic).
        preferred_domain: Prefer questions from this domain ('emotion', 'goal',
                          'planning', 'feedback', 'constraint', 'context', 'general').
        top_n: Number of candidates to return.
        exclude_ids: Set of question_ids already asked in this session.

    Returns list of ClarificationResult (highest score first).
    """
    bank = _load_question_bank()
    excluded = set(exclude_ids or [])

    combined = f'{message} {context}'.strip()
    query_tokens = set(_tokenize(combined))

    scored: list[tuple[float, dict[str, Any]]] = []
    for entry in bank:
        if entry['question_id'] in excluded:
            continue
        base_score = _overlap_score(query_tokens, entry['keywords'])

        # Domain preference boost
        domain_boost = 0.0
        if preferred_domain and entry.get('domain') == preferred_domain:
            domain_boost = 0.15

        # Coaching questions get a slight boost for coaching-domain context
        if entry['question_id'].startswith('CQ'):
            coaching_signals = re.findall(
                r'\b(feel|plan|goal|step|help|stuck|confused|afraid|чувствую|цель|шаг|помоги|застрял)\b',
                combined.lower()
            )
            if coaching_signals:
                domain_boost += 0.08

        scored.append((base_score + domain_boost, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    need_score = score_clarification_need(message)
    results: list[ClarificationResult] = []

    for score, entry in scored[:max(top_n, 2)]:
        results.append(ClarificationResult(
            needs_clarification=need_score >= 0.25,
            score=need_score,
            selected_question=entry['question'],
            question_id=entry['question_id'],
            question_domain=entry.get('domain', 'general'),
            fallback_question='Could you tell me a bit more about what you mean?',
        ))

    if not results:
        results.append(ClarificationResult(
            needs_clarification=need_score >= 0.25,
            score=need_score,
            selected_question='Could you tell me a bit more about what you mean?',
            question_id='FALLBACK',
            question_domain='general',
        ))

    return results[:top_n]


def get_clarification_for_action(
    message: str,
    context: str = '',
    *,
    event_type: str = '',
    session_asked_ids: set[str] | None = None,
) -> ClarificationResult:
    """
    High-level entry point for the action layer.

    Maps event_type to a preferred question domain and returns the
    single best clarifying question.
    """
    domain_map = {
        'planning_request':    'planning',
        'feedback_report':     'feedback',
        'social_shame':        'emotion',
        'social_rejection':    'emotion',
        'overload':            'emotion',
        'threat':              'constraint',
        'neutral':             'context',
    }
    preferred = domain_map.get(event_type, '')

    results = select_clarifying_question(
        message,
        context,
        preferred_domain=preferred,
        top_n=1,
        exclude_ids=session_asked_ids,
    )
    return results[0]


def build_clarification_summary(session_asked_ids: set[str]) -> str:
    """
    Return a compact summary of which question domains have been covered,
    so the system avoids repeating the same type of question.
    """
    bank = _load_question_bank()
    domain_counts: dict[str, int] = {}
    for entry in bank:
        if entry['question_id'] in session_asked_ids:
            d = entry.get('domain', 'general')
            domain_counts[d] = domain_counts.get(d, 0) + 1

    if not domain_counts:
        return 'No clarifying questions asked yet.'
    parts = [f'{d}×{c}' for d, c in sorted(domain_counts.items())]
    return f'Asked: {", ".join(parts)}'


def load_clariq_examples(
    split: str = 'train',
    *,
    min_need: int = 1,
    max_need: int = 4,
    limit: int = 0,
) -> list[dict[str, Any]]:
    """
    Load ClariQ examples for a given split.

    Useful for tests and calibration.
    Returns list of dicts with keys:
        topic_id, initial_request, clarification_need (int), question, answer
    """
    path_map = {
        'train': _CLARIQ_TRAIN_PATH,
        'dev':   _DATASETS_DIR / 'dev.tsv',
    }
    path = path_map.get(split)
    if path is None or not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    seen_topics: set[str] = set()
    try:
        with path.open(encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                tid = str(row.get('topic_id') or '')
                need = int(row.get('clarification_need') or 0)
                if not (min_need <= need <= max_need):
                    continue
                # One example per topic to avoid duplication
                if tid in seen_topics:
                    continue
                seen_topics.add(tid)
                rows.append({
                    'topic_id':           tid,
                    'initial_request':    str(row.get('initial_request') or ''),
                    'clarification_need': need,
                    'question':           str(row.get('question') or ''),
                    'answer':             str(row.get('answer') or ''),
                })
                if limit and len(rows) >= limit:
                    break
    except Exception:
        pass
    return rows


def load_multiturn_examples(limit: int = 0) -> list[dict[str, Any]]:
    """
    Load ClariQ multi-turn human-generated clarification dialogues.

    Returns list of dicts with:
        topic_id, initial_request, turns: list of {question, answer}
    """
    path = _DATASETS_DIR / 'multi_turn_human_generated_data.tsv'
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                turns = []
                for i in range(1, 6):
                    q = str(row.get(f'question{i}') or '').strip()
                    a = str(row.get(f'answer{i}') or '').strip()
                    if q and a:
                        turns.append({'question': q, 'answer': a})
                if turns:
                    rows.append({
                        'topic_id':       str(row.get('topic_id') or ''),
                        'initial_request': str(row.get('initial_request') or '').strip(),
                        'turns':          turns,
                    })
                if limit and len(rows) >= limit:
                    break
    except Exception:
        pass
    return rows
