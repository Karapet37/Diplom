from __future__ import annotations

from typing import Any

from .duplicate_resolver import normalize_name
from .models import HeadBundle, MessageAnalysis, Situation

_FOCUS_LEXICON: dict[str, tuple[str, ...]] = {
    'identity': (
        'who',
        'yourself',
        'identity',
        'person',
        'background',
        'introduce',
        'self',
        'kind',
        'nature',
        'character',
    ),
    'work': (
        'work',
        'job',
        'role',
        'routine',
        'day',
        'shift',
        'practice',
        'task',
        'responsib',
        'profession',
        'do',
    ),
    'decision': (
        'decid',
        'judge',
        'reason',
        'think',
        'choose',
        'weigh',
        'sort',
        'handle',
        'approach',
        'priorit',
    ),
    'memory': (
        'remember',
        'recall',
        'past',
        'history',
        'anchor',
        'memory',
        'once',
        'learned',
        'grew',
        'kept',
    ),
    'values': (
        'value',
        'principle',
        'believe',
        'important',
        'protect',
        'care',
        'matter',
        'prefer',
    ),
    'relations': (
        'trust',
        'family',
        'friend',
        'ally',
        'rival',
        'with',
        'toward',
        'relationship',
    ),
    'conflict': (
        'conflict',
        'struggle',
        'tension',
        'fear',
        'resent',
        'torn',
        'ashamed',
        'avoid',
    ),
}

_STOP_TOKENS = {
    'a',
    'an',
    'the',
    'and',
    'or',
    'but',
    'if',
    'then',
    'so',
    'to',
    'of',
    'for',
    'in',
    'on',
    'at',
    'with',
    'from',
    'by',
    'is',
    'are',
    'was',
    'were',
    'be',
    'been',
    'being',
    'do',
    'does',
    'did',
    'how',
    'what',
    'when',
    'where',
    'why',
    'your',
    'you',
    'me',
    'my',
    'i',
    'it',
    'this',
    'that',
    'these',
    'those',
    'like',
    'again',
    'here',
}


def _tokens(text: str) -> list[str]:
    return [token for token in normalize_name(text).split() if token and token not in _STOP_TOKENS]


def _stem_hits(tokens: list[str], stems: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    for token in tokens:
        if any(token.startswith(stem) or stem.startswith(token) for stem in stems):
            hits.append(token)
    return list(dict.fromkeys(hits))


def _persona_category_tokens(bundle: HeadBundle | None) -> dict[str, set[str]]:
    if bundle is None:
        return {key: set() for key in _FOCUS_LEXICON}
    form = dict(bundle.persona_form or {})
    values: dict[str, list[str]] = {
        'identity': [
            str(bundle.name or ''),
            str(form.get('biography') or ''),
            str(form.get('identity_class') or ''),
            *[str(item) for item in list(form.get('core_dispositions') or [])],
            *[str(item) for item in list(form.get('social_roles') or [])],
        ],
        'work': [
            str(form.get('biography') or ''),
            *[str(item) for item in list(form.get('work_habits') or [])],
            *[str(item) for item in list(form.get('habits') or [])],
            *[str(item) for item in list(form.get('knowledge_domains') or [])],
            *[str(item.get('target') or '') for item in list(bundle.relations or []) if isinstance(item, dict)],
        ],
        'decision': [
            *[str(item) for item in list(form.get('decision_patterns') or [])],
            *[str(item) for item in list(form.get('response_priorities') or [])],
            *[str(item) for item in list(form.get('trust_model') or [])],
            *[str(item) for item in list(form.get('reaction_patterns') or [])],
        ],
        'memory': [
            *[str(item) for item in list(form.get('memory_anchors') or [])],
            *[str(item) for item in list(form.get('memories') or [])],
            *[str(item) for item in list(form.get('personal_history') or [])],
            *[str(item) for item in list(bundle.examples or [])],
        ],
        'values': [
            *[str(item) for item in list(form.get('values') or [])],
            *[str(item) for item in list(form.get('strengths') or [])],
        ],
        'relations': [
            *[str(item) for item in list(form.get('trust_model') or [])],
            *[str(item.get('target') or '') for item in list(bundle.relations or []) if isinstance(item, dict)],
        ],
        'conflict': [
            *[str(item) for item in list(form.get('conflicts') or [])],
            *[str(item) for item in list(form.get('weaknesses') or [])],
            *[str(item) for item in list(form.get('conflict_behavior') or [])],
        ],
    }
    out: dict[str, set[str]] = {}
    for key, rows in values.items():
        out[key] = {token for row in rows for token in _tokens(row)}
    return out


def infer_semantic_focus(
    *,
    question: str,
    persona_bundle: HeadBundle | None = None,
    analysis: MessageAnalysis | None = None,
    situation: Situation | None = None,
) -> dict[str, Any]:
    question_tokens = _tokens(question)
    category_tokens = _persona_category_tokens(persona_bundle)
    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    for focus, stems in _FOCUS_LEXICON.items():
        score = 0.0
        notes: list[str] = []
        stem_matches = _stem_hits(question_tokens, stems)
        if stem_matches:
            score += 0.45
            notes.append('matched semantic lexicon: ' + ', '.join(stem_matches[:4]))
        overlap = category_tokens.get(focus, set()) & set(question_tokens)
        if overlap:
            score += min(0.45, 0.12 * len(overlap))
            notes.append('overlaps persona-structure tokens: ' + ', '.join(sorted(overlap)[:4]))
        if focus == 'identity' and persona_bundle is not None and analysis is not None and analysis.user_state.signal('contains_persona_reference') > 0.0:
            score += 0.18
            notes.append('question is directed toward the persona')
        if focus == 'decision' and situation is not None and situation.type in {'abnormal_behavior', 'neutral_query'}:
            score += 0.08
        if focus == 'memory' and any(token in {'before', 'after', 'still', 'again'} for token in question_tokens):
            score += 0.1
            notes.append('temporal wording implies recall or continuity')
        scores[focus] = round(score, 6)
        evidence[focus] = notes
    selected = [
        focus
        for focus, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if score >= 0.34
    ][:3]
    identity_fallback = {'who', 'yourself', 'background', 'introduce'} & set(question_tokens)
    if not selected and persona_bundle is not None and analysis is not None and analysis.user_state.intent == 'question' and identity_fallback:
        selected = ['identity']
        evidence['identity'] = evidence.get('identity', []) + ['fallback: persona-directed question with identity cues']
        scores['identity'] = max(scores.get('identity', 0.0), 0.34)
    return {
        'focus': selected,
        'scores': dict(scores),
        'evidence': {key: list(value) for key, value in evidence.items() if value},
    }


def render_semantic_focus_guidance(focus_payload: dict[str, Any]) -> list[str]:
    focus = [str(item).strip() for item in list(dict(focus_payload or {}).get('focus') or []) if str(item).strip()]
    guidance: list[str] = []
    if 'identity' in focus or 'work' in focus:
        guidance.append('This is a direct personal question about the persona. Start with a concrete first-person answer about biography, routine, memory, or decision process before any framing.')
    if 'work' in focus:
        guidance.append('Because the user is asking about a normal day or work routine, begin with where you work, what responsibilities fill the day, and how you make practical decisions under pressure before mentioning symbolic memory anchors.')
    if 'decision' in focus:
        guidance.append('This question probes judgment. Explain how the persona weighs facts, values, trust, and pressure when choosing.')
    if 'memory' in focus:
        guidance.append('This question probes memory. Use remembered episodes, anchors, and durable personal continuity rather than slogans.')
    if 'values' in focus:
        guidance.append('This question probes values. Name what matters, what is protected, and what lines the persona does not cross.')
    if 'relations' in focus:
        guidance.append('This question probes social ties. Ground the answer in trust, loyalty, alliances, rivalry, or dependency.')
    if 'conflict' in focus:
        guidance.append('This question probes inner conflict. Surface the tension honestly instead of flattening it into a neat assistant answer.')
    return guidance
