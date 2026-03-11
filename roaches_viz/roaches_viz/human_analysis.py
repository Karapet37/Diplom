from __future__ import annotations

import re
from typing import Any


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_ACTOR_RE = re.compile(
    r"\b(?:he|she|they|you|we|i|boss|manager|friend|partner|coworker|client|neighbor|teacher|parent|mother|father|husband|wife|girlfriend|boyfriend|employee|employer|doctor|lawyer|police)\b",
    re.IGNORECASE,
)
_REPORTED_SPEECH_RE = re.compile(
    r"\b(?:said|says|told|heard|apparently|allegedly|rumor|rumour|gossip|claimed|claims|according to)\b",
    re.IGNORECASE,
)
_DIRECT_OBSERVATION_RE = re.compile(
    r"\b(?:i saw|i heard|i read|i checked|i have|i got|i received|witnessed|recorded|screenshot|message|email|video|photo|document|contract)\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"\b(?:today|yesterday|tonight|tomorrow|last\s+\w+|next\s+\w+|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}:\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)
_SITUATIONAL_RE = re.compile(
    r"\b(?:because|after|before|during|while|when|under|due to|since|at work|at home|in public|in private|during the meeting|under pressure)\b",
    re.IGNORECASE,
)
_TRAIT_RE = re.compile(
    r"\b(?:he|she|they|you|[A-Z][a-z]+)\s+(?:is|was|seems|looks|acts)\s+(?:always\s+|never\s+)?"
    r"(?:lazy|selfish|manipulative|dishonest|crazy|rude|toxic|dangerous|unstable|irresponsible|unreliable|mean|aggressive|abusive|controlling)\b",
)
_TRAIT_WORD_RE = re.compile(
    r"\b(?:lazy|selfish|manipulative|dishonest|crazy|rude|toxic|dangerous|unstable|irresponsible|unreliable|mean|aggressive|abusive|controlling)\b",
    re.IGNORECASE,
)
_QUOTE_RE = re.compile(r"[\"“”'‘’][^\"“”'‘’]{2,}[\"“”'‘’]")
_INFERENCE_RE = re.compile(
    r"\b(?:maybe|perhaps|probably|seems|looks like|i think|i guess|apparently|allegedly|must be|clearly)\b",
    re.IGNORECASE,
)
_LEGAL_ISSUE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "employment": ("job", "boss", "manager", "salary", "fired", "coworker", "employer", "employee", "hr"),
    "threat_or_safety": ("threat", "violent", "violence", "stalk", "hit", "attack", "unsafe", "afraid", "abuse"),
    "harassment": ("harass", "harassment", "bully", "bullying", "hostile", "unwanted", "pressure"),
    "privacy_or_consent": ("private", "recording", "photo", "screenshot", "secret", "consent", "permission"),
    "reputation_or_defamation": ("rumor", "rumour", "gossip", "lie", "lying", "slander", "reputation"),
    "money_or_contract": ("contract", "money", "debt", "loan", "invoice", "rent", "pay", "payment"),
    "discrimination": ("race", "gender", "religion", "disabled", "pregnant", "nationality", "discrimination"),
}

_RESEARCH_BASIS: list[dict[str, str]] = [
    {
        "id": "attribution",
        "concept": "Person-situation caution",
        "basis": "Jones & Harris 1967; Ross 1977",
        "why_it_matters": "Do not jump from behavior to character without weighing situational constraints.",
    },
    {
        "id": "relevance",
        "concept": "Fact relevance",
        "basis": "Federal Rule of Evidence 401",
        "why_it_matters": "Separate details that actually change the assessment from decorative noise.",
    },
    {
        "id": "hearsay",
        "concept": "Source quality / hearsay caution",
        "basis": "Federal Rule of Evidence 802",
        "why_it_matters": "Reported speech is weaker than direct observation and should not be treated as equal proof.",
    },
    {
        "id": "character",
        "concept": "Character-trait caution",
        "basis": "Federal Rule of Evidence 404",
        "why_it_matters": "Avoid treating trait labels as proof of what happened on the specific occasion.",
    },
]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _level(score: float) -> str:
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"


def _split_sentences(text: str) -> list[str]:
    raw = [part.strip() for part in _SENTENCE_SPLIT_RE.split(str(text or "").strip())]
    return [part for part in raw if part]


def _short_list(items: list[str], limit: int = 6) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item.strip())
        if len(unique) >= limit:
            break
    return unique


def _cue_count(cues: list[dict[str, Any]], kind: str) -> int:
    return sum(1 for cue in cues if str(cue.get("kind") or "") == kind)


def analyze_human_context(
    text: str,
    *,
    cues: list[dict[str, Any]] | None = None,
    top_hypotheses: list[dict[str, Any]] | None = None,
    mode: str = "build",
) -> dict[str, Any]:
    raw_text = str(text or "").strip()
    sentences = _split_sentences(raw_text)
    cues = list(cues or [])
    top_hypotheses = list(top_hypotheses or [])
    mode_name = str(mode or "").strip().lower() or "build"

    actors = _short_list([match.group(0) for match in _ACTOR_RE.finditer(raw_text)], limit=8)
    reported_speech = [sentence for sentence in sentences if _REPORTED_SPEECH_RE.search(sentence)]
    direct_observation = [sentence for sentence in sentences if _DIRECT_OBSERVATION_RE.search(sentence)]
    time_anchors = _short_list([match.group(0) for match in _TIME_RE.finditer(raw_text)], limit=8)
    situational_details = [sentence for sentence in sentences if _SITUATIONAL_RE.search(sentence)]
    trait_claims = [
        sentence
        for sentence in sentences
        if _TRAIT_RE.search(sentence) or (_TRAIT_WORD_RE.search(sentence) and (_ACTOR_RE.search(sentence) or _QUOTE_RE.search(sentence)))
    ]
    fact_like = [
        sentence
        for sentence in sentences
        if sentence
        and not _INFERENCE_RE.search(sentence)
        and (
            _DIRECT_OBSERVATION_RE.search(sentence)
            or _TIME_RE.search(sentence)
            or any(ch.isdigit() for ch in sentence)
        )
    ]
    inference_like = [
        sentence
        for sentence in sentences
        if sentence and (_INFERENCE_RE.search(sentence) or sentence in trait_claims or sentence in reported_speech)
    ]

    issue_spots: list[str] = []
    lowered = raw_text.lower()
    for issue_id, keywords in _LEGAL_ISSUE_KEYWORDS.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword.lower())}\b", lowered):
                issue_spots.append(issue_id)
                break

    ambiguity_score = _clamp(
        (_cue_count(cues, "hedge_marker") * 0.18)
        + (_cue_count(cues, "context_shift_marker") * 0.12)
        + (_cue_count(cues, "mixed_punctuation") * 0.14)
        + (_cue_count(cues, "repeated_question") * 0.1)
        + (_cue_count(cues, "ellipsis") * 0.08)
        + (_cue_count(cues, "irony_quote") * 0.08)
    )
    conflict_score = _clamp(
        (_cue_count(cues, "contrast_marker") * 0.18)
        + (_cue_count(cues, "sentiment_conflict") * 0.22)
        + (_cue_count(cues, "negation") * 0.08)
    )
    attribution_risk = _clamp(
        (len(trait_claims) * 0.34)
        + (0.16 if trait_claims and not situational_details else 0.0)
        + (0.08 if not actors else 0.0)
        + (0.08 if _cue_count(cues, "sarcasm_marker") else 0.0)
    )
    evidence_quality = _clamp(
        0.22
        + (len(direct_observation) * 0.2)
        + (len(time_anchors) * 0.08)
        + (0.08 if actors else 0.0)
        + (len(fact_like) * 0.05)
        - (len(reported_speech) * 0.12)
        - (len(inference_like) * 0.06)
        - (ambiguity_score * 0.16)
    )
    decision_risk = _clamp(
        (ambiguity_score * 0.36)
        + (conflict_score * 0.26)
        + ((1.0 - evidence_quality) * 0.28)
        + ((0.1 if issue_spots else 0.0) + (0.08 if reported_speech else 0.0))
    )

    top_labels = [str(item.get("label") or item.get("id") or "") for item in top_hypotheses[:3]]
    missing_context: list[str] = []
    if not actors:
        missing_context.append("Actors are underspecified.")
    if not time_anchors:
        missing_context.append("Time anchors are missing or too vague.")
    if not direct_observation and reported_speech:
        missing_context.append("Most support is reported speech rather than direct observation.")
    if attribution_risk >= 0.45:
        missing_context.append("There is a risk of judging character before mapping situational constraints.")
    if ambiguity_score >= 0.45:
        missing_context.append("Meaning is unstable enough that a clarifying question should come before a conclusion.")

    if ambiguity_score >= 0.45 and evidence_quality < 0.45:
        core_problem = "The main problem is acting on a socially or legally weak interpretation before facts and context are separated."
    elif issue_spots:
        core_problem = f"The main problem is a potentially sensitive {issue_spots[0].replace('_', ' ')} situation with incomplete source quality."
    elif conflict_score >= 0.35:
        core_problem = "The main problem is conflicting signals about intent, motive, or emotional meaning."
    else:
        core_problem = "The main problem is incomplete context around what happened, why it happened, and how sure we should be."

    next_questions: list[str] = []
    if not direct_observation:
        next_questions.append("What did you directly observe, read, or receive yourself?")
    if not time_anchors:
        next_questions.append("When exactly did this happen?")
    if reported_speech:
        next_questions.append("What was said verbatim, and who is the primary source?")
    if trait_claims and not situational_details:
        next_questions.append("What situational pressures or constraints were present?")
    if not next_questions:
        next_questions.append("Which single detail, if confirmed, would most change the conclusion?")

    cautions: list[str] = []
    if reported_speech:
        cautions.append("Treat reported speech as weaker support than primary-source material.")
    if trait_claims:
        cautions.append("Do not treat trait labels as proof of what happened on this specific occasion.")
    if issue_spots:
        cautions.append("Preserve timestamps, messages, screenshots, and documents before drawing hard conclusions.")
    if ambiguity_score >= 0.45:
        cautions.append("Clarify intent before escalating or accusing.")

    plan_adjustments = {
        "clarity": round(-(ambiguity_score * 0.18), 6),
        "alignment": round(-(attribution_risk * 0.14), 6),
        "progress": round(-((1.0 - evidence_quality) * 0.08), 6),
        "rapport": round(-(conflict_score * 0.06), 6),
        "risk": round((decision_risk * 0.2) + (0.05 if issue_spots else 0.0), 6),
    }

    return {
        "mode": mode_name,
        "person_analysis": {
            "actors": actors,
            "trait_claims": trait_claims[:5],
            "situational_details": situational_details[:5],
            "attribution_risk": {
                "score": round(attribution_risk, 6),
                "level": _level(attribution_risk),
                "plain_explanation": (
                    "The text leans toward explaining behavior by character more than by situation."
                    if attribution_risk >= 0.45
                    else "The text contains some situational grounding and is less dependent on trait-based judgment."
                ),
            },
        },
        "situation_analysis": {
            "tone_instability": {
                "score": round(max(ambiguity_score, conflict_score), 6),
                "level": _level(max(ambiguity_score, conflict_score)),
            },
            "ambiguity_score": round(ambiguity_score, 6),
            "conflict_score": round(conflict_score, 6),
            "time_anchors": time_anchors,
            "direct_observation_count": len(direct_observation),
            "reported_speech_count": len(reported_speech),
            "top_interpretations": top_labels,
            "missing_context": missing_context,
        },
        "problem_analysis": {
            "core_problem": core_problem,
            "decision_risk": {
                "score": round(decision_risk, 6),
                "level": _level(decision_risk),
            },
            "competing_interpretations": top_labels,
            "next_best_questions": next_questions[:4],
        },
        "legal_analysis": {
            "issue_spots": issue_spots,
            "fact_like_statements": fact_like[:5],
            "inference_like_statements": inference_like[:5],
            "reported_speech": reported_speech[:5],
            "evidence_quality": {
                "score": round(evidence_quality, 6),
                "level": _level(evidence_quality),
                "plain_explanation": (
                    "The account has enough direct or time-anchored detail to support cautious analysis."
                    if evidence_quality >= 0.55
                    else "The account relies heavily on inference, missing timestamps, or second-hand reports."
                ),
            },
            "cautions": cautions,
            "not_legal_advice": True,
        },
        "analysis_adjustments": {
            "planning_state_delta": plan_adjustments,
        },
        "research_basis": _RESEARCH_BASIS,
    }
