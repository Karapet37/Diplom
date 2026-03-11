from __future__ import annotations

import re
from typing import Any


_SARCASM_PATTERNS = (
    re.compile(r"\byeah right\b", re.IGNORECASE),
    re.compile(r"\bas if\b", re.IGNORECASE),
    re.compile(r"\bsure\b", re.IGNORECASE),
    re.compile(r"\bobviously\b", re.IGNORECASE),
    re.compile(r"\btotally\b", re.IGNORECASE),
)
_HEDGE_PATTERNS = (
    re.compile(r"\bmaybe\b", re.IGNORECASE),
    re.compile(r"\bperhaps\b", re.IGNORECASE),
    re.compile(r"\bprobably\b", re.IGNORECASE),
    re.compile(r"\bkind of\b", re.IGNORECASE),
    re.compile(r"\bsort of\b", re.IGNORECASE),
    re.compile(r"\bapparently\b", re.IGNORECASE),
)
_CONTRAST_PATTERNS = (
    re.compile(r"\bbut\b", re.IGNORECASE),
    re.compile(r"\bhowever\b", re.IGNORECASE),
    re.compile(r"\bthough\b", re.IGNORECASE),
    re.compile(r"\byet\b", re.IGNORECASE),
    re.compile(r"\balthough\b", re.IGNORECASE),
)
_CONTEXT_SHIFT_PATTERNS = (
    re.compile(r"\bon second thought\b", re.IGNORECASE),
    re.compile(r"\bto be fair\b", re.IGNORECASE),
    re.compile(r"\bactually\b", re.IGNORECASE),
)
_LITERAL_PATTERNS = (
    re.compile(r"\bbecause\b", re.IGNORECASE),
    re.compile(r"\btherefore\b", re.IGNORECASE),
    re.compile(r"\bevidence\b", re.IGNORECASE),
    re.compile(r"\bdata\b", re.IGNORECASE),
    re.compile(r"\bfirst\b", re.IGNORECASE),
    re.compile(r"\bsecond\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\b"),
)
_NEGATION_RE = re.compile(r"\b(?:not|never|no|can't|cannot|won't|don't|didn't)\b", re.IGNORECASE)
_JOKE_RE = re.compile(r"(?:/s\b|\bjk\b|\bjust kidding\b)", re.IGNORECASE)
_QUOTE_RE = re.compile(r"[\"“”'‘’][^\"“”'‘’]{2,}[\"“”'‘’]")
_SENTIMENT_POS = re.compile(r"\b(?:love|great|awesome|amazing|nice)\b", re.IGNORECASE)
_SENTIMENT_NEG = re.compile(r"\b(?:hate|awful|bad|terrible|worst)\b", re.IGNORECASE)
_EMOJI_SARCASM = ("🙄", "😏", "😉", "🙃")
_EMOJI_NEG = ("😒", "😤", "😑", "😬")
_EMOJI_POS = ("😊", "🙂", "😁", "😄")


def _push(
    sink: list[dict[str, Any]],
    *,
    cue_id: str,
    kind: str,
    evidence: str,
    weight: float,
    start: int,
) -> None:
    sink.append(
        {
            "id": cue_id,
            "kind": kind,
            "evidence": evidence,
            "weight": float(weight),
            "start": int(start),
        }
    )


def extract_cues(text: str) -> dict[str, Any]:
    raw = str(text or "")
    cues: list[dict[str, Any]] = []

    for pattern in _SARCASM_PATTERNS:
        for match in pattern.finditer(raw):
            _push(
                cues,
                cue_id="sarcasm_marker",
                kind="sarcasm_marker",
                evidence=match.group(0),
                weight=1.0,
                start=match.start(),
            )
    for pattern in _HEDGE_PATTERNS:
        for match in pattern.finditer(raw):
            _push(
                cues,
                cue_id="hedge_marker",
                kind="hedge_marker",
                evidence=match.group(0),
                weight=1.0,
                start=match.start(),
            )
    for pattern in _CONTRAST_PATTERNS:
        for match in pattern.finditer(raw):
            _push(
                cues,
                cue_id="contrast_marker",
                kind="contrast_marker",
                evidence=match.group(0),
                weight=1.0,
                start=match.start(),
            )
    for pattern in _CONTEXT_SHIFT_PATTERNS:
        for match in pattern.finditer(raw):
            _push(
                cues,
                cue_id="context_shift_marker",
                kind="context_shift_marker",
                evidence=match.group(0),
                weight=1.0,
                start=match.start(),
            )
    for pattern in _LITERAL_PATTERNS:
        for match in pattern.finditer(raw):
            _push(
                cues,
                cue_id="literal_marker",
                kind="literal_marker",
                evidence=match.group(0),
                weight=1.0,
                start=match.start(),
            )
    for match in _NEGATION_RE.finditer(raw):
        _push(
            cues,
            cue_id="negation",
            kind="negation",
            evidence=match.group(0),
            weight=0.8,
            start=match.start(),
        )
    for match in _JOKE_RE.finditer(raw):
        _push(
            cues,
            cue_id="joke_marker",
            kind="joke_marker",
            evidence=match.group(0),
            weight=1.3,
            start=match.start(),
        )
    for match in _QUOTE_RE.finditer(raw):
        _push(
            cues,
            cue_id="irony_quote",
            kind="irony_quote",
            evidence=match.group(0),
            weight=1.0,
            start=match.start(),
        )

    if "..." in raw:
        _push(cues, cue_id="ellipsis", kind="ellipsis", evidence="...", weight=0.9, start=raw.find("..."))
    if "?!" in raw or "!?" in raw:
        idx = raw.find("?!")
        if idx < 0:
            idx = raw.find("!?")
        _push(cues, cue_id="mixed_punctuation", kind="mixed_punctuation", evidence=raw[idx : idx + 2], weight=1.1, start=idx)
    if raw.count("?") >= 2:
        _push(cues, cue_id="repeated_question", kind="repeated_question", evidence="?" * raw.count("?"), weight=1.0, start=raw.find("?"))
    if raw.count("!") >= 2:
        _push(cues, cue_id="repeated_exclamation", kind="uppercase_emphasis", evidence="!" * raw.count("!"), weight=0.7, start=raw.find("!"))

    for emoji in _EMOJI_SARCASM:
        if emoji in raw:
            _push(cues, cue_id="emoji_sarcasm", kind="emoji_sarcasm", evidence=emoji, weight=1.0, start=raw.find(emoji))
    for emoji in _EMOJI_NEG:
        if emoji in raw:
            _push(cues, cue_id="emoji_negative", kind="emoji_negative", evidence=emoji, weight=0.7, start=raw.find(emoji))
    for emoji in _EMOJI_POS:
        if emoji in raw:
            _push(cues, cue_id="emoji_positive", kind="emoji_positive", evidence=emoji, weight=0.6, start=raw.find(emoji))

    alpha = [ch for ch in raw if ch.isalpha()]
    if alpha:
        upper_ratio = sum(1 for ch in alpha if ch.isupper()) / len(alpha)
        if len(alpha) >= 8 and upper_ratio >= 0.35:
            _push(
                cues,
                cue_id="uppercase_emphasis",
                kind="uppercase_emphasis",
                evidence=f"upper_ratio={upper_ratio:.2f}",
                weight=min(1.5, upper_ratio * 2),
                start=0,
            )
    if _SENTIMENT_POS.search(raw) and _SENTIMENT_NEG.search(raw):
        _push(
            cues,
            cue_id="sentiment_conflict",
            kind="sentiment_conflict",
            evidence="positive+negative sentiment",
            weight=1.2,
            start=0,
        )

    ordered = sorted(cues, key=lambda item: (item["start"], item["id"], item["evidence"]))
    # Keep duplicate cues by position, but return summary counts for deterministic explainability.
    counts: dict[str, int] = {}
    for cue in ordered:
        counts[cue["kind"]] = counts.get(cue["kind"], 0) + 1
    return {
        "text": raw,
        "cues": ordered,
        "summary": {
            "cue_count": len(ordered),
            "counts_by_kind": counts,
        },
    }
