from __future__ import annotations

import math
from typing import Any

from ..human_analysis import analyze_human_context
from .cues import extract_cues
from .hypotheses import normalized_entropy, seed_hypotheses

_MODE_PRIOR_DELTAS: dict[str, dict[str, float]] = {
    "build": {"literal": 0.25, "ambiguous": 0.15, "sarcastic": -0.05, "conflicted": 0.05},
    "review": {"literal": -0.1, "ambiguous": 0.1, "sarcastic": 0.2, "conflicted": 0.3},
    "psychology": {"literal": -0.05, "ambiguous": 0.18, "sarcastic": 0.05, "conflicted": 0.14},
    "legal": {"literal": 0.15, "ambiguous": 0.14, "sarcastic": -0.06, "conflicted": 0.2},
}
_CUE_WEIGHTS: dict[str, dict[str, float]] = {
    "sarcasm_marker": {"literal": -1.2, "sarcastic": 2.2, "ambiguous": 0.3, "conflicted": 0.1},
    "joke_marker": {"literal": -1.0, "sarcastic": 2.0, "ambiguous": 0.2, "conflicted": 0.0},
    "hedge_marker": {"literal": -0.8, "sarcastic": 0.1, "ambiguous": 1.8, "conflicted": 0.2},
    "contrast_marker": {"literal": -0.2, "sarcastic": 0.2, "ambiguous": 0.7, "conflicted": 1.6},
    "sentiment_conflict": {"literal": -0.5, "sarcastic": 0.4, "ambiguous": 0.8, "conflicted": 1.8},
    "mixed_punctuation": {"literal": -0.7, "sarcastic": 1.0, "ambiguous": 1.1, "conflicted": 0.4},
    "repeated_question": {"literal": -0.3, "sarcastic": 0.2, "ambiguous": 1.5, "conflicted": 0.3},
    "ellipsis": {"literal": -0.2, "sarcastic": 0.3, "ambiguous": 1.2, "conflicted": 0.1},
    "irony_quote": {"literal": -0.4, "sarcastic": 1.2, "ambiguous": 0.8, "conflicted": 0.0},
    "emoji_sarcasm": {"literal": -0.8, "sarcastic": 1.6, "ambiguous": 0.4, "conflicted": 0.2},
    "emoji_negative": {"literal": -0.1, "sarcastic": 0.1, "ambiguous": 0.2, "conflicted": 0.8},
    "emoji_positive": {"literal": 0.3, "sarcastic": 0.1, "ambiguous": -0.1, "conflicted": -0.1},
    "uppercase_emphasis": {"literal": -0.3, "sarcastic": 0.6, "ambiguous": 0.4, "conflicted": 0.2},
    "negation": {"literal": -0.1, "sarcastic": 0.1, "ambiguous": 0.4, "conflicted": 0.9},
    "literal_marker": {"literal": 1.1, "sarcastic": -0.5, "ambiguous": -0.4, "conflicted": -0.2},
    "context_shift_marker": {"literal": -0.3, "sarcastic": 0.2, "ambiguous": 1.0, "conflicted": 0.6},
}
_PAIR_QUESTION_MAP: dict[tuple[str, str], str] = {
    ("literal", "sarcastic"): "Are you being literal here, or is this intentional irony?",
    ("literal", "ambiguous"): "Should I treat this as a firm statement or as tentative?",
    ("literal", "conflicted"): "Do you want one direct interpretation, or should I model the contradiction?",
    ("sarcastic", "ambiguous"): "Is the tone ironic, or are you uncertain and exploratory?",
    ("sarcastic", "conflicted"): "Is the contrast a joke, or a genuine conflict in meaning?",
    ("ambiguous", "conflicted"): "Which meaning should have priority in this context?",
}


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [v / total for v in exps]


def _best_question(hypotheses: list[dict[str, Any]], uncertainty: float) -> dict[str, Any]:
    if len(hypotheses) < 2:
        return {
            "question": "Can you provide one concrete example from context?",
            "target_hypotheses": [hypotheses[0]["id"]] if hypotheses else [],
            "expected_information_gain": 0.1,
        }
    first = hypotheses[0]
    second = hypotheses[1]
    pair = tuple(sorted((first["id"], second["id"])))
    margin = max(0.0, float(first["probability"]) - float(second["probability"]))
    voi = max(0.05, min(1.0, (1.0 - margin) * (0.5 + (uncertainty * 0.5))))
    return {
        "question": _PAIR_QUESTION_MAP.get(pair, "Which interpretation is closer to your intent?"),
        "target_hypotheses": [first["id"], second["id"]],
        "expected_information_gain": round(voi, 4),
    }


def update_hypotheses(
    hypotheses: list[dict[str, Any]],
    cues: dict[str, Any],
    *,
    top_k: int | None = None,
    mode: str = "build",
) -> dict[str, Any]:
    if not hypotheses:
        hypotheses = seed_hypotheses("", k=max(1, top_k or 3))

    ordered_cues = list(cues.get("cues") or [])
    mode_name = str(mode or "").strip().lower() or "build"
    mode_deltas = _MODE_PRIOR_DELTAS.get(mode_name, _MODE_PRIOR_DELTAS["build"])
    scored: list[dict[str, Any]] = []
    logits: list[float] = []

    for hyp in hypotheses:
        hid = str(hyp["id"])
        base = float(hyp.get("prior_logit", hyp.get("logit", 0.0)))
        logit = base + float(mode_deltas.get(hid, 0.0))
        evidence: list[dict[str, Any]] = []
        for cue in ordered_cues:
            cue_kind = str(cue.get("kind") or "")
            cue_strength = float(cue.get("weight") or 1.0)
            delta = _CUE_WEIGHTS.get(cue_kind, {}).get(hid, 0.0) * cue_strength
            if abs(delta) > 1e-9:
                logit += delta
            if delta > 0.15:
                evidence.append(
                    {
                        "cue": cue_kind,
                        "evidence": cue.get("evidence", ""),
                        "delta": round(delta, 4),
                    }
                )
        scored.append(
            {
                **hyp,
                "logit": round(logit, 6),
                "evidence": evidence,
            }
        )
        logits.append(logit)

    probs = _softmax(logits)
    for idx, hyp in enumerate(scored):
        hyp["probability"] = round(probs[idx], 6)

    ranked = sorted(scored, key=lambda item: (-float(item["probability"]), str(item["id"])))
    k = max(1, min(int(top_k or len(ranked)), len(ranked)))
    top = ranked[:k]

    uncertainty = round(normalized_entropy([float(h["probability"]) for h in ranked]), 6)
    question = _best_question(top, uncertainty)

    return {
        "top_hypotheses": top,
        "all_hypotheses": ranked,
        "uncertainty": uncertainty,
        "best_clarifying_question": question,
    }


def interpret_text(text: str, *, k: int = 3, mode: str = "build") -> dict[str, Any]:
    mode_name = str(mode or "").strip().lower() or "build"
    cues = extract_cues(text)
    seeded = seed_hypotheses(text, k=max(k, 4))
    updated = update_hypotheses(seeded, cues, top_k=k, mode=mode_name)
    analysis = analyze_human_context(
        text,
        cues=list(cues["cues"]),
        top_hypotheses=list(updated["top_hypotheses"]),
        mode=mode_name,
    )
    return {
        "text": text,
        "mode": mode_name,
        "cues": cues["cues"],
        "cue_summary": cues.get("summary", {}),
        "top_hypotheses": updated["top_hypotheses"],
        "all_hypotheses": updated["all_hypotheses"],
        "uncertainty": updated["uncertainty"],
        "best_clarifying_question": updated["best_clarifying_question"],
        "analysis": analysis,
    }
