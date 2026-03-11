from __future__ import annotations

import math
from typing import Any

_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "id": "literal",
        "label": "literal_informational",
        "description": "Speaker is literal and direct.",
        "prior_logit": 0.2,
    },
    {
        "id": "sarcastic",
        "label": "sarcastic_ironic",
        "description": "Speaker uses irony or sarcasm.",
        "prior_logit": -0.05,
    },
    {
        "id": "ambiguous",
        "label": "ambiguous_uncertain",
        "description": "Message is tentative or underspecified.",
        "prior_logit": -0.1,
    },
    {
        "id": "conflicted",
        "label": "contradictory_or_shifted",
        "description": "Message carries conflict or context reversal.",
        "prior_logit": -0.2,
    },
)


def _normalized_probabilities(logits: list[float]) -> list[float]:
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(logits)] * len(logits)
    return [v / total for v in exps]


def normalized_entropy(probabilities: list[float]) -> float:
    if not probabilities:
        return 0.0
    eps = 1e-12
    ent = -sum(p * math.log(p + eps) for p in probabilities)
    max_ent = math.log(len(probabilities))
    if max_ent <= 0:
        return 0.0
    return max(0.0, min(1.0, ent / max_ent))


def seed_hypotheses(text: str, k: int = 3) -> list[dict[str, Any]]:
    _ = text
    size = max(1, min(int(k or 3), len(_TEMPLATES)))
    logits = [float(item["prior_logit"]) for item in _TEMPLATES[:size]]
    probs = _normalized_probabilities(logits)
    seeded: list[dict[str, Any]] = []
    for idx, tpl in enumerate(_TEMPLATES[:size]):
        seeded.append(
            {
                "id": tpl["id"],
                "label": tpl["label"],
                "description": tpl["description"],
                "prior_logit": tpl["prior_logit"],
                "logit": tpl["prior_logit"],
                "probability": probs[idx],
                "evidence": [],
            }
        )
    return seeded
