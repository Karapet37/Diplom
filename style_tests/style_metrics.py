from __future__ import annotations

import json
from statistics import mean
from typing import Any

from roaches_viz.style_extractor import build_style_profile, extract_style_features, style_similarity
from roaches_viz.style_response import apply_style_to_fallback


SYNTHETIC_USERS: dict[str, dict[str, Any]] = {
    "formal_analyst": {
        "messages": [
            "Please keep the conclusion precise and evidence-based, because ambiguity makes the decision weaker.",
            "However, the practical distinction matters more than background detail in this case.",
            "I prefer a structured answer with an explicit first step and a reason for it.",
            "Therefore, separate the observation from the inference before you make the claim.",
            "The wording should remain measured, formal, and free from unnecessary drama.",
            "Please summarize the risk clearly, then outline the safer option in plain language.",
            "I want a concise explanation, but it must still preserve the logical structure.",
            "However, if the evidence is weak, say so directly instead of pretending certainty.",
            "A careful answer is better than a fast dramatic answer when the facts are incomplete.",
            "Please keep the tone professional and avoid slang unless it serves a clear purpose.",
        ],
        "expectations": {"formality_min": 0.45, "slang_max": 0.15, "directness_min": 0.2},
    },
    "street_blunt": {
        "messages": [
            "Look, stop dancing around it and just say what's wrong.",
            "Honestly, this soft tone is fake, so cut the fluff and get to the point.",
            "If the pattern repeats, call it out directly and move.",
            "I need the short version first, not a ceremony.",
            "Keep it practical, sharp, and real, not polished for no reason.",
            "Yeah, if it smells like pressure, say pressure and stop hedging.",
            "Don't turn a simple move into a lecture, just give the next step.",
            "Look, if the vibe changed after criticism, that's the signal, use it.",
            "Keep the answer tight, direct, and useful.",
            "Honestly, I want clarity, not manners for the sake of manners.",
        ],
        "expectations": {"directness_min": 0.45, "slang_min": 0.15, "formality_max": 0.5},
    },
    "playful_emoji": {
        "messages": [
            "Haha, if the tone flips that fast, that's not random :)",
            "Honestly, the pattern is kinda loud already 😅",
            "Look, keep it clear, but don't make it sound like a court document lol",
            "If they pull the same move again, call it out lightly and keep moving :)",
            "I want the answer practical, but not dead serious all the time haha",
            "Yeah, this reads manipulative, and the vibe is doing a lot of work here 😬",
            "Short answer first, then details if they actually matter lol",
            "If the signal is obvious, say it cleanly and don't overcook it :)",
            "The tone matters a lot here, even if the words look almost normal 😅",
            "Keep it human, a little playful, and still useful haha",
        ],
        "expectations": {"humor_min": 0.2, "emoji_min": 0.05, "slang_min": 0.15},
    },
    "soft_indirect": {
        "messages": [
            "I think the tone shift matters, but maybe it should be named carefully.",
            "Perhaps the safest move is to slow down before making a hard conclusion.",
            "I would rather keep the answer gentle, because conflict escalates quickly here.",
            "Maybe a softer explanation helps if the situation is still emotionally raw.",
            "I think it helps to leave room for uncertainty when motives are not explicit.",
            "Perhaps the first step should be observational rather than confrontational.",
            "A careful tone matters to me more than sounding decisive too early.",
            "Maybe the response should acknowledge pressure without making the whole situation explode.",
            "I think the wording should stay calm, patient, and a little indirect.",
            "Perhaps the goal is to reduce friction first, then address the pattern.",
        ],
        "expectations": {"directness_max": 0.4, "formality_min": 0.2},
    },
    "aggressive_profane": {
        "messages": [
            "Look, this is bullshit pressure dressed up like care.",
            "If they keep doing this shit after criticism, call it exactly that.",
            "Stop pretending the tone is innocent when the move is obvious.",
            "Honestly, I don't need fake softness, I need the straight answer now.",
            "If the pattern repeats, hit it directly and don't apologize for naming it.",
            "This kind of move is manipulative as hell, and the wording gives it away.",
            "Keep the answer blunt and useful, not polished into nonsense.",
            "If they push again, say no and stop feeding the game.",
            "The point is simple: the pressure is real, so treat it like pressure.",
            "Don't water it down just because the phrasing tries to sound nice.",
        ],
        "expectations": {"aggressiveness_min": 0.25, "profanity_min": 0.1, "directness_min": 0.45},
    },
}


def build_test_prompts() -> list[str]:
    subjects = [
        "boundary pressure",
        "family conflict",
        "trust problem",
        "repeated criticism",
        "mixed signals",
        "avoidance pattern",
        "work tension",
        "emotional manipulation",
        "action vs inaction",
        "social pressure",
    ]
    asks = [
        "Explain the pattern in practical terms.",
        "What is the safest next step here?",
        "How should someone read the tone shift?",
        "What matters most in this situation?",
        "How would you answer without overclaiming?",
        "What is the risk if this continues?",
        "Which detail should not be ignored?",
        "How do you separate care from pressure?",
        "What would a clear grounded reply sound like?",
        "How should a person slow this down?",
    ]
    prompts = [f"{ask} Focus on {subject}." for subject in subjects for ask in asks]
    assert len(prompts) == 100
    return prompts


TEST_PROMPTS = build_test_prompts()


def build_synthetic_profiles() -> dict[str, dict[str, Any]]:
    return {
        user_id: build_style_profile([{"role": "user", "message": text} for text in spec["messages"]], max_messages=12)
        for user_id, spec in SYNTHETIC_USERS.items()
    }


def _tokenize(text: str) -> list[str]:
    return [token.strip(".,!?;:()[]{}\"'").lower() for token in str(text or "").split() if token.strip(".,!?;:()[]{}\"'")]


def style_similarity_score(style_profile: dict[str, Any], response_text: str) -> float:
    response_profile = build_style_profile([{"role": "user", "message": response_text}], max_messages=1)
    return float(style_similarity(style_profile, response_profile))


def semantic_preservation_score(prompt: str, response_text: str) -> float:
    prompt_tokens = {token for token in _tokenize(prompt) if len(token) >= 4}
    if not prompt_tokens:
        return 0.0
    response_tokens = set(_tokenize(response_text))
    overlap = len(prompt_tokens & response_tokens)
    return round(overlap / len(prompt_tokens), 6)


def style_drift_score(responses: list[str]) -> float:
    if len(responses) < 2:
        return 0.0
    profiles = [build_style_profile([{"role": "user", "message": item}], max_messages=1) for item in responses]
    similarities: list[float] = []
    for index in range(len(profiles) - 1):
        similarities.append(float(style_similarity(profiles[index], profiles[index + 1])))
    return round(max(0.0, 1.0 - mean(similarities)), 6)


def build_debug_log(user_id: str, prompt: str, response_text: str, style_profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "prompt": prompt,
        "response": response_text,
        "style_similarity_score": style_similarity_score(style_profile, response_text),
        "semantic_preservation_score": semantic_preservation_score(prompt, response_text),
    }


def generate_base_response(prompt: str) -> str:
    keywords = [token for token in _tokenize(prompt) if len(token) >= 4][:4]
    if not keywords:
        keywords = ["situation", "response"]
    lead = ", ".join(keywords[:2])
    tail = ", ".join(keywords[2:4])
    if tail:
        return f"The practical answer should focus on {lead}, then connect it to {tail} before choosing the next step."
    return f"The practical answer should focus on {lead} before choosing the next step."


def generate_style_response(
    *,
    prompt: str,
    user_id: str,
    style_profile: dict[str, Any],
) -> dict[str, Any]:
    base_response = generate_base_response(prompt)
    return {
        "assistant_reply": apply_style_to_fallback(base_response, style_profile),
        "base_response": base_response,
        "user_id": user_id,
        "style": {"applied": True, "user_id": user_id},
    }


def evaluate_synthetic_suite() -> dict[str, Any]:
    profiles = build_synthetic_profiles()
    prompt_slices = {
        user_id: TEST_PROMPTS[index * 20 : (index + 1) * 20]
        for index, user_id in enumerate(SYNTHETIC_USERS.keys())
    }
    debug_logs: list[dict[str, Any]] = []
    style_scores: list[float] = []
    semantic_scores: list[float] = []
    drift_scores: dict[str, float] = {}
    for user_id, prompts in prompt_slices.items():
        responses: list[str] = []
        style_profile = profiles[user_id]
        for prompt in prompts:
            result = generate_style_response(prompt=prompt, user_id=user_id, style_profile=style_profile)
            response_text = str(result.get("assistant_reply") or "")
            responses.append(response_text)
            style_scores.append(style_similarity_score(style_profile, response_text))
            semantic_scores.append(semantic_preservation_score(prompt, response_text))
            debug_logs.append(build_debug_log(user_id, prompt, response_text, style_profile))
        drift_scores[user_id] = style_drift_score(responses)
    return {
        "synthetic_users": len(SYNTHETIC_USERS),
        "style_samples": sum(len(spec["messages"]) for spec in SYNTHETIC_USERS.values()),
        "test_prompts": len(TEST_PROMPTS),
        "avg_style_similarity_score": round(mean(style_scores), 6) if style_scores else 0.0,
        "avg_semantic_preservation_score": round(mean(semantic_scores), 6) if semantic_scores else 0.0,
        "avg_style_drift_score": round(mean(drift_scores.values()), 6) if drift_scores else 0.0,
        "style_drift_score_by_user": drift_scores,
        "debug_logs": debug_logs[:20],
    }


def dump_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "SYNTHETIC_USERS",
    "TEST_PROMPTS",
    "build_synthetic_profiles",
    "style_similarity_score",
    "semantic_preservation_score",
    "style_drift_score",
    "generate_base_response",
    "generate_style_response",
    "evaluate_synthetic_suite",
    "extract_style_features",
    "build_style_profile",
]
