from __future__ import annotations

from typing import Any


def build_style_prompt_block(style_profile: dict[str, Any] | None) -> str:
    if not style_profile:
        return ""
    features = dict(style_profile.get("features") or {})
    speech_dna = dict(style_profile.get("speech_dna") or {})
    examples = list(style_profile.get("style_examples") or [])
    lines = [
        "User style profile is active. Match style, not content.",
        "Replicate stylistic patterns only. Do not copy factual content or private details from the examples.",
        f"Sentence length target: {float(features.get('sentence_length') or 0.0):.2f}",
        f"Slang level: {float(features.get('slang_level') or 0.0):.2f}",
        f"Formality: {float(features.get('formality') or 0.0):.2f}",
        f"Aggressiveness: {float(features.get('aggressiveness') or 0.0):.2f}",
        f"Humor level: {float(features.get('humor_level') or 0.0):.2f}",
        f"Punctuation style: {float(features.get('punctuation_style') or 0.0):.2f}",
        f"Directness: {float(features.get('directness') or 0.0):.2f}",
        f"Profanity tolerance: {float(features.get('profanity_tolerance') or 0.0):.2f}",
    ]
    if speech_dna.get("vocabulary_bias"):
        lines.append("Vocabulary bias: " + ", ".join(list(speech_dna.get("vocabulary_bias") or [])[:8]))
    if speech_dna.get("punctuation_profile"):
        lines.append("Punctuation profile: " + ", ".join(f"{key}={value}" for key, value in sorted(dict(speech_dna.get("punctuation_profile") or {}).items())[:6]))
    if examples:
        lines.append("Style examples:")
        for example in examples[:3]:
            lines.append(f"- {example}")
    return "\n".join(lines)


def apply_style_to_fallback(reply: str, style_profile: dict[str, Any] | None) -> str:
    text = str(reply or "").strip()
    if not text or not style_profile:
        return text
    features = dict(style_profile.get("features") or {})
    slang = float(features.get("slang_level") or 0.0)
    directness = float(features.get("directness") or 0.0)
    formality = float(features.get("formality") or 0.0)
    aggressiveness = float(features.get("aggressiveness") or 0.0)
    humor = float(features.get("humor_level") or 0.0)
    emoji = float(features.get("emoji_usage") or 0.0)
    profanity = float(features.get("profanity_tolerance") or 0.0)
    punctuation = dict((style_profile.get("speech_dna") or {}).get("punctuation_profile") or {})

    if formality >= 0.62 and directness <= 0.45 and not text.startswith(("From my point of view,", "In practical terms,")):
        text = f"From my point of view, {text[:1].lower() + text[1:] if text else text}"
    elif directness >= 0.68 and not text.startswith(("Look,", "Honestly,", "Listen,")):
        text = f"Look, {text[:1].lower() + text[1:] if text else text}"
    elif directness <= 0.28 and not text.startswith(("Maybe ", "I think ")):
        text = f"I think {text[:1].lower() + text[1:] if text else text}"

    if slang >= 0.45 and "Honestly," not in text and "Look," not in text and "From my point of view," not in text:
        text = f"Honestly, {text[:1].lower() + text[1:] if text else text}"

    if formality >= 0.65:
        text = text.replace("The practical answer", "The practical response", 1)
        text = text.replace("the next step", "the next considered step", 1)

    if aggressiveness >= 0.28 and "straight answer" not in text.lower():
        text = text.replace("The practical answer", "The straight answer", 1)
        text = text.replace("The practical response", "The straight response", 1)

    if profanity >= 0.12 and "damn" not in text.lower():
        text = text.replace("The straight answer", "The damn straight answer", 1)

    if humor >= 0.35 and "haha" not in text.lower():
        text = text.rstrip(".") + " haha."

    if emoji >= 0.08 and ":)" not in text and "😅" not in text:
        text = text.rstrip(".!") + " :)"

    exclam = float(punctuation.get("!", 0.0) or 0.0)
    ellipsis = float(punctuation.get(".", 0.0) or 0.0)
    if exclam >= 0.18 and not text.endswith(("!", "!!")):
        text = text.rstrip(".") + "!"
    elif ellipsis >= 0.28 and not text.endswith("..."):
        text = text.rstrip(".") + "..."
    return text
