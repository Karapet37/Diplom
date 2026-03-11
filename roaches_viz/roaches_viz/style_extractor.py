from __future__ import annotations

import json
import math
import re
from collections import Counter
from statistics import mean
from typing import Any


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9']+", flags=re.UNICODE)
_QUESTION_HINTS = {
    "why",
    "what",
    "how",
    "when",
    "where",
    "who",
    "which",
    "зачем",
    "почему",
    "что",
    "как",
    "когда",
    "где",
    "кто",
}
_SLANG_HINTS = {
    "lol",
    "lmao",
    "wtf",
    "bro",
    "dude",
    "nah",
    "yeah",
    "блин",
    "черт",
    "капец",
    "типа",
    "короче",
    "ага",
}
_PROFANITY_HINTS = {
    "fuck",
    "shit",
    "damn",
    "hell",
    "сука",
    "хрен",
    "черт",
    "бляд",
}
_FORMAL_HINTS = {
    "please",
    "therefore",
    "however",
    "sincerely",
    "пожалуйста",
    "следовательно",
    "однако",
    "уважаемый",
}
_HUMOR_HINTS = {
    "haha",
    "hehe",
    "lol",
    "lmao",
    "xd",
    ")))",
    ":)",
}
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\u2600-\u27BF]",
    flags=re.UNICODE,
)


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(str(text or ""))]


def _is_noise(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True
    if len(raw) < 12:
        return True
    if raw.lower().startswith(("http://", "https://")):
        return True
    if re.fullmatch(r"[\W_]+", raw, flags=re.UNICODE):
        return True
    return False


def _is_question(text: str) -> bool:
    raw = str(text or "").strip().lower()
    if "?" in raw:
        return True
    tokens = _tokenize(raw)
    return bool(tokens and tokens[0] in _QUESTION_HINTS)


def _is_short_response(text: str) -> bool:
    tokens = _tokenize(text)
    return len(tokens) < 4 or len(str(text or "").strip()) < 20


def collect_style_samples(messages: list[dict[str, Any]] | list[str], *, max_messages: int = 12) -> list[str]:
    samples: list[str] = []
    rows = list(messages or [])[-max_messages:]
    for item in rows:
        if isinstance(item, str):
            role = "user"
            text = item
        elif isinstance(item, dict):
            role = str(item.get("role") or "user").strip().lower()
            text = str(item.get("message") or item.get("text") or "")
        else:
            continue
        if role != "user":
            continue
        if _is_noise(text) or _is_short_response(text) or _is_question(text):
            continue
        samples.append(str(text).strip())
    return samples


def extract_style_features(samples: list[str]) -> dict[str, float | dict[str, float] | list[str]]:
    if not samples:
        return {
            "sentence_length": 0.0,
            "average_word_length": 0.0,
            "punctuation_style": 0.0,
            "slang_level": 0.0,
            "directness": 0.0,
            "formality": 0.0,
            "aggressiveness": 0.0,
            "humor_level": 0.0,
            "emoji_usage": 0.0,
            "profanity_tolerance": 0.0,
            "punctuation_profile": {},
            "sentence_rhythm": [],
        }

    word_lengths: list[int] = []
    sentence_lengths: list[int] = []
    punctuation_counts = Counter()
    slang_hits = 0
    direct_hits = 0
    formal_hits = 0
    humor_hits = 0
    emoji_hits = 0
    profanity_hits = 0
    aggressive_hits = 0
    total_tokens = 0

    for sample in samples:
        tokens = _tokenize(sample)
        if not tokens:
            continue
        total_tokens += len(tokens)
        sentence_lengths.append(len(tokens))
        word_lengths.extend(len(token) for token in tokens)
        lowered = sample.lower()
        slang_hits += sum(1 for token in tokens if token in _SLANG_HINTS)
        formal_hits += sum(1 for token in tokens if token in _FORMAL_HINTS)
        profanity_hits += sum(1 for token in tokens if any(token.startswith(hint) for hint in _PROFANITY_HINTS))
        humor_hits += sum(1 for hint in _HUMOR_HINTS if hint in lowered)
        emoji_hits += len(_EMOJI_RE.findall(sample))
        aggressive_hits += sample.count("!") + sum(1 for token in tokens if token in {"never", "always", "must", "должен", "никогда", "всегда"})
        direct_hits += sum(1 for token in tokens if token in {"do", "stop", "listen", "look", "смотри", "слушай", "делай", "хватит"})
        for char in sample:
            if char in "!?.,;:-()":
                punctuation_counts[char] += 1

    token_base = max(total_tokens, 1)
    sentence_base = max(len(sentence_lengths), 1)
    punctuation_density = sum(punctuation_counts.values()) / max(sum(len(sample) for sample in samples), 1)
    rhythm = [round(length / max(mean(sentence_lengths), 1), 4) for length in sentence_lengths[:8]]

    return {
        "sentence_length": round(mean(sentence_lengths), 6) if sentence_lengths else 0.0,
        "average_word_length": round(mean(word_lengths), 6) if word_lengths else 0.0,
        "punctuation_style": round(min(punctuation_density * 18.0, 1.0), 6),
        "slang_level": round(min(slang_hits / token_base * 6.0, 1.0), 6),
        "directness": round(min((direct_hits + aggressive_hits * 0.25) / sentence_base * 0.6, 1.0), 6),
        "formality": round(min((formal_hits / token_base * 7.0) + (mean(word_lengths) / 12.0 if word_lengths else 0.0), 1.0), 6),
        "aggressiveness": round(min((aggressive_hits / sentence_base) * 0.28 + (profanity_hits / token_base) * 2.2, 1.0), 6),
        "humor_level": round(min(humor_hits / sentence_base * 0.45, 1.0), 6),
        "emoji_usage": round(min(emoji_hits / sentence_base * 0.35, 1.0), 6),
        "profanity_tolerance": round(min(profanity_hits / token_base * 5.5, 1.0), 6),
        "punctuation_profile": {
            key: round(value / max(sum(punctuation_counts.values()), 1), 6)
            for key, value in sorted(punctuation_counts.items())
        },
        "sentence_rhythm": rhythm,
    }


def build_style_embedding(features: dict[str, Any]) -> list[float]:
    return [
        round(float(features.get("sentence_length") or 0.0) / 32.0, 6),
        round(float(features.get("slang_level") or 0.0), 6),
        round(float(features.get("formality") or 0.0), 6),
        round(float(features.get("aggressiveness") or 0.0), 6),
        round(float(features.get("humor_level") or 0.0), 6),
        round(float(features.get("punctuation_style") or 0.0), 6),
        round(float(features.get("directness") or 0.0), 6),
        round(float(features.get("profanity_tolerance") or 0.0), 6),
    ]


def _top_phrases(samples: list[str], *, limit: int = 3) -> list[str]:
    phrases = []
    for sample in samples:
        stripped = str(sample).strip()
        if stripped and stripped not in phrases:
            phrases.append(stripped)
        if len(phrases) >= limit:
            break
    return phrases


def _vocabulary_bias(samples: list[str], *, limit: int = 12) -> list[str]:
    counts = Counter()
    for sample in samples:
        for token in _tokenize(sample):
            if len(token) >= 4:
                counts[token] += 1
    return [token for token, _count in counts.most_common(limit)]


def build_speech_dna(samples: list[str], features: dict[str, Any]) -> dict[str, Any]:
    punctuation_profile = dict(features.get("punctuation_profile") or {})
    rhythm = list(features.get("sentence_rhythm") or [])
    return {
        "style_embedding": build_style_embedding(features),
        "example_phrases": _top_phrases(samples, limit=3),
        "vocabulary_bias": _vocabulary_bias(samples, limit=10),
        "punctuation_profile": punctuation_profile,
        "sentence_rhythm": rhythm[:6],
    }


def build_style_profile(messages: list[dict[str, Any]] | list[str], *, max_messages: int = 12) -> dict[str, Any]:
    samples = collect_style_samples(messages, max_messages=max_messages)
    features = extract_style_features(samples)
    return {
        "sample_count": len(samples),
        "style_embedding": build_style_embedding(features),
        "speech_dna": build_speech_dna(samples, features),
        "style_examples": _top_phrases(samples, limit=3),
        "features": features,
    }


def style_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    a = list(left.get("style_embedding") or [])
    b = list(right.get("style_embedding") or [])
    if not a or not b or len(a) != len(b):
        return 0.0
    distance = math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
    return round(max(0.0, 1.0 - distance / math.sqrt(len(a))), 6)


def profile_to_json(profile: dict[str, Any]) -> str:
    return json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True)
