from __future__ import annotations

import re
from typing import Any


_EMOTION_RULES = {
    "guilt_loaded": {
        "label": "Guilt-loaded emotional pressure",
        "description": "The sentence tries to create guilt so the other person feels pushed to comply.",
        "cues": ["if you loved me", "if you cared", "если бы ты любил", "после всего что я сделал"],
    },
    "criticism_pressure": {
        "label": "Critical pressure",
        "description": "The wording carries blame, correction, or moral pressure.",
        "cues": ["your fault", "из-за тебя", "ты виноват", "should", "должен", "must"],
    },
    "hurt_rejection": {
        "label": "Hurt or rejection sensitivity",
        "description": "The sentence carries pain around distance, rejection, or being ignored.",
        "cues": ["ignored", "reject", "left me", "игнор", "отверж", "бросил"],
    },
    "anger_irritation": {
        "label": "Anger or irritation",
        "description": "The tone shows sharp frustration, attack, or heated pressure.",
        "cues": ["always", "never", "всегда", "никогда", "!"],
    },
    "uncertainty_hesitation": {
        "label": "Uncertainty or hesitation",
        "description": "The sentence signals doubt, hedging, or unstable confidence.",
        "cues": ["maybe", "perhaps", "i guess", "может", "кажется", "...", "?"],
    },
    "care_attachment": {
        "label": "Care or attachment bid",
        "description": "The sentence uses closeness, care, or attachment language to influence the exchange.",
        "cues": ["love", "care", "need you", "люблю", "забоч", "нужен"],
    },
}

_LOGIC_RULES = {
    "conditional_leverage": {
        "label": "Conditional structure",
        "description": "The sentence is built around an if-then condition or leverage pattern.",
        "cues": ["if ", "если "],
    },
    "causal_claim": {
        "label": "Cause-and-effect claim",
        "description": "The sentence explains behavior through a claimed cause.",
        "cues": ["because", "because of", "потому", "из-за"],
    },
    "contrast_turn": {
        "label": "Contrast or reversal",
        "description": "The sentence shifts direction with contrast, correction, or reversal.",
        "cues": ["but", "however", "но", "однако"],
    },
    "direct_request": {
        "label": "Direct request or demand",
        "description": "The sentence pushes for an action, duty, or response.",
        "cues": ["please", "can you", "сделай", "должен", "нужно"],
    },
    "negation_frame": {
        "label": "Negation frame",
        "description": "The sentence defines meaning by refusal, denial, or absence.",
        "cues": [" not ", " never ", "нет", "не ", "никогда"],
    },
    "quoted_or_reported_speech": {
        "label": "Quoted or reported speech",
        "description": "The sentence contains a quotation or a reported statement from someone.",
        "cues": ['"', "“", "”", "said", "говор", "сказал"],
    },
    "generalization": {
        "label": "Sweeping generalization",
        "description": "The wording makes a broad rule instead of staying with one case.",
        "cues": ["always", "never", "everyone", "всегда", "никогда", "все"],
    },
    "question_probe": {
        "label": "Question probe",
        "description": "The sentence is structured as a probe for information, reassurance, or reaction.",
        "cues": ["?"],
    },
}

_PATTERN_LABELS = {
    "guilt_pressure": ("Guilt pressure candidate", "The sentence likely belongs to a guilt-based pressure pattern."),
    "blame_shifting": ("Blame shifting candidate", "The sentence likely moves responsibility away from the speaker."),
    "silent_treatment": ("Silent treatment candidate", "The sentence suggests withdrawal being used as pressure."),
    "boundary_testing": ("Boundary testing candidate", "The sentence suggests repeated pressure against limits or refusal."),
    "conflict_avoidance": ("Conflict avoidance candidate", "The sentence suggests peacekeeping at the cost of direct honesty."),
    "behavioral_pressure": ("Behavioral pressure candidate", "The sentence suggests pressure or coercive influence."),
    "avoidant_response": ("Avoidant response candidate", "The sentence suggests withdrawal or evasive response."),
}

_TRAIT_RULES = {
    "control_seeking": {
        "label": "Control-seeking style",
        "description": "The behavior candidate suggests a need to steer the other person rather than negotiate openly.",
    },
    "defensive_style": {
        "label": "Defensive style",
        "description": "The behavior candidate suggests protecting the self through blame, retreat, or justification.",
    },
    "approval_sensitive": {
        "label": "Approval-sensitive style",
        "description": "The behavior candidate suggests strong dependence on reassurance or acceptance.",
    },
    "conflict_avoidant": {
        "label": "Conflict-avoidant style",
        "description": "The behavior candidate suggests preserving calm even when direct discussion is needed.",
    },
    "ambiguity_sensitive": {
        "label": "Ambiguity-sensitive style",
        "description": "The behavior candidate suggests discomfort with uncertainty, mixed signals, or unclear intentions.",
    },
    "boundary_blind": {
        "label": "Boundary-blind style",
        "description": "The behavior candidate suggests low respect for limits or a habit of pushing past them.",
    },
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9а-яё]+", "_", value.lower()).strip("_") or "signal"


def _match_cues(lowered: str, raw_sentence: str, cues: list[str]) -> list[str]:
    matched: list[str] = []
    for cue in cues:
        if cue == "?" and "?" in raw_sentence:
            matched.append("?")
        elif cue == "!" and "!" in raw_sentence:
            matched.append("!")
        elif cue == "..." and "..." in raw_sentence:
            matched.append("...")
        elif cue.lower() in lowered:
            matched.append(cue)
    return matched


def _signal(key: str, label: str, description: str, cues: list[str], *, confidence: float) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "description": description,
        "cues": cues,
        "confidence": confidence,
    }


def _semantic_signals(concepts: list[str], patterns: list[str], triggers: list[str]) -> list[dict[str, Any]]:
    if patterns:
        key = patterns[0]
        if key == "guilt_pressure":
            return [_signal("semantic_affection_as_leverage", "Affection used as leverage", "The core meaning is not simple affection; it is affection turned into leverage.", [key], confidence=0.9)]
        if key == "blame_shifting":
            return [_signal("semantic_responsibility_shift", "Responsibility shifted outward", "The core meaning is pushing fault away from the speaker and onto someone else.", [key], confidence=0.86)]
        if key == "boundary_testing":
            return [_signal("semantic_boundary_push", "Limits are being pushed", "The core meaning is pressure against a limit, refusal, or personal boundary.", [key], confidence=0.84)]
        if key == "conflict_avoidance":
            return [_signal("semantic_surface_peace", "Surface peace over direct honesty", "The sentence centers on avoiding open conflict rather than resolving it clearly.", [key], confidence=0.82)]
    parts = concepts[:2] or triggers[:1]
    if parts:
        readable = ", ".join(part.replace("_", " ") for part in parts)
        return [_signal(f"semantic_{_slug(readable)}", f"Meaning centers on {readable}", "The sentence is mainly about this cluster of meanings and practical themes.", parts, confidence=0.7)]
    return [_signal("semantic_plain_observation", "Plain observation", "The sentence mainly reports a situation without a strong specialized semantic frame.", [], confidence=0.55)]


def _emotional_signals(sentence: str, patterns: list[str], triggers: list[str]) -> list[dict[str, Any]]:
    lowered = f" {sentence.lower()} "
    results: list[dict[str, Any]] = []
    for key, config in _EMOTION_RULES.items():
        cues = _match_cues(lowered, sentence, list(config["cues"]))
        if key == "guilt_loaded" and "guilt_pressure" in patterns:
            cues.append("pattern:guilt_pressure")
        if key == "hurt_rejection" and "rejection" in triggers:
            cues.append("trigger:rejection")
        if cues:
            confidence = min(0.95, 0.6 + 0.08 * len(set(cues)))
            results.append(_signal(key, str(config["label"]), str(config["description"]), sorted(set(cues)), confidence=confidence))
    if results:
        return sorted(results, key=lambda item: (-float(item["confidence"]), str(item["key"])))
    return [_signal("neutral_or_flat_affect", "Neutral or flat affect", "The sentence does not expose a strong emotional charge, so the emotional layer stays weak.", [], confidence=0.45)]


def _logical_structure_signals(sentence: str) -> list[dict[str, Any]]:
    lowered = f" {sentence.lower()} "
    results: list[dict[str, Any]] = []
    for key, config in _LOGIC_RULES.items():
        cues = _match_cues(lowered, sentence, list(config["cues"]))
        if cues:
            confidence = min(0.95, 0.58 + 0.07 * len(set(cues)))
            results.append(_signal(key, str(config["label"]), str(config["description"]), sorted(set(cues)), confidence=confidence))
    if results:
        return sorted(results, key=lambda item: (-float(item["confidence"]), str(item["key"])))
    return [_signal("plain_assertion", "Plain assertion", "The sentence mostly states something directly without a more complex logical frame.", [], confidence=0.5)]


def _hidden_intent_signals(patterns: list[str], emotional: list[dict[str, Any]], logical: list[dict[str, Any]]) -> list[dict[str, Any]]:
    emotion_keys = {str(item["key"]) for item in emotional}
    logic_keys = {str(item["key"]) for item in logical}
    results: list[dict[str, Any]] = []
    if "guilt_pressure" in patterns or "guilt_loaded" in emotion_keys:
        results.append(_signal("compliance_pressure", "Push toward compliance", "The hidden intent looks like getting compliance through emotional leverage rather than direct agreement.", ["guilt_pressure"], confidence=0.9))
    if "blame_shifting" in patterns or "criticism_pressure" in emotion_keys:
        results.append(_signal("blame_deflection", "Deflect blame", "The hidden intent looks like reducing personal responsibility by moving fault elsewhere.", ["blame_shifting"], confidence=0.86))
    if "boundary_testing" in patterns:
        results.append(_signal("access_expansion", "Push past limits", "The hidden intent looks like expanding access, permission, or control beyond the other person's limit.", ["boundary_testing"], confidence=0.85))
    if "conflict_avoidance" in patterns:
        results.append(_signal("harmony_preservation", "Preserve surface peace", "The hidden intent looks like avoiding open conflict even if clarity is sacrificed.", ["conflict_avoidance"], confidence=0.82))
    if "question_probe" in logic_keys and "uncertainty_hesitation" in emotion_keys:
        results.append(_signal("reassurance_seeking", "Seek reassurance", "The hidden intent looks like asking for emotional certainty, not only information.", ["question", "uncertainty"], confidence=0.77))
    elif "question_probe" in logic_keys:
        results.append(_signal("information_probe", "Probe for information", "The hidden intent looks like testing the other person's position, reaction, or knowledge.", ["question"], confidence=0.65))
    if "direct_request" in logic_keys and "care_attachment" in emotion_keys and not any(item["key"] == "compliance_pressure" for item in results):
        results.append(_signal("attachment_leverage", "Use closeness to shape behavior", "The hidden intent looks like using attachment language to steer the other person's action.", ["direct_request", "care"], confidence=0.8))
    if results:
        return sorted(results, key=lambda item: (-float(item["confidence"]), str(item["key"])))
    return [_signal("observation_only", "No strong hidden intent", "The sentence reads more like observation or reporting than an active hidden move.", [], confidence=0.5)]


def _behavior_candidates(patterns: list[str], hidden_intents: list[dict[str, Any]], triggers: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for pattern in patterns:
        label, description = _PATTERN_LABELS.get(pattern, (pattern.replace("_", " ").title(), "The sentence suggests this repeatable behavior candidate."))
        results.append(_signal(pattern, label, description, [pattern], confidence=0.86))
    if not results:
        intent_keys = {str(item["key"]) for item in hidden_intents}
        if "compliance_pressure" in intent_keys or "attachment_leverage" in intent_keys:
            label, description = _PATTERN_LABELS["behavioral_pressure"]
            results.append(_signal("behavioral_pressure", label, description, ["hidden_intent"], confidence=0.74))
        if "blame_deflection" in intent_keys:
            label, description = _PATTERN_LABELS["blame_shifting"]
            results.append(_signal("blame_shifting", label, description, ["hidden_intent"], confidence=0.74))
        if "harmony_preservation" in intent_keys:
            label, description = _PATTERN_LABELS["conflict_avoidance"]
            results.append(_signal("conflict_avoidance", label, description, ["hidden_intent"], confidence=0.72))
    if "criticism" in triggers and not results:
        label, description = _PATTERN_LABELS["behavioral_pressure"]
        results.append(_signal("behavioral_pressure", label, description, ["trigger:criticism"], confidence=0.68))
    if results:
        return sorted(results, key=lambda item: (-float(item["confidence"]), str(item["key"])))
    return [_signal("weak_pattern_signal", "Weak pattern signal", "The sentence contains too little repeated structure to support a strong pattern candidate.", [], confidence=0.45)]


def _trait_candidates(hidden_intents: list[dict[str, Any]], emotional: list[dict[str, Any]], logical: list[dict[str, Any]]) -> list[dict[str, Any]]:
    intent_keys = {str(item["key"]) for item in hidden_intents}
    emotion_keys = {str(item["key"]) for item in emotional}
    logical_keys = {str(item["key"]) for item in logical}
    keys: list[str] = []
    if "compliance_pressure" in intent_keys or "attachment_leverage" in intent_keys:
        keys.append("control_seeking")
    if "blame_deflection" in intent_keys:
        keys.append("defensive_style")
    if "reassurance_seeking" in intent_keys:
        keys.append("approval_sensitive")
    if "harmony_preservation" in intent_keys:
        keys.append("conflict_avoidant")
    if "uncertainty_hesitation" in emotion_keys or "question_probe" in logical_keys:
        keys.append("ambiguity_sensitive")
    if "access_expansion" in intent_keys:
        keys.append("boundary_blind")
    results: list[dict[str, Any]] = []
    for key in keys:
        config = _TRAIT_RULES[key]
        results.append(_signal(key, str(config["label"]), str(config["description"]), [key], confidence=0.66))
    return sorted(results, key=lambda item: (-float(item["confidence"]), str(item["key"])))


def extract_micro_signals(sentence: str, *, concepts: list[str], patterns: list[str], triggers: list[str]) -> dict[str, list[dict[str, Any]]]:
    semantic = _semantic_signals(concepts, patterns, triggers)
    emotional = _emotional_signals(sentence, patterns, triggers)
    logical = _logical_structure_signals(sentence)
    hidden_intent = _hidden_intent_signals(patterns, emotional, logical)
    behavior_candidates = _behavior_candidates(patterns, hidden_intent, triggers)
    trait_candidates = _trait_candidates(hidden_intent, emotional, logical)
    return {
        "semantic_meaning": semantic,
        "emotional_signals": emotional,
        "logical_structure": logical,
        "hidden_intent": hidden_intent,
        "behavior_pattern_candidates": behavior_candidates,
        "trait_candidates": trait_candidates,
    }
