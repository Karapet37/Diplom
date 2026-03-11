from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .graph_model import Edge, Evidence, Node, Source, edge_key
from .micro_signals import extract_micro_signals
from .store import GraphStore


_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?", re.MULTILINE)
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё]{4,}")
_NAME_RE = re.compile(r"\b([A-ZА-ЯЁ][a-zа-яё]{2,})\b")
_STOPWORDS = {
    "это", "если", "когда", "потому", "который", "которые", "может", "нужно", "после", "before", "after",
    "this", "that", "with", "from", "into", "about", "would", "could", "because", "their", "there",
    "human", "people", "person", "через", "только", "очень", "даже", "somebody", "someone", "thing",
    "said", "says", "note", "notes",
}
_DOMAIN_RULES = {
    "psychology": ["психолог", "психологи", "trauma", "family", "boundar", "attachment", "emotion", "relationship", "манипул", "травм", "семь", "границ", "эмоци"],
    "law": ["law", "legal", "right", "rights", "duty", "duties", "обязан", "прав", "закон", "суд"],
    "communication": ["communicat", "dialog", "speak", "tone", "sarcasm", "ambigu", "общени", "диалог", "тон", "сарказ"],
}
_PATTERN_RULES = {
    "guilt_pressure": ["if you loved me", "if you cared", "после всего что я сделал", "если бы ты любил", "если бы ты заботился"],
    "blame_shifting": ["your fault", "ты виноват", "because of you", "из-за тебя"],
    "silent_treatment": ["stopped talking", "ignore me", "игнорирует", "молчит неделями"],
    "boundary_testing": ["keep pushing", "давит", "нарушает границы", "won't take no"],
    "conflict_avoidance": ["avoid conflict", "избегает конфликта", "keeps peace at any cost"],
}
_TRIGGER_TERMS = {
    "rejection": ["rejection", "reject", "отверж", "ignored", "игнор"],
    "change": ["change", "новое", "перемен", "routine", "распоряд"],
    "criticism": ["criticism", "critic", "критик", "shame", "стыд"],
}
_PROFESSION_RULES = {
    "law": {
        "label": "Law",
        "keywords": ["lawyer", "attorney", "legal", "contract", "court", "advocate", "юрист", "адвокат", "суд", "договор"],
        "speech": {"formality": 0.82, "slang_level": 0.08, "directness": 0.72, "profanity_tolerance": 0.05},
    },
    "business": {
        "label": "Business",
        "keywords": ["manager", "business", "sales", "client", "market", "founder", "менеджер", "клиент", "бизнес", "продажи"],
        "speech": {"formality": 0.58, "slang_level": 0.34, "directness": 0.68, "profanity_tolerance": 0.18},
    },
    "strategy": {
        "label": "Strategy",
        "keywords": ["strategy", "strategist", "planner", "plan", "операция", "стратег", "планировщик", "стратегия"],
        "speech": {"formality": 0.64, "slang_level": 0.16, "directness": 0.62, "profanity_tolerance": 0.08},
    },
    "therapy": {
        "label": "Therapy",
        "keywords": ["therapist", "therapy", "counselor", "psychologist", "психолог", "терапевт", "консультант"],
        "speech": {"formality": 0.66, "slang_level": 0.06, "directness": 0.42, "profanity_tolerance": 0.02},
    },
    "research": {
        "label": "Research",
        "keywords": ["research", "study", "scientist", "analysis", "исслед", "учен", "анализ"],
        "speech": {"formality": 0.7, "slang_level": 0.05, "directness": 0.5, "profanity_tolerance": 0.02},
    },
}
_SIGNAL_CATEGORY_LABELS = {
    "semantic_meaning": "Semantic meaning",
    "emotional_signals": "Emotional signal",
    "logical_structure": "Logical structure",
    "hidden_intent": "Hidden intent",
    "behavior_pattern_candidates": "Behavior pattern candidate",
}


def split_sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]


def source_id_from_text(text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"src:{digest}"


def _normalize_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zа-яё0-9]+", "_", value.lower(), flags=re.IGNORECASE)
    return cleaned.strip("_") or "item"


def _pick_domain(text: str) -> str:
    lowered = text.lower()
    scored = []
    for domain, hints in _DOMAIN_RULES.items():
        score = sum(1 for hint in hints if hint in lowered)
        scored.append((score, domain))
    scored.sort(reverse=True)
    if scored and scored[0][0] > 0:
        return scored[0][1]
    return "behavioral_knowledge"


def _top_tokens(sentence: str, *, top_k: int) -> list[str]:
    tokens = [item.lower() for item in _TOKEN_RE.findall(sentence)]
    tokens = [item for item in tokens if item.lower() not in _STOPWORDS]
    ranked = sorted(Counter(tokens).items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _count in ranked[:top_k]]


def _person_names(sentence: str) -> list[str]:
    blocked = {"psychology", "family", "manipulation", "communication", "law"}
    candidates = {item for item in _NAME_RE.findall(sentence) if item.lower() not in _STOPWORDS and item.lower() not in blocked}
    lowered = sentence.lower()
    if any(token in lowered for token in [" said", "says", "говор", "\"", "“", "”"]):
        return sorted(candidates)
    return []


def _pattern_keys(sentence: str) -> list[str]:
    lowered = sentence.lower()
    matches = [key for key, cues in _PATTERN_RULES.items() if any(cue in lowered for cue in cues)]
    if matches:
        return sorted(matches)
    if any(word in lowered for word in ["control", "pressure", "манипул", "давлен", "застав"]):
        return ["behavioral_pressure"]
    if any(word in lowered for word in ["withdraw", "avoid", "avoidance", "избега", "замыка"]):
        return ["avoidant_response"]
    return []


def _trigger_keys(sentence: str) -> list[str]:
    lowered = sentence.lower()
    return sorted({key for key, cues in _TRIGGER_TERMS.items() if any(cue in lowered for cue in cues)})


def _domain_node(domain_key: str) -> Node:
    name = domain_key.replace("_", " ").title()
    return Node(
        id=f"domain:{domain_key}",
        type="DOMAIN",
        name=name,
        description=f"{name} is the working domain that groups related concepts, patterns, examples, people, and micro-signals.",
        what_it_is=f"{name} is a practical knowledge area used to organize recurring human behavior and interpretation.",
        how_it_works="The domain groups concepts, patterns, signals, triggers, and people so related examples can be explored together.",
        how_to_recognize=f"If a note keeps returning to {name.lower()} themes, it belongs in this domain.",
        tags_json=json_list(["domain", domain_key]),
        logic_weight=0.85,
        relevance_weight=0.75,
    )


def _concept_node(domain_key: str, token: str) -> Node:
    name = token.replace("_", " ").title()
    return Node(
        id=f"concept:{domain_key}:{_normalize_slug(token)}",
        type="PATTERN",
        name=name,
        description=f"{name} is a reusable concept inside the {domain_key.replace('_', ' ')} domain.",
        what_it_is=f"{name} names an idea that helps classify behavior, motives, or recurring situations.",
        how_it_works=f"{name} works as a label that groups related patterns and examples under one meaning.",
        how_to_recognize=f"You recognize {name.lower()} when multiple notes keep pointing to the same recurring meaning.",
        tags_json=json_list([domain_key, "semantic_concept"]),
        logic_weight=0.8,
        relevance_weight=0.78,
    )


def _pattern_node(domain_key: str, pattern_key: str) -> Node:
    name = pattern_key.replace("_", " ").title()
    return Node(
        id=f"pattern:{domain_key}:{pattern_key}",
        type="PATTERN",
        name=name,
        description=f"{name} is a repeated behavioral move rather than a one-off event.",
        what_it_is=f"{name} is a repeatable way of acting, pressuring, avoiding, or reacting.",
        how_it_works="A pattern shows up when similar actions or phrases keep appearing across situations.",
        how_to_recognize="Look for repeated phrasing, repeated pressure, repeated avoidance, or the same trigger-response loop.",
        tags_json=json_list([domain_key, "pattern"]),
        behavior_patterns_json=json_list([name]),
        logic_weight=0.72,
        risk_weight=0.82,
        relevance_weight=0.74,
    )


def _trigger_node(domain_key: str, trigger_key: str) -> Node:
    name = trigger_key.replace("_", " ").title()
    return Node(
        id=f"trigger:{domain_key}:{trigger_key}",
        type="SIGNAL",
        name=name,
        description=f"{name} is a trigger that can set off a predictable reaction.",
        what_it_is=f"{name} is the condition or cue that tends to start a behavior pattern.",
        how_it_works="When the trigger appears, the same defensive or controlling move often follows.",
        how_to_recognize="Find the situation that keeps showing up right before the reaction.",
        tags_json=json_list([domain_key, "trigger_signal"]),
        triggers_json=json_list([name]),
        emotion_weight=0.68,
        risk_weight=0.86,
        relevance_weight=0.7,
    )


def _signal_node(domain_key: str, category_key: str, signal: dict[str, object]) -> Node:
    signal_key = str(signal["key"])
    label = str(signal["label"])
    description = str(signal["description"])
    cues = [str(item) for item in signal.get("cues") or []]
    category_label = _SIGNAL_CATEGORY_LABELS.get(category_key, category_key.replace("_", " ").title())
    return Node(
        id=f"signal:{domain_key}:{category_key}:{_normalize_slug(signal_key)}",
        type="SIGNAL",
        name=label,
        description=description,
        what_it_is=f"{category_label} extracted from a single sentence as a micro-level clue worth tracking.",
        how_it_works="Small linguistic and emotional details become explicit graph nodes so weak but repeated cues can accumulate into patterns.",
        how_to_recognize="Look for the exact wording, punctuation, or framing cues listed in this node.",
        examples_json=json_list(cues),
        tags_json=json_list([domain_key, "signal", category_key]),
        emotion_weight=0.74 if category_key in {"emotional_signals", "hidden_intent"} else 0.58,
        logic_weight=0.74 if category_key in {"semantic_meaning", "logical_structure"} else 0.52,
        risk_weight=0.76 if category_key in {"hidden_intent", "behavior_pattern_candidates"} else 0.5,
        relevance_weight=0.82,
        confidence=float(signal.get("confidence") or 0.7),
        plain_explanation=description,
    )


def _trait_node(domain_key: str, trait_key: str, label: str, description: str) -> Node:
    return Node(
        id=f"trait:{domain_key}:{trait_key}",
        type="TRAIT",
        name=label,
        description=description,
        what_it_is="A trait node stores a personality-style candidate inferred from repeated signals and patterns.",
        how_it_works="Traits are only candidates. They collect repeated evidence from signals, patterns, and examples rather than diagnosing a person from one line.",
        how_to_recognize="Treat the trait as stronger only when many examples keep supporting the same style candidate.",
        tags_json=json_list([domain_key, "trait"]),
        emotion_weight=0.84,
        relevance_weight=0.76,
        plain_explanation=description,
    )


def _profession_keys(sentence: str, domain_key: str) -> list[str]:
    lowered = sentence.lower()
    keys = [key for key, config in _PROFESSION_RULES.items() if any(keyword in lowered for keyword in config["keywords"])]
    if keys:
        return sorted(set(keys))
    if domain_key == "law":
        return ["law"]
    if domain_key == "psychology":
        return ["therapy"]
    return []


def _profession_node(profession_key: str) -> Node:
    config = _PROFESSION_RULES[profession_key]
    return Node(
        id=f"profession:{profession_key}",
        type="PROFESSION",
        name=str(config["label"]),
        description=f"{config['label']} is a professional logic domain with its own language, risk model, and response habits.",
        what_it_is="A profession node captures the role logic, domain pressure, and communication style that shape human reasoning.",
        how_it_works="Profession nodes help the system simulate not only what someone knows, but how their work changes what they notice and how they answer.",
        how_to_recognize="Use profession nodes when the person's way of speaking or deciding follows an occupational logic.",
        tags_json=json_list(["profession", profession_key]),
        logic_weight=0.82,
        risk_weight=0.66,
        relevance_weight=0.75,
    )


def _infer_speech_profile(sentence: str, professions: list[str], micro: dict[str, object]) -> tuple[dict[str, float], str]:
    lowered = sentence.lower()
    if professions:
        config = _PROFESSION_RULES[professions[0]]
        base = dict(config["speech"])
    else:
        base = {"formality": 0.45, "slang_level": 0.22, "directness": 0.55, "profanity_tolerance": 0.08}
    if '"' in sentence or "“" in sentence or "”" in sentence:
        base["directness"] = min(1.0, base["directness"] + 0.08)
    if "?" in sentence:
        base["directness"] = max(0.0, base["directness"] - 0.06)
    if any(word in lowered for word in ["damn", "hell", "черт", "блин"]):
        base["slang_level"] = min(1.0, base["slang_level"] + 0.22)
        base["profanity_tolerance"] = min(1.0, base["profanity_tolerance"] + 0.28)
    if any(word in lowered for word in ["please", "could you", "пожалуйста", "будьте добры"]):
        base["formality"] = min(1.0, base["formality"] + 0.12)
        base["directness"] = max(0.0, base["directness"] - 0.05)
    hidden_intents = list((micro or {}).get("hidden_intent") or [])
    temperament = "guarded"
    if any(str(item.get("key")) in {"compliance_pressure", "blame_deflection"} for item in hidden_intents):
        temperament = "controlling_defensive"
    elif any(str(item.get("key")) == "harmony_preservation" for item in hidden_intents):
        temperament = "conflict_avoidant"
    elif any(str(item.get("key")) == "reassurance_seeking" for item in hidden_intents):
        temperament = "approval_sensitive"
    return base, temperament


def _person_node(name: str, *, sentence: str, patterns: list[str], triggers: list[str], professions: list[str], micro: dict[str, object]) -> Node:
    behavior_patterns = [item.replace("_", " ") for item in patterns]
    trigger_names = [item.replace("_", " ") for item in triggers]
    speech_style, temperament = _infer_speech_profile(sentence, professions, micro)
    hidden_intents = [str(item.get("label") or item.get("key") or "") for item in list(micro.get("hidden_intent") or [])[:4]]
    emotion_signals = [str(item.get("label") or item.get("key") or "") for item in list(micro.get("emotional_signals") or [])[:4]]
    values = []
    lowered = sentence.lower()
    if "love" in lowered or "care" in lowered or "люб" in lowered:
        values.append("attachment")
    if "truth" in lowered or "honest" in lowered or "чест" in lowered:
        values.append("honesty")
    tolerance_threshold = 0.42 if "criticism" in triggers else 0.58
    return Node(
        id=f"person:{_normalize_slug(name)}",
        type="PERSON",
        name=name,
        description=f"{name} is a person node connected to observed examples, patterns, triggers, and personality-style candidates.",
        what_it_is="A person node stores observed behavior rather than an abstract label.",
        how_it_works="The person is linked to quoted examples, recurring patterns, triggers, micro-signals, and trait candidates gathered from text.",
        how_to_recognize="Use this node when the text clearly points to a named person or stable personality target.",
        tags_json=json_list(["person"]),
        background=f"Observed through text examples in the {professions[0] if professions else 'general'} context." if professions else "Observed through text examples.",
        profession=professions[0] if professions else "",
        speech_style_json=_json_object({"summary": temperament, **speech_style}),
        temperament=temperament,
        tolerance_threshold=tolerance_threshold,
        formality=float(speech_style["formality"]),
        slang_level=float(speech_style["slang_level"]),
        directness=float(speech_style["directness"]),
        profanity_tolerance=float(speech_style["profanity_tolerance"]),
        speech_patterns_json=json_list([sentence[:160]]),
        behavior_patterns_json=json_list(behavior_patterns),
        triggers_json=json_list(trigger_names),
        possible_intents_json=json_list(hidden_intents),
        emotion_signals_json=json_list(emotion_signals),
        values_json=json_list(values),
        emotion_weight=0.82,
        risk_weight=0.64,
        relevance_weight=0.88,
    )


def _example_node(domain_key: str, source_id: str, index: int, sentence: str, *, micro: dict[str, object]) -> Node:
    digest = hashlib.sha1(f"{source_id}:{index}:{sentence}".encode("utf-8")).hexdigest()[:12]
    hidden_intents = [str(item.get("label") or item.get("key") or "") for item in list(micro.get("hidden_intent") or [])[:5]]
    emotional = [str(item.get("label") or item.get("key") or "") for item in list(micro.get("emotional_signals") or [])[:5]]
    conflict_level = min(
        1.0,
        0.15
        + 0.18 * len(list(micro.get("behavior_pattern_candidates") or []))
        + 0.14 * len(list(micro.get("hidden_intent") or []))
        + (0.18 if any("pressure" in item.lower() for item in emotional + hidden_intents) else 0.0),
    )
    irony_probability = 0.0
    lowered = sentence.lower()
    if "..." in sentence or "?" in sentence and "!" in sentence:
        irony_probability += 0.18
    if any(word in lowered for word in ["yeah right", "sure", "конечно", "ага"]):
        irony_probability += 0.34
    return Node(
        id=f"example:{domain_key}:{digest}",
        type="EXAMPLE",
        name=(sentence[:72] + "...") if len(sentence) > 72 else sentence,
        description="Concrete sentence from source text, kept as a behavioral example.",
        what_it_is="An example preserves the concrete wording that demonstrates a concept, micro-signal, or behavioral pattern.",
        how_it_works="Examples anchor the graph in real language, so concepts do not drift into vague theory.",
        how_to_recognize="If the sentence shows a real phrase, behavior, or concrete situation, store it as an example.",
        examples_json=json_list([sentence]),
        tags_json=json_list([domain_key, "example"]),
        possible_intents_json=json_list(hidden_intents),
        emotion_signals_json=json_list(emotional),
        conflict_level=round(conflict_level, 6),
        irony_probability=round(min(irony_probability, 0.95), 6),
        relevance_weight=0.8,
        plain_explanation=sentence,
    )


def json_list(values: list[str]) -> str:
    return json.dumps([item for item in values if item], ensure_ascii=False)


def _json_object(value: dict[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _merge_edge(edges: dict[tuple[str, str, str], Edge], edge: Edge) -> None:
    key = (edge.src_id, edge.dst_id, edge.type)
    if key not in edges:
        edges[key] = edge
        return
    current = edges[key]
    merged_meta = {}
    if current.metadata_json and current.metadata_json != "{}":
        merged_meta.update(json.loads(current.metadata_json))
    if edge.metadata_json and edge.metadata_json != "{}":
        merged_meta.update(json.loads(edge.metadata_json))
    edges[key] = Edge(
        src_id=current.src_id,
        dst_id=current.dst_id,
        type=current.type,
        weight=current.weight + edge.weight,
        confidence=max(current.confidence, edge.confidence),
        metadata_json=_json_object(merged_meta) if merged_meta else "{}",
    )


def build_ingest_payload(source_id: str, text: str, *, top_tokens_per_sentence: int = 3) -> tuple[list[Node], list[Edge], list[Evidence]]:
    domain_key = _pick_domain(text)
    nodes_by_id: dict[str, Node] = {}
    edges_by_key: dict[tuple[str, str, str], Edge] = {}
    evidences: list[Evidence] = []
    nodes_by_id[f"domain:{domain_key}"] = _domain_node(domain_key)

    for index, sentence in enumerate(split_sentences(text)):
        patterns = _pattern_keys(sentence)
        triggers = _trigger_keys(sentence)
        people = _person_names(sentence)
        people_tokens = {_normalize_slug(name) for name in people}
        concepts = [token for token in _top_tokens(sentence, top_k=top_tokens_per_sentence + len(people_tokens)) if _normalize_slug(token) not in people_tokens][:top_tokens_per_sentence]
        micro = extract_micro_signals(sentence, concepts=concepts, patterns=patterns, triggers=triggers)
        professions = _profession_keys(sentence, domain_key)
        example = _example_node(domain_key, source_id, index, sentence, micro=micro)
        nodes_by_id[example.id] = example

        for concept in concepts:
            concept_node = _concept_node(domain_key, concept)
            nodes_by_id[concept_node.id] = concept_node
            _merge_edge(edges_by_key, Edge(src_id=f"domain:{domain_key}", dst_id=concept_node.id, type="RELATED_TO", weight=1.0, confidence=0.9))
            _merge_edge(edges_by_key, Edge(src_id=concept_node.id, dst_id=example.id, type="EXAMPLE_OF", weight=0.6, confidence=0.7))

        for pattern_key in patterns:
            pattern = _pattern_node(domain_key, pattern_key)
            nodes_by_id[pattern.id] = pattern
            _merge_edge(edges_by_key, Edge(src_id=f"domain:{domain_key}", dst_id=pattern.id, type="RELATED_TO", weight=1.0, confidence=0.9))
            _merge_edge(edges_by_key, Edge(src_id=pattern.id, dst_id=example.id, type="EXAMPLE_OF", weight=1.0, confidence=0.9))
            for concept in concepts[:2]:
                _merge_edge(
                    edges_by_key,
                    Edge(
                        src_id=f"concept:{domain_key}:{_normalize_slug(concept)}",
                        dst_id=pattern.id,
                        type="RELATED_TO",
                        weight=0.8,
                        confidence=0.75,
                    ),
                )

        for trigger_key in triggers:
            trigger = _trigger_node(domain_key, trigger_key)
            nodes_by_id[trigger.id] = trigger
            _merge_edge(edges_by_key, Edge(src_id=f"domain:{domain_key}", dst_id=trigger.id, type="RELATED_TO", weight=0.8, confidence=0.8))
            for pattern_key in patterns or ["behavioral_pressure"]:
                pattern_id = f"pattern:{domain_key}:{pattern_key}"
                if pattern_id not in nodes_by_id:
                    nodes_by_id[pattern_id] = _pattern_node(domain_key, pattern_key)
                _merge_edge(edges_by_key, Edge(src_id=trigger.id, dst_id=pattern_id, type="RELATED_TO", weight=0.9, confidence=0.8))

        for profession_key in professions:
            profession_node = _profession_node(profession_key)
            nodes_by_id[profession_node.id] = profession_node
            _merge_edge(edges_by_key, Edge(src_id=profession_node.id, dst_id=f"domain:{domain_key}", type="WORKS_IN_DOMAIN", weight=0.88, confidence=0.82))
            _merge_edge(edges_by_key, Edge(src_id=profession_node.id, dst_id=example.id, type="EXAMPLE_OF", weight=0.44, confidence=0.62))

        category_order = [
            "semantic_meaning",
            "emotional_signals",
            "logical_structure",
            "hidden_intent",
            "behavior_pattern_candidates",
        ]
        for category_key in category_order:
            for signal in micro.get(category_key, []):
                signal_node = _signal_node(domain_key, category_key, signal)
                nodes_by_id[signal_node.id] = signal_node
                _merge_edge(
                    edges_by_key,
                    Edge(
                        src_id=example.id,
                        dst_id=signal_node.id,
                        type="RELATED_TO",
                        weight=float(signal.get("confidence") or 0.7),
                        confidence=float(signal.get("confidence") or 0.7),
                        metadata_json=_json_object(
                            {
                                "category": category_key,
                                "signal_key": str(signal.get("key") or ""),
                                "cues": signal.get("cues") or [],
                            }
                        ),
                    ),
                )
                for concept in concepts[:2]:
                    _merge_edge(
                        edges_by_key,
                        Edge(
                            src_id=signal_node.id,
                            dst_id=f"concept:{domain_key}:{_normalize_slug(concept)}",
                            type="RELATED_TO",
                            weight=0.55,
                            confidence=float(signal.get("confidence") or 0.7),
                            metadata_json=_json_object({"category": category_key}),
                        ),
                    )

        behavior_signal_keys = {
            str(signal.get("key"))
            for signal in micro.get("behavior_pattern_candidates", [])
            if str(signal.get("key"))
        }
        candidate_pattern_keys = sorted((set(patterns) | (behavior_signal_keys & set(_PATTERN_RULES)) | (behavior_signal_keys & {"behavioral_pressure", "avoidant_response"})))
        hidden_intent_signal_ids = [
            f"signal:{domain_key}:hidden_intent:{_normalize_slug(str(signal.get('key')))}"
            for signal in micro.get("hidden_intent", [])
            if str(signal.get("key"))
        ]
        for pattern_key in candidate_pattern_keys:
            pattern_id = f"pattern:{domain_key}:{pattern_key}"
            if pattern_id not in nodes_by_id:
                nodes_by_id[pattern_id] = _pattern_node(domain_key, pattern_key)
            for signal in micro.get("behavior_pattern_candidates", []):
                signal_id = f"signal:{domain_key}:behavior_pattern_candidates:{_normalize_slug(str(signal.get('key')))}"
                if signal_id in nodes_by_id:
                    _merge_edge(
                        edges_by_key,
                        Edge(
                            src_id=signal_id,
                            dst_id=pattern_id,
                            type="RELATED_TO",
                            weight=float(signal.get("confidence") or 0.7),
                            confidence=float(signal.get("confidence") or 0.7),
                            metadata_json=_json_object({"basis": "behavior_pattern_candidate"}),
                        ),
                    )
            for hidden_signal_id in hidden_intent_signal_ids:
                if hidden_signal_id in nodes_by_id:
                    _merge_edge(
                        edges_by_key,
                        Edge(
                            src_id=hidden_signal_id,
                            dst_id=pattern_id,
                            type="RELATED_TO",
                            weight=0.62,
                            confidence=0.66,
                            metadata_json=_json_object({"basis": "hidden_intent"}),
                        ),
                    )

        for trait in micro.get("trait_candidates", []):
            trait_key = _normalize_slug(str(trait["key"]))
            trait_node = _trait_node(domain_key, trait_key, str(trait["label"]), str(trait["description"]))
            nodes_by_id[trait_node.id] = trait_node
            for category_key in ["hidden_intent", "emotional_signals", "logical_structure"]:
                signal_id = f"signal:{domain_key}:{category_key}:{_normalize_slug(str(trait.get('key')))}"
                if signal_id in nodes_by_id:
                    _merge_edge(
                        edges_by_key,
                        Edge(
                            src_id=signal_id,
                            dst_id=trait_node.id,
                            type="RELATED_TO",
                            weight=float(trait.get("confidence") or 0.66),
                            confidence=float(trait.get("confidence") or 0.66),
                            metadata_json=_json_object({"basis": "trait_candidate"}),
                        ),
                    )
            for pattern_key in patterns:
                _merge_edge(
                    edges_by_key,
                    Edge(
                        src_id=f"pattern:{domain_key}:{pattern_key}",
                        dst_id=trait_node.id,
                        type="RELATED_TO",
                        weight=0.45,
                        confidence=0.6,
                        metadata_json=_json_object({"basis": "pattern_to_trait_candidate"}),
                    ),
                )

        for person_name in people:
            person = _person_node(person_name, sentence=sentence, patterns=patterns, triggers=triggers, professions=professions, micro=micro)
            nodes_by_id[person.id] = person
            _merge_edge(
                edges_by_key,
                Edge(
                    src_id=person.id,
                    dst_id=example.id,
                    type="SAID_EXAMPLE",
                    weight=1.0,
                    confidence=0.9,
                    metadata_json=_json_object(
                        {
                            "tone": list(micro.get("emotional_signals") or [{}])[0].get("label") if list(micro.get("emotional_signals") or []) else "",
                            "context": domain_key,
                            "intent": list(micro.get("hidden_intent") or [{}])[0].get("label") if list(micro.get("hidden_intent") or []) else "",
                        }
                    ),
                ),
            )
            for profession_key in professions:
                _merge_edge(edges_by_key, Edge(src_id=person.id, dst_id=f"profession:{profession_key}", type="RELATED_TO", weight=0.72, confidence=0.74))
            for pattern_key in patterns:
                _merge_edge(edges_by_key, Edge(src_id=person.id, dst_id=f"pattern:{domain_key}:{pattern_key}", type="USES_PATTERN", weight=0.7, confidence=0.7))
            for trigger_key in triggers:
                _merge_edge(edges_by_key, Edge(src_id=person.id, dst_id=f"trigger:{domain_key}:{trigger_key}", type="RELATED_TO", weight=0.6, confidence=0.65))
            for trait in micro.get("trait_candidates", []):
                trait_id = f"trait:{domain_key}:{_normalize_slug(str(trait.get('key')))}"
                if trait_id in nodes_by_id:
                    _merge_edge(
                        edges_by_key,
                        Edge(
                            src_id=person.id,
                            dst_id=trait_id,
                            type="HAS_TRAIT",
                            weight=0.52,
                            confidence=float(trait.get("confidence") or 0.66),
                            metadata_json=_json_object({"basis": "observed_candidate_from_sentence"}),
                        ),
                    )
            for category_key in ["emotional_signals", "hidden_intent"]:
                for signal in micro.get(category_key, []):
                    signal_id = f"signal:{domain_key}:{category_key}:{_normalize_slug(str(signal.get('key')))}"
                    if signal_id in nodes_by_id:
                        _merge_edge(
                            edges_by_key,
                            Edge(
                                src_id=person.id,
                                dst_id=signal_id,
                                type="RELATED_TO",
                                weight=0.4,
                                confidence=float(signal.get("confidence") or 0.7),
                                metadata_json=_json_object({"basis": "personality_model_input"}),
                            ),
                        )

        evidence_target_ids = [example.id]
        evidence_target_ids.extend(f"pattern:{domain_key}:{pattern_key}" for pattern_key in candidate_pattern_keys)
        evidence_target_ids.extend(f"concept:{domain_key}:{_normalize_slug(concept)}" for concept in concepts[:2])
        evidence_target_ids.extend(
            f"signal:{domain_key}:{category_key}:{_normalize_slug(str(signal.get('key')))}"
            for category_key in category_order
            for signal in micro.get(category_key, [])
        )
        evidence_target_ids.extend(
            f"trait:{domain_key}:{_normalize_slug(str(signal.get('key')))}"
            for signal in micro.get("trait_candidates", [])
        )
        for edge in list(edges_by_key.values()):
            if edge.src_id in evidence_target_ids or edge.dst_id in evidence_target_ids:
                evidences.append(
                    Evidence(
                        edge_key=edge_key(edge.src_id, edge.type, edge.dst_id),
                        source_id=source_id,
                        snippet_text=sentence,
                        offset_start=0,
                        offset_end=len(sentence),
                    )
                )

    nodes = [nodes_by_id[node_id] for node_id in sorted(nodes_by_id)]
    edges = [edges_by_key[key] for key in sorted(edges_by_key)]
    dedup_evidence: dict[tuple[str, str, str, int, int], Evidence] = {}
    for item in evidences:
        key = (item.edge_key, item.source_id, item.snippet_text, item.offset_start, item.offset_end)
        dedup_evidence[key] = item
    return nodes, edges, [dedup_evidence[key] for key in sorted(dedup_evidence)]


def ingest_text(store: GraphStore, source_id: str, text: str, *, top_tokens_per_sentence: int = 3) -> dict[str, int | str]:
    nodes, edges, evidence = build_ingest_payload(source_id, text, top_tokens_per_sentence=top_tokens_per_sentence)
    stats = store.apply_batch(source=Source(source_id=source_id, raw_text=text), nodes=nodes, edges=edges, evidence=evidence)
    return {"source_id": source_id, **stats}


def main() -> None:
    import argparse

    from .config import default_settings

    parser = argparse.ArgumentParser(description="Ingest behavioral knowledge into the graph")
    parser.add_argument("--source", required=True, help="Path to input text file")
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    text = Path(args.source).read_text(encoding="utf-8")
    source_id = args.source_id or source_id_from_text(text)

    settings = default_settings(Path(__file__).resolve().parents[1])
    store = GraphStore(settings.db_path)
    try:
        result = ingest_text(store, source_id, text, top_tokens_per_sentence=args.top_k)
    finally:
        store.close()
    print(result)


if __name__ == "__main__":
    main()
