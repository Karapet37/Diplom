"""Graph-first workspace service for autonomous system simulation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping
import zlib

from src.autonomous_graph.api import GraphAPI, build_graph_engine_from_env
from src.living_system.core_engine import LivingSystemEngine
from src.web.autoruns_import import parse_autoruns_text
from src.web.client_introspection import build_client_profile


@dataclass(frozen=True)
class GraphMetrics:
    node_count: int
    edge_count: int
    relation_counts: dict[str, int]
    node_type_counts: dict[str, int]


def _profile_key_token(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).casefold()


_PROFILE_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "first_name": ("имя", "name", "first name", "firstname", "անուն"),
    "last_name": ("фамилия", "surname", "last name", "lastname", "ազգանուն"),
    "full_name": ("фио", "full name", "fullname", "անուն ազգանուն"),
    "date_of_birth": ("дата рождения", "date of birth", "birth date", "dob", "ծննդյան օր"),
    "age": ("возраст", "age", "տարիք"),
    "height_cm": ("рост", "height", "հասակ"),
    "weight_kg": ("вес", "weight", "քաշ"),
    "education": ("образование", "education", "կրթություն"),
    "employment": (
        "работа",
        "employment",
        "job",
        "occupation",
        "career",
        "աշխատանք",
    ),
    "job_status": (
        "должность",
        "позиция",
        "role",
        "status",
        "profession",
        "профессия",
        "պաշտոն",
    ),
    "company_name": ("компания", "company", "organization", "ընկերություն"),
    "preferences": ("предпочтения", "preferences", "likes", "նախընտրություններ"),
    "values": ("ценности", "values", "արժեքներ"),
    "views": ("взгляды", "beliefs", "views", "убеждения", "հայացքներ"),
    "fears": ("страхи", "fears", "worries", "վախեր"),
    "goals": ("цели", "goals", "aims", "նպատակներ"),
    "desires": ("желания", "desires", "wants", "ցանկություններ"),
    "traits": ("черты", "traits", "особенности", "personality", "հատկություններ"),
    "principles": ("принципы", "principles", "credo", "убеждения"),
    "opportunities": ("возможности", "opportunities", "chances", "опции"),
    "abilities": ("умения", "abilities", "ability", "способности", "skills"),
    "access": ("доступ", "access", "permissions", "право доступа"),
    "knowledge": ("знания", "knowledge", "expertise", "опыт"),
    "assets": ("имущество", "assets", "property", "ресурсы", "ownership"),
    "reminders": ("напоминания", "reminders", "հիշեցումներ"),
    "known_languages": ("языки", "languages", "known languages", "լեզուներ"),
    "primary_language": ("основной язык", "primary language", "основний язык", "հիմնական լեզու"),
}

_PROFILE_LIST_KEYS: set[str] = {
    "education",
    "preferences",
    "values",
    "views",
    "fears",
    "goals",
    "desires",
    "traits",
    "principles",
    "opportunities",
    "abilities",
    "access",
    "knowledge",
    "assets",
    "reminders",
    "known_languages",
}

_PROFILE_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical_key, _aliases in _PROFILE_KEY_ALIASES.items():
    _PROFILE_ALIAS_TO_CANONICAL[_profile_key_token(_canonical_key)] = _canonical_key
    for _alias in _aliases:
        _token = _profile_key_token(_alias)
        if _token:
            _PROFILE_ALIAS_TO_CANONICAL.setdefault(_token, _canonical_key)

_USER_DIMENSION_BINDINGS: dict[str, dict[str, str]] = {
    "fears": {
        "node_type": "fear",
        "relation_type": "has_fear",
        "display": "Fear",
    },
    "desires": {
        "node_type": "desire",
        "relation_type": "desires",
        "display": "Desire",
    },
    "goals": {
        "node_type": "goal",
        "relation_type": "pursues_goal",
        "display": "Goal",
    },
    "principles": {
        "node_type": "principle",
        "relation_type": "holds_principle",
        "display": "Principle",
    },
    "opportunities": {
        "node_type": "opportunity",
        "relation_type": "sees_opportunity",
        "display": "Opportunity",
    },
    "abilities": {
        "node_type": "ability",
        "relation_type": "has_ability",
        "display": "Ability",
    },
    "access": {
        "node_type": "access",
        "relation_type": "has_access",
        "display": "Access",
    },
    "knowledge": {
        "node_type": "knowledge_area",
        "relation_type": "knows",
        "display": "Knowledge",
    },
    "assets": {
        "node_type": "asset",
        "relation_type": "owns_asset",
        "display": "Asset",
    },
    "values": {
        "node_type": "value",
        "relation_type": "values",
        "display": "Value",
    },
    "preferences": {
        "node_type": "preference",
        "relation_type": "prefers",
        "display": "Preference",
    },
    "traits": {
        "node_type": "trait",
        "relation_type": "has_trait",
        "display": "Trait",
    },
}

_AUTORUNS_RISK_LOCATION_HINTS: tuple[str, ...] = (
    "runonce",
    "currentversion\\run",
    "scheduled task",
    "service",
    "drivers",
    "boot",
    "winlogon",
)

_PROFILE_ALIAS_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = []
for _canonical_key, _aliases in _PROFILE_KEY_ALIASES.items():
    for _alias in _aliases:
        _PROFILE_ALIAS_PATTERNS.append(
            (
                _canonical_key,
                _alias,
                re.compile(
                    rf"(?<!\w){re.escape(_alias)}(?!\w)\s*[:=]",
                    flags=re.IGNORECASE | re.UNICODE,
                ),
            )
        )
_PROFILE_ALIAS_PATTERNS.sort(key=lambda item: len(item[1]), reverse=True)

_LIST_SPLIT_RE = re.compile(r"\s*[,;\n|/]\s*")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
_EMPLOYMENT_SCORE_RE = re.compile(
    r"(?:importance|score|weight|вес|важность)\s*[:=]?\s*([-+]?\d+(?:[.,]\d+)?)",
    flags=re.IGNORECASE | re.UNICODE,
)
_EMPLOYMENT_AT_RE = re.compile(
    r"^(?P<status>.+?)\s+(?:@|at|в|во|in)\s+(?P<company>.+)$",
    flags=re.IGNORECASE | re.UNICODE,
)
_EMPLOYMENT_STATUS_FIELD_RE = re.compile(
    r"(?:status|role|должность|позиция|профессия|պաշտոն)\s*[:=]\s*([^,;]+)",
    flags=re.IGNORECASE | re.UNICODE,
)
_EMPLOYMENT_COMPANY_FIELD_RE = re.compile(
    r"(?:company|компания|organization|организация|ընկերություն)\s*[:=]\s*([^,;]+)",
    flags=re.IGNORECASE | re.UNICODE,
)

_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bmy name is\s+(?P<first>[A-Za-z][A-Za-z'`-]{1,40})(?:\s+(?P<last>[A-Za-z][A-Za-z'`-]{1,40}))?",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bменя зовут\s+(?P<first>[A-Za-zА-Яа-яЁё][\w'`-]{1,40})(?:\s+(?P<last>[A-Za-zА-Яа-яЁё][\w'`-]{1,40}))?",
        flags=re.IGNORECASE | re.UNICODE,
    ),
    re.compile(
        r"իմ անունը\s+(?P<first>[^\s,.;:]{2,40})(?:\s+(?P<last>[^\s,.;:]{2,40}))?",
        flags=re.IGNORECASE | re.UNICODE,
    ),
)

_WORK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bI\s+work\s+as\s+(?P<status>[^.,;\n]{2,50})\s+(?:at|in)\s+(?P<company>[^.,;\n]{2,80})",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bI\s+work\s+at\s+(?P<company>[^.,;\n]{2,80})",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bя\s+работаю\s+как\s+(?P<status>[^.,;\n]{2,50})\s+в\s+(?P<company>[^.,;\n]{2,80})",
        flags=re.IGNORECASE | re.UNICODE,
    ),
    re.compile(
        r"\bя\s+работаю\s+в\s+(?P<company>[^.,;\n]{2,80})",
        flags=re.IGNORECASE | re.UNICODE,
    ),
    re.compile(
        r"աշխատում եմ\s+(?P<company>[^.,;\n]{2,80})",
        flags=re.IGNORECASE | re.UNICODE,
    ),
)

_LLM_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", flags=re.IGNORECASE)
_LLM_JSON_OBJECT_RE = re.compile(r"(\{[\s\S]*\})", flags=re.MULTILINE)
_LLM_JSON_ARRAY_RE = re.compile(r"(\[[\s\S]*\])", flags=re.MULTILINE)

_LEVEL_WEIGHT_HINTS: dict[str, float] = {
    "native": 1.0,
    "fluent": 0.9,
    "advanced": 0.85,
    "intermediate": 0.65,
    "beginner": 0.4,
    "novice": 0.3,
    "expert": 0.95,
    "senior": 0.85,
    "junior": 0.55,
    "c2": 0.95,
    "c1": 0.85,
    "b2": 0.72,
    "b1": 0.6,
    "a2": 0.45,
    "a1": 0.32,
}

_RELATION_TYPE_HINTS: tuple[str, ...] = (
    "likes",
    "dislikes",
    "fears",
    "avoids",
    "respects",
    "interested_in",
    "influenced_by",
    "prefers",
    "trusts",
    "distrusts",
    "motivated_by",
    "stressed_by",
    "associated_with",
    "positive_association",
    "negative_association",
    "neutral_association",
)

_DEBATE_ROLES_ALLOWED: tuple[str, ...] = (
    "general",
    "analyst",
    "creative",
    "planner",
    "coder_architect",
    "coder_reviewer",
    "coder_refactor",
    "coder_debug",
)

_DEBATE_DEFAULT_ROLES: dict[str, str] = {
    "proposer": "creative",
    "critic": "analyst",
    "judge": "planner",
}

_HALLUCINATION_BRANCH_NAME = "hallucination_hunter"

_HALLUCINATION_SEVERITY_ALLOWED: tuple[str, ...] = (
    "low",
    "medium",
    "high",
    "critical",
)

_ARCHIVE_UPDATE_BRANCH_NAME = "archive_verified_updates"

_ARCHIVE_UPDATE_OPERATIONS_ALLOWED: tuple[str, ...] = (
    "upsert",
    "append",
    "correct",
    "deprecate",
)

_ARCHIVE_VERIFICATION_MODES_ALLOWED: tuple[str, ...] = (
    "strict",
    "balanced",
)

_PERSONALIZATION_STYLE_ALLOWED: tuple[str, ...] = (
    "adaptive",
    "concise",
    "balanced",
    "deep",
)

_PERSONALIZATION_DEPTH_ALLOWED: tuple[str, ...] = (
    "quick",
    "balanced",
    "deep",
)

_PERSONALIZATION_RISK_ALLOWED: tuple[str, ...] = (
    "low",
    "medium",
    "high",
)

_PERSONALIZATION_TONE_ALLOWED: tuple[str, ...] = (
    "neutral",
    "direct",
    "empathetic",
    "challenging",
)

_GLOBAL_CONCEPT_CATALOG: dict[int, dict[str, Any]] = {
    202: {
        "name": "music",
        "aliases": ["музыка", "երաժշտություն", "音樂", "música", "música"],
    },
    203: {
        "name": "chess",
        "aliases": ["шахматы", "շախմատ", "国际象棋", "ajedrez", "xadrez"],
    },
    204: {
        "name": "programming",
        "aliases": ["программирование", "ծրագրավորում", "编程", "programación", "programação"],
    },
    205: {
        "name": "business",
        "aliases": ["бизнес", "բիզնես", "商业", "negocio", "negócio"],
    },
    206: {
        "name": "freedom",
        "aliases": ["свобода", "ազատություն", "自由", "libertad", "liberdade"],
    },
    207: {
        "name": "jazz",
        "aliases": ["джаз", "ջազ", "爵士乐"],
    },
    208: {
        "name": "history",
        "aliases": ["история", "պատմություն", "历史", "historia"],
    },
    209: {
        "name": "film",
        "aliases": ["фильм", "կինո", "电影", "película", "filme"],
    },
}

_FOUNDATIONAL_DOMAIN_GRAPH: dict[str, tuple[str, ...]] = {
    "Mathematics": ("Algebra", "Calculus", "Geometry", "Probability", "Logic"),
    "Physics": ("Mechanics", "Energy", "Thermodynamics", "Relativity", "Quantum Mechanics"),
    "Biology": ("Cell", "Genetics", "Evolution", "Ecology", "Physiology"),
    "Computer Science": ("Algorithms", "Data Structures", "Databases", "Networks", "Machine Learning"),
    "Philosophy": ("Ethics", "Epistemology", "Metaphysics", "Ontology", "Reason"),
    "Psychology": ("Cognition", "Memory", "Emotion", "Behavior", "Motivation"),
    "Sociology": ("Society", "Culture", "Institutions", "Norms", "Social Structure"),
    "Theology": ("Faith", "Doctrine", "Hermeneutics", "Comparative Religion", "Theodicy"),
    "Economics": ("Market", "Demand", "Supply", "Capital", "Trade"),
    "Linguistics": ("Phonology", "Morphology", "Syntax", "Semantics", "Pragmatics"),
    "Arts": ("Music", "Film", "Painting", "Poetry", "Theater"),
}

_FOUNDATIONAL_DOMAIN_RELATIONS: tuple[tuple[str, str, str, float], ...] = (
    ("Mathematics", "Computer Science", "enables", 0.95),
    ("Mathematics", "Physics", "supports", 0.9),
    ("Psychology", "Sociology", "influences", 0.82),
    ("Linguistics", "Computer Science", "enables", 0.76),
    ("Philosophy", "Theology", "interacts_with", 0.67),
    ("Economics", "Sociology", "depends_on", 0.72),
    ("Music", "Psychology", "affects", 0.73),
)

_FOUNDATIONAL_DOMAIN_DETAILS: dict[str, dict[str, Any]] = {
    "Mathematics": {
        "description": (
            "Foundational domain: Mathematics. Studies quantity, structure, change, space, uncertainty and proof; "
            "provides formal language for precise reasoning across sciences and engineering."
        ),
        "history_intro": (
            "From Babylonian and Egyptian arithmetic through Greek axiomatic geometry, mathematics expanded with "
            "algebra, calculus, probability, modern logic and abstract structures in the 19th-20th centuries."
        ),
        "connections": [
            "Physics",
            "Computer Science",
            "Economics",
            "Philosophy (logic)",
            "Engineering",
        ],
        "usage": [
            "Modeling and simulation",
            "Cryptography and security",
            "Data analysis and statistics",
            "Optimization and control systems",
            "Machine learning foundations",
        ],
    },
    "Physics": {
        "description": (
            "Foundational domain: Physics. Explains matter, energy, motion and interactions from subatomic to "
            "cosmological scales using theory, experiment and mathematical models."
        ),
        "history_intro": (
            "Classical mechanics and electromagnetism were unified in early modern science; relativity and quantum "
            "theory transformed 20th-century understanding of space, time and microscopic behavior."
        ),
        "connections": [
            "Mathematics",
            "Engineering",
            "Computer Science (simulation)",
            "Chemistry",
            "Philosophy of science",
        ],
        "usage": [
            "Energy systems and electronics",
            "Materials and semiconductor design",
            "Medical imaging technologies",
            "Space and satellite systems",
            "Climate and environmental modeling",
        ],
    },
    "Biology": {
        "description": (
            "Foundational domain: Biology. Studies living systems, from molecular mechanisms and cells to organisms, "
            "ecosystems and evolution."
        ),
        "history_intro": (
            "Natural history evolved into modern biology through cell theory, Darwinian evolution, Mendelian genetics "
            "and molecular biology, then expanded with genomics and systems biology."
        ),
        "connections": [
            "Chemistry",
            "Medicine",
            "Psychology and neuroscience",
            "Ecology and environmental science",
            "Computer Science (bioinformatics)",
        ],
        "usage": [
            "Healthcare and diagnostics",
            "Drug discovery and biotechnology",
            "Agriculture and food systems",
            "Conservation and ecosystem management",
            "Public health planning",
        ],
    },
    "Computer Science": {
        "description": (
            "Foundational domain: Computer Science. Studies computation, algorithms, data, software systems and "
            "machine intelligence."
        ),
        "history_intro": (
            "From formal computability theory and early programmable machines, the field expanded into operating systems, "
            "networks, databases, software engineering, AI and modern distributed platforms."
        ),
        "connections": [
            "Mathematics",
            "Linguistics (NLP)",
            "Psychology (cognition models)",
            "Economics (digital markets)",
            "Engineering",
        ],
        "usage": [
            "Software and web platforms",
            "Cybersecurity and infrastructure",
            "Data platforms and analytics",
            "Automation and robotics",
            "AI systems and decision support",
        ],
    },
    "Philosophy": {
        "description": (
            "Foundational domain: Philosophy. Examines knowledge, reality, reasoning, ethics and meaning; clarifies "
            "assumptions and conceptual frameworks behind all disciplines."
        ),
        "history_intro": (
            "Developed from ancient Greek, Indian and Chinese traditions, then advanced through medieval scholasticism, "
            "modern rationalism/empiricism and contemporary analytic and continental schools."
        ),
        "connections": [
            "Theology",
            "Mathematics (logic and foundations)",
            "Psychology (mind and cognition)",
            "Sociology (ethics and institutions)",
            "Computer Science (AI ethics)",
        ],
        "usage": [
            "Ethical decision frameworks",
            "Argument analysis and critical thinking",
            "Policy and governance principles",
            "Scientific methodology critique",
            "AI alignment and responsible design",
        ],
    },
    "Psychology": {
        "description": (
            "Foundational domain: Psychology. Studies behavior, cognition, emotion, motivation and mental health at "
            "individual and group levels."
        ),
        "history_intro": (
            "Emerging from philosophy and physiology, psychology became an experimental science in the 19th century and "
            "later integrated behaviorist, cognitive, social and clinical approaches."
        ),
        "connections": [
            "Biology and neuroscience",
            "Sociology",
            "Linguistics",
            "Economics (behavioral economics)",
            "Computer Science (human-computer interaction)",
        ],
        "usage": [
            "Mental health interventions",
            "Education and learning design",
            "Workplace and team performance",
            "Product UX and behavior modeling",
            "Personal development and coaching",
        ],
    },
    "Sociology": {
        "description": (
            "Foundational domain: Sociology. Studies social structures, institutions, norms and collective behavior in "
            "communities and societies."
        ),
        "history_intro": (
            "Formed in the 19th century as industrialization changed social life; major traditions include structural, "
            "interpretive and critical approaches to institutions and inequality."
        ),
        "connections": [
            "Psychology",
            "Economics",
            "Political science",
            "Anthropology",
            "Linguistics (discourse and identity)",
        ],
        "usage": [
            "Public policy analysis",
            "Community and organizational design",
            "Risk and inequality assessment",
            "Media and culture research",
            "Social impact evaluation",
        ],
    },
    "Theology": {
        "description": (
            "Foundational domain: Theology. Studies religious beliefs, doctrines, interpretation traditions and "
            "spiritual worldviews."
        ),
        "history_intro": (
            "Built through scriptural interpretation and philosophical reflection in Abrahamic and other traditions; "
            "modern theology engages comparative religion and contemporary ethical questions."
        ),
        "connections": [
            "Philosophy",
            "History",
            "Sociology of religion",
            "Linguistics (text interpretation)",
            "Ethics",
        ],
        "usage": [
            "Moral and spiritual reflection",
            "Comparative religion studies",
            "Cultural and historical interpretation",
            "Dialogue across worldviews",
            "Community guidance frameworks",
        ],
    },
    "Economics": {
        "description": (
            "Foundational domain: Economics. Studies production, exchange, incentives, resource allocation and market "
            "dynamics under scarcity."
        ),
        "history_intro": (
            "From classical political economy through marginal analysis and macroeconomics, the field diversified into "
            "game theory, econometrics, behavioral and institutional economics."
        ),
        "connections": [
            "Mathematics and statistics",
            "Sociology",
            "Political science",
            "Psychology (decision making)",
            "Computer Science (platform economics)",
        ],
        "usage": [
            "Business strategy and forecasting",
            "Policy and regulation analysis",
            "Market and pricing models",
            "Development and welfare planning",
            "Resource optimization decisions",
        ],
    },
    "Linguistics": {
        "description": (
            "Foundational domain: Linguistics. Studies language structure and meaning across sounds, words, "
            "grammar and context; history spans ancient grammar traditions to modern structural, "
            "cognitive and computational approaches."
        ),
        "history_intro": (
            "Early foundations appeared in Panini's grammar and Greek philology; modern linguistics expanded in "
            "the 19th-20th centuries via comparative, structural and generative schools."
        ),
        "connections": [
            "Cognitive science",
            "Psychology",
            "Computer Science (NLP)",
            "Philosophy of language",
            "Sociology",
        ],
        "usage": [
            "Natural language processing",
            "Education and language learning",
            "Speech technologies",
            "Cross-cultural communication",
            "Knowledge representation",
        ],
    },
    "Arts": {
        "description": (
            "Foundational domain: Arts. Studies and practices creative expression through sound, image, movement, text "
            "and performance, shaping symbolic meaning and human experience."
        ),
        "history_intro": (
            "From prehistoric visual symbols and oral performance to classical, modern and digital art forms, artistic "
            "practice has co-evolved with technologies, institutions and cultural identities."
        ),
        "connections": [
            "Psychology (emotion and perception)",
            "Sociology (culture and identity)",
            "Linguistics (poetics and narrative)",
            "Technology (digital media)",
            "Philosophy (aesthetics)",
        ],
        "usage": [
            "Education and cultural literacy",
            "Therapy and wellbeing support",
            "Creative industries and media",
            "Communication and storytelling",
            "Public space and community identity",
        ],
    },
}

_DAILY_GOAL_HINTS: tuple[str, ...] = (
    "goal",
    "goals",
    "цель",
    "цели",
    "хочу",
    "plan",
    "план",
    "intend",
    "надо",
    "нужно",
    "aim",
)
_DAILY_PROBLEM_HINTS: tuple[str, ...] = (
    "problem",
    "issue",
    "risk",
    "stress",
    "проблем",
    "сложно",
    "трудно",
    "боюсь",
    "ошибка",
    "устал",
    "anxiety",
    "blocked",
)
_DAILY_WIN_HINTS: tuple[str, ...] = (
    "done",
    "completed",
    "finished",
    "achieved",
    "сделал",
    "завершил",
    "получилось",
    "удачно",
    "progress",
    "win",
)
_DAILY_POSITIVE_HINTS: tuple[str, ...] = (
    "calm",
    "focused",
    "good",
    "great",
    "energized",
    "доволен",
    "спокойно",
    "хорошо",
    "продуктивно",
)
_DAILY_NEGATIVE_HINTS: tuple[str, ...] = (
    "tired",
    "burnout",
    "anxious",
    "stuck",
    "bad",
    "недоволен",
    "устал",
    "выгорел",
    "тревожно",
    "плохо",
)


def _catalog_token(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).casefold()


_CONCEPT_ALIAS_TO_ID: dict[str, int] = {}
for _cid, _entry in _GLOBAL_CONCEPT_CATALOG.items():
    _raw_aliases = [str(_entry.get("name", "") or "").strip()] + [
        str(item or "").strip() for item in (_entry.get("aliases") or [])
    ]
    for _alias in _raw_aliases:
        _token = _catalog_token(_alias)
        if _token:
            _CONCEPT_ALIAS_TO_ID.setdefault(_token, int(_cid))

PROFILE_PROMPT_TEMPLATE = (
    "You are a structured JSON generator for a knowledge graph.\n"
    "Input: free-form text describing a person and surroundings.\n"
    "Output: ONLY valid JSON. No markdown, no comments, no extra text.\n"
    "Requirements:\n"
    "1) Build a person node.\n"
    "2) Build concept relations using existing global concept IDs when known.\n"
    "3) Do not redefine known concepts, only reference concept_id.\n"
    "4) If concept is not in global catalog, generate temporary concept_id and set new_concept=true.\n"
    "5) Every relation must contain relation_type, embedding_vector (float array), confidence and details.\n"
    "6) Use psychologically meaningful relation types beyond like/dislike/fear.\n"
    "Allowed relation type examples: {relation_types}\n"
    "Global concept catalog (ID -> concept):\n"
    "{concept_catalog}\n"
    "JSON schema:\n"
    "{\n"
    '  "person": {\n'
    '    "id": 0,\n'
    '    "first_name": "string",\n'
    '    "last_name": "string",\n'
    '    "bio": "string",\n'
    '    "birth_date": "YYYY-MM-DD"\n'
    "  },\n"
    '  "concept_relations": [\n'
    "    {\n"
    '      "concept_id": 0,\n'
    '      "concept_name": "string",\n'
    '      "new_concept": false,\n'
    '      "relation_type": "string",\n'
    '      "embedding_vector": [0.0, 0.0, 0.0, 0.0],\n'
    '      "confidence": 0.0,\n'
    '      "details": {"additional_info": "string"}\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Constraints:\n"
    "- Keep concept_relations present even when empty.\n"
    "- confidence in [0,1].\n"
    "- embedding_vector must be numeric floats.\n"
    "- Use existing concept_id from catalog when concept is known.\n"
    "Entity type hint: {entity_type_hint}\n"
    "User text:\n"
    "{text}\n"
    "JSON:"
)


class GraphWorkspaceService:
    """High-level graph workspace facade for web controllers."""

    def __init__(
        self,
        *,
        use_env_adapter: bool = True,
        profile_llm_fn: Callable[[str], str] | None = None,
        role_llm_resolver: Callable[[str], Callable[[str], str] | None] | None = None,
        model_llm_resolver: Callable[[str], Callable[[str], str] | None] | None = None,
        enable_living_system: bool = True,
        living_system_db_path: str = "data/living_system.db",
        workspace_root: str = ".",
    ):
        engine = build_graph_engine_from_env() if use_env_adapter else None
        self.api = GraphAPI(engine)
        self.profile_llm_fn = profile_llm_fn if profile_llm_fn is not None else self._build_profile_llm_fn()
        self.role_llm_resolver = (
            role_llm_resolver if role_llm_resolver is not None else self._build_role_llm_resolver()
        )
        self.model_llm_resolver = (
            model_llm_resolver if model_llm_resolver is not None else self._build_model_llm_resolver()
        )
        self.living_system = (
            LivingSystemEngine(
                db_path=living_system_db_path,
                workspace_root=workspace_root,
                prompt_llm_fn=self.profile_llm_fn,
            )
            if enable_living_system
            else None
        )
        self._bootstrap_runtime_graph()

    @staticmethod
    def _build_profile_llm_fn() -> Callable[[str], str] | None:
        try:
            from src.utils.local_llm_provider import build_local_llm_fn
        except Exception:
            return None
        try:
            return build_local_llm_fn()
        except Exception:
            return None

    @staticmethod
    def _build_role_llm_resolver() -> Callable[[str], Callable[[str], str] | None] | None:
        try:
            from src.utils.local_llm_provider import build_role_llm_fn
        except Exception:
            return None
        return build_role_llm_fn

    @staticmethod
    def _build_model_llm_resolver() -> Callable[[str], Callable[[str], str] | None] | None:
        try:
            from src.utils.local_llm_provider import build_model_llm_fn
        except Exception:
            return None
        return build_model_llm_fn

    @staticmethod
    def _env_flag(name: str, default: bool = True) -> bool:
        raw = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
        if not raw:
            return bool(default)
        return raw not in {"0", "false", "no", "off"}

    def _bootstrap_runtime_graph(self) -> None:
        """Initialize baseline graph and foundational SQL knowledge on startup."""
        should_seed = self._env_flag("AUTOGRAPH_BOOTSTRAP_FOUNDATION", True)
        if should_seed and not self.api.engine.store.nodes:
            self.seed_foundational_graph()
            try:
                self.api.persist()
            except Exception:
                pass

        if self.living_system is None:
            return
        if not self._env_flag("AUTOGRAPH_BOOTSTRAP_LIVING_FOUNDATION", True):
            return
        try:
            counts = self.living_system.store.table_counts()
            if int(counts.get("nodes", 0)) > 0:
                return
            self.living_knowledge_initialize(
                {
                    "user_id": "foundation_user",
                    "display_name": "Foundation User",
                    "language": "en",
                    "branch_id": "foundation",
                    "apply_changes": True,
                }
            )
        except Exception:
            # Keep startup resilient even when living layer is unavailable.
            return

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _to_state(payload: Mapping[str, Any] | None) -> dict[str, float]:
        if not payload:
            return {}
        out: dict[str, float] = {}
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                out[str(key)] = float(value)
            else:
                try:
                    out[str(key)] = float(value)
                except Exception:
                    continue
        return out

    @staticmethod
    def _normalize_edge_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "from_node": int(payload.get("from_node")),
            "to_node": int(payload.get("to_node")),
            "relation_type": str(payload.get("relation_type", "") or "").strip(),
            "weight": float(payload.get("weight", 1.0) or 1.0),
            "direction": str(payload.get("direction", "directed") or "directed"),
            "logic_rule": str(payload.get("logic_rule", "explicit") or "explicit"),
        }

    @staticmethod
    def _safe_json_loads(value: Any, default: Any):
        if isinstance(value, (dict, list)):
            return value
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        try:
            parsed = json.loads(text)
        except Exception:
            return default
        return parsed

    @staticmethod
    def _clean_field_value(value: str) -> str:
        cleaned = str(value or "").strip()
        cleaned = cleaned.strip(" \t\r\n,;")
        if cleaned.startswith("{") and cleaned.endswith("}"):
            cleaned = cleaned[1:-1].strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1].strip()
        return cleaned.strip(" \t\r\n,;")

    @staticmethod
    def _split_list_values(value: str) -> list[str]:
        cleaned = GraphWorkspaceService._clean_field_value(value)
        if not cleaned:
            return []
        parts = [
            token.strip(" \t\r\n,.;")
            for token in _LIST_SPLIT_RE.split(cleaned)
            if token.strip(" \t\r\n,.;")
        ]
        return parts

    @staticmethod
    def _extract_number(value: str) -> float | None:
        match = _NUMBER_RE.search(str(value or ""))
        if not match:
            return None
        token = match.group(0).replace(",", ".")
        try:
            return float(token)
        except Exception:
            return None

    @staticmethod
    def _sanitize_company_name(value: str) -> str:
        company = str(value or "").strip(" \t\r\n,.;")
        if not company:
            return ""
        lowered = company.casefold()
        splitters = (" and ", " и ", " ու ", " with ", " где ", " which ")
        for splitter in splitters:
            idx = lowered.find(splitter)
            if idx > 0:
                company = company[:idx].strip(" \t\r\n,.;")
                break
        return company

    @staticmethod
    def _parse_structured_profile(text: str) -> dict[str, list[str]]:
        source = str(text or "")
        if not source.strip():
            return {}

        matches: list[tuple[int, int, str]] = []
        for canonical_key, _alias, pattern in _PROFILE_ALIAS_PATTERNS:
            for match in pattern.finditer(source):
                matches.append((match.start(), match.end(), canonical_key))

        if not matches:
            return {}

        matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        normalized: list[tuple[int, int, str]] = []
        for item in matches:
            if normalized and normalized[-1][0] == item[0]:
                continue
            normalized.append(item)

        out: dict[str, list[str]] = {}
        for idx, (_, end, key) in enumerate(normalized):
            next_start = normalized[idx + 1][0] if idx + 1 < len(normalized) else len(source)
            value = GraphWorkspaceService._clean_field_value(source[end:next_start])
            if not value:
                continue
            out.setdefault(key, []).append(value)
        return out

    @staticmethod
    def _parse_employment_free_text(text: str) -> list[dict[str, Any]]:
        source = str(text or "").strip()
        if not source:
            return []

        chunks = [
            item.strip(" \t\r\n,;")
            for item in re.split(r"[;\n]+", source)
            if item.strip(" \t\r\n,;")
        ]
        out: list[dict[str, Any]] = []
        for chunk in chunks:
            importance_score = 1.0
            score_match = _EMPLOYMENT_SCORE_RE.search(chunk)
            if score_match:
                score = GraphWorkspaceService._extract_number(score_match.group(1))
                if score is not None:
                    importance_score = max(0.0, min(1.0, float(score)))
                chunk = _EMPLOYMENT_SCORE_RE.sub("", chunk).strip(" \t\r\n,;")

            status = ""
            company_name = ""
            at_match = _EMPLOYMENT_AT_RE.match(chunk)
            if at_match:
                status = str(at_match.group("status") or "").strip(" \t\r\n,;")
                company_name = GraphWorkspaceService._sanitize_company_name(
                    str(at_match.group("company") or "").strip(" \t\r\n,;")
                )
            else:
                status_match = _EMPLOYMENT_STATUS_FIELD_RE.search(chunk)
                company_match = _EMPLOYMENT_COMPANY_FIELD_RE.search(chunk)
                if status_match:
                    status = str(status_match.group(1) or "").strip(" \t\r\n,;")
                if company_match:
                    company_name = GraphWorkspaceService._sanitize_company_name(
                        str(company_match.group(1) or "").strip(" \t\r\n,;")
                    )
                if not status:
                    status = chunk.strip(" \t\r\n,;")

            if not status and not company_name:
                continue
            out.append(
                {
                    "status": status,
                    "importance_score": max(0.0, min(1.0, float(importance_score))),
                    "company_name": company_name,
                    "company_attributes": {},
                }
            )
        return out

    @staticmethod
    def _infer_name_from_text(text: str) -> tuple[str, str]:
        source = str(text or "")
        for pattern in _NAME_PATTERNS:
            match = pattern.search(source)
            if not match:
                continue
            first_name = str(match.groupdict().get("first", "") or "").strip(" \t\r\n,.;")
            last_name = str(match.groupdict().get("last", "") or "").strip(" \t\r\n,.;")
            if first_name:
                return first_name, last_name
        return "", ""

    @staticmethod
    def _infer_employment_from_text(text: str) -> list[dict[str, Any]]:
        source = str(text or "")
        for pattern in _WORK_PATTERNS:
            match = pattern.search(source)
            if not match:
                continue
            status = str(match.groupdict().get("status", "") or "").strip(" \t\r\n,.;")
            company_name = GraphWorkspaceService._sanitize_company_name(
                str(match.groupdict().get("company", "") or "").strip(" \t\r\n,.;")
            )
            if not status and not company_name:
                continue
            return [
                {
                    "status": status or "employee",
                    "importance_score": 0.8,
                    "company_name": company_name,
                    "company_attributes": {},
                }
            ]
        return []

    @staticmethod
    def _normalize_token(value: Any) -> str:
        return " ".join(str(value or "").strip().split()).casefold()

    @staticmethod
    def _to_list_of_strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [item for item in GraphWorkspaceService._split_list_values(value) if item]
        if not isinstance(value, (list, tuple, set)):
            return []
        out: list[str] = []
        for row in value:
            text = " ".join(str(row or "").split()).strip()
            if text:
                out.append(text)
        return out

    @staticmethod
    def _canonical_profile_key(value: Any) -> str:
        token = _profile_key_token(value)
        if not token:
            return ""
        return _PROFILE_ALIAS_TO_CANONICAL.get(token, token)

    @staticmethod
    def _dedupe_strings(items: list[str], *, limit: int = 256) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            cleaned = " ".join(str(item or "").split()).strip()
            if not cleaned:
                continue
            token = cleaned.casefold()
            if token in seen:
                continue
            seen.add(token)
            out.append(cleaned)
            if len(out) >= max(1, int(limit)):
                break
        return out

    def _extract_user_dimensions(self, payload: Mapping[str, Any]) -> dict[str, list[str]]:
        root = self._as_mapping(payload)
        profile = self._as_mapping(root.get("profile"))
        personality = self._as_mapping(root.get("personality"))

        profile_text_parts = [
            str(root.get("profile_text", "") or "").strip(),
            str(root.get("text", "") or "").strip(),
            str(profile.get("profile_text", "") or "").strip(),
            str(profile.get("text", "") or "").strip(),
        ]
        structured = self._parse_structured_profile("\n".join(item for item in profile_text_parts if item))

        out: dict[str, list[str]] = {key: [] for key in _USER_DIMENSION_BINDINGS}
        for container in (root, profile, personality):
            for raw_key, raw_value in container.items():
                canonical = self._canonical_profile_key(raw_key)
                if canonical not in out:
                    continue
                out[canonical].extend(self._to_list_of_strings(raw_value))

        for key in out:
            for row in structured.get(key, []):
                out[key].extend(self._split_list_values(row))
            out[key] = self._dedupe_strings(out[key], limit=256)
        return out

    @staticmethod
    def _extract_json_from_llm_output(raw_output: str) -> dict[str, Any] | list[Any] | None:
        text = str(raw_output or "").strip()
        if not text:
            return None

        candidates: list[str] = []

        # First pass: regex extraction from fenced json blocks.
        for match in _LLM_JSON_FENCE_RE.finditer(text):
            chunk = str(match.group(1) or "").strip()
            if chunk:
                candidates.append(chunk)

        # Second pass: regex extraction for inline object / array blocks.
        for pattern in (_LLM_JSON_OBJECT_RE, _LLM_JSON_ARRAY_RE):
            match = pattern.search(text)
            if match:
                chunk = str(match.group(1) or "").strip()
                if chunk:
                    candidates.append(chunk)

        # Third pass: balanced scan from each regex-located start token.
        for start_match in re.finditer(r"[\{\[]", text):
            start = start_match.start()
            open_char = text[start]
            close_char = "}" if open_char == "{" else "]"
            depth = 0
            in_string = False
            escaped = False
            for idx in range(start, len(text)):
                ch = text[idx]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == "\"":
                        in_string = False
                    continue
                if ch == "\"":
                    in_string = True
                    continue
                if ch == open_char:
                    depth += 1
                    continue
                if ch == close_char:
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : idx + 1].strip())
                        break

        seen: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, (dict, list)):
                return parsed

        return None

    @staticmethod
    def _level_to_weight(level: Any, *, default: float = 0.6) -> float:
        if isinstance(level, (int, float)):
            value = float(level)
            if value > 1.0:
                if value <= 100.0:
                    value = value / 100.0
                elif value <= 10.0:
                    value = value / 10.0
            return max(0.0, min(1.0, value))

        raw = str(level or "").strip().lower()
        if not raw:
            return max(0.0, min(1.0, float(default)))

        for hint, weight in _LEVEL_WEIGHT_HINTS.items():
            if hint in raw:
                return weight

        numeric = GraphWorkspaceService._extract_number(raw)
        if numeric is not None:
            value = float(numeric)
            if "%" in raw and value > 1.0:
                value = value / 100.0
            elif value > 1.0:
                if value <= 10.0:
                    value = value / 10.0
                elif value <= 100.0:
                    value = value / 100.0
            return max(0.0, min(1.0, value))
        return max(0.0, min(1.0, float(default)))

    @staticmethod
    def _normalize_language_code(name: str, code: str) -> str:
        code_clean = re.sub(r"[^a-z0-9_-]+", "", str(code or "").strip().lower())
        if code_clean:
            return code_clean

        token = GraphWorkspaceService._normalize_token(name)
        mapping = {
            "english": "en",
            "английский": "en",
            "անգլերեն": "en",
            "russian": "ru",
            "русский": "ru",
            "русский язык": "ru",
            "рус": "ru",
            "армянский": "hy",
            "armenian": "hy",
            "հայերեն": "hy",
            "arabic": "ar",
            "арабский": "ar",
            "العربية": "ar",
            "chinese": "zh",
            "китайский": "zh",
            "中文": "zh",
            "spanish": "es",
            "испанский": "es",
            "portuguese": "pt",
            "португальский": "pt",
        }
        if token in mapping:
            return mapping[token]

        fallback = re.sub(r"[^a-z0-9]+", "", token)
        if not fallback:
            return "unknown"
        return fallback[:12]

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        if not text:
            return False
        return text not in {"0", "false", "no", "off", "null", "none"}

    @staticmethod
    def _catalog_for_prompt() -> str:
        rows: list[str] = []
        for concept_id in sorted(_GLOBAL_CONCEPT_CATALOG):
            item = _GLOBAL_CONCEPT_CATALOG[concept_id]
            name = str(item.get("name", "") or "").strip() or f"concept_{concept_id}"
            aliases = [str(alias or "").strip() for alias in (item.get("aliases") or []) if str(alias or "").strip()]
            alias_text = ", ".join(aliases[:6])
            if alias_text:
                rows.append(f"- {concept_id}: {name} (aliases: {alias_text})")
            else:
                rows.append(f"- {concept_id}: {name}")
        return "\n".join(rows)

    @staticmethod
    def _stable_temp_concept_id(name: str, *, offset: int = 0) -> int:
        token = _catalog_token(name)
        if not token:
            return 900000000 + int(offset)
        checksum = int(zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF)
        return 900000000 + (checksum % 90000000)

    @staticmethod
    def _normalize_embedding_vector(value: Any) -> list[float]:
        if not isinstance(value, (list, tuple)):
            return [0.0, 0.0, 0.0, 0.0]
        out: list[float] = []
        for item in value:
            try:
                out.append(float(item))
            except Exception:
                continue
        if not out:
            return [0.0, 0.0, 0.0, 0.0]
        return out

    def _resolve_concept_reference(
        self,
        *,
        concept_id: Any,
        concept_name: Any,
        new_concept_hint: bool,
        sequence_idx: int,
    ) -> tuple[int, str, bool]:
        cid = self._to_int(concept_id, default=0)
        raw_name = str(concept_name or "").strip()
        normalized_name = self._normalize_token(raw_name)

        if cid in _GLOBAL_CONCEPT_CATALOG:
            catalog_name = str(_GLOBAL_CONCEPT_CATALOG[cid].get("name", "") or "").strip()
            return int(cid), (raw_name or catalog_name or f"concept_{cid}"), False

        if normalized_name and normalized_name in _CONCEPT_ALIAS_TO_ID:
            mapped_id = int(_CONCEPT_ALIAS_TO_ID[normalized_name])
            mapped_name = str(_GLOBAL_CONCEPT_CATALOG[mapped_id].get("name", "") or "").strip()
            return mapped_id, (raw_name or mapped_name or f"concept_{mapped_id}"), False

        force_new = bool(new_concept_hint)
        if cid > 0 and not force_new:
            return int(cid), (raw_name or f"concept_{cid}"), True

        if raw_name:
            temp_id = self._stable_temp_concept_id(raw_name, offset=sequence_idx)
            return temp_id, raw_name, True
        temp_id = 900000000 + int(sequence_idx)
        return temp_id, f"concept_{temp_id}", True

    @staticmethod
    def _append_unique(rows: list[str], value: str) -> None:
        token = GraphWorkspaceService._normalize_token(value)
        if not token:
            return
        for item in rows:
            if GraphWorkspaceService._normalize_token(item) == token:
                return
        rows.append(value.strip())

    def profile_prompt_template(self, *, text: str, entity_type_hint: str = "human") -> str:
        hint = str(entity_type_hint or "human").strip().lower() or "human"
        source = str(text or "").strip()
        return (
            PROFILE_PROMPT_TEMPLATE.replace("{entity_type_hint}", hint)
            .replace("{text}", source)
            .replace("{relation_types}", ", ".join(_RELATION_TYPE_HINTS))
            .replace("{concept_catalog}", self._catalog_for_prompt())
        )

    @staticmethod
    def _as_mapping(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return []

    @staticmethod
    def _debate_role(role: Any, *, fallback: str) -> str:
        token = re.sub(r"[^a-z0-9_]+", "_", str(role or "").strip().lower())
        if token in _DEBATE_ROLES_ALLOWED:
            return token
        return fallback

    @staticmethod
    def _confidence(value: Any, default: float = 0.55) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return max(0.0, min(1.0, float(default)))

    @staticmethod
    def _to_title(text: str, *, fallback: str, limit: int = 72) -> str:
        source = " ".join(str(text or "").strip().split())
        if not source:
            return fallback
        chunk = source[:limit].strip()
        if len(source) > limit:
            chunk = f"{chunk}..."
        return chunk

    @staticmethod
    def _pick_allowed_token(value: Any, *, allowed: tuple[str, ...], default: str) -> str:
        token = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
        if token in allowed:
            return token
        return str(default or "")

    def _sanitize_personalization(self, payload: Any) -> dict[str, Any]:
        root = self._as_mapping(payload)
        if not root:
            return {}

        llm_roles_raw = self._as_mapping(root.get("llm_roles") or root.get("roles"))
        llm_roles = {
            "proposer": self._debate_role(
                llm_roles_raw.get("proposer"),
                fallback=_DEBATE_DEFAULT_ROLES["proposer"],
            ),
            "critic": self._debate_role(
                llm_roles_raw.get("critic"),
                fallback=_DEBATE_DEFAULT_ROLES["critic"],
            ),
            "judge": self._debate_role(
                llm_roles_raw.get("judge"),
                fallback=_DEBATE_DEFAULT_ROLES["judge"],
            ),
        }

        focus_goals = self._dedupe_strings(
            self._to_list_of_strings(root.get("focus_goals") or root.get("goals")),
            limit=24,
        )
        domain_focus = self._dedupe_strings(
            self._to_list_of_strings(root.get("domain_focus") or root.get("domains")),
            limit=24,
        )
        avoid_topics = self._dedupe_strings(
            self._to_list_of_strings(root.get("avoid_topics") or root.get("avoid")),
            limit=24,
        )
        memory_notes = " ".join(str(root.get("memory_notes", "") or "").split()).strip()
        language_preference = " ".join(str(root.get("language", "") or "").split()).strip()

        out: dict[str, Any] = {
            "response_style": self._pick_allowed_token(
                root.get("response_style") or root.get("style"),
                allowed=_PERSONALIZATION_STYLE_ALLOWED,
                default="adaptive",
            ),
            "reasoning_depth": self._pick_allowed_token(
                root.get("reasoning_depth") or root.get("depth"),
                allowed=_PERSONALIZATION_DEPTH_ALLOWED,
                default="balanced",
            ),
            "risk_tolerance": self._pick_allowed_token(
                root.get("risk_tolerance") or root.get("risk"),
                allowed=_PERSONALIZATION_RISK_ALLOWED,
                default="medium",
            ),
            "tone": self._pick_allowed_token(
                root.get("tone"),
                allowed=_PERSONALIZATION_TONE_ALLOWED,
                default="neutral",
            ),
            "llm_roles": llm_roles,
        }
        if focus_goals:
            out["focus_goals"] = focus_goals
        if domain_focus:
            out["domain_focus"] = domain_focus
        if avoid_topics:
            out["avoid_topics"] = avoid_topics
        if memory_notes:
            out["memory_notes"] = memory_notes[:1200]
        if language_preference:
            out["language"] = language_preference[:24]
        return out

    def _normalize_feedback_items(self, value: Any) -> list[dict[str, Any]]:
        rows = self._as_list(value)
        out: list[dict[str, Any]] = []
        for item in rows[:24]:
            if isinstance(item, str):
                message = " ".join(item.split()).strip()
                if not message:
                    continue
                out.append(
                    {
                        "message": message[:260],
                        "score": 0.0,
                        "decision": "",
                        "target": "",
                    }
                )
                continue

            mapped = self._as_mapping(item)
            if not mapped:
                continue
            message = " ".join(str(mapped.get("message", "") or "").split()).strip()
            target = " ".join(str(mapped.get("target", "") or "").split()).strip()
            decision = re.sub(
                r"[^a-z_]+",
                "_",
                str(mapped.get("decision", "") or "").strip().lower(),
            ).strip("_")
            score = self._confidence(
                mapped.get("score", mapped.get("rating", mapped.get("value", 0.0))),
                0.0,
            )
            if not message and not target and not decision:
                continue
            out.append(
                {
                    "message": message[:260],
                    "score": score,
                    "decision": decision[:32],
                    "target": target[:96],
                }
            )
        return out

    def _personalization_prompt_context(self, personalization: Mapping[str, Any] | None) -> str:
        prefs = self._as_mapping(personalization)
        if not prefs:
            return ""

        parts: list[str] = ["Personalization profile:"]
        parts.append(f"- response_style: {prefs.get('response_style', 'adaptive')}")
        parts.append(f"- reasoning_depth: {prefs.get('reasoning_depth', 'balanced')}")
        parts.append(f"- risk_tolerance: {prefs.get('risk_tolerance', 'medium')}")
        parts.append(f"- tone: {prefs.get('tone', 'neutral')}")

        for key in ("focus_goals", "domain_focus", "avoid_topics"):
            rows = self._to_list_of_strings(prefs.get(key))
            if rows:
                parts.append(f"- {key}: {', '.join(rows[:6])}")

        roles = self._as_mapping(prefs.get("llm_roles"))
        if roles:
            parts.append(
                "- llm_roles: proposer={proposer}, critic={critic}, judge={judge}".format(
                    proposer=str(roles.get("proposer", _DEBATE_DEFAULT_ROLES["proposer"])),
                    critic=str(roles.get("critic", _DEBATE_DEFAULT_ROLES["critic"])),
                    judge=str(roles.get("judge", _DEBATE_DEFAULT_ROLES["judge"])),
                )
            )

        memory_notes = " ".join(str(prefs.get("memory_notes", "") or "").split()).strip()
        if memory_notes:
            parts.append(f"- memory_notes: {memory_notes[:360]}")
        return "\n".join(parts)

    @staticmethod
    def _hallucination_signature(*parts: Any) -> str:
        rows: list[str] = []
        for part in parts:
            token = GraphWorkspaceService._normalize_token(part)
            if token:
                rows.append(token)
        if not rows:
            return ""
        payload = " | ".join(rows)
        checksum = int(zlib.crc32(payload.encode("utf-8")) & 0xFFFFFFFF)
        return f"h{checksum:08x}"

    @staticmethod
    def _token_set(value: Any) -> set[str]:
        source = str(value or "").strip().lower()
        if not source:
            return set()
        cleaned = re.sub(r"[^\w]+", " ", source, flags=re.UNICODE)
        return {item for item in cleaned.split() if len(item) >= 3}

    @staticmethod
    def _jaccard_similarity(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left | right
        if not union:
            return 0.0
        return float(len(left & right)) / float(len(union))

    def _sanitize_hallucination_payload(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        root = self._as_mapping(payload)
        prompt = " ".join(
            str(
                root.get("prompt")
                or root.get("question")
                or root.get("query")
                or root.get("text")
                or ""
            ).split()
        ).strip()
        llm_answer = " ".join(
            str(
                root.get("llm_answer")
                or root.get("answer")
                or root.get("hallucinated_answer")
                or root.get("wrong_answer")
                or ""
            ).split()
        ).strip()
        correct_answer = " ".join(
            str(
                root.get("correct_answer")
                or root.get("reference_answer")
                or root.get("ground_truth")
                or ""
            ).split()
        ).strip()
        source = " ".join(str(root.get("source", "") or "").split()).strip()
        severity = self._pick_allowed_token(
            root.get("severity"),
            allowed=_HALLUCINATION_SEVERITY_ALLOWED,
            default="medium",
        )
        tags = self._dedupe_strings(self._to_list_of_strings(root.get("tags")), limit=16)
        confidence = self._confidence(root.get("confidence", 0.8), 0.8)
        metadata = self._as_mapping(root.get("metadata"))
        return {
            "prompt": prompt[:1800],
            "llm_answer": llm_answer[:3200],
            "correct_answer": correct_answer[:3200],
            "source": source[:600],
            "severity": severity,
            "tags": tags,
            "confidence": confidence,
            "metadata": metadata,
        }

    def _ensure_hallucination_branch_node(self, *, user_id: str):
        branch_key = f"{user_id}:{_HALLUCINATION_BRANCH_NAME}"
        branch_node, _ = self._ensure_shared_node(
            node_type="llm_hallucination_branch",
            identity_key="branch_key",
            identity_value=branch_key,
            attributes={
                "branch_key": branch_key,
                "branch_name": _HALLUCINATION_BRANCH_NAME,
                "user_id": user_id,
                "name": f"Hallucination Hunter Branch ({user_id})",
                "description": "Dedicated branch to track hallucinations and verified corrections.",
            },
        )
        branch_node.attributes["last_seen_at"] = float(time.time())
        return branch_node

    def _hallucination_case_nodes(self, *, user_id: str) -> list[Any]:
        out: list[Any] = []
        user_token = " ".join(str(user_id or "").split()).strip()
        for node in self.api.engine.store.nodes.values():
            if str(node.type) != "llm_hallucination_case":
                continue
            owner = " ".join(str(node.attributes.get("user_id", "") or "").split()).strip()
            if user_token and owner and owner != user_token:
                continue
            out.append(node)
        return out

    def _match_hallucination_cases(
        self,
        *,
        user_id: str,
        prompt: str,
        llm_answer: str = "",
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        prompt_sig = self._hallucination_signature(user_id, prompt)
        prompt_tokens = self._token_set(prompt)
        answer_tokens = self._token_set(llm_answer)

        scored: list[dict[str, Any]] = []
        for node in self._hallucination_case_nodes(user_id=user_id):
            attrs = self._as_mapping(node.attributes)
            score = 0.0
            reasons: list[str] = []

            case_prompt_sig = str(attrs.get("prompt_signature", "") or "").strip()
            if prompt_sig and case_prompt_sig and prompt_sig == case_prompt_sig:
                score += 1.0
                reasons.append("prompt_signature_match")

            case_prompt_tokens = self._token_set(attrs.get("prompt", ""))
            prompt_overlap = self._jaccard_similarity(prompt_tokens, case_prompt_tokens)
            if prompt_overlap > 0.0:
                score += 0.82 * prompt_overlap
                if prompt_overlap >= 0.45:
                    reasons.append("prompt_overlap")

            case_wrong_tokens = self._token_set(attrs.get("hallucinated_answer", ""))
            answer_overlap = self._jaccard_similarity(answer_tokens, case_wrong_tokens)
            if answer_overlap > 0.0:
                score += 0.58 * answer_overlap
                if answer_overlap >= 0.40:
                    reasons.append("answer_overlap")

            if score < 0.18:
                continue
            scored.append(
                {
                    "case_node_id": int(node.id),
                    "score": round(max(0.0, min(1.0, score)), 4),
                    "prompt": str(attrs.get("prompt", "") or ""),
                    "hallucinated_answer": str(attrs.get("hallucinated_answer", "") or ""),
                    "correct_answer": str(attrs.get("correct_answer", "") or ""),
                    "source": str(attrs.get("source", "") or ""),
                    "severity": str(attrs.get("severity", "medium") or "medium"),
                    "tags": self._to_list_of_strings(attrs.get("tags")),
                    "occurrence_count": self._to_int(attrs.get("occurrence_count", 1), 1),
                    "reasons": reasons,
                }
            )

        scored.sort(
            key=lambda row: (
                float(row.get("score", 0.0)),
                int(row.get("occurrence_count", 0)),
            ),
            reverse=True,
        )
        return scored[: max(1, min(10, int(top_k)))]

    def _hallucination_guard_context(self, matches: list[Mapping[str, Any]]) -> str:
        if not matches:
            return ""
        rows: list[str] = [
            "Hallucination guard memory:",
            "Avoid repeating known wrong claims and prioritize corrected facts below.",
        ]
        for idx, item in enumerate(matches[:3]):
            wrong = " ".join(str(item.get("hallucinated_answer", "") or "").split()).strip()
            correct = " ".join(str(item.get("correct_answer", "") or "").split()).strip()
            source = " ".join(str(item.get("source", "") or "").split()).strip()
            score = float(item.get("score", 0.0) or 0.0)
            rows.append(f"- Case {idx + 1} (score={score:.2f})")
            if wrong:
                rows.append(f"  wrong: {wrong[:220]}")
            if correct:
                rows.append(f"  correct: {correct[:220]}")
            if source:
                rows.append(f"  source: {source[:120]}")
        return "\n".join(rows)

    @staticmethod
    def _stable_json(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(value)

    def _sanitize_archive_chat_payload(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        root = self._as_mapping(payload)
        message = " ".join(
            str(root.get("message", "") or root.get("prompt", "") or root.get("text", "") or "").split()
        ).strip()
        context = " ".join(str(root.get("context", "") or root.get("notes", "") or "").split()).strip()
        model_path = str(root.get("model_path", "") or "").strip()
        model_role = self._debate_role(root.get("model_role"), fallback="general")
        verification_mode = self._pick_allowed_token(
            root.get("verification_mode"),
            allowed=_ARCHIVE_VERIFICATION_MODES_ALLOWED,
            default="strict",
        )
        top_k = max(1, min(8, self._to_int(root.get("top_k", 3), 3)))
        apply_to_graph = self._to_bool(root.get("apply_to_graph", True))
        return {
            "message": message[:2400],
            "context": context[:2400],
            "model_path": model_path,
            "model_role": model_role,
            "verification_mode": verification_mode,
            "top_k": top_k,
            "apply_to_graph": apply_to_graph,
        }

    def _normalize_archive_updates(self, value: Any) -> list[dict[str, Any]]:
        rows = self._as_list(value)
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in rows:
            row = self._as_mapping(item)
            if not row:
                continue
            entity = " ".join(str(row.get("entity", "") or row.get("target", "") or "").split()).strip()
            field = " ".join(str(row.get("field", "") or row.get("path", "") or row.get("key", "") or "").split()).strip()
            if not entity or not field:
                continue
            operation = self._pick_allowed_token(
                row.get("operation"),
                allowed=_ARCHIVE_UPDATE_OPERATIONS_ALLOWED,
                default="upsert",
            )
            update_key = self._hallucination_signature(entity, field, operation)
            if not update_key or update_key in seen:
                continue
            seen.add(update_key)
            out.append(
                {
                    "entity": entity[:160],
                    "field": field[:160],
                    "operation": operation,
                    "value": row.get("value"),
                    "reason": " ".join(str(row.get("reason", "") or row.get("rationale", "") or "").split()).strip()[:800],
                    "source": " ".join(str(row.get("source", "") or row.get("evidence", "") or "").split()).strip()[:800],
                    "confidence": self._confidence(row.get("confidence", 0.65), 0.65),
                    "tags": self._dedupe_strings(self._to_list_of_strings(row.get("tags")), limit=16),
                }
            )
        return out

    def _ensure_archive_branch_node(self, *, user_id: str):
        branch_key = f"{user_id}:{_ARCHIVE_UPDATE_BRANCH_NAME}"
        branch_node, _ = self._ensure_shared_node(
            node_type="llm_archive_update_branch",
            identity_key="branch_key",
            identity_value=branch_key,
            attributes={
                "branch_key": branch_key,
                "branch_name": _ARCHIVE_UPDATE_BRANCH_NAME,
                "user_id": user_id,
                "name": f"Archive Update Verification Branch ({user_id})",
                "description": "Stores verified JSON archive updates generated by selected local LLM models.",
            },
        )
        branch_node.attributes["last_seen_at"] = float(time.time())
        return branch_node

    def _archive_record_nodes(self, *, user_id: str) -> list[Any]:
        out: list[Any] = []
        user_token = " ".join(str(user_id or "").split()).strip()
        for node in self.api.engine.store.nodes.values():
            if str(node.type) != "llm_archive_update_record":
                continue
            owner = " ".join(str(node.attributes.get("user_id", "") or "").split()).strip()
            if user_token and owner and owner != user_token:
                continue
            out.append(node)
        return out

    def _resolve_archive_chat_llm(
        self,
        *,
        model_path: str,
        model_role: str,
    ) -> tuple[Callable[[str], str] | None, str, str]:
        path_token = str(model_path or "").strip()
        role_token = self._debate_role(model_role, fallback="general")
        if path_token and self.model_llm_resolver is not None:
            try:
                fn = self.model_llm_resolver(path_token)
                if fn is not None:
                    return fn, path_token, "explicit_model_path"
            except Exception:
                pass
        role_fn = self._resolve_debate_llm(role_token)
        if role_fn is not None:
            return role_fn, "", f"role:{role_token}"
        return None, "", "unavailable"

    def _verify_archive_updates(
        self,
        *,
        user_id: str,
        message: str,
        updates: list[dict[str, Any]],
        verification_mode: str,
        top_k: int,
    ) -> dict[str, Any]:
        strict_mode = verification_mode == "strict"
        issues: list[str] = []
        warnings: list[str] = []
        checked = len(updates)
        min_confidence = 0.62 if strict_mode else 0.5

        if checked <= 0:
            issues.append("no_updates")

        for idx, row in enumerate(updates):
            value = row.get("value")
            if value in ("", None, [], {}):
                issues.append(f"update_{idx + 1}_empty_value")
            if not str(row.get("reason", "") or "").strip():
                issues.append(f"update_{idx + 1}_missing_reason")
            if strict_mode and not str(row.get("source", "") or "").strip():
                issues.append(f"update_{idx + 1}_missing_source")
            if self._confidence(row.get("confidence", 0.0), 0.0) < min_confidence:
                issues.append(f"update_{idx + 1}_low_confidence")

        llm_answer_text = self._stable_json({"archive_updates": updates})
        hallucination_matches = self._match_hallucination_cases(
            user_id=user_id,
            prompt=message,
            llm_answer=llm_answer_text,
            top_k=top_k,
        )
        if hallucination_matches:
            issues.append("known_hallucination_overlap")

        known_records = self._archive_record_nodes(user_id=user_id)
        value_conflicts: list[dict[str, Any]] = []
        for row in updates:
            entity = self._normalize_token(row.get("entity"))
            field = self._normalize_token(row.get("field"))
            if not entity or not field:
                continue
            new_value_key = self._stable_json(row.get("value"))
            for node in known_records:
                attrs = self._as_mapping(node.attributes)
                if self._normalize_token(attrs.get("entity")) != entity:
                    continue
                if self._normalize_token(attrs.get("field")) != field:
                    continue
                existing_value = attrs.get("value")
                existing_value_key = self._stable_json(existing_value)
                if existing_value_key == new_value_key:
                    continue
                value_conflicts.append(
                    {
                        "record_node_id": int(node.id),
                        "entity": str(attrs.get("entity", "") or ""),
                        "field": str(attrs.get("field", "") or ""),
                        "existing_value": existing_value,
                        "candidate_value": row.get("value"),
                    }
                )
        if value_conflicts:
            if strict_mode:
                issues.append("conflicts_with_verified_archive")
            else:
                warnings.append("conflicts_with_verified_archive")

        score = 1.0 - (0.2 * float(len(issues))) - (0.08 * float(len(warnings)))
        score = max(0.0, min(1.0, score))
        return {
            "verified": len(issues) == 0,
            "verification_mode": verification_mode,
            "schema_valid": checked > 0,
            "issue_count": len(issues),
            "warning_count": len(warnings),
            "issues": issues,
            "warnings": warnings,
            "checked_updates": checked,
            "hallucination_guard_hits": len(hallucination_matches),
            "hallucination_matches": hallucination_matches,
            "value_conflicts": value_conflicts[:8],
            "score": round(score, 4),
        }

    def _coerce_archive_updates_input(self, value: Any) -> list[dict[str, Any]]:
        source = value
        if isinstance(source, str):
            parsed = self._extract_json_from_llm_output(source)
            if isinstance(parsed, Mapping):
                source = parsed.get("archive_updates", parsed.get("updates", []))
            elif isinstance(parsed, list):
                source = parsed
            else:
                source = []
        elif isinstance(source, Mapping):
            source = source.get("archive_updates", source.get("updates", []))
        return self._normalize_archive_updates(source)

    def _attach_archive_updates_to_graph(
        self,
        *,
        user_id: str,
        session_id: str,
        message: str,
        context: str,
        summary: str,
        updates: list[dict[str, Any]],
        verification: Mapping[str, Any],
        model_role: str,
        model_path: str,
        resolution_mode: str,
        raw_output: str,
        verification_mode: str,
        apply_to_graph: bool,
    ) -> dict[str, Any]:
        graph_binding: dict[str, Any] = {
            "attached": False,
            "branch_node_id": 0,
            "session_node_id": 0,
            "update_node_ids": [],
        }
        if not apply_to_graph:
            return graph_binding

        branch_node = self._ensure_archive_branch_node(user_id=user_id)
        graph_binding["attached"] = True
        graph_binding["branch_node_id"] = int(branch_node.id)

        session_node = self.api.engine.create_node(
            "llm_archive_update_session",
            attributes={
                "name": self._to_title(message, fallback="Archive Chat Session", limit=96),
                "user_id": user_id,
                "session_id": session_id,
                "message": message,
                "context": context,
                "summary": summary,
                "model_role": model_role,
                "model_path": model_path,
                "resolution_mode": resolution_mode,
                "verification": dict(verification),
                "raw_output": raw_output[:6000],
            },
            state={"confidence": self._confidence(verification.get("score", 0.0), 0.0)},
        )
        graph_binding["session_node_id"] = int(session_node.id)
        self._connect_nodes(
            from_node=branch_node.id,
            to_node=session_node.id,
            relation_type="tracks_archive_session",
            weight=self._confidence(verification.get("score", 0.0), 0.0),
            logic_rule="archive_update_chat",
            metadata={
                "verified": bool(verification.get("verified", False)),
                "verification_mode": verification_mode,
            },
        )

        update_node_ids: list[int] = []
        for idx, row in enumerate(updates):
            entity = str(row.get("entity", "") or "").strip()
            field = str(row.get("field", "") or "").strip()
            operation = str(row.get("operation", "upsert") or "upsert").strip() or "upsert"
            node_type = (
                "llm_archive_update_record"
                if bool(verification.get("verified", False))
                else "llm_archive_update_candidate"
            )
            update_key = self._hallucination_signature(user_id, entity, field, operation)
            update_node, created = self._ensure_shared_node(
                node_type=node_type,
                identity_key="update_key",
                identity_value=update_key,
                attributes={
                    "update_key": update_key,
                    "user_id": user_id,
                    "entity": entity,
                    "field": field,
                    "operation": operation,
                    "value": row.get("value"),
                    "reason": str(row.get("reason", "") or ""),
                    "source": str(row.get("source", "") or ""),
                    "tags": self._to_list_of_strings(row.get("tags")),
                    "verification_mode": verification_mode,
                    "verified": bool(verification.get("verified", False)),
                    "first_seen_at": float(time.time()),
                },
            )
            update_node.attributes["entity"] = entity
            update_node.attributes["field"] = field
            update_node.attributes["operation"] = operation
            update_node.attributes["value"] = row.get("value")
            update_node.attributes["reason"] = str(row.get("reason", "") or "")
            update_node.attributes["source"] = str(row.get("source", "") or "")
            update_node.attributes["tags"] = self._to_list_of_strings(row.get("tags"))
            update_node.attributes["verified"] = bool(verification.get("verified", False))
            update_node.attributes["last_seen_at"] = float(time.time())
            prev_count = self._to_int(update_node.attributes.get("occurrence_count", 0), 0)
            update_node.attributes["occurrence_count"] = max(1, prev_count + 1)
            if created:
                update_node.attributes["created_at"] = float(time.time())

            self._connect_nodes(
                from_node=session_node.id,
                to_node=update_node.id,
                relation_type="proposes_archive_update",
                weight=self._confidence(row.get("confidence", 0.65), 0.65),
                logic_rule=(
                    "archive_update_verified"
                    if bool(verification.get("verified", False))
                    else "archive_update_candidate"
                ),
                metadata={
                    "index": idx + 1,
                    "operation": operation,
                    "verified": bool(verification.get("verified", False)),
                },
            )
            update_node_ids.append(int(update_node.id))
        graph_binding["update_node_ids"] = update_node_ids
        return graph_binding

    def _build_archive_chat_reply(
        self,
        *,
        message: str,
        summary: str,
        updates: list[Mapping[str, Any]],
        verification: Mapping[str, Any],
    ) -> str:
        is_russian = bool(re.search(r"[А-Яа-яЁё]", str(message or "")))
        verified = bool(verification.get("verified", False))
        issue_count = int(verification.get("issue_count", 0) or 0)
        update_count = len(updates)

        preview_tokens: list[str] = []
        for row in updates[:3]:
            entity = str(row.get("entity", "") or "").strip()
            field = str(row.get("field", "") or "").strip()
            if entity and field:
                preview_tokens.append(f"{entity}.{field}")
            elif entity:
                preview_tokens.append(entity)
        preview = ", ".join(preview_tokens)

        if is_russian:
            parts: list[str] = []
            if summary:
                parts.append(summary)
            if verified:
                parts.append(f"Подготовил {update_count} проверенных обновлений архива.")
            else:
                parts.append(
                    f"Сформировал {update_count} обновлений, но проверка нашла проблемы ({issue_count})."
                )
            if preview:
                parts.append(f"Основные изменения: {preview}.")
            issues = self._to_list_of_strings(verification.get("issues"))[:3]
            if issues:
                parts.append(f"Нужно поправить: {', '.join(issues)}.")
            parts.append("Проверь и отредактируй выводы в блоке review справа.")
            return " ".join(item for item in parts if item).strip()

        parts = []
        if summary:
            parts.append(summary)
        if verified:
            parts.append(f"I prepared {update_count} verified archive updates.")
        else:
            parts.append(f"I drafted {update_count} updates, but verification found issues ({issue_count}).")
        if preview:
            parts.append(f"Main changes: {preview}.")
        issues = self._to_list_of_strings(verification.get("issues"))[:3]
        if issues:
            parts.append(f"Please resolve: {', '.join(issues)}.")
        parts.append("Review and edit the conclusions in the review panel.")
        return " ".join(item for item in parts if item).strip()

    @staticmethod
    def _split_sentences(text: str, *, limit: int = 6) -> list[str]:
        parts = re.split(r"[.!?\n]+", str(text or ""))
        out: list[str] = []
        for item in parts:
            token = " ".join(item.split()).strip()
            if not token:
                continue
            out.append(token)
            if len(out) >= limit:
                break
        return out

    def _resolve_debate_llm(self, role: str) -> Callable[[str], str] | None:
        if self.role_llm_resolver is not None:
            try:
                fn = self.role_llm_resolver(role)
                if fn is not None:
                    return fn
            except Exception:
                pass
        if role == "general":
            return self.profile_llm_fn
        return self.profile_llm_fn

    def _fallback_hypotheses(self, *, topic: str, count: int) -> list[dict[str, Any]]:
        seeds = self._split_sentences(topic, limit=max(3, count * 2))
        if not seeds:
            seeds = [str(topic or "Define a practical plan").strip() or "Define a practical plan"]

        out: list[dict[str, Any]] = []
        for index in range(max(1, count)):
            base = seeds[index % len(seeds)]
            claim = (
                base
                if len(base) > 12
                else f"{base}. Convert it into measurable actions and graph checks."
            )
            out.append(
                {
                    "index": index + 1,
                    "title": self._to_title(claim, fallback=f"Hypothesis {index + 1}"),
                    "claim": claim,
                    "rationale": "Deterministic fallback hypothesis (role model unavailable).",
                    "confidence": self._confidence(0.52 + (0.05 * (index % 3)), 0.55),
                }
            )
        return out[: max(1, count)]

    def _fallback_critique(self, *, hypothesis: Mapping[str, Any]) -> dict[str, Any]:
        claim = str(hypothesis.get("claim", "") or "")
        issues: list[str] = []
        if len(claim) < 40:
            issues.append("Hypothesis is underspecified and needs measurable constraints.")
        if "always" in claim.lower() or "never" in claim.lower():
            issues.append("Contains absolute language; likely brittle across contexts.")
        if not issues:
            issues.append("No direct contradiction detected; validate with empirical checks.")
        risk_score = 0.42 if len(issues) == 1 else 0.58
        return {
            "issues": issues,
            "contradictions": [item for item in issues if "contradiction" in item.lower()],
            "risk_score": self._confidence(risk_score, 0.5),
            "confidence": self._confidence(0.6, 0.6),
            "recommendation": "revise" if risk_score >= 0.55 else "accept_with_checks",
        }

    def _fallback_verdict(
        self,
        *,
        hypotheses: list[dict[str, Any]],
        critiques: list[dict[str, Any]],
    ) -> dict[str, Any]:
        best_index = 0
        best_score = -1.0
        ranking: list[dict[str, Any]] = []
        for idx, hypothesis in enumerate(hypotheses):
            confidence = self._confidence(hypothesis.get("confidence", 0.5), 0.5)
            risk = self._confidence((critiques[idx] if idx < len(critiques) else {}).get("risk_score", 0.5), 0.5)
            score = max(0.0, min(1.0, (0.72 * confidence) + (0.28 * (1.0 - risk))))
            ranking.append(
                {
                    "index": idx + 1,
                    "score": round(score, 4),
                }
            )
            if score > best_score:
                best_index = idx
                best_score = score
        ranking.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        chosen = hypotheses[best_index] if hypotheses else {"title": "", "claim": ""}
        return {
            "selected_index": best_index + 1,
            "decision": str(chosen.get("claim", "") or ""),
            "consensus": (
                "Fallback judge selected the highest confidence/lowest risk hypothesis. "
                "Review manually before operational changes."
            ),
            "confidence": self._confidence(best_score, 0.55),
            "ranking": ranking,
        }

    def _normalize_person_concept_payload(
        self,
        root: Mapping[str, Any],
        *,
        source_text: str,
        entity_type_hint: str,
    ) -> dict[str, Any]:
        person = self._as_mapping(root.get("person"))
        first_name = str(person.get("first_name", "") or "").strip()
        last_name = str(person.get("last_name", "") or "").strip()
        if not first_name:
            inferred_first, inferred_last = self._infer_name_from_text(source_text)
            first_name = inferred_first
            last_name = last_name or inferred_last

        birth_date = str(person.get("birth_date", "") or "").strip()
        bio = str(person.get("bio", "") or source_text).strip()
        person_id = self._to_int(person.get("id"), default=0)

        preferences: list[str] = []
        values: list[str] = []
        fears: list[str] = []
        goals: list[str] = []
        traits: list[str] = []

        links: list[dict[str, Any]] = []
        concept_relations: list[dict[str, Any]] = []
        raw_relations = self._as_list(root.get("concept_relations"))
        for idx, row in enumerate(raw_relations):
            item = self._as_mapping(row)
            details = self._as_mapping(item.get("details"))
            relation_type = str(item.get("relation_type", "") or "").strip().lower() or "associated_with"
            confidence = max(
                0.0,
                min(
                    1.0,
                    self._level_to_weight(item.get("confidence", 0.6), default=0.6),
                ),
            )
            concept_name_candidate = (
                item.get("concept_name")
                or details.get("concept_name")
                or details.get("name")
                or ""
            )
            concept_id, concept_name, is_new_concept = self._resolve_concept_reference(
                concept_id=item.get("concept_id"),
                concept_name=concept_name_candidate,
                new_concept_hint=self._to_bool(
                    item.get("new_concept", details.get("new_concept", False))
                ),
                sequence_idx=idx + 1,
            )
            embedding_vector = self._normalize_embedding_vector(item.get("embedding_vector"))
            additional_info = str(details.get("additional_info", "") or "").strip()

            relation_payload = {
                "concept_id": int(concept_id),
                "concept_name": concept_name,
                "new_concept": bool(is_new_concept),
                "relation_type": relation_type,
                "embedding_vector": embedding_vector,
                "confidence": confidence,
                "details": {
                    "additional_info": additional_info,
                    **{
                        str(k): v
                        for k, v in details.items()
                        if str(k) not in {"additional_info", "concept_name", "name", "new_concept"}
                    },
                },
            }
            concept_relations.append(relation_payload)

            if relation_type in {"likes", "prefers", "interested_in", "positive_association"}:
                self._append_unique(preferences, concept_name)
            if relation_type in {"motivated_by", "respects", "trusts"}:
                self._append_unique(values, concept_name)
            if relation_type in {"fears", "avoids", "stressed_by", "negative_association"}:
                self._append_unique(fears, concept_name)
            if relation_type in {"goal", "goal_related", "aspires_to"}:
                self._append_unique(goals, concept_name)
            if relation_type in {"associated_with", "influenced_by"}:
                self._append_unique(traits, concept_name)

            links.append(
                {
                    "relation_type": relation_type,
                    "target_type": "concept",
                    "target_name": concept_name,
                    "target_identifier": f"concept:{concept_id}",
                    "weight": confidence,
                    "description": additional_info,
                    "concept_id": int(concept_id),
                    "new_concept": bool(is_new_concept),
                    "embedding_vector": embedding_vector,
                    "confidence": confidence,
                    "details": relation_payload["details"],
                }
            )

        display_name = " ".join(part for part in (first_name, last_name) if part).strip()
        return {
            "entity": {
                "type": "human",
                "name": display_name,
                "first_name": first_name,
                "last_name": last_name,
                "industry": "",
                "description": bio,
            },
            "person": {
                "id": person_id,
                "first_name": first_name,
                "last_name": last_name,
                "bio": bio,
                "birth_date": birth_date,
            },
            "personality": {
                "traits": traits,
                "values": values,
                "fears": fears,
                "goals": goals,
                "preferences": preferences,
                "desires": self._to_list_of_strings(root.get("desires")),
                "principles": self._to_list_of_strings(root.get("principles")),
                "opportunities": self._to_list_of_strings(root.get("opportunities")),
                "abilities": self._to_list_of_strings(root.get("abilities")),
                "access": self._to_list_of_strings(root.get("access")),
                "knowledge": self._to_list_of_strings(root.get("knowledge")),
                "assets": self._to_list_of_strings(root.get("assets")),
            },
            "employment": [],
            "skills": [],
            "languages": [],
            "capabilities": [],
            "concept_relations": concept_relations,
            "links": links,
        }

    def _normalize_profile_payload(
        self,
        payload: Mapping[str, Any] | list[Any],
        *,
        source_text: str,
        entity_type_hint: str,
    ) -> dict[str, Any]:
        if isinstance(payload, list):
            payload = {"skills": payload}
        root = self._as_mapping(payload)
        if "person" in root or "concept_relations" in root:
            return self._normalize_person_concept_payload(
                root,
                source_text=source_text,
                entity_type_hint=entity_type_hint,
            )

        entity = self._as_mapping(root.get("entity"))
        personality = self._as_mapping(root.get("personality"))

        entity_type = str(
            entity.get("type")
            or root.get("entity_type")
            or root.get("type")
            or entity_type_hint
            or "human"
        ).strip().lower()
        if entity_type not in {"human", "company", "technology", "generic"}:
            entity_type = "generic"

        entity_name = str(
            entity.get("name") or root.get("name") or ""
        ).strip()
        first_name = str(entity.get("first_name") or root.get("first_name") or "").strip()
        last_name = str(entity.get("last_name") or root.get("last_name") or "").strip()
        if not first_name and entity_type == "human":
            inferred_first, inferred_last = self._infer_name_from_text(source_text)
            first_name = inferred_first
            last_name = last_name or inferred_last
        if not first_name and entity_type == "human" and entity_name:
            name_parts = [item for item in entity_name.split() if item]
            if name_parts:
                first_name = name_parts[0]
                if len(name_parts) > 1 and not last_name:
                    last_name = " ".join(name_parts[1:])

        traits = self._to_list_of_strings(personality.get("traits") or root.get("traits"))
        values = self._to_list_of_strings(personality.get("values") or root.get("values"))
        fears = self._to_list_of_strings(personality.get("fears") or root.get("fears"))
        goals = self._to_list_of_strings(personality.get("goals") or root.get("goals"))
        preferences = self._to_list_of_strings(
            personality.get("preferences") or root.get("preferences")
        )
        desires = self._to_list_of_strings(personality.get("desires") or root.get("desires"))
        principles = self._to_list_of_strings(personality.get("principles") or root.get("principles"))
        opportunities = self._to_list_of_strings(personality.get("opportunities") or root.get("opportunities"))
        abilities = self._to_list_of_strings(personality.get("abilities") or root.get("abilities"))
        access = self._to_list_of_strings(personality.get("access") or root.get("access"))
        knowledge = self._to_list_of_strings(personality.get("knowledge") or root.get("knowledge"))
        assets = self._to_list_of_strings(personality.get("assets") or root.get("assets"))

        employment: list[dict[str, Any]] = []
        for row in self._as_list(root.get("employment")):
            item = self._as_mapping(row)
            status = str(item.get("status", "") or "").strip()
            company_name = self._sanitize_company_name(item.get("company_name", ""))
            if not status and not company_name:
                continue
            employment.append(
                {
                    "status": status or "employee",
                    "company_name": company_name,
                    "importance_score": self._level_to_weight(
                        item.get("importance_score", 0.75),
                        default=0.75,
                    ),
                    "company_attributes": self._as_mapping(item.get("company_attributes")),
                }
            )

        skills: list[dict[str, Any]] = []
        for row in self._as_list(root.get("skills")):
            if isinstance(row, str):
                name = row.strip()
                if not name:
                    continue
                skills.append(
                    {
                        "name": name,
                        "category": "general",
                        "level": "",
                        "weight": 0.6,
                        "description": "",
                        "evidence": "",
                    }
                )
                continue
            item = self._as_mapping(row)
            name = str(item.get("name", "") or "").strip()
            if not name:
                continue
            level = str(item.get("level", "") or "").strip()
            skills.append(
                {
                    "name": name,
                    "category": str(item.get("category", "") or "").strip() or "general",
                    "level": level,
                    "weight": self._level_to_weight(level or item.get("score"), default=0.6),
                    "description": str(item.get("description", "") or "").strip(),
                    "evidence": str(item.get("evidence", "") or "").strip(),
                }
            )

        languages: list[dict[str, Any]] = []
        for row in self._as_list(root.get("languages")):
            if isinstance(row, str):
                name = row.strip()
                if not name:
                    continue
                code = self._normalize_language_code(name, "")
                languages.append(
                    {
                        "name": name,
                        "code": code,
                        "proficiency": "",
                        "weight": 0.6,
                        "family": "",
                        "origin": "",
                        "script": "",
                        "description": "",
                    }
                )
                continue
            item = self._as_mapping(row)
            name = str(item.get("name", "") or "").strip()
            code = self._normalize_language_code(name, str(item.get("code", "") or ""))
            if not name and code:
                name = code.upper()
            if not name:
                continue
            proficiency = str(item.get("proficiency", "") or "").strip()
            languages.append(
                {
                    "name": name,
                    "code": code,
                    "proficiency": proficiency,
                    "weight": self._level_to_weight(proficiency or item.get("level"), default=0.6),
                    "family": str(item.get("family", "") or "").strip(),
                    "origin": str(item.get("origin", "") or "").strip(),
                    "script": str(item.get("script", "") or "").strip(),
                    "description": str(item.get("description", "") or "").strip(),
                }
            )

        capabilities: list[dict[str, Any]] = []
        for row in self._as_list(root.get("capabilities")):
            if isinstance(row, str):
                name = row.strip()
                if not name:
                    continue
                capabilities.append(
                    {
                        "name": name,
                        "category": "general",
                        "level": "",
                        "weight": 0.6,
                        "description": "",
                    }
                )
                continue
            item = self._as_mapping(row)
            name = str(item.get("name", "") or "").strip()
            if not name:
                continue
            level = str(item.get("level", "") or "").strip()
            capabilities.append(
                {
                    "name": name,
                    "category": str(item.get("category", "") or "").strip() or "general",
                    "level": level,
                    "weight": self._level_to_weight(level or item.get("score"), default=0.6),
                    "description": str(item.get("description", "") or "").strip(),
                }
            )

        links: list[dict[str, Any]] = []
        for row in self._as_list(root.get("links")):
            item = self._as_mapping(row)
            target_name = str(item.get("target_name", "") or "").strip()
            target_identifier = str(item.get("target_identifier", "") or "").strip()
            if not target_name and not target_identifier:
                continue
            links.append(
                {
                    "relation_type": str(item.get("relation_type", "") or "").strip() or "related_to",
                    "target_type": str(item.get("target_type", "") or "").strip().lower() or "generic",
                    "target_name": target_name,
                    "target_identifier": target_identifier,
                    "weight": self._level_to_weight(item.get("weight", 0.6), default=0.6),
                    "description": str(item.get("description", "") or "").strip(),
                }
            )

        summary = str(
            entity.get("description") or root.get("summary") or source_text
        ).strip()

        return {
            "entity": {
                "type": entity_type,
                "name": entity_name,
                "first_name": first_name,
                "last_name": last_name,
                "industry": str(entity.get("industry", "") or root.get("industry", "") or "").strip(),
                "description": summary,
            },
            "personality": {
                "traits": traits,
                "values": values,
                "fears": fears,
                "goals": goals,
                "preferences": preferences,
                "desires": desires,
                "principles": principles,
                "opportunities": opportunities,
                "abilities": abilities,
                "access": access,
                "knowledge": knowledge,
                "assets": assets,
            },
            "employment": employment,
            "skills": skills,
            "languages": languages,
            "capabilities": capabilities,
            "links": links,
        }

    def _find_node_by_identity(self, *, node_type: str, key: str, value: str):
        needle = self._normalize_token(value)
        if not needle:
            return None
        for node in self.api.engine.store.nodes.values():
            if node.type != node_type:
                continue
            candidate = node.attributes.get(key, "")
            if self._normalize_token(candidate) == needle:
                return node
        return None

    def _ensure_shared_node(
        self,
        *,
        node_type: str,
        identity_key: str,
        identity_value: str,
        attributes: Mapping[str, Any] | None = None,
    ):
        existing = self._find_node_by_identity(
            node_type=node_type,
            key=identity_key,
            value=identity_value,
        )
        attrs = dict(attributes or {})
        attrs.setdefault(identity_key, identity_value)
        if existing is not None:
            for key, value in attrs.items():
                if key not in existing.attributes and value not in ("", None, [], {}):
                    existing.attributes[key] = value
            return existing, False
        created = self.api.engine.create_node(
            node_type,
            attributes=attrs,
            state={},
        )
        return created, True

    def _connect_nodes(
        self,
        *,
        from_node: int,
        to_node: int,
        relation_type: str,
        weight: float,
        logic_rule: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.api.connect(
            int(from_node),
            int(to_node),
            relation_type=str(relation_type or "related_to"),
            weight=max(0.0, min(1.0, float(weight))),
            direction="directed",
            logic_rule=str(logic_rule or "profile_import"),
            metadata=dict(metadata or {}),
        )

    def _ensure_user_profile_node(self, *, user_id: str, display_name: str):
        node, _ = self._ensure_shared_node(
            node_type="person_profile",
            identity_key="user_id",
            identity_value=user_id,
            attributes={
                "user_id": user_id,
                "name": display_name or user_id,
                "description": "User profile root for semantic personal dimensions.",
            },
        )
        if display_name:
            node.attributes["name"] = display_name
        return node

    def _bind_user_dimensions(
        self,
        *,
        root_node_id: int,
        dimensions: Mapping[str, Any],
        logic_rule: str,
    ) -> dict[str, Any]:
        created_nodes = 0
        created_edges = 0
        node_ids_by_dimension: dict[str, list[int]] = {}

        for key, config in _USER_DIMENSION_BINDINGS.items():
            node_ids_by_dimension[key] = []
            values = self._to_list_of_strings(dimensions.get(key))
            for value in self._dedupe_strings(values, limit=256):
                identifier = re.sub(r"[^a-z0-9]+", "_", self._normalize_token(value)).strip("_") or value
                scoped_identifier = f"{key}:{identifier}"
                target_node, created = self._ensure_shared_node(
                    node_type=str(config.get("node_type", "concept") or "concept"),
                    identity_key="identifier",
                    identity_value=scoped_identifier,
                    attributes={
                        "identifier": scoped_identifier,
                        "name": value,
                        "dimension": key,
                        "description": f"{config.get('display', key)} for user profile",
                    },
                )
                if created:
                    created_nodes += 1
                node_ids_by_dimension[key].append(int(target_node.id))
                self._connect_nodes(
                    from_node=int(root_node_id),
                    to_node=int(target_node.id),
                    relation_type=str(config.get("relation_type", "related_to") or "related_to"),
                    weight=0.78,
                    logic_rule=logic_rule,
                )
                created_edges += 1

        return {
            "created_nodes_estimate": created_nodes,
            "created_edges_estimate": created_edges,
            "node_ids_by_dimension": node_ids_by_dimension,
        }

    def _extract_history_fragments(self, text: str, *, limit: int = 5) -> list[str]:
        source = str(text or "").strip()
        if not source:
            return []
        history_hints = (
            "childhood",
            "детств",
            "истори",
            "background",
            "family",
            "семья",
            "школ",
            "универс",
            "опыт",
        )
        rows: list[str] = []
        for sentence in self._split_daily_sentences(source):
            lowered = sentence.casefold()
            if any(hint in lowered for hint in history_hints):
                rows.append(sentence)
        if not rows:
            rows = self._split_daily_sentences(source)[:2]
        return self._dedupe_limited(rows, limit=max(1, int(limit)))

    @staticmethod
    def _merge_dimensions(
        base: Mapping[str, Any] | None,
        extra: Mapping[str, Any] | None,
    ) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for key in _USER_DIMENSION_BINDINGS:
            out[key] = []
            base_rows = base.get(key) if isinstance(base, Mapping) else []
            extra_rows = extra.get(key) if isinstance(extra, Mapping) else []
            for source in (base_rows, extra_rows):
                if isinstance(source, (list, tuple, set)):
                    for item in source:
                        token = " ".join(str(item or "").split()).strip()
                        if token:
                            out[key].append(token)
                else:
                    token = " ".join(str(source or "").split()).strip()
                    if token:
                        out[key].append(token)
            # Deduplicate case-insensitively.
            seen: set[str] = set()
            deduped: list[str] = []
            for item in out[key]:
                marker = item.casefold()
                if marker in seen:
                    continue
                seen.add(marker)
                deduped.append(item)
            out[key] = deduped
        return out

    def _infer_user_update_json_from_text(
        self,
        *,
        text: str,
        display_name: str,
        language: str,
        use_llm_profile: bool,
    ) -> dict[str, Any]:
        source_text = str(text or "").strip()
        if not source_text:
            return {
                "source": "none",
                "confidence": 0.0,
                "llm_error": "",
                "profile": {},
                "personality": {},
                "dimensions": {key: [] for key in _USER_DIMENSION_BINDINGS},
                "history_fragments": [],
            }

        entity: dict[str, Any] = {}
        personality: dict[str, Any] = {}
        source = "heuristic"
        confidence = 0.56
        llm_error = ""

        if use_llm_profile and self.profile_llm_fn is not None:
            try:
                inferred = self.infer_profile_from_text(
                    {
                        "text": source_text,
                        "entity_type_hint": "human",
                        "create_graph": False,
                        "save_json": False,
                    }
                )
                normalized = self._as_mapping(inferred.get("profile_json"))
                entity = self._as_mapping(normalized.get("entity"))
                personality = self._as_mapping(normalized.get("personality"))
                source = "llm"
                confidence = 0.82
            except Exception as exc:
                llm_error = str(exc)

        first_name = str(entity.get("first_name", "") or "").strip()
        last_name = str(entity.get("last_name", "") or "").strip()
        if not first_name:
            first_name, inferred_last = self._infer_name_from_text(source_text)
            if not last_name:
                last_name = inferred_last

        summary = str(entity.get("description", "") or "").strip()
        if not summary:
            summary = source_text[:420]
        summary = " ".join(summary.split()).strip()

        history_fragments = self._extract_history_fragments(source_text, limit=6)
        history_summary = " ".join(history_fragments).strip()

        profile_patch: dict[str, Any] = {
            "name": str(entity.get("name", "") or display_name or "").strip() or display_name,
            "first_name": first_name,
            "last_name": last_name,
            "description": summary,
            "profile_text": source_text,
            "history": history_summary,
            "language": str(language or "en").strip() or "en",
            "updated_via": source,
        }

        dimensions = self._extract_user_dimensions(
            {
                "text": source_text,
                "profile_text": source_text,
                "profile": {
                    "text": source_text,
                },
                "personality": personality,
            }
        )

        return {
            "source": source,
            "confidence": confidence,
            "llm_error": llm_error,
            "profile": profile_patch,
            "personality": personality,
            "dimensions": dimensions,
            "history_fragments": history_fragments,
        }

    @staticmethod
    def _autoruns_auto_rows_from_profile(profile: Mapping[str, Any], *, query: str = "") -> list[dict[str, Any]]:
        os_name = str(((profile.get("device") or {}).get("os", "") if isinstance(profile, Mapping) else "") or "")
        browser = str(((profile.get("device") or {}).get("browser", "") if isinstance(profile, Mapping) else "") or "")
        platform_name = os_name.casefold()
        query_text = " ".join(str(query or "").split()).strip()

        def row(
            name: str,
            location: str,
            *,
            category: str,
            description: str,
            publisher: str,
            image_path: str,
            launch: str,
            verified: str,
            signer: str,
            virus_total: str,
            enabled: bool = True,
        ) -> dict[str, Any]:
            return {
                "entry_name": name,
                "entry_location": location,
                "enabled": enabled,
                "category": category,
                "profile": "current_user",
                "description": description,
                "publisher": publisher,
                "image_path": image_path,
                "launch_string": launch,
                "timestamp_utc": "",
                "signer": signer,
                "verified": verified,
                "virus_total": virus_total,
                "sha1": "",
                "md5": "",
                "source_query": query_text,
                "source_mode": "client_process_inference",
            }

        if "windows" in platform_name:
            return [
                row(
                    "OneDrive",
                    r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
                    category="Logon",
                    description="Cloud sync startup entry detected by platform profile.",
                    publisher="Microsoft Corporation",
                    image_path=r"C:\Program Files\Microsoft OneDrive\OneDrive.exe",
                    launch=r"\"C:\Program Files\Microsoft OneDrive\OneDrive.exe\"",
                    verified="Signed",
                    signer="Microsoft Corporation",
                    virus_total="0/74",
                ),
                row(
                    "SecurityHealthSystray",
                    r"HKLM\Software\Microsoft\Windows\CurrentVersion\Run",
                    category="Logon",
                    description="Windows security tray service.",
                    publisher="Microsoft Corporation",
                    image_path=r"C:\Windows\System32\SecurityHealthSystray.exe",
                    launch=r"C:\Windows\System32\SecurityHealthSystray.exe",
                    verified="Signed",
                    signer="Microsoft Corporation",
                    virus_total="0/74",
                ),
                row(
                    "BrowserUpdateAgent",
                    "Scheduled Tasks",
                    category="Scheduled Tasks",
                    description=f"Updater task inferred from browser profile: {browser or 'Unknown Browser'}.",
                    publisher=browser or "Unknown",
                    image_path=r"C:\Program Files\Browser\updater.exe",
                    launch=r"C:\Program Files\Browser\updater.exe --startup-task",
                    verified="Unknown",
                    signer="",
                    virus_total="0/74",
                ),
            ]
        if "linux" in platform_name:
            return [
                row(
                    "systemd --user session",
                    "~/.config/systemd/user",
                    category="User Services",
                    description="User-level systemd services inferred from Linux session.",
                    publisher="systemd",
                    image_path="/usr/lib/systemd/systemd",
                    launch="/usr/lib/systemd/systemd --user",
                    verified="Signed",
                    signer="Linux Distribution",
                    virus_total="0/74",
                ),
                row(
                    "desktop autostart",
                    "~/.config/autostart",
                    category="Desktop Autostart",
                    description="Desktop autostart entries potentially active for current user profile.",
                    publisher="Desktop Environment",
                    image_path="~/.config/autostart/*.desktop",
                    launch="xdg-autostart",
                    verified="Unknown",
                    signer="",
                    virus_total="0/74",
                ),
                row(
                    "browser background service",
                    "~/.config",
                    category="Background Service",
                    description=f"Background browser task inferred from detected browser: {browser or 'Unknown Browser'}.",
                    publisher=browser or "Unknown",
                    image_path="~/.config/browser/background",
                    launch="browser --background",
                    verified="Unknown",
                    signer="",
                    virus_total="0/74",
                ),
            ]
        if "android" in platform_name:
            return [
                row(
                    "BOOT_COMPLETED receiver",
                    "Android Manifest",
                    category="Boot Receiver",
                    description="App component that can react on device boot.",
                    publisher="Android Application",
                    image_path="apk://manifest/BOOT_COMPLETED",
                    launch="android.intent.action.BOOT_COMPLETED",
                    verified="Unknown",
                    signer="",
                    virus_total="0/74",
                ),
                row(
                    "Foreground service restart",
                    "WorkManager / Service",
                    category="Service",
                    description="Persistent foreground/background restart behavior inferred for Android app context.",
                    publisher="Android Application",
                    image_path="apk://service/foreground",
                    launch="startForegroundService",
                    verified="Unknown",
                    signer="",
                    virus_total="0/74",
                ),
            ]
        return [
            row(
                "session startup task",
                "session profile",
                category="Startup",
                description="Generic startup behavior inferred from client telemetry.",
                publisher="Unknown",
                image_path="unknown",
                launch="startup",
                verified="Unknown",
                signer="",
                virus_total="0/74",
            )
        ]

    @staticmethod
    def _autoruns_row_label(row: Mapping[str, Any]) -> str:
        entry_name = str(row.get("entry_name", "") or "").strip()
        if entry_name:
            return entry_name
        image_path = str(row.get("image_path", "") or "").strip().replace("\\", "/")
        if image_path:
            name = image_path.rsplit("/", 1)[-1].strip()
            if name:
                return name
        launch = str(row.get("launch_string", "") or "").strip().replace("\\", "/")
        if launch:
            token = launch.split()[0].strip("\"' ")
            if token:
                return token.rsplit("/", 1)[-1]
        return "autorun_entry"

    @staticmethod
    def _autoruns_risk_score(row: Mapping[str, Any]) -> float:
        score = 0.08
        positives = int(row.get("vt_positives", 0) or 0)
        total = int(row.get("vt_total", 0) or 0)
        if positives > 0:
            score += 0.28
            score += min(0.34, float(positives) / float(max(1, total)))

        verified = str(row.get("verified", "") or "").strip().casefold()
        signer = str(row.get("signer", "") or "").strip()
        if not verified or "not verified" in verified or "unsigned" in verified:
            score += 0.16
        if not signer:
            score += 0.12

        location = str(row.get("entry_location", "") or "").casefold()
        if any(hint in location for hint in _AUTORUNS_RISK_LOCATION_HINTS):
            score += 0.14

        enabled = row.get("enabled", None)
        if enabled is False:
            score -= 0.05
        return max(0.0, min(1.0, score))

    @staticmethod
    def _autoruns_risk_level(score: float) -> str:
        value = max(0.0, min(1.0, float(score)))
        if value >= 0.75:
            return "high"
        if value >= 0.45:
            return "medium"
        return "low"

    def _build_profile_graph(self, profile: Mapping[str, Any]) -> dict[str, Any]:
        entity = self._as_mapping(profile.get("entity"))
        personality = self._as_mapping(profile.get("personality"))
        entity_type = str(entity.get("type", "human") or "human").strip().lower()

        if entity_type == "human":
            person = self._as_mapping(profile.get("person"))
            root = self.create_node(
                {
                    "node_type": "human",
                    "first_name": str(entity.get("first_name", "") or "").strip(),
                    "last_name": str(entity.get("last_name", "") or "").strip(),
                    "bio": str(entity.get("description", "") or "").strip(),
                    "employment": self._as_list(profile.get("employment")),
                    "attributes": {
                        "name": str(entity.get("name", "") or "").strip(),
                        "traits": self._as_list(personality.get("traits")),
                        "values": self._as_list(personality.get("values")),
                        "fears": self._as_list(personality.get("fears")),
                        "goals": self._as_list(personality.get("goals")),
                        "preferences": self._as_list(personality.get("preferences")),
                        "desires": self._as_list(personality.get("desires")),
                        "principles": self._as_list(personality.get("principles")),
                        "opportunities": self._as_list(personality.get("opportunities")),
                        "abilities": self._as_list(personality.get("abilities")),
                        "access": self._as_list(personality.get("access")),
                        "knowledge": self._as_list(personality.get("knowledge")),
                        "assets": self._as_list(personality.get("assets")),
                        "person_id": self._to_int(person.get("id"), default=0),
                        "birth_date": str(person.get("birth_date", "") or "").strip(),
                    },
                    "state": {"influence": 0.5, "trust": 0.5},
                }
            )
        elif entity_type == "company":
            root = self.create_node(
                {
                    "node_type": "company",
                    "name": str(entity.get("name", "") or "").strip(),
                    "industry": str(entity.get("industry", "") or "").strip(),
                    "description": str(entity.get("description", "") or "").strip(),
                    "attributes": {
                        "values": self._as_list(personality.get("values")),
                        "goals": self._as_list(personality.get("goals")),
                        "principles": self._as_list(personality.get("principles")),
                        "opportunities": self._as_list(personality.get("opportunities")),
                    },
                    "state": {"influence": 0.5},
                }
            )
        else:
            root_name = str(entity.get("name", "") or "").strip()
            root = self.create_node(
                {
                    "node_type": entity_type if entity_type in {"technology", "generic"} else "generic",
                    "attributes": {
                        "name": root_name,
                        "description": str(entity.get("description", "") or "").strip(),
                        "industry": str(entity.get("industry", "") or "").strip(),
                        "values": self._as_list(personality.get("values")),
                        "goals": self._as_list(personality.get("goals")),
                        "principles": self._as_list(personality.get("principles")),
                    },
                    "state": {"influence": 0.5},
                }
            )

        root_id = int(root["node"]["id"])
        created_nodes = 1
        created_edges = 0

        for job in self._as_list(profile.get("employment")):
            item = self._as_mapping(job)
            company_name = self._sanitize_company_name(item.get("company_name", ""))
            if not company_name:
                continue
            company_node, created = self._ensure_shared_node(
                node_type="company",
                identity_key="name",
                identity_value=company_name,
                attributes={
                    "name": company_name,
                    "industry": str(item.get("industry", "") or "").strip(),
                    "description": str(item.get("description", "") or "").strip(),
                },
            )
            if created:
                created_nodes += 1
            self._connect_nodes(
                from_node=root_id,
                to_node=company_node.id,
                relation_type="works_at",
                weight=self._level_to_weight(item.get("importance_score", 0.75), default=0.75),
                logic_rule="profile_employment",
            )
            created_edges += 1

        for skill in self._as_list(profile.get("skills")):
            item = self._as_mapping(skill)
            name = str(item.get("name", "") or "").strip()
            if not name:
                continue
            identifier = re.sub(r"[^a-z0-9]+", "_", self._normalize_token(name)).strip("_") or name
            skill_node, created = self._ensure_shared_node(
                node_type="skill",
                identity_key="identifier",
                identity_value=identifier,
                attributes={
                    "identifier": identifier,
                    "name": name,
                    "category": str(item.get("category", "") or "").strip(),
                    "description": str(item.get("description", "") or "").strip(),
                },
            )
            if created:
                created_nodes += 1
            self._connect_nodes(
                from_node=root_id,
                to_node=skill_node.id,
                relation_type="has_skill",
                weight=self._level_to_weight(item.get("weight", item.get("level", 0.6)), default=0.6),
                logic_rule="profile_skill",
            )
            created_edges += 1

        for language in self._as_list(profile.get("languages")):
            item = self._as_mapping(language)
            code = self._normalize_language_code(
                str(item.get("name", "") or "").strip(),
                str(item.get("code", "") or "").strip(),
            )
            if not code:
                continue
            language_node, created = self._ensure_shared_node(
                node_type="language",
                identity_key="code",
                identity_value=code,
                attributes={
                    "code": code,
                    "name": str(item.get("name", "") or code.upper()).strip(),
                    "family": str(item.get("family", "") or "").strip(),
                    "origin": str(item.get("origin", "") or "").strip(),
                    "script": str(item.get("script", "") or "").strip(),
                    "description": str(item.get("description", "") or "").strip(),
                },
            )
            if created:
                created_nodes += 1
            self._connect_nodes(
                from_node=root_id,
                to_node=language_node.id,
                relation_type="speaks_language",
                weight=self._level_to_weight(item.get("weight", item.get("proficiency", 0.6)), default=0.6),
                logic_rule="profile_language",
            )
            created_edges += 1

        for capability in self._as_list(profile.get("capabilities")):
            item = self._as_mapping(capability)
            name = str(item.get("name", "") or "").strip()
            if not name:
                continue
            identifier = re.sub(r"[^a-z0-9]+", "_", self._normalize_token(name)).strip("_") or name
            capability_node, created = self._ensure_shared_node(
                node_type="capability",
                identity_key="identifier",
                identity_value=identifier,
                attributes={
                    "identifier": identifier,
                    "name": name,
                    "category": str(item.get("category", "") or "").strip(),
                    "description": str(item.get("description", "") or "").strip(),
                },
            )
            if created:
                created_nodes += 1
            self._connect_nodes(
                from_node=root_id,
                to_node=capability_node.id,
                relation_type="has_capability",
                weight=self._level_to_weight(item.get("weight", item.get("level", 0.6)), default=0.6),
                logic_rule="profile_capability",
            )
            created_edges += 1

        for link in self._as_list(profile.get("links")):
            item = self._as_mapping(link)
            target_type = str(item.get("target_type", "") or "generic").strip().lower() or "generic"
            target_name = str(item.get("target_name", "") or "").strip()
            target_identifier = str(item.get("target_identifier", "") or "").strip()
            if not target_name and not target_identifier:
                continue
            concept_id = self._to_int(item.get("concept_id"), default=0)
            if target_type == "concept" and concept_id > 0:
                identity_key = "concept_id"
                identity_value = str(concept_id)
            else:
                identity_key = "identifier"
                identity_value = target_identifier or target_name
            target_node, created = self._ensure_shared_node(
                node_type=target_type,
                identity_key=identity_key,
                identity_value=identity_value,
                attributes={
                    "identifier": target_identifier or identity_value,
                    "name": target_name or identity_value,
                    "description": str(item.get("description", "") or "").strip(),
                    "concept_id": concept_id if concept_id > 0 else None,
                    "new_concept": self._to_bool(item.get("new_concept", False)),
                },
            )
            if created:
                created_nodes += 1
            edge_metadata = {
                "concept_id": concept_id if concept_id > 0 else None,
                "new_concept": self._to_bool(item.get("new_concept", False)),
                "embedding_vector": self._normalize_embedding_vector(item.get("embedding_vector")),
                "confidence": self._level_to_weight(item.get("confidence", item.get("weight", 0.6)), default=0.6),
                "details": self._as_mapping(item.get("details")),
            }
            self._connect_nodes(
                from_node=root_id,
                to_node=target_node.id,
                relation_type=str(item.get("relation_type", "") or "related_to"),
                weight=self._level_to_weight(item.get("weight", 0.6), default=0.6),
                logic_rule="profile_link",
                metadata=edge_metadata,
            )
            created_edges += 1

        dimension_binding = self._bind_user_dimensions(
            root_node_id=root_id,
            dimensions=personality,
            logic_rule="profile_dimension",
        )
        created_nodes += int(dimension_binding.get("created_nodes_estimate", 0))
        created_edges += int(dimension_binding.get("created_edges_estimate", 0))

        return {
            "root_node_id": root_id,
            "created_nodes_estimate": created_nodes,
            "created_edges_estimate": created_edges,
            "dimension_binding": dimension_binding,
            **self.snapshot_payload(),
        }

    @staticmethod
    def _profile_exports_dir() -> Path:
        return Path("data/profile_exports")

    def _write_profile_export(
        self,
        *,
        source_text: str,
        prompt: str,
        raw_output: str,
        profile_json: Mapping[str, Any],
    ) -> str:
        export_dir = self._profile_exports_dir()
        export_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = int((time.time() % 1) * 1000)
        path = export_dir / f"profile_{stamp}_{suffix:03d}.json"
        payload = {
            "created_at": time.time(),
            "source_text": source_text,
            "prompt": prompt,
            "raw_output": raw_output,
            "profile_json": dict(profile_json),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)

    def infer_profile_from_text(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        source_text = str(payload.get("text", "") or "").strip()
        if not source_text:
            raise ValueError("text is required")
        entity_type_hint = str(payload.get("entity_type_hint", "human") or "human").strip().lower() or "human"
        create_graph = str(payload.get("create_graph", "1")).strip().lower() not in {"0", "false", "no", "off"}
        save_json = str(payload.get("save_json", "1")).strip().lower() not in {"0", "false", "no", "off"}

        if self.profile_llm_fn is None:
            raise ValueError(
                "local profile LLM is unavailable. Put GGUF model files into ./models "
                "or set LOCAL_GGUF_MODEL=/absolute/path/to/model.gguf"
            )

        prompt = self.profile_prompt_template(
            text=source_text,
            entity_type_hint=entity_type_hint,
        )
        raw_output = str(self.profile_llm_fn(prompt) or "").strip()
        parsed = self._extract_json_from_llm_output(raw_output)
        if parsed is None:
            raise ValueError("LLM output does not contain parseable JSON")

        normalized = self._normalize_profile_payload(
            parsed,
            source_text=source_text,
            entity_type_hint=entity_type_hint,
        )

        graph_payload: dict[str, Any] = self.snapshot_payload()
        if create_graph:
            graph_payload = self._build_profile_graph(normalized)

        export_path = ""
        if save_json:
            export_path = self._write_profile_export(
                source_text=source_text,
                prompt=prompt,
                raw_output=raw_output,
                profile_json=normalized,
            )

        return {
            "prompt": prompt,
            "raw_output": raw_output,
            "profile_json": normalized,
            "profile_json_file": export_path,
            **graph_payload,
        }

    def _metrics(self) -> GraphMetrics:
        snapshot = self.api.engine.snapshot()
        nodes = dict(snapshot.get("nodes", {}) or {})
        edges = list(snapshot.get("edges", []) or [])

        relation_counts = Counter(
            str(item.get("relation_type", "") or "") for item in edges if isinstance(item, Mapping)
        )
        node_type_counts = Counter(
            str((item or {}).get("type", "generic") or "generic")
            for item in nodes.values()
            if isinstance(item, Mapping)
        )

        return GraphMetrics(
            node_count=len(nodes),
            edge_count=len(edges),
            relation_counts=dict(sorted(relation_counts.items())),
            node_type_counts=dict(sorted(node_type_counts.items())),
        )

    @staticmethod
    def _serialize_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
        nodes_raw = dict(snapshot.get("nodes", {}) or {})
        edges_raw = list(snapshot.get("edges", []) or [])

        nodes: list[dict[str, Any]] = []
        for raw_id, payload in nodes_raw.items():
            try:
                node_id = int(raw_id)
            except Exception:
                continue
            if not isinstance(payload, Mapping):
                continue
            nodes.append(
                {
                    "id": node_id,
                    "type": str(payload.get("type", "generic") or "generic"),
                    "attributes": dict(payload.get("attributes", {}) or {}),
                    "state": dict(payload.get("state", {}) or {}),
                }
            )

        edges: list[dict[str, Any]] = []
        for item in edges_raw:
            if not isinstance(item, Mapping):
                continue
            try:
                edges.append(
                    {
                        "from": int(item.get("from")),
                        "to": int(item.get("to")),
                        "relation_type": str(item.get("relation_type", "") or ""),
                        "weight": float(item.get("weight", 1.0) or 1.0),
                        "direction": str(item.get("direction", "directed") or "directed"),
                        "logic_rule": str(item.get("logic_rule", "explicit") or "explicit"),
                        "metadata": dict(item.get("metadata", {}) or {}),
                    }
                )
            except Exception:
                continue

        nodes.sort(key=lambda row: row["id"])
        return {
            "nodes": nodes,
            "edges": edges,
        }

    def snapshot_payload(self) -> dict[str, Any]:
        snapshot = self.api.engine.snapshot()
        metrics = self._metrics()
        return {
            "snapshot": self._serialize_snapshot(snapshot),
            "metrics": {
                "node_count": metrics.node_count,
                "edge_count": metrics.edge_count,
                "relation_counts": metrics.relation_counts,
                "node_type_counts": metrics.node_type_counts,
            },
        }

    def list_node_types(self) -> list[str]:
        return sorted(self.api.engine.node_types.keys())

    def add_graph_event_listener(self, listener: Callable[[Any], None]) -> None:
        self.api.engine.add_event_listener(listener)

    def remove_graph_event_listener(self, listener: Callable[[Any], None]) -> None:
        self.api.engine.remove_event_listener(listener)

    def create_node(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        node_type = str(payload.get("node_type", "generic") or "generic").strip().lower() or "generic"
        attributes = dict(self._safe_json_loads(payload.get("attributes"), {}) or {})
        state = self._to_state(self._safe_json_loads(payload.get("state"), {}) or {})

        if node_type == "human":
            bio_text = str(payload.get("bio", attributes.get("bio", "")) or "").strip()
            profile_text = str(payload.get("profile_text", attributes.get("profile_text", "")) or "").strip()
            employment_text = str(payload.get("employment_text", attributes.get("employment_text", "")) or "").strip()
            profile_source = "\n".join(part for part in (profile_text, bio_text) if part).strip()
            structured = self._parse_structured_profile(
                "\n".join(part for part in (profile_text, bio_text, employment_text) if part).strip()
            )

            first_name = str(payload.get("first_name", attributes.get("first_name", "")) or "").strip()
            last_name = str(payload.get("last_name", attributes.get("last_name", "")) or "").strip()

            if not first_name:
                first_name = self._clean_field_value((structured.get("first_name") or [""])[0])
            if not last_name:
                last_name = self._clean_field_value((structured.get("last_name") or [""])[0])

            full_name = self._clean_field_value((structured.get("full_name") or [""])[0])
            if full_name and (not first_name or not last_name):
                parts = [part for part in re.split(r"\s+", full_name) if part]
                if parts:
                    first_name = first_name or parts[0]
                    if len(parts) >= 2 and not last_name:
                        last_name = " ".join(parts[1:])

            if not first_name:
                inferred_first, inferred_last = self._infer_name_from_text(profile_source)
                first_name = inferred_first or first_name
                last_name = last_name or inferred_last

            if profile_text:
                attributes.setdefault("profile_text", profile_text)
            if employment_text:
                attributes.setdefault("employment_text", employment_text)
            if first_name:
                attributes.setdefault("first_name", first_name)
            if last_name:
                attributes.setdefault("last_name", last_name)
            if bio_text:
                attributes.setdefault("bio", bio_text)

            for key in _PROFILE_LIST_KEYS:
                if attributes.get(key):
                    continue
                items: list[str] = []
                for raw in structured.get(key, []):
                    items.extend(self._split_list_values(raw))
                if items:
                    attributes[key] = items

            for key in ("date_of_birth", "primary_language"):
                if attributes.get(key):
                    continue
                value = self._clean_field_value((structured.get(key) or [""])[0])
                if value:
                    attributes[key] = value

            for key in ("age", "height_cm", "weight_kg"):
                if attributes.get(key) not in (None, ""):
                    continue
                raw_value = self._clean_field_value((structured.get(key) or [""])[0])
                numeric = self._extract_number(raw_value)
                if numeric is None:
                    continue
                if key == "age":
                    attributes[key] = int(round(numeric))
                else:
                    attributes[key] = float(numeric)

            employment_raw = self._safe_json_loads(payload.get("employment"), []) or []
            employment: list[dict[str, Any]] = []
            if isinstance(employment_raw, list):
                for row in employment_raw:
                    if not isinstance(row, Mapping):
                        continue
                    employment.append(
                        {
                            "status": str(row.get("status", "") or "").strip(),
                            "importance_score": self._to_float(row.get("importance_score", 1.0), 1.0),
                            "company_name": str(row.get("company_name", "") or "").strip(),
                            "company_attributes": dict(row.get("company_attributes", {}) or {}),
                        }
                    )

            if not employment and employment_text:
                employment.extend(self._parse_employment_free_text(employment_text))

            if not employment:
                for row in structured.get("employment", []):
                    employment.extend(self._parse_employment_free_text(row))

            if not employment:
                status_hint = self._clean_field_value((structured.get("job_status") or [""])[0])
                company_hint = self._clean_field_value((structured.get("company_name") or [""])[0])
                if status_hint or company_hint:
                    employment.append(
                        {
                            "status": status_hint or "employee",
                            "importance_score": 0.8,
                            "company_name": company_hint,
                            "company_attributes": {},
                        }
                    )

            if not employment:
                employment.extend(self._infer_employment_from_text(profile_source))

            deduped_employment: list[dict[str, Any]] = []
            seen_employment: set[tuple[str, str]] = set()
            for row in employment:
                status = str(row.get("status", "") or "").strip()
                company_name = str(row.get("company_name", "") or "").strip()
                if not status and not company_name:
                    continue
                key = (status.casefold(), company_name.casefold())
                if key in seen_employment:
                    continue
                seen_employment.add(key)
                deduped_employment.append(
                    {
                        "status": status or "employee",
                        "importance_score": max(0.0, min(1.0, self._to_float(row.get("importance_score", 1.0), 1.0))),
                        "company_name": company_name,
                        "company_attributes": dict(row.get("company_attributes", {}) or {}),
                    }
                )
            employment = deduped_employment

            node = self.api.create_human(
                first_name=first_name,
                last_name=last_name,
                bio=bio_text,
                employment=employment,
                attributes=attributes,
                state=state,
            )
        elif node_type == "company":
            node = self.api.create_company(
                name=str(payload.get("name", attributes.get("name", "")) or "").strip(),
                industry=str(payload.get("industry", attributes.get("industry", "")) or "").strip(),
                description=str(payload.get("description", attributes.get("description", "")) or "").strip(),
                attributes=attributes,
                state=state,
            )
        else:
            node = self.api.engine.create_node(
                node_type,
                attributes=attributes,
                state=state,
            )

        return {
            "node": {
                "id": node.id,
                "type": node.type,
                "attributes": dict(node.attributes),
                "state": dict(node.state),
            },
            **self.snapshot_payload(),
        }

    def create_edge(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        edge_payload = self._normalize_edge_payload(payload)
        if not edge_payload["relation_type"]:
            raise ValueError("relation_type is required")

        created = self.api.connect(
            edge_payload["from_node"],
            edge_payload["to_node"],
            relation_type=edge_payload["relation_type"],
            weight=edge_payload["weight"],
            direction=edge_payload["direction"],
            logic_rule=edge_payload["logic_rule"],
        )
        return {
            "created": bool(created),
            "edge": edge_payload,
            **self.snapshot_payload(),
        }

    def update_node(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        node_id = self._to_int(payload.get("node_id", 0), 0)
        if node_id <= 0:
            raise ValueError("node_id is required")
        node = self.api.engine.get_node(node_id)
        if node is None:
            raise ValueError(f"node {node_id} not found")

        attributes_raw = self._safe_json_loads(payload.get("attributes"), {})
        if isinstance(attributes_raw, Mapping):
            for key, value in attributes_raw.items():
                node.attributes[str(key)] = value

        state_raw = self._safe_json_loads(payload.get("state"), {})
        if isinstance(state_raw, Mapping):
            for key, value in self._to_state(dict(state_raw)).items():
                node.state[str(key)] = value

        # Keep compatibility with shortcut fields used by create_node payloads.
        for field in ("first_name", "last_name", "bio", "name", "industry", "description"):
            if field in payload and str(payload.get(field, "")).strip():
                node.attributes[field] = str(payload.get(field, "")).strip()

        self.api.engine._record_event(  # noqa: SLF001
            "node_updated",
            {
                "node_id": node.id,
                "node_type": node.type,
                "node": {
                    "id": node.id,
                    "type": node.type,
                    "attributes": dict(node.attributes),
                    "state": dict(node.state),
                },
            },
        )
        return {
            "updated": True,
            "node": {
                "id": node.id,
                "type": node.type,
                "attributes": dict(node.attributes),
                "state": dict(node.state),
            },
            **self.snapshot_payload(),
        }

    def delete_node(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        node_id = self._to_int(payload.get("node_id", 0), 0)
        if node_id <= 0:
            raise ValueError("node_id is required")
        node = self.api.engine.get_node(node_id)
        if node is None:
            raise ValueError(f"node {node_id} not found")

        before_edges = len(self.api.engine.store.edges)
        removed_edge_rows = [
            {
                "from": int(edge.from_node),
                "to": int(edge.to_node),
                "relation_type": str(edge.relation_type),
                "direction": str(edge.direction),
            }
            for edge in self.api.engine.store.edges
            if edge.from_node == node_id or edge.to_node == node_id
        ]
        self.api.engine.store.edges = [
            edge
            for edge in self.api.engine.store.edges
            if edge.from_node != node_id and edge.to_node != node_id
        ]
        removed_edges = before_edges - len(self.api.engine.store.edges)
        self.api.engine.store.nodes.pop(node_id, None)
        self.api.engine._record_event(  # noqa: SLF001
            "node_deleted",
            {
                "node_id": node_id,
                "node_type": node.type,
                "removed_edges": removed_edges,
                "removed_edge_refs": removed_edge_rows,
            },
        )
        return {
            "deleted": True,
            "node_id": node_id,
            "removed_edges": removed_edges,
            **self.snapshot_payload(),
        }

    def _find_edge_record(
        self,
        *,
        from_node: int,
        to_node: int,
        relation_type: str,
        direction: str,
    ):
        for edge in self.api.engine.store.edges:
            if (
                int(edge.from_node) == int(from_node)
                and int(edge.to_node) == int(to_node)
                and str(edge.relation_type) == str(relation_type)
                and str(edge.direction) == str(direction)
            ):
                return edge
        return None

    def update_edge(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        edge_payload = self._normalize_edge_payload(payload)
        edge = self._find_edge_record(
            from_node=edge_payload["from_node"],
            to_node=edge_payload["to_node"],
            relation_type=edge_payload["relation_type"],
            direction=edge_payload["direction"],
        )
        if edge is None:
            raise ValueError("edge not found")

        edge.weight = max(0.0, min(1.0, self._to_float(payload.get("weight", edge.weight), edge.weight)))
        edge.logic_rule = str(payload.get("logic_rule", edge.logic_rule) or edge.logic_rule)
        metadata_raw = self._safe_json_loads(payload.get("metadata"), edge.metadata)
        if isinstance(metadata_raw, Mapping):
            edge.metadata = dict(metadata_raw)

        self.api.engine._record_event(  # noqa: SLF001
            "edge_updated_manual",
            {
                "from": edge.from_node,
                "to": edge.to_node,
                "relation_type": edge.relation_type,
                "direction": edge.direction,
                "weight": edge.weight,
                "logic_rule": edge.logic_rule,
            },
        )
        return {
            "updated": True,
            "edge": {
                "from": edge.from_node,
                "to": edge.to_node,
                "relation_type": edge.relation_type,
                "weight": edge.weight,
                "direction": edge.direction,
                "logic_rule": edge.logic_rule,
                "metadata": dict(edge.metadata),
            },
            **self.snapshot_payload(),
        }

    def delete_edge(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        edge_payload = self._normalize_edge_payload(payload)
        before = len(self.api.engine.store.edges)
        self.api.engine.store.edges = [
            edge
            for edge in self.api.engine.store.edges
            if not (
                int(edge.from_node) == edge_payload["from_node"]
                and int(edge.to_node) == edge_payload["to_node"]
                and str(edge.relation_type) == edge_payload["relation_type"]
                and str(edge.direction) == edge_payload["direction"]
            )
        ]
        removed = before - len(self.api.engine.store.edges)
        self.api.engine._record_event(  # noqa: SLF001
            "edge_deleted",
            {
                "from": edge_payload["from_node"],
                "to": edge_payload["to_node"],
                "relation_type": edge_payload["relation_type"],
                "direction": edge_payload["direction"],
                "removed": removed,
            },
        )
        return {
            "deleted": removed > 0,
            "removed": removed,
            **self.snapshot_payload(),
        }

    def simulate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        seed_node_ids = payload.get("seed_node_ids", [])
        if isinstance(seed_node_ids, str):
            parsed = []
            for part in seed_node_ids.split(","):
                token = part.strip()
                if not token:
                    continue
                try:
                    parsed.append(int(token))
                except Exception:
                    continue
            seed_node_ids = parsed
        if not isinstance(seed_node_ids, list):
            seed_node_ids = []

        out = self.api.simulate(
            seed_node_ids=[int(item) for item in seed_node_ids if str(item).strip()],
            recursive_depth=max(1, self._to_int(payload.get("recursive_depth", 2), 2)),
            propagation_steps=max(1, self._to_int(payload.get("propagation_steps", 3), 3)),
            damping=max(0.0, min(1.0, self._to_float(payload.get("damping", 0.15), 0.15))),
            activation=str(payload.get("activation", "tanh") or "tanh"),
            infer_rounds=max(1, self._to_int(payload.get("infer_rounds", 1), 1)),
        )
        return {
            "result": out,
            **self.snapshot_payload(),
        }

    def list_events(self, *, limit: int = 200, event_type: str = "") -> list[dict[str, Any]]:
        rows = self.api.get_events(limit=max(1, min(2000, int(limit))), event_type=(event_type or None))
        return [
            {
                "id": event.id,
                "event_type": event.event_type,
                "timestamp": event.timestamp,
                "payload": dict(event.payload),
            }
            for event in rows
        ]

    def reward_event(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        event_id = self._to_int(payload.get("event_id", -1), -1)
        reward = self._to_float(payload.get("reward", 0.0), 0.0)
        learning_rate = self._to_float(payload.get("learning_rate", 0.15), 0.15)
        changed = self.api.reward_event(event_id, reward=reward, learning_rate=learning_rate)
        return {
            "changed": bool(changed),
            "event_id": event_id,
            "reward": reward,
            "learning_rate": learning_rate,
            **self.snapshot_payload(),
        }

    def reinforce_relation(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        relation_type = str(payload.get("relation_type", "") or "").strip()
        reward = self._to_float(payload.get("reward", 0.0), 0.0)
        learning_rate = self._to_float(payload.get("learning_rate", 0.15), 0.15)
        updated = self.api.reinforce_relation(
            relation_type,
            reward=reward,
            learning_rate=learning_rate,
        )
        return {
            "updated": int(updated),
            "relation_type": relation_type,
            "reward": reward,
            "learning_rate": learning_rate,
            **self.snapshot_payload(),
        }

    def persist(self) -> dict[str, Any]:
        ok = self.api.persist()
        return {
            "ok": bool(ok),
            **self.snapshot_payload(),
        }

    def load(self) -> dict[str, Any]:
        ok = self.api.load()
        return {
            "ok": bool(ok),
            **self.snapshot_payload(),
        }

    def clear(self) -> dict[str, Any]:
        self.api.engine.store.nodes.clear()
        self.api.engine.store.edges.clear()
        self.api.engine.clear_event_log()
        self.api.engine._next_node_id = 1  # noqa: SLF001
        return self.snapshot_payload()

    def seed_foundational_graph(self) -> dict[str, Any]:
        created_nodes = 0
        created_edges = 0
        domain_ids: dict[str, int] = {}

        def _resolve_named_node(name: str):
            node = self._find_node_by_identity(node_type="domain", key="name", value=name)
            if node is not None:
                return node
            return self._find_node_by_identity(node_type="concept", key="name", value=name)

        for domain, concepts in _FOUNDATIONAL_DOMAIN_GRAPH.items():
            domain_details = dict(_FOUNDATIONAL_DOMAIN_DETAILS.get(domain, {}) or {})
            domain_description = str(
                domain_details.get("description", f"Foundational domain: {domain}")
            ).strip() or f"Foundational domain: {domain}"
            domain_node, created = self._ensure_shared_node(
                node_type="domain",
                identity_key="name",
                identity_value=domain,
                attributes={
                    "name": domain,
                    "description": domain_description,
                    "history_intro": str(domain_details.get("history_intro", "") or "").strip(),
                    "connections": list(domain_details.get("connections", []) or []),
                    "usage": list(domain_details.get("usage", []) or []),
                },
            )
            if domain_details:
                # Always refresh enriched editorial fields for known domains in existing graphs.
                domain_node.attributes["description"] = domain_description
                domain_node.attributes["history_intro"] = str(domain_details.get("history_intro", "") or "").strip()
                domain_node.attributes["connections"] = list(domain_details.get("connections", []) or [])
                domain_node.attributes["usage"] = list(domain_details.get("usage", []) or [])
            domain_ids[domain] = int(domain_node.id)
            if created:
                created_nodes += 1

            for concept_name in concepts:
                concept_node, concept_created = self._ensure_shared_node(
                    node_type="concept",
                    identity_key="name",
                    identity_value=concept_name,
                    attributes={
                        "name": concept_name,
                        "domain": domain,
                        "description": f"Core concept in {domain}",
                        "concept_id": self._stable_temp_concept_id(f"{domain}:{concept_name}"),
                    },
                )
                if concept_created:
                    created_nodes += 1
                self._connect_nodes(
                    from_node=domain_node.id,
                    to_node=concept_node.id,
                    relation_type="contains_concept",
                    weight=0.9,
                    logic_rule="foundation_seed",
                )
                created_edges += 1

        for concept_id, item in _GLOBAL_CONCEPT_CATALOG.items():
            name = str(item.get("name", "") or "").strip() or f"concept_{concept_id}"
            concept_node, created = self._ensure_shared_node(
                node_type="concept",
                identity_key="name",
                identity_value=name,
                attributes={
                    "name": name,
                    "aliases": list(item.get("aliases", []) or []),
                    "concept_id": int(concept_id),
                    "description": "Global concept catalog seed",
                },
            )
            if created:
                created_nodes += 1
            arts_domain_id = domain_ids.get("Arts")
            if arts_domain_id:
                self._connect_nodes(
                    from_node=arts_domain_id,
                    to_node=concept_node.id,
                    relation_type="includes",
                    weight=0.78,
                    logic_rule="catalog_seed",
                )
                created_edges += 1

        for source_name, target_name, relation_type, weight in _FOUNDATIONAL_DOMAIN_RELATIONS:
            source_node = _resolve_named_node(source_name)
            target_node = _resolve_named_node(target_name)
            if source_node is None or target_node is None:
                continue
            self._connect_nodes(
                from_node=source_node.id,
                to_node=target_node.id,
                relation_type=relation_type,
                weight=weight,
                logic_rule="foundation_relations",
            )
            created_edges += 1

        return {
            "created_nodes_estimate": created_nodes,
            "created_edges_estimate": created_edges,
            **self.snapshot_payload(),
        }

    @staticmethod
    def _split_daily_sentences(text: str) -> list[str]:
        rows = re.split(r"[\n\r]+|(?<=[.!?;:])\s+", str(text or ""))
        out: list[str] = []
        for row in rows:
            cleaned = " ".join(str(row).split()).strip(" \t\r\n-•*;")
            if cleaned:
                out.append(cleaned)
        return out

    @staticmethod
    def _contains_hint(text: str, hints: tuple[str, ...]) -> bool:
        source = str(text or "").casefold()
        for hint in hints:
            if str(hint or "").casefold() in source:
                return True
        return False

    @staticmethod
    def _dedupe_limited(items: list[str], *, limit: int = 6) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            cleaned = " ".join(str(item or "").split()).strip()
            if not cleaned:
                continue
            token = cleaned.casefold()
            if token in seen:
                continue
            seen.add(token)
            out.append(cleaned)
            if len(out) >= max(1, int(limit)):
                break
        return out

    def _extract_daily_signals(self, text: str) -> dict[str, list[str]]:
        goals: list[str] = []
        problems: list[str] = []
        wins: list[str] = []

        for sentence in self._split_daily_sentences(text):
            lowered = sentence.casefold()
            if self._contains_hint(lowered, _DAILY_GOAL_HINTS):
                goals.append(sentence)
            if self._contains_hint(lowered, _DAILY_PROBLEM_HINTS):
                problems.append(sentence)
            if self._contains_hint(lowered, _DAILY_WIN_HINTS):
                wins.append(sentence)

        for line in str(text or "").splitlines():
            lowered = line.casefold().strip()
            if lowered.startswith(("goal:", "цель:", "цели:", "план:", "plan:", "want:")):
                goals.append(line.split(":", 1)[-1])
            if lowered.startswith(("problem:", "issue:", "проблема:", "риск:", "risk:")):
                problems.append(line.split(":", 1)[-1])
            if lowered.startswith(("done:", "result:", "итог:", "сделал:", "win:")):
                wins.append(line.split(":", 1)[-1])

        return {
            "goals": self._dedupe_limited(goals, limit=8),
            "problems": self._dedupe_limited(problems, limit=8),
            "wins": self._dedupe_limited(wins, limit=8),
        }

    @staticmethod
    def _score_clamp(value: float) -> int:
        return max(0, min(100, int(round(float(value)))))

    def _build_daily_scores(
        self,
        *,
        text: str,
        goals: list[str],
        problems: list[str],
        wins: list[str],
    ) -> dict[str, Any]:
        source = str(text or "").casefold()
        goal_factor = min(1.0, len(goals) / 4.0)
        problem_factor = min(1.0, len(problems) / 4.0)
        win_factor = min(1.0, len(wins) / 4.0)

        positive_hits = sum(1 for hint in _DAILY_POSITIVE_HINTS if hint.casefold() in source)
        negative_hits = sum(1 for hint in _DAILY_NEGATIVE_HINTS if hint.casefold() in source)
        learning_hits = sum(
            1
            for hint in ("learn", "study", "read", "research", "practice", "учусь", "изучаю", "читаю")
            if hint in source
        )

        focus = self._score_clamp(56 + (24 * goal_factor) + (10 * win_factor) - (17 * problem_factor))
        energy = self._score_clamp(54 + (8 * positive_hits) - (8 * negative_hits) + (7 * win_factor))
        consistency = self._score_clamp(50 + (19 * goal_factor) + (16 * win_factor) - (11 * problem_factor))
        stress_management = self._score_clamp(61 - (26 * problem_factor) - (7 * negative_hits) + (5 * positive_hits))
        learning_progress = self._score_clamp(46 + (8 * learning_hits) + (12 * goal_factor) + (6 * win_factor))
        wellbeing = self._score_clamp(55 + (10 * positive_hits) - (11 * negative_hits) - (8 * problem_factor))

        scores = {
            "focus": focus,
            "energy": energy,
            "consistency": consistency,
            "stress_management": stress_management,
            "learning_progress": learning_progress,
            "wellbeing": wellbeing,
        }
        overall = self._score_clamp(sum(scores.values()) / max(1, len(scores)))
        sorted_low = sorted(scores.items(), key=lambda item: item[1])
        improvement_targets = [
            {
                "metric": name,
                "score": value,
                "target": min(100, value + 15),
            }
            for name, value in sorted_low[:3]
        ]
        return {
            "overall": overall,
            "metrics": scores,
            "improvement_targets": improvement_targets,
        }

    def _build_daily_recommendations(
        self,
        *,
        goals: list[str],
        problems: list[str],
        wins: list[str],
        scores: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        metrics = dict(scores.get("metrics", {}) or {})

        for idx, problem in enumerate(problems[:3], start=1):
            rows.append(
                {
                    "id": f"risk_{idx}",
                    "type": "risk_mitigation",
                    "priority": "high",
                    "confidence": round(max(0.55, 0.84 - (idx * 0.06)), 2),
                    "title": "Reduce friction around key problem",
                    "advice": f"Address: {problem}",
                    "rationale": "Detected recurring blocker in daily journal.",
                }
            )

        for idx, goal in enumerate(goals[:3], start=1):
            rows.append(
                {
                    "id": f"goal_{idx}",
                    "type": "goal_execution",
                    "priority": "medium",
                    "confidence": round(max(0.58, 0.87 - (idx * 0.05)), 2),
                    "title": "Turn goal into next executable step",
                    "advice": f"Next step for goal: {goal}",
                    "rationale": "Goals are strongest when linked to concrete immediate actions.",
                }
            )

        if int(metrics.get("stress_management", 100)) < 60:
            rows.append(
                {
                    "id": "stress_protocol",
                    "type": "self_regulation",
                    "priority": "high",
                    "confidence": 0.79,
                    "title": "Run short anti-stress protocol",
                    "advice": "Schedule two focused 25-minute blocks and one 10-minute reset between them.",
                    "rationale": "Stress score is below stable threshold.",
                }
            )

        if int(metrics.get("consistency", 100)) < 62:
            rows.append(
                {
                    "id": "consistency_loop",
                    "type": "process",
                    "priority": "medium",
                    "confidence": 0.74,
                    "title": "Stabilize daily loop",
                    "advice": "Define a fixed start ritual and one fixed completion ritual for every workday.",
                    "rationale": "Consistency score indicates execution variance.",
                }
            )

        if wins:
            rows.append(
                {
                    "id": "reinforce_wins",
                    "type": "reinforcement",
                    "priority": "low",
                    "confidence": 0.7,
                    "title": "Reinforce what already works",
                    "advice": f"Repeat patterns behind this result: {wins[0]}",
                    "rationale": "Successful behavior should be explicitly reused.",
                }
            )

        rows = self._dedupe_limited([json.dumps(item, sort_keys=True) for item in rows], limit=32)
        parsed_rows = [json.loads(item) for item in rows]

        bounded = max(3, min(5, int(limit)))
        while len(parsed_rows) < 3:
            idx = len(parsed_rows) + 1
            parsed_rows.append(
                {
                    "id": f"generic_plan_{idx}",
                    "type": "planning",
                    "priority": "medium",
                    "confidence": 0.62,
                    "title": "Define one priority outcome",
                    "advice": "Pick one measurable result for today and close it before context switching.",
                    "rationale": "Fallback recommendation when journal signal is sparse.",
                }
            )
        return parsed_rows[:bounded]

    def _bind_daily_mode_graph(
        self,
        *,
        user_id: str,
        display_name: str,
        text: str,
        goals: list[str],
        problems: list[str],
        wins: list[str],
        recommendations: list[Mapping[str, Any]],
        scores: Mapping[str, Any],
    ) -> dict[str, Any]:
        now = time.time()
        checksum = int(zlib.crc32(str(text).encode("utf-8")) & 0xFFFFFFFF)
        entry_id = f"{user_id}:{int(now)}:{checksum:08x}"

        person_node, _ = self._ensure_shared_node(
            node_type="person_profile",
            identity_key="user_id",
            identity_value=user_id,
            attributes={
                "user_id": user_id,
                "name": display_name,
                "description": "Personal adaptive profile for daily decision support.",
            },
        )
        entry_node, _ = self._ensure_shared_node(
            node_type="daily_entry",
            identity_key="entry_id",
            identity_value=entry_id,
            attributes={
                "entry_id": entry_id,
                "name": f"Daily Journal {time.strftime('%Y-%m-%d')}",
                "text": text,
                "goals": goals,
                "problems": problems,
                "wins": wins,
                "scores": dict(scores),
                "created_at": now,
            },
        )
        self._connect_nodes(
            from_node=person_node.id,
            to_node=entry_node.id,
            relation_type="logged_day",
            weight=0.95,
            logic_rule="daily_mode",
        )

        goal_ids: list[int] = []
        for goal in goals:
            goal_node, _ = self._ensure_shared_node(
                node_type="goal",
                identity_key="name",
                identity_value=goal,
                attributes={"name": goal, "description": "Daily extracted goal"},
            )
            goal_ids.append(int(goal_node.id))
            self._connect_nodes(
                from_node=entry_node.id,
                to_node=goal_node.id,
                relation_type="targets_goal",
                weight=0.86,
                logic_rule="daily_mode",
            )

        problem_ids: list[int] = []
        for problem in problems:
            problem_node, _ = self._ensure_shared_node(
                node_type="problem",
                identity_key="name",
                identity_value=problem,
                attributes={"name": problem, "description": "Daily extracted blocker/risk"},
            )
            problem_ids.append(int(problem_node.id))
            self._connect_nodes(
                from_node=entry_node.id,
                to_node=problem_node.id,
                relation_type="reports_problem",
                weight=0.88,
                logic_rule="daily_mode",
            )

        recommendation_ids: list[int] = []
        for item in recommendations:
            title = str(item.get("title", "") or "").strip()
            advice = str(item.get("advice", "") or "").strip()
            if not title and not advice:
                continue
            identifier = f"{entry_id}:{len(recommendation_ids)}"
            rec_node, _ = self._ensure_shared_node(
                node_type="recommendation",
                identity_key="identifier",
                identity_value=identifier,
                attributes={
                    "identifier": identifier,
                    "name": title or "Recommendation",
                    "advice": advice,
                    "priority": str(item.get("priority", "medium") or "medium"),
                    "confidence": self._to_float(item.get("confidence", 0.6), 0.6),
                    "rationale": str(item.get("rationale", "") or "").strip(),
                },
            )
            recommendation_ids.append(int(rec_node.id))
            self._connect_nodes(
                from_node=entry_node.id,
                to_node=rec_node.id,
                relation_type="advises",
                weight=max(0.4, min(1.0, self._to_float(item.get("confidence", 0.6), 0.6))),
                logic_rule="daily_mode",
            )

        return {
            "person_node_id": int(person_node.id),
            "entry_node_id": int(entry_node.id),
            "goal_node_ids": goal_ids,
            "problem_node_ids": problem_ids,
            "recommendation_node_ids": recommendation_ids,
        }

    def project_daily_mode(
        self,
        payload: Mapping[str, Any],
        *,
        request_headers: Mapping[str, Any] | None = None,
        request_ip: str = "",
    ) -> dict[str, Any]:
        text = str(payload.get("text", "") or "").strip()
        if not text:
            raise ValueError("text is required")

        user_id = str(payload.get("user_id", "default_user") or "default_user").strip() or "default_user"
        display_name = str(payload.get("display_name", user_id) or user_id).strip() or user_id
        language = str(payload.get("language", "en") or "en").strip() or "en"
        session_id = str(payload.get("session_id", "daily") or "daily").strip() or "daily"
        auto_snapshot = self._to_bool(payload.get("auto_snapshot", True))
        recommendation_count = max(3, min(5, self._to_int(payload.get("recommendation_count", 4), 4)))
        run_knowledge = self._to_bool(payload.get("run_knowledge_analysis", True))
        apply_profile_update = self._to_bool(payload.get("apply_profile_update", True))
        use_llm_profile = self._to_bool(payload.get("use_llm_profile", True))
        include_client_profile = self._to_bool(payload.get("include_client_profile", False))
        client_payload = self._as_mapping(payload.get("client"))

        signals = self._extract_daily_signals(text)
        goals = list(signals.get("goals", []) or [])
        problems = list(signals.get("problems", []) or [])
        wins = list(signals.get("wins", []) or [])
        scores = self._build_daily_scores(text=text, goals=goals, problems=problems, wins=wins)
        recommendations = self._build_daily_recommendations(
            goals=goals,
            problems=problems,
            wins=wins,
            scores=scores,
            limit=recommendation_count,
        )
        graph_binding = self._bind_daily_mode_graph(
            user_id=user_id,
            display_name=display_name,
            text=text,
            goals=goals,
            problems=problems,
            wins=wins,
            recommendations=recommendations,
            scores=scores,
        )

        living_result: dict[str, Any] = {}
        knowledge_result: dict[str, Any] = {}
        profile_update: dict[str, Any] = {}
        profile_update_error = ""

        if apply_profile_update:
            try:
                profile_update = self.project_user_graph_update(
                    {
                        "user_id": user_id,
                        "display_name": display_name,
                        "text": text,
                        "language": language,
                        "session_id": session_id,
                        "use_llm_profile": use_llm_profile,
                        "include_client_profile": include_client_profile,
                        "client": client_payload,
                        "profile_text": text,
                        "goals": goals,
                        "fears": problems,
                        "knowledge": wins,
                    },
                    request_headers=request_headers,
                    request_ip=request_ip,
                )
            except Exception as exc:
                profile_update_error = str(exc)

        if self.living_system is not None:
            living_result = self.living_process(
                {
                    "text": text,
                    "user_id": user_id,
                    "display_name": display_name,
                    "language": language,
                    "session_id": session_id,
                    "auto_snapshot": auto_snapshot,
                }
            )
            if run_knowledge:
                knowledge_result = self.living_knowledge_analyze(
                    {
                        "text": text,
                        "user_id": user_id,
                        "display_name": display_name,
                        "language": language,
                        "branch_id": "daily",
                        "apply_changes": False,
                        "sources": [],
                    }
                )

        if self.living_system is not None:
            project_status = self.project_evaluate({"user_id": user_id})
        else:
            project_status = {
                "graph": self.snapshot_payload(),
                "events": self.list_events(limit=120),
                "living_health": {"enabled": False},
                "knowledge_evaluation": {"enabled": False},
            }

        return {
            "journal_entry": {
                "user_id": user_id,
                "display_name": display_name,
                "text": text,
                "language": language,
                "session_id": session_id,
                "created_at": time.time(),
            },
            "signals": {
                "goals": goals,
                "problems": problems,
                "wins": wins,
            },
            "recommendations": recommendations,
            "improvement_scores": scores,
            "graph_binding": graph_binding,
            "profile_update": profile_update,
            "profile_update_json": self._as_mapping(profile_update.get("profile_update_json")),
            "profile_update_error": profile_update_error,
            "living": living_result,
            "knowledge": knowledge_result,
            "project_status": project_status,
        }

    @staticmethod
    def _default_demo_narrative(persona_name: str, language: str = "ru") -> str:
        name = str(persona_name or "Alexa").strip() or "Alexa"
        lang = str(language or "en").strip().lower()
        if lang.startswith("ru"):
            return (
                f"Меня зовут {name}. Я инженер автономных систем и создатель небольшого AI-сервиса. "
                "В детстве я рос в семье преподавателей, любил музыку и математику, участвовал в кружке робототехники. "
                "Сейчас я доволен тем, что могу строить полезные продукты и помогать людям учиться быстрее. "
                "Я хочу создать устойчивую платформу знаний, усилить приватность пользователей и выйти на международный рынок."
            )
        if lang.startswith("hy"):
            return (
                f"Իմ անունը {name} է։ Ես ինքնավար համակարգերի ինժեներ եմ և փոքր AI ծառայության ստեղծող։ "
                "Մանկության տարիներին մեծացել եմ ուսուցիչների ընտանիքում, սիրել եմ երաժշտություն ու մաթեմատիկա, "
                "մասնակցել եմ ռոբոտաշինության խմբակին։ Հիմա գոհ եմ, որ կարող եմ օգտակար պրոդուկտներ կառուցել և մարդկանց օգնել "
                "ավելի արագ սովորել։ Ցանկանում եմ ստեղծել կայուն գիտելիքային հարթակ, ուժեղացնել օգտատերերի գաղտնիությունը "
                "և դուրս գալ միջազգային շուկա։"
            )
        if lang.startswith("zh"):
            return (
                f"我叫{name}。我是自治系统工程师，也是一个小型 AI 服务的创建者。"
                "我在教师家庭中长大，喜欢音乐和数学，小时候参加过机器人社团。"
                "现在我很满意自己能构建有用的产品并帮助人们更快学习。"
                "我希望打造一个可持续的知识平台，强化用户隐私，并进入国际市场。"
            )
        if lang.startswith("es"):
            return (
                f"Me llamo {name}. Soy ingeniero de sistemas autónomos y creador de un pequeño servicio de IA. "
                "En mi infancia crecí en una familia de docentes, me gustaban la música y las matemáticas, y participé "
                "en un club de robótica. Hoy me satisface poder construir productos útiles y ayudar a las personas a aprender "
                "más rápido. Quiero crear una plataforma de conocimiento sostenible, reforzar la privacidad de los usuarios "
                "y llegar al mercado internacional."
            )
        if lang.startswith("pt"):
            return (
                f"Meu nome é {name}. Sou engenheiro de sistemas autônomos e criador de um pequeno serviço de IA. "
                "Na infância cresci em uma família de professores, gostava de música e matemática e participei de um clube "
                "de robótica. Hoje fico satisfeito por conseguir construir produtos úteis e ajudar pessoas a aprender mais rápido. "
                "Quero criar uma plataforma de conhecimento sustentável, fortalecer a privacidade dos usuários e entrar no mercado internacional."
            )
        return (
            f"My name is {name}. I am an autonomous systems engineer and creator of a small AI service. "
            "As a child, I grew up in a family of teachers, loved music and mathematics, and joined a robotics club. "
            "Now I am satisfied that I can build useful products and help people learn faster. "
            "I want to create a resilient knowledge platform, strengthen user privacy, and enter the international market."
        )

    def watch_demo(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        data = dict(payload or {})
        persona_name = str(data.get("persona_name", "Alexa") or "Alexa").strip() or "Alexa"
        language = str(data.get("language", "ru") or "ru").strip() or "ru"
        reset_graph = self._to_bool(data.get("reset_graph", True))
        use_llm = self._to_bool(data.get("use_llm", True))
        narrative = (
            str(data.get("narrative", "") or "").strip()
            or self._default_demo_narrative(persona_name=persona_name, language=language)
        )

        if reset_graph:
            self.clear()
            self.seed_foundational_graph()

        llm_mode = "fallback"
        llm_error = ""
        root_node_id = 0

        if use_llm and self.profile_llm_fn is not None:
            try:
                inferred = self.infer_profile_from_text(
                    {
                        "text": narrative,
                        "entity_type_hint": "human",
                        "create_graph": True,
                        "save_json": False,
                    }
                )
                root_node_id = self._to_int(inferred.get("root_node_id", 0), 0)
                llm_mode = "llm"
            except Exception as exc:
                llm_error = str(exc)

        if llm_mode != "llm":
            alexa = self.create_node(
                {
                    "node_type": "human",
                    "first_name": persona_name,
                    "last_name": "Demov",
                    "bio": (
                        "Autonomous systems engineer focused on resilient product architecture "
                        "and semantic knowledge workflows."
                    ),
                    "profile_text": (
                        "детство: семья преподавателей, кружок робототехники, любовь к музыке и математике\n"
                        "доволен: полезные продукты, сильная команда, рост клиентов\n"
                        "цели: построить платформу знаний, улучшить приватность, расширить проект"
                    ),
                    "employment": [
                        {
                            "status": "founder",
                            "importance_score": 0.95,
                            "company_name": "Autograph Labs",
                            "company_attributes": {
                                "industry": "AI Platform",
                                "description": "Semantic graph runtime with local LLM support.",
                            },
                        }
                    ],
                    "state": {
                        "influence": 0.72,
                        "trust": 0.64,
                        "satisfaction": 0.78,
                        "ambition": 0.91,
                    },
                }
            )
            root_node_id = int(alexa["node"]["id"])

            stage_node, _ = self._ensure_shared_node(
                node_type="life_stage",
                identity_key="name",
                identity_value="Childhood",
                attributes={
                    "name": "Childhood",
                    "description": "Early family and school period with foundational habits.",
                },
            )
            wants_node, _ = self._ensure_shared_node(
                node_type="goal",
                identity_key="name",
                identity_value="Build a resilient knowledge platform",
                attributes={
                    "name": "Build a resilient knowledge platform",
                    "description": "Long-term system objective for explainable knowledge operations.",
                },
            )
            satisfied_node, _ = self._ensure_shared_node(
                node_type="satisfaction",
                identity_key="name",
                identity_value="Shipping useful products",
                attributes={
                    "name": "Shipping useful products",
                    "description": "Current satisfaction source for demo persona.",
                },
            )

            for concept_name, relation_type, weight in (
                ("Music", "loves", 0.84),
                ("Mathematics", "studies", 0.88),
                ("Computer Science", "builds", 0.95),
            ):
                concept_node, _ = self._ensure_shared_node(
                    node_type="concept",
                    identity_key="name",
                    identity_value=concept_name,
                    attributes={"name": concept_name},
                )
                self._connect_nodes(
                    from_node=root_node_id,
                    to_node=concept_node.id,
                    relation_type=relation_type,
                    weight=weight,
                    logic_rule="demo_persona_link",
                )

            self._connect_nodes(
                from_node=root_node_id,
                to_node=stage_node.id,
                relation_type="experienced",
                weight=0.79,
                logic_rule="demo_persona_link",
            )
            self._connect_nodes(
                from_node=root_node_id,
                to_node=wants_node.id,
                relation_type="wants",
                weight=0.93,
                logic_rule="demo_persona_link",
            )
            self._connect_nodes(
                from_node=root_node_id,
                to_node=satisfied_node.id,
                relation_type="satisfied_with",
                weight=0.86,
                logic_rule="demo_persona_link",
            )

        self.simulate(
            {
                "seed_node_ids": [root_node_id] if root_node_id > 0 else [],
                "recursive_depth": 2,
                "propagation_steps": 3,
                "damping": 0.12,
                "activation": "tanh",
                "infer_rounds": 2,
            }
        )
        return {
            "demo": {
                "persona_name": persona_name,
                "narrative": narrative,
                "mode": llm_mode,
                "llm_error": llm_error,
                "root_node_id": root_node_id,
            },
            **self.snapshot_payload(),
        }

    def seed_demo(self) -> dict[str, Any]:
        # Backward-compatible endpoint used by UI buttons and tests.
        return self.watch_demo({"persona_name": "Alexa", "reset_graph": True, "use_llm": True})

    def capture_client_profile(
        self,
        payload: Mapping[str, Any] | None,
        *,
        request_headers: Mapping[str, Any],
        request_ip: str,
    ) -> dict[str, Any]:
        profile = build_client_profile(
            request_headers=request_headers,
            request_client_ip=request_ip,
            payload=payload,
        )

        session_id = str(profile.get("session_id", "") or "").strip()
        if not session_id:
            basis = f"{request_ip}|{profile['device']['os']}|{profile['device']['browser']}"
            session_id = f"s_{zlib.crc32(basis.encode('utf-8')) & 0xFFFFFFFF:08x}"
            profile["session_id"] = session_id

        user_id = str(profile.get("user_id", "") or "").strip() or f"client_{session_id}"
        profile["user_id"] = user_id

        session_node, _ = self._ensure_shared_node(
            node_type="client_session",
            identity_key="session_id",
            identity_value=session_id,
            attributes={
                "session_id": session_id,
                "user_id": user_id,
                "name": f"Client Session {session_id}",
            },
        )
        ip_node, _ = self._ensure_shared_node(
            node_type="ip_address",
            identity_key="ip",
            identity_value=str(profile["network"]["ip"]["ip"]),
            attributes={
                "ip": str(profile["network"]["ip"]["ip"]),
                "private": bool(profile["network"]["ip"].get("private", False)),
                "name": f"IP {profile['network']['ip']['ip']}",
            },
        )
        os_node, _ = self._ensure_shared_node(
            node_type="operating_system",
            identity_key="name",
            identity_value=str(profile["device"]["os"]),
            attributes={"name": str(profile["device"]["os"])},
        )
        browser_node, _ = self._ensure_shared_node(
            node_type="browser",
            identity_key="name",
            identity_value=str(profile["device"]["browser"]),
            attributes={"name": str(profile["device"]["browser"])},
        )
        network_node, _ = self._ensure_shared_node(
            node_type="network_profile",
            identity_key="identifier",
            identity_value=session_id,
            attributes={
                "identifier": session_id,
                "vpn_proxy_suspected": bool(profile["network"]["vpn_proxy_suspected"]),
                "vpn_proxy_reasons": list(profile["network"]["vpn_proxy_reasons"]),
                "forward_chain": list(profile["network"]["forward_chain"]),
            },
        )
        network_node.attributes["vpn_proxy_suspected"] = bool(profile["network"]["vpn_proxy_suspected"])
        network_node.attributes["vpn_proxy_reasons"] = list(profile["network"]["vpn_proxy_reasons"])
        network_node.attributes["forward_chain"] = list(profile["network"]["forward_chain"])
        network_node.attributes["last_seen_at"] = float(profile["timestamp"])
        session_node.attributes["last_seen_at"] = float(profile["timestamp"])
        ip_node.attributes["last_seen_at"] = float(profile["timestamp"])
        os_node.attributes["last_seen_at"] = float(profile["timestamp"])
        browser_node.attributes["last_seen_at"] = float(profile["timestamp"])

        self._connect_nodes(
            from_node=session_node.id,
            to_node=ip_node.id,
            relation_type="connected_from",
            weight=0.95,
            logic_rule="client_introspection",
        )
        self._connect_nodes(
            from_node=session_node.id,
            to_node=os_node.id,
            relation_type="uses_os",
            weight=0.88,
            logic_rule="client_introspection",
        )
        self._connect_nodes(
            from_node=session_node.id,
            to_node=browser_node.id,
            relation_type="uses_browser",
            weight=0.88,
            logic_rule="client_introspection",
        )
        self._connect_nodes(
            from_node=session_node.id,
            to_node=network_node.id,
            relation_type="observed_network",
            weight=0.82,
            logic_rule="client_introspection",
        )

        return {
            "profile": profile,
            "semantic_binding": {
                "session_node_id": int(session_node.id),
                "ip_node_id": int(ip_node.id),
                "os_node_id": int(os_node.id),
                "browser_node_id": int(browser_node.id),
                "network_node_id": int(network_node.id),
            },
            **self.snapshot_payload(),
        }

    def project_user_graph_update(
        self,
        payload: Mapping[str, Any],
        *,
        request_headers: Mapping[str, Any] | None = None,
        request_ip: str = "",
    ) -> dict[str, Any]:
        user_id = str(payload.get("user_id", "default_user") or "default_user").strip() or "default_user"
        display_name = str(payload.get("display_name", user_id) or user_id).strip() or user_id
        language = str(payload.get("language", "en") or "en").strip() or "en"
        session_id = str(payload.get("session_id", "") or "").strip()
        text = str(payload.get("text", "") or "").strip()
        use_llm_profile = self._to_bool(payload.get("use_llm_profile", True))
        include_client_profile = self._to_bool(payload.get("include_client_profile", True))

        explicit_profile = self._as_mapping(payload.get("profile"))
        explicit_dimensions = self._extract_user_dimensions(payload)
        personalization = self._sanitize_personalization(payload.get("personalization"))
        feedback_items = self._normalize_feedback_items(payload.get("feedback_items"))
        if personalization:
            explicit_dimensions = self._merge_dimensions(
                explicit_dimensions,
                {
                    "goals": self._to_list_of_strings(personalization.get("focus_goals")),
                    "knowledge": self._to_list_of_strings(personalization.get("domain_focus")),
                },
            )
        inferred_update = self._infer_user_update_json_from_text(
            text=text,
            display_name=display_name,
            language=language,
            use_llm_profile=use_llm_profile,
        )
        inferred_dimensions = self._as_mapping(inferred_update.get("dimensions"))
        dimensions = self._merge_dimensions(explicit_dimensions, inferred_dimensions)
        profile_node = self._ensure_user_profile_node(user_id=user_id, display_name=display_name)

        inferred_profile = self._as_mapping(inferred_update.get("profile"))

        def _pick_text(*values: Any) -> str:
            for value in values:
                token = " ".join(str(value or "").split()).strip()
                if token:
                    return token
            return ""

        profile_node.attributes["name"] = _pick_text(
            payload.get("name"),
            explicit_profile.get("name"),
            inferred_profile.get("name"),
            display_name,
            user_id,
        )
        profile_node.attributes["display_name"] = display_name
        profile_node.attributes["language"] = language

        first_name = _pick_text(
            payload.get("first_name"),
            explicit_profile.get("first_name"),
            inferred_profile.get("first_name"),
        )
        last_name = _pick_text(
            payload.get("last_name"),
            explicit_profile.get("last_name"),
            inferred_profile.get("last_name"),
        )
        description = _pick_text(
            payload.get("description"),
            explicit_profile.get("description"),
            inferred_profile.get("description"),
        )
        history = _pick_text(
            payload.get("history"),
            explicit_profile.get("history"),
            inferred_profile.get("history"),
        )
        profile_text = _pick_text(
            payload.get("profile_text"),
            explicit_profile.get("profile_text"),
            text,
            inferred_profile.get("profile_text"),
        )

        if first_name:
            profile_node.attributes["first_name"] = first_name
        if last_name:
            profile_node.attributes["last_name"] = last_name
        if description:
            profile_node.attributes["description"] = description
        if history:
            profile_node.attributes["history"] = history
        if profile_text:
            profile_node.attributes["profile_text"] = profile_text

        profile_node.attributes["profile_update_source"] = str(
            inferred_update.get("source", "heuristic") or "heuristic"
        )
        profile_node.attributes["profile_update_confidence"] = round(
            self._to_float(inferred_update.get("confidence", 0.0), 0.0),
            4,
        )
        llm_error = str(inferred_update.get("llm_error", "") or "").strip()
        if llm_error:
            profile_node.attributes["profile_update_error"] = llm_error

        if personalization:
            profile_node.attributes["personalization"] = personalization
            profile_node.attributes["response_style"] = str(
                personalization.get("response_style", "adaptive")
            )
            profile_node.attributes["reasoning_depth"] = str(
                personalization.get("reasoning_depth", "balanced")
            )
            profile_node.attributes["risk_tolerance"] = str(
                personalization.get("risk_tolerance", "medium")
            )
            profile_node.attributes["tone"] = str(personalization.get("tone", "neutral"))
            if personalization.get("memory_notes"):
                profile_node.attributes["memory_notes"] = str(
                    personalization.get("memory_notes", "")
                )[:1200]

        for key, values in dimensions.items():
            if values:
                profile_node.attributes[key] = list(values)

        feedback_summary = {
            "new_items": 0,
            "accepted": 0,
            "rejected": 0,
            "stored_total": self._to_int(profile_node.attributes.get("feedback_total"), 0),
        }
        if feedback_items:
            accepted = 0
            rejected = 0
            for row in feedback_items:
                decision = str(row.get("decision", "") or "").strip()
                score = self._confidence(row.get("score", 0.0), 0.0)
                if decision in {"accept", "accepted", "like", "liked"} or score >= 0.66:
                    accepted += 1
                if decision in {"reject", "rejected", "dislike", "discard"} or score <= 0.34:
                    rejected += 1

            prev_total = self._to_int(profile_node.attributes.get("feedback_total"), 0)
            prev_accepted = self._to_int(profile_node.attributes.get("feedback_accepted"), 0)
            prev_rejected = self._to_int(profile_node.attributes.get("feedback_rejected"), 0)
            profile_node.attributes["feedback_total"] = prev_total + len(feedback_items)
            profile_node.attributes["feedback_accepted"] = prev_accepted + accepted
            profile_node.attributes["feedback_rejected"] = prev_rejected + rejected
            profile_node.attributes["feedback_last_at"] = float(time.time())
            profile_node.attributes["feedback_recent"] = feedback_items[-8:]
            feedback_summary = {
                "new_items": len(feedback_items),
                "accepted": accepted,
                "rejected": rejected,
                "stored_total": int(profile_node.attributes["feedback_total"]),
            }
        profile_node.attributes["updated_at"] = float(time.time())

        binding = self._bind_user_dimensions(
            root_node_id=int(profile_node.id),
            dimensions=dimensions,
            logic_rule="user_profile_update",
        )
        non_empty = {key: values for key, values in dimensions.items() if values}
        profile_update_json = {
            "source": str(inferred_update.get("source", "heuristic") or "heuristic"),
            "confidence": round(self._to_float(inferred_update.get("confidence", 0.0), 0.0), 4),
            "llm_error": llm_error,
            "profile": {
                "name": str(profile_node.attributes.get("name", "") or "").strip(),
                "first_name": str(profile_node.attributes.get("first_name", "") or "").strip(),
                "last_name": str(profile_node.attributes.get("last_name", "") or "").strip(),
                "description": str(profile_node.attributes.get("description", "") or "").strip(),
                "history": str(profile_node.attributes.get("history", "") or "").strip(),
                "profile_text": str(profile_node.attributes.get("profile_text", "") or "").strip(),
                "language": str(profile_node.attributes.get("language", "") or language),
            },
            "dimensions": non_empty,
            "personalization": personalization,
            "feedback_summary": feedback_summary,
            "history_fragments": self._as_list(inferred_update.get("history_fragments")),
        }

        client_binding: dict[str, Any] = {}
        client_profile: dict[str, Any] = {}
        client_profile_error = ""
        if include_client_profile:
            try:
                client_result = self.capture_client_profile(
                    {
                        "session_id": session_id,
                        "user_id": user_id,
                        "client": self._as_mapping(payload.get("client")),
                    },
                    request_headers=request_headers or {},
                    request_ip=str(request_ip or ""),
                )
                client_profile = self._as_mapping(client_result.get("profile"))
                client_binding = self._as_mapping(client_result.get("semantic_binding"))
                session_node_id = self._to_int(client_binding.get("session_node_id"), 0)
                if session_node_id > 0:
                    self._connect_nodes(
                        from_node=int(profile_node.id),
                        to_node=session_node_id,
                        relation_type="observed_in_session",
                        weight=0.84,
                        logic_rule="user_profile_update",
                    )
            except Exception as exc:
                client_profile_error = str(exc)

        return {
            "user_profile": {
                "user_id": user_id,
                "display_name": display_name,
                "profile_node_id": int(profile_node.id),
                "dimensions": non_empty,
            },
            "binding": binding,
            "profile_update_json": profile_update_json,
            "client_profile": client_profile,
            "client_semantic_binding": client_binding,
            "client_profile_error": client_profile_error,
            "personalization_applied": bool(personalization),
            "feedback_summary": feedback_summary,
            **self.snapshot_payload(),
        }

    def project_autoruns_import(
        self,
        payload: Mapping[str, Any],
        *,
        request_headers: Mapping[str, Any] | None = None,
        request_ip: str = "",
    ) -> dict[str, Any]:
        raw_text = str(payload.get("text", "") or "").strip()
        auto_detect = self._to_bool(payload.get("auto_detect", True))
        delimiter = str(payload.get("delimiter", "") or "").strip()
        query = str(payload.get("query", "") or "").strip()
        language = str(payload.get("language", "en") or "en").strip() or "en"
        user_id = str(payload.get("user_id", "default_user") or "default_user").strip() or "default_user"
        session_id = str(payload.get("session_id", "autoruns_session") or "autoruns_session").strip() or "autoruns_session"
        host_label = str(payload.get("host_label", "") or "").strip() or user_id
        max_rows = max(1, min(5000, self._to_int(payload.get("max_rows", 1000), 1000)))
        client_payload = self._as_mapping(payload.get("client"))

        parsed_rows = parse_autoruns_text(raw_text, delimiter=delimiter) if raw_text else []
        mode = "parsed_text"
        source_label = "sysinternals_autoruns"
        client_profile: dict[str, Any] = {}
        client_semantic_binding: dict[str, Any] = {}
        client_profile_error = ""

        if parsed_rows:
            rows = parsed_rows[:max_rows]
        else:
            if raw_text and not auto_detect:
                raise ValueError("autoruns payload could not be parsed")
            if not raw_text and not auto_detect:
                raise ValueError("text is required when auto_detect is false")

            mode = "auto_detected"
            source_label = "semantic_client_autoruns_inference"
            try:
                client_capture = self.capture_client_profile(
                    {
                        "session_id": session_id,
                        "user_id": user_id,
                        "client": client_payload,
                    },
                    request_headers=request_headers or {},
                    request_ip=str(request_ip or ""),
                )
                client_profile = self._as_mapping(client_capture.get("profile"))
                client_semantic_binding = self._as_mapping(client_capture.get("semantic_binding"))
            except Exception as exc:
                client_profile_error = str(exc)
                client_profile = build_client_profile(
                    request_headers=request_headers or {},
                    request_client_ip=str(request_ip or ""),
                    payload={
                        "session_id": session_id,
                        "user_id": user_id,
                        "client": client_payload,
                    },
                )
            rows = self._autoruns_auto_rows_from_profile(
                client_profile,
                query=(query or raw_text),
            )[:max_rows]
            if not rows:
                raise ValueError("autoruns auto-detect produced no entries")

        now = float(time.time())
        checksum_basis = raw_text or json.dumps(rows, ensure_ascii=False, sort_keys=True)
        checksum = int(zlib.crc32(checksum_basis.encode("utf-8")) & 0xFFFFFFFF)
        scan_id = f"{user_id}:{session_id}:{checksum:08x}"

        profile_node = self._ensure_user_profile_node(user_id=user_id, display_name=host_label)
        scan_node, _ = self._ensure_shared_node(
            node_type="autoruns_scan",
            identity_key="scan_id",
            identity_value=scan_id,
            attributes={
                "scan_id": scan_id,
                "user_id": user_id,
                "session_id": session_id,
                "host_label": host_label,
                "name": f"Autoruns Scan {session_id}",
                "created_at": now,
                "rows_total": len(rows),
                "query": query,
                "language": language,
                "mode": mode,
            },
        )
        scan_node.attributes["rows_total"] = len(rows)
        scan_node.attributes["source"] = source_label
        scan_node.attributes["updated_at"] = now

        self._connect_nodes(
            from_node=profile_node.id,
            to_node=scan_node.id,
            relation_type="has_startup_scan",
            weight=0.91,
            logic_rule="autoruns_import",
        )
        session_node_id = self._to_int(client_semantic_binding.get("session_node_id"), 0)
        if session_node_id > 0:
            self._connect_nodes(
                from_node=int(profile_node.id),
                to_node=session_node_id,
                relation_type="observed_in_session",
                weight=0.82,
                logic_rule="autoruns_import",
            )

        created_nodes = 0
        created_edges = 1 + (1 if session_node_id > 0 else 0)
        risk_counts = {"low": 0, "medium": 0, "high": 0}
        vt_positive_count = 0
        top_risky: list[dict[str, Any]] = []
        entry_node_ids: list[int] = []
        risk_node_ids: list[int] = []

        for idx, row in enumerate(rows):
            entry_name = self._autoruns_row_label(row)
            entry_location = str(row.get("entry_location", "") or "").strip()
            image_path = str(row.get("image_path", "") or "").strip()
            launch_string = str(row.get("launch_string", "") or "").strip()
            enabled = row.get("enabled", None)
            risk_score = self._autoruns_risk_score(row)
            risk_level = self._autoruns_risk_level(risk_score)
            risk_counts[risk_level] += 1

            if int(row.get("vt_positives", 0) or 0) > 0:
                vt_positive_count += 1

            identity_basis = "|".join([entry_name, entry_location, image_path, launch_string]).strip("|")
            if not identity_basis:
                identity_basis = f"entry:{idx + 1}"
            entry_identifier = f"autoruns:{zlib.crc32(identity_basis.encode('utf-8')) & 0xFFFFFFFF:08x}"

            entry_node, created = self._ensure_shared_node(
                node_type="autorun_entry",
                identity_key="identifier",
                identity_value=entry_identifier,
                attributes={
                    "identifier": entry_identifier,
                    "name": entry_name,
                    "entry_location": entry_location,
                    "category": str(row.get("category", "") or "").strip(),
                    "profile": str(row.get("profile", "") or "").strip(),
                    "description": str(row.get("description", "") or "").strip(),
                    "publisher": str(row.get("publisher", "") or "").strip(),
                    "image_path": image_path,
                    "launch_string": launch_string,
                    "timestamp_utc": str(row.get("timestamp_utc", "") or "").strip(),
                    "signer": str(row.get("signer", "") or "").strip(),
                    "verified": str(row.get("verified", "") or "").strip(),
                    "virus_total": str(row.get("virus_total", "") or "").strip(),
                    "vt_positives": int(row.get("vt_positives", 0) or 0),
                    "vt_total": int(row.get("vt_total", 0) or 0),
                    "sha1": str(row.get("sha1", "") or "").strip(),
                    "md5": str(row.get("md5", "") or "").strip(),
                    "enabled": enabled,
                },
            )
            if created:
                created_nodes += 1
            entry_node.attributes["risk_score"] = round(risk_score, 4)
            entry_node.attributes["risk_level"] = risk_level
            entry_node.attributes["last_scan_id"] = scan_id
            entry_node.attributes["last_seen_at"] = now

            self._connect_nodes(
                from_node=scan_node.id,
                to_node=entry_node.id,
                relation_type="observed_autorun",
                weight=max(0.45, min(0.98, 0.58 + risk_score * 0.4)),
                logic_rule="autoruns_import",
                metadata={
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "enabled": enabled,
                    "entry_location": entry_location,
                    "source": "autoruns",
                },
            )
            created_edges += 1
            entry_node_ids.append(int(entry_node.id))

            publisher = str(row.get("publisher", "") or "").strip()
            if publisher:
                publisher_identifier = re.sub(r"[^a-z0-9]+", "_", self._normalize_token(publisher)).strip("_") or publisher
                publisher_node, created_pub = self._ensure_shared_node(
                    node_type="publisher",
                    identity_key="identifier",
                    identity_value=f"publisher:{publisher_identifier}",
                    attributes={
                        "identifier": f"publisher:{publisher_identifier}",
                        "name": publisher,
                    },
                )
                if created_pub:
                    created_nodes += 1
                self._connect_nodes(
                    from_node=entry_node.id,
                    to_node=publisher_node.id,
                    relation_type="published_by",
                    weight=0.66,
                    logic_rule="autoruns_import",
                )
                created_edges += 1

            if risk_level in {"high", "medium"}:
                risk_identifier = f"autoruns_risk:{entry_identifier}"
                risk_node, created_risk = self._ensure_shared_node(
                    node_type="security_risk",
                    identity_key="identifier",
                    identity_value=risk_identifier,
                    attributes={
                        "identifier": risk_identifier,
                        "name": f"Startup Risk: {entry_name}",
                        "category": "startup_persistence",
                        "description": "Potential startup persistence or integrity risk from Autoruns scan.",
                    },
                )
                if created_risk:
                    created_nodes += 1
                risk_node.attributes["risk_score"] = round(risk_score, 4)
                risk_node.attributes["risk_level"] = risk_level
                risk_node.attributes["last_scan_id"] = scan_id
                self._connect_nodes(
                    from_node=entry_node.id,
                    to_node=risk_node.id,
                    relation_type="has_risk",
                    weight=max(0.45, min(1.0, risk_score)),
                    logic_rule="autoruns_import",
                )
                created_edges += 1
                risk_node_ids.append(int(risk_node.id))

            top_risky.append(
                {
                    "name": entry_name,
                    "identifier": entry_identifier,
                    "risk_score": round(risk_score, 4),
                    "risk_level": risk_level,
                    "entry_location": entry_location,
                    "enabled": enabled,
                    "virus_total": str(row.get("virus_total", "") or "").strip(),
                }
            )

        top_risky.sort(key=lambda item: float(item.get("risk_score", 0.0)), reverse=True)
        top_risky = top_risky[:20]

        return {
            "scan": {
                "scan_id": scan_id,
                "session_id": session_id,
                "user_id": user_id,
                "host_label": host_label,
                "mode": mode,
                "source": source_label,
                "rows_parsed": len(parsed_rows),
                "rows_processed": len(rows),
            },
            "summary": {
                "risk_counts": risk_counts,
                "virus_total_positive_entries": vt_positive_count,
                "high_risk_entries": int(risk_counts.get("high", 0)),
                "top_risky": top_risky,
            },
            "semantic_binding": {
                "profile_node_id": int(profile_node.id),
                "scan_node_id": int(scan_node.id),
                "entry_node_ids": entry_node_ids,
                "risk_node_ids": risk_node_ids,
            },
            "client_profile": client_profile,
            "client_semantic_binding": client_semantic_binding,
            "client_profile_error": client_profile_error,
            "created_nodes_estimate": created_nodes,
            "created_edges_estimate": created_edges,
            **self.snapshot_payload(),
        }

    def project_db_schema(self) -> dict[str, Any]:
        if self.living_system is None:
            return {
                "database_path": "",
                "generated_at": time.time(),
                "tables": [],
                "note": "living_system is disabled",
            }
        return self.living_system.store.describe_schema()

    def _living_required(self) -> LivingSystemEngine:
        if self.living_system is None:
            raise ValueError("living_system is disabled")
        return self.living_system

    def living_architecture(self) -> dict[str, Any]:
        return self._living_required().architecture_overview()

    def living_health(self) -> dict[str, Any]:
        return self._living_required().health_report()

    def living_process(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text", "") or "").strip()
        if not text:
            raise ValueError("text is required")
        user_id = str(payload.get("user_id", "default_user") or "default_user").strip() or "default_user"
        language = str(payload.get("language", "hy") or "hy").strip() or "hy"
        session_id = str(payload.get("session_id", "default") or "default").strip() or "default"
        auto_snapshot = self._to_bool(payload.get("auto_snapshot", True))

        display_name = str(payload.get("display_name", user_id) or user_id).strip() or user_id
        self._living_required().bootstrap_user(
            user_id=user_id,
            display_name=display_name,
            primary_language=language,
        )
        return self._living_required().process_input(
            text=text,
            user_id=user_id,
            language=language,
            session_id=session_id,
            auto_snapshot=auto_snapshot,
        )

    def living_graph_view(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        user_id = ""
        if payload:
            user_id = str(payload.get("user_id", "") or "").strip()
        return self._living_required().graph_view(user_id=user_id)

    def living_snapshot(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        reason = str(payload.get("reason", "manual") or "manual").strip() or "manual"
        user_id = str(payload.get("user_id", "") or "").strip()
        snapshot_id = self._living_required().create_snapshot(reason=reason, user_id=user_id)
        return {"snapshot_id": snapshot_id, "reason": reason, "user_id": user_id}

    def living_rollback(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        snapshot_id = self._to_int(payload.get("snapshot_id", 0), 0)
        if snapshot_id <= 0:
            raise ValueError("snapshot_id is required")
        return self._living_required().rollback(snapshot_id)

    def living_safe_mode(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        enabled = self._to_bool(payload.get("enabled", True))
        reason = str(payload.get("reason", "") or "").strip()
        return self._living_required().set_safe_mode(enabled, reason=reason)

    def living_human_override(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        enabled = self._to_bool(payload.get("enabled", True))
        reason = str(payload.get("reason", "") or "").strip()
        return self._living_required().set_human_override(enabled, reason=reason)

    def living_feedback(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._living_required().feedback_event(dict(payload))

    def living_evolution_plan(self) -> dict[str, Any]:
        return self._living_required().evolution_plan()

    def living_prompt_catalog(self) -> dict[str, Any]:
        return {"prompts": self._living_required().prompt_catalog()}

    def living_prompt_run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        prompt_name = str(payload.get("prompt_name", "") or "").strip()
        if not prompt_name:
            raise ValueError("prompt_name is required")
        variables = dict(self._safe_json_loads(payload.get("variables"), {}) or {})
        user_id = str(payload.get("user_id", "") or "").strip()
        session_id = str(payload.get("session_id", "") or "").strip()
        return self._living_required().run_prompt(
            prompt_name=prompt_name,
            variables=variables,
            user_id=user_id,
            session_id=session_id,
        )

    def living_project_map(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        max_files = 600
        if payload is not None:
            max_files = max(10, min(5000, self._to_int(payload.get("max_files", 600), 600)))
        return self._living_required().project_map(max_files=max_files)

    def living_file_create(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        relative_path = str(payload.get("relative_path", "") or "").strip()
        content = str(payload.get("content", "") or "")
        user_id = str(payload.get("user_id", "") or "").strip()
        if not relative_path:
            raise ValueError("relative_path is required")
        return self._living_required().create_file(
            relative_path=relative_path,
            content=content,
            user_id=user_id,
        )

    def living_file_update(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        relative_path = str(payload.get("relative_path", "") or "").strip()
        content = str(payload.get("content", "") or "")
        user_id = str(payload.get("user_id", "") or "").strip()
        if not relative_path:
            raise ValueError("relative_path is required")
        return self._living_required().update_file(
            relative_path=relative_path,
            content=content,
            user_id=user_id,
        )

    def living_file_delete(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        relative_path = str(payload.get("relative_path", "") or "").strip()
        user_id = str(payload.get("user_id", "") or "").strip()
        if not relative_path:
            raise ValueError("relative_path is required")
        return self._living_required().delete_file(
            relative_path=relative_path,
            user_id=user_id,
        )

    def living_knowledge_analyze(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text", "") or "").strip()
        if not text:
            raise ValueError("text is required")
        user_id = str(payload.get("user_id", "default_user") or "default_user").strip() or "default_user"
        language = str(payload.get("language", "en") or "en").strip() or "en"
        branch_id = str(payload.get("branch_id", "main") or "main").strip() or "main"
        apply_changes = self._to_bool(payload.get("apply_changes", False))
        display_name = str(payload.get("display_name", user_id) or user_id).strip() or user_id
        self._living_required().bootstrap_user(
            user_id=user_id,
            display_name=display_name,
            primary_language=language,
        )

        raw_sources = payload.get("sources", [])
        normalized_sources: list[dict[str, Any]] = []
        if isinstance(raw_sources, list):
            for row in raw_sources:
                if isinstance(row, Mapping):
                    normalized_sources.append(dict(row))
                elif str(row).strip():
                    normalized_sources.append({"url": str(row).strip()})

        return self._living_required().analyze_knowledge(
            text=text,
            user_id=user_id,
            sources=normalized_sources,
            branch_id=branch_id,
            apply_changes=apply_changes,
        )

    def living_knowledge_initialize(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        item = dict(payload or {})
        user_id = str(item.get("user_id", "foundation_user") or "foundation_user").strip() or "foundation_user"
        language = str(item.get("language", "en") or "en").strip() or "en"
        branch_id = str(item.get("branch_id", "foundation") or "foundation").strip() or "foundation"
        apply_changes = self._to_bool(item.get("apply_changes", True))
        display_name = str(item.get("display_name", "Foundation User") or "Foundation User").strip()
        self._living_required().bootstrap_user(
            user_id=user_id,
            display_name=display_name,
            primary_language=language,
        )
        return self._living_required().initialize_foundational_knowledge(
            user_id=user_id,
            branch_id=branch_id,
            apply_changes=apply_changes,
        )

    def living_knowledge_evaluate(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        user_id = ""
        if payload:
            user_id = str(payload.get("user_id", "") or "").strip()
        return self._living_required().evaluate_knowledge_graph(user_id=user_id)

    def living_knowledge_branch(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        branch_name = str(payload.get("branch_name", "") or "").strip()
        if not branch_name:
            raise ValueError("branch_name is required")
        user_id = str(payload.get("user_id", "default_user") or "default_user").strip() or "default_user"
        return self._living_required().create_knowledge_branch(
            user_id=user_id,
            branch_name=branch_name,
        )

    def living_knowledge_merge(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        user_id = str(payload.get("user_id", "default_user") or "default_user").strip() or "default_user"
        base_snapshot_id = self._to_int(payload.get("base_snapshot_id", 0), 0)
        target_snapshot_id = self._to_int(payload.get("target_snapshot_id", 0), 0)
        apply_changes = self._to_bool(payload.get("apply_changes", False))
        if base_snapshot_id <= 0 or target_snapshot_id <= 0:
            raise ValueError("base_snapshot_id and target_snapshot_id are required")
        return self._living_required().merge_knowledge_branches(
            user_id=user_id,
            base_snapshot_id=base_snapshot_id,
            target_snapshot_id=target_snapshot_id,
            apply_changes=apply_changes,
        )

    def project_overview(self) -> dict[str, Any]:
        graph_payload = self.snapshot_payload()
        living_enabled = self.living_system is not None

        living_architecture: dict[str, Any] = {}
        living_health: dict[str, Any] = {}
        if living_enabled:
            living_architecture = self.living_architecture()
            living_health = self.living_health()

        snapshot_nodes = list(graph_payload.get("snapshot", {}).get("nodes", []) or [])
        foundation_domains = sum(1 for row in snapshot_nodes if str(row.get("type")) == "domain")
        foundation_concepts = sum(1 for row in snapshot_nodes if str(row.get("type")) == "concept")

        return {
            "project": "autonomous-knowledge-platform",
            "graph": graph_payload,
            "living_enabled": living_enabled,
            "living_architecture": living_architecture,
            "living_health": living_health,
            "foundation": {
                "domain_nodes": foundation_domains,
                "concept_nodes": foundation_concepts,
            },
            "features": {
                "graph_workspace": True,
                "living_runtime": living_enabled,
                "prompt_brain": living_enabled,
                "universal_knowledge": living_enabled,
                "client_introspection": True,
                "graph_editing": True,
                "db_schema_json": living_enabled,
                "daily_mode": True,
                "user_dimension_graph": True,
                "autoruns_import": True,
                "model_advisors": True,
                "llm_role_debate": True,
                "hallucination_hunter": True,
                "archive_verified_chat": True,
            },
        }

    def project_model_advisors(self) -> dict[str, Any]:
        try:
            from src.utils.local_llm_provider import list_model_advisors
            advisor_payload = list_model_advisors()
        except Exception as exc:
            advisor_payload = {
                "models_dir": str(os.getenv("LOCAL_MODELS_DIR", "models/gguf") or "models/gguf"),
                "detected_models": [],
                "advisors": [],
                "translator_policy": "translator_gguf_only",
                "translator_priority": "madlad400",
                "error": str(exc),
            }

        prompt_catalog: list[dict[str, Any]] = []
        if self.living_system is not None:
            try:
                prompt_catalog = self._living_required().prompt_catalog()
            except Exception:
                prompt_catalog = []

        return {
            "advisors": advisor_payload,
            "prompts": prompt_catalog,
            "timestamp": time.time(),
        }

    def project_hallucination_report(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        user_id = str(payload.get("user_id", "default_user") or "default_user").strip() or "default_user"
        session_id = str(payload.get("session_id", "") or "").strip()
        sanitized = self._sanitize_hallucination_payload(payload)

        prompt = str(sanitized.get("prompt", "") or "").strip()
        llm_answer = str(sanitized.get("llm_answer", "") or "").strip()
        correct_answer = str(sanitized.get("correct_answer", "") or "").strip()
        if not prompt:
            raise ValueError("prompt is required")
        if not llm_answer:
            raise ValueError("llm_answer is required")
        if not correct_answer:
            raise ValueError("correct_answer is required")

        branch_node = self._ensure_hallucination_branch_node(user_id=user_id)
        case_key = self._hallucination_signature(user_id, prompt, llm_answer)
        prompt_signature = self._hallucination_signature(user_id, prompt)
        wrong_signature = self._hallucination_signature(user_id, llm_answer)
        correct_signature = self._hallucination_signature(user_id, correct_answer)

        case_node, created = self._ensure_shared_node(
            node_type="llm_hallucination_case",
            identity_key="case_key",
            identity_value=case_key,
            attributes={
                "case_key": case_key,
                "user_id": user_id,
                "session_id": session_id,
                "name": self._to_title(prompt, fallback="LLM Hallucination Case", limit=110),
                "prompt": prompt,
                "hallucinated_answer": llm_answer,
                "correct_answer": correct_answer,
                "source": str(sanitized.get("source", "") or ""),
                "severity": str(sanitized.get("severity", "medium") or "medium"),
                "confidence": self._confidence(sanitized.get("confidence", 0.8), 0.8),
                "tags": self._to_list_of_strings(sanitized.get("tags")),
                "metadata": self._as_mapping(sanitized.get("metadata")),
                "prompt_signature": prompt_signature,
                "wrong_signature": wrong_signature,
                "correct_signature": correct_signature,
                "occurrence_count": 0,
            },
        )

        case_node.attributes["prompt"] = prompt
        case_node.attributes["hallucinated_answer"] = llm_answer
        case_node.attributes["correct_answer"] = correct_answer
        case_node.attributes["source"] = str(sanitized.get("source", "") or "")
        case_node.attributes["severity"] = str(sanitized.get("severity", "medium") or "medium")
        case_node.attributes["confidence"] = self._confidence(sanitized.get("confidence", 0.8), 0.8)
        case_node.attributes["tags"] = self._to_list_of_strings(sanitized.get("tags"))
        case_node.attributes["metadata"] = self._as_mapping(sanitized.get("metadata"))
        case_node.attributes["prompt_signature"] = prompt_signature
        case_node.attributes["wrong_signature"] = wrong_signature
        case_node.attributes["correct_signature"] = correct_signature
        case_node.attributes["updated_at"] = float(time.time())
        if created:
            case_node.attributes["created_at"] = float(time.time())

        prev_occurrence = self._to_int(case_node.attributes.get("occurrence_count", 0), 0)
        case_node.attributes["occurrence_count"] = max(1, prev_occurrence + 1)

        self._connect_nodes(
            from_node=int(branch_node.id),
            to_node=int(case_node.id),
            relation_type="tracks_hallucination",
            weight=self._confidence(case_node.attributes.get("confidence", 0.8), 0.8),
            logic_rule="hallucination_hunter_report",
            metadata={
                "severity": str(case_node.attributes.get("severity", "medium") or "medium"),
                "occurrence_count": int(case_node.attributes.get("occurrence_count", 1) or 1),
            },
        )

        self.api.engine._record_event(  # noqa: SLF001
            "hallucination_reported",
            {
                "user_id": user_id,
                "session_id": session_id,
                "case_node_id": int(case_node.id),
                "created": bool(created),
                "severity": str(case_node.attributes.get("severity", "medium") or "medium"),
                "occurrence_count": int(case_node.attributes.get("occurrence_count", 1) or 1),
            },
        )

        similar = self._match_hallucination_cases(
            user_id=user_id,
            prompt=prompt,
            llm_answer=llm_answer,
            top_k=3,
        )
        return {
            "hallucination_branch": {
                "branch_name": _HALLUCINATION_BRANCH_NAME,
                "branch_node_id": int(branch_node.id),
                "user_id": user_id,
            },
            "case": {
                "case_node_id": int(case_node.id),
                "case_key": case_key,
                "created": bool(created),
                "occurrence_count": int(case_node.attributes.get("occurrence_count", 1) or 1),
                "severity": str(case_node.attributes.get("severity", "medium") or "medium"),
                "confidence": self._confidence(case_node.attributes.get("confidence", 0.8), 0.8),
            },
            "guard_matches": similar,
            **self.snapshot_payload(),
        }

    def project_hallucination_check(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        user_id = str(payload.get("user_id", "default_user") or "default_user").strip() or "default_user"
        prompt = " ".join(
            str(payload.get("prompt", "") or payload.get("question", "") or payload.get("text", "") or "").split()
        ).strip()
        llm_answer = " ".join(str(payload.get("llm_answer", "") or payload.get("answer", "") or "").split()).strip()
        if not prompt and not llm_answer:
            raise ValueError("prompt or llm_answer is required")
        top_k = max(1, min(10, self._to_int(payload.get("top_k", 3), 3)))

        query_text = prompt or llm_answer
        matches = self._match_hallucination_cases(
            user_id=user_id,
            prompt=query_text,
            llm_answer=llm_answer,
            top_k=top_k,
        )
        guard = self._hallucination_guard_context(matches)

        self.api.engine._record_event(  # noqa: SLF001
            "hallucination_checked",
            {
                "user_id": user_id,
                "matches": len(matches),
                "query_present": bool(prompt),
                "answer_present": bool(llm_answer),
            },
        )

        return {
            "user_id": user_id,
            "query": {
                "prompt": prompt,
                "llm_answer": llm_answer,
            },
            "matches": matches,
            "guard_context": guard,
            "has_known_hallucination_risk": bool(matches),
            "match_count": len(matches),
            **self.snapshot_payload(),
        }

    def project_archive_verified_chat(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        user_id = str(payload.get("user_id", "default_user") or "default_user").strip() or "default_user"
        session_id = str(payload.get("session_id", "") or "").strip()
        sanitized = self._sanitize_archive_chat_payload(payload)
        message = str(sanitized.get("message", "") or "").strip()
        if not message:
            raise ValueError("message is required")

        context = str(sanitized.get("context", "") or "").strip()
        model_path = str(sanitized.get("model_path", "") or "").strip()
        model_role = str(sanitized.get("model_role", "general") or "general").strip() or "general"
        top_k = max(1, min(8, self._to_int(sanitized.get("top_k", 3), 3)))
        verification_mode = str(sanitized.get("verification_mode", "strict") or "strict").strip() or "strict"
        apply_to_graph = self._to_bool(sanitized.get("apply_to_graph", True))

        llm_fn, selected_model_path, resolution_mode = self._resolve_archive_chat_llm(
            model_path=model_path,
            model_role=model_role,
        )
        if llm_fn is None:
            raise ValueError("archive chat model is unavailable; configure LOCAL_GGUF_MODEL or provide valid model_path")

        update_schema = (
            "{\n"
            '  "summary": "short summary",\n'
            '  "archive_updates": [\n'
            "    {\n"
            '      "entity": "string",\n'
            '      "field": "string",\n'
            '      "operation": "upsert|append|correct|deprecate",\n'
            '      "value": "any-json-value",\n'
            '      "reason": "why this update is needed",\n'
            '      "source": "verified source or evidence",\n'
            '      "confidence": 0.0,\n'
            '      "tags": ["optional"]\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

        prompt = (
            "You are a verification-first archive update assistant.\n"
            "Task: produce only safe JSON updates for knowledge archive maintenance.\n"
            "Rules:\n"
            "- Output JSON only, no markdown.\n"
            "- Do not invent facts; if uncertain, set lower confidence and include why.\n"
            "- Include `source` for each update.\n"
            "- Keep updates concise and practical.\n"
            f"Verification mode: {verification_mode}\n"
            f"User message: {message}\n"
            f"Context: {context}\n"
            "Return JSON with this schema:\n"
            f"{update_schema}"
        )

        raw_output = ""
        try:
            raw_output = str(llm_fn(prompt) or "").strip()
        except Exception:
            raw_output = ""

        parsed_output = self._extract_json_from_llm_output(raw_output)
        updates: list[dict[str, Any]] = []
        summary = ""
        if isinstance(parsed_output, Mapping):
            summary = " ".join(str(parsed_output.get("summary", "") or "").split()).strip()
            updates = self._normalize_archive_updates(
                parsed_output.get("archive_updates", parsed_output.get("updates", []))
            )
        elif isinstance(parsed_output, list):
            updates = self._normalize_archive_updates(parsed_output)

        verification = self._verify_archive_updates(
            user_id=user_id,
            message=message,
            updates=updates,
            verification_mode=verification_mode,
            top_k=top_k,
        )

        selected_path = selected_model_path or model_path
        graph_binding = self._attach_archive_updates_to_graph(
            user_id=user_id,
            session_id=session_id,
            message=message,
            context=context,
            summary=summary,
            updates=updates,
            verification=verification,
            model_role=model_role,
            model_path=selected_path,
            resolution_mode=resolution_mode,
            raw_output=raw_output,
            verification_mode=verification_mode,
            apply_to_graph=apply_to_graph,
        )
        assistant_reply = self._build_archive_chat_reply(
            message=message,
            summary=summary,
            updates=updates,
            verification=verification,
        )

        self.api.engine._record_event(  # noqa: SLF001
            "archive_update_chat_completed",
            {
                "user_id": user_id,
                "session_id": session_id,
                "model_role": model_role,
                "model_path": selected_path,
                "resolution_mode": resolution_mode,
                "verified": bool(verification.get("verified", False)),
                "updates_count": len(updates),
                "hallucination_guard_hits": int(verification.get("hallucination_guard_hits", 0) or 0),
                "verification_mode": verification_mode,
                "attached_to_graph": bool(graph_binding.get("attached", False)),
            },
        )

        return {
            "user_id": user_id,
            "session_id": session_id,
            "model": {
                "requested_model_path": model_path,
                "selected_model_path": selected_path,
                "model_role": model_role,
                "resolution_mode": resolution_mode,
            },
            "input": {
                "message": message,
                "context": context,
            },
            "assistant_reply": assistant_reply,
            "summary": summary,
            "archive_updates": updates,
            "verification": verification,
            "graph_binding": graph_binding,
            "review": {
                "summary": summary,
                "archive_updates": updates,
                "verification": verification,
            },
            "raw_output": raw_output[:6000],
            **self.snapshot_payload(),
        }

    def project_archive_review_apply(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        user_id = str(payload.get("user_id", "default_user") or "default_user").strip() or "default_user"
        session_id = str(payload.get("session_id", "") or "").strip()
        message = " ".join(
            str(payload.get("message", "") or payload.get("prompt", "") or "Manual archive review draft").split()
        ).strip()
        context = " ".join(str(payload.get("context", "") or "").split()).strip()
        verification_mode = self._pick_allowed_token(
            payload.get("verification_mode"),
            allowed=_ARCHIVE_VERIFICATION_MODES_ALLOWED,
            default="strict",
        )
        top_k = max(1, min(8, self._to_int(payload.get("top_k", 3), 3)))
        apply_to_graph = self._to_bool(payload.get("apply_to_graph", True))
        summary = " ".join(str(payload.get("summary", "") or "").split()).strip()

        updates_input = payload.get("archive_updates", payload.get("updates", payload.get("review_json", [])))
        updates = self._coerce_archive_updates_input(updates_input)
        if not updates:
            raise ValueError("archive_updates is required")

        verification = self._verify_archive_updates(
            user_id=user_id,
            message=message,
            updates=updates,
            verification_mode=verification_mode,
            top_k=top_k,
        )
        graph_binding = self._attach_archive_updates_to_graph(
            user_id=user_id,
            session_id=session_id,
            message=message,
            context=context,
            summary=summary,
            updates=updates,
            verification=verification,
            model_role="general",
            model_path="manual_review",
            resolution_mode="manual_review",
            raw_output=self._stable_json({"archive_updates": updates}),
            verification_mode=verification_mode,
            apply_to_graph=apply_to_graph,
        )
        assistant_reply = self._build_archive_chat_reply(
            message=message,
            summary=summary,
            updates=updates,
            verification=verification,
        )

        self.api.engine._record_event(  # noqa: SLF001
            "archive_update_review_applied",
            {
                "user_id": user_id,
                "session_id": session_id,
                "verified": bool(verification.get("verified", False)),
                "updates_count": len(updates),
                "verification_mode": verification_mode,
                "attached_to_graph": bool(graph_binding.get("attached", False)),
            },
        )

        return {
            "user_id": user_id,
            "session_id": session_id,
            "assistant_reply": assistant_reply,
            "summary": summary,
            "archive_updates": updates,
            "verification": verification,
            "graph_binding": graph_binding,
            "review": {
                "summary": summary,
                "archive_updates": updates,
                "verification": verification,
            },
            **self.snapshot_payload(),
        }

    def project_llm_debate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        topic = str(payload.get("topic", "") or payload.get("text", "") or "").strip()
        if not topic:
            raise ValueError("topic is required")

        user_id = str(payload.get("user_id", "default_user") or "default_user").strip() or "default_user"
        session_id = str(payload.get("session_id", "") or "").strip()
        hypothesis_count = max(1, min(6, self._to_int(payload.get("hypothesis_count", 3), 3)))
        attach_to_graph = self._to_bool(payload.get("attach_to_graph", True))
        personalization = self._sanitize_personalization(payload.get("personalization"))
        personalization_roles = self._as_mapping(personalization.get("llm_roles"))
        feedback_items = self._normalize_feedback_items(payload.get("feedback_items"))

        proposer_input = payload.get("proposer_role")
        critic_input = payload.get("critic_role")
        judge_input = payload.get("judge_role")
        if not str(proposer_input or "").strip():
            proposer_input = personalization_roles.get("proposer")
        if not str(critic_input or "").strip():
            critic_input = personalization_roles.get("critic")
        if not str(judge_input or "").strip():
            judge_input = personalization_roles.get("judge")

        proposer_role = self._debate_role(
            proposer_input,
            fallback=_DEBATE_DEFAULT_ROLES["proposer"],
        )
        critic_role = self._debate_role(
            critic_input,
            fallback=_DEBATE_DEFAULT_ROLES["critic"],
        )
        judge_role = self._debate_role(
            judge_input,
            fallback=_DEBATE_DEFAULT_ROLES["judge"],
        )

        personalization_context = self._personalization_prompt_context(personalization)
        hallucination_matches = self._match_hallucination_cases(
            user_id=user_id,
            prompt=topic,
            llm_answer="",
            top_k=3,
        )
        hallucination_guard = self._hallucination_guard_context(hallucination_matches)

        proposer_llm = self._resolve_debate_llm(proposer_role)
        critic_llm = self._resolve_debate_llm(critic_role)
        judge_llm = self._resolve_debate_llm(judge_role)

        self.api.engine._record_event(  # noqa: SLF001
            "debate_started",
            {
                "user_id": user_id,
                "session_id": session_id,
                "topic": topic,
                "hypothesis_count": hypothesis_count,
                "roles": {
                    "proposer": proposer_role,
                    "critic": critic_role,
                    "judge": judge_role,
                },
                "personalization_enabled": bool(personalization),
                "hallucination_guard_hits": len(hallucination_matches),
            },
        )

        proposer_prompt = (
            "You are a hypothesis proposer for graph-based reasoning.\n"
            f"Task: {topic}\n"
            f"{personalization_context}\n"
            f"{hallucination_guard}\n"
            f"Generate {hypothesis_count} distinct hypotheses.\n"
            "Return JSON only:\n"
            "{\n"
            '  "hypotheses": [\n'
            '    {"title":"", "claim":"", "rationale":"", "confidence":0.0}\n'
            "  ]\n"
            "}\n"
            "Constraints: confidence in [0,1], concise and actionable claims."
        )

        hypotheses: list[dict[str, Any]] = []
        proposer_raw = ""
        if proposer_llm is not None:
            try:
                proposer_raw = str(proposer_llm(proposer_prompt) or "").strip()
            except Exception:
                proposer_raw = ""
        parsed_proposer = self._extract_json_from_llm_output(proposer_raw)
        if isinstance(parsed_proposer, Mapping):
            raw_hypotheses = self._as_list(parsed_proposer.get("hypotheses"))
            for idx, item in enumerate(raw_hypotheses):
                row = self._as_mapping(item)
                claim = str(row.get("claim", "") or row.get("hypothesis", "") or "").strip()
                if not claim:
                    continue
                hypotheses.append(
                    {
                        "index": idx + 1,
                        "title": self._to_title(
                            str(row.get("title", "") or claim),
                            fallback=f"Hypothesis {idx + 1}",
                        ),
                        "claim": claim,
                        "rationale": str(row.get("rationale", "") or "").strip(),
                        "confidence": self._confidence(row.get("confidence", 0.6), 0.6),
                    }
                )
        if not hypotheses:
            hypotheses = self._fallback_hypotheses(topic=topic, count=hypothesis_count)

        hypotheses = hypotheses[:hypothesis_count]
        for idx, hypothesis in enumerate(hypotheses):
            self.api.engine._record_event(  # noqa: SLF001
                "debate_hypothesis_generated",
                {
                    "index": idx + 1,
                    "title": str(hypothesis.get("title", "") or ""),
                    "confidence": self._confidence(hypothesis.get("confidence", 0.55), 0.55),
                },
            )

        critiques: list[dict[str, Any]] = []
        for idx, hypothesis in enumerate(hypotheses):
            critic_prompt = (
                "You are a contradiction and risk critic.\n"
                f"Task: {topic}\n"
                f"{personalization_context}\n"
                f"{hallucination_guard}\n"
                f"Hypothesis #{idx + 1}: {hypothesis.get('claim', '')}\n"
                "Return JSON only:\n"
                "{\n"
                '  "issues":[""],\n'
                '  "contradictions":[""],\n'
                '  "risk_score":0.0,\n'
                '  "confidence":0.0,\n'
                '  "recommendation":"accept|revise|reject"\n'
                "}\n"
                "risk_score/confidence in [0,1]."
            )
            critic_raw = ""
            if critic_llm is not None:
                try:
                    critic_raw = str(critic_llm(critic_prompt) or "").strip()
                except Exception:
                    critic_raw = ""
            parsed_critic = self._extract_json_from_llm_output(critic_raw)
            if isinstance(parsed_critic, Mapping):
                parsed_row = self._as_mapping(parsed_critic)
                critique = {
                    "issues": self._to_list_of_strings(parsed_row.get("issues")) or ["No issue provided."],
                    "contradictions": self._to_list_of_strings(parsed_row.get("contradictions")),
                    "risk_score": self._confidence(parsed_row.get("risk_score", 0.5), 0.5),
                    "confidence": self._confidence(parsed_row.get("confidence", 0.6), 0.6),
                    "recommendation": str(parsed_row.get("recommendation", "revise") or "revise").strip(),
                }
            else:
                critique = self._fallback_critique(hypothesis=hypothesis)
            critiques.append(critique)
            self.api.engine._record_event(  # noqa: SLF001
                "debate_hypothesis_criticized",
                {
                    "index": idx + 1,
                    "risk_score": self._confidence(critique.get("risk_score", 0.5), 0.5),
                    "recommendation": str(critique.get("recommendation", "revise") or "revise"),
                },
            )

        judge_prompt = (
            "You are the final judge for a multi-role LLM debate.\n"
            f"Task: {topic}\n"
            f"{personalization_context}\n"
            f"{hallucination_guard}\n"
            f"Hypotheses: {json.dumps(hypotheses, ensure_ascii=False)}\n"
            f"Critiques: {json.dumps(critiques, ensure_ascii=False)}\n"
            "Return JSON only:\n"
            "{\n"
            '  "selected_index":1,\n'
            '  "decision":"",\n'
            '  "consensus":"",\n'
            '  "confidence":0.0,\n'
            '  "ranking":[{"index":1,"score":0.0}]\n'
            "}\n"
            "confidence/score in [0,1]."
        )
        judge_raw = ""
        if judge_llm is not None:
            try:
                judge_raw = str(judge_llm(judge_prompt) or "").strip()
            except Exception:
                judge_raw = ""
        parsed_judge = self._extract_json_from_llm_output(judge_raw)
        if isinstance(parsed_judge, Mapping):
            mapped_judge = self._as_mapping(parsed_judge)
            ranking_rows: list[dict[str, Any]] = []
            for item in self._as_list(mapped_judge.get("ranking")):
                row = self._as_mapping(item)
                rank_idx = self._to_int(row.get("index"), 0)
                if rank_idx <= 0:
                    continue
                ranking_rows.append(
                    {
                        "index": rank_idx,
                        "score": self._confidence(row.get("score", 0.5), 0.5),
                    }
                )
            verdict = {
                "selected_index": max(1, min(len(hypotheses), self._to_int(mapped_judge.get("selected_index", 1), 1))),
                "decision": str(mapped_judge.get("decision", "") or "").strip(),
                "consensus": str(mapped_judge.get("consensus", "") or "").strip(),
                "confidence": self._confidence(mapped_judge.get("confidence", 0.6), 0.6),
                "ranking": ranking_rows,
            }
            if not verdict["decision"] and hypotheses:
                verdict["decision"] = str(hypotheses[verdict["selected_index"] - 1].get("claim", "") or "")
        else:
            verdict = self._fallback_verdict(hypotheses=hypotheses, critiques=critiques)

        feedback_summary = {
            "items": len(feedback_items),
            "accepted": 0,
            "rejected": 0,
        }
        if feedback_items:
            accepted = 0
            rejected = 0
            for row in feedback_items:
                decision = str(row.get("decision", "") or "").strip()
                score = self._confidence(row.get("score", 0.0), 0.0)
                if decision in {"accept", "accepted", "like", "liked"} or score >= 0.66:
                    accepted += 1
                if decision in {"reject", "rejected", "dislike", "discard"} or score <= 0.34:
                    rejected += 1
            feedback_summary = {
                "items": len(feedback_items),
                "accepted": accepted,
                "rejected": rejected,
            }

        graph_binding: dict[str, Any] = {
            "attached": False,
            "debate_node_id": 0,
            "hypothesis_node_ids": [],
            "critique_node_ids": [],
            "judge_node_id": 0,
        }
        if attach_to_graph:
            debate_node = self.api.engine.create_node(
                "llm_debate_session",
                attributes={
                    "name": self._to_title(topic, fallback="LLM Debate Session", limit=90),
                    "topic": topic,
                    "user_id": user_id,
                    "session_id": session_id,
                    "roles": {
                        "proposer": proposer_role,
                        "critic": critic_role,
                        "judge": judge_role,
                    },
                    "personalization": personalization,
                    "feedback_summary": feedback_summary,
                    "hallucination_guard_hits": len(hallucination_matches),
                },
                state={"confidence": self._confidence(verdict.get("confidence", 0.6), 0.6)},
            )
            graph_binding["attached"] = True
            graph_binding["debate_node_id"] = int(debate_node.id)

            hypothesis_node_ids: list[int] = []
            critique_node_ids: list[int] = []
            for idx, hypothesis in enumerate(hypotheses):
                branch_token = f"debate:{debate_node.id}:h{idx + 1}"
                hypothesis_node = self.api.engine.create_node(
                    "llm_hypothesis",
                    attributes={
                        "branch_id": branch_token,
                        "title": str(hypothesis.get("title", "") or f"Hypothesis {idx + 1}"),
                        "claim": str(hypothesis.get("claim", "") or ""),
                        "rationale": str(hypothesis.get("rationale", "") or ""),
                        "proposer_role": proposer_role,
                    },
                    state={"confidence": self._confidence(hypothesis.get("confidence", 0.5), 0.5)},
                )
                hypothesis_node_ids.append(int(hypothesis_node.id))
                self._connect_nodes(
                    from_node=debate_node.id,
                    to_node=hypothesis_node.id,
                    relation_type="debate_branch",
                    weight=self._confidence(hypothesis.get("confidence", 0.5), 0.5),
                    logic_rule="llm_debate_proposer",
                    metadata={
                        "index": idx + 1,
                        "branch_id": branch_token,
                    },
                )

                critique = critiques[idx] if idx < len(critiques) else {}
                critique_node = self.api.engine.create_node(
                    "llm_critique",
                    attributes={
                        "branch_id": branch_token,
                        "issues": self._to_list_of_strings(critique.get("issues")),
                        "contradictions": self._to_list_of_strings(critique.get("contradictions")),
                        "recommendation": str(critique.get("recommendation", "revise") or "revise"),
                        "critic_role": critic_role,
                    },
                    state={
                        "risk": self._confidence(critique.get("risk_score", 0.5), 0.5),
                        "confidence": self._confidence(critique.get("confidence", 0.6), 0.6),
                    },
                )
                critique_node_ids.append(int(critique_node.id))
                self._connect_nodes(
                    from_node=hypothesis_node.id,
                    to_node=critique_node.id,
                    relation_type="criticized_by",
                    weight=max(0.0, 1.0 - self._confidence(critique.get("risk_score", 0.5), 0.5)),
                    logic_rule="llm_debate_critic",
                    metadata={"index": idx + 1},
                )

            judge_node = self.api.engine.create_node(
                "llm_judgement",
                attributes={
                    "decision": str(verdict.get("decision", "") or ""),
                    "consensus": str(verdict.get("consensus", "") or ""),
                    "judge_role": judge_role,
                    "ranking": self._as_list(verdict.get("ranking")),
                },
                state={"confidence": self._confidence(verdict.get("confidence", 0.6), 0.6)},
            )
            graph_binding["judge_node_id"] = int(judge_node.id)
            graph_binding["hypothesis_node_ids"] = hypothesis_node_ids
            graph_binding["critique_node_ids"] = critique_node_ids
            self._connect_nodes(
                from_node=debate_node.id,
                to_node=judge_node.id,
                relation_type="judged_by",
                weight=self._confidence(verdict.get("confidence", 0.6), 0.6),
                logic_rule="llm_debate_judge",
                metadata={"selected_index": int(verdict.get("selected_index", 1) or 1)},
            )
            selected_index = max(1, min(len(hypothesis_node_ids), int(verdict.get("selected_index", 1) or 1)))
            if hypothesis_node_ids:
                selected_hypothesis_node_id = hypothesis_node_ids[selected_index - 1]
                self._connect_nodes(
                    from_node=judge_node.id,
                    to_node=selected_hypothesis_node_id,
                    relation_type="selects",
                    weight=self._confidence(verdict.get("confidence", 0.6), 0.6),
                    logic_rule="llm_debate_selection",
                    metadata={"selected_index": selected_index},
                )

        self.api.engine._record_event(  # noqa: SLF001
            "debate_completed",
            {
                "hypothesis_count": len(hypotheses),
                "selected_index": int(verdict.get("selected_index", 1) or 1),
                "confidence": self._confidence(verdict.get("confidence", 0.6), 0.6),
                "attached_to_graph": bool(graph_binding.get("attached", False)),
                "personalization_enabled": bool(personalization),
                "hallucination_guard_hits": len(hallucination_matches),
            },
        )

        return {
            "topic": topic,
            "user_id": user_id,
            "session_id": session_id,
            "personalization": personalization,
            "feedback_summary": feedback_summary,
            "hallucination_guard": {
                "hits": len(hallucination_matches),
                "matches": hallucination_matches,
                "context": hallucination_guard,
            },
            "roles": {
                "proposer": proposer_role,
                "critic": critic_role,
                "judge": judge_role,
            },
            "role_models_available": {
                "proposer": proposer_llm is not None,
                "critic": critic_llm is not None,
                "judge": judge_llm is not None,
            },
            "hypotheses": hypotheses,
            "critiques": critiques,
            "verdict": verdict,
            "graph_binding": graph_binding,
            **self.snapshot_payload(),
        }

    def project_pipeline(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text", "") or "").strip()
        if not text:
            raise ValueError("text is required")

        user_id = str(payload.get("user_id", "default_user") or "default_user").strip() or "default_user"
        display_name = str(payload.get("display_name", user_id) or user_id).strip() or user_id
        language = str(payload.get("language", "en") or "en").strip() or "en"
        session_id = str(payload.get("session_id", "default") or "default").strip() or "default"
        auto_snapshot = self._to_bool(payload.get("auto_snapshot", True))
        branch_id = str(payload.get("branch_id", "main") or "main").strip() or "main"
        apply_knowledge_changes = self._to_bool(payload.get("apply_knowledge_changes", False))

        raw_sources = payload.get("sources", [])
        normalized_sources: list[dict[str, Any]] = []
        if isinstance(raw_sources, list):
            for row in raw_sources:
                if isinstance(row, Mapping):
                    normalized_sources.append(dict(row))
                elif str(row).strip():
                    normalized_sources.append({"url": str(row).strip()})

        living_output = self.living_process(
            {
                "text": text,
                "user_id": user_id,
                "display_name": display_name,
                "language": language,
                "session_id": session_id,
                "auto_snapshot": auto_snapshot,
            }
        )
        knowledge_output = self.living_knowledge_analyze(
            {
                "text": text,
                "user_id": user_id,
                "display_name": display_name,
                "language": language,
                "branch_id": branch_id,
                "apply_changes": apply_knowledge_changes,
                "sources": normalized_sources,
            }
        )

        return {
            "living": living_output,
            "knowledge": knowledge_output,
            "project_status": self.project_evaluate({"user_id": user_id}),
        }

    def project_bootstrap(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        data = dict(payload or {})
        user_id = str(data.get("user_id", "foundation_user") or "foundation_user").strip() or "foundation_user"
        display_name = str(data.get("display_name", "Foundation User") or "Foundation User").strip() or "Foundation User"
        language = str(data.get("language", "en") or "en").strip() or "en"
        branch_id = str(data.get("branch_id", "foundation") or "foundation").strip() or "foundation"
        apply_changes = self._to_bool(data.get("apply_changes", True))
        seed_graph_demo = self._to_bool(data.get("seed_graph_demo", True))

        foundation = self.living_knowledge_initialize(
            {
                "user_id": user_id,
                "display_name": display_name,
                "language": language,
                "branch_id": branch_id,
                "apply_changes": apply_changes,
            }
        )
        graph_seed: dict[str, Any] = {}
        if seed_graph_demo:
            graph_seed = self.seed_demo()

        return {
            "foundation": foundation,
            "graph_seed": graph_seed,
            "project_status": self.project_evaluate({"user_id": user_id}),
        }

    def project_evaluate(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        data = dict(payload or {})
        user_id = str(data.get("user_id", "") or "").strip()
        living_health: dict[str, Any]
        knowledge_eval: dict[str, Any]
        if self.living_system is not None:
            living_health = self.living_health()
            knowledge_eval = self.living_knowledge_evaluate({"user_id": user_id})
        else:
            living_health = {"enabled": False}
            knowledge_eval = {"enabled": False}
        return {
            "graph": self.snapshot_payload(),
            "events": self.list_events(limit=100),
            "living_health": living_health,
            "knowledge_evaluation": knowledge_eval,
        }


__all__ = ["GraphWorkspaceService"]
