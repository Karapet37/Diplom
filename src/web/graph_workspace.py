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

_DEFAULT_STRUCTURED_INPUT_MODEL_PATH = "models/gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
_DEFAULT_GRAPH_MONITOR_MODEL_PATH = "models/gguf/textGen/mistral-7b-instruct-v0.3-q4_k_m.gguf"

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

_PERSONAL_TREE_BRANCH_NAME = "personal_info_tree"
_PERSONAL_TREE_SOURCE_TYPES_ALLOWED: tuple[str, ...] = (
    "article",
    "law",
    "text",
    "note",
    "other",
)
_PERSONAL_TREE_NODE_TYPES: tuple[str, ...] = (
    "personal_info_tree_root",
    "thought_tree_session",
    "thought_summary_node",
    "thought_point_node",
    "personal_note_node",
    "source_reference",
)
_PERSONAL_TREE_EDGE_TYPES: tuple[str, ...] = (
    "tree_session",
    "has_summary",
    "supports_point",
    "based_on_source",
    "tree_note",
    "note_child_of",
    "about_topic",
    "references_source",
)
_PERSONAL_TREE_CITATION_RE = re.compile(
    r"(?:article|art\.?|статья|ст\.?|հոդված)\s*\d+[a-zа-я0-9.\-]*",
    flags=re.IGNORECASE | re.UNICODE,
)
_MEMORY_NAMESPACES_ALLOWED: tuple[str, ...] = (
    "global",
    "personal",
    "session",
    "experiment",
    "trash",
)
_LLM_POLICY_MODES_ALLOWED: tuple[str, ...] = (
    "propose_only",
    "confirm_required",
    "assisted_autonomy",
)
_CONTRADICTION_NEGATION_HINTS: tuple[str, ...] = (
    " not ",
    " no ",
    " never ",
    " without ",
    " cannot ",
    " can't ",
    " doesn't ",
    " isn't ",
    " нет ",
    " не ",
    " без ",
    " never",
    " ոչ ",
    " չի ",
)
_GARBAGE_MANAGER_DEFAULT_ROLE = "coder_reviewer"
_PACKAGE_ACTIONS_ALLOWED: tuple[str, ...] = (
    "store",
    "list",
    "purge",
    "restore",
)
_MEMORY_SCOPE_ALLOWED: tuple[str, ...] = (
    "all",
    "owned",
)
_TASK_RISK_LEVELS: tuple[str, ...] = (
    "low",
    "medium",
    "high",
    "critical",
)
_WRAPPER_PROFILE_NODE_TYPE = "llm_wrapper_profile"
_WRAPPER_CONTEXT_NODE_TYPES_DENY: tuple[str, ...] = (
    "llm_wrapper_feedback",
    "llm_wrapper_session",
)
_WRAPPER_GOSSIP_MODE_ALLOWED: tuple[str, ...] = (
    "auto",
    "off",
    "force",
)
_INTERACTION_TRIAGE_CATEGORIES_ALLOWED: tuple[str, ...] = (
    "action",
    "fact",
    "risk",
    "style",
    "noise",
)
_INTEGRATION_LAYER_ACTIONS_ALLOWED: tuple[str, ...] = (
    "wrapper.respond",
    "archive.chat",
    "user_graph.update",
    "personal_tree.ingest",
)
_INTEGRATION_LAYER_HOSTS_ALLOWED: tuple[str, ...] = (
    "generic",
    "vscode",
    "chat_agent",
    "image_creator",
    "web",
)
_GOSSIP_MARKERS: tuple[str, ...] = (
    "gossip",
    "rumor",
    "rumour",
    "hearsay",
    "talking behind",
    "перемыва",
    "сплет",
    "слух",
    "говорят",
    "բամբաս",
)
_SUBJECT_NAME_HINT_RE: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:about|on|regarding)\s+([A-Za-z][A-Za-z0-9 ._'-]{1,64})",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:о|про|насчет|по поводу)\s+([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9 ._'-]{1,64})",
        flags=re.IGNORECASE | re.UNICODE,
    ),
    re.compile(
        r"(?:մասին)\s+([^\s,.;:!?]{2,64}(?:\s+[^\s,.;:!?]{2,64})?)",
        flags=re.IGNORECASE | re.UNICODE,
    ),
)
_DIALECT_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁёԱ-Ֆա-ֆ0-9][A-Za-zА-Яа-яЁёԱ-Ֆա-ֆ0-9_'-]{1,31}", flags=re.UNICODE)
_DIALECT_STOPWORDS: tuple[str, ...] = (
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "you",
    "your",
    "have",
    "has",
    "are",
    "was",
    "were",
    "что",
    "это",
    "как",
    "для",
    "или",
    "она",
    "они",
    "его",
    "ее",
    "мне",
    "меня",
    "когда",
    "если",
    "где",
    "и",
    "в",
    "на",
    "не",
    "то",
    "это",
    "кто",
    "как",
    "для",
    "он",
    "она",
    "они",
    "ես",
    "դու",
    "նա",
    "մենք",
    "դուք",
    "նրանք",
    "ու",
    "է",
    "եմ",
    "թե",
    "որ",
    "ինչ",
    "ինչպես",
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
        self._llm_policy: dict[str, Any] = {
            "mode": str(os.getenv("AUTOGRAPH_LLM_POLICY_MODE", "confirm_required") or "confirm_required").strip()
            or "confirm_required",
            "trusted_sessions": [],
            "trusted_users": [],
            "allow_apply_for_actions": [],
            "updated_at": time.time(),
        }
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

    def _input_extraction_status(self, value: Any) -> dict[str, Any]:
        root = self._as_mapping(value)
        verification = self._as_mapping(root.get("verification"))
        graph_binding = self._as_mapping(root.get("graph_binding"))
        subject_binding = self._as_mapping(graph_binding.get("subject_binding"))
        model = self._as_mapping(root.get("model"))
        return {
            "enabled": bool(root.get("enabled", False)),
            "source": str(root.get("source", "") or "").strip(),
            "summary": " ".join(str(root.get("summary", "") or "").split()).strip(),
            "updates_count": len(self._as_list(root.get("updates"))),
            "verified": bool(verification.get("verified", False)),
            "issue_count": self._to_int(verification.get("issue_count", 0), 0),
            "warning_count": self._to_int(verification.get("warning_count", 0), 0),
            "score": round(self._to_float(verification.get("score", 0.0), 0.0), 4),
            "graph_attached": bool(graph_binding.get("attached", False)),
            "branch_node_id": self._to_int(graph_binding.get("branch_node_id", 0), 0),
            "session_node_id": self._to_int(graph_binding.get("session_node_id", 0), 0),
            "subject_branch_count": len(self._as_list(subject_binding.get("subject_branch_node_ids"))),
            "used_fallback": bool(model.get("used_fallback", False)),
            "model_path": str(
                model.get("selected_model_path", model.get("requested_model_path", "")) or ""
            ).strip(),
        }

    def _graph_monitor_status(self, value: Any) -> dict[str, Any]:
        root = self._as_mapping(value)
        model = self._as_mapping(root.get("model"))
        node_patch_count = self._to_int(root.get("node_patch_count", 0), 0)
        edge_patch_count = self._to_int(root.get("edge_patch_count", 0), 0)
        return {
            "enabled": bool(root.get("enabled", False)),
            "attached": bool(root.get("attached", False)),
            "session_node_id": self._to_int(root.get("session_node_id", 0), 0),
            "node_patch_count": node_patch_count,
            "edge_patch_count": edge_patch_count,
            "patch_total": max(0, node_patch_count + edge_patch_count),
            "used_fallback": bool(model.get("used_fallback", False)),
            "model_path": str(
                model.get("selected_model_path", model.get("requested_model_path", "")) or ""
            ).strip(),
        }

    def _execution_status(
        self,
        *,
        action: str,
        persisted: Any,
        input_extraction: Any = None,
        graph_monitor: Any = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        extraction = self._input_extraction_status(input_extraction)
        monitor = self._graph_monitor_status(graph_monitor)
        persisted_flag = bool(persisted)
        status = "persisted" if persisted_flag else "in_memory"
        if not persisted_flag and (extraction.get("graph_attached") or monitor.get("attached")):
            status = "graph_updated_not_persisted"
        payload = {
            "action": str(action or "").strip(),
            "status": status,
            "persisted": persisted_flag,
            "input_extraction": extraction,
            "graph_monitor": monitor,
        }
        if extra:
            payload.update(dict(extra))
        return payload

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

    def _normalize_namespace(self, value: Any, *, default: str = "global") -> str:
        token = self._pick_allowed_token(
            value,
            allowed=_MEMORY_NAMESPACES_ALLOWED,
            default=default,
        )
        return token or default

    def _node_namespace(self, node: Any, *, default: str = "global") -> str:
        attrs = self._as_mapping(getattr(node, "attributes", {}))
        return self._normalize_namespace(attrs.get("namespace", default), default=default)

    @staticmethod
    def _node_text_blob(node: Any) -> str:
        attrs = GraphWorkspaceService._as_mapping(getattr(node, "attributes", {}))
        chunks: list[str] = []
        for key in (
            "name",
            "title",
            "description",
            "summary",
            "point",
            "claim",
            "note",
            "details",
            "topic",
            "source_title",
            "source_url",
        ):
            value = attrs.get(key)
            if isinstance(value, str) and value.strip():
                chunks.append(value.strip())
        return " ".join(chunks).strip()

    @staticmethod
    def _node_belongs_to_user(node: Any, user_id: str) -> bool:
        attrs = GraphWorkspaceService._as_mapping(getattr(node, "attributes", {}))
        owner = str(attrs.get("user_id", "") or "").strip()
        if not owner:
            return False
        return owner == str(user_id or "").strip()

    def _llm_policy_snapshot(self) -> dict[str, Any]:
        mode = self._pick_allowed_token(
            self._llm_policy.get("mode"),
            allowed=_LLM_POLICY_MODES_ALLOWED,
            default="confirm_required",
        )
        return {
            "mode": mode,
            "trusted_sessions": self._dedupe_strings(self._to_list_of_strings(self._llm_policy.get("trusted_sessions"))),
            "trusted_users": self._dedupe_strings(self._to_list_of_strings(self._llm_policy.get("trusted_users"))),
            "allow_apply_for_actions": self._dedupe_strings(
                self._to_list_of_strings(self._llm_policy.get("allow_apply_for_actions"))
            ),
            "updated_at": float(self._llm_policy.get("updated_at", time.time()) or time.time()),
        }

    def _llm_policy_decision(
        self,
        *,
        user_id: str,
        session_id: str,
        action: str,
        requested_apply: bool,
        confirmation: Any = None,
    ) -> dict[str, Any]:
        policy = self._llm_policy_snapshot()
        mode = str(policy.get("mode", "confirm_required") or "confirm_required")
        token = str(confirmation or "").strip().lower()
        is_confirmed = token in {"confirm", "confirmed", "yes", "true", "1", "apply"}
        trusted_session = str(session_id or "").strip() in set(policy.get("trusted_sessions", []))
        trusted_user = str(user_id or "").strip() in set(policy.get("trusted_users", []))
        action_allowed = str(action or "").strip() in set(policy.get("allow_apply_for_actions", []))

        apply_allowed = bool(requested_apply)
        reason = "requested_false"
        requires_confirmation = False
        if not requested_apply:
            apply_allowed = False
            reason = "requested_false"
        elif mode == "propose_only":
            apply_allowed = False
            reason = "policy_propose_only"
        elif mode == "confirm_required":
            if is_confirmed:
                apply_allowed = True
                reason = "confirmed"
            else:
                apply_allowed = False
                requires_confirmation = True
                reason = "confirmation_required"
        else:  # assisted_autonomy
            if trusted_session or trusted_user or action_allowed or is_confirmed:
                apply_allowed = True
                reason = "assisted_autonomy_allowed"
            else:
                apply_allowed = False
                requires_confirmation = True
                reason = "assisted_autonomy_needs_trust_or_confirm"

        return {
            "mode": mode,
            "requested_apply": bool(requested_apply),
            "apply_allowed": bool(apply_allowed),
            "requires_confirmation": bool(requires_confirmation),
            "confirmation_received": bool(is_confirmed),
            "trusted_session": bool(trusted_session),
            "trusted_user": bool(trusted_user),
            "action_allowlisted": bool(action_allowed),
            "reason": reason,
            "action": str(action or ""),
        }

    def _resolve_role_or_model_llm(
        self,
        *,
        role: str = "general",
        model_path: str = "",
    ) -> tuple[Callable[[str], str] | None, dict[str, Any]]:
        requested_model_path = str(model_path or "").strip()
        requested_role = str(role or "general").strip() or "general"
        if requested_model_path and self.model_llm_resolver is not None:
            fn = self.model_llm_resolver(requested_model_path)
            if fn is not None:
                return fn, {
                    "mode": "explicit_model_path",
                    "requested_model_path": requested_model_path,
                    "selected_model_path": requested_model_path,
                    "requested_role": requested_role,
                }
        if self.role_llm_resolver is not None:
            fn = self.role_llm_resolver(requested_role)
            if fn is not None:
                return fn, {
                    "mode": "role_resolver",
                    "requested_model_path": requested_model_path,
                    "selected_model_path": "",
                    "requested_role": requested_role,
                }
        return None, {
            "mode": "unavailable",
            "requested_model_path": requested_model_path,
            "selected_model_path": "",
            "requested_role": requested_role,
        }

    @staticmethod
    def _safe_slug(value: Any, *, default: str) -> str:
        token = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
        return token or default

    def _heuristic_input_updates(
        self,
        *,
        source: str,
        text: str,
        context: str,
        limit: int = 4,
    ) -> tuple[str, list[dict[str, Any]]]:
        clean_text = " ".join(str(text or "").split()).strip()
        clean_context = " ".join(str(context or "").split()).strip()
        source_token = self._safe_slug(source, default="input")
        summary = self._to_title(clean_text, fallback=f"{source_token} input captured", limit=160)
        sentences = self._split_sentences(clean_text, limit=max(1, int(limit)))
        if not sentences and clean_text:
            sentences = [clean_text[:320]]

        updates: list[dict[str, Any]] = []
        for idx, sentence in enumerate(sentences[: max(1, int(limit))], start=1):
            field = "intent" if idx == 1 else f"detail_{idx}"
            updates.append(
                {
                    "entity": source_token,
                    "field": field,
                    "operation": "append",
                    "value": sentence[:320],
                    "reason": "Structured fallback capture from user input text.",
                    "source": "input_capture_fallback",
                    "confidence": 0.58,
                    "tags": [source_token, "input_capture", "fallback"],
                }
            )
        if clean_context:
            updates.append(
                {
                    "entity": source_token,
                    "field": "context",
                    "operation": "append",
                    "value": clean_context[:320],
                    "reason": "Context line linked to the captured input.",
                    "source": "input_capture_context",
                    "confidence": 0.61,
                    "tags": [source_token, "context"],
                }
            )
        return summary, updates[: max(1, int(limit))]

    def _link_input_capture_to_related_nodes(
        self,
        *,
        graph_binding: Mapping[str, Any],
        related_node_ids: list[int],
        source: str,
    ) -> None:
        session_node_id = self._to_int(self._as_mapping(graph_binding).get("session_node_id", 0), 0)
        if session_node_id <= 0:
            return
        update_node_ids = [
            self._to_int(item, 0)
            for item in self._as_list(self._as_mapping(graph_binding).get("update_node_ids"))
            if self._to_int(item, 0) > 0
        ]
        relation_type = f"{self._safe_slug(source, default='input')}_input"
        for related_node_id in related_node_ids:
            if related_node_id <= 0:
                continue
            self._connect_nodes(
                from_node=session_node_id,
                to_node=int(related_node_id),
                relation_type=f"interprets_{relation_type}",
                weight=0.74,
                logic_rule="input_capture_link",
            )
            for update_node_id in update_node_ids[:6]:
                self._connect_nodes(
                    from_node=update_node_id,
                    to_node=int(related_node_id),
                    relation_type="updates_related_context",
                    weight=0.68,
                    logic_rule="input_capture_link",
                )

    def _run_graph_monitor(
        self,
        *,
        user_id: str,
        session_id: str,
        source: str,
        focus_node_ids: list[int],
        hint_text: str = "",
        model_path: str = "",
        model_role: str = "planner",
        apply_changes: bool = True,
    ) -> dict[str, Any]:
        normalized_focus = [
            self._to_int(item, 0)
            for item in list(focus_node_ids or [])
            if self._to_int(item, 0) > 0
        ]
        seen_focus: set[int] = set()
        focus_ids: list[int] = []
        for node_id in normalized_focus:
            if node_id in seen_focus:
                continue
            seen_focus.add(node_id)
            focus_ids.append(node_id)
            if len(focus_ids) >= 12:
                break
        if not focus_ids:
            return {
                "enabled": False,
                "attached": False,
                "session_node_id": 0,
                "patch_node_ids": [],
                "node_patches": [],
                "edge_patches": [],
                "node_patch_count": 0,
                "edge_patch_count": 0,
                "raw_output": "",
                "model": {
                    "requested_model_path": str(model_path or ""),
                    "selected_model_path": "",
                    "requested_role": str(model_role or "planner"),
                    "resolution": {"mode": "skipped"},
                    "used_fallback": True,
                },
            }

        focus_nodes = [self.api.engine.get_node(node_id) for node_id in focus_ids]
        focus_nodes = [node for node in focus_nodes if node is not None]
        if not focus_nodes:
            return {
                "enabled": False,
                "attached": False,
                "session_node_id": 0,
                "patch_node_ids": [],
                "node_patches": [],
                "edge_patches": [],
                "node_patch_count": 0,
                "edge_patch_count": 0,
                "raw_output": "",
                "model": {
                    "requested_model_path": str(model_path or ""),
                    "selected_model_path": "",
                    "requested_role": str(model_role or "planner"),
                    "resolution": {"mode": "skipped"},
                    "used_fallback": True,
                },
            }

        focus_payload = []
        allowed_focus_ids = {int(node.id) for node in focus_nodes}
        for node in focus_nodes:
            attrs = self._as_mapping(getattr(node, "attributes", {}))
            focus_payload.append(
                {
                    "node_id": int(node.id),
                    "type": str(getattr(node, "type", "generic") or "generic"),
                    "name": str(attrs.get("name", "") or attrs.get("title", "") or f"Node {node.id}"),
                    "summary": self._node_text_blob(node)[:280],
                    "state": self._as_mapping(getattr(node, "state", {})),
                }
            )

        focus_edges = []
        for edge in self.api.engine.store.edges:
            from_node = int(getattr(edge, "from_node", 0) or 0)
            to_node = int(getattr(edge, "to_node", 0) or 0)
            if from_node not in allowed_focus_ids or to_node not in allowed_focus_ids:
                continue
            focus_edges.append(
                {
                    "from_node": from_node,
                    "to_node": to_node,
                    "relation_type": str(getattr(edge, "relation_type", "related_to") or "related_to"),
                    "weight": round(self._to_float(getattr(edge, "weight", 0.0), 0.0), 4),
                }
            )
            if len(focus_edges) >= 16:
                break

        requested_model_path = str(model_path or "").strip() or _DEFAULT_GRAPH_MONITOR_MODEL_PATH
        resolved_role = self._debate_role(model_role, fallback="planner")
        llm_fn, resolution = self._resolve_role_or_model_llm(
            role=resolved_role,
            model_path=requested_model_path,
        )
        raw_output = ""
        parsed_output: dict[str, Any] = {}
        used_fallback = True
        if llm_fn is not None:
            prompt = (
                "You are a graph monitor.\n"
                "Given a focused graph fragment, suggest small safe node/edge patches.\n"
                "Return JSON only:\n"
                '{ "node_patches":[{"node_id":0,"summary":"","confidence":0.0,"reason":""}],'
                '"edge_patches":[{"from_node":0,"to_node":0,"relation_type":"related_to","weight":0.0,"action":"create|update","reason":""}] }\n'
                "Keep patches minimal, only reference provided node ids, and never invent node ids.\n"
                f"Source: {source}\n"
                f"Hint: {' '.join(str(hint_text or '').split())[:600]}\n"
                f"Nodes: {self._stable_json(focus_payload)}\n"
                f"Edges: {self._stable_json(focus_edges)}\n"
            )
            try:
                raw_output = str(llm_fn(prompt) or "").strip()
                parsed_candidate = self._extract_json_from_llm_output(raw_output)
                if isinstance(parsed_candidate, Mapping):
                    parsed_output = dict(parsed_candidate)
                    used_fallback = False
            except Exception:
                parsed_output = {}
                used_fallback = True

        node_patches_in = self._as_list(parsed_output.get("node_patches", []))
        edge_patches_in = self._as_list(parsed_output.get("edge_patches", []))
        if not node_patches_in:
            node_patches_in = [
                {
                    "node_id": int(node["node_id"]),
                    "summary": str(node.get("summary", "") or node.get("name", "") or "")[:240],
                    "confidence": 0.62,
                    "reason": "heuristic_monitor_summary",
                }
                for node in focus_payload[:4]
                if str(node.get("summary", "") or node.get("name", "") or "").strip()
            ]
        if not edge_patches_in:
            heuristic_edges: list[dict[str, Any]] = []
            for idx in range(len(focus_nodes)):
                left = focus_nodes[idx]
                left_tokens = self._token_set(self._node_text_blob(left))
                if not left_tokens:
                    continue
                for jdx in range(idx + 1, len(focus_nodes)):
                    right = focus_nodes[jdx]
                    right_tokens = self._token_set(self._node_text_blob(right))
                    if not right_tokens:
                        continue
                    overlap = self._jaccard_similarity(left_tokens, right_tokens)
                    if overlap < 0.18:
                        continue
                    heuristic_edges.append(
                        {
                            "from_node": int(left.id),
                            "to_node": int(right.id),
                            "relation_type": "related_to",
                            "weight": round(max(0.42, min(0.84, 0.42 + overlap)), 4),
                            "action": "update",
                            "reason": "heuristic_text_overlap",
                        }
                    )
                    if len(heuristic_edges) >= 4:
                        break
                if len(heuristic_edges) >= 4:
                    break
            edge_patches_in = heuristic_edges

        node_patches: list[dict[str, Any]] = []
        for row in node_patches_in[:6]:
            item = self._as_mapping(row)
            node_id = self._to_int(item.get("node_id", 0), 0)
            if node_id not in allowed_focus_ids:
                continue
            summary = " ".join(str(item.get("summary", "") or "").split()).strip()
            if not summary:
                continue
            node_patches.append(
                {
                    "node_id": node_id,
                    "summary": summary[:240],
                    "confidence": self._confidence(item.get("confidence", 0.62), 0.62),
                    "reason": " ".join(str(item.get("reason", "") or "").split()).strip()[:320],
                }
            )

        edge_patches: list[dict[str, Any]] = []
        for row in edge_patches_in[:6]:
            item = self._as_mapping(row)
            from_node = self._to_int(item.get("from_node", 0), 0)
            to_node = self._to_int(item.get("to_node", 0), 0)
            if from_node not in allowed_focus_ids or to_node not in allowed_focus_ids:
                continue
            if from_node == to_node:
                continue
            relation_type = self._safe_slug(item.get("relation_type", "related_to"), default="related_to")
            action = "update" if str(item.get("action", "update") or "update").strip().lower() == "update" else "create"
            edge_patches.append(
                {
                    "from_node": from_node,
                    "to_node": to_node,
                    "relation_type": relation_type,
                    "weight": round(self._confidence(item.get("weight", 0.58), 0.58), 4),
                    "action": action,
                    "reason": " ".join(str(item.get("reason", "") or "").split()).strip()[:320],
                }
            )

        if not apply_changes:
            return {
                "enabled": True,
                "attached": False,
                "session_node_id": 0,
                "patch_node_ids": [],
                "node_patches": node_patches,
                "edge_patches": edge_patches,
                "node_patch_count": len(node_patches),
                "edge_patch_count": len(edge_patches),
                "raw_output": raw_output[:6000],
                "model": {
                    "requested_model_path": requested_model_path,
                    "selected_model_path": str(resolution.get("selected_model_path", "") or requested_model_path),
                    "requested_role": resolved_role,
                    "resolution": dict(resolution),
                    "used_fallback": bool(used_fallback),
                },
            }

        session_node = self.api.engine.create_node(
            "graph_monitor_session",
            attributes={
                "user_id": user_id,
                "session_id": session_id,
                "source": source,
                "name": f"Graph Monitor {self._to_title(source, fallback='session', limit=48)}",
                "hint_text": " ".join(str(hint_text or "").split())[:900],
                "focus_node_ids": list(focus_ids),
                "requested_model_path": requested_model_path,
                "selected_model_path": str(resolution.get("selected_model_path", "") or requested_model_path),
                "requested_role": resolved_role,
                "resolution": dict(resolution),
                "used_fallback": bool(used_fallback),
                "raw_output": raw_output[:6000],
                "created_at": float(time.time()),
            },
            state={"confidence": 0.66 if not used_fallback else 0.52},
        )
        patch_node_ids: list[int] = []
        for focus_node_id in focus_ids:
            self._connect_nodes(
                from_node=int(session_node.id),
                to_node=int(focus_node_id),
                relation_type="monitors_graph_node",
                weight=0.64,
                logic_rule="graph_monitor",
            )

        for patch in node_patches:
            target = self.api.engine.get_node(int(patch["node_id"]))
            if target is None:
                continue
            attrs = self._as_mapping(getattr(target, "attributes", {}))
            current_summary = " ".join(str(attrs.get("summary", "") or "").split()).strip()
            target.attributes["monitor_summary"] = patch["summary"]
            target.attributes["monitor_reason"] = patch["reason"]
            target.attributes["monitor_updated_at"] = float(time.time())
            if not current_summary:
                target.attributes["summary"] = patch["summary"]
            target.state["monitor_confidence"] = patch["confidence"]

            patch_node = self.api.engine.create_node(
                "graph_monitor_patch",
                attributes={
                    "user_id": user_id,
                    "session_id": session_id,
                    "source": source,
                    "patch_type": "node",
                    "target_node_id": int(target.id),
                    "summary": patch["summary"],
                    "reason": patch["reason"],
                    "created_at": float(time.time()),
                },
                state={"confidence": patch["confidence"]},
            )
            patch_node_ids.append(int(patch_node.id))
            self._connect_nodes(
                from_node=int(session_node.id),
                to_node=int(patch_node.id),
                relation_type="records_monitor_patch",
                weight=patch["confidence"],
                logic_rule="graph_monitor",
            )
            self._connect_nodes(
                from_node=int(patch_node.id),
                to_node=int(target.id),
                relation_type="patches_node",
                weight=patch["confidence"],
                logic_rule="graph_monitor",
            )

        for patch in edge_patches:
            target_edge = None
            for edge in self.api.engine.store.edges:
                if int(getattr(edge, "from_node", 0) or 0) != int(patch["from_node"]):
                    continue
                if int(getattr(edge, "to_node", 0) or 0) != int(patch["to_node"]):
                    continue
                target_edge = edge
                if str(getattr(edge, "relation_type", "") or "") == str(patch["relation_type"]):
                    break
            if target_edge is None:
                self._connect_nodes(
                    from_node=int(patch["from_node"]),
                    to_node=int(patch["to_node"]),
                    relation_type=str(patch["relation_type"]),
                    weight=float(patch["weight"]),
                    logic_rule="graph_monitor",
                    metadata={
                        "source": source,
                        "reason": patch["reason"],
                        "action": patch["action"],
                    },
                )
            else:
                target_edge.relation_type = str(patch["relation_type"])
                target_edge.weight = float(patch["weight"])
                target_edge.logic_rule = "graph_monitor"
                target_edge.metadata = {
                    **self._as_mapping(getattr(target_edge, "metadata", {})),
                    "source": source,
                    "reason": patch["reason"],
                    "action": patch["action"],
                }

            patch_node = self.api.engine.create_node(
                "graph_monitor_patch",
                attributes={
                    "user_id": user_id,
                    "session_id": session_id,
                    "source": source,
                    "patch_type": "edge",
                    "from_node": int(patch["from_node"]),
                    "to_node": int(patch["to_node"]),
                    "relation_type": str(patch["relation_type"]),
                    "reason": patch["reason"],
                    "created_at": float(time.time()),
                },
                state={"confidence": float(patch["weight"])},
            )
            patch_node_ids.append(int(patch_node.id))
            self._connect_nodes(
                from_node=int(session_node.id),
                to_node=int(patch_node.id),
                relation_type="records_monitor_patch",
                weight=float(patch["weight"]),
                logic_rule="graph_monitor",
            )
            self._connect_nodes(
                from_node=int(patch_node.id),
                to_node=int(patch["from_node"]),
                relation_type="patches_relation_source",
                weight=float(patch["weight"]),
                logic_rule="graph_monitor",
            )
            self._connect_nodes(
                from_node=int(patch_node.id),
                to_node=int(patch["to_node"]),
                relation_type="patches_relation_target",
                weight=float(patch["weight"]),
                logic_rule="graph_monitor",
            )

        self.api.engine._record_event(  # noqa: SLF001
            "graph_monitor_applied",
            {
                "user_id": user_id,
                "session_id": session_id,
                "source": source,
                "focus_nodes": len(focus_ids),
                "node_patches": len(node_patches),
                "edge_patches": len(edge_patches),
                "used_fallback": bool(used_fallback),
            },
        )

        return {
            "enabled": True,
            "attached": True,
            "session_node_id": int(session_node.id),
            "patch_node_ids": patch_node_ids,
            "node_patches": node_patches,
            "edge_patches": edge_patches,
            "node_patch_count": len(node_patches),
            "edge_patch_count": len(edge_patches),
            "raw_output": raw_output[:6000],
            "model": {
                "requested_model_path": requested_model_path,
                "selected_model_path": str(resolution.get("selected_model_path", "") or requested_model_path),
                "requested_role": resolved_role,
                "resolution": dict(resolution),
                "used_fallback": bool(used_fallback),
            },
        }

    def _capture_input_intelligence(
        self,
        *,
        user_id: str,
        session_id: str,
        source: str,
        text: str,
        context: str = "",
        related_node_ids: list[int] | None = None,
        apply_to_graph: bool = True,
        model_path: str = "",
        model_role: str = "analyst",
    ) -> dict[str, Any]:
        clean_text = " ".join(str(text or "").split()).strip()
        clean_context = " ".join(str(context or "").split()).strip()
        related_ids = [
            self._to_int(item, 0)
            for item in list(related_node_ids or [])
            if self._to_int(item, 0) > 0
        ]
        if not clean_text:
            return {
                "enabled": False,
                "source": source,
                "summary": "",
                "updates": [],
                "updates_count": 0,
                "verification": {
                    "verified": False,
                    "verification_mode": "balanced",
                    "schema_valid": False,
                    "issue_count": 0,
                    "warning_count": 0,
                    "issues": [],
                    "warnings": [],
                    "checked_updates": 0,
                    "hallucination_guard_hits": 0,
                    "hallucination_matches": [],
                    "value_conflicts": [],
                    "score": 0.0,
                },
                "graph_binding": {
                    "attached": False,
                    "branch_node_id": 0,
                    "session_node_id": 0,
                    "update_node_ids": [],
                    "subject_binding": {
                        "attached": False,
                        "gossip_detected": False,
                        "mode": "off",
                        "subject_branch_node_ids": [],
                        "claim_node_ids": [],
                    },
                },
                "raw_output": "",
                "model": {
                    "requested_model_path": str(model_path or ""),
                    "selected_model_path": "",
                    "requested_role": str(model_role or "analyst"),
                    "resolution": {"mode": "skipped"},
                    "used_fallback": True,
                },
                "graph_monitor": {
                    "enabled": False,
                    "attached": False,
                    "session_node_id": 0,
                    "patch_node_ids": [],
                    "node_patch_count": 0,
                    "edge_patch_count": 0,
                    "raw_output": "",
                    "model": {
                        "requested_model_path": _DEFAULT_GRAPH_MONITOR_MODEL_PATH,
                        "selected_model_path": "",
                        "requested_role": "planner",
                        "resolution": {"mode": "skipped"},
                        "used_fallback": True,
                    },
                },
            }

        requested_model_path = str(model_path or "").strip() or _DEFAULT_STRUCTURED_INPUT_MODEL_PATH
        resolved_role = self._debate_role(model_role, fallback="analyst")
        llm_fn, resolution = self._resolve_role_or_model_llm(
            role=resolved_role,
            model_path=requested_model_path,
        )
        raw_output = ""
        summary = ""
        updates: list[dict[str, Any]] = []
        used_fallback = True
        if llm_fn is not None:
            prompt = (
                "You convert user input into structured archive updates for a graph memory.\n"
                "Return JSON only.\n"
                "Schema:\n"
                '{ "summary":"", "archive_updates":[{ "entity":"", "field":"", "operation":"upsert|append|correct|deprecate", '
                '"value":"any-json-value", "reason":"", "source":"", "confidence":0.0, "tags":["optional"] }] }\n'
                "Use concise values, keep confidence realistic, and do not invent unsupported facts.\n"
                f"Source: {source}\n"
                f"Input: {clean_text[:2600]}\n"
                f"Context: {clean_context[:1200]}\n"
            )
            try:
                raw_output = str(llm_fn(prompt) or "").strip()
                parsed_output = self._extract_json_from_llm_output(raw_output)
                if isinstance(parsed_output, Mapping):
                    summary = " ".join(str(parsed_output.get("summary", "") or "").split()).strip()
                    updates = self._normalize_archive_updates(
                        parsed_output.get("archive_updates", parsed_output.get("updates", []))
                    )
                    used_fallback = not bool(updates)
                elif isinstance(parsed_output, list):
                    updates = self._normalize_archive_updates(parsed_output)
                    used_fallback = not bool(updates)
            except Exception:
                summary = ""
                updates = []
                used_fallback = True

        if not updates:
            summary, updates = self._heuristic_input_updates(
                source=source,
                text=clean_text,
                context=clean_context,
                limit=4,
            )

        verification = self._verify_archive_updates(
            user_id=user_id,
            message=clean_text,
            updates=updates,
            verification_mode="balanced",
            top_k=3,
        )
        selected_model_path = str(resolution.get("selected_model_path", "") or requested_model_path)
        graph_binding = self._attach_archive_updates_to_graph(
            user_id=user_id,
            session_id=session_id,
            message=clean_text,
            context=clean_context,
            summary=summary,
            updates=updates,
            verification=verification,
            model_role=resolved_role,
            model_path=selected_model_path,
            resolution_mode=str(resolution.get("mode", "unavailable") or "unavailable"),
            raw_output=raw_output,
            verification_mode="balanced",
            apply_to_graph=apply_to_graph,
            subject_name="",
            gossip_mode="off",
            allow_subject_branch_write=False,
        )
        if apply_to_graph and graph_binding.get("attached", False):
            self._link_input_capture_to_related_nodes(
                graph_binding=graph_binding,
                related_node_ids=related_ids,
                source=source,
            )

        monitor_focus_ids = [
            *related_ids,
            self._to_int(self._as_mapping(graph_binding).get("session_node_id", 0), 0),
            *[
                self._to_int(item, 0)
                for item in self._as_list(self._as_mapping(graph_binding).get("update_node_ids"))
                if self._to_int(item, 0) > 0
            ],
        ]
        graph_monitor = self._run_graph_monitor(
            user_id=user_id,
            session_id=session_id,
            source=f"{source}_monitor",
            focus_node_ids=monitor_focus_ids,
            hint_text=f"{summary}\n{clean_text[:1200]}",
            model_path=_DEFAULT_GRAPH_MONITOR_MODEL_PATH,
            model_role="planner",
            apply_changes=apply_to_graph,
        )

        return {
            "enabled": True,
            "source": source,
            "summary": summary,
            "updates": updates,
            "updates_count": len(updates),
            "verification": verification,
            "graph_binding": graph_binding,
            "raw_output": raw_output[:6000],
            "model": {
                "requested_model_path": requested_model_path,
                "selected_model_path": selected_model_path,
                "requested_role": resolved_role,
                "resolution": dict(resolution),
                "used_fallback": bool(used_fallback),
            },
            "graph_monitor": graph_monitor,
        }

    def _normalize_triage_items(
        self,
        value: Any,
        *,
        fallback_category: str = "fact",
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        rows = value if isinstance(value, list) else []
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            item = self._as_mapping(row)
            if not item and isinstance(row, str):
                item = {"summary": row}
            summary = " ".join(
                str(
                    item.get(
                        "summary",
                        item.get("text", item.get("title", item.get("content", ""))),
                    )
                    or ""
                ).split()
            ).strip()
            if not summary:
                continue
            category = self._pick_allowed_token(
                item.get("category", fallback_category),
                allowed=_INTERACTION_TRIAGE_CATEGORIES_ALLOWED,
                default=fallback_category,
            )
            reason = " ".join(str(item.get("reason", "") or "").split()).strip()
            confidence_default = 0.68 if category in {"action", "risk"} else 0.6
            confidence = self._confidence(item.get("confidence", confidence_default), confidence_default)
            fingerprint = f"{category}:{self._normalize_token(summary)[:220]}"
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            out.append(
                {
                    "category": category,
                    "summary": summary[:360],
                    "confidence": round(confidence, 4),
                    "reason": reason[:420],
                }
            )
            if len(out) >= max(1, int(limit)):
                break
        return out

    def _heuristic_interaction_triage_items(
        self,
        *,
        message: str,
        reply: str,
        updates: list[dict[str, Any]] | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        action_hints = (
            "should",
            "need to",
            "next",
            "plan",
            "action",
            "todo",
            "нужно",
            "надо",
            "сдел",
            "план",
            "следующ",
            "պետք է",
        )
        risk_hints = (
            "risk",
            "issue",
            "blocker",
            "failure",
            "опас",
            "риск",
            "угроз",
            "խնդիր",
            "վտանգ",
        )
        noise_hints = (
            "rumor",
            "gossip",
            "сплет",
            "слух",
            "перемыва",
            "junk",
            "trash",
            "noise",
            "tmp",
            "irrelevant",
        )
        style_hints = (
            "style",
            "tone",
            "format",
            "коротк",
            "стиль",
            "тон",
            "ձև",
            "ոճ",
        )
        source_parts: list[str] = []
        source_parts.extend(self._split_sentences(message, limit=6))
        source_parts.extend(self._split_sentences(reply, limit=6))
        for row in list(updates or [])[:4]:
            item = self._as_mapping(row)
            entity = str(item.get("entity", "") or "").strip()
            field = str(item.get("field", "") or "").strip()
            reason = str(item.get("reason", "") or "").strip()
            value = str(item.get("value", "") or "").strip()
            update_line = " ".join(part for part in (entity, field, reason, value) if part).strip()
            if update_line:
                source_parts.append(update_line)

        heuristics: list[dict[str, Any]] = []
        for sentence in source_parts:
            normalized = " ".join(str(sentence or "").split()).strip()
            if not normalized:
                continue
            lowered = normalized.casefold()
            category = "fact"
            reason = "general_signal"
            confidence = 0.58
            if any(hint in lowered for hint in action_hints):
                category = "action"
                reason = "action_hint"
                confidence = 0.76
            elif any(hint in lowered for hint in risk_hints):
                category = "risk"
                reason = "risk_hint"
                confidence = 0.72
            elif any(hint in lowered for hint in noise_hints):
                category = "noise"
                reason = "noise_hint"
                confidence = 0.78
            elif any(hint in lowered for hint in style_hints):
                category = "style"
                reason = "style_hint"
                confidence = 0.66

            heuristics.append(
                {
                    "category": category,
                    "summary": normalized,
                    "confidence": confidence,
                    "reason": reason,
                }
            )
            if len(heuristics) >= max(3, int(limit) * 2):
                break
        return self._normalize_triage_items(heuristics, fallback_category="fact", limit=limit)

    def _auto_interaction_triage(
        self,
        *,
        user_id: str,
        session_id: str,
        source: str,
        message: str,
        reply: str,
        updates: list[dict[str, Any]] | None = None,
        auto_triage: bool = True,
        triage_with_llm: bool = True,
        model_role: str = "analyst",
        model_path: str = "",
        attach_to_graph: bool = True,
        related_node_id: int = 0,
        llm_fn: Callable[[str], str] | None = None,
        llm_resolution: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not auto_triage:
            return {
                "enabled": False,
                "source": str(source or "wrapper"),
                "items": [],
                "counts": {},
                "signal_score": 0.0,
                "noise_ratio": 0.0,
                "graph": {
                    "attached": False,
                    "session_node_id": 0,
                    "item_node_ids": [],
                },
                "llm": {
                    "requested": False,
                    "used": False,
                    "resolution": dict(llm_resolution or {}),
                },
            }

        normalized_updates = [self._as_mapping(row) for row in list(updates or [])]
        heuristic_items = self._heuristic_interaction_triage_items(
            message=message,
            reply=reply,
            updates=normalized_updates,
            limit=8,
        )

        llm_requested = bool(triage_with_llm)
        llm_used = False
        llm_raw_output = ""
        llm_items: list[dict[str, Any]] = []
        llm_resolution_payload = self._as_mapping(llm_resolution)
        resolved_llm_fn = llm_fn
        if llm_requested and resolved_llm_fn is None:
            resolved_llm_fn, llm_resolution_payload = self._resolve_role_or_model_llm(
                role=self._debate_role(model_role, fallback="analyst"),
                model_path=model_path,
            )

        if llm_requested and resolved_llm_fn is not None:
            triage_prompt = (
                "You are a signal triage assistant.\n"
                "Classify interaction fragments into categories: action, fact, risk, style, noise.\n"
                "Return JSON only:\n"
                '{ "items":[{ "category":"action|fact|risk|style|noise", "summary":"", "confidence":0.0, "reason":"" }] }\n'
                "Keep max 8 concise items and avoid duplicates.\n"
                f"Message: {message}\n"
                f"Reply: {reply}\n"
                f"Structured updates: {self._stable_json(normalized_updates)[:1600]}\n"
            )
            try:
                llm_raw_output = str(resolved_llm_fn(triage_prompt) or "").strip()
                parsed = self._extract_json_from_llm_output(llm_raw_output)
                parsed_items = []
                if isinstance(parsed, Mapping):
                    parsed_items = parsed.get("items", parsed.get("triage_items", []))
                elif isinstance(parsed, list):
                    parsed_items = parsed
                llm_items = self._normalize_triage_items(parsed_items, fallback_category="fact", limit=8)
                llm_used = bool(llm_items)
            except Exception:
                llm_items = []
                llm_used = False

        items = llm_items or heuristic_items
        counts = Counter(str(row.get("category", "fact") or "fact") for row in items)
        total = max(1, len(items))
        signal_total = float(counts.get("action", 0) + counts.get("fact", 0) + counts.get("risk", 0))
        noise_total = float(counts.get("noise", 0))
        signal_score = max(0.0, min(1.0, signal_total / float(total)))
        noise_ratio = max(0.0, min(1.0, noise_total / float(total)))

        graph_session_node_id = 0
        graph_item_node_ids: list[int] = []
        graph_attached = False
        if attach_to_graph and items:
            triage_session_node = self.api.engine.create_node(
                "llm_triage_session",
                attributes={
                    "user_id": user_id,
                    "session_id": session_id,
                    "source": str(source or "wrapper"),
                    "message": message[:1600],
                    "reply": reply[:2400],
                    "counts": dict(counts),
                    "signal_score": round(signal_score, 4),
                    "noise_ratio": round(noise_ratio, 4),
                    "llm_used": bool(llm_used),
                    "created_at": float(time.time()),
                },
                state={"confidence": max(0.3, 1.0 - (noise_ratio * 0.65))},
            )
            graph_session_node_id = int(triage_session_node.id)
            graph_attached = True
            if related_node_id > 0 and self.api.engine.get_node(related_node_id) is not None:
                self._connect_nodes(
                    from_node=int(related_node_id),
                    to_node=int(triage_session_node.id),
                    relation_type="triaged_by",
                    weight=max(0.45, signal_score),
                    logic_rule="interaction_triage",
                )

            for index, item in enumerate(items, start=1):
                category = self._pick_allowed_token(
                    item.get("category", "fact"),
                    allowed=_INTERACTION_TRIAGE_CATEGORIES_ALLOWED,
                    default="fact",
                )
                triage_item_node = self.api.engine.create_node(
                    "llm_triage_item",
                    attributes={
                        "user_id": user_id,
                        "session_id": session_id,
                        "source": str(source or "wrapper"),
                        "name": self._to_title(
                            f"{category}: {str(item.get('summary', '') or '')}",
                            fallback=f"Triage Item {index}",
                            limit=96,
                        ),
                        "category": category,
                        "summary": str(item.get("summary", "") or "")[:600],
                        "reason": str(item.get("reason", "") or "")[:600],
                        "rank": index,
                        "created_at": float(time.time()),
                    },
                    state={"confidence": self._confidence(item.get("confidence", 0.6), 0.6)},
                )
                graph_item_node_ids.append(int(triage_item_node.id))
                self._connect_nodes(
                    from_node=int(triage_session_node.id),
                    to_node=int(triage_item_node.id),
                    relation_type="triage_item",
                    weight=self._confidence(item.get("confidence", 0.6), 0.6),
                    logic_rule="interaction_triage",
                    metadata={"category": category, "rank": index},
                )

        return {
            "enabled": True,
            "source": str(source or "wrapper"),
            "items": items,
            "counts": dict(counts),
            "signal_score": round(signal_score, 4),
            "noise_ratio": round(noise_ratio, 4),
            "graph": {
                "attached": bool(graph_attached),
                "session_node_id": int(graph_session_node_id),
                "item_node_ids": graph_item_node_ids,
            },
            "llm": {
                "requested": bool(llm_requested),
                "used": bool(llm_used),
                "resolution": llm_resolution_payload,
                "raw_output": llm_raw_output[:3000],
            },
        }

    def _iter_user_nodes(
        self,
        *,
        user_id: str = "",
        namespace: str = "",
        scope: str = "all",
    ) -> list[Any]:
        owner = str(user_id or "").strip()
        mode = self._pick_allowed_token(scope, allowed=_MEMORY_SCOPE_ALLOWED, default="all")
        namespace_token = ""
        if str(namespace or "").strip():
            namespace_token = self._normalize_namespace(namespace, default="global")

        out: list[Any] = []
        for node in self.api.engine.store.nodes.values():
            if namespace_token and self._node_namespace(node, default="global") != namespace_token:
                continue
            if mode == "owned":
                if not owner:
                    continue
                if not self._node_belongs_to_user(node, owner):
                    continue
            out.append(node)
        return out

    def _graph_degree_map(self, node_ids: set[int] | None = None) -> dict[int, int]:
        degrees: dict[int, int] = {}
        selected = set(node_ids or set())
        for edge in self.api.engine.store.edges:
            left = int(edge.from_node)
            right = int(edge.to_node)
            if selected and left not in selected and right not in selected:
                continue
            degrees[left] = degrees.get(left, 0) + 1
            degrees[right] = degrees.get(right, 0) + 1
        return degrees

    def _has_negation_hint(self, text: str) -> bool:
        source = f" {str(text or '').strip().lower()} "
        return any(hint in source for hint in _CONTRADICTION_NEGATION_HINTS)

    @staticmethod
    def _project_backups_dir() -> Path:
        return Path("data/project_backups")

    def _write_project_backup(self, payload: Mapping[str, Any], *, label: str = "") -> str:
        base = self._project_backups_dir()
        base.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        token = self._safe_slug(label, default="manual")
        path = base / f"project_backup_{stamp}_{token}.json"
        suffix = 1
        while path.exists():
            path = base / f"project_backup_{stamp}_{token}_{suffix:02d}.json"
            suffix += 1
        path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def _collect_audit_events(self, *, limit: int = 200) -> list[dict[str, Any]]:
        interesting = {
            "project_package_action",
            "memory_namespace_applied",
            "graph_rag_query",
            "contradiction_scan_completed",
            "task_risk_board_generated",
            "timeline_replay_requested",
            "llm_policy_updated",
            "quality_harness_ran",
            "project_backup_created",
            "project_backup_restored",
        }
        rows = self.list_events(limit=max(10, min(4000, int(limit))))
        out = [row for row in rows if str(row.get("event_type", "")) in interesting]
        return out[-max(1, min(1000, int(limit))):]

    def _restore_graph_from_snapshot(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        nodes = self._as_list(snapshot.get("nodes"))
        edges = self._as_list(snapshot.get("edges"))
        self.api.engine.store.nodes.clear()
        self.api.engine.store.edges.clear()
        self.api.engine.clear_event_log()
        self.api.engine._next_node_id = 1  # noqa: SLF001

        created_nodes = 0
        created_edges = 0
        for item in sorted(nodes, key=lambda row: self._to_int(self._as_mapping(row).get("id", 0), 0)):
            row = self._as_mapping(item)
            node_id = self._to_int(row.get("id", 0), 0)
            if node_id <= 0:
                continue
            node_type = str(row.get("type", "generic") or "generic").strip() or "generic"
            attrs = self._as_mapping(row.get("attributes"))
            state = self._as_mapping(row.get("state"))
            try:
                self.api.engine.create_node(
                    node_type,
                    node_id=node_id,
                    attributes=attrs,
                    state=state,
                )
                created_nodes += 1
            except Exception:
                continue

        for item in edges:
            row = self._as_mapping(item)
            from_node = self._to_int(row.get("from", 0), 0)
            to_node = self._to_int(row.get("to", 0), 0)
            relation_type = str(row.get("relation_type", "") or "").strip()
            if from_node <= 0 or to_node <= 0 or not relation_type:
                continue
            if self.api.engine.get_node(from_node) is None or self.api.engine.get_node(to_node) is None:
                continue
            self._connect_nodes(
                from_node=from_node,
                to_node=to_node,
                relation_type=relation_type,
                weight=self._confidence(row.get("weight", 0.6), 0.6),
                logic_rule=str(row.get("logic_rule", "backup_restore") or "backup_restore"),
                metadata=self._as_mapping(row.get("metadata")),
            )
            created_edges += 1

        return {
            "created_nodes": int(created_nodes),
            "created_edges": int(created_edges),
        }

    def _wrapper_profile_defaults(self, *, user_id: str) -> dict[str, Any]:
        now = time.time()
        return {
            "user_id": user_id,
            "name": f"Wrapper Profile · {user_id}",
            "response_style": "adaptive",
            "reasoning_depth": "balanced",
            "risk_tolerance": "medium",
            "tone": "neutral",
            "focus_goals": [],
            "domain_focus": [],
            "avoid_topics": [],
            "memory_notes": "",
            "llm_roles": {
                "proposer": _DEBATE_DEFAULT_ROLES["proposer"],
                "critic": _DEBATE_DEFAULT_ROLES["critic"],
                "judge": _DEBATE_DEFAULT_ROLES["judge"],
            },
            "preferred_role": "general",
            "preferred_model_path": "",
            "memory_scope": "owned",
            "feedback_total": 0,
            "feedback_positive": 0,
            "feedback_negative": 0,
            "feedback_recent": [],
            "dialect_dictionary": [],
            "dialect_last_update_at": 0.0,
            "created_at": now,
            "updated_at": now,
        }

    def _ensure_wrapper_profile_node(self, *, user_id: str):
        defaults = self._wrapper_profile_defaults(user_id=user_id)
        node, _ = self._ensure_shared_node(
            node_type=_WRAPPER_PROFILE_NODE_TYPE,
            identity_key="user_id",
            identity_value=user_id,
            attributes=defaults,
        )
        attrs = self._as_mapping(node.attributes)
        for key, value in defaults.items():
            if key not in attrs:
                node.attributes[key] = value
        node.attributes["updated_at"] = time.time()
        return node

    def _wrapper_profile_payload(self, node: Any) -> dict[str, Any]:
        attrs = self._as_mapping(getattr(node, "attributes", {}))
        personalization = self._sanitize_personalization(
            {
                "response_style": attrs.get("response_style", "adaptive"),
                "reasoning_depth": attrs.get("reasoning_depth", "balanced"),
                "risk_tolerance": attrs.get("risk_tolerance", "medium"),
                "tone": attrs.get("tone", "neutral"),
                "focus_goals": attrs.get("focus_goals", []),
                "domain_focus": attrs.get("domain_focus", []),
                "avoid_topics": attrs.get("avoid_topics", []),
                "memory_notes": attrs.get("memory_notes", ""),
                "llm_roles": attrs.get("llm_roles", {}),
            }
        )
        preferred_role = self._debate_role(attrs.get("preferred_role"), fallback="general")
        preferred_model_path = str(attrs.get("preferred_model_path", "") or "").strip()
        memory_scope = self._pick_allowed_token(
            attrs.get("memory_scope", "owned"),
            allowed=_MEMORY_SCOPE_ALLOWED,
            default="owned",
        )
        return {
            "user_id": str(attrs.get("user_id", "") or "").strip(),
            "name": str(attrs.get("name", "") or "").strip(),
            "preferred_role": preferred_role,
            "preferred_model_path": preferred_model_path,
            "memory_scope": memory_scope,
            "personalization": personalization,
            "feedback": {
                "total": self._to_int(attrs.get("feedback_total", 0), 0),
                "positive": self._to_int(attrs.get("feedback_positive", 0), 0),
                "negative": self._to_int(attrs.get("feedback_negative", 0), 0),
                "recent": self._as_list(attrs.get("feedback_recent"))[-12:],
                "updated_at": float(attrs.get("updated_at", 0.0) or 0.0),
            },
            "dialect": {
                "dictionary": self._as_list(attrs.get("dialect_dictionary"))[:60],
                "updated_at": float(attrs.get("dialect_last_update_at", 0.0) or 0.0),
            },
        }

    def _merge_personalization(
        self,
        base: Mapping[str, Any] | None,
        override: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        left = self._sanitize_personalization(base)
        right = self._sanitize_personalization(override)
        if not left:
            return right
        if not right:
            return left

        out = dict(left)
        for key in ("response_style", "reasoning_depth", "risk_tolerance", "tone", "language"):
            if key in right and str(right.get(key, "")).strip():
                out[key] = right[key]

        for key in ("focus_goals", "domain_focus", "avoid_topics"):
            merged = self._dedupe_strings(
                self._to_list_of_strings(left.get(key))
                + self._to_list_of_strings(right.get(key)),
                limit=24,
            )
            if merged:
                out[key] = merged

        notes = " ".join(
            str(right.get("memory_notes", "") or left.get("memory_notes", "") or "").split()
        ).strip()
        if notes:
            out["memory_notes"] = notes[:1200]

        roles = {
            **self._as_mapping(left.get("llm_roles")),
            **self._as_mapping(right.get("llm_roles")),
        }
        if roles:
            out["llm_roles"] = roles
        return self._sanitize_personalization(out)

    def _wrapper_memory_context(
        self,
        *,
        user_id: str,
        query: str,
        scope: str = "owned",
        namespace: str = "",
        top_k: int = 6,
    ) -> list[dict[str, Any]]:
        query_tokens = self._token_set(query)
        nodes = self._iter_user_nodes(user_id=user_id, namespace=namespace, scope=scope)
        if not nodes:
            nodes = list(self.api.engine.store.nodes.values())
        degree_map = self._graph_degree_map({int(node.id) for node in nodes})

        rows: list[dict[str, Any]] = []
        for node in nodes:
            if str(node.type or "") in _WRAPPER_CONTEXT_NODE_TYPES_DENY:
                continue
            text_blob = self._node_text_blob(node)
            if not text_blob:
                continue
            node_tokens = self._token_set(text_blob)
            overlap = self._jaccard_similarity(query_tokens, node_tokens) if query_tokens else 0.0
            degree_boost = min(1.0, float(degree_map.get(int(node.id), 0)) / 12.0)
            user_boost = 0.08 if self._node_belongs_to_user(node, user_id) else 0.0
            score = (0.78 * overlap) + (0.22 * degree_boost) + user_boost
            if query_tokens and score < 0.08:
                continue
            attrs = self._as_mapping(node.attributes)
            summary = (
                str(attrs.get("summary", "") or attrs.get("description", "") or attrs.get("note", "") or "").strip()
            )
            if len(summary) > 240:
                summary = f"{summary[:237].rstrip()}..."
            rows.append(
                {
                    "node_id": int(node.id),
                    "type": str(node.type or "generic"),
                    "name": str(attrs.get("name", "") or attrs.get("title", "") or "").strip() or f"Node {node.id}",
                    "summary": summary,
                    "namespace": self._node_namespace(node, default="global"),
                    "score": round(max(0.0, min(1.0, score)), 4),
                }
            )
        rows.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        return rows[: max(1, min(24, int(top_k)))]

    def _wrapper_prompt(
        self,
        *,
        message: str,
        personalization: Mapping[str, Any] | None,
        memory_context: list[Mapping[str, Any]],
    ) -> str:
        profile_block = self._personalization_prompt_context(personalization)
        context_lines: list[str] = []
        for idx, row in enumerate(memory_context[:8], start=1):
            title = str(row.get("name", "") or f"Node {row.get('node_id', idx)}")
            summary = str(row.get("summary", "") or "").strip()
            context_lines.append(
                f"[{idx}] {title} (node:{row.get('node_id')}, ns:{row.get('namespace', 'global')}, score:{row.get('score')})"
            )
            if summary:
                context_lines.append(f"    {summary}")
        context_block = "\n".join(context_lines).strip() or "No strong graph context retrieved."

        return (
            "You are an efficiency-first LLM wrapper assistant.\n"
            "Rules:\n"
            "- Be practical and concise.\n"
            "- Do not fabricate facts. If uncertain, state uncertainty explicitly.\n"
            "- Prioritize actionable next steps.\n"
            "- Use retrieved graph context when relevant.\n"
            f"{profile_block}\n"
            "Retrieved graph context:\n"
            f"{context_block}\n"
            "User message:\n"
            f"{message}\n"
            "Answer:"
        )

    def _wrapper_fallback_reply(
        self,
        *,
        message: str,
        memory_context: list[Mapping[str, Any]],
    ) -> str:
        lead = self._to_title(message, fallback="User request", limit=120)
        if memory_context:
            hints = ", ".join(str(item.get("name", "")) for item in memory_context[:3] if str(item.get("name", "")).strip())
            if hints:
                return f"{lead}. Context from your graph: {hints}. Next: refine goal, constraints, and first measurable action."
        return f"{lead}. Next: define goal, constraints, and first measurable action."

    def _apply_wrapper_feedback_to_profile(
        self,
        *,
        profile_node: Any,
        feedback_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        attrs = self._as_mapping(profile_node.attributes)
        accepted = 0
        rejected = 0
        combined_messages: list[str] = []
        for row in feedback_items:
            decision = str(row.get("decision", "") or "").strip()
            score = self._confidence(row.get("score", 0.0), 0.0)
            if decision in {"accept", "accepted", "like", "liked"} or score >= 0.66:
                accepted += 1
            if decision in {"reject", "rejected", "dislike", "discard"} or score <= 0.34:
                rejected += 1
            message = " ".join(str(row.get("message", "") or "").split()).strip()
            if message:
                combined_messages.append(message.lower())
        message_blob = " ".join(combined_messages)

        if any(hint in message_blob for hint in ("короче", "short", "concise", "кратко")):
            profile_node.attributes["response_style"] = "concise"
            profile_node.attributes["reasoning_depth"] = "quick"
        if any(hint in message_blob for hint in ("подроб", "detail", "deeper", "глубже")):
            profile_node.attributes["response_style"] = "deep"
            profile_node.attributes["reasoning_depth"] = "deep"
        if any(hint in message_blob for hint in ("осторож", "safe", "risk", "conservative")):
            profile_node.attributes["risk_tolerance"] = "low"
        if any(hint in message_blob for hint in ("смел", "aggressive", "bold")):
            profile_node.attributes["risk_tolerance"] = "high"

        prev_total = self._to_int(attrs.get("feedback_total", 0), 0)
        prev_pos = self._to_int(attrs.get("feedback_positive", 0), 0)
        prev_neg = self._to_int(attrs.get("feedback_negative", 0), 0)
        profile_node.attributes["feedback_total"] = prev_total + len(feedback_items)
        profile_node.attributes["feedback_positive"] = prev_pos + accepted
        profile_node.attributes["feedback_negative"] = prev_neg + rejected
        history = self._as_list(attrs.get("feedback_recent")) + feedback_items
        profile_node.attributes["feedback_recent"] = history[-20:]
        profile_node.attributes["updated_at"] = time.time()
        return {
            "new_items": len(feedback_items),
            "accepted": accepted,
            "rejected": rejected,
            "feedback_total": int(profile_node.attributes.get("feedback_total", 0) or 0),
        }

    def _normalize_gossip_mode(self, value: Any, *, default: str = "auto") -> str:
        return self._pick_allowed_token(
            value,
            allowed=_WRAPPER_GOSSIP_MODE_ALLOWED,
            default=default,
        )

    @staticmethod
    def _identity_slug(value: Any, *, fallback: str = "item") -> str:
        token = re.sub(r"[^\w]+", "_", str(value or "").strip().casefold(), flags=re.UNICODE).strip("_")
        return token or fallback

    def _looks_like_gossip(self, text: Any) -> bool:
        source = self._normalize_token(text)
        if not source:
            return False
        return any(marker in source for marker in _GOSSIP_MARKERS)

    def _extract_subject_candidates(
        self,
        *,
        message: str,
        explicit_subject: Any = "",
        updates: list[Mapping[str, Any]] | None = None,
    ) -> list[str]:
        candidates: list[str] = []
        explicit = " ".join(str(explicit_subject or "").split()).strip()
        if explicit:
            candidates.append(explicit[:120])

        for row in updates or []:
            entity = " ".join(str(row.get("entity", "") or "").split()).strip()
            if entity:
                candidates.append(entity[:120])

        source = str(message or "")
        for pattern in _SUBJECT_NAME_HINT_RE:
            for match in pattern.finditer(source):
                candidate = " ".join(str(match.group(1) or "").split()).strip(" \t\r\n,.;:!?")
                if not candidate:
                    continue
                if len(candidate) < 2:
                    continue
                candidates.append(candidate[:120])

        out = self._dedupe_strings(candidates, limit=8)
        return [item for item in out if len(item) >= 2]

    def _ensure_subject_branch_node(self, *, user_id: str, subject_name: str):
        clean = " ".join(str(subject_name or "").split()).strip()
        if not clean:
            clean = "Unknown Subject"
        subject_slug = self._identity_slug(clean, fallback="subject")
        subject_key = f"{user_id}:{subject_slug}"
        node, _ = self._ensure_shared_node(
            node_type="subject_profile_branch",
            identity_key="subject_key",
            identity_value=subject_key,
            attributes={
                "subject_key": subject_key,
                "user_id": user_id,
                "subject_name": clean,
                "name": clean,
                "description": "Subject-centric branch for socially sourced claims and context.",
                "created_at": float(time.time()),
            },
        )
        node.attributes["subject_name"] = clean
        node.attributes["name"] = clean
        node.attributes["updated_at"] = float(time.time())
        return node

    def _bind_subject_conversation(
        self,
        *,
        user_id: str,
        session_node_id: int,
        message: str,
        reply: str,
        explicit_subject: Any,
        updates: list[Mapping[str, Any]] | None,
        verification: Mapping[str, Any] | None,
        gossip_mode: str,
        allow_write: bool,
    ) -> dict[str, Any]:
        detected = self._looks_like_gossip(message)
        mode = self._normalize_gossip_mode(gossip_mode, default="auto")
        subjects = self._extract_subject_candidates(
            message=message,
            explicit_subject=explicit_subject,
            updates=updates,
        )

        binding: dict[str, Any] = {
            "attached": False,
            "gossip_detected": bool(detected),
            "mode": mode,
            "subject_branch_node_ids": [],
            "claim_node_ids": [],
        }
        if not allow_write:
            binding["blocked"] = "subject_branch_write_disabled"
            return binding
        if mode == "off":
            return binding
        if mode == "auto" and not detected and not subjects:
            return binding
        if not subjects:
            fallback_subject = self._to_title(
                message,
                fallback="Conversation Subject",
                limit=48,
            )
            subjects = [fallback_subject]

        verified = bool((verification or {}).get("verified", False))
        claim_rows = self._as_list(updates) if isinstance(updates, list) else []
        claim_node_ids: list[int] = []
        subject_node_ids: list[int] = []
        now = float(time.time())
        for subject in subjects[:6]:
            subject_node = self._ensure_subject_branch_node(user_id=user_id, subject_name=subject)
            subject_node_ids.append(int(subject_node.id))
            binding["attached"] = True

            if session_node_id > 0:
                self._connect_nodes(
                    from_node=int(session_node_id),
                    to_node=int(subject_node.id),
                    relation_type="discusses_subject",
                    weight=0.76 if detected else 0.62,
                    logic_rule="subject_branch_bind",
                    metadata={
                        "gossip_detected": bool(detected),
                        "mode": mode,
                    },
                )

            selected_rows: list[Mapping[str, Any]] = []
            for row in claim_rows:
                entity = self._normalize_token(row.get("entity"))
                if entity and entity == self._normalize_token(subject):
                    selected_rows.append(row)
            if not selected_rows:
                selected_rows = claim_rows[:3]

            if not selected_rows:
                selected_rows = [
                    {
                        "entity": subject,
                        "field": "conversation_summary",
                        "operation": "append",
                        "value": self._to_title(message, fallback="discussion", limit=180),
                        "reason": self._to_title(reply, fallback="conversation reply", limit=180),
                        "source": "conversation_memory",
                        "confidence": 0.4 if detected else 0.58,
                        "tags": ["gossip"] if detected else ["conversation"],
                    }
                ]

            for row in selected_rows[:8]:
                field = " ".join(str(row.get("field", "") or "").split()).strip() or "detail"
                operation = " ".join(str(row.get("operation", "") or "").split()).strip() or "append"
                reason = " ".join(str(row.get("reason", "") or "").split()).strip()
                source = " ".join(str(row.get("source", "") or "").split()).strip()
                claim_text = reason
                if not claim_text:
                    claim_text = f"{field}: {self._stable_json(row.get('value'))[:240]}"
                if not claim_text:
                    continue
                claim_key = self._hallucination_signature(user_id, subject, field, operation, claim_text)
                claim_node, _ = self._ensure_shared_node(
                    node_type="subject_claim_node",
                    identity_key="claim_key",
                    identity_value=claim_key,
                    attributes={
                        "claim_key": claim_key,
                        "user_id": user_id,
                        "subject_name": subject,
                        "field": field,
                        "operation": operation,
                        "claim": claim_text[:500],
                        "value": row.get("value"),
                        "source": source[:500],
                        "verified": bool(verified),
                        "gossip_flag": bool(detected),
                        "tags": self._to_list_of_strings(row.get("tags")),
                        "created_at": now,
                    },
                )
                claim_node.attributes["claim"] = claim_text[:500]
                claim_node.attributes["value"] = row.get("value")
                claim_node.attributes["source"] = source[:500]
                claim_node.attributes["verified"] = bool(verified)
                claim_node.attributes["gossip_flag"] = bool(detected)
                claim_node.attributes["subject_name"] = subject
                claim_node.attributes["updated_at"] = now
                claim_node.attributes["confidence"] = self._confidence(row.get("confidence", 0.55), 0.55)

                self._connect_nodes(
                    from_node=int(subject_node.id),
                    to_node=int(claim_node.id),
                    relation_type="has_subject_claim",
                    weight=self._confidence(row.get("confidence", 0.55), 0.55),
                    logic_rule="subject_branch_claim",
                    metadata={
                        "verified": bool(verified),
                        "gossip_flag": bool(detected),
                    },
                )
                claim_node_ids.append(int(claim_node.id))

        binding["subject_branch_node_ids"] = self._dedupe_strings([str(item) for item in subject_node_ids], limit=128)
        binding["subject_branch_node_ids"] = [int(item) for item in binding["subject_branch_node_ids"] if str(item).isdigit()]
        binding["claim_node_ids"] = self._dedupe_strings([str(item) for item in claim_node_ids], limit=256)
        binding["claim_node_ids"] = [int(item) for item in binding["claim_node_ids"] if str(item).isdigit()]
        return binding

    def _extract_dialect_terms(self, text: str) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for raw in _DIALECT_TOKEN_RE.findall(str(text or "")):
            token = self._normalize_token(raw).replace(" ", "")
            if not token:
                continue
            if token in _DIALECT_STOPWORDS:
                continue
            if token.isdigit():
                continue
            if len(token) < 2:
                continue
            counter[token] += 1
        out: dict[str, int] = {}
        for token, count in counter.most_common(80):
            if count <= 0:
                continue
            if count == 1 and len(token) > 8:
                continue
            out[token] = int(count)
        return out

    def _capture_wrapper_dialect(
        self,
        *,
        user_id: str,
        profile_node: Any,
        message: str,
        reply: str,
        attach_to_graph: bool,
    ) -> dict[str, Any]:
        merged = f"{message}\n{reply}".strip()
        terms = self._extract_dialect_terms(merged)
        summary = {
            "captured_terms": 0,
            "dictionary_size": 0,
            "top_terms": [],
            "dictionary_node_id": 0,
        }
        if not terms:
            return summary

        attrs = self._as_mapping(profile_node.attributes)
        existing_rows = self._as_list(attrs.get("dialect_dictionary"))
        by_term: dict[str, dict[str, Any]] = {}
        for row in existing_rows:
            item = self._as_mapping(row)
            term = self._normalize_token(item.get("term")).replace(" ", "")
            if not term:
                continue
            by_term[term] = {
                "term": term,
                "count": self._to_int(item.get("count", 0), 0),
                "last_seen_at": float(item.get("last_seen_at", 0.0) or 0.0),
            }

        now = float(time.time())
        for term, count in terms.items():
            row = by_term.get(term, {"term": term, "count": 0, "last_seen_at": 0.0})
            row["count"] = self._to_int(row.get("count", 0), 0) + int(count)
            row["last_seen_at"] = now
            by_term[term] = row

        dictionary_rows = list(by_term.values())
        dictionary_rows.sort(
            key=lambda row: (
                int(row.get("count", 0) or 0),
                float(row.get("last_seen_at", 0.0) or 0.0),
            ),
            reverse=True,
        )
        dictionary_rows = dictionary_rows[:160]
        profile_node.attributes["dialect_dictionary"] = dictionary_rows
        profile_node.attributes["dialect_last_update_at"] = now

        dictionary_node_id = 0
        if attach_to_graph:
            dictionary_node, _ = self._ensure_shared_node(
                node_type="user_dialect_dictionary",
                identity_key="user_id",
                identity_value=user_id,
                attributes={
                    "user_id": user_id,
                    "name": f"Dialect Dictionary ({user_id})",
                    "description": "Adaptive short-style/slang lexicon extracted from user interactions.",
                    "created_at": now,
                },
            )
            dictionary_node.attributes["terms_count"] = len(dictionary_rows)
            dictionary_node.attributes["top_terms"] = [row.get("term") for row in dictionary_rows[:20]]
            dictionary_node.attributes["updated_at"] = now
            dictionary_node_id = int(dictionary_node.id)

            self._connect_nodes(
                from_node=int(profile_node.id),
                to_node=int(dictionary_node.id),
                relation_type="tracks_dialect_dictionary",
                weight=0.83,
                logic_rule="wrapper_dialect_capture",
            )

            for row in dictionary_rows[:24]:
                term = str(row.get("term", "") or "").strip()
                if not term:
                    continue
                term_key = f"{user_id}:{self._identity_slug(term, fallback='term')}"
                term_node, _ = self._ensure_shared_node(
                    node_type="dialect_term",
                    identity_key="term_key",
                    identity_value=term_key,
                    attributes={
                        "term_key": term_key,
                        "user_id": user_id,
                        "term": term,
                        "created_at": now,
                    },
                )
                term_node.attributes["term"] = term
                term_node.attributes["count"] = self._to_int(row.get("count", 0), 0)
                term_node.attributes["last_seen_at"] = float(row.get("last_seen_at", now) or now)
                self._connect_nodes(
                    from_node=int(dictionary_node.id),
                    to_node=int(term_node.id),
                    relation_type="has_dialect_term",
                    weight=min(1.0, 0.3 + (0.06 * float(self._to_int(row.get("count", 1), 1)))),
                    logic_rule="wrapper_dialect_capture",
                )

        summary["captured_terms"] = len(terms)
        summary["dictionary_size"] = len(dictionary_rows)
        summary["top_terms"] = [str(row.get("term", "") or "") for row in dictionary_rows[:12]]
        summary["dictionary_node_id"] = int(dictionary_node_id)
        return summary

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
        subject_name = " ".join(str(root.get("subject_name", "") or "").split()).strip()
        gossip_mode = self._normalize_gossip_mode(root.get("gossip_mode"), default="auto")
        allow_subject_branch_write = self._to_bool(root.get("allow_subject_branch_write", True))
        capture_dialect = self._to_bool(root.get("capture_dialect", True))
        auto_triage = self._to_bool(root.get("auto_triage", True))
        triage_with_llm = self._to_bool(root.get("triage_with_llm", True))
        return {
            "message": message[:2400],
            "context": context[:2400],
            "model_path": model_path,
            "model_role": model_role,
            "verification_mode": verification_mode,
            "top_k": top_k,
            "apply_to_graph": apply_to_graph,
            "subject_name": subject_name[:140],
            "gossip_mode": gossip_mode,
            "allow_subject_branch_write": allow_subject_branch_write,
            "capture_dialect": capture_dialect,
            "auto_triage": auto_triage,
            "triage_with_llm": triage_with_llm,
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
        subject_name: str = "",
        gossip_mode: str = "auto",
        allow_subject_branch_write: bool = True,
    ) -> dict[str, Any]:
        graph_binding: dict[str, Any] = {
            "attached": False,
            "branch_node_id": 0,
            "session_node_id": 0,
            "update_node_ids": [],
            "subject_binding": {
                "attached": False,
                "gossip_detected": False,
                "mode": self._normalize_gossip_mode(gossip_mode, default="auto"),
                "subject_branch_node_ids": [],
                "claim_node_ids": [],
            },
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
        subject_binding = self._bind_subject_conversation(
            user_id=user_id,
            session_node_id=int(session_node.id),
            message=message,
            reply=summary or "",
            explicit_subject=subject_name,
            updates=updates,
            verification=verification,
            gossip_mode=gossip_mode,
            allow_write=allow_subject_branch_write,
        )
        graph_binding["subject_binding"] = subject_binding
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
        persisted = self._auto_persist_after_write()

        return {
            "node": {
                "id": node.id,
                "type": node.type,
                "attributes": dict(node.attributes),
                "state": dict(node.state),
            },
            "persisted": bool(persisted),
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
        persisted = self._auto_persist_after_write() if created else False
        return {
            "created": bool(created),
            "persisted": bool(persisted),
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
        persisted = self._auto_persist_after_write()
        return {
            "updated": True,
            "persisted": bool(persisted),
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
        persisted = self._auto_persist_after_write()
        return {
            "deleted": True,
            "persisted": bool(persisted),
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
        persisted = self._auto_persist_after_write()
        return {
            "updated": True,
            "persisted": bool(persisted),
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
        persisted = self._auto_persist_after_write() if removed > 0 else False
        return {
            "deleted": removed > 0,
            "removed": removed,
            "persisted": bool(persisted),
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
        persisted = self._auto_persist_after_write()
        return {
            "result": out,
            "persisted": bool(persisted),
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
        persisted = self._auto_persist_after_write() if changed else False
        return {
            "changed": bool(changed),
            "event_id": event_id,
            "reward": reward,
            "learning_rate": learning_rate,
            "persisted": bool(persisted),
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
        persisted = self._auto_persist_after_write() if updated > 0 else False
        return {
            "updated": int(updated),
            "relation_type": relation_type,
            "reward": reward,
            "learning_rate": learning_rate,
            "persisted": bool(persisted),
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
        payload = self.snapshot_payload()
        payload["persisted"] = bool(self._auto_persist_after_write())
        return payload

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

        persisted = self._auto_persist_after_write()
        return {
            "created_nodes_estimate": created_nodes,
            "created_edges_estimate": created_edges,
            "persisted": bool(persisted),
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

        input_extraction = self._as_mapping(profile_update.get("input_extraction"))
        graph_monitor = self._as_mapping(profile_update.get("graph_monitor"))
        if not input_extraction:
            related_node_ids = [
                self._to_int(graph_binding.get("person_node_id", 0), 0),
                self._to_int(graph_binding.get("entry_node_id", 0), 0),
            ]
            related_node_ids.extend(
                self._to_int(item, 0)
                for item in self._as_list(graph_binding.get("recommendation_node_ids"))
                if self._to_int(item, 0) > 0
            )
            input_extraction = self._capture_input_intelligence(
                user_id=user_id,
                session_id=session_id,
                source="daily_mode",
                text=text,
                context=f"display_name={display_name}; language={language}",
                related_node_ids=related_node_ids[:6],
                apply_to_graph=True,
            )
            graph_monitor = self._as_mapping(input_extraction.get("graph_monitor"))
        if isinstance(project_status, dict):
            project_status["graph"] = self.snapshot_payload()

        persisted = self._auto_persist_after_write()
        daily_summary = {
            "overall_score": round(self._to_float(scores.get("overall", 0.0), 0.0), 4),
            "recommendation_count": len(recommendations),
            "recommendation_titles": [
                self._to_title(str(item.get("title", "") or ""), fallback=f"Recommendation {idx + 1}", limit=64)
                for idx, item in enumerate(recommendations[:6])
                if isinstance(item, Mapping)
            ],
            "signal_counts": {
                "goals": len(goals),
                "problems": len(problems),
                "wins": len(wins),
            },
            "profile_updated": bool(profile_update),
            "profile_update_error": profile_update_error,
        }
        execution = self._execution_status(
            action="daily_mode",
            persisted=persisted,
            input_extraction=input_extraction,
            graph_monitor=graph_monitor,
            extra={
                "recommendation_count": len(recommendations),
                "overall_score": daily_summary["overall_score"],
            },
        )
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
            "input_extraction": input_extraction,
            "graph_monitor": graph_monitor,
            "summary": daily_summary,
            "execution": execution,
            "persisted": bool(persisted),
        }

    @staticmethod
    def _default_demo_narrative(persona_name: str, language: str = "ru") -> str:
        name = str(persona_name or "You").strip() or "You"
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
        persona_name = str(data.get("persona_name", "You") or "You").strip() or "You"
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
        input_extraction = self._capture_input_intelligence(
            user_id=f"demo_{self._safe_slug(persona_name, default='persona')}",
            session_id=f"demo_{self._safe_slug(persona_name, default='persona')}",
            source="demo_narrative",
            text=narrative,
            context=f"persona={persona_name}; language={language}",
            related_node_ids=[root_node_id] if root_node_id > 0 else [],
            apply_to_graph=True,
        )
        graph_monitor = self._as_mapping(input_extraction.get("graph_monitor"))
        persisted = self._auto_persist_after_write()
        summary = {
            "persona_name": persona_name,
            "mode": llm_mode,
            "root_node_id": int(root_node_id),
            "llm_error": llm_error,
            "narrative_length": len(narrative),
        }
        execution = self._execution_status(
            action="demo_watch",
            persisted=persisted,
            input_extraction=input_extraction,
            graph_monitor=graph_monitor,
            extra={
                "root_node_id": int(root_node_id),
                "narrative_length": len(narrative),
            },
        )
        return {
            "demo": {
                "persona_name": persona_name,
                "narrative": narrative,
                "mode": llm_mode,
                "llm_error": llm_error,
                "root_node_id": root_node_id,
            },
            "input_extraction": input_extraction,
            "graph_monitor": graph_monitor,
            "summary": summary,
            "execution": execution,
            "persisted": bool(persisted),
            **self.snapshot_payload(),
        }

    def seed_demo(self) -> dict[str, Any]:
        # Backward-compatible endpoint used by UI buttons and tests.
        return self.watch_demo({"persona_name": "You", "reset_graph": True, "use_llm": True})

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

        related_node_ids = [int(profile_node.id)]
        session_node_id = self._to_int(client_binding.get("session_node_id", 0), 0)
        if session_node_id > 0:
            related_node_ids.append(session_node_id)
        input_extraction = self._capture_input_intelligence(
            user_id=user_id,
            session_id=session_id or f"profile_{user_id}",
            source="user_graph_update",
            text=profile_text or text,
            context=f"display_name={display_name}; language={language}",
            related_node_ids=related_node_ids,
            apply_to_graph=True,
        )

        graph_monitor = self._as_mapping(input_extraction.get("graph_monitor"))
        persisted = self._auto_persist_after_write()
        user_graph_summary = {
            "user_id": user_id,
            "display_name": display_name,
            "profile_node_id": int(profile_node.id),
            "dimension_count": len(non_empty),
            "dimension_names": list(non_empty.keys()),
            "client_profile_captured": bool(client_profile),
            "personalization_applied": bool(personalization),
            "feedback_items": int(feedback_summary.get("total", 0) or 0),
        }
        execution = self._execution_status(
            action="user_graph_update",
            persisted=persisted,
            input_extraction=input_extraction,
            graph_monitor=graph_monitor,
            extra={
                "profile_node_id": int(profile_node.id),
                "dimension_count": len(non_empty),
            },
        )
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
            "input_extraction": input_extraction,
            "graph_monitor": graph_monitor,
            "summary": user_graph_summary,
            "execution": execution,
            "persisted": bool(persisted),
            **self.snapshot_payload(),
        }

    def _persist_graph_safe(self) -> bool:
        try:
            return bool(self.api.persist())
        except Exception:
            return False

    def _auto_persist_after_write(self) -> bool:
        if self.api.engine.graph_adapter is None:
            return False
        if not self._env_flag("AUTOGRAPH_AUTO_PERSIST_ON_WRITE", True):
            return False
        return self._persist_graph_safe()

    def _ensure_personal_tree_root(self, *, user_id: str):
        tree_key = f"{user_id}:{_PERSONAL_TREE_BRANCH_NAME}"
        root, _ = self._ensure_shared_node(
            node_type="personal_info_tree_root",
            identity_key="tree_key",
            identity_value=tree_key,
            attributes={
                "tree_key": tree_key,
                "user_id": user_id,
                "branch_name": _PERSONAL_TREE_BRANCH_NAME,
                "name": f"Personal Tree · {user_id}",
                "description": "Persistent personal information tree with notes, summaries and thought branches.",
            },
        )
        root.attributes["updated_at"] = time.time()
        return root

    def _extract_personal_tree_takeaways(
        self,
        *,
        text: str,
        source_type: str,
        max_points: int,
    ) -> dict[str, Any]:
        source = str(text or "").strip()
        rows = self._split_daily_sentences(source)
        if not rows:
            return {
                "summary": "",
                "points": [],
                "citations": [],
            }

        hints_by_type: dict[str, tuple[str, ...]] = {
            "law": ("law", "legal", "article", "статья", "закон", "հոդված", "իրավ"),
            "article": ("study", "research", "article", "paper", "analysis", "evidence"),
            "text": ("because", "therefore", "result", "impact", "plan", "risk"),
            "note": ("todo", "fix", "improve", "idea", "next"),
        }
        active_hints = hints_by_type.get(source_type, hints_by_type["text"])

        scored: list[tuple[float, int, str]] = []
        for idx, sentence in enumerate(rows):
            cleaned = " ".join(str(sentence or "").split()).strip()
            if not cleaned:
                continue
            score = min(1.0, len(cleaned) / 190.0)
            if idx < 2:
                score += 0.28
            if self._contains_hint(cleaned, active_hints):
                score += 0.32
            if _PERSONAL_TREE_CITATION_RE.search(cleaned):
                score += 0.42
            scored.append((score, idx, cleaned))

        if not scored:
            scored = [(0.5, idx, row) for idx, row in enumerate(rows)]
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

        points: list[str] = []
        for _, _, row in scored:
            if len(points) >= max(1, min(12, int(max_points))):
                break
            points.append(row)
        points = self._dedupe_limited(points, limit=max(1, min(12, int(max_points))))

        citations = self._dedupe_limited(
            [match.group(0).strip() for match in _PERSONAL_TREE_CITATION_RE.finditer(source)],
            limit=12,
        )

        summary_basis = points[:2] if points else rows[:2]
        summary = " ".join(summary_basis).strip()
        if len(summary) > 720:
            summary = f"{summary[:717].rstrip()}..."

        return {
            "summary": summary,
            "points": points,
            "citations": citations,
        }

    def _serialize_personal_tree(self, *, user_id: str, focus_node_id: int = 0, max_nodes: int = 160) -> dict[str, Any]:
        root = self._ensure_personal_tree_root(user_id=user_id)
        focus = self._to_int(focus_node_id, 0)
        limit = max(20, min(500, self._to_int(max_nodes, 160)))

        candidate_ids: set[int] = {int(root.id)}
        for node in self.api.engine.store.nodes.values():
            if str(node.type or "") not in _PERSONAL_TREE_NODE_TYPES:
                continue
            attrs = self._as_mapping(node.attributes)
            if str(attrs.get("user_id", "") or "").strip() == user_id:
                candidate_ids.add(int(node.id))

        eligible_edges = [
            edge
            for edge in self.api.engine.store.edges
            if str(edge.relation_type or "") in _PERSONAL_TREE_EDGE_TYPES
            and (int(edge.from_node) in candidate_ids or int(edge.to_node) in candidate_ids)
        ]
        for edge in eligible_edges:
            candidate_ids.add(int(edge.from_node))
            candidate_ids.add(int(edge.to_node))

        eligible_edges = [
            edge
            for edge in self.api.engine.store.edges
            if str(edge.relation_type or "") in _PERSONAL_TREE_EDGE_TYPES
            and int(edge.from_node) in candidate_ids
            and int(edge.to_node) in candidate_ids
        ]

        selected_ids: set[int]
        if focus > 0 and focus in candidate_ids:
            adjacency: dict[int, set[int]] = {}
            for edge in eligible_edges:
                left = int(edge.from_node)
                right = int(edge.to_node)
                adjacency.setdefault(left, set()).add(right)
                adjacency.setdefault(right, set()).add(left)
            queue: list[int] = [focus]
            visited: set[int] = set()
            while queue and len(visited) < limit:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                for nxt in adjacency.get(current, set()):
                    if nxt not in visited:
                        queue.append(nxt)
            visited.add(int(root.id))
            selected_ids = visited
        else:
            selected_ids = set(sorted(candidate_ids)[:limit])
            selected_ids.add(int(root.id))

        nodes_out: list[dict[str, Any]] = []
        for node_id in sorted(selected_ids):
            node = self.api.engine.get_node(node_id)
            if node is None:
                continue
            nodes_out.append(
                {
                    "id": int(node.id),
                    "type": str(node.type or "generic"),
                    "attributes": dict(node.attributes or {}),
                    "state": dict(node.state or {}),
                }
            )

        edges_out: list[dict[str, Any]] = []
        for edge in eligible_edges:
            from_id = int(edge.from_node)
            to_id = int(edge.to_node)
            if from_id not in selected_ids or to_id not in selected_ids:
                continue
            edges_out.append(
                {
                    "from": from_id,
                    "to": to_id,
                    "relation_type": str(edge.relation_type or ""),
                    "weight": float(edge.weight or 0.0),
                    "direction": str(edge.direction or "directed"),
                    "logic_rule": str(edge.logic_rule or ""),
                    "metadata": dict(edge.metadata or {}),
                }
            )

        sources: list[dict[str, Any]] = []
        notes: list[dict[str, Any]] = []
        for row in nodes_out:
            node_type = str(row.get("type", "") or "")
            attrs = self._as_mapping(row.get("attributes"))
            if node_type == "source_reference":
                sources.append(
                    {
                        "node_id": int(row.get("id", 0) or 0),
                        "title": str(attrs.get("title", "") or attrs.get("name", "") or "").strip(),
                        "url": str(attrs.get("url", "") or "").strip(),
                        "source_type": str(attrs.get("source_type", "") or "text").strip(),
                    }
                )
            if node_type == "personal_note_node":
                notes.append(
                    {
                        "node_id": int(row.get("id", 0) or 0),
                        "title": str(attrs.get("title", "") or attrs.get("name", "") or "").strip(),
                        "tags": self._to_list_of_strings(attrs.get("tags")),
                        "links": self._to_list_of_strings(attrs.get("links")),
                    }
                )

        return {
            "branch_name": _PERSONAL_TREE_BRANCH_NAME,
            "user_id": user_id,
            "root_node_id": int(root.id),
            "focus_node_id": int(focus if focus in selected_ids else root.id),
            "nodes": nodes_out,
            "edges": edges_out,
            "sources": sources,
            "notes": notes,
            "stats": {
                "node_count": len(nodes_out),
                "edge_count": len(edges_out),
                "source_count": len(sources),
                "note_count": len(notes),
            },
        }

    def project_personal_tree_note(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        user_id = str(payload.get("user_id", "default_user") or "default_user").strip() or "default_user"
        session_id = str(payload.get("session_id", "") or "").strip()
        title = " ".join(str(payload.get("title", "") or "").split()).strip()
        note_text = " ".join(str(payload.get("note", "") or payload.get("text", "") or "").split()).strip()
        if not title and note_text:
            title = self._to_title(note_text, fallback="Note", limit=86)
        if not title:
            raise ValueError("title or note is required")

        tags = self._dedupe_strings(self._to_list_of_strings(payload.get("tags")), limit=40)
        links = self._dedupe_strings(self._to_list_of_strings(payload.get("links")), limit=40)
        parent_node_id = self._to_int(payload.get("parent_node_id", 0), 0)
        note_id = self._to_int(payload.get("note_id", 0), 0)
        source_url = " ".join(str(payload.get("source_url", "") or "").split()).strip()
        source_title = " ".join(str(payload.get("source_title", "") or "").split()).strip()
        source_type = self._pick_allowed_token(
            payload.get("source_type"),
            allowed=_PERSONAL_TREE_SOURCE_TYPES_ALLOWED,
            default="note",
        )

        root = self._ensure_personal_tree_root(user_id=user_id)
        now = time.time()
        note_node = None
        if note_id > 0:
            existing = self.api.engine.get_node(note_id)
            if existing is not None and str(existing.type or "") == "personal_note_node":
                existing_owner = str(self._as_mapping(existing.attributes).get("user_id", "") or "").strip()
                if existing_owner and existing_owner != user_id:
                    raise ValueError("note belongs to another user_id")
                note_node = existing

        attrs = {
            "user_id": user_id,
            "title": title,
            "name": title,
            "note": note_text,
            "tags": tags,
            "links": links,
            "session_id": session_id,
            "updated_at": now,
        }
        if note_node is None:
            note_node = self.api.engine.create_node(
                "personal_note_node",
                attributes={
                    **attrs,
                    "created_at": now,
                },
                state={"weight": 0.66},
            )
            created = True
        else:
            note_node.attributes.update(attrs)
            created = False

        self._connect_nodes(
            from_node=int(root.id),
            to_node=int(note_node.id),
            relation_type="tree_note",
            weight=0.92,
            logic_rule="personal_tree_note",
            metadata={"session_id": session_id},
        )

        if parent_node_id > 0 and self.api.engine.get_node(parent_node_id) is not None:
            self._connect_nodes(
                from_node=int(parent_node_id),
                to_node=int(note_node.id),
                relation_type="note_child_of",
                weight=0.84,
                logic_rule="personal_tree_note",
            )

        source_node_id = 0
        if source_url or source_title:
            source_identity = source_url or source_title
            source_key = f"{user_id}:src:{zlib.crc32(source_identity.encode('utf-8')) & 0xFFFFFFFF:08x}"
            source_node, _ = self._ensure_shared_node(
                node_type="source_reference",
                identity_key="source_key",
                identity_value=source_key,
                attributes={
                    "source_key": source_key,
                    "user_id": user_id,
                    "name": source_title or source_url or "Source",
                    "title": source_title or source_url or "Source",
                    "url": source_url,
                    "source_type": source_type,
                    "updated_at": now,
                },
            )
            source_node.attributes["updated_at"] = now
            source_node_id = int(source_node.id)
            self._connect_nodes(
                from_node=int(note_node.id),
                to_node=int(source_node.id),
                relation_type="references_source",
                weight=0.82,
                logic_rule="personal_tree_note",
            )

        persisted = self._persist_graph_safe()
        tree = self._serialize_personal_tree(
            user_id=user_id,
            focus_node_id=int(note_node.id),
            max_nodes=self._to_int(payload.get("max_nodes", 180), 180),
        )

        self.api.engine._record_event(  # noqa: SLF001
            "personal_tree_note_saved",
            {
                "user_id": user_id,
                "session_id": session_id,
                "note_node_id": int(note_node.id),
                "created": bool(created),
                "source_node_id": int(source_node_id),
            },
        )

        return {
            "user_id": user_id,
            "session_id": session_id,
            "created": bool(created),
            "persisted": bool(persisted),
            "note": {
                "node_id": int(note_node.id),
                "title": title,
                "tags": tags,
                "links": links,
                "source_node_id": int(source_node_id),
            },
            "tree": tree,
            **self.snapshot_payload(),
        }

    def project_personal_tree_ingest(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        user_id = str(payload.get("user_id", "default_user") or "default_user").strip() or "default_user"
        session_id = str(payload.get("session_id", "") or "").strip()
        topic = " ".join(str(payload.get("topic", "") or "").split()).strip()
        title = " ".join(
            str(payload.get("title", "") or payload.get("source_title", "") or topic or "Thought Tree Session").split()
        ).strip()
        text = str(payload.get("text", "") or "").strip()
        if not text:
            raise ValueError("text is required")
        source_type = self._pick_allowed_token(
            payload.get("source_type"),
            allowed=_PERSONAL_TREE_SOURCE_TYPES_ALLOWED,
            default="text",
        )
        source_url = " ".join(str(payload.get("source_url", "") or "").split()).strip()
        source_title = " ".join(str(payload.get("source_title", "") or "").split()).strip()
        max_points = max(2, min(12, self._to_int(payload.get("max_points", 6), 6)))
        parent_node_id = self._to_int(payload.get("parent_node_id", 0), 0)

        extraction = self._extract_personal_tree_takeaways(
            text=text,
            source_type=source_type,
            max_points=max_points,
        )
        summary_text = str(extraction.get("summary", "") or "").strip()
        points = self._to_list_of_strings(extraction.get("points"))
        citations = self._to_list_of_strings(extraction.get("citations"))

        root = self._ensure_personal_tree_root(user_id=user_id)
        now = time.time()
        session_node = self.api.engine.create_node(
            "thought_tree_session",
            attributes={
                "user_id": user_id,
                "name": title,
                "title": title,
                "topic": topic,
                "source_type": source_type,
                "source_title": source_title,
                "source_url": source_url,
                "session_id": session_id,
                "created_at": now,
                "updated_at": now,
            },
            state={"confidence": 0.74},
        )
        self._connect_nodes(
            from_node=int(root.id),
            to_node=int(session_node.id),
            relation_type="tree_session",
            weight=0.92,
            logic_rule="personal_tree_ingest",
        )
        if parent_node_id > 0 and self.api.engine.get_node(parent_node_id) is not None:
            self._connect_nodes(
                from_node=int(parent_node_id),
                to_node=int(session_node.id),
                relation_type="about_topic",
                weight=0.84,
                logic_rule="personal_tree_ingest",
            )

        source_node_id = 0
        if source_url or source_title:
            source_identity = source_url or source_title
            source_key = f"{user_id}:src:{zlib.crc32(source_identity.encode('utf-8')) & 0xFFFFFFFF:08x}"
            source_node, _ = self._ensure_shared_node(
                node_type="source_reference",
                identity_key="source_key",
                identity_value=source_key,
                attributes={
                    "source_key": source_key,
                    "user_id": user_id,
                    "name": source_title or source_url or title,
                    "title": source_title or source_url or title,
                    "url": source_url,
                    "source_type": source_type,
                    "updated_at": now,
                },
            )
            source_node.attributes["updated_at"] = now
            source_node_id = int(source_node.id)
            self._connect_nodes(
                from_node=int(session_node.id),
                to_node=int(source_node.id),
                relation_type="based_on_source",
                weight=0.86,
                logic_rule="personal_tree_ingest",
            )

        summary_node = self.api.engine.create_node(
            "thought_summary_node",
            attributes={
                "user_id": user_id,
                "name": self._to_title(summary_text or title, fallback=title, limit=96),
                "summary": summary_text,
                "citations": citations,
                "source_type": source_type,
                "source_url": source_url,
                "created_at": now,
                "updated_at": now,
            },
            state={"confidence": 0.77},
        )
        self._connect_nodes(
            from_node=int(session_node.id),
            to_node=int(summary_node.id),
            relation_type="has_summary",
            weight=0.9,
            logic_rule="personal_tree_ingest",
        )

        point_node_ids: list[int] = []
        for idx, point in enumerate(points, start=1):
            point_node = self.api.engine.create_node(
                "thought_point_node",
                attributes={
                    "user_id": user_id,
                    "name": self._to_title(point, fallback=f"Point {idx}", limit=96),
                    "point": point,
                    "rank": idx,
                    "source_type": source_type,
                    "created_at": now,
                    "updated_at": now,
                },
                state={"confidence": max(0.45, 0.84 - idx * 0.06)},
            )
            point_node_ids.append(int(point_node.id))
            self._connect_nodes(
                from_node=int(summary_node.id),
                to_node=int(point_node.id),
                relation_type="supports_point",
                weight=max(0.42, 0.92 - idx * 0.08),
                logic_rule="personal_tree_ingest",
                metadata={"rank": idx},
            )

        persisted = self._persist_graph_safe()
        tree = self._serialize_personal_tree(
            user_id=user_id,
            focus_node_id=int(summary_node.id),
            max_nodes=self._to_int(payload.get("max_nodes", 200), 200),
        )

        self.api.engine._record_event(  # noqa: SLF001
            "personal_tree_ingested",
            {
                "user_id": user_id,
                "session_id": session_id,
                "source_type": source_type,
                "summary_node_id": int(summary_node.id),
                "points": len(point_node_ids),
                "citations": len(citations),
            },
        )

        related_node_ids = [int(summary_node.id), int(session_node.id)]
        if source_node_id > 0:
            related_node_ids.append(int(source_node_id))
        related_node_ids.extend(point_node_ids[:4])
        input_extraction = self._capture_input_intelligence(
            user_id=user_id,
            session_id=session_id or f"tree_{user_id}",
            source="personal_tree_ingest",
            text=text,
            context=f"title={title}; topic={topic}; source_type={source_type}",
            related_node_ids=related_node_ids,
            apply_to_graph=True,
        )
        graph_monitor = self._as_mapping(input_extraction.get("graph_monitor"))
        execution = self._execution_status(
            action="personal_tree_ingest",
            persisted=persisted,
            input_extraction=input_extraction,
            graph_monitor=graph_monitor,
            extra={
                "summary_node_id": int(summary_node.id),
                "point_count": len(point_node_ids),
            },
        )

        return {
            "user_id": user_id,
            "session_id": session_id,
            "persisted": bool(persisted),
            "ingest": {
                "title": title,
                "topic": topic,
                "source_type": source_type,
                "source_url": source_url,
                "source_title": source_title,
                "max_points": max_points,
            },
            "summary": {
                "summary_node_id": int(summary_node.id),
                "point_count": len(point_node_ids),
                "citation_count": len(citations),
                "source_type": source_type,
                "title": title,
                "topic": topic,
            },
            "extraction": {
                "summary": summary_text,
                "points": points,
                "citations": citations,
            },
            "semantic_binding": {
                "root_node_id": int(root.id),
                "session_node_id": int(session_node.id),
                "summary_node_id": int(summary_node.id),
                "point_node_ids": point_node_ids,
                "source_node_id": int(source_node_id),
            },
            "tree": tree,
            "input_extraction": input_extraction,
            "graph_monitor": graph_monitor,
            "execution": execution,
            **self.snapshot_payload(),
        }

    def project_personal_tree_view(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        root = self._as_mapping(payload)
        user_id = str(root.get("user_id", "default_user") or "default_user").strip() or "default_user"
        focus_node_id = self._to_int(root.get("focus_node_id", 0), 0)
        max_nodes = max(20, min(500, self._to_int(root.get("max_nodes", 180), 180)))
        tree = self._serialize_personal_tree(
            user_id=user_id,
            focus_node_id=focus_node_id,
            max_nodes=max_nodes,
        )
        return {
            "tree": tree,
            **self.snapshot_payload(),
        }

    def project_packages_manage(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        user_id = str(payload.get("user_id", "default_user") or "default_user").strip() or "default_user"
        session_id = str(payload.get("session_id", "") or "").strip()
        package_name = self._safe_slug(payload.get("package_name", "inbox"), default="inbox")
        action = self._pick_allowed_token(
            payload.get("action", "list"),
            allowed=_PACKAGE_ACTIONS_ALLOWED,
            default="list",
        )
        model_role = self._debate_role(payload.get("model_role"), fallback=_GARBAGE_MANAGER_DEFAULT_ROLE)
        model_path = str(payload.get("model_path", "") or "").strip()
        classify_with_llm = self._to_bool(payload.get("classify_with_llm", True))

        bucket_key = f"{user_id}:{package_name}"
        now = time.time()
        bucket_node, _ = self._ensure_shared_node(
            node_type="user_package_bucket",
            identity_key="bucket_key",
            identity_value=bucket_key,
            attributes={
                "bucket_key": bucket_key,
                "user_id": user_id,
                "name": package_name,
                "namespace": "personal",
                "updated_at": now,
            },
        )
        bucket_node.attributes["updated_at"] = now

        def _package_items() -> list[Any]:
            out: list[Any] = []
            for node in self.api.engine.store.nodes.values():
                if str(node.type or "") != "user_package_item":
                    continue
                attrs = self._as_mapping(node.attributes)
                if str(attrs.get("user_id", "") or "").strip() != user_id:
                    continue
                if str(attrs.get("package_name", "") or "").strip() != package_name:
                    continue
                out.append(node)
            out.sort(
                key=lambda row: float(self._as_mapping(row.attributes).get("created_at", 0.0) or 0.0),
                reverse=True,
            )
            return out

        llm_fn: Callable[[str], str] | None = None
        manager_resolution: dict[str, Any] = {
            "mode": "disabled",
            "requested_model_path": model_path,
            "requested_role": model_role,
            "selected_model_path": "",
        }
        if classify_with_llm and action == "store":
            llm_fn, manager_resolution = self._resolve_role_or_model_llm(
                role=model_role,
                model_path=model_path,
            )

        created_items: list[dict[str, Any]] = []
        changed_item_ids: list[int] = []
        if action == "store":
            raw_items = payload.get("items", payload.get("entries", []))
            if isinstance(raw_items, str):
                raw_items = [raw_items]
            if not isinstance(raw_items, list):
                raw_items = []
            if not raw_items:
                fallback_item = " ".join(
                    str(payload.get("item", "") or payload.get("text", "") or payload.get("content", "") or "").split()
                ).strip()
                if fallback_item:
                    raw_items = [fallback_item]
            if not raw_items:
                raise ValueError("items or item is required for store action")

            for idx, raw in enumerate(raw_items):
                content = " ".join(str(raw or "").split()).strip()
                if not content:
                    continue
                trash_score = 0.25
                classification_reason = "heuristic_default"
                if any(hint in content.lower() for hint in ("tmp", "todo", "old", "junk", "trash", "noise")):
                    trash_score = 0.76
                    classification_reason = "heuristic_keyword"
                llm_output = ""
                if llm_fn is not None:
                    prompt = (
                        "Classify package item for personal memory hygiene.\n"
                        "Return JSON only:\n"
                        '{ "trash_score":0.0, "status":"active|candidate_trash", "label":"", "reason":"" }\n'
                        "trash_score in [0,1].\n"
                        f"Item: {content}\n"
                    )
                    try:
                        llm_output = str(llm_fn(prompt) or "").strip()
                    except Exception:
                        llm_output = ""
                    parsed = self._extract_json_from_llm_output(llm_output)
                    parsed_row = self._as_mapping(parsed)
                    if parsed_row:
                        trash_score = self._confidence(parsed_row.get("trash_score", trash_score), trash_score)
                        classification_reason = "llm_classification"

                namespace = "trash" if trash_score >= 0.67 else "personal"
                status = "candidate_trash" if trash_score >= 0.67 else "active"
                item_node = self.api.engine.create_node(
                    "user_package_item",
                    attributes={
                        "user_id": user_id,
                        "session_id": session_id,
                        "package_name": package_name,
                        "name": self._to_title(content, fallback=f"Item {idx + 1}", limit=80),
                        "content": content,
                        "namespace": namespace,
                        "status": status,
                        "trash_score": round(trash_score, 4),
                        "classification_reason": classification_reason,
                        "created_at": now,
                        "updated_at": now,
                    },
                    state={"confidence": max(0.2, 1.0 - trash_score)},
                )
                changed_item_ids.append(int(item_node.id))
                self._connect_nodes(
                    from_node=int(bucket_node.id),
                    to_node=int(item_node.id),
                    relation_type="contains_item",
                    weight=max(0.2, 1.0 - trash_score * 0.5),
                    logic_rule="packages_manage_store",
                )
                created_items.append(
                    {
                        "node_id": int(item_node.id),
                        "content": content,
                        "namespace": namespace,
                        "status": status,
                        "trash_score": round(trash_score, 4),
                    }
                )

        elif action in {"purge", "restore"}:
            apply_changes = self._to_bool(payload.get("apply_changes", False))
            confirmation = payload.get("confirmation", payload.get("security_decision", ""))
            policy_decision = self._llm_policy_decision(
                user_id=user_id,
                session_id=session_id,
                action=f"packages_{action}",
                requested_apply=apply_changes,
                confirmation=confirmation,
            )
            target_ids = {
                self._to_int(item, 0)
                for item in self._as_list(payload.get("item_node_ids", payload.get("node_ids", [])))
                if self._to_int(item, 0) > 0
            }
            for node in _package_items():
                attrs = self._as_mapping(node.attributes)
                node_id = int(node.id)
                if target_ids and node_id not in target_ids:
                    continue
                if action == "purge":
                    if not target_ids and str(attrs.get("status", "active")) != "candidate_trash":
                        continue
                    if policy_decision.get("apply_allowed", False):
                        node.attributes["status"] = "purged"
                        node.attributes["namespace"] = "trash"
                        node.attributes["updated_at"] = now
                        node.attributes["purged_at"] = now
                    changed_item_ids.append(node_id)
                else:  # restore
                    if policy_decision.get("apply_allowed", False):
                        node.attributes["status"] = "active"
                        node.attributes["namespace"] = "personal"
                        node.attributes["updated_at"] = now
                    changed_item_ids.append(node_id)
        else:
            # list action
            pass

        items_payload: list[dict[str, Any]] = []
        namespace_counts: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()
        for node in _package_items():
            attrs = self._as_mapping(node.attributes)
            namespace = str(attrs.get("namespace", "personal") or "personal").strip() or "personal"
            status = str(attrs.get("status", "active") or "active").strip() or "active"
            namespace_counts[namespace] += 1
            status_counts[status] += 1
            items_payload.append(
                {
                    "node_id": int(node.id),
                    "name": str(attrs.get("name", "") or ""),
                    "content": str(attrs.get("content", "") or ""),
                    "namespace": namespace,
                    "status": status,
                    "trash_score": self._confidence(attrs.get("trash_score", 0.0), 0.0),
                    "updated_at": float(attrs.get("updated_at", 0.0) or 0.0),
                }
            )

        persisted = False
        if action in {"store", "purge", "restore"} and (created_items or changed_item_ids):
            persisted = self._persist_graph_safe()

        self.api.engine._record_event(  # noqa: SLF001
            "project_package_action",
            {
                "action": action,
                "user_id": user_id,
                "session_id": session_id,
                "package_name": package_name,
                "changed_items": len(changed_item_ids),
                "created_items": len(created_items),
            },
        )

        return {
            "action": action,
            "user_id": user_id,
            "session_id": session_id,
            "package": {
                "node_id": int(bucket_node.id),
                "name": package_name,
            },
            "manager": {
                "model_role": model_role,
                "model_path": model_path,
                "resolution": manager_resolution,
            },
            "created_items": created_items,
            "changed_item_ids": changed_item_ids,
            "items": items_payload,
            "stats": {
                "item_count": len(items_payload),
                "namespace_counts": dict(namespace_counts),
                "status_counts": dict(status_counts),
            },
            "persisted": bool(persisted),
            **self.snapshot_payload(),
        }

    def project_memory_namespace_apply(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        user_id = str(payload.get("user_id", "default_user") or "default_user").strip() or "default_user"
        session_id = str(payload.get("session_id", "") or "").strip()
        namespace = self._normalize_namespace(payload.get("namespace"), default="personal")
        scope = self._pick_allowed_token(payload.get("scope", "owned"), allowed=_MEMORY_SCOPE_ALLOWED, default="owned")
        query = " ".join(str(payload.get("query", "") or payload.get("text", "") or "").split()).strip()
        query_tokens = self._token_set(query)
        min_score = max(0.0, min(1.0, self._to_float(payload.get("min_score", 0.2), 0.2)))
        apply_changes = self._to_bool(payload.get("apply_changes", True))
        confirmation = payload.get("confirmation", payload.get("security_decision", ""))
        policy_decision = self._llm_policy_decision(
            user_id=user_id,
            session_id=session_id,
            action="memory_namespace_apply",
            requested_apply=apply_changes,
            confirmation=confirmation,
        )

        node_ids = {
            self._to_int(item, 0)
            for item in self._as_list(payload.get("node_ids", []))
            if self._to_int(item, 0) > 0
        }
        candidates = self._iter_user_nodes(
            user_id=user_id,
            namespace=str(payload.get("source_namespace", "") or ""),
            scope=scope,
        )
        affected: list[dict[str, Any]] = []
        for node in candidates:
            node_id = int(node.id)
            if node_ids and node_id not in node_ids:
                continue
            if query_tokens:
                score = self._jaccard_similarity(query_tokens, self._token_set(self._node_text_blob(node)))
                if score < min_score:
                    continue
            attrs = self._as_mapping(node.attributes)
            old_namespace = self._normalize_namespace(attrs.get("namespace", "global"), default="global")
            if policy_decision.get("apply_allowed", False):
                node.attributes["namespace"] = namespace
                node.attributes["namespace_updated_at"] = time.time()
                node.attributes.setdefault("user_id", user_id)
            affected.append(
                {
                    "node_id": node_id,
                    "node_type": str(node.type or "generic"),
                    "name": str(attrs.get("name", "") or attrs.get("title", "") or "").strip(),
                    "old_namespace": old_namespace,
                    "new_namespace": namespace,
                }
            )

        persisted = False
        if affected and policy_decision.get("apply_allowed", False):
            persisted = self._persist_graph_safe()

        self.api.engine._record_event(  # noqa: SLF001
            "memory_namespace_applied",
            {
                "user_id": user_id,
                "session_id": session_id,
                "namespace": namespace,
                "scope": scope,
                "affected": len(affected),
                "apply_allowed": bool(policy_decision.get("apply_allowed", False)),
            },
        )

        return {
            "user_id": user_id,
            "session_id": session_id,
            "namespace": namespace,
            "scope": scope,
            "query": query,
            "affected": affected,
            "affected_count": len(affected),
            "policy": policy_decision,
            "persisted": bool(persisted),
            **self.snapshot_payload(),
        }

    def project_memory_namespace_view(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        root = self._as_mapping(payload)
        user_id = str(root.get("user_id", "") or "").strip()
        scope = self._pick_allowed_token(root.get("scope", "all"), allowed=_MEMORY_SCOPE_ALLOWED, default="all")
        nodes = self._iter_user_nodes(user_id=user_id, namespace="", scope=scope)
        max_nodes = max(10, min(800, self._to_int(root.get("max_nodes", 220), 220)))
        nodes = nodes[:max_nodes]
        counts: Counter[str] = Counter()
        rows: list[dict[str, Any]] = []
        for node in nodes:
            attrs = self._as_mapping(node.attributes)
            namespace = self._normalize_namespace(attrs.get("namespace", "global"), default="global")
            counts[namespace] += 1
            rows.append(
                {
                    "node_id": int(node.id),
                    "type": str(node.type or "generic"),
                    "namespace": namespace,
                    "name": str(attrs.get("name", "") or attrs.get("title", "") or "").strip(),
                    "user_id": str(attrs.get("user_id", "") or "").strip(),
                }
            )
        rows.sort(key=lambda item: (item["namespace"], item["node_id"]))
        return {
            "user_id": user_id,
            "scope": scope,
            "namespace_counts": dict(counts),
            "nodes": rows,
            "total_nodes": len(rows),
            **self.snapshot_payload(),
        }

    def project_graph_rag_query(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        query = " ".join(str(payload.get("query", "") or payload.get("text", "") or "").split()).strip()
        if not query:
            raise ValueError("query is required")
        user_id = str(payload.get("user_id", "") or "").strip()
        scope = self._pick_allowed_token(payload.get("scope", "all"), allowed=_MEMORY_SCOPE_ALLOWED, default="all")
        namespace = str(payload.get("namespace", "") or "").strip()
        top_k = max(1, min(20, self._to_int(payload.get("top_k", 6), 6)))
        use_llm = self._to_bool(payload.get("use_llm", True))
        model_role = self._debate_role(payload.get("model_role"), fallback="analyst")
        model_path = str(payload.get("model_path", "") or "").strip()

        nodes = self._iter_user_nodes(user_id=user_id, namespace=namespace, scope=scope)
        if not nodes:
            nodes = list(self.api.engine.store.nodes.values())

        node_ids = {int(node.id) for node in nodes}
        degree_map = self._graph_degree_map(node_ids=node_ids)
        query_tokens = self._token_set(query)
        scored: list[dict[str, Any]] = []
        for node in nodes:
            attrs = self._as_mapping(node.attributes)
            text_blob = self._node_text_blob(node)
            if not text_blob:
                continue
            overlap = self._jaccard_similarity(query_tokens, self._token_set(text_blob))
            if overlap <= 0.0:
                continue
            degree_boost = min(1.0, float(degree_map.get(int(node.id), 0)) / 10.0)
            score = (0.82 * overlap) + (0.18 * degree_boost)
            if score < 0.12:
                continue
            scored.append(
                {
                    "node_id": int(node.id),
                    "type": str(node.type or "generic"),
                    "name": str(attrs.get("name", "") or attrs.get("title", "") or "").strip(),
                    "summary": str(attrs.get("summary", "") or attrs.get("description", "") or "").strip(),
                    "namespace": self._node_namespace(node, default="global"),
                    "score": round(max(0.0, min(1.0, score)), 4),
                }
            )

        scored.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        hits = scored[:top_k]
        context_lines: list[str] = []
        for idx, hit in enumerate(hits, start=1):
            title = str(hit.get("name", "") or f"Node {hit.get('node_id', 0)}")
            summary = str(hit.get("summary", "") or "").strip()
            if len(summary) > 240:
                summary = f"{summary[:237].rstrip()}..."
            context_lines.append(f"[{idx}] {title} (node:{hit.get('node_id')}, score={hit.get('score')})")
            if summary:
                context_lines.append(f"    {summary}")

        llm_fn: Callable[[str], str] | None = None
        resolution: dict[str, Any] = {
            "mode": "disabled",
            "requested_role": model_role,
            "requested_model_path": model_path,
            "selected_model_path": "",
        }
        answer = ""
        raw_output = ""
        if use_llm:
            llm_fn, resolution = self._resolve_role_or_model_llm(role=model_role, model_path=model_path)
        if llm_fn is not None and context_lines:
            prompt = (
                "You are a graph-RAG assistant.\n"
                "Use only provided context nodes and mention uncertainty when context is insufficient.\n"
                f"Query: {query}\n"
                "Context:\n"
                f"{chr(10).join(context_lines)}\n"
                "Return JSON only:\n"
                '{ "answer":"", "reasoning_summary":"", "citations":[{"node_id":0,"why":""}] }\n'
            )
            try:
                raw_output = str(llm_fn(prompt) or "").strip()
            except Exception:
                raw_output = ""
            parsed = self._extract_json_from_llm_output(raw_output)
            parsed_row = self._as_mapping(parsed)
            if parsed_row:
                answer = " ".join(str(parsed_row.get("answer", "") or "").split()).strip()
        if not answer:
            if hits:
                answer = (
                    "Top graph evidence suggests focusing on: "
                    + ", ".join(str(item.get("name", "") or item.get("node_id", 0)) for item in hits[:3])
                )
            else:
                answer = "No strong matching nodes found for this query."

        self.api.engine._record_event(  # noqa: SLF001
            "graph_rag_query",
            {
                "user_id": user_id,
                "scope": scope,
                "namespace": namespace,
                "hits": len(hits),
                "model_role": model_role,
            },
        )

        return {
            "query": query,
            "user_id": user_id,
            "scope": scope,
            "namespace": namespace,
            "answer": answer,
            "hits": hits,
            "model": {
                "role": model_role,
                "path": model_path,
                "resolution": resolution,
                "used": llm_fn is not None,
            },
            "raw_output": raw_output[:6000],
            **self.snapshot_payload(),
        }

    def project_contradiction_scan(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        user_id = str(payload.get("user_id", "") or "").strip()
        scope = self._pick_allowed_token(payload.get("scope", "all"), allowed=_MEMORY_SCOPE_ALLOWED, default="all")
        namespace = str(payload.get("namespace", "") or "").strip()
        max_nodes = max(10, min(240, self._to_int(payload.get("max_nodes", 120), 120)))
        top_k = max(1, min(120, self._to_int(payload.get("top_k", 20), 20)))
        min_overlap = max(0.1, min(0.95, self._to_float(payload.get("min_overlap", 0.32), 0.32)))
        apply_to_graph = self._to_bool(payload.get("apply_to_graph", False))
        session_id = str(payload.get("session_id", "") or "").strip()
        confirmation = payload.get("confirmation", payload.get("security_decision", ""))
        policy_decision = self._llm_policy_decision(
            user_id=user_id,
            session_id=session_id,
            action="contradiction_scan_apply",
            requested_apply=apply_to_graph,
            confirmation=confirmation,
        )

        nodes = self._iter_user_nodes(user_id=user_id, namespace=namespace, scope=scope)[:max_nodes]
        prepared: list[tuple[Any, str, set[str], bool]] = []
        for node in nodes:
            text = self._node_text_blob(node)
            tokens = self._token_set(text)
            if not text or not tokens:
                continue
            prepared.append((node, text, tokens, self._has_negation_hint(text)))

        issues: list[dict[str, Any]] = []
        for idx in range(len(prepared)):
            left_node, left_text, left_tokens, left_neg = prepared[idx]
            for jdx in range(idx + 1, len(prepared)):
                right_node, right_text, right_tokens, right_neg = prepared[jdx]
                overlap = self._jaccard_similarity(left_tokens, right_tokens)
                if overlap < min_overlap:
                    continue
                left_lower = f" {left_text.lower()} "
                right_lower = f" {right_text.lower()} "
                explicit_opposite = (
                    (" must " in left_lower and " must not " in right_lower)
                    or (" must not " in left_lower and " must " in right_lower)
                    or (" should " in left_lower and " should not " in right_lower)
                    or (" should not " in left_lower and " should " in right_lower)
                )
                negation_conflict = left_neg != right_neg
                if not explicit_opposite and not negation_conflict:
                    continue
                score = overlap + (0.28 if explicit_opposite else 0.0) + (0.16 if negation_conflict else 0.0)
                issues.append(
                    {
                        "left_node_id": int(left_node.id),
                        "right_node_id": int(right_node.id),
                        "left_preview": self._to_title(left_text, fallback=f"Node {left_node.id}", limit=120),
                        "right_preview": self._to_title(right_text, fallback=f"Node {right_node.id}", limit=120),
                        "overlap": round(overlap, 4),
                        "negation_conflict": bool(negation_conflict),
                        "explicit_opposite": bool(explicit_opposite),
                        "score": round(max(0.0, min(1.0, score)), 4),
                        "suggestion": (
                            "Split into context-specific branches or add explicit condition/time window."
                            if explicit_opposite
                            else "Add qualifier to clarify when each statement is true."
                        ),
                    }
                )
        issues.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        issues = issues[:top_k]

        binding: dict[str, Any] = {
            "attached": False,
            "scan_node_id": 0,
            "issue_node_ids": [],
        }
        if issues and policy_decision.get("apply_allowed", False):
            scan_node = self.api.engine.create_node(
                "contradiction_scan",
                attributes={
                    "user_id": user_id,
                    "session_id": session_id,
                    "name": f"Contradiction Scan ({len(issues)} issues)",
                    "namespace": namespace or "global",
                    "created_at": time.time(),
                },
                state={"confidence": max(0.2, 1.0 - (len(issues) / max(1, len(prepared))))},
            )
            binding["attached"] = True
            binding["scan_node_id"] = int(scan_node.id)
            issue_node_ids: list[int] = []
            for row in issues:
                issue_node = self.api.engine.create_node(
                    "contradiction_issue",
                    attributes={
                        "user_id": user_id,
                        "left_node_id": int(row.get("left_node_id", 0)),
                        "right_node_id": int(row.get("right_node_id", 0)),
                        "suggestion": str(row.get("suggestion", "") or ""),
                        "score": self._confidence(row.get("score", 0.5), 0.5),
                        "created_at": time.time(),
                    },
                    state={"risk": self._confidence(row.get("score", 0.5), 0.5)},
                )
                issue_node_ids.append(int(issue_node.id))
                self._connect_nodes(
                    from_node=int(scan_node.id),
                    to_node=int(issue_node.id),
                    relation_type="tracks_issue",
                    weight=self._confidence(row.get("score", 0.5), 0.5),
                    logic_rule="contradiction_scan",
                )
            binding["issue_node_ids"] = issue_node_ids
            self._persist_graph_safe()

        self.api.engine._record_event(  # noqa: SLF001
            "contradiction_scan_completed",
            {
                "user_id": user_id,
                "scope": scope,
                "namespace": namespace,
                "issues": len(issues),
                "attached": bool(binding.get("attached", False)),
            },
        )

        return {
            "user_id": user_id,
            "scope": scope,
            "namespace": namespace,
            "prepared_nodes": len(prepared),
            "issues": issues,
            "issue_count": len(issues),
            "graph_binding": binding,
            "policy": policy_decision,
            **self.snapshot_payload(),
        }

    def project_task_risk_board(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        user_id = str(payload.get("user_id", "default_user") or "default_user").strip() or "default_user"
        session_id = str(payload.get("session_id", "") or "").strip()
        tasks_raw = payload.get("tasks", [])
        if isinstance(tasks_raw, str):
            tasks_raw = [{"title": token.strip()} for token in re.split(r"[;\n]+", tasks_raw) if token.strip()]
        tasks_list = self._as_list(tasks_raw)
        if not tasks_list:
            source_text = " ".join(str(payload.get("text", "") or "").split()).strip()
            if source_text:
                tasks_list = [{"title": item} for item in self._split_daily_sentences(source_text)[:12]]
        if not tasks_list:
            raise ValueError("tasks or text is required")

        apply_to_graph = self._to_bool(payload.get("apply_to_graph", True))
        confirmation = payload.get("confirmation", payload.get("security_decision", ""))
        policy_decision = self._llm_policy_decision(
            user_id=user_id,
            session_id=session_id,
            action="task_risk_apply",
            requested_apply=apply_to_graph,
            confirmation=confirmation,
        )

        board_rows: list[dict[str, Any]] = []
        for idx, item in enumerate(tasks_list[:50], start=1):
            row = self._as_mapping(item)
            title = " ".join(str(row.get("title", "") or row.get("task", "") or "").split()).strip()
            details = " ".join(str(row.get("description", "") or row.get("details", "") or "").split()).strip()
            if not title:
                continue
            base_risk = self._confidence(row.get("risk", row.get("risk_score", 0.35)), 0.35)
            text_blob = f"{title} {details}".lower()
            if any(hint in text_blob for hint in ("legal", "law", "deadline", "security", "payment", "prod", "risk")):
                base_risk = min(1.0, base_risk + 0.22)
            if any(hint in text_blob for hint in ("research", "draft", "idea", "optional")):
                base_risk = max(0.0, base_risk - 0.14)
            level = "low"
            if base_risk >= 0.8:
                level = "critical"
            elif base_risk >= 0.62:
                level = "high"
            elif base_risk >= 0.38:
                level = "medium"
            if level not in _TASK_RISK_LEVELS:
                level = "medium"
            board_rows.append(
                {
                    "index": idx,
                    "title": title,
                    "details": details,
                    "risk_score": round(base_risk, 4),
                    "risk_level": level,
                    "action": (
                        "Add mitigation plan and checkpoint"
                        if level in {"high", "critical"}
                        else "Proceed with normal monitoring"
                    ),
                }
            )
        board_rows.sort(key=lambda row: float(row.get("risk_score", 0.0)), reverse=True)

        graph_binding: dict[str, Any] = {
            "attached": False,
            "board_node_id": 0,
            "task_node_ids": [],
            "risk_node_ids": [],
        }
        if board_rows and policy_decision.get("apply_allowed", False):
            board_node = self.api.engine.create_node(
                "task_risk_board",
                attributes={
                    "user_id": user_id,
                    "session_id": session_id,
                    "name": f"Task Risk Board ({len(board_rows)} tasks)",
                    "created_at": time.time(),
                },
                state={"confidence": 0.76},
            )
            graph_binding["attached"] = True
            graph_binding["board_node_id"] = int(board_node.id)
            task_node_ids: list[int] = []
            risk_node_ids: list[int] = []
            for row in board_rows:
                task_node = self.api.engine.create_node(
                    "task_item",
                    attributes={
                        "user_id": user_id,
                        "title": str(row.get("title", "") or ""),
                        "details": str(row.get("details", "") or ""),
                        "risk_level": str(row.get("risk_level", "medium") or "medium"),
                        "risk_score": self._confidence(row.get("risk_score", 0.5), 0.5),
                        "created_at": time.time(),
                    },
                    state={"risk": self._confidence(row.get("risk_score", 0.5), 0.5)},
                )
                task_node_ids.append(int(task_node.id))
                self._connect_nodes(
                    from_node=int(board_node.id),
                    to_node=int(task_node.id),
                    relation_type="tracks_task",
                    weight=max(0.2, 1.0 - self._confidence(row.get("risk_score", 0.5), 0.5)),
                    logic_rule="task_risk_board",
                )
                risk_node = self.api.engine.create_node(
                    "task_risk",
                    attributes={
                        "user_id": user_id,
                        "title": f"Risk · {row.get('title', '')}",
                        "risk_level": str(row.get("risk_level", "medium") or "medium"),
                        "risk_score": self._confidence(row.get("risk_score", 0.5), 0.5),
                        "mitigation": str(row.get("action", "") or ""),
                    },
                    state={"risk": self._confidence(row.get("risk_score", 0.5), 0.5)},
                )
                risk_node_ids.append(int(risk_node.id))
                self._connect_nodes(
                    from_node=int(task_node.id),
                    to_node=int(risk_node.id),
                    relation_type="has_risk",
                    weight=self._confidence(row.get("risk_score", 0.5), 0.5),
                    logic_rule="task_risk_board",
                )
            graph_binding["task_node_ids"] = task_node_ids
            graph_binding["risk_node_ids"] = risk_node_ids
            self._persist_graph_safe()

        level_counts: Counter[str] = Counter(str(row.get("risk_level", "low")) for row in board_rows)
        self.api.engine._record_event(  # noqa: SLF001
            "task_risk_board_generated",
            {
                "user_id": user_id,
                "session_id": session_id,
                "tasks": len(board_rows),
                "attached": bool(graph_binding.get("attached", False)),
                "level_counts": dict(level_counts),
            },
        )

        return {
            "user_id": user_id,
            "session_id": session_id,
            "tasks": board_rows,
            "task_count": len(board_rows),
            "risk_level_counts": dict(level_counts),
            "graph_binding": graph_binding,
            "policy": policy_decision,
            **self.snapshot_payload(),
        }

    def project_timeline_replay(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        root = self._as_mapping(payload)
        user_id = str(root.get("user_id", "") or "").strip()
        session_id = str(root.get("session_id", "") or "").strip()
        event_type = str(root.get("event_type", "") or "").strip()
        limit = max(1, min(5000, self._to_int(root.get("limit", 600), 600)))
        from_ts = self._to_float(root.get("from_ts", 0.0), 0.0)
        to_ts = self._to_float(root.get("to_ts", 0.0), 0.0)

        events = self.list_events(limit=limit, event_type=event_type)
        filtered: list[dict[str, Any]] = []
        for event in events:
            timestamp = self._to_float(event.get("timestamp", 0.0), 0.0)
            if from_ts > 0.0 and timestamp < from_ts:
                continue
            if to_ts > 0.0 and timestamp > to_ts:
                continue
            payload_map = self._as_mapping(event.get("payload"))
            if user_id:
                owner = str(payload_map.get("user_id", "") or "").strip()
                if owner != user_id:
                    continue
            if session_id:
                sid = str(payload_map.get("session_id", "") or "").strip()
                if sid != session_id:
                    continue
            filtered.append(event)

        event_counts: Counter[str] = Counter(str(item.get("event_type", "")) for item in filtered)
        timeline: list[dict[str, Any]] = []
        for idx, event in enumerate(filtered, start=1):
            payload_map = self._as_mapping(event.get("payload"))
            summary = ""
            for key in ("name", "title", "topic", "action", "reason", "relation_type", "phase"):
                token = " ".join(str(payload_map.get(key, "") or "").split()).strip()
                if token:
                    summary = token
                    break
            if not summary:
                summary = f"event {event.get('event_type', '')}"
            timeline.append(
                {
                    "index": idx,
                    "id": int(event.get("id", 0) or 0),
                    "event_type": str(event.get("event_type", "") or ""),
                    "timestamp": self._to_float(event.get("timestamp", 0.0), 0.0),
                    "summary": summary,
                    "payload": payload_map,
                }
            )

        self.api.engine._record_event(  # noqa: SLF001
            "timeline_replay_requested",
            {
                "user_id": user_id,
                "session_id": session_id,
                "event_type": event_type,
                "returned_events": len(timeline),
            },
        )

        return {
            "filters": {
                "user_id": user_id,
                "session_id": session_id,
                "event_type": event_type,
                "limit": limit,
                "from_ts": from_ts,
                "to_ts": to_ts,
            },
            "event_counts": dict(event_counts),
            "timeline": timeline,
            "total_events": len(timeline),
        }

    def project_llm_policy_get(self, _payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "policy": self._llm_policy_snapshot(),
            "modes_allowed": list(_LLM_POLICY_MODES_ALLOWED),
        }

    def project_llm_policy_update(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        root = self._as_mapping(payload)
        merge_lists = self._to_bool(root.get("merge_lists", True))
        mode = self._pick_allowed_token(
            root.get("mode", self._llm_policy.get("mode", "confirm_required")),
            allowed=_LLM_POLICY_MODES_ALLOWED,
            default="confirm_required",
        )

        def _next_list(key: str) -> list[str]:
            incoming = self._dedupe_strings(self._to_list_of_strings(root.get(key)))
            if not merge_lists:
                return incoming
            current = self._dedupe_strings(self._to_list_of_strings(self._llm_policy.get(key)))
            return self._dedupe_strings(current + incoming)

        self._llm_policy["mode"] = mode
        if "trusted_sessions" in root:
            self._llm_policy["trusted_sessions"] = _next_list("trusted_sessions")
        if "trusted_users" in root:
            self._llm_policy["trusted_users"] = _next_list("trusted_users")
        if "allow_apply_for_actions" in root:
            self._llm_policy["allow_apply_for_actions"] = _next_list("allow_apply_for_actions")
        self._llm_policy["updated_at"] = time.time()

        self.api.engine._record_event(  # noqa: SLF001
            "llm_policy_updated",
            {
                "mode": mode,
                "trusted_sessions": len(self._to_list_of_strings(self._llm_policy.get("trusted_sessions"))),
                "trusted_users": len(self._to_list_of_strings(self._llm_policy.get("trusted_users"))),
                "allow_actions": len(self._to_list_of_strings(self._llm_policy.get("allow_apply_for_actions"))),
            },
        )
        return {
            "ok": True,
            "policy": self._llm_policy_snapshot(),
            "modes_allowed": list(_LLM_POLICY_MODES_ALLOWED),
        }

    def project_quality_harness(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        root = self._as_mapping(payload)
        user_id = str(root.get("user_id", "") or "").strip()
        sample_queries = self._to_list_of_strings(root.get("sample_queries"))[:12]
        if not sample_queries:
            sample_queries = [
                "what are my top priorities",
                "show risky contradictions",
                "what to improve next",
            ]

        snapshot = self.snapshot_payload()
        nodes = self._as_list(snapshot.get("snapshot", {}).get("nodes"))
        edges = self._as_list(snapshot.get("snapshot", {}).get("edges"))
        node_ids = {self._to_int(row.get("id", 0), 0) for row in nodes}
        dangling_edges = 0
        for edge in edges:
            row = self._as_mapping(edge)
            left = self._to_int(row.get("from", 0), 0)
            right = self._to_int(row.get("to", 0), 0)
            if left not in node_ids or right not in node_ids:
                dangling_edges += 1

        namespace_counts: Counter[str] = Counter()
        for row in nodes:
            attrs = self._as_mapping(self._as_mapping(row).get("attributes"))
            namespace_counts[self._normalize_namespace(attrs.get("namespace", "global"), default="global")] += 1

        contradiction_probe = self.project_contradiction_scan(
            {
                "user_id": user_id,
                "scope": "owned" if user_id else "all",
                "max_nodes": min(120, len(nodes)),
                "top_k": 20,
                "apply_to_graph": False,
            }
        )
        contradiction_count = int(contradiction_probe.get("issue_count", 0) or 0)

        rag_results: list[dict[str, Any]] = []
        rag_hits = 0
        for query in sample_queries:
            rag = self.project_graph_rag_query(
                {
                    "query": query,
                    "user_id": user_id,
                    "scope": "owned" if user_id else "all",
                    "top_k": 4,
                    "use_llm": False,
                }
            )
            hit_count = len(self._as_list(rag.get("hits")))
            if hit_count > 0:
                rag_hits += 1
            rag_results.append(
                {
                    "query": query,
                    "hits": hit_count,
                }
            )

        score = 100.0
        score -= min(30.0, dangling_edges * 3.0)
        score -= min(24.0, contradiction_count * 1.2)
        coverage = 0.0 if not sample_queries else float(rag_hits) / float(len(sample_queries))
        score += 10.0 * coverage
        if len(nodes) > 10000:
            score -= 8.0
        score = max(0.0, min(100.0, score))

        recommendations: list[str] = []
        if dangling_edges > 0:
            recommendations.append("Fix dangling edges: nodes were deleted while relations remained.")
        if contradiction_count > 0:
            recommendations.append("Review contradiction issues and split statements by context.")
        if coverage < 0.5:
            recommendations.append("Improve node summaries/titles to boost graph-RAG retrieval quality.")
        if len(nodes) > 10000:
            recommendations.append("Add pagination/indexing for large graphs (10k+ nodes).")
        if not recommendations:
            recommendations.append("Quality is stable. Continue with periodic contradiction and retrieval checks.")

        self.api.engine._record_event(  # noqa: SLF001
            "quality_harness_ran",
            {
                "user_id": user_id,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "score": round(score, 2),
                "dangling_edges": dangling_edges,
                "contradictions": contradiction_count,
            },
        )

        return {
            "user_id": user_id,
            "score": round(score, 2),
            "checks": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "dangling_edges": dangling_edges,
                "contradiction_count": contradiction_count,
                "namespace_counts": dict(namespace_counts),
                "rag_probe_hits": rag_hits,
                "rag_probe_total": len(sample_queries),
            },
            "rag_results": rag_results,
            "recommendations": recommendations,
            **self.snapshot_payload(),
        }

    def project_backup_create(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        root = self._as_mapping(payload)
        label = " ".join(str(root.get("label", "manual") or "manual").split()).strip() or "manual"
        user_id = str(root.get("user_id", "") or "").strip()
        include_events = self._to_bool(root.get("include_events", True))
        event_limit = max(0, min(10000, self._to_int(root.get("event_limit", 2000), 2000)))
        backup_payload = {
            "meta": {
                "created_at": time.time(),
                "label": label,
                "user_id": user_id,
                "policy": self._llm_policy_snapshot(),
            },
            "graph": self.api.engine.snapshot(),
            "events": self.list_events(limit=event_limit) if include_events else [],
        }
        path = self._write_project_backup(backup_payload, label=label)
        self.api.engine._record_event(  # noqa: SLF001
            "project_backup_created",
            {
                "path": path,
                "label": label,
                "user_id": user_id,
                "include_events": include_events,
                "event_limit": event_limit,
            },
        )
        return {
            "ok": True,
            "path": path,
            "label": label,
            "include_events": include_events,
            "event_count": len(self._as_list(backup_payload.get("events"))),
            "graph_counts": {
                "nodes": len(self._as_mapping(backup_payload.get("graph")).get("nodes", {})),
                "edges": len(self._as_list(self._as_mapping(backup_payload.get("graph")).get("edges"))),
            },
        }

    def project_backup_restore(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        root = self._as_mapping(payload)
        requested_path = str(root.get("path", "") or root.get("backup_path", "") or "").strip()
        if not requested_path and self._to_bool(root.get("latest", True)):
            backup_dir = self._project_backups_dir()
            candidates = sorted(backup_dir.glob("project_backup_*.json"))
            if candidates:
                requested_path = str(candidates[-1])
        if not requested_path:
            raise ValueError("path is required")
        path = Path(requested_path)
        if not path.exists() or not path.is_file():
            raise ValueError("backup file not found")

        raw = path.read_text(encoding="utf-8")
        parsed = self._safe_json_loads(raw, {})
        data = self._as_mapping(parsed)
        graph_snapshot = self._as_mapping(data.get("graph"))
        raw_nodes = graph_snapshot.get("nodes")
        if isinstance(raw_nodes, Mapping):
            nodes_in = len(raw_nodes)
        elif isinstance(raw_nodes, list):
            nodes_in = len(raw_nodes)
        else:
            nodes_in = 0
        edges_in = len(self._as_list(graph_snapshot.get("edges")))

        user_id = str(root.get("user_id", "") or "").strip()
        session_id = str(root.get("session_id", "") or "").strip()
        apply_changes = self._to_bool(root.get("apply_changes", False))
        confirmation = root.get("confirmation", root.get("security_decision", ""))
        policy_decision = self._llm_policy_decision(
            user_id=user_id,
            session_id=session_id,
            action="backup_restore",
            requested_apply=apply_changes,
            confirmation=confirmation,
        )

        restore_result = {
            "created_nodes": 0,
            "created_edges": 0,
        }
        if policy_decision.get("apply_allowed", False):
            serialized_nodes: list[dict[str, Any]] = []
            if isinstance(raw_nodes, Mapping):
                for key, row in raw_nodes.items():
                    item = self._as_mapping(row)
                    serialized_nodes.append(
                        {
                            "id": self._to_int(key, self._to_int(item.get("id", 0), 0)),
                            "type": str(item.get("type", "generic") or "generic"),
                            "attributes": self._as_mapping(item.get("attributes")),
                            "state": self._as_mapping(item.get("state")),
                        }
                    )
            elif isinstance(raw_nodes, list):
                for row in raw_nodes:
                    item = self._as_mapping(row)
                    serialized_nodes.append(
                        {
                            "id": self._to_int(item.get("id", 0), 0),
                            "type": str(item.get("type", "generic") or "generic"),
                            "attributes": self._as_mapping(item.get("attributes")),
                            "state": self._as_mapping(item.get("state")),
                        }
                    )
            restore_result = self._restore_graph_from_snapshot(
                {
                    "nodes": serialized_nodes,
                    "edges": self._as_list(graph_snapshot.get("edges")),
                }
            )
            if self._to_bool(root.get("restore_policy", True)):
                meta = self._as_mapping(data.get("meta"))
                policy = self._as_mapping(meta.get("policy"))
                if policy:
                    self._llm_policy["mode"] = self._pick_allowed_token(
                        policy.get("mode"),
                        allowed=_LLM_POLICY_MODES_ALLOWED,
                        default="confirm_required",
                    )
                    self._llm_policy["trusted_sessions"] = self._dedupe_strings(
                        self._to_list_of_strings(policy.get("trusted_sessions"))
                    )
                    self._llm_policy["trusted_users"] = self._dedupe_strings(
                        self._to_list_of_strings(policy.get("trusted_users"))
                    )
                    self._llm_policy["allow_apply_for_actions"] = self._dedupe_strings(
                        self._to_list_of_strings(policy.get("allow_apply_for_actions"))
                    )
                    self._llm_policy["updated_at"] = time.time()
            self._persist_graph_safe()

        self.api.engine._record_event(  # noqa: SLF001
            "project_backup_restored",
            {
                "path": str(path),
                "apply_allowed": bool(policy_decision.get("apply_allowed", False)),
                "requested_apply": bool(apply_changes),
                "nodes_in_backup": nodes_in,
                "edges_in_backup": edges_in,
                "restored_nodes": int(restore_result.get("created_nodes", 0)),
                "restored_edges": int(restore_result.get("created_edges", 0)),
            },
        )

        return {
            "path": str(path),
            "preview": {
                "nodes": nodes_in,
                "edges": edges_in,
            },
            "policy": policy_decision,
            "applied": bool(policy_decision.get("apply_allowed", False)),
            "result": restore_result,
            **self.snapshot_payload(),
        }

    def project_audit_logs(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        root = self._as_mapping(payload)
        limit = max(1, min(2000, self._to_int(root.get("limit", 200), 200)))
        include_backups = self._to_bool(root.get("include_backups", True))
        events = self._collect_audit_events(limit=limit)
        backups: list[dict[str, Any]] = []
        if include_backups:
            for path in sorted(self._project_backups_dir().glob("project_backup_*.json"))[-200:]:
                try:
                    stat = path.stat()
                    backups.append(
                        {
                            "path": str(path),
                            "size_bytes": int(stat.st_size),
                            "modified_at": float(stat.st_mtime),
                        }
                    )
                except Exception:
                    continue
        return {
            "events": events[-limit:],
            "event_count": len(events[-limit:]),
            "backups": backups,
            "backup_count": len(backups),
        }

    def project_wrapper_profile_get(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        root = self._as_mapping(payload)
        user_id = str(root.get("user_id", "default_user") or "default_user").strip() or "default_user"
        node = self._ensure_wrapper_profile_node(user_id=user_id)
        return {
            "profile": self._wrapper_profile_payload(node),
            "roles_allowed": list(_DEBATE_ROLES_ALLOWED),
            "memory_scopes_allowed": list(_MEMORY_SCOPE_ALLOWED),
        }

    def project_wrapper_profile_update(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        root = self._as_mapping(payload)
        user_id = str(root.get("user_id", "default_user") or "default_user").strip() or "default_user"
        node = self._ensure_wrapper_profile_node(user_id=user_id)
        current_profile = self._wrapper_profile_payload(node)
        current_personalization = self._as_mapping(current_profile.get("personalization"))

        personalization_source = root.get("personalization")
        if not isinstance(personalization_source, Mapping):
            personalization_source = root
        requested_personalization = self._sanitize_personalization(personalization_source)
        merged = self._merge_personalization(current_personalization, requested_personalization)

        attrs = self._as_mapping(node.attributes)
        for key in ("response_style", "reasoning_depth", "risk_tolerance", "tone"):
            if key in merged:
                node.attributes[key] = merged.get(key)
        for key in ("focus_goals", "domain_focus", "avoid_topics"):
            if key in merged:
                node.attributes[key] = self._to_list_of_strings(merged.get(key))
        if "memory_notes" in merged:
            node.attributes["memory_notes"] = str(merged.get("memory_notes", "") or "")[:1200]
        if "llm_roles" in merged:
            node.attributes["llm_roles"] = self._as_mapping(merged.get("llm_roles"))

        preferred_role = self._debate_role(
            root.get("preferred_role", attrs.get("preferred_role", "general")),
            fallback="general",
        )
        preferred_model_path = str(root.get("preferred_model_path", attrs.get("preferred_model_path", "")) or "").strip()
        memory_scope = self._pick_allowed_token(
            root.get("memory_scope", attrs.get("memory_scope", "owned")),
            allowed=_MEMORY_SCOPE_ALLOWED,
            default="owned",
        )
        node.attributes["preferred_role"] = preferred_role
        node.attributes["preferred_model_path"] = preferred_model_path
        node.attributes["memory_scope"] = memory_scope
        node.attributes["updated_at"] = time.time()

        persisted = self._persist_graph_safe()
        self.api.engine._record_event(  # noqa: SLF001
            "wrapper_profile_updated",
            {
                "user_id": user_id,
                "preferred_role": preferred_role,
                "memory_scope": memory_scope,
            },
        )
        return {
            "ok": True,
            "persisted": bool(persisted),
            "profile": self._wrapper_profile_payload(node),
            "roles_allowed": list(_DEBATE_ROLES_ALLOWED),
            "memory_scopes_allowed": list(_MEMORY_SCOPE_ALLOWED),
            **self.snapshot_payload(),
        }

    def project_wrapper_feedback(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        root = self._as_mapping(payload)
        user_id = str(root.get("user_id", "default_user") or "default_user").strip() or "default_user"
        session_id = str(root.get("session_id", "") or "").strip()
        feedback_items = self._normalize_feedback_items(
            root.get("feedback_items", root.get("items", []))
        )
        if not feedback_items:
            feedback_items = self._normalize_feedback_items(
                [
                    {
                        "message": str(root.get("message", "") or "").strip(),
                        "score": self._confidence(root.get("score", 0.0), 0.0),
                        "decision": str(root.get("decision", "") or "").strip(),
                        "target": str(root.get("target", "") or "").strip(),
                    }
                ]
            )
        if not feedback_items:
            raise ValueError("feedback_items is required")

        node = self._ensure_wrapper_profile_node(user_id=user_id)
        summary = self._apply_wrapper_feedback_to_profile(
            profile_node=node,
            feedback_items=feedback_items,
        )
        node.attributes["updated_at"] = time.time()

        attach_to_graph = self._to_bool(root.get("attach_to_graph", False))
        feedback_node_id = 0
        if attach_to_graph:
            feedback_node = self.api.engine.create_node(
                "llm_wrapper_feedback",
                attributes={
                    "user_id": user_id,
                    "session_id": session_id,
                    "items": feedback_items,
                    "summary": summary,
                    "created_at": time.time(),
                },
                state={"confidence": 0.72},
            )
            feedback_node_id = int(feedback_node.id)
            self._connect_nodes(
                from_node=int(node.id),
                to_node=int(feedback_node.id),
                relation_type="wrapper_feedback",
                weight=0.84,
                logic_rule="wrapper_feedback",
            )

        persisted = self._persist_graph_safe()
        self.api.engine._record_event(  # noqa: SLF001
            "wrapper_feedback_saved",
            {
                "user_id": user_id,
                "session_id": session_id,
                "items": len(feedback_items),
                "feedback_node_id": feedback_node_id,
            },
        )
        return {
            "ok": True,
            "persisted": bool(persisted),
            "summary": summary,
            "feedback_node_id": int(feedback_node_id),
            "profile": self._wrapper_profile_payload(node),
            **self.snapshot_payload(),
        }

    def project_wrapper_respond(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        root = self._as_mapping(payload)
        user_id = str(root.get("user_id", "default_user") or "default_user").strip() or "default_user"
        session_id = str(root.get("session_id", "") or "").strip()
        message = " ".join(
            str(root.get("message", root.get("prompt", root.get("text", ""))) or "").split()
        ).strip()
        if not message:
            raise ValueError("message is required")

        profile_node = self._ensure_wrapper_profile_node(user_id=user_id)
        profile = self._wrapper_profile_payload(profile_node)
        base_personalization = self._as_mapping(profile.get("personalization"))
        override_personalization = self._sanitize_personalization(root.get("personalization"))
        personalization = self._merge_personalization(base_personalization, override_personalization)

        preferred_role = str(profile.get("preferred_role", "general") or "general").strip() or "general"
        requested_role = root.get("role", root.get("model_role", preferred_role))
        role = self._debate_role(requested_role, fallback=preferred_role or "general")
        model_path = str(root.get("model_path", profile.get("preferred_model_path", "")) or "").strip()
        use_memory = self._to_bool(root.get("use_memory", True))
        memory_scope = self._pick_allowed_token(
            root.get("memory_scope", profile.get("memory_scope", "owned")),
            allowed=_MEMORY_SCOPE_ALLOWED,
            default="owned",
        )
        memory_namespace = str(root.get("memory_namespace", "") or "").strip()
        memory_top_k = max(1, min(24, self._to_int(root.get("memory_top_k", 6), 6)))
        subject_name = " ".join(str(root.get("subject_name", "") or "").split()).strip()
        gossip_mode = self._normalize_gossip_mode(root.get("gossip_mode"), default="auto")
        allow_subject_branch_write = self._to_bool(root.get("allow_subject_branch_write", True))
        capture_dialect = self._to_bool(root.get("capture_dialect", True))
        auto_triage = self._to_bool(root.get("auto_triage", True))
        triage_with_llm = self._to_bool(root.get("triage_with_llm", True))
        memory_context = (
            self._wrapper_memory_context(
                user_id=user_id,
                query=message,
                scope=memory_scope,
                namespace=memory_namespace,
                top_k=memory_top_k,
            )
            if use_memory
            else []
        )

        llm_fn, resolution = self._resolve_role_or_model_llm(
            role=role,
            model_path=model_path,
        )
        prompt = self._wrapper_prompt(
            message=message,
            personalization=personalization,
            memory_context=memory_context,
        )
        raw_output = ""
        reply = ""
        used_fallback = False
        if llm_fn is not None:
            try:
                raw_output = str(llm_fn(prompt) or "").strip()
            except Exception:
                raw_output = ""
            reply = " ".join(raw_output.split()).strip()
        if not reply:
            used_fallback = True
            reply = self._wrapper_fallback_reply(
                message=message,
                memory_context=memory_context,
            )

        apply_profile_update = self._to_bool(root.get("apply_profile_update", True))
        if apply_profile_update:
            profile_node.attributes["response_style"] = personalization.get("response_style", "adaptive")
            profile_node.attributes["reasoning_depth"] = personalization.get("reasoning_depth", "balanced")
            profile_node.attributes["risk_tolerance"] = personalization.get("risk_tolerance", "medium")
            profile_node.attributes["tone"] = personalization.get("tone", "neutral")
            profile_node.attributes["focus_goals"] = self._to_list_of_strings(personalization.get("focus_goals"))
            profile_node.attributes["domain_focus"] = self._to_list_of_strings(personalization.get("domain_focus"))
            profile_node.attributes["avoid_topics"] = self._to_list_of_strings(personalization.get("avoid_topics"))
            profile_node.attributes["memory_notes"] = str(personalization.get("memory_notes", "") or "")[:1200]
            profile_node.attributes["llm_roles"] = self._as_mapping(personalization.get("llm_roles"))
            profile_node.attributes["preferred_role"] = role
            profile_node.attributes["preferred_model_path"] = model_path
            profile_node.attributes["memory_scope"] = memory_scope
            profile_node.attributes["last_query"] = message[:1600]
            profile_node.attributes["last_reply"] = reply[:2400]
            profile_node.attributes["last_context_nodes"] = [int(row.get("node_id", 0)) for row in memory_context[:12]]
            profile_node.attributes["query_total"] = self._to_int(profile_node.attributes.get("query_total", 0), 0) + 1
            profile_node.attributes["updated_at"] = time.time()

        feedback_items = self._normalize_feedback_items(root.get("feedback_items"))
        feedback_summary: dict[str, Any] = {}
        if feedback_items:
            feedback_summary = self._apply_wrapper_feedback_to_profile(
                profile_node=profile_node,
                feedback_items=feedback_items,
            )

        store_interaction = self._to_bool(root.get("store_interaction", False))
        session_node_id = 0
        if store_interaction:
            session_node = self.api.engine.create_node(
                "llm_wrapper_session",
                attributes={
                    "user_id": user_id,
                    "session_id": session_id,
                    "message": message,
                    "reply": reply,
                    "role": role,
                    "model_path": model_path,
                    "memory_context": memory_context[:8],
                    "used_fallback": bool(used_fallback),
                    "created_at": time.time(),
                },
                state={"confidence": 0.74 if not used_fallback else 0.45},
            )
            session_node_id = int(session_node.id)
            self._connect_nodes(
                from_node=int(profile_node.id),
                to_node=int(session_node.id),
                relation_type="wrapper_interaction",
                weight=0.84 if not used_fallback else 0.55,
                logic_rule="wrapper_respond",
            )

        subject_binding = self._bind_subject_conversation(
            user_id=user_id,
            session_node_id=int(session_node_id),
            message=message,
            reply=reply,
            explicit_subject=subject_name,
            updates=[],
            verification={},
            gossip_mode=gossip_mode,
            allow_write=allow_subject_branch_write,
        )
        dialect_summary = self._capture_wrapper_dialect(
            user_id=user_id,
            profile_node=profile_node,
            message=message,
            reply=reply,
            attach_to_graph=bool(capture_dialect and (store_interaction or apply_profile_update)),
        ) if capture_dialect else {"captured_terms": 0, "dictionary_size": 0, "top_terms": [], "dictionary_node_id": 0}
        triage = self._auto_interaction_triage(
            user_id=user_id,
            session_id=session_id,
            source="wrapper",
            message=message,
            reply=reply,
            updates=[],
            auto_triage=auto_triage,
            triage_with_llm=triage_with_llm,
            model_role=role,
            model_path=model_path,
            attach_to_graph=bool(store_interaction or apply_profile_update),
            related_node_id=int(session_node_id or profile_node.id),
            llm_fn=llm_fn,
            llm_resolution=resolution,
        )
        input_extraction = self._capture_input_intelligence(
            user_id=user_id,
            session_id=session_id or f"wrapper_{user_id}",
            source="wrapper_input",
            text=message,
            context=f"role={role}; used_fallback={used_fallback}; reply={reply[:600]}",
            related_node_ids=[int(profile_node.id), int(session_node_id)] if int(session_node_id) > 0 else [int(profile_node.id)],
            apply_to_graph=True,
        )

        should_persist = bool(
            apply_profile_update
            or feedback_items
            or store_interaction
            or bool(subject_binding.get("attached", False))
            or int(dialect_summary.get("captured_terms", 0) or 0) > 0
            or bool(self._as_mapping(triage.get("graph")).get("attached", False))
            or bool(self._as_mapping(input_extraction.get("graph_binding")).get("attached", False))
            or bool(self._as_mapping(input_extraction.get("graph_monitor")).get("attached", False))
        )
        persisted = self._persist_graph_safe() if should_persist else False

        self.api.engine._record_event(  # noqa: SLF001
            "wrapper_responded",
            {
                "user_id": user_id,
                "session_id": session_id,
                "role": role,
                "model_path": model_path,
                "llm_available": llm_fn is not None,
                "used_fallback": bool(used_fallback),
                "context_nodes": len(memory_context),
                "session_node_id": int(session_node_id),
                "gossip_detected": bool(subject_binding.get("gossip_detected", False)),
                "subject_branch_attached": bool(subject_binding.get("attached", False)),
                "dialect_terms_captured": int(dialect_summary.get("captured_terms", 0) or 0),
                "triage_items": len(self._as_list(triage.get("items"))),
                "triage_noise_ratio": self._to_float(triage.get("noise_ratio", 0.0), 0.0),
            },
        )
        graph_monitor = self._as_mapping(input_extraction.get("graph_monitor"))
        summary = {
            "reply_length": len(reply),
            "memory_context_count": len(memory_context),
            "feedback_items": int(feedback_summary.get("total", 0) or 0),
            "triage_items": len(self._as_list(triage.get("items"))),
            "gossip_detected": bool(subject_binding.get("gossip_detected", False)),
        }
        execution = self._execution_status(
            action="wrapper_respond",
            persisted=persisted,
            input_extraction=input_extraction,
            graph_monitor=graph_monitor,
            extra={
                "session_node_id": int(session_node_id),
                "triage_items": summary["triage_items"],
            },
        )
        return {
            "user_id": user_id,
            "session_id": session_id,
            "reply": reply,
            "model": {
                "role": role,
                "model_path": model_path,
                "resolution": resolution,
                "llm_available": llm_fn is not None,
                "used_fallback": bool(used_fallback),
            },
            "memory": {
                "enabled": bool(use_memory),
                "scope": memory_scope,
                "namespace": memory_namespace,
                "top_k": memory_top_k,
                "context": memory_context,
            },
            "profile": self._wrapper_profile_payload(profile_node),
            "feedback": feedback_summary,
            "subject_binding": subject_binding,
            "gossip_detected": bool(subject_binding.get("gossip_detected", False)),
            "dialect": dialect_summary,
            "triage": triage,
            "input_extraction": input_extraction,
            "graph_monitor": graph_monitor,
            "summary": summary,
            "execution": execution,
            "persisted": bool(persisted),
            "session_node_id": int(session_node_id),
            "raw_output": raw_output[:6000],
            **self.snapshot_payload(),
        }

    def project_integration_layer_manifest(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        root = self._as_mapping(payload)
        requested_host = self._safe_slug(root.get("host", "generic"), default="generic")
        host = requested_host if requested_host in _INTEGRATION_LAYER_HOSTS_ALLOWED else "generic"
        app_id = self._safe_slug(root.get("app_id", "external_app"), default="external_app")

        model_advisors: dict[str, Any] = {}
        try:
            from src.utils.local_llm_provider import list_model_advisors

            model_advisors = self._as_mapping(list_model_advisors())
        except Exception:
            model_advisors = {
                "detected_models": [],
                "advisors": [],
            }

        capabilities = [
            {
                "action": "wrapper.respond",
                "description": "Conversational response with memory + triage.",
                "writes_graph": True,
                "supports_chat_response": True,
                "supports_structured_result": True,
            },
            {
                "action": "archive.chat",
                "description": "Verification-first archive update chat.",
                "writes_graph": True,
                "supports_chat_response": True,
                "supports_structured_result": True,
            },
            {
                "action": "user_graph.update",
                "description": "Apply profile/questionnaire updates into semantic user graph.",
                "writes_graph": True,
                "supports_chat_response": True,
                "supports_structured_result": True,
            },
            {
                "action": "personal_tree.ingest",
                "description": "Extract summary + thought points and write into personal tree.",
                "writes_graph": True,
                "supports_chat_response": True,
                "supports_structured_result": True,
            },
        ]
        actions = [
            {
                "key": str(item.get("action", "") or "").strip(),
                "description": str(item.get("description", "") or "").strip(),
                "writes_graph": bool(item.get("writes_graph", False)),
                "supports_chat_response": bool(item.get("supports_chat_response", False)),
                "supports_structured_result": bool(item.get("supports_structured_result", False)),
            }
            for item in capabilities
            if str(item.get("action", "") or "").strip()
        ]
        manifest_summary = {
            "host": host,
            "app_id": app_id,
            "action_count": len(actions),
            "writes_graph_actions": sum(1 for item in actions if item.get("writes_graph")),
            "chat_actions": sum(1 for item in actions if item.get("supports_chat_response")),
            "detected_model_count": len(self._to_list_of_strings(model_advisors.get("detected_models"))),
        }

        return {
            "layer": "autograph_integration_layer",
            "version": "1.1.0",
            "generated_at": float(time.time()),
            "host": host,
            "app_id": app_id,
            "endpoint": {
                "manifest": "/api/integration/layer/manifest",
                "invoke": "/api/integration/layer/invoke",
            },
            "hosts_allowed": list(_INTEGRATION_LAYER_HOSTS_ALLOWED),
            "actions_allowed": list(_INTEGRATION_LAYER_ACTIONS_ALLOWED),
            "actions": actions,
            "capabilities": capabilities,
            "summary": manifest_summary,
            "ui_contract": {
                "chat_response_field": "chat_response",
                "structured_result_field": "structured_result",
                "raw_result_field": "result",
                "editable_output_paths": [
                    "result.archive_updates",
                    "result.input_extraction.updates",
                    "result.triage.items",
                    "result.review.archive_updates",
                ],
            },
            "defaults": {
                "auto_triage": True,
                "triage_with_llm": True,
                "apply_to_graph": True,
                "use_llm_profile": True,
            },
            "models": {
                "detected": self._to_list_of_strings(model_advisors.get("detected_models")),
                "advisors": self._as_list(model_advisors.get("advisors")),
            },
            "examples": {
                "wrapper.respond": {
                    "action": "wrapper.respond",
                    "host": host,
                    "app_id": app_id,
                    "user_id": "demo_user",
                    "session_id": "sess_demo_wrapper",
                    "input": {
                        "message": "Give me a concise action plan for my next step.",
                    },
                    "options": {
                        "role": "general",
                        "use_memory": True,
                        "auto_triage": True,
                    },
                },
            },
        }

    def project_integration_layer_invoke(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        root = self._as_mapping(payload)
        raw_action = " ".join(str(root.get("action", "wrapper.respond") or "wrapper.respond").split()).strip().lower()
        action_alias = raw_action.replace("/", ".").replace(":", ".").replace("__", ".")
        if action_alias not in _INTEGRATION_LAYER_ACTIONS_ALLOWED:
            raise ValueError(f"unsupported action: {raw_action}")
        action = action_alias

        requested_host = self._safe_slug(root.get("host", "generic"), default="generic")
        host = requested_host if requested_host in _INTEGRATION_LAYER_HOSTS_ALLOWED else "generic"
        app_id = self._safe_slug(root.get("app_id", "external_app"), default="external_app")
        user_id = str(root.get("user_id", "default_user") or "default_user").strip() or "default_user"
        session_id = str(root.get("session_id", "") or "").strip() or f"{host}_session"

        input_payload = self._as_mapping(root.get("input"))
        options = self._as_mapping(root.get("options"))

        message = " ".join(
            str(
                input_payload.get(
                    "message",
                    input_payload.get("text", root.get("message", root.get("text", ""))),
                )
                or ""
            ).split()
        ).strip()
        context = " ".join(str(input_payload.get("context", root.get("context", "")) or "").split()).strip()
        model_path = str(
            options.get("model_path", input_payload.get("model_path", root.get("model_path", ""))) or ""
        ).strip()
        model_role = self._debate_role(
            options.get("model_role", input_payload.get("model_role", root.get("model_role", "general"))),
            fallback="general",
        )
        auto_triage = self._to_bool(options.get("auto_triage", root.get("auto_triage", True)))
        triage_with_llm = self._to_bool(options.get("triage_with_llm", root.get("triage_with_llm", True)))

        chat_response = ""
        structured_result: dict[str, Any] = {}
        result: dict[str, Any] = {}

        if action == "wrapper.respond":
            if not message:
                raise ValueError("input.message is required for wrapper.respond")
            result = self.project_wrapper_respond(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "message": message,
                    "role": self._debate_role(options.get("role", model_role), fallback=model_role),
                    "model_path": model_path,
                    "use_memory": self._to_bool(options.get("use_memory", True)),
                    "memory_scope": self._pick_allowed_token(
                        options.get("memory_scope", "owned"),
                        allowed=_MEMORY_SCOPE_ALLOWED,
                        default="owned",
                    ),
                    "memory_namespace": str(options.get("memory_namespace", "") or "").strip(),
                    "memory_top_k": max(1, min(24, self._to_int(options.get("memory_top_k", 6), 6))),
                    "apply_profile_update": self._to_bool(options.get("apply_profile_update", True)),
                    "store_interaction": self._to_bool(options.get("store_interaction", True)),
                    "auto_triage": auto_triage,
                    "triage_with_llm": triage_with_llm,
                }
            )
            chat_response = " ".join(str(result.get("reply", "") or "").split()).strip()
            structured_result = {
                "triage": self._as_mapping(result.get("triage")),
                "memory": self._as_mapping(result.get("memory")),
                "subject_binding": self._as_mapping(result.get("subject_binding")),
                "dialect": self._as_mapping(result.get("dialect")),
                "input_extraction": self._as_mapping(result.get("input_extraction")),
                "graph_monitor": self._as_mapping(result.get("graph_monitor")),
            }

        elif action == "archive.chat":
            if not message:
                raise ValueError("input.message is required for archive.chat")
            result = self.project_archive_verified_chat(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "message": message,
                    "context": context,
                    "model_path": model_path,
                    "model_role": model_role,
                    "apply_to_graph": self._to_bool(options.get("apply_to_graph", True)),
                    "verification_mode": self._pick_allowed_token(
                        options.get("verification_mode", "strict"),
                        allowed=_ARCHIVE_VERIFICATION_MODES_ALLOWED,
                        default="strict",
                    ),
                    "top_k": max(1, min(8, self._to_int(options.get("top_k", 3), 3))),
                    "auto_triage": auto_triage,
                    "triage_with_llm": triage_with_llm,
                }
            )
            chat_response = " ".join(str(result.get("assistant_reply", "") or "").split()).strip()
            structured_result = {
                "verification": self._as_mapping(result.get("verification")),
                "archive_updates": self._as_list(result.get("archive_updates")),
                "triage": self._as_mapping(result.get("triage")),
                "graph_binding": self._as_mapping(result.get("graph_binding")),
                "graph_monitor": self._as_mapping(result.get("graph_monitor")),
            }

        elif action == "user_graph.update":
            text = message
            if not text:
                text = " ".join(
                    self._to_list_of_strings(
                        input_payload.get(
                            "profile_lines",
                            input_payload.get("facts", []),
                        )
                    )
                ).strip()
            if not text:
                raise ValueError("input.message or input.profile_lines is required for user_graph.update")

            result = self.project_user_graph_update(
                {
                    "user_id": user_id,
                    "display_name": str(input_payload.get("display_name", user_id) or user_id).strip() or user_id,
                    "text": text,
                    "language": str(input_payload.get("language", "en") or "en").strip() or "en",
                    "session_id": session_id,
                    "use_llm_profile": self._to_bool(options.get("use_llm_profile", True)),
                    "include_client_profile": self._to_bool(options.get("include_client_profile", False)),
                    "profile_text": text,
                    "fears": input_payload.get("fears", []),
                    "desires": input_payload.get("desires", []),
                    "goals": input_payload.get("goals", []),
                    "principles": input_payload.get("principles", []),
                    "opportunities": input_payload.get("opportunities", []),
                    "abilities": input_payload.get("abilities", []),
                    "access": input_payload.get("access", []),
                    "knowledge": input_payload.get("knowledge", []),
                    "assets": input_payload.get("assets", []),
                }
            )
            profile_json = self._as_mapping(result.get("profile_update_json"))
            dimensions = self._as_mapping(profile_json.get("dimensions"))
            non_empty_dimensions = sum(
                1 for rows in dimensions.values() if isinstance(rows, list) and len(rows) > 0
            )
            chat_response = (
                f"User graph updated. Non-empty dimensions: {non_empty_dimensions}. "
                f"Profile source: {str(profile_json.get('source', 'heuristic') or 'heuristic')}."
            )
            structured_result = {
                "profile_update_json": profile_json,
                "semantic_binding": self._as_mapping(result.get("semantic_binding")),
                "input_extraction": self._as_mapping(result.get("input_extraction")),
                "graph_monitor": self._as_mapping(result.get("graph_monitor")),
            }

        elif action == "personal_tree.ingest":
            text = message
            if not text:
                raise ValueError("input.message is required for personal_tree.ingest")
            result = self.project_personal_tree_ingest(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "title": str(input_payload.get("title", "") or "").strip(),
                    "topic": str(input_payload.get("topic", "") or "").strip(),
                    "text": text,
                    "source_type": self._pick_allowed_token(
                        input_payload.get("source_type", "text"),
                        allowed=_PERSONAL_TREE_SOURCE_TYPES_ALLOWED,
                        default="text",
                    ),
                    "source_url": str(input_payload.get("source_url", "") or "").strip(),
                    "source_title": str(input_payload.get("source_title", "") or "").strip(),
                    "max_points": max(2, min(12, self._to_int(input_payload.get("max_points", 6), 6))),
                    "max_nodes": max(40, min(300, self._to_int(options.get("max_nodes", 180), 180))),
                }
            )
            extraction = self._as_mapping(result.get("extraction"))
            chat_response = " ".join(
                str(extraction.get("summary", "") or result.get("ingest", {}).get("title", "Ingest completed")).split()
            ).strip()
            if not chat_response:
                chat_response = "Ingest completed."
            structured_result = {
                "extraction": extraction,
                "semantic_binding": self._as_mapping(result.get("semantic_binding")),
                "tree": self._as_mapping(result.get("tree")),
                "input_extraction": self._as_mapping(result.get("input_extraction")),
                "graph_monitor": self._as_mapping(result.get("graph_monitor")),
            }

        self.api.engine._record_event(  # noqa: SLF001
            "integration_layer_invoked",
            {
                "host": host,
                "app_id": app_id,
                "action": action,
                "user_id": user_id,
                "session_id": session_id,
                "auto_triage": bool(auto_triage),
            },
        )
        input_extraction = self._as_mapping(
            structured_result.get("input_extraction", result.get("input_extraction"))
        )
        graph_monitor = self._as_mapping(
            structured_result.get("graph_monitor", result.get("graph_monitor"))
        )
        persisted = bool(result.get("persisted", False))
        execution = self._execution_status(
            action=action,
            persisted=persisted,
            input_extraction=input_extraction,
            graph_monitor=graph_monitor,
        )
        invoke_summary = {
            "action": action,
            "chat_response_present": bool(chat_response),
            "chat_response_length": len(chat_response),
            "structured_keys": sorted(str(key) for key in structured_result.keys()),
            "persisted": persisted,
            "writes_graph": bool(action in {"wrapper.respond", "archive.chat", "user_graph.update", "personal_tree.ingest"}),
        }
        action_summary = self._as_mapping(result.get("summary"))
        if action_summary:
            invoke_summary["action_summary"] = action_summary

        return {
            "ok": True,
            "layer": "autograph_integration_layer",
            "host": host,
            "app_id": app_id,
            "action": action,
            "user_id": user_id,
            "session_id": session_id,
            "chat_response": chat_response,
            "summary": invoke_summary,
            "execution": execution,
            "structured_result": structured_result,
            "result": result,
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
        security_decision = str(payload.get("security_decision", "") or "").strip()
        if not security_decision and self._to_bool(payload.get("force_execute", False)):
            security_decision = "proceed"
        return self._living_required().run_prompt(
            prompt_name=prompt_name,
            variables=variables,
            user_id=user_id,
            session_id=session_id,
            security_decision=security_decision,
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
                "graph_foundation_builder": True,
                "graph_node_assist": True,
                "graph_edge_assist": True,
                "model_advisors": True,
                "llm_role_debate": True,
                "hallucination_hunter": True,
                "archive_verified_chat": True,
                "packages_manager": True,
                "memory_namespaces": True,
                "graph_rag": True,
                "contradiction_scan": True,
                "task_risk_board": True,
                "timeline_replay": True,
                "llm_policy_layer": True,
                "quality_harness": True,
                "backup_restore": True,
                "audit_logs": True,
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
        subject_name = str(sanitized.get("subject_name", "") or "").strip()
        gossip_mode = self._normalize_gossip_mode(sanitized.get("gossip_mode"), default="auto")
        allow_subject_branch_write = self._to_bool(sanitized.get("allow_subject_branch_write", True))
        capture_dialect = self._to_bool(sanitized.get("capture_dialect", True))
        auto_triage = self._to_bool(sanitized.get("auto_triage", True))
        triage_with_llm = self._to_bool(sanitized.get("triage_with_llm", True))

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
            subject_name=subject_name,
            gossip_mode=gossip_mode,
            allow_subject_branch_write=allow_subject_branch_write,
        )
        assistant_reply = self._build_archive_chat_reply(
            message=message,
            summary=summary,
            updates=updates,
            verification=verification,
        )
        profile_node = self._ensure_wrapper_profile_node(user_id=user_id)
        dialect_summary = self._capture_wrapper_dialect(
            user_id=user_id,
            profile_node=profile_node,
            message=message,
            reply=assistant_reply,
            attach_to_graph=bool(apply_to_graph and capture_dialect),
        ) if capture_dialect else {"captured_terms": 0, "dictionary_size": 0, "top_terms": [], "dictionary_node_id": 0}
        if capture_dialect:
            profile_node.attributes["updated_at"] = float(time.time())
        triage = self._auto_interaction_triage(
            user_id=user_id,
            session_id=session_id,
            source="archive",
            message=message,
            reply=assistant_reply,
            updates=updates,
            auto_triage=auto_triage,
            triage_with_llm=triage_with_llm,
            model_role=model_role,
            model_path=selected_path,
            attach_to_graph=bool(apply_to_graph),
            related_node_id=self._to_int(graph_binding.get("session_node_id", 0), 0),
            llm_fn=llm_fn,
            llm_resolution={
                "mode": resolution_mode,
                "selected_model_path": selected_path,
                "requested_model_path": model_path,
                "requested_role": model_role,
            },
        )
        graph_monitor = self._run_graph_monitor(
            user_id=user_id,
            session_id=session_id or f"archive_{user_id}",
            source="archive_chat",
            focus_node_ids=[
                self._to_int(graph_binding.get("session_node_id", 0), 0),
                *[
                    self._to_int(item, 0)
                    for item in self._as_list(graph_binding.get("update_node_ids"))
                    if self._to_int(item, 0) > 0
                ],
            ],
            hint_text=f"{summary}\n{message[:1200]}",
            model_path=_DEFAULT_GRAPH_MONITOR_MODEL_PATH,
            model_role="planner",
            apply_changes=bool(apply_to_graph),
        )
        persisted = self._auto_persist_after_write() if apply_to_graph else False

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
                "gossip_detected": bool(
                    self._as_mapping(graph_binding.get("subject_binding")).get("gossip_detected", False)
                ),
                "dialect_terms_captured": int(dialect_summary.get("captured_terms", 0) or 0),
                "triage_items": len(self._as_list(triage.get("items"))),
                "triage_noise_ratio": self._to_float(triage.get("noise_ratio", 0.0), 0.0),
            },
        )
        execution = self._execution_status(
            action="archive_chat",
            persisted=persisted,
            input_extraction={
                "enabled": True,
                "source": "archive_chat",
                "summary": summary,
                "updates": updates,
                "verification": verification,
                "graph_binding": graph_binding,
                "graph_monitor": graph_monitor,
                "model": {
                    "requested_model_path": model_path,
                    "selected_model_path": selected_path,
                    "requested_role": model_role,
                    "used_fallback": bool(resolution_mode != "model"),
                },
            },
            graph_monitor=graph_monitor,
            extra={
                "updates_count": len(updates),
                "verified": bool(verification.get("verified", False)),
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
            "subject_binding": self._as_mapping(graph_binding.get("subject_binding")),
            "gossip_detected": bool(
                self._as_mapping(graph_binding.get("subject_binding")).get("gossip_detected", False)
            ),
            "dialect": dialect_summary,
            "triage": triage,
            "graph_monitor": graph_monitor,
            "profile": self._wrapper_profile_payload(profile_node),
            "execution": execution,
            "persisted": bool(persisted),
            "review": {
                "summary": summary,
                "archive_updates": updates,
                "verification": verification,
            },
            "raw_output": raw_output[:6000],
            **self.snapshot_payload(),
        }

    def project_graph_node_assist(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        root = self._as_mapping(payload)
        node_id = self._to_int(root.get("node_id", 0), 0)
        if node_id <= 0:
            raise ValueError("node_id is required")

        action = self._pick_allowed_token(
            root.get("action", "improve"),
            allowed={"explain", "improve", "risks", "tasks", "memory"},
            default="improve",
        )

        graph_snapshot = self.snapshot_payload()
        snapshot = self._as_mapping(graph_snapshot.get("snapshot"))
        nodes = [self._as_mapping(row) for row in self._as_list(snapshot.get("nodes"))]
        edges = [self._as_mapping(row) for row in self._as_list(snapshot.get("edges"))]
        target = next((row for row in nodes if self._to_int(row.get("id", 0), 0) == node_id), None)
        if target is None:
            raise ValueError("target node was not found")

        attrs = self._as_mapping(target.get("attributes"))
        state = self._as_mapping(target.get("state"))
        node_type = str(target.get("type", "generic") or "generic").strip() or "generic"
        label = " ".join(
            str(
                attrs.get("title")
                or attrs.get("name")
                or attrs.get("profile_name")
                or attrs.get("summary")
                or attrs.get("description")
                or f"Node {node_id}"
            ).split()
        ).strip() or f"Node {node_id}"

        incoming = sum(1 for edge in edges if self._to_int(edge.get("to", 0), 0) == node_id)
        outgoing = sum(1 for edge in edges if self._to_int(edge.get("from", 0), 0) == node_id)
        related_ids = [
            self._to_int(edge.get("from", 0), 0)
            for edge in edges
            if self._to_int(edge.get("to", 0), 0) == node_id and self._to_int(edge.get("from", 0), 0) > 0
        ]
        related_ids.extend(
            self._to_int(edge.get("to", 0), 0)
            for edge in edges
            if self._to_int(edge.get("from", 0), 0) == node_id and self._to_int(edge.get("to", 0), 0) > 0
        )
        related_names: list[str] = []
        seen_related: set[int] = set()
        for related_id in related_ids:
            if related_id <= 0 or related_id in seen_related:
                continue
            seen_related.add(related_id)
            related_node = next(
                (row for row in nodes if self._to_int(row.get("id", 0), 0) == related_id),
                None,
            )
            if related_node is None:
                continue
            related_attrs = self._as_mapping(related_node.get("attributes"))
            related_label = " ".join(
                str(
                    related_attrs.get("title")
                    or related_attrs.get("name")
                    or related_attrs.get("profile_name")
                    or related_attrs.get("summary")
                    or f"Node {related_id}"
                ).split()
            ).strip()
            if related_label:
                related_names.append(related_label)
            if len(related_names) >= 6:
                break

        text_fragments: list[str] = []
        for key in (
            "summary",
            "details",
            "description",
            "notes",
            "desired_output",
            "prompt",
            "correct_answer",
            "hallucinated_answer",
            "message",
        ):
            value = " ".join(str(attrs.get(key, "") or "").split()).strip()
            if value and value not in text_fragments:
                text_fragments.append(value)
        for key in ("likes", "dislikes", "style_examples", "tool_examples", "mitigation_steps", "tags"):
            rows = self._to_list_of_strings(attrs.get(key))
            if rows:
                text_fragments.append(", ".join(rows[:8]))
        if not text_fragments:
            text_fragments.append(label)

        action_prompt_map = {
            "explain": f"Explain the selected graph node '{label}' in plain language and highlight what matters most.",
            "improve": f"Suggest concrete improvements for the selected graph node '{label}'. Focus on useful next steps.",
            "risks": f"Analyze risks, contradictions, and verification gaps around the selected graph node '{label}'.",
            "tasks": f"Turn the selected graph node '{label}' into an execution-oriented task plan with checkpoints.",
            "memory": f"Extract only durable, high-value memory from the selected graph node '{label}' and drop noise.",
        }
        user_message = " ".join(
            str(root.get("message", root.get("prompt", "")) or "").split()
        ).strip()
        if not user_message:
            user_message = action_prompt_map.get(action, action_prompt_map["improve"])

        extra_context = " ".join(str(root.get("context", "") or "").split()).strip()
        context_lines = [
            f"Target graph node #{node_id}",
            f"Type: {node_type}",
            f"Label: {label}",
            f"Incoming edges: {incoming}",
            f"Outgoing edges: {outgoing}",
        ]
        if related_names:
            context_lines.append(f"Related nodes: {', '.join(related_names)}")
        context_lines.append(f"Node content: {' | '.join(text_fragments[:6])}")
        if state:
            state_tokens = [f"{key}={value}" for key, value in list(state.items())[:6]]
            if state_tokens:
                context_lines.append(f"Node state: {', '.join(state_tokens)}")
        if extra_context:
            context_lines.append(f"User context: {extra_context}")

        default_role = "planner" if action == "tasks" else "analyst" if action in {"risks", "memory"} else "general"
        result = self.project_archive_verified_chat(
            {
                "user_id": str(root.get("user_id", "default_user") or "default_user").strip() or "default_user",
                "session_id": str(root.get("session_id", "") or "").strip() or f"graph_node_{node_id}",
                "message": user_message,
                "context": "\n".join(context_lines),
                "model_path": str(root.get("model_path", "") or "").strip(),
                "model_role": str(root.get("model_role", default_role) or default_role).strip() or default_role,
                "apply_to_graph": self._to_bool(root.get("apply_to_graph", True)),
                "verification_mode": self._pick_allowed_token(
                    root.get("verification_mode", "balanced"),
                    allowed=_ARCHIVE_VERIFICATION_MODES_ALLOWED,
                    default="balanced",
                ),
                "top_k": max(1, min(8, self._to_int(root.get("top_k", 5), 5))),
                "capture_dialect": self._to_bool(root.get("capture_dialect", True)),
                "auto_triage": self._to_bool(root.get("auto_triage", True)),
                "triage_with_llm": self._to_bool(root.get("triage_with_llm", True)),
            }
        )

        graph_binding = self._as_mapping(result.get("graph_binding"))
        session_node_id = self._to_int(graph_binding.get("session_node_id", 0), 0)
        update_node_ids = [
            self._to_int(item, 0)
            for item in self._as_list(graph_binding.get("update_node_ids"))
            if self._to_int(item, 0) > 0
        ]
        apply_to_graph = self._to_bool(root.get("apply_to_graph", True))
        graph_changed = False
        extra_persisted = False
        if apply_to_graph and session_node_id > 0:
            self._connect_nodes(
                from_node=node_id,
                to_node=session_node_id,
                relation_type="requested_node_assist",
                weight=0.88,
                logic_rule="graph_node_assist",
                metadata={"action": action, "node_type": node_type},
            )
            graph_changed = True
            self._connect_nodes(
                from_node=session_node_id,
                to_node=node_id,
                relation_type="targets_graph_node",
                weight=0.84,
                logic_rule="graph_node_assist",
                metadata={"action": action},
            )
            graph_changed = True
            for update_node_id in update_node_ids:
                self._connect_nodes(
                    from_node=update_node_id,
                    to_node=node_id,
                    relation_type="suggests_change_for",
                    weight=0.82,
                    logic_rule="graph_node_assist",
                    metadata={"action": action},
                )
                graph_changed = True
            extra_persisted = bool(self._persist_graph_safe())

        self.api.engine._record_event(  # noqa: SLF001
            "graph_node_assist_completed",
            {
                "node_id": node_id,
                "action": action,
                "session_node_id": session_node_id,
                "update_nodes": len(update_node_ids),
                "persisted": bool(result.get("persisted", False) or extra_persisted),
            },
        )

        out = dict(result)
        out["node_assist"] = {
            "node_id": node_id,
            "action": action,
            "node_label": label,
            "node_type": node_type,
            "incoming_edges": incoming,
            "outgoing_edges": outgoing,
            "related_nodes": related_names,
            "session_node_id": session_node_id,
            "update_node_ids": update_node_ids,
        }
        out["persisted"] = bool(result.get("persisted", False) or extra_persisted)
        if graph_changed:
            out.update(self.snapshot_payload())
        execution = self._as_mapping(out.get("execution"))
        if execution:
            execution["persisted"] = bool(out["persisted"])
            execution["status"] = "persisted" if bool(out["persisted"]) else str(
                execution.get("status", "in_memory") or "in_memory"
            )
            out["execution"] = execution
        return out

    def project_graph_edge_assist(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        root = self._as_mapping(payload)
        edge_payload = self._normalize_edge_payload(root)
        if not edge_payload["relation_type"]:
            raise ValueError("relation_type is required")

        edge = self._find_edge_record(
            from_node=edge_payload["from_node"],
            to_node=edge_payload["to_node"],
            relation_type=edge_payload["relation_type"],
            direction=edge_payload["direction"],
        )
        if edge is None:
            raise ValueError("edge not found")

        action = self._pick_allowed_token(
            root.get("action", "improve"),
            allowed={"explain", "improve", "risks", "merge", "split"},
            default="improve",
        )

        graph_snapshot = self.snapshot_payload()
        snapshot = self._as_mapping(graph_snapshot.get("snapshot"))
        nodes = [self._as_mapping(row) for row in self._as_list(snapshot.get("nodes"))]
        from_node = next(
            (row for row in nodes if self._to_int(row.get("id", 0), 0) == edge_payload["from_node"]),
            None,
        )
        to_node = next(
            (row for row in nodes if self._to_int(row.get("id", 0), 0) == edge_payload["to_node"]),
            None,
        )
        if from_node is None or to_node is None:
            raise ValueError("edge endpoint nodes were not found")

        from_attrs = self._as_mapping(from_node.get("attributes"))
        to_attrs = self._as_mapping(to_node.get("attributes"))

        def _node_label(node_attrs: Mapping[str, Any], node_id: int) -> str:
            return " ".join(
                str(
                    node_attrs.get("title")
                    or node_attrs.get("name")
                    or node_attrs.get("profile_name")
                    or node_attrs.get("summary")
                    or node_attrs.get("description")
                    or f"Node {node_id}"
                ).split()
            ).strip() or f"Node {node_id}"

        from_label = _node_label(from_attrs, edge_payload["from_node"])
        to_label = _node_label(to_attrs, edge_payload["to_node"])
        relation_label = " ".join(str(edge_payload["relation_type"]).replace("_", " ").split()).strip()
        metadata = self._as_mapping(edge.metadata)
        metadata_keys = list(metadata.keys())[:8]

        action_prompt_map = {
            "explain": (
                f"Explain what the relation '{relation_label}' means between '{from_label}' and '{to_label}'. "
                "Clarify why it exists and how it should be interpreted."
            ),
            "improve": (
                f"Suggest a clearer, stronger version of the relation '{relation_label}' between '{from_label}' and '{to_label}'. "
                "Focus on semantics, evidence, and confidence."
            ),
            "risks": (
                f"Analyze risks, contradictions, ambiguity, and weak evidence in the relation '{relation_label}' "
                f"between '{from_label}' and '{to_label}'."
            ),
            "merge": (
                f"Assess whether '{from_label}' and '{to_label}' should be merged into one node or kept separate, "
                f"using the relation '{relation_label}' as evidence."
            ),
            "split": (
                f"Assess whether the relation '{relation_label}' between '{from_label}' and '{to_label}' should be "
                "split into multiple more precise edges."
            ),
        }
        user_message = " ".join(str(root.get("message", root.get("prompt", "")) or "").split()).strip()
        if not user_message:
            user_message = action_prompt_map.get(action, action_prompt_map["improve"])

        extra_context = " ".join(str(root.get("context", "") or "").split()).strip()
        context_lines = [
            f"Target graph edge {edge_payload['from_node']} -> {edge_payload['to_node']}",
            f"From node: {from_label} ({str(from_node.get('type', 'generic') or 'generic').strip() or 'generic'})",
            f"To node: {to_label} ({str(to_node.get('type', 'generic') or 'generic').strip() or 'generic'})",
            f"Relation type: {edge_payload['relation_type']}",
            f"Direction: {edge_payload['direction']}",
            f"Weight: {self._to_float(edge.weight, 0.0):.2f}",
            f"Logic rule: {str(edge.logic_rule or 'explicit')}",
        ]
        if metadata_keys:
            context_lines.append(f"Metadata keys: {', '.join(metadata_keys)}")
        if from_attrs:
            from_hint = " ".join(
                str(
                    from_attrs.get("summary")
                    or from_attrs.get("description")
                    or from_attrs.get("details")
                    or from_attrs.get("title")
                    or from_attrs.get("name")
                    or ""
                ).split()
            ).strip()
            if from_hint:
                context_lines.append(f"From node content: {from_hint[:500]}")
        if to_attrs:
            to_hint = " ".join(
                str(
                    to_attrs.get("summary")
                    or to_attrs.get("description")
                    or to_attrs.get("details")
                    or to_attrs.get("title")
                    or to_attrs.get("name")
                    or ""
                ).split()
            ).strip()
            if to_hint:
                context_lines.append(f"To node content: {to_hint[:500]}")
        if extra_context:
            context_lines.append(f"User context: {extra_context}")

        default_role = "planner" if action in {"merge", "split"} else "analyst" if action == "risks" else "general"
        result = self.project_archive_verified_chat(
            {
                "user_id": str(root.get("user_id", "default_user") or "default_user").strip() or "default_user",
                "session_id": str(root.get("session_id", "") or "").strip()
                or f"graph_edge_{edge_payload['from_node']}_{edge_payload['to_node']}",
                "message": user_message,
                "context": "\n".join(context_lines),
                "model_path": str(root.get("model_path", "") or "").strip(),
                "model_role": str(root.get("model_role", default_role) or default_role).strip() or default_role,
                "apply_to_graph": self._to_bool(root.get("apply_to_graph", True)),
                "verification_mode": self._pick_allowed_token(
                    root.get("verification_mode", "balanced"),
                    allowed=_ARCHIVE_VERIFICATION_MODES_ALLOWED,
                    default="balanced",
                ),
                "top_k": max(1, min(8, self._to_int(root.get("top_k", 5), 5))),
                "capture_dialect": self._to_bool(root.get("capture_dialect", True)),
                "auto_triage": self._to_bool(root.get("auto_triage", True)),
                "triage_with_llm": self._to_bool(root.get("triage_with_llm", True)),
            }
        )

        graph_binding = self._as_mapping(result.get("graph_binding"))
        session_node_id = self._to_int(graph_binding.get("session_node_id", 0), 0)
        update_node_ids = [
            self._to_int(item, 0)
            for item in self._as_list(graph_binding.get("update_node_ids"))
            if self._to_int(item, 0) > 0
        ]
        apply_to_graph = self._to_bool(root.get("apply_to_graph", True))
        graph_changed = False
        extra_persisted = False
        if apply_to_graph:
            updated_metadata = dict(metadata)
            updated_metadata["last_edge_assist_action"] = action
            updated_metadata["last_edge_assist_at"] = float(time.time())
            updated_metadata["last_edge_assist_summary"] = str(result.get("summary", "") or "")[:600]
            if session_node_id > 0:
                updated_metadata["last_edge_assist_session_node_id"] = session_node_id
            edge.metadata = updated_metadata
            if action == "merge":
                edge.logic_rule = "edge_merge_candidate"
            elif action == "split":
                edge.logic_rule = "edge_split_candidate"
            graph_changed = True

        if apply_to_graph and session_node_id > 0:
            relation_metadata = {
                "action": action,
                "relation_type": edge_payload["relation_type"],
                "direction": edge_payload["direction"],
            }
            self._connect_nodes(
                from_node=edge_payload["from_node"],
                to_node=session_node_id,
                relation_type="requested_edge_assist",
                weight=0.86,
                logic_rule="graph_edge_assist",
                metadata=relation_metadata,
            )
            graph_changed = True
            self._connect_nodes(
                from_node=session_node_id,
                to_node=edge_payload["to_node"],
                relation_type="targets_graph_edge",
                weight=0.82,
                logic_rule="graph_edge_assist",
                metadata=relation_metadata,
            )
            graph_changed = True
            for update_node_id in update_node_ids:
                self._connect_nodes(
                    from_node=update_node_id,
                    to_node=edge_payload["from_node"],
                    relation_type="suggests_edge_change_from",
                    weight=0.8,
                    logic_rule="graph_edge_assist",
                    metadata={"action": action, "relation_type": edge_payload["relation_type"]},
                )
                self._connect_nodes(
                    from_node=update_node_id,
                    to_node=edge_payload["to_node"],
                    relation_type="suggests_edge_change_to",
                    weight=0.8,
                    logic_rule="graph_edge_assist",
                    metadata={"action": action, "relation_type": edge_payload["relation_type"]},
                )
                graph_changed = True
            extra_persisted = bool(self._persist_graph_safe())

        self.api.engine._record_event(  # noqa: SLF001
            "graph_edge_assist_completed",
            {
                "from_node": edge_payload["from_node"],
                "to_node": edge_payload["to_node"],
                "relation_type": edge_payload["relation_type"],
                "direction": edge_payload["direction"],
                "action": action,
                "session_node_id": session_node_id,
                "update_nodes": len(update_node_ids),
                "persisted": bool(result.get("persisted", False) or extra_persisted),
            },
        )

        out = dict(result)
        out["edge_assist"] = {
            "from_node": edge_payload["from_node"],
            "to_node": edge_payload["to_node"],
            "relation_type": edge_payload["relation_type"],
            "direction": edge_payload["direction"],
            "action": action,
            "from_label": from_label,
            "to_label": to_label,
            "session_node_id": session_node_id,
            "update_node_ids": update_node_ids,
        }
        out["persisted"] = bool(result.get("persisted", False) or extra_persisted)
        if graph_changed:
            out.update(self.snapshot_payload())
        execution = self._as_mapping(out.get("execution"))
        if execution:
            execution["persisted"] = bool(out["persisted"])
            execution["status"] = "persisted" if bool(out["persisted"]) else str(
                execution.get("status", "in_memory") or "in_memory"
            )
            out["execution"] = execution
        return out

    def project_graph_foundation_create(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        root = self._as_mapping(payload)
        target_node_id = self._to_int(root.get("target_node_id", 0), 0)
        depth = max(1, min(3, self._to_int(root.get("depth", 2), 2)))
        concept_limit = max(2, min(6, self._to_int(root.get("concept_limit", 4), 4)))
        clean_topic = " ".join(str(root.get("topic", root.get("message", "")) or "").split()).strip()
        clean_context = " ".join(str(root.get("context", "") or "").split()).strip()
        requested_model_path = str(root.get("model_path", "") or "").strip() or _DEFAULT_GRAPH_MONITOR_MODEL_PATH
        resolved_role = self._debate_role(root.get("model_role", "planner"), fallback="planner")

        snapshot = self._as_mapping(self.snapshot_payload().get("snapshot"))
        nodes = [self._as_mapping(row) for row in self._as_list(snapshot.get("nodes"))]
        target_row = next(
            (row for row in nodes if self._to_int(row.get("id", 0), 0) == target_node_id),
            None,
        )
        target_attrs = self._as_mapping(target_row.get("attributes")) if target_row else {}
        target_label = ""
        target_summary = ""
        if target_row:
            target_label = " ".join(
                str(
                    target_attrs.get("title")
                    or target_attrs.get("name")
                    or target_attrs.get("profile_name")
                    or target_attrs.get("summary")
                    or f"Node {target_node_id}"
                ).split()
            ).strip() or f"Node {target_node_id}"
            target_summary = " ".join(
                str(
                    target_attrs.get("summary")
                    or target_attrs.get("description")
                    or target_attrs.get("details")
                    or target_attrs.get("notes")
                    or ""
                ).split()
            ).strip()

        foundation_topic = clean_topic or target_label
        if not foundation_topic:
            raise ValueError("topic is required or select a node to deepen")

        llm_fn, resolution = self._resolve_role_or_model_llm(
            role=resolved_role,
            model_path=requested_model_path,
        )
        raw_output = ""
        parsed_output: dict[str, Any] = {}
        used_fallback = True
        if llm_fn is not None:
            prompt_parts = [
                "You design a compact knowledge-graph foundation for a user workspace.",
                "Return JSON only.",
                'Schema: { "title":"", "summary":"", "concepts":[{ "name":"", "summary":"", "reason":"", "confidence":0.0, "children":[{ "name":"", "summary":"", "reason":"", "confidence":0.0 }] }] }',
                f"Generate up to {concept_limit} top-level concepts and at most {concept_limit} child concepts per node.",
                f"Depth limit: {depth}",
                "Keep names concrete, concise, and useful for graph editing.",
                f"Topic: {foundation_topic[:800]}",
            ]
            if target_label:
                prompt_parts.append(f"Selected node: {target_label}")
            if target_summary:
                prompt_parts.append(f"Selected node summary: {target_summary[:800]}")
            if clean_context:
                prompt_parts.append(f"Context: {clean_context[:1200]}")
            try:
                raw_output = str(llm_fn("\n".join(prompt_parts)) or "").strip()
                parsed_candidate = self._extract_json_from_llm_output(raw_output)
                if isinstance(parsed_candidate, Mapping):
                    parsed_output = dict(parsed_candidate)
                    used_fallback = False
            except Exception:
                parsed_output = {}
                used_fallback = True

        def _normalize_foundation_concepts(value: Any, *, level: int) -> list[dict[str, Any]]:
            rows = value if isinstance(value, list) else []
            out: list[dict[str, Any]] = []
            seen: set[str] = set()
            for row in rows:
                item = self._as_mapping(row)
                if not item and isinstance(row, str):
                    item = {"name": row}
                name = " ".join(
                    str(item.get("name", item.get("title", item.get("label", ""))) or "").split()
                ).strip()
                if not name:
                    continue
                token = name.casefold()
                if token in seen:
                    continue
                seen.add(token)
                summary = " ".join(
                    str(item.get("summary", item.get("description", "")) or "").split()
                ).strip()
                reason = " ".join(str(item.get("reason", "") or "").split()).strip()
                children: list[dict[str, Any]] = []
                if level < depth:
                    child_value = item.get("children", item.get("subconcepts", item.get("details", [])))
                    children = _normalize_foundation_concepts(child_value, level=level + 1)
                out.append(
                    {
                        "name": name[:120],
                        "summary": summary[:360],
                        "reason": reason[:320],
                        "confidence": self._confidence(item.get("confidence", 0.66), 0.66),
                        "children": children[:concept_limit],
                    }
                )
                if len(out) >= concept_limit:
                    break
            return out

        blueprint_title = " ".join(str(parsed_output.get("title", "") or "").split()).strip()
        if not blueprint_title:
            if target_label:
                blueprint_title = self._to_title(
                    f"{target_label} foundation",
                    fallback="Concept foundation",
                    limit=72,
                )
            else:
                blueprint_title = self._to_title(foundation_topic, fallback="Concept foundation", limit=72)
        blueprint_summary = " ".join(str(parsed_output.get("summary", "") or "").split()).strip()
        if not blueprint_summary:
            if clean_context:
                blueprint_summary = self._to_title(clean_context, fallback="Structured concept foundation", limit=180)
            elif target_summary:
                blueprint_summary = self._to_title(target_summary, fallback="Structured concept foundation", limit=180)
            else:
                blueprint_summary = self._to_title(
                    f"Structured concept foundation for {foundation_topic}",
                    fallback="Structured concept foundation",
                    limit=180,
                )

        concepts = _normalize_foundation_concepts(parsed_output.get("concepts", []), level=1)
        if not concepts:
            source_text = "\n".join(part for part in (foundation_topic, clean_context, target_summary) if part).strip()
            raw_chunks = [
                " ".join(chunk.split()).strip()
                for chunk in re.split(r"[\n;,|]+", source_text)
                if " ".join(str(chunk or "").split()).strip()
            ]
            name_candidates = [
                chunk
                for chunk in raw_chunks
                if 3 <= len(chunk) <= 80
            ]
            if len(name_candidates) < concept_limit:
                fallback_words = [
                    token
                    for token in re.findall(r"[\w-]{4,}", source_text, flags=re.UNICODE)
                    if len(token) >= 4
                ]
                for token in fallback_words:
                    if len(name_candidates) >= concept_limit:
                        break
                    if any(token.casefold() == item.casefold() for item in name_candidates):
                        continue
                    name_candidates.append(token)
            if not name_candidates:
                name_candidates = ["core concept", "key detail", "next step"][:concept_limit]
            concepts = []
            for index, name in enumerate(name_candidates[:concept_limit], start=1):
                child_rows: list[dict[str, Any]] = []
                if depth > 1:
                    child_rows.append(
                        {
                            "name": f"{name} detail",
                            "summary": "Fallback deeper layer added because LLM structure was unavailable.",
                            "reason": "fallback_depth_expansion",
                            "confidence": 0.52,
                            "children": [],
                        }
                    )
                concepts.append(
                    {
                        "name": self._to_title(name, fallback=f"Concept {index}", limit=120),
                        "summary": f"Fallback concept extracted from the foundation request #{index}.",
                        "reason": "fallback_concept_extraction",
                        "confidence": 0.56,
                        "children": child_rows,
                    }
                )
            used_fallback = True

        selected_model_path = str(resolution.get("selected_model_path", "") or requested_model_path)
        root_node = self.api.engine.create_node(
            "foundation_branch",
            attributes={
                "title": blueprint_title,
                "summary": blueprint_summary,
                "topic": foundation_topic,
                "context": clean_context,
                "source": "graph_foundation_builder",
                "depth": depth,
                "concept_limit": concept_limit,
                "target_node_id": target_node_id if target_node_id > 0 else 0,
                "target_label": target_label,
                "requested_model_path": requested_model_path,
                "selected_model_path": selected_model_path,
                "requested_role": resolved_role,
                "resolution": dict(resolution),
                "used_fallback": bool(used_fallback),
            },
            state={
                "depth": float(depth),
                "concept_limit": float(concept_limit),
                "confidence": 0.74 if not used_fallback else 0.56,
            },
        )
        root_node_id = int(root_node.id)
        created_node_ids: list[int] = [root_node_id]
        created_edge_count = 0

        if target_node_id > 0 and target_row is not None:
            self._connect_nodes(
                from_node=target_node_id,
                to_node=root_node_id,
                relation_type="expands_concept",
                weight=0.86,
                logic_rule="graph_foundation_builder",
                metadata={"topic": foundation_topic, "depth": depth},
            )
            created_edge_count += 1

        def _create_concept_branch(parent_node_id: int, concept_rows: list[dict[str, Any]], *, level: int) -> None:
            nonlocal created_edge_count
            for item in concept_rows[:concept_limit]:
                concept_node = self.api.engine.create_node(
                    "concept",
                    attributes={
                        "name": str(item.get("name", "") or "").strip(),
                        "summary": str(item.get("summary", "") or "").strip(),
                        "reason": str(item.get("reason", "") or "").strip(),
                        "foundation_root_id": root_node_id,
                        "foundation_topic": foundation_topic,
                        "concept_level": level,
                    },
                    state={
                        "confidence": self._confidence(item.get("confidence", 0.62), 0.62),
                        "concept_level": float(level),
                    },
                )
                concept_node_id = int(concept_node.id)
                created_node_ids.append(concept_node_id)
                self._connect_nodes(
                    from_node=parent_node_id,
                    to_node=concept_node_id,
                    relation_type="contains_concept" if level == 1 else "deepens_concept",
                    weight=max(0.5, 0.9 - (0.08 * max(0, level - 1))),
                    logic_rule="graph_foundation_builder",
                    metadata={"level": level, "root_node_id": root_node_id},
                )
                created_edge_count += 1
                children = [
                    self._as_mapping(child)
                    for child in self._as_list(self._as_mapping(item).get("children"))
                ]
                if children and level < depth:
                    _create_concept_branch(concept_node_id, children, level=level + 1)

        _create_concept_branch(root_node_id, concepts, level=1)

        monitor_focus_ids = [root_node_id, *created_node_ids[1:]]
        if target_node_id > 0:
            monitor_focus_ids.append(target_node_id)
        graph_monitor = self._run_graph_monitor(
            user_id=str(root.get("user_id", "default_user") or "default_user").strip() or "default_user",
            session_id=str(root.get("session_id", "") or "").strip() or f"graph_foundation_{root_node_id}",
            source="graph_foundation_builder",
            focus_node_ids=monitor_focus_ids,
            hint_text=f"{blueprint_summary}\n{foundation_topic}",
            model_path=selected_model_path,
            model_role="planner",
            apply_changes=True,
        )
        persisted = self._persist_graph_safe()
        execution = self._execution_status(
            action="graph_foundation_create",
            persisted=persisted,
            graph_monitor=graph_monitor,
            extra={
                "created_nodes": len(created_node_ids),
                "created_edges": created_edge_count,
                "focus_node_id": root_node_id,
            },
        )

        self.api.engine._record_event(  # noqa: SLF001
            "graph_foundation_created",
            {
                "root_node_id": root_node_id,
                "target_node_id": target_node_id,
                "created_nodes": len(created_node_ids),
                "created_edges": created_edge_count,
                "top_level_concepts": len(concepts),
                "depth": depth,
                "persisted": bool(persisted),
            },
        )

        return {
            "ok": True,
            "foundation": {
                "title": blueprint_title,
                "summary": blueprint_summary,
                "topic": foundation_topic,
                "root_node_id": root_node_id,
                "focus_node_id": root_node_id,
                "target_node_id": target_node_id,
                "target_label": target_label,
                "created_node_ids": created_node_ids,
                "created_nodes": len(created_node_ids),
                "created_edges": created_edge_count,
                "top_level_concepts": len(concepts),
                "depth": depth,
                "concept_limit": concept_limit,
            },
            "summary": {
                "title": blueprint_title,
                "summary": blueprint_summary,
                "created_nodes": len(created_node_ids),
                "created_edges": created_edge_count,
                "focus_node_id": root_node_id,
                "model_path": selected_model_path,
            },
            "blueprint": {
                "title": blueprint_title,
                "summary": blueprint_summary,
                "concepts": concepts,
            },
            "graph_monitor": graph_monitor,
            "model": {
                "requested_model_path": requested_model_path,
                "selected_model_path": selected_model_path,
                "requested_role": resolved_role,
                "resolution": dict(resolution),
                "used_fallback": bool(used_fallback),
            },
            "execution": execution,
            "persisted": bool(persisted),
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
