"""
Personality update parser.

Given a text input (chat message, document excerpt, graph extraction result,
tool output, structured follow-up), classify its content into a list of
UpdateCandidate objects ready for the merge engine.

Classification categories:
    profile_field      → a psychological trait, fear, goal, defense, etc.
    biography_fact     → an explicit factual statement about the person's life
    temporary_state    → a transient emotion / situation reported right now
    uncertain_inference → tentative pattern, low confidence, not yet stable
    rejected_noise     → irrelevant / too vague / system noise

Rules that MUST hold:
    - "I am afraid of being used."         → profile_field (fears)
    - "I studied at YSU."                  → biography_fact (education)
    - "Yesterday I argued with X."         → biography_fact (life_event) OR temporary_state
    - "I am upset right now."              → temporary_state
    - "Maybe he feels jealous."            → uncertain_inference
    - Generic greetings / noise            → rejected_noise

The parser is deterministic and does NOT call the LLM.
It uses a cascading rule set with keyword patterns.
Each candidate carries the exact evidence excerpt and classification reason.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from .personality_schema import UpdateCandidate


# ---------------------------------------------------------------------------
# Source channels
# ---------------------------------------------------------------------------

SOURCE_CHANNEL_CHAT_USER = 'chat_user'
SOURCE_CHANNEL_CHAT_ASSISTANT = 'chat_assistant'
SOURCE_CHANNEL_UPLOADED_DOCUMENT = 'uploaded_document'
SOURCE_CHANNEL_GRAPH_EXTRACTION = 'graph_extraction'
SOURCE_CHANNEL_MANUAL_EDIT = 'manual_edit'
SOURCE_CHANNEL_PERSONA_IMPORT = 'persona_import'
SOURCE_CHANNEL_TOOL_OUTPUT = 'tool_output'


# ---------------------------------------------------------------------------
# Temporary state patterns
# A sentence qualifies as "temporary" if it is anchored to right-now / today.
# ---------------------------------------------------------------------------

_TEMP_ANCHORS: tuple[str, ...] = (
    r'\bright now\b',
    r'\bat this moment\b',
    r'\btoday\s+i\b',
    r'\btoday\s+he\b',
    r'\btoday\s+she\b',
    r'\btoday\s+they\b',
    r'\bcurrently\s+feeling\b',
    r'\bi(?:\'m| am)\s+(?:feeling|upset|angry|sad|happy|anxious|stressed|worried|excited|nervous)\b',
    r'\bhe(?:\'s| is)\s+(?:feeling|upset|angry|sad|happy|anxious|stressed|worried)\b',
    r'\bshe(?:\'s| is)\s+(?:feeling|upset|angry|sad|happy|anxious|stressed|worried)\b',
    r'\bjust\s+now\b',
    r'\bthis\s+session\b',
    r'\btoday\b.{0,20}\b(?:upset|angry|sad|stressed|anxious|nervous|excited|distressed)\b',
    # Russian
    r'\bсейчас\b',
    r'\bв данный момент\b',
    r'\bсегодня\s+(?:расстроен|злюсь|грущу|тревожусь)\b',
)

_TEMP_ANCHOR_RE: list[re.Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in _TEMP_ANCHORS]

_TRANSIENT_EMOTION_WORDS: frozenset[str] = frozenset({
    'upset', 'stressed', 'distressed', 'panicking', 'furious', 'terrified',
    'overjoyed', 'elated', 'devastated', 'overwhelmed', 'panicked',
    'расстроен', 'злюсь', 'грущу', 'тревожусь', 'паникую',
})


def _is_temporary_state(text: str) -> tuple[bool, str]:
    lowered = text.strip().lower()
    for pattern in _TEMP_ANCHOR_RE:
        m = pattern.search(lowered)
        if m:
            return True, f'temporal anchor matched: {m.group(0)!r}'
    for word in _TRANSIENT_EMOTION_WORDS:
        if word in lowered:
            # Only if no stable bio markers present — handled below in cascade
            return True, f'transient emotion word: {word!r}'
    return False, ''


# ---------------------------------------------------------------------------
# Uncertain inference patterns
# A statement is "uncertain" if it uses hedged language.
# ---------------------------------------------------------------------------

_UNCERTAIN_PATTERNS: tuple[str, ...] = (
    r'\bmaybe\b',
    r'\bperhaps\b',
    r'\bpossibly\b',
    r'\bprobably\b',
    r'\bmight\s+(?:be|feel|have)\b',
    r'\bcould\s+(?:be|feel|have)\b',
    r'\bseems\s+(?:to\s+)?(?:be|like|afraid|scared)\b',
    r'\bappears\s+to\b',
    r'\bI think\b',
    r'\bI suspect\b',
    r'\bI guess\b',
    r'\bI believe\b',
    r'\bit\s+(?:seems|appears|looks)\s+like\b',
    r'\bcould\s+indicate\b',
    r'\bmay\s+suggest\b',
    # Russian
    r'\bможет\s+быть\b',
    r'\bвозможно\b',
    r'\bнаверное\b',
    r'\bскорее\s+всего\b',
    r'\bкажется\b',
    r'\bдумаю,?\s+что\b',
)

_UNCERTAIN_RE: list[re.Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in _UNCERTAIN_PATTERNS]


def _is_uncertain(text: str) -> tuple[bool, str]:
    for pattern in _UNCERTAIN_RE:
        m = pattern.search(text)
        if m:
            return True, f'hedge phrase: {m.group(0)!r}'
    return False, ''


# ---------------------------------------------------------------------------
# Biography fact patterns
# ---------------------------------------------------------------------------

# Maps a regex to (fact_type, confidence_multiplier)
_BIO_PATTERNS: tuple[tuple[str, str, float], ...] = (
    # Education
    (r'\b(?:studied|study|studying|graduated|attended|enrolled)\s+(?:at|in|from)\b', 'education', 0.92),
    (r'\b(?:university|college|school|institute|academy|курс|университет|институт|школа)\b', 'education', 0.82),
    (r'\bдиплом\b|\bстепень\b|\bобразование\b', 'education', 0.82),
    # Occupation / work
    (r'\b(?:work|worked|working|job|employed|occupation|profession|career|profession|freelance)\b', 'occupation', 0.82),
    (r'\b(?:i am|i\'m|i was)\s+a\s+\w+\b', 'occupation', 0.75),
    (r'\bработ(?:аю|ал|аю\s+в|ал\s+в)\b', 'occupation', 0.82),
    (r'\b(?:engineer|developer|designer|doctor|teacher|lawyer|nurse|manager|director|analyst)\b', 'occupation', 0.80),
    # Location
    (r'\b(?:live|lived|living|moved|relocated|born|grew up|reside)\s+(?:in|at|to)\b', 'location', 0.88),
    (r'\b(?:city|town|country|village|region|district)\b', 'location', 0.75),
    (r'\bживу\b|\bжил\b|\bродился\b|\bпереехал\b', 'location', 0.85),
    # Family
    (r'\b(?:my\s+(?:mother|father|sister|brother|son|daughter|wife|husband|partner|spouse|family|parents|grandparent|uncle|aunt|cousin))\b', 'family', 0.90),
    (r'\b(?:моя?\s+(?:мать|мама|отец|папа|сестра|брат|сын|дочь|жена|муж|семья|дети))\b', 'family', 0.90),
    # Relationships
    (r'\b(?:my\s+(?:friend|colleague|boss|mentor|partner|ex|boyfriend|girlfriend|lover))\b', 'relationship', 0.85),
    # Life events
    (r'\b(?:was born|born in|birth|death|married|divorced|graduated|retired|promoted|fired|hired)\b', 'life_event', 0.88),
    (r'\b(?:родился|умер|поженился|развелся|окончил|вышел на пенсию)\b', 'life_event', 0.88),
    (r'\b(?:accident|surgery|illness|hospital|diagnosis|diagnosed)\b', 'life_event', 0.85),
    # Achievements
    (r'\b(?:won|award|prize|medal|certificate|published|launched|founded|built|created)\b', 'achievement', 0.80),
    # Timeline entries (years or dates)
    (r'\b(?:in\s+(?:19|20)\d{2}|(?:19|20)\d{2}\s+(?:was|were|i|he|she))\b', 'timeline_entry', 0.85),
    (r'\b(?:в|в\s+году)\s+(?:19|20)\d{2}\b', 'timeline_entry', 0.85),
    # Identity facts
    (r'\b(?:my\s+name\s+is|i\s+am\s+called|they\s+call\s+me)\b', 'identity_fact', 0.95),
    (r'\b(?:меня\s+зовут|моё\s+имя)\b', 'identity_fact', 0.95),
    # Stated preferences (borderline biography / profile)
    (r'\b(?:i\s+(?:love|hate|enjoy|like|dislike|prefer|always|never)\s+\w+)\b', 'stated_preference', 0.75),
    (r'\b(?:я\s+(?:люблю|ненавижу|обожаю|не\s+люблю|предпочитаю)\s+\w+)\b', 'stated_preference', 0.75),
)

_BIO_COMPILED: list[tuple[re.Pattern[str], str, float]] = [
    (re.compile(p, re.IGNORECASE), ftype, conf)
    for p, ftype, conf in _BIO_PATTERNS
]


def _detect_biography_fact(text: str) -> tuple[bool, str, float]:
    """Returns (matched, fact_type, confidence)."""
    lowered = text.strip()
    for pattern, fact_type, conf in _BIO_COMPILED:
        if pattern.search(lowered):
            return True, fact_type, conf
    return False, 'factual_statement', 0.6


# ---------------------------------------------------------------------------
# Psychological profile field patterns
# ---------------------------------------------------------------------------

# Maps regex → profile field name
_PROFILE_PATTERNS: tuple[tuple[str, str], ...] = (
    # fears
    (r'\b(?:afraid\s+of|fear\s+of|fears?\s+\w+|scared\s+of|terrified\s+of|дрожу\s+перед|боится|боюсь|страшусь)\b', 'fears'),
    (r'\b(?:i am afraid|i fear|i\'m scared)\b', 'fears'),
    # needs
    (r'\b(?:needs?\s+to|i\s+need|he\s+needs|she\s+needs|нуждается\s+в|мне\s+нужно)\b', 'needs'),
    # core_goals
    (r'\b(?:my\s+(?:main\s+)?goal\s+is|i\s+want\s+to\s+achieve|strives?\s+to|primary\s+goal)\b', 'core_goals'),
    (r'\b(?:main\s+purpose|life\s+goal|driving\s+goal)\b', 'core_goals'),
    # secondary_goals
    (r'\b(?:also\s+wants?\s+to|secondary\s+goal|wishes?\s+to|hopes?\s+to)\b', 'secondary_goals'),
    # desires
    (r'\b(?:deeply\s+desires?|longs?\s+for|yearns?\s+for|craves?|мечтает\s+о)\b', 'desires'),
    # values
    (r'\b(?:values?\s+\w+|believes?\s+in|principles?\s+of|морально|ценит)\b', 'values'),
    # shame_points
    (r'\b(?:ashamed\s+of|shame\s+about|humiliated\s+by|embarrassed\s+by|стыдится|стыдно\s+за)\b', 'shame_points'),
    # vulnerabilities
    (r'\b(?:vulnerable\s+(?:to|when)|weakness\s+is|weak\s+point|ахиллесова\s+пята)\b', 'vulnerabilities'),
    # internal_conflicts
    (r'\b(?:torn\s+between|conflicted\s+about|inner\s+conflict|part\s+of\s+me\s+wants|внутренний\s+конфликт)\b', 'internal_conflicts'),
    (r'\b(?:wants\s+.{3,30}\s+but\s+(?:also|at\s+the\s+same\s+time)\s+(?:fears?|avoids?))\b', 'internal_conflicts'),
    # constraints_internal
    (r'\b(?:cannot\s+bring\s+(?:myself|himself|herself)\s+to|refuses?\s+to|won\'t\s+allow\s+(?:myself|himself|herself))\b', 'constraints_internal'),
    # constraints_social
    (r'\b(?:social\s+constraint|expected\s+to|role\s+demands?|society\s+(?:expects?|requires?))\b', 'constraints_social'),
    # defense_mechanisms
    (r'\b(?:deflects?|rationalizes?|intellectualizes?|projects?|denies?|suppresses?|displaced|defense\s+mechanism)\b', 'defense_mechanisms'),
    (r'\b(?:defends?\s+(?:by|through|with)|protects?\s+(?:by|through|with))\b', 'defense_mechanisms'),
    # communication_style
    (r'\b(?:communicates?\s+(?:by|through|in\s+a)|speaks?\s+in\s+a|tends?\s+to\s+(?:speak|talk|communicate))\b', 'communication_style'),
    (r'\b(?:terse|verbose|indirect|direct|guarded|open)\s+(?:communicat|speak|talk|style|manner)\b', 'communication_style'),
    # attachment_patterns
    (r'\b(?:attachment\s+(?:style|pattern)|anxious\s+attachment|avoidant|fearful\s+attachment)\b', 'attachment_patterns'),
    (r'\b(?:clings?\s+to|pushes?\s+away|needs?\s+(?:closeness|distance)|тянется\s+к|отстраняется)\b', 'attachment_patterns'),
    # triggers
    (r'\b(?:triggered\s+by|trigger\s+is|gets?\s+triggered|provoked\s+by|реагирует\s+на|триггер)\b', 'triggers'),
    (r'\b(?:when\s+(?:someone|people)\s+\w+,?\s+(?:he|she|they)\s+(?:becomes?|gets?|feels?))\b', 'triggers'),
    # pressure_response
    (r'\b(?:under\s+pressure|when\s+pressured|responds?\s+to\s+pressure|под\s+давлением)\b', 'pressure_response'),
    (r'\b(?:withdraws?\s+when|shuts?\s+down\s+when|lashes?\s+out\s+when)\b', 'pressure_response'),
    # maladaptive_methods
    (r'\b(?:maladaptive|unhealthy\s+coping|self.?destruct|sabotages?|avoids?\s+by|escapes?\s+through)\b', 'maladaptive_methods'),
    # allowed_methods
    (r'\b(?:healthy\s+coping|constructive\s+way|good\s+approach|allowed\s+to)\b', 'allowed_methods'),
    # growth_pattern
    (r'\b(?:grows?\s+when|learns?\s+to|over\s+time\s+(?:he|she|they)|development\s+pattern)\b', 'growth_pattern'),
)

_PROFILE_COMPILED: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), field_name)
    for p, field_name in _PROFILE_PATTERNS
]


def _detect_profile_field(text: str) -> tuple[bool, str]:
    """Returns (matched, field_name)."""
    for pattern, field_name in _PROFILE_COMPILED:
        if pattern.search(text):
            return True, field_name
    return False, ''


# ---------------------------------------------------------------------------
# Noise / rejection patterns
# ---------------------------------------------------------------------------

_NOISE_PATTERNS: tuple[str, ...] = (
    r'^(?:hi|hello|hey|ok|okay|yes|no|sure|thanks?|thank\s+you|bye|goodbye)\s*[.!]?\s*$',
    r'^(?:привет|здравствуй|ок|окей|да|нет|спасибо|до\s+свидания)\s*[.!]?\s*$',
    r'^\s*$',
)

_NOISE_RE: list[re.Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in _NOISE_PATTERNS]


def _is_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) < 4:
        return True
    for pattern in _NOISE_RE:
        if pattern.match(stripped):
            return True
    return False


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences; also handle line breaks."""
    lines = str(text or '').strip().splitlines()
    sentences: list[str] = []
    for line in lines:
        for sent in _SENTENCE_SPLIT_RE.split(line.strip()):
            clean = sent.strip()
            if clean:
                sentences.append(clean)
    return sentences if sentences else [text.strip()]


# ---------------------------------------------------------------------------
# Main parser entry point
# ---------------------------------------------------------------------------

def parse_update_candidates(
    text: str,
    *,
    source_channel: str = SOURCE_CHANNEL_CHAT_USER,
    session_id: str = '',
    timestamp: str = '',
    confidence_override: float | None = None,
) -> list[UpdateCandidate]:
    """
    Parse a text block into classified UpdateCandidate objects.

    Each sentence is evaluated independently.  A sentence can produce at most
    one candidate.  The classification cascade is:

        1. is_noise?              → rejected_noise
        2. is_temporary_state?    → temporary_state  (checked BEFORE profile, so
                                     "today I am afraid" → temp, not fears)
        3. is_biography_fact?     → biography_fact
        4. is_profile_field?      → profile_field (or uncertain_inference if hedged)
        5. is_uncertain?          → uncertain_inference  (fallback catch)
        6. default                → rejected_noise

    Arguments:
        text             — the input text to parse
        source_channel   — where this came from
        session_id       — current session identifier
        timestamp        — ISO timestamp string; generated if empty
        confidence_override — override the confidence for all candidates
    """
    if not timestamp:
        timestamp = datetime.now(UTC).isoformat()

    sentences = _split_sentences(str(text or ''))
    candidates: list[UpdateCandidate] = []

    for sentence in sentences:
        candidate = _classify_sentence(
            sentence,
            source_channel=source_channel,
            session_id=session_id,
            timestamp=timestamp,
            confidence_override=confidence_override,
        )
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def _classify_sentence(
    sentence: str,
    *,
    source_channel: str,
    session_id: str,
    timestamp: str,
    confidence_override: float | None,
) -> UpdateCandidate | None:
    """Classify a single sentence into an UpdateCandidate (or None for rejected)."""
    stripped = sentence.strip()
    if not stripped:
        return None

    # --- Step 1: Noise check ---
    if _is_noise(stripped):
        return None  # Drop silently; no candidate for pure noise

    # --- Step 2: Temporary state ---
    is_temp, temp_reason = _is_temporary_state(stripped)
    if is_temp:
        return UpdateCandidate(
            update_class='temporary_state',
            value=stripped,
            confidence=confidence_override if confidence_override is not None else 0.75,
            source_channel=source_channel,
            session_id=session_id,
            timestamp=timestamp,
            evidence_excerpt=stripped,
            reason=f'temporary_state: {temp_reason}',
        )

    # --- Step 3: Biography fact ---
    bio_matched, fact_type, bio_conf = _detect_biography_fact(stripped)
    if bio_matched:
        uncertain, uncertain_reason = _is_uncertain(stripped)
        final_conf = confidence_override if confidence_override is not None else bio_conf
        if uncertain:
            final_conf = min(final_conf, 0.5)
            return UpdateCandidate(
                update_class='uncertain_inference',
                value=stripped,
                confidence=final_conf,
                source_channel=source_channel,
                session_id=session_id,
                timestamp=timestamp,
                evidence_excerpt=stripped,
                is_uncertain=True,
                fact_type=fact_type,
                reason=f'biography_with_hedge ({uncertain_reason})',
            )
        return UpdateCandidate(
            update_class='biography_fact',
            fact_type=fact_type,
            value=stripped,
            confidence=final_conf,
            source_channel=source_channel,
            session_id=session_id,
            timestamp=timestamp,
            evidence_excerpt=stripped,
            reason=f'biography_fact: {fact_type}',
        )

    # --- Step 4: Psychological profile field ---
    prof_matched, field_name = _detect_profile_field(stripped)
    if prof_matched:
        uncertain, uncertain_reason = _is_uncertain(stripped)
        final_conf = confidence_override if confidence_override is not None else (0.55 if uncertain else 0.80)
        return UpdateCandidate(
            update_class='uncertain_inference' if uncertain else 'profile_field',
            field_name=field_name,
            value=stripped,
            confidence=final_conf,
            source_channel=source_channel,
            session_id=session_id,
            timestamp=timestamp,
            evidence_excerpt=stripped,
            is_uncertain=uncertain,
            reason=f'profile_field:{field_name}' + (f' (uncertain: {uncertain_reason})' if uncertain else ''),
        )

    # --- Step 5: Catch-all uncertain (hedged text without clear field) ---
    uncertain, uncertain_reason = _is_uncertain(stripped)
    if uncertain and len(stripped) > 12:
        return UpdateCandidate(
            update_class='uncertain_inference',
            value=stripped,
            confidence=confidence_override if confidence_override is not None else 0.40,
            source_channel=source_channel,
            session_id=session_id,
            timestamp=timestamp,
            evidence_excerpt=stripped,
            is_uncertain=True,
            reason=f'uncertain_inference: {uncertain_reason}',
        )

    # --- Default: reject ---
    return None


# ---------------------------------------------------------------------------
# Document / tool extraction helper
# ---------------------------------------------------------------------------

def parse_document_candidates(
    text: str,
    *,
    session_id: str = '',
    timestamp: str = '',
) -> list[UpdateCandidate]:
    """
    Parse a document or tool output for biography facts.

    Documents are trusted more than chat for biography facts (0.88 default confidence).
    Psychological profile fields from documents are stored as uncertain_inference
    unless they use definitive language.
    """
    return parse_update_candidates(
        text,
        source_channel=SOURCE_CHANNEL_UPLOADED_DOCUMENT,
        session_id=session_id,
        timestamp=timestamp,
    )


def parse_graph_extraction_candidates(
    text: str,
    *,
    session_id: str = '',
    timestamp: str = '',
) -> list[UpdateCandidate]:
    """
    Parse graph extraction output for personality updates.

    Graph extractions targeting psychological patterns come in as
    uncertain_inference unless they have high evidence support.
    """
    candidates = parse_update_candidates(
        text,
        source_channel=SOURCE_CHANNEL_GRAPH_EXTRACTION,
        session_id=session_id,
        timestamp=timestamp,
    )
    # Graph-extracted psychological fields are always uncertain until confirmed
    for c in candidates:
        if c.update_class == 'profile_field':
            c.update_class = 'uncertain_inference'
            c.is_uncertain = True
            c.confidence = min(c.confidence, 0.60)
            c.reason = f'graph_extraction_as_uncertain: {c.reason}'
    return candidates


# ---------------------------------------------------------------------------
# Structured field update helper (for direct programmatic updates)
# ---------------------------------------------------------------------------

def build_direct_profile_update(
    field_name: str,
    value: str,
    *,
    source_channel: str = SOURCE_CHANNEL_MANUAL_EDIT,
    session_id: str = '',
    confidence: float = 0.90,
    evidence_excerpt: str = '',
) -> UpdateCandidate:
    """
    Build a high-confidence direct update candidate for a named profile field.

    Used for structured follow-up corrections or operator-provided updates
    where the field is known and the value is explicit.
    """
    return UpdateCandidate(
        update_class='profile_field',
        field_name=str(field_name or ''),
        value=str(value or ''),
        confidence=confidence,
        source_channel=source_channel,
        session_id=session_id,
        timestamp=datetime.now(UTC).isoformat(),
        evidence_excerpt=evidence_excerpt or value,
        is_uncertain=False,
        reason='direct_structured_update',
    )


def build_direct_biography_update(
    fact_type: str,
    value: str,
    *,
    source_channel: str = SOURCE_CHANNEL_MANUAL_EDIT,
    session_id: str = '',
    confidence: float = 0.90,
    evidence_excerpt: str = '',
) -> UpdateCandidate:
    """
    Build a high-confidence direct biography fact candidate.

    Used for operator-provided or document-derived facts when fact_type is known.
    """
    return UpdateCandidate(
        update_class='biography_fact',
        fact_type=str(fact_type or 'factual_statement'),
        value=str(value or ''),
        confidence=confidence,
        source_channel=source_channel,
        session_id=session_id,
        timestamp=datetime.now(UTC).isoformat(),
        evidence_excerpt=evidence_excerpt or value,
        is_uncertain=False,
        reason='direct_biography_update',
    )
