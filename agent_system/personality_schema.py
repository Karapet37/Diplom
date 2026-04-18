"""
Personality construction schema.

This module defines the structured data models for the personality construction
system.  A personality is NOT a text blob.  It is a layered object with:

    A.  identity         — who the person is
    B.  profile          — typed psychological fields with provenance
    C.  biography        — explicit factual life information (separate from inferences)
    D.  state_meta       — system metadata, confidence, completeness

Critical distinction:
    - profile fields   = psychological interpretation (fears, defenses, goals, …)
    - biography facts  = explicit factual statements (education, job, relationships, …)
    - temporary state  = session-scoped emotion / transient report (never stored in stable profile)
    - uncertain inference = low-confidence pattern, not elevated to stable truth until repeated

None of these four categories must ever be collapsed into a single list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

SourceChannel = Literal[
    'chat_user',
    'chat_assistant',
    'uploaded_document',
    'graph_extraction',
    'manual_edit',
    'persona_import',
    'tool_output',
]

UpdateClass = Literal[
    'profile_field',
    'biography_fact',
    'temporary_state',
    'uncertain_inference',
    'rejected_noise',
]

ProfileFieldName = Literal[
    'core_goals',
    'secondary_goals',
    'desires',
    'fears',
    'needs',
    'values',
    'shame_points',
    'vulnerabilities',
    'internal_conflicts',
    'constraints_internal',
    'constraints_social',
    'constraints_hard_system',
    'allowed_methods',
    'maladaptive_methods',
    'defense_mechanisms',
    'communication_style',
    'attachment_patterns',
    'refusal_style',
    'triggers',
    'pressure_response',
    'growth_pattern',
]

PROFILE_FIELD_NAMES: tuple[str, ...] = (
    'core_goals',
    'secondary_goals',
    'desires',
    'fears',
    'needs',
    'values',
    'shame_points',
    'vulnerabilities',
    'internal_conflicts',
    'constraints_internal',
    'constraints_social',
    'constraints_hard_system',
    'allowed_methods',
    'maladaptive_methods',
    'defense_mechanisms',
    'communication_style',
    'attachment_patterns',
    'refusal_style',
    'triggers',
    'pressure_response',
    'growth_pattern',
)

BiographyFactType = Literal[
    'education',
    'occupation',
    'location',
    'family',
    'relationship',
    'life_event',
    'identity_fact',
    'experience',
    'achievement',
    'loss',
    'timeline_entry',
    'document_fact',
    'stated_preference',
    'factual_statement',
]

BIOGRAPHY_FACT_TYPES: tuple[str, ...] = (
    'education',
    'occupation',
    'location',
    'family',
    'relationship',
    'life_event',
    'identity_fact',
    'experience',
    'achievement',
    'loss',
    'timeline_entry',
    'document_fact',
    'stated_preference',
    'factual_statement',
)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceRecord:
    """Tracks where a field entry or fact came from."""

    source: str = 'chat_user'          # SourceChannel value
    source_channel: str = 'chat_user'  # free-text label (same as source normally)
    session_id: str = ''
    timestamp: str = ''
    evidence_excerpt: str = ''         # Verbatim text that produced this entry

    def to_dict(self) -> dict[str, Any]:
        return {
            'source': self.source,
            'source_channel': self.source_channel,
            'session_id': self.session_id,
            'timestamp': self.timestamp,
            'evidence_excerpt': self.evidence_excerpt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ProvenanceRecord':
        return cls(
            source=str(data.get('source') or 'chat_user'),
            source_channel=str(data.get('source_channel') or 'chat_user'),
            session_id=str(data.get('session_id') or ''),
            timestamp=str(data.get('timestamp') or ''),
            evidence_excerpt=str(data.get('evidence_excerpt') or ''),
        )


# ---------------------------------------------------------------------------
# Conflict record
# ---------------------------------------------------------------------------

@dataclass
class ConflictRecord:
    """Records a detected contradiction between two facts/entries."""

    conflict_id: str
    conflicting_fact_id: str        # ID of the other fact/entry that contradicts
    description: str = ''
    resolved: bool = False
    resolution: str = ''            # How it was resolved (if at all)

    def to_dict(self) -> dict[str, Any]:
        return {
            'conflict_id': self.conflict_id,
            'conflicting_fact_id': self.conflicting_fact_id,
            'description': self.description,
            'resolved': bool(self.resolved),
            'resolution': self.resolution,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ConflictRecord':
        return cls(
            conflict_id=str(data.get('conflict_id') or ''),
            conflicting_fact_id=str(data.get('conflicting_fact_id') or ''),
            description=str(data.get('description') or ''),
            resolved=bool(data.get('resolved', False)),
            resolution=str(data.get('resolution') or ''),
        )


# ---------------------------------------------------------------------------
# Profile field entry
# ---------------------------------------------------------------------------

@dataclass
class ProfileFieldEntry:
    """
    A single item within a psychological profile field.

    Carries its own provenance so we always know where it came from.
    Distinguishes stable truths from uncertain inferences.
    """

    entry_id: str
    value: str
    confidence: float = 0.8
    provenance: list[ProvenanceRecord] = field(default_factory=list)
    is_uncertain: bool = False  # True → inference not yet stable
    is_temporary: bool = False  # True → session-scoped, not persisted to stable profile
    conflicts: list[ConflictRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'entry_id': self.entry_id,
            'value': self.value,
            'confidence': round(float(self.confidence or 0.8), 6),
            'provenance': [p.to_dict() for p in self.provenance],
            'is_uncertain': bool(self.is_uncertain),
            'is_temporary': bool(self.is_temporary),
            'conflicts': [c.to_dict() for c in self.conflicts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ProfileFieldEntry':
        return cls(
            entry_id=str(data.get('entry_id') or ''),
            value=str(data.get('value') or ''),
            confidence=float(data.get('confidence') or 0.8),
            provenance=[ProvenanceRecord.from_dict(p) for p in list(data.get('provenance') or [])],
            is_uncertain=bool(data.get('is_uncertain', False)),
            is_temporary=bool(data.get('is_temporary', False)),
            conflicts=[ConflictRecord.from_dict(c) for c in list(data.get('conflicts') or [])],
        )


# ---------------------------------------------------------------------------
# Psychological profile
# ---------------------------------------------------------------------------

@dataclass
class PersonalityProfile:
    """
    Structured psychological profile.

    Every field is a list of ProfileFieldEntry — not raw strings — so that
    provenance is preserved per-item and uncertain entries can be distinguished
    from stable truths.

    RULE: Do not mix biography facts into these fields.
    RULE: Do not store temporary session state in these fields.
    """

    core_goals: list[ProfileFieldEntry] = field(default_factory=list)
    secondary_goals: list[ProfileFieldEntry] = field(default_factory=list)
    desires: list[ProfileFieldEntry] = field(default_factory=list)
    fears: list[ProfileFieldEntry] = field(default_factory=list)
    needs: list[ProfileFieldEntry] = field(default_factory=list)
    values: list[ProfileFieldEntry] = field(default_factory=list)
    shame_points: list[ProfileFieldEntry] = field(default_factory=list)
    vulnerabilities: list[ProfileFieldEntry] = field(default_factory=list)
    internal_conflicts: list[ProfileFieldEntry] = field(default_factory=list)
    constraints_internal: list[ProfileFieldEntry] = field(default_factory=list)
    constraints_social: list[ProfileFieldEntry] = field(default_factory=list)
    constraints_hard_system: list[ProfileFieldEntry] = field(default_factory=list)
    allowed_methods: list[ProfileFieldEntry] = field(default_factory=list)
    maladaptive_methods: list[ProfileFieldEntry] = field(default_factory=list)
    defense_mechanisms: list[ProfileFieldEntry] = field(default_factory=list)
    communication_style: list[ProfileFieldEntry] = field(default_factory=list)
    attachment_patterns: list[ProfileFieldEntry] = field(default_factory=list)
    refusal_style: list[ProfileFieldEntry] = field(default_factory=list)
    triggers: list[ProfileFieldEntry] = field(default_factory=list)
    pressure_response: list[ProfileFieldEntry] = field(default_factory=list)
    growth_pattern: list[ProfileFieldEntry] = field(default_factory=list)

    def get_field(self, field_name: str) -> list[ProfileFieldEntry]:
        return list(getattr(self, field_name, []))

    def set_field(self, field_name: str, entries: list[ProfileFieldEntry]) -> None:
        if field_name in PROFILE_FIELD_NAMES:
            object.__setattr__(self, field_name, entries)

    def all_field_names(self) -> tuple[str, ...]:
        return PROFILE_FIELD_NAMES

    def stable_entries(self, field_name: str) -> list[ProfileFieldEntry]:
        """Return only non-temporary, non-uncertain entries for a field."""
        return [e for e in self.get_field(field_name) if not e.is_temporary and not e.is_uncertain]

    def to_dict(self) -> dict[str, Any]:
        return {
            name: [e.to_dict() for e in self.get_field(name)]
            for name in PROFILE_FIELD_NAMES
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PersonalityProfile':
        obj = cls()
        for name in PROFILE_FIELD_NAMES:
            raw = list(data.get(name) or [])
            entries = [ProfileFieldEntry.from_dict(item) for item in raw if isinstance(item, dict)]
            setattr(obj, name, entries)
        return obj


# ---------------------------------------------------------------------------
# Biography fact
# ---------------------------------------------------------------------------

@dataclass
class BiographyFact:
    """
    A single structured factual statement about the person's life.

    Must NOT contain psychological interpretations.
    Must NOT contain uncertain inferences unless confidence < 0.5.

    Examples:
        fact_type='education', value='studied at YSU'
        fact_type='occupation', value='software engineer at Acme Corp'
        fact_type='life_event', value='moved to Paris in 2018'
    """

    fact_id: str
    fact_type: str = 'factual_statement'    # BiographyFactType value
    value: str = ''
    confidence: float = 0.8
    source: str = 'chat_user'               # SourceChannel value
    source_channel: str = 'chat_user'
    timestamp: str = ''
    session_id: str = ''
    evidence_excerpt: str = ''
    conflicts: list[ConflictRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'fact_id': self.fact_id,
            'fact_type': self.fact_type,
            'value': self.value,
            'confidence': round(float(self.confidence or 0.8), 6),
            'source': self.source,
            'source_channel': self.source_channel,
            'timestamp': self.timestamp,
            'session_id': self.session_id,
            'evidence_excerpt': self.evidence_excerpt,
            'conflicts': [c.to_dict() for c in self.conflicts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'BiographyFact':
        return cls(
            fact_id=str(data.get('fact_id') or ''),
            fact_type=str(data.get('fact_type') or 'factual_statement'),
            value=str(data.get('value') or ''),
            confidence=float(data.get('confidence') or 0.8),
            source=str(data.get('source') or 'chat_user'),
            source_channel=str(data.get('source_channel') or 'chat_user'),
            timestamp=str(data.get('timestamp') or ''),
            session_id=str(data.get('session_id') or ''),
            evidence_excerpt=str(data.get('evidence_excerpt') or ''),
            conflicts=[ConflictRecord.from_dict(c) for c in list(data.get('conflicts') or [])],
        )


# ---------------------------------------------------------------------------
# Biography
# ---------------------------------------------------------------------------

@dataclass
class PersonalityBiography:
    """
    Factual life information store.

    Separated from the psychological profile by design.
    Documents and chat can add facts here.  Inferences do NOT belong here.

    Three sub-collections:
        facts         — general factual statements
        timeline      — time-ordered life events
        relationships — facts about known relationships
    """

    facts: list[BiographyFact] = field(default_factory=list)
    timeline: list[BiographyFact] = field(default_factory=list)
    relationships: list[BiographyFact] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)  # Optional human-readable narratives

    def to_dict(self) -> dict[str, Any]:
        return {
            'facts': [f.to_dict() for f in self.facts],
            'timeline': [f.to_dict() for f in self.timeline],
            'relationships': [f.to_dict() for f in self.relationships],
            'summaries': list(self.summaries),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PersonalityBiography':
        def _load_facts(raw: Any) -> list[BiographyFact]:
            return [BiographyFact.from_dict(item) for item in list(raw or []) if isinstance(item, dict)]
        return cls(
            facts=_load_facts(data.get('facts')),
            timeline=_load_facts(data.get('timeline')),
            relationships=_load_facts(data.get('relationships')),
            summaries=[str(s) for s in list(data.get('summaries') or [])],
        )


# ---------------------------------------------------------------------------
# Temporary state
# ---------------------------------------------------------------------------

@dataclass
class TemporaryStateEntry:
    """
    Session-scoped temporary state.

    Must NEVER be merged into the stable profile or biography.
    Examples: "today I am upset", "feeling anxious right now".

    Cleared at session end or when contradicted by more stable data.
    """

    entry_id: str
    value: str
    source_channel: str = 'chat_user'
    session_id: str = ''
    timestamp: str = ''
    reason: str = ''  # Why this is classified as temporary

    def to_dict(self) -> dict[str, Any]:
        return {
            'entry_id': self.entry_id,
            'value': self.value,
            'source_channel': self.source_channel,
            'session_id': self.session_id,
            'timestamp': self.timestamp,
            'reason': self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'TemporaryStateEntry':
        return cls(
            entry_id=str(data.get('entry_id') or ''),
            value=str(data.get('value') or ''),
            source_channel=str(data.get('source_channel') or 'chat_user'),
            session_id=str(data.get('session_id') or ''),
            timestamp=str(data.get('timestamp') or ''),
            reason=str(data.get('reason') or ''),
        )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

@dataclass
class PersonalityIdentity:
    """Identity block of the personality object."""

    label: str = ''
    aliases: list[str] = field(default_factory=list)
    persona_type: str = 'hybrid'
    stable_self_description: str = ''
    source_texts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'label': self.label,
            'aliases': list(self.aliases),
            'persona_type': self.persona_type,
            'stable_self_description': self.stable_self_description,
            'source_texts': list(self.source_texts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PersonalityIdentity':
        return cls(
            label=str(data.get('label') or ''),
            aliases=[str(a) for a in list(data.get('aliases') or [])],
            persona_type=str(data.get('persona_type') or 'hybrid'),
            stable_self_description=str(data.get('stable_self_description') or ''),
            source_texts=[str(t) for t in list(data.get('source_texts') or [])],
        )


# ---------------------------------------------------------------------------
# State meta
# ---------------------------------------------------------------------------

@dataclass
class PersonalityStateMeta:
    """System metadata — confidence, completeness, provenance summary."""

    confidence: float = 0.0
    completeness: float = 0.0          # 0.0–1.0 how many required fields are populated
    readiness: str = 'draft'           # 'seed' | 'draft' | 'full'
    validation_status: str = 'partial' # 'valid' | 'partial' | 'rejected'
    hover_summary: str = ''
    source_channels: list[str] = field(default_factory=list)
    updated_at: str = ''
    update_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            'confidence': round(float(self.confidence or 0.0), 6),
            'completeness': round(float(self.completeness or 0.0), 6),
            'readiness': self.readiness,
            'validation_status': self.validation_status,
            'hover_summary': self.hover_summary,
            'source_channels': list(self.source_channels),
            'updated_at': self.updated_at,
            'update_count': int(self.update_count or 0),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PersonalityStateMeta':
        return cls(
            confidence=float(data.get('confidence') or 0.0),
            completeness=float(data.get('completeness') or 0.0),
            readiness=str(data.get('readiness') or 'draft'),
            validation_status=str(data.get('validation_status') or 'partial'),
            hover_summary=str(data.get('hover_summary') or ''),
            source_channels=[str(s) for s in list(data.get('source_channels') or [])],
            updated_at=str(data.get('updated_at') or ''),
            update_count=int(data.get('update_count') or 0),
        )


# ---------------------------------------------------------------------------
# Top-level personality object
# ---------------------------------------------------------------------------

@dataclass
class PersonalityObject:
    """
    Top-level structured personality container.

    This is the single source of truth for a person's personality in the system.
    It is NOT a narrative essay.  It is a typed, versioned, provenance-tracked object.

    Groups:
        identity    — stable identity facts
        profile     — psychological profile fields (fears, goals, defenses, …)
        biography   — factual life information
        state_meta  — system metadata

    Link to persona:
        persona_name links back to the persona slug in the persona engine
        if this personality is attached to an active persona.
    """

    personality_id: str = ''
    persona_name: str = ''   # slug of the linked persona, if any
    identity: PersonalityIdentity = field(default_factory=PersonalityIdentity)
    profile: PersonalityProfile = field(default_factory=PersonalityProfile)
    biography: PersonalityBiography = field(default_factory=PersonalityBiography)
    state_meta: PersonalityStateMeta = field(default_factory=PersonalityStateMeta)

    def to_dict(self) -> dict[str, Any]:
        return {
            'personality_id': self.personality_id,
            'persona_name': self.persona_name,
            'identity': self.identity.to_dict(),
            'profile': self.profile.to_dict(),
            'biography': self.biography.to_dict(),
            'state_meta': self.state_meta.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PersonalityObject':
        return cls(
            personality_id=str(data.get('personality_id') or ''),
            persona_name=str(data.get('persona_name') or ''),
            identity=PersonalityIdentity.from_dict(dict(data.get('identity') or {})),
            profile=PersonalityProfile.from_dict(dict(data.get('profile') or {})),
            biography=PersonalityBiography.from_dict(dict(data.get('biography') or {})),
            state_meta=PersonalityStateMeta.from_dict(dict(data.get('state_meta') or {})),
        )


# ---------------------------------------------------------------------------
# Update candidate (output of the parser, input to the merge engine)
# ---------------------------------------------------------------------------

@dataclass
class UpdateCandidate:
    """
    A classified candidate update produced by the update parser.

    This is an intermediate object — it is not stored directly.
    The merge engine consumes these and decides what to do with each one.

    update_class values:
        profile_field      → goes to PersonalityProfile.<field_name>
        biography_fact     → goes to PersonalityBiography
        temporary_state    → goes to session-scoped temp store, NOT stable profile
        uncertain_inference → stored as low-confidence entry in profile (is_uncertain=True)
        rejected_noise     → discarded with a reason
    """

    update_class: str = 'rejected_noise'   # UpdateClass value
    field_name: str = ''                   # which profile field, for 'profile_field'
    value: str = ''
    confidence: float = 0.8
    source_channel: str = 'chat_user'      # SourceChannel value
    session_id: str = ''
    timestamp: str = ''
    evidence_excerpt: str = ''
    is_uncertain: bool = False
    fact_type: str = 'factual_statement'   # BiographyFactType, for 'biography_fact'
    reason: str = ''                       # Parser reasoning

    def to_dict(self) -> dict[str, Any]:
        return {
            'update_class': self.update_class,
            'field_name': self.field_name,
            'value': self.value,
            'confidence': round(float(self.confidence or 0.8), 6),
            'source_channel': self.source_channel,
            'session_id': self.session_id,
            'timestamp': self.timestamp,
            'evidence_excerpt': self.evidence_excerpt,
            'is_uncertain': bool(self.is_uncertain),
            'fact_type': self.fact_type,
            'reason': self.reason,
        }


# ---------------------------------------------------------------------------
# Update result
# ---------------------------------------------------------------------------

@dataclass
class PersonalityUpdateResult:
    """Summary of what the merge engine did with a set of update candidates."""

    personality_id: str = ''
    applied: list[UpdateCandidate] = field(default_factory=list)
    rejected: list[UpdateCandidate] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)            # entry_ids that were merged
    conflicts_created: list[str] = field(default_factory=list) # conflict_ids created
    temp_state: list[TemporaryStateEntry] = field(default_factory=list)
    summary: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'personality_id': self.personality_id,
            'applied': [c.to_dict() for c in self.applied],
            'rejected': [c.to_dict() for c in self.rejected],
            'merged': list(self.merged),
            'conflicts_created': list(self.conflicts_created),
            'temp_state': [e.to_dict() for e in self.temp_state],
            'summary': self.summary,
        }
