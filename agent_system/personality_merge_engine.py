"""
Personality merge engine.

Takes UpdateCandidate objects (from the update parser) and applies them to a
PersonalityObject according to explicit merge/update rules.

Merge rules:
    1. Semantic duplicate check   — if a new entry says roughly the same thing
                                    as an existing one, merge (don't duplicate).
    2. Contradiction check        — if a new entry contradicts an existing one,
                                    keep BOTH with a ConflictRecord; do not silently drop either.
    3. Confidence upgrade         — if a new entry provides better/more-structured
                                    version of a vague existing entry, upgrade confidence.
    4. Temporary routing          — temporary_state candidates go to a separate
                                    temp list, never into the stable profile.
    5. Uncertain inference        — stored as is_uncertain=True; not elevated to
                                    stable truth unless confidence > PROMOTE_THRESHOLD
                                    and repeated evidence has accumulated.
    6. Provenance preservation    — every write records where it came from.
    7. No silent overwrites       — the entire profile is never replaced on a single update.

STABLE_PROFILE_UPDATE_CONFIDENCE_MINIMUM: below this threshold, entries are
    stored as uncertain, regardless of classification.
PROMOTE_THRESHOLD: the minimum confidence at which an uncertain inference can
    be promoted to a stable entry during a merge.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Any

from .personality_schema import (
    BiographyFact,
    ConflictRecord,
    PersonalityBiography,
    PersonalityObject,
    PersonalityProfile,
    PersonalityStateMeta,
    PersonalityUpdateResult,
    ProfileFieldEntry,
    ProvenanceRecord,
    TemporaryStateEntry,
    UpdateCandidate,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STABLE_PROFILE_UPDATE_CONFIDENCE_MINIMUM = 0.60
PROMOTE_THRESHOLD = 0.75       # uncertain entry confidence floor to promote to stable
SIMILARITY_THRESHOLD = 0.55    # token-overlap similarity to call entries "duplicate"
MAX_ENTRIES_PER_FIELD = 24     # guard rail against unbounded growth
MAX_BIOGRAPHY_FACTS = 128      # guard rail for biography


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _short_id(text: str, prefix: str = '') -> str:
    """Deterministic short id from content hash."""
    h = hashlib.sha1(str(text or '').encode('utf-8', errors='replace')).hexdigest()[:10]
    return f'{prefix}{h}' if prefix else h


def _conflict_id(a_id: str, b_id: str) -> str:
    pair = ':'.join(sorted([str(a_id), str(b_id)]))
    return _short_id(pair, 'conflict_')


def _normalize_for_compare(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    t = str(text or '').lower()
    t = re.sub(r'[^\w\s]', ' ', t)
    return ' '.join(t.split())


def _token_overlap(a: str, b: str) -> float:
    """Jaccard-style token overlap between two strings."""
    a_tokens = set(_normalize_for_compare(a).split())
    b_tokens = set(_normalize_for_compare(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / union if union else 0.0


def _is_semantic_duplicate(new_value: str, existing_value: str) -> bool:
    """True if two field entry values are semantically equivalent."""
    norm_new = _normalize_for_compare(new_value)
    norm_existing = _normalize_for_compare(existing_value)
    if norm_new == norm_existing:
        return True
    return _token_overlap(norm_new, norm_existing) >= SIMILARITY_THRESHOLD


def _is_contradiction(new_value: str, existing_value: str) -> bool:
    """
    Detect likely contradictions using negation patterns.

    Heuristic: the new value contains an explicit negation of a key phrase
    in the existing value, or vice versa.
    """
    _CONTRADICTION_OVERLAP_THRESHOLD = 0.40  # lower than dedup threshold

    _NEGATION_PATTERNS = (
        (r'\bnot\s+', ''),
        (r'\bnever\s+', ''),
        (r'\bno\s+longer\s+', ''),
        (r'\bno\s+', ''),
        (r'\bdon\'?t\s+', ''),
        (r'\bdoesn\'?t\s+', ''),
        (r'\bwon\'?t\s+', ''),
        (r'\bнет\s+', ''),
        (r'\bне\s+', ''),
        (r'\bникогда\s+', ''),
    )
    norm_new = _normalize_for_compare(new_value)
    norm_existing = _normalize_for_compare(existing_value)
    # Quick check: if neither contains negation words, no contradiction
    has_negation = any(
        re.search(pat, norm_new) or re.search(pat, norm_existing)
        for pat, _ in _NEGATION_PATTERNS
    )
    if not has_negation:
        return False
    # Check if one is the negation of the other
    for neg_pat, replacement in _NEGATION_PATTERNS:
        stripped_new = re.sub(neg_pat, replacement, norm_new)
        stripped_existing = re.sub(neg_pat, replacement, norm_existing)
        if stripped_new and stripped_existing:
            if _token_overlap(stripped_new, norm_existing) > _CONTRADICTION_OVERLAP_THRESHOLD:
                return True
            if _token_overlap(norm_new, stripped_existing) > _CONTRADICTION_OVERLAP_THRESHOLD:
                return True
    return False


def _make_provenance(candidate: UpdateCandidate) -> ProvenanceRecord:
    return ProvenanceRecord(
        source=str(candidate.source_channel or 'chat_user'),
        source_channel=str(candidate.source_channel or 'chat_user'),
        session_id=str(candidate.session_id or ''),
        timestamp=str(candidate.timestamp or _now_iso()),
        evidence_excerpt=str(candidate.evidence_excerpt or candidate.value or ''),
    )


# ---------------------------------------------------------------------------
# Profile field merge
# ---------------------------------------------------------------------------

def _merge_into_profile_field(
    entries: list[ProfileFieldEntry],
    candidate: UpdateCandidate,
    *,
    result: PersonalityUpdateResult,
) -> list[ProfileFieldEntry]:
    """
    Merge a single profile-field candidate into an existing entries list.

    Returns the updated list.
    """
    new_value = str(candidate.value or '').strip()
    if not new_value:
        result.rejected.append(candidate)
        return entries

    is_uncertain = bool(candidate.is_uncertain) or candidate.confidence < STABLE_PROFILE_UPDATE_CONFIDENCE_MINIMUM
    new_conf = float(candidate.confidence or 0.8)
    prov = _make_provenance(candidate)
    new_id = _short_id(new_value + candidate.field_name, 'pfe_')

    # --- Check for duplicates ---
    for existing in entries:
        if _is_semantic_duplicate(new_value, existing.value):
            # Merge: add provenance to existing entry, upgrade confidence if better
            if prov not in existing.provenance:
                existing.provenance.append(prov)
            if new_conf > existing.confidence:
                existing.confidence = new_conf
            # If new data shows this entry is now stable, upgrade
            if not candidate.is_uncertain and existing.is_uncertain and new_conf >= PROMOTE_THRESHOLD:
                existing.is_uncertain = False
            result.merged.append(existing.entry_id)
            return entries

    # --- Check for contradictions ---
    for existing in entries:
        if _is_contradiction(new_value, existing.value):
            cid = _conflict_id(existing.entry_id, new_id)
            conflict = ConflictRecord(
                conflict_id=cid,
                conflicting_fact_id=existing.entry_id,
                description=f'Contradiction: {new_value!r} vs {existing.value!r}',
            )
            # Mark both entries
            existing.conflicts.append(conflict)
            new_entry = ProfileFieldEntry(
                entry_id=new_id,
                value=new_value,
                confidence=new_conf,
                provenance=[prov],
                is_uncertain=is_uncertain,
                conflicts=[ConflictRecord(
                    conflict_id=cid,
                    conflicting_fact_id=existing.entry_id,
                    description=conflict.description,
                )],
            )
            entries.append(new_entry)
            result.conflicts_created.append(cid)
            result.applied.append(candidate)
            return entries

    # --- No duplicate or conflict: append ---
    # Guard rail
    stable_count = sum(1 for e in entries if not e.is_temporary)
    if stable_count >= MAX_ENTRIES_PER_FIELD:
        # Drop lowest confidence uncertain entry to make room
        uncertain_entries = [e for e in entries if e.is_uncertain]
        if uncertain_entries:
            lowest = min(uncertain_entries, key=lambda e: e.confidence)
            entries = [e for e in entries if e.entry_id != lowest.entry_id]

    entries.append(ProfileFieldEntry(
        entry_id=new_id,
        value=new_value,
        confidence=new_conf,
        provenance=[prov],
        is_uncertain=is_uncertain,
    ))
    result.applied.append(candidate)
    return entries


# ---------------------------------------------------------------------------
# Biography merge
# ---------------------------------------------------------------------------

def _merge_into_biography(
    biography: PersonalityBiography,
    candidate: UpdateCandidate,
    *,
    result: PersonalityUpdateResult,
) -> PersonalityBiography:
    """
    Merge a biography_fact candidate into the biography object.

    Routes to .timeline, .relationships, or .facts based on fact_type.
    """
    new_value = str(candidate.value or '').strip()
    if not new_value:
        result.rejected.append(candidate)
        return biography

    fact_type = str(candidate.fact_type or 'factual_statement')
    new_id = _short_id(new_value + fact_type, 'bfact_')
    conf = float(candidate.confidence or 0.8)

    # Choose which sub-collection to use
    if fact_type in ('life_event', 'timeline_entry', 'achievement', 'loss'):
        target_list = biography.timeline
    elif fact_type == 'relationship':
        target_list = biography.relationships
    else:
        target_list = biography.facts

    # Enforce guard
    if len(target_list) >= MAX_BIOGRAPHY_FACTS:
        result.rejected.append(candidate)
        return biography

    # Duplicate check across all sub-collections
    for fact_list in (biography.facts, biography.timeline, biography.relationships):
        for existing_fact in fact_list:
            if _is_semantic_duplicate(new_value, existing_fact.value):
                # Merge: just add provenance and possibly update confidence
                if existing_fact.confidence < conf:
                    existing_fact.confidence = conf
                result.merged.append(existing_fact.fact_id)
                return biography

    # Contradiction check
    for fact_list in (biography.facts, biography.timeline, biography.relationships):
        for existing_fact in fact_list:
            if existing_fact.fact_type == fact_type and _is_contradiction(new_value, existing_fact.value):
                cid = _conflict_id(existing_fact.fact_id, new_id)
                conflict = ConflictRecord(
                    conflict_id=cid,
                    conflicting_fact_id=existing_fact.fact_id,
                    description=f'Contradiction: {new_value!r} vs {existing_fact.value!r}',
                )
                existing_fact.conflicts.append(conflict)
                new_fact = BiographyFact(
                    fact_id=new_id,
                    fact_type=fact_type,
                    value=new_value,
                    confidence=conf,
                    source=str(candidate.source_channel or 'chat_user'),
                    source_channel=str(candidate.source_channel or 'chat_user'),
                    timestamp=str(candidate.timestamp or _now_iso()),
                    session_id=str(candidate.session_id or ''),
                    evidence_excerpt=str(candidate.evidence_excerpt or new_value),
                    conflicts=[ConflictRecord(
                        conflict_id=cid,
                        conflicting_fact_id=existing_fact.fact_id,
                        description=conflict.description,
                    )],
                )
                target_list.append(new_fact)
                result.conflicts_created.append(cid)
                result.applied.append(candidate)
                return biography

    # Fresh fact
    new_fact = BiographyFact(
        fact_id=new_id,
        fact_type=fact_type,
        value=new_value,
        confidence=conf,
        source=str(candidate.source_channel or 'chat_user'),
        source_channel=str(candidate.source_channel or 'chat_user'),
        timestamp=str(candidate.timestamp or _now_iso()),
        session_id=str(candidate.session_id or ''),
        evidence_excerpt=str(candidate.evidence_excerpt or new_value),
    )
    target_list.append(new_fact)
    result.applied.append(candidate)
    return biography


# ---------------------------------------------------------------------------
# Temporary state routing
# ---------------------------------------------------------------------------

def _route_temporary_state(
    candidate: UpdateCandidate,
    *,
    result: PersonalityUpdateResult,
) -> None:
    """Route a temporary_state candidate to the result's temp_state list."""
    entry = TemporaryStateEntry(
        entry_id=_short_id(candidate.value, 'tmp_'),
        value=str(candidate.value or ''),
        source_channel=str(candidate.source_channel or 'chat_user'),
        session_id=str(candidate.session_id or ''),
        timestamp=str(candidate.timestamp or _now_iso()),
        reason=str(candidate.reason or 'temporary_state'),
    )
    result.temp_state.append(entry)
    # NOT added to result.applied — temp state is not a stable personality update


# ---------------------------------------------------------------------------
# State meta recalculation
# ---------------------------------------------------------------------------

def _recalculate_state_meta(personality: PersonalityObject) -> None:
    """Recalculate completeness, confidence, and readiness from current data."""
    profile = personality.profile
    total_fields = len(profile.all_field_names())
    populated_fields = sum(
        1 for name in profile.all_field_names()
        if any(not e.is_temporary for e in profile.get_field(name))
    )
    completeness = populated_fields / total_fields if total_fields else 0.0

    bio = personality.biography
    bio_count = len(bio.facts) + len(bio.timeline) + len(bio.relationships)

    # Average confidence across non-temporary profile entries
    conf_values = [
        e.confidence
        for name in profile.all_field_names()
        for e in profile.get_field(name)
        if not e.is_temporary and not e.is_uncertain
    ]
    avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0.0

    meta = personality.state_meta
    meta.completeness = round(completeness, 4)
    meta.confidence = round(avg_conf, 4)
    meta.updated_at = _now_iso()
    meta.update_count = (meta.update_count or 0) + 1

    # Readiness
    if completeness >= 0.6 and avg_conf >= 0.7 and bio_count >= 1:
        meta.readiness = 'full'
    elif completeness >= 0.3 or bio_count >= 1:
        meta.readiness = 'draft'
    else:
        meta.readiness = 'seed'

    # Validation status
    conflicts_present = any(
        e.conflicts
        for name in profile.all_field_names()
        for e in profile.get_field(name)
    )
    if not conflicts_present and completeness >= 0.5 and avg_conf >= 0.65:
        meta.validation_status = 'valid'
    elif completeness < 0.1:
        meta.validation_status = 'partial'
    else:
        meta.validation_status = 'partial'


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_update_candidates(
    personality: PersonalityObject,
    candidates: list[UpdateCandidate],
) -> PersonalityUpdateResult:
    """
    Apply a list of UpdateCandidate objects to a PersonalityObject in-place.

    Returns a PersonalityUpdateResult describing what was applied, rejected,
    merged, or flagged as temporary state.

    Rules:
        - temporary_state    → routed to result.temp_state only, NOT stored in profile
        - uncertain_inference → stored with is_uncertain=True, not promoted unless
                                confidence >= PROMOTE_THRESHOLD
        - profile_field      → merged into profile.<field_name> with deduplication
        - biography_fact     → merged into biography.facts/timeline/relationships
        - rejected_noise     → added to result.rejected

    The personality object is mutated in-place.  Callers should persist it
    after this function returns.
    """
    result = PersonalityUpdateResult(personality_id=str(personality.personality_id or ''))

    for candidate in candidates:
        update_class = str(candidate.update_class or 'rejected_noise')

        if update_class == 'rejected_noise':
            result.rejected.append(candidate)
            continue

        if update_class == 'temporary_state':
            _route_temporary_state(candidate, result=result)
            continue

        if update_class == 'biography_fact':
            personality.biography = _merge_into_biography(
                personality.biography, candidate, result=result,
            )
            # Track source channel in meta
            ch = str(candidate.source_channel or 'chat_user')
            if ch not in personality.state_meta.source_channels:
                personality.state_meta.source_channels.append(ch)
            continue

        if update_class in ('profile_field', 'uncertain_inference'):
            field_name = str(candidate.field_name or '').strip()
            if not field_name or field_name not in personality.profile.all_field_names():
                # Field name unknown: store as uncertain in 'vulnerabilities' as a catch-all
                # if it's an uncertain inference, otherwise reject
                if update_class == 'uncertain_inference':
                    candidate = UpdateCandidate(
                        update_class='uncertain_inference',
                        field_name='vulnerabilities',
                        value=candidate.value,
                        confidence=min(candidate.confidence, 0.45),
                        source_channel=candidate.source_channel,
                        session_id=candidate.session_id,
                        timestamp=candidate.timestamp,
                        evidence_excerpt=candidate.evidence_excerpt,
                        is_uncertain=True,
                        reason=f'unknown_field_fallback: {candidate.reason}',
                    )
                    field_name = 'vulnerabilities'
                else:
                    result.rejected.append(candidate)
                    continue

            current_entries = list(personality.profile.get_field(field_name))
            updated_entries = _merge_into_profile_field(current_entries, candidate, result=result)
            personality.profile.set_field(field_name, updated_entries)

            ch = str(candidate.source_channel or 'chat_user')
            if ch not in personality.state_meta.source_channels:
                personality.state_meta.source_channels.append(ch)
            continue

        # Unknown class: reject
        result.rejected.append(candidate)

    # Recalculate state meta after batch
    _recalculate_state_meta(personality)

    # Build summary
    n_applied = len(result.applied)
    n_rejected = len(result.rejected)
    n_merged = len(result.merged)
    n_conflicts = len(result.conflicts_created)
    n_temp = len(result.temp_state)
    result.summary = (
        f'applied={n_applied} merged={n_merged} temp={n_temp} '
        f'conflicts={n_conflicts} rejected={n_rejected}'
    )
    return result


# ---------------------------------------------------------------------------
# Targeted field override (for structured corrections)
# ---------------------------------------------------------------------------

def override_profile_field_entry(
    personality: PersonalityObject,
    field_name: str,
    entry_id: str,
    new_value: str,
    *,
    source_channel: str = 'manual_edit',
    session_id: str = '',
    confidence: float = 0.90,
) -> bool:
    """
    Override a specific entry by entry_id within a profile field.

    Used for structured follow-up corrections where the user explicitly
    refines a specific field entry.  Only the targeted entry is changed;
    other entries are preserved.

    Returns True if the entry was found and updated, False otherwise.
    """
    entries = list(personality.profile.get_field(field_name))
    for entry in entries:
        if entry.entry_id == entry_id:
            prov = ProvenanceRecord(
                source=source_channel,
                source_channel=source_channel,
                session_id=session_id,
                timestamp=_now_iso(),
                evidence_excerpt=new_value,
            )
            entry.value = new_value
            entry.confidence = confidence
            entry.is_uncertain = False
            entry.provenance.append(prov)
            personality.profile.set_field(field_name, entries)
            _recalculate_state_meta(personality)
            return True
    return False


def resolve_conflict(
    personality: PersonalityObject,
    conflict_id: str,
    resolution: str,
    *,
    keep_entry_id: str = '',
) -> bool:
    """
    Mark a conflict as resolved.

    If keep_entry_id is provided, the opposing conflicting entry is removed
    (the winner survives).  Otherwise, both entries are kept and the conflict
    is only marked resolved.
    """
    resolved_any = False

    # Search in profile fields
    for field_name in personality.profile.all_field_names():
        entries = list(personality.profile.get_field(field_name))
        entries_to_remove = []
        for entry in entries:
            for conflict in entry.conflicts:
                if conflict.conflict_id == conflict_id:
                    conflict.resolved = True
                    conflict.resolution = resolution
                    resolved_any = True
            # If loser: remove
            if (
                keep_entry_id
                and entry.entry_id != keep_entry_id
                and any(c.conflict_id == conflict_id for c in entry.conflicts)
            ):
                entries_to_remove.append(entry.entry_id)
        if entries_to_remove:
            entries = [e for e in entries if e.entry_id not in entries_to_remove]
            personality.profile.set_field(field_name, entries)

    # Search in biography
    for fact_list in (personality.biography.facts, personality.biography.timeline, personality.biography.relationships):
        facts_to_remove = []
        for fact in fact_list:
            for conflict in fact.conflicts:
                if conflict.conflict_id == conflict_id:
                    conflict.resolved = True
                    conflict.resolution = resolution
                    resolved_any = True
            if (
                keep_entry_id
                and fact.fact_id != keep_entry_id
                and any(c.conflict_id == conflict_id for c in fact.conflicts)
            ):
                facts_to_remove.append(fact.fact_id)
        if facts_to_remove:
            surviving = [f for f in fact_list if f.fact_id not in facts_to_remove]
            fact_list.clear()
            fact_list.extend(surviving)

    if resolved_any:
        _recalculate_state_meta(personality)
    return resolved_any
