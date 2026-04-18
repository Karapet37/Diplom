"""
Personality summarizer.

Generates human-readable summaries from a PersonalityObject:
    - hover_summary       : one short line for UI tooltips
    - compact_summary     : operator-readable 5-7 bullet points
    - label               : short identity label

These summaries are derived purely from the structured data —
no LLM call is needed.  They are stored in state_meta.hover_summary
after generation.
"""

from __future__ import annotations

from typing import Any

from .personality_schema import PersonalityObject


# ---------------------------------------------------------------------------
# Core summary logic
# ---------------------------------------------------------------------------

def _top_values(profile_field: list[Any], n: int = 3) -> list[str]:
    """Return up to n highest-confidence stable entry values."""
    stable = [e for e in profile_field if not e.is_temporary]
    sorted_entries = sorted(stable, key=lambda e: e.confidence, reverse=True)
    return [e.value for e in sorted_entries[:n]]


def generate_hover_summary(personality: PersonalityObject) -> str:
    """
    Generate a single-line hover summary.

    Format: "<label> — fears: X; wants: Y; defends by: Z"
    """
    label = personality.identity.label or personality.persona_name or 'Unknown'
    profile = personality.profile

    parts: list[str] = []

    top_fears = _top_values(profile.fears, 1)
    if top_fears:
        fear_text = top_fears[0][:60]
        parts.append(f'fears: {fear_text}')

    top_goals = _top_values(profile.core_goals, 1)
    if top_goals:
        goal_text = top_goals[0][:60]
        parts.append(f'wants: {goal_text}')

    top_defense = _top_values(profile.defense_mechanisms, 1)
    if top_defense:
        defense_text = top_defense[0][:60]
        parts.append(f'defends by: {defense_text}')

    # Try needs as fallback if no goal
    if not top_goals:
        top_needs = _top_values(profile.needs, 1)
        if top_needs:
            parts.append(f'needs: {top_needs[0][:60]}')

    body = '; '.join(parts)
    if body:
        return f'{label} — {body}'
    return label


def generate_compact_summary(personality: PersonalityObject) -> dict[str, Any]:
    """
    Generate an operator-readable compact summary dict.

    Contains:
        label, readiness, completeness, top fears, top goals, top defenses,
        top biography facts, source channels, conflicts count.
    """
    profile = personality.profile
    bio = personality.biography
    meta = personality.state_meta

    def top_stable(field_entries: list[Any], n: int = 3) -> list[str]:
        stable = [e for e in field_entries if not e.is_temporary and not e.is_uncertain]
        sorted_entries = sorted(stable, key=lambda e: e.confidence, reverse=True)
        return [e.value for e in sorted_entries[:n]]

    all_bio = list(bio.facts) + list(bio.timeline) + list(bio.relationships)
    top_bio = sorted(all_bio, key=lambda f: f.confidence, reverse=True)[:4]

    # Count unresolved conflicts
    conflict_count = sum(
        1
        for field_name in profile.all_field_names()
        for entry in profile.get_field(field_name)
        for c in entry.conflicts
        if not c.resolved
    ) + sum(
        1
        for fact in all_bio
        for c in fact.conflicts
        if not c.resolved
    )

    return {
        'label': personality.identity.label or personality.persona_name,
        'persona_name': personality.persona_name,
        'personality_id': personality.personality_id,
        'readiness': meta.readiness,
        'completeness': round(meta.completeness, 2),
        'confidence': round(meta.confidence, 2),
        'validation_status': meta.validation_status,
        'top_fears': top_stable(profile.fears),
        'top_goals': top_stable(profile.core_goals) or top_stable(profile.secondary_goals),
        'top_needs': top_stable(profile.needs),
        'top_defenses': top_stable(profile.defense_mechanisms),
        'top_triggers': top_stable(profile.triggers),
        'top_biography_facts': [f.value for f in top_bio],
        'source_channels': list(meta.source_channels),
        'unresolved_conflicts': conflict_count,
        'updated_at': meta.updated_at,
    }


def generate_operator_label(personality: PersonalityObject) -> str:
    """
    Generate a short operator-readable label for display in lists.

    Examples:
        "Guarded Pride (draft, 40% complete)"
        "Anya Yusupova (full, 85% complete)"
    """
    label = personality.identity.label or personality.persona_name or 'Unnamed'
    meta = personality.state_meta
    pct = int(round(meta.completeness * 100))
    return f'{label} ({meta.readiness}, {pct}% complete)'


def update_hover_summary(personality: PersonalityObject) -> None:
    """Regenerate and store hover_summary in state_meta (in-place)."""
    personality.state_meta.hover_summary = generate_hover_summary(personality)
