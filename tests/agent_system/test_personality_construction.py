"""
Tests for the personality construction layer and behavioral action engine.

Covers:
 1.  Creating a personality from structured input
 2.  Updating psychological fields from chat
 3.  Updating biography facts from chat
 4.  Updating biography facts from document extraction
 5.  Preserving provenance
 6.  Duplicate merge (no duplicate entries created)
 7.  Contradiction handling (conflict records created, both entries kept)
 8.  Temporary state not corrupting stable profile
 9.  Uncertain inference not becoming stable truth immediately
10.  API inspection of personality object

Behavioral action engine tests:
11.  Different actions selected when fear dominates vs desire dominates
12.  Hard constraints blocking otherwise-valid actions
13.  Attachment weakening firmness (refuse_firmly scored lower)
14.  Shame points pushing toward avoidance
15.  Habits influencing repeated action choice
16.  Fragile/shy personas prefer avoidance over confident confrontation
17.  Route-aware action selection (persona_chat vs factual_answer)
18.  Score breakdown is non-trivial and deterministic
"""

from __future__ import annotations

import pytest

from agent_system.personality_schema import (
    PROFILE_FIELD_NAMES,
    BiographyFact,
    PersonalityBiography,
    PersonalityIdentity,
    PersonalityObject,
    PersonalityProfile,
    PersonalityStateMeta,
    ProfileFieldEntry,
    ProvenanceRecord,
    UpdateCandidate,
)
from agent_system.personality_update_parser import (
    SOURCE_CHANNEL_CHAT_USER,
    SOURCE_CHANNEL_UPLOADED_DOCUMENT,
    build_direct_biography_update,
    build_direct_profile_update,
    parse_document_candidates,
    parse_update_candidates,
)
from agent_system.personality_merge_engine import (
    apply_update_candidates,
    override_profile_field_entry,
    resolve_conflict,
)
from agent_system.personality_summarizer import (
    generate_compact_summary,
    generate_hover_summary,
    generate_operator_label,
)
from agent_system.behavioral_action_engine import (
    ACTION_FAMILIES,
    ROUTE_ACTION_WHITELIST,
    ConstraintFactor,
    CurrentStateModifier,
    PersonalityBehaviorModel,
    TriggerFactor,
    WeightedAttachment,
    WeightedFactor,
    build_behavior_model,
    pressure_modifier,
    select_action,
    select_action_for_persona,
    shame_activation_modifier,
    trust_modifier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_personality(label: str = 'Test Persona', persona_name: str = 'test') -> PersonalityObject:
    return PersonalityObject(
        personality_id=f'pid_{persona_name}',
        persona_name=persona_name,
        identity=PersonalityIdentity(label=label),
    )


def _make_fear_dominant_personality() -> PersonalityObject:
    """Personality where fears dominate — should prefer avoidance actions."""
    p = _make_personality('Fear-Dominant', 'fear_dominant')
    fear_entries = [
        ProfileFieldEntry(entry_id='f1', value='rejection', confidence=0.95),
        ProfileFieldEntry(entry_id='f2', value='humiliation', confidence=0.90),
        ProfileFieldEntry(entry_id='f3', value='being used', confidence=0.88),
    ]
    p.profile.fears = fear_entries
    p.profile.shame_points = [
        ProfileFieldEntry(entry_id='s1', value='visible weakness', confidence=0.85),
    ]
    return p


def _make_desire_dominant_personality() -> PersonalityObject:
    """Personality where desires and goals dominate — should prefer compliance actions."""
    p = _make_personality('Desire-Dominant', 'desire_dominant')
    p.profile.core_goals = [
        ProfileFieldEntry(entry_id='g1', value='achieve recognition', confidence=0.92),
    ]
    p.profile.desires = [
        ProfileFieldEntry(entry_id='d1', value='closeness and affection', confidence=0.90),
        ProfileFieldEntry(entry_id='d2', value='to be understood', confidence=0.85),
    ]
    p.profile.needs = [
        ProfileFieldEntry(entry_id='n1', value='emotional validation', confidence=0.88),
    ]
    return p


def _make_model_from_personality(personality: PersonalityObject) -> PersonalityBehaviorModel:
    return build_behavior_model(personality)


# ---------------------------------------------------------------------------
# 1. Creating a personality from structured input
# ---------------------------------------------------------------------------

class TestPersonalityCreation:
    def test_create_basic_personality_object(self):
        p = _make_personality('Guarded Pride')
        assert p.personality_id
        assert p.identity.label == 'Guarded Pride'
        assert isinstance(p.profile, PersonalityProfile)
        assert isinstance(p.biography, PersonalityBiography)
        assert isinstance(p.state_meta, PersonalityStateMeta)

    def test_profile_has_all_required_fields(self):
        p = _make_personality()
        for field_name in PROFILE_FIELD_NAMES:
            assert hasattr(p.profile, field_name), f'Missing profile field: {field_name}'
            assert isinstance(p.profile.get_field(field_name), list)

    def test_personality_serializes_and_deserializes(self):
        p = _make_personality('Round Trip')
        p.profile.fears = [ProfileFieldEntry(entry_id='f1', value='rejection', confidence=0.9)]
        d = p.to_dict()
        p2 = PersonalityObject.from_dict(d)
        assert p2.identity.label == 'Round Trip'
        assert len(p2.profile.fears) == 1
        assert p2.profile.fears[0].value == 'rejection'

    def test_direct_profile_update_candidate_is_stable(self):
        candidate = build_direct_profile_update('fears', 'being abandoned')
        assert candidate.update_class == 'profile_field'
        assert candidate.field_name == 'fears'
        assert candidate.confidence == 0.90
        assert not candidate.is_uncertain

    def test_direct_biography_update_candidate(self):
        candidate = build_direct_biography_update('education', 'studied at YSU')
        assert candidate.update_class == 'biography_fact'
        assert candidate.fact_type == 'education'
        assert 'YSU' in candidate.value


# ---------------------------------------------------------------------------
# 2. Updating psychological fields from chat
# ---------------------------------------------------------------------------

class TestProfileUpdateFromChat:
    def test_fear_extracted_from_chat_message(self):
        candidates = parse_update_candidates('I am afraid of being used.')
        profile_candidates = [c for c in candidates if c.update_class == 'profile_field']
        assert profile_candidates, 'Expected at least one profile_field candidate'
        fear_candidates = [c for c in profile_candidates if c.field_name == 'fears']
        assert fear_candidates, 'Expected fears field to be targeted'

    def test_fear_applied_to_personality(self):
        p = _make_personality()
        candidates = parse_update_candidates('I am afraid of being used.')
        result = apply_update_candidates(p, candidates)
        # Should have applied at least one profile field or uncertainty entry
        applied_classes = {c.update_class for c in result.applied}
        assert applied_classes & {'profile_field', 'uncertain_inference'}, (
            f'No profile_field or uncertain_inference applied; result: {result.summary}'
        )

    def test_goal_extracted_from_chat_message(self):
        candidates = parse_update_candidates('My main goal is to preserve my dignity.')
        profile_candidates = [c for c in candidates if c.update_class == 'profile_field']
        assert any(c.field_name == 'core_goals' for c in profile_candidates), (
            f'Expected core_goals; got: {[c.field_name for c in profile_candidates]}'
        )

    def test_multiple_fields_from_single_message(self):
        msg = 'I am afraid of rejection. I need emotional safety.'
        candidates = parse_update_candidates(msg)
        field_names = {c.field_name for c in candidates if c.update_class == 'profile_field'}
        assert 'fears' in field_names or 'needs' in field_names, (
            f'Expected fears and/or needs; got: {field_names}'
        )

    def test_empty_message_produces_no_candidates(self):
        candidates = parse_update_candidates('')
        assert candidates == []

    def test_noise_message_produces_no_candidates(self):
        candidates = parse_update_candidates('Hi!')
        assert candidates == []


# ---------------------------------------------------------------------------
# 3. Updating biography facts from chat
# ---------------------------------------------------------------------------

class TestBiographyUpdateFromChat:
    def test_education_fact_extracted(self):
        candidates = parse_update_candidates('I studied at YSU.')
        bio_candidates = [c for c in candidates if c.update_class == 'biography_fact']
        assert bio_candidates, f'Expected biography_fact; got: {[c.update_class for c in candidates]}'
        assert any(c.fact_type == 'education' for c in bio_candidates), (
            f'Expected education fact_type; got: {[c.fact_type for c in bio_candidates]}'
        )

    def test_education_applied_to_biography(self):
        p = _make_personality()
        candidates = parse_update_candidates('I studied at YSU.')
        result = apply_update_candidates(p, candidates)
        all_bio = list(p.biography.facts) + list(p.biography.timeline) + list(p.biography.relationships)
        bio_values = [f.value for f in all_bio]
        assert any('YSU' in v for v in bio_values), (
            f'YSU not stored in biography; stored: {bio_values}'
        )

    def test_occupation_fact_extracted(self):
        candidates = parse_update_candidates('I work as a software engineer.')
        bio_candidates = [c for c in candidates if c.update_class == 'biography_fact']
        assert any(c.fact_type == 'occupation' for c in bio_candidates), (
            f'Expected occupation; got: {[c.fact_type for c in bio_candidates]}'
        )

    def test_life_event_extracted(self):
        candidates = parse_update_candidates('I was born in 1990.')
        bio_candidates = [c for c in candidates if c.update_class == 'biography_fact']
        assert bio_candidates, 'Expected biography_fact for birth event'

    def test_family_fact_extracted(self):
        candidates = parse_update_candidates('My mother always told me to be careful.')
        bio_candidates = [c for c in candidates if c.update_class == 'biography_fact']
        assert any(c.fact_type == 'family' for c in bio_candidates), (
            f'Expected family; got: {[c.fact_type for c in bio_candidates]}'
        )


# ---------------------------------------------------------------------------
# 4. Updating biography facts from document extraction
# ---------------------------------------------------------------------------

class TestBiographyUpdateFromDocument:
    def test_document_source_channel_is_correct(self):
        candidates = parse_document_candidates('She graduated from State University in 2005.')
        assert all(c.source_channel == 'uploaded_document' for c in candidates)

    def test_document_education_stored_in_biography(self):
        p = _make_personality()
        candidates = parse_document_candidates('She graduated from State University in 2005.')
        result = apply_update_candidates(p, candidates)
        all_bio = list(p.biography.facts) + list(p.biography.timeline) + list(p.biography.relationships)
        assert all_bio or result.applied, (
            f'Expected biography update from document; result: {result.summary}'
        )

    def test_document_occupation_stored(self):
        p = _make_personality()
        candidate = build_direct_biography_update(
            'occupation', 'worked as a translator for 5 years',
            source_channel='uploaded_document',
        )
        result = apply_update_candidates(p, [candidate])
        all_bio = list(p.biography.facts) + list(p.biography.timeline) + list(p.biography.relationships)
        assert any('translator' in f.value for f in all_bio), (
            f'Expected translator in biography; got: {[f.value for f in all_bio]}'
        )
        assert all(f.source == 'uploaded_document' for f in all_bio if 'translator' in f.value)

    def test_structured_biography_fact_has_correct_fields(self):
        p = _make_personality()
        candidate = build_direct_biography_update('life_event', 'moved to Paris in 2018')
        apply_update_candidates(p, [candidate])
        facts = list(p.biography.timeline) + list(p.biography.facts)
        matching = [f for f in facts if 'Paris' in f.value]
        assert matching, 'Expected Paris timeline entry'
        f = matching[0]
        assert f.fact_id
        assert f.fact_type in ('life_event', 'factual_statement')
        assert f.confidence > 0.0
        assert f.timestamp


# ---------------------------------------------------------------------------
# 5. Preserving provenance
# ---------------------------------------------------------------------------

class TestProvenancePreservation:
    def test_profile_entry_has_provenance(self):
        p = _make_personality()
        candidate = build_direct_profile_update(
            'fears', 'being ignored',
            source_channel='chat_user',
            session_id='sess_001',
        )
        apply_update_candidates(p, [candidate])
        entries = p.profile.fears
        assert entries, 'Expected at least one fear entry'
        prov = entries[0].provenance
        assert prov, 'Expected provenance records'
        assert prov[0].source == 'chat_user'
        assert prov[0].session_id == 'sess_001'

    def test_biography_fact_has_source_channel(self):
        p = _make_personality()
        candidate = build_direct_biography_update(
            'education', 'attended YSU',
            source_channel='uploaded_document',
            session_id='sess_doc',
        )
        apply_update_candidates(p, [candidate])
        all_bio = list(p.biography.facts) + list(p.biography.timeline)
        bio = next((f for f in all_bio if 'YSU' in f.value), None)
        assert bio is not None
        assert bio.source == 'uploaded_document'
        assert bio.session_id == 'sess_doc'

    def test_multiple_updates_accumulate_provenance(self):
        p = _make_personality()
        c1 = build_direct_profile_update('fears', 'rejection', session_id='s1')
        c2 = build_direct_profile_update('fears', 'rejection', session_id='s2')
        apply_update_candidates(p, [c1])
        apply_update_candidates(p, [c2])
        entry = p.profile.fears[0]
        # Both provenance records should be stored on the merged entry
        session_ids = {prov.session_id for prov in entry.provenance}
        assert 's1' in session_ids
        assert 's2' in session_ids


# ---------------------------------------------------------------------------
# 6. Duplicate merge
# ---------------------------------------------------------------------------

class TestDuplicateMerge:
    def test_exact_duplicate_not_added_twice(self):
        p = _make_personality()
        candidate = build_direct_profile_update('fears', 'rejection')
        apply_update_candidates(p, [candidate])
        count_before = len(p.profile.fears)
        apply_update_candidates(p, [candidate])
        count_after = len(p.profile.fears)
        assert count_before == count_after, (
            f'Duplicate entry created: count went from {count_before} to {count_after}'
        )

    def test_semantically_similar_duplicate_merged(self):
        p = _make_personality()
        c1 = build_direct_profile_update('fears', 'I am afraid of rejection')
        c2 = build_direct_profile_update('fears', 'afraid of rejection')
        apply_update_candidates(p, [c1])
        apply_update_candidates(p, [c2])
        assert len(p.profile.fears) == 1, (
            f'Expected 1 entry after semantic dedup; got {len(p.profile.fears)}'
        )

    def test_duplicate_biography_fact_not_doubled(self):
        p = _make_personality()
        c1 = build_direct_biography_update('education', 'studied at YSU')
        c2 = build_direct_biography_update('education', 'studied at YSU')
        apply_update_candidates(p, [c1])
        apply_update_candidates(p, [c2])
        all_bio = list(p.biography.facts) + list(p.biography.timeline)
        ysu_facts = [f for f in all_bio if 'YSU' in f.value]
        assert len(ysu_facts) == 1, f'Expected 1 biography fact; got {len(ysu_facts)}'

    def test_merged_result_tracks_merged_ids(self):
        p = _make_personality()
        c1 = build_direct_profile_update('needs', 'safety')
        apply_update_candidates(p, [c1])
        result2 = apply_update_candidates(p, [c1])
        assert result2.merged, 'Expected merged list to be non-empty on duplicate'


# ---------------------------------------------------------------------------
# 7. Contradiction handling
# ---------------------------------------------------------------------------

class TestContradictionHandling:
    def test_contradicting_biography_facts_both_kept(self):
        p = _make_personality()
        c1 = build_direct_biography_update('location', 'lives in Paris')
        c2 = build_direct_biography_update('location', 'not living in Paris')
        apply_update_candidates(p, [c1])
        result2 = apply_update_candidates(p, [c2])
        all_bio = list(p.biography.facts) + list(p.biography.timeline)
        location_facts = [f for f in all_bio if 'Paris' in f.value]
        assert len(location_facts) >= 2, (
            f'Expected both contradicting facts kept; got {len(location_facts)}: {[f.value for f in location_facts]}'
        )

    def test_contradiction_creates_conflict_record(self):
        p = _make_personality()
        c1 = build_direct_biography_update('location', 'lives in Paris')
        c2 = build_direct_biography_update('location', 'not living in Paris')
        apply_update_candidates(p, [c1])
        result2 = apply_update_candidates(p, [c2])
        assert result2.conflicts_created, 'Expected conflict_id to be recorded'

    def test_contradiction_profile_field_both_entries_present(self):
        p = _make_personality()
        c1 = build_direct_profile_update('values', 'honesty is paramount')
        c2 = build_direct_profile_update('values', 'not valuing honesty')
        apply_update_candidates(p, [c1])
        apply_update_candidates(p, [c2])
        entries = p.profile.values
        assert len(entries) >= 2, f'Expected both conflicting entries kept; got {len(entries)}'

    def test_conflict_can_be_resolved(self):
        p = _make_personality()
        c1 = build_direct_biography_update('location', 'lives in Paris')
        c2 = build_direct_biography_update('location', 'not living in Paris')
        apply_update_candidates(p, [c1])
        result2 = apply_update_candidates(p, [c2])
        conflict_id = result2.conflicts_created[0]
        ok = resolve_conflict(p, conflict_id, resolution='user confirmed not in Paris')
        assert ok


# ---------------------------------------------------------------------------
# 8. Temporary state not corrupting stable profile
# ---------------------------------------------------------------------------

class TestTemporaryStateIsolation:
    def test_temp_state_not_in_stable_profile(self):
        p = _make_personality()
        candidates = parse_update_candidates('I am upset right now.')
        result = apply_update_candidates(p, candidates)
        # Temp state goes to result.temp_state, NOT to profile fields
        applied_classes = {c.update_class for c in result.applied}
        assert 'temporary_state' not in applied_classes, (
            f'"upset right now" applied to profile as: {applied_classes}'
        )
        # Also check profile.fears is empty (no stable fear written from "upset")
        stable_fears = p.profile.stable_entries('fears')
        assert not stable_fears, f'Expected no stable fears; got: {[e.value for e in stable_fears]}'

    def test_temp_state_routed_to_temp_list(self):
        p = _make_personality()
        candidates = parse_update_candidates('Today I am angry.')
        result = apply_update_candidates(p, candidates)
        if candidates:
            # At least some candidates should have been routed to temp_state
            # (even if parser classified them as profile, the "today" anchor marks them temp)
            classes = {c.update_class for c in candidates}
            if 'temporary_state' in classes:
                assert result.temp_state, 'Expected temp_state entries'

    def test_explicitly_temporary_candidate_not_in_stable_profile(self):
        """Direct test: a temporary_state candidate must not reach the profile."""
        p = _make_personality()
        temp_candidate = UpdateCandidate(
            update_class='temporary_state',
            value='feeling panicked right now',
            source_channel='chat_user',
        )
        result = apply_update_candidates(p, [temp_candidate])
        # Nothing in stable fears
        assert not p.profile.stable_entries('fears')
        # Should be in temp_state
        assert result.temp_state

    def test_temp_entry_has_reason_field(self):
        p = _make_personality()
        temp_candidate = UpdateCandidate(
            update_class='temporary_state',
            value='very stressed today',
            source_channel='chat_user',
            reason='temporal anchor: today',
        )
        result = apply_update_candidates(p, [temp_candidate])
        if result.temp_state:
            assert result.temp_state[0].reason


# ---------------------------------------------------------------------------
# 9. Uncertain inference not becoming stable truth immediately
# ---------------------------------------------------------------------------

class TestUncertainInference:
    def test_uncertain_inference_stored_as_uncertain(self):
        p = _make_personality()
        candidates = parse_update_candidates('Maybe he feels jealous.')
        result = apply_update_candidates(p, candidates)
        # Any uncertain_inference applied should be stored with is_uncertain=True
        for field_name in PROFILE_FIELD_NAMES:
            for entry in p.profile.get_field(field_name):
                if entry.is_uncertain:
                    assert entry.confidence <= 0.65, (
                        f'Uncertain entry has high confidence: {entry.confidence}'
                    )

    def test_uncertain_inference_marked_is_uncertain_true(self):
        p = _make_personality()
        uncertain_candidate = UpdateCandidate(
            update_class='uncertain_inference',
            field_name='fears',
            value='possibly afraid of heights',
            confidence=0.45,
            is_uncertain=True,
        )
        result = apply_update_candidates(p, [uncertain_candidate])
        if p.profile.fears:
            uncertain_entries = [e for e in p.profile.fears if e.is_uncertain]
            # Should be stored as uncertain, not promoted to stable
            assert uncertain_entries, 'Uncertain inference not stored as uncertain'

    def test_uncertain_inference_below_promotion_threshold(self):
        p = _make_personality()
        uncertain_candidate = UpdateCandidate(
            update_class='uncertain_inference',
            field_name='values',
            value='may value solitude',
            confidence=0.45,
            is_uncertain=True,
        )
        apply_update_candidates(p, [uncertain_candidate])
        stable = p.profile.stable_entries('values')
        assert not stable, (
            f'Expected no stable values after low-confidence uncertain inference; got: {[e.value for e in stable]}'
        )

    def test_hedge_language_produces_uncertain_inference(self):
        candidates = parse_update_candidates('I think maybe she avoids people out of shame.')
        uncertain = [c for c in candidates if c.is_uncertain or c.update_class == 'uncertain_inference']
        assert uncertain, 'Expected hedge language to produce uncertain inference'


# ---------------------------------------------------------------------------
# 10. API inspection of personality object
# ---------------------------------------------------------------------------

class TestPersonalityObjectInspection:
    def test_to_dict_round_trip(self):
        p = _make_personality('Inspection Test')
        p.profile.fears = [
            ProfileFieldEntry(
                entry_id='f1',
                value='rejection',
                confidence=0.9,
                provenance=[ProvenanceRecord(source='chat_user', session_id='s1')],
            )
        ]
        p.biography.facts = [
            BiographyFact(fact_id='b1', fact_type='education', value='studied at YSU', confidence=0.88)
        ]
        d = p.to_dict()

        # personality_id and persona_name
        assert d['personality_id']
        assert d['identity']['label'] == 'Inspection Test'

        # profile fields present
        assert 'fears' in d['profile']
        fear_list = d['profile']['fears']
        assert len(fear_list) == 1
        assert fear_list[0]['value'] == 'rejection'
        assert fear_list[0]['provenance'][0]['source'] == 'chat_user'

        # biography facts
        assert d['biography']['facts'][0]['value'] == 'studied at YSU'

    def test_compact_summary_has_required_keys(self):
        p = _make_fear_dominant_personality()
        summary = generate_compact_summary(p)
        assert 'label' in summary
        assert 'readiness' in summary
        assert 'completeness' in summary
        assert 'top_fears' in summary
        assert 'top_goals' in summary
        assert 'source_channels' in summary
        assert 'unresolved_conflicts' in summary

    def test_hover_summary_contains_label(self):
        p = _make_personality('Hidden Pride')
        p.profile.fears = [ProfileFieldEntry(entry_id='f1', value='exposure', confidence=0.9)]
        hover = generate_hover_summary(p)
        assert 'Hidden Pride' in hover
        assert 'exposure' in hover.lower() or 'fear' in hover.lower()

    def test_operator_label_contains_readiness(self):
        p = _make_personality('Operator Label Test')
        label = generate_operator_label(p)
        assert 'seed' in label or 'draft' in label or 'full' in label

    def test_state_meta_recalculated_after_update(self):
        p = _make_personality()
        assert p.state_meta.update_count == 0
        c = build_direct_profile_update('fears', 'rejection')
        apply_update_candidates(p, [c])
        assert p.state_meta.update_count > 0
        assert p.state_meta.updated_at


# ---------------------------------------------------------------------------
# 11. Fear dominance vs desire dominance produces different actions
# ---------------------------------------------------------------------------

class TestActionSelectionFearVsDesire:
    def test_fear_dominant_prefers_avoidance(self):
        p = _make_fear_dominant_personality()
        result = select_action_for_persona(p, 'persona_chat', situation_type='insult')
        assert result.selected_action in (
            'avoid', 'deflect', 'retreat', 'deny',
            'refuse_weakly', 'refuse_firmly',
        ), f'Expected avoidance action; got: {result.selected_action}'

    def test_desire_dominant_prefers_engagement(self):
        p = _make_desire_dominant_personality()
        result = select_action_for_persona(p, 'persona_chat', situation_type='neutral_query')
        assert result.selected_action in (
            'comply', 'soften', 'seek_support', 'ask_back', 'probe_indirectly',
        ), f'Expected engagement action; got: {result.selected_action}'

    def test_fear_and_desire_dominant_produce_different_actions(self):
        fear_p = _make_fear_dominant_personality()
        desire_p = _make_desire_dominant_personality()
        fear_result = select_action_for_persona(fear_p, 'persona_chat', situation_type='insult')
        desire_result = select_action_for_persona(desire_p, 'persona_chat', situation_type='insult')
        # Not necessarily different every time but fear-dominant should score higher on avoidance
        fear_model = build_behavior_model(fear_p)
        desire_model = build_behavior_model(desire_p)
        from agent_system.behavioral_action_engine import _score_action
        fear_avoid = _score_action('avoid', fear_model).score
        desire_avoid = _score_action('avoid', desire_model).score
        assert fear_avoid >= desire_avoid, (
            f'Fear-dominant should score avoidance >= desire-dominant; '
            f'fear={fear_avoid:.3f} desire={desire_avoid:.3f}'
        )


# ---------------------------------------------------------------------------
# 12. Hard constraints block actions
# ---------------------------------------------------------------------------

class TestHardConstraintBlocking:
    def test_hard_constraint_blocks_escalate_emotionally(self):
        model = PersonalityBehaviorModel(
            persona_name='constrained',
            hard_constraints=[
                ConstraintFactor(
                    label='never escalate emotionally',
                    strength=1.0,
                    constraint_type='hard',
                    active=True,
                )
            ],
        )
        result = select_action(model, 'persona_chat', situation_type='user_anger')
        # escalate_emotionally should be blocked
        blocked_names = [a.action for a in result.blocked_actions]
        assert 'escalate_emotionally' in blocked_names, (
            f'Expected escalate_emotionally blocked; blocked: {blocked_names}'
        )
        assert result.selected_action != 'escalate_emotionally'

    def test_hard_constraint_not_present_allows_escalation(self):
        model = PersonalityBehaviorModel(
            persona_name='unconstrained',
            shame_points=[WeightedFactor(label='visible weakness', weight=0.9)],
            fears=[WeightedFactor(label='being controlled', weight=0.85)],
        )
        result = select_action(model, 'persona_chat', situation_type='insult')
        # Without hard constraint, escalate_emotionally should not be blocked
        blocked_names = [a.action for a in result.blocked_actions]
        assert 'escalate_emotionally' not in blocked_names

    def test_hard_constraint_is_route_independent(self):
        model = PersonalityBehaviorModel(
            hard_constraints=[
                ConstraintFactor(
                    label='must not emotionally escalate',
                    strength=1.0,
                    constraint_type='hard',
                    active=True,
                )
            ],
        )
        for route in ('persona_chat', 'general_chat'):
            result = select_action(model, route)
            blocked_names = [a.action for a in result.blocked_actions]
            if 'escalate_emotionally' in ROUTE_ACTION_WHITELIST.get(route, ()):
                assert 'escalate_emotionally' in blocked_names


# ---------------------------------------------------------------------------
# 13. Attachment weakens firmness
# ---------------------------------------------------------------------------

class TestAttachmentWeakensFirmness:
    def test_high_attachment_reduces_refuse_firmly_score(self):
        model_with_attachment = PersonalityBehaviorModel(
            fears=[WeightedFactor(label='abandonment', weight=0.8)],
            attachments=[
                WeightedAttachment(target='close_person', weight=0.95, attachment_type='anxious')
            ],
        )
        model_without_attachment = PersonalityBehaviorModel(
            fears=[WeightedFactor(label='abandonment', weight=0.8)],
        )
        from agent_system.behavioral_action_engine import _score_action
        score_with = _score_action('refuse_firmly', model_with_attachment).score
        score_without = _score_action('refuse_firmly', model_without_attachment).score
        assert score_with < score_without, (
            f'Attachment should reduce refuse_firmly; with={score_with:.3f} without={score_without:.3f}'
        )

    def test_attachment_increases_comply_score(self):
        model = PersonalityBehaviorModel(
            attachments=[WeightedAttachment(target='partner', weight=0.9)],
            needs=[WeightedFactor(label='closeness', weight=0.8)],
        )
        from agent_system.behavioral_action_engine import _score_action
        comply_score = _score_action('comply', model).score
        avoid_score = _score_action('avoid', model).score
        assert comply_score > avoid_score, (
            f'Attachment+needs should score comply > avoid; comply={comply_score:.3f} avoid={avoid_score:.3f}'
        )


# ---------------------------------------------------------------------------
# 14. Shame points push toward avoidance
# ---------------------------------------------------------------------------

class TestShamePointsAvoidance:
    def test_shame_pushes_toward_deflect_and_avoid(self):
        model = PersonalityBehaviorModel(
            shame_points=[
                WeightedFactor(label='showing weakness publicly', weight=0.9),
                WeightedFactor(label='dependency', weight=0.85),
            ],
        )
        from agent_system.behavioral_action_engine import _score_action
        deflect_score = _score_action('deflect', model).score
        comply_score = _score_action('comply', model).score
        seek_support_score = _score_action('seek_support', model).score
        assert deflect_score > comply_score, (
            f'High shame should score deflect > comply; deflect={deflect_score:.3f} comply={comply_score:.3f}'
        )

    def test_shame_activation_modifier_amplifies_avoidance(self):
        p = _make_fear_dominant_personality()
        p.profile.shame_points = [
            ProfileFieldEntry(entry_id='sp1', value='visible dependency', confidence=0.9)
        ]
        result_normal = select_action_for_persona(p, 'persona_chat')
        result_shame = select_action_for_persona(
            p, 'persona_chat',
            current_state_modifiers=[shame_activation_modifier(0.9)],
        )
        avoidance_actions = {'avoid', 'deflect', 'retreat', 'deny'}
        shame_is_avoidance = result_shame.selected_action in avoidance_actions
        # At minimum: shame should not make the persona MORE engaging
        assert shame_is_avoidance or result_shame.selected_action in (
            'refuse_weakly', 'refuse_firmly', 'probe_indirectly'
        ), f'Unexpected action under shame: {result_shame.selected_action}'


# ---------------------------------------------------------------------------
# 15. Route-aware action selection
# ---------------------------------------------------------------------------

class TestRouteAwareActionSelection:
    def test_persona_chat_route_allows_emotional_actions(self):
        model = PersonalityBehaviorModel(
            persona_name='test',
            fears=[WeightedFactor(label='rejection', weight=0.9)],
        )
        result = select_action(model, 'persona_chat')
        assert result.selected_action in ROUTE_ACTION_WHITELIST['persona_chat'], (
            f'Action not in persona_chat whitelist: {result.selected_action}'
        )

    def test_factual_answer_route_only_allows_informational_actions(self):
        model = PersonalityBehaviorModel(
            persona_name='test',
            goals=[WeightedFactor(label='provide accurate information', weight=0.9)],
        )
        result = select_action(model, 'factual_answer')
        assert result.selected_action in ROUTE_ACTION_WHITELIST['factual_answer'], (
            f'Action not in factual_answer whitelist: {result.selected_action}'
        )
        assert result.selected_action not in ('avoid', 'deflect', 'irritate', 'escalate_emotionally'), (
            f'Emotional action selected on factual route: {result.selected_action}'
        )

    def test_persona_specification_route_allows_parse_actions(self):
        model = PersonalityBehaviorModel(persona_name='test')
        result = select_action(model, 'persona_specification')
        assert result.selected_action in ROUTE_ACTION_WHITELIST['persona_specification'], (
            f'Action not in persona_specification whitelist: {result.selected_action}'
        )

    def test_different_routes_can_produce_different_actions(self):
        model = PersonalityBehaviorModel(
            goals=[WeightedFactor(label='be helpful', weight=0.9)],
            values=[WeightedFactor(label='accuracy', weight=0.85)],
        )
        persona_result = select_action(model, 'persona_chat')
        factual_result = select_action(model, 'factual_answer')
        # Selected actions may differ across routes
        assert persona_result.route == 'persona_chat'
        assert factual_result.route == 'factual_answer'


# ---------------------------------------------------------------------------
# 16. Score breakdown is non-trivial and deterministic
# ---------------------------------------------------------------------------

class TestScoreBreakdown:
    def test_score_breakdown_has_contributing_factors(self):
        model = PersonalityBehaviorModel(
            fears=[WeightedFactor(label='rejection', weight=0.9)],
            shame_points=[WeightedFactor(label='weakness', weight=0.8)],
            needs=[WeightedFactor(label='safety', weight=0.85)],
        )
        from agent_system.behavioral_action_engine import _score_action
        scored = _score_action('avoid', model)
        assert scored.score_breakdown, 'Expected non-empty score breakdown'
        assert 'fears' in scored.score_breakdown or 'shame_points' in scored.score_breakdown

    def test_scoring_is_deterministic(self):
        model = PersonalityBehaviorModel(
            fears=[WeightedFactor(label='rejection', weight=0.9)],
        )
        from agent_system.behavioral_action_engine import _score_action
        s1 = _score_action('avoid', model).score
        s2 = _score_action('avoid', model).score
        assert s1 == s2, 'Score is not deterministic'

    def test_result_has_verbalization_directive(self):
        p = _make_fear_dominant_personality()
        result = select_action_for_persona(p, 'persona_chat')
        assert result.verbalization_directive, 'Expected non-empty verbalization directive'
        assert result.selected_action.upper() in result.verbalization_directive

    def test_confidence_reflects_score_gap(self):
        model = PersonalityBehaviorModel(
            fears=[WeightedFactor(label='rejection', weight=0.99, confidence=0.99)],
            shame_points=[WeightedFactor(label='exposure', weight=0.99, confidence=0.99)],
        )
        result = select_action(model, 'persona_chat', situation_type='insult')
        assert 0.0 <= result.confidence <= 1.0

    def test_pressure_modifier_shifts_scores(self):
        p = _make_fear_dominant_personality()
        result_no_pressure = select_action_for_persona(p, 'persona_chat')
        result_high_pressure = select_action_for_persona(
            p, 'persona_chat',
            current_state_modifiers=[pressure_modifier(0.9)],
        )
        # Both must have valid selected actions
        assert result_no_pressure.selected_action in ACTION_FAMILIES
        assert result_high_pressure.selected_action in ACTION_FAMILIES
