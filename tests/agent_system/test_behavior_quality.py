"""
Behavior quality tests.

Tests cover spec categories:
- personality updates stored correctly, biography vs profile separation
- temporary state does not overwrite stable profile
- delete personality works safely (removes index, files, no stale pointers)
- action selector changes behavior when forces change
- user feedback changes next plan (feedback → step intensity)
"""

from __future__ import annotations

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Personality model integrity tests
# ---------------------------------------------------------------------------

class TestPersonalityModelIntegrity:
    def test_profile_and_biography_stay_separate(self, tmp_path, monkeypatch):
        """
        Profile fields (fears, goals) and biography facts (education, occupation)
        must never be in the same collection.
        """
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.personality_schema import (
            PersonalityObject, ProfileFieldEntry, BiographyFact,
        )
        from agent_system.personality_store import save_personality, load_personality

        personality = PersonalityObject(persona_name='test_person')
        personality.profile.fears.append(ProfileFieldEntry(
            entry_id='f1',
            value='fear of failure',
            confidence=0.9,
        ))
        personality.biography.facts.append(BiographyFact(
            fact_id='b1',
            fact_type='occupation',
            value='software engineer',
            confidence=0.95,
        ))

        pid = save_personality(personality)
        loaded = load_personality(pid)
        assert loaded is not None

        fears = loaded.profile.get_field('fears')
        assert any(e.value == 'fear of failure' for e in fears)
        bio_values = [f.value for f in loaded.biography.facts]
        assert 'software engineer' in bio_values
        # Biography must NOT leak into profile
        for field_name in loaded.profile.all_field_names():
            for entry in loaded.profile.get_field(field_name):
                assert 'software engineer' not in entry.value, \
                    f'Biography fact leaked into profile field {field_name}'

    def test_temporary_state_does_not_overwrite_stable_profile(self, tmp_path, monkeypatch):
        """
        Temporary state entries (is_temporary=True) must not replace stable profile entries.
        """
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.personality_schema import PersonalityObject, ProfileFieldEntry
        from agent_system.personality_store import save_personality, load_personality

        personality = PersonalityObject(persona_name='temp_test')
        personality.profile.fears.append(ProfileFieldEntry(
            entry_id='stable_fear',
            value='stable fear of rejection',
            confidence=0.9,
            is_temporary=False,
        ))
        personality.profile.fears.append(ProfileFieldEntry(
            entry_id='temp_fear',
            value='temporary fear after bad day',
            confidence=0.7,
            is_temporary=True,
        ))
        pid = save_personality(personality)
        loaded = load_personality(pid)
        assert loaded is not None

        all_fears = loaded.profile.get_field('fears')
        stable_fears = loaded.profile.stable_entries('fears')

        assert len(all_fears) == 2
        assert len(stable_fears) == 1
        assert stable_fears[0].value == 'stable fear of rejection'
        assert any(e.value == 'stable fear of rejection' for e in all_fears)

    def test_delete_personality_cleans_index(self, tmp_path, monkeypatch):
        """
        delete_personality removes the file AND cleans the index.
        No stale pointers remain.
        """
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.personality_schema import PersonalityObject
        from agent_system.personality_store import (
            save_personality, load_personality, delete_personality,
            load_personality_for_persona, list_personalities,
        )

        personality = PersonalityObject(persona_name='doomed_persona')
        pid = save_personality(personality)

        assert load_personality(pid) is not None
        assert load_personality_for_persona('doomed_persona') is not None
        listed = list_personalities()
        assert any(p['personality_id'] == pid for p in listed)

        result = delete_personality(pid)
        assert result is True

        assert load_personality(pid) is None
        assert load_personality_for_persona('doomed_persona') is None
        listed_after = list_personalities()
        assert not any(p['personality_id'] == pid for p in listed_after)

    def test_delete_nonexistent_personality_returns_false(self, tmp_path, monkeypatch):
        """Deleting a non-existent personality returns False."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.personality_store import delete_personality
        result = delete_personality('personality_does_not_exist_xyz')
        assert result is False

    def test_provenance_is_preserved(self, tmp_path, monkeypatch):
        """Each profile entry retains its provenance record."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.personality_schema import (
            PersonalityObject, ProfileFieldEntry, ProvenanceRecord,
        )
        from agent_system.personality_store import save_personality, load_personality

        personality = PersonalityObject(persona_name='provenance_test')
        entry = ProfileFieldEntry(
            entry_id='prov_entry',
            value='wants recognition',
            confidence=0.85,
            provenance=[
                ProvenanceRecord(
                    source='chat_user',
                    session_id='session_xyz',
                    evidence_excerpt='I really want to be recognized',
                )
            ],
        )
        personality.profile.core_goals.append(entry)
        pid = save_personality(personality)
        loaded = load_personality(pid)
        assert loaded is not None

        goals = loaded.profile.get_field('core_goals')
        assert len(goals) == 1
        assert len(goals[0].provenance) == 1
        assert goals[0].provenance[0].evidence_excerpt == 'I really want to be recognized'
        assert goals[0].provenance[0].session_id == 'session_xyz'


# ---------------------------------------------------------------------------
# Action selector behavior change tests
# ---------------------------------------------------------------------------

class TestActionSelectorChanges:
    def test_action_changes_when_forces_change(self):
        """
        When personality forces change (dopamine high vs low),
        the selected action must differ.
        """
        from agent_system.situation_regulator import select_action, RegulatorState

        energized = RegulatorState(
            dopamine=0.85, cortisol=0.2, serotonin=0.7,
            fatigue=0.1, adrenaline=0.5, testosterone=0.6, oxytocin=0.6,
        )
        best_energized, _ = select_action(
            regulator_state=energized,
            event_type='planning_request',
            overload_level=0.05,
            feedback_class='success',
        )

        depleted = RegulatorState(
            dopamine=0.15, cortisol=0.85, serotonin=0.2,
            fatigue=0.9, adrenaline=0.2, testosterone=0.3, oxytocin=0.3,
        )
        best_depleted, _ = select_action(
            regulator_state=depleted,
            event_type='overload',
            overload_level=0.85,
            feedback_class='overload',
        )

        assert best_energized != best_depleted, (
            f'Action should differ between energized ({best_energized}) '
            f'and depleted ({best_depleted}) states'
        )

    def test_planning_blocked_during_crisis(self):
        """plan_small_step action must be blocked during crisis/overload."""
        from agent_system.situation_regulator import score_action, RegulatorState

        overloaded = RegulatorState(cortisol=0.95, fatigue=0.9, dopamine=0.1,
                                     serotonin=0.1, adrenaline=0.1, testosterone=0.3, oxytocin=0.2)
        plan_score = score_action(
            'plan_small_step',
            regulator_state=overloaded,
            event_type='crisis',
            overload_level=0.9,
            feedback_class='crisis',
        )
        assert plan_score.blocked is True, 'plan_small_step must be blocked during crisis'

    def test_resistance_triggers_explore_action(self):
        """When feedback is resistance, explore_resistance scores in top 3."""
        from agent_system.situation_regulator import select_action, RegulatorState

        moderate_state = RegulatorState.baseline()
        best, scores = select_action(
            regulator_state=moderate_state,
            event_type='neutral',
            overload_level=0.2,
            feedback_class='resistance',
        )
        top3 = [s.action for s in scores[:3]]
        assert 'explore_resistance' in top3, f'explore_resistance should be in top 3. Got: {top3}'

    def test_overload_lowers_step_intensity(self):
        """
        Overload level from repeated failures results in more cautious intensity.
        """
        from agent_system.planning_engine import estimate_overload_level, select_step_intensity

        no_overload = estimate_overload_level([], [])
        high_overload = estimate_overload_level(
            ['failure', 'failure', 'overload', 'failure'], ['very stressed today']
        )
        no_overload_intensity = select_step_intensity(no_overload)
        high_overload_intensity = select_step_intensity(high_overload)

        intensity_order = ['normal', 'gentle', 'minimal', 'micro']
        no_idx = intensity_order.index(no_overload_intensity)
        high_idx = intensity_order.index(high_overload_intensity)
        assert high_idx > no_idx, (
            f'High overload should produce lower intensity than no overload. '
            f'Got no_overload={no_overload_intensity}, high_overload={high_overload_intensity}'
        )


# ---------------------------------------------------------------------------
# Feedback loop: feedback changes next plan
# ---------------------------------------------------------------------------

class TestFeedbackLoop:
    def test_success_feedback_maintains_step_intensity(self):
        """After success feedback, step intensity is not MORE conservative than baseline."""
        from agent_system.planning_engine import estimate_overload_level, select_step_intensity
        baseline = select_step_intensity(estimate_overload_level([], []))
        after_success = select_step_intensity(estimate_overload_level(['success', 'success'], []))
        intensity_order = ['normal', 'gentle', 'minimal', 'micro']
        baseline_idx = intensity_order.index(baseline)
        success_idx = intensity_order.index(after_success)
        assert success_idx <= baseline_idx, \
            f'Success should not make steps more conservative: {baseline} → {after_success}'

    def test_failure_feedback_increases_caution(self):
        """After failure feedback, step intensity is more conservative than after success."""
        from agent_system.planning_engine import estimate_overload_level, select_step_intensity
        after_success = select_step_intensity(estimate_overload_level(['success', 'success'], []))
        after_failure = select_step_intensity(estimate_overload_level(['failure', 'failure', 'failure'], []))
        intensity_order = ['normal', 'gentle', 'minimal', 'micro']
        success_idx = intensity_order.index(after_success)
        failure_idx = intensity_order.index(after_failure)
        assert failure_idx >= success_idx, \
            f'Failure should produce more cautious steps than success: {after_failure} vs {after_success}'

    def test_planning_output_changes_with_feedback_history(self, tmp_path, monkeypatch):
        """Planning output step_intensity differs after success vs failure history."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.planning_engine import build_planning_output

        output_success = build_planning_output(
            message='what next?',
            session_id='feedback_test_sess',
            recent_feedback=['success', 'success'],
        )
        output_failure = build_planning_output(
            message='what next?',
            session_id='feedback_test_sess',
            recent_feedback=['failure', 'failure', 'failure'],
        )
        intensity_order = ['normal', 'gentle', 'minimal', 'micro']
        success_idx = intensity_order.index(output_success.step_intensity)
        failure_idx = intensity_order.index(output_failure.step_intensity)
        assert failure_idx >= success_idx, (
            f'Failure history should produce more cautious intensity than success. '
            f'Got: success={output_success.step_intensity}, failure={output_failure.step_intensity}'
        )


# ---------------------------------------------------------------------------
# Regulator state serialization
# ---------------------------------------------------------------------------

class TestRegulatorStateSerialization:
    def test_regulator_state_roundtrips(self):
        """RegulatorState serializes and deserializes correctly."""
        from agent_system.situation_regulator import RegulatorState
        original = RegulatorState(
            dopamine=0.7, cortisol=0.4, serotonin=0.6,
            adrenaline=0.5, oxytocin=0.8, testosterone=0.3, fatigue=0.2,
        )
        d = original.to_dict()
        restored = RegulatorState.from_dict(d)
        assert abs(restored.dopamine - 0.7) < 0.001
        assert abs(restored.cortisol - 0.4) < 0.001
        assert abs(restored.oxytocin - 0.8) < 0.001

    def test_regulator_output_to_dict(self):
        """RegulatorOutput.to_dict() produces the expected structure."""
        from agent_system.situation_regulator import run_regulator_pipeline
        output = run_regulator_pipeline(message='I need a plan')
        d = output.to_dict()
        assert 'event_type' in d
        assert 'regulator_state' in d
        assert 'selected_action' in d
        assert 'action_scores' in d
        assert isinstance(d['regulator_state']['dopamine'], float)


# ---------------------------------------------------------------------------
# Notes + planning context integration
# ---------------------------------------------------------------------------

class TestNotesInPlanningContext:
    def test_notes_appear_in_planning_prompt_section(self, tmp_path, monkeypatch):
        """When notes are saved, they appear in the planning prompt section."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.notes_store import save_note
        save_note('integration_sess', 'I am afraid of being judged by my manager')
        save_note('integration_sess', 'My goal is to get promoted next quarter')

        from agent_system.planning_engine import build_planning_output
        output = build_planning_output(
            message='what should I do this week?',
            session_id='integration_sess',
        )
        section = output.render_planning_prompt_section()
        assert 'afraid of being judged' in section
        assert 'promoted' in section

    def test_planning_prompt_contains_5_section_structure(self, tmp_path, monkeypatch):
        """Planning prompt section contains the 5-section output structure."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.planning_engine import build_planning_output
        output = build_planning_output(
            message='help me figure out next steps',
            session_id='instructions_sess',
        )
        section = output.render_planning_prompt_section()
        assert 'What matters now' in section
        assert 'Forces in conflict' in section
        assert 'Next micro-step' in section
        assert 'Fallback smaller step' in section
        assert 'Follow-up question' in section
