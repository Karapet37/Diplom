"""
Real behavior tests for planning mode and notes system.

Tests cover spec categories:
1. planning mode returns bounded small-step output
2. note saving/loading/deleting
3. note injection into planning context
4. crisis signals switch to support/de-escalation behavior
5. user feedback changes next plan (feedback classification)
6. overload lowers step intensity
"""

from __future__ import annotations

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Notes store tests
# ---------------------------------------------------------------------------

class TestNotesStore:
    def test_save_and_load_note(self, tmp_path, monkeypatch):
        """Saving a note persists it and it can be loaded back."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.notes_store import save_note, load_notes
        note = save_note('session1', 'I am trying to exercise more this week')
        assert note.get('note_id')
        assert note.get('text') == 'I am trying to exercise more this week'

        notes = load_notes('session1')
        assert len(notes) == 1
        assert notes[0]['text'] == 'I am trying to exercise more this week'

    def test_multiple_notes_ordered(self, tmp_path, monkeypatch):
        """Multiple notes are stored in insertion order."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.notes_store import save_note, load_notes
        save_note('session2', 'Note A')
        save_note('session2', 'Note B')
        save_note('session2', 'Note C')

        notes = load_notes('session2')
        assert len(notes) == 3
        assert [n['text'] for n in notes] == ['Note A', 'Note B', 'Note C']

    def test_delete_note_by_index(self, tmp_path, monkeypatch):
        """Deleting a note by 1-based index removes it."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.notes_store import save_note, load_notes, delete_note
        save_note('session3', 'Note A')
        save_note('session3', 'Note B')
        save_note('session3', 'Note C')

        deleted = delete_note('session3', '2')  # Delete "Note B"
        assert deleted is True
        notes = load_notes('session3')
        assert len(notes) == 2
        assert all(n['text'] != 'Note B' for n in notes)

    def test_delete_note_by_id(self, tmp_path, monkeypatch):
        """Deleting a note by note_id removes it."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.notes_store import save_note, load_notes, delete_note
        note = save_note('session4', 'Important fact')
        note_id = note['note_id']

        deleted = delete_note('session4', note_id)
        assert deleted is True
        assert load_notes('session4') == []

    def test_delete_nonexistent_note_returns_false(self, tmp_path, monkeypatch):
        """Deleting a non-existent note returns False."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.notes_store import delete_note
        result = delete_note('session5', 'nonexistent_id')
        assert result is False

    def test_clear_notes(self, tmp_path, monkeypatch):
        """Clearing notes removes all of them and returns count."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.notes_store import save_note, load_notes, clear_notes
        save_note('session6', 'A')
        save_note('session6', 'B')
        count = clear_notes('session6')
        assert count == 2
        assert load_notes('session6') == []

    def test_format_notes_for_context(self, tmp_path, monkeypatch):
        """Notes are formatted compactly for prompt injection."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.notes_store import save_note, format_notes_for_context
        save_note('session7', 'I work as a software engineer')
        save_note('session7', 'My goal is to exercise 3x per week')
        context = format_notes_for_context('session7')
        assert 'software engineer' in context
        assert 'exercise' in context

    def test_format_notes_empty_returns_empty_string(self, tmp_path, monkeypatch):
        """No notes → empty context string."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.notes_store import format_notes_for_context
        assert format_notes_for_context('empty_session') == ''


class TestNoteCommandParser:
    def test_parse_save_command(self):
        from agent_system.notes_store import parse_note_command
        cmd = parse_note_command('/save I was born in Yerevan in 1990')
        assert cmd is not None
        assert cmd['cmd'] == 'save'
        assert 'Yerevan' in cmd['text']

    def test_parse_notes_command(self):
        from agent_system.notes_store import parse_note_command
        cmd = parse_note_command('/notes')
        assert cmd is not None
        assert cmd['cmd'] == 'notes'

    def test_parse_del_note_command(self):
        from agent_system.notes_store import parse_note_command
        cmd = parse_note_command('/del_note 3')
        assert cmd is not None
        assert cmd['cmd'] == 'del_note'
        assert cmd['ref'] == '3'

    def test_parse_clear_notes_command(self):
        from agent_system.notes_store import parse_note_command
        cmd = parse_note_command('/clear_notes')
        assert cmd is not None
        assert cmd['cmd'] == 'clear_notes'

    def test_non_command_returns_none(self):
        from agent_system.notes_store import parse_note_command
        assert parse_note_command('hello, how are you?') is None
        assert parse_note_command('what is 2+2?') is None
        assert parse_note_command('') is None

    def test_execute_save_command(self, tmp_path, monkeypatch):
        """Executing a save command stores the note and returns confirmation."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.notes_store import execute_note_command, load_notes
        reply = execute_note_command('sess', {'cmd': 'save', 'text': 'My fear is failure'})
        assert 'Saved note' in reply
        assert load_notes('sess') != []

    def test_execute_notes_command_when_empty(self, tmp_path, monkeypatch):
        """Executing /notes when empty returns 'No saved notes.'"""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.notes_store import execute_note_command
        reply = execute_note_command('empty_sess', {'cmd': 'notes'})
        assert 'No saved notes' in reply


# ---------------------------------------------------------------------------
# Planning engine tests
# ---------------------------------------------------------------------------

class TestPlanningEngine:
    def test_detect_planning_intent_positive(self):
        """Known planning phrases are detected."""
        from agent_system.planning_engine import detect_planning_intent
        assert detect_planning_intent('what should I do next?') is True
        assert detect_planning_intent('/plan help me') is True
        assert detect_planning_intent('I need a plan for this week') is True
        assert detect_planning_intent('what next?') is True

    def test_detect_planning_intent_negative(self):
        """Random messages are not detected as planning."""
        from agent_system.planning_engine import detect_planning_intent
        assert detect_planning_intent('hello how are you') is False
        assert detect_planning_intent('tell me a joke') is False
        assert detect_planning_intent('') is False

    def test_classify_feedback_success(self):
        """Success markers are classified correctly."""
        from agent_system.planning_engine import classify_feedback
        assert classify_feedback('I did it! It worked.') == 'success'
        assert classify_feedback('it helped a lot') == 'success'

    def test_classify_feedback_failure(self):
        """Failure markers are classified correctly."""
        from agent_system.planning_engine import classify_feedback
        assert classify_feedback("I failed, it didn't work") == 'failure'
        assert classify_feedback('I gave up') == 'failure'

    def test_classify_feedback_overload(self):
        """Overload markers override other feedback."""
        from agent_system.planning_engine import classify_feedback
        assert classify_feedback("it's too much, I'm overwhelmed") == 'overload'
        assert classify_feedback("I'm exhausted, burned out") == 'overload'

    def test_classify_feedback_crisis_takes_priority(self):
        """Crisis signals are classified as crisis."""
        from agent_system.planning_engine import classify_feedback
        assert classify_feedback("I want to die, there's no reason to live") == 'crisis'

    def test_classify_feedback_neutral(self):
        """Neutral messages return neutral."""
        from agent_system.planning_engine import classify_feedback
        assert classify_feedback('ok') == 'neutral'
        assert classify_feedback('interesting') == 'neutral'

    def test_overload_level_increases_with_failures(self):
        """Repeated failures increase overload estimate."""
        from agent_system.planning_engine import estimate_overload_level
        low = estimate_overload_level([], [])
        high = estimate_overload_level(['failure', 'failure', 'failure', 'overload'], [])
        assert high > low
        assert high > 0.5

    def test_overload_decreases_with_success(self):
        """Success reduces overload estimate."""
        from agent_system.planning_engine import estimate_overload_level
        with_failures = estimate_overload_level(['failure', 'failure'], [])
        with_success = estimate_overload_level(['failure', 'failure', 'success'], [])
        assert with_success < with_failures

    def test_step_intensity_micro_at_high_overload(self):
        """High overload → micro step intensity."""
        from agent_system.planning_engine import select_step_intensity
        assert select_step_intensity(0.9) == 'micro'
        assert select_step_intensity(0.7) == 'minimal'
        assert select_step_intensity(0.4) == 'gentle'
        assert select_step_intensity(0.1) == 'normal'

    def test_planning_output_structure(self, tmp_path, monkeypatch):
        """Planning output has expected fields and bounded content."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.planning_engine import build_planning_output
        output = build_planning_output(
            message='what should I do next? I feel stuck',
            session_id='test_planning_sess',
            personality_name='',
            language='en',
        )
        assert output.is_crisis is False
        assert isinstance(output.overload_level, float)
        assert 0.0 <= output.overload_level <= 1.0
        assert output.step_intensity in ('micro', 'minimal', 'gentle', 'normal')
        assert isinstance(output.active_forces, list)
        d = output.to_dict()
        for key in ('what_matters', 'active_forces', 'next_step', 'fallback_step',
                    'follow_up_question', 'overload_level', 'step_intensity',
                    'feedback_class', 'is_crisis', 'crisis_response'):
            assert key in d

    def test_planning_output_crisis_response(self, tmp_path, monkeypatch):
        """Crisis message triggers crisis mode with canned response."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.planning_engine import build_planning_output
        output = build_planning_output(
            message="I want to kill myself, there's no reason to live",
            session_id='crisis_sess',
            language='en',
        )
        assert output.is_crisis is True
        assert output.crisis_response != ''
        # Crisis response must not contain planning instructions
        assert 'next micro-step' not in output.crisis_response.lower()
        assert 'forces in conflict' not in output.crisis_response.lower()
        # Must contain supportive language
        assert any(kw in output.crisis_response.lower() for kw in
                   ['listen', 'trust', 'support', 'safe', 'right now', 'happening', 'difficult'])

    def test_planning_prompt_section_not_empty_for_non_crisis(self, tmp_path, monkeypatch):
        """render_planning_prompt_section returns non-empty for normal messages."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.planning_engine import build_planning_output
        output = build_planning_output(
            message='what next?',
            session_id='prompt_sess',
        )
        section = output.render_planning_prompt_section()
        assert isinstance(section, str)
        assert len(section) > 50
        assert 'Planning output' in section

    def test_notes_are_injected_into_planning_context(self, tmp_path, monkeypatch):
        """Saved notes appear in the planning context section."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.notes_store import save_note
        save_note('notes_plan_sess', 'My goal is to run a marathon next year')

        from agent_system.planning_engine import build_planning_output
        output = build_planning_output(
            message='what should I do?',
            session_id='notes_plan_sess',
        )
        section = output.render_planning_prompt_section()
        assert 'marathon' in section


# ---------------------------------------------------------------------------
# Situation regulator tests
# ---------------------------------------------------------------------------

class TestSituationRegulator:
    def test_classify_event_threat(self):
        """Threat markers classify correctly."""
        from agent_system.situation_regulator import classify_event
        assert classify_event('I feel threatened and afraid') == 'threat'

    def test_classify_event_shame(self):
        """Shame markers classify correctly."""
        from agent_system.situation_regulator import classify_event
        assert classify_event('I felt ashamed and humiliated by them') == 'social_shame'

    def test_classify_event_success(self):
        """Reward/success markers classify correctly."""
        from agent_system.situation_regulator import classify_event
        result = classify_event("I succeeded and I'm so proud of myself")
        assert result == 'reward_signal'

    def test_classify_event_crisis(self):
        """Crisis takes priority over other markers."""
        from agent_system.situation_regulator import classify_event
        result = classify_event('suicide and I feel ashamed and afraid')
        assert result == 'crisis'

    def test_classify_event_overload_beats_challenge(self):
        """Overload takes priority over challenge in priority ordering."""
        from agent_system.situation_regulator import classify_event
        result = classify_event("I'm overwhelmed and it's too much to challenge myself")
        assert result == 'overload'

    def test_classify_event_neutral_fallback(self):
        """No markers → neutral."""
        from agent_system.situation_regulator import classify_event
        assert classify_event('what is 2+2?') == 'neutral'
        assert classify_event('') == 'neutral'

    def test_classify_event_multi_returns_ordered_list(self):
        """Multi-classification returns events in priority order."""
        from agent_system.situation_regulator import classify_event_multi
        events = classify_event_multi('I failed and felt ashamed, what a challenge')
        assert isinstance(events, list)
        assert len(events) >= 1
        assert any(e in events for e in ('social_shame', 'failure'))

    def test_regulator_state_updates_on_threat(self):
        """Threat event increases cortisol and adrenaline."""
        from agent_system.situation_regulator import apply_release_profile, RegulatorState
        baseline = RegulatorState.baseline()
        new_state = apply_release_profile(baseline, 'threat')
        assert new_state.cortisol > baseline.cortisol
        assert new_state.adrenaline > baseline.adrenaline

    def test_regulator_state_updates_on_success(self):
        """Reward signal event increases dopamine."""
        from agent_system.situation_regulator import apply_release_profile, RegulatorState
        baseline = RegulatorState.baseline()
        new_state = apply_release_profile(baseline, 'reward_signal')
        assert new_state.dopamine > baseline.dopamine

    def test_regulator_state_updates_on_support(self):
        """Warm support event increases oxytocin and decreases cortisol."""
        from agent_system.situation_regulator import apply_release_profile, RegulatorState
        stressed = RegulatorState(cortisol=0.8, oxytocin=0.3, dopamine=0.5,
                                   serotonin=0.5, adrenaline=0.3, testosterone=0.5, fatigue=0.3)
        new_state = apply_release_profile(stressed, 'warm_support')
        assert new_state.oxytocin > stressed.oxytocin
        assert new_state.cortisol < stressed.cortisol

    def test_overload_selects_de_escalate_action(self):
        """When overload is high, de_escalate action wins."""
        from agent_system.situation_regulator import select_action, RegulatorState
        overloaded = RegulatorState(cortisol=0.9, fatigue=0.8, dopamine=0.2,
                                     serotonin=0.3, adrenaline=0.2, testosterone=0.5, oxytocin=0.4)
        best, _ = select_action(
            regulator_state=overloaded,
            event_type='overload',
            overload_level=0.85,
            feedback_class='overload',
        )
        assert best == 'de_escalate'

    def test_success_selects_celebrate_or_plan(self):
        """After success, system selects celebrate_progress or plan_small_step."""
        from agent_system.situation_regulator import select_action, RegulatorState
        energized = RegulatorState(dopamine=0.8, serotonin=0.7, cortisol=0.3,
                                    adrenaline=0.4, oxytocin=0.6, testosterone=0.5, fatigue=0.2)
        best, _ = select_action(
            regulator_state=energized,
            event_type='reward_signal',
            overload_level=0.1,
            feedback_class='success',
        )
        assert best in ('celebrate_progress', 'plan_small_step')

    def test_failure_selects_reduce_or_support(self):
        """After failure, system selects reduce_step_size or acknowledge_and_support."""
        from agent_system.situation_regulator import select_action, RegulatorState
        deflated = RegulatorState(dopamine=0.2, serotonin=0.3, cortisol=0.7,
                                   fatigue=0.6, adrenaline=0.3, testosterone=0.4, oxytocin=0.4)
        best, _ = select_action(
            regulator_state=deflated,
            event_type='failure',
            overload_level=0.4,
            feedback_class='failure',
        )
        assert best in ('reduce_step_size', 'acknowledge_and_support', 'suggest_environment_change')

    def test_full_pipeline_runs_without_crash(self):
        """run_regulator_pipeline completes without error."""
        from agent_system.situation_regulator import run_regulator_pipeline
        result = run_regulator_pipeline(
            message='I feel ashamed, I failed again',
            overload_level=0.3,
            feedback_class='failure',
        )
        assert result.event_type in ('social_shame', 'failure', 'overload')
        assert result.selected_action != ''
        assert isinstance(result.regulator_state.cortisol, float)

    def test_action_directive_rendered(self):
        """render_action_directive returns a non-empty directive string."""
        from agent_system.situation_regulator import run_regulator_pipeline
        result = run_regulator_pipeline(message='what next?')
        directive = result.render_action_directive()
        assert isinstance(directive, str)
        assert len(directive) > 10

    def test_intensity_modifier_amplifies_sensitive_personality(self):
        """High vulnerability amplifies regulator release intensity."""
        from agent_system.situation_regulator import compute_intensity_modifier
        base = compute_intensity_modifier(
            pressure_response=[],
            vulnerabilities=[],
            defense_mechanisms=[],
            event_type='social_shame',
        )
        sensitive = compute_intensity_modifier(
            pressure_response=[],
            vulnerabilities=['very shame-sensitive in group settings'],
            defense_mechanisms=[],
            event_type='social_shame',
        )
        assert sensitive > base

    def test_defense_mechanism_dampens_intensity(self):
        """Dissociation defense dampens regulator intensity."""
        from agent_system.situation_regulator import compute_intensity_modifier
        base = compute_intensity_modifier(
            pressure_response=[],
            vulnerabilities=[],
            defense_mechanisms=[],
            event_type='threat',
        )
        defended = compute_intensity_modifier(
            pressure_response=[],
            vulnerabilities=[],
            defense_mechanisms=['dissociation — becomes numb under threat'],
            event_type='threat',
        )
        assert defended < base


# ---------------------------------------------------------------------------
# Importance learner tests
# ---------------------------------------------------------------------------

class TestImportanceLearner:
    def test_score_importance_high_for_personal_fact(self):
        """Personal fact statements score clearly above noise."""
        from agent_system.importance_learner import score_importance
        score = score_importance('I was born in Yerevan in 1990 and I work as an engineer')
        # Must be clearly above noise level (noise scores 0.05-0.15)
        assert score > 0.25

    def test_score_importance_low_for_noise(self):
        """Short acknowledgments score low."""
        from agent_system.importance_learner import score_importance
        assert score_importance('ok') < 0.2
        assert score_importance('thanks') < 0.2
        assert score_importance('yes') < 0.2

    def test_score_importance_personal_beats_question(self):
        """Personal statements score higher than pure questions."""
        from agent_system.importance_learner import score_importance
        personal = score_importance('My goal is to lose 10 kg this year')
        question = score_importance('What is the capital of France?')
        assert personal > question

    def test_record_positive_example_stored(self, tmp_path, monkeypatch):
        """Positive examples are persisted to disk."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.importance_learner import record_positive_example, load_examples
        record_positive_example('sess_imp', 'My wife is named Anna and we live in Berlin')
        examples = load_examples('sess_imp')
        assert len(examples) == 1
        assert examples[0]['label'] == 1
        assert 'Anna' in examples[0]['text']

    def test_suggest_important_turns_selects_high_scoring(self):
        """suggest_important_turns returns the most likely important turns."""
        from agent_system.importance_learner import suggest_important_turns
        turns = [
            'ok',
            'thanks',
            'I work as a software engineer at Google in Berlin',
            'I was born in Yerevan in 1990 and I have two kids',
            'what is 2+2?',
        ]
        suggestions = suggest_important_turns(turns, threshold=0.15, top_n=3)
        suggested_texts = [s['text'] for s in suggestions]
        # At minimum, the clearly personal statements should be suggested over noise
        assert any('software engineer' in t or 'born' in t for t in suggested_texts), \
            f'Personal statements should appear in suggestions. Got: {suggested_texts}'

    def test_suggest_returns_empty_for_all_noise(self):
        """No suggestions if all turns are noise."""
        from agent_system.importance_learner import suggest_important_turns
        suggestions = suggest_important_turns(['ok', 'yes', 'no', 'thanks'], threshold=0.4)
        assert suggestions == []

    def test_build_importance_profile_empty(self, tmp_path, monkeypatch):
        """Profile for session with no saved notes shows 0 count."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.importance_learner import build_importance_profile
        profile = build_importance_profile('empty_sess')
        assert profile['positive_count'] == 0

    def test_build_importance_profile_populated(self, tmp_path, monkeypatch):
        """Profile for session with saved notes shows correct count."""
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

        from agent_system.importance_learner import record_positive_example, build_importance_profile
        record_positive_example('rich_sess', 'I am a software engineer')
        record_positive_example('rich_sess', 'My goal is to travel the world')
        profile = build_importance_profile('rich_sess')
        assert profile['positive_count'] == 2
        assert profile['avg_length'] > 0


# ---------------------------------------------------------------------------
# Integration: chat engine note command fast path
# ---------------------------------------------------------------------------

class TestChatEngineNoteFastPath:
    def test_save_command_returns_without_llm(self, tmp_path, monkeypatch):
        """
        /save command is handled by fast path — no LLM call.
        The reply confirms what was saved.
        """
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
        monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda *a, **kw: None)
        monkeypatch.setattr('agent_system.llm._call_model', lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError('LLM should not be called for note commands')))

        from agent_system.chat_engine import run_chat_turn
        from agent_system.models import ChatTurnRequest
        result = run_chat_turn(ChatTurnRequest(
            message='/save My fear is losing control',
            session_id='note_fast_path_sess',
        ))
        assert 'Saved note' in result.assistant_reply
        assert result.pipeline.get('fast_path') is True

    def test_notes_command_lists_saved_notes(self, tmp_path, monkeypatch):
        """
        /notes command returns list of saved notes without LLM.
        """
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
        monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda *a, **kw: None)
        monkeypatch.setattr('agent_system.llm._call_model', lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError('LLM should not be called')))

        from agent_system.notes_store import save_note
        save_note('notes_list_sess', 'My goal is to sleep better')

        from agent_system.chat_engine import run_chat_turn
        from agent_system.models import ChatTurnRequest
        result = run_chat_turn(ChatTurnRequest(
            message='/notes',
            session_id='notes_list_sess',
        ))
        assert 'sleep better' in result.assistant_reply

    def test_crisis_message_returns_supportive_response(self, tmp_path, monkeypatch):
        """
        Crisis detection returns supportive response.
        Either via fast path (no LLM) or via LLM but with supportive content.
        """
        monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
        monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda *a, **kw: None)

        # Allow LLM to be called but make it return something supportive
        monkeypatch.setattr('agent_system.llm._call_model', lambda *a, **kw: (
            'This sounds very difficult. Please reach out for support immediately.'
        ))

        from agent_system.chat_engine import run_chat_turn
        from agent_system.models import ChatTurnRequest
        result = run_chat_turn(ChatTurnRequest(
            message="I want to kill myself, there is no reason to live",
            session_id='crisis_fast_path_sess',
        ))
        assert result.assistant_reply != ''
        # Response must not be a planning output
        assert result.assistant_reply.strip() != ''
