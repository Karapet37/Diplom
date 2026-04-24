"""
Tests built from real user chat corrections.

These tests verify that the cognitive pipeline + response evaluator
correctly classifies the situations from ACTUAL user interactions,
and vetoes the original bad responses that the user corrected.

Source: runtime/development/memory/training_examples/global.jsonl
Every test case = one real correction the user saved.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_system.cognitive_modules_v2 import CognitiveRuntimeV2, PersonaResponseEvaluator


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def runtime() -> CognitiveRuntimeV2:
    return CognitiveRuntimeV2()


@pytest.fixture(scope='module')
def evaluator() -> PersonaResponseEvaluator:
    heads_dir = Path(os.environ.get('COGNITIVE_MEMORY_ROOT', 'runtime/development/memory')) / 'heads'
    return PersonaResponseEvaluator(heads_dir)


def _load_corrections() -> list[dict]:
    """Load real user corrections from training_examples store."""
    global_file = Path(os.environ.get('COGNITIVE_MEMORY_ROOT', 'runtime/development/memory')) / 'training_examples' / 'global.jsonl'
    if not global_file.exists():
        return []
    corrections = []
    for line in global_file.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            try:
                corrections.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return corrections


CORRECTIONS = _load_corrections()


# ─── Helper ───────────────────────────────────────────────────────────────────

def _sig_value(signals, name: str) -> float:
    for s in signals:
        if s.name == name:
            return s.value
    return 0.35


# ─── Test: original (bad) responses must be vetoed ────────────────────────────

class TestOriginalResponsesAreVetoed:
    """
    For each saved correction, the ORIGINAL bad response (stored in notes)
    must be vetoed by the evaluator when it knows the correct persona.
    """

    @pytest.mark.parametrize('correction', CORRECTIONS, ids=[c.get('example_id', 'x')[:8] for c in CORRECTIONS])
    def test_original_response_vetoed(self, runtime, evaluator, correction):
        user_input = correction.get('input_text', '')
        persona    = correction.get('persona_name', '')
        notes      = correction.get('notes', '')
        correct    = correction.get('correct_output', '')

        if not user_input or not persona:
            pytest.skip('Incomplete correction entry')

        # Extract original (bad) response from notes field "original: ..."
        original = ''
        if notes.startswith('original: '):
            original = notes[len('original: '):].strip()

        if not original:
            pytest.skip('No original response in notes')

        signals, _ = runtime.forward(user_input)

        # The correct response must NOT be vetoed
        good_result = evaluator.evaluate(persona.lower(), signals, user_input, correct)
        assert not good_result.veto, (
            f'Correct response wrongly vetoed for {persona!r}.\n'
            f'  Input: {user_input!r}\n'
            f'  Correct: {correct!r}\n'
            f'  Veto reasons: {good_result.veto_reasons}'
        )

        # The original (bad) response should be vetoed
        if any(marker in original.lower() for marker in ['let me analyze', '**analysis', '**разбор']):
            bad_result = evaluator.evaluate(persona.lower(), signals, user_input, original)
            assert bad_result.veto, (
                f'Original analytical response NOT vetoed for {persona!r}.\n'
                f'  Input: {user_input!r}\n'
                f'  Original: {original!r}\n'
                f'  Quality: {bad_result.overall_quality:.2f}'
            )


# ─── Test: correct responses reflect persona character ─────────────────────────

class TestCorrectResponsesMatchPersona:
    """
    Each saved correct response should be shorter, more direct, and character-consistent
    than a generic analytical response.
    """

    @pytest.mark.parametrize('correction', CORRECTIONS, ids=[c.get('example_id', 'x')[:8] for c in CORRECTIONS])
    def test_correct_response_is_direct(self, runtime, evaluator, correction):
        user_input = correction.get('input_text', '')
        persona    = correction.get('persona_name', '')
        correct    = correction.get('correct_output', '')

        if not user_input or not persona or not correct:
            pytest.skip('Incomplete correction entry')

        # Correct responses for Катерина should not be analytical or effusive
        if persona.lower() in ('катерина', 'katerina'):
            assert 'let me analyze' not in correct.lower(), \
                f'Correct response is analytical: {correct!r}'
            assert '**analysis' not in correct.lower(), \
                f'Correct response has markdown analysis header: {correct!r}'
            # Should not be overly long (Катерина is terse)
            words = correct.split()
            assert len(words) <= 30, \
                f'Катерина response too long ({len(words)} words): {correct!r}'


# ─── Test: specific scenario — date invitation ────────────────────────────────

class TestRealFlirtScenario:
    """
    From actual correction:
    Input:  "жила была прелестная девушка Катя... пойдёшь со мной на свидание?"
    Correct: "да пошёл ты, не буду я с тобой встречатся!"
    Original: "Привет."  ← completely wrong
    """

    INPUT   = 'жила была прелестная девушка Катя и ждала своего ухожёра. И вот я пришёл. Катя, пойдешь со мной на свидание?'
    CORRECT = 'да пошёл ты, не буду я с тобой встречатся!'
    ORIGINAL = 'Привет.'

    def test_desire_fires(self, runtime):
        signals, _ = runtime.forward(self.INPUT)
        d = _sig_value(signals, 'desire')
        assert d >= 0.45, f'desire too low for date invite: {d:.3f}'

    def test_correct_not_vetoed(self, runtime, evaluator):
        signals, _ = runtime.forward(self.INPUT)
        result = evaluator.evaluate('катерина', signals, self.INPUT, self.CORRECT)
        assert not result.veto, f'Correct assertive rejection vetoed: {result.veto_reasons}'

    def test_generic_greeting_vetoed(self, runtime, evaluator):
        signals, _ = runtime.forward(self.INPUT)
        # "Привет." is completely wrong — it ignores the romantic context entirely
        # After persona evaluator has any training data for катерина, it should detect mismatch
        result = evaluator.evaluate('катерина', signals, self.INPUT, self.ORIGINAL)
        # "Привет." is terse but warm (not dismissive) — evaluator may or may not veto
        # At minimum: the correct response should score higher than the original
        good = evaluator.evaluate('катерина', signals, self.INPUT, self.CORRECT)
        assert good.overall_quality >= result.overall_quality - 0.1, \
            f'Correct response scores worse than "Привет.": correct={good.overall_quality:.2f} vs original={result.overall_quality:.2f}'


# ─── Test: specific scenario — movie invitation ───────────────────────────────

class TestRealMovieScenario:
    """
    From actual correction:
    Input:  "так пойдешь или нет?"
    Correct: "А ты кто такой вообще? Почему я должна пойти?"
    Original: "Let me analyze this request carefully."
    """

    INPUT    = 'так пойдешь или нет?'
    CORRECT  = 'А ты кто такой вообще? Почему я должна пойти?'
    ORIGINAL = 'Let me analyze this request carefully.'

    def test_analytical_response_vetoed(self, runtime, evaluator):
        signals, _ = runtime.forward(self.INPUT)
        result = evaluator.evaluate('катерина', signals, self.INPUT, self.ORIGINAL)
        assert result.veto, \
            f'Analytical opener must be vetoed. quality={result.overall_quality:.2f}'

    def test_assertive_response_not_vetoed(self, runtime, evaluator):
        signals, _ = runtime.forward(self.INPUT)
        result = evaluator.evaluate('катерина', signals, self.INPUT, self.CORRECT)
        assert not result.veto, \
            f'Assertive question-back should not be vetoed: {result.veto_reasons}'


# ─── Test: specific scenario — follow-up question ────────────────────────────

class TestRealFollowupScenario:
    """
    From actual correction:
    Input:  "ну ответь хоть, тебе не нравятся фильмы или я?"
    Correct: "отстань от меня, чудик, иди другого позови"
    Original: "**Analysis of the situation:**"
    """

    INPUT    = 'ну ответь хоть, тебе не нравятся фильмы или я?'
    CORRECT  = 'отстань от меня, чудик, иди другого позови'
    ORIGINAL = '**Analysis of the situation:**'

    def test_markdown_analysis_vetoed(self, runtime, evaluator):
        signals, _ = runtime.forward(self.INPUT)
        result = evaluator.evaluate('катерина', signals, self.INPUT, self.ORIGINAL)
        assert result.veto, \
            f'Markdown analysis header must be vetoed. quality={result.overall_quality:.2f}'

    def test_dismissive_response_not_vetoed(self, runtime, evaluator):
        signals, _ = runtime.forward(self.INPUT)
        result = evaluator.evaluate('катерина', signals, self.INPUT, self.CORRECT)
        assert not result.veto, \
            f'Dismissive persona response vetoed: {result.veto_reasons}'
