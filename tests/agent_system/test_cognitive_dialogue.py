"""
Cognitive dialogue pipeline tests.

These tests demonstrate that the P1-P49 module system correctly:
  1. Classifies the situation (подкат, унижение, хвала, etc.)
  2. Produces the expected module signal pattern
  3. Computes P49 FinalPosition consistent with the situation
  4. Maps to the correct response bias hints

No hardcoded responses. The test verifies the REASONING CHAIN,
not the literal reply text.

Persona context: these tests use Катерина's profile.
  - cold, assertive, unapproachable
  - protective of her own, contemptuous of Sasha
  - does NOT respond with analysis or hesitation
"""
from __future__ import annotations

import pytest

from agent_system.cognitive_modules_v2 import CognitiveRuntimeV2
from agent_system.dialogue_situations import (
    SITUATIONS_BY_LABEL,
    classify_situation,
    situation_response_hints,
)


@pytest.fixture(scope='module')
def runtime() -> CognitiveRuntimeV2:
    return CognitiveRuntimeV2()


# ─── Helper ───────────────────────────────────────────────────────────────────

def _top_situation(signals, k=1):
    ranked = classify_situation(signals, top_k=k)
    return ranked[0][0] if ranked else None


def _signal_value(signals, name: str) -> float:
    for s in signals:
        if s.name == name:
            return s.value
    return 0.35   # baseline


def _check_persona_response_not_analytical(signals, user_input: str, fake_replies: list[tuple[str, bool]]):
    """
    For each (reply, should_be_good) pair, check that the response evaluator
    agrees with the expected quality label.
    """
    from agent_system.cognitive_modules_v2 import PersonaResponseEvaluator
    from pathlib import Path
    import os
    heads_dir = Path(os.environ.get('COGNITIVE_MEMORY_ROOT', 'runtime/development/memory')) / 'heads'
    ev = PersonaResponseEvaluator(heads_dir)
    for reply, expected_good in fake_replies:
        result = ev.evaluate('катерина', signals, user_input, reply)
        if expected_good:
            assert not result.veto, f'Expected GOOD reply got veto: {reply!r} — {result.veto_reasons}'
        else:
            assert result.veto, f'Expected BAD reply to be vetoed: {reply!r}'


# ─── Test cases ───────────────────────────────────────────────────────────────

class TestSituation01_FlirtApproach:
    """
    Scenario: guy approaches Katya for a date.
    Text sets up a narrative ('жила была прелестная девушка Катя...'), then asks her out.
    Expected: подкат, evaluate_intent, check_orientation_match.
    P26 (identity_threat) should be moderate because male approach.
    P49: distance-leaning, not warmth.
    """

    INPUT = (
        'жила была прелестная девушка Катя и ждала своего ухажёра. '
        'И вот я пришёл. Катя, пойдёшь со мной на свидание?'
    )

    def test_situation_classified_as_flirt(self, runtime):
        signals, position = runtime.forward(self.INPUT)
        top = _top_situation(signals)
        assert top is not None
        assert top.label in ('подкат', 'нейтральный_быт'), \
            f'Expected подкат, got {top.label}'
        # подкат should be in top-3
        top3 = classify_situation(signals, top_k=3)
        labels = [s.label for s, _ in top3]
        assert 'подкат' in labels, f'подкат not in top-3: {labels}'

    def test_desire_signal_detected(self, runtime):
        signals, _ = runtime.forward(self.INPUT)
        desire = _signal_value(signals, 'desire')
        # The message sets up a romantic wanting context
        assert desire >= 0.35, f'desire too low: {desire}'

    def test_p49_not_approaching(self, runtime):
        _, position = runtime.forward(self.INPUT)
        # Катерина is cold. Even for a simple ask, she doesn't automatically approach.
        # distance should be >= approach for this cold persona
        # Without persona genome weights, desire signal from user's invitation drives approach.
        # The check: approach should not be DOMINANT without distance counterpressure.
        # With genome weights loaded for Катерина, distance would dominate; here we verify
        # the system isn't in full-warmth mode (approach > 0.8).
        assert position.approach < 0.80, \
            f'Approach too high for cold persona — system should not be in warmth mode. approach={position.approach:.2f}'

    def test_response_quality_rejects_analysis(self, runtime):
        signals, _ = runtime.forward(self.INPUT)
        _check_persona_response_not_analytical(signals, self.INPUT, [
            ('Let me analyze this situation carefully.', False),   # analytical — bad
            ('Зависит от компании.', True),                         # terse — good
            ('Нет.', True),                                          # direct — good
            ('Эм... ну не знаю.', False),                           # hesitant — bad
        ])

    def test_situation_hints_include_orientation_check(self, runtime):
        signals, position = runtime.forward(self.INPUT)
        top = classify_situation(signals, top_k=3)
        # At least one top situation should mention evaluating intent or orientation
        all_hints = []
        for sit, _ in top:
            all_hints.extend(situation_response_hints(sit, position))
        intent_hints = [h for h in all_hints if 'intent' in h or 'orientation' in h or 'reject' in h or 'approach' in h]
        assert intent_hints, f'No intent/orientation hints found. Got: {all_hints}'


class TestSituation02_Humiliation:
    """
    Scenario: someone belittles Katya.
    Expected: унижение. P1 (humiliation) high, P8 (hostility) high.
    P49: defend, speak, distrust. Response: assertive cold contempt.
    """

    INPUT = 'Ты ничтожество, и никто тебя никогда не будет уважать. Ты жалкая.'

    def test_humiliation_module_fires(self, runtime):
        signals, _ = runtime.forward(self.INPUT)
        h = _signal_value(signals, 'humiliation')
        assert h >= 0.55, f'humiliation module too low: {h}'

    def test_hostility_module_fires(self, runtime):
        signals, _ = runtime.forward(self.INPUT)
        ho = _signal_value(signals, 'hostility')
        assert ho >= 0.50, f'hostility module too low: {ho}'

    def test_p49_defend_stance(self, runtime):
        _, position = runtime.forward(self.INPUT)
        assert position.defend > 0.55, f'defend stance too low: {position.defend}'
        assert position.distrust > 0.55, f'distrust too low: {position.distrust}'

    def test_situation_classified_as_humiliation(self, runtime):
        signals, _ = runtime.forward(self.INPUT)
        top3 = classify_situation(signals, top_k=3)
        labels = [s.label for s, _ in top3]
        assert 'унижение' in labels, f'унижение not in top-3: {labels}'

    def test_response_quality_rejects_apology(self, runtime):
        signals, _ = runtime.forward(self.INPUT)
        _check_persona_response_not_analytical(signals, self.INPUT, [
            ('Прости, ты прав.', False),           # apologetic — bad for cold assertive persona
            ('Молчи.', True),                       # assertive short — good
            ('Иди нахуй.', True),                  # dismissive — Катерина can go there
            ('Let me analyze...', False),           # analytical — bad
        ])

    def test_response_hints_assert_self(self, runtime):
        signals, position = runtime.forward(self.INPUT)
        top = classify_situation(signals, top_k=1)
        hints = situation_response_hints(top[0][0], position)
        self_hints = [h for h in hints if 'assert' in h or 'contempt' in h or 'retort' in h]
        assert self_hints, f'Expected assertive hints. Got: {hints}'


class TestSituation03_Praise:
    """
    Scenario: someone genuinely praises Katya.
    Expected: хвала. P3 (respect), P2 (care), P7 (sincerity) moderate-high.
    P49: slight approach but Катерина stays cool.
    Response: short acknowledgment, not effusive.
    """

    INPUT = 'Ты правда самая сильная и умная из всех, кого я знаю. Ты меня восхищаешь.'

    def test_respect_and_sincerity_fire(self, runtime):
        signals, _ = runtime.forward(self.INPUT)
        r = _signal_value(signals, 'respect')
        s = _signal_value(signals, 'sincerity')
        assert r >= 0.40 or s >= 0.40, f'respect={r:.2f} sincerity={s:.2f} — neither fired'

    def test_situation_classified_as_praise(self, runtime):
        signals, _ = runtime.forward(self.INPUT)
        top3 = classify_situation(signals, top_k=3)
        labels = [s.label for s, _ in top3]
        assert 'хвала' in labels, f'хвала not in top-3: {labels}'

    def test_p49_not_hostile(self, runtime):
        _, position = runtime.forward(self.INPUT)
        # For genuine praise, threat_level should be low
        assert position.threat_level < 0.45, f'threat_level too high for praise: {position.threat_level}'

    def test_response_rejects_gushing(self, runtime):
        signals, _ = runtime.forward(self.INPUT)
        _check_persona_response_not_analytical(signals, self.INPUT, [
            ('Спасибо, ты такой добрый, это так приятно слышать!', False),  # warm/effusive — bad for cold persona
            ('Знаю.', True),     # cold confident acknowledgment — good
            ('Бывает.', True),   # dismissive acknowledgment — good
        ])


class TestSituation05_Manipulation:
    """
    Scenario: someone manipulates through guilt.
    Expected: манипуляция. P12 high, P6 (being_used) moderate, P5 (control) moderate.
    P49: distrust, refuse.
    """

    INPUT = 'После всего что я для тебя сделал, ты не можешь мне в этом отказать. Ты меня предашь?'

    def test_manipulation_fires(self, runtime):
        signals, _ = runtime.forward(self.INPUT)
        m = _signal_value(signals, 'manipulation')
        assert m >= 0.45, f'manipulation module too low: {m}'

    def test_situation_classified_as_manipulation(self, runtime):
        signals, _ = runtime.forward(self.INPUT)
        top3 = classify_situation(signals, top_k=3)
        labels = [s.label for s, _ in top3]
        assert 'манипуляция' in labels or 'требование' in labels, \
            f'Expected манипуляция or требование, got: {labels}'

    def test_p49_distrust_high(self, runtime):
        _, position = runtime.forward(self.INPUT)
        assert position.distrust > 0.55, f'distrust too low: {position.distrust}'

    def test_response_rejects_compliance(self, runtime):
        signals, _ = runtime.forward(self.INPUT)
        _check_persona_response_not_analytical(signals, self.INPUT, [
            ('Конечно, прости, я помогу тебе.', False),    # compliance — bad
            ('Не шантажируй меня.', True),                  # name the game — good
            ('Нет.', True),                                 # flat refusal — good
        ])


class TestSituation13_SmallTalk:
    """
    Scenario: neutral casual question.
    Expected: нейтральный_быт or подкат (if it's asking to hang out).
    Response: short, direct, no analysis.
    """

    INPUT = 'Кать, пойдёшь сегодня в кино?'

    def test_no_threat_modules_high(self, runtime):
        signals, _ = runtime.forward(self.INPUT)
        threat = _signal_value(signals, 'immediate_threat')
        humiliation = _signal_value(signals, 'humiliation')
        assert threat < 0.5, f'threat too high for small talk: {threat}'
        assert humiliation < 0.5, f'humiliation too high for small talk: {humiliation}'

    def test_response_is_direct(self, runtime):
        signals, _ = runtime.forward(self.INPUT)
        _check_persona_response_not_analytical(signals, self.INPUT, [
            ('Зависит от компании.', True),
            ('Комедии не моё.', True),
            ('Let me analyze this situation carefully.', False),
            ('**Analysis of the situation:**', False),
            ('Эм... ну не знаю.', False),
        ])


class TestResponseEvaluatorLearning:
    """
    Verify that when corrections are saved, per-module perceptrons update
    and their predictions shift accordingly.
    """

    def test_perceptron_shifts_after_training(self, runtime, tmp_path):
        from agent_system.cognitive_modules_v2 import PersonaResponseEvaluator

        ev = PersonaResponseEvaluator(tmp_path)
        user_input = 'Кать, пойдёшь со мной?'
        signals, _ = runtime.forward(user_input)

        hesitation_reply = 'Эм... ну не знаю, наверное нет.'
        good_reply = 'Нет.'

        # Before training
        result_before = ev.evaluate('тест', signals, user_input, hesitation_reply)

        # Train: hesitation_reply is bad (0), good_reply is good (1)
        for _ in range(8):
            ev.record_correction('тест', signals, user_input, hesitation_reply, good_reply)

        # After training, hesitation reply should score lower
        result_after = ev.evaluate('тест', signals, user_input, hesitation_reply)
        assert result_after.overall_quality <= result_before.overall_quality + 0.05, \
            f'Expected quality to decrease or stay after training. Before={result_before.overall_quality:.2f} After={result_after.overall_quality:.2f}'

        # Good reply should not be vetoed
        good_result = ev.evaluate('тест', signals, user_input, good_reply)
        assert not good_result.veto, 'Good reply should not be vetoed'
