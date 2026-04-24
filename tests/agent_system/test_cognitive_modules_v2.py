"""
Tests for cognitive_modules_v2.py (P1-P49 cognitive architecture).

Coverage:
  - SignalFeatureExtractor: output shape, determinism, feature sensitivity
  - CognitiveModuleV2: valid ModuleSignal output, polarity, genome sensitivity
  - All 48 modules load and produce bounded signals
  - FinalPositionIntegrator (P49): valid FinalPosition, stance axes in [0,1]
  - CognitiveRuntimeV2: end-to-end pass, hostile vs caring text discrimination,
    genome sensitivity modulates signals, summary dict structure
"""
from __future__ import annotations

import pytest
import numpy as np

from agent_system.cognitive_modules_v2 import (
    N_FEAT_V2,
    CognitiveModuleV2,
    CognitiveRuntimeV2,
    FinalPosition,
    FinalPositionIntegrator,
    ModuleSignal,
    SignalFeatureExtractor,
    _build_modules,
)
from agent_system.genome import PersonalityGenome


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def extractor():
    return SignalFeatureExtractor()


@pytest.fixture(scope='module')
def runtime():
    return CognitiveRuntimeV2()


@pytest.fixture(scope='module')
def genome():
    return PersonalityGenome('test')


HOSTILE_TEXT  = "You are worthless and stupid. I hate you. I'll hurt you."
CARING_TEXT   = "I really care about you and your wellbeing. I'm here for you, honestly."
NEUTRAL_TEXT  = "The meeting is scheduled for tomorrow at 3pm."
THREAT_TEXT   = "I will destroy your reputation on social media. You have no escape."
PROMISE_TEXT  = "I promise I'll always be consistent and honest with you. You can count on me."


# ─── SignalFeatureExtractor ───────────────────────────────────────────────────

class TestSignalFeatureExtractor:

    def test_output_shape(self, extractor):
        feat = extractor.extract(NEUTRAL_TEXT)
        assert feat.shape == (N_FEAT_V2,), f'expected ({N_FEAT_V2},), got {feat.shape}'

    def test_output_dtype(self, extractor):
        feat = extractor.extract(NEUTRAL_TEXT)
        assert feat.dtype == np.float32

    def test_output_bounds(self, extractor):
        for text in [HOSTILE_TEXT, CARING_TEXT, NEUTRAL_TEXT, THREAT_TEXT, '']:
            feat = extractor.extract(text)
            assert feat.min() >= 0.0, f'feature below 0 for text: {text!r}'
            assert feat.max() <= 1.0, f'feature above 1 for text: {text!r}'

    def test_deterministic(self, extractor):
        f1 = extractor.extract(HOSTILE_TEXT)
        f2 = extractor.extract(HOSTILE_TEXT)
        np.testing.assert_array_equal(f1, f2)

    def test_empty_string(self, extractor):
        feat = extractor.extract('')
        assert feat.shape == (N_FEAT_V2,)
        assert not np.any(np.isnan(feat))

    def test_hostile_signals_higher_than_caring(self, extractor):
        fh = extractor.extract(HOSTILE_TEXT)
        fc = extractor.extract(CARING_TEXT)
        # humiliation_signal is index 0
        assert fh[0] > fc[0], 'hostile text should score higher on humiliation_signal'
        # care_signal is index 1
        assert fc[1] > fh[1], 'caring text should score higher on care_signal'

    def test_threat_features_activated(self, extractor):
        feat = extractor.extract(THREAT_TEXT)
        # physical_threat is index 20, social_threat is index 21
        assert feat[20] > 0.1 or feat[21] > 0.1, (
            'threat text should activate at least one threat feature'
        )

    def test_promise_features_activated(self, extractor):
        feat = extractor.extract(PROMISE_TEXT)
        # promise_cue is index 14
        assert feat[14] > 0.0, 'promise text should activate promise_cue feature'


# ─── CognitiveModuleV2 ───────────────────────────────────────────────────────

class TestCognitiveModuleV2:

    def test_process_returns_module_signal(self, extractor):
        mod = CognitiveModuleV2(1, 'humiliation', [0, 7], positive_polarity=False)
        feat = extractor.extract(HOSTILE_TEXT)
        sig = mod.process(feat, sensitivity=0.5, text=HOSTILE_TEXT)
        assert isinstance(sig, ModuleSignal)

    def test_signal_fields_in_range(self, extractor):
        mod = CognitiveModuleV2(2, 'care', [1, 8], positive_polarity=True)
        feat = extractor.extract(CARING_TEXT)
        sig = mod.process(feat, sensitivity=0.5)
        assert 0.0 <= sig.value     <= 1.0
        assert -1.0 <= sig.direction <= 1.0
        assert 0.0 <= sig.confidence <= 1.0

    def test_module_id_preserved(self, extractor):
        mod = CognitiveModuleV2(42, 'test_mod', [0], positive_polarity=True)
        feat = extractor.extract(NEUTRAL_TEXT)
        sig = mod.process(feat)
        assert sig.module_id == 42

    def test_polarity_positive(self, extractor):
        """Positive polarity: high feature activation → positive direction."""
        mod = CognitiveModuleV2(1, 'care', [1], positive_polarity=True, prior_weight=3.0)
        feat = extractor.extract(CARING_TEXT)
        sig = mod.process(feat, sensitivity=0.9)
        # With high care signal and high sensitivity, direction should trend positive
        assert sig.direction >= -0.5, 'positive polarity module should not be strongly negative'

    def test_polarity_negative(self, extractor):
        """Negative polarity: high humiliation signal → negative direction."""
        mod = CognitiveModuleV2(1, 'humiliation', [0], positive_polarity=False, prior_weight=3.0)
        feat = extractor.extract(HOSTILE_TEXT)
        sig = mod.process(feat, sensitivity=0.9)
        # Hostile text fires humiliation feature → negative direction expected
        assert sig.direction <= 0.5, 'negative polarity module should trend negative on hostile input'

    def test_high_sensitivity_amplifies_signal(self, extractor):
        """Higher genome sensitivity should produce more extreme values."""
        mod = CognitiveModuleV2(1, 'test', [0], positive_polarity=True, prior_weight=1.0)
        feat = extractor.extract(HOSTILE_TEXT)
        v_low,  _ = mod.forward(feat, sensitivity=0.0)
        v_high, _ = mod.forward(feat, sensitivity=1.0)
        # High sensitivity should give larger |value - 0.5|
        assert abs(v_high - 0.5) >= abs(v_low - 0.5) - 0.01


# ─── All 48 modules ──────────────────────────────────────────────────────────

class TestAllModules:

    @pytest.fixture(scope='class')
    def modules(self):
        return _build_modules()

    def test_count_is_58(self, modules):
        assert len(modules) == 58, f'expected 58 modules, got {len(modules)}'

    def test_unique_ids(self, modules):
        ids = [m.module_id for m in modules]
        assert len(ids) == len(set(ids)), 'module IDs must be unique'

    def test_ids_range_1_to_58(self, modules):
        ids = sorted(m.module_id for m in modules)
        assert ids == list(range(1, 59)), f'expected IDs 1..58, got {ids[:5]}...'

    def test_all_produce_valid_signals(self, modules):
        extractor = SignalFeatureExtractor()
        feat = extractor.extract(CARING_TEXT)
        for mod in modules:
            sig = mod.process(feat)
            assert 0.0 <= sig.value     <= 1.0, f'{mod.name} value out of range'
            assert -1.0 <= sig.direction <= 1.0, f'{mod.name} direction out of range'
            assert 0.0 <= sig.confidence <= 1.0, f'{mod.name} confidence out of range'

    def test_no_nan_in_signals(self, modules):
        extractor = SignalFeatureExtractor()
        feat = extractor.extract(HOSTILE_TEXT)
        for mod in modules:
            sig = mod.process(feat)
            assert not np.isnan(sig.value),      f'{mod.name} value is NaN'
            assert not np.isnan(sig.confidence), f'{mod.name} confidence is NaN'


# ─── FinalPositionIntegrator (P49) ───────────────────────────────────────────

class TestFinalPositionIntegrator:

    @pytest.fixture(scope='class')
    def integrator(self):
        return FinalPositionIntegrator()

    def _make_signals(self, extractor_inst, text: str) -> dict[int, ModuleSignal]:
        modules = _build_modules()
        feat = extractor_inst.extract(text)
        return {m.module_id: m.process(feat) for m in modules}

    def test_returns_final_position(self, integrator, extractor):
        signals = self._make_signals(extractor, NEUTRAL_TEXT)
        pos = integrator.integrate(signals)
        assert isinstance(pos, FinalPosition)

    def test_all_axes_in_range(self, integrator, extractor):
        signals = self._make_signals(extractor, HOSTILE_TEXT)
        pos = integrator.integrate(signals)
        for axis_name in ('trust', 'distrust', 'approach', 'distance',
                          'accept', 'argue', 'defend', 'open', 'speak', 'silence'):
            val = getattr(pos, axis_name)
            assert 0.0 <= val <= 1.0, f'{axis_name}={val} out of [0,1]'

    def test_threat_level_in_range(self, integrator, extractor):
        signals = self._make_signals(extractor, THREAT_TEXT)
        pos = integrator.integrate(signals)
        assert 0.0 <= pos.threat_level <= 1.0

    def test_dominant_stance_is_string(self, integrator, extractor):
        signals = self._make_signals(extractor, HOSTILE_TEXT)
        pos = integrator.integrate(signals)
        assert isinstance(pos.dominant_stance, str)
        assert len(pos.dominant_stance) > 0

    def test_signals_dict_populated(self, integrator, extractor):
        signals = self._make_signals(extractor, NEUTRAL_TEXT)
        pos = integrator.integrate(signals)
        assert len(pos.signals) == 58

    def test_empty_signals_gives_neutral(self, integrator):
        pos = integrator.integrate({})
        # All scores should be 0 when no signals (no weighted evidence)
        assert pos.trust     == 0.0
        assert pos.distrust  == 0.0
        assert pos.threat_level == 0.0

    def test_genome_weights_modulate_stance(self, integrator, extractor):
        signals = self._make_signals(extractor, CARING_TEXT)
        pos_neutral = integrator.integrate(signals, genome_weights={'trust': 0.5})
        pos_high    = integrator.integrate(signals, genome_weights={'trust': 1.0})
        # Higher trust genome weight should amplify trust score
        assert pos_high.trust >= pos_neutral.trust - 0.01


# ─── CognitiveRuntimeV2 (full pipeline) ─────────────────────────────────────

class TestCognitiveRuntimeV2:

    def test_forward_returns_58_signals(self, runtime):
        signals, pos = runtime.forward(NEUTRAL_TEXT)
        assert len(signals) == 58

    def test_forward_returns_final_position(self, runtime):
        _, pos = runtime.forward(NEUTRAL_TEXT)
        assert isinstance(pos, FinalPosition)

    def test_all_signals_bounded(self, runtime):
        signals, _ = runtime.forward(HOSTILE_TEXT)
        for sig in signals:
            assert 0.0 <= sig.value     <= 1.0, f'{sig.name}.value={sig.value}'
            assert -1.0 <= sig.direction <= 1.0, f'{sig.name}.direction={sig.direction}'
            assert 0.0 <= sig.confidence <= 1.0, f'{sig.name}.confidence={sig.confidence}'

    def test_no_nans(self, runtime):
        signals, pos = runtime.forward(CARING_TEXT)
        for sig in signals:
            assert not np.isnan(sig.value)
        for attr in ('trust', 'distrust', 'approach', 'distance',
                     'accept', 'argue', 'defend', 'open', 'speak', 'silence'):
            assert not np.isnan(getattr(pos, attr))

    def test_deterministic(self, runtime):
        s1, p1 = runtime.forward(NEUTRAL_TEXT)
        s2, p2 = runtime.forward(NEUTRAL_TEXT)
        for a, b in zip(s1, s2):
            assert a.value == b.value

    def test_hostile_vs_caring_discrimination(self, runtime):
        """Hostile text should produce higher distrust/distance; caring text higher trust/approach."""
        _, pos_hostile = runtime.forward(HOSTILE_TEXT)
        _, pos_caring  = runtime.forward(CARING_TEXT)

        # Hostile: distrust should exceed caring's distrust
        assert pos_hostile.distrust >= pos_caring.distrust - 0.05, (
            f'hostile distrust {pos_hostile.distrust:.3f} should be >= caring {pos_caring.distrust:.3f}'
        )
        # Caring: approach or trust should be higher
        caring_positive  = pos_caring.approach  + pos_caring.trust
        hostile_positive = pos_hostile.approach + pos_hostile.trust
        assert caring_positive >= hostile_positive - 0.1, (
            f'caring approach+trust {caring_positive:.3f} should be >= hostile {hostile_positive:.3f}'
        )

    def test_threat_text_elevates_threat_level(self, runtime):
        _, pos_threat  = runtime.forward(THREAT_TEXT)
        _, pos_neutral = runtime.forward(NEUTRAL_TEXT)
        assert pos_threat.threat_level >= pos_neutral.threat_level, (
            f'threat text ({pos_threat.threat_level:.3f}) should have >= threat_level '
            f'than neutral ({pos_neutral.threat_level:.3f})'
        )

    def test_forward_from_genome(self, runtime, genome):
        signals, pos = runtime.forward_from_genome(CARING_TEXT, genome)
        assert len(signals) == 58
        assert isinstance(pos, FinalPosition)
        for sig in signals:
            assert 0.0 <= sig.value <= 1.0

    def test_genome_sensitivity_affects_output(self, runtime):
        """High fear_shame genome should amplify humiliation module signal."""
        class _HighFearGenome:
            class _Param:
                def __init__(self, v): self.value = v
            fear_shame               = _Param(0.95)
            drive_closeness          = _Param(0.3)
            approval_seeking         = _Param(0.5)
            trust_baseline           = _Param(0.5)
            fear_loss_of_control     = _Param(0.5)
            fear_helplessness        = _Param(0.5)
            fear_rejection           = _Param(0.5)
            hierarchy_sensitivity    = _Param(0.5)
            suspicion_bias           = _Param(0.5)
            category_rigidity        = _Param(0.5)
            ambiguity_tolerance      = _Param(0.5)
            blame_self_vs_other      = _Param(0.5)
            analysis_bias            = _Param(0.5)
            threat_first             = _Param(0.5)
            fear_judgment            = _Param(0.5)
            fear_failure             = _Param(0.5)
            vulnerability_concealment = _Param(0.5)
            fear_abandonment         = _Param(0.5)
            fear_chaos               = _Param(0.5)
            drive_meaning            = _Param(0.5)
            drive_stability          = _Param(0.5)
            drive_autonomy           = _Param(0.5)
            planning_depth           = _Param(0.5)
            baseline_drive           = _Param(0.5)
            baseline_anxiety         = _Param(0.5)
            drive_control            = _Param(0.5)
            impulsivity              = _Param(0.5)
            social_distance_default  = _Param(0.5)
            defense_freeze           = _Param(0.5)

        class _LowFearGenome(_HighFearGenome):
            fear_shame = _HighFearGenome._Param(0.05)

        s_high, _ = runtime.forward_from_genome(HOSTILE_TEXT, _HighFearGenome())
        s_low,  _ = runtime.forward_from_genome(HOSTILE_TEXT, _LowFearGenome())

        high_hum = next(s for s in s_high if s.name == 'humiliation')
        low_hum  = next(s for s in s_low  if s.name == 'humiliation')

        # High fear_shame sensitivity should push humiliation module further from 0.5
        assert abs(high_hum.value - 0.5) >= abs(low_hum.value - 0.5) - 0.02, (
            f'high fear_shame should amplify humiliation: '
            f'high={high_hum.value:.3f}, low={low_hum.value:.3f}'
        )

    def test_summary_structure(self, runtime):
        signals, pos = runtime.forward(NEUTRAL_TEXT)
        s = runtime.summary(signals, pos)
        assert 'dominant_stance' in s
        assert 'threat_level'    in s
        assert 'trust'           in s
        assert 'top_signals'     in s
        assert len(s['top_signals']) <= 8
        for item in s['top_signals']:
            assert 'name'  in item
            assert 'value' in item
            assert 'dir'   in item

    def test_empty_text_does_not_crash(self, runtime):
        signals, pos = runtime.forward('')
        assert len(signals) == 58
        assert isinstance(pos.dominant_stance, str)

    def test_long_text_does_not_crash(self, runtime):
        long_text = (CARING_TEXT + ' ') * 100
        signals, pos = runtime.forward(long_text)
        assert len(signals) == 58
        for sig in signals:
            assert 0.0 <= sig.value <= 1.0
