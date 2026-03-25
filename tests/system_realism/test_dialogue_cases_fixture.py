from __future__ import annotations

from tests.system_realism.dialogue_cases import canonical_dialogue_benchmark, canonical_dialogue_cases, canonical_dialogue_case_groups


def test_canonical_dialogue_benchmark_structure() -> None:
    groups = canonical_dialogue_case_groups()
    assert set(groups) == {
        'identity',
        'style',
        'memory_continuity',
        'stress_adversarial',
        'decision_making',
        'repeated_topic_consistency',
    }

    cases = canonical_dialogue_cases()
    case_ids = [case.case_id for case in cases]
    assert len(case_ids) == len(set(case_ids))
    assert all(case.trait_probe for case in cases)
    assert all(case.generic_llm_failure_signals for case in cases)
    assert all(case.persona_success_signals for case in cases)

    benchmark = canonical_dialogue_benchmark()
    assert benchmark['benchmark_id'] == 'canonical_persona_fidelity_v1'
    assert benchmark['case_count'] == len(cases)
