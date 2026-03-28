from __future__ import annotations

from tests.system_realism.dialogue_cases import (
    canonical_dialogue_benchmark,
    canonical_dialogue_cases,
    canonical_dialogue_case_groups,
    exploratory_dialogue_case_pool,
    sample_exploratory_dialogue_cases,
)


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


def test_exploratory_dialogue_pool_is_rich_and_seeded() -> None:
    pool = exploratory_dialogue_case_pool()
    assert len(pool) >= 10
    assert all(case.category == 'exploratory' for case in pool)

    sample_a = sample_exploratory_dialogue_cases(seed=17, count=5)
    sample_b = sample_exploratory_dialogue_cases(seed=17, count=5)
    sample_c = sample_exploratory_dialogue_cases(seed=23, count=5)

    assert [case.case_id for case in sample_a] == [case.case_id for case in sample_b]
    assert [case.case_id for case in sample_a] != [case.case_id for case in sample_c]
