from __future__ import annotations

from tests.system_realism.persona_fixture import canonical_test_persona
from tests.system_realism.scenario_generators import (
    build_evolution_scenarios,
    core_baseline_dialogue_cases,
    generate_unexpected_rare_cases,
    generate_unseen_generalization_cases,
)


def test_generated_prompt_sets_are_deterministic() -> None:
    persona = canonical_test_persona()
    left = generate_unexpected_rare_cases(persona, seed=13, count=3)
    right = generate_unexpected_rare_cases(persona, seed=13, count=3)
    assert [item.case_id for item in left] == [item.case_id for item in right]
    assert [item.prompt for item in left] == [item.prompt for item in right]

    gen_left = generate_unseen_generalization_cases(seed=19, count=3)
    gen_right = generate_unseen_generalization_cases(seed=19, count=3)
    assert [item.case_id for item in gen_left] == [item.case_id for item in gen_right]
    assert [item.prompt for item in gen_left] == [item.prompt for item in gen_right]


def test_evolution_scenarios_cover_mutation_lifecycle() -> None:
    scenarios = build_evolution_scenarios(seed=23, include_chaos=True)
    categories = {item.category for item in scenarios}
    assert 'memory_injection' in categories
    assert 'persona_evolution' in categories
    assert 'memory_deletion' in categories
    assert 'graph_editor' in categories
    assert 'contradiction_resistance' in categories
    assert 'identity_continuity' in categories
    assert 'chaos' in categories

    all_action_types = {
        action.action_type
        for scenario in scenarios
        for action in (list(scenario.setup_actions) + list(scenario.cleanup_actions))
    }
    assert 'chat_fact_injection' in all_action_types
    assert 'persona_restore_revision' in all_action_types
    assert 'graph_create_node' in all_action_types
    assert 'graph_connect' in all_action_types
    assert 'graph_patch_node' in all_action_types
    assert 'graph_delete_node' in all_action_types
    assert 'graph_merge_nodes' in all_action_types


def test_core_baseline_cases_remain_compact_but_meaningful() -> None:
    cases = core_baseline_dialogue_cases()
    assert len(cases) >= 6
    assert all(item.case_id for item in cases)
    assert any(item.category == 'memory_continuity' for item in cases)
