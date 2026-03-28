from __future__ import annotations

from typing import Any

from .models import EvolutionScenarioObservation, RealismRunConfig
from .mutation_utils import LiveSystemOperator
from .persona_fixture import CanonicalTestPersona
from .runtime_launcher import RuntimeLauncher
from .scenario_generators import build_evolution_scenarios, generate_unexpected_rare_cases, generate_unseen_generalization_cases


def run_advanced_realism_suite(
    *,
    config: RealismRunConfig,
    launcher: RuntimeLauncher,
    persona: CanonicalTestPersona,
    memory_root,
    report_tag: str,
) -> dict[str, Any]:
    operator = LiveSystemOperator(
        launcher=launcher,
        memory_root=memory_root,
        persona_name=persona.name,
        persona_slug=persona.slug,
        base_session_id=f'system-realism-{report_tag}-advanced',
        language='en',
        request_timeout_s=config.request_timeout_s,
    )
    generated_case_seed = int(config.exploratory_seed) + 101
    unexpected_cases = generate_unexpected_rare_cases(
        persona,
        seed=generated_case_seed,
        count=max(0, int(config.unexpected_case_count or 0)),
    )
    generalization_cases = generate_unseen_generalization_cases(
        seed=generated_case_seed + 17,
        count=max(0, int(config.generalization_case_count or 0)),
    )
    generated_observations = [
        operator.probe_case(case, session_id=f'{operator.base_session_id}-generated')
        for case in (unexpected_cases + generalization_cases)
    ]

    scenario_observations: list[EvolutionScenarioObservation] = []
    advanced_dialogues = list(generated_observations)
    scenarios = build_evolution_scenarios(
        seed=generated_case_seed + 31,
        include_chaos=bool(config.include_chaos),
        subset=config.mutation_subset,
    )
    for scenario in scenarios:
        raw = operator.run_scenario(scenario, session_id=f'{operator.base_session_id}-{scenario.scenario_id}')
        observation = EvolutionScenarioObservation(
            scenario=raw['scenario'],
            setup_records=list(raw['setup_records']),
            probe_observations=list(raw['probe_observations']),
            cleanup_records=list(raw['cleanup_records']),
            state_snapshot=dict(raw['state_snapshot']),
        )
        scenario_observations.append(observation)
        advanced_dialogues.extend(observation.probe_observations)

    setup_records = [record for item in scenario_observations for record in item.setup_records]
    cleanup_records = [record for item in scenario_observations for record in item.cleanup_records]
    probe_health = launcher.probe_runtime_health(timeout_s=8.0)
    return {
        'generated_cases': {
            'unexpected_rare': [case.to_dict() for case in unexpected_cases],
            'unseen_generalization': [case.to_dict() for case in generalization_cases],
        },
        'generated_observations': [item.to_dict() for item in generated_observations],
        'scenario_observations': [item.to_dict() for item in scenario_observations],
        'dialogue_observations': advanced_dialogues,
        'mutation_summary': {
            'setup_action_count': len(setup_records),
            'setup_failures': sum(1 for item in setup_records if not item.ok),
            'cleanup_action_count': len(cleanup_records),
            'cleanup_failures': sum(1 for item in cleanup_records if not item.ok),
            'scenario_count': len(scenario_observations),
            'chaos_enabled': bool(config.include_chaos),
        },
        'post_suite_health': probe_health.to_dict(),
        'state_snapshot': dict(operator.state),
    }
