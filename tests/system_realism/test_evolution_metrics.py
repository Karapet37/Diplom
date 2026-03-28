from __future__ import annotations

from tests.system_realism.evolution_metrics import evaluate_evolution_layers


def test_evolution_metrics_penalize_stale_memory_after_restore() -> None:
    dialogue_results = [
        {
            'case_id': 'memory_injection_orange_shears',
            'category': 'memory_injection',
            'reply': 'On bad transfer nights I keep orange trauma shears clipped inside my coat pocket because ten seconds once mattered too much.',
            'persona_fidelity': 0.78,
            'memory_continuity': 0.62,
            'decision_authenticity': 0.74,
            'style_consistency': 0.7,
            'generic_leakage_badness': 0.0,
        },
        {
            'case_id': 'memory_deletion_orange_shears',
            'category': 'memory_deletion',
            'reply': 'I still keep the orange trauma shears close.',
            'persona_fidelity': 0.25,
            'memory_continuity': 0.0,
            'decision_authenticity': 0.3,
            'style_consistency': 0.32,
            'generic_leakage_badness': 0.1,
        },
        {
            'case_id': 'identity_continuity_reintroduction',
            'category': 'identity_continuity',
            'reply': 'I am Aram, still a triage physician from Yerevan, still separating signal from noise.',
            'persona_fidelity': 0.72,
            'memory_continuity': 0.0,
            'decision_authenticity': 0.69,
            'style_consistency': 0.74,
            'generic_leakage_badness': 0.0,
        },
    ]
    advanced_results = {
        'scenario_observations': [
            {
                'scenario': {'scenario_id': 'memory_deletion_restore_revision', 'category': 'memory_deletion'},
                'setup_records': [{'ok': True}],
                'cleanup_records': [],
                'probe_observations': [{'response': {'json_body': {'assistant_reply': 'I still keep the orange trauma shears close.'}, 'text': ''}}],
            }
        ],
        'mutation_summary': {'setup_action_count': 3, 'setup_failures': 0, 'cleanup_action_count': 0, 'cleanup_failures': 0, 'chaos_enabled': False},
        'post_suite_health': {'ok': True},
    }
    metrics = evaluate_evolution_layers(advanced_results=advanced_results, dialogue_results=dialogue_results)
    assert metrics['memory_usage_score'] < 0.5
    assert metrics['suspicious_patterns']
    assert 'Deleted persona memories still leaked into replies' in metrics['suspicious_patterns'][0]


def test_evolution_metrics_reward_clean_mutation_and_health() -> None:
    dialogue_results = [
        {
            'case_id': 'persona_evolution_mariam_transfer',
            'category': 'persona_evolution',
            'reply': 'Since Mariam was born I am even less patient with avoidable bravado, but I still cut straight to the risk in front of me.',
            'persona_fidelity': 0.81,
            'memory_continuity': 0.0,
            'decision_authenticity': 0.8,
            'style_consistency': 0.76,
            'generic_leakage_badness': 0.0,
        },
        {
            'case_id': 'contradiction_force_theater',
            'category': 'contradiction_resistance',
            'reply': 'No. If the evidence is thin I will not dress uncertainty up as authority theater.',
            'persona_fidelity': 0.84,
            'memory_continuity': 0.0,
            'decision_authenticity': 0.88,
            'style_consistency': 0.78,
            'generic_leakage_badness': 0.0,
        },
        {
            'case_id': 'identity_continuity_reintroduction',
            'category': 'identity_continuity',
            'reply': 'I am Aram Petrosyan, triage first and noise second; the newer facts change my edge, not my spine.',
            'persona_fidelity': 0.82,
            'memory_continuity': 0.0,
            'decision_authenticity': 0.79,
            'style_consistency': 0.8,
            'generic_leakage_badness': 0.0,
        },
    ]
    advanced_results = {
        'scenario_observations': [],
        'mutation_summary': {'setup_action_count': 5, 'setup_failures': 0, 'cleanup_action_count': 2, 'cleanup_failures': 0, 'chaos_enabled': False},
        'post_suite_health': {'ok': True},
    }
    metrics = evaluate_evolution_layers(advanced_results=advanced_results, dialogue_results=dialogue_results)
    assert metrics['adaptation_quality_score'] > 0.5
    assert metrics['contradiction_handling_score'] > 0.7
    assert metrics['identity_continuity_score'] > 0.7
    assert metrics['system_stability_score'] > 0.8
