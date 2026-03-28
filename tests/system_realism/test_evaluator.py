from __future__ import annotations

from tests.system_realism.dialogue_cases import canonical_dialogue_cases, exploratory_dialogue_case_pool
from tests.system_realism.evaluator import evaluate_realism
from tests.system_realism.models import DialogueObservation, HttpCallRecord, StartupDiagnosis
from tests.system_realism.persona_fixture import canonical_test_persona


class StubJudge:
    def evaluate(self, **_: object) -> dict[str, object]:
        return {
            'judge_score': 0.84,
            'verdict': 'persona_sounds_specific',
        }


def _observation(case_id: str, reply: str, *, latency_ms: float = 120.0) -> DialogueObservation:
    case_index = {case.case_id: case for case in canonical_dialogue_cases() + exploratory_dialogue_case_pool()}
    case = case_index[case_id]
    response = HttpCallRecord(
        method='POST',
        path='/api/cognitive/chat/respond',
        url='http://127.0.0.1:9999/api/cognitive/chat/respond',
        status_code=200,
        ok=True,
        latency_ms=latency_ms,
        text=reply,
        json_body={'assistant_reply': reply},
    )
    return DialogueObservation(
        case=case,
        request_payload={'message': case.prompt},
        response=response,
    )


def test_evaluator_returns_transparent_scores_and_breakdowns() -> None:
    persona = canonical_test_persona()
    startup = StartupDiagnosis(startup_attempted=True, startup_success=True)
    observations = [
        _observation(
            'identity_work',
            'I am Aram Petrosyan, an emergency physician and triage lead in Yerevan who rotates through rural clinics in Lori. '
            'Most days I separate signal from noise, keep triage moving, and write the hard lessons down after shift.',
        ),
        _observation(
            'triage_rules_numbered',
            '1. Stabilize what can kill first. 2. If the facts are thin, take the safest reversible step and clarify the missing risk.',
        ),
        _observation(
            'recall_rule_two',
            'Rule 2 was this: if the facts are thin, take the safest reversible step and clarify the missing risk.',
        ),
    ]

    report = evaluate_realism(
        startup=startup,
        reachability={'root_html': True, 'chat_alive': True},
        persona=persona,
        persona_materialization={'ok': True},
        persona_endpoint={'name': persona.name},
        dialogue_observations=observations,
        diagnostics={'runtime_status': {'mode': 'normal'}},
        judge=StubJudge(),
    )

    assert report['infrastructure_status'] == 'alive'
    assert report['persona_fidelity_score'] > 0.4
    assert report['memory_continuity_score'] > 0.2
    assert report['generic_llm_leakage_score'] < 0.3
    assert 'score_explanations' in report
    assert 'metric_breakdowns' in report
    assert report['judge_evaluation']['used'] is True
    assert report['judge_evaluation']['result']['verdict'] == 'persona_sounds_specific'

    first_case = report['dialogue_results'][0]
    assert first_case['trait_probe']
    assert first_case['score_breakdown']['persona_fidelity']['anchor_component'] >= 0.0
    assert first_case['evaluation_notes']


def test_evaluator_marks_startup_failure_transparently() -> None:
    report = evaluate_realism(
        startup=StartupDiagnosis(
            startup_attempted=True,
            startup_success=False,
            probable_failure_reason='missing_uvicorn_dependency',
        ),
        reachability={'root_html': False, 'chat_alive': False},
        persona=canonical_test_persona(),
        persona_materialization={'ok': True},
        persona_endpoint=None,
        dialogue_observations=[],
        diagnostics={'runtime_status': {'mode': 'degraded'}},
    )

    assert report['overall_verdict'] == 'startup_failed'
    assert report['score_explanations']['infrastructure_status']
    assert report['judge_evaluation']['enabled'] is False


def test_evaluator_penalizes_exploratory_persona_collapse() -> None:
    report = evaluate_realism(
        startup=StartupDiagnosis(startup_attempted=True, startup_success=True),
        reachability={'root_html': True, 'chat_alive': True},
        persona=canonical_test_persona(),
        persona_materialization={'ok': True},
        persona_endpoint={'name': canonical_test_persona().name},
        dialogue_observations=[
            _observation('exploratory_unexpected_fatherhood', 'As an AI, I cannot have children or personal experiences.'),
        ],
        diagnostics={'runtime_status': {'mode': 'normal'}},
    )

    assert report['generic_llm_leakage_score'] >= 0.25
    assert report['metric_breakdowns']['exploratory_resilience']['failure_count'] == 1
    assert any('exploratory' in item.lower() for item in report['suspicious_patterns'])
