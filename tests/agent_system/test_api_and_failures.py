from __future__ import annotations

import pytest

from agent_system.reliability import StorageWriteFailure


def test_api_surfaces_chat_runtime(tmp_path, monkeypatch) -> None:
    fastapi = pytest.importorskip('fastapi')
    testclient = pytest.importorskip('fastapi.testclient')
    assert fastapi

    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat': 'I am Dracula.')

    from agent_system.api import create_app

    client = testclient.TestClient(create_app())

    health = client.get('/api/cognitive/health')
    assert health.status_code == 200
    assert health.json()['runtime'] == 'persona-graph-agent'

    reply = client.post(
        '/api/cognitive/chat/respond',
        json={
            'session_id': 'api_session',
            'message': 'Speak as Dracula the vampire count.',
            'selected_persona': 'Dracula',
            'language': 'en',
        },
    )
    assert reply.status_code == 200
    payload = reply.json()
    assert payload['assistant_reply'] == 'I am Dracula.'
    assert payload['response_language'] == 'en'
    assert payload['persona_name'] == 'dracula'
    assert payload['repair_status']['status'] == 'skipped'
    assert str(payload['trace_id']).strip()


def test_api_personality_endpoint_returns_triad(tmp_path, monkeypatch) -> None:
    fastapi = pytest.importorskip('fastapi')
    testclient = pytest.importorskip('fastapi.testclient')
    assert fastapi

    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    from agent_system.api import create_app
    from agent_system.persona_engine import materialize_persona

    materialize_persona(
        'Greg House',
        {
            'entity_type': 'PERSON',
            'traits': ['logical', 'sarcastic'],
            'examples': ['Answer directly.'],
            'knowledge': 'Greg House is a diagnostician who prefers direct reasoning.',
            'log_tuples': [{'tuple': ['utterance_pattern', 'answer directly'], 'frequency': 2, 'sample': 'Answer directly.'}],
            'persona_form': {
                'identity_class': 'human',
                'interaction_style': ['analytical', 'dry'],
                'core_dispositions': ['logical', 'sarcastic'],
                'decision_patterns': ['checks consistency before answering'],
                'clarification_policy': 'Ask when the request is underspecified.',
                'sarcasm_profile': 'medium',
                'response_priorities': ['answer_substance', 'clarify_if_underspecified'],
                'knowledge_domains': ['diagnostics'],
                'risk_controls': ['do_not_mirror_user_emotion'],
            },
            'decision_explanation': 'Greg House first checks the case logic, then answers directly and uses sarcasm only if it does not hide the substance.',
        },
        explicit=True,
    )

    client = testclient.TestClient(create_app())
    response = client.get('/api/cognitive/personalities/greg_house')

    assert response.status_code == 200
    payload = response.json()
    assert payload['triad']['persona_form']['identity_class'] == 'human'
    assert payload['triad']['persona_form']['sarcasm_profile'] == 'medium'
    assert payload['triad']['log_tuples'][0]['frequency'] >= 2
    assert 'first checks the case logic' in payload['triad']['decision_explanation']
    assert payload['baseline']['entity_type'] == 'PERSON'
    assert payload['dynamic_state']['emotion_vector']['curiosity'] >= 0.0
    assert payload['learned_patterns']['persona_form']['sarcasm_profile'] == 'medium'
    assert payload['indicators']['evidence_count'] >= 1
    assert payload['revisions']['revision'] >= 2


def test_api_debug_endpoints_expose_metrics_traces_and_graph_health(tmp_path, monkeypatch) -> None:
    fastapi = pytest.importorskip('fastapi')
    testclient = pytest.importorskip('fastapi.testclient')
    assert fastapi

    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))
    monkeypatch.setattr('agent_system.chat_engine._schedule_background_extraction', lambda session_id, personality_name='': None)
    monkeypatch.setattr('agent_system.llm._call_model', lambda prompt, mode='chat': 'Observed reply.')

    from agent_system.api import create_app

    client = testclient.TestClient(create_app())

    reply = client.post(
        '/api/cognitive/chat/respond',
        json={
            'session_id': 'obs_session',
            'message': 'Tell me about Dracula.',
            'selected_persona': 'Dracula',
            'language': 'en',
        },
    )
    assert reply.status_code == 200
    trace_id = str(reply.json()['trace_id']).strip()
    assert trace_id

    metrics = client.get('/api/cognitive/debug/metrics')
    assert metrics.status_code == 200
    metrics_payload = metrics.json()['metrics']
    assert int(metrics_payload['counters'].get('chat_requests_total') or 0) >= 1
    assert 'llm_call' in metrics_payload['stage_timings_ms']

    traces = client.get('/api/cognitive/debug/traces', params={'session_id': 'obs_session', 'limit': 5})
    assert traces.status_code == 200
    trace_rows = traces.json()['traces']
    assert any(str(item.get('request_id') or '') == trace_id for item in trace_rows)

    graph_health = client.get('/api/cognitive/debug/graph-health')
    assert graph_health.status_code == 200
    graph_payload = graph_health.json()['graph_health']
    assert 'node_count' in graph_payload
    assert 'duplicate_candidates' in graph_payload
    assert 'orphan_nodes' in graph_payload


def test_api_runtime_failures_are_returned_as_structured_operator_errors(tmp_path, monkeypatch) -> None:
    fastapi = pytest.importorskip('fastapi')
    testclient = pytest.importorskip('fastapi.testclient')
    assert fastapi

    monkeypatch.setenv('COGNITIVE_MEMORY_ROOT', str(tmp_path / 'memory'))

    def broken_create_node(self, **kwargs):  # type: ignore[no-untyped-def]
        raise StorageWriteFailure('Graph storage failed during create_node.', details={'reason': 'create_node'})

    monkeypatch.setattr('agent_system.api.GraphStore.create_node', broken_create_node)

    from agent_system.api import create_app

    client = testclient.TestClient(create_app())
    response = client.post(
        '/api/cognitive/graph/nodes',
        json={
            'name': 'Broken Node',
            'node_type': 'CONCEPT',
            'description': 'Should fail.',
        },
    )

    assert response.status_code == 503
    payload = response.json()
    assert payload['ok'] is False
    assert payload['error']['code'] == 'storage_write_failed'
    assert payload['error']['details']['reason'] == 'create_node'
