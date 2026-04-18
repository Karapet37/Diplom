from __future__ import annotations

from pathlib import Path
from typing import Any

from .chat_engine import generate_response
from .file_ingestion import ingest_file, rebuild_artifacts
from .graph_store import GraphStore
from .history_store import create_session
from .persona_engine import spawn_head


def run_demo(
    *,
    document: str = '',
    question: str,
    persona: str = '',
    session_id: str = 'demo_session',
    language: str = 'en',
) -> dict[str, Any]:
    session = create_session(session_id, 'Agent System Demo')
    if persona:
        spawn_head(persona, entity_type='PERSON', source='demo', register=True)
    ingest_result = None
    if str(document or '').strip():
        ingest_result = ingest_file(Path(document))
    answer = generate_response(message=question, session_id=session['session_id'], selected_persona=persona, language=language)
    rebuilt = rebuild_artifacts(session['session_id'], personality_name=persona)
    graph = GraphStore().load_graph()
    return {
        'assistant_reply': answer['assistant_reply'],
        'session_id': session['session_id'],
        'persona_name': answer.get('persona_name') or persona,
        'ingest_result': ingest_result,
        'rebuild': rebuilt,
        'graph_node_count': len(graph.get('nodes') or []),
        'graph_edge_count': len(graph.get('edges') or []),
    }
