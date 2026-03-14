from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .context_builder import build_chat_context, current_entity_hint, infer_personality_name, personality_exists
from .graph_store import GraphStore
from .history_store import append_turn, create_session, infer_current_entity, parse_session, recent_dialogue
from .knowledge_extractor import process_session_artifacts, request_missing_personality
from .llm import generate_chat_reply

_MISSING_PERSONALITY_RU = 'Контекст личности не найден. Запрошено создание анкеты по описанию.'
_BACKGROUND_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mvp-extract')


def _schedule_background_work(session_id: str, personality_name: str = '') -> None:
    def _runner() -> None:
        try:
            process_session_artifacts(session_id, personality_name=personality_name)
        except Exception:
            return

    _BACKGROUND_EXECUTOR.submit(_runner)


def generate_dialogue_response(
    *,
    message: str,
    session_id: str = '',
    context: str = '',
    language: str = 'ru',
    llm_role: str = 'general',
    personality_name: str = '',
    user_id: str = '',
) -> dict[str, Any]:
    del llm_role, user_id
    clean_message = str(message or '').strip()
    session = create_session(session_id or '')
    session_id = str(session.get('session_id') or session_id or '')
    graph_store = GraphStore()

    session_context = recent_dialogue(session_id)
    current_entity = current_entity_hint(clean_message, session_context) or infer_current_entity(session_id)
    resolved_personality = infer_personality_name(clean_message, selected_name=personality_name, current_entity=current_entity)
    explicit_personality_requested = bool(str(personality_name or '').strip())

    if resolved_personality and not personality_exists(resolved_personality):
        graph_has_entity = graph_store.entity_exists(resolved_personality)
        if explicit_personality_requested or not graph_has_entity:
            request_missing_personality(
                resolved_personality,
                reason='User selected or mentioned a personality missing from graph and personality files.',
                session_id=session_id,
                excerpt=clean_message,
            )
            assistant_reply = _MISSING_PERSONALITY_RU if (language or 'ru').startswith('ru') else 'Personality context was not found. A profile request has been created from the description.'
            append_turn(session_id, clean_message, assistant_reply)
            _schedule_background_work(session_id, personality_name=resolved_personality)
            session = parse_session(session_id) or session
            return {
                'assistant_reply': assistant_reply,
                'session_id': session_id,
                'session': session,
                'personality_name': resolved_personality,
                'graph_context': '',
                'current_entity': current_entity or resolved_personality,
                'proposal_requested': True,
            }
        resolved_personality = ''

    built_context = build_chat_context(
        message=clean_message,
        recent_dialogue=session_context,
        selected_personality=resolved_personality,
        current_entity=current_entity,
        explicit_context=context,
        store=graph_store,
    )
    assistant_reply = generate_chat_reply(
        message=clean_message,
        session_context=built_context['session_context'],
        graph_context=built_context['graph_context'],
        personality_prompt=built_context['personality_prompt'],
        language=language or 'ru',
    )
    append_turn(session_id, clean_message, assistant_reply)
    _schedule_background_work(session_id, personality_name=built_context.get('personality_name') or '')
    session = parse_session(session_id) or session
    return {
        'assistant_reply': assistant_reply,
        'session_id': session_id,
        'session': session,
        'personality_name': built_context.get('personality_name') or '',
        'graph_context': built_context.get('graph_context') or '',
        'current_entity': built_context.get('current_entity') or '',
        'proposal_requested': False,
    }
