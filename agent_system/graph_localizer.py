from __future__ import annotations

from typing import Any

from .graph_store import GraphStore
from .language_tools import detect_language_code, normalize_language_code
from .llm import translate_text
from .models import GraphModelToolPolicy


def _policy(policy: GraphModelToolPolicy | None = None) -> GraphModelToolPolicy:
    return policy or GraphModelToolPolicy()


def _node_context(node: dict[str, Any]) -> dict[str, Any]:
    return dict(node.get('context') or {})


def _source_explanation(node: dict[str, Any]) -> str:
    context = _node_context(node)
    return (
        str(context.get('plain_explanation') or '').strip()
        or str(node.get('description') or '').strip()
        or ' | '.join(str(item).strip() for item in list(node.get('facts') or []) if str(item).strip())
    ).strip()


def ensure_node_localization(
    node_id: str,
    *,
    language: str = 'en',
    store: GraphStore | None = None,
    policy: GraphModelToolPolicy | None = None,
) -> dict[str, Any]:
    graph_store = store or GraphStore()
    node = graph_store.get_node_by_id(node_id)
    if node is None:
        return {'ok': False, 'reason': 'node_not_found', 'node_id': node_id}

    tool_policy = _policy(policy)
    target_language = normalize_language_code(language, fallback=tool_policy.canonical_language)
    context = _node_context(node)
    localized_explanations = dict(context.get('localized_explanations') or {})
    changed = False

    canonical_english = str(context.get('canonical_english_explanation') or '').strip()
    if not canonical_english:
        source_text = _source_explanation(node)
        source_language = detect_language_code(source_text, fallback=tool_policy.canonical_language)
        if source_text:
            if source_language == tool_policy.canonical_language or not tool_policy.allow_translation:
                canonical_english = source_text
            else:
                canonical_english = translate_text(
                    source_text,
                    target_language=tool_policy.canonical_language,
                    source_language=source_language,
                    role=tool_policy.translation_role,
                ).strip() or source_text
            changed = True

    localized_explanation = ''
    if target_language != tool_policy.canonical_language:
        localized_explanation = str(localized_explanations.get(target_language) or '').strip()
        if not localized_explanation and canonical_english and tool_policy.allow_translation:
            localized_explanation = translate_text(
                canonical_english,
                target_language=target_language,
                source_language=tool_policy.canonical_language,
                role=tool_policy.translation_role,
            ).strip()
            if localized_explanation and localized_explanation != canonical_english:
                localized_explanations[target_language] = localized_explanation
                changed = True
            elif localized_explanation == canonical_english:
                localized_explanation = ''

    if changed:
        graph_store.patch_node(
            str(node.get('id') or ''),
            context_patch={
                **context,
                'canonical_english_explanation': canonical_english,
                'localized_explanations': localized_explanations,
                'last_localized_language': target_language,
            },
        )
        node = graph_store.get_node_by_id(node_id) or node
        context = _node_context(node)
        localized_explanations = dict(context.get('localized_explanations') or localized_explanations)

    return {
        'ok': True,
        'node_id': node_id,
        'canonical_english_explanation': str(context.get('canonical_english_explanation') or canonical_english).strip(),
        'localized_explanation': str(localized_explanations.get(target_language) or localized_explanation).strip(),
        'localized_language': target_language,
    }


def localized_node_view(
    node_id: str,
    *,
    language: str = 'en',
    store: GraphStore | None = None,
    policy: GraphModelToolPolicy | None = None,
) -> dict[str, Any] | None:
    graph_store = store or GraphStore()
    view = graph_store.answerable_node_view(node_id)
    if view is None:
        return None
    localization = ensure_node_localization(node_id, language=language, store=graph_store, policy=policy)
    if not localization.get('ok'):
        return view
    enriched = dict(view)
    what = dict(enriched.get('what_is_it_like') or {})
    context = dict(what.get('context') or {})
    context.update(
        {
            'canonical_english_explanation': localization.get('canonical_english_explanation') or '',
            'localized_explanation': localization.get('localized_explanation') or '',
            'localized_language': localization.get('localized_language') or normalize_language_code(language),
        }
    )
    what['context'] = context
    enriched['what_is_it_like'] = what
    return enriched
