from __future__ import annotations

from typing import Any

from .duplicate_resolver import normalize_name
from .models import ClassificationDecision, MessageAnalysis, PersonaSelectionExplanation
from .models import HEAD_ENTITY_TYPES
from .persona_engine import load_persona, spawn_head


def prepare_heads(
    *,
    analysis: MessageAnalysis,
    classifications: list[ClassificationDecision],
    graph_store: Any,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    selected_token = normalize_name(analysis.selected_head)
    for decision in classifications:
        is_selected_entity = bool(selected_token and normalize_name(decision.entity_name) == selected_token)
        profession_head_allowed = decision.entity_type == 'PROFESSION' and (
            is_selected_entity
            or len(str(decision.entity_name or '').split()) >= 2
            or load_persona(decision.entity_name) is not None
        )
        allow_head = decision.entity_type in {'PERSON', 'FICTIONAL_CHARACTER'} or profession_head_allowed or is_selected_entity
        node = (
            graph_store.upsert_entity(
                name=decision.entity_name,
                entity_type=decision.entity_type,
                aliases=[],
                description='',
                confidence=decision.confidence,
                source='chat',
            )
            if allow_head
            else graph_store.get_node(decision.entity_name)
        )
        head = (
            spawn_head(
                decision.entity_name,
                entity_type=decision.entity_type,
                aliases=list(node.get('aliases') or []),
                source='chat',
            )
            if allow_head
            else None
        )
        prepared.append({'decision': decision, 'node': node, 'head': head})
    return prepared


def select_primary_head(
    *,
    analysis: MessageAnalysis,
    prepared_heads: list[dict[str, Any]],
) -> dict[str, Any] | None:
    selected = str(analysis.selected_head or analysis.primary_entity or '').strip().lower()
    if selected:
        for item in prepared_heads:
            head = item.get('head')
            if head is None:
                continue
            if str(head.meta.get('slug') or '').lower() == selected or str(head.name or '').lower() == selected:
                decision = item.get('decision')
                enriched = dict(item)
                enriched['selection_explanation'] = PersonaSelectionExplanation(
                    persona_name=head.name,
                    source='explicit_selection',
                    reason='Selected persona matched an explicit user-selected head.',
                    evidence=[
                        f'selected_head={analysis.selected_head}',
                        f'entity_name={head.name}',
                        f'entity_type={head.entity_type}',
                        f'confidence={decision.confidence:.3f}' if decision is not None else 'confidence=existing',
                    ],
                )
                return enriched
    for item in prepared_heads:
        if item.get('head') is not None:
            head = item['head']
            decision = item.get('decision')
            enriched = dict(item)
            enriched['selection_explanation'] = PersonaSelectionExplanation(
                persona_name=head.name,
                source='prepared_head_fallback',
                reason='The first valid spawned head was selected as the current primary persona.',
                evidence=[
                    f'entity_name={head.name}',
                    f'entity_type={head.entity_type}',
                    f'confidence={decision.confidence:.3f}' if decision is not None else 'confidence=existing',
                ],
            )
            return enriched
    if analysis.primary_entity:
        bundle = load_persona(analysis.primary_entity)
        if bundle:
            return {
                'decision': None,
                'node': None,
                'head': bundle,
                'selection_explanation': PersonaSelectionExplanation(
                    persona_name=bundle.name,
                    source='existing_persona_lookup',
                    reason='Existing persona head matched the primary entity inferred from the message.',
                    evidence=[
                        f'primary_entity={analysis.primary_entity}',
                        f'entity_type={bundle.entity_type}',
                    ],
                ),
            }
    return None
