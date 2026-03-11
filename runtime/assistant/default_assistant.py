from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _needs_clarification(query: str, lookup: dict[str, Any]) -> tuple[bool, str]:
    cleaned = _normalize_text(query)
    token_count = len([token for token in cleaned.split() if token])
    if token_count <= 2:
        return True, "What exactly do you want to understand or decide here?"
    if not list(lookup.get("ranked_nodes") or []) and token_count <= 6:
        return True, "Give one concrete example or one sentence of context so I can ground the answer."
    if cleaned.endswith("?") and any(word in cleaned.lower() for word in ("it", "this", "that", "это", "это?", "он", "она")) and token_count <= 8:
        return True, "What does 'it' refer to in this situation?"
    return False, ""


@dataclass(slots=True)
class AssistantDecision:
    clarification_needed: bool
    clarification_question: str
    plan_steps: list[str]
    requires_graph_update: bool
    reason: str
    lookup: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "clarification_needed": self.clarification_needed,
            "clarification_question": self.clarification_question,
            "plan_steps": list(self.plan_steps),
            "requires_graph_update": self.requires_graph_update,
            "reason": self.reason,
            "lookup": dict(self.lookup),
        }


class DefaultAssistant:
    """Stateless assistant that decides clarification, plan, and graph-update need."""

    def assess(
        self,
        *,
        query: str,
        context: str,
        lookup: dict[str, Any],
        apply_to_graph: bool,
    ) -> AssistantDecision:
        effective_query = "\n\n".join(part for part in [_normalize_text(query), _normalize_text(context)] if part).strip()
        clarification_needed, clarification_question = _needs_clarification(effective_query, lookup)
        ranked = list(lookup.get("ranked_nodes") or [])
        missing_tokens = list(lookup.get("missing_tokens") or [])
        plan_steps = [
            "extract fast signals from the query",
            "search verified graph with a small context window",
            "rank relevant nodes by overlap and importance",
        ]
        if clarification_needed:
            plan_steps.append("ask for the minimum missing clarification")
            return AssistantDecision(
                clarification_needed=True,
                clarification_question=clarification_question,
                plan_steps=plan_steps,
                requires_graph_update=False,
                reason="clarification_needed",
                lookup=lookup,
            )
        plan_steps.append("generate a fast grounded reply")
        if apply_to_graph:
            if ranked and not missing_tokens:
                plan_steps.append("skip graph build because verified context is already sufficient")
                requires_graph_update = False
                reason = "verified_graph_sufficient"
            else:
                plan_steps.append("queue background graph build for missing knowledge only")
                requires_graph_update = True
                reason = "missing_verified_context"
        else:
            requires_graph_update = False
            reason = "graph_write_disabled"
        return AssistantDecision(
            clarification_needed=False,
            clarification_question="",
            plan_steps=plan_steps,
            requires_graph_update=requires_graph_update,
            reason=reason,
            lookup=lookup,
        )

