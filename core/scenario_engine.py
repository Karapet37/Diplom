from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DialogueScenario:
    scenario_id: str
    scenario_type: str
    title: str
    prompt_frame: str
    constraints: tuple[str, ...]
    expected_tension: float
    recommended_agents: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_type": self.scenario_type,
            "title": self.title,
            "prompt_frame": self.prompt_frame,
            "constraints": list(self.constraints),
            "expected_tension": self.expected_tension,
            "recommended_agents": list(self.recommended_agents),
        }


class ScenarioEngine:
    def pick(self, query: str, *, profession: str = "") -> DialogueScenario:
        text = str(query or "").lower()
        profession = str(profession or "").lower()
        if any(token in text for token in ("street", "stranger", "sidewalk", "fight", "двор", "улиц")):
            return DialogueScenario(
                scenario_id="street_interaction",
                scenario_type="street_interaction",
                title="Street interaction",
                prompt_frame="Read fast social cues, threat, disrespect, and status shifts before answering.",
                constraints=("short reaction time", "high uncertainty", "social risk"),
                expected_tension=0.72,
                recommended_agents=("strategy",),
            )
        if any(token in text for token in ("contract", "court", "law", "legal", "суд", "закон", "договор")) or profession == "law":
            return DialogueScenario(
                scenario_id="professional_consultation",
                scenario_type="professional_consultation",
                title="Professional consultation",
                prompt_frame="Answer as a professional consultation grounded in obligations, evidence, and practical next steps.",
                constraints=("evidence discipline", "avoid overclaiming", "high consequence"),
                expected_tension=0.45,
                recommended_agents=("law", "strategy"),
            )
        if any(token in text for token in ("conflict", "argument", "pressure", "manipulat", "ссора", "давлен", "конфликт")):
            return DialogueScenario(
                scenario_id="conflict_dialogue",
                scenario_type="conflict_dialogue",
                title="Conflict dialogue",
                prompt_frame="Read power, pressure, hurt, and escalation risk before producing a response.",
                constraints=("high emotional load", "hidden intent likely", "repair vs escalation tradeoff"),
                expected_tension=0.81,
                recommended_agents=("strategy", "business"),
            )
        return DialogueScenario(
            scenario_id="casual_conversation",
            scenario_type="casual_conversation",
            title="Casual conversation",
            prompt_frame="Keep the answer human and grounded, but avoid overloading it with formal structure.",
            constraints=("low tension", "social pacing matters"),
            expected_tension=0.22,
            recommended_agents=("strategy",),
        )
