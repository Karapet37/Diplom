from __future__ import annotations

from core import DialogueEngine, PersonalityCore, ScenarioEngine, SpeechDNA, SpeechStyleCore, StyleEngine, default_agent_roles
from core.graph_core import GraphEdge, GraphNode, ImportanceVector, NodeCore, RAMContextGraph, RankedContextNode


def test_style_engine_learns_only_on_trigger(tmp_path) -> None:
    engine = StyleEngine(base_dir=tmp_path)
    messages = [
        {"role": "user", "message": "Look, stop hedging and say it directly."},
        {"role": "user", "message": "Honestly, keep it short and practical."},
        {"role": "user", "message": "If the pattern repeats, call it out plainly."},
    ]
    skipped = engine.learn(user_id="u1", messages=messages, learn_style_button=False)
    assert skipped.learned is False
    assert skipped.profile is None

    learned = engine.learn(user_id="u1", messages=messages, learn_style_button=True)
    assert learned.learned is True
    assert learned.profile is not None
    assert learned.profile["speech_dna_core"]["style_embedding"]


def test_speech_dna_contains_expected_fields() -> None:
    profile = {
        "style_embedding": [0.3, 0.7, 0.2, 0.1, 0.4, 0.5, 0.8, 0.0],
        "style_examples": ["Look, keep it practical."],
        "speech_dna": {
            "vocabulary_bias": ["look", "practical"],
            "punctuation_profile": {"!": 0.2},
            "sentence_rhythm": [1.0, 0.9],
        },
    }
    dna = SpeechDNA.from_style_profile(profile)
    assert dna.typical_phrases == ("Look, keep it practical.",)
    assert dna.vocabulary_patterns == ("look", "practical")
    assert dna.punctuation_profile["!"] == 0.2


def test_dialogue_engine_builds_personality_and_style_aware_contract() -> None:
    node = GraphNode(
        node_id="pattern:guilt_pressure",
        node_core=NodeCore(
            node_id="pattern:guilt_pressure",
            node_type="PATTERN",
            name="Guilt pressure",
            description="Pressure framed as proof of care.",
            importance_vector=ImportanceVector(0.9, 0.7, 0.8, 0.85),
        ),
    )
    ram = RAMContextGraph(
        query="Is this pressure manipulative?",
        signals={"query_signals": {"intent": ["control"]}},
        nodes=(node,),
        edges=(GraphEdge(src_id="pattern:guilt_pressure", dst_id="domain:psychology", edge_type="IS_PART_OF", weight=0.9),),
        ranked_nodes=(RankedContextNode(node=node, score=2.4, reasons=("signal:control",)),),
    )
    personality = PersonalityCore(
        temperament="guarded",
        values=("clarity", "self_respect"),
        speech_style=SpeechStyleCore(formality=0.2, slang_level=0.3, directness=0.8, profanity_tolerance=0.1),
        reasoning_style="grounded",
        risk_tolerance=0.4,
        aggression_level=0.2,
        humor_level=0.1,
    )
    speech_dna = SpeechDNA(
        style_embedding=(0.3, 0.6, 0.2, 0.1, 0.3, 0.2, 0.8, 0.1),
        typical_phrases=("Look,",),
        vocabulary_patterns=("direct", "practical"),
    )
    roles = tuple(default_agent_roles().values())
    contract = DialogueEngine().build_contract(
        query="Is this pressure manipulative?",
        personality_core=personality,
        speech_dna=speech_dna,
        ram_graph=ram,
        agent_roles=roles[:2],
        scenario={"scenario_type": "conflict_dialogue"},
    )
    payload = contract.as_dict()
    assert payload["personality_summary"]["temperament"] == "guarded"
    assert payload["style_summary"]["typical_phrases"] == ["Look,"]
    assert payload["graph_summary"]["ranked_nodes"][0]["node_id"] == "pattern:guilt_pressure"
    assert payload["agent_roles"]


def test_scenario_engine_and_agent_roles_cover_required_professional_paths() -> None:
    scenarios = ScenarioEngine()
    assert scenarios.pick("There is contract risk here.").scenario_type == "professional_consultation"
    assert scenarios.pick("He keeps pushing after criticism and conflict.").scenario_type == "conflict_dialogue"
    roles = default_agent_roles()
    assert {"law", "business", "strategy"} <= set(roles.keys())
