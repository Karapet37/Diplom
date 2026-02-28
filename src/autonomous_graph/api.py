"""High-level API layer for autonomous graph simulation."""

from __future__ import annotations

import os
from typing import Any, Mapping

from src.autonomous_graph.core import (
    Edge,
    EmploymentRecord,
    GraphEvent,
    GraphEngine,
    Human,
    Node,
)
from src.autonomous_graph.storage import JsonGraphDBAdapter, Neo4jGraphDBAdapter


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name, "1" if default else "0").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _autoload_engine(engine: GraphEngine) -> GraphEngine:
    if engine.graph_adapter is None:
        return engine
    if not _bool_env("AUTOGRAPH_AUTO_LOAD_ON_START", True):
        return engine
    try:
        engine.load_from_adapter()
    except Exception:
        pass
    return engine


def build_graph_engine_from_env() -> GraphEngine:
    """
    Build GraphEngine with storage adapter from environment.

    Priority:
    1) Neo4j (if enabled and credentials are present)
    2) JSON adapter
    3) In-memory only
    """
    adapter_mode = os.getenv("AUTOGRAPH_STORAGE_ADAPTER", "auto").strip().lower() or "auto"

    neo4j_allowed = adapter_mode in {"auto", "neo4j"}
    if neo4j_allowed and _bool_env("AUTOGRAPH_NEO4J_ENABLE", False):
        uri = os.getenv("NEO4J_URI", "").strip()
        user = os.getenv("NEO4J_USER", "").strip()
        password = os.getenv("NEO4J_PASSWORD", "").strip()
        database = os.getenv("NEO4J_DATABASE", "neo4j").strip() or "neo4j"
        if uri and user and password:
            try:
                adapter = Neo4jGraphDBAdapter(
                    uri=uri,
                    user=user,
                    password=password,
                    database=database,
                )
                return _autoload_engine(GraphEngine(graph_adapter=adapter))
            except Exception:
                pass

    json_allowed = adapter_mode in {"auto", "json"}
    if json_allowed and _bool_env("AUTOGRAPH_JSON_ENABLE", True):
        path = os.getenv("AUTOGRAPH_JSON_PATH", "data/autonomous_graph_snapshot.json").strip()
        adapter = JsonGraphDBAdapter(path)
        return _autoload_engine(GraphEngine(graph_adapter=adapter))

    return GraphEngine()


class GraphAPI:
    """Facade around GraphEngine for external integration."""

    def __init__(self, engine: GraphEngine | None = None):
        self.engine = engine or GraphEngine()

    def create_human(
        self,
        *,
        first_name: str,
        last_name: str = "",
        bio: str = "",
        employment: list[Mapping[str, Any]] | None = None,
        attributes: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> Human:
        jobs = [
            EmploymentRecord(
                status=str(item.get("status", "") or "").strip(),
                importance_score=float(item.get("importance_score", 1.0) or 1.0),
                company_name=str(item.get("company_name", "") or "").strip(),
                company_attributes=dict(item.get("company_attributes", {}) or {}),
            )
            for item in (employment or [])
        ]
        payload = dict(attributes or {})
        payload.setdefault("first_name", first_name)
        payload.setdefault("last_name", last_name)
        payload.setdefault("bio", bio)
        payload.setdefault("employment_status", [
            {
                "status": job.status,
                "importance_score": job.importance_score,
                "company_name": job.company_name,
                "company_attributes": dict(job.company_attributes),
            }
            for job in jobs
        ])
        node = self.engine.create_node(
            "human",
            first_name=None,
            last_name=None,
            bio=bio,
            employment_status=jobs,
            attributes=payload,
            state=state,
        )
        if isinstance(node, Human):
            return node
        raise TypeError("human factory returned unexpected node type")

    def create_company(
        self,
        *,
        name: str,
        industry: str = "",
        description: str = "",
        attributes: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> Node:
        payload = dict(attributes or {})
        payload.setdefault("name", name)
        payload.setdefault("industry", industry)
        payload.setdefault("description", description)
        return self.engine.create_node(
            "company",
            name=name,
            industry=industry,
            description=description,
            attributes=payload,
            state=state,
        )

    def connect(
        self,
        from_node: int,
        to_node: int,
        *,
        relation_type: str,
        weight: float = 1.0,
        direction: str = "directed",
        logic_rule: str = "explicit",
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        return self.engine.add_edge(
            Edge(
                from_node=from_node,
                to_node=to_node,
                relation_type=relation_type,
                weight=weight,
                direction=direction,
                logic_rule=logic_rule,
                metadata=dict(metadata or {}),
            )
        )

    def simulate(
        self,
        *,
        seed_node_ids: list[int] | None = None,
        recursive_depth: int = 2,
        propagation_steps: int = 3,
        damping: float = 0.15,
        activation: str = "tanh",
        infer_rounds: int = 1,
    ) -> dict[str, Any]:
        self.engine._record_event(  # noqa: SLF001
            "simulation_started",
            {
                "seed_node_ids": list(seed_node_ids or []),
                "recursive_depth": int(recursive_depth),
                "propagation_steps": int(propagation_steps),
                "damping": float(damping),
                "activation": str(activation or "tanh"),
                "infer_rounds": int(max(1, int(infer_rounds))),
            },
        )
        autodetect_logs = self.engine.recursive_generation_operator(
            seed_node_ids=seed_node_ids,
            max_depth=recursive_depth,
        )
        self.engine._record_event(  # noqa: SLF001
            "simulation_phase",
            {
                "phase": "recursive_generation",
                "processed_nodes": len(autodetect_logs),
                "created_nodes": sum(len(item.new_nodes) for item in autodetect_logs),
                "created_edges": sum(len(item.new_edges) for item in autodetect_logs),
            },
        )
        inferred: list[dict[str, Any]] = []
        rounds = max(1, int(infer_rounds))
        for round_index in range(rounds):
            new_edges = self.engine.logical_inference_operator()
            for edge in new_edges:
                inferred.append(
                    {
                        "from": edge.from_node,
                        "to": edge.to_node,
                        "relation_type": edge.relation_type,
                        "logic_rule": edge.logic_rule,
                        "weight": edge.weight,
                    }
                )
            self.engine._record_event(  # noqa: SLF001
                "simulation_infer_round",
                {
                    "round": round_index + 1,
                    "rounds_total": rounds,
                    "added_edges": len(new_edges),
                },
            )
            if not new_edges:
                break

        state = self.engine.state_propagation_operator(
            steps=propagation_steps,
            damping=damping,
            activation=activation,
        )
        snapshot = self.engine.snapshot()
        self.engine._record_event(  # noqa: SLF001
            "simulation_completed",
            {
                "inferred_edges": len(inferred),
                "autodetect_notes": sum(len(item.notes) for item in autodetect_logs),
                "nodes": len(snapshot.get("nodes", {}) or {}),
                "edges": len(snapshot.get("edges", []) or []),
            },
        )
        return {
            "autodetect_notes": [note for item in autodetect_logs for note in item.notes],
            "inferred_edges": inferred,
            "state": state,
            "events": [
                {
                    "id": event.id,
                    "type": event.event_type,
                    "timestamp": event.timestamp,
                }
                for event in self.engine.get_event_log(limit=100)
            ],
            "snapshot": snapshot,
        }

    def get_events(
        self,
        *,
        limit: int | None = None,
        event_type: str | None = None,
    ) -> list[GraphEvent]:
        return self.engine.get_event_log(limit=limit, event_type=event_type)

    def reward_event(
        self,
        event_id: int,
        *,
        reward: float,
        learning_rate: float = 0.15,
    ) -> bool:
        return self.engine.reward_event(
            event_id,
            reward=reward,
            learning_rate=learning_rate,
        )

    def reinforce_relation(
        self,
        relation_type: str,
        *,
        reward: float,
        learning_rate: float = 0.15,
    ) -> int:
        return self.engine.reinforce_relations(
            relation_type,
            reward=reward,
            learning_rate=learning_rate,
        )

    def persist(self) -> bool:
        return self.engine.persist()

    def load(self) -> bool:
        return self.engine.load_from_adapter()


def run_demo_simulation(*, use_env_adapter: bool = True) -> dict[str, Any]:
    """Reference simulation scenario."""
    engine = build_graph_engine_from_env() if use_env_adapter else GraphEngine()
    api = GraphAPI(engine)
    human = api.create_human(
        first_name="Elena",
        last_name="Markova",
        bio=(
            "I design data systems for product teams. "
            "I care about long-term growth and rigorous analysis."
        ),
        employment=[
            {
                "status": "senior data scientist",
                "importance_score": 0.92,
                "company_name": "Aurora Analytics",
                "company_attributes": {
                    "industry": "AI SaaS",
                    "description": "Provides analytical platforms for retail forecasting.",
                },
            }
        ],
        state={"influence": 0.65, "trust": 0.55},
    )
    aurora = api.create_company(
        name="Aurora Analytics",
        industry="AI SaaS",
        description="Provides analytical platforms for retail forecasting.",
        state={"influence": 0.55, "trust": 0.45},
    )
    holding = api.create_company(
        name="Northlight Group",
        industry="Holding",
        description="Strategic holding with multi-market subsidiaries.",
        state={"influence": 0.8},
    )
    api.connect(
        from_node=aurora.id,
        to_node=holding.id,
        relation_type="part_of",
        weight=0.9,
        logic_rule="manual_setup",
    )
    api.connect(
        from_node=holding.id,
        to_node=aurora.id,
        relation_type="influences",
        weight=0.75,
        logic_rule="board_influence",
    )
    result = api.simulate(seed_node_ids=[human.id], recursive_depth=3, propagation_steps=4)
    if _bool_env("AUTOGRAPH_DEMO_PERSIST", False):
        result["persisted"] = api.persist()
    return result
