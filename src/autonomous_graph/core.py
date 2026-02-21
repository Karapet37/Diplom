"""Core abstractions for autonomous logical graph systems."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
import time
from typing import Any, Callable, Mapping, Protocol

StateVector = dict[str, float]
EdgeTransformFn = Callable[[StateVector, StateVector, "Edge"], StateVector]

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+", re.UNICODE)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_state_vector(payload: Mapping[str, Any] | None) -> StateVector:
    if not payload:
        return {}
    out: StateVector = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
    return out


@dataclass(frozen=True)
class WeightedValue:
    value: Any
    importance_score: float = 1.0

    def importance(self) -> float:
        return _clamp01(self.importance_score)


@dataclass
class EmploymentRecord:
    status: str
    importance_score: float = 1.0
    company_name: str = ""
    relation_to_company_node: int | None = None
    company_attributes: dict[str, Any] = field(default_factory=dict)

    def importance(self) -> float:
        return _clamp01(self.importance_score)


@dataclass(frozen=True)
class TextAnalysis:
    sentiment: float
    topics: tuple[str, ...]
    mindset: str
    outlook: str
    cognitive_profile: dict[str, float]


class TextAnalyzer(Protocol):
    def analyze(
        self,
        text: str,
        *,
        node_type: str,
        hints: Mapping[str, Any] | None = None,
    ) -> TextAnalysis:
        ...


class GraphDBAdapter(Protocol):
    def persist_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        ...

    def load_snapshot(self) -> Mapping[str, Any]:
        ...


class HeuristicTextAnalyzer:
    """Fast offline fallback analyzer."""

    POSITIVE = {
        "growth",
        "improve",
        "optimistic",
        "creative",
        "innovation",
        "progress",
        "collaborate",
        "успех",
        "рост",
        "развитие",
        "улучшить",
    }
    NEGATIVE = {
        "risk",
        "stress",
        "fear",
        "failure",
        "loss",
        "anxiety",
        "problem",
        "риск",
        "страх",
        "провал",
        "убыток",
        "проблема",
    }
    ANALYTICAL = {
        "analysis",
        "logic",
        "system",
        "data",
        "model",
        "algorithm",
        "метрика",
        "логика",
        "анализ",
        "структура",
        "система",
    }
    CREATIVE = {
        "design",
        "art",
        "idea",
        "imagine",
        "music",
        "story",
        "креатив",
        "идея",
        "дизайн",
        "творчество",
    }
    SOCIAL = {
        "team",
        "people",
        "community",
        "friend",
        "user",
        "клиент",
        "команда",
        "люди",
        "пользователь",
    }

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in TOKEN_RE.findall(text or "")]

    @staticmethod
    def _score(tokens: list[str], lexicon: set[str]) -> float:
        if not tokens:
            return 0.0
        hits = sum(1 for token in tokens if token in lexicon)
        return min(1.0, hits / max(1.0, len(tokens) * 0.25))

    def analyze(
        self,
        text: str,
        *,
        node_type: str,
        hints: Mapping[str, Any] | None = None,
    ) -> TextAnalysis:
        tokens = self._tokenize(text)
        if not tokens:
            return TextAnalysis(
                sentiment=0.0,
                topics=("unknown",),
                mindset="insufficient_data",
                outlook="neutral",
                cognitive_profile={
                    "analytical": 0.0,
                    "creative": 0.0,
                    "social": 0.0,
                },
            )

        pos = self._score(tokens, self.POSITIVE)
        neg = self._score(tokens, self.NEGATIVE)
        analytical = self._score(tokens, self.ANALYTICAL)
        creative = self._score(tokens, self.CREATIVE)
        social = self._score(tokens, self.SOCIAL)
        sentiment = max(-1.0, min(1.0, pos - neg))

        topics: list[str] = []
        if analytical >= 0.2:
            topics.append("analysis")
        if creative >= 0.2:
            topics.append("creativity")
        if social >= 0.2:
            topics.append("social")
        if neg >= 0.25:
            topics.append("risk")
        if not topics:
            topics.append(node_type or "general")

        if analytical >= creative + 0.1:
            mindset = "analytical"
        elif creative >= analytical + 0.1:
            mindset = "creative"
        else:
            mindset = "balanced"

        if sentiment > 0.2:
            outlook = "optimistic"
        elif sentiment < -0.2:
            outlook = "defensive"
        else:
            outlook = "neutral"

        return TextAnalysis(
            sentiment=sentiment,
            topics=tuple(topics),
            mindset=mindset,
            outlook=outlook,
            cognitive_profile={
                "analytical": analytical,
                "creative": creative,
                "social": social,
            },
        )


class LLMTextAnalyzer:
    """LLM-first analyzer with heuristic fallback."""

    def __init__(
        self,
        *,
        llm_fn: Callable[[str], str] | None = None,
        fallback: TextAnalyzer | None = None,
    ):
        self.fallback = fallback or HeuristicTextAnalyzer()
        if llm_fn is None:
            try:
                from src.utils.local_llm_provider import build_local_llm_fn

                llm_fn = build_local_llm_fn()
            except Exception:
                llm_fn = None
        self.llm_fn = llm_fn

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        payload = str(text or "").strip()
        if not payload:
            return None
        start = payload.find("{")
        end = payload.rfind("}")
        if start < 0 or end <= start:
            return None
        chunk = payload[start : end + 1]
        try:
            parsed = json.loads(chunk)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def analyze(
        self,
        text: str,
        *,
        node_type: str,
        hints: Mapping[str, Any] | None = None,
    ) -> TextAnalysis:
        if not self.llm_fn:
            return self.fallback.analyze(text, node_type=node_type, hints=hints)

        prompt = (
            "You are a compact cognitive profiler for graph nodes.\n"
            "Return only JSON with keys: sentiment, topics, mindset, outlook, cognitive_profile.\n"
            "sentiment must be in [-1,1].\n"
            "cognitive_profile must include analytical, creative, social scores in [0,1].\n"
            f"Node type: {node_type}\n"
            f"Hints: {dict(hints or {})}\n"
            f"Text: {text}\n"
            "JSON:"
        )
        try:
            raw = str(self.llm_fn(prompt) or "")
        except Exception:
            return self.fallback.analyze(text, node_type=node_type, hints=hints)

        parsed = self._extract_json(raw)
        if not parsed:
            return self.fallback.analyze(text, node_type=node_type, hints=hints)

        sentiment = max(-1.0, min(1.0, _safe_float(parsed.get("sentiment"), 0.0)))
        topics_raw = parsed.get("topics", ())
        if isinstance(topics_raw, str):
            topics = tuple(token.strip() for token in topics_raw.split(",") if token.strip())
        elif isinstance(topics_raw, list):
            topics = tuple(str(item).strip() for item in topics_raw if str(item).strip())
        else:
            topics = ()
        if not topics:
            topics = (node_type or "general",)

        mindset = str(parsed.get("mindset", "balanced") or "balanced").strip()
        outlook = str(parsed.get("outlook", "neutral") or "neutral").strip()

        profile_raw = parsed.get("cognitive_profile", {})
        profile: dict[str, float] = {
            "analytical": _clamp01(_safe_float(getattr(profile_raw, "get", lambda *_: 0.0)("analytical"))),
            "creative": _clamp01(_safe_float(getattr(profile_raw, "get", lambda *_: 0.0)("creative"))),
            "social": _clamp01(_safe_float(getattr(profile_raw, "get", lambda *_: 0.0)("social"))),
        }
        return TextAnalysis(
            sentiment=sentiment,
            topics=topics,
            mindset=mindset,
            outlook=outlook,
            cognitive_profile=profile,
        )


@dataclass
class Edge:
    from_node: int
    to_node: int
    relation_type: str
    weight: float = 1.0
    direction: str = "directed"
    logic_rule: str = "explicit"
    transform: EdgeTransformFn | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_weight(self) -> float:
        return _clamp01(self.weight)


@dataclass
class AutodetectResult:
    new_nodes: list["Node"] = field(default_factory=list)
    new_edges: list[Edge] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GraphEvent:
    id: int
    event_type: str
    timestamp: float
    payload: dict[str, Any]


class Node:
    """Universal autonomous node."""

    def __init__(
        self,
        id: int,
        *,
        node_type: str = "generic",
        attributes: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
        text_analyzer: TextAnalyzer | None = None,
    ):
        self.id = int(id)
        self.type = str(node_type or "generic")
        self.attributes: dict[str, Any] = dict(attributes or {})
        self.state: StateVector = _to_state_vector(state)
        self.text_analyzer: TextAnalyzer = text_analyzer or HeuristicTextAnalyzer()
        self.function = self.autodetect

    def _iter_text_values(self, value: Any) -> list[str]:
        if isinstance(value, str):
            normalized = " ".join(value.split()).strip()
            return [normalized] if normalized else []
        if isinstance(value, Mapping):
            out: list[str] = []
            for item in value.values():
                out.extend(self._iter_text_values(item))
            return out
        if isinstance(value, (list, tuple, set)):
            out = []
            for item in value:
                out.extend(self._iter_text_values(item))
            return out
        return []

    def collect_text_payload(self) -> str:
        parts: list[str] = []
        for value in self.attributes.values():
            parts.extend(self._iter_text_values(value))
        return "\n".join(part for part in parts if part).strip()

    def analyze_text_fields(self) -> TextAnalysis | None:
        payload = self.collect_text_payload()
        if not payload:
            return None
        analysis = self.text_analyzer.analyze(
            payload,
            node_type=self.type,
            hints=self.attributes,
        )
        self.attributes["mindset"] = analysis.mindset
        self.attributes["outlook"] = analysis.outlook
        self.attributes["cognitive_profile"] = dict(analysis.cognitive_profile)
        self.attributes["topics"] = list(analysis.topics)
        self.state["sentiment"] = float(analysis.sentiment)
        self.state["analytical"] = analysis.cognitive_profile.get("analytical", 0.0)
        self.state["creative"] = analysis.cognitive_profile.get("creative", 0.0)
        self.state["social"] = analysis.cognitive_profile.get("social", 0.0)
        return analysis

    def autodetect(
        self,
        *,
        recursive: bool = True,
        graph: "GraphEngine | None" = None,
        depth: int = 0,
        max_depth: int = 3,
    ) -> AutodetectResult:
        result = AutodetectResult()
        analysis = self.analyze_text_fields()
        if analysis is not None:
            result.notes.append(
                f"node={self.id} type={self.type} mindset={analysis.mindset} outlook={analysis.outlook}"
            )
        return result


class Human(Node):
    """Specialized human node with weighted profile fields."""

    def __init__(
        self,
        id: int,
        *,
        first_name: WeightedValue | None = None,
        last_name: WeightedValue | None = None,
        date_of_birth: WeightedValue | None = None,
        education: list[str] | None = None,
        employment_status: list[EmploymentRecord | Mapping[str, Any]] | None = None,
        bio: str = "",
        attributes: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
        text_analyzer: TextAnalyzer | None = None,
    ):
        base = dict(attributes or {})
        super().__init__(
            id,
            node_type="human",
            attributes=base,
            state=state,
            text_analyzer=text_analyzer,
        )
        self.first_name = first_name or WeightedValue(base.get("first_name", ""), base.get("first_name_importance", 0.8))
        self.last_name = last_name or WeightedValue(base.get("last_name", ""), base.get("last_name_importance", 0.7))
        self.date_of_birth = date_of_birth or WeightedValue(base.get("date_of_birth", ""), base.get("date_of_birth_importance", 0.6))
        self.education = list(education or base.get("education", []))
        self.bio = str(bio or base.get("bio", ""))
        self.employment_status: list[EmploymentRecord] = self._normalize_employment(
            employment_status or base.get("employment_status", [])
        )
        self.attributes["bio"] = self.bio
        self.attributes["education"] = list(self.education)
        self.attributes["employment_status"] = [
            {
                "status": item.status,
                "importance_score": item.importance(),
                "company_name": item.company_name,
                "relation_to_company_node": item.relation_to_company_node,
            }
            for item in self.employment_status
        ]

    @staticmethod
    def _normalize_employment(
        rows: list[EmploymentRecord | Mapping[str, Any]],
    ) -> list[EmploymentRecord]:
        normalized: list[EmploymentRecord] = []
        for row in rows:
            if isinstance(row, EmploymentRecord):
                normalized.append(row)
                continue
            if not isinstance(row, Mapping):
                continue
            normalized.append(
                EmploymentRecord(
                    status=str(row.get("status", "") or "").strip(),
                    importance_score=_safe_float(row.get("importance_score", 1.0), 1.0),
                    company_name=str(row.get("company_name", "") or "").strip(),
                    relation_to_company_node=row.get("relation_to_company_node"),
                    company_attributes=dict(row.get("company_attributes", {}) or {}),
                )
            )
        return normalized

    def _identity_strength(self) -> float:
        values = [self.first_name, self.last_name, self.date_of_birth]
        weighted = 0.0
        total = 0.0
        for item in values:
            if str(item.value).strip():
                weighted += item.importance()
                total += 1.0
        if total <= 0:
            return 0.0
        return _clamp01(weighted / total)

    def _derive_profile(self, analysis: TextAnalysis | None) -> None:
        profile = dict((analysis.cognitive_profile if analysis else {}))
        analytical = _clamp01(_safe_float(profile.get("analytical"), 0.0))
        creative = _clamp01(_safe_float(profile.get("creative"), 0.0))
        social = _clamp01(_safe_float(profile.get("social"), 0.0))
        identity = self._identity_strength()
        education_factor = _clamp01(len(self.education) / 5.0)
        analytical_creative = _clamp01((0.65 * analytical) + (0.35 * creative))
        cognitive = {
            "analytical": analytical,
            "creative": creative,
            "social": social,
            "identity_stability": identity,
            "education_factor": education_factor,
        }
        self.attributes["mindset"] = (
            analysis.mindset
            if analysis is not None
            else ("analytical" if analytical >= creative else "creative")
        )
        self.attributes["outlook"] = analysis.outlook if analysis is not None else "neutral"
        self.attributes["cognitive_profile"] = cognitive
        self.attributes["analytical_creative_score"] = analytical_creative
        self.state["analytical_creative_score"] = analytical_creative
        self.state["identity_stability"] = identity
        self.state["education_factor"] = education_factor

    def autodetect(
        self,
        *,
        recursive: bool = True,
        graph: "GraphEngine | None" = None,
        depth: int = 0,
        max_depth: int = 3,
    ) -> AutodetectResult:
        result = super().autodetect(
            recursive=recursive,
            graph=graph,
            depth=depth,
            max_depth=max_depth,
        )
        analysis = self.analyze_text_fields()
        self._derive_profile(analysis)
        if not recursive or graph is None or depth >= max_depth:
            return result

        for job in self.employment_status:
            company_name = " ".join(job.company_name.split()).strip()
            if not company_name:
                continue
            company_node = graph.find_company_by_name(company_name)
            if company_node is None:
                company_node = graph.create_node(
                    "company",
                    attributes={
                        "name": company_name,
                        "industry": job.company_attributes.get("industry", ""),
                        "description": job.company_attributes.get("description", ""),
                    },
                    state={
                        "influence": 0.1 * job.importance(),
                    },
                )
                result.new_nodes.append(company_node)
                result.notes.append(f"created_company:{company_node.id}:{company_name}")

            job.relation_to_company_node = company_node.id
            works_edge = Edge(
                from_node=self.id,
                to_node=company_node.id,
                relation_type="works_at",
                weight=job.importance(),
                direction="directed",
                logic_rule="employment_link",
            )
            if graph.add_edge(works_edge):
                result.new_edges.append(works_edge)
            if "owner" in job.status.lower() or "founder" in job.status.lower():
                owns_edge = Edge(
                    from_node=self.id,
                    to_node=company_node.id,
                    relation_type="owns",
                    weight=max(0.55, job.importance()),
                    direction="directed",
                    logic_rule="ownership_from_employment_status",
                )
                if graph.add_edge(owns_edge):
                    result.new_edges.append(owns_edge)

        self.attributes["employment_status"] = [
            {
                "status": item.status,
                "importance_score": item.importance(),
                "company_name": item.company_name,
                "relation_to_company_node": item.relation_to_company_node,
            }
            for item in self.employment_status
        ]
        return result


class Company(Node):
    """Specialized company node."""

    def __init__(
        self,
        id: int,
        *,
        name: str = "",
        industry: str = "",
        description: str = "",
        attributes: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
        text_analyzer: TextAnalyzer | None = None,
    ):
        base = dict(attributes or {})
        merged = dict(base)
        merged["name"] = str(name or base.get("name", ""))
        merged["industry"] = str(industry or base.get("industry", ""))
        merged["description"] = str(description or base.get("description", ""))
        super().__init__(
            id,
            node_type="company",
            attributes=merged,
            state=state,
            text_analyzer=text_analyzer,
        )

    @property
    def name(self) -> str:
        return str(self.attributes.get("name", "")).strip()

    def autodetect(
        self,
        *,
        recursive: bool = True,
        graph: "GraphEngine | None" = None,
        depth: int = 0,
        max_depth: int = 3,
    ) -> AutodetectResult:
        result = super().autodetect(
            recursive=recursive,
            graph=graph,
            depth=depth,
            max_depth=max_depth,
        )
        analysis = self.analyze_text_fields()
        if analysis is not None:
            self.state["market_confidence"] = _clamp01(0.5 + (0.5 * analysis.sentiment))
            self.state["innovation"] = analysis.cognitive_profile.get("creative", 0.0)

        if not recursive or graph is None or depth >= max_depth:
            return result

        parent_name = " ".join(str(self.attributes.get("parent_company", "")).split()).strip()
        if parent_name:
            parent = graph.find_company_by_name(parent_name)
            if parent is None:
                parent = graph.create_node(
                    "company",
                    attributes={"name": parent_name},
                    state={"influence": 0.2},
                )
                result.new_nodes.append(parent)
                result.notes.append(f"created_parent_company:{parent.id}:{parent_name}")
            edge = Edge(
                from_node=self.id,
                to_node=parent.id,
                relation_type="part_of",
                weight=0.8,
                direction="directed",
                logic_rule="company_hierarchy",
            )
            if graph.add_edge(edge):
                result.new_edges.append(edge)
        return result


class InMemoryGraphStore:
    """Simple storage layer for graph nodes and edges."""

    def __init__(self):
        self.nodes: dict[int, Node] = {}
        self.edges: list[Edge] = []


class GraphEngine:
    """
    Hybrid graph engine:
    - graph storage
    - recursive node generation
    - logical inference
    - state propagation
    """

    RELATION_EFFECTS: dict[str, dict[str, float]] = {
        "works_at": {"influence": 0.45, "trust": 0.35},
        "owns": {"influence": 0.80, "trust": 0.20},
        "influences": {"influence": 0.70, "risk": 0.10},
        "competes_with": {"risk": 0.55, "influence": 0.15},
        "parent_of": {"trust": 0.60, "influence": 0.20},
        "part_of": {"influence": 0.40, "trust": 0.25},
    }

    def __init__(
        self,
        *,
        store: InMemoryGraphStore | None = None,
        text_analyzer: TextAnalyzer | None = None,
        graph_adapter: GraphDBAdapter | None = None,
    ):
        self.store = store or InMemoryGraphStore()
        self.text_analyzer = text_analyzer or HeuristicTextAnalyzer()
        self.graph_adapter = graph_adapter
        self.node_types: dict[str, type[Node]] = {
            "generic": Node,
            "human": Human,
            "company": Company,
        }
        self.event_log: list[GraphEvent] = []
        self._event_listeners: list[Callable[[GraphEvent], None]] = []
        self._next_event_id = 1
        self._next_node_id = 1
        if self.store.nodes:
            self._next_node_id = max(self.store.nodes) + 1

    def _record_event(self, event_type: str, payload: Mapping[str, Any]) -> GraphEvent:
        event = GraphEvent(
            id=self._next_event_id,
            event_type=str(event_type),
            timestamp=time.time(),
            payload=dict(payload),
        )
        self._next_event_id += 1
        self.event_log.append(event)
        # Best-effort dispatch to runtime subscribers (e.g. websocket stream).
        for listener in list(self._event_listeners):
            try:
                listener(event)
            except Exception:
                continue
        return event

    def add_event_listener(self, listener: Callable[[GraphEvent], None]) -> None:
        if listener in self._event_listeners:
            return
        self._event_listeners.append(listener)

    def remove_event_listener(self, listener: Callable[[GraphEvent], None]) -> None:
        self._event_listeners = [item for item in self._event_listeners if item != listener]

    def register_node_type(self, name: str, node_cls: type[Node]) -> None:
        key = str(name or "").strip().lower()
        if not key:
            raise ValueError("node type name is required")
        if not issubclass(node_cls, Node):
            raise TypeError("node_cls must inherit Node")
        self.node_types[key] = node_cls

    def _allocate_node_id(self) -> int:
        value = self._next_node_id
        self._next_node_id += 1
        return value

    def add_node(self, node: Node) -> Node:
        self.store.nodes[node.id] = node
        if node.id >= self._next_node_id:
            self._next_node_id = node.id + 1
        self._record_event(
            "node_added",
            {
                "node_id": node.id,
                "node_type": node.type,
                "node": {
                    "id": node.id,
                    "type": node.type,
                    "attributes": dict(node.attributes),
                    "state": dict(node.state),
                },
            },
        )
        return node

    def create_node(
        self,
        node_type: str,
        *,
        node_id: int | None = None,
        attributes: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> Node:
        key = str(node_type or "generic").strip().lower() or "generic"
        cls = self.node_types.get(key, Node)
        assigned_id = node_id if node_id is not None else self._allocate_node_id()
        analyzer = kwargs.pop("text_analyzer", self.text_analyzer)
        if cls is Node:
            node = cls(
                assigned_id,
                node_type=key,
                attributes=attributes,
                state=state,
                text_analyzer=analyzer,
            )
        else:
            node = cls(
                assigned_id,
                attributes=attributes,
                state=state,
                text_analyzer=analyzer,
                **kwargs,
            )
        if key not in {"generic", "human", "company"}:
            node.type = key
        return self.add_node(node)

    def get_node(self, node_id: int) -> Node | None:
        return self.store.nodes.get(node_id)

    def find_company_by_name(self, name: str) -> Company | None:
        normalized = " ".join(str(name or "").lower().split()).strip()
        if not normalized:
            return None
        for node in self.store.nodes.values():
            if node.type != "company":
                continue
            candidate = " ".join(str(node.attributes.get("name", "")).lower().split()).strip()
            if candidate == normalized and isinstance(node, Company):
                return node
            if candidate == normalized:
                return Company(
                    node.id,
                    attributes=node.attributes,
                    state=node.state,
                    text_analyzer=node.text_analyzer,
                )
        return None

    def add_edge(self, edge: Edge) -> bool:
        if edge.from_node not in self.store.nodes or edge.to_node not in self.store.nodes:
            raise ValueError("both nodes must exist before adding edge")
        key = (
            int(edge.from_node),
            int(edge.to_node),
            str(edge.relation_type),
            str(edge.direction),
        )
        for existing in self.store.edges:
            existing_key = (
                existing.from_node,
                existing.to_node,
                existing.relation_type,
                existing.direction,
            )
            if existing_key == key:
                existing.weight = max(existing.weight, edge.weight)
                if edge.logic_rule and existing.logic_rule == "explicit":
                    existing.logic_rule = edge.logic_rule
                if edge.metadata:
                    existing.metadata.update(dict(edge.metadata))
                self._record_event(
                    "edge_updated",
                    {
                        "from": existing.from_node,
                        "to": existing.to_node,
                        "relation_type": existing.relation_type,
                        "direction": existing.direction,
                        "weight": existing.weight,
                        "logic_rule": existing.logic_rule,
                        "metadata": dict(existing.metadata),
                    },
                )
                return False
        edge.weight = edge.normalized_weight()
        self.store.edges.append(edge)
        self._record_event(
            "edge_added",
            {
                "from": edge.from_node,
                "to": edge.to_node,
                "relation_type": edge.relation_type,
                "direction": edge.direction,
                "weight": edge.weight,
                "logic_rule": edge.logic_rule,
                "metadata": dict(edge.metadata),
            },
        )
        return True

    def neighbors(self, node_id: int, *, direction: str = "out") -> list[Node]:
        out: list[Node] = []
        for edge in self.store.edges:
            if direction == "out" and edge.from_node == node_id:
                node = self.get_node(edge.to_node)
                if node is not None:
                    out.append(node)
            elif direction == "in" and edge.to_node == node_id:
                node = self.get_node(edge.from_node)
                if node is not None:
                    out.append(node)
            elif direction == "both" and (edge.to_node == node_id or edge.from_node == node_id):
                other_id = edge.to_node if edge.from_node == node_id else edge.from_node
                node = self.get_node(other_id)
                if node is not None:
                    out.append(node)
        return out

    def recursive_generation_operator(
        self,
        *,
        seed_node_ids: list[int] | None = None,
        max_depth: int = 3,
    ) -> list[AutodetectResult]:
        queue: list[tuple[int, int]] = []
        visited: set[tuple[int, int]] = set()
        if seed_node_ids:
            for node_id in seed_node_ids:
                queue.append((int(node_id), 0))
        else:
            for node_id in self.store.nodes:
                queue.append((node_id, 0))

        results: list[AutodetectResult] = []
        while queue:
            node_id, depth = queue.pop(0)
            marker = (node_id, depth)
            if marker in visited:
                continue
            visited.add(marker)
            node = self.get_node(node_id)
            if node is None:
                continue
            result = node.autodetect(
                recursive=True,
                graph=self,
                depth=depth,
                max_depth=max_depth,
            )
            results.append(result)
            if depth >= max_depth:
                continue
            for created in result.new_nodes:
                queue.append((created.id, depth + 1))
            for edge in result.new_edges:
                queue.append((edge.to_node, depth + 1))
        self._record_event(
            "recursive_generation",
            {
                "seed_node_ids": list(seed_node_ids or []),
                "max_depth": int(max_depth),
                "processed_nodes": len(results),
                "created_nodes": sum(len(item.new_nodes) for item in results),
                "created_edges": sum(len(item.new_edges) for item in results),
            },
        )
        return results

    def _relation_message(
        self,
        edge: Edge,
        source_state: StateVector,
        target_state: StateVector,
    ) -> StateVector:
        if edge.transform is not None:
            try:
                return dict(edge.transform(source_state, target_state, edge))
            except Exception:
                return {}

        effect = self.RELATION_EFFECTS.get(edge.relation_type, {})
        weight = edge.normalized_weight()
        if not effect:
            return {key: (value * 0.25 * weight) for key, value in source_state.items()}

        out: StateVector = {}
        for key, gain in effect.items():
            source_value = source_state.get(key, source_state.get("influence", 0.0))
            out[key] = source_value * gain * weight
        if "influence" not in out:
            out["influence"] = source_state.get("influence", 0.0) * 0.20 * weight
        return out

    @staticmethod
    def _activate(value: float, mode: str) -> float:
        if mode == "identity":
            return float(value)
        if mode == "relu":
            return max(0.0, float(value))
        if mode == "sigmoid":
            try:
                return 1.0 / (1.0 + math.exp(-float(value)))
            except Exception:
                return 0.0
        # tanh by default
        try:
            return math.tanh(float(value))
        except Exception:
            return 0.0

    @staticmethod
    def _accumulate(target: StateVector, delta: Mapping[str, float]) -> None:
        for key, value in delta.items():
            target[key] = target.get(key, 0.0) + float(value)

    def state_propagation_operator(
        self,
        *,
        steps: int = 1,
        damping: float = 0.15,
        activation: str = "tanh",
    ) -> dict[int, StateVector]:
        rounds = max(1, int(steps))
        for idx in range(rounds):
            incoming: dict[int, StateVector] = {
                node_id: {} for node_id in self.store.nodes
            }
            for edge in self.store.edges:
                source = self.get_node(edge.from_node)
                target = self.get_node(edge.to_node)
                if source is None or target is None:
                    continue
                msg = self._relation_message(edge, source.state, target.state)
                self._accumulate(incoming[target.id], msg)
                if edge.direction == "undirected":
                    reverse_msg = self._relation_message(edge, target.state, source.state)
                    self._accumulate(incoming[source.id], reverse_msg)

            for node_id, node in self.store.nodes.items():
                previous = dict(node.state)
                merged_keys = set(previous) | set(incoming[node_id])
                updated: StateVector = {}
                for key in merged_keys:
                    value = ((1.0 - damping) * previous.get(key, 0.0)) + incoming[node_id].get(key, 0.0)
                    updated[key] = self._activate(value, activation)
                node.state = updated
            self._record_event(
                "state_propagation_step",
                {
                    "step": idx + 1,
                    "steps_total": rounds,
                    "damping": float(damping),
                    "activation": activation,
                    "nodes": len(self.store.nodes),
                    "edges": len(self.store.edges),
                },
            )
        return {node_id: dict(node.state) for node_id, node in self.store.nodes.items()}

    def logical_inference_operator(self, *, max_new_edges: int = 128) -> list[Edge]:
        added: list[Edge] = []
        by_from: dict[int, list[Edge]] = {}
        for edge in self.store.edges:
            by_from.setdefault(edge.from_node, []).append(edge)

        candidates: list[Edge] = []
        for edge in self.store.edges:
            if edge.relation_type == "works_at":
                for next_edge in by_from.get(edge.to_node, []):
                    if next_edge.relation_type == "part_of":
                        candidates.append(
                            Edge(
                                from_node=edge.from_node,
                                to_node=next_edge.to_node,
                                relation_type="part_of",
                                weight=edge.weight * next_edge.weight * 0.85,
                                logic_rule="transitive(works_at,part_of)",
                            )
                        )
            if edge.relation_type == "owns":
                for next_edge in by_from.get(edge.to_node, []):
                    if next_edge.relation_type == "competes_with":
                        candidates.append(
                            Edge(
                                from_node=edge.from_node,
                                to_node=next_edge.to_node,
                                relation_type="influences",
                                weight=edge.weight * next_edge.weight * 0.60,
                                logic_rule="ownership_competition_influence",
                            )
                        )
            if edge.relation_type == "influences":
                for next_edge in by_from.get(edge.to_node, []):
                    if next_edge.relation_type == "influences":
                        candidates.append(
                            Edge(
                                from_node=edge.from_node,
                                to_node=next_edge.to_node,
                                relation_type="influences",
                                weight=edge.weight * next_edge.weight * 0.75,
                                logic_rule="transitive(influences,influences)",
                            )
                        )

        for edge in candidates:
            if len(added) >= max_new_edges:
                break
            if self.add_edge(edge):
                added.append(edge)
        self._record_event(
            "logical_inference",
            {
                "max_new_edges": int(max_new_edges),
                "candidate_edges": len(candidates),
                "added_edges": len(added),
            },
        )
        return added

    def snapshot(self) -> dict[str, Any]:
        return {
            "nodes": {
                node_id: {
                    "type": node.type,
                    "attributes": dict(node.attributes),
                    "state": dict(node.state),
                }
                for node_id, node in self.store.nodes.items()
            },
            "edges": [
                {
                    "from": edge.from_node,
                    "to": edge.to_node,
                    "relation_type": edge.relation_type,
                    "weight": edge.weight,
                    "direction": edge.direction,
                    "logic_rule": edge.logic_rule,
                    "metadata": dict(edge.metadata),
                }
                for edge in self.store.edges
            ],
        }

    def _find_edge(
        self,
        *,
        from_node: int,
        to_node: int,
        relation_type: str,
        direction: str = "directed",
    ) -> Edge | None:
        for edge in self.store.edges:
            if (
                edge.from_node == from_node
                and edge.to_node == to_node
                and edge.relation_type == relation_type
                and edge.direction == direction
            ):
                return edge
        return None

    def get_event_log(
        self,
        *,
        limit: int | None = None,
        event_type: str | None = None,
    ) -> list[GraphEvent]:
        rows = self.event_log
        if event_type:
            rows = [item for item in rows if item.event_type == event_type]
        if limit is not None and limit >= 0:
            return rows[-limit:]
        return list(rows)

    def clear_event_log(self) -> None:
        self.event_log.clear()

    def reward_event(
        self,
        event_id: int,
        *,
        reward: float,
        learning_rate: float = 0.15,
    ) -> bool:
        target = None
        for event in self.event_log:
            if event.id == int(event_id):
                target = event
                break
        if target is None:
            return False
        payload = target.payload
        relation = str(payload.get("relation_type", "") or "")
        if not relation:
            return False
        edge = self._find_edge(
            from_node=int(payload.get("from", -1)),
            to_node=int(payload.get("to", -1)),
            relation_type=relation,
            direction=str(payload.get("direction", "directed")),
        )
        if edge is None:
            return False
        old_weight = edge.weight
        delta = float(learning_rate) * float(reward)
        if delta >= 0.0:
            edge.weight = _clamp01(edge.weight + (delta * (1.0 - edge.weight)))
        else:
            edge.weight = _clamp01(edge.weight + (delta * edge.weight))
        self._record_event(
            "edge_weight_feedback",
            {
                "source_event_id": int(event_id),
                "from": edge.from_node,
                "to": edge.to_node,
                "relation_type": edge.relation_type,
                "old_weight": old_weight,
                "new_weight": edge.weight,
                "reward": float(reward),
                "learning_rate": float(learning_rate),
            },
        )
        return True

    def reinforce_relations(
        self,
        relation_type: str,
        *,
        reward: float,
        learning_rate: float = 0.15,
    ) -> int:
        rel = str(relation_type or "").strip()
        if not rel:
            return 0
        updated = 0
        for edge in self.store.edges:
            if edge.relation_type != rel:
                continue
            old_weight = edge.weight
            delta = float(learning_rate) * float(reward)
            if delta >= 0.0:
                edge.weight = _clamp01(edge.weight + (delta * (1.0 - edge.weight)))
            else:
                edge.weight = _clamp01(edge.weight + (delta * edge.weight))
            updated += 1
            self._record_event(
                "edge_weight_feedback",
                {
                    "mode": "relation_batch",
                    "from": edge.from_node,
                    "to": edge.to_node,
                    "relation_type": edge.relation_type,
                    "old_weight": old_weight,
                    "new_weight": edge.weight,
                    "reward": float(reward),
                    "learning_rate": float(learning_rate),
                },
            )
        return updated

    def persist(self) -> bool:
        if self.graph_adapter is None:
            return False
        self.graph_adapter.persist_snapshot(self.snapshot())
        self._record_event(
            "persist_snapshot",
            {
                "nodes": len(self.store.nodes),
                "edges": len(self.store.edges),
            },
        )
        return True

    def load_from_adapter(self) -> bool:
        if self.graph_adapter is None:
            return False
        snapshot = dict(self.graph_adapter.load_snapshot() or {})
        if not snapshot:
            return False
        self._load_snapshot(snapshot)
        self._record_event(
            "load_snapshot",
            {
                "nodes": len(self.store.nodes),
                "edges": len(self.store.edges),
            },
        )
        return True

    def _load_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        self.store.nodes.clear()
        self.store.edges.clear()
        nodes_raw = dict(snapshot.get("nodes", {}) or {})
        edges_raw = list(snapshot.get("edges", []) or [])
        max_id = 0
        for raw_id, data in nodes_raw.items():
            try:
                node_id = int(raw_id)
            except Exception:
                continue
            if not isinstance(data, Mapping):
                continue
            node_type = str(data.get("type", "generic") or "generic")
            attributes = dict(data.get("attributes", {}) or {})
            state = _to_state_vector(dict(data.get("state", {}) or {}))
            self.create_node(
                node_type,
                node_id=node_id,
                attributes=attributes,
                state=state,
            )
            max_id = max(max_id, node_id)
        for item in edges_raw:
            if not isinstance(item, Mapping):
                continue
            try:
                edge = Edge(
                    from_node=int(item.get("from")),
                    to_node=int(item.get("to")),
                    relation_type=str(item.get("relation_type", "") or "").strip(),
                    weight=_safe_float(item.get("weight", 1.0), 1.0),
                    direction=str(item.get("direction", "directed") or "directed"),
                    logic_rule=str(item.get("logic_rule", "explicit") or "explicit"),
                    metadata=dict(item.get("metadata", {}) or {}),
                )
            except Exception:
                continue
            if not edge.relation_type:
                continue
            try:
                self.add_edge(edge)
            except Exception:
                continue
        self._next_node_id = max(self._next_node_id, max_id + 1)
