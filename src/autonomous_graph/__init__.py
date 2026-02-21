"""Autonomous logical graph package."""

from src.autonomous_graph.api import GraphAPI, build_graph_engine_from_env, run_demo_simulation
from src.autonomous_graph.core import (
    AutodetectResult,
    Company,
    Edge,
    EmploymentRecord,
    GraphDBAdapter,
    GraphEvent,
    GraphEngine,
    HeuristicTextAnalyzer,
    Human,
    InMemoryGraphStore,
    LLMTextAnalyzer,
    Node,
    TextAnalysis,
    WeightedValue,
)
from src.autonomous_graph.storage import JsonGraphDBAdapter, Neo4jGraphDBAdapter

__all__ = [
    "AutodetectResult",
    "Company",
    "Edge",
    "EmploymentRecord",
    "GraphAPI",
    "build_graph_engine_from_env",
    "GraphDBAdapter",
    "GraphEvent",
    "GraphEngine",
    "HeuristicTextAnalyzer",
    "Human",
    "InMemoryGraphStore",
    "JsonGraphDBAdapter",
    "LLMTextAnalyzer",
    "Neo4jGraphDBAdapter",
    "Node",
    "TextAnalysis",
    "WeightedValue",
    "run_demo_simulation",
]
