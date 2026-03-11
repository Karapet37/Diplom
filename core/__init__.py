"""Core-first architecture package for the behavioral graph system."""

from .agent_core import AgentCore, AgentMutationProposal
from .agent_roles import AgentRole, default_agent_roles
from .context_core import ContextCore, ContextQuery, ContextSignals, GraphHydrationPort, GraphSearchPort, RAMContextGraph, SearchHit, SignalExtractorPort
from .dialogue_engine import DialogueContract, DialogueEngine
from .graph_core import GraphEdge, GraphMemory, GraphNode, ImportanceVector, NodeBranches, NodeContext, NodeCore, RankedContextNode
from .graph_initializer import GraphInitializationResult, GraphInitializer
from .graph_traversal import GraphTraversalEngine, TraversalResult
from .personality_core import PersonalityCore, SpeechStyleCore
from .scenario_engine import DialogueScenario, ScenarioEngine
from .speech_dna import SpeechDNA
from .style_engine import StyleEngine, StyleLearningResult
from .system_core import SystemCore

__all__ = [
    "AgentCore",
    "AgentRole",
    "AgentMutationProposal",
    "ContextCore",
    "ContextQuery",
    "ContextSignals",
    "default_agent_roles",
    "DialogueContract",
    "DialogueEngine",
    "DialogueScenario",
    "GraphEdge",
    "GraphHydrationPort",
    "GraphInitializationResult",
    "GraphInitializer",
    "GraphMemory",
    "GraphNode",
    "GraphSearchPort",
    "GraphTraversalEngine",
    "ImportanceVector",
    "NodeBranches",
    "NodeContext",
    "NodeCore",
    "PersonalityCore",
    "RAMContextGraph",
    "RankedContextNode",
    "ScenarioEngine",
    "SearchHit",
    "SignalExtractorPort",
    "SpeechDNA",
    "SpeechStyleCore",
    "StyleEngine",
    "StyleLearningResult",
    "SystemCore",
    "TraversalResult",
]
