"""Universal semantic knowledge graph extension layer.

This module parses arbitrary input into atomic concepts, mechanisms and relations,
tracks uncertainty, and produces SQL + Cypher representations before optional apply.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
import time
from typing import Any
import uuid

from src.living_system.embedding import HashEmbeddingService
from src.living_system.knowledge_sql import KnowledgeSQLStore


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё][\w\-/]{1,80}", flags=re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"[\n\r]+|(?<=[.!?;:])\s+")

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "you",
    "are",
    "must",
    "not",
    "all",
    "each",
    "any",
    "without",
    "over",
    "under",
    "about",
    "their",
    "them",
    "they",
    "its",
    "was",
    "were",
    "can",
    "should",
    "will",
    "have",
    "has",
    "had",
    "also",
    "then",
    "than",
    "but",
    "or",
    "of",
    "to",
    "in",
    "on",
    "by",
    "as",
    "at",
}

_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Mathematics": ("algebra", "calculus", "geometry", "logic", "number", "probability", "set", "topology"),
    "Physics": ("energy", "force", "quantum", "thermodynamics", "motion", "particle", "field", "relativity"),
    "Biology": ("cell", "gene", "evolution", "organism", "protein", "ecosystem", "metabolism", "dna"),
    "Computer Science": ("algorithm", "data", "graph", "database", "software", "network", "machine", "programming"),
    "Philosophy": ("ethics", "epistemology", "metaphysics", "ontology", "reason", "truth", "knowledge"),
    "Psychology": ("behavior", "cognition", "emotion", "memory", "personality", "motivation", "attention"),
    "Sociology": ("society", "institution", "culture", "social", "community", "norm", "group"),
    "Theology": ("religion", "faith", "doctrine", "sacred", "divine", "spiritual", "theology"),
    "Economics": ("market", "capital", "demand", "supply", "inflation", "trade", "finance", "utility"),
    "Linguistics": ("language", "syntax", "semantics", "phonology", "morphology", "pragmatics", "grammar"),
}

_FOUNDATIONAL_DOMAIN_CONCEPTS: dict[str, list[dict[str, str]]] = {
    "Mathematics": [
        {"name": "Set Theory", "definition": "Formal language of collections and membership."},
        {"name": "Algebra", "definition": "Symbolic manipulation of structures and equations."},
        {"name": "Calculus", "definition": "Study of change via derivatives and integrals."},
        {"name": "Probability", "definition": "Quantification of uncertainty and random events."},
        {"name": "Logic", "definition": "Rules of valid inference and proof."},
    ],
    "Physics": [
        {"name": "Classical Mechanics", "definition": "Dynamics of bodies under forces."},
        {"name": "Thermodynamics", "definition": "Energy, entropy, and state transformations."},
        {"name": "Electromagnetism", "definition": "Interaction of electric and magnetic fields."},
        {"name": "Quantum Mechanics", "definition": "Behavior of matter and energy at small scales."},
        {"name": "Relativity", "definition": "Spacetime structure and gravitation."},
    ],
    "Biology": [
        {"name": "Cell Theory", "definition": "Cells are the fundamental units of life."},
        {"name": "Genetics", "definition": "Inheritance and variation through genetic material."},
        {"name": "Evolution", "definition": "Population change via selection and adaptation."},
        {"name": "Ecology", "definition": "Interactions between organisms and environment."},
        {"name": "Physiology", "definition": "Functions of living systems and organs."},
    ],
    "Computer Science": [
        {"name": "Algorithms", "definition": "Stepwise procedures for computation."},
        {"name": "Data Structures", "definition": "Organized data representations enabling operations."},
        {"name": "Databases", "definition": "Structured storage, retrieval, and consistency control."},
        {"name": "Networks", "definition": "Communication protocols and distributed exchange."},
        {"name": "Machine Learning", "definition": "Statistical models that improve from data."},
    ],
    "Philosophy": [
        {"name": "Epistemology", "definition": "Study of knowledge and justification."},
        {"name": "Metaphysics", "definition": "Nature of reality and existence."},
        {"name": "Ethics", "definition": "Normative principles of right and wrong."},
        {"name": "Logic", "definition": "Formal structure of reasoning."},
        {"name": "Philosophy of Mind", "definition": "Nature of consciousness and cognition."},
    ],
    "Psychology": [
        {"name": "Cognitive Psychology", "definition": "Mental processes including memory and attention."},
        {"name": "Behavioral Psychology", "definition": "Observable behavior and reinforcement dynamics."},
        {"name": "Developmental Psychology", "definition": "Lifespan changes in behavior and cognition."},
        {"name": "Social Psychology", "definition": "Interpersonal effects on thought and behavior."},
        {"name": "Clinical Psychology", "definition": "Assessment and treatment of mental disorders."},
    ],
    "Sociology": [
        {"name": "Social Structure", "definition": "Patterned social relationships and institutions."},
        {"name": "Institutions", "definition": "Stabilized norms governing collective behavior."},
        {"name": "Stratification", "definition": "Inequality distribution across social classes."},
        {"name": "Culture", "definition": "Shared values, symbols, and practices."},
        {"name": "Collective Behavior", "definition": "Group dynamics in social processes."},
    ],
    "Theology": [
        {"name": "Doctrine", "definition": "Systematic religious teachings."},
        {"name": "Hermeneutics", "definition": "Interpretation of sacred texts."},
        {"name": "Comparative Religion", "definition": "Cross-tradition analysis of beliefs and practices."},
        {"name": "Spiritual Practice", "definition": "Ritual and contemplative methods in faith."},
        {"name": "Theodicy", "definition": "Philosophical treatment of evil and divine goodness."},
    ],
    "Economics": [
        {"name": "Microeconomics", "definition": "Individual and firm decision dynamics."},
        {"name": "Macroeconomics", "definition": "Aggregate output, inflation, and employment."},
        {"name": "Game Theory", "definition": "Strategic interaction among rational agents."},
        {"name": "Market Design", "definition": "Mechanism construction for allocation and pricing."},
        {"name": "Behavioral Economics", "definition": "Economic behavior with cognitive constraints."},
    ],
    "Linguistics": [
        {"name": "Phonology", "definition": "Sound systems in language."},
        {"name": "Morphology", "definition": "Word formation and structure."},
        {"name": "Syntax", "definition": "Rules of sentence structure."},
        {"name": "Semantics", "definition": "Meaning representation in language."},
        {"name": "Pragmatics", "definition": "Context-dependent interpretation of utterances."},
    ],
}

_RELATION_HINTS: tuple[tuple[str, str], ...] = (
    ("causes", "causes"),
    ("because", "causes"),
    ("leads to", "causes"),
    ("depends", "depends_on"),
    ("requires", "depends_on"),
    ("enables", "enables"),
    ("prevents", "prevents"),
    ("contradicts", "contradicts"),
    ("conflicts", "contradicts"),
    ("supports", "supports"),
    ("improves", "improves"),
    ("optimizes", "improves"),
)

_CAUSAL_TYPES = {"causes", "prevents", "enables"}
_DEPENDENCY_TYPES = {"depends_on", "requires"}

_CONTRADICTORY_TYPES: tuple[tuple[str, str], ...] = (
    ("causes", "prevents"),
    ("supports", "contradicts"),
    ("enables", "disables"),
    ("depends_on", "independent_of"),
    ("improves", "degrades"),
)


@dataclass(frozen=True)
class _ConceptCandidate:
    name: str
    evidence: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_token(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).casefold()


def _safe_sql_literal(value: Any) -> str:
    text = str(value or "")
    return text.replace("'", "''")


def _safe_json(value: Any) -> str:
    return _safe_sql_literal(json.dumps(value, ensure_ascii=False, sort_keys=True))


class UniversalKnowledgeAgent:
    """Builds and evolves a semantic knowledge graph with explicit uncertainty."""

    def __init__(self, store: KnowledgeSQLStore, embeddings: HashEmbeddingService):
        self.store = store
        self.embeddings = embeddings

    @staticmethod
    def _extract_candidates(text: str) -> list[_ConceptCandidate]:
        source = str(text or "")
        if not source.strip():
            return []

        rows: list[_ConceptCandidate] = []

        for line in source.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("-", "*", "•")):
                item = stripped[1:].strip(" \t-•*;:,")
                if 2 <= len(item.split()) <= 10:
                    rows.append(_ConceptCandidate(name=item, evidence=stripped))
            if ":" in stripped:
                _, tail = stripped.split(":", 1)
                for chunk in re.split(r"[,;]", tail):
                    item = chunk.strip(" \t-•*;:,")
                    if 2 <= len(item.split()) <= 8:
                        rows.append(_ConceptCandidate(name=item, evidence=stripped))

        words = [token for token in _WORD_RE.findall(source) if len(token) >= 3]
        top_tokens = Counter(_normalize_token(item) for item in words if _normalize_token(item) not in _STOPWORDS)
        for token, _count in top_tokens.most_common(40):
            rows.append(_ConceptCandidate(name=token, evidence="token_frequency"))

        out: list[_ConceptCandidate] = []
        seen: set[str] = set()
        for row in rows:
            normalized = _normalize_token(row.name)
            if not normalized or normalized in _STOPWORDS:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            out.append(_ConceptCandidate(name=" ".join(str(row.name).split()), evidence=row.evidence))
            if len(out) >= 80:
                break
        return out

    @staticmethod
    def _classify_domain(name: str) -> str:
        token = _normalize_token(name)
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in token:
                    return domain
        return "General Systems"

    @staticmethod
    def _classify_type(name: str, domain: str) -> str:
        token = _normalize_token(name)
        if domain in _FOUNDATIONAL_DOMAIN_CONCEPTS:
            if any(keyword in token for keyword in ("theory", "mechanics", "economics", "linguistics", "psychology")):
                return "discipline"
        if any(keyword in token for keyword in ("mechanism", "pipeline", "workflow", "architecture", "system")):
            return "mechanism_component"
        if any(keyword in token for keyword in ("agent", "engine", "model", "graph")):
            return "model"
        return "concept"

    @staticmethod
    def _source_quality(sources: list[dict[str, Any]]) -> tuple[int, float]:
        if not sources:
            return 0, 0.3
        signatures: set[str] = set()
        for row in sources:
            domain = str(row.get("domain", "") or "").strip().lower()
            url = str(row.get("url", "") or "").strip().lower()
            sig = domain or url or str(row)
            signatures.add(sig)
        independent = len(signatures)
        quality = min(1.0, max(0.2, independent / 2.0))
        return independent, quality

    def _concept_confidence(self, name: str, *, source_quality: float) -> float:
        token = _normalize_token(name)
        length = len(token)
        lexical = 0.45
        if length >= 6:
            lexical += 0.1
        if " " in token:
            lexical += 0.1
        if any(ch.isdigit() for ch in token):
            lexical -= 0.08
        return max(0.05, min(0.99, lexical * 0.55 + source_quality * 0.45))

    @staticmethod
    def _relation_type(sentence: str) -> str:
        source = _normalize_token(sentence)
        for hint, relation in _RELATION_HINTS:
            if hint in source:
                return relation
        return "related_to"

    @staticmethod
    def _relation_weight(relation_type: str, conf_a: float, conf_b: float) -> float:
        base = (float(conf_a) + float(conf_b)) / 2.0
        if relation_type in _CAUSAL_TYPES:
            base += 0.12
        if relation_type in _DEPENDENCY_TYPES:
            base += 0.08
        if relation_type == "contradicts":
            base -= 0.06
        return max(0.05, min(0.99, base))

    @staticmethod
    def _relation_trust(weight: float, source_quality: float) -> float:
        value = 0.55 * float(weight) + 0.45 * float(source_quality)
        return max(0.05, min(0.99, value))

    def _build_concepts(
        self,
        *,
        text: str,
        user_id: str,
        sources: list[dict[str, Any]],
        branch_id: str,
    ) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
        candidates = self._extract_candidates(text)
        independent_sources, source_quality = self._source_quality(sources)

        concepts: list[dict[str, Any]] = []
        by_name: dict[str, str] = {}
        uncertainties: list[dict[str, Any]] = []

        now_iso = _now_iso()
        for row in candidates:
            normalized_name = _normalize_token(row.name)
            if not normalized_name:
                continue

            existing = self.store.find_node(user_id=user_id, node_type="concept", display_name=row.name)
            if existing is not None:
                concept_id = str(existing.get("node_id"))
                confidence = max(float(existing.get("confidence", 0.5)), self._concept_confidence(row.name, source_quality=source_quality))
                reused = True
            else:
                concept_id = str(uuid.uuid4())
                confidence = self._concept_confidence(row.name, source_quality=source_quality)
                reused = False

            domain = self._classify_domain(row.name)
            concept_type = self._classify_type(row.name, domain)
            definition = f"Atomic concept extracted from input text: {row.name}."
            embedding = self.embeddings.embed_text(row.name)

            concept_payload = {
                "id": concept_id,
                "name": row.name,
                "domain": domain,
                "type": concept_type,
                "definition": definition,
                "source": {
                    "kind": "user_input",
                    "evidence": row.evidence,
                    "sources": sources,
                    "independent_sources": independent_sources,
                },
                "confidence": round(float(confidence), 4),
                "embedding": embedding,
                "temporal_validity": {
                    "from": now_iso,
                    "to": None,
                },
                "provenance": {
                    "user_id": user_id,
                    "branch_id": branch_id,
                    "reused": reused,
                },
            }
            concepts.append(concept_payload)
            by_name[normalized_name] = concept_id

            if confidence < 0.6:
                uncertainties.append(
                    {
                        "kind": "low_confidence_concept",
                        "concept_id": concept_id,
                        "name": row.name,
                        "confidence": round(float(confidence), 4),
                        "reason": "insufficient evidence or ambiguous extraction",
                    }
                )

        if independent_sources < 2:
            uncertainties.append(
                {
                    "kind": "source_verification",
                    "independent_sources": independent_sources,
                    "required_minimum": 2,
                    "reason": "input did not include >=2 independent sources",
                }
            )

        return concepts, by_name, uncertainties

    def _build_relations(
        self,
        *,
        text: str,
        concepts: list[dict[str, Any]],
        by_name: dict[str, str],
        sources: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        concept_by_id = {str(item["id"]): item for item in concepts}
        independent_sources, source_quality = self._source_quality(sources)

        uncertainties: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        seen_rel: set[tuple[str, str, str]] = set()

        sentences = [chunk.strip() for chunk in _SENTENCE_SPLIT_RE.split(text or "") if chunk.strip()]
        if not sentences:
            sentences = [str(text or "").strip()]

        concept_name_pairs = [(item["id"], _normalize_token(item["name"])) for item in concepts]

        for sentence in sentences:
            sentence_norm = _normalize_token(sentence)
            if not sentence_norm:
                continue

            matched_ids: list[str] = []
            for concept_id, concept_name in concept_name_pairs:
                if concept_name and concept_name in sentence_norm:
                    matched_ids.append(str(concept_id))

            if len(matched_ids) < 2:
                continue

            relation_type = self._relation_type(sentence)
            for index in range(len(matched_ids) - 1):
                source_id = matched_ids[index]
                target_id = matched_ids[index + 1]
                if source_id == target_id:
                    continue
                key = (source_id, target_id, relation_type)
                if key in seen_rel:
                    continue
                seen_rel.add(key)

                source_conf = float(concept_by_id[source_id]["confidence"])
                target_conf = float(concept_by_id[target_id]["confidence"])
                weight = self._relation_weight(relation_type, source_conf, target_conf)
                trust = self._relation_trust(weight, source_quality)

                relation_payload = {
                    "id": str(uuid.uuid4()),
                    "source": source_id,
                    "target": target_id,
                    "type": relation_type,
                    "weight": round(float(weight), 4),
                    "trust": round(float(trust), 4),
                    "confidence": round(float((source_conf + target_conf) / 2.0), 4),
                    "evidence": {
                        "sentence": sentence,
                        "sources": sources,
                        "independent_sources": independent_sources,
                    },
                    "embedding": self.embeddings.embed_text(
                        f"{concept_by_id[source_id]['name']} {relation_type} {concept_by_id[target_id]['name']}"
                    ),
                    "causal": relation_type in _CAUSAL_TYPES,
                    "dependency": relation_type in _DEPENDENCY_TYPES,
                }
                relations.append(relation_payload)

                if trust < 0.6:
                    uncertainties.append(
                        {
                            "kind": "low_trust_relation",
                            "relation_id": relation_payload["id"],
                            "source": source_id,
                            "target": target_id,
                            "relation_type": relation_type,
                            "trust": relation_payload["trust"],
                        }
                    )

        if not relations:
            uncertainties.append(
                {
                    "kind": "relation_sparsity",
                    "reason": "not enough concept interactions were detected",
                }
            )

        return relations, uncertainties

    def _build_mechanisms(
        self,
        *,
        concepts: list[dict[str, Any]],
        relations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        concept_by_id = {str(item["id"]): item for item in concepts}
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for relation in relations:
            key = str(relation.get("type", "related_to") or "related_to")
            buckets[key].append(relation)

        mechanisms: list[dict[str, Any]] = []
        for relation_type, bucket in buckets.items():
            concept_ids: list[str] = []
            for relation in bucket:
                concept_ids.append(str(relation["source"]))
                concept_ids.append(str(relation["target"]))
            deduped_ids: list[str] = []
            seen: set[str] = set()
            for concept_id in concept_ids:
                if concept_id in seen:
                    continue
                seen.add(concept_id)
                deduped_ids.append(concept_id)

            mechanism_confidence = 0.0
            if bucket:
                mechanism_confidence = sum(float(item.get("confidence", 0.0)) for item in bucket) / len(bucket)

            mechanisms.append(
                {
                    "id": str(uuid.uuid4()),
                    "name": f"{relation_type}_mechanism",
                    "relation_type": relation_type,
                    "concepts": deduped_ids,
                    "concept_names": [concept_by_id[item]["name"] for item in deduped_ids if item in concept_by_id],
                    "relations": [str(item["id"]) for item in bucket],
                    "description": f"Mechanism inferred from {len(bucket)} '{relation_type}' interactions.",
                    "confidence": round(float(mechanism_confidence), 4),
                }
            )

        return mechanisms

    @staticmethod
    def _contradiction_pairs() -> dict[str, set[str]]:
        out: dict[str, set[str]] = defaultdict(set)
        for a, b in _CONTRADICTORY_TYPES:
            out[a].add(b)
            out[b].add(a)
        return out

    def _detect_conflicts(self, *, edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        contradictory = self._contradiction_pairs()
        by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
        for edge in edges:
            source = str(edge.get("source") or edge.get("from_node") or edge.get("from") or "")
            target = str(edge.get("target") or edge.get("to_node") or edge.get("to") or "")
            relation_type = str(edge.get("type") or edge.get("relation_type") or "").strip().lower()
            if not source or not target or not relation_type:
                continue
            by_pair[(source, target)].add(relation_type)

        conflicts: list[dict[str, Any]] = []
        for (source, target), rels in by_pair.items():
            for relation in rels:
                opposed = contradictory.get(relation, set())
                overlap = sorted(opposed.intersection(rels))
                if not overlap:
                    continue
                conflicts.append(
                    {
                        "source": source,
                        "target": target,
                        "relation": relation,
                        "conflicts_with": overlap,
                    }
                )
        return conflicts

    def evaluate_graph(self, *, user_id: str = "") -> dict[str, Any]:
        state = self.store.graph_state(user_id=user_id)
        nodes = list(state.get("nodes", []) or [])
        edges = list(state.get("edges", []) or [])

        weak_nodes = [
            {
                "node_id": str(row.get("node_id", "")),
                "name": str(row.get("display_name", "")),
                "confidence": float(row.get("confidence", 0.0)),
            }
            for row in nodes
            if float(row.get("confidence", 0.0)) < 0.55
        ]

        low_conf_edges = [
            {
                "source": str(row.get("from_node", "")),
                "target": str(row.get("to_node", "")),
                "relation_type": str(row.get("relation_type", "")),
                "confidence": float(row.get("confidence", 0.0)),
                "weight": float(row.get("weight", 0.0)),
            }
            for row in edges
            if float(row.get("confidence", 0.0)) < 0.55 or float(row.get("weight", 0.0)) < 0.55
        ]

        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in edges:
            source = str(edge.get("from_node", ""))
            target = str(edge.get("to_node", ""))
            if source and target:
                adjacency[source].add(target)
                adjacency[target].add(source)

        node_ids = [str(row.get("node_id", "")) for row in nodes if str(row.get("node_id", ""))]
        visited: set[str] = set()
        isolated_clusters: list[dict[str, Any]] = []
        for node_id in node_ids:
            if node_id in visited:
                continue
            stack = [node_id]
            component: list[str] = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                for nxt in adjacency.get(current, set()):
                    if nxt not in visited:
                        stack.append(nxt)
            if len(component) <= 1:
                isolated_clusters.append(
                    {
                        "cluster_size": len(component),
                        "nodes": component,
                    }
                )

        conflicts = self._detect_conflicts(edges=edges)

        improvements: list[str] = []
        if weak_nodes:
            improvements.append("Collect additional independent evidence for weak concepts.")
        if low_conf_edges:
            improvements.append("Revalidate low-confidence relations using cross-domain sources.")
        if isolated_clusters:
            improvements.append("Connect isolated clusters by adding causal or dependency mechanisms.")
        if conflicts:
            improvements.append("Maintain conflicting relations as parallel hypotheses with probability tracking.")
        if not improvements:
            improvements.append("Graph quality is stable; continue periodic topology scans.")

        return {
            "weak_nodes": weak_nodes,
            "low_confidence_edges": low_conf_edges,
            "isolated_clusters": isolated_clusters,
            "logical_conflicts": conflicts,
            "improvements": improvements,
            "summary": {
                "nodes": len(nodes),
                "edges": len(edges),
                "weak_nodes": len(weak_nodes),
                "low_confidence_edges": len(low_conf_edges),
                "isolated_clusters": len(isolated_clusters),
                "conflicts": len(conflicts),
            },
        }

    def _simulate_impact(
        self,
        *,
        user_id: str,
        concepts: list[dict[str, Any]],
        relations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        state = self.store.graph_state(user_id=user_id)
        existing_nodes = {str(row.get("node_id", "")) for row in state.get("nodes", [])}
        existing_edges = {
            (
                str(row.get("from_node", "")),
                str(row.get("to_node", "")),
                str(row.get("relation_type", "")),
            )
            for row in state.get("edges", [])
        }

        new_nodes = [row for row in concepts if str(row.get("id", "")) not in existing_nodes]
        new_edges = [
            row
            for row in relations
            if (
                str(row.get("source", "")),
                str(row.get("target", "")),
                str(row.get("type", "")),
            )
            not in existing_edges
        ]

        synthetic_edges = list(state.get("edges", [])) + [
            {
                "from": row.get("source"),
                "to": row.get("target"),
                "relation_type": row.get("type"),
            }
            for row in new_edges
        ]
        predicted_conflicts = self._detect_conflicts(edges=synthetic_edges)

        return {
            "current_nodes": len(existing_nodes),
            "current_edges": len(existing_edges),
            "predicted_nodes": len(existing_nodes) + len(new_nodes),
            "predicted_edges": len(existing_edges) + len(new_edges),
            "new_nodes": len(new_nodes),
            "new_edges": len(new_edges),
            "predicted_conflicts": predicted_conflicts,
        }

    def _build_sql_preview(
        self,
        *,
        user_id: str,
        branch_id: str,
        concepts: list[dict[str, Any]],
        relations: list[dict[str, Any]],
    ) -> list[str]:
        sql_rows: list[str] = []
        for concept in concepts:
            sql_rows.append(
                "INSERT INTO nodes(node_id, user_id, node_type, display_name, confidence, version, metadata_json, created_at, updated_at) "
                f"VALUES('{_safe_sql_literal(concept['id'])}', '{_safe_sql_literal(user_id)}', 'concept', "
                f"'{_safe_sql_literal(concept['name'])}', {float(concept['confidence']):.6f}, 1, "
                f"'{_safe_json({'domain': concept['domain'], 'branch_id': branch_id, 'type': concept['type']})}', "
                "strftime('%s','now'), strftime('%s','now')) "
                "ON CONFLICT(node_id) DO UPDATE SET confidence=excluded.confidence, updated_at=excluded.updated_at;"
            )
            sql_rows.append(
                "INSERT INTO node_properties(node_id, prop_key, prop_value_json, language_code, version, updated_at) "
                f"VALUES('{_safe_sql_literal(concept['id'])}', 'definition', '{_safe_json(concept['definition'])}', 'en', 1, strftime('%s','now'));"
            )

        for relation in relations:
            sql_rows.append(
                "INSERT INTO edges(user_id, from_node, to_node, relation_type, confidence, weight, version, metadata_json, created_at, updated_at) "
                f"VALUES('{_safe_sql_literal(user_id)}', '{_safe_sql_literal(relation['source'])}', '{_safe_sql_literal(relation['target'])}', "
                f"'{_safe_sql_literal(relation['type'])}', {float(relation['confidence']):.6f}, {float(relation['weight']):.6f}, 1, "
                f"'{_safe_json({'trust': relation['trust'], 'branch_id': branch_id, 'evidence': relation['evidence']})}', "
                "strftime('%s','now'), strftime('%s','now'));"
            )

        return sql_rows

    @staticmethod
    def _sanitize_cypher_rel_type(value: str) -> str:
        token = re.sub(r"[^A-Za-z0-9_]", "_", str(value or "related_to").upper())
        token = re.sub(r"_+", "_", token).strip("_")
        return token or "RELATED_TO"

    def _build_cypher_preview(
        self,
        *,
        branch_id: str,
        concepts: list[dict[str, Any]],
        relations: list[dict[str, Any]],
    ) -> list[str]:
        rows: list[str] = []
        for concept in concepts:
            rows.append(
                "MERGE (c:Concept {id: '" + _safe_sql_literal(concept["id"]) + "'}) "
                "SET c.name = '" + _safe_sql_literal(concept["name"]) + "', "
                "c.domain = '" + _safe_sql_literal(concept["domain"]) + "', "
                "c.type = '" + _safe_sql_literal(concept["type"]) + "', "
                "c.confidence = " + f"{float(concept['confidence']):.6f}" + ", "
                "c.branch_id = '" + _safe_sql_literal(branch_id) + "';"
            )
        for relation in relations:
            rel_type = self._sanitize_cypher_rel_type(str(relation["type"]))
            rows.append(
                "MATCH (a:Concept {id: '" + _safe_sql_literal(relation["source"]) + "'}), "
                "(b:Concept {id: '" + _safe_sql_literal(relation["target"]) + "'}) "
                "MERGE (a)-[r:" + rel_type + "]->(b) "
                "SET r.weight = " + f"{float(relation['weight']):.6f}" + ", "
                "r.trust = " + f"{float(relation['trust']):.6f}" + ", "
                "r.confidence = " + f"{float(relation['confidence']):.6f}" + ";"
            )
        return rows

    def _apply_update(
        self,
        *,
        user_id: str,
        branch_id: str,
        concepts: list[dict[str, Any]],
        relations: list[dict[str, Any]],
        mechanisms: list[dict[str, Any]],
    ) -> dict[str, Any]:
        inserted_concepts: list[dict[str, Any]] = []
        inserted_relations: list[dict[str, Any]] = []

        for concept in concepts:
            node_id = self.store.upsert_node(
                {
                    "node_id": concept["id"],
                    "user_id": user_id,
                    "node_type": "concept",
                    "display_name": concept["name"],
                    "confidence": concept["confidence"],
                    "language_code": "en",
                    "properties": {
                        "name": concept["name"],
                        "definition": concept["definition"],
                        "domain": concept["domain"],
                        "type": concept["type"],
                        "source": concept["source"],
                        "temporal_validity": concept["temporal_validity"],
                    },
                    "metadata": {
                        "branch_id": branch_id,
                        "provenance": concept["provenance"],
                    },
                }
            )
            concept["id"] = node_id
            self.store.store_embedding(
                owner_type="node",
                owner_id=node_id,
                vector=list(concept["embedding"]),
                model_name=self.embeddings.model_name,
                version=1,
            )
            inserted_concepts.append(
                {
                    "node_id": node_id,
                    "name": concept["name"],
                    "confidence": concept["confidence"],
                }
            )

        for relation in relations:
            edge_id = self.store.upsert_edge(
                {
                    "user_id": user_id,
                    "from_node": relation["source"],
                    "to_node": relation["target"],
                    "relation_type": relation["type"],
                    "confidence": relation["confidence"],
                    "weight": relation["weight"],
                    "metadata": {
                        "trust": relation["trust"],
                        "evidence": relation["evidence"],
                        "branch_id": branch_id,
                    },
                }
            )
            self.store.store_embedding(
                owner_type="edge",
                owner_id=str(edge_id),
                vector=list(relation["embedding"]),
                model_name=self.embeddings.model_name,
                version=1,
            )
            inserted_relations.append(
                {
                    "edge_id": edge_id,
                    "source": relation["source"],
                    "target": relation["target"],
                    "type": relation["type"],
                    "weight": relation["weight"],
                    "trust": relation["trust"],
                }
            )

        version_id = self.store.record_component_version(
            component="universal_knowledge_update",
            version=str(int(time.time() * 1000)),
            checksum="",
            metadata={
                "user_id": user_id,
                "branch_id": branch_id,
                "concepts": len(concepts),
                "relations": len(relations),
                "mechanisms": len(mechanisms),
            },
        )

        latest_snapshot = self.store.latest_snapshot_id(user_id=user_id)
        snapshot_id = self.store.save_snapshot(
            "knowledge_update",
            self.store.graph_state(user_id=user_id),
            user_id=user_id,
            parent_snapshot_id=latest_snapshot,
        )

        return {
            "new_concepts": inserted_concepts,
            "new_relations": inserted_relations,
            "updated_confidence": [
                {"id": row["node_id"], "confidence": row["confidence"]}
                for row in inserted_concepts
            ],
            "versions": [
                {
                    "version_id": version_id,
                    "snapshot_id": snapshot_id,
                    "component": "universal_knowledge_update",
                }
            ],
        }

    def analyze_input(
        self,
        *,
        text: str,
        user_id: str,
        sources: list[dict[str, Any]] | None = None,
        branch_id: str = "main",
        apply_changes: bool = False,
    ) -> dict[str, Any]:
        source_rows = []
        for row in list(sources or []):
            if isinstance(row, dict):
                source_rows.append(dict(row))
            else:
                source_rows.append({"url": str(row)})

        concepts, by_name, uncertainties_concepts = self._build_concepts(
            text=text,
            user_id=user_id,
            sources=source_rows,
            branch_id=branch_id,
        )
        relations, uncertainties_relations = self._build_relations(
            text=text,
            concepts=concepts,
            by_name=by_name,
            sources=source_rows,
        )
        mechanisms = self._build_mechanisms(concepts=concepts, relations=relations)

        confidence_values = [float(row.get("confidence", 0.0)) for row in concepts + relations]
        overall_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0

        uncertainties = uncertainties_concepts + uncertainties_relations
        simulation = self._simulate_impact(user_id=user_id, concepts=concepts, relations=relations)

        sql_preview = self._build_sql_preview(
            user_id=user_id,
            branch_id=branch_id,
            concepts=concepts,
            relations=relations,
        )
        cypher_preview = self._build_cypher_preview(
            branch_id=branch_id,
            concepts=concepts,
            relations=relations,
        )

        apply_result = {
            "new_concepts": [],
            "new_relations": [],
            "updated_confidence": [],
            "versions": [],
        }
        if apply_changes:
            apply_result = self._apply_update(
                user_id=user_id,
                branch_id=branch_id,
                concepts=concepts,
                relations=relations,
                mechanisms=mechanisms,
            )

        conflicts = self._detect_conflicts(
            edges=[
                {
                    "source": row.get("source"),
                    "target": row.get("target"),
                    "type": row.get("type"),
                }
                for row in relations
            ]
        )

        return {
            "concepts": concepts,
            "mechanisms": mechanisms,
            "relations": relations,
            "causal_links": [row for row in relations if bool(row.get("causal"))],
            "dependencies": [row for row in relations if bool(row.get("dependency"))],
            "uncertainties": uncertainties,
            "confidence": round(float(overall_confidence), 4),
            "simulation": simulation,
            "evaluation_preview": {
                "conflicts": conflicts,
                "requires_confirmation": bool(uncertainties) or overall_confidence < 0.6,
            },
            "sql": sql_preview,
            "cypher": cypher_preview,
            "metadata": {
                "user_id": user_id,
                "branch_id": branch_id,
                "timestamp": _now_iso(),
                "source_count": len(source_rows),
                "independent_sources": self._source_quality(source_rows)[0],
                "apply_changes": bool(apply_changes),
            },
            "new_concepts": apply_result["new_concepts"],
            "new_relations": apply_result["new_relations"],
            "updated_confidence": apply_result["updated_confidence"],
            "versions": apply_result["versions"],
            "notes": "Knowledge parsed with explicit uncertainty and confidence scoring.",
        }

    def initialize_foundational_domains(
        self,
        *,
        user_id: str,
        branch_id: str = "foundation",
        apply_changes: bool = True,
    ) -> dict[str, Any]:
        concepts: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        mechanisms: list[dict[str, Any]] = []

        now_iso = _now_iso()
        domain_roots: dict[str, str] = {}
        for domain, rows in _FOUNDATIONAL_DOMAIN_CONCEPTS.items():
            domain_root_id = str(uuid.uuid4())
            domain_roots[domain] = domain_root_id
            domain_root = {
                "id": domain_root_id,
                "name": domain,
                "domain": domain,
                "type": "domain",
                "definition": f"Foundational domain: {domain}.",
                "source": {
                    "kind": "system_bootstrap",
                    "evidence": "foundational initialization",
                    "sources": [{"name": "bootstrap"}],
                    "independent_sources": 1,
                },
                "confidence": 0.95,
                "embedding": self.embeddings.embed_text(domain),
                "temporal_validity": {"from": now_iso, "to": None},
                "provenance": {"user_id": user_id, "branch_id": branch_id, "reused": False},
            }
            concepts.append(domain_root)

            for item in rows:
                concept_id = str(uuid.uuid4())
                concept = {
                    "id": concept_id,
                    "name": item["name"],
                    "domain": domain,
                    "type": "foundational_concept",
                    "definition": item["definition"],
                    "source": {
                        "kind": "system_bootstrap",
                        "evidence": "foundational initialization",
                        "sources": [{"name": "bootstrap"}],
                        "independent_sources": 1,
                    },
                    "confidence": 0.9,
                    "embedding": self.embeddings.embed_text(item["name"]),
                    "temporal_validity": {"from": now_iso, "to": None},
                    "provenance": {"user_id": user_id, "branch_id": branch_id, "reused": False},
                }
                concepts.append(concept)

                relation = {
                    "id": str(uuid.uuid4()),
                    "source": domain_root_id,
                    "target": concept_id,
                    "type": "contains",
                    "weight": 0.92,
                    "trust": 0.88,
                    "confidence": 0.9,
                    "evidence": {
                        "sentence": f"{domain} contains {item['name']}",
                        "sources": [{"name": "bootstrap"}],
                        "independent_sources": 1,
                    },
                    "embedding": self.embeddings.embed_text(f"{domain} contains {item['name']}"),
                    "causal": False,
                    "dependency": False,
                }
                relations.append(relation)

            mechanisms.append(
                {
                    "id": str(uuid.uuid4()),
                    "name": f"{domain}_foundational_mechanism",
                    "relation_type": "contains",
                    "concepts": [domain_root_id],
                    "concept_names": [domain],
                    "relations": [],
                    "description": f"Foundational organization mechanism for {domain}.",
                    "confidence": 0.9,
                }
            )

        cross_domain_edges = [
            ("Mathematics", "Physics", "supports"),
            ("Mathematics", "Computer Science", "supports"),
            ("Biology", "Psychology", "influences"),
            ("Sociology", "Economics", "influences"),
            ("Philosophy", "Theology", "contradicts"),
            ("Linguistics", "Psychology", "depends_on"),
        ]
        for source_domain, target_domain, relation_type in cross_domain_edges:
            source_id = domain_roots.get(source_domain)
            target_id = domain_roots.get(target_domain)
            if not source_id or not target_id:
                continue
            relations.append(
                {
                    "id": str(uuid.uuid4()),
                    "source": source_id,
                    "target": target_id,
                    "type": relation_type,
                    "weight": 0.78,
                    "trust": 0.74,
                    "confidence": 0.8,
                    "evidence": {
                        "sentence": f"{source_domain} {relation_type} {target_domain}",
                        "sources": [{"name": "bootstrap"}],
                        "independent_sources": 1,
                    },
                    "embedding": self.embeddings.embed_text(f"{source_domain} {relation_type} {target_domain}"),
                    "causal": relation_type in _CAUSAL_TYPES,
                    "dependency": relation_type in _DEPENDENCY_TYPES,
                }
            )

        apply_result = {
            "new_concepts": [],
            "new_relations": [],
            "updated_confidence": [],
            "versions": [],
        }
        if apply_changes:
            apply_result = self._apply_update(
                user_id=user_id,
                branch_id=branch_id,
                concepts=concepts,
                relations=relations,
                mechanisms=mechanisms,
            )

        return {
            "concepts": concepts,
            "mechanisms": mechanisms,
            "relations": relations,
            "confidence": 0.9,
            "uncertainties": [
                {
                    "kind": "bootstrap_source_count",
                    "independent_sources": 1,
                    "note": "Foundational initialization uses internal curated seed set.",
                }
            ],
            "sql": self._build_sql_preview(
                user_id=user_id,
                branch_id=branch_id,
                concepts=concepts,
                relations=relations,
            ),
            "cypher": self._build_cypher_preview(
                branch_id=branch_id,
                concepts=concepts,
                relations=relations,
            ),
            "metadata": {
                "timestamp": _now_iso(),
                "user_id": user_id,
                "branch_id": branch_id,
                "domain_count": len(_FOUNDATIONAL_DOMAIN_CONCEPTS),
                "apply_changes": bool(apply_changes),
            },
            "new_concepts": apply_result["new_concepts"],
            "new_relations": apply_result["new_relations"],
            "updated_confidence": apply_result["updated_confidence"],
            "versions": apply_result["versions"],
            "notes": "Foundational domains initialized with core concepts and cross-domain relations.",
        }

    def branch_graph(self, *, user_id: str, branch_name: str) -> dict[str, Any]:
        latest = self.store.latest_snapshot_id(user_id=user_id)
        snapshot_id = self.store.save_snapshot(
            snapshot_type=f"branch:{branch_name}",
            state=self.store.graph_state(user_id=user_id),
            user_id=user_id,
            parent_snapshot_id=latest,
        )
        return {
            "branch_name": branch_name,
            "snapshot_id": snapshot_id,
            "parent_snapshot_id": latest,
            "timestamp": _now_iso(),
        }

    def merge_branches(
        self,
        *,
        user_id: str,
        base_snapshot_id: int,
        target_snapshot_id: int,
        apply_changes: bool = False,
    ) -> dict[str, Any]:
        base = self.store.load_snapshot(base_snapshot_id)
        target = self.store.load_snapshot(target_snapshot_id)
        if not isinstance(base, dict) or not isinstance(target, dict):
            raise ValueError("both snapshots must exist")

        base_nodes = {str(row.get("node_id", "")): row for row in list(base.get("nodes", []) or []) if str(row.get("node_id", ""))}
        target_nodes = {str(row.get("node_id", "")): row for row in list(target.get("nodes", []) or []) if str(row.get("node_id", ""))}
        base_edges = {
            (
                str(row.get("from_node", "")),
                str(row.get("to_node", "")),
                str(row.get("relation_type", "")),
            ): row
            for row in list(base.get("edges", []) or [])
            if str(row.get("from_node", "")) and str(row.get("to_node", ""))
        }
        target_edges = {
            (
                str(row.get("from_node", "")),
                str(row.get("to_node", "")),
                str(row.get("relation_type", "")),
            ): row
            for row in list(target.get("edges", []) or [])
            if str(row.get("from_node", "")) and str(row.get("to_node", ""))
        }

        nodes_to_add = [row for node_id, row in target_nodes.items() if node_id not in base_nodes]
        edges_to_add = [row for key, row in target_edges.items() if key not in base_edges]

        node_conflicts: list[dict[str, Any]] = []
        for node_id, target_node in target_nodes.items():
            if node_id not in base_nodes:
                continue
            base_conf = float(base_nodes[node_id].get("confidence", 0.0))
            target_conf = float(target_node.get("confidence", 0.0))
            if abs(base_conf - target_conf) > 0.2:
                node_conflicts.append(
                    {
                        "node_id": node_id,
                        "base_confidence": base_conf,
                        "target_confidence": target_conf,
                    }
                )

        edge_conflicts = self._detect_conflicts(
            edges=[
                {
                    "source": key[0],
                    "target": key[1],
                    "type": key[2],
                }
                for key in set(base_edges).union(set(target_edges))
            ]
        )

        apply_result: dict[str, Any] = {"applied": False, "version_id": None, "snapshot_id": None}
        if apply_changes:
            for row in nodes_to_add:
                self.store.upsert_node(
                    {
                        "node_id": row.get("node_id"),
                        "user_id": user_id,
                        "node_type": row.get("node_type", "concept"),
                        "display_name": row.get("display_name", ""),
                        "confidence": row.get("confidence", 0.5),
                        "properties": dict(row.get("properties", {}) or {}),
                        "metadata": dict(row.get("metadata", {}) or {}),
                    }
                )
            for row in edges_to_add:
                self.store.upsert_edge(
                    {
                        "user_id": user_id,
                        "from_node": row.get("from_node"),
                        "to_node": row.get("to_node"),
                        "relation_type": row.get("relation_type", "related_to"),
                        "confidence": row.get("confidence", 0.5),
                        "weight": row.get("weight", 0.5),
                        "metadata": dict(row.get("metadata", {}) or {}),
                    }
                )

            version_id = self.store.record_component_version(
                component="knowledge_branch_merge",
                version=str(int(time.time() * 1000)),
                metadata={
                    "base_snapshot_id": base_snapshot_id,
                    "target_snapshot_id": target_snapshot_id,
                    "nodes_added": len(nodes_to_add),
                    "edges_added": len(edges_to_add),
                },
            )
            latest = self.store.latest_snapshot_id(user_id=user_id)
            snapshot_id = self.store.save_snapshot(
                snapshot_type="merge_result",
                state=self.store.graph_state(user_id=user_id),
                user_id=user_id,
                parent_snapshot_id=latest,
            )
            apply_result = {
                "applied": True,
                "version_id": version_id,
                "snapshot_id": snapshot_id,
            }

        return {
            "base_snapshot_id": int(base_snapshot_id),
            "target_snapshot_id": int(target_snapshot_id),
            "nodes_to_add": nodes_to_add,
            "edges_to_add": edges_to_add,
            "node_conflicts": node_conflicts,
            "edge_conflicts": edge_conflicts,
            "simulation": {
                "base_nodes": len(base_nodes),
                "base_edges": len(base_edges),
                "predicted_nodes": len(base_nodes) + len(nodes_to_add),
                "predicted_edges": len(base_edges) + len(edges_to_add),
                "conflicts": len(node_conflicts) + len(edge_conflicts),
            },
            "apply": apply_result,
            "notes": "Merge keeps conflicting models explicit and does not collapse contradictions.",
        }
