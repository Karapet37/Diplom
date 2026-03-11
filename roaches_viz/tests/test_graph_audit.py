from __future__ import annotations

from roaches_viz.graph_audit import build_graph_audit


def test_graph_audit_flags_weak_nodes_duplicates_and_orphans() -> None:
    graph = {
        "nodes": [
            {
                "id": "concept:psychology",
                "type": "CONCEPT",
                "name": "Psychology",
                "description": "Psychology",
                "what_it_is": "",
                "how_it_works": "",
                "how_to_recognize": "",
            },
            {
                "id": "concept:psychology_copy",
                "type": "CONCEPT",
                "name": "Psychology",
                "description": "Field of knowledge about behavior.",
                "what_it_is": "A practical field for understanding behavior.",
                "how_it_works": "It groups concepts, patterns, and examples.",
                "how_to_recognize": "Use it for behavior-focused notes.",
            },
            {
                "id": "pattern:guilt_pressure",
                "type": "PATTERN",
                "name": "Guilt pressure",
                "description": "Pressure.",
                "what_it_is": "A coercive move using guilt.",
                "how_it_works": "It pushes compliance through guilt.",
                "how_to_recognize": "Look for affection turned into leverage.",
            },
            {
                "id": "example:1",
                "type": "EXAMPLE",
                "name": "If you loved me, you would do this.",
                "description": "Concrete sentence.",
                "what_it_is": "A concrete example.",
            },
        ],
        "edges": [
            {"src_id": "pattern:guilt_pressure", "dst_id": "example:1", "type": "EXAMPLE_OF", "weight": 1.0, "confidence": 0.9},
            {"src_id": "pattern:guilt_pressure", "dst_id": "pattern:guilt_pressure", "type": "RELATED_TO", "weight": 0.2, "confidence": 0.2},
        ],
    }

    report = build_graph_audit(graph)
    assert report["ok"] is True
    assert report["checks"]["duplicate_candidates"] >= 1
    assert report["checks"]["weak_descriptions"] >= 1
    assert report["checks"]["missing_practical_fields"] >= 1
    assert report["checks"]["orphan_nodes"] >= 1
    assert report["checks"]["low_confidence_edges"] >= 1
    assert report["checks"]["contradictions"] >= 1
