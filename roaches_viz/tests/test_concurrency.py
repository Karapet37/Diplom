from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from roaches_viz.concurrency.graph_actor import GraphActor
from roaches_viz.concurrency.actor import Actor


class _SlowActor(Actor):
    def __init__(self) -> None:
        super().__init__(name="slow-actor")

    def handle_message(self, msg):
        time.sleep(0.2)
        return {"ok": True}


def test_graph_actor_concurrent_ingest_consistency(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.sqlite3"
    actor = GraphActor(db_path=db_path, top_tokens_per_sentence=4)
    actor.start()
    try:
        texts = [
            "Coffee aroma is floral and sweet.",
            "Coffee roast can be dark or medium.",
            "Sarcasm in chat changes perceived intent.",
            "Ambiguous punctuation may alter tone.",
        ]

        def submit(idx: int) -> dict[str, object]:
            return actor.ask("ingest", {"source_id": f"src:{idx}", "text": texts[idx % len(texts)]})

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(submit, range(20)))

        snap = actor.ask("snapshot", {})
        nodes = {node["id"] for node in snap["nodes"]}
        assert nodes

        edge_keys = set()
        for edge in snap["edges"]:
            assert edge["src_id"] in nodes
            assert edge["dst_id"] in nodes
            key = (edge["src_id"], edge["dst_id"], edge["type"])
            assert key not in edge_keys
            edge_keys.add(key)
    finally:
        actor.shutdown()


def test_graph_actor_rebuild_is_deterministic_and_consistent(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.sqlite3"
    actor = GraphActor(db_path=db_path, top_tokens_per_sentence=4)
    actor.start()
    try:
        actor.ask("ingest", {"source_id": "src:1", "text": "Coffee can be sweet. Coffee can be bitter."})
        actor.ask("ingest", {"source_id": "src:2", "text": "Tone shifts with punctuation. Context matters."})

        first = actor.ask("rebuild", {"mode": "full"})
        assert first["ok"] is True
        snap_a = actor.ask("snapshot", {})

        second = actor.ask("rebuild", {"mode": "full"})
        assert second["ok"] is True
        snap_b = actor.ask("snapshot", {})

        edges_a = sorted((edge["src_id"], edge["dst_id"], edge["type"], edge["weight"]) for edge in snap_a["edges"])
        edges_b = sorted((edge["src_id"], edge["dst_id"], edge["type"], edge["weight"]) for edge in snap_b["edges"])
        assert edges_a == edges_b
    finally:
        actor.shutdown()


def test_graph_actor_snapshot_reads_are_immutable(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.sqlite3"
    actor = GraphActor(db_path=db_path, top_tokens_per_sentence=4)
    actor.start()
    try:
        actor.ask("ingest", {"source_id": "src:1", "text": "Coffee can be sweet. Tone can invert meaning."})
        first = actor.ask("snapshot", {})
        first["nodes"][0]["label"] = "MUTATED"
        first["edges"][0]["weight"] = 999.0
        second = actor.ask("snapshot", {})
        assert second["nodes"][0]["label"] != "MUTATED"
        assert second["edges"][0]["weight"] != 999.0
    finally:
        actor.shutdown()


def test_actor_ask_raises_timeout_error_with_command_name() -> None:
    actor = _SlowActor()
    actor.start()
    try:
        try:
            actor.ask("slow_command", {}, timeout=0.01)
            assert False, "expected timeout"
        except TimeoutError as exc:
            assert "slow_command" in str(exc)
            assert "slow-actor" in str(exc)
    finally:
        actor.stop()
        actor.join(timeout=1.0)


def test_dialogue_turn_returns_fast_and_applies_graph_job_in_background(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "graph.sqlite3"
    actor = GraphActor(db_path=db_path, top_tokens_per_sentence=4)
    actor.start()
    try:
        monkeypatch.setattr(
            "roaches_viz.concurrency.graph_actor.generate_behavioral_dialogue",
            lambda *args, **kwargs: {
                "assistant_reply": "Short fast answer.",
                "assistant_reply_en": "Short fast answer.",
                "ram_graph": {"ranked_context": [], "edges": [], "nodes": [], "micro_signals": {}},
                "context_nodes": [],
                "agent_plan": [],
                "style": {"applied": False},
                "core": {"speech_dna": {}, "scenario": {}, "dialogue_contract": {}},
            },
        )
        monkeypatch.setattr(
            "roaches_viz.concurrency.graph_actor.build_graph_update_scenario",
            lambda *args, **kwargs: {
                "should_update": True,
                "reason": "missing_context_detected",
                "missing_tokens": ["pressure", "pattern"],
                "requests": [
                    "Capture the behavior pattern and connect it to the current dialogue example.",
                    "Add a practical concept node for guilt pressure.",
                ],
            },
        )
        monkeypatch.setattr(
            "roaches_viz.concurrency.graph_actor.materialize_graph_update_requests",
            lambda *args, **kwargs: {
                "requests": [
                    "Capture the behavior pattern and connect it to the current dialogue example.",
                    "Add a practical concept node for guilt pressure.",
                ],
                "memory_text": "Guilt pressure is a behavior pattern. If you loved me you would do this is an example.",
                "source_preview": "Guilt pressure is a behavior pattern.",
            },
        )

        result = actor.ask(
            "dialogue_turn",
            {
                "message": "Is this guilt pressure?",
                "context": "It came right after criticism.",
                "apply_to_graph": True,
                "save_to_graph": True,
                "chat_model_role": "general",
            },
            timeout=5.0,
        )

        assert result["ok"] is True
        assert result["assistant_reply"] == "Short fast answer."
        assert result["graph_job"]["status"] == "scheduled"
        job_id = result["graph_job"]["job_id"]

        status = {}
        for _ in range(30):
            status = actor.ask("graph_job_status", {"job_id": job_id})
            if (status.get("job") or {}).get("status") in {"done", "skipped", "failed"}:
                break
            time.sleep(0.05)

        assert status["job"]["status"] == "done"
        snapshot = actor.ask("snapshot", {})
        assert snapshot["nodes"]
    finally:
        actor.shutdown()
