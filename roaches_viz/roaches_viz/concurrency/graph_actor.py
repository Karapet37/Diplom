from __future__ import annotations

import copy
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..analysis_loops import detect_loops
from ..controller import GraphMutationController
from ..graph_audit import build_graph_audit
from ..graph_hygiene import build_sanitized_node, sanitize_graph_nodes
from ..graph_model import Edge, Evidence, Node
from ..graph_rag import (
    build_agent_plan,
    build_graph_update_scenario,
    build_ram_graph,
    ensure_system_agents,
    generate_behavioral_dialogue,
    materialize_graph_update_requests,
)
from ..ingest import ingest_text, source_id_from_text
from ..foundations import load_builtin_foundation, load_foundation_from_payload, seed_foundation
from ..rebuild import rebuild_graph_from_sources
from ..store import GraphStore
from ..style_profiles import load_style_profile
from .actor import Actor, ActorMessage


class GraphActor(Actor):
    """Single-writer actor for behavioral knowledge graph mutations."""

    def __init__(self, *, db_path: Path, top_tokens_per_sentence: int = 5):
        super().__init__(name="graph-actor")
        self._store = GraphStore(db_path)
        ensure_system_agents(self._store)
        self.db_path = db_path
        self._top_tokens_per_sentence = top_tokens_per_sentence
        self._controller = GraphMutationController()
        self._snapshot_lock = threading.Lock()
        self._snapshot = self._store.snapshot()
        self._last_hygiene_at = ""
        self._last_hygiene_reason = ""
        self._last_hygiene_updates: dict[str, Any] = {"scanned_nodes": 0, "updated_nodes": 0, "updates": []}
        self._last_hygiene_monotonic = 0.0
        self._graph_jobs_lock = threading.Lock()
        self._graph_jobs: dict[str, dict[str, Any]] = {}

    def _refresh_runtime_graph(self) -> None:
        ensure_system_agents(self._store)
        self._swap_snapshot()

    def handle_message(self, msg: ActorMessage) -> Any:
        if msg.command == "ingest":
            text = str(msg.payload.get("text") or "").strip()
            if not text:
                return {"ok": False, "error": "empty text"}
            source_id = str(msg.payload.get("source_id") or source_id_from_text(text))
            result = ingest_text(self._store, source_id, text, top_tokens_per_sentence=self._top_tokens_per_sentence)
            self._refresh_runtime_graph()
            return {"ok": True, **result}

        if msg.command == "snapshot":
            edge_type = msg.payload.get("edge_type")
            min_weight = float(msg.payload.get("min_weight") or 0.0)
            snapshot = self.read_snapshot()
            edges = [edge for edge in snapshot["edges"] if float(edge.get("weight") or 0.0) >= min_weight]
            if edge_type:
                edges = [edge for edge in edges if edge.get("type") == edge_type]
            nodes = [snapshot["nodes"][node_id] for node_id in sorted(snapshot["nodes"]) if node_id in snapshot["nodes"]]
            return {"ok": True, "nodes": nodes, "edges": edges, "adjacency": snapshot["adjacency"]}

        if msg.command == "search_nodes":
            query = str(msg.payload.get("query") or "").strip()
            limit = int(msg.payload.get("limit") or 20)
            return {"ok": True, "nodes": self._store.search_nodes(query, limit=limit), "query": query}

        if msg.command == "query_subgraph":
            query = str(msg.payload.get("query") or "").strip()
            limit = int(msg.payload.get("limit") or 24)
            hops = max(0, min(3, int(msg.payload.get("hops") or 1)))
            if not query:
                return {"ok": True, "query": query, "nodes": [], "edges": [], "seed_node_ids": []}
            search_hits = self._store.search_nodes(query, limit=limit)
            seed_node_ids = [str(item.get("id") or item.get("node_id") or "") for item in search_hits if str(item.get("id") or item.get("node_id") or "").strip()]
            snapshot = self.read_snapshot()
            adjacency = snapshot["adjacency"]
            selected: set[str] = set(seed_node_ids)
            frontier = set(seed_node_ids)
            for _ in range(hops):
                next_frontier: set[str] = set()
                for node_id in frontier:
                    for neighbor in adjacency.get(node_id, ()):
                        if neighbor not in selected:
                            selected.add(neighbor)
                            next_frontier.add(neighbor)
                frontier = next_frontier
                if not frontier:
                    break
            nodes = [snapshot["nodes"][node_id] for node_id in sorted(selected) if node_id in snapshot["nodes"]]
            edges = [
                edge
                for edge in snapshot["edges"]
                if str(edge.get("src_id") or "") in selected and str(edge.get("dst_id") or "") in selected
            ]
            return {
                "ok": True,
                "query": query,
                "seed_node_ids": seed_node_ids,
                "nodes": nodes,
                "edges": edges,
            }

        if msg.command == "list_domains":
            return {"ok": True, "domains": self._store.list_domains()}

        if msg.command == "list_persons":
            return {"ok": True, "persons": self._store.list_nodes_by_type("PERSON")}

        if msg.command == "list_agents":
            return {"ok": True, "agents": self._store.list_nodes_by_type("AGENT")}

        if msg.command == "list_professions":
            return {"ok": True, "professions": self._store.list_nodes_by_type("PROFESSION")}

        if msg.command == "create_node":
            node_id = str(msg.payload.get("node_id") or "").strip()
            if not node_id:
                return {"ok": False, "error": "node_id is required"}
            if self._store.get_node(node_id):
                return {"ok": False, "error": f"node '{node_id}' already exists"}
            node = Node(
                id=node_id,
                type=str(msg.payload.get("type") or "CONCEPT"),
                name=str(msg.payload.get("name") or msg.payload.get("label") or node_id),
                label=str(msg.payload.get("label") or msg.payload.get("name") or node_id),
                description=str(msg.payload.get("description") or msg.payload.get("short_gloss") or ""),
                short_gloss=str(msg.payload.get("short_gloss") or msg.payload.get("description") or ""),
                plain_explanation=str(msg.payload.get("plain_explanation") or msg.payload.get("what_it_is") or ""),
                what_it_is=str(msg.payload.get("what_it_is") or msg.payload.get("plain_explanation") or ""),
                how_it_works=str(msg.payload.get("how_it_works") or ""),
                how_to_recognize=str(msg.payload.get("how_to_recognize") or ""),
                examples_json=json.dumps(msg.payload.get("examples") if isinstance(msg.payload.get("examples"), list) else [], ensure_ascii=False),
                tags_json=json.dumps(msg.payload.get("tags") if isinstance(msg.payload.get("tags"), list) else [], ensure_ascii=False),
                logic_weight=float(msg.payload.get("logic_weight") if msg.payload.get("logic_weight") is not None else 0.5),
                emotion_weight=float(msg.payload.get("emotion_weight") if msg.payload.get("emotion_weight") is not None else 0.5),
                risk_weight=float(msg.payload.get("risk_weight") if msg.payload.get("risk_weight") is not None else 0.5),
                relevance_weight=float(msg.payload.get("relevance_weight") if msg.payload.get("relevance_weight") is not None else 0.5),
                confidence=float(msg.payload.get("confidence") if msg.payload.get("confidence") is not None else 0.7),
            )
            self._store.upsert_node(node)
            self._store.conn.commit()
            self._swap_snapshot()
            return {"ok": True, "node": self._store.get_node(node_id)}

        if msg.command == "update_node":
            node_id = str(msg.payload.get("node_id") or "").strip()
            current = self._store.get_node(node_id)
            if not current:
                return {"ok": False, "error": f"unknown node_id '{node_id}'"}
            decision = self._controller.validate_node_patch(msg.payload)
            if not decision.ok:
                return {"ok": False, "error": decision.reason, "controller": decision.details}
            examples = msg.payload.get("examples")
            tags = msg.payload.get("tags")
            node = Node(
                id=node_id,
                type=str(msg.payload.get("type") or current.get("type") or "PATTERN"),
                name=str(msg.payload.get("name") or msg.payload.get("label") or current.get("name") or current.get("label") or node_id),
                label=str(msg.payload.get("label") or msg.payload.get("name") or current.get("label") or current.get("name") or node_id),
                description=str(msg.payload.get("description") or msg.payload.get("short_gloss") or current.get("description") or current.get("short_gloss") or ""),
                short_gloss=str(msg.payload.get("short_gloss") or msg.payload.get("description") or current.get("short_gloss") or current.get("description") or ""),
                plain_explanation=str(msg.payload.get("plain_explanation") or msg.payload.get("what_it_is") or current.get("plain_explanation") or current.get("what_it_is") or ""),
                what_it_is=str(msg.payload.get("what_it_is") or msg.payload.get("plain_explanation") or current.get("what_it_is") or current.get("plain_explanation") or ""),
                how_it_works=str(msg.payload.get("how_it_works") or current.get("how_it_works") or ""),
                how_to_recognize=str(msg.payload.get("how_to_recognize") or current.get("how_to_recognize") or ""),
                examples_json=json.dumps(examples if isinstance(examples, list) else current.get("examples") or [], ensure_ascii=False),
                tags_json=json.dumps(tags if isinstance(tags, list) else current.get("tags") or [], ensure_ascii=False),
                speech_patterns_json=json.dumps(msg.payload.get("speech_patterns") if isinstance(msg.payload.get("speech_patterns"), list) else current.get("speech_patterns") or [], ensure_ascii=False),
                behavior_patterns_json=json.dumps(msg.payload.get("behavior_patterns") if isinstance(msg.payload.get("behavior_patterns"), list) else current.get("behavior_patterns") or [], ensure_ascii=False),
                triggers_json=json.dumps(msg.payload.get("triggers") if isinstance(msg.payload.get("triggers"), list) else current.get("triggers") or [], ensure_ascii=False),
                values_json=json.dumps(msg.payload.get("values") if isinstance(msg.payload.get("values"), list) else current.get("values") or [], ensure_ascii=False),
                preferences_json=json.dumps(msg.payload.get("preferences") if isinstance(msg.payload.get("preferences"), list) else current.get("preferences") or [], ensure_ascii=False),
                reaction_logic_json=json.dumps(msg.payload.get("reaction_logic") if isinstance(msg.payload.get("reaction_logic"), list) else current.get("reaction_logic") or [], ensure_ascii=False),
                tolerance_thresholds_json=json.dumps(msg.payload.get("tolerance_thresholds") if isinstance(msg.payload.get("tolerance_thresholds"), dict) else current.get("tolerance_thresholds") or {}, ensure_ascii=False),
                conflict_patterns_json=json.dumps(msg.payload.get("conflict_patterns") if isinstance(msg.payload.get("conflict_patterns"), list) else current.get("conflict_patterns") or [], ensure_ascii=False),
                logic_weight=float(msg.payload.get("logic_weight") if msg.payload.get("logic_weight") is not None else (current.get("importance_vector") or {}).get("logic_weight", current.get("logic_weight", 0.5))),
                emotion_weight=float(msg.payload.get("emotion_weight") if msg.payload.get("emotion_weight") is not None else (current.get("importance_vector") or {}).get("emotion_weight", current.get("emotion_weight", 0.5))),
                risk_weight=float(msg.payload.get("risk_weight") if msg.payload.get("risk_weight") is not None else (current.get("importance_vector") or {}).get("risk_weight", current.get("risk_weight", 0.5))),
                relevance_weight=float(msg.payload.get("relevance_weight") if msg.payload.get("relevance_weight") is not None else (current.get("importance_vector") or {}).get("relevance_weight", current.get("relevance_weight", 0.5))),
                confidence=float(msg.payload.get("confidence") if msg.payload.get("confidence") is not None else current.get("confidence", 0.7)),
            )
            self._store.upsert_node(node)
            self._store.conn.commit()
            self._swap_snapshot()
            return {"ok": True, "node": self._store.get_node(node_id)}

        if msg.command == "update_edge":
            src_id = str(msg.payload.get("src_id") or "").strip()
            dst_id = str(msg.payload.get("dst_id") or "").strip()
            edge_type = str(msg.payload.get("type") or "").strip()
            current = self._store.get_edge(src_id, dst_id, edge_type)
            if not current:
                return {"ok": False, "error": f"unknown edge '{src_id}|{edge_type}|{dst_id}'"}
            edge = Edge(
                src_id=src_id,
                dst_id=dst_id,
                type=edge_type,
                weight=float(msg.payload.get("weight") if msg.payload.get("weight") is not None else current.get("weight", 1.0)),
                confidence=float(msg.payload.get("confidence") if msg.payload.get("confidence") is not None else current.get("confidence", 0.7)),
                metadata_json=json.dumps(current.get("metadata") or {}, ensure_ascii=False, sort_keys=True),
            )
            self._store.replace_edge(edge)
            self._swap_snapshot()
            return {"ok": True, "edge": self._store.get_edge(src_id, dst_id, edge_type)}

        if msg.command == "create_edge":
            src_id = str(msg.payload.get("src_id") or "").strip()
            dst_id = str(msg.payload.get("dst_id") or "").strip()
            edge_type = str(msg.payload.get("type") or "").strip()
            if not src_id or not dst_id or not edge_type:
                return {"ok": False, "error": "src_id, dst_id, and type are required"}
            if not self._store.get_node(src_id) or not self._store.get_node(dst_id):
                return {"ok": False, "error": "source and target nodes must exist"}
            if self._store.get_edge(src_id, dst_id, edge_type):
                return {"ok": False, "error": f"edge '{src_id}|{edge_type}|{dst_id}' already exists"}
            edge = Edge(
                src_id=src_id,
                dst_id=dst_id,
                type=edge_type,
                weight=float(msg.payload.get("weight") if msg.payload.get("weight") is not None else 1.0),
                confidence=float(msg.payload.get("confidence") if msg.payload.get("confidence") is not None else 0.7),
                metadata_json=json.dumps({}, ensure_ascii=False, sort_keys=True),
            )
            self._store.replace_edge(edge)
            self._swap_snapshot()
            return {"ok": True, "edge": self._store.get_edge(src_id, dst_id, edge_type)}

        if msg.command == "health":
            return {
                "ok": True,
                "nodes": len(self._snapshot.nodes),
                "edges": len(self._snapshot.edges),
                "maintenance": {
                    "last_hygiene_at": self._last_hygiene_at,
                    "last_hygiene_reason": self._last_hygiene_reason,
                    "last_hygiene_updates": dict(self._last_hygiene_updates),
                    "graph_jobs": self._graph_job_summary(),
                },
                "controller": {
                    "max_nodes_per_request": self._controller.max_nodes_per_request,
                    "max_edges_per_request": self._controller.max_edges_per_request,
                    "allow_deletes": self._controller.allow_deletes,
                },
            }

        if msg.command == "graph_hygiene_tick":
            reason = str(msg.payload.get("reason") or "manual").strip() or "manual"
            now = time.monotonic()
            if now - self._last_hygiene_monotonic < 8.0:
                return {"ok": True, "skipped": True, "reason": "cooldown", "maintenance": {"last_hygiene_at": self._last_hygiene_at, "last_hygiene_reason": self._last_hygiene_reason, "last_hygiene_updates": dict(self._last_hygiene_updates)}}
            export = self._store.export_graph()
            result = sanitize_graph_nodes(list(export["nodes"]))
            updated_nodes: list[dict[str, Any]] = []
            if result.updated_nodes:
                node_map = {str(node.get("id") or ""): node for node in export["nodes"]}
                for update in result.updates:
                    node = node_map.get(str(update.get("id") or ""))
                    if not node:
                        continue
                    self._store.upsert_node(build_sanitized_node(node, update))
                    updated_nodes.append(update)
                self._store.conn.commit()
                self._swap_snapshot()
            self._last_hygiene_monotonic = now
            self._last_hygiene_at = datetime.now(timezone.utc).isoformat()
            self._last_hygiene_reason = reason
            self._last_hygiene_updates = {"scanned_nodes": result.scanned_nodes, "updated_nodes": len(updated_nodes), "updates": updated_nodes[:20]}
            return {"ok": True, "skipped": False, "reason": reason, "maintenance": {"last_hygiene_at": self._last_hygiene_at, "last_hygiene_reason": self._last_hygiene_reason, "last_hygiene_updates": dict(self._last_hygiene_updates)}}

        if msg.command == "rebuild":
            mode = str(msg.payload.get("mode") or "full").lower()
            source_ids = msg.payload.get("source_ids")
            if not isinstance(source_ids, list):
                source_ids = None
            result = rebuild_graph_from_sources(self._store, mode=mode, source_ids=source_ids, top_tokens_per_sentence=self._top_tokens_per_sentence)
            if result.get("ok"):
                self._refresh_runtime_graph()
            return result

        if msg.command == "analysis_loops":
            snapshot = self.read_snapshot()
            return {"ok": True, **detect_loops({"edges": snapshot["edges"]})}

        if msg.command == "graph_audit":
            snapshot = self.read_snapshot()
            return build_graph_audit({"nodes": list(snapshot["nodes"].values()), "edges": list(snapshot["edges"])})

        if msg.command == "sources":
            rows = self._store.list_sources()
            items = [{"source_id": row["source_id"], "created_at": row["created_at"], "text_length": len(row["raw_text"]), "preview": row["raw_text"][:160]} for row in rows]
            return {"ok": True, "sources": items, "count": len(items)}

        if msg.command == "graph_job_status":
            job_id = str(msg.payload.get("job_id") or "").strip()
            return {"ok": True, "job": self._graph_job_payload(job_id)}

        if msg.command == "chat_graph":
            msg = ActorMessage(command="dialogue_turn", payload=msg.payload, reply=msg.reply)

        if msg.command == "ram_preview":
            message = str(msg.payload.get("message") or "").strip()
            context = str(msg.payload.get("context") or "").strip()
            person_id = str(msg.payload.get("person_id") or msg.payload.get("persona_id") or "").strip() or None
            if not message:
                return {"ok": False, "error": "message is required"}
            snapshot = self._store.snapshot()
            graph_payload = {"nodes": list(snapshot.nodes.values()), "edges": list(snapshot.edges)}
            ram_graph = build_ram_graph(
                graph_payload,
                query="\n\n".join(part for part in [message, context] if part).strip(),
                recent_history=[message, context] if context else [message],
                person_id=person_id,
            )
            agent_plan = build_agent_plan(graph_payload, "\n\n".join(part for part in [message, context] if part).strip(), person_id=person_id)
            return {"ok": True, "ram_graph": ram_graph, "agent_plan": agent_plan}

        if msg.command == "dialogue_turn":
            message = str(msg.payload.get("message") or "").strip()
            context = str(msg.payload.get("context") or "").strip()
            llm_role = str(msg.payload.get("chat_model_role") or msg.payload.get("llm_role") or "general").strip() or "general"
            persona_id = str(msg.payload.get("person_id") or msg.payload.get("persona_id") or "").strip() or None
            user_id = str(msg.payload.get("user_id") or "").strip() or None
            save_to_graph = bool(msg.payload.get("apply_to_graph", msg.payload.get("save_to_graph", True)))
            if not message:
                return {"ok": False, "error": "message is required"}
            before = self._store.snapshot()
            graph_payload = {"nodes": list(before.nodes.values()), "edges": list(before.edges)}
            style_profile = load_style_profile(user_id) if user_id else None
            chat = generate_behavioral_dialogue(
                graph_payload,
                query="\n\n".join(part for part in [message, context] if part).strip(),
                recent_history=[message, context] if context else [message],
                person_id=persona_id,
                llm_role=llm_role,
                user_id=user_id,
                style_profile=style_profile,
            )
            graph_memory_text, graph_guard = self._build_chat_graph_memory_text(message=message, context=context, chat=chat, applied=save_to_graph)
            ingest_result: dict[str, Any] = {"ok": True, "nodes": 0, "edges": 0, "evidence": 0, "source_id": ""}
            scenario = build_graph_update_scenario(
                graph_payload,
                query="\n\n".join(part for part in [message, context] if part).strip(),
                assistant_reply=str(chat.get("assistant_reply") or ""),
                person_id=persona_id,
            )
            controller = self._controller.validate_ai_change(
                nodes_to_add=max(1, len(scenario.get("requests") or [])) * 2,
                edges_to_add=max(1, len(scenario.get("requests") or [])) * 3,
            )
            applied = False
            graph_job = None
            if save_to_graph and scenario.get("should_update") and controller.ok:
                graph_job = self._schedule_graph_job(
                    query=message,
                    context=context,
                    assistant_reply=str(chat.get("assistant_reply") or ""),
                    person_id=persona_id,
                    llm_role=llm_role,
                    requests=list(scenario.get("requests") or []),
                    graph_payload=graph_payload,
                    source_id=str(msg.payload.get("source_id") or ""),
                )
            elif save_to_graph and not scenario.get("should_update"):
                graph_job = {
                    "status": "skipped",
                    "reason": str(scenario.get("reason") or "graph_context_sufficient"),
                    "requests": [],
                    "missing_tokens": list(scenario.get("missing_tokens") or []),
                }
            elif save_to_graph and not controller.ok:
                graph_job = {
                    "status": "blocked",
                    "reason": controller.reason,
                    "requests": list(scenario.get("requests") or []),
                    "missing_tokens": list(scenario.get("missing_tokens") or []),
                }
            after = self._store.snapshot()
            graph_diff = self._build_graph_diff(before, after, attached=applied)
            return {
                "ok": True,
                **chat,
                "ingest_result": ingest_result,
                "graph_binding": {"attached": applied, "source_id": ingest_result.get("source_id") or ""},
                "graph_guard": {**graph_guard, "controller": controller.details, "controller_ok": controller.ok, "controller_reason": controller.reason},
                "graph_diff": graph_diff,
                "archive_updates": self._build_archive_updates(graph_diff),
                "web_context": {"snippets": []},
                "style": dict(chat.get("style") or {}),
                "graph_job": graph_job,
                "graph_scenario": {
                    "requests": list(scenario.get("requests") or []),
                    "missing_tokens": list(scenario.get("missing_tokens") or []),
                    "reason": str(scenario.get("reason") or ""),
                    "should_update": bool(scenario.get("should_update")),
                },
            }

        if msg.command == "apply_dialogue_graph_job":
            job_id = str(msg.payload.get("job_id") or "").strip()
            memory_text = str(msg.payload.get("memory_text") or "").strip()
            source_id = str(msg.payload.get("source_id") or "").strip() or source_id_from_text(memory_text)
            query = str(msg.payload.get("query") or "").strip()
            assistant_reply = str(msg.payload.get("assistant_reply") or "").strip()
            person_id = str(msg.payload.get("person_id") or "").strip() or None
            current_snapshot = self._store.snapshot()
            current_payload = {"nodes": list(current_snapshot.nodes.values()), "edges": list(current_snapshot.edges)}
            scenario = build_graph_update_scenario(
                current_payload,
                query=query,
                assistant_reply=assistant_reply,
                person_id=person_id,
            )
            if not scenario.get("should_update"):
                self._update_graph_job(job_id, status="skipped", reason=str(scenario.get("reason") or "graph_context_sufficient"), graph_diff={"attached": False, "nodes": [], "edges": [], "node_count": 0, "edge_count": 0})
                return {"ok": True, "skipped": True, "job_id": job_id}
            if not memory_text:
                self._update_graph_job(job_id, status="failed", reason="empty_graph_job_material")
                return {"ok": False, "error": "empty graph job material", "job_id": job_id}
            before = current_snapshot
            ingest_result = ingest_text(self._store, source_id, memory_text, top_tokens_per_sentence=self._top_tokens_per_sentence)
            self._refresh_runtime_graph()
            after = self._store.snapshot()
            graph_diff = self._build_graph_diff(before, after, attached=True)
            self._update_graph_job(job_id, status="done", source_id=source_id, ingest_result=ingest_result, graph_diff=graph_diff)
            return {"ok": True, "job_id": job_id, "ingest_result": ingest_result, "graph_diff": graph_diff}

        if msg.command == "apply_materialized_graph":
            memory_text = str(msg.payload.get("memory_text") or "").strip()
            source_id = str(msg.payload.get("source_id") or "").strip() or source_id_from_text(memory_text)
            query = str(msg.payload.get("query") or "").strip()
            assistant_reply = str(msg.payload.get("assistant_reply") or "").strip()
            person_id = str(msg.payload.get("person_id") or "").strip() or None
            current_snapshot = self._store.snapshot()
            current_payload = {"nodes": list(current_snapshot.nodes.values()), "edges": list(current_snapshot.edges)}
            scenario = build_graph_update_scenario(
                current_payload,
                query=query,
                assistant_reply=assistant_reply,
                person_id=person_id,
            )
            if not scenario.get("should_update"):
                return {
                    "ok": True,
                    "skipped": True,
                    "reason": str(scenario.get("reason") or "graph_context_sufficient"),
                    "graph_diff": {"attached": False, "nodes": [], "edges": [], "node_count": 0, "edge_count": 0},
                }
            if not memory_text:
                return {"ok": False, "error": "empty graph job material"}
            before = current_snapshot
            ingest_result = ingest_text(self._store, source_id, memory_text, top_tokens_per_sentence=self._top_tokens_per_sentence)
            self._refresh_runtime_graph()
            after = self._store.snapshot()
            graph_diff = self._build_graph_diff(before, after, attached=True)
            return {"ok": True, "source_id": source_id, "ingest_result": ingest_result, "graph_diff": graph_diff}

        if msg.command == "seed_series":
            dataset_id = str(msg.payload.get("dataset_id") or "").strip()
            replace_graph = bool(msg.payload.get("replace_graph", True))
            result = seed_foundation(self._store, load_builtin_foundation(dataset_id), replace_graph=replace_graph)
            if result.get("ok"):
                self._refresh_runtime_graph()
            return result

        if msg.command == "import_series_payload":
            payload = msg.payload.get("payload")
            if not isinstance(payload, dict):
                return {"ok": False, "error": "payload must be an object"}
            replace_graph = bool(msg.payload.get("replace_graph", True))
            if isinstance(payload.get("nodes"), list) and isinstance(payload.get("edges"), list):
                if not replace_graph:
                    return {"ok": False, "error": "merge import is not supported for direct graph payloads"}
                evidence = payload.get("evidence") if isinstance(payload.get("evidence"), list) else []
                if not evidence:
                    seen: set[str] = set()
                    for edge in payload.get("edges") or []:
                        if not isinstance(edge, dict):
                            continue
                        edge_key = str(edge.get("edge_key") or f"{edge.get('src_id')}|{edge.get('type')}|{edge.get('dst_id')}")
                        for item in edge.get("evidence") or []:
                            if not isinstance(item, dict):
                                continue
                            record = {
                                "edge_key": edge_key,
                                "source_id": str(item.get("source_id") or "import:graph"),
                                "snippet_text": str(item.get("snippet_text") or ""),
                                "offset_start": int(item.get("offset_start") or 0),
                                "offset_end": int(item.get("offset_end") or len(str(item.get("snippet_text") or ""))),
                            }
                            dedupe = f"{record['edge_key']}|{record['source_id']}|{record['snippet_text']}|{record['offset_start']}|{record['offset_end']}"
                            if dedupe in seen:
                                continue
                            seen.add(dedupe)
                            evidence.append(record)
                result = self._store.replace_graph(
                    nodes=list(payload.get("nodes") or []),
                    edges=list(payload.get("edges") or []),
                    evidence=list(evidence),
                )
                self._store.conn.commit()
                self._refresh_runtime_graph()
                return {"ok": True, "seed": str(payload.get("id") or payload.get("dataset_id") or "graph_import"), **result}
            result = seed_foundation(self._store, load_foundation_from_payload(payload), replace_graph=replace_graph)
            if result.get("ok"):
                self._refresh_runtime_graph()
            return result

        raise ValueError(f"Unknown command: {msg.command}")

    def _build_graph_diff(self, before: Any, after: Any, *, attached: bool) -> dict[str, Any]:
        before_nodes = dict(before.nodes)
        after_nodes = dict(after.nodes)
        before_edges = {f"{edge.get('src_id')}|{edge.get('type')}|{edge.get('dst_id')}": edge for edge in before.edges}
        after_edges = {f"{edge.get('src_id')}|{edge.get('type')}|{edge.get('dst_id')}": edge for edge in after.edges}
        new_node_ids = sorted(set(after_nodes) - set(before_nodes))
        new_edge_keys = sorted(set(after_edges) - set(before_edges))
        return {
            "attached": attached,
            "node_count": len(new_node_ids),
            "edge_count": len(new_edge_keys),
            "nodes": [{"id": node_id, "type": str(after_nodes[node_id].get("type") or ""), "label": str(after_nodes[node_id].get("label") or node_id), "short_gloss": str(after_nodes[node_id].get("short_gloss") or ""), "plain_explanation": str(after_nodes[node_id].get("plain_explanation") or "")} for node_id in new_node_ids],
            "edges": [{"src_id": str(after_edges[key].get("src_id") or ""), "dst_id": str(after_edges[key].get("dst_id") or ""), "type": str(after_edges[key].get("type") or ""), "weight": float(after_edges[key].get("weight") or 0.0), "src_label": str(after_nodes.get(str(after_edges[key].get("src_id") or ""), {}).get("label") or after_edges[key].get("src_id") or ""), "dst_label": str(after_nodes.get(str(after_edges[key].get("dst_id") or ""), {}).get("label") or after_edges[key].get("dst_id") or "")} for key in new_edge_keys],
        }

    def _build_archive_updates(self, graph_diff: dict[str, Any]) -> list[dict[str, Any]]:
        updates: list[dict[str, Any]] = []
        for item in graph_diff.get("nodes") or []:
            updates.append({"entity": str(item.get("type") or "node").lower(), "field": str(item.get("label") or item.get("id") or ""), "operation": "upsert", "confidence": 0.92, "reason": f"Added node {item.get('label') or item.get('id')}"})
        for item in graph_diff.get("edges") or []:
            updates.append({"entity": "edge", "field": str(item.get("type") or ""), "operation": "upsert", "confidence": 0.88, "reason": f"Linked {item.get('src_label')} to {item.get('dst_label')}"})
        return updates[:24]

    def _build_chat_graph_memory_text(self, *, message: str, context: str, chat: dict[str, Any], applied: bool) -> tuple[str, dict[str, Any]]:
        ram_graph = dict(chat.get("ram_graph") or {})
        ranked_context = list(ram_graph.get("ranked_context") or [])[:5]
        agent_plan = list(chat.get("agent_plan") or [])[:4]
        micro = dict(ram_graph.get("micro_signals") or {})
        parts: list[str] = [f"User request: {message.strip()[:280]}"]
        if context.strip():
            parts.append(f"Relevant context: {context.strip()[:320]}")
        for item in ranked_context:
            parts.append(f"Context node: {str(item.get('name') or item.get('node_id') or '')[:180]} | {str(item.get('description') or '')[:200]}")
        for category, values in micro.items():
            if not values:
                continue
            label = str(category).replace("_", " ")
            top = values[0]
            parts.append(f"Signal {label}: {str(top.get('label') or top.get('key') or '')[:180]}")
        for item in agent_plan:
            parts.append(f"Agent: {str(item.get('name') or item.get('agent_id') or '')[:120]}")
        assistant_reply = str(chat.get("assistant_reply_en") or chat.get("assistant_reply") or "").strip()
        if assistant_reply:
            parts.append(f"Assistant conclusion: {assistant_reply[:360]}")
        memory_text = "\n".join(item for item in parts if item).strip()[:1200]
        return memory_text, {"mode": "conservative_summary", "max_chars": 1200, "applied": bool(applied), "summary_preview": memory_text[:280]}

    def _swap_snapshot(self) -> None:
        new_snapshot = self._store.snapshot()
        with self._snapshot_lock:
            self._snapshot = new_snapshot

    def _graph_job_summary(self) -> dict[str, Any]:
        with self._graph_jobs_lock:
            jobs = list(self._graph_jobs.values())
        pending = sum(1 for item in jobs if item.get("status") in {"scheduled", "running", "materialized"})
        done = sum(1 for item in jobs if item.get("status") == "done")
        failed = sum(1 for item in jobs if item.get("status") in {"failed", "blocked"})
        return {"pending": pending, "done": done, "failed": failed, "recent": jobs[-6:]}

    def _graph_job_payload(self, job_id: str) -> dict[str, Any] | None:
        if not job_id:
            return None
        with self._graph_jobs_lock:
            job = self._graph_jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    def _update_graph_job(self, job_id: str, **changes: Any) -> None:
        with self._graph_jobs_lock:
            current = dict(self._graph_jobs.get(job_id) or {})
            current.update(changes)
            current["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._graph_jobs[job_id] = current

    def _schedule_graph_job(
        self,
        *,
        query: str,
        context: str,
        assistant_reply: str,
        person_id: str | None,
        llm_role: str,
        requests: list[str],
        graph_payload: dict[str, Any],
        source_id: str,
    ) -> dict[str, Any]:
        job_id = f"graph-job-{int(time.time() * 1000)}"
        created_at = datetime.now(timezone.utc).isoformat()
        base_payload = {
            "job_id": job_id,
            "status": "scheduled",
            "created_at": created_at,
            "updated_at": created_at,
            "query": query,
            "context": context,
            "assistant_reply": assistant_reply,
            "person_id": person_id or "",
            "requests": list(requests or []),
            "source_id": source_id,
            "graph_diff": {"attached": False, "nodes": [], "edges": [], "node_count": 0, "edge_count": 0},
        }
        with self._graph_jobs_lock:
            self._graph_jobs[job_id] = dict(base_payload)

        def _runner() -> None:
            self._update_graph_job(job_id, status="running")
            try:
                materialized = materialize_graph_update_requests(
                    graph_payload,
                    query="\n\n".join(part for part in [query, context] if part).strip(),
                    assistant_reply=assistant_reply,
                    requests=list(requests or []),
                    person_id=person_id,
                    llm_role=llm_role,
                )
                self._update_graph_job(
                    job_id,
                    status="materialized",
                    source_preview=str(materialized.get("source_preview") or ""),
                    requests=list(materialized.get("requests") or []),
                )
                self.submit(
                    "apply_dialogue_graph_job",
                    {
                        "job_id": job_id,
                        "memory_text": str(materialized.get("memory_text") or ""),
                        "source_id": source_id,
                        "query": query,
                        "assistant_reply": assistant_reply,
                        "person_id": person_id or "",
                    },
                )
            except Exception as exc:
                self._update_graph_job(job_id, status="failed", reason=str(exc))

        threading.Thread(target=_runner, name=f"{self.name}-graph-job", daemon=True).start()
        return dict(base_payload)

    def read_snapshot(self) -> dict[str, Any]:
        with self._snapshot_lock:
            return copy.deepcopy({"nodes": self._snapshot.nodes, "edges": self._snapshot.edges, "adjacency": self._snapshot.adjacency})

    def shutdown(self) -> None:
        self.stop()
        self.join(timeout=5.0)
        self._store.close()
