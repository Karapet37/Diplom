from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import default_settings
from .graph_rag import ensure_system_agents, generate_behavioral_dialogue
from .ingest import ingest_text, source_id_from_text
from .interpret.updater import interpret_text
from .foundations import list_builtin_foundations, load_builtin_foundation, load_foundation_from_payload, seed_foundation
from .planner.expectimax import plan_expectimax
from .planner.utility import normalize_state
from .prompt_compose import compose_prompt, list_prompt_modes
from .rebuild import rebuild_graph_from_sources
from .store import GraphStore
from .style_profiles import learn_style_profile, load_style_profile


def _settings():
    return default_settings(Path(__file__).resolve().parents[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavioral Graph-RAG CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    compose_parser = sub.add_parser("compose", help="Compose system prompt blocks")
    compose_parser.add_argument("--mode", default="build", choices=list_prompt_modes())
    compose_parser.add_argument("--task", required=True)

    foundations_parser = sub.add_parser("load-foundation", help="Load a built-in foundation dataset into the graph")
    foundations_parser.add_argument("--dataset-id", default="human_foundations", choices=list_builtin_foundations())
    foundations_parser.add_argument("--merge", action="store_true")

    import_parser = sub.add_parser("import-graph", help="Import a graph dataset JSON payload")
    import_parser.add_argument("--file", required=True)
    import_parser.add_argument("--merge", action="store_true")

    ingest_parser = sub.add_parser("ingest", help="Ingest text into the behavioral graph")
    ingest_parser.add_argument("--source", required=True)
    ingest_parser.add_argument("--source-id", default=None)
    ingest_parser.add_argument("--top-k", type=int, default=5)

    rebuild_parser = sub.add_parser("rebuild", help="Rebuild graph from stored sources")
    rebuild_parser.add_argument("--all", action="store_true")
    rebuild_parser.add_argument("--source-id", action="append", default=[])
    rebuild_parser.add_argument("--top-k", type=int, default=5)

    export_parser = sub.add_parser("export-graph", help="Export graph snapshot to JSON")
    export_parser.add_argument("--output", required=True)

    interpret_parser = sub.add_parser("interpret", help="Extract competing interpretations and micro-signals")
    interpret_parser.add_argument("--text", required=True)
    interpret_parser.add_argument("--k", type=int, default=3)
    interpret_parser.add_argument("--mode", default="build", choices=list_prompt_modes())

    plan_parser = sub.add_parser("plan", help="Run lookahead planning on grounded hypotheses")
    plan_parser.add_argument("--goal", required=True)
    plan_parser.add_argument("--text", default=None)
    plan_parser.add_argument("--depth", type=int, default=3)
    plan_parser.add_argument("--beam-width", type=int, default=4)
    plan_parser.add_argument("--mode", default="build", choices=list_prompt_modes())
    plan_parser.add_argument("--clarity", type=float, default=0.5)
    plan_parser.add_argument("--alignment", type=float, default=0.5)
    plan_parser.add_argument("--progress", type=float, default=0.5)
    plan_parser.add_argument("--rapport", type=float, default=0.5)
    plan_parser.add_argument("--risk", type=float, default=0.5)

    dialogue_parser = sub.add_parser("dialogue", help="Generate a graph-grounded dialogue turn from the current graph")
    dialogue_parser.add_argument("--message", required=True)
    dialogue_parser.add_argument("--context", default="")
    dialogue_parser.add_argument("--person-id", default=None)
    dialogue_parser.add_argument("--llm-role", default="general")
    dialogue_parser.add_argument("--user-id", default="")

    chat_parser = sub.add_parser("chat", help=argparse.SUPPRESS)
    chat_parser.add_argument("--message", required=True)
    chat_parser.add_argument("--context", default="")
    chat_parser.add_argument("--person-id", default=None)
    chat_parser.add_argument("--persona-id", default=None)
    chat_parser.add_argument("--llm-role", default="general")
    chat_parser.add_argument("--user-id", default="")

    style_learn_parser = sub.add_parser("style-learn", help="Learn a user style profile from explicit message samples")
    style_learn_parser.add_argument("--user-id", required=True)
    style_learn_parser.add_argument("--message", action="append", default=[])
    style_learn_parser.add_argument("--messages-file", default="")
    style_learn_parser.add_argument("--max-messages", type=int, default=12)

    style_show_parser = sub.add_parser("style-show", help="Show an existing style profile")
    style_show_parser.add_argument("--user-id", required=True)

    args = parser.parse_args()

    if args.command == "compose":
        settings = _settings()
        print(compose_prompt(mode=args.mode, task=args.task, db_path=settings.db_path))
        return

    settings = _settings()
    store = GraphStore(settings.db_path)
    try:
        ensure_system_agents(store)
        if args.command == "load-foundation":
            result = seed_foundation(store, load_builtin_foundation(args.dataset_id), replace_graph=not bool(args.merge))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        if args.command == "import-graph":
            payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
            if isinstance(payload.get("nodes"), list) and isinstance(payload.get("edges"), list):
                if args.merge:
                    raise SystemExit("--merge is not supported for direct graph JSON imports")
                evidence = payload.get("evidence") if isinstance(payload.get("evidence"), list) else []
                if not evidence:
                    for edge in payload.get("edges") or []:
                        if not isinstance(edge, dict):
                            continue
                        edge_key = str(edge.get("edge_key") or f"{edge.get('src_id')}|{edge.get('type')}|{edge.get('dst_id')}")
                        for item in edge.get("evidence") or []:
                            if not isinstance(item, dict):
                                continue
                            evidence.append({
                                "edge_key": edge_key,
                                "source_id": str(item.get("source_id") or "import:graph"),
                                "snippet_text": str(item.get("snippet_text") or ""),
                                "offset_start": int(item.get("offset_start") or 0),
                                "offset_end": int(item.get("offset_end") or len(str(item.get("snippet_text") or ""))),
                            })
                result = store.replace_graph(nodes=list(payload.get("nodes") or []), edges=list(payload.get("edges") or []), evidence=list(evidence))
                ensure_system_agents(store)
                print(json.dumps({"ok": True, "seed": str(payload.get("id") or payload.get("dataset_id") or "graph_import"), **result}, ensure_ascii=False, indent=2))
                return
            result = seed_foundation(store, load_foundation_from_payload(payload), replace_graph=not bool(args.merge))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        if args.command == "ingest":
            source_path = Path(args.source)
            text = source_path.read_text(encoding="utf-8")
            source_id = args.source_id or source_id_from_text(text)
            result = ingest_text(store, source_id=source_id, text=text, top_tokens_per_sentence=args.top_k)
            print(json.dumps({"ok": True, **result}, ensure_ascii=False, indent=2))
            return

        if args.command == "rebuild":
            if args.all:
                result = rebuild_graph_from_sources(store, mode="full", source_ids=None, top_tokens_per_sentence=args.top_k)
            else:
                result = rebuild_graph_from_sources(store, mode="scoped", source_ids=list(args.source_id or []), top_tokens_per_sentence=args.top_k)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        if args.command == "export-graph":
            output = Path(args.output)
            store.export_json(output)
            payload = store.export_graph()
            result = {"ok": True, "nodes": len(payload["nodes"]), "edges": len(payload["edges"]), "path": str(output)}
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        if args.command == "interpret":
            print(json.dumps(interpret_text(args.text, k=args.k, mode=args.mode), ensure_ascii=False, indent=2))
            return

        if args.command == "plan":
            interpretation = interpret_text(args.text, k=3, mode=args.mode) if args.text else None
            hypotheses = interpretation["top_hypotheses"] if interpretation else None
            result = plan_expectimax(
                normalize_state({
                    "clarity": args.clarity,
                    "alignment": args.alignment,
                    "progress": args.progress,
                    "rapport": args.rapport,
                    "risk": args.risk,
                }),
                args.goal,
                depth=args.depth,
                mode=args.mode,
                hypotheses=hypotheses,
                beam_width=args.beam_width,
            )
            if interpretation is not None:
                result["interpretation"] = {
                    "top_hypotheses": interpretation["top_hypotheses"],
                    "uncertainty": interpretation["uncertainty"],
                    "best_clarifying_question": interpretation["best_clarifying_question"],
                }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        if args.command == "style-learn":
            messages: list[object] = []
            if args.messages_file:
                raw = Path(args.messages_file).read_text(encoding="utf-8").strip()
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = [line.strip() for line in raw.splitlines() if line.strip()]
                if isinstance(parsed, list):
                    messages.extend(parsed)
            messages.extend([{"role": "user", "message": text} for text in list(args.message or []) if str(text).strip()])
            result = learn_style_profile(
                args.user_id,
                messages,
                learn_style_button=True,
                max_messages=args.max_messages,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        if args.command == "style-show":
            print(json.dumps({"ok": True, "user_id": args.user_id, "profile": load_style_profile(args.user_id)}, ensure_ascii=False, indent=2))
            return

        if args.command in {"dialogue", "chat"}:
            payload = store.export_graph()
            person_id = getattr(args, "person_id", None) or getattr(args, "persona_id", None)
            result = generate_behavioral_dialogue(
                payload,
                query="\n\n".join(part for part in [args.message, args.context] if str(part).strip()),
                recent_history=[args.message, args.context] if args.context else [args.message],
                person_id=person_id,
                llm_role=args.llm_role,
                user_id=(getattr(args, "user_id", "") or "").strip() or None,
                style_profile=load_style_profile((getattr(args, "user_id", "") or "").strip()) if (getattr(args, "user_id", "") or "").strip() else None,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
    finally:
        store.close()


if __name__ == "__main__":
    main()
