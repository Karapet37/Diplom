from __future__ import annotations

import json
from pathlib import Path

from roaches_viz import cli
import roaches_viz.graph_rag as graph_rag
from roaches_viz.config import Settings


def _patch_settings(monkeypatch, db_path: Path) -> None:
    monkeypatch.setattr(cli, "default_settings", lambda _base_dir=None: Settings(db_path=db_path))


def test_cli_load_foundation_prints_seed(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "graph.sqlite3"
    _patch_settings(monkeypatch, db_path)
    monkeypatch.setattr("sys.argv", ["roaches_viz", "load-foundation", "--dataset-id", "psychology_foundations"])

    cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["seed"] == "psychology_foundations"


def test_cli_import_graph_supports_direct_graph_payload(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "graph.sqlite3"
    payload_path = tmp_path / "graph.json"
    payload_path.write_text(
        json.dumps(
            {
                "id": "cli_graph",
                "nodes": [
                    {
                        "id": "domain:cli",
                        "type": "DOMAIN",
                        "label": "CLI domain",
                        "name": "CLI domain",
                        "description": "CLI import root.",
                        "what_it_is": "A domain imported through the CLI.",
                        "how_it_works": "Accepts direct graph JSON.",
                        "how_to_recognize": "Loaded during CLI test.",
                        "examples": ["CLI example"],
                        "tags": ["cli"],
                    },
                    {
                        "id": "pattern:cli_signal",
                        "type": "PATTERN",
                        "label": "CLI pattern",
                        "name": "CLI pattern",
                        "description": "CLI import pattern.",
                        "what_it_is": "A pattern node for CLI import testing.",
                        "how_it_works": "Links to the domain.",
                        "how_to_recognize": "Appears in exported graph after import.",
                        "examples": ["CLI pattern example"],
                        "tags": ["cli", "pattern"],
                    },
                ],
                "edges": [
                    {
                        "src_id": "domain:cli",
                        "dst_id": "pattern:cli_signal",
                        "type": "RELATED_TO",
                        "weight": 1.0,
                        "confidence": 0.8,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _patch_settings(monkeypatch, db_path)
    monkeypatch.setattr("sys.argv", ["roaches_viz", "import-graph", "--file", str(payload_path)])

    cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["seed"] == "cli_graph"
    assert payload["nodes"] == 2


def test_cli_interpret_and_plan_are_behavioral(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "graph.sqlite3"
    _patch_settings(monkeypatch, db_path)

    monkeypatch.setattr(
        "sys.argv",
        ["roaches_viz", "interpret", "--text", "If you loved me you would do this.", "--k", "3", "--mode", "build"],
    )
    cli.main()
    interpreted = json.loads(capsys.readouterr().out)
    assert interpreted["top_hypotheses"]
    assert interpreted["analysis"]

    monkeypatch.setattr(
        "sys.argv",
        [
            "roaches_viz",
            "plan",
            "--goal",
            "understand the pressure pattern and choose a safe next step",
            "--text",
            "If you loved me you would do this.",
            "--depth",
            "3",
            "--beam-width",
            "4",
        ],
    )
    cli.main()
    planned = json.loads(capsys.readouterr().out)
    assert planned["best_line"]
    assert planned["trace"]


def test_cli_dialogue_returns_grounded_reply(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "graph.sqlite3"
    _patch_settings(monkeypatch, db_path)

    monkeypatch.setattr("sys.argv", ["roaches_viz", "load-foundation", "--dataset-id", "human_foundations"])
    cli.main()
    capsys.readouterr()
    monkeypatch.setattr(graph_rag, "build_reasoning_llm", lambda *_args, **_kwargs: None)

    monkeypatch.setattr(
        "sys.argv",
        [
            "roaches_viz",
            "dialogue",
            "--message",
            "What is the difference between action and inaction?",
            "--context",
            "Use the graph and be concise.",
        ],
    )
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["assistant_reply"]
    assert payload["ram_graph"]["ranked_context"]
    assert payload["agent_plan"]


def test_cli_style_learn_and_show(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "graph.sqlite3"
    _patch_settings(monkeypatch, db_path)
    monkeypatch.setenv("STYLE_PROFILES_DIR", str(tmp_path))

    monkeypatch.setattr(
        "sys.argv",
        [
            "roaches_viz",
            "style-learn",
            "--user-id",
            "cli_user",
            "--message",
            "Look, stop hedging and say the thing directly.",
            "--message",
            "Honestly, keep it short and practical.",
        ],
    )
    cli.main()
    learned = json.loads(capsys.readouterr().out)
    assert learned["learned"] is True
    assert learned["profile"]["style_examples"]

    monkeypatch.setattr("sys.argv", ["roaches_viz", "style-show", "--user-id", "cli_user"])
    cli.main()
    shown = json.loads(capsys.readouterr().out)
    assert shown["ok"] is True
    assert shown["profile"]["style_embedding"]
