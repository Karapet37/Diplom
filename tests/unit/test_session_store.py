from __future__ import annotations

from pathlib import Path

from runtime.sessions.session_store import SessionStore


def test_session_store_creates_and_persists_json(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    created = store.create(title="Test session", tools={"user_id": "u1"})
    assert created["session_id"].startswith("session-")
    assert (tmp_path / "sessions" / f"{created['session_id']}.json").exists()

    saved = store.save(
        {
            "session_id": created["session_id"],
            "title": "Updated title",
            "last_query": "what is guilt pressure",
            "tools": {"user_id": "u1", "extra_context": "family conflict"},
            "messages": [{"role": "user", "message": "hello"}],
        }
    )
    assert saved["title"] == "Updated title"
    assert saved["last_query"] == "what is guilt pressure"
    assert len(saved["messages"]) == 1

    loaded = store.load(created["session_id"])
    assert loaded is not None
    assert loaded["title"] == "Updated title"
    assert loaded["tools"]["extra_context"] == "family conflict"

    listed = store.list()
    assert listed
    assert listed[0]["session_id"] == created["session_id"]
