from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActorMessage:
    command: str
    payload: dict[str, Any] = field(default_factory=dict)
    reply_to: queue.Queue[Any] | None = None


class Actor(threading.Thread):
    def __init__(self, *, name: str, queue_size: int = 2048):
        super().__init__(name=name, daemon=True)
        self._mailbox: queue.Queue[ActorMessage] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()

    def submit(self, command: str, payload: dict[str, Any] | None = None) -> None:
        self._mailbox.put(ActorMessage(command=command, payload=payload or {}))

    def ask(self, command: str, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> Any:
        reply: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._mailbox.put(ActorMessage(command=command, payload=payload or {}, reply_to=reply))
        try:
            result = reply.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"Actor '{self.name}' timed out waiting for command '{command}' after {timeout:.1f}s") from exc
        if isinstance(result, Exception):
            raise result
        return result

    def stop(self) -> None:
        self._stop_event.set()
        self._mailbox.put(ActorMessage(command="__shutdown__"))

    def run(self) -> None:
        while not self._stop_event.is_set():
            msg = self._mailbox.get()
            if msg.command == "__shutdown__":
                if msg.reply_to is not None:
                    msg.reply_to.put({"ok": True})
                break
            try:
                result = self.handle_message(msg)
                if msg.reply_to is not None:
                    msg.reply_to.put(result)
            except Exception as exc:  # pragma: no cover - defensive path
                if msg.reply_to is not None:
                    msg.reply_to.put(exc)

    def handle_message(self, msg: ActorMessage) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError
