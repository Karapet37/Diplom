from __future__ import annotations

import os
import subprocess
from typing import Callable


def build_ollama_llm_fn(model_name: str) -> Callable[[str], str] | None:
    model = str(model_name or "").strip()
    if not model:
        return None

    timeout_seconds = max(5, min(300, int(str(os.getenv("OLLAMA_TIMEOUT_SECONDS", "90") or "90"))))

    def _run(prompt: str) -> str:
        completed = subprocess.run(
            ["ollama", "run", model, str(prompt)],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return str(completed.stdout or "").strip()

    return _run
