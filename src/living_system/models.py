"""Typed payloads shared across living-system layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LayerHealth:
    layer: str
    status: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TelemetryEvent:
    component: str
    event_type: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    level: str = "info"
    user_id: str = ""


@dataclass(frozen=True)
class FailureRecord:
    signature: str
    error_type: str
    message: str
    traceback: str
    component: str
    severity: str = "error"


@dataclass(frozen=True)
class RecoveryAction:
    action: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReasoningTrace:
    session_id: str
    input_text: str
    output_text: str
    confidence: float
    trace: dict[str, Any] = field(default_factory=dict)
    user_id: str = ""


@dataclass(frozen=True)
class PromptRun:
    prompt_id: int
    input_payload: dict[str, Any]
    output_text: str
    confidence: float
    user_id: str = ""


@dataclass(frozen=True)
class OperationPolicy:
    workspace_root: str
    allow_create: bool = True
    allow_update: bool = True
    allow_delete: bool = False
    blocked_prefixes: tuple[str, ...] = (".git", "models/gguf/llama.cpp")
    max_file_bytes: int = 1_000_000


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    display_name: str
    primary_language: str = "hy"
    secondary_languages: tuple[str, ...] = ("ru", "en", "pt", "ar", "zh")
    preferences: dict[str, Any] = field(default_factory=dict)
    behavior_model: dict[str, Any] = field(default_factory=dict)
