from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .graph_model import Node


_CITATION_RE = re.compile(r"\[\d+\]")
_WHITESPACE_RE = re.compile(r"\s+")
_LABEL_SUFFIX_RE = re.compile(
    r"(?:\s+(?:description|article|wiki|wikipedia|описание|статья|википедия))+$",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class HygieneResult:
    scanned_nodes: int
    updated_nodes: int
    updates: list[dict[str, Any]]


def _clean_text(value: str) -> str:
    text = _CITATION_RE.sub("", str(value or ""))
    text = text.replace("\u00a0", " ").replace("\r", " ").replace("\n", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _trim_sentences(text: str, *, max_sentences: int, max_chars: int) -> str:
    clean = _clean_text(text)
    if not clean:
        return ""
    sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(clean) if item.strip()]
    if not sentences:
        return clean[:max_chars].strip()
    picked = " ".join(sentences[:max_sentences]).strip()
    if len(picked) <= max_chars:
        return picked
    return picked[: max_chars - 1].rstrip(" ,;:-") + "..."


def _clean_label(label: str, plain_explanation: str) -> str:
    clean = _clean_text(label)
    if not clean:
        clean = _trim_sentences(plain_explanation, max_sentences=1, max_chars=72)
    clean = _LABEL_SUFFIX_RE.sub("", clean).strip(" -,:;")
    if len(clean) > 96:
        clean = _trim_sentences(clean, max_sentences=1, max_chars=96)
    return clean


def _decode_json_list(raw: str) -> list[Any]:
    try:
        parsed = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def sanitize_node_payload(node: dict[str, Any]) -> dict[str, Any] | None:
    original_label = str(node.get("label") or "")
    original_short = str(node.get("short_gloss") or "")
    original_plain = str(node.get("plain_explanation") or "")

    cleaned_plain = _trim_sentences(original_plain, max_sentences=3, max_chars=680)
    cleaned_label = _clean_label(original_label, cleaned_plain)
    short_source = original_short
    if not short_source or _LABEL_SUFFIX_RE.search(short_source) or _clean_text(short_source) == _clean_text(original_label):
        short_source = cleaned_plain or cleaned_label
    cleaned_short = _trim_sentences(short_source, max_sentences=1, max_chars=180)
    if not cleaned_plain:
        cleaned_plain = cleaned_short or cleaned_label

    if (
        cleaned_label == original_label.strip()
        and cleaned_short == _clean_text(original_short)
        and cleaned_plain == _clean_text(original_plain)
    ):
        return None

    return {
        "id": str(node.get("id") or ""),
        "before": {
            "label": original_label,
            "short_gloss": original_short,
            "plain_explanation": original_plain,
        },
        "after": {
            "label": cleaned_label or original_label,
            "short_gloss": cleaned_short,
            "plain_explanation": cleaned_plain,
        },
    }


def sanitize_graph_nodes(nodes: list[dict[str, Any]]) -> HygieneResult:
    updates: list[dict[str, Any]] = []
    for node in nodes:
        item = sanitize_node_payload(node)
        if item is not None:
            updates.append(item)
    return HygieneResult(scanned_nodes=len(nodes), updated_nodes=len(updates), updates=updates)


def build_sanitized_node(node: dict[str, Any], update: dict[str, Any]) -> Node:
    return Node(
        id=str(node.get("id") or ""),
        type=str(node.get("type") or "CONCEPT"),
        label=str(update["after"]["label"]),
        short_gloss=str(update["after"]["short_gloss"]),
        plain_explanation=str(update["after"]["plain_explanation"]),
        examples_json=json.dumps(_decode_json_list(str(node.get("examples_json") or "[]")), ensure_ascii=False),
        tags_json=json.dumps(_decode_json_list(str(node.get("tags_json") or "[]")), ensure_ascii=False),
    )
