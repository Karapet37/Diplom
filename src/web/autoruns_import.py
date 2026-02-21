"""Parser helpers for Sysinternals Autoruns/Autorunsc exports."""

from __future__ import annotations

import csv
from io import StringIO
import re
from typing import Any


_HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "entry_name": ("entry", "autorun entry", "name"),
    "entry_location": ("entry location", "location"),
    "enabled": ("enabled",),
    "category": ("category",),
    "profile": ("profile", "user"),
    "description": ("description",),
    "publisher": ("publisher",),
    "image_path": ("image path", "path", "image"),
    "launch_string": ("launch string", "command line", "command"),
    "timestamp_utc": ("timestamp", "time", "utc"),
    "signer": ("signer", "verified signer"),
    "verified": ("verified",),
    "virus_total": ("virustotal", "virus total"),
    "sha1": ("sha-1", "sha1"),
    "md5": ("md5",),
}

_TRUE_TOKENS = {"enabled", "yes", "true", "1", "on"}
_FALSE_TOKENS = {"disabled", "no", "false", "0", "off"}
_VT_RE = re.compile(r"(?P<pos>\d+)\s*/\s*(?P<total>\d+)")


def _normalize_header(value: Any) -> str:
    token = str(value or "").strip().lower()
    token = token.replace("_", " ")
    token = re.sub(r"\s+", " ", token)
    return token


def _guess_delimiter(text: str, explicit: str = "") -> str:
    if explicit in {",", "\t", ";", "|"}:
        return explicit

    sample = str(text or "")[:4000]
    if "\t" in sample and sample.count("\t") >= sample.count(","):
        return "\t"
    if ";" in sample and sample.count(";") > sample.count(","):
        return ";"
    try:
        guessed = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return str(guessed.delimiter or ",")
    except Exception:
        return ","


def _map_headers(header_row: list[str]) -> dict[str, int]:
    normalized = [_normalize_header(item) for item in header_row]
    mapped: dict[str, int] = {}
    for canonical, aliases in _HEADER_ALIASES.items():
        best_idx = -1
        best_len = -1
        for idx, header in enumerate(normalized):
            for alias in aliases:
                token = _normalize_header(alias)
                if not token:
                    continue
                if header == token or token in header:
                    if len(token) > best_len:
                        best_idx = idx
                        best_len = len(token)
        if best_idx >= 0:
            mapped[canonical] = best_idx
    return mapped


def _cell(row: list[str], mapping: dict[str, int], key: str) -> str:
    idx = mapping.get(key, -1)
    if idx < 0 or idx >= len(row):
        return ""
    return str(row[idx] or "").strip()


def _to_enabled(value: str) -> bool | None:
    token = str(value or "").strip().lower()
    if not token:
        return None
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    return None


def _parse_vt_ratio(value: str) -> tuple[int, int]:
    token = str(value or "").strip()
    if not token:
        return (0, 0)
    match = _VT_RE.search(token)
    if not match:
        return (0, 0)
    try:
        positives = int(match.group("pos"))
        total = int(match.group("total"))
    except Exception:
        return (0, 0)
    return (max(0, positives), max(0, total))


def parse_autoruns_text(text: str, *, delimiter: str = "") -> list[dict[str, Any]]:
    """Parse Autorunsc CSV/TSV text into normalized rows."""
    raw = str(text or "").lstrip("\ufeff")
    if not raw.strip():
        return []

    delim = _guess_delimiter(raw, explicit=delimiter)
    reader = csv.reader(StringIO(raw), delimiter=delim)
    rows = [list(row) for row in reader if any(str(cell or "").strip() for cell in row)]
    if len(rows) < 2:
        return []

    header = rows[0]
    mapping = _map_headers(header)
    if "entry_name" not in mapping and "launch_string" not in mapping:
        return []

    out: list[dict[str, Any]] = []
    for row in rows[1:]:
        entry_name = _cell(row, mapping, "entry_name")
        entry_location = _cell(row, mapping, "entry_location")
        launch_string = _cell(row, mapping, "launch_string")
        image_path = _cell(row, mapping, "image_path")
        if not (entry_name or launch_string or image_path):
            continue
        enabled_raw = _cell(row, mapping, "enabled")
        vt_raw = _cell(row, mapping, "virus_total")
        vt_pos, vt_total = _parse_vt_ratio(vt_raw)
        out.append(
            {
                "entry_name": entry_name,
                "entry_location": entry_location,
                "enabled": _to_enabled(enabled_raw),
                "enabled_raw": enabled_raw,
                "category": _cell(row, mapping, "category"),
                "profile": _cell(row, mapping, "profile"),
                "description": _cell(row, mapping, "description"),
                "publisher": _cell(row, mapping, "publisher"),
                "image_path": image_path,
                "launch_string": launch_string,
                "timestamp_utc": _cell(row, mapping, "timestamp_utc"),
                "signer": _cell(row, mapping, "signer"),
                "verified": _cell(row, mapping, "verified"),
                "virus_total": vt_raw,
                "vt_positives": vt_pos,
                "vt_total": vt_total,
                "sha1": _cell(row, mapping, "sha1"),
                "md5": _cell(row, mapping, "md5"),
                "raw_row": list(row),
            }
        )
    return out

