"""Client telemetry introspection helpers for semantic graph integration."""

from __future__ import annotations

import ipaddress
import re
import time
from typing import Any, Mapping


_OS_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"windows nt 11", re.IGNORECASE), "Windows 11"),
    (re.compile(r"windows nt 10", re.IGNORECASE), "Windows 10"),
    (re.compile(r"windows nt 6\.[23]", re.IGNORECASE), "Windows 8"),
    (re.compile(r"windows nt 6\.[01]", re.IGNORECASE), "Windows 7"),
    (re.compile(r"android\s+([\d.]+)", re.IGNORECASE), "Android"),
    (re.compile(r"iphone os\s+([\d_]+)", re.IGNORECASE), "iOS"),
    (re.compile(r"ipad; cpu os\s+([\d_]+)", re.IGNORECASE), "iPadOS"),
    (re.compile(r"mac os x\s+([\d_]+)", re.IGNORECASE), "macOS"),
    (re.compile(r"\blinux\b", re.IGNORECASE), "Linux"),
)

_BROWSER_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"edg/([\d.]+)", re.IGNORECASE), "Edge"),
    (re.compile(r"opr/([\d.]+)", re.IGNORECASE), "Opera"),
    (re.compile(r"chrome/([\d.]+)", re.IGNORECASE), "Chrome"),
    (re.compile(r"firefox/([\d.]+)", re.IGNORECASE), "Firefox"),
    (re.compile(r"version/([\d.]+)\s+safari", re.IGNORECASE), "Safari"),
)


def _as_map(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _as_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out
    text = str(value or "").strip()
    return [text] if text else []


def _ip_classification(raw_ip: str) -> dict[str, Any]:
    text = str(raw_ip or "").strip()
    if not text:
        return {"ip": "unknown", "valid": False, "type": "unknown", "private": False}
    try:
        ip = ipaddress.ip_address(text)
    except Exception:
        return {"ip": text, "valid": False, "type": "unknown", "private": False}
    return {
        "ip": text,
        "valid": True,
        "type": "ipv6" if ip.version == 6 else "ipv4",
        "private": bool(ip.is_private),
        "loopback": bool(ip.is_loopback),
        "reserved": bool(ip.is_reserved),
    }


def _detect_os(user_agent: str, platform_hint: str = "") -> str:
    source = f"{user_agent} {platform_hint}".strip()
    for pattern, label in _OS_RULES:
        match = pattern.search(source)
        if not match:
            continue
        version = ""
        if match.lastindex:
            version = str(match.group(match.lastindex) or "").strip().replace("_", ".")
        return f"{label} {version}".strip()
    return "Unknown OS"


def _detect_browser(user_agent: str) -> str:
    source = str(user_agent or "")
    for pattern, label in _BROWSER_RULES:
        match = pattern.search(source)
        if not match:
            continue
        version = str(match.group(1) or "").strip()
        return f"{label} {version}".strip()
    return "Unknown Browser"


def _collect_forward_chain(headers: Mapping[str, Any]) -> list[str]:
    forwarded = str(headers.get("x-forwarded-for", "") or "").strip()
    if not forwarded:
        return []
    out: list[str] = []
    for part in forwarded.split(","):
        token = part.strip()
        if token:
            out.append(token)
    return out


def build_client_profile(
    *,
    request_headers: Mapping[str, Any],
    request_client_ip: str,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build normalized client profile from request + browser payload."""
    body = dict(payload or {})
    client = _as_map(body.get("client"))
    headers = {str(k).lower(): str(v) for k, v in dict(request_headers).items()}

    reported_ip = str(client.get("reported_ip", "") or "").strip()
    ip_info = _ip_classification(reported_ip or request_client_ip)
    forward_chain = _collect_forward_chain(headers)
    via_header = str(headers.get("via", "") or "").strip()
    real_ip_header = str(headers.get("x-real-ip", "") or "").strip()
    cf_ip = str(headers.get("cf-connecting-ip", "") or "").strip()

    suspicion_reasons: list[str] = []
    if len(forward_chain) >= 2:
        suspicion_reasons.append("multiple_forward_hops")
    if via_header:
        suspicion_reasons.append("via_header_present")
    if real_ip_header and real_ip_header != ip_info["ip"]:
        suspicion_reasons.append("x_real_ip_mismatch")
    if cf_ip and cf_ip != ip_info["ip"]:
        suspicion_reasons.append("cf_ip_mismatch")
    if str(headers.get("x-forwarded-proto", "")).strip().lower() == "http":
        suspicion_reasons.append("forwarded_plain_http")

    ua = str(client.get("user_agent", "") or str(headers.get("user-agent", "") or "")).strip()
    platform_hint = str(client.get("platform", "") or "").strip()
    os_name = _detect_os(ua, platform_hint)
    browser_name = _detect_browser(ua)

    language = str(client.get("language", "") or str(headers.get("accept-language", "")).split(",")[0]).strip()
    languages = _as_list(client.get("languages"))
    if not languages and language:
        languages = [language]

    profile = {
        "timestamp": time.time(),
        "session_id": str(body.get("session_id", "") or "").strip(),
        "user_id": str(body.get("user_id", "") or "").strip(),
        "network": {
            "ip": ip_info,
            "forward_chain": forward_chain,
            "vpn_proxy_suspected": bool(suspicion_reasons),
            "vpn_proxy_reasons": suspicion_reasons,
            "x_real_ip": real_ip_header,
            "cf_connecting_ip": cf_ip,
        },
        "device": {
            "os": os_name,
            "browser": browser_name,
            "user_agent": ua,
            "platform": platform_hint,
            "vendor": str(client.get("vendor", "") or "").strip(),
            "hardware_concurrency": int(client.get("hardware_concurrency", 0) or 0),
            "device_memory_gb": float(client.get("device_memory_gb", 0.0) or 0.0),
            "max_touch_points": int(client.get("max_touch_points", 0) or 0),
            "screen": _as_map(client.get("screen")),
            "viewport": _as_map(client.get("viewport")),
        },
        "locale": {
            "language": language,
            "languages": languages,
            "timezone": str(client.get("timezone", "") or "").strip(),
        },
        "browser_signals": {
            "online": bool(client.get("online", True)),
            "cookies_enabled": bool(client.get("cookies_enabled", True)),
            "do_not_track": str(client.get("do_not_track", "") or "").strip(),
            "webdriver": bool(client.get("webdriver", False)),
            "connection": _as_map(client.get("connection")),
        },
        "request_headers": {
            "host": str(headers.get("host", "") or ""),
            "accept": str(headers.get("accept", "") or ""),
            "accept_language": str(headers.get("accept-language", "") or ""),
            "x_forwarded_for": str(headers.get("x-forwarded-for", "") or ""),
            "x_forwarded_proto": str(headers.get("x-forwarded-proto", "") or ""),
            "x_real_ip": real_ip_header,
            "via": via_header,
        },
    }
    return profile
