"""Authentication and rate-limiting adapters for FastAPI integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import base64
import hashlib
import hmac
import ipaddress
import json
import logging
import os
from threading import Lock
import time
from typing import Any

try:
    from fastapi.responses import JSONResponse
except Exception:  # pragma: no cover - allows unit tests without FastAPI installed
    class JSONResponse:  # type: ignore[override]
        def __init__(self, *, status_code: int, content: dict[str, Any], headers: dict[str, str] | None = None):
            self.status_code = status_code
            self.content = content
            self.headers = headers or {}

LOGGER = logging.getLogger(__name__)

_WEAK_SECRET_TOKENS: set[str] = {
    "",
    "change-me",
    "changeme",
    "default",
    "secret",
    "password",
    "admin",
    "autograph",
    "123456",
    "qwerty",
}


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "1" if default else "0").strip().lower()
    return value not in {"0", "false", "no", "off", ""}


def _csv_env(name: str, default: str) -> set[str]:
    raw = os.getenv(name, default)
    out: set[str] = set()
    for token in str(raw or "").split(","):
        cleaned = token.strip()
        if cleaned:
            out.add(cleaned)
    return out


@dataclass(frozen=True)
class SecuritySettings:
    auth_enable: bool = False
    auth_protect_write_only: bool = True
    jwt_secret: str = ""
    jwt_issuer: str = "autograph"
    jwt_audience: str = ""
    jwt_exp_minutes: int = 60
    auth_exempt_paths: set[str] = field(default_factory=set)
    trust_proxy_headers: bool = False
    trusted_proxy_ips: set[str] = field(default_factory=set)

    rate_limit_enable: bool = False
    rate_limit_backend: str = "memory"
    rate_limit_per_minute: int = 120
    rate_limit_exempt_paths: set[str] = field(default_factory=set)

    @classmethod
    def from_env(cls) -> "SecuritySettings":
        default_exempt = ",".join(
            [
                "/api/health",
                "/api/modules",
                "/api/status",
                "/api/graph/snapshot",
                "/api/graph/node-types",
                "/api/graph/events",
                "/api/auth/token",
                "/api/privacy/noise/report",
                "/api/client/introspect",
                "/api/project/db/schema",
                "/api/project/model-advisors",
                "/metrics",
            ]
        )
        default_rl_exempt = ",".join(
            [
                "/api/health",
                "/metrics",
            ]
        )

        exp_raw = os.getenv("AUTH_JWT_EXP_MINUTES", "60").strip()
        limit_raw = os.getenv("RATE_LIMIT_PER_MINUTE", "120").strip()
        try:
            exp_minutes = max(1, int(exp_raw))
        except Exception:
            exp_minutes = 60
        try:
            per_min = max(1, int(limit_raw))
        except Exception:
            per_min = 120

        return cls(
            auth_enable=_bool_env("AUTH_ENABLE", False),
            auth_protect_write_only=_bool_env("AUTH_PROTECT_WRITE_ONLY", True),
            jwt_secret=str(os.getenv("AUTH_JWT_SECRET", "")).strip(),
            jwt_issuer=str(os.getenv("AUTH_JWT_ISSUER", "autograph")).strip() or "autograph",
            jwt_audience=str(os.getenv("AUTH_JWT_AUDIENCE", "")).strip(),
            jwt_exp_minutes=exp_minutes,
            auth_exempt_paths=_csv_env("AUTH_EXEMPT_PATHS", default_exempt),
            trust_proxy_headers=_bool_env("TRUST_PROXY_HEADERS", False),
            trusted_proxy_ips=_csv_env("TRUSTED_PROXY_IPS", "127.0.0.1,::1"),
            rate_limit_enable=_bool_env("RATE_LIMIT_ENABLE", False),
            rate_limit_backend=str(os.getenv("RATE_LIMIT_BACKEND", "memory")).strip().lower() or "memory",
            rate_limit_per_minute=per_min,
            rate_limit_exempt_paths=_csv_env("RATE_LIMIT_EXEMPT_PATHS", default_rl_exempt),
        )


class JWTError(ValueError):
    pass


def is_strong_secret(secret: str, *, min_length: int = 24) -> bool:
    raw = str(secret or "").strip()
    if len(raw) < max(8, int(min_length)):
        return False
    token = raw.casefold()
    if token in _WEAK_SECRET_TOKENS:
        return False
    if len(set(raw)) <= 2:
        return False
    return True


def is_strong_password(password: str, *, min_length: int = 12) -> bool:
    raw = str(password or "").strip()
    if len(raw) < max(8, int(min_length)):
        return False
    token = raw.casefold()
    if token in _WEAK_SECRET_TOKENS:
        return False
    if len(set(raw)) <= 2:
        return False
    return True


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(token: str) -> bytes:
    data = token.encode("ascii")
    missing = (4 - (len(data) % 4)) % 4
    data += b"=" * missing
    return base64.urlsafe_b64decode(data)


def create_access_token(
    *,
    subject: str,
    secret: str,
    issuer: str,
    audience: str = "",
    expires_minutes: int = 60,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": str(subject),
        "iss": str(issuer),
        "iat": now,
        "exp": now + (max(1, int(expires_minutes)) * 60),
    }
    if audience:
        payload["aud"] = str(audience)
    if extra_claims:
        payload.update(dict(extra_claims))

    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    signature_b64 = _b64url_encode(signature)
    return f"{header_b64}.{payload_b64}.{signature_b64}"


def decode_access_token(
    token: str,
    *,
    secret: str,
    issuer: str,
    audience: str = "",
) -> dict[str, Any]:
    parts = str(token or "").split(".")
    if len(parts) != 3:
        raise JWTError("invalid token format")

    header_b64, payload_b64, signature_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected_sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    actual_sig = _b64url_decode(signature_b64)
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise JWTError("invalid token signature")

    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception as exc:
        raise JWTError("invalid token payload") from exc

    now = int(time.time())
    exp = int(payload.get("exp", 0) or 0)
    if exp <= now:
        raise JWTError("token is expired")

    if str(payload.get("iss", "")) != str(issuer):
        raise JWTError("invalid issuer")

    if audience and str(payload.get("aud", "")) != str(audience):
        raise JWTError("invalid audience")

    subject = str(payload.get("sub", "")).strip()
    if not subject:
        raise JWTError("missing subject")
    return payload


def _path_match(path: str, patterns: set[str]) -> bool:
    normalized = str(path or "")
    for pattern in patterns:
        rule = str(pattern).strip()
        if not rule:
            continue
        if rule.endswith("*"):
            if normalized.startswith(rule[:-1]):
                return True
        elif normalized == rule:
            return True
    return False


def requires_auth(*, settings: SecuritySettings, method: str, path: str) -> bool:
    safe_methods = {"GET", "HEAD", "OPTIONS"}
    if not settings.auth_enable:
        return False
    if not path.startswith("/api/"):
        return False
    if _path_match(path, settings.auth_exempt_paths):
        return False
    if settings.auth_protect_write_only and method.upper() in safe_methods:
        return False
    return True


def auth_error_response(message: str, *, status_code: int = 401) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


def extract_bearer_token(auth_header: str) -> str:
    header = str(auth_header or "")
    if not header.startswith("Bearer "):
        return ""
    return header[7:].strip()


def verify_request_token(token: str, *, settings: SecuritySettings) -> dict[str, Any]:
    return decode_access_token(
        token,
        secret=settings.jwt_secret,
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
    )


def _coerce_ip(value: str) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    try:
        ipaddress.ip_address(token)
    except Exception:
        return ""
    return token


def extract_client_ip(request, *, settings: SecuritySettings | None = None) -> str:  # type: ignore[no-untyped-def]
    direct_ip = str(getattr(getattr(request, "client", None), "host", "") or "").strip() or "unknown"

    security = settings if settings is not None else SecuritySettings()
    if not security.trust_proxy_headers:
        return direct_ip

    trusted_sources = set(security.trusted_proxy_ips or set())
    if trusted_sources and direct_ip not in trusted_sources:
        return direct_ip

    headers = getattr(request, "headers", {}) or {}
    forwarded = str(headers.get("x-forwarded-for", "") or "").strip()
    if forwarded:
        candidate = _coerce_ip(forwarded.split(",")[0].strip())
        if candidate:
            return candidate

    for header_name in ("x-real-ip", "cf-connecting-ip"):
        candidate = _coerce_ip(str(headers.get(header_name, "") or ""))
        if candidate:
            return candidate
    return direct_ip


def should_rate_limit(*, settings: SecuritySettings, path: str) -> bool:
    if not settings.rate_limit_enable:
        return False
    if not path.startswith("/api/"):
        return False
    if _path_match(path, settings.rate_limit_exempt_paths):
        return False
    return True


class InMemoryRateLimiter:
    """Per-IP minute-window rate limiter used as safe default backend."""

    def __init__(self, per_minute: int):
        self.per_minute = max(1, int(per_minute))
        self._lock = Lock()
        self._history: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            bucket = self._history.setdefault(key, [])
            cutoff = now - 60.0
            bucket[:] = [item for item in bucket if item >= cutoff]
            if len(bucket) >= self.per_minute:
                return False
            bucket.append(now)
            return True


def try_enable_slowapi(app, settings: SecuritySettings) -> bool:
    """Enable slowapi default limits when package is available.

    Not enabled unless RATE_LIMIT_BACKEND=slowapi.
    """
    if not settings.rate_limit_enable or settings.rate_limit_backend != "slowapi":
        return False

    try:
        from slowapi import Limiter
        from slowapi.errors import RateLimitExceeded
        from slowapi.middleware import SlowAPIMiddleware
        from slowapi.util import get_remote_address
    except Exception:
        LOGGER.warning("slowapi is not available; fallback limiter should be used")
        return False

    limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.rate_limit_per_minute}/minute"])
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit_handler(request, exc):  # type: ignore[no-untyped-def]
        return JSONResponse(
            status_code=429,
            content={
                "detail": "rate limit exceeded",
                "limit_per_minute": settings.rate_limit_per_minute,
            },
        )

    app.add_middleware(SlowAPIMiddleware)
    return True
