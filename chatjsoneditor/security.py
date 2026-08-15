"""Request guards for the local-only editor server.

The API has no authentication and can delete session files, which is only safe
while the sole reachable client is the local user's browser tab. Two attacks
break that assumption from any web page the user happens to visit:

* DNS rebinding — attacker.example is made to resolve to 127.0.0.1, so the
  attacker's page talks to the API as a same-origin document. Requests carry
  ``Host: attacker.example``, so rejecting non-loopback Host values blocks it.
* CSRF — a cross-origin page issues POST/DELETE requests to the loopback
  address. Those carry an ``Origin`` that never matches the server, so
  cross-origin requests are rejected.
"""
from __future__ import annotations

from urllib.parse import urlsplit

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse

LOOPBACK_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})

# Set from the CLI when the user deliberately binds a non-loopback address:
# the Host check can no longer tell a rebinding attempt from a legitimate LAN
# name, so it is dropped and only the same-origin check remains.
_any_host_allowed = False


def allow_any_host() -> None:
    global _any_host_allowed
    _any_host_allowed = True


def is_loopback_host(host: str) -> bool:
    hostname = _hostname(host)
    return hostname in LOOPBACK_HOSTNAMES or hostname.startswith("127.")


def _hostname(value: str) -> str:
    """Hostname of a ``host[:port]`` header value, lowercased, sans brackets."""
    host = (value or "").strip().lower()
    if host.startswith("["):  # [::1]:8642
        return host[1:].split("]", 1)[0]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def _host_allowed(value: str) -> bool:
    if not value:  # HTTP/1.0 client without a Host header
        return True
    return _any_host_allowed or is_loopback_host(value)


def _same_origin(origin: str, host: str) -> bool:
    """True if ``origin`` names the same host the request was addressed to."""
    if origin == "null":
        return False
    parts = urlsplit(origin)
    if parts.scheme not in ("http", "https"):
        return False
    return _hostname(parts.netloc) == _hostname(host)


class LocalOriginGuard(BaseHTTPMiddleware):
    """Reject requests that did not originate from the local editor page."""

    async def dispatch(self, request, call_next):
        if not _host_allowed(request.headers.get("host", "")):
            return PlainTextResponse(
                "Forbidden: unexpected Host header (possible DNS rebinding). "
                "Open the editor at http://127.0.0.1:<port>/.",
                status_code=403,
            )
        origin = request.headers.get("origin")
        if origin is not None and not _same_origin(origin, request.headers.get("host", "")):
            return PlainTextResponse(
                "Forbidden: cross-origin request", status_code=403
            )
        return await call_next(request)
