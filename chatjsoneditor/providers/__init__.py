"""Session source providers (Claude, Grok, Gemini, Antigravity)."""
from __future__ import annotations

from .base import SessionProvider, get_provider, list_sources, register_provider

# Import order is registration order (explorer top-to-bottom).
from . import claude as _claude  # noqa: F401
from . import grok as _grok  # noqa: F401
from . import gemini as _gemini  # noqa: F401
from . import antigravity as _antigravity  # noqa: F401

__all__ = [
    "SessionProvider",
    "get_provider",
    "list_sources",
    "register_provider",
]
