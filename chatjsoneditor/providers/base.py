"""Provider registry and shared protocol for multi-source session editing."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

_REGISTRY: dict[str, "SessionProvider"] = {}


@runtime_checkable
class SessionProvider(Protocol):
    id: str
    label: str
    product_name: str  # for conflict messages

    def list_projects(self) -> list[dict]: ...
    def list_sessions(self, slug: str) -> list[dict]: ...
    def session_payload(self, slug: str, sid: str, fold_synthetic: bool = True) -> dict: ...
    def perform_delete(
        self,
        slug: str,
        sid: str,
        turn_ids: list[str],
        expected_hash: str,
        fold_synthetic: bool = True,
    ) -> None: ...
    def perform_undo(self, slug: str, sid: str, expected_hash: str) -> None: ...
    def perform_redo(self, slug: str, sid: str, expected_hash: str) -> None: ...


def register_provider(provider: SessionProvider) -> None:
    _REGISTRY[provider.id] = provider


def get_provider(source: str) -> SessionProvider:
    try:
        return _REGISTRY[source]
    except KeyError as e:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"unknown source {source!r}; known: {known}") from e


def list_sources() -> list[dict]:
    return [
        {"id": p.id, "label": p.label, "productName": p.product_name}
        for p in _REGISTRY.values()
    ]
