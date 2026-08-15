"""Claude Code session provider (JSONL under ~/.claude/projects)."""
from __future__ import annotations

from .. import sessions as S
from .base import register_provider
from .common import session_payload


class ClaudeProvider:
    id = "claude"
    label = "Claude Code"
    product_name = "Claude Code"

    def list_projects(self) -> list[dict]:
        return S.list_projects()

    def list_sessions(self, slug: str) -> list[dict]:
        return S.list_sessions(slug)

    def session_payload(self, slug: str, sid: str, fold_synthetic: bool = True) -> dict:
        path = S.session_path(slug, sid)
        if not path.is_file():
            raise FileNotFoundError(f"no such session: {sid}")
        doc = S.load_session(path)
        return session_payload(
            self.id, slug, sid,
            hash=S.file_hash(path),
            nbytes=path.stat().st_size,
            turns=S.summarize_session(doc, fold_synthetic),
        )

    def perform_delete(
        self,
        slug: str,
        sid: str,
        turn_ids: list[str],
        expected_hash: str,
        fold_synthetic: bool = True,
    ) -> None:
        S.perform_delete(slug, sid, turn_ids, expected_hash, fold_synthetic=fold_synthetic)

    def perform_delete_session(self, slug: str, sid: str) -> None:
        S.perform_delete_session(slug, sid)

    def perform_undo(self, slug: str, sid: str, expected_hash: str) -> None:
        S.perform_undo(slug, sid, expected_hash)

    def perform_redo(self, slug: str, sid: str, expected_hash: str) -> None:
        S.perform_redo(slug, sid, expected_hash)


register_provider(ClaudeProvider())
