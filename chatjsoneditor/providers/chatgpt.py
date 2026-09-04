"""ChatGPT/Codex rollout session provider.

Codex stores local conversation rollouts as JSONL under ``~/.codex/sessions``.
Entries between two human messages belong to the first message's turn.  We keep
the original line text for every retained record (including malformed JSON), so
editing a turn never reserializes or silently discards the rest of a rollout.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .. import sessions as S
from .base import register_provider
from .common import NormMsg, NormTurn, cap_tool, text_from_content

log = logging.getLogger(__name__)


def _root() -> Path:
    return S.chatgpt_sessions_root()


def _iter_sessions() -> list[tuple[str, Path]]:
    root = _root()
    if not root.is_dir():
        return []
    out = []
    for path in root.glob("*/*/*/*.jsonl"):
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        # The date is both an understandable project grouping and a safely
        # reconstructable path component for API requests.
        out.append(("-".join(rel.parts[:3]), path))
    return out


def _session_path(slug: str, sid: str) -> Path:
    slug, sid = S.safe_name(slug), S.safe_name(sid)
    parts = slug.split("-")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"invalid ChatGPT session date: {slug!r}")
    path = _root().joinpath(*parts, sid)
    if not path.is_file():
        raise FileNotFoundError(f"no such session: {sid}")
    try:
        path.resolve().relative_to(_root().resolve())
    except ValueError as e:
        raise ValueError("path escape") from e
    return path


def _payload(entry: S.Entry) -> dict | None:
    data = entry.data
    payload = data.get("payload") if data else None
    return payload if isinstance(payload, dict) else None


def _is_user(entry: S.Entry) -> bool:
    payload = _payload(entry)
    return bool(
        entry.type == "response_item"
        and payload
        and payload.get("type") == "message"
        and payload.get("role") == "user"
    )


def _content_text(content) -> str:
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in ("input_text", "output_text", "text"):
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(text_from_content(item))
            else:
                parts.append(text_from_content(item))
        return "\n".join(part for part in parts if part)
    return text_from_content(content)


def _entry_messages(entry: S.Entry) -> list[NormMsg]:
    payload = _payload(entry)
    if not payload or entry.type != "response_item":
        return []
    kind = payload.get("type")
    if kind == "message":
        text = _content_text(payload.get("content"))
        role = payload.get("role")
        if role == "user":
            return [NormMsg(kind="user", text=text, timestamp=(entry.data or {}).get("timestamp"))]
        if role == "assistant":
            return [NormMsg(kind="assistant", text=text)] if text else []
        if role == "developer":
            return [NormMsg(kind="system", text=text, synthetic=True)] if text else []
    if kind in ("function_call", "custom_tool_call"):
        args = payload.get("arguments", payload.get("input", ""))
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False, indent=2)
        return [NormMsg(
            kind="tool_use", id=payload.get("call_id") or payload.get("id"),
            name=payload.get("name") or payload.get("namespace") or "?", input=cap_tool(args),
        )]
    if kind in ("function_call_output", "custom_tool_call_output"):
        output = payload.get("output", "")
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False)
        return [NormMsg(kind="tool_result", tool_use_id=payload.get("call_id"), text=cap_tool(output))]
    if kind == "reasoning":
        text = _content_text(payload.get("summary") or payload.get("content"))
        return [NormMsg(kind="thinking", text=text)] if text else []
    return []


def _turn_ranges(doc: S.SessionDoc) -> dict[str, tuple[int, int]]:
    starts = [i for i, entry in enumerate(doc.entries) if _is_user(entry)]
    ranges = {}
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(doc.entries)
        payload = _payload(doc.entries[start]) or {}
        ranges[str(payload.get("id") or f"c{start}")] = (start, end)
    return ranges


def _doc_text(doc: S.SessionDoc) -> str:
    """Render retained raw lines without inventing a dangling CR on CRLF input."""
    text = "\n".join(entry.raw for entry in doc.entries)
    if doc.trailing_newline and text:
        return text + "\n"
    # ``load_session`` retains the CR as part of each pre-final line.  If the
    # last original line was deleted, its predecessor becomes final and must
    # lose that former line terminator.
    return text[:-1] if text.endswith("\r") else text


def _group_turns(doc: S.SessionDoc) -> list[NormTurn]:
    ranges = _turn_ranges(doc)
    if not ranges:
        messages = [m for entry in doc.entries for m in _entry_messages(entry)]
        return [NormTurn(id=S.HEADER_TURN_ID, deletable=False, messages=messages)] if messages else []
    starts = [start for start, _ in ranges.values()]
    turns = []
    if starts[0]:
        header = [m for entry in doc.entries[:starts[0]] for m in _entry_messages(entry)]
        turns.append(NormTurn(id=S.HEADER_TURN_ID, deletable=False, messages=header))
    for tid, (start, end) in ranges.items():
        messages = [m for entry in doc.entries[start:end] for m in _entry_messages(entry)]
        timestamp = (doc.entries[start].data or {}).get("timestamp")
        turns.append(NormTurn(id=tid, deletable=True, messages=messages, timestamp=timestamp))
    return turns


class ChatGPTProvider:
    id = "chatgpt"
    label = "ChatGPT / Codex"
    product_name = "ChatGPT / Codex"

    def list_projects(self) -> list[dict]:
        counts: dict[str, int] = {}
        for slug, _ in _iter_sessions():
            counts[slug] = counts.get(slug, 0) + 1
        return [
            {"slug": slug, "label": slug, "sessionCount": count}
            for slug, count in sorted(counts.items(), reverse=True)
        ]

    def list_sessions(self, slug: str) -> list[dict]:
        S.safe_name(slug)
        out = []
        for session_slug, path in _iter_sessions():
            if session_slug != slug:
                continue
            try:
                doc = S.load_session(path)
                turns = _group_turns(doc)
                prompts = [
                    message.text for turn in turns if turn.deletable
                    for message in turn.messages if message.kind == "user" and message.text.strip()
                ]
                turn_count = sum(turn.deletable for turn in turns)
                title = prompts[0].splitlines()[0][:80] if prompts else path.stem
            except (OSError, UnicodeDecodeError):
                log.warning("listing %s without details", path, exc_info=True)
                title, turn_count = "(unreadable)", 0
            stat = path.stat()
            out.append({"sid": path.name, "title": title, "mtime": stat.st_mtime,
                        "bytes": stat.st_size, "turnCount": turn_count})
        return sorted(out, key=lambda session: session["mtime"], reverse=True)

    def session_payload(self, slug: str, sid: str, fold_synthetic: bool = True) -> dict:
        path = _session_path(slug, sid)
        doc = S.load_session(path)
        return {"source": self.id, "slug": slug, "sid": sid, "hash": S.file_hash(path),
                "bytes": path.stat().st_size,
                "turns": [turn.to_summary(i) for i, turn in enumerate(_group_turns(doc))],
                **S.History(slug, sid, source=self.id).status()}

    def perform_delete(self, slug: str, sid: str, turn_ids: list[str], expected_hash: str,
                       fold_synthetic: bool = True) -> None:
        path = _session_path(slug, sid)
        S.check_hash(path, expected_hash, self.product_name)
        doc = S.load_session(path)
        ranges = _turn_ranges(doc)
        unknown = [turn_id for turn_id in turn_ids if turn_id not in ranges]
        if unknown:
            raise ValueError(f"unknown or undeletable turn ids: {unknown}")
        deleted = {index for turn_id in turn_ids for index in range(*ranges[turn_id])}
        edited = S.SessionDoc(path, [entry for i, entry in enumerate(doc.entries) if i not in deleted],
                              doc.trailing_newline)
        S.History(slug, sid, source=self.id).record_and_write(path, _doc_text(edited))

    def perform_delete_session(self, slug: str, sid: str) -> None:
        path = _session_path(slug, sid)
        S.archive_deleted_session(self.id, slug, sid, {path.name: path.read_bytes()})
        path.unlink()

    def perform_undo(self, slug: str, sid: str, expected_hash: str) -> None:
        path = _session_path(slug, sid)
        S.check_hash(path, expected_hash, self.product_name)
        S.History(slug, sid, source=self.id).undo(path)

    def perform_redo(self, slug: str, sid: str, expected_hash: str) -> None:
        path = _session_path(slug, sid)
        S.check_hash(path, expected_hash, self.product_name)
        S.History(slug, sid, source=self.id).redo(path)


register_provider(ChatGPTProvider())
