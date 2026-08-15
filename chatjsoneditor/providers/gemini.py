"""Gemini CLI session provider.

Sessions live under ~/.gemini/tmp/<project_hash>/chats/ as JSON or JSONL.
Format varies across CLI versions; this adapter is intentionally tolerant.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .. import jsonio
from .. import sessions as S
from .base import register_provider
from .common import (
    NormMsg,
    NormTurn,
    cap_tool,
    deleted_indices,
    first_prompt_title,
    session_payload,
    text_from_content,
    turn_spans,
)

log = logging.getLogger(__name__)


def _root() -> Path:
    return S.gemini_tmp_root()


def _iter_session_files() -> list[tuple[str, Path]]:
    """Return (project_hash, session_file) pairs."""
    root = _root()
    out = []
    if not root.is_dir():
        return out
    for proj in sorted(root.iterdir()):
        if not proj.is_dir():
            continue
        chats = proj / "chats"
        scan_dirs = [chats] if chats.is_dir() else [proj]
        for d in scan_dirs:
            for p in sorted(d.iterdir()):
                if not p.is_file():
                    continue
                if p.suffix.lower() in (".json", ".jsonl") and not p.name.startswith("."):
                    # skip obvious non-session logs if clearly named
                    if p.name in ("logs.json", "settings.json"):
                        continue
                    out.append((proj.name, p))
    return out


def _project_label(project_hash: str) -> str:
    root = _root() / project_hash
    # optional path markers used by some CLI builds
    for name in (".project_root", "project_root", "cwd", ".cwd"):
        p = root / name
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                log.warning("could not read project marker %s", p, exc_info=True)
                continue
            if text:
                return text.splitlines()[0].strip()
    short = project_hash[:12] + "…" if len(project_hash) > 12 else project_hash
    return f"project {short}"


def _only_messages(path: Path, items: list) -> list[dict]:
    """Message objects, refusing a transcript that also holds other values.

    Anything dropped here would be lost the next time the file is rewritten.
    """
    messages = [m for m in items if isinstance(m, dict)]
    if len(messages) != len(items):
        raise S.CorruptDataError(
            f"{path.name} contains {len(items) - len(messages)} entr(ies) that are "
            "not messages; repair the file before editing this session"
        )
    return messages


def _load_messages(path: Path) -> tuple[list[dict], dict]:
    """Return (messages, envelope) where envelope holds outer metadata."""
    text = jsonio.read_text(path)
    if path.suffix.lower() == ".jsonl" or (
        not text.lstrip().startswith("{") and not text.lstrip().startswith("[")
    ):
        messages = []
        skipped = 0
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            obj = jsonio.parse_object(line)
            if obj is None:
                skipped += 1
                continue
            # unwrap {"message": {...}} envelopes; otherwise take the line as-is
            if (
                "role" not in obj and "type" not in obj and "parts" not in obj
                and isinstance(obj.get("message"), dict)
            ):
                obj = obj["message"]
            messages.append(obj)
        if skipped:
            # unparseable lines are dropped on rewrite, so never do it silently
            raise S.CorruptDataError(
                f"{path.name} has {skipped} unparseable or non-message line(s); "
                "repair the file before editing this session"
            )
        return messages, {"format": "jsonl"}

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # Returning "no messages" here made a delete rewrite the file as an
        # empty transcript, destroying a session we simply could not read.
        raise S.CorruptDataError(f"{path.name} is not valid JSON: {e}") from e

    if isinstance(data, list):
        return _only_messages(path, data), {"format": "list"}
    if not isinstance(data, dict):
        raise S.CorruptDataError(
            f"{path.name} holds a {type(data).__name__}, not a session transcript"
        )

    for key in ("messages", "history", "chat", "items"):
        if isinstance(data.get(key), list):
            return _only_messages(path, data[key]), data
    # single message object
    if "role" in data or "parts" in data:
        return [data], data
    return [], data


def _role_of(msg: dict) -> str:
    r = (msg.get("role") or msg.get("type") or msg.get("author") or "").lower()
    if r in ("user", "human", "model", "assistant", "system", "tool", "function"):
        return "model" if r == "assistant" else r
    # gemini sometimes uses "user"/"model"
    return r or "unknown"


def _parts_text(msg: dict) -> str:
    if "parts" in msg and isinstance(msg["parts"], list):
        parts = []
        for p in msg["parts"]:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if "text" in p:
                    parts.append(p.get("text") or "")
                elif p.get("functionCall") or p.get("function_call"):
                    continue
                elif p.get("functionResponse") or p.get("function_response"):
                    fr = p.get("functionResponse") or p.get("function_response")
                    parts.append(text_from_content(fr))
        return "\n".join(x for x in parts if x)
    return text_from_content(msg.get("content") or msg.get("text") or msg.get("message"))


def _msg_to_norm(msg: dict) -> list[NormMsg]:
    role = _role_of(msg)
    out: list[NormMsg] = []
    # function calls in parts
    parts = msg.get("parts") if isinstance(msg.get("parts"), list) else []
    text = _parts_text(msg)
    if role in ("user", "human"):
        out.append(NormMsg(kind="user", text=text))
    elif role in ("system",):
        if text.strip():
            out.append(NormMsg(kind="system", text=text, synthetic=True))
    elif role in ("tool", "function"):
        out.append(NormMsg(
            kind="tool_result",
            tool_use_id=msg.get("tool_call_id") or msg.get("id") or msg.get("name"),
            text=cap_tool(text),
            is_error=bool(msg.get("is_error")),
        ))
    else:
        # model / assistant
        if text.strip():
            out.append(NormMsg(kind="assistant", text=text))
        for p in parts:
            if not isinstance(p, dict):
                continue
            fc = p.get("functionCall") or p.get("function_call")
            if fc and isinstance(fc, dict):
                args = fc.get("args") or fc.get("arguments") or {}
                out.append(NormMsg(
                    kind="tool_use",
                    id=fc.get("id") or fc.get("name"),
                    name=fc.get("name", "?"),
                    input=cap_tool(jsonio.pretty(args)),
                ))
            fr = p.get("functionResponse") or p.get("function_response")
            if fr and isinstance(fr, dict) and not text:
                out.append(NormMsg(
                    kind="tool_result",
                    tool_use_id=fr.get("id") or fr.get("name"),
                    text=cap_tool(text_from_content(fr.get("response") or fr)),
                ))
        # thoughts
        thought = msg.get("thought") or msg.get("thinking")
        if thought:
            out.insert(0, NormMsg(kind="thinking", text=text_from_content(thought)))
    if not out and text:
        out.append(NormMsg(kind="raw", text=text))
    return out


def _is_user_msg(msg: dict) -> bool:
    return _role_of(msg) in ("user", "human")


def _turn_id(messages: list[dict], start: int) -> str:
    mid = messages[start].get("id") or messages[start].get("messageId")
    return str(mid) if mid else f"g{start}"


def _turn_ranges(messages: list[dict]) -> dict[str, tuple[int, int]]:
    """Deletable turn id → [start, end) message index range."""
    starts = [i for i, m in enumerate(messages) if _is_user_msg(m)]
    return {
        _turn_id(messages, start): (start, end)
        for is_header, start, end in turn_spans(starts, len(messages))
        if not is_header
    }


def _group_turns(messages: list[dict], fold_synthetic: bool = True) -> list[NormTurn]:
    starts = [i for i, m in enumerate(messages) if _is_user_msg(m)]
    turns: list[NormTurn] = []
    for is_header, start, end in turn_spans(starts, len(messages)):
        msgs: list[NormMsg] = []
        for m in messages[start:end]:
            msgs.extend(_msg_to_norm(m))
        tid = S.HEADER_TURN_ID if is_header else _turn_id(messages, start)
        turns.append(NormTurn(id=tid, deletable=not is_header, messages=msgs))
    return turns


def _rewrite_file(path: Path, envelope: dict, messages: list[dict]) -> bytes:
    if path.suffix.lower() == ".jsonl" or envelope.get("format") == "jsonl":
        return jsonio.join_lines(
            json.dumps(m, ensure_ascii=False) for m in messages
        )
    if envelope.get("format") == "list":
        return jsonio.dump_json(messages)
    # object envelope
    data = dict(envelope)
    data.pop("format", None)
    written = False
    for key in ("messages", "history", "chat", "items"):
        if key in data or (not written and key == "messages"):
            data[key] = messages
            written = True
            break
    if not written:
        data["messages"] = messages
    return jsonio.dump_json(data)


class GeminiProvider:
    id = "gemini"
    label = "Gemini CLI"
    product_name = "Gemini CLI"

    def list_projects(self) -> list[dict]:
        counts: dict[str, int] = {}
        for ph, _ in _iter_session_files():
            counts[ph] = counts.get(ph, 0) + 1
        return [
            {"slug": ph, "label": _project_label(ph), "sessionCount": n}
            for ph, n in sorted(counts.items(), key=lambda x: _project_label(x[0]).lower())
        ]

    def _session_path(self, slug: str, sid: str) -> Path:
        # sid is the file stem or full filename
        root = _root() / S.safe_name(slug)
        sid = S.safe_name(sid)
        candidates = [
            root / "chats" / sid,
            root / "chats" / f"{sid}.json",
            root / "chats" / f"{sid}.jsonl",
            root / sid,
            root / f"{sid}.json",
            root / f"{sid}.jsonl",
        ]
        for c in candidates:
            if c.is_file():
                # path components validated via construction under root
                try:
                    c.resolve().relative_to(root.resolve())
                except ValueError as e:
                    raise ValueError("path escape") from e
                return c
        raise FileNotFoundError(f"no such session: {sid}")

    def list_sessions(self, slug: str) -> list[dict]:
        S.safe_name(slug)
        out = []
        for ph, path in _iter_session_files():
            if ph != slug:
                continue
            try:
                messages, _ = _load_messages(path)
            except (S.CorruptDataError, OSError):
                log.warning("listing %s without turn counts", path, exc_info=True)
                messages = []
            turns = _group_turns(messages)
            turn_count = sum(1 for t in turns if t.deletable)
            st = path.stat()
            out.append({
                "sid": path.name,  # keep extension for disambiguation
                "title": first_prompt_title(turns, path.stem),
                "mtime": st.st_mtime,
                "bytes": st.st_size,
                "turnCount": turn_count,
            })
        out.sort(key=lambda s: s["mtime"], reverse=True)
        return out

    def session_payload(self, slug: str, sid: str, fold_synthetic: bool = True) -> dict:
        path = self._session_path(slug, sid)
        messages, _ = _load_messages(path)
        turns = _group_turns(messages, fold_synthetic)
        return session_payload(
            self.id, slug, sid,
            hash=S.file_hash(path),
            nbytes=path.stat().st_size,
            turns=[t.to_summary(i) for i, t in enumerate(turns)],
        )

    def perform_delete_session(self, slug: str, sid: str) -> None:
        path = self._session_path(slug, sid)
        S.archive_deleted_session(self.id, slug, sid, {path.name: path.read_bytes()})
        path.unlink()

    def perform_delete(
        self,
        slug: str,
        sid: str,
        turn_ids: list[str],
        expected_hash: str,
        fold_synthetic: bool = True,
    ) -> None:
        path = self._session_path(slug, sid)
        S.check_hash(path, expected_hash, self.product_name)
        messages, envelope = _load_messages(path)
        del_idx = deleted_indices(_turn_ranges(messages), turn_ids)
        new_messages = [m for i, m in enumerate(messages) if i not in del_idx]
        new_bytes = _rewrite_file(path, envelope, new_messages)
        # write as text if utf-8
        S.History(slug, sid, source=self.id).record_and_write(
            path, new_bytes.decode("utf-8")
        )

    def perform_undo(self, slug: str, sid: str, expected_hash: str) -> None:
        path = self._session_path(slug, sid)
        S.check_hash(path, expected_hash, self.product_name)
        S.History(slug, sid, source=self.id).undo(path)

    def perform_redo(self, slug: str, sid: str, expected_hash: str) -> None:
        path = self._session_path(slug, sid)
        S.check_hash(path, expected_hash, self.product_name)
        S.History(slug, sid, source=self.id).redo(path)


register_provider(GeminiProvider())
