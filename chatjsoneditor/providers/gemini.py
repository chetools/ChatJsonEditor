"""Gemini CLI session provider.

Sessions live under ~/.gemini/tmp/<project_hash>/chats/ as JSON or JSONL.
Format varies across CLI versions; this adapter is intentionally tolerant.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import sessions as S
from .base import register_provider
from .common import NormMsg, NormTurn, cap_tool, text_from_content


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
                if text:
                    return text.splitlines()[0].strip()
            except OSError:
                pass
    short = project_hash[:12] + "…" if len(project_hash) > 12 else project_hash
    return f"project {short}"


def _load_messages(path: Path) -> tuple[list[dict], dict]:
    """Return (messages, envelope) where envelope holds outer metadata."""
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    if path.suffix.lower() == ".jsonl" or (
        not text.lstrip().startswith("{") and not text.lstrip().startswith("[")
    ):
        messages = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                # wrap line as message if it looks like one
                if "role" in obj or "type" in obj or "parts" in obj:
                    messages.append(obj)
                elif "message" in obj and isinstance(obj["message"], dict):
                    messages.append(obj["message"])
                else:
                    messages.append(obj)
        return messages, {"format": "jsonl"}

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [], {"format": "broken"}

    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)], {"format": "list"}
    if not isinstance(data, dict):
        return [], {"format": "unknown"}

    for key in ("messages", "history", "chat", "items"):
        if isinstance(data.get(key), list):
            return [m for m in data[key] if isinstance(m, dict)], data
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
                    input=cap_tool(json.dumps(args, ensure_ascii=False, indent=2)),
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


def _group_turns(messages: list[dict], fold_synthetic: bool = True) -> list[NormTurn]:
    starts = [i for i, m in enumerate(messages) if _is_user_msg(m)]
    if not starts:
        if messages:
            msgs = []
            for m in messages:
                msgs.extend(_msg_to_norm(m))
            return [NormTurn(id=S.HEADER_TURN_ID, deletable=False, messages=msgs)]
        return []
    turns: list[NormTurn] = []
    if starts[0] > 0:
        msgs = []
        for m in messages[: starts[0]]:
            msgs.extend(_msg_to_norm(m))
        turns.append(NormTurn(id=S.HEADER_TURN_ID, deletable=False, messages=msgs))
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(messages)
        msgs = []
        for m in messages[start:end]:
            msgs.extend(_msg_to_norm(m))
        tid = f"g{start}"
        # optional message id
        mid = messages[start].get("id") or messages[start].get("messageId")
        if mid:
            tid = str(mid)
        turns.append(NormTurn(id=tid, deletable=True, messages=msgs))
    return turns


def _rewrite_file(path: Path, envelope: dict, messages: list[dict]) -> bytes:
    if path.suffix.lower() == ".jsonl" or envelope.get("format") == "jsonl":
        lines = [json.dumps(m, ensure_ascii=False) for m in messages]
        text = "\n".join(lines)
        if text:
            text += "\n"
        return text.encode("utf-8")
    if envelope.get("format") == "list":
        return (json.dumps(messages, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
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
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


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
        S.safe_name(slug)
        # sid is the file stem or full filename
        root = _root() / S.safe_name(slug)
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
            messages, _ = _load_messages(path)
            turns = _group_turns(messages)
            turn_count = sum(1 for t in turns if t.deletable)
            title = path.stem
            for t in turns:
                if t.deletable:
                    for m in t.messages:
                        if m.kind == "user" and m.text.strip():
                            title = m.text.strip().splitlines()[0][:80]
                            break
                    break
            st = path.stat()
            out.append({
                "sid": path.name,  # keep extension for disambiguation
                "title": title,
                "mtime": st.st_mtime,
                "bytes": st.st_size,
                "turnCount": turn_count,
            })
        out.sort(key=lambda s: s["mtime"], reverse=True)
        return out

    def session_payload(self, slug: str, sid: str, fold_synthetic: bool = True) -> dict:
        path = self._session_path(slug, sid)
        messages, _ = _load_messages(path)
        turns = [t.to_summary(i) for i, t in enumerate(_group_turns(messages, fold_synthetic))]
        return {
            "source": self.id,
            "slug": slug,
            "sid": sid,
            "hash": S.file_hash(path),
            "bytes": path.stat().st_size,
            "turns": turns,
            **S.History(slug, sid, source=self.id).status(),
        }

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
        turns = _group_turns(messages, fold_synthetic)
        by_id = {t.id: t for t in turns if t.deletable}
        # map turn id back to message index ranges
        starts = [i for i, m in enumerate(messages) if _is_user_msg(m)]
        id_to_range: dict[str, tuple[int, int]] = {}
        # rebuild ranges aligned with _group_turns
        if starts:
            if starts[0] > 0:
                pass  # header non-deletable
            for k, start in enumerate(starts):
                end = starts[k + 1] if k + 1 < len(starts) else len(messages)
                tid = f"g{start}"
                mid = messages[start].get("id") or messages[start].get("messageId")
                if mid:
                    tid = str(mid)
                id_to_range[tid] = (start, end)
        unknown = [tid for tid in turn_ids if tid not in id_to_range]
        if unknown:
            raise ValueError(f"unknown or undeletable turn ids: {unknown}")
        del_idx: set[int] = set()
        for tid in turn_ids:
            a, b = id_to_range[tid]
            del_idx.update(range(a, b))
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
