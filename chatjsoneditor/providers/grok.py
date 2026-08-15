"""Grok / Grok Build session provider.

Both products share ~/.grok/sessions/<encoded-cwd>/<sid>/ with:
  summary.json, updates.jsonl, chat_history.jsonl, …
Classification into source ids ``grok`` vs ``grok-build`` uses agent_name /
model_id heuristics (Grok Build wins on overlap).
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from urllib.parse import unquote

from .. import sessions as S
from ..sessions import HEADER_TURN_ID, Turn
from .base import register_provider
from .common import (
    NormMsg,
    NormTurn,
    cap_tool,
    extract_user_query,
    is_grok_synthetic_user,
    text_from_content,
)

log = logging.getLogger(__name__)

# Identity files hashed + snapshotted for edit safety.
_IDENTITY = ("updates.jsonl", "chat_history.jsonl", "summary.json")

_BUILD_AGENTS = frozenset({
    "grok-build-plan",
    "grok-build",
    "build",
})


def _is_build_session(summary: dict) -> bool:
    agent = (summary.get("agent_name") or "").lower()
    model = (summary.get("current_model_id") or "").lower()
    if agent in _BUILD_AGENTS or agent.startswith("grok-build"):
        return True
    if model.startswith("grok-build") or model == "grok-build":
        return True
    return False


def _session_dirs(root: Path) -> list[tuple[str, Path, dict]]:
    """Yield (project_slug, session_dir, summary_dict) for every session."""
    out = []
    if not root.is_dir():
        return out
    for proj in sorted(root.iterdir()):
        if not proj.is_dir():
            continue
        # skip non-project files like prompt_history.jsonl at project level
        for sess in sorted(proj.iterdir()):
            if not sess.is_dir():
                continue
            summary_path = sess / "summary.json"
            if not summary_path.is_file():
                continue
            summary = _read_summary(summary_path)
            out.append((proj.name, sess, summary))
    return out


def _read_summary(path: Path) -> dict:
    """summary.json contents, or {} if it is missing/damaged (display only)."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        log.warning("ignoring unreadable %s", path, exc_info=True)
        return {}
    if not isinstance(data, dict):
        log.warning("ignoring %s: expected an object, got %s", path, type(data).__name__)
        return {}
    return data


def _cwd_label(slug: str, summary: dict) -> str:
    cwd = (summary.get("info") or {}).get("cwd")
    if cwd:
        return cwd
    return unquote(slug)


def _identity_paths(sess_dir: Path) -> dict[str, Path]:
    return {name: sess_dir / name for name in _IDENTITY}


def _hash_session(sess_dir: Path) -> str:
    return S.multi_file_hash([sess_dir / n for n in _IDENTITY])


def _load_jsonl(path: Path) -> list[tuple[str, dict | None]]:
    if not path.is_file():
        return []
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as e:
        raise S.CorruptDataError(f"{path.name} is not valid UTF-8: {e}") from e
    trailing = text.endswith("\n")
    body = text[:-1] if trailing else text
    rows = []
    for line in body.split("\n") if body else []:
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                data = None
        except json.JSONDecodeError:
            data = None
        rows.append((line, data))
    return rows


def _write_jsonl(rows: list[tuple[str, dict | None]], trailing: bool = True) -> bytes:
    lines = [raw for raw, _ in rows]
    text = "\n".join(lines)
    if trailing and text:
        text += "\n"
    elif trailing and not text:
        text = ""
    return text.encode("utf-8")


def _chat_user_text(data: dict) -> str:
    return text_from_content(data.get("content"))


def _is_real_chat_user(data: dict) -> bool:
    if data.get("type") != "user":
        return False
    text = _chat_user_text(data)
    return extract_user_query(text) is not None or not is_grok_synthetic_user(text)


def _is_turn_start_chat(data: dict, fold_synthetic: bool) -> bool:
    if data.get("type") != "user":
        return False
    text = _chat_user_text(data)
    if extract_user_query(text) is not None:
        return True
    if is_grok_synthetic_user(text):
        return not fold_synthetic
    return True


def _messages_from_chat_span(rows: list[tuple[str, dict | None]], start: int, end: int) -> list[NormMsg]:
    msgs: list[NormMsg] = []
    for raw, data in rows[start:end]:
        if not data:
            msgs.append(NormMsg(kind="raw", text=raw[: S.MAX_TOOL_TEXT]))
            continue
        t = data.get("type")
        if t == "system":
            text = text_from_content(data.get("content"))
            if text.strip():
                msgs.append(NormMsg(kind="system", text=text, synthetic=True))
        elif t == "user":
            text = _chat_user_text(data)
            q = extract_user_query(text)
            display = q if q is not None else text
            syn = is_grok_synthetic_user(text) and q is None
            msgs.append(NormMsg(kind="user", text=display, synthetic=syn))
        elif t == "reasoning":
            # summary blocks or plain text
            summary = data.get("summary")
            if isinstance(summary, list):
                parts = [
                    b.get("text", "") for b in summary
                    if isinstance(b, dict) and b.get("type") in ("summary_text", "text")
                ]
                think = "\n".join(p for p in parts if p)
            else:
                think = text_from_content(data.get("content") or data.get("text") or "")
            if think.strip():
                msgs.append(NormMsg(kind="thinking", text=think))
        elif t == "assistant":
            content = data.get("content")
            text = text_from_content(content) if not isinstance(content, str) else content
            if text and text.strip():
                msgs.append(NormMsg(kind="assistant", text=text))
            for tc in data.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                args = tc.get("arguments", tc.get("input", {}))
                if isinstance(args, str):
                    try:
                        args_obj = json.loads(args)
                        inp = json.dumps(args_obj, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        inp = args
                else:
                    inp = json.dumps(args or {}, ensure_ascii=False, indent=2)
                msgs.append(NormMsg(
                    kind="tool_use",
                    id=tc.get("id"),
                    name=tc.get("name", "?"),
                    input=cap_tool(inp),
                ))
        elif t == "tool_result":
            msgs.append(NormMsg(
                kind="tool_result",
                tool_use_id=data.get("tool_call_id") or data.get("tool_use_id"),
                is_error=bool(data.get("is_error") or data.get("isError")),
                text=cap_tool(text_from_content(data.get("content"))),
            ))
        elif t == "backend_tool_call":
            kind = data.get("kind") or {}
            name = kind.get("tool_type") or "backend_tool"
            msgs.append(NormMsg(
                kind="tool_use",
                id=data.get("id") or f"backend-{len(msgs)}",
                name=str(name),
                input=cap_tool(json.dumps(kind, ensure_ascii=False, indent=2)),
            ))
    return msgs


def _group_chat_turns(
    rows: list[tuple[str, dict | None]], fold_synthetic: bool
) -> list[tuple[Turn, list[NormMsg]]]:
    starts = []
    for i, (_, data) in enumerate(rows):
        if data and _is_turn_start_chat(data, fold_synthetic):
            starts.append(i)
    if not starts:
        if rows:
            msgs = _messages_from_chat_span(rows, 0, len(rows))
            t = Turn(HEADER_TURN_ID, 0, len(rows), False)
            return [(t, msgs)]
        return []
    turns: list[tuple[Turn, list[NormMsg]]] = []
    if starts[0] > 0:
        msgs = _messages_from_chat_span(rows, 0, starts[0])
        turns.append((Turn(HEADER_TURN_ID, 0, starts[0], False), msgs))
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(rows)
        data = rows[start][1] or {}
        text = _chat_user_text(data)
        q = extract_user_query(text) or text
        # stable across processes: PYTHONHASHSEED randomizes hash() per run,
        # which would invalidate every turn id the client is holding
        digest = hashlib.sha256(q[:80].encode("utf-8")).hexdigest()[:8]
        tid = f"u{start}-{digest}"
        msgs = _messages_from_chat_span(rows, start, end)
        turns.append((Turn(tid, start, end, True), msgs))
    return turns


def _group_update_turns(
    rows: list[tuple[str, dict | None]],
) -> list[tuple[int, int, str]]:
    """Return (start, end, prompt_preview) spans over updates.jsonl."""
    starts: list[tuple[int, str]] = []
    for i, (_, data) in enumerate(rows):
        if not data:
            continue
        update = (data.get("params") or {}).get("update") or {}
        if update.get("sessionUpdate") == "user_message_chunk":
            # only start a turn on first chunk of a user burst
            if starts and starts[-1][0] == i - 1:
                # extend handled later; mark only first of consecutive
                prev = rows[i - 1][1] or {}
                prev_u = (prev.get("params") or {}).get("update") or {}
                if prev_u.get("sessionUpdate") == "user_message_chunk":
                    continue
            content = update.get("content") or {}
            text = content.get("text") if isinstance(content, dict) else ""
            starts.append((i, text or ""))
    if not starts:
        return []
    spans = []
    for k, (start, prompt) in enumerate(starts):
        # merge consecutive user_message_chunk into prompt
        j = start
        prompt_parts = [prompt]
        while j + 1 < len(rows):
            nd = rows[j + 1][1] or {}
            nu = (nd.get("params") or {}).get("update") or {}
            if nu.get("sessionUpdate") != "user_message_chunk":
                break
            c = nu.get("content") or {}
            chunk = c.get("text") if isinstance(c, dict) else ""
            prompt_parts.append(chunk or "")
            j += 1
        end = starts[k + 1][0] if k + 1 < len(starts) else len(rows)
        spans.append((start, end, "".join(prompt_parts)))
    return spans


def _summarize_from_chat(
    chat_rows: list[tuple[str, dict | None]], fold_synthetic: bool
) -> list[dict]:
    grouped = _group_chat_turns(chat_rows, fold_synthetic)
    result = []
    for idx, (turn, msgs) in enumerate(grouped):
        nt = NormTurn(id=turn.id, deletable=turn.deletable, messages=msgs)
        # timestamp: none in chat lines usually
        result.append(nt.to_summary(idx))
    return result


def _delete_chat_turns(
    rows: list[tuple[str, dict | None]],
    turn_ids: list[str],
    fold_synthetic: bool,
) -> list[tuple[str, dict | None]]:
    grouped = _group_chat_turns(rows, fold_synthetic)
    by_id = {t.id: (t, msgs) for t, msgs in grouped if t.deletable}
    unknown = [tid for tid in turn_ids if tid not in by_id]
    if unknown:
        raise ValueError(f"unknown or undeletable turn ids: {unknown}")
    del_idx: set[int] = set()
    for tid in turn_ids:
        t, _ = by_id[tid]
        del_idx.update(range(t.start, t.end))
    return [row for i, row in enumerate(rows) if i not in del_idx]


def _delete_update_turns_by_order(
    rows: list[tuple[str, dict | None]],
    deleted_prompt_order: set[int],
) -> list[tuple[str, dict | None]]:
    """Delete updates spans whose user-prompt order index is in the set."""
    spans = _group_update_turns(rows)
    if not spans:
        return rows
    del_idx: set[int] = set()
    for order, (start, end, _) in enumerate(spans):
        if order in deleted_prompt_order:
            del_idx.update(range(start, end))
    return [row for i, row in enumerate(rows) if i not in del_idx]


def _real_prompt_orders(
    chat_rows: list[tuple[str, dict | None]], fold_synthetic: bool
) -> list[str]:
    """Turn ids of deletable real (or show-all synthetic) user turns in order."""
    return [t.id for t, _ in _group_chat_turns(chat_rows, fold_synthetic) if t.deletable]


class _GrokBase:
    """Shared implementation; subclasses set id/label/filter."""

    id: str
    label: str
    product_name: str
    build_only: bool  # True → grok-build filter; False → non-build only

    def _root(self) -> Path:
        return S.grok_sessions_root()

    def _iter_matching(self) -> list[tuple[str, Path, dict]]:
        matched = []
        for slug, sess, summary in _session_dirs(self._root()):
            is_build = _is_build_session(summary)
            if self.build_only and is_build:
                matched.append((slug, sess, summary))
            elif not self.build_only and not is_build:
                matched.append((slug, sess, summary))
        return matched

    def list_projects(self) -> list[dict]:
        by_slug: dict[str, dict] = {}
        for slug, sess, summary in self._iter_matching():
            if slug not in by_slug:
                by_slug[slug] = {
                    "slug": slug,
                    "label": _cwd_label(slug, summary),
                    "sessionCount": 0,
                }
            by_slug[slug]["sessionCount"] += 1
        return sorted(by_slug.values(), key=lambda p: p["label"].lower())

    def _sess_dir(self, slug: str, sid: str) -> Path:
        d = self._root() / S.safe_name(slug) / S.safe_name(sid)
        if not d.is_dir():
            raise FileNotFoundError(f"no such session: {sid}")
        # verify classification
        summary = _read_summary(d / "summary.json")
        is_build = _is_build_session(summary)
        if self.build_only and not is_build:
            raise FileNotFoundError(f"session {sid} is not a Grok Build session")
        if not self.build_only and is_build:
            raise FileNotFoundError(f"session {sid} is a Grok Build session")
        return d

    def list_sessions(self, slug: str) -> list[dict]:
        S.safe_name(slug)
        out = []
        for s_slug, sess, summary in self._iter_matching():
            if s_slug != slug:
                continue
            title = (
                summary.get("generated_title")
                or summary.get("session_summary")
                or sess.name
            )
            chat = sess / "chat_history.jsonl"
            turn_count = 0
            if chat.is_file():
                rows = _load_jsonl(chat)
                turn_count = sum(
                    1 for t, _ in _group_chat_turns(rows, True) if t.deletable
                )
            # size: identity files
            nbytes = sum((sess / n).stat().st_size for n in _IDENTITY if (sess / n).is_file())
            mtime = sess.stat().st_mtime
            for n in _IDENTITY:
                p = sess / n
                if p.is_file():
                    mtime = max(mtime, p.stat().st_mtime)
            out.append({
                "sid": sess.name,
                "title": title,
                "mtime": mtime,
                "bytes": nbytes,
                "turnCount": turn_count,
                "model": summary.get("current_model_id"),
                "agent": summary.get("agent_name"),
            })
        out.sort(key=lambda s: s["mtime"], reverse=True)
        return out

    def session_payload(self, slug: str, sid: str, fold_synthetic: bool = True) -> dict:
        d = self._sess_dir(slug, sid)
        chat_rows = _load_jsonl(d / "chat_history.jsonl")
        turns = _summarize_from_chat(chat_rows, fold_synthetic)
        nbytes = sum((d / n).stat().st_size for n in _IDENTITY if (d / n).is_file())
        return {
            "source": self.id,
            "slug": slug,
            "sid": sid,
            "hash": _hash_session(d),
            "bytes": nbytes,
            "turns": turns,
            **S.History(slug, sid, source=self.id).status(),
        }

    def perform_delete_session(self, slug: str, sid: str) -> None:
        import shutil

        d = self._sess_dir(slug, sid)
        files: dict[str, bytes] = {}
        for p in sorted(d.rglob("*")):
            if p.is_file():
                try:
                    rel = str(p.relative_to(d)).replace("\\", "/")
                except ValueError:
                    rel = p.name
                files[rel.replace("/", "__")] = p.read_bytes()
        S.archive_deleted_session(self.id, slug, sid, files)
        shutil.rmtree(d)

    def perform_delete(
        self,
        slug: str,
        sid: str,
        turn_ids: list[str],
        expected_hash: str,
        fold_synthetic: bool = True,
    ) -> None:
        d = self._sess_dir(slug, sid)
        S.check_hash_value(_hash_session(d), expected_hash, self.product_name)

        chat_path = d / "chat_history.jsonl"
        updates_path = d / "updates.jsonl"
        summary_path = d / "summary.json"

        chat_rows = _load_jsonl(chat_path)
        # Which ordered prompt indices are being deleted?
        ordered_ids = _real_prompt_orders(chat_rows, fold_synthetic)
        deleted_orders = {i for i, tid in enumerate(ordered_ids) if tid in turn_ids}
        # Deleting by prompt order is what keeps updates.jsonl aligned with
        # chat_history.jsonl, so a turn id that resolves to no order (or to
        # several) would silently trim the wrong span of updates.
        if len(deleted_orders) != len(set(turn_ids) & set(ordered_ids)):
            raise ValueError(
                "ambiguous turn ids: cannot map the selection onto prompt order; "
                "reload the session and retry"
            )

        new_chat = _delete_chat_turns(chat_rows, turn_ids, fold_synthetic)
        updates_rows = _load_jsonl(updates_path)
        new_updates = _delete_update_turns_by_order(updates_rows, deleted_orders)

        # summary patch
        summary = _read_summary(summary_path)
        summary["num_chat_messages"] = len(new_chat)
        summary["num_messages"] = len(new_updates)
        summary_bytes = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

        paths = _identity_paths(d)
        new_data = {
            "chat_history.jsonl": _write_jsonl(new_chat, trailing=True),
            "updates.jsonl": _write_jsonl(new_updates, trailing=True),
            "summary.json": summary_bytes,
        }
        # best-effort rewind_points trim
        rp = d / "rewind_points.jsonl"
        if rp.is_file() and deleted_orders:
            rp_rows = _load_jsonl(rp)
            kept = []
            for raw, data in rp_rows:
                if data and data.get("prompt_index") in deleted_orders:
                    continue
                kept.append((raw, data))
            # also include rewind in snapshot if we touch it
            paths["rewind_points.jsonl"] = rp
            new_data["rewind_points.jsonl"] = _write_jsonl(kept, trailing=True)

        S.History(slug, sid, source=self.id).record_and_write_bundle(paths, new_data)

    def perform_undo(self, slug: str, sid: str, expected_hash: str) -> None:
        d = self._sess_dir(slug, sid)
        S.check_hash_value(_hash_session(d), expected_hash, self.product_name)
        paths = _identity_paths(d)
        rp = d / "rewind_points.jsonl"
        if rp.is_file():
            paths["rewind_points.jsonl"] = rp
        S.History(slug, sid, source=self.id).undo_bundle(paths)

    def perform_redo(self, slug: str, sid: str, expected_hash: str) -> None:
        d = self._sess_dir(slug, sid)
        S.check_hash_value(_hash_session(d), expected_hash, self.product_name)
        paths = _identity_paths(d)
        rp = d / "rewind_points.jsonl"
        if rp.is_file():
            paths["rewind_points.jsonl"] = rp
        S.History(slug, sid, source=self.id).redo_bundle(paths)


class GrokProvider(_GrokBase):
    id = "grok"
    label = "Grok"
    product_name = "Grok"
    build_only = False


class GrokBuildProvider(_GrokBase):
    id = "grok-build"
    label = "Grok Build"
    product_name = "Grok Build"
    build_only = True


register_provider(GrokProvider())
register_provider(GrokBuildProvider())
