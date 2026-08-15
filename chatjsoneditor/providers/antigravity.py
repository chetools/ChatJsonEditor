"""Antigravity conversation provider.

Authoritative store: ~/.gemini/antigravity/conversations/<uuid>.db (SQLite,
protobuf step payloads). Readable parallel log under brain/<uuid>/…/transcript.jsonl.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from .. import jsonio
from .. import sessions as S
from .base import register_provider
from .common import (
    NormMsg,
    NormTurn,
    cap_tool,
    first_prompt_title,
    group_projects,
    session_payload,
)

log = logging.getLogger(__name__)

# Verified correlation: step_type 14 ↔ USER_INPUT in transcript.
STEP_USER_INPUT = 14


def _root() -> Path:
    return S.antigravity_root()


def _conversations_dir() -> Path:
    return _root() / "conversations"


def _brain_dir() -> Path:
    return _root() / "brain"


def _config_projects() -> dict[str, str]:
    """Map folder path → project name from ~/.gemini/config/projects."""
    cfg = S.env_root("ANTIGRAVITY_CONFIG_PROJECTS", ".gemini", "config", "projects")
    out: dict[str, str] = {}
    if not cfg.is_dir():
        return out
    for p in cfg.glob("*.json"):
        if p.name == "outside-of-project.json":
            continue
        data = jsonio.read_json_or_default(p)
        if not data:
            continue
        name = data.get("name") or p.stem
        resources = (data.get("projectResources") or {}).get("resources") or []
        for r in resources:
            uri = r.get("folderUri") or ""
            if uri.startswith("file:"):
                path = unquote(urlparse(uri).path)
                # Windows file:///c:/... → /c:/...
                if re.match(r"^/[A-Za-z]:", path):
                    path = path[1:]
                path = path.replace("/", "\\") if "\\" in path or path[1:3] == ":\\" else path
                out[path.lower()] = name
                out[path.replace("\\", "/").lower()] = name
    return out


def _cwd_from_db(db_path: Path) -> str | None:
    try:
        con = sqlite3.connect(str(db_path), timeout=2.0)
        try:
            row = con.execute(
                "SELECT data FROM trajectory_metadata_blob LIMIT 1"
            ).fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        log.warning("could not read project path from %s", db_path, exc_info=True)
        return None
    if not row or not row[0]:
        return None
    blob = row[0] if isinstance(row[0], (bytes, memoryview)) else bytes(row[0])
    for t in re.findall(rb"file://[^\x00-\x1f]{5,300}", bytes(blob)):
        uri = t.decode("utf-8", "replace")
        path = unquote(urlparse(uri).path)
        if re.match(r"^/[A-Za-z]:", path):
            path = path[1:]
        return path
    # printable path-like strings
    for s in re.findall(rb"[A-Za-z]:[\\/][^\x00-\x1f]{3,200}", bytes(blob)):
        return s.decode("utf-8", "replace")
    return None


def _transcript_path(cid: str) -> Path | None:
    base = _brain_dir() / cid / ".system_generated" / "logs"
    for name in ("transcript.jsonl", "transcript_full.jsonl"):
        p = base / name
        if p.is_file():
            return p
    return None


def _transcript_full_path(cid: str) -> Path | None:
    p = _brain_dir() / cid / ".system_generated" / "logs" / "transcript_full.jsonl"
    return p if p.is_file() else None


def _load_transcript(cid: str) -> list[dict]:
    path = _transcript_path(cid)
    return jsonio.load_objects(path) if path else []


def _bundle_paths(cid: str, db: Path) -> dict[str, Path]:
    """Snapshot bundle: the SQLite store plus whichever transcripts exist."""
    paths = {"conversation.db": db}
    tpath = _transcript_path(cid)
    tfpath = _transcript_full_path(cid)
    if tpath:
        paths["transcript.jsonl"] = tpath
    if tfpath:
        paths["transcript_full.jsonl"] = tfpath
    return paths


def _transcript_without_steps(path: Path, del_idxs: set[int]) -> bytes:
    """Transcript bytes with lines for the deleted step indices removed."""
    kept = []
    for line in jsonio.read_text(path).splitlines():
        if not line.strip():
            continue
        obj = jsonio.parse_object(line)
        if obj is not None:
            si = obj.get("step_index")
            if si is not None and int(si) in del_idxs:
                continue
        kept.append(line)
    return jsonio.join_lines(kept)


def _open_db(db_path: Path, readonly: bool = False) -> sqlite3.Connection:
    uri = db_path.resolve().as_uri()
    if readonly:
        con = sqlite3.connect(f"{uri}?mode=ro", uri=True, timeout=2.0)
    else:
        con = sqlite3.connect(str(db_path), timeout=5.0)
    con.row_factory = sqlite3.Row
    return con


def _step_rows(db_path: Path) -> list[sqlite3.Row]:
    try:
        con = _open_db(db_path, readonly=True)
        try:
            return list(con.execute(
                "SELECT idx, step_type, status FROM steps ORDER BY idx"
            ))
        finally:
            con.close()
    except sqlite3.Error as e:
        raise S.ConflictError(
            f"Could not open Antigravity database (locked or corrupt): {e}"
        ) from e


def _user_step_indices(steps: list[sqlite3.Row], transcript: list[dict]) -> list[int]:
    """Return step idx values that start a user turn.

    Prefer each USER_INPUT's ``step_index`` from the transcript. Do **not** use
    the transcript *line number* as an index into ``steps`` — Antigravity often
    leaves gaps in step_index (e.g. line 5 → si=6), so line-index mapping shifts
    every later turn and desyncs the summary pane from the center transcript.
    """
    if transcript:
        idxs: list[int] = []
        seen: set[int] = set()
        for t in transcript:
            if t.get("type") != "USER_INPUT":
                continue
            si = t.get("step_index")
            if si is None:
                continue
            si = int(si)
            if si in seen:
                continue
            seen.add(si)
            idxs.append(si)
        if idxs:
            return idxs
    return [r["idx"] for r in steps if r["step_type"] == STEP_USER_INPUT]


def _turn_ranges(
    steps: list[sqlite3.Row], user_idxs: list[int]
) -> list[tuple[str, int, int]]:
    """Return (turn_id, start_idx, end_idx_exclusive) over step idx values."""
    if not steps:
        return []
    all_idxs = [r["idx"] for r in steps]
    if not user_idxs:
        return [(S.HEADER_TURN_ID, all_idxs[0], all_idxs[-1] + 1)]
    ranges = []
    # header before first user
    first_user_pos = next(i for i, r in enumerate(steps) if r["idx"] == user_idxs[0])
    if first_user_pos > 0:
        ranges.append((S.HEADER_TURN_ID, all_idxs[0], user_idxs[0]))
    for k, uidx in enumerate(user_idxs):
        end = user_idxs[k + 1] if k + 1 < len(user_idxs) else all_idxs[-1] + 1
        ranges.append((f"ag-{uidx}", uidx, end))
    return ranges


def _user_request_text(content: str) -> str:
    """Extract the human-visible request from a USER_INPUT payload."""
    text = content or ""
    m = re.search(r"<USER_REQUEST>\s*([\s\S]*?)\s*</USER_REQUEST>", text)
    if m:
        text = m.group(1).strip()
    # Drop trailing machine metadata blocks when no USER_REQUEST wrapper
    text = re.sub(
        r"\s*<ADDITIONAL_METADATA>[\s\S]*?</ADDITIONAL_METADATA>\s*",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s*<ADDITIONAL_METADATA>[\s\S]*$", "", text, flags=re.I)
    return text.strip()


def _entry_to_msg(entry: dict) -> list[NormMsg]:
    t = entry.get("type") or ""
    content = entry.get("content") or ""
    thinking = entry.get("thinking") or ""
    out: list[NormMsg] = []
    if t == "USER_INPUT":
        text = _user_request_text(content)
        out.append(NormMsg(kind="user", text=text, timestamp=entry.get("created_at")))
    elif t in ("PLANNER_RESPONSE", "MODEL_RESPONSE"):
        if thinking.strip():
            out.append(NormMsg(kind="thinking", text=thinking))
        if content.strip():
            out.append(NormMsg(kind="assistant", text=content))
    elif t in ("EPHEMERAL_MESSAGE", "CONVERSATION_HISTORY", "SYSTEM_MESSAGE"):
        if content.strip():
            out.append(NormMsg(kind="system", text=content, synthetic=True))
    elif t == "ERROR_MESSAGE":
        out.append(NormMsg(kind="system", text=content or "error", synthetic=True))
    else:
        # Tool / action steps (CODE_ACTION, RUN_COMMAND, SEARCH_WEB, GENERIC
        # task notifications, …). Never surface as assistant prose — that made
        # the summary pane show "Created At: …" as the reply and cluttered
        # reading view.
        name = t.lower() if t else "action"
        out.append(NormMsg(
            kind="tool_use",
            id=f"{t}-{entry.get('step_index')}",
            name=name,
            input=cap_tool(content if content else json.dumps(
                {k: entry.get(k) for k in entry if k not in ("content", "thinking")},
                ensure_ascii=False, indent=2, default=str,
            )),
        ))
        if content.strip() and t not in ("CODE_ACTION",):
            out.append(NormMsg(
                kind="tool_result",
                tool_use_id=f"{t}-{entry.get('step_index')}",
                text=cap_tool(content),
            ))
    return out


def _build_turns(cid: str, db_path: Path, fold_synthetic: bool = True) -> list[NormTurn]:
    steps = _step_rows(db_path)
    transcript = _load_transcript(cid)
    user_idxs = _user_step_indices(steps, transcript)
    ranges = _turn_ranges(steps, user_idxs)

    turns: list[NormTurn] = []
    for tid, start, end in ranges:
        deletable = tid != S.HEADER_TURN_ID
        msgs: list[NormMsg] = []
        ts = None
        # Collect transcript entries in [start, end) in file order so the user
        # prompt always precedes its assistant reply within the turn.
        for e in transcript:
            si = e.get("step_index")
            if si is None:
                continue
            si = int(si)
            if not (start <= si < end):
                continue
            for m in _entry_to_msg(e):
                if m.timestamp and not ts:
                    ts = m.timestamp
                msgs.append(m)
        if not msgs and not transcript:
            # fallback: opaque placeholder per step
            for r in steps:
                if start <= r["idx"] < end:
                    msgs.append(NormMsg(
                        kind="system",
                        text=f"step {r['idx']} type={r['step_type']}",
                        synthetic=True,
                    ))
        turns.append(NormTurn(id=tid, deletable=deletable, messages=msgs, timestamp=ts))
    return turns


def _db_path(sid: str) -> Path:
    return _conversations_dir() / f"{S.safe_name(sid)}.db"


def _project_slug_for_cwd(cwd: str | None, projects: dict[str, str]) -> tuple[str, str]:
    if not cwd:
        return "_unknown", "(unknown project)"
    key = cwd.replace("\\", "/").lower()
    key2 = cwd.lower()
    name = projects.get(key) or projects.get(key2)
    # also try suffix match
    if not name:
        for p, n in projects.items():
            if key.endswith(p.replace("\\", "/")) or p.replace("\\", "/").endswith(key):
                name = n
                break
    label = name or cwd
    # slug: sanitized path-ish
    slug = re.sub(r"[^A-Za-z0-9._%-]+", "-", cwd).strip("-")
    if len(slug) > 120:
        slug = slug[:120]
    return slug or "_unknown", label


class AntigravityProvider:
    id = "antigravity"
    label = "Antigravity"
    product_name = "Antigravity"

    def _all_sessions(self) -> list[dict]:
        """Internal: list session metadata with project grouping fields."""
        conv = _conversations_dir()
        if not conv.is_dir():
            return []
        projects = _config_projects()
        out = []
        for db in sorted(conv.glob("*.db")):
            if db.name.endswith("-shm") or db.name.endswith("-wal"):
                continue
            sid = db.stem
            cwd = _cwd_from_db(db)
            slug, label = _project_slug_for_cwd(cwd, projects)
            st = db.stat()
            turns: list[NormTurn] = []
            try:
                turns = _build_turns(sid, db, True)
                turn_count = sum(1 for t in turns if t.deletable)
            except (S.ConflictError, S.CorruptDataError, OSError, sqlite3.Error):
                log.warning("listing %s without turn counts", db, exc_info=True)
                turn_count = 0
            title = first_prompt_title(turns, sid)
            # try task.md / brain
            task = _brain_dir() / sid / "task.md"
            if task.is_file():
                try:
                    lines = task.read_text(encoding="utf-8").strip().splitlines()
                except (OSError, UnicodeDecodeError):
                    log.warning("could not read %s", task, exc_info=True)
                    lines = []
                if lines and lines[0]:
                    title = lines[0][:80]
            out.append({
                "sid": sid,
                "slug": slug,
                "label": label,
                "title": title,
                "mtime": st.st_mtime,
                "bytes": st.st_size,
                "turnCount": turn_count,
                "cwd": cwd,
            })
        return out

    def list_projects(self) -> list[dict]:
        return group_projects(
            (s["slug"], s["label"]) for s in self._all_sessions()
        )

    def list_sessions(self, slug: str) -> list[dict]:
        S.safe_name(slug)
        out = []
        for s in self._all_sessions():
            if s["slug"] != slug:
                continue
            out.append({
                "sid": s["sid"],
                "title": s["title"],
                "mtime": s["mtime"],
                "bytes": s["bytes"],
                "turnCount": s["turnCount"],
            })
        out.sort(key=lambda x: x["mtime"], reverse=True)
        return out

    def _resolve(self, slug: str, sid: str) -> Path:
        # The conversation id alone identifies the database; the slug is a
        # display grouping derived from the cwd, so a mismatch (renamed or
        # moved project) is logged rather than treated as a missing session.
        db = _db_path(sid)
        if not db.is_file():
            raise FileNotFoundError(f"no such session: {sid}")
        for s in self._all_sessions():
            if s["sid"] == sid and s["slug"] != slug:
                log.info("session %s is now grouped under %r, not %r", sid, s["slug"], slug)
        return db

    def session_payload(self, slug: str, sid: str, fold_synthetic: bool = True) -> dict:
        db = self._resolve(slug, sid)
        turns = _build_turns(sid, db, fold_synthetic)
        return session_payload(
            self.id, slug, sid,
            hash=S.file_hash(db),
            nbytes=db.stat().st_size,
            turns=[t.to_summary(i) for i, t in enumerate(turns)],
        )

    def perform_delete_session(self, slug: str, sid: str) -> None:
        db = self._resolve(slug, sid)
        files = {p.name: p.read_bytes() for p in _bundle_paths(sid, db).values()}
        S.archive_deleted_session(self.id, slug, sid, files)
        db.unlink()
        # remove auxiliary brain tree for this conversation id
        brain = _brain_dir() / S.safe_name(sid)
        if brain.is_dir():
            shutil.rmtree(brain, ignore_errors=True)
            if brain.exists():
                # the conversation itself is gone; leftovers are reported, not fatal
                log.warning("could not fully remove auxiliary data at %s", brain)

    def perform_delete(
        self,
        slug: str,
        sid: str,
        turn_ids: list[str],
        expected_hash: str,
        fold_synthetic: bool = True,
    ) -> None:
        db = self._resolve(slug, sid)
        S.check_hash(db, expected_hash, self.product_name)
        steps = _step_rows(db)
        transcript = _load_transcript(sid)
        user_idxs = _user_step_indices(steps, transcript)
        ranges = _turn_ranges(steps, user_idxs)
        by_id = {tid: (a, b) for tid, a, b in ranges if tid != S.HEADER_TURN_ID}
        unknown = [tid for tid in turn_ids if tid not in by_id]
        if unknown:
            raise ValueError(f"unknown or undeletable turn ids: {unknown}")
        del_idxs: set[int] = set()
        for tid in turn_ids:
            a, b = by_id[tid]
            # collect concrete step idxs in range
            for r in steps:
                if a <= r["idx"] < b:
                    del_idxs.add(r["idx"])

        # Perform deletion in a temp copy then replace
        parent = db.parent
        fd, tmp_db = tempfile.mkstemp(dir=str(parent), prefix=db.name + ".", suffix=".tmp")
        os.close(fd)
        try:
            shutil.copy2(db, tmp_db)
            con = sqlite3.connect(tmp_db, timeout=5.0)
            try:
                con.execute("BEGIN IMMEDIATE")
                has_gen_metadata = con.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='gen_metadata'"
                ).fetchone() is not None
                for idx in sorted(del_idxs):
                    con.execute("DELETE FROM steps WHERE idx = ?", (idx,))
                    if has_gen_metadata:
                        con.execute("DELETE FROM gen_metadata WHERE idx = ?", (idx,))
                con.commit()
            except sqlite3.Error as e:
                con.rollback()
                raise S.ConflictError(
                    f"Could not edit the Antigravity database (locked or corrupt): {e}"
                ) from e
            finally:
                con.close()

            path_map = _bundle_paths(sid, db)
            new_data: dict[str, bytes] = {"conversation.db": Path(tmp_db).read_bytes()}
            for rel, p in path_map.items():
                if rel != "conversation.db":
                    new_data[rel] = _transcript_without_steps(p, del_idxs)
            S.History(slug, sid, source=self.id).record_and_write_bundle(path_map, new_data)
        finally:
            try:
                os.unlink(tmp_db)
            except OSError:
                log.warning("could not remove temp file %s", tmp_db, exc_info=True)

    def _guarded_bundle(self, slug: str, sid: str, expected_hash: str) -> dict[str, Path]:
        """Hash-guarded snapshot bundle for undo/redo."""
        db = self._resolve(slug, sid)
        S.check_hash(db, expected_hash, self.product_name)
        return _bundle_paths(sid, db)

    def perform_undo(self, slug: str, sid: str, expected_hash: str) -> None:
        paths = self._guarded_bundle(slug, sid, expected_hash)
        S.History(slug, sid, source=self.id).undo_bundle(paths)

    def perform_redo(self, slug: str, sid: str, expected_hash: str) -> None:
        paths = self._guarded_bundle(slug, sid, expected_hash)
        S.History(slug, sid, source=self.id).redo_bundle(paths)


register_provider(AntigravityProvider())
