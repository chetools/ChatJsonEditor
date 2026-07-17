"""Parsing and editing of Claude Code session JSONL files.

Design principles:
- Untouched lines are preserved byte-for-byte; only entries whose
  parentUuid/leafUuid must be repaired after a deletion are re-serialized.
- Deletion is whole-turn only: a human prompt plus everything it triggered,
  so tool_use/tool_result pairs and sidechains are always removed together.
- Every mutation snapshots the whole file first, giving exact undo/redo.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

HEADER_TURN_ID = "__header__"

# Allow URL-encoded path segments (Grok uses %3A etc.) and UUIDs.
_NAME_RE = re.compile(r"^[A-Za-z0-9._%-]+$")


class ConflictError(Exception):
    """The file on disk no longer matches what the client last saw."""


def projects_root() -> Path:
    override = os.environ.get("CLAUDE_PROJECTS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".claude" / "projects"


def backups_root() -> Path:
    override = os.environ.get("CHATJSONEDITOR_BACKUPS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".claude" / "chatjsoneditor-backups"


def config_root() -> Path:
    override = os.environ.get("CHATJSONEDITOR_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".claude" / "chatjsoneditor"


def grok_sessions_root() -> Path:
    override = os.environ.get("GROK_SESSIONS_DIR")
    if override:
        return Path(override)
    home = os.environ.get("GROK_HOME")
    if home:
        return Path(home) / "sessions"
    return Path.home() / ".grok" / "sessions"


def gemini_tmp_root() -> Path:
    override = os.environ.get("GEMINI_TMP_DIR")
    if override:
        return Path(override)
    return Path.home() / ".gemini" / "tmp"


def antigravity_root() -> Path:
    override = os.environ.get("ANTIGRAVITY_ROOT")
    if override:
        return Path(override)
    return Path.home() / ".gemini" / "antigravity"


# Default keyboard shortcuts (action -> combo string). Combos are normalized as
# optional "Ctrl+"/"Alt+"/"Shift+"/"Meta+" prefixes + the KeyboardEvent .key
# (with " " represented as "Space"). All are rebindable via the in-app panel or
# by editing keybindings.json directly.
DEFAULT_KEYBINDINGS: dict[str, str] = {
    "prevTurn": "ArrowUp",
    "nextTurn": "ArrowDown",
    "scrollUp": "Shift+ArrowUp",
    "scrollDown": "Shift+ArrowDown",
    "toggleSelect": "Space",
    "deleteSelected": "d",
    "undo": "Ctrl+z",
    "redo": "Ctrl+y",
    "firstTurn": "Home",
    "lastTurn": "End",
    "selectAll": "a",
    "clearSelection": "Escape",
    "help": "?",
}


def keybindings_path() -> Path:
    return config_root() / "keybindings.json"


def load_keybindings() -> dict[str, str]:
    """Defaults merged with the on-disk overrides (unknown actions ignored)."""
    result = dict(DEFAULT_KEYBINDINGS)
    p = keybindings_path()
    if p.is_file():
        try:
            saved = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                for action, combo in saved.items():
                    if action in DEFAULT_KEYBINDINGS and isinstance(combo, str) and combo:
                        result[action] = combo
        except (json.JSONDecodeError, OSError):
            pass
    return result


def save_keybindings(mapping: dict) -> dict[str, str]:
    """Validate + persist a keybindings map, returning the merged result."""
    if not isinstance(mapping, dict):
        raise ValueError("keybindings must be an object")
    cleaned = {}
    for action, combo in mapping.items():
        if action not in DEFAULT_KEYBINDINGS:
            raise ValueError(f"unknown action: {action!r}")
        if not isinstance(combo, str) or not combo:
            raise ValueError(f"invalid combo for {action!r}")
        cleaned[action] = combo
    p = keybindings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(p, json.dumps(cleaned, indent=2))
    return load_keybindings()


def safe_name(name: str) -> str:
    if not _NAME_RE.match(name) or ".." in name:
        raise ValueError(f"unsafe path component: {name!r}")
    return name


def session_path(slug: str, sid: str) -> Path:
    return projects_root() / safe_name(slug) / f"{safe_name(sid)}.jsonl"


def decode_slug(slug: str) -> str:
    """Best-effort readable label for a project directory slug."""
    m = re.match(r"^([A-Za-z])--(.*)$", slug)
    if m:
        return f"{m.group(1)}:/" + m.group(2).replace("-", "/")
    return slug.lstrip("-").replace("-", "/")


# ---------------------------------------------------------------- parsing

@dataclass
class Entry:
    raw: str  # exact original line text, no trailing newline
    data: dict | None  # parsed JSON object, None if unparseable

    @property
    def uuid(self):
        return self.data.get("uuid") if self.data else None

    @property
    def parent_uuid(self):
        return self.data.get("parentUuid") if self.data else None

    @property
    def type(self):
        return self.data.get("type") if self.data else None


@dataclass
class SessionDoc:
    path: Path
    entries: list[Entry]
    trailing_newline: bool

    def text(self) -> str:
        s = "\n".join(e.raw for e in self.entries)
        if self.trailing_newline and s:
            s += "\n"
        return s


def load_session(path: Path) -> SessionDoc:
    # read_bytes + decode: no universal-newline translation, exact round-trip
    text = path.read_bytes().decode("utf-8")
    trailing = text.endswith("\n")
    body = text[:-1] if trailing else text
    entries = []
    for line in body.split("\n") if body else []:
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                data = None
        except json.JSONDecodeError:
            data = None
        entries.append(Entry(raw=line, data=data))
    return SessionDoc(path=path, entries=entries, trailing_newline=trailing)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomic-ish binary write; falls back to in-place rewrite on Windows locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Destination may be briefly locked (e.g. SQLite on Windows).
            with open(path, "wb") as f:
                f.write(data)
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------- turns

# User "prompt" entries whose string content is machinery the human never
# typed: local slash-command scaffolding / stdout, and task notifications.
# These must not start a turn or render as a user message.
_SYNTHETIC_USER_RE = re.compile(
    r"^\s*<(?:local-command-[a-z]+"
    r"|command-(?:name|message|args|stdout|contents)"
    r"|task-notification)\b",
    re.IGNORECASE,
)


def is_synthetic_user_text(content) -> bool:
    """True if a user entry's string content is synthetic (not human-typed)."""
    return isinstance(content, str) and bool(_SYNTHETIC_USER_RE.match(content))


def is_human_prompt(data: dict | None) -> bool:
    """True for user entries that begin a turn (real prompts, not tool results)."""
    if not data or data.get("type") != "user":
        return False
    if data.get("isSidechain") or data.get("isMeta"):
        return False
    msg = data.get("message")
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, str):
        return not is_synthetic_user_text(content)
    if isinstance(content, list):
        return not any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        )
    return False


@dataclass
class Turn:
    id: str  # uuid of the human user entry, or HEADER_TURN_ID
    start: int  # first line index, inclusive
    end: int  # last line index, exclusive
    deletable: bool


def _is_turn_start(data: dict | None, fold_synthetic: bool) -> bool:
    """A line begins a turn if it's a real human prompt, or — when synthetic
    entries are *not* folded (the "show all"/original view) — a synthetic user
    entry (slash-command scaffolding, local-command output, task notification)."""
    if is_human_prompt(data):
        return True
    if not fold_synthetic and data and data.get("type") == "user":
        content = (data.get("message") or {}).get("content")
        return is_synthetic_user_text(content)
    return False


def group_turns(entries: list[Entry], fold_synthetic: bool = True) -> list[Turn]:
    starts = [i for i, e in enumerate(entries) if _is_turn_start(e.data, fold_synthetic)]
    if not starts:
        if entries:
            return [Turn(HEADER_TURN_ID, 0, len(entries), False)]
        return []
    # pull contiguous queue-operation lines that precede a prompt into its turn
    adjusted = []
    for i in starts:
        j = i
        while j > 0 and entries[j - 1].type == "queue-operation":
            j -= 1
        adjusted.append((j, i))
    turns: list[Turn] = []
    first_start = adjusted[0][0]
    if first_start > 0:
        turns.append(Turn(HEADER_TURN_ID, 0, first_start, False))
    for k, (start, prompt_idx) in enumerate(adjusted):
        end = adjusted[k + 1][0] if k + 1 < len(adjusted) else len(entries)
        tid = entries[prompt_idx].uuid or f"line-{prompt_idx}"
        turns.append(Turn(tid, start, end, True))
    return turns


# ------------------------------------------------------------- summaries

def _text_of_blocks(content) -> str:
    """Flatten a message content field (str or block list) to displayable text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                parts.append(b.get("text", ""))
            elif bt == "tool_result":
                parts.append(_text_of_blocks(b.get("content")))
        return "\n".join(p for p in parts if p)
    return ""


# Per-block cap for tool inputs/results so a single giant payload (e.g. a
# multi-MB file write) can't bloat the JSON response. Rich text (assistant
# prose, thinking, user prompts) is small and sent in full.
MAX_TOOL_TEXT = 64 * 1024


def _preview(s: str, n: int) -> str:
    """Short single-line-ish preview with an ellipsis (for list/turn headers)."""
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


def _cap(s: str, n: int = MAX_TOOL_TEXT) -> str:
    """Cap a string to n bytes, appending a human-readable truncation marker."""
    s = s or ""
    if len(s) <= n:
        return s
    kb = len(s) // 1024
    return s[:n] + f"\n… (truncated, {kb} KB total)"


def turn_messages(entries: list[Entry], turn: Turn) -> list[dict]:
    """Renderable message list for one turn, in file order.

    The frontend groups this flat list into a desktop-style transcript and
    pairs each tool_use with its tool_result by id. Rich text is sent in full;
    only tool inputs/results are capped (MAX_TOOL_TEXT)."""
    out: list[dict] = []
    for e in entries[turn.start: turn.end]:
        d = e.data
        if d is None:
            out.append({"kind": "raw", "text": _cap(e.raw)})
            continue
        t = d.get("type")
        if t == "user":
            msg = d.get("message") or {}
            content = msg.get("content")
            if is_human_prompt(d) or isinstance(content, str):
                m = {"kind": "user", "text": _text_of_blocks(content)}
                # machinery the human never typed (slash-command scaffolding,
                # local-command output, task notifications): tagged so the UI
                # can hide it. It never starts a turn (see is_human_prompt).
                if is_synthetic_user_text(content):
                    m["synthetic"] = True
                out.append(m)
            elif isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        out.append({
                            "kind": "tool_result",
                            "toolUseId": b.get("tool_use_id"),
                            "isError": bool(b.get("is_error")),
                            "text": _cap(_text_of_blocks(b.get("content"))),
                        })
        elif t == "assistant":
            msg = d.get("message") or {}
            for b in msg.get("content") or []:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    out.append({"kind": "assistant", "text": b.get("text", "")})
                elif bt == "thinking":
                    # some sessions persist only a signature with empty text;
                    # skip those so the transcript isn't littered with empty toggles
                    think = b.get("thinking", "")
                    if think.strip():
                        out.append({"kind": "thinking", "text": think})
                elif bt == "tool_use":
                    out.append({
                        "kind": "tool_use",
                        "id": b.get("id"),
                        "name": b.get("name", "?"),
                        "input": _cap(json.dumps(b.get("input", {}), ensure_ascii=False, indent=2)),
                    })
        elif t == "system":
            out.append({"kind": "system", "text": d.get("subtype", "")})
        # metadata lines (queue-operation, ai-title, ...) are not shown
    return out


def summarize_session(doc: SessionDoc, fold_synthetic: bool = True) -> list[dict]:
    turns = group_turns(doc.entries, fold_synthetic)
    result = []
    for idx, t in enumerate(turns):
        span = doc.entries[t.start: t.end]
        prompt = ""
        timestamp = None
        n_assistant = 0
        n_tools = 0
        for e in span:
            d = e.data
            if d is None:
                continue
            if timestamp is None and d.get("timestamp"):
                timestamp = d["timestamp"]
            if not prompt and d.get("type") == "user":
                content = (d.get("message") or {}).get("content")
                # a real prompt, or (in "show all" turns) the synthetic entry
                if is_human_prompt(d) or is_synthetic_user_text(content):
                    prompt = _text_of_blocks(content)
            if d.get("type") == "assistant":
                n_assistant += 1
                for b in (d.get("message") or {}).get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        n_tools += 1
        nbytes = sum(len(e.raw.encode("utf-8")) + 1 for e in span)
        result.append({
            "id": t.id,
            "index": idx,
            "deletable": t.deletable,
            "timestamp": timestamp,
            "prompt": _preview(prompt if t.deletable else "(session header)", 300),
            "assistantCount": n_assistant,
            "toolCount": n_tools,
            "lineCount": t.end - t.start,
            "bytes": nbytes,
            "messages": turn_messages(doc.entries, t),
        })
    return result


# -------------------------------------------------------------- deletion

def delete_turns(doc: SessionDoc, turn_ids: list[str], fold_synthetic: bool = True) -> list[Entry]:
    """Return a new entry list with the given turns removed and the
    uuid chain repaired. Raises ValueError for unknown/undeletable ids.

    `fold_synthetic` must match the grouping the client saw, so the turn ids
    resolve to the same spans (in "show all" mode synthetic entries are their
    own deletable turns; folded they belong to the preceding turn)."""
    turns = group_turns(doc.entries, fold_synthetic)
    by_id = {t.id: t for t in turns if t.deletable}
    unknown = [tid for tid in turn_ids if tid not in by_id]
    if unknown:
        raise ValueError(f"unknown or undeletable turn ids: {unknown}")

    del_idx: set[int] = set()
    for tid in turn_ids:
        t = by_id[tid]
        del_idx.update(range(t.start, t.end))

    deleted_uuids = {
        e.uuid for i, e in enumerate(doc.entries) if i in del_idx and e.uuid
    }
    parent_map = {e.uuid: e.parent_uuid for e in doc.entries if e.uuid}

    def resolve(uuid):
        seen = set()
        while uuid in deleted_uuids:
            if uuid in seen:
                return None
            seen.add(uuid)
            uuid = parent_map.get(uuid)
        return uuid

    new_entries: list[Entry] = []
    for i, e in enumerate(doc.entries):
        if i in del_idx:
            continue
        d = e.data
        if d is not None:
            patched = None
            if d.get("parentUuid") in deleted_uuids:
                patched = dict(d)
                patched["parentUuid"] = resolve(d["parentUuid"])
            if d.get("leafUuid") in deleted_uuids:
                patched = patched or dict(d)
                patched["leafUuid"] = resolve(d["leafUuid"])
            if patched is not None:
                e = Entry(
                    raw=json.dumps(patched, ensure_ascii=False, separators=(",", ":")),
                    data=patched,
                )
        new_entries.append(e)
    return new_entries


# ---------------------------------------------------------------- listing

def list_projects() -> list[dict]:
    root = projects_root()
    out = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        count = len(list(d.glob("*.jsonl")))
        if count == 0:
            continue
        out.append({"slug": d.name, "label": decode_slug(d.name), "sessionCount": count})
    return out


def list_sessions(slug: str) -> list[dict]:
    d = projects_root() / safe_name(slug)
    out = []
    for p in d.glob("*.jsonl"):
        title = ""
        turn_count = 0
        try:
            doc = load_session(p)
            for e in doc.entries:
                if e.type == "ai-title" and e.data.get("aiTitle"):
                    title = e.data["aiTitle"]
                elif e.type == "summary" and not title and e.data.get("summary"):
                    title = e.data["summary"]
            turns = group_turns(doc.entries)
            turn_count = sum(1 for t in turns if t.deletable)
            if not title:
                for e in doc.entries:
                    if is_human_prompt(e.data):
                        title = _preview(_text_of_blocks((e.data.get("message") or {}).get("content")), 80)
                        break
        except (OSError, UnicodeDecodeError):
            title = "(unreadable)"
        stat = p.stat()
        out.append({
            "sid": p.stem,
            "title": title or "(untitled)",
            "mtime": stat.st_mtime,
            "bytes": stat.st_size,
            "turnCount": turn_count,
        })
    out.sort(key=lambda s: s["mtime"], reverse=True)
    return out


# ------------------------------------------------------------ undo/redo

class History:
    """Snapshot stacks for one session, persisted on disk.

    Supports a single text file (Claude/Gemini) or a named multi-file bundle
    (Grok, Antigravity). Legacy single-file snapshots are `NNNN.jsonl` bytes;
    bundles are directories `NNNN/` containing the named files.
    """

    def __init__(self, slug: str, sid: str, source: str = "claude"):
        self.source = source
        self.dir = backups_root() / safe_name(source) / safe_name(slug) / safe_name(sid)
        self.state_path = self.dir / "state.json"

    def _load(self) -> dict:
        if self.state_path.is_file():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"undo": [], "redo": [], "counter": 0}

    def _save(self, state: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state), encoding="utf-8")

    def _snapshot_bytes(self, state: dict, content: bytes, suffix: str = ".jsonl") -> str:
        state["counter"] += 1
        name = f"{state['counter']:04d}{suffix}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / name).write_bytes(content)
        return name

    def _snapshot_bundle(self, state: dict, files: dict[str, bytes]) -> str:
        state["counter"] += 1
        name = f"{state['counter']:04d}"
        snap = self.dir / name
        snap.mkdir(parents=True, exist_ok=True)
        for rel, data in files.items():
            (snap / rel).write_bytes(data)
        return name

    def _read_bundle(self, name: str) -> dict[str, bytes]:
        snap = self.dir / name
        if not snap.is_dir():
            raise ValueError(f"missing snapshot bundle: {name}")
        return {p.name: p.read_bytes() for p in snap.iterdir() if p.is_file()}

    def status(self) -> dict:
        state = self._load()
        return {"canUndo": bool(state["undo"]), "canRedo": bool(state["redo"])}

    def record_and_write(self, path: Path, new_text: str) -> None:
        state = self._load()
        name = self._snapshot_bytes(state, path.read_bytes())
        state["undo"].append(name)
        state["redo"] = []
        atomic_write(path, new_text)
        self._save(state)

    def record_and_write_bytes(self, path: Path, new_bytes: bytes) -> None:
        state = self._load()
        name = self._snapshot_bytes(state, path.read_bytes(), suffix=".bin")
        state["undo"].append(name)
        state["redo"] = []
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(new_bytes)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self._save(state)

    def record_and_write_bundle(
        self, paths: dict[str, Path], new_data: dict[str, bytes]
    ) -> None:
        """Snapshot current files, then write new_data (rel name → bytes)."""
        state = self._load()
        current = {rel: p.read_bytes() for rel, p in paths.items() if p.is_file()}
        name = self._snapshot_bundle(state, current)
        state["undo"].append(name)
        state["redo"] = []
        for rel, data in new_data.items():
            p = paths[rel]
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(data, str):
                atomic_write(p, data)
            else:
                _atomic_write_bytes(p, data)
        self._save(state)

    def undo(self, path: Path) -> None:
        state = self._load()
        if not state["undo"]:
            raise ValueError("nothing to undo")
        current = self._snapshot_bytes(state, path.read_bytes())
        name = state["undo"].pop()
        restored = (self.dir / name).read_bytes()
        state["redo"].append(current)
        # binary-safe when snapshot is .bin; text otherwise
        if name.endswith(".bin"):
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(restored)
                os.replace(tmp, path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        else:
            atomic_write(path, restored.decode("utf-8"))
        self._save(state)

    def redo(self, path: Path) -> None:
        state = self._load()
        if not state["redo"]:
            raise ValueError("nothing to redo")
        current = self._snapshot_bytes(state, path.read_bytes())
        name = state["redo"].pop()
        restored = (self.dir / name).read_bytes()
        state["undo"].append(current)
        if name.endswith(".bin"):
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(restored)
                os.replace(tmp, path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        else:
            atomic_write(path, restored.decode("utf-8"))
        self._save(state)

    def undo_bundle(self, paths: dict[str, Path]) -> None:
        state = self._load()
        if not state["undo"]:
            raise ValueError("nothing to undo")
        current = {rel: p.read_bytes() for rel, p in paths.items() if p.is_file()}
        cur_name = self._snapshot_bundle(state, current)
        name = state["undo"].pop()
        restored = self._read_bundle(name)
        state["redo"].append(cur_name)
        for rel, data in restored.items():
            p = paths.get(rel)
            if p is None:
                continue
            _atomic_write_bytes(p, data)
        self._save(state)

    def redo_bundle(self, paths: dict[str, Path]) -> None:
        state = self._load()
        if not state["redo"]:
            raise ValueError("nothing to redo")
        current = {rel: p.read_bytes() for rel, p in paths.items() if p.is_file()}
        cur_name = self._snapshot_bundle(state, current)
        name = state["redo"].pop()
        restored = self._read_bundle(name)
        state["undo"].append(cur_name)
        for rel, data in restored.items():
            p = paths.get(rel)
            if p is None:
                continue
            _atomic_write_bytes(p, data)
        self._save(state)


# ------------------------------------------------------------ operations

def check_hash(path: Path, expected: str, product: str = "the app") -> None:
    if file_hash(path) != expected:
        raise ConflictError(
            f"The session file changed on disk (is it open in {product}?). "
            "Reload before editing."
        )


def check_hash_value(actual: str, expected: str, product: str = "the app") -> None:
    if actual != expected:
        raise ConflictError(
            f"The session file changed on disk (is it open in {product}?). "
            "Reload before editing."
        )


def perform_delete(slug: str, sid: str, turn_ids: list[str], expected_hash: str,
                   fold_synthetic: bool = True) -> None:
    path = session_path(slug, sid)
    check_hash(path, expected_hash, "Claude Code")
    doc = load_session(path)
    new_entries = delete_turns(doc, turn_ids, fold_synthetic)
    new_doc = SessionDoc(path=path, entries=new_entries, trailing_newline=doc.trailing_newline)
    History(slug, sid, source="claude").record_and_write(path, new_doc.text())


def perform_undo(slug: str, sid: str, expected_hash: str) -> None:
    path = session_path(slug, sid)
    check_hash(path, expected_hash, "Claude Code")
    History(slug, sid, source="claude").undo(path)


def perform_redo(slug: str, sid: str, expected_hash: str) -> None:
    path = session_path(slug, sid)
    check_hash(path, expected_hash, "Claude Code")
    History(slug, sid, source="claude").redo(path)


def multi_file_hash(paths: list[Path]) -> str:
    """Stable hash over several files (missing files contribute empty)."""
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        if p.is_file():
            h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()
