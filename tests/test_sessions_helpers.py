"""Unit tests for sessions.py helpers: path roots, IO, listing and History."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import chatjsoneditor.sessions as S


@pytest.fixture
def roots(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    projects.mkdir()
    monkeypatch.setenv("CLAUDE_PROJECTS_DIR", str(projects))
    monkeypatch.setenv("CHATJSONEDITOR_BACKUPS_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("CHATJSONEDITOR_CONFIG_DIR", str(tmp_path / "config"))
    return projects


# ------------------------------------------------------------- path roots

def test_roots_use_env_overrides(tmp_path, monkeypatch):
    for var, fn in [
        ("CLAUDE_PROJECTS_DIR", S.projects_root),
        ("CHATJSONEDITOR_BACKUPS_DIR", S.backups_root),
        ("CHATJSONEDITOR_CONFIG_DIR", S.config_root),
        ("GROK_SESSIONS_DIR", S.grok_sessions_root),
        ("GEMINI_TMP_DIR", S.gemini_tmp_root),
        ("ANTIGRAVITY_ROOT", S.antigravity_root),
    ]:
        monkeypatch.setenv(var, str(tmp_path / var.lower()))
        assert fn() == tmp_path / var.lower()


def test_roots_fall_back_to_home(monkeypatch):
    for var in [
        "CLAUDE_PROJECTS_DIR",
        "CHATJSONEDITOR_BACKUPS_DIR",
        "CHATJSONEDITOR_CONFIG_DIR",
        "GROK_SESSIONS_DIR",
        "GROK_HOME",
        "GEMINI_TMP_DIR",
        "ANTIGRAVITY_ROOT",
    ]:
        monkeypatch.delenv(var, raising=False)
    home = Path.home()
    assert S.projects_root() == home / ".claude" / "projects"
    assert S.backups_root() == home / ".claude" / "chatjsoneditor-backups"
    assert S.config_root() == home / ".claude" / "chatjsoneditor"
    assert S.grok_sessions_root() == home / ".grok" / "sessions"
    assert S.gemini_tmp_root() == home / ".gemini" / "tmp"
    assert S.antigravity_root() == home / ".gemini" / "antigravity"


def test_grok_root_derived_from_grok_home(tmp_path, monkeypatch):
    monkeypatch.delenv("GROK_SESSIONS_DIR", raising=False)
    monkeypatch.setenv("GROK_HOME", str(tmp_path / "grokhome"))
    assert S.grok_sessions_root() == tmp_path / "grokhome" / "sessions"


@pytest.mark.parametrize("name", ["ok-name", "a.b_c%3A", "0123"])
def test_safe_name_accepts_path_segments(name):
    assert S.safe_name(name) == name


@pytest.mark.parametrize("name", ["..", "a/b", "a\\b", "", "sp ace", "a..b/../c"])
def test_safe_name_rejects_traversal(name):
    with pytest.raises(ValueError):
        S.safe_name(name)


def test_session_path_composition(roots):
    assert S.session_path("C--x", "sid") == roots / "C--x" / "sid.jsonl"


@pytest.mark.parametrize(
    "slug,expected",
    [
        ("C--Users-me-proj", "C:/Users/me/proj"),
        ("-home-me-proj", "home/me/proj"),
        ("plainslug", "plainslug"),
    ],
)
def test_decode_slug(slug, expected):
    assert S.decode_slug(slug) == expected


# -------------------------------------------------------------------- IO

def test_load_session_tolerates_unparseable_and_non_object_lines(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"uuid":"a"}\nnot json\n[1,2]\n', encoding="utf-8")
    doc = S.load_session(p)
    assert [e.data is None for e in doc.entries] == [False, True, True]
    assert doc.trailing_newline is True
    assert doc.text().encode("utf-8") == p.read_bytes()
    # None-data entries expose no chain fields
    assert doc.entries[1].uuid is None
    assert doc.entries[1].parent_uuid is None
    assert doc.entries[1].type is None


def test_load_session_empty_and_no_trailing_newline(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    doc = S.load_session(empty)
    assert doc.entries == [] and doc.text() == ""

    bare = tmp_path / "bare.jsonl"
    bare.write_text('{"uuid":"a"}', encoding="utf-8")
    doc = S.load_session(bare)
    assert doc.trailing_newline is False
    assert doc.text() == '{"uuid":"a"}'


def test_atomic_write_replaces_and_leaves_no_temp_files(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("old", encoding="utf-8")
    S.atomic_write(p, "new\n")
    assert p.read_text(encoding="utf-8") == "new\n"
    assert list(tmp_path.iterdir()) == [p]


def test_atomic_write_cleans_up_temp_on_failure(tmp_path, monkeypatch):
    p = tmp_path / "f.txt"
    p.write_text("old", encoding="utf-8")

    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(S.os, "replace", boom)
    with pytest.raises(OSError):
        S.atomic_write(p, "new")
    assert p.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.iterdir()) == [p]


def test_atomic_write_bytes_creates_parents(tmp_path):
    p = tmp_path / "nested" / "dir" / "f.bin"
    S._atomic_write_bytes(p, b"\x00\xff")
    assert p.read_bytes() == b"\x00\xff"


def test_atomic_write_bytes_falls_back_when_destination_locked(tmp_path, monkeypatch):
    """Windows can hold a lock on the destination; we rewrite in place instead."""
    p = tmp_path / "f.bin"
    p.write_bytes(b"old")

    def locked(*a, **kw):
        raise PermissionError("locked")

    monkeypatch.setattr(S.os, "replace", locked)
    S._atomic_write_bytes(p, b"new")
    assert p.read_bytes() == b"new"
    assert list(tmp_path.iterdir()) == [p]


def test_file_hash_and_multi_file_hash(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    assert S.file_hash(a) != S.file_hash(b)
    h = S.multi_file_hash([a, b])
    assert len(h) == 64
    # order- and content-sensitive; missing files contribute empty content
    assert S.multi_file_hash([b, a]) != h
    b.write_bytes(b"two!")
    assert S.multi_file_hash([a, b]) != h
    assert len(S.multi_file_hash([a, tmp_path / "gone"])) == 64


# --------------------------------------------------------------- previews

def test_preview_and_cap():
    assert S._preview("short", 10) == "short"
    assert S._preview("abcdefghij", 5) == "abcd…"
    assert S._preview(None, 5) == ""
    assert S._cap("abc", 10) == "abc"
    capped = S._cap("x" * 3000, 1024)
    assert capped.startswith("x" * 1024)
    assert capped.endswith("(truncated, 2 KB total)")
    assert S._cap(None) == ""


def test_text_of_blocks_flattens_nested_content():
    assert S._text_of_blocks("plain") == "plain"
    assert S._text_of_blocks(None) == ""
    assert S._text_of_blocks(42) == ""
    blocks = [
        {"type": "text", "text": "hello"},
        "ignored non-dict",
        {"type": "tool_result", "content": [{"type": "text", "text": "inner"}]},
        {"type": "thinking", "thinking": "hidden"},
        {"type": "text", "text": ""},
    ]
    assert S._text_of_blocks(blocks) == "hello\ninner"


# ------------------------------------------------------------------ turns

def test_synthetic_user_detection():
    assert S.is_synthetic_user_text("<local-command-stdout>out</local-command-stdout>")
    assert S.is_synthetic_user_text("<command-name>/foo</command-name>")
    assert S.is_synthetic_user_text("<task-notification>done</task-notification>")
    assert not S.is_synthetic_user_text("just a prompt")
    assert not S.is_synthetic_user_text(["not", "a", "string"])


def test_is_human_prompt_rules():
    assert S.is_human_prompt({"type": "user", "message": {"content": "hi"}})
    assert not S.is_human_prompt(None)
    assert not S.is_human_prompt({"type": "assistant", "message": {"content": "hi"}})
    assert not S.is_human_prompt(
        {"type": "user", "isSidechain": True, "message": {"content": "hi"}}
    )
    assert not S.is_human_prompt({"type": "user", "isMeta": True, "message": {"content": "hi"}})
    assert not S.is_human_prompt(
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t"}]}}
    )
    assert S.is_human_prompt({"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}})
    # a non-dict message has no prompt content
    assert not S.is_human_prompt({"type": "user", "message": "hi"})


def _entry(data: dict) -> S.Entry:
    return S.Entry(raw=json.dumps(data, ensure_ascii=False), data=data)


def test_group_turns_edge_cases():
    assert S.group_turns([]) == []
    # no prompt at all → one undeletable header turn spanning the file
    only_meta = [_entry({"type": "ai-title", "aiTitle": "t"})]
    turns = S.group_turns(only_meta)
    assert len(turns) == 1
    assert turns[0].id == S.HEADER_TURN_ID and not turns[0].deletable
    # leading non-prompt lines that aren't queue-operations become a header turn
    entries = [
        _entry({"type": "summary", "summary": "s"}),
        _entry({"type": "user", "uuid": "u1", "message": {"content": "hi"}}),
    ]
    turns = S.group_turns(entries)
    assert [t.id for t in turns] == [S.HEADER_TURN_ID, "u1"]
    assert (turns[0].start, turns[0].end) == (0, 1)
    assert (turns[1].start, turns[1].end) == (1, 2)


def test_group_turns_uses_line_id_when_uuid_missing():
    entries = [_entry({"type": "user", "message": {"content": "hi"}})]
    assert S.group_turns(entries)[0].id == "line-0"


def test_group_turns_show_all_splits_synthetic_entries():
    entries = [
        _entry({"type": "user", "uuid": "u1", "message": {"content": "real"}}),
        _entry({
            "type": "user",
            "uuid": "s1",
            "message": {"content": "<local-command-stdout>out</local-command-stdout>"},
        }),
    ]
    assert [t.id for t in S.group_turns(entries, fold_synthetic=True)] == ["u1"]
    assert [t.id for t in S.group_turns(entries, fold_synthetic=False)] == ["u1", "s1"]


def test_turn_messages_renders_raw_system_and_synthetic():
    entries = [
        S.Entry(raw="not json", data=None),
        _entry({"type": "user", "message": {"content": "real prompt"}}),
        _entry({
            "type": "user",
            "message": {"content": "<command-name>/clear</command-name>"},
        }),
        _entry({"type": "system", "subtype": "compact_boundary"}),
        _entry({"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": "   "},  # blank thinking is skipped
            {"type": "text", "text": "answer"},
            "not-a-dict",
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
        ]}}),
        _entry({"type": "queue-operation", "operation": "enqueue"}),
    ]
    turn = S.Turn(id="x", start=0, end=len(entries), deletable=True)
    msgs = S.turn_messages(entries, turn)
    assert [m["kind"] for m in msgs] == [
        "raw", "user", "user", "system", "assistant", "tool_use",
    ]
    assert msgs[0]["text"] == "not json"
    assert "synthetic" not in msgs[1]
    assert msgs[2]["synthetic"] is True
    assert msgs[3]["text"] == "compact_boundary"
    assert json.loads(msgs[5]["input"]) == {"command": "ls"}


def test_summarize_session_of_header_only_file(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"type":"ai-title","aiTitle":"t"}\n', encoding="utf-8")
    turns = S.summarize_session(S.load_session(p))
    assert len(turns) == 1
    assert turns[0]["deletable"] is False
    assert turns[0]["prompt"] == "(session header)"
    assert turns[0]["assistantCount"] == 0 and turns[0]["toolCount"] == 0
    assert turns[0]["bytes"] > 0


def test_summarize_session_counts_tools_and_timestamp(tmp_path):
    p = tmp_path / "s.jsonl"
    lines = [
        {"type": "user", "uuid": "u1", "timestamp": "2026-01-01T00:00:00Z",
         "message": {"content": "prompt"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1", "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}},
            {"type": "text", "text": "hi"},
        ]}},
        "unparseable line",
    ]
    p.write_text(
        "\n".join(l if isinstance(l, str) else json.dumps(l) for l in lines) + "\n",
        encoding="utf-8",
    )
    summary = S.summarize_session(S.load_session(p))
    assert len(summary) == 1
    assert summary[0]["prompt"] == "prompt"
    assert summary[0]["timestamp"] == "2026-01-01T00:00:00Z"
    assert summary[0]["assistantCount"] == 1
    assert summary[0]["toolCount"] == 1
    assert summary[0]["lineCount"] == 3


def test_delete_turns_breaks_parent_cycles(tmp_path):
    """A parentUuid cycle inside a deleted turn must resolve to null, not hang."""
    p = tmp_path / "s.jsonl"
    lines = [
        {"type": "user", "uuid": "u1", "parentUuid": "a1", "message": {"content": "one"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1", "message": {"content": []}},
        {"type": "user", "uuid": "u2", "parentUuid": "a1", "message": {"content": "two"}},
    ]
    p.write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")
    doc = S.load_session(p)
    new_entries = S.delete_turns(doc, ["u1"])
    assert [e.uuid for e in new_entries] == ["u2"]
    assert new_entries[0].parent_uuid is None


# ---------------------------------------------------------------- listing

def test_list_projects_skips_files_and_empty_dirs(roots):
    (roots / "empty").mkdir()
    (roots / "with-session").mkdir()
    (roots / "with-session" / "a.jsonl").write_text("{}\n", encoding="utf-8")
    (roots / "stray.txt").write_text("x", encoding="utf-8")
    projects = S.list_projects()
    assert [p["slug"] for p in projects] == ["with-session"]
    assert projects[0]["sessionCount"] == 1


def test_list_projects_when_root_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_PROJECTS_DIR", str(tmp_path / "nope"))
    assert S.list_projects() == []


def test_list_sessions_titles_and_sorting(roots):
    d = roots / "proj"
    d.mkdir()
    (d / "titled.jsonl").write_text(
        json.dumps({"type": "ai-title", "aiTitle": "AI title"}) + "\n", encoding="utf-8"
    )
    (d / "summarized.jsonl").write_text(
        json.dumps({"type": "summary", "summary": "Summary title"}) + "\n", encoding="utf-8"
    )
    (d / "prompted.jsonl").write_text(
        json.dumps({"type": "user", "uuid": "u1", "message": {"content": "prompt title"}}) + "\n",
        encoding="utf-8",
    )
    (d / "untitled.jsonl").write_text("{}\n", encoding="utf-8")

    mtimes = {"titled": 400, "summarized": 300, "prompted": 200, "untitled": 100}
    import os
    for stem, mtime in mtimes.items():
        os.utime(d / f"{stem}.jsonl", (mtime, mtime))

    sessions = S.list_sessions("proj")
    assert [s["sid"] for s in sessions] == ["titled", "summarized", "prompted", "untitled"]
    by_sid = {s["sid"]: s for s in sessions}
    assert by_sid["titled"]["title"] == "AI title"
    assert by_sid["summarized"]["title"] == "Summary title"
    assert by_sid["prompted"]["title"] == "prompt title"
    assert by_sid["prompted"]["turnCount"] == 1
    assert by_sid["untitled"]["title"] == "(untitled)"


def test_list_sessions_marks_unreadable_files(roots):
    d = roots / "proj"
    d.mkdir()
    (d / "bad.jsonl").write_bytes(b"\xff\xfe not utf-8")
    sessions = S.list_sessions("proj")
    assert [s["title"] for s in sessions] == ["(unreadable)"]


# --------------------------------------------------------------- History

def test_history_status_ignores_corrupt_state(roots):
    h = S.History("proj", "sid")
    assert h.status() == {"canUndo": False, "canRedo": False}
    h.dir.mkdir(parents=True, exist_ok=True)
    h.state_path.write_text("{ not json", encoding="utf-8")
    assert h.status() == {"canUndo": False, "canRedo": False}


def test_history_text_snapshot_roundtrip(roots):
    f = roots / "f.jsonl"
    f.write_text("v1\n", encoding="utf-8")
    h = S.History("proj", "sid")
    h.record_and_write(f, "v2\n")
    assert f.read_text(encoding="utf-8") == "v2\n"
    assert h.status() == {"canUndo": True, "canRedo": False}
    h.undo(f)
    assert f.read_text(encoding="utf-8") == "v1\n"
    assert h.status() == {"canUndo": False, "canRedo": True}
    h.redo(f)
    assert f.read_text(encoding="utf-8") == "v2\n"
    with pytest.raises(ValueError):
        h.redo(f)
    h.undo(f)
    with pytest.raises(ValueError):
        h.undo(f)


def test_history_binary_snapshot_roundtrip(roots):
    f = roots / "f.db"
    f.write_bytes(b"\x00binary-v1")
    h = S.History("proj", "sid", source="antigravity")
    h.record_and_write_bytes(f, b"\x00binary-v2")
    assert f.read_bytes() == b"\x00binary-v2"
    h.undo(f)
    assert f.read_bytes() == b"\x00binary-v1"
    h.redo(f)
    assert f.read_bytes() == b"\x00binary-v2"


def test_history_bundle_roundtrip(roots):
    a = roots / "sess" / "a.jsonl"
    b = roots / "sess" / "b.json"
    a.parent.mkdir(parents=True)
    a.write_bytes(b"a1\n")
    b.write_bytes(b"b1\n")
    paths = {"a.jsonl": a, "b.json": b, "missing.json": roots / "sess" / "missing.json"}

    h = S.History("proj", "sid", source="grok")
    h.record_and_write_bundle(paths, {"a.jsonl": b"a2\n", "b.json": "b2\n"})
    assert a.read_bytes() == b"a2\n"
    assert b.read_text(encoding="utf-8") == "b2\n"

    h.undo_bundle(paths)
    assert (a.read_bytes(), b.read_bytes()) == (b"a1\n", b"b1\n")
    h.redo_bundle(paths)
    assert (a.read_bytes(), b.read_bytes()) == (b"a2\n", b"b2\n")

    with pytest.raises(ValueError):
        h.redo_bundle(paths)
    h.undo_bundle(paths)
    with pytest.raises(ValueError):
        h.undo_bundle(paths)


def test_history_bundle_undo_reports_missing_snapshot(roots):
    f = roots / "sess" / "a.jsonl"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"a1\n")
    paths = {"a.jsonl": f}
    h = S.History("proj", "sid", source="grok")
    h.record_and_write_bundle(paths, {"a.jsonl": b"a2\n"})
    state = json.loads(h.state_path.read_text(encoding="utf-8"))
    state["undo"] = ["9999"]
    h.state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError):
        h.undo_bundle(paths)


def test_history_bundle_ignores_unknown_relative_names(roots):
    """Snapshot entries with no matching path are skipped on restore."""
    f = roots / "sess" / "a.jsonl"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"a1\n")
    h = S.History("proj", "sid", source="grok")
    h.record_and_write_bundle({"a.jsonl": f}, {"a.jsonl": b"a2\n"})
    # simulate a bundle that also holds a file the caller no longer tracks
    snap = h.dir / "0001"
    (snap / "gone.jsonl").write_bytes(b"stale\n")
    h.undo_bundle({"a.jsonl": f})
    assert f.read_bytes() == b"a1\n"
    assert not (roots / "sess" / "gone.jsonl").exists()


# ------------------------------------------------------------- operations

def test_check_hash_and_check_hash_value(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"data")
    S.check_hash(p, S.file_hash(p), "Claude Code")
    with pytest.raises(S.ConflictError) as e:
        S.check_hash(p, "deadbeef", "Grok")
    assert "Grok" in str(e.value)

    S.check_hash_value("abc", "abc")
    with pytest.raises(S.ConflictError):
        S.check_hash_value("abc", "def")


def test_archive_deleted_session_flattens_names(roots):
    dest = S.archive_deleted_session(
        "claude", "proj", "sid", {"nested/dir/file.jsonl": b"content"}
    )
    assert dest.parent == S.backups_root() / "claude" / "proj" / "sid"
    assert dest.name.startswith("deleted-")
    assert (dest / "file.jsonl").read_bytes() == b"content"


def test_perform_delete_session_missing_file(roots):
    (roots / "proj").mkdir()
    with pytest.raises(FileNotFoundError):
        S.perform_delete_session("proj", "ghost")


def test_keybindings_ignores_corrupt_and_unknown_entries(roots):
    p = S.keybindings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not json", encoding="utf-8")
    assert S.load_keybindings() == S.DEFAULT_KEYBINDINGS

    p.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    assert S.load_keybindings() == S.DEFAULT_KEYBINDINGS

    p.write_text(
        json.dumps({"undo": "Ctrl+u", "bogus": "x", "redo": "", "help": 3}),
        encoding="utf-8",
    )
    loaded = S.load_keybindings()
    assert loaded["undo"] == "Ctrl+u"
    assert loaded["redo"] == S.DEFAULT_KEYBINDINGS["redo"]
    assert loaded["help"] == S.DEFAULT_KEYBINDINGS["help"]
    assert "bogus" not in loaded


def test_save_keybindings_rejects_non_dict(roots):
    with pytest.raises(ValueError):
        S.save_keybindings(["undo", "Ctrl+z"])
