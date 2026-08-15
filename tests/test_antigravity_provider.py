"""Unit tests for the Antigravity provider (SQLite steps + brain transcripts)."""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

import chatjsoneditor.providers.antigravity as AG
from chatjsoneditor import sessions as S

FIX = Path(__file__).parent / "fixtures" / "antigravity"


@pytest.fixture
def anti(tmp_path, monkeypatch):
    root = tmp_path / "antigravity"
    shutil.copytree(FIX, root)
    monkeypatch.setenv("ANTIGRAVITY_ROOT", str(root))
    monkeypatch.setenv("CHATJSONEDITOR_BACKUPS_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("ANTIGRAVITY_CONFIG_PROJECTS", str(tmp_path / "no-projects"))
    return AG.AntigravityProvider(), root


class Row(dict):
    """Stand-in for sqlite3.Row (only __getitem__ is used)."""


def _steps(*specs) -> list[Row]:
    return [Row(idx=idx, step_type=st, status=3) for idx, st in specs]


# ----------------------------------------------------------- config/projects

def test_config_projects_maps_folder_uris(tmp_path, monkeypatch):
    cfg = tmp_path / "projects"
    cfg.mkdir()
    (cfg / "p1.json").write_text(json.dumps({
        "name": "My Project",
        "projectResources": {"resources": [{"folderUri": "file:///c:/work/My%20Proj"}]},
    }), encoding="utf-8")
    # ignored: reserved name, unparseable file, non-file uri, missing resources
    (cfg / "outside-of-project.json").write_text(json.dumps({"name": "nope"}), encoding="utf-8")
    (cfg / "broken.json").write_text("{ not json", encoding="utf-8")
    (cfg / "p2.json").write_text(json.dumps({
        "projectResources": {"resources": [{"folderUri": "vscode://other"}]},
    }), encoding="utf-8")
    monkeypatch.setenv("ANTIGRAVITY_CONFIG_PROJECTS", str(cfg))

    mapping = AG._config_projects()
    assert mapping["c:/work/my proj"] == "My Project"
    assert all(v == "My Project" for v in mapping.values())


def test_config_projects_when_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_CONFIG_PROJECTS", str(tmp_path / "nope"))
    assert AG._config_projects() == {}


@pytest.mark.parametrize(
    "cwd,projects,expected_slug,expected_label",
    [
        (None, {}, "_unknown", "(unknown project)"),
        ("C:/work/proj", {"c:/work/proj": "Named"}, "C-work-proj", "Named"),
        ("C:\\work\\proj", {"c:\\work\\proj": "Win"}, "C-work-proj", "Win"),
        ("/home/me/deep/proj", {"deep/proj": "Suffix"}, "home-me-deep-proj", "Suffix"),
        ("/home/me/other", {}, "home-me-other", "/home/me/other"),
    ],
)
def test_project_slug_for_cwd(cwd, projects, expected_slug, expected_label):
    slug, label = AG._project_slug_for_cwd(cwd, projects)
    assert (slug, label) == (expected_slug, expected_label)


def test_project_slug_is_truncated():
    slug, _ = AG._project_slug_for_cwd("/" + "a" * 300, {})
    assert len(slug) == 120


# ------------------------------------------------------------------ database

def _make_db(path: Path, steps=((0, 14), (1, 15)), metadata_blob: bytes | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    try:
        con.execute("CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type INTEGER, status INTEGER)")
        con.executemany("INSERT INTO steps VALUES (?, ?, 3)", steps)
        if metadata_blob is not None:
            con.execute("CREATE TABLE trajectory_metadata_blob (id TEXT PRIMARY KEY, data BLOB)")
            con.execute("INSERT INTO trajectory_metadata_blob VALUES ('a', ?)", (metadata_blob,))
        con.commit()
    finally:
        con.close()
    return path


def test_cwd_from_db_variants(tmp_path):
    # no metadata table at all
    assert AG._cwd_from_db(_make_db(tmp_path / "a.db")) is None
    # table present but empty payload
    assert AG._cwd_from_db(_make_db(tmp_path / "b.db", metadata_blob=b"")) is None
    # file:// URI wins, with a Windows drive prefix stripped
    db = _make_db(tmp_path / "c.db", metadata_blob=b"\x00\x12file:///c:/work/My%20Proj\x00")
    assert AG._cwd_from_db(db) == "c:/work/My Proj"
    # otherwise a bare drive-letter path is recovered
    db = _make_db(tmp_path / "d.db", metadata_blob=b"\x00D:\\work\\proj\x00")
    assert AG._cwd_from_db(db) == "D:\\work\\proj"
    # unrecognizable payload
    db = _make_db(tmp_path / "e.db", metadata_blob=b"\x01\x02\x03")
    assert AG._cwd_from_db(db) is None
    # not a database
    broken = tmp_path / "f.db"
    broken.write_bytes(b"not sqlite")
    assert AG._cwd_from_db(broken) is None


def test_step_rows_reports_conflict_for_unreadable_db(tmp_path):
    broken = tmp_path / "broken.db"
    broken.write_bytes(b"not a database")
    with pytest.raises(S.ConflictError):
        AG._step_rows(broken)


# ---------------------------------------------------------------- transcripts

def test_transcript_paths_prefer_transcript_jsonl(anti):
    _, root = anti
    logs = root / "brain" / "ag-sess-1" / ".system_generated" / "logs"
    assert AG._transcript_path("ag-sess-1") == logs / "transcript.jsonl"
    assert AG._transcript_full_path("ag-sess-1") is None
    assert AG._transcript_path("missing-cid") is None

    (logs / "transcript.jsonl").unlink()
    (logs / "transcript_full.jsonl").write_text("", encoding="utf-8")
    assert AG._transcript_path("ag-sess-1") == logs / "transcript_full.jsonl"
    assert AG._transcript_full_path("ag-sess-1") == logs / "transcript_full.jsonl"


def test_load_transcript_skips_unusable_lines(anti):
    _, root = anti
    logs = root / "brain" / "ag-sess-1" / ".system_generated" / "logs"
    (logs / "transcript.jsonl").write_text(
        '{"step_index":0,"type":"USER_INPUT"}\n\nnot json\n[1,2]\n', encoding="utf-8"
    )
    assert AG._load_transcript("ag-sess-1") == [{"step_index": 0, "type": "USER_INPUT"}]
    assert AG._load_transcript("missing-cid") == []


# --------------------------------------------------------------------- turns

def test_user_step_indices_falls_back_to_step_type():
    steps = _steps((0, AG.STEP_USER_INPUT), (1, 15), (2, AG.STEP_USER_INPUT))
    assert AG._user_step_indices(steps, []) == [0, 2]
    # transcript entries without a usable step_index don't override the fallback
    transcript = [{"type": "USER_INPUT"}, {"type": "PLANNER_RESPONSE", "step_index": 1}]
    assert AG._user_step_indices(steps, transcript) == [0, 2]


def test_user_step_indices_dedupes_repeated_step_index():
    transcript = [
        {"type": "USER_INPUT", "step_index": 4},
        {"type": "USER_INPUT", "step_index": 4},
        {"type": "USER_INPUT", "step_index": 7},
    ]
    assert AG._user_step_indices(_steps((4, 14), (7, 14)), transcript) == [4, 7]


def test_turn_ranges_edge_cases():
    assert AG._turn_ranges([], []) == []
    # no user steps → one undeletable header spanning every step
    assert AG._turn_ranges(_steps((3, 15), (5, 15)), []) == [(S.HEADER_TURN_ID, 3, 6)]
    # leading non-user steps become a header turn
    steps = _steps((0, 15), (2, 14), (3, 15), (6, 14))
    assert AG._turn_ranges(steps, [2, 6]) == [
        (S.HEADER_TURN_ID, 0, 2),
        ("ag-2", 2, 6),
        ("ag-6", 6, 7),
    ]
    # a session starting with a user step has no header
    assert AG._turn_ranges(_steps((0, 14), (1, 15)), [0]) == [("ag-0", 0, 2)]


@pytest.mark.parametrize(
    "content,expected",
    [
        ("<USER_REQUEST>\n  hello \n</USER_REQUEST>", "hello"),
        ("plain text", "plain text"),
        ("keep <ADDITIONAL_METADATA>drop</ADDITIONAL_METADATA> keep2", "keepkeep2"),
        ("keep <ADDITIONAL_METADATA>unterminated", "keep"),
        ("", ""),
        (None, ""),
    ],
)
def test_user_request_text(content, expected):
    assert AG._user_request_text(content) == expected


def test_entry_to_msg_message_kinds():
    user = AG._entry_to_msg({
        "type": "USER_INPUT",
        "content": "<USER_REQUEST>\nhi\n</USER_REQUEST>",
        "created_at": "2026-01-01T00:00:00Z",
    })
    assert [m.kind for m in user] == ["user"]
    assert user[0].text == "hi" and user[0].timestamp == "2026-01-01T00:00:00Z"

    planner = AG._entry_to_msg({
        "type": "PLANNER_RESPONSE", "content": "answer", "thinking": "thought",
    })
    assert [m.kind for m in planner] == ["thinking", "assistant"]
    assert AG._entry_to_msg({"type": "MODEL_RESPONSE", "content": "  "}) == []

    for t in ("EPHEMERAL_MESSAGE", "CONVERSATION_HISTORY", "SYSTEM_MESSAGE"):
        msgs = AG._entry_to_msg({"type": t, "content": "noise"})
        assert [m.kind for m in msgs] == ["system"]
        assert msgs[0].synthetic is True
        assert AG._entry_to_msg({"type": t, "content": " "}) == []

    err = AG._entry_to_msg({"type": "ERROR_MESSAGE", "content": ""})
    assert [m.kind for m in err] == ["system"] and err[0].text == "error"


def test_entry_to_msg_action_steps_become_tool_calls():
    msgs = AG._entry_to_msg({"type": "RUN_COMMAND", "content": "ls -la", "step_index": 7})
    assert [m.kind for m in msgs] == ["tool_use", "tool_result"]
    assert msgs[0].id == "RUN_COMMAND-7" and msgs[0].name == "run_command"
    assert msgs[1].tool_use_id == "RUN_COMMAND-7"

    # CODE_ACTION content is only shown as the tool input, never as a result
    code = AG._entry_to_msg({"type": "CODE_ACTION", "content": "diff", "step_index": 1})
    assert [m.kind for m in code] == ["tool_use"]

    # a typeless step still renders, with its metadata as the input payload
    bare = AG._entry_to_msg({"step_index": 2, "created_at": "2026-01-01T00:00:00Z"})
    assert [m.kind for m in bare] == ["tool_use"]
    assert bare[0].name == "action"
    assert json.loads(bare[0].input)["created_at"] == "2026-01-01T00:00:00Z"


def test_build_turns_from_fixture(anti):
    _, root = anti
    db = root / "conversations" / "ag-sess-1.db"
    turns = AG._build_turns("ag-sess-1", db)
    assert [t.id for t in turns] == ["ag-0", "ag-3"]
    assert turns[0].timestamp == "2026-07-01T00:00:00Z"
    assert [m.kind for m in turns[0].messages] == ["user", "system", "thinking", "assistant"]
    assert turns[0].messages[0].text == "first antigravity turn"


def test_build_turns_without_transcript_uses_step_placeholders(tmp_path, monkeypatch):
    root = tmp_path / "antigravity"
    monkeypatch.setenv("ANTIGRAVITY_ROOT", str(root))
    db = _make_db(root / "conversations" / "cid.db", steps=((0, 14), (1, 15), (2, 15)))
    turns = AG._build_turns("cid", db)
    assert [t.id for t in turns] == ["ag-0"]
    assert [m.kind for m in turns[0].messages] == ["system"] * 3
    assert turns[0].messages[0].text == "step 0 type=14"


def test_build_turns_of_headerless_session_without_user_steps(tmp_path, monkeypatch):
    root = tmp_path / "antigravity"
    monkeypatch.setenv("ANTIGRAVITY_ROOT", str(root))
    db = _make_db(root / "conversations" / "cid.db", steps=((0, 15),))
    turns = AG._build_turns("cid", db)
    assert [(t.id, t.deletable) for t in turns] == [(S.HEADER_TURN_ID, False)]


# ------------------------------------------------------------------ provider

def test_list_projects_and_sessions(anti):
    provider, root = anti
    projects = provider.list_projects()
    assert len(projects) == 1
    slug = projects[0]["slug"]
    assert projects[0]["sessionCount"] == 1

    sessions = provider.list_sessions(slug)
    assert [s["sid"] for s in sessions] == ["ag-sess-1"]
    assert sessions[0]["title"] == "first antigravity turn"
    assert sessions[0]["turnCount"] == 2
    assert provider.list_sessions("other-slug") == []
    with pytest.raises(ValueError):
        provider.list_sessions("../escape")


def test_list_sessions_prefers_task_md_title(anti):
    provider, root = anti
    (root / "brain" / "ag-sess-1" / "task.md").write_text(
        "Task: refactor the parser\nmore\n", encoding="utf-8"
    )
    slug = provider.list_projects()[0]["slug"]
    assert provider.list_sessions(slug)[0]["title"] == "Task: refactor the parser"


def test_list_sessions_ignores_wal_files_and_unreadable_dbs(anti):
    provider, root = anti
    conv = root / "conversations"
    (conv / "ag-sess-1.db-wal").write_bytes(b"")
    (conv / "ag-sess-1.db-shm").write_bytes(b"")
    (conv / "broken.db").write_bytes(b"not a database")
    sessions = {s["sid"]: s for s in provider._all_sessions()}
    assert set(sessions) == {"ag-sess-1", "broken"}
    assert sessions["broken"]["turnCount"] == 0
    assert sessions["broken"]["title"] == "broken"


def test_list_projects_when_conversations_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_ROOT", str(tmp_path / "nope"))
    provider = AG.AntigravityProvider()
    assert provider.list_projects() == []
    assert provider.list_sessions("any") == []


def test_resolve_missing_session(anti):
    provider, _ = anti
    with pytest.raises(FileNotFoundError):
        provider.session_payload("slug", "ghost")


def test_session_payload_shape(anti):
    provider, root = anti
    slug = provider.list_projects()[0]["slug"]
    payload = provider.session_payload(slug, "ag-sess-1")
    assert payload["source"] == "antigravity"
    assert payload["canUndo"] is False and payload["canRedo"] is False
    assert payload["bytes"] == (root / "conversations" / "ag-sess-1.db").stat().st_size
    assert [t["prompt"] for t in payload["turns"]] == [
        "first antigravity turn", "second antigravity turn",
    ]


def test_perform_delete_rejects_unknown_turn_and_stale_hash(anti):
    provider, _ = anti
    slug = provider.list_projects()[0]["slug"]
    payload = provider.session_payload(slug, "ag-sess-1")
    with pytest.raises(ValueError):
        provider.perform_delete(slug, "ag-sess-1", ["ag-999"], payload["hash"])
    with pytest.raises(ValueError):
        provider.perform_delete(slug, "ag-sess-1", [S.HEADER_TURN_ID], payload["hash"])
    with pytest.raises(S.ConflictError):
        provider.perform_delete(slug, "ag-sess-1", ["ag-0"], "deadbeef")
    with pytest.raises(S.ConflictError):
        provider.perform_undo(slug, "ag-sess-1", "deadbeef")
    with pytest.raises(S.ConflictError):
        provider.perform_redo(slug, "ag-sess-1", "deadbeef")


def test_delete_undo_redo_keeps_transcript_full_in_sync(anti):
    provider, root = anti
    logs = root / "brain" / "ag-sess-1" / ".system_generated" / "logs"
    full = logs / "transcript_full.jsonl"
    shutil.copy(logs / "transcript.jsonl", full)
    # an unparseable line must survive deletion untouched
    with full.open("a", encoding="utf-8") as f:
        f.write("not json\n")
    original_full = full.read_bytes()

    slug = provider.list_projects()[0]["slug"]
    payload = provider.session_payload(slug, "ag-sess-1")
    provider.perform_delete(slug, "ag-sess-1", ["ag-0"], payload["hash"])

    kept = [l for l in full.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert "not json" in kept
    assert all(json.loads(l)["step_index"] >= 3 for l in kept if l != "not json")

    after = provider.session_payload(slug, "ag-sess-1")
    assert [t["prompt"] for t in after["turns"]] == ["second antigravity turn"]

    provider.perform_undo(slug, "ag-sess-1", after["hash"])
    assert full.read_bytes() == original_full
    restored = provider.session_payload(slug, "ag-sess-1")
    provider.perform_redo(slug, "ag-sess-1", restored["hash"])
    assert len(provider.session_payload(slug, "ag-sess-1")["turns"]) == 1


def test_perform_delete_session_archives_and_clears_brain(anti):
    provider, root = anti
    slug = provider.list_projects()[0]["slug"]
    db = root / "conversations" / "ag-sess-1.db"
    brain = root / "brain" / "ag-sess-1"
    logs = brain / ".system_generated" / "logs"
    shutil.copy(logs / "transcript.jsonl", logs / "transcript_full.jsonl")

    provider.perform_delete_session(slug, "ag-sess-1")
    assert not db.exists() and not brain.exists()
    archive = next(
        (S.backups_root() / "antigravity" / slug / "ag-sess-1").glob("deleted-*")
    )
    assert {p.name for p in archive.iterdir()} == {
        "ag-sess-1.db", "transcript.jsonl", "transcript_full.jsonl",
    }
