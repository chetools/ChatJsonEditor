"""Unit tests for the Grok / Grok Build provider helpers and editing paths."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import chatjsoneditor.providers.grok as GK
from chatjsoneditor import sessions as S

FIX = Path(__file__).parent / "fixtures" / "grok"


@pytest.fixture
def grok(tmp_path, monkeypatch):
    root = tmp_path / "grok"
    shutil.copytree(FIX, root)
    monkeypatch.setenv("GROK_SESSIONS_DIR", str(root))
    monkeypatch.setenv("CHATJSONEDITOR_BACKUPS_DIR", str(tmp_path / "backups"))
    return GK.GrokProvider(), GK.GrokBuildProvider(), root


def _rows(*objs) -> list[tuple[str, dict | None]]:
    return [(json.dumps(o), o) for o in objs]


def _user_chunk(text: str) -> dict:
    return {"params": {"update": {
        "sessionUpdate": "user_message_chunk",
        "content": {"type": "text", "text": text},
    }}}


def _other_update(kind: str = "agent_message_chunk") -> dict:
    return {"params": {"update": {"sessionUpdate": kind}}}


# ------------------------------------------------------------ classification

@pytest.mark.parametrize(
    "summary,is_build",
    [
        ({"agent_name": "grok-build"}, True),
        ({"agent_name": "GROK-BUILD-PLAN"}, True),
        ({"agent_name": "build"}, True),
        ({"agent_name": "grok-build-something"}, True),
        ({"current_model_id": "grok-build-fast-1"}, True),
        ({"agent_name": "cursor", "current_model_id": "grok-4.5"}, False),
        ({}, False),
    ],
)
def test_is_build_session(summary, is_build):
    assert GK._is_build_session(summary) is is_build


def test_session_dirs_skips_stray_files_and_broken_summaries(tmp_path):
    root = tmp_path / "grok"
    (root / "proj" / "sess1").mkdir(parents=True)
    (root / "proj" / "sess1" / "summary.json").write_text("{}", encoding="utf-8")
    (root / "proj" / "sess-broken").mkdir()
    (root / "proj" / "sess-broken" / "summary.json").write_text("{ nope", encoding="utf-8")
    (root / "proj" / "sess-nosummary").mkdir()
    (root / "proj" / "prompt_history.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "stray.jsonl").write_text("{}\n", encoding="utf-8")

    found = {(slug, sess.name, tuple(summary)) for slug, sess, summary in GK._session_dirs(root)}
    assert found == {("proj", "sess1", ()), ("proj", "sess-broken", ())}
    assert GK._session_dirs(tmp_path / "missing") == []


def test_cwd_label_prefers_summary_then_decodes_slug():
    assert GK._cwd_label("x", {"info": {"cwd": "C:\\work"}}) == "C:\\work"
    assert GK._cwd_label("C%3A%5Cwork", {}) == "C:\\work"
    assert GK._cwd_label("plain", {"info": {}}) == "plain"


# -------------------------------------------------------------------- jsonl

def test_load_and_write_jsonl_roundtrip(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text('{"a":1}\nnot json\n[1]\n', encoding="utf-8")
    rows = GK._load_jsonl(p)
    assert [data for _, data in rows] == [{"a": 1}, None, None]
    assert GK._write_jsonl(rows) == p.read_bytes()

    assert GK._load_jsonl(tmp_path / "missing.jsonl") == []
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert GK._load_jsonl(empty) == []
    assert GK._write_jsonl([]) == b""
    assert GK._write_jsonl(rows, trailing=False) == p.read_bytes().rstrip(b"\n")


# ----------------------------------------------------------- chat normalizing

def test_is_real_chat_user_and_turn_starts():
    real = {"type": "user", "content": "<user_query>hi</user_query>"}
    synthetic = {"type": "user", "content": "<user_info>ws</user_info>"}
    typed = {"type": "user", "content": "typed directly"}
    assert GK._is_real_chat_user(real)
    assert GK._is_real_chat_user(typed)
    assert not GK._is_real_chat_user(synthetic)
    assert not GK._is_real_chat_user({"type": "assistant", "content": "x"})

    assert GK._is_turn_start_chat(real, True)
    assert GK._is_turn_start_chat(typed, True)
    assert not GK._is_turn_start_chat(synthetic, True)
    assert GK._is_turn_start_chat(synthetic, False)
    assert not GK._is_turn_start_chat({"type": "system", "content": "x"}, False)


def test_messages_from_chat_span_covers_every_line_type():
    rows = [("garbage line", None)] + _rows(
        {"type": "system", "content": "You are Grok."},
        {"type": "system", "content": "  "},
        {"type": "user", "content": "<user_query>real</user_query>"},
        {"type": "user", "content": "<user_info>ws</user_info>"},
        {"type": "reasoning", "summary": [
            {"type": "summary_text", "text": "step one"},
            {"type": "other", "text": "skipped"},
            "not-a-dict",
        ]},
        {"type": "reasoning", "text": "plain thinking"},
        {"type": "reasoning", "summary": []},
        {"type": "assistant", "content": "prose", "tool_calls": [
            {"id": "c1", "name": "list_dir", "arguments": '{"a":1}'},
            {"id": "c2", "name": "run", "arguments": "not json"},
            {"id": "c3", "name": "run", "input": {"b": 2}},
            "not-a-dict",
        ]},
        {"type": "assistant", "content": [{"type": "text", "text": "block prose"}]},
        {"type": "assistant", "content": "   "},
        {"type": "tool_result", "tool_use_id": "c1", "isError": True, "content": "out"},
        {"type": "backend_tool_call", "id": "b1", "kind": {"tool_type": "web_search"}},
        {"type": "backend_tool_call", "kind": {}},
        {"type": "unknown_type", "content": "ignored"},
    )
    msgs = GK._messages_from_chat_span(rows, 0, len(rows))
    assert [m.kind for m in msgs] == [
        "raw", "system", "user", "user",
        "thinking", "thinking",
        "assistant", "tool_use", "tool_use", "tool_use",
        "assistant",
        "tool_result", "tool_use", "tool_use",
    ]
    by_kind = {}
    for m in msgs:
        by_kind.setdefault(m.kind, []).append(m)
    assert by_kind["user"][0].text == "real" and by_kind["user"][0].synthetic is False
    assert by_kind["user"][1].synthetic is True
    assert by_kind["thinking"][0].text == "step one"
    assert by_kind["thinking"][1].text == "plain thinking"
    assert json.loads(by_kind["tool_use"][0].input) == {"a": 1}
    assert by_kind["tool_use"][1].input == "not json"
    assert json.loads(by_kind["tool_use"][2].input) == {"b": 2}
    assert by_kind["tool_result"][0].is_error is True
    assert by_kind["tool_use"][3].name == "web_search"
    assert by_kind["tool_use"][4].name == "backend_tool"


def test_group_chat_turns_header_and_ids():
    assert GK._group_chat_turns([], True) == []
    # no user line at all → one undeletable header turn
    grouped = GK._group_chat_turns(_rows({"type": "assistant", "content": "hi"}), True)
    assert [t.id for t, _ in grouped] == [S.HEADER_TURN_ID]

    rows = _rows(
        {"type": "system", "content": "preamble"},
        {"type": "user", "content": "<user_query>one</user_query>"},
        {"type": "assistant", "content": "reply"},
        {"type": "user", "content": "<user_info>ws</user_info>"},
        {"type": "user", "content": "<user_query>two</user_query>"},
    )
    folded = GK._group_chat_turns(rows, True)
    assert [t.deletable for t, _ in folded] == [False, True, True]
    assert [(t.start, t.end) for t, _ in folded] == [(0, 1), (1, 4), (4, 5)]

    shown = GK._group_chat_turns(rows, False)
    # the synthetic user line becomes its own deletable turn
    assert [(t.start, t.end) for t, _ in shown] == [(0, 1), (1, 3), (3, 4), (4, 5)]
    assert all(t.deletable for t, _ in shown[1:])


def test_group_update_turns_merges_consecutive_user_chunks():
    rows = _rows(
        _other_update("agent_message_chunk"),
        _user_chunk("first "),
        _user_chunk("prompt"),
        _other_update("turn_completed"),
        _user_chunk("second prompt"),
        _other_update(),
    )
    rows.insert(0, ("garbage", None))
    spans = GK._group_update_turns(rows)
    assert spans == [(2, 5, "first prompt"), (5, 7, "second prompt")]
    assert GK._group_update_turns(_rows(_other_update())) == []


def test_group_update_turns_handles_missing_content():
    rows = _rows(
        {"params": {"update": {"sessionUpdate": "user_message_chunk", "content": "str"}}},
        {"params": {"update": {"sessionUpdate": "user_message_chunk"}}},
    )
    assert GK._group_update_turns(rows) == [(0, 2, "")]


def test_delete_chat_turns_validates_ids():
    rows = _rows(
        {"type": "user", "content": "<user_query>one</user_query>"},
        {"type": "assistant", "content": "reply"},
        {"type": "user", "content": "<user_query>two</user_query>"},
    )
    ids = GK._real_prompt_orders(rows, True)
    assert len(ids) == 2
    kept = GK._delete_chat_turns(rows, [ids[0]], True)
    assert [data["content"][0]["text"] if isinstance(data["content"], list) else data["content"]
            for _, data in kept] == ["<user_query>two</user_query>"]
    with pytest.raises(ValueError):
        GK._delete_chat_turns(rows, ["nope"], True)


def test_delete_update_turns_by_order():
    rows = _rows(_user_chunk("one"), _other_update(), _user_chunk("two"), _other_update())
    assert len(GK._delete_update_turns_by_order(rows, {0})) == 2
    assert GK._delete_update_turns_by_order(rows, set()) == rows
    # a file with no user chunks is returned untouched
    plain = _rows(_other_update())
    assert GK._delete_update_turns_by_order(plain, {0}) == plain


# ------------------------------------------------------------------ provider

def test_projects_and_sessions_are_partitioned(grok):
    g, b, _ = grok
    assert [p["slug"] for p in g.list_projects()] == ["proj"]
    assert [p["slug"] for p in b.list_projects()] == ["proj-build"]
    assert g.list_projects()[0]["label"] == "C:\\test\\Project"

    sessions = g.list_sessions("proj")
    assert [s["sid"] for s in sessions] == ["sess1"]
    assert sessions[0]["title"] == "Test Grok session"
    assert sessions[0]["model"] == "grok-4.5" and sessions[0]["agent"] == "cursor"
    assert sessions[0]["bytes"] > 0
    assert g.list_sessions("proj-build") == []
    with pytest.raises(ValueError):
        g.list_sessions("bad!slug")


def test_list_sessions_falls_back_to_dir_name_for_title(grok):
    g, _, root = grok
    (root / "proj" / "sess2").mkdir()
    (root / "proj" / "sess2" / "summary.json").write_text("{}", encoding="utf-8")
    titles = {s["sid"]: s["title"] for s in g.list_sessions("proj")}
    assert titles["sess2"] == "sess2"


def test_sess_dir_enforces_classification(grok):
    g, b, _ = grok
    with pytest.raises(FileNotFoundError):
        g._sess_dir("proj-build", "sess-build")
    with pytest.raises(FileNotFoundError):
        b._sess_dir("proj", "sess1")
    with pytest.raises(FileNotFoundError):
        g._sess_dir("proj", "ghost")
    assert b._sess_dir("proj-build", "sess-build").name == "sess-build"


def test_sess_dir_tolerates_broken_summary(grok):
    g, _, root = grok
    d = root / "proj" / "sess2"
    d.mkdir()
    (d / "summary.json").write_text("{ broken", encoding="utf-8")
    # unparseable summary → treated as non-build, so the Grok provider accepts it
    assert g._sess_dir("proj", "sess2") == d


def test_session_payload_and_summary_patch(grok):
    g, _, root = grok
    sess = root / "proj" / "sess1"
    payload = g.session_payload("proj", "sess1")
    assert payload["source"] == "grok"
    turns = [t for t in payload["turns"] if t["deletable"]]
    assert [t["prompt"] for t in turns] == ["first prompt", "second prompt"]

    g.perform_delete("proj", "sess1", [turns[0]["id"]], payload["hash"])
    summary = json.loads((sess / "summary.json").read_text(encoding="utf-8"))
    chat_lines = (sess / "chat_history.jsonl").read_text(encoding="utf-8").splitlines()
    update_lines = (sess / "updates.jsonl").read_text(encoding="utf-8").splitlines()
    assert summary["num_chat_messages"] == len(chat_lines)
    assert summary["num_messages"] == len(update_lines)
    # the session's own metadata survives the patch
    assert summary["generated_title"] == "Test Grok session"


def test_perform_delete_trims_rewind_points(grok):
    g, _, root = grok
    sess = root / "proj" / "sess1"
    rp = sess / "rewind_points.jsonl"
    rp.write_text(
        "\n".join(json.dumps({"prompt_index": i}) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    payload = g.session_payload("proj", "sess1")
    first = [t for t in payload["turns"] if t["deletable"]][0]["id"]
    g.perform_delete("proj", "sess1", [first], payload["hash"])
    kept = [json.loads(l) for l in rp.read_text(encoding="utf-8").splitlines()]
    assert [k["prompt_index"] for k in kept] == [1, 2]

    # rewind_points is part of the snapshot bundle, so undo restores it too
    after = g.session_payload("proj", "sess1")
    g.perform_undo("proj", "sess1", after["hash"])
    restored = [json.loads(l) for l in rp.read_text(encoding="utf-8").splitlines()]
    assert [k["prompt_index"] for k in restored] == [0, 1, 2]


def test_perform_delete_rejects_unknown_turn_and_stale_hash(grok):
    g, _, _ = grok
    payload = g.session_payload("proj", "sess1")
    with pytest.raises(ValueError):
        g.perform_delete("proj", "sess1", ["nope"], payload["hash"])
    with pytest.raises(S.ConflictError):
        g.perform_delete("proj", "sess1", ["nope"], "deadbeef")
    with pytest.raises(S.ConflictError):
        g.perform_undo("proj", "sess1", "deadbeef")
    with pytest.raises(S.ConflictError):
        g.perform_redo("proj", "sess1", "deadbeef")


def test_perform_delete_session_archives_nested_files(grok):
    g, _, root = grok
    sess = root / "proj" / "sess1"
    (sess / "nested").mkdir()
    (sess / "nested" / "extra.json").write_text("{}", encoding="utf-8")
    g.perform_delete_session("proj", "sess1")
    assert not sess.exists()
    archive = next((S.backups_root() / "grok" / "proj" / "sess1").glob("deleted-*"))
    assert "nested__extra.json" in {p.name for p in archive.iterdir()}


def test_show_all_mode_can_delete_synthetic_turns(grok):
    g, _, root = grok
    payload = g.session_payload("proj", "sess1", fold_synthetic=False)
    turns = [t for t in payload["turns"] if t["deletable"]]
    # the <user_info> line is its own turn in show-all mode
    assert len(turns) == 3
    g.perform_delete("proj", "sess1", [turns[0]["id"]], payload["hash"], fold_synthetic=False)
    after = g.session_payload("proj", "sess1", fold_synthetic=False)
    assert len([t for t in after["turns"] if t["deletable"]]) == 2
