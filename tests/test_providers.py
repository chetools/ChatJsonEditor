"""Tests for multi-source providers (Grok, Gemini, Antigravity)."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from chatjsoneditor import sessions as S

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def multi_env(tmp_path, monkeypatch):
    grok = tmp_path / "grok"
    gemini = tmp_path / "gemini_tmp"
    anti = tmp_path / "antigravity"
    chatgpt = tmp_path / "chatgpt_sessions"
    backups = tmp_path / "backups"
    config = tmp_path / "config"

    shutil.copytree(FIX / "grok", grok)
    shutil.copytree(FIX / "gemini", gemini)
    shutil.copytree(FIX / "antigravity", anti)

    monkeypatch.setenv("GROK_SESSIONS_DIR", str(grok))
    monkeypatch.setenv("GEMINI_TMP_DIR", str(gemini))
    monkeypatch.setenv("ANTIGRAVITY_ROOT", str(anti))
    monkeypatch.setenv("CHATGPT_SESSIONS_DIR", str(chatgpt))
    monkeypatch.setenv("CHATJSONEDITOR_BACKUPS_DIR", str(backups))
    monkeypatch.setenv("CHATJSONEDITOR_CONFIG_DIR", str(config))
    # empty claude projects so legacy listing is empty
    monkeypatch.setenv("CLAUDE_PROJECTS_DIR", str(tmp_path / "claude_projects"))

    # re-import providers so nothing is sticky (providers read env at call time)
    from chatjsoneditor.providers import get_provider, list_sources
    return {
        "grok": get_provider("grok"),
        "grok-build": get_provider("grok-build"),
        "gemini": get_provider("gemini"),
        "antigravity": get_provider("antigravity"),
        "chatgpt": get_provider("chatgpt"),
        "list_sources": list_sources,
        "paths": {"grok": grok, "gemini": gemini, "anti": anti, "chatgpt": chatgpt},
    }


def test_sources_registered(multi_env):
    ids = {s["id"] for s in multi_env["list_sources"]()}
    assert {"claude", "chatgpt", "grok", "grok-build", "gemini", "antigravity"} <= ids


def test_chatgpt_rollout_delete_preserves_raw_lines_and_undo_redo(multi_env):
    """ChatGPT/Codex JSONL is range-deleted without reserializing retained rows."""
    root = multi_env["paths"]["chatgpt"] / "2026" / "09" / "04"
    root.mkdir(parents=True)
    path = root / "rollout-test.jsonl"
    rows = [
        '{"type":"session_meta","payload":{"session_id":"s"}}',
        '{"type":"response_item","timestamp":"2026-09-04T10:00:00Z","payload":{"type":"message","id":"u1","role":"user","content":[{"type":"input_text","text":"first prompt"}]}}',
        'this malformed raw line must survive',
        'null',
        '{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"first reply"}]}}',
        '{"type":"response_item","payload":{"type":"function_call","call_id":"call1","name":"read_file","arguments":"{\\"path\\":\\"a\\"}"}}',
        '{"type":"response_item","payload":{"type":"function_call_output","call_id":"call1","output":"ok"}}',
        '{"type":"response_item","timestamp":"2026-09-04T10:01:00Z","payload":{"type":"message","id":"u2","role":"user","content":[{"type":"input_text","text":"second prompt"}]}}',
        '{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"second reply"}]}}',
    ]
    original = "\r\n".join(rows)  # deliberately no final newline
    path.write_bytes(original.encode("utf-8"))

    provider = multi_env["chatgpt"]
    assert provider.list_projects() == [
        {"slug": "2026-09-04", "label": "2026-09-04", "sessionCount": 1}
    ]
    session = provider.list_sessions("2026-09-04")[0]
    assert session["sid"] == "rollout-test.jsonl"
    payload = provider.session_payload("2026-09-04", session["sid"])
    turns = [turn for turn in payload["turns"] if turn["deletable"]]
    assert [turn["id"] for turn in turns] == ["u1", "u2"]
    assert "first prompt" in turns[0]["prompt"]
    assert {message["kind"] for message in turns[0]["messages"]} >= {
        "user", "assistant", "tool_use", "tool_result"
    }

    provider.perform_delete("2026-09-04", session["sid"], ["u2"], payload["hash"])
    after = path.read_bytes()
    assert after == "\r\n".join(rows[:7]).encode("utf-8")
    assert b"this malformed raw line must survive" in after
    assert not after.endswith(b"\n")
    reduced = provider.session_payload("2026-09-04", session["sid"])
    assert [turn["id"] for turn in reduced["turns"] if turn["deletable"]] == ["u1"]

    provider.perform_undo("2026-09-04", session["sid"], reduced["hash"])
    assert path.read_bytes() == original.encode("utf-8")
    restored = provider.session_payload("2026-09-04", session["sid"])
    provider.perform_redo("2026-09-04", session["sid"], restored["hash"])
    assert path.read_bytes() == after


def test_chatgpt_rejects_unknown_and_header_turn_ids(multi_env):
    root = multi_env["paths"]["chatgpt"] / "2026" / "09" / "04"
    root.mkdir(parents=True)
    path = root / "rollout-test.jsonl"
    path.write_text(
        '{"type":"session_meta","payload":{}}\n'
        '{"type":"response_item","payload":{"type":"message","id":"u1","role":"user","content":"hi"}}\n',
        encoding="utf-8",
    )
    provider = multi_env["chatgpt"]
    payload = provider.session_payload("2026-09-04", path.name)
    with pytest.raises(ValueError):
        provider.perform_delete("2026-09-04", path.name, [S.HEADER_TURN_ID], payload["hash"])
    with pytest.raises(ValueError):
        provider.perform_delete("2026-09-04", path.name, ["missing"], payload["hash"])


def test_grok_vs_build_partition(multi_env):
    g = multi_env["grok"]
    b = multi_env["grok-build"]
    g_projects = {p["slug"] for p in g.list_projects()}
    b_projects = {p["slug"] for p in b.list_projects()}
    assert "proj" in g_projects
    assert "proj-build" in b_projects
    # no cross-listing
    assert "proj-build" not in g_projects
    assert "proj" not in b_projects


def test_grok_list_and_payload(multi_env):
    g = multi_env["grok"]
    sessions = g.list_sessions("proj")
    assert len(sessions) == 1
    assert sessions[0]["sid"] == "sess1"
    assert sessions[0]["turnCount"] == 2
    payload = g.session_payload("proj", "sess1")
    assert payload["source"] == "grok"
    turns = [t for t in payload["turns"] if t["deletable"]]
    assert len(turns) == 2
    assert "first prompt" in turns[0]["prompt"]
    kinds = [m["kind"] for m in turns[0]["messages"]]
    assert "user" in kinds
    assert "assistant" in kinds
    assert "tool_use" in kinds
    assert "tool_result" in kinds


def test_grok_delete_undo_redo(multi_env):
    g = multi_env["grok"]
    payload = g.session_payload("proj", "sess1")
    turns = [t for t in payload["turns"] if t["deletable"]]
    first_id = turns[0]["id"]
    h = payload["hash"]
    chat = multi_env["paths"]["grok"] / "proj" / "sess1" / "chat_history.jsonl"
    updates = multi_env["paths"]["grok"] / "proj" / "sess1" / "updates.jsonl"
    original_chat = chat.read_bytes()
    original_updates = updates.read_bytes()

    g.perform_delete("proj", "sess1", [first_id], h)
    after = g.session_payload("proj", "sess1")
    assert len([t for t in after["turns"] if t["deletable"]]) == 1
    assert "second prompt" in after["turns"][-1]["prompt"]
    assert chat.read_bytes() != original_chat
    # updates also trimmed
    assert updates.read_bytes() != original_updates
    # summary counts
    summary = json.loads(
        (multi_env["paths"]["grok"] / "proj" / "sess1" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["num_chat_messages"] == len(chat.read_text(encoding="utf-8").splitlines())

    g.perform_undo("proj", "sess1", after["hash"])
    assert chat.read_bytes() == original_chat
    assert updates.read_bytes() == original_updates

    restored = g.session_payload("proj", "sess1")
    g.perform_redo("proj", "sess1", restored["hash"])
    assert len([t for t in g.session_payload("proj", "sess1")["turns"] if t["deletable"]]) == 1


def test_grok_stale_hash(multi_env):
    g = multi_env["grok"]
    with pytest.raises(S.ConflictError):
        g.perform_delete("proj", "sess1", ["x"], "deadbeef")


def test_grok_delete_entire_session(multi_env):
    g = multi_env["grok"]
    sess_dir = multi_env["paths"]["grok"] / "proj" / "sess1"
    assert sess_dir.is_dir()
    g.perform_delete_session("proj", "sess1")
    assert not sess_dir.exists()
    assert g.list_sessions("proj") == []


def test_gemini_delete_entire_session(multi_env):
    gem = multi_env["gemini"]
    sessions = gem.list_sessions("hash123")
    assert sessions
    sid = sessions[0]["sid"]
    path = multi_env["paths"]["gemini"] / "hash123" / "chats" / sid
    if not path.is_file():
        # sid may be stem-only in some layouts
        path = multi_env["paths"]["gemini"] / "hash123" / "chats" / (
            sid if sid.endswith((".json", ".jsonl")) else f"{sid}.json"
        )
    assert path.is_file()
    gem.perform_delete_session("hash123", sid)
    assert not path.is_file()
    assert gem.list_sessions("hash123") == []


def test_gemini_list_load_delete(multi_env):
    gem = multi_env["gemini"]
    projects = gem.list_projects()
    assert len(projects) == 1
    assert projects[0]["slug"] == "hash123"
    sessions = gem.list_sessions("hash123")
    assert len(sessions) == 1
    sid = sessions[0]["sid"]
    payload = gem.session_payload("hash123", sid)
    turns = [t for t in payload["turns"] if t["deletable"]]
    assert len(turns) >= 2
    first = turns[0]["id"]
    gem.perform_delete("hash123", sid, [first], payload["hash"])
    after = gem.session_payload("hash123", sid)
    assert len([t for t in after["turns"] if t["deletable"]]) == len(turns) - 1


def test_gemini_malformed_session_refuses_to_rewrite(multi_env):
    """A file we cannot fully parse must not be rewritten from a partial parse."""
    gem = multi_env["gemini"]
    sid = gem.list_sessions("hash123")[0]["sid"]
    path = multi_env["paths"]["gemini"] / "hash123" / "chats" / "session-1.json"
    payload = gem.session_payload("hash123", sid)
    first = [t for t in payload["turns"] if t["deletable"]][0]["id"]

    broken = path.read_text(encoding="utf-8")[:-5]
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(S.CorruptDataError):
        gem.session_payload("hash123", sid)
    with pytest.raises(S.CorruptDataError):
        gem.perform_delete("hash123", sid, [first], S.file_hash(path))
    assert path.read_text(encoding="utf-8") == broken
    # listing still works, just without turn counts
    assert gem.list_sessions("hash123")[0]["turnCount"] == 0


def test_grok_turn_ids_are_stable_across_processes(multi_env):
    g = multi_env["grok"]
    ids = [t["id"] for t in g.session_payload("proj", "sess1")["turns"]]
    code = (
        "import json;from chatjsoneditor.providers import get_provider;"
        "print(json.dumps([t['id'] for t in "
        "get_provider('grok').session_payload('proj','sess1')['turns']]))"
    )
    env = {
        **os.environ,
        "PYTHONHASHSEED": "12345",
        "GROK_SESSIONS_DIR": str(multi_env["paths"]["grok"]),
    }
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, check=True)
    assert json.loads(out.stdout) == ids


def test_grok_unknown_turn_id_rejected(multi_env):
    g = multi_env["grok"]
    payload = g.session_payload("proj", "sess1")
    with pytest.raises(ValueError):
        g.perform_delete("proj", "sess1", ["u0-deadbeef"], payload["hash"])


def test_antigravity_turn_boundaries_use_step_index_not_line_index(multi_env, tmp_path):
    """Regression: gapped step_index must not shift later turns.

    Antigravity transcripts often skip indices (line i has step_index > i).
    Mapping USER_INPUT *line numbers* into the steps table desyncs summary
    prompts from the center pane (user text missing / previous reply attached).
    """
    import chatjsoneditor.providers.antigravity as AG
    from chatjsoneditor.providers.antigravity import _user_step_indices

    # Synthetic transcript with a gap (no step_index 5)
    transcript = [
        {"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>\none\n</USER_REQUEST>"},
        {"step_index": 1, "type": "PLANNER_RESPONSE", "content": "reply one"},
        {"step_index": 6, "type": "USER_INPUT", "content": "<USER_REQUEST>\ntwo\n</USER_REQUEST>"},
        {"step_index": 7, "type": "PLANNER_RESPONSE", "content": "reply two"},
        {"step_index": 10, "type": "USER_INPUT", "content": "<USER_REQUEST>\nthree\n</USER_REQUEST>"},
        {"step_index": 11, "type": "PLANNER_RESPONSE", "content": "reply three"},
    ]
    # steps table idx matches step_index values (with gaps)
    class R(dict):
        def __getitem__(self, k):
            return dict.__getitem__(self, k)
    steps = [R(idx=i, step_type=14 if i in (0, 6, 10) else 15, status=3) for i in (0, 1, 6, 7, 10, 11)]
    idxs = _user_step_indices(steps, transcript)
    assert idxs == [0, 6, 10], idxs

    # Old buggy logic would have used line indices 0,2,4 → steps[2].idx=6, steps[4].idx=10
    # and still look partly ok — use a case where line index != step_index value:
    transcript2 = [
        {"step_index": 0, "type": "USER_INPUT", "content": "a"},
        {"step_index": 2, "type": "PLANNER_RESPONSE", "content": "ra"},
        {"step_index": 3, "type": "USER_INPUT", "content": "b"},  # line 2, si 3
        {"step_index": 4, "type": "PLANNER_RESPONSE", "content": "rb"},
    ]
    steps2 = [R(idx=i, step_type=14 if i in (0, 3) else 15, status=3) for i in (0, 2, 3, 4)]
    assert _user_step_indices(steps2, transcript2) == [0, 3]


def test_antigravity_list_load_delete(multi_env):
    ag = multi_env["antigravity"]
    projects = ag.list_projects()
    assert projects, "expected at least one antigravity project"
    slug = projects[0]["slug"]
    sessions = ag.list_sessions(slug)
    assert len(sessions) == 1
    sid = sessions[0]["sid"]
    assert sid == "ag-sess-1"
    payload = ag.session_payload(slug, sid)
    turns = [t for t in payload["turns"] if t["deletable"]]
    assert len(turns) == 2
    assert "first antigravity" in turns[0]["prompt"]

    db = multi_env["paths"]["anti"] / "conversations" / "ag-sess-1.db"

    def step_count() -> int:
        con = sqlite3.connect(str(db))
        try:
            return con.execute("select count(*) from steps").fetchone()[0]
        finally:
            con.close()

    before_count = step_count()
    assert before_count == 5

    first = turns[0]["id"]
    ag.perform_delete(slug, sid, [first], payload["hash"])
    after = ag.session_payload(slug, sid)
    assert len([t for t in after["turns"] if t["deletable"]]) == 1
    after_count = step_count()
    assert after_count < before_count

    # transcript trimmed
    tpath = multi_env["paths"]["anti"] / "brain" / "ag-sess-1" / ".system_generated" / "logs" / "transcript.jsonl"
    lines = [json.loads(l) for l in tpath.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert all(l.get("step_index") not in (0, 1, 2) for l in lines) or len(lines) < 5

    ag.perform_undo(slug, sid, after["hash"])
    restored_count = step_count()
    assert restored_count == before_count
