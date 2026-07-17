"""Tests for multi-source providers (Grok, Gemini, Antigravity)."""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def multi_env(tmp_path, monkeypatch):
    grok = tmp_path / "grok"
    gemini = tmp_path / "gemini_tmp"
    anti = tmp_path / "antigravity"
    backups = tmp_path / "backups"
    config = tmp_path / "config"

    shutil.copytree(FIX / "grok", grok)
    shutil.copytree(FIX / "gemini", gemini)
    shutil.copytree(FIX / "antigravity", anti)

    monkeypatch.setenv("GROK_SESSIONS_DIR", str(grok))
    monkeypatch.setenv("GEMINI_TMP_DIR", str(gemini))
    monkeypatch.setenv("ANTIGRAVITY_ROOT", str(anti))
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
        "list_sources": list_sources,
        "paths": {"grok": grok, "gemini": gemini, "anti": anti},
    }


def test_sources_registered(multi_env):
    ids = {s["id"] for s in multi_env["list_sources"]()}
    assert {"claude", "grok", "grok-build", "gemini", "antigravity"} <= ids


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
    from chatjsoneditor import sessions as S
    with pytest.raises(S.ConflictError):
        g.perform_delete("proj", "sess1", ["x"], "deadbeef")


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
