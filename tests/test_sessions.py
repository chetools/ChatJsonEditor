import json
import shutil
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "sample.jsonl"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated projects + backups roots with the fixture installed as a session."""
    projects = tmp_path / "projects"
    backups = tmp_path / "backups"
    proj = projects / "C--test-Project"
    proj.mkdir(parents=True)
    shutil.copy(FIXTURE, proj / "sess1.jsonl")
    monkeypatch.setenv("CLAUDE_PROJECTS_DIR", str(projects))
    monkeypatch.setenv("CHATJSONEDITOR_BACKUPS_DIR", str(backups))
    monkeypatch.setenv("CHATJSONEDITOR_CONFIG_DIR", str(tmp_path / "config"))
    import chatjsoneditor.sessions as S
    return S, proj / "sess1.jsonl"


def chain_is_consistent(entries):
    uuids = {e.uuid for e in entries if e.uuid}
    for e in entries:
        if e.uuid and e.parent_uuid is not None:
            assert e.parent_uuid in uuids, f"dangling parentUuid {e.parent_uuid}"


def _tool_ids(entries):
    uses, results = set(), set()
    for e in entries:
        d = e.data or {}
        for b in (d.get("message") or {}).get("content") or []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                uses.add(b["id"])
            elif b.get("type") == "tool_result":
                results.add(b["tool_use_id"])
    return uses, results


def tool_pairing_ok(entries):
    """No tool_result survives without its tool_use (the corruption that
    breaks resume). A trailing orphan tool_use is tolerated — it occurs
    naturally when a human interrupts a pending tool call."""
    uses, results = _tool_ids(entries)
    assert results <= uses, f"orphaned tool_result(s): {results - uses}"


def no_new_orphans(before, after):
    """Deletion must not introduce orphans that weren't already present."""
    bu, br = _tool_ids(before)
    au, ar = _tool_ids(after)
    assert (ar - au) <= (br - bu), "deletion created a new orphaned tool_result"


def test_roundtrip_is_byte_identical(env):
    S, path = env
    doc = S.load_session(path)
    assert doc.text().encode("utf-8") == path.read_bytes()


def test_keybindings_defaults_and_merge(env):
    S, _ = env
    # no file yet → pure defaults
    assert S.load_keybindings() == S.DEFAULT_KEYBINDINGS
    # save a partial override → merged over defaults
    merged = S.save_keybindings({"undo": "Ctrl+u", "deleteSelected": "Backspace"})
    assert merged["undo"] == "Ctrl+u"
    assert merged["deleteSelected"] == "Backspace"
    assert merged["nextTurn"] == S.DEFAULT_KEYBINDINGS["nextTurn"]  # untouched
    # persisted to disk and reloaded
    assert S.load_keybindings()["undo"] == "Ctrl+u"
    # unknown action rejected
    with pytest.raises(ValueError):
        S.save_keybindings({"frobnicate": "x"})
    # empty combo rejected
    with pytest.raises(ValueError):
        S.save_keybindings({"undo": ""})


def test_turn_messages_shape_and_pairing(env):
    S, path = env
    doc = S.load_session(path)
    turns = [t for t in S.group_turns(doc.entries) if t.deletable]
    # turn 1 (u1) has: user, thinking, tool_use(t1), tool_result(t1), assistant
    msgs = S.turn_messages(doc.entries, turns[0])
    kinds = [m["kind"] for m in msgs]
    assert kinds == ["user", "thinking", "tool_use", "tool_result", "assistant"]
    use = next(m for m in msgs if m["kind"] == "tool_use")
    res = next(m for m in msgs if m["kind"] == "tool_result")
    assert use["id"] == "t1" and use["name"] == "Bash"
    assert res["toolUseId"] == "t1" and res["isError"] is False
    # rich text is preserved verbatim, including unicode
    assert msgs[-1]["text"] == "done with turn one — ünïcode ✓"


def test_tool_text_is_capped(env):
    S, path = env
    doc = S.load_session(path)
    turn = [t for t in S.group_turns(doc.entries) if t.deletable][0]
    # craft an oversized tool_result and confirm it gets the truncation marker
    big = "x" * (S.MAX_TOOL_TEXT + 5000)
    entry = S.Entry(raw="", data={
        "type": "user",
        "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": big},
        ]},
    })
    entries = [entry]
    t = S.Turn(id="x", start=0, end=1, deletable=True)
    out = S.turn_messages(entries, t)
    assert out[0]["kind"] == "tool_result"
    assert out[0]["text"].endswith("KB total)")
    assert len(out[0]["text"]) < len(big)


def test_turn_grouping(env):
    S, path = env
    doc = S.load_session(path)
    turns = S.group_turns(doc.entries)
    deletable = [t for t in turns if t.deletable]
    assert len(deletable) == 3
    assert [t.id for t in deletable] == ["u1", "u3", "u5"]
    # queue-operation lines fold into the first turn, so no header block
    assert turns[0].deletable and turns[0].start == 0


def test_delete_middle_turn_repairs_chain(env):
    S, path = env
    doc = S.load_session(path)
    new_entries = S.delete_turns(doc, ["u3"])
    chain_is_consistent(new_entries)
    tool_pairing_ok(new_entries)
    # u5 must now hang off a3 (last surviving uuid of turn 1)
    u5 = next(e for e in new_entries if e.uuid == "u5")
    assert u5.parent_uuid == "a3"
    # sidechain entries of the deleted turn are gone
    assert not any(e.uuid in ("sc1", "sc2", "u3", "a4", "u4", "a5") for e in new_entries)
    # untouched lines are byte-identical to the originals
    originals = {e.uuid: e.raw for e in doc.entries if e.uuid}
    for e in new_entries:
        if e.uuid and e.uuid != "u5":
            assert e.raw == originals[e.uuid]


def test_delete_first_turn_nulls_parent(env):
    S, path = env
    doc = S.load_session(path)
    new_entries = S.delete_turns(doc, ["u1"])
    chain_is_consistent(new_entries)
    tool_pairing_ok(new_entries)
    u3 = next(e for e in new_entries if e.uuid == "u3")
    assert u3.parent_uuid is None
    # the last-prompt line pointing at deleted a3 is remapped to null
    lp = [e for e in new_entries if e.type == "last-prompt"]
    assert all(e.data["leafUuid"] in (None, "a5", "a6") for e in lp)


def test_delete_unknown_turn_rejected(env):
    S, path = env
    doc = S.load_session(path)
    with pytest.raises(ValueError):
        S.delete_turns(doc, ["nope"])


def test_perform_delete_undo_redo_roundtrip(env):
    S, path = env
    original = path.read_bytes()
    h = S.file_hash(path)
    S.perform_delete("C--test-Project", "sess1", ["u3"], h)
    deleted = path.read_bytes()
    assert deleted != original

    S.perform_undo("C--test-Project", "sess1", S.file_hash(path))
    assert path.read_bytes() == original

    S.perform_redo("C--test-Project", "sess1", S.file_hash(path))
    assert path.read_bytes() == deleted

    st = S.History("C--test-Project", "sess1").status()
    assert st["canUndo"] and not st["canRedo"]


def test_stale_hash_conflict(env):
    S, path = env
    with pytest.raises(S.ConflictError):
        S.perform_delete("C--test-Project", "sess1", ["u1"], "deadbeef")


def test_delete_all_turns_leaves_valid_file(env):
    S, path = env
    doc = S.load_session(path)
    new_entries = S.delete_turns(doc, ["u1", "u3", "u5"])
    chain_uuids = [e for e in new_entries if e.uuid]
    assert chain_uuids == []


def test_real_session_if_available():
    """Extra safety net: run invariants against every local real session."""
    import chatjsoneditor.sessions as S
    root = Path.home() / ".claude" / "projects"
    if not root.is_dir():
        pytest.skip("no real sessions")
    checked = 0
    for p in root.glob("*/*.jsonl"):
        doc = S.load_session(p)
        assert doc.text().encode("utf-8") == p.read_bytes(), f"roundtrip failed: {p}"
        turns = S.group_turns(doc.entries)
        deletable = [t for t in turns if t.deletable]
        if len(deletable) >= 3:
            new_entries = S.delete_turns(doc, [deletable[1].id])
            chain_is_consistent(new_entries)
            tool_pairing_ok(new_entries)
            no_new_orphans(doc.entries, new_entries)
        checked += 1
    assert checked > 0
