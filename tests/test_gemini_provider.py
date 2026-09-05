"""Unit tests for the Gemini CLI provider (format tolerance + editing paths)."""
from __future__ import annotations

import json

import pytest

import chatjsoneditor.providers.gemini as G
from chatjsoneditor import sessions as S


@pytest.fixture
def gem(tmp_path, monkeypatch):
    root = tmp_path / "gemini_tmp"
    root.mkdir()
    monkeypatch.setenv("GEMINI_TMP_DIR", str(root))
    monkeypatch.setenv("CHATJSONEDITOR_BACKUPS_DIR", str(tmp_path / "backups"))
    return G.GeminiProvider(), root


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, str):
        path.write_text(obj, encoding="utf-8")
    else:
        path.write_text(json.dumps(obj), encoding="utf-8")
    return path


# ------------------------------------------------------------- discovery

def test_iter_session_files_skips_noise(gem):
    _, root = gem
    _write(root / "h1" / "chats" / "a.json", {"messages": []})
    _write(root / "h1" / "chats" / "logs.json", {"messages": []})
    _write(root / "h1" / "chats" / "settings.json", {"messages": []})
    _write(root / "h1" / "chats" / ".hidden.json", {"messages": []})
    _write(root / "h1" / "chats" / "notes.txt", "x")
    (root / "h1" / "chats" / "subdir").mkdir()
    # a project without a chats/ dir is scanned directly
    _write(root / "h2" / "flat.jsonl", '{"role":"user","parts":[{"text":"hi"}]}\n')
    _write(root / "stray.json", {"messages": []})  # files at the root are ignored

    found = {(ph, p.name) for ph, p in G._iter_session_files()}
    assert found == {("h1", "a.json"), ("h2", "flat.jsonl")}


def test_iter_session_files_when_root_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_TMP_DIR", str(tmp_path / "nope"))
    assert G._iter_session_files() == []


def test_project_label_prefers_path_marker(gem):
    _, root = gem
    (root / "h1").mkdir()
    assert G._project_label("h1") == "project h1"
    (root / "h1" / ".project_root").write_text("C:/work/proj\nignored\n", encoding="utf-8")
    assert G._project_label("h1") == "C:/work/proj"


def test_project_label_ignores_empty_marker_and_shortens_hash(gem):
    _, root = gem
    long_hash = "0123456789abcdef0123"
    (root / long_hash).mkdir()
    (root / long_hash / "cwd").write_text("   \n", encoding="utf-8")
    assert G._project_label(long_hash) == "project 0123456789ab…"


def test_list_projects_sorted_by_label(gem):
    provider, root = gem
    _write(root / "h1" / "chats" / "a.json", {"messages": []})
    _write(root / "h2" / "chats" / "b.json", {"messages": []})
    _write(root / "h2" / "chats" / "c.json", {"messages": []})
    (root / "h1" / "cwd").write_text("zeta", encoding="utf-8")
    (root / "h2" / "cwd").write_text("alpha", encoding="utf-8")
    projects = provider.list_projects()
    assert [p["label"] for p in projects] == ["alpha", "zeta"]
    assert {p["slug"]: p["sessionCount"] for p in projects} == {"h1": 1, "h2": 2}


# ---------------------------------------------------------------- parsing

def test_load_messages_object_envelope_keys(tmp_path):
    for key in ("messages", "history", "chat", "items"):
        p = _write(tmp_path / f"{key}.json", {key: [{"role": "user"}], "meta": 1})
        messages, envelope = G._load_messages(p)
        assert messages == [{"role": "user"}]
        assert envelope["meta"] == 1


def test_load_messages_list_and_single_object(tmp_path):
    p = _write(tmp_path / "list.json", [{"role": "user"}, "junk"])
    assert G._load_messages(p) == ([{"role": "user"}], {"format": "list"})

    p = _write(tmp_path / "single.json", {"role": "user", "parts": [{"text": "hi"}]})
    messages, envelope = G._load_messages(p)
    assert messages == [envelope]


def test_load_messages_broken_and_unknown_shapes(tmp_path):
    p = _write(tmp_path / "broken.json", "{not json")
    assert G._load_messages(p) == ([], {"format": "broken"})

    # a scalar body doesn't look like JSON at all → treated as (empty) JSONL
    p = _write(tmp_path / "scalar.json", 42)
    assert G._load_messages(p) == ([], {"format": "jsonl"})

    p = _write(tmp_path / "nomessages.json", {"other": 1})
    assert G._load_messages(p) == ([], {"other": 1})


def test_load_messages_jsonl_variants(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(
        "\n".join([
            json.dumps({"role": "user", "parts": [{"text": "hi"}]}),
            "",
            "not json",
            json.dumps({"message": {"role": "model", "parts": [{"text": "yo"}]}}),
            json.dumps({"anything": "else"}),
            json.dumps(["a list line"]),
        ]) + "\n",
        encoding="utf-8",
    )
    messages, envelope = G._load_messages(p)
    assert envelope == {"format": "jsonl"}
    assert messages == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "yo"}]},
        {"anything": "else"},
    ]


def test_load_messages_treats_bare_text_as_jsonl(tmp_path):
    """A .json file whose body isn't a JSON object/array is scanned line-wise."""
    p = tmp_path / "s.json"
    p.write_text('role: user\n', encoding="utf-8")
    assert G._load_messages(p) == ([], {"format": "jsonl"})


@pytest.mark.parametrize(
    "msg,expected",
    [
        ({"role": "user"}, "user"),
        ({"role": "assistant"}, "model"),
        ({"type": "model"}, "model"),
        ({"author": "system"}, "system"),
        ({"role": "TOOL"}, "tool"),
        ({"role": "weird"}, "weird"),
        ({}, "unknown"),
    ],
)
def test_role_of(msg, expected):
    assert G._role_of(msg) == expected


def test_parts_text_variants():
    assert G._parts_text({"parts": ["a", {"text": "b"}, 3]}) == "a\nb"
    assert G._parts_text({"parts": [{"functionCall": {"name": "f"}}]}) == ""
    assert G._parts_text({"parts": [{"functionResponse": {"text": "out"}}]}) == "out"
    # a nested response payload isn't flattened here
    assert G._parts_text(
        {"parts": [{"functionResponse": {"response": {"text": "out"}}}]}
    ) == ""
    assert G._parts_text({"content": "plain"}) == "plain"
    assert G._parts_text({"text": "plain"}) == "plain"
    assert G._parts_text({"message": "plain"}) == "plain"
    assert G._parts_text({}) == ""


def test_msg_to_norm_roles():
    assert [m.kind for m in G._msg_to_norm({"role": "user", "parts": [{"text": "hi"}]})] == ["user"]
    assert [m.kind for m in G._msg_to_norm({"role": "system", "content": "sys"})] == ["system"]
    # whitespace-only system text isn't a system message, but is still shown raw
    assert [m.kind for m in G._msg_to_norm({"role": "system", "content": "  "})] == ["raw"]
    tool = G._msg_to_norm(
        {"role": "tool", "tool_call_id": "c1", "content": "out", "is_error": True}
    )
    assert [m.kind for m in tool] == ["tool_result"]
    assert tool[0].tool_use_id == "c1" and tool[0].is_error is True


def test_msg_to_norm_model_with_function_call_and_thought():
    msgs = G._msg_to_norm({
        "role": "model",
        "thought": "planning",
        "parts": [
            {"text": "answer"},
            {"functionCall": {"id": "fc1", "name": "run", "args": {"a": 1}}},
            "not-a-dict",
        ],
    })
    assert [m.kind for m in msgs] == ["thinking", "assistant", "tool_use"]
    use = msgs[-1]
    assert use.id == "fc1" and use.name == "run"
    assert json.loads(use.input) == {"a": 1}


def test_msg_to_norm_function_response_without_text():
    msgs = G._msg_to_norm({
        "role": "model",
        "parts": [{"functionResponse": {"name": "run", "response": {"text": ""}}}],
    })
    assert [m.kind for m in msgs] == ["tool_result"]
    assert msgs[0].tool_use_id == "run"


def test_msg_to_norm_unknown_role_falls_back_to_raw():
    msgs = G._msg_to_norm({"role": "weird", "content": "text"})
    # a "weird" role is treated as model prose
    assert [m.kind for m in msgs] == ["assistant"]
    assert G._msg_to_norm({"role": "weird"}) == []


def test_group_turns_header_and_ids():
    assert G._group_turns([]) == []
    # no user message at all → single undeletable header turn
    only_model = G._group_turns([{"role": "model", "content": "hi"}])
    assert len(only_model) == 1 and not only_model[0].deletable

    turns = G._group_turns([
        {"role": "system", "content": "preamble"},
        {"role": "user", "content": "one"},
        {"role": "model", "content": "reply"},
        {"role": "user", "id": "m9", "content": "two"},
    ])
    assert [t.id for t in turns] == [S.HEADER_TURN_ID, "g1", "m9"]
    assert [t.deletable for t in turns] == [False, True, True]
    assert [m.kind for m in turns[1].messages] == ["user", "assistant"]


def test_rewrite_file_per_format(tmp_path):
    msgs = [{"role": "user", "content": "hi"}]

    jl = G._rewrite_file(tmp_path / "s.jsonl", {"format": "jsonl"}, msgs)
    assert jl == (json.dumps(msgs[0], ensure_ascii=False) + "\n").encode("utf-8")
    assert G._rewrite_file(tmp_path / "s.jsonl", {"format": "jsonl"}, []) == b""

    lst = G._rewrite_file(tmp_path / "s.json", {"format": "list"}, msgs)
    assert json.loads(lst) == msgs

    obj = G._rewrite_file(tmp_path / "s.json", {"messages": [], "meta": 1}, msgs)
    parsed = json.loads(obj)
    assert parsed["messages"] == msgs and parsed["meta"] == 1
    assert "format" not in parsed

    # "messages" always wins, even when the envelope used another key
    other = json.loads(G._rewrite_file(tmp_path / "s.json", {"history": []}, msgs))
    assert other["messages"] == msgs and other["history"] == []

    # an envelope with no known key gets a messages array
    bare = json.loads(G._rewrite_file(tmp_path / "s.json", {"meta": 1}, msgs))
    assert bare["messages"] == msgs


# -------------------------------------------------------------- provider

def _session(root, name="chat.json", messages=None):
    messages = messages if messages is not None else [
        {"role": "user", "parts": [{"text": "first prompt"}]},
        {"role": "model", "parts": [{"text": "first reply"}]},
        {"role": "user", "parts": [{"text": "second prompt"}]},
        {"role": "model", "parts": [{"text": "second reply"}]},
    ]
    return _write(root / "h1" / "chats" / name, {"messages": messages})


def test_session_path_accepts_stem_or_filename(gem):
    provider, root = gem
    _session(root)
    assert provider._session_path("h1", "chat.json").name == "chat.json"
    assert provider._session_path("h1", "chat").name == "chat.json"
    with pytest.raises(FileNotFoundError):
        provider._session_path("h1", "missing")
    with pytest.raises(ValueError):
        provider._session_path("bad!slug", "chat")


def test_session_path_finds_files_outside_chats_dir(gem):
    provider, root = gem
    _write(root / "h2" / "flat.jsonl", '{"role":"user","parts":[{"text":"hi"}]}\n')
    assert provider._session_path("h2", "flat").name == "flat.jsonl"


def test_list_sessions_titles_and_validation(gem):
    provider, root = gem
    _session(root)
    _write(root / "h1" / "chats" / "empty.json", {"messages": []})
    sessions = {s["sid"]: s for s in provider.list_sessions("h1")}
    assert sessions["chat.json"]["title"] == "first prompt"
    assert sessions["chat.json"]["turnCount"] == 2
    assert sessions["empty.json"]["title"] == "empty"
    assert sessions["empty.json"]["turnCount"] == 0
    assert provider.list_sessions("nosuch") == []
    with pytest.raises(ValueError):
        provider.list_sessions("bad!slug")


def test_session_payload_and_delete_undo_redo(gem):
    provider, root = gem
    path = _session(root)
    original = path.read_bytes()

    payload = provider.session_payload("h1", "chat.json")
    assert payload["source"] == "gemini"
    assert payload["bytes"] == path.stat().st_size
    assert payload["canUndo"] is False
    turns = [t for t in payload["turns"] if t["deletable"]]
    assert [t["prompt"] for t in turns] == ["first prompt", "second prompt"]

    provider.perform_delete("h1", "chat.json", [turns[0]["id"]], payload["hash"])
    after = provider.session_payload("h1", "chat.json")
    remaining = [t for t in after["turns"] if t["deletable"]]
    assert [t["prompt"] for t in remaining] == ["second prompt"]
    assert json.loads(path.read_text(encoding="utf-8"))["messages"][0]["parts"][0]["text"] == (
        "second prompt"
    )

    provider.perform_undo("h1", "chat.json", after["hash"])
    assert path.read_bytes() == original
    restored = provider.session_payload("h1", "chat.json")
    provider.perform_redo("h1", "chat.json", restored["hash"])
    assert len([t for t in provider.session_payload("h1", "chat.json")["turns"]
                if t["deletable"]]) == 1


def test_perform_delete_rejects_unknown_turn_and_stale_hash(gem):
    provider, root = gem
    _session(root)
    payload = provider.session_payload("h1", "chat.json")
    with pytest.raises(ValueError):
        provider.perform_delete("h1", "chat.json", ["nope"], payload["hash"])
    with pytest.raises(S.ConflictError):
        provider.perform_delete("h1", "chat.json", ["g0"], "deadbeef")
    with pytest.raises(S.ConflictError):
        provider.perform_undo("h1", "chat.json", "deadbeef")
    with pytest.raises(S.ConflictError):
        provider.perform_redo("h1", "chat.json", "deadbeef")


def test_perform_delete_on_session_without_user_messages(gem):
    provider, root = gem
    _session(root, messages=[{"role": "model", "parts": [{"text": "hi"}]}])
    payload = provider.session_payload("h1", "chat.json")
    assert [t["deletable"] for t in payload["turns"]] == [False]
    with pytest.raises(ValueError):
        provider.perform_delete("h1", "chat.json", ["g0"], payload["hash"])


def test_perform_delete_session_archives_file(gem):
    provider, root = gem
    path = _session(root)
    provider.perform_delete_session("h1", "chat.json")
    assert not path.is_file()
    archives = list(
        (S.backups_root() / "gemini" / "h1" / "chat.json").glob("deleted-*/chat.json")
    )
    assert archives and archives[0].read_bytes()
