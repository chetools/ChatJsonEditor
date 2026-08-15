"""Unit tests for providers/common.py normalization helpers."""
from __future__ import annotations

import json

from chatjsoneditor.providers.common import (
    NormMsg,
    NormTurn,
    cap_tool,
    extract_user_query,
    is_grok_synthetic_user,
    text_from_content,
)
from chatjsoneditor.sessions import MAX_TOOL_TEXT


def test_norm_msg_to_dict_per_kind():
    assert NormMsg(kind="assistant", text="hi").to_dict() == {"kind": "assistant", "text": "hi"}
    assert NormMsg(kind="user", text="hi").to_dict() == {"kind": "user", "text": "hi"}
    assert NormMsg(kind="user", text="<x/>", synthetic=True).to_dict() == {
        "kind": "user", "text": "<x/>", "synthetic": True,
    }
    # synthetic only surfaces for user messages
    assert NormMsg(kind="assistant", text="hi", synthetic=True).to_dict() == {
        "kind": "assistant", "text": "hi",
    }
    assert NormMsg(kind="tool_use", id="t1", name="Bash", input="{}").to_dict() == {
        "kind": "tool_use", "id": "t1", "name": "Bash", "input": "{}",
    }
    # missing name/input get placeholders
    assert NormMsg(kind="tool_use", id="t1").to_dict() == {
        "kind": "tool_use", "id": "t1", "name": "?", "input": "",
    }
    assert NormMsg(kind="tool_result", tool_use_id="t1", text="out", is_error=True).to_dict() == {
        "kind": "tool_result", "toolUseId": "t1", "isError": True, "text": "out",
    }


def test_norm_turn_summary_prefers_the_real_prompt():
    turn = NormTurn(
        id="turn-1",
        deletable=True,
        timestamp="2026-01-01T00:00:00Z",
        messages=[
            NormMsg(kind="user", text="<user_info/>", synthetic=True),
            NormMsg(kind="user", text="real prompt"),
            NormMsg(kind="assistant", text="answer"),
            NormMsg(kind="tool_use", id="t1", name="Bash", input='{"a":1}'),
            NormMsg(kind="tool_result", tool_use_id="t1", text="out"),
        ],
    )
    s = turn.to_summary(3)
    assert s["id"] == "turn-1" and s["index"] == 3
    assert s["deletable"] is True
    assert s["timestamp"] == "2026-01-01T00:00:00Z"
    assert s["prompt"] == "real prompt"
    assert s["assistantCount"] == 1
    assert s["toolCount"] == 1
    assert s["lineCount"] == 5
    assert s["bytes"] > 0
    assert [m["kind"] for m in s["messages"]] == [
        "user", "user", "assistant", "tool_use", "tool_result",
    ]


def test_norm_turn_summary_falls_back_to_synthetic_prompt():
    turn = NormTurn(
        id="t",
        deletable=True,
        messages=[NormMsg(kind="user", text="<user_info/>", synthetic=True)],
    )
    assert turn.to_summary(0)["prompt"] == "<user_info/>"


def test_norm_turn_summary_of_header_turn():
    turn = NormTurn(id="__header__", deletable=False, messages=[])
    s = turn.to_summary(0)
    assert s["prompt"] == "(session header)"
    assert s["bytes"] == 0 and s["lineCount"] == 0


def test_norm_turn_summary_prompt_is_previewed():
    long_prompt = "x" * 500
    turn = NormTurn(id="t", deletable=True, messages=[NormMsg(kind="user", text=long_prompt)])
    prompt = turn.to_summary(0)["prompt"]
    assert len(prompt) == 300 and prompt.endswith("…")


def test_norm_turn_summary_counts_multibyte_bytes():
    turn = NormTurn(id="t", deletable=True, messages=[NormMsg(kind="user", text="ü")])
    assert turn.to_summary(0)["bytes"] == 3  # 2-byte char + separator


def test_text_from_content_variants():
    assert text_from_content(None) == ""
    assert text_from_content("plain") == "plain"
    assert text_from_content(["a", "b"]) == "a\nb"
    assert text_from_content([{"type": "text", "text": "hello"}]) == "hello"
    assert text_from_content([{"text": "typeless"}]) == "typeless"
    assert text_from_content(
        [{"type": "tool_result", "content": [{"type": "text", "text": "inner"}]}]
    ) == "inner"
    assert text_from_content([{"content": "str content"}]) == "str content"
    assert text_from_content([{"type": "image", "source": {}}]) == ""
    assert text_from_content([1, None]) == ""
    assert text_from_content({"text": "dict text"}) == "dict text"
    assert text_from_content({"content": "dict content"}) == "dict content"
    assert text_from_content({"other": 1}) == ""
    assert text_from_content(42) == "42"


def test_cap_tool_truncates_oversized_payloads():
    assert cap_tool("") == ""
    assert cap_tool(None) == ""
    assert cap_tool("small") == "small"
    big = cap_tool("x" * (MAX_TOOL_TEXT + 10))
    assert big.endswith("KB total)")
    assert len(big) < MAX_TOOL_TEXT + 100


def test_extract_user_query():
    assert extract_user_query("<user_query>\n  do a thing \n</user_query>") == "do a thing"
    assert extract_user_query("<USER_QUERY>caps</USER_QUERY>") == "caps"
    assert extract_user_query("prefix <user_query>q</user_query> suffix") == "q"
    assert extract_user_query("no tags here") is None
    assert extract_user_query("") is None
    assert extract_user_query(None) is None


def test_is_grok_synthetic_user():
    assert is_grok_synthetic_user("")
    assert is_grok_synthetic_user(None)
    assert is_grok_synthetic_user("<user_info>cwd=/tmp</user_info>")
    assert is_grok_synthetic_user("<system-reminder>be nice</system-reminder>")
    assert is_grok_synthetic_user("<agent_info>x</agent_info>")
    assert is_grok_synthetic_user("<environment_details>x</environment_details>")
    # any other leading tag without a user_query is machinery too
    assert is_grok_synthetic_user("  <something_else>x</something_else>")
    # a wrapped real prompt is human, even inside machinery
    assert not is_grok_synthetic_user("<user_info>x</user_info><user_query>hi</user_query>")
    assert not is_grok_synthetic_user("plain typed prompt")


def test_norm_turn_summary_is_json_serializable():
    turn = NormTurn(
        id="t",
        deletable=True,
        messages=[NormMsg(kind="tool_use", id="t1", name="Bash", input=json.dumps({"a": 1}))],
    )
    assert json.loads(json.dumps(turn.to_summary(0)))["messages"][0]["name"] == "Bash"
