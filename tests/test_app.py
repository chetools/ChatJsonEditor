"""HTTP-level tests for the FastAPI routes (multi-source + legacy Claude)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FIXTURE = Path(__file__).parent / "fixtures" / "sample.jsonl"
SLUG = "C--test-Project"
SID = "sess1"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient over scratch roots with the Claude fixture installed."""
    projects = tmp_path / "projects"
    proj = projects / SLUG
    proj.mkdir(parents=True)
    shutil.copy(FIXTURE, proj / f"{SID}.jsonl")
    monkeypatch.setenv("CLAUDE_PROJECTS_DIR", str(projects))
    monkeypatch.setenv("CHATJSONEDITOR_BACKUPS_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("CHATJSONEDITOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GROK_SESSIONS_DIR", str(tmp_path / "grok"))
    monkeypatch.setenv("GEMINI_TMP_DIR", str(tmp_path / "gemini"))
    monkeypatch.setenv("ANTIGRAVITY_ROOT", str(tmp_path / "antigravity"))

    from chatjsoneditor.app import app
    with TestClient(app) as c:
        yield c, proj / f"{SID}.jsonl"


def _detail(c, **params):
    r = c.get(f"/api/claude/sessions/{SLUG}/{SID}", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def test_index_serves_html(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_sources_lists_every_provider(client):
    c, _ = client
    r = c.get("/api/sources")
    assert r.status_code == 200
    by_id = {s["id"]: s for s in r.json()}
    assert {"claude", "grok", "grok-build", "gemini", "antigravity"} <= set(by_id)
    assert by_id["claude"]["label"] == "Claude Code"
    assert by_id["claude"]["productName"] == "Claude Code"


def test_unknown_source_is_400(client):
    c, _ = client
    assert c.get("/api/nosuch/projects").status_code == 400
    assert c.get(f"/api/nosuch/projects/{SLUG}/sessions").status_code == 400
    assert c.get(f"/api/nosuch/sessions/{SLUG}/{SID}").status_code == 400


def test_projects_and_sessions_listing(client):
    c, _ = client
    projects = c.get("/api/claude/projects").json()
    assert [p["slug"] for p in projects] == [SLUG]
    assert projects[0]["sessionCount"] == 1
    assert projects[0]["label"] == "C:/test/Project"

    sessions = c.get(f"/api/claude/projects/{SLUG}/sessions").json()
    assert [s["sid"] for s in sessions] == [SID]
    assert sessions[0]["title"] == "Sample session"
    assert sessions[0]["turnCount"] == 3


def test_unsafe_slug_is_400(client):
    c, _ = client
    r = c.get("/api/claude/projects/bad!slug/sessions")
    assert r.status_code == 400
    assert c.get("/api/claude/sessions/bad!slug/sess1").status_code == 400


def test_session_detail_show_all_toggle(client):
    c, _ = client
    folded = _detail(c)
    assert folded["source"] == "claude"
    assert folded["slug"] == SLUG and folded["sid"] == SID
    assert len(folded["hash"]) == 64
    assert folded["bytes"] > 0
    assert folded["canUndo"] is False and folded["canRedo"] is False
    assert [t["id"] for t in folded["turns"] if t["deletable"]] == ["u1", "u3", "u5"]

    shown = _detail(c, showAll=1)
    assert len(shown["turns"]) >= len(folded["turns"])


def test_missing_session_is_404(client):
    c, _ = client
    assert c.get(f"/api/claude/sessions/{SLUG}/nope").status_code == 404


def test_delete_undo_redo_over_http(client):
    c, path = client
    original = path.read_bytes()
    payload = _detail(c)

    r = c.post(
        f"/api/claude/sessions/{SLUG}/{SID}/delete",
        json={"turnIds": ["u3"], "hash": payload["hash"]},
    )
    assert r.status_code == 200, r.text
    after = r.json()
    assert [t["id"] for t in after["turns"] if t["deletable"]] == ["u1", "u5"]
    assert after["canUndo"] is True and after["canRedo"] is False
    assert path.read_bytes() != original

    r = c.post(f"/api/claude/sessions/{SLUG}/{SID}/undo", json={"hash": after["hash"]})
    assert r.status_code == 200, r.text
    undone = r.json()
    assert path.read_bytes() == original
    assert undone["canRedo"] is True

    r = c.post(f"/api/claude/sessions/{SLUG}/{SID}/redo", json={"hash": undone["hash"]})
    assert r.status_code == 200, r.text
    assert [t["id"] for t in r.json()["turns"] if t["deletable"]] == ["u1", "u5"]


def test_delete_requires_turn_ids(client):
    c, _ = client
    r = c.post(
        f"/api/claude/sessions/{SLUG}/{SID}/delete",
        json={"turnIds": [], "hash": _detail(c)["hash"]},
    )
    assert r.status_code == 400


def test_stale_hash_is_409(client):
    c, _ = client
    body = {"turnIds": ["u1"], "hash": "deadbeef"}
    assert c.post(f"/api/claude/sessions/{SLUG}/{SID}/delete", json=body).status_code == 409
    assert c.post(
        f"/api/claude/sessions/{SLUG}/{SID}/undo", json={"hash": "deadbeef"}
    ).status_code == 409
    assert c.post(
        f"/api/claude/sessions/{SLUG}/{SID}/redo", json={"hash": "deadbeef"}
    ).status_code == 409


def test_unknown_turn_id_is_400(client):
    c, _ = client
    r = c.post(
        f"/api/claude/sessions/{SLUG}/{SID}/delete",
        json={"turnIds": ["nope"], "hash": _detail(c)["hash"]},
    )
    assert r.status_code == 400


def test_undo_with_empty_history_is_400(client):
    c, _ = client
    r = c.post(f"/api/claude/sessions/{SLUG}/{SID}/undo", json={"hash": _detail(c)["hash"]})
    assert r.status_code == 400


def test_mutating_a_missing_session_is_404(client):
    c, _ = client
    r = c.post(
        f"/api/claude/sessions/{SLUG}/ghost/delete",
        json={"turnIds": ["u1"], "hash": "x"},
    )
    assert r.status_code == 404


def test_delete_whole_session(client):
    c, path = client
    r = c.delete(f"/api/claude/sessions/{SLUG}/{SID}")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "source": "claude", "slug": SLUG, "sid": SID}
    assert not path.is_file()
    assert c.delete(f"/api/claude/sessions/{SLUG}/{SID}").status_code == 404


def test_delete_whole_session_rejects_unsafe_slug(client):
    c, _ = client
    assert c.delete("/api/claude/sessions/bad!slug/sess1").status_code == 400


def test_legacy_claude_routes(client):
    c, path = client
    assert [p["slug"] for p in c.get("/api/projects").json()] == [SLUG]
    assert [s["sid"] for s in c.get(f"/api/projects/{SLUG}/sessions").json()] == [SID]
    assert c.get("/api/projects/bad!slug/sessions").status_code == 400

    detail = c.get(f"/api/sessions/{SLUG}/{SID}").json()
    assert detail["source"] == "claude"
    assert c.get(f"/api/sessions/{SLUG}/nope").status_code == 404
    assert c.get("/api/sessions/bad!slug/sess1").status_code == 400

    r = c.post(
        f"/api/sessions/{SLUG}/{SID}/delete",
        json={"turnIds": ["u3"], "hash": detail["hash"]},
    )
    assert r.status_code == 200
    after = r.json()
    r = c.post(f"/api/sessions/{SLUG}/{SID}/undo", json={"hash": after["hash"]})
    assert r.status_code == 200
    r = c.post(f"/api/sessions/{SLUG}/{SID}/redo", json={"hash": r.json()["hash"]})
    assert r.status_code == 200

    assert c.delete(f"/api/sessions/{SLUG}/{SID}").status_code == 200
    assert not path.is_file()


def test_keybindings_get_and_post(client):
    c, _ = client
    from chatjsoneditor import sessions as S

    assert c.get("/api/keybindings").json() == S.DEFAULT_KEYBINDINGS

    r = c.post("/api/keybindings", json={"undo": "Ctrl+u"})
    assert r.status_code == 200
    assert r.json()["undo"] == "Ctrl+u"
    assert r.json()["redo"] == S.DEFAULT_KEYBINDINGS["redo"]
    assert c.get("/api/keybindings").json()["undo"] == "Ctrl+u"

    assert c.post("/api/keybindings", json={"frobnicate": "x"}).status_code == 400
    assert c.post("/api/keybindings", json={"undo": ""}).status_code == 400


def test_main_starts_server_without_browser(monkeypatch):
    """`main()` wires argv into uvicorn.run and honours --no-browser."""
    import uvicorn

    from chatjsoneditor import app as app_module

    calls = {}
    monkeypatch.setattr(uvicorn, "run", lambda a, **kw: calls.update(kw, app=a))
    monkeypatch.setattr(
        app_module.webbrowser, "open", lambda *a: calls.setdefault("browser", a)
    )
    monkeypatch.setattr(
        "sys.argv", ["chatjsoneditor", "--port", "9999", "--host", "0.0.0.0", "--no-browser"]
    )

    app_module.main()
    assert calls["app"] is app_module.app
    assert calls["host"] == "0.0.0.0" and calls["port"] == 9999
    assert "browser" not in calls


def test_main_opens_browser_by_default(monkeypatch):
    import uvicorn

    from chatjsoneditor import app as app_module

    timers = []

    class FakeTimer:
        def __init__(self, delay, fn, args=()):
            timers.append((delay, fn, args))

        def start(self):
            pass

    monkeypatch.setattr(uvicorn, "run", lambda a, **kw: None)
    monkeypatch.setattr(app_module.threading, "Timer", FakeTimer)
    monkeypatch.setattr("sys.argv", ["chatjsoneditor"])

    app_module.main()
    assert len(timers) == 1
    _, fn, args = timers[0]
    assert fn is app_module.webbrowser.open
    assert args == ("http://127.0.0.1:8642/",)
