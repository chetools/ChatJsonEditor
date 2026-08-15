"""API-level checks that domain failures reach the client with a real status."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chatjsoneditor import sessions as S

FIXTURE = Path(__file__).parent / "fixtures" / "sample.jsonl"


@pytest.fixture
def client(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    proj = projects / "C--test-Project"
    proj.mkdir(parents=True)
    shutil.copy(FIXTURE, proj / "sess1.jsonl")
    monkeypatch.setenv("CLAUDE_PROJECTS_DIR", str(projects))
    monkeypatch.setenv("CHATJSONEDITOR_BACKUPS_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("CHATJSONEDITOR_CONFIG_DIR", str(tmp_path / "config"))
    from chatjsoneditor.app import app
    return TestClient(app, base_url="http://127.0.0.1:8642"), proj / "sess1.jsonl"


def test_unknown_source_is_400(client):
    c, _ = client
    r = c.get("/api/nosuchsource/projects")
    assert r.status_code == 400
    assert r.json()["detail"]


def test_missing_session_is_404(client):
    c, _ = client
    r = c.get("/api/claude/sessions/C--test-Project/nope")
    assert r.status_code == 404


def test_stale_hash_is_409(client):
    c, _ = client
    r = c.post(
        "/api/claude/sessions/C--test-Project/sess1/delete",
        json={"turnIds": ["u1"], "hash": "deadbeef"},
    )
    assert r.status_code == 409
    assert "changed on disk" in r.json()["detail"]


def test_unknown_turn_is_400(client):
    c, path = client
    r = c.post(
        "/api/claude/sessions/C--test-Project/sess1/delete",
        json={"turnIds": ["nope"], "hash": S.file_hash(path)},
    )
    assert r.status_code == 400
    assert "nope" in r.json()["detail"]


def test_undo_without_history_is_400(client):
    c, path = client
    r = c.post(
        "/api/claude/sessions/C--test-Project/sess1/undo",
        json={"hash": S.file_hash(path)},
    )
    assert r.status_code == 400
    assert "nothing to undo" in r.json()["detail"]
