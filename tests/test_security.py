"""Guards that keep the unauthenticated local API unreachable from web pages."""
import pytest
from fastapi.testclient import TestClient

from chatjsoneditor.app import app
from chatjsoneditor.sessions import safe_name


@pytest.fixture
def client():
    return TestClient(app, base_url="http://127.0.0.1:8642")


def test_rebound_host_header_rejected(client):
    r = client.get("/api/sources", headers={"host": "evil.example"})
    assert r.status_code == 403


@pytest.mark.parametrize("host", ["127.0.0.1:8642", "localhost:8642", "[::1]:8642"])
def test_loopback_hosts_allowed(client, host):
    assert client.get("/api/sources", headers={"host": host}).status_code == 200


def test_cross_origin_request_rejected(client):
    r = client.post(
        "/api/claude/sessions/slug/sid/delete",
        json={"turnIds": ["x"], "hash": "y"},
        headers={"origin": "https://evil.example"},
    )
    assert r.status_code == 403


def test_same_origin_request_allowed(client):
    r = client.get("/api/sources", headers={"origin": "http://127.0.0.1:8642"})
    assert r.status_code == 200


@pytest.mark.parametrize(
    "name",
    ["..", "../etc", "a/b", "a\\b", "%2e%2e", "%2e%2e%5cwindows", "", "a?b"],
)
def test_safe_name_rejects_traversal(name):
    with pytest.raises(ValueError):
        safe_name(name)


@pytest.mark.parametrize("name", ["sess1", "C--test-Project", "C%3A%5CUsers%5Cme", "a.json"])
def test_safe_name_accepts_real_components(name):
    assert safe_name(name) == name


def test_gemini_session_id_is_validated(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_TMP_DIR", str(tmp_path))
    (tmp_path / "hash123" / "chats").mkdir(parents=True)
    secret = tmp_path / "secret.json"
    secret.write_text("{}", encoding="utf-8")

    from chatjsoneditor.providers.gemini import GeminiProvider

    with pytest.raises(ValueError):
        GeminiProvider()._session_path("hash123", "..\\..\\secret.json")
