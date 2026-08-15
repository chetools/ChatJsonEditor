"""FastAPI server for multi-source chat session editor."""
from __future__ import annotations

import argparse
import logging
import threading
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import sessions as S
from .providers import get_provider, list_sources

log = logging.getLogger(__name__)

app = FastAPI(title="ChatJsonEditor")

STATIC_DIR = Path(__file__).parent / "static"


# Errors raised anywhere below (routes, providers, sessions) are mapped to a
# status code here instead of being caught and reworded per route, so no
# failure reaches the client as an opaque 500 with an empty body.

def _error(status: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detail})


@app.exception_handler(S.ConflictError)
def _conflict_handler(request: Request, exc: S.ConflictError) -> JSONResponse:
    return _error(409, str(exc))


@app.exception_handler(S.CorruptDataError)
def _corrupt_handler(request: Request, exc: S.CorruptDataError) -> JSONResponse:
    log.warning("%s %s: unparseable session data: %s", request.method, request.url.path, exc)
    return _error(422, str(exc))


@app.exception_handler(FileNotFoundError)
def _not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    return _error(404, str(exc) or "not found")


@app.exception_handler(ValueError)
def _value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return _error(400, str(exc))


@app.exception_handler(OSError)
def _os_error_handler(request: Request, exc: OSError) -> JSONResponse:
    log.exception("%s %s failed", request.method, request.url.path)
    return _error(500, f"filesystem error: {exc}")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/sources")
def sources():
    return list_sources()


def _provider(source: str):
    return get_provider(source)


@app.get("/api/{source}/projects")
def projects(source: str):
    return _provider(source).list_projects()


@app.get("/api/{source}/projects/{slug}/sessions")
def project_sessions(source: str, slug: str):
    return _provider(source).list_sessions(slug)


@app.get("/api/{source}/sessions/{slug}/{sid}")
def session_detail(source: str, slug: str, sid: str, showAll: bool = False):
    return _provider(source).session_payload(slug, sid, fold_synthetic=not showAll)


class DeleteBody(BaseModel):
    turnIds: list[str]
    hash: str
    showAll: bool = False


class HashBody(BaseModel):
    hash: str
    showAll: bool = False


def _mutate(source: str, slug: str, sid: str, fn, fold_synthetic: bool = True):
    fn()
    return _provider(source).session_payload(slug, sid, fold_synthetic=fold_synthetic)


@app.post("/api/{source}/sessions/{slug}/{sid}/delete")
def delete_turns(source: str, slug: str, sid: str, body: DeleteBody):
    if not body.turnIds:
        raise HTTPException(400, "no turns selected")
    fold = not body.showAll
    p = _provider(source)
    return _mutate(
        source, slug, sid,
        lambda: p.perform_delete(slug, sid, body.turnIds, body.hash, fold_synthetic=fold),
        fold_synthetic=fold,
    )


@app.delete("/api/{source}/sessions/{slug}/{sid}")
def delete_session(source: str, slug: str, sid: str):
    """Permanently delete an entire session (archived under backups first)."""
    _provider(source).perform_delete_session(slug, sid)
    return {"ok": True, "source": source, "slug": slug, "sid": sid}


@app.post("/api/{source}/sessions/{slug}/{sid}/undo")
def undo(source: str, slug: str, sid: str, body: HashBody):
    p = _provider(source)
    return _mutate(
        source, slug, sid,
        lambda: p.perform_undo(slug, sid, body.hash),
        fold_synthetic=not body.showAll,
    )


@app.post("/api/{source}/sessions/{slug}/{sid}/redo")
def redo(source: str, slug: str, sid: str, body: HashBody):
    p = _provider(source)
    return _mutate(
        source, slug, sid,
        lambda: p.perform_redo(slug, sid, body.hash),
        fold_synthetic=not body.showAll,
    )


# ---- Legacy Claude-only routes (default source=claude) ----

@app.get("/api/projects")
def legacy_projects():
    return _provider("claude").list_projects()


@app.get("/api/projects/{slug}/sessions")
def legacy_project_sessions(slug: str):
    return _provider("claude").list_sessions(slug)


@app.get("/api/sessions/{slug}/{sid}")
def legacy_session_detail(slug: str, sid: str, showAll: bool = False):
    return _provider("claude").session_payload(slug, sid, fold_synthetic=not showAll)


@app.post("/api/sessions/{slug}/{sid}/delete")
def legacy_delete(slug: str, sid: str, body: DeleteBody):
    return delete_turns("claude", slug, sid, body)


@app.delete("/api/sessions/{slug}/{sid}")
def legacy_delete_session(slug: str, sid: str):
    return delete_session("claude", slug, sid)


@app.post("/api/sessions/{slug}/{sid}/undo")
def legacy_undo(slug: str, sid: str, body: HashBody):
    return undo("claude", slug, sid, body)


@app.post("/api/sessions/{slug}/{sid}/redo")
def legacy_redo(slug: str, sid: str, body: HashBody):
    return redo("claude", slug, sid, body)


@app.get("/api/keybindings")
def get_keybindings():
    return S.load_keybindings()


@app.post("/api/keybindings")
def set_keybindings(mapping: dict):
    return S.save_keybindings(mapping)


def main():
    import uvicorn

    parser = argparse.ArgumentParser(
        description="Local multi-source chat session editor "
        "(Claude Code, Grok, Grok Build, Gemini CLI, Antigravity)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    parser.add_argument(
        "--log-level", default="info",
        choices=("debug", "info", "warning", "error"),
        help="verbosity of app logs (server access logs stay quiet)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    print(f"ChatJsonEditor running at {url}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
