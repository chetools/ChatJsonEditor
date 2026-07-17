"""FastAPI server for the Claude Code session editor."""
from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import sessions as S

app = FastAPI(title="ChatJsonEditor")

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# serve vendored JS/CSS (marked, DOMPurify, MathJax) and any other static assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/projects")
def projects():
    return S.list_projects()


@app.get("/api/projects/{slug}/sessions")
def project_sessions(slug: str):
    try:
        return S.list_sessions(slug)
    except ValueError as e:
        raise HTTPException(400, str(e))


def _session_payload(slug: str, sid: str, fold_synthetic: bool = True) -> dict:
    path = S.session_path(slug, sid)
    if not path.is_file():
        raise HTTPException(404, f"no such session: {sid}")
    doc = S.load_session(path)
    return {
        "slug": slug,
        "sid": sid,
        "hash": S.file_hash(path),
        "bytes": path.stat().st_size,
        "turns": S.summarize_session(doc, fold_synthetic),
        **S.History(slug, sid).status(),
    }


@app.get("/api/sessions/{slug}/{sid}")
def session_detail(slug: str, sid: str, showAll: bool = False):
    try:
        return _session_payload(slug, sid, fold_synthetic=not showAll)
    except ValueError as e:
        raise HTTPException(400, str(e))


class DeleteBody(BaseModel):
    turnIds: list[str]
    hash: str
    showAll: bool = False


class HashBody(BaseModel):
    hash: str
    showAll: bool = False


def _mutate(fn, slug: str, sid: str, fold_synthetic: bool = True):
    try:
        fn()
    except S.ConflictError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError:
        raise HTTPException(404, "session file not found")
    return _session_payload(slug, sid, fold_synthetic=fold_synthetic)


@app.post("/api/sessions/{slug}/{sid}/delete")
def delete_turns(slug: str, sid: str, body: DeleteBody):
    if not body.turnIds:
        raise HTTPException(400, "no turns selected")
    fold = not body.showAll
    return _mutate(
        lambda: S.perform_delete(slug, sid, body.turnIds, body.hash, fold_synthetic=fold),
        slug, sid, fold_synthetic=fold,
    )


@app.post("/api/sessions/{slug}/{sid}/undo")
def undo(slug: str, sid: str, body: HashBody):
    return _mutate(lambda: S.perform_undo(slug, sid, body.hash), slug, sid,
                   fold_synthetic=not body.showAll)


@app.post("/api/sessions/{slug}/{sid}/redo")
def redo(slug: str, sid: str, body: HashBody):
    return _mutate(lambda: S.perform_redo(slug, sid, body.hash), slug, sid,
                   fold_synthetic=not body.showAll)


@app.get("/api/keybindings")
def get_keybindings():
    return S.load_keybindings()


@app.post("/api/keybindings")
def set_keybindings(mapping: dict):
    try:
        return S.save_keybindings(mapping)
    except ValueError as e:
        raise HTTPException(400, str(e))


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Claude Code session editor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    print(f"ChatJsonEditor running at {url}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
