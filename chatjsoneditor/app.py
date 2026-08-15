"""FastAPI server for multi-source chat session editor."""
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
from .providers import get_provider, list_sources
from .security import LocalOriginGuard, allow_any_host, is_loopback_host

app = FastAPI(title="ChatJsonEditor")
app.add_middleware(LocalOriginGuard)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/sources")
def sources():
    return list_sources()


def _provider(source: str):
    try:
        return get_provider(source)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/{source}/projects")
def projects(source: str):
    return _provider(source).list_projects()


@app.get("/api/{source}/projects/{slug}/sessions")
def project_sessions(source: str, slug: str):
    try:
        return _provider(source).list_sessions(slug)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/{source}/sessions/{slug}/{sid}")
def session_detail(source: str, slug: str, sid: str, showAll: bool = False):
    try:
        return _provider(source).session_payload(slug, sid, fold_synthetic=not showAll)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except S.ConflictError as e:
        raise HTTPException(409, str(e))


class DeleteBody(BaseModel):
    turnIds: list[str]
    hash: str
    showAll: bool = False


class HashBody(BaseModel):
    hash: str
    showAll: bool = False


def _mutate(source: str, slug: str, sid: str, fn, fold_synthetic: bool = True):
    try:
        fn()
    except S.ConflictError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError:
        raise HTTPException(404, "session file not found")
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
    p = _provider(source)
    try:
        p.perform_delete_session(slug, sid)
    except FileNotFoundError:
        raise HTTPException(404, "session file not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
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
    try:
        return _provider("claude").list_sessions(slug)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/sessions/{slug}/{sid}")
def legacy_session_detail(slug: str, sid: str, showAll: bool = False):
    try:
        return _provider("claude").session_payload(slug, sid, fold_synthetic=not showAll)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


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
    try:
        return S.save_keybindings(mapping)
    except ValueError as e:
        raise HTTPException(400, str(e))


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
        "--allow-remote",
        action="store_true",
        help="required to bind a non-loopback address; the API is unauthenticated "
        "and can delete session files, so anyone who can reach it has full access",
    )
    args = parser.parse_args()

    if not is_loopback_host(args.host):
        if not args.allow_remote:
            parser.error(
                f"refusing to bind {args.host}: the editor is unauthenticated and can "
                "delete session files. Use --host 127.0.0.1, or pass --allow-remote "
                "if the network is trusted."
            )
        allow_any_host()
        print(
            f"WARNING: listening on {args.host} — anyone who can reach this port can "
            "read and delete your chat sessions."
        )

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    print(f"ChatJsonEditor running at {url}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
