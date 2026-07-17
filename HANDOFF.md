# ChatJsonEditor — Session Handoff

_Last updated: 2026-07-17_

## What this project is

A **local web app to delete and undo deletions of earlier turns** in multi-source
chat sessions (Claude Code, Grok, Grok Build, Gemini CLI, Antigravity). Long multi-turn
sessions accumulate stale early turns that bloat context on resume; this tool lets you
prune whole turns safely and reversibly, with a Claude-Desktop-like reading UI.

- **Stack:** Python + `uv`, FastAPI + uvicorn backend, single vanilla-JS HTML page (no build step).
- **Granularity:** whole-turn deletion (a human prompt + all assistant/tool activity it
  triggered), which keeps tool_use/tool_result pairs valid for resume.
- **Read-only transcript** (no live message composer / no API calls).
- **Offline:** rendering libs are vendored under `static/vendor/` — no CDN/internet at runtime.

## Sources

| Source id | Label | On-disk root | Format |
|-----------|-------|--------------|--------|
| `claude` | Claude Code | `~/.claude/projects/<slug>/<sid>.jsonl` | JSONL + uuid/parentUuid chain |
| `grok` | Grok | `~/.grok/sessions/<encoded-cwd>/<sid>/` | `updates.jsonl` + `chat_history.jsonl` + `summary.json` |
| `grok-build` | Grok Build | same tree as Grok | classified by `agent_name` / `model_id` (Build wins on overlap) |
| `gemini` | Gemini CLI | `~/.gemini/tmp/<project_hash>/chats/` | session JSON / JSONL |
| `antigravity` | Antigravity | `~/.gemini/antigravity/conversations/<uuid>.db` | SQLite steps + brain transcripts |

Env overrides (tests use scratch copies only):

```
CLAUDE_PROJECTS_DIR
GROK_SESSIONS_DIR / GROK_HOME
GEMINI_TMP_DIR
ANTIGRAVITY_ROOT
CHATJSONEDITOR_BACKUPS_DIR   # namespaced as {backups}/{source}/{slug}/{sid}/
CHATJSONEDITOR_CONFIG_DIR
```

## Global environment rules (from user's CLAUDE.md)

- Use `uv pip install` — **never** `pip install`.
- Use `uv run` to run things.
- Platform: Windows 11, PowerShell primary shell.
- **Never touch the user's live session files in tests.** Point roots at scratch dirs.

## How to run / test

```bash
uv run pytest -q                      # 19 passing
uv run chatjsoneditor                 # default port 8642
uv run chatjsoneditor --port 8646 --no-browser
```

## File map

```
pyproject.toml
chatjsoneditor/
  app.py                      # FastAPI: multi-source + legacy Claude routes
  sessions.py                 # Claude JSONL logic, History, path roots, keybindings
  providers/
    base.py                   # registry
    claude.py / grok.py / gemini.py / antigravity.py
    common.py                 # NormMsg / NormTurn helpers
  static/index.html           # multi-source explorer UI
  static/vendor/              # marked, DOMPurify, MathJax SVG
tests/
  test_sessions.py            # Claude
  test_providers.py           # Grok / Gemini / Antigravity
  fixtures/                   # sample.jsonl + grok/gemini/antigravity trees
HANDOFF.md
```

## API

```
GET  /api/sources
GET  /api/{source}/projects
GET  /api/{source}/projects/{slug}/sessions
GET  /api/{source}/sessions/{slug}/{sid}?showAll=0|1
POST /api/{source}/sessions/{slug}/{sid}/delete   {turnIds, hash, showAll}
POST /api/{source}/sessions/{slug}/{sid}/undo|redo {hash, showAll}
GET|POST /api/keybindings
```

Legacy Claude routes without `{source}` still work (`/api/projects`, `/api/sessions/...`).

## Safety invariants (do not regress)

- **Whole-turn delete** only.
- **SHA-256 hash guard** → 409 Conflict if disk changed.
- **Atomic / snapshot undo** per source (single file or multi-file bundle).
- **Grok:** delete keeps `updates.jsonl` and `chat_history.jsonl` aligned by user-prompt order; patches `summary.json` counts.
- **Antigravity:** delete step idx ranges in SQLite + matching transcript lines; WAL lock → 409.
- Tests never write live roots.

## UI

Explorer roots = sources → projects (path trie) → lazy sessions.
Reading view vs Show all, summary pane, keybindings unchanged in spirit.

## Grok Build classification

A session is Grok Build when `agent_name` is in `{grok-build-plan, grok-build, …}` or
`current_model_id` starts with `grok-build`. Otherwise it lists under Grok. No session appears in both.

## Known gotchas

- Resume the source app after editing (most do not hot-reload open sessions).
- Antigravity protobuf payloads are not fully decoded; display prefers `transcript.jsonl`.
- Gemini CLI sessions may be absent until the CLI has been used; empty source is OK.
- URL-encoded Grok project slugs need `encodeURIComponent` in API paths.
