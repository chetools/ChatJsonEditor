# ChatJsonEditor — Session Handoff

_Last updated: 2026-07-17_

## What this project is

A **local web app to delete and undo deletions of earlier turns** in Claude Code
session `.jsonl` files. Long multi-turn sessions accumulate stale early turns that bloat
context on `--resume`; this tool lets you prune whole turns safely and reversibly, with a
Claude-Desktop-like reading UI.

- **Stack:** Python + `uv`, FastAPI + uvicorn backend, single vanilla-JS HTML page (no build step).
- **Granularity:** whole-turn deletion (a human prompt + all assistant/tool activity it
  triggered), which guarantees tool_use/tool_result pairs stay valid for resume.
- **Read-only transcript** (no live message composer / no Claude API calls).
- **Offline:** rendering libs are vendored under `static/vendor/` — no CDN/internet at runtime.

## Global environment rules (from user's CLAUDE.md)

- Use `uv pip install` — **never** `pip install`.
- Use `uv run` to run things.
- Platform: Windows 11, PowerShell primary shell (Bash tool also available).
- **Never touch the user's live session files.** All testing uses scratch copies under the
  scratchpad dir with `CLAUDE_PROJECTS_DIR` / `CHATJSONEDITOR_BACKUPS_DIR` /
  `CHATJSONEDITOR_CONFIG_DIR` pointed there.

## How to run / test

```bash
# tests
uv run pytest -q                      # 12 passing

# run the app (opens browser)
uv run chatjsoneditor                 # default port 8642
uv run chatjsoneditor --port 8646 --no-browser
```

Scratch-copy launch pattern (used for all UI verification):
```bash
export CLAUDE_PROJECTS_DIR="<scratch>/projects"
export CHATJSONEDITOR_BACKUPS_DIR="<scratch>/backups"
export CHATJSONEDITOR_CONFIG_DIR="<scratch>/config"
uv run chatjsoneditor --port 8646 --no-browser
```

## File map

```
pyproject.toml              # fastapi, uvicorn; dev: pytest, httpx; script: chatjsoneditor = app:main
chatjsoneditor/
  app.py                    # FastAPI: endpoints + main() (argparse --host/--port/--no-browser)
  sessions.py               # ALL core logic (parsing, turns, delete, chain repair, undo/redo, keybindings)
  static/index.html         # ENTIRE UI (CSS + vanilla JS, no build)
  static/vendor/            # marked.min.js, purify.min.js, tex-svg.js (MathJax SVG), README.md (provenance)
tests/
  test_sessions.py          # 12 tests
  fixtures/sample.jsonl      # synthetic 3-human-turn session covering all entry types
HANDOFF.md                  # this file
```

Plan file (plan-mode artifact, base + 3 appended sections): `~/.claude/plans/i-want-to-be-transient-shannon.md`

## Session file format (verified against real files)

- One JSON object per line. Chain entries (`user`,`assistant`,`attachment`,`system`) have
  `uuid`+`parentUuid` forming a linked list in file order.
- Metadata lines (`queue-operation`,`last-prompt`,`ai-title`,`mode`,`summary`) have **no** uuid.
- A **turn** starts at a human `user` prompt (`origin.kind=="human"` or string `message.content`,
  not a `tool_result` list) and runs to the next such prompt.
- **Delete = remove the turn's lines + repair the chain:** first surviving chain entry after the
  cut gets its `parentUuid` rewritten to the last surviving `uuid` before the cut (null if the
  first turn was deleted). `last-prompt.leafUuid` pointing into deleted entries is remapped too.

## Safety invariants (do not regress)

- **Byte-exact round-trip:** untouched lines are written back byte-identical; only repaired
  entries are re-serialized.
- **Atomic writes** (temp file + `os.replace`).
- **Whole-file snapshot undo/redo** to `CHATJSONEDITOR_BACKUPS_DIR`; undo restores byte-identical
  original, redo restores the deleted state.
- **SHA-256 hash guard:** delete/undo/redo take the client's file hash; 409 `ConflictError` if the
  file changed on disk (e.g. session live in Claude). Frontend shows a reload banner.
- **No orphaned tool_results.** `tool_pairing_ok` asserts `results <= uses` (a trailing orphan
  tool_use is tolerated — happens when a human interrupts a pending tool call). `no_new_orphans`
  asserts deletion introduces none.

## API (app.py)

- `GET /api/projects` → `[{slug, label (decoded path), sessionCount}]`
- `GET /api/projects/{slug}/sessions` → `[{sid, title, timestamp, bytes, turnCount}]`
- `GET /api/sessions/{slug}/{sid}` → `{slug, sid, hash, bytes, turns[], canUndo, canRedo}`
- `POST /api/sessions/{slug}/{sid}/delete` (body `{turnIds, hash}`)
- `POST .../undo`, `POST .../redo` (body `{hash}`)
- `GET /api/keybindings`, `POST /api/keybindings` (body = action→combo map; merged + validated)

`turn_messages` emits per-message dicts: `user{kind,text}`, `assistant{kind,text}`,
`thinking{kind,text}` (skipped if empty), `tool_use{kind,id,name,input}`,
`tool_result{kind,toolUseId,isError,text}`. `MAX_TOOL_TEXT = 64KB` cap on tool input/result.

## UI state (index.html)

Three panes: `#explorerPane` (hierarchical tree) │ splitter │ `#conversationPane` (flex) │
right-splitter │ `#summaryPane`. Pane widths persist in `localStorage['cje.paneWidths']`.

- **Explorer:** compacted directory trie (VS Code "compact folders") built client-side from
  project `label`s; sessions lazy-load on project expand; expanded state in `cje.expanded`.
- **Reading view ⟷ Show all** (toolbar `#showAllBtn`, `state.showAll`,
  `localStorage['cje.showAll']`, default = reading view). Toggling **refetches** the session
  because turn grouping itself depends on the mode (see below).
  - **Reading view (default):** only actual conversation is shown — human prompts + assistant
    replies. Thinking, tool_use/tool_result, `system`, and **synthetic** user entries are hidden
    (`isReadingMsg` in index.html). Synthetic entries also **fold** into the preceding turn
    (`fold_synthetic=True`), so they don't create their own turns.
  - **Show all:** everything renders (thinking/tools/system/synthetic), and synthetic entries are
    restored as their **own turns** — the original pre-filter grouping (`fold_synthetic=False`).
  - **Synthetic detection:** `is_synthetic_user_text` (sessions.py) matches user string content
    that is machinery the human never typed — `<local-command-*>`, `<task-notification>`,
    slash-command wrappers (`<command-name|message|args|stdout|contents>`), start-anchored,
    case-insensitive. Emitted per-message as `synthetic:true`.
  - **Mode threads through the backend:** `group_turns/summarize_session/delete_turns/
    perform_delete` take `fold_synthetic`; API carries it as `?showAll=1` (GET) and
    `showAll` in delete/undo/redo bodies, so returned payloads + delete's turn-id resolution
    match whichever view the client is in. In show-all mode synthetic turns are individually
    deletable; in reading mode deleting a turn also removes its folded machinery.
- **Conversation:** turn groups; tool_use merged with its tool_result by id into one `<details>`;
  thinking/tool blocks collapsed by default; math-safe render pipeline (extract math → marked →
  restore → DOMPurify → MathJax typeset). Click anywhere on a turn selects it (excludes clicks on
  `summary, a, pre, .io, details` and active text selection). Shift-click extends range.
- **Summary pane (right):** per-turn outline — user text + reply text (assistant text only, no
  tools/thinking), each clamped to *k* lines via CSS `-webkit-line-clamp: var(--sumLines)`.
  `k` adjustable via header number input (1–12, default 2), persisted `cje.sumLines`. Clicking an
  entry navigates + marks current; current turn highlighted and scrolled into view; selection
  syncs a red marker.
- **Current-turn cursor** `state.current`: `setCurrent(i)` applies `.current` ring and
  `scrollTurnToTop(el)` — anchors the destination turn's **top ~44px below the viewport top**
  (leaves a sliver of the previous turn; a turn taller than the viewport still lands near the top,
  not centered). This replaced the old `scrollIntoView({block:'center'})`.

### Default keybindings (all rebindable via `⌨` panel → on-disk `keybindings.json`)

```
prevTurn ArrowUp   nextTurn ArrowDown          # move cursor across turns
scrollUp Shift+ArrowUp  scrollDown Shift+ArrowDown  # line-scroll transcript ±120px (browser-style)
toggleSelect Space   deleteSelected d
undo Ctrl+z   redo Ctrl+y
firstTurn Home   lastTurn End
selectAll a   clearSelection Escape   help ?
```

## Current status

**All requested work is complete and verified.** Feature history:
1. Base editor (delete/undo/redo, chain repair, conflict guard). ✅
2. Claude-Desktop-style redesign (3-pane, Markdown + MathJax-SVG, collapsible tool/thinking). ✅
3. Unified hierarchical explorer tree, click-anywhere-to-select, configurable shortcuts. ✅
4. **(latest)** Top-anchored turn scrolling + adjustable-width Summary/outline pane. ✅

Last verification (13-turn BodyTracker session, scratch copy): 12/12 tests pass; three panes
render; summary shows correct user+reply text with 2/2 line-clamp; `k`=5 re-clamps and persists;
right-splitter drag grows/shrinks + persists; summary-click navigates; ↑/↓ move cursor;
**Shift+↓ = exactly +120px without moving cursor**; scroll anchors normal turns at 44px (40px prev
sliver) and tall turns near the top; selection syncs to summary marker.

## Known gotchas

- **Browser-pane verification:** the screenshot tool sometimes times out or opens a tiny viewport
  (e.g. 464×149) where 340+300px of side panes collapse the conversation column to 0 width and turns
  render absurdly tall — **resize to a real size** (`resize_window width 1400 height 900`) before
  measuring. Prefer `javascript_tool`/DOM inspection over screenshots.
- **Measuring scroll:** `#transcript` has `scroll-behavior: smooth`; reading `scrollTop`/rects
  mid-animation gives garbage. Temporarily set `scrollBehavior='auto'` and wait 2 rAFs before
  asserting final positions.
- PowerShell process-kill of the background server sometimes returns exit 255 but still stops it;
  confirm the port is free rather than trusting the exit code.

## Suggested next steps (none pending; only if user asks)

- Visual polish pass once a reliable screenshot path exists.
- Optional: reflect selected turns more richly in the summary pane, or add multi-select from it.
- Optional: keyboard shortcut to focus/jump the summary pane.
