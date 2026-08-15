"""Shared helpers for non-Claude providers."""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..sessions import MAX_TOOL_TEXT, History, _cap, _preview


@dataclass
class NormMsg:
    """Normalized message used to build UI turn payloads."""
    kind: str
    text: str = ""
    synthetic: bool = False
    id: str | None = None
    name: str | None = None
    tool_use_id: str | None = None
    is_error: bool = False
    input: str | None = None
    timestamp: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"kind": self.kind}
        if self.kind == "tool_use":
            d["id"] = self.id
            d["name"] = self.name or "?"
            d["input"] = self.input or ""
        elif self.kind == "tool_result":
            d["toolUseId"] = self.tool_use_id
            d["isError"] = self.is_error
            d["text"] = self.text
        else:
            d["text"] = self.text
            if self.synthetic and self.kind == "user":
                d["synthetic"] = True
        return d


@dataclass
class NormTurn:
    id: str
    deletable: bool
    messages: list[NormMsg] = field(default_factory=list)
    timestamp: str | None = None

    def to_summary(self, index: int) -> dict:
        real_prompt = next(
            (m.text for m in self.messages if m.kind == "user" and not m.synthetic),
            None,
        )
        syn_prompt = next(
            (m.text for m in self.messages if m.kind == "user" and m.synthetic),
            None,
        )
        prompt = real_prompt if real_prompt is not None else (syn_prompt or "")
        if not self.deletable:
            prompt = prompt or "(session header)"
        n_assistant = sum(1 for m in self.messages if m.kind == "assistant")
        n_tools = sum(1 for m in self.messages if m.kind == "tool_use")
        nbytes = 0
        for m in self.messages:
            blob = m.text or m.input or ""
            nbytes += len(blob.encode("utf-8", errors="replace")) + 1
        return {
            "id": self.id,
            "index": index,
            "deletable": self.deletable,
            "timestamp": self.timestamp,
            "prompt": _preview(prompt if self.deletable else "(session header)", 300),
            "assistantCount": n_assistant,
            "toolCount": n_tools,
            "lineCount": len(self.messages),
            "bytes": nbytes,
            "messages": [m.to_dict() for m in self.messages],
        }


def text_from_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                if b.get("type") == "text" or "text" in b:
                    parts.append(b.get("text") or "")
                elif b.get("type") == "tool_result":
                    parts.append(text_from_content(b.get("content")))
                elif "content" in b and isinstance(b["content"], str):
                    parts.append(b["content"])
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return text_from_content(content.get("text") or content.get("content") or "")
    return str(content)


def cap_tool(s: str) -> str:
    return _cap(s or "", MAX_TOOL_TEXT)


def session_payload(
    source: str, slug: str, sid: str, *, hash: str, nbytes: int, turns: list[dict]
) -> dict:
    """The `/api/{source}/sessions/{slug}/{sid}` response envelope."""
    return {
        "source": source,
        "slug": slug,
        "sid": sid,
        "hash": hash,
        "bytes": nbytes,
        "turns": turns,
        **History(slug, sid, source=source).status(),
    }


def turn_spans(starts: Sequence[int], total: int) -> list[tuple[bool, int, int]]:
    """(is_header, start, end) spans covering [0, total) given turn start indices.

    Anything before the first start is a non-deletable header span; each start
    owns everything up to the next one.
    """
    if not starts:
        return [(True, 0, total)] if total else []
    spans = []
    if starts[0] > 0:
        spans.append((True, 0, starts[0]))
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else total
        spans.append((False, start, end))
    return spans


def deleted_indices(
    ranges_by_id: dict[str, tuple[int, int]], turn_ids: list[str]
) -> set[int]:
    """Indices covered by the requested turns; raises on unknown ids."""
    unknown = [tid for tid in turn_ids if tid not in ranges_by_id]
    if unknown:
        raise ValueError(f"unknown or undeletable turn ids: {unknown}")
    out: set[int] = set()
    for tid in turn_ids:
        start, end = ranges_by_id[tid]
        out.update(range(start, end))
    return out


def group_projects(entries: Iterable[tuple[str, str]]) -> list[dict]:
    """Collapse (slug, label) pairs into project rows with session counts."""
    by_slug: dict[str, dict] = {}
    for slug, label in entries:
        row = by_slug.setdefault(
            slug, {"slug": slug, "label": label, "sessionCount": 0}
        )
        row["sessionCount"] += 1
    return sorted(by_slug.values(), key=lambda p: p["label"].lower())


def first_prompt_title(turns: Iterable[NormTurn], default: str) -> str:
    """First line of the first real user prompt, for session lists."""
    for t in turns:
        if not t.deletable:
            continue
        for m in t.messages:
            if m.kind == "user" and m.text.strip():
                return m.text.strip().splitlines()[0][:80]
        break
    return default


def newest_mtime(paths: Iterable[Path], fallback: float) -> float:
    return max([p.stat().st_mtime for p in paths if p.is_file()] + [fallback])


def total_bytes(paths: Iterable[Path]) -> int:
    return sum(p.stat().st_size for p in paths if p.is_file())


_USER_QUERY_RE = re.compile(r"<user_query>\s*([\s\S]*?)\s*</user_query>", re.I)
_SYNTHETIC_GROK_RE = re.compile(
    r"^\s*<(?:user_info|system-reminder|agent_info|environment_details)\b",
    re.I,
)


def extract_user_query(text: str) -> str | None:
    m = _USER_QUERY_RE.search(text or "")
    return m.group(1).strip() if m else None


def is_grok_synthetic_user(text: str) -> bool:
    if not text:
        return True
    if extract_user_query(text) is not None:
        return False
    if _SYNTHETIC_GROK_RE.match(text):
        return True
    if text.lstrip().startswith("<") and "user_query" not in text:
        return True
    return False
