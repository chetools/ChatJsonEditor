"""Shared helpers for non-Claude providers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..sessions import MAX_TOOL_TEXT, _cap, _preview


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
