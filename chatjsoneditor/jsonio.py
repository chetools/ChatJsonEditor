"""Tolerant JSON / JSONL reading and writing shared by every source adapter.

All sources store transcripts as JSONL (or JSON) files written by another
program, so parsing must never raise on a malformed line and rewriting must
preserve untouched lines byte-for-byte. Whole-file damage is different: a file
that is not even UTF-8 cannot be edited without destroying content, so reading
it raises `CorruptDataError`.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

log = logging.getLogger(__name__)


class CorruptDataError(Exception):
    """On-disk data could not be parsed, so editing it would destroy content."""


def split_lines(text: str) -> tuple[list[str], bool]:
    """Split a JSONL document into lines plus whether it ended with a newline."""
    trailing = text.endswith("\n")
    body = text[:-1] if trailing else text
    return (body.split("\n") if body else []), trailing


def join_lines(lines: Iterable[str], trailing: bool = True) -> bytes:
    text = "\n".join(lines)
    if trailing and text:
        text += "\n"
    return text.encode("utf-8")


def parse_object(line: str) -> dict | None:
    """Parse one JSONL line, returning None unless it is a JSON object."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def read_text(path: Path) -> str:
    """Read a file as UTF-8 without newline translation (exact round-trip)."""
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as e:
        raise CorruptDataError(f"{path.name} is not valid UTF-8: {e}") from e


def load_rows(path: Path) -> list[tuple[str, dict | None]]:
    """(raw_line, parsed_object_or_None) for each line; [] if path is missing."""
    if not path.is_file():
        return []
    lines, _ = split_lines(read_text(path))
    return [(line, parse_object(line)) for line in lines]


def load_objects(path: Path) -> list[dict]:
    """Parsed objects only; blank lines are ignored and damaged lines logged."""
    if not path.is_file():
        return []
    out = []
    skipped = 0
    for line in read_text(path).splitlines():
        if not line.strip():
            continue
        obj = parse_object(line)
        if obj is None:
            skipped += 1
            continue
        out.append(obj)
    if skipped:
        log.warning("%s: %d unreadable line(s) not rendered", path, skipped)
    return out


def read_json_or_default(path: Path, default: dict | None = None) -> dict:
    """A JSON object read for display only: damage is logged, never raised."""
    fallback = {} if default is None else default
    if not path.is_file():
        return fallback
    try:
        data = json.loads(read_text(path))
    except (json.JSONDecodeError, OSError, CorruptDataError):
        log.warning("ignoring unreadable %s", path, exc_info=True)
        return fallback
    if not isinstance(data, dict):
        log.warning("ignoring %s: expected an object, got %s", path, type(data).__name__)
        return fallback
    return data


def dump_json(data, indent: int = 2) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=indent) + "\n").encode("utf-8")


def pretty(data) -> str:
    """Indented JSON text used for tool inputs shown in the UI."""
    return json.dumps(data, ensure_ascii=False, indent=2)
