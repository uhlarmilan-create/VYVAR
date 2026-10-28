"""JSON with ``//`` line comments (config.json is written with one help comment per key). Leaf module."""

from __future__ import annotations

import json
from typing import Any


# --- moved from config (LAYER-2026-10) ---


def strip_jsonc_comments(text: str) -> str:
    """Remove ``//`` line comments that sit OUTSIDE string literals (JSONC-lite).

    A small character state machine so ``//`` inside a string value (e.g. a URL like
    ``"http://x"``) is preserved. Only ``//`` to end-of-line is stripped; block comments
    ``/* */`` and trailing commas are intentionally NOT supported so the format stays
    strict elsewhere and diffs/tools stay sane. Newlines are preserved so line numbers in
    validator error messages stay accurate.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_str = False
    escape = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] not in "\r\n":
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def parse_config_text(text: str) -> dict[str, Any]:
    """Parse config.json text tolerating ``//`` line comments; raises json.JSONDecodeError."""
    parsed = json.loads(strip_jsonc_comments(text))
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("config.json top level must be a JSON object", text, 0)
    return parsed
