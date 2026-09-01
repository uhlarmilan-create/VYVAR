#!/usr/bin/env python3
"""CONSOLIDATE-01D D1: inventory all config_runtime keys + grep consumers.

Read-only. Writes config_prerez_table.json (skeleton; class/proposal filled later).
"""
from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SRC = REPO / "src_py"
sys.path.insert(0, str(SRC))

from config import AppConfig  # noqa: E402
from params_registry import load_registry  # noqa: E402

OUT = Path(__file__).resolve().parent / "config_prerez_table.json"
SRC_FILES = list((REPO / "src_py").rglob("*.py"))


def _default_of(f: dataclasses.Field) -> str:
    v = f.default
    if v is dataclasses.MISSING:
        if f.default_factory is not dataclasses.MISSING:
            try:
                return repr(f.default_factory())
            except Exception:
                return "<factory>"
        return "<missing>"
    return repr(v)


def _type_of(f: dataclasses.Field) -> str:
    t = f.type
    return getattr(t, "__name__", str(t).replace("typing.", ""))


def _consumers(key: str) -> list[str]:
    pat = re.compile(rf"\b{re.escape(key)}\b")
    hits: list[str] = []
    for p in SRC_FILES:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if pat.search(line):
                rel = p.relative_to(REPO).as_posix()
                hits.append(f"{rel}:{i}")
                if len(hits) >= 40:
                    return hits
    return hits


def main() -> None:
    reg = load_registry()
    fields = {f.name: f for f in dataclasses.fields(AppConfig)}
    keys = sorted(k for k, e in reg.items() if e.get("owner") == "config_runtime")
    rows = []
    for k in keys:
        f = fields.get(k)
        rows.append(
            {
                "key": k,
                "type_default": (
                    f"{_type_of(f)} = {_default_of(f)}" if f is not None else "NOT_ON_APPCONFIG"
                ),
                "consumers": _consumers(k),
                "class": "",
                "proposal": "",
                "risk": "",
                "d2_evidence": "",
            }
        )
    payload = {
        "n": len(rows),
        "note": "class/proposal/risk filled by CONSOLIDATE-01D after consumer review",
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} n={len(rows)}")


if __name__ == "__main__":
    main()
