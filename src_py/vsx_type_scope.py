# -*- coding: ascii -*-
"""VSX type out-of-scope token matching (PUBLICATION-PREP Part B).

Matching semantics (canonical; keep in sync with PROCESS/PARAMS/FLOW/registry help):

A target's VSX type string is tokenized on the separators ``|``, ``/``, ``+`` and
whitespace; each token is uppercased and a trailing ``:`` (VSX uncertain
classification mark) is stripped. Config entries are normalized the same way.
A target is OUT OF SCOPE when ANY of its tokens equals ANY configured entry.
Substring matching is NEVER used.
"""
from __future__ import annotations

import re
from typing import Iterable

_TOKEN_SPLIT = re.compile(r"[|/+]+|\s+")


def normalize_vsx_type_token(token: str) -> str:
    t = str(token or "").strip().upper()
    if t.endswith(":"):
        t = t[:-1].strip()
    return t


def tokenize_vsx_type(vsx_type: str) -> list[str]:
    raw = str(vsx_type or "").strip()
    if not raw:
        return []
    parts = _TOKEN_SPLIT.split(raw)
    out: list[str] = []
    for p in parts:
        n = normalize_vsx_type_token(p)
        if n:
            out.append(n)
    return out


def normalize_vsx_out_of_scope_types(entries: Iterable[str] | None) -> list[str]:
    if not entries:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for e in entries:
        n = normalize_vsx_type_token(str(e))
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def vsx_type_is_out_of_scope(vsx_type: str, configured: Iterable[str] | None) -> bool:
    """True when any token of ``vsx_type`` equals any normalized config entry."""
    cfg = normalize_vsx_out_of_scope_types(configured)
    if not cfg:
        return False
    tokens = tokenize_vsx_type(vsx_type)
    if not tokens:
        return False
    cfg_set = set(cfg)
    return any(t in cfg_set for t in tokens)


def is_vsx_auto_selected_target(row: dict | object) -> bool:
    """VSX auto-selected rows are filterable; manual / exoplanet / user are not.

    ``catalog == "VSX"`` (case-insensitive) => auto.
    Explicit non-VSX catalogs (MANUAL/USER/EXOPLANET/...) => never filtered.
    Legacy rows without ``catalog``: treat as VSX auto when ``vsx_name`` is set.
    """
    get = row.get if hasattr(row, "get") else lambda k, d=None: getattr(row, k, d)
    cat = str(get("catalog", "") or "").strip().upper()
    if cat == "VSX":
        return True
    if cat in {"MANUAL", "USER", "CUSTOM", "EXOPLANET", "HAND"}:
        return False
    if cat:
        return False
    return bool(str(get("vsx_name", "") or "").strip())
