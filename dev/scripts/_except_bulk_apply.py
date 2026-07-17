#!/usr/bin/env python3
"""Apply EXCEPT-BULK dispositions across the VYVAR census (conservative policy, 2026-07-08).

Reads docs/VYVAR_EXCEPT_CENSUS.md, maps each EXC site to delete-dead / log / comment /
approved narrow actions, edits production modules, and marks census rows disposition-DONE.

Does NOT run automatically; invoke explicitly with --phase and optional --dry-run.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
REPO_ROOT = _bootstrap.REPO_ROOT
CENSUS_PATH = REPO_ROOT / "docs" / "VYVAR_EXCEPT_CENSUS.md"
BULK_STAMP = "EXCEPT-BULK-2 2026-07-08"
COMMENT_PREFIX = "# EXC-"

Phase = Literal["delete-dead", "log", "comment", "narrow", "all"]

# ---------------------------------------------------------------------------
# FIX batches - already landed; bulk pass must not touch these sites.
# ---------------------------------------------------------------------------
FIXED_MARKERS = ("FIXED",)

FIX1_IDS = frozenset(
    {
        "EXC-0043",
        "EXC-0044",
        "EXC-0045",
        "EXC-0132",
        "EXC-0136",
        "EXC-0166",
        "EXC-0198",
        "EXC-0449",
        "EXC-0452",
        "EXC-0455",
        "EXC-0626",
    }
)
FIX2_IDS = frozenset(
    {
        "EXC-0275",
        "EXC-0312",
        "EXC-0317",
        "EXC-0331",
        "EXC-0339",
        "EXC-0342",
        "EXC-0350",
        "EXC-0389",
        "EXC-0415",
        "EXC-0433",
    }
)
FIX3_IDS = frozenset(
    {
        "EXC-0010",
        "EXC-0089",
        "EXC-0090",
        "EXC-0092",
        "EXC-0094",
        "EXC-0095",
        "EXC-0100",
        "EXC-0102",
        "EXC-0586",
        "EXC-0605",
        "EXC-0625",
    }
)
FIX4_IDS = frozenset(
    {
        "EXC-0022",
        "EXC-0079",
        "EXC-0111",
        "EXC-0115",
        "EXC-0486",
        "EXC-0487",
        "EXC-0491",
        "EXC-0561",
    }
)
FIX_BATCH_IDS = FIX1_IDS | FIX2_IDS | FIX3_IDS | FIX4_IDS

# ---------------------------------------------------------------------------
# Tranche 4 group defaults (mechanical census rows overridden by these ranges).
# Individual overrides below take precedence over group rows.
# ---------------------------------------------------------------------------
TRANCHE4_GROUPS: list[tuple[int, int, str, str]] = [
    (225, 274, "T3", "comment"),
    (492, 562, "T3", "comment"),  # UI block; individual overrides except FIX-4 etc.
    (105, 110, "T3", "comment"),
    (1, 5, "T3", "comment"),
    (563, 568, "T4", "comment"),
    (478, 485, "T4", "comment"),
]

# Individual tranche 4 / 4b overrides from CURSOR TASK (highest priority for listed IDs).
INDIVIDUAL_OVERRIDES: dict[str, str] = {
    "EXC-0486": "FIX-4 #1",
    "EXC-0487": "FIX-4 #2",
    "EXC-0488": "keep + comment",
    "EXC-0489": "narrow to (TypeError, ValueError) + comment",
    "EXC-0490": "narrow to sqlite3.Error + comment",
    "EXC-0569": "comment",
    "EXC-0570": "keep + logging.warning",
    "EXC-0571": "narrow to (TypeError, ValueError) + comment",
    "EXC-0572": "keep + comment",
    "EXC-0573": "keep + comment",
    "EXC-0574": "narrow to (TypeError, ValueError) + comment",
    "EXC-0575": "keep + logging.warning",
    "EXC-0576": "keep + logging.debug",
    "EXC-0577": "comment",
    "EXC-0578": "narrow to (TypeError, ValueError) + comment",
    "EXC-0049": "keep + logging.warning",
    "EXC-0050": "comment",
    "EXC-0051": "keep + logging.debug",
    "EXC-0052": "narrow to (TypeError, ValueError) + comment",
    "EXC-0053": "narrow to (TypeError, ValueError) + comment",
    "EXC-0054": "keep + logging.warning",
    "EXC-0116": "keep + logging.warning",
    "EXC-0117": "keep + logging.debug",
    "EXC-0118": "keep + logging.warning",
    "EXC-0119": "narrow to sqlite3.Error + comment",
    "EXC-0469": "comment",
    "EXC-0470": "comment",
    "EXC-0471": "comment",
    "EXC-0472": "keep + logging.warning",
    "EXC-0473": "narrow to (TypeError, ValueError) + comment",
    "EXC-0474": "narrow to (TypeError, ValueError) + comment",
    "EXC-0475": "comment",
    "EXC-0476": "comment",
    "EXC-0477": "comment",
    "EXC-0020": "narrow to sqlite3.Error + comment",
    "EXC-0021": "keep + comment",
    "EXC-0022": "FIX-4 #8",
    "EXC-0023": "keep + logging.debug",
    "EXC-0024": "narrow to IndexError + comment",
    "EXC-0025": "keep + logging.debug",
    "EXC-0026": "keep + logging.warning",
    "EXC-0027": "comment",
    "EXC-0028": "keep + logging.debug",
    "EXC-0029": "narrow to (TypeError, ValueError) + comment",
    "EXC-0048": "comment",
    "EXC-0055": "keep + logging.debug",
    "EXC-0076": "keep + logging.warning",
    "EXC-0077": "comment",
    "EXC-0111": "FIX-4 #5",
    "EXC-0112": "comment",
    "EXC-0113": "keep + logging.warning",
    "EXC-0114": "comment",
    "EXC-0115": "FIX-4 #7",
    "EXC-0445": "keep + comment",
    "EXC-0467": "comment",
    "EXC-0468": "keep + comment",
    "EXC-0491": "FIX-4 #3",
    "EXC-0078": "comment",
    "EXC-0079": "FIX-4 #6",
    "EXC-0080": "comment",
    "EXC-0081": "narrow to (TypeError, ValueError) + comment",
    "EXC-0082": "narrow to (TypeError, ValueError) + comment",
    "EXC-0083": "narrow to (TypeError, ValueError) + comment",
    "EXC-0084": "keep + comment",
    "EXC-0085": "comment",
    "EXC-0086": "comment",
    "EXC-0087": "comment",
    "EXC-0538": "keep + log_event warning",
    "EXC-0539": "keep + log_event warning",
    "EXC-0561": "FIX-4 #4",
}

# Explicit narrow tuples from evidence tables (conservative allow-list).
EVIDENCE_NARROW: dict[str, tuple[str, ...]] = {
    "EXC-0061": ("sqlite3.Error",),
    "EXC-0062": ("sqlite3.Error",),
    "EXC-0065": ("sqlite3.Error",),
    "EXC-0067": ("sqlite3.Error",),
    "EXC-0073": ("TypeError", "ValueError"),
    "EXC-0074": ("sqlite3.Error",),
    "EXC-0075": ("sqlite3.Error",),
    "EXC-0588": ("OSError", "pickle.UnpicklingError", "KeyError", "TypeError", "ValueError"),
}

EVIDENCE_ROW = re.compile(
    r"^\|\s*(EXC-\d{4})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$"
)
MECHANICAL_ROW = re.compile(
    r"^\|\s*(EXC-\d{4})\s*\|\s*`([^`]+)`\s*\|"
)


@dataclass
class CensusSite:
    exc_id: str
    rel_path: str
    line: int
    tier: str
    effect: str
    disposition_raw: str
    source: str  # evidence | mechanical
    narrow_types: tuple[str, ...] | None = None
    log_level: str | None = None
    actions: set[str] = field(default_factory=set)
    skip_reason: str | None = None

    @property
    def abs_path(self) -> Path:
        return REPO_ROOT / self.rel_path


def _exc_num(exc_id: str) -> int:
    return int(exc_id.split("-")[1])


def _in_tranche4_group(exc_id: str) -> tuple[str, str] | None:
    n = _exc_num(exc_id)
    for lo, hi, tier, disp in TRANCHE4_GROUPS:
        if lo <= n <= hi:
            return tier, disp
    return None


def _parse_loc(loc: str, *, current_file: str | None = None) -> tuple[str, int]:
    loc = loc.strip().strip("`")
    # strip approximate marker: (~128) or ~128
    loc = re.sub(r"\(~\d+\)", "", loc).strip()
    loc = loc.replace("~", "").strip()
    if ":" in loc:
        path_part, line_part = loc.rsplit(":", 1)
        line_part = re.sub(r"[^\d]", "", line_part.split()[0])
        return path_part.strip(), int(line_part)
    # platesolver style: bs:117
    if re.match(r"^[a-z]+:\d+", loc):
        alias, line_s = loc.split(":", 1)
        line_s = re.sub(r"[^\d]", "", line_s.split()[0])
        alias_map = {"bs": "vyvar_blind_series.py", "bsol": "vyvar_blind_solver.py"}
        return alias_map.get(alias, f"{alias}.py"), int(line_s)
    # evidence table line-only: "61" or "131 (~128)"
    if re.match(r"^\d+", loc):
        if not current_file:
            raise ValueError(f"line-only loc {loc!r} without file context")
        line_no = int(re.match(r"(\d+)", loc).group(1))
        return current_file, line_no
    raise ValueError(f"unparseable loc: {loc!r}")


def _parse_mechanical_site(site_field: str) -> tuple[str, int]:
    return _parse_loc(site_field.strip())


def _normalize_disposition(raw: str) -> str:
    return " ".join(raw.strip().split())


def _extract_narrow_types(disp: str) -> tuple[str, ...] | None:
    m = re.search(r"\(([^)]+)\)", disp)
    if m and ("TypeError" in m.group(1) or "ValueError" in m.group(1)):
        parts = [p.strip() for p in m.group(1).split(",")]
        return tuple(parts)
    if "sqlite3.Error" in disp:
        return ("sqlite3.Error",)
    if "IndexError" in disp and "narrow" in disp.lower():
        return ("IndexError",)
    if "pickle.UnpicklingError" in disp or "OSError" in disp:
        parts = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", disp)
        allowed = (
            "OSError",
            "pickle.UnpicklingError",
            "KeyError",
            "TypeError",
            "ValueError",
        )
        picked = tuple(p for p in allowed if p in parts)
        return picked or None
    return None


def _resolve_actions(site: CensusSite) -> None:
    disp = _normalize_disposition(site.disposition_raw)
    low = disp.lower()

    if any(m in disp for m in FIXED_MARKERS) or site.exc_id in FIX_BATCH_IDS:
        site.skip_reason = "FIXED or FIX batch"
        return
    if "fix-4" in low or "fix-3" in low or "fix-2" in low or "fix-1" in low:
        site.skip_reason = "FIX batch marker in disposition"
        return
    if disp.endswith("-DONE") or "disposition-done" in low:
        site.skip_reason = "already disposition-DONE"
        return

    if site.narrow_types is None:
        site.narrow_types = EVIDENCE_NARROW.get(site.exc_id)
    if site.narrow_types is None:
        site.narrow_types = _extract_narrow_types(disp)

    if "delete-dead" in low:
        site.actions.add("delete-dead")
        return

    if re.search(r"keep\s*\+\s*logging\.(error|warning|debug)", low):
        site.log_level = re.search(r"logging\.(error|warning|debug)", low).group(1)  # type: ignore[union-attr]
        site.actions.add("log")
    elif "narrow+log-error" in low.replace(" ", ""):
        site.log_level = "error"
        site.actions.add("log")
    elif re.search(r"narrow\s*\+\s*logging\.(error|warning|debug)", low):
        site.log_level = re.search(r"logging\.(error|warning|debug)", low).group(1)  # type: ignore[union-attr]
        site.actions.add("log")
    elif re.search(r"narrow\s*\+\s*log\b", low) and "comment" not in low:
        site.log_level = "warning"
        site.actions.add("log")
    elif "keep + logging" in low:
        if "warning" in low:
            site.log_level = "warning"
        elif "debug" in low:
            site.log_level = "debug"
        else:
            site.log_level = "error"
        site.actions.add("log")

    # Conservative: only narrow when an approved tuple is explicit.
    if site.narrow_types:
        site.actions.add("narrow")

    # comment for leave+comment, narrow+comment, keep+comment, plain comment, log_event keep rows
    if (
        "comment" in low
        or "leave+comment" in low.replace(" ", "")
        or "keep + comment" in low
        or "keep+comment" in low.replace(" ", "")
        or disp == "comment"
        or "keep + log_event" in low
    ):
        site.actions.add("comment")

    if not site.actions:
        # fix-now rows outside FIX batches - comment-only fallback (should not happen)
        if "fix-now" in low or "fix now" in low:
            site.actions.add("comment")
        else:
            site.actions.add("comment")


FILE_SECTION = re.compile(r"^###\s+([a-zA-Z0-9_./ -]+\.py)")


def parse_census(text: str) -> dict[str, CensusSite]:
    sites: dict[str, CensusSite] = {}
    in_census = False
    current_file: str | None = None

    for line in text.splitlines():
        if line.startswith("## Census"):
            in_census = True
            current_file = None

        m_file = FILE_SECTION.match(line)
        if m_file:
            raw = m_file.group(1).strip()
            if "/" in raw:
                current_file = None  # per-row bs:/bsol: loc aliases
            else:
                current_file = raw

        m_ev = EVIDENCE_ROW.match(line)
        if m_ev:
            exc_id, loc, tier, effect, disp = m_ev.groups()
            rel, line_no = _parse_loc(loc, current_file=current_file)
            sites[exc_id] = CensusSite(
                exc_id=exc_id,
                rel_path=rel,
                line=line_no,
                tier=tier.strip(),
                effect=effect.strip(),
                disposition_raw=disp.strip(),
                source="evidence",
            )
            continue

        if in_census:
            m_m = MECHANICAL_ROW.match(line)
            if not m_m:
                continue
            exc_id, site_field = m_m.groups()
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 8:
                continue
            tier = parts[5]
            effect = parts[6]
            disp = parts[7]
            rel, line_no = _parse_mechanical_site(site_field)
            if exc_id not in sites:
                sites[exc_id] = CensusSite(
                    exc_id=exc_id,
                    rel_path=rel,
                    line=line_no,
                    tier=tier,
                    effect=effect,
                    disposition_raw=disp,
                    source="mechanical",
                )
            else:
                # Refresh line/path from mechanical scan; keep evidence disposition if present.
                existing = sites[exc_id]
                existing.rel_path = rel
                existing.line = line_no
                if existing.source == "mechanical":
                    existing.tier = tier
                    existing.effect = effect
                    existing.disposition_raw = disp

    # Tranche 4 group + individual overrides (only when mechanical / unset evidence).
    for exc_id, site in list(sites.items()):
        if exc_id in INDIVIDUAL_OVERRIDES:
            site.disposition_raw = INDIVIDUAL_OVERRIDES[exc_id]
            site.source = "override"
        else:
            grp = _in_tranche4_group(exc_id)
            if grp and site.source == "mechanical":
                tier, disp = grp
                if site.tier.startswith("?") or site.tier == "?":
                    site.tier = tier
                site.disposition_raw = disp

    for site in sites.values():
        _resolve_actions(site)

    return sites


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


@dataclass
class HandlerMatch:
    try_node: ast.Try
    handler: ast.ExceptHandler
    try_start: int
    try_end: int
    handler_start: int
    handler_end: int


def _is_broad_handler(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    if isinstance(handler.type, ast.Name):
        return handler.type.id in ("Exception", "BaseException")
    if isinstance(handler.type, ast.Tuple):
        names = [elt.id for elt in handler.type.elts if isinstance(elt, ast.Name)]
        return "Exception" in names or "BaseException" in names
    return False


def _handler_match(node: ast.Try, handler: ast.ExceptHandler) -> HandlerMatch:
    return HandlerMatch(
        try_node=node,
        handler=handler,
        try_start=node.lineno,
        try_end=node.end_lineno or node.lineno,
        handler_start=handler.lineno,
        handler_end=handler.end_lineno or handler.lineno,
    )


def _all_broad_handler_lines(source: str) -> list[int]:
    tree = ast.parse(source)
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            if _is_broad_handler(handler):
                lines.append(handler.lineno)
    return sorted(set(lines))


def _resolve_drift_lines(source: str, file_sites: list[CensusSite], *, max_delta: int = 200) -> None:
    """Map drifted census lines to unique nearest broad-except handlers (BULK-2)."""
    broad = _all_broad_handler_lines(source)
    if not broad:
        return
    claimed: set[int] = set()
    for site in sorted(file_sites, key=lambda s: s.line):
        candidates = [ln for ln in broad if ln not in claimed and abs(ln - site.line) <= max_delta]
        if not candidates:
            continue
        best = min(candidates, key=lambda ln: (abs(ln - site.line), ln))
        site.line = best
        claimed.add(best)


def _find_handler(source: str, target_line: int, *, max_line_delta: int = 0) -> tuple[HandlerMatch | None, int | None]:
    """Locate except handler at target_line, or nearest broad handler within max_line_delta."""
    tree = ast.parse(source)
    exact: HandlerMatch | None = None
    broad: list[tuple[int, HandlerMatch]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            if not _is_broad_handler(handler):
                continue
            hm = _handler_match(node, handler)
            broad.append((handler.lineno, hm))
            if handler.lineno <= target_line <= getattr(handler, "end_lineno", handler.lineno):
                exact = hm
            elif handler.lineno == target_line:
                exact = hm

    if exact is not None:
        return exact, None
    if not broad:
        return None, None
    nearest_line, nearest = min(broad, key=lambda t: abs(t[0] - target_line))
    if abs(nearest_line - target_line) <= max_line_delta:
        return nearest, nearest_line
    return None, None


def _handler_body_lines(source: str, match: HandlerMatch) -> list[str]:
    lines = source.splitlines()
    body = match.handler.body
    if not body:
        return []
    start = body[0].lineno - 1
    end = (body[-1].end_lineno or body[-1].lineno) - 1
    return lines[start : end + 1]


def _try_body_lines(source: str, match: HandlerMatch) -> list[str]:
    lines = source.splitlines()
    body = match.try_node.body
    if not body:
        return []
    start = body[0].lineno - 1
    end = (body[-1].end_lineno or body[-1].lineno) - 1
    return lines[start : end + 1]


def _is_pass_only_handler(source: str, match: HandlerMatch) -> bool:
    body = match.handler.body
    if len(body) == 1 and isinstance(body[0], ast.Pass):
        return True
    if len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        return body[0].value.value is Ellipsis
    return False


def _is_log_or_print_call(node: ast.AST) -> bool:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id in {"print", "log_event"}:
            return True
        if isinstance(func, ast.Attribute) and func.attr in {"print", "debug", "info", "warning"}:
            return True
    return False


def _stmt_log_or_print_only(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Expr):
        return _is_log_or_print_call(stmt.value)
    if isinstance(stmt, ast.If):
        return all(_stmt_log_or_print_only(s) for s in stmt.body + stmt.orelse)
    return False


def _try_body_log_or_print_only(source: str, match: HandlerMatch) -> bool:
    body = match.try_node.body
    if not body:
        return False
    return all(_stmt_log_or_print_only(stmt) for stmt in body)


def _already_bulk(source: str, match: HandlerMatch) -> bool:
    region = "\n".join(_handler_body_lines(source, match) + _try_body_lines(source, match))
    return BULK_STAMP in region or COMMENT_PREFIX + "EXC-" in region and BULK_STAMP.split()[0] in region


def _short_reason(site: CensusSite) -> str:
    text = site.effect.replace("|", "/").strip()
    text = text.replace('"', "'").replace("\n", " ").replace("\\", "/")
    if len(text) > 90:
        text = text[:87] + "..."
    return text


def _log_line(site: CensusSite, indent: str) -> str:
    level = site.log_level or "warning"
    msg = f"[{site.exc_id}] {_short_reason(site)}: %s"
    return f'{indent}logging.{level}({msg!r}, exc)'


def _comment_line(site: CensusSite, indent: str) -> str:
    tier = site.tier.split("-")[0] if site.tier else "?"
    return f'{indent}{COMMENT_PREFIX}{site.exc_id[4:]}: {tier} -- {_short_reason(site)} ({BULK_STAMP})'


def _logger_name(rel_path: str) -> str:
    return Path(rel_path).stem


def _ensure_exc_binding(source: str, match: HandlerMatch) -> tuple[str, bool]:
    """Return (possibly updated) except line text; bool = changed."""
    lines = source.splitlines()
    hline = lines[match.handler_start - 1]
    if re.search(r"\bas\s+\w+", hline):
        return hline, False
    if "except Exception" in hline:
        new = re.sub(r"except Exception(\s*:\s*)", r"except Exception as exc\1", hline, count=1)
        return new, new != hline
    return hline, False


def _narrow_except_line(source: str, match: HandlerMatch, types: tuple[str, ...]) -> tuple[str, bool]:
    lines = source.splitlines()
    hline = lines[match.handler_start - 1]
    type_str = ", ".join(types)
    if len(types) == 1:
        repl = f"except {types[0]}"
    else:
        repl = f"except ({type_str})"
    if " as exc" not in hline and " as " not in hline:
        repl += " as exc"
    new = re.sub(r"except\s+(?:Exception|\([^)]+\)|[\w.]+)", repl, hline, count=1)
    return new, new != hline


def _insert_after_line(lines: list[str], lineno: int, new_lines: list[str]) -> None:
    idx = lineno  # 1-based lineno -> insert after this line index (0-based: lineno)
    lines[idx:idx] = new_lines


def _replace_range(lines: list[str], start: int, end: int, new_lines: list[str]) -> None:
    del lines[start - 1 : end]
    for i, nl in enumerate(new_lines):
        lines.insert(start - 1 + i, nl)


def apply_delete_dead(source: str, site: CensusSite, match: HandlerMatch) -> tuple[str, bool]:
    if not (_is_pass_only_handler(source, match) and _try_body_log_or_print_only(source, match)):
        return source, False
    lines = source.splitlines()
    try_lines = _try_body_lines(source, match)
    if not try_lines:
        return source, False
    try_indent = _indent_of(lines[match.try_start - 1])
    body_indent = _indent_of(try_lines[0])
    if not body_indent.startswith(try_indent):
        return source, False
    dedent_len = len(body_indent) - len(try_indent)
    dedented: list[str] = []
    for tl in try_lines:
        if not tl.strip():
            dedented.append(tl)
        elif len(tl) >= dedent_len:
            dedented.append(try_indent + tl[dedent_len:])
        else:
            dedented.append(tl.lstrip())
    start = match.try_start - 1
    end = (match.try_node.end_lineno or match.handler_end) - 1
    new_lines = lines[:start] + dedented + lines[end + 1 :]
    return "\n".join(new_lines) + ("\n" if source.endswith("\n") else ""), True


def apply_comment(source: str, site: CensusSite, match: HandlerMatch) -> tuple[str, bool]:
    lines = source.splitlines()
    body = match.handler.body
    if not body:
        return source, False
    insert_at = body[0].lineno
    indent = _indent_of(lines[insert_at - 1])
    comment = _comment_line(site, indent)
    block = _handler_body_lines(source, match)
    if any(BULK_STAMP in ln for ln in block):
        return source, False
    _insert_after_line(lines, insert_at - 1, [comment])
    return "\n".join(lines) + ("\n" if source.endswith("\n") else ""), True


def apply_log(source: str, site: CensusSite, match: HandlerMatch) -> tuple[str, bool]:
    lines = source.splitlines()
    body = match.handler.body
    if not body:
        return source, False
    insert_at = body[0].lineno
    indent = _indent_of(lines[insert_at - 1])
    block = _handler_body_lines(source, match)
    if any(f"[{site.exc_id}]" in ln for ln in block):
        return source, False

    new_lines: list[str] = []
    exc_line, exc_changed = _ensure_exc_binding(source, match)
    if exc_changed:
        lines[match.handler_start - 1] = exc_line
        source = "\n".join(lines) + ("\n" if source.endswith("\n") else "")
        match, _ = _find_handler(source, site.line)
        if match is None:
            return source, False
        lines = source.splitlines()
        body = match.handler.body
        insert_at = body[0].lineno
        indent = _indent_of(lines[insert_at - 1])

    # Ensure logging import exists - insert at module level if missing (caller adds separately).
    log_block = [_log_line(site, indent)]
    _insert_after_line(lines, insert_at - 1, log_block)
    return "\n".join(lines) + ("\n" if source.endswith("\n") else ""), True


def apply_narrow(source: str, site: CensusSite, match: HandlerMatch) -> tuple[str, bool]:
    if not site.narrow_types:
        return source, False
    lines = source.splitlines()
    new_line, changed = _narrow_except_line(source, match, site.narrow_types)
    if not changed:
        return source, False
    lines[match.handler_start - 1] = new_line
    return "\n".join(lines) + ("\n" if source.endswith("\n") else ""), True


def _ensure_logging_import(source: str) -> tuple[str, bool]:
    if re.search(r"^import logging\b|^from logging import", source, re.M):
        return source, False
    lines = source.splitlines()
    insert_at = 0
    for i, ln in enumerate(lines):
        if ln.startswith("import ") or ln.startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, "import logging")
    return "\n".join(lines) + ("\n" if source.endswith("\n") else ""), True


def apply_site(source: str, site: CensusSite, phase: Phase, *, dry_run: bool) -> tuple[str, list[str]]:
    notes: list[str] = []
    if site.skip_reason:
        notes.append(f"SKIP {site.exc_id}: {site.skip_reason}")
        return source, notes

    match, resolved_line = _find_handler(source, site.line, max_line_delta=0)
    if match is None:
        notes.append(f"SKIP {site.exc_id}: no except handler at line {site.line}")
        return source, notes
    if resolved_line is not None and resolved_line != site.line:
        notes.append(f"LINE-REFRESH {site.exc_id}: {site.line} -> {resolved_line}")
        site.line = resolved_line

    if _already_bulk(source, match):
        notes.append(f"SKIP {site.exc_id}: already has EXCEPT-BULK marker")
        return source, notes

    actions = set(site.actions)
    if phase != "all":
        if phase not in actions:
            return source, notes
        actions = {phase}

    changed = False
    for action in ("delete-dead", "log", "narrow", "comment"):
        if action not in actions:
            continue
        before = source
        if action == "delete-dead":
            source, did = apply_delete_dead(source, site, match)
            if did:
                try:
                    ast.parse(source)
                except SyntaxError as exc:
                    notes.append(f"ROLLBACK {site.exc_id}: syntax error after delete-dead: {exc}")
                    source = before
                    did = False
            if not did:
                notes.append(f"DOWNGRADE {site.exc_id}: delete-dead -> comment-only (unsafe try body)")
                source, cdid = apply_comment(source, site, match)
                if cdid:
                    try:
                        ast.parse(source)
                    except SyntaxError as exc:
                        notes.append(f"ROLLBACK {site.exc_id}: syntax error after downgraded comment: {exc}")
                        source = before
                        cdid = False
                    else:
                        changed = True
                        notes.append(f"APPLY {site.exc_id}: comment (downgraded from delete-dead)")
                        match, _ = _find_handler(source, site.line, max_line_delta=0)
                continue
        elif action == "log":
            source, did = apply_log(source, site, match)
            if did and not dry_run:
                source, _ = _ensure_logging_import(source)
        elif action == "narrow":
            source, did = apply_narrow(source, site, match)
        elif action == "comment":
            source, did = apply_comment(source, site, match)
        else:
            did = False
        if did:
            try:
                ast.parse(source)
            except SyntaxError as exc:
                notes.append(f"ROLLBACK {site.exc_id}: syntax error after {action}: {exc}")
                source = before
                did = False
        if did:
            changed = True
            notes.append(f"APPLY {site.exc_id}: {action}")
            match, _ = _find_handler(source, site.line)
            if match is None and action != "delete-dead":
                notes.append(f"WARN {site.exc_id}: handler lost after {action}")
                break

    if not changed and not notes:
        notes.append(f"NOOP {site.exc_id}: phase={phase} actions={sorted(site.actions)}")
    return source, notes


def _git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def update_census_dispositions(census_text: str, applied_ids: Iterable[str]) -> str:
    applied = set(applied_ids)
    commit = _git_head()
    out_lines: list[str] = []

    for line in census_text.splitlines():
        updated = line
        for exc_id in applied:
            if exc_id not in line:
                continue
            if not line.startswith("|"):
                continue
            if not line.startswith(f"| {exc_id} "):
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            disp_col = parts[-2].strip()
            if "DONE" in disp_col:
                continue
            new_disp = f"disposition-DONE ({commit})"
            parts[-2] = f" {new_disp} "
            updated = "|".join(parts)
            break
        out_lines.append(updated)

    return "\n".join(out_lines) + ("\n" if census_text.endswith("\n") else "")


def run(phase: Phase, *, dry_run: bool, only_ids: frozenset[str] | None = None) -> int:
    census_text = CENSUS_PATH.read_text(encoding="utf-8")
    sites = parse_census(census_text)

    by_file: dict[str, list[CensusSite]] = {}
    for site in sites.values():
        if site.skip_reason:
            continue
        if only_ids is not None and site.exc_id not in only_ids:
            continue
        by_file.setdefault(site.rel_path, []).append(site)

    summary = {
        "applied": 0,
        "skipped": 0,
        "noop": 0,
        "errors": 0,
    }
    applied_ids: list[str] = []
    all_notes: list[str] = []

    for rel_path, file_sites in sorted(by_file.items()):
        abs_path = REPO_ROOT / rel_path
        if not abs_path.is_file():
            for s in file_sites:
                all_notes.append(f"ERROR {s.exc_id}: missing file {rel_path}")
                summary["errors"] += 1
            continue

        source = abs_path.read_text(encoding="utf-8")
        if only_ids is not None:
            _resolve_drift_lines(source, file_sites)
        file_changed = False

        for site in sorted(file_sites, key=lambda s: s.line, reverse=True):
            new_source, notes = apply_site(source, site, phase, dry_run=dry_run)
            all_notes.extend(notes)
            site_applied = False
            site_skipped = False
            site_noop = False
            site_error = False
            for note in notes:
                if note.startswith("SKIP"):
                    site_skipped = True
                elif note.startswith("APPLY"):
                    site_applied = True
                elif note.startswith("NOOP"):
                    site_noop = True
                elif note.startswith("ERROR"):
                    site_error = True
            if site_error:
                summary["errors"] += 1
            elif site_applied:
                summary["applied"] += 1
                applied_ids.append(site.exc_id)
                file_changed = True
            elif site_skipped:
                summary["skipped"] += 1
            elif site_noop:
                summary["noop"] += 1
            source = new_source

        if file_changed and not dry_run:
            abs_path.write_text(source, encoding="utf-8", newline="\n")

    if applied_ids and not dry_run:
        new_census = update_census_dispositions(census_text, applied_ids)
        CENSUS_PATH.write_text(new_census, encoding="utf-8", newline="\n")

    print(f"EXCEPT-BULK phase={phase} dry_run={dry_run}")
    print(f"  applied={summary['applied']} skipped={summary['skipped']} noop={summary['noop']} errors={summary['errors']}")
    if applied_ids:
        print(f"  touched IDs ({len(applied_ids)}): {', '.join(sorted(set(applied_ids))[:20])}{'...' if len(applied_ids)>20 else ''}")
    for note in all_notes[:200]:
        print(f"  {note}")
    if len(all_notes) > 200:
        print(f"  ... {len(all_notes) - 200} more notes")

    return 1 if summary["errors"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["delete-dead", "log", "comment", "narrow", "all"],
        default="all",
        help="Which disposition action(s) to apply (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned edits without writing files or census",
    )
    parser.add_argument(
        "--only-ids-file",
        type=Path,
        default=None,
        help="Apply only EXC IDs listed one per line (EXCEPT-BULK-2 drift rows)",
    )
    args = parser.parse_args(argv)
    only_ids: frozenset[str] | None = None
    if args.only_ids_file is not None:
        only_ids = frozenset(
            ln.strip()
            for ln in args.only_ids_file.read_text(encoding="utf-8").splitlines()
            if ln.strip().startswith("EXC-")
        )
    return run(args.phase, dry_run=args.dry_run, only_ids=only_ids)


if __name__ == "__main__":
    sys.exit(main())
