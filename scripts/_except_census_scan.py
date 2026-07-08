#!/usr/bin/env python3
"""Scan production .py for silent broad-except handlers; emit census markdown."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {"tests", "scripts", "tmp", "sandbox", "Archive", ".git", "__pycache__", "agent-tools", "GAIA_DR3", ".worktrees"}
EXCLUDE_FILES: set[str] = set()

SCIENCE_MODULES = {
    "pipeline.py",
    "photometry_core.py",
    "photometry_phase2a.py",
    "photometry_phase1.py",
    "photometry.py",
    "comp_selection_per_target.py",
    "comp_qa_core.py",
    "comp_qa.py",
    "comp_pool_rms.py",
    "calibration_core.py",
    "k2_extinction.py",
    "psf_photometry.py",
    "crowding_index.py",
    "trust_flag_core.py",
    "param_resolver.py",
    "time_utils.py",
    "masterstar_context.py",
    "hrd_analysis.py",
    "band_classify.py",
    "vyvar_platesolver.py",
    "vyvar_blind_solver.py",
}

INTEGRITY_MODULES = {
    "photometry_report.py",
    "export_reports.py",
    "pdf_report.py",
    "database.py",
    "draft_provenance.py",
    "hrd_analysis.py",
}

UI_PREFIXES = ("ui_", "vyvar_ui")
UI_MODULES = {
    "app.py",
    "ui_components.py",
    "ui_calibration_library.py",
    "ui_database_explorer.py",
    "ui_photometry.py",
    "ui_photometry_quality.py",
    "ui_settings.py",
    "vyvar_ui_status.py",
}


@dataclass
class Site:
    file: str
    line: int
    exc_types: str
    handler: str
    handler_detail: str
    context: str


def iter_production_py() -> list[Path]:
    out: list[Path] = []
    for p in sorted(ROOT.rglob("*.py")):
        rel = p.relative_to(ROOT)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if rel.as_posix() in EXCLUDE_FILES:
            continue
        out.append(p)
    return out


def is_broad_except(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    if isinstance(handler.type, ast.Name):
        return handler.type.id in ("Exception", "BaseException")
    if isinstance(handler.type, ast.Tuple):
        names = []
        for elt in handler.type.elts:
            if isinstance(elt, ast.Name):
                names.append(elt.id)
        return "Exception" in names or "BaseException" in names
    return False


def classify_handler(body: list[ast.stmt]) -> tuple[str, str]:
    if not body:
        return "empty", ""
    if len(body) == 1:
        st = body[0]
        if isinstance(st, ast.Pass):
            return "pass", ""
        if isinstance(st, ast.Continue):
            return "continue", ""
        if isinstance(st, ast.Return) and st.value is None:
            return "return", ""
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
            return _classify_call(st.value)
    # Multi-statement: check if only pass/continue/return or sub-ERROR logs
    kinds = []
    for st in body:
        if isinstance(st, ast.Pass):
            kinds.append("pass")
        elif isinstance(st, ast.Continue):
            kinds.append("continue")
        elif isinstance(st, ast.Return):
            kinds.append("return")
        elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
            k, d = _classify_call(st.value)
            kinds.append(k if k else "call")
        else:
            kinds.append("other")
    if all(k in ("pass", "continue", "return", "log-below-ERROR", "log_event") for k in kinds):
        return "mixed-silent", "+".join(kinds)
    return "other", "+".join(kinds)


def _classify_call(call: ast.Call) -> tuple[str, str]:
    func = call.func
    name = ""
    if isinstance(func, ast.Attribute):
        name = func.attr
        if isinstance(func.value, ast.Name) and func.value.id == "logging":
            level = name
            if level in ("debug", "info", "warning"):
                return "log-below-ERROR", f"logging.{level}"
            if level in ("error", "exception", "critical"):
                return "log-ERROR+", f"logging.{level}"
    elif isinstance(func, ast.Name):
        name = func.id
        if name == "log_event":
            return "log_event", "log_event"
    return "", name


def is_silent(handler_kind: str) -> bool:
    return handler_kind in {
        "pass",
        "continue",
        "return",
        "log-below-ERROR",
        "log_event",
        "mixed-silent",
        "empty",
    }


def exc_type_str(handler: ast.ExceptHandler) -> str:
    if handler.type is None:
        return "bare except"
    return ast.unparse(handler.type)


def snippet(lines: list[str], lineno: int, radius: int = 1) -> str:
    i0 = max(0, lineno - 1 - radius)
    i1 = min(len(lines), lineno + radius)
    return " / ".join(l.strip() for l in lines[i0:i1] if l.strip())


def infer_tier(rel: str, context: str) -> str:
    base = Path(rel).name
    if base in SCIENCE_MODULES or rel.startswith("orchestrator/"):
        # orchestrator may be mixed; mark ? unless obvious
        if "ui" in base.lower():
            return "T3-UI"
        if base in INTEGRITY_MODULES:
            return "T2-INTEGRITY"
        return "T1-SCIENCE"
    if base in INTEGRITY_MODULES:
        return "T2-INTEGRITY"
    if base.startswith(UI_PREFIXES) or base in UI_MODULES:
        return "T3-UI"
    if base.startswith("ui_") or "plotly" in context.lower() or "streamlit" in context.lower():
        return "T3-UI"
    if base in {"config.py", "infolog.py", "citations.py", "gaia_catalog_id.py"}:
        return "T4-LEGIT"
    if "best effort" in context.lower() or "fallback" in context.lower():
        return "T4-LEGIT"
    return "?"


def propose_disposition(tier: str, handler: str, rel: str) -> str:
    if tier == "T1-SCIENCE":
        return "fix now"
    if tier == "T2-INTEGRITY":
        return "narrow+log"
    if tier == "T3-UI":
        return "leave+comment"
    if tier == "T4-LEGIT":
        return "narrow+comment"
    return "triage (?)"


def what_lost(tier: str, context: str, handler: str) -> str:
    ctx = context[:120]
    if tier == "T1-SCIENCE":
        return f"science path may skip or use stale defaults ({ctx})"
    if tier == "T2-INTEGRITY":
        return f"report/export may omit or misstate ({ctx})"
    if tier == "T3-UI":
        return f"UI diagnostic/plot only ({ctx})"
    if tier == "T4-LEGIT":
        return f"optional enrichment skipped ({ctx})"
    return f"intent unclear ({ctx})"


def scan_file(path: Path) -> list[Site]:
    rel = path.relative_to(ROOT).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []

    sites: list[Site] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            if not is_broad_except(handler):
                continue
            kind, detail = classify_handler(handler.body)
            if not is_silent(kind):
                continue
            ctx = snippet(lines, handler.lineno, radius=2)
            sites.append(
                Site(
                    file=rel,
                    line=handler.lineno,
                    exc_types=exc_type_str(handler),
                    handler=kind,
                    handler_detail=detail,
                    context=ctx,
                )
            )
    return sites


CENSUS_MARKER = "## Census"
_ROW_RE = re.compile(r"^\|\s*(EXC-\d+)\s*\|\s*`([^:`]+):(\d+)`\s*\|.*\|\s*[^|]*\|\s*$")


@dataclass
class OldRow:
    idn: int
    file: str
    line: int
    disp: str
    raw: str  # verbatim census row (curated prose preserved)


def parse_existing_census(path: Path):
    """Return (preamble, old_rows, tail) preserving verbatim row text.

    - preamble: everything up to and including the census table header separator row.
    - old_rows: list of OldRow in document order (raw = verbatim line).
    - tail: any trailing non-row lines after the last EXC row (kept verbatim).

    Stable-ID mode reuses existing EXC IDs and curated prose, updating ONLY the
    ``file:line`` in each row so line numbers track the tree. Rows whose site is no
    longer silent (disposition contains FIXED) keep their historical line untouched.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    preamble_end = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith(CENSUS_MARKER):
            for j in range(i, min(i + 6, len(lines))):
                if set(lines[j].strip()) <= {"|", "-"} and "-" in lines[j]:
                    preamble_end = j
                    break
            break
    if preamble_end is None:
        raise RuntimeError("Could not locate '## Census' table header in existing census")
    preamble = "\n".join(lines[: preamble_end + 1])

    old_rows: list[OldRow] = []
    for ln in lines[preamble_end + 1 :]:
        m = _ROW_RE.match(ln)
        if not m:
            continue
        idn = int(m.group(1).split("-")[1])
        fname, lineno = m.group(2), int(m.group(3))
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        disp = cells[6] if len(cells) >= 7 else ""
        old_rows.append(OldRow(idn=idn, file=fname, line=lineno, disp=disp, raw=ln))
    return preamble, old_rows


def main() -> None:
    all_sites: list[Site] = []
    for p in iter_production_py():
        all_sites.extend(scan_file(p))

    all_sites.sort(key=lambda s: (s.file, s.line))

    out = ROOT / "docs" / "VYVAR_EXCEPT_CENSUS.md"

    # Stable-ID mode: if a census already exists, preserve its EXC IDs AND its curated
    # tranche prose/dispositions, updating ONLY the `file:line` per row so line numbers
    # track the current tree. Matching is by within-file line order: the new scan is
    # the old set minus rows already surfaced (disposition contains FIXED). For files
    # where that count invariant does not hold, lines are left untouched and flagged.
    if out.exists():
        try:
            preamble, old_rows = parse_existing_census(out)
        except RuntimeError:
            preamble = None
        if out.exists() and preamble is not None:
            from collections import defaultdict

            new_by_file: dict[str, list[int]] = defaultdict(list)
            for s in all_sites:
                new_by_file[s.file].append(s.line)
            for lst in new_by_file.values():
                lst.sort()

            # surviving (non-FIXED) old rows per file, in document (line) order
            surviving_by_file: dict[str, list[OldRow]] = defaultdict(list)
            for r in old_rows:
                if "FIXED" not in r.disp.upper():
                    surviving_by_file[r.file].append(r)

            newline_map: dict[int, int] = {}  # id(OldRow) -> new line
            mismatched_files: list[str] = []
            updated = 0
            for fname, survivors in surviving_by_file.items():
                new_lines = new_by_file.get(fname, [])
                if len(survivors) == len(new_lines):
                    for r, nl in zip(survivors, new_lines):
                        newline_map[id(r)] = nl
                        if nl != r.line:
                            updated += 1
                else:
                    mismatched_files.append(
                        f"{fname} (surviving-rows={len(survivors)}, scan-sites={len(new_lines)})"
                    )

            out_rows: list[str] = []
            for r in old_rows:
                nl = newline_map.get(id(r))
                if nl is not None and nl != r.line:
                    out_rows.append(r.raw.replace(f"`{r.file}:{r.line}`", f"`{r.file}:{nl}`", 1))
                else:
                    out_rows.append(r.raw)

            out.write_text(preamble + "\n" + "\n".join(out_rows) + "\n", encoding="utf-8")
            print(f"Stable-ID line refresh: {len(old_rows)} rows kept, {updated} line numbers updated.")
            print(f"  scan found {len(all_sites)} silent sites in tree.")
            if mismatched_files:
                print("  DEFERRED (lines left as-is, count invariant unmet):")
                for mf in mismatched_files:
                    print(f"    {mf}")
            return

    tiers: dict[str, int] = {}
    rows: list[str] = []
    for i, s in enumerate(all_sites, start=1):
        sid = f"EXC-{i:04d}"
        tier = infer_tier(s.file, s.context)
        tiers[tier] = tiers.get(tier, 0) + 1
        disp = propose_disposition(tier, s.handler, s.file)
        lost = what_lost(tier, s.context, s.handler)
        handler_s = s.handler
        if s.handler_detail:
            handler_s = f"{s.handler} ({s.handler_detail})"
        rows.append(
            f"| {sid} | `{s.file}:{s.line}` | {s.exc_types} | {handler_s} | {tier} | {lost} | {disp} |"
        )

    header = """# VYVAR - silent broad-except census (EXCEPT-BATCH-S0)

Working ledger for the **299-defensive cluster**, **F-EXCEPT-TIER1**, and **TIER1-UI-DEBT**
reconciliation. Generated by `sandbox/_except_census_scan.py` (re-run to refresh counts).

**Scope:** production `.py` under repo root, excluding `tests/`, `scripts/`, `tmp/`, `sandbox/`,
`Archive/`. **Criterion:** `except:` / `except Exception` / `except BaseException` whose handler
is `pass`, `continue`, bare `return`, `log_event`, or `logging` below ERROR.

**Tier legend:** T1-SCIENCE (photometry/calibration/comp) - T2-INTEGRITY (reports/IO/DB) -
T3-UI (cosmetic) - T4-LEGIT (correct best-effort fallback) - **?** (needs human triage).

## Summary counts

| Tier | Count |
|------|------:|
"""
    tier_order = ["T1-SCIENCE", "T2-INTEGRITY", "T3-UI", "T4-LEGIT", "?"]
    for t in tier_order:
        header += f"| {t} | {tiers.get(t, 0)} |\n"
    header += f"| **TOTAL** | **{len(all_sites)}** |\n\n"
    header += (
        "**Reconciliation notes:** F-EXCEPT-TIER1 cited **160** sites - matches "
        "`pipeline.py` pass/continue broad-except (~95) + `photometry_core.py` (~66). "
        "This census is wider: all production modules, all silent handler kinds "
        "(pass/continue/return/log-below-ERROR/log_event). TIER1-UI-DEBT cited **38** SAFE "
        "UI/plotly `pass` sites (subset of T3-UI here). **299-defensive cluster** = phased "
        "fix queue for T1-SCIENCE + T2-INTEGRITY dispositions marked fix-now / narrow+log.\n\n"
    )
    header += "## Census\n\n"
    header += "| ID | Site | Except | Handler | Tier | What gets silently lost | Disposition |\n"
    header += "|----|------|--------|---------|------|-------------------------|-------------|\n"
    out.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(all_sites)} sites to {out}")
    for t in tier_order:
        print(f"  {t}: {tiers.get(t, 0)}")


if __name__ == "__main__":
    main()
