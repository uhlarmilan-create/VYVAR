"""One-shot merge script for VYVAR_STATE.md - UTF-8 safe."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = Path(r"C:\Users\uhlar\Downloads\VYVAR_STATE (1).md")
OUT = ROOT / "docs" / "VYVAR_STATE.md"


def git_show(path: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"HEAD:{path}"],
        capture_output=True,
        check=True,
    )
    return r.stdout.decode("utf-8")


def line_index(lines: list[str], marker: str, start: int = 0) -> int:
    for i in range(start, len(lines)):
        if lines[i].strip() == marker.strip():
            return i
    raise ValueError(f"Marker not found: {marker!r} (from line {start})")


def slice_lines(lines: list[str], start: int, end: int) -> str:
    chunk = lines[start:end]
    while chunk and not chunk[-1].strip():
        chunk.pop()
    while chunk and chunk[0].strip() in ("", "---"):
        chunk.pop(0)
    return "\n".join(chunk).strip()


def strip_section_heading(block: str, heading: str) -> str:
    lines = block.splitlines()
    if lines and lines[0].strip() == heading.strip():
        lines = lines[1:]
    return "\n".join(lines).strip()


def main() -> None:
    repo_lines = git_show("docs/VYVAR_STATE.md").splitlines()
    local_lines = LOCAL.read_text(encoding="utf-8").splitlines()

    i_repo_open = line_index(repo_lines, "## Open TODOs (backlog)")
    i_repo_known = line_index(repo_lines, "## Known issues / poznamky")

    session_27 = slice_lines(repo_lines, 0, i_repo_open)
    open_todos_repo = strip_section_heading(
        slice_lines(repo_lines, i_repo_open, i_repo_known), "## Open TODOs (backlog)"
    )
    known_repo = strip_section_heading(
        slice_lines(repo_lines, i_repo_known, len(repo_lines)),
        "## Known issues / poznamky",
    )

    i_loc_open = line_index(local_lines, "## Open TODOs (backlog)")
    i_loc_known = line_index(local_lines, "## Known issues / next session")
    i_loc_20 = line_index(local_lines, "# VYVAR SESSION SUMMARY - 20.5.2026")
    i_loc_19 = line_index(local_lines, "# VYVAR SESSION SUMMARY - 19.5.2026")
    i_loc_18 = line_index(local_lines, "# VYVAR SESSION SUMMARY - 18.5.2026")

    session_21 = slice_lines(local_lines, 0, i_loc_open)
    open_todos_local = strip_section_heading(
        slice_lines(local_lines, i_loc_open, i_loc_known), "## Open TODOs (backlog)"
    )
    known_local = strip_section_heading(
        slice_lines(local_lines, i_loc_known, i_loc_20), "## Known issues / next session"
    )
    session_20 = slice_lines(local_lines, i_loc_20, i_loc_19)
    def no_trailing_hr(s: str) -> str:
        s = s.rstrip()
        while s.endswith("---"):
            s = s[: -3].rstrip()
        return s

    session_19 = no_trailing_hr(slice_lines(local_lines, i_loc_19, i_loc_18))
    archive = slice_lines(local_lines, i_loc_18, len(local_lines))

    merged_open = f"""## Open TODOs (backlog)

_Merged from repo (27.5.2026) and local archive (21.5.2026). CLOSED items marked [OK]._

### Active (repo - 27.5.2026)

{open_todos_repo}

### Active (local - 21.5.2026)

{open_todos_local}

### Reference - CLOSED / completed (from local archive)

| ID | Status | Notes |
|----|--------|-------|
| TODO-GS6b | [OK] CLOSED | AAVSO Extended Format validator (`scripts/validate_aavso_export.py`) - 20.5.2026 |
| CQ-1, CQ-2, CQ-4 | [OK] CLOSED | `run_phase2a`, `render_live_view`, `solve_wcs_with_local_gaia` splits - 20.5.2026 |
| PERF-9 | [OK] CLOSED | Vectorized haversine VSX match - 20.5.2026 |
| TODO-23, TODO-25, TODO-16, TODO-17 | [OK] CLOSED | Adaptive match radius, Gaia completeness UI, crossmatch coords - 20.5.2026 |
| TODO-ALG-2, TODO-ALG-3, TODO-ALG-4, TODO-ALG-5 | [OK] CLOSED | Savitzky-Golay, temporal binning, Democratic Detrender, PyTICS - 20.5.2026 |
| TODO-44, TODO-8 | [OK] CLOSED | Role-aware aperture; ePSF infrastructure - 20.5.2026 (Bootes -> TODO-8-BOO) |
| PERF-1 ... PERF-10 | [OK] CLOSED | Performance series - 19.5.2026 |
| CQ-3, TODO-35 | [OK] CLOSED | Comp selection split; SysRem MVP - 19.5.2026 |
| TODO-ALG-2 ... TODO-ALG-5, TODO-44, TODO-8 | [OK] CLOSED | See 19.5.2026 session backlog table |
"""

    merged_known = f"""## Known issues / next session

_Merged from repo (27.5.2026) and local (21.5.2026)._

### Repo (27.5.2026)

{known_repo}

### Local (21.5.2026)

{known_local}
"""

    parts = [
        "# VYVAR - Development State",
        "",
        "Last updated: 2026-05-27 (merged: repo session 27.5.2026 + local archive 17.5-21.5.2026)",
        "",
        "---",
        "",
        session_27,
        "",
        "---",
        "",
        merged_open.strip(),
        "",
        "---",
        "",
        merged_known.strip(),
        "",
        "---",
        "",
        session_21,
        "",
        "---",
        "",
        session_20,
        "",
        "---",
        "",
        session_19,
        "",
        "---",
        "",
        archive,
    ]

    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(OUT.read_text(encoding='utf-8').splitlines())} lines)")


if __name__ == "__main__":
    main()
