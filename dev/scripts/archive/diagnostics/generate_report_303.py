#!/usr/bin/env python3
"""Generate Summary Measure Report PDF for draft_000303 (no pipeline run)."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DRAFT_DIR = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000303")


def _find_obs_group(draft_dir: Path) -> str:
    ps = draft_dir / "platesolve"
    if ps.is_dir():
        setups = sorted(
            p.name for p in ps.iterdir() if p.is_dir() and (p / "photometry" / "photometry_summary.csv").is_file()
        )
        if setups:
            return setups[0]
    lights = draft_dir / "detrended_aligned" / "lights"
    if lights.is_dir():
        for p in sorted(lights.iterdir()):
            if p.is_dir() and any(p.glob("proc_*.csv")):
                return p.name
    raise FileNotFoundError(f"No obs_group with photometry_summary.csv under {draft_dir}")


def main() -> int:
    draft_dir = DRAFT_DIR.resolve()
    if not draft_dir.is_dir():
        print(f"ERROR: draft directory not found: {draft_dir}", file=sys.stderr)
        return 1

    obs_group = _find_obs_group(draft_dir)
    print(f"draft_dir: {draft_dir}")
    print(f"obs_group: {obs_group}")

    from photometry_report import generate_photometry_report

    pdf_path = generate_photometry_report(
        draft_dir=draft_dir,
        obs_group=obs_group,
        output_pdf=None,
        report_draft_label="draft_000303",
        report_title="VYVAR - Summary Measure Report",
    )
    if pdf_path is None:
        print("ERROR: generate_photometry_report returned None (reportlab missing?)", file=sys.stderr)
        return 1

    out = Path(pdf_path).resolve()
    print(f"SUCCESS: {out}")
    return 0 if out.is_file() else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
