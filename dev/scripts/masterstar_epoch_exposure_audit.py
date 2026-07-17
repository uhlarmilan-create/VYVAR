#!/usr/bin/env python3
"""Exposure audit: masterstar proc CSV and phantom LC epochs across archive drafts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from proc_frame_store import is_masterstar_proc_name  # noqa: E402


def _draft_dirs(archive_root: Path) -> list[Path]:
    drafts = archive_root / "Drafts"
    if not drafts.is_dir():
        return []
    out = sorted(drafts.glob("draft_*"))
    return [p for p in out if p.is_dir() and "snapshot" not in p.name.lower()]


def _setups(draft_dir: Path) -> list[str]:
    ps = draft_dir / "platesolve"
    if not ps.is_dir():
        return []
    return sorted(d.name for d in ps.iterdir() if d.is_dir())


def audit_setup(draft_dir: Path, setup: str) -> dict:
    proc_hits: list[str] = []
    lights = draft_dir / "detrended_aligned" / "lights" / setup
    if lights.is_dir():
        for p in lights.rglob("proc_*.csv"):
            if is_masterstar_proc_name(p):
                proc_hits.append(str(p.relative_to(draft_dir)))

    lc_with_ms = 0
    lc_files = 0
    phot = draft_dir / "platesolve" / setup / "photometry" / "lightcurves"
    if phot.is_dir():
        for lc in phot.glob("lightcurve_*.csv"):
            try:
                sf = pd.read_csv(lc, usecols=["source_file"])["source_file"].astype(str)
                lc_files += 1
                if any(is_masterstar_proc_name(s) for s in sf.unique()):
                    lc_with_ms += 1
            except (OSError, ValueError, KeyError):
                continue

    return {
        "setup": setup,
        "masterstar_proc_paths": proc_hits,
        "masterstar_proc_present": bool(proc_hits),
        "lc_files": lc_files,
        "lc_with_masterstar_epoch": lc_with_ms,
    }


def audit_draft(draft_dir: Path) -> dict:
    draft_id = draft_dir.name.replace("draft_", "")
    rows = [audit_setup(draft_dir, s) for s in _setups(draft_dir)]
    return {"draft": draft_id, "setups": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-root", type=Path, default=_ROOT / "Archive")
    ap.add_argument("--out", type=Path, default=_ROOT / "tmp" / "masterstar_epoch_audit.json")
    args = ap.parse_args()

    report = {
        "archive_root": str(args.archive_root.resolve()),
        "drafts": [audit_draft(d) for d in _draft_dirs(args.archive_root)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.out)
    for d in report["drafts"]:
        for row in d["setups"]:
            if row["masterstar_proc_present"] or row["lc_with_masterstar_epoch"]:
                print(
                    d["draft"],
                    row["setup"],
                    "proc",
                    row["masterstar_proc_present"],
                    "lc_phantom",
                    row["lc_with_masterstar_epoch"],
                    "/",
                    row["lc_files"],
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
