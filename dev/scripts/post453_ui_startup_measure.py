#!/usr/bin/env python3
"""POST-453 Part 2: measure pre-artifact wall time (scan + import copy)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
sys.path.insert(0, str(SRC))

CTX = REPO / "dev" / "results" / "context" / "session_20260727_post453"


def _gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024**3)


def main() -> None:
    from config import AppConfig
    from importer import smart_scan_source
    from pipeline import AstroPipeline

    cfg = AppConfig()
    # BO CVn raw on this machine (same night as drafts 451-453)
    candidates = [
        REPO / "Archive" / "Raw" / "BO_CVn_2026-07-26",
        REPO / "Archive" / "Raw" / "BO_CVn",
        Path(r"C:\ASTRO\data\BO_CVn"),
    ]
    source = next((p for p in candidates if p.is_dir()), None)
    if source is None:
        # fall back: draft 451 non_calibrated copy source from provenance
        draft451 = REPO / "Archive" / "Drafts" / "draft_000451" / "non_calibrated" / "lights"
        if draft451.is_dir():
            source = draft451.parent.parent
        else:
            print("No raw source found")
            sys.exit(1)

    CTX.mkdir(parents=True, exist_ok=True)
    phases: list[tuple[str, float, str]] = []

    t_wall = time.perf_counter()
    t0 = time.perf_counter()
    scan = smart_scan_source(str(source), cfg=cfg)
    phases.append(("smart_scan_source", time.perf_counter() - t0, f"n_files={len(getattr(scan, 'files', []) or [])}"))

    # Import copy only - use dry tmp archive
    import tempfile

    tmp_archive = Path(tempfile.mkdtemp(prefix="vyvar_post453_"))
    pipeline = AstroPipeline(cfg)
    t0 = time.perf_counter()
    from importer import smart_import_session

    plan = scan
    t_import_start = time.perf_counter()
    result = smart_import_session(
        plan=plan,
        pipeline=pipeline,
        id_equipment=1,
        id_telescope=1,
        id_location=None,
        cfg=cfg,
    )
    t_import = time.perf_counter() - t_import_start
    phases.append(("smart_import_session", t_import, f"draft_id={getattr(result, 'draft_id', None)}"))

    draft_dir = Path(str(result.archive_path)).resolve()
    if draft_dir.name.casefold() == "non_calibrated":
        draft_dir = draft_dir.parent
    first_art = None
    t_first = None
    for fp in sorted(draft_dir.rglob("*")):
        if fp.is_file():
            first_art = fp
            t_first = fp.stat().st_mtime
            break

    wall = time.perf_counter() - t_wall
    vol_gb = _gb(draft_dir / "non_calibrated" / "lights") if (draft_dir / "non_calibrated" / "lights").is_dir() else _gb(draft_dir)

    lines = [
        "phase,seconds,notes",
        *[f"{a},{b:.3f},{c}" for a, b, c in phases],
        f"wall_to_first_artifact,{wall:.3f},first={first_art}",
        f"copy_volume_gb,{vol_gb:.4f},",
        f"throughput_gb_s,{vol_gb / max(t_import, 1e-6):.4f},import_only",
    ]
    (CTX / "ui_startup_phases.csv").write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    print("Wrote", CTX / "ui_startup_phases.csv")


if __name__ == "__main__":
    main()
