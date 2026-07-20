#!/usr/bin/env python3
"""Anchor #3 protocol v2 retry: TWO full photometry passes on draft_435 at ONE HEAD.

Pass-1 is a Cursor-run full pipeline (overwrites prior UI photometry). Milan UI backup
remains at tmp/anchor435_protocol_v2/pass1_photometry_backup until snapshot is cut.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DRAFT_ID = 435
SETUP = "NoFilter_60_2"
OUT = _ROOT / "tmp" / "anchor435_protocol_v2"


def _git_head(full: bool = True) -> str:
    args = ["rev-parse", "HEAD"] if full else ["rev-parse", "--short", "HEAD"]
    return subprocess.check_output(["git", *args], cwd=_ROOT, text=True).strip()


def _clear_photometry_outputs(phot_dir: Path) -> None:
    for name in (
        "lightcurves",
        "lightcurves_reports",
        "photometry_summary.csv",
        "pipeline_meta.json",
        "field_map.png",
        "hockey_stick_report.png",
        "suspected_variables.csv",
        "variability_candidates.csv",
        "active_targets.csv",
        "comparison_stars_per_target.csv",
        "excluded_targets.csv",
        "field_density.json",
        "_report_cache",
        "_hrd_cache",
        "_crossmatch",
    ):
        p = phot_dir / name
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.is_file():
            p.unlink(missing_ok=True)


def _read_prov(meta_path: Path) -> dict[str, Any]:
    if not meta_path.is_file():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "provenance": meta.get("provenance") or {},
        "matched_world2pix_identity_p95_px": meta.get("matched_world2pix_identity_p95_px"),
        "matched_world2pix_identity_p99_px": meta.get("matched_world2pix_identity_p99_px"),
        "matched_world2pix_identity_n": meta.get("matched_world2pix_identity_n"),
    }


def _run_full(cfg, draft: Path, ps: Path, lights: Path, phot: Path) -> None:
    from database import VyvarDatabase
    from photometry_core import run_full_photometry_pipeline

    db = VyvarDatabase(cfg.database_path)
    try:
        run_full_photometry_pipeline(
            masterstar_fits_path=ps / "MASTERSTAR.fits",
            variable_targets_csv=ps / "variable_targets.csv",
            masterstars_csv=ps / "masterstars_full_match.csv",
            per_frame_csv_dir=lights,
            detrended_aligned_dir=lights,
            output_dir=phot,
            cfg=cfg,
            db=db,
            draft_id=DRAFT_ID,
            progress_cb=lambda m: print(m, flush=True),
        )
    finally:
        db.conn.close()


def main() -> int:
    from config import AppConfig
    from tests.photometry_sha import compute_photometry_sha

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig()
    draft = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    ps = draft / "platesolve" / SETUP
    lights = draft / "detrended_aligned" / "lights" / SETUP
    phot = ps / "photometry"
    meta_path = phot / "pipeline_meta.json"

    head = _git_head(True)
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": head,
        "draft_id": DRAFT_ID,
        "setup": SETUP,
        "draft_root": str(draft),
        "note": "both passes are Cursor full-photometry at this HEAD",
    }

    # Preserve Milan UI backup if not already present
    ui_backup = OUT / "pass1_photometry_backup"
    if not ui_backup.is_dir() and phot.is_dir():
        shutil.copytree(phot, ui_backup)
        print(f"Saved UI/prior photometry backup -> {ui_backup}", flush=True)
    else:
        print(f"Keeping existing backup at {ui_backup}", flush=True)

    # --- PASS 1 ---
    print("=== PASS1 run_full_photometry_pipeline ===", flush=True)
    _clear_photometry_outputs(phot)
    phot.mkdir(parents=True, exist_ok=True)
    _run_full(cfg, draft, ps, lights, phot)
    core1, n1 = compute_photometry_sha(draft, include_comp_qa=False)
    ext1, ne1 = compute_photometry_sha(draft, include_comp_qa=True)
    pass1_meta = _read_prov(meta_path)
    report["pass1"] = {
        "core_sha": core1,
        "core_n": n1,
        "extended_sha": ext1,
        "extended_n": ne1,
        "meta": pass1_meta,
    }
    pass1_phot_backup = OUT / "cursor_pass1_photometry_backup"
    if pass1_phot_backup.exists():
        shutil.rmtree(pass1_phot_backup)
    shutil.copytree(phot, pass1_phot_backup)
    (OUT / "pass1_report.json").write_text(
        json.dumps(report["pass1"], indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS1 core={core1[:16]}... n={n1} ext={ext1[:16]}... n={ne1}", flush=True)
    print(f"PASS1 provenance: {json.dumps(pass1_meta.get('provenance') or {}, indent=2)}", flush=True)

    # --- PASS 2 ---
    print("=== PASS2 run_full_photometry_pipeline ===", flush=True)
    _clear_photometry_outputs(phot)
    phot.mkdir(parents=True, exist_ok=True)
    _run_full(cfg, draft, ps, lights, phot)
    core2, n2 = compute_photometry_sha(draft, include_comp_qa=False)
    ext2, ne2 = compute_photometry_sha(draft, include_comp_qa=True)
    pass2_meta = _read_prov(meta_path)
    report["pass2"] = {
        "core_sha": core2,
        "core_n": n2,
        "extended_sha": ext2,
        "extended_n": ne2,
        "meta": pass2_meta,
    }
    print(f"PASS2 core={core2[:16]}... n={n2} ext={ext2[:16]}... n={ne2}", flush=True)
    print(f"PASS2 provenance: {json.dumps(pass2_meta.get('provenance') or {}, indent=2)}", flush=True)

    p1 = pass1_meta.get("provenance") or {}
    p2 = pass2_meta.get("provenance") or {}
    byte_core = core1 == core2 and n1 == n2
    byte_ext = ext1 == ext2 and ne1 == ne2
    same_git = str(p1.get("git_hash") or "") == str(p2.get("git_hash") or "") == head
    dirty_ok = (p1.get("git_dirty_code") is False) and (p2.get("git_dirty_code") is False)
    labbe_ok = p1.get("labbe_rng_seed_policy") == p2.get("labbe_rng_seed_policy") == "content_frame_hash_v1"
    report["sha_gate"] = {
        "byte_identical_core": byte_core,
        "byte_identical_extended": byte_ext,
        "same_git_hash": same_git,
        "git_dirty_code_false": dirty_ok,
        "labbe_policy_ok": labbe_ok,
        "pass": bool(byte_core and byte_ext and same_git and dirty_ok and labbe_ok),
    }
    (OUT / "protocol_v2_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["sha_gate"], indent=2), flush=True)

    if not report["sha_gate"]["pass"]:
        print("STOP: SHA gate FAILED - restoring Cursor pass1 photometry", flush=True)
        if phot.exists():
            shutil.rmtree(phot, ignore_errors=True)
        shutil.copytree(pass1_phot_backup, phot)
        return 2
    print("SHA gate PASS - draft photometry left as pass2 (=pass1)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
