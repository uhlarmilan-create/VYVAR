#!/usr/bin/env python3
"""ANCHOR-RECUT sigma-notes: Step 1 diff proof + Step 2 double --full pipeline runs."""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))
sys.path.insert(0, str(ROOT / "dev"))
sys.path.insert(0, str(ROOT / "dev" / "scripts"))
sys.path.insert(0, str(ROOT / "dev" / "tests"))

from tests.photometry_sha import compute_photometry_sha, photometry_sha_files  # noqa: E402

SETUP = "NoFilter_60_2"
SNAPSHOT = ROOT / "Archive" / "Drafts" / "draft_000435_snapshot_skysurface_20260716"
OUT = ROOT / "tmp" / "anchor_recut"


def step1_proof(*, run_dir: Path) -> dict:
    snap = SNAPSHOT
    if not run_dir.is_dir():
        raise SystemExit(f"missing run_dir {run_dir}")
    if not snap.is_dir():
        raise SystemExit(f"missing snapshot {snap}")
    rf = {p.relative_to(run_dir).as_posix(): p for p in photometry_sha_files(run_dir)}
    sf = {p.relative_to(snap).as_posix(): p for p in photometry_sha_files(snap)}
    only_r = sorted(set(rf) - set(sf))
    only_s = sorted(set(sf) - set(rf))
    diffs: list[dict] = []
    for k in sorted(set(rf) & set(sf)):
        rb, sb = rf[k].read_bytes(), sf[k].read_bytes()
        if rb != sb:
            diffs.append({"rel": k, "run_bytes": len(rb), "snap_bytes": len(sb), "delta": len(rb) - len(sb)})
    cq = [d for d in diffs if "comp_quality_" in d["rel"]]
    lc = [d for d in diffs if "lightcurve_" in d["rel"]]
    other = [d for d in diffs if d not in cq and d not in lc]
    proof = {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "only_run": only_r,
        "only_snap": only_s,
        "n_diff": len(diffs),
        "comp_quality_diffs": cq,
        "lightcurve_diffs": lc,
        "other_diffs": other,
        "sample_unified_diffs": [],
    }
    if other:
        proof["STOP"] = "non-comp_quality diffs present"
        return proof
    if len(cq) != 19:
        proof["STOP"] = f"expected 19 comp_quality diffs, got {len(cq)}"
        return proof
    if lc:
        proof["STOP"] = "lightcurve diffs present"
        return proof
    for d in cq[:3]:
        k = d["rel"]
        rtxt = rf[k].read_text(encoding="utf-8")
        stxt = sf[k].read_text(encoding="utf-8")
        udiff = list(
            difflib.unified_diff(
                stxt.splitlines(),
                rtxt.splitlines(),
                fromfile="snapshot",
                tofile="run",
                lineterm="",
            )
        )
        proof["sample_unified_diffs"].append({"file": k, "lines": udiff})
        for line in udiff:
            if line.startswith("+") and not line.startswith("+++"):
                if "\u03c3" in line or "sigma" not in line:
                    if "note" in line.lower():
                        proof["STOP"] = f"unexpected + line in {k}: {line!r}"
    core_r, n_r = compute_photometry_sha(run_dir)
    core_s, n_s = compute_photometry_sha(snap)
    ext_r, ne_r = compute_photometry_sha(run_dir, include_comp_qa=True)
    ext_s, ne_s = compute_photometry_sha(snap, include_comp_qa=True)
    proof["run_core_sha"] = core_r
    proof["snap_core_sha"] = core_s
    proof["run_extended_sha"] = ext_r
    proof["snap_extended_sha"] = ext_s
    proof["run_core_n"] = n_r
    proof["run_extended_n"] = ne_r
    return proof


def _run_pipeline_once(tag: str) -> Path:
    from config import AppConfig  # noqa: PLC0415
    from database import VyvarDatabase  # noqa: PLC0415
    from except_fix_counters import reset_except_fix_counters  # noqa: PLC0415
    from photometry_core import run_full_photometry_pipeline  # noqa: PLC0415
    from tools.reference_seed import seed_reference_observatory  # noqa: PLC0415

    cfg = AppConfig()
    cfg.k2_mode = "literature"
    cfg.save_lightcurve_png = False
    draft = Path(cfg.archive_root) / "Drafts" / "draft_000435"
    ps = draft / "platesolve" / SETUP
    lights = draft / "detrended_aligned" / "lights" / SETUP
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = OUT / f"{tag}_{ts}"
    out_phot = work / "platesolve" / SETUP / "photometry"
    out_phot.mkdir(parents=True, exist_ok=True)
    reset_except_fix_counters()
    db = VyvarDatabase(cfg.database_path)
    seed_reference_observatory(db)
    try:
        run_full_photometry_pipeline(
            masterstar_fits_path=ps / "MASTERSTAR.fits",
            variable_targets_csv=ps / "variable_targets.csv",
            masterstars_csv=ps / "masterstars_full_match.csv",
            per_frame_csv_dir=lights,
            detrended_aligned_dir=lights,
            output_dir=out_phot,
            cfg=cfg,
            db=db,
            draft_id=435,
        )
    finally:
        try:
            db.conn.close()
        except Exception:  # noqa: BLE001
            pass
    return work


def step2_double_run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== anchor recut full run A ===", flush=True)
    run_a = _run_pipeline_once("runA")
    print("=== anchor recut full run B ===", flush=True)
    run_b = _run_pipeline_once("runB")
    core_a, n_a = compute_photometry_sha(run_a)
    core_b, n_b = compute_photometry_sha(run_b)
    ext_a, ne_a = compute_photometry_sha(run_a, include_comp_qa=True)
    ext_b, ne_b = compute_photometry_sha(run_b, include_comp_qa=True)
    same_core = core_a == core_b and n_a == n_b
    same_ext = ext_a == ext_b and ne_a == ne_b
    return {
        "run_a": str(run_a.relative_to(ROOT)),
        "run_b": str(run_b.relative_to(ROOT)),
        "core_sha_a": core_a,
        "core_sha_b": core_b,
        "extended_sha_a": ext_a,
        "extended_sha_b": ext_b,
        "core_n": n_a,
        "extended_n": ne_a,
        "byte_identical_core": same_core,
        "byte_identical_extended": same_ext,
    }


def patch_snapshot_from_run(run_dir: Path) -> dict:
    """Copy SHA-set files that differ from snapshot (expected: 19 comp_quality only)."""
    snap = SNAPSHOT
    rf = {p.relative_to(run_dir).as_posix(): p for p in photometry_sha_files(run_dir)}
    sf = {p.relative_to(snap).as_posix(): p for p in photometry_sha_files(snap)}
    copied: list[str] = []
    for k in sorted(set(rf) & set(sf)):
        if rf[k].read_bytes() != sf[k].read_bytes():
            dest = snap / k
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(rf[k].read_bytes())
            copied.append(k)
    core, nc = compute_photometry_sha(snap)
    ext, ne = compute_photometry_sha(snap, include_comp_qa=True)
    return {"copied": copied, "snapshot_core_sha": core, "snapshot_extended_sha": ext, "core_n": nc, "extended_n": ne}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1-only", action="store_true")
    ap.add_argument("--run-dir", type=Path, default=ROOT / "tmp" / "session_baseline" / "20260720T153735Z")
    ap.add_argument("--step2", action="store_true", help="Two fresh pipeline runs + compare")
    ap.add_argument("--patch-snapshot", type=Path, help="After step2, patch snapshot from this run dir")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"generated_at_utc": datetime.now(timezone.utc).isoformat()}
    if args.step1_only or not args.step2:
        report["step1"] = step1_proof(run_dir=args.run_dir)
        print(json.dumps(report["step1"], indent=2))
        if report["step1"].get("STOP"):
            (OUT / "step1_STOP.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return 1
    if args.step2:
        report["step2"] = step2_double_run()
        print(json.dumps(report["step2"], indent=2))
        if not report["step2"]["byte_identical_core"]:
            (OUT / "step2_STOP.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return 2
    if args.patch_snapshot:
        report["patch"] = patch_snapshot_from_run(args.patch_snapshot)
        print(json.dumps(report["patch"], indent=2))
    (OUT / "anchor_recut_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
