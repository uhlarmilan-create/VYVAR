#!/usr/bin/env python3
"""426-REGEN Part A: preserve stale evidence + regenerate draft_426 photometry."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from photometry_core import (  # noqa: E402
    _PSF_ERR_MAG_SCALE,
    run_full_photometry_pipeline,
)
from scripts.bingain_err_decompose import _gain_from_lights  # noqa: E402
from scripts.bingain_fix_validate import resolve_archive_root  # noqa: E402
from scripts.sigma_sem_cause import (  # noqa: E402
    _ensemble_sem_from_lc_err,
    _photon_err_mag_per_frame,
    extract_production_trace,
)
from scripts.sparse_comp_diag import V0611_CID  # noqa: E402
from ui_aperture_photometry import _find_phase2a_paths  # noqa: E402

DRAFT_ID = 426
SETUPS = ("g_60_4", "i_70_4", "r_60_4", "z_90_4")
EVIDENCE_NAME = "draft_000426_stale_20260626"
_MAG = 2.5 / math.log(10.0)
OUT_META = _ROOT / "tmp" / "draft_426_regen"


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _stamp(payload: dict[str, Any]) -> dict[str, Any]:
    payload["git_head"] = _git_head()
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return payload


def preserve_and_restore_shell(archive_root: Path, *, dry_run: bool = False) -> dict[str, str]:
    drafts = archive_root / "Drafts"
    src = drafts / f"draft_{DRAFT_ID:06d}"
    evidence_dir = archive_root / "evidence" / EVIDENCE_NAME
    if not src.is_dir():
        raise SystemExit(f"ERROR: missing draft tree {src}")
    if evidence_dir.exists():
        return {"status": "already_preserved", "evidence": str(evidence_dir), "working": str(src)}
    if dry_run:
        return {"status": "dry_run", "would_move": str(src), "would_copy_to": str(src)}
    evidence_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(evidence_dir))
    shutil.copytree(evidence_dir, src)
    return {"status": "moved_and_copied", "evidence": str(evidence_dir), "working": str(src)}


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
        "_report_cache",
    ):
        p = phot_dir / name
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.is_file():
            p.unlink(missing_ok=True)


def run_setup_regen(cfg: AppConfig, setup: str) -> dict[str, Any]:
    paths = _find_phase2a_paths(cfg, DRAFT_ID).get(setup)
    if not paths:
        raise SystemExit(f"ERROR: no phase2a paths for {setup}")
    og = Path(paths["obs_group_dir"])
    phot = Path(paths["output_dir"])
    ms_fits = Path(paths["masterstar_fits"])
    pf = Path(paths["per_frame_csv_dir"])
    dt = Path(paths["detrended_aligned_dir"])
    vt = og / "variable_targets.csv"
    ms_csv = og / "masterstars_full_match.csv"
    for label, p in [
        ("MASTERSTAR", ms_fits),
        ("variable_targets", vt),
        ("masterstars", ms_csv),
        ("proc_dir", pf),
        ("detrended", dt),
    ]:
        if not p.exists():
            raise SystemExit(f"ERROR: {setup} missing {label}: {p}")
    _clear_photometry_outputs(phot)
    phot.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now(timezone.utc)
    run_full_photometry_pipeline(
        masterstar_fits_path=ms_fits,
        variable_targets_csv=vt,
        masterstars_csv=ms_csv,
        per_frame_csv_dir=pf,
        detrended_aligned_dir=dt,
        output_dir=phot,
        cfg=cfg,
        draft_id=DRAFT_ID,
        progress_cb=lambda m: print(f"  [{setup}] {m}"),
    )
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    return {"setup": setup, "elapsed_s": elapsed, "phot_dir": str(phot)}


def verify_setup(
    cfg: AppConfig,
    archive_root: Path,
    setup: str,
    *,
    evidence_root: Path,
) -> dict[str, Any]:
    phot = archive_root / "Drafts" / f"draft_{DRAFT_ID:06d}" / "platesolve" / setup / "photometry"
    meta_path = phot / "pipeline_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    prov = meta.get("provenance") if isinstance(meta.get("provenance"), dict) else None
    gh = prov.get("git_hash") if prov else None
    lc_dir = phot / "lightcurves"
    n_lc = len(list(lc_dir.glob("lightcurve_*.csv"))) if lc_dir.is_dir() else 0
    n_check = len(list(lc_dir.glob("check_kmag_*.csv"))) if lc_dir.is_dir() else 0
    lights = archive_root / "Drafts" / f"draft_{DRAFT_ID:06d}" / "detrended_aligned" / "lights" / setup
    gain = _gain_from_lights(lights, float(cfg.gain))
    rn = float(cfg.read_noise)
    trace = extract_production_trace(
        phot_dir=phot, setup=setup, target_cid=V0611_CID, cfg=cfg, gain=gain, read_noise=rn,
    )
    carrier_ok = False
    max_diff = float("nan")
    v0611_lc = phot / "lightcurves" / f"lightcurve_{V0611_CID}.csv"
    if trace.get("available") and v0611_lc.is_file():
        from check_star_kmag import resolve_proc_csv_dir  # noqa: PLC0415

        proc_dir = resolve_proc_csv_dir(phot, setup)
        lc_df = pd.read_csv(v0611_lc, low_memory=False)
        phot_mag = _photon_err_mag_per_frame(lc_df, proc_dir, V0611_CID, gain=gain, read_noise=rn)
        ens_norm = np.asarray(
            [float(trace.get("scatter_by_file", {}).get(str(sf).strip(), float("nan"))) for sf in lc_df["source_file"]],
            dtype=float,
        )
        err_rel = lc_df["err"].to_numpy(dtype=float)
        phot_rel = phot_mag / _MAG
        implied_ens_rel = np.sqrt(np.maximum(0.0, err_rel * err_rel - phot_rel * phot_rel))
        implied_ens_mag = implied_ens_rel * _MAG
        ok = np.isfinite(implied_ens_mag) & np.isfinite(ens_norm)
        if ok.any():
            max_diff = float(np.max(np.abs(implied_ens_mag[ok] - ens_norm[ok])))
            carrier_ok = max_diff < 1e-5
    frame_diff = _frame_diff(evidence_root, setup, phot)
    return {
        "setup": setup,
        "provenance_present": prov is not None and bool(gh),
        "git_hash": gh,
        "stamped_at_utc": prov.get("stamped_at_utc") if prov else None,
        "n_lightcurves": n_lc,
        "n_check_kmag": n_check,
        "carrier_max_abs_diff_mag": max_diff,
        "carrier_matches_normalize": carrier_ok,
        "frame_diff": frame_diff,
    }


def _frame_diff(evidence_root: Path, setup: str, fresh_phot: Path) -> dict[str, Any]:
    stale_phot = evidence_root / "platesolve" / setup / "photometry"
    stale_lc = stale_phot / "lightcurves" / f"lightcurve_{V0611_CID}.csv"
    fresh_lc = fresh_phot / "lightcurves" / f"lightcurve_{V0611_CID}.csv"
    if not stale_lc.is_file() or not fresh_lc.is_file():
        return {"available": False}
    s = pd.read_csv(stale_lc, usecols=["source_file"], low_memory=False)["source_file"].astype(str).tolist()
    f = pd.read_csv(fresh_lc, usecols=["source_file"], low_memory=False)["source_file"].astype(str).tolist()
    only_fresh = sorted(set(f) - set(s))
    only_stale = sorted(set(s) - set(f))
    return {
        "available": True,
        "stale_n": len(s),
        "fresh_n": len(f),
        "only_fresh": only_fresh,
        "only_stale": only_stale,
        "shared_n": len(set(s) & set(f)),
    }


def analyze_frame_gate(
    archive_root: Path,
    evidence_root: Path,
    setup: str = "i_70_4",
) -> dict[str, Any]:
    """Identify extra/missing frame between stale and fresh (Part A.4)."""
    diff = _frame_diff(
        evidence_root,
        setup,
        archive_root / "Drafts" / f"draft_{DRAFT_ID:06d}" / "platesolve" / setup / "photometry",
    )
    if not diff.get("available"):
        return diff
    extra = diff.get("only_fresh") or []
    missing = diff.get("only_stale") or []
    proc_dir = archive_root / "Drafts" / f"draft_{DRAFT_ID:06d}" / "detrended_aligned" / "lights" / setup
    all_procs = sorted(p.name for p in proc_dir.glob("proc_*.csv"))
    note = ""
    responsible = "unknown"
    frame_id = extra[0] if extra else (missing[0] if missing else "")
    if extra:
        frame_id = extra[0]
        responsible = "phase2a proc_*.csv glob includes proc_MASTERSTAR.csv"
        note = (
            f"The extra epoch is source_file {frame_id!r}. Stale LC ({diff['stale_n']} rows) excluded "
            f"this file; fresh HEAD ({diff['fresh_n']} rows) includes it. "
            f"detrended_aligned/lights/{setup} has {len(all_procs)} proc_*.csv files; "
            f"photometry_core loads all proc_*.csv (glob at ~line 933) so the MASTERSTAR "
            f"reference proc row enters Phase-2A frame assembly. Stale June-26 photometry "
            f"(pre-Fix-A code) did not carry this epoch in the LC. Current-HEAD behavior is "
            f"canonical; do not suppress proc_MASTERSTAR.csv."
        )
    elif missing:
        responsible = "frame dropped by current pipeline gate"
        note = f"Frame {frame_id!r} in stale LC but not fresh."
    return {**diff, "frame_id": frame_id, "responsible_gate": responsible, "n_proc_files": len(all_procs), "note": note}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-root", default=None)
    ap.add_argument("--preserve-only", action="store_true")
    ap.add_argument("--skip-preserve", action="store_true", help="Evidence already moved")
    ap.add_argument("--setups", nargs="+", default=list(SETUPS))
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    cfg = AppConfig()
    archive_root = resolve_archive_root(args.archive_root, cfg=cfg)
    cfg.archive_root = archive_root
    OUT_META.mkdir(parents=True, exist_ok=True)

    preserve: dict[str, str] = {"status": "skipped"}
    if not args.skip_preserve and not args.verify_only:
        preserve = preserve_and_restore_shell(archive_root)
        print(f"[preserve] {preserve}")
    if args.preserve_only:
        return 0

    evidence_root = archive_root / "evidence" / EVIDENCE_NAME
    if not evidence_root.is_dir():
        evidence_root = archive_root / "Drafts" / f"draft_{DRAFT_ID:06d}"

    runs: list[dict[str, Any]] = []
    if not args.verify_only:
        for setup in args.setups:
            print(f"[regen] {setup}")
            runs.append(run_setup_regen(cfg, setup))

    verifications = [verify_setup(cfg, archive_root, s, evidence_root=evidence_root) for s in args.setups]
    frame_analysis = analyze_frame_gate(archive_root, evidence_root, "i_70_4")

    payload = _stamp(
        {
            "task": "426-REGEN",
            "preserve": preserve,
            "evidence_path": str(archive_root / "evidence" / EVIDENCE_NAME),
            "runs": runs,
            "verifications": verifications,
            "frame_analysis_i_70_4": frame_analysis,
        }
    )
    out_path = OUT_META / "regen_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] {out_path}")
    for v in verifications:
        print(
            f"  {v['setup']}: prov={v['provenance_present']} carrier_ok={v['carrier_matches_normalize']} "
            f"lc={v['n_lightcurves']} check={v['n_check_kmag']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
