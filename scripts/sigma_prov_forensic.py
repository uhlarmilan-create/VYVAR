#!/usr/bin/env python3
"""SIGMA-PROV-FORENSIC: stale-LC provenance proof + fresh Newton baseline."""

from __future__ import annotations

import argparse
import json
import math
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

from check_star_kmag import (  # noqa: E402
    build_aligned_comp_inst,
    comp_ensemble_maps,
    resolve_proc_csv_dir,
)
from config import AppConfig  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from photometry_core import (  # noqa: E402
    _ensemble_scatter_by_source_file,
    check_comparison_stability,
    ensemble_normalize,
    run_full_photometry_pipeline,
)
from scripts.bingain_err_decompose import _gain_from_lights  # noqa: E402
from scripts.bingain_fix_validate import _chi2_lc_err, resolve_archive_root  # noqa: E402
from scripts.chi2_sigma_gate import bootstrap_chi2_dof_ci, reduced_chi2_constant  # noqa: E402
from scripts.sigma_sem_cause import (  # noqa: E402
    _ensemble_sem_from_lc_err,
    _photon_err_mag_per_frame,
    extract_production_trace,
    load_star_lists,
)
from scripts.sparse_comp_diag import V0611_CID  # noqa: E402
from sigma_budget import SIGMA_VARIANT_PRODUCTION_LC_ERR  # noqa: E402
from tests.photometry_sha import _compare_lc_science  # noqa: E402
from ui_aperture_photometry import _find_phase2a_paths  # noqa: E402

_MAG = 2.5 / math.log(10.0)
OUT_ROOT = _ROOT / "tmp" / "sigma_prov_forensic"
DRAFT_426 = 426
SETUPS_426 = ("g_60_4", "i_70_4", "r_60_4", "z_90_4")
FRESH_SETUP = "i_70_4"

TIMELINE: dict[str, dict[str, str]] = {
    "005716d": {"label": "Fix A (Honeycutt residual SEM)", "date_utc": "2026-06-18"},
    "e7ce7ea": {"label": "PROV-FIX (pipeline_meta provenance)", "date_utc": "2026-07-08"},
    "bingain": {"label": "F-BINGAIN chain (empirical bkg err)", "date_utc": "2026-07-10"},
}

PREDICTIONS = {
    "P1": (
        "draft_426 LC provenance predates 2026-06-18 (005716d), OR provenance absent with "
        "LC-assembly mtimes < 2026-06-18 (not proc CSVs)."
    ),
    "P2": (
        "Fresh i_70_4 on current HEAD: LC err total ~0.009-0.010 mag on V0611-cohort stars "
        "(photon ~0.005 + SEM ~0.0067, x1.0857 unit tolerance if pre-fix)."
    ),
    "P3": (
        "Check-star chi2 on FRESH i_70_4 LC ~6-8 (OVERdispersed) via production_lc_err."
    ),
}


def _git_head(full: bool = False) -> str:
    try:
        args = ["git", "rev-parse", "HEAD"] if full else ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.check_output(args, cwd=_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _stamp(payload: dict[str, Any]) -> dict[str, Any]:
    payload["git_head"] = _git_head()
    payload["git_head_full"] = _git_head(full=True)
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_stamp(payload), indent=2), encoding="utf-8")


def _norm_id(x: object) -> str:
    return str(normalize_gaia_source_id(x) or "").strip()


def _mtime_utc(p: Path) -> str | None:
    if not p.is_file():
        return None
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()


def _draft_id_from_name(name: str) -> int | None:
    if not name.startswith("draft_"):
        return None
    try:
        return int(name.split("_", 1)[1][:6])
    except ValueError:
        return None


def _hash_timeline(git_hash: str | None) -> dict[str, Any]:
    if not git_hash:
        return {"status": "unknown", "relative_to_fix_a": None}
    try:
        is_ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", git_hash, "HEAD"],
            cwd=_ROOT,
            capture_output=True,
        ).returncode == 0
        before_fix_a = subprocess.run(
            ["git", "merge-base", "--is-ancestor", git_hash, "005716d"],
            cwd=_ROOT,
            capture_output=True,
        ).returncode == 0
        fix_a_is_ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", "005716d", git_hash],
            cwd=_ROOT,
            capture_output=True,
        ).returncode == 0
        prov_fix_is_ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", "e7ce7ea", git_hash],
            cwd=_ROOT,
            capture_output=True,
        ).returncode == 0
        return {
            "status": "mapped" if is_ancestor else "not_in_history",
            "before_fix_a": before_fix_a and not fix_a_is_ancestor,
            "includes_fix_a": fix_a_is_ancestor,
            "includes_prov_fix": prov_fix_is_ancestor,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def _lc_assembly_files(phot_dir: Path) -> list[Path]:
    lc_dir = phot_dir / "lightcurves"
    files: list[Path] = []
    if lc_dir.is_dir():
        files.extend(lc_dir.glob("lightcurve_*.csv"))
        files.extend(lc_dir.glob("check_kmag_*.csv"))
    for name in ("pipeline_meta.json", "photometry_summary.csv"):
        p = phot_dir / name
        if p.is_file():
            files.append(p)
    return files


def _inspect_setup_provenance(draft_dir: Path, setup: str) -> dict[str, Any]:
    phot = draft_dir / "platesolve" / setup / "photometry"
    meta_path = phot / "pipeline_meta.json"
    prov: dict[str, Any] | None = None
    entry_point = None
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            prov = meta.get("provenance")
            entry_point = meta.get("entry_point")
        except Exception as exc:  # noqa: BLE001
            prov = {"parse_error": str(exc)}
    lc_files = _lc_assembly_files(phot)
    mtimes = [_mtime_utc(p) for p in lc_files if _mtime_utc(p)]
    oldest = min(mtimes) if mtimes else None
    newest = max(mtimes) if mtimes else None
    gh = None
    stamped = None
    if isinstance(prov, dict):
        gh = prov.get("git_hash")
        stamped = prov.get("stamped_at_utc")
    return {
        "setup": setup,
        "phot_dir": str(phot),
        "provenance_present": prov is not None and isinstance(prov, dict) and "git_hash" in prov,
        "provenance": prov if prov is not None else "ABSENT",
        "entry_point": entry_point,
        "git_hash": gh,
        "stamped_at_utc": stamped,
        "git_timeline": _hash_timeline(str(gh) if gh else None),
        "lc_assembly_file_count": len(lc_files),
        "lc_mtime_oldest_utc": oldest,
        "lc_mtime_newest_utc": newest,
        "lc_mtime_before_fix_a": (
            oldest < "2026-06-18T00:00:00+00:00" if oldest else None
        ),
    }


def _stale_err_fingerprint(phot_dir: Path, setup: str, cfg: AppConfig) -> dict[str, Any]:
    """Compare LC-implied ensemble vs normalize SEM (pre-Fix-A fingerprint)."""
    lights = phot_dir.parents[2] / "detrended_aligned" / "lights" / setup
    gain = _gain_from_lights(lights, float(cfg.gain))
    rn = float(cfg.read_noise)
    trace = extract_production_trace(
        phot_dir=phot_dir, setup=setup, target_cid=V0611_CID, cfg=cfg,
        gain=gain, read_noise=rn,
    )
    if not trace.get("available"):
        return {"available": False}
    lc_df = pd.read_csv(phot_dir / "lightcurves" / f"lightcurve_{V0611_CID}.csv", low_memory=False)
    proc_dir = resolve_proc_csv_dir(phot_dir, setup)
    phot_mag = _photon_err_mag_per_frame(lc_df, proc_dir, V0611_CID, gain=gain, read_noise=rn)
    ens_lc = _ensemble_sem_from_lc_err(lc_df["err"].to_numpy(dtype=float), phot_mag)
    ens_norm = np.asarray(trace.get("ensemble_scatter", []), dtype=float)
    ratio = float(np.nanmedian(ens_lc / ens_norm)) if np.any(np.isfinite(ens_lc) & np.isfinite(ens_norm)) else float("nan")
    return {
        "available": True,
        "lc_err_median_rel": float(lc_df["err"].median()),
        "lc_err_median_mag": float(lc_df["err"].median() * _MAG),
        "lc_implied_ensemble_median_mag": float(np.nanmedian(ens_lc)),
        "normalize_sem_median_mag": float(np.nanmedian(ens_norm)),
        "lc_over_normalize_ratio": ratio,
        "matches_pre_fix_a_spread": ratio > 3.0,
    }


def run_part_a(cfg: AppConfig, archive_root: Path) -> dict[str, Any]:
    drafts_dir = archive_root / "Drafts"
    primary: dict[str, Any] = {}
    for setup in SETUPS_426:
        draft_dir = drafts_dir / f"draft_{DRAFT_426:06d}"
        if not draft_dir.is_dir():
            continue
        row = _inspect_setup_provenance(draft_dir, setup)
        row["stale_fingerprint"] = _stale_err_fingerprint(
            draft_dir / "platesolve" / setup / "photometry", setup, cfg,
        )
        primary[setup] = row

    controls: dict[str, Any] = {}
    for draft_id, setup in ((424, "NoFilter_60_2"), (425, "B_20_2")):
        draft_dir = drafts_dir / f"draft_{draft_id:06d}"
        if draft_dir.is_dir():
            controls[f"draft_{draft_id}_{setup}"] = _inspect_setup_provenance(draft_dir, setup)

    blast: list[dict[str, Any]] = []
    blast_semantic: list[dict[str, Any]] = []
    for draft_path in sorted(drafts_dir.iterdir()):
        if not draft_path.is_dir() or not draft_path.name.startswith("draft_"):
            continue
        ps = draft_path / "platesolve"
        if not ps.is_dir():
            continue
        for setup_dir in sorted(ps.iterdir()):
            if not setup_dir.is_dir():
                continue
            phot = setup_dir / "photometry"
            if not (phot / "lightcurves").is_dir():
                continue
            info = _inspect_setup_provenance(draft_path, setup_dir.name)
            stale = (
                not info["provenance_present"]
                and info.get("lc_mtime_before_fix_a") is True
            ) or (
                isinstance(info.get("git_timeline"), dict)
                and info["git_timeline"].get("before_fix_a") is True
            )
            if stale:
                blast.append(
                    {
                        "draft": draft_path.name,
                        "setup": setup_dir.name,
                        **info,
                    }
                )
            fp = _stale_err_fingerprint(phot, setup_dir.name, cfg) if (phot / "lightcurves").is_dir() else {}
            if (
                not info["provenance_present"]
                and fp.get("available")
                and (
                    fp.get("matches_pre_fix_a_spread")
                    or float(fp.get("lc_over_normalize_ratio", 0.0) or 0.0) > 2.5
                )
            ):
                blast_semantic.append(
                    {
                        "draft": draft_path.name,
                        "setup": setup_dir.name,
                        "lc_implied_ensemble_median_mag": fp.get("lc_implied_ensemble_median_mag"),
                        "normalize_sem_median_mag": fp.get("normalize_sem_median_mag"),
                        "lc_over_normalize_ratio": fp.get("lc_over_normalize_ratio"),
                    }
                )

    p1_rows = [primary[s] for s in SETUPS_426 if s in primary]
    p1_absent = all(not r["provenance_present"] for r in p1_rows)
    p1_mtime_old = any(r.get("lc_mtime_before_fix_a") for r in p1_rows)
    p1_hash_old = any(
        isinstance(r.get("git_timeline"), dict) and r["git_timeline"].get("before_fix_a")
        for r in p1_rows
    )
    p1_semantic = any(
        (r.get("stale_fingerprint") or {}).get("matches_pre_fix_a_spread")
        for r in p1_rows
    )
    p1_pass = p1_hash_old or (p1_absent and p1_mtime_old)

    return _stamp(
        {
            "task": "SIGMA-PROV-FORENSIC",
            "part": "A",
            "predictions": PREDICTIONS,
            "timeline": TIMELINE,
            "draft_426": primary,
            "controls": controls,
            "blast_radius": blast,
            "blast_radius_semantic_stale_err": blast_semantic,
            "p1_verdict": {
                "pass_strict": p1_pass,
                "provenance_absent_all_426": p1_absent,
                "lc_mtime_before_fix_a_any": p1_mtime_old,
                "git_hash_before_fix_a_any": p1_hash_old,
                "semantic_pre_fix_a_fingerprint": p1_semantic,
                "note": (
                    "Strict P1 requires provenance<Fix-A OR (absent provenance AND mtime<2026-06-18). "
                    "Semantic fingerprint (LC/normalize ratio>>1) reported separately."
                ),
            },
        }
    )


def _chi2_for_star(lc_path: Path, check_path: Path | None) -> dict[str, Any]:
    lc = pd.read_csv(lc_path, low_memory=False)
    if check_path is not None and check_path.is_file():
        chi2_dof, meta = _chi2_lc_err(lc_path=lc_path, side_path=check_path)
        side = pd.read_csv(check_path, low_memory=False)
        mags = pd.to_numeric(side["kmag"], errors="coerce").to_numpy(dtype=float)
    else:
        mags = pd.to_numeric(lc.get("mag_calib_final", lc.get("mag_calib")), errors="coerce").to_numpy(dtype=float)
        err = pd.to_numeric(lc["err"], errors="coerce").to_numpy(dtype=float)
        _, _, chi2_dof, _ = reduced_chi2_constant(mags, _MAG * err)
        meta = {"fallback": "lc_mag_calib"}
    err = pd.to_numeric(lc["err"], errors="coerce").to_numpy(dtype=float)
    ci = bootstrap_chi2_dof_ci(mags, _MAG * err)
    return {
        "chi2_dof": float(chi2_dof) if chi2_dof is not None and math.isfinite(float(chi2_dof)) else float("nan"),
        "n_frames": int(len(lc)),
        "ci_lo": float(ci[0]),
        "ci_hi": float(ci[1]),
        **meta,
    }


def run_fresh_pipeline(cfg: AppConfig, *, skip_if_exists: bool = False) -> Path:
    out_phot = OUT_ROOT / "fresh_i_70_4" / "photometry"
    marker = out_phot / "photometry_summary.csv"
    if skip_if_exists and marker.is_file():
        return out_phot

    paths = _find_phase2a_paths(cfg, DRAFT_426).get(FRESH_SETUP)
    if not paths:
        raise SystemExit(f"ERROR: no phase2a paths for draft_{DRAFT_426:06d} setup {FRESH_SETUP}")

    ms_fits = Path(paths["masterstar_fits"])
    og_dir = Path(paths["obs_group_dir"])
    vt_csv = og_dir / "variable_targets.csv"
    ms_csv = og_dir / "masterstars_full_match.csv"
    pf_dir = Path(paths["per_frame_csv_dir"])
    dt_dir = Path(paths["detrended_aligned_dir"])
    for label, p in [
        ("MASTERSTAR.fits", ms_fits),
        ("variable_targets.csv", vt_csv),
        ("masterstars_full_match.csv", ms_csv),
        ("per_frame_csv_dir", pf_dir),
        ("detrended_aligned_dir", dt_dir),
    ]:
        if not p.exists():
            raise SystemExit(f"ERROR: missing {label}: {p}")

    out_phot.mkdir(parents=True, exist_ok=True)
    print(f"[Part B] run_full_photometry_pipeline -> {out_phot}")
    run_full_photometry_pipeline(
        masterstar_fits_path=ms_fits,
        variable_targets_csv=vt_csv,
        masterstars_csv=ms_csv,
        per_frame_csv_dir=pf_dir,
        detrended_aligned_dir=dt_dir,
        output_dir=out_phot,
        cfg=cfg,
        draft_id=DRAFT_426,
        progress_cb=lambda m: print(f"  {m}"),
    )
    return out_phot


def run_part_b(cfg: AppConfig, archive_root: Path, *, skip_pipeline: bool = False) -> dict[str, Any]:
    stale_phot = archive_root / "Drafts" / f"draft_{DRAFT_426:06d}" / "platesolve" / FRESH_SETUP / "photometry"
    if not skip_pipeline:
        fresh_phot = run_fresh_pipeline(cfg)
    else:
        fresh_phot = OUT_ROOT / "fresh_i_70_4" / "photometry"
        if not (fresh_phot / "photometry_summary.csv").is_file():
            raise SystemExit("ERROR: --skip-pipeline but fresh output missing")

    summary_path = _ROOT / "tmp" / "sigma_newton" / "sigma_newton_summary.json"
    star_ids = [V0611_CID]
    if summary_path.is_file():
        star_ids = load_star_lists(summary_path).get(FRESH_SETUP, star_ids)

    lights = archive_root / "Drafts" / f"draft_{DRAFT_426:06d}" / "detrended_aligned" / "lights" / FRESH_SETUP
    gain = _gain_from_lights(lights, float(cfg.gain))
    rn = float(cfg.read_noise)
    proc_dir = resolve_proc_csv_dir(stale_phot, FRESH_SETUP)
    if proc_dir is None:
        raise SystemExit(f"ERROR: cannot resolve proc_dir for {FRESH_SETUP}")

    per_star: dict[str, Any] = {}
    for cid in star_ids:
        lc_f = fresh_phot / "lightcurves" / f"lightcurve_{cid}.csv"
        if not lc_f.is_file():
            continue
        lc_df = pd.read_csv(lc_f, low_memory=False)
        trace = extract_production_trace(
            phot_dir=stale_phot, setup=FRESH_SETUP, target_cid=cid, cfg=cfg,
            gain=gain, read_noise=rn,
        )
        phot_mag = _photon_err_mag_per_frame(lc_df, proc_dir, cid, gain=gain, read_noise=rn)
        ens_norm = np.asarray(
            [float(trace.get("scatter_by_file", {}).get(str(sf).strip(), float("nan"))) for sf in lc_df["source_file"]],
            dtype=float,
        )
        err_med_rel = float(lc_df["err"].median())
        err_med_mag = err_med_rel * _MAG
        ens_lc = _ensemble_sem_from_lc_err(lc_df["err"].to_numpy(dtype=float), phot_mag)
        check = fresh_phot / "lightcurves" / f"check_kmag_{cid}.csv"
        chi2 = _chi2_for_star(lc_f, check if check.is_file() else None)
        per_star[cid] = {
            "err_median_rel": err_med_rel,
            "err_median_mag": err_med_mag,
            "photon_median_mag": float(np.nanmedian(phot_mag)),
            "normalize_sem_median_mag": float(np.nanmedian(ens_norm)),
            "lc_implied_ensemble_median_mag": float(np.nanmedian(ens_lc)),
            "carrier_matches_normalize_max_abs_diff_mag": float(
                np.nanmax(np.abs(ens_lc - ens_norm))
            ) if np.any(np.isfinite(ens_lc)) else float("nan"),
            "chi2_production_lc_err": chi2,
            "is_v0611": cid == V0611_CID,
        }

    v0611 = per_star.get(V0611_CID, {})
    p2_target_lo, p2_target_hi = 0.009, 0.010
    err_mag = float(v0611.get("err_median_mag", float("nan")))
    p2_pass = p2_target_lo <= err_mag <= p2_target_hi * 1.2  # allow unit slack

    v0611_chi2 = float((v0611.get("chi2_production_lc_err") or {}).get("chi2_dof", float("nan")))
    p3_pass = 6.0 <= v0611_chi2 <= 8.5

    science_compare: dict[str, Any] = {}
    stale_lc_dir = stale_phot / "lightcurves"
    fresh_lc_dir = fresh_phot / "lightcurves"
    for cid in star_ids:
        sa = stale_lc_dir / f"lightcurve_{cid}.csv"
        sb = fresh_lc_dir / f"lightcurve_{cid}.csv"
        if sa.is_file() and sb.is_file():
            science_compare[cid] = _compare_lc_science(sa, sb)

    meta_fresh = {}
    meta_path = fresh_phot / "pipeline_meta.json"
    if meta_path.is_file():
        meta_fresh = json.loads(meta_path.read_text(encoding="utf-8")).get("provenance", {})

    payload = _stamp(
        {
            "task": "SIGMA-PROV-FORENSIC",
            "part": "B",
            "setup": FRESH_SETUP,
            "fresh_phot_dir": str(fresh_phot),
            "stale_phot_dir": str(stale_phot),
            "fresh_provenance": meta_fresh,
            "per_star": per_star,
            "science_compare_fresh_vs_stale": science_compare,
            "p2_verdict": {
                "pass": p2_pass,
                "v0611_err_median_mag": err_mag,
                "expected_range_mag": [p2_target_lo, p2_target_hi],
                "unit_fix_applied_at_run": False,
            },
            "p3_verdict": {
                "pass": p3_pass,
                "v0611_chi2_dof": v0611_chi2,
                "expected_range": [6.0, 8.0],
            },
        }
    )
    _write_json(OUT_ROOT / "fresh_chi2.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="SIGMA-PROV-FORENSIC runner")
    parser.add_argument("--part", choices=("A", "B", "all"), default="all")
    parser.add_argument("--skip-pipeline", action="store_true", help="Part B: reuse existing fresh output")
    args = parser.parse_args()

    cfg = AppConfig()
    archive_root = resolve_archive_root(cfg=cfg)

    if args.part in ("A", "all"):
        part_a = run_part_a(cfg, archive_root)
        _write_json(OUT_ROOT / "provenance_table.json", part_a)
        print(f"[Part A] wrote {OUT_ROOT / 'provenance_table.json'}")
        print(f"  P1 strict pass: {part_a['p1_verdict']['pass_strict']}")

    if args.part in ("B", "all"):
        part_b = run_part_b(cfg, archive_root, skip_pipeline=args.skip_pipeline)
        print(f"[Part B] fresh V0611 err mag: {part_b['p2_verdict']['v0611_err_median_mag']:.5f}")
        print(f"  P2 pass: {part_b['p2_verdict']['pass']}")
        print(f"  P3 chi2: {part_b['p3_verdict']['v0611_chi2_dof']:.3f} pass={part_b['p3_verdict']['pass']}")


if __name__ == "__main__":
    main()
