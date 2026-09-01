#!/usr/bin/env python3
"""F-BINGAIN-1 FIX acceptance: re-export proc CSVs + Phase 2A with empirical err."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from photometry_core import (  # noqa: E402
    ERR_BKG_SOURCE_COL,
    ERR_BKG_SOURCE_EMPIRICAL,
    ERR_BKG_SOURCE_HOWELL_FALLBACK,
    SIGMA_BKG_AP_COL,
    run_phase2a,
)
from pipeline import export_per_frame_catalogs  # noqa: E402
from scripts.bingain_fix_validate import resolve_archive_root  # noqa: E402
from scripts.bingain_patch_sigma_bkg import patch_setup  # noqa: E402


ACCEPTANCE_CASES: list[tuple[int, str]] = [
    (424, "NoFilter_60_2"),
    (425, "B_20_2"),
    (425, "V_20_2"),
    (425, "R_20_2"),
    (426, "g_60_4"),
    (426, "i_70_4"),
    (426, "r_60_4"),
    (426, "z_90_4"),
]

NON_ERR_PROC_COLS = [
    "catalog_id",
    "name",
    "x",
    "y",
    "dao_flux",
    "flux",
    "noise_floor_adu",
    "sky_adu_per_px_annulus",
    "aperture_r_px",
    "peak_max_adu",
    "fwhm_estimate_px",
    "flux_small",
    "flux_large",
]


def _fwhm_from_header(hdr: fits.Header) -> float:
    for key in ("VY_FWHM_GAUSS", "VY_FWHM_GAUSSIAN", "VY_FWHM"):
        try:
            v = float(hdr.get(key, float("nan")))
            if math.isfinite(v) and 0.5 < v < 30.0:
                return v
        except (TypeError, ValueError):
            continue
    return 4.5


def _provenance_stats(proc_dir: Path) -> dict[str, Any]:
    from photometry_core import ERR_BKG_SOURCE_HOWELL_SCALED

    files = sorted(proc_dir.glob("proc_*.csv"))
    n_rows = 0
    n_emp = 0
    n_fb = 0
    n_scaled = 0
    n_other = 0
    for p in files:
        try:
            df = pd.read_csv(p, usecols=[ERR_BKG_SOURCE_COL], low_memory=False)
        except ValueError:
            continue
        except Exception:  # noqa: BLE001
            continue
        col = df[ERR_BKG_SOURCE_COL].astype(str)
        n_rows += int(len(col))
        n_emp += int((col == ERR_BKG_SOURCE_EMPIRICAL).sum())
        n_fb += int((col == ERR_BKG_SOURCE_HOWELL_FALLBACK).sum())
        n_scaled += int((col == ERR_BKG_SOURCE_HOWELL_SCALED).sum())
        n_other += int(
            len(col)
            - (col == ERR_BKG_SOURCE_EMPIRICAL).sum()
            - (col == ERR_BKG_SOURCE_HOWELL_FALLBACK).sum()
            - (col == ERR_BKG_SOURCE_HOWELL_SCALED).sum()
        )
    pct_emp = 100.0 * n_emp / n_rows if n_rows else float("nan")
    pct_fb = 100.0 * n_fb / n_rows if n_rows else float("nan")
    pct_scaled = 100.0 * n_scaled / n_rows if n_rows else float("nan")
    return {
        "n_proc_files": len(files),
        "n_rows": n_rows,
        "pct_empirical": pct_emp,
        "pct_howell_fallback": pct_fb,
        "pct_howell_scaled": pct_scaled,
        "pct_other": 100.0 * n_other / n_rows if n_rows else float("nan"),
        "fallback_flag": bool(pct_fb > 20.0) if math.isfinite(pct_fb) else False,
        "mask_params": {
            "err_empty_apertures_n": "config default 64 clamp 16..256",
            "err_empty_apertures_min": "config default 16",
            "exclusion": "r_out + margin_px around each detected star; edge margin r_out+r_ap+1",
        },
    }


def _byte_identity_proc(
    proc_dir: Path,
    backup_dir: Path,
) -> dict[str, Any]:
    """Compare non-err science columns before vs after re-export."""
    mismatches: list[str] = []
    compared = 0
    for new_p in sorted(proc_dir.glob("proc_*.csv")):
        old_p = backup_dir / new_p.name
        if not old_p.is_file():
            continue
        try:
            old = pd.read_csv(old_p, low_memory=False)
            new = pd.read_csv(new_p, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            mismatches.append(f"{new_p.name}: read error {exc}")
            continue
        cols = [c for c in NON_ERR_PROC_COLS if c in old.columns and c in new.columns]
        if not cols:
            continue
        if len(old) != len(new):
            mismatches.append(f"{new_p.name}: row count {len(old)} -> {len(new)}")
            continue
        for c in cols:
            o = pd.to_numeric(old[c], errors="coerce")
            n = pd.to_numeric(new[c], errors="coerce")
            if not np.allclose(o.to_numpy(), n.to_numpy(), rtol=0, atol=1e-5, equal_nan=True):
                mismatches.append(f"{new_p.name}: column {c} diverged")
                break
        compared += 1
    return {
        "n_compared": compared,
        "byte_identical_non_err": len(mismatches) == 0,
        "mismatches": mismatches[:20],
    }


def _pedestal_fit_setup(
    lights_dir: Path,
    proc_dir: Path,
    *,
    gain: float,
    n_frames: int = 12,
    n_patches: int = 40,
    patch: int = 32,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Photon-transfer fit var = (level - P)/g with free intercept P [ADU]."""
    fits_files = sorted(lights_dir.glob("*.fits"))[: max(1, n_frames)]
    levels: list[float] = []
    vars_: list[float] = []
    header_p: list[float] = []
    for fp in fits_files:
        try:
            with fits.open(fp, memmap=True) as hd:
                data = np.asarray(hd[0].data, dtype=np.float64)
                hdr = hd[0].header
        except Exception:  # noqa: BLE001
            continue
        for key in ("PEDESTAL", "OFFSET"):
            try:
                v = float(hdr.get(key, float("nan")))
                if math.isfinite(v):
                    header_p.append(v)
            except (TypeError, ValueError):
                pass
        ny, nx = data.shape
        for _ in range(n_patches):
            y0 = int(rng.integers(0, max(1, ny - patch)))
            x0 = int(rng.integers(0, max(1, nx - patch)))
            patch_arr = data[y0 : y0 + patch, x0 : x0 + patch]
            patch_arr = patch_arr[np.isfinite(patch_arr)]
            if patch_arr.size < patch * patch // 2:
                continue
            med = float(np.median(patch_arr))
            var = float(np.var(patch_arr, ddof=1))
            if math.isfinite(med) and math.isfinite(var) and var > 0:
                levels.append(med)
                vars_.append(var)
    if len(levels) < 8:
        return {"n_points": len(levels), "P_adu": None, "P_ci_lo": None, "P_ci_hi": None, "header_pedestal": header_p}
    x = np.asarray(levels, dtype=np.float64)
    y = np.asarray(vars_, dtype=np.float64)
    # Linear fit y = (x - P)/g  => y = x/g - P/g  => slope=1/g, intercept=-P/g
    A = np.column_stack([x, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(A, y * gain, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    p_est = -intercept / slope if abs(slope) > 1e-12 else float("nan")
    resid = y * gain - (x - p_est)
    sig = float(np.std(resid, ddof=1)) if resid.size > 2 else float("nan")
    p_se = sig / math.sqrt(len(x)) if math.isfinite(sig) else float("nan")
    return {
        "n_points": int(len(x)),
        "P_adu": float(p_est) if math.isfinite(p_est) else None,
        "P_ci_lo": float(p_est - 1.96 * p_se) if math.isfinite(p_est) and math.isfinite(p_se) else None,
        "P_ci_hi": float(p_est + 1.96 * p_se) if math.isfinite(p_est) and math.isfinite(p_se) else None,
        "gain_used": float(gain),
        "header_pedestal_values": header_p[:5],
        "header_pedestal_median": float(np.median(header_p)) if header_p else None,
    }


def _gain_from_lights(lights_dir: Path, fallback: float) -> float:
    for fp in sorted(lights_dir.glob("*.fits"))[:5]:
        try:
            with fits.open(fp, memmap=False) as hd:
                for key in ("GAIN", "EGAIN", "VY_GAIN"):
                    try:
                        v = float(hd[0].header.get(key, float("nan")))
                        if math.isfinite(v) and v > 0:
                            return v
                    except (TypeError, ValueError):
                        continue
        except Exception:  # noqa: BLE001
            continue
    return float(fallback)


def _read_flux_err_ratio(
    proc_dir: Path,
    *,
    gain: float,
    read_noise: float,
    n_files: int = 8,
    n_stars: int = 12,
) -> dict[str, Any]:
    """Per-row photon err ratio empirical/howell at read_flux level (wide-rig gate)."""
    from photometry_core import ERR_BKG_MODE_EMPIRICAL, ERR_BKG_MODE_HOWELL, read_flux_from_csv

    ratios: list[float] = []
    for proc_path in sorted(proc_dir.glob("proc_*.csv"))[: max(1, n_files)]:
        try:
            df = pd.read_csv(proc_path, low_memory=False, dtype={"catalog_id": str})
        except Exception:  # noqa: BLE001
            continue
        if df.empty or "catalog_id" not in df.columns:
            continue
        ids = [str(x) for x in df["catalog_id"].dropna().astype(str).tolist()[: max(1, n_stars)]]
        if not ids:
            continue
        apertures = {}
        if "aperture_r_px" in df.columns:
            for cid in ids:
                sub = df.loc[df["catalog_id"].astype(str) == cid, "aperture_r_px"]
                if not sub.empty:
                    try:
                        apertures[cid] = float(pd.to_numeric(sub.iloc[0], errors="coerce"))
                    except (TypeError, ValueError):
                        pass
        emp = read_flux_from_csv(
            proc_path, ids, apertures, gain=gain, read_noise=read_noise, err_background_mode=ERR_BKG_MODE_EMPIRICAL
        )
        how = read_flux_from_csv(
            proc_path, ids, apertures, gain=gain, read_noise=read_noise, err_background_mode=ERR_BKG_MODE_HOWELL
        )
        if emp.empty or how.empty:
            continue
        m = emp.merge(how, on="catalog_id", suffixes=("_emp", "_how"))
        e0 = pd.to_numeric(m.get("err_how"), errors="coerce")
        e1 = pd.to_numeric(m.get("err_emp"), errors="coerce")
        ok = e0.notna() & e1.notna() & (e0 > 0)
        if ok.any():
            ratios.extend((e1[ok] / e0[ok]).tolist())
    if not ratios:
        return {"n": 0, "median": None, "p25": None, "p75": None}
    arr = np.asarray(ratios, dtype=np.float64)
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
    }


def _run_setup(
    *,
    draft_id: int,
    setup: str,
    archive_root: Path,
    work_root: Path,
    cfg: AppConfig,
    db: VyvarDatabase,
    skip_export: bool,
    patch_only: bool,
) -> dict[str, Any]:
    draft_dir = archive_root / "Drafts" / f"draft_{draft_id:06d}"
    ps_dir = draft_dir / "platesolve" / setup
    lights_dir = draft_dir / "detrended_aligned" / "lights" / setup
    phot_in = ps_dir / "photometry"
    out_phot = work_root / f"draft_{draft_id:06d}" / setup / "photometry"
    backup_dir = work_root / f"draft_{draft_id:06d}" / setup / "proc_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {"draft_id": draft_id, "setup": setup}
    if not lights_dir.is_dir() or not ps_dir.is_dir():
        result["error"] = f"missing lights or platesolve: {lights_dir} / {ps_dir}"
        return result

    # Backup proc CSVs once
    if not any(backup_dir.glob("proc_*.csv")):
        for p in sorted(lights_dir.glob("proc_*.csv")):
            shutil.copy2(p, backup_dir / p.name)

    t_export = t_patch = 0.0
    if patch_only:
        t0 = time.perf_counter()
        rng = np.random.default_rng(draft_id + hash(setup) % 10000)
        result["patch_stats"] = patch_setup(lights_dir=lights_dir, cfg=cfg, rng=rng)
        t_patch = time.perf_counter() - t0
    elif not skip_export:
        t0 = time.perf_counter()
        export_per_frame_catalogs(
            frames_root=lights_dir,
            platesolve_dir=ps_dir,
            masterstars_csv=ps_dir / "masterstars_full_match.csv",
            masterstar_fits=ps_dir / "MASTERSTAR.fits",
            write_sidecar_csv_next_to_fits=True,
            app_config=cfg,
            draft_id=draft_id,
        )
        t_export = time.perf_counter() - t0
    result["export_seconds"] = t_export
    result["patch_seconds"] = t_patch

    result["provenance"] = _provenance_stats(lights_dir)
    result["proc_byte_identity"] = _byte_identity_proc(lights_dir, backup_dir)

    ms_fits = ps_dir / "MASTERSTAR.fits"
    with fits.open(ms_fits, memmap=False) as hd:
        fwhm = _fwhm_from_header(hd[0].header)
    gain = _gain_from_lights(lights_dir, float(cfg.gain))
    read_noise = float(cfg.read_noise)

    out_phot.mkdir(parents=True, exist_ok=True)
    t2a_emp = t2a_howell = 0.0
    for mode, key in [("empirical", "phase2a_empirical"), ("howell", "phase2a_howell")]:
        cfg_mode = AppConfig()
        cfg_mode.archive_root = archive_root
        cfg_mode.err_background_mode = mode
        cfg_mode.save_lightcurve_png = False
        sub_out = out_phot / mode
        t0 = time.perf_counter()
        run_phase2a(
            masterstar_fits_path=ms_fits,
            active_targets_csv=phot_in / "active_targets.csv",
            comparison_stars_csv=phot_in / "comparison_stars_per_target.csv",
            per_frame_csv_dir=lights_dir,
            detrended_aligned_dir=lights_dir,
            output_dir=sub_out,
            fwhm_px=float(fwhm),
            cfg=cfg_mode,
            db=db,
            draft_id=draft_id,
        )
        elapsed = time.perf_counter() - t0
        if mode == "empirical":
            t2a_emp = elapsed
        else:
            t2a_howell = elapsed
        result[key] = {"seconds": elapsed, "output_dir": str(sub_out)}

    result["phase2a_empirical_seconds"] = t2a_emp
    result["phase2a_howell_seconds"] = t2a_howell

    if draft_id == 426:
        result["pedestal_fit"] = _pedestal_fit_setup(
            lights_dir, lights_dir, gain=gain, rng=np.random.default_rng(draft_id + hash(setup) % 10000)
        )

    result["read_flux_err_ratio_emp_over_howell"] = _read_flux_err_ratio(
        lights_dir, gain=gain, read_noise=read_noise
    )

    # err ratio wide-rig / all: sample active target LCs
    emp_lc = out_phot / "empirical" / "lightcurves"
    how_lc = out_phot / "howell" / "lightcurves"
    target_ids: list[str] = []
    at_p = phot_in / "active_targets.csv"
    if at_p.is_file():
        at = pd.read_csv(at_p, low_memory=False, dtype={"catalog_id": str})
        if "catalog_id" in at.columns:
            target_ids = [str(x) for x in at["catalog_id"].dropna().astype(str).tolist()[:12]]
    ratios: list[float] = []
    for cid in target_ids:
        p0 = how_lc / f"lightcurve_{cid}.csv"
        p1 = emp_lc / f"lightcurve_{cid}.csv"
        if not p0.is_file() or not p1.is_file():
            continue
        m = pd.read_csv(p0, usecols=["source_file", "err"], low_memory=False).merge(
            pd.read_csv(p1, usecols=["source_file", "err"], low_memory=False),
            on="source_file",
            suffixes=("_h", "_e"),
        )
        e0 = pd.to_numeric(m["err_h"], errors="coerce")
        e1 = pd.to_numeric(m["err_e"], errors="coerce")
        ok = e0.notna() & e1.notna() & (e0 > 0)
        if ok.any():
            ratios.extend((e1[ok] / e0[ok]).tolist())
    if ratios:
        arr = np.asarray(ratios, dtype=np.float64)
        result["err_ratio_emp_over_howell"] = {
            "n": int(arr.size),
            "median": float(np.median(arr)),
            "p25": float(np.quantile(arr, 0.25)),
            "p75": float(np.quantile(arr, 0.75)),
        }
    else:
        result["err_ratio_emp_over_howell"] = {"n": 0}

    # symlink canonical after path for validate script
    canonical = work_root / f"draft_{draft_id:06d}" / setup / "photometry" / "lightcurves"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    if canonical.is_symlink() or canonical.exists():
        if canonical.is_symlink():
            canonical.unlink()
        elif canonical.is_dir():
            shutil.rmtree(canonical)
    shutil.copytree(emp_lc, canonical, dirs_exist_ok=True)

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-root", type=str, default=None)
    ap.add_argument("--work-root", type=Path, default=Path("tmp/bingain_acceptance"))
    ap.add_argument("--drafts", type=int, nargs="*", default=None)
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument(
        "--patch-only",
        action="store_true",
        help="Patch sigma_bkg_ap onto existing proc CSVs (no DAO re-export). Implies --skip-export.",
    )
    ap.add_argument("--out", type=Path, default=Path("tmp/bingain_acceptance/run_report.json"))
    args = ap.parse_args()

    cfg = AppConfig()
    archive_root = resolve_archive_root(args.archive_root, cfg=cfg)
    cfg.archive_root = archive_root
    cfg.save_lightcurve_png = False

    cases = ACCEPTANCE_CASES
    if args.drafts:
        wanted = set(args.drafts)
        cases = [(d, s) for d, s in ACCEPTANCE_CASES if d in wanted]

    work_root = args.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    db = VyvarDatabase(cfg.database_path)

    report: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "archive_root": str(archive_root),
        "work_root": str(work_root),
        "setups": {},
    }
    try:
        for draft_id, setup in cases:
            key = f"draft_{draft_id:06d}/{setup}"
            print(f"[ACCEPT] {key} ...", flush=True)
            report["setups"][key] = _run_setup(
                draft_id=draft_id,
                setup=setup,
                archive_root=archive_root,
                work_root=work_root,
                cfg=cfg,
                db=db,
                skip_export=bool(args.skip_export or args.patch_only),
                patch_only=bool(args.patch_only),
            )
    finally:
        try:
            db.conn.close()
        except Exception:  # noqa: BLE001
            pass

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
