#!/usr/bin/env python3
"""Patch sigma_bkg_ap / err_bkg_source onto existing proc CSVs (no DAO re-export)."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

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
from photometry_core import (  # noqa: E402
    ERR_BKG_SOURCE_COL,
    ERR_BKG_SOURCE_EMPIRICAL,
    ERR_BKG_SOURCE_HOWELL_FALLBACK,
    ERR_BKG_SOURCE_HOWELL_SCALED,
    SIGMA_BKG_AP_COL,
    finalize_hybrid_bkg_fallback_proc_dir,
    measure_empty_aperture_sigma_bkg,
)
from scripts.bingain_fix_validate import resolve_archive_root  # noqa: E402


def _annulus_radii(r_ap: float, fw: float, inner_fwhm: float, outer_fwhm: float) -> tuple[float, float]:
    r_in = max(float(r_ap) + 0.5, float(inner_fwhm) * fw)
    r_out = max(r_in + 0.5, float(outer_fwhm) * fw)
    return r_in, r_out


def patch_setup(
    *,
    lights_dir: Path,
    cfg: AppConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    inner = float(cfg.annulus_inner_fwhm)
    outer = float(cfg.annulus_outer_fwhm)
    n_empty = int(cfg.err_empty_apertures_n)
    n_min = int(cfg.err_empty_apertures_min)
    stats = {"n_files": 0, "pct_empirical": 0.0, "pct_fallback": 0.0}
    n_emp = n_fb = n_rows = 0

    for proc_path in sorted(lights_dir.glob("proc_*.csv")):
        fits_path = proc_path.with_suffix(".fits")
        if not fits_path.is_file():
            stem = proc_path.stem
            if stem.startswith("proc_"):
                fits_path = lights_dir / f"{stem[5:]}.fits"
        if not fits_path.is_file():
            continue
        df = pd.read_csv(proc_path, low_memory=False)
        if df.empty:
            continue
        with fits.open(fits_path, memmap=True) as hd:
            data = np.asarray(hd[0].data, dtype=np.float64)
            hdr = hd[0].header
        fw = 4.5
        for key in ("VY_FWHM_GAUSS", "VY_FWHM"):
            try:
                v = float(hdr.get(key, float("nan")))
                if math.isfinite(v) and v > 0:
                    fw = v
                    break
            except (TypeError, ValueError):
                pass
        xs = pd.to_numeric(df.get("x"), errors="coerce").to_numpy(dtype=np.float64)
        ys = pd.to_numeric(df.get("y"), errors="coerce").to_numpy(dtype=np.float64)
        if "aperture_r_px" in df.columns:
            r_ap_arr = pd.to_numeric(df["aperture_r_px"], errors="coerce").to_numpy(dtype=np.float64)
        else:
            r_ap_arr = np.full(len(df), max(0.5, float(cfg.aperture_fwhm_factor) * fw))
        sigma_col = np.full(len(df), np.nan, dtype=np.float64)
        src_col = np.full(len(df), ERR_BKG_SOURCE_HOWELL_FALLBACK, dtype=object)
        unique_r = np.unique(np.round(r_ap_arr[np.isfinite(r_ap_arr) & (r_ap_arr > 0)], 4))
        cache: dict[float, tuple[float, str]] = {}
        for r_u in unique_r:
            ri, ro = _annulus_radii(float(r_u), fw, inner, outer)
            sig, nv, _reason = measure_empty_aperture_sigma_bkg(
                data, xs, ys, float(r_u), ri, ro, n_apertures=n_empty, min_valid=n_min, rng=rng
            )
            if math.isfinite(sig) and sig >= 0:
                cache[float(r_u)] = (float(sig), ERR_BKG_SOURCE_EMPIRICAL)
            else:
                cache[float(r_u)] = (float("nan"), ERR_BKG_SOURCE_HOWELL_FALLBACK)
        for i, r_ap in enumerate(r_ap_arr):
            key = round(float(r_ap), 4) if math.isfinite(r_ap) else float("nan")
            sig_v, src_v = cache.get(key, (float("nan"), ERR_BKG_SOURCE_HOWELL_FALLBACK))
            sigma_col[i] = sig_v
            src_col[i] = src_v
        df[SIGMA_BKG_AP_COL] = sigma_col
        df[ERR_BKG_SOURCE_COL] = src_col
        df.to_csv(proc_path, index=False)
        stats["n_files"] += 1
        n_rows += len(df)
        n_emp += int((src_col == ERR_BKG_SOURCE_EMPIRICAL).sum())
        n_fb += int((src_col == ERR_BKG_SOURCE_HOWELL_FALLBACK).sum())

    if n_rows:
        stats["pct_empirical"] = 100.0 * n_emp / n_rows
        stats["pct_fallback"] = 100.0 * n_fb / n_rows
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-root", type=str, default=None)
    ap.add_argument("--draft", type=int, required=True)
    ap.add_argument("--setup", type=str, required=True)
    ap.add_argument("--restore-backup", type=Path, default=None, help="Restore proc CSV from backup dir first")
    args = ap.parse_args()
    cfg = AppConfig()
    root = resolve_archive_root(args.archive_root, cfg=cfg)
    lights = root / "Drafts" / f"draft_{args.draft:06d}" / "detrended_aligned" / "lights" / args.setup
    if args.restore_backup is not None:
        for p in Path(args.restore_backup).glob("proc_*.csv"):
            (lights / p.name).write_bytes(p.read_bytes())
    rng = np.random.default_rng(args.draft + hash(args.setup) % 10000)
    stats = patch_setup(lights_dir=lights, cfg=cfg, rng=rng)
    stats["hybrid_finalize"] = finalize_hybrid_bkg_fallback_proc_dir(
        lights,
        gain=float(cfg.gain),
        read_noise=float(cfg.read_noise),
        setup_label=f"draft_{args.draft:06d}/{args.setup}",
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
