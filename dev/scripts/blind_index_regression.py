#!/usr/bin/env python3
"""Regression: blind solve with legacy vs density-matched mag14 index."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig
from vyvar_blind_solver import find_blind_candidates, find_blind_hint
from vyvar_platesolver import _verify_blind_candidates


def _field_center(masterstar: Path) -> tuple[float, float, float, int, int]:
    with fits.open(masterstar, memmap=True) as hdul:
        hdr = hdul[0].header
        ny, nx = hdul[0].data.shape
    w = WCS(hdr)
    ra, dec = w.all_pix2world(nx / 2.0, ny / 2.0, 0)
    vy_plts = hdr.get("VY_PLTS")
    if vy_plts is not None:
        try:
            scale = float(vy_plts)
        except (TypeError, ValueError):
            scale = float(abs(w.pixel_scale_matrix).mean() * 3600.0)
    else:
        scale = float(abs(w.pixel_scale_matrix).mean() * 3600.0)
    return float(ra), float(dec), scale, int(nx), int(ny)


def _dao(fits_path: Path) -> tuple:
    with fits.open(fits_path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
    _, med, std = sigma_clipped_stats(data, sigma=3.0)
    finder = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)
    srcs = finder(data - med)
    if srcs is None or len(srcs) < 3:
        return None, data.shape
    import pandas as pd

    df = srcs.to_pandas().rename(columns={"xcentroid": "x", "ycentroid": "y"})
    if "peak" in df.columns:
        df["flux"] = df["peak"]
    return df.sort_values("flux", ascending=False), data.shape


def _sep(ra1, dec1, ra2, dec2) -> float:
    dra = (ra1 - ra2) * math.cos(math.radians((dec1 + dec2) / 2.0))
    ddec = dec1 - dec2
    return math.sqrt(dra * dra + ddec * ddec)


def _blind_hint(
    *,
    dao_df,
    index_path: Path,
    plate_scale: float,
    fov_deg: float,
    nx: int,
    ny: int,
    cfg: AppConfig,
    verify: bool,
) -> tuple[float, float] | None:
    if verify:
        cands = find_blind_candidates(
            dao_df,
            index_path,
            n_top=30,
            top_n=int(cfg.blind_verify_top_n),
            plate_scale_arcsec_per_px=plate_scale,
            fov_deg=fov_deg,
            app_config=cfg,
        )
        gaia_db = Path(cfg.gaia_db_path).expanduser()
        if gaia_db.is_file() and cands:
            hint = _verify_blind_candidates(
                cands,
                dao_df=dao_df,
                gaia_db_path=gaia_db,
                fov_deg=fov_deg,
                naxis1=nx,
                naxis2=ny,
                pixel_pitch_um=None,
                focal_length_mm=None,
                max_cat_mag=16.0,
                app_config=cfg,
            )
            if hint is not None:
                return hint
        return find_blind_hint(
            dao_df,
            index_path,
            n_top=30,
            min_votes=3,
            plate_scale_arcsec_per_px=plate_scale,
            fov_deg=fov_deg,
            app_config=cfg,
        )
    return find_blind_hint(
        dao_df,
        index_path,
        n_top=30,
        min_votes=3,
        plate_scale_arcsec_per_px=plate_scale,
        fov_deg=fov_deg,
        app_config=cfg,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masterstar", type=Path, required=True)
    ap.add_argument("--legacy-index", type=Path, default=_ROOT / "GAIA_DR3" / "gaia_triangles.pkl")
    ap.add_argument("--new-index", type=Path, default=_ROOT / "GAIA_DR3" / "gaia_triangles_mag14.pkl")
    ap.add_argument("--max-sep-deg", type=float, default=2.0)
    ap.add_argument("--truth-ra", type=float, default=None)
    ap.add_argument("--truth-dec", type=float, default=None)
    ap.add_argument(
        "--plate-scale",
        type=float,
        default=None,
        help="arcsec/px (default: WCS mean; use 1.3 for draft_380 Chi_and_H)",
    )
    args = ap.parse_args()

    cfg = AppConfig()
    ra_wcs, dec_wcs, ps_wcs, nx, ny = _field_center(args.masterstar)
    ps = float(args.plate_scale) if args.plate_scale is not None else ps_wcs
    ra_t = float(args.truth_ra) if args.truth_ra is not None else ra_wcs
    dec_t = float(args.truth_dec) if args.truth_dec is not None else dec_wcs
    fov = max(nx, ny) * ps / 3600.0
    dao_df, shape = _dao(args.masterstar)
    if dao_df is None:
        print("FAIL: too few DAO stars")
        return 1

    rows = []
    for label, idx, verify in (
        ("legacy_premdensity", args.legacy_index, True),
        ("mag14_uniform", args.new_index, True),
        ("mag14_vote_only", args.new_index, False),
    ):
        cfg_v = AppConfig()
        cfg_v.blind_verify_enabled = verify
        if not verify:
            cfg_v = cfg_v  # find_blind_hint path
        hint = _blind_hint(
            dao_df=dao_df,
            index_path=idx,
            plate_scale=ps,
            fov_deg=fov,
            nx=nx,
            ny=ny,
            cfg=cfg_v if verify else cfg_v,
            verify=verify,
        )
        sep = _sep(hint[0], hint[1], ra_t, dec_t) if hint else float("nan")
        ok = hint is not None and sep <= args.max_sep_deg
        rows.append((label, hint, sep, ok))
        print(f"{label}: hint={hint} sep={sep:.3f}° {'OK' if ok else 'FAIL'}")

    legacy_ok = rows[0][3]
    new_ok = rows[1][3]
    parity_ok = rows[1][2] <= args.max_sep_deg + 0.5 if rows[1][1] and rows[2][1] else False
    print(
        f"truth: RA={ra_t:.4f} Dec={dec_t:.4f} scale={ps:.3f} arcsec/px "
        f"(WCS center RA={ra_wcs:.4f} Dec={dec_wcs:.4f})"
    )
    if not new_ok:
        return 1
    if not legacy_ok:
        print("WARN: legacy index MISS on this field (wide-rig may never have blind-HIT)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
