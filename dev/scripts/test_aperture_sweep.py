#!/usr/bin/env python3
"""Sweep aperture radius on proc FITS; compare differential LC RMS to AIJ.

Temporary diagnostic - do not commit unless promoted.

Usage (from repo root):
  python scripts/test_aperture_sweep.py
  python scripts/test_aperture_sweep.py --target FW --radii 3.318,5,7,10
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from validate_lc_crossval import (  # noqa: E402
    MIN_COMP_PER_FRAME,
    MIN_FRAMES,
    _norm_cid,
    differential_lc_rms,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

DEFAULT_DRAFT = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000311")
DEFAULT_SETUP = "NoFilter_60_2"

TARGETS = {
    "FW": ("1497343732462852864", "FW_CVn_aij.tbl"),
    "BO": ("1498613634033133184", "BO_CVn_aij.tbl"),
}

GAIN = 3.17


def _norm_cid_local(x: object) -> str:
    return _norm_cid(x)


def load_comp_cids(comp_csv: Path, target_cid: str) -> list[str]:
    comp = pd.read_csv(comp_csv, dtype={"catalog_id": str, "target_catalog_id": str})
    sub = comp[comp["target_catalog_id"].map(_norm_cid_local) == _norm_cid_local(target_cid)]
    return [_norm_cid_local(x) for x in sub["catalog_id"].tolist() if str(x).strip()]


def aij_rms_ppt(aij_tbl: Path) -> tuple[float, float]:
    """RMS of AIJ rel_flux_T1 (demeaned), in mag and ppt."""
    aij = pd.read_csv(aij_tbl, sep="\t")
    f = pd.to_numeric(aij["rel_flux_T1"], errors="coerce").to_numpy(dtype=float)
    f = f[np.isfinite(f) & (f > 0)]
    mag = -2.5 * np.log10(f)
    mag_dm = mag - np.median(mag)
    rms_mag = float(np.std(mag_dm))
    return rms_mag, rms_mag * 1000.0


def annulus_radii(r_ap: float, fwhm_px: float, mode: str) -> tuple[float, float]:
    if mode == "vyvar":
        return 4.75 * fwhm_px, 9.0 * fwhm_px
    # scaled: inner ~1.4x ap, width ~0.9x ap (typical AIJ-like proportions)
    r_in = max(r_ap * 1.4, r_ap + 2.0)
    r_out = r_in + max(r_ap * 0.9, 4.0)
    return r_in, r_out


def photometry_frame(
    data: np.ndarray,
    sky: float,
    positions: list[tuple[float, float]],
    r_ap: float,
    r_in: float,
    r_out: float,
) -> np.ndarray:
    flux = np.full(len(positions), np.nan, dtype=float)
    if not positions:
        return flux
    data2d = np.ascontiguousarray(np.squeeze(data), dtype=np.float64)
    for i, (x, y) in enumerate(positions):
        ap = CircularAperture((x, y), r=r_ap)
        ann = CircularAnnulus((x, y), r_in=r_in, r_out=r_out)
        try:
            src = ApertureStats(data2d, ap)
            sky = ApertureStats(data2d, ann)
            net = float(src.sum) - float(sky.median) * float(ap.area)
            if np.isfinite(net) and net > 0:
                flux[i] = net
        except Exception:  # noqa: BLE001
            continue
    return flux


def load_xy_positions(
    active_csv: Path,
    comp_csv: Path,
    target_cid: str,
    comp_cids: list[str],
) -> dict[str, tuple[float, float]]:
    """Chip x,y from active_targets (target) and comparison_stars_per_target (comps)."""
    out: dict[str, tuple[float, float]] = {}
    tcid = _norm_cid_local(target_cid)

    if active_csv.is_file():
        at = pd.read_csv(active_csv, dtype={"catalog_id": str})
        at["catalog_id"] = at["catalog_id"].map(_norm_cid_local)
        row = at[at["catalog_id"] == tcid]
        if not row.empty:
            x = float(pd.to_numeric(row["x"].iloc[0], errors="coerce"))
            y = float(pd.to_numeric(row["y"].iloc[0], errors="coerce"))
            if np.isfinite(x) and np.isfinite(y):
                out[tcid] = (x, y)

    if comp_csv.is_file():
        comp = pd.read_csv(comp_csv, dtype={"catalog_id": str, "target_catalog_id": str})
        comp["catalog_id"] = comp["catalog_id"].map(_norm_cid_local)
        comp["target_catalog_id"] = comp["target_catalog_id"].map(_norm_cid_local)
        sub = comp[comp["target_catalog_id"] == tcid]
        for cid in comp_cids:
            row = sub[sub["catalog_id"] == _norm_cid_local(cid)]
            if row.empty:
                continue
            x = float(pd.to_numeric(row["x"].iloc[0], errors="coerce"))
            y = float(pd.to_numeric(row["y"].iloc[0], errors="coerce"))
            if np.isfinite(x) and np.isfinite(y):
                out[_norm_cid_local(cid)] = (x, y)

    return out


def build_flux_matrix(
    star_cids: list[str],
    xy_map: dict[str, tuple[float, float]],
    fits_files: list[Path],
    r_ap: float,
    fwhm_px: float,
    ann_mode: str,
) -> dict[str, np.ndarray]:
    n = len(fits_files)
    cids_order = [c for c in star_cids if c in xy_map]
    positions = [xy_map[c] for c in cids_order]
    matrix = {cid: np.full(n, np.nan, dtype=float) for cid in cids_order}
    if not cids_order:
        return matrix
    r_in, r_out = annulus_radii(r_ap, fwhm_px, ann_mode)

    for i, fits_path in enumerate(fits_files):
        with fits.open(fits_path, memmap=True) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
        fluxes = photometry_frame(data, 0.0, positions, r_ap, r_in, r_out)
        for cid, fl in zip(cids_order, fluxes, strict=False):
            matrix[cid][i] = fl

        if (i + 1) % 40 == 0:
            LOGGER.info("  frame %d/%d r_ap=%.2f", i + 1, n, r_ap)

    return matrix


def paired_paths(proc_dir: Path) -> tuple[list[Path], list[Path]]:
    csv_files = sorted(proc_dir.glob("proc_*.csv"))
    fits_files: list[Path] = []
    fits_dir = proc_dir.parent.parent.parent / "processed" / "lights" / proc_dir.name
    if not fits_dir.is_dir():
        fits_dir = proc_dir
    for csv_path in csv_files:
        stem = csv_path.stem
        fit = fits_dir / f"{stem}.fits"
        if not fit.is_file():
            fit = proc_dir / f"{stem}.fits"
        if not fit.is_file():
            raise FileNotFoundError(f"Missing FITS for {csv_path.name}")
        fits_files.append(fit)
    return csv_files, fits_files


def main() -> int:
    ap = argparse.ArgumentParser(description="Aperture sweep vs AIJ LC RMS")
    ap.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    ap.add_argument("--setup", default=DEFAULT_SETUP)
    ap.add_argument("--target", choices=("FW", "BO", "both"), default="FW")
    ap.add_argument("--radii", default="3.318,5,7,10", help="Comma-separated aperture radii [px]")
    ap.add_argument(
        "--annulus",
        choices=("vyvar", "scaled"),
        default="scaled",
        help="vyvar = 4.75/9.0xFWHM sky annulus; scaled = inner~1.4xr_ap",
    )
    ap.add_argument("--fwhm", type=float, default=None, help="FWHM [px]; default from MASTERSTAR")
    args = ap.parse_args()

    draft = args.draft.resolve()
    proc_dir = draft / "detrended_aligned" / "lights" / args.setup
    phot_dir = draft / "platesolve" / args.setup / "photometry"
    lights_aij = draft / "detrended_aligned" / "lights"
    comp_csv = phot_dir / "comparison_stars_per_target.csv"

    if args.fwhm is not None and np.isfinite(args.fwhm):
        fwhm_px = float(args.fwhm)
    else:
        from ui_aperture_photometry import _load_fwhm  # noqa: PLC0415

        fwhm_px = float(_load_fwhm(draft / "platesolve" / args.setup / "MASTERSTAR.fits"))

    radii = [float(x.strip()) for x in str(args.radii).split(",") if x.strip()]
    csv_files, fits_files = paired_paths(proc_dir)
    active_csv = phot_dir / "active_targets.csv"
    LOGGER.info(
        "Frames: %d  FWHM=%.3f px  annulus=%s  xy=active_targets",
        len(fits_files),
        fwhm_px,
        args.annulus,
    )

    targets = list(TARGETS.items()) if args.target == "both" else [(args.target, TARGETS[args.target.upper()])]

    print("=" * 70)
    print(f"Draft {draft.name}  setup={args.setup}")
    print(f"VYVAR default aperture ~3.3 px (SNR table); AIJ Source_Radius ~7 px")
    print("=" * 70)

    for label, (cid, aij_name) in targets:
        comp_cids = load_comp_cids(comp_csv, cid)
        star_cids = [_norm_cid_local(cid)] + comp_cids
        aij_tbl = lights_aij / aij_name
        aij_mag, aij_ppt = aij_rms_ppt(aij_tbl)

        vy = pd.read_csv(phot_dir / "photometry_summary.csv", dtype={"catalog_id": str})
        row = vy[vy["catalog_id"].map(_norm_cid_local) == _norm_cid_local(cid)]
        vy_rms = float(row["lc_rms"].iloc[0]) if not row.empty else float("nan")
        vy_ap = float(row["aperture_px"].iloc[0]) if not row.empty and "aperture_px" in row else float("nan")

        print(f"\n--- {label} CVn ({cid}) ---")
        print(f"  comps: {len(comp_cids)}  AIJ rel_flux RMS: {aij_ppt:.2f} ppt ({aij_mag:.4f} mag)")
        print(f"  VYVAR summary: lc_rms={vy_rms:.4f}  aperture_px={vy_ap:.3f}")
        print(f"  {'r_ap':>6}  {'RMS_mag':>8}  {'ppt':>8}  {'n_frame':>7}  vs AIJ")
        print("  " + "-" * 44)

        xy_map = load_xy_positions(active_csv, comp_csv, cid, comp_cids)
        print(f"  positions loaded: {len(xy_map)}/{len(star_cids)} stars")
        comp_cids = [c for c in comp_cids if c in xy_map]
        if len(comp_cids) < MIN_COMP_PER_FRAME:
            print("  ERROR: too few comp positions in active_targets")
            continue

        for r_ap in radii:
            matrix = build_flux_matrix(
                star_cids, xy_map, fits_files, r_ap, fwhm_px, args.annulus
            )
            rms, n_used = differential_lc_rms(cid, comp_cids, matrix, len(fits_files))
            ppt = rms * 1000.0 if np.isfinite(rms) else float("nan")
            tag = ""
            if abs(r_ap - 7.0) < 0.01:
                tag = " <- AIJ Source_Radius"
            if abs(r_ap - vy_ap) < 0.05:
                tag += " <- VYVAR"
            delta_ppt = ppt - aij_ppt if np.isfinite(ppt) else float("nan")
            print(f"  {r_ap:6.2f}  {rms:8.4f}  {ppt:8.2f}  {n_used:7d}  d={delta_ppt:+.1f} ppt{tag}")

    print("\nDone. If RMS drops near 7 px toward AIJ (~13 ppt for FW), aperture mismatch explains the UI look.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
