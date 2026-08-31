#!/usr/bin/env python3
"""Build differential LC from proc FITS (photutils) at fixed aperture - AIJ replication test.

Writes lightcurve_{catalog_id}_ap{N}.csv and comparison PNGs vs AIJ .tbl.

Usage:
  python scripts/build_lc_from_fits_aperture.py
  python scripts/build_lc_from_fits_aperture.py --radius 7 --targets FW,BO

Temporary diagnostic - do not commit unless promoted.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from validate_lc_crossval import MIN_COMP_PER_FRAME, MIN_FRAMES, _norm_cid_int_dotzero, differential_lc_rms

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

DEFAULT_DRAFT = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000311")
DEFAULT_SETUP = "NoFilter_60_2"

STARS: dict[str, tuple[str, str, str]] = {
    "FW": ("FW CVn", "1497343732462852864", "FW_CVn_aij.tbl"),
    "BO": ("BO CVn", "1498613634033133184", "BO_CVn_aij.tbl"),
}


def _norm(x: object) -> str:
    return _norm_cid_int_dotzero(x)


def annulus_radii(r_ap: float, mode: str, fwhm_px: float) -> tuple[float, float]:
    if mode == "vyvar":
        return 4.75 * fwhm_px, 9.0 * fwhm_px
    r_in = max(r_ap * 1.4, r_ap + 2.0)
    r_out = r_in + max(r_ap * 0.9, 4.0)
    return r_in, r_out


def net_flux(data: np.ndarray, x: float, y: float, r_ap: float, r_in: float, r_out: float) -> float:
    ap = CircularAperture((x, y), r=r_ap)
    ann = CircularAnnulus((x, y), r_in=r_in, r_out=r_out)
    src = ApertureStats(data, ap)
    sky = ApertureStats(data, ann)
    net = float(src.sum) - float(sky.median) * float(ap.area)
    if np.isfinite(net) and net > 0:
        return net
    return float("nan")


def load_xy(active_csv: Path, comp_csv: Path, target_cid: str, comp_cids: list[str]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    tcid = _norm(target_cid)
    at = pd.read_csv(active_csv, dtype={"catalog_id": str})
    at["catalog_id"] = at["catalog_id"].map(_norm)
    row = at[at["catalog_id"] == tcid]
    if not row.empty:
        x = float(row["x"].iloc[0])
        y = float(row["y"].iloc[0])
        if np.isfinite(x) and np.isfinite(y):
            out[tcid] = (x, y)
    comp = pd.read_csv(comp_csv, dtype={"catalog_id": str, "target_catalog_id": str})
    comp["catalog_id"] = comp["catalog_id"].map(_norm)
    comp["target_catalog_id"] = comp["target_catalog_id"].map(_norm)
    sub = comp[comp["target_catalog_id"] == tcid]
    for cid in comp_cids:
        r = sub[sub["catalog_id"] == _norm(cid)]
        if r.empty:
            continue
        x = float(r["x"].iloc[0])
        y = float(r["y"].iloc[0])
        if np.isfinite(x) and np.isfinite(y):
            out[_norm(cid)] = (x, y)
    return out


def paired_paths(proc_dir: Path, fits_source: str) -> tuple[list[Path], list[Path]]:
    csv_files = sorted(proc_dir.glob("proc_*.csv"))
    if fits_source == "aligned":
        fits_dir = proc_dir
    else:
        fits_dir = proc_dir.parent.parent.parent / "processed" / "lights" / proc_dir.name
    fits_files: list[Path] = []
    for csv_path in csv_files:
        fit = fits_dir / f"{csv_path.stem}.fits"
        if not fit.is_file():
            raise FileNotFoundError(f"Missing FITS for {csv_path.name} in {fits_dir}")
        fits_files.append(fit)
    return csv_files, fits_files


def load_aij_xy(aij_path: Path, target_label: str = "T1") -> dict[str, list[tuple[float, float]]]:
    """Per-frame FITS centroids from AIJ table (X(FITS)_*, Y(FITS)_*)."""
    aij = pd.read_csv(aij_path, sep="\t")
    out: dict[str, list[tuple[float, float]]] = {target_label: []}
    comp_keys = [f"C{i}" for i in range(2, 10)]
    for ck in comp_keys:
        out[ck] = []
    for _, row in aij.iterrows():
        x = float(pd.to_numeric(row.get(f"X(FITS)_{target_label}"), errors="coerce"))
        y = float(pd.to_numeric(row.get(f"Y(FITS)_{target_label}"), errors="coerce"))
        out[target_label].append((x, y) if np.isfinite(x) and np.isfinite(y) else (float("nan"), float("nan")))
        for ck in comp_keys:
            x = float(pd.to_numeric(row.get(f"X(FITS)_{ck}"), errors="coerce"))
            y = float(pd.to_numeric(row.get(f"Y(FITS)_{ck}"), errors="coerce"))
            out[ck].append((x, y) if np.isfinite(x) and np.isfinite(y) else (float("nan"), float("nan")))
    return out


def build_lc_aij_coords(
    aij_path: Path,
    csv_files: list[Path],
    fits_files: list[Path],
    r_ap: float,
    r_in: float,
    r_out: float,
) -> pd.DataFrame:
    """Photometry at AIJ X(FITS)/Y(FITS); differential vs AIJ comp apertures."""
    aij = pd.read_csv(aij_path, sep="\t")
    xy = load_aij_xy(aij_path, "T1")
    comp_keys = [f"C{i}" for i in range(2, 10)]
    rows: list[dict[str, object]] = []

    for i, (csv_path, fit_path) in enumerate(zip(csv_files, fits_files, strict=True)):
        meta = load_frame_meta(csv_path)
        with fits.open(fit_path, memmap=True) as hdul:
            data = np.ascontiguousarray(np.squeeze(hdul[0].data), dtype=np.float64)

        xt, yt = xy["T1"][i]
        if not (np.isfinite(xt) and np.isfinite(yt)):
            continue
        ft = net_flux(data, xt, yt, r_ap, r_in, r_out)
        comp_fluxes: list[float] = []
        for ck in comp_keys:
            if i >= len(xy[ck]):
                break
            xc, yc = xy[ck][i]
            if not (np.isfinite(xc) and np.isfinite(yc)):
                continue
            fc = net_flux(data, xc, yc, r_ap, r_in, r_out)
            if np.isfinite(fc) and fc > 0:
                comp_fluxes.append(fc)

        if not (np.isfinite(ft) and ft > 0) or len(comp_fluxes) < MIN_COMP_PER_FRAME:
            continue

        fc_mean = float(np.mean(comp_fluxes))
        mag_inst = -2.5 * np.log10(ft)
        mag_diff = mag_inst - float(np.mean([-2.5 * np.log10(f) for f in comp_fluxes]))
        rel_flux = ft / fc_mean

        aij_row = aij.iloc[i] if i < len(aij) else None
        rel_aij = float("nan")
        if aij_row is not None:
            rel_aij = float(pd.to_numeric(aij_row.get("rel_flux_T1"), errors="coerce"))

        rows.append(
            {
                **meta,
                "flux_target": ft,
                "flux_comp_mean": fc_mean,
                "n_comp_used": len(comp_fluxes),
                "mag_inst": mag_inst,
                "mag_diff": mag_diff,
                "rel_flux": rel_flux,
                "rel_flux_aij": rel_aij,
                "mag_calib": mag_diff,
                "aperture_r_px": r_ap,
                "flag": "normal",
                "method": f"photutils_aij_xy_r{int(r_ap) if r_ap == int(r_ap) else r_ap}",
            }
        )

    return pd.DataFrame(rows).sort_values("bjd").reset_index(drop=True)


def load_frame_meta(csv_path: Path) -> dict[str, float | str]:
    df = pd.read_csv(
        csv_path,
        usecols=lambda c: c in {"bjd_tdb_mid", "hjd_mid", "jd_mid", "airmass", "source_file"},
    )
    row = df.iloc[0]
    return {
        "bjd": float(pd.to_numeric(row.get("bjd_tdb_mid"), errors="coerce")),
        "hjd": float(pd.to_numeric(row.get("hjd_mid"), errors="coerce")),
        "jd": float(pd.to_numeric(row.get("jd_mid"), errors="coerce")),
        "airmass": float(pd.to_numeric(row.get("airmass"), errors="coerce")),
        "source_file": str(row.get("source_file", csv_path.name)),
    }


def build_lc(
    target_cid: str,
    comp_cids: list[str],
    xy: dict[str, tuple[float, float]],
    csv_files: list[Path],
    fits_files: list[Path],
    r_ap: float,
    r_in: float,
    r_out: float,
) -> pd.DataFrame:
    tcid = _norm(target_cid)
    cids = [tcid] + [_norm(c) for c in comp_cids if _norm(c) in xy]
    rows: list[dict[str, object]] = []

    for csv_path, fit_path in zip(csv_files, fits_files, strict=True):
        meta = load_frame_meta(csv_path)
        with fits.open(fit_path, memmap=True) as hdul:
            data = np.ascontiguousarray(np.squeeze(hdul[0].data), dtype=np.float64)

        fluxes: dict[str, float] = {}
        for cid in cids:
            if cid not in xy:
                continue
            x, y = xy[cid]
            fluxes[cid] = net_flux(data, x, y, r_ap, r_in, r_out)

        ft = fluxes.get(tcid, float("nan"))
        comp_fluxes = [fluxes[c] for c in cids if c != tcid and np.isfinite(fluxes.get(c, float("nan")))]
        if not (np.isfinite(ft) and ft > 0) or len(comp_fluxes) < MIN_COMP_PER_FRAME:
            continue

        fc_mean = float(np.mean(comp_fluxes))
        mag_inst = -2.5 * np.log10(ft)
        comp_mag_mean = float(np.mean([-2.5 * np.log10(f) for f in comp_fluxes]))
        mag_diff = mag_inst - comp_mag_mean
        rel_flux = ft / fc_mean if fc_mean > 0 else float("nan")

        rows.append(
            {
                **meta,
                "flux_target": ft,
                "flux_comp_mean": fc_mean,
                "n_comp_used": len(comp_fluxes),
                "mag_inst": mag_inst,
                "mag_diff": mag_diff,
                "rel_flux": rel_flux,
                "mag_calib": mag_diff,
                "aperture_r_px": r_ap,
                "flag": "normal",
                "method": f"photutils_r{int(r_ap) if r_ap == int(r_ap) else r_ap}",
            }
        )

    return pd.DataFrame(rows).sort_values("bjd").reset_index(drop=True)


def compare_aij(lc: pd.DataFrame, aij_path: Path) -> dict[str, float]:
    aij = pd.read_csv(aij_path, sep="\t").sort_values("BJD_TDB").reset_index(drop=True)
    n = min(len(lc), len(aij))
    if n < 5:
        return {}
    rf_aij = pd.to_numeric(aij["rel_flux_T1"].iloc[:n], errors="coerce").to_numpy()
    rf_vy = lc["rel_flux"].iloc[:n].to_numpy(dtype=float)
    mag_aij = -2.5 * np.log10(np.clip(rf_aij, 1e-12, None))
    mag_vy = lc["mag_diff"].iloc[:n].to_numpy(dtype=float)
    m_aij = mag_aij - np.nanmedian(mag_aij)
    m_vy = mag_vy - np.nanmedian(mag_vy)
    ok = np.isfinite(rf_aij) & np.isfinite(rf_vy) & (rf_aij > 0) & (rf_vy > 0)
    out = {
        "n_frames": float(n),
        "corr_rel_flux": float(np.corrcoef(rf_vy[ok], rf_aij[ok])[0, 1]) if ok.sum() > 3 else float("nan"),
        "corr_mag_demean": float(np.corrcoef(m_vy[ok], m_aij[ok])[0, 1]) if ok.sum() > 3 else float("nan"),
        "rms_rel_flux_vy_ppt": float(np.std(rf_vy[ok] / np.nanmedian(rf_vy[ok]))) * 1000.0,
        "rms_rel_flux_aij_ppt": float(np.std(rf_aij[ok] / np.nanmedian(rf_aij[ok]))) * 1000.0,
        "rms_mag_diff_vy_ppt": float(np.std(m_vy[ok])) * 1000.0,
        "rms_mag_diff_aij_ppt": float(np.std(m_aij[ok])) * 1000.0,
        "mean_ratio_vy_over_aij": float(np.nanmean(rf_vy[ok] / rf_aij[ok])),
    }
    return out


def plot_replication(
    name: str,
    cid: str,
    lc: pd.DataFrame,
    aij_path: Path,
    vyvar_lc_path: Path | None,
    stats: dict[str, float],
    out_png: Path,
    r_ap: float,
) -> None:
    aij = pd.read_csv(aij_path, sep="\t").sort_values("BJD_TDB")
    t_aij = pd.to_numeric(aij["JD_UTC"], errors="coerce") - 2461154.0
    t_vy = lc["bjd"] - 2461154.0
    n = min(len(lc), len(aij))

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(
        f"{name} - FITS ap={r_ap}px vs AIJ (draft_311)\n"
        f"corr(rel_flux)={stats.get('corr_rel_flux', float('nan')):.4f}  "
        f"RMS vy={stats.get('rms_mag_diff_vy_ppt', float('nan')):.1f} ppt  "
        f"AIJ={stats.get('rms_mag_diff_aij_ppt', float('nan')):.1f} ppt",
        fontsize=10,
    )

    rf_aij = pd.to_numeric(aij["rel_flux_T1"].iloc[:n], errors="coerce")
    rf_vy = lc["rel_flux"].iloc[:n]
    axes[0].plot(t_aij.iloc[:n], rf_aij / np.nanmedian(rf_aij), ".", ms=4, label="AIJ rel_flux_T1 norm")
    axes[0].plot(t_vy.iloc[:n], rf_vy / np.nanmedian(rf_vy), ".", ms=4, alpha=0.7, label=f"VYVAR photutils r={r_ap}")
    axes[0].set_ylabel("rel. flux / median")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    mag_aij_dm = -2.5 * np.log10(np.clip(rf_aij, 1e-12, None))
    mag_aij_dm = mag_aij_dm - np.nanmedian(mag_aij_dm)
    mag_vy_dm = lc["mag_diff"].iloc[:n] - np.nanmedian(lc["mag_diff"].iloc[:n])
    axes[1].plot(t_aij.iloc[:n], mag_aij_dm, ".", ms=4, label="AIJ mag demean")
    axes[1].plot(t_vy.iloc[:n], mag_vy_dm, ".", ms=4, alpha=0.7, label="VYVAR mag_diff demean")
    axes[1].set_ylabel("demeaned mag")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    resid = mag_vy_dm - mag_aij_dm
    axes[2].plot(t_vy.iloc[:n], resid, ".", ms=4, color="#9467bd")
    axes[2].axhline(0, color="gray", lw=0.8)
    axes[2].set_ylabel("VYVAR - AIJ mag")
    axes[2].grid(True, alpha=0.3)

    if vyvar_lc_path is not None and vyvar_lc_path.is_file():
        vy = pd.read_csv(vyvar_lc_path).sort_values("bjd")
        tv = vy["bjd"] - 2461154.0
        m_old = vy["mag_calib"] - np.nanmedian(vy["mag_calib"])
        m_new = lc["mag_calib"] - np.nanmedian(lc["mag_calib"])
        axes[3].plot(tv, m_old, ".", ms=3, alpha=0.5, label=f"VYVAR pipeline (ap~{vy['aperture_r_px'].iloc[0]:.2f})")
        axes[3].plot(t_vy, m_new, ".", ms=3, alpha=0.8, label=f"FITS photutils ap={r_ap}")
        axes[3].set_ylabel("mag_calib demean")
        axes[3].legend(fontsize=7)
    else:
        axes[3].plot(t_vy, mag_vy_dm, ".", ms=4)
        axes[3].set_ylabel("mag_diff demean")
    axes[3].set_xlabel("JD - 2461154")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    ap.add_argument("--setup", default=DEFAULT_SETUP)
    ap.add_argument("--radius", type=float, default=7.0)
    ap.add_argument("--annulus", choices=("scaled", "vyvar"), default="scaled")
    ap.add_argument("--targets", default="FW,BO", help="FW, BO or both")
    ap.add_argument(
        "--coords",
        choices=("aij", "vyvar"),
        default="aij",
        help="aij = X(FITS)/Y(FITS) from .tbl; vyvar = active_targets + comp CSV",
    )
    ap.add_argument(
        "--fits",
        choices=("aligned", "processed"),
        default="aligned",
        help="aligned = detrended_aligned/lights (same as AIJ proc path)",
    )
    args = ap.parse_args()

    draft = args.draft.resolve()
    proc_dir = draft / "detrended_aligned" / "lights" / args.setup
    phot = draft / "platesolve" / args.setup / "photometry"
    lc_dir = phot / "lightcurves"
    out_plot = phot / f"_lc_fits_ap{int(args.radius) if args.radius == int(args.radius) else args.radius}"
    out_plot.mkdir(parents=True, exist_ok=True)

    from ui_aperture_photometry import _load_fwhm  # noqa: PLC0415

    fwhm = float(_load_fwhm(draft / "platesolve" / args.setup / "MASTERSTAR.fits"))
    r_in, r_out = annulus_radii(args.radius, args.annulus, fwhm)
    csv_files, fits_files = paired_paths(proc_dir, args.fits)
    comp_csv = phot / "comparison_stars_per_target.csv"
    active_csv = phot / "active_targets.csv"

    keys = [k.strip().upper() for k in args.targets.split(",") if k.strip()]
    LOGGER.info("Frames=%d  r_ap=%.3f  annulus=%s (%.2f-%.2f px)", len(csv_files), args.radius, args.annulus, r_in, r_out)

    ap_tag = f"ap{int(args.radius) if args.radius == int(args.radius) else args.radius}"

    print("=" * 70)
    print(f"Build LC from FITS - {draft.name} - r={args.radius} px")
    print(f"  coords={args.coords}  fits={args.fits}  annulus={args.annulus}")
    print("=" * 70)

    for key in keys:
        if key not in STARS:
            LOGGER.warning("Unknown target key %s", key)
            continue
        name, cid, aij_file = STARS[key]
        aij_path = draft / "detrended_aligned" / "lights" / aij_file

        if args.coords == "aij":
            lc = build_lc_aij_coords(aij_path, csv_files, fits_files, args.radius, r_in, r_out)
            comp_cids = []
        else:
            comp = pd.read_csv(comp_csv, dtype={"catalog_id": str, "target_catalog_id": str})
            comp_cids = comp[comp["target_catalog_id"].map(_norm) == _norm(cid)]["catalog_id"].map(_norm).tolist()
            xy = load_xy(active_csv, comp_csv, cid, comp_cids)
            LOGGER.info("%s: %d positions / %d comps", name, len(xy), len(comp_cids))
            lc = build_lc(cid, comp_cids, xy, csv_files, fits_files, args.radius, r_in, r_out)
        if lc.empty:
            print(f"  {name}: ERROR - no frames")
            continue

        out_csv = lc_dir / f"lightcurve_{cid}_{ap_tag}.csv"
        lc.to_csv(out_csv, index=False)

        md = lc["mag_diff"].to_numpy(dtype=float)
        rms_broeg = float(np.std(md - np.nanmedian(md)))
        n_used = int(np.isfinite(md).sum())

        vyvar_lc = lc_dir / f"lightcurve_{cid}.csv"
        stats = compare_aij(lc, aij_path)
        stats["rms_broeg_mag_ppt"] = rms_broeg * 1000.0
        if "rel_flux_aij" in lc.columns:
            ok = np.isfinite(lc["rel_flux"]) & np.isfinite(lc["rel_flux_aij"])
            if ok.sum() > 3:
                stats["corr_rel_flux_direct"] = float(
                    np.corrcoef(lc.loc[ok, "rel_flux"], lc.loc[ok, "rel_flux_aij"])[0, 1]
                )
                stats["median_ratio_vy_aij"] = float(np.nanmedian(lc.loc[ok, "rel_flux"] / lc.loc[ok, "rel_flux_aij"]))

        png = out_plot / f"aij_replication_{cid[-8:]}_{ap_tag}.png"
        plot_replication(name, cid, lc, aij_path, vyvar_lc if vyvar_lc.is_file() else None, stats, png, args.radius)

        print(f"\n{name} ({cid})")
        print(f"  CSV: {out_csv}")
        print(f"  PNG: {png}")
        print(f"  Frames: {len(lc)}  Broeg RMS: {rms_broeg:.4f} mag ({rms_broeg*1000:.1f} ppt)")
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}" if np.isfinite(v) else f"  {k}: nan")

    print(f"\nDone. Plots in: {out_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
