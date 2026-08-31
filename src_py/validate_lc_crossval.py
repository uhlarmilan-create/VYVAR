#!/usr/bin/env python3
"""
LC cross-validation: VYVAR photometry_summary lc_rms vs dao_flux differential LC.

Reads dao_flux from proc_*.csv (same flux as VYVAR Phase 2A), applies the same
comp-star ensemble as comparison_stars_per_target.csv (Broeg 2005 equal weights),
then compares RMS of sigma-clipped differential magnitudes to lc_rms in summary.

Default paths target Lenovo / Public/VYVAR layout (draft_000310).
"""

from __future__ import annotations

import glob
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# PyRAF - optional
iraf = None  # type: ignore[assignment,misc]
IRAF_AVAILABLE = False

try:
    from pyraf import iraf as _iraf  # type: ignore[import-untyped]

    _iraf.noao(_doprint=0)
    _iraf.digiphot(_doprint=0)
    _iraf.apphot(_doprint=0)
    iraf = _iraf
    IRAF_AVAILABLE = True
    logging.info("PyRAF available - IRAF LC will be computed")
except Exception as exc:  # noqa: BLE001
    iraf = None
    IRAF_AVAILABLE = False
    logging.warning("PyRAF not available - lc_rms_iraf will be NaN (%s)", exc)

# -- Hardcoded paths (Lenovo, draft_000310) -------------------------------------
DRAFT_DIR = Path("/home/milan/Public/VYVAR/draft_000310")
PROC_DIR = DRAFT_DIR / "detrended_aligned/lights/NoFilter_60_2"
FITS_DIR = DRAFT_DIR / "processed/lights/NoFilter_60_2"
PHOT_DIR = DRAFT_DIR / "platesolve/NoFilter_60_2/photometry"
SUMMARY_CSV = PHOT_DIR / "photometry_summary.csv"
ACTIVE_CSV = PHOT_DIR / "active_targets.csv"
COMP_PER_TARGET_CSV = PHOT_DIR / "comparison_stars_per_target.csv"
OUTPUT_CSV = Path("/home/milan/Public/VYVAR/lc_crossval_results.csv")
OUTPUT_PNG = Path("/home/milan/Public/VYVAR/lc_crossval_plot.png")

MIN_FRAMES = 5
MIN_COMP_PER_FRAME = 2
IRAF_GAIN = 3.17
IRAF_RN = 7.6
R_MIN_PX = 1.918
R_MAX_PX = 5.994

SNR_TABLE: dict[float, float] = {
    7.0: 3.818,
    7.5: 3.718,
    8.0: 3.568,
    8.5: 3.418,
    9.0: 3.318,
    9.5: 3.168,
    10.0: 3.018,
    10.5: 2.868,
    11.0: 2.718,
    11.5: 2.568,
    12.0: 2.418,
    12.5: 2.318,
    13.0: 2.168,
    13.5: 2.068,
    14.0: 1.968,
    14.5: 1.918,
}

FWHM_PX = 2.3976  # from aperture_snr_table.json
ANNULUS_INNER_PX = 4.75 * FWHM_PX  # 11.39px - matches VYVAR annulus_inner_fwhm
ANNULUS_OUTER_PX = 9.0 * FWHM_PX  # 21.58px - matches VYVAR annulus_outer_fwhm
DANNULUS_PX = ANNULUS_OUTER_PX - ANNULUS_INNER_PX  # 10.19px

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def get_aperture_px(mag: float) -> float:
    """SNR-optimal radius [px] for IRAF apphot (matches VYVAR SNR table)."""
    m = float(min(max(float(mag), 7.0), 14.5))
    bin_key = round(m * 2) / 2
    r = SNR_TABLE.get(bin_key, 1.918)
    return float(np.clip(r, R_MIN_PX, R_MAX_PX))


def _row_mag(row: pd.Series) -> float:
    for col in ("phot_g_mean_mag", "gaia_mag", "mag", "catalog_mag"):
        if col in row.index:
            v = float(pd.to_numeric(row.get(col), errors="coerce"))
            if np.isfinite(v):
                return v
    return 12.0


def _norm_cid_int_dotzero(x: object) -> str:
    """Strip; collapse ``123.0`` / ``123.00`` to ``123``. Not Gaia-canonical."""
    s = str(x or "").strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    if re.fullmatch(r"\d+\.0+", s):
        return s.split(".", 1)[0]
    return s


def _pick_mag_column(df: pd.DataFrame) -> str:
    for col in ("gaia_mag", "phot_g_mean_mag", "mag", "gaia_g_mag"):
        if col in df.columns and df[col].notna().any():
            return col
    raise KeyError("No magnitude column in active_targets (expected gaia_mag or phot_g_mean_mag)")


def load_star_list() -> pd.DataFrame:
    """Merge active_targets with VYVAR lc_rms from photometry_summary."""
    active = pd.read_csv(ACTIVE_CSV, dtype={"catalog_id": str, "name": str})
    summary = pd.read_csv(SUMMARY_CSV, dtype={"catalog_id": str})

    active["catalog_id"] = active["catalog_id"].map(_norm_cid_int_dotzero)
    summary["catalog_id"] = summary["catalog_id"].map(_norm_cid_int_dotzero)
    active = active[active["catalog_id"] != ""].copy()
    summary = summary[summary["catalog_id"] != ""].copy()

    mag_col = _pick_mag_column(active)
    use_cols = ["catalog_id", "name", "x", "y", mag_col]
    for c in ("ra", "dec", "ra_deg", "dec_deg"):
        if c in active.columns:
            use_cols.append(c)
    stars = active[use_cols].rename(columns={mag_col: "gaia_mag"})
    stars = stars.merge(
        summary[["catalog_id", "lc_rms"]].rename(columns={"lc_rms": "lc_rms_vyvar"}),
        on="catalog_id",
        how="inner",
    )
    stars["gaia_mag"] = pd.to_numeric(stars["gaia_mag"], errors="coerce")
    stars["lc_rms_vyvar"] = pd.to_numeric(stars["lc_rms_vyvar"], errors="coerce")
    stars["x"] = pd.to_numeric(stars["x"], errors="coerce")
    stars["y"] = pd.to_numeric(stars["y"], errors="coerce")
    return stars.reset_index(drop=True)


def load_comp_map() -> dict[str, list[str]]:
    """target_catalog_id -> list of comp catalog_ids (comparison_stars_per_target.csv)."""
    comp_df = pd.read_csv(
        COMP_PER_TARGET_CSV,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    # catalog_id = comp star, target_catalog_id = target
    comp_df["catalog_id"] = comp_df["catalog_id"].map(_norm_cid_int_dotzero)
    comp_df["target_catalog_id"] = comp_df["target_catalog_id"].map(_norm_cid_int_dotzero)
    comp_df = comp_df[
        (comp_df["target_catalog_id"] != "") & (comp_df["catalog_id"] != "")
    ]
    comp_map = comp_df.groupby("target_catalog_id")["catalog_id"].apply(list).to_dict()
    # dedupe comp lists per target
    comp_map = {k: list(dict.fromkeys(v)) for k, v in comp_map.items()}
    LOGGER.info(
        "comp_map: %d targets, example keys: %s",
        len(comp_map),
        list(comp_map.keys())[:3],
    )
    return comp_map


def discover_frames() -> tuple[list[Path], list[Path]]:
    fits_files = sorted(Path(p) for p in glob.glob(str(FITS_DIR / "proc_BO_CVn_Light_*.fits")))
    csv_files = sorted(Path(p) for p in glob.glob(str(PROC_DIR / "proc_BO_CVn_Light_*.csv")))
    if not fits_files:
        raise FileNotFoundError(f"No proc_BO_CVn_Light_*.fits in {FITS_DIR}")
    if not csv_files:
        raise FileNotFoundError(f"No proc_BO_CVn_Light_*.csv in {PROC_DIR}")
    if len(fits_files) != len(csv_files):
        raise FileNotFoundError(
            f"FITS/CSV frame count mismatch: {len(fits_files)} fits vs {len(csv_files)} csv"
        )
    csv_by_stem = {p.stem: p for p in csv_files}
    paired_csv: list[Path] = []
    for fp in fits_files:
        cp = csv_by_stem.get(fp.stem)
        if cp is None:
            raise FileNotFoundError(f"No proc CSV for FITS stem {fp.stem}")
        paired_csv.append(cp)
    return fits_files, paired_csv


def build_flux_matrix(
    all_cids: list[str],
    csv_files: list[Path],
) -> dict[str, np.ndarray]:
    """flux_matrix[cid][frame_idx] = dao_flux from proc CSV or NaN."""
    n_frames = len(csv_files)
    flux_matrix = {cid: np.full(n_frames, np.nan, dtype=float) for cid in all_cids}

    for i, csv_path in enumerate(csv_files):
        if i == 0 or (i + 1) % 20 == 0:
            LOGGER.info("Loading frame %d/%d: %s", i + 1, n_frames, csv_path.name)

        frame_cat = pd.read_csv(csv_path, dtype={"catalog_id": str, "name": str})
        frame_cat["catalog_id"] = frame_cat["catalog_id"].map(_norm_cid_int_dotzero)
        frame_cat = frame_cat[
            frame_cat["catalog_id"].notna()
            & (frame_cat["catalog_id"] != "")
            & (~frame_cat["catalog_id"].str.startswith("DET_"))
        ]
        frame_cat = frame_cat.set_index("catalog_id", drop=False)

        for cid in all_cids:
            if cid not in frame_cat.index:
                continue
            row = frame_cat.loc[cid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            dao_flux = row.get("dao_flux", np.nan)
            if pd.notna(dao_flux) and float(dao_flux) > 0:
                flux_matrix[cid][i] = float(dao_flux)

    return flux_matrix


def build_iraf_flux_matrix(
    all_cids: list[str],
    fits_files: list[Path],
    csv_files: list[Path],
    iraf_module=None,
) -> dict[str, np.ndarray]:
    """flux_matrix[cid][frame_idx] = IRAF net flux (per-star apphot) or NaN."""
    n_frames = len(fits_files)
    iraf_flux_matrix = {cid: np.full(n_frames, np.nan, dtype=float) for cid in all_cids}

    if not IRAF_AVAILABLE or iraf_module is None:
        return iraf_flux_matrix

    LOGGER.info("Building IRAF flux matrix (per-star)...")
    iraf_work_dir = tempfile.mkdtemp(prefix="vyvar_iraf_")

    try:
        iraf_module.set(uparm=iraf_work_dir + os.sep)
        iraf_module.apphot.phot.setParam("interactive", "no")
        iraf_module.apphot.phot.setParam("verify", "no")
        iraf_module.apphot.phot.setParam("verbose", "no")
        iraf_module.apphot.phot.setParam("salgorithm", "median")
        iraf_module.apphot.datapars.setParam("epadu", IRAF_GAIN)
        iraf_module.apphot.datapars.setParam("gain", IRAF_GAIN)
        iraf_module.apphot.datapars.setParam("readnoise", IRAF_RN)
        iraf_module.apphot.phot.setParam("zmag", 25.0)

        frame_lookups: list[pd.DataFrame] = []
        for csv_file in csv_files:
            fc = pd.read_csv(csv_file, dtype={"catalog_id": str, "name": str})
            fc["catalog_id"] = fc["catalog_id"].map(_norm_cid_int_dotzero)
            fc = fc[
                fc["catalog_id"].notna()
                & (fc["catalog_id"] != "")
                & (~fc["catalog_id"].str.startswith("DET_"))
            ].set_index("catalog_id")
            fc = fc[~fc.index.duplicated(keep="first")]
            frame_lookups.append(fc)

        n_stars = len(all_cids)
        for si, cid in enumerate(all_cids):
            if si == 0 or (si + 1) % 50 == 0:
                LOGGER.info("IRAF star %d/%d: %s", si + 1, n_stars, cid)

            star_frames: list[tuple[int, float, float, float]] = []
            for i, fc in enumerate(frame_lookups):
                if cid not in fc.index:
                    continue
                row = fc.loc[cid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                x = float(pd.to_numeric(row["x"], errors="coerce"))
                y = float(pd.to_numeric(row["y"], errors="coerce"))
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                mag = float(row.get("phot_g_mean_mag", 12.0) or 12.0)
                star_frames.append((i, x, y, mag))

            if not star_frames:
                continue

            cid_safe = re.sub(r"[^\w.-]", "_", str(cid))
            for i, x, y, mag in star_frames:
                r_ap = get_aperture_px(mag)
                coords_f = os.path.join(iraf_work_dir, f"c_{cid_safe}_{i:03d}.txt")
                phot_f = os.path.join(iraf_work_dir, f"p_{cid_safe}_{i:03d}.mag")

                with open(coords_f, "w", encoding="utf-8") as f:
                    f.write(f"{x:.3f} {y:.3f}\n")

                if os.path.exists(phot_f):
                    os.remove(phot_f)

                try:
                    iraf_module.apphot.phot(
                        image=str(fits_files[i]),
                        coords=coords_f,
                        output=phot_f,
                        apertures=f"{r_ap:.2f}",
                        annulus=ANNULUS_INNER_PX,
                        dannulus=DANNULUS_PX,
                        Stdout="/dev/null",
                        Stderr="/dev/null",
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("IRAF phot failed cid=%s frame=%d: %s", cid, i, exc)
                    continue

                try:
                    out = iraf_module.txdump(
                        textfiles=phot_f,
                        fields="FLUX,MSKY,AREA",
                        expr="yes",
                        Stdout=1,
                    )
                    for line in out:
                        parts = line.strip().split()
                        if len(parts) < 3 or "INDEF" in parts:
                            continue
                        gross = float(parts[0])
                        msky = float(parts[1])
                        area = float(parts[2])
                        net = gross - msky * area
                        if net > 0:
                            iraf_flux_matrix[cid][i] = net
                        break
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("IRAF txdump failed cid=%s frame=%d: %s", cid, i, exc)

        LOGGER.info("IRAF flux matrix done")

    finally:
        shutil.rmtree(iraf_work_dir, ignore_errors=True)

    return iraf_flux_matrix


def differential_lc_rms(
    target_cid: str,
    comp_cids: list[str],
    flux_matrix: dict[str, np.ndarray],
    n_frames: int,
) -> tuple[float, int]:
    """Broeg-style differential mag RMS (equal-weight comp ensemble, 3sigma MAD clip)."""
    target_flux = flux_matrix.get(target_cid, np.full(n_frames, np.nan))
    diff_mags: list[float] = []

    for i in range(n_frames):
        t_flux = target_flux[i]
        if not (np.isfinite(t_flux) and t_flux > 0):
            continue

        comp_inst_mags: list[float] = []
        for ccid in comp_cids:
            c_flux = flux_matrix.get(ccid, np.full(n_frames, np.nan))[i]
            if np.isfinite(c_flux) and c_flux > 0:
                comp_inst_mags.append(-2.5 * np.log10(c_flux))

        if len(comp_inst_mags) < MIN_COMP_PER_FRAME:
            continue

        ensemble_mag = float(np.mean(comp_inst_mags))
        diff_mag = -2.5 * np.log10(t_flux) - ensemble_mag
        diff_mags.append(diff_mag)

    n_used = len(diff_mags)
    if n_used < MIN_FRAMES:
        return float("nan"), n_used

    arr = np.asarray(diff_mags, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad > 0 and np.isfinite(mad):
        mask = np.abs(arr - med) < 3.0 * 1.4826 * mad
        if int(mask.sum()) >= MIN_FRAMES:
            arr = arr[mask]
            n_used = int(mask.sum())

    return float(np.std(arr)), n_used


def build_results(
    stars: pd.DataFrame,
    comp_map: dict[str, list[str]],
    flux_matrix: dict[str, np.ndarray],
    iraf_flux_matrix: dict[str, np.ndarray],
    n_frames: int,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for _, star in stars.iterrows():
        cid = _norm_cid_int_dotzero(star["catalog_id"])
        gaia_mag = star["gaia_mag"]
        lc_rms_vyvar = float(star["lc_rms_vyvar"])

        comp_cids = comp_map.get(cid, [])
        if len(comp_cids) < MIN_COMP_PER_FRAME:
            LOGGER.warning("Star %s: fewer than %d comp stars, skipping", cid, MIN_COMP_PER_FRAME)
            continue

        lc_rms_phot, n_used = differential_lc_rms(cid, comp_cids, flux_matrix, n_frames)
        if not np.isfinite(lc_rms_phot):
            LOGGER.warning(
                "Star %s (%s): dao_flux differential LC has only %d/%d valid frames",
                cid,
                star.get("name", ""),
                n_used,
                n_frames,
            )

        if IRAF_AVAILABLE:
            lc_rms_iraf, n_iraf = differential_lc_rms(cid, comp_cids, iraf_flux_matrix, n_frames)
            if not np.isfinite(lc_rms_iraf):
                LOGGER.debug(
                    "Star %s (%s): IRAF differential LC has only %d/%d valid frames",
                    cid,
                    star.get("name", ""),
                    n_iraf,
                    n_frames,
                )
        else:
            lc_rms_iraf = float("nan")

        results.append(
            {
                "catalog_id": cid,
                "name": star.get("name", ""),
                "gaia_mag": gaia_mag,
                "lc_rms_vyvar": lc_rms_vyvar,
                "lc_rms_photutils": lc_rms_phot,
                "lc_rms_iraf": lc_rms_iraf,
                "delta_phot": lc_rms_phot - lc_rms_vyvar
                if np.isfinite(lc_rms_phot) and np.isfinite(lc_rms_vyvar)
                else np.nan,
                "delta_iraf": lc_rms_iraf - lc_rms_vyvar
                if np.isfinite(lc_rms_iraf) and np.isfinite(lc_rms_vyvar)
                else np.nan,
                "n_frames": n_used,
            }
        )
    return results


def log_mag_bin_summary(df: pd.DataFrame) -> None:
    for mag_lo, mag_hi in [(8, 10), (10, 11), (11, 12), (12, 13), (13, 14)]:
        sub = df[
            (df["gaia_mag"] >= mag_lo)
            & (df["gaia_mag"] < mag_hi)
            & df["delta_phot"].notna()
        ]
        if len(sub):
            LOGGER.info(
                "Mag %d-%d: N=%d | median Delta=%.4f | std Delta=%.4f",
                mag_lo,
                mag_hi,
                len(sub),
                float(sub["delta_phot"].median()),
                float(sub["delta_phot"].std()),
            )


def save_plot(df: pd.DataFrame, out_png: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    v = df["lc_rms_vyvar"].to_numpy(dtype=float)
    p = df["lc_rms_photutils"].to_numpy(dtype=float)
    i = df["lc_rms_iraf"].to_numpy(dtype=float)
    ok_p = np.isfinite(v) & np.isfinite(p) & (v > 0) & (p > 0)
    ok_i = np.isfinite(v) & np.isfinite(i) & (v > 0) & (i > 0)

    if ok_p.any():
        ax1.scatter(v[ok_p], p[ok_p], s=18, alpha=0.7, c="steelblue", label="photutils")
    if ok_i.any():
        ax1.scatter(v[ok_i], i[ok_i], s=18, alpha=0.7, c="darkorange", label="IRAF")
    if ok_p.any() or ok_i.any():
        lo = float(np.nanmin(np.r_[v[ok_p | ok_i], p[ok_p], i[ok_i]])) * 0.8
        hi = float(np.nanmax(np.r_[v[ok_p | ok_i], p[ok_p], i[ok_i]])) * 1.2
        if np.isfinite(lo) and np.isfinite(hi) and lo > 0:
            ax1.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
            ax1.set_xscale("log")
            ax1.set_yscale("log")
    ax1.set_xlabel("lc_rms VYVAR [mag]")
    ax1.set_ylabel("lc_rms tool [mag]")
    ax1.set_title("LC RMS cross-validation")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    mag = df["gaia_mag"].to_numpy(dtype=float)
    dp = df["delta_phot"].to_numpy(dtype=float)
    di = df["delta_iraf"].to_numpy(dtype=float)
    ok_dp = np.isfinite(mag) & np.isfinite(dp)
    ok_di = np.isfinite(mag) & np.isfinite(di)
    if ok_dp.any():
        ax2.scatter(mag[ok_dp], dp[ok_dp], s=18, alpha=0.7, c="steelblue", label="photutils - VYVAR")
    if ok_di.any():
        ax2.scatter(mag[ok_di], di[ok_di], s=18, alpha=0.7, c="darkorange", label="IRAF - VYVAR")
    ax2.axhline(0.0, color="k", ls="--", lw=1)
    ax2.set_xlabel("Gaia mag")
    ax2.set_ylabel("Delta lc_rms [mag]")
    ax2.set_title("RMS offset vs magnitude")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    LOGGER.info("Loading star list from %s + %s", ACTIVE_CSV, SUMMARY_CSV)
    stars = load_star_list()
    LOGGER.info("Targets in summary: %d", len(stars))

    LOGGER.info("Loading comp assignments from %s", COMP_PER_TARGET_CSV)
    comp_map = load_comp_map()

    all_comp_cids: set[str] = set()
    for comps in comp_map.values():
        all_comp_cids.update(comps)
    target_cids = [_norm_cid_int_dotzero(x) for x in stars["catalog_id"]]
    all_cids = list(dict.fromkeys([c for c in target_cids if c] + sorted(all_comp_cids)))
    LOGGER.info("Flux matrix: %d stars (targets + comps)", len(all_cids))

    fits_files, csv_files = discover_frames()
    LOGGER.info("Frames: %d FITS + CSV pairs", len(fits_files))

    flux_matrix = build_flux_matrix(all_cids, csv_files)
    iraf_flux_matrix = build_iraf_flux_matrix(
        all_cids, fits_files, csv_files, iraf_module=iraf
    )

    results = build_results(
        stars,
        comp_map,
        flux_matrix,
        iraf_flux_matrix,
        len(fits_files),
    )
    out = pd.DataFrame(results)
    log_mag_bin_summary(out)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    LOGGER.info("Wrote %s (%d rows)", OUTPUT_CSV, len(out))

    save_plot(out, OUTPUT_PNG)
    LOGGER.info("Wrote %s", OUTPUT_PNG)

    valid_p = out["lc_rms_photutils"].notna()
    if valid_p.any():
        dp = out.loc[valid_p, "delta_phot"]
        LOGGER.info(
            "photutils dao_flux (differential) vs VYVAR: median Delta=%.4f mag, std Delta=%.4f, N=%d",
            float(dp.median()),
            float(dp.std()),
            int(valid_p.sum()),
        )

    iraf_valid = int(out["lc_rms_iraf"].notna().sum())
    if iraf_valid > 0:
        iraf_delta = (out["lc_rms_iraf"] - out["lc_rms_vyvar"]).dropna()
        LOGGER.info(
            "IRAF (differential) vs VYVAR: median Delta=%.4f mag, std Delta=%.4f, N=%d",
            float(iraf_delta.median()),
            float(iraf_delta.std()),
            iraf_valid,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
