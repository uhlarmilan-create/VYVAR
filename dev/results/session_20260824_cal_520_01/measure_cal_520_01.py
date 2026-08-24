"""CAL-520-01 read-only radiometric + library + comp-forensics measure.

Does not reclassify, recalibrate, or write science FITS.
Rig: Brno AZ800 / C5A-150M. ASCII only.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import binary_dilation, gaussian_filter, shift as ndi_shift

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src_py"))

from comp_pool_rms import compute_global_pool_rms_map  # noqa: E402
from invariants_runtime import (  # noqa: E402
    PREPROCESS_LARGE_SMALL_RATIO_WARN,
    preprocess_large_small_ratio,
)

OUT = Path(__file__).resolve().parent
G520 = REPO / "Archive" / "Drafts" / "draft_000520" / "non_calibrated" / "lights" / "g_60_4"
REF516 = (
    REPO
    / "Archive"
    / "Drafts"
    / "draft_000516"
    / "calibrated"
    / "lights"
    / "NoFilter_60_2"
    / "BO_CVn_Light_001.fits"
)
EPSF516 = (
    REPO
    / "Archive"
    / "Drafts"
    / "draft_000516"
    / "platesolve"
    / "NoFilter_60_2"
    / "masterstar_epsf.fits"
)
LIB = REPO / "CalibrationLibrary"
DB_PATH = REPO / "vyvar.sqlite3"
MS_CSV = REPO / "Archive" / "Drafts" / "draft_000520" / "platesolve" / "g_60_4" / "masterstars_full_match.csv"
COMP_PT = (
    REPO
    / "Archive"
    / "Drafts"
    / "draft_000520"
    / "platesolve"
    / "g_60_4"
    / "photometry"
    / "comparison_stars_per_target.csv"
)
PROC_DIR = REPO / "Archive" / "Drafts" / "draft_000520" / "detrended_aligned" / "lights" / "g_60_4"
LC_V0612 = (
    REPO
    / "Archive"
    / "Drafts"
    / "draft_000520"
    / "platesolve"
    / "g_60_4"
    / "photometry"
    / "lightcurves"
    / "lightcurve_1111749368289526912.csv"
)
TARGET_CID = "1111749368289526912"
EXPECTED_EPSF_SHA = "172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_frame(path: Path) -> tuple[np.ndarray, dict]:
    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        hdr = hdul[0].header
        cards = {
            "DATE-OBS": str(hdr.get("DATE-OBS") or hdr.get("DATE") or ""),
            "FILTER": str(hdr.get("FILTER") or hdr.get("FILTER2") or ""),
            "XBINNING": hdr.get("XBINNING"),
            "YBINNING": hdr.get("YBINNING"),
            "EXPTIME": hdr.get("EXPTIME") or hdr.get("EXPOSURE"),
            "CCD-TEMP": hdr.get("CCD-TEMP") or hdr.get("CCDTEMP") or hdr.get("SET-TEMP"),
            "GAIN": hdr.get("GAIN"),
            "INSTRUME": str(hdr.get("INSTRUME") or ""),
            "TELESCOP": str(hdr.get("TELESCOP") or ""),
            "IMAGETYP": str(hdr.get("IMAGETYP") or ""),
            "NAXIS1": int(hdr.get("NAXIS1") or data.shape[1]),
            "NAXIS2": int(hdr.get("NAXIS2") or data.shape[0]),
        }
    return data, cards


def star_mask(arr: np.ndarray, *, k: float = 6.0, dilate_iter: int = 10) -> np.ndarray:
    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite])) if int(np.count_nonzero(finite)) else 0.0
    work = np.where(finite, arr, fill)
    lp = gaussian_filter(work, sigma=25.0)
    resid = work - lp
    med = float(np.median(resid[finite]))
    mad = float(np.median(np.abs(resid[finite] - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(resid[finite]))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    stars = (resid > (med + k * sigma)) & finite
    if dilate_iter > 0:
        stars = binary_dilation(stars, iterations=int(dilate_iter))
    return stars


def large_scale_metrics(arr: np.ndarray, *, sigmas: tuple[float, ...] = (50.0, 80.0, 150.0)) -> dict:
    finite = np.isfinite(arr)
    stars = star_mask(arr)
    sky_pix = finite & (~stars)
    n_sky = int(np.count_nonzero(sky_pix))
    sky = float(np.median(arr[sky_pix])) if n_sky > 100 else float(np.nanmedian(arr[finite]))
    fill = sky if np.isfinite(sky) else 0.0
    work = np.where(sky_pix, arr, fill)
    out: dict = {
        "n_sky": n_sky,
        "n_star_mask": int(np.count_nonzero(stars)),
        "sky_median_adu": sky,
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
    }
    for sig in sigmas:
        blur = gaussian_filter(work, sigma=float(sig))
        vals = blur[sky_pix]
        med_b = float(np.median(vals))
        rms = float(np.std(vals))
        p1, p99 = (float(x) for x in np.percentile(vals, [1.0, 99.0]))
        ptp = p99 - p1
        out[f"sigma_{int(sig)}"] = {
            "rms_adu": rms,
            "p99_p1_adu": ptp,
            "rel_rms_to_sky": (rms / sky) if sky else float("nan"),
            "rel_p99_p1_to_sky": (ptp / sky) if sky else float("nan"),
            "blur_median_adu": med_b,
        }
    out["inv_prep01_large_small_ratio"] = float(preprocess_large_small_ratio(arr))
    return out


def ratio_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    finite = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    med_a = float(np.median(a[finite]))
    med_b = float(np.median(b[finite]))
    scale = med_a / med_b if med_b else 1.0
    ratio = np.full_like(a, np.nan, dtype=np.float64)
    ratio[finite] = (b[finite] * scale) / a[finite]
    stars = star_mask(a) | star_mask(b)
    sky = finite & (~stars)
    r = ratio[sky]
    r = r[np.isfinite(r)]
    fill = 1.0
    work = np.where(sky, np.where(np.isfinite(ratio), ratio, fill), fill)
    blur = gaussian_filter(work, sigma=80.0)
    bv = blur[sky]
    rms = float(np.std(bv))
    p1, p99 = (float(x) for x in np.percentile(bv, [1.0, 99.0]))
    return {
        "median_a": med_a,
        "median_b": med_b,
        "scale_b_to_a": scale,
        "n_sky": int(np.count_nonzero(sky)),
        "ratio_median": float(np.median(r)),
        "ratio_lp80_rms": rms,
        "ratio_lp80_p99_p1": p99 - p1,
        "ratio_minus_1_lp80_rms": rms,
        "blur": blur,
        "sky_mask": sky,
        "ratio": ratio,
    }


def save_ratio_plot(path: Path, frame_lp: np.ndarray, ratio_lp: np.ndarray, sky: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _show(ax, img, title, cmap, vmin, vmax):
        vis = np.array(img, copy=True)
        vis[~sky] = np.nan
        im = ax.imshow(vis, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    fvals = frame_lp[sky]
    rvals = ratio_lp[sky]
    f_lo, f_hi = np.percentile(fvals, [2.0, 98.0])
    r_span = float(np.percentile(np.abs(rvals - 1.0), 98.0))
    _show(axes[0], frame_lp, "520 g_0000 star-masked low-pass (80 px)", "gray", f_lo, f_hi)
    _show(
        axes[1],
        ratio_lp,
        "g_0096/g_0000 ratio low-pass (80 px)",
        "coolwarm",
        1.0 - r_span,
        1.0 + r_span,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def header_jsonable(cards: dict) -> dict:
    out = {}
    for k, v in cards.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v) if isinstance(v, np.floating) else int(v)
        else:
            out[k] = v
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g_files = sorted(G520.glob("SSCam_*_g_*.fits"))
    if len(g_files) < 2:
        raise SystemExit(f"expected 520 g lights under {G520}")
    first, last = g_files[0], g_files[-1]
    second = g_files[1]

    a, ha = load_frame(first)
    b, hb = load_frame(last)
    c, _hc = load_frame(second)
    mid, hmid = load_frame(g_files[len(g_files) // 2])
    ref, href = load_frame(REF516)

    m_first = large_scale_metrics(a)
    m_mid = large_scale_metrics(mid)
    m_last = large_scale_metrics(b)
    m_ref = large_scale_metrics(ref)

    inv_all = []
    for fp in g_files:
        arr, _ = load_frame(fp)
        inv_all.append(
            {
                "file": fp.name,
                "large_small_ratio": float(preprocess_large_small_ratio(arr)),
            }
        )

    r_fl = ratio_metrics(a, b)
    r_fs = ratio_metrics(a, c)

    # Integer-pixel star align of last onto first (bright-residual cross-corr, +-32 px).
    stars_a = star_mask(a, k=5.0, dilate_iter=0)
    lp_a = gaussian_filter(np.where(np.isfinite(a), a, np.nanmedian(a)), 25.0)
    hp_a = np.where(stars_a, a - lp_a, 0.0)
    lp_b = gaussian_filter(np.where(np.isfinite(b), b, np.nanmedian(b)), 25.0)
    stars_b = star_mask(b, k=5.0, dilate_iter=0)
    hp_b = np.where(stars_b, b - lp_b, 0.0)
    maxlag = 32
    from numpy.fft import fft2, ifft2

    fa = fft2(hp_a)
    fb = fft2(hp_b)
    corr = np.real(ifft2(fa * np.conj(fb)))
    ny, nx = corr.shape
    corr = np.roll(np.roll(corr, ny // 2, 0), nx // 2, 1)
    cy, cx = ny // 2, nx // 2
    win = corr[cy - maxlag : cy + maxlag + 1, cx - maxlag : cx + maxlag + 1]
    iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)
    dy = int(iy - maxlag)
    dx = int(ix - maxlag)
    b_shift = ndi_shift(b, shift=(-dy, -dx), order=1, mode="nearest")
    r_aligned = ratio_metrics(a, b_shift)

    blur_first = gaussian_filter(
        np.where(r_fl["sky_mask"], a, float(np.median(a[r_fl["sky_mask"]]))),
        80.0,
    )
    plot_path = OUT / "two_light_ratio_g0000_g0096.png"
    save_ratio_plot(plot_path, blur_first, r_fl["blur"], r_fl["sky_mask"])

    # Drop large arrays from JSON.
    r_fl_j = {k: v for k, v in r_fl.items() if k not in ("blur", "sky_mask", "ratio")}
    r_fs_j = {k: v for k, v in r_fs.items() if k not in ("blur", "sky_mask", "ratio")}
    r_al_j = {k: v for k, v in r_aligned.items() if k not in ("blur", "sky_mask", "ratio")}

    m2 = {
        "g_n_lights": len(g_files),
        "g_first": first.name,
        "g_last": last.name,
        "g_first_header": header_jsonable(ha),
        "g_last_header": header_jsonable(hb),
        "ref516": REF516.name,
        "ref516_header": header_jsonable(href),
        "large_scale": {
            "g_0000": m_first,
            "g_mid": m_mid,
            "g_last": m_last,
            "ref516_BO_CVn_Light_001": m_ref,
        },
        "inv_prep01_warn_threshold": PREPROCESS_LARGE_SMALL_RATIO_WARN,
        "inv_prep01_this_measure_all_g": inv_all,
        "inv_prep01_g_median": float(np.median([x["large_small_ratio"] for x in inv_all])),
        "inv_prep01_recorded_in_run": {
            "g_60_4": 0.02,
            "i_70_4": 0.02,
            "r_60_4": 0.01,
            "z_90_4": 0.06,
            "warn": 10.0,
            "source": "infolog_20260824_204055.txt 18:43:19-18:43:25",
        },
        "two_light": {
            "pair_first_last_pixel": r_fl_j,
            "pair_first_second_pixel": r_fs_j,
            "star_xcorr_shift_last_minus_first_px": {"dy": dy, "dx": dx},
            "pair_first_last_star_aligned": r_al_j,
            "plot": str(plot_path.name),
        },
    }
    (OUT / "m2_radiometric.json").write_text(json.dumps(m2, indent=2), encoding="utf-8")

    # M3 library
    lib_files = []
    for p in sorted(LIB.glob("*.fits")):
        arr, cards = load_frame(p)
        lib_files.append(
            {
                "name": p.name,
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
                "header": header_jsonable(cards),
            }
        )

    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cal_rows = [dict(r) for r in conn.execute("SELECT * FROM CALIBRATION_LIBRARY").fetchall()]
    eq_rows = []
    for table, q in (
        ("EQUIPMENTS", "SELECT ID, CAMERANAME, PIXELSIZE FROM EQUIPMENTS"),
        ("TELESCOPE", "SELECT ID, TELESCOPENAME, FOCAL FROM TELESCOPE"),
    ):
        try:
            eq_rows.append({"table": table, "rows": [dict(r) for r in conn.execute(q).fetchall()]})
        except sqlite3.Error as exc:
            eq_rows.append({"table": table, "error": str(exc)})

    light_xb = int(ha.get("XBINNING") or 4)
    light_exp = float(ha.get("EXPTIME") or 60.0)
    light_temp = ha.get("CCD-TEMP")
    try:
        light_temp_f = float(light_temp) if light_temp is not None else None
    except (TypeError, ValueError):
        light_temp_f = None
    try:
        light_gain = int(float(ha.get("GAIN") or 0))
    except (TypeError, ValueError):
        light_gain = 0

    def _best(kind: str, xb: int, flt: str) -> str | None:
        if kind == "dark":
            if light_temp_f is None:
                return None
            q = """
                SELECT FILE_PATH FROM CALIBRATION_LIBRARY
                WHERE KIND = 'dark' AND XBINNING = ? AND EXPTIME = ?
                  AND COALESCE(GAIN, 0) = ? AND COALESCE(FILTER_NAME, '') = ''
                  AND CCD_TEMP IS NOT NULL AND ABS(CCD_TEMP - ?) <= 0.5
                  AND ID_EQUIPMENTS = 4 AND ID_TELESCOPE = 6
            """
            params = (int(xb), float(light_exp), int(light_gain), float(light_temp_f))
        else:
            q = """
                SELECT FILE_PATH FROM CALIBRATION_LIBRARY
                WHERE KIND = 'flat' AND XBINNING = ?
                  AND COALESCE(GAIN, 0) = ? AND FILTER_NAME = ?
                  AND ID_EQUIPMENTS = 4 AND ID_TELESCOPE = 6
            """
            params = (int(xb), int(light_gain), str(flt))
        hits = [str(r["FILE_PATH"]) for r in conn.execute(q, params).fetchall()]
        hits = [h for h in hits if Path(h).is_file()]
        return hits[0] if hits else None

    dark_path = _best("dark", 1, "") or _best("dark", light_xb, "")
    flat_path = _best("flat", 1, "g") or _best("flat", light_xb, "g")
    dark_any = dark_path
    flat_nofilter = _best("flat", 1, "NoFilter") or _best("flat", light_xb, "NoFilter")
    conn.close()

    m3 = {
        "light_match_keys": {
            "xbinning": light_xb,
            "exptime": light_exp,
            "ccd_temp": light_temp_f,
            "gain": light_gain,
            "filter": "g",
            "id_equipments": 4,
            "id_telescope": 6,
            "light_date": ha.get("DATE-OBS"),
        },
        "validity_windows_days": {"dark": 90, "flat": 200},
        "library_fits_on_disk": lib_files,
        "calibration_library_db_n": len(cal_rows),
        "calibration_library_db_rows": cal_rows,
        "equipment_tables": eq_rows,
        "find_best_eq4_tel6": {
            "dark_bin4_g_60s": dark_path,
            "flat_bin4_g": flat_path,
            "dark_prefer_unbinned": dark_any,
            "flat_nofilter_unbinned": flat_nofilter,
        },
    }
    (OUT / "m3_calibration_library.json").write_text(json.dumps(m3, indent=2, default=str), encoding="utf-8")

    # M4 comps
    ms = pd.read_csv(MS_CSV, dtype={"catalog_id": str, "name": str}, low_memory=False)
    ms["phot_g_mean_mag"] = pd.to_numeric(ms["phot_g_mean_mag"], errors="coerce")
    ms["catalog_id"] = ms["catalog_id"].astype(str).str.strip()
    june_band = ms[(ms["phot_g_mean_mag"] >= 11.6) & (ms["phot_g_mean_mag"] <= 13.9)].copy()
    pt = pd.read_csv(COMP_PT, dtype={"catalog_id": str, "target_catalog_id": str}, low_memory=False)
    pt["catalog_id"] = pt["catalog_id"].astype(str).str.strip()
    pt["target_catalog_id"] = pt["target_catalog_id"].astype(str).str.strip()
    v0612 = pt[pt["target_catalog_id"] == TARGET_CID].copy()
    selected_ids = set(v0612["catalog_id"].astype(str))
    june_ids = set(june_band["catalog_id"].astype(str))
    want_ids = selected_ids | june_ids

    proc_paths = sorted(PROC_DIR.glob("proc_*.csv"))
    cache: dict[str, pd.DataFrame] = {}
    usecols = [
        "name",
        "catalog_id",
        "bjd_tdb_mid",
        "flux",
        "dao_flux",
        "noise_floor_adu",
        "aperture_r_px",
        "snr50_ok",
        "vsx_known_variable",
        "likely_saturated",
        "x",
        "y",
        "peak_dao",
        "fwhm_estimate_px",
        "psf_chi2",
        "phot_g_mean_mag",
    ]
    for p in proc_paths:
        dfp = pd.read_csv(p, usecols=lambda c: c in usecols, dtype={"catalog_id": str, "name": str})
        cache[str(p)] = dfp

    rms_map = compute_global_pool_rms_map(
        want_ids,
        ms,
        proc_paths,
        cache,
        flux_col="dao_flux",
        apply_rms_prefilter=False,
        max_comp_rms=99.0,
    )

    rows = []
    for cid, gmag in zip(ms["catalog_id"], ms["phot_g_mean_mag"]):
        if cid not in want_ids:
            continue
        sub = v0612[v0612["catalog_id"] == cid]
        rows.append(
            {
                "catalog_id": cid,
                "phot_g_mean_mag": float(gmag) if pd.notna(gmag) else None,
                "band": (
                    "june_G_11.6_13.9"
                    if cid in june_ids
                    else "selected_faint"
                ),
                "selected_for_V0612": bool(cid in selected_ids),
                "comp_rms_on_selected_row": (
                    float(sub["comp_rms"].iloc[0]) if len(sub) and pd.notna(sub["comp_rms"].iloc[0]) else None
                ),
                "comp_rms_fieldwide_today": rms_map.get(cid),
                "exclusion_reason": (
                    None
                    if cid not in set(ms["catalog_id"])
                    else (
                        None
                        if "exclusion_reason" not in ms.columns
                        else (
                            str(ms.loc[ms["catalog_id"] == cid, "exclusion_reason"].iloc[0])
                            if len(ms.loc[ms["catalog_id"] == cid])
                            else None
                        )
                    )
                ),
            }
        )
    # also selected faint not already in ms loop (they are)
    df_comp = pd.DataFrame(rows).sort_values(["band", "phot_g_mean_mag"])
    df_comp.to_csv(OUT / "m4_comp_forensics.csv", index=False)

    june_rms = [x for x in df_comp.loc[df_comp["band"] == "june_G_11.6_13.9", "comp_rms_fieldwide_today"] if pd.notna(x)]
    sel_rms = [x for x in df_comp.loc[df_comp["selected_for_V0612"], "comp_rms_fieldwide_today"] if pd.notna(x)]
    m4 = {
        "target": "V0612 Cam",
        "target_catalog_id": TARGET_CID,
        "june_band_n_in_field": int(len(june_ids)),
        "selected_n": int(len(selected_ids)),
        "june_band_comp_rms_today_median": float(np.median(june_rms)) if june_rms else None,
        "june_band_comp_rms_today_min": float(np.min(june_rms)) if june_rms else None,
        "june_band_comp_rms_today_max": float(np.max(june_rms)) if june_rms else None,
        "selected_comp_rms_today_median": float(np.median(sel_rms)) if sel_rms else None,
        "selected_comp_rms_today_min": float(np.min(sel_rms)) if sel_rms else None,
        "selected_comp_rms_today_max": float(np.max(sel_rms)) if sel_rms else None,
        "selected_phot_g_min": float(np.nanmin(v0612["phot_g_mean_mag"])) if len(v0612) else None,
        "selected_phot_g_max": float(np.nanmax(v0612["phot_g_mean_mag"])) if len(v0612) else None,
        "june_gaia_ids_from_june_table": "not_recovered",
        "lc_on_disk_time_base_unique": sorted(
            pd.read_csv(LC_V0612, usecols=["time_base"])["time_base"].astype(str).unique().tolist()
        ),
    }
    (OUT / "m4_comp_forensics_summary.json").write_text(json.dumps(m4, indent=2), encoding="utf-8")

    epsf_sha = sha256_file(EPSF516) if EPSF516.is_file() else None
    gates = {
        "g1_head": None,
        "g2_epsf_sha": epsf_sha,
        "g2_expected": EXPECTED_EPSF_SHA,
        "g2_match": epsf_sha == EXPECTED_EPSF_SHA,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print("INV-PREP g median", m2["inv_prep01_g_median"])
    print("g0000 rel_p99_p1 sigma80", m_first["sigma_80"]["rel_p99_p1_to_sky"])
    print("516 rel_p99_p1 sigma80", m_ref["sigma_80"]["rel_p99_p1_to_sky"])
    print("ratio first-last lp80 rms", r_fl_j["ratio_lp80_rms"])
    print("xcorr dx dy", dx, dy)
    print("library n", len(lib_files), "db rows", len(cal_rows))
    print("dark", dark_path, "flat", flat_path)
    print("june n", m4["june_band_n_in_field"], "june median rms", m4["june_band_comp_rms_today_median"])
    print("selected median rms", m4["selected_comp_rms_today_median"])
    print("epsf", epsf_sha, "match", gates["g2_match"])


if __name__ == "__main__":
    main()
