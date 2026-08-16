#!/usr/bin/env python3
"""WIDE-ERR A2b: measure effective gain from raw data (read-only)."""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))
sys.path.insert(0, str(REPO / "dev" / "scripts"))
import _bootstrap  # noqa: E402,F401

from calibration import resample_master_to_light_binning  # noqa: E402
from config import AppConfig  # noqa: E402
from param_resolver import resolve_gain, resolve_read_noise  # noqa: E402

OUT = REPO / "tmp" / "wide_err_a2b"
DRAFT = REPO / "Archive" / "Drafts" / "draft_000435_snapshot_skysurface_20260716"
RAW_LIGHTS = DRAFT / "Raw" / "lights" / "NoFilter_60_2"
FLAT_DIR = Path(r"D:\FLAT56\Flat")
DARK_DIR = Path(r"D:\DARKS\60s")
CAL_LIB = REPO / "CalibrationLibrary"
CHECK_CID = "1499906247391001088"
W1W2_LC = REPO / "tmp" / "wide_err_w1w2" / "diag_check_lc"
MAG_ERR_SCALE = 1000.0
MAD_SCALE = 1.4826


def _regions(shape: tuple[int, int], n: int = 5, size: int = 200) -> list[tuple[int, int]]:
    h, w = shape
    cy, cx = h // 2, w // 2
    offs = [(-size, -size), (size, -size), (0, 0), (-size, size), (size, size), (0, -size), (0, size)]
    out: list[tuple[int, int]] = []
    for dy, dx in offs:
        y0 = int(np.clip(cy + dy - size // 2, 0, h - size))
        x0 = int(np.clip(cx + dx - size // 2, 0, w - size))
        out.append((y0, x0))
        if len(out) >= n:
            break
    return out


def _clip_region(arr: np.ndarray, sigma: float = 3.0) -> tuple[float, float, float]:
    flat = arr.astype(np.float64).ravel()
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med)))
    if mad <= 0:
        ok = np.isfinite(flat)
        return float(np.mean(flat[ok])), float(np.var(flat[ok])), 1.0
    z = 0.6745 * (flat - med) / mad
    keep = np.abs(z) < sigma
    kept = flat[keep & np.isfinite(flat)]
    frac = float(kept.size / flat.size) if flat.size else 0.0
    if kept.size < 10:
        kept = flat[np.isfinite(flat)]
    return float(np.mean(kept)), float(np.var(kept, ddof=1)), frac


def m0(cfg: AppConfig) -> dict[str, Any]:
    meta = json.loads((DRAFT / "platesolve" / "NoFilter_60_2" / "photometry" / "pipeline_meta.json").read_text())
    dyn = meta.get("dynamic_params", {})
    raw_path = RAW_LIGHTS / "BO_CVn_Light_001.fits"
    hdr = fits.getheader(raw_path)
    from database import VyvarDatabase  # noqa: PLC0415

    db = VyvarDatabase(cfg.database_path)
    g = resolve_gain(hdr, db=db, equipment_id=1, cfg=cfg)
    rn = resolve_read_noise(hdr, db=db, equipment_id=1, cfg=cfg)
    # Did bin2 DB scaling fire? header_index_mapped bypasses _scale_bin1_db_for_header for gain.
    scale_fired_gain = g.source == "db" and int(hdr.get("XBINNING", 1)) == 2
    scale_fired_rn = rn.source == "db" and float(rn.value) != float(db.get_equipment_cosmic_params(1)[1] or 0)

    conn = sqlite3.connect(str(cfg.database_path))
    conn.row_factory = sqlite3.Row
    equip = dict(conn.execute("SELECT * FROM EQUIPMENTS WHERE ID=1").fetchone())
    conn.close()

    hdr_keys = [
        "GAIN", "EGAIN", "XBINNING", "YBINNING", "XPIXSZ", "READOUTM", "READMODE",
        "OFFSET", "EXPTIME", "DATE-OBS", "APTDIA", "FOCALLEN",
    ]
    return {
        "pipeline_meta_gain": dyn.get("gain"),
        "pipeline_meta_read_noise": dyn.get("read_noise"),
        "resolved_gain": g.value,
        "resolved_gain_source": g.source,
        "resolved_gain_key": g.key,
        "resolved_rn": rn.value,
        "resolved_rn_source": rn.source,
        "scale_bin1_db_for_header_fired_gain": scale_fired_gain,
        "scale_bin1_db_for_header_fired_rn": scale_fired_rn,
        "note_gain": (
            "header_index_mapped: _scale_bin1_to_binning NOT applied to gain on this run"
            if g.source == "header_index_mapped"
            else "see resolved_gain_source"
        ),
        "raw_header": {k: hdr.get(k) for k in hdr_keys if k in hdr},
        "equipments_id1": equip,
        "cal_diag_convention": meta.get("cal_diag", {}).get("keys", {}),
    }


def m1() -> dict[str, Any]:
    bias_path = CAL_LIB / "Dark_60s_Dark_0G_-10deg_Bin1_20260422.fits"
    bias = fits.getdata(bias_path, mmap=True).astype(np.float64)
    flats = sorted(FLAT_DIR.glob("FLAT56_Flat_*.fits"))
    pairs = [(flats[i], flats[i + 1]) for i in range(0, min(18, len(flats) - 1), 2)]
    pair_rows: list[dict[str, Any]] = []
    g1_all: list[float] = []
    s_all: list[float] = []
    var_all: list[float] = []
    for fa, fb in pairs[:3]:
        da = fits.getdata(fa, mmap=True).astype(np.float64)
        db = fits.getdata(fb, mmap=True).astype(np.float64)
        ha, hb = fits.getheader(fa), fits.getheader(fb)
        reg_g: list[float] = []
        for y0, x0 in _regions(da.shape, n=5, size=200):
            ra = da[y0 : y0 + 200, x0 : x0 + 200] - bias[y0 : y0 + 200, x0 : x0 + 200]
            rb = db[y0 : y0 + 200, x0 : x0 + 200] - bias[y0 : y0 + 200, x0 : x0 + 200]
            sa = float(np.mean(ra))
            sb = float(np.mean(rb))
            s = 0.5 * (sa + sb)
            diff = ra - rb
            var_diff = float(np.var(diff, ddof=1))
            if var_diff <= 0 or s <= 0:
                continue
            g1 = s / (var_diff / 2.0)
            reg_g.append(g1)
            g1_all.append(g1)
            s_all.append(s)
            var_all.append(var_diff / 2.0)
        pair_rows.append(
            {
                "pair": (fa.name, fb.name),
                "date_obs_a": str(ha.get("DATE-OBS")),
                "date_obs_b": str(hb.get("DATE-OBS")),
                "exptime": float(ha.get("EXPTIME", 0)),
                "mean_level_a": float(np.mean(da)),
                "mean_level_b": float(np.mean(db)),
                "g1_per_region": reg_g,
                "g1_median": float(np.median(reg_g)) if reg_g else float("nan"),
            }
        )
    rn_adu = float(np.std(bias) * np.median(g1_all)) if g1_all else float("nan")
    slope = intercept = r2 = float("nan")
    if len(s_all) >= 5:
        lr = stats.linregress(s_all, var_all)
        slope, intercept, r2 = float(lr.slope), float(lr.intercept), float(lr.rvalue ** 2)
    return {
        "flat_source": str(FLAT_DIR),
        "bias_source": str(bias_path),
        "pairs": pair_rows,
        "g1_median_all_regions": float(np.median(g1_all)) if g1_all else float("nan"),
        "g1_iqr": [
            float(np.quantile(g1_all, 0.25)),
            float(np.quantile(g1_all, 0.5)),
            float(np.quantile(g1_all, 0.75)),
        ]
        if g1_all
        else [float("nan")] * 3,
        "rn_e_bin1": rn_adu,
        "var_vs_signal_slope": slope,
        "var_vs_signal_intercept": intercept,
        "var_vs_signal_r2": r2,
    }


def m2() -> dict[str, Any]:
    dark_path = CAL_LIB / "Dark_60s_Dark_0G_-10deg_Bin1_20260422.fits"
    dark1 = fits.getdata(dark_path, mmap=True).astype(np.float64)
    lights = sorted(RAW_LIGHTS.glob("BO_CVn_Light_*.fits"))
    means: list[float] = []
    vars_: list[float] = []
    fracs: list[float] = []
    dates: list[str] = []
    for lp in lights:
        img = fits.getdata(lp, mmap=True).astype(np.float64)
        dark2, _ = resample_master_to_light_binning(
            dark1, master_binning=1, light_binning=2, kind="dark",
        )
        corr = img - dark2
        for y0, x0 in _regions(corr.shape, n=3, size=200):
            patch = corr[y0 : y0 + 200, x0 : x0 + 200]
            m, v, f = _clip_region(patch)
            if m > 0 and math.isfinite(v):
                means.append(m)
                vars_.append(v)
                fracs.append(f)
        dates.append(str(fits.getheader(lp).get("DATE-OBS")))
    m_arr = np.asarray(means, dtype=np.float64)
    v_arr = np.asarray(vars_, dtype=np.float64)
    ok = np.isfinite(m_arr) & np.isfinite(v_arr) & (m_arr > 0) & (v_arr > 0)
    m_arr = m_arr[ok]
    v_arr = v_arr[ok]
    if m_arr.size < 20:
        return {"error": "insufficient points", "n": int(m_arr.size)}
    lr = stats.linregress(m_arr, v_arr)
    slope = float(lr.slope)
    intercept = float(lr.intercept)
    g_eff = 1.0 / slope if slope > 0 else float("nan")
    rn_eff_adu = math.sqrt(max(0.0, intercept / slope)) if slope > 0 else float("nan")
    # CI on slope via bootstrap
    boots: list[float] = []
    rng = np.random.default_rng(42)
    idx = np.arange(m_arr.size)
    for _ in range(500):
        j = rng.choice(idx, size=idx.size, replace=True)
        b = stats.linregress(m_arr[j], v_arr[j]).slope
        if b > 0:
            boots.append(1.0 / b)
    ci = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))] if boots else [float("nan")] * 2
    return {
        "raw_lights_path": str(RAW_LIGHTS),
        "match_note": "Snapshot Raw/lights/NoFilter_60_2 BO_CVn_Light_*.fits; DATE-OBS 2026-04-23 matches draft night",
        "n_frames": len(lights),
        "n_points": int(m_arr.size),
        "mean_frac_kept": float(np.median(fracs)),
        "sky_mean_p05_p50_p95": [float(x) for x in np.quantile(m_arr, [0.05, 0.5, 0.95])],
        "g_eff": g_eff,
        "g_eff_ci95": ci,
        "rn_eff_adu": rn_eff_adu,
        "fit_slope": slope,
        "fit_intercept": intercept,
        "fit_r2": float(lr.rvalue ** 2),
        "fit_pvalue": float(lr.pvalue),
    }


def _mad_sigma(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float("nan")
    med = float(np.median(v))
    return float(MAD_SCALE * np.median(np.abs(v - med)))


_MAG_ERR = 2.5 / math.log(10)


def _load_proc_index(proc_dir: Path) -> dict[str, pd.DataFrame]:
    from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

    out: dict[str, pd.DataFrame] = {}
    for p in sorted(proc_dir.glob("proc_*.csv")):
        df = pd.read_csv(p, low_memory=False, dtype={"catalog_id": str})
        id_col = "catalog_id" if "catalog_id" in df.columns else "name"
        df["_nid"] = df[id_col].map(lambda x: str(normalize_gaia_source_id(x) or "").strip())
        out[p.name] = df
    return out


def _proc_row(proc_index: dict[str, pd.DataFrame], source_file: str, catalog_id: str) -> pd.Series | None:
    from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

    key = Path(str(source_file).strip()).name
    df = proc_index.get(key)
    if df is None:
        return None
    cid = str(normalize_gaia_source_id(catalog_id) or "").strip()
    sub = df.loc[df["_nid"] == cid]
    return None if sub.empty else sub.iloc[0]


def _target_err_decompose(
    lc_path: Path,
    proc_index: dict[str, pd.DataFrame],
    target_cid: str,
    *,
    gain: float,
    read_noise: float,
) -> dict[str, Any] | None:
    from photometry_core import SIGMA_BKG_AP_COL, _howell_bkg_variance_adu2, _sky_pp_for_photometric_error  # noqa: PLC0415

    lc = pd.read_csv(lc_path, low_memory=False)
    phot_rel: list[float] = []
    ens_rel: list[float] = []
    err_rel: list[float] = []
    mags: list[float] = []
    for _, row in lc.iterrows():
        sf = str(row.get("source_file", "")).strip()
        err_lc = float(pd.to_numeric(row.get("err"), errors="coerce"))
        mag = float(pd.to_numeric(row.get("mag_calib_final"), errors="coerce"))
        proc_row = _proc_row(proc_index, sf, target_cid)
        if proc_row is None or not (math.isfinite(err_lc) and err_lc > 0 and math.isfinite(mag)):
            continue
        flux = float(pd.to_numeric(proc_row.get("dao_flux"), errors="coerce"))
        if not (math.isfinite(flux) and flux > 0):
            continue
        sky = _sky_pp_for_photometric_error(proc_row)
        area = float(pd.to_numeric(proc_row.get("aperture_area_px"), errors="coerce"))
        if not (math.isfinite(area) and area > 0):
            r_ap = float(pd.to_numeric(proc_row.get("aperture_r_px"), errors="coerce"))
            area = math.pi * r_ap * r_ap if math.isfinite(r_ap) and r_ap > 0 else float("nan")
        sig_bkg = float(pd.to_numeric(proc_row.get(SIGMA_BKG_AP_COL), errors="coerce"))
        g = gain if gain > 0 else 1.0
        var_photon = flux / g
        if math.isfinite(sig_bkg):
            var_bkg = sig_bkg * sig_bkg
        else:
            var_bkg = _howell_bkg_variance_adu2(sky, area, gain=g, read_noise=read_noise)
        var_pt = var_photon + var_bkg
        if not (math.isfinite(var_pt) and var_pt > 0):
            continue
        ep = math.sqrt(var_pt) / flux
        ee = max(0.0, err_lc * err_lc - ep * ep)
        er = math.sqrt(ee)
        phot_rel.append(ep)
        ens_rel.append(er)
        err_rel.append(err_lc)
        mags.append(mag)
    if len(mags) < 10:
        return None
    pr = np.asarray(phot_rel, dtype=np.float64)
    er = np.asarray(ens_rel, dtype=np.float64)
    return {
        "mags": np.asarray(mags, dtype=np.float64),
        "phot_rel_med": float(np.median(pr)),
        "ens_rel_med": float(np.median(er)),
        "err_rel_med": float(np.median(err_rel)),
    }


def m4(gain_used: float, g_eff: float, cfg: AppConfig, *, read_noise: float) -> dict[str, Any]:
    from check_star_kmag import resolve_proc_csv_dir  # noqa: PLC0415

    ps = DRAFT / "platesolve" / "NoFilter_60_2"
    phot = ps / "photometry"
    proc_dir = resolve_proc_csv_dir(phot, "NoFilter_60_2")
    proc_index = _load_proc_index(proc_dir) if proc_dir else {}
    ms = pd.read_csv(ps / "masterstars_full_match.csv", dtype={"catalog_id": str})

    factor = math.sqrt(gain_used / g_eff) if g_eff > 0 else float("nan")
    stars: list[dict[str, Any]] = []
    for lc_path in sorted((phot / "lightcurves").glob("lightcurve_*.csv")):
        target_cid = lc_path.stem.replace("lightcurve_", "").split("_")[0]
        if target_cid == CHECK_CID:
            continue
        dec = _target_err_decompose(
            lc_path, proc_index, target_cid, gain=gain_used, read_noise=read_noise,
        )
        if dec is None:
            continue
        if not (dec["ens_rel_med"] < 0.5 * dec["phot_rel_med"]):
            continue
        gmag = float("nan")
        row = ms.loc[ms["catalog_id"].astype(str).str.strip() == target_cid]
        if not row.empty:
            gmag = float(pd.to_numeric(row["phot_g_mean_mag"].iloc[0], errors="coerce"))
        sig = _mad_sigma(dec["mags"])
        em = dec["err_rel_med"]
        ec = math.sqrt((dec["phot_rel_med"] * factor) ** 2 + dec["ens_rel_med"] ** 2)
        stars.append(
            {
                "catalog_id": target_cid,
                "mag_g": gmag,
                "ratio_orig": sig / em if em > 0 else float("nan"),
                "ratio_corr": sig / ec if ec > 0 else float("nan"),
                "phot_mmag": dec["phot_rel_med"] * MAG_ERR_SCALE,
                "ens_mmag": dec["ens_rel_med"] * MAG_ERR_SCALE,
            }
        )
    stars.sort(key=lambda r: r.get("mag_g", -999), reverse=True)
    faint5 = stars[:5]

    ck_ratios = {"ratio_orig": float("nan"), "ratio_corr": float("nan"), "catalog_id": CHECK_CID}
    ck_path = next(W1W2_LC.glob(f"*/lightcurve_{CHECK_CID}.csv"), None)
    if ck_path:
        lc = pd.read_csv(ck_path, low_memory=False)
        m = pd.to_numeric(lc["mag_calib_final"], errors="coerce").to_numpy(dtype=np.float64)
        e = pd.to_numeric(lc["err"], errors="coerce").to_numpy(dtype=np.float64)
        ep = pd.to_numeric(lc["err_photon"], errors="coerce").to_numpy(dtype=np.float64)
        es = pd.to_numeric(lc["err_sem_rel"], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
        sig = _mad_sigma(m[ok])
        em = float(np.median(e[ok]))
        other = np.maximum(0.0, e[ok] ** 2 - ep[ok] ** 2 - es[ok] ** 2)
        e_corr = np.sqrt((ep[ok] * factor) ** 2 + es[ok] ** 2 + other)
        ec = float(np.median(e_corr))
        ck_ratios = {
            "ratio_orig": sig / em,
            "ratio_corr": sig / ec,
            "catalog_id": CHECK_CID,
            "median_ep_mmag": float(np.median(ep[ok]) * MAG_ERR_SCALE),
            "median_es_mmag": float(np.median(es[ok]) * MAG_ERR_SCALE),
        }
    return {
        "gain_used": gain_used,
        "g_eff": g_eff,
        "correction_factor_sqrt_gain_ratio": factor,
        "n_photon_dominated_candidates": len(stars),
        "photon_dominated_faint5": faint5,
        "faint5_median_ratio_orig": float(np.median([s["ratio_orig"] for s in faint5])) if faint5 else float("nan"),
        "faint5_median_ratio_corr": float(np.median([s["ratio_corr"] for s in faint5])) if faint5 else float("nan"),
        "check_star": ck_ratios,
    }


def main() -> int:
    cfg = AppConfig()
    OUT.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"M0": m0(cfg), "M1": m1(), "M2": m2()}
    gain_used = float(out["M0"]["resolved_gain"])
    g_eff = float(out["M2"].get("g_eff", float("nan")))
    g1 = float(out["M1"].get("g1_median_all_regions", float("nan")))
    out["M3"] = {
        "g1_bin1_median": g1,
        "g_eff_bin2": g_eff,
        "ratio_g_eff_over_g1": (g_eff / g1 if g1 > 0 else float("nan")),
        "gain_used": gain_used,
        "gain_used_over_g_eff": (gain_used / g_eff if g_eff > 0 else float("nan")),
        "implied_photon_factor_sqrt": math.sqrt(gain_used / g_eff) if g_eff > 0 else float("nan"),
    }
    out["M4"] = m4(gain_used, g_eff, cfg, read_noise=float(out["M0"]["resolved_rn"]))
    (OUT / "wide_err_a2b.json").write_text(json.dumps(out, indent=2) + "\n", encoding="ascii")
    print(json.dumps({k: out[k] for k in ("M0", "M1", "M2", "M3", "M4")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
