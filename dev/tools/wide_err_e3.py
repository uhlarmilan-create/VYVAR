#!/usr/bin/env python3
"""WIDE-ERR E3: missing per-star noise term on target (read-only)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))
sys.path.insert(0, str(REPO / "dev" / "scripts"))
import _bootstrap  # noqa: E402,F401

from config import AppConfig  # noqa: E402
from photometry_core import (  # noqa: E402
    SIGMA_BKG_AP_COL,
    _group_comp_mag_inst_from_proc_csvs,
    _howell_bkg_variance_adu2,
    _sky_pp_for_photometric_error,
    check_comparison_stability,
    temporal_bin_comp_lc,
)
from sigma_floor_core import ensemble_sem_mag_from_residuals, resolve_sigma_sys_mag  # noqa: E402

# reuse field builder from e2
sys.path.insert(0, str(REPO / "dev" / "tools"))
from wide_err_e2 import _field_comp_lc, _iqr  # noqa: E402


def _mad_sigma_arr(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float("nan")
    med = float(np.median(v))
    return float(MAD_SCALE * np.median(np.abs(v - med)))

SETUP = "NoFilter_60_2"
DRAFT = REPO / "Archive" / "Drafts" / "draft_000435_snapshot_skysurface_20260716"
CHECK_CID = "1499906247391001088"
CHECK_G = 8.74
OUT = REPO / "tmp" / "wide_err_e3"
W1W2_JSON = REPO / "tmp" / "wide_err_w1w2" / "wide_err_w1w2.json"
MAG_ERR_SCALE = 1000.0
MAD_SCALE = 1.4826
GAIN = 3.17
READ_NOISE = 15.2


def _load_proc_index(proc_dir: Path) -> dict[str, pd.DataFrame]:
    from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

    out: dict[str, pd.DataFrame] = {}
    for p in sorted(proc_dir.glob("proc_*.csv")):
        df = pd.read_csv(p, low_memory=False, dtype={"catalog_id": str})
        id_col = "catalog_id" if "catalog_id" in df.columns else "name"
        df["_nid"] = df[id_col].map(lambda x: str(normalize_gaia_source_id(x) or "").strip())
        out[p.name] = df
    return out


def _photon_rel_median(
    proc_index: dict[str, pd.DataFrame],
    cid: str,
    csv_files: list[Path],
    comp_mag: np.ndarray,
) -> float:
    """Median photon err (rel flux) from proc rows aligned to comp mag epochs."""
    vals: list[float] = []
    for i, cp in enumerate(csv_files):
        if i >= len(comp_mag) or not math.isfinite(float(comp_mag[i])):
            continue
        key = cp.name
        df = proc_index.get(key)
        if df is None:
            continue
        sub = df.loc[df["_nid"] == cid]
        if sub.empty:
            continue
        row = sub.iloc[0]
        flux = float(pd.to_numeric(row.get("dao_flux"), errors="coerce"))
        if not (math.isfinite(flux) and flux > 0):
            continue
        sky = _sky_pp_for_photometric_error(row)
        area = float(pd.to_numeric(row.get("aperture_area_px"), errors="coerce"))
        if not (math.isfinite(area) and area > 0):
            r_ap = float(pd.to_numeric(row.get("aperture_r_px"), errors="coerce"))
            area = math.pi * r_ap * r_ap if math.isfinite(r_ap) and r_ap > 0 else float("nan")
        sig_bkg = float(pd.to_numeric(row.get(SIGMA_BKG_AP_COL), errors="coerce"))
        var_p = flux / GAIN
        if math.isfinite(sig_bkg):
            var_b = sig_bkg * sig_bkg
        else:
            var_b = _howell_bkg_variance_adu2(sky, area, gain=GAIN, read_noise=READ_NOISE)
        var_pt = var_p + var_b
        if var_pt > 0:
            vals.append(math.sqrt(var_pt) / flux)
    return float(np.median(vals)) if vals else float("nan")


def _comp_sigma_eps_excess(
    comp_mag: np.ndarray,
    comp_ref: float,
    phot_rel: float,
) -> float:
    """Robust scatter of comp-ref time series, excess over photon, in mmag."""
    m = np.asarray(comp_mag, dtype=np.float64)
    ok = np.isfinite(m)
    if int(np.count_nonzero(ok)) < 5:
        return float("nan")
    resid = m[ok] - comp_ref
    sig = MAD_SCALE * float(np.median(np.abs(resid - np.median(resid))))
    sig_mmag = sig * MAG_ERR_SCALE
    phot_mmag = phot_rel * MAG_ERR_SCALE if math.isfinite(phot_rel) else float("nan")
    if not math.isfinite(phot_mmag):
        return float("nan")
    return float(math.sqrt(max(0.0, sig_mmag * sig_mmag - phot_mmag * phot_mmag)))


def _field_variance_decompose(field: dict[str, Any], csv_files: list[Path]) -> dict[str, float]:
    resid = field["resid_mat"]
    n_comp, n_frames = resid.shape
    frame_means: list[float] = []
    frame_within: list[float] = []
    sem_quoted: list[float] = []
    good_ids = field["good_ids"]
    comp_lc = field["comp_lc"]
    comp_ref = field["comp_ref"]

    for i in range(n_frames):
        col = resid[:, i]
        ok = np.isfinite(col)
        if int(np.count_nonzero(ok)) < 2:
            continue
        v = col[ok]
        frame_means.append(float(np.mean(v)))
        frame_within.append(float(np.std(v, ddof=1)))
        pairs = [(cid, float(comp_lc[cid][i])) for cid in good_ids if cid in comp_ref and math.isfinite(comp_lc[cid][i])]
        if len(pairs) >= 2:
            cr = [m - comp_ref[cid] for cid, m in pairs]
            sem_quoted.append(float(ensemble_sem_mag_from_residuals(cr)))

    sigma_c = float(np.std(frame_means, ddof=1)) if len(frame_means) >= 3 else float("nan")
    sigma_eps = float(np.median(frame_within)) if frame_within else float("nan")
    sigma_eps_mmag = sigma_eps * MAG_ERR_SCALE if math.isfinite(sigma_eps) else float("nan")
    sem_med_mmag = float(np.median(sem_quoted)) * MAG_ERR_SCALE if sem_quoted else float("nan")
    n = field["n_comp"]
    expected_sigma_eps = sem_med_mmag * math.sqrt(n) if math.isfinite(sem_med_mmag) and n > 0 else float("nan")
    return {
        "sigma_c_mmag": sigma_c * MAG_ERR_SCALE if math.isfinite(sigma_c) else float("nan"),
        "sigma_eps_mmag": sigma_eps_mmag,
        "sem_quoted_median_mmag": sem_med_mmag,
        "expected_sigma_eps_from_sem_mmag": expected_sigma_eps,
        "n_comp": n,
    }


def main() -> int:
    cfg = AppConfig()
    OUT.mkdir(parents=True, exist_ok=True)
    ps = DRAFT / "platesolve" / SETUP
    phot = ps / "photometry"
    lights = DRAFT / "detrended_aligned" / "lights" / SETUP
    proc_dir = lights
    proc_index = _load_proc_index(proc_dir)
    comp_all = pd.read_csv(
        phot / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    ms = pd.read_csv(ps / "masterstars_full_match.csv", dtype={"catalog_id": str})
    gmap = {
        str(r["catalog_id"]).strip(): float(pd.to_numeric(r.get("phot_g_mean_mag"), errors="coerce"))
        for _, r in ms.iterrows()
    }
    csv_files = sorted(lights.glob("proc_*.csv"))
    all_frames_stub = pd.DataFrame({"catalog_id": [], "bjd": []})

    target_ids = sorted(
        t
        for t in {str(x).strip() for x in comp_all["target_catalog_id"].astype(str)}
        if t and (phot / "lightcurves" / f"check_kmag_{t}.csv").is_file()
    )

    e31_rows: list[dict[str, float]] = []
    matched_eps: list[float] = []
    all_eps: list[float] = []
    g_bins: list[dict[str, Any]] = []

    for t in target_ids:
        field = _field_comp_lc(t, comp_all, csv_files, cfg, all_frames_stub)
        if field is None:
            continue
        e31_rows.append(_field_variance_decompose(field, csv_files))

        for cid in field["good_ids"]:
            g = gmap.get(cid, float("nan"))
            arr = field["comp_lc"].get(cid)
            ref = field["comp_ref"].get(cid, float("nan"))
            if arr is None or not math.isfinite(ref):
                continue
            phot_rel = _photon_rel_median(proc_index, cid, csv_files, arr)
            eps = _comp_sigma_eps_excess(arr, ref, phot_rel)
            if not math.isfinite(eps):
                continue
            all_eps.append(eps)
            g_bins.append({"g": g, "sigma_eps_mmag": eps, "field": t})
            if math.isfinite(g) and abs(g - CHECK_G) <= 0.3:
                matched_eps.append(eps)

    widen_note = "+/-0.3 mag"
    if len(matched_eps) < 10:
        matched_eps = []
        widen_note = "+/-0.5 mag (widened from 0.3)"
        for row in g_bins:
            g = row["g"]
            if math.isfinite(g) and abs(g - CHECK_G) <= 0.5:
                matched_eps.append(float(row["sigma_eps_mmag"]))

    # E3.3 trace
    sigma_sys_eq1 = resolve_sigma_sys_mag(1, cfg, rig_label=SETUP)
    sigma_sys_eq4 = resolve_sigma_sys_mag(4, cfg, rig_label=SETUP)

    from sigma_budget import resolve_rig_scintillation_params, scintillation_sigma  # noqa: PLC0415

    rig_scint = resolve_rig_scintillation_params(draft_id=435, setup=SETUP, cfg=cfg, pipeline_meta=None)
    sigma_eps_use = float(np.median(matched_eps)) if matched_eps else float("nan")
    eps_rel = sigma_eps_use / MAG_ERR_SCALE if math.isfinite(sigma_eps_use) else float("nan")

    pop_rows: list[dict[str, float]] = []
    for lc_path in sorted((phot / "lightcurves").glob("lightcurve_*.csv")):
        target_cid = lc_path.stem.replace("lightcurve_", "").split("_")[0]
        if target_cid == CHECK_CID:
            continue
        lc = pd.read_csv(lc_path, low_memory=False)
        phot_r: list[float] = []
        sem_r: list[float] = []
        scint_r: list[float] = []
        err_r: list[float] = []
        mags: list[float] = []
        for _, row in lc.iterrows():
            sf = str(row.get("source_file", "")).strip()
            e = float(pd.to_numeric(row.get("err"), errors="coerce"))
            m = float(pd.to_numeric(row.get("mag_calib_final"), errors="coerce"))
            if not (math.isfinite(e) and e > 0 and math.isfinite(m)):
                continue
            key = Path(sf).name
            df = proc_index.get(key)
            if df is None:
                continue
            sub = df.loc[df["_nid"] == target_cid]
            if sub.empty:
                continue
            pr = sub.iloc[0]
            flux = float(pd.to_numeric(pr.get("dao_flux"), errors="coerce"))
            if not (math.isfinite(flux) and flux > 0):
                continue
            sky = _sky_pp_for_photometric_error(pr)
            area = float(pd.to_numeric(pr.get("aperture_area_px"), errors="coerce"))
            if not (math.isfinite(area) and area > 0):
                r_ap = float(pd.to_numeric(pr.get("aperture_r_px"), errors="coerce"))
                area = math.pi * r_ap * r_ap if math.isfinite(r_ap) and r_ap > 0 else float("nan")
            sig_bkg = float(pd.to_numeric(pr.get(SIGMA_BKG_AP_COL), errors="coerce"))
            var_p = flux / GAIN
            var_b = sig_bkg * sig_bkg if math.isfinite(sig_bkg) else _howell_bkg_variance_adu2(sky, area, gain=GAIN, read_noise=READ_NOISE)
            var_pt = var_p + var_b
            if var_pt <= 0:
                continue
            ep = math.sqrt(var_pt) / flux
            am = float(pd.to_numeric(row.get("airmass"), errors="coerce"))
            sc_m = (
                scintillation_sigma(
                    telescope_diameter_m=rig_scint.telescope_diameter_m,
                    airmass=am,
                    exposure_s=rig_scint.exposure_s,
                    altitude_m=rig_scint.altitude_m,
                    c_y=rig_scint.c_y,
                )
                if math.isfinite(am) and am >= 1.0
                else 0.0
            )
            sc = sc_m / MAG_ERR_SCALE
            e = float(pd.to_numeric(row.get("err"), errors="coerce"))
            ens = math.sqrt(max(0.0, e * e - ep * ep - sc * sc))
            phot_r.append(ep)
            sem_r.append(ens)
            scint_r.append(sc)
            err_r.append(e)
            mags.append(m)
        if len(mags) < 10:
            continue
        mo = np.asarray(mags, dtype=np.float64)
        eo = np.asarray(err_r, dtype=np.float64)
        ok = np.isfinite(mo) & np.isfinite(eo) & (eo > 0)
        sig = _mad_sigma_arr(mo[ok])
        em = float(np.median(eo[ok]))
        meas_ratio = sig / em if em > 0 else float("nan")
        pr = np.asarray(phot_r, dtype=np.float64)
        sr = np.asarray(sem_r, dtype=np.float64)
        sc = np.asarray(scint_r, dtype=np.float64)
        base = np.sqrt(pr * pr + sr * sr + sc * sc)
        pred = np.sqrt(pr * pr + sr * sr + sc * sc + eps_rel * eps_rel)
        pred_ratio = float(np.median(pred[base > 0] / base[base > 0])) if np.any(base > 0) else float("nan")
        med_ep = float(np.median(pr))
        med_es = float(np.median(sr))
        gmag = gmap.get(target_cid, float("nan"))
        pop_rows.append(
            {
                "target_cid": target_cid,
                "mag_g": gmag,
                "meas_ratio": meas_ratio,
                "pred_ratio": pred_ratio,
                "phot_dominated": med_es < 0.5 * med_ep,
            }
        )

    df31 = pd.DataFrame(e31_rows)
    df34 = pd.DataFrame(pop_rows)
    phot_sub = df34.loc[df34["phot_dominated"]]
    ens_sub = df34.loc[~df34["phot_dominated"]]

    def _spearman(a: pd.Series, b: pd.Series) -> dict[str, float]:
        ok = a.notna() & b.notna()
        if int(ok.sum()) < 5:
            return {"rho": float("nan"), "p": float("nan"), "n": int(ok.sum())}
        r, p = stats.spearmanr(a[ok], b[ok])
        return {"rho": float(r), "p": float(p), "n": int(ok.sum())}

    e34_spear = _spearman(df34["pred_ratio"], df34["meas_ratio"])

    # G bins for sigma_eps
    gbin_rows: list[dict[str, Any]] = []
    if g_bins:
        gdf = pd.DataFrame(g_bins)
        gdf = gdf.loc[gdf["g"].notna()]
        if not gdf.empty:
            gdf["gbin"] = pd.cut(gdf["g"], bins=[8, 10, 11, 12, 13, 14, 16])
            for b, sub in gdf.groupby("gbin", observed=True):
                if sub.empty:
                    continue
                gbin_rows.append(
                    {
                        "g_bin": str(b),
                        "n": int(len(sub)),
                        "sigma_eps_median_mmag": float(sub["sigma_eps_mmag"].median()),
                    }
                )

    out = {
        "E3_0": {
            "correction": "E2.1 rho_bar is across-frame comp correlation (common mode c(t)); not within-frame SEM divisor",
            "e2_2_check_pc1": "check_pc1_corr ~ 0 confirms c(t) cancelled target vs ensemble",
            "withdraw_causal_claim": "WIDE-ERR-CORRELATED-COMPS withdrawn; 1.90 vs 1.83 numerical agreement unexplained",
            "keep_fact": "rho_bar median 0.393 measured",
        },
        "E3_1": {
            "n_fields": len(df31),
            "sigma_c_mmag_median": float(df31["sigma_c_mmag"].median()),
            "sigma_c_iqr": _iqr(df31["sigma_c_mmag"].to_numpy(dtype=np.float64)),
            "sigma_eps_mmag_median": float(df31["sigma_eps_mmag"].median()),
            "sigma_eps_iqr": _iqr(df31["sigma_eps_mmag"].to_numpy(dtype=np.float64)),
            "sem_quoted_median_mmag": float(df31["sem_quoted_median_mmag"].median()),
            "expected_sigma_eps_from_sem_mmag": float(df31["expected_sigma_eps_from_sem_mmag"].median()),
            "budget_note": "SEM ~ sigma_eps/sqrt(n) implies sigma_eps ~ 8.88*sqrt(8) ~ 25 mmag",
        },
        "E3_2": {
            "check_g": CHECK_G,
            "required_sigma_eps_mmag": 15.1,
            "match_window": widen_note,
            "n_brightness_matched_comps": len(matched_eps),
            "matched_sigma_eps_mmag_median": float(np.median(matched_eps)) if matched_eps else float("nan"),
            "matched_iqr": _iqr(np.asarray(matched_eps, dtype=np.float64)) if matched_eps else [float("nan")] * 3,
            "all_comps_sigma_eps_median_mmag": float(np.median(all_eps)) if all_eps else float("nan"),
            "all_comps_iqr": _iqr(np.asarray(all_eps, dtype=np.float64)) if all_eps else [float("nan")] * 3,
            "g_bins": gbin_rows,
            "constrained": len(matched_eps) >= 5,
        },
        "E3_3": {
            "err_model": "err_photon^2 + sem_rel^2 + scint_rel^2 + sigma_sys_rel^2 (photometry_core.py:3550)",
            "terms": [
                {"term": "err_photon", "category": "a target photon", "file": "photometry_core.py:9833-9885"},
                {"term": "err_sem_rel / ensemble_scatter", "category": "b ensemble ZP", "file": "photometry_core.py:3438,9889"},
                {"term": "err_scint_rel", "category": "c atmosphere", "file": "photometry_core.py:9866-9878"},
                {"term": "err_sigma_sys_rel / sigma_sys_mag", "category": "d per-star systematic target", "file": "sigma_floor_core.py:64-86,129-157"},
            ],
            "sigma_sys_mag_equipment_id_1": sigma_sys_eq1,
            "sigma_sys_mag_equipment_id_4": sigma_sys_eq4,
            "category_d_nonzero_this_run": bool(sigma_sys_eq1 > 0),
        },
        "E3_4": {
            "sigma_eps_used_mmag": sigma_eps_use,
            "n_stars": len(df34),
            "pred_vs_meas_spearman": e34_spear,
            "phot_dominated_n": int(len(phot_sub)),
            "phot_dominated_pred_median": float(phot_sub["pred_ratio"].median()) if len(phot_sub) else float("nan"),
            "phot_dominated_meas_median": float(phot_sub["meas_ratio"].median()) if len(phot_sub) else float("nan"),
            "ensemble_dominated_n": int(len(ens_sub)),
            "ensemble_dominated_pred_median": float(ens_sub["pred_ratio"].median()) if len(ens_sub) else float("nan"),
            "ensemble_dominated_meas_median": float(ens_sub["meas_ratio"].median()) if len(ens_sub) else float("nan"),
        },
    }
    (OUT / "wide_err_e3.json").write_text(json.dumps(out, indent=2) + "\n", encoding="ascii")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
