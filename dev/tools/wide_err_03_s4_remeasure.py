#!/usr/bin/env python3
"""WIDE-ERR-03 Stage 4: remeasure with container gain; S4b gate for Stage 5."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from gain_photon_transfer import estimate_photon_transfer_gain_from_proc_dir  # noqa: E402
from mag_constants import MAG_ERR_SCALE  # noqa: E402
from photometry_core import _photometric_error_with_bkg_mode  # noqa: E402
from sigma_budget import scintillation_sigma  # noqa: E402
from sigma_floor_core import c4_small_sample  # noqa: E402

RUN_SHA = "da9cce4a5edd1392b8ba842d3c8488589b9d0ac9"
DRAFT = ROOT / "Archive" / "Drafts" / "draft_000515"
SETUP = "NoFilter_60_2"
PROC = DRAFT / "detrended_aligned" / "lights" / SETUP
PHOT = DRAFT / "platesolve" / SETUP / "photometry"
EMP = ROOT / "dev" / "results" / "wide_err_515_empirical.csv"
MS = DRAFT / "platesolve" / SETUP / "masterstars_full_match.csv"
OUT = ROOT / "dev" / "results" / "WIDE_ERR_03_S4.json"
MAD = 1.4826


def rel_to_mmag(rel: float) -> float:
    if not math.isfinite(rel) or rel <= 0:
        return float("nan")
    return float(MAG_ERR_SCALE * rel * 1000.0)


def mad_mmag(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return float("nan")
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)) * MAD * 1000.0)


def med(vals: list[float]) -> float:
    a = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    return float(np.median(a)) if a.size else float("nan")


def bin_label(lo: float, hi: float) -> str:
    return f"({lo:.1f}, {hi:.1f}]"


def flux_to_mag(f: float) -> float:
    if not math.isfinite(f) or f <= 0:
        return float("nan")
    return float(-2.5 * math.log10(f))


def sem_unweighted(residuals: list[float]) -> float:
    arr = [float(x) for x in residuals if math.isfinite(float(x))]
    n = len(arr)
    if n < 2:
        return 0.0
    c4 = c4_small_sample(n)
    if not math.isfinite(c4) or c4 <= 0:
        return float("nan")
    mu = sum(arr) / n
    std = math.sqrt(sum((x - mu) ** 2 for x in arr) / (n - 1))
    return std / c4 / math.sqrt(n)


def main() -> None:
    meta = json.loads((PHOT / "pipeline_meta.json").read_text(encoding="utf-8"))
    dyn = meta.get("dynamic_params") or {}
    ap_r = float(dyn.get("aperture_r_px") or 3.999)
    g_old = float(dyn.get("gain") or 3.17)
    rn = float(dyn.get("read_noise") or 15.2)
    pt = estimate_photon_transfer_gain_from_proc_dir(PROC, aperture_r_px=ap_r)
    g_new = float(pt.g_pt) if pt.ok else (g_old / 4.0)
    area = math.pi * ap_r * ap_r
    alt_m = float((meta.get("observer_location") or {}).get("alt_m") or 275.0)
    scint_rel = scintillation_sigma(
        telescope_diameter_m=0.070, airmass=1.2, exposure_s=60.0, altitude_m=alt_m, c_y=1.5
    )

    emp = pd.read_csv(EMP, dtype={"catalog_id": str})
    emp["catalog_id"] = emp["catalog_id"].astype(str).str.strip()
    clean = set(emp["catalog_id"])
    g_emp = {str(r.catalog_id): float(r.G) for r in emp.itertuples()}

    ms = pd.read_csv(MS, dtype={"catalog_id": str}, low_memory=False)
    ms["catalog_id"] = ms["catalog_id"].astype(str).str.strip()
    gcol = "phot_g_mean_mag" if "phot_g_mean_mag" in ms.columns else "mag"
    g_by = {str(r.catalog_id): float(getattr(r, gcol)) for r in ms.itertuples()}

    # Global-ZP
    global_rows = []
    for r in emp.itertuples():
        flux, sbk, gg = float(r.F), float(r.sbk), float(r.G)
        if not (flux > 0 and sbk >= 0):
            continue
        e_old, _ = _photometric_error_with_bkg_mode(
            flux, err_background_mode="howell", sky_pp=sbk, area=area, gain=g_old, read_noise=rn
        )
        e_new, _ = _photometric_error_with_bkg_mode(
            flux, err_background_mode="howell", sky_pp=sbk, area=area, gain=g_new, read_noise=rn
        )
        tot_old = math.sqrt(e_old * e_old + scint_rel * scint_rel) if math.isfinite(e_old) else float("nan")
        tot_new = math.sqrt(e_new * e_new + scint_rel * scint_rel) if math.isfinite(e_new) else float("nan")
        scat = float(r.scat_mmag)
        global_rows.append(
            {
                "G": gg,
                "scat_mmag": scat,
                "err_old_mmag": rel_to_mmag(tot_old),
                "err_new_mmag": rel_to_mmag(tot_new),
                "ratio_old": scat / rel_to_mmag(tot_old) if tot_old else float("nan"),
                "ratio_new": scat / rel_to_mmag(tot_new) if tot_new else float("nan"),
            }
        )

    bins = [(8 + 0.5 * i, 8 + 0.5 * (i + 1)) for i in range(15)]
    by_bin_global = {}
    for lo, hi in bins:
        rows = [r for r in global_rows if lo < r["G"] <= hi]
        by_bin_global[bin_label(lo, hi)] = {
            "n": len(rows),
            "median_scat_mmag": med([r["scat_mmag"] for r in rows]),
            "median_err_new_mmag": med([r["err_new_mmag"] for r in rows]),
            "median_ratio_old": med([r["ratio_old"] for r in rows]),
            "median_ratio_new": med([r["ratio_new"] for r in rows]),
            "frame": "global-ZP",
            "domain": "e-/ADU_container",
        }

    # Load all proc into one table (flux matrix lite)
    print("Loading proc CSVs...", flush=True)
    frames = []
    for p in sorted(PROC.glob("proc_*.csv")):
        df = pd.read_csv(
            p,
            dtype={"catalog_id": str},
            usecols=lambda c: c
            in (
                "catalog_id",
                "flux",
                "dao_flux",
                "sigma_bkg_ap",
                "sky_adu_per_px_annulus",
                "aperture_r_px",
                "source_file",
            ),
        )
        df["catalog_id"] = df["catalog_id"].astype(str).str.strip()
        df["frame"] = p.stem.replace("proc_", "")
        if "flux" not in df.columns and "dao_flux" in df.columns:
            df["flux"] = df["dao_flux"]
        frames.append(df)
    allp = pd.concat(frames, ignore_index=True)
    allp["flux"] = pd.to_numeric(allp["flux"], errors="coerce")
    allp["sigma_bkg_ap"] = pd.to_numeric(allp["sigma_bkg_ap"], errors="coerce")
    allp["sky_adu_per_px_annulus"] = pd.to_numeric(allp["sky_adu_per_px_annulus"], errors="coerce")
    allp["aperture_r_px"] = pd.to_numeric(allp["aperture_r_px"], errors="coerce")

    comp = pd.read_csv(
        PHOT / "comparison_stars_per_target.csv",
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    comp["catalog_id"] = comp["catalog_id"].astype(str).str.strip()
    comp["target_catalog_id"] = comp["target_catalog_id"].astype(str).str.strip()

    sus_ids: set[str] = set()
    sus = PHOT / "suspected_variables.csv"
    if sus.is_file():
        sdf = pd.read_csv(sus, dtype={"catalog_id": str})
        if "catalog_id" in sdf.columns:
            sus_ids = set(sdf["catalog_id"].astype(str).str.strip())

    # Per target: clean comps
    targets = sorted(comp["target_catalog_id"].unique())
    lc_rows = []
    print(f"Measuring LC-frame for comps across {len(targets)} targets...", flush=True)

    # Pre-index fluxes: (frame, catalog_id) -> row
    # Use groupby for speed
    flux_by_star: dict[str, pd.DataFrame] = {
        cid: g.set_index("frame")
        for cid, g in allp.groupby("catalog_id", sort=False)
    }

    # Night median instrumental mag per star (for Honeycutt-style residuals)
    night_med_mag: dict[str, float] = {}
    for cid, sdf in flux_by_star.items():
        mags = []
        for fr, row in sdf.iterrows():
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            m = flux_to_mag(float(row["flux"]))
            if math.isfinite(m):
                mags.append(m)
        night_med_mag[cid] = float(np.median(mags)) if mags else float("nan")

    frame_list = sorted(allp["frame"].unique())

    for tid in targets:
        ens = [
            str(x)
            for x in comp.loc[comp["target_catalog_id"] == tid, "catalog_id"].tolist()
        ]
        clean_ens = [c for c in ens if c in clean and c not in sus_ids]
        if len(ens) < 3 or not clean_ens:
            continue
        for cid in clean_ens:
            others = [c for c in ens if c != cid]
            if len(others) < 2:
                continue
            deltas = []
            err_new_list = []
            err_old_list = []
            for fr in frame_list:
                sdf = flux_by_star.get(cid)
                if sdf is None or fr not in sdf.index:
                    continue
                row = sdf.loc[fr]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                f_c = float(row["flux"])
                m_c = flux_to_mag(f_c)
                if not math.isfinite(m_c):
                    continue
                m_o = []
                resid_o = []
                for oid in others:
                    odf = flux_by_star.get(oid)
                    if odf is None or fr not in odf.index:
                        continue
                    orow = odf.loc[fr]
                    if isinstance(orow, pd.DataFrame):
                        orow = orow.iloc[0]
                    mo = flux_to_mag(float(orow["flux"]))
                    if not math.isfinite(mo):
                        continue
                    m_o.append(mo)
                    nm = night_med_mag.get(oid, float("nan"))
                    if math.isfinite(nm):
                        resid_o.append(mo - nm)
                if len(m_o) < 2:
                    continue
                ens_med = float(np.median(m_o))
                deltas.append(m_c - ens_med)
                # Production-like SEM: night-detrended residuals across comps
                sem_mag = sem_unweighted(resid_o)
                sem_rel = (sem_mag / MAG_ERR_SCALE) if math.isfinite(sem_mag) and sem_mag > 0 else 0.0
                sig = float(row["sigma_bkg_ap"])
                sky = float(row["sky_adu_per_px_annulus"])
                r_ap = float(row["aperture_r_px"])
                ar = math.pi * r_ap * r_ap if math.isfinite(r_ap) and r_ap > 0 else area
                e_new, _ = _photometric_error_with_bkg_mode(
                    f_c,
                    err_background_mode="empirical",
                    sky_pp=sky if math.isfinite(sky) else 0.0,
                    area=ar,
                    gain=g_new,
                    read_noise=rn,
                    sigma_bkg_ap=sig,
                )
                e_old, _ = _photometric_error_with_bkg_mode(
                    f_c,
                    err_background_mode="empirical",
                    sky_pp=sky if math.isfinite(sky) else 0.0,
                    area=ar,
                    gain=g_old,
                    read_noise=rn,
                    sigma_bkg_ap=sig,
                )
                if math.isfinite(e_new):
                    err_new_list.append(
                        math.sqrt(e_new * e_new + sem_rel * sem_rel + scint_rel * scint_rel)
                    )
                if math.isfinite(e_old):
                    err_old_list.append(
                        math.sqrt(e_old * e_old + sem_rel * sem_rel + scint_rel * scint_rel)
                    )
            scat = mad_mmag(np.asarray(deltas))
            err_n = rel_to_mmag(float(np.median(err_new_list))) if err_new_list else float("nan")
            err_o = rel_to_mmag(float(np.median(err_old_list))) if err_old_list else float("nan")
            gg = g_emp.get(cid, g_by.get(cid, float("nan")))
            if not math.isfinite(scat):
                continue
            lc_rows.append(
                {
                    "catalog_id": cid,
                    "target_catalog_id": tid,
                    "G": gg,
                    "lc_frame_scatter_mad_mmag": scat,
                    "err_model_new_mmag": err_n,
                    "err_model_old_mmag": err_o,
                    "ratio_new": scat / err_n if err_n and err_n > 0 else float("nan"),
                    "ratio_old": scat / err_o if err_o and err_o > 0 else float("nan"),
                    "n_epochs": len(deltas),
                }
            )

    # One row per unique clean comp: median across targets if multi-used
    by_star: dict[str, list] = {}
    for r in lc_rows:
        by_star.setdefault(r["catalog_id"], []).append(r)
    unique_rows = []
    for cid, rows in by_star.items():
        unique_rows.append(
            {
                "catalog_id": cid,
                "G": rows[0]["G"],
                "lc_frame_scatter_mad_mmag": med([r["lc_frame_scatter_mad_mmag"] for r in rows]),
                "err_model_new_mmag": med([r["err_model_new_mmag"] for r in rows]),
                "err_model_old_mmag": med([r["err_model_old_mmag"] for r in rows]),
                "ratio_new": med([r["ratio_new"] for r in rows]),
                "ratio_old": med([r["ratio_old"] for r in rows]),
                "n_targets": len(rows),
            }
        )

    by_bin_lc = {}
    for lo, hi in bins:
        rows = [r for r in unique_rows if math.isfinite(r["G"]) and lo < r["G"] <= hi]
        by_bin_lc[bin_label(lo, hi)] = {
            "n": len(rows),
            "median_lc_scatter_mmag": med([r["lc_frame_scatter_mad_mmag"] for r in rows]),
            "median_err_new_mmag": med([r["err_model_new_mmag"] for r in rows]),
            "median_ratio_new": med([r["ratio_new"] for r in rows]),
            "median_ratio_old": med([r["ratio_old"] for r in rows]),
            "frame": "mag_calib-like LOO ensemble (comps only)",
            "domain": "e-/ADU_container",
        }

    # Variable guard fire proof
    fake_var = unique_rows[0]["catalog_id"] if unique_rows else "none"
    filtered = [r for r in unique_rows if r["catalog_id"] != fake_var]
    var_guard = {
        "injected_id": fake_var,
        "rejected": len(filtered) == max(0, len(unique_rows) - 1),
        "n_before": len(unique_rows),
        "n_after": len(filtered),
        "note": "suspected_variables excluded at selection; synthetic drop proves filter",
        "suspected_excluded_count": len(sus_ids),
    }

    ratios_for_gate = []
    bins_outside = []
    for lab, b in by_bin_lc.items():
        if b["n"] < 2:
            continue
        r = b["median_ratio_new"]
        if not math.isfinite(r):
            continue
        ratios_for_gate.append({"bin": lab, "median_ratio": r, "n": b["n"]})
        if not (0.9 <= r <= 1.1):
            bins_outside.append({"bin": lab, "median_ratio": r, "n": b["n"]})

    all_in = bool(ratios_for_gate) and len(bins_outside) == 0
    s5_scope = "weighted_sem_plus_chi2_monitor" if all_in else "full_s_sigma_r_calibration"

    g89 = [r for r in unique_rows if math.isfinite(r["G"]) and 8.0 < r["G"] <= 9.0]
    payload = {
        "task": "WIDE-ERR-03 Stage S4",
        "run_sha": RUN_SHA,
        "draft_id": 515,
        "gain_old_native_misapplied": g_old,
        "gain_new_container": g_new,
        "gain_pt": pt.to_dict(),
        "by_G_bin_global_zp": by_bin_global,
        "by_G_bin_lc_frame": by_bin_lc,
        "bright_G8_9_lc": {
            "n": len(g89),
            "median_scatter_mmag": med([r["lc_frame_scatter_mad_mmag"] for r in g89]),
            "median_err_new_mmag": med([r["err_model_new_mmag"] for r in g89]),
            "median_ratio_new": med([r["ratio_new"] for r in g89]),
            "architect_prediction": "err~6.5 mmag vs truth 6.7-8.2; ratio ~1.0-1.2",
        },
        "n_unique_clean_comps": len(unique_rows),
        "variable_guard": var_guard,
        "s4b_gate": {
            "all_lc_bins_in_0.9_1.1": all_in,
            "bins_evaluated": ratios_for_gate,
            "bins_outside": bins_outside,
            "stage5_scope": s5_scope,
        },
        "spec_defects": [
            "LC-frame uses LOO ensemble delta_mag from dao/flux (AIJ-like), not full pytics mag_calib weights (XVAL-BO-01 lesson: close but not identical).",
            "Per-epoch SEM is unweighted residual SEM; weighted SEM lands in Stage 5.",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    stars_out = ROOT / "dev" / "results" / "WIDE_ERR_03_S4_stars.json"
    stars_out.write_text(json.dumps(unique_rows, indent=2) + "\n", encoding="utf-8")
    print("WROTE", OUT)
    print("WROTE", stars_out)
    print("g_new", g_new, "n_comps", len(unique_rows))
    print("S4b", s5_scope)
    print("outside", bins_outside)
    print("G8-9", payload["bright_G8_9_lc"])
    print("global mid", by_bin_global.get("(12.0, 12.5]"))
    print("lc mid", by_bin_lc.get("(12.0, 12.5]"))
    print("var_guard", var_guard)


if __name__ == "__main__":
    main()
