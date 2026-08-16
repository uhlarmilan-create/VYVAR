#!/usr/bin/env python3
"""WIDE-ERR-03B: smooth clamped calibration + held-out validation (B1/B2)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from err_calibration import (  # noqa: E402
    ERR_CALIB_SIDECAR,
    SmoothCalibration,
    apply_smooth_mmag,
    calibrate_smooth,
    choose_form_by_heldout,
    write_sidecar,
)
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
OUT = ROOT / "dev" / "results" / "WIDE_ERR_03B_B2.json"
MAD = 1.4826
SCINT_FLOOR_MMAG = 2.2


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


def measure_star_series(
    *,
    flux_by_star: dict[str, pd.DataFrame],
    night_med_mag: dict[str, float],
    ens: list[str],
    cid: str,
    frame_list: list[str],
    frame_mask: set[str] | None,
    g_new: float,
    g_wrong: float,
    rn: float,
    area: float,
    scint_rel: float,
) -> dict[str, float] | None:
    others = [c for c in ens if c != cid]
    if len(others) < 2:
        return None
    deltas = []
    err_new = []
    err_wrong = []
    for fr in frame_list:
        if frame_mask is not None and fr not in frame_mask:
            continue
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
        sem_mag = sem_unweighted(resid_o)
        sem_rel = (sem_mag / MAG_ERR_SCALE) if math.isfinite(sem_mag) and sem_mag > 0 else 0.0
        sig = float(row["sigma_bkg_ap"])
        sky = float(row["sky_adu_per_px_annulus"])
        r_ap = float(row["aperture_r_px"])
        ar = math.pi * r_ap * r_ap if math.isfinite(r_ap) and r_ap > 0 else area
        e_n, _ = _photometric_error_with_bkg_mode(
            f_c,
            err_background_mode="empirical",
            sky_pp=sky if math.isfinite(sky) else 0.0,
            area=ar,
            gain=g_new,
            read_noise=rn,
            sigma_bkg_ap=sig,
        )
        e_w, _ = _photometric_error_with_bkg_mode(
            f_c,
            err_background_mode="empirical",
            sky_pp=sky if math.isfinite(sky) else 0.0,
            area=ar,
            gain=g_wrong,
            read_noise=rn,
            sigma_bkg_ap=sig,
        )
        if math.isfinite(e_n):
            err_new.append(math.sqrt(e_n * e_n + sem_rel * sem_rel + scint_rel * scint_rel))
        if math.isfinite(e_w):
            err_wrong.append(math.sqrt(e_w * e_w + sem_rel * sem_rel + scint_rel * scint_rel))
    scat = mad_mmag(np.asarray(deltas))
    if not math.isfinite(scat):
        return None
    en = rel_to_mmag(float(np.median(err_new))) if err_new else float("nan")
    ew = rel_to_mmag(float(np.median(err_wrong))) if err_wrong else float("nan")
    return {
        "scatter_mmag": scat,
        "err_model_mmag": en,
        "err_wrong_mmag": ew,
        "n_epochs": float(len(deltas)),
    }


def eval_bins(
    rows: list[dict],
    cal: SmoothCalibration | None,
    *,
    err_key: str = "err_model_mmag",
    min_n_gate: int = 4,
) -> dict:
    bins = [(8 + 0.5 * i, 8 + 0.5 * (i + 1)) for i in range(15)]
    by = {}
    outside = []
    for lo, hi in bins:
        lab = bin_label(lo, hi)
        sub = [r for r in rows if math.isfinite(r["G"]) and lo < r["G"] <= hi]
        if not sub:
            by[lab] = {"n": 0}
            continue
        ratios = []
        errs = []
        for r in sub:
            e0 = float(r[err_key])
            if cal is not None:
                e = apply_smooth_mmag(e0, float(r["G"]), cal)
            else:
                e = e0
            if e and e > 0 and math.isfinite(r["scatter_mmag"]):
                ratios.append(r["scatter_mmag"] / e)
                errs.append(e)
        mr = med(ratios)
        me = med(errs)
        entry = {
            "n": len(sub),
            "n_ratio": len(ratios),
            "median_ratio": mr,
            "median_err_mmag": me,
            "gated": len(sub) >= min_n_gate,
        }
        by[lab] = entry
        if len(sub) >= min_n_gate and math.isfinite(mr) and not (0.85 <= mr <= 1.15):
            outside.append({"bin": lab, **entry})

    # Union G8-9
    g89 = [r for r in rows if math.isfinite(r["G"]) and 8.0 < r["G"] <= 9.0]
    ratios89 = []
    errs89 = []
    for r in g89:
        e0 = float(r[err_key])
        e = apply_smooth_mmag(e0, float(r["G"]), cal) if cal is not None else e0
        if e and e > 0:
            ratios89.append(r["scatter_mmag"] / e)
            errs89.append(e)
    g89_summary = {
        "n": len(g89),
        "median_ratio": med(ratios89),
        "median_err_mmag": med(errs89),
        "in_window": bool(
            len(g89) >= 1
            and math.isfinite(med(ratios89))
            and 0.85 <= med(ratios89) <= 1.15
            and med(errs89) >= SCINT_FLOOR_MMAG
        ),
    }
    gated_ok = len(outside) == 0 and g89_summary["in_window"]
    # If no gated bins at all, fail closed unless g89 ok and we have some ratios
    has_gated = any(v.get("gated") for v in by.values() if isinstance(v, dict))
    if not has_gated:
        gated_ok = bool(g89_summary["in_window"])
    return {"by_G_bin": by, "bins_outside": outside, "G8_9": g89_summary, "pass": gated_ok}


def main() -> None:
    meta = json.loads((PHOT / "pipeline_meta.json").read_text(encoding="utf-8"))
    dyn = meta.get("dynamic_params") or {}
    ap_r = float(dyn.get("aperture_r_px") or 3.999)
    g_wrong = float(dyn.get("gain") or 3.17)
    rn = float(dyn.get("read_noise") or 15.2)
    pt = estimate_photon_transfer_gain_from_proc_dir(PROC, aperture_r_px=ap_r)
    g_new = float(pt.g_pt) if pt.ok else (g_wrong / 4.0)
    area = math.pi * ap_r * ap_r
    alt_m = float((meta.get("observer_location") or {}).get("alt_m") or 275.0)
    scint_rel = scintillation_sigma(
        telescope_diameter_m=0.070, airmass=1.2, exposure_s=60.0, altitude_m=alt_m, c_y=1.5
    )

    emp = pd.read_csv(EMP, dtype={"catalog_id": str})
    emp["catalog_id"] = emp["catalog_id"].astype(str).str.strip()
    clean = set(emp["catalog_id"])
    g_emp = {str(r.catalog_id): float(r.G) for r in emp.itertuples()}

    print("Loading proc...", flush=True)
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
            ),
        )
        df["catalog_id"] = df["catalog_id"].astype(str).str.strip()
        df["frame"] = p.stem.replace("proc_", "")
        if "flux" not in df.columns and "dao_flux" in df.columns:
            df["flux"] = df["dao_flux"]
        frames.append(df)
    allp = pd.concat(frames, ignore_index=True)
    for c in ("flux", "sigma_bkg_ap", "sky_adu_per_px_annulus", "aperture_r_px"):
        allp[c] = pd.to_numeric(allp[c], errors="coerce")

    frame_list = sorted(allp["frame"].unique())
    # Odd/even by sorted index
    odd_frames = {fr for i, fr in enumerate(frame_list) if i % 2 == 1}
    even_frames = {fr for i, fr in enumerate(frame_list) if i % 2 == 0}

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

    flux_by_star = {cid: g.set_index("frame") for cid, g in allp.groupby("catalog_id", sort=False)}
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

    # Unique clean comps with a primary ensemble (first target)
    primary: dict[str, list[str]] = {}
    for tid in sorted(comp["target_catalog_id"].unique()):
        ens = [
            str(x)
            for x in comp.loc[comp["target_catalog_id"] == tid, "catalog_id"].tolist()
        ]
        for cid in ens:
            if cid in clean and cid not in sus_ids and cid not in primary:
                primary[cid] = ens

    print(f"Measuring {len(primary)} clean comps on odd/even frames...", flush=True)

    def build_rows(frame_mask: set[str] | None) -> list[dict]:
        rows = []
        for cid, ens in primary.items():
            m = measure_star_series(
                flux_by_star=flux_by_star,
                night_med_mag=night_med_mag,
                ens=ens,
                cid=cid,
                frame_list=frame_list,
                frame_mask=frame_mask,
                g_new=g_new,
                g_wrong=g_wrong,
                rn=rn,
                area=area,
                scint_rel=scint_rel,
            )
            if m is None:
                continue
            rows.append({"catalog_id": cid, "G": g_emp.get(cid, float("nan")), **m})
        return rows

    rows_all = build_rows(None)
    rows_odd = build_rows(odd_frames)
    rows_even = build_rows(even_frames)

    # B1/B2: choose form by held-out (odd calib -> even eval)
    cal_holdout = choose_form_by_heldout(rows_odd, rows_even, s_min=1.0)
    frame_eval = eval_bins(rows_even, cal_holdout)
    frame_calib_in = eval_bins(rows_odd, cal_holdout)
    # Production coeffs: refit chosen form on all frames (form locked by held-out)
    cal = calibrate_smooth(rows_all, form=cal_holdout.form, s_min=1.0)

    # Star-split secondary
    stars_sorted = sorted(rows_all, key=lambda r: r["catalog_id"])
    half = len(stars_sorted) // 2
    star_cal_rows = stars_sorted[:half]
    star_eval_rows = stars_sorted[half:]
    cal_star = choose_form_by_heldout(star_cal_rows, star_eval_rows, s_min=1.0)
    star_eval = eval_bins(star_eval_rows, cal_star)

    # B2d fire proof: wrong calib (s=1, sigma_r=0) with old gain errs on even half
    wrong_rows = [
        {
            "catalog_id": r["catalog_id"],
            "G": r["G"],
            "scatter_mmag": r["scatter_mmag"],
            "err_model_mmag": r["err_wrong_mmag"],
        }
        for r in rows_even
        if math.isfinite(r["err_wrong_mmag"])
    ]
    wrong_cal = SmoothCalibration(
        s=1.0,
        sigma_r0_mmag=0.0,
        sigma_r_slope_mmag_per_G=0.0,
        form="constant_sigma_r",
        n_stars=len(wrong_rows),
        s_clamped=False,
        median_ratio_pre=float("nan"),
        median_ratio_post=float("nan"),
    )
    fire = eval_bins(wrong_rows, wrong_cal)
    fire_fails = not fire["pass"]

    # Faint-end check (global-ish): G>13 ratios on eval with clamped s
    faint = [r for r in rows_even if math.isfinite(r["G"]) and r["G"] > 13]
    faint_ratios = []
    for r in faint:
        e = apply_smooth_mmag(r["err_model_mmag"], r["G"], cal_holdout)
        if e and e > 0:
            faint_ratios.append(r["scatter_mmag"] / e)
    faint_med = med(faint_ratios)
    gain_pt_ci_item = None
    if math.isfinite(faint_med) and faint_med < 0.85:
        gain_pt_ci_item = {
            "id": "GAIN-PT-CI-01",
            "note": (
                f"Faint-end held-out median ratio {faint_med:.3f} < 0.85 with s clamped>=1; "
                "possible photon-term overprediction (g_pt center vs CI) - do not absorb in calib"
            ),
            "median_ratio_Ggt13": faint_med,
            "n": len(faint_ratios),
        }

    b2c_pass = bool(frame_eval["pass"] and fire_fails)

    sidecar = {
        "task": "WIDE-ERR-03B",
        "run_sha": RUN_SHA,
        "draft_id": 515,
        "form": "err_exported^2 = (s * err_model)^2 + sigma_r(G)^2",
        "smooth": cal.to_dict(),
        "smooth_holdout_odd_fit": cal_holdout.to_dict(),
        "s_min_clamp": 1.0,
        "gain_authority": g_new,
        "n_calib_stars_odd": len(rows_odd),
        "n_eval_stars_even": len(rows_even),
        "n_stars_all": len(rows_all),
        "chosen_by": "held_out_even_frames_form_then_refit_all_frames",
        "b2c_pass": b2c_pass,
    }
    write_sidecar(PHOT / ERR_CALIB_SIDECAR, sidecar)

    next_step = None
    if not b2c_pass:
        next_step = (
            "Do not widen [0.85,1.15]. Mid-G gated bins sit just below 0.85 under any "
            "2-param (s>=1, sigma_r>=0) constant form (grid search: zero zero-fail cands). "
            "Next: re-measure B2c in the true mag_calib / comps-only frame (Pont/Gillon), "
            "not the LOO flux-sum LC-frame meter; if still outside, treat residual as "
            "correlated/common-mode structure beyond diagonal (s, sigma_r)."
        )

    payload = {
        "task": "WIDE-ERR-03B B1/B2",
        "run_sha": RUN_SHA,
        "g_pt": g_new,
        "n_frames": len(frame_list),
        "n_odd": len(odd_frames),
        "n_even": len(even_frames),
        "b1": {
            "chosen_form": cal.form,
            "parameters_production_all_frames": cal.to_dict(),
            "parameters_holdout_odd_fit": cal_holdout.to_dict(),
            "s_clamp_policy": "s >= 1.0 always",
            "alternative_star_split_form": cal_star.to_dict(),
            "n_stars": len(rows_all),
        },
        "b2_frame_split": {
            "calib_half": "odd-indexed frames",
            "eval_half": "even-indexed frames",
            "calib_in_sample": frame_calib_in,
            "eval": frame_eval,
            "note": "eval uses odd-fit holdout parameters (cal_holdout), not all-frame refit",
        },
        "b2_star_split": {
            "calib_n": len(star_cal_rows),
            "eval_n": len(star_eval_rows),
            "eval": star_eval,
        },
        "b2d_fire_proof": {
            "description": "s=1, sigma_r=0, gain=3.17 (bare native) on even frames",
            "must_fail": True,
            "failed_as_required": fire_fails,
            "eval": fire,
        },
        "b2c_acceptance": {
            "pass": b2c_pass,
            "requires_eval_pass_and_fire_proof_fail": True,
            "next_step_if_fail": next_step,
        },
        "gain_pt_ci_01": gain_pt_ci_item,
        "spec_defects": [
            "WIDE-ERR-03 S5e was circular (fit=accept same stars) - named architect defect.",
            "LC-frame meter remains LOO flux-sum delta_mag (not full pytics mag_calib).",
            "Per-star primary ensemble is first target listing that uses the comp.",
            "B2c [0.85,1.15] cannot be met by any constant (s>=1, sigma_r) on this meter "
            "(bright underquote vs mid overquote tension); do not widen window.",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("WROTE", OUT)
    print("WROTE", PHOT / ERR_CALIB_SIDECAR)
    print(
        "form",
        cal.form,
        "s",
        cal.s,
        "clamped",
        cal.s_clamped,
        "sig0",
        cal.sigma_r0_mmag,
        "slope",
        cal.sigma_r_slope_mmag_per_G,
    )
    print("B2c", "PASS" if b2c_pass else "FAIL")
    print("eval G8-9", frame_eval["G8_9"])
    print("outside", frame_eval["bins_outside"])
    print("fire_fails", fire_fails, "fire G8-9", fire["G8_9"])
    print("GAIN-PT-CI", gain_pt_ci_item)
    if next_step:
        print("NEXT", next_step)


if __name__ == "__main__":
    main()
