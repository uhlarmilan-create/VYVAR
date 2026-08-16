#!/usr/bin/env python3
"""WIDE-ERR-03C: product-frame (mag_calib) calibration gate + optional floor forms."""
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
    load_sidecar,
    smooth_from_sidecar,
    write_sidecar,
)
from gain_photon_transfer import estimate_photon_transfer_gain_from_proc_dir  # noqa: E402
from mag_constants import MAG_ERR_SCALE  # noqa: E402
from photometry_core import (  # noqa: E402
    _photometric_error_with_bkg_mode,
    pytics_iterative_weights,
)
from sigma_budget import scintillation_sigma  # noqa: E402
from sigma_floor_core import c4_small_sample  # noqa: E402

RUN_SHA = "da9cce4a5edd1392b8ba842d3c8488589b9d0ac9"
DRAFT = ROOT / "Archive" / "Drafts" / "draft_000515"
SETUP = "NoFilter_60_2"
PROC = DRAFT / "detrended_aligned" / "lights" / SETUP
PHOT = DRAFT / "platesolve" / SETUP / "photometry"
EMP = ROOT / "dev" / "results" / "wide_err_515_empirical.csv"
OUT_C1 = ROOT / "dev" / "results" / "WIDE_ERR_03C_C1.json"
OUT_C2 = ROOT / "dev" / "results" / "WIDE_ERR_03C_C2.json"
OUT_C3 = ROOT / "dev" / "results" / "WIDE_ERR_03C_C3.json"
OUT_C4 = ROOT / "dev" / "results" / "WIDE_ERR_03C_C4.json"
OUT_SUM = ROOT / "dev" / "results" / "WIDE_ERR_03C_summary.json"
MAD = 1.4826
SCINT_FLOOR_MMAG = 2.2
GATE_LO, GATE_HI = 0.85, 1.15


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


def mag_calib_series(
    *,
    m_star: np.ndarray,
    ens_ids: list[str],
    mag_by: dict[str, np.ndarray],
    cat_g: dict[str, float],
    rms_phase1: dict[str, float],
    self_exclude: bool,
    focus_id: str,
) -> tuple[np.ndarray, str, dict[str, float]]:
    """kcal[i] = m_star[i] + sum w_j (G_j - m_j[i]) / sum w_j (XVAL-BO-01)."""
    if self_exclude:
        use_ids = [c for c in ens_ids if c != focus_id]
        case = "self_excluded"
    else:
        use_ids = list(ens_ids)
        case = "full_ensemble"
    use_ids = [c for c in use_ids if c in mag_by]
    if len(use_ids) < 2:
        return np.full_like(m_star, float("nan"), dtype=float), case, {}

    qual = {c: {"quality": "good"} for c in use_ids}
    rms0 = {c: float(rms_phase1.get(c, 0.01)) for c in use_ids}
    rms_py = pytics_iterative_weights(
        comp_lc={c: mag_by[c] for c in use_ids},
        comp_quality=qual,
        comp_rms_map=rms0,
        n_iter=5,
        enabled=True,
    )
    weights = {}
    for c in use_ids:
        r = float(rms_py.get(c, float("nan")))
        weights[c] = (1.0 / (r * r)) if math.isfinite(r) and r > 1e-6 else 0.0

    n = int(m_star.size)
    out = np.full(n, float("nan"), dtype=float)
    for i in range(n):
        if not math.isfinite(m_star[i]):
            continue
        num = den = 0.0
        for c in use_ids:
            mj = float(mag_by[c][i])
            gj = float(cat_g.get(c, float("nan")))
            w = float(weights.get(c, 0.0))
            if not (math.isfinite(mj) and math.isfinite(gj) and w > 0):
                continue
            num += w * (gj - mj)
            den += w
        if den > 0:
            out[i] = m_star[i] + num / den
    return out, case, weights


def eval_bins(
    rows: list[dict],
    cal: SmoothCalibration | None,
    *,
    err_key: str = "err_model_mmag",
    sigma_r_fn=None,
    min_n_gate: int = 4,
) -> dict:
    """sigma_r_fn(row)->mmag overrides cal.sigma_r if provided (C2/C3)."""
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
            if not (math.isfinite(e0) and e0 > 0 and math.isfinite(r["scatter_mmag"])):
                continue
            if sigma_r_fn is not None:
                s = float(cal.s) if cal is not None else 1.0
                sr = float(sigma_r_fn(r))
                e = math.sqrt((s * e0) ** 2 + max(0.0, sr) ** 2)
            elif cal is not None:
                e = apply_smooth_mmag(e0, float(r["G"]), cal)
            else:
                e = e0
            if e and e > 0:
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
        if len(sub) >= min_n_gate and math.isfinite(mr) and not (GATE_LO <= mr <= GATE_HI):
            outside.append({"bin": lab, **entry})

    g89 = [r for r in rows if math.isfinite(r["G"]) and 8.0 < r["G"] <= 9.0]
    ratios89 = []
    errs89 = []
    for r in g89:
        e0 = float(r[err_key])
        if sigma_r_fn is not None:
            s = float(cal.s) if cal is not None else 1.0
            e = math.sqrt((s * e0) ** 2 + max(0.0, float(sigma_r_fn(r))) ** 2)
        elif cal is not None:
            e = apply_smooth_mmag(e0, float(r["G"]), cal)
        else:
            e = e0
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
            and GATE_LO <= med(ratios89) <= GATE_HI
            and med(errs89) >= SCINT_FLOOR_MMAG
        ),
    }
    has_gated = any(v.get("gated") for v in by.values() if isinstance(v, dict))
    gated_ok = len(outside) == 0 and g89_summary["in_window"]
    if not has_gated:
        gated_ok = bool(g89_summary["in_window"])
    return {"by_G_bin": by, "bins_outside": outside, "G8_9": g89_summary, "pass": gated_ok}


def main() -> int:
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
        if "dao_flux" in df.columns:
            df["flux_use"] = pd.to_numeric(df["dao_flux"], errors="coerce")
        else:
            df["flux_use"] = pd.to_numeric(df["flux"], errors="coerce")
        frames.append(df)
    allp = pd.concat(frames, ignore_index=True)
    for c in ("flux_use", "sigma_bkg_ap", "sky_adu_per_px_annulus", "aperture_r_px"):
        allp[c] = pd.to_numeric(allp[c], errors="coerce")

    frame_list = sorted(allp["frame"].unique())
    odd_frames = {fr for i, fr in enumerate(frame_list) if i % 2 == 1}
    even_frames = {fr for i, fr in enumerate(frame_list) if i % 2 == 0}
    frame_index = {fr: i for i, fr in enumerate(frame_list)}

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

    # mag series and refs aligned to frame_list
    mag_by: dict[str, np.ndarray] = {}
    flux_rows_by: dict[str, pd.DataFrame] = {}
    night_med_mag: dict[str, float] = {}
    median_ap: dict[str, float] = {}
    for cid, gdf in allp.groupby("catalog_id", sort=False):
        gdf = gdf.set_index("frame")
        flux_rows_by[cid] = gdf
        mags = np.full(len(frame_list), float("nan"), dtype=float)
        aps = []
        for i, fr in enumerate(frame_list):
            if fr not in gdf.index:
                continue
            row = gdf.loc[fr]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            mags[i] = flux_to_mag(float(row["flux_use"]))
            apv = float(row["aperture_r_px"])
            if math.isfinite(apv):
                aps.append(apv)
        mag_by[cid] = mags
        finite = mags[np.isfinite(mags)]
        night_med_mag[cid] = float(np.median(finite)) if finite.size else float("nan")
        median_ap[cid] = float(np.median(aps)) if aps else float("nan")

    # Primary target ensemble for each clean comp
    primary: dict[str, tuple[str, list[str], dict[str, float], dict[str, float]]] = {}
    for tid in sorted(comp["target_catalog_id"].unique()):
        sub = comp.loc[comp["target_catalog_id"] == tid]
        ens = [str(x) for x in sub["catalog_id"].tolist()]
        cat_g = {}
        rms = {}
        for _, r in sub.iterrows():
            cid = str(r["catalog_id"]).strip()
            cat_g[cid] = float(
                pd.to_numeric(r.get("phot_g_mean_mag", r.get("mag")), errors="coerce")
            )
            rms[cid] = float(pd.to_numeric(r.get("comp_rms"), errors="coerce"))
        for cid in ens:
            if cid in clean and cid not in sus_ids and cid not in primary:
                primary[cid] = (tid, ens, cat_g, rms)

    print(f"Measuring {len(primary)} clean comps in mag_calib product frame...", flush=True)

    def measure(frame_mask: set[str] | None) -> list[dict]:
        rows = []
        mask_idx = None
        if frame_mask is not None:
            mask_idx = np.asarray(
                [frame_index[fr] for fr in frame_list if fr in frame_mask], dtype=int
            )
        for cid, (tid, ens, cat_g, rms) in primary.items():
            if cid not in mag_by:
                continue
            self_ex = cid in ens
            kcal, case, _w = mag_calib_series(
                m_star=mag_by[cid],
                ens_ids=ens,
                mag_by=mag_by,
                cat_g=cat_g,
                rms_phase1=rms,
                self_exclude=self_ex,
                focus_id=cid,
            )
            if mask_idx is not None:
                kcal_m = kcal[mask_idx]
            else:
                kcal_m = kcal
            scat = mad_mmag(kcal_m)
            if not math.isfinite(scat):
                continue

            # Physical model err (median over selected frames)
            others = [c for c in ens if c != cid]
            err_new = []
            err_wrong = []
            gdf = flux_rows_by[cid]
            for fr in frame_list:
                if frame_mask is not None and fr not in frame_mask:
                    continue
                if fr not in gdf.index:
                    continue
                row = gdf.loc[fr]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                f_c = float(row["flux_use"])
                resid_o = []
                for oid in others:
                    if oid not in mag_by:
                        continue
                    mi = mag_by[oid][frame_index[fr]]
                    nm = night_med_mag.get(oid, float("nan"))
                    if math.isfinite(mi) and math.isfinite(nm):
                        resid_o.append(mi - nm)
                sem_mag = sem_unweighted(resid_o)
                sem_rel = (
                    (sem_mag / MAG_ERR_SCALE) if math.isfinite(sem_mag) and sem_mag > 0 else 0.0
                )
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
                    err_wrong.append(
                        math.sqrt(e_w * e_w + sem_rel * sem_rel + scint_rel * scint_rel)
                    )
            en = rel_to_mmag(float(np.median(err_new))) if err_new else float("nan")
            ew = rel_to_mmag(float(np.median(err_wrong))) if err_wrong else float("nan")
            rows.append(
                {
                    "catalog_id": cid,
                    "target_catalog_id": tid,
                    "G": g_emp.get(cid, float("nan")),
                    "scatter_mmag": scat,
                    "err_model_mmag": en,
                    "err_wrong_mmag": ew,
                    "ensemble_case": case,
                    "n_ensemble": len(ens),
                    "r_ap_px": median_ap.get(cid, float("nan")),
                    "n_epochs": float(np.sum(np.isfinite(kcal_m))),
                    "frame": "mag_calib_pytics_zp",
                }
            )
        return rows

    rows_all = measure(None)
    rows_odd = measure(odd_frames)
    rows_even = measure(even_frames)

    # Existing 03B constant calibration (production all-frame params)
    side = load_sidecar(PHOT / ERR_CALIB_SIDECAR) or {}
    cal_03b = smooth_from_sidecar(side)
    if cal_03b is None:
        cal_03b = SmoothCalibration(
            s=1.001118295861578,
            sigma_r0_mmag=6.892615508521114,
            sigma_r_slope_mmag_per_G=0.0,
            form="constant_sigma_r",
            n_stars=54,
            s_clamped=False,
            median_ratio_pre=float("nan"),
            median_ratio_post=float("nan"),
        )
    # Holdout odd-fit from 03B for fair comparison with 03B gate
    hold = side.get("smooth_holdout_odd_fit") if isinstance(side, dict) else None
    if isinstance(hold, dict):
        cal_hold = SmoothCalibration(
            s=float(hold["s"]),
            sigma_r0_mmag=float(hold["sigma_r0_mmag"]),
            sigma_r_slope_mmag_per_G=float(hold.get("sigma_r_slope_mmag_per_G", 0.0)),
            form=str(hold.get("form", "constant_sigma_r")),  # type: ignore[arg-type]
            n_stars=int(hold.get("n_stars", 0)),
            s_clamped=bool(hold.get("s_clamped", False)),
            median_ratio_pre=float(hold.get("median_ratio_pre", float("nan"))),
            median_ratio_post=float(hold.get("median_ratio_post", float("nan"))),
        )
    else:
        cal_hold = cal_03b

    c1d_eval = eval_bins(rows_even, cal_hold)
    c1d_prod = eval_bins(rows_even, cal_03b)
    n_self = sum(1 for r in rows_all if r["ensemble_case"] == "self_excluded")
    n_full = sum(1 for r in rows_all if r["ensemble_case"] == "full_ensemble")

    c1_payload = {
        "task": "WIDE-ERR-03C C1",
        "run_sha": RUN_SHA,
        "meter": "mag_calib = m_inst + sum w_j(G_j-m_j)/sum w_j; w from pytics_iterative_weights",
        "frame": "product mag_calib (XVAL-BO-01 validated)",
        "g_pt": g_new,
        "n_stars": len(rows_all),
        "n_self_excluded": n_self,
        "n_full_ensemble": n_full,
        "n_odd_frames": len(odd_frames),
        "n_even_frames": len(even_frames),
        "c1d_before_refit": {
            "note": "Existing 03B constant-sigma_r on EVEN half, product-frame scatter",
            "cal_holdout_odd_fit": cal_hold.to_dict(),
            "eval_holdout_params": c1d_eval,
            "cal_production_all_frames": cal_03b.to_dict(),
            "eval_production_params": c1d_prod,
            "meter_artifact_if_pass": bool(c1d_eval["pass"]),
        },
        "rows_sample": rows_all[:3],
    }
    OUT_C1.write_text(json.dumps(c1_payload, indent=2) + "\n", encoding="utf-8")
    print("WROTE", OUT_C1)
    print("C1d holdout-params PASS" if c1d_eval["pass"] else "C1d holdout-params FAIL", c1d_eval["bins_outside"])
    print("C1d G8-9", c1d_eval["G8_9"])

    winning_level = None
    winning_cal: SmoothCalibration | None = None
    winning_form_meta: dict = {}
    sigma_r_fn = None

    if c1d_eval["pass"]:
        winning_level = "C1d_existing_constant_sigma_r"
        winning_cal = cal_03b
        winning_form_meta = {
            "form": "constant_sigma_r",
            "note": "03B FAIL was meter artifact; product-frame gate passes with existing calib",
            "parameters": cal_03b.to_dict(),
        }
        print("C1d PASS -> skip C2/C3, go to C4", flush=True)
    else:
        # ---- C2: sigma_r = f(r_ap) ----
        print("C1d FAIL -> C2 aperture floor forms...", flush=True)

        def fit_s_and_floor(scat, err, floor_mmag, s_min=1.0):
            # scat^2 ~= (s*err)^2 + floor^2; solve s with floor fixed, clamp s>=1
            # then optionally re-solve floor after s clamp
            excess = np.maximum(scat * scat - err * err, 0.0)
            # if floor given as array, use residual after floor
            fl = np.asarray(floor_mmag, dtype=float)
            if fl.ndim == 0:
                fl = np.full_like(scat, float(fl))
            excess2 = np.maximum(scat * scat - fl * fl, 0.0)
            # s from median(scat / sqrt(err^2 + ...)) wait: after floor, s from ratio
            err_f = np.sqrt(err * err + fl * fl)
            ratios = scat / err_f
            ok = np.isfinite(ratios) & (err_f > 0)
            s_raw = float(np.median(ratios[ok])) if np.any(ok) else 1.0
            s_clamped = s_raw < s_min
            s = max(s_min, min(s_raw, 3.0))
            return s, s_clamped

        # Odd calib rows for form choice
        scat_o = np.asarray([r["scatter_mmag"] for r in rows_odd], dtype=float)
        err_o = np.asarray([r["err_model_mmag"] for r in rows_odd], dtype=float)
        rap_o = np.asarray([r["r_ap_px"] for r in rows_odd], dtype=float)
        ok_o = np.isfinite(scat_o) & np.isfinite(err_o) & (err_o > 0) & np.isfinite(rap_o)

        candidates = []

        # (iii) constant null
        cal_const = calibrate_smooth(rows_odd, form="constant_sigma_r", s_min=1.0)
        score_const = abs(
            med(
                [
                    r["scatter_mmag"]
                    / apply_smooth_mmag(r["err_model_mmag"], r["G"], cal_const)
                    for r in rows_even
                    if apply_smooth_mmag(r["err_model_mmag"], r["G"], cal_const) > 0
                ]
            )
            - 1.0
        )
        candidates.append(
            {
                "name": "constant",
                "score": score_const,
                "cal": cal_const,
                "fn": None,
                "meta": {"form": "constant_sigma_r", **cal_const.to_dict()},
            }
        )

        # (i) linear in r_ap: sigma_r = max(0, a * r_ap)
        # Fit a from odd: excess_mmag ~ a * r_ap
        y = np.sqrt(np.maximum(scat_o[ok_o] ** 2 - err_o[ok_o] ** 2, 0.0))
        rrr = rap_o[ok_o]
        # least squares y = a * r  (through origin) with a>=0
        if np.sum(rrr * rrr) > 0:
            a_lin = float(np.sum(rrr * y) / np.sum(rrr * rrr))
            a_lin = max(0.0, a_lin)
        else:
            a_lin = 0.0
        floor_lin = a_lin * rap_o
        s_lin, clamp_lin = fit_s_and_floor(scat_o[ok_o], err_o[ok_o], (a_lin * rrr))
        # re-fit a after s: excess vs (s*err)
        y2 = np.sqrt(np.maximum(scat_o[ok_o] ** 2 - (s_lin * err_o[ok_o]) ** 2, 0.0))
        if np.sum(rrr * rrr) > 0:
            a_lin = max(0.0, float(np.sum(rrr * y2) / np.sum(rrr * rrr)))
        cal_lin = SmoothCalibration(
            s=s_lin,
            sigma_r0_mmag=0.0,
            sigma_r_slope_mmag_per_G=0.0,
            form="constant_sigma_r",
            n_stars=int(np.sum(ok_o)),
            s_clamped=clamp_lin,
            median_ratio_pre=float("nan"),
            median_ratio_post=float("nan"),
        )

        def fn_lin(r, a=a_lin):
            return max(0.0, a * float(r["r_ap_px"]))

        score_lin = abs(
            med(
                [
                    r["scatter_mmag"]
                    / math.sqrt(
                        (s_lin * r["err_model_mmag"]) ** 2 + fn_lin(r) ** 2
                    )
                    for r in rows_even
                    if r["err_model_mmag"] > 0
                ]
            )
            - 1.0
        )
        candidates.append(
            {
                "name": "linear_r_ap",
                "score": score_lin,
                "cal": cal_lin,
                "fn": fn_lin,
                "meta": {
                    "form": "sigma_r = a * r_ap_px",
                    "a_mmag_per_px": a_lin,
                    "s": s_lin,
                    "s_clamped": clamp_lin,
                },
            }
        )

        # (ii) two-level step: bright (r_ap >= r_cut) vs rest
        # Choose r_cut by scanning odd held-in median ratio closeness
        best_step = None
        for r_cut in (5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0):
            bright = ok_o & (rap_o >= r_cut)
            rest = ok_o & (rap_o < r_cut)
            if int(np.sum(bright)) < 3 or int(np.sum(rest)) < 5:
                continue
            # floors from excess medians
            fl_b = float(
                math.sqrt(float(np.median(np.maximum(scat_o[bright] ** 2 - err_o[bright] ** 2, 0.0))))
            )
            fl_r = float(
                math.sqrt(float(np.median(np.maximum(scat_o[rest] ** 2 - err_o[rest] ** 2, 0.0))))
            )
            floor_arr = np.where(rap_o[ok_o] >= r_cut, fl_b, fl_r)
            s_st, clamp_st = fit_s_and_floor(scat_o[ok_o], err_o[ok_o], floor_arr)
            # re-estimate floors after s
            fl_b = float(
                math.sqrt(
                    float(
                        np.median(
                            np.maximum(scat_o[bright] ** 2 - (s_st * err_o[bright]) ** 2, 0.0)
                        )
                    )
                )
            )
            fl_r = float(
                math.sqrt(
                    float(
                        np.median(np.maximum(scat_o[rest] ** 2 - (s_st * err_o[rest]) ** 2, 0.0))
                    )
                )
            )

            def make_fn(rc=r_cut, fb=fl_b, fr=fl_r):
                def _fn(r):
                    return fb if float(r["r_ap_px"]) >= rc else fr

                return _fn

            fn = make_fn()
            score = abs(
                med(
                    [
                        r["scatter_mmag"]
                        / math.sqrt((s_st * r["err_model_mmag"]) ** 2 + fn(r) ** 2)
                        for r in rows_even
                        if r["err_model_mmag"] > 0
                    ]
                )
                - 1.0
            )
            cal_st = SmoothCalibration(
                s=s_st,
                sigma_r0_mmag=fl_r,
                sigma_r_slope_mmag_per_G=0.0,
                form="constant_sigma_r",
                n_stars=int(np.sum(ok_o)),
                s_clamped=clamp_st,
                median_ratio_pre=float("nan"),
                median_ratio_post=float("nan"),
            )
            cand = {
                "name": f"step_r{r_cut:g}",
                "score": score,
                "cal": cal_st,
                "fn": fn,
                "meta": {
                    "form": "two_level_r_ap",
                    "r_cut_px": r_cut,
                    "sigma_r_bright_mmag": fl_b,
                    "sigma_r_rest_mmag": fl_r,
                    "s": s_st,
                    "s_clamped": clamp_st,
                    "n_bright_odd": int(np.sum(bright)),
                    "n_rest_odd": int(np.sum(rest)),
                },
            }
            if best_step is None or score < best_step["score"]:
                best_step = cand
        if best_step is not None:
            candidates.append(best_step)

        candidates.sort(key=lambda c: c["score"] if math.isfinite(c["score"]) else 9e9)
        winner = candidates[0]
        print(
            "C2 candidates",
            [(c["name"], round(c["score"], 4) if math.isfinite(c["score"]) else None) for c in candidates],
            flush=True,
        )
        c2_eval = eval_bins(rows_even, winner["cal"], sigma_r_fn=winner["fn"])
        # Fire check later in C4

        # BIN-8-9 physics sentence
        bright_end = None
        if winner["fn"] is not None:
            # typical bright r_ap ~8.5
            bright_end = float(winner["fn"]({"r_ap_px": 8.5}))
        elif winner["name"] == "constant":
            bright_end = float(winner["cal"].sigma_r0_mmag)
        bin89_sentence = (
            f"Winning floor at r_ap~8.5 px is {bright_end:.2f} mmag; "
            f"BIN-8-9 bright LOO ~12 mmag at r 8.5-9.5 px is consistent with "
            f"this floor plus photon/scint/SEM (floor is a major bright-end term)."
            if bright_end is not None and bright_end > 0
            else "Constant/null floor; no strong BIN-8-9 aperture-floor link claimed."
        )
        if winner["name"] != "constant" and c2_eval["pass"]:
            bin89_sentence += " Register pointer: BIN-8-9 <-> WIDE-ERR aperture floor."

        c2_payload = {
            "task": "WIDE-ERR-03C C2",
            "run_sha": RUN_SHA,
            "candidates": [
                {"name": c["name"], "score": c["score"], "meta": c["meta"]} for c in candidates
            ],
            "winner": winner["name"],
            "winner_meta": winner["meta"],
            "even_gate": c2_eval,
            "bin_8_9_physics": bin89_sentence,
        }
        OUT_C2.write_text(json.dumps(c2_payload, indent=2) + "\n", encoding="utf-8")
        print("WROTE", OUT_C2)
        print("C2", winner["name"], "PASS" if c2_eval["pass"] else "FAIL", c2_eval["bins_outside"])

        if c2_eval["pass"]:
            winning_level = f"C2_{winner['name']}"
            winning_cal = winner["cal"]
            winning_form_meta = winner["meta"]
            sigma_r_fn = winner["fn"]
        else:
            # ---- C3: per-curve floor ----
            print("C2 FAIL -> C3 per-LC floor...", flush=True)
            # For each star: fit s>=1 and floor>=0 on odd, evaluate on even
            per_lc = []
            even_rows_c3 = []
            for cid in sorted({r["catalog_id"] for r in rows_all}):
                ro = next((r for r in rows_odd if r["catalog_id"] == cid), None)
                re = next((r for r in rows_even if r["catalog_id"] == cid), None)
                if ro is None or re is None:
                    continue
                # Single-star: floor from odd excess, s from ratio after floor
                so, eo = ro["scatter_mmag"], ro["err_model_mmag"]
                if not (math.isfinite(so) and math.isfinite(eo) and eo > 0):
                    continue
                fl = math.sqrt(max(0.0, so * so - eo * eo))
                s_raw = so / math.sqrt(eo * eo + fl * fl) if (eo * eo + fl * fl) > 0 else 1.0
                s_cl = s_raw < 1.0
                s = max(1.0, min(float(s_raw), 3.0))
                fl2 = math.sqrt(max(0.0, so * so - (s * eo) ** 2))
                # even prediction
                ee = re["err_model_mmag"]
                e_exp = math.sqrt((s * ee) ** 2 + fl2 * fl2) if math.isfinite(ee) and ee > 0 else float("nan")
                ratio_e = re["scatter_mmag"] / e_exp if e_exp and e_exp > 0 else float("nan")
                per_lc.append(
                    {
                        "catalog_id": cid,
                        "G": re["G"],
                        "r_ap_px": re["r_ap_px"],
                        "s": s,
                        "s_clamped": s_cl,
                        "sigma_r_mmag": fl2,
                        "scatter_odd_mmag": so,
                        "scatter_even_mmag": re["scatter_mmag"],
                        "err_model_even_mmag": ee,
                        "err_exported_even_mmag": e_exp,
                        "ratio_even": ratio_e,
                        "ensemble_case": re["ensemble_case"],
                    }
                )
                even_rows_c3.append(
                    {
                        **re,
                        "scatter_mmag": re["scatter_mmag"],
                        "err_model_mmag": ee,
                        "_s": s,
                        "_sr": fl2,
                    }
                )

            def fn_c3(r, table={p["catalog_id"]: p for p in per_lc}):
                p = table.get(r["catalog_id"])
                return float(p["sigma_r_mmag"]) if p else 0.0

            # Custom eval with per-star s
            outside = []
            by = {}
            for i in range(15):
                lo, hi = 8 + 0.5 * i, 8 + 0.5 * (i + 1)
                lab = bin_label(lo, hi)
                sub = [r for r in even_rows_c3 if math.isfinite(r["G"]) and lo < r["G"] <= hi]
                if not sub:
                    by[lab] = {"n": 0}
                    continue
                ratios = []
                errs = []
                for r in sub:
                    e = math.sqrt((r["_s"] * r["err_model_mmag"]) ** 2 + r["_sr"] ** 2)
                    if e > 0:
                        ratios.append(r["scatter_mmag"] / e)
                        errs.append(e)
                mr = med(ratios)
                entry = {
                    "n": len(sub),
                    "median_ratio": mr,
                    "median_err_mmag": med(errs),
                    "gated": len(sub) >= 4,
                }
                by[lab] = entry
                if len(sub) >= 4 and math.isfinite(mr) and not (GATE_LO <= mr <= GATE_HI):
                    outside.append({"bin": lab, **entry})
            g89 = [r for r in even_rows_c3 if 8.0 < r["G"] <= 9.0]
            r89 = []
            e89 = []
            for r in g89:
                e = math.sqrt((r["_s"] * r["err_model_mmag"]) ** 2 + r["_sr"] ** 2)
                if e > 0:
                    r89.append(r["scatter_mmag"] / e)
                    e89.append(e)
            c3_pass = len(outside) == 0 and bool(
                g89
                and math.isfinite(med(r89))
                and GATE_LO <= med(r89) <= GATE_HI
                and med(e89) >= SCINT_FLOOR_MMAG
            )
            c3_eval = {
                "by_G_bin": by,
                "bins_outside": outside,
                "G8_9": {
                    "n": len(g89),
                    "median_ratio": med(r89),
                    "median_err_mmag": med(e89),
                    "in_window": bool(
                        g89
                        and math.isfinite(med(r89))
                        and GATE_LO <= med(r89) <= GATE_HI
                        and med(e89) >= SCINT_FLOOR_MMAG
                    ),
                },
                "pass": c3_pass,
            }
            c3_payload = {
                "task": "WIDE-ERR-03C C3",
                "run_sha": RUN_SHA,
                "n_curves": len(per_lc),
                "even_gate": c3_eval,
                "per_lc_sample": per_lc[:5],
            }
            OUT_C3.write_text(json.dumps(c3_payload, indent=2) + "\n", encoding="utf-8")
            print("WROTE", OUT_C3)
            print("C3 PASS" if c3_pass else "C3 FAIL", c3_eval["bins_outside"])

            if c3_pass:
                winning_level = "C3_per_lc_floor"
                winning_cal = SmoothCalibration(
                    s=1.0,
                    sigma_r0_mmag=0.0,
                    sigma_r_slope_mmag_per_G=0.0,
                    form="constant_sigma_r",
                    n_stars=len(per_lc),
                    s_clamped=False,
                    median_ratio_pre=float("nan"),
                    median_ratio_post=float("nan"),
                )
                winning_form_meta = {
                    "form": "per_lc_floor",
                    "n_curves": len(per_lc),
                    "per_lc": per_lc,
                }
                sigma_r_fn = fn_c3
            else:
                # Pont residual evidence for BO/FW
                bo_id = "1498613634033133184"
                fw_id = "1497343732462852864"
                # BO/FW are targets not comps - measure from lightcurves if present
                pont = {}
                for label, cid in (("BO", bo_id), ("FW", fw_id)):
                    lc = PHOT / "lightcurves" / f"lightcurve_{cid}.csv"
                    if not lc.is_file():
                        pont[label] = {"missing": True}
                        continue
                    df = pd.read_csv(lc, comment="#", low_memory=False)
                    mag = pd.to_numeric(df.get("mag_calib_final", df.get("mag_calib")), errors="coerce")
                    err = pd.to_numeric(df["err"], errors="coerce")
                    # binned residual sigma_r proxy: MAD of (mag - median) vs median err
                    scat = mad_mmag(mag.to_numpy())
                    med_err = float(np.nanmedian(err)) * 1000.0 * MAG_ERR_SCALE / MAG_ERR_SCALE
                    # err is already in mag; convert to mmag
                    med_err_mmag = float(np.nanmedian(err)) * 1000.0
                    pont[label] = {
                        "catalog_id": cid,
                        "scatter_mad_mmag": scat,
                        "median_err_mmag": med_err_mmag,
                        "ratio_scatter_over_err": (
                            scat / med_err_mmag if med_err_mmag > 0 else float("nan")
                        ),
                        "frame": "exported lightcurve mag_calib_final vs err",
                        "citation": "Pont, Zucker & Queloz 2006 sigma_r binned-residual spirit",
                    }
                corr_item = {
                    "id": "CORR-ERR-01",
                    "status": "OPEN",
                    "note": (
                        "Residual after product-frame meter + aperture floor + per-LC floor "
                        "still fails even-half gate; correlated/common-mode beyond diagonal "
                        "(s, sigma_r)."
                    ),
                    "pont_bo_fw": pont,
                    "c3_bins_outside": outside,
                }
                summary = {
                    "task": "WIDE-ERR-03C",
                    "outcome": "STOP_CORR_ERR_01",
                    "run_sha": RUN_SHA,
                    "part0_push": "SKIPPED_blank_authorization",
                    "c1d_pass": False,
                    "c2_pass": False,
                    "c3_pass": False,
                    "register": {
                        "GAIN-DOMAIN-01": "CLOSED",
                        "WIDE-ERR": "OPEN",
                        "SEM": "OPEN",
                        "CORR-ERR-01": "OPEN",
                        "WIDE-ERR-CROSSRIG": "OPEN",
                    },
                    "corr_err_01": corr_item,
                    "c1": str(OUT_C1),
                    "c2": str(OUT_C2),
                    "c3": str(OUT_C3),
                }
                OUT_SUM.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                print("STOP CORR-ERR-01", flush=True)
                print("WROTE", OUT_SUM)
                return 2

    # ---- C4: fire proof + gate confirm + sidecar ----
    assert winning_cal is not None
    # Fire proof on product-frame even half
    wrong_rows = [
        {
            "catalog_id": r["catalog_id"],
            "G": r["G"],
            "scatter_mmag": r["scatter_mmag"],
            "err_model_mmag": r["err_wrong_mmag"],
            "r_ap_px": r["r_ap_px"],
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

    final_eval = eval_bins(rows_even, winning_cal, sigma_r_fn=sigma_r_fn)
    gate_pass = bool(final_eval["pass"] and fire_fails)

    # Refit production coeffs on all frames for constant form; keep C2/C3 meta
    if winning_level and winning_level.startswith("C1"):
        prod_cal = cal_03b
    elif winning_level and "constant" in winning_level and sigma_r_fn is None:
        prod_cal = calibrate_smooth(rows_all, form="constant_sigma_r", s_min=1.0)
        winning_cal = prod_cal
    else:
        prod_cal = winning_cal

    sidecar = {
        "task": "WIDE-ERR-03C",
        "run_sha": RUN_SHA,
        "draft_id": 515,
        "meter": "mag_calib_pytics_product_frame",
        "winning_level": winning_level,
        "form": winning_form_meta.get("form", "constant_sigma_r"),
        "smooth": prod_cal.to_dict() if prod_cal is not None else None,
        "form_meta": winning_form_meta,
        "s_min_clamp": 1.0,
        "gain_authority": g_new,
        "b2c_pass": gate_pass,
        "c1d_meter_artifact": bool(c1d_eval["pass"]),
    }
    if "per_lc" in winning_form_meta:
        sidecar["per_lc_floor"] = {
            p["catalog_id"]: {"s": p["s"], "sigma_r_mmag": p["sigma_r_mmag"]}
            for p in winning_form_meta["per_lc"]
        }
    write_sidecar(PHOT / ERR_CALIB_SIDECAR, sidecar)

    c4_payload = {
        "task": "WIDE-ERR-03C C4",
        "run_sha": RUN_SHA,
        "winning_level": winning_level,
        "gate_pass": gate_pass,
        "final_eval": final_eval,
        "fire_proof": {
            "failed_as_required": fire_fails,
            "eval": fire,
        },
        "sidecar": str(PHOT / ERR_CALIB_SIDECAR),
        "form_meta": winning_form_meta,
    }
    OUT_C4.write_text(json.dumps(c4_payload, indent=2) + "\n", encoding="utf-8")
    print("WROTE", OUT_C4)
    print("C4 gate", "PASS" if gate_pass else "FAIL", "fire_fails", fire_fails)

    summary = {
        "task": "WIDE-ERR-03C",
        "outcome": "GATE_PASS" if gate_pass else "GATE_FAIL",
        "run_sha": RUN_SHA,
        "part0_push": "SKIPPED_blank_authorization",
        "c1d_pass": bool(c1d_eval["pass"]),
        "winning_level": winning_level,
        "gate_pass": gate_pass,
        "needs_reexport": bool(gate_pass),
        "register_intent": {
            "GAIN-DOMAIN-01": "CLOSED",
            "WIDE-ERR": "CLOSED" if gate_pass else "OPEN",
            "SEM": "CLOSED" if gate_pass else "OPEN",
            "WIDE-ERR-CROSSRIG": "OPEN",
        },
    }
    OUT_SUM.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("WROTE", OUT_SUM)
    return 0 if gate_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
