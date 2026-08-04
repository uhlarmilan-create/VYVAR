#!/usr/bin/env python3
"""Wide-rig error-budget diagnostic H1 vs H2 (batch D GATE 1 follow-up)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dev" / "scripts"))
import _bootstrap  # noqa: E402,F401

from config import AppConfig  # noqa: E402
from photometry_core import (  # noqa: E402
    _phase2a_prepare_shared_state,
    check_comparison_stability,
    ensemble_member_ids,
    parse_comp_quality_json_map,
    photometer_check_star_production_path,
)
from scripts.audit_stage3_part1c_robust_chi2 import (  # noqa: E402
    CLIP_MAXITERS,
    CLIP_SIGMA,
    _iterative_sigma_clip_chi2,
    _mad_robust_chi2,
)
from scripts.chi2_sigma_gate import reduced_chi2_constant  # noqa: E402
from check_star_kmag import build_aligned_comp_inst, comp_ensemble_maps, resolve_proc_csv_dir  # noqa: E402
from scripts.select_constant_calibrators import compute_loo_production_ensemble_scatter  # noqa: E402
from sigma_budget import resolve_rig_scintillation_params, scintillation_sigma  # noqa: E402

SETUP = "NoFilter_60_2"
DRAFT_ID = 435
EQUIPMENT_ID = 1
MAG_ERR_SCALE = 1000.0


def _weighted_scatter(mags: np.ndarray, errs: np.ndarray) -> float:
    m = np.asarray(mags, dtype=np.float64)
    e = np.asarray(errs, dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
    m = m[ok]
    e = e[ok]
    if m.size < 3:
        return float("nan")
    w = 1.0 / (e * e)
    ref = float(np.sum(w * m) / np.sum(w))
    resid = m - ref
    return float(np.std(resid, ddof=1))


def _linfit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[ok]
    y = y[ok]
    if x.size < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "n": int(x.size)}
    A = np.column_stack([x, np.ones(x.size)])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return {"slope": float(coef[0]), "intercept": float(coef[1]), "n": int(x.size)}


def _err_terms_row(lc_df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for col, key in (
        ("err_photon", "photon_mmag"),
        ("err_sem_rel", "ensemble_mmag"),
        ("err_scint_rel", "scint_mmag"),
        ("err_sigma_sys_rel", "sys_mmag"),
        ("err", "total_mmag"),
    ):
        if col not in lc_df.columns:
            continue
        v = pd.to_numeric(lc_df[col], errors="coerce")
        med = float(np.nanmedian(v.to_numpy(dtype=np.float64)))
        if col == "err":
            out[key] = med * MAG_ERR_SCALE
        else:
            out[key] = med * MAG_ERR_SCALE
    return out


def _ensemble_t4(
    *,
    phot_dir: Path,
    setup: str,
    target_cid: str,
    check_cid: str,
    lc_df: pd.DataFrame,
    cfg: AppConfig,
) -> dict[str, Any]:
    comp_all = pd.read_csv(
        phot_dir / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    comp_df = comp_all.loc[comp_all["target_catalog_id"].astype(str).str.strip() == str(target_cid)]
    comp_ids = [str(c).strip() for c in comp_df["catalog_id"].tolist() if str(c).strip()]
    n_comp = int(len(comp_ids))
    comp_rms: list[dict[str, Any]] = []
    for _, row in comp_df.iterrows():
        cid = str(row["catalog_id"]).strip()
        cr = float(pd.to_numeric(row.get("comp_rms"), errors="coerce"))
        comp_rms.append({"catalog_id": cid, "comp_rms_catalog": cr, "quality": ""})
    loo_scatter = float("nan")
    # LOO ensemble scatter is expensive; computed for representative target only in main().
    return {
        "n_comp": n_comp,
        "comp_ids": comp_ids,
        "comp_rms_rows": comp_rms,
        "loo_ensemble_scatter_mmag_median": loo_scatter,
    }


def main() -> int:
    cfg = AppConfig()
    draft = Path(cfg.archive_root) / "Drafts" / "draft_000435_snapshot_skysurface_20260716"
    ps = draft / "platesolve" / SETUP
    phot = ps / "photometry"
    lc_dir = phot / "lightcurves"
    lights = draft / "detrended_aligned" / "lights" / SETUP

    rig = resolve_rig_scintillation_params(draft_id=DRAFT_ID, setup=SETUP, cfg=cfg)
    scint_x1_mmag = (
        float(
            1000.0
            * scintillation_sigma(
                telescope_diameter_m=float(rig.telescope_diameter_m),
                airmass=1.0,
                exposure_s=float(rig.exposure_s),
                altitude_m=float(rig.altitude_m),
                c_y=float(rig.c_y),
            )
        )
        if rig is not None
        else float("nan")
    )

    state = _phase2a_prepare_shared_state(
        output_dir=phot,
        lc_dir=lc_dir,
        masterstar_fits_path=ps / "MASTERSTAR.fits",
        comparison_stars_csv=phot / "comparison_stars_per_target.csv",
        per_frame_csv_dir=lights,
        progress_cb=None,
        active_targets_csv=ps / "variable_targets.csv",
        detrended_aligned_dir=lights,
        fwhm_px=3.2,
        cfg=cfg,
        db=None,
        draft_id=DRAFT_ID,
    )

    rows: list[dict[str, Any]] = []
    for ck_path in sorted(lc_dir.glob("check_kmag_*.csv")):
        target_cid = ck_path.stem.replace("check_kmag_", "")
        ckdf = pd.read_csv(ck_path, low_memory=False)
        if ckdf.empty:
            continue
        id_col = "check_catalog_id" if "check_catalog_id" in ckdf.columns else "check_cid"
        check_cid = str(ckdf[id_col].iloc[0]).strip()
        mag_g = float("nan")
        for _gc in ("phot_g_mean_mag", "mag", "check_mag"):
            if _gc in ckdf.columns:
                mag_g = float(pd.to_numeric(ckdf[_gc].iloc[0], errors="coerce"))
                if math.isfinite(mag_g):
                    break
        if not math.isfinite(mag_g):
            comp_path = phot / "comparison_stars_per_target.csv"
            if comp_path.is_file():
                comp_all = pd.read_csv(comp_path, low_memory=False, dtype={"catalog_id": str})
                sub = comp_all.loc[comp_all["catalog_id"].astype(str).str.strip() == check_cid]
                if not sub.empty:
                    mag_g = float(pd.to_numeric(sub["mag"].iloc[0], errors="coerce"))
        lc_df = photometer_check_star_production_path(
            state=state,
            parent_target_cid=target_cid,
            check_cid=check_cid,
            masterstar_fits_path=ps / "MASTERSTAR.fits",
            lc_dir=phot / "diag_check_lc" / target_cid,
            output_dir=phot,
        )
        if lc_df is None or "mag_calib_final" not in lc_df.columns:
            continue
        m = pd.to_numeric(lc_df["mag_calib_final"], errors="coerce").to_numpy(dtype=np.float64)
        e = pd.to_numeric(lc_df["err"], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
        if int(np.count_nonzero(ok)) < 3:
            continue
        scatter = _weighted_scatter(m[ok], e[ok])
        err_med = float(np.median(e[ok]))
        _, _, chi2_clip, _ = reduced_chi2_constant(m[ok], e[ok])
        chi2_clip2, _, _ = _iterative_sigma_clip_chi2(m[ok], e[ok])
        ens = _ensemble_t4(
            phot_dir=phot, setup=SETUP, target_cid=target_cid, check_cid=check_cid, lc_df=lc_df, cfg=cfg,
        )
        row = {
            "target_cid": target_cid,
            "check_cid": check_cid,
            "mag_g": mag_g,
            "N_epochs": int(np.count_nonzero(ok)),
            "scatter_mag": scatter,
            "scatter_mmag": scatter * MAG_ERR_SCALE,
            "err_median_mmag": err_med * MAG_ERR_SCALE,
            "chi2_red_clipped": float(chi2_clip2),
            "chi2_red_raw": float(chi2_clip),
            "ratio_scatter_over_err": scatter / err_med if err_med > 0 else float("nan"),
            **_err_terms_row(lc_df),
            **ens,
        }
        rows.append(row)

    if not rows:
        print(json.dumps({"error": "no check stars"}, indent=2))
        return 1

    df = pd.DataFrame(rows)
    mags = df["mag_g"].to_numpy(dtype=np.float64)
    q25, q50, q75 = np.nanquantile(mags, [0.25, 0.5, 0.75])

    def _qstats(label: str, mask: np.ndarray) -> dict[str, Any]:
        sub = df.loc[mask]
        if sub.empty:
            return {"label": label, "n": 0}
        return {
            "label": label,
            "n": int(len(sub)),
            "mag_g_range": [float(sub["mag_g"].min()), float(sub["mag_g"].max())],
            "scatter_mmag_median": float(sub["scatter_mmag"].median()),
            "err_mmedian_mmag": float(sub["err_median_mmag"].median()),
            "chi2_clip_median": float(sub["chi2_red_clipped"].median()),
        }

    t1 = {
        "bright_quartile": _qstats("bright", mags <= q25),
        "middle_half": _qstats("middle", (mags > q25) & (mags <= q75)),
        "faint_quartile": _qstats("faint", mags > q75),
        "mag_quartiles_g": [float(q25), float(q50), float(q75)],
    }

    x = df["err_median_mmag"].to_numpy(dtype=np.float64)
    y = df["scatter_mmag"].to_numpy(dtype=np.float64)
    fit = _linfit(x, y)
    fit_through_origin = _linfit(x, y * 0 + y)  # dummy
    # slope through origin
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    slope0 = float(np.sum(x[ok] * y[ok]) / np.sum(x[ok] ** 2)) if ok.sum() >= 3 else float("nan")
    t2 = {
        "linear_fit_scatter_vs_quoted_err_mmag": fit,
        "slope_through_origin": slope0,
        "median_ratio_scatter_over_err": float(np.median(df["ratio_scatter_over_err"])),
        "p95_ratio_scatter_over_err": float(np.quantile(df["ratio_scatter_over_err"], 0.95)),
    }

    bright = df.loc[mags <= q25].sort_values("mag_g")
    faint = df.loc[mags > q75].sort_values("mag_g", ascending=False)
    rep_bright = bright.iloc[0].to_dict() if not bright.empty else {}
    rep_faint = faint.iloc[0].to_dict() if not faint.empty else {}
    t3 = {
        "representative_bright": {k: rep_bright.get(k) for k in (
            "check_cid", "mag_g", "photon_mmag", "ensemble_mmag", "scint_mmag", "sys_mmag", "total_mmag",
            "scatter_mmag", "err_median_mmag", "chi2_red_clipped",
        )},
        "representative_faint": {k: rep_faint.get(k) for k in (
            "check_cid", "mag_g", "photon_mmag", "ensemble_mmag", "scint_mmag", "sys_mmag", "total_mmag",
            "scatter_mmag", "err_median_mmag", "chi2_red_clipped",
        )},
    }

    n_comp = df["n_comp"].to_numpy(dtype=np.float64)
    high_rms = [
        r for rows_c in df["comp_rms_rows"] for r in (rows_c or []) if float(r.get("comp_rms_catalog", 0) or 0) > 0.05
    ]
    # LOO ensemble scatter for one representative field (anchor-style target with most comps)
    rep_row = df.sort_values("n_comp", ascending=False).iloc[0]
    loo_rep = float("nan")
    try:
        loo_arr = compute_loo_production_ensemble_scatter(
            str(rep_row["check_cid"]),
            phot_dir=phot,
            setup=SETUP,
            anchor_target=str(rep_row["target_cid"]),
            cfg=cfg,
        )
        if loo_arr is not None and len(loo_arr):
            loo_rep = float(np.nanmedian(np.asarray(loo_arr, dtype=np.float64) * MAG_ERR_SCALE))
    except Exception:  # noqa: BLE001
        loo_rep = float("nan")
    t4 = {
        "n_comp_median": float(np.nanmedian(n_comp)),
        "n_comp_min": float(np.nanmin(n_comp)),
        "n_comp_max": float(np.nanmax(n_comp)),
        "loo_ensemble_scatter_mmag_representative": loo_rep,
        "loo_representative_target_cid": str(rep_row["target_cid"]),
        "fields_with_n_comp_lt_5": int(np.sum(n_comp < 5)),
        "high_comp_rms_comps": high_rms[:20],
    }

    # Pre-registered verdict
    slope = fit.get("slope", float("nan"))
    med_ratio = t2["median_ratio_scatter_over_err"]
    bright_scatter = t1["bright_quartile"].get("scatter_mmag_median", float("nan"))
    faint_scatter = t1["faint_quartile"].get("scatter_mmag_median", float("nan"))
    flat_t1 = (
        math.isfinite(bright_scatter)
        and math.isfinite(faint_scatter)
        and abs(bright_scatter - faint_scatter) < 5.0
    )
    rising_t1 = math.isfinite(faint_scatter) and math.isfinite(bright_scatter) and faint_scatter > bright_scatter + 3.0
    const_mult = math.isfinite(med_ratio) and 1.6 <= med_ratio <= 2.4
    on_11 = math.isfinite(slope) and 0.75 <= slope <= 1.35 and abs(fit.get("intercept", 999)) < 5.0

    if flat_t1 and const_mult:
        verdict = "H1-global"
        mechanism = "quoted error budget scaled ~{:.2f}x low vs measured scatter (T2)".format(med_ratio)
    elif t4["fields_with_n_comp_lt_5"] > 20 or t4["high_comp_rms_comps"]:
        verdict = "H1-ensemble"
        mechanism = "ensemble underquoted: n_comp median {:.0f}; variable/high-RMS comps present".format(
            t4["n_comp_median"]
        )
    elif rising_t1 and on_11:
        verdict = "H2"
        mechanism = "scatter rises toward faint stars; slope~{:.2f} on scatter vs quoted err".format(slope)
    elif flat_t1:
        verdict = "H1-global"
        mechanism = "flat ~{:.0f} mmag scatter vs magnitude; budget underquoted".format(bright_scatter)
    else:
        verdict = "H2-mixed"
        mechanism = "mixed: some stars above 1:1, magnitude-dependent scatter"

    out = {
        "draft_id": DRAFT_ID,
        "equipment_id": EQUIPMENT_ID,
        "n_check_fields": int(len(df)),
        "scintillation_mmag_at_X1": scint_x1_mmag,
        "median_chi2_red_clipped": float(df["chi2_red_clipped"].median()),
        "median_scatter_mmag": float(df["scatter_mmag"].median()),
        "median_quoted_err_mmag": float(df["err_median_mmag"].median()),
        "T1_scatter_vs_magnitude": t1,
        "T2_scatter_vs_quoted_err": t2,
        "T3_error_decomposition": t3,
        "T4_ensemble": t4,
        "verdict": verdict,
        "mechanism": mechanism,
        "floor_applied": False,
        "per_check_star": rows,
    }
    out_path = REPO / "tmp" / "wide_error_budget_diag.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k != "per_check_star"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
