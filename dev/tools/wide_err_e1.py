#!/usr/bin/env python3
"""WIDE-ERR E1: locate ensemble-term underquote (read-only)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))
sys.path.insert(0, str(REPO / "dev" / "scripts"))
import _bootstrap  # noqa: E402,F401

from config import AppConfig  # noqa: E402
from photometry_core import (  # noqa: E402
    _comp_maps_from_comparison_stars_csv,
    _group_comp_mag_inst_from_proc_csvs,
    _mad_sigma,
    check_comparison_stability,
    ensemble_normalize,
    temporal_bin_comp_lc,
)
from sigma_floor_core import c4_small_sample, ensemble_sem_mag_from_residuals  # noqa: E402

SETUP = "NoFilter_60_2"
DRAFT = REPO / "Archive" / "Drafts" / "draft_000435_snapshot_skysurface_20260716"
CHECK_CID = "1499906247391001088"
T3_FIELD = "1485540612577549568"
OUT = REPO / "tmp" / "wide_err_e1"
MAD_SCALE = 1.4826
MAG_ERR_SCALE = 1000.0


def _iqr(x: np.ndarray) -> list[float]:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [float("nan")] * 3
    q25, q50, q75 = np.quantile(v, [0.25, 0.5, 0.75])
    return [float(q25), float(q50), float(q75)]


def _mad_sigma_arr(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float("nan")
    return float(MAD_SCALE * np.median(np.abs(v - np.median(v))))


def _per_frame_sets(
    *,
    comp_mag_inst: dict[str, np.ndarray],
    comp_catalog_mag: dict[str, float],
    comp_quality: dict[str, dict],
    comp_rms_map: dict[str, float],
    n_comp_min: int,
    n_comp_max: int,
    frame_i: int,
) -> dict[str, Any]:
    """Mirror ensemble_normalize comp selection for one frame."""
    p2p_thr = float("nan")
    for q in comp_quality.values():
        t = q.get("p2p_threshold")
        if t is not None and math.isfinite(float(t)):
            p2p_thr = float(t)
            break
    usable_all = [cid for cid, q in comp_quality.items() if q.get("quality") in ("good", "suspect")]
    usable_sorted = sorted(
        usable_all,
        key=lambda c: (
            0 if comp_quality[c].get("quality") == "good" else 1,
            float(comp_rms_map.get(c, float("inf"))),
            str(c),
        ),
    )
    selected: list[str] = []
    for cid in usable_sorted:
        if len(selected) >= n_comp_max:
            break
        p2p = float(comp_quality[cid].get("rms_p2p", float("nan")))
        if (
            len(selected) < n_comp_min
            or (math.isfinite(p2p_thr) and math.isfinite(p2p) and p2p < p2p_thr)
            or not math.isfinite(p2p_thr)
        ):
            selected.append(cid)
    good_ids = selected[:n_comp_max]

    comp_ref_map: dict[str, float] = {}
    for cid in good_ids:
        arr = comp_mag_inst.get(cid)
        if arr is None:
            continue
        fin = arr[np.isfinite(arr)]
        if fin.size:
            comp_ref_map[cid] = float(np.median(fin))

    comp_pairs: list[tuple[str, float]] = []
    for cid in good_ids:
        if cid not in comp_mag_inst:
            continue
        mv = float(comp_mag_inst[cid][frame_i])
        if math.isfinite(mv):
            comp_pairs.append((cid, mv))

    flux_ids = [cid for cid, _ in comp_pairs]
    sem_ids = [
        cid
        for cid, m in comp_pairs
        if cid in comp_ref_map and math.isfinite(comp_ref_map[cid])
    ]
    return {
        "good_ids": good_ids,
        "comp_pairs": comp_pairs,
        "flux_ids": flux_ids,
        "sem_ids": sem_ids,
        "comp_ref_map": comp_ref_map,
    }


def _frame_sem_honeycutt(comp_pairs: list[tuple[str, float]], comp_ref_map: dict[str, float]) -> float:
    comp_resid = [
        m - comp_ref_map[cid]
        for cid, m in comp_pairs
        if cid in comp_ref_map and math.isfinite(comp_ref_map[cid])
    ]
    if len(comp_resid) < 2:
        return float("nan")
    return float(ensemble_sem_mag_from_residuals(comp_resid))


def _frame_actual_about_ens_med(comp_pairs: list[tuple[str, float]]) -> tuple[float, float]:
    """Return (raw_std, honeycutt_sem) of comps about flux-sum ensemble mean."""
    if len(comp_pairs) < 2:
        return float("nan"), float("nan")
    fluxes = [10 ** (-0.4 * m) for _, m in comp_pairs]
    s = float(np.sum(fluxes))
    if not (math.isfinite(s) and s > 0):
        return float("nan"), float("nan")
    ens_med = float(-2.5 * math.log10(s))
    resid = [m - ens_med for _, m in comp_pairs]
    n = len(resid)
    std = float(np.std(resid, ddof=1))
    c4 = c4_small_sample(n)
    sem = std / c4 / math.sqrt(n) if c4 > 0 else float("nan")
    return std, sem


def _field_pipeline(
    *,
    target_cid: str,
    comp_df: pd.DataFrame,
    csv_files: list[Path],
    cfg: AppConfig,
    all_frames_stub: pd.DataFrame,
) -> dict[str, Any]:
    comp_ids = [
        str(c).strip()
        for c in comp_df.loc[comp_df["target_catalog_id"].astype(str).str.strip() == target_cid, "catalog_id"]
        if str(c).strip() and str(c).strip() != CHECK_CID
    ]
    if len(comp_ids) < 3:
        return {"skip": True, "reason": "few_comps"}

    comp_mag_raw = _group_comp_mag_inst_from_proc_csvs(comp_ids, csv_files)
    comp_rms_map = {
        str(r["catalog_id"]).strip(): float(pd.to_numeric(r.get("comp_rms"), errors="coerce"))
        for _, r in comp_df.loc[comp_df["target_catalog_id"].astype(str).str.strip() == target_cid].iterrows()
        if str(r["catalog_id"]).strip() in comp_ids
    }
    comp_csv_path = (
        DRAFT / "platesolve" / SETUP / "photometry" / "comparison_stars_per_target.csv"
    )
    _, comp_catalog_mag, comp_quality_csv = _comp_maps_from_comparison_stars_csv(comp_csv_path)
    comp_catalog_mag = {k: v for k, v in comp_catalog_mag.items() if k in comp_ids}
    comp_quality_csv = {k: v for k, v in comp_quality_csv.items() if k in comp_ids}

    n_comp_min = max(1, int(getattr(cfg, "phase01_comparison_n_comp_min", 3)))
    n_comp_max = int(cfg.phase01_comparison_n_comp_max)
    stability_sigma = float(getattr(cfg, "stability_sigma", 3.0))

    comp_bjd = {cid: np.arange(len(comp_mag_raw[cid]), dtype=np.float64) for cid in comp_ids}
    comp_lc_bin = temporal_bin_comp_lc(
        comp_lc=comp_mag_raw,
        comp_quality={},
        all_frames=all_frames_stub,
        window=int(cfg.temporal_bin_window),
        enabled=bool(cfg.temporal_binning_enabled),
    )
    comp_quality = check_comparison_stability(
        comp_lc_bin,
        comp_rms_map=comp_rms_map,
        comp_bjd=comp_bjd,
        n_comp_min=n_comp_min,
        outlier_sigma=stability_sigma,
        max_comp_slope_mmag_hr=float(cfg.comp_max_slope_mmag_hr),
        comp_slope_significance_k=float(getattr(cfg, "comp_slope_significance_k", 3.0)),
        common_mode_detrend=True,
    )

    n_frames = len(csv_files)
    target_stub = np.zeros(n_frames, dtype=np.float64)
    _, _, ens_prod = ensemble_normalize(
        target_stub,
        comp_lc_bin,
        comp_catalog_mag,
        comp_quality,
        comp_rms_map=comp_rms_map,
        n_comp_min=n_comp_min,
        n_comp_max=n_comp_max,
    )

    # Unclipped: all comps, no stability exclusion, no n_comp_max cap
    comp_quality_all = {cid: {"quality": "good", "rms_p2p": 0.0, "p2p_threshold": float("nan")} for cid in comp_ids}
    _, _, ens_unclip = ensemble_normalize(
        target_stub,
        comp_lc_bin,
        comp_catalog_mag,
        comp_quality_all,
        comp_rms_map=comp_rms_map,
        n_comp_min=1,
        n_comp_max=max(len(comp_ids), 999),
    )

    clip_ratios: list[float] = []
    actual_ratios: list[float] = []
    membership_rows: list[dict[str, Any]] = []

    for i in range(n_frames):
        info = _per_frame_sets(
            comp_mag_inst=comp_lc_bin,
            comp_catalog_mag=comp_catalog_mag,
            comp_quality=comp_quality,
            comp_rms_map=comp_rms_map,
            n_comp_min=n_comp_min,
            n_comp_max=n_comp_max,
            frame_i=i,
        )
        if info["comp_pairs"]:
            membership_rows.append(
                {
                    "frame": i,
                    "n_flux": len(info["flux_ids"]),
                    "n_sem": len(info["sem_ids"]),
                    "diff": len(info["flux_ids"]) - len(info["sem_ids"]),
                }
            )
        sp = float(ens_prod[i]) if i < ens_prod.size else float("nan")
        su = float(ens_unclip[i]) if i < ens_unclip.size else float("nan")
        if math.isfinite(sp) and sp > 0 and math.isfinite(su):
            clip_ratios.append(su / sp)

        _, actual_sem = _frame_actual_about_ens_med(info["comp_pairs"])
        if math.isfinite(sp) and sp > 0 and math.isfinite(actual_sem):
            actual_ratios.append(actual_sem / sp)

    return {
        "target_cid": target_cid,
        "n_comps_pool": len(comp_ids),
        "n_frames": n_frames,
        "clip_ratio_median": float(np.median(clip_ratios)) if clip_ratios else float("nan"),
        "clip_ratio_iqr": _iqr(np.asarray(clip_ratios)),
        "actual_over_quoted_median": float(np.median(actual_ratios)) if actual_ratios else float("nan"),
        "actual_over_quoted_iqr": _iqr(np.asarray(actual_ratios)),
        "membership_diff_max": int(max((r["diff"] for r in membership_rows), default=0)),
        "membership_sample": membership_rows[:3],
    }


def e12_trace() -> dict[str, Any]:
    """Static code-path trace + membership on 5 representative fields."""
    return {
        "trace": {
            "comp_resid_clip_before_sem": (
                "photometry_core.py:3430-3438 comp_resid from comp_pairs minus comp_ref_map; "
                "NO sigma-clip on comp_resid before ensemble_sem_mag_from_residuals"
            ),
            "comp_ref_map": (
                "photometry_core.py:3382-3390 median of ALL finite frames per comp (not clipped)"
            ),
            "comp_pairs_vs_ens_med": (
                "photometry_core.py:3392-3418 same comp_pairs list feeds flux sum (ens_med) and comp_resid"
            ),
            "stability_exclusion": (
                "check_comparison_stability photometry_core.py:3007-3040 MAD p2p filter marks comps "
                "excluded BEFORE ensemble_normalize; excluded comps drop from BOTH flux and SEM"
            ),
            "zp_sigma_clip": (
                "photometry_core.py:3468-3486 iterative 3-sigma clip on ZP offsets affects mag_calib "
                "ONLY, not ensemble_scatter"
            ),
            "temporal_binning": (
                "temporal_bin_comp_lc photometry_core.py:2567-2623 smooths comp LC before stability; "
                "not clipping but reduces high-frequency scatter feeding p2p"
            ),
        },
        "file_lines": [
            "photometry_core.py:3430-3438 comp_resid / ensemble_scatter",
            "photometry_core.py:3382-3390 comp_ref_map",
            "photometry_core.py:3340-3365 good_ids selection from comp_quality",
            "photometry_core.py:9493-9525 production caller chain",
            "sigma_floor_core.py:37-49 ensemble_sem_mag_from_residuals (std/c4/sqrt(n), no clip)",
        ],
    }


def main() -> int:
    cfg = AppConfig()
    OUT.mkdir(parents=True, exist_ok=True)
    ps = DRAFT / "platesolve" / SETUP
    phot = ps / "photometry"
    lights = DRAFT / "detrended_aligned" / "lights" / SETUP
    comp_all = pd.read_csv(
        phot / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    csv_files = sorted(lights.glob("proc_*.csv"))
    all_frames_stub = pd.DataFrame({"catalog_id": [], "bjd": []})

    target_ids = sorted(
        {
            str(x).strip()
            for x in comp_all["target_catalog_id"].astype(str)
            if str(x).strip()
        }
    )
    check_targets = [t for t in target_ids if CHECK_CID not in t]  # fields hosting check star
    # Use targets that have check_kmag sidecar
    fields = [
        t
        for t in check_targets
        if (phot / "lightcurves" / f"check_kmag_{t}.csv").is_file()
    ]

    field_rows: list[dict[str, Any]] = []
    for t in fields:
        row = _field_pipeline(
            target_cid=t,
            comp_df=comp_all,
            csv_files=csv_files,
            cfg=cfg,
            all_frames_stub=all_frames_stub,
        )
        if not row.get("skip"):
            field_rows.append(row)

    clip_pool = np.asarray([r["clip_ratio_median"] for r in field_rows], dtype=np.float64)
    actual_pool = np.asarray([r["actual_over_quoted_median"] for r in field_rows], dtype=np.float64)

    # T3 single-field check star LC stats
    t3_lc = pd.read_csv(
        REPO / "tmp" / "wide_err_w1w2" / "diag_check_lc" / T3_FIELD / f"lightcurve_{CHECK_CID}.csv"
    )
    m = pd.to_numeric(t3_lc["mag_calib_final"], errors="coerce").to_numpy(dtype=np.float64)
    e = pd.to_numeric(t3_lc["err"], errors="coerce").to_numpy(dtype=np.float64)
    ep = pd.to_numeric(t3_lc.get("err_photon"), errors="coerce").to_numpy(dtype=np.float64)
    es = pd.to_numeric(t3_lc.get("err_sem_rel"), errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
    sig = _mad_sigma_arr(m[ok])
    em = float(np.median(e[ok]))

    # E1.2 membership on 5 representative fields
    rep = [T3_FIELD] + [r["target_cid"] for r in field_rows[:4] if r["target_cid"] != T3_FIELD]
    e12_membership: list[dict[str, Any]] = []
    for t in rep[:5]:
        fr = next((r for r in field_rows if r["target_cid"] == t), None)
        if fr:
            e12_membership.append(
                {
                    "target_cid": t,
                    "membership_diff_max": fr["membership_diff_max"],
                    "sample": fr["membership_sample"],
                }
            )

    out = {
        "E1_1": {
            "gain_retraction": {
                "m2_issue": "sky PTC on unflat-fielded raw frames; PRNU + vignetting inflate variance",
                "m1_issue": "60 s dark subtracted from 0.15 s flats (pedestal mismatch)",
                "science_bound": "faint5 scatter 1.119*201=225 mmag vs phot floor 200*sqrt(3.17/0.96)=364 mmag",
                "g_lower_bound": "g >= 2.50 e-/ADU from 200*sqrt(3.17/g) <= 225",
                "gain_used_consistent": "gain_used 3.17 >= 2.50 bound; not implicated",
                "relative_survives": "g_eff/g1 ~ 0.95 supports SUM binning (relative only)",
            },
            "m4_check_star_correction": {
                "field": T3_FIELD,
                "label": "single T3 bright-representative field (NOT W1 median over 163 fields)",
                "ratio_orig": sig / em,
                "median_ep_mmag": float(np.median(ep[ok]) * MAG_ERR_SCALE),
                "median_es_mmag": float(np.median(es[ok]) * MAG_ERR_SCALE),
                "w1_median_ratio_total_robust": 1.828,
            },
        },
        "E1_2": {**e12_trace(), "membership_5_fields": e12_membership},
        "E1_3": {
            "n_fields": len(field_rows),
            "sem_unclipped_over_production_median": float(np.median(clip_pool)),
            "sem_unclipped_over_production_iqr": _iqr(clip_pool),
            "clip_present": "stability exclusion + n_comp_max cap; NOT sigma-clip on comp_resid",
        },
        "E1_4": {
            "n_fields": len(field_rows),
            "actual_over_quoted_median": float(np.median(actual_pool)),
            "actual_over_quoted_iqr": _iqr(actual_pool),
            "definition_actual": "Honeycutt SEM of (comp - ens_med) per frame, ens_med = flux-sum",
            "definition_quoted": "production ensemble_scatter from (comp - comp_ref_median_night)",
        },
        "field_rows_sample": field_rows[:8],
    }
    (OUT / "wide_err_e1.json").write_text(json.dumps(out, indent=2) + "\n", encoding="ascii")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
