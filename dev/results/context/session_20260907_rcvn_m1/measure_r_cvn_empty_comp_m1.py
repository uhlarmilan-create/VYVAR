# -*- coding: ascii -*-
"""R-CVN-EMPTY-COMP-M1: read-only measurement. Session evidence only.

Does not write Archive. Does not instantiate AppConfig (mkdir risk).
Calls production filter functions with a SimpleNamespace cfg.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
SRC = REPO / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from comp_selection_per_target import (  # noqa: E402
    _filter_comp_candidates_spatial_static,
    _resolve_target_color_for_comp_selection,
)
from k2_extinction import (  # noqa: E402
    SMITH_K2_NATIVE,
    SLOPE_GR_PER_BPRP,
    computed_k2_bprp_for_token,
    resolve_k2_bprp_value,
)

SETUP = "NoFilter_60_2"
LIVE_PS = REPO / "Archive" / "Drafts" / "draft_000516" / "platesolve" / SETUP
LIVE_MS = LIVE_PS / "masterstars_full_match.csv"
LIVE_LC = LIVE_PS / "photometry" / "lightcurves"
LIVE_COMPS = LIVE_PS / "photometry" / "comparison_stars_per_target.csv"
LIVE_LIGHTS = REPO / "Archive" / "Drafts" / "draft_000516" / "detrended_aligned" / "lights" / SETUP
VSX_DB = REPO / "VSX" / "vyvar_vsx_local_v2.db"
PIN_CSV = REPO / "dev" / "validation" / "pinned_ensembles.csv"
OUT = Path(__file__).resolve().parent

TASK_IDS = [
    "1498842882207281152",
    "1499842372636900992",
    "1500410236033012352",
]
R_CVN = "1496795041799526400"
EMPTY_DROP_IDS = [
    "1497245497969274240",
    "1498425548825498112",
    "1497227287309482624",
]
ALL_IDS = TASK_IDS + [R_CVN] + EMPTY_DROP_IDS
G4 = {"csv": "bfa24039", "fits": "13e77cf8", "epsf": "172f9540"}
CAP = 0.79
TIER_CFG = (0.15, 0.30, 0.55, 1.10)  # config.json comp_color_tiers
CHIP = (2183, 1498)  # C-EXPORT-GAP Phase-1 EDGE CHECK
MARGIN = 50
MAG_DIFF = 1.5
N_MIN = 3
MAX_COMP_RMS = 0.10  # config.json phase01_comparison_max_comp_rms


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def vsx_nearest(ra: float, dec: float) -> dict:
    if not VSX_DB.is_file() or not (math.isfinite(ra) and math.isfinite(dec)):
        return {}
    con = sqlite3.connect(f"file:{VSX_DB.as_posix()}?mode=ro", uri=True)
    d = 30.0 / 3600.0
    rows = con.execute(
        "SELECT name, var_type, mag_max, mag_min, ra_deg, dec_deg FROM vsx_data "
        "WHERE ra_deg BETWEEN ? AND ? AND dec_deg BETWEEN ? AND ?",
        (
            ra - d / max(math.cos(math.radians(dec)), 0.2),
            ra + d / max(math.cos(math.radians(dec)), 0.2),
            dec - d,
            dec + d,
        ),
    ).fetchall()
    con.close()
    if not rows:
        return {"vsx_name": "", "vsx_type": "", "vsx_sep_arcsec": float("nan")}
    best = None
    best_sep = 1e9
    for name, vtype, mx, mn, vra, vdec in rows:
        sep = math.hypot((vra - ra) * math.cos(math.radians(dec)), vdec - dec) * 3600.0
        if sep < best_sep:
            best_sep = sep
            best = {
                "vsx_name": name,
                "vsx_type": vtype,
                "vsx_mag_max": mx,
                "vsx_mag_min": mn,
                "vsx_sep_arcsec": sep,
            }
    return best or {}


def g4() -> dict:
    rows = {
        "csv": LIVE_MS,
        "fits": LIVE_PS / "MASTERSTAR.fits",
        "epsf": LIVE_PS / "masterstar_epsf.fits",
    }
    out = {}
    ok = True
    for k, p in rows.items():
        d = _sha(p) if p.is_file() else ""
        v = "PASS" if d.startswith(G4[k]) else "FAIL"
        if v != "PASS":
            ok = False
        out[k] = {"sha256": d, "verdict": v}
    out["pass"] = ok
    return out


def make_cfg() -> SimpleNamespace:
    def limits() -> list[float]:
        return list(TIER_CFG)

    return SimpleNamespace(
        comp_max_delta_bprp=CAP,
        comp_tier_bprp_limits=limits,
        phase01_comparison_max_mag_diff_absolute=3.0,
        phase01_comparison_n_comp_min=N_MIN,
        snr_cog_isolation_fwhm=3.0,
        vsx_local_db_path="",
        gaia_db_path="",
    )


def funnel_one(ms: pd.DataFrame, cid: str, cfg: SimpleNamespace, vt_ids: set[str]) -> dict:
    row = ms[ms["catalog_id"].astype(str) == cid]
    if row.empty:
        return {"catalog_id": cid, "error": "missing_in_masterstars"}
    target = row.iloc[0]
    n_field = int(len(ms))
    ctx = _resolve_target_color_for_comp_selection(
        target,
        vsx_local_db_path=None,
        gaia_db_path=None,
        cfg=cfg,
    )
    t_bprp = float(ctx["target_bprp_eff"])
    mag_t = float(ctx["mag_t"])
    work = ms.copy()
    work, base, det = _filter_comp_candidates_spatial_static(
        work,
        ra_t=float(target["ra_deg"]),
        dec_t=float(target["dec_deg"]),
        mag_t=mag_t,
        target_cid=cid,
        target_bprp_eff=t_bprp,
        max_delta_bprp_cfg=CAP,
        max_dist_deg=1.0,
        min_dist_arcsec=60.0,
        exclude_gaia_nss=True,
        exclude_gaia_extobj=True,
        chip_fw=CHIP[0],
        chip_fh=CHIP[1],
        chip_interior_margin_px=MARGIN,
        variable_target_catalog_ids=vt_ids,
        use_pixel_dist=True,
        x_t=float(pd.to_numeric(target.get("x"), errors="coerce")),
        y_t=float(pd.to_numeric(target.get("y"), errors="coerce")),
        plate_scale_arcsec=9.774,
    )
    n_base = int(base.sum())
    n_det = int(det.sum())
    n_spatial = int((base | det).sum())
    # Live 516 masterstars lacks D3 columns (vy_identity_gate, gaia_dao_resid_px,
    # snr_ap_pixscaled). C-EXPORT-GAP sandbox log: D3 n_in=n_out (no-op) on this
    # field. _adaptive_mag_filter is also a no-op (COMP-ADMIT-03, :290-304).
    pre = work.loc[base | det].copy()
    used_tol = MAG_DIFF
    n_mag = int(len(pre))
    n_d3 = n_spatial
    if n_mag and math.isfinite(t_bprp) and "bp_rp" in pre.columns:
        dlt = (pd.to_numeric(pre["bp_rp"], errors="coerce") - t_bprp).abs()
        pre = pre.copy()
        pre["_d_bprp"] = dlt
    else:
        pre = pre.copy()
        pre["_d_bprp"] = np.nan
    n_t1 = int((pre["_d_bprp"] <= TIER_CFG[0]).sum()) if n_mag else 0
    n_t2 = int((pre["_d_bprp"] <= TIER_CFG[1]).sum()) if n_mag else 0
    n_t3 = int((pre["_d_bprp"] <= TIER_CFG[2]).sum()) if n_mag else 0
    n_cap = int((pre["_d_bprp"] <= CAP).sum()) if n_mag else 0
    finite = pre[np.isfinite(pre["_d_bprp"])] if n_mag else pre
    dist = {}
    if len(finite):
        q = finite["_d_bprp"].quantile([0, 0.25, 0.5, 0.75, 1.0])
        dist = {f"p{int(k * 100)}": float(v) for k, v in q.items()}
        dist["n_finite_color"] = int(len(finite))
        dist["n_nan_color"] = int(n_mag - len(finite))
        dist["min_d_bprp"] = float(finite["_d_bprp"].min())
    kill = "color_cap_0.79_empties" if n_cap == 0 and n_mag > 0 else (
        "empty_after_mag" if n_mag == 0 else "cap_does_not_empty"
    )
    return {
        "catalog_id": cid,
        "n_field": n_field,
        "n_after_base_sat_var_geom_mindist": n_base,
        "n_det_extra": n_det,
        "n_after_spatial_base_or_det": n_spatial,
        "n_after_d3": n_d3,
        "d3_note": "live_MS_missing_D3_cols; treated as no-op per C-EXPORT-GAP log",
        "n_after_d3_and_adaptive_mag": n_mag,
        "mag_filter_note": "no-op COMP-ADMIT-03 comp_selection_per_target.py:290-304",
        "used_mag_tol": used_tol,
        "target_bprp_eff": t_bprp,
        "target_mag_used": mag_t,
        "n_within_t1_0.15": n_t1,
        "n_within_t2_0.30": n_t2,
        "n_within_t3_0.55": n_t3,
        "n_within_cap_0.79": n_cap,
        "n_unbounded_t4": n_mag,
        "killing_stage_color_path": kill,
        "color_dist_pre_cap": dist,
        "_pre": pre,
    }


def nearest_color(pre: pd.DataFrame, k: int) -> pd.DataFrame:
    if pre is None or pre.empty or "_d_bprp" not in pre.columns:
        return pre.iloc[0:0].copy() if pre is not None else pd.DataFrame()
    ok = pre[np.isfinite(pre["_d_bprp"])].sort_values("_d_bprp", kind="mergesort")
    return ok.head(k).copy()


def proc_snr_map(cids: list[str]) -> dict[str, float]:
    wanted = {str(c) for c in cids}
    if not wanted:
        return {}
    acc: dict[str, list[float]] = {c: [] for c in wanted}
    sample = next(iter(sorted(LIVE_LIGHTS.glob("proc_*.csv"))), None)
    if sample is None:
        return {c: float("nan") for c in wanted}
    hdr = pd.read_csv(sample, nrows=0)
    cols = [
        c
        for c in (
            "catalog_id",
            "flux",
            "sigma_bkg_ap",
            "psf_flux",
            "psf_flux_err",
            "psf_snr",
        )
        if c in hdr.columns
    ]
    for p in sorted(LIVE_LIGHTS.glob("proc_*.csv")):
        df = pd.read_csv(p, usecols=cols, dtype={"catalog_id": str})
        sub = df[df["catalog_id"].astype(str).isin(wanted)]
        for _, r in sub.iterrows():
            cid = str(r["catalog_id"])
            snr = float(pd.to_numeric(r.get("psf_snr"), errors="coerce"))
            if math.isfinite(snr) and snr > 0:
                acc[cid].append(snr)
                continue
            f = float(pd.to_numeric(r.get("psf_flux"), errors="coerce"))
            e = float(pd.to_numeric(r.get("psf_flux_err"), errors="coerce"))
            if math.isfinite(f) and math.isfinite(e) and e > 0 and f > 0:
                acc[cid].append(f / e)
                continue
            f = float(pd.to_numeric(r.get("flux"), errors="coerce"))
            e = float(pd.to_numeric(r.get("sigma_bkg_ap"), errors="coerce"))
            if math.isfinite(f) and math.isfinite(e) and e > 0:
                acc[cid].append(f / e)
    return {c: float(np.median(v)) if v else float("nan") for c, v in acc.items()}


def exportable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Local copy of export_reports._select_export_lc_rows (finite mag + BJD)."""
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    mag_col = "mag_calib"
    if "mag_calib_final" in work.columns:
        mag_col = "mag_calib_final"
    mag = pd.to_numeric(work.get(mag_col), errors="coerce")
    bjd = pd.to_numeric(work.get("bjd"), errors="coerce")
    finite = mag.notna() & np.isfinite(mag.to_numpy(dtype=float)) & bjd.notna() & np.isfinite(
        bjd.to_numpy(dtype=float)
    )
    if "flag" in work.columns:
        fl = work["flag"].astype(str).str.strip().str.lower()
        good = fl.isin(("normal", "")) | fl.isna()
        bad = fl.isin(("no_data", "saturated", "edge_fail", "nondetection"))
        mask = finite & (good | ~bad)
        out = work.loc[mask].copy()
        if out.empty and finite.any():
            out = work.loc[finite].copy()
        return out
    return work.loc[finite].copy()


def lc_report(cid: str) -> dict:
    path = LIVE_LC / f"lightcurve_{cid}.csv"
    if not path.is_file():
        return {"catalog_id": cid, "path": str(path.as_posix()), "exists": False}
    raw = path.read_text(encoding="utf-8", errors="replace")
    has_skip = "skip_reason" in raw[:8000] or "ac_skip_reason" in raw[:8000]
    df = pd.read_csv(path, comment="#")
    exp = exportable_rows(df)
    empty_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").isna().all()]
    flags = df["flag"].astype(str).value_counts().to_dict() if "flag" in df.columns else {}
    return {
        "catalog_id": cid,
        "path": str(path.as_posix()),
        "exists": True,
        "n_rows": int(len(df)),
        "n_exportable": int(len(exp)),
        "n_finite_mag_final": int(pd.to_numeric(df.get("mag_calib_final"), errors="coerce").notna().sum())
        if "mag_calib_final" in df.columns
        else 0,
        "flag_counts": flags,
        "all_nan_numeric_cols": empty_cols,
        "skip_reason_in_file": has_skip,
        "header_comment": raw.startswith("#"),
        "k2_source_mode": str(df["k2_source"].iloc[0]) if "k2_source" in df.columns and len(df) else "",
    }


def pin_report(pin: pd.DataFrame, ms: pd.DataFrame, cid: str, t_bprp: float) -> dict:
    sub = pin[pin["target_catalog_id"].astype(str) == cid]
    comps = []
    for _, r in sub.iterrows():
        cc = str(r.get("comp_catalog_id", ""))
        hit = ms[ms["catalog_id"].astype(str) == cc]
        bp = float(pd.to_numeric(hit.iloc[0]["bp_rp"], errors="coerce")) if not hit.empty else float("nan")
        g = float(pd.to_numeric(hit.iloc[0]["phot_g_mean_mag"], errors="coerce")) if not hit.empty else float("nan")
        dlt = abs(bp - t_bprp) if math.isfinite(bp) and math.isfinite(t_bprp) else float("nan")
        comps.append({
            "comp_catalog_id": cc,
            "comp_tier_pin": int(pd.to_numeric(r.get("comp_tier"), errors="coerce") or 4),
            "bp_rp": bp,
            "phot_g_mean_mag": g,
            "d_bprp": dlt,
        })
    return {
        "catalog_id": cid,
        "n_pin": int(len(sub)),
        "pinned": int(len(sub)) > 0,
        "comps": comps,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    g4v = g4()
    ms = pd.read_csv(LIVE_MS, dtype={"catalog_id": str})
    vt = set()
    if "vsx_known_variable" in ms.columns:
        vt = set(ms.loc[ms["vsx_known_variable"].astype(bool), "catalog_id"].astype(str))
    cfg = make_cfg()
    pin = pd.read_csv(PIN_CSV, dtype=str) if PIN_CSV.is_file() else pd.DataFrame()
    live_comps = (
        pd.read_csv(LIVE_COMPS, dtype={"catalog_id": str, "target_catalog_id": str})
        if LIVE_COMPS.is_file()
        else pd.DataFrame()
    )

    m1 = []
    for cid in ALL_IDS:
        hit = ms[ms["catalog_id"] == cid]
        rec = {"catalog_id": cid, "in_masterstars": not hit.empty, "is_r_cvn": cid == R_CVN}
        if hit.empty:
            m1.append(rec)
            continue
        r = hit.iloc[0]
        rec.update({
            "name_col": str(r.get("name", "")),
            "ra_deg": float(r["ra_deg"]),
            "dec_deg": float(r["dec_deg"]),
            "phot_g_mean_mag": float(pd.to_numeric(r.get("phot_g_mean_mag"), errors="coerce")),
            "bp_rp": float(pd.to_numeric(r.get("bp_rp"), errors="coerce")),
            "x": float(pd.to_numeric(r.get("x"), errors="coerce")),
            "y": float(pd.to_numeric(r.get("y"), errors="coerce")),
            "vsx_known_variable": bool(r.get("vsx_known_variable", False)) if "vsx_known_variable" in r.index else None,
            "cohort": (
                "task_export_fail" if cid in TASK_IDS else (
                    "r_cvn" if cid == R_CVN else "phase2a_empty_comp_drop"
                )
            ),
        })
        rec.update(vsx_nearest(rec["ra_deg"], rec["dec_deg"]))
        rec["pinned_n"] = int((pin["target_catalog_id"] == cid).sum()) if not pin.empty else 0
        if not live_comps.empty:
            lc = live_comps[live_comps["target_catalog_id"] == cid]
            rec["live_n_comps"] = int(lc["catalog_id"].nunique()) if not lc.empty else 0
            rec["live_selected_tier"] = (
                ",".join(sorted(lc["selected_tier"].astype(str).unique())) if not lc.empty and "selected_tier" in lc.columns else ""
            )
        m1.append(rec)
    pd.DataFrame(m1).to_csv(OUT / "m1_identity.csv", index=False)

    funnels = []
    pres = {}
    pins = []
    for cid in ALL_IDS:
        fun = funnel_one(ms, cid, cfg, vt)
        pre = fun.pop("_pre", pd.DataFrame())
        pres[cid] = pre
        funnels.append(fun)
        pins.append(pin_report(pin, ms, cid, float(fun.get("target_bprp_eff", float("nan")))))
    pd.DataFrame(funnels).to_csv(OUT / "m2_funnel.csv", index=False)
    (OUT / "m2_pins.json").write_text(json.dumps(pins, indent=2, default=str), encoding="utf-8")

    need: list[str] = []
    nearest_frames = []
    for cid in ALL_IDS:
        pre = pres.get(cid, pd.DataFrame())
        top8 = nearest_color(pre, 8)
        need.extend(str(x) for x in top8.get("catalog_id", pd.Series(dtype=str)).astype(str).tolist())
        nearest_frames.append((cid, top8))
    snr = proc_snr_map(sorted(set(need)))
    m3_rows = []
    for cid, top8 in nearest_frames:
        for k in (3, 5, 8):
            for _, r in top8.head(k).iterrows():
                cc = str(r.get("catalog_id", ""))
                m3_rows.append({
                    "target_id": cid,
                    "k": k,
                    "catalog_id": cc,
                    "d_bprp": float(r["_d_bprp"]),
                    "bp_rp": float(pd.to_numeric(r.get("bp_rp"), errors="coerce")),
                    "phot_g_mean_mag": float(pd.to_numeric(r.get("phot_g_mean_mag"), errors="coerce")),
                    "mag": float(pd.to_numeric(r.get("mag"), errors="coerce")),
                    "snr_median_proc": snr.get(cc, float("nan")),
                    "snr_source": "psf_snr_else_flux_over_sigma_bkg_ap",
                })
    pd.DataFrame(m3_rows).to_csv(OUT / "m3_nearest_color.csv", index=False)

    m4 = [lc_report(cid) for cid in TASK_IDS + [R_CVN] + EMPTY_DROP_IDS]
    pd.DataFrame(m4).to_csv(OUT / "m4_lc_csv.csv", index=False)

    k2_val, k2_src = resolve_k2_bprp_value(None, "NoFilter_60_2")
    am = []
    for cid in [R_CVN] + TASK_IDS:
        p = LIVE_LC / f"lightcurve_{cid}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, comment="#")
        a = pd.to_numeric(df.get("airmass"), errors="coerce")
        a = a[np.isfinite(a)]
        if len(a):
            am.append({
                "catalog_id": cid,
                "am_min": float(a.min()),
                "am_max": float(a.max()),
                "am_span": float(a.max() - a.min()),
            })
    am_df = pd.DataFrame(am)
    am_span = float(am_df["am_span"].max()) if not am_df.empty else float("nan")
    # Hypothetical literature k2 if this night had been Sloan g/r (NOT the actual band).
    k2_g = computed_k2_bprp_for_token("g")  # Sloan g (lowercase); uppercase R/G are Johnson
    k2_r = computed_k2_bprp_for_token("r")
    m5 = []
    for row in m3_rows:
        if int(row["k"]) != 8:
            continue
        d = float(row["d_bprp"])
        term_actual = 0.0
        term_g = abs(float(k2_g)) * am_span * d if math.isfinite(float(k2_g or float("nan"))) and math.isfinite(am_span) else float("nan")
        term_r = abs(float(k2_r)) * am_span * d if math.isfinite(float(k2_r or float("nan"))) and math.isfinite(am_span) else float("nan")
        m5.append({
            "target_id": row["target_id"],
            "comp_id": row["catalog_id"],
            "d_bprp": d,
            "k2_bprp_nofilter": None if not math.isfinite(float(k2_val)) else float(k2_val),
            "k2_source_nofilter": str(k2_src),
            "airmass_span": am_span,
            "k2_term_mag_nofilter": term_actual,
            "k2_bprp_if_sloan_g": k2_g,
            "k2_term_mag_if_sloan_g": term_g,
            "k2_bprp_if_sloan_r": k2_r,
            "k2_term_mag_if_sloan_r": term_r,
            "smith_k2_native_g": SMITH_K2_NATIVE.get("G"),
            "slope_gr_per_bprp": SLOPE_GR_PER_BPRP,
            "note": "NoFilter -> band_failsafe_clear -> literature k2 NONE; term=0. Sloan g/r columns are counterfactual only.",
        })
    pd.DataFrame(m5).to_csv(OUT / "m5_k2_term.csv", index=False)
    am_df.to_csv(OUT / "m5_airmass.csv", index=False)

    live_comp_rows = []
    if not live_comps.empty:
        for cid in ALL_IDS:
            sub = live_comps[live_comps["target_catalog_id"] == cid]
            for _, r in sub.iterrows():
                live_comp_rows.append({
                    "target_catalog_id": cid,
                    "comp_catalog_id": str(r.get("catalog_id", "")),
                    "delta_bprp_abs": float(pd.to_numeric(r.get("delta_bprp_abs"), errors="coerce")),
                    "comp_tier": r.get("comp_tier"),
                    "selected_tier": r.get("selected_tier"),
                    "phot_g_mean_mag": float(pd.to_numeric(r.get("phot_g_mean_mag"), errors="coerce")),
                    "comp_rms": float(pd.to_numeric(r.get("comp_rms"), errors="coerce")),
                    "selection_note": str(r.get("selection_note", ""))[:240],
                })
    pd.DataFrame(live_comp_rows).to_csv(OUT / "m2_live_comps.csv", index=False)

    summary = {
        "task": "R-CVN-EMPTY-COMP-M1",
        "extracted_utc": _now(),
        "g4": g4v,
        "refute": {
            "r_cvn_id": R_CVN,
            "r_cvn_in_task_ids": False,
            "task_ids_are_export_failures_not_empty_comp": True,
            "empty_comp_drop_ids": EMPTY_DROP_IDS,
            "empty_comp_kill": "pinned_ensemble n_survivors=2 < n_min=3 after rms_violation on 1500467303261764096 (not color cap)",
            "tier_limits_config_json": list(TIER_CFG),
            "architect_remembered_tiers": [0.25, 0.48, 0.79],
            "spatial_filter_does_not_apply_cap": "comp_selection_per_target.py:406-426 COMP-ADMIT-03",
            "base_mask_still_applies_min_dist": "comp_selection_per_target.py:509-510",
            "ladder_last_rung": "comp_max_delta_bprp 0.79 (photometry_comp.py:1138-1157); T4 1.10 not on ladder",
            "r_cvn_phase1": "pinned 8-comp overlay; dBP-RP median 4.852 (cexportgap log:70-73)",
            "architect_cap_cite_1691_1750": "REFUTED as color-cap site; those lines are derived global-pool admission",
        },
        "k2": {"value": k2_val, "source": str(k2_src), "airmass_span": am_span, "k2_g": k2_g, "k2_r": k2_r},
        "m1": m1,
        "m2": funnels,
        "m2_pins": pins,
        "m4": m4,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "g4": g4v["pass"],
        "m1_vsx": [(r.get("catalog_id"), r.get("vsx_name"), r.get("vsx_type"), r.get("bp_rp"), r.get("pinned_n")) for r in m1],
        "m2_kill": [(f["catalog_id"], f.get("killing_stage_color_path"), f.get("n_within_cap_0.79"), f.get("n_after_d3_and_adaptive_mag"), f.get("target_bprp_eff")) for f in funnels],
        "m4_exportable": [(x["catalog_id"], x.get("exists"), x.get("n_rows"), x.get("n_exportable")) for x in m4],
        "k2": str(k2_src),
        "n_nearest": len(m3_rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
