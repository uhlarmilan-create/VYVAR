# -*- coding: ascii -*-
"""C3-STOP sandbox predictions: P-C3-1/2/3. No live Archive writes."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))

from comp_rms_loo import (  # noqa: E402
    COMP_RMS_LOO_PHOTON_K_DEFAULT,
    compute_loo_mag_rms_map,
    loo_ceiling_mag,
    photon_sigma_mag,
)
from masterstar_gaia_accounting import _norm_cid  # noqa: E402
from pipeline import _annotate_masterstars_flux_zones  # noqa: E402

OUTDIR = Path(__file__).resolve().parent
B3 = ROOT / "dev" / "results" / "session_20260825_sel_ghost_01_b3"
R2 = B3 / "t3_r2"
T4 = B3 / "t4_520"
LIVE516 = ROOT / "Archive" / "Drafts" / "draft_000516"
LIVE520 = ROOT / "Archive" / "Drafts" / "draft_000520"
SETUP516 = "NoFilter_60_2"
SETUP520 = "g_60_4"
K = COMP_RMS_LOO_PHOTON_K_DEFAULT
ABS_MAX = 0.1
SEVEN = [
    "1112113680298377344",  # G=7.63
    "1111920204908702336",
    "1112110695298081664",
    "1111749157833870208",
    "1112121862213003648",
    "1112121067641532160",
    "1111737033143440768",  # G=13.87
]
G13 = "1111737033143440768"
G763 = "1112113680298377344"
V0612 = "1111749368289526912"


def cid(v: object) -> str:
    return _norm_cid(v)


def load_proc_dir(phot: Path) -> tuple[list[Path], dict[str, pd.DataFrame]]:
    paths: list[Path] = []
    cache: dict[str, pd.DataFrame] = {}
    for p in sorted(phot.glob("proc_*.csv")):
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue
        paths.append(p)
        cache[str(p)] = df
    return paths, cache


def mag_from_flux(f: float) -> float:
    if not (math.isfinite(f) and f > 0):
        return float("nan")
    return -2.5 * math.log10(f)


def ensemble_lc_rms(target: str, comps: list[str], cache: dict[str, pd.DataFrame]) -> float:
    dmags: list[float] = []
    for df in cache.values():
        col = "catalog_id" if "catalog_id" in df.columns else "name"
        flux_col = "dao_flux" if "dao_flux" in df.columns else "flux"
        if col not in df.columns or flux_col not in df.columns:
            continue
        work = df[[col, flux_col]].copy()
        work["_cid"] = work[col].map(cid)
        work["_flux"] = pd.to_numeric(work[flux_col], errors="coerce")
        fmap = {
            str(a): float(b)
            for a, b in zip(work["_cid"], work["_flux"])
            if str(a) and math.isfinite(float(b)) and float(b) > 0
        }
        mt = mag_from_flux(fmap.get(target, float("nan")))
        others = [mag_from_flux(fmap[c]) for c in comps if c in fmap]
        others = [m for m in others if math.isfinite(m)]
        if not math.isfinite(mt) or len(others) < 3:
            continue
        dmags.append(mt - float(np.median(np.asarray(others, dtype=np.float64))))
    if len(dmags) < 5:
        return float("nan")
    a = np.asarray(dmags, dtype=np.float64)
    a = a - float(np.median(a))
    return float(np.sqrt(np.mean(a * a)))


def old_relflux_flag_ids(
    pool_ids: set[str], cache: dict[str, pd.DataFrame], outlier_sigma: float = 3.0
) -> set[str]:
    """Replicate pre-C3 suspected_variables: RMS of detrended rel-flux vs 1.0."""
    flux_map: dict[str, list[float]] = {x: [] for x in pool_ids}
    n_frames = 0
    for df in cache.values():
        col = "catalog_id" if "catalog_id" in df.columns else "name"
        flux_col = "dao_flux" if "dao_flux" in df.columns else "flux"
        if col not in df.columns or flux_col not in df.columns:
            continue
        work = df[[col, flux_col]].copy()
        work["_cid"] = work[col].map(cid)
        work["_flux"] = pd.to_numeric(work[flux_col], errors="coerce")
        work = work[work["_cid"].isin(pool_ids) & work["_flux"].gt(0)]
        if work.empty:
            continue
        n_frames += 1
        med = float(work["_flux"].median())
        if not (math.isfinite(med) and med > 0):
            continue
        for sid, fl in zip(work["_cid"], work["_flux"]):
            flux_map[str(sid)].append(float(fl) / med)
    min_f = max(3, int(n_frames * 0.3))
    rms_map: dict[str, float] = {}
    for sid, vals in flux_map.items():
        if len(vals) < min_f:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        t = np.arange(len(arr), dtype=np.float64)
        if arr.size >= 5:
            try:
                coeffs = np.polyfit(t, arr, 2)
                trend = np.polyval(coeffs, t)
                safe = np.where(np.abs(trend) > 1e-9, trend, 1.0)
                arr = arr / safe
                med_dt = float(np.median(arr))
                if math.isfinite(med_dt) and med_dt > 0:
                    arr = arr / med_dt
            except Exception:
                pass
        rms = float(np.sqrt(np.mean((arr - 1.0) ** 2)))
        if math.isfinite(rms):
            rms_map[sid] = rms
    if not rms_map:
        return set()
    arr = np.asarray(list(rms_map.values()), dtype=np.float64)
    med = float(np.median(arr))
    mad_raw = float(np.median(np.abs(arr - med)))
    mad_sigma = (mad_raw / 0.6745) if mad_raw > 0 else (float(np.std(arr)) / 0.6745 or 1e-9)
    thr = med + outlier_sigma * mad_sigma
    return {k for k, v in rms_map.items() if v > thr}


def main() -> None:
    out: dict = {"k": K, "abs_max": ABS_MAX}

    # --- 520 P-C3-1 ---
    ms520_path = T4 / "platesolve" / SETUP520 / "masterstars_full_match.csv"
    if not ms520_path.is_file():
        ms520_path = LIVE520 / "platesolve" / SETUP520 / "masterstars_full_match.csv"
    ms520 = pd.read_csv(ms520_path, low_memory=False)
    ms520["_cid"] = ms520["catalog_id"].map(cid) if "catalog_id" in ms520.columns else ms520["name"].map(cid)
    phot520 = T4 / "platesolve" / SETUP520 / "photometry"
    proc520_dir = phot520
    if not list(proc520_dir.glob("proc_*.csv")):
        # T4 may keep proc next to aligned lights
        alt = T4 / "detrended_aligned" / "lights" / SETUP520
        if alt.is_dir():
            proc520_dir = alt
        else:
            proc520_dir = LIVE520 / "platesolve" / SETUP520 / "photometry"
    paths520, cache520 = load_proc_dir(proc520_dir)
    if not cache520:
        live_phot = LIVE520 / "platesolve" / SETUP520 / "photometry"
        paths520, cache520 = load_proc_dir(live_phot)
        proc520_dir = live_phot
    loo520, basis520 = compute_loo_mag_rms_map(set(SEVEN), paths520, cache520)
    rows520 = []
    pass_ids = []
    for sid in SEVEN:
        row = ms520[ms520["_cid"] == sid]
        snr = float("nan")
        gmag = float("nan")
        peak = float("nan")
        zone = ""
        if not row.empty:
            r0 = row.iloc[0]
            for c in ("snr_ap_pixscaled", "snr_ap", "snr"):
                if c in row.columns:
                    snr = float(pd.to_numeric(r0[c], errors="coerce"))
                    if math.isfinite(snr) and snr > 0:
                        break
            gmag = float(pd.to_numeric(r0.get("phot_g_mean_mag", r0.get("mag")), errors="coerce"))
            peak = float(pd.to_numeric(r0.get("peak_max_adu", r0.get("peak_dao")), errors="coerce"))
            zone = str(r0.get("zone", "") or "")
        # C2 photon table fallback when MS lacks SNR
        c2_ph = {
            "1112113680298377344": 0.00127,
            "1111920204908702336": 0.00511,
            "1112110695298081664": 0.00632,
            "1111749157833870208": 0.00683,
            "1112121862213003648": 0.00697,
            "1112121067641532160": 0.00824,
            "1111737033143440768": 0.0252,
        }
        ph = photon_sigma_mag(snr)
        if not math.isfinite(ph):
            ph = c2_ph.get(sid, float("nan"))
            snr_used = 1.0857362047581294 / ph if math.isfinite(ph) and ph > 0 else float("nan")
        else:
            snr_used = snr
        loo = float(loo520.get(sid, float("nan")))
        try:
            ceil = loo_ceiling_mag(snr_used, k=K, abs_max=ABS_MAX)
            passes = math.isfinite(loo) and loo <= ceil
        except ValueError:
            ceil = float("nan")
            passes = False
        rec = {
            "catalog_id": sid,
            "G": gmag,
            "loo": loo,
            "snr": snr_used,
            "photon": ph,
            "ceil": ceil,
            "r": (loo / ph) if math.isfinite(loo) and math.isfinite(ph) and ph > 0 else float("nan"),
            "passes": passes,
            "peak": peak,
            "zone_on_disk": zone,
        }
        rows520.append(rec)
        if passes:
            pass_ids.append(sid)
    lc_rms = ensemble_lc_rms(V0612, pass_ids, cache520)
    out["p_c3_1"] = {
        "proc_dir": str(proc520_dir),
        "n_proc": len(cache520),
        "frames_basis": basis520,
        "n_pass": len(pass_ids),
        "pass_ids": pass_ids,
        "g13_selected": G13 in pass_ids,
        "g763_selected": G763 in pass_ids,
        "lc_rms": lc_rms,
        "rows": rows520,
        "hit_ensemble_ge5": len(pass_ids) >= 5,
        "hit_lc_rms_le_0p06": math.isfinite(lc_rms) and lc_rms <= 0.06,
        "hit_g13_not_selected": G13 not in pass_ids,
    }

    # ZONE-SAT-01 on 520 G=7.63 (in-memory, no write)
    sat_row = ms520[ms520["_cid"] == G763].copy()
    zone_out = None
    if not sat_row.empty:
        zdf = _annotate_masterstars_flux_zones(
            sat_row,
            noise_floor_adu=40.0,
            equipment_saturate_adu=65535.0,
            saturate_limit_adu_fallback=65535.0,
            n_stack=1,
            saturate_limit_fraction=1.0,
            sigma_px=40.0,
            sky_median_adu=float(pd.to_numeric(sat_row.iloc[0].get("sky_median_adu"), errors="coerce") or 1400.0),
            dao_detection_n_equiv=4.5,
            empirical_clip_adu=None,
        )
        zone_out = {
            "zone": str(zdf.iloc[0].get("zone", "")),
            "zone_peak_column": str(zdf.iloc[0].get("zone_peak_column", "")),
            "zone_sat_limit_used": zdf.iloc[0].get("zone_sat_limit_used"),
            "peak_max_adu": float(pd.to_numeric(sat_row.iloc[0].get("peak_max_adu"), errors="coerce")),
        }
    out["zone_520_g763"] = zone_out

    # --- 516 P-C3-2 ---
    ms516_path = R2 / "platesolve" / SETUP516 / "masterstars_full_match.csv"
    ms516 = pd.read_csv(ms516_path, low_memory=False)
    ms516["_cid"] = ms516["catalog_id"].map(cid) if "catalog_id" in ms516.columns else ms516["name"].map(cid)
    phot516 = R2 / "platesolve" / SETUP516 / "photometry"
    comp_csv = phot516 / "comparison_stars_per_target.csv"
    if not comp_csv.is_file():
        comp_csv = LIVE516 / "platesolve" / SETUP516 / "photometry" / "comparison_stars_per_target.csv"
    comps = pd.read_csv(comp_csv, low_memory=False)
    live_ids: set[str] = set()
    for col in comps.columns:
        if "comp" in col.lower() and "id" in col.lower():
            live_ids.update(cid(v) for v in comps[col] if cid(v))
    if "catalog_id" in comps.columns and not live_ids:
        live_ids = {cid(v) for v in comps["catalog_id"] if cid(v)}
    # also ensembles columns
    for col in comps.columns:
        s = comps[col].astype(str)
        if s.str.fullmatch(r"\d{15,20}").any():
            live_ids.update(cid(v) for v in comps[col] if cid(v) and cid(v).isdigit() and len(cid(v)) >= 15)

    proc516_dir = R2 / "detrended_aligned" / "lights" / SETUP516
    paths516, cache516 = load_proc_dir(proc516_dir)
    if not cache516:
        paths516, cache516 = load_proc_dir(phot516)

    # D5: snr_ap_pixscaled >= 10
    snr_col = "snr_ap_pixscaled" if "snr_ap_pixscaled" in ms516.columns else None
    d5_ids = []
    fail_rows = []
    live_list = sorted(x for x in live_ids if x)
    for sid in live_list:
        row = ms516[ms516["_cid"] == sid]
        snr = float("nan")
        if not row.empty and snr_col:
            snr = float(pd.to_numeric(row.iloc[0][snr_col], errors="coerce"))
        if math.isfinite(snr) and snr >= 10.0:
            d5_ids.append(sid)

    # candidate pool = D5 live comps (selector scores the pool)
    pool = set(d5_ids) if d5_ids else set(live_list)
    loo516, basis516 = compute_loo_mag_rms_map(pool, paths516, cache516)
    for sid in sorted(pool):
        row = ms516[ms516["_cid"] == sid]
        snr = float("nan")
        if not row.empty and snr_col:
            snr = float(pd.to_numeric(row.iloc[0][snr_col], errors="coerce"))
        loo = float(loo516.get(sid, float("nan")))
        ph = photon_sigma_mag(snr)
        try:
            ceil = loo_ceiling_mag(snr, k=K, abs_max=ABS_MAX) if math.isfinite(snr) else float("nan")
            passes = math.isfinite(loo) and math.isfinite(ceil) and loo <= ceil
        except ValueError:
            ceil = float("nan")
            passes = False
        r = (loo / ph) if math.isfinite(loo) and math.isfinite(ph) and ph > 0 else float("nan")
        if not passes:
            fail_rows.append(
                {
                    "catalog_id": sid,
                    "loo": loo,
                    "snr": snr,
                    "photon": ph,
                    "ceil": ceil,
                    "r": r,
                }
            )

    n_sat_r2 = int((ms516["zone"].astype(str).str.lower() == "saturated").sum()) if "zone" in ms516.columns else None
    # in-memory re-zone 516
    n_sat_new = None
    try:
        zall = _annotate_masterstars_flux_zones(
            ms516,
            noise_floor_adu=40.0,
            equipment_saturate_adu=65535.0,
            saturate_limit_adu_fallback=65535.0,
            n_stack=1,
            saturate_limit_fraction=1.0,
            sigma_px=40.0,
            sky_median_adu=1400.0,
            dao_detection_n_equiv=4.5,
            empirical_clip_adu=None,
        )
        n_sat_new = int((zall["zone"].astype(str).str.lower() == "saturated").sum())
    except Exception as exc:
        n_sat_new = str(exc)

    out["p_c3_2"] = {
        "n_live_ids_parsed": len(live_list),
        "n_d5": len(d5_ids),
        "n_proc": len(cache516),
        "frames_basis": basis516,
        "n_fail": len(fail_rows),
        "fails": fail_rows,
        "all_d5_pass": len(fail_rows) == 0 and len(d5_ids) > 0,
        "n_sat_on_disk": n_sat_r2,
        "n_sat_rezone": n_sat_new,
        "snr_col": snr_col,
        "comp_csv": str(comp_csv),
        "comp_cols": list(comps.columns)[:20],
    }

    # --- P-C3-3 suspected_variables ---
    old_flags = old_relflux_flag_ids(pool, cache516)
    loo_vals = np.asarray(list(loo516.values()), dtype=np.float64)
    new_flags: set[str] = set()
    if loo_vals.size:
        med = float(np.median(loo_vals))
        mad_raw = float(np.median(np.abs(loo_vals - med)))
        mad_sigma = (mad_raw / 0.6745) if mad_raw > 0 else (float(np.std(loo_vals)) / 0.6745 or 1e-9)
        thr = med + 3.0 * mad_sigma
        new_flags = {k for k, v in loo516.items() if v > thr}
    live_new_flagged = sorted(new_flags & pool)
    live_sv = phot516 / "suspected_variables.csv"
    if not live_sv.is_file():
        live_sv = LIVE516 / "platesolve" / SETUP516 / "photometry" / "suspected_variables.csv"
    n_old_file = None
    old_file_ids: set[str] = set()
    if live_sv.is_file():
        sv = pd.read_csv(live_sv, low_memory=False)
        n_old_file = int(len(sv))
        idc = "catalog_id" if "catalog_id" in sv.columns else sv.columns[0]
        old_file_ids = {cid(v) for v in sv[idc] if cid(v)}
    out["p_c3_3"] = {
        "n_old_relflux_flags_recomputed": len(old_flags),
        "n_new_loo_flags": len(new_flags),
        "n_live_comp_newly_flagged": len(live_new_flagged),
        "live_comp_newly_flagged": live_new_flagged,
        "n_old_file": n_old_file,
        "n_old_file_overlap_new": len(old_file_ids & new_flags) if old_file_ids else None,
        "sv_path": str(live_sv),
    }

    # 520 suspected
    pool520 = set(SEVEN)
    old520 = old_relflux_flag_ids(pool520, cache520)
    loo_a = np.asarray(list(loo520.values()), dtype=np.float64)
    new520: set[str] = set()
    if loo_a.size:
        med = float(np.median(loo_a))
        mad_raw = float(np.median(np.abs(loo_a - med)))
        mad_sigma = (mad_raw / 0.6745) if mad_raw > 0 else (float(np.std(loo_a)) / 0.6745 or 1e-9)
        thr = med + 3.0 * mad_sigma
        new520 = {k for k, v in loo520.items() if v > thr}
    sv520 = T4 / "platesolve" / SETUP520 / "photometry" / "suspected_variables.csv"
    n_sv520 = int(len(pd.read_csv(sv520))) if sv520.is_file() else None
    out["p_c3_3_520"] = {
        "n_old_recomputed": len(old520),
        "n_new": len(new520),
        "overlap": sorted(old520 & new520),
        "n_old_file": n_sv520,
        "new_ids": sorted(new520),
        "old_ids": sorted(old520),
    }

    dest = OUTDIR / "c3_pred.json"
    dest.write_text(json.dumps(out, indent=2, default=str), encoding="ascii")
    print(json.dumps({k: out[k] if k not in ("p_c3_1",) else {kk: out[k][kk] for kk in out[k] if kk != "rows"} for k in out}, indent=2, default=str))
    print("wrote", dest)


if __name__ == "__main__":
    main()
