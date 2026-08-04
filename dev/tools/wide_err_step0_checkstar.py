#!/usr/bin/env python3
"""WIDE-ERR STEP 0: check-star validity gate (read-only diagnostic)."""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))
sys.path.insert(0, str(REPO / "dev" / "scripts"))
import _bootstrap  # noqa: E402,F401

from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from config import AppConfig  # noqa: E402
from check_star_kmag import (  # noqa: E402
    _apply_crowding_exclusion,
    _drop_rms_artefacts,
    _exclude_ensemble_members,
    select_check_star,
)
from photometry_core import (  # noqa: E402
    _phase2a_prepare_shared_state,
    photometer_check_star_production_path,
)
from pipeline import _query_vsx_local  # noqa: E402

SETUP = "NoFilter_60_2"
DRAFT_NAME = "draft_000435_snapshot_skysurface_20260716"
DRAFT_ID = 435
DIAG_LC_ROOT = REPO / "tmp" / "wide_err_step0_checkstar" / "diag_check_lc"
MAG_ERR_SCALE = 1000.0
WIDE_ERR_STAR = "1499906247391001088"
SIDECAR_DOMINANT = "1497265907653703680"


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


def _p2p_scatter(mags: np.ndarray) -> float:
    m = np.asarray(mags, dtype=np.float64)
    ok = np.isfinite(m)
    m = m[ok]
    if m.size < 3:
        return float("nan")
    diffs = np.diff(m)
    return float(np.std(diffs, ddof=1) / math.sqrt(2.0))


def _rank_check_candidates(comp_df: pd.DataFrame, cfg: AppConfig, rank: int) -> str | None:
    """Return catalog_id of check candidate at 1-based rank by p2p_rms (tier 1-3)."""
    if comp_df is None or comp_df.empty:
        return None
    df = comp_df.copy()
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.strip().str.lower() == "good"]
    df = _exclude_ensemble_members(df, set())
    df = _drop_rms_artefacts(df, cfg=cfg, floor_override=None)
    df = _apply_crowding_exclusion(df, cfg)
    tier_col = "tier" if "tier" in df.columns else ("comp_tier" if "comp_tier" in df.columns else None)
    if tier_col is not None:
        df[tier_col] = pd.to_numeric(df[tier_col], errors="coerce")
    if "p2p_rms" not in df.columns:
        return None
    df["p2p_rms"] = pd.to_numeric(df["p2p_rms"], errors="coerce")
    tiers = (1, 2, 3) if tier_col is not None else (None,)
    pool: list[pd.Series] = []
    for tier in tiers:
        cand = df
        if tier is not None:
            cand = cand[(cand[tier_col].notna()) & (cand[tier_col].astype(int) == int(tier))]
        cand = cand[cand["p2p_rms"].notna()].sort_values("p2p_rms", ascending=True)
        for _, row in cand.iterrows():
            pool.append(row)
    if rank < 1 or rank > len(pool):
        return None
    return str(pool[rank - 1]["catalog_id"]).strip()


def _production_lc(
    *,
    state: Any,
    target_cid: str,
    check_cid: str,
    ps: Path,
    phot: Path,
    allow_recompute: bool = True,
) -> pd.DataFrame | None:
    diag = DIAG_LC_ROOT / target_cid
    cached = diag / f"lightcurve_{check_cid}.csv"
    if cached.is_file():
        return pd.read_csv(cached, low_memory=False)
    if not allow_recompute:
        return None
    return photometer_check_star_production_path(
        state=state,
        parent_target_cid=target_cid,
        check_cid=check_cid,
        masterstar_fits_path=ps / "MASTERSTAR.fits",
        lc_dir=diag,
        output_dir=phot,
    )


def _lc_metrics(lc_df: pd.DataFrame | None) -> dict[str, float]:
    if lc_df is None or "mag_calib_final" not in lc_df.columns:
        return {
            "sigma_total": float("nan"),
            "sigma_p2p": float("nan"),
            "err_med_mmag": float("nan"),
            "ratio_total": float("nan"),
            "ratio_p2p": float("nan"),
            "n": 0,
        }
    m = pd.to_numeric(lc_df["mag_calib_final"], errors="coerce").to_numpy(dtype=np.float64)
    e = pd.to_numeric(lc_df["err"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
    if int(np.count_nonzero(ok)) < 3:
        return {
            "sigma_total": float("nan"),
            "sigma_p2p": float("nan"),
            "err_med_mmag": float("nan"),
            "ratio_total": float("nan"),
            "ratio_p2p": float("nan"),
            "n": int(np.count_nonzero(ok)),
        }
    mo = m[ok]
    eo = e[ok]
    st = _weighted_scatter(mo, eo)
    sp = _p2p_scatter(mo)
    em = float(np.median(eo))
    return {
        "sigma_total": st,
        "sigma_p2p": sp,
        "err_med_mmag": em * MAG_ERR_SCALE,
        "ratio_total": (st / em) if em > 0 else float("nan"),
        "ratio_p2p": (sp / em) if em > 0 else float("nan"),
        "n": int(mo.size),
    }


def _lomb_scargle(lc_df: pd.DataFrame) -> dict[str, float]:
    try:
        from astropy.timeseries import LombScargle
    except ImportError:
        return {"error": "LombScargle unavailable"}
    t = pd.to_numeric(lc_df.get("bjd"), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(lc_df.get("mag_calib_final"), errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(y)
    t = t[ok]
    y = y[ok]
    if t.size < 5:
        return {"n": int(t.size)}
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    baseline_days = float(t[-1] - t[0])
    dt_med = float(np.median(np.diff(t))) if t.size > 1 else float("nan")
    nyq_h = (0.5 / dt_med) if dt_med > 0 else float("nan")
    f_min = 1.0 / baseline_days if baseline_days > 0 else 1.0 / 24.0
    f_max = min(nyq_h, 1.0 / (2.0 * dt_med)) if dt_med > 0 else 1.0 / (2.0 * 3600.0)
    if not (math.isfinite(f_min) and math.isfinite(f_max) and f_max > f_min):
        f_min, f_max = 1.0 / baseline_days, 10.0 / baseline_days
    ls = LombScargle(t, y)
    freq = np.linspace(f_min, f_max, 5000)
    power = ls.power(freq)
    idx = int(np.argmax(power))
    best_f = float(freq[idx])
    period_h = (1.0 / best_f) * 24.0 if best_f > 0 else float("nan")
    fap = float(ls.false_alarm_probability(power[idx]))
    phase = (t * best_f) % 1.0
    bins = np.linspace(0, 1, 21)
    folded = []
    for i in range(len(bins) - 1):
        m = (phase >= bins[i]) & (phase < bins[i + 1])
        if np.any(m):
            folded.append(float(np.median(y[m])))
    amp = float(max(folded) - min(folded)) if len(folded) >= 2 else float("nan")
    return {
        "baseline_days": baseline_days,
        "f_min_cpd": f_min,
        "f_max_cpd": f_max,
        "best_period_h": period_h,
        "peak_power": float(power[idx]),
        "fap": fap,
        "folded_pp_amp_mag": amp,
        "n_epochs": int(t.size),
    }


def step1_id_census(lc_dir: Path, phot: Path, cfg: AppConfig) -> dict[str, Any]:
    files = sorted(lc_dir.glob("check_kmag_*.csv"))
    side_ids: list[tuple[str, str]] = []
    for f in files:
        target = f.stem.replace("check_kmag_", "")
        df = pd.read_csv(f, nrows=1, low_memory=False)
        col = "check_catalog_id" if "check_catalog_id" in df.columns else "check_cid"
        side_ids.append((target, str(df[col].iloc[0]).strip()))

    side_counter = Counter(cid for _, cid in side_ids)
    comp = pd.read_csv(
        phot / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )

    select_counter: Counter[str] = Counter()
    mismatches: list[dict[str, str]] = []
    for target, sid in side_ids:
        sub = comp.loc[comp["target_catalog_id"].astype(str).str.strip() == target]
        row = select_check_star(sub, cfg=cfg)
        sel = str(row["catalog_id"]).strip() if row is not None else "NONE"
        select_counter[sel] += 1
        if sel != sid:
            mismatches.append({"target": target, "sidecar_id": sid, "select_check_star_id": sel})

    cross_samples = mismatches[:3]
    return {
        "n_files": len(files),
        "n_distinct_sidecar_ids": len(side_counter),
        "sidecar_id_counts": side_counter.most_common(),
        "n_distinct_select_check_star": len(select_counter),
        "select_check_star_counts_top10": select_counter.most_common(10),
        "n_sidecar_vs_select_mismatch": len(mismatches),
        "cross_check_samples": cross_samples,
        "has_is_check_star_column": "is_check_star" in comp.columns,
    }


def step2_vsx(cfg: AppConfig, ms_path: Path, source_id: str) -> dict[str, Any]:
    ms = pd.read_csv(ms_path, low_memory=False, dtype={"catalog_id": str})
    row = ms.loc[ms["catalog_id"].astype(str).str.strip() == str(source_id)]
    if row.empty:
        # try Gaia DB
        db = Path(cfg.gaia_db_path)
        conn = sqlite3.connect(db)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(gaia_source)").fetchall()]
        tbl = "gaia_source" if "gaia_source" in [
            x[0] for x in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ] else None
        ra = dec = float("nan")
        if tbl:
            r = conn.execute(
                f"SELECT ra, dec FROM {tbl} WHERE source_id=?",
                (int(source_id),),
            ).fetchone()
            if r:
                ra, dec = float(r[0]), float(r[1])
        conn.close()
    else:
        ra = float(row["ra_deg"].iloc[0])
        dec = float(row["dec_deg"].iloc[0])

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    vsx_path = Path(cfg.vsx_local_db_path)
    vsx_df = _query_vsx_local(center=coord, radius_deg=0.05, vsx_db_path=vsx_path)
    out: dict[str, Any] = {"query_ra_deg": ra, "query_dec_deg": dec, "source_id": source_id}
    if vsx_df is None or vsx_df.empty:
        out["vsx_match_within_10arcsec"] = False
        out["note"] = "NO VSX MATCH within cone query"
        return out

    v_ra = pd.to_numeric(vsx_df["ra_deg"], errors="coerce").to_numpy(dtype=float)
    v_dec = pd.to_numeric(vsx_df["dec_deg"], errors="coerce").to_numpy(dtype=float)
    vcoord = SkyCoord(ra=v_ra * u.deg, dec=v_dec * u.deg, frame="icrs")
    sep = coord.separation(vcoord).arcsec
    j = int(np.argmin(sep))
    out["nearest_sep_arcsec"] = float(sep[j])
    out["vsx_match_within_10arcsec"] = bool(sep[j] <= 10.0)
    for col in ("name", "type", "period", "mag_range", "max_mag", "min_mag"):
        if col in vsx_df.columns:
            out[f"vsx_{col}"] = vsx_df.iloc[j][col]
    return out


def step3_gaia(cfg: AppConfig, source_id: str, draft: Path) -> dict[str, Any]:
    db = Path(cfg.gaia_db_path)
    conn = sqlite3.connect(db)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    tbl = "gaia_source" if "gaia_source" in tables else tables[0]
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({tbl})").fetchall()]
    want = [
        "var_flag",
        "phot_variable_flag",
        "non_single_star",
        "g_mag",
        "bp_rp",
        "teff_gspphot",
        "logg_gspphot",
        "source_id",
    ]
    presence = {c: c in cols for c in want}
    sel_cols = [c for c in want if c in cols]
    row = None
    if "source_id" in cols:
        q = f"SELECT {', '.join(sel_cols)} FROM {tbl} WHERE source_id=?"
        row = conn.execute(q, (int(source_id),)).fetchone()
    row_dict = dict(zip(sel_cols, row)) if row else {}
    var_dist: list[tuple[Any, int]] = []
    if "var_flag" in cols:
        var_dist = conn.execute(f"SELECT var_flag, COUNT(*) FROM {tbl} GROUP BY var_flag").fetchall()
    conn.close()
    return {
        "table": tbl,
        "column_presence": presence,
        "row": row_dict,
        "var_flag_distribution_full_table": var_dist,
        "428_claim": "phot_variable_flag absent; var_flag alias per build_gaia_catalog.py",
    }


def step4_gate(ms_path: Path) -> dict[str, Any]:
    ms = pd.read_csv(ms_path, low_memory=False)
    vsx_true = int(ms["vsx_known_variable"].fillna(False).astype(bool).sum()) if "vsx_known_variable" in ms.columns else -1
    gvar_true = (
        int(ms["gaia_dr3_variable_catalog"].fillna(False).astype(bool).sum())
        if "gaia_dr3_variable_catalog" in ms.columns
        else -1
    )
    return {
        "gaia_variable_df_loader": (
            "_prefetch_export_shared_catalog_for_process_pool sets g_df = gaia_variable_df "
            "or empty DataFrame(); no query populates gaia_variable_df anywhere in pipeline.py"
        ),
        "gaia_variable_df_predicate": "N/A -- never loaded; gvar_hit always False",
        "n_gaia_variable_in_field_on_draft435": 0,
        "masterstars_n_rows": len(ms),
        "vsx_known_variable_true": vsx_true,
        "gaia_dr3_variable_catalog_true": gvar_true,
        "gate_finding": (
            "Gaia arm of catalog_known_variable is a silent no-op on this data; "
            "only VSX proximity contributes"
        ),
    }


def main() -> int:
    cfg = AppConfig()
    draft = Path(cfg.archive_root) / "Drafts" / DRAFT_NAME
    ps = draft / "platesolve" / SETUP
    phot = ps / "photometry"
    lc_dir = phot / "lightcurves"
    lights = draft / "detrended_aligned" / "lights" / SETUP
    ms_path = ps / "masterstars_full_match.csv"

    s1 = step1_id_census(lc_dir, phot, cfg)

    # Primary star for VSX/Gaia: WIDE-ERR dominant from prior JSON; fallback sidecar dominant
    vsx_star = WIDE_ERR_STAR
    s2_wide = step2_vsx(cfg, ms_path, vsx_star)
    s2_side = step2_vsx(cfg, ms_path, SIDECAR_DOMINANT)
    s3 = step3_gaia(cfg, vsx_star, draft)
    s3_side = step3_gaia(cfg, SIDECAR_DOMINANT, draft)
    s4 = step4_gate(ms_path)

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
    comp = pd.read_csv(
        phot / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )

    # STEP 5 bulk: cached diag LCs for current sidecar ids
    step5_rows: list[dict[str, Any]] = []
    for ck_path in sorted(lc_dir.glob("check_kmag_*.csv")):
        target = ck_path.stem.replace("check_kmag_", "")
        ckdf = pd.read_csv(ck_path, nrows=1, low_memory=False)
        id_col = "check_catalog_id" if "check_catalog_id" in ckdf.columns else "check_cid"
        check_cid = str(ckdf[id_col].iloc[0]).strip()
        lc_df = _production_lc(
            state=state,
            target_cid=target,
            check_cid=check_cid,
            ps=ps,
            phot=phot,
            allow_recompute=False,
        )
        met = _lc_metrics(lc_df)
        if met["n"] >= 3:
            step5_rows.append({"target_cid": target, "check_cid": check_cid, **met})

    # Prior WIDE-ERR run (162 fields): production LCs keyed to sidecar id at run time
    prior_json = REPO / "tmp" / "wide_error_budget_diag.json"
    prior_rows: list[dict[str, Any]] = []
    prior_note: dict[str, Any] = {}
    if prior_json.is_file():
        prior = json.loads(prior_json.read_text(encoding="utf-8"))
        pc = Counter(r["check_cid"] for r in prior.get("per_check_star", []))
        prior_note = {
            "prior_n_fields": prior.get("n_check_fields"),
            "prior_check_cid_counts": pc.most_common(),
        }
        for row in prior.get("per_check_star", []):
            sc = float(row.get("scatter_mmag", float("nan"))) / MAG_ERR_SCALE
            em = float(row.get("err_median_mmag", float("nan"))) / MAG_ERR_SCALE
            prior_rows.append(
                {
                    "target_cid": row.get("target_cid"),
                    "check_cid": row.get("check_cid"),
                    "sigma_total": sc,
                    "err_med_mmag": float(row.get("err_mmedian_mmag", row.get("err_median_mmag", float("nan")))),
                    "ratio_total": float(row.get("ratio_scatter_over_err", sc / em if em > 0 else float("nan"))),
                }
            )

    # Recompute one representative prior-star LC for sigma_p2p + Lomb-Scargle
    prior_rep_target = None
    prior_rep_lc = None
    prior_rep_metrics: dict[str, float] = {}
    prior_ls: dict[str, Any] = {}
    if prior_rows:
        bulk = [r for r in prior_rows if str(r.get("check_cid")) == WIDE_ERR_STAR]
        if bulk:
            prior_rep_target = str(bulk[0]["target_cid"])
            prior_rep_lc = _production_lc(
                state=state,
                target_cid=prior_rep_target,
                check_cid=WIDE_ERR_STAR,
                ps=ps,
                phot=phot,
                allow_recompute=True,
            )
            prior_rep_metrics = _lc_metrics(prior_rep_lc)
            if prior_rep_lc is not None:
                prior_ls = _lomb_scargle(prior_rep_lc)

    ratios_total = [r["ratio_total"] for r in step5_rows if math.isfinite(r["ratio_total"])]
    ratios_p2p = [r["ratio_p2p"] for r in step5_rows if math.isfinite(r["ratio_p2p"])]
    prior_ratios_total = [r["ratio_total"] for r in prior_rows if math.isfinite(r["ratio_total"])]

    rep_target = step5_rows[0]["target_cid"] if step5_rows else prior_rep_target
    rep_lc = None
    rep_metrics = {}
    ls_out: dict[str, Any] = {}
    if step5_rows:
        rep_check = step5_rows[0]["check_cid"]
        rep_lc = _production_lc(
            state=state,
            target_cid=rep_target,
            check_cid=rep_check,
            ps=ps,
            phot=phot,
            allow_recompute=False,
        )
        rep_metrics = _lc_metrics(rep_lc)
        if rep_lc is not None:
            ls_out = _lomb_scargle(rep_lc)
    elif prior_rep_lc is not None:
        rep_metrics = prior_rep_metrics
        ls_out = prior_ls

    # STEP 6: 20 representative fields, ranks 1-3
    ok_targets = [r["target_cid"] for r in step5_rows]
    sample_targets = ok_targets[:: max(1, len(ok_targets) // 20)][:20]
    rank_medians: dict[int, list[float]] = {1: [], 2: [], 3: []}
    for target in sample_targets:
        sub = comp.loc[comp["target_catalog_id"].astype(str).str.strip() == target]
        for rank in (1, 2, 3):
            cid = _rank_check_candidates(sub, cfg, rank)
            if not cid:
                continue
            lc_df = _production_lc(
                state=state,
                target_cid=target,
                check_cid=cid,
                ps=ps,
                phot=phot,
                allow_recompute=True,
            )
            met = _lc_metrics(lc_df)
            if math.isfinite(met["ratio_total"]):
                rank_medians[rank].append(met["ratio_total"])

    s6 = {
        "n_sample_fields": len(sample_targets),
        "median_sigma_total_over_err_rank1": float(np.median(rank_medians[1])) if rank_medians[1] else float("nan"),
        "median_sigma_total_over_err_rank2": float(np.median(rank_medians[2])) if rank_medians[2] else float("nan"),
        "median_sigma_total_over_err_rank3": float(np.median(rank_medians[3])) if rank_medians[3] else float("nan"),
        "n_rank1": len(rank_medians[1]),
        "n_rank2": len(rank_medians[2]),
        "n_rank3": len(rank_medians[3]),
    }

    # Prior wide-error JSON comparison (prior_note filled above)
    out = {
        "step1": s1,
        "step2_wide_err_star": s2_wide,
        "step2_sidecar_dominant_star": s2_side,
        "step3_wide_err_star": s3,
        "step3_sidecar_dominant": s3_side,
        "step4": s4,
        "step5": {
            "n_cached_production_lcs_current_sidecar": len(step5_rows),
            "median_ratio_sigma_total_over_err_cached": float(np.median(ratios_total))
            if ratios_total
            else float("nan"),
            "median_ratio_sigma_p2p_over_err_cached": float(np.median(ratios_p2p))
            if ratios_p2p
            else float("nan"),
            "prior_wide_err_n_fields": len(prior_rows),
            "prior_median_ratio_sigma_total_over_err": float(np.median(prior_ratios_total))
            if prior_ratios_total
            else float("nan"),
            "prior_rep_target": prior_rep_target,
            "prior_rep_check_cid": WIDE_ERR_STAR if prior_rep_target else None,
            "prior_rep_metrics": prior_rep_metrics,
            "prior_rep_lomb_scargle": prior_ls,
            "representative_target_cached": rep_target,
            "representative_metrics_cached": rep_metrics,
            "lomb_scargle_cached_sidecar_star": ls_out,
            "check_cid_counts_cached": Counter(r["check_cid"] for r in step5_rows).most_common(),
        },
        "step6": s6,
        "prior_wide_error_json": prior_note,
    }

    out_path = REPO / "tmp" / "wide_err_step0_checkstar.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
