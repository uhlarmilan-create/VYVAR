#!/usr/bin/env python3
"""XVAL-BO-01: dump BO frame 001 and explain check_kmag vs proc flux-sum."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from photometry_core import ensemble_normalize, pytics_iterative_weights  # noqa: E402

RUN_SHA = "da9cce4a5edd1392b8ba842d3c8488589b9d0ac9"
DRAFT = ROOT / "Archive" / "Drafts" / "draft_000515"
SETUP = "NoFilter_60_2"
PROC = DRAFT / "detrended_aligned" / "lights" / SETUP
PHOT = DRAFT / "platesolve" / SETUP / "photometry"
LC = PHOT / "lightcurves"

BO = "1498613634033133184"
FW = "1497343732462852864"
CHECK = "1497613731286514432"
FRAME001 = "proc_BO_CVn_Light_001.csv"
# FW uses same night frames; first frame name may differ - use BO frame for BO dump

OUT_DUMP = ROOT / "dev" / "results" / "XVAL_BO_01_frame001_dump.json"
OUT_MD = ROOT / "dev" / "results" / "CURSOR_RESULT_XVAL_BO_01.md"

MAD_SCALE = 1.4826


def mad_mmag(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(np.abs(a - np.median(a))) * MAD_SCALE * 1000.0)


def load_series(cids: list[str], order: list[str], col: str = "dao_flux") -> dict[str, np.ndarray]:
    series = {c: [] for c in cids}
    for name in order:
        df = pd.read_csv(PROC / name, dtype={"catalog_id": str}, low_memory=False)
        df["catalog_id"] = df["catalog_id"].astype(str).str.strip()
        flux = pd.to_numeric(df[col], errors="coerce")
        m = -2.5 * np.log10(np.where(flux.to_numpy() > 0, flux.to_numpy(), np.nan))
        by = dict(zip(df["catalog_id"], m, strict=False))
        flux_by = dict(zip(df["catalog_id"], flux.to_numpy(), strict=False))
        for c in cids:
            series[c].append(float(by[c]) if c in by and np.isfinite(by[c]) else float("nan"))
        # stash fluxes only for last frame handled externally
        _ = flux_by
    return {c: np.asarray(v, dtype=float) for c, v in series.items()}


def comps_of(comp: pd.DataFrame, tid: str) -> pd.DataFrame:
    return comp[comp["target_catalog_id"].astype(str).str.strip() == tid].copy()


def main() -> None:
    comp = pd.read_csv(
        PHOT / "comparison_stars_per_target.csv",
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    bo_comps = comps_of(comp, BO)
    fw_comps = comps_of(comp, FW)
    bo_ids = [str(x).strip() for x in bo_comps["catalog_id"]]
    fw_ids = [str(x).strip() for x in fw_comps["catalog_id"]]

    chk_bo = pd.read_csv(LC / f"check_kmag_{BO}.csv")
    chk_fw = pd.read_csv(LC / f"check_kmag_{FW}.csv")
    order = list(chk_bo["source_file"].astype(str))

    # --- Frame 001 dump (BO) ---
    df1 = pd.read_csv(PROC / FRAME001, dtype={"catalog_id": str}, low_memory=False)
    df1["catalog_id"] = df1["catalog_id"].astype(str).str.strip()
    want = set(bo_ids + [CHECK, BO])
    rows_dump = []
    for cid in sorted(want):
        sub = df1[df1["catalog_id"] == cid]
        if sub.empty:
            rows_dump.append({"catalog_id": cid, "present": False})
            continue
        r = sub.iloc[0]
        flux = float(pd.to_numeric(r.get("flux"), errors="coerce"))
        dao = float(pd.to_numeric(r.get("dao_flux"), errors="coerce"))
        ap = float(pd.to_numeric(r.get("aperture_r_px"), errors="coerce"))
        role = "check" if cid == CHECK else ("target" if cid == BO else "comp")
        rms = None
        w = None
        cat = None
        if cid in set(bo_ids):
            crow = bo_comps[bo_comps["catalog_id"].astype(str).str.strip() == cid].iloc[0]
            rms = float(pd.to_numeric(crow.get("comp_rms"), errors="coerce"))
            cat = float(pd.to_numeric(crow.get("phot_g_mean_mag", crow.get("mag")), errors="coerce"))
            if math.isfinite(rms) and rms > 1e-6:
                w = 1.0 / (rms**2)
        rows_dump.append(
            {
                "catalog_id": cid,
                "role": role,
                "present": True,
                "aperture_r_px": ap,
                "flux": flux,
                "dao_flux": dao,
                "flux_equals_dao": bool(
                    math.isfinite(flux) and math.isfinite(dao) and abs(flux - dao) < 1e-6
                ),
                "mag_inst_from_dao": (
                    float(-2.5 * math.log10(dao)) if math.isfinite(dao) and dao > 0 else None
                ),
                "catalog_G_mag": cat,
                "comp_rms_mag": rms,
                "zp_weight_1_over_rms2": w,
            }
        )

    # Epoch formulas on frame 001
    def flux_of(cid: str, col: str = "dao_flux") -> float:
        sub = df1[df1["catalog_id"] == cid]
        if sub.empty:
            return float("nan")
        return float(pd.to_numeric(sub.iloc[0][col], errors="coerce"))

    f_chk = flux_of(CHECK)
    f_comps = [flux_of(c) for c in bo_ids]
    sum_c = float(np.nansum([f for f in f_comps if math.isfinite(f) and f > 0]))
    kmag_fluxsum = (
        float(-2.5 * math.log10(f_chk / sum_c))
        if math.isfinite(f_chk) and f_chk > 0 and sum_c > 0
        else float("nan")
    )
    # mag_calib style for one epoch
    m_chk = -2.5 * math.log10(f_chk) if f_chk > 0 else float("nan")
    zp_num = zp_den = 0.0
    zp_detail = []
    for c, f in zip(bo_ids, f_comps, strict=False):
        if not (math.isfinite(f) and f > 0):
            continue
        crow = bo_comps[bo_comps["catalog_id"].astype(str).str.strip() == c].iloc[0]
        cm = float(pd.to_numeric(crow.get("phot_g_mean_mag", crow.get("mag")), errors="coerce"))
        rms = float(pd.to_numeric(crow.get("comp_rms"), errors="coerce"))
        m_j = -2.5 * math.log10(f)
        d = cm - m_j
        w = 1.0 / (rms**2) if math.isfinite(rms) and rms > 1e-6 else float("nan")
        zp_detail.append({"catalog_id": c, "m_inst": m_j, "cat_G": cm, "zp_off": d, "w": w})
        if math.isfinite(w) and w > 0 and math.isfinite(d):
            zp_num += w * d
            zp_den += w
    kmag_magcalib_phase1 = m_chk + (zp_num / zp_den) if zp_den > 0 else float("nan")

    # Sidecar epoch for same source_file
    side_row = chk_bo[chk_bo["source_file"].astype(str) == FRAME001]
    kmag_prod_epoch = (
        float(pd.to_numeric(side_row.iloc[0]["kmag"], errors="coerce"))
        if not side_row.empty
        else None
    )

    # --- Full-series reconstruction ---
    def series_stats(tid: str, ids: list[str], sub: pd.DataFrame, side: pd.DataFrame) -> dict:
        order_t = list(side["source_file"].astype(str))
        cids = ids + [CHECK]
        mag = load_series(cids, order_t, col="dao_flux")
        cat = {}
        rms = {}
        qual = {}
        for _, r in sub.iterrows():
            cid = str(r["catalog_id"]).strip()
            cat[cid] = float(pd.to_numeric(r.get("phot_g_mean_mag", r.get("mag")), errors="coerce"))
            rms[cid] = float(pd.to_numeric(r.get("comp_rms"), errors="coerce"))
            qual[cid] = {"quality": "good"}
        # Pure flux-sum differential (architect Layer-3 / delta_mag)
        stack = np.vstack([mag[c] for c in ids])
        flux_c = np.nansum(10.0 ** (-0.4 * stack), axis=0)
        m_ens = -2.5 * np.log10(np.where(flux_c > 0, flux_c, np.nan))
        delta = mag[CHECK] - m_ens
        # mag_calib via ensemble_normalize (Phase-1 rms, no weight map)
        mc, dm, _ = ensemble_normalize(
            mag[CHECK],
            {c: mag[c] for c in ids},
            cat,
            qual,
            comp_rms_map=rms,
            n_comp_min=2,
            n_comp_max=8,
        )
        # pytics-inflated rms then mag_calib (mirrors Phase 2A order before check sidecar)
        rms2 = pytics_iterative_weights(
            comp_lc={c: mag[c] for c in ids},
            comp_quality=qual,
            comp_rms_map=dict(rms),
            n_iter=5,
            enabled=True,
        )
        mc_py, dm_py, _ = ensemble_normalize(
            mag[CHECK],
            {c: mag[c] for c in ids},
            cat,
            qual,
            comp_rms_map=rms2,
            n_comp_min=2,
            n_comp_max=8,
        )
        k_prod = pd.to_numeric(side["kmag"], errors="coerce").to_numpy()

        def corr(a, b):
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 5:
                return None
            return float(np.corrcoef(a[m], b[m])[0, 1])

        def medabs(a, b):
            m = np.isfinite(a) & np.isfinite(b)
            return float(np.nanmedian(np.abs(a[m] - b[m])) * 1000.0) if m.any() else None

        return {
            "n_comp": len(ids),
            "comp_ids": ids,
            "production_check_mad_mmag": mad_mmag(k_prod),
            "delta_mag_fluxsum_mad_mmag": mad_mmag(delta),
            "ensemble_normalize_delta_mag_mad_mmag": mad_mmag(dm),
            "mag_calib_phase1_rms_mad_mmag": mad_mmag(mc),
            "mag_calib_pytics_rms_mad_mmag": mad_mmag(mc_py),
            "corr_prod_vs_delta_mag": corr(k_prod, delta),
            "corr_prod_vs_mag_calib_phase1": corr(k_prod, mc),
            "corr_prod_vs_mag_calib_pytics": corr(k_prod, mc_py),
            "medabs_prod_vs_delta_mmag": medabs(k_prod, delta),
            "medabs_prod_vs_mag_calib_phase1_mmag": medabs(k_prod, mc),
            "medabs_prod_vs_mag_calib_pytics_mmag": medabs(k_prod, mc_py),
            "pytics_rms_before_after": {
                c: {"before": rms[c], "after": float(rms2.get(c, float("nan")))} for c in ids
            },
        }

    bo_stats = series_stats(BO, bo_ids, bo_comps, chk_bo)
    fw_stats = series_stats(FW, fw_ids, fw_comps, chk_fw)

    dump = {
        "task": "XVAL-BO-01",
        "run_sha": RUN_SHA,
        "frame": FRAME001,
        "target_catalog_id": BO,
        "check_catalog_id": CHECK,
        "quantity_notes": {
            "flux": "proc CSV column 'flux'",
            "dao_flux": "proc CSV column 'dao_flux' (Phase 2A read_flux_from_csv uses this)",
            "kmag_fluxsum": "-2.5*log10(F_check / sum F_comp) from dao_flux [mag]",
            "kmag_magcalib_phase1": (
                "m_check + sum(w*(G_cat - m_comp))/sum(w), w=1/comp_rms^2 [mag]"
            ),
            "production_sidecar_kmag": "check_kmag CSV kmag column for this source_file [mag]",
        },
        "stars": rows_dump,
        "epoch_formulas": {
            "kmag_fluxsum_mag": kmag_fluxsum,
            "kmag_magcalib_phase1_rms_mag": kmag_magcalib_phase1,
            "production_sidecar_kmag_mag": kmag_prod_epoch,
            "zp_detail": zp_detail,
            "delta_fluxsum_minus_prod_mmag": (
                (kmag_fluxsum - kmag_prod_epoch) * 1000.0
                if kmag_prod_epoch is not None and math.isfinite(kmag_fluxsum)
                else None
            ),
            "magcalib_minus_prod_mmag": (
                (kmag_magcalib_phase1 - kmag_prod_epoch) * 1000.0
                if kmag_prod_epoch is not None and math.isfinite(kmag_magcalib_phase1)
                else None
            ),
        },
        "full_series": {"BO_CVn": bo_stats, "FW_CVn": fw_stats},
        "code_path": {
            "writer": "photometry_core._phase2a_process_one_target -> compute_check_ensemble_mag_calib -> ensemble_normalize -> save_check_kmag_sidecar",
            "flux_authority": "read_flux_from_csv: dao_flux (not 'flux' by name; on this night they match for dumped stars)",
            "aperture": "aperture_r_px from proc CSV row (DAO aperture used when dao_flux was measured)",
            "combination": (
                "ens_med = -2.5 log10(sum 10^(-0.4 m_i)) unweighted flux sum; "
                "delta_mag = m_check - ens_med; "
                "sidecar stores mag_calib = m_check + weighted_mean(G_cat - m_comp) with "
                "w=1/rms^2 (pytics-updated rms), NOT delta_mag"
            ),
            "per_target_branch": (
                "No BO/FW special case. Same function. Differences arise from ensemble "
                "membership, n_comp, per-comp rms (incl. pytics), and catalog G values."
            ),
        },
    }
    OUT_DUMP.write_text(json.dumps(dump, indent=2) + "\n", encoding="utf-8")
    print("WROTE", OUT_DUMP, flush=True)
    print("BO", json.dumps(bo_stats, indent=2)[:1200], flush=True)
    print("FW", json.dumps({k: fw_stats[k] for k in fw_stats if "pytics_rms" not in k}, indent=2), flush=True)
    print(
        "frame001 fluxsum",
        kmag_fluxsum,
        "magcalib",
        kmag_magcalib_phase1,
        "prod",
        kmag_prod_epoch,
        flush=True,
    )


if __name__ == "__main__":
    main()
