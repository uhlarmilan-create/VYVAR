#!/usr/bin/env python3
"""Newton bin4 gain/RN forensics + ensemble SEM=0 trace (sandbox, report-only)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from astropy.io import fits  # noqa: E402

from check_star_kmag import (  # noqa: E402
    build_aligned_comp_inst,
    comp_ensemble_maps,
    resolve_proc_csv_dir,
)
from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from param_resolver import (  # noqa: E402
    _binning_from_header,
    _scale_bin1_to_binning,
    resolve_gain,
    resolve_read_noise,
)
from photometry_core import (  # noqa: E402
    _photometric_error,
    check_comparison_stability,
    ensemble_normalize,
    parse_comp_quality_json_map,
)
from scripts.chi2_sigma_gate import (  # noqa: E402
    evaluate_lc_chi2_variants,
    load_proc_row_for_source,
    reduced_chi2_constant,
    sigma_arrays_from_lc_and_proc,
    write_summary_json,
)
from scripts.select_constant_calibrators import compute_loo_production_ensemble_scatter  # noqa: E402
from scripts.sparse_comp_diag import SS_CAM_CID, V0611_CID, _check_star_chi2_rows  # noqa: E402
from sigma_budget import (  # noqa: E402
    SIGMA_VARIANT_HOWELL_ONLY,
    relative_flux_err_to_mag_sigma,
    resolve_rig_scintillation_params,
)

SIGMA_VARIANT_BIN4_GAIN_HYPOTHESIS = "bin4_gain_rn_hypothesis"


def _norm_id(x: object) -> str:
    return str(normalize_gaia_source_id(x) or "").strip()


def _draft_equipment_id(db: VyvarDatabase, draft_id: int) -> int | None:
    row = db.conn.execute(
        "SELECT ID_EQUIPMENTS FROM OBS_DRAFT WHERE ID = ?;",
        (int(draft_id),),
    ).fetchone()
    if row is None or row["ID_EQUIPMENTS"] is None:
        return None
    return int(row["ID_EQUIPMENTS"])


def _equipment_db_cosmic(db: VyvarDatabase, equipment_id: int) -> dict[str, float | str | None]:
    row = db.conn.execute(
        "SELECT ID, CAMERANAME, SENSORTYPE, GAIN_ADU, READNOISE_E FROM EQUIPMENTS WHERE ID = ?;",
        (int(equipment_id),),
    ).fetchone()
    return dict(row) if row else {}


def _find_example_fits(phot_dir: Path, setup: str) -> Path | None:
    lights = phot_dir.parent.parent.parent / "non_calibrated" / "lights" / setup
    if lights.is_dir():
        for pat in ("*.fits", "*.fit", "*.fts"):
            hits = sorted(lights.glob(pat))
            if hits:
                return hits[0]
    det = phot_dir.parent.parent.parent / "detrended_aligned" / "lights" / setup
    if det.is_dir():
        for pat in ("*.fits", "*.fit", "*.fts"):
            hits = sorted(det.glob(pat))
            if hits:
                return hits[0]
    return None


def expected_bin_scaled(
    gain_bin1: float,
    rn_bin1: float,
    binning: int,
    *,
    sum_mode: bool = True,
) -> tuple[float, float, str]:
    """Expected effective gain/RN for software-summed 2D binning (VYVAR policy)."""
    b = int(binning)
    if sum_mode:
        gain_exp = _scale_bin1_to_binning(gain_bin1, b, exponent=2)
        rn_exp = _scale_bin1_to_binning(rn_bin1, b, exponent=1)
        note = "software_sum: gain*=b^2, RN*=b (param_resolver exponents 2/1)"
    else:
        gain_exp = float(gain_bin1)
        rn_exp = float(rn_bin1) / math.sqrt(max(b * b, 1))
        note = "average_mode_hypothesis: gain unchanged, RN/sqrt(b^2)"
    return float(gain_exp), float(rn_exp), note


def forensics_setup(
    draft_id: int,
    setup: str,
    *,
    cfg: AppConfig,
) -> dict[str, Any]:
    db = VyvarDatabase(cfg.database_path)
    phot_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}" / "platesolve" / setup / "photometry"
    proc_dir = resolve_proc_csv_dir(phot_dir, setup)
    eq_id = _draft_equipment_id(db, draft_id)
    eq_row = _equipment_db_cosmic(db, int(eq_id)) if eq_id is not None else {}
    fits_path = _find_example_fits(phot_dir, setup)
    header = None
    if fits_path is not None and fits_path.is_file():
        with fits.open(fits_path, memmap=False) as hdul:
            header = hdul[0].header

    binning = _binning_from_header(header)
    gain_res = resolve_gain(header, db=db, equipment_id=eq_id, cfg=cfg)
    rn_res = resolve_read_noise(header, db=db, equipment_id=eq_id, cfg=cfg)

    gain_db = float(eq_row.get("GAIN_ADU")) if eq_row.get("GAIN_ADU") is not None else float("nan")
    rn_db = float(eq_row.get("READNOISE_E")) if eq_row.get("READNOISE_E") is not None else float("nan")
    b = int(binning or 4)
    gain_exp, rn_exp, scale_note = expected_bin_scaled(gain_db, rn_db, b, sum_mode=True)

    ratio_gain = float(gain_res.value) / gain_exp if gain_exp > 0 and math.isfinite(float(gain_res.value or float("nan"))) else float("nan")
    ratio_rn = float(rn_res.value) / rn_exp if rn_exp > 0 and math.isfinite(float(rn_res.value or float("nan"))) else float("nan")

    # representative star from first proc csv
    rep_flux = rep_sky = rep_area = float("nan")
    if proc_dir is not None:
        csvs = sorted(proc_dir.glob("proc_*.csv"))
        if csvs:
            df = pd.read_csv(csvs[0], low_memory=False)
            if not df.empty:
                row = df.iloc[0]
                rep_flux = float(pd.to_numeric(row.get("dao_flux"), errors="coerce"))
                from photometry_core import _sky_pp_for_photometric_error  # noqa: PLC0415

                rep_sky = float(_sky_pp_for_photometric_error(row))
                rep_area = float(pd.to_numeric(row.get("aperture_area_px"), errors="coerce"))
                if not math.isfinite(rep_area) or rep_area <= 0:
                    r = float(pd.to_numeric(row.get("aperture_r_px"), errors="coerce"))
                    rep_area = math.pi * r * r if math.isfinite(r) and r > 0 else float("nan")

    sigma_used = sigma_exp = ratio_sigma = chi2_pred = float("nan")
    if all(math.isfinite(v) and v > 0 for v in (rep_flux, rep_sky, rep_area)):
        g_used = float(gain_res.value) if gain_res.value is not None else 1.0
        rn_used = float(rn_res.value) if rn_res.value is not None else 10.0
        sigma_used = float(_photometric_error(rep_flux, rep_sky, rep_area, gain=g_used, read_noise=rn_used))
        sigma_exp = float(_photometric_error(rep_flux, rep_sky, rep_area, gain=gain_exp, read_noise=rn_exp))
        if sigma_exp > 0:
            ratio_sigma = sigma_used / sigma_exp
            chi2_pred = 1.0 / (ratio_sigma * ratio_sigma)

    hdr_keys = {}
    if header is not None:
        for k in ("XBINNING", "YBINNING", "BINNING", "GAIN", "EGAIN", "READNOISE", "RDNOISE", "READOUTM", "ROWORDN"):
            if k in header:
                hdr_keys[k] = str(header[k])

    return {
        "draft_id": draft_id,
        "setup": setup,
        "equipment_id": eq_id,
        "equipment": eq_row,
        "fits_example": str(fits_path) if fits_path else None,
        "header_keys": hdr_keys,
        "binning_resolved": binning,
        "gain_used": gain_res.value,
        "gain_source": gain_res.source,
        "read_noise_used": rn_res.value,
        "read_noise_source": rn_res.source,
        "gain_db_bin1": gain_db,
        "read_noise_db_bin1": rn_db,
        "gain_expected_bin": gain_exp,
        "read_noise_expected_bin": rn_exp,
        "scaling_formula": scale_note,
        "ratio_gain_used_over_expected": ratio_gain,
        "ratio_rn_used_over_expected": ratio_rn,
        "sigma_used_rel": sigma_used,
        "sigma_expected_rel": sigma_exp,
        "sigma_ratio_used_over_expected": ratio_sigma,
        "chi2_predicted_from_sigma_ratio": chi2_pred,
        "sensor_note": (
            "Archive draft_426 equipment is "
            f"{eq_row.get('CAMERANAME')} ({eq_row.get('SENSORTYPE')}); "
            "task C3-26000/IMX571 reference is equipment_id=2 in DB"
        ),
    }


def trace_ensemble_sem_zero(
    draft_id: int,
    setup: str,
    target_cid: str,
    *,
    cfg: AppConfig,
) -> dict[str, Any]:
    phot_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}" / "platesolve" / setup / "photometry"
    lc_dir = phot_dir / "lightcurves"
    lc_path = lc_dir / f"lightcurve_{target_cid}.csv"
    side = lc_dir / f"check_kmag_{target_cid}.csv"
    if not side.is_file() or not lc_path.is_file():
        return {"available": False}
    lc_df = pd.read_csv(lc_path, low_memory=False)
    side_df = pd.read_csv(side, low_memory=False)
    chk_cid = str(side_df["check_catalog_id"].iloc[0]) if "check_catalog_id" in side_df.columns else ""
    comp_all = pd.read_csv(
        phot_dir / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    comp_df = comp_all.loc[comp_all["target_catalog_id"].map(_norm_id) == _norm_id(target_cid)]
    cat, tier, rms, tw = comp_ensemble_maps(comp_df, cfg)
    comp_ids = [_norm_id(c) for c in comp_df["catalog_id"].tolist() if _norm_id(c)]
    proc_dir = resolve_proc_csv_dir(phot_dir, setup)
    if proc_dir is None or not chk_cid:
        return {"available": False}
    if chk_cid not in comp_ids:
        comp_ids.append(chk_cid)
    comp_lc = build_aligned_comp_inst(
        proc_dir, comp_ids, lc_df["source_file"].astype(str).tolist(), cfg, "aperture",
    )
    if chk_cid not in comp_lc:
        return {"available": False, "reason": "check star absent from proc alignment"}
    other_ids = [c for c in comp_ids if c != chk_cid and c in comp_lc]
    other_lc = {c: comp_lc[c] for c in other_ids}
    comp_quality = check_comparison_stability(
        other_lc, comp_rms_map=rms, n_comp_min=3, outlier_sigma=3.0, common_mode_detrend=True,
    )
    other_cat = {c: cat[c] for c in other_ids if c in cat}
    other_quality = {
        c: comp_quality[c]
        for c in other_ids
        if c in comp_quality and str(comp_quality[c].get("quality", "")).strip().lower() != "excluded"
    }
    _, _, ensemble_scatter = ensemble_normalize(
        comp_lc[chk_cid],
        other_lc,
        other_cat,
        other_quality,
        comp_rms_map=rms,
        comp_tier_map=tier,
        tier_weights=tw,
        n_comp_min=3,
        n_comp_max=int(cfg.phase01_comparison_n_comp_max),
    )

    # manual trace comp_ref_map logic from photometry_core
    good_ids = [c for c in other_ids if c in other_quality]
    comp_ref_map: dict[str, float] = {}
    for cid in good_ids:
        arr = comp_lc.get(cid)
        if arr is None:
            continue
        fin = np.asarray(arr, dtype=np.float64)
        fin = fin[np.isfinite(fin)]
        if fin.size:
            comp_ref_map[cid] = float(np.median(fin))

    n_resid_list: list[int] = []
    n_pairs_list: list[int] = []
    missing_ref_frames = 0
    for i in range(len(lc_df)):
        comp_pairs: list[tuple[str, float]] = []
        for cid in good_ids:
            if cid not in comp_lc:
                continue
            mv = float(comp_lc[cid][i])
            if math.isfinite(mv):
                comp_pairs.append((cid, mv))
        comp_resid = [
            (m - comp_ref_map[cid_j])
            for cid_j, m in comp_pairs
            if cid_j in comp_ref_map and math.isfinite(comp_ref_map[cid_j])
        ]
        n_pairs_list.append(len(comp_pairs))
        n_resid_list.append(len(comp_resid))
        if len(comp_resid) < 2:
            missing_ref_frames += 1

    arr_n = np.asarray(n_resid_list, dtype=int)
    sc = np.asarray(ensemble_scatter, dtype=np.float64)
    return {
        "available": True,
        "target_catalog_id": target_cid,
        "check_catalog_id": chk_cid,
        "n_other_comps": len(other_ids),
        "n_good_ids": len(good_ids),
        "comp_ref_map_size": len(comp_ref_map),
        "frames_n_resid_lt2": int(missing_ref_frames),
        "frames_total": int(len(lc_df)),
        "n_resid_per_frame_p50": float(np.quantile(arr_n, 0.5)) if arr_n.size else None,
        "n_resid_per_frame_p95": float(np.quantile(arr_n, 0.95)) if arr_n.size else None,
        "ensemble_scatter_zero_fraction": float(np.mean(sc == 0.0)) if sc.size else None,
        "ensemble_scatter_nan_fraction": float(np.mean(~np.isfinite(sc))) if sc.size else None,
        "root_cause": (
            "ensemble_normalize sets scatter=0 when len(comp_resid)<2 "
            "(Honeycutt SEM needs >=2 comps with finite comp_ref_map residuals)"
        ),
    }


def check_chi2_with_gain_hypothesis(
    draft_id: int,
    setup: str,
    target_cid: str,
    *,
    cfg: AppConfig,
    gain_exp: float,
    rn_exp: float,
    f_resid_e: float,
    sigma_floor_e: float,
) -> dict[str, Any]:
    phot_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}" / "platesolve" / setup / "photometry"
    lc_dir = phot_dir / "lightcurves"
    lc_path = lc_dir / f"lightcurve_{target_cid}.csv"
    proc_dir = resolve_proc_csv_dir(phot_dir, setup)
    meta = json.loads((phot_dir / "pipeline_meta.json").read_text(encoding="utf-8")) if (phot_dir / "pipeline_meta.json").is_file() else {}
    rig = resolve_rig_scintillation_params(draft_id=draft_id, setup=setup, cfg=cfg, pipeline_meta=meta)
    side = lc_dir / f"check_kmag_{target_cid}.csv"
    if not side.is_file() or proc_dir is None or not lc_path.is_file():
        return {"available": False}
    lc_df = pd.read_csv(lc_path, low_memory=False)
    side_df = pd.read_csv(side, low_memory=False)
    rows, sem_summary = _check_star_chi2_rows(
        phot_dir=phot_dir,
        setup=setup,
        target_cid=target_cid,
        lc_df=lc_df,
        side_df=side_df,
        proc_dir=proc_dir,
        rig=rig,
        cfg=cfg,
        f_resid_e=f_resid_e,
        sigma_floor_e=sigma_floor_e,
    )
    chk_cid = sem_summary.get("check_catalog_id", "")
    work = side_df.copy()
    work["delta_mag"] = pd.to_numeric(work["kmag"], errors="coerce")
    work["source_file"] = lc_df["source_file"].astype(str).iloc[: len(work)].tolist()
    work["airmass"] = pd.to_numeric(lc_df["airmass"], errors="coerce").iloc[: len(work)].tolist()
    if "err" not in work.columns and "err" in lc_df.columns:
        work["err"] = pd.to_numeric(lc_df["err"], errors="coerce").iloc[: len(work)].tolist()
    prod_scatter = compute_loo_production_ensemble_scatter(
        str(chk_cid), phot_dir=phot_dir, setup=setup, anchor_target=target_cid, cfg=cfg,
    )
    mags, variants, sh, ss, sem_meta = sigma_arrays_from_lc_and_proc(
        work,
        proc_dir,
        str(chk_cid),
        rig_params=rig,
        f_resid=f_resid_e,
        sigma_floor_mag=sigma_floor_e,
        production_ensemble_scatter=prod_scatter,
        gain=float(gain_exp),
        read_noise=float(rn_exp),
    )
    sem = sem_meta["ensemble_sem_primary"]
    n = len(work)
    hyp_sig = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if math.isfinite(sh[i]) and math.isfinite(ss[i]):
            sem_i = float(sem[i]) if math.isfinite(float(sem[i])) else 0.0
            from sigma_budget import combine_sigma_mag_quadrature  # noqa: PLC0415

            hyp_sig[i] = combine_sigma_mag_quadrature(
                sh[i], f_resid_e * ss[i], sigma_floor_mag=sigma_floor_e, ensemble_sem_mag=sem_i,
            )
    variants[SIGMA_VARIANT_BIN4_GAIN_HYPOTHESIS] = hyp_sig
    bjd = pd.to_numeric(lc_df.get("bjd"), errors="coerce").to_numpy(dtype=np.float64)
    hyp_rows = [r.to_dict() for r in evaluate_lc_chi2_variants(mags, variants, catalog_id=str(chk_cid), mag_g=None, bjd=bjd)]
    return {
        "available": True,
        "draft_id": draft_id,
        "setup": setup,
        "target_catalog_id": target_cid,
        "gain_hypothesis": gain_exp,
        "read_noise_hypothesis": rn_exp,
        "baseline_check_chi2": rows,
        "hypothesis_check_chi2": hyp_rows,
        "ensemble_sem": sem_summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("tmp/sigma_budget"))
    args = ap.parse_args()
    cfg = AppConfig()
    out_dir = Path(args.out_dir)

    setups = [("g_60_4", SS_CAM_CID), ("i_70_4", SS_CAM_CID), ("r_60_4", SS_CAM_CID), ("z_90_4", SS_CAM_CID)]
    gain_tables = [forensics_setup(426, s, cfg=cfg) for s, _ in setups]

    sem_traces = [trace_ensemble_sem_zero(426, s, t, cfg=cfg) for s, t in setups if s in ("g_60_4", "i_70_4")]
    sem_traces.append(trace_ensemble_sem_zero(426, "g_60_4", V0611_CID, cfg=cfg))

    # load joint (e) from A3
    summary_path = out_dir / "calibrator_chi2_summary.json"
    f_resid_e, sigma_floor_e = 0.0, 0.0065
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        setup424 = next((s for s in payload.get("setups", []) if int(s.get("draft_id", -1)) == 424), None)
        if setup424 and setup424.get("joint_fit_ensemble"):
            je = setup424["joint_fit_ensemble"]
            f_resid_e = float(je.get("f_resid", f_resid_e))
            sigma_floor_e = float(je.get("sigma_floor_mag", sigma_floor_e))

    hyp_cases = []
    for s, t in [("g_60_4", SS_CAM_CID), ("i_70_4", SS_CAM_CID), ("r_60_4", SS_CAM_CID), ("g_60_4", V0611_CID)]:
        fore = next((f for f in gain_tables if f["setup"] == s), None)
        if fore is None:
            continue
        hyp_cases.append(
            check_chi2_with_gain_hypothesis(
                426,
                s,
                t,
                cfg=cfg,
                gain_exp=float(fore["gain_expected_bin"]),
                rn_exp=float(fore["read_noise_expected_bin"]),
                f_resid_e=f_resid_e,
                sigma_floor_e=sigma_floor_e,
            )
        )

    payload = {
        "gain_rn_forensics": gain_tables,
        "ensemble_sem_traces": sem_traces,
        "hypothesis_chi2_cases": hyp_cases,
    }
    path = write_summary_json(payload, out_dir / "bin4_sigma_forensics.json")
    print(path)


if __name__ == "__main__":
    main()
