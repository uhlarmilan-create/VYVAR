#!/usr/bin/env python3
"""Stage 3 Part 1 measurements: check-star chi2_red, variance budget, D1-2 linearity."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src_py", REPO / "dev", REPO):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import AppConfig  # noqa: E402
from mag_constants import MAG_ERR_SCALE  # noqa: E402
from photometry_core import (  # noqa: E402
    ERR_BKG_SOURCE_COL,
    SIGMA_BKG_AP_COL,
    _photometric_error_with_bkg_mode,
    _sky_pp_for_photometric_error,
)
from sigma_budget import (  # noqa: E402
    OSBORN_CY_DEFAULT,
    resolve_rig_scintillation_params,
    scintillation_sigma,
)
from sigma_floor_core import mag_sigma_to_rel, resolve_sigma_sys_mag  # noqa: E402

# Reuse chi2 helper from existing harness
sys.path.insert(0, str(REPO / "dev" / "scripts"))
from chi2_sigma_gate import (  # noqa: E402
    _proc_aperture_area_px,
    _relative_flux_sigma_with_bkg,
    load_proc_row_for_source,
    reduced_chi2_constant,
)

SETUP = "NoFilter_60_2"
T1_TARGETS = [
    "1485540612577549568",
    "1485552329248338816",
    "1485574899299782528",
    "1485609538212672000",
    "1485913828055470592",
]
T1_CHI2_PRIOR = [0.577, 1.265, 1.345, 1.578, 1.477]


def _proc_cache(proc_dir: Path) -> dict[str, pd.DataFrame]:
    from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

    cache: dict[str, pd.DataFrame] = {}
    for proc_path in proc_dir.glob("proc_*.csv"):
        try:
            df = pd.read_csv(proc_path, low_memory=False, dtype={"catalog_id": str})
            id_col = "catalog_id" if "catalog_id" in df.columns else "name"
            df["_nid"] = df[id_col].map(lambda x: str(normalize_gaia_source_id(x) or "").strip())
            cache[proc_path.name] = df
        except Exception:  # noqa: BLE001
            continue
    return cache


def _proc_row_cached(cache: dict[str, pd.DataFrame], source_file: str, catalog_id: str) -> pd.Series | None:
    from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

    df = cache.get(str(source_file).strip())
    if df is None:
        return None
    cid = str(normalize_gaia_source_id(catalog_id) or "").strip()
    id_col = "catalog_id" if "catalog_id" in df.columns else "name"
    if "_nid" not in df.columns:
        df = df.copy()
        df["_nid"] = df[id_col].map(lambda x: str(normalize_gaia_source_id(x) or "").strip())
    sub = df.loc[df["_nid"] == cid]
    return None if sub.empty else sub.iloc[0]


def _photon_err_rel(row: pd.Series, *, gain: float = 1.0, read_noise: float = 10.0) -> float:
    flux = float(pd.to_numeric(row.get("dao_flux"), errors="coerce"))
    if not (math.isfinite(flux) and flux > 0):
        return float("nan")
    sky = float(_sky_pp_for_photometric_error(row))
    area = _proc_aperture_area_px(row)
    sig_bkg = float(pd.to_numeric(row.get(SIGMA_BKG_AP_COL), errors="coerce"))
    sig_bkg_ap = sig_bkg if math.isfinite(sig_bkg) else None
    err_bkg_source = str(row.get(ERR_BKG_SOURCE_COL, "")).strip() or None
    err_rel, _ = _relative_flux_sigma_with_bkg(
        flux,
        sky,
        area,
        sigma_bkg_ap=sig_bkg_ap,
        err_bkg_source=err_bkg_source,
        gain=gain,
        read_noise=read_noise,
    )
    return float(err_rel)


def _decompose_target_lc_err(
    lc_df: pd.DataFrame,
    proc_cache: dict[str, pd.DataFrame],
    target_cid: str,
    *,
    sigma_sys_mag: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(lc_df)
    ep = np.full(n, np.nan)
    sem = np.full(n, np.nan)
    sys = np.full(n, mag_sigma_to_rel(float(sigma_sys_mag)))
    err_mag = np.full(n, np.nan)
    for i, sf in enumerate(lc_df["source_file"].astype(str).tolist()):
        row = _proc_row_cached(proc_cache, sf, target_cid)
        err_lc = float(pd.to_numeric(lc_df.iloc[i]["err"], errors="coerce"))
        if math.isfinite(err_lc) and err_lc > 0:
            err_mag[i] = MAG_ERR_SCALE * err_lc
        if row is None:
            continue
        er = _photon_err_rel(row)
        if math.isfinite(er) and er > 0:
            ep[i] = er
        if math.isfinite(err_mag[i]) and math.isfinite(ep[i]):
            diff = err_mag[i] ** 2 - (MAG_ERR_SCALE * ep[i]) ** 2 - (MAG_ERR_SCALE * sys[i]) ** 2
            sem[i] = math.sqrt(diff) / MAG_ERR_SCALE if diff > 0 else 0.0
    return ep, sem, sys, err_mag


def _check_err_mag(
    check_cid: str,
    source_files: list[str],
    proc_cache: dict[str, pd.DataFrame],
    sem_rel: np.ndarray,
    sys_rel: np.ndarray,
) -> np.ndarray:
    out = np.full(len(source_files), np.nan)
    for i, sf in enumerate(source_files):
        row = _proc_row_cached(proc_cache, sf, check_cid)
        if row is None:
            continue
        ep = _photon_err_rel(row)
        if not math.isfinite(ep):
            continue
        sr = float(sys_rel[i]) if i < len(sys_rel) else float("nan")
        sm = float(sem_rel[i]) if i < len(sem_rel) else float("nan")
        if not math.isfinite(sr):
            sr = 0.0
        if not math.isfinite(sm):
            sm = 0.0
        comb = math.sqrt(ep * ep + sm * sm + sr * sr)
        out[i] = MAG_ERR_SCALE * comb
    return out


def measure_check_stars(
    draft_root: Path,
    *,
    setup: str,
    proc_dir: Path,
    cfg: AppConfig,
    rig_params,
    equipment_id: int | None,
    proc_cache: dict[str, pd.DataFrame],
) -> dict:
    lc_dir = draft_root / "platesolve" / setup / "photometry" / "lightcurves"
    sigma_sys = resolve_sigma_sys_mag(equipment_id, cfg, rig_label=setup.split("_")[0])
    rows: list[dict] = []
    for ck_path in sorted(lc_dir.glob("check_kmag_*.csv")):
        target_cid = ck_path.stem.replace("check_kmag_", "")
        lc_path = lc_dir / f"lightcurve_{target_cid}.csv"
        if not lc_path.is_file():
            continue
        ck = pd.read_csv(ck_path, low_memory=False)
        lc = pd.read_csv(lc_path, low_memory=False)
        if ck.empty or lc.empty:
            continue
        check_cid = str(ck["check_catalog_id"].iloc[0]).strip()
        kmag = pd.to_numeric(ck["kmag"], errors="coerce").to_numpy(dtype=float)
        src = ck["source_file"].astype(str).tolist()
        ep, sem, sys, _ = _decompose_target_lc_err(lc, proc_cache, target_cid, sigma_sys_mag=sigma_sys)
        # Align sem/sys to check sidecar rows via source_file
        lc_idx = {str(s).strip(): i for i, s in enumerate(lc["source_file"].astype(str).tolist())}
        sem_ck = np.array(
            [sem[lc_idx[str(s).strip()]] if str(s).strip() in lc_idx else float("nan") for s in src]
        )
        sys_ck = np.full(len(src), mag_sigma_to_rel(float(sigma_sys)))
        err_mag_check = _check_err_mag(check_cid, src, proc_cache, sem_ck, sys_ck)
        err_mag_target = np.array(
            [
                MAG_ERR_SCALE * float(pd.to_numeric(lc.iloc[lc_idx[str(s).strip()]]["err"], errors="coerce"))
                if str(s).strip() in lc_idx
                else float("nan")
                for s in src
            ],
            dtype=float,
        )
        _, dof, chi2_red, _ = reduced_chi2_constant(kmag, err_mag_check)
        _, _, chi2_red_target_err, _ = reduced_chi2_constant(kmag, err_mag_target)
        ep_ck = []
        for s in src:
            row = _proc_row_cached(proc_cache, s, check_cid)
            ep_ck.append(_photon_err_rel(row) if row is not None else float("nan"))
        ep_ck = np.array(ep_ck, dtype=float)
        frac_p = (
            float(np.nanmean(ep_ck ** 2 / ((err_mag_check / MAG_ERR_SCALE) ** 2)))
            if np.isfinite(err_mag_check).any()
            else float("nan")
        )
        frac_s = (
            float(np.nanmean(sys_ck ** 2 / ((err_mag_check / MAG_ERR_SCALE) ** 2)))
            if np.isfinite(err_mag_check).any()
            else float("nan")
        )
        frac_e = (
            float(np.nanmean(sem_ck ** 2 / ((err_mag_check / MAG_ERR_SCALE) ** 2)))
            if np.isfinite(err_mag_check).any()
            else float("nan")
        )
        am = float(pd.to_numeric(lc["airmass"].median(), errors="coerce"))
        if not math.isfinite(am) or am < 1:
            am = 1.2
        scint = scintillation_sigma(
            telescope_diameter_m=rig_params.telescope_diameter_m,
            airmass=am,
            exposure_s=rig_params.exposure_s,
            altitude_m=rig_params.altitude_m,
            c_y=rig_params.c_y,
        )
        rows.append(
            {
                "target_cid": target_cid,
                "check_cid": check_cid,
                "n": int(np.sum(np.isfinite(kmag) & np.isfinite(err_mag_check) & (err_mag_check > 0))),
                "chi2_red": chi2_red,
                "chi2_red_target_err_proxy": chi2_red_target_err,
                "dof": dof,
                "err_median_mag": float(np.nanmedian(err_mag_check)),
                "frac_var_photon": frac_p,
                "frac_var_sys": frac_s,
                "frac_var_sem": frac_e,
                "scint_would_be_rel": scint,
            }
        )
    df = pd.DataFrame(rows)
    med = float(df["chi2_red"].median()) if not df.empty else float("nan")
    med_proxy = float(df["chi2_red_target_err_proxy"].median()) if not df.empty else float("nan")
    return {
        "n_check_fields": len(rows),
        "median_chi2_red": med,
        "median_chi2_red_target_err_proxy": med_proxy,
        "per_field": rows,
    }


def measure_target_chi2_invalid(draft_root: Path, setup: str) -> list[dict]:
    lc_dir = draft_root / "platesolve" / setup / "photometry" / "lightcurves"
    out = []
    for tid, prior in zip(T1_TARGETS, T1_CHI2_PRIOR):
        p = lc_dir / f"lightcurve_{tid}.csv"
        if not p.is_file():
            continue
        lc = pd.read_csv(p, low_memory=False)
        m = pd.to_numeric(lc["mag_calib_final"], errors="coerce").to_numpy(dtype=float)
        e = pd.to_numeric(lc["err"], errors="coerce").to_numpy(dtype=float) * MAG_ERR_SCALE
        _, dof, chi2_red, _ = reduced_chi2_constant(m, e)
        out.append({"target_id": tid, "n": dof + 1 if dof >= 0 else 0, "chi2_red_targets": chi2_red, "t1_prior": prior})
    return out


def measure_d1_2_linearity(
    proc_cache: dict[str, pd.DataFrame],
    masterstars_csv: Path,
    *,
    n_bins: int = 10,
) -> dict:
    ms = pd.read_csv(masterstars_csv, dtype=str, low_memory=False)
    id_col = "catalog_id" if "catalog_id" in ms.columns else "gaia_source_id"
    allowed = {str(x).strip() for x in ms[id_col].dropna().astype(str)}
    records: list[dict] = []
    for sf, df in proc_cache.items():
        if df.empty:
            continue
        sub = df[df["catalog_id"].astype(str).isin(allowed)].copy()
        if sub.empty:
            continue
        inst = -2.5 * np.log10(pd.to_numeric(sub["dao_flux"], errors="coerce").clip(lower=1e-12))
        cat = pd.to_numeric(sub["mag"], errors="coerce")
        peak = pd.to_numeric(sub["peak_max_adu"], errors="coerce")
        ok = np.isfinite(inst) & np.isfinite(cat) & np.isfinite(peak) & (peak > 0)
        if not ok.any():
            continue
        zp = float(np.nanmedian(inst[ok] - cat[ok]))
        resid = inst - cat - zp
        sat_lim = float(pd.to_numeric(sub["saturate_limit_adu"].iloc[0], errors="coerce"))
        for r, p in zip(resid[ok], peak[ok]):
            records.append({"source_file": sf, "residual": float(r), "peak_adu": float(p), "saturate_limit_adu": sat_lim})
    if not records:
        return {"error": "no proc records"}
    tab = pd.DataFrame(records)
    sat = float(tab["saturate_limit_adu"].median())
    hi = min(sat, float(tab["peak_adu"].quantile(0.995)))
    edges = np.linspace(0, hi, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi_b = edges[i], edges[i + 1]
        sub = tab[(tab["peak_adu"] >= lo) & (tab["peak_adu"] < hi_b)]
        if sub.empty:
            continue
        bins.append(
            {
                "peak_adu_lo": float(lo),
                "peak_adu_hi": float(hi_b),
                "n": int(len(sub)),
                "resid_mean": float(sub["residual"].mean()),
                "resid_std": float(sub["residual"].std(ddof=1)) if len(sub) > 1 else float("nan"),
                "resid_median": float(sub["residual"].median()),
            }
        )
    # Trend onset: first bin where |mean| > 2*std/sqrt(n) of lowest-peak reference
    ref = tab[tab["peak_adu"] <= tab["peak_adu"].quantile(0.25)]
    ref_scatter = float(ref["residual"].std(ddof=1)) if len(ref) > 2 else float("nan")
    trend_adu = None
    for b in bins:
        if b["peak_adu_lo"] < 500:
            continue
        se = b["resid_std"] / math.sqrt(b["n"]) if b["n"] > 0 and math.isfinite(b["resid_std"]) else float("nan")
        if math.isfinite(ref_scatter) and abs(b["resid_mean"]) > max(0.01, 2.0 * ref_scatter):
            trend_adu = b["peak_adu_lo"]
            break
    return {
        "n_points": int(len(tab)),
        "n_frames": int(tab["source_file"].nunique()),
        "saturate_limit_adu_median": sat,
        "resid_overall_std_mmag": float(tab["residual"].std(ddof=1) * 1000),
        "trend_onset_peak_adu": trend_adu,
        "bins": bins,
        "linearity_spec_imx294_imx571": "UNVERIFIED - manufacturer datasheet not retrieved this session",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-id", type=int, default=435)
    ap.add_argument("--setup", default=SETUP)
    ap.add_argument("--snapshot", action="store_true", help="Use snapshot draft tree")
    ap.add_argument("--out", type=Path, default=REPO / "tmp" / "audit_stage3_part1.json")
    args = ap.parse_args()

    cfg = AppConfig()
    name = f"draft_{args.draft_id:06d}_snapshot_skysurface_20260716" if args.snapshot else f"draft_{args.draft_id:06d}"
    draft_root = Path(cfg.archive_root) / "Drafts" / name
    proc_dir = draft_root / "detrended_aligned" / "lights" / args.setup
    meta_path = draft_root / "platesolve" / args.setup / "photometry" / "pipeline_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    rig = resolve_rig_scintillation_params(draft_id=args.draft_id, setup=args.setup, pipeline_meta=meta)
    equip = (meta.get("config_snapshot") or {}).get("equipment_id")
    print(f"Loading proc cache from {proc_dir}...", flush=True)
    proc_cache = _proc_cache(proc_dir)
    print(f"Proc cache: {len(proc_cache)} frames", flush=True)

    payload = {
        "draft_root": str(draft_root),
        "setup": args.setup,
        "rig_scintillation": rig.to_dict(),
        "osborn_c_y": OSBORN_CY_DEFAULT,
        "check_star_chi2": measure_check_stars(
            draft_root,
            setup=args.setup,
            proc_dir=proc_dir,
            cfg=cfg,
            rig_params=rig,
            equipment_id=equip,
            proc_cache=proc_cache,
        ),
        "target_chi2_invalid_baseline": measure_target_chi2_invalid(draft_root, args.setup),
        "d1_2_linearity_residual_vs_peak": measure_d1_2_linearity(
            proc_cache,
            draft_root / "platesolve" / args.setup / "masterstars_full_match.csv",
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.out)
    print(f"check fields={payload['check_star_chi2']['n_check_fields']} median chi2_red={payload['check_star_chi2']['median_chi2_red']:.4f}")


if __name__ == "__main__":
    main()
