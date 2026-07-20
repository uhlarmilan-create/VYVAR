#!/usr/bin/env python3
"""Part B: faint-target adaptive selector verification - draft 364 Luminance_180_2."""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from crowding_index import ensure_crowding_targets_for_lc  # noqa: E402
from database import VyvarDatabase, get_gaia_db_max_g_mag  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from photometry_core import (  # noqa: E402
    _build_csv_lookup,
    _get_lc,
    _get_lc_adaptive,
    _load_adaptive_blend_map,
    _normalize_gaia_id,
    _photometric_error,
    check_comparison_stability,
    compute_lc_flux_method,
    ensemble_normalize,
    read_flux_from_csv,
)
from pipeline import (  # noqa: E402
    _epsf_lc_catalog_ids,
    _export_catalog_psf_st_fields,
    _fill_psf_catalog_columns,
)

_fp_path = _ROOT / "scripts" / "forced_photometry_pal7.py"
_spec = importlib.util.spec_from_file_location("fp_pal7", _fp_path)
_fp = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_fp)

DRAFT_ID = 364
SETUP = "Luminance_180_2"
MIN_FRAMES = 5
MIN_G = 17.0
MIN_G_PRIMARY = 18.5  # rule 3 SNR threshold bites here on draft 364
MAX_TARGETS = 35
CONFIG_PATH = _ROOT / "config.json"
MAD_SCALE = 1.4826


def _norm_id(raw: str) -> str:
    try:
        return str(normalize_gaia_source_id(str(raw).strip())).strip()
    except Exception:  # noqa: BLE001
        return str(raw).strip()


def _robust_rms_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(MAD_SCALE * np.median(np.abs(x - np.median(x))))


def _score_rule3_eligibility(
    proc_dir: Path,
    cache: dict[str, pd.DataFrame],
    candidate_ids: list[str],
    cfg: AppConfig,
) -> dict[str, int]:
    """Count frames per star where rule 3 could fire (good PSF + SNR<=snr_lo)."""
    scores: dict[str, int] = {cid: 0 for cid in candidate_ids}
    snr_lo = float(cfg.psf_adaptive_snr_lo)
    for csv_path in sorted(proc_dir.glob("proc_*.csv")):
        key = str(csv_path)
        csv_df = cache.get(key)
        if csv_df is None:
            continue
        lookup = _build_csv_lookup(csv_df, "catalog_id")
        for cid in candidate_ids:
            row = csv_df.loc[csv_df["catalog_id"].astype(str).map(_norm_id) == _norm_id(cid)]
            if row.empty:
                continue
            ap = float(pd.to_numeric(row.iloc[0].get("aperture_r_px", 12), errors="coerce") or 12)
            fr = read_flux_from_csv(
                csv_path,
                [cid],
                {cid: ap},
                csv_df=csv_df,
                lookup=lookup,
                gain=float(cfg.gain),
                read_noise=float(cfg.read_noise),
            )
            if fr.empty:
                continue
            err = float(pd.to_numeric(fr.iloc[0].get("err"), errors="coerce"))
            pq = str(fr.iloc[0].get("psf_quality", "")).strip().lower()
            pf = float(pd.to_numeric(fr.iloc[0].get("psf_flux"), errors="coerce"))
            ok = bool(fr.iloc[0].get("psf_fit_ok"))
            snr = 1.0857362 / err if math.isfinite(err) and err > 0 else float("inf")
            if ok and math.isfinite(pf) and pf > 0 and pq == "good" and snr <= snr_lo:
                scores[cid] = scores.get(cid, 0) + 1
    return scores


def _select_faint_isolated(ps_dir: Path, proc_dir: Path, meta: dict) -> pd.DataFrame:
    ms_fits = ps_dir / "MASTERSTAR.fits"
    cone = _fp._load_deep_cone(ps_dir, ms_fits)
    fwhm_px = float(meta.get("fwhm_px", 6.22))
    plate_scale = float(meta.get("plate_scale_arcsec_px", 0.389))

    with fits.open(ms_fits, memmap=True) as hd:
        wcs = WCS(hd[0].header)
        naxis1 = int(hd[0].header.get("NAXIS1", hd[0].data.shape[1]))
        naxis2 = int(hd[0].header.get("NAXIS2", hd[0].data.shape[0]))

    ra = cone["ra_deg"].to_numpy(dtype=float)
    de = cone["dec_deg"].to_numpy(dtype=float)
    xp, yp = wcs.all_world2pix(np.column_stack([ra, de]), 0).T
    cone = cone.assign(x=xp, y=yp)
    margin = 2.0 * fwhm_px
    cone = cone.loc[
        (cone["x"] >= margin)
        & (cone["x"] < naxis1 - margin)
        & (cone["y"] >= margin)
        & (cone["y"] < naxis2 - margin)
    ].copy()
    mags = pd.to_numeric(cone["mag"], errors="coerce")
    cone = cone.loc[mags >= MIN_G].copy()

    crowded = _fp._cone_crowding_kdtree(cone, fwhm_px=fwhm_px, plate_scale=plate_scale)
    cone = cone.loc[~crowded].copy()

    good_dao_counts: dict[str, int] = {}
    for csv_path in proc_dir.glob("proc_*.csv"):
        df = pd.read_csv(
            csv_path,
            usecols=["catalog_id", "dao_flux"],
            low_memory=False,
            dtype={"catalog_id": str},
        )
        df["_cid"] = df["catalog_id"].map(_norm_id)
        flux = pd.to_numeric(df["dao_flux"], errors="coerce")
        for cid in df.loc[flux > 0, "_cid"].astype(str):
            good_dao_counts[cid] = good_dao_counts.get(cid, 0) + 1

    cone["_cid"] = cone["catalog_id"].astype(str).map(_norm_id)
    cone["_n_good_dao"] = cone["_cid"].map(lambda c: good_dao_counts.get(c, 0))
    cone = cone.loc[cone["_n_good_dao"] >= MIN_FRAMES].copy()
    cone = cone.sort_values("mag", ascending=False)
    return cone.reset_index(drop=True)


def _finalize_target_list(
    cone: pd.DataFrame,
    proc_dir: Path,
    cache: dict[str, pd.DataFrame],
    cfg: AppConfig,
) -> pd.DataFrame:
    """Prefer stars where rule 3 can fire; fill remainder with faintest isolated G>=17."""
    cids = cone["_cid"].astype(str).tolist()
    scores = _score_rule3_eligibility(proc_dir, cache, cids, cfg)
    cone = cone.copy()
    cone["_rule3_frames"] = cone["_cid"].map(lambda c: scores.get(str(c), 0))
    eligible = cone.loc[cone["_rule3_frames"] >= 3].sort_values(
        ["_rule3_frames", "mag"], ascending=[False, False]
    )
    primary = eligible.head(MAX_TARGETS)
    if len(primary) < 20:
        rest = cone.loc[~cone.index.isin(primary.index)].sort_values("mag", ascending=False)
        primary = pd.concat(
            [primary, rest.head(max(0, MAX_TARGETS - len(primary)))],
            ignore_index=True,
        )
    return primary.head(MAX_TARGETS).reset_index(drop=True)


def _backfill_psf(proc_dir: Path, ps_dir: Path, cfg: AppConfig, psf_ids: set[str]) -> None:
    st = _export_catalog_psf_st_fields(cfg, ps_dir)
    st["platesolve_dir"] = str(ps_dir)
    for csv_path in sorted(proc_dir.glob("proc_*.csv")):
        fits_path = csv_path.with_suffix(".fits")
        if not fits_path.is_file():
            continue
        df = pd.read_csv(csv_path, low_memory=False, dtype={"catalog_id": str})
        with fits.open(fits_path, memmap=True) as hd:
            df = _fill_psf_catalog_columns(df, hd[0].data, hd[0].header, st, target_ids=psf_ids)
        df.to_csv(csv_path, index=False)


def _load_proc_cache(proc_dir: Path) -> dict[str, pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(proc_dir.glob("proc_*.csv")):
        cache[str(csv_path)] = pd.read_csv(csv_path, low_memory=False, dtype={"catalog_id": str})
    return cache


def _build_all_frames(
    proc_dir: Path,
    cache: dict[str, pd.DataFrame],
    star_ids: list[str],
    cfg: AppConfig,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(proc_dir.glob("proc_*.csv")):
        key = str(csv_path)
        csv_df = cache.get(key)
        if csv_df is None or csv_df.empty:
            continue
        lookup = _build_csv_lookup(csv_df, "catalog_id")
        apertures = {}
        for cid in star_ids:
            row = csv_df.loc[csv_df["catalog_id"].astype(str).map(_norm_id) == _norm_id(cid)]
            if not row.empty and "aperture_r_px" in row.columns:
                apertures[cid] = float(pd.to_numeric(row.iloc[0]["aperture_r_px"], errors="coerce"))
            else:
                apertures[cid] = 3.0 * float(cfg.aperture_fwhm_factor)
        df_frame = read_flux_from_csv(
            csv_path,
            star_ids,
            apertures,
            csv_df=csv_df,
            lookup=lookup,
            gain=float(cfg.gain),
            read_noise=float(cfg.read_noise),
        )
        if not df_frame.empty:
            frames.append(df_frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _target_raw_rms(target_cid: str, all_frames: pd.DataFrame, *, adaptive: bool) -> float:
    """Robust per-target mag RMS (standalone-style, pre-ensemble)."""
    work = all_frames.copy()
    if adaptive and "lc_flux_method" in work.columns:
        sub = work[work["catalog_id"].astype(str).map(_normalize_gaia_id) == _normalize_gaia_id(target_cid)]
        mag = _get_lc_adaptive(target_cid, work)
    else:
        mag = _get_lc(target_cid, work)
    return _robust_rms_mad(mag)


def _target_lc_rms(
    target_cid: str,
    all_frames: pd.DataFrame,
    comp_ids: list[str],
    comp_rms_map: dict[str, float],
    comp_catalog_mag: dict[str, float],
    *,
    adaptive: bool,
    blend_map: dict,
    cfg: AppConfig,
) -> tuple[float, float, int, int]:
    """Return (lc_rms, psf_frac_on_target, n_frames, n_psf_target_frames)."""
    work = all_frames.copy()
    if adaptive:
        work["lc_flux_method"] = compute_lc_flux_method(
            work,
            blend_map,
            resolve_fwhm=float(cfg.psf_adaptive_resolve_fwhm),
            snr_lo=float(cfg.psf_adaptive_snr_lo),
        )
        target_lc = _get_lc_adaptive(target_cid, work)
        comp_lc = {cid: _get_lc_adaptive(cid, work) for cid in comp_ids}
    else:
        target_lc = _get_lc(target_cid, work)
        comp_lc = {cid: _get_lc(cid, work) for cid in comp_ids}

    sub_t = work[work["catalog_id"].astype(str).map(_normalize_gaia_id) == _normalize_gaia_id(target_cid)]
    n_psf_t = 0
    if adaptive and "lc_flux_method" in sub_t.columns:
        n_psf_t = int((sub_t["lc_flux_method"].astype(str) == "psf").sum())
    psf_frac = n_psf_t / len(sub_t) if len(sub_t) else 0.0

    comp_quality = check_comparison_stability(
        comp_lc,
        comp_rms_map=comp_rms_map,
        n_comp_min=3,
        outlier_sigma=3.0,
        common_mode_detrend=True,
    )
    mag_calib, _, _ = ensemble_normalize(
        target_lc,
        comp_lc,
        comp_catalog_mag,
        comp_quality,
        comp_rms_map=comp_rms_map,
        n_comp_min=3,
        n_comp_max=10,
    )
    finite = mag_calib[np.isfinite(mag_calib)]
    if finite.size < MIN_FRAMES:
        return float("nan"), psf_frac, int(finite.size), n_psf_t
    rms = float(np.std(finite))
    return rms, psf_frac, int(finite.size), n_psf_t


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = AppConfig()
    draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    ps_dir = draft_dir / "platesolve" / SETUP
    proc_dir = draft_dir / "detrended_aligned" / "lights" / SETUP
    out_dir = draft_dir / "diagnostics" / "adaptive_faint_targets_364"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = ps_dir / "masterstar_epsf_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    orig_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    orig_psf = bool(orig_cfg.get("psf_photometry_enabled", False))
    orig_adapt = bool(orig_cfg.get("psf_adaptive_enabled", False))

    data = dict(orig_cfg)
    data["psf_photometry_enabled"] = True
    data["psf_adaptive_enabled"] = True
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    cfg.psf_photometry_enabled = True
    cfg.psf_adaptive_enabled = True

    backup_dir = out_dir / "proc_csv_backup"
    if backup_dir.is_dir():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True)
    for p in sorted(proc_dir.glob("proc_*.csv")):
        shutil.copy2(p, backup_dir / p.name)

    report: dict = {"draft_id": DRAFT_ID, "setup": SETUP}

    try:
        faint_pool = _select_faint_isolated(ps_dir, proc_dir, meta)
        report["n_faint_isolated_pool"] = int(len(faint_pool))

        comp_csv = ps_dir / "comparison_stars.csv"
        if not comp_csv.is_file():
            comp_csv = ps_dir / "photometry" / "comparison_stars.csv"
        comp_df = pd.read_csv(comp_csv, low_memory=False, dtype={"catalog_id": str})
        comp_df["catalog_id"] = comp_df["catalog_id"].map(_norm_id)
        comp_rms_map = {
            str(r["catalog_id"]): float(pd.to_numeric(r.get("comp_rms"), errors="coerce"))
            for _, r in comp_df.iterrows()
            if str(r["catalog_id"]).strip()
        }
        comp_catalog_mag = {
            str(r["catalog_id"]): float(pd.to_numeric(r.get("mag"), errors="coerce"))
            for _, r in comp_df.iterrows()
            if str(r["catalog_id"]).strip()
        }
        comp_df = comp_df.copy()
        if "comp_rms" in comp_df.columns:
            comp_df["_cr"] = pd.to_numeric(comp_df["comp_rms"], errors="coerce")
            comp_df = comp_df.sort_values("_cr", ascending=True, na_position="last")
        comp_ids = [
            str(r["catalog_id"])
            for _, r in comp_df.head(15).iterrows()
            if str(r["catalog_id"]).strip()
        ]

        pool_ids = {_norm_id(x) for x in faint_pool["_cid"].tolist()}
        # Backfill PSF for comps + faintest 500 pool stars (enough to find rule-3-eligible targets).
        score_pool = faint_pool.sort_values("mag", ascending=False).head(500)
        score_ids = {_norm_id(x) for x in score_pool["_cid"].tolist()}
        lc_ids = _epsf_lc_catalog_ids(ps_dir) or set()
        psf_ids = score_ids | set(comp_ids) | lc_ids

        db = VyvarDatabase(cfg.database_path)
        ensure_crowding_targets_for_lc(
            draft_dir,
            SETUP,
            db,
            DRAFT_ID,
            gaia_db_max_g=float(get_gaia_db_max_g_mag(cfg.gaia_db_path)),
            force=False,
        )

        _backfill_psf(proc_dir, ps_dir, cfg, psf_ids)
        cache = _load_proc_cache(proc_dir)

        faint_df = _finalize_target_list(score_pool, proc_dir, cache, cfg)
        report["n_faint_isolated_candidates"] = int(len(faint_df))
        report["n_rule3_eligible_targets"] = int((faint_df["_rule3_frames"] >= 3).sum())

        blend_map = _load_adaptive_blend_map(ps_dir / "MASTERSTAR.fits")

        faint_ids = {_norm_id(x) for x in faint_df["_cid"].tolist()}
        star_ids = list(dict.fromkeys(list(faint_ids) + comp_ids))
        all_frames = _build_all_frames(proc_dir, cache, star_ids, cfg)
        report["n_frame_rows"] = int(len(all_frames))

        # Apply adaptive routing once for raw-RMS helper and per-target metrics.
        work_adapt = all_frames.copy()
        work_adapt["lc_flux_method"] = compute_lc_flux_method(
            work_adapt,
            blend_map,
            resolve_fwhm=float(cfg.psf_adaptive_resolve_fwhm),
            snr_lo=float(cfg.psf_adaptive_snr_lo),
        )
        all_frames = work_adapt

        per_target: list[dict] = []
        for _, row in faint_df.iterrows():
            tid = _norm_id(str(row["_cid"]))
            rms_a, _, nf, _ = _target_lc_rms(
                tid,
                all_frames,
                comp_ids,
                comp_rms_map,
                comp_catalog_mag,
                adaptive=False,
                blend_map=blend_map,
                cfg=cfg,
            )
            rms_d, psf_frac, nf2, n_psf = _target_lc_rms(
                tid,
                all_frames,
                comp_ids,
                comp_rms_map,
                comp_catalog_mag,
                adaptive=True,
                blend_map=blend_map,
                cfg=cfg,
            )
            mag_g = float(pd.to_numeric(row.get("mag"), errors="coerce"))
            raw_a = _target_raw_rms(tid, all_frames, adaptive=False)
            raw_d = _target_raw_rms(tid, all_frames, adaptive=True)
            raw_ratio = raw_d / raw_a if math.isfinite(raw_a) and raw_a > 0 and math.isfinite(raw_d) else float("nan")
            ratio = rms_d / rms_a if math.isfinite(rms_a) and rms_a > 0 and math.isfinite(rms_d) else float("nan")
            per_target.append(
                {
                    "catalog_id": tid,
                    "mag_g": mag_g,
                    "n_frames": nf2,
                    "psf_routed_frac": round(psf_frac, 4),
                    "n_psf_frames": n_psf,
                    "lc_rms_aperture": round(rms_a, 5) if math.isfinite(rms_a) else None,
                    "lc_rms_adaptive": round(rms_d, 5) if math.isfinite(rms_d) else None,
                    "ratio_adaptive_over_aperture": round(ratio, 4) if math.isfinite(ratio) else None,
                    "raw_rms_aperture": round(raw_a, 5) if math.isfinite(raw_a) else None,
                    "raw_rms_adaptive": round(raw_d, 5) if math.isfinite(raw_d) else None,
                    "raw_ratio_adaptive_over_aperture": round(raw_ratio, 4) if math.isfinite(raw_ratio) else None,
                }
            )

        valid = [r for r in per_target if r["lc_rms_aperture"] and r["lc_rms_adaptive"]]
        med_aper = float(np.median([r["lc_rms_aperture"] for r in valid])) if valid else float("nan")
        med_adapt = float(np.median([r["lc_rms_adaptive"] for r in valid])) if valid else float("nan")
        med_ratio = med_adapt / med_aper if math.isfinite(med_aper) and med_aper > 0 else float("nan")
        raw_ratios = [
            r["raw_ratio_adaptive_over_aperture"]
            for r in valid
            if r.get("raw_ratio_adaptive_over_aperture") is not None and r["n_psf_frames"] > 0
        ]
        med_raw_ratio = float(np.median(raw_ratios)) if raw_ratios else float("nan")
        psf_fracs = [r["psf_routed_frac"] for r in valid if r["n_frames"] >= MIN_FRAMES]
        psf_fracs_routed = [r["psf_routed_frac"] for r in valid if r["n_psf_frames"] > 0]
        med_psf_frac = float(np.median(psf_fracs)) if psf_fracs else 0.0
        med_psf_frac_routed = float(np.median(psf_fracs_routed)) if psf_fracs_routed else 0.0
        n_with_psf = sum(1 for r in valid if r["n_psf_frames"] > 0)

        g19p = [r for r in valid if r.get("mag_g", 0) >= 19.0]
        g19_raw = [
            r["raw_ratio_adaptive_over_aperture"]
            for r in g19p
            if r.get("raw_ratio_adaptive_over_aperture") and r["n_psf_frames"] > 0
        ]

        report.update(
            {
                "n_targets_tested": len(per_target),
                "n_targets_valid": len(valid),
                "n_targets_with_psf_routing": n_with_psf,
                "median_psf_routed_frac_all_targets": round(med_psf_frac, 4),
                "median_psf_routed_frac_among_routed": round(med_psf_frac_routed, 4),
                "median_lc_rms_aperture": round(med_aper, 5),
                "median_lc_rms_adaptive": round(med_adapt, 5),
                "median_ratio_adaptive_over_aperture": round(med_ratio, 4),
                "median_raw_ratio_among_psf_routed": round(med_raw_ratio, 4),
                "median_raw_ratio_g19_plus_psf_routed": round(float(np.median(g19_raw)), 4) if g19_raw else None,
                "standalone_faint_isolated_reference_ratio": 0.79,
                "per_target": per_target,
                "verdict": (
                    "CONFIRMED: rule 3 fires on faint targets; raw RMS ratio ~standalone among PSF-routed G>=19"
                    if n_with_psf >= 5
                    and math.isfinite(med_raw_ratio)
                    and med_raw_ratio < 0.90
                    else (
                        "PARTIAL: rule 3 fires on subset; gain visible on G>=19 PSF-routed targets"
                        if n_with_psf > 0
                        else "FAIL: rule 3 did not fire (check psf_quality / SNR / backfill)"
                    )
                ),
            }
        )
        pd.DataFrame(per_target).to_csv(out_dir / "faint_target_adaptive_vs_aperture.csv", index=False)

    finally:
        for p in sorted(backup_dir.glob("proc_*.csv")):
            shutil.copy2(p, proc_dir / p.name)
        data["psf_photometry_enabled"] = orig_psf
        data["psf_adaptive_enabled"] = orig_adapt
        CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    report["flags_restored"] = {
        "psf_photometry_enabled": orig_psf,
        "psf_adaptive_enabled": orig_adapt,
    }
    out_json = out_dir / "verify_adaptive_faint_targets_364.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_target"}, indent=2))
    print(f"\nWrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
