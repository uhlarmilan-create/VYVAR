#!/usr/bin/env python3
"""DAO-PHYS-1/2: read-only physical basis measurement for DAO detection thresholds.

Uses stored MASTERSTAR frames and pipeline DAO helpers (import only; no pipeline runs).
DAO-PHYS-2 adds noise-scale correction, Q-statistic, persistence, SNR-floor curves.
Output: JSON summary to stdout; use --write-json for a file path.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from param_resolver import (  # noqa: E402
    _binning_from_header,
    _scale_bin1_db_for_header,
    resolve_gain,
    resolve_read_noise,
)
from pipeline import (  # noqa: E402
    _dao_auto_binning_factor,
    _dao_convolved_background_rms_adu,
    _dao_detection_threshold_adu,
    _dao_xy_binned_to_full,
    _mean_bin2d_for_dao,
    _prefilter_dao_table_brightest,
)
from utils import DAO_STAR_FINDER_NO_ROUNDNESS_FILTER, dao_detection_fwhm_pixels  # noqa: E402

PHI = lambda k: 0.5 * (1.0 + math.erf(k / math.sqrt(2.0)))  # noqa: E731


@dataclass
class DraftSpec:
    draft_id: str
    label: str
    masterstar_fits: Path
    masterstars_csv: Path
    infolog: Path | None
    recorded_pass1: int | None
    recorded_merged: int | None
    recorded_after_snr: int | None
    draft_db_id: int
    anchored: bool = True
    reference_source: str = ""
    plate_scale_arcsec_px: float | None = None
    lights_dir: Path | None = None
    proc_csv_dir: Path | None = None
    draft_manifest: Path | None = None
    reference_light_fits: Path | None = None


DRAFTS: list[DraftSpec] = [
    DraftSpec(
        draft_id="draft_000501",
        label="501 Newton V",
        masterstar_fits=REPO / "Archive/Drafts/draft_000501/platesolve/V_60_2/MASTERSTAR.fits",
        masterstars_csv=REPO / "Archive/Drafts/draft_000501/platesolve/V_60_2/masterstars_full_match.csv",
        infolog=REPO / "Archive/Drafts/draft_000501/infolog_20260805_113441.txt",
        recorded_pass1=1654,
        recorded_merged=1670,
        recorded_after_snr=1668,
        draft_db_id=501,
        reference_source="TOI-1131.01.b_2025-04-22_23-59-57_V.fits (infolog)",
        plate_scale_arcsec_px=1.3010910511796954,
        lights_dir=REPO / "Archive/Drafts/draft_000501/non_calibrated/lights/V_60_2",
        proc_csv_dir=REPO / "Archive/Drafts/draft_000501/detrended_aligned/lights/V_60_2",
        draft_manifest=REPO / "Archive/Drafts/draft_000501/draft_manifest.json",
        reference_light_fits=REPO
        / "Archive/Drafts/draft_000501/non_calibrated/lights/V_60_2/TOI-1131.01.b_2025-04-22_23-05-09_V.fits",
    ),
    DraftSpec(
        draft_id="draft_000435_snapshot_skysurface_20260716",
        label="435 wide anchor",
        masterstar_fits=REPO
        / "Archive/Drafts/draft_000435_snapshot_skysurface_20260716/platesolve/NoFilter_60_2/MASTERSTAR.fits",
        masterstars_csv=REPO
        / "Archive/Drafts/draft_000435_snapshot_skysurface_20260716/platesolve/NoFilter_60_2/masterstars_full_match.csv",
        infolog=REPO
        / "Archive/Drafts/draft_000435_snapshot_skysurface_20260716/infolog_20260716_123126.txt",
        recorded_pass1=2552,
        recorded_merged=3777,
        recorded_after_snr=2951,
        draft_db_id=435,
        reference_source="MASTERSTAR.fits platesolve/NoFilter_60_2 (infolog DAO on ref frame)",
        plate_scale_arcsec_px=9.772785373657268,
        lights_dir=REPO
        / "Archive/Drafts/draft_000435_snapshot_skysurface_20260716/detrended_aligned/lights/NoFilter_60_2",
        proc_csv_dir=REPO
        / "Archive/Drafts/draft_000435_snapshot_skysurface_20260716/detrended_aligned/lights/NoFilter_60_2",
        draft_manifest=REPO
        / "Archive/Drafts/draft_000435_snapshot_skysurface_20260716/draft_manifest.json",
        reference_light_fits=REPO
        / "Archive/Drafts/draft_000435_snapshot_skysurface_20260716/platesolve/NoFilter_60_2/MASTERSTAR.fits",
    ),
    DraftSpec(
        draft_id="draft_000500",
        label="500 wide",
        masterstar_fits=REPO / "Archive/Drafts/draft_000500/platesolve/NoFilter_60_2/MASTERSTAR.fits",
        masterstars_csv=REPO / "Archive/Drafts/draft_000500/platesolve/NoFilter_60_2/masterstars_full_match.csv",
        infolog=None,
        recorded_pass1=None,
        recorded_merged=None,
        recorded_after_snr=4122,
        draft_db_id=500,
        anchored=False,
        reference_source="MASTERSTAR.fits (no infolog; unanchored)",
        plate_scale_arcsec_px=9.7741059180782,
        lights_dir=REPO / "Archive/Drafts/draft_000500/detrended_aligned/lights/NoFilter_60_2",
        proc_csv_dir=REPO / "Archive/Drafts/draft_000500/detrended_aligned/lights/NoFilter_60_2",
        draft_manifest=REPO / "Archive/Drafts/draft_000500/draft_manifest.json",
        reference_light_fits=REPO / "Archive/Drafts/draft_000500/platesolve/NoFilter_60_2/MASTERSTAR.fits",
    ),
]

PERSISTENCE_MAX_FRAMES = 12
PERSISTENCE_DAO_CAP = 200

K_SWEEPS = (3.0, 3.78, 4.0, 4.5, 5.0)
LAMBDA_M = 550e-9
FWHM_ATM_ARCSEC = 2.5  # stated assumption, not fitted
PIXEL_FLOOR_PX = 1.2


def _load_frame(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        hdr = hdul[0].header.copy()
    return data, hdr


def _psf_predicted_q(fwhm_px: float) -> float:
    sigma = float(fwhm_px) / 2.3548
    if sigma <= 0:
        return float("nan")
    return 4.0 * math.exp(-0.5 / sigma**2) + 4.0 * math.exp(-1.0 / sigma**2)


def _source_mask(shape: tuple[int, int], xs: np.ndarray, ys: np.ndarray, *, radius_px: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w), dtype=bool)
    r2 = float(radius_px) ** 2
    for x, y in zip(xs, ys, strict=False):
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        dx = xx - float(x)
        dy = yy - float(y)
        mask |= (dx * dx + dy * dy) <= r2
    return mask


def _block_mean_pool(arr: np.ndarray, factor: int) -> np.ndarray:
    f = int(factor)
    h, w = arr.shape
    h2, w2 = h - (h % f), w - (w % f)
    sl = arr[:h2, :w2]
    return sl.reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3))


def _noise_binning_slope(
    data0: np.ndarray,
    *,
    star_x: np.ndarray,
    star_y: np.ndarray,
    fwhm_px: float,
    factors: tuple[int, ...] = (1, 2, 4, 8, 16),
) -> dict[str, Any]:
    """log sigma vs log bin factor; slope -1 => white, > -1 => correlated excess."""
    mask_r = max(3.0, 2.0 * float(fwhm_px))
    src = _source_mask(data0.shape, star_x, star_y, radius_px=mask_r)
    clean = np.where(src, np.nan, data0.astype(np.float64))
    sigmas: list[float] = []
    used: list[int] = []
    for f in factors:
        if min(data0.shape) // f < 32:
            continue
        pooled = _block_mean_pool(np.nan_to_num(clean, nan=0.0), f)
        src_p = _block_mean_pool(src.astype(np.float32), f) > 0.05
        vals = pooled[~src_p]
        vals = vals[np.isfinite(vals)]
        if vals.size < 100:
            continue
        _, _, s = sigma_clipped_stats(vals, sigma=3.0, maxiters=2)
        if math.isfinite(s) and s > 0:
            sigmas.append(float(s))
            used.append(int(f))
    if len(sigmas) < 2:
        return {"usable": False, "reason": "insufficient bin levels", "n_levels": len(sigmas)}
    lx = np.log(np.asarray(used, dtype=np.float64))
    ly = np.log(np.asarray(sigmas, dtype=np.float64))
    slope = float(np.polyfit(lx, ly, 1)[0])
    return {
        "usable": True,
        "factors": used,
        "sigma_clip_by_factor": sigmas,
        "log_log_slope": slope,
        "verdict": "white" if abs(slope + 1.0) <= 0.15 else "correlated",
    }


def _gain_rn_convention(hdr: fits.Header, db: VyvarDatabase, equipment_id: int | None) -> dict[str, Any]:
    db_g, db_rn = (None, None)
    if equipment_id is not None:
        db_g, db_rn = db.get_equipment_cosmic_params(int(equipment_id))
    binning = _binning_from_header(hdr)
    g_res = resolve_gain(hdr, db=db, equipment_id=equipment_id)
    rn_res = resolve_read_noise(hdr, db=db, equipment_id=equipment_id)
    g_scaled = _scale_bin1_db_for_header(float(db_g), hdr, exponent=2, param_label="gain") if db_g else None
    rn_scaled = _scale_bin1_db_for_header(float(db_rn), hdr, exponent=1, param_label="read_noise") if db_rn else None
    note = (
        "DB values documented as bin1 per param_resolver; header XBINNING scales DB fallbacks. "
        "Header-mapped gain bypasses DB scaling."
    )
    qhy_double = None
    if equipment_id == 1 and binning == 2 and db_rn is not None:
        # JOURNAL draft_303 NoFilter_60_2 session values match DB at bin2 (gain=3.17 RN=7.6).
        qhy_double = {
            "likely_double_count_rn": abs(float(rn_res.value) - 2.0 * float(db_rn)) < 0.5,
            "evidence": "draft_303 NoFilter_60_2 measured RN=7.6 e- at bin2; resolver scales bin1->bin2 again",
            "rn_if_db_already_bin2": float(db_rn),
        }
    c3_ok = None
    if equipment_id == 2 and db_g is not None and db_rn is not None and binning == 2:
        c3_ok = {
            "gain_db_x4_matches_header": math.isclose(float(db_g) * 4.0, float(g_res.value), rel_tol=0.05),
            "rn_db_x2_matches_resolved": math.isclose(float(db_rn) * 2.0, float(rn_res.value), rel_tol=0.05),
            "mutually_consistent": True,
        }
    return {
        "header_binning": binning,
        "db_gain_bin1": db_g,
        "db_read_noise_bin1": db_rn,
        "db_gain_scaled_for_header": g_scaled,
        "db_read_noise_scaled_for_header": rn_scaled,
        "resolved_gain": float(g_res.value),
        "resolved_gain_source": getattr(g_res, "source", str(g_res)),
        "resolved_read_noise": float(rn_res.value),
        "resolved_read_noise_source": getattr(rn_res, "source", str(rn_res)),
        "c3_26000_bin2_check": c3_ok,
        "qhy294mm_rn_check": qhy_double,
        "note": note,
    }


def _masterstar_provenance(
    spec: DraftSpec,
    *,
    master_hdr: fits.Header,
    std_clip_master: float,
) -> dict[str, Any]:
    ncombine = master_hdr.get("NCOMBINE") or master_hdr.get("COMNUM")
    manifest_mode = None
    if spec.draft_manifest and spec.draft_manifest.exists():
        try:
            manifest_mode = json.loads(spec.draft_manifest.read_text(encoding="utf-8")).get("calibration_mode")
        except Exception:  # noqa: BLE001
            manifest_mode = None
    single_std = None
    single_path = None
    if spec.reference_light_fits and spec.reference_light_fits.exists():
        single_path = str(spec.reference_light_fits)
        arr, _ = _load_frame(spec.reference_light_fits)
        _, _, single_std = sigma_clipped_stats(arr, sigma=3.0, maxiters=3)
    return {
        "ncombine_header": int(ncombine) if ncombine not in (None, "") else 1,
        "is_stack": bool(ncombine not in (None, "", 1, "1")),
        "calibration_mode": manifest_mode,
        "reference_light_fits": single_path,
        "std_clip_master_adu": float(std_clip_master),
        "std_clip_single_light_adu": float(single_std) if single_std is not None else None,
        "ratio_master_over_single": (
            float(std_clip_master / single_std) if single_std and single_std > 0 else None
        ),
        "verdict": "single reference frame (not noise-reducing stack)",
    }


def _bpm_presence(spec: DraftSpec) -> dict[str, Any]:
    draft_root = spec.masterstar_fits.parents[2]
    sidecars = sorted(draft_root.rglob("*_dark_bpm.json"))
    proc_dir = spec.proc_csv_dir or spec.lights_dir
    on_bad_frac = None
    n_proc = 0
    n_bad = 0
    if proc_dir and proc_dir.exists():
        for proc in proc_dir.glob("proc_*.csv"):
            try:
                df = pd.read_csv(proc, usecols=lambda c: c == "on_bad_column")
            except (ValueError, pd.errors.EmptyDataError):
                continue
            if "on_bad_column" not in df.columns:
                continue
            n_proc += len(df)
            n_bad += int(df["on_bad_column"].astype(str).str.lower().isin(["true", "1", "t", "yes"]).sum())
        if n_proc > 0:
            on_bad_frac = float(n_bad / n_proc)
    return {
        "dark_bpm_sidecars_in_draft": [str(p) for p in sidecars],
        "n_dark_bpm_sidecars": len(sidecars),
        "proc_rows_on_bad_column_frac": on_bad_frac,
        "proc_rows_sampled": n_proc,
    }


def _pixel_q(data0: np.ndarray, x: float, y: float, *, bg: float = 0.0) -> float:
    h, w = data0.shape
    ix, iy = int(round(x)), int(round(y))
    if ix < 1 or iy < 1 or ix >= w - 1 or iy >= h - 1:
        return float("nan")
    c = float(data0[iy, ix]) - bg
    if c <= 0:
        return float("nan")
    nbr = (
        data0[iy - 1, ix - 1],
        data0[iy - 1, ix],
        data0[iy - 1, ix + 1],
        data0[iy, ix - 1],
        data0[iy, ix + 1],
        data0[iy + 1, ix - 1],
        data0[iy + 1, ix],
        data0[iy + 1, ix + 1],
    )
    num = sum(float(v) - bg for v in nbr)
    return float(num / c)


def _q_tradeoff(q: np.ndarray, dao_mask: np.ndarray, gaia_mask: np.ndarray) -> list[dict[str, Any]]:
    q = np.asarray(q, dtype=np.float64)
    dao_mask = np.asarray(dao_mask, dtype=bool)
    gaia_mask = np.asarray(gaia_mask, dtype=bool)
    finite = np.isfinite(q)
    if int(finite.sum()) == 0:
        return []
    qs = np.unique(np.round(np.linspace(float(np.nanmin(q[finite])), float(np.nanmax(q[finite])), 25), 3))
    rows: list[dict[str, Any]] = []
    n_dao = max(int(dao_mask.sum()), 1)
    n_g = max(int(gaia_mask.sum()), 1)
    for lb in qs:
        keep = q >= lb
        rows.append(
            {
                "q_lower_bound": float(lb),
                "dao_only_removed_frac": float(np.sum(dao_mask & ~keep) / n_dao),
                "gaia_removed_frac": float(np.sum(gaia_mask & ~keep) / n_g),
            }
        )
    return rows


def _persistence_native(
    spec: DraftSpec,
    ms: pd.DataFrame,
    dao_only_mask: pd.Series,
    *,
    cfg: AppConfig,
) -> dict[str, Any]:
    if spec.draft_id != "draft_000501" or not spec.lights_dir or not spec.lights_dir.exists():
        return {"usable": False, "reason": "persistence test only implemented for draft_501 with lights_dir"}
    fits_files = sorted(
        p for p in spec.lights_dir.glob("*.fits") if p.name.upper() != "MASTERSTAR.FITS"
    )[:PERSISTENCE_MAX_FRAMES]
    if len(fits_files) < 3:
        return {"usable": False, "reason": f"need >=3 light frames, found {len(fits_files)}"}

    xs = pd.to_numeric(ms["x"], errors="coerce").to_numpy(dtype=np.float64)
    ys = pd.to_numeric(ms["y"], errors="coerce").to_numpy(dtype=np.float64)
    dao = dao_only_mask.to_numpy(dtype=bool)
    gaia = ~dao

    det_by_frame: list[np.ndarray] = []
    for fp in fits_files:
        arr, hdr_f = _load_frame(fp)
        _, data_dao, _, _, bfac_f, _, _ = _preprocess_dao(arr)
        fw_f = _fwhm_from_header(hdr_f, cfg)
        f_eff = max(1.2, fw_f / float(bfac_f))
        _, _, _, _, tbl = _dao_pass1_count(
            data_dao,
            fwhm_eff=f_eff,
            n_equiv=float(cfg.dao_detection_n_equiv),
            cfg=cfg,
            bfac=bfac_f,
            max_catalog_rows=PERSISTENCE_DAO_CAP,
            return_table=True,
        )
        if tbl is None or len(tbl) == 0:
            det_by_frame.append(np.zeros((0, 2), dtype=np.float64))
            continue
        xd = np.asarray(tbl["x_centroid"], dtype=np.float64)
        yd = np.asarray(tbl["y_centroid"], dtype=np.float64)
        if int(bfac_f) > 1:
            xd, yd = _dao_xy_binned_to_full(xd, yd, int(bfac_f))
        det_by_frame.append(np.column_stack([xd, yd]))

    def _recurrence(pop_mask: np.ndarray, *, max_n: int = 200) -> dict[str, float]:
        idx = np.where(pop_mask & np.isfinite(xs) & np.isfinite(ys))[0]
        if idx.size > max_n:
            idx = np.random.default_rng(501).choice(idx, size=max_n, replace=False)
        if idx.size == 0:
            return {"n": 0, "mean_recurrence_frac": float("nan"), "p95_recurrence_frac": float("nan")}
        rates: list[float] = []
        for j in idx:
            ix, iy = float(xs[j]), float(ys[j])
            hits = 0
            for dets in det_by_frame:
                if dets.size == 0:
                    continue
                if float(np.min(np.hypot(dets[:, 0] - ix, dets[:, 1] - iy))) <= 1.0:
                    hits += 1
            rates.append(hits / float(len(fits_files)))
        arr_r = np.asarray(rates, dtype=np.float64)
        return {
            "n": int(idx.size),
            "mean_recurrence_frac": float(np.mean(arr_r)),
            "p95_recurrence_frac": float(np.percentile(arr_r, 95)),
        }

    return {
        "usable": True,
        "n_frames": len(fits_files),
        "frames_sample": [p.name for p in fits_files[:5]],
        "method": "one DAO pass/frame (cap 200); reference (x,y) recurrence within 1px on native lights",
        "dao_only": _recurrence(dao, max_n=200),
        "gaia_matched": _recurrence(gaia, max_n=120),
    }


def _negative_flux_analysis(ms: pd.DataFrame, dao_only_mask: pd.Series, *, med: float) -> dict[str, Any]:
    flux = pd.to_numeric(ms.get("flux"), errors="coerce")
    peak = pd.to_numeric(ms.get("peak_max_adu"), errors="coerce")
    x = pd.to_numeric(ms.get("x"), errors="coerce")
    y = pd.to_numeric(ms.get("y"), errors="coerce")
    pop = dao_only_mask
    neg = pop & (flux < 0)
    n_neg = int(neg.sum())
    if n_neg == 0:
        return {"n_negative": 0}
    peak_above = peak[neg] - float(med)
    # spatial spread vs all dao_only
    xd = x[pop & np.isfinite(x)]
    yd = y[pop & np.isfinite(y)]
    xn = x[neg & np.isfinite(x)]
    yn = y[neg & np.isfinite(y)]
    spatial = {
        "neg_centroid_x": float(np.mean(xn)) if len(xn) else float("nan"),
        "neg_centroid_y": float(np.mean(yn)) if len(yn) else float("nan"),
        "dao_only_centroid_x": float(np.mean(xd)) if len(xd) else float("nan"),
        "dao_only_centroid_y": float(np.mean(yd)) if len(yd) else float("nan"),
        "neg_spatial_std_px": float(np.hypot(np.std(xn), np.std(yn))) if len(xn) > 2 else float("nan"),
        "dao_only_spatial_std_px": float(np.hypot(np.std(xd), np.std(yd))) if len(xd) > 2 else float("nan"),
    }
    return {
        "n_negative": n_neg,
        "frac_of_dao_only": float(n_neg / max(int(pop.sum()), 1)),
        "peak_above_median_adu_median": float(np.median(peak_above)),
        "peak_above_median_adu_p05": float(np.percentile(peak_above, 5)) if n_neg else float("nan"),
        "interpretation": (
            "negative flux with positive peak_max_adu implies local background > aperture sum "
            "(oversubtracted neighbourhood or mis-centred aperture on a weak gradient)"
        ),
        "spatial": spatial,
    }


def _snr_floor_analysis(
    ms: pd.DataFrame,
    dao_only_mask: pd.Series,
    *,
    med: float,
    std_clip: float,
    k_configured: float,
    recorded_merged: int | None,
    recorded_after_snr: int | None,
) -> dict[str, Any]:
    peak = pd.to_numeric(ms.get("peak_max_adu"), errors="coerce")
    sigma_u = (peak - float(med)) / float(std_clip) if std_clip > 0 else pd.Series(dtype=float)
    dao = dao_only_mask.to_numpy(dtype=bool)
    gaia = ~dao
    floor_adu = float(med) + float(k_configured) * float(std_clip)
    below = peak < floor_adu
    ks = np.unique(np.round(np.linspace(0.5, 12.0, 24), 2))
    curve: list[dict[str, Any]] = []
    n_dao = max(int(dao.sum()), 1)
    n_g = max(int(gaia.sum()), 1)
    for k in ks:
        rem_d = float(np.sum(dao & (sigma_u < k).to_numpy()) / n_dao)
        rem_g = float(np.sum(gaia & (sigma_u < k).to_numpy()) / n_g)
        curve.append({"k_floor": float(k), "dao_only_removed_frac": rem_d, "gaia_removed_frac": rem_g})
    best = None
    for pt in curve:
        if pt["gaia_removed_frac"] <= 0.02 and pt["dao_only_removed_frac"] >= 0.5:
            best = pt
            break
    reconciled = None
    if recorded_merged is not None and recorded_after_snr is not None:
        reconciled = {
            "recorded_merged": int(recorded_merged),
            "recorded_after_snr": int(recorded_after_snr),
            "recorded_removed": int(recorded_merged) - int(recorded_after_snr),
            "floor_adu_at_k_configured": floor_adu,
            "below_floor_in_current_csv": int(below.sum()),
            "note": "current CSV is post-SNR; pre-SNR merged catalogue not archived for 435",
        }
    return {
        "k_configured": float(k_configured),
        "floor_adu": floor_adu,
        "sigma_units": {
            "dao_only": _percentiles(sigma_u[dao].to_numpy()),
            "gaia_matched": _percentiles(sigma_u[gaia].to_numpy()),
        },
        "k_floor_curve": curve,
        "knee_ge50_dao_le2_gaia": best,
        "snr_reconciliation": reconciled,
    }


def _measure_phys2(
    spec: DraftSpec,
    *,
    cfg: AppConfig,
    db: VyvarDatabase,
    arr: np.ndarray,
    hdr: fits.Header,
    data0: np.ndarray,
    ms: pd.DataFrame,
    dao_only_mask: pd.Series,
    med: float,
    std_clip: float,
    rms_conv: float,
    gain: float,
    read_noise: float,
    base_fw: float,
    fwhm_eff: float,
    bfac: int,
    equipment_id: int | None,
    tbl: Any,
    joined: pd.DataFrame,
) -> dict[str, Any]:
    sigma_pred = math.sqrt(max(med, 0.0) * gain + read_noise**2) / gain if gain > 0 else float("nan")
    R_rms = rms_conv / sigma_pred if sigma_pred > 0 else float("nan")
    R_std = std_clip / sigma_pred if sigma_pred > 0 else float("nan")

    xs = pd.to_numeric(ms["x"], errors="coerce").to_numpy(dtype=np.float64)
    ys = pd.to_numeric(ms["y"], errors="coerce").to_numpy(dtype=np.float64)

    q_psf = _psf_predicted_q(base_fw)
    q_vals: dict[str, Any] = {}
    q_curve: list[dict[str, Any]] = []
    if not joined.empty:
        q_arr = np.asarray(
            [_pixel_q(data0, r["x_det"], r["y_det"]) for _, r in joined.iterrows()],
            dtype=np.float64,
        )
        joined_q = joined.copy()
        joined_q["Q"] = q_arr
        for pop, mask in (("dao_only", joined_q["dao_only"]), ("gaia_matched", ~joined_q["dao_only"])):
            sub = joined_q.loc[mask, "Q"].to_numpy(dtype=np.float64)
            q_vals[pop] = {**_percentiles(sub), "n": int(np.isfinite(sub).sum())}
        q_curve = _q_tradeoff(
            joined_q["Q"].to_numpy(),
            joined_q["dao_only"].to_numpy(),
            (~joined_q["dao_only"]).to_numpy(),
        )

    persist = _persistence_native(spec, ms, dao_only_mask, cfg=cfg)

    return {
        "noise_R": {
            "R_rms_conv": R_rms,
            "R_std_clip": R_std,
            "sigma_pred_pedestal0_adu": sigma_pred,
            "pedestal_conclusion_survives": bool(R_std < 0.5),
        },
        "noise_binning_slope": _noise_binning_slope(data0, star_x=xs, star_y=ys, fwhm_px=base_fw),
        "gain_rn_convention": _gain_rn_convention(hdr, db, equipment_id),
        "masterstar_provenance": _masterstar_provenance(spec, master_hdr=hdr, std_clip_master=std_clip),
        "bpm": _bpm_presence(spec),
        "Q_statistic": {
            "psf_predicted": q_psf,
            "fwhm_px_used": base_fw,
            "populations": q_vals,
            "tradeoff_curve": q_curve,
        },
        "persistence": persist,
        "negative_flux": _negative_flux_analysis(ms, dao_only_mask, med=med),
        "snr_floor": _snr_floor_analysis(
            ms,
            dao_only_mask,
            med=med,
            std_clip=std_clip,
            k_configured=float(cfg.masterstar_prematch_peak_sigma_floor),
            recorded_merged=spec.recorded_merged,
            recorded_after_snr=spec.recorded_after_snr,
        ),
    }


def _preprocess_dao(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, int, float, float]:
    """Sky-subtract, bin, return (data0, data_dao, med, std_raw, bfac, fwhm_eff, rms_conv)."""
    mean, med, std = sigma_clipped_stats(arr, sigma=3.0, maxiters=3)
    data0 = np.nan_to_num((arr - med).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    bfac = _dao_auto_binning_factor(*data0.shape)
    data_dao, bfac = _mean_bin2d_for_dao(data0, bfac)
    return data0, data_dao, float(med), float(std), int(bfac), float(mean), float(std)


def _fwhm_from_header(hdr: fits.Header, cfg: AppConfig) -> float:
    fb = float(cfg.sips_dao_fwhm_px)
    base = dao_detection_fwhm_pixels(hdr, configured_fallback=fb)
    return float(base)


def _dao_pass1_count(
    data_dao: np.ndarray,
    *,
    fwhm_eff: float,
    n_equiv: float,
    cfg: AppConfig,
    bfac: int = 1,
    max_catalog_rows: int = 15000,
    invert: bool = False,
    return_table: bool = False,
) -> tuple[int, int, float, Any]:
    img = -data_dao if invert else data_dao
    rms_conv, _rel = _dao_convolved_background_rms_adu(img, fwhm_px=fwhm_eff)

    class _Cfg:
        dao_detection_n_equiv = float(n_equiv)

    thr, _ = _dao_detection_threshold_adu(rms_conv, cfg=_Cfg(), dao_threshold_sigma=float(n_equiv))
    finder = DAOStarFinder(
        fwhm=float(fwhm_eff),
        threshold=float(thr),
        scale_threshold=False,
        n_brightest=None,
        **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    )
    tbl = finder(img)
    n_raw = int(len(tbl)) if tbl is not None else 0
    if tbl is not None and len(tbl) > 0:
        tbl = _prefilter_dao_table_brightest(tbl, max(int(max_catalog_rows) * 12, 36_000))
        # Full-frame coordinates for catalogue join (pipeline stores binned centroids scaled back).
        if int(bfac) > 1:
            xf, yf = _dao_xy_binned_to_full(
                np.asarray(tbl["x_centroid"], dtype=np.float64),
                np.asarray(tbl["y_centroid"], dtype=np.float64),
                int(bfac),
            )
            tbl["x_centroid"] = xf
            tbl["y_centroid"] = yf
    n_pref = int(len(tbl)) if tbl is not None else 0
    if return_table:
        return n_raw, n_pref, float(rms_conv), float(thr), tbl
    return n_raw, n_pref, float(rms_conv), float(thr)


def _percentiles(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"median": float("nan"), "p05": float("nan"), "p95": float("nan")}
    return {
        "median": float(np.median(x)),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
    }


def _match_detections_to_csv(tbl: Any, ms: pd.DataFrame, *, max_sep_px: float = 5.0) -> pd.DataFrame:
    if tbl is None or len(tbl) == 0:
        return pd.DataFrame()
    det = pd.DataFrame(
        {
            "x_det": np.asarray(tbl["x_centroid"], dtype=np.float64),
            "y_det": np.asarray(tbl["y_centroid"], dtype=np.float64),
            "flux_det": np.asarray(tbl["flux"], dtype=np.float64),
            "sharpness": np.asarray(tbl["sharpness"], dtype=np.float64),
            "roundness1": np.asarray(tbl["roundness1"], dtype=np.float64),
            "roundness2": np.asarray(tbl["roundness2"], dtype=np.float64),
        }
    )
    xs = pd.to_numeric(ms["x"], errors="coerce").to_numpy(dtype=np.float64)
    ys = pd.to_numeric(ms["y"], errors="coerce").to_numpy(dtype=np.float64)
    cid = ms.get("catalog_id", pd.Series([""] * len(ms))).astype(str).str.strip()
    dao_only = cid.eq("") | cid.eq("nan")
    out_rows: list[dict[str, Any]] = []
    for i in range(len(det)):
        dx = xs - det.loc[i, "x_det"]
        dy = ys - det.loc[i, "y_det"]
        sep = np.hypot(dx, dy)
        j = int(np.argmin(sep)) if sep.size else -1
        if j < 0 or float(sep[j]) > max_sep_px:
            continue
        out_rows.append(
            {
                **det.iloc[i].to_dict(),
                "ms_idx": j,
                "dao_only": bool(dao_only.iloc[j]),
                "sep_px": float(sep[j]),
            }
        )
    return pd.DataFrame(out_rows)


def _tradeoff_curve(sharp: np.ndarray, dao_mask: np.ndarray, gaia_mask: np.ndarray) -> list[dict[str, Any]]:
    sharp = np.asarray(sharp, dtype=np.float64)
    dao_mask = np.asarray(dao_mask, dtype=bool)
    gaia_mask = np.asarray(gaia_mask, dtype=bool)
    if sharp.size == 0:
        return []
    qs = np.unique(np.round(np.linspace(float(np.nanmin(sharp)), float(np.nanmax(sharp)), 25), 4))
    rows: list[dict[str, Any]] = []
    n_dao = max(int(dao_mask.sum()), 1)
    n_g = max(int(gaia_mask.sum()), 1)
    for ub in qs:
        if not math.isfinite(ub):
            continue
        keep = sharp <= ub
        rows.append(
            {
                "sharpness_ub": float(ub),
                "dao_only_removed_frac": float(np.sum(dao_mask & ~keep) / n_dao),
                "gaia_removed_frac": float(np.sum(gaia_mask & ~keep) / n_g),
            }
        )
    return rows


def _roundness_tradeoff(r1: np.ndarray, r2: np.ndarray, dao_mask: np.ndarray, gaia_mask: np.ndarray) -> list[dict[str, Any]]:
    roundness = np.hypot(np.asarray(r1, dtype=np.float64), np.asarray(r2, dtype=np.float64))
    dao_mask = np.asarray(dao_mask, dtype=bool)
    gaia_mask = np.asarray(gaia_mask, dtype=bool)
    if roundness.size == 0:
        return []
    qs = np.unique(np.round(np.linspace(0.0, float(np.nanpercentile(roundness, 99)), 20), 3))
    rows: list[dict[str, Any]] = []
    n_dao = max(int(dao_mask.sum()), 1)
    n_g = max(int(gaia_mask.sum()), 1)
    for ub in qs:
        keep = roundness <= ub
        rows.append(
            {
                "roundness_ub": float(ub),
                "dao_only_removed_frac": float(np.sum(dao_mask & ~keep) / n_dao),
                "gaia_removed_frac": float(np.sum(gaia_mask & ~keep) / n_g),
            }
        )
    return rows


def _ptc_pedestal(arr: np.ndarray, gain: float, read_noise: float, *, n_bins: int = 16) -> dict[str, Any]:
    """Photon transfer curve on spatial bins; returns fit metadata or unusable flag."""
    a = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(a)
    if int(np.count_nonzero(finite)) < 1000:
        return {"usable": False, "reason": "too few finite pixels"}
    h, w = a.shape
    bh, bw = max(h // 4, 1), max(w // 4, 1)
    medians: list[float] = []
    sigmas: list[float] = []
    for y0 in range(0, h - bh + 1, bh):
        for x0 in range(0, w - bw + 1, bw):
            sl = a[y0 : y0 + bh, x0 : x0 + bw]
            slf = sl[np.isfinite(sl)]
            if slf.size < 64:
                continue
            _, m, s = sigma_clipped_stats(slf, sigma=3.0, maxiters=2)
            if math.isfinite(m) and math.isfinite(s) and s > 0:
                medians.append(float(m))
                sigmas.append(float(s))
    if len(medians) < 6:
        return {"usable": False, "reason": f"only {len(medians)} bins with leverage"}
    med = np.asarray(medians, dtype=np.float64)
    sig = np.asarray(sigmas, dtype=np.float64)
    var = sig**2
    # fit var = (1/g)*median + (RN/g)^2  -> slope ~ 1/g
    A = np.column_stack([med, np.ones_like(med)])
    coef, _, _, _ = np.linalg.lstsq(A, var, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    g_fit = (1.0 / slope) if slope > 1e-12 else float("nan")
    rn2_g2 = intercept
    pedestal_x_intercept = (-intercept / slope) if abs(slope) > 1e-12 else float("nan")
    return {
        "usable": True,
        "n_bins": len(medians),
        "slope_var_vs_median": slope,
        "intercept": intercept,
        "gain_from_slope_e_adu": g_fit,
        "rn_e_from_intercept": math.sqrt(max(rn2_g2, 0.0)) * g_fit if g_fit > 0 else float("nan"),
        "pedestal_adu_x_intercept": pedestal_x_intercept,
    }


def _fwhm_predicted_arcsec(diameter_mm: float, lambda_m: float = LAMBDA_M, fwhm_atm: float = FWHM_ATM_ARCSEC) -> float:
    d_m = float(diameter_mm) / 1000.0
    fwhm_diff_arcsec = 1.03 * lambda_m / d_m * (180.0 / math.pi) * 3600.0
    return math.sqrt(fwhm_atm**2 + fwhm_diff_arcsec**2)


def _equipment(db: VyvarDatabase, draft_id: int) -> dict[str, Any]:
    row = db.conn.execute(
        "SELECT ID_EQUIPMENTS, ID_TELESCOPE FROM OBS_DRAFT WHERE ID=?", (int(draft_id),)
    ).fetchone()
    if not row:
        return {}
    eid, tid = int(row[0]), int(row[1])
    eq = db.conn.execute(
        "SELECT CAMERANAME, PIXELSIZE, GAIN_ADU, READNOISE_E FROM EQUIPMENTS WHERE ID=?", (eid,)
    ).fetchone()
    tel = db.conn.execute(
        "SELECT TELESCOPENAME, DIAMETER, FOCAL FROM TELESCOPE WHERE ID=?", (tid,)
    ).fetchone()
    return {
        "equipment_id": eid,
        "telescope_id": tid,
        "camera": eq[0] if eq else None,
        "pixel_um": float(eq[1]) if eq and eq[1] else None,
        "gain_db": float(eq[2]) if eq and eq[2] is not None else None,
        "read_noise_db": float(eq[3]) if eq and eq[3] is not None else None,
        "telescope": tel[0] if tel else None,
        "diameter_mm": float(tel[1]) if tel and tel[1] else None,
        "focal_mm": float(tel[2]) if tel and tel[2] else None,
    }


def measure_draft(spec: DraftSpec, cfg: AppConfig, db: VyvarDatabase) -> dict[str, Any]:
    arr, hdr = _load_frame(spec.masterstar_fits)
    ms = pd.read_csv(spec.masterstars_csv, low_memory=False)
    cid = ms.get("catalog_id", pd.Series([""] * len(ms))).astype(str).str.strip()
    dao_only_mask = cid.eq("") | cid.eq("nan")
    n_dao_only = int(dao_only_mask.sum())
    n_gaia = int((~dao_only_mask).sum())
    n_total_ms = int(len(ms))

    data0, data_dao, med, std_raw, bfac, _mean, _std = _preprocess_dao(arr)
    base_fw = _fwhm_from_header(hdr, cfg)
    fwhm_eff = max(1.2, base_fw / float(bfac))
    n_pix = int(arr.shape[0] * arr.shape[1])

    eq = _equipment(db, spec.draft_db_id)
    row = db.conn.execute("SELECT ID_EQUIPMENTS FROM OBS_DRAFT WHERE ID=?", (spec.draft_db_id,)).fetchone()
    eid = int(row[0]) if row else None
    g_res = resolve_gain(hdr, db=db, equipment_id=eid)
    rn_res = resolve_read_noise(hdr, db=db, equipment_id=eid)
    gain = float(g_res.value if hasattr(g_res, "value") else g_res)
    read_noise = float(rn_res.value if hasattr(rn_res, "value") else rn_res)

    _, n_repro, rms_conv, thr = _dao_pass1_count(
        data_dao,
        fwhm_eff=fwhm_eff,
        n_equiv=float(cfg.dao_detection_n_equiv),
        cfg=cfg,
        bfac=bfac,
    )
    anchor_ok = None
    anchor_pct = None
    if spec.recorded_pass1 is not None:
        anchor_pct = 100.0 * n_repro / float(spec.recorded_pass1)
        anchor_ok = abs(n_repro - spec.recorded_pass1) / float(spec.recorded_pass1) <= 0.03

    # inverted + k sweep
    k_curve: list[dict[str, Any]] = []
    for k in K_SWEEPS:
        _, n_det, rms_k, thr_k = _dao_pass1_count(
            data_dao, fwhm_eff=fwhm_eff, n_equiv=k, cfg=cfg, bfac=bfac
        )
        _, n_inv, _, _ = _dao_pass1_count(
            data_dao, fwhm_eff=fwhm_eff, n_equiv=k, cfg=cfg, bfac=bfac, invert=True
        )
        k_curve.append(
            {
                "k": k,
                "n_detected_pass1": n_det,
                "n_inverted": n_inv,
                "rms_conv": rms_k,
                "threshold_adu": thr_k,
            }
        )

    n_inverted_default = next(x["n_inverted"] for x in k_curve if abs(x["k"] - 3.78) < 0.01)
    n_positive_only = max(0, n_dao_only - n_inverted_default)

    sigma_psf = base_fw / 2.3548
    n_res = n_pix / (2.0 * math.pi * sigma_psf**2)
    n_fa_378 = n_res * (1.0 - PHI(3.78))

    # noise R (DAO-PHYS-1 used rms_conv; DAO-PHYS-2 adds std_clip - per-pixel comparable)
    sigma_meas_conv = rms_conv
    _, _, std_clip = sigma_clipped_stats(arr, sigma=3.0, maxiters=3)
    sigma_pred = math.sqrt(max(med, 0.0) * gain + read_noise**2) / gain if gain > 0 else float("nan")
    R_rms_conv = sigma_meas_conv / sigma_pred if sigma_pred > 0 else float("nan")
    R_std_clip = float(std_clip) / sigma_pred if sigma_pred > 0 else float("nan")
    se = max((float(std_clip) * gain) ** 2 - read_noise**2, 0.0)
    pedestal_inv = med - se / gain if gain > 0 else float("nan")
    ptc = _ptc_pedestal(arr, gain, read_noise)

    # shape stats: re-detect at k=3.78 with table
    _, _, _, _, tbl = _dao_pass1_count(
        data_dao,
        fwhm_eff=fwhm_eff,
        n_equiv=float(cfg.dao_detection_n_equiv),
        cfg=cfg,
        bfac=bfac,
        return_table=True,
    )
    joined = _match_detections_to_csv(tbl, ms)
    shape: dict[str, Any] = {}
    if not joined.empty:
        for pop, mask in (("dao_only", joined["dao_only"]), ("gaia_matched", ~joined["dao_only"])):
            sub = joined.loc[mask]
            shape[pop] = {
                "n_matched_det": int(len(sub)),
                "sharpness": _percentiles(sub["sharpness"].to_numpy()),
                "roundness1": _percentiles(sub["roundness1"].to_numpy()),
                "roundness2": _percentiles(sub["roundness2"].to_numpy()),
            }
        sharp_curve = _tradeoff_curve(
            joined["sharpness"].to_numpy(),
            joined["dao_only"].to_numpy(),
            (~joined["dao_only"]).to_numpy(),
        )
        round_curve = _roundness_tradeoff(
            joined["roundness1"].to_numpy(),
            joined["roundness2"].to_numpy(),
            joined["dao_only"].to_numpy(),
            (~joined["dao_only"]).to_numpy(),
        )
    else:
        sharp_curve, round_curve = [], []

    # trivial filters from CSV
    flux = pd.to_numeric(ms.get("flux"), errors="coerce")
    neg_flux = flux < 0
    snr_ok = ms.get("snr50_ok")
    edge_ok = ms.get("edge_safe_10px")
    if snr_ok is not None:
        snr_ok_b = snr_ok.astype(str).str.lower().isin(["true", "1", "t", "yes"])
    else:
        snr_ok_b = pd.Series([True] * len(ms))
    if edge_ok is not None:
        edge_ok_b = edge_ok.astype(str).str.lower().isin(["true", "1", "t", "yes"])
    else:
        edge_ok_b = pd.Series([True] * len(ms))

    def _frac(mask: pd.Series, pop: pd.Series) -> float:
        m = pop & mask
        d = pop.sum()
        return float(m.sum() / d) if d else float("nan")

    pop_dao = dao_only_mask
    pop_gaia = ~dao_only_mask
    union = neg_flux | (~snr_ok_b) | (~edge_ok_b)
    trivial = {
        "dao_only": {
            "n": n_dao_only,
            "neg_flux_frac": _frac(neg_flux, pop_dao),
            "fail_snr50_frac": _frac(~snr_ok_b, pop_dao),
            "fail_edge_frac": _frac(~edge_ok_b, pop_dao),
            "union_removed_frac": _frac(union, pop_dao),
        },
        "gaia_matched": {
            "n": n_gaia,
            "neg_flux_frac": _frac(neg_flux, pop_gaia),
            "fail_snr50_frac": _frac(~snr_ok_b, pop_gaia),
            "fail_edge_frac": _frac(~edge_ok_b, pop_gaia),
            "union_removed_frac": _frac(union, pop_gaia),
        },
    }

    # FWHM prediction
    d_mm = eq.get("diameter_mm") or float("nan")
    fwhm_pred_arcsec = _fwhm_predicted_arcsec(d_mm) if math.isfinite(d_mm) and d_mm > 0 else float("nan")
    scale = spec.plate_scale_arcsec_px
    if scale is None:
        try:
            sc = proj_plane_pixel_scales(WCS(hdr))
            scale = float(np.mean(np.abs(sc)) * 3600.0)
        except Exception:  # noqa: BLE001
            scale = float("nan")
    fwhm_pred_px = max(PIXEL_FLOOR_PX, fwhm_pred_arcsec / scale) if scale and scale > 0 else float("nan")

    phys2 = _measure_phys2(
        spec,
        cfg=cfg,
        db=db,
        arr=arr,
        hdr=hdr,
        data0=data0,
        ms=ms,
        dao_only_mask=dao_only_mask,
        med=med,
        std_clip=float(std_clip),
        rms_conv=rms_conv,
        gain=gain,
        read_noise=read_noise,
        base_fw=base_fw,
        fwhm_eff=fwhm_eff,
        bfac=bfac,
        equipment_id=eid,
        tbl=tbl,
        joined=joined,
    )

    return {
        "draft_id": spec.draft_id,
        "label": spec.label,
        "anchored": spec.anchored,
        "reference_source": spec.reference_source,
        "masterstar_fits": str(spec.masterstar_fits),
        "infolog": str(spec.infolog) if spec.infolog else None,
        "frame_shape": list(arr.shape),
        "n_pix": n_pix,
        "vy_fwhm_header": float(hdr.get("VY_FWHM", float("nan"))),
        "bfac": bfac,
        "fwhm_eff_px": fwhm_eff,
        "median_adu": med,
        "std_clip_adu": float(std_clip),
        "rms_conv_adu": rms_conv,
        "threshold_adu": thr,
        "recorded_pass1": spec.recorded_pass1,
        "reproduced_pass1": n_repro,
        "anchor_ok": anchor_ok,
        "anchor_pct": anchor_pct,
        "recorded_merged": spec.recorded_merged,
        "recorded_after_snr": spec.recorded_after_snr,
        "masterstars_rows": n_total_ms,
        "dao_only_rows": n_dao_only,
        "gaia_matched_rows": n_gaia,
        "inverted_decomposition": {
            "n_inverted_at_3.78": n_inverted_default,
            "dao_only_observed": n_dao_only,
            "n_positive_only_artifacts_est": n_positive_only,
            "n_real_below_catalog_cap_est": max(
                0, n_dao_only - n_inverted_default - n_positive_only
            ),
        },
        "k_curve": k_curve,
        "analytic": {
            "sigma_psf_px": sigma_psf,
            "n_res": n_res,
            "n_fa_gaussian_3.78": n_fa_378,
            "ratio_inverted_to_n_fa": n_inverted_default / n_fa_378 if n_fa_378 > 0 else float("nan"),
        },
        "noise": {
            "gain_e_adu": gain,
            "read_noise_e": read_noise,
            "gain_source": getattr(g_res, "source", str(g_res)),
            "read_noise_source": getattr(rn_res, "source", str(rn_res)),
            "sigma_meas_rms_conv_adu": sigma_meas_conv,
            "sigma_meas_std_clip_adu": float(std_clip),
            "sigma_pred_pedestal0_adu": sigma_pred,
            "R_rms_conv": R_rms_conv,
            "R_std_clip": R_std_clip,
            "R_pedestal0": R_rms_conv,
            "pedestal_inverted_adu": pedestal_inv,
            "ptc": ptc,
        },
        "shape": shape,
        "sharpness_tradeoff": sharp_curve,
        "roundness_tradeoff": round_curve,
        "trivial_filters": trivial,
        "fwhm_optics": {
            "diameter_mm": d_mm,
            "fwhm_atm_assumed_arcsec": FWHM_ATM_ARCSEC,
            "fwhm_predicted_arcsec": fwhm_pred_arcsec,
            "plate_scale_arcsec_px": scale,
            "fwhm_predicted_px": fwhm_pred_px,
            "fwhm_measured_px": base_fw,
            "ratio_measured_over_predicted": base_fw / fwhm_pred_px if fwhm_pred_px > 0 else float("nan"),
        },
        "equipment": eq,
        "phys2": phys2,
    }


def _interp_k_curve(k_curve: list[dict[str, Any]], k_req: float, key: str = "n_detected_pass1") -> float:
    pts = sorted((float(p["k"]), float(p[key])) for p in k_curve if math.isfinite(float(p["k"])))
    if not pts or not math.isfinite(k_req):
        return float("nan")
    if k_req <= pts[0][0]:
        return pts[0][1]
    if k_req >= pts[-1][0]:
        return pts[-1][1]
    for (k0, v0), (k1, v1) in zip(pts[:-1], pts[1:]):
        if k0 <= k_req <= k1:
            if k1 == k0:
                return v0
            t = (k_req - k0) / (k1 - k0)
            return v0 + t * (v1 - v0)
    return float("nan")


def _k_for_n_fa(n_res: float, n_fa_target: float) -> float:
    """Invert N_FA = N_res * (1-Phi(k)) for k."""
    if n_res <= 0 or n_fa_target <= 0:
        return float("nan")
    p = 1.0 - min(max(n_fa_target / n_res, 1e-12), 1.0 - 1e-12)
    # p = Phi(k) -> k = sqrt(2)*erfinv(2p-1)
    from scipy.special import erfinv

    return float(math.sqrt(2.0) * erfinv(2.0 * p - 1.0))


def build_decision_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in results:
        did = r["draft_id"]
        dao = r["dao_only_rows"]
        gaia = r["gaia_matched_rows"]
        triv = r["trivial_filters"]
        # negative flux
        rows.append(
            {
                "draft": did,
                "measure": "reject negative flux",
                "dao_only_removed_frac": triv["dao_only"]["neg_flux_frac"],
                "gaia_matched_lost_frac": triv["gaia_matched"]["neg_flux_frac"],
                "would_prevent_501_outcome": did == "draft_000501" and triv["dao_only"]["neg_flux_frac"] > 0.15,
                "risk": "low on Gaia if fraction small",
            }
        )
        # union trivial
        rows.append(
            {
                "draft": did,
                "measure": "union trivial (neg+snr50+edge)",
                "dao_only_removed_frac": triv["dao_only"]["union_removed_frac"],
                "gaia_matched_lost_frac": triv["gaia_matched"]["union_removed_frac"],
                "would_prevent_501_outcome": did == "draft_000501" and triv["dao_only"]["union_removed_frac"] > 0.25,
                "risk": "check Gaia loss",
            }
        )
        # R gate
        R = r["noise"].get("R_std_clip", r["noise"].get("R_pedestal0"))
        rows.append(
            {
                "draft": did,
                "measure": "noise-consistency R gate (frame-level)",
                "dao_only_removed_frac": "n/a (frame reject)",
                "gaia_matched_lost_frac": "n/a",
                "would_prevent_501_outcome": did == "draft_000501" and R < 0.5,
                "risk": f"R={R:.3f} flag pre-cal pedestal",
            }
        )
        # sharpness best point
        sc = r.get("sharpness_tradeoff") or []
        best = None
        for pt in sc:
            if pt["gaia_removed_frac"] <= 0.02 and pt["dao_only_removed_frac"] >= 0.5:
                best = pt
                break
        rows.append(
            {
                "draft": did,
                "measure": "restore sharpness upper bound",
                "dao_only_removed_frac": best["dao_only_removed_frac"] if best else "no knee found",
                "gaia_matched_lost_frac": best["gaia_removed_frac"] if best else "n/a",
                "would_prevent_501_outcome": did == "draft_000501" and bool(best),
                "risk": "comatic stars if bound too tight on roundness (disabled by design)",
            }
        )
        n_res = r["analytic"]["n_res"]
        for target in (1.0, 10.0):
            k_req = _k_for_n_fa(n_res, target)
            n_det = _interp_k_curve(r["k_curve"], k_req)
            n_inv = _interp_k_curve(r["k_curve"], k_req, key="n_inverted")
            rows.append(
                {
                    "draft": did,
                    "measure": f"FAR-derived k (N_FA={int(target)})",
                    "k_implied": k_req,
                    "dao_only_removed_frac": "indirect (lowers all detections)",
                    "n_detected_pass1_at_k": n_det,
                    "n_inverted_at_k": n_inv,
                    "would_prevent_501_outcome": "depth/purity trade-off -- not decided here",
                    "risk": "catalog depth loss",
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="DAO-PHYS-1/2 measurement")
    parser.add_argument("--write-json", type=Path, default=None)
    args = parser.parse_args()

    cfg = AppConfig()
    db = VyvarDatabase(cfg.database_path)
    try:
        results = [measure_draft(spec, cfg, db) for spec in DRAFTS]
        out = {"drafts": results, "decision_table": build_decision_table(results)}
        text = json.dumps(out, indent=2, default=str)
        if args.write_json:
            args.write_json.write_text(text, encoding="utf-8")
            print(f"wrote {args.write_json}")
        else:
            print(text)
    finally:
        db.conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
