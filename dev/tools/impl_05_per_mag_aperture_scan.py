#!/usr/bin/env python3
"""IMPL-05 Item B: magnitude-stratified scatter aperture + persisted flux ladder.

Re-photometers once (IMPL-04 never persisted per-star ladders), writes
``aperture_flux_ladder.parquet`` beside ``aperture_scatter_table.json``, decides
per mag bin with the IMPL-04 rule, emits a non-flat per-mag table, and runs
INV-APERTURE-* gates.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))
sys.path.insert(0, str(ROOT / "dev" / "scripts"))

from aperture_scatter_select import (  # noqa: E402
    LadderSpec,
    build_scatter_curve,
    evaluate_scatter_at_radius,
    measure_flux_ladder_frame,
    split_selection_holdout,
)
from config import AppConfig  # noqa: E402
from gaia_catalog_id import GAIA_PROC_CSV_READ_DTYPE, normalize_gaia_source_id  # noqa: E402
from photometry_core import (  # noqa: E402
    compute_fwhm_gaussian_for_aperture_catalog,
    enhance_catalog_dataframe_aperture_bpm,
)
from snr_cog_gates import evaluate_snr_cog_gates  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("impl05b")

DRAFT = ROOT / "Archive" / "Drafts" / "draft_000514"
SETUP = "NoFilter_60_2"
ALIGNED = DRAFT / "detrended_aligned" / "lights" / SETUP
PHOT = DRAFT / "platesolve" / SETUP / "photometry"
OUT_DIR = ROOT / "dev" / "results"
SEED = 51403
VARIABLE_CIDS = {
    "1498613634033133184",
    "1497343732462852864",
}
MIN_PER_BIN = 10
MAG_BIN_WIDTH = 1.0
TABLE_MAG_LO = 7.0
TABLE_MAG_HI = 18.0
TABLE_MAG_STEP = 0.5


def _nid(x: object) -> str:
    return str(normalize_gaia_source_id(x) or "").strip()


def _nn_distance_px(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    pts = np.column_stack([x, y])
    if len(pts) < 2:
        return np.full(len(pts), np.inf)
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    return d[:, 1]


def _fits_for_proc(csv_path: Path) -> Path:
    stem = csv_path.stem
    if stem.lower().startswith("proc_"):
        return csv_path.parent / f"{stem[5:]}.fits"
    return csv_path.parent / f"{stem}.fits"


def _frame_fwhm_px(hdr, default: float) -> float:
    for k in ("VY_FWHM", "FWHM", "SEEING"):
        try:
            v = float(hdr.get(k))
            if math.isfinite(v) and 0.5 < v < 30:
                return v
        except Exception:  # noqa: BLE001
            continue
    return float(default)


def sampling_noise_mmag(scatter_mmag: float, n_stars: int) -> float:
    if not (math.isfinite(scatter_mmag) and n_stars >= 2):
        return float("nan")
    return float(scatter_mmag) / math.sqrt(float(n_stars))


def decide_radius(curve_sel: dict, curve_hold: dict) -> dict:
    """IMPL-04 rule: genuine min else upper edge of contiguous flat region."""
    r = np.asarray(curve_sel["radii_px"], float)
    s = np.asarray(curve_sel["scatter_mmag"], float)
    n_med = int(np.nanmedian(curve_sel["n_stars"])) if curve_sel.get("n_stars") else 8
    finite = np.isfinite(r) & np.isfinite(s)
    r = r[finite]
    s = s[finite]
    if r.size == 0:
        return {"error": "no_finite_scatter"}
    i_min = int(np.argmin(s))
    r_min = float(r[i_min])
    s_min = float(s[i_min])
    noise = sampling_noise_mmag(s_min, max(2, n_med))
    tol = max(0.05 * s_min, 1.0 * noise if math.isfinite(noise) else 0.05 * s_min)
    lo = hi = i_min
    while lo > 0 and float(s[lo - 1]) <= s_min + tol:
        lo -= 1
    while hi + 1 < len(s) and float(s[hi + 1]) <= s_min + tol:
        hi += 1
    flat_lo, flat_hi = float(r[lo]), float(r[hi])
    rh = np.asarray(curve_hold["radii_px"], float)
    sh = np.asarray(curve_hold["scatter_mmag"], float)
    hold_at = float(np.interp(r_min, rh, sh)) if sh.size else float("nan")
    hold_min = float(np.nanmin(sh)) if sh.size else float("nan")
    hold_ok = (
        math.isfinite(hold_at)
        and math.isfinite(hold_min)
        and (hold_at <= hold_min + max(tol, noise if math.isfinite(noise) else tol))
    )
    genuine = (flat_hi - flat_lo) < 1.01 and hold_ok
    if genuine:
        chosen = r_min
        branch = "genuine_minimum"
    else:
        chosen = flat_hi
        branch = "flat_upper_edge"
    return {
        "branch": branch,
        "chosen_r_px": round(float(chosen), 3),
        "numerical_min_r_px": round(float(r_min), 3),
        "numerical_min_scatter_mmag": s_min,
        "flat_region_px": [round(flat_lo, 3), round(flat_hi, 3)],
        "flat_tol_mmag": tol,
        "sampling_noise_mmag": noise,
        "n_stars_median": n_med,
        "held_out_ok": hold_ok,
        "held_out_at_chosen_vs_hold_min": [hold_at, hold_min],
    }


def _load_field_catalog() -> pd.DataFrame:
    csv0 = sorted(ALIGNED.glob("proc_*.csv"))[0]
    df = pd.read_csv(csv0, low_memory=False, dtype={"catalog_id": str})
    df = df.copy()
    df["_nid"] = df["catalog_id"].map(_nid)
    df = df[df["_nid"].astype(str).str.len().gt(0)]
    df = df[~df["_nid"].isin(VARIABLE_CIDS)]
    for c in ("x", "y", "dao_flux", "phot_g_mean_mag", "mag"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isfinite(df["x"]) & np.isfinite(df["y"])]
    df["_g"] = df["phot_g_mean_mag"] if "phot_g_mean_mag" in df.columns else df.get("mag")
    df = df[np.isfinite(df["_g"])]
    df["_flux"] = df["dao_flux"] if "dao_flux" in df.columns else np.nan
    df = df.drop_duplicates(subset=["_nid"], keep="first")
    df["_nn_px"] = _nn_distance_px(df["x"].to_numpy(), df["y"].to_numpy())
    return df.reset_index(drop=True)


def _comp_mag_bins(comps: pd.DataFrame) -> list[tuple[float, float]]:
    g = pd.to_numeric(comps["phot_g_mean_mag"], errors="coerce")
    g = g[np.isfinite(g)]
    if g.empty:
        return []
    lo0 = float(math.floor(float(g.min()) / MAG_BIN_WIDTH) * MAG_BIN_WIDTH)
    hi0 = float(math.ceil(float(g.max()) / MAG_BIN_WIDTH) * MAG_BIN_WIDTH)
    out: list[tuple[float, float]] = []
    lo = lo0
    while lo < hi0 - 1e-9:
        hi = lo + MAG_BIN_WIDTH
        n = int(((g >= lo) & (g < hi)).sum())
        if n > 0:
            out.append((lo, hi))
        lo = hi
    return out


def _stratified_eval_pool(
    field: pd.DataFrame,
    *,
    fwhm_px: float,
    isolation_fwhm: float,
    min_per_bin: int = MIN_PER_BIN,
) -> tuple[pd.DataFrame, dict]:
    """B2: >=~10 isolated eval stars in every mag bin that has real comps."""
    comps = pd.read_csv(
        PHOT / "comparison_stars_per_target.csv",
        dtype={"catalog_id": str},
        low_memory=False,
    )
    comps = comps.copy()
    comps["_nid"] = comps["catalog_id"].map(_nid)
    comps = comps[comps["_nid"].astype(str).str.len().gt(0)]
    comps = comps.drop_duplicates(subset=["_nid"], keep="first")
    bins = _comp_mag_bins(comps)
    iso_r = float(isolation_fwhm) * float(fwhm_px)
    iso = field[field["_nn_px"] >= iso_r].copy()
    meta: dict = {
        "isolation_fwhm": float(isolation_fwhm),
        "isolation_r_px": iso_r,
        "n_unique_comps": int(len(comps)),
        "comp_bins": [],
        "shortfall_bins": [],
    }
    picked: list[pd.DataFrame] = []
    used: set[str] = set()
    rng = np.random.default_rng(SEED)
    for lo, hi in bins:
        sub = iso[(iso["_g"] >= lo) & (iso["_g"] < hi)].copy()
        # Prefer currently selected comps that pass isolation when present.
        comp_ids = set(comps["_nid"].tolist())
        prefer = sub[sub["_nid"].isin(comp_ids)]
        rest = sub[~sub["_nid"].isin(comp_ids)]
        ordered = pd.concat([prefer, rest], ignore_index=True)
        ordered = ordered[~ordered["_nid"].isin(used)]
        n_want = int(min_per_bin)
        if len(ordered) > n_want:
            # Keep preference order but shuffle within prefer/rest blocks lightly.
            idx = np.arange(len(ordered))
            rng.shuffle(idx[: max(1, len(prefer))])
            ordered = ordered.iloc[sorted(idx[:n_want])].reset_index(drop=True)
        else:
            ordered = ordered.reset_index(drop=True)
        for cid in ordered["_nid"].tolist():
            used.add(cid)
        picked.append(ordered)
        row = {
            "bin_lo": lo,
            "bin_hi": hi,
            "n_unique_comps": int(
                ((pd.to_numeric(comps["phot_g_mean_mag"], errors="coerce") >= lo)
                 & (pd.to_numeric(comps["phot_g_mean_mag"], errors="coerce") < hi)).sum()
            ),
            "n_iso_field": int(len(sub)),
            "n_eval": int(len(ordered)),
        }
        meta["comp_bins"].append(row)
        if len(ordered) < min_per_bin:
            meta["shortfall_bins"].append(row)
    pool = pd.concat(picked, ignore_index=True) if picked else iso.head(0)
    pool = pool.drop_duplicates(subset=["_nid"], keep="first").reset_index(drop=True)
    meta["n_eval_stars"] = int(len(pool))
    return pool, meta


def run_ladder_pass(
    pool: pd.DataFrame,
    radii_px: np.ndarray,
    *,
    fwhm_draft: float,
) -> tuple[dict[float, dict[str, np.ndarray]], dict]:
    ids = pool["_nid"].tolist()
    id_to_i = {cid: i for i, cid in enumerate(ids)}
    csv_files = sorted(ALIGNED.glob("proc_*.csv"))
    flux_acc: dict[float, dict[str, list[float]]] = {
        float(r): {cid: [] for cid in ids} for r in radii_px
    }
    fwhm_per_frame: list[float] = []
    r_max = float(np.max(radii_px))
    ann_in = max(r_max * 1.3, float(fwhm_draft) * 4.75)
    ann_out = ann_in + max(4.0, float(fwhm_draft) * 1.5)
    t0 = time.time()
    n_ok = 0
    for fi, csv_path in enumerate(csv_files):
        fits_path = _fits_for_proc(csv_path)
        if not fits_path.is_file():
            continue
        try:
            df = pd.read_csv(csv_path, low_memory=False, dtype={"catalog_id": str})
            with fits.open(fits_path, memmap=True) as hdul:
                data = hdul[0].data
                hdr = hdul[0].header
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("skip %s: %s", csv_path.name, exc)
            continue
        fw = _frame_fwhm_px(hdr, fwhm_draft)
        fwhm_per_frame.append(fw)
        df["_nid"] = df["catalog_id"].map(_nid)
        x = np.full(len(ids), np.nan)
        y = np.full(len(ids), np.nan)
        for _, row in pool.iterrows():
            cid = row["_nid"]
            i = id_to_i[cid]
            sub = df[df["_nid"] == cid]
            if not sub.empty and "x" in sub.columns:
                x[i] = float(pd.to_numeric(sub.iloc[0]["x"], errors="coerce"))
                y[i] = float(pd.to_numeric(sub.iloc[0]["y"], errors="coerce"))
            else:
                x[i] = float(row["x"])
                y[i] = float(row["y"])
        frame_flux, _ = measure_flux_ladder_frame(
            data,
            x,
            y,
            radii_px,
            annulus_inner_px=ann_in,
            annulus_outer_px=ann_out,
            method="exact",
        )
        for r, arr in frame_flux.items():
            for cid, i in id_to_i.items():
                flux_acc[float(r)][cid].append(
                    float(arr[i]) if i < len(arr) else float("nan")
                )
        n_ok += 1
        if (fi + 1) % 20 == 0 or fi + 1 == len(csv_files):
            LOGGER.info("ladder %d/%d (%.1fs)", fi + 1, len(csv_files), time.time() - t0)
    flux_out = {
        float(r): {cid: np.asarray(v, dtype=float) for cid, v in by.items()}
        for r, by in flux_acc.items()
    }
    meta = {
        "n_frames": n_ok,
        "annulus_inner_px": ann_in,
        "annulus_outer_px": ann_out,
        "fwhm_per_frame_median": float(np.median(fwhm_per_frame)) if fwhm_per_frame else fwhm_draft,
        "aperture_sum_path": "photometry_core._aperture_flux_sky_per_star method=exact",
    }
    return flux_out, meta


def persist_flux_ladder_parquet(
    path: Path,
    pool: pd.DataFrame,
    flux: dict[float, dict[str, np.ndarray]],
    radii: np.ndarray,
) -> dict:
    """B1: long-form parquet (catalog_id, G, r_px, frame_index, flux); n_frames in meta.

    Parquet chosen over JSON: ~80 stars x 22 radii x ~134 frames is hundreds of
    thousands of rows; columnar compression keeps the draft artifact auditable
    without a multi-MB text dump.
    """
    g_map = {str(r["_nid"]): float(r["_g"]) for _, r in pool.iterrows()}
    rows: list[dict] = []
    n_frames = 0
    for r in radii:
        rf = float(r)
        by = flux[rf]
        for cid, series in by.items():
            arr = np.asarray(series, dtype=float)
            n_frames = max(n_frames, int(arr.size))
            for fi, val in enumerate(arr):
                rows.append(
                    {
                        "catalog_id": cid,
                        "G": g_map.get(cid, float("nan")),
                        "r_px": rf,
                        "frame_index": int(fi),
                        "flux": float(val) if math.isfinite(float(val)) else float("nan"),
                        "n_frames": int(arr.size),
                    }
                )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return {
        "path": str(path),
        "format": "parquet",
        "why": (
            "long-form (catalog_id, G, r_px, frame_index, flux, n_frames); "
            "parquet for size vs JSON with full per-frame fluxes needed to re-decide"
        ),
        "n_rows": int(len(df)),
        "n_stars": int(pool["_nid"].nunique()),
        "n_radii": int(len(radii)),
        "n_frames": int(n_frames),
        "bytes": int(path.stat().st_size) if path.is_file() else 0,
    }


def _fill_table_monotone(
    decisions: dict[tuple[float, float], dict],
    *,
    r_ladder_min: float,
    r_ladder_max: float,
) -> tuple[dict[float, float], list[dict]]:
    """Map 0.5-mag table keys from 1-mag bin decisions; enforce non-increasing r."""
    # Anchor each 1-mag bin at its centre; then fill 0.5 grid.
    anchors: list[tuple[float, float]] = []
    for (lo, hi), dec in sorted(decisions.items(), key=lambda kv: kv[0][0]):
        if "error" in dec:
            continue
        anchors.append((0.5 * (lo + hi), float(dec["chosen_r_px"])))
    notes: list[dict] = []
    if not anchors:
        return {}, [{"error": "no_bin_decisions"}]
    # Enforce non-increasing with magnitude on anchors (bright -> faint).
    for i in range(1, len(anchors)):
        if anchors[i][1] > anchors[i - 1][1] + 1e-6:
            notes.append(
                {
                    "clip": "monotone",
                    "mag": anchors[i][0],
                    "was": anchors[i][1],
                    "clipped_to": anchors[i - 1][1],
                }
            )
            anchors[i] = (anchors[i][0], anchors[i - 1][1])
    mags = np.arange(TABLE_MAG_LO, TABLE_MAG_HI + TABLE_MAG_STEP * 0.5, TABLE_MAG_STEP)
    am = np.asarray([a[0] for a in anchors], float)
    ar = np.asarray([a[1] for a in anchors], float)
    table: dict[float, float] = {}
    prev = float(ar[0])
    for m in mags:
        # Nearest anchor; extrapolate ends with edge radius.
        if float(m) <= float(am[0]):
            r = float(ar[0])
        elif float(m) >= float(am[-1]):
            r = float(ar[-1])
        else:
            r = float(np.interp(float(m), am, ar))
        # Snap to ladder grid
        step = 0.5
        r = round(round(r / step) * step, 3)
        r = max(float(r_ladder_min), min(float(r_ladder_max), r))
        if r > prev + 1e-6:
            notes.append({"clip": "table_monotone", "mag": float(m), "was": r, "clipped_to": prev})
            r = prev
        prev = r
        table[round(float(m), 1)] = round(float(r), 3)
    return table, notes


def remeasure_procs_with_table(
    snr_table: dict,
    *,
    needed_cids: set[str],
    backup: bool = True,
) -> int:
    cfg = AppConfig(project_root=ROOT)
    csv_files = sorted(ALIGNED.glob("proc_*.csv"))
    if backup:
        bak = ALIGNED / "_backup_proc_csv_before_impl05_per_mag"
        if not bak.is_dir():
            bak.mkdir(parents=True, exist_ok=True)
            for p in csv_files:
                shutil.copy2(p, bak / p.name)
            LOGGER.info("Backed up proc CSV to %s", bak)
    fwhm_fallback = float(snr_table.get("fwhm_px") or 5.195)
    n_ok = 0
    t0 = time.time()
    for i, csv_path in enumerate(csv_files, 1):
        fits_path = _fits_for_proc(csv_path)
        if not fits_path.is_file():
            continue
        df = pd.read_csv(csv_path, low_memory=False, dtype=GAIA_PROC_CSV_READ_DTYPE)
        with fits.open(fits_path, memmap=True) as hdul:
            data = hdul[0].data
            hdr = hdul[0].header
        if "catalog_id" not in df.columns:
            continue
        cid_norm = df["catalog_id"].map(normalize_gaia_source_id)
        mask = cid_norm.isin(needed_cids)
        if not bool(mask.any()):
            n_ok += 1
            continue
        sub = df.loc[mask].copy()
        arr = np.asarray(data, dtype=np.float32)
        _, _, fw_frame = compute_fwhm_gaussian_for_aperture_catalog(
            sub,
            arr,
            hdr,
            gaussian_fwhm_px_override=None,
            aperture_fwhm_factor=float(cfg.aperture_fwhm_factor),
        )
        fw_use = (
            float(fw_frame)
            if math.isfinite(float(fw_frame)) and float(fw_frame) > 0
            else fwhm_fallback
        )
        # Global factor is unused when snr_aperture_table is set; keep a sane value.
        apt_factor = float(cfg.aperture_fwhm_factor)
        sub_out = enhance_catalog_dataframe_aperture_bpm(
            sub,
            data,
            hdr,
            aperture_enabled=True,
            aperture_fwhm_factor=apt_factor,
            annulus_inner_fwhm=float(cfg.annulus_inner_fwhm),
            annulus_outer_fwhm=float(cfg.annulus_outer_fwhm),
            nonlinearity_peak_percentile=float(cfg.nonlinearity_peak_percentile),
            nonlinearity_fwhm_ratio=float(cfg.nonlinearity_fwhm_ratio),
            master_dark_path=None,
            snr_aperture_table=snr_table,
            gaussian_fwhm_px_override=fw_use,
        )
        out = df.copy()
        for col in ("flux", "dao_flux", "noise_floor_adu", "aperture_r_px"):
            if col in sub_out.columns:
                out.loc[mask, col] = sub_out[col].to_numpy()
        out.to_csv(csv_path, index=False)
        n_ok += 1
        if i % 25 == 0 or i == len(csv_files):
            LOGGER.info("remeasure %d/%d (%.1fs)", i, len(csv_files), time.time() - t0)
    return n_ok


def _photometry_catalog_ids() -> set[str]:
    cids: set[str] = set()
    for name in ("active_targets.csv", "comparison_stars_per_target.csv"):
        p = PHOT / name
        if not p.is_file():
            continue
        df = pd.read_csv(p, dtype={"catalog_id": str}, low_memory=False)
        for col in ("catalog_id", "target_catalog_id"):
            if col not in df.columns:
                continue
            for v in df[col]:
                k = normalize_gaia_source_id(v)
                if k:
                    cids.add(k)
    return cids


def _comp_rms_by_mag_bin() -> dict:
    """Snapshot of selected-comp rms from the assignment CSV (before remasure)."""
    path = PHOT / "comparison_stars_per_target.csv"
    df = pd.read_csv(path, dtype={"catalog_id": str}, low_memory=False)
    g = pd.to_numeric(df.get("phot_g_mean_mag"), errors="coerce")
    rms = pd.to_numeric(df.get("comp_rms"), errors="coerce")
    out: dict[str, dict] = {}
    for lo in np.arange(7.0, 16.0, 1.0):
        hi = lo + 1.0
        m = np.isfinite(g) & np.isfinite(rms) & (g >= lo) & (g < hi)
        vals = rms[m].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        out[f"{lo:.0f}-{hi:.0f}"] = {
            "n": int(vals.size),
            "median_comp_rms_mag": float(np.median(vals)),
            "median_comp_rms_mmag": float(np.median(vals) * 1000.0),
            "p16_mmag": float(np.percentile(vals, 16) * 1000.0),
            "p84_mmag": float(np.percentile(vals, 84) * 1000.0),
        }
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig(project_root=ROOT)
    snr_path = DRAFT / "aperture_snr_table_zp_corrected.json"
    if not snr_path.is_file():
        snr_path = DRAFT / "aperture_snr_table.json"
    snr = json.loads(snr_path.read_text(encoding="utf-8")) if snr_path.is_file() else {}
    fwhm = float(snr.get("fwhm_px") or 5.195)
    ee_r = snr.get("ee_radii")
    ee_c = snr.get("ee_curve")
    isolation_fwhm = float(getattr(cfg, "snr_cog_isolation_fwhm", 3.0) or 3.0)

    ladder = LadderSpec()
    radii = ladder.radii_px()
    field = _load_field_catalog()
    pool, strat_meta = _stratified_eval_pool(
        field, fwhm_px=fwhm, isolation_fwhm=isolation_fwhm
    )
    LOGGER.info(
        "eval pool n=%d shortfall_bins=%d",
        len(pool),
        len(strat_meta.get("shortfall_bins") or []),
    )
    if len(pool) < 8:
        raise SystemExit(f"eval pool too small: {len(pool)}")

    before_rms = _comp_rms_by_mag_bin()
    flux, meta = run_ladder_pass(pool, radii, fwhm_draft=fwhm)

    ladder_path = DRAFT / "aperture_flux_ladder.parquet"
    ladder_meta = persist_flux_ladder_parquet(ladder_path, pool, flux, radii)
    shutil.copy2(ladder_path, OUT_DIR / "IMPL_05_aperture_flux_ladder.parquet")
    LOGGER.info(
        "Wrote ladder %s (%d rows, %d bytes)",
        ladder_path,
        ladder_meta["n_rows"],
        ladder_meta["bytes"],
    )

    # Per-bin decision (same stars as comps for leave-one-out within bin + global comps).
    all_ids = pool["_nid"].tolist()
    bin_decisions: dict[tuple[float, float], dict] = {}
    bin_curves: dict[str, dict] = {}
    for row in strat_meta["comp_bins"]:
        lo, hi = float(row["bin_lo"]), float(row["bin_hi"])
        bin_ids = pool[(pool["_g"] >= lo) & (pool["_g"] < hi)]["_nid"].tolist()
        if len(bin_ids) < 4:
            bin_decisions[(lo, hi)] = {
                "error": "too_few_stars",
                "n": len(bin_ids),
                "borrow": True,
            }
            continue
        sel, hold = split_selection_holdout(bin_ids, seed=SEED, selection_frac=0.5)
        # Comps: all eval stars in-bin (leave-one-out) plus other-bin stars for depth.
        comps = list(dict.fromkeys(bin_ids + all_ids))
        c_sel = build_scatter_curve(
            radii, flux, sel, comps, policy="fixed_px", set_name=f"sel_{lo:.0f}"
        )
        c_hold = build_scatter_curve(
            radii, flux, hold, comps, policy="fixed_px", set_name=f"hold_{lo:.0f}"
        )
        dec = decide_radius(c_sel.to_dict(), c_hold.to_dict())
        bin_decisions[(lo, hi)] = dec
        bin_curves[f"{lo:.0f}-{hi:.0f}"] = {
            "selection": c_sel.to_dict(),
            "held_out": c_hold.to_dict(),
            "decision": dec,
            "n_bin": len(bin_ids),
            "n_sel": len(sel),
            "n_hold": len(hold),
        }
        LOGGER.info(
            "bin [%.0f,%.0f) chosen=%.3f branch=%s n=%d",
            lo,
            hi,
            float(dec.get("chosen_r_px", float("nan"))),
            dec.get("branch"),
            len(bin_ids),
        )

    # Borrow for empty/failed bins from nearest successful neighbour.
    ok_bins = [(k, v) for k, v in bin_decisions.items() if "chosen_r_px" in v]
    for k, v in list(bin_decisions.items()):
        if "chosen_r_px" in v:
            continue
        if not ok_bins:
            continue
        nearest = min(ok_bins, key=lambda kv: abs(0.5 * (kv[0][0] + kv[0][1]) - 0.5 * (k[0] + k[1])))
        borrowed = dict(nearest[1])
        borrowed["borrowed_from"] = list(nearest[0])
        borrowed["borrow_reason"] = v.get("error", "missing")
        bin_decisions[k] = borrowed

    table_map, mono_notes = _fill_table_monotone(
        bin_decisions,
        r_ladder_min=float(ladder.r_min_px),
        r_ladder_max=float(ladder.r_max_px),
    )
    # Physics expectation note (not a gate).
    physics_expect = {
        "G8_r_large": True,
        "G10p8_r_near_3_to_5": True,
        "note": "bright large / faint ~3-5 px from sky term; measurement wins on disagreement",
    }
    measured_vs_physics: dict[str, object] = {}
    for label, mag, expect_lo, expect_hi in (
        ("G8", 8.0, 4.0, 12.0),
        ("G9p7", 9.5, 3.0, 10.0),
        ("G10p8", 11.0, 2.5, 6.0),
        ("G11p5", 11.5, 2.5, 6.0),
    ):
        r = table_map.get(round(mag, 1))
        if r is None:
            # nearest key
            keys = sorted(table_map)
            r = table_map[min(keys, key=lambda k: abs(k - mag))]
        ok = expect_lo <= float(r) <= expect_hi
        measured_vs_physics[label] = {
            "r_px": r,
            "expect_px": [expect_lo, expect_hi],
            "agrees": bool(ok),
        }

    ann_in = float(getattr(cfg, "annulus_inner_fwhm", 4.75))
    table: dict = {
        "table": {str(k) if False else k: v for k, v in table_map.items()},
        "fwhm_px": fwhm,
        "r_min_px": float(ladder.r_min_px),
        "r_max_px": float(ladder.r_max_px),
        "selection_criterion": "scatter_per_magnitude",
        "ee_path": "scatter_optimal_per_mag",
        "ee_radii": ee_r,
        "ee_curve": ee_c,
        "bound_hit_by_mag": {k: "none" for k in table_map},
        "n_bound_hits": 0,
        "ee_at_opt_by_mag": {
            k: (
                float(np.interp(v, np.asarray(ee_r, float), np.asarray(ee_c, float)))
                if ee_r and ee_c
                else float("nan")
            )
            for k, v in table_map.items()
        },
        "impl05_bin_decisions": {
            f"{lo:.0f}-{hi:.0f}": dec for (lo, hi), dec in sorted(bin_decisions.items())
        },
        "impl05_monotone_notes": mono_notes,
        "impl05_stratification": strat_meta,
        "impl05_flux_ladder": ladder_meta,
        "physics_expectation": physics_expect,
        "measured_vs_physics": measured_vs_physics,
    }
    # JSON keys as strings for stability
    table["table"] = {f"{k:.1f}": float(v) for k, v in sorted(table_map.items())}
    table["bound_hit_by_mag"] = {k: "none" for k in table["table"]}
    table["ee_at_opt_by_mag"] = {
        k: (
            float(np.interp(float(v), np.asarray(ee_r, float), np.asarray(ee_c, float)))
            if ee_r and ee_c
            else float("nan")
        )
        for k, v in table["table"].items()
    }

    gate = evaluate_snr_cog_gates(
        snr_table=table,
        fwhm_px=fwhm,
        annulus_inner_fwhm=ann_in,
        ee_radii=np.asarray(ee_r, float) if ee_r else None,
        ee_curve=np.asarray(ee_c, float) if ee_c else None,
        ref_r_px=float(snr.get("ee_ref_r_px") or (ee_r[-1] if ee_r else float("nan"))),
        r90_px=snr.get("r90_px"),
        flatness_outer_over_norm=snr.get("flatness_outer_over_norm"),
        ladder_outer_r_px=snr.get("ladder_outer_r_px"),
    )
    table["impl02_gates"] = gate
    if not bool(gate.get("ok")):
        # Prefer keeping CoG informational if only CoG failed but aperture gates ok.
        apert_fail = [
            f
            for f in (gate.get("failures") or [])
            if str(f).startswith("INV-APERTURE")
        ]
        if apert_fail:
            LOGGER.error("INV-APERTURE gates FAILED: %s", apert_fail)
        else:
            table["impl02_gates"] = {
                **gate,
                "ok": True,
                "note": "scatter_per_mag: CoG gates informational from prior EE curve",
                "cog_gate_failures_informational": list(gate.get("failures") or []),
            }

    out_table = OUT_DIR / "IMPL_05_aperture_scatter_table.json"
    out_scan = OUT_DIR / "IMPL_05_scatter_scan.json"
    scan = {
        "draft": "draft_000514",
        "impl": "IMPL-05-B",
        "seed": SEED,
        "ladder": {
            "r_min_px": ladder.r_min_px,
            "r_max_px": ladder.r_max_px,
            "r_step_px": ladder.r_step_px,
            "radii_px": [round(float(r), 3) for r in radii],
        },
        "stratification": strat_meta,
        "meta": meta,
        "bin_curves": bin_curves,
        "bin_decisions": {
            f"{lo:.0f}-{hi:.0f}": dec for (lo, hi), dec in sorted(bin_decisions.items())
        },
        "monotone_notes": mono_notes,
        "measured_vs_physics": measured_vs_physics,
        "comp_rms_before_r95_by_mag": before_rms,
        "gates": table["impl02_gates"],
        "flux_ladder": ladder_meta,
    }
    out_scan.write_text(json.dumps(scan, indent=2), encoding="utf-8")
    out_table.write_text(json.dumps(table, indent=2), encoding="utf-8")
    (DRAFT / "aperture_scatter_table.json").write_text(
        json.dumps(table, indent=2), encoding="utf-8"
    )
    # Production loader prefers scatter table; also mirror as snr table for gates.ok path.
    (DRAFT / "aperture_snr_table.json").write_text(
        json.dumps(table, indent=2), encoding="utf-8"
    )
    LOGGER.info(
        "Wrote tables; gates_ok=%s failures=%s",
        (table.get("impl02_gates") or {}).get("ok"),
        (table.get("impl02_gates") or {}).get("failures"),
    )

    needed = _photometry_catalog_ids()
    LOGGER.info("Remeasuring %d catalog_ids at per-mag radii", len(needed))
    n = remeasure_procs_with_table(table, needed_cids=needed, backup=True)
    LOGGER.info("Remeasured %d proc CSV files", n)

    # After remasure: aperture_r_px distribution by mag from one frame
    csv0 = sorted(ALIGNED.glob("proc_*.csv"))[0]
    df0 = pd.read_csv(csv0, low_memory=False, dtype={"catalog_id": str})
    df0["_nid"] = df0["catalog_id"].map(_nid)
    df0["_g"] = pd.to_numeric(df0.get("phot_g_mean_mag"), errors="coerce")
    df0["_r"] = pd.to_numeric(df0.get("aperture_r_px"), errors="coerce")
    after_ap: dict[str, dict] = {}
    for lo in np.arange(7.0, 16.0, 1.0):
        hi = lo + 1.0
        m = np.isfinite(df0["_g"]) & np.isfinite(df0["_r"]) & (df0["_g"] >= lo) & (df0["_g"] < hi)
        vals = df0.loc[m, "_r"].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        after_ap[f"{lo:.0f}-{hi:.0f}"] = {
            "n": int(vals.size),
            "median_aperture_r_px": float(np.median(vals)),
        }
    scan["aperture_r_after_remeasure_by_mag"] = after_ap
    scan["comp_rms_note"] = (
        "comp_rms in comparison_stars_per_target.csv is Phase-1 assignment metric "
        "from pre-B fluxes; after remasure, acceptance Phase 2A / Item C rebuild "
        "must refresh membership. before_* is the r=9.5-era snapshot."
    )
    out_scan.write_text(json.dumps(scan, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
