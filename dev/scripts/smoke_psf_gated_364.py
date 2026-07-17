#!/usr/bin/env python3
"""Cheap smoke test for gated PSF components — draft 364 Luminance_180_2 (read-only data).

Temporarily toggles config flags, runs minimal checks, restores all flags.
Does NOT fix bugs — report only.
"""
from __future__ import annotations

import json
import math
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from photutils.psf.groupers import SourceGrouper

_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = _ROOT / "config.json"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from photometry_core import compute_lc_flux_method, _load_adaptive_blend_map  # noqa: E402
from psf_photometry import (  # noqa: E402
    _grouped_psf_fit,
    assess_psf_quality,
    build_epsf_grid_model,
    get_epsf_fwhm_from_context,
)
from photutils.psf import ImagePSF  # noqa: E402

DRAFT_ID = 364
SETUP = "Luminance_180_2"
MAX_FRAMES_ADAPTIVE = 3
MAX_GROUP_FITS = 5
FIELD_DB = _ROOT / "GAIA_DR3" / "vyvar_gaia_dr3_pal7_field.db"


def _load_config_raw() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _save_config_raw(data: dict[str, Any]) -> None:
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _set_flags(**kwargs: Any) -> dict[str, Any]:
    """Patch config.json keys; return snapshot of touched keys."""
    data = _load_config_raw()
    snap = {k: data.get(k) for k in kwargs}
    for k, v in kwargs.items():
        data[k] = v
    _save_config_raw(data)
    return snap


def _restore_flags(snap: dict[str, Any]) -> None:
    data = _load_config_raw()
    for k, v in snap.items():
        data[k] = v
    _save_config_raw(data)


def _draft_paths() -> dict[str, Path]:
    cfg = AppConfig()
    db = VyvarDatabase(cfg.database_path)
    row = db.fetch_obs_draft_by_id(DRAFT_ID)
    draft_dir = Path(row["ARCHIVE_PATH"]) if row and row.get("ARCHIVE_PATH") else (
        Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    )
    ps = draft_dir / "platesolve" / SETUP
    aligned = draft_dir / "detrended_aligned" / "lights" / SETUP
    return {"draft_dir": draft_dir, "ps": ps, "aligned": aligned}


def _smoke_grouper(paths: dict[str, Path], fwhm_px: float, cfg: AppConfig) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False}
    sep_fwhm = float(cfg.psf_group_sep_fwhm)
    inc_fwhm = float(cfg.psf_neighbor_include_fwhm)
    sep_px = sep_fwhm * fwhm_px
    inc_px = inc_fwhm * fwhm_px

    csvs = sorted(paths["aligned"].glob("proc_*.csv"))
    fits_map = {p.stem: p for p in paths["aligned"].glob("proc_*.fits")}
    if not csvs:
        out["error"] = "no proc CSV"
        return out

    csv_path = csvs[0]
    fits_path = fits_map.get(csv_path.stem)
    if fits_path is None:
        out["error"] = "no paired FITS"
        return out

    df = pd.read_csv(csv_path, low_memory=False, dtype={"catalog_id": str})
    df = df[df.get("photometry_ok", True).astype(str).str.lower().isin(["true", "1", "yes"])]
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df[np.isfinite(df["x"]) & np.isfinite(df["y"])]
    if df.empty:
        out["error"] = "no stars in frame"
        return out

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    grouper = SourceGrouper(min_separation=float(sep_px))
    gids = np.asarray(grouper(x, y), dtype=int)
    sizes = Counter(gids)
    n_groups = len(sizes)
    size_dist = dict(sorted(Counter(sizes.values()).items()))
    max_size = max(sizes.values()) if sizes else 0
    multi_mask = np.array([sizes[g] > 1 for g in gids], dtype=bool)
    n_multi = int(multi_mask.sum())

    # Neighbour-inclusion count (would enter joint fit pool)
    n_joint_candidates = 0
    for i in range(len(x)):
        d = np.hypot(x - x[i], y - y[i])
        n_near = int(np.sum((d <= inc_px) & (d > 0.5 * fwhm_px)))
        if n_near > 0:
            n_joint_candidates += 1

    out.update(
        {
            "frame": csv_path.name,
            "n_sources": int(len(df)),
            "n_groups": n_groups,
            "group_size_distribution": size_dist,
            "max_group_size": max_size,
            "n_in_multi_source_groups": n_multi,
            "n_with_neighbors_within_include_radius": n_joint_candidates,
            "sep_fwhm": sep_fwhm,
            "include_fwhm": inc_fwhm,
            "sep_px": sep_px,
            "include_px": inc_px,
        }
    )

    # Optional: fit a few joint groups
    epsf_path = paths["ps"] / "masterstar_epsf.fits"
    meta_path = paths["ps"] / "masterstar_epsf_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    osamp = int(meta.get("oversampling", 2))
    fit_shape = tuple(meta.get("fit_shape", [15, 15]))
    psf_data = np.asarray(fits.getdata(epsf_path), dtype=np.float64)
    psf_model = ImagePSF(psf_data, oversampling=osamp)

    with fits.open(fits_path, memmap=True) as hd:
        data = np.asarray(hd[0].data, dtype=np.float64)

    nb_xy = df[["x", "y"]].to_numpy(dtype=float)
    flux_col = pd.to_numeric(df.get("dao_flux", pd.Series(np.nan, index=df.index)), errors="coerce")
    nb_flux = flux_col.to_numpy(dtype=float)

    fit_results: list[dict[str, Any]] = []
    tried = 0
    for i in np.where(multi_mask)[0][:MAX_GROUP_FITS]:
        tried += 1
        try:
            res = _grouped_psf_fit(
                data,
                None,
                float(x[i]),
                float(y[i]),
                fwhm_px=fwhm_px,
                fit_shape=fit_shape,
                psf_model=psf_model,
                neighbor_xy=nb_xy,
                neighbor_flux=nb_flux,
                group_sep_fwhm=sep_fwhm,
                neighbor_include_fwhm=inc_fwhm,
                chi2_limit=float(cfg.psf_chi2_threshold),
            )
            fit_results.append(
                {
                    "idx": int(i),
                    "ok": res is not None,
                    "n_group": int(res.get("n_group", 0)) if res else 0,
                    "psf_fit_ok": bool(res.get("psf_fit_ok")) if res else False,
                }
            )
        except Exception as exc:  # noqa: BLE001
            fit_results.append({"idx": int(i), "ok": False, "error": str(exc)})

    out["group_fits_attempted"] = tried
    out["group_fits"] = fit_results
    out["ok"] = True
    return out


def _smoke_gridded(paths: dict[str, Path], cfg: AppConfig) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False}
    db = VyvarDatabase(cfg.database_path)
    ms_fits = paths["ps"] / "MASTERSTAR.fits"
    ms_csv = paths["ps"] / "masterstars.csv"
    try:
        grid = build_epsf_grid_model(
            ms_fits,
            ms_csv,
            db,
            DRAFT_ID,
            grid=str(cfg.psf_spatial_grid),
            oversampling=2,
            min_stars_per_cell=int(cfg.psf_spatial_min_stars_per_cell),
        )
        out["grid_nx"] = grid["grid_nx"]
        out["grid_ny"] = grid["grid_ny"]
        out["n_isolated_candidates"] = grid["n_isolated"]
        out["cell_n_stars"] = grid["cell_n_stars"]
        out["cell_fallback"] = grid["cell_fallback"]
        out["n_fallback"] = grid["n_fallback"]
        out["gridded_model_built"] = grid["gridded_model"] is not None
        out["global_fwhm_ratio"] = grid.get("global_qc", {}).get("epsf_vs_input_fwhm_ratio")
        starved = [
            (i, n)
            for i, (n, fb) in enumerate(zip(grid["cell_n_stars"], grid["cell_fallback"], strict=True))
            if fb or n < int(cfg.psf_spatial_min_stars_per_cell)
        ]
        out["starved_or_fallback_cells"] = starved
        out["all_cells_built_native"] = int(grid["n_fallback"]) == 0
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
        out["traceback"] = traceback.format_exc()
    return out


def _derive_psf_quality(df: pd.DataFrame, fwhm_px: float) -> pd.Series:
    qs = []
    for _, r in df.iterrows():
        chi2 = float(pd.to_numeric(r.get("psf_chi2"), errors="coerce"))
        pflux = float(pd.to_numeric(r.get("psf_flux"), errors="coerce"))
        perr = float(pd.to_numeric(r.get("psf_flux_err"), errors="coerce"))
        snr = pflux / perr if math.isfinite(pflux) and math.isfinite(perr) and perr > 0 else float("nan")
        qs.append(assess_psf_quality(chi2, snr, None, fwhm_px, None))
    return pd.Series(qs, index=df.index)


def _quick_crowded_flags(df: pd.DataFrame, fwhm_px: float, plate_scale: float = 0.389) -> pd.Series:
    """Proxy crowded: nearest neighbour within 2×FWHM in px."""
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    iso_r = 2.0 * fwhm_px
    crowded = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        d = np.hypot(x - x[i], y - y[i])
        crowded[i] = bool(np.any((d > 0.5) & (d <= iso_r)))
    return pd.Series(crowded, index=df.index)


def _smoke_adaptive(paths: dict[str, Path], cfg: AppConfig, fwhm_px: float) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False}
    csvs = sorted(paths["aligned"].glob("proc_*.csv"))[:MAX_FRAMES_ADAPTIVE]
    if not csvs:
        out["error"] = "no proc CSV"
        return out

    chunks: list[pd.DataFrame] = []
    for p in csvs:
        df = pd.read_csv(p, low_memory=False, dtype={"catalog_id": str})
        df = df[df.get("photometry_ok", True).astype(str).str.lower().isin(["true", "1", "yes"])]
        df["frame"] = p.name
        chunks.append(df)
    all_df = pd.concat(chunks, ignore_index=True)

    # Derive Phase-2A-like columns missing from proc CSV
    dao = pd.to_numeric(all_df["dao_flux"], errors="coerce")
    all_df["mag_inst"] = np.where(dao > 0, -2.5 * np.log10(dao), np.nan)
    perr = pd.to_numeric(all_df.get("psf_flux_err"), errors="coerce")
    all_df["err"] = np.where(perr > 0, 1.0857362 / (dao / perr), np.nan)
    all_df["psf_quality"] = _derive_psf_quality(all_df, fwhm_px)

    blend_map = _load_adaptive_blend_map(paths["ps"] / "MASTERSTAR.fits")
    out["blend_map_size"] = len(blend_map)
    out["crowding_targets_present"] = (paths["ps"] / "crowding_targets.csv").is_file()

    methods = compute_lc_flux_method(
        all_df,
        blend_map,
        resolve_fwhm=float(cfg.psf_adaptive_resolve_fwhm),
        snr_lo=float(cfg.psf_adaptive_snr_lo),
    )
    all_df["lc_flux_method"] = methods

    # Rule attribution
    psf_ok = all_df["psf_fit_ok"].astype(str).str.lower().isin(["true", "1", "yes"])
    pf = pd.to_numeric(all_df["psf_flux"], errors="coerce")
    psf_usable = psf_ok & np.isfinite(pf) & (pf > 0) & (all_df["psf_quality"].astype(str) != "bad")
    err = pd.to_numeric(all_df["err"], errors="coerce")
    snr_aper = np.where(np.isfinite(err) & (err > 0), 1.0857362 / err, np.inf)
    cid = all_df["catalog_id"].astype(str).str.strip()
    is_blended = cid.map(lambda c: bool(blend_map.get(c, (False, float("nan")))[0]))
    nn = cid.map(lambda c: float(blend_map.get(c, (False, float("nan")))[1]))
    rule2 = psf_usable & is_blended & np.isfinite(nn.to_numpy()) & (nn.to_numpy() >= float(cfg.psf_adaptive_resolve_fwhm))
    rule3 = psf_usable & (snr_aper <= float(cfg.psf_adaptive_snr_lo)) & (all_df["psf_quality"].astype(str) == "good")
    all_df["adaptive_rule"] = np.where(rule2, "rule2_blend", np.where(rule3, "rule3_faint", "default_aperture"))

    n_psf = int((methods == "psf").sum())
    n_aper = int((methods == "aperture").sum())
    out["n_rows"] = len(all_df)
    out["n_frames"] = len(csvs)
    out["n_psf"] = n_psf
    out["n_aperture"] = n_aper
    out["rule_counts"] = dict(Counter(all_df.loc[methods == "psf", "adaptive_rule"]))

    mag = pd.to_numeric(all_df.get("phot_g_mean_mag", all_df.get("mag")), errors="coerce")
    all_df["_mag_bin"] = pd.cut(mag, bins=[0, 13, 15, 17, 20, 99], labels=["G<13", "G13-15", "G15-17", "G17-20", "G20+"])
    all_df["_crowded"] = _quick_crowded_flags(all_df, fwhm_px)

    by_mag = (
        all_df.groupby("_mag_bin", observed=True)["lc_flux_method"]
        .value_counts()
        .unstack(fill_value=0)
        .to_dict()
    )
    by_crowd = (
        all_df.groupby("_crowded")["lc_flux_method"]
        .value_counts()
        .unstack(fill_value=0)
        .to_dict()
    )
    out["by_mag"] = by_mag
    out["by_crowded"] = by_crowd
    out["ok"] = True
    return out


def main() -> None:
    paths = _draft_paths()
    db = VyvarDatabase(AppConfig().database_path)
    fwhm_px = float(get_epsf_fwhm_from_context(paths["ps"] / "MASTERSTAR.fits", db, DRAFT_ID))

    orig = _load_config_raw()
    flag_keys = [
        "psf_photometry_enabled",
        "psf_grouper_enabled",
        "psf_spatial_enabled",
        "psf_adaptive_enabled",
    ]
    snap = {k: orig.get(k) for k in flag_keys}

    results: dict[str, Any] = {"draft": DRAFT_ID, "setup": SETUP, "fwhm_px": fwhm_px}

    # B1 — Grouper ON
    print("=== B1 GROUPER SMOKE ===", flush=True)
    _set_flags(psf_grouper_enabled=True)
    cfg = AppConfig()
    try:
        results["grouper"] = _smoke_grouper(paths, fwhm_px, cfg)
    except Exception as exc:  # noqa: BLE001
        results["grouper"] = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}

    # B2 — Gridded ePSF ON
    print("=== B2 GRIDDED ePSF SMOKE ===", flush=True)
    _set_flags(psf_spatial_enabled=True)
    cfg = AppConfig()
    try:
        results["gridded"] = _smoke_gridded(paths, cfg)
    except Exception as exc:  # noqa: BLE001
        results["gridded"] = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}

    # B3 — Adaptive ON (needs psf columns; uses proc CSVs directly)
    print("=== B3 ADAPTIVE SMOKE ===", flush=True)
    _set_flags(psf_adaptive_enabled=True, psf_photometry_enabled=True)
    cfg = AppConfig()
    try:
        results["adaptive"] = _smoke_adaptive(paths, cfg, fwhm_px)
    except Exception as exc:  # noqa: BLE001
        results["adaptive"] = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}

    _restore_flags(snap)
    chk = _load_config_raw()
    results["flags_restored"] = {k: chk.get(k) for k in flag_keys}
    results["restore_ok"] = all(chk.get(k) == snap[k] for k in flag_keys)

    out_json = paths["draft_dir"] / "diagnostics" / "psf_gated_smoke_364.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
