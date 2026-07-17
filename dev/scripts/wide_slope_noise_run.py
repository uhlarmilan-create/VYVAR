#!/usr/bin/env python3
"""WIDE-SLOPE-NOISE: report-only decomposition of wide rig b_X scatter (draft_424)."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from wide_slope_noise_core import (  # noqa: E402
    brightness_tertile_slices,
    excess_variance_by_tertile,
    fwhm_sensitivity,
    p4_noise_consistency_check,
    physical_effect_size_table,
    pre_registered_outcome,
    rms_predicted_mmag,
    slope_se_audit_steps,
    slope_se_pair,
    star_drift_metrics,
    univariate_hypothesis_scan,
    variance_decomposition_regression,
)

# Reuse K2 cohort data plumbing (report-only; no production import).
from scripts.k2_cohort_run import (  # noqa: E402
    AIRMASS_RANGE_MIN,
    MIN_EPOCHS,
    _batch_loo_delta_mag,
    _bp_rp_map,
    _build_k2_star_record,
    _host_context,
    _norm_id,
    _photon_err_mag_cached,
    _pick_host_target,
    _proc_row_cached,
    _sparse_target_ids,
)

DRAFT_ID = 424
SETUP = "NoFilter_60_2"
CELL_KEY = "wide_CLEAR"
COHORT_JSON = _ROOT / "tmp" / "k2_cohort" / "cohort_table.json"
PZQ_SIGMA_R_MM = 5.5
PZQ_SIGMA_R_CI = (4.7, 6.5)
RIG_CONSTANT_MM = 4.5


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _stamp(payload: dict[str, Any]) -> dict[str, Any]:
    payload["git_head"] = _git_head()
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return payload


def _load_wide_cell() -> dict[str, Any]:
    if not COHORT_JSON.is_file():
        raise FileNotFoundError(f"Missing cohort table: {COHORT_JSON}")
    data = json.loads(COHORT_JSON.read_text(encoding="utf-8"))
    for cell in data.get("cells", []):
        if cell.get("cell_key") == CELL_KEY:
            return cell
    raise KeyError(f"Cell {CELL_KEY} not found in {COHORT_JSON}")


def _epoch_geometry(
    loo_stub: pd.DataFrame,
    proc_dir: Path,
    catalog_id: str,
    csv_cache: dict[str, pd.DataFrame],
) -> dict[str, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    fwhms: list[float] = []
    apertures: list[float] = []
    for sf in loo_stub["source_file"].astype(str).tolist():
        row = _proc_row_cached(proc_dir, sf, catalog_id, csv_cache)
        if row is None:
            xs.append(float("nan"))
            ys.append(float("nan"))
            fwhms.append(float("nan"))
            apertures.append(float("nan"))
            continue
        xs.append(float(pd.to_numeric(row.get("x"), errors="coerce")))
        ys.append(float(pd.to_numeric(row.get("y"), errors="coerce")))
        fwhms.append(float(pd.to_numeric(row.get("fwhm_estimate_px"), errors="coerce")))
        apertures.append(float(pd.to_numeric(row.get("aperture_r_px"), errors="coerce")))
    return {
        "x": np.asarray(xs, dtype=np.float64),
        "y": np.asarray(ys, dtype=np.float64),
        "fwhm": np.asarray(fwhms, dtype=np.float64),
        "aperture": np.asarray(apertures, dtype=np.float64),
    }


RIG_CONSTANT_MM = 4.5
DETECTOR_PATH_NOTE = (
    "Path (c): DAO cutout centroid on calibrated pre-alignment lights "
    "(Archive/.../calibrated/lights/NoFilter_60_2/). "
    "Paths (a)(b) unavailable: alignment_report.csv has no per-frame shift columns."
)


def _proc_to_cal_name(source_file: str) -> str:
    base = Path(source_file).name
    if base.startswith("proc_"):
        base = base[5:]
    if base.endswith(".csv"):
        base = base[:-4] + ".fits"
    elif not base.endswith(".fits"):
        base = f"{Path(base).stem}.fits"
    return base


def _alignment_transform_audit(cfg: AppConfig) -> dict[str, Any]:
    """Report whether stored alignment transforms exist for draft_424."""
    draft = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    report = draft / "alignment_report.csv"
    out: dict[str, Any] = {
        "alignment_report_exists": report.is_file(),
        "has_shift_columns": False,
        "path_used": "c",
        "path_note": DETECTOR_PATH_NOTE,
    }
    if report.is_file():
        df = pd.read_csv(report, nrows=5)
        shift_cols = [c for c in df.columns if "shift" in c.lower() or c.lower() in ("dx", "dy", "tx", "ty")]
        out["has_shift_columns"] = bool(shift_cols)
        out["alignment_report_columns"] = list(df.columns)
    return out


def _detector_drift_for_epochs(
    source_files: list[str],
    cal_dir: Path,
    seed_x: float,
    seed_y: float,
    airmass: np.ndarray,
) -> dict[str, Any]:
    """Track detector-frame centroids on calibrated pre-alignment lights."""
    from astropy.io import fits

    from wide_slope_noise_core import centroid_cutout_detector  # noqa: PLC0415

    cx, cy = float(seed_x), float(seed_y)
    xs: list[float] = []
    ys: list[float] = []
    for i, sf in enumerate(source_files):
        cal_path = cal_dir / _proc_to_cal_name(sf)
        if not cal_path.is_file():
            xs.append(float("nan"))
            ys.append(float("nan"))
            continue
        with fits.open(cal_path, memmap=True) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
        half = 32 if i == 0 else 16
        cx, cy = centroid_cutout_detector(data, cx, cy, half=half)
        xs.append(cx)
        ys.append(cy)
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    order = np.argsort(airmass)
    metrics = star_drift_metrics(x_arr[order], y_arr[order], airmass[order])
    return {
        "det_x_med": metrics["x_med"],
        "det_y_med": metrics["y_med"],
        "det_drift_span_px": metrics["drift_span_px"],
        "det_drift_x_corr": metrics["drift_x_corr"],
        "det_corr_x_am": metrics["corr_x_am"],
        "det_corr_y_am": metrics["corr_y_am"],
    }


def _pick_worked_stars_per_tertile(stars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One representative star per brightness tertile for SE audit."""
    mags = np.asarray([float(s["mag_g"]) for s in stars if s.get("mag_g") is not None], dtype=np.float64)
    if len(mags) < 6:
        return []
    picked: list[dict[str, Any]] = []
    for label, lo, hi in brightness_tertile_slices(mags):
        sub = [
            s for s in stars
            if s.get("mag_g") is not None and lo <= float(s["mag_g"]) < hi and s.get("_audit_lc") is not None
        ]
        if not sub:
            continue
        sub.sort(key=lambda s: abs(float(s.get("b_X", 0.0))))
        picked.append({"tertile": label, "star": sub[len(sub) // 2]})
    return picked


def _refresh_detector_drift(stars: list[dict[str, Any]], cfg: AppConfig) -> list[dict[str, Any]]:
    """Recompute detector-frame drift on cached stars (skip full LOO rebuild)."""
    cal_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}" / "calibrated" / "lights" / SETUP
    phot_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}" / "platesolve" / SETUP / "photometry"
    from check_star_kmag import resolve_proc_csv_dir  # noqa: PLC0415

    proc_dir = resolve_proc_csv_dir(phot_dir, SETUP)
    if proc_dir is None:
        raise FileNotFoundError("proc dir missing for draft_424")
    comp_all = pd.read_csv(
        phot_dir / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    sparse_targets = _sparse_target_ids(phot_dir)
    csv_cache: dict[str, pd.DataFrame] = {}
    host_cache: dict[str, dict[str, Any] | None] = {}
    out: list[dict[str, Any]] = []
    for row in stars:
        cid = str(row.get("catalog_id", ""))
        host = _pick_host_target(comp_all, cid, sparse_targets=sparse_targets)
        if host is None:
            out.append(row)
            continue
        if host not in host_cache:
            host_cache[host] = _host_context(
                host, phot_dir=phot_dir, comp_all=comp_all, proc_dir=proc_dir, cfg=cfg,
                csv_cache=csv_cache,
            )
        ctx = host_cache[host]
        if ctx is None:
            out.append(row)
            continue
        lc_df = ctx["lc_df"]
        airmass = ctx["lc_airmass"]
        source_files = lc_df["source_file"].astype(str).tolist()
        m_len = min(len(source_files), len(airmass))
        first_sf = source_files[0]
        proc_row = _proc_row_cached(proc_dir, first_sf, cid, csv_cache)
        if proc_row is None:
            out.append(row)
            continue
        seed_x = float(pd.to_numeric(proc_row.get("x"), errors="coerce"))
        seed_y = float(pd.to_numeric(proc_row.get("y"), errors="coerce"))
        am_ok = np.asarray(airmass[:m_len], dtype=np.float64)
        sf_ok = source_files[:m_len]
        updated = dict(row)
        if math.isfinite(seed_x) and math.isfinite(seed_y):
            updated.update(_detector_drift_for_epochs(sf_ok, cal_dir, seed_x, seed_y, am_ok))
        out.append(updated)
    return out


def enrich_stars(cfg: AppConfig, cell: dict[str, Any]) -> list[dict[str, Any]]:
    phot_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}" / "platesolve" / SETUP / "photometry"
    cal_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}" / "calibrated" / "lights" / SETUP
    if not cal_dir.is_dir():
        raise FileNotFoundError(f"Missing calibrated lights: {cal_dir}")
    from check_star_kmag import resolve_proc_csv_dir  # noqa: PLC0415

    proc_dir = resolve_proc_csv_dir(phot_dir, SETUP)
    if proc_dir is None:
        raise FileNotFoundError("proc dir missing for draft_424")

    meta = json.loads((phot_dir / "pipeline_meta.json").read_text(encoding="utf-8"))
    gain = float(meta.get("gain") or 1.0)
    rn = float(meta.get("read_noise") or 10.0)
    comp_all = pd.read_csv(
        phot_dir / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    bp_rp = _bp_rp_map(phot_dir)
    br_vals = [v for v in bp_rp.values() if math.isfinite(v)]
    bp_rp_med = float(np.median(br_vals)) if br_vals else float("nan")
    sparse_targets = _sparse_target_ids(phot_dir)

    cohort_by_id = {str(s["catalog_id"]): s for s in cell.get("stars", [])}
    csv_cache: dict[str, pd.DataFrame] = {}
    host_cache: dict[str, dict[str, Any] | None] = {}
    enriched: list[dict[str, Any]] = []

    for cid, base in cohort_by_id.items():
        if base.get("t1_lever_excluded"):
            continue
        host = _pick_host_target(comp_all, cid, sparse_targets=sparse_targets)
        if host is None:
            continue
        if host not in host_cache:
            host_cache[host] = _host_context(
                host, phot_dir=phot_dir, comp_all=comp_all, proc_dir=proc_dir, cfg=cfg,
                csv_cache=csv_cache,
            )
        ctx = host_cache[host]
        if ctx is None:
            continue
        if _norm_id(cid) not in ctx["comp_ids"]:
            ctx["comp_ids"] = list(ctx["comp_ids"]) + [_norm_id(cid)]
            from check_star_kmag import build_aligned_comp_inst  # noqa: PLC0415

            ctx["comp_lc"] = build_aligned_comp_inst(
                proc_dir, ctx["comp_ids"], ctx["lc_df"]["source_file"].astype(str).tolist(),
                cfg, "aperture", csv_cache=csv_cache,
            )
            host_cache[host] = ctx

        built = _build_k2_star_record(
            cid,
            proc_dir=proc_dir,
            lc_df=ctx["lc_df"],
            lc_airmass_full=ctx["lc_airmass"],
            comp_lc=ctx["comp_lc"],
            comp_ids=ctx["comp_ids"],
            cat=ctx["cat"],
            tier=ctx["tier"],
            rms=ctx["rms"],
            tw=ctx["tw"],
            cfg=cfg,
            gain=gain,
            read_noise=rn,
            csv_cache=csv_cache,
        )
        if built is None:
            continue

        n = int(built["n_epochs"])
        loo_stub = ctx["lc_df"].iloc[: len(ctx["lc_df"])].copy()
        mags_full = _batch_loo_delta_mag(
            cid, lc_df=ctx["lc_df"], comp_lc=ctx["comp_lc"], comp_ids=ctx["comp_ids"],
            cat=ctx["cat"], tier=ctx["tier"], rms=ctx["rms"], tw=ctx["tw"], cfg=cfg,
        )
        if mags_full is None:
            continue
        m_len = min(len(mags_full), len(ctx["lc_airmass"]), len(loo_stub))
        loo_stub = loo_stub.iloc[:m_len].copy()
        photon_full = _photon_err_mag_cached(
            loo_stub, proc_dir, cid, gain=gain, read_noise=rn, csv_cache=csv_cache,
        )
        airmass_full = ctx["lc_airmass"][:m_len]
        mags_full = np.asarray(mags_full[:m_len], dtype=np.float64)
        geom = _epoch_geometry(loo_stub, proc_dir, cid, csv_cache)
        ok = (
            np.isfinite(mags_full)
            & np.isfinite(airmass_full)
            & np.isfinite(photon_full)
            & (photon_full > 0)
        )
        m_ok = mags_full[ok]
        am_ok = airmass_full[ok]
        err_ok = photon_full[ok]
        x_ok = geom["x"][:m_len][ok]
        y_ok = geom["y"][:m_len][ok]
        fwhm_ok = geom["fwhm"][:m_len][ok]
        ap_ok = geom["aperture"][:m_len][ok]

        se_info = slope_se_pair(
            m_ok, am_ok, err_ok,
            bootstrap_draws=300,
            seed=hash(cid) % 100000,
            min_airmass_range=AIRMASS_RANGE_MIN,
        )
        se_use = se_info["se_use"]

        order = np.argsort(am_ok)
        drift = star_drift_metrics(x_ok[order], y_ok[order], am_ok[order])
        source_files = loo_stub["source_file"].astype(str).tolist()
        ok_idx = np.where(ok)[0]
        sf_ok = [source_files[i] for i in ok_idx]
        seed_x = float(x_ok[0]) if len(x_ok) else float("nan")
        seed_y = float(y_ok[0]) if len(y_ok) else float("nan")
        det_drift = (
            _detector_drift_for_epochs(sf_ok, cal_dir, seed_x, seed_y, am_ok)
            if math.isfinite(seed_x) and math.isfinite(seed_y)
            else {}
        )
        fwhm_info = fwhm_sensitivity(m_ok, fwhm_ok, am_ok)
        ap_fwhm = float(np.nanmedian(ap_ok / fwhm_ok)) if np.any(np.isfinite(ap_ok) & np.isfinite(fwhm_ok) & (fwhm_ok > 0)) else float("nan")
        fwhm_info["aperture_over_fwhm"] = ap_fwhm
        fwhm_range = float(np.nanmax(fwhm_ok) - np.nanmin(fwhm_ok)) if np.any(np.isfinite(fwhm_ok)) else float("nan")

        br = bp_rp.get(_norm_id(cid), float("nan"))
        colour_signed = float(br - bp_rp_med) if math.isfinite(br) and math.isfinite(bp_rp_med) else float("nan")

        row = dict(base)
        row.update({
            "catalog_id": cid,
            "b_X": se_info["b_X"],
            "b_X_se_cohort": base.get("b_X_se"),
            "se_propagated": se_info["se_propagated"],
            "se_bootstrap": se_info["se_bootstrap"],
            "se_use": se_use,
            "N_epochs": se_info["n_epochs"],
            "mag_g": base.get("mag_g"),
            "colour_offset_signed": colour_signed if math.isfinite(colour_signed) else base.get("colour_offset_signed"),
            "sigma_r": base.get("sigma_r"),
            "_fwhm_epochs": fwhm_ok.tolist(),
            "fwhm_range": fwhm_range,
            "_audit_lc": {
                "mags": m_ok.tolist(),
                "airmass": am_ok.tolist(),
                "err": err_ok.tolist(),
            },
        })
        row.update(drift)
        row.update(det_drift)
        row.update(fwhm_info)
        enriched.append(row)

    return enriched


def _plot_field_map(stars: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    bx = [float(s["b_X"]) for s in stars if s.get("b_X") is not None]
    x = [float(s["x_med"]) for s in stars if s.get("x_med") is not None]
    y = [float(s["y_med"]) for s in stars if s.get("y_med") is not None]
    if len(bx) < 3:
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return
    sc = ax.scatter(x, y, c=bx, cmap="coolwarm", s=18, alpha=0.85)
    fig.colorbar(sc, ax=ax, label="b_X (mag/airmass)")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title(f"b_X field map - {CELL_KEY}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_excess_tertiles(tertiles: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [t["tertile"] for t in tertiles]
    excess = [float(t["excess_variance"]) for t in tertiles]
    lo = [float(t["excess_ci_lo"]) for t in tertiles]
    hi = [float(t["excess_ci_hi"]) for t in tertiles]
    yerr = [
        [e - l if math.isfinite(e) and math.isfinite(l) else 0 for e, l in zip(excess, lo, strict=True)],
        [h - e if math.isfinite(e) and math.isfinite(h) else 0 for e, h in zip(excess, hi, strict=True)],
    ]
    ax.bar(labels, excess, yerr=yerr, capsize=4, color=["#4c72b0", "#55a868", "#c44e52"])
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("Excess variance (mag/airmass)^2")
    ax.set_title("H0 excess variance by brightness tertile")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_group_shares(decomp: dict[str, Any], out: Path) -> None:
    groups = decomp.get("group_shares") or {}
    names = list(groups.keys())
    shares = [float(groups[g].get("share_of_total_ss", 0) or 0) for g in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, shares, color="#8172b3")
    ax.set_ylabel("Share of total SS")
    ax.set_title("P2 variance decomposition (nested groups)")
    ax.set_ylim(0, max(shares + [0.05]) * 1.2)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _write_result_md(
    path: Path,
    *,
    stars: list[dict[str, Any]],
    tertiles: list[dict[str, Any]],
    tertiles_legacy: list[dict[str, Any]],
    se_audit: list[dict[str, Any]],
    detector_audit: dict[str, Any],
    p3: dict[str, Any],
    decomp: dict[str, Any],
    outcome: dict[str, Any],
    p4: dict[str, Any],
    univariate: dict[str, Any],
    superseded: bool = True,
) -> None:
    sd_bx = float(np.std([float(s["b_X"]) for s in stars], ddof=1)) if len(stars) >= 2 else float("nan")
    lines = [
        "# WIDE-SLOPE-NOISE result (WSN-FIX corrected)",
        "",
    ]
    if superseded:
        lines.extend([
            "**SUPERSEDES:** prior EXCESS_UNATTRIBUTED verdict in CURSOR_RESULT_wide_slope_noise.md "
            "(inverted tertile labels + WLS residual SE). See CURSOR_RESULT_wsn_fix.md.",
            "",
        ])
    lines.extend([
        f"Generated: {datetime.now(timezone.utc).isoformat()} UTC",
        f"Git: {_git_head()}",
        f"Cell: {CELL_KEY} (draft_{DRAFT_ID}, n={len(stars)}, SD(b_X)={sd_bx:.4f} mag/airmass)",
        "",
        "## Detector drift path",
        "",
        f"- {detector_audit.get('path_note', DETECTOR_PATH_NOTE)}",
        f"- alignment_report exists: {detector_audit.get('alignment_report_exists')}, "
        f"shift columns: {detector_audit.get('has_shift_columns')}",
        "",
        "## SE audit (one worked star per tertile)",
        "",
        "| Tertile | catalog_id | N | SD(X) | med err | hand SE | analytic SE | bootstrap SE | se_use |",
        "|---------|------------|---:|------:|--------:|--------:|------------:|-------------:|-------:|",
    ])
    for row in se_audit:
        lines.append(
            f"| {row['tertile']} | {row.get('catalog_id', '')} | {row['N']} | {row['SD_X']:.4f} | "
            f"{row['median_err_epoch']:.5f} | {row['se_hand_formula']:.5f} | "
            f"{row['se_analytic_propagated']:.5f} | {row['se_bootstrap']:.5f} | {row['se_use']:.5f} |"
        )
    lines.extend([
        "",
        "## Tertile label check (corrected: lower mag_g = brighter)",
        "",
        "| Tertile | n | mag_min | mag_max |",
        "|---------|---:|--------:|--------:|",
    ])
    for t in tertiles:
        lines.append(
            f"| {t['tertile']} | {t['n']} | {t.get('mag_min', float('nan')):.2f} | "
            f"{t.get('mag_max', float('nan')):.2f} |"
        )
    lines.extend([
        "",
        "## P3 physical effect-size table (pre-registered, before P2)",
        "",
        f"Drift keys: span={p3.get('drift_span_key')}, corr={p3.get('drift_corr_key')}",
        "",
        f"| Quantity | Value |",
        f"|----------|------:|",
        f"| Measurement floor median SE | {p3.get('measurement_floor_median_se', float('nan')):.5f} |",
        f"| H1 drift span px (p90) | {p3.get('H1_drift_span_px_p90', float('nan')):.2f} |",
        f"| H1 |corr(pos,X)| (p90) | {p3.get('H1_drift_x_corr_abs_p90', float('nan')):.4f} |",
    ])
    for row in p3.get("H1_flat_scenarios", []):
        lines.append(
            f"| H1 attainable b_X @ eps={row['eps_flat']:.3f} | {row['attainable_bX_p90']:.5f} "
            f"(testable={row['testable']}) |"
        )
    lines.extend([
        f"| H2 attainable b_X (p90) | {p3.get('H2_attainable_bX_p90', float('nan')):.5f} "
        f"(testable={p3.get('H2_testable')}) |",
        f"| H4 colour bound (control) | {p3.get('H4_colour_bound_mag_airmass', 0.031):.3f} |",
        "",
        "## P1 noise floor (H0) -- corrected vs legacy inverted labels",
        "",
        "| Tertile | n | SD_obs | median SE | excess var | noise-dom | mag range |",
        "|---------|---:|-------:|----------:|-----------:|----------|-----------|",
    ])
    for t in tertiles:
        lines.append(
            f"| {t['tertile']} (new) | {t['n']} | {t['sd_obs']:.4f} | {t['median_se']:.5f} | "
            f"{t['excess_variance']:.6f} | {t['noise_dominated']} | "
            f"{t.get('mag_min', float('nan')):.1f}-{t.get('mag_max', float('nan')):.1f} |"
        )
    for t in tertiles_legacy:
        lines.append(
            f"| {t['tertile']} (old) | {t['n']} | {t['sd_obs']:.4f} | {t['median_se']:.5f} | "
            f"{t['excess_variance']:.6f} | {t['noise_dominated']} | "
            f"{t.get('mag_min', float('nan')):.1f}-{t.get('mag_max', float('nan')):.1f} |"
        )
    lines.extend([
        "",
        "## P2 regression decomposition",
        "",
        f"n={decomp.get('n')}, chi2_red={decomp.get('chi2_red', float('nan')):.2f}, "
        f"overdispersion scale={decomp.get('overdispersion_scale', float('nan')):.2f}, "
        f"CV R2 full={decomp.get('cv_r2_full', float('nan')):.3f}",
        "",
        "| Group | share SS | q (FDR) | reject | share boot CI | CV R2 |",
        "|-------|---------:|--------:|--------|---------------|------:|",
    ])
    for g, info in (decomp.get("group_shares") or {}).items():
        cv_g = (decomp.get("cv_r2_by_group") or {}).get(g, float("nan"))
        lines.append(
            f"| {g} | {info.get('share_of_total_ss', float('nan')):.3f} | "
            f"{info.get('q_value', float('nan')):.4f} | {info.get('reject_fdr')} | "
            f"[{info.get('share_bootstrap_ci_lo', float('nan')):.3f}, "
            f"{info.get('share_bootstrap_ci_hi', float('nan')):.3f}] | {cv_g:.3f} |"
        )
    if decomp.get("warnings"):
        lines.append("")
        lines.append("Warnings: " + "; ".join(decomp["warnings"]))
    lines.extend([
        "",
        "## P2b univariate hypothesis scan (exploratory)",
        "",
        "| Proxy | n | Spearman rho | linear R^2 |",
        "|-------|---:|-------------:|-----------:|",
    ])
    for g, info in univariate.items():
        if g == "n" or not isinstance(info, dict):
            continue
        lines.append(
            f"| {g} | {info.get('n', 0)} | {info.get('rho', float('nan')):.4f} | "
            f"{info.get('r2_linear', float('nan')):.4f} |"
        )
    lines.extend([
        "",
        "## P4 cross-checks",
        "",
        f"- PZQ sigma_r reference: {PZQ_SIGMA_R_MM:.1f} mmag [{PZQ_SIGMA_R_CI[0]}, {PZQ_SIGMA_R_CI[1]}]",
        f"- Rig constant reference: {RIG_CONSTANT_MM:.1f} mmag",
        f"- RMS spatial+detector-drift+fwhm fitted component: {p4.get('rms_spatial_drift_fwhm_mmag', float('nan')):.2f} mmag",
        f"- P4 consistency: {p4.get('consistency_detail', 'n/a')} (ratio={p4.get('consistency_ratio', float('nan')):.2f})",
        f"- Colour slope spread: {decomp.get('colour_slope_spread', float('nan')):.4f} mag/airmass (H4 bound <= 0.031)",
        "",
        "## P5 outcome",
        "",
        f"**{outcome.get('verdict')}** -- {outcome.get('detail')}",
        "",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _se_audit_one_star(cfg: AppConfig, catalog_id: str, tertile: str) -> dict[str, Any] | None:
    """Recompute step-by-step SE audit for a single star (no full cohort pass)."""
    cell = _load_wide_cell()
    base = next((s for s in cell.get("stars", []) if str(s.get("catalog_id")) == catalog_id), None)
    if base is None:
        return None
    mini = enrich_stars(cfg, {"stars": [base], "cell_key": CELL_KEY})
    if not mini:
        return None
    star = mini[0]
    lc = star.get("_audit_lc") or {}
    if not lc:
        return None
    steps = slope_se_audit_steps(
        np.asarray(lc["mags"], dtype=np.float64),
        np.asarray(lc["airmass"], dtype=np.float64),
        np.asarray(lc["err"], dtype=np.float64),
        bootstrap_draws=300,
        seed=424,
    )
    steps["tertile"] = tertile
    steps["catalog_id"] = catalog_id
    return steps


def _build_se_audit(cfg: AppConfig, stars: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    se_audit: list[dict[str, Any]] = []
    for pick in _pick_worked_stars_per_tertile(stars):
        lc = pick["star"].get("_audit_lc") or {}
        if lc:
            steps = slope_se_audit_steps(
                np.asarray(lc["mags"], dtype=np.float64),
                np.asarray(lc["airmass"], dtype=np.float64),
                np.asarray(lc["err"], dtype=np.float64),
                bootstrap_draws=300,
                seed=424,
            )
            steps["tertile"] = pick["tertile"]
            steps["catalog_id"] = pick["star"].get("catalog_id")
            se_audit.append(steps)
    if se_audit:
        return se_audit
    prior = out_dir / "se_audit.json"
    if prior.is_file():
        old = json.loads(prior.read_text(encoding="utf-8")).get("steps", [])
        if old:
            return old
    # Recompute for one star per tertile from mag_g tertiles on cached table.
    mags = np.asarray([float(s["mag_g"]) for s in stars if s.get("mag_g") is not None], dtype=np.float64)
    if len(mags) < 6:
        return se_audit
    for label, lo, hi in brightness_tertile_slices(mags):
        sub = [s for s in stars if s.get("mag_g") is not None and lo <= float(s["mag_g"]) < hi]
        if not sub:
            continue
        sub.sort(key=lambda s: abs(float(s.get("b_X", 0.0))))
        rep = sub[len(sub) // 2]
        cid = str(rep.get("catalog_id", ""))
        if not cid:
            continue
        one = _se_audit_one_star(cfg, cid, label)
        if one is not None:
            se_audit.append(one)
    return se_audit


def run_analysis(out_dir: Path, cfg: AppConfig, *, from_cache: bool = False, refresh_detector: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    detector_audit = _alignment_transform_audit(cfg)
    cache_path = out_dir / "per_star_table.json"
    if from_cache and cache_path.is_file():
        stars = json.loads(cache_path.read_text(encoding="utf-8")).get("stars", [])
        if refresh_detector:
            stars = _refresh_detector_drift(stars, cfg)
    else:
        cell = _load_wide_cell()
        stars = enrich_stars(cfg, cell)
    if len(stars) < 10:
        raise RuntimeError(f"Too few enriched stars: {len(stars)}")

    # SE audit: per-star LC cache, prior file, or targeted single-star recompute.
    se_audit = _build_se_audit(cfg, stars, out_dir)

    tertiles = excess_variance_by_tertile(stars, seed=424)
    tertiles_legacy = excess_variance_by_tertile(stars, seed=424, legacy_inverted_labels=True)
    p3 = physical_effect_size_table(stars)
    decomp = variance_decomposition_regression(stars, seed=424)
    univariate = univariate_hypothesis_scan(stars)
    outcome = pre_registered_outcome(tertiles, decomp)

    coef = decomp.get("coef_overdisp_scaled") or decomp.get("coef") or {}
    spatial_keys = [
        "x_med", "y_med", "r2_norm",
        "det_drift_x_corr", "det_drift_span_px",
        "fwhm_sens",
    ]
    rms_comp = rms_predicted_mmag(coef, stars, spatial_keys)
    p4_check = p4_noise_consistency_check(rms_comp, sigma_r_ref_mmag=PZQ_SIGMA_R_MM)
    p4 = {
        "rms_spatial_drift_fwhm_mmag": rms_comp,
        "consistency_passed": p4_check["passed"],
        "consistency_ratio": p4_check["ratio"],
        "consistency_detail": p4_check["detail"],
        "colour_share": float((decomp.get("group_shares") or {}).get("colour", {}).get("share_of_total_ss", float("nan"))),
        "median_sigma_r_mmag": float(np.nanmedian([
            float(s["sigma_r"]) * 1000.0 for s in stars if s.get("sigma_r") is not None and math.isfinite(float(s["sigma_r"]))
        ])),
        "det_drift_span_p90": float(np.nanpercentile([
            float(s["det_drift_span_px"]) for s in stars
            if s.get("det_drift_span_px") is not None and math.isfinite(float(s["det_drift_span_px"]))
        ], 90)) if stars else float("nan"),
        "aligned_drift_span_p90": float(np.nanpercentile([
            float(s["drift_span_px"]) for s in stars
            if s.get("drift_span_px") is not None and math.isfinite(float(s["drift_span_px"]))
        ], 90)) if stars else float("nan"),
    }

    per_star_path = out_dir / "per_star_table.json"
    export_stars = [{k: v for k, v in s.items() if not str(k).startswith("_")} for s in stars]
    per_star_path.write_text(json.dumps(_stamp({"stars": export_stars}), indent=2), encoding="utf-8")

    decomposition = _stamp({
        "detector_path_audit": detector_audit,
        "se_audit": se_audit,
        "tertiles_h0": tertiles,
        "tertiles_h0_legacy_inverted": tertiles_legacy,
        "p3_effect_sizes": p3,
        "p2_decomposition": decomp,
        "p2_univariate_scan": univariate,
        "p4_cross_checks": p4,
        "p5_outcome": outcome,
        "n_stars": len(stars),
        "sd_bx": float(np.std([float(s["b_X"]) for s in stars], ddof=1)),
    })
    (out_dir / "decomposition.json").write_text(json.dumps(decomposition, indent=2), encoding="utf-8")
    (out_dir / "se_audit.json").write_text(json.dumps(_stamp({"steps": se_audit}), indent=2), encoding="utf-8")

    _plot_field_map(stars, fig_dir / "bx_field_map.png")
    _plot_excess_tertiles(tertiles, fig_dir / "excess_by_tertile.png")
    if decomp.get("group_shares"):
        _plot_group_shares(decomp, fig_dir / "group_shares.png")

    _write_result_md(
        out_dir / "WIDE_SLOPE_NOISE_result.md",
        stars=stars,
        tertiles=tertiles,
        tertiles_legacy=tertiles_legacy,
        se_audit=se_audit,
        detector_audit=detector_audit,
        p3=p3,
        decomp=decomp,
        outcome=outcome,
        p4=p4,
        univariate=univariate,
    )
    return decomposition


def main() -> None:
    parser = argparse.ArgumentParser(description="WIDE-SLOPE-NOISE report-only analysis")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "tmp" / "wide_slope_noise",
    )
    parser.add_argument("--from-cache", action="store_true", help="Reuse per_star_table.json")
    parser.add_argument(
        "--refresh-detector",
        action="store_true",
        help="With --from-cache: recompute detector drift only (fixed cal FITS mapping)",
    )
    args = parser.parse_args()
    cfg = AppConfig()
    summary = run_analysis(
        args.out_dir, cfg, from_cache=args.from_cache, refresh_detector=args.refresh_detector,
    )
    print(json.dumps({"out_dir": str(args.out_dir), "verdict": summary["p5_outcome"]["verdict"], "n": summary["n_stars"]}, indent=2))


if __name__ == "__main__":
    main()
