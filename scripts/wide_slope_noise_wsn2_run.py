#!/usr/bin/env python3
"""WSN-2: corrected P4 excess integration, neighbor contamination, final verdict + park."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
import astropy.units as u  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from database import query_local_gaia  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from k2_cohort_core import weighted_linear_regression  # noqa: E402
from wide_slope_noise_core import (  # noqa: E402
    analytic_slope_se,
    attainable_neighbor_slope,
    contamination_fraction,
    excess_variance_by_tertile,
    final_wsn_outcome,
    fwhm_to_sigma,
    neighbor_attainable_table,
    neighbor_sensitivity_mag_per_fwhm_px,
    p4_excess_integration,
    p4_noise_consistency_check,
    pre_registered_outcome,
    rms_predicted_mmag,
    variance_decomposition_regression,
)
from scripts.wide_slope_noise_run import (  # noqa: E402
    CELL_KEY,
    COHORT_JSON,
    DRAFT_ID,
    PZQ_SIGMA_R_CI,
    PZQ_SIGMA_R_MM,
    RIG_CONSTANT_MM,
    SETUP,
    _git_head,
    _load_wide_cell,
    _stamp,
)

PLATE_SCALE_ARCSEC = 9.77  # wide rig draft_424 (pre-registered in spec context)
OUT_DIR_DEFAULT = _ROOT / "tmp" / "wide_slope_noise"


def _cohort_airmass_sd_x(cfg: AppConfig) -> float:
    from check_star_kmag import resolve_proc_csv_dir  # noqa: PLC0415
    from scripts.k2_cohort_run import _host_context, _pick_host_target, _sparse_target_ids  # noqa: PLC0415

    phot_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}" / "platesolve" / SETUP / "photometry"
    proc_dir = resolve_proc_csv_dir(phot_dir, SETUP)
    comp_all = pd.read_csv(
        phot_dir / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    host = _pick_host_target(comp_all, str(comp_all.iloc[0]["catalog_id"]), sparse_targets=_sparse_target_ids(phot_dir))
    ctx = _host_context(host, phot_dir=phot_dir, comp_all=comp_all, proc_dir=proc_dir, cfg=cfg, csv_cache={})
    am = np.asarray(ctx["lc_airmass"], dtype=np.float64)
    return float(np.std(am[np.isfinite(am)], ddof=1))


def _night_fwhm_airmass_regression(cfg: AppConfig) -> dict[str, Any]:
    from check_star_kmag import resolve_proc_csv_dir  # noqa: PLC0415
    from scripts.k2_cohort_run import _host_context, _pick_host_target, _sparse_target_ids  # noqa: PLC0415

    phot_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}" / "platesolve" / SETUP / "photometry"
    proc_dir = resolve_proc_csv_dir(phot_dir, SETUP)
    comp_all = pd.read_csv(
        phot_dir / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    host = _pick_host_target(comp_all, str(comp_all.iloc[0]["catalog_id"]), sparse_targets=_sparse_target_ids(phot_dir))
    ctx = _host_context(host, phot_dir=phot_dir, comp_all=comp_all, proc_dir=proc_dir, cfg=cfg, csv_cache={})
    lc = ctx["lc_df"]
    am = np.asarray(ctx["lc_airmass"], dtype=np.float64)
    fwhms: list[float] = []
    airm: list[float] = []
    apertures: list[float] = []
    for i, sf in enumerate(lc["source_file"].astype(str).tolist()):
        if i >= len(am):
            break
        path = proc_dir / sf
        if not path.is_file():
            continue
        df = pd.read_csv(path, usecols=["fwhm_estimate_px", "aperture_r_px"])
        if df.empty:
            continue
        f = float(np.nanmedian(pd.to_numeric(df["fwhm_estimate_px"], errors="coerce")))
        a = float(np.nanmedian(pd.to_numeric(df["aperture_r_px"], errors="coerce")))
        if math.isfinite(f) and math.isfinite(am[i]):
            fwhms.append(f)
            airm.append(float(am[i]))
            if math.isfinite(a):
                apertures.append(a)
    f_arr = np.asarray(fwhms, dtype=np.float64)
    x_arr = np.asarray(airm, dtype=np.float64)
    fit = weighted_linear_regression(x_arr, f_arr, np.ones(len(f_arr)))
    resid = f_arr - fit["intercept"] - fit["slope"] * x_arr
    return {
        "fwhm_median_px": float(np.median(f_arr)),
        "fwhm_p90_px": float(np.percentile(f_arr, 90)),
        "r_ap_median_px": float(np.median(apertures)) if apertures else 3.818,
        "dfwhm_dairmass_px": float(fit["slope"]),
        "dfwhm_dairmass_se": float(fit["slope_se"]),
        "fwhm_airmass_scatter_px": float(np.std(resid, ddof=1)),
        "n_epochs": int(len(f_arr)),
        "sigma_psf_p90_px": fwhm_to_sigma(float(np.percentile(f_arr, 90))),
    }


def _star_positions(cfg: AppConfig) -> dict[str, dict[str, float]]:
    phot_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}" / "platesolve" / SETUP / "photometry"
    comp = pd.read_csv(
        phot_dir / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str},
    )
    out: dict[str, dict[str, float]] = {}
    for _, row in comp.iterrows():
        cid = normalize_gaia_source_id(row.get("catalog_id"))
        if not cid:
            continue
        gmag = row.get("phot_g_mean_mag")
        if gmag is None or (isinstance(gmag, float) and not math.isfinite(gmag)):
            gmag = row.get("mag")
        out[cid] = {
            "ra_deg": float(row["ra_deg"]),
            "dec_deg": float(row["dec_deg"]),
            "g_mag": float(pd.to_numeric(gmag, errors="coerce")),
        }
    return out


def _gaia_neighbors(
    cfg: AppConfig,
    ra_deg: float,
    dec_deg: float,
    *,
    r_search_px: float,
    self_id: str,
) -> list[dict[str, float]]:
    r_deg = float(r_search_px * PLATE_SCALE_ARCSEC / 3600.0)
    pad = r_deg * 1.2
    rows = query_local_gaia(
        cfg.gaia_db_path,
        ra_min=ra_deg - pad,
        ra_max=ra_deg + pad,
        dec_min=dec_deg - pad,
        dec_max=dec_deg + pad,
        mag_limit=None,
    )
    c0 = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    neighbors: list[dict[str, float]] = []
    for row in rows:
        sid = normalize_gaia_source_id(row.get("source_id"))
        if sid == self_id:
            continue
        c1 = SkyCoord(ra=float(row["ra"]) * u.deg, dec=float(row["dec"]) * u.deg)
        sep_px = float(c0.separation(c1).arcsec / PLATE_SCALE_ARCSEC)
        if sep_px <= 0.05 or sep_px > r_search_px:
            continue
        g = float(row.get("g_mag", float("nan")))
        if not math.isfinite(g):
            continue
        neighbors.append({"sep_px": sep_px, "g_mag": g, "source_id": sid})
    return neighbors


def enrich_neighbor_hypothesis(
    stars: list[dict[str, Any]],
    cfg: AppConfig,
    fwhm_stats: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positions = _star_positions(cfg)
    fwhm_p90 = float(fwhm_stats["fwhm_p90_px"])
    fwhm_med = float(fwhm_stats["fwhm_median_px"])
    r_ap = float(fwhm_stats["r_ap_median_px"])
    sigma_p90 = float(fwhm_stats["sigma_psf_p90_px"])
    r_search_px = r_ap + 3.0 * sigma_p90
    r_search_arcsec = r_search_px * PLATE_SCALE_ARCSEC
    dfwhm_dx = float(fwhm_stats["dfwhm_dairmass_px"])

    enriched: list[dict[str, Any]] = []
    b_attain_vals: list[float] = []
    for s in stars:
        cid = normalize_gaia_source_id(s.get("catalog_id"))
        row = dict(s)
        pos = positions.get(cid)
        if not cid or pos is None:
            row.update({"neighbor_S": float("nan"), "neighbor_b_attain": float("nan"), "neighbor_fc_p90": float("nan")})
            enriched.append(row)
            continue
        nbs = _gaia_neighbors(
            cfg, pos["ra_deg"], pos["dec_deg"],
            r_search_px=r_search_px, self_id=cid,
        )
        g_tgt = float(s.get("mag_g", pos.get("g_mag", float("nan"))))
        fc_p90 = contamination_fraction(nbs, r_ap_px=r_ap, fwhm_px=fwhm_p90, target_g_mag=g_tgt)
        fc_med = contamination_fraction(nbs, r_ap_px=r_ap, fwhm_px=fwhm_med, target_g_mag=g_tgt)
        sens = neighbor_sensitivity_mag_per_fwhm_px(fc_p90, fc_med, fwhm_p90, fwhm_med)
        b_att = attainable_neighbor_slope(sens, dfwhm_dx)
        row.update({
            "neighbor_n": len(nbs),
            "neighbor_fc_p90": fc_p90,
            "neighbor_fc_med": fc_med,
            "neighbor_S": sens,
            "neighbor_b_attain": b_att,
        })
        if math.isfinite(b_att):
            b_attain_vals.append(b_att)
        enriched.append(row)

    meta = {
        "plate_scale_arcsec_per_px": PLATE_SCALE_ARCSEC,
        "r_ap_px": r_ap,
        "fwhm_p90_px": fwhm_p90,
        "fwhm_median_px": fwhm_med,
        "sigma_psf_p90_px": sigma_p90,
        "neighbor_search_radius_px": r_search_px,
        "neighbor_search_radius_arcsec": r_search_arcsec,
        "overlap_method": "2D Gaussian disk quadrature (grid_n=48); O_j = integral_aperture PSF_j",
        "dfwhm_dairmass_px": dfwhm_dx,
        "fwhm_airmass_regression": fwhm_stats,
    }
    return enriched, {"meta": meta, "b_attain_values": b_attain_vals}


def _plot_sigma_slope_scatter(p4: dict[str, Any], out: Path) -> None:
    stars = p4.get("stars") or []
    if not stars:
        return
    x = [float(s["sigma_r_mmag"]) for s in stars if math.isfinite(float(s.get("sigma_r_mmag", float("nan"))))]
    y = [float(s["sigma_slope_pt_tertile_mmag"]) for s in stars if math.isfinite(float(s.get("sigma_r_mmag", float("nan"))))]
    if len(x) < 3:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=12, alpha=0.7, c="#4c72b0")
    ax.axhline(PZQ_SIGMA_R_MM, color="green", ls="--", lw=0.8, label=f"PZQ sigma_r={PZQ_SIGMA_R_MM}")
    ax.axhline(RIG_CONSTANT_MM, color="orange", ls=":", lw=0.8, label=f"rig const={RIG_CONSTANT_MM}")
    ax.plot([0, max(x + y)], [0, max(x + y)], "k--", lw=0.5, alpha=0.4)
    ax.set_xlabel("sigma_r per star (mmag)")
    ax.set_ylabel("sigma_slope_pt tertile (mmag)")
    ax.set_title("P4 excess integration vs sigma_r")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _bounds_table(
    p3: dict[str, Any] | None,
    neighbor: dict[str, Any],
    p4: dict[str, Any],
) -> dict[str, str]:
    att = neighbor.get("attainable") or {}
    return {
        "colour": "<= 0.031 mag/airmass (K2 bound)",
        "spatial_drift_detector_fwhm": "each < 4% in-sample SS; CV collapse",
        "H1_flat_drift_detector": f"p90 span {p3.get('H1_drift_span_px_p90', float('nan')):.1f} px; eps=1% testable",
        "neighbor_contamination": (
            f"untestable-here (p90 |b_attain|={att.get('p90_abs_b_attain', float('nan')):.4f})"
            if not att.get("testable")
            else f"testable p90 |b_attain|={att.get('p90_abs_b_attain', float('nan')):.4f}"
        ),
        "sigma_slope_pt_cohort_median_mmag": f"{p4.get('cohort_median_sigma_slope_pt_mmag', float('nan')):.2f}",
        "sigma_r_reference_mmag": f"{PZQ_SIGMA_R_MM} [{PZQ_SIGMA_R_CI[0]}, {PZQ_SIGMA_R_CI[1]}]",
    }


def run_wsn2(out_dir: Path, cfg: AppConfig) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    cache = out_dir / "per_star_table.json"
    if not cache.is_file():
        raise FileNotFoundError(f"Missing {cache}; run wide_slope_noise_run.py first.")
    stars = json.loads(cache.read_text(encoding="utf-8")).get("stars", [])

    sd_x = _cohort_airmass_sd_x(cfg)
    fwhm_stats = _night_fwhm_airmass_regression(cfg)
    tertiles = excess_variance_by_tertile(stars, seed=424)

    p4 = p4_excess_integration(
        tertiles, stars,
        sd_x_airmass=sd_x,
        sigma_r_ref_mmag=PZQ_SIGMA_R_MM,
        sigma_r_ci=PZQ_SIGMA_R_CI,
        rig_constant_mmag=RIG_CONSTANT_MM,
    )
    _plot_sigma_slope_scatter(p4, fig_dir / "p4_sigma_slope_vs_sigma_r.png")

    # Legacy fitted P4 (superseded, kept for audit).
    decomp_path = out_dir / "decomposition.json"
    legacy_p4 = {}
    if decomp_path.is_file():
        legacy_p4 = json.loads(decomp_path.read_text(encoding="utf-8")).get("p4_cross_checks", {})

    stars_nb, nb_meta = enrich_neighbor_hypothesis(stars, cfg, fwhm_stats)
    tertile_sd = {t["tertile"]: float(t["sd_obs"]) for t in tertiles}
    med_se = float(np.median([float(s["se_use"]) for s in stars if s.get("se_use")]))
    attainable = neighbor_attainable_table(
        nb_meta["b_attain_values"],
        tertile_sd_obs=tertile_sd,
        measurement_floor=med_se,
    )
    attainable.update(nb_meta["meta"])

    neighbor_result: dict[str, Any] = {
        "attainable": attainable,
        "testable": bool(attainable.get("testable")),
    }

    if attainable.get("testable"):
        term_groups = {
            "colour": ["colour_offset_signed"],
            "spatial": ["x_med", "y_med", "r2_norm"],
            "drift_aligned": ["drift_x_corr"],
            "drift_detector": ["det_drift_x_corr", "det_drift_span_px"],
            "fwhm": ["fwhm_sens", "aperture_over_fwhm"],
            "mag": ["mag_g"],
            "neighbor": ["neighbor_S", "neighbor_b_attain"],
        }
        decomp = variance_decomposition_regression(stars_nb, term_groups=term_groups, seed=424)
        nb_share = float((decomp.get("group_shares") or {}).get("neighbor", {}).get("share_of_total_ss", 0) or 0)
        nb_reject = bool((decomp.get("group_shares") or {}).get("neighbor", {}).get("reject_fdr"))
        neighbor_result["regression"] = decomp
        neighbor_result["regression_share"] = nb_share
        neighbor_result["regression_reject_fdr"] = nb_reject
        # Direct weighted correlation b_X vs b_attain (sign predicted positive).
        pairs_stars = [
            s for s in stars_nb
            if s.get("neighbor_b_attain") is not None and s.get("b_X") is not None
            and math.isfinite(float(s["neighbor_b_attain"])) and math.isfinite(float(s["b_X"]))
            and s.get("se_use") is not None and float(s["se_use"]) > 0
        ]
        if len(pairs_stars) >= 5:
            xs = np.asarray([float(s["neighbor_b_attain"]) for s in pairs_stars], dtype=np.float64)
            ys = np.asarray([float(s["b_X"]) for s in pairs_stars], dtype=np.float64)
            w = np.asarray([1.0 / (float(s["se_use"]) ** 2) for s in pairs_stars], dtype=np.float64)
            fit = weighted_linear_regression(xs, ys, w)
            neighbor_result["direct_corr"] = {
                "slope": float(fit["slope"]),
                "slope_se": float(fit["slope_se"]),
                "n": len(pairs_stars),
                "sign_positive": bool(fit["slope"] > 0),
            }
    else:
        neighbor_result["stopped_at"] = (
            f"p90 |b_attain|={attainable.get('p90_abs_b_attain', float('nan')):.4f} "
            f"< floor {attainable.get('measurement_floor', float('nan')):.4f}; no correlation computed."
        )

    p3 = {}
    if decomp_path.is_file():
        p3 = json.loads(decomp_path.read_text(encoding="utf-8")).get("p3_effect_sizes", {})

    neighbor_result["bounds_table"] = _bounds_table(p3, neighbor_result, p4)

    decomp_base = {}
    if decomp_path.is_file():
        decomp_base = json.loads(decomp_path.read_text(encoding="utf-8")).get("p2_decomposition", {})
    outcome = final_wsn_outcome(tertiles, decomp_base, p4, neighbor_result)

    summary = _stamp({
        "p4_excess_integration": p4,
        "p4_legacy_fitted_superseded": legacy_p4,
        "neighbor_contamination": neighbor_result,
        "fwhm_night_regression": fwhm_stats,
        "p5_final_outcome": outcome,
        "n_stars": len(stars),
    })
    (out_dir / "wsn2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Update per-star table with neighbor fields.
    export = [{k: v for k, v in s.items() if not str(k).startswith("_")} for s in stars_nb]
    (out_dir / "per_star_table.json").write_text(
        json.dumps(_stamp({"stars": export}), indent=2), encoding="utf-8",
    )

    _write_wsn2_md(out_dir / "WIDE_SLOPE_NOISE_wsn2_result.md", summary)
    return summary


def _write_wsn2_md(path: Path, summary: dict[str, Any]) -> None:
    p4 = summary.get("p4_excess_integration") or {}
    nb = summary.get("neighbor_contamination") or {}
    att = nb.get("attainable") or {}
    outcome = summary.get("p5_final_outcome") or {}
    fwhm = summary.get("fwhm_night_regression") or {}
    lines = [
        "# WIDE-SLOPE-NOISE WSN-2 result",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()} UTC",
        f"Git: {_git_head()}",
        "",
        "## P4 corrected excess integration (supersedes fitted-RMS FAIL)",
        "",
        f"SD(X) airmass = {p4.get('sd_x_airmass', float('nan')):.4f} mag/airmass",
        f"Cohort median sigma_slope_pt = {p4.get('cohort_median_sigma_slope_pt_mmag', float('nan')):.2f} mmag",
        f"Status: {p4.get('status')} -- {p4.get('unification_detail')}",
        f"Caveat: {p4.get('caveat')}",
        "",
        "| Tertile | excess var | sigma_slope_pt (mmag) | CI (mmag) |",
        "|---------|----------:|----------------------:|----------:|",
    ]
    for t in p4.get("tertiles") or []:
        lo = t.get("sigma_slope_pt_ci_lo_mmag", float("nan"))
        hi = t.get("sigma_slope_pt_ci_hi_mmag", float("nan"))
        lines.append(
            f"| {t.get('tertile')} | {t.get('excess_variance', float('nan')):.5f} | "
            f"{t.get('sigma_slope_pt_mmag', float('nan')):.2f} | [{lo:.2f}, {hi:.2f}] |"
        )
    lines.extend([
        "",
        "Star-by-star scatter: figures/p4_sigma_slope_vs_sigma_r.png",
        "",
        "## Neighbor contamination (pre-registered)",
        "",
        f"Search radius: {att.get('neighbor_search_radius_px', float('nan')):.2f} px "
        f"({att.get('neighbor_search_radius_arcsec', float('nan')):.2f} arcsec)",
        f"FWHM vs airmass: dFWHM/dX = {fwhm.get('dfwhm_dairmass_px', float('nan')):.4f} px/airmass "
        f"(scatter {fwhm.get('fwhm_airmass_scatter_px', float('nan')):.2f} px, n={fwhm.get('n_epochs')})",
        "",
        "### Attainable table (before test)",
        "",
        f"testable={att.get('testable')}; p50 |b_attain|={att.get('p50_abs_b_attain', float('nan')):.5f}; "
        f"p90={att.get('p90_abs_b_attain', float('nan')):.5f}; max={att.get('max_abs_b_attain', float('nan')):.5f}",
        f"measurement floor (median SE)={att.get('measurement_floor', float('nan')):.4f} mag/airmass",
        "",
    ])
    if nb.get("testable"):
        dc = nb.get("direct_corr") or {}
        lines.extend([
            "### Neighbor test (testable)",
            "",
            f"Regression neighbor share SS={nb.get('regression_share', float('nan')):.3f}; "
            f"FDR reject={nb.get('regression_reject_fdr')}",
            f"Direct b_X vs b_attain slope={dc.get('slope', float('nan')):.4f} "
            f"(sign positive={dc.get('sign_positive')})",
            "",
        ])
    else:
        lines.extend([
            "### Neighbor test",
            "",
            f"STOP: {nb.get('stopped_at', 'untestable-here')}",
            "",
        ])
    lines.extend([
        "## P5 final verdict",
        "",
        f"**{outcome.get('verdict')}** -- {outcome.get('detail')}",
        "",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description="WSN-2 final analysis")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    args = parser.parse_args()
    cfg = AppConfig()
    summary = run_wsn2(args.out_dir, cfg)
    print(json.dumps({
        "verdict": summary["p5_final_outcome"]["verdict"],
        "p4_status": summary["p4_excess_integration"]["status"],
        "neighbor_testable": summary["neighbor_contamination"]["testable"],
    }, indent=2))


if __name__ == "__main__":
    main()
