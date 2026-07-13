#!/usr/bin/env python3
"""SIGMA-NEWTON gate run: draft_426 fine-scale rigs + anomaly diagnostics."""

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

from check_star_kmag import (  # noqa: E402
    build_aligned_comp_inst,
    comp_ensemble_maps,
    resolve_proc_csv_dir,
)
from config import AppConfig  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from photometry_core import (  # noqa: E402
    ERR_BKG_SOURCE_COL,
    SIGMA_BKG_AP_COL,
    check_comparison_stability,
    ensemble_normalize,
)
from scripts.bingain_err_decompose import (  # noqa: E402
    _gain_from_lights,
    decompose_target_lc,
)
from scripts.bingain_fix_validate import (  # noqa: E402
    _chi2_lc_err,
    resolve_archive_root,
)
from scripts.chi2_sigma_gate import (  # noqa: E402
    bootstrap_chi2_dof_ci,
    evaluate_lc_chi2_variants,
    plot_chi2_vs_g,
    reduced_chi2_constant,
    saturation_margin_distribution,
    sigma_arrays_from_lc_and_proc,
    write_summary_json,
)
from scripts.sparse_comp_diag import V0611_CID  # noqa: E402
from sigma_budget import (  # noqa: E402
    SIGMA_VARIANT_PRODUCTION_LC_ERR,
    resolve_rig_scintillation_params,
)

_MAG_ERR_SCALE = 2.5 / math.log(10.0)
DRAFT_ID = 426
SETUPS = ("g_60_4", "i_70_4", "r_60_4")
IR_SETUPS = ("i_70_4", "r_60_4")


def _git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _norm_id(x: object) -> str:
    return str(normalize_gaia_source_id(x) or "").strip()


def _verify_setup_artifacts(archive_root: Path, setup: str) -> Path:
    phot = archive_root / "Drafts" / f"draft_{DRAFT_ID:06d}" / "platesolve" / setup / "photometry"
    lc_dir = phot / "lightcurves"
    meta = phot / "pipeline_meta.json"
    proc_dir = resolve_proc_csv_dir(phot, setup)
    missing: list[str] = []
    if not lc_dir.is_dir():
        missing.append(str(lc_dir))
    if not meta.is_file():
        missing.append(str(meta))
    if proc_dir is None or not proc_dir.is_dir():
        missing.append(f"proc_dir under {phot}")
    v0611_lc = lc_dir / f"lightcurve_{V0611_CID}.csv"
    if not v0611_lc.is_file():
        missing.append(str(v0611_lc))
    if missing:
        raise SystemExit(
            "ERROR: draft_426 archive artifacts missing for setup "
            f"{setup}:\n  " + "\n  ".join(missing)
        )
    return phot


def _list_check_star_ids(lc_dir: Path) -> list[str]:
    ids: list[str] = []
    for side in sorted(lc_dir.glob("check_kmag_*.csv")):
        cid = side.stem.replace("check_kmag_", "", 1)
        lc_path = lc_dir / f"lightcurve_{cid}.csv"
        if lc_path.is_file():
            ids.append(cid)
    return ids


def _work_lc_for_star(
    *,
    phot_dir: Path,
    lc_dir: Path,
    catalog_id: str,
    is_v0611: bool,
) -> tuple[pd.DataFrame, float | None]:
    lc_path = lc_dir / f"lightcurve_{catalog_id}.csv"
    lc_df = pd.read_csv(lc_path, low_memory=False)
    mag_g: float | None = None
    side = lc_dir / f"check_kmag_{catalog_id}.csv"
    if side.is_file():
        side_df = pd.read_csv(side, low_memory=False)
        work = side_df.copy()
        work["delta_mag"] = pd.to_numeric(work["kmag"], errors="coerce")
        work["source_file"] = lc_df["source_file"].astype(str).iloc[: len(work)].tolist()
        work["airmass"] = pd.to_numeric(lc_df["airmass"], errors="coerce").iloc[: len(work)].tolist()
        if "err" not in work.columns and "err" in lc_df.columns:
            work["err"] = pd.to_numeric(lc_df["err"], errors="coerce").iloc[: len(work)].tolist()
        if "bjd" not in work.columns and "bjd" in lc_df.columns:
            work["bjd"] = pd.to_numeric(lc_df["bjd"], errors="coerce").iloc[: len(work)].tolist()
        if "phot_g_mean_mag" in side_df.columns:
            g = float(pd.to_numeric(side_df["phot_g_mean_mag"].iloc[0], errors="coerce"))
            if math.isfinite(g):
                mag_g = g
        return work, mag_g
    if is_v0611:
        work = lc_df.copy()
        if "delta_mag" not in work.columns and "mag_calib" in work.columns:
            work["delta_mag"] = pd.to_numeric(work["mag_calib"], errors="coerce")
        return work, mag_g
    return lc_df, mag_g


def run_star_gate(
    *,
    phot_dir: Path,
    setup: str,
    catalog_id: str,
    out_dir: Path,
    cfg: AppConfig,
    is_v0611: bool = False,
) -> dict[str, Any]:
    lc_dir = phot_dir / "lightcurves"
    proc_dir = resolve_proc_csv_dir(phot_dir, setup)
    assert proc_dir is not None
    meta_path = phot_dir / "pipeline_meta.json"
    meta_json = json.loads(meta_path.read_text(encoding="utf-8"))
    rig = resolve_rig_scintillation_params(draft_id=DRAFT_ID, setup=setup, cfg=cfg, pipeline_meta=meta_json)
    lights = phot_dir.parents[2] / "detrended_aligned" / "lights" / setup
    if not lights.is_dir():
        lights = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}" / "detrended_aligned" / "lights" / setup
    gain = _gain_from_lights(lights, float(cfg.gain))
    rn = float(cfg.read_noise)

    work, mag_g = _work_lc_for_star(
        phot_dir=phot_dir, lc_dir=lc_dir, catalog_id=catalog_id, is_v0611=is_v0611,
    )
    mags, variants, _, _, sem_meta = sigma_arrays_from_lc_and_proc(
        work, proc_dir, catalog_id, rig_params=rig, gain=gain, read_noise=rn,
    )
    bjd = pd.to_numeric(work.get("bjd"), errors="coerce").to_numpy(dtype=np.float64)
    results = evaluate_lc_chi2_variants(
        mags, variants, catalog_id=catalog_id, mag_g=mag_g, bjd=bjd,
    )
    decomp = decompose_target_lc(
        lc_path=lc_dir / f"lightcurve_{catalog_id}.csv",
        proc_dir=proc_dir,
        target_cid=catalog_id,
        gain=gain,
        read_noise=rn,
    )
    sat = saturation_margin_distribution(work, proc_dir, catalog_id)
    payload: dict[str, Any] = {
        "draft_id": DRAFT_ID,
        "setup": setup,
        "catalog_id": catalog_id,
        "is_v0611": is_v0611,
        "rig": rig.to_dict(),
        "gain": gain,
        "read_noise": rn,
        "results": [r.to_dict() for r in results],
        "err_decomposition": decomp,
        "saturation_margin": sat,
        "bkg_term_source": sem_meta.get("bkg_term_source"),
        "ensemble_sem_meta": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in sem_meta.items()
            if k != "ensemble_sem_primary"
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_summary_json(payload, out_dir / f"chi2_gate_{catalog_id}.json")
    plot_path = plot_chi2_vs_g(
        results,
        out_dir / f"chi2_vs_g_{catalog_id}.png",
        title=f"{setup} {catalog_id}",
    )
    payload["json_path"] = json_path
    payload["plot_path"] = plot_path
    return payload


def _production_chi2_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    for r in payload.get("results", []):
        if str(r.get("variant")) == SIGMA_VARIANT_PRODUCTION_LC_ERR:
            return r
    return None


def _ensemble_frame_trace(
    *,
    phot_dir: Path,
    setup: str,
    catalog_id: str,
    cfg: AppConfig,
    gain: float | None = None,
    read_noise: float | None = None,
) -> dict[str, Any]:
    """Per-frame ensemble SEM vs raw comp residual scatter (B3 diagnostics)."""
    lc_dir = phot_dir / "lightcurves"
    lc_path = lc_dir / f"lightcurve_{catalog_id}.csv"
    side = lc_dir / f"check_kmag_{catalog_id}.csv"
    if not lc_path.is_file() or not side.is_file():
        return {"available": False}
    lc_df = pd.read_csv(lc_path, low_memory=False)
    side_df = pd.read_csv(side, low_memory=False)
    chk_cid = str(side_df["check_catalog_id"].iloc[0]) if "check_catalog_id" in side_df.columns else catalog_id
    proc_dir = resolve_proc_csv_dir(phot_dir, setup)
    if proc_dir is None:
        return {"available": False}
    g = float(gain) if gain is not None and gain > 0 else float(cfg.gain)
    rn = float(read_noise) if read_noise is not None else float(cfg.read_noise)
    comp_all = pd.read_csv(
        phot_dir / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    comp_df = comp_all.loc[comp_all["target_catalog_id"].map(_norm_id) == _norm_id(catalog_id)]
    cat, tier, rms, tw = comp_ensemble_maps(comp_df, cfg)
    comp_ids = [_norm_id(c) for c in comp_df["catalog_id"].tolist() if _norm_id(c)]
    if chk_cid not in comp_ids:
        comp_ids.append(chk_cid)
    comp_lc = build_aligned_comp_inst(
        proc_dir, comp_ids, lc_df["source_file"].astype(str).tolist(), cfg, "aperture",
    )
    if chk_cid not in comp_lc:
        return {"available": False, "reason": "check star missing from proc alignment"}
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

    frames: list[dict[str, Any]] = []
    err_lc = pd.to_numeric(lc_df.get("err"), errors="coerce").to_numpy(dtype=float)
    kmag = pd.to_numeric(side_df.get("kmag"), errors="coerce").to_numpy(dtype=float)
    kmag_fin = kmag[np.isfinite(kmag)]
    emp_scatter = float(np.std(kmag_fin, ddof=1)) if kmag_fin.size >= 2 else float("nan")
    err_med = float(np.nanmedian(err_lc[np.isfinite(err_lc) & (err_lc > 0)])) if err_lc.size else float("nan")
    err_mag_med = _MAG_ERR_SCALE * err_med if math.isfinite(err_med) else float("nan")

    for i in range(len(lc_df)):
        sf = str(lc_df.iloc[i].get("source_file", "")).strip()
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
        raw_scatter = float(np.std(np.asarray(comp_resid, dtype=float), ddof=1)) if len(comp_resid) >= 2 else float("nan")
        n_resid = len(comp_resid)
        ens = float(ensemble_scatter[i]) if i < len(ensemble_scatter) else float("nan")
        sem_over_raw = ens / raw_scatter if math.isfinite(ens) and math.isfinite(raw_scatter) and raw_scatter > 0 else float("nan")
        inv_sqrt_n = 1.0 / math.sqrt(n_resid) if n_resid > 0 else float("nan")
        e_lc = float(err_lc[i]) if i < len(err_lc) else float("nan")
        frames.append(
            {
                "source_file": sf,
                "n_comps": n_resid,
                "ensemble_scatter_mag": ens,
                "raw_comp_resid_scatter_mag": raw_scatter,
                "sem_over_raw_scatter": sem_over_raw,
                "inv_sqrt_n_comps": inv_sqrt_n,
                "err_lc_rel": e_lc,
                "err_lc_mag": _MAG_ERR_SCALE * e_lc if math.isfinite(e_lc) and e_lc > 0 else float("nan"),
            }
        )

    # comp photon contribution estimate per frame (median over comps)
    comp_photon_shares: list[float] = []
    for fr in frames:
        i_sf = fr["source_file"]
        proc_path = proc_dir / i_sf
        if not proc_path.is_file():
            continue
        try:
            pdf = pd.read_csv(proc_path, low_memory=False, dtype={"catalog_id": str})
        except Exception:  # noqa: BLE001
            continue
        pdf["_nid"] = pdf["catalog_id"].map(_norm_id)
        phot_vars: list[float] = []
        for cid in good_ids:
            sub = pdf.loc[pdf["_nid"] == cid]
            if sub.empty:
                continue
            flux = float(pd.to_numeric(sub.iloc[0].get("dao_flux"), errors="coerce"))
            sig_bkg = float(pd.to_numeric(sub.iloc[0].get(SIGMA_BKG_AP_COL), errors="coerce"))
            if not math.isfinite(flux) or flux <= 0:
                continue
            var_p = flux / g
            var_b = sig_bkg * sig_bkg if math.isfinite(sig_bkg) else float("nan")
            if math.isfinite(var_b):
                phot_vars.append(var_p + var_b)
        if phot_vars and math.isfinite(fr["err_lc_rel"]) and fr["err_lc_rel"] > 0:
            med_var = float(np.median(phot_vars))
            comp_phot_rel = math.sqrt(med_var) / 12000.0  # scale proxy; share uses ratio
            ens_mag = fr["ensemble_scatter_mag"]
            err_mag = fr["err_lc_mag"]
            if math.isfinite(ens_mag) and math.isfinite(err_mag) and err_mag > 0:
                comp_photon_shares.append((_MAG_ERR_SCALE * comp_phot_rel / err_mag) ** 2)

    sem_or_raw = [f["sem_over_raw_scatter"] for f in frames if math.isfinite(f["sem_over_raw_scatter"])]
    inv_sqrt = [f["inv_sqrt_n_comps"] for f in frames if math.isfinite(f["inv_sqrt_n_comps"])]
    ratio_err_emp = err_mag_med / emp_scatter if math.isfinite(err_mag_med) and math.isfinite(emp_scatter) and emp_scatter > 0 else float("nan")

    return {
        "available": True,
        "catalog_id": catalog_id,
        "check_catalog_id": chk_cid,
        "empirical_kmag_scatter_mag": emp_scatter,
        "lc_err_median_mag": err_mag_med,
        "err_over_empirical_scatter": ratio_err_emp,
        "chi2_predicted_if_err_2x_empirical": (1.0 / (ratio_err_emp**2)) if math.isfinite(ratio_err_emp) and ratio_err_emp > 0 else float("nan"),
        "sem_over_raw_scatter_median": float(np.median(sem_or_raw)) if sem_or_raw else None,
        "inv_sqrt_n_comps_median": float(np.median(inv_sqrt)) if inv_sqrt else None,
        "sem_behaves_as_sqrt_n": (
            abs(float(np.median(sem_or_raw)) - float(np.median(inv_sqrt))) < 0.05
            if sem_or_raw and inv_sqrt
            else None
        ),
        "comp_photon_share_of_ensemble_median": float(np.median(comp_photon_shares)) if comp_photon_shares else None,
        "frames": frames,
        "citations": {
            "ensemble_sem": "photometry_core.py:3113-3115 std(comp_resid)/sqrt(n)",
            "err_combine": "photometry_core.py:3205 _combine_err_with_ensemble_scatter_keyed",
        },
    }


def _plot_underdispersion(
    setup: str,
    stars: list[dict[str, Any]],
    out_path: Path,
) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels: list[str] = []
    err_meds: list[float] = []
    emp_sc: list[float] = []
    ens_sh: list[float] = []
    for st in stars:
        dec = st.get("err_decomposition", {}).get("medians", {})
        tr = st.get("ensemble_trace", {})
        labels.append(st["catalog_id"][-6:])
        err_meds.append(float(tr.get("lc_err_median_mag", float("nan"))))
        emp_sc.append(float(tr.get("empirical_kmag_scatter_mag", float("nan"))))
        ens_sh.append(float(dec.get("ensemble_share", float("nan"))) * 100.0)
    x = np.arange(len(labels))
    w = 0.35
    axes[0].bar(x - w / 2, err_meds, w, label="LC err median (mag)")
    axes[0].bar(x + w / 2, emp_sc, w, label="empirical kmag scatter")
    axes[0].set_xticks(x, labels, rotation=45, ha="right")
    axes[0].set_ylabel("mag")
    axes[0].set_title(f"{setup}: modeled err vs empirical scatter")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, ens_sh, color="#B279A2")
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1].set_ylabel("ensemble share (%)")
    axes[1].set_title("LC err decomposition")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)


def run_underdispersion_ir(
    *,
    archive_root: Path,
    cfg: AppConfig,
    star_payloads: dict[str, dict[str, dict[str, Any]]],
    out_dir: Path,
    gain_by_setup: dict[str, float],
) -> dict[str, Any]:
    report: dict[str, Any] = {"setups": {}, "summary": {}}
    for setup in IR_SETUPS:
        phot = archive_root / "Drafts" / f"draft_{DRAFT_ID:06d}" / "platesolve" / setup / "photometry"
        lc_dir = phot / "lightcurves"
        gain = gain_by_setup.get(setup, float(cfg.gain))
        stars_out: list[dict[str, Any]] = []
        for cid in _list_check_star_ids(lc_dir):
            pl = star_payloads.get(setup, {}).get(cid, {})
            trace = _ensemble_frame_trace(
                phot_dir=phot, setup=setup, catalog_id=cid, cfg=cfg, gain=gain,
            )
            entry = {
                "catalog_id": cid,
                "production_lc_err": _production_chi2_from_payload(pl),
                "err_decomposition": pl.get("err_decomposition", {}),
                "ensemble_trace": trace,
            }
            stars_out.append(entry)
        v0611_pl = star_payloads.get(setup, {}).get(V0611_CID, {})
        stars_out.insert(
            0,
            {
                "catalog_id": V0611_CID,
                "production_lc_err": _production_chi2_from_payload(v0611_pl),
                "err_decomposition": v0611_pl.get("err_decomposition", {}),
                "ensemble_trace": _ensemble_frame_trace(
                    phot_dir=phot, setup=setup, catalog_id=V0611_CID, cfg=cfg, gain=gain,
                ),
            },
        )
        fig_path = _plot_underdispersion(setup, stars_out, out_dir / f"underdispersion_{setup}.png")
        ratios = [
            float(s["ensemble_trace"].get("err_over_empirical_scatter"))
            for s in stars_out
            if s.get("ensemble_trace", {}).get("available")
            and math.isfinite(float(s["ensemble_trace"].get("err_over_empirical_scatter", float("nan"))))
        ]
        report["setups"][setup] = {
            "stars": stars_out,
            "figure": fig_path,
            "median_err_over_empirical_scatter": float(np.median(ratios)) if ratios else None,
        }
    v0611_ir_chi2: list[float] = []
    for s in IR_SETUPS:
        pl = star_payloads.get(s, {}).get(V0611_CID, {})
        prod = _production_chi2_from_payload(pl)
        if prod and prod.get("chi2_dof") is not None and math.isfinite(float(prod["chi2_dof"])):
            v0611_ir_chi2.append(float(prod["chi2_dof"]))
    report["summary"] = {
        "v0611_ir_production_lc_err": v0611_ir_chi2,
        "hypothesis": "ensemble SEM overestimate if err/empirical_scatter ~ 2",
    }
    path = write_summary_json(report, out_dir / "underdispersion_ir.json")
    report["json_path"] = path
    return report


def run_g60_heterogeneity(
    *,
    star_payloads: dict[str, dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    g_stars = star_payloads.get("g_60_4", {})
    rows: list[dict[str, Any]] = []
    for cid, pl in g_stars.items():
        prod = _production_chi2_from_payload(pl)
        if prod is None:
            continue
        med = pl.get("err_decomposition", {}).get("medians", {})
        sat = pl.get("saturation_margin", {})
        rows.append(
            {
                "catalog_id": cid,
                "chi2_dof": prod.get("chi2_dof"),
                "chi2_dof_ci_lo": prod.get("chi2_dof_ci_lo"),
                "chi2_dof_ci_hi": prod.get("chi2_dof_ci_hi"),
                "n_frames": prod.get("n_frames"),
                "ensemble_share": med.get("ensemble_share"),
                "photon_share": med.get("photon_share"),
                "background_share": med.get("background_share"),
                "saturation_fill_p95": sat.get("fill_p95"),
                "is_v0611": pl.get("is_v0611", False),
            }
        )
    rows.sort(key=lambda r: float(r.get("chi2_dof") or 0.0), reverse=True)
    chi2_vals = [float(r["chi2_dof"]) for r in rows if r.get("chi2_dof") is not None]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(chi2_vals, bins=min(12, max(4, len(chi2_vals))), edgecolor="black", alpha=0.75)
    ax.axvline(1.0, color="gray", linestyle=":")
    v0611_c2 = next((r["chi2_dof"] for r in rows if r.get("is_v0611")), None)
    if v0611_c2 is not None:
        ax.axvline(float(v0611_c2), color="#F58518", linestyle="--", label=f"V0611 {v0611_c2:.2f}")
    ax.set_xlabel("production_lc_err chi2/dof")
    ax.set_ylabel("count")
    ax.set_title("g_60_4 pooled check stars")
    ax.legend(fontsize=8)
    fig.tight_layout()
    hist_path = out_dir / "g60_heterogeneity_hist.png"
    fig.savefig(hist_path, dpi=120)
    plt.close(fig)

    over = [r for r in rows if float(r.get("chi2_dof") or 0) > 1.5]
    under = [r for r in rows if float(r.get("chi2_dof") or 0) < 0.6]
    pooled = [c for c in chi2_vals if math.isfinite(c)]
    trimmed = sorted(c for c in pooled if c < 20)
    verdict = "broad-based"
    drivers: list[dict[str, Any]] = []
    if over and len(over) <= max(2, len(rows) // 3):
        verdict = "subset-driven (overdispersed tail)"
        drivers = over
    elif under and len(under) <= max(2, len(rows) // 3) and not over:
        verdict = "subset-driven (underdispersed tail)"
        drivers = under
    elif over and under:
        verdict = "bimodal: overdispersed outlier(s) + underdispersed majority"
        drivers = over + under

    payload = {
        "setup": "g_60_4",
        "n_stars": len(rows),
        "pooled_median_chi2": float(np.median(pooled)) if pooled else None,
        "pooled_median_chi2_trimmed_lt20": float(np.median(trimmed)) if trimmed else None,
        "pooled_mean_chi2": float(np.mean(pooled)) if pooled else None,
        "stars_sorted": rows,
        "overdispersed_tail": over,
        "underdispersed_tail": under,
        "verdict": verdict,
        "suspected_drivers": drivers,
        "histogram": str(hist_path),
    }
    path = write_summary_json(payload, out_dir / "g60_heterogeneity.json")
    payload["json_path"] = path
    return payload


def n4_harness_sanity(archive_root: Path, cfg: AppConfig) -> dict[str, Any]:
    """Wide-rig draft_424 check star: production_lc_err vs bingain_fix_validate LC err chi2."""
    setup = "NoFilter_60_2"
    phot = archive_root / "Drafts" / "draft_000424" / "platesolve" / setup / "photometry"
    lc_dir = phot / "lightcurves"
    check_ids = _list_check_star_ids(lc_dir)
    if not check_ids:
        return {"available": False, "reason": "no check stars on draft_424 wide rig"}
    cid = check_ids[0]
    lc = lc_dir / f"lightcurve_{cid}.csv"
    side = lc_dir / f"check_kmag_{cid}.csv"
    chi2_ref, meta_ref = _chi2_lc_err(lc_path=lc, side_path=side)
    out_dir = Path("tmp/sigma_newton") / "n4_sanity"
    new_pl = run_star_gate(
        phot_dir=phot,
        setup=setup,
        catalog_id=cid,
        out_dir=out_dir,
        cfg=cfg,
        is_v0611=False,
    )
    prod = _production_chi2_from_payload(new_pl)
    chi2_new = float(prod["chi2_dof"]) if prod else float("nan")
    ci_lo = prod.get("chi2_dof_ci_lo") if prod else None
    ci_hi = prod.get("chi2_dof_ci_hi") if prod else None
    within = (
        chi2_ref is not None
        and math.isfinite(chi2_new)
        and ci_lo is not None
        and ci_hi is not None
        and float(ci_lo) <= float(chi2_ref) <= float(ci_hi)
    )
    return {
        "draft_id": 424,
        "setup": setup,
        "catalog_id": cid,
        "chi2_acceptance_ref": chi2_ref,
        "chi2_harness_production_lc_err": chi2_new,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "within_ci": within,
        "meta_ref": meta_ref,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-root", type=str, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("tmp/sigma_newton"))
    args = ap.parse_args()

    cfg = AppConfig()
    archive_root = resolve_archive_root(args.archive_root, cfg=cfg)
    cfg.archive_root = archive_root
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    star_payloads: dict[str, dict[str, dict[str, Any]]] = {}
    summary_setups: dict[str, Any] = {}
    gain_by_setup: dict[str, float] = {}

    for setup in SETUPS:
        phot = _verify_setup_artifacts(archive_root, setup)
        lc_dir = phot / "lightcurves"
        setup_dir = out_dir / setup
        star_payloads[setup] = {}
        lights = archive_root / "Drafts" / f"draft_{DRAFT_ID:06d}" / "detrended_aligned" / "lights" / setup
        gain_by_setup[setup] = _gain_from_lights(lights, float(cfg.gain))
        targets = list(dict.fromkeys([V0611_CID] + _list_check_star_ids(lc_dir)))
        per_star_table: list[dict[str, Any]] = []
        chi2_pool: list[float] = []
        ens_shares: list[float] = []
        bkg_counts: dict[str, int] = {}

        for cid in targets:
            pl = run_star_gate(
                phot_dir=phot,
                setup=setup,
                catalog_id=cid,
                out_dir=setup_dir,
                cfg=cfg,
                is_v0611=(cid == V0611_CID),
            )
            star_payloads[setup][cid] = pl
            prod = _production_chi2_from_payload(pl)
            med = pl.get("err_decomposition", {}).get("medians", {})
            ens = med.get("ensemble_share")
            if ens is not None and math.isfinite(float(ens)):
                ens_shares.append(float(ens))
            bts = pl.get("bkg_term_source", {}) or {}
            for k, v in (bts.get("counts") or {}).items():
                bkg_counts[k] = bkg_counts.get(k, 0) + int(v)
            if prod and prod.get("chi2_dof") is not None and math.isfinite(float(prod["chi2_dof"])):
                row = {
                    "catalog_id": cid,
                    "is_v0611": cid == V0611_CID,
                    "chi2_dof": prod.get("chi2_dof"),
                    "ci_lo": prod.get("chi2_dof_ci_lo"),
                    "ci_hi": prod.get("chi2_dof_ci_hi"),
                    "n_frames": prod.get("n_frames"),
                    "ensemble_share_median": ens,
                    "background_share_median": med.get("background_share"),
                    "photon_share_median": med.get("photon_share"),
                }
                per_star_table.append(row)
                if cid != V0611_CID:
                    chi2_pool.append(float(prod["chi2_dof"]))

        v0611_prod = _production_chi2_from_payload(star_payloads[setup].get(V0611_CID, {}))
        summary_setups[setup] = {
            "n_stars": len(targets),
            "v0611_production_lc_err": v0611_prod,
            "pooled_check_stars_production_lc_err": {
                "n": len(chi2_pool),
                "median": float(np.median(chi2_pool)) if chi2_pool else None,
                "values": chi2_pool,
            },
            "per_star_table": sorted(per_star_table, key=lambda r: float(r.get("chi2_dof") or 0), reverse=True),
            "ensemble_share_median": float(np.median(ens_shares)) if ens_shares else None,
            "bkg_term_source_counts": bkg_counts,
        }

    under = run_underdispersion_ir(
        archive_root=archive_root,
        cfg=cfg,
        star_payloads=star_payloads,
        out_dir=out_dir,
        gain_by_setup=gain_by_setup,
    )
    g60 = run_g60_heterogeneity(star_payloads=star_payloads, out_dir=out_dir)
    n4 = n4_harness_sanity(archive_root, cfg)

    summary = {
        "task": "SIGMA-NEWTON",
        "draft_id": DRAFT_ID,
        "setups": summary_setups,
        "underdispersion_ir": {
            "json": under.get("json_path"),
            "summary": under.get("summary"),
        },
        "g60_heterogeneity": {
            "json": g60.get("json_path"),
            "verdict": g60.get("verdict"),
            "pooled_median_chi2": g60.get("pooled_median_chi2"),
        },
        "gate_n4_harness_sanity": n4,
        "git_head": _git_head(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = write_summary_json(summary, out_dir / "sigma_newton_summary.json")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
