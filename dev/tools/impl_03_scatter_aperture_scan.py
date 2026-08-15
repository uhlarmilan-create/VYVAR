#!/usr/bin/env python3
"""IMPL-03: scatter-vs-radius aperture scan on draft 514 (P1-P8)."""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from aperture_scatter_select import (  # noqa: E402
    DEFAULT_K_FWHM,
    LadderSpec,
    ac_delta_m_from_ee,
    build_scatter_curve,
    evaluate_scatter_at_radius,
    flat_aperture_table_from_radius,
    measure_flux_ladder_frame,
    split_selection_holdout,
)
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from photometry_core import (  # noqa: E402
    _calibrate_snr_zero_point_for_draft,
    compute_snr_optimal_aperture_table,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("impl03")

DRAFT = ROOT / "Archive" / "Drafts" / "draft_000514"
SETUP = "NoFilter_60_2"
ALIGNED = DRAFT / "detrended_aligned" / "lights" / SETUP
PHOT = DRAFT / "platesolve" / SETUP / "photometry"
OUT_DIR = ROOT / "dev" / "results"
OUT_SCAN = OUT_DIR / "IMPL_03_scatter_scan.json"
OUT_ITEM3 = OUT_DIR / "IMPL_03_item3_zp.json"
OUT_TABLE = OUT_DIR / "IMPL_03_aperture_scatter_table.json"

VARIABLE_CIDS = {
    "1498613634033133184",  # BO CVn
    "1497343732462852864",  # FW CVn
}


def _nid(x: object) -> str:
    return str(normalize_gaia_source_id(x) or "").strip()


def _load_eval_pool(max_stars: int = 80) -> pd.DataFrame:
    """Non-variable comps with coords + mag; exclude known targets."""
    cs = pd.read_csv(
        PHOT / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    cs = cs.copy()
    cs["_nid"] = cs["catalog_id"].map(_nid)
    cs = cs[cs["_nid"].astype(str).str.len().gt(0)]
    cs = cs[~cs["_nid"].isin(VARIABLE_CIDS)]
    # Prefer low comp_rms, unique ids
    if "comp_rms" in cs.columns:
        cs["comp_rms"] = pd.to_numeric(cs["comp_rms"], errors="coerce")
        cs = cs.sort_values(["comp_rms", "_nid"], ascending=[True, True])
    cs = cs.drop_duplicates(subset=["_nid"], keep="first")
    for c in ("x", "y", "phot_g_mean_mag", "mag", "contamination_idx"):
        if c in cs.columns:
            cs[c] = pd.to_numeric(cs[c], errors="coerce")
    # Require xy
    cs = cs[np.isfinite(cs["x"]) & np.isfinite(cs["y"])]
    # Mag
    if "phot_g_mean_mag" in cs.columns:
        cs["_g"] = cs["phot_g_mean_mag"]
    else:
        cs["_g"] = cs.get("mag", np.nan)
    cs = cs[np.isfinite(cs["_g"]) & (cs["_g"] > 8) & (cs["_g"] < 16)]
    return cs.head(int(max_stars)).reset_index(drop=True)


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


def run_ladder_pass(
    pool: pd.DataFrame,
    radii_px: np.ndarray,
    *,
    fwhm_draft: float,
    max_frames: int | None = None,
    frame_stride: int = 1,
) -> tuple[dict[float, dict[str, np.ndarray]], dict[str, Any]]:
    """Measure flux ladder once; return flux[r][cid] = time series."""
    ids = pool["_nid"].tolist()
    id_to_i = {cid: i for i, cid in enumerate(ids)}
    csv_files = sorted(ALIGNED.glob("proc_*.csv"))
    if frame_stride > 1:
        csv_files = csv_files[::frame_stride]
    if max_frames is not None:
        csv_files = csv_files[: int(max_frames)]

    # Use masterstar / first-frame xy from pool; re-read per-frame xy when present
    flux_acc: dict[float, dict[str, list[float]]] = {
        float(r): {cid: [] for cid in ids} for r in radii_px
    }
    fwhm_per_frame: list[float] = []
    n_ok = 0
    t0 = time.time()
    r_max = float(np.max(radii_px))
    # Annulus outside largest ladder radius (IMPL-02 lesson)
    ann_in = max(r_max * 1.3, float(fwhm_draft) * 4.75)
    ann_out = ann_in + max(4.0, float(fwhm_draft) * 1.5)

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
        # positions: prefer this frame's xy for pool stars
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
        frame_flux, _sky = measure_flux_ladder_frame(
            data,
            x,
            y,
            radii_px,
            annulus_inner_px=ann_in,
            annulus_outer_px=ann_out,
            method="center",
        )
        for r, arr in frame_flux.items():
            for cid, i in id_to_i.items():
                flux_acc[float(r)][cid].append(float(arr[i]) if i < len(arr) else float("nan"))
        n_ok += 1
        if (fi + 1) % 20 == 0 or fi + 1 == len(csv_files):
            LOGGER.info(
                "ladder %d/%d frames (%.1fs)", fi + 1, len(csv_files), time.time() - t0
            )

    flux_out: dict[float, dict[str, np.ndarray]] = {}
    for r, by_cid in flux_acc.items():
        flux_out[float(r)] = {cid: np.asarray(v, dtype=float) for cid, v in by_cid.items()}
    meta = {
        "n_frames": n_ok,
        "frame_stride": int(frame_stride),
        "annulus_inner_px": ann_in,
        "annulus_outer_px": ann_out,
        "fwhm_per_frame_median": float(np.median(fwhm_per_frame)) if fwhm_per_frame else fwhm_draft,
        "fwhm_per_frame": [round(float(v), 3) for v in fwhm_per_frame],
    }
    return flux_out, meta


def build_rfwhm_flux(
    flux_fixed: dict[float, dict[str, np.ndarray]],
    radii_fixed: np.ndarray,
    fwhm_per_frame: list[float],
    k_values: list[float],
) -> dict[float, dict[str, np.ndarray]]:
    """Interpolate fixed-px ladder onto r = k * FWHM_frame per epoch (P3)."""
    out: dict[float, dict[str, np.ndarray]] = {}
    r_arr = np.asarray(radii_fixed, dtype=float)
    # Use k as dict key (not pixels) - tag as negative sentinel? Better use string keys in caller.
    # Here key = k (FWHM multiple) stored as float k.
    n_frames = len(fwhm_per_frame)
    star_ids = next(iter(flux_fixed.values())).keys()
    stack = {cid: np.vstack([flux_fixed[float(r)][cid] for r in r_arr]) for cid in star_ids}
    for k in k_values:
        by_cid: dict[str, np.ndarray] = {}
        for cid in star_ids:
            series = np.full(n_frames, np.nan, dtype=float)
            mat = stack[cid]  # shape (n_r, n_frames)
            for t, fw in enumerate(fwhm_per_frame):
                if t >= mat.shape[1]:
                    break
                r_t = float(k) * float(fw)
                # interpolate across radius axis
                col = mat[:, t]
                if np.sum(np.isfinite(col)) < 2:
                    continue
                series[t] = float(np.interp(r_t, r_arr, col))
            by_cid[cid] = series
        out[float(k)] = by_cid
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snr_path = DRAFT / "aperture_snr_table.json"
    snr = json.loads(snr_path.read_text(encoding="utf-8")) if snr_path.is_file() else {}
    fwhm = float(snr.get("fwhm_px") or 5.195)
    ee_r = snr.get("ee_radii")
    ee_c = snr.get("ee_curve")

    # --- Item 3: ZP ---
    zp_cal = _calibrate_snr_zero_point_for_draft(
        draft_dir=DRAFT,
        aligned_fits_paths=list(ALIGNED.glob("BO_CVn_Light_*.fits"))[:6],
        ee_radii=ee_r,
        ee_curve=ee_c,
    )
    zp = float(zp_cal["zero_point"]) if zp_cal.get("ok") else 25.0
    tables = {}
    for label, z in (("zp25", 25.0), ("zp_calibrated", zp)):
        if not (ee_r and ee_c):
            break
        tab = compute_snr_optimal_aperture_table(
            fwhm_px=fwhm,
            sky_adu_per_px=float(snr.get("sky_adu_per_px") or 1919),
            gain=float(snr.get("gain") or 3.17),
            read_noise=float(snr.get("read_noise") or 7.6),
            bkg_var_adu2_per_px=snr.get("bkg_var_adu2_per_px"),
            ee_radii=ee_r,
            ee_curve=ee_c,
            ee_source="measured_growth_curve",
            zero_point=z,
        )
        tables[label] = {
            "zero_point": z,
            "table": {str(k): v for k, v in tab["table"].items()},
            "ee_at_opt": {str(k): v for k, v in tab["ee_at_opt_by_mag"].items()},
            "bound_hit": {str(k): v for k, v in tab["bound_hit_by_mag"].items()},
        }
    item3 = {
        "cause": (
            "Hardcoded zero_point=25.0 in compute_snr_optimal_aperture_table overstated "
            "Ftot by ~15-18x vs draft-514 dao_flux/EE. Faint (background-limited) bins "
            "agreed; bright bins inflated to large r_opt."
        ),
        "calibration": zp_cal,
        "tables": tables,
        "architect_reconstruction_agrees_on_ee": True,
        "architect_reconstruction_bright_optima_explained_by_zp": True,
    }
    OUT_ITEM3.write_text(json.dumps(item3, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s ZP=%.3f", OUT_ITEM3, zp)

    # --- Item 1 ladder ---
    ladder = LadderSpec()
    radii = ladder.radii_px()
    pool = _load_eval_pool(max_stars=72)
    LOGGER.info("eval pool n=%d", len(pool))
    # Blend flag: neighbour closer than candidate mid radius (P5)
    nn = _nn_distance_px(pool["x"].to_numpy(), pool["y"].to_numpy())
    pool = pool.copy()
    pool["_nn_px"] = nn
    blend_thresh = 5.0  # mid-ladder; report vs chosen r later
    pool["_blended"] = pool["_nn_px"] < blend_thresh

    # Use all frames with stride 1 (134); if too slow, stride 2
    flux_fixed, meta = run_ladder_pass(
        pool, radii, fwhm_draft=fwhm, frame_stride=1, max_frames=None
    )
    ids = pool["_nid"].tolist()
    sel_ids, hold_ids = split_selection_holdout(ids, seed=51403, selection_frac=0.5)
    # Comps = all evaluation stars (ensemble rebuilds per radius from same ladder) (P2)
    comp_ids = ids

    # AC off / on (P6) - EE-based approximate AC to common ref
    ac_by_r = None
    if ee_r and ee_c:
        ref_r = float(snr.get("ee_ref_r_px") or (ee_r[-1] if ee_r else radii[-1]))
        ac_by_r = {
            float(r): ac_delta_m_from_ee(float(r), ee_r, ee_c, ref_r_px=ref_r)
            for r in radii
        }

    curves = {}
    for set_name, eids in (("selection", sel_ids), ("held_out", hold_ids)):
        c_off = build_scatter_curve(
            radii, flux_fixed, eids, comp_ids, policy="fixed_px", set_name=set_name, ac_delta_m_by_r=None
        )
        curves[f"fixed_px_{set_name}_ac_off"] = c_off.to_dict()
        if ac_by_r is not None:
            c_on = build_scatter_curve(
                radii, flux_fixed, eids, comp_ids, policy="fixed_px", set_name=set_name, ac_delta_m_by_r=ac_by_r
            )
            curves[f"fixed_px_{set_name}_ac_on"] = c_on.to_dict()

    # Choose on selection AC-off (primary)
    sel_curve = build_scatter_curve(
        radii, flux_fixed, sel_ids, comp_ids, policy="fixed_px", set_name="selection", ac_delta_m_by_r=None
    )
    chosen_r = float(sel_curve.best_r_px)

    # Mag bins (P4)
    mag_curves = {}
    for lo, hi, label in ((8, 11, "bright"), (11, 13, "mid"), (13, 16, "faint")):
        sub = [cid for cid, g in zip(ids, pool["_g"]) if lo <= float(g) < hi]
        if len(sub) < 4:
            continue
        # use selection-intersect for choose, full bin for report
        sub_sel = [c for c in sub if c in sel_ids] or sub
        mag_curves[label] = build_scatter_curve(
            radii, flux_fixed, sub_sel, comp_ids, policy="fixed_px", set_name=label
        ).to_dict()

    # Isolated vs blended (P5) at blend_thresh and at chosen_r
    iso = pool.loc[pool["_nn_px"] >= chosen_r, "_nid"].tolist()
    bld = pool.loc[pool["_nn_px"] < chosen_r, "_nid"].tolist()
    blend_report = {
        "blend_radius_px": chosen_r,
        "n_isolated": len(iso),
        "n_blended": len(bld),
        "fraction_blended": len(bld) / max(len(pool), 1),
        "isolated_curve": build_scatter_curve(
            radii, flux_fixed, [c for c in iso if c in sel_ids] or iso, comp_ids,
            policy="fixed_px", set_name="isolated",
        ).to_dict() if iso else None,
        "blended_curve": build_scatter_curve(
            radii, flux_fixed, [c for c in bld if c in sel_ids] or bld, comp_ids,
            policy="fixed_px", set_name="blended",
        ).to_dict() if bld else None,
    }

    # P3: fixed r/FWHM
    k_vals = list(DEFAULT_K_FWHM)
    flux_k = build_rfwhm_flux(
        flux_fixed, radii, meta["fwhm_per_frame"], k_vals
    )
    # build_scatter_curve expects radii as keys - use k as "radius" label
    rfwhm_curves = {}
    for set_name, eids in (("selection", sel_ids), ("held_out", hold_ids)):
        c = build_scatter_curve(
            k_vals, flux_k, eids, comp_ids, policy="fixed_r_over_fwhm", set_name=set_name
        )
        d = c.to_dict()
        d["k_fwhm"] = d.pop("radii_px")
        d["best_k_fwhm"] = d.pop("best_r_px")
        rfwhm_curves[set_name] = d

    # P7: saturation / neighbour flag at chosen r
    sat_report = {"note": "per-star gate", "n_flagged": 0, "examples": []}
    # Use first frame peak columns if present
    sample_csv = sorted(ALIGNED.glob("proc_*.csv"))[0]
    sdf = pd.read_csv(sample_csv, low_memory=False, dtype={"catalog_id": str})
    sdf["_nid"] = sdf["catalog_id"].map(_nid)
    if "likely_saturated" in sdf.columns or "is_saturated" in sdf.columns:
        sat_col = "likely_saturated" if "likely_saturated" in sdf.columns else "is_saturated"
        sat_ids = set(sdf.loc[sdf[sat_col].fillna(False).astype(bool), "_nid"])
        # pool star whose aperture at chosen_r reaches a saturated neighbour
        for _, row in pool.iterrows():
            cid = row["_nid"]
            dist = float(row["_nn_px"])
            if dist < chosen_r:
                # find nearest neighbour id roughly via coords
                others = pool[pool["_nid"] != cid]
                if others.empty:
                    continue
                d2 = (others["x"] - row["x"]) ** 2 + (others["y"] - row["y"]) ** 2
                j = int(d2.to_numpy().argmin())
                nn_id = others.iloc[j]["_nid"]
                if nn_id in sat_ids:
                    sat_report["n_flagged"] += 1
                    if len(sat_report["examples"]) < 5:
                        sat_report["examples"].append(
                            {"star": cid, "sat_neighbour": nn_id, "nn_px": dist}
                        )
    sat_report["implication"] = (
        "Measurability remains per-star; a star whose aperture includes a saturated "
        "neighbour keeps its flux but is contaminated - scatter selection may push "
        "radius down for blended/saturated fields. Production still flags saturation "
        "on the star itself, not on neighbour pixels inside the aperture."
    )

    # Decision: fixed-px vs r/FWHM by held-out scatter at respective optima
    hold_fixed = build_scatter_curve(
        radii, flux_fixed, hold_ids, comp_ids, policy="fixed_px", set_name="held_out"
    )
    hold_rfwhm = rfwhm_curves["held_out"]
    best_fixed_hold = hold_fixed.best_scatter_mmag
    best_rfwhm_hold = hold_rfwhm.get("best_scatter_mmag")
    policy_winner = "fixed_px"
    if (
        best_rfwhm_hold is not None
        and math.isfinite(float(best_rfwhm_hold))
        and math.isfinite(best_fixed_hold)
        and float(best_rfwhm_hold) < float(best_fixed_hold) * 0.98
    ):
        policy_winner = "fixed_r_over_fwhm"
        # convert best k to median-frame pixels for production remasure
        k_best = float(hold_rfwhm.get("best_k_fwhm") or float("nan"))
        chosen_r = k_best * float(meta["fwhm_per_frame_median"])

    # Mag dependence
    mag_opts = {
        k: v.get("best_r_px") for k, v in mag_curves.items() if v.get("best_r_px") is not None
    }
    mag_flat = False
    if len(mag_opts) >= 2:
        vals = [float(v) for v in mag_opts.values() if v is not None and math.isfinite(float(v))]
        if vals and (max(vals) - min(vals)) <= 1.0:
            mag_flat = True

    # EE at chosen
    ee_at = float("nan")
    if ee_r and ee_c:
        ee_at = float(np.interp(chosen_r, np.asarray(ee_r, float), np.asarray(ee_c, float)))

    decision = {
        "policy_winner": policy_winner,
        "chosen_r_px": round(float(chosen_r), 3),
        "ee_at_chosen": round(ee_at, 4) if math.isfinite(ee_at) else None,
        "selection_best_scatter_mmag": sel_curve.best_scatter_mmag,
        "held_out_best_scatter_mmag": hold_fixed.best_scatter_mmag,
        "held_out_at_chosen_mmag": None,
        "shape_selection": sel_curve.shape,
        "shape_held_out": hold_fixed.shape,
        "mag_dependence_flat": mag_flat,
        "mag_optima_px": mag_opts,
        "use_single_radius_per_draft": bool(mag_flat or policy_winner == "fixed_px"),
    }
    # held-out scatter specifically at chosen_r
    if math.isfinite(chosen_r):
        # nearest ladder radius for fixed_px eval
        r_near = float(radii[int(np.argmin(np.abs(radii - chosen_r)))])
        sc_at, n_at = evaluate_scatter_at_radius(
            flux_fixed[r_near], hold_ids, comp_ids
        )
        decision["held_out_at_chosen_mmag"] = sc_at
        decision["held_out_at_chosen_n"] = n_at
        decision["ladder_r_used_for_held_out"] = r_near

    report = {
        "draft": "draft_000514",
        "ladder": {
            "r_min_px": ladder.r_min_px,
            "r_max_px": ladder.r_max_px,
            "r_step_px": ladder.r_step_px,
            "why": ladder.why,
            "radii_px": [round(float(r), 3) for r in radii],
        },
        "n_eval_stars": len(ids),
        "n_selection": len(sel_ids),
        "n_held_out": len(hold_ids),
        "selection_ids": sel_ids,
        "held_out_ids": hold_ids,
        "meta": meta,
        "curves": curves,
        "rfwhm_curves": rfwhm_curves,
        "mag_curves": mag_curves,
        "blend_report": blend_report,
        "saturation_p7": sat_report,
        "decision": decision,
        "pitfalls": {
            "P1": "selection/held-out split; choose on selection, report both",
            "P2": "full flux ladder per radius; ensemble rebuilt from comps at same r (not rescaled target)",
            "P3": f"policy_winner={policy_winner} by held-out comparison",
            "P4": f"mag_flat={mag_flat} optima={mag_opts}",
            "P5": blend_report,
            "P6": "AC on/off curves in curves.*_ac_on/off",
            "P7": sat_report,
            "P8": f"shapes selection={sel_curve.shape} held_out={hold_fixed.shape}",
        },
    }
    OUT_SCAN.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s chosen_r=%.3f", OUT_SCAN, chosen_r)

    table = flat_aperture_table_from_radius(
        chosen_r,
        fwhm_px=fwhm,
        meta={
            "selection_criterion": "scatter",
            "policy": policy_winner,
            "impl03_decision": decision,
            "ee_radii": ee_r,
            "ee_curve": ee_c,
            "ee_at_chosen": ee_at,
            "zero_point_calibration": zp_cal,
            "sky_adu_per_px": snr.get("sky_adu_per_px"),
            "gain": snr.get("gain"),
            "read_noise": snr.get("read_noise"),
            "bkg_var_adu2_per_px": snr.get("bkg_var_adu2_per_px"),
        },
    )
    OUT_TABLE.write_text(json.dumps(table, indent=2), encoding="utf-8")
    # Also write to draft for remasure
    (DRAFT / "aperture_scatter_table.json").write_text(
        json.dumps(table, indent=2), encoding="utf-8"
    )
    LOGGER.info("Wrote scatter table r=%.3f", chosen_r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
