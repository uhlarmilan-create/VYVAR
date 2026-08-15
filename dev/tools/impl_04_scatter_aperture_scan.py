#!/usr/bin/env python3
"""IMPL-04: clean scatter-vs-radius rescan (exact masking) + blended set."""
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("impl04")

DRAFT = ROOT / "Archive" / "Drafts" / "draft_000514"
SETUP = "NoFilter_60_2"
ALIGNED = DRAFT / "detrended_aligned" / "lights" / SETUP
PHOT = DRAFT / "platesolve" / SETUP / "photometry"
OUT_DIR = ROOT / "dev" / "results"
OUT_SCAN = OUT_DIR / "IMPL_04_scatter_scan.json"
OUT_TABLE = OUT_DIR / "IMPL_04_aperture_scatter_table.json"

VARIABLE_CIDS = {
    "1498613634033133184",
    "1497343732462852864",
}
SEED = 51403


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


def _load_field_catalog() -> pd.DataFrame:
    """One row per catalog_id from a proc CSV (field positions + measured flux)."""
    csv0 = sorted(ALIGNED.glob("proc_*.csv"))[0]
    df = pd.read_csv(csv0, low_memory=False, dtype={"catalog_id": str})
    df = df.copy()
    df["_nid"] = df["catalog_id"].map(_nid)
    df = df[df["_nid"].astype(str).str.len().gt(0)]
    df = df[~df["_nid"].isin(VARIABLE_CIDS)]
    for c in ("x", "y", "dao_flux", "phot_g_mean_mag", "mag", "contamination_idx"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isfinite(df["x"]) & np.isfinite(df["y"])]
    df["_flux"] = df["dao_flux"] if "dao_flux" in df.columns else np.nan
    df = df[np.isfinite(df["_flux"]) & (df["_flux"] > 0)]
    df = df.drop_duplicates(subset=["_nid"], keep="first")
    df["_nn_px"] = _nn_distance_px(df["x"].to_numpy(), df["y"].to_numpy())
    return df.reset_index(drop=True)


def _pick_pools(field: pd.DataFrame, *, blend_r: float = 4.5, n_each: int = 36) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Isolated vs blended pools matched on measured flux (not catalogue G)."""
    iso = field[field["_nn_px"] >= float(blend_r)].copy()
    bld = field[field["_nn_px"] < float(blend_r)].copy()
    # Match blended to isolated flux distribution: nearest flux partners.
    iso = iso.sort_values("_flux", ascending=False).reset_index(drop=True)
    bld = bld.sort_values("_flux", ascending=False).reset_index(drop=True)
    if iso.empty or bld.empty:
        return iso.head(n_each), bld.head(n_each)
    # Take top-flux isolated and flux-matched blends
    iso_sel = iso.head(int(n_each)).copy()
    targets = iso_sel["_flux"].to_numpy()
    bld_flux = bld["_flux"].to_numpy()
    used = set()
    rows = []
    for tf in targets:
        order = np.argsort(np.abs(bld_flux - float(tf)))
        for j in order:
            j = int(j)
            if j in used:
                continue
            used.add(j)
            rows.append(bld.iloc[j])
            break
        if len(rows) >= int(n_each):
            break
    bld_sel = pd.DataFrame(rows).reset_index(drop=True) if rows else bld.head(0)
    return iso_sel.reset_index(drop=True), bld_sel


def run_ladder_pass(
    pool: pd.DataFrame,
    radii_px: np.ndarray,
    *,
    fwhm_draft: float,
    frame_stride: int = 1,
) -> tuple[dict[float, dict[str, np.ndarray]], dict]:
    ids = pool["_nid"].tolist()
    id_to_i = {cid: i for i, cid in enumerate(ids)}
    csv_files = sorted(ALIGNED.glob("proc_*.csv"))[:: max(1, int(frame_stride))]
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
                flux_acc[float(r)][cid].append(float(arr[i]) if i < len(arr) else float("nan"))
        n_ok += 1
        if (fi + 1) % 20 == 0 or fi + 1 == len(csv_files):
            LOGGER.info("ladder %d/%d (%.1fs)", fi + 1, len(csv_files), time.time() - t0)
    flux_out = {
        float(r): {cid: np.asarray(v, dtype=float) for cid, v in by.items()}
        for r, by in flux_acc.items()
    }
    meta = {
        "n_frames": n_ok,
        "frame_stride": int(frame_stride),
        "annulus_inner_px": ann_in,
        "annulus_outer_px": ann_out,
        "fwhm_per_frame_median": float(np.median(fwhm_per_frame)) if fwhm_per_frame else fwhm_draft,
        "fwhm_per_frame": [round(float(v), 3) for v in fwhm_per_frame],
        "aperture_sum_path": "photometry_core._aperture_flux_sky_per_star method=exact",
    }
    return flux_out, meta


def build_rfwhm_flux(flux_fixed, radii_fixed, fwhm_per_frame, k_values):
    out = {}
    r_arr = np.asarray(radii_fixed, dtype=float)
    n_frames = len(fwhm_per_frame)
    star_ids = next(iter(flux_fixed.values())).keys()
    stack = {cid: np.vstack([flux_fixed[float(r)][cid] for r in r_arr]) for cid in star_ids}
    for k in k_values:
        by_cid = {}
        for cid in star_ids:
            series = np.full(n_frames, np.nan)
            mat = stack[cid]
            for t, fw in enumerate(fwhm_per_frame):
                if t >= mat.shape[1]:
                    break
                series[t] = float(np.interp(float(k) * float(fw), r_arr, mat[:, t]))
            by_cid[cid] = series
        out[float(k)] = by_cid
    return out


def parity_means(radii, scatter):
    r = np.asarray(radii, float)
    s = np.asarray(scatter, float)
    ok = np.isfinite(r) & np.isfinite(s)
    r, s = r[ok], s[ok]
    ii = np.isclose(r % 1.0, 0.0)
    hh = np.isclose(r % 1.0, 0.5)
    return {
        "mean_integer_mmag": float(np.mean(s[ii])) if ii.any() else None,
        "mean_half_integer_mmag": float(np.mean(s[hh])) if hh.any() else None,
        "split_mmag": (
            abs(float(np.mean(s[ii])) - float(np.mean(s[hh])))
            if ii.any() and hh.any()
            else None
        ),
    }


def sampling_noise_mmag(scatter_mmag: float, n_stars: int) -> float:
    """Rough SEM of the median-of-n-stars scatter estimate."""
    if not (math.isfinite(scatter_mmag) and n_stars >= 2):
        return float("nan")
    return float(scatter_mmag) / math.sqrt(float(n_stars))


def decide_radius(curve_sel, curve_hold, radii, ee_r, ee_c, fwhm):
    """Pre-stated rule: genuine min else upper edge of contiguous flat region."""
    r = np.asarray(curve_sel["radii_px"], float)
    s = np.asarray(curve_sel["scatter_mmag"], float)
    n_med = int(np.nanmedian(curve_sel["n_stars"])) if curve_sel.get("n_stars") else 36
    finite = np.isfinite(r) & np.isfinite(s)
    r = r[finite]
    s = s[finite]
    if r.size == 0:
        return {"error": "no_finite_scatter"}
    i_min = int(np.argmin(s))
    r_min = float(r[i_min])
    s_min = float(s[i_min])
    noise = sampling_noise_mmag(s_min, n_med)
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
    sens = None
    if ee_r and ee_c:
        ee = float(np.interp(chosen, np.asarray(ee_r, float), np.asarray(ee_c, float)))
        ee_lo = float(np.interp(chosen - 0.25, np.asarray(ee_r, float), np.asarray(ee_c, float)))
        ee_hi = float(np.interp(chosen + 0.25, np.asarray(ee_r, float), np.asarray(ee_c, float)))
        d_ee_dr = (ee_hi - ee_lo) / 0.5
        sens = {
            "ee_at_chosen": ee,
            "d_ee_dr_per_px": d_ee_dr,
            "d_ee_d_r_over_fwhm": d_ee_dr * float(fwhm),
            "note": (
                "Larger r on a flat scatter band reduces |dEE/dr| exposure to seeing/"
                "centroid error; prefer upper edge when scatter is flat within noise."
            ),
        }
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
        "sensitivity": sens,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snr_path = DRAFT / "aperture_snr_table_zp_corrected.json"
    if not snr_path.is_file():
        snr_path = DRAFT / "aperture_snr_table.json"
    snr = json.loads(snr_path.read_text(encoding="utf-8")) if snr_path.is_file() else {}
    fwhm = float(snr.get("fwhm_px") or 5.195)
    ee_r = snr.get("ee_radii")
    ee_c = snr.get("ee_curve")

    ladder = LadderSpec()
    radii = ladder.radii_px()
    field = _load_field_catalog()
    iso, bld = _pick_pools(field, blend_r=4.5, n_each=36)
    LOGGER.info("pools iso=%d bld=%d field=%d", len(iso), len(bld), len(field))

    # Primary scan on isolated pool (matched to IMPL-03 intent, but not RMS-biased)
    flux_iso, meta = run_ladder_pass(iso, radii, fwhm_draft=fwhm, frame_stride=1)
    ids = iso["_nid"].tolist()
    sel, hold = split_selection_holdout(ids, seed=SEED, selection_frac=0.5)
    comps = ids

    ac_by_r = None
    if ee_r and ee_c:
        ref_r = float(snr.get("ee_ref_r_px") or ee_r[-1])
        ac_by_r = {float(r): ac_delta_m_from_ee(float(r), ee_r, ee_c, ref_r_px=ref_r) for r in radii}

    curves = {}
    for set_name, eids in (("selection", sel), ("held_out", hold)):
        c_off = build_scatter_curve(
            radii, flux_iso, eids, comps, policy="fixed_px", set_name=set_name, ac_delta_m_by_r=None
        )
        curves[f"fixed_px_{set_name}_ac_off"] = c_off.to_dict()
        curves[f"fixed_px_{set_name}_ac_off"]["parity"] = parity_means(
            c_off.radii_px, c_off.scatter_mmag
        )
        if ac_by_r is not None:
            c_on = build_scatter_curve(
                radii, flux_iso, eids, comps, policy="fixed_px", set_name=set_name, ac_delta_m_by_r=ac_by_r
            )
            curves[f"fixed_px_{set_name}_ac_on"] = c_on.to_dict()

    sel_c = curves["fixed_px_selection_ac_off"]
    hold_c = curves["fixed_px_held_out_ac_off"]
    decision = decide_radius(sel_c, hold_c, radii, ee_r, ee_c, fwhm)

    # r/FWHM policy
    k_vals = list(DEFAULT_K_FWHM)
    flux_k = build_rfwhm_flux(flux_iso, radii, meta["fwhm_per_frame"], k_vals)
    rfwhm = {}
    for set_name, eids in (("selection", sel), ("held_out", hold)):
        c = build_scatter_curve(
            k_vals, flux_k, eids, comps, policy="fixed_r_over_fwhm", set_name=set_name
        )
        d = c.to_dict()
        d["k_fwhm"] = d.pop("radii_px")
        d["best_k_fwhm"] = d.pop("best_r_px")
        rfwhm[set_name] = d

    # Policy comparison on held-out at each policy's selection optimum
    hold_fixed_at = None
    r_dec = float(decision["chosen_r_px"])
    r_near = float(radii[int(np.argmin(np.abs(radii - r_dec)))])
    hold_fixed_at, _ = evaluate_scatter_at_radius(flux_iso[r_near], hold, comps)
    k_best = float(rfwhm["selection"].get("best_k_fwhm") or float("nan"))
    hold_rfwhm_sc = rfwhm["held_out"].get("best_scatter_mmag")
    # Evaluate held-out at selection's best k
    if math.isfinite(k_best) and float(k_best) in flux_k:
        hold_rfwhm_at, _ = evaluate_scatter_at_radius(flux_k[float(k_best)], hold, comps)
    else:
        hold_rfwhm_at = hold_rfwhm_sc
    policy_winner = "fixed_px"
    if (
        hold_rfwhm_at is not None
        and math.isfinite(float(hold_rfwhm_at))
        and math.isfinite(float(hold_fixed_at))
        and float(hold_rfwhm_at) < float(hold_fixed_at) * 0.98
    ):
        policy_winner = "fixed_r_over_fwhm"
        decision["chosen_r_px"] = round(k_best * float(meta["fwhm_per_frame_median"]), 3)
        decision["policy_note"] = f"r/FWHM k={k_best} -> px"

    decision["policy_winner"] = policy_winner
    decision["held_out_fixed_px_at_chosen_mmag"] = hold_fixed_at
    decision["held_out_rfwhm_at_sel_opt_mmag"] = hold_rfwhm_at

    # Blended set (Item 2)
    blend_report = {"n_blended_pool": len(bld), "n_isolated_pool": len(iso)}
    if len(bld) >= 8:
        flux_b, meta_b = run_ladder_pass(bld, radii, fwhm_draft=fwhm, frame_stride=1)
        ids_b = bld["_nid"].tolist()
        sel_b, hold_b = split_selection_holdout(ids_b, seed=SEED + 1, selection_frac=0.5)
        c_b = build_scatter_curve(
            radii, flux_b, sel_b, ids_b, policy="fixed_px", set_name="blended_selection"
        )
        c_bh = build_scatter_curve(
            radii, flux_b, hold_b, ids_b, policy="fixed_px", set_name="blended_held_out"
        )
        blend_report["selection"] = c_b.to_dict()
        blend_report["held_out"] = c_bh.to_dict()
        blend_report["decision"] = decide_radius(
            c_b.to_dict(), c_bh.to_dict(), radii, ee_r, ee_c, fwhm
        )
        blend_report["flux_match"] = {
            "iso_median_flux": float(np.median(iso["_flux"])),
            "bld_median_flux": float(np.median(bld["_flux"])),
            "ratio": float(np.median(bld["_flux"]) / np.median(iso["_flux"])),
        }
        blend_report["meta"] = meta_b

    report = {
        "draft": "draft_000514",
        "impl": "IMPL-04",
        "ladder": {
            "r_min_px": ladder.r_min_px,
            "r_max_px": ladder.r_max_px,
            "r_step_px": ladder.r_step_px,
            "why": ladder.why,
            "radii_px": [round(float(r), 3) for r in radii],
        },
        "seed": SEED,
        "n_selection": len(sel),
        "n_held_out": len(hold),
        "selection_ids": sel,
        "held_out_ids": hold,
        "meta": meta,
        "curves": curves,
        "rfwhm_curves": rfwhm,
        "decision": decision,
        "blend_report": blend_report,
        "parity_selection": curves["fixed_px_selection_ac_off"]["parity"],
        "parity_held_out": curves["fixed_px_held_out_ac_off"]["parity"],
        "literature_masking": (
            "photutils aperture_photometry method='exact' (default): fractional "
            "pixel overlap. method='center' is binary and caused IMPL-03 parity "
            "sawtooth. Production and ladder now share _aperture_flux_sky_per_star "
            "with method='exact'."
        ),
    }
    OUT_SCAN.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info(
        "Wrote %s chosen_r=%.3f branch=%s policy=%s parity_split=%.3f",
        OUT_SCAN,
        float(decision["chosen_r_px"]),
        decision.get("branch"),
        policy_winner,
        float(report["parity_selection"]["split_mmag"] or -1),
    )

    table = flat_aperture_table_from_radius(
        float(decision["chosen_r_px"]),
        fwhm_px=fwhm,
        meta={
            "selection_criterion": "scatter",
            "impl04_decision": decision,
            "ee_radii": ee_r,
            "ee_curve": ee_c,
            "policy": policy_winner,
        },
    )
    OUT_TABLE.write_text(json.dumps(table, indent=2), encoding="utf-8")
    (DRAFT / "aperture_scatter_table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
