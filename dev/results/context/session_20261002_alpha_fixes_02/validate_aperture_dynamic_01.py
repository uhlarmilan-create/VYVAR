# -*- coding: ascii -*-
"""APERTURE-DYNAMIC-01 validation on drafts 516 and 521.

Measures f-grid on aligned FITS (r = f x FWHM_frame), picks f* per target
(Abbe p2p), reports:
  - f* vs magnitude / G bin; aperture_f_edge counts
  - p2p gain vs f_fixed=1.35 by mag bin
  - D5-1: residual(f*) vs seeing correlation
  - AIJ gate on BO (516) at f* with per-frame r = f* x FWHM_frame
  - grid runtime
Does NOT rewrite live draft LCs (validation-only measure).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))

from aperture_pertarget import (  # noqa: E402
    DEFAULT_APERTURE_F_GRID,
    choose_f_for_target,
    equal_weight_ensemble_delta,
    flux_to_mag,
    measure_night_grid,
    series_from_grid,
    write_per_target_choices,
    _aperture_flux_uniform,
)
from aperture_policy import resolve_aperture_geometry  # noqa: E402

OUT = ROOT / "dev" / "results" / "context" / "session_20261002_alpha_fixes_02"
SETUP = "NoFilter_60_2"
F_FIXED = 1.35
F_REF = 2.5
ANN_IN = 2.7
ANN_OUT = 5.2
BO_CID = "1498613634033133184"
AIJ_CSV = ROOT / "dev" / "results" / "XVAL_AIJ_01_bo_compare.csv"
AIJ_TBL = ROOT / "dev" / "results" / "XVAL_AIJ_01_Table.tbl"
GATE_MMAG = 2.8


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def frame_key(s: str) -> str:
    stem = Path(str(s)).stem
    if stem.startswith("proc_"):
        stem = stem[5:]
    return stem.lower()


def sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    dra = (ra1 - ra2) * math.cos(math.radians(0.5 * (dec1 + dec2)))
    return math.hypot(dra, dec1 - dec2) * 3600.0


def night_fwhm_from_fits(frames_dir: Path) -> float:
    vals = []
    for fp in sorted(frames_dir.glob("*.fits")):
        if fp.stem.upper() == "MASTERSTAR":
            continue
        try:
            h = fits.getheader(fp)
            v = h.get("VY_FWHM")
            if v is not None and math.isfinite(float(v)):
                vals.append(float(v))
        except Exception:  # noqa: BLE001
            continue
    if not vals:
        return 5.0
    return float(np.median(vals))


def per_frame_fwhm(frames_dir: Path, order: list[str]) -> np.ndarray:
    out = np.full(len(order), float("nan"), dtype=float)
    for i, stem in enumerate(order):
        fp = frames_dir / f"{stem}.fits"
        if not fp.is_file():
            continue
        try:
            v = fits.getheader(fp).get("VY_FWHM")
            if v is not None and math.isfinite(float(v)):
                out[i] = float(v)
        except Exception:  # noqa: BLE001
            continue
    return out


def read_aij_ensemble(tbl: Path) -> dict:
    with tbl.open(encoding="utf-8", errors="replace", newline="") as f:
        header = f.readline().rstrip("\n").split("\t")
        row = f.readline().rstrip("\n").split("\t")
    idx = {h: i for i, h in enumerate(header)}
    stars = {}
    for role in ("T1", "C2", "C3", "C4", "C5", "C6"):
        rk, dk = f"RA_{role}", f"DEC_{role}"
        ra_h = float(row[idx[rk]])
        dec = float(row[idx[dk]])
        stars[role] = {"ra_deg": ra_h * 15.0, "dec_deg": dec}
    return {
        "source_radius_px": float(row[idx["Source_Radius"]]) if "Source_Radius" in idx else None,
        "stars": stars,
    }


def match_role(ms: pd.DataFrame, ra: float, dec: float) -> str | None:
    best = None
    for _, row in ms.iterrows():
        try:
            s = sep_arcsec(ra, dec, float(row["ra_deg"]), float(row["dec_deg"]))
        except (TypeError, ValueError, KeyError):
            continue
        cid = str(row["catalog_id"]).strip()
        if best is None or s < best[0]:
            best = (s, cid)
    if best is None or best[0] > 8.0:
        return None
    return best[1]


def mag_bin_label(g: float) -> str:
    if not math.isfinite(g):
        return "unknown"
    if g < 12:
        return "G<12"
    if g < 14:
        return "12-14"
    if g < 16:
        return "14-16"
    return "G>=16"


def validate_draft(draft_id: int) -> dict:
    draft = ROOT / "Archive" / "Drafts" / f"draft_{draft_id:06d}"
    frames_dir = draft / "detrended_aligned" / "lights" / SETUP
    photo = draft / "platesolve" / SETUP / "photometry"
    at = pd.read_csv(photo / "active_targets.csv", dtype={"catalog_id": str}, low_memory=False)
    comps = pd.read_csv(
        photo / "comparison_stars_per_target.csv",
        dtype={"catalog_id": str, "target_catalog_id": str},
        low_memory=False,
    )
    mag_map = {
        str(r["catalog_id"]): float(
            pd.to_numeric(r.get("phot_g_mean_mag", r.get("mag")), errors="coerce")
        )
        for _, r in at.iterrows()
    }
    if "phot_g_mean_mag" in comps.columns:
        for _, r in comps.iterrows():
            cid = str(r["catalog_id"])
            g = float(pd.to_numeric(r["phot_g_mean_mag"], errors="coerce"))
            if math.isfinite(g):
                mag_map[cid] = g

    targets = sorted(set(comps["target_catalog_id"].astype(str)))
    star_ids = set(targets) | set(comps["catalog_id"].astype(str))
    fwhm = night_fwhm_from_fits(frames_dir)
    policy_path = photo / "aperture_policy.json"
    if policy_path.is_file():
        try:
            pol = json.loads(policy_path.read_text(encoding="utf-8"))
            if pol.get("fwhm_night_median_px"):
                fwhm = float(pol["fwhm_night_median_px"])
        except Exception:  # noqa: BLE001
            pass

    t0 = time.perf_counter()
    grid = measure_night_grid(
        frames_dir=frames_dir,
        catalog_ids=sorted(star_ids),
        f_grid=list(DEFAULT_APERTURE_F_GRID),
        fwhm_night_px=fwhm,
        annulus_inner_fwhm=ANN_IN,
        annulus_outer_fwhm=ANN_OUT,
    )
    grid_s = float(time.perf_counter() - t0)
    order = sorted(grid.frames.keys())
    fwhm_arr = per_frame_fwhm(frames_dir, order)

    choices = {}
    rows = []
    for tid in targets:
        cids = (
            comps.loc[comps["target_catalog_id"].astype(str) == tid, "catalog_id"]
            .astype(str)
            .tolist()
        )
        ch = choose_f_for_target(grid, target_cid=tid, comp_ids=cids, frame_order=order)
        choices[tid] = ch
        p2p_fixed = float(ch.p2p_by_f.get(f"{F_FIXED:.4g}", float("nan")))
        if not math.isfinite(p2p_fixed):
            for k, v in ch.p2p_by_f.items():
                if abs(float(k) - F_FIXED) < 1e-9:
                    p2p_fixed = float(v)
                    break
        p2p_star = float(ch.p2p_by_f.get(f"{ch.f_star:.4g}", float("nan")))
        if not math.isfinite(p2p_star):
            for k, v in ch.p2p_by_f.items():
                if abs(float(k) - float(ch.f_star)) < 1e-9:
                    p2p_star = float(v)
                    break
        t_mag_star = flux_to_mag(series_from_grid(grid, tid, ch.f_star, order))
        c_mags_star = {
            c: flux_to_mag(series_from_grid(grid, c, ch.f_star, order)) for c in cids
        }
        delta_star = equal_weight_ensemble_delta(t_mag_star, c_mags_star)
        t_mag_ref = flux_to_mag(series_from_grid(grid, tid, F_REF, order))
        c_mags_ref = {
            c: flux_to_mag(series_from_grid(grid, c, F_REF, order)) for c in cids
        }
        delta_ref = equal_weight_ensemble_delta(t_mag_ref, c_mags_ref)
        ok = np.isfinite(delta_star) & np.isfinite(delta_ref)
        if int(ok.sum()) >= 8:
            dlevel_mmag = float(np.median(delta_star[ok] - delta_ref[ok]) * 1000.0)
        else:
            dlevel_mmag = float("nan")
        resid = delta_star - (
            np.nanmedian(delta_star) if np.isfinite(delta_star).any() else 0.0
        )
        ok2 = np.isfinite(resid) & np.isfinite(fwhm_arr)
        if int(ok2.sum()) >= 12 and float(np.std(fwhm_arr[ok2])) > 1e-6:
            rho = float(np.corrcoef(resid[ok2], fwhm_arr[ok2])[0, 1])
        else:
            rho = float("nan")
        gmag = float(mag_map.get(tid, float("nan")))
        rows.append(
            {
                "catalog_id": tid,
                "g_mag": gmag if math.isfinite(gmag) else None,
                "mag_bin": mag_bin_label(gmag),
                "f_star": ch.f_star,
                "aperture_f_edge": bool(ch.f_edge),
                "r_ap_px_night_scale": ch.r_ap_px,
                "p2p_fixed_mag": p2p_fixed,
                "p2p_star_mag": p2p_star,
                "p2p_gain_mmag": (
                    (p2p_fixed - p2p_star) * 1000.0
                    if math.isfinite(p2p_fixed) and math.isfinite(p2p_star)
                    else None
                ),
                "dlevel_vs_f25_mmag": dlevel_mmag if math.isfinite(dlevel_mmag) else None,
                "rho_resid_seeing": rho if math.isfinite(rho) else None,
                "n_comps": ch.n_comps,
                "n_frames": ch.n_frames,
                "same_r_target_comps": True,
            }
        )

    write_per_target_choices(
        OUT / f"aperture_per_target_draft{draft_id}.json",
        choices,
        f_grid=grid.f_grid,
        fwhm_night_px=fwhm,
        elapsed_s=grid_s,
    )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"fstar_table_draft{draft_id}.csv", index=False)

    bin_summary = []
    for b in ["G<12", "12-14", "14-16", "G>=16", "unknown"]:
        sub = df[df["mag_bin"] == b]
        if sub.empty:
            continue
        gains = pd.to_numeric(sub["p2p_gain_mmag"], errors="coerce")
        bin_summary.append(
            {
                "mag_bin": b,
                "n": int(len(sub)),
                "median_f_star": float(np.nanmedian(sub["f_star"])),
                "median_p2p_gain_mmag": float(np.nanmedian(gains)),
                "frac_gain_gt_0": float(np.nanmean(gains > 0)),
                "n_f_edge": int(sub["aperture_f_edge"].astype(bool).sum()),
            }
        )

    f_hist = (
        df["f_star"].round(4).value_counts().sort_index().to_dict()
        if len(df)
        else {}
    )
    f_hist = {str(k): int(v) for k, v in f_hist.items()}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    g = pd.to_numeric(df["g_mag"], errors="coerce").to_numpy()
    fs = pd.to_numeric(df["f_star"], errors="coerce").to_numpy()
    ok = np.isfinite(g) & np.isfinite(fs)
    ax.scatter(g[ok], fs[ok], s=18, alpha=0.75, c="#1f4e79")
    ax.axhline(F_FIXED, color="#c44", ls="--", lw=1, label=f"f_fixed={F_FIXED}")
    ax.set_xlabel("Gaia G (mag)")
    ax.set_ylabel("f*")
    ax.set_title(f"draft {draft_id}: f* vs magnitude (APERTURE-DYNAMIC-01)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"fstar_vs_mag_draft{draft_id}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    gain = pd.to_numeric(df["p2p_gain_mmag"], errors="coerce").to_numpy()
    ok = np.isfinite(g) & np.isfinite(gain)
    ax.scatter(g[ok], gain[ok], s=18, alpha=0.75, c="#2a7")
    ax.axhline(0, color="#333", ls="-", lw=0.8)
    ax.set_xlabel("Gaia G (mag)")
    ax.set_ylabel("p2p gain (mmag) = fixed - per_target")
    ax.set_title(f"draft {draft_id}: Abbe p2p gain vs magnitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"p2p_gain_vs_mag_draft{draft_id}.png", dpi=120)
    plt.close(fig)

    rhos = pd.to_numeric(df["rho_resid_seeing"], errors="coerce")
    n_edge = int(df["aperture_f_edge"].astype(bool).sum()) if len(df) else 0
    return {
        "draft_id": draft_id,
        "fwhm_night_px": round(fwhm, 6),
        "n_targets": len(targets),
        "n_stars_measured": len(star_ids),
        "n_frames": grid.n_frames_measured,
        "n_fwhm_fallback_night": int(grid.n_fwhm_fallback_night),
        "grid_elapsed_s": round(grid_s, 2),
        "f_fixed": F_FIXED,
        "n_f_edge": n_edge,
        "f_star_hist": f_hist,
        "median_f_star": float(np.nanmedian(df["f_star"])) if len(df) else None,
        "median_p2p_gain_mmag": float(
            np.nanmedian(pd.to_numeric(df["p2p_gain_mmag"], errors="coerce"))
        ),
        "median_abs_rho_resid_seeing": float(np.nanmedian(np.abs(rhos))),
        "frac_abs_rho_lt_0_3": float(np.nanmean(np.abs(rhos) < 0.3)),
        "bin_summary": bin_summary,
        "bo_row": df[df["catalog_id"] == BO_CID].to_dict(orient="records")[0]
        if BO_CID in set(df["catalog_id"])
        else None,
        "choices_path": str(OUT / f"aperture_per_target_draft{draft_id}.json"),
        "table_path": str(OUT / f"fstar_table_draft{draft_id}.csv"),
    }


def aij_gate_dynamic(draft_id: int, f_star: float, fwhm_night: float) -> dict:
    """AIJ RMS(diff) at f* with per-frame r = f* x FWHM_frame (DYNAMIC default)."""
    draft = ROOT / "Archive" / "Drafts" / f"draft_{draft_id:06d}"
    ms_path = draft / "platesolve" / SETUP / "masterstars_full_match.csv"
    if not ms_path.is_file():
        ms_path = (
            ROOT
            / "Archive"
            / "Drafts"
            / "draft_000516_snapshot_era04_20260826"
            / "platesolve"
            / SETUP
            / "masterstars_full_match.csv"
        )
    frames_dir = draft / "detrended_aligned" / "lights" / SETUP
    ens = read_aij_ensemble(AIJ_TBL)
    ms = pd.read_csv(ms_path, dtype={"catalog_id": str}, low_memory=False)
    matched = {}
    for role, rec in ens["stars"].items():
        matched[role] = match_role(ms, rec["ra_deg"], rec["dec_deg"])
    if any(v is None for v in matched.values()):
        return {"ok": False, "reason": "match_failed", "matched": matched}
    ids = [matched["T1"]] + [matched[f"C{i}"] for i in range(2, 7)]
    proc0 = next(frames_dir.glob("proc_*.csv"))
    pdf = pd.read_csv(proc0, dtype={"catalog_id": str}, low_memory=False)
    xy = []
    for cid in ids:
        row = pdf[pdf["catalog_id"].astype(str) == cid]
        if row.empty:
            return {"ok": False, "reason": f"xy_missing_{cid}"}
        xy.append((float(row.iloc[0]["x"]), float(row.iloc[0]["y"])))
    pos = np.asarray(xy, dtype=float)

    lights = sorted(
        p for p in frames_dir.glob("*.fits") if p.stem.upper() != "MASTERSTAR"
    )
    flux = {cid: [] for cid in ids}
    keys = []
    r_list = []
    n_fb = 0
    t0 = time.perf_counter()
    for fp in lights:
        with fits.open(fp, memmap=True) as hdul:
            data = np.asarray(hdul[0].data, dtype=float)
            raw = hdul[0].header.get("VY_FWHM")
            try:
                fw = float(raw) if raw is not None else float("nan")
            except (TypeError, ValueError):
                fw = float("nan")
            if not math.isfinite(fw) or fw <= 0:
                fw = float(fwhm_night)
                n_fb += 1
        r_ap, r_in, r_out = resolve_aperture_geometry(
            f=float(f_star),
            fwhm_px=float(fw),
            annulus_inner_fwhm=ANN_IN,
            annulus_outer_fwhm=ANN_OUT,
        )
        r_list.append(float(r_ap))
        keys.append(frame_key(fp.name))
        fl = _aperture_flux_uniform(data, pos, r_ap, r_in, r_out)
        for j, cid in enumerate(ids):
            flux[cid].append(float(fl[j]))
    tflux = np.asarray(flux[ids[0]], dtype=float)
    csum = np.zeros(len(lights), dtype=float)
    for c in ids[1:]:
        csum += np.asarray(flux[c], dtype=float)
    vy_rel = tflux / csum
    aij = pd.read_csv(AIJ_CSV)
    aij["Label"] = aij["Label"].map(frame_key)
    j = aij.merge(
        pd.DataFrame({"Label": keys, "vy_rel_pin": vy_rel}), on="Label", how="inner"
    )
    aij_rel = pd.to_numeric(j["rel_flux_T1"], errors="coerce").to_numpy()
    vy = pd.to_numeric(j["vy_rel_pin"], errors="coerce").to_numpy()
    ok = np.isfinite(aij_rel) & np.isfinite(vy) & (aij_rel > 0) & (vy > 0)
    rms = float("nan")
    if int(ok.sum()) >= 8:
        a = aij_rel[ok] / float(np.median(aij_rel[ok]))
        b = vy[ok] / float(np.median(vy[ok]))
        diff = -2.5 * np.log10(a / b) * 1000.0
        rms = float(np.sqrt(np.mean(diff * diff)))
    return {
        "ok": True,
        "mode": "per_target_dynamic_per_frame_r",
        "f": float(f_star),
        "fwhm_night_px": float(fwhm_night),
        "r_ap_median_px": float(np.median(r_list)) if r_list else None,
        "r_ap_min_px": float(np.min(r_list)) if r_list else None,
        "r_ap_max_px": float(np.max(r_list)) if r_list else None,
        "n_fwhm_fallback": int(n_fb),
        "matched": matched,
        "n_join": int(ok.sum()),
        "rms_diff_mmag": round(rms, 4) if math.isfinite(rms) else None,
        "gate_mmag": GATE_MMAG,
        "pass": bool(math.isfinite(rms) and rms <= GATE_MMAG),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "aij_csv_sha256": sha256_file(AIJ_CSV)[:16],
        "aij_tbl_sha256": sha256_file(AIJ_TBL)[:16],
    }


def aij_gate_fixed_night(draft_id: int, f_star: float, fwhm: float) -> dict:
    """Reference: fixed night FWHM (APERTURE-01c style) for comparison."""
    draft = ROOT / "Archive" / "Drafts" / f"draft_{draft_id:06d}"
    ms_path = draft / "platesolve" / SETUP / "masterstars_full_match.csv"
    if not ms_path.is_file():
        ms_path = (
            ROOT
            / "Archive"
            / "Drafts"
            / "draft_000516_snapshot_era04_20260826"
            / "platesolve"
            / SETUP
            / "masterstars_full_match.csv"
        )
    frames_dir = draft / "detrended_aligned" / "lights" / SETUP
    ens = read_aij_ensemble(AIJ_TBL)
    ms = pd.read_csv(ms_path, dtype={"catalog_id": str}, low_memory=False)
    matched = {}
    for role, rec in ens["stars"].items():
        matched[role] = match_role(ms, rec["ra_deg"], rec["dec_deg"])
    if any(v is None for v in matched.values()):
        return {"ok": False, "reason": "match_failed", "matched": matched}
    ids = [matched["T1"]] + [matched[f"C{i}"] for i in range(2, 7)]
    proc0 = next(frames_dir.glob("proc_*.csv"))
    pdf = pd.read_csv(proc0, dtype={"catalog_id": str}, low_memory=False)
    xy = []
    for cid in ids:
        row = pdf[pdf["catalog_id"].astype(str) == cid]
        if row.empty:
            return {"ok": False, "reason": f"xy_missing_{cid}"}
        xy.append((float(row.iloc[0]["x"]), float(row.iloc[0]["y"])))
    pos = np.asarray(xy, dtype=float)
    r_ap, r_in, r_out = resolve_aperture_geometry(
        f=float(f_star), fwhm_px=float(fwhm), annulus_inner_fwhm=ANN_IN, annulus_outer_fwhm=ANN_OUT
    )
    lights = sorted(
        p for p in frames_dir.glob("*.fits") if p.stem.upper() != "MASTERSTAR"
    )
    flux = {cid: [] for cid in ids}
    keys = []
    t0 = time.perf_counter()
    for fp in lights:
        with fits.open(fp, memmap=True) as hdul:
            data = np.asarray(hdul[0].data, dtype=float)
        keys.append(frame_key(fp.name))
        fl = _aperture_flux_uniform(data, pos, r_ap, r_in, r_out)
        for j, cid in enumerate(ids):
            flux[cid].append(float(fl[j]))
    tflux = np.asarray(flux[ids[0]], dtype=float)
    csum = np.zeros(len(lights), dtype=float)
    for c in ids[1:]:
        csum += np.asarray(flux[c], dtype=float)
    vy_rel = tflux / csum
    aij = pd.read_csv(AIJ_CSV)
    aij["Label"] = aij["Label"].map(frame_key)
    j = aij.merge(
        pd.DataFrame({"Label": keys, "vy_rel_pin": vy_rel}), on="Label", how="inner"
    )
    aij_rel = pd.to_numeric(j["rel_flux_T1"], errors="coerce").to_numpy()
    vy = pd.to_numeric(j["vy_rel_pin"], errors="coerce").to_numpy()
    ok = np.isfinite(aij_rel) & np.isfinite(vy) & (aij_rel > 0) & (vy > 0)
    rms = float("nan")
    if int(ok.sum()) >= 8:
        a = aij_rel[ok] / float(np.median(aij_rel[ok]))
        b = vy[ok] / float(np.median(vy[ok]))
        diff = -2.5 * np.log10(a / b) * 1000.0
        rms = float(np.sqrt(np.mean(diff * diff)))
    return {
        "ok": True,
        "mode": "f_fixed_night",
        "f": float(f_star),
        "fwhm_px": float(fwhm),
        "r_ap_px": float(r_ap),
        "matched": matched,
        "n_join": int(ok.sum()),
        "rms_diff_mmag": round(rms, 4) if math.isfinite(rms) else None,
        "gate_mmag": GATE_MMAG,
        "pass": bool(math.isfinite(rms) and rms <= GATE_MMAG),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


def product_hashes_516() -> dict:
    """Current live 516 product hashes (pre re-export; for era07 planning)."""
    draft = ROOT / "Archive" / "Drafts" / "draft_000516"
    photo = draft / "platesolve" / SETUP / "photometry"
    paths = [
        photo / "photometry_summary.csv",
        photo / "aperture_policy.json",
        photo / "active_targets.csv",
        photo / "comparison_stars_per_target.csv",
        photo / "lightcurves" / f"lightcurve_{BO_CID}.csv",
    ]
    out = {}
    for p in paths:
        if p.is_file():
            out[str(p.relative_to(ROOT))] = sha256_file(p)
        else:
            out[str(p.relative_to(ROOT))] = None
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "task": "APERTURE-DYNAMIC-01",
        "default_mode": "per_target",
        "f_grid": list(DEFAULT_APERTURE_F_GRID),
        "production_r": "f_star x FWHM_frame",
    }
    for did in (516, 521):
        print(f"=== validating draft {did} ===", flush=True)
        summary[f"draft_{did}"] = validate_draft(did)
        print(json.dumps(summary[f"draft_{did}"], indent=2), flush=True)

    d516 = summary["draft_516"]
    fwhm = float(d516["fwhm_night_px"])
    bo_f = F_FIXED
    if d516.get("bo_row") and d516["bo_row"].get("f_star"):
        bo_f = float(d516["bo_row"]["f_star"])
    print("=== AIJ gate f_fixed_night f=1.35 ===", flush=True)
    summary["aij_f_fixed_night"] = aij_gate_fixed_night(516, F_FIXED, fwhm)
    print(summary["aij_f_fixed_night"], flush=True)
    print(f"=== AIJ gate DYNAMIC per_target f*={bo_f} (per-frame r) ===", flush=True)
    summary["aij_dynamic_default"] = aij_gate_dynamic(516, bo_f, fwhm)
    print(summary["aij_dynamic_default"], flush=True)

    summary["draft_516_product_hashes_pre_reexport"] = product_hashes_516()

    g516 = d516.get("median_p2p_gain_mmag")
    bins = {b["mag_bin"]: b for b in d516.get("bin_summary") or []}
    faint = bins.get("G>=16") or bins.get("14-16")
    bright = bins.get("G<12")
    reading_parts = []
    if g516 is not None:
        reading_parts.append(f"516 median p2p gain {g516:.2f} mmag overall")
    if faint:
        reading_parts.append(
            f"faint bin {faint['mag_bin']}: median gain {faint['median_p2p_gain_mmag']:.2f} mmag "
            f"(median f*={faint['median_f_star']:.2f})"
        )
    if bright:
        reading_parts.append(
            f"bright bin {bright['mag_bin']}: median gain {bright['median_p2p_gain_mmag']:.2f} mmag "
            f"(median f*={bright['median_f_star']:.2f})"
        )
    aij_pt = summary.get("aij_dynamic_default") or {}
    if aij_pt.get("rms_diff_mmag") is not None:
        reading_parts.append(
            f"AIJ BO RMS(diff) DYNAMIC f*={bo_f}: {aij_pt['rms_diff_mmag']:.3f} mmag "
            f"(prev night-FWHM f*=1.0 was 5.255; f_fixed 1.35 was 2.887)"
        )
    reading_parts.append(
        f"n_f_edge 516={d516.get('n_f_edge')} 521={summary['draft_521'].get('n_f_edge')}"
    )
    summary["reading"] = "; ".join(reading_parts) if reading_parts else "no gain measured"
    summary["runtime_note"] = (
        f"per_target grid: 516={d516['grid_elapsed_s']}s "
        f"({d516['n_stars_measured']} stars x {d516['n_frames']} frames x "
        f"{len(DEFAULT_APERTURE_F_GRID)} f); "
        f"521={summary['draft_521']['grid_elapsed_s']}s."
    )
    (OUT / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="ascii"
    )
    print("READING:", summary["reading"], flush=True)
    print("RUNTIME:", summary["runtime_note"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
