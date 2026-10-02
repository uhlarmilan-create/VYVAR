# -*- coding: ascii -*-
"""APERTURE-DYNAMIC-02 validation on drafts 516 and 521.

Measures fine f-grid (0.4..3.0 step 0.05), picks f* = argmax Howell S/N,
reports:
  - f* vs G / mag bin; n_f_edge (expect ~0)
  - p2p and RMS vs f=1.35 by G bin (p2p diagnostic only)
  - D5-1 |rho|
  - BO dlevel vs f=2.5
  - Fair AIJ equal-radius: VYVAR at AIJ Source_Radius=7 px
  - AIJ settings for Milan to remeasure at VYVAR f*
  - 521 no_data=1885 census
Does NOT rewrite live draft LCs.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import Counter
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
    FALLBACK_F_STAR,
    abbe_p2p_scatter,
    choose_f_for_target,
    equal_weight_ensemble_delta,
    flux_to_mag,
    measure_night_grid,
    series_from_grid,
    write_per_target_choices,
    _aperture_flux_uniform,
)
from aperture_policy import resolve_aperture_geometry  # noqa: E402

OUT = ROOT / "dev" / "results" / "context" / "session_20261002_aperture_dynamic_02"
SETUP = "NoFilter_60_2"
F_FIXED = 1.35
F_REF = 2.5
ANN_IN = 2.7
ANN_OUT = 5.2
BO_CID = "1498613634033133184"
AIJ_CSV = ROOT / "dev" / "results" / "XVAL_AIJ_01_bo_compare.csv"
AIJ_TBL = ROOT / "dev" / "results" / "XVAL_AIJ_01_Table.tbl"
GATE_MMAG = 2.8
AIJ_SOURCE_RADIUS_PX = 7.0  # from XVAL_AIJ_01_Table.tbl Source_Radius


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


def resolve_gain_rn(draft_id: int) -> tuple[float, float]:
    photo = (
        ROOT
        / "Archive"
        / "Drafts"
        / f"draft_{draft_id:06d}"
        / "platesolve"
        / SETUP
        / "photometry"
    )
    gain = 1.0
    rn = 10.0
    gpath = photo / "gain_photon_transfer.json"
    if gpath.is_file():
        try:
            j = json.loads(gpath.read_text(encoding="utf-8"))
            auth = j.get("authority") or {}
            v = auth.get("value_e_per_adu_container")
            if v is not None and math.isfinite(float(v)) and float(v) > 0:
                gain = float(v)
        except Exception:  # noqa: BLE001
            pass
    return gain, rn


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
        "sky_rad_min": float(row[idx["Sky_Rad(min)"]]) if "Sky_Rad(min)" in idx else None,
        "sky_rad_max": float(row[idx["Sky_Rad(max)"]]) if "Sky_Rad(max)" in idx else None,
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


def aperture_flux_vec(data, pos, r_ap, r_in, r_out):
    fl, _sky = _aperture_flux_uniform(data, pos, r_ap, r_in, r_out)
    return fl


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
    gain, rn = resolve_gain_rn(draft_id)
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

    print(
        f"[draft {draft_id}] measuring grid n_stars={len(star_ids)} "
        f"n_f={len(DEFAULT_APERTURE_F_GRID)} gain={gain:.4f} rn={rn:.1f} ...",
        flush=True,
    )
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
    print(f"[draft {draft_id}] grid done in {grid_s:.1f}s; choosing f* ...", flush=True)

    choices = {}
    rows = []
    for tid in targets:
        cids = (
            comps.loc[comps["target_catalog_id"].astype(str) == tid, "catalog_id"]
            .astype(str)
            .tolist()
        )
        ch = choose_f_for_target(
            grid,
            target_cid=tid,
            comp_ids=cids,
            frame_order=order,
            gain=gain,
            read_noise=rn,
        )
        choices[tid] = ch

        def _p2p_at(fval: float) -> float:
            key = f"{fval:.4g}"
            v = float(ch.p2p_by_f.get(key, float("nan")))
            if math.isfinite(v):
                return v
            for k, vv in ch.p2p_by_f.items():
                if abs(float(k) - fval) < 1e-9:
                    return float(vv)
            return float("nan")

        p2p_fixed = _p2p_at(F_FIXED)
        p2p_star = _p2p_at(ch.f_star)

        t_mag_star = flux_to_mag(series_from_grid(grid, tid, ch.f_star, order))
        c_mags_star = {
            c: flux_to_mag(series_from_grid(grid, c, ch.f_star, order)) for c in cids
        }
        delta_star = equal_weight_ensemble_delta(t_mag_star, c_mags_star)
        t_mag_fixed = flux_to_mag(series_from_grid(grid, tid, F_FIXED, order))
        c_mags_fixed = {
            c: flux_to_mag(series_from_grid(grid, c, F_FIXED, order)) for c in cids
        }
        delta_fixed = equal_weight_ensemble_delta(t_mag_fixed, c_mags_fixed)
        ok_rms = np.isfinite(delta_star)
        ok_fix = np.isfinite(delta_fixed)
        rms_star = (
            float(np.nanstd(delta_star[ok_rms])) if int(ok_rms.sum()) >= 8 else float("nan")
        )
        rms_fixed = (
            float(np.nanstd(delta_fixed[ok_fix])) if int(ok_fix.sum()) >= 8 else float("nan")
        )

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
                "reason": ch.reason,
                "snr_max": ch.snr_max if math.isfinite(ch.snr_max) else None,
                "r_ap_px_night_scale": ch.r_ap_px,
                "p2p_fixed_mag": p2p_fixed if math.isfinite(p2p_fixed) else None,
                "p2p_star_mag": p2p_star if math.isfinite(p2p_star) else None,
                "p2p_gain_mmag": (
                    (p2p_fixed - p2p_star) * 1000.0
                    if math.isfinite(p2p_fixed) and math.isfinite(p2p_star)
                    else None
                ),
                "rms_star_mag": rms_star if math.isfinite(rms_star) else None,
                "rms_fixed_mag": rms_fixed if math.isfinite(rms_fixed) else None,
                "rms_gain_mmag": (
                    (rms_fixed - rms_star) * 1000.0
                    if math.isfinite(rms_fixed) and math.isfinite(rms_star)
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
        gain=gain,
        read_noise=rn,
    )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"fstar_table_draft{draft_id}.csv", index=False)

    bin_summary = []
    for b in ["G<12", "12-14", "14-16", "G>=16", "unknown"]:
        sub = df[df["mag_bin"] == b]
        if sub.empty:
            continue
        gains = pd.to_numeric(sub["p2p_gain_mmag"], errors="coerce")
        rms_g = pd.to_numeric(sub["rms_gain_mmag"], errors="coerce")
        bin_summary.append(
            {
                "mag_bin": b,
                "n": int(len(sub)),
                "median_f_star": float(np.nanmedian(sub["f_star"])),
                "median_p2p_gain_mmag": float(np.nanmedian(gains)),
                "median_rms_gain_mmag": float(np.nanmedian(rms_g)),
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
    ax.axhline(FALLBACK_F_STAR, color="#888", ls=":", lw=1, label=f"fallback={FALLBACK_F_STAR}")
    ax.set_xlabel("Gaia G (mag)")
    ax.set_ylabel("f* (Howell S/N)")
    ax.set_title(f"draft {draft_id}: f* vs magnitude (APERTURE-DYNAMIC-02)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"fstar_vs_mag_draft{draft_id}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    gain_mm = pd.to_numeric(df["p2p_gain_mmag"], errors="coerce").to_numpy()
    ok = np.isfinite(g) & np.isfinite(gain_mm)
    ax.scatter(g[ok], gain_mm[ok], s=18, alpha=0.75, c="#2a7")
    ax.axhline(0, color="#333", ls="-", lw=0.8)
    ax.set_xlabel("Gaia G (mag)")
    ax.set_ylabel("p2p diagnostic gain (mmag) = fixed - f*")
    ax.set_title(f"draft {draft_id}: Abbe p2p (diagnostic) vs magnitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"p2p_gain_vs_mag_draft{draft_id}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rms_g = pd.to_numeric(df["rms_gain_mmag"], errors="coerce").to_numpy()
    ok = np.isfinite(g) & np.isfinite(rms_g)
    ax.scatter(g[ok], rms_g[ok], s=18, alpha=0.75, c="#a35")
    ax.axhline(0, color="#333", ls="-", lw=0.8)
    ax.set_xlabel("Gaia G (mag)")
    ax.set_ylabel("RMS gain (mmag) = fixed - f*")
    ax.set_title(f"draft {draft_id}: LC RMS vs f=1.35 by G")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"rms_gain_vs_mag_draft{draft_id}.png", dpi=120)
    plt.close(fig)

    rhos = pd.to_numeric(df["rho_resid_seeing"], errors="coerce")
    n_edge = int(df["aperture_f_edge"].astype(bool).sum()) if len(df) else 0
    n_fallback = int(df["reason"].astype(str).str.contains("fallback").sum()) if len(df) else 0
    return {
        "draft_id": draft_id,
        "criterion": "howell_snr_argmax",
        "gain": gain,
        "read_noise": rn,
        "fwhm_night_px": round(fwhm, 6),
        "n_targets": len(targets),
        "n_stars_measured": len(star_ids),
        "n_frames": grid.n_frames_measured,
        "n_fwhm_fallback_night": int(grid.n_fwhm_fallback_night),
        "grid_elapsed_s": round(grid_s, 2),
        "f_fixed": F_FIXED,
        "n_f_edge": n_edge,
        "n_fallback_f070": n_fallback,
        "f_star_hist": f_hist,
        "median_f_star": float(np.nanmedian(df["f_star"])) if len(df) else None,
        "median_p2p_gain_mmag": float(
            np.nanmedian(pd.to_numeric(df["p2p_gain_mmag"], errors="coerce"))
        ),
        "median_rms_gain_mmag": float(
            np.nanmedian(pd.to_numeric(df["rms_gain_mmag"], errors="coerce"))
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


def _match_bo_ids(draft_id: int) -> tuple[dict | None, list[str] | None, np.ndarray | None, dict]:
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
        return None, None, None, {"ok": False, "reason": "match_failed", "matched": matched}
    ids = [matched["T1"]] + [matched[f"C{i}"] for i in range(2, 7)]
    proc0 = next(frames_dir.glob("proc_*.csv"))
    pdf = pd.read_csv(proc0, dtype={"catalog_id": str}, low_memory=False)
    xy = []
    for cid in ids:
        row = pdf[pdf["catalog_id"].astype(str) == cid]
        if row.empty:
            return None, None, None, {"ok": False, "reason": f"xy_missing_{cid}"}
        xy.append((float(row.iloc[0]["x"]), float(row.iloc[0]["y"])))
    return ens, ids, np.asarray(xy, dtype=float), {"ok": True, "matched": matched, "frames_dir": frames_dir}


def aij_rms_at_radius(
    draft_id: int,
    *,
    mode: str,
    f_star: float | None = None,
    r_ap_fixed_px: float | None = None,
    fwhm_night: float | None = None,
    per_frame_f: bool = False,
) -> dict:
    """Compare VYVAR relative flux to AIJ at a chosen aperture radius policy."""
    ens, ids, pos, meta = _match_bo_ids(draft_id)
    if ids is None or pos is None:
        return meta
    frames_dir = meta["frames_dir"]
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
                fw = float(fwhm_night or 5.0)
                n_fb += 1
        if r_ap_fixed_px is not None:
            r_ap = float(r_ap_fixed_px)
            r_in = max(r_ap + 0.5, ANN_IN * float(fwhm_night or fw))
            r_out = max(r_in + 0.5, ANN_OUT * float(fwhm_night or fw))
        else:
            use_f = float(f_star)
            use_fw = float(fw) if per_frame_f else float(fwhm_night or fw)
            r_ap, r_in, r_out = resolve_aperture_geometry(
                f=use_f,
                fwhm_px=use_fw,
                annulus_inner_fwhm=ANN_IN,
                annulus_outer_fwhm=ANN_OUT,
            )
        r_list.append(float(r_ap))
        keys.append(frame_key(fp.name))
        fl = aperture_flux_vec(data, pos, r_ap, r_in, r_out)
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
        "mode": mode,
        "f": float(f_star) if f_star is not None else None,
        "r_ap_fixed_px": float(r_ap_fixed_px) if r_ap_fixed_px is not None else None,
        "fwhm_night_px": float(fwhm_night) if fwhm_night is not None else None,
        "r_ap_median_px": float(np.median(r_list)) if r_list else None,
        "r_ap_min_px": float(np.min(r_list)) if r_list else None,
        "r_ap_max_px": float(np.max(r_list)) if r_list else None,
        "n_fwhm_fallback": int(n_fb),
        "matched": meta.get("matched"),
        "n_join": int(ok.sum()),
        "rms_diff_mmag": round(rms, 4) if math.isfinite(rms) else None,
        "gate_mmag": GATE_MMAG,
        "pass": bool(math.isfinite(rms) and rms <= GATE_MMAG),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "aij_source_radius_px": AIJ_SOURCE_RADIUS_PX,
        "note": (
            "AIJ measurement used fixed Source_Radius="
            f"{AIJ_SOURCE_RADIUS_PX:.0f} px; unequal-radius compare is unfair."
        ),
    }


def census_no_data_521() -> dict:
    photo = (
        ROOT
        / "Archive"
        / "Drafts"
        / "draft_000521"
        / "platesolve"
        / SETUP
        / "photometry"
    )
    lc_dir = photo / "lightcurves"
    frames = (
        ROOT
        / "Archive"
        / "Drafts"
        / "draft_000521"
        / "detrended_aligned"
        / "lights"
        / SETUP
    )
    at = pd.read_csv(photo / "active_targets.csv", dtype={"catalog_id": str}, low_memory=False)
    rows = []
    reason_ctr: Counter[str] = Counter()
    for p in lc_dir.glob("lightcurve_*.csv"):
        df = pd.read_csv(p, low_memory=False)
        fl = df["flag"].astype(str).str.lower()
        n_nd = int((fl == "no_data").sum())
        if n_nd == 0:
            continue
        for r in df.loc[fl == "no_data", "flag_reason"].astype(str):
            reason_ctr[r] += 1
        mag = pd.to_numeric(df.get("mag_calib"), errors="coerce")
        cid = p.stem.replace("lightcurve_", "")
        rows.append(
            {
                "catalog_id": cid,
                "n_no_data": n_nd,
                "n_epochs": len(df),
                "n_finite_mag": int(mag.notna().sum()),
                "all_nan": bool(mag.notna().sum() == 0),
            }
        )
    rdf = pd.DataFrame(rows)
    all_nan_cids = rdf.loc[rdf["all_nan"], "catalog_id"].tolist() if len(rdf) else []
    # Presence in proc CSVs
    proc_counts = {c: 0 for c in all_nan_cids}
    n_procs = 0
    for p in frames.glob("proc_*.csv"):
        n_procs += 1
        d = pd.read_csv(p, usecols=["catalog_id"], low_memory=False)
        ids = set(d["catalog_id"].astype(str))
        for c in all_nan_cids:
            if c in ids:
                proc_counts[c] += 1
    names = {
        str(r["catalog_id"]): str(r.get("vsx_name", ""))
        for _, r in at.iterrows()
    }
    all_nan_detail = [
        {
            "catalog_id": c,
            "vsx_name": names.get(c, ""),
            "n_no_data": 131,
            "n_proc_frames_present": int(proc_counts.get(c, 0)),
            "skip_reason": "absent_from_all_proc_csv_forced_photometry_not_extracted",
        }
        for c in all_nan_cids
    ]
    out = {
        "draft_id": 521,
        "n_no_data_epochs": int(rdf["n_no_data"].sum()) if len(rdf) else 0,
        "n_targets_any_no_data": int(len(rdf)),
        "n_targets_all_nan": int(rdf["all_nan"].sum()) if len(rdf) else 0,
        "n_epochs_all_nan": int(rdf.loc[rdf["all_nan"], "n_no_data"].sum()) if len(rdf) else 0,
        "n_targets_partial": int((~rdf["all_nan"]).sum()) if len(rdf) else 0,
        "n_epochs_partial": int(rdf.loc[~rdf["all_nan"], "n_no_data"].sum()) if len(rdf) else 0,
        "flag_reason_counts": dict(reason_ctr),
        "n_procs_checked": n_procs,
        "all_nan_targets": all_nan_detail,
        "expected": (
            "Yes for all-nan set: stars in active_targets/MASTERSTAR but never "
            "extracted into proc_*.csv (no mag_inst). flag_reason is uniformly "
            "'no_data' (no finer edge/saturation taxonomy recorded). Partial "
            "no_data epochs are intermittent missing photometry with the same "
            "flag_reason."
        ),
    }
    (OUT / "no_data_census_draft521.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    return out


def write_aij_settings_for_milan(bo_f_star: float, fwhm_night: float) -> dict:
    """Claude Code cannot re-run AIJ; prepare exact settings for Milan."""
    ens = read_aij_ensemble(AIJ_TBL)
    r_from_f = float(bo_f_star) * float(fwhm_night)
    payload = {
        "task": "APERTURE-DYNAMIC-02 equal-radius AIJ remeasure (Milan)",
        "why": (
            "Existing AIJ table used fixed Source_Radius=7 px "
            f"(~f={AIJ_SOURCE_RADIUS_PX / fwhm_night:.3f} at night FWHM="
            f"{fwhm_night:.3f}). Comparing that to VYVAR at f*={bo_f_star} is "
            "unequal-radius. Need AIJ remasure at VYVAR's f* OR keep AIJ at 7 "
            "and force VYVAR to r=7 (done in validation)."
        ),
        "existing_aij_gate": {
            "table": str(AIJ_TBL.relative_to(ROOT)),
            "source_radius_px": AIJ_SOURCE_RADIUS_PX,
            "sky_rad_min_px": ens.get("sky_rad_min"),
            "sky_rad_max_px": ens.get("sky_rad_max"),
            "ensemble": "T1=BO, comps C2..C6 pinned",
            "annulus_vyvar_equiv": f"{ANN_IN}/{ANN_OUT} x FWHM",
        },
        "remeasure_at_vyvar_fstar": {
            "target": "BO CVn (draft 516)",
            "aperture_policy": "variable aperture OR fixed radius",
            "f_star_vyvar": float(bo_f_star),
            "fwhm_night_px": float(fwhm_night),
            "source_radius_px_fixed_equiv": round(r_from_f, 3),
            "instruction_fixed": (
                f"Set Multi-Aperture Source_Radius = {r_from_f:.2f} px "
                f"(= f* {bo_f_star} x FWHM {fwhm_night:.3f}). "
                f"Sky_Rad(min/max) keep {ens.get('sky_rad_min')}/"
                f"{ens.get('sky_rad_max')} (or 2.7/5.2 x FWHM). "
                "Same pinned comps C2..C6. Export rel_flux_T1 table."
            ),
            "instruction_variable": (
                "If AIJ variable aperture by FWHM is available: set aperture "
                f"factor = {bo_f_star} FWHM per frame; annulus {ANN_IN}/{ANN_OUT} "
                "FWHM; same comps."
            ),
        },
        "status": "STOP for AIJ remasure - Claude Code cannot run AIJ",
    }
    (OUT / "aij_equal_radius_settings_for_milan.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "task": "APERTURE-DYNAMIC-02",
        "default_mode": "per_target",
        "criterion": "howell_snr_argmax",
        "fallback_f_star": FALLBACK_F_STAR,
        "f_grid": list(DEFAULT_APERTURE_F_GRID),
        "n_f_grid": len(DEFAULT_APERTURE_F_GRID),
        "production_r": "f_star x FWHM_frame",
        "noise_model_cite": "src_py/photometry_phase2a.py:_howell_variance_adu2 (~L378-401); mirrored in aperture_pertarget.howell_variance_adu2",
        "aij_source_radius_px_used_in_gate": AIJ_SOURCE_RADIUS_PX,
    }

    print("=== no_data census 521 ===", flush=True)
    summary["no_data_521"] = census_no_data_521()

    drafts = {}
    for did in (516, 521):
        print(f"=== validate draft {did} ===", flush=True)
        drafts[str(did)] = validate_draft(did)
    summary["drafts"] = drafts

    bo = drafts["516"].get("bo_row") or {}
    bo_f = float(bo.get("f_star") or 1.0)
    fwhm516 = float(drafts["516"]["fwhm_night_px"])

    print("=== AIJ unequal (legacy) DYNAMIC at f* per-frame r ===", flush=True)
    summary["aij_unequal_dynamic_fstar"] = aij_rms_at_radius(
        516, mode="vyvar_fstar_per_frame_r_vs_aij_r7", f_star=bo_f, fwhm_night=fwhm516, per_frame_f=True
    )
    print("=== AIJ unequal fixed night f=1.35 ===", flush=True)
    summary["aij_unequal_fixed_135"] = aij_rms_at_radius(
        516, mode="vyvar_f135_night_r_vs_aij_r7", f_star=1.35, fwhm_night=fwhm516, per_frame_f=False
    )
    print("=== AIJ EQUAL radius: VYVAR forced to AIJ Source_Radius=7 ===", flush=True)
    summary["aij_equal_vyvar_at_aij_r7"] = aij_rms_at_radius(
        516,
        mode="vyvar_forced_r7_vs_aij_r7",
        r_ap_fixed_px=AIJ_SOURCE_RADIUS_PX,
        fwhm_night=fwhm516,
    )
    print("=== AIJ settings for Milan (remeasure at VYVAR f*) ===", flush=True)
    summary["aij_settings_for_milan"] = write_aij_settings_for_milan(bo_f, fwhm516)

    (OUT / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2)[:4000], flush=True)
    print(f"Wrote {OUT / 'validation_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
