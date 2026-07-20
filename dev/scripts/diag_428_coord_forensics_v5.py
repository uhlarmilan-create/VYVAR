#!/usr/bin/env python3
"""F-428-COORD forensics v5: peak test + direction stats (read-only)."""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.patches import Circle, Rectangle

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig
from gaia_catalog_id import normalize_gaia_source_id
from pipeline import _all_pix2world_icrs_deg
from repair_catalog_ids import _pick_gaia_table, _sep_arcsec

PRIORITY_TARGETS = [
    "FY CVn",
    "FZ CVn",
    "CSS_J134925.3+393524",
    "CSS_J140918.7+423422",
    "NSVS 5096293",
    "RX CVn",
    "R CVn",
]

STALE_WCS_PX = 1.5
PLATE_SCALE = 2.6
STAMP_HALF = 20  # 41x41
# Aperture radius ~2 px; Gaia WCS projection within this of ms x/y => same physical star.
SAME_STAR_PX = 2.5
DISTINCT_STAR_PX = 3.0


def _load_gaia_by_id(con: sqlite3.Connection, table: str, ids: set[str]) -> dict[str, tuple[float, float, float | None]]:
    out: dict[str, tuple[float, float, float | None]] = {}
    if not ids:
        return out
    mag_col = "phot_g_mean_mag" if "phot_g_mean_mag" in {r[1] for r in con.execute(f"PRAGMA table_info({table})")} else "g_mag"
    for i in range(0, len(sorted(ids)), 400):
        part = sorted(ids)[i : i + 400]
        ph = ",".join("?" * len(part))
        try:
            rows = con.execute(
                f"SELECT source_id, ra, dec, {mag_col} FROM {table} WHERE source_id IN ({ph})",
                part,
            ).fetchall()
        except sqlite3.OperationalError:
            rows = [(a, b, c, None) for a, b, c in con.execute(
                f"SELECT source_id, ra, dec FROM {table} WHERE source_id IN ({ph})", part
            ).fetchall()]
        for sid, ra, dec, gmag in rows:
            key = normalize_gaia_source_id(str(sid))
            if key:
                g = float(gmag) if gmag is not None and math.isfinite(float(gmag)) else None
                out[key] = (float(ra), float(dec), g)
    return out


def _delta_arcsec(ra_w: float, dec_w: float, ra_g: float, dec_g: float) -> tuple[float, float, float]:
    dra = (ra_w - ra_g) * math.cos(math.radians(dec_w)) * 3600.0
    dde = (dec_w - dec_g) * 3600.0
    mag = math.hypot(dra, dde)
    return dra, dde, mag


def _field_center_dist_arcmin(ra: np.ndarray, dec: np.ndarray, ra_c: float, dec_c: float) -> np.ndarray:
    dra = (ra - ra_c) * np.cos(np.radians(dec_c)) * 60.0
    dde = (dec - dec_c) * 60.0
    return np.sqrt(dra * dra + dde * dde)


def _nearest_sep_vec(ra: np.ndarray, dec: np.ndarray, gra: np.ndarray, gdec: np.ndarray) -> np.ndarray:
    n = len(ra)
    out = np.full(n, np.nan, dtype=np.float64)
    chunk = 128
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        ra_q = ra[i0:i1, None]
        dec_q = dec[i0:i1, None]
        cd = np.cos(np.radians(dec[i0:i1]))[:, None]
        dra = (gra[None, :] - ra_q) * cd
        dde = gdec[None, :] - dec_q
        out[i0:i1] = np.min(np.sqrt(dra * dra + dde * dde) * 3600.0, axis=1)
    return out


def _stamp_peak(data: np.ndarray, xc: float, yc: float, half: int = STAMP_HALF) -> dict[str, Any]:
    """Extract stamp and find brightest peak relative to stored centroid."""
    h, w = data.shape
    xi = int(round(xc))
    yi = int(round(yc))
    x0, x1 = max(0, xi - half), min(w, xi + half + 1)
    y0, y1 = max(0, yi - half), min(h, yi + half + 1)
    if x1 <= x0 or y1 <= y0:
        return {"ok": False, "reason": "out_of_bounds"}
    patch = np.asarray(data[y0:y1, x0:x1], dtype=np.float64)
    if patch.size == 0:
        return {"ok": False, "reason": "empty_patch"}
    iy, ix = np.unravel_index(int(np.nanargmax(patch)), patch.shape)
    px = float(x0 + ix)
    py = float(y0 + iy)
    peak_flux = float(patch[iy, ix])
    sep_px = math.hypot(px - xc, py - yc)
    # local background annulus proxy: corners of patch
    bg = float(np.nanmedian(np.concatenate([patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]])))
    snr = (peak_flux - bg) / max(1.0, float(np.nanstd(patch)))
    return {
        "ok": True,
        "nom_x": xc,
        "nom_y": yc,
        "peak_x": px,
        "peak_y": py,
        "peak_flux": peak_flux,
        "bg_median": bg,
        "snr": snr,
        "sep_peak_from_nominal_px": sep_px,
        "patch": patch,
        "origin": (x0, y0),
    }


def _has_significant_peak(st: dict[str, Any], *, min_snr: float = 3.0) -> bool:
    if not st.get("ok"):
        return False
    return bool(st.get("snr", 0) >= min_snr and st.get("peak_flux", 0) > st.get("bg_median", 0))


def _classify_peak_t2(
    st_xy: dict[str, Any],
    st_gaia: dict[str, Any],
    *,
    sep_proj_px: float,
    min_snr: float = 3.0,
    distinct_peak_px: float = 1.5,
) -> str:
    """Classify pixel identity: same-star overlap vs distinct second peak vs phantom."""
    pk_xy = _has_significant_peak(st_xy, min_snr=min_snr)
    if not pk_xy:
        return "PHANTOM"
    # Gaia catalog position projects onto the same pixel as ms x/y (within aperture).
    if sep_proj_px <= SAME_STAR_PX:
        return "CATALOG_PROJECTION_OFF"
    pk_g = _has_significant_peak(st_gaia, min_snr=min_snr)
    if not pk_g:
        return "CATALOG_PROJECTION_OFF"
    # Distinct stamp centers: require peak centroids separated from nominal positions.
    px_xy = (
        math.hypot(float(st_xy.get("peak_x", 0)) - float(st_xy.get("nom_x", 0)),
                   float(st_xy.get("peak_y", 0)) - float(st_xy.get("nom_y", 0)))
        if "nom_x" in st_xy
        else 0.0
    )
    px_g = (
        math.hypot(float(st_gaia.get("peak_x", 0)) - float(st_gaia.get("nom_x", 0)),
                   float(st_gaia.get("peak_y", 0)) - float(st_gaia.get("nom_y", 0)))
        if "nom_x" in st_gaia
        else 0.0
    )
    peak_sep = math.hypot(
        float(st_xy.get("peak_x", 0)) - float(st_gaia.get("peak_x", 0)),
        float(st_xy.get("peak_y", 0)) - float(st_gaia.get("peak_y", 0)),
    )
    if peak_sep >= distinct_peak_px and px_xy <= distinct_peak_px and px_g <= distinct_peak_px:
        return "GENUINE_CONFUSION"
    return "CATALOG_PROJECTION_OFF"


def _match_origin(row: pd.Series, sep_stored_gaia: float, sep_wcs_gaia: float, sep_wcs_stored: float) -> str:
    msep = pd.to_numeric(row.get("match_sep_arcsec"), errors="coerce")
    if sep_stored_gaia <= 1.0 and sep_wcs_gaia > 4.0:
        return "detection_gaia_coords_on_row"
    if sep_wcs_stored <= 1.0 and float(msep) if pd.notna(msep) else 999.0 <= 8.0:
        return "detection_catalog_match_coords"
    if sep_wcs_stored <= 1.0 and pd.notna(msep) and float(msep) > 8.0:
        return "optimizer_write_match_loose"
    if pd.notna(msep) and float(msep) > 8.0:
        return "optimizer_write_match_loose"
    return "detection_catalog_match_coords"


def _collect_violations(ms: pd.DataFrame, vt: pd.DataFrame, at: pd.DataFrame, threshold: float = 2.0) -> list[str]:
    ms_by_cid = {str(r["cid_norm"]): r for _, r in ms.iterrows() if str(r.get("cid_norm", ""))}
    ids: list[str] = []
    for cid in sorted(set(vt["_cid_norm"].dropna()) | set(at["_cid_norm"].dropna())):
        if not cid or cid not in ms_by_cid:
            continue
        vrows = vt[vt["_cid_norm"] == cid]
        arows = at[at["_cid_norm"] == cid]
        vrow = vrows.iloc[0] if not vrows.empty else (arows.iloc[0] if not arows.empty else None)
        if vrow is None:
            continue
        mrow = ms_by_cid[cid]
        if _sep_arcsec(float(vrow["ra_deg"]), float(vrow["dec_deg"]), float(mrow["ra_deg"]), float(mrow["dec_deg"])) > threshold:
            ids.append(cid)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description="F-428 coord forensics v5")
    ap.add_argument("--masterstars", type=Path, default=_ROOT / "Archive/Drafts/draft_000428/platesolve/NoFilter_60_2/masterstars_full_match.csv")
    ap.add_argument("--variable-targets", type=Path, default=_ROOT / "Archive/Drafts/draft_000428/platesolve/NoFilter_60_2/variable_targets.csv")
    ap.add_argument("--active-targets", type=Path, default=_ROOT / "Archive/Drafts/draft_000428/platesolve/NoFilter_60_2/photometry/active_targets.csv")
    ap.add_argument("--masterstar-fits", type=Path, default=_ROOT / "Archive/Drafts/draft_000428/platesolve/NoFilter_60_2/MASTERSTAR.fits")
    ap.add_argument("--photometry-dir", type=Path, default=_ROOT / "Archive/Drafts/draft_000428/platesolve/NoFilter_60_2/photometry")
    ap.add_argument("--out", type=Path, default=_ROOT / "tmp/f428_coord_forensics_v5.txt")
    ap.add_argument("--row-csv", type=Path, default=_ROOT / "tmp/f428_coord_v5_rows.csv")
    ap.add_argument("--contact-png", type=Path, default=_ROOT / "tmp/f428_priority_contact_sheet.png")
    ap.add_argument("--det-scatter-png", type=Path, default=_ROOT / "tmp/f428_det_scatter_v5.png")
    ap.add_argument("--gaia-db", type=Path, default=None)
    args = ap.parse_args()

    cfg = AppConfig()
    gdb = Path(args.gaia_db or cfg.gaia_db_path or "")

    ms = pd.read_csv(args.masterstars, low_memory=False, dtype={"catalog_id": str, "name": str})
    vt = pd.read_csv(args.variable_targets, dtype={"catalog_id": str})
    at = pd.read_csv(args.active_targets, dtype={"catalog_id": str})
    ms["cid_norm"] = ms["catalog_id"].map(normalize_gaia_source_id)
    vt["_cid_norm"] = vt["catalog_id"].map(normalize_gaia_source_id)
    at["_cid_norm"] = at["catalog_id"].map(normalize_gaia_source_id)

    violation_ids = _collect_violations(ms, vt, at)
    ms_by_cid = {str(r["cid_norm"]): r for _, r in ms.iterrows() if str(r.get("cid_norm", ""))}
    vt_by_cid = {str(r["_cid_norm"]): r for _, r in vt.iterrows() if str(r.get("_cid_norm", ""))}
    pri_cids: set[str] = set()
    for name in PRIORITY_TARGETS:
        arows = at[at["name"].astype(str) == name]
        if arows.empty:
            continue
        cid = normalize_gaia_source_id(arows.iloc[0]["catalog_id"])
        if cid:
            pri_cids.add(cid)

    with fits.open(args.masterstar_fits, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        hdr = hdul[0].header
        w = WCS(hdr)
        wpx = int(hdr.get("NAXIS1", data.shape[1]))
        hpx = int(hdr.get("NAXIS2", data.shape[0]))

    if not getattr(w, "has_celestial", False):
        print("No celestial WCS", file=sys.stderr)
        return 2

    foot_x = np.array([0.0, wpx, wpx, 0.0])
    foot_y = np.array([0.0, 0.0, hpx, hpx])
    foot_ra, foot_dec = w.all_pix2world(foot_x, foot_y, 0)
    ra_c = float(np.mean(foot_ra))
    dec_c = float(np.mean(foot_dec))

    con = sqlite3.connect(str(gdb))
    table = _pick_gaia_table(con)
    all_cids = set(violation_ids) | {c for c in ms["cid_norm"].dropna().astype(str) if c}
    gaia_by_id = _load_gaia_by_id(con, table, all_cids)
    con.close()

    stale_thr = STALE_WCS_PX * PLATE_SCALE
    rows_out: list[dict[str, Any]] = []

    for cid in sorted(ms_by_cid.keys()):
        if not cid:
            continue
        mrow = ms_by_cid[cid]
        g = gaia_by_id.get(cid)
        if g is None:
            continue
        g_ra, g_de, g_mag = g
        x = float(mrow["x"])
        y = float(mrow["y"])
        ra_w, dec_w = _all_pix2world_icrs_deg(w, np.array([x]), np.array([y]))
        ra_w, dec_w = float(ra_w[0]), float(dec_w[0])
        dra, dde, sep_wg = _delta_arcsec(ra_w, dec_w, g_ra, g_de)
        sep_sg = _sep_arcsec(float(mrow["ra_deg"]), float(mrow["dec_deg"]), g_ra, g_de)
        sep_ws = _sep_arcsec(ra_w, dec_w, float(mrow["ra_deg"]), float(mrow["dec_deg"]))
        if sep_wg <= stale_thr:
            v4_class = "CONSISTENT"
        elif sep_wg > stale_thr and sep_sg > stale_thr:
            v4_class = "MISASSIGNED-ID"
        else:
            v4_class = "STALE-COORDS"
        if cid not in violation_ids and v4_class == "MISASSIGNED-ID":
            v4_class = "CONSISTENT"

        gx, gy = w.world_to_pixel_values(g_ra, g_de)
        sep_proj_px = math.hypot(gx - x, gy - y)
        st_xy = _stamp_peak(data, x, y)
        st_gaia = _stamp_peak(data, float(gx), float(gy))
        t2_class = _classify_peak_t2(st_xy, st_gaia, sep_proj_px=sep_proj_px)
        origin = _match_origin(mrow, sep_sg, sep_wg, sep_ws)

        vt_name = ""
        vrow = vt_by_cid.get(cid)
        if vrow is not None:
            vt_name = str(vrow.get("vsx_name") or "")
        sep_vt_gaia = float("nan")
        if vrow is not None:
            sep_vt_gaia = _sep_arcsec(float(vrow["ra_deg"]), float(vrow["dec_deg"]), g_ra, g_de)

        rows_out.append(
            {
                "cid": cid,
                "name": vt_name or str(mrow.get("name") or cid),
                "v4_class": v4_class if cid in violation_ids or v4_class == "CONSISTENT" else "OTHER",
                "t2_class": t2_class,
                "match_origin": origin,
                "x": x,
                "y": y,
                "x_gaia_proj": float(gx),
                "y_gaia_proj": float(gy),
                "sep_proj_px": sep_proj_px,
                "sep_wcs_gaia_arcsec": sep_wg,
                "sep_vt_gaia_arcsec": sep_vt_gaia,
                "sep_stored_gaia_arcsec": sep_sg,
                "sep_wcs_stored_arcsec": sep_ws,
                "delta_ra_cosdec_arcsec": dra,
                "delta_dec_arcsec": dde,
                "peak_sep_from_xy_px": float(st_xy.get("sep_peak_from_nominal_px", float("nan"))),
                "peak_flux_xy": float(st_xy.get("peak_flux", float("nan"))),
                "peak_flux_gaia": float(st_gaia.get("peak_flux", float("nan"))),
                "peak_snr_xy": float(st_xy.get("snr", float("nan"))),
                "peak_snr_gaia": float(st_gaia.get("snr", float("nan"))),
                "match_sep_arcsec": pd.to_numeric(mrow.get("match_sep_arcsec"), errors="coerce"),
                "dist_field_center_arcmin": float(
                    _field_center_dist_arcmin(np.array([ra_w]), np.array([dec_w]), ra_c, dec_c)[0]
                ),
            }
        )

    df_rows = pd.DataFrame(rows_out)
    mis = df_rows[df_rows["v4_class"] == "MISASSIGNED-ID"].copy()
    con20 = df_rows[(df_rows["v4_class"] == "CONSISTENT") & df_rows["cid"].isin(violation_ids)].copy()

    def _dir_stats(sub: pd.DataFrame, label: str) -> dict[str, Any]:
        if sub.empty:
            return {"label": label, "n": 0}
        mags = np.hypot(sub["delta_ra_cosdec_arcsec"], sub["delta_dec_arcsec"])
        vm_ra = float(sub["delta_ra_cosdec_arcsec"].mean())
        vm_de = float(sub["delta_dec_arcsec"].mean())
        vm = float(math.hypot(vm_ra, vm_de))
        mean_abs = float(mags.mean())
        dist = sub["dist_field_center_arcmin"].to_numpy(dtype=float)
        corr = float(np.corrcoef(mags, dist)[0, 1]) if len(sub) > 2 else float("nan")
        ang = np.degrees(np.arctan2(sub["delta_dec_arcsec"], sub["delta_ra_cosdec_arcsec"]))
        hist, _ = np.histogram(ang, bins=8, range=(-180, 180))
        return {
            "label": label,
            "n": int(len(sub)),
            "vector_mean_dra": vm_ra,
            "vector_mean_ddec": vm_de,
            "vector_mean_mag": vm,
            "mean_abs_delta": mean_abs,
            "corr_mag_vs_center_dist": corr,
            "angle_hist_8bin": hist.tolist(),
        }

    t1_mis = _dir_stats(mis, "MISASSIGNED_164")
    t1_con = _dir_stats(con20, "CONSISTENT_20")

    origin_counts = mis["match_origin"].value_counts().to_dict() if not mis.empty else {}

    # T3 LC position evidence
    t3_lines: list[str] = []
    lc_dir = Path(args.photometry_dir) / "lightcurves"
    for name in PRIORITY_TARGETS:
        vrows = vt[vt["vsx_name"].astype(str).str.contains(name.split()[0], na=False)]
        if vrows.empty:
            vrows = vt[vt["vsx_name"] == name]
        if vrows.empty:
            continue
        cid = normalize_gaia_source_id(vrows.iloc[0]["catalog_id"])
        lc_path = lc_dir / f"lightcurve_{cid}.csv"
        if not lc_path.is_file():
            t3_lines.append(f"  {name}: no LC file")
            continue
        lc = pd.read_csv(lc_path, nrows=1)
        cols = set(lc.columns)
        if "x" in cols or "ra_deg" in cols:
            t3_lines.append(f"  {name}: per-frame position columns present")
        else:
            t3_lines.append(f"  {name}: LC has no x/y - n_frames={len(pd.read_csv(lc_path))} source_file only")

    align_path = Path(args.photometry_dir).parent / "alignment_report.csv"
    if align_path.is_file():
        ar = pd.read_csv(align_path)
        t3_lines.append(f"  alignment_report: {len(ar)} rows, cols={list(ar.columns)} - no per-target centroid scatter")

    # T4 unmatched DET + control
    det_mask = ms["name"].astype(str).str.match(r"^DET_\d+$", na=False) & (
        ms["catalog_id"].fillna("").astype(str).str.strip().isin(("", "nan", "None"))
    )
    unmatched = ms.loc[det_mask].copy()
    cone_path = args.masterstars.parent / "field_catalog_cone.csv"
    cone_f = pd.read_csv(cone_path, usecols=["ra_deg", "dec_deg"])
    cone_f = cone_f[
        (cone_f["ra_deg"] >= float(np.min(foot_ra)) - 0.01)
        & (cone_f["ra_deg"] <= float(np.max(foot_ra)) + 0.01)
        & (cone_f["dec_deg"] >= float(np.min(foot_dec)) - 0.01)
        & (cone_f["dec_deg"] <= float(np.max(foot_dec)) + 0.01)
    ]
    gra = cone_f["ra_deg"].to_numpy(dtype=float)
    gdec = cone_f["dec_deg"].to_numpy(dtype=float)
    u_ra = pd.to_numeric(unmatched["ra_deg"], errors="coerce").to_numpy(dtype=float)
    u_de = pd.to_numeric(unmatched["dec_deg"], errors="coerce").to_numpy(dtype=float)
    ux = pd.to_numeric(unmatched["x"], errors="coerce").to_numpy(dtype=float)
    uy = pd.to_numeric(unmatched["y"], errors="coerce").to_numpy(dtype=float)
    seps_det = _nearest_sep_vec(u_ra, u_de, gra, gdec)

    rng = np.random.default_rng(4285)
    n_ctrl = len(unmatched)
    cx = rng.uniform(0, wpx, size=n_ctrl)
    cy = rng.uniform(0, hpx, size=n_ctrl)
    cra, cde = w.all_pix2world(cx, cy, 0)
    seps_ctrl = _nearest_sep_vec(np.asarray(cra, float), np.asarray(cde, float), gra, gdec)

    p50_det = float(np.nanpercentile(seps_det, 50))
    p50_ctrl = float(np.nanpercentile(seps_ctrl, 50))
    area_deg2 = (wpx * PLATE_SCALE / 3600.0) * (hpx * PLATE_SCALE / 3600.0)
    n_cat = len(cone_f)
    poisson_est = float(3600.0 * 0.5 * math.sqrt(area_deg2 / max(1, n_cat)))  # rough char sep arcsec

    # edge ring fraction: within 50px of border
    edge = (
        (ux <= 50) | (ux >= wpx - 50) | (uy <= 50) | (uy >= hpx - 50)
    )
    edge_frac = float(np.mean(edge))

    ratio_det_ctrl = p50_det / max(p50_ctrl, 1e-6)
    # Poisson analytic uses uniform catalog density; field_catalog_cone is G-limited cone export
    # (100k cap, ~13.6 deg radius) with strong central concentration - random control NN p50 tracks
    # the actual in-frame catalog geometry (~122 arcsec) not the naive uniform-density estimate (~33 arcsec).
    poisson_vs_control_ratio = p50_ctrl / max(poisson_est, 1e-6)
    t4_poisson_note = (
        f"T4 Poisson reconciliation: analytic_uniform_p50~{poisson_est:.1f}\" assumes "
        f"uniform n={n_cat}/area={area_deg2:.3f}deg2; random_control_p50={p50_ctrl:.1f}\" "
        f"(ratio control/Poisson={poisson_vs_control_ratio:.2f}). Cone export is spatially "
        f"non-uniform (100k cap, G<=15.26, ~13.6 deg radius) - control matches catalog geometry; "
        f"not a units bug. Feeds F-428-A3-RADIUS / GAIA-DR4 depth track only."
    )
    if ratio_det_ctrl > 2.5 and abs(p50_ctrl - poisson_est) / max(poisson_est, 1) > 0.5:
        t4_verdict = "DIAG-BUG"
    elif edge_frac > 0.35:
        t4_verdict = "EDGE-CLUSTERED"
    elif ratio_det_ctrl < 1.8:
        t4_verdict = "SPURIOUS-UNIFORM"
    else:
        t4_verdict = "COVERAGE-MISMATCH"

    # T2 histogram for violation set
    t2_hist = mis["t2_class"].value_counts().to_dict() if not mis.empty else {}
    pri = df_rows[df_rows["cid"].isin(pri_cids)].copy()

    # Contact sheet
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    pri_targets: list[tuple[str, pd.Series]] = []
    for name in PRIORITY_TARGETS:
        arows = at[at["name"].astype(str) == name]
        if arows.empty:
            continue
        cid = normalize_gaia_source_id(arows.iloc[0]["catalog_id"])
        sub = df_rows[df_rows["cid"] == cid]
        if not sub.empty:
            pri_targets.append((name, sub.iloc[0]))
    for ax, (name, row) in zip(axes, pri_targets[:7]):
        st = _stamp_peak(data, float(row["x"]), float(row["y"]))
        if st.get("ok"):
            patch = st["patch"]
            ax.imshow(patch, origin="lower", cmap="gray", vmin=np.percentile(patch, 5), vmax=np.percentile(patch, 99.5))
            ox, oy = st["origin"]
            ax.plot(float(row["x"]) - ox, float(row["y"]) - oy, "r+", ms=12, mew=2, label="ms x/y")
            ax.plot(float(row["x_gaia_proj"]) - ox, float(row["y_gaia_proj"]) - oy, "c+", ms=12, mew=2, label="Gaia proj")
            ax.plot(st["peak_x"] - ox, st["peak_y"] - oy, "yo", ms=6, label="peak")
        ax.set_title(f"{name}\n{row['t2_class']}\n{row['sep_proj_px']:.0f}px", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    if len(pri_targets) < len(axes):
        axes[-1].axis("off")
    fig.suptitle("F-428 v5 priority targets: red=ms x/y  cyan=Gaia[cid] WCS proj  yellow=local peak", fontsize=10)
    fig.tight_layout()
    args.contact_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.contact_png, dpi=120)
    plt.close(fig)

    # DET scatter PNG
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.scatter(gra, gdec, s=1, c="lightgray", alpha=0.5, label=f"field_catalog n={n_cat}")
    ax2.scatter(u_ra, u_de, s=4, c=seps_det, cmap="viridis", alpha=0.6, label="unmatched DET")
    ax2.plot(foot_ra, foot_dec, "k-", lw=2, label="frame footprint")
    ax2.set_xlabel("RA deg")
    ax2.set_ylabel("Dec deg")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_title(f"Unmatched DET scatter (p50 NN={p50_det:.0f}\" control={p50_ctrl:.0f}\")")
    fig2.tight_layout()
    args.det_scatter_png.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(args.det_scatter_png, dpi=120)
    plt.close(fig2)

    # Gate
    t2_dom = t2_hist.get("CATALOG_PROJECTION_OFF", 0)
    t2_conf = t2_hist.get("GENUINE_CONFUSION", 0)
    pri_conf = int((pri["t2_class"] == "GENUINE_CONFUSION").sum()) if not pri.empty else 0

    lines: list[str] = []
    lines.append("# F-428-COORD forensics v5 (read-only)")
    lines.append(f"inputs: masterstars={args.masterstars.resolve()}")
    lines.append(f"inputs: MASTERSTAR.fits={args.masterstar_fits.resolve()}")
    lines.append(f"inputs: gaia_db={gdb.resolve()}")
    lines.append("")
    lines.append("# T1 - Direction statistics")
    lines.append(json.dumps(t1_mis, indent=2))
    lines.append(json.dumps(t1_con, indent=2))
    lines.append(f"match_origin_counts_MISASSIGNED: {json.dumps(origin_counts)}")
    if t1_mis.get("vector_mean_mag", 999) < 0.5 * t1_mis.get("mean_abs_delta", 1):
        lines.append("T1 interpretation: MISASSIGNED set isotropic (vector mean << mean |delta|) - not rigid shift")
    if abs(t1_mis.get("corr_mag_vs_center_dist", 0)) > 0.3:
        lines.append(f"T1 interpretation: radius-correlated |delta| corr={t1_mis.get('corr_mag_vs_center_dist'):.3f}")
    else:
        lines.append(f"T1 interpretation: weak radius correlation corr={t1_mis.get('corr_mag_vs_center_dist', float('nan')):.3f}")

    lines.append("")
    lines.append("# T2 - Peak test on MASTERSTAR.fits")
    lines.append(f"t2_class_histogram (MISASSIGNED n={len(mis)}): {json.dumps(t2_hist)}")
    pri_med_sep_proj = float(pri["sep_proj_px"].median()) if not pri.empty else float("nan")
    pri_med_sep_wcs = float(pri["sep_wcs_gaia_arcsec"].median()) if not pri.empty else float("nan")
    pri_med_vt_gaia = float(pri["sep_vt_gaia_arcsec"].median()) if not pri.empty else float("nan")
    lines.append(
        f"priority_summary: n={len(pri)} median_sep_proj_px={pri_med_sep_proj:.3f} "
        f"median_sep_wcs_gaia={pri_med_sep_wcs:.1f}\" median_sep_vt_gaia={pri_med_vt_gaia:.2f}\""
    )
    lines.append(
        f"priority_targets_t2: {pri[['name','t2_class','sep_proj_px','sep_wcs_gaia_arcsec','sep_vt_gaia_arcsec','peak_sep_from_xy_px']].to_string(index=False) if not pri.empty else 'none'}"
    )
    lines.append(f"contact_sheet: {args.contact_png.resolve()}")

    lines.append("")
    lines.append("# T3 - Per-frame LC position evidence")
    lines.extend(t3_lines if t3_lines else ["  (no data)"])

    lines.append("")
    lines.append("# T4 - 81\" population closure")
    lines.append(f"unmatched_DET: {len(unmatched)}")
    lines.append(f"field_catalog_in_frame: n={n_cat} area_deg2={area_deg2:.4f} poisson_est_p50~{poisson_est:.1f}\"")
    lines.append(f"det_nearest_catalog_p50: {p50_det:.3f}\"")
    lines.append(f"random_control_p50: {p50_ctrl:.3f}\" ratio det/control={ratio_det_ctrl:.2f}")
    lines.append(f"edge_margin_50px_fraction: {edge_frac:.3f}")
    lines.append(t4_poisson_note)
    lines.append(f"T4_verdict: {t4_verdict}")
    lines.append(f"det_scatter_png: {args.det_scatter_png.resolve()}")

    fail: list[str] = []
    gate = "STOP"
    if t4_verdict == "DIAG-BUG":
        fail.append("T4 control test suggests diagnostic bias - quarantine v3/v4 separations")
        gate = "QUARANTINE"
    elif pri_conf > 0:
        fail.append(f"T2 GENUINE_CONFUSION on {pri_conf} priority targets")
        gate = "STOP"
    elif t2_dom >= max(1, int(0.6 * len(mis))) and t1_mis.get("vector_mean_mag", 999) < 0.5 * t1_mis.get("mean_abs_delta", 1):
        gate = "RECLASSIFY-PROJECTION"
        lines.append("")
        lines.append("GATE: RECLASSIFY-PROJECTION - peak at ms x/y, not at Gaia projection; systematic not identity")
        lines.append("draft_428 flux likely correct; T4 re-run + anchor UNBLOCK pending Milan contact sheet review")
    else:
        fail.append("T2 mixed/ambiguous - Milan review required")
        gate = "STOP"

    lines.append("")
    lines.append(f"GATE_VERDICT: {gate}")
    if fail:
        lines.append("DIAG SELF-CHECK FAIL")
        for f in fail:
            lines.append(f"  - {f}")
    else:
        lines.append("DIAG SELF-CHECK PASS (v5 gate)")

    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    mis.to_csv(args.row_csv, index=False)
    df_rows.to_csv(args.row_csv.with_name("f428_coord_v5_all_rows.csv"), index=False)
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    return 1 if gate in ("STOP", "QUARANTINE") else 0


if __name__ == "__main__":
    raise SystemExit(main())
