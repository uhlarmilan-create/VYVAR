#!/usr/bin/env python3
"""Wide field: true (WCS+Gaia) triangle shapes — center vs edge, flat vs gnomonic vs sky."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder
from scipy.spatial import KDTree

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from database import query_local_gaia  # noqa: E402
from vyvar_blind_solver import _triangle_sides_arcsec  # noqa: E402


def _angular_sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    c1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg, frame="icrs")
    c2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg, frame="icrs")
    return float(c1.separation(c2).arcsec)


def _triangle_shape_from_sides_arcsec(
    L1: float, L2: float, L3: float
) -> tuple[float, float, float] | None:
    if L3 < 0.1 or L1 / L3 < 0.15:
        return None
    return float(L1 / L3), float(L2 / L3), float(math.log10(L3))


def _triangle_shape_sky(
    ra0: float,
    dec0: float,
    ra1: float,
    dec1: float,
    ra2: float,
    dec2: float,
) -> tuple[float, float, float] | None:
    s01 = _angular_sep_arcsec(ra0, dec0, ra1, dec1)
    s12 = _angular_sep_arcsec(ra1, dec1, ra2, dec2)
    s02 = _angular_sep_arcsec(ra0, dec0, ra2, dec2)
    L1, L2, L3 = sorted([s01, s12, s02])
    return _triangle_shape_from_sides_arcsec(L1, L2, L3)


def _triangle_shape_img(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    x_cen: float,
    y_cen: float,
    plate_scale: float,
    use_gnomonic: bool,
) -> tuple[float, float, float] | None:
    L1, L2, L3 = _triangle_sides_arcsec(
        p0,
        p1,
        p2,
        x_cen=x_cen,
        y_cen=y_cen,
        plate_scale_arcsec_per_px=plate_scale,
        use_gnomonic=use_gnomonic,
    )
    return _triangle_shape_from_sides_arcsec(L1, L2, L3)


def _match_dao_gaia(
    *,
    dao_df: pd.DataFrame,
    wcs: WCS,
    gaia_rows: list[dict[str, Any]],
    max_sep_px: float,
) -> pd.DataFrame:
    if dao_df.empty or not gaia_rows:
        return pd.DataFrame()
    ra = np.asarray([float(r["ra"]) for r in gaia_rows], dtype=np.float64)
    dec = np.asarray([float(r["dec"]) for r in gaia_rows], dtype=np.float64)
    px, py = wcs.all_world2pix(ra, dec, 0)
    gdf = pd.DataFrame({"ra": ra, "dec": dec, "g_mag": [r.get("g_mag") for r in gaia_rows], "px": px, "py": py})
    gtree = KDTree(gdf[["px", "py"]].to_numpy(dtype=np.float64))
    hits = []
    for _, row in dao_df.iterrows():
        x, y = float(row["x"]), float(row["y"])
        d, idx = gtree.query([x, y], k=1)
        if float(d) <= max_sep_px and 0 <= int(idx) < len(gdf):
            g = gdf.iloc[int(idx)]
            hits.append(
                {
                    "x": x,
                    "y": y,
                    "flux": float(row.get("flux", 1.0)),
                    "ra": float(g["ra"]),
                    "dec": float(g["dec"]),
                    "g_mag": float(g["g_mag"]) if g["g_mag"] is not None else float("nan"),
                    "match_px": float(d),
                }
            )
    return pd.DataFrame(hits)


def _sample_triangles(
    matched: pd.DataFrame,
    *,
    x_cen: float,
    y_cen: float,
    plate_scale: float,
    k_neighbors: int,
    max_stars: int,
    region_mask: np.ndarray,
    max_samples: int = 8000,
) -> list[dict[str, Any]]:
    sub = matched.loc[region_mask].copy()
    if len(sub) < 3:
        return []
    sub = sub.sort_values("flux", ascending=False).head(max_stars)
    stars_xy = sub[["x", "y"]].to_numpy(dtype=np.float64)
    ra_dec = sub[["ra", "dec"]].to_numpy(dtype=np.float64)
    tree = KDTree(stars_xy)
    k = min(int(k_neighbors), len(sub))
    _, nn_idx = tree.query(stars_xy, k=k)
    if nn_idx.ndim == 1:
        nn_idx = nn_idx.reshape(-1, 1)
    combos = list(itertools.combinations(range(k), 3))
    out: list[dict[str, Any]] = []
    owner = np.arange(len(sub), dtype=np.int64)
    for i in range(len(sub)):
        i_tri = nn_idx[i]
        for c in combos:
            tri_local = i_tri[list(c)]
            if owner[i] != int(np.min(tri_local)):
                continue
            p0, p1, p2 = stars_xy[tri_local[0]], stars_xy[tri_local[1]], stars_xy[tri_local[2]]
            r0, d0 = ra_dec[tri_local[0]]
            r1, d1 = ra_dec[tri_local[1]]
            r2, d2 = ra_dec[tri_local[2]]
            sky = _triangle_shape_sky(r0, d0, r1, d1, r2, d2)
            flat = _triangle_shape_img(p0, p1, p2, x_cen=x_cen, y_cen=y_cen, plate_scale=plate_scale, use_gnomonic=False)
            gno = _triangle_shape_img(p0, p1, p2, x_cen=x_cen, y_cen=y_cen, plate_scale=plate_scale, use_gnomonic=True)
            if sky is None or flat is None or gno is None:
                continue
            out.append(
                {
                    "r1_sky": sky[0],
                    "r2_sky": sky[1],
                    "log_l3_sky": sky[2],
                    "r1_flat": flat[0],
                    "r2_flat": flat[1],
                    "log_l3_flat": flat[2],
                    "r1_gno": gno[0],
                    "r2_gno": gno[1],
                    "log_l3_gno": gno[2],
                    "dr1_flat": abs(flat[0] - sky[0]),
                    "dr2_flat": abs(flat[1] - sky[1]),
                    "dl3_flat": abs(flat[2] - sky[2]),
                    "dr1_gno": abs(gno[0] - sky[0]),
                    "dr2_gno": abs(gno[1] - sky[1]),
                    "dl3_gno": abs(gno[2] - sky[2]),
                }
            )
            if len(out) >= max_samples:
                return out
    return out


def _summarize(samples: list[dict[str, Any]]) -> dict[str, float]:
    if not samples:
        return {}
    arr = {k: np.asarray([s[k] for s in samples], dtype=np.float64) for k in samples[0]}
    return {
        "n": float(len(samples)),
        "dr1_flat_med": float(np.median(arr["dr1_flat"])),
        "dr2_flat_med": float(np.median(arr["dr2_flat"])),
        "dl3_flat_med": float(np.median(arr["dl3_flat"])),
        "dr1_gno_med": float(np.median(arr["dr1_gno"])),
        "dr2_gno_med": float(np.median(arr["dr2_gno"])),
        "dl3_gno_med": float(np.median(arr["dl3_gno"])),
        "dr1_flat_p90": float(np.percentile(arr["dr1_flat"], 90)),
        "dr2_flat_p90": float(np.percentile(arr["dr2_flat"], 90)),
        "dl3_flat_p90": float(np.percentile(arr["dl3_flat"], 90)),
        "dr1_gno_p90": float(np.percentile(arr["dr1_gno"], 90)),
        "dr2_gno_p90": float(np.percentile(arr["dr2_gno"], 90)),
        "dl3_gno_p90": float(np.percentile(arr["dl3_gno"], 90)),
    }


def _index_lookup_stats(
    samples: list[dict[str, Any]],
    *,
    index_path: Path,
    truth_ra: float,
    truth_dec: float,
) -> dict[str, float]:
    if not samples or not index_path.is_file():
        return {}
    with index_path.open("rb") as f:
        idx = pickle.load(f)
    hash_tree = idx["tree"]
    meta = np.asarray(idx["metadata"], dtype=np.float64)
    log_L3_min = float(idx["log_L3_min"])
    log_L3_max = float(idx["log_L3_max"])
    log_range = max(log_L3_max - log_L3_min, 1e-6)
    seps: list[float] = []
    votes_2 = 0
    votes_5 = 0
    for s in samples:
        log_norm = (float(s["log_l3_sky"]) - log_L3_min) / log_range
        if log_norm < 0 or log_norm > 1:
            continue
        dist, mi = hash_tree.query([s["r1_sky"], s["r2_sky"], log_norm], k=1)
        if not np.isfinite(dist):
            continue
        mi = int(mi)
        if mi < 0 or mi >= len(meta):
            continue
        cra, cdec = float(meta[mi, 0]), float(meta[mi, 1])
        dra = (cra - truth_ra) * math.cos(math.radians((cdec + truth_dec) / 2.0))
        ddec = cdec - truth_dec
        sep = math.sqrt(dra * dra + ddec * ddec)
        seps.append(sep)
        if sep < 2.0:
            votes_2 += 1
        if sep < 5.0:
            votes_5 += 1
    if not seps:
        return {}
    arr = np.asarray(seps, dtype=np.float64)
    return {
        "n_index_hits": float(len(seps)),
        "vote_sep_deg_med": float(np.median(arr)),
        "vote_sep_deg_min": float(np.min(arr)),
        "votes_near_truth_2deg": float(votes_2),
        "votes_near_truth_5deg": float(votes_5),
    }


def _recommendation(
    center: dict[str, float],
    edge: dict[str, float],
    idx_stats: dict[str, float],
) -> str:
    if not center or not edge:
        return "insufficient_matched_stars"
    c_flat = max(center.get("dr1_flat_p90", 0), center.get("dr2_flat_p90", 0), center.get("dl3_flat_p90", 0))
    e_flat = max(edge.get("dr1_flat_p90", 0), edge.get("dr2_flat_p90", 0), edge.get("dl3_flat_p90", 0))
    c_gno = max(center.get("dr1_gno_p90", 0), center.get("dr2_gno_p90", 0), center.get("dl3_gno_p90", 0))
    e_gno = max(edge.get("dr1_gno_p90", 0), edge.get("dr2_gno_p90", 0), edge.get("dl3_gno_p90", 0))
    shape_tol = 0.02
    v2 = int(idx_stats.get("votes_near_truth_2deg", 0))
    vmin = float(idx_stats.get("vote_sep_deg_min", 999.0))
    if e_flat > shape_tol and e_gno < shape_tol and c_gno < shape_tol:
        return "gnomonic_required_at_edge; consider_central_region_kNN_for_wide"
    if c_flat < shape_tol and e_flat < shape_tol:
        if v2 > 0 and vmin < 2.0:
            return (
                "shape_ok; index_hashes_hit_truth_with_correct_sky_triangles — "
                "prioritize_quads_or_vote_clustering_not_central_kNN"
            )
        return "shape_ok; index_sparse_or_wrong_hashes — tune_wide_index_or_quads"
    if e_gno < e_flat * 0.5:
        return "gnomonic_helps_edge; deploy_gnomonic_plus_central_kNN"
    if max(c_flat, e_flat) > 0.05:
        return "quads_or_deeper_index_tune"
    return "marginal; try_vote_clustering_first"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=str, default="draft_000365")
    ap.add_argument("--setup", type=str, default="NoFilter_60_2")
    ap.add_argument("--match-tol-px", type=float, default=3.0)
    ap.add_argument("--central-frac", type=float, default=0.35, help="radius fraction for 'center'")
    ap.add_argument("--edge-frac", type=float, default=0.65, help="min radius fraction for 'edge'")
    ap.add_argument("--k-neighbors", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--index",
        type=Path,
        default=_ROOT / "GAIA_DR3/gaia_triangles_wide.pkl",
    )
    ap.add_argument("--truth-ra", type=float, default=241.53869)
    ap.add_argument("--truth-dec", type=float, default=50.29571)
    args = ap.parse_args()

    cfg = AppConfig()
    draft_dir = Path(cfg.archive_root) / "Drafts" / args.draft
    ms = draft_dir / "platesolve" / args.setup / "MASTERSTAR.fits"
    if not ms.is_file():
        ms = next(draft_dir.rglob("MASTERSTAR.fits"), None)
    if ms is None or not ms.is_file():
        print(f"ERROR: no MASTERSTAR under {draft_dir}")
        return 1

    with fits.open(ms, memmap=True) as hdul:
        hdr = hdul[0].header
        data = np.asarray(hdul[0].data, dtype=np.float64)
    ny, nx = data.shape
    wcs = WCS(hdr)
    if not wcs.is_celestial:
        print("ERROR: MASTERSTAR has no celestial WCS")
        return 1

    ps = float(hdr.get("VY_PLTS", abs(wcs.pixel_scale_matrix).mean() * 3600.0))
    fov_deg = max(nx, ny) * ps / 3600.0
    cx, cy = nx / 2.0, ny / 2.0
    half_diag = math.hypot(nx, ny) / 2.0

    _, med, std = sigma_clipped_stats(data, sigma=3.0)
    finder = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)
    srcs = finder(data - med)
    n_dao = 0 if srcs is None else len(srcs)
    dao_df = pd.DataFrame()
    if srcs is not None and n_dao > 0:
        dao_df = srcs.to_pandas().rename(columns={"xcentroid": "x", "ycentroid": "y"})
        if "peak" in dao_df.columns:
            dao_df["flux"] = dao_df["peak"]
        dao_df = dao_df.sort_values("flux", ascending=False)

    crval_ra = float(wcs.wcs.crval[0])
    crval_dec = float(wcs.wcs.crval[1])
    cone = max(0.5, fov_deg * 0.55)
    ra_min = crval_ra - cone / max(math.cos(math.radians(crval_dec)), 1e-6)
    ra_max = crval_ra + cone / max(math.cos(math.radians(crval_dec)), 1e-6)
    gaia_rows = query_local_gaia(
        cfg.gaia_db_path,
        ra_min=ra_min,
        ra_max=ra_max,
        dec_min=crval_dec - cone,
        dec_max=crval_dec + cone,
        mag_limit=14.0,
        max_rows=int(cfg.catalog_query_max_rows),
    )

    matched = _match_dao_gaia(
        dao_df=dao_df,
        wcs=wcs,
        gaia_rows=gaia_rows,
        max_sep_px=float(args.match_tol_px),
    )
    if matched.empty:
        print("ERROR: no DAO–Gaia matches")
        return 1

    dist_px = np.sqrt((matched["x"] - cx) ** 2 + (matched["y"] - cy) ** 2)
    matched = matched.assign(dist_frac=dist_px / max(half_diag, 1e-6))
    center_mask = matched["dist_frac"].to_numpy() <= float(args.central_frac)
    edge_mask = matched["dist_frac"].to_numpy() >= float(args.edge_frac)

    center_s = _sample_triangles(
        matched,
        x_cen=cx,
        y_cen=cy,
        plate_scale=ps,
        k_neighbors=int(args.k_neighbors),
        max_stars=80,
        region_mask=center_mask,
    )
    edge_s = _sample_triangles(
        matched,
        x_cen=cx,
        y_cen=cy,
        plate_scale=ps,
        k_neighbors=int(args.k_neighbors),
        max_stars=80,
        region_mask=edge_mask,
    )
    center_sum = _summarize(center_s)
    edge_sum = _summarize(edge_s)
    all_s = center_s + edge_s
    idx_stats = _index_lookup_stats(
        all_s,
        index_path=args.index,
        truth_ra=float(args.truth_ra),
        truth_dec=float(args.truth_dec),
    )
    rec = _recommendation(center_sum, edge_sum, idx_stats)

    flux = matched["flux"].to_numpy(dtype=np.float64)
    gmag = matched["g_mag"].to_numpy(dtype=np.float64)

    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "draft": args.draft,
        "masterstar": str(ms),
        "wcs_crval": [crval_ra, crval_dec],
        "plate_scale_arcsec_px": ps,
        "fov_deg": fov_deg,
        "n_dao_total": int(n_dao),
        "n_gaia_cone": len(gaia_rows),
        "n_matched": int(len(matched)),
        "match_frac_dao": float(len(matched) / max(n_dao, 1)),
        "match_px_med": float(np.median(matched["match_px"])),
        "flux_matched_p10": float(np.percentile(flux, 10)) if len(flux) else None,
        "flux_matched_med": float(np.median(flux)) if len(flux) else None,
        "g_mag_matched_med": float(np.nanmedian(gmag)) if len(gmag) else None,
        "n_bright_g_le_12": int(np.sum(np.isfinite(gmag) & (gmag <= 12))),
        "center": center_sum,
        "edge": edge_sum,
        "index_vote_from_true_sky_triangles": idx_stats,
        "recommendation": rec,
    }

    out = args.out or (
        draft_dir / "diag" / "blind_solver" / "wide_true_triangle_shape.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
