"""VYVAR script #2 — Gaia blind triangle index (fine + wide PKL tiers).

Builds density-matched triangle hash indexes from local ``gaia_dr3`` SQLite.
Default run produces ``gaia_triangles_fine.pkl`` and ``gaia_triangles_wide.pkl``.

Examples:
  python GAIA_DR3/build_blind_index.py --help
  python GAIA_DR3/build_blind_index.py --tier fine --out GAIA_DR3/_smoke.pkl \\
      --dec-min 89 --dec-max 90 --mag-limit 14 --stars-per-cell 10
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import pickle
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm

_GAIA_DIR = Path(__file__).resolve().parent

MIN_RATIO = 0.15
K_NEIGHBORS_DEFAULT = 8
TOLERANCE_UPPER_DEFAULT = 0.002
DEC_BAND_DEG_DEFAULT = 1.0

DEFAULT_FINE_OUT = _GAIA_DIR / "gaia_triangles_fine.pkl"
DEFAULT_WIDE_OUT = _GAIA_DIR / "gaia_triangles_wide.pkl"
DEFAULT_DB = _GAIA_DIR / "vyvar_gaia_dr3.db"

FINE_CELL_DEG = 1.0
FINE_STARS_PER_CELL = 95
WIDE_CELL_DEG = 2.0
WIDE_STARS_PER_CELL = 95
DEFAULT_MAG_LIMIT = 14.0


def assign_cell_bins(
    ra: np.ndarray,
    dec: np.ndarray,
    cell_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Dec bands + RA bins scaled by cos(dec) (no extra dependencies)."""
    cell = float(cell_deg)
    if not np.isfinite(cell) or cell <= 0:
        raise ValueError(f"cell_deg must be positive, got {cell_deg!r}")
    dec_bin = np.floor((dec + 90.0) / cell).astype(np.int64)
    cos_dec = np.cos(np.radians(dec))
    ra_width = cell / np.maximum(cos_dec, 0.1)
    ra_bin = np.floor(ra / ra_width).astype(np.int64)
    return dec_bin, ra_bin


def cap_brightest_per_cell(
    df: pd.DataFrame,
    *,
    cell_deg: float,
    stars_per_cell: int,
    mag_col: str = "g_mag",
) -> pd.DataFrame:
    """Keep brightest ``stars_per_cell`` rows per sky cell (smallest ``g_mag``)."""
    if df.empty:
        return df.copy()
    n = max(1, int(stars_per_cell))
    dec_bin, ra_bin = assign_cell_bins(
        df["ra"].to_numpy(dtype=np.float64),
        df["dec"].to_numpy(dtype=np.float64),
        cell_deg,
    )
    work = df.copy()
    work["_dec_bin"] = dec_bin
    work["_ra_bin"] = ra_bin
    capped = (
        work.sort_values(mag_col, ascending=True)
        .groupby(["_dec_bin", "_ra_bin"], sort=False)
        .head(n)
        .drop(columns=["_dec_bin", "_ra_bin"])
    )
    return capped.reset_index(drop=True)


def triangle_hash_batch(
    pts: np.ndarray,
    *,
    min_ratio: float = MIN_RATIO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized 3D triangle hash for ``pts`` shape (m, 3, 3). Returns mask, r1, r2, log_L3."""
    p0, p1, p2 = pts[:, 0], pts[:, 1], pts[:, 2]
    d01 = np.linalg.norm(p0 - p1, axis=1)
    d12 = np.linalg.norm(p1 - p2, axis=1)
    d02 = np.linalg.norm(p0 - p2, axis=1)
    sides = np.sort(np.stack([d01, d12, d02], axis=1), axis=1)
    L1, L2, L3 = sides[:, 0], sides[:, 1], sides[:, 2]
    ok = (L3 >= 1e-8) & (min_ratio <= L1 / L3)
    L3_arcsec = L3 * (180.0 / math.pi) * 3600.0
    ok &= L3_arcsec >= 0.1
    r1 = np.zeros_like(L3)
    r2 = np.zeros_like(L3)
    log_l3 = np.zeros_like(L3)
    if np.any(ok):
        r1[ok] = L1[ok] / L3[ok]
        r2[ok] = L2[ok] / L3[ok]
        log_l3[ok] = np.log10(L3_arcsec[ok])
    return ok, r1, r2, log_l3


def triangle_hash(p0, p1, p2, *, min_ratio: float = MIN_RATIO) -> tuple[float, float, float] | None:
    d01 = math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2)
    d12 = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
    d02 = math.sqrt((p0[0] - p2[0]) ** 2 + (p0[1] - p2[1]) ** 2 + (p0[2] - p2[2]) ** 2)
    sides = sorted([d01, d12, d02])
    L1, L2, L3 = sides
    if L3 < 1e-8:
        return None
    r1, r2 = L1 / L3, L2 / L3
    if r1 < min_ratio:
        return None
    L3_arcsec = L3 * (180.0 / math.pi) * 3600.0
    if L3_arcsec < 0.1:
        return None
    return (float(r1), float(r2), float(math.log10(L3_arcsec)))


def log_l3_distribution(log_l3: np.ndarray) -> dict[str, float]:
    arr = np.asarray(log_l3, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "min": float("nan"),
            "p10": float("nan"),
            "med": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
            "min_arcsec": float("nan"),
            "med_arcsec": float("nan"),
            "max_arcsec": float("nan"),
        }
    return {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "med": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "min_arcsec": float(10 ** np.min(arr)),
        "med_arcsec": float(10 ** np.median(arr)),
        "max_arcsec": float(10 ** np.max(arr)),
    }


def load_capped_stars_dec_bands(
    db_path: str,
    *,
    mag_limit: float,
    cell_deg: float,
    stars_per_cell: int,
    dec_band_deg: float = DEC_BAND_DEG_DEFAULT,
    dec_min: float = -90.0,
    dec_max: float = 90.0,
) -> pd.DataFrame:
    """Memory-safe: query Gaia in Dec strips, cap per cell, concatenate."""
    conn = sqlite3.connect(db_path)
    chunks: list[pd.DataFrame] = []
    dec = float(dec_min)
    dec_hi_cap = float(dec_max)
    n_raw_total = 0
    while dec < dec_hi_cap:
        dec_hi = min(dec_hi_cap, dec + float(dec_band_deg))
        q = (
            "SELECT ra, dec, g_mag FROM gaia_dr3 "
            "WHERE g_mag <= ? AND dec >= ? AND dec < ?"
        )
        band = pd.read_sql_query(q, conn, params=(mag_limit, dec, dec_hi))
        n_raw_total += len(band)
        if not band.empty:
            chunks.append(
                cap_brightest_per_cell(
                    band,
                    cell_deg=cell_deg,
                    stars_per_cell=stars_per_cell,
                    mag_col="g_mag",
                )
            )
        dec = dec_hi
    conn.close()
    if not chunks:
        return pd.DataFrame(columns=["ra", "dec", "g_mag"])
    out = pd.concat(chunks, ignore_index=True)
    out = out.drop_duplicates(subset=["ra", "dec"], keep="first")
    print(
        f"  Dec-band load: raw={n_raw_total:,} capped={len(out):,} "
        f"(mag<={mag_limit}, cell={cell_deg}, N/cell={stars_per_cell}, "
        f"dec={dec_min}..{dec_max})"
    )
    return out


def build_triangle_index(
    df: pd.DataFrame,
    *,
    k_neighbors: int = K_NEIGHBORS_DEFAULT,
    tolerance: float = TOLERANCE_UPPER_DEFAULT,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Build hash tree + metadata; return (index_data, log_L3 stats on physical dex)."""
    n_stars = len(df)
    if n_stars < 3:
        raise ValueError(f"Need >= 3 stars after capping, got {n_stars}")

    ra_rad = np.radians(df["ra"].values)
    dec_rad = np.radians(df["dec"].values)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    coords_search = np.column_stack([x, y, z]).astype(np.float64)
    coords_real = df[["ra", "dec"]].values.astype(np.float32)

    print("KDTree susedov...")
    neighbor_tree = KDTree(coords_search)
    k = min(int(k_neighbors), n_stars)
    _q = neighbor_tree.query(coords_search, k=k)
    if isinstance(_q, tuple):
        _idx = np.asarray(_q[1], dtype=np.int64)
    else:
        _idx = np.asarray(_q, dtype=np.int64)
    if _idx.ndim == 1:
        _idx = _idx.reshape(1, -1) if n_stars == 1 else _idx.reshape(-1, 1)
    all_indices = _idx
    combos = list(itertools.combinations(range(k), 3))
    owners = np.arange(n_stars, dtype=np.int64)

    hash_chunks: list[np.ndarray] = []
    meta_chunks: list[np.ndarray] = []

    print(f"Trojuholniky (k={k}, vectorized, {len(combos)} combos/star)...")
    for c in tqdm(combos, desc="combos"):
        i_tri = all_indices[:, list(c)]
        keep = owners == np.min(i_tri, axis=1)
        if not np.any(keep):
            continue
        tri_idx = i_tri[keep]
        pts = coords_search[tri_idx]
        ok, r1, r2, log_l3 = triangle_hash_batch(pts)
        if not np.any(ok):
            continue
        cr = coords_real[tri_idx[ok]]
        ra_c = cr[:, :, 0].mean(axis=1)
        dec_c = cr[:, :, 1].mean(axis=1)
        meta = np.column_stack(
            [
                ra_c,
                dec_c,
                cr[:, 0, 0],
                cr[:, 0, 1],
                cr[:, 1, 0],
                cr[:, 1, 1],
                cr[:, 2, 0],
                cr[:, 2, 1],
            ]
        ).astype(np.float32)
        hashes = np.column_stack([r1[ok], r2[ok], log_l3[ok]]).astype(np.float32)
        hash_chunks.append(hashes)
        meta_chunks.append(meta)

    if not hash_chunks:
        raise ValueError("No triangles generated")
    hashes_arr = np.vstack(hash_chunks)
    metadata_arr = np.vstack(meta_chunks)
    log_l3_samples = hashes_arr[:, 2].copy()
    l3_stats = log_l3_distribution(np.asarray(log_l3_samples))

    log_L3_min = float(hashes_arr[:, 2].min())
    log_L3_max = float(hashes_arr[:, 2].max())
    log_L3_range = max(log_L3_max - log_L3_min, 1e-6)
    hashes_arr[:, 2] = (hashes_arr[:, 2] - log_L3_min) / log_L3_range

    print("Final 3D hash-tree...")
    hash_tree = KDTree(hashes_arr)

    index_data: dict[str, Any] = {
        "tree": hash_tree,
        "metadata": metadata_arr,
        "tolerance": float(tolerance),
        "hash_dim": 3,
        "log_L3_min": log_L3_min,
        "log_L3_max": log_L3_max,
        "k_neighbors": int(k_neighbors),
        "n_stars": int(n_stars),
        "n_triangles": int(len(hashes_arr)),
    }
    return index_data, l3_stats


def build_and_save(
    *,
    db_path: str,
    output_pkl: str,
    mag_limit: float,
    cell_deg: float,
    stars_per_cell: int,
    k_neighbors: int = K_NEIGHBORS_DEFAULT,
    tolerance: float = TOLERANCE_UPPER_DEFAULT,
    dec_band_deg: float = DEC_BAND_DEG_DEFAULT,
    dec_min: float = -90.0,
    dec_max: float = 90.0,
) -> dict[str, Any]:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    t0 = time.time()
    print(f"{'=' * 60}\n VYVAR Blind Index Build\n{'=' * 60}")
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)

    df = load_capped_stars_dec_bands(
        db_path,
        mag_limit=mag_limit,
        cell_deg=cell_deg,
        stars_per_cell=stars_per_cell,
        dec_band_deg=dec_band_deg,
        dec_min=dec_min,
        dec_max=dec_max,
    )
    index_core, l3_stats = build_triangle_index(df, k_neighbors=k_neighbors, tolerance=tolerance)
    index_data = {
        **index_core,
        "mag_limit": float(mag_limit),
        "cell_deg": float(cell_deg),
        "stars_per_cell": int(stars_per_cell),
        "target_density_deg2": float(stars_per_cell) / max(float(cell_deg) ** 2, 1e-12),
    }

    with open(output_pkl, "wb") as f:
        pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    pkl_mb = os.path.getsize(output_pkl) / (1024 * 1024)
    elapsed = time.time() - t0
    summary = {
        "output_pkl": output_pkl,
        "elapsed_s": elapsed,
        "pkl_mb": pkl_mb,
        "stars_total": int(index_data["n_stars"]),
        "n_triangles": int(index_data["n_triangles"]),
        "mag_limit": mag_limit,
        "cell_deg": cell_deg,
        "stars_per_cell": stars_per_cell,
        "k_neighbors": k_neighbors,
        "log_L3_index": l3_stats,
    }
    print(
        f"OK {os.path.basename(output_pkl)} | {elapsed:.1f}s | "
        f"stars={summary['stars_total']:,} tri={summary['n_triangles']:,} | {pkl_mb:.1f} MB"
    )
    print(
        f"   log_L3 index (dex): min={l3_stats['min']:.3f} p10={l3_stats['p10']:.3f} "
        f"med={l3_stats['med']:.3f} p90={l3_stats['p90']:.3f} max={l3_stats['max']:.3f}"
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Gaia blind triangle index PKL (fine/wide tiers).")
    p.add_argument("--db", type=Path, default=DEFAULT_DB, help="Gaia SQLite with gaia_dr3 table.")
    p.add_argument(
        "--tier",
        choices=("fine", "wide", "both"),
        default="both",
        help="Which tier(s) to build (default: both fine + wide).",
    )
    p.add_argument("--mag-limit", type=float, default=DEFAULT_MAG_LIMIT)
    p.add_argument("--fine-out", type=Path, default=DEFAULT_FINE_OUT)
    p.add_argument("--wide-out", type=Path, default=DEFAULT_WIDE_OUT)
    p.add_argument("--out", type=Path, default=None, help="Single-tier output (overrides --fine-out/--wide-out).")
    p.add_argument("--cell-deg", type=float, default=None, help="Override cell size for single-tier build.")
    p.add_argument(
        "--stars-per-cell",
        type=int,
        default=None,
        help="Override stars/cell for single-tier build.",
    )
    p.add_argument("--fine-cell-deg", type=float, default=FINE_CELL_DEG)
    p.add_argument("--fine-stars-per-cell", type=int, default=FINE_STARS_PER_CELL)
    p.add_argument("--wide-cell-deg", type=float, default=WIDE_CELL_DEG)
    p.add_argument("--wide-stars-per-cell", type=int, default=WIDE_STARS_PER_CELL)
    p.add_argument("--dec-min", type=float, default=-90.0)
    p.add_argument("--dec-max", type=float, default=90.0)
    p.add_argument("--k-neighbors", type=int, default=K_NEIGHBORS_DEFAULT)
    p.add_argument("--tolerance", type=float, default=TOLERANCE_UPPER_DEFAULT)
    p.add_argument("--dec-band-deg", type=float, default=DEC_BAND_DEG_DEFAULT)
    return p.parse_args(argv)


def _build_one_tier(
    *,
    db_path: Path,
    out_path: Path,
    mag_limit: float,
    cell_deg: float,
    stars_per_cell: int,
    dec_min: float,
    dec_max: float,
    k_neighbors: int,
    tolerance: float,
    dec_band_deg: float,
) -> dict[str, Any]:
    return build_and_save(
        db_path=str(db_path),
        output_pkl=str(out_path),
        mag_limit=mag_limit,
        cell_deg=cell_deg,
        stars_per_cell=stars_per_cell,
        k_neighbors=k_neighbors,
        tolerance=tolerance,
        dec_band_deg=dec_band_deg,
        dec_min=dec_min,
        dec_max=dec_max,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        raise SystemExit(f"Gaia DB missing: {db_path}")

    tier = str(args.tier)
    common = {
        "db_path": db_path,
        "mag_limit": float(args.mag_limit),
        "dec_min": float(args.dec_min),
        "dec_max": float(args.dec_max),
        "k_neighbors": int(args.k_neighbors),
        "tolerance": float(args.tolerance),
        "dec_band_deg": float(args.dec_band_deg),
    }

    if tier in ("fine", "wide"):
        out = Path(args.out) if args.out else (args.fine_out if tier == "fine" else args.wide_out)
        if args.cell_deg is not None and args.stars_per_cell is not None:
            cell_deg = float(args.cell_deg)
            spc = int(args.stars_per_cell)
        elif tier == "fine":
            cell_deg = float(args.fine_cell_deg)
            spc = int(args.fine_stars_per_cell)
        else:
            cell_deg = float(args.wide_cell_deg)
            spc = int(args.wide_stars_per_cell)
        _build_one_tier(out_path=out.resolve(), cell_deg=cell_deg, stars_per_cell=spc, **common)
        return 0

    fine_out = Path(args.out or args.fine_out).resolve()
    wide_out = Path(args.wide_out).resolve()
    _build_one_tier(
        out_path=fine_out,
        cell_deg=float(args.fine_cell_deg),
        stars_per_cell=int(args.fine_stars_per_cell),
        **common,
    )
    _build_one_tier(
        out_path=wide_out,
        cell_deg=float(args.wide_cell_deg),
        stars_per_cell=int(args.wide_stars_per_cell),
        **common,
    )
    print(f"Done: {fine_out.name}, {wide_out.name}")
    print("Set blind_index_fine_path / blind_index_wide_path in config.json (Settings).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
