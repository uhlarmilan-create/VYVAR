"""
VYVAR Performance Benchmark
Measures wall-clock time for key pipeline stages.
Usage: python scripts/benchmark_pipeline.py --draft <draft_path> --setup <setup_name>
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

# Add the VYVAR module roots to path (src_py + dev) for standalone execution.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)

logging.basicConfig(level=logging.WARNING)  # suppress pipeline logs during bench

RESULTS: list[tuple[str, float]] = []


@contextmanager
def bench(name: str):
    """Context manager to time a block."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    RESULTS.append((name, elapsed))
    print(f"  {name:<55} {elapsed:7.3f}s")


def run_benchmarks(draft_path: str, setup: str) -> None:
    draft_path = str(Path(draft_path).expanduser().resolve())
    proc_dir = os.path.join(draft_path, "detrended_aligned", "lights", setup)
    phot_dir = os.path.join(draft_path, "platesolve", setup, "photometry")
    ms_fits = os.path.join(draft_path, "platesolve", setup, "MASTERSTAR.fits")

    csv_files = sorted(glob.glob(os.path.join(proc_dir, "proc_*.csv")))
    n_frames = len(csv_files)
    if n_frames == 0:
        print(f"ERROR: No proc_*.csv found in {proc_dir}")
        return

    print(f"\n{'=' * 70}")
    print("VYVAR Performance Benchmark")
    print(f"Draft:  {draft_path}")
    print(f"Setup:  {setup}")
    print(f"Frames: {n_frames}")
    print(f"{'=' * 70}\n")

    # ------------------------------------------------------------------ #
    # 1. CSV I/O
    # ------------------------------------------------------------------ #
    print("[ 1. CSV I/O ]")

    from gaia_catalog_id import read_vyvar_csv

    with bench("Read 1 proc_*.csv (cold)"):
        df1 = read_vyvar_csv(csv_files[0])

    with bench(f"Read all {n_frames} proc_*.csv sequentially"):
        all_dfs = [read_vyvar_csv(f) for f in csv_files]

    total_rows = sum(len(d) for d in all_dfs)
    total_mb = sum(os.path.getsize(f) for f in csv_files) / 1e6
    print(f"  {'Total rows / MB':<55} {total_rows:,} rows / {total_mb:.1f} MB")

    pivot = pd.DataFrame()
    with bench("Build flux pivot (stars × frames)"):
        long = pd.concat(
            [
                d[["catalog_id", "dao_flux"]].assign(frame=i)
                for i, d in enumerate(all_dfs)
                if "dao_flux" in d.columns and "catalog_id" in d.columns
            ],
            ignore_index=True,
        )
        pivot = long.pivot_table(
            index="catalog_id",
            columns="frame",
            values="dao_flux",
            aggfunc="first",
        )
    print(f"  {'Pivot shape':<55} {pivot.shape}")

    # ------------------------------------------------------------------ #
    # 2. FITS I/O
    # ------------------------------------------------------------------ #
    print("\n[ 2. FITS I/O ]")

    fits_files = sorted(
        glob.glob(os.path.join(draft_path, "processed", "lights", setup, "proc_*.fits"))
    )
    if fits_files:
        from astropy.io import fits

        with bench("Read 1 FITS header only"):
            with fits.open(fits_files[0]) as h:
                _ = h[0].header.copy()

        with bench("Read 1 FITS full data"):
            _ = fits.getdata(fits_files[0])

        if os.path.exists(ms_fits):
            with bench("Read MASTERSTAR.fits (header + data)"):
                with fits.open(ms_fits, memmap=False) as h:
                    _ = h[0].header.copy()
                    _ = h[0].data.copy() if h[0].data is not None else None
    else:
        print("  (no processed FITS found — skipping)")

    # ------------------------------------------------------------------ #
    # 3. Gaia DB
    # ------------------------------------------------------------------ #
    print("\n[ 3. Gaia DB lookup ]")

    from config import AppConfig

    cfg = AppConfig()

    active_csv = os.path.join(phot_dir, "active_targets.csv")
    if os.path.exists(active_csv) and os.path.exists(str(cfg.gaia_db_path)):
        from gaia_catalog_id import normalize_gaia_source_id

        at = read_vyvar_csv(active_csv)
        cids = [
            normalize_gaia_source_id(x)
            for x in at["catalog_id"].dropna().tolist()[:50]
            if normalize_gaia_source_id(x)
        ]

        with bench(f"Per-target SQLite (N={min(len(cids), 10)} individual connects)"):
            import sqlite3

            for cid in cids[:10]:
                sid = int(cid) if str(cid).isdigit() else cid
                with sqlite3.connect(cfg.gaia_db_path, timeout=5) as conn:
                    conn.execute(
                        "SELECT bp_rp FROM gaia_dr3 WHERE source_id=? LIMIT 1;",
                        (sid,),
                    ).fetchone()

        rows: list = []
        with bench(f"Batch SQLite (N={len(cids)} targets, 1 connect)"):
            import sqlite3

            ids_int = [int(x) for x in cids if str(x).isdigit()]
            if ids_int:
                placeholders = ",".join("?" * len(ids_int))
                with sqlite3.connect(cfg.gaia_db_path, timeout=5) as conn:
                    rows = conn.execute(
                        f"SELECT source_id, bp_rp FROM gaia_dr3 "
                        f"WHERE source_id IN ({placeholders})",
                        ids_int,
                    ).fetchall()
        print(f"  {'Batch returned':<55} {len(rows)} rows")
    else:
        print("  (active_targets.csv or Gaia DB not found — skipping)")

    # ------------------------------------------------------------------ #
    # 4. Phase 2A simulation
    # ------------------------------------------------------------------ #
    print("\n[ 4. Phase 2A simulation ]")

    comp_csv = os.path.join(phot_dir, "comparison_stars_per_target.csv")
    summary_csv = os.path.join(phot_dir, "photometry_summary.csv")

    if os.path.exists(comp_csv) and os.path.exists(summary_csv) and not pivot.empty:
        summary_df = read_vyvar_csv(summary_csv)
        t_n = len(summary_df)
        target_cids = summary_df["catalog_id"].dropna().tolist()[:20]

        with bench(f"Per-target flux lookup — Python loop (T={min(t_n, 20)}, N={n_frames})"):
            for cid in target_cids:
                for df in all_dfs:
                    if "catalog_id" not in df.columns or "dao_flux" not in df.columns:
                        continue
                    row = df[df["catalog_id"] == cid]
                    if not row.empty:
                        _ = row["dao_flux"].iloc[0]

        with bench(f"Per-target flux lookup — pivot slice (T={min(t_n, 20)})"):
            for cid in target_cids:
                if cid in pivot.index:
                    _ = pivot.loc[cid].values
    else:
        print("  (comparison_stars or summary CSV not found — skipping)")

    # ------------------------------------------------------------------ #
    # 5. Variability matrix
    # ------------------------------------------------------------------ #
    print("\n[ 5. Variability matrix ]")

    with bench("load_field_flux_matrix from disk (fresh read)"):
        try:
            from variability_detector import load_field_flux_matrix

            mat, meta, _bjd = load_field_flux_matrix(Path(proc_dir))
            print(f"  {'Matrix shape':<55} {mat.shape if hasattr(mat, 'shape') else 'n/a'}")
        except Exception as exc:  # noqa: BLE001
            print(f"  (skipped: {exc})")

    # Test cache path speedup (TODO-PERF-6 — same as Phase 2A after RUN VYVAR)
    with bench("load_field_flux_matrix WITH cache (simulated)"):
        try:
            cache = {str(csv_files[i]): all_dfs[i] for i in range(len(csv_files))}
            from variability_detector import load_field_flux_matrix

            mat2, meta2, _bjd2 = load_field_flux_matrix(
                Path(proc_dir),
                flux_pivot=pivot,
                csv_cache=cache,
            )
            print(f"  {'Matrix shape (cached)':<55} {mat2.shape if hasattr(mat2, 'shape') else 'n/a'}")
        except Exception as exc:  # noqa: BLE001
            print(f"  (skipped: {exc})")

    if not pivot.empty:
        with bench("Variability matrix from pre-built pivot (reuse)"):
            _ = pivot.values.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    total = sum(t for _, t in RESULTS)
    for name, t in RESULTS:
        pct = 100 * t / total if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {name:<55} {t:7.3f}s  {pct:5.1f}%  {bar}")
    print(f"\n  {'TOTAL':<55} {total:7.3f}s")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VYVAR Performance Benchmark")
    parser.add_argument("--draft", required=True, help="Path to draft directory")
    parser.add_argument("--setup", required=True, help="Setup name (e.g. NoFilter_60_2)")
    args = parser.parse_args()
    run_benchmarks(args.draft, args.setup)
