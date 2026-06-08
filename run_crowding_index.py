"""Runner for the parallel depth-aware crowding index (no pipeline wire-up).

Runs crowding_index.compute_crowding_index over a fixed draft list, writes
crowding_index.json + crowding_targets.csv per draft, and prints one cross-draft
summary table. Read-only against the pipeline; only writes the two new artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import AppConfig
from crowding_index import compute_crowding_index
from database import VyvarDatabase, get_gaia_db_max_g_mag

DRAFTS = [311, 321, 358, 359, 360, 361, 362]
SETUP = "NoFilter_60_2"
ARCHIVE = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts")


def main() -> None:
    cfg = AppConfig()
    db = VyvarDatabase(cfg.database_path)
    gaia_max_g = get_gaia_db_max_g_mag(cfg.gaia_db_path)
    print(f"Gaia DB max g_mag (catalog cap) = {gaia_max_g:.3f}\n")

    summary_rows = []
    for d in DRAFTS:
        draft_dir = ARCHIVE / f"draft_{d:06d}"
        ps = draft_dir / "platesolve" / SETUP
        if not (ps / "MASTERSTAR.fits").is_file():
            print(f"[draft {d}] missing MASTERSTAR.fits — skip")
            continue
        try:
            res, tgt = compute_crowding_index(draft_dir, SETUP, db, d, gaia_db_max_g=gaia_max_g)
        except Exception as e:  # noqa: BLE001
            print(f"[draft {d}] ERROR: {e!r}")
            continue
        out_json = ps / "crowding_index.json"
        out_csv = ps / "crowding_targets.csv"
        out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
        if len(tgt):
            tgt.to_csv(out_csv, index=False)
        summary_rows.append(res)
        print(f"[draft {d}] frame_limit={res['frame_limit_mag']} eff={res['effective_limit']} "
              f"bottleneck={res['catalog_is_bottleneck']} dens={res['gaia_density_per_arcmin2']} "
              f"blend1={res['blend_frac_1fwhm']} comp={res['completeness_on_frame']} "
              f"thr_miss={res['threshold_miss_frac']} blend_miss={res['blend_miss_frac']} "
              f"-> {out_json}")

    if not summary_rows:
        print("No drafts processed.")
        return

    print("\n" + "=" * 140)
    print("CROSS-DRAFT SUMMARY")
    print("=" * 140)
    cols = [
        ("draft", "draft_id", ""),
        ("frame_lim", "frame_limit_mag", ""),
        ("cone_max_g", "cone_max_g_100k", ""),
        ("cat_bottleneck", "catalog_is_bottleneck", ""),
        ("gaia/arcmin2", "gaia_density_per_arcmin2", ""),
        ("blend_1fwhm", "blend_frac_1fwhm", ""),
        ("blend_2fwhm", "blend_frac_2fwhm", ""),
        ("compl_onframe", "completeness_on_frame", ""),
        ("thr_miss_frac", "threshold_miss_frac", ""),
        ("blend_miss_frac", "blend_miss_frac", ""),
        ("ceiling", "achievable_ceiling", ""),
    ]
    header = " | ".join(f"{h:>15}" for h, _, _ in cols)
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        line = " | ".join(f"{str(r.get(k)):>15}" for _, k, _ in cols)
        print(line)

    # also dump a tidy CSV next to the runner for convenience
    pd.DataFrame(summary_rows).to_csv(ARCHIVE.parent / "crowding_index_summary.csv", index=False)
    print(f"\nWrote cross-draft CSV: {ARCHIVE.parent / 'crowding_index_summary.csv'}")


if __name__ == "__main__":
    main()
