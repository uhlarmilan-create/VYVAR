from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    phot_dir = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000278\platesolve\NoFilter_60_2\photometry")

    # 1) comparison_stars_per_target.csv rows for BO CVn
    comp = pd.read_csv(
        phot_dir / "comparison_stars_per_target.csv",
        dtype={"catalog_id": str, "name": str, "target_catalog_id": str},  # Gaia ID musí byť str — float64 stráca cifry
    )
    bo_cid = "1498613634033133184"
    bo_comp = comp[comp["target_catalog_id"] == bo_cid]
    print(f"comparison_stars_per_target.csv — BO CVn riadky: {len(bo_comp)}")

    cols = [c for c in ["catalog_id", "mag", "b_v", "tier", "comp_tier", "status"] if c in bo_comp.columns]
    if cols:
        print(bo_comp[cols].to_string(index=False))
    else:
        print(bo_comp.to_string(index=False))

    # 2) photometry_summary
    summ = pd.read_csv(
        phot_dir / "photometry_summary.csv",
        dtype={"catalog_id": str, "name": str},  # Gaia ID musí byť str — float64 stráca cifry
    )
    bo_summ = summ[summ["catalog_id"] == bo_cid]
    print(f"\nphotometry_summary — n_good_comp: {bo_summ['n_good_comp'].values}")
    print(f"lc_rms: {bo_summ['lc_rms'].values}")

    # 3) lightcurve
    lc = pd.read_csv(phot_dir / "lightcurves" / f"lightcurve_{bo_cid}.csv")
    normal = (lc["flag"].astype(str) == "normal").sum()
    print(f"\nlightcurve — n_frames: {len(lc)}, normal: {normal}")


if __name__ == "__main__":
    main()

