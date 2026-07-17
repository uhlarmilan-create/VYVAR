from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _print_safe(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(str(msg).encode("ascii", "backslashreplace").decode("ascii"))


def main() -> None:
    phot_dir = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000278\platesolve\NoFilter_60_2\photometry")
    bo_cid = "1498613634033133184"

    # 1) comp_quality JSON
    json_path = phot_dir / "lightcurves" / f"comp_quality_{bo_cid}.json"
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        _print_safe(f"comp_quality JSON — keys: {list(data.keys())[:5]}")
        _print_safe(f"Count items in JSON: {len(data)}")
        for i, (k, v) in enumerate(data.items()):
            if i >= 5:
                break
            _print_safe(f"  {k}: {v}")
    else:
        _print_safe("comp_quality JSON missing")

    # 2) comparison_stars_per_target.csv
    comp = pd.read_csv(
        phot_dir / "comparison_stars_per_target.csv",
        dtype={"target_catalog_id": str, "catalog_id": str},
        low_memory=False,
    )
    bo_comp = comp[comp["target_catalog_id"] == bo_cid]
    ids = bo_comp["catalog_id"].astype(str).tolist() if "catalog_id" in bo_comp.columns else []
    _print_safe(f"\ncomparison_stars_per_target.csv IDs: {ids}")


if __name__ == "__main__":
    main()

