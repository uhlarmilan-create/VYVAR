import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from comp_selection_per_target import _angular_distance_deg_vectorized

ms = pd.read_csv(
    _ROOT / "Archive/Drafts/draft_000343/platesolve/NoFilter_60_2/masterstars_full_match.csv",
    dtype={"catalog_id": str, "name": str},
)
at = pd.read_csv(
    _ROOT
    / "Archive/Drafts/draft_000343/platesolve/NoFilter_60_2/photometry/active_targets.csv",
    dtype={"catalog_id": str},
)

cat_only = at[at["zone_flag"] == "catalog_only"][
    ["name", "vsx_name", "ra_deg", "dec_deg", "mag", "catalog_id"]
].copy()

print("catalog_only targets:", len(cat_only))
print("mag distribucia:")
print(cat_only["mag"].describe())

print("\nmasterstars columns:", ms.columns.tolist())
print("\nmasterstars zone dist:")
print(ms["zone"].value_counts() if "zone" in ms.columns else "no zone col")

MATCH_ARCSEC = 3.9  # 3 * phase01_plate_scale 1.3

ms["ra_deg"] = pd.to_numeric(ms["ra_deg"], errors="coerce")
ms["dec_deg"] = pd.to_numeric(ms["dec_deg"], errors="coerce")
cat_only["ra_deg"] = pd.to_numeric(cat_only["ra_deg"], errors="coerce")
cat_only["dec_deg"] = pd.to_numeric(cat_only["dec_deg"], errors="coerce")

ms_ok = ms[ms["ra_deg"].notna() & ms["dec_deg"].notna()]
ms_arr = ms_ok[["ra_deg", "dec_deg"]].to_numpy(dtype=float)

seps = []
for _, r in cat_only.iterrows():
    ra, de = float(r["ra_deg"]), float(r["dec_deg"])
    if not (math.isfinite(ra) and math.isfinite(de)):
        seps.append(float("nan"))
        continue
    d = _angular_distance_deg_vectorized(ra, de, ms_arr[:, 0], ms_arr[:, 1]) * 3600.0
    seps.append(float(np.min(d)))

cat_only["nearest_dao_arcsec"] = seps
s = cat_only["nearest_dao_arcsec"].dropna()

print(f"\n=== Nearest DAO distance (threshold match = {MATCH_ARCSEC}\") ===")
print(s.describe().round(3))
print("count <= 3.9 arcsec:", int((s <= MATCH_ARCSEC).sum()))
print("count 3.9-10 arcsec:", int(((s > MATCH_ARCSEC) & (s <= 10)).sum()))
print("count 10-30 arcsec:", int(((s > 10) & (s <= 30)).sum()))
print("count > 30 arcsec:", int((s > 30).sum()))

print("\nPercentiles (arcsec):")
for p in (5, 10, 25, 50, 75, 90, 95):
    print(f"  p{p}: {np.percentile(s, p):.2f}\"")

print("\nClosest 15 catalog_only to any DAO star:")
print(
    cat_only.nsmallest(15, "nearest_dao_arcsec")[
        ["vsx_name", "mag", "nearest_dao_arcsec"]
    ].to_string(index=False)
)
