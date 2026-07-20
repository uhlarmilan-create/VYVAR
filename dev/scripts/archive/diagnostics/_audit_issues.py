import pandas as pd
import os

print("=== Issue 2: BO CVn ===")
base = r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000346"
phot = base + r"\platesolve\NoFilter_60_2\photometry"

at = pd.read_csv(phot + r"\active_targets.csv")
print("=== active_targets zone_flag ===")
print(at["zone_flag"].value_counts())

mask = pd.Series(False, index=at.index)
if "name" in at.columns:
    mask |= at["name"].astype(str).str.contains("BO", case=False, na=False)
if "vsx_name" in at.columns:
    mask |= at["vsx_name"].astype(str).str.contains("BO CVn", case=False, na=False)
bo = at[mask]
print("\n=== BO CVn in active_targets ===")
cols = [c for c in ["name", "vsx_name", "zone_flag", "skip_photometry", "mag", "catalog_id"] if c in at.columns]
print(bo[cols].to_string() if len(bo) else "NOT FOUND")

s = pd.read_csv(phot + r"\photometry_summary.csv")
mask_s = pd.Series(False, index=s.index)
if "vsx_name" in s.columns:
    mask_s |= s["vsx_name"].astype(str).str.contains("BO", case=False, na=False)
if "name" in s.columns:
    mask_s |= s["name"].astype(str).str.contains("BO", case=False, na=False)
bo_s = s[mask_s]
print("\n=== BO CVn in summary ===")
print(bo_s.to_string() if len(bo_s) > 0 else "NOT FOUND")

csp = phot + r"\comparison_stars_per_target.csv"
if os.path.exists(csp):
    cs = pd.read_csv(csp)
    if "target_vsx_name" in cs.columns:
        bo_cs = cs[cs["target_vsx_name"].astype(str).str.contains("BO", case=False, na=False)]
    elif "target_name" in cs.columns:
        bo_cs = cs[cs["target_name"].astype(str).str.contains("BO", case=False, na=False)]
    elif len(bo) and "target_catalog_id" in cs.columns:
        bo_cs = cs[cs["target_catalog_id"].astype(str) == str(bo.iloc[0]["catalog_id"])]
    else:
        bo_cs = cs.iloc[0:0]
    print(f"\n=== BO CVn comp assignments ({len(bo_cs)} rows) ===")
    print(bo_cs.head(20).to_string() if len(bo_cs) > 0 else "NO COMP ASSIGNMENTS")

s321 = pd.read_csv(
    r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000321\platesolve\NoFilter_60_2\photometry\photometry_summary.csv"
)
bo321 = s321[s321["vsx_name"].astype(str).str.contains("BO CVn", case=False, na=False)]
if len(bo321):
    cid = str(int(float(bo321.iloc[0]["catalog_id"])))
    print(f"\nBO CVn catalog_id from draft_321: {cid}")
    in_at = at[at["catalog_id"].astype(str) == cid]
    print("In draft_346 active_targets:")
    print(in_at[cols].to_string() if len(in_at) else "NOT IN active_targets")
    in_s = s[s["catalog_id"].astype(str) == cid]
    print("In draft_346 summary:")
    print(in_s.to_string() if len(in_s) else "NOT IN summary")

print("\n=== Issue 3: zone_flag 321 vs 346 ===")
at321 = pd.read_csv(
    r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000321\platesolve\NoFilter_60_2\photometry\active_targets.csv"
)
at346 = pd.read_csv(phot + r"\active_targets.csv")

print("=== draft_321 active_targets zone_flag ===")
print(at321["zone_flag"].value_counts())
print(f"total: {len(at321)}")

print("\n=== draft_346 active_targets zone_flag ===")
print(at346["zone_flag"].value_counts())
print(f"total: {len(at346)}")

common = set(at321["catalog_id"].astype(str)) & set(at346["catalog_id"].astype(str))
print(f"\nCommon catalog_ids: {len(common)}")

merged = at321[["catalog_id", "zone_flag", "mag"]].merge(
    at346[["catalog_id", "zone_flag"]].rename(columns={"zone_flag": "zone_346"}),
    on="catalog_id",
    how="inner",
)
changed = merged[merged["zone_flag"] != merged["zone_346"]]
print(f"\nStars that changed zone_flag (321->346): {len(changed)}")
if len(changed):
    print("\nTransition counts (zone_321 -> zone_346):")
    print(changed.groupby(["zone_flag", "zone_346"]).size().sort_values(ascending=False))
    print("\nFirst 25 changed rows:")
    print(changed.head(25).to_string())

# lc_source on 346 summary
if "lc_source" in s.columns:
    print("\n=== draft_346 photometry_summary lc_source ===")
    print(s["lc_source"].value_counts())
