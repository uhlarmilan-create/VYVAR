"""One-off comp consistency diagnostic: draft_321 vs draft_348."""
import numpy as np
import pandas as pd

phot321 = r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000321\platesolve\NoFilter_60_2\photometry"
phot348 = r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000348\platesolve\NoFilter_60_2\photometry"

cs321 = pd.read_csv(
    phot321 + r"\comparison_stars_per_target.csv",
    dtype={"catalog_id": str, "target_catalog_id": str},
)
cs348 = pd.read_csv(
    phot348 + r"\comparison_stars_per_target.csv",
    dtype={"catalog_id": str, "target_catalog_id": str},
)
s321 = pd.read_csv(phot321 + r"\photometry_summary.csv", dtype={"catalog_id": str})
s348 = pd.read_csv(phot348 + r"\photometry_summary.csv", dtype={"catalog_id": str})

print(f"draft_321 summary: {len(s321)} targets")
print(f"draft_348 summary: {len(s348)} targets")

ids321 = set(s321["catalog_id"].astype(str))
ids348 = set(s348["catalog_id"].astype(str))
common = ids321 & ids348
print(f"\nCommon targets in both summaries: {len(common)}")

same_comps = []
diff_comps = []
only_321 = []
only_348 = []

for tid in common:
    c321 = set(cs321[cs321["target_catalog_id"].astype(str) == tid]["catalog_id"].astype(str))
    c348 = set(cs348[cs348["target_catalog_id"].astype(str) == tid]["catalog_id"].astype(str))

    if not c321 and not c348:
        continue
    if c321 == c348:
        same_comps.append(tid)
    elif c321 and c348:
        diff_comps.append(
            {
                "target_id": tid,
                "n_321": len(c321),
                "n_348": len(c348),
                "only_in_321": len(c321 - c348),
                "only_in_348": len(c348 - c321),
                "shared": len(c321 & c348),
            }
        )
    elif c321 and not c348:
        only_321.append(tid)
    else:
        only_348.append(tid)

print("\n=== Comp assignment comparison (common targets) ===")
print(f"Identical comp sets:     {len(same_comps)}")
print(f"Different comp sets:     {len(diff_comps)}")
print(f"Comps only in 321:       {len(only_321)}")
print(f"Comps only in 348:       {len(only_348)}")

if diff_comps:
    df_diff = pd.DataFrame(diff_comps)
    print("\nDifferent comp sets detail:")
    print(df_diff.to_string())
    print(
        f"\nShared comps: mean={df_diff['shared'].mean():.1f}, "
        f"min={df_diff['shared'].min()}, max={df_diff['shared'].max()}"
    )

print("\n=== RMS impact of different comp sets ===")
for label, tids in [
    ("same comps", same_comps),
    ("diff comps", [d["target_id"] for d in diff_comps]),
]:
    rms_321 = []
    rms_348 = []
    for tid in tids:
        try:
            r321 = float(s321[s321["catalog_id"].astype(str) == tid]["lc_rms"].values[0])
            r348 = float(s348[s348["catalog_id"].astype(str) == tid]["lc_rms"].values[0])
            if pd.notna(r321) and pd.notna(r348):
                rms_321.append(r321)
                rms_348.append(r348)
        except Exception:
            pass
    if rms_321:
        print(f"\n{label} ({len(rms_321)} targets):")
        print(f"  321 median RMS: {np.median(rms_321):.4f}")
        print(f"  348 median RMS: {np.median(rms_348):.4f}")
        print(
            f"  348 worse than 321: "
            f"{sum(r48 > r21 for r48, r21 in zip(rms_348, rms_321))}/{len(rms_321)}"
        )
