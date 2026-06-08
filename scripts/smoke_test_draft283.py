"""
VYVAR Smoke Test — draft_000283 border filter + proximity veto
Run: python scripts/smoke_test_draft283.py
"""

import json
import sys
from pathlib import Path

DRAFT = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000283")
PLATESOLVE = DRAFT / "platesolve" / "NoFilter_60_2"

PLAN = PLATESOLVE / "photometry_plan.json"
COMP_CSV = PLATESOLVE / "comparison_stars.csv"
ALIGNED_DIR = DRAFT / "detrended_aligned" / "lights" / "NoFilter_60_2"

OK = "✅"
FAIL = "❌"
WARN = "⚠️"

errors = 0


def check(label, condition, detail=""):
    global errors
    if condition:
        print(f"  {OK} {label}")
    else:
        print(f"  {FAIL} {label}{': ' + detail if detail else ''}")
        errors += 1


def warn(label, detail=""):
    print(f"  {WARN} {label}{': ' + detail if detail else ''}")


print("\n=== VYVAR Smoke Test — draft_000283 ===\n")

# 1. photometry_plan.json exists
print("[1] photometry_plan.json")
check("File exists", PLAN.exists())

if PLAN.exists():
    plan = json.loads(PLAN.read_text())

    safe_bbox = plan.get("safe_bbox_px")
    r_out = plan.get("safe_bbox_r_out_px")

    check(
        "safe_bbox_px present",
        safe_bbox is not None,
        "border filter did not compute bbox — RAM flush timing still broken?",
    )

    if safe_bbox is not None:
        x0, y0, x1, y1 = safe_bbox
        check("safe_bbox is valid (x1>x0, y1>y0)", x1 > x0 and y1 > y0, f"got {safe_bbox}")
        check(
            "safe_bbox not full frame (filter applied)",
            x0 > 0 or y0 > 0 or x1 < 2082 or y1 < 1397,
            f"bbox = {safe_bbox} — looks like no shrinkage was applied",
        )
        print(f"       safe_bbox_px = {safe_bbox}")
        print(f"       safe_bbox_r_out_px = {r_out}")


# 2. aligned frames on disk
print("\n[2] Aligned frames on disk")
aligned = sorted(ALIGNED_DIR.glob("proc_*.fits"))
check("Aligned dir exists", ALIGNED_DIR.exists())
check("At least 100 aligned frames", len(aligned) >= 100, f"found {len(aligned)}")


# 3. comparison_stars.csv — no star outside safe_bbox
print("\n[3] comparison_stars.csv — border filter applied")
check("File exists", COMP_CSV.exists())

if COMP_CSV.exists() and safe_bbox is not None:
    import csv

    x0, y0, x1, y1 = safe_bbox
    outside = []
    with open(COMP_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cx = float(row["x"])
                cy = float(row["y"])
                if cx < x0 or cx > x1 or cy < y0 or cy > y1:
                    outside.append((row.get("catalog_id", "?"), cx, cy))
            except (KeyError, ValueError):
                pass

    check("No comp stars outside safe_bbox", len(outside) == 0, f"{len(outside)} stars outside bbox: {outside[:3]}")


# 4. proximity veto — check comparison_stars has no VSX targets
print("\n[4] Proximity veto — comp ∩ variable_targets")
VAR_CSV = PLATESOLVE / "variable_targets.csv"

if COMP_CSV.exists() and VAR_CSV.exists():
    import csv

    def load_radec(path):
        result = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    result[row.get("catalog_id", "?")] = (float(row["ra_deg"]), float(row["dec_deg"]))
                except (KeyError, ValueError):
                    pass
        return result

    comp_radec = load_radec(COMP_CSV)
    var_radec = load_radec(VAR_CSV)

    from math import cos, radians, sqrt

    def sep_arcsec(ra1, dec1, ra2, dec2):
        d_ra = (ra1 - ra2) * cos(radians((dec1 + dec2) / 2)) * 3600
        d_dec = (dec1 - dec2) * 3600
        return sqrt(d_ra**2 + d_dec**2)

    contaminated = []
    for cid, (cra, cdec) in comp_radec.items():
        for vid, (vra, vdec) in var_radec.items():
            sep = sep_arcsec(cra, cdec, vra, vdec)
            if sep < 10:
                contaminated.append((cid, vid, round(sep, 2)))

    check(
        "No comp within 10 arcsec of VSX target",
        len(contaminated) == 0,
        f"{len(contaminated)} pairs: {contaminated[:3]}",
    )


# Summary
print(f"\n{'='*40}")
if errors == 0:
    print(f"{OK} All checks passed!")
else:
    print(f"{FAIL} {errors} check(s) FAILED")
print()

sys.exit(0 if errors == 0 else 1)

