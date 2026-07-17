from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


TARGET_CID = "1498613634033133184"  # BO CVn
DRAFT_DIR = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000278")


def _find_setup_dir(draft_dir: Path) -> Path:
    ps = draft_dir / "platesolve"
    if not ps.is_dir():
        raise FileNotFoundError(f"Missing platesolve directory: {ps}")
    for sub in sorted(ps.iterdir()):
        if not sub.is_dir():
            continue
        phot = sub / "photometry"
        if (phot / "active_targets.csv").is_file() and (phot / "comparison_stars_per_target.csv").is_file():
            return sub
    raise FileNotFoundError(f"No setup with photometry CSVs under {ps}")


def _is_finite(v: object) -> bool:
    try:
        return math.isfinite(float(v))
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    setup_dir = _find_setup_dir(DRAFT_DIR)
    phot_dir = setup_dir / "photometry"
    at_path = phot_dir / "active_targets.csv"
    comp_path = phot_dir / "comparison_stars_per_target.csv"

    print(f"[INFO] setup={setup_dir.name}")
    print(f"[INFO] active_targets={at_path}")
    print(f"[INFO] comparison_stars={comp_path}")

    at = pd.read_csv(
        at_path,
        dtype={"catalog_id": str, "name": str},
        low_memory=False,
    )
    comp = pd.read_csv(
        comp_path,
        dtype={"catalog_id": str, "name": str, "target_catalog_id": str},
        low_memory=False,
    )

    bo = at[at["catalog_id"] == TARGET_CID]
    if len(bo) != 1:
        print("[FAIL] BO CVn nenájdená v active_targets.csv")
        return
    row = bo.iloc[0]
    try:
        assert 0.40 < float(row["bp_rp"]) < 0.52
        assert "b_v" not in at.columns or pd.isna(row.get("b_v", float("nan")))
        print(f"[OK] active_targets BO CVn: bp_rp={float(row['bp_rp']):.4f}")
    except Exception:  # noqa: BLE001
        print(
            "[FAIL] active_targets BO CVn:",
            f"bp_rp={row.get('bp_rp')}",
            f"has_b_v_col={'b_v' in at.columns}",
        )

    bo_comp = comp[comp["target_catalog_id"] == TARGET_CID]
    if len(bo_comp) < 3:
        print(f"[FAIL] Málo comp riadkov: {len(bo_comp)}")
        return
    if "bp_rp" not in bo_comp.columns or not bo_comp["bp_rp"].apply(_is_finite).any():
        print("[FAIL] Žiadna comp nemá finite bp_rp")
        return

    tier_col = "tier" if "tier" in bo_comp.columns else ("comp_tier" if "comp_tier" in bo_comp.columns else None)
    if tier_col is None:
        print("[FAIL] Chýba tier/comp_tier stĺpec")
        return
    tiers = bo_comp[tier_col].dropna().astype(int)
    if not tiers.between(1, 4).all():
        print(f"[FAIL] Tier mimo rozsahu 1-4: {tiers.unique()}")

    dbprp_finite = bo_comp["delta_bprp_abs"].apply(_is_finite) if "delta_bprp_abs" in bo_comp.columns else pd.Series(False)
    if not dbprp_finite.any():
        print("[FAIL] Žiadna comp nemá finite delta_bprp_abs")
    print(
        f"[OK] comparison_stars BO CVn: {len(bo_comp)} comp, "
        f"tiers={sorted(tiers.unique())}, "
        f"bp_rp finite={int(bo_comp['bp_rp'].apply(_is_finite).sum())}, "
        f"delta_bprp finite={int(dbprp_finite.sum())}"
    )

    nan_targets = at[at["bp_rp"].isna()] if "bp_rp" in at.columns else at.iloc[0:0]
    if len(nan_targets) > 0:
        test_cid = str(nan_targets.iloc[0]["catalog_id"])
        nan_comp = comp[comp["target_catalog_id"] == test_cid]
        if len(nan_comp) > 0:
            tiers_nan = nan_comp[tier_col].dropna().astype(int)
            if (tiers_nan == 4).all():
                print(f"[OK] NaN target bp_rp → tier=4 pre {test_cid}")
            else:
                print(f"[FAIL] Pre target s NaN bp_rp očakávam tier=4, dostal: {tiers_nan.unique()}")
        else:
            print(f"[SKIP] No comps for NaN target {test_cid}")
    else:
        print("[SKIP] No target with NaN bp_rp")


if __name__ == "__main__":
    main()
