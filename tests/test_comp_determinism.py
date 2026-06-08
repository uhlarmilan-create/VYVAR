"""
TODO-MULTISET regression test: comp star selection must be deterministic
and stable across pipeline runs on the same input data.

Uses draft_000344 vs draft_000345: same night; draft_345 is re-photometry on
copied draft_344 input (common-mode detrend added) — the correct "same data,
two runs" pair. Those tests require identical comp sets.

Also uses draft_000321 vs draft_000348 when present on disk: different pipeline
runs (not same-input reruns). Those tests only assert non-empty comps and that
comp counts do not collapse between runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

DRAFT_344 = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000344")
DRAFT_345 = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000345")
PHOT_344 = DRAFT_344 / "platesolve/NoFilter_60_2/photometry"
PHOT_345 = DRAFT_345 / "platesolve/NoFilter_60_2/photometry"

DRAFTS_AVAILABLE = (
    (DRAFT_344 / "platesolve" / "NoFilter_60_2" / "photometry" / "comparison_stars_per_target.csv").exists()
    and (DRAFT_345 / "platesolve" / "NoFilter_60_2" / "photometry" / "comparison_stars_per_target.csv").exists()
)
SKIP_NO_DATA = pytest.mark.skipif(
    not DRAFTS_AVAILABLE,
    reason="draft_000344 and draft_000345 not present in Archive/Drafts — copy them to run determinism tests",
)

DRAFT_321 = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000321")
DRAFT_348 = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000348")
PHOT_321 = DRAFT_321 / "platesolve/NoFilter_60_2/photometry"
PHOT_348 = DRAFT_348 / "platesolve/NoFilter_60_2/photometry"

DRAFTS_321_348_AVAILABLE = (
    (DRAFT_321 / "platesolve" / "NoFilter_60_2" / "photometry" / "comparison_stars_per_target.csv").exists()
    and (DRAFT_348 / "platesolve" / "NoFilter_60_2" / "photometry" / "comparison_stars_per_target.csv").exists()
)
SKIP_NO_321_348 = pytest.mark.skipif(
    not DRAFTS_321_348_AVAILABLE,
    reason="draft_000321 and draft_000348 comparison_stars_per_target.csv not in Archive/Drafts",
)


def load_comp_sets(phot_dir: Path) -> dict[str, set[str]]:
    """Load comparison_stars_per_target as dict: target_id -> set of comp_ids."""
    csv = phot_dir / "comparison_stars_per_target.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv, dtype={"catalog_id": str, "target_catalog_id": str})
    result: dict[str, set[str]] = {}
    for tid, grp in df.groupby(df["target_catalog_id"].astype(str)):
        result[tid] = set(grp["catalog_id"].astype(str))
    return result


@SKIP_NO_DATA
def test_comp_sets_are_deterministic():
    """Same field + same config must produce identical comp sets."""
    cs344 = load_comp_sets(PHOT_344)
    cs345 = load_comp_sets(PHOT_345)

    common = set(cs344) & set(cs345)
    assert len(common) > 0, "No common targets between drafts — check input data"

    mismatches = []
    for tid in common:
        if cs344[tid] != cs345[tid]:
            mismatches.append(
                {
                    "target": tid,
                    "only_344": cs344[tid] - cs345[tid],
                    "only_345": cs345[tid] - cs344[tid],
                    "shared": cs344[tid] & cs345[tid],
                }
            )

    assert len(mismatches) == 0, (
        f"{len(mismatches)}/{len(common)} targets have different comp sets:\n"
        + "\n".join(
            f"  target {m['target']}: shared={len(m['shared'])}, "
            f"only_344={len(m['only_344'])}, only_345={len(m['only_345'])}"
            for m in mismatches[:10]
        )
    )


@SKIP_NO_DATA
def test_comp_count_stable():
    """Comp count per target should not drop significantly between runs."""
    cs344 = load_comp_sets(PHOT_344)
    cs345 = load_comp_sets(PHOT_345)

    common = set(cs344) & set(cs345)
    regressions = []
    for tid in common:
        n344 = len(cs344[tid])
        n345 = len(cs345[tid])
        if n345 < n344 - 1:  # allow 1 comp difference
            regressions.append((tid, n344, n345))

    assert len(regressions) == 0, (
        f"{len(regressions)} targets lost comps:\n"
        + "\n".join(f"  {t}: {n344} → {n345}" for t, n344, n345 in regressions[:10])
    )


@SKIP_NO_DATA
def test_bo_cvn_has_comps():
    """BO CVn (primary target of BO CVn field) must always have comp stars."""
    cs345 = load_comp_sets(PHOT_345)
    s345 = pd.read_csv(PHOT_345 / "photometry_summary.csv")

    if "vsx_name" in s345.columns:
        bo = s345[s345["vsx_name"] == "BO CVn"]
    else:
        bo = s345[s345["name"].str.contains("BO CVn", na=False)]

    assert len(bo) > 0, "BO CVn not in photometry_summary — primary target missing"
    assert bo["n_good_comp"].values[0] >= 3, (
        f"BO CVn has only {bo['n_good_comp'].values[0]} good comps (need >=3)"
    )


@SKIP_NO_321_348
def test_comp_sets_nonempty_321_348():
    """Each common target must have at least one comp in both 321 and 348 runs."""
    cs321 = load_comp_sets(PHOT_321)
    cs348 = load_comp_sets(PHOT_348)

    common = set(cs321) & set(cs348)
    assert len(common) > 0, "No common targets between draft_000321 and draft_000348"

    empty = []
    for tid in sorted(common):
        if len(cs321[tid]) < 1 or len(cs348[tid]) < 1:
            empty.append(
                (tid, len(cs321[tid]), len(cs348[tid])),
            )

    assert len(empty) == 0, (
        f"{len(empty)}/{len(common)} targets have zero comps in at least one run:\n"
        + "\n".join(
            f"  {t}: n_321={n321}, n_348={n348}"
            for t, n321, n348 in empty[:10]
        )
    )


@SKIP_NO_321_348
def test_comp_count_regression_321_348():
    """348 should not lose more than half the comps vs 321 for any common target."""
    cs321 = load_comp_sets(PHOT_321)
    cs348 = load_comp_sets(PHOT_348)

    common = set(cs321) & set(cs348)
    assert len(common) > 0, "No common targets between draft_000321 and draft_000348"

    regressions = []
    for tid in sorted(common):
        n321 = len(cs321[tid])
        n348 = len(cs348[tid])
        # len(comps_348) >= len(comps_321) * 0.5, with 1-comp slack for integer caps
        min_348 = max(1.0, n321 * 0.5 - 1.0)
        if n321 > 0 and n348 < min_348:
            regressions.append((tid, n321, n348))

    assert len(regressions) == 0, (
        f"{len(regressions)}/{len(common)} targets lost more than half their comps (321 → 348):\n"
        + "\n".join(f"  {t}: {n321} → {n348}" for t, n321, n348 in regressions[:10])
    )
