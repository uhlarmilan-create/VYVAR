# VYVAR — Check-star Selection Hardening Spec (CS-2 / CS-3 / CS-4)

Scope: fix how the **check star is selected** in `check_star_kmag.py:select_check_star`. The
gate (scatter thresholds) was hardened separately; this spec makes sure the star being gated
is an **independent, non-artefact** witness.

INVARIANTS:
- **Photometry byte-identity holds.** `check_kmag_*.csv` is NOT in the photometry SHA set.
- **Trust re-baselines** (intended).
- English/ASCII only.

## CS-3 (PRIORITY) — ensemble exclusion

Pass `ensemble_ids: set[str]` from Phase-2A `ensemble_member_ids()` at call sites. Never
select an ensemble member as check star. If fewer than `n_comp_min` independent candidates
remain -> `None`.

## CS-2 — artefact floor

Config `check_select_rms_floor` (default `1e-4`); also `0.1 * median(comp_rms)` of pool.
Drop candidates at/below floor before sorting.

## CS-4 — crowding (when metric present)

If `contamination_idx` in comp_df, exclude above `aperture_correction_max_contamination`.
Otherwise deferred.

## Tests

`tests/test_check_star_selection.py`

## draft_000387 validation

SHA unchanged; report CS-3 footprint and trust re-baseline.
