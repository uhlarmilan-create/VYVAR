# VYVAR - Check-star Selection Hardening Spec (CS-2 / CS-3 / CS-4)

Scope: fix how the **check star is selected** in `check_star_kmag.py:select_check_star`. The
gate (scatter thresholds) was hardened separately; this spec makes sure the star being gated
is an **independent, non-artefact** witness.

INVARIANTS:
- **Photometry byte-identity holds.** `check_kmag_*.csv` is NOT in the photometry SHA set.
- **Trust re-baselines** (intended).
- English/ASCII only.

## CS-3 (PRIORITY) - ensemble exclusion

Pass `ensemble_ids: set[str]` from Phase-2A `ensemble_member_ids()` at call sites. Never
select an ensemble member as check star. No independent candidate remains -> `None`.

Amended 2026-10-31 (CHECK-ONE-2026-10, G08-41, Milan approved): the former rule "fewer than
`n_comp_min` independent candidates -> `None`" is removed. The purpose of CS-3 was the ensemble
exclusion, not a count; ONE clean independent candidate is the check star (AAVSO extended format:
one KNAME / KMAG; AAVSO Guide to Photometric Uncertainty, Technique 2). `select_check_star` has no
`n_comp_min` argument. A check star more than 2 mag (Gaia G) from the target is recorded
(`check_star.check_minus_target_g_mag` in the comp_quality JSON) with a WARNING, not a gate.

## CS-2 - artefact floor

Config `check_select_rms_floor` (default `1e-4`); also `0.1 * median(comp_rms)` of pool.
Drop candidates at/below floor before sorting.

## CS-4 - crowding (when metric present)

If `contamination_idx` in comp_df, exclude above `aperture_correction_max_contamination`.
Otherwise deferred.

## Tests

`tests/test_check_star_selection.py`

## draft_000387 validation

SHA unchanged; report CS-3 footprint and trust re-baseline.
