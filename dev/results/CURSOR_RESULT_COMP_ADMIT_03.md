# CURSOR RESULT - COMP-ADMIT-03

Date: 2026-08-15
Repo tip at issue: 691d3be (+ GATE-REGIME-01 18c770e, GATE-OWNERSHIP-01 5612f42)
Commit: (filled after local commit)
Type: SCIENCE CHANGE. Push: NO.

## Premise

Compared: comparison-star admission (rank cuts + tiers + hard RMS/colour/mag/dilution)
vs continuous Broeg-family weights. Difference: only three gates remain
(saturation, known variable, geometry); scatter/colour/distance become
`sigma_eff` weights. Physical argument recorded under INV-GATE-REMOVAL (not
byte-identity).

## Forced photometry (required answer)

Per-frame comparison photometry is **not** forced aperture at WCS/catalogue XY
every frame. Stars must be DAO-detected (pass-1 or targeted pass-2), matched to
MASTERSTAR, then aperture-measured at the locked MASTERSTAR grid
(`pipeline.py` detect/match ~7799-8316; aperture `photometry_core.py` ~12347+).
If DAO never finds the star that frame, there is no `proc_*.csv` row.

Therefore `detect_frac` was a symptom of detection-dependent photometry. It is
deleted as an admission criterion. A follow-up task is needed if fixed membership
across every frame is required without a Honeycutt global least-squares ensemble.

## Broeg iteration

`ensemble_normalize` is one-shot `1/sigma^2` for the catalog ZP (AIJ/Honeycutt
unweighted flux sum for `delta_mag`). It does not iterate residuals into weights.
Adding colour/distance terms to `sigma_eff` does not change that. An iterative
*fit* that down-weights is not the same as discarding members.

## What changed

- `admit_pool_stars`: only VSX / Gaia variable rejects
- Deleted as cuts: p84 MAD/IQR/inv_eta, detect_frac, dilution ladder, faint/bright,
  colour hard cut, mag adaptive cut, max_comp_rms hard cut, SNR/PSF/FWHM/GS11 hard
  filters, tier truncation / mag_proxy tier authority, p2p/slope exclusion,
  distance hard cut on candidate masks, tier multiplier on Phase-1 `comp_weight`
- New `comp_weights.py`: `sigma_eff^2 = rms^2 + (c_col*|dBP-RP|)^2 + (c_dist*r)^2`
- `ensemble_normalize`: full membership + continuous weights (no tier multiplier,
  no n_comp_max truncation)
- Config/registry/UI: `comp_weight_c_col_mag_per_bprp`, `comp_weight_c_dist_mag_per_deg`,
  `comp_weight_airmass_span` (all with units)
- DECISIONS: COMP-ADMIT-03 entry

## Universality tests

`dev/tests/test_comp_weights_universal.py`: permutation exact, subset exact,
uniformly-good (new admits all; old rank/faint fire proof rejects), injected
50 mmag variable suppresses only itself by `(0.01/0.05)^2`.

## Measurements

| Draft | After global pool | Empty? |
|------:|------------------:|:------:|
| 512 | **141** | no |
| 513 | **1238** | no |
| 435 | **783** | **no** (was emptied under p84) |

`c_col` / `c_dist`: both **0.0** with named gaps on these harness runs
(`c_col=0 (no k2_bprp)`; `c_dist=0 named_gap:no_regression_inputs`).
PSF colour term is a named zero (`C_COL_PSF_TERM_GAP`). Rig dependence not
measured (no Newton draft in harness).

BO CVn archived comps present in new pools on 512/513/435. Intersection of
*archived assignment sets* 512 vs 513 is empty (different prior assignments);
pool membership no longer blocks BO-ENSEMBLE-01. Full Phase-1/2A rebuild needed
to close assigned-ensemble overlap.

FW CVn: archived n_comps not resolved in harness; rebuild needed for check-star
scatter / ELL shape.

Raw JSON: `dev/results/COMP_ADMIT_03_measurements.json`.

## Residual rejection (behaviour sweep)

`scan_gates_inventory.py --validate`: OK (59 gates). Inventory still *lists*
five `rank_statistic` sites under `comp_pool` from COMP-POOL-01 Stage 2; those
cuts are no longer applied in `admit_pool_stars` (inventory refresh is a
follow-up). Code-path residuals that still reject on the comparison path:

Allowed three gates:
- saturation / nonlinear / likely_saturated / zone / is_noisy (measurability)
- VSX / Gaia variable / variable_targets (known variable)
- chip margin / safe_bbox / edge-annulus (geometry)

Catalogue contaminants still G1 (report for Milan; not in the three-gate table):
- Gaia NSS / QSO / GAL when config exclude flags are on

`comp_qa` / trust: post-Phase-2A grading only (`n_clean`, GREEN threshold). No
admission reject wired in this change.

## `--fast`

OVERALL PASS at pre-commit tip `e9fc9f1` (1369 passed, 27 skipped). Re-verify
after this commit SHA.

## Files

- `src_py/comp_weights.py` (new)
- `src_py/comp_pool_noise.py`, `photometry_core.py`, `comp_selection_per_target.py`
- `src_py/config.py`, `ui_settings.py`
- `docs/VYVAR_DECISIONS.md`, `docs/VYVAR_PARAMS.md`
- `dev/validation/params_registry.json`
- `dev/tests/test_comp_weights_universal.py` + updated regression tests
- `dev/tools/comp_admit_03_measure.py`, `dev/results/COMP_ADMIT_03_measurements.json`
- `dev/results/CURSOR_RESULT_COMP_ADMIT_03.md`
