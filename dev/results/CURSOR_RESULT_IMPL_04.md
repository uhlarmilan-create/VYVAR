# CURSOR RESULT - IMPL-04

Date: 2026-08-15
Baseline: c2bab5c (IMPL-03), stamp 1630beb
Tip: see commit after this file (fill SHA on stamp)
Push: NO

## What I did

Removed the aperture-mask pixelisation that produced the IMPL-03 scatter
sawtooth, unified ladder and production on one exact-overlap aperture-sum path,
rescanned the IMPL-03 design on clean curves, chose radius by the pre-stated
flat-upper-edge rule, measured a flux-matched blended eval set, fixed check-star
selection that produced the twin 222 mmag meters, remasured draft 514 at the
final radius, rebuilt the acceptance Phase 2A subset, and re-backfilled check
sidecars after deleting stale ones that pinned the bad check id.

## Item 1 - Cause and fix (sawtooth)

**Cause (code authority):** IMPL-03 ladder `measure_flux_ladder_frame` defaulted
to photutils `method="center"` (binary centre-in-circle). Production already used
`method="exact"`. Integer/half-integer parity is the documented centre-masking
signature, not physics.

**One aperture-sum path (shared):**
`photometry_core._aperture_flux_sky_batch` -> (heterogeneous radii)
`_aperture_flux_sky_per_star`, both with photutils
`aperture_photometry(..., method="exact")`. Ladder calls the batch helper.
Annulus sky still uses `to_mask(method="center")` for median sampling only.

**Literature (masking):** photutils aperture guide / `aperture_photometry` API:
`exact` (default) = exact fractional pixel overlap; `center` = binary by pixel
centre; `subpixel` = grid approximation. Production stays on `exact` as the
precise overlap method that removes radius-parity artefacts.

**Fire proof:**
`dev/tests/test_impl_02_snr_cog_gates.py::test_fire_aperture_mask_exact_smooth_vs_center_sawtooth`
- centre fails smoothness / shows sawtooth; exact passes.

### Clean rescan (seed 51403)

Artifact: `dev/results/IMPL_04_scatter_scan.json`

Parity (mean scatter integer vs half-integer):

| set | int mean [mmag] | half mean [mmag] | split |
|---|---:|---:|---:|
| selection AC-off | 10.138 | 10.139 | **-0.001** |
| held-out AC-off | 15.197 | 16.039 | -0.84 (inside noise) |

Sampling noise (selection): SEM ~ **1.85 mmag** (n_stars median 18).

**Decision rule (pre-stated):**

- Numerical min r=5.0 (7.85 mmag) fails held-out validation -> reject sharp-min.
- Contiguous flat region: **4.5-9.5 px** (tol = max(5% s_min, 1x SEM)).
- Flat branch -> upper edge: **r = 9.5 px** (fixed px).
- Sensitivity at 9.5: EE=0.974; d(EE)/dr ~ 0.0094 /px; d(EE)/d(r/FWHM) ~ 0.049.
- Policy: fixed-px held-out at chosen 10.77 mmag beats r/FWHM at its selection
  opt 14.54 mmag -> **fixed_px wins** (clean re-decision; IMPL-03 P3 was premature).

Corrected findings: r=4.5 won by parity; sharp_min was an artifact; P3 must be
read from this rescan.

## Item 2 - Blended evaluation set

Flux-matched iso vs blend pools (n=36). See `blend_report` in the scan JSON.

| population | chosen r [px] | flat region |
|---|---:|---|
| isolated | 9.5 | 4.5-9.5 |
| blended | 12.0 | 8.0-12.0 |

Material difference (blended prefers larger). No per-star policy this task.
Production single radius remains **9.5 px**.

## Item 3 - Twin 222 mmag check stars

Targets `1498278351706325248`, `1499084499887740160` shared check
**1496871217340450304** (comp_rms~0.125, not VSX/Gaia-variable/suspected).
Cause: tier-1 preference over quieter T2/T3 above the existing RMS ceiling.
Fix: `select_check_star` applies `phase01_comparison_max_comp_rms` before tier
pick. After deleting stale sidecars that pinned the bad id, re-backfill:

| target | check_id | check scatter [mmag] |
|---|---|---:|
| 1498278351706325248 | 1497974027502858240 | 34.0 |
| 1499084499887740160 | 1499200223486564608 | 9.0 |

Shared broken meter removed. Residual ~34 mmag on one target is a quieter check
still imperfect under uncapped COMP-ADMIT membership (COMP-ASSIGN-01 next).

## Production before/after (acceptance, r=9.5)

`dev/results/IMPL_04_production_scatter.json` (mag_calib std mmag; before =
IMPL-03 after at r=4.5):

| Target | before (4.5) | after (9.5) | check after |
|---|---:|---:|---:|
| BO CVn | 146.2 | 145.8 | 9.1 |
| FW CVn | 14.4 | 15.1 | 8.6 |
| 1498278351706325248 | 16.1 | **9.6** | 34.0 (was 222) |
| 1499084499887740160 | (quiet) | 68.3 | **9.0** (was 222) |
| 1500461157165243648 | 14.0 | **9.1** | 9.3 |

Mixed: some quieter at larger EE, some worse (sky / blend). BO unchanged
(intrinsic). Check meter for the twin 222 targets is fixed.

## Files changed

- `src_py/photometry_core.py` - `_aperture_flux_sky_batch` / exact sums
- `src_py/aperture_scatter_select.py` - ladder uses shared batch path
- `src_py/check_star_kmag.py` - RMS ceiling before tier pick
- `dev/tests/test_impl_02_snr_cog_gates.py` - smoothness fire proof
- `dev/tools/impl_04_scatter_aperture_scan.py`
- `dev/tools/impl_04_phase2a_acceptance.py`
- `dev/results/IMPL_04_*.json`, this result

## --fast

OVERALL PASS (1402 passed, 27 skipped) via `session_baseline_check.py --fast`.

## Errors

Phase 2A Comp QA hung after all 10 LCs written; process killed; active_targets
restored from backup. Acceptance LCs complete at r=9.5.
