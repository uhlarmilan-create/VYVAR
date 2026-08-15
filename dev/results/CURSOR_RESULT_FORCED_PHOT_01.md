# CURSOR RESULT - FORCED-PHOT-01 + COMP-WEIGHT-COEFF-01

Date: 2026-08-15
Baseline: b6e0e29 (COMP-ADMIT-03)
Commit: (filled after local commit)
Push: NO. Waiting for Milan.

Ordering: FORCED-PHOT-01 implemented first; COMP-WEIGHT c_dist measured on
pre-forced COMP-POOL-02 residual products and marked for re-verify after a
forced rebuild. Honeycutt global LS was not required.

---

## FORCED-PHOT-01

### Premise

Compared: DAO-conditional per-frame presence vs forced aperture at locked
MASTERSTAR XY for force-eligible members. Difference: presence no longer depends
on detection.

### The required answer

**No.** After this change, a force-eligible pool member's per-frame presence is
not conditional on anything other than geometry.

Code path: `detect_stars_match_master_reference` (DAO unchanged) ->
`forced_photometry.inject_forced_masterstar_rows` in
`_export_per_frame_run_catalog_core` -> aperture enhance -> proc CSV.

### Ensemble treatment of per-frame saturation

Rows are kept with the saturation flag. `ensemble_normalize` accepts
`comp_likely_saturated` and excludes those epochs from the zeropoint flux sum
explicitly (measurability: invalid that frame). Membership `good_ids` unchanged.

### Centroid bound

Peak refine search radius = ceil(fwhm * `forced_photometry_centroid_bound_fwhm`)
default 2.5 FWHM; recorded in inject meta (`max_refine_shift_px`, bound).

### DAO untouched

Injection is after DAO match. `n_matched` / `n_raw_dao` come from the DAO path
and are not modified by forced injection.

### Check-star scatter before/after

Not measured on a rebuilt draft in this workspace: Archive draft photometry
trees for 512/513/435 were not found under `Archive/Drafts` / local DB paths
queried here. Acceptance unit tests prove fixed contributing-set equality when
all members have finite forced mags (`dev/tests/test_forced_phot_and_weights.py`).
A rebuild on the operator machine is required for the ZP-CLIP-REMOVAL-style
scatter matrix.

### Tests

- `test_forced_inject_fills_missing_members_and_records_geometry`
- `test_ensemble_contributing_set_equal_when_all_finite`
- `test_ensemble_sat_excluded_explicitly_membership_unchanged`
- `test_is_noisy_is_force_eligible_nss_qso_are_not`

---

## COMP-WEIGHT-COEFF-01

### c_col (non-null on 512/513/435)

| Term | Value | Unit | Derivation |
|------|------:|------|------------|
| k2 (CLEAR) | 0.0 | mag/BP-RP | literature NONE for CLEAR/unfiltered; no CHOSEN CMOS default |
| PSF EE | **0.029485** | mag/BP-RP | MEASURED: 29.485 mmag / Delta(BP-RP)=1.0 (0.5->1.5), COMP-POOL-02 Item 4 |
| combined | **0.029485** | mag/BP-RP | quadrature(k2, psf) |
| mirror prediction | 0.0 | mag/BP-RP | `optics_kind=mirror` |

Uncertainty on PSF term: single published EE estimate (no formal SE in Item 4);
sensitivity below.

### c_dist (measured, not a gap)

OLS `scatter_mad` vs field-centre separation (plate scale 9.55169 "/px):

| Draft | c_dist | unc | note |
|------:|-------:|----:|------|
| 512 | **0.0** | 0.00123 | MEASURED consistent with zero (n=297) |
| 435 | **0.0** | 0.00448 | MEASURED non-positive / zero (n=1277) |
| 510 | **0.0** | 0.00126 | MEASURED consistent with zero (scatter path) |
| 513 | **0.0** | -- | same refractive rig; no residual CSV; inherit measured zero |

Newton/Boyden: no residual products in harness; mirror `c_col_psf=0` predicted.

### FW CVn (513) weight suppression (equal rms=0.01 synthetic)

With c_col=0.029485: C03 at dBP-RP=0.044 has w=9834; three comps at dBP-RP=0.4
have w=4182 each (~**42.5%** of C03). Meaningful suppression.

### Sensitivity of colour bias proxy vs c_col factor

| factor | c_col | weighted colour bias (mag) |
|-------:|------:|---------------------------:|
| 0.5 | 0.0147 | 0.0043 |
| 1.0 | 0.0295 | 0.0072 |
| 1.5 | 0.0442 | 0.0087 |

Bias moves ~mmag under +/-50% on c_col; method is moderately sensitive but not
unstable.

### Universality

Exact subset/permutation invariance still holds with non-zero coeffs
(`test_universality_with_nonzero_coeffs_exact` + existing
`test_comp_weights_universal.py`).

### COMP-ADMIT-03 corrections

- `is_noisy` removed from admission/force gates (weight via scatter).
- Gaia NSS = known variable; QSO/GAL = measurability.

Machine: `dev/results/COMP_WEIGHT_COEFF_01_measurements.json`,
`dev/results/COMP_WEIGHT_COEFF_01_c_dist.json`.

---

## `--fast`

Record after commit.
