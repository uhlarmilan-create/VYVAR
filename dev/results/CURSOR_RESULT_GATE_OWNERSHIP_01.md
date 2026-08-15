# CURSOR RESULT - GATE-OWNERSHIP-01

Date: 2026-08-15
Repo tip at issue: 691d3be
Commit: 5612f42 (local; not pushed)
Type: INVESTIGATION + INSTRUMENTATION inventory. No science-path behaviour change from this task alone.
Push: NO

## Premise

Compared: science-path decision sites (gates) as implemented in `src_py/` vs the requirement that one physical quantity have one owning stage.
Difference: gates were added where symptoms appeared; inventory records every G1/G2/G3 site found, without retuning or removing any.

Baseline tip at issue: 691d3be. GATE-REGIME-01 committed separately as 18c770e (failure-path only; does not retune thresholds).

Definition used (reproducible): G1 remove rows; G2 set flag later used to remove rows; G3 clamp/substitute on threshold (incl. silent except fallbacks). Diagnostics/plots/log-only are not gates. Ambiguous sites kept with `ambiguous: true`.

## What I did

1. Built curated inventory `dev/validation/gates_inventory.json` (59 gates) from code read + explore pass.
2. Added validator/scanner `dev/tools/scan_gates_inventory.py` (`--validate`).
3. Cross-checked `threshold_source=rank_statistic` vs `derived_fit` (percentile-of-filtered-population vs estimator).
4. Compared registry `phase` vs Milan stage vocabulary (R4).
5. Literature check on rank cuts for comparison-star admission (R2 appendix).

Companion GATE-REGIME-01 (separate commit/task) wires INV-NO-SILENT-EMPTY at one site and does not change thresholds.

## Deliverable counts

| | |
|--|--|
| Gates | 59 |
| G1 / G2 / G3 | 42 / 10 / 7 |
| `rank_statistic` | 5 |
| `instrumented` (persisted n_in/n_out) | 5 |
| `can_empty_population` | 50 |
| `has_empty_guard` | 20 |
| Ambiguous | 4 |
| Validator | OK |

`--fast` OVERALL PASS (post working tree): 1364 passed, 27 skipped. Science path untouched by this inventory task.

---

## R1 - Conflicts

Grouped by quantity. Each group below has more than one independently thresholded gate. Not resolved.

### R1.1 Detector saturation (triplicate+)

Same physical idea (star too bright for linear photometry), independent authorities:

| Gate | File:line | Unit / form |
|------|-----------|-------------|
| `pool_zone_saturated_nonlinear` | `src_py/photometry_core.py:13932` | categorical zone |
| `pool_cand_mask_is_saturated` | `src_py/photometry_core.py:13936` | boolean `is_saturated` / `likely_saturated` |
| `masterstar_zone_saturated_annotation` | `src_py/pipeline.py:6426` | peak vs `saturate_limit_fraction` * sat |
| `likely_saturated_vectorized` | `src_py/pipeline.py:5747` | peak\|plateau vs sat frac |
| `sat_diag_apply_raw_peaks` | `src_py/sat_diag.py:912` | raw peak columns (parallel) |
| `admit_sat_frac_per_target` | `src_py/comp_selection_per_target.py:1131` | frame frac over 0.70*sat |
| `admit_sat_frac_comp_pool_rms` | `src_py/comp_pool_rms.py:290` | duplicate admission logic |
| `spatial_static_sat_filter` | `src_py/comp_selection_per_target.py:418` | re-reads same flags |
| `zone_skip_photometry_saturated` | `src_py/photometry_core.py:13347` | target skip |

Can one admit what another rejects? Yes. Example: zone=`linear` with `likely_saturated=True` (peak/plateau path) is rejected by pool cand_mask but would pass a zone-only consumer; admission 70%-over-10% can reject a star that never tripped MASTERSTAR zone on the stack.

### R1.2 Comparison stability (multi-unit)

| Gate | File:line | Unit |
|------|-----------|------|
| `phase01_max_comp_rms_hard_cut` | `src_py/comp_selection_per_target.py:1565` | mag (absolute RMS) |
| `check_comparison_stability_p2p` | `src_py/photometry_core.py:3213` | mag p2p (reuses `phase01_comparison_max_comp_rms`) |
| `check_comparison_stability_slope` | `src_py/photometry_core.py:3255` | mmag/hr |
| `derived_admission_p84_mad_excess` | `src_py/comp_pool_noise.py:869` | dimensionless excess vs model |
| `derived_admission_p84_iqr_excess` | `src_py/comp_pool_noise.py:871` | dimensionless |
| `derived_admission_p84_inv_eta` | `src_py/comp_pool_noise.py:873` | dimensionless |

Absolute 0.1 mag vs rank excess: a star with RMS 0.08 mag can fail p84 inv_eta on a uniformly good night (draft 435 pattern), and a star above 0.1 mag can still be below p84 if the field is noisy. Independently set; either can admit what the other rejects.

### R1.3 Dilution (dual authority)

| Gate | File:line | Threshold |
|------|-----------|-----------|
| `derived_dilution_percentile` | `src_py/comp_pool_noise.py:814` | p16/p10/p05 of D (rank) |
| `gs11_dilution_hard_cut` | `src_py/comp_selection_per_target.py:1241` | fixed 0.90 / suspect 0.98 |

Same quantity D; different owners and units of decision (order statistic vs literal).

### R1.4 Significance / detection floors (fourth conflict class)

Named here as a **fourth conflict class** beyond saturation triplicate and stability multi-unit:

| Gate | File:line | Estimator |
|------|-----------|-----------|
| `dao_starfinder_threshold` | `src_py/pipeline.py:7267` | k * scene std |
| `prematch_peak_snr` | `src_py/pipeline.py:8919` | k * sky MAD |
| `median_snr_lt_5` / `pool_median_snr_lt_5` | `comp_selection_per_target.py:1177` / `comp_pool_rms.py:311` | median SNR < 5 |
| `masterstar_zone_linear_noise` | `src_py/pipeline.py:6458` | `dao_detection_n_equiv` |

All answer "is this detection usable?" with incompatible noise estimators. One can keep what another drops.

### R1.5 Duplicate same-rule sites

- Admission sat frac: `comp_selection_per_target.py:1131` and `comp_pool_rms.py:290` (copy).
- Median SNR < 5 and edge frac: per-target and pool RMS mirrors.
- VSX known variable: pool cand_mask, admit_pool_stars, spatial filter.

---

## R2 - Rank cuts

Every `threshold_source = rank_statistic` gate:

| Gate | File:line | Rejection fraction by construction | Uniformly-good population |
|------|-----------|------------------------------------|---------------------------|
| `derived_admission_p84_mad_excess` | `comp_pool_noise.py:869` | Rejects above p84 of (MAD/sigma_total) among bulk => ~16% of that bulk fail this cut alone (exact fraction depends on ties and NaNs) | Still rejects the worst ~16% even if all stars are photon-noise limited; "good" is relative to peers, not absolute |
| `derived_admission_p84_iqr_excess` | `comp_pool_noise.py:871` | Same ~16% construction on IQR excess | Same |
| `derived_admission_p84_inv_eta` | `comp_pool_noise.py:873` | Same ~16% on inv_eta | Draft 435: tighter inv_eta p84 emptied archived BO comps while draft 512 admitted them (`CURSOR_RESULT_COMP_POOL_01_S2.md`) |
| `derived_detect_frac_p16` | `comp_pool_noise.py:755` | Floor = p16 of detect_frac among mag_g<=14; rejects stars below that floor | When all bright stars have detect_frac=1.000 (draft 512), floor becomes 1.000 and any incomplete detection fails; gate does not relax when the field is uniformly complete |
| `derived_dilution_percentile` | `comp_pool_noise.py:814` | p16 (else p10/p05) of D; rejects D below thr | When D piles at 1.0 through p05, threshold becomes inert (named); otherwise still rejects a fixed lower tail of isolation even if all D are scientifically fine |

### Literature (rank cuts vs comparison admission)

- **Broeg et al. 2005** (AN 326, 134): recursive absolute dispersion / inverse-variance weights; variables down-weighted or removed by empirical std vs instrumental error -- not a fixed population percentile admission cut.
- **Sokolovsky et al. 2017** (MNRAS 464, 274): deliberately use IQR and 1/eta with magnitude-peer ranking to **find variables** (reject the non-variable bulk). That is the inverse problem of comparison-star admission. Rank cuts are defended for variability search, not for admitting a universal comparison pool.
- **photutils / SEP**: detection thresholds in sigma units of background; no p84 pool admission.
- **AstroImageJ**: user-selected comps; quality is absolute photometry metrics, not draft-relative percentiles.
- **Kepler/TESS PDC**: ensemble based on similar stars and CBVs; variability/quality cuts are not "always drop top 16% of this night's pool."

Conclusion: the argument against using rank cuts as the **owning** admission rule for a universal photometry tool survives contact with this literature. Rank indices (MAD, IQR, inv_eta) remain useful **diagnostics** and for variable-star search; using them as draft-relative hard admission enforces a rejection fraction even when the null (all comps good) is true.

---

## R3 - Unguarded emptiers

Gates with `can_empty_population=true` and `has_empty_guard=false` (from inventory validator counts: 50 can_empty, 20 guarded => 30 unguarded). Full list is the inventory filter; highest-risk examples:

| Gate | File:line |
|------|-----------|
| `admit_sat_frac_per_target` | `comp_selection_per_target.py:1131` |
| `admit_sat_frac_comp_pool_rms` | `comp_pool_rms.py:290` |
| `phase01_max_comp_rms_hard_cut` | `comp_selection_per_target.py:1565` (warns / returns None below n_comp_min -- soft, not population assert) |
| `derived_admission_p84_*` (pre-REGIME) | emptied silently via empty set; **REGIME wires guard at aggregator** |
| `dao_starfinder_threshold` | `pipeline.py:7267` |
| `prematch_peak_snr` | `pipeline.py:8919` |
| `proc_drop_unmatched` | `pipeline.py:7703` |
| `gs11_dilution_hard_cut` | `comp_selection_per_target.py:1241` |
| `median_snr_lt_5` / pool mirror | `:1177` / `comp_pool_rms.py:311` |
| `fwhm_is_rejected_frame_filter` | `pipeline.py:2407` |

Complete machine-readable set: every inventory row with those two flags.

---

## R4 - Registry mismatch

### Vocabulary

Milan stage strings: `calibration`, `best_frame`, `platesolve`, `alignment`, `dao_detection`, `comp_pool`, `comp_assignment`, `photometry`, `reporting`.

Registry `phase` strings (274 params): `detection`, `photometry`, `comp_selection`, `qc`, `reports`, `paths`, `trust`, `observer`, `extinction`, `calibration`, `alignment`, `export`, `system`.

Shared names only: `calibration`, `alignment`, `photometry`. No registry phase equals `comp_pool`, `comp_assignment`, `dao_detection`, `best_frame`, `platesolve`, or `reporting` (registry uses `detection` / `comp_selection` / `qc` / `reports`).

### Disagreements (code `stage` vs registry `phase` for gate `param_names`)

45 gate-param pairs disagree. Unique (param, code_stage, registry_phase) samples:

| Param | Code stage | Registry phase | Example gate |
|-------|------------|----------------|--------------|
| `saturate_limit_fraction` | dao_detection | detection | masterstar_zone_saturated_annotation `pipeline.py:6426` |
| `admission_sat_peak_frac` | comp_assignment | comp_selection | admit_sat_frac_per_target `:1131` |
| `phase01_comparison_max_comp_rms` | comp_assignment / photometry | comp_selection | phase01_max_comp_rms_hard_cut; check_comparison_stability_p2p |
| `comp_pool_derived_admission` | comp_pool | comp_selection | derived_admission_p84_* |
| `gs11_comp_max_dilution` | comp_assignment | photometry | gs11_dilution_hard_cut |
| `gain` / `read_noise` | comp_pool | photometry | derived_faint_limit |
| `frame_align_residual_*` | alignment | qc | align_residual_gate |
| `nonlinearity_peak_percentile` | dao_detection | photometry | likely_nonlinear_annotation |

### Params referenced by gates but absent from registry

`saturate_level_fraction`, `edge_bad_frame_frac_max`, `chip_interior_margin_px`, `dao_threshold_sigma`, `prematch_peak_sigma_floor`, `max_catalog_rows` (8 references).

### Unit coverage

At issue time: **6 of 274** parameters have a non-null `unit` in `dev/validation/params_registry.json` (confirmed 2026-08-15). Remaining 268 have `unit: null`, including all magnitude/RMS/fraction cuts that need units for ownership work.

---

## R5 - Ownership proposal (for Milan; not a change)

For each quantity, one owning stage; other gates become read-only consumers of a recorded decision. INV-GATE-REMOVAL: removing a gate needs a physical argument, not byte-identity.

| Quantity | Proposed owner stage | Recorded decision | Consumers (read-only) | Needs physical argument to retire? |
|----------|----------------------|-------------------|------------------------|------------------------------------|
| Detector saturation | `dao_detection` (raw peak authority via SAT-DIAG / placed aperture) | single `sat_state` per star (linear / nonlinear / saturated) + optional per-frame clean_frac | comp_pool zone/bool filters; admission frac; spatial static; target skip; ePSF funnel | Yes -- must show other thresholds are implied by the owner or accept risk |
| Comp stability / variability | `comp_pool` | single admit/reject with physical threshold (mag or excess vs noise model with **absolute** ceiling), not draft percentile | Phase-1 max_comp_rms; Phase-2A p2p/slope; Broeg weights | Yes -- rank p84 retirement needs physics (see R2 literature); identity check insufficient |
| Dilution / isolation | `comp_pool` | one D threshold (physical or fixed science constant) | GS11 hard cut; derived percentile | Yes if dropping one of the two |
| Detection significance | `dao_detection` | one SNR/threshold policy | prematch, median SNR5, zone N_equiv | Yes |
| Frame quality | `best_frame` / QC allowlist | frame keep set | FWHM reject, align residual (if ON), calibration abort | Partial -- CAL-DIAG already owned |
| Catalog contaminants (VSX/NSS/ext) | `comp_pool` | flags only | spatial re-filters | Low risk if flags identical |
| Geometry (bbox/margin/edge) | `alignment` | safe footprint | pool margin; annulus frac | Yes for dual edge tests |

Iron rule note: consolidating R1.1/R1.2 by deleting consumer filters on "outputs unchanged" evidence alone violates INV-GATE-REMOVAL. Prefer: owner writes decision once; consumers assert consistency or become no-ops only after a physical proof that the consumer condition is implied.

---

## Files

- `dev/validation/gates_inventory.json`
- `dev/tools/scan_gates_inventory.py`
- `dev/results/CURSOR_RESULT_GATE_OWNERSHIP_01.md` (this file)

## Errors

None.
