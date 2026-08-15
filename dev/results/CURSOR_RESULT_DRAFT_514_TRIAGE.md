# CURSOR RESULT - DRAFT-514-TRIAGE

Date: 2026-08-15
Baseline at issue: b731320 (FORCED-PHOT-01 + COMP-WEIGHT-COEFF-01)
Repo HEAD at report: 011fff7
A1/A2 fixes: present in working tree (not committed; Push: NO)
Type: TRIAGE + MEASUREMENT. No ensemble-size cut implemented.
`--fast` OVERALL: PASS (1382 passed, 27 skipped)

Machine-readable: `dev/results/DRAFT_514_TRIAGE_{B1,B2,B3,C1,C2,D}.json`

---

## Standing constraint

No top-N / cumulative-weight membership cut was implemented. Cumulative-weight
truncation appears only as a measuring instrument in B2.

---

## A1. Crash: `math range error`

### Preflight log defect (INV-NO-SILENT-EMPTY class)

`logs/run_preflight_error_20260815_110852.log` recorded:

```
exception: RuntimeError: NoFilter_60_2: math range error
traceback:
NoneType: None
```

Finding: the handler stored no traceback. Fixed in `run_preflight_log.py` to use
`traceback.format_exception(type(exc), exc, exc.__traceback__)`, and `app.py`
preserves the original exception when wrapping a single-setup failure. Regression:
`dev/tests/test_preflight_traceback.py`.

### Real site and input

Site: `src_py/sigma_floor_core.py` `c4_small_sample` -> `math.gamma`.

`OverflowError: math range error` on Windows when `Gamma(n/2)` exceeds ~1e308
(n greater than about 342). After COMP-ADMIT-03, ensembles have n_comps ~ 1292,
so `ensemble_sem_mag_from_residuals` -> `c4_small_sample(1292)` overflows.

Not the candidate sites in `comp_selection_per_target` / `sigma_budget` /
`crowding_index` for this crash.

Fix: compute c4 via `math.lgamma` ratio (overflow-safe). Regression calls the
real function for n in {2,3,5,10,50,200,342,500,1292,5000}:
`dev/tests/test_sigma_floor.py`.

### Relation to `[AC] skipped: no_comp_rms`

Unrelated red herring. On draft 514 all admitted comps are TIER4; filtering to
T1/T2 yielded an empty DataFrame that dropped columns under pandas, so AC falsely
reported `no_comp_rms`. Disk `comparison_stars_per_target.csv` has finite
`comp_rms` on all 125316 pair rows.

Missing `comp_rms` path: `sigma_eff_mag` returns NaN; `weight_from_sigma_eff`
returns 0.0. It does **not** produce unbounded `1/sigma_eff^2`. Tiny rms is
floored at 1e-6 mag inside `sigma_eff_mag`.

AC column check fixed to inspect the full frame and fall back to all comps when
T1/T2 is empty (`photometry_core.compute_aperture_correction`).

### Rerun

Phase 2A resumes past R CVn after the c4 fix. At report time ~21+ lightcurves
written (~58 s/target wall). Resume harness:
`dev/tools/draft_514_resume_phase2a_skip_done.py` (skips targets that already
have LC CSVs). Full 97 photometry targets still completing in background.

---

## A2. Duplicate rows in per-frame product

Measured on `proc_BO_CVn_Light_001.csv` (pre-fix file on disk):

| quantity | value |
|---|---|
| rows | 3627 |
| unique `catalog_id` | 3500 |
| extra duplicate rows | 127 |

Duplicates share identical `flux`, `x`, `y`; `forced_photometry=False`.

Cause: `_export_per_frame_run_catalog_core` did **not** call
`_proc_deduplicate_matched_catalog_rows` before write. An alternate export path
(~11150) did. Multiple DAO detections matched the same Gaia id.

Pool helper `_dedupe_comp_pool_by_gaia_key` is unrelated; the gap is the per-frame
export path.

Fix: call `_proc_deduplicate_matched_catalog_rows` in
`_export_per_frame_run_catalog_core`. Regression:
`dev/tests/test_proc_dedupe_catalog_id.py`.

On-disk draft 514 proc CSVs still contain duplicates until re-exported; the code
path is fixed for the next export.

---

## B1. Weight distribution (97 photometry targets)

218 active targets; 121 have `skip_photometry` (linear/sat zone). The other 97
each have ~1292 admitted comps (125316 pairs).

| metric | min | p16 | median | p84 | max |
|---|---:|---:|---:|---:|---:|
| N_eff | 150.3 | 170.0 | 193.9 | 340.8 | 590.0 |
| heaviest weight fraction | 0.0087 | | 0.0189 | | 0.0438 |

Median comps carrying cumulative weight:

| cumulative weight | median n_comps |
|---|---:|
| 50% | 74 |
| 90% | 366 |
| 99% | 811 |
| 99.9% | 1094 |

No target has N_eff near 1. Worst single-star dominance is R CVn at 4.4% of
total weight. Differential photometry is not collapsing to one star.

JSON: `DRAFT_514_TRIAGE_B1.json` (per-target + summary).

---

## B2. Truncation sensitivity (measuring instrument only)

Synthetic common-mode + per-comp noise; production `sigma_eff` weights;
`ensemble_normalize` full membership. Unit: mmag. SHA: 011fff7.

| target | X | n kept | ZP diff RMS (mmag) | ZP mean offset (mmag) | LC scatter change (mmag) | wall (s) vs full |
|---|---:|---:|---:|---:|---:|---:|
| BO CVn | 99.9% | 1094 | 0.027 | -0.006 | ~0.01 | 0.39 / 0.53 |
| BO CVn | 99% | 812 | 0.09-0.12 | ~0.01 | ~0.03 | ~0.3 / 0.5 |
| BO CVn | 95% | 508 | 0.27 | -0.05 | | |
| BO CVn | 50% | 78 | 0.93 | -0.19 | | |
| FW CVn | 99% | 766 | 0.13 | 0.014 | | |
| R CVn | 99% | 1124 | 0.52 | -0.018 | | |

### Verdict (plain)

For typical targets (BO, FW): keeping comps that carry the top 99% of weight
changes the zeropoint by well under 1 mmag (~0.1 mmag RMS). The low-weight tail
is numerically irrelevant for science at this precision. Any cut is a
**performance** decision, not a noise-model repair.

At 50% weight the ZP moves by ~1 mmag: truncating that hard does change the
result. R CVn (extreme red, one-sided colour, flatter weights) is more sensitive
(~0.5 mmag at 99%).

This does **not** authorize a cumulative-weight or top-N cut (still
population-dependent). See proposal below.

JSON: `DRAFT_514_TRIAGE_B2.json`.

---

## B3. Cost

Observed Phase 2A: ~58 s wall per target with ~1292 comps and 134 frames
(from `comp_quality_*.json` write times).

| n_comps | `ensemble_normalize` alone (134 frames, s) |
|---:|---:|
| 50 | 0.012 |
| 250 | 0.083 |
| 812 | 0.62 |
| 1292 | 2.17 |

Weight recompute for one target: ~0.08 s.

Where time goes: Phase 2A per-target light-curve build (frame I/O, photometry
assembly, quality), not Phase 0+1 scoring and not the ZP solve alone. A
membership cut would relieve the ~2 s ensemble SEM/ZP loop somewhat, but most of
the ~58 s is elsewhere. A cut is only weakly motivated by cost unless frame-level
comp work also scales with n_comps.

JSON: `DRAFT_514_TRIAGE_B3.json`.

---

## C1. Colour balance

`delta_colour_ensemble = sum(w_i (BP-RP)_i)/sum(w_i) - (BP-RP)_target`

| | value (BP-RP) |
|---|---|
| median | -0.189 |
| p16 / p84 | -1.045 / +0.268 |
| worst | R CVn -4.714 (target BP-RP 5.675; 1289 blue / 0 red) |

FW CVn: target BP-RP 0.815, delta_colour +0.072, coverage both sides
(407 blue / 882 red). Weighting already balances FW better than the archived
four-comp story.

JSON: `DRAFT_514_TRIAGE_C1.json` (and B1 per-target fields).

---

## C2. Colour imbalance vs airmass drift

Regress LC residual (mag about median) vs airmass; correlate slope with
`delta_colour_ensemble`. Partial Phase 2A sample (n=21 LCs including BO, FW, R).

| subset | n | corr | coeff (mmag / BP-RP / airmass) |
|---|---:|---:|---:|
| all | 21 | -0.025 | -29 |
| abs(delta_colour) < 2 | 19 | -0.16 | (unstable leverage) |
| abs(slope) < 150 mmag/airmass | 8 | +0.79 | +87 |

Compare: CLEAR k'' = NONE; `c_col` PSF = 0.029485 mag/BP-RP (not per airmass).

Full-sample correlation is consistent with **no measurable colour-imbalance ->
airmass drift** on this rig once variables with huge slopes (BO -404 mmag/airmass,
R -279) dominate the residual. Mild-slope subset is too small and unstable to
choose design (b) or (c).

### Design choice from this measurement

**(a) weighting alone is enough** for colour on this evidence, with the caveat
that Phase 2A was incomplete at report time and pathological variables must not
be read as colour-term detections. Revisit C2 when all 97 LCs exist if desired;
do not implement (b) or (c) from this sample.

JSON: `DRAFT_514_TRIAGE_C2.json`.

---

## C3. Colour coverage

Per-target blue/red counts are in B1/C1. One-sided coverage: **2 / 97**
(R CVn and CV CVn: both extremely red; field has no redder comps). For those
targets imbalance is a property of the field, not of the algorithm. Most targets
have comps on both sides of the target colour.

---

## D1. Catalogue blends (confirmed)

From `DRAFT_514_TRIAGE_D.json` on `proc_BO_CVn_Light_001.csv` (deduped unique ids):

| | |
|---|---|
| plate scale | 9.55 arcsec/px |
| median `fwhm_estimate_px` | 6.13 px = 58.6 arcsec |
| median `aperture_r_px` | 2.71 px |
| stars with neighbour inside one aperture | 1780 / 3500 = 51% |
| neighbour sep / FWHM p50 | ~0 (packed inside PSF) |
| pool members with neighbour inside aperture | 629 / 1292 |

Interpretation confirmed: Gaia resolves sources the 200 mm lens cannot;
multiple catalogue IDs attach to one optical blob. Explains ~3500 catalogue rows
vs ~1600 SIPS-scale detections. No merging implemented (report only).

---

## D2. FWHM triad and aperture (bigger than triage title)

Three values on this draft:

| label | px | role |
|---|---:|---|
| Phase 2A `FWHM=3.301` | 3.301 | `VY_FWHM_GAUSS` preferred in `run_phase2a` header read |
| AUTO FWHM median | 5.311 | UI quality auto-limit (`ui_quality_dashboard`), not photometry sizing |
| SNR table `vy_fwhm_dao_px` | 5.195 | MASTERSTAR `VY_FWHM` (DAO), recorded |
| SNR table `fwhm_px` | 3.389 | per-draft median frame DAO-moment (sizing for SNR table) |
| proc `fwhm_estimate_px` median | 6.13 | per-star moment on catalogue positions (blend-inflated) |

Production aperture for faint stars: **SNR-table `r_min_px` = 2.711**, which
equals the measured median `aperture_r_px`. Config `aperture_fwhm_factor=1.9`
would give 1.9*3.301 = 6.27 px if used globally; it is not what the median star
gets.

Policy contradiction (report as first-class finding):

- `resolve_fwhm_px_for_snr_aperture_table` documents `VY_FWHM_GAUSS` as
  **record-only**, not sizing authority.
- Phase 2A still prefers `VY_FWHM_GAUSS` when setting `fwhm_px` for the LC path
  (`photometry_core` ~8580-8601).

If the DAO / AUTO / moment scale (~5.2-6.1 px) is closer to true seeing, then
`r/FWHM ~ 0.44-0.52` and Gaussian enclosed fraction at the production radius is
~42%, matching draft 435's undersized pattern (`r/FWHM=0.60` then). Aperture
losses that vary with seeing are a WIDE-ERR candidate.

JSON: `DRAFT_514_TRIAGE_D.json`.

---

## Proposal only (not implemented): field-independent cut

If a performance cut is wanted later, prefer an **absolute ceiling on
`sigma_eff` (mag)** relative to the target (and/or absolute floor on weight
`w = 1/sigma_eff^2`), evaluated per (target, star) only.

That is subset-invariant: adding or removing another star does not change
whether star i is admitted. It survives the COMP-ADMIT-03 subset test. A
"top N by weight" or "cumulative 99%" rule does not.

B2 says a soft absolute cut that removes only the far tail (stars that would
fall past ~99% cumulative on this field) should change ZP by << 1 mmag if the
`sigma_eff` model is honest. If such a cut moves ZP by mmag, the missing term is
in `sigma_eff`, not in ensemble size.

---

## Files changed (A1/A2; working tree)

- `src_py/sigma_floor_core.py` - lgamma c4
- `src_py/run_preflight_log.py` - real traceback
- `src_py/app.py` - preserve original exception
- `src_py/photometry_core.py` - AC `no_comp_rms` false positive
- `src_py/pipeline.py` - proc catalog dedupe in export core
- `dev/tests/test_sigma_floor.py`, `test_preflight_traceback.py`,
  `test_proc_dedupe_catalog_id.py`
- Tools/results: `dev/tools/draft_514_*.py`, `dev/results/DRAFT_514_TRIAGE_*.json`,
  this file

No push.
