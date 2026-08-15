# CURSOR RESULT - IMPL-03

Date: 2026-08-15
Baseline: 9762240 (IMPL-02)
Tip: (see commit after --fast)
Push: NO

## What I did

Changed aperture selection from Howell single-frame SNR to measured
light-curve scatter of non-variables (literature-aligned), fixed the SNR
bright-end ZP defect (Item 3), remasured draft 514 at the scatter-chosen
radius, rebuilt production LCs for the acceptance set, and restored
check_kmag sidecars.

## Literature (criterion)

| Source | Practice | Adopted? |
|---|---|---|
| **eleanor / TESS** (Feinstein et al. 2019, PASP) | Library of apertures; pick min **CDPP** on corrected LC | Yes - scatter/CDPP, not Howell |
| **Kepler / Smith et al. 2016** | Optimize apertures with CDPP (LC noise) | Yes |
| **C-Munipack / Muniwin** | Multi-aperture; user picks min Std.Dev on LC | Yes |
| **AstroImageJ** | User fixed px **or** FWHM-tracking; ~1.7xFWHM suggestion from radial profile | P3 measured both policies |
| **photutils / SExtractor** | Extraction tools; SNR-vs-r for single-frame SNR, not differential LC | Not used as authority |
| **Broeg-family** | Ensemble differential photometry; aperture usually fixed by user/seeing | Ensemble-per-r confirmed (P2) |

Architect's scatter-min design matches eleanor/Kepler/C-Munipack. No defect
found that required abandoning it.

## Item 1 - Scatter-optimal radius

**Ladder:** 1.5-12 px, step 0.5 px (matches CoG ladder granularity; spans deep
PSF core through ~2xFWHM at FWHM~5). One photometry pass; offline scatter eval.

**Chosen (draft 514):** **r = 4.5 px** (fixed pixels), EE(4.5) = **0.829**.
Selection-set MAD ~9.95 mmag; held-out at chosen ~12.34 mmag (held-out absolute
min at 5.5 px with 12.02 mmag - close; P1 reports both).

Artifacts:
- `dev/results/IMPL_03_scatter_scan.json`
- `dev/results/IMPL_03_aperture_scatter_table.json`
- draft `aperture_scatter_table.json` / production `aperture_snr_table.json` (flat)

### Pitfalls P1-P8

| ID | Finding |
|---|---|
| **P1** | Selection/held-out split (seed 51403). Choose on selection (4.5); report held-out at 4.5 (12.34 mmag) and held-out min (5.5 / 12.02). |
| **P2** | Full flux ladder per radius; ensemble rebuilt from comps at same r (not rescaled target). Confirmed in `aperture_scatter_select.py`. |
| **P3** | Fixed-px beat fixed-r/FWHM on held-out (12.02 vs 12.70 mmag). **Winner: fixed pixels for the draft.** Record for WIDE-ERR: seeing-tracking aperture did not win here. |
| **P4** | Mag-bin optima bright 3.5 / mid 2.5 (spread <=1 px); faint bin sparse. Treated as flat enough for **one radius per draft** (AIJ-style). |
| **P5** | Eval pool was isolation-biased (0% blended). Field catalogue: 53% blended at 2.711, **54% at 4.5**, 59% at 10.5. Chosen r is for isolated non-variables; do not treat as optimal for the blended half. |
| **P6** | AC on/off curves identical: star-independent AC offset cancels in differential mag. Method B does not change differential scatter. |
| **P7** | Measurability remains per-star. Neighbour saturation inside the aperture contaminates flux without a per-pixel gate; flagged in scan when nn < r and neighbour saturated. |
| **P8** | Shape: **sharp_min** on both selection and held-out (not flat). Minimum is meaningful. |

## Item 2 - Production before/after (draft 514)

Remasured proc CSVs at r=4.5 (`force_aperture_r_px.py` FITS pairing fixed for
`proc_*.csv` -> science FITS), Phase 2A on the 10-target acceptance subset.

`dev/results/IMPL_03_production_scatter.json` (mag_calib std mmag):

| Target | before | after | before r | after r |
|---|---:|---:|---:|---:|
| BO CVn | 146.5 | 146.2 | 4.211 | 4.5 |
| FW CVn | 14.83 | 14.38 | 4.411 | 4.5 |
| quiet best | 16.56 | 16.09 | 4.611 | 4.5 |
| quiet (was 2.711) | 28.25 | **13.97** | 2.711 | 4.5 |
| several fainter quiet | 17-60 | some worse | 2.7-3.4 | 4.5 |

**Finding:** Uniform 4.5 helps stars that were badly undersized; mixed/worse for
some that already sat near their local optimum. BO unchanged (intrinsic
variability). Aperture is not the whole noise budget - report, do not explain away.

### check_kmag

**Cause of null:** Phase 2A skip path logged only at DEBUG; string `"good"` in
exported quality maps could AttributeError inside ensemble; empty ensemble
returned None with no warning. Sidecars never written (0 files).

**Fix:** Accept string quality in `compute_check_ensemble_mag_calib`; WARNING
logs when check missing/empty; backfill wrote **49** sidecars.

Check scatter (after backfill), examples: BO check **10.4 mmag**, FW check
**12.9 mmag**. Some quiet checks are pathological (~222 mmag) - selection quality
issue separate from aperture radius.

## Item 3 - Bright-end SNR inconsistency

**Cause named:** hardcoded `zero_point=25.0` in `compute_snr_optimal_aperture_table`.
Draft-514 calibrated ZP from dao_flux/EE: **ZP = 22.42** (n=7063, MAD 0.42).
Factor ~10^((25-22.42)/2.5) ~ **10.8x** flux overstatement -> bright r_opt too large.
Architect EE reconstruction agrees; architect bright optima were right given real fluxes.

Corrected SNR table (still for SNR mode / diagnostics; scatter is aperture authority):

| G | ZP=25 r | ZP=22.42 r |
|---:|---:|---:|
| 8 | 8.499 | **4.499** |
| 10 | 5.499 | **2.499** |
| 12 | 2.999 | **1.999** |
| 14 | 1.999 | **1.499** |
| 16 | 1.699 | **1.499** |

Artifacts: `dev/results/IMPL_03_item3_zp.json`,
`dev/results/IMPL_03_aperture_snr_table_zp_corrected.json`,
draft `aperture_snr_table_zp_corrected.json`.

## Code / config

- `src_py/aperture_scatter_select.py` - ladder, curves, ZP calib, flat table
- `src_py/photometry_core.py` - ZP into SNR precompute; load prefers scatter table;
  check-kmag warnings + quality hardening
- `src_py/check_star_kmag.py` - string quality accepted
- `src_py/config.py` + `params_registry.json` - `aperture_selection_criterion`,
  `aperture_scatter_r_{min,max,step}_px` (units: enum / px)
- `dev/scripts/force_aperture_r_px.py` - proc_/FITS pairing
- `dev/tools/impl_03_scatter_aperture_scan.py`
- `dev/tests/test_impl_03_scatter_aperture.py`

## --fast

(see stamp after suite)

## Files changed

(list + SHA after commit)
