CURSOR RESULT - COG-A1-01 (2026-08-14)

Curves of growth on draft 512; FWHM authority; INV gate fixes E1-E3.
Measurement and gate fixes are separate, uncommitted. No pipeline re-run; no aperture redesign implemented.

**Trial-run caveat (C-R4 / X-R3):** draft 512 was produced from `git_hash=4a3e855` with `git_dirty=true`. Results are valid physics on the archived FITS; they are not a reproducible reference until re-measured on a committed tree.

---

## Deliverables

| Artifact | Path |
|----------|------|
| This memo | `dev/results/CURSOR_RESULT_COG_A1_01.md` |
| COG curves (star x frame x radius) | `dev/results/draft512_cog_curves.csv` |
| COG per-star summary | `dev/results/draft512_cog_summary.csv` |
| Part C time series | `dev/results/draft512_cog_ee_timeseries.csv` |
| Part C stats JSON | `dev/results/draft512_cog_part_c_stats.json` |
| Measurement script | `dev/tools/cog_draft512_measure.py` |
| Gate tests (E1/E2 fire proof) | `dev/tests/test_inv_cal_sat_gates.py` |

---

## Part A -- FWHM authority and where 3.301 px comes from

### A.1 FWHM quantities in `photometry_core.py`

| Quantity | Definition | Measured on | Stage | Consumed by |
|----------|------------|-------------|-------|-------------|
| **`fwhm_per`** (per star) | Second-moment FWHM via `_fwhm_moment_at` on a local cutout | Aligned science pixel grid, star `(x,y)` | Per-frame catalog (`compute_fwhm_gaussian_for_aperture_catalog`) | Diagnostic column `fwhm_estimate_px`; median feeds fallback |
| **`fwhm_moment_med`** | `nanmedian(fwhm_per)` over catalog stars | Same frame | Same | Fallback `fwhm_gaussian = fwhm_moment_med * 0.619` when no override/header |
| **`fwhm_gaussian`** | Sizing FWHM: (1) `gaussian_fwhm_px_override`, else (2) `VY_FWHM * (1/1.5)`, else (3) `fwhm_moment_med * 0.619` | Override from MASTERSTAR header; `VY_FWHM` from frame header; moment from data | Per-frame export; **same override every frame** | `global_aperture_r_px`, annulus `r_in = max(r_ap+0.5, annulus_inner_fwhm * fw)`; exported as `fwhm_px_for_aperture` |
| **`VY_FWHM`** (header) | DAO-style moment FWHM stamped at QC/calibration | Calibrated (and aligned) FITS headers | QC + photometry | QC prefilter; fallback sizing; infolog median 5.195 px on 150 frames |
| **`VY_FWHM_GAUSS`** (MASTERSTAR) | 2D Gaussian fit on stacked MASTERSTAR | MASTERSTAR pixel grid | MASTERSTAR stack | **Record-only for SNR table**; drives `gaussian_fwhm_px_override` in `pipeline.py` |
| **SNR table `fwhm_px`** | Median DAO moment FWHM over first 12 aligned frames (`estimate_median_dao_fwhm_px_for_snr_table`) | Aligned science grid | Pre-export SNR table build | `compute_snr_optimal_aperture_table` only (radii 0.8-2.5 x this FWHM) |
| **Manifest `inspection.fwhm`** | DAOStarFinder robust FWHM at analyze | Raw/cal preview crop | Analyze / auto FWHM limit | `compute_auto_fwhm_limit` input (150 frames) |

**Gaussian vs moment:** For a pure Gaussian, second-moment FWHM and Gaussian-fit FWHM agree. Real PSFs have wings; the moment is sensitive to wings, the core fit is not. The code uses `DAO_TO_GAUSSIAN = 1/1.5` (Stetson/DAOPHOT convention) to map DAO `VY_FWHM` to an equivalent Gaussian core width for a circular aperture model.

### A.2 Which scale is correct for aperture sizing?

For **encircled-energy / growth-curve** sizing (Decision 4), neither header Gaussian nor a single draft constant is sufficient: the physically correct scale is the **per-frame PSF** (or per-star growth curve on that frame). For **SNR-optimal** sizing under a Gaussian model, the model FWHM must match the PSF scale **on each frame** (Howell 1989; Naylor 1998).

Using **`VY_FWHM_GAUSS` from MASTERSTAR (3.301 px)** while per-frame **`VY_FWHM ~ 5.19 px`** sizes apertures for a core ~57% narrower than the night's seeing. That is the root mismatch (O1).

### A.3 Where 3.3014 px comes from (D5-1 Q2 answered)

Trace on draft 512:

1. `Archive/Drafts/draft_000512/aperture_snr_table.json`: `vy_fwhm_gauss_px = 3.3014`, `vy_fwhm_dao_px = 5.19465`, SNR authority `fwhm_px = 3.389` (12-frame DAO moment median).
2. `pipeline.py` sets `gaussian_fwhm_px_override` from MASTERSTAR `VY_FWHM_GAUSS` (priority over `VY_FWHM * 0.667`).
3. Per-frame proc CSVs: `fwhm_px_for_aperture = 3.3014`, `fwhm_px_scope = per_draft_gaussian_override`.
4. Per-star radii **3.411-4.211 px** come from **`compute_snr_optimal_aperture_table`** (magnitude-dependent SNR optimum at `fwhm_px = 3.389`), not from the 3.301 factor alone.

**Draft 510 cross-check:** identical `fwhm_px_for_aperture = 3.3014` and `vy_fwhm_gauss_px = 3.3014` (same raw SHA256 as 512). The value is **inherited from the MASTERSTAR stack / SNR table path**, not from per-frame QC seeing on either draft.

**D5-1 Q1 (answered by data):** annulus and override FWHM are **per-draft frozen**; `aperture_r_px`, `r_in_derived`, `sky_annulus_r_out_px` are constant across all 134 frames (O2).

---

## Part B -- Measured curves of growth

**Method:** `dev/tools/cog_draft512_measure.py` on `detrended_aligned/lights/NoFilter_60_2/*.fits` (134 QC-ok frames). Positions from proc CSV `(x,y)`. Sky: plain annulus median (`_sky_pp_from_annulus_mask`, SKY-CLIP-01). Radii 0.5-15.43 px step 0.25 ( capped below `r_in = 15.682 px`). Science pixels unmodified.

### B.1 Asymptote

- **Primary:** median of last 4 radii when consecutive relative increments all `< 1%` (`tail_median_flat`).
- **Otherwise:** linear extrapolation by +0.5 px (`linear_extrap_+0.5px`) or last point (`last_point_not_flat`).

| asymptote_method | star-frame count |
|------------------|------------------|
| tail_median_flat | 696 / 804 (86.6%) |
| linear_extrap_+0.5px | 104 / 804 |
| last_point_not_flat | 4 / 804 |

**C-R1:** For **86.6%** of star-frames the curve reaches a flat tail inside `r_in`. **13.4%** depend on extrapolation; worst star **`1497974027502858240`** (flat fraction **19%** only) -- comp near annulus/crowding limit. Target and other comps: **100%** flat inside limit.

### B.2 Enclosed fraction at pipeline radius (`ee_at_pipeline_r`)

| catalog_id | role | `aperture_r_px` | median EE | median r90 (px) |
|------------|------|-----------------|-----------|-----------------|
| 1498613634033133184 | target | 4.211 | **0.846** | 5.31 |
| 1497771992240531712 | comp | 4.011 | 0.822 | 5.83 |
| 1499200223486564608 | comp | 4.211 | 0.828 | 5.61 |
| 1497368849430107904 | comp | 3.411 | 0.792 | 4.98 |
| 1499053747922698240 | comp | 3.611 | 0.811 | 5.13 |
| 1497974027502858240 | comp | 3.611 | 0.772 | 6.65 |

**Decision (4) check:** 90% enclosed needs **r90 ~ 5.0-5.75 px** on this data for most stars; at pipeline radii the target captures **~84.6%**, not 90%. Growth curves **support** the prior r90 range and **do not** support 90% EE at current radii.

---

## Part C -- Systematic in the light curve

### C.0 Detection floor (stated before amplitude)

**Method:** FWHM-correlated amplitude = `|slope(log10(ee_target/ee_ensemble) vs fwhm_px)| x (fwhm_max - fwhm_min) x 2500` mmag. **Floor:** 95th percentile of that amplitude under block bootstrap (5000 resamples, frame resampling). **Citation:** Efron & Tibshirani (1993); Kunsch (1989) block resampling.

| Quantity | mmag |
|----------|------|
| **Floor (95% bootstrap)** | **18.21** |
| Observed FWHM-correlated amplitude | 4.36 |
| **Above floor?** | **No** |

### C.1 Time series

Built from flux-weighted comp ensemble EE vs target EE at pipeline radii (`draft512_cog_ee_timeseries.csv`).

| Metric | Value |
|--------|-------|
| Static offset (median `delta_mmag`) | +39.1 mmag (target EE > ensemble EE) |
| Total p2p of `delta_mmag` | 139.0 mmag |
| Total std | 19.2 mmag |
| **FWHM-correlated amplitude** | **4.36 mmag** |

**Correlations (Pearson r):** `fwhm_px` **-0.04**; airmass **-0.10**; sky **-0.18**; target EE vs FWHM **-0.10**. Variation is **not** seeing-dominated.

**Vs draft scatter:** FWHM-correlated term (**4.4 mmag**) is below **check_scatter 9.3 mmag**, **ac_scatter 13.3 mmag**, **lc_rms_ooe 46.7 mmag**.

---

## Part D -- Cause, literature, recommendation

### D.1 Cause (single statement)

**Per-frame photometry uses `gaussian_fwhm_px_override = VY_FWHM_GAUSS` (3.3014 px) from MASTERSTAR for annulus geometry and exported `fwhm_px_for_aperture`, while the night's PSF in data is DAO/VY_FWHM ~ 5.19 px.** SNR table radii use a separate 12-frame DAO moment median (3.389 px). Both are frozen per draft; neither tracks per-frame seeing.

### D.2 Other packages (aperture sizing)

| Tool | Sizing rule | Per-frame? | Reference |
|------|-------------|------------|-----------|
| **DAOPHOT/DAOFIND** | `psfscale` / measured core FWHM | Yes | Stetson 1987, 1990 |
| **IRAF apphot** | User `scale` or measured FWHM | Typically per-image | IRAF apphot docs |
| **SExtractor** | `PHOT_AUTOM` optional; else fixed or FWHM from detection | Can be per-object/frame | Bertin & Arnouts 1996 |
| **photutils** | User-chosen radius; tutorials use FWHM from `DAOStarFinder` per image | User responsibility | Brady & Lauer 2020 |
| **AstroImageJ** | Fixed or **variable aperture ~ FWHM x k** | Variable mode: per frame | AIJ help / Collins et al. |
| **C-Munipack** | Configurable multiples of FWHM | Per frame option | Munipack docs |
| **VaST** | Aperture from FWHM estimate | Per frame | Sokolovsky & Lebedev 2018 |

### D.3 Options (systematics vs noise)

| Option | Systematics | Noise cost |
|--------|-------------|------------|
| **Per-frame FWHM-scaled radii** | Removes seeing drift in EE; comps/target scale together | Larger apertures in good seeing -> more sky variance (Howell 1989 trade-off) |
| **Fixed EE fraction (Decision 4, r90 from COG)** | Puts all stars on common flux scale; matches Stetson COG practice | Larger radii for faint comps -> higher sky term |
| **COG correction (`cog_aperture_correction_enabled`)** | Corrects current small-aperture bias in software | Needs bright unsaturated COG stars; adds model noise if `< min_stars` |

### D.4 Recommendation

1. **Stop using `VY_FWHM_GAUSS` as `gaussian_fwhm_px_override`** for science export; use per-frame `VY_FWHM` (or moment) for annulus and sizing authority (extends A-1 decision 2 consistently).
2. **Decision (4) remains the EE target:** size to measured **r90** from growth curves when Milan authorizes a sizing arc (this measurement gives r90 ~ 5.0-5.8 px).
3. **No LC fix implemented here:** Part C FWHM-correlated systematic **4.4 mmag < 18.2 mmag floor (C-R2)** -- no seeing-correlated term established at n=134 despite ~84% EE.

---

## Part E -- Bounded defects

### E1 -- INV-CAL-02 false PASS

**Finding:** Gate reported `cal_stage not stamped` while `VY_CALSTAGE=SKYSF_2`, `VY_CALDATASUM` verifies on calibrated FITS.

**Cause:** `check_cal_stage` only read `meta["cal_stage"]`; never inspected draft disk.

**Fix:** `_sample_cal_stage_from_disk` + evaluate headers/`cal_stage.json`. **Fire proof:** `dev/tests/test_inv_cal_sat_gates.py::test_fire_proof_draft512_sat_and_cal_gates` PASS.

### E2 -- INV-SAT-01 false PASS

**Finding:** Gate reported `sat_diag not stamped` while `Archive/Drafts/draft_000512/sat_diag.json` exists (schema 2, `pileup_detected=true`).

**Cause:** Path probe used `photometry_dir.parent.parent` (platesolve folder), not draft root.

**Fix:** `_draft_root_from_photometry_dir` + load `sat_diag.json`. **Fire proof:** same test module PASS on draft 512.

**Pileup vs 6 stars:** Pileup is field-wide (2617 pixels at 65535 ADU in 30 sampled raw frames). Measured stars peak **21360 ADU** max (target) << `lin_adu` 55705; **`likely_saturated=False`** on all 804 rows is correct for these stars. Pileup affects other field objects, not this differential set.

### E3 -- Auto-FWHM median inconsistency

**Not a math bug.** Two different populations:

| Stat | Value | Population |
|------|-------|------------|
| `median_fwhm` 5.311 px | manifest `inspection.fwhm` | 150 frames, DAOStarFinder at analyze |
| Limit 5.362 px | `median + k * MAD * 1.4826`, k=1.5 | same 150 values |
| VY_FWHM median 5.195 px | QC calibrated headers | Different estimator/stage |
| Retained max VY_FWHM 5.305 px | 134 QC-ok frames | Prefilter applied to **VY_FWHM**, not inspection.fwhm |

Reported `k=1.5` is **not** `median * k` (which would be ~7.97 px). **16** frames exceed limit on **VY_FWHM** at QC, consistent with `n_cut=16`.

**Code change:** none (documentation only).

---

## Pre-registered rule fired

**C-R2:** *"If the part C amplitude lies below the detection floor from the same section, the conclusion is that no seeing-correlated systematic is established at this frame count."*

FWHM-correlated amplitude **4.36 mmag < floor 18.21 mmag**. Also **C-R1** partial: 13.4% of COG curves extrapolation-dependent. **C-R4** stated above.

**C-R0:** Architect 11.5 mmag Gaussian prior was not used for validation. Measured FWHM-correlated term is **4.4 mmag**, below prior scale and below floor.

---

## Register diff (authorization)

| ID | Prior | Proposed after COG-A1-01 |
|----|-------|---------------------------|
| **COG-A1-01** | NEW | **CLOSED** (measurement + memo) |
| **A-1** | PARTIAL (510) | **CLOSED** (512 COG + FWHM authority documented; override bug identified) |
| **D5-1** | OPEN Q1/Q2 | **CLOSED** (Q1 per-draft; Q2 SNR table + MASTERSTAR GAUSS) |
| **Decision (4)** | Proposed | **ADVANCE** (r90 measured; implement awaits Milan) |
| **INV-CAL-02** | wired | **FIXED** disk detection + fire proof |
| **INV-SAT-01** | wired | **FIXED** draft-root path + fire proof |

---

## Files changed (uncommitted)

| File | Change |
|------|--------|
| `src_py/invariants_runtime.py` | E1/E2 gate disk detection |
| `dev/tests/test_inv_cal_sat_gates.py` | Fire proof tests |
| `dev/tools/cog_draft512_measure.py` | COG measurement |
| `dev/results/draft512_cog_*.csv/json` | Measurement outputs |
| `dev/results/CURSOR_RESULT_COG_A1_01.md` | This memo |

No commit/push without Milan authorization.
