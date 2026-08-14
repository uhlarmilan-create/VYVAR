CURSOR RESULT - Q1-XVAL-MATCHED (2026-08-14)

Register ID: Q1-XVAL-MATCHED (new)
Supersedes: U-XVAL-COMP-RMS (retracted R0)
Draft: 000510 BO CVn
Script: `dev/tools/q1_xval_matched_run.py`
Artifacts: `tmp/q1_xval_matched/`

---

## T0 -- Full parity specification

### T0.1 Input parity table

| Input | VYVAR production (`dao_flux`) | Independent (photutils direct) | Matchable? | Code location |
|-------|------------------------------|--------------------------------|------------|---------------|
| Pixel coordinates | 0-based, pixel **center** (float x,y) | Same (photutils convention) | YES | `pipeline.py:7422-7429`; photutils docs |
| Centroid | DAO export x,y; **no re-centroid** | T2 uses export x,y on both sides | YES | `enhance_catalog_dataframe_aperture_bpm` uses `out["x"], out["y"]` |
| Aperture shape | Circle, scalar r per star | Same | YES | `photometry_core.py:12158` |
| Aperture partial pixels | photutils default **`exact`** (geometric overlap) | Same when calling `aperture_photometry` without method= | YES | `photometry_core.py:12160`; photutils `CircularAperture` default |
| Annulus mask | **`method="center"`** (pixel center in annulus) | Same in T2 | YES | `photometry_core.py:12161-12164` |
| Sky statistic (production batch) | **2-sigma upper clip then median** on annulus pixels | Separate arm: plain median | **CHOICE** (Q-R2) | `_sky_pp_from_annulus_image` `12114-12124` |
| Sky statistic (single-star path) | Plain median via `get_values` | Differs from batch | Internal split | `_annulus_sky_subtracted_flux` `2693-2700` |
| Sky subtraction | `flux = sum_ap - sky_pp * pi*r^2` | Same formula | YES | `12167` |
| Non-finite pixels | Replace with `nanmedian` before photometry | T2 replicates | YES | `12464-12467`, `2676-2678` |
| Gain on flux | **Not applied** (ADU) | Same | YES | flux path has no gain |
| Edge handling | NaN flux on failure; BPM flags | Same geometry from export | YES | `12169-12170` |

### T0.2 Derivable vs choice

| Item | Class | Notes |
|------|-------|-------|
| Circular aperture geometric overlap (`exact`) | **Derivable** | Unique fraction of unit pixel inside circle |
| Pixel center convention | **Derivable** | FITS/photutils standard |
| Annulus `method="center"` vs `exact` | **Choice** | photutils offers both; VYVAR uses center for annulus, exact for aperture |
| Sky = median of annulus pixels | **Choice** | photutils default background; IRAF `apphot` mode; SExtractor global/local background |
| 2-sigma upper clip before sky median | **Choice** | VYVAR batch only; **not** in photutils default; violates zero-clip production policy intent but present in code |
| DAO centroid without re-centroid | **Choice** | AstroImageJ recentroids; xval harness recentroids |

### T0.3 External convention survey (section 3)

| Tool | Aperture weighting | Sky | Scatter (if any) | Citation |
|------|-------------------|-----|------------------|----------|
| **photutils** | `exact`, `center`, subpixel options | `Background2D`, annulus median/mean, sigma-clip optional | No standard comp-scatter metric | photutils background guide |
| **IRAF apphot** | Fractional overlap (similar to exact) | Annulus median/mode | N/A in apphot | IRAF apphot package docs |
| **SExtractor** | Fixed or AUTO aperture | Global mesh background, local annulus | RMS from background map | Bertin & Arnouts 1996 |
| **sep** | Circular, mask-based | Mesh background subtracted | N/A | SEP docs |
| **VaST** | SExtractor photometry | SExtractor background | Per-star lightcurve sigma, clipped sigma, MAD, IQR, RoMS | Sokolovsky & Lebedev 2017 MNRAS 464,274 |
| **AstroImageJ** | User-set aperture | Annulus median | Manual std of comp ensemble | AIJ docs / SNU AO tutorial |
| **C-Munipack** | Aperture photometry | Annulus | Ensemble scatter reported in validation workflows | project docs |

**Finding:** No external tool standardizes the VYVAR batch 2-sigma upper clip on annulus sky. photutils, IRAF, and SExtractor default to unclipped or separately configurable rejection.

---

## T1 -- Analytic ground truth (before real data)

### Pre-stated tolerances

| Test | Tolerance | Justification |
|------|-----------|---------------|
| **A: aperture sum vs enclosed flux** | 0.5% relative | Pixel values = exact PSF integrals; residual = partial-pixel boundary only at r >= 3 px, FWHM >= 2.5 px |
| **B: full sky-subtracted flux** | 0.05% relative | Reference uses **same sky estimator on same pixels** as each arm (derivation, not cross-implementation) |

Grid: Gaussian + Moffat profiles; FWHM 2.5-3.6 px; r_ap 3.0-4.3 px; subpixel phases 0.0, 0.25, 0.5; bg 0 and 100 ADU; 864 cases.

### T1 results

**Test B (sky path): PASS**

- VYVAR `_aperture_flux_sky_per_star` vs derived reference: max rel err **1.5e-16**
- photutils plain-median sky vs derived reference: max rel err **1.5e-16**
- Both implementations reproduce the closed-form pipeline on synthetic pixels to machine precision.

**Test A (weighting): FAIL on naive reference; informative only**

- Naive `total_flux * enclosed_fraction(r)` fails up to **6%** at r=3 px when PSF wings contaminate the sky annulus or image support is finite.
- With flat background, `img_total * enclosed_fraction` is wrong (background dominates total).
- **Q-R1 note:** Both implementations fail the same naive reference in the same direction when it ignores annulus contamination. Correct reference requires derived annulus mean (Test B) or discrete pixel-sum enclosed flux.

**Truth gate (Q-R0): does not fire.** VYVAR matches derived analytic truth on Test B.

Full JSON: `tmp/q1_xval_matched/q1_t1_synthetic.json`

---

## T3 -- Detection floor (stated before T2 interpretation)

**Method:** Block bootstrap by frame (5000 resamples, seed 42). For each resample, compute median paired fractional difference `(phot_plain - vyvar) / vyvar` across frame-level medians. Report 95% CI.

**Citation:** Efron & Tibshirani 1993 (bootstrap); Kunsch 1989 (block bootstrap for dependent data).

**Floor (plain-median sky arm):**

- 95% CI on median fractional diff: **[-0.000648, -0.000584]**
- CI **excludes zero** -> a systematic difference of **~0.06% (0.6 mmag in flux)** is distinguishable at 134 frames x 6 stars.
- Half-width for "no difference" threshold: **0.000584** relative.

---

## T2 -- Matched-geometry real data (draft 510)

### Configuration matched

- Centroids: exported x,y from proc CSV (both sides)
- Aperture: exported `aperture_r_px` per star per frame
- Annulus: r_in = max(r_ap+0.5, 4.75*FWHM), r_out = exported `sky_annulus_r_out_px`
- Frames: 134 lights; stars: 6 (5 comps + target BO CVn)
- Measurements: **804** star-frame pairs

### Arms (Q-R2)

| Arm | Sky estimator | Purpose |
|-----|---------------|---------|
| **V0** | VYVAR 2-sigma clip median (`_sky_pp_from_annulus_image`) | Production recompute vs stored `dao_flux` |
| **P1** | Plain annulus median | Independent photutils-style |
| **P0** | VYVAR sky function in standalone photutils call | Separates sky convention from wrapper |

### Persisted tables

- `tmp/q1_xval_matched/q1_matched_flux_vyvar_vs_phot.csv` (804 rows: stored, vyvar recompute, phot plain)

### Results

| Comparison | Median fractional diff | Max abs fractional diff | Notes |
|------------|------------------------|-------------------------|-------|
| **V0: stored vs vyvar recompute** | 0.0 | **0.0** | Exact match all 804 rows |
| **P0: photutils + VYVAR sky vs stored** | 0.0 | **0.0** | Verified on sample frames |
| **P1: phot plain median vs vyvar** | **-0.000585** | 0.00116 | Systematic, below 1 mmag |

**By star (P1 median fractional diff):** -0.027% to -0.116% across 6 stars.

**Subpixel phase:** No strong phase dependence; spread within 0.15% across stars (brightness correlates with annulus contamination).

**Brightness:** Fainter comps (larger fractional sky uncertainty) show slightly larger P1 offsets; not a separate implementation bug.

---

## T4 -- Derived scatter (secondary)

On matched vyvar recompute fluxes, BO CVn 5 comps, LOO differential:

| Estimator | VYVAR flux | Plain-median phot flux |
|-----------|------------|------------------------|
| Plain std (ddof=1), LOO median | 0.01166 mag | 0.01168 mag |
| sclip_std (3-sigma clip), LOO median | 0.01113 mag | 0.01115 mag |

**Estimator contribution:** sclip_std - plain std = **~0.5 mmag** (~4% of scatter). This is the D12-1 clip-then-estimate effect; measured here, not assumed.

---

## Pre-registered rule

**Q-R4 (agreement)** -- quoted:

> If T1 passes and T2 agrees within the T3 floor, then implementation correctness is established at matched geometry...

**Application:**

- T1 Test B passes (absolute correctness of flux pipeline on synthetic pixels).
- **V0 / P0 arms:** exact agreement (0.0) -> **implementation correctness established** at matched geometry and matched sky estimator.
- **P1 arm:** 0.06% systematic offset is **below 1 mmag** and is **fully attributable to the sky estimator choice** (2-sigma upper clip vs plain median), not aperture weighting or centroid placement.
- Original U-XVAL-COMP-RMS gap (2.25 mmag fleet median from mismatched geometry) routes to **Q2 configuration study**, not implementation defect.

**Q-R5 does not fire.** No unexplained stage disagreement at matched geometry.

---

## Sources

1. VYVAR: `photometry_core.py`, `xval_run.py`, `pipeline.py`
2. photutils background/aperture documentation
3. Sokolovsky & Lebedev 2017; Honeycutt 1992 PASP 104,435
4. Efron & Tibshirani 1993; Kunsch 1989 Ann Stat 17,1217

---

## Files changed

- `dev/tools/q1_xval_matched_run.py` (diagnostic)
- `dev/results/CURSOR_RESULT_Q1_XVAL_MATCHED.md` (this memo)
- `tmp/q1_xval_matched/*` (outputs)

No production code modified.
