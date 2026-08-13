# VYVAR audit 2026 ù Wave 4: scientific correctness

**Date:** 2026-08-13  
**Priority order:** flux/aperture/background ? ensemble/ZP ? errors ? calibration ? astrometry ? time/extinction ? detection/exports.

Literature citations are primary where stated; VYVAR implementation via `file:line`. Agreement classes: **Y** yes ù **D** deliberate ù **N** no ù **U** unmeasured.

---

## 4.1 Literature vs implementation matrix (condensed)

| Topic | Literature | VYVAR | Agree? | Class |
|-------|------------|-------|--------|-------|
| Aperture flux | Howell (1989) eq. 1ù2; Mighell 1999 | Circular aperture sum ? annulus sky; `photometry_core.py` dao/aperture paths | Y | ù |
| Background annulus | Labbe et al. 2003 empty-aperture sigma | Median annulus sky (`_annulus_sky_subtracted_flux`); no sigma-clip on annulus pixels. Sigma-clipped MAD only for empty-aperture error term, not sky subtraction. sigma_pp dropped. | Y | - |
| Ensemble ZP | Broeg et al. 2005 weights | Weighted median ZP; no MAD clip post-fix | Y | FIXED |
| Differential LC | Honeycutt 1992 | LOO comp median | Y | U on wide-rig err |
| Error budget | Howell 1989; Merline 1986 scintillation | Hybrid Poisson+RN+scint; I-11 path fixed | Y | WIDE-ERR open |
| Dark scaling | Howell 2006 ù4.2 | SUM resample when binning; CAL-DIAG derives | Y | U ?_p=0 edge |
| Flat combine | Standard median stack | Raw flat median; dark not subtracted at stack | D | D1-3 documented |
| Saturation | Full-well physics | SAT-DIAG placed aperture + 0.70 admission | Y | U linearity knee |
| CR rejection | van Dokkum 2001 | ABSENT (removed 0ab686f 2026-08-12); iron rule 2 - no science-pixel CR clean | D deliberate | DECISIONS |
| DAO detection | Stetson 1987 | DAOStarFinder; N_equiv on resampled | Y | T4-1 FIXED |
| Astrometry | Lang et al. 2010; SIP Calabretta | TAN+SIP fit + Grip optimizer | Y | ù |
| Time | Eastman et al. 2010 | astropy LTT ? HJD/BJD_TDB | Y | measured xval |
| Gaia transforms | Casagrande & VandenBerg 2018 | `gaia_johnson.py` | U | not re-run here |
| Colour extinction K2 | Casati et al. 2021 | `k2_extinction.py` literature mode | D | config |
| Airmass | Young 1994 | Standard formula via headers | U | not xval this run |
| COG correction | Stetson 1990 | Identified for A-1; **not applied** | N | A-1 OPEN |

Full July domain synthesis remains authoritative for fixed items: `docs/VYVAR_AUDIT_FINAL.md`.

---

## 4.2 Priority findings (open carry-forward)

### A-1 ù FWHM estimator disagreement (largest open science question)

| Estimator | draft 435 | draft 510 | Used for aperture |
|-----------|-----------|-----------|-------------------|
| MASTERSTAR Gaussian fit | ~2.40 px | ~3.30 px | **Yes** (SNR table) |
| Per-frame moment FWHM | ~5.14 px | ~5.31 px | No (QC only) |
| xval aligned-frame (2026-08-13) | ù | **2.96 px** | harness only |

**Factor:** 1.6ù2.1ù between moment and MASTERSTAR. If moment is correct, enclosed flux ~54% at current r (Step 1d fixture ~144 mmag G 8ù9).

**Verdict:** **UNRESOLVED.** Third estimator (xval 2.96 px) sits between extremes ù suggests aligned-frame seeing ? MASTERSTAR stack PSF ? per-frame QC FWHM definitions differ, not merely a bug.

**Disposition:** DOCUMENTED; COG/Stetson correction deferred.

### INV-CAL-01 ù degenerate `sigma_p = 0`

draft 510 cal_diag: `pedestal_sigma_p: 0.0` while PASS. Gate passes but uncertainty on pedestal is zero ù **condition does not enforce measurement quality** when intercept fit collapses.

**Class:** C/U ù **Status:** OPEN

### SAT-DIAG linearity

`lin_source=DEFAULT_FRAC` (85% of sat) ù spec-correct not to exclude on Tier-3 alone; warning recorded. **One-sided compatibility test** vs measured knee: U.

### WIDE-ERR

Wide-rig check-star errors ~2ù underquoted (Honeycutt SEM path). Fluxes unaffected. **OPEN.**

### T4-1

N_equiv=3.78 effective threshold on resampled frames ù **FIXED** batch E.

### P-02 / I-11 / I-04

**FIXED** batch D.

---

## 4.3 Numerical cross-validation (2026-08-13)

**Harness:** `src_py/xval_run.py` on draft_510 BO CVn (134 frames, 72 comp ids measured).

**Tolerance stated in advance:** comp RMS agreement **< 5 mmag** (differential method noise floor); target RMS **< 10 mmag** for confirmed variables.

### Results

| Metric | VYVAR | photutils | sep | DAO indep |
|--------|-------|-----------|-----|-----------|
| BO CVn target RMS | 0.1454 | 0.1456 | 0.1448 | 0.1455 |
| BO CVn comp RMS | 0.0105 | 0.0100 | 0.0100 | 0.0105 |
| Median target \|?\| vs VYVAR | ù | **0.0030 mag** | ù | ù |
| Median comp RMS (all targets) | dao 0.0102 | **0.0078** | **0.0076** | ù |

**Interpretation:**
- **Target differential LC:** photutils/sep/DAO agree with VYVAR at **? 3 mmag** median ù **PASS** tolerance.
- **Comp ensemble scatter:** photutils/sep **slightly lower** (~2 mmag) than VYVAR dao path ù likely background/centroid definition, not flux formula error.
- **FWHM estimate in harness:** 2.96 px vs SNR table 2.395 px ù **0.57 px offset** (19%) ù feeds A-1.

### Not run this session (Milan follow-up)

| Test | Blocker |
|------|---------|
| Centroid vs `photutils.centroid_sources` | Needs per-star position dump script |
| Background vs `Background2D` / `sep` mesh | Partially embedded in xval |
| Time vs astropy independent | Spot-check recommended on 3 frames |
| Airmass independent formula | Needs header dump |
| Gaia coeffs hand-eval | Needs band table export |
| Detection vs DAOStarFinder/sep.extract | Needs MASTERSTAR frame isolate |

---

## 4.4 Stage-by-stage verification status

| Stage | Verified how | Result |
|-------|--------------|--------|
| Aperture photometry | xval photutils 134 frames | **3 mmag** median target agreement |
| Ensemble/ZP | BO CVn check 0.008629; xval comp RMS | GREEN; 2 mmag comp offset |
| Error budget | Literature review + WIDE-ERR diag | anchor OK; wide OPEN |
| Calibration | CAL-DIAG PASS; INV-CAL-01 P1/P2 byte identity | PASS with ?_p caveat |
| Astrometry | INV-WCS-01 WARN band; closure history | DOCUMENTED |
| Time | astropy path; export BJD_TDB gate | IMPLEMENTED |
| Detection | T4-1 N_equiv | FIXED |
| Exports | OSC-03, BJD gate code review | IMPLEMENTED not exercised |

---

## Wave 4 closing

**Surprised:** xval shows **BO CVn target RMS matches photutils to 0.0002 mag** ù differential path validated independent of A-1 absolute aperture sizing debate.

**Could not determine:** Which FWHM estimator is "correct" without COG ground truth on same stars same frames.

**Next:** Hand-run centroid/Background2D spot checks; resolve A-1 with unified PSF model on MASTERSTAR vs single frames.
