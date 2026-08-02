# VYVAR -- Science audit (12 domains, final synthesis)

**Date:** 2026-07-30 (tranches 1--4) through 2026-07-31 (Stage 3 forensics)
**Status:** AUDIT COMPLETE -- remediation tracked in `VYVAR_AUDIT_CLOSURE_REGISTER.md`
**Method:** Read-only code audit + literature verification (R1--R5) + production-path measurement
**Evidence:** `dev/results/CURSOR_RESULT_audit_t{1,2,3,4}.md`, `dev/results/CURSOR_RESULT_audit_stage{0,1,2,3}_*.md`

This document consolidates the science audit across twelve domains. Individual tranche reports
retain full derivations; here each domain states the verdict, severity, and disposition.

---

## Domain map

| # | Domain | Primary IDs | Severity (open) |
|---|--------|-------------|-----------------|
| 1 | Calibration (dark/flat/linearity) | D1-1, D1-2, D1-3 | MED |
| 2 | Noise model / background statistics | I-11, P-02, T4-1 | HIGH |
| 3 | Preprocess / sky surface | P-10 | **FIXED** |
| 4 | Astrometry / Gaia epoch | I-12, GAIA-1 | MED (PM deferred DR4) |
| 5 | Aperture photometry / curve of growth | D5-1 | MED (measured) |
| 6 | PSF / ePSF | -- | LOW (not re-opened) |
| 7 | Detection / DAO / MASTERSTAR | T4-1, S3 | HIGH (blocks re-cut) |
| 8 | Ensemble / differential photometry | I-04 | MED |
| 9 | Error budget / check-star chi2 | P-02, 1c | MED |
| 10 | Magnitude system / transforms | D10-1, D10-2 | DECISION (CR vs CV) |
| 11 | Dilution / crowding | -- | INFO |
| 12 | Statistics / export truth | T1-B, clipping | **FIXED** (export) |

---

## 1 -- Calibration

**D1-3 (master flat construction):** Documented in `VYVAR_DECISIONS.md`. Master flats are median
stacks of **raw** flats without dark subtraction at stack time; `VYFLNRD=1` normalizes at
calibrate-after-resample. Open gap vs Howell requirement that additive terms be removed before
normalization -- scope documented, not verified end-to-end in builder UI.

**D1-2 (detector linearity):** Batch B (2026-08-02) **B-open**. B2 partial deficit~peak | r50 = +0.37
(G 9-11 reference), below +0.4 pre-registered threshold. Faint-half reference slope -0.18
(invalid). Status: **DEFERRED**.

**Closure batch B (2026-08-02):** B-open -- B1 VOID (sanity gate); B2 inconclusive. D5-2 mechanism
**DEFERRED**. Report: `dev/results/CURSOR_RESULT_batch_B.md`.

**Closure batch A (2026-08-02):** A-1, A-9, D1-1, U-09 **DOCUMENTED**; `docs/VYVAR_LIMITATIONS.md`
created. Report: `dev/results/CURSOR_RESULT_batch_A.md`.

**D5-2 (production flux vs catalogue magnitude):** Confirmed. Raw slope **-0.296** (`flux`),
**-0.269** (`flux_large` fixed radius). Localisation: **G 8-9** bin. Mechanism **DEFERRED**
(batch B-open). Status: **MEASURED**.

**D1-1 (cosmic-ray rejection):** **DOCUMENTED**. Absent from `src_py`. Scheduled batch E
(van Dokkum 2001 L.A.Cosmic or astroscrappy). CR-1 **QUEUED**.

---

## 2 -- Noise model / background statistics

**I-11 (Howell sky term on sky-subtracted frames):** HIGH on fallback path. Annulus sky on
subtracted image collapses sky Poisson term; hybrid clamp `BKG_SCALE_R_CLAMP_HI=2.0` can still
under-quote ~2x. **0 production epochs** on anchor today; engages in crowded fields where
empirical sigma fails. Fix options documented (pre-subtraction `sky_surface_bg_median_adu` preferred).

**P-02 (variance budget):** Inconsistent between setups -- NoFilter_60_2 check stars chi2_red > 1
(under-quoted err); Newton bin4 shows deficit. Scintillation formula implemented but **not wired**
(DECISION REQUIRED).

**T4-1 (correlated noise after astroalign):** Nominal 3.8 sigma threshold becomes ~3.3--3.58
effective on resampled `detrended_aligned` frames. Neither sigma_pp nor Background2D fixes
correlation. Options A--D in Tranche 4; **blocks anchor re-cut** until Milan chooses.

**Estimator (sigma_pp vs sigma_clipped_stats):** Post-P-10 measurement: median ratio 1.034,
twilight frame 001 at +8.3%. Milan decision: **drop sigma_pp**, revert to sigma_clipped_stats
(Stage 3).

---

## 3 -- Preprocess / sky surface

**P-10 (CRITICAL -- FIXED):** `_fit_subtract_preprocess_sky_surface` fitted `z = bg_median - work`
then subtracted, **doubling** large-scale gradient (ratio ~2.00 on synthetic test; SKYSF forensics
283/136.8 = 2.07). Fixed: `z = work - bg_median`. Independent regression test fails pre-fix /
passes post-fix. **SKYSF-DOUBLE guard** (2026-07-30) prevents in-place double subtract.

Byte-identity gates did not catch P-10 (reproducibility != correctness). Doctrine: at least one
gate per physical step must compare against independent expectation.

---

## 4 -- Astrometry / Gaia epoch

**I-12 (PM correction -- logging FIXED):** `_apply_proper_motion` math correct; local DR3 build
lacks `pmra`/`pmdec` so correction is silent no-op. WARNING logged when PM columns absent.
Centroid-vs-catalog residual gate suggested; not implemented.

**GAIA-1/GAIA-2:** Deferred to Gaia DR4 build (~Dec 2026) per DECISIONS.

**Stage 3 Part 0e finding:** Per-frame DAO centroid instability (`detect_stars_match_master_reference`)
can mis-centre apertures on correct `catalog_id` -- distinct from WCS or neighbour confusion.

---

## 5 -- Aperture photometry

**D5-1:** Stage 2 measured per-frame FWHM tracking (10.8% spread); target aperture 2.716 px
constant; comp median 2.416 px. LC residual vs FWHM slope median -3.5 mmag/px (scattered).
Stage 1.2 provenance columns added (`aperture_factor_applied`, `fwhm_px_for_aperture`, etc.).

**Closure Step 1 (2026-07-31, finding A-1):** Anchor draft_435 uses SNR-table radii from
`aperture_snr_table.json` (`fwhm_px = 2.395`, `r_min_px = 1.916`). Focus target at clamp for
all frames. Register inference via `aperture_fwhm_factor` was wrong code path (H0 confirmed).

**Closure Step 1d (2026-07-31):** Unit bug fixed (`delta_ap` in **mmag**, `* 1000`). Independent
fixture `closure_a1_reference_fixture.py` (L2 photutils) expects **+144 mmag** G 8-9 range over
anchor r50 span. Step 1d reported max **203 mmag** -> **A-1b CONFIRMED**; **203 mmag VOID** (V8).

**Closure Step 1e (2026-08-01):** Measurement-method repair (photutils exact COG, Gaussian
centroid). G7 PASS; G6 failed due to full-curve admissibility defect (VOID contamination claim;
fixed in Step 1f).

**Closure Step 1g (2026-08-01):** F1 restores A-1 differential configuration (proxies G 11.5-13.0
at clamp 1.916 px, excluded from comp subsets). G7/G8/G9 PASS; **G6 FAIL** (proxy p95-p5 spread
4.8x on G 8-9). Brightest proxy G 11.52: G 8-9 range **94 mmag** vs fixture **144.3 mmag**.
Step 1f **48.0 mmag** VOID (proxies inside comp set). Exact consolidated magnitude **open**.

**Closure Step 1i (2026-08-01):** Mechanism for EE(1.916) excursions (I3). E5: placement not
dominant. HIGH EE: normalisation path; LOW EE: rare placement (one frame 3.17 px). Step 1h
"SNR-driven" label rejected.

**Closure Step 1j (2026-08-01):** J-a -- F(12) fails catalogue consistency (J1 slope **-0.285**
vs **-0.4**); G 11.52 vs 11.53 (dG = 0.006 mag) F(12) ratio **2.6x** from per-star sky offset
(~33 ADU/px). Step 1i production annulus claim **withdrawn** (25-45 px is harness-only;
production uses 4.75-9.0 x FWHM per star). J3: narrow/production annulus reduces EE std for
G 11.53. J2: field-centre correlation -0.53 on annulus sky offset.

**Closure Step 1k-1n (2026-08-01):** D5-2 confirmed; mechanism open after Step 1n (N-none).

**D5-1 Q1 (does aperture FWHM track per-frame seeing?):** **No.** Single draft-constant
`VY_FWHM_GAUSS` from MASTERSTAR drives SNR table; `aperture_r_px` constant per star across
frames.

**D5-1 Q2 (are role factors applied to science radii?):** **No** on per-frame export path;
`aperture_comp_factor` appears in label strings only (S3). The "target 1.0x vs comp 1.1x radius"
framing is **superseded** -- differing radii come from magnitude bins in the SNR table, not role factors.

Prior Stage 2 statement "radii track per-frame FWHM" is **VOID** for anchor draft_435 (R7);
radii track magnitude bins from a draft-constant FWHM table.

---

## 6 -- PSF / ePSF

Not re-audited in July 2026 tranches. Prior `VYVAR_EPSF_AUDIT.md` stands. ePSF is prerequisite for
TODO-B proper coaddition, not beneficiary of stack reference alone.

---

## 7 -- Detection / DAO / MASTERSTAR

**Threshold recalibration:** `masterstar_dao_threshold_sigma` 2.1 -> 3.8 after P-10 bundle
(restores effective detection depth once gradient inflation removed).

**Tranche 4:** DAOFIND `scale_threshold=True` convention **correct**; M3 (Background2D as fix)
**withdrawn**. Real defect is post-alignment noise correlation.

**Part 2b sweep (correct path):** log-log slope -1.58; DAO_ONLY moves with N on
`detect_stars_and_match_catalog`; **no N selected** (R5). Legacy N_equiv ~3.78--4.71 depending
on rel_err measurement.

**Single-frame MASTERSTAR:** Non-standard vs literature (Stetson 1994 ALLFRAME). See
`VYVAR_MASTERSTAR_REFERENCE_ARCHITECTURE.md` and `VYVAR_TODO_MASTERSTAR_REFERENCE.md`.

**Part 0c/0d/0e:** Delta-tail forensics found Part 0c **positional pairing bug** (not source_file);
3.36 mag headline invalid; correct pairing median p95 |delta| 0.104 mag. Part 0e: focus target tail
from **DAO centroid shift** (3.48 px on frame 063), not ensemble or neighbour swap.

---

## 8 -- Ensemble / differential photometry

**I-04 (ensemble scatter on unmatched epochs):** Anchor has 0 epochs with
`err_scatter_unmatched=True`. Policy choice pending: NaN+exclude vs flagged inflation.

Ensemble recomposition explains part of valid-pairing delta tail (Part 0d); sub-aperture shifts
explain extreme cases (Part 0e).

---

## 9 -- Error budget / check-star chi2

**Part 1b indexing bug:** Reported chi2=649 was **total chi2**, not chi2_red (index [0] vs [2]).

**Part 1c corrected:** Median chi2_red ~4.7 (not ~649). Clipped median ~4.0. Top outliers from
**shared bad frames** (e.g. frame 123 saturation), not cosmic-ray spikes.

**sigma_sys_mag = 0** on equipment_id=1 (BO CVn wide rig) -- documented; literature floor debate
open for differential photometry.

---

## 10 -- Magnitude system / transforms

**D10-2 (Gaia->Johnson range):** Guard matches Gaia DR3 Table 5.10; 1 comp star G=7.99 outside
range on anchor; 39 masterstars bright outliers.

**D10-1 (unfiltered CV vs CR):** R transform flatter (slope +0.107 vs V -0.386; scatter 0.043 vs
0.129 mag). Milan decision: **switch unfiltered comparison to Cousins R** (Stage 3 implemented).

---

## 11 -- Dilution / crowding

Crowded-field empirical sigma fallback (I-11 interaction) documented. No new July finding beyond
prior comp-selection and dilution specs.

---

## 12 -- Statistics / export truth

**T1 Group B (FIXED):** AAVSO/VarAstro export refuses non-BJD_TDB time bases; `#DATE=BJD` only
after guard passes.

**T1 Group A (FIXED):** Docstrings, HJD/BJD error logging, Kasten-Young citation wiring.

Clipping bias in check-star chi2: 1.4% median outlier fraction at 3-sigma iterative clip.

---

## Audit completion statement

All twelve domains have been reviewed. Critical sign error P-10 is **fixed and tested**.
Measurement stages 0--3 and tranches 1--4 are **committed** (`dev/results/`). Open work is
**not** further audit discovery but **closure** of registered items -- see
`VYVAR_AUDIT_CLOSURE_REGISTER.md`. Anchor re-cut remains **blocked** on T4-1 detection-noise
decision and DAO-centroid / pairing fixes from Stage 3 Part 0d--0e.

**Next work item:** Aperture closure **Step 2** (A-2/A-3 placement; DAO centroid coupling).
A-1 **DOCUMENTED** (batch A); **A-9 DOCUMENTED**. D5-2 **MEASURED**, mechanism **DEFERRED** (batch B-open).
MASTERSTAR stack **A-1** (`I_j`) queued separately.

---

## Primary references

| Report | Content |
|--------|---------|
| `CURSOR_RESULT_audit_t1.md` | Export truth, time base, err scatter surfacing |
| `CURSOR_RESULT_audit_t2.md` | P-10, I-11, I-12 |
| `CURSOR_RESULT_audit_t3.md` | Bundle causal chain, threshold 3.8 |
| `CURSOR_RESULT_audit_t4.md` | Correlated noise, M3 retraction |
| `CURSOR_RESULT_audit_stage0.md` | Estimator measurement, P-10 verify |
| `CURSOR_RESULT_audit_stage1.md` | D10-2, D5-1 provenance, D1-3 |
| `CURSOR_RESULT_audit_stage2.md` | D5-1, D10-1, D1-2, P-02, U-09, T4-1 |
| `CURSOR_RESULT_closure_step1.md` | A-1 Step 1 (superseded VOID items) |
| `CURSOR_RESULT_closure_step1c.md` | A-1 Step 1c harness audit; delta_ap 0.203 mmag; A-1 DOCUMENTED final |
| `CURSOR_RESULT_audit_stage3_part0c.md` -- `part0e.md` | Rebuild delta tail forensics |
