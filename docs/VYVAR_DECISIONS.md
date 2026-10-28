# VYVAR - Decisions & rationale

Durable design decisions and *why* they hold. This is the reference for "why is it like
this" - it should not be reopened without a new decision recorded here. Per-draft validation
numbers and the day-by-day record live in `VYVAR_JOURNAL.md`; open work in `VYVAR_ROADMAP.md`.

---

## D-EPSF-CHI2-LOCUS-01 (Milan LOCK 2026-09-30) - ENACTED

Fixed ``psf_chi2_threshold=50`` SET replaced by data-derived chi2(flux)
locus (night slope k + per-frame intercept a_f; ``psf_chi2_locus_nsigma``
default 5.0). Evidence: SAT-CHI2-01 (bright comps MODEL not SAT),
EPSF-PIN-FITOK-01 (product-wide pin collapse), Phase 1/1b
`CURSOR_RESULT_EPSF_CHI2_LOCUS_01.md`.

era06 product: epsf01 **c94cf4fe** n=53; G3 BO/FW demeaned residual
RMS **15.372 / 5.360** mmag, n_full=134 each. Aperture unchanged
era05 **87197716** / **dd92e99d**. era05 epsf01 **552ace75**
SUPERSEDED (collapsed under fixed chi2=50). EPSF-XVAL-01 and
D-EPSF-XVAL-CLOSE-TEXT-02 stay CLOSED/ENACTED with the note that
era05 PSF LCs were collapsed and era06 is the verified product.

``psf_chi2_threshold`` remains LEGACY in the params registry (removal
is CHI2-THRESH-LEGACY-01 debt, D-style).

## D-EPSF-XVAL-DOD-06 (Milan 2026-09-28) - criterion 2 met; closure sequence fixed

EPSF-AC-02 (`672a6a7`): R-AC1 PASS (b = +2.68 +/- 21.5 mmag/mag,
simultaneous with colour; n=18 isolated stars), R-AC2 PASS
(split-half per-star offset stability 5.59 odd/even, 6.60
chronological, noise expectation 4.83 mmag - at the noise floor),
R-AC4: criterion 2 met on 516. Architect re-verification: values
bit-match the per-star table.

Caveat on the record: two of 36 split tests exceed 3x their own
noise (1498735778606786816 chrono -14.5; 1497145751650265600
odd/even -14.6 mmag). Per-star offsets may wander by ~10-15 mmag
for a minority of bright stars; not a FAIL; listed as a limitation
in D-EPSF-XVAL-CLOSE-TEXT-02.

Architect error 31: the CORE-03 T1 phase surface was predicted to
drive the per-star common-scale offsets (+/-36 mmag); AC-02 Part C
gives rho = 0.13, p = 0.61 - NOT confirmed. The phase mechanism
remains established only for the target's per-epoch component
(CORE-01 phase rank R^2 0.221; ~2-5 mmag RMS). The ~37 mmag
per-star residual after colour is UNATTRIBUTED (bright-end model
mismatch is the plausible but unmeasured cause;
PSF-COMMON-SCALE-RESIDUAL-01).

Status on 516: criterion 1 PASS (VAL-02); criterion 2 PASS
(EPSF-AC-02); criterion 3 pending ERA-520-RECUT-01. Closure
sequence: ERA-520-RECUT-01 (criterion 3) -> re-run of the VAL-02
(criteria 1) and AC-02 (criterion 2) harnesses on the 520 products
(a re-check, not a re-registration) -> enact
D-EPSF-XVAL-CLOSE-TEXT-02 (supersedes CLOSE-TEXT-01).

Evidence: `CURSOR_RESULT_EPSF_AC_02.md`,
`CURSOR_RESULT_LEDGER_EPSF_DOD_06.md` (error 31).

## D-EPSF-XVAL-CLOSE-TEXT-02 (Milan 2026-09-28) - ENACTED 2026-09-29 (ERA-520-RECUT-01)

Enacted after criteria 1-3 held: criterion 1 (VAL-02 re-check on
era05: R-W1+R-W2 PASS; lc_flux_method VERIFIABLE), criterion 2
(AC-02 R-AC1/R-AC2 PASS), criterion 3 (FIXPOS/FIT-OK/GAIN unit
tests + era05 cut). Evidence: CURSOR_RESULT_ERA_520_RECUT_01.md.
G3 residual n_full collapse post FIT-OK recorded there (Milan
sequel); not a criterion 1-3 FAIL.

EPSF-XVAL-01 closes with this statement, verbatim, in DECISIONS and
in the methods paper:

  VYVAR ePSF photometry is implemented correctly (model construction
  verified against an independent PSFEx reference: FWHM within 2%,
  shape not a limiting factor; fit machinery self-consistent to
  ~1 mmag), externally cross-validated (closer to the AIJ-validated
  aperture path than PSFEx on the same 134 epochs), and validated
  for its intended, differential use: precision equal to aperture
  for G >= 9.5 (rms ratio median 1.05, max 1.42), bright-end fits
  gated by psf_fit_ok with aperture fallback verified end-to-end,
  per-star scale offsets stable at the noise floor (5.6-6.6 mmag
  split-half) and therefore absorbed by ensemble normalization,
  and a flux-scale slope versus brightness consistent with zero
  within +/-20 mmag/mag (isolated stars) and within ~15 mmag/mag
  (95-star catalogue-tied core).
  Known limitations, stated: (i) common-scale (absolute) PSF
  magnitudes carry a ~40 mmag per-star systematic, of which a
  chromatic term of +138 +/- 48 mmag per mag of BP-RP is measured
  and ~37 mmag remains unattributed; absolute PSF magnitudes are not
  a product; (ii) an undersampling pixel-phase component of
  ~2-5 mmag RMS on aligned frames; (iii) a minority of bright stars
  show per-star offset wander of ~10-15 mmag; (iv) intra-pixel
  sensitivity and brighter-fatter are unmeasured; (v) the ePSF
  builder at oversampling >= 3 is unusable (EPSF-BUILD-OSAMP-01).
  Evidence chain: VYVAR_AUDIT_FINAL; EPSF-XVAL-A2-COMPARE;
  EPSF-SHAPE-01; EPSF-CORE-01..04; EPSF-VAL-01..03; EPSF-AC-02;
  VYVAR_LITERATURE_CHECK_EPSF_20260915; ERA-520 gates.

Supersedes D-EPSF-XVAL-CLOSE-TEXT-01.

## D-EPSF-XVAL-DOD-05 (Milan 2026-09-16) - amends DOD-04 criterion 2; closure sequence amended by DOD-06

Basis: VAL-03 R-X1 FAIL on robust scatter (58.6 mmag; n=18 isolated
stars, G 8.7-11.2, BP-RP 0.6-1.4; |b|=4.93 within limit). Architect
re-analysis of `accuracy_cog_psf.csv`: simultaneous fit in G and colour
gives c = +130 mmag/mag, b = +2.3, robust scatter 42 mmag. The
remaining star-to-star scatter is accounted for by mechanisms
already measured: per-star constant sub-pixel phase on aligned frames
(CORE-03 T1 surface, 72 mmag ptp -> up to +/-36 mmag per star), the
bright-end model mismatch (these stars are the fit_ok-poor domain),
and the colour term. Literature: per-star PSF-vs-aperture systematics
of 0.05-0.15 mag are reported for imperfect PSFs (Dolphin 2000;
Stetson 1992). Architect error 30: criterion 2 was evaluated on
uncorrected PSF magnitudes in the PSF path's weakest domain with a
threshold (15 mmag) set without predicting the phase-surface
contribution.

Design fact on the record (AC-DESIGN-01): VYVAR has no per-star
aperture correction. `_compute_aperture_correction`
(`psf_photometry.py:366`) is a field-wide median ref/PSF factor, legacy
fallback only; the sole ePSF AC policy is `p4_none` (chi2_lt5 branch
deleted, CONSOLIDATE-01D, ZP-OK v2 / P4). The ePSF path is a
differential-only design by decision: per-star scale offsets are
absorbed by ensemble normalization, not by AC. Enabling AC is
therefore not a configuration change; a DAOGROW-style per-star AC
would be a new feature and is NOT pursued for closure.

EPSF-XVAL-01 remains OPEN. Criteria 1 and 3 unchanged. Criterion 2
of DOD-04 is SPLIT:

- 2a LINEARITY (flux scale vs brightness): on the VAL-03 isolated
  set, Theil-Sen slope of d = m_psf_inst - m_L_inst vs (G - 10),
  fitted simultaneously with colour: |b| <= 5.0 mmag/mag. Bootstrap
  std reported; PASS requires |b| <= 5.0 and |b| - 2*std <= 5.0 is
  NOT required (n is small; report both).
- 2b STABILITY of per-star offsets (what differential photometry
  relies on): split-half test - d_s derived on odd epochs, applied
  to even epochs (and vice versa); the robust scatter across stars
  of the residual (d_s,even - d_s,odd) <= 10 mmag. If 2b holds, the
  per-star scale offsets are constants absorbed by ensemble
  normalization and cannot enter a differential LC.
- RECORD (not judged): the per-star offset scatter itself (VAL-03:
  42 mmag after colour, 59 raw) as the PSF path's common-scale
  systematic, with its decomposition into phase and colour
  (EPSF-AC-02 Part C), and the colour term c as CHROMATIC-PSF-01.

Status on 516: criterion 1 PASS (VAL-02); criteria 2a/2b PASS
(EPSF-AC-02 R-AC1/R-AC2); criterion 3 pending ERA-520-RECUT-01.
**Amended by D-EPSF-XVAL-DOD-06 (2026-09-28):** closure sequence =
ERA-520-RECUT-01 + VAL-02/AC-02 harness re-check on 520 products ->
enact D-EPSF-XVAL-CLOSE-TEXT-02 (not CLOSE-TEXT-01). Phase-surface
attribution of the per-star residual RETRACTED (error 31;
PSF-COMMON-SCALE-RESIDUAL-01).

Evidence: `CURSOR_RESULT_EPSF_VAL_03.md` (R-X1 FAIL under DOD-04;
explained here), `CURSOR_RESULT_LEDGER_EPSF_DOD_05.md` (error 30),
`CURSOR_RESULT_EPSF_AC_02.md`, `CURSOR_RESULT_LEDGER_EPSF_DOD_06.md`
(error 31), `CURSOR_RESULT_EPSF_CORE_03.md` (T1 phase surface;
per-epoch only).

## AC-DESIGN-01 (Milan 2026-09-16) - record; not a defect

VYVAR has no per-star aperture correction on the ePSF path. Field-wide
`_compute_aperture_correction` (`psf_photometry.py:366`) is legacy
fallback only; sole policy `psf_ac_policy = p4_none`. Differential-
only design: per-star scale offsets absorbed by ensemble
normalization. A DAOGROW-style per-star AC is a new feature and is
NOT pursued for EPSF-XVAL-01 closure. See D-EPSF-XVAL-DOD-05.

## CHROMATIC-PSF-01 (Milan 2026-09-16) - record

PSF minus large-aperture flux depends on colour at c = +137.5 +/- 48.3
mmag per mag of BP-RP on the same photons (EPSF-AC-02 bootstrap on the
VAL-03 isolated set; architect pre-estimate was +130; consistent with
VAL-02: PSF c = +140 vs aperture c = -21 against Johnson V). Red stars
measure fainter by PSF than by aperture: expected for an unfiltered
wide-band system (seeing FWHM ~ lambda^-0.2; optical chromatism) with
a single ePSF built at the ensemble mean colour. Constant per star;
mitigated in differential photometry by colour-matched comparison
selection (`comp_color_tiers`). Literature anchor to add at citation
time: chromatic PSF / DCR treatments (Meyers & Burchat 2015 and refs
therein). Feeds the methods paper and D10-1.

Evidence: `CURSOR_RESULT_EPSF_AC_02.md`, `CURSOR_RESULT_EPSF_VAL_03.md`,
`CURSOR_RESULT_EPSF_VAL_02.md`, `CURSOR_RESULT_LEDGER_EPSF_DOD_05.md`.

## D-EPSF-XVAL-CLOSE-TEXT-01 (Milan 2026-09-16) - SUPERSEDED by CLOSE-TEXT-02 (DOD-06)

Historical close text (amended by DOD-05). **Not to be enacted.**
Live close text is D-EPSF-XVAL-CLOSE-TEXT-02.

  VYVAR ePSF photometry is implemented correctly (model construction
  verified against an independent PSFEx reference; fit machinery
  self-consistent to ~1 mmag), externally cross-validated (closer to
  the AIJ-validated aperture path than PSFEx), and validated for its
  intended domain (precision equal to aperture for G >= 9.5; flux
  scale linear vs curve-of-growth-tied aperture; per-star scale
  offsets stable and absorbed by ensemble normalization; bright-end
  fits gated by psf_fit_ok with aperture fallback). Known limitations:
  undersampling pixel-phase component, unmeasured intra-pixel
  sensitivity, builder at oversampling >= 3; common-scale (absolute)
  PSF magnitudes carry a ~40-60 mmag per-star systematic (phase,
  chromatic PSF) and are not a product.

Superseded 2026-09-28 by D-EPSF-XVAL-CLOSE-TEXT-02 (error 31
retracts phase attribution of the per-star residual; adds minority
wander and unattributed ~37 mmag limitation).

## D-EPSF-XVAL-DOD-04 (Milan 2026-09-16) - amends DOD-03 criterion 2; criterion 2 amended by D-EPSF-XVAL-DOD-05

Basis: VAL-02 R-W3 (|b|=6.4, resid RMS 519 mmag) is INCONCLUSIVE, not
a PSF-path finding. Architect re-analysis of
`accuracy_vs_catalog_psf.csv` / `_ap.csv`: 3-sigma-MAD clipping leaves a
95-star core with robust scatter 56 mmag, b = 11.8 +/- 5.2 mmag/mag
(bootstrap), c = +140 mmag/mag; the 9 clipped stars (-180 to -3350
mmag) are outliers on BOTH paths (blends / catalogue association),
not PSF failures. The 56 mmag core scatter is the unfiltered-CMOS ->
Johnson V transformation with a single linear colour term (D10-1),
not photometric error. Criterion 2 as written carried the colour
transformation and unscreened blends in the reference - architect
error 29.

EPSF-XVAL-01 remains OPEN. Criteria 1 and 3 of DOD-03 are unchanged.
Criterion 2 is REPLACED by the literature-standard flux-scale test
(Stetson 1990 DAOGROW; Dolphin 2000):

2. ACCURACY (curve-of-growth-tied large aperture). **AMENDED by
   D-EPSF-XVAL-DOD-05 (2026-09-16):** split into 2a LINEARITY and 2b
   STABILITY; the DOD-04 single-bar form (|b|<=5 and robust scatter
   <=15 mmag) is historical. VAL-03 R-X1 FAIL under that bar is
   explained (error 30); superseded by DOD-05 2a/2b via EPSF-AC-02.
   Historical DOD-04 text retained below for the record.

Historical DOD-04 criterion 2 text:

2. ACCURACY (curve-of-growth-tied large aperture). On isolated,
   unsaturated, constant bright stars, d(s) =
   median_epochs(m_psf_inst - m_L_inst), where m_L is a
   large-aperture (r_L ~ 4 x FWHM) instrumental magnitude tied by a
   measured growth curve. Fit d = a + b*(G - 10): |b| <= 5.0
   mmag/mag and robust scatter (1.4826 * MAD) <= 15 mmag. The colour
   slope c of d vs BP-RP is RECORDED (expected ~0; a nonzero value
   would indicate PSF colour dependence), not judged. No catalogue
   transformation and no blended star enters the reference.

Thresholds were Milan's to adjust before EPSF-VAL-03; frozen at
VAL-03 for the historical bar. Status: criterion 1 PASS (VAL-02);
historical criterion 2 measured by VAL-03 (R-X1 FAIL; explained by
DOD-05); live criterion 2 is DOD-05 2a/2b via EPSF-AC-02; criterion 3
pending the 520 era re-cut.

Evidence: `CURSOR_RESULT_EPSF_VAL_02.md` (R-W1/R-W2 PASS; R-W3
INCONCLUSIVE under this amendment),
`CURSOR_RESULT_LEDGER_EPSF_DOD_04.md` (error 29),
`docs/VYVAR_LITERATURE_CHECK_EPSF_20260915.md` C4/C6.

## D-EPSF-XVAL-DOD-03 (Milan 2026-09-15) - supersedes D-EPSF-XVAL-DOD-02; criterion 2 amended by D-EPSF-XVAL-DOD-04

Basis: VAL-01 (R-V1/R-V2 FAIL) read against the literature check
(C1-C7; `docs/VYVAR_LITERATURE_CHECK_EPSF_20260915.md`). Criterion 2
of DOD-02 measured the aperture path's missing curve-of-growth
correction (audit D5-1; Howell 1989; Stetson 1990), not PSF
accuracy - architect error 27. Criterion 1 failed only on the bright
end, which is the documented behaviour of the method class
(Sokolovsky et al. 2017: aperture ~2x more accurate than PSFEx on
the brightest stars; Nardiello 2015; Libralato 2016; Irwin 1997).

EPSF-XVAL-01 remains OPEN. The PSF path is VALIDATED when, on an
identical epoch set and the pinned ensemble, all three hold:

1. PRECISION (per G bin). On constant stars selected as in VAL-01,
   r(s) = RMS_med(PSF diff LC) / RMS_med(aperture diff LC), V-prod
   admission:
   - Domain D = stars with G >= 9.5 (the PSF path's justified
     domain): median r <= 1.25 and max r <= 1.50.
   - Bright end G < 9.5: no ratio criterion. Instead an ADMISSION
     test: epochs with psf_fit_ok == False must be routed to the
     aperture method by the science-method picker
     (photometry_lightcurve.py:2393 region), verified in code and
     end-to-end on >= 2 bright stars in the 516 products. Any
     psf_fit_ok == False epoch that reaches a science LC as PSF is a
     FAIL.
2. ACCURACY (common-scale reference). **AMENDED by D-EPSF-XVAL-DOD-04
   (2026-09-16).** Historical DOD-03 text (catalogue m_cat; |b|<=5,
   resid RMS<=25; c recorded) is superseded for the live bar; see
   DOD-04 for the CoG-tied large-aperture replacement. VAL-02 R-W3
   under this historical wording is INCONCLUSIVE (method; error 29).
3. CODE: as DOD-02 criterion 3 (FIT-OK-ADMISSION-01, GAIN-FALSY-01,
   FIXPOS-NOOP-01 fixed and gated at the 520 era re-cut).

Thresholds for criterion 1 were frozen at EPSF-VAL-02 (R-W1/R-W2
PASS). Closure of EPSF-XVAL-01 = criteria 1 (VAL-02) + criterion 2
(DOD-05 2a/2b via EPSF-AC-02; historical DOD-04/VAL-03 R-X1 FAIL
explained by error 30) + criterion 3 (520 re-cut); then enact
D-EPSF-XVAL-CLOSE-TEXT-01 (amended by DOD-05). VAL-01 under DOD-02
stays historical.

Scope statement for the paper: on this rig (FWHM ~2.4 px,
9.77"/px, undersampled), the aperture path is the science product
for bright targets; the PSF path is a faint/crowded-field method
with a literature-expected bright-end precision deficit handled by
psf_fit_ok admission and picker fallback.

Until closure, the PSF path stays internal-diagnostic
(NOT FOR AAVSO/VARASTRO SUBMISSION sidecar wording unchanged).

### Fix list bound to the 520 era re-cut (unchanged from DOD-02)

- FIT-OK-ADMISSION-01 (`psf_internal_lc.py:124-134` mask).
- GAIN-FALSY-01 (`psf_photometry.py:2268-2270` `value or 1.0`;
  authority g_pt 0.637067 / RN 15.2 per
  `gain_photon_transfer.json` and pipeline_meta
  `resolved_facts`).
- FIXPOS-NOOP-01 (`psf_photometry.py:2431-2440` on
  IterativePSFPhotometry; EXC-0454).
- Already bound: RED-TARGET-T4 reland + micro-measurement;
  LC-CSV `skip_reason` unification.

Evidence: `docs/VYVAR_LITERATURE_CHECK_EPSF_20260915.md`,
`CURSOR_RESULT_EPSF_VAL_01.md` (R-V1/R-V2/R-V3 under DOD-02),
`CURSOR_RESULT_LEDGER_EPSF_DOD_03.md` (errors 27-28),
`CURSOR_RESULT_EPSF_CORE_04.md`, `CURSOR_RESULT_EPSF_CORE_03.md`.

## D-EPSF-PHASE-ROUTES-01 (Milan 2026-09-15)

Two literature-backed routes for the phase component (CORE-03 T1),
neither required for the aperture science product:
- Route A: correct osamp >= 3 ePSF build (RASTI 2025: ~0.1% pixel-
  phase error at osamp 3 for Gaussian-like ePSF) - BLOCKED on
  EPSF-BUILD-OSAMP-01.
- Route B: dithered Anderson & King (2000) build - TODO-A sub-item.

Literature: `docs/VYVAR_LITERATURE_CHECK_EPSF_20260915.md` C2-C3.
Evidence: `CURSOR_RESULT_EPSF_CORE_03.md` (T1),
`CURSOR_RESULT_EPSF_CORE_04.md` (R-P2 re-read),
`CURSOR_RESULT_LEDGER_EPSF_DOD_03.md`.

## D-EPSF-XVAL-DOD-02 (Milan 2026-09-15) - SUPERSEDED by D-EPSF-XVAL-DOD-03

Historical. Closed-on SUCCESS was: PRECISION ratio vs aperture on
>=4 check-class stars (median <= 1.25, max <= 1.50); ACCURACY
PSF-minus-aperture slopes vs G and BP-RP; CODE =
FIT-OK-ADMISSION-01 / GAIN-FALSY-01 / FIXPOS-NOOP-01 at the 520
re-cut. Superseded: criterion 2 used raw per-star apertures as
reference (audit D5-1 / error 27); criterion 1 had no G-domain
split for the literature-expected bright-end deficit. VAL-01
R-V1/R-V2/R-V3 FAIL under this bar is historical.

Original DOD-02 text (kept for the record):

Rationale: the DOD-01 bar (<= 3.0 mmag vs PSFEx) was transferred
from an aperture-vs-aperture precedent (1.9503 mmag vs AIJ: same
pixels, same weights, photon noise cancels). PSF vs any other
estimator does not cancel photon noise; on this set the aperture
check-star LC alone has RMS_med 8.23 mmag (CORE-04 A4), so 3 mmag
between two different estimators was unreachable even for a
perfect fit. CORE-04 triangle (RMS_med, target / check, 134
identical epochs, pinned ensemble): VYVAR PSF vs aperture
8.50 / 14.07; PSFEx vs aperture 12.44 / 15.69; VYVAR vs PSFEx
10.48 / 21.41. K3 (uniform weights) moves VYVAR toward PSFEx
(check 21.3 -> 9.9): most of R-A2-3 is an estimator-weighting
difference, not a defect.

EPSF-XVAL-01 remains OPEN. The PSF path is VALIDATED when, on an
identical epoch set and the pinned ensemble, all three hold:

1. PRECISION: on a set of >= 4 bright constant stars (check
   class; VSX-clean; not ensemble members),
   RMS_med(PSF diff LC) / RMS_med(aperture diff LC) has median
   <= 1.25 and no star > 1.50.
2. ACCURACY: per-star median(PSF_inst - aperture_inst) over the
   epochs, across stars with n_ok >= 100 and 8.5 <= G <= 12.5,
   fits d = a + b*(G - 10) with |b| <= 5.0 mmag/mag and residual
   RMS <= 10 mmag; and vs BP-RP with |c| <= 10 mmag/mag.
3. CODE: the three recorded defects are fixed and gated
   (FIT-OK-ADMISSION-01 = CORE-01 R-C0 fit_ok_for_zp admits
   finite flux+chi2 regardless of psf_fit_ok; GAIN-FALSY-01;
   FIXPOS-NOOP-01), and the era re-cut passes its gates.

Thresholds are Milan's numbers to adjust before EPSF-VAL-01 is
read; once EPSF-VAL-01 runs they are frozen. Closure of
EPSF-XVAL-01 = this DoD via EPSF-VAL-01 (measurement, issued
separately) + the fix list at the 520 era re-cut. A2 result
R-A2-3 stays on the record as historical (superseded bar).

Until closure, the PSF path stays internal-diagnostic
(NOT FOR AAVSO/VARASTRO SUBMISSION sidecar wording unchanged).
The aperture path is unaffected and remains the science product
for bright targets on this rig (FWHM ~2.4 px, 9.77"/px; Howell
1989/2006: aperture near-optimal at high SNR). The PSF path's
justified use is crowded / faint fields; the phase component
(CORE-03 T1: 72 mmag ptp surface, ~2-5 mmag RMS in the live
window; undersampling) is deferred to a dithered ePSF build
(Anderson & King 2000) under TODO-A, only if that use is needed.

### Fix list bound to the 520 era re-cut (no other changes ride it)

- FIT-OK-ADMISSION-01 (`psf_internal_lc.py:124-134` mask).
- GAIN-FALSY-01 (`psf_photometry.py:2268-2270` `value or 1.0`;
  authority g_pt 0.637067 / RN 15.2 per
  `gain_photon_transfer.json` and pipeline_meta
  `resolved_facts`).
- FIXPOS-NOOP-01 (`psf_photometry.py:2431-2440` on
  IterativePSFPhotometry; EXC-0454).
- Already bound: RED-TARGET-T4 reland + micro-measurement;
  LC-CSV `skip_reason` unification.

Evidence: `CURSOR_RESULT_EPSF_CORE_04.md` (R-R2, R-P2, R-K0),
`CURSOR_RESULT_EPSF_CORE_03.md` (R-Q3, R-Q4; error 25),
`CURSOR_RESULT_LEDGER_EPSF_DOD_02.md` (errors 25-26),
`CURSOR_RESULT_EPSF_XVAL_A2_COMPARE.md` (R-A2-3, historical).

## D-EPSF-XVAL-DOD-01 (Milan 2026-09-14) - SUPERSEDED by D-EPSF-XVAL-DOD-02

Historical. Closed-on SUCCESS was: after a root-cause fix, an
UNCHANGED A2-COMPARE re-run (same pinned ensemble
1497771992240531712 / 1499200223486564608 /
1497974027502858240 / 1497368849430107904, same target
1498613634033133184, same check 1497613731286514432, same
construction and thresholds) fires R-A2-1: target AND check
RMS <= 3.0 mmag. Superseded: that bar was unreachable by
construction on this set (architect error 26; see DOD-02).

Evidence at the time: `CURSOR_RESULT_EPSF_XVAL_A2_COMPARE.md`
(R-A2-3), `CURSOR_RESULT_EPSF_SHAPE_01.md` (R-SH3),
`CURSOR_RESULT_LEDGER_EPSF_XVAL_DOD_01.md`,
`CURSOR_RESULT_EPSF_CORE_01.md` through
`CURSOR_RESULT_EPSF_CORE_04.md`.

## D-EPSF-SWAP-DIFF-01 (Milan 2026-09-14)

The CORE-03 model swap is a DIFFERENTIAL measurement inside the
harness: B1(harness, live ePSF) vs B2(harness, PSFEx model) vs
PSFEx catalog, all on the same call path. The
harness-vs-production floor is the resolution limit (CORE-02 /
CORE-03 B1: 1.396 mmag on 804 rows). The CORE-01 gate of 1.0
mmag is NOT relaxed; Part B still tries to pass it. CORE-03
Part B did not; Part C ran under this decision with the
achieved B1 as the floor.

Evidence: `CURSOR_RESULT_EPSF_CORE_03.md`.

## D-RED-TARGET-T4-01 (Milan 2026-09-08)

(i-a) automatic tier4 fallback. In
`_select_comps_by_rms_then_color`'s color ladder
(`photometry_comp.py` ladder build / rung loop), when the last
rung (`comp_max_delta_bprp` cap) yields n < n_comp_min AND the
target has finite BP-RP: append one final rung T4_FALLBACK.
Candidates are the SAME post-MAD / post-RMS-ceiling /
post-isolation set the other rungs see. That quality chain IS
the SNR floor: RMS ceiling 0.080 mag ~ SNR 13.6 per epoch.
Ranked ascending by |delta BP-RP|, admitted up to the same
n_comp target count. If still n < n_comp_min, existing
no_comps path is unchanged. NaN-BP-RP targets keep the
`_delta_bprp_abs=0.0` bypass; do not "fix" it here. Pinned
targets: pin overlay early-return stays first; fallback never
sees them. When the fallback fires: sel_note
`color_rms_t4_fallback`; `comp_delta_bprp_map` already carries
per-comp deltas; surface `max_delta_bprp_used` and
`color_fallback=true` on the per-target selection output /
Phase 2A QA sidecar only when the rung fires.

Measured basis: M1 (R CVn cap empties; 0/3300 within the live
ladder last rung 0.79; min achievable |dBP-RP| 2.219; pinned
comps outperform nearest-color low-SNR stars). Live config
tiers are 0.15/0.30/0.55/1.10; T4 1.10 is not on the ladder
(architect error 19). Applies to future runs after an anchor
plan. **Not in production HEAD:** A3 claimed no unpinned 516
target has an empty quality-filtered cap and G2 would be
byte-identical. G2 on `817f1f9` refuted that --
`color_rms_t4_fallback` fired for CV CVn
(`1497007144465726080`) and HAT-188-0002048
(`1497683996951418880`); core SHA `f6cb0416` vs snap
`d55fcc9d` n=53; ext `8282f3ce` vs `cc8b532e` n=157; science
compare failed on HAT-188-0002048. A reverted (`8af9a69`).
Reland is **RED-TARGET-T4-RELAND** (needs era recut / Milan
GO). Evidence: `CURSOR_RESULT_R_CVN_EMPTY_COMP_M1.md`,
`CURSOR_RESULT_RED_TARGET_T4_01.md`.
Reland plan (Milan 2026-09-08): tied to the 520 era re-cut;
scoped G2 exceptions rejected; reland carries the RMS-first vs
color-first LC-quality micro-measurement for the two affected
targets.
Reland COMPLETE (ERA-520-RECUT-01, d79945f): colour-first HELPS
med |dBP-RP| (HAT 1.49->1.06; CV 2.03->1.61). CV CVn has no
science LC. Evidence: CURSOR_RESULT_ERA_520_RECUT_01.md.

## D-LC-SKIP-MANIFEST-01 (Milan 2026-09-08)

(iii) artifact honesty via skip-manifest sidecar now. One file
per setup: `photometry/lightcurves_skip_manifest.json`, written
at Phase 2A finalize (`photometry_phase2a._phase2a_finalize_exports`).
Lists EVERY target whose LC is absent or empty: catalog_id,
name, class (`no_comps_stub` | `pin_rms_abort` | `export_empty`
| `other`), `ac_skip_reason` where present, n_rows/n_exportable
when a CSV exists, timestamp, producing stage. The name is
outside both `photometry_sha_files` glob sets
(`dev/tests/photometry_sha.py:100-105`):
`**/photometry/**/lightcurve_*.csv`,
`**/photometry/**/comp_quality_*.json`,
`**/platesolve/**/comparison_stars_per_target.csv`,
`**/photometry/**/lightcurves/comp_qa_*.json`. LC CSV bytes are
not touched. Always written; `targets: []` when nothing is
skipped. Deferred end state: a `skip_reason` column lands in
the LC CSVs at the next natural era re-cut, not now.
Evidence: `CURSOR_RESULT_RED_TARGET_T4_01.md`.

## D-NSTARS-DIAG-01 (Milan 2026-09-07)

n_stars frame QC is DIAGNOSTIC-ONLY. Per night, on qc_metrics
status=ok rows, frames outside median +/- 5.0*sigma_MAD of
n_stars_detected are logged as warnings and KEPT. Measured basis
(FRAME-QC-PARITY-02B, live 516): the n_stars outlier (Light_029,
263 vs median 98) is photometrically clean (check |z|=1.865), while
the only photometric outlier epoch (Light_034, z=4.10) has normal
n_stars - the metrics decorrelate, so drop authority on n_stars
would remove a healthy frame and keep the unhealthy one. No
comparable package wires such a drop by default (AIJ none;
C-Munipack manual ~50% operator heuristic; VaST matching-failure
mechanism). k=5.0 chosen against the measured tight distribution
(ok-set MAD=1) to warn rarely and meaningfully. Promotion to drop
authority is a new decision with its own measured delta and anchor
plan.

## D-FRAME-QC-AUTH-01 (Milan 2026-09-07)

Frame drop authority is Layer B only: the DB DAO-FWHM prefilter ->
qc_metrics.csv status -> allowlist chain. Layer A calibration QC
(_post_calibration_qc_eval: HFR, star count, background RMS) is
diagnostic: it stamps VYQCPASS/VY_QCHFR/VY_QCNS headers and logs,
and must not drop frames. Log wording states this (option 2 of
FRAME-QC-PARITY-01; "REJECTED" wording retired on annotate-only
paths). Any future promotion of a Layer A metric to drop authority
is a new decision with its own measured delta and anchor plan.

## D-MP-CALIB-OFF-01 (Milan 2026-09-07)

Parallel calibration (VYVAR_CALIBRATE_MP=1) is permanently OFF and
unsupported in production. The E5 latent spawn defect
(_init_calibrate_batch_worker 1-tuple vs 3-tuple initargs) is fixed
and guarded by test_calibrate_batch_mp_spawn_passthrough_roundtrip;
the guard test stays. Any future enablement requires BOTH a new Milan
decision AND a serial-vs-MP byte-parity gate on a small real draft
(hash equality of all calibrated products). The MP-CALIB-PARITY-01
proposal is closed by this decision without a parity campaign.

## D-FWHM-AUTH-01-CLOSE (Milan 2026-09-07)

FWHM-AUTH-01 ledger line (PENDING since 2026-08-27) is closed.
Authority for aperture geometry FWHM is the per-frame QC moment FWHM
resolved through the file chain only:
qc_metrics.csv -> frame header card VY_FWHM -> night median
(aperture_policy.resolve_frame_fwhm_px). The sqlite DB is not a
source of FWHM authority and will not become one (files-only
direction; DB-RETIRE-01). Measured max delta card vs qc_metrics
2.5e-4 px = FITS card rounding, physically negligible
(sub-microamag at f=1.35). Not VY_FWHM_GAUSS, not DAO_TO_GAUSSIAN,
not the SNR-table draft constant (unchanged from APERTURE-01).
A-1-OVERRIDE (remove VY_FWHM_GAUSS override) remains a separate
open ROADMAP item with its own measured-delta requirement.
Carve-out: the safe-bbox precompute (pipeline_astrometry.py,
Variant A2) keeps MASTERSTAR header VY_FWHM as its FWHM source per
CONSOLIDATE-01B A2 MEASURE FIRST (era04 byte-identity; 0.015 px
r_out bookkeeping at the chip edge, no photometric effect). This
decision does not mandate swapping it to the qc chain. Evidence:
dev/results/context/session_20260831_c01b/REPORT.md (A2 STOP);
code comment src_py/pipeline_astrometry.py:1957-1959.

## D-FACADE-PERMANENT-01 (Milan 2026-09-03)

`pipeline.py` and `photometry_core.py` remain PERMANENT thin re-export
facades. `dev/` scripts keep importing through them. Facade removal is
off the table. Call-time facade imports in the ePSF-graph modules
(`epsf_stage`, `psf_photometry`, `psf_internal_lc`, `epsf_psf_merge`,
`epsf_zp_ok`, `psf_runner`) are acceptable permanent state.

## D-CONSTANTS-LEAF-01 (Milan 2026-09-03)

ALL pipeline-physical constants live in leaf module
`pipeline_constants.py` (imports nothing from VYVAR; anything may
import it at module level). The SAT_LIMIT numeric twin in
`pipeline_catalog.py` is dismantled. The facade re-exports the leaf
names so external readers are unchanged.

## D-RUNFULL-HOME-01 (Milan 2026-09-03)

`run_full_photometry_pipeline` stays physically in
`photometry_core.py` (production entry; C-D made permanent).
`analyze_calibrated_qc` + `_analyze_calibrated_qc_one` stay physically
in `pipeline.py` beside `AstroPipeline` (C-C).

## CONSOLIDATE-01D P2 - FLOW doc sync (2026-09-01)

FLOW builder + `flow_doc_facts.py` + `docs/VYVAR_FLOW_CZ.pdf` drop removed config keys (`err_background_mode` was the G1 `test_flow_doc_config_facts` miss).

## CONSOLIDATE-01D P2 - dao_gaia flatten (2026-09-01)

`dao_gaia_stage_01.py` / `_iter2.py` / `_iter3.py` deleted; helpers live in `dao_gaia_common.py`. `dao_gaia_stage_01_iter4.py` is not renamed; follow-up `dao_gaia_stage.py` only if Milan asks.

## CONSOLIDATE-01D P2 - phase01_use_bprp_primary removed (2026-09-01)

`phase01_use_bprp_primary` removed; BP-RP is the only color display basis (selection already used BP-RP).

## CONSOLIDATE-01D P2 - ALIAS 3 fields dropped (2026-09-01)

`comp_iterative_clip_enabled`, `blind_index_path`, `aavso_observer_code` removed as AppConfig fields; JSON loader still maps them onto `comp_sparse_fallback_enabled`, `blind_index_fine_path`, `observer_code`.

## CONSOLIDATE-01D P2 - DEAD persist-only keys removed (2026-09-01)

`qc_fwhm_limit`, `qc_elong_limit`, `psf_spatial_grid`, `psf_spatial_min_stars_per_cell`, `gs11_comp_suspect_dilution` removed; persist-only, never consumed.

## CONSOLIDATE-01D P2 - psf_ac_policy removed (2026-09-01)

`psf_ac_policy` removed; p4_none is the only ePSF AC policy (ZP-OK v2 / P4).

## CONSOLIDATE-01D P2 - masterstar_accept_mode removed (2026-09-01)

`masterstar_accept_mode` removed; MASTERSTAR odds gate is the only accept policy.

## CONSOLIDATE-01D P2 - err_background_mode removed (2026-09-01)

`err_background_mode` removed; empirical empty-aperture term is always on (F-BINGAIN-1); Howell math remains as missing-sigma fallback.

## CONSOLIDATE-01D P2 - export_err_mode removed (2026-09-01)

`export_err_mode` removed; exported err is always ERR-CALIB calibrated.

## CONSOLIDATE-01D P2 - global_comp_pool_enabled removed (2026-09-01)

`global_comp_pool_enabled` removed; production is always the COMP-POOL-01 global pool.

## VL-ANCHOR-EPSF01 accepted as first ePSF repeatability anchor (2026-08-28)

science validation pending; external independent gate for ePSF (AIJ-class) on ROADMAP.

## era04 hash v2 (content), v1 9367f998/d3cefff3 kept as history (2026-08-27)

Milan: redefine the --full anchor statistic as a content hash.
`compute_photometry_sha` drops `# git_hash=` / `# git_dirty=` /
`# files=` / `# generated=` / `# timestamp=` / `# vyvar_version=`
header lines. Frozen era04 bytes are not rewritten. v1 raw-byte
core 9367f998 n=160 / ext d3cefff3 n=210 stay in the ledger as
history (reproducible only at dffe859 because PSF LCs stamp HEAD).
v2 (gate): core af218acd n=160 / ext ada5caff n=210. Provenance
is a separate PASS/DRIFT line, never a SHA FAIL.
Evidence: `dev/results/context/session_20260827_anchor_hash/REPORT.md`.

## APERTURE-01d LOCK - annulus 2.7/5.2; era04 locked (2026-08-27)

Annulus set to 2.7/5.2 FWHM (AIJ 14/27 px at 516 night FWHM 5.191733;
Howell: just beyond aperture wings). Production r_ap stays f=1.35 x
FWHM = 7.0088 px; stamped r_in=14.0177 r_out=26.997. Independent AIJ
gate RMS(diff)=1.9503 mmag PASS (gate <=2.8; 01c was 2.7833 at
annulus 4.75/9). Recut era04 (candidate3 = 01c f=1.35 tree, kept).
Ledger v6: 60/60 tagged; 0 UNNAMED. POOL-STARVE (3) and EDGE-ANNULUS
(4) are limitations, not lock blockers. CSS_J135929 and
HAT-188-0003359 did not return (0 epochs); tagged [CROWDING] (VSX
not force-injected; detection assigned to a neighbour). BO level vs
era03 is +41.734 mmag (v5 was +49.458; move -7.72 mmag, outside the
<5 mmag prediction). FW +32.853 (v5 +28.213; move +4.64). C6-2
`--full` twice byte-identical core 9367f998 n=160 / ext d3cefff3
n=210. SNAPSHOT_NAME is era04. era03 freeze kept on disk.
Evidence: `CURSOR_RESULT_APERTURE_01D.md`.

## APERTURE-01c STOP - f=1.35 set; era04 not locked (2026-08-26)

Milan GO: production f=1.35, mode (a) f_fixed_night. Scatter
ladder is diagnostic only (r_min = 0.75 FWHM). Independent
unconfounded AIJ gate (VYVAR BO LC with comps pinned to
XVAL_AIJ_01_Table.tbl C2-C6, same r=7): RMS(diff)=2.7833 mmag
PASS (gate <=4). That number is not the APERTURE-01b pool-ZP
14.04 mmag. Recut era04 at r=7.0088 (candidate2 renamed, kept).
Ledger v5 vs era03: [APERTURE-01 f=1.35] with level residual
per target; GH also [CROWDING] (D11-1). 6 UNNAMED remain
(lost LC at r=7, Phase1 n_survivors<3, or dmag undefined).
Anything without a tag = STOP. era04 not locked. SNAPSHOT_NAME
stays era03 (9902d918 / 472bc9e4). config f stays 1.35.
Evidence: `CURSOR_RESULT_APERTURE_01C.md`.

## APERTURE-01b STOP - no f* on the accuracy grid (2026-08-26)

f is not taken from the scatter ladder. Pre-registered rule:
f* = smallest f in {0.75, 1.0, 1.25, 1.35, 1.5, 1.75, 2.0, 2.5}
with ALL of: dEE(f)=|EE_Q1-EE_Q4|<0.01; |level(f)-level(2.5)|<3 mmag
for BO, FW, and GH; AIJ RMS(diff)<=3.3 mmag on BO. Among survivors,
lowest joint demeaned RMS. Mode (a), 516, night QC FWHM 5.191733 px.
This table is the methods-paper figure for f. Rule not relaxed.

f     EE_Q1   EE_Q4   dEE     dBO    dFW    dGH    AIJ    RMS
0.75  0.828   0.750   0.078   -57.1  -8.8   -544   10.45  93.2
1.00  0.900   0.844   0.056   -35.0  -11.4  -475   12.56  95.7
1.25  0.943   0.902   0.040   -19.4  -8.6   -443   14.16  100.6
1.35  0.955   0.919   0.036   -16.2  -7.1   -422   14.04  103.5
1.50  0.968   0.940   0.028   -15.7  -5.1   -377   11.05  108.5
1.75  0.983   0.965   0.019   -8.9   +0.5   -342   12.76  119.9
2.00  0.992   0.980   0.012   -9.2   +0.6   -227   12.66  136.7
2.50  0.999   0.995   0.004    0.0    0.0    0.0   15.36  208.9

d* = mag_calib level vs f=2.5 (mmag). AIJ = RMS(diff) mmag.
Architect expected f* in 1.25-1.5; none of those rows pass.
f* = NONE. config f stays 0.385228. Evidence:
`CURSOR_RESULT_APERTURE_01B.md`.

## APERTURE-01 STOP - era04 not locked (2026-08-26)

Milan GO option (i) wired: one radius per frame for every star
= f x FWHM(frame), annulus scaled by the same FWHM. FWHM-AUTH-01:
authority is qc_metrics.fwhm_px (VY_FWHM). Modes (a) f_fixed_night
and (b) f_per_frame are both wired; config default (a). f comes
from the 516 scatter ladder once (joint demeaned RMS BO/FW/GH);
no per-magnitude table in production. SNR table is diagnostic
only. Continuous ee_r90 is diagnostic only.

Recut at ladder f=0.385228 (r=2.0 px) failed P-A1 vs era03 mixed
radii (BO +65.6, FW +24.0, GH -387.9 mmag) and P-A3 vs AIJ r=7
(18.18 mmag). Do not retune f from those numbers. Lock waits a
Milan pick: keep the RMS-optimal small r, or a larger f, or
option (ii) COG. Default mode (a) if (a)/(b) tie.

## EDGE-ANNULUS-01 CLOSED-DECIDED (Milan 2026-08-31)

Edge stars are not used (Milan 2026-08-31). Aperture and annulus
must lie fully on-chip; no partial-annulus mode. Consequence
accepted: FR CVn and the 4 EDGE ids stay outside the 516 product.

Predicate: `aperture_policy.star_fits_on_chip` (CONSOLIDATE-01B).
PHASE0-BORDER-MARGIN-GEOMETRY is a different margin (fixed 50 px
`phase01_chip_interior_margin_px` vs EDGE `r_out` ~27 px at 5.2 x
FWHM) and is not merged into this closure.

## C6-3d [APERTURE-RADIUS] STOP - era04 not locked (2026-08-26)

SNR-optimal per-star radii without COG produced the era03->era04
mag_calib 2.8/10.8/6.8 mmag (P3 restores era03 radii -> 0.0 mmag).
Governing mover: measured growth curve in aperture_snr_table.json
(ee_r90 6.0->5.5), not FWHM-AUTH-01, not ZP. D5-1 CONFIRMED:
Howell 1989 pairs optimum apertures with growth curves; Collins
2017 uses one r per frame for all stars. Fix is Milan GO (i)
common f x FWHM or (ii) COG on; lock after that recut. [EDGE]:
named variables with annulus off-chip are out_of_frame when
safe_bbox_px is set (era03 had null). Evidence:
`CURSOR_RESULT_ANCHOR_ERA04.md`.

## C6-3c STOP - era04 not locked (2026-08-26)

[WCS-APERTURE] refused: per-frame aperture centres are MASTERSTAR
snap + peak refine (`pipeline.py:7491`), not world2pix; proc x,y
era03 vs era04 are identical; D2 six cards did not move centres;
centroid truth residuals equal (p50 0.797 px). [CT-REF] in:
`ct_bp_rp_comp_med` is the ZP ensemble, never `comparison_stars.csv`
(`cfffa82`). After phot-only, BO `mag_calib_final` +2.787 mmag equals
`mag_calib`; 10/12 CT-dominated LCs collapse. [EDGE]: four era03-only
targets (`out_of_frame` / safe_bbox), including FR CVn. mag_calib
2.8/10.8/6.8 mmag remains UNNAMED (measured aperture_r 5.999->5.499
on BO). Anchor remains era03. Evidence:
`CURSOR_RESULT_ANCHOR_ERA04.md`.

## C6-3 STOP - era04 not locked (2026-08-26)

C6-2 `--full` twice OVERALL PASS on
`draft_000516_snapshot_era04_20260826` (core **961d590f** n=169, ext
**59206a24** n=222). C6-3 vs era03: 60/60 LC targets changed; 55
UNNAMED. Same-ensemble per-epoch `mag_calib` is not 0 and not the
~2.4 mmag AC overlay constant (BO CVn 4=4 comps, +2.8 mmag median,
std 1.6 mmag, `mag_calib_final` +59 mmag). Named only FRAME-29,
D3-D5, C3-K5, NAME-FIX. No new tag invented. Anchor remains era03:
core **9902d918** n=121, ext **472bc9e4** n=179. R1'' is
informational only (shim recorded; chain skipped on KeyError
`frame`). Evidence: `CURSOR_RESULT_ANCHOR_ERA04.md`.

## C6-0 R1' STOP - era04 not locked (2026-08-26)

Resume (A-only `git diff --name-status c592ecf 0684ba9`): six files
copied; import OK; chain FAIL `ValueError` unpack 4 vs 3 on
`lock_existing_and_leftover_assign`. c592ecf accounting is 3-tuple;
0684ba9 M `masterstar_gaia_accounting.py` is 4-tuple and was not
copied. C6-1 not started. Anchor remains era03: core **9902d918**
n=121, ext **472bc9e4** n=179. Evidence:
`CURSOR_RESULT_ANCHOR_ERA04.md`.

## SEL-GHOST-01 - one identity, one geometry, no match-rate widening (2026-08-25)

Part A (draft 520 `g_60_4`) binding facts:

F1  Sky-domain match widens (12" -> 18" logged; silent 0.95 loop to
    96" unlogged). 347/692 pairs at gate time; greedy closest-first.
F2  Identity gate ran once, correctly: ok=52 warn=9 fail=286 at
    FWHM 1.25 px. It clears catalog_id / catalog / match_sep_arcsec
    and leaves `name` (wcs_invertibility.py:337-344).
F3  _vyvar_df_to_csv (pipeline.py:5938-5940) calls
    catalog_id_series_for_masterstars_export (gaia_catalog_id.py:
    197-208) which copies a Gaia-looking `name` back onto catalog_id.
    The 286 stripped IDs are restored on export. Optimizer input sees
    347 nonempty IDs; Grip1 SIP fit on those pairs: rms_lin 81.0 px,
    rms_sip 84.7 px; INV-WCS-01 WARN p95 146.8 px. Final header WCS is
    unusable (10/11 G<12 stars fail the gate against it).
F4  All 8 ghost comps: vy_match_mode=locked (born-owned lock skips
    lock_tol_px). Optimizer identity (astrometry_optimizer.py:475-480)
    gates only NEW writes, with header VY_FWHM=2.5 (full-res) rather
    than the DAO-domain _fwhm_used=1.25.
F5  Honest match rate (d_px <= 3 FWHM) 0.088 at gate time vs 0.522
    reported on the final table. No literature precedent for
    match-rate-driven radius widening (Marrese et al. 2019, A&A 621
    A144; Bertin SCAMP/SExtractor).

Milan GO (chat, 2026-08-25) on D1-D4:

D1  Catalog-match radius must not chase a match rate. The 0.70 -> x1.5
    loop and the 0.95 widening loop are REMOVED. Radius =
    max(12", 3 x FWHM_dao_px x plate_scale), one pass; the
    tighten-to-4.5" step stays. Low match rate remains a WARN.
    `solve_rms_px` is stamped as a diagnostic only (it belongs to the
    identity-gate WCS, not to a radius the 3xFWHM gate strips anyway).
    T1 (B-STOP-3, 2026-08-25): 516/520 post-gate catalog_id sets
    identical to S3/S6; 516 used radius stayed 152.32" because
    solve_rms was already below FWHM. Basis: Part A F1/F5, Marrese
    et al. 2019 A&A 621 A144, B-STOP-2 D1 architect error.
D2  Astrometry optimizer refit acceptance: reject when rms_sip >
    max(3 x FWHM_dao_px, 3 px) or n_honest_pairs < 10. A refit that
    worsens round-trip p95 versus the entry WCS is never written.
    Before the first optimizer write, the entry WCS is backed up.
    Basis: Part A F3, B1b M4 (no backup existed; 84.7 px SIP written).
D3  Comparison candidacy before the rms ceiling: source_state in
    {DETECTED_P1, DETECTED_P2}; vy_identity_gate != fail;
    gaia_dao_resid_px <= max(3 x FWHM_dao_px, 2 x solve_rms_px);
    MASTERSTAR snr_ap_pixscaled >= 10 (pixel-scaled estimate
    sqrt(F/g + (sigma_pix * sqrt(pi r^2))^2), not empty-aperture
    empirical error; D5: threshold stays 10). phase01_comparison_max_comp_rms unchanged
    (COMP-RMS-DEF-01). Basis: REG-520-01, Part A, B-STOP-1.
D4  One geometry question, one threshold. Lock PREFERENCE keeps the
    derived lock tolerance (certificate); lock REJECTION (B1e) uses the
    identity-gate fail threshold 3 x FWHM_dao_px. Basis: B1b M2c (ten
    516 edge stars at 2.05-2.65 px rejected at 2.5 px while the
    identity gate passes them at 3 x FWHM).
D5  D3 aperture SNR threshold stays 10. Two live 516 comps
    (1500579870061241088 snr_ap 8.29; 1498964240802993408 snr_ap 7.57;
    photon ~0.13 mag) leave the ensembles. The gated column is
    `snr_ap_pixscaled` (pixel-scaled err_ap), not empty-aperture
    empirical error. Basis: B3 T2-P1 table; CLOSE-OUT C0b.

Architect retraction (verbatim): "H-MATCH-WIDEN
(2026-08-25) named WCS-refine rematch (H2) as the path that returned
stripped IDs to the table. Measured: refine was rejected 3x and never
rematched on 520 g_60_4. The return path is name-based rehydration on
CSV export (F3). H2 is retracted as cause; the code path remains real
and is closed by B1 as defence, not as the fix."

Part A 2.5 honest match rate (B-STOP-3 T5, 2026-08-25). Defined on
DETECTED rows only (catalog_membership rows are catalog, not matches):

  honest   = n(DETECTED with d <= 3 FWHM) / n(DETECTED with cid)
  reported = n(DETECTED with cid) / n(DAO retained)

B-STOP-2 P-520-4 restated under this definition on 520 g_60_4 (n_DAO
retained = 742): honest = 1.0 (61/61 DETECTED-with-cid), reported =
0.082 (61/742). The old Part A mix of membership rows into the
numerator is retired.

Pointers: INV-MATCH-IDENTITY-01, INV-SOURCE-STATE-01, INV-WCS-01 write
guard (S4b), D3 SNR note. Evidence: `CURSOR_RESULT_SEL_GHOST_01_B2.md`,
`CURSOR_RESULT_SEL_GHOST_01_B3.md`, CLOSE-OUT C0/C1/C2.

## ANCHOR-DRIFT-01 - R1 vs R0 is freeze lag, not a hidden MS commit (2026-08-25)

**Cause (measured, STOP C1).** Fresh full chain at `c592ecf` (R1) vs
frozen live 516 (R0, freeze git `ad19e14`): 0/60 LC SHA equal because
`--full` never rebuilds MASTERSTAR. C1c: no commit in
`ad19e14..c592ecf` moves the pre-expand detection CSV (SHA
`2eae423d...`, n=2643 at four bisect points). Frozen 3610 vs R1 3606
is census membership. Name DET_ vs cid, greedy ID swaps, and two
headline LC deltas (bogus frozen epoch; ZP on identical mag_inst) are
authorized-unrecut. Census 4+1 catalog_id membership remains
unexplained at ID grain. No defect-fixed-here. Re-cut is C6, Milan GO
in chat, full-chain (not photometry-only).

Evidence: `CURSOR_RESULT_ANCHOR_DRIFT_01.md`.

## COMP-RMS-DEF-01-A - selector comp_rms is not differential mag (2026-08-25)

**Cause (measured, STOP C2; wiring is C3).** Selector `comp_rms`
(`comp_pool_rms.compute_global_pool_rms_map` 74-387) is 1.4826*MAD of
mag-bin relative flux (fractional), gated as if mag (ceiling 0.1).
Leave-one-out MAD of differential mag (iv) puts the bright 520 pool
near a few times photon; selector 4.33 can sit in a 5% ensemble.
`suspected_variables` uses the same relative-flux series with RMS not
MAD (5/7 flagged). ZONE-SAT-01: peak test reads `peak_max_adu` but
skips when `saturate_limit_adu_85pct` is NaN, so G=7.63 peak 88781
stays `zone=linear`. Fix direction (GO): LOO differential scatter,
ceiling k*photon, zone reads pixel peak vs saturate limit. k not
computed on frozen 516 (no SNR column).

Evidence: `CURSOR_RESULT_COMP_RMS_DEF_01_A.md`.

## EPSF-ZP-OK-01-WIRE v2 - wide rig only (2026-08-26)

Wired for Carl-Zeiss 200 mm + QHY294MM (drafts 516/517). Rig identity
is `equipment_id:telescope_id` = `1:1`, not `scanning_id`. Production
`psf_zp_membership=fit_ok_for_zp` on that allow-list. Other rigs stay
`fit_ok_strict` until EPSF-ZP-OK-XRIG-01. T1 no longer rewrites live
516 PSF LCs (`output_directory`). Closed: locate-fail / "ZP-OK v2
undecided" / "T1 rewrites live 516".

W2 meters HIT: BO 134/134 demeaned 8.495 mmag; FW 134/134 demeaned
5.218 mmag. ePSF SHA unchanged.

Evidence: `CURSOR_RESULT_EPSF_ZP_OK_01_WIRE_v2.md`.

## 2026-08-25 push incident - do not rewrite origin/main (2026-08-26)

`git push origin HEAD` advanced `origin/main` from `b1f5b8c` to `7c086e8`
without authorization. Milan recommendation accepted: do **not**
force-push main back. Every commit on that SHA passed `--fast --clean`.
Rule: pushes name the ref (`git push origin <local>:<remote>`).
`git push origin HEAD` and bare `git push` are banned. Updating main
requires gitignored `dev/PUSH_AUTH_main_<YYYYMMDD>.txt` with the target
SHA; `dev/scripts/push_guard.py` refuses otherwise.

Evidence: CLOSE-OUT part 3; `CURSOR_RESULT_CLOSEOUT_C8.md`.

## COMP-RMS-DEF-01-B - LOO mag RMS, k=5, zone peak test (2026-08-25)

Milan GO (chat, 2026-08-25): wire C2f. C3-0 k is p90 of
(iv/photon) rounded up; 3<=p90<=5 => k=5. Measured p90=3.672 so
**k=5**. Gated statistic is `comp_rms_loo_mag` (MAD*1.4826 of
leave-one-out differential mag vs the candidate pool, all loadable
proc frames, no clipping). `comp_relflux_mad` is diagnostic.
Ceiling is `min(0.1 mag, k x photon)` with photon from
`snr_ap_pixscaled` (raise if missing). `phase01_comparison_max_comp_rms`
is an absolute cap only. ZONE-SAT-01: peak test uses `peak_max_adu`
(fallback peak_dao, fallback flux) and still runs against
`saturate_limit_adu` when 85pct is NaN. Wide-rig 516 24 saturated
rows stay; 520 G=7.63 becomes saturated.

Evidence: `CURSOR_RESULT_COMP_RMS_DEF_01_B.md`.

## EPSF-ZP-OK-01-WIRE v2 - locate fail (2026-08-25)

**SUPERSEDED 2026-08-26** by the wide-rig wire (same filename STOP,
locate table kept as appendix). Parked v2 was re-issued in chat
2026-08-25 and executed in CLOSE-OUT C4.

## REG-520-01 - non-cal is first-class; 0.39 vs 0.06 is selection (2026-08-24)

**Cause (measured, STOP).** Same SS Cam night, same `RUN VYVAR (non-cal)`
button: June lc_rms 0.0622 vs today 0.3949. S1 matching A/B/C does not
change bright DETECTED counts. Selected comps are DETECTED_P1 at 59 px
from Gaia; June-band stars fail `max_comp_rms=0.1`. CAL-520-01
H-CAL-MISCLASS is superseded as the causal story. Library "no AZ800
masters" remains a fact, not the regression.

**Decision (Milan testimony, binding until a later GO).** Non-cal is a
first-class route. PRECAL radiometric metric informs, does not block.
A declared-mode stamp/banner is the product direction (not executed in
REG-520-01). Selection-input (rms ceiling + residual/visibility gate)
is the evidence-ranked fix for this draft's lc_rms.

Evidence: `CURSOR_RESULT_REG_520_01.md`.

## DAO-GAIA-XFER-01 - pin STAGE-01 sandbox gate to hand params (2026-08-24)

**Cause (REGRESS-01, accepted).** Draft 520 Brno AZ800/C5A-150M failed
DAO-GAIA-CALIBRATION by ~0.6-2.0 pp because the STAGE-01 iter4 sandbox
(draft 516 WIDE CVn, FWHM 5.2 px, 9.77 arcsec/px) was re-scored with
**Brno-derived** centroid tols (1.0 px from FWHM 1.25 px). Hand CSV was
locked 2026-08-19 at seed/pass2 = 2.0 px on those 516 frames. Catalog
fingerprint and scoring code were unchanged (H-GATE-XFER, not H-KAT /
H-CODE / H-DATA).

**Decision (Milan GO, 2026-08-24, binding).** Option 1 + identity stamps.

1. The STAGE-01 validation gate **always** re-scores the 516 sandbox with
   `ValidationParams.hand_validated()` (match 3.0 / pass2 2.0 / seed 2.0 /
   p1=4.5 / p2=4.0). Draft-derived centroid tols are **never** substituted
   into that compare. `MAX_REGRESSION_PP=0.005` and fail-closed raise
   unchanged.
2. Derived tols remain in force for **production photometry of the current
   set** and are recorded on the certificate as separate fields
   (`derived_pass2_center_tol_px`, `derived_forced_seed_centroid_max_px`,
   `production_tolerances.scope=production_photometry_current_set`).
3. Certificate write is fail-loud without identity stamps: Gaia/VSX cheap
   fingerprints, sandbox draft id + SHA256 of MASTERSTAR and Light_001/076/148
   actually scored, hand CSV path+SHA256, lock-rig plate scale and FWHM
   (516). Missing stamp source raises `DAO-GAIA-IDENTITY`.
4. Informational WARN when |derived/hand| ratio >= 2x (does not block).

**Gate-design principle (general).** A reference-bearing gate must pin
**and stamp** the full context the reference stands on: params, rig
scale/FWHM, catalog identity, and input SHAs. Same lesson class as
CATALOG-PROVENANCE (catalog identity) and the census pattern note
(denominator identity) -- this is the parameter-space instance.

**Evidence:** `dev/results/CURSOR_RESULT_DAO_GAIA_REGRESS_01.md`,
`dev/results/CURSOR_RESULT_DAO_GAIA_XFER_01.md`. 520 z follow-up
(read-only, not a gate change): projecting the g_60_4 VERIFIED WCS onto
z_90_4 MASTERSTAR yields catalog_recovery_gate 2.7% (n_tight=3); z
p99-median contrast 9.06 ADU vs g 40.0 ADU. Sibling seed would not
rescue this set. Carry MULTIFILTER-WCS-01 for the general seed-not-trust
design; do not lower `masterstar_catalog_recovery_min`.

---

## EPSF-AC-01 - chi2<5 AC gate is a brightness cut (2026-08-24)

**Finding (measurement, binding until a new AC decision).** The F6 ePSF aperture
correction (`_compute_aperture_correction`, chi2<5) is a brightness cut in
disguise: on draft 516, 20-frame subset, 0/30 brightest science-set stars are
ever admitted; night-median admitted mag 13.35 vs science-set 12.90.
Night-median AC from that ensemble is 0.500 vs 0.567 from all fit_ok vs
0.78/0.64/0.59/0.47/0.46 per M1 mag bin. Uncorrected PSF/DAO is mag-sloped
(bins 1.27 -> 2.21; corr=0.69; RMS~=|median| in every bin = offset, not
scatter). A scalar AC cannot be honest on this model.

**Policy (recommendation; not wired -- STOP for Milan).**
(a) Production F6 merge: keep current chi2<5 behavior as an explicit named
fallback until GO. Do not treat it as a quality gate. Preferred GO options:
P4/AC-off (stamp uncorrected; absolute scale stays untrusted) or P2 mag-binned
if a nearer-to-1 PSF/DAO per bin is wanted. P3 (all fit_ok) is the least-wrong
scalar but still cannot flatten A1.
(b) Internal PSF LC: P4 (no AC). Per-frame scalar cancels in the ensemble ZP;
P0 vs P4 `psf_delta_mag` identical to 1e-12. Mag-slope residual for BO CVn vs
pinned-comp mean mag is ~76 mmag (A1 slope term), separate from the existing
~614 mmag RMS vs aperture driven by ensemble fit_ok drops.

**Evidence:** `dev/results/CURSOR_RESULT_EPSF_AC_01.md`.
H5 fitter-scale in `CURSOR_RESULT_EPSF_SHAPE_01_M.md` is superseded
(SHAPE-01-F F2). F2b pedestal mechanism retracted (outer-annulus median 0).

---

## EPSF-AC-02 - Milan GO: P4 production AC + LC-PIN-01 (2026-08-24)

**Decision (Milan GO, 2026-08-24, binding).** Four items, each with its own
rationale.

**(a) Production F6 merge AC policy = P4 (`psf_ac_policy=p4_none`).**
Stamp uncorrected fit flux (`psf_ac_factor=1`, `psf_ac_applied=false`). The
chi2<5 DAO/PSF median remains available as the named fallback
`chi2_lt5_legacy`. A scalar trained on a brightness-cut ensemble cannot be an
honest absolute scale on this ePSF (EPSF-AC-01 A1/A2). Relative photometry
does not need that scale: a per-frame scalar cancels in the ensemble ZP.

**(b) Internal PSF LC policy = P4.** Same reason as (a), plus the measured
P0/P4 `psf_delta_mag` invariance (AC-01 A3, 5e-15). Wiring a scalar AC into
the diagnostic LC would change provenance without changing differential mags.

**(c) LC-PIN-01: same-membership-or-NaN ensemble ZP (INV-PSF-LC-PIN-01).**
If any pinned comparison star lacks `psf_fit_ok` or finite `psf_flux` on an
epoch, that epoch is NaN `psf_delta_mag` with
`psf_epoch_drop_reason=comp_psf_fail:<gaia_id>`. Aperture columns stay filled.
No partial-ensemble fallback. AC-01 A3 614 mmag RMS was membership drift
(frame 004: 0.385 vs 0.866), the PSF-branch analogue of ERA-03 INV-PIN.

**(d) EPSF-CORE-01 ROADMAP row gets the literature parameters.** The remaining
open defect is the narrow ePSF core (FWHM 2.36 vs 3.30 px), not the AC scalar.
Build recipe: multi-frame samples, oversampling vs >=4 gridpoints/FWHM,
minimal smoothing at low osamp, 2D polynomial gridpoint estimation, watch
photutils for Godden & Blundell 2025 fixes. Newton remains desired but is no
longer the sole unlock.

**Canonical AC (literature).** Absolute PSF scale, when wanted, is to be
derived DAOGROW/DOLPHOT-style from bright isolated stars versus growth-curve
total magnitudes (Stetson 1990), never from a chi2-gated per-star DAO ratio
(defect A2 on record).

**Evidence:** `dev/results/CURSOR_RESULT_EPSF_AC_02_WIRE.md`.

---

**Finding (measurement, binding until a new AC decision).** The F6 ePSF aperture
correction (`_compute_aperture_correction`, chi2<5) is a brightness cut in
disguise: on draft 516, 20-frame subset, 0/30 brightest science-set stars are
ever admitted; night-median admitted mag 13.35 vs science-set 12.90.
Night-median AC from that ensemble is 0.500 vs 0.567 from all fit_ok vs
0.78/0.64/0.59/0.47/0.46 per M1 mag bin. Uncorrected PSF/DAO is mag-sloped
(bins 1.27 -> 2.21; corr=0.69; RMS~=|median| in every bin = offset, not
scatter). A scalar AC cannot be honest on this model.

**Policy (recommendation; not wired -- STOP for Milan).**
(a) Production F6 merge: keep current chi2<5 behavior as an explicit named
fallback until GO. Do not treat it as a quality gate. Preferred GO options:
P4/AC-off (stamp uncorrected; absolute scale stays untrusted) or P2 mag-binned
if a nearer-to-1 PSF/DAO per bin is wanted. P3 (all fit_ok) is the least-wrong
scalar but still cannot flatten A1.
(b) Internal PSF LC: P4 (no AC). Per-frame scalar cancels in the ensemble ZP;
P0 vs P4 `psf_delta_mag` identical to 1e-12. Mag-slope residual for BO CVn vs
pinned-comp mean mag is ~76 mmag (A1 slope term), separate from the existing
~614 mmag RMS vs aperture driven by ensemble fit_ok drops.

**Evidence:** `dev/results/CURSOR_RESULT_EPSF_AC_01.md`.
H5 fitter-scale in `CURSOR_RESULT_EPSF_SHAPE_01_M.md` is superseded
(SHAPE-01-F F2). F2b pedestal mechanism retracted (outer-annulus median 0).

---

## EPSF-LC-LOG-01 - internal PSF LC authority (2026-08-24)


**Decision (Milan, binding).** Phase 2A no longer writes submission-shaped
``lightcurve_*_psf.csv`` files. Those globs in ``photometry_core``,
``photometry_report``, and ``export_reports`` now mean the internal diagnostic
product written by ``psf_internal_lc.py`` (RUN ePSF job path and/or the ePSF
dashboard button / headless flag).

**Rationale.** A Phase 2A method-variant LC would collide with the diagnostic
schema (provenance header, ``psf_delta_mag``, NaN coverage rows) and with the
trust boundary: PSF relative photometry is production-usable; PSF absolute
scale on bright stars is untrusted pending EPSF-SHAPE-01. Science exports
remain aperture-only. One writer owns the filename.

**Evidence:** ``dev/results/CURSOR_RESULT_EPSF_LC_LOG_01.md``.

---

## INV-PSF-SUBMIT-01 - unconditional AAVSO/VarAstro guard (2026-08-24)

**Decision (Milan, binding).** AAVSO and VarAstro submission writers hard-fail
when the effective ``lc_method`` is ``psf`` or ``adaptive``. No config escape
hatch. Internal PSF light curves must not route through those writers.

**Rationale.** Review found ``export_reports.py`` could feed ``_export_method in
("psf", "adaptive")`` into the AAVSO citation context, i.e. the submission
path could in principle run with a PSF method. SESSION-CLOSE 2026-08-23 trust
boundary: science exports stay aperture-only. An escape hatch would re-open
the same risk site. The guard is ``InvariantViolation("INV-PSF-SUBMIT-01")``
before any export bytes are written; the batch exporter iterates aperture only.

**Evidence:** ``docs/VYVAR_INVARIANTS.md`` (INV-PSF-SUBMIT-01);
``dev/tests/test_psf_internal_lc.py`` T3.

---

## MS-SOURCES-RETIRE - files over DB for per-draft science (2026-08-21)

**Decision (Milan, binding).** No database repair and no re-population of corrupt
`MASTER_SOURCES`. Draft directory files (`masterstars_full_match.csv` + enrichment columns
written at MAKE MASTERSTAR) are the source of truth for ePSF input, comp-selection enrichment,
and Photometric Grid QA. Shared SQLite holds equipment/location/calibration/cache only.

**Architect review decisions (2026-08-21, binding for MS-SOURCES-RETIRE-02):**

1. ePSF uses existing CSV-quality selection path as-is (conscious widening vs DB safe-comp +
   targeting + augment); PSF-star refinement belongs to **ePSF-VALID-01**.
2. MAKE MASTERSTAR persists enrichment fields to CSV; EXC-0347 silent pass replaced with
   fail-loud log + plan provenance when columns absent on old drafts.
3. SHA gates unchanged for frozen anchor snapshot (9902d918); new columns affect newly built
   masterstars only.
4. `LOCATION_OLD` excluded from hygiene swap copy list (legacy mirror; no active path).
   `update_master_source_safety` deleted (unwired).
5. `PRAGMA quick_check` on `vyvar.sqlite3` is part of `--fast` gate after Phase 4.

**2026-08-22 addendum (Milan, binding).** DB hygiene swap is **dropped** as a prerequisite
for file-based work. Direction extends files-over-DB: corrupt `MASTER_SOURCES` btrees sit in
a store code no longer reads; blocking file jobs on a retired store is process debt. Consequences:

- `db-quick-check` is **waived** (explicit committed marker in
  `dev/validation/db_quick_check_waiver.json`); gate stays, verdict downgrades to WARN with
  waiver text on every run (INV-GATE-REMOVAL).
- **DB-RETIRE-01** (FUTURE): migrate remaining DB stores (EQUIPMENTS, TELESCOPE, LOCATION,
  CALIBRATION_LIBRARY, FITS_HEADER_CACHE) to file-based storage; retire `vyvar.sqlite3` entirely.
- DB-writing jobs remain discouraged until DB-RETIRE-01; file-based jobs (RUN ePSF, uploads,
  exports) are **unblocked**.

**Reports:** `dev/results/CURSOR_RESULT_MS_SOURCES_RETIRE_01_AUDIT.md`,
`dev/results/CURSOR_RESULT_MS_SOURCES_RETIRE_02.md`.

---

## D10-1-CLOSE - unfiltered band letter CV affirmed (2026-08-21)

**Decision (Milan, binding).** Unfiltered wide-rig exports use AAVSO/VarAstro band code
**CV** (Johnson V comparison magnitudes). Zero code change: existing
NOFILTER/CLEAR/L/LUM/LUMINANCE -> CV mapping in `export_reports.py` stays as-is.

**Evidence (pre-registered dual measurement on frozen era-516 NoFilter snapshot).**

- **D10-1** ensemble fit (n=2076, BP-RP span 2.34): b_V = +1.6 +/- 67, b_R = +495 +/- 67
  mmag/mag; mechanical rule -> CV.
- **D10-1b** raw-counts probe (n=1567, G<=14, BP-RP span 0.46-2.80): b_V = -29 +/- 65,
  b_G = +358 +/- 65, b_R = +466 +/- 65 mmag/mag; identity gates b_R-b_V and b_G-b_V
  passed; same rule -> CV.
- **Physical finding:** Zeiss 200 mm + QHY294MM NoFilter effective band is V-like (I-V flat,
  I-G sloped) - real bandpass physics, **not** a construction artifact (D10-1b provenance
  audit: Gaia G only on LHS ZP; b_G discriminator; raw mag_inst confirms pattern without ZP).
- **Architect retraction on record:** artifact hypothesis raised in D10-1 review was tested
  and disproven by the raw-counts probe.
- **Colour systematics at pinned submission targets under CV:** sub-mmag (BO +0.9, FW -1.0,
  GH +1.5 mmag); CR would inject ~15-25 mmag.

**Reports:** `dev/results/CURSOR_RESULT_D10_1_CV_CR.md`,
`dev/results/CURSOR_RESULT_D10_1B_CV_CR.md`.

**Methods-paper caveat.** Slopes are 5-sigma robust but averaged over large per-star residual
scatter (transform sigma, faint end); state explicitly when publishing.

**D10-2 closure (same date).** Gaia->Johnson validity guard verified present and enforced
(`gaia_johnson.py:38-42`, `140-155`; BP-RP [-0.5, 5.1], G [8, 16]); exercised in D10-1
selection (11+3 stars excluded).

---

## Canonical --full / P1 mode and clean-rebuild freeze (ANCHOR-516-04, 2026-08-18)

**Canonical headless mode.** `--full`, P1 golden, and product rebuilds run with
PFS ON, `export_err_mode=calibrated`, empirical background ERR
(INV-ERR-MODE-01 fail-loud if `sigma_bkg_ap` missing), and
`saturate_limit_fraction=0.80` (peak-test 52428 ADU). ROT in
`vsx_out_of_scope_types` is deliberate policy, not a defect.

**One-authority sat-limit includes the MASTERSTAR writer.** Live
`saturate_limit_fraction` is not enough: `_annotate_masterstars_flux_zones`
and per-frame `saturate_limit_adu_85pct` / leftover `linearity_limit_adu`
copies must carry 52428. A product that consumes a 0.85-annotated
`masterstars_full_match.csv` is not a clean rebuild (be6191e0 lesson).

**Anchor-from-clean-rebuild.** The `--full` gate photometers *frozen
snapshot inputs* copied into `tmp/session_baseline/<ts>/`, never a live
draft. A golden that photometers a live draft under evolving config is
not a reproducibility gate (435 lesson). Reference product SHA
**477dc8cf** (MAG = de6f7c8; ERR empirical). P1 mini is
`draft_000516_p1mini` and must include `cal_diag.json` where INV-CAL-01
looks (draft root and `platesolve/<setup>/`).

**435 retired by design.** Not repaired. DELETE-OK list is for Milan.

---

## ERR-518-02 - sigma_bkg_ap key transport + RAM finalize (2026-08-21)

**Implemented (Milan decision):** options 1+2 from ERR-518-01. `_sigma_bkg_r_key()`
canonicalizes every `_sigma_by_r` store and lookup; INV-ERR-SIGMA-ACCT-01 raises
`InvariantViolation` when Labbe succeeds but zero rows receive empirical source.
RAM handoff invokes `_finalize_hybrid_bkg_fallback_sidecar` once after deferred
CSV flush (direct-write path unchanged; no double finalize because defer path
skips the in-export call).

**Rejected (Milan 2026-08-21):** option 3 (block empirical without accepted SNR
table) - would hard-block Newton drafts that legitimately run `global_fixed` until
SNR gate passes; option 5 (ops workaround regen SNR table only) - treats symptom,
not transport bug.

**Deferred:** PRECAL-INPUT-CONTRACT-01 (option 4) stays ROADMAP; Howell sky_pp on
externally pre-calibrated pedestals is not fully resolved here.
CAL-520-01 (2026-08-24, measure): third evidence instance -- draft 520
clicked `[folder] RUN VYVAR (non-cal)` on `RAW_NON_CALIBRATED` Brno
lights; INV-PREP-01 logged 0.02x and stayed informational; no eq=4/tel=6
masters in CalibrationLibrary. Gate, when wired, must be star-masked and
bias-aware (M2.1), not INV-PREP-01's unmasked variance ratio.

Proof: draft 518 re-export 62352/62352 empirical; anchor 516 core SHA unchanged
9902d918.

---

## INV-ERR-MODE-01 - empirical ERR fail-loud (2026-08-18)

Headless Phase 2A with `err_background_mode=empirical` raises if
`sigma_bkg_ap` is missing/NaN instead of silently falling through to
Howell. Cache columns are derived from the named requirements map.
all 134 proc CSVs; sem/scint/sys deltas vs de6f7c8 = 0.

---

## GAIN-PT-RADIUS-01 - pin photon-transfer aperture at 4.0 px (2026-08-17)

**Mechanism.** PT empty-aperture radius is always
`PHOTON_TRANSFER_APERTURE_R_PX = 4.0` with source string
`pinned_sky_dominated_4px` (`resolve_photon_transfer_aperture_r_px`).
Leftover `pipeline_meta.dynamic_params.aperture_r_px` is never read for
PT. `force_aperture_px` sizes star photometry only and does not override
PT. Sidecar records `aperture_r_px` and `aperture_r_px_source`. No new
config key (291).

**Why.** Reading leftover meta before this-run apertures exist selected
r=2.499 on 36a53b0, widened CI to 6.22, and fell back to DB/scale
(GAIN-AUTH-VERIFY-01). WIDE-ERR-03B B3: PT needs sky-dominated ~4 px.

**Proof.** On draft 515, pinned PT gives g_pt=0.63707, CI factor 2.468,
authority=g_pt (byte-identical to WIDE-ERR-04). ERR-only Phase 2A:
MAG 48/48 byte-identical; product SHA **de6f7c8** (err columns).
SUBMIT-01 checklist all PASS. Evidence:
`dev/results/CURSOR_RESULT_GAIN_PT_RADIUS_01.md`.

---

## U-09 - DATE-OBS is shutter-open; export is mid-exposure BJD (2026-08-17)

**Closed (verdict a)** on the home wide rig (draft 515, NoFilter_60_2,
QHY294PROM, 60 s). Real-frame DATE-OBS comment is "UTC start date of
observation". Consecutive BO CVn DATE-OBS gaps are 121.000 s median
(EXPTIME 60 s + 61 s dead time; 149/149 gaps >= EXPTIME; 0 overlaps).
VYVAR adds EXPTIME/2 in `time_utils.mid_exposure_jd` (not at export).
Measured jd_mid - jd(DATE-OBS) = 30.000 s; AAVSO DATE is BJD_TDB of that
mid (jd_export - jd(DATE-OBS) = 398.847 s median = 30 s + TDB/LTT).
Ekos/INDI logs were not on this PC; start vs end would still want a log
excerpt, but mid-exposure DATE-OBS is incompatible with the on-frame
comment. Other rigs stay unverified. Evidence:
`dev/results/CURSOR_RESULT_U09_GAIN_AUTH.md`.

---

## GAIN-AUTH-VERIFY-01 - 36a53b0 used DB/scale because PT CI was wide (2026-08-17)

**CLOSED** by GAIN-PT-RADIUS-01. Root cause was leftover
`dynamic_params.aperture_r_px=2.499` overriding the intended ~4 px PT
radius (CI factor 6.22 > 3). Pin at 4.0 restores g_pt=0.637 authority.
Pointer: `CURSOR_RESULT_U09_GAIN_AUTH.md` (diagnosis);
`CURSOR_RESULT_GAIN_PT_RADIUS_01.md` (fix).

---

## PFS-SEMANTICS-01 - rescue keyed on skip_reason; one limit authority (2026-08-17)

**Teach, do not disable.** `per_frame_saturation_enabled` remains a valid
per-run override (AppConfig default false; 291 keys). Rescue is keyed on
the recorded `skip_reason` / saturation zone, never on bare
`skip_photometry`.

**Never rescue** `zone_noise` or `below_target_depth`. TARGET-DEPTH-02
outranks PFS. Phase 2A re-force always keeps noise out; saturation-zone
skip is re-forced only when PFS is OFF.

**One limit authority.** Per-frame clean test uses
`inv_sat_limit_peak_test_adu()` (INV-SAT-LIMIT effective peak-test,
currently 0.80 x 65535 = 52428 ADU on this rig). Container clip 65535 ADU
is a separate named field (hard clip vs nonlinearity). Provenance cites
the source string.

**8f107cf** is quarantined (VL-PFS-8F107CF): PFS rescued non-saturation
skips (96 LCs). Do not cite for export or acceptance.

**SAT-RERANK-01** DONE on SHA 36a53b0 (48 LCs; CV CVn gated by
clean_frac=0.448 < 0.5). Pointers: `CURSOR_RESULT_PFS_SEMANTICS_01.md`,
`CURSOR_RESULT_SAT_RERANK_01B.md`. SAT-LIMIT-01 stays CLOSED. D1-2 OPEN.

---

## SAT-LIMIT-01 - never silently admit a missing saturation clip (2026-08-17)

**Hole.** Draft 515 MASTERSTAR catalog had `saturate_limit_adu` NaN on
3621/3621 rows. EQUIPMENTS id=1 SATURATE_ADU is NULL (wrong-16384 wipe);
MASTERSTAR FITS is BITPIX -32 so BITPIX inference is None; SAT-DIAG 65535
DERIVED was not wired into MASTERSTAR annotate. `peak > NaN` is False, so
`is_saturated` never fired. C2 (G=7.99, peak 64350 ADU = 98.2% of 65535)
sat in the BO ensemble as zone=linear.

**INV-SAT-LIMIT.** Unresolved clip -> WARN-named conservative default
(GAIN-DOMAIN-01 container 65535, peak-test 0.80 x clip = 52428 ADU when the
D1-2 knee is unmeasured). Never silent admit. Float-stack overshoot of a few
percent does not veto the camera clip; a true unit change (frame_max > 1.20 x
raw clip) still skips.

**Knee.** Cheap residual-vs-peak on 515 did not resolve a linearity knee.
D1-2 stays OPEN (dome-flat ramp).

**515 impact.** 24 stars reclassified saturated, including C2. Product-frame
check MAD without C2 and without a replacement rose 7.05 -> 8.58 mmag
(meter match to D515-ACCEPT-01B). SAT gate still excludes C2; a quieter
ensemble needs Phase 1 re-rank. Production comparison CSV not rewritten.

**Artifacts.** `dev/results/CURSOR_RESULT_SAT_LIMIT_01.md`,
`SAT_LIMIT_01_summary.json`.

---

## XVAL-AIJ-01 - three-link photometry cross-check (2026-08-17)

**Chain.** VYVAR differential photometry is now closed against three independent
reductions on the same data:

1. **Library:** photutils/sep vs VYVAR dao_flux, ~3 mmag RMS (audit Wave 7 /
   TODO-SEP-XVAL harness).
2. **Product formula:** architect independent reconstruction of mag_calib,
   0.0001 mmag (XVAL-BO-01; formula identity, not a second photometry engine).
3. **External tool:** AstroImageJ 6.0.10 vs VYVAR, 3.3 mmag RMS per epoch
   over 134 frames of BO CVn (0.47 mag ptp eclipser), identical five-star
   ensembles. Evidence: `dev/results/XVAL_AIJ_01_bo_compare.csv`,
   `CURSOR_RESULT_XVAL_AIJ_01.md`.

JAAVSO methods sentence: VYVAR and AstroImageJ agree to 3.3 mmag RMS per
epoch over 134 frames of a 0.47-mag-amplitude eclipser using identical
comparison ensembles.

---

## XVAL-AIJ-02 - production 4-comp ensemble and two frame states (2026-08-17)

**Chain (extends XVAL-AIJ-01).** library (photutils/sep ~3 mmag) ->
product formula (0.0001 mmag, XVAL-BO-01) -> external tool on a matched
5-star ensemble (3.3 mmag, XVAL-AIJ-01) -> external tool on the
**production clean 4-comp ensemble** and on **two frame states**
(calibrated unaligned vs detrended_aligned). Saturated C2 is absent.
Production comps (RA/DEC-matched): `1497771992240531712`,
`1499200223486564608`, `1497974027502858240`,
`1497368849430107904`. Domain: BO CVn 2026-04-23, 134 epochs, VYVAR
product SHA **de6f7c8**, AIJ 6.0.10.

**Matrix (per-epoch mmag; sanity-read from committed tables).**

| cell | RMS mmag | note |
|------|---------:|------|
| AIJ run3 vs architect 4c rebuild of run1 | 0.00 | byte-equal (drop saturated C2 from run1 C3-C6 = run3 C2-C5) |
| frame-set: run3 vs run2 | 3.80 | 131 common epochs; 1 broken cal frame (FWHM=0) excluded; r(airmass/time) ~0 |
| weighting: pytics vs flux-sum on AIJ fluxes | 2.52 | production `comp_weight` on AIJ Source-Sky; spec 2.70 not reproduced |
| tools, old 5c (XVAL-AIJ-01) | 3.27 | reference row, unchanged |
| tools, production 4c: run3 vs mag_calib_final | 4.86 | 134/134; depth 462.7 vs 469.6 mmag |
| AIJ run3 vs delta_mag (flux-sum layer) | 4.74 | same epochs; spec 4.72 |

The rise 3.27 -> 4.86 is expected physics, not regression: the old
flux-sum was dominated by bright C2 in both tools, so common-mode noise
cancelled in the difference; the clean 4-comp sum of fainter stars
averages per-tool measurement noise less. 4.86 mmag sits inside the
combined error budget (VYVAR 8.9 mmag median + AIJ CCD-eq ~7.3 mmag
median on this 4c table; XVAL-AIJ-01 5c was ~6.2 mmag).

**QC finding.** Run2 (calibrated, 145 frames) contains 13 epochs absent
from the aligned/detrended AIJ table (VYVAR QC dropped them from the
aligned set). Residual vs a 7-point running-median local curve: median
|resid| 8.56 mmag vs 3.64 mmag scaled MAD (x 1.4826) on the accepted
epochs (~2.4x worse) -- independent-tool evidence that the QC filter
selects genuinely bad frames. Frames 049 and 111 of those 13 are
photometrically fine in AIJ (resid ~0 mmag), consistent with
alignment-reason rejection, not photometric.

Evidence: `dev/results/XVAL_AIJ_02_bo_compare.csv`,
`XVAL_AIJ_02_Table_calibrated.tbl`,
`XVAL_AIJ_02_Table_detr_aligned.tbl`,
`CURSOR_RESULT_XVAL_AIJ_02.md`.

JAAVSO methods sentence: VYVAR and AstroImageJ agree to 3.3 mmag RMS
(matched 5-star ensembles) and 4.9 mmag RMS on the production clean
4-comp ensemble over 134 epochs of a 0.47-mag eclipser; processing-chain
(alignment+detrending) effect bounded at 3.8 mmag RMS with no
airmass/time correlation.

---

## WIDE-ERR-04 - close at physical model (2026-08-16)

**Closure.** Wide-rig exported errors closed at the **physical model**:
container-domain `g_pt` authority (GAIN-DOMAIN-01) + weighted ensemble SEM,
with identity calibration `s=1`, `sigma_r=0` (no empirical floor). The 03B
~6.9 mmag constant floor over-inflated the product frame; C2 held-out chose
identity. Machinery `(s, sigma_r)` and `export_err_mode=calibrated|model`
remain as the documented per-rig hook.

**Accuracy (product-frame even-half, draft 515, SHA da9cce4 photometry).**
Across G 8-13 gated bins, median(scatter/err) is typically within ~+/-15% of
1; (11.0, 11.5] n=5 remains ~25% conservative (ratio 0.746) - see CORR-ERR-01.
Under-quoting (the scientifically harmful direction) is eliminated.

**Literature context.** AAVSO amateur practice often quotes SNR-based errors
that the AAVSO forum itself acknowledges under-estimate. Professional
transit/variable work rescales to chi2~1 and/or adds a red-noise floor
(Pont, Zucker & Queloz 2006; Gillon 2009). VYVAR now sits at or above both
bars: physical budget without under-quote; residual conservatism documented
rather than silently absorbed by s<1.

**Decision rule.** Conservative errors are acceptable; under-quoted errors
are not. CORR-ERR-01 stays OPEN as a LOW research note (covariance-aware
budget out of v1.0).

**Artifacts.** `dev/results/CURSOR_RESULT_WIDE_ERR_04.md`, accuracy + reexport JSON.

---

## WIDE-ERR-03C - product-frame meter + CORR-ERR-01 (2026-08-16)

**Product-frame meter rule (XVAL lesson generalized).** Calibration and
acceptance gates must measure the product that ships: pytics-weighted
`mag_calib` (dao_flux -> m_inst; zp = sum w_j (G_j - m_j) / sum w_j), not the
LOO flux-sum / delta_mag diagnostic frame. INV-CALIB-HOLDOUT still applies
(odd/even frames; fire proof required).

**Escalation outcome.** C1d: existing 03B constant floor on the product meter
FAILED harder (ratios ~0.70-0.78) - not a meter artifact that would rescue
03B. C2: held-out chose constant with s clamped to 1.0 and sigma_r=0 (physical
model alone); still FAIL at (11.0,11.5] ratio 0.746. C3: per-LC floor FAIL.
STOP: open **CORR-ERR-01** - observed product-frame scatter lies below the
diagonal physical err_model (s=1, sigma_r=0) in gated bins; common-mode
cancelled by the catalog ZP is not a diagonal (s, sigma_r) residual.
WIDE-ERR + SEM stay OPEN. WIDE-ERR-CROSSRIG stays OPEN. No forced pass.

**Artifacts.** `dev/results/CURSOR_RESULT_WIDE_ERR_03C.md`, `WIDE_ERR_03C_C*.json`.

---

## WIDE-ERR-03B - smooth clamped calib + INV-CALIB-HOLDOUT (2026-08-16)

**s >= 1 clamp.** Per-draft err calibration may only inflate errors beyond the
physical model (`s >= 1` always). Never deflate via s < 1. A genuine model
overprediction (e.g. faint-end ratio 0.75-0.85) is a MODEL question (g_pt
center vs CI; register GAIN-PT-CI-01), not something a few comps may absorb.

**Smooth form, few parameters.** Replace per-bin (s, sigma_r) tables with a
smooth form on all calibration stars at once: one constant s and sigma_r(G)
either constant or a monotone non-increasing-in-flux smooth function (e.g.
linear in G, non-negative). Rationale: ~50 clean comps cannot support ~20 free
parameters; they can support 2-3. Form choice is by held-out performance, not
in-sample fit alone.

**INV-CALIB-HOLDOUT (standing rule).** Any future per-draft err calibration
must be validated non-circularly: fit coefficients on a held-out split
(prefer odd/even frames when star count is scarce; also report star-split),
and accept only on the evaluation half. A gate that fits and accepts on the
same stars is not a verification (WIDE-ERR-03 S5e defect). Deliberately wrong
calibration must fail the same gate (fire proof).

**Artifacts.** `dev/results/CURSOR_RESULT_WIDE_ERR_03B.md`, `WIDE_ERR_03B_B2.json`,
`WIDE_ERR_03B_B3.json`.

---

## WIDE-ERR-03 - container-domain gain + photon-transfer authority + calibrated err (2026-08-16)

**Unit chain (S1c, verbatim).** QHY294MM digitizes a 14-bit native sample and
stores it in a 16-bit FITS container by a left shift of two bits, so container
ADU = 4 * native ADU (raw lights sit on a residue class mod 4; S1a). The
equipment DB gain 3.17 e-/ADU is the native-domain conversion; the matching
container-domain gain is therefore 3.17/4 = 0.7925 e-/ADU_container
(photon-transfer on draft 515 gives g_pt ~ 0.635 e-/ADU_container inside CI
[0.44, 1.09], which excludes bare 3.17). Hardware/software 2x2 binning on this
CMOS path sums container-domain electrons and ADU together, leaving
e-/ADU_container unchanged to first order; bias/dark/flat calibration and later
resampling mix the integer grid into floats but do not change the ADU scale away
from the container domain. Production photon and SNR terms that consume flux and
sky in container ADU must therefore use g_container (g_pt when available, else
DB/4), never bare DB gain; read noise remains in electrons and enters only as
(RN/g)^2*npix in ADU^2.

**Gain authority.** Data-derived `g_pt` (Theil-Sen empty-aperture var vs
npix*sky) when CI finite and width factor <=~3; else DB_native/container_scale.
Never bare DB on container ADU. Log authority (INV-NO-SILENT). WARN if g_pt and
DB/scale disagree >2x.

**Retraction.** Architect fitted effective gain 0.24-0.32 (WIDE-ERR-02 W1c STOP)
is superseded: g-k_sky degeneracy; sky dominates photon above G~11. Standing
number is the photon-transfer CI.

**Err export.** `err_exported^2 = (s * err_model)^2 + sigma_r^2` per draft x rig
x G bin on clean comps in the mag_calib-like frame (Pont+2006 / Gillon 2009),
plus weighted ensemble SEM matching ZP weights (SEM-WEIGHT-01). Config
`export_err_mode=calibrated|model`. SysRem stays out of v1 (modifies LCs).

**Citations.** Tamuz+2005; Pont+2006; Kovacs+2005; Gillon 2009; Winn 2008.

**Artifacts.** `dev/results/CURSOR_RESULT_WIDE_ERR_03.md`, S1/S2/S4/S5 JSON,
`gain_photon_transfer.json`, `err_calibration.json`.

---

## WIDE-ERR-02 - STOP at W1c (gain); calibrated err + SEM not shipped (2026-08-16)

**Task intent.** One wave: Pont/Gillon-style
`err_exported^2 = (s * err_model)^2 + sigma_r^2` calibrated in the
**mag_calib / comps-only** frame (XVAL-BO-01 lesson), plus the weighted SEM
fix (measured ratio 0.677). SysRem remains out of v1 scope (modifies LCs).

**Gate fired.** Part W1c: production gain on draft 515 NoFilter_60_2 is
**3.17 e-/ADU** (`draft_manifest.rig.equipment_id=1` -> EQUIPMENTS QHY294MM /
`header_index_mapped` GAIN setting 0; stamped in `pipeline_meta.dynamic_params`).
Architect fitted effective gain from the empirical photon term is **0.24-0.32**.
Disagreement >2x (ratio to mid ~11). Task rule: STOP and report before
implementing calibration - a white scale + red floor would mask a wrong photon
term.

**W1b did not stop.** Production sys+scint hypot at G8-9 is ~4.0 mmag (sys=0
for equipment 1; scint alone), reproducing the ~2x gap vs LC-frame ~6-8 mmag.

**Disposition.** W3 (calibrated export + weighted SEM) **not implemented**.
WIDE-ERR and SEM stay OPEN. WIDE-ERR-CROSSRIG stays OPEN. Next prerequisite:
resolve why production e-/ADU and the fitted effective differ by an order of
magnitude (units, flux domain, or DB gain) before any err-bar calibration wave.

**Artifacts.** `dev/results/CURSOR_RESULT_WIDE_ERR_02.md`,
`WIDE_ERR_02_prod_components.json`, `WIDE_ERR_02_summary.json`.

---

## PUSH-STAMP-01 - content tip vs origin SHA (2026-08-16)

**Rule.** A close result records the **CONTENT tip** (last substantive commit of
the close) and the pushed range up to it. The final `origin/main` SHA is
**never** written into a committed file, because committing that string changes
HEAD and forces another stamp.

**Precedent this ends.** Stamp-on-stamp chains: PUSH-02 (`5872634` -> `738a24e`
-> `691d3be`); SESSION-CLOSE (`1521d3c` -> `1b8fcab` -> `9f139ef`). One docs
commit per close is enough; do not chase self-referential tip SHAs.

**Practice.** STATE/ROADMAP/RESULT may name the science tip and the content
pack tip. After `git push`, the report text (uncommitted or chat) may mention
the received remote SHA; that SHA is not a reason for a further docs commit.

---

## RUN-WORKER-01 - subprocess job worker (open; 2026-08-16)

**Incident (HANG-P1-T16-01).** Draft 515 RUN VYVAR executed Phase 0+1 on the
Streamlit script thread. At Phase 1 target 16/97 the Variability auto-TESS
branch issued `st.rerun()`, silently aborting the RUN. py-spy later showed an
idle Streamlit process; the Gaia box-scan hypothesis was dead (indexes present,
0.4 ms query). The UI footer stayed stale ("8/97") with no `[RUN]` exit line.

**Mitigations shipped in RUN-HARDEN-01 (B+C):** (1) `run_callable_with_exit_log`
around the UI RUN entry emits `[RUN] finished OK` / `aborted: ...` /
`interrupted by script rerun`. (2) Variability and Aperture auto-crossmatch /
auto-TESS branches no-op while `vyvar_footer_state.running` is true.
(3) Phase 1 progress logs every target.

**Open architecture (do not implement until Milan decides):** RUN-WORKER-01 -
move Phase 0+1+2A (and optionally the full RUN VYVAR chain) into a **subprocess
or dedicated job worker** that writes a status file (PID, step, pct, last
heartbeat, exit code) under the draft or `tmp/`; the Streamlit UI only queues
the job and polls the status file. Browser refresh, session timeout, and tab
auto-`st.rerun()` then cannot kill photometry. B+C remain as in-process
guards for any residual UI-thread work.

---

## GATE-OWNERSHIP v1 scope - D3 unit + D4 universality (2026-08-15)

Recorded from GATE-OWNERSHIP-01 (`5612f42`); not a retune. v1.0 scope for how
gates are judged when ownership work continues.

**D3 - threshold-must-carry-unit.** Every science-path gate threshold must
declare a physical unit (mag, ADU, dimensionless excess, fraction, mmag/hr,
...). A bare number in a registry or inventory row is incomplete. At inventory
time only 6 of 274 params had non-null `unit`; ownership work may not close a
conflict without naming the unit of decision.

**D4 - universality-as-test.** A star's admit/reject fate may depend only on
that star and the target (and fixed physical constants), never on the
composition of the peer pool for that draft. Rank-statistic cuts fail this
test by construction (GATE-OWNERSHIP R2; COMP-ADMIT-03 removal). Field-
relative percentiles remain diagnostics / variable-search tools, not admission
owners.

Report: `dev/results/CURSOR_RESULT_GATE_OWNERSHIP_01.md`.

---

## COMP-ASSIGN-01 - stability is a post-LC verdict, never a selector (2026-08-15)

**Decision.** `check_comparison_stability` runs **after** light-curve write.
It may write sidecars and trust flags; it must **not** drop or reshuffle the
comparison set used by `ensemble_normalize`. Membership is decided in step 2
(RMS/colour/distance + ceiling + 3-8 clamp); stability is a verdict on that
fixed set.

**Lineage.** Extends ZP-CLIP-REMOVAL: varying membership mid-ensemble (or
post-selecting by residual scatter) reintroduces the two-state ZP failure mode
that ZP-CLIP-REMOVAL removed. Comp QA (IMPL-05 D) remains a guarded post-LC
consumer of the step-2 pool only.

Report: `dev/results/CURSOR_RESULT_COMP_ASSIGN_01.md`. Tip stamp `635404f`.

---

## IMPL-01 - no ensemble size cut in v1.0 (2026-08-15)

**Decision.** Do **not** cut the comparison ensemble by cumulative weight, N_eff, or
rank in v1.0. Keep the full admitted pool with existing `1/sigma_eff^2` weights.

**Evidence (PRE-IMPL-01 / DRAFT-514-TRIAGE):**

- **Science.** Check-star scatter is flat from the full ensemble through 90 percent of
  cumulative weight (16.49 to 16.92 mmag). The far tail does not improve scatter.
- **Cost.** `ensemble_normalize` is about 2 s of roughly 58 s per-target wall time at
  n=1292 (~3.5 percent). A cut does not buy meaningful runtime.
- **Constructibility.** An absolute, field-independent inclusion ceiling from `sigma_eff`
  was not established. Q1's `sigma_obs/sigma_eff` trend is **contaminated** (see OPEN
  PRE-IMPL-01-Q1 below): absence of evidence for a scale, not evidence against one.

**Future cuts.** Any later ensemble cut must be **field-independent**: a star's
inclusion may depend only on that star and the target, never on the rest of the pool.
Candidates with an absolute scale: `comp_rms` (mag) or median SNR. Cumulative-weight
or N_eff truncations remain performance experiments, not v1.0 policy.

Report: `dev/results/CURSOR_RESULT_IMPL_01.md`.

---

## OPEN - PRE-IMPL-01 measurement corrections (register only; 2026-08-15)

Architect review: four conclusions do not follow from their measurements. Recorded
OPEN so they are not later cited as settled. **Do not act** in IMPL-01.

1. **PRE-IMPL-01-Q1 contaminated.** Q2 replaced flux-sum `loo_diff_series` with
   weighted-mean peers; Q1 kept the flux-sum estimator. In the Q1 sample `sigma_obs`
   is ~60 mmag flat while `comp_rms` is 6-9 mmag (reference floor). A constant
   `sigma_obs` divided by magnitude-dependent `sigma_eff` produces the reported
   2.85 to 1.08 trend automatically and makes `excess2` constant. Conclusion that
   `sigma_eff` has no absolute scale is **unsupported** (and so is its opposite).
   Re-run with weighted-mean peers when convenient.

2. **PRE-IMPL-01-Q2 read one column.** Scatter improves at 50 percent truncation
   (16.49 to 12.79 mmag) but airmass slope worsens (101 to 255 mmag per airmass).
   A 255 mmag/airmass drift contaminates light-curve shape worse than 3.7 mmag of
   scatter. The claim "far tail carries a missing `sigma_eff` term" used the scatter
   column only.

3. **PRE-IMPL-01-Q5 matched on the wrong quantity.** Blends looked 16 mmag quieter
   than isolated stars at matched catalogue G. A blend carries more flux than its
   catalogue G implies, so matching on catalogue magnitude does not match signal.
   Match on **measured flux**. Deferring blend merging is probably still right; that
   decision does not follow from this measurement.

4. **PRE-IMPL-01-Q4 night EE quantity.** Q4 reported night EE variation MAD
   0.008 mmag. The measured quantity is **frame-to-frame absolute EE of one isolated
   star at fixed production radius** (common-mode aperture-loss amplitude), demeaned
   MAD - **not** the residual after ensemble common-mode removal. Whether 0.008 mmag
   is physical (stable seeing that night) or method-limited remains open; it must not
   be cited as a differential residual. See IMPL-01 Item 1 artifact.

---

## CLOSED - EMPTY-DAO-01 (was register-only; 2026-08-16; closed 2026-08-21)

IMPL-05 Item A fixed the dtype crash in `_proc_deduplicate_matched_catalog_rows`
when `peak_max_adu` / `dao_flux` / `flux` are absent. **Closed 2026-08-21:**
empty-sky false-fill addressed by era DAO-Gaia audits (517 certificate
empty_sky pass2 0.14%, 3/2200). See CLOSED section at file end and ROADMAP.

---

## Wave 6 PROPOSE items - Milan authorized 2026-08-13 (implement later)

| ID | Decision |
|----|----------|
| W6-PROP-03 | **Option A:** stamp `VY_QCBG_PRE` at cal QC; write `VY_QCBG` at preprocess in same FITS write (INV-CAL-02). |
| W6-PROP-01 | **Option A:** `detect_outliers` clips constants only; variables stay mask-protected. |
| W6-PROP-05 | Wire `count_final_data_for_*` into Calibration Library delete dialog before OK. |
| W6-PROP-02 | Rename to `preprocess_calibrated_lights_in_place`; one-release deprecated shim. |
| W6-PROP-04 | Reachability doc: `unwired_ui`=1 (`ui_finalization` only). |

---

## A-1 aperture EE - decision (2) implemented on draft 510 (2026-08-14)

Growth curves (2026-08-13) measured EE at production radius **81-86%** (510) vs **67-73%** (435), below a 90% target. **Decision (2):** SNR aperture table FWHM authority moved from stack `VY_FWHM_GAUSS` to **per-draft median of per-frame DAO moment FWHM** (`fwhm_px_scope`: `per_draft_median_frame_dao_moment`). `VY_FWHM_GAUSS` recorded for provenance only.

**Wave 7 verification (draft 510):** re-export 135 proc sidecars + Phase 2A; `aperture_px` **4.141 -> 4.261**; check_scatter **0.008629 -> 0.008638**; EE **+0.8 pp**; xval target agreement still **~3 mmag**. Draft **435 not re-cut** (no aligned science FITS). Closure: `docs/VYVAR_AUDIT_2026_CLOSURE.md`.

**Deferred:** decision (3) COG correction; decision (4) fixed 90% enclosed fraction (~5.0-5.75 px on typical 510 comps).

Report chain: `dev/results/CURSOR_RESULT_a1_growth_curves.md`, `dev/results/CURSOR_RESULT_a1_snr_dao_authority.md`.

---

## SNR-GATE-01 - prematch peak noise scale + pass-2 exempt (2026-08-14)

**Problem.** After `c9e1f8f`, `plain_mean_med_std` returned unclipped full-frame sample std
(~570 ADU on draft 512 MASTERSTAR) while call sites still looked like clipped background
stats. Prematch `median + 1.8*std` therefore operated at ~12x real sky noise and discarded
pass-2 Gaia recoveries that F2 showed are real stars through ~G15.

**Decision.** (1) Prematch noise scale = `sky_mad_sigma_adu` (MAD on pixels <= median),
validated by `sigma^2` vs sky linearity (implied g~2.94 vs QHY294MM 3.17). (2) Pass-2
detections are exempt from the global peak gate; local annulus test + catalogue seed stand.
Pass-1 keeps the sky-MAD peak floor. Do not retune `k` to hit a star count (G-R3).
(3) SNR-GATE-02: `_noise_floor_adu_from_image_array` stays on legacy plain sample std so
SNR aperture radii do not move in this commit.

**Measurement (draft 512 MASTERSTAR):** 735 -> 3614 rows; Delta A carries the count; Delta B
is zero numerically on this night but retained. Full LC reprocess not run in the task.

Report: `dev/results/CURSOR_RESULT_SNR_GATE_01.md`. Commit/push await Milan.

---

## SKY-CLIP-01 - plain annulus median sky, no one-sided clip (2026-08-14)

**Problem.** Production batch aperture photometry used `_sky_pp_from_annulus_image` with a
one-sided upper clip (`sky_pixels < median + 2*std` then re-median). The single-star path
`_annulus_sky_subtracted_flux` used plain median. Q1-XVAL-MATCHED arm P1 measured
`phot_plain - vyvar = -0.000585` (VYVAR flux higher => sky lower), consistent with downward
bias from the clip.

**Literature / tools.** IRAF `apphot` annulus sky modes are median/mode (no asymmetric clip).
DAOPHOT uses sky-plane estimation without one-sided upper rejection on the annulus sample.
SExtractor/photutils default to robust estimators (median) with optional symmetric sigma
rejection in background meshes, not a one-sided annulus cut. AstroImageJ/C-Munipack: annulus
median. The literature's standard answer to PSF-wing contamination in the annulus is the
**mode** estimator (DAOPHOT; SExtractor `BACK_TYPE` / local background mode). Mode requires
no rejection step. It was identified in the survey and **was never measured** on VYVAR
data. Only median, mean, and the one-sided clip were measured. The estimator question is
not closed.

**Synthetic trade-off (CLOSE-IRON-GATES).** On clean sky the plain median is unbiased.
Under PSF-wing contamination the plain median bias is **+0.276 ADU** against the clipped
estimator's **+0.069 ADU** -- roughly four times the contamination bias. Moving to the
median bought unbiasedness on clean sky and compliance with iron rule 1, and paid that
contamination cost. BO CVn is not a crowded field; a crowded field is a different case.

**Decision.** Replace clip with **plain median** of annulus pixels (`method="center"` mask).
Unify all production call sites on `_sky_pp_from_annulus_image`. Register: SKY-CLIP-01,
IRON-GATES-01 INV-NOCLIP-01.

**Measurement (draft 510, FITS recomputation).** Median fractional flux change
new-old = **-0.058%** (-0.027% target, -0.070% comps). Stored proc CSVs match old clip
recompute exactly (max rel diff 0.0) until photometry re-export.

**Retraction.** Q1-XVAL-MATCHED arm P1 (0.06% clip vs median offset) is superseded by this
fix; replacement evidence: `dev/results/CURSOR_RESULT_CLOSE_IRON_GATES.md`.

---

## COMP-ADMIT-03 - admission is weights, not rank cuts (2026-08-15)

**Problem.** COMP-POOL-01 Stage 2 (`696c849`) added five rank-based comparison
admission cuts (p84 MAD/IQR/inv_eta, p16 detect_frac, dilution ladder, faint/bright).
They sat upstream of the word "admitted" in ZP-CLIP-REMOVAL and therefore did not
violate its letter, but they violated its decision: Broeg-family pipelines favour
using as many comps as possible with weights, not a tiny hand-picked set.
Rank cuts also make a star's fate depend on the composition of the field
(universality failure; GATE-OWNERSHIP-01 R2).

**Physical argument (INV-GATE-REMOVAL).** Each deleted gate discretised a continuous
quality already expressed by a weight:

- scatter -> `sigma_rms` in `sigma_eff`
- colour difference -> `c_col * |delta(BP-RP)|`
- angular distance -> `c_dist * r`

Rank-based cuts additionally reject a fixed fraction of a uniformly good population.
Byte-identity is not evidence for removal; the physics is.

**Gates that remain (exactly three):** saturation/non-linearity (measurement
invalid), known variable (VSX/Gaia flag), geometry (aligned footprint). Everything
else is weight:

```
sigma_eff^2 = sigma_rms^2 + (c_col * |dBP-RP|)^2 + (c_dist * r_deg)^2
w = 1 / sigma_eff^2
```

**Forced photometry.** Superseded by FORCED-PHOT-01 (2026-08-15): pool-eligible
MASTERSTAR members are injected and aperture-measured every frame at locked
MASTERSTAR XY after DAO match. Detection no longer gates presence.

**Broeg iteration.** VYVAR's `ensemble_normalize` uses one-shot `1/sigma^2` for the
catalog zeropoint (AIJ/Honeycutt unweighted flux sum for `delta_mag`). It does not
iterate residuals into weights. Adding colour/distance systematic terms to
`sigma_eff` does not introduce iteration; an iterative fit that down-weights is
not the same as discarding members (iron rule 1).

**Rule.** Comparison admission: three gates only. Continuous qualities are weights.
See INV-COMP-MEMBERSHIP. Extends ZP-CLIP-REMOVAL from the zeropoint to admission.

**COMP-ADMIT-03 review corrections (same date, FORCED-PHOT / COMP-WEIGHT tasks):**
- ``is_noisy`` is not a measurability gate (DAO peak significance); it belongs in
  weight via scatter.
- Gaia NSS -> known-variable gate; Gaia QSO/GAL -> measurability (extended source).

---

## FORCED-PHOT-01 - fixed membership by forced aperture (2026-08-15)

**Problem.** After COMP-ADMIT-03 removed `detect_frac`, pool membership grew while
per-frame presence remained DAO-conditional. ZP-CLIP-REMOVAL proved that varying
membership with a fixed-membership weighted mean is the defect; Honeycutt global
LS is parked. INV-COMP-MEMBERSHIP was unenforceable in practice.

**Decision.** Once a star is force-eligible for the draft (three gates), it is
measured in every frame at MASTERSTAR ``(x,y)`` on the aligned grid (bounded peak
refine, bound recorded in FWHM units). DAO discovery is unchanged. Geometry
out-of-footprint yields no measurement (recorded). Low SNR is kept. Per-frame
saturation keeps the row and excludes that epoch from the zeropoint explicitly.

**Answer.** After this change, a force-eligible pool member's per-frame presence
is **not** conditional on anything other than geometry (code:
`forced_photometry.inject_forced_masterstar_rows` after DAO in
`_export_per_frame_run_catalog_core`).

---

## COMP-WEIGHT-COEFF-01 - measure c_col and c_dist (2026-08-15)

**c_col.** CLEAR/unfiltered: literature k'' is NONE (no CHOSEN CMOS default).
PSF/EE term MEASURED on refractive Zeiss 200 mm: 29.485 mmag enclosed-flux
difference over BP-RP span 1.0 (0.5->1.5) from COMP-POOL-02 Item 4 =>
`c_col_psf = 0.029485 mag/BP-RP`. Combined in quadrature with `|k2|*DeltaX`.
Mirror optics: `c_col_psf = 0` predicted. Rig via `comp_weight_optics_kind`.

**c_dist.** OLS of `scatter_mad` vs field-centre separation on COMP-POOL-02
residuals: **measured zero** on drafts 512 and 435 (and 510 scatter path),
consistent with zero within uncertainty. Named measured zero, not a gap.
Ordering: measured on pre-forced products; re-verify after forced rebuild.

---

## ZP-CLIP-REMOVAL - no per-frame MAD rejection in ensemble_normalize (2026-08-12)

**Problem.** Draft 509 (same raw as draft 435, HEAD `682f40c`) produced check-star
scatter 0.025 vs 0.008 on 435. Instrumental photometry was fine on both (~0.008-0.009).
Cause: `ensemble_normalize` applied a 3x MAD clip to the per-frame zeropoint when
`len(z) >= 4`. Draft 435 had 3 comps and never entered it; draft 509 had 5 (after
`phase01_comparison_max_mag_diff` 1.5->2.0) and intermittently rejected bright TIER1
`1497771992240531712` on 37/134 frames, creating a ~50 mmag two-state ZP.

**Evidence (counterfactual matrix on 509 instrumental fluxes):**

| variant | comps | ZP clip | check `...1001088` | check `...4892800` | target res | shape |
|---------|-------|---------|--------------------|--------------------|------------|-------|
| A | 5 | ON | 0.0190 | 0.0187 | 0.020 | ZP bimodal |
| B | 5 | OFF | 0.0068 | 0.0085 | 0.012 | unimodal |
| C | 3 | OFF | 0.0079 | 0.0092 | 0.013 | unimodal |
| D | 3 | ON | =C | =C | =C | =C |
| E | 5 | OFF Broeg | 0.0073 | 0.0086 | 0.013 | unimodal |

Case: **B ~ C ~ 435 quality**. Clip is the entire cause. Broeg weights kept (not rejection).
`phase01_comparison_max_mag_diff` deliberately **not** reverted in this commit.

**Mechanism (measured).** Rejections were scattered (30 blocks, max run 3). On reject
frames the bright star's residual vs other comps was **quieter** (0.007) than on kept
frames (0.010). Absolute |z-med| stayed ~100-110 mmag; MAD collapsed (0.018 vs 0.037),
shrinking the 3-sigma boundary ~166 -> 80 mmag. 33/37 rejects had MAD below the overall
p25. Estimator noise at N=5, not a physical event on that star.

**Literature basis (methods-paper / decision rationale).**

1. **Broeg et al. 2005** (AN 326, 134) -- the method VYVAR already cites -- builds an
   artificial comparison as a weighted average. Variability weight and the decision to
   drop a star are properties of the star **across the whole series**, not of a single
   frame. Broeg-family pipelines (SPECULOOS-South / Murray et al. arXiv:2005.02423;
   Jena `chphot` arXiv:0905.1833; Cep-Cas / REM Orion follow-ons) drop low-weight or
   variable stars once; they do not eject a fixed member from the ZP on a subset of
   frames. They also favour using as many comps as possible with weights, not a tiny
   hand-picked set. The removed clip made a per-frame in/out decision on a star Broeg
   treats as a fixed member with a fixed weight -- that is not the cited method.

2. **Honeycutt 1992** (PASP 104, 435) is the place variable membership is legitimate:
   stars are genuinely **absent** from some exposures (heterogeneous archives), handled
   by a **global** least-squares solution over all stars and frames jointly. Varying
   membership plus a simple per-frame weighted mean (as though membership were fixed)
   appears nowhere in the literature and was the actual defect. A future Honeycutt-style
   global ensemble (parked with WIDE-ERR) would make controlled membership variation
   legitimate; that is a separate larger change, not this decision.

3. **Comparable tools:** AstroImageJ (Collins et al. 2017), C-Munipack/Muniwin, and
   Broeg-family pipelines use a fixed ensemble (or whole-series drop). VaST (Sokolovsky
   & Lebedev 2018) does robust per-frame calibration, but against **hundreds to
   thousands** of matched stars -- a different statistical regime. No tool in this set
   automatically ejects a comparison star from a **small** ensemble ZP per frame.

4. **Why N=5 decides it.** MAD asymptotic Gaussian efficiency ~37% (vs Sn ~58%, Qn
   ~82%; Akinshin arXiv:2209.12268; Rousseeuw & Croux 1993). At n=5 finite-sample
   efficiency is worse and MAD underestimates scale if the asymptotic 1.4826 factor is
   used (consistency factor nearer ~1.72) -- the rejection boundary was systematically
   **too tight** exactly where the clip activated. Literature does not ban robust
   rejection in general (VaST is right at large N); it bans this combination: small N,
   MAD scale, per-frame membership flip, then a fixed-membership weighted mean.

**Decision 2 (admission) -- not coupled.** Broeg-family + Astrokit (Burdanov et al.
arXiv:1408.0664) support using many comps with |dmag| within ~2 mag;
`phase01_comparison_max_mag_diff = 2.0` sits at that stated limit and need not revert
to 1.5 for scatter. Caveat: the matrix measured scatter only. Looser admission can
admit colour systematics (smooth airmass-correlated drift) invisible to scatter;
`comp_max_delta_bprp` 0.79->0.99 moved in the same generation. On BO CVn all five comps
had dBP-RP <= 0.15, so colour did not bite here -- field-specific, not a parameter proof.
Future colour-tolerance decisions need residual-vs-airmass / residual-vs-colour tests.

**Rule.** Per-frame zeropoint uses all admitted comps with Broeg 1/sigma^2 (+ tier) weights.
No rejection step. See INV-COMP-MEMBERSHIP.

**History note.** Commit `c9e1f8f` ("remove all science-path sigma-clip") did **not** remove
this clip. Token search missed it (no `sigma_clip` string). Future rejection sweeps must be
by behaviour, not by name.

**Open / not in this commit:** `detect_outliers` (4589-4602), plate-solver SIP pair clip
(`vyvar_platesolver.py:685-693`), I-04 unmatched-epoch drop, `phase01_comparison_max_mag_diff`.

Reports: `dev/results/CURSOR_RESULT_zp_clip_closeout.md`,
`dev/results/MEMO_ensemble_zp_clip_literature.md`.

---

## INFOLOG-AUTHORITY - durable session log over ring buffer (2026-07-29) - superseded by INFOLOG-ONE-2026-10

**Problem.** Two infolog files per run with the same nominal role: a durable session append log
(written from run start) and a late ring-buffer export (8000-line cap). Guard lines such as
`INV-PREP-01` appeared in the durable file but not in the export the UI auto-saved after RUN.

**Rule.** `write_run_infolog()` **finalizes the durable session file** when one is open: flush,
close, prepend `# authoritative: durable session log (complete operator record)` and the milestone
block. Ring-buffer-only saves (no session) are tagged `# partial: ring-buffer tail only`. The
Streamlit tab may still show the live ring buffer for tail viewing; disk artefact for operators is
the durable file.

---

## CATALOG-PROVENANCE - fingerprint catalog DBs in run meta (2026-07-29)

**Problem.** Anchor `plan_regen_fingerprint` depends on Gaia cone contents (15085 sources at
G<=17.5 on anchor night). Rebuilding the Gaia DB (e.g. GAIA-PM-COLUMNS) would fail the anchor
with an anonymous SHA mismatch.

**Rule.** Every photometry run records `provenance.catalog_databases` with path, size, mtime,
row count, and `fingerprint_sha256` via `sha256(size + first_1MiB + last_1MiB)` (full file hash
over ~53 GB impractical). Anchor `--full` gate `full-catalog-provenance` compares run vs snapshot;
on mismatch reports **input catalogue changed** with field-level deltas before photometry SHA fail.

---

## ANCHOR-GATE-BLIND-SPOT - `--full` cannot verify preprocess or detection (2026-07-30)

**Finding (Audit Stage 3 Part 0b).** The anchor byte-identity gate
(`session_baseline_check.py --full`, ledger VL-ANCHOR-WCSINV) starts from **frozen**
`detrended_aligned/lights`, `MASTERSTAR.fits`, and `masterstars_full_match.csv` copied from
the anchor snapshot. It calls `run_full_photometry_pipeline` only. It never runs calibration,
preprocess, alignment, MASTERSTAR stacking, or DAO detection/catalogue construction.

**Consequence.** Two active audit fixes survived every gate without verification on the science path:

| Change | Blind to `--full`? |
|--------|-------------------|
| P-10 preprocess sky-surface sign fix | **Yes** - preprocess does not run |
| DAO threshold / noise estimator (3.8, `sigma_pp`) | **Yes** - frozen catalogue, no DAO |

This is the same blind spot recorded in `CURSOR_RESULT_masterstar_count_diag.md`; Stage 0
incorrectly attributed a photometry SHA change to P-10 + threshold when neither executed in
the `--full` run.

**Rule.** Treat `--full` as verifying **photometry determinism and downstream science outputs**
given fixed aligned lights and MASTERSTAR inputs. Preprocess, alignment, stacking, and detection
changes require a **full-chain rebuild harness** from calibrated lights (Part 0b) before
anchor re-cut. Registry: `INV-ANCHOR-00` in `VYVAR_INVARIANTS.md`.

**Stages covered vs not covered** - see INV-ANCHOR-00 table in `VYVAR_INVARIANTS.md`.

---

## DAO-THRESHOLD-OPTION-B - convolved RMS threshold (2026-07-30, N pending)

**Architecture (Part 2 implemented).** DAO detection threshold uses option B:
`scale_threshold=False`, `threshold = N x rms_conv`, where `rms_conv` is the robust RMS of
the FIND-kernel convolved detection image. `_pixel_noise_sigma_pp_adu` / `_dao_noise_sigma_adu`
remain **diagnostic only** (not wired to threshold).

**Literature basis.** photutils `DAOStarFinder` (`scale_threshold=False`); Fruchter & Hook
(2002) resampling correlation; Casertano et al. (2000) AJ 120, 2747 App. A undersampled noise.

**Pending Milan decision.** `masterstar_dao_threshold_sigma` must be recalibrated: under option B
the number means **N sigma in the convolved image on correlated noise** (third recalibration;
intended final stable meaning). Sweep: `dev/results/CURSOR_RESULT_audit_stage3_part2.md`.

---

## RELEASE-TREE-HYGIENE - post-bundle compiled-artifact cleanup (2026-07-23)

**Problem.** WSL/MSVC Cython builds leave `.pyd`/`.so` beside sources (and occasionally
at repo root when relocation fails). These must never appear in `git status` or confuse
dev interpreted runs.

**Rule.** After every successful `build_bundle.py` run, **`run_clean()` runs automatically**
(removes `src_py/*.pyd|*.so`, repo-root droppings, `build/lib.*`, `build/_cython_out/`).
Opt out for debugging only: `build_bundle.py --no-post-clean`. Manual reset anytime:
`python build/setup_cython.py clean` (Windows and WSL).

**Gitignore.** `*.pyd`, `*.so`, `build/lib.*/`, root `/*.pyd`, `/*.so` are ignored; physical
delete is still required so interpreted dev does not shadow stale binaries.

---

## LATENT-NAMES-COMPILE-GATE - latent NameError sweep before Cython release (2026-07-21)

**Trigger.** Cython compile-as-static-analysis + pyflakes/ruff F821 gate ahead of
CYTHON-RELEASE closed-source bundle.

**Fixed (10 + EXC-0030 accessor).**
1. `_get_lc_psf_strict` deleted as collateral in `0b5eb8b` while consumers remained
   (`photometry_core:7419`, `method_lc_output._build_flux` lazy import) -- restored with
   `psf_ac_applied` guard (T2 never-silent AC rule).
2-9. Eight EXC-batch log templates referenced `exc` while handlers bound `e`, `_e`,
   `_ap_exc`, `_alt_exc`, `_meth_exc`, `_wcs_exc` (`photometry_core` x7, `pipeline` x1).
10. `calibration.py`: missing `import sqlite3` (latent NameError on equipment OSC path).
11. `comp_selection_per_target.py:173`: `sqlite3.Row.get` -- **root cause of recurring
    EXC-0030 noise** in `--full` logs (`'sqlite3.Row' object has no attribute 'get'`).

**Systemic guard.** ruff **F821** enabled repo-wide (`pyproject.toml`); `import *` shims
(`photometry.py`, `photometry_phase2a.py`) remain -- F821 blind in those two files only.

**Recurrence.** New guard: ruff F821 (dangling-name class, 2nd+ occurrence).

---

## CYTHON-RELEASE-2 - bundle + embedded runtime + install docs (2026-07-23)

**Decisions (Milan, R1-R6).** Release ships bundled Python 3.12 on Win (embeddable) and
Linux (python-build-standalone); catalogs never shipped; LICENSE reused; preview tags
`preview-YYYYMMDD` (pre-release); equipment via DB Explorer; install docs CZ+EN.

**Data-dir separation (B2).** `resolve_data_root()` in `config.py`: `VYVAR_DATA_DIR` env
override; git dev checkout keeps install root as data root (dev-neutral); bundled installs
default to `%LOCALAPPDATA%\\VYVAR` / `~/.local/share/vyvar`. First-launch bootstrap in
`vyvar_runtime.py`. **Requires --full anchor** after path-resolution change.

**Bundle tooling.** `dev/tools/cython_release/bundle/` (`build_bundle.py`, runtime pins,
launchers, `--selftest`, `smoke_bundle.py`). Public repo content staged under
`release/public_repo/` (Milan copies to `VYVAR-release`; Cursor never pushes public repo).

**Next.** Milan: WSL Linux bundle + GitHub pre-release upload per `docs/VYVAR_RELEASE_RUNBOOK.md`.

---


**Scope (S1-S4).** Release compilation is a packaging step only: `src_py/` stays
interpreted in git; `.pyd`/`.so` and `build/_cython_out/` are gitignored. Dev/CI
anchors run interpreted; compiled builds must reproduce the **same** anchor SHAs.

**MODULE_LIST rule (S2).** Single source: `dev/tools/cython_release/module_list.py`.
All `src_py/*.py` except `app.py`, `ui_*.py`, plus `EXPLICIT_EXCLUDE` (one-line
reason per entry). OSC modules (`osc_extract`, `osc_align`, `gaia_johnson`) included.
84 science modules compiled at RELEASE-1 (`b4c372a`, 2026-07-23); **85** after RELEASE-2
added `vyvar_runtime.py` for bundled first-launch bootstrap (`3369832`); **86** after
BUNDLE-FIELD-FIXES-2 added `run_preflight_log.py` for pre-infolog RUN error capture.

**Pinned flags (S3).** Plain compile only: `annotation_typing=False`,
`Options.docstrings=False`, `language_level=3`, `embedsignature=False`.
`build_release.py` refuses to run if these drift.

**Release gate (S4).** Compiled `--full` must match interpreted anchor byte-identity:
core `03d8fb64...` n=333, extended `bbfcc92e...` n=499, science compare 0 failures.
P1 golden compiled: 7/7 byte-identical (`VL-P1-GOLD`).

**Conditional-skip policy.** Tests using `inspect.getsource` or `.py` source scans on
compiled modules skip only when the module loads from `.pyd`/`.so`
(`dev/tests/cython_compat.py`). No ad-hoc test weakening.

**Tooling.** `dev/tools/cython_release/` (build, clean, latent sweep, smoke, MP verify);
`build/setup_cython.py` shim. Recurrence: `test_cython_annotation_typing.py` (full list
+ flag-drift guard).

**Next.** RELEASE-2: bundling/installer (mixed compiled science + interpreted UI).

---


**Finding.** Cython 3 pure-Python mode defaults `annotation_typing=True`, turning PEP-484
annotations into C type declarations. Plain compile of `comp_selection_per_target` without
the pin changed P1 science (core SHA `4ecbae9f...` vs VL-P1-GOLD `074ae881...`; up to
167/169 `phase2a_empty_comp_drop`).

**Decision.** All release compiles use `annotation_typing=False` in
`build/setup_cython.py` `COMPILER_DIRECTIVES`. PEP annotations remain metadata only.
Typed speedups later use explicit `cython.*` declarations in dedicated modules, never
via PEP-484 annotations on production paths.

**Verification.** With the pin: `comp_selection_per_target` alone and all three spike
modules (`photometry_core`, `comp_selection_per_target`, `photometry_phase2a`) compile;
P1 golden 7/7 byte-identical on Windows (MSVC 14.51, Cython 3.2.8).

**Recurrence test.** `dev/tests/test_cython_annotation_typing.py` pins the directive.

---

## DEPS-CYCLE-2 - photutils 3.0 + astropy 8.0 (+ numpy 2.4.4+) (2026-07-20)

**Pins.** `photutils>=3.0,<4` (installed 3.0.0); `astropy>=8.0,<9` (8.0.1);
`numpy>=2.4.4,<2.5` (validated on 2.4.4; 2.5.1 rejected 2026-07-21, see below).

**Migration (scout checklist).** Kwarg migration (`sharpness_range`, `roundness_range`,
`n_brightest`); explicit `min_separation=0` at every `DAOStarFinder` site to freeze 2.x
detection (#2216); column readers migrated to `x_centroid`/`y_centroid`; PSF module verified
(EPSFBuilder/ImagePSF/PSFPhotometry unchanged API).

**M2 freeze decision.** Adopting photutils 3.0's implicit 2.5xFWHM crowding filter is a
**future science arc** (dense-field program), never an upgrade side effect. All production
DAO sites pass `min_separation=0` deliberately.

**G3 outcome (path a).** `--full` vs VL-ANCHOR-WCSINV **byte-identical**: core
`03d8fb64...` n=333, extended `bbfcc92e...` n=499; science-compare 166/166 PASS.
No anchor or P1 re-cut. Runtime ~2025 s vs ~2296 s pre-upgrade (~12% faster).
See `dev/results/CURSOR_RESULT_deps_cycle2.md`.

**numpy 2.5.1 validation (2026-07-21).** Fresh-install path tested in
`tmp/deps_cycle2_venv` (numpy 2.5.1 + astropy 8.0.1 + photutils 3.0.0): `--full`
science-compare PASS (166/166) but photometry SHA core/extended **FAIL** (run
`842443c7...` / `9cabd82d...` vs anchor `03d8fb64...` / `bbfcc92e...`). Pin
tightened to `numpy>=2.4.4,<2.5`; 2.5.x is a future gated candidate, not blessed
for the compiled release bundle.

## ANCHOR-RECUT-SIGMA-NOTES - VL-ANCHOR-WCSINV re-cut after ASCII slope notes (2026-07-20)

**Origin.** PUBLICATION-PREP Part B `--full` STOP: 19 `comp_quality_*.json` files differed from
`draft_000435_snapshot_skysurface_20260716` only in slope-exclusion note text: ASCII `sigma`
(run) vs Unicode U+03C3 (snapshot). All 166 `lightcurve_*.csv` byte-identical;
`full-science-compare` PASS. Cause: ENCODING-POLICY commit `ecbae90` transliterated the
comp-stability note template in `photometry_core.py`; policy stands, not reverted.

**Gate-spec miss (plain).** The ENCODING-POLICY gate set did not require `--full` before push;
latent SHA drift shipped until Part B touched the shared Phase 0 path and ran `--full`.
**PROCESS fix:** string-literal changes in `src_py` science modules now require `--full` before
push (comments exempt).

**Coverage hole.** P1 golden mini (16 frames) never emits slope notes, so `VL-P1-GOLD` did not
catch the drift. **Test fix:** `dev/tests/test_recur_shatext_templates.py` pins comp_quality
note template strings.

**Re-cut.** Two independent fresh `--full` runs byte-identical; new anchor SHAs locked via
ledger refresh on `VL-ANCHOR-WCSINV` (superseded core `3d26f469...` / extended `6420f1da...`
retained in notes). See `dev/results/CURSOR_RESULT_anchor_recut.md`.

## EXPORT-HEADER-SLIM - methods matrix + slim AAVSO/VarAstro headers (2026-07-20)

**Decision.** AAVSO and VarAstro text exports drop unconditional citation blocks
(`[CORE]`, `[CATALOGS & TIME]`, `[SOFTWARE]`, `[FIELD ASTROPHYSICS]`). They keep
format/site/observer fields, a machine-generated `# METHODS MATRIX (this run):`
ON/OFF block, `# [METHODS - this run]` one-line citations only for methods that
are ON, and a pointer to the SUMMARY MEASURE REPORT (PDF). The full citation
block plus the same matrix remain in the PDF methods section.

**Single-source rule.** The matrix is built from the same `RunCitationContext`
flag->method mapping that gates conditional citations in `citations.py` (extend
that mapping when a gated method is missing -- one source, no drift).

**Citation corrections (CITATIONS.bib, verified vs FLOW ch 19 / ADS):**
- AstroImageJ: Collins, Kielkopf, Stassun & Hessman (2017) AJ 153, 77
  (bibcode 2017AJ....153...77C) -- was wrongly PASP 129, 144502 / Stelzer.
- Jordi et al. (2010): A&A 523, A48 (bibcode 2010A&A...523A..48J) -- was wrongly
  A&A 515, A16. FLOW ch 19 already had 523/A48.
- Emitting notes are plain-ASCII (strip LaTeX `\_`, `\Delta` / `\ensuremath\Delta`).

**OBSCODE warning:** warn ONLY when observer code is missing/empty in config.
Any explicitly set value (including UMIA) emits no warning.

## INVARIANTS-P3P4-CLOSEOUT - recurrence/forensic rubric + honest scope (2026-07-20)

**Delivered.** Invariants program P1-P4 closed.

- **P3:** PROCESS "Recurrence and forensic discipline" (recurrence rule + RESULT
  `Recurrence:` field; forensic PROMOTE/ARCHIVE/DELETE rubric; weekly
  `dev/tools/invariants_report.py` cadence). Pilot: `forensic_disc_ui_match2.py`
  -> **PROMOTE+ARCHIVE** (cat_match_arc=2.0 parity asserted in
  `test_ui_chain_byte_identity`; script header state ARCHIVED).
- **P4:** STATE pinned "Invariants program -- honest scope" (GUARANTEED vs NOT
  guaranteed; claim-verified). GAPS D2 void addendum (EXCEPT-BULK-2 already
  closed 2026-07-08).
- Result: `dev/results/CURSOR_RESULT_invariants_P3P4.md`.

## K2-NIGHT-FIT-V2-IMPLEMENTED - gated NIGHT_FIT fitter + synthetic recovery (2026-07-20)

**Scope finding.** The v2 fit path was never in production: `k2_extinction.py` had v1
literature machinery only; the four `k2_fit_*` keys existed since v1 for forward
compat. This ships v2 per `dev/results/specs/VYVAR_K2_DESIGN_SPEC.md` v1.1 S5/S6.
Remaining activation blocker is **only** the data night (GAPS B2 / K2-DATA-BLOCKER).
GAPS C1 (prove the fitter) is delivered by the synthetic suite against the real
production fitter.

**Adherence.** Model on Honeycutt residuals uses the S5-identifiable form
`k2*(C-Cref)*dX` (dX = X-mean(X) after frame+star CM removal). Pre-gate: monotonic
airmass refuse; detectability `sigma_k2_pred <= |k2_lit|/k2_fit_min_detectability`;
outer colour/brightness tertile + arc consistency within
`k2_fit_consistency_sigma * max(sigma_boot, 0.10.|k2|)`; plausibility ceiling +
literature factor/sign. Any failure -> LITERATURE_DEFAULT with
`k2_fit_refuse_reason`. Default `k2_fit_enabled=false` is byte-identical.

**Synthetic validation (summary).** Recovery sweep over k2in{0,0.02,0.05,0.08} x
spreadin{0.2,0.5,1.0} x noisein{5,15,30} mmag: accepted fits within
`max(2.sigma_boot, 0.005)`; non-detectable cells refuse with `detectability`. Explicit
REFUSE: monotonic X; absurd k2=0.5; tertile-split injection; zero-signal clean.
Draft-427 fixture: `fixture_source=synthesized_from_decisions` (sandbox JSON gone;
signature from SPEC S1/S6 + recon notes) fails items 3/4.

**Activation.** Still blocked on B2 data night. Do not flip `k2_fit_enabled` here.

---

## PER-FRAME-SAT-GATED - per-frame target saturation behind flag (2026-07-19)

**Context re-scope.** The original M67 evidence (76 Green / 49 Red whole-star drops)
came from an ASTROPHOTO dataset with long exposures - an extreme case, not
representative of photometric nights (Milan, 2026-07-19). Severity re-graded
**HIGH -> MED**. The principle stands: a star's usability is a per-frame fact
(`peak_max_adu` is already measured per frame), while today the TARGET-level drop
comes from the MASTERSTAR zone of one reference exposure (`skip_photometry` via
`zone_flag`). Failure mode both ways: a target saturated on 30% of frames is dropped
wholly (loses 70% good data); a borderline target clean on the master can carry
saturated rows. The M67 dataset is gone; **real validation is DEFERRED** to the
next dataset containing saturated bright stars (ROADMAP revisit trigger).

**Design shipped (default OFF).** New flags `per_frame_saturation_enabled=false`
and `per_frame_sat_min_clean_frac=0.5` (clamp [0.1, 1.0]). When OFF: byte-identical
to today's zone-based `skip_photometry` (comps keep their existing per-frame >10%
rule). When ON, for TARGETS only: master zone `saturated` / `likely_saturated` is
advisory; compute clean fraction from per-frame sat flags; >= threshold -> measure
(saturated rows keep flag `saturated`, mask-first); < threshold -> skip with
`skip_reason=per_frame_saturation`; missing per-frame peak data -> fail-safe fallback
to zone behavior + `per_frame_sat_fallback=true`. Provenance: `sat_clean_frac` on
`photometry_summary` (outside photometry SHA / science comparator file set).
INV-CFG-01 extended: flag OFF => none of the new markers appear.

---

## Dependency policy + CYCLE 1 in-range refresh (numpy 2.4.4) (2026-07-18)

**Context (DEPS-SCOUT / DEPS-POLICY).** Grounded in Claude's dependency-landscape
scout (`dev/results/DEPS_SCOUT.md`), Cursor-verified against tree `d437bcd`.

- **Policy adopted.** Upgrades are a gated ritual, not accidents. `docs/DEPS_POLICY.md`
  is the reference: pin majors in `requirements.txt`; a candidate cycle is fresh env
  -> `pytest` -> `session_baseline_check.py --full` (byte-identical anchor gate).
  Byte-identical = free upgrade + (pin move if needed) + DECISIONS entry; a diff is a
  documented finding (adopt-and-re-anchor, or hold and report upstream). Neither
  outcome is a failure. `--fast` gains an **informational** `deps-outdated` line
  (WARN/PASS/SKIP only, offline-tolerant, never blocks).
- **CYCLE 1 result = byte-identical PASS.** numpy 2.4.3 -> 2.4.4 (in-range, within
  the existing `numpy>=2.4,<3` pin, so no pin move). `pytest` 963 passed / 19 skipped;
  `--full` vs anchor 435 PASS at HEAD `30c803f`: `full-science-compare` n_lc=166
  failures=0, `full-photometry-sha-core` `3d26f4692ac81fc5...` (n=333), extended
  `6420f1daa53a0d5d...` (n=499), pipeline 2278 s. Ledger anchor items auto-stamped
  `commit=30c803f, last_verified=2026-07-18`. numpy 2.4.4 is validated for production.
- **Finding for the next cycle.** The scout's table listed numpy latest as 2.4.4, but
  the live index now offers **numpy 2.5.1** (a minor bump, still within `<3`). CYCLE 1
  was scoped to the 2.4.x patch, so 2.5.1 was deliberately **not** adopted here; it is
  a candidate for a future in-range cycle (minor bump -> still needs pytest + `--full`).
  astropy 8.0.1 and photutils 3.0.0 remain gated cross-major work (CYCLE 2, after the
  INSTALL/Lenovo arc).

---

## License, repo visibility & distribution model (2026-07-18)

**Milan decision (DOCS-FIX-ARC1).**

- **License = proprietary, all rights reserved.** VYVAR ships under a short
  proprietary `LICENSE` at the repo root: Copyright (c) 2026 Milan Uhlar; no use,
  copying, modification, or distribution without prior written permission; no
  warranty. The `docs/README_FULL.md` / `docs/README_CZ.md` license sections match this text.
  Rationale: VYVAR is not (yet) an open-source release; keep all rights with the
  author until a deliberate licensing decision is made.
- **Repository visibility = private** (already applied on the remote). The GitHub
  front-door README is written for eventual public reading, but the repo stays
  private for now.
- **Compiled-library distribution = deferred (recorded, not adopted).** The idea of
  shipping VYVAR as a compiled/obfuscated library instead of source was considered
  and explicitly parked; revisit only if/when a distribution need arises.

---

## WAVE-B-PARAM-REDUCTION - cut/merge/wire the 304-key surface to 269 (2026-07-18)

**Milan sign-off (post PARAM-BUDGET-AUDIT).** Acting on the audit dispositions in
`dev/results/param_budget_audit.csv`, the registered-parameter surface was reduced
from 304 to 269 entries (config.json persists 249). Executed as anchor-gated steps,
one commit each, full pytest green throughout, and closed by the `--full` anchor gate
vs `draft_435` (fallback removals touch the science path; byte-identical was required).

**What was cut and why.**
- **DELETE-DEAD (4):** `aperture_fwhm_factor_medium`, `masterstar_log_astroalign`,
  `phase01_comparison_proximity_tiebreak`, `phase01_comparison_rms_bin_mag` - no live
  readers; removed from AppConfig, registry, generated docs, config.json.
- **MERGE 14 -> 3:** eight `comp_tier{1..4}_bprp_limit/_weight` scalars -> one
  `comp_color_tiers` list-of-dicts; four `phase01_tier{1..4}_mag` -> one `phase01_tiers`
  list; `aperture_fwhm_factor_small/_large` -> one `aperture_snr_sizing` mapping. Loader
  accepts the OLD scalar keys for one transition release (deprecation log) and maps them
  into the structured key; save writes only the new form.
- **DELETE-DB-DUP (9):** `gain`, `read_noise`, `plate_scale_arcsec_per_px`,
  `phase01_plate_scale_arcsec_per_px`, `export_arcsec_per_px`, and the observer mirrors
  (`observer_lat/lon/alt_m/location_name`) removed from config.json persistence in both
  directions. The AppConfig fields remain as run-time hydrated mirrors; the DB/FITS
  resolver is now the ONLY authoritative source. The vestigial DB `SETTINGS` table (no
  production readers) is dropped via idempotent migration; dead `get_setting_int` /
  `set_setting` removed. Masterdark/flat validity days confirmed config-authoritative.
- **HARDCODE (20):** blind/plate-solve solver internals that were never tuned in practice
  moved to module-level constants in `vyvar_blind_solver.py`, `vyvar_platesolver.py`,
  `pipeline.py`; removed from AppConfig, registry, config.json, generated docs.

**WIRE-IN finding (bug fix).** `calibration_master_ccd_temp_tolerance_c` was registered
but never passed to `find_best_calibration_library_path`; the dark-selection tolerance was
silently hardcoded. Now wired from `cfg` at both importer call sites. No-op for current
runs (key absent from Milan's config.json -> effective value stays 0.5).

**INTERNALIZE (2).** `frame_width_px` / `frame_height_px` stay as AppConfig fields (they
still resolve from FITS NAXIS at run time) but leave the USER parameter space:
owner=internal, widget=hidden, tier=expert, and dropped from config.json persistence.

**Focal-length precedence.** The two resolution paths were audited and already resolve
DB-optics-first with FITS-header fallback (`param_resolver.resolve_focal_mm`); no code
change was needed - the audit note was stale.

**The 80-key "never-touched expert" hardcode pool: REJECTED - stays KEEP.** Universality
argument: those keys are legitimate scientific knobs that apply across all sites/targets;
the fact that Milan has not re-tuned them from defaults is evidence they are well-chosen
defaults, not evidence they are dead. They remain user-visible expert parameters.

**sips_dao_fwhm clarification.** The audit's bonus `sips_dao_fwhm` claim is RETRACTED: the
code uses the registered `sips_dao_fwhm_px` everywhere (config.py, pipeline.py,
photometry_core.py, vyvar_platesolver.py, night_run.py, app.py) - there is no bare
`sips_dao_fwhm` key.

---

## COMP-TRUST-MIN-COMPS - 3 vs code default 5 is INTENTIONAL (2026-07-17)

`comp_trust_min_comps=3` (config.json) vs code default 5: INTENTIONAL. Since 1c80219
(2026-06-16, Phase-1 graceful comp degradation) the key is the GREEN trust threshold, not
a hard RED floor; 1-4 good comps yield YELLOW with sigma scaling by N (see
VYVAR_COMP_DEGRADATION_SPEC.md). GREEN at 3 vs spec example 5 validated by matrix 164157.
Config page in SUMMARY MEASURE REPORT will permanently list this as a deviation from code
defaults - that is correct behavior.

---

## INVARIANTS-P1-GOLDEN-MINI (2026-07-19)

**Mini source.** In-Archive `draft_000435` (live draft). Fail-early requires
calibrated lights + detrended_aligned proc products + platesolve
MASTERSTAR/catalogs. Raw lights exist (150) but in-draft `Raw/darks` and
`Raw/flats` are empty (masters come from CalibrationLibrary at import time).

**Scope note.** Mini starts at **photometry-ready** stage: 16 calibrated + matching
`proc_*.fits/csv` + parent platesolve catalogs/MASTERSTAR. Chain coverage matches
`session_baseline_check --full` (`run_full_photometry_pipeline`). Calibrate -> QC ->
align are not re-run on the mini (would require local masters + QC DB state); the
UI-order test still exercises UI path discovery (`_find_phase2a_paths`) vs headless
direct paths for the photometry composition. Dual-entry divergence remains an
F-431-class STOP (report, do not silent-fix).

**Size / selection.** 16 frames, even DATE-OBS stride across the time-sorted list of
frames that have both calibrated and aligned products (QC-rejected frames excluded),
first frame included. Same masters/catalogs/config as parent night.

**Expected at n=16.** Variability detection no-ops (`variability_min_frames=30`);
`lc_quality` may classify short-baseline - both correct, not failures.

**Reproducibility before lock.** Build + full headless run twice from scratch;
science outputs byte-identical; only then register `VL-P1-GOLD`. Confirmed
2026-07-19 (run1 611.6 s, run2 574.7 s; core/ext SHAs identical).

---

## VYVAR-INVARIANTS - machine-enforced contracts (2026-07-16)

**Premise (Milan sign-off by commissioning).** Static audits cannot catch integration-class
defects (ordering, path assumptions, transform relationships, cross-entry divergence,
unprovenanced code). Every defect found in the F-428/F-431 arc was caught by a
**measurement**, not by reading code. This program converts VYVAR's physics/math contracts
into machine-enforced invariants.

**Goal (honest):** not "zero bugs" (unattainable); rather (a) no defect reaches science output
silently, and (b) no defect class recurs once fixed. After P1-P3 the guarantee is: silent-wrong
science requires simultaneously fooling independent guards (flux conservation, WCS identity,
census bands, SHA determinism, golden UI<->night_run equivalence); every ledger defect class has
a recurrence guard. Audits continue for dead code / style / docs / obvious logic only.

**Timing.** Starts **after Anchor #3 closes**. Phases land as separate commits; pytest green
each. Order: P1 golden E2E equivalence -> P2 contract registry / runtime gates -> P3 PROCESS
recurrence discipline -> P4 STATE scope statement.

**Out of scope:** UI Settings block (parallel), new science features.

### INVARIANTS-P2-REGISTRY (2026-07-19)

**Delivered.** Human registry `docs/VYVAR_INVARIANTS.md` + runtime library
`src_py/invariants_runtime.py`. Wired gates (check-only): INV-FLUX-01/02 (FAIL),
INV-FLAT-01 (WARN), INV-WCS-01 (WARN), INV-DAG-01 (FAIL; cold-start / gap allowed for
sparse stamps), INV-RNG-01 (TEST AST; zero hits), INV-PROV-01 + INV-CFG-01 (FAIL end-of-run).
INV-FLUX-03 / INV-SHA-01 / INV-WCS-00 are registry-only pointers.

**Meta / SHA scope.** `pipeline_meta.json` is outside core/extended photometry SHA and the
`--full` science comparator (`dev/tests/photometry_sha.py`). Invariants/stages blocks are
written into `pipeline_meta.json` freely; science outputs remain the byte-identity contract.

**FLOW 4.5 correction (doc-drift evidence).** Builder prose claimed sky-surface preprocess was
"flux-conserving / mean-preserving". Code and T3 decision subtract the **full** fitted surface
including the constant term (pedestal; Delta median ~ -96 ADU on the reference frame). Registry
INV-FLUX-03 now owns that claim; FLOW regenerated accordingly. This is the class of
prose-physics drift the facts guard cannot catch.

---

## T3-PREPROCESS-SKY-SURFACE - order-2 shared preprocess (2026-07-16)

**Milan sign-off (commissioning T3-RESTORE).** Per calibrated light frame, fit a 2D polynomial
surface of order 2 to the background (source-masked + sigma-clipped fit), subtract it in the
shared ``calibrated -> processed`` step used by BOTH chains (UI and headless route through
``preprocess_calibrated_to_processed`` -> ``_preprocess_calibrated_one``). Config:
``preprocess_sky_surface_order`` (int, 0 = off, **default 2**). Subtract the **full** fitted
surface including the constant term (pedestal convention: Delta median ~ -96 ADU on BO CVn
Light_008 vs draft_429).

**Rationale.** DAOStarFinder and Labbe background assume a locally flat field. The lost
429-era preprocess step removed large-scale gradient; without it the deterministic repo path
inflates pass-1 DAO (~8927) and flips automatic ``DENSITY_OVERRIDES``. Restoring the surface in
shared preprocess targets **429-class** census/quality without per-chain drift.

**Provenance.** Per-frame ``VYSKYORD`` / ``VYSKYP2P`` FITS headers; preprocess row stats in QC
dataframe.

**Dirty gate (paired).** ``git_dirty_code`` (import-relevant ``*.py`` only) gates anchor /
FAIL-CLOSED; ``git_dirty`` + scratch lists remain for transparency.

**Empirical gap (honest).** Cal-only surface fit vs draft_429 oracle MS: DAO pass-1 sim **2579**
vs logged **2816** (~8%); smooth-field residual p99 **173 ADU** vs oracle **52 ADU**. Frame
median shift differs from meta ``sky_adu`` (~1478 Labbe annulus). Full MS residual unavailable
at preprocess time; Milan UI run is acceptance gate.

---

## F-431-HEADLESS-DIVERGENCE - CLOSED (2026-07-16 / T3)

**Root cause:** lost unprovenanced order-2 preprocess ADU step (``cal==proc`` on deterministic
path). **Resolution:** shared ``_fit_subtract_preprocess_sky_surface`` in
``preprocess_calibrated_to_processed``. See `CURSOR_RESULT_t3_restore_anchor.md`.

**Recurrence (2026-07-22 / SKIPPROC, second occurrence):** ``013cb0c`` retired the
``processed/lights`` copy tree and removed the in-place order-2 sky-surface subtract from the
mono calibration path. Same defect class as F-431; see **SKY-SURFACE-RESTORE** (2026-07-27).
First closure shipped **without a regression guard**, so the step could return silently until
**INV-PREP-01** (preprocess gradient / large-small ratio on calibrated frames) made a third
occurrence detectable from the saved infolog.

---

## F-428-PASS2-CONTAMINATION - draft_428 census products unreliable (2026-07-16)

**Status (amended 2026-07-16 / F-431):** **DOWNGRADED** as the *primary census driver*.
Draft_433 (`715391b`, FIX 2 / WCS-INV active, clean tree, UI-parity match sep) still reproduces the
**6699-class** inflated census. Primary driver of pass-1 inflation is therefore the **absent
preprocess ADU mutation** that distinguishes 429 (`cal!=proc` Light_008 / MASTERSTAR) from the
deterministic repo path (`cal==proc`). Keep both facts below.

**Original finding (still true, secondary for census).** On draft_428, Gaia-targeted DAO pass-2
ran through a corrupt forward SIP (~12 arcsec / ~5 px bookkeeping offset per v5). Evidence retained:
428 `n_raw_dao=8927` vs 429 `2816`; unmatched 2724 vs 179; 1172 matched catalog_ids only in 428
(median mag 15.1, median sep-to-Gaia 17 arcsec). **Per-target photometry of well-detected stars is
unaffected** (v5 flux/identity correct). The WCS inversion asymmetry on 428 was a **real, fixed**
defect (`F-428-WCS-INV`); it is **not** sufficient to explain the 8927->2816 pass-1 collapse once
FIX 2 is live (see F-431-HEADLESS-DIVERGENCE / `CURSOR_RESULT_headless_forensics.md`).

**Density override coupling.** Inflated matched census still flips automatic `DENSITY_OVERRIDES`
(428/431: `annulus_inner_fwhm=5.75`, tighter comp gates vs 429 defaults). Monitor
`config_snapshot` near the density threshold.

**429 provenance.** draft_429 remains the **census QUALITY target** (3054 / 2875), but is an
**unprovenanced anomaly** (`git_dirty=true`, dirty file list never recorded). Deterministic repo
state today is the **6699-class**. Do not treat 429 as a golden clean-SHA run until the lost ADU
transform is restored in committed shared preprocess.

**Anchor implication.** New anchor requires healthy-census tree under protocol v2 +
`git_dirty=false` hard gate. Anchor arc **BLOCKED** until F-431 root closure.

---

## F-428-WCS-INV - MASTERSTAR WCS invertibility gate + coordinate finalization (2026-07-16)

**Decision.** After F-428 COORD v5 (`RECLASSIFY-PROJECTION`), encode three durable guards:

1. **Round-trip invertibility gate** (`wcs_invertibility.py`): after every plate solve (WARN +
   provenance flag) and after every optimizer SIP refit (FAIL-CLOSED - keep previous WCS). Metric:
   p99\|pix - world2pix(pix2world(pix))\| on a 9x9 grid; threshold 0.2 px. Persist
   `wcs_roundtrip_p99_px` / `wcs_roundtrip_pass` in `pipeline_meta.json`.

2. **SIP inverse regeneration** (FIX 2): forward SIP fit (`A/B`) must ship with fitted `AP/BP` in
   the same step (`ensure_sip_inverse_coefficients`). Optimizer SIP refit pairs DAO `(x,y)` with
   **Gaia catalog sky** (not stale `ra_deg`/`dec_deg` on the row) - root cause of v4/v5 bookkeeping
   offset while `world2pix(Gaia)` agreed with pixels.

3. **Coordinate finalization** (`finalize_masterstar_sky_coords`): matched rows -> Gaia catalog
   `ra_deg`/`dec_deg` + `coord_source=gaia_catalog`; unmatched -> `final_wcs` pix2world. Post-match
   identity sep gate in pixel space (WARN > 1.5xFWHM, drop assignment > 3xFWHM).

**Why.** v5 showed `world2pix(Gaia[cid])` within ~1.3 px of ms `x/y` while `pix2world(x,y)` was
~12 arcsec from Gaia - science flux/identity correct, sky columns wrong. Internal round-trip can still
pass (astropy numerical world2pix); the identity gate + finalization close the bookkeeping chain.
Angle histogram concentration (97/160 in two bins) is SIP-field distortion pattern, not isotropic
confusion. Photometry/LC science columns unchanged by this batch.

**Radius / field DB.** F-428-A3-RADIUS remains OPEN (deeper/wider field DB; GAIA-DR4 ~Dec 2026).

---

## Second-order extinction k'' - band-aware v1 (2026-07-07)

**Decision.** Ship k'' as a **deterministic, provenance-tagged correction** (not a per-night free
parameter in v1). Source hierarchy: `night_fit` (v2 only, pre-gate) -> `literature_default` -> `none`.

**Q1 - default `k2_mode=literature`.** Population-mean Smith/Henden coefficients; explicit
`k2_source`/`k2_value` on every LC row; independent of `apply_color_term` (CT default remains off).

**Q2 - v1 scope:** `band_classify` CT wiring + CV/CR flip + literature k'' path. NIGHT_FIT machinery
stays in sandbox; `k2_fit_enabled=OFF` until a feasible dataset exists (K2-DATA-BLOCKER).

**Q3 - draft_425 validation** under snapshot discipline (manifest + raw-light checksum sample).

**OSC tri-colour (BLUE/GREEN/RED, TG/TB/TR):** `STANDARD_FILTER` for CT eligibility only; **k2=none**
- no citable k'' for Bayer RGB bandpasses (`k2_none_tokens` second check in `k2_extinction.py`).

**Mandatory ordering.** k'' on comp instrumental mags **before** `fit_color_term_c1`; k'' on target
`mag_calib` **before** `apply_color_term` - otherwise fitted c1 absorbs k''xX_bar (CT bias).

**Missing data (fail-soft).** No finite Gaia BP-RP -> no k'' (`k2_source=none`); no finite airmass ->
skip that frame; never invent default colour.

**Converter.** `k2_bprp = k''_native x d(C_native)/d(C_bprp)` (Jordi et al. 2010 slopes in
`k2_extinction.py`; B-band uses cited `d(B-V)/d(BP-RP)`).

Spec: `docs/VYVAR_K2_DESIGN_SPEC.md` v1.1.

---

## Colour: raw Gaia BP-RP only; B-V / Riello citation removed (2026-06-25)

**Decision.** Johnson B-V is **deprecated** for tiering and reporting. Colour comes from raw
Gaia `bp_rp` on matched stars; no BP-RP -> B-V transform is implemented or desired. The PDF
report and citation emitter no longer cite Riello 2021 as a B-V transform. The legacy `b_v`
column plumbing remains (mostly NaN) - removal is a separate bounded task.

**Err model (F-HOWELL-3, Stage C FIXED 2026-06-25).** Proc export writes explicit
`sky_adu_per_px_annulus` (annulus median ADU/px) alongside legacy `noise_floor_adu` (detection
floor on MASTERSTAR). `_photometric_error` prefers the annulus column; falls back to
`noise_floor_adu` on older proc CSVs. Verified on draft_424: canonical LC science byte-identical;
sky-dominated faint targets show ~12-14% err inflation if detection floor were used instead of
annulus. Edge case (`photometry_mode=epsf` without ePSF): aperture enhance skipped - rare;
structural insurance via explicit column + fallback.

**Time provenance (F-BJD-1, Stage D FIXED 2026-06-25).** `_recompute_bjd_hjd_with_status` returns
`(bjd, hjd, time_base)` where `time_base` is `BJD_TDB` on astropy success or `JD_FALLBACK` on
the three documented fallback paths (invalid coords; observer 0/0; astropy batch failure). The
2-tuple wrapper preserves sandbox/test callers. Production Phase 2A writes constant per-frame
`time_base` into `lightcurve_*.csv`; numeric time columns unchanged. `compare_photometry_science_meaningful`
excludes `time_base` from science diffs.

---

**Decision.** VYVAR measures variable targets only when they have a **masterstar (DAO+Gaia) cross-match**.
VSX entries without such a match are **excluded upstream** in Faza 0 (`select_active_targets`), not
measured via forced aperture at VSX catalog coordinates. The former `catalog_only` / `forced_aperture`
Phase 2A branch, `lc_source` provenance column, and `catalog_only_n_comps` config are **removed**.

**Variable-target flux rule (strict).** Active variable targets are measured **only on a direct DAO
`catalog_id` hit** in the per-frame proc CSV (`read_flux_from_csv`). If the target `catalog_id` is
missing from the frame catalog -> **nondetection / NaN** (`no_data`). **No XY fallback** for variable
targets (prevents wrong-neighbor grabs on intermittent DAO frames). Comp stars retain the legacy
`_lookup_star_in_csv` path with the bright-neighbor reject (`fallback_mag > -8.0`).

**Removed knobs.** `phase2a_variable_xy_fallback_mag_tol`, `_phase2a_variable_xy_fallback_expected_mag`.

**Shared helper preserved.** `_catalog_only_fixed_aperture_flux` was **renamed** to
`_annulus_sky_subtracted_flux` (body unchanged) - it remains the annulus sky-subtraction primitive for
DAO PSF local-sky (`psf_photometry.py`) and neighbor-sub residual aperture (`psf_neighbor_sub.py`).

**Unchanged.** Saturated targets still get `skip_photometry=True` / zero-frame skip in Phase 2A.

**Supersedes.** G2-F001 forced-aperture routing (`f42ce89`) and G2-F002 catalog_only WCS placement
(`0dd59d7`) - both paths deleted rather than maintained in parallel.

**Validation (strip-FORCED + Phase-2A vs draft 419, Jirny comp lists).** `mag_inst` byte-identical
**360/360** B stable pool; **346/350** R (4 frames = intermittent-DAO NaN gaps, intended). `mag_calib`
byte-identical **357/360** B and **344/350** R. One B target (`458415401545371264`) shows a **uniform
~30 mmag zeropoint shift** on all 12 frames (`mag_inst` unchanged): comp_qa correctly **excludes**
intermittent forced-only comp `458412790204894208` (10/12 DAO, 2 frames FORCED_APERTURE-only in 419) ->
ensemble 8->7 comps. **Accepted** as intended DAO-only effect (variability shape preserved; `lc_rms`
slightly improved). R outlier `458470858164631936` frame 0041: wrong-neighbor XY closed -> NaN gap.

**Future backlog (not this fix).** Comp selection should exclude intermittently-DAO comps upfront
(e.g. `458412790204894208`, listed `comp_n_frames=12` but DAO-only 10/12) so comp_qa need not drop
them post-hoc.

---

## Photometry: canonical published magnitude `mag_calib_final` (Path A, 2026-06-22)

**Decision.** One canonical calibrated magnitude for all **publication-facing** outputs:
`mag_calib_final = mag_calib + ct_correction (if ct_ok) + delta_m_corr (if ac_ok)`, stored in
each `lightcurve_*.csv` alongside provenance columns (`mag_calib`, `mag_calib_ct`,
`mag_calib_ac`, `mag_calib_raw`).

**Consumers.** AAVSO export, VarAstro MAG column, main per-star PDF LC plots, candidate PDF LC
figures - all read `mag_calib_final` (`export_reports.py`, `photometry_report.py`).
VarAstro `delta_mag` remains the ensemble differential (not CT/AC adjusted).

**Not changed.** `lc_rms`, `lc_rms_ooe`, comp_qa, trust gate continue to use `mag_calib`
(CT/AC are per-target/night constants -> scatter invariant). Variability detection uses
instrumental `dao_flux` from proc CSVs.

**Supersedes.** Split consumer model (export AC-only vs PDF CT-only) flagged as G5-F011;
G5-F003 candidate-figure AC precedence is subsumed by this column.

**Implementation.** `be3e193`; full data-flow doc: `docs/VYVAR_CALIBRATION.md`.

---

## Fix C diagnosed (C1): run-414 bad frames are PSF/FWHM-bloated, not mis-aligned; recovery N/A (2026-06-18)

**Finding (measured, run-414 g, production alignment path; `CURSOR_RESULT_fixC_diag.md`).** The 14
late-night frames Fix B drops are **PSF-degraded**, not "good data that only failed alignment". bad-14
median **FWHM 8.60 px = 1.85x the good baseline 4.64 px**; concentration flux_large/flux **13.1 vs 1.65**
(8x worse); **corr(FWHM, alignment-residual) = 0.95** across 161 frames. The bloated-donut centroid noise
(~2.4 px) is the single root: it breaks astroalign's asterism matching (the **misalignment is a symptom**)
and it is exactly what B.2 (concentration) and Fix-B (alignment residual) measure - two downstream
symptoms of one cause. Most likely cause: **late-night focus drift on the deliberately-defocused rig**
(a pure transparency/flux drop would not bloat FWHM); the post-flip half may not have been refocused
(observer question).

**Decision - Fix C "alignment recovery" is NOT APPLICABLE; close it.** The frames are **not recoverable
to sub-px**: residual geometry on the matched subset is incoherent ~2.4 px scatter (rotation ~0 deg, scale
~1.0, |t| ~0.1 px; a similarity fit does not reduce it; no radial trend) + ~50 % source loss - centroid
noise, not a fixable transform. Their centroid floor (~2.4 px) is **above the 1.37 px gate**, so even a
perfect alignment would still be flagged. Candidate tests confirm it: control-point cap 50 -> 3/14, cap 80
-> 2/14, cap+isolation -> 1/14; per-frame WCS is absent (0/162) so WCS-reproject is unavailable;
translation-refinement is inapplicable to incoherent scatter. Force-aligning bloated PSFs would not yield
science-grade photometry and would risk the 147 working frames / all rigs. **Diagnose-before-fix
prevented shipping a useless, risky alignment change.**

**Decision - Fix B + B.2 are the correct, PERMANENT handling.** They reject genuinely unusable frames;
there is nothing to "recover", so the residual gate is a **permanent quality gate**, not a stop-gap that
self-deactivates once "Fix C" succeeds. (The earlier "self-deactivating" wording is superseded below.)

**Decision - log a SEPARATE perf/robustness ticket (not Fix C recovery).** astroalign at the production
mcp~200 on dense fields is ~654 s/frame (and still fails); cap to ~50 (astroalign's design point) ->
~3-10 s. Two integration shapes - additive recovery rung (keeps the 147 byte-identical, does **not** fix
slowness) vs primary cap (fixes slowness, changes the 147 transforms -> needs a **cross-rig regression,
home + narrow rigs**, before adoption). Deferred until cross-rig data is available. See ROADMAP.

**Watch-item.** Mildly-bloated near-threshold frames kept by the gate (e.g. g_0231, FWHM 5.12 = 1.10x):
differential photometry largely cancels common-mode FWHM changes, so likely benign - watch the LC near
the good->bad transition.

---

## Fix B: reject-on-alignment-residual frame gate is rig-agnostic + cause-correct; default-OFF (2026-06-18)

> **[Updated by C1, 2026-06-18.]** The "Problem"/"always-on"/"Self-deactivating" clauses below assumed
> the frames were *recoverable* once "Fix C" fixed alignment. **C1 refuted that** - the frames are
> PSF/FWHM-bloated (FWHM 1.85x; corr(FWHM,residual)=0.95) and **unrecoverable** (centroid floor ~2.4 px >
> 1.37 px gate). Fix B is therefore a **permanent quality gate**, and the alignment residual is a
> *symptom* of PSF bloat (alongside B.2), not a signal awaiting a future alignment fix. See the C1
> decision above.

**Problem.** run-414 D-A/D-B: the catastrophic V0454 LC outliers are 13 phase_correlation frames
mis-aligned ~2.1 px (translation-only fallback after astroalign failed on the dense field). The frames'
photometry data is fine - only the alignment failed. ~~They are recoverable once alignment is fixed
(Fix C), but until then they must not reach photometry.~~ **[C1: not recoverable - PSF/FWHM bloat;
permanent gate. See C1 decision above.]**

**Decision 1 - record a per-frame alignment residual as always-on QC.** Compute, at the Phase-2A
frame-selection point, a per-frame residual = median (over bright matched sources, 10<=mag<=13) of the
deviation of (x,y) from each source's robust across-night median position, and write it as
`align_residual_px` in `alignment_report.csv`. *Why this metric:* the across-night median reference is
dominated by the well-aligned majority, so a translation-mis-aligned frame's residual ~ its full shift
- it reproduces the diagnostic's clean separation (astroalign ~0.36 px vs phase_corr ~2.13 px) without
needing the alignment transform's internal control points. *Why always-on:* it is additive QC metadata
(does not feed photometry -> baseline byte-identical) and is a direct measure of frame quality.
**[C1: it is a *symptom* of PSF/FWHM bloat (corr=0.95), not a signal a future "Fix C" will lower.]**

**Decision 2 - gate threshold is RELATIVE (rig-agnostic), not a fixed pixel value.** Reject if
residual > `frame_align_residual_max_frac x science-aperture-radius-px`. *Why relative:* a fixed px
threshold would mis-generalize across rigs/plate-scales/focus; expressing it as a fraction of the
aperture radius ties it to where flux physically leaves the aperture. *Default 0.25:* on run-414 the
good/bad gap (1.206->1.450 px) is 0.22-0.27 x the 5.47 px aperture radius, and physically residual
>=~0.2x aperture-radius is where the defocused-donut flux starts leaving the science aperture; 0.25
(->1.37 px) sits in the gap. Safety floor (`min_keep_frames`, default 10) makes the gate a no-op rather
than nuke a marginal night.

**Decision 3 - default-OFF; method-agnostic; keep distinct from B.2.** Default-OFF => byte-identical
(verified: 70 targets 0 diff, V0454 max|diff|=0). ON drops 14 frames = all 13 phase_correlation **+ 1
mis-aligned astroalign** (dr=1.648, itself an LC outlier): the gate keys on *measured residual*, not on
the alignment method, so it correctly catches a bad astroalign frame and would spare a well-aligned
phase_correlation frame - this is desired, not a defect. **Relationship to B.2 (kept separate):** the
residual gate (alignment quality) is the cause-correct signal; B.2 (flux_large/flux concentration) is
the aperture-integrity symptom and also catches genuine transparency collapse the residual gate would
miss. On run-414 the residual gate is a strict superset of B.2 (overlap 13, residual-only the 1
astroalign, B.2-only 0). Consolidation is deferred (a future question), not done here. ~~**Self-
deactivating:** once Fix C makes astroalign succeed on these frames their residual drops below
threshold and the guard stops rejecting them - a safety net, not a permanent exclusion.~~ **[Corrected
by C1, 2026-06-18: the gate is PERMANENT - the frames are PSF/FWHM-bloated (centroid floor ~2.4 px >
1.37 px threshold) and unrecoverable; there is no "Fix C" that lowers their residual. See C1 above.]**

---

## Per-point `err` = photon (+) ensemble-ZP residual SEM (NOT std of comp instrumental mags) (2026-06-18)

**Bug.** The LC `err` term-3 was `ensemble_scatter/sqrtn_ens` with `ensemble_scatter = np.std(comp_vals)`
on the comparison stars' **instrumental** magnitudes (`photometry_core.py:2552,:2567`). The std of
comps' instrumental mags is dominated by their **brightness spread** - a fixed ensemble-composition
number, ~constant frame-to-frame - not by per-point uncertainty. For V0454's 2 comps (instrumental Delta
1.655 mag) it injected a constant **0.585 mag** onto every point, ~23x the empirical 0.025 mag
differential scatter. The LC centres were always correct (flux-sum zeropoint `ens_med`); only the error
bars were wrong. This is a distinct error-propagation bug that **shares the sparse/colour-mismatched-comp
structural root** with trust-RED (a thin, brightness-spread ensemble amplifies it) but is not the same
issue.

**Decision (bug fix; corrected formula is the default, no flag).** Term-3 is now the per-frame
**ensemble zeropoint standard error** from the comps' residuals: for each comp, residual = its
instrumental mag minus its **own across-night median** (`comp_ref_map`), so the comps' brightness and
constant colour offsets cancel and only the genuine per-frame zeropoint scatter remains; the per-point
contribution is `std(residuals, ddof=1)/sqrtn` (Honeycutt 1992 PASP 104:435, `honeycutt1992`). The former
`comp_rms_med/sqrtn_ens` term-2 is **dropped** to avoid double-counting the same ensemble-ZP quantity. The
photon/SNR base term-1 is **kept** (correctly large/NaN when a mis-centred aperture tanks SNR - that is
Fix B's domain). Small-n robustness: a near-zero residual SEM leaves `err` = photon base (the floor).

**Consumers (Step-1 audit).** `err` does **not** feed the trust verdict, `lc_rms` (empirical
`np.std(mags)`), the production Broeg ensemble combine (`1/comp_rms^2` weights), or production
sigma-clipping - so trust and production weights are unaffected. It feeds AAVSO/VarAstro export MAGERR +
the PDF median-error (intended; now correct) and **SysRem's inverse-variance weights**
(`run_sysrem_field`, `W=1/err^2`, Tamuz 2005) - `sysrem_enabled` is **default-OFF**, and the fix
*improves* its weighting (mis-aligned frames down-weighted instead of the old ~uniform weighting). No
SysRem code change.

**Verification (run-414 g, re-run vs committed artifacts).** `mag_calib`/`delta_mag` **byte-identical**
(max|diff|=0); V0454 `err` 0.581->0.013 vs empirical 0.025 (mis-cal 23.5x->0.5x); err now tracks
brightness (corr(err,mag) +0.75), no fixed baseline; the 13 mis-aligned frames still flagged via the
photon term. Evidence: `CURSOR_RESULT_414_diag.md`, `tmp/fixA_verify.py`.

---

## V0454 CrA amplitude = real eclipse egress + a position-dependent meridian-flip step (2026-06-17)

**Finding (diagnose-only, no code change).** V0454 CrA's ~0.45 mag gate-ON rise on draft_413 g
decomposes into a **real eclipse egress that dominates ~4:1** plus a **secondary ~+0.1 mag
post-fainter meridian-flip step**. The egress and the flip coincide in time (egress is pre-flip,
plateau is post-flip) but are cleanly separated by their comp-dependence:

- **Egress is real (comp-invariant).** Differential curves vs comps spanning 141-1840 px overlay in
  the pre-flip branch: pre-flip rise mean -0.288, **std 0.088 mag**; ~0.369 mag of the rise occurs
  *within the single pre-flip orientation* (no flip involved); SIPS independently reproduces the same
  egress->plateau on the same aligned frames.
- **Flip step is instrumental (comp/position-dependent).** Post-flip offsets fan out with comp
  position: near-centre comps -0.25, mid/far -0.38...-0.53 (**std 0.174 mag** across comps). Check stars
  near V0454 show a cross-flip step **median +0.100 mag (post-fainter)**, growing with separation
  (~0 co-located -> +0.2 at ~450 px). Boundary continuity (D3) shows the post-flip plateau sits +0.144
  mag below the extrapolated egress (part natural turnover, part step).

**Consequence.** The VYVAR-0.450 vs SIPS-0.548 amplitude gap (0.098 mag) is a **comp-choice / flip-step
artifact, not a pixel-data disagreement** - SIPS used the identical aligned frames; its comp simply
carried a flip step opposite to VYVAR's near-centre ensemble (measuring V0454 against far comps yields
~0.55 mag, SIPS-like). Root cause = **uncorrected flat-field under the 180 deg p->-p flip mapping**,
exacerbated by non-cal data. This grounds the ROADMAP **flip-aware comp selection** candidate (prefer
near-target comps / per-side normalization when a flip is detected). Evidence: `tmp/v0454_flip_diag.md`
(throwaway), `docs/round2_figs/v0454_flip_diag.png`, JOURNAL 2026-06-17 end-of-day.

---

## Aperture-skirt fix REJECTED; transparency frame-quality gate ADOPTED (default-OFF) (2026-06-17)

**B.1 aperture-skirt - not implemented.** The SNR-optimal science aperture (~5 px) captures only
EE~0.65 of a defocused star (curve-of-growth on bright isolated pre-flip stars), so the premise
"aperture under-captures the skirt" is true. But the decisive test refutes the *fix*: differential
frame-to-frame scatter is **flat** from r=5 px to the EE>=0.99 plateau (~24->27 mmag; minimum ~21 mmag
at r~6 px, within noise), and a per-frame FWHM-adaptive aperture is **worse** (30-32 mmag). The
+-5-7% skirt-fraction swing (~48 mmag nominal) is **common-mode PSF breathing** that differential
photometry already cancels; the bright-star floor is not aperture-limited. Grounded decision: widening
the aperture is not justified (a 5->6 px bump is within noise). Evidence: `CURSOR_RESULT_round2.md`,
`tmp/b1_cog_diag.py`.

**B.2 transparency frame-quality gate - adopted behind a default-OFF flag.** A per-frame
PSF-concentration statistic, median `flux_large / flux` over bright unsaturated sources, cleanly
separates transparency/PSF-collapsed frames (ratio ~11-16, FWHM at the rail, science aperture
catching only noise) from clear-but-faint frames (ratio stays ~2.7 as flux falls equally in both
apertures). Grounded in the curve-of-growth / SNR-optimal aperture framework (Howell 1989). The gate
rejects whole frames whose ratio is a robust outlier (`z>k`, primary) guarded by `FWHM >= factor.median`
(spares sharp frames); safety floor on min kept frames. **Default OFF => baseline byte-identical**
(preserves the UI test). Params `frame_quality_{gate_enabled,ratio_k,fwhm_factor,min_keep_frames}`
(config + ui_settings + VYVAR_PARAMS); `howell1989` gated. Isolated measurement (draft_413 g, ON vs
OFF): LC scatter for bright targets drops **median -257 mmag** (14/15 improved; a flat field star
0.342->0.035 mag); **trust unchanged (RED)** because RED is set by the structural check-star/thin-comp/
colour-term-off gates, not LC scatter - the gate is a precision win, not a trust-flip. Scope: Phase 2A
(target LC/trust); Phase-0+1 comp selection not gated yet. Evidence: `CURSOR_RESULT_round2.md`,
`tmp/b2_measure.py`, `docs/round2_figs/`.

---

## variable_targets selection is spatial-first (frame bbox), and it also purges variables from the comp pool (2026-06-17)

VSX `variable_targets` are selected from the **frame footprint** (frame bbox + the 50 px in-frame
margin) via `_query_vsx_local_frame_bbox`, **not** the 3.5 deg cone box. The cone hit
`catalog_query_max_rows=15000` with **no `ORDER BY`** and silently dropped a contiguous Dec slice
(the northern half of the field, incl. bright named variables - V0454 CrA m9.9, KQ/KM/KT CrA).
Completeness must not depend on row order; the bbox is sub-degree so the SQL result is tiny and the
cap never bites.

**Accepted consequence (Milan, 2026-06-17): this is NOT a pure target-add.** The same
`variable_targets` list drives the **global comparison-pool veto**
(`build_global_comp_pool(..., variable_target_catalog_ids=...)`). The now-complete list correctly
removes newly-recognised variables from the comparison ensemble, which shifts a minority of
previously-measured targets (draft_413 g: 6/19, max |Deltamag|=0.122) - both by dropping a
variable-as-comp directly and by lowering surviving comps' field-wide RMS (the variables had
inflated ensemble scatter) so weights re-rank. This is a comp-purity **improvement**, deterministic,
and accepted as the correct behaviour. Byte-identity vs the capped baseline is therefore *not* a
goal for B. The plan-level `comparison_stars.csv` (the unconditional cone veto in
`write_photometry_plan_files`) is byte-identical old-vs-new; the shift is entirely in the phase-1
global-pool path. Evidence: `CURSOR_RESULT_round1.md`, `tmp/fix2_e2e_oldnew.py`,
`tmp/fix2_mechanism3.py`.

## Completeness gate scores measurable targets, not raw active count (2026-06-17)

`audit_photometry_completeness` (`night_run.py`) is a **false-success / truncation** guard, not a
yield gate. Its verdict is taken against **measurable** targets: a missing target (active, no
summary row) counts as honest *unmeasurable* (does NOT fail) when it is fainter than the achieved
detection depth (faintest measured target's catalog mag); a missing target at-or-above depth is a
*measurable miss* and still fails (the silent-truncation guard - draft_383/385-style cut-short runs
must fail). Depth is derived from the data, so **no new threshold/param** is introduced.
Conservative fallbacks (no mag, nothing measured) treat misses as measurable so truncation can never
masquerade as honest. This unblocks honest RED nights (g 19/22 = 86.4% now PASS) without weakening
the guard. Evidence: `tests/test_completeness_gate_measurable.py` (4/4).

---

## Product scope: light curves in, period science out (2026-06-09)

**VYVAR scope:** produce, validate, and prepare light curves for submission (AAVSO / VarAstro / VSX).
Scientific analysis of those light curves -- period finding, classification -- is **OUT of scope**
and left to downstream tools (Peranso, VStar, Period04).

Internal Lomb-Scargle / BLS use is **not** VYVAR analyzing its own LCs as a science product. It
runs only on:

1. **External TESS cutouts** in `tess_verify.py` -- to confirm a variable-star candidate against an
   independent survey.
2. **Catalog-period display** (VSX / ASAS-SN / ZTF) in the variability UI -- detection/validation
   aids, not folded LC products.

Do **not** expand these into the PDF report as a period product (this descopes TODO-GS9).
Citations `lomb1976` / `scargle1982` / `vanderplas2018` **stay** -- they back the `tess_verify`
TESS cross-check.

## Product scope: single-night is the canonical publishable unit (2026-06-25)

**Decision (Milan).** The unit VYVAR produces, validates, and submits is a **single-night
light curve**. This is the regime cross-validated against SIPS/AIJ (V0612, wide rig) and the
unit a user processes per session. **Multi-night global matching / inter-night zeropoint
(TODO-GS8) is descoped from HIGH to FUTURE / nice-to-have** - built only if a long-baseline
science case requires it. Consistent with the TODO-GS9 (downstream period analysis) scope
boundary: stitching and long-baseline analysis live outside VYVAR's core submission path.

## CAL-DIAG - calibration-time radiometry gate (agreed 2026-07-07)

**Decision (Milan).** Add a camera-agnostic calibration-time diagnostic. Spec:
`docs/VYVAR_CAL_DIAG_SPEC.md` (v1.1, 2026-07-07). VYVAR must verify radiometry from data
when users build masters at arbitrary binning -- not assume per-camera SUM/MEAN conventions.

**Scope (planned):**
- Post-dark-subtraction sky-median sanity (> 0, plausible).
- Pre-subtraction `median(light)` vs `median(resampled dark)` cross-check; loud auto-correction
  (SUM -> MEAN retry) or fail-closed abort.
- Provenance flag `dark_resample` = `SUM` | `MEAN_AUTOCORRECTED` | `PASSTHROUGH`.

**Gap today:** `calibration.py` `get_processed_master` guards geometric resampling only; no
radiometric validation. **F-BINGAIN-1 RN sub-question** (db 7.6 e- scaled x2 -> 15.2 e- if DB
already read-mode-0) resolves here, not via ad-hoc param_resolver exponent change.

**Status:** **IMPLEMENTED (2026-07-07); re-validated 2026-07-14 on HEAD 13341b3.** Spec v1.1
**APPROVED (Milan, 2026-07-14).** **REMOVED 2026-08-11** in commit `967f835` (config
parameter reduction). Spec retained: `dev/results/specs/VYVAR_CAL_DIAG_SPEC.md`. See
ROADMAP **SUPERSEDED - CAL-DIAG**.

---

## Calibration masters: manual library build only; no auto-stack on import (2026-07-07)

**Decision (Milan).** When an imported session folder contains raw darks/flats, VYVAR copies
them to `Raw/darks|flats` in the archive but does **NOT** auto-stack them into masters at
import time. Master creation stays **manual via the CalibrationLibrary UI**
(`generate_master_dark_from_source_dir` / `generate_master_flat_from_source_dir`,
importer.py:1483-1608; ui_calibration_library.py:366,398), which registers the result in
`CALIBRATION_LIBRARY` for reuse.

**Why.** The library is the single curated source of validated masters (scoped per
equipment+telescope, validity-windowed). Auto-stacking arbitrary session frames at import
would silently admit unvetted masters (wrong temp match, too few frames, light leaks) into
science calibration -- against the "trust in the numbers" mission. The user builds masters
deliberately, once, and the library serves them everywhere.

**Context.** The product description previously implied auto-build ("folder with
lights/darks/flats creates new masters"); the 2026-07-07 flow trace
(`CURSOR_RESULT_caldiag_flow.md` Q3) showed the code never did this. Milan confirmed the
code behavior as intended -- this entry codifies it so the gap between description and code
does not resurface as a "bug".

## Calibration masters: library precedence over session raw (2026-07-07)

**Decision (Milan).** When a session provides raw darks/flats AND a valid scoped library
master exists for the same obs_group, the **library master wins** for calibration
(importer.py:1438-1439). Session raw frames are archived under `Raw/darks|flats` as
provenance, not stacked or preferred.

**Why.** A registered library master is a known quantity: scoped to the equipment set,
within its validity window (90 d dark / 200 d flat), built from a deliberate stack. A
handful of session raw frames is not automatically better -- and choosing it silently would
make calibration depend on what happened to be in the source folder. Deterministic
precedence beats per-session guessing. The user who wants session-specific masters builds
them via the UI (previous decision), which then wins the scoped lookup as the freshest
valid entry.

## READNOISE_E is per-pixel at bin1; RN_eff = RN_db * bin is correct for software binning (2026-07-07)

**Decision.** The DB `EQUIPMENTS.READNOISE_E` value is defined as the **per-pixel read noise
at bin1** (native read mode). The EQUIP-BINNING scaling `RN_eff = RN_db * binning`
(exponent 1; `_scale_bin1_to_binning`, param_resolver.py:154-159, applied in
`resolve_read_noise` param_resolver.py:493-498) is **physically correct** for
software/digitally binned CMOS: a bin-b superpixel sums b^2 independent pixel reads, so
read noise adds in quadrature, sigma = RN_px * sqrt(b^2) = RN_px * b. For the QHY294MM,
7.6 e- (per-pixel, per spec sheet) -> 15.2 e- effective at bin2. There is **no
double-count** under this semantic; a double-count would require the DB to store an
already-binned effective value, which nothing in code or docs claims.

**Closes the F-BINGAIN-1 RN sub-question directionally.** Final empirical closure = photon
transfer on **bin2 flats** (Milan data item; the 2026-07-07 field-light attempt was
inconclusive, g_eff~0.9). No param_resolver exponent change; the guard is CAL-DIAG's
data-driven radiometry gate, not per-camera constants.

**Implementation note:** semantic comment at param_resolver.py:155 and in the
`set_equipment_cosmic_params` docstring (database.py:2928), landed with the CAL-DIAG PR.

---

## Photometry method & scale

### Plate scale is WCS-derived (~ 9.77 arcsec/px on the wide rig), not 1.3
The project-wide `1.3 arcsec/px` belief was wrong - it was a Newton 300/1200 + C3-26000 (binned 2x)
placeholder leaking onto the wide Carl-Zeiss/QHY294MM field via a global config default. The
resolver is **solved WCS/CD authoritative -> config only as last resort** (sane clamp widened
to `0.1-30.0`). WCS-dependent geometry (ePSF isolation, FOV / `max_dist_deg`, TESS context) is
now correct; pixel-based geometry (aperture / annulus / SNR-optimal table / field density) was
immune. **Status: settled (2026-05-29/30).** Any residual `1.3` in old `pipeline_meta` is stale
run metadata, overwritten on a clean re-run (see ROADMAP: WIDE-RIG-REPROCESS).

### Brno / external data: characterize before PSF or NEIGHBOR-SUB (2026-06-08)

Before relying on PSF or NEIGHBOR-SUB for publishable output on incoming Brno University data (or any
new field), run the standard characterization gate: plate scale + pixel sampling, ePSF-vs-star Moffat
mismatch (decisive), and crowding (`compute_crowding_index`). NEIGHBOR-SUB is **validated at fine
scale** (draft 367: mismatch ~1.0, A9 HV ~83%, FAIL-SILENT 0). If new data is **coarse or
under-sampled** (mismatch > ~3%), it falls back to the **SAFE_LOW_YIELD** regime -- bright-neighbour
blends will correctly **REFUSE**, not be silently deblended. That is the publishable-safe behaviour;
do not force deblending outside the validated regime.

### PSF fit weights: sky + read noise only (2026-06-09)

Mid-mag PSF bias on V3d (+4.5%) was **flux-dependent weighting**: including source Poisson in
photutils fit `error` makes relative pixel weights depend on brightness, so the bright/faint flux
ratio becomes PSF-model-dependent and biases point-source fluxes (Astier et al. 2013; Lacroix /
Regnault 2025). Production fix: `psf_weight_mode=sky_only` -- one estimator for all magnitudes,
uniform per-stamp sigma from sky + read noise only. Accepted small bright-end precision cost vs
object-weighted fits. Forced position (Guy et al. 2010) not required at fine scale after Fix 1.
Residual hardware systematics (brighter-fatter / pocket effect; Lacroix 2025) remain out of scope.

### Sandwich reported PSF uncertainty (2026-06-09)

With sky-only fit weights, reported `psf_flux_err` must propagate **true pixel variance**
(sky + source Poisson + read noise) through the **actual** weights used in the fit
(`psf_err_mode=sandwich_skyonly`). This calibrates error bars (V3d P3 ~1 mag<=17) without
changing fluxes. Stetson 1987 / Mighell 1999 cited for ensemble context; sandwich is the
production implementation for per-star PSF errors.

### EPSF-1 robust FWHM QC (2026-06-08, diagnostic only)

`epsf_fwhm_native` uses an azimuthally-binned radial profile with linear 0.5 crossing (not
first-pixel half-max). QC warning band tightened to **[0.80, 1.25]**. Does not enter flux path
or `assess_psf_quality`; numeric SHA unchanged. Validated via harness V3e.

### NEIGHBOR-SUB shape: PSF subtract contaminant, aperture measure target (2026-06-09)

For blended targets, use ePSF to fit and subtract a bright neighbour, then run the existing
aperture path on the residual stamp. PSF does not replace aperture for science flux; it only
removes contamination. Does not revive grouped PSF / rule 2 (mutually exclusive thresholds).
Gated `psf_neighbor_sub_enabled` OFF. Synthetic validation: **VALIDATED_FINE_SCALE_IDLE**
(A9 HV ~83%, FAIL-SILENT 0 on draft 367). Real-field enablement after Brno characterization.
Full design: `docs/VYVAR_NEIGHBOR_SUB_DESIGN.md`.

### Fail-safety hygiene #4 (2026-06-08, Milan confirmed)

**MASTERSTAR WCS persist (`vyvar_platesolver.py`):** failed `fits.writeto` (or read/update) on
MASTERSTAR is **fail-closed** for that draft -- returns `solved=False` via `_SolveWcsWriteError`,
`LOGGER.error`, pipeline blocks Phase 2A (no silent stale WCS). Other frames in the batch are
unaffected.

**Edge-ok filter (`photometry_core._edge_ok_from_masterstar_pipeline`):** on check failure,
**fail-open** (all stars treated edge-ok so detection is not zeroed) but **loud** --
`edge_filter_failed=True` + `edge_filter_note` on `variability_candidates.csv` only (not on
byte-identity SHA files: `lightcurve_*.csv`, `comp_quality_*.json`,
`comparison_stars_per_target.csv`). Report cover shows edge-filter status when flagged.

**Dead UI modules:** `ui_photometry_results.py` and `ui_suspected_lightcurves.py` deleted;
function covered by `ui_aperture_photometry` + `ui_variability`.

### Crowding + ePSF FWHM context use measured core (VY_FWHM_GAUSS), not DAO search scale (2026-06-09)
`VY_FWHM` on MASTERSTAR is the DAOStarFinder search parameter (~3.4-3.8 px on h & chi Per L);
`VY_FWHM_GAUSS` is the 2D Gaussian core fit (~2.7 px) already used by aperture photometry
(`pipeline.py:9206`). Crowding (`_load_wcs_meta`) and ePSF build context
(`get_epsf_fwhm_from_context`) previously read `VY_FWHM` only, inflating blend disks and
deflating ePSF QC ratios. **Decision:** shared `header_core_fwhm_px` prefers
`VY_FWHM_GAUSS` -> `VY_FWHM_GAUSSIAN` -> `VY_FWHM` at exactly those two sites; display-only
and plate-solve hint readers keep `VY_FWHM`. Validated: numeric SHA `770966c3...` unchanged;
h & chi Per L crowding 77/87 -> 58/53 is_blended.

### Aperture is the validated workhorse; PSF validated-but-gated
At 9.77 arcsec/px the PSF is well-sampled and stable across the field, so a single ePSF already
captures it and aperture wins. Every PSF variant - single ePSF, spatial `GriddedPSFModel`,
`SourceGrouper` joint fit, per-star adaptive selector - was implemented and **lost to aperture**
at this scale (single ePSF ~3x worse comp RMS; grid starves cells; grouper diverges on
sub-resolution blends). **Decision: keep all PSF flags OFF in production on the wide rig.**
On fine-scale synthetic truth (draft-367-like), PSF is now **publication-grade** (accuracy,
precision, sandwich P3) but remains **gated OFF** until real Newton / Brno data passes the
characterization gate. **Status: settled on wide; fine-scale validated-but-gated.**

### Full-frame DAO over fixed-position stamp photometry (overnight-batch model)
Every frame runs full-frame `DAOStarFinder` + match (the "master fast path" skips the per-frame
Gaia cone, not the detection). The SIPS-style speedup considered 2026-04-22 - read flux only at
fixed catalog positions, no full-frame finder - is **deliberately not adopted.** Full-frame
detection buys QC that fixed positions lose: per-frame local centroiding (absorbs WCS / drift /
resampling offsets), shape-roundness rejection of cosmics, hot pixels and CCD columns,
match-count sanity against a bad solve, new-source detection, and completeness diagnostics. The
compute cost is accepted because **VYVAR runs as an overnight batch** after the session ends,
while the observer sleeps - the binding constraint is a *trustworthy* Summary Measure Report by
morning, not wall-clock throughput. The same logic licenses PSF's extra cost when it is enabled
on fine-scale data: accuracy and robustness over speed. **Corollary:** performance work is
welcome only where it does not trade away QC (I/O, parallelism, caching); the per-frame
full-frame detection itself is a feature, not a bottleneck to remove. **Status: settled
2026-06-02.** (See also: SIPS comparison below.)

### Per-frame catalog: drop unmatched DAO before aperture; Moffat gated on ePSF (2026-06-12)
**DAO detection stays full-frame** (QC unchanged). After `detect_stars_match_master_reference`,
rows with empty `catalog_id` are dropped **before** aperture / Moffat / PSF work
(`_proc_drop_unmatched_dao_rows`; key on `catalog_id`, not `source_type`). They were never written
to `proc_*.csv` anyway (final `_proc_catalog_keep_matched_rows_only`); this is pure wasted compute.

**Moffat fit is Step 1 of the two-step ePSF path only** - gate `if _run_epsf:` (not
`_run_aperture`). In aperture-only production (`psf_photometry_enabled=False`), `moffat_*` columns
are omitted; LC / comp-QA readers do not consume them. **LC byte-identical** when `VY_FWHM` /
`VY_FWHM_GAUSS` drives aperture radius (verified `draft_000389` B_60_1). Chi_and_H photometry SHA
unchanged (`proc_*.csv` not in SHA set).

### What VYVAR deliberately does NOT adopt from SIPS (2026-06-02)
Comparison against SIPS (Moravian Instruments; v4.4 manual + Pejcha & Cagas 2022, A&A 667,
A53) confirmed the two tools share a photometric **family** - both do full-frame per-frame star
detection, intensity-weighted sub-pixel centroids, per-star automatic apertures, robust
background, saturation invalidation, and a flux-summed ensemble. SIPS's speed comes from
**native C/C++ + multicore**, not an algorithmic shortcut - which *validates* VYVAR's full-frame
DAO choice rather than contradicting it.

**Deliberately not adopted:**
- **Fixed-position stamp photometry** (read flux at catalog x,y only; no per-frame finder). See
  the full-frame DAO decision above - QC and trust over throughput.
- **Neural-network variable detection (VDI/NN).** Against the trust mission - VYVAR keeps
  explainable statistics (Sokolovsky indices, RMS hockey-stick, independent cross-validation),
  not a black box.
- **UCAC4-based calibration.** A step back from VYVAR's native Gaia DR3 + BP-RP (deeper, modern,
  colour-complete).
- **SPL scripting / REST API.** Different stack; VYVAR is Python/Streamlit.

**Worth borrowing - scoped in ROADMAP, not core changes:**
- **Wide-field WCS distortion check** (MEDIUM): confirm SIP / higher-order terms on the wide
  rig; SIPS uses a 3rd-order 2D polynomial (Monomial/Legendre).
- **Spatial term in ensemble calibration** (LOW): SIPS's `x1.X + y1.Y + ...` field-gradient terms
  - relevant only for a future whole-field absolute mode, not the current per-target differential
  path.

The Pejcha & Cagas paper is a **citation / positioning reference for the GS7 paper**, not a
source to copy. Where VYVAR is already ahead: Gaia-native catalog, independent cross-val +
per-target trust gate, comp_qa, literature-backed comp selection, and reproducibility/citation
discipline. **Status: settled 2026-06-02.**

### What VYVAR does / does not adopt from CoLiTecVS (2026-06-09)
Comparison in the same spirit as the SIPS entry. Sources: Savanevych/Briukhovetskyi/Khlamov/
Kudzej/Dubovsky/Parimucha et al. -- Astron. Nachr. 2019;340:68-70; CAOSP 49,151 (2019);
2022A&C....4000605S; Dubovsky et al. 2017 OEJV 180 (inverse-median-filter detail). CoLiTecVS is
from the same community Milan works in (Kolonica Saddle / Vihorlat Obs. / UPJS Kosice).

**Shared photometric family.** Both are automated differential aperture-photometry pipelines
that take raw frames to AAVSO-style light curves with ensemble comparison stars and minimal
manual step-by-step handling. Both were validated against the C-Munipack / Muniwin class of
tools and reach comparable scatter (CoLiTecVS aperture uncertainty < 0.04 mag; on the MASTER OT
J174305 field CoLiTecVS auto-ensemble SD ~0.0078 vs C-Munipack+MCV ~0.0067, i.e. parity).
Same problem domain, same accuracy class.

**Where VYVAR is ahead:**
- **Gaia DR3-native + colour.** CoLiTecVS selects comparison stars from AAVSO charts (LookSky
  tool). VYVAR is Gaia DR3-native with BP-RP colour, colour-term handling, and colour-aware comp
  selection -- a more modern catalog/colour basis.
- **Independent QA + per-target trust verdict.** CoLiTecVS reports a global aperture uncertainty
  and validates in aggregate (one mean/SD table); it exposes no per-target machine-checkable
  verdict and no independent second-extractor cross-check in-product. VYVAR adds comp_qa
  (Sokolovsky leave-one-out locus), a **production per-target trust gate** (comp health +
  check-star scatter + `lc_quality_flag`; see Trust & validation below), and **offline-only**
  SEP cross-validation (~0.2%/frame via `xval_run.py` - validation study, not a pipeline trust axis).
- **Reproducibility / provenance.** VYVAR has SHA-256 byte-identity on photometry artifacts, a
  citation emitter, and a decision log. CoLiTecVS is a compact closed all-in-one -- by the
  authors' own note you cannot isolate and test a single internal stage.
- **Modularity / auditability.** VYVAR stages are individually inspectable and read-only
  auditable; CoLiTecVS is monolithic (raw -> LC, turnkey).

**Where CoLiTecVS is ahead (VYVAR gap):**
- **Inverse-median-filter brightness equalization** (their signature). Removes large-scale
  illumination non-uniformities from Moonlight / scattered light that dark+flat do NOT correct.
  The authors report it usually beats classical flat-field for background equalization, with no
  measurable photometric non-linearity (Dubovsky et al. 2017). VYVAR is flat-only and has NO
  scattered-light / large-scale gradient equalization stage. Real gap for moonlit nights and
  light-polluted sites.
- **Online / real-time mode (OLDAS-Night):** processes data live off the sensor. VYVAR is offline
  batch only.
- **Maturity / proven scale:** CoLiTec lineage (700k+ observations in the asteroid-detection
  heritage); CoLiTecVS tested on 100+ time series (20-600 frames) and in regular operational use
  at Kolonica. VYVAR is a single-observer pipeline.
- **LookSky one-click AAVSO-chart comp selection + reusable per-target task-file** (slick repeat-
  target UX). Workflow convenience, not necessarily better science.
- **Compact turnkey UX** (minimal interaction). VYVAR needs more setup and understanding.

**Worth borrowing -- scoped in ROADMAP, not core changes:**
- **PRIMARY: optional large-scale illumination-gradient removal** (inverse-median-filter or
  equivalent background equalization), pre-photometry, OFF by default and gated. Directly addresses
  the one real capability gap above; ties to existing items TODO-LC-TREND (differential extinction /
  moonless-night note) and the LOW SIPS "spatial term". VYVAR can adopt it more safely than
  CoLiTecVS validated it: byte-identity SHA, comp_qa locus, SEP cross-val, and check-star scatter
  are the acceptance harness -- enable on a moonlit/gradient draft, confirm locus + check-star
  improve (or are unchanged) and constant-star differential RMS does not degrade, with numeric SHA
  tracked as a separate baseline. **Risks to gate:** median-background subtraction can remove real
  extended flux and perturb faint-star annulus estimation; must stay optional, never silently alter
  the photometry path, and pass the trust-gate acceptance before default-on (mirrors PSF "OFF until
  validated" discipline).
- **SECONDARY (optional, UX):** reusable per-target "task-file" (fixed target + comp set reused
  every reduction). Only if VYVAR lacks an equivalent per-target config; low priority, workflow not
  science.

**Deliberately not adopted:**
- **Monolithic compact architecture** -- VYVAR's modular, auditable design is a deliberate strength.
- **Online OLDAS mode** -- out of scope for VYVAR's offline-rigor model.
- **AAVSO-chart comp selection** -- VYVAR's Gaia DR3 + BP-RP colour basis is more modern;
  switching to chart-based selection would be a step back.

**Net:** borrow ONE idea (optional gated illumination-gradient equalization), validated through
VYVAR's existing trust harness; keep everything else as VYVAR already does it. **Status: settled
2026-06-09.**

### COG (curve-of-growth) aperture correction: implemented, default OFF
Per-frame encircled-energy correction removes the constant target<->comp enclosed-flux bias and
the seeing-correlated systematic from per-star SNR-optimal radii. Byte-identical when OFF.
**Decision: ship gated, leave OFF** until validation on real nights. Mixed-frame guard is
wired (see `APCORR-MIXEDFRAME-ALLORNOTHING` below).

### APCORR-MIXEDFRAME-ALLORNOTHING (2026-07-19)
**Context.** With `cog_aperture_correction_enabled=True`, a night that only partially succeeds
at COG (`cog_ok=False` on some frames) would otherwise route corrected flux on some epochs and
raw `dao_flux` on others -> a step in the light curve that is not astrophysical.

**Decision.** Night-level all-or-nothing gate after per-frame COG AC computation: if **any**
science frame of the night lacks a usable correction (`cog_ok=False` or missing `cog_ok`),
COG application is **disabled for the entire night** - every Phase 2A row takes the standard
Metoda B AC chain. Log:
`[APCORR] COG night fallback: N/M frames without cog_ok -> whole night uses standard AC`.
Provenance: `cog_night_fallback=true` (plus counts) in `photometry/pipeline_meta.json`.
Per-frame `fallback_ee` remains a FUTURE refinement (noted on ROADMAP closure).

**Status:** implemented; COG still default OFF pending enablement validation.

### SKIPPROC-PERMANENT (2026-07-22)
``skip_processed_directory`` removed; in-place QC + ``qc_metrics.csv`` allowlist is the only
preprocess path. The ``processed/lights`` copy tree is retired (it existed only as a check
artifact; allowlist supersedes). Resume/skip-drafts: preprocess complete when
``calibrated/lights/qc_metrics.csv`` exists (or ``pipeline_meta`` preprocess stage stamp).
**Status:** implemented.

### SKY-SURFACE-RESTORE (2026-07-27)
Accidental drop of order-2 sky-surface subtract on mono frames in `013cb0c` (SKIPPROC) is
**reversed**. This is the **second occurrence** of F-431-HEADLESS-DIVERGENCE (lost preprocess
ADU step); the first closure (2026-07-16) did not add a guard, so SKIPPROC could remove the step
again without tripping any gate. T3-PREPROCESS-SKY-SURFACE (2026-07-16) remains authoritative: when
``preprocess_sky_surface_order > 0``, ``_qc_enrich_calibrated_in_place`` calls
``_fit_subtract_preprocess_sky_surface`` for all non-mosaic lights (no ``VY_CHANNEL`` gate);
OSC Bayer mosaics are still skipped to avoid double subtract on channels. Headers:
``VY_SKYSF``, ``VYSKYORD``, ``VYSKYP2P``. Guards: **INV-PREP-01** (large_small_ratio WARN on
calibrated-frame gradient; detects missing or ineffective sky-surface subtract from infolog),
**INV-MS-01** (DAO_ONLY fraction WARN/FAIL). C.4 acceptance on BO CVn raw path pending Milan
raw frames on disk. **Status:** code landed; science validation pending.

### SKYSF-DOUBLE-GUARD (2026-07-30)

**Problem.** In-place preprocess (``_qc_enrich_calibrated_in_place``, since ``013cb0c``) always
re-ran ``_fit_subtract_preprocess_sky_surface`` when ``preprocess_sky_surface_order > 0`` even if
the frame already carried ``VY_SKYSF=True``. Headers were write-only; UI preprocess twice,
MAKE MASTERSTAR re-entry, or interrupted re-runs could double-subtract. Measured defect cost on
draft 452 bench: **508.97 ADU** ``max_abs_diff`` for **one** extra pass (452 calibrated vs 452
re-preprocessed in place) -- star-mask non-idempotency removes real signal on the second fit, not
a float-rounding issue.

**Rule.** One-shot in-place sky-surface subtract per calibrated light, guarded by headers on the
**same** FITS being modified:
- ``VY_SKYSF`` absent (legacy copy-tree calibrated): subtract and write markers.
- ``VY_SKYSF`` present and ``VYSKYORD == preprocess_sky_surface_order``: **skip** (INFO + run-summary counter).
- ``VY_SKYSF`` present and ``VYSKYORD != order``: **abort**; message requires recalibration from raw.
- ``preprocess_sky_surface_force_reapply=True``: bypass guard; record override in row provenance + WARNING in run summary.

**``apply_sky_surface`` removed** (was dead after T3 restore): mono and OSC channel paths gate on
``preprocess_sky_surface_order`` only; mosaics still excluded via ``BAYERPAT`` without ``VY_CHANNEL``.

**Anchor note.** Copy-tree draft 435 verified single subtract; no contamination on drafts
checkable on disk; in-place-era drafts unavailable, status unknown. **Status:** implemented;
regression tests ``dev/tests/test_skysf_double_guard.py``.

### SKYSF-UNKNOWN-PROVENANCE-WINDOW (2026-07-30)

**Window.** Commits ``013cb0c`` (2026-07-22, in-place skip-only preprocess on calibrated FITS)
through ``84174ae`` (2026-07-30, ``VY_SKYSF`` / ``VYSKYORD`` idempotency guard).

**Defect.** Without the guard, any repeated preprocess pass on the same calibrated light could run
``_fit_subtract_preprocess_sky_surface`` again. Measured cost on draft 452 bench: **508.97 ADU**
``max_abs_diff`` for **one** extra pass (star-mask non-idempotency; not a rounding artefact).

**Drafts inside the window (status UNKNOWN).** ``draft_000449`` through ``draft_000455`` (and any
other post-``013cb0c`` run not re-verifiable on disk). These drafts are **gone locally** and
**absent from backups** on the forensic machine as of 2026-07-30.

**Rule.** Quantitative results committed from that window (``f8285c7`` forensic bundle and
derivatives) carry a **PROVENANCE WARNING** banner. Treat as indicative, not validated, until a
draft is restored and headers/pixels re-checked. Pre-``013cb0c`` anchor drafts (``draft_000435``
family) are outside this window and remain anchor-validated where Part 1 forensics passed.

**Status:** bookkeeping + banners applied; no remediation or reprocessing.

### P-10-SKYSURF-SIGN + PREPROCESS-REGIMES-JULY (2026-07-30)

**P-10 defect.** ``_fit_subtract_preprocess_sky_surface`` fitted ``z = bg_median - work`` then
subtracted ``surf``, **doubling** large-scale gradient while preserving pedestal. SKYSF-DOUBLE
forensics ratio ``283/136.8 ~ 2.07`` on refit p2p is this signature, not mask non-idempotency alone.

**Fix (local, bundled with sigma_pp DAO estimator):** fit ``z = work - bg_median``, keep
``out = work - surf``. **Not pushed alone** - see Audit Tranche 3.

**Three distinct preprocess regimes in July (none physically correct pre-fix):**

| Era | Behaviour | Large-scale content |
|-----|-----------|---------------------|
| pre-``013cb0c`` (435 family) | sky-surface applied | gradient **doubled** (``2g``) |
| ``013cb0c`` -> T3 restore | sky-surface **absent** for mono (SKIPPROC) | natural ``g`` |
| restore -> ``84174ae`` | sky-surface applied (sign bug) + double-pass risk | ``2g`` (+ double risk) |

**Anchor status.** ``VL-ANCHOR-WCSINV`` fingerprints (``b7f980c0...``, ``n_raw_dao=2552``, etc.)
describe **reproducible doubled-gradient data**, not physically flattened backgrounds. Anchor is a
**reproducibility reference until re-cut** after P-10 + estimator bundle + delta measurement.

**Byte-identity doctrine.** Identity gates prove the pipeline repeats itself, not that the transform
is correct. At least one gate per physical step should compare against an independent expectation
(synthetic gradient test in ``test_preprocess_sky_surface.py`` for sky-surface).

**Status:** fix + sigma_pp estimator + ``masterstar_dao_threshold_sigma=3.8`` recalibration
implemented locally; anchor re-run and ledger re-cut pending.

### D1-3-MASTER-FLAT-CONSTRUCTION (2026-07-30, audit Stage 1)

**Question.** Is the CalibrationLibrary master flat dark/bias-subtracted before stacking, and is
normalization at stack time or calibrate time?

**Evidence (code).** ``importer._write_master_to_library`` / ``_generate_master_flat``: per-pixel
**median** stack of raw flat FITS with **no** dark or bias subtraction step
(``generate_master_flat_from_source_dir`` docstring, ``importer.py`` ~1639-1700, ~2122-2133).
New masters receive ``VYFLNRD=1`` with comment *Median normalization deferred to calibrate after
resample* (``importer.py`` ~1122-1126). ``calibration.py`` module docstring: legacy flats
normalized at stack time; new flats normalized in RAM at calibrate via ``normalize_flat_master``.

**Literature.** Howell, *Handbook of CCD Astronomy* (flat-field chapter): flats must have additive
terms (bias/dark) removed before treating the frame as a pure multiplicative illumination map;
otherwise bias does not scale with signal and breaks ``(light-dark)/flat``.

**Status.** **Open scope gap:** VYVAR does not dark-subtract individual flats before median stack.
Operator must supply bias-subtracted or dark-subtracted sky flats externally, or accept that
twilight/dark current in raw flats is folded into the master. Normalization: **at calibrate** for
``VYFLNRD=1`` masters (current library convention).

### D10-2-GAIA-JOHNSON-RANGE (2026-07-30, audit Stage 1)

**Literature.** Gaia DR3 CU5 Table 5.9 polynomials (``gaia_johnson.py``); applicability range
Table 5.10: **G in [8, 16] mag**, **BP-RP in [-0.5, 5.1] mag** (pinned as ``G_MAG_MIN/MAX``,
``BPRP_MIN/MAX``).

**Code.** ``transform_gaia_to_johnson`` rejects out-of-range inputs (``gaia_johnson.py`` ~140-155);
``transform_comp_row_for_osc_band`` excludes comps from OSC ensemble with logged reason.

**Anchor (draft_435 snapshot).** 1/148 comparison stars and 39/2951 masterstars outside published
G range (bright stars); **0** outside BP-RP range in sampled failures.

**Status.** Guard **present**; finding **latent** for colour, **active** for a handful of bright comps.

### CONFIG-PATH-DATA-ROOT (2026-07-27)

**Problem.** Config path-valued keys stored as relative strings (e.g.
``exoplanets/vyvar_exoplanet_local.db``, ``VSX/vyvar_vsx_local_v2.db``) were resolved with
``Path.resolve()`` against the **process CWD**, not ``data_root``. When night runs or Streamlit
sessions set CWD under ``Archive/Drafts/`` (common since ``c99fcec`` data-dir bootstrap), a
configured file under ``<data_root>/`` was missed; VSX/exoplanet queries returned empty and the
pipeline reported success (draft 450, 2026-07-24).

**Trigger.** Latent CWD-relative resolution predates the regression; first production symptom on
draft 450 coincides with ``c99fcec`` (2026-07-24) separating ``data_root`` from install root while
night-run CWD stayed under ``Archive/Drafts/``. ``3c31bfa`` materialized relative paths in
``config.json`` but did not change resolution semantics.

**Rule.** One resolution function for all config paths: ``resolve_config_path(raw, data_root)`` in
``config.py`` -- absolute paths as-is, relative paths against ``data_root``, never CWD. Catalog DB
open paths use ``database._resolve_catalog_db_path`` + ``require_*_db_path`` helpers; a configured
path that does not exist or has zero rows (VSX) **must fail loud** with the config key name. Guard:
``dev/tests/test_catalog_db_path_resolution.py``. **Status:** implemented (POST-451 closeout).

### PHASE0-ACTIVE-COUNT-VT-CONTEXT (2026-07-27)

**Problem.** POST-451 C.4 acceptance table listed active targets **~160-175** by extrapolating from
the anchor's 165 actives. That band was wrong: anchor actives come from a **frozen**
``variable_targets.csv`` built when ``vsx_variable_targets_mag_limit = 14.5`` still existed (245
VT rows, faintest G=14.50). The live full-path VT has **no mag limit** (873 VSX rows). Comparing
live active counts to anchor without stating the VT context repeats the apples-to-oranges error
made earlier with the ``~283`` masterstars target.

**Verified decomposition (draft_452, live VT):**

- VT rows with G <= 14.5: **243** (vs 245 on frozen anchor VT).
- Active total **201** = Group A **163** (shared with anchor 165, minus 2 zone/skip) + Group B
  **38** (live-path-only actives).
- Group B actives carry ``gaia_match_source`` **masterstars** (37) or **masterstars_exo** (1) -
  not ``gaia_dr3_direct`` (Phase 0 rejects ``gaia_dr3_direct`` as ``not_target_eligible``).

**Rule.** When citing anchor active count (165) as a regression target for full-path runs, always
state the frozen VT / mag-limit context. Do not treat 165 as the expected live-path active count
without that caveat. **Status:** recorded; no code change.

### DETECTION-DEPTH-VS-LC-USABILITY (2026-07-27)

**Principle.** A star detectable on a single stacked MASTERSTAR frame at the DAO threshold is
**not** guaranteed to yield a scientifically usable differential light curve across the series.
Group B on draft_452 (38 actives): median ``lc_rms`` **0.398 mag**, **20** flagged RED - real
variables, genuinely detected, correctly identified, but not submission-grade at this rig and
exposure.

**Rule.** No magnitude cut, no target limit, no new parameter. The existing trust system (RED /
``lc_quality_flag``) is the correct handling. Do not reopen as a pipeline defect when faint
actives appear with poor curves. **Status:** recorded; no code change.

### GUARD-HEADLESS-OBSERVABILITY (2026-07-27)

**Problem.** ``INV-PREP-01`` and ``INV-MS-01`` called ``log_event()`` only, which writes to the
in-memory Infolog ring buffer. Headless ``run_night_pipeline`` did not mirror Part B's
``logging.info`` + ``log_event`` dual path; guard output never reached night-run stdout or
``infolog_*.txt`` on disk.

**Fix.** Guards emit ``LOGGER.info`` + ``log_event`` (same dual path as Part B VSX-GAIA / FAZA 0
funnel). ``log_milestone()`` helper added to ``infolog.py`` for future Cython rebuild. Headless
``run_night_pipeline`` calls ``ensure_infolog_logging()`` at start and ``save_infolog_to_disk()``
on success. Tests: ``test_inv_ms01_milestone_reaches_headless_logger``,
``test_inv_prep01_milestone_reaches_headless_logger``. **Status:** implemented.

### QC-ALLOWLIST-AUTHORITY
``qc_metrics.csv`` under the draft lights root is the **authoritative frame allowlist** for
alignment: only rows with ``status=ok`` (exact match) enter ``astrometry_align_and_build_masterstar``.
Frames on disk but missing from the CSV, or with any non-ok status, are excluded; a missing CSV is
fail-closed. ``VY_QC`` FITS headers remain **diagnostic only** (including prefilter stamps such as
``rejected_prefilter_fwhm``); they are not alignment gates. In-place QC visits **every** light
frame and records segmentation FWHM/elongation as diagnostics; it does **not** self-reject on
segmentation FWHM. Frame selection authority is the DB DAO-FWHM prefilter (Analyze Auto FWHM limit)
feeding ``prefilter_rejected`` status rows. Rationale: ``dev/results/CURSOR_RESULT_skipproc_qc_leak.md``.
Wired gate: **QC-01**. **Status:** unconditional since SKIPPROC-PERMANENT.

### VSX-AUTO-MAGLIM (2026-07-22)
``vsx_variable_targets_mag_limit`` removed. VSX rows enter ``variable_targets.csv`` from the
frame bbox query without a static ``mag_max`` pre-filter; **measurement scope** remains
detection-limited: a VSX star is active only with a DAO detection cross-matched to Gaia on
MASTERSTAR (unchanged Phase 0 criterion). Rationale: catalog ``mag_max`` band heterogeneity
made the static threshold semantically fuzzy; detection limit is the honest gate. Comp-pool
veto coverage grows with the fuller ``variable_targets`` list (correctness). **Status:** implemented.

### VSX-GAIA-MATCHER-TWO-STEP (2026-07-26, PHASE0-IDENTITY-GATE / MATCHER-FIX)

**Decision.** Plan-time cross-match is a **two-step separation** (design A):

1. **VSX -> Gaia DR3** over the frame bbox (deep local catalogue via ``query_local_gaia``;
   same geometry as ``_query_vsx_local_frame_bbox``). Measured ``rho`` sets acceptance radius
   ``r_max = sqrt(0.01 / (pi * rho))`` (1% contamination budget). Mixture fit yields ``Q``, ``w``,
   ``sigma_narrow``, ``sigma_broad`` for **ranking** multi-candidate cases; Sutherland reliability
   tiers are quality indicators only (not accept/reject gates).
2. **Phase 0** promotes only when the accepted Gaia ``source_id`` is present in
   ``masterstars_full_match.csv`` (``gaia_match_source=masterstars``). Matches without DAO
   detection remain in ``variable_targets.csv`` as ``gaia_dr3_direct`` (comparison-pool veto only).

**Why not masterstars as match RHS.** Masterstars are detection-limited (``Q ~ 0.32`` on anchor
night); fitting nearest-neighbour separations against them locks ``sigma`` onto the chance scale
(~100 arcsec), not astrometry. ``Q`` is a **fitted** mixture parameter (Sutherland & Saunders),
not assumed 1.

**True-match separation model (MATCHER-FIX-2).** A single Rayleigh is insufficient for VSX: the
catalogue is heterogeneous (774/873 in-frame entries on the anchor night are Gaia-identified
names with near-zero true separation; classical entries carry mixed epoch/quality astrometry),
and Gaia DR3 (epoch 2016.0) vs VSX (generally J2000) introduces a proper-motion tail to ~1-2 arcsec.
The true-match term is therefore a **two-component Rayleigh mixture** (``w``, ``sigma_narrow``,
``sigma_broad``) plus measured ``rho`` and fitted ``Q``. The <=1% contamination gate
(``mean(rho * pi * (sep/3600)^2)``) was **never relaxed** in this arc.

**Acceptance rule (MATCHER-FIX-3).** Acceptance is defined solely by the pre-registered 1%
contamination budget, not by a reliability threshold. The acceptance radius follows from measured
field density:

``r_max = sqrt(0.01 / (pi * rho))``  [``r`` in deg; ``rho`` in deg^-2]

A Gaia source is a **candidate** when it lies within ``r_max`` of the VSX position. Exactly one
candidate is accepted; when several lie inside ``r_max``, the mixture-based Sutherland likelihood
ratio **ranks** them and the best is taken. Reliability is still computed per row for
``gaia_match_quality`` tiers but **must not reject** a positional candidate. This mirrors the Gaia
DR3 cross-match design: the figure of merit ranks neighbours; positional compatibility gates them.
An arbitrary FoM cutoff was explicitly avoided by the Gaia authors because it would be arbitrary.

**PM path.** When ``pmra``/``pmdec`` columns are present and finite in the local Gaia DB,
positions propagate from ``GAIA_EPOCH=2016.0`` to ``VSX_MATCH_EPOCH=2000.0`` (J2000 assumption).
When absent, ``pm_path=broadened`` with a small quadrature term; separation quantiles before/after
are logged.

**Degeneracy guard.** ``sigma_broad > 0.25/sqrt(rho)`` [deg] plus astrometric-tail checks on the
sub-10 arcsec core (chance-scale p50 lock-on). At plan-time match, degeneracy is **WARN** only
(ranking may degrade; acceptance via ``r_max`` is unaffected). Strict FAIL remains in unit tests
on direct mixture fit.

**Outcome check (G3).** WARN when ``masterstars_accepted / masterstars_eligible < 80%``, where
``masterstars_eligible`` counts VSX rows with any DAO ``source_id`` candidate within ``r_max``.

**Comparison baselines (plan regen).** Withdrawn: frozen 245-row anchor VT histogram
``masterstars=178, gaia_dr3_direct=64, no_match=1, masterstars_exo=2`` (14.5 mag limit era).
Correct like-for-like: ``draft_000450`` regen on same night/field without mag limit:
``873 rows | masterstars 283 | gaia_dr3_direct 443 | no_match 147``.

**Status:** implemented (uncommitted pending Milan STOP clearance).

### IDENTITY-GATE-WRONG-STAR-NAMING (2026-07-28, GOLDEN-ASSET-RECUT audit)

**Finding.** Before the Phase 0 identity gate, four ``gaia_dr3_direct`` targets received light curves
named after **neighbouring** Gaia IDs (matched masterstar ``catalog_id`` adopted instead of the
variable's). The stale P1 mini files are the clearest artefact:

| VSX / variable Gaia target | Stale LC filename Gaia ID |
|----------------------------|---------------------------|
| 1499883638682689920 | 1499883638682689408 |
| 1500410236033012352 | 1500410613990135296 |
| 1498513166158147968 | 1498513269237363456 |
| 1499064433800590592 | 1499064399440851968 |

Those files are light curves of neighbouring stars, not the variables they were created for.
Independent confirmation of the identity gate from filename geometry, not only from active-count
arithmetic. Recorded in ``VL-ANCHOR-WCSINV.identity_gate_dropped_targets``.

**Status:** recorded; gate fix shipped in PHASE0-IDENTITY-GATE CLOSE-2.

### OSC-CHANNEL-EXTRACTION (2026-07-22, phase 1/3)
OSC Bayer mosaics: calibrate on CFA (dark subtract + flat divide on raw mosaic; flat via
``normalize_flat_master`` per-tile using ``EQUIPMENTS.BAYERMASK``). **No** demosaic/interpolation.
After calibration, plane-split extraction to four obs-groups (``oneRGGB``, ``R``, ``G``, ``B``) with
AVERAGE semantics; optional ``osc_channel_binning`` NxN average post-extraction. Sky-surface fit runs
**only** on extracted channels, never on the mosaic. ``oneRGGB`` is internal/diagnostics only (not
AAVSO). Effective gain/RN stamped per channel (``VY_EGAIN``, ``VY_RDNOIS``). Wired gate **OSC-01**
(no mosaic without ``VY_CHANNEL`` in alignment). Literature: AAVSO DSLR Observing Manual (channel
separation); Carrasco/Riello Johnson transforms deferred to OSC-3.
**Status:** phase 1 implemented; see OSC-WCS-SOLVE-ONCE for phase 2; OSC-3 (band mapping,
TG/TB/TR exports) queued.

### OSC-WCS-SOLVE-ONCE (2026-07-22, phase 2/3)
OSC multi-band drafts: **unified frame set** across oneRGGB/R/G/B (QC verdict replicated from
oneRGGB with per-channel diagnostics preserved; ``qc_source=oneRGGB``). Plate-solve runs **once**
on the oneRGGB MASTERSTAR; WCS is propagated to R/G/B masterstars and aligned frames. Registration
transforms are computed on oneRGGB only and stored in ``platesolve/<oneRGGB>/osc_registration_handoff.json``
for verbatim reuse on sibling channel frames (D2 artifact). Channel MASTERSTAR photometry (DAO
catalog, comp selection) uses each channel's own pixels; only geometry/WCS is shared (D4). Per-channel
DAO-Gaia match rate logged after propagation; below ``MASTERSTAR_PLATESOLVE_MIN_MATCH_RATE`` (0.60,
same gate as ``generate_masterstar_and_catalog``) -> WARN with channel numbers; FAIL only if
oneRGGB itself failed (B channel fewer stars expected). Band tokens: R->TR, G->TG, B->TB,
oneRGGB->CLEAR (k'' none-token path). Known systematic: Bayer channel
planes sit at sub-cell offsets (<=0.5 raw px = 0.25 superpixel); accepted for catalog crossmatch
tolerances - no per-channel WCS micro-shift. Wired gate **OSC-02** (identical frame ID sets).
**Status:** phase 2 implemented; see OSC-BAND-EXPORT for phase 3.

### OSC-BAND-EXPORT (2026-07-22, phase 3/3)

E1. AAVSO band codes for OSC DSLR tri-color (untransformed): R->TR, G->TG, B->TB per
AAVSO practice. ``oneRGGB`` is **internal-only** (LC + report diagnostics); hard excluded
from AAVSO and VarAstro exports (mixed-channel mags undefined).

E2. Comp/check catalog magnitudes per band: TG ensemble reduces against Johnson V comps,
TB against B, TR against Cousins R_C (AAVSO comp-mag rule).

E3. Gaia DR3 G + BP-RP -> Johnson/Cousins via ``src_py/gaia_johnson.py``. **Coefficient
source:** Gaia DR3 documentation CU5 Table 5.9 (GBP-GRP polynomials for G-V, G-B, G-R;
sigma column 0.03017 / 0.0633 / 0.03167 mag). **Validation reference:** Ruelas-Mayorga
et al. 2025 RASTI 4:37 (doi:10.1093/rasti/rzaf037; Landolt-standard independent fit,
Crossref-verified author list). Validity 8<G<16, BP-RP in pinned range. Out-of-validity
comps excluded with logged reason; Table 5.9 sigma added in quadrature. Riello et al.
2021 EDR3 Table C.2-style fallback in unit tests only.

E4. Jordi B-V legacy path (mono k2 slope) **unchanged**; ``gaia_johnson`` consumed only by
OSC band exports in this phase.

E5. No new config parameters. Wired gate **OSC-03** (no oneRGGB in export files).

**Status:** phase 3 implemented; M71 E2E acceptance pending Milan.

## Comp-star selection & QA

### Selection priority: stability > colour > proximity; proximity is a GATE, not a rank
- **Stability** is enforced as a hard `max_comp_rms` cap + iterative MAD filter, and as the
  Broeg (2005) `1/sigma^2` ensemble weight - which dominates the actual measurement.
- **Colour** (`|DeltaBP-RP|`) is a hard cut (<= 0.79) + tier, and is the **primary pick sort key**.
  For **NoFilter/broadband** this is justified: without a filter the colour term is a
  *first-order* systematic (no bandpass to cancel it), and colour systematics don't average
  down like random scatter. Grounded in Henden & Kaitchuck (1982) and the AAVSO CCD Guide.
- **Proximity** is enforced only as the distance gate (`max_dist_deg` ~ 1.5 deg, `min_dist_arcsec`
  60 arcsec). A proximity ranking tie-break was implemented and **reverted**: because Broeg `1/sigma^2`
  weights are order-independent, any RMS-bin tie-break necessarily trades stability for
  proximity, violating the agreed order. **Status: settled 2026-06-02** (revert restored the
  comp set byte-for-byte).

### comp_qa - Sokolovsky leave-one-out self-consistency QA
Per-comp variability indices on the zero-median LOO differential light curve: `sigma_IQR`
(amplitude), `1/eta` von Neumann (slow drift), `spike` (dropouts), flagged against a
**magnitude-dependent locus** (0.5-mag bins) rather than a flat floor. Grounded in Broeg (2005)
+ Sokolovsky et al. (2017) + Howell (1989). The flat-floor v1 over-flagged faint comps; the
locus version is correct. **Status: productionized 2026-06-02** as a read-only post-Phase-2A
stage (photometry byte-identical); outputs per-comp flags + per-target `n_clean`.

### Crowding classifier: gated infra, NOT enabled on the wide rig
Detection-independent signals (Gaia density, `blend_frac` at depth, comp availability) replace
the erratic stars/Mpx class; LOOSEN on comp scarcity, TIGHTEN on real blend fraction, with a
**sampling gate (FWHM >= 3 px)**. The wide rig is floor-limited (scintillation/undersampling at
FWHM ~ 2.6 px), so tightening there only cuts good comps. **Decision: keep OFF for the wide rig;
enable on the well-sampled Newton cluster.**

## Trust & validation

### Production trust gate - hard/soft model, inform-only (v1)

Per-target GREEN/YELLOW/RED from **comp-set quality** (`n_clean` from comp_qa Sokolovsky LOO),
**check-star scatter**, and **`lc_quality_flag`**. **No SEP / xval axis in production** - SEP vs
DAO cross-validation is an offline harness only (`xval_run.py`, `assign_sep_confidence` in
`xval_harness_core.py`); see Cross-validation CLOSED below.

**Hard** warnings (real red flags: `n_clean == 0`, no check star, bad `lc_quality` e.g. saturated,
check >= 0.05) vs **soft** (thin comps, sparse_fallback, marginal check 0.02-0.05, short_baseline).
RED = any hard OR >= 3 escalating soft; YELLOW = any soft; GREEN = `n_clean >= strong` + check OK +
no warnings. Thresholds derive from `comp_trust_min_comps` / `phase01_comparison_n_comp_max`.
**Unevaluated targets -> RED** (fail-closed). **`lc_quality="noisy"` is informational only** -
variability-driven (counting it as a hard warning wrongly demoted real variables). **v1 is
inform-only** (RED is surfaced, not auto-dropped from exports). **Status: shipped 2026-06-02;
SEP axis removed from production 2026-06-03** (trust gate v2 - supersedes the earlier harness-era
gate that also read `xval_results.csv` / `sep_confidence`).

### Cross-validation CLOSED (aperture path); SEP is the independent witness
draft_000365 triple-validated (photutils + sep + dao): the science number reproduces to ~1 %,
and **sep matches VYVAR extraction to 0.2 %/frame** (a SExtractor mesh-background pipeline ==
VYVAR aperture photometry). photutils-annulus inflates on crowded/faint targets and is NOT a
reliable independent witness; sep is. VYVAR `lc_rms` is consistent with and slightly
conservative vs the raw differential floor (never under-reports noise). **IRAF/PyRAF closed as
unnecessary** (no independent axis; not feasible on Py3.12/Ubuntu24). **PSF cross-val deferred**
to a PSF-heavy/faint draft. The `xval_run.py` harness is validated and reusable.

### In-pipeline `sep_xval` stage retired; trust gate re-anchored on comp-stability (2026-06-03)
The draft-level production stage (`sep_xval_core`, `sep_xval_*` config, per-target
`sep_confidence` in `photometry_summary.csv`) is **removed**. Rationale: the validated
independent witness (SEP via `xval_run.py`) remains available offline; running a second full
extraction pass on every production draft duplicated harness work without adding a distinct
trust axis once comp_qa (Sokolovsky LOO + magnitude locus) is productionized. The **trust gate
v2** uses only: comp health (`n_clean` from comp_qa), check-star scatter, and
`lc_quality_flag`. **`lc_quality="noisy"` stays informational only.** Runtime citations are driven by
`comp_qa` / trust / photometry paths only (no SEP axis). SEP/SExtractor entries remain in
`CITATIONS.bib` for the offline `xval_run.py` harness. Historical cross-val rationale above is
unchanged. Harness helpers live in `xval_harness_core.py` (shared with `xval_run.py`).

## Calibration & parameters

### Single authoritative parameter resolver (provenance)
`param_resolver.py`: equipment-intrinsic (gain/RN/pixel/focal/saturation) = **DB(valid) ->
header(cross-check warn) -> config**; observation-specific = header -> DB -> config; site
(lat/lon/elev) = **per-draft `ID_LOCATION` -> header -> config (flagged, never silent)**. BJD /
airmass are now **config-independent** (derive from the draft's own site). `config.json`
`observer_location` is UI / last-session state only - moot for the science. **Status: settled
2026-05-30.** Closes TODO-GEO (ROADMAP 2026-06-09).

## Catalogs

### Retire APASS/Tycho B-V -> pure Gaia BP-RP
APASS/Tycho B-V is reached only via the last-resort `lookup_bv_from_local_db` fallback; no
production algorithm needs it (colour term + tiers are already BP-RP-native; ~3 % of targets
on 362). **Stage 1 done** (fallback disconnected in 4 callers; determinism verified - the
earlier ~45 % comp diff was run-context drift, not the disconnect). **Recommended scope = A+B**
(also retire all Johnson B-V dual-mode).

**B-V A+B executed (2026-06-03):** Johnson B-V retired; comp tiering and hard colour filter
are **Gaia BP-RP only**. Targets without `bp_rp` -> T4 / magnitude-proxy (accepted minority).
`lookup_bv_from_local_db`, `bp_rp_to_bv`, `teff_to_bv`, dual-mode config (`phase01_use_bprp_primary`,
`*_bv_limit`, `phase01_tier*_bv`) removed from production. `VSX/vsx_make.py` builds VSX-only;
regenerate `vyvar_vsx_local.db` on the catalog machine (see JOURNAL). Stages 2-4 complete.

## Reporting & export

### AAVSO / VarAstro correctness
MTYPE = **STD** with `TRANS=NO` (every prior file was mislabeled DIFF); table-driven FILT map
with `#WARNING` on unknown filters (no silent CV); honest `meth=` label; **KMAG = measured
ensemble-standardized check-star magnitude** (per-row sidecar; check star excluded from its own
ensemble). **Routing: eclipsing -> VarAstro (LC); pulsating/all -> AAVSO.**

### Citations: `CITATIONS.bib` is the single source of truth
One conditional emitter (`citations.py`) shared by the AAVSO export, VarAstro export, and PDF
Methods - cites **only methods that actually ran**. CORE (always) + a gated **DATA-QUALITY
GATE** section (Sokolovsky / von Neumann when comp_qa/trust on). SEP/SExtractor citations are
offline-harness only (see *In-pipeline sep_xval retired*, 2026-06-03). Comp-selection rationale cited via Broeg, Henden & Kaitchuck, AAVSO CCD
Guide. **Status: settled 2026-06-02.** Runtime DATA-QUALITY GATE cites Sokolovsky / von Neumann
when comp_qa/trust run; SEP citations are harness-only after 2026-06-03 (see DECISIONS:
*In-pipeline sep_xval retired*).

### PDF: R1 overflow guarantee
Wrapping `Paragraph` + pagination + layout guard -> **0 overflow violations**. Aperture-only
default output is byte-stable. Trust badge + per-method overlays are additive and must preserve
the 0-overflow guarantee.

## Strategic

### Comet photometry - feasible, but a future parallel phase (do NOT start yet)
Architecture is sound and reuses the front-end (calibrate -> platesolve -> star-stack -> Gaia ZP),
then forks into comet-rate stacking + extended coma photometry + ICQ/COBS export. Mature tools
exist (KOPR, Tycho-Tracker, Comphot); VYVAR's value is workflow integration + a Gaia zeropoint,
not novel science. **Decision: analysis only; start only after the variable-star pipeline is
finished.** The B-V/APASS removal does not block it (Gaia->V Riello gives V-equivalent comps).

### Brand / paper title - locked
The name **VYVAR** is final. Working title: *VYVAR: An Automated Differential Photometry
Pipeline for Amateur Variable Star Observers* (PASP / AN).

### APCORR-COLOR - extrapolation hard-block; NoFilter CT still off (2026-06-03)
**Prototype (draft_000366, NoFilter):** `VYVAR_CT_PROTOTYPE=1` measured would-be CT without
changing production LCs. Findings supersede earlier roadmap estimates (`c1 ~ -1.0`,
cat-inst ~0.12-0.16 mag): median c1 ~ -0.07 (all) / -0.36 (nonzero) / -0.53 (gate-passers);
median |ct_corr| ~ 0.019 mag (p90 ~ 0.5 mag); cat-inst scatter 0.078->0.053 mag; only ~11%
pass numeric gates; worst cases up to ~4.8 mag on extrapolated red targets. **NoFilter CT
enable remains parked.**

**Correctness fix (all filters):** `_check_color_term_extrapolation` now **returns False**
when target BP-RP lies outside the comp BP-RP range (+- `phase01_ct_extrapolation_tol`, default
0). Call site skips `apply_color_term` (`ct_ok=False`, uncorrected fallback) instead of
warn-only. Targets are never dropped or NaN'd. `should_apply_color_term` NoFilter skip unchanged.

## Colour term - validation & production (2026-06-03)

### In-range CT apply path validated (machinery + science-grade)
Machinery-grade on M67 Blue (astro-RGB); science-grade on h & chi Per photometric B/V/Rc.
**Acceptance evidence = comp-scatter reduction scaling with `|c1|` + physical `|c1|` ordering
(Rc<V<B) + `stderr_ratio` <= 0.5** - not `|c1|` magnitude.

### Retraction: "photometric -> small `|c1|<<1`" expectation is wrong
c1 is relative to Gaia G; a large B term (~1) is physical. Validate via fit quality + scatter
reduction + ordering, not absolute c1 size.

### `phase01_ct_min_comp = 7` default retained
Do **not** flip on single-field evidence. M67 Green favoured lower; h & chi Per (n_comp~140) shows
the gate is moot in rich fields. Settle via cross-field experiment. `stderr_ratio <= 0.5` is the
real quality guard.

### Exposure-merge not adopted as a CT fix
Refuted on M67 - degrades c1 stderr. Same-filter/different-exposure stays exposure-aware for the
c1 fit.

### Red CT on M67 is data-limited (saturation)
Shorter Red exposures are the lever, not algorithm changes.

### Colour term decoupled from target selection (production)
Colour term is an applied-correction **toggle** (`apply_color_term`: auto/on/off). Photometry
always runs the full VSX field for every filter. `VYVAR_CT_PROTOTYPE` presel is an opt-in
validation mode ONLY, never the production path.

### Non-cal mode: no `calibrated/` directory
Frames live in `non_calibrated/lights/`; all consumers read via one source root.
`calibration_mode=pre_calibrated` recorded end-to-end.

### Conceptual (science output): G-referenced magnitudes
VYVAR colour terms are relative to Gaia G -> corrected magnitude is G-referenced, not standard
Johnson/Sloan. AAVSO-standard B/V/Rc requires a standard catalog (APASS) or a documented
G->standard transform - to resolve before science submission.

### trust_flag_core Phase E (2026-06-08)

- **Finding A:** summary/export targets absent from `trust_map` default to **RED** with reason
  `not evaluated (no comp QA / missing from trust map)`; `LOGGER.warning` when any summary id
  is missing. Conservative mission-safe default (was GREEN).
- **Finding B:** `classify_warnings` adds soft note `no check-star verification available` when
  `check_scatter` is nan; can shift GREEN->YELLOW when no other warnings. Max 2 soft preserved.
  **2026-06-10 (draft_382 check-star audit):** 15 hard-RED check-star targets on 12-15 frame
  sessions are not genuine variables (8 crowding-blend, 4 short-baseline-outlier, 2 metric-mismatch,
  1 thin-pool). Follow-ups CS-1..4 logged in ROADMAP (frame-blind 0.05 gate, select/gate metric
  mismatch, ensemble-exclusion gap, crowding caveat) - record only; not fixed with #3.
- **Finding C (C1 chosen):** keep `np.nanstd(km)` ddof=0; 0.02/0.05 thresholds calibrated to
  population std. Revisit ddof+threshold co-calibration on ROADMAP (not this pass).
- **Finding D:** `len(soft) >= 3 -> RED` kept as forward guard (today max 2 soft).
- **Finding E:** deferred -- lc_quality-missing soft note (would make D reachable). **2026-06-10
  (rev b):** `short_baseline` is a **non-escalating** soft (excluded from `len(soft)>=3 -> RED`);
  Finding E **stays OPEN** -- not the third escalating soft source. See
  `VYVAR_LC_QUALITY_SHORT_BASELINE_SPEC.md`.
- **draft_000366 trust re-run:** 10 GREEN->YELLOW, 0 GREEN->RED; numeric LC/comp_quality unchanged.

### B905 zip strict policy (2026-06-08, Phase D)

`strict=True` only where paired iterables are equal-length by construction (parallel
per-frame arrays, pairwise boundaries, same-length Series `.tolist()` pairs).
`strict=False` where ragged length is intentional (`.get(col, pd.Series())` fallbacks,
cross-DataFrame UI zips) or on untested display code. `strict=False` preserves today's
truncate-to-shortest behavior; `strict=True` adds a defensive length assertion only.

### comp_qa_core CQ-C - fix-once magnitude locus (2026-06-09, Phase F)

The comp QA magnitude locus was rebuilt from an accumulating `dropped_global`, coupling per-target
flag thresholds to target processing order (circular: drops shaped the locus that shaped drops).
**Decision:** use the **fix-once** pass-1 locus (`build_locus` over the full pass-1 pool) for all
per-comp `locus_at` / spike / flag evaluation; `dropped_global` remains for survivor bookkeeping only.

**Validation (draft_000366):** order-independence PASS (>=5 shuffled target orders -> byte-identical
QA payload). Bounded diff vs iterative locus: **1** comp flag flip, **1** target `n_clean` +1,
**0** trust-label changes (borderline only). `lightcurve_*.csv`, `comp_quality_*.json`, and
`comparison_stars_per_target.csv` unchanged.

**SHA transition:** core photometry subset (283 files) stays **`770966c3...`**; reference baseline
expanded to include `comp_qa_*.json` sidecars (426 files) -> **`edbd97e7...`** (intentional
CQ-C re-baseline, not a photometry regression). Sibling: ddof+threshold co-calibration (ROADMAP).

### Gaia DR3 catalog ingest -- GAIA-3 Riello G correction (2026-06-10)

DR3 `phot_g_mean_mag` already includes the Riello et al. 2021 milli-mag correction for
6-param and 2-param solutions; **do not** re-apply. Prior "missing correction" concern closed.
See `VYVAR_GAIA_DR3_AUDIT.md` (GAIA-3).

### Gaia audit GAIA-1 / GAIA-2 deferred to DR4 (2026-06-10)

`pmra`/`pmdec` (PM propagation) and `ruwe`/`duplicated_source` (astrometric-quality filter) will
be added in the **Gaia DR4** catalog build (DR4 ~Dec 2026, ref epoch J2017.5), not by restarting
the in-progress DR3 rebuild. The DR3 build completes as-is on the existing schema.

**Rationale.** Gaia DR4 requires a fresh full-sky build regardless; restarting a ~50 h DR3 build
for an interim catalog superseded within ~6 months is not worth the sunk cost.

**Accepted interim risk (until DR4).** Platesolver PM propagation (`_apply_proper_motion`,
`GAIA_EPOCH = 2016.0`) stays a no-op against the DR3 catalog; no `ruwe`-based comp filtering.
Wide rig (~9.77"/px): negligible. Fine rig (~0.65"/px) in dense fields: GAIA-1 mis-association
risk remains **unmitigated** -- treat fine-scale dense-field reference magnitudes with this caveat
until DR4.

**DR4 migration hooks (act at DR4 build time):**
1. Reference epoch J2016.0 (DR3) -> **J2017.5** (DR4): `GAIA_EPOCH` at `vyvar_platesolver.py:63`
   must update; prefer sourcing from catalog metadata.
2. DR4 `build_gaia_catalog.py`: SELECT + `_ROW_COLUMNS` + `init_db` + INSERT must include
   `pmra`, `pmdec`, `ruwe` (+ optional `duplicated_source`); downstream already tolerates them
   (`database.py` :210-212).
3. Re-verify lite-table column availability per DR4 datamodel.
4. DR4 ~2.5B sources with reliability split; G <= 16.5 cut keeps VYVAR in the reliable subset.

(GAIA-3 already closed: G-band correction baked into DR3 values; do not re-apply.)
See `VYVAR_GAIA_DR3_AUDIT.md`.

### Short-baseline LC quality `short_baseline` (#3, 2026-06-10, spec ready)

New terminal `lc_quality` class for `[lc_quality_short_min_frames, lc_quality_min_frames)` with
OK normal fraction. Defaults: short=**3**, min=**20** (LPV/Mira few-frame nights submittable).
Terminal (no noisy/good sub-verdict); YELLOW trust; **exportable** to AAVSO; **excluded** from
`len(soft)>=3` RED escalation. Implementation: `VYVAR_LC_QUALITY_SHORT_BASELINE_SPEC.md`.
Follow-up: vsx_type-aware frame thresholds (out of scope).

### Comp selection - proximity tie-break reverted (2026-06-08)

`dist_score` removed from `comp_selection_per_target.py`: the proximity tie-break was
deliberately reverted (Broeg 2005; Henden & Kaitchuck 1982; AAVSO CCD Guide) - proximity
belongs as a **gate**, not a ranking criterion - so the orphaned local and its
`optimal_dist_arcsec` helper were removed. Final comp ranking sorts by `comp_rms`
(`out.sort_values(["comp_rms", "catalog_id"], ...)`).

## Blind plate-solve - rig prior (2026-06-04)

Telescope + camera are always known in VYVAR -> **plate scale and FOV are legitimate priors**, not
unknowns to search (ASTAP / astrometry.net `--scale` model). `blind_use_rig_prior=True` (default)
enforces: (1) pre-vote `L3_image/L3_catalog` ratio gate, (2) post-fit WCS scale consistency in verify,
(3) FOV-derived central selection and triangle size caps (not index `log_L3_max`), (4) gnomonic
triangle sides when FOV >= 2 deg. Full scale-blind mode remains via `blind_use_rig_prior=False`. Index
**series** (`fine` / `wide` density tiers, all mag14) selects tier from known scale.

## Trust / comp QA - Chi_and_H diagnostic (2026-06-04)

- **CT result locked:** reproduced on clean fresh run draft_380 (B -1.08, V -0.38, Rc -0.02);
  CT-toggle/decoupling verified end-to-end on all filters (~371 targets, 0 "nan", comps + check-star LCs).
- **n_clean=0 / trust RED on Chi_and_H is draft-specific plumbing, NOT a cleaning-gate regression**
  (draft_366 baseline reproduces original n_clean/trust with current code). Root cause: hardcoded
  `proc_*_Light_*.csv` glob in `load_proc_pivot` - the **pre-cal-naming class** again. The fix belongs in
  a **single canonical pre-cal proc-CSV resolution**, not a one-off per consumer.

## Chi_and_H catalog policy - zaloha-only (2026-06-11)

**Adopted:** `chiandh_night_run_bvr.py` and the anchor recipe use **only** paths from
`config.json` pointing at `GAIA_DR3/zaloha/` (G<=16) + zaloha blind PKLs. **No field DB, no
TAP, no astroquery** in the Chi_and_H night-run path. `build_gaia_catalog.py` adaptive-split
remains DEFERRED until the next full-sky build.

**Retired anchors:** `d246a5be` / `30a2f461` (draft_382 TAP G<=19.5); `f4bcc0ee` / `bd0b1792`
(draft_385 truncated photometry / false success).

## Confirm-reproducibility-before-locking (2026-06-11)

Standing discipline: **two independent fresh runs must be byte-identical** on photometry SHA
before recording a new anchor (`draft_386 == draft_387` for the current cut). Record SHAs,
recipe, and `git rev-parse HEAD` in STATE/JOURNAL. Trust/QA changes must re-verify photometry
SHA unchanged; trust counts may move (intended).

## Night-run completeness gate (2026-06-11)

`night_run.audit_photometry_completeness` fails `night_run_success` when any setup's
`photometry_summary` covers <90% of `active_targets`. Guards the silent-truncation-as-success
class (draft_385, draft_383).

## Trust / check-star correctness (2026-06-11)

**Findings A/B closed** (`VYVAR_TRUST_CHECKSTAR_HARDENING_SPEC.md`): un-evaluated trust -> RED;
`check_star_min_epochs=5`; `check_star_scatter` uses `ddof=1`. **Finding E re-checked:**
`short_baseline` remains non-escalating YELLOW.

**CS-3 (circular check):** Phase-2A check-star selection used column-based ensemble exclusion
that was dead code - on draft_387 ~97% of selected checks were still ensemble members. Fix:
`ensemble_member_ids()` + `select_check_star(..., ensemble_ids=...)`. Spec:
`VYVAR_CHECKSTAR_SELECTION_SPEC.md`.

**CS-2:** `check_select_rms_floor` guards artefact `comp_rms~0` rankings. **CS-4:** drop
candidates with `contamination_idx > aperture_correction_max_contamination` when column present.

**Reserved check-star (hold-one-out by design):** PARKED - would change which stars enter the
ensemble and **move the photometry anchor**; requires explicit re-cut if ever adopted.

## Broad-except regression guard (2026-06-11)

**Enforced:** ruff `BLE001` + `E722` in `pyproject.toml` `select`; `.pre-commit-config.yaml`;
`tests/test_ble001_regression.py`. Existing sites grandfathered with `# noqa: BLE001` (168 added);
4 bare `except:` narrowed to `except Exception:`; 8 `photometry_core` parse paths narrowed.
Critical LC/completeness path reviewed - no silent swallow found.

## Comparison-star trust floor - ADOPTED (2026-06-11, Option B)

**Spec:** `VYVAR_COMP_FLOOR_POLICY_SPEC.md`. **Trust-only split** - byte-identity-neutral w.r.t.
photometry (anchor `203254fd` / `95a5515a` unchanged).

**Adopted:**

- `comp_trust_min_comps = 5` (config + Settings Data quality) - trust RED floor via `n_clean`.
- `phase01_comparison_n_comp_min` stays **3** (Phase-1 selection / ensemble unchanged).
- Trust `strong = min(comp_trust_min_comps + 2, phase01_comparison_n_comp_max)` -> **7** at defaults.
- `n_clean` 5-6 -> thin comp soft YELLOW; 3-4 -> RED.
- **Trust baseline on draft_387:** **1382 YELLOW / 106 RED** (floor-5; 1488 rows). Pre-floor-5
  1400/88 superseded.

**Not adopted (Option A):** raising Phase-1 selection floor to 5 - would move photometry SHA
(draft_387 footprint: 45 per-setup hits at 3-4 comps); anchor re-cut required if ever pursued.

**Literature rationale:** Broeg weighted ensemble; AAVSO ~12-20 good comps; robustness floor ~5+.
See spec for citations.

**Check-star coupling:** CS-3 left 60 independent checks on draft_387; reserved-check design still
parked (ROADMAP).

## Comp-slope stability on common-mode-removed residual (2026-06-11)

**Adopted (science-changing on long baselines):**

- **Common-mode detrend before slope test** (`check_comparison_stability`, default ON): fit a line
  to the per-frame median of active comp LCs, subtract for slope evaluation only - ensemble
  magnitudes unchanged. Formal basis: Honeycutt (1992) ensemble `em(e)` term; Broeg (2005)
  artificial-comparison framing; Sokolovsky (2017) indices judged vs the comp population.
- **BJD sort before `np.interp`** (B2): frame/proc order is not monotonic in BJD; unsorted xp
  corrupted the common-mode estimate (Step A: 97 vs 237 mmag/hr on DY Peg).
- **Significance gate** (B1): exclude on slope only if **both** `|slope| > comp_max_slope_mmag_hr`
  **and** `|slope|/stderr >= comp_slope_significance_k` (default 3.0) on the **post-detrend**
  residual. Large-but-insignificant slopes (noise / imperfect common-mode removal) are kept.
- **`comp_slope_significance_k`** in config + Settings + `VYVAR_PARAMS.md`.
- **Honeycutt 1992** citation emitted only when common-mode stability detrend runs
  (`pipeline_meta.common_mode_stability_detrend`); remains in VarAstro flux-sum line (Collins +
  Honeycutt combination).

**Thin fields unchanged:** when `n_good < n_comp_min`, slope/p2p flags -> `suspect` with
`kept: n_good<min` - ensemble membership unchanged. DY Peg (2 comps) stays RED via comp_qa
`n_clean` skip, not this change.

**Anchor footprint (`draft_000387`):** 12 frames/setup (<20 guard) -> detrend + slope paths never
exercised; **0** historical `slope=` comp notes; **LC byte-identical** on re-run expectation.
Re-baseline (Step D) waits for Milan acceptance of bounded diff on longer-baseline validation
(e.g. DY Peg `draft_000390`: slope notes removed, ensemble unchanged).

### Sparse-only comp fallback (2026-06-12)

**Decision:** Wholesale iterative comp clip rejected on anchor `draft_000387` (marginal churn, not net
precision). Ship the same CM-residual clip machinery as a **per-target sparse-only fallback** behind
`comp_sparse_fallback_enabled` (**default ON** from 2026-06-11 re-baseline lock; alias
`comp_iterative_clip_enabled`).

1. Run default a-priori selection unchanged; if `>= comp_sparse_fallback_min` comps -> stop.
2. Else if flag ON: generous masterstars pool (no global RMS pre-filter / no a-priori comp_rms gate),
   iterative leave-one-out 5sigma-MAD on CM-removed residuals; recover LC if `>= n_comp_min`.
3. Provenance: `comp_path`, funnel columns on `comparison_stars_per_target.csv`.
4. Trust: all `sparse_fallback` targets -> **YELLOW**; default-path targets unchanged.

**Anchor re-baseline (2026-06-11):** Three-way reconcile (`203254fd...` vs current code flag-OFF
class vs flag-ON) on in-SHA artefacts. Rich `draft_000387` has **0 default-starved targets** in
full photometry - fallback **inert** (all `comp_path=default`; 0 recovery LCs). Raw SHA moves to
`3f7c9e7a...` / `d5b72d08...` (two-run repro). Drift vs old cut is benign: `comp_path` provenance,
BJD/HJD ~1.9x10-^9 d, per-frame `err` QC recalc (~2.5x scale, mag/flux byte-identical). Accept via
`compare_photometry_science_meaningful` (PROCESS) - not raw `filecmp` vs `203254fd...`.

**Baseline comparison method:** raw byte SHA for lock/repro; science-meaningful tolerance gate for
regression vs prior anchor (provenance + QC excluded; BJD/HJD <=1e-6 d; mag/flux <=1e-6).

### Plate-solver: scoped robustness + Brno production fix (2026-06-14)

**Decision (locked):** Production defaults in `solve_wcs_with_local_gaia`:

| Flag | Default | Rationale |
|------|---------|-----------|
| `solver_use_cone_for_sip` | ON | SIP pass 2 rematches on **deep Gaia cone** (not triangle slice) |
| `solver_fits_header_hint_sep_escape` | ON | Verified-strong escape only (see below) - not match% alone |
| `solver_apply_roworder_yflip` | **OFF** | **Rejected** - regression gate: ~**320 px** home-rig star displacement (77% LCs broken) |
| `solver_legacy_masterstar_mirror_sweep` | ON | Single orientation resolver (home + Brno); mirror sweep retained |

**ROWORDER `BOTTOM-UP` Y-flip rejected** - anchor regression gate showed it displaces home-rig stars
~320 px. Kept **OFF**; legacy mirror sweep is the orientation resolver.

**Brno 83.1% match retracted as a target** - draft_399 / lower detection count (154 vs 250) artifact;
never production-validated on `generate_masterstar_and_catalog`. **Policy:** do not chase high match%;
pass an **overlay-confirmed** correct-but-distortion-limited solve at lower match when appropriate.

**Stale-hint cone recenter (real Brno blocker):** Gaia cone was built at `VY_TARG` while the linear
WCS center was **0.228 deg** off. When header hint vs solved center offset **>= 0.05 deg**, solver
re-queries Gaia at the **solved WCS center** and re-runs full-pair refit.

**hint_sep escape only on verified-strong solves:** cone recenter applied + **>= 75%** brightest-N
match + RMS **<= 2 px** (+ overlay confirmation for distortion-limited passes) - **never on match%
alone**.

**Anchor gate:** same-harness legacy-vs-scoped re-cut on `draft_000387`: **0 science failures**
(B) vs (A); B WCS **~0.003 px**. Re-cut vs archive alone is **not** a reliable gate (~2.26 mag B
harness drift, internally deterministic) - use **`sandbox/anchor387_legacy_vs_scoped_gate.py`**.

**Anchor re-baseline:** **3f7c9e7a (core) / d5b72d08 (full)** with sparse-only fallback default ON.
**Science-meaningful comparator** adopted (numeric tolerance on BJD/mag/flux; excludes provenance
columns).

**SIP guard:** `force_apply` on MASTERSTAR requires `rms_sip <= rms_linear`. Distortion-limited fields
may remain linear when SIP regresses.

**Equipment:** C5A-150M (id=4, 3.76 um), AZ800 (id=6, F=5480 mm) seeded in `initialize_database()`.

### Per-set astrometry fault isolation (2026-06-14)

**Decision:** In multi-group drafts, a plate-solve / MASTERSTAR failure in one filter/setup must **not**
abort astrometry for sibling sets or block photometry on sets that already produced catalogs.

**Mechanism:** `astrometry_align_and_build_masterstar` loops jobs with try/except; merges survivor
reports via `_merge_astrometry_group_reports`; attaches `skipped_subgroups` for failed setups.
All-fail still raises. **Single-group path unchanged** - one set, nothing to continue to.

**RUN VYVAR:** photometry stage hard-fails only when **no** set completed; partial success logs OK +
skipped/failed sets (including astrometry skips from `skipped_subgroups`).

**Fail-closed on skipped set:** exception before catalog / `per_frame_catalog_index.csv` write - no
half-written MASTERSTAR downstream.

**TASK 2 (shipped 2026-06-14):** catalog-recovery verification gate + hint-as-prior on MASTERSTAR.

**Accept gate (VERIFIED):** `catalog_recovery_tight >= masterstar_catalog_recovery_min` (default **0.65**),
`n_matched_tight >= masterstar_min_matched_floor` (default **40**), and distortion healthy
(`distortion_limited_benign` **or** `centre_rms <= masterstar_centre_rms_max_px`, default **1.20 px**).
Detection-denominated `_match_rate` / brightest-N remain **informational only**.

**hint_sep:** once VERIFIED, stale pointing offset is **`hint_sep_warn`** (non-fatal; PDF cover note via
`VY_HSWN`). Hard reject only when **not VERIFIED** and `hint_sep > max(1.5 deg, fov_diameter_deg)`.
Stacked FITS-header escape blocks (>=85% match + RMS <=2 px) **removed** - superseded by this rule.

**Distortion benign ratio:** edge/centre cap **2.50 -> 3.20** (`masterstar_distortion_benign_ratio_max`;
Brno `r` ratio ~3.0).

**Citations:** Lang et al. 2010 (Astrometry.net) emitted when catalog-recovery verification runs.

**Supersedes** hint_sep escape paragraphs above (>=75% brightest-N + RMS <=2 px widen) and TASK 2 blocked note.

### Plate-solver: scoped robustness lock (2026-06-14, superseded)

### Iterative ensemble-relative comp clip (2026-06-12, superseded by sparse-only fallback)

**Decision:** Retire binding a-priori `comp_rms` cuts (global pool pre-filter + per-target gate)
for sparse-field recovery. Replace with **generous candidate intake + iterative 5sigma-MAD clip on
CM-removed ensemble residuals** (Gilliland & Brown 1988; Broeg 2005; Honeycutt 1992 common-mode
detrend; Burdanov et al. 2014 / epsilon Indi 2020 practice; Everett & Howell 2001).

**Superseded:** wholesale flag - use sparse-only fallback above.

---

## Photometry math / simple differential (2026-06-15)

- ALG-3 comp temporal binning (`temporal_bin_comp_lc`) is incorrect for VYVAR's regime; **default
  OFF**. Proven root cause of non-home-set chaos (mechanism: per-frame common-mode breakage;
  corr(injection, transparency HF)=0.9995).
- Color term (c1) to be **dropped** in favor of color-matched comp selection (min |delta BP-RP|):
  removes the color systematic at source.
- Comp selection criterion = **min |delta(BP-RP)| + min RMS**; plain per-frame ensemble; no
  temporal binning, no color term, no complex weighting.
- Trust RED/YELLOW **temporarily disabled** during photometry tuning; to be re-derived on corrected
  numbers afterward.
- Legacy fields/anchors (h&chi Per, DY Peg, BO CVn) and old-SHA re-cut framing are **retired**; we
  are on the new catalog + new pkl.
- Fix mis-attribution: ALG-3 is **Hartley & Wilson 2023, MNRAS 526, 3482** (not Broeg-Bischoff &
  Dreizler) at docstring + config.py:452 + dev/results/config_schema.md:145 (archived), and the UI caption ("after
  ensemble" -> ALG-3 runs BEFORE ensemble).
- **Supersede** proposed `comp_color_window_bprp` param (PARAMS 2026-06-15): reuse existing
  tier ladder (`comp_tier1_bprp_limit` 0.15 -> tier2 0.30 -> tier3 0.55 -> cap
  `comp_max_delta_bprp` 0.79) in `_select_comps_by_color_then_rms`; no lone 0.2 step, no new key.
- Phase-1 comp rank artefact floor: **`comp_select_rms_floor` = 1e-6** (drop isolated_bin comps
  before RMS ranking; mirrors CS-2 `check_select_rms_floor` pattern at 1e-4).
- **Workstream A landed (2026-06-15):** dataclass + config.json defaults (`temporal_binning_enabled`
  False, `apply_color_term` off); Phase-1 routes through `_select_comps_by_color_then_rms` in
  `_assign_comp_tiers_to_pool`; tier load-clamp fixed (0.15/0.30/0.55 survive JSON). DoD-A PASS
  V0612 `delta_mag` 0.0113 / 0.949 / 7 comps.
- **Gate:** >=1 additional ground-truth field recommended before treating V0612-only as global
  default risk closure (Milan risk call).

---

## Decision-grounding rule (2026-06-15, ADOPTED - Milan)

Any design fork Claude brings to Milan must be grounded in physics/math, peer-reviewed literature,
or documented field practice. Bare engineering preference is not sufficient; no "recommended" label
without a cited basis. Grounding may supersede earlier recommendations. Method citations belong in
`CITATIONS.bib` at call sites when code changes land.

---

## Reporting-column fix - grounded synthesis (2026-06-15, supersedes B1/B2)

**Earlier B1/B2 framing withdrawn:** "guard the airmass detrend" treated a non-physical step as
load-bearing; not grounded in differential-photometry physics.

**Code audit (read-only, 2026-06-15):**

| function | file:line | finding |
|----------|-----------|---------|
| `airmass_detrend_lc` | *(removed)* | Least-squares fit **`mag = a.airmass + b` on the target's own curve**. **Not** a comp-derived extinction coefficient. **[2026-06-19 - fully removed]** wiring helper `_apply_airmass_detrend_helper` (T1-2) and functions `airmass_detrend_lc`/`airmass_detrend_lc_piecewise` (T1-7) deleted; per-target airmass detrend no longer runs (Phase-2A summary reports `am_detrended=False`), per the grounded fix below. |
| `detect_outliers` | `photometry_core.py:3323-3360` | Global median + MAD on all finite mags; **no VSX/feature mask**. Eclipse dimming -> `outlier_lo` (`mag > med + thr`, `:3354-3356`). V0612 DoD-A LC: **2x `outlier_lo`** (ingress). |
| `delta_mag` export | `save_lightcurve_csv` `:7594`, `:3814` | **Unchanged** by outlier/airmass stages; only `mag_calib*` columns are rewritten (`:7486-7496`). |
| Shape preservation | DoD-A LC `tmp/phase10/.../lightcurve_1111749368289526912.csv` | `corr(delta_mag, mag_calib_raw)` **0.998**; historical `corr(delta_mag, mag_calib)` **0.59** after target-fit airmass detrend (slope ~ **0.78** mag/airmass) - detrend path since removed. |

**Grounded fix (three parts):**

1. **Reported mag = validated differential + ensemble zero-point** (`delta_mag + ZP_ensemble` per
   frame from colour-matched comps; Honeycutt 1992 ensemble - already cited). For V0612,
   pre-detrend `mag_calib_raw` already matches `delta_mag` shape (corr 0.998); implementation must
   make that the shipping curve, not hope post-hoc guards salvage a target-fit detrend.
2. **Remove per-target airmass detrend from the variable reporting path** - redundant after
   colour-matched differential (Plavchan et al. arXiv:0704.3584; Dhillon PHY217); signal-absorbing
   when fitted to the target (confirmed above). Any residual extinction -> comp ensemble, not target LSQ.
3. **Mask-first known-variable guard on `detect_outliers`** - clip out-of-eclipse only; extend mask
   around ingress/egress (TESS subdwarf recipe arXiv:2402.16018; democratic detrender clips
   out-of-transit only - arXiv:2411.09753). Required regardless of (1-2).

**DoD-B (2026-06-15): PASS** - ``apply_reporting_postprocess``; V0612 ``mag_calib`` corr **0.958** /
pre **0.011** (was 0.57); ingress 24/24 ``normal``. Harness: ``tmp/phase11/dod_b_workstream_b.json``.

**Tier-2 (PARKED):** comp-ensemble-derived k for wide delta-airmass - ROADMAP.

---

## Canonical ensemble combination - A vs B resolved (2026-06-15)

**Decision-grounding:** Gauss-Markov / Broeg (2005) AN 326:134; SPECULOOS-South arXiv:2005.02423;
Howell (1989) sigma budget. Flux-sum equals inverse-variance weighting only in the photon-limited,
all-constant limit.

**Resolution (conditional, not taste):**

1. **Canonical science product = Broeg inverse-variance estimate** - *when* sigma is complete and
   error bars are validated (chi^2/dof ~ 1 on a constant star).
2. **`delta_mag` (flux-sum) retained as AIJ-validation / diagnostic column** (`tot_C_cnts` parity);
   not the primary science export once sigma is trusted. The ~0.002 corr gap vs ``mag_calib`` on
   V0612 is the expected weighting difference, not a bug.
3. **Load-bearing work = sigma budget** (photon + read + sky + scintillation + Broeg intrinsic
   inflation) - same machinery required for TODO-GS8 / TODO-MULTISET multi-rig combine.

### Read-only audit - current code vs Broeg-canonical (2026-06-15)

| Question | Finding | Anchor |
|----------|---------|--------|
| **1. What sigma feeds `ZP_weighted`?** | **Not** the per-frame Howell CCD ``err``. Weights use **night-level `comp_rms`** = RMS of **detrended relative flux** around 1.0 (dimensionless stability metric from Phase 1 / global pool), mapped into ``w = 1/rms^2 x tier_weight``. **No scintillation**; dark only via read-noise in the separate LC ``err`` column, not in weights. | ``comp_pool_rms.py:356-380``; ``comp_selection_per_target.py:1556``; ``ensemble_normalize`` ``:2437-2446``; ``_photometric_error`` ``:636-656`` (photon+sky+read only) |
| **2. Broeg iteration / variable comp inflation?** | **Partial.** ``pytics_iterative_weights`` (default **on**, ``config.json``) iteratively **inflates `comp_rms`** from per-comp residual scatter vs weighted ZP - Broeg-like, but on **stability RMS**, not per-frame photon sigma. ``check_comparison_stability`` MAD-filters high p2p comps (excludes/suspects), does not iteratively drop variables inside ``ensemble_normalize``. **Ensemble combination itself is flux-sum** (explicitly *not* Broeg-weighted - comment: 1/rms^2 deforms extinction slope). | ``:2409-2418``; ``pytics_iterative_weights`` ``:1821-1906``; ``check_comparison_stability`` ``:1914+`` |
| **3. Error bars validated (chi^2/dof ~ 1)?** | **No production gate.** LC ``err`` = Howell photon+sky+read per frame; **not** propagated into ensemble weights; **no** chi^2/dof check on constant stars in the Phase-2A export path (Mighell chi^2-gamma cited export-only per ``VYVAR_MATH_PHYS_AUDIT.md``). DoD-B constant gate used no-regression + RMS ratio, not chi^2. | ``:1428-1434``; ``VYVAR_MATH_PHYS_AUDIT.md`` Mighell row |

**Outcome:** sigma **incomplete** for Broeg-canonical ensemble combine -> **hold flux-sum for `delta_mag`**
(AI/diagnostic); **reporting `mag_calib` already uses partial Broeg (ZP offset only)**. Do **not**
promote inverse-variance **ensemble combine** until: (a) weights use validated per-frame sigma
(Howell + scintillation + inflation), (b) chi^2/dof ~ 1 on a constant calibrator. That sigma fix
is load-bearing for GS8/MULTISET regardless.

**Citations added (sandbox, 2026-06-15):** `young1967`, `osborn2015`, `dravins1998`,
`murray2020speculoos` in `CITATIONS.bib`. Spec: `docs/VYVAR_SIGMA_BUDGET_SPEC.md`;
sandbox: `tmp/phase12/`.

### Sigma-budget work item - chi^2 audit + sandbox (2026-06-15)

**Read-only chi^2 audit:** Mighell (1999) is **export-only** (`citations.py` PSF block); no
production chi^2/dof on constant stars. PSF `reduced_chi2` and trust `check_star_scatter` are
unrelated. **Verdict:** promote **new** reduced-chi^2/dof gate (not Mighell chi^2-gamma as-is).

**Sandbox shipped:** Osborn eq. (7) scintillation + Howell quadrature + Broeg inflation helpers;
chi^2 gate harness. **Not production** until chi^2/dof ~ 1 on verified-constant calibrator.
`delta_mag` unchanged.

---

## draft_409 trust/consistency cleanup - Fixes 1-3 (2026-06-16)

### Comp stability on ensemble residual (not raw `mag_inst`)

**Problem:** `check_comparison_stability` peak-to-peak on raw instrumental mag included
night-level common-mode drift (~0.35 mag), flagging all comps suspect despite GREEN LOO trust.

**Decision:** Assess stability on **per-frame ensemble residual** (median-subtracted differential
quantity) before optional common-mode detrend. Aligns comp QA labels with trust-line intent.

### Measured aperture + observed-band SNR sizing

**Decision:** PDF card and LC export report **measured** proc `aperture_r_px`, not Phase-2A replan.
SNR-opt sizing prefers observed-band catalog `mag` over Gaia G (`_APERTURE_SIZING_MAG_COLS`).

### `lc_rms (OOE)` for variables

**Decision:** Headline precision on variable target cards = **out-of-eclipse** scatter
(`lc_rms_ooe`, brightest tertile). Full undemeaned `lc_rms` retained but not the headline for
variables (eclipse-dominated otherwise).

**Validation:** draft_409 V0612 cross-validated vs SIPS - eclipse shape + single bright outlier at
~JD 2461200.385 match in both reductions (frame-level artifact, not VYVAR bug).

---

## Development process harness (2026-06-16)

**Decision:** Adopt JSON pass/fail validation ledger + session-start baseline check
(**DEV-PROCESS-A/B**), and load **CLAUDE_OPERATING_PRINCIPLES** at session init. Grounded in
Anthropic context-engineering / long-running-agent harness guidance (get your bearings, verify a
known-good baseline before new work, structured pass/fail ledger, leave a clean state).

**Charter:** `docs/VYVAR_CLAUDE_OPERATING_PRINCIPLES.md` -- session-init required read; governs
pre-claim checklist, diagnose-then-fix, and reproducibility (archive proc before re-proc; exact
19-digit source_id matching).

**Open (spec only, ROADMAP):**
- **DEV-PROCESS-A** -- JSON ledger of end-to-end validation items (`passes: true/false`); agents
  may edit status only; no deleting or weakening tests.
- **DEV-PROCESS-B** -- session-start script reproducing reference draft / regression baseline before
  new work (motivating case: lost V0612 proc).

Definition-of-Done discipline remains in `VYVAR_PROCESS.md`; old **TODO-DEV-PROCESS** folded/replaced
by A/B plus charter session-start checklist.

---

## Phase-1 comp degradation trust bands (2026-06-16)

**Decision:** sparse_fallback + check sidecar currently caps trust at **YELLOW** even when sparse comps
are unusable (degenerate / V0611 sparse `comp_rms` ~1.9-3.3 mag). An **absolute sanity ceiling** on
sparse comps is **deferred to Phase-2** as a conscious decision (ROADMAP item 1 after Phase-1b).

**Correction logged:** prior validation note "expected degenerate -> RED" was wrong for the current
trust model. Locked rule: sparse path + check present -> **YELLOW** (not RED), until Phase-2 ceiling
lands.

**Known issue (b):** per-target `comp_rms` gate not authoritative for N_good; RMS fallback + routing
fix is **Phase-1b** (immediate next), not part of the Phase-1 commit.

---

## Audit closure decisions 5-9 (2026-08-02, batch C + B-revised)

Milan approved recommendations from `docs/VYVAR_DECISION_BRIEF.md` and D5-2 mechanism from
`dev/results/CURSOR_RESULT_batch_B_revised.md`.

### 5 -- I-11 (register 21): Howell sky term on subtracted frames

**Decision:** **Option 1** -- carry pre-subtraction `sky_surface_bg_median_adu` into the Howell
sky Poisson term on the legacy fallback path.

**Why:** Sky Poisson noise is set by photons that arrived before subtraction, not by the
post-subtraction annulus residual (Howell 2006 CCD equation).

**Implement:** batch D (D.2). Confirming metric: crowded-field chi2_red rises toward correctness;
anchor unchanged (0 legacy-path epochs today).

### 6 -- I-04 (register 22): ensemble scatter on unmatched epochs

**Decision:** **Option 1** -- NaN + exclude epoch when ensemble scatter cannot be computed;
export flag required.

**Why:** Fail-optimistic zero substitution is unacceptable for publication-grade errors (Honeycutt
1992 propagate-or-drop rule).

**Implement:** batch D (D.1). Confirming metric: byte-identical anchor (0 unmatched epochs).

### 7 -- P-02 / A-6 (register 25): scintillation and systematic floor

**Decision:** **Option 3** -- (a) wire scintillation per rig from Young/Osborn formula using
per-epoch header airmass; (b) re-measure chi2_red; (c) if median chi2_red still > 1.2, add
per-rig `sigma_sys_mag` floor; **report floor and achieved chi2_red separately** (R8).

**Why:** Part 1c median chi2_red ~4.7; scintillation alone may not close gap on all rigs; Everett
& Howell (2001) 2-5 mmag systematic floor is literature-backed.

**Implement:** batch D (D.4). Confirming metric: median chi2_red within ~20% of 1.0 on >= 2 rigs.

### 8 -- T4-1 (register 10): detection noise on resampled frames

**Decision:** **Option B** -- single measured `N_equiv` correction so effective post-resample
threshold matches intent. **Confirm 3.78 vs 4.71** from Part 2b measured `rel_err` before re-cut.

**Why:** Correlated noise after astroalign (Casertano 2000; Fruchter & Hook 2002); fixed kernel
implies single rig-level factor.

**Implement:** batch E (E.4). Blocks anchor re-cut until applied.

### 9 -- D5-2 (register 16b): bright-end flux compression mechanism

**Decision:** Mechanism = **bright-end saturation / detector non-linearity** (G 8-9 bin, stars to
~97% full well). **Fix = saturation admission gate (C-1/C-2)** at **70%** of
`saturate_limit_adu_85pct` (to be validated in implementation). **Not** an in-code linearity curve.

**Why:** Production-column M1 (B-revised): G 10-13 slope -0.421; G 8-9 bin -0.258; brightest star
peak 54231 ADU (97.4% of 55705). Batch B B-open superseded (FITS instrument invalid).

**D1-2 alternative:** per-sensor linearity curve (Howell 2006 sec 4.4) **DEFERRED** -- requires
dome-flat ramp per sensor; observing-plan item, not code fabrication.

**Implement:** C-1/C-2 admission gate in batch E alongside existing admission-gate work.

---

## Decision 7 addendum -- wide-rig sigma_sys floor fit (2026-08-03, batch D GATE 1)

**Population:** Part 1c check-star harness, draft_435, equipment_id **1**, n=**162** fields.

**Median quoted err (post-scintillation):** **9.4 mmag**. Scintillation at X=1: **1.73 mmag**
(negligible in quadrature).

| stage | median chi2_red_clipped | notes |
|-------|-------------------------|-------|
| before batch D (no scint) | **3.55** | `tmp/batch_d_chi2_before.json` (prior session) |
| scintillation only | **3.55** | `tmp/batch_d_part1c_post_scint.json`; unchanged |
| scintillation + fitted floor | **~1.0** (simulated) | requires **~15 mmag** floor -- **not applied** |

**Fitted sigma_sys_mag (NOT applied):**

- From chi2 formula (`true_scatter = sqrt(chi2) x err`): **~14.7 mmag**
- From measured residual RMS (`scatter_mag` median 20.1 mmag): **~15.7 mmag**
- Constant-calibrator cohort (`dev/tools/batch_d_wide_floor_fit.py`, n=12): **8.33 mmag**
  (chi2_before ~0.99 on that subset; also outside 2-5 mmag sanity)

**R8 verdict:** Floor outside Everett & Howell (2001) **2-5 mmag** band on the check-star
population. **Do not apply** without Milan review -- likely photon/ensemble mis-scaling or
frame-correlated scatter not captured by a constant quadrature floor. Batch E **blocked** per
GATE 1 task rule.

**H1-global verdict (2026-08-04 diagnostic):** Part 1c check stars (n=162): median
scatter/quoted-err ratio **~1.96** (~2x); slope through origin **1.83** (T2). Median
scatter **20.1 mmag** vs quoted err **9.4 mmag**; chi2_red **3.55** unchanged. **Not H2**
(honest 20 mmag intrinsic scatter). **No floor applied.** Primary fix routed: audit
ensemble SEM + photon term propagation (post-batch-E numeric item). Secondary: comp QA
(high catalog comp_rms comps). See `dev/results/CURSOR_RESULT_wide_error_diag.md`.

**Fingerprints pushed (batch D re-cut #1):** core `b9c9489aa88b1df815bf6157911b35af5bb1c42a3b0eaf58995042fcdd007a39`
(n=325); extended `65bc826cac433453f689dbc5ab2883e783b7a7c7563092c02cfa443058f48cc2` (n=487).
Superseded: `b7f980c0...` / `2c43bbbf...`. Science compare PASS (162 LC; flux/mag/WCS unchanged).

## INV-MS-01-REMOVAL (2026-08-05)

**Decision:** Remove runtime invariant `INV-MS-01` (masterstar DAO_ONLY fraction WARN/FAIL gate).

**Rationale:** Thresholds (WARN 0.10 / FAIL 0.25) were calibrated on a single wide-rig
VYVAR-calibrated anchor (measured 0.0369). A legitimate Newton/eq4 pre-calibrated run
(draft_501) measures 0.417 under unchanged science goals. The metric is not portable across
rigs or calibration modes; re-tuning would only move the kill boundary.

**A2 finding:** FAIL raised `InvariantViolation`, which was swallowed by a broad handler that
also wrapped `_vyvar_df_to_csv` -- the annotated masterstars CSV was never written, causing
0-LC runs with a misleading log line. Fix: remove the gate; move CSV write outside the
annotate try block; retain fraction as informational census only.

**Detail:** `docs/VYVAR_LIMITATIONS.md` (INV-MS-01-REMOVED). Anchor regression retained in
`dev/tests/test_invariants_p2.py` fixture test. A-6 magnitude-split diagnostic **closed**
2026-08-07 (`docs/VYVAR_DAO_DETECTION.md`).

## DAO-ONLY-CLASSIFY-REPORT-NOT-FILTER (2026-08-07)

**Decision:** DAO_ONLY rows are **classified and reported**, not filtered at detection or
MASTERSTAR write time.

**Rationale:** The DAO-PHYS / A-6 campaign showed no detection-stage filter separates artifact
from astrophysical candidates without unacceptable depth loss. Magnitude-vs-Gaia-cap classes
(`artifact_negative`, `unmatched_in_range`, `ambiguous_depth`, `beyond_catalogue`, `indeterminate`)
plus `sigma_g_row` bands give operators visibility. Consumption remains gated by `snr50_ok` and
downstream photometry filters (`comp_selection_per_target.py:514-523`,
`variability_detector.py:101`). Class counts are installation-specific (local Gaia DB) and must
not become runtime gates.

**Reference:** `docs/VYVAR_DAO_DETECTION.md`, `src_py/dao_reconcile.py`, `dev/results/CURSOR_RESULT_dao_close.md`.

## SAT-DIAG - saturation and linearity limit gate (IMPLEMENTED - Milan authorized 2026-08-13)

**Status:** **IMPLEMENTED** (2026-08-13). Milan authorized implementation same date.
Module: `src_py/sat_diag.py`; invariant **INV-SAT-01** wired. Spec:
`dev/results/specs/VYVAR_SAT_DIAG_SPEC.md`. Report:
`dev/results/CURSOR_RESULT_sat_diag_implement.md`.
`dev/results/MEMO_saturation_limit_literature.md`,
`dev/results/CURSOR_RESULT_saturation_peak_reconcile.md`.

**Governing principle (architect):** the strength of the action must follow the
provenance of the number. Measured limits may exclude; derived or defaulted limits
warn only.

**Four decisions -- architect recommendations (2026-08-13, not yet authorized):**

1. **Interim limit source and CONFLICT policy** -- resolution order: header, then
   equipment row, then derived pile-up, then BITPIX container bound. Stated ceiling
   refuted when below max raw pixel (compatibility test). CONFLICT: adapt and
   continue loudly (`CONFLICT_DERIVED`); fail closed only when nothing stated,
   nothing derived, no BITPIX bound. No-pile-up: container bound only; never
   brightest-star ceiling.
2. **Target structure** -- two levels keyed by `(equipment, readmode, XBINNING,
   YBINNING)`; DB row is hint not authority. Columns: `sat_adu`, `lin_adu`,
   `lin_source`, `sat_source`, `measured_utc`, `tolerance_pct`. Migration: null out
   `EQUIPMENTS.SATURATE_ADU=16384`; do not carry forward.
3. **Exposure ramp** -- proceed with `DEFAULT_FRAC=0.85` + mandatory WARN; ramp
   when convenient (AAVSO procedure, per rig config). `DEFAULT_FRAC` must not
   exclude (Decision 4 Tier 3).
4. **Consumer policies** -- three tiers keyed on provenance: (1) hard saturation
   from MEASURED/HEADER/DERIVED may exclude pool/AC/PSF; (2) MEASURED linearity
   may exclude pool/AC; (3) DEFAULT_FRAC or DERIVED linearity warn only. Exclusion
   decided once per draft (`INV-COMP-MEMBERSHIP`); target epochs flagged not dropped.

**Measured consequence (2026-08-13):** at the currently active limit of **16384**,
**62 of 140** comparison stars fail static admission; **2 of 5** comps that
produced the good draft-509 BO CVn light curve would be excluded. Raw ceiling
measured **65535**; peaks currently taken from aligned frames reaching **~69000**.
See memo authoritative-numbers table.

## SAT-DIAG placed aperture (2026-08-13)

**Decision:** Raw saturation peaks use **determined** positions (aligned DAO `(x,y)`
on raw grid + optional 11 px COM centroid), not brightest-pixel search on comps.
Removed anchor disk, plausibility ratio, ring self-check, `RAW_VERIFIED` /
`ALIGNED_INTERIM`, per-frame `peak_loc_ok` accounting. Variable-target mag-guided
centroid retained for **frame drift diagnostic only**. Literature:
`dev/results/MEMO_peak_location_literature.md`; spec section 8.2.

**Measured:** draft 510 BO CVn drop comp median raw peak **5436 ADU** (not
**49000** hijack); bright ref saturated **132/134**; ensemble restored to **5**
comps matching 509.

## CAL-DIAG reinstatement (OPEN - separate from SAT-DIAG)

**Status:** **SUPERSEDED -- implemented as INV-CAL-01 / CAL-DIAG v2 (2026-08-13).** See
`dev/results/specs/VYVAR_CAL_DIAG_V2_SPEC.md` and `dev/results/CURSOR_RESULT_inv_cal_01_impl.md`.

## INV-CAL-01 / CAL-DIAG v2 (authorized 2026-08-13)

**Status:** **IMPLEMENTED (not pushed).** Milan authorized 2026-08-13 with Decision 2 amendment.

### Why v2 is not a revert of `967f835`

Commit `967f835` removed CAL-DIAG during config reduction (five keys, auto-MEAN, 2% median test).
v2 **derives** SUM vs MEAN from measured pedestal `P`, `Delta_meas`, `R`, and Check B -- not a
stated convention with tolerance. It drops all five config keys (gate always on when dark applied),
removes silent auto-MEAN, and registers **INV-CAL-01** so a future parameter audit cannot delete
the gate as "configuration." Byte-identical anchor output when convention is SUM is the acceptance
criterion (check-only, not arithmetic change).

### Five decisions (+ section 0 amendment)

| # | Decision | Outcome |
|---|----------|---------|
| **D1** | Derive convention from physics (Check P/C/B), not 2% median tolerance | Implemented in `cal_diag.py`; zero config keys |
| **D2** | INDETERMINATE split by cause | `INDETERMINATE_NEGLIGIBLE` when `Delta_dark < resolv_limit`; `INDETERMINATE_UNMEASURED` when pedestal not measurable or Check P fails on intercept. **Case B behaviour:** implementation proceeds with SUM + `WARN` + loud log (`ui_error`); Milan to confirm vs fail-closed (recommendation in impl report) |
| **D3** | No auto-MEAN correction | ABORT names `P`, `Delta_pred`, `Delta_meas`, `R`, `s_SUM`, `s_MEAN`; no silent resample flip |
| **D4** | Check B standalone | Post-dark sky sanity runs even when `bf=1` (no Check C) |
| **D5** | Provenance | `VY_DKRSMP`, `VY_DKRSMP_SRC`, `VY_CPED`, `VY_CDSKY`, `VY_CDSTAT`; `cal_diag.json`; merged into `pipeline_meta` |

### Pedestal / dark-library findings (from spec section 11)

**11.1 Pedestal not in headers (QHY294MM):** FITS `OFFSET=0.0` while data carry ~24.5 ADU/bin1
pedestal. SAT-DIAG, noise model, and RN/sky algebra must **measure P from dark/bias data**, not
read `OFFSET`. Recorded in `dev/results/specs/VYVAR_SAT_DIAG_SPEC.md` section 4 (pedestal note).

**11.2 Pedestal-dominated dark at -10 C:** 60 s and 120 s masters share median **24.4706 ADU**;
**k ~ 0.001 ADU/s**. Exposure-time matching buys ~0.06 ADU over 60 s -- negligible vs sky.
CalibrationLibrary dark matching need not require multi-exptime libraries at this temperature for
QHY294MM wide rig.

**11.3 Mutable `calibrated/` two-stage product:** **FIXED by INV-CAL-02 (2026-08-13).**
Option A: `VY_CALSTAGE` + FITS `VY_CALDATASUM` in same flush as pixel mutation;
`resolve_calibrated_stage` returns `INDETERMINATE_*` when legacy headers ambiguous;
compare gates refuse rather than assume `PURE`. Force reapply stamps `SKYSF_N_R{pass}`.
Spec: `dev/results/specs/VYVAR_CAL_STAGE_SPEC.md`.

**INV-CAL-02-OPTION-A (2026-08-13, Milan authorized):** Keep in-place mutation on
`calibrated/lights/` rather than Option B separate tree (~1.75 GB/draft at 150 frames x
11.6 MB). Field convention (DRAGONS/LSST/ccdproc) prefers immutable per-stage products;
VYVAR departs deliberately because stamps + DATASUM close the reader-ambiguity gap at
negligible storage cost. **Amendments:** (0.1) force reapply kept, stage token encodes pass
count; (0.2) legacy `VYSKYP2P` without `VY_SKYSF` -> `INDETERMINATE_LEGACY`; (0.3)
`preprocess_calibrated_to_processed` renamed to `qc_enrich_calibrated_lights_in_place`.

**Previous hazard text (pre-INV-CAL-02):** Not fixed by INV-CAL-01. This cost time twice
(P-10 sky-surface sign; INV-CAL-01 P2 predicate).

**Previous text (historical):**
(config reduction; P1 core SHA byte-identical). No dark-resample radiometry check
survives; `INV-FLUX-01` covers arithmetic only. Draft 435 has `VY_DKRSMP=SUM` /
`cal_diag.json`; drafts 509/510 do not.

**Recommended process invariant (`INV-GATE-REMOVAL`):** a verification gate may not
be removed on byte-identity evidence alone; removal requires either proof the
checked condition is now impossible, or an explicit recorded decision accepting the
unverified condition with risk stated. See `docs/VYVAR_INVARIANTS.md`.

**Decision for Milan:** whether to reinstate the CAL-DIAG dark-resample convention
check. Not bundled with SAT-DIAG authorization.

---

## COMP-ASSIGN-02 - n_comp_max is a ceiling, not a target (2026-08-15)

**Context:** COMP-ASSIGN-01 restored colour-then-RMS ordering but applied only
`comp_select_rms_floor` before `head(n_comp_max)`. When a colour bin held exactly
eight candidates, noisy comps (FW CVn up to `comp_rms=0.46`) entered the set;
check scatter rose BO 9.1->17.5 mmag, FW 8.6->20.3 mmag.

**Arithmetic (unweighted flux-sum ensemble, AIJ / `ensemble_normalize`):** for
roughly equal-brightness comps, `sigma_ens ~ sqrt(sum sigma_i^2)/N`. Predicted
from the COMP-ASSIGN-01 selected eights: BO 23.6 mmag (best-3: 11.3), FW 62.8
(best-3: 13.5) vs measured check after 17.5 / 20.3. The RMS composition predicts
the damage; therefore **membership must never pad to eight with above-ceiling
comps**.

**Decision (restores documented known-issue (b) gate):**
`phase01_comparison_max_comp_rms` is a hard ceiling inside
`_select_comps_by_color_then_rms`, applied **before** the colour ladder and
**before** `head(n_comp_max)`. The set is filled only with under-ceiling comps;
it is **not** padded to `n_comp_max`. `n_comp_max=8` is a ceiling, not a target
- three clean comps beat eight where five are noisy.

**Colour vs RMS trade:** measured colour LEVEL term (~-0.373 mag/BP-RP) prices
delta(BP-RP)=0.05 at ~19 mmag of level (corrected at export). An admitted
comp at rms 0.46 costs ~57 mmag of scatter per star in an 8-star unweighted sum.
Ladder widen after the ceiling (slightly worse colour, clean RMS) is the intended
outcome.

**Relax path (existing, not redesigned):**
`COMP_ASSIGNMENT_RELAX_ORDER` =
`colour_tier_widen_T1_to_T4` > `adaptive_delta_mag` > `sparse_fallback_path`.
Inside step-2 selection, if no ladder step reaches `n_comp_min` under the
ceiling, the code relaxes to the full **under-ceiling** candidate set (log +
provenance) and keeps a thin set if still `< n_comp_min` (graceful degradation
warning). Above-ceiling comps are never re-admitted. Empty / zero gate-passers
still route to the existing sparse_fallback recursion.

## COMP-ASSIGN-03 - RMS first, single-source comps (2026-08-16)

**Context:** COMP-ASSIGN-01/02 ranked colour before RMS. On this rig the colour
LEVEL term is corrected at export (IMPL-01, verified) and the shape term is
null (-83 +/- 99 mmag/BP-RP/airmass). High `comp_rms` is always expensive in the
unweighted flux sum: `sigma_ens = sqrt(sum sigma_i^2)/N`. Milan (2026-08-16):
reorder to **RMS -> |delta(BP-RP)| -> distance**, keep the COMP-ASSIGN-02 RMS
ceiling, and admit only **single-source** comps.

**Single-source:** a comparison star must have no other catalogue entry within
`snr_cog_isolation_fwhm` (default 3.0) x FWHM - the same isolation criterion as
the CoG / SNR eval path. A blend's Gaia BP-RP is not the colour of the optical
blob, either component can vary independently, and its EE differs from a point
source. With an uncapped pool and only 3-8 comps needed, the constraint costs
nothing. This supersedes the PRE-IMPL Q5 "defer merging" conclusion **for comps
only** (Q5 concerned the pool at large and matched on the wrong quantity);
merging for targets remains deferred.

**Implementation:** `_select_comps_by_rms_then_color` (honest rename of
`_select_comps_by_color_then_rms`; thin alias retained). Colour ladder widens on
colour while RMS orders within a step. Blends filtered before the ladder.

**Status:** IMPL-05 Item C. Tip stamp `4fe84b4`. SESSION-CLOSE 2026-08-16 push.

## DAO-GAIA-ERA-01 M1-M4 - catalog-derived MASTERSTAR membership (2026-08-19)

**Problem (Part C retry).** MS rebuild built membership FROM detection (~3621 -> 2644
rows). That violated Part-F catalog-derived additive membership: pool shrank, ensembles
swapped, ZP moved 50-250 mmag. Detection/certificate work is not implicated; the
membership model was.

**M1 - Membership rows (M1-amend).** Membership envelope = every on-chip Gaia star with
G <= `target_depth_g` (15.0) + DAO_ONLY reals + bright/sat handling as today. Row existence
NEVER depends on detection outcome. Census accounting stays deeper (17.5); G in 15.0-17.5 is
census-only, not membership. `source_state` carries provenance
(`DETECTED_P1` / `DETECTED_P2` / `FORCED_SEED` / `SEED_REJECTED` / `TOO_FAINT` /
`BLENDED` / `SAT` / `EDGE`). Production: `expand_detection_to_catalog_membership()` at
G<=15.0 in `masterstar_gaia_accounting.py`, wired in `generate_masterstar_and_catalog()`
before zone annotation; enrich/certificate at G<=17.5.

**M2 - Per-frame measurement (M2-amend).** `source_state` is PROVENANCE only, never
measurement permission. Per-frame forced injection measures ALL MS members under existing
`force_eligible_masterstar_mask()` rules (sat/variable/extended exclusions unchanged).
`SEED_REJECTED` and `TOO_FAINT` members are measured exactly as the baseline era measured
them (aperture at Gaia position). Per-frame QA gates (zone etc.) remain the quality authority.

**M3 - Pool candidacy (this task).** State-agnostic; same zone/sat/variable/mag gates
as the `477dc8cf` era. Detection state is NOT a pool criterion yet.
State-based pool policy is **MS-POOL-POLICY-01** (separate task, before first new-field
draft).

**M4 - Detection completeness.** Diagnostic only (census + per-bin curve in
certificate/UI). Not a membership gate. Certificate reports `census_accounting` (100%
on-chip named) separately from `detection_completeness` vs iter4 sandbox reference
(~96% at G<=14.5).

**M-guard - INV-MS-EXPAND-01.** Catalog expand path is fail-loud: expand must report
`n_catalog_rows_added`, write census + certificate with fresh mtimes, or raise. A swallowed
exception on this path is the WCS-bug class and must be impossible. Enforced by
`verify_ms_expand_guard()` after certificate write.

**Part C L-table recalibration.** L5-new: census accounting = 100%. L1-final: baseline
48-target set must be a SUBSET of the new LC set (each of the 48 with an LC); additional
targets reported as additive, not failures; any of the 48 missing = DEVIATE with named
reason. L2/L3/L6 unchanged. L4: eval harness label merge fix (offset-corrected XVAL).

**Status:** Panel GREEN 2026-08-20; anchor recut authorized and executed
(VL-ANCHOR-WCSINV -> 9902d918). PUSH-STAMP-01 pending Milan.

## DAO-GAIA-ERA-03 CLOSE - pinned ensembles + check-star pinning (2026-08-20)

**Context.** Production close of DAO-Gaia era on live draft 516 after panel-red
fixes (INV-PIN-04 color authority, check-pin distance fallback).

**Decisions (binding).**

1. **Pinned-output principle.** For the 48 baseline LC targets,
   `pinned_ensembles.csv` is the membership authority; re-validation uses
   data-derived rules only (zone/sat/distance/RMS; color tier limits via
   INV-PIN-04, not global BP-RP ceiling).
2. **Check-star pinning = KNAME continuity.** Each pinned target carries
   `check_catalog_id` in the pin file; export KNAME must match anchor check
   (`1497613731286514432` for BO/FW on NoFilter_60_2). Sidecar
   `check_kmag_*.csv` is outside core photometry SHA scope.
3. **SHA scope 121.** Core hash includes: all `lightcurve_*.csv`,
   `comp_quality_*.json`, and `comparison_stars_per_target.csv`. +14 additive
   LCs/comp_quality vs 477dc8cf era; -2 honest zone_noise missing. Extended
   n=179 adds comp_qa sidecars.
4. **L2 rescoped (unchanged from ERA-02).** Same-branch regression only; era
   migration uses T0-T3 + pinned-comp + check-meter bands, not byte-identical
   `mag_calib_final` vs 477dc8cf for all metadata columns.
5. **INV-PIN-01..04** wired in `invariants_runtime` + `VYVAR_INVARIANTS.md`.

**Acceptance (2026-08-20).** Panel OVERALL PASS. Product SHA **9902d918**
n=121. BO/FW check MAD 7.151 / 8.201 mmag (band 6.08-8.22). CV CVn
`per_frame_saturation`. SUBMIT-01 PASS on BO export post KNAME fix.

**F2 note.** Full Phase-2A re-run on live draft moved core SHA (9902d918 ->
a751183c) before INV-DAG-01 stop; surgical restore + check_kmag patch only
preserved SHA. Phase-2A-only is not SHA-safe on live draft.

## DAO-GAIA-ERA-02 ERA-ACCEPT - tiered acceptance for pipeline-era changes (2026-08-19)

**Context.** ERA-02-OPEN investigation + execute run on preserved
`draft_000516_era_candidate`. Pinned-comp experiment confirmed selection-only
mechanism (BO +227.6 mmag unpinned -> 0.0 mmag pinned; FW +199.9 -> 0.0).
L4 228 mmag was harness offset bug (fixed P0).

**Tier model (binding for era migration; T4 for same-branch only).**

| Tier | Gate | Era candidate (2026-08-19) |
|------|------|----------------------------|
| **T0** | Certificate PASS, census 100%, empty-sky inv | **PASS** |
| **T1** | Era-native XVAL RMS (BO/FW) + check MAD bands | **Mixed** (BO XVAL 6.24 vs 4.86; FW 2.64 vs 1.52; BO/FW MAD OK) |
| **T1-abs** | \|median_calib - Gaia->Johnson V\| for BO/FW/GH; era closer to catalog wins; >50 mmag regression flags | **Mixed** (BO/FW candidate wins; GH baseline wins +192 mmag candidate regression) |
| **T2** | Post-offset shape: max epoch residual <= 10 mmag vs baseline | Unpinned 18/48; **Pinned 48/48** |
| **T3** | Pool/per-target comp Jaccard (diagnostic) | Pool 0.704; per-target median 0.333 unpinned |
| **T4** | L2 byte-continuity (\|median Delta\|<=2, max<=10 mmag) | **Same-branch regression ONLY** - not era migration |

**T1-abs rationale.** VYVAR exports calibrated magnitudes; differential ensemble
offset is not arbitrary for product consumers who anchor to catalog transforms.
Gaia DR3 G + BP-RP -> Johnson V via `gaia_johnson.transform_gaia_to_johnson`
(Table 5.9). Compare median `mag_calib_final` per check star vs transformed
catalog V; report both eras side-by-side. Regression threshold: candidate
\|Delta\| worse than baseline by **> 50 mmag** on any of BO/FW/GH.

**L2 rescope.** `L2` continuity gate retained for **same-branch hotfix
regression** only. Era migration must use T0-T3 + T1-abs + pinned-comp
sensitivity experiment. Byte-identical `mag_calib_final` vs prior anchor is
**not** a valid era acceptance criterion (Honeycutt 1992 ensemble rebase).

**Pinned-comp protocol.** Before era sign-off: freeze baseline
`comparison_stars_per_target.csv`, re-run Phase-2A on candidate MS/flux.
Collapse to mmag -> selection-only; persist -> flux-path investigation.

**Status:** Panel delivered; architect/Milan decide era acceptance. No anchor
recut authorized from this task.

---

## Product model and pool policy (2026-08-20, Milan, binding)

**VYVAR 1.x is a deterministic draft generator.** User flow: run -> draft ->
submit -> delete draft. No cross-draft memory, no persistent state layer in 1.x.

**Pins are internal-only** (anchor-continuity tool for the 48 legacy targets).
Users never see or manage them. Per-field auto-pin / comp-history DB deferred to
release 2.0 (**COMP-HISTORY-DB** on ROADMAP).

**Selection rules stay immutable; MS-POOL-POLICY-01 stabilizes inputs only**
(catalog-derived mags/colors, qualification thresholds, hysteresis, pin-time
`n_min` refill cap).

**`n_comp_min` / `n_comp_max` are USER settings (expectations), not physics.**
The pool has no min/max; per-target selection has none; min/max gate the final
expectation only. At n=1, `err_sem_rel` is undefined -> the error model must
handle this loudly (no silent zero; I-04 class); NOTES must state single-comp.
A pin carries its pin-time `n_min`; a config change never silently refills a pin.

**ePSF validation constraints:** measure ONLY the science set (targets + comps +
checks + BLENDED sample), never all stars; PSF-star selection = composition of
existing gates; split-half self-test certificate instead of star-count constants;
time-mode (fixed / FWHM-scaled / per-frame) chosen by measurement per setup;
mono cameras only for now (OSC deferred).

**2026-08-21 addendum (draft 518 XVAL).** Draft 518 cross-validation vs AstroImageJ was a
**TEST only** (no ledger entry). Formal Newton-rig XVAL waits for VYVAR-native calibration
data, not pre-calibrated external input.

---

## EPSF-VALID-02 S6 - gated production ePSF swap (2026-08-22)

**Decision:** Draft 516 production `masterstar_epsf.fits` replaced with Part C gated
science-comp build (67 stars, edge-star guard). Pre-gated 1475-star model archived as
`masterstar_epsf_pre_gated_20260822.fits`. Draft 517 receives first gated production build
(66 stars). Milan authorized swap after STOP-B (S5b aligned certificates PASS).

**INV-PSF-ADDITIVE-01:** RUN ePSF uses F6 merge path only; PSF columns additive to existing
proc sidecars; aperture and all non-`psf_*` columns untouched.

**N policy (production + certificate):**
- Production builds use the **full Part C gated science-comp pool** (not INTERIM top-N=200).
- Certificate metric: **scale-aligned** per-star RMS delta < 3x median ERR budget of compared
  stars; raw inter-model flux offsets are bookkeeping (differential photometry cancels them).
- INTERIM top-N=200 remains **disabled**.

**Edge-star build guard:** On non-finite EPSFBuilder result, drop edge-nearest isolated star
(logged in `masterstar_epsf_meta.json` -> `build_guard`). Build **FAILS LOUDLY** if guard would
drop >10% of the gated pool (no silent erosion).

Evidence: `dev/results/CURSOR_RESULT_EPSF_VALID_02_S6.md`, `CURSOR_RESULT_EPSF_VALID_02_S5B.md`.

---

## EPSF-BRIGHT-01 FD-A - full CCD PSF fit variance (2026-08-23)

**Decision (Milan GO):** Replace sky-only photutils fit error maps with the full per-pixel
CCD variance model for PSF-branch chi2 gating and sandwich flux errors:

`sigma^2 = F_model/g + sky/g + (RN/g)^2` (ADU^2 domain)

**Implementation:** One-pass **(i)** - static error map built from DAO aperture flux
(`ref_fluxes`) distributed by the ePSF profile at the catalog position before fit
(DAOPHOT / Anderson & King practice; photutils `PSFPhotometry` requires a fixed
per-fit error array). Gain and read noise resolved via production
`_psf_resolve_gain_read_noise` / `param_resolver.resolve_gain` (never hardcoded).

**Provenance strings:** `psf_weight_mode=full_ccd`, `psf_err_mode=sandwich_full_ccd`.

**Supersedes for chi2 gating only:** the 2026-06-09 sky-only **fit-weight** decision
remains documented for V3d flux-bias context; FD-A restores source Poisson in weights
because sky-only reduced chi2 acted as a brightness cut on the PSF branch (M1/M2
confirmed). Threshold **50** unchanged - genuine outlier gate under recalibrated statistic.

**Scope:** PSF columns only; aperture path untouched; INV-PSF-ADDITIVE-01 guards merge.

Evidence: `dev/results/CURSOR_RESULT_EPSF_BRIGHT_01.md`, `CURSOR_RESULT_EPSF_BRIGHT_01_P3.md`.

---

## CLOSED - EMPTY-DAO-01 (2026-08-21)

Register-only item from 2026-08-16 (empty DAO table / forced-only proc rows).
**Closed:** empty-sky false-fill rate addressed by era DAO-Gaia audits (draft 517
certificate empty_sky pass2 **0.14%**, 3/2200; pre-era class ~55%). Remaining
forced-only export path is governed by era census accounting (100% on 517). See
ROADMAP EMPTY-DAO-01 closure pointer and `CURSOR_RESULT_DRAFT_517_REVIEW.md`.


---

## NIGHT-FWHM-2026-10 (2026-10-10)

One FWHM for neighbour geometry: the night median of `fwhm_px_for_aperture` over the
observation group's own per-frame CSVs (`photometry_comp.night_median_fwhm_px`).
Used by: Phase-1 isolation rule (nearest neighbour < `snr_cog_isolation_fwhm` x FWHM,
brightness-blind, kept), minimum target-comp separation
((annulus_outer_fwhm + k_max) x FWHM), pool neighbour light (BLEND-K), and the Phase-2A
night value (`aperture_policy.json` fwhm_night_median_px, grid fallback).
**Not** used for the D3 astrometric gate, which stays on the MASTERSTAR FWHM
(`fwhm_d3_px`), because that residual is measured on the MASTERSTAR image.

Why: MASTERSTAR is the best frame (524 g: 4.64 px vs night median 5.31 px), so the rules
were too lenient on a typical frame; the draft-level `qc_metrics.csv` mixes filters
(524: g 5.31, r 4.91, mixed 5.14). Literature: Kepler crowding per quarter with the
actual PRF; Irwin et al. 2007 (Monitor) - blending varies with seeing, radius from the
typical FWHM; Artusi 2016 (JAAVSO) mean FWHM; Naylor 2002 median-FWHM star.
524 effect (pool 1405, G < 16.4): isolation drops 244 -> 335. Brightness-aware isolation
(ATLAS, Heinze 2018) considered and deferred: the geometric rule also covers DAO_ONLY
neighbours not in the catalogue and worse-than-median seeing.
Tests: `dev/tests/test_night_fwhm_2026_10.py`.

---

## UNCAL-LC-2026-10 (2026-10-11)

A target with no comparison star holding a valid band magnitude (band_system NONE; Gaia
DR3 -> g/r relation valid only for BP-RP <= 2.0, so red targets with red tier-1 comps) kept
mag_calib all NaN (export guard) but was then flagged no_data on every frame, got lc_rms NaN,
no check star and was dropped from the PDF - although its differential light curve
delta_mag (= mag_calib - constant, ensemble_normalize) was valid. Regression from BANDAUTH
(08d6a98), which removed the old Gaia-G zero-point fallback (wrong band for a g/r filter).

Now: LC flags (assign_lc_flags), lc_rms, lc_rms_ooe and the check star use delta_mag when
not calibrated; mag_calib stays NaN; lc_median_mag stays NaN (no level); PDF and GUI plot
delta_mag labelled "not calibrated", KMAG shown as not calibrated; export still refuses.
No Gaia-G fallback. K2/CT per-frame terms are not in delta_mag (<= 0.6 mmag on 525 g).
Check-star relative mode only for uncalibrated targets (allow_relative).

Real-data A/B on draft 525 g (98 targets, same Phase-1 input): 21 NONE targets; PDF pages
0 -> 19 (2 have no measured frame at all); check star 0 -> 21; 38 calibrated LCs, all
summary rows of other targets and all AAVSO/VarAstro files byte-identical; 21 of 21 NONE
refused by export. Tests: dev/tests/test_uncal_lc_2026_10.py.

---

## GSPC-ZP-2026-10 (2026-10-11)

Milan: the standard light curve stays calibrated from Gaia DR3 G, BP-RP (Riello et al. 2021); a
second scale from the Gaia DR3 Synthetic Photometry Catalogue (GSPC; Gaia Collaboration,
Montegriffo et al. 2023) is added where the standard calibration is a problem, with a note.
No local database: one online query to the Gaia archive per group, cached as
`photometry/gspc_cache.csv` (rebuilt if its columns differ). No new config key.

- Comps accepted for GSPC: flux/error > 30, g_sdss_flag = 1 (validated G/BP-RP range; 0 =
  extrapolation), |C*| < 3 sigma(G) (Riello 2021 Eq. 18), BP blended transits < 10 %; >= 2 comps.
  Per-comp sigma = photon (+) 10 mmag (App. D: red giants/dwarfs agree within ~10 mmag).
- Zero point: same estimator and Broeg weights / Honeycutt levels as the ensemble. mag_gspc =
  standard curve shifted by (C_GSPC - C_std) with the band colour term recomputed on the GSPC
  comps inside the CT fit range; uncalibrated: delta_mag + ensemble level + C_GSPC.
- Notes: A = no standard zero point (band NONE; not exported); B = |GSPC - standard| >= 0.02 mag
  (2 x GSPC accuracy); B-heuristic = GSPC not measurable, red comps (median BP-RP > 1.5) at
  |b| < 10 deg. PDF note + right-hand GSPC axis; AAVSO header + NOTES ZPNOTE; VarAstro header.
- UNCAL-LC: K2 (k'' X dC) is now applied to delta_mag of uncalibrated curves too.
- apply_color_term default "auto" (standard filter on, L/Clear off).
- PDF-LAYOUT: comp table height measured before planning the plot; table continues on a new
  page when it does not fit (notes were drawn below the page edge).

Real data 525 g (98 targets): standard mag_calib_final, flags and summary of all 38 calibrated
targets byte-identical to before; notes A 21 (GSPC level for 17), B 8 (dZP -0.060..+0.029),
B-heuristic 20; all 47 notes visible in the PDF; exports identical except ZPNOTE (28 files).
Revisit with Gaia DR4 (GSPC-ZP-OPEN). Tests: dev/tests/test_gspc_zp_exp.py.

## 2026-10-11 -- exports of uncalibrated targets, M3 star page, informational aperture tab, fixes

**UNCAL-EXPORT-2026-10 / GSPC-STD-2026-10 (Milan 2026-10-11).** A target without a standard
zero point (band_system NONE, note A) is always exported. With a GSPC zero point (>= 2 accepted
comps, `mag_gspc` in the LC) it is exported as STANDARD magnitudes on the GSPC scale (AAVSO
MTYPE=STD, TRANS=NO, NOTES `ZP=GaiaDR3-GSPC|ZPERR=..`, KMAG/KREFMAG of the check star on the
same scale; VarAstro magnitude column = mag_gspc). Without it (e.g. Gaia archive unreachable:
the module is skipped and the note says so) as DIFFERENTIAL magnitudes (AAVSO MTYPE=DIF,
CNAME=ENSEMBLE, CMAG=na, CHART=na, NOTES `ZP=none-differential`; VarAstro `dif_mag`).
Targets with a standard zero point keep the Gaia G, BP-RP polynomial (option a); option b (GSPC
for all) is postponed to Gaia DR4 (ROADMAP GSPC-ZP-OPEN).
Evidence (field 525, Gaia archive, `claude/VYVAR_XVAL_AIJ_IRAF_20261011.md` and chat): vs
APASS DR9 g' (504 isolated stars) GSPC +5..+27 mmag for BP-RP 0.75-2.0, polynomial +30..+87;
vs Pan-STARRS1 g -> SDSS (Tonry 2012; 819 stars) GSPC -41..-26 mmag flat (rsig 7-21) up to
BP-RP 3.0, polynomial -14..+28 colour-dependent. ZTF J212128.80 check star on the GSPC scale
15.983 vs its own GSPC magnitude 15.972.

**STAR-PAGE-M3-2026-10 (Milan 2026-10-11).** Per-star PDF page = mock-up M3: name by source
(VSX / exoplanet host TOI / Gaia DR3 id; also the export STARID/file name), light curve with
the check-star strip and a UT (HH:MM) top axis, flagged points shown in red, compact comp table
(Broeg weights, GSPC magnitudes, short status), facts, aperture/annulus cutout, field map,
calibration note box. `src_py/star_page_m3.py`.

**UI-INFO-2026-10 (Milan 2026-10-11).** The Aperture Photometry tab is informational only; the
RUN button was removed (runs: VAR-STREM / night runner).

**Fixes 2026-10-11** (found while verifying; Milan: always fix and report):
FIX-LOWSNR (low_snr verdict was never applied to the LC flags), FIX-K2ROWS, FIX-KMAG1
(single-comp check KMAG on the catalogue scale), FIX-ERRUNIT (LC `err` is the relative flux
error: PSF path, PSF routing SNR, PDF MERR, GUI/PNG error bars, spike residual),
FIX-FITSQA-NAN, FIX-PNGFLAGS, FIX-APREPORT (summary aperture = radius actually used),
FIX-EXPORTALLBAD (no export of epochs that all carry exclusion flags).
Cross-validation (ZTF J212128.80, 29 frames): IRAF phot median sky 0.35 mmag rms vs VYVAR;
differences to AIJ / IRAF mode come only from the sky estimator.


## 2026-10-11 (evening) -- drafts 525/526 feedback: forced targets, trust coverage, SMART comps, local PSF

Milan's review of drafts 525 MN Boo / 526 FI Boo (points 1-8 approved 2026-10-11 15:50).
Verified on a copy of draft 526 (122 frames, 248 targets): per-frame forced rows emulated
on the existing aligned frames, Phase 1 + Phase 2A + PDF re-run (camera gain 3.17 as in his DB).

**FIX-VARASTRO-MAP.** The VarAstro export lost the field map (`# FIELD IMAGE:` line + PNG next to
the .txt) in the 08d6a98 rewrite; restored for every export mode. Maps copied into both drafts.

**FORCED-TARGET-2026-10.** Known variables are force-measured at the MASTERSTAR position in every
frame (before: only when DAO detected them; draft 526: 53 of 81 targets lost > 50 % of the
epochs as no_data). Saturated and extended sources stay excluded.

**SAT-TARGET-2026-10.** A target saturated in more than half of its frames
(clean fraction < per_frame_saturation_min_clean_frac 0.5) is not measured, also when its
MASTERSTAR zone is linear/unknown (526: FG Boo 71/122 frames saturated, NSV 20299 G 5.6).
Summary skip_reason=per_frame_saturation; trust "not measured: per_frame_saturation" (a light
curve left on disk by an earlier run is ignored). A re-run keeps the skip (the recorded reason
made the target "rescue-eligible" and it was measured again - review finding, fixed).

**TRUST-COVERAGE-2026-10.** Trust follows the epochs that are actually exported:
RED "insufficient data" when n_export < max(min(10, n/2), 0.2 n) (export refused, compact PDF
block with the dominant flag), YELLOW below 0.5 n. n_export is counted with the exporter's own
row selection (export_reports._select_export_lc_rows). 526: 45 targets with 0 usable epochs were
GREEN before; all low_snr (median relative error 0.2-1.5 at G 13.5-15 with sky 3900 ADU/px;
cross-checked by hand: G 13.89 sky-limited error 0.16 per frame).

**TARGET-BLEND-2026-10.** Neighbour light inside the target aperture (Moffat model of
comp_blend, same as the comp pool): > 1 % soft, > 5 % hard (RED). 526: CSS J153101.6 48 %
(G 10.2 neighbour at 5 px), ZTF J153239.42 2197 %, TIC 309849817 273 %.

**PSF-LOCAL-2026-10.** Aperture radius scaled by the target's own profile width when its
second-moment FWHM is > 1.25 x that of isolated constant stars of the SAME magnitude
(+-0.5 mag, >= 10 stars; cap 2.5). Magnitude matching is required: the moment FWHM grows with
magnitude (526: ratio 0.95 at G 9 -> 1.37 at G 14), a bright-star reference scaled 100 of 248
targets. Not scaled: targets with a catalogue neighbour within 3 FWHM (blend) and saturated /
nonlinear targets (flat core looks broad). The scaled radius is used everywhere (grid, LC,
k choice S/N, TARGET-BLEND). 526: ASASSN-V J152756.18 x1.33 and KW Dra x1.33 (both measured).
Bug found and fixed in the test: a non-round scale was never measured (rounding mismatch ->
no_data).

**COMP-SMART-2026-10.** Phase 1 selects the comps exactly as before (n_comp_max) and, as a
separate selection with n_comp_max + 8, the next 8 candidates in that selection's rank order
as the reserve (comparison_reserve_per_target.csv; review finding: splitting one larger
selection changed the primary set of red targets because the result is re-sorted by RMS). Phase 2A runs the stability test on primary + reserve
(iterative: worst suspect removed, rest re-tested), takes the clean ones in rank order (max
8), then verifies the final set on its own (a comp good among 16 was suspect among the final
8) and replaces/drops suspects; only when fewer than 3 clean exist the least suspect ones are
kept and marked. comparison_stars_per_target.csv is rewritten (Phase 1 version kept as
_phase1.csv), comp_smart_log.json, PDF "(reserve)" marks. 526: targets with a suspect comp in
the final ensemble 10 -> 2 (both have only 3 candidates); Phase-1 comp sets identical to the
previous run for all 113 targets.
Literature: comparison stars must be constant; suspect ones are rejected and replaced from the
candidate list (Henden & Kaitchuck 1982; AAVSO CCD Photometry Guide ch. 5).

**HRD-PAGE-2026-10.** HRD enrichment: SIMBAD queried even when Gaia TAP fails; Gaia adds
distance_gspphot, ag_gspphot; TAP timeout 60 s (hrd_enrich_tap_timeout_s). Inverted field image
with dark markers; table with wrapped cells fitted to the page (Category, Gaia DR3,
SIMBAD (type), SpT, G, M_G, BP-RP, Teff, logg, d [pc] (1/parallax or GSP-Phot), src); caption
below both images.

**CHECK-FACTS-2026-10.** M3 page: check star name + full Gaia id, G and BP-RP, measured KMAG
vs catalogue KMAG (difference), scatter vs expected (median KMAG error, ratio; > 2 in orange);
aperture row notes the own-PSF scaling; blend row.

## 2026-10-12 -- PDF without no-data targets, blank page, cross-validation of 6 drafts

**PDF-NODATA-2026-10 (Milan 2026-10-12).** Targets with insufficient data (TRUST-COVERAGE) get no
page in the PDF; they stay in the summary table of all targets (526 test: 85 -> 62 pages).
**FIX-BLANKPAGE-2026-10.** The HRD section started with an extra showPage after a section that had
already ended its page -> one empty page in every report (found in Milan's 526 report, p. 17).
**XVAL-2026-10-12.** IRAF apphot (median / mode sky) and AstroImageJ (mean / remove-stars) on the
same pixels, positions, radii, comps and Broeg weights as the VYVAR LC, every 5th frame, one target
per draft (BO CVn, FI Boo, MN Boo, S4 Cyg = ZTF J212128.80 g and r, V0842 Her, CX CVn): IRAF median
sky reproduces the differential LC to 0.4-1.1 mmag rms; mode / remove-stars 0.5-3.9 mmag (sky
estimator only); AIJ plain mean sky biased by annulus stars (up to +154 mmag). Muniwin (colleague,
own comps) vs VYVAR for S4 Cyg: same shape (r 0.988 g / 0.997 r), residual rms 16.2 mmag g /
8.2 mmag r = Muniwin's own point errors; identical JD (0.0 s). Findings for Milan: sigma_sys_mag
0.018 in the config map dominates the S4 Cyg error bars (19-21 mmag vs measured point-to-point
scatter 8-10 mmag); frame QC rejected 11 g / 8 r frames (borderline elongation / non-uniformity
limits) that Muniwin used; one S4 Cyg comp 29 px from the chip edge has its sky annulus partly
off-chip in poor-seeing frames.

## 2026-10-12 (afternoon) -- error floor per night, frame QC floors, comp edge / linearity / S/N

Approved by Milan 2026-10-12 14:41 (after the IRAF / AIJ / Muniwin cross-validation).

**SIGMA-SYS-NIGHT-2026-10.** The systematic floor in the per-point error is measured every night
on the final comparison stars (k = 1.25): leave-one-out differential LC against the other comps,
robust point-to-point scatter minus the predicted photon/sky/read noise, median over comps with
photon error < 10 mmag (>= 5 needed). Replaces the fixed config `sigma_sys_mag` (now only a
fallback). Evidence: S4 Cyg config 18 mmag made error bars 19-21 mmag vs measured 8-10 mmag
scatter; FI Boo 526 test night: floor 4.5 mmag from 39 comps (Everett & Howell 2001: 2-5 mmag),
median scatter / error 0.88. AAVSO uncertainty guide: uncertainty from measured scatter.
Reported in the PDF methods ("this night's systematic floor X mmag") and sigma_sys_night.json.

**QC-FLOORS-2026-10.** Frame QC rejects only above max(group limit, physical floor):
elongation 1.30, transparency loss 1.0 mag, quadrant non-uniformity 0.20 mag. S4 Cyg 526: all
11 g + 8 r previously rejected frames (elongation <= 1.28, dZP <= 0.62, non-uniformity <= 0.16)
are kept; Muniwin used them (scatter 1.0-1.7x, errors 1.35x the rest).

**COMP-EDGE-2026-10.** EDGE-ANNULUS-01 for comps in every frame: the comp pool bbox is shrunk by
annulus_outer_fwhm x the night's largest frame FWHM (was the MASTERSTAR FWHM). S4 Cyg: comp
2172293994744599040 (29 px from the edge, r_out up to 42 px) now excluded in g and r. The
target per-frame edge check includes the own-PSF scale.

**COMP-SAT-80-2026-10.** Comparison stars: raw peak < 0.80 x saturation level (MASTERSTAR and
per frame); targets keep 0.90 (AAVSO CCD/CMOS guide 3.3.5 / 3.3.7).

**COMP-SNR-50-2026-10 / SNR-MIN (targets).** Comparison stars need MASTERSTAR aperture S/N >= 50;
targets S/N >= 10 (snr_min; AAVSO guide 2.1.4/2.1.8: S/N 100 excellent, < 10 poor). Targets
without a DAO flux now get their S/N measured on MASTERSTAR (before: no value -> no gate; 526:
46 targets measured with per-frame S/N 0.4-9). 526 test: below_snr_min 28 -> 94, light curves
81 -> 55 (all with usable data), comp pool 285 stars (S/N >= 50), 122 of 155 target ensembles
with 8 comps. Master S/N agrees with the LC photon S/N (e.g. G 13.89: 4.55 vs 4.58).
Review fixes (independent review 2026-10-12): the target S/N fill uses the exact Gaia id from
`name` (catalog_id float-damaged in 2540 rows of 526); the comp gates also apply when the
global pool is empty and the masterstars fallback is used (apply_comp_quality_gates); the night
floor is measured at the grid k nearest the median chosen k* (stored in sigma_sys_night.json);
per-frame FWHM for the comp bbox is clamped like the Phase-2A grid; pipeline_meta
sigma_sys_night is reset to "config" at the start of every Phase 2A.

**PHASE1-SKIP / PROGRESS-MEASURED-2026-10 (Milan 2026-10-12).** Phase 1 selects comparison
stars only for targets that can be measured: final skips (FINAL_SKIP_REASONS = VSX type out of
scope, S/N below snr_min on MASTERSTAR, depth/noise) get none; saturation skips still do (the
per-frame test of Phase 2A can rescue them). The status line counts measured targets only
("ciel i/N (M preskocenych: S/N, typ)"). FI Boo 526: Phase 1 32 -> 7.5 min, light curves,
trust, comps and magnitudes identical (max |dmag| 0).

**WORK-RULE: remove old code (Milan 2026-10-12).** Every implementation first maps what it
touches in the code. Old paths that do the same job wrongly (duplicates, fallbacks, dead gates,
their config keys, UI, docs and generators) are removed or fixed in the same change - they are
not left running next to the new path.

**ALIGN-VERIFY-2026-10 (Milan 2026-10-12).** Draft 526 (BO CVn, NoFilter_60_2): astroalign failed
on frames 040, 041, 043, 044, 046 and the translation-only phase-correlation fallback was accepted
although the frames are rotated 0.31-0.40 deg against the post-flip reference (frame 103): stars
1-7 px off the apertures (check star flux -79 instead of 132893 ADU, k100/k200 0.1-0.4 instead
of 0.72), all frames flagged aligned with align_residual_px = 0.0 (computed on positions locked
to the MASTERSTAR grid -> always 0). The 5 frames made every comp look unstable (p2p 0.16-0.54),
dropped two G 12 comps by the detection-fraction rule, inflated the ensemble error to 48 mmag and
put spurious points into the LC. The same frames aligned correctly in the morning run (draft
525, error 7.8 mmag); in the container astroalign succeeds on them for 50/50 seeds - the
failure depends on the environment, which is why it must never pass silently.
Now: every registration is verified on the stars. The frame's detections are mapped through the
transform and paired with the reference detections (mutual nearest neighbour within 2 FWHM); the
90th percentile of the pair distances must be <= max(0.25 FWHM, 0.5 px) - the percentile because a
rotation left in a shift-only registration grows towards the chip edge, the px floor for DAO
centroid noise on undersampled rigs - with >= max(10, 0.3 x possible) pairs (pipeline_constants
ALIGN_*). Correct registrations on 526 measure p90 0.15 px. When astroalign fails or does not
verify, a similarity transform (shift + rotation + scale) is fitted on star pairs, seeded by the
astroalign matrix, phase correlation, identity and a 180 deg (meridian flip) guess. A frame that
cannot be verified is not aligned and not measured (alignment_report reason
alignment_not_verified); its aligned FITS / proc CSV from an earlier run are deleted so a re-run
cannot measure them; the count reaches the LC trust flag ("N epochs failed alignment").
Non-finite transforms or star positions never crash the alignment (the frame is skipped).
Removed: translation-only phase-correlation, WCS-CRVAL shift (audit AST-5: wrong direction) and
identity fallbacks; the circular Phase-2A residual (_compute_frame_align_residuals,
_record_align_residuals_to_report, _frame_align_residual_gate_select) with its config keys
frame_align_residual_* (KNOWN_REMOVED_KEYS), UI, registry, docs; the K2 fit-frame residual
subset (it never matched a frame: proc_X vs X keys). OSC channel handoff reuses only matrix
registrations: the donor's verified frame->reference matrix (astroalign, wcs_reproject or
star_similarity) is applied to the channel frame; without a finite matrix the frame is not aligned. The
astroalign error text is logged. alignment_report.csv now carries align_residual_px and
align_pairs from the alignment itself. Real frames of 526 (040-046, astroalign forced to fail):
star_similarity rotation 0.31-0.42 deg, residual 0.06 px on 159 pairs (astroalign 0.05-0.07 px);
aperture fluxes equal to astroalign within 0.1 %; the old translation-only shift measures
3.1-3.2 px and is rejected.

**CHECK-NOT-SUSPECT-2026-10 (Milan 2026-10-12).** A comparison star that COMP-SMART dropped as
unstable for a target is not that target's check star (526 BO CVn: the dropped comp became the
check star; check_star_kmag.drop_smart_rejected_check_candidates).

**COMP-SNR-20-2026-10 (Milan 2026-10-12).** The comp MASTERSTAR S/N minimum is 20 (was 50).
On the 60 s wide-field rig S/N 50 is G ~11.5: it removed 2924 of 3272 pool stars (pool
507 -> 126) and the G 12.0 / 12.2 comps BO CVn used before. AAVSO CCD/CMOS guide 2.1.4: below
S/N 10-20 is poor.

**FITS-SCALE-2026-10 (Milan 2026-10-13).** Draft 532 (QHY294, lights used without VYVAR
calibration -> ``non_calibrated/lights``): the in-place QC / sky-surface write opened the 16-bit
original (BZERO=32768) in update mode and assigned float data; astropy keeps BZERO/BSCALE and
stores the floats unscaled, so every in-place write added +32768 ADU (scan sky 1664 -> 34 100 in
532, 66 800 after a second run). Consequences: sky noise in the error model from a 34 000 ADU sky
(error bars 2-7x the real scatter, most targets RED "0 of 30 epochs usable"), no saturation level.
utils.set_float_image_data_in_place removes BZERO/BSCALE/BLANK before assigning physical float32
data; used by the QC in-place writer and the platesolver stack write (the only two update-mode
data writes). Frames already written with the offset keep it: such drafts are re-imported from
the original frames.

**SAT-PRECAL-2026-10 (Milan 2026-10-13).** One saturation authority for every draft: SAT-DIAG on
the draft's own light frames (``Raw/lights`` or ``non_calibrated/lights``; before only Raw ->
532 found none and stopped at INV-SAT-01 after the exports). Integer frames: pixel pile-up as
before. Float frames (before refused): star-peak plateau - >= 10 star peaks (local maxima) within
1 % of the highest peak and >= 10x more than in the band just below; 532 (30 frames): 509 peaks
at 65 507 ADU, 10 below. Float frames without saturated stars: MAX_OBSERVED (unverified, never
excludes). The MASTERSTAR build measures SAT-DIAG before setting its zones (before it read
sat_diag.json, which is written later); a measured level wins over EQUIPMENTS / header and over
the per-image clip in MASTERSTAR zones, MASTERSTAR detection and per-frame catalogs; the Phase-2A
per-frame target test uses 0.90 x the SAT-DIAG level (draft_peak_test_adu). Removed: the guess of a
65535 / 32767 clip from pixels near those values (_detect_empirical_clip_level_adu), the REFUSE
path for float frames, the late sat_diag.json lookup (_sat_adu_from_draft_sat_diag). Calibrated
drafts: unchanged (526: MASTERSTAR and per-frame limits 58 981.5 = 0.90 x measured 65 535 before
and after).

**PER-FRAME-ONE-PATH-2026-10 (Milan 2026-10-13).** The per-frame catalog runs the same function
(_export_per_frame_run_catalog_core) for one or many workers. The one-worker copy
(frame_export._run_one_catalog) lacked the MASTERSTAR join (zone, bp_rp, phot_g_mean_mag, ...),
forced photometry and ePSF records; on a 1-2 core machine Phase 1 failed in the comp pool
admission. Removed with its helpers and imports.

Test (container, 532 every 5th frame = 30 frames, reconstructed 16-bit originals, full run from
import, old code c22ec8e vs new): old = Milan's failure (sky 34 141 ADU, SAT-DIAG none,
INV-SAT-01 FAIL), errors 4.4x the point-to-point scatter (median), GREEN 10, AAVSO 16; new =
sky 1 373 ADU, saturation 65 507 ADU measured, run completes with PDF, errors = scatter (median
ratio 1.0), sigma_sys night 3.8 mmag, GREEN 26, AAVSO 44, same 44 light curves.

## TARGET-RULES-2026-10 (Milan 2026-10-13)

**Rule.** A photometry target is a VSX star DAO-detected on MASTERSTAR. Catalogue-membership
rows (DAO-GAIA-ERA-01 M1: Gaia position added without a detection, flux NaN) are catalogue rows,
not image data -> excluded (`no_dao_detection`). Then three final cuts: `saturated` (MASTERSTAR
zone, upper cut), `below_snr_min` (MASTERSTAR aperture S/N < snr_min, or no S/N; lower cut,
drowned in noise), `vsx_type_out_of_scope` (ROT). Final = no comparison selection in Phase 1,
never measured, never revived (FINAL_SKIP_REASONS). Kept: SAT-TARGET (linear on MASTERSTAR but
saturated in most frames -> `per_frame_saturation`). Same path for calibrated and uncalibrated drafts.

**Why.** Draft 532 sent 283 targets to Phase 1 instead of 240: 32 catalogue rows without DAO
(NaN S/N filled from the image; CS Cam G 4.1 and NSV 1158 G 6.4 saturated on MASTERSTAR but zone
`unknown` -> no upper cut) and 11 saturated targets kept for the per-frame "rescue".

**Removed (old code doing the job wrongly / never running):** PFS rescue (pfs_rescue_eligible,
decide_target_saturation_policy, _keep_recorded_skip_reason, PFS_* reason sets, enabled flags,
per_frame_sat_fallback, dead INV-CFG-01 block); _fill_missing_master_snr; _active_target_zone_flag
and the labels catalog_only / noisy1-3 / neznama_zona / zone_noise / below_target_depth /
saturovany ciel; snr50_ok branch (column never written); TARGET-DEPTH-02 derivation, unused
select_active_targets params and target_depth.json (computed, ignored); comp selection: never
returned cand_mask + Filter-A logs, det_mask (DET rows never pass D3), BO CVn debug prints,
_select_comps_tiered + deprecated alias, _flux_series_to_mag_bjd, _common_mode_detrend_mag_lcs,
_target_row_is_vsx_known_variable, comp_pool_rms._norm_id_series. Funnel log: dao_detected counts
real DAO joins. UI badge names the actual skip reason. Phase-2A-only re-run of an old
active_targets.csv maps legacy reasons and keeps MASTERSTAR saturation final (review fix).

**Test** (container, 532 every 7th frame = 18 frames, full run from import, e62c454 vs new):
Phase 1 targets 297 -> 255 (= rule count); comparison sets identical for all 255 common targets;
magnitudes identical (max |dmag| 0), errors within 0.12 mmag (night floor from a smaller comp
union); the 34 light curves no longer produced were all RED (30 catalogue rows, saturated stars).

**Open (not changed, Milan to decide):** identity catalogue truncation - optics-floor cone uses
header XPIXSZ (already binned) x binning -> 13.64 deg instead of 4.68 deg; brightest-100 000 cap ->
532 identity catalogue G <= 13.46; fixed 4.5" final match tightening (0.46 px at 9.8"/px, switched
on/off by a 92 % rule). Per-target Phase-1 frame accumulation whose SAT/EDGE "filters" do not
change the selection (pool already gates; ~48 s per target).


## CATALOG-FOOTPRINT-2026-10 (Milan 2026-10-13)

**Rule.** The Gaia catalogue for MASTERSTAR identity is every local Gaia DR3 star in the cone of
the solved WCS footprint (`_effective_field_catalog_cone_radius_deg`), no brightest-first row cap.
The SQL box covers the cone at any declination (`cone_sky_box`: RA half-width r / cos(dec) at the
edge nearest the pole, all RA when the cone reaches a pole) and across RA 0 (`_ra_intervals_deg`,
`query_local_gaia` per interval). The cone cache is invalidated by spec + radius.

**Why.** Draft 532: the optics-floor cone used header XPIXSZ (already binned in ASCOM / N.I.N.A.
headers) x binning -> 13.64 deg instead of 4.68 deg; with the 100 000-row brightest-first cap the
identity catalogue stopped at G 13.46 (Milan's real run: 6297 identities, none fainter). The RA
box was +-r (at Dec 61 a 4.7 deg cone lost the chip edges); a field across RA 0 read nothing.

**Removed.** `_gaia_catalog_cone_radius_optics_floor_deg`, `_catalog_df_cap_brightest_by_mag`,
FOV-hint chains (plate_solve_fov_deg in the catalogue / export paths; the solver keeps its own),
dead gn/gq/gg arrays, `_icrs_center_radius_from_hdr_data`, `_export_first_icrs_center_radius`.

## IDENTITY-2026-10 (Milan 2026-10-13)

**Rule.** DAO -> Gaia identity radius = max(1 px, full-resolution FWHM), the same physical
tolerance on every rig. Detections are taken brightest first; each takes the BRIGHTEST unclaimed
Gaia star in the radius (it dominates the detected flux), never a fainter neighbour that happens
to be nearer. Second pass: a Gaia star more than 2.0 mag fainter than the detection flux implies
(frame ZP = median(G + 2.5 log10 flux) over pass-1, non-variable identities) is rejected; known
variables (VSX / Gaia variable at the position) are exempt (a Mira near maximum). One path for any
catalogue size (the < 120-row nearest-neighbour fallback is gone). A DAO detection must lie on the
chip (`detection_on_chip`; 532 had x = -0.4 identified as G 17.3 and its forced position wiped the
aperture photometry of every frame). The DAO flux sign is not a criterion: it is summed against the
global sky, 532 has 1 671 real faint stars with DAO flux <= 0 (95 % positive aperture flux). Full-resolution
FWHM is used consistently (gate, S/N aperture, reconcile, meta); DAO may run 2x binned.

**Removed.** fixed 4.5" final tightening and its 92 % switch (0.46 px at 9.8"/px, 8 px at
0.56"/px), max(12", 3 FWHM) radius, HISTORY-parsing solve RMS (now VY_SOLRM), weak-match warning
(could not fire), `nearest_sky_nn_kdtree`, `wcs_distortion_log_suffix`, the config override of the
gaia_db_path argument (the argument wins, config is the fallback).

## WCS-CONFIRM-2026-10 (Milan 2026-10-13)

**Rule.** Every MASTERSTAR WCS is verified by astrosolve. A WCS handed over from another MASTERSTAR
(sibling filter, OSC channel) goes through `astrosolve.confirm_fits`: Bayesian log-odds against the
Gaia stars of its own footprint (accept at ln 1e9), then the same refinement as a solve (1:1
matching, TAN+SIP by BIC). A WCS that does not verify stops the build - it would give wrong
identities.

**Why.** Sibling / OSC MASTERSTARs skipped the solve and then ran the astrometry optimizer
(re-match with fixed arcsec radii 5" ... 60", nearest pixel, brightest-100 000 Gaia rows) and the
catalog_match TAN-NN refit (match radius up to 0.88 x chip size) - rig-dependent identities.

**Removed.** `astrometry_optimizer.py` (1 567 lines), the platesolve-pair merge, the second
post-merge identity gate, the TAN-NN WCS refit loop in catalog_match, `assert_inv_match_identity_01`.

## MASTER-LOCK-2026-10 (Milan 2026-10-13)

**Rule.** Per-frame catalogues are matched only to the MASTERSTAR catalogue of the setup
(identities from MASTERSTAR; positions / fluxes from the frame, forced photometry at MASTERSTAR
positions) - also for sibling-filter and OSC-channel setups, which before ran an independent
per-frame Gaia-cone identity. Without a MASTERSTAR the alignment stops early with a clear message.
Aperture photometry measures rows with a finite position; one bad row no longer aborts the frame.

**Removed.** per-frame Gaia-cone path (`use_master_fast_path`, `master_only_mode`, cone prefetch
`_prefetch_export_shared_catalog_for_process_pool`, kd-pack state), unread worker state
(`ref_wcs`, `masterstar_data_shape`), dead `catalog_local_gaia_only` parameter chain.

## PHASE1-ONE-PASS-2026-10 / COMP-NSS-2026-10 (Milan 2026-10-13)

**Rule.** `comp_frame_stats`: one pass over the per-frame catalogues gives n_frames / n_peak /
n_over per comparison candidate. With a verified saturation level, a comparison whose raw peak is
>= 0.80 SAT in more than 10 % of its frames (n_peak >= 10) leaves the global pool
(COMP_SAT_80_PER_FRAME). Gaia `non_single_star > 0` is no comparison star (the old rule read a
`gaia_nss` column that is never written). Per-target per-frame accumulation only when the pool has
no RMS (fallback).

**Removed.** `compute_global_pool_rms_map` (field pass whose filters only fed a diagnostic
column), per-target SAT/EDGE "filters" that did not change the selection, GS11 comp dilution cut,
`exclude_gaia_extobj` + QSO/GAL checks (columns never written), `max_fwhm_factor`,
`rms_outlier_sigma`, config keys gs11_comp_max_dilution / phase01_comparison_exclude_gaia_extobj /
phase01_comparison_max_fwhm_factor (KNOWN_REMOVED_KEYS), the never-produced "nonlinear" zone.

**Test (2026-10-13, container).** Identity on Milan's MASTERSTARs (G-bin completeness of on-chip
Gaia): 532 (Zeiss 200 mm, 9.8"/px) 6 297 -> 11 634 identities, G 13-14 28 % -> 72 %, G 14-15
0 -> 17 %; BO CVn 2 362 -> 2 420; S4 Cyg g / r (Newton 300/1200, 0.56"/px) 2 605 -> 2 592 /
2 830 -> 2 811 (the 2-mag rule drops a few blends). Handed-over WCS: S4 Cyg r given the g WCS
(4-5 px off) -> confirmed, 0.000 px from its own solve, catalogue identical row for row (5 010);
own WCS shifted 2-5 px / rotated 0.05 deg on 532, BO CVn, S4 -> confirmed to < 0.003 px; wrong
WCS (300 px shift, 20 deg rotation) -> rejected. 532 every 7th frame (18) full run e62c454 vs new:
same 255 targets; comparison sets differ (complete catalogue: COMP_BLEND_NEIGHBOUR_LIGHT now sees
the faint neighbours, 2 210 -> 989 pool; 7 Gaia binaries no longer comparisons; field aperture
k 1.25 -> 1.00 by the LOO rule); light curves: median level shift -5 mmag (90 % within 62 mmag;
the largest, +0.8 mag, is a target blended with a brighter neighbour inside the larger aperture),
point-to-point scatter 29.7 -> 28.8 mmag, |mag - G| 0.242 -> 0.225 mag. Runtime: Phase 1 + 2A
106 -> 35 min; MASTERSTAR stage 15 -> 26 min (the DAO-Gaia certificate diagnostic and census now
see every on-chip Gaia star to G 17.5, ~130 000 on 532). Full suite: 1 764 passed, the 9 failed +
2 errors are the known environment set (fail on HEAD too).

**Open (Milan to decide).** (1) Certificate diagnostic `_diagnostic_identity_separations_px` runs
pass-2 DAO on every unowned on-chip Gaia star (all G, `target_depth_g` / `edge_margin_px` are
accepted but unused) -> ~11 min on 532; limit to the production pass-2 depth? (2) Census / forced
seed depth 17.5 is a constant (M1-amend) - on a wide rig (G_lim50 13.9) 3.5 mag below the limit.
(3) ~500 532 identities at G > 15 are blends (brightest Gaia star in 1.5 px, within the 2-mag rule).
(4) Per-frame DAO match radius 2.5 px (floor 3") is harmless under the MASTERSTAR lock (positions
locked to MASTERSTAR) but is still a fixed arcsec floor. (5) dev/scripts one-off session scripts
still pass removed parameters (not production).

## FRAME-DEPTH-2026-10 (Milan 2026-10-14) - IMPLEMENTED (step 1 limit + depth, step 2 coverage + online top-up, one depth)

**Rule.** The faint limit of the DAO <-> Gaia matching phase comes from the MASTERSTAR itself, no
static magnitude parameter:
1. Plate solve (astrosolve, bright stars only), identity on the local Gaia DB.
2. Completeness fit DAO vs Gaia by G (erf, Fleming 1995; already computed as G_lim_50, now used):
   frame limit G_lim = G50 + 3 sigma, capped by the depth the DB has in this field.
3. When the DB ends before the completeness drop (fit censored), the limit is first estimated
   from the frame: flux -> G zero point of the identified stars and the flux of the faintest
   detections at the detector S/N threshold; then Gaia is fetched to that estimate (step 4) and
   the fit is repeated - it gives the final limit.
4. If G_lim is deeper than the DB in this field, VYVAR fetches the missing Gaia DR3 stars online
   (field + 1 deg around the edges, down to G_lim) and writes them into the main Gaia DB. Offline:
   the run continues with the local DB and logs / reports that identity is limited by DB depth.
5. A coverage table in the main DB records region + complete depth (no refetch for the
   observer's repeated fields; census never counts "missing in the DB" as "not detected by DAO").
   The existing DB is registered as "all sky to G 17.5".
6. G_lim replaces the constant 17.5 (census, forced seeds, certificate diagnostic) and the config
   key masterstar_gaia_census_target_depth_g (15) for pass-2. G_lim is written to pipeline meta
   and the report.

**Comparison stars.** The Gaia NSS (non_single_star) exclusion is dropped: a binary that does not
vary is a valid comparison; what matters is that a comparison is not variable (VSX, Gaia
variability flag, own stability test - unchanged).

**Gaia files.** One main Gaia DB (depth / declination range chosen by the user at install) plus
`vyvar_astro_index/` (plate-solve index derived from it). The old solver index (gaia_triangles_*.pkl)
is removed by Milan; the larger clean-up (old solver code) comes later.

**Step 1 implemented (2026-10-14).** `frame_depth.measure_frame_depth` on the pass-1 detections
(brightest-first 1:1, chance-corrected completeness; detector estimate from G = ZP_peak - 2.5 log
peak at the DAO threshold). The matching depth (min(G_lim, catalogue)) drives pass-2, catalogue
membership, census, forced seeds and the certificate diagnostic (which now honours target_depth_g /
edge_margin_px). Written to pipeline_meta `frame_depth` and the report facts. Config keys
masterstar_gaia_census_target_depth_g and phase01_comparison_exclude_gaia_nss removed.
Test: 532 fit G50 13.88 / sigma 0.57 -> limit 15.57; with the DB cut at G 15 / 13 the detector
estimate gives 15.35 / 15.55. BO CVn fit 15.28; cut 15 / 13 -> 15.14 / 15.02. S4 Cyg 0.8 m (g / r):
19.1 / 19.1 (DB 17.5 -> "limited by the catalogue"); cut 15 / 13 -> 19.2-20.0 / 19.0-19.1. Fit vs
detector on uncensored data: 532 15.57 vs 15.54, BO CVn 15.28 vs 15.15. 532 every 7th frame full
run: MASTERSTAR stage 26 -> 17.5 min (OLD 14.7), photometry 34 min (OLD 106); same 255 targets;
pool 989 -> 1081 (binaries back); LC scatter 29.1 mmag (OLD 29.7), level shift median -3 mmag.
Note: on S4 the measured completeness already drops to 77 % at G 17.25 while the detector model
says ~98 % (crowding / blends) - the detector estimate is the deep side, acceptable for fetching.

**Step 2 implemented (2026-10-14).** `gaia_coverage.py`:
* table `gaia_coverage` in the main DB (box RA/Dec, g_max, source, n_stars, time). A DB built
  before it is registered on first use as RA 0-360, Dec MIN..MAX(dec), G <= MAX(g_mag);
  `GAIA_DR3/build_gaia_catalog.py` records its own box + limit after a complete build and shares
  the schema / insert code with VYVAR (one definition).
* depth of a field = the deepest record whose box contains the whole field box (cone box of the
  chip footprint); a field outside the Dec range has no depth (TODO-GAIA-COVERAGE-DEC).
* MASTERSTAR: if the frame limit is deeper than the field depth, `ensure_field_depth` fetches
  `gaiadr3.gaia_source` for the field + 1 deg, g_from < G <= frame limit + 1 sigma (the catalogue
  then holds the whole completeness drop, so after the top-up the fit - not the detector estimate -
  gives the final limit; BO CVn with the DB cut at 15: estimate 15.14, fit 15.28), INSERT OR
  IGNORE into the main DB, records the box, and the catalogue match is repeated. Offline / archive
  down: the run continues with the local DB, pipeline_meta `gaia_topup` + report row say so. The
  coverage row is written only after all rows are stored.
* identity only to the matching depth: a detection never takes a Gaia star fainter than the frame
  can detect (known variables exempt). With the DB to 17.5, 532 had 335 such chance identities
  (frame limit 15.57) and the result depended on how deep the DB was.
* the matching phase (pass-1 residuals, pass-2 seeds, point-source q floor) sees the catalogue to
  the matching depth only, and its tolerances come from the identities: residuals within the
  identity radius max(1 px, FWHM) (532: p95 0.60 px -> match / lock radius 2.0 px). The former
  "coarse" radius (from catalog_match_max_sep_arcsec) let chance neighbours into the p95 (3.4 px with
  the DB to 17.5, 6.3 px to the frame limit -> match radius 5.5 / 10.5 px at FWHM 1.5 px), so the
  pass-2 seeds followed the DB depth (532: 8 VSX / Gaia variables G 13.8-14.9 lost their pass-2
  detection). Pass-2 seed = Gaia star with no pass-1 detection within centre tolerance + identity
  radius (nearer, pass-2 only finds that pass-1 detection again: identity radius alone gave 419
  such duplicates on 532). match_sep_arcsec is now only recorded in the meta.

Verification (2026-10-14, MASTERSTAR builds, simulated archive = the full local DB): 532 DB to 17.5
vs DB cut at 13 + top-up (to G 16.0, 290 709 stars, 10 s): P1 11022 / 11020, P2 342 / 343,
membership 23457 / 23458, frame limit 15.573 both (completeness fit). BO CVn DB 17.5 vs 13 + top-up
vs 15 + top-up: identical (P1 2279, P2 159, membership 1709, limit 15.277). Offline (real, no route
to the archive): S4 Cyg g / r limit 19.11 / 19.14, top-up 'offline', matching to the DB depth 17.5;
532 with DB 13 offline: matching depth 13.0, reported. 532 night run (every 7th frame): same 255
exported light curves as step 1 (offset 0.000, scatter 29.1 mmag, OLD 29.7); active targets 903 ->
900: 7 dropped are G 15.7-16.8 (beyond the frame limit, chance identities), 4 added G 13.0-15.2
(pass-2 recoveries); 8 VSX / Gaia variables G 13.8-14.9 that the intermediate version had lost
are back as pass-2 detections. Full pytest: 7 failed + 2 errors, all environment (baseline).

**One depth (2026-10-14, Milan: "nech to je ujednotene").** Removed: the global MAX(g_mag) of the
DB as a depth (`get_gaia_db_max_g_mag` and its caps in query_local_gaia, VSX bbox, blind verify),
the MASTERSTAR floor `faintest_mag_limit` 18 (whole parameter chain, always None from night_run),
the second Fleming fit + censoring of `dao_reconcile` (G_lim_50/90 now = frame G50 / G50 - 1.28
sigma; the reconcile counts DAO detections, not catalogue-membership rows, which had made the final
catalogue "100 % complete" and its G_lim_50 equal to the membership edge), the duplicate reconcile
in catalog_match, the SNR5 frame limit + row-cap logic of `crowding_index` (reads `frame_depth`),
the dead report helpers vsx_limit_vs_depth_status / load_field_depth_metrics. DAO_ONLY confirmable
depth = min(catalogue depth of the field, matching depth). The VSX cross-match diagnostic reports
the catalogue depth of the frame box. Provenance fingerprints the coverage record.


## COMP-SELECT-2026-10 (Milan 2026-10-17) - KEPT AS IS (study closed) + COMP-TIER4-2026-10 cleanup

**Decision.** Comparison selection stays as implemented (COMP-ASSIGN-03): pool from MASTERSTAR
(DETECTED, not VSX/Gaia variable, isolation, saturation, frame coverage and edge gates), colour
class first (0.15 / 0.30 / 0.55 / comp_max_delta_bprp 0.79; the first class holding
n_comp_min), inside it LOO RMS, then colour, then distance; 3..8 comps, a target may end with
fewer than 8. Red targets without n_comp_min inside the cap: T4_FALLBACK ranked by |dBP-RP|,
flagged `color_rms_t4_fallback` - the user knows the target is very red. Aperture stays
`comp_loo_scatter_min` (one k per target per night, r = k x FWHM of each frame, same k for
target, comps and check star). No colour-extinction (k'') correction: colour-matched comps make
it unnecessary. No stored comp sequence: selection is per run on the actual frames.

**Why (offline study on drafts 533/534/535/536, project doc VYVAR_COMP_SELECTION_STUDY_20261017).**
- Aperture by target S/N instead of comp scatter: rms OOE median 0.91-1.00 x today, worse by
  >2 % for 9-31 % of targets; MERR lower by construction (it minimises that error model); it
  sits at the lower grid edge k = 1.0 where PSF / centring systematics were not tested.
- Variable brightness does not argue for changing the aperture: same r for target and comps per
  frame keeps the enclosed fraction equal (PSF, not brightness); on 156 real variables
  (amp >= 0.3 mag, >= 8x noise) the S/N-optimal k is the same in the faint and bright third of
  the night for 84 %, the night choice equals the faint-state optimum for 92 %, median cost 0 %.
- Measured k'' (constant field stars vs field ensemble): g ~ -14, r ~ -3, NoFilter (532) ~ -110
  mmag/airmass/mag BP-RP; relevant only for colour-mismatched comps.

**Cleanup COMP-TIER4-2026-10.** Row 4 of `comp_color_tiers` carried bprp 1.10 that selected
nothing (ladder ends at comp_max_delta_bprp; T4 = above it). Row 4 now mirrors
comp_max_delta_bprp on load and in the UI (Tier4 limit slider removed; its weight stays);
stale UI caption (COMP-ADMIT-03 "tiers no longer reject comps") and fallback tier defaults
(0.25/0.48/0.79) aligned with the real rule; registry help / config.json comments / config
guides no longer claim density adaptation (removed CLEAN-STEP5-2026-10). Test:
test_tier4_row_follows_colour_cap.

## CFG-CLEAN-2026-10-17 (Milan 2026-10-17) - config.json review and dead-code cleanup (commits 2380f15, bacddf6, stage 3, stage 4)

**Decision.** Every config key must have a reader; one key per meaning, in pixels (the unit the
code uses). Removed keys go to `KNOWN_REMOVED_KEYS` (load logs INFO once). Dead code is deleted,
not kept "for API compatibility": unused functions, unused and explicitly ignored parameters
(`_ = x`, `del x`), dead locals, dead imports and facade re-exports nobody uses. ePSF/PSF modules
are out of scope and untouched.

**Removed config keys.** 9 unit companions (*_arcsec / *_fwhm / qc_max_hfr_fwhm_ratio, CFG-UNITS),
auto_fwhm_k_min/_max, comp_tier_weights and the `w` column of comp_color_tiers (ensemble weights
are Broeg; COMP-SELECT's "its weight stays" no longer applies), forced_photometry_centroid_bound_fwhm,
saturate_level_fraction. UI weight sliders removed. psf_adaptive_resolve_fwhm stays: only the PSF
research scripts read it (PSF out of scope).

**Wired.** GS11 dilution per target in the PDF report and in AAVSO NOTES (GS11D=<D>).

**Bugs found by the cleanup.** NightRunParams.max_control_points / max_extra_platesolve,
astrometry_api_key and platesolve_backend were passed through to the aligner and ignored there
(alignment_max_control_points from config is the only control). Color-term fit ran the polyfit
twice (first result discarded). Several values were computed and dropped (night_run, importer,
masterstar_build, catalog_match, sparse_trust_core) - removed.

**Verification.** Draft 526 rerun from MASTERSTAR after each stage: light curves, AAVSO/VarAstro
exports, summaries byte-identical to the baseline; only timestamps, paths and the provenance echo
of removed keys differ. Static checks: pyflakes clean (no undefined names, no unused locals),
no **kwargs dict carries an unknown key, no initializer/submit/partial arity mismatch.

## BPM-REMOVE-2026-10 (Milan 2026-10-17) - dark bad-pixel map retired

**Decision.** VYVAR does not use a bad-pixel / bad-column map. `bpm_dark_mad_sigma` is removed
(KNOWN_REMOVED), with everything that read the `*_dark_bpm.json` sidecar: the column flag
`on_bad_column` (per-frame catalogs, masterstars CSV, comparison exclusion "Bad column", sat_diag),
`bad_columns_for_light_frame`, and the master-dark path that was carried from the night run through
alignment and catalog export only for this. No code ever wrote the sidecar, so the flag was always 0.
`enhance_catalog_dataframe_aperture_bpm` is now `enhance_catalog_dataframe_aperture`. Verified on
draft 526: photometry outputs identical.

## MSQA-2026-10 (Milan 2026-10-17) - MASTERSTAR QA tab is informative only

**Decision.** The tab shows, per observation setup (filter / exposure / binning; the user picks
which MASTERSTAR): whether plate solving succeeded (solver, mode, log-odds, SIP order), frame size
and binning, field centre, resolution, field of view, rotation, solution RMS and stars in the fit;
DAO <-> Gaia matching (detections, matched with %, detections without Gaia, median / p95 position
residual); and the map with two layers, Show Gaia (catalogue to the matching depth) and Show
matches (matched DAO green, unmatched DAO orange), full frame or a 3x centre detail. Removed: VSX
overlay and slider, completeness / census tables, reconcile metrics, unmatched-DAO class breakdown,
Gaia query-radius warning, density / "LOCK" banners, measured-vs-reprojected toggle, ID labels,
proc-CSV overlay resolution and the WCS/DAO diagnostic expander. Logic in
masterstar_qa_plot.load_masterstar_qa / render_masterstar_overlay (test_masterstar_qa_tab).

## AP-TAB-2026-10 (Milan 2026-10-17) - Aperture Photometry tab = the PDF target page, interactive

**Decision.** The tab shows one target at a time, laid out like its PDF page (header, TRUST line,
light curve with check-star strip and UT top axis, facts, aperture cutout, field map, comparison
stars, zero-point note). The light curve is an interactive Plotly chart with three switches: Show
airmass, Show flagged points, All setups in one chart (the overlay stays; to be reworked later).
Comparison table: the Gaia DR3 id links to the star in Gaia DR3 (VizieR I/355/gaiadr3); a tier
column (Phase-1 colour class) is added. Target picker: searchable list + previous / next. Removed:
LC quality filter, green/noisy badges, field density, FWHM line, setup status panel, PDF generation
from the tab (download only), "load light curves into memory" switch, bottom metric row.

**One source.** target_page.TargetPages builds the page content for both the PDF report and the tab
(moved out of photometry_report: table preparation, LC stats, check-star facts, zero-point note,
uncalibrated test, GSPC rows, field neighbours, the facts list). The whole observation group is
loaded into memory once per draft (st.cache_resource); a target switch takes ~0.2 s. Field HRD is
a view switch instead of st.tabs (Streamlit runs every tab body on each rerun; HRD cost ~5 s).
LC stats (points used, MERR) now read the canonical lightcurve_<cid>.csv, not the absolute path
stored in the summary, which broke when a draft was opened on another computer.
Verified: draft 535 PDF text identical to the previous code except those LC stats; tab checked in
a Streamlit browser screenshot and AppTest. pdf_report.py (only used by the removed button) and
vyvar_ui_status.py deleted; find_photometry_setups / masterstar_fwhm_px / setup_label moved to
platesolve_ui_paths.

---

## INFOLOG-ONE-2026-10 (Milan 2026-10-17) - one infolog per run, Infolog tab removed

**Problem.** Drafts 533-536 each had two `infolog_*.txt` files. Since JOB-2026-10 RUN VYVAR runs in
its own process: that process wrote the run log (`night_run` -> `write_run_infolog`), and after the
job ended the Streamlit process called `write_run_infolog` again; it had no open session there, so it
dumped its own ring buffer (Streamlit-side lines plus the calibration JSON snapshot) as a second
`# partial: ring-buffer tail only` file. The run log itself repeated every milestone three times
(milestone block, direct line, `pipeline` logger line) and missed `Night run complete`, written
after the file was closed.

**Decision.** Milan reads the log outside VYVAR, so the Infolog tab, the in-memory ring buffer,
`save_infolog_to_disk`, `write_run_infolog`, the authoritative-header rewrite and
`last_job_snapshot` are deleted. The run process owns the only file: `night_run` opens it once the
draft folder exists and closes it in `finally` after the last line. Each line is written once;
milestones logged before the draft folder exists are written when the file opens. Streamlit-side
messages go to the app logger (console). Tests read the real file (`infolog_lines` fixture).

---

## VAR-TAB-2026-10 (Milan 2026-10-18) - interactive Variability tab, TESS folder

**Decision.** Every plot of the tab is plotly, nothing is a PNG:
- hockey-stick of every star; hover shows VSX name / type / period, Gaia DR3, G, BP-RP, scatter vs expected, z values;
  layers switch on and off; a click on a variable opens it;
- one table "Variable stars in this field": new candidates first (by z), then the known VSX variables the search
  recovered (VSX name, type, period, Gaia DR3 link); the old candidate table, known-variables expander and the
  candidates-only field image are gone;
- the star: night light curve with FWHM and airmass strips, Peranso export; TESS download and analysis;
- period / phase view like Peranso: period field, nudge by 1 or 10 steps of 1e-3..1e-6 d, P/2 and 2P, reset to
  the TESS fit or the VSX period, a +-0.5 % fine-scan slider, epoch T0 with "minimum -> phase 0", binned mean,
  overlay of the night, live PDM Theta (Stellingwerf 1978), periodogram click takes the peak, phased Peranso export;
  for a new candidate the adopted period goes to `photometry/variability_periods.json` and onto its PDF page.

**TESS folder.** `photometry/TESS/<gaia id>/` (was `_tess`). The analysis also writes `periodogram.csv`
(frequency, LS power, AoV Theta) and `pixel_map.json` for the interactive plots; the PNGs stay for the PDF. The
result JSON no longer stores absolute file paths (`files`); readers locate files through `tess_dir_for`.

**Shared files** moved out of the UI module into `variability_store.py` (the PDF no longer imports Streamlit).
`noise_curves` in `variability_plots.py` feeds both the PDF figure (pixel-identical on drafts 535 and 536) and the tab.

---

## HRD-TAB-2026-10 (Milan 2026-10-18) - field HRD view rebuilt, colour field replaced

**Decision.** The Field HRD view of the Aperture Photometry tab:
- HR diagram in three variants (A colour-magnitude with a spectral-class band, B Teff-luminosity, C coloured by
  distance); field stars use the BP-RP palette of the colour image; VSX variables, measured targets and labels switch
  on/off; stars without a reliable parallax get their own panel (apparent G is not comparable with M_G - before they
  were mixed into the same axes);
- one "Notable stars" table (category with confirmed/candidate, SIMBAD name, spectral and object type, Gaia DR3 link,
  G, M_G, BP-RP, Teff, log g, distance with parallax S/N, A_G, VSX variability, photometry status, RA/Dec) in place of
  the table + details expander;
- the field image as one interactive viewer (hover = Gaia data of every star, zoom, pan) with three modes: inverted,
  catalogue colour (vivid) and physical blackbody. A star picked in the diagram, table or image is highlighted everywhere.

**Colour field.** The old render was almost all white because of the field-median white point and clipped cores.
`hrd_field_render.py` takes luminance from the MASTERSTAR (asinh), colour from Gaia BP-RP splatted around each star,
and applies a hue-preserving stretch (Lupton et al. 2004): a pixel whose brightest channel would exceed 1 is scaled as a
whole, so bright stars keep their colour; colour only on light 3 sigma above the sky. `hrd_colorfield.py` and its seven
`hrd_color_*` keys are removed (KNOWN_REMOVED); images are cached per MASTERSTAR in `photometry/_hrd_cache/field_<mode>_*.jpg`.
The PDF report keeps its own annotated field image.


---

## EPSF-UPGRADE-2026-10 (Milan 2026-10-18) - photutils 3.1.1, ePSF build without the sky in the star flux

**Origin.** ePSF review on draft 535 (FWHM 2.03 px, 49 build stars): the production MASTERSTAR ePSF had an edge
artifact (edge/peak 1.9; column 0 from -3.7 to +1.9 at peak 0.35) and PSF/aperture 0.37-0.43. Two causes, both removed:

1. **VYVAR:** `_epsf_prepare_stars` cut the stars out of the raw frame and injected the sky-subtracted data into the
   private `EPSFStar._data` afterwards, so every star kept an initial flux that included the sky pedestal (and a
   failure of that injection was swallowed). Now `_epsf_sky_subtracted_stars` creates each `EPSFStar` from the
   sky-subtracted cutout (public API), so the flux the builder normalizes by is the star flux.
2. **photutils 3.0:** the builder renormalized the stars every iteration with the first-iteration flux (#2386); with
   cause 1 the build diverged after ~5 iterations. photutils >= 3.1.1 (`requirements.txt`) fixes it, plus the zero
   edge row/column (#2418) and the checkerboard from heterogeneous stars (#2419/#2465).

**Builder settings = photutils 3.1 defaults** (Anderson & King 2000; Anderson 2016): `smoothing_kernel='auto'`
(quartic kernel scaled to the ePSF FWHM, none below ~7 grid points per FWHM, i.e. none at 535), `fit_shape='auto'`
(2 x FWHM; the old fixed 5 px box used only the flat core of the FWHM 4-5 px 533 stars), `centroid_symmetry`
recentering, convergence on relative star-centre changes (95 % of the stars < 0.001 px). `maxiters=50` is a cap only:
535 converges at iteration 18 (the photutils default 10 stops it unconverged). The VYVAR `quadratic`/`quartic`
override, the `smoothing_kernel`/`builder_kwargs` arguments and the `InstrumentedEPSFBuilder` subclass of the private
`_process_iteration` are removed; the meta now records `photutils_version`, `smoothing_kernel` (`auto:<kernel>`),
`build_converged`, `build_n_excluded_stars`, `build_fit_shape`, `build_center_asymmetry_px` and `build_iterations`
(photutils `iteration_info`).

**Smoothing choice checked against truth.** Synthetic pixel-integrated Moffat (beta 3) fields at FWHM 2.0 and 2.6 px,
49 noisy build stars, 400 test stars with known flux: flux scatter 0.39 % (bright) / 1.5 % (faint) identically for
`auto`, `quadratic` and `quartic`; `quadratic` adds 0.5 % more grid-flux deficit (broader core). On the 535 MASTERSTAR
hold-out the shape residual of the bright stars is 0.021 (`auto`) vs 0.036 (`quadratic`); 49 vs 300 build stars makes no
difference. On 533 g (FWHM 4.6 px) all variants are equal.

**Build guard retired** (INV-EPSF-BUILD-GUARD-01). It dropped edge stars when photutils 3.0 raised `NonFiniteValueError`
for a NaN inside a star fit region; 3.1.1 treats such a star as a fit failure (#2423). Shown by
`test_nan_inside_a_fit_region_does_not_abort_the_build` (3.0.0 raises, 3.1.1 builds), per INV-GATE-REMOVAL.

**Gate (DEPS_POLICY, anchor suspended -> A/B on draft 535).** Photometry stage old code + 3.0.0 vs new code + 3.1.1:
171/171 light curves byte-identical (30 diagnostic JSON differ in the last float ulp). DAO detections, centroids, peak,
sharpness and roundness identical on 535 and 533 frames; DAO `flux` is now the sum inside the kernel footprint (0.80x at
FWHM 2 px, 1.00x at 4.6 px) and is used only for brightest-first ordering (top-100 overlap 100 %). RUN ePSF on 10 frames
of 535: 220/220 stars fit per frame in both; PSF/aperture 0.36-0.43 (old) -> 0.92-1.14 (new); ePSF FWHM 1.88 px
(old: not measurable); about 2 min per frame for 220 stars (old 3 min).

**Still open (next steps of the review).** The PSF/aperture ratio of the fixed MASTERSTAR ePSF follows the frame seeing
(1.14 at FWHM 1.87 px -> 0.92 at 2.69 px on 535): per-frame ePSF (step 2, star selection limited to E-type targets per
Milan), per-frame aperture correction and spatial PSF (step 3), noise model (step 4). Evidence:
`claude/VYVAR_EPSF_REVIEW_20261018.md` (project), `dev/tests/test_epsf_upgrade_2026_10.py`.

---

## EPSF-FRAME-2026-10 (Milan 2026-10-18) - per-frame ePSF, RUN ePSF fits E-type targets

**Decision.** RUN ePSF fits every science frame with an ePSF built from that frame (`build_frame_epsf`, models in
`platesolve/<setup>/epsf_frames/<frame>_epsf.fits` + `_meta.json`, list in `epsf_frames_manifest.json`). The MASTERSTAR
build (Step ePSF-1) selects the build stars (`masterstar_epsf_build_stars.csv`) and stays as the reference model for the
dashboard; the aligned frames share its pixel grid, so the same stars are cut from every frame and the builder recentres
them. Literature: ground-based PSFs change from exposure to exposure, ePSF per exposure (Anderson et al. 2006, A&A 454,
1029); photutils 3.1 warns against combining frames of different seeing.

**Fit set (Milan: per-frame ePSF is costly, limit the measured stars to E%).** `build_epsf_fit_set`: active targets
whose VSX type starts with `E` (EA, EB, EW, ELL, uncertain `E...:`) outside the `catalog_only` zone, plus their
comparison stars from `comparison_stars_per_target.csv`. Draft 535: 10 targets + 28 comps = 38 stars (was the 220-star
science set), 19 s per frame including the build. No E-type target -> fit and internal PSF light curves are skipped with
the reason in the stage output. The internal PSF light curves are written for the fit-set targets only; their provenance
header names the manifest and its sha256. `psf_photometry_stars` reads `<model>_meta.json` and takes the photometry dir
explicitly (gain / RN authority no longer derived from the model location).

**Evidence (draft 535, 146 frames, FWHM 1.88-2.86 px, same 38 stars, A/B fixed MASTERSTAR model vs per-frame).**
- All per-frame builds converged (median 19 iterations); ePSF FWHM 1.88-2.54 px, median 2.23 px; 100 % fits OK.
- Seeing trend of PSF minus aperture differential magnitude on the E-type targets: fixed model +83 to +173 mmag/px on
  the six faint (G 12-13.4) targets (r 0.26-0.53); per-frame -18 to +42 mmag/px (|r| <= 0.31, typically ~0.1).
- Frame-common PSF/aperture(2 FWHM) scale: slope -27 %/px (fixed) -> -15 %/px (per-frame); it cancels in differential
  photometry and is left to the per-frame aperture correction (next step).
- Point-to-point scatter: per-frame PSF equal to aperture on the bright targets (FW CVn 5.6 vs 5.2, BO CVn 8.8 vs
  8.7 mmag) and 10-38 % lower on the faint ones (e.g. NSVS 5096293 31.8 vs 50.2, CSS J1409 39.1 vs 56.5 mmag).

**Open.** Residual seeing trend of ~30-40 mmag/px on the faintest targets and the frame-common scale (aperture
correction, step 3); spatial PSF variation (step 3); noise model (step 4); the MASTERSTAR model's remaining role is
reviewed with the dead-code step.

---

## EPSF-SPATIAL-2026-10 (Milan 2026-10-18) - MASTERSTAR for aperture only; frame ePSF + night spatial perturbation; frame-level fit

**Decision (Milan).** ePSF is computed on every frame; MASTERSTAR is only for aperture photometry. The MASTERSTAR ePSF
model is no longer built (alignment-time build removed, RUN ePSF Step 1 only selects the build stars on the MASTERSTAR
catalogue, `select_epsf_build_stars`). PSF fitting during the per-frame catalogue export (`_fill_psf_catalog_columns`,
Moffat path, `photometry_mode`) is removed: the catalogues carry aperture photometry, PSF columns come only from RUN
ePSF. `photometry_mode` is removed (KNOWN_REMOVED).

**Build stars.** Isolated (no Gaia neighbour within 3 FWHM and 2.5 mag), unsaturated, not variable, clean source state,
edge-safe, the brightest 600 over the whole field (was: the 49 comparison stars of the science set + an INTERIM top-200
that contradicted N-policy). Draft 535: 600 stars, 58-73 per 3x3 cell.

**Model of a frame** (`epsf_frame.build_frame_model`): the frame's ePSF from the build stars + a night-averaged spatial
perturbation per cell (`night_spatial_delta`: median over every k-th frame, max 30, of cell ePSF minus field ePSF), as a
photutils GriddedPSFModel; grid 3x3 with >= 450 build stars, 2x2 with >= 200, else none (>= 50 stars per PSF, Anderson et
al. 2006). Scheme of Anderson & King (2006, ACS ISR 2006-01): stable spatial library + per-exposure core. Draft 535:
perturbation up to 4-16 % of the peak per cell.

**Fit** (`epsf_frame.fit_frame_psf`): one photutils PSFPhotometry call per frame in global coordinates, local sky
median in a 4-6 FWHM annulus, fit box 2 FWHM + 1, pass 1 errors from the data, pass 2 from the model
((sky + model)/gain + (RN/gain)^2), `psf_flux_err` = covariance error x sqrt(max(1, chi2_red)), `psf_converged` = no
photutils flag 2/4/8/16; `psf_fit_ok` from the night chi2 locus (D-EPSF-CHI2-LOCUS-01, unchanged). The locus rewrite now
uses the guarded sidecar writer (was a raw to_csv) and runs once (the stage ran it a second time). Sidecars keep only
the new PSF columns (psf_flux, psf_flux_err, psf_chi2, psf_flags, psf_converged, psf_fit_ok, psf_snr, psf_sky, x_fit,
y_fit + locus columns).

**Evidence (draft 535, 146 frames, 28 constant comparison stars, each against the rest of its ensemble).**

| method | comp RMS median [mmag] | |seeing trend| median [mmag/px] |
|---|---|---|
| aperture | 7.25 | 8.1 |
| old VYVAR fitter, 49-star per-frame ePSF (step 2) | 7.77 | 8.4 |
| frame-level fit, 600-star per-frame ePSF, no spatial | 8.02 | 6.4-7.4 |
| per-frame 3x3 grid built on each frame | 12.85 | 11.5 |
| **production: per-frame ePSF + night 3x3 perturbation** | **7.72** | **5.0** |

E-type targets point-to-point: 10-35 % below aperture on the faint ones (FY CVn 52.9 vs 68.7, CSS J1409 38.2 vs 56.5,
GH CVn 42.5 vs 54.1 mmag), equal on the bright ones. PSF cannot beat aperture on bright isolated stars (source-noise
limited, Cramer-Rao). Run time 36 min for 146 frames (perturbation 13 min). 146/146 builds converged (median 9
iterations), 5548/5548 fits OK.

**Removed:** `build_epsf_model`, `build_frame_epsf`, the MASTERSTAR ePSF model and its alignment-time build,
`_fill_psf_catalog_columns`, `_epsf_fit_catalog_ids`, `_epsf_target_catalog_ids`, `_photometry_mode_run_flags`,
`_export_catalog_psf_st_fields`, `merge_psf_into_sidecar`, the p4_none AC stamping helpers, `photometry_mode`.
Evidence scripts: /root/epsfrev/tools (exp_spatial*, exp_fitter*, ana_fitter.py); project doc
`claude/VYVAR_EPSF_SPATIAL_20261018.md`.

---

## EPSF-ERR-2026-10 (2026-10-18) - PSF flux error = fit (Fisher) error; scintillation added in the PSF light curve

**Measured** (draft 535, 28 constant comparison stars, each against the sum of the rest of its ensemble, 146 frames,
slow trends removed with an 11-frame running median; frame-level fit, gain 0.7925 e-/ADU, RN 10 e-):
scatter / predicted error median **1.17** with the photutils covariance error (absolute sigma). The bright stars
(predicted 3.5-5 mmag) show 1.3-1.5: ~4 mmag in quadrature beyond the fit error, the scintillation level of the rig.
With the error inflated by sqrt(max(1, chi2_red)) the ratio was 0.40 (errors 2.5x too large): chi2_red 5-40 on bright
stars measures the model-shape mismatch, which does not show up as flux scatter.

**Decision.** `psf_flux_err` is the fit error (no chi2 inflation; chi2 stays in `psf_chi2` for the night locus). The
internal PSF light curve adds the aperture light curve's per-epoch `err_scint_rel` (scintillation belongs to the star
and the epoch, not to the measuring method) via `combine_production_err_mag(..., sigma_scint_mag=...)`.
Test: `test_psf_lc_error_adds_the_aperture_scintillation_term`. Note: the sandbox harness runs with gain 0.25 (its DB
lacks the rig's gain), so its production runs overstate the errors by ~1.8x; the calibration above used the rig gain.

## EPSF-CLEAN-2026-10 (Milan 2026-10-18) - dead PSF code removed after the per-frame ePSF

The per-frame ePSF and the frame-level fit (EPSF-FRAME / EPSF-SPATIAL) left the old PSF code without callers or
without inputs. Removed (causes, not consequences):

- **Per-star cutout fitter** in `psf_photometry.py` (`psf_photometry_stars`, grouped fit, Moffat override, sandwich and
  full-CCD error maps, fixed-position path, residual-annulus sky, old spatial grid, quality grade, legacy FWHM
  estimator). The module keeps the build-star selection, the ePSF builder helpers and the gain/RN authority
  (3258 -> ~1100 lines). `psf_runner.py` is a thin CLI over `epsf_stage.run_epsf_stage`.
- **`psf_neighbor_sub.py`** (no production caller) with its A9 envelope harness and tests.
- **Adaptive aperture/PSF picker** (Phase 2A `lc_flux_method` routing, blend map, `crowding_targets.csv` writer,
  `_get_lc_adaptive`, `BlendMapEntry`, `_route_lc_per_frame_err`, `compute_lc_flux_method`): its inputs
  (`psf_ac_applied`, `psf_quality`) are no longer produced. The `adaptive` report/LC method is gone; AAVSO/VarAstro
  writers now refuse every method except `aperture` (INV-PSF-SUBMIT-01 tightened). `lc_flux_method` stays in the LC
  schema (= `method`); the dead `err_method` / `lc_flux_method` parameters of `save_lightcurve_csv` were removed.
- **Phase 2A / proc store PSF columns** (`psf_*`, `psf_ac_*`, `psf_quality*`): no consumer; RUN ePSF reads and writes
  the proc sidecars directly.
- **21 config keys** (KNOWN_REMOVED, a stale key in an old config.json only logs a warning): `psf_spatial_order`,
  `psf_spatial_enabled`, `psf_chi2_threshold`, `psf_grouper_enabled`, `psf_group_sep_fwhm`,
  `psf_neighbor_include_fwhm`, `psf_quality_fallback_enabled`, `psf_neighbor_sub_enabled`, the ten `neighbor_sub_*`,
  `psf_adaptive_enabled`, `psf_adaptive_resolve_fwhm`, `psf_adaptive_snr_lo`.
- Dev scripts and harnesses of the old fitter (V3d, V3e legacy comparison, sky sweep, xval_psfex epsf_core/val,
  xval_pythonphot a1b, 364 pilots); their results stay in `dev/results`.

**Provenance fixed.** The internal PSF LC header said `psf_weight_mode=full_ccd`, `psf_err_mode=sandwich_full_ccd`,
`psf_ac_policy=p4_none` (defaults of the removed fitter). It now states the method actually used:
`psf_fit=frame_psfphotometry_epsf_spatial_local_bkg`, `psf_err_mode=fisher_model_variance`; the empty `n_group`
column was dropped. PSF citations: Stetson 1987, Anderson & King 2000, Anderson et al. 2006 (A&A 454, 1029: per-image
ePSF with a spatially varying perturbation), Anderson & King 2006 (ACS ISR 2006-01: PSF library + per-exposure
perturbation); Moffat, Astier 2013, Lacroix 2025, Guy 2010, Mighell 1999 belonged to the
removed fitter.

**Bibliography fails loudly.** `load_citations_bib` returned `{}` when `CITATIONS.bib` was missing and
`citation_line` then wrote `Unknown (?) -` lines into the AAVSO/VarAstro headers and the PDF methods text (seen when a
sandbox copy of `src_py` ran without the repo root). A missing bibliography now raises `FileNotFoundError`, an unknown
key raises `KeyError` (`test_missing_bibliography_or_key_fails_loudly`). The release bundle ships `CITATIONS.bib`.

**Verified on draft 535** (photometry stage, ed11311 vs this change, same inputs): all 71 light-curve / KMAG CSVs
byte-identical; active targets, comparison stars, variability products identical; AAVSO/VarAstro data lines identical;
`photometry_summary.csv` differs only in the run directory path; `pipeline_meta.json` only in timestamps and the
removed config keys of the snapshot.

**Kept, open:** `crowding_index.py` has no production caller now (diagnostic for dense fields) - Milan to decide.

## EPSF-LC-COMPARE-2026-10 (Milan 2026-10-18) - aperture vs ePSF light curves side by side

Milan: "urcite bude dobre porovnat tie svetelne krivky aperturna a ePSF voci sebe". The ePSF tab overlaid the
calibrated aperture curve with raw PSF instrumental magnitudes (no ensemble: the transparency of the night went
into the PSF series). Replaced by `epsf_lc_compare.py` + an interactive plotly panel comparing the two
**differential** curves (same comparison stars): per star p2p scatter (1.4826 MAD of successive differences /
sqrt 2), p2p / median quoted error, and the PSF - aperture difference (its p2p and slope vs the frame ePSF FWHM).
`low_snr` aperture epochs stay in (they are measurements; the faint targets would otherwise drop out); hard
failures (saturated, edge_fail, no_data, nondetection) and PSF epochs without `psf_fit_ok` are left out.
The median PSF - aperture difference is only a level: each ensemble carries Broeg weights from its own method's
scatter (535 BO CVn -205 mmag while the PSF/aperture flux ratio of target and comparison stars agree within 2 %),
so the panel removes it and does not report it as a bias.

**Draft 535** (E-type targets, 146 epochs; harness gain 0.25 so both p2p/err ~0.5-0.7):

| star | G | AP p2p | PSF p2p | PSF-AP p2p | PSF-AP vs FWHM [mmag/px] |
|---|---|---|---|---|---|
| FW CVn | 9.2 | 5.0 | 5.0 | 1.6 | +1 |
| BO CVn | 9.7 | 7.8 | 7.8 | 3.5 | +4 |
| FU CVn | 11.0 | 12.9 | 8.9 | 7.8 | -33 |
| ASASSN-V J135757 | 12.1 | 36.3 | 27.2 | 28.5 | -48 |
| NSVS 5096293 | 12.1 | 49.2 | 29.9 | 53.2 | +50 |
| CSS J140918 | 12.7 | 58.5 | 38.3 | 50.6 | +20 |
| GH CVn | 12.9 | 50.4 | 44.2 | 27.4 | +9 |
| FY CVn | 13.2 | 70.8 | 51.1 | 49.6 | +18 |
| FZ CVn | 13.4 | 88.2 | 77.7 | 56.3 | +50 |

Bright targets: the two methods agree to 2-4 mmag; fainter than G ~11 PSF lowers the scatter by 12-40 %.
Tests: `dev/tests/test_epsf_lc_compare.py`.

## CROWDING-INDEX-REMOVE-2026-10 (Milan 2026-10-18) - crowding_index.py removed

Milan: "ak je to useless funkcia tak zmazat". After the adaptive picker went (EPSF-CLEAN-2026-10) nothing in the
pipeline, UI or reports called `crowding_index.py`; `crowding_index.json` was no longer written. Neighbour light of
comparison stars is measured by `comp_blend.py` (BLEND-K-2026-10, 1 % gate). Removed with it: the validation items
A1/A2 (they tested only this module's own blend flag) and `crowding_audit_367.py`.

## EPSF-BLEND-2026-10 (Milan 2026-10-18) - joint PSF fit with Gaia neighbours, 12xFWHM model

Milan: ePSF must handle a faint star next to a bright one, where aperture photometry fails ("ano pokracuj na
tie body 1-5"). Globular clusters are a job for a finer-scale rig (300/1200, C3-26000), not the wide rig.

**Finding.** RUN ePSF fitted only the fit set, each star with a free position and no neighbours: a faint
star next to a bright one slid onto the neighbour (535: up to 3 px) and came out 2-4 mag too bright.
The DAO-Gaia match itself was correct (a faint star's DAO centroid is pulled ~0.6 px by the wing).

**Decision (measured on draft 535).**
1. `fit_frame_psf` fits the fit set together with every Gaia star down to the frame limit
   (`frame_depth.g_lim_frame`) whose model reaches a fit star's fit box (`epsf_frame.frame_fit_sources`);
   stars closer than two fit boxes form a group (photutils `SourceGrouper`). Positions fixed at Gaia
   through the MASTERSTAR WCS (frames are aligned to it), only fluxes fitted.
2. ePSF cutout / model = 12 x FWHM (was max(15, 5 x FWHM)): the growth curve of isolated bright stars
   reaches 0.94 at 7 px and levels off at ~12 px.
3. Tested and **rejected**: sky from the residual image after subtracting the neighbours (comp RMS 7.31 vs
   6.85 mmag, blend bias unchanged); build stars with <= 1 % neighbour light in the cutout (288 instead of 600
   stars, 2x2 grid, comp RMS 7.18 vs 6.91 mmag, blend bias unchanged - EPSFBuilder's sigma clipping removes
   the contamination). The 3xFWHM / 2.5 mag build-star isolation stays.

**Injection-recovery** (10 frames; a real isolated star's stamp, scaled 2-5 mag fainter than a G 8.5-10.5
host, placed 3-12 px from it, photon noise added; bias relative to isolated injected controls):

| separation | old PSF fit | aperture | joint fit, 15 px model | joint fit, 25 px model |
|---|---|---|---|---|
| 3 px | -3.10 mag | -2.80 mag | -169 mmag | -84 mmag |
| 4 px | -2.85 mag | -2.16 mag | -109 mmag | -15 mmag |
| 6 px | -0.44 mag | -0.80 mag | -134 mmag | +35 mmag |
| 8 px | -0.12 mag | -0.27 mag | -119 mmag | +35 mmag |
| 12 px | 0 | +0.02 mag | +13 mmag | +12 mmag |

Real Gaia blends (14 stars G 11.5-13.5 with a >= 2.5 mag brighter star at 3-8 px, residual vs Gaia G):
old fit -0.82 mag (spread 1.26), joint fit + 25 px model -0.01 mag (spread 0.22).
Full RUN ePSF on 535 (146 frames, 5548/5548 fits OK, 36 neighbours fitted with the 38 fit-set stars):
comp RMS 7.72 -> 7.64 mmag, seeing slope 8.6 -> 8.0 mmag/px; the isolated E targets unchanged within
1 mmag except FZ CVn p2p 77.7 -> 72.2 mmag. Runtime 49 min (was 36).
Tests: `test_faint_star_next_to_a_bright_one_needs_the_joint_fit`,
`test_fit_sources_use_gaia_positions_and_catalogue_neighbours`, `test_build_stars_isolated_and_model_covers_the_wings`.

## EPSF-SAT-ERR-2026-10 (Milan 2026-10-18) - saturation mask, honest errors for blends

**Saturation.** `fit_frame_psf` masks pixels at or above the frame's `saturate_limit_adu_85pct` (proc
sidecar, INV-SAT-LIMIT) and non-finite pixels; a saturated core is fitted from its wings. Injections on 535
(real-star stamps clipped at 65535, 10 frames): peak 1.5 / 2.5 / 4 / 7 x the limit -> without mask +62 / +258 /
+519 / +851 mmag, with mask +40 / +87 / +101 / +114 mmag (a larger fit box did not help: 2xFWHM +140, 3xFWHM
+157 mmag at 7x; clean build stars neither). `psf_sat_masked` marks such fits; the internal PSF light curve
drops them (`psf_epoch_drop_reason=target_saturated`, header `psf_lc_n_epochs_target_saturated`), as the
aperture light curve drops saturated epochs. The mask matters most for saturated neighbours of a fit star.

**Errors in blends.** On the injections the Fisher error described isolated stars (pull 1.21) but not blends
(pull 6.3 / 5.0 / 2.3 / 1.5 at 3 / 4 / 6 / 8 px): the model-shape error of the neighbour's light is not in it.
`psf_flux_err` = sqrt(fisher^2 + (0.113 sqrt(f_nb) F)^2), f_nb = model light of the other sources in the fit box
/ own model light (`psf_nb_light_frac`); `psf_flux_err_fit` keeps the Fisher error. k = 0.113 fitted on frames
0, 15, ..., validated on frames 6, 21, ...: pull 0.98 / 1.28 / 1.05 / 1.02 at 3 / 4 / 6 / 8 px. Calibrated on the
wide rig only - to be re-checked on a new rig. Full RUN ePSF 535: unchanged (no fit-set star saturated or
blended: f_nb < 0.1 everywhere, comp RMS 7.64 mmag).
(Superseded by EPSF-BLEND-ERR2-2026-10: sigma_rel = 0.040 f_nb, calibrated on two rigs.)
Also measured: field-only ePSF (no spatial grid) doubles the blend scatter at 3-4 px (0.77 vs 0.35 mag);
oversampling 4 vs 2 changes the blend bias within its noise and does not converge in 50 iterations - kept 2.

## EXPORT-GUARD-TOL-2026-10 (2026-10-18) - INV-EXPORT-READ-ONLY-01 compares values, not a digit hash

RUN ePSF on Brno draft 533 g tripped INV-EXPORT-READ-ONLY-01 although no non-psf column was touched: the
CSV writer keeps ~11 significant digits, so a value on a rounding tie (flux_k125 2.5538216505083e-05 ->
2.5538216505e-05) changed the `.10g` fingerprint of the re-read file. Cause removed: `hash_non_psf_columns`
replaced by `non_psf_columns_equal(before, after, rtol=1e-9)` - columns whose entries are all numbers on both
sides (object columns from a pyarrow round-trip included) compare numerically with rtol 1e-9; text columns and
integer IDs longer than 15 digits (Gaia source_id - a tolerance would hide a changed ID) compare as exact text.
Real-data check: 132 proc CSVs of drafts 533/534 written and re-read - 0 trips; a changed value, flag or ID is
still caught (dev/tests/test_export_parity_01.py).

## EPSF-FOOTPRINT-2026-10 (2026-10-19) - ePSF and PSF fits only on pixels the aligned frame covers

Brno 533 r and 534 g: RUN ePSF stopped in the night spatial perturbation ("Cannot normalize ePSF: the data
sum (-2.04e16) ..."). Cause: the aligner fills the reference-grid pixels the shifted frame does not cover with a
constant (533 r_0199: x < 48 px, r_0187: x < 30 px); two build stars of cell (0,0) of r_0199 lay in that fill
(flux 115 / 384 ADU vs median 7136, peak 6-8 ADU), their normalization blew up the EPSFBuilder iteration.
`epsf_frame.aligned_footprint(header, shape)` maps the raw frame's corners through `VY_AM0..8`
(ALIGN-MATRIX-2026-10) onto the aligned grid (checked on 533 r: first data column 30 / 48 px vs quadrilateral
29.4-33.8 / 47.8-51.7 px); raises when the transform is missing (all 132 Brno and 146 draft-535 science frames
carry it). The frame ePSF and every cell of the night perturbation are built only from build stars whose whole
cutout is covered; `fit_frame_psf(covered=...)` masks uncovered pixels (in the fit and in the sky annulus - photutils passes the mask
to the local background) and `psf_edge_masked` marks a star whose fit box reaches them or whose annulus keeps less
than half of its pixels (a first version that flagged any annulus contact dropped 21 of 33 epochs of a 533 g
target at y = 54 px whose fit box was always on data); the internal PSF light curve drops such target epochs
(`psf_epoch_drop_reason=target_outside_footprint`, header `psf_lc_n_epochs_target_outside_footprint`), a comparison
star's epoch fails the pin rule as before.
The aperture side followed in AP-FOOTPRINT-2026-10.

## EPSF-MERGE-2026-10 (2026-10-19) - unresolved fainter companions are merged with the fit star

Brno 533 g: ZTF J212128.80+514855.6 (G 13.41) has a Gaia neighbour G 17.15 at 2.0 px (0.43 FWHM, 1.1"). The joint
fit traded flux between the two (neighbour light fraction -0.04..+0.11 frame to frame, target 10 % brighter when it
was negative): p2p 33.9 mmag vs 9.9 mmag aperture on the same epochs. Injections on 533 g (real-star stamps,
sub-pixel placed, 2-5 mag fainter, next to G 12.5-14.5 stars, 11 frames FWHM 3.8-7.4 px), scatter of the host flux
caused by the companion at 0.25 / 0.5 / 0.75 / 1.0 / 1.5 FWHM: joint fit 110-250 / 55-70 / 20-33 / 8-19 / 15-23
mmag (pull 1.8-2.8 at 0.25: the errors do not cover it), merged 3-11 / 3-12 / 2-15 / 3-36 / 1-15 mmag (dmag >= 3;
the merged light is a constant dilution as in the aperture). `frame_fit_sources` does not fit a neighbour that is
fainter than the fit star and closer than 0.5 x the night median frame FWHM (`night_median_fwhm_px` over the group's
proc sidecars, NIGHT-FWHM-2026-10; raises without per-frame FWHM). One decision per night - a per-frame decision
would put dilution jumps into the light curve; 0.5 x the median keeps the night's best frames below the 0.75-1 FWHM
crossover (533 g best / median 3.77 / 4.50 px, 534 g 3.34 / 5.48 px). The MASTERSTAR FWHM was tried first and is
wrong here: 534 g MASTERSTAR 3.44 px gave a 1.72 px radius and left the pair (2.02 px) in the joint fit - p2p 21.3
vs 13.1 mmag aperture, PSF - aperture p2p 19.7 mmag; 533 r merged it only by 0.007 px. `merged_light_frac` (Gaia G flux ratio) goes to the sidecar as `psf_merged_light_frac`, the
pairs to the manifest (`fit_sources.merged_neighbours`). A brighter neighbour that close stays in the joint fit.
Result on that star: p2p 33.9 -> 14.8 mmag; no other star of the Brno fit sets has a neighbour within 1 FWHM.

Measured on Brno and kept unchanged:
* PSF local sky: annulus median 4-6 FWHM (A) vs catalogue stars masked 2 FWHM (B) vs median of the pass-1 residual
  image (C), 533 g, 15 E targets / 40 comps: target p2p 51.7 / 57.0 / 53.2 mmag, comp RMS 12.56 / 12.93 / 13.18 mmag.
  A stays (C was also worse on 535).
* Bright-star halo: injections 15-35 px from G 10.5-12 stars, 4-6 mag fainter, 11 frames: PSF bias vs isolated
  -14..+30 mmag at FWHM < 5 and -6..+60 at FWHM > 6; aperture -232 / -14 / +40 mmag at 15 / 20 / 25 px (FWHM < 5) and
  -1736 / -421 mmag at 15 / 20 px (FWHM > 6).

## EPSF-SYS-2026-10 (2026-10-19) - the PSF light-curve error carries the night's systematic floor

Brno 533 g comparison stars (each against the sum of the other comparison stars of its target, 33 frames): PSF
scatter over the quoted PSF error 1.55 overall and 5.1 / 2.2 / 1.7 / 1.3 / 1.3 at G 10-12 / 12-13 / 13-14 / 14-15
/ 15-17 - a multiplicative floor missing from the error, common to both methods (comparison RMS aperture 13.3,
PSF 12.5 mmag; flat field and the stars' drift over the chip, ~50 px in the night). The aperture light curve
already carries it: SIGMA-SYS-NIGHT measures it on the comparison stars (533 g: 6.4 mmag) and writes it per epoch
as `err_sigma_sys_rel`. Like scintillation (EPSF-ERR-2026-10) it belongs to the star and the epoch, not to the
method: `psf_delta_mag_err` adds it from the aperture light curve. With it the comparison-star pull is 1.10
(0.92 / 0.93 / 1.06 / 1.25 / 1.20 by magnitude). The PSF fit error itself was checked on the sky: measured sky
noise 0.83-0.86 x the CCD-equation value (gain 12.48 e-/ADU, RN 14.1 e- from the header), neighbouring pixels
correlated 0.16-0.19 by the alignment resampling.

## AP-FOOTPRINT-2026-10 (Milan 2026-10-19) - aperture photometry only on pixels the aligned frame covers

Same cause as EPSF-FOOTPRINT-2026-10: the aligned frame's uncovered border holds the aligner's constant fill. The
target edge check (`phase2a_target`) used the chip rectangle and the annulus sky median took the fill as sky.
`frame_footprint.py` (moved out of epsf_frame: `aligned_footprint`, `has_alignment_transform`, `off_footprint`) is
used by both methods. Aperture: the annulus sky median takes only covered pixels (`aperture_pertarget.
measure_flux_on_f_grid(covered=...)` - the light-curve fluxes - and the proc-CSV photometry in `photometry_shared`);
a star whose largest grid aperture reaches the fill or whose annulus keeps less than half of its pixels is marked
per frame (`ApertureGridNight.edge_by_frame`) and `apply_grid_fluxes_to_frames` sets `mag_inst` NaN,
`flag=edge_fail`. A frame without `VY_AM0..8` was not resampled (MASTERSTAR): footprint not checked, logged
(`n_frames_no_transform`). The chip-rectangle check stays for runs without the per-target grid.
Brno 533/534, all 132 frames, targets + comps (19 338 measurements): 394 changed sky (8 stars near the bottom edge,
0.07-0.66 ADU), 101 now `edge_fail`; on the epochs that stay, the flux changes by median 14 mmag, per star up to
62-134 mmag median (faint targets G 17 at y = 48-55 px), max 1.57 mag - a drift-dependent systematic the aperture
light curves of those targets carried. One comparison star (2172365188123907328, y = 48.6 px) is affected: comp
selection checks the chip, not the footprint; the ensemble handles its missing epochs (per-comp levels).

## EPSF-BLEND-ERR2-2026-10 (Milan 2026-10-19) - blend error linear in the neighbour light, checked on two rigs

EPSF-SAT-ERR-2026-10 calibrated sigma_rel = 0.113 sqrt(f_nb) on draft 535 only. Brno injections (533 g and 534 r,
11 frames each, faint stars 2-5 mag below G 11-13.5 hosts at 3 / 4 / 6 / 8 / 12 px, real gain 12.48 e-/ADU and RN
14.1 e-; the faint star drawn from the frame's local model with Poisson noise - a real-star stamp moved across the
Brno field carries the field's PSF variation, isolated pull 2.67 instead of 1.16): pull without the blend term
21 / 20 / 12 / 6.3 / 2.3 - the term is needed - but with 0.113 sqrt(f_nb) 1.35 / 1.02 / 0.70 / 0.65 / 0.45. A model-
shape error is a fixed fraction of the neighbour's light in the box, sigma_F = k F_nb, i.e. sigma_rel = k f_nb.
Power law k f_nb^a fitted on both rigs: a = 0.9-1.0. With a = 1, k fitted on 535 alone 0.0406, on Brno alone 0.0399;
each tested on the other rig: Brno 1.05 / 0.95 / 0.78 / 1.13 / 1.13, 535 0.99 / 1.03 / 1.17 / 0.98 / 1.22.
`_BLEND_ERR_K = 0.040`, `psf_flux_err = sqrt(fisher^2 + (0.040 f_nb F)^2)`. Real Brno fit sets (7920 fits): median
change of the error 0, 1 % of the fits (bright stars with a faint neighbour, f_nb 0.01-0.05, where the old sqrt term
dominated) 46-71 % smaller; comparison-star pull with the night floor 1.10 / 1.11 / 1.20 / 1.21 before, 1.11 / 1.12 /
1.20 / 1.21 after (533 g / 533 r / 534 g / 534 r).

## EPSF-TARGETS-2026-10 (Milan 2026-10-19) - RUN ePSF measures the selected catalogue variables; ePSF tab

Replaces the E-type-only fit set of EPSF-FRAME-2026-10. Candidates (`epsf_science_set.epsf_target_candidates`):
targets of `active_targets.csv` with `catalog` VSX whose aperture light curve exists - targets out of the configured
type scope (`vsx_out_of_scope_types`) or skipped for S/N / saturation have none, `catalog_only` is left out;
exoplanet hosts and the variability-search candidates are not catalogue variables. Selection (`build_epsf_fit_set(
platesolve_dir, selection)`): `ALL` (or empty), or comma / semicolon / newline separated tokens - a VSX type token
(`EA` matches `EA`, `EA:`, `EA|RS`, same tokenizer as vsx_type_scope), a star name or VSX name, a catalog id; `*`
and `?` wildcards (`E*`, `ZTF*`), case-insensitive. Unmatched tokens are reported. New config key `epsf_targets`
(default `ALL`) is used when `epsf_auto_run` runs the stage after the aperture photometry (night run and RUN VYVAR
from the UI, both `run_night_pipeline`); `run_epsf_stage(targets=...)`, the ePSF tab and `psf_runner --targets`
override it. The stage builds the fit set once and hands it to `run_epsf_psf_merge_job(fit_set=...)` (the job no
longer builds its own; its never-used `write_internal_lc` branch and `write_internal_psf_lightcurves_after_epsf_job`
were removed). Manifest `fit_set` records `selection`, `unmatched_tokens`, `target_names`.
ePSF tab (`ui_epsf_dashboard.py`, rewritten): config state, target field with a live preview (n targets + comps,
unmatched tokens, table), RUN ePSF (job carries `epsf_targets`), then the ePSF light curve of each measured target
with the aperture light curve as an option (PSF drawn at the aperture level), a ePSF-vs-aperture scatter table and the
per-frame model FWHM in expanders. Removed with it: the per-star metrics table, the three-panel comparison figure
and the separate "Write internal PSF light curves" button (RUN ePSF writes them).
Brno 533 g: ALL = 54 targets + 79 comparison stars (was 15 + 40 for E types).

## EPSF-MULTISETUP-2026-10 (Milan 2026-10-19) - RUN ePSF runs every setup of the night

The ePSF tab measured only the setup chosen in its selectbox; a night with g_120_4 and r_100_4 needed two runs.
The tab now builds one plan for all setups of the draft (`ui_epsf_dashboard.epsf_run_plan`: per setup its paths,
`build_epsf_fit_set(selection)` and what is missing) and shows it as a table (targets, candidates, comparison
stars, status); a token is reported as unmatched only when no setup matches it (EW may exist only in r). RUN
ePSF puts every runnable setup into one job (`pending["setups"]`, same `epsf_targets`); `app.py` runs
`run_epsf_stage` per setup in turn (progress `setup k/n <name>`) and reports one line per setup. The setup
selectbox now only chooses the light curves shown. The night run already ran the stage per obs group.

## EPSF-PDF-2026-10 (Milan 2026-10-19) - ePSF page right after the target's aperture page

When a target has an ePSF light curve (>= 5 kept epochs), the PDF report adds a page right after its aperture page
(full and compact layouts): aperture curve with the ePSF curve drawn at the aperture level, a strip with
ePSF - aperture per epoch [mmag], and a table - ePSF epochs and drops (saturated / off data / frame not fitted or a
comparison star missing), common epochs, p2p scatter and median error of both methods, ePSF - aperture p2p and
slope vs seeing, comparison stars, unresolved neighbour, ePSF models, the run's selection
(`star_page_m3.epsf_page_data`, `plot_lc_epsf`, `photometry_report._report_epsf_star_page`). The PSF overlay on the
aperture light curve image and its helpers were removed. Fixed with it: `epsf_lc_compare.load_pair` read the
aperture `err` (relative flux) as magnitudes - now x 2.5/ln10; `pair_stats` takes scatter and median error of
both methods on the common epochs (on a variable star the p2p grows with the time step, so an aperture curve with
98 epochs and an ePSF curve with 33 were not comparable: 13.0 vs 32.5 mmag -> 28.8 vs 32.5 mmag on Brno 533 g,
ZTF J212030.85+514732.6, sandbox copy with 33 of the 98 frames). Brno 533 g E*: 15 ePSF pages, each after its aperture page.

## EPSF-EXPORT-2026-10 (Milan 2026-10-19) - ePSF light curve export for the user

RUN ePSF writes for each measured target `photometry/lightcurves_reports/epsf/<name>_<YYYYMMDD>_<setup>_epsf.txt`
(`epsf_export.write_epsf_user_lightcurves`, called by `run_epsf_stage` after the internal LCs): columns
`bjd_tdb,delta_mag,err` (kept epochs, sorted), header with star, catalog id, VSX type and period, coordinates, G,
observer code, setup / filter / exposure / binning, night, time system, the meaning of delta_mag and err, method,
ePSF frames and FWHM, model sha256, comparison-star ids, unresolved neighbour, epoch accounting, p2p scatter,
selection, software version. A rerun replaces the target's file. The ePSF tab offers it as a download.
Milan 2026-10-19: user product only, next to the `aavso` / `varastro` folders. Config `epsf_export_auto` (default
true) writes the files when RUN ePSF finishes (tab or night run); with it off - or for light curves of an earlier
run - the tab's "Generate ePSF light-curve files" button writes them for every setup with ePSF light curves
(`ui_epsf_dashboard.generate_epsf_files`; Brno 533: g 5 + r 5 files). It is not a
submission file: INV-PSF-SUBMIT-01 (AAVSO / VarAstro aperture-only) is unchanged - an ePSF AAVSO export would reverse
it and is Milan's decision (open).

## EPSF-STALE-2026-10 (2026-10-19) - a RUN ePSF replaces the previous ePSF products of the setup

The fit + merge drops every psf_* column of the setup's proc sidecars before writing the new fit set, and the ePSF
manifest is replaced. ePSF light curves of targets outside the new selection (and their user exports) therefore
described fits that no longer exist - on Brno 533 g a run with `EW` after `E*` left 10 E/EA curves whose header
pointed to the old model while the manifest said 4 frames / `EW`, and the PDF would have shown them with the new
run's facts. `epsf_stage.remove_stale_epsf_products` removes them after the fit + merge; the RUN ePSF help text says
so.

## CFG-REVIEW-2026-10 (Milan 2026-10-19) - config.json review: dead keys and the old plate solver removed

Method: every AppConfig field traced at runtime (a full Phase 0+1+2A run of 535 BO CVn and the test suite, with a
tracing `__getattribute__` in a throw-away copy) plus a static name-based reachability of every top-level
function / class from the entry points (app, night_run, psf_runner, the index builders and `scripts/`).

**The old VYVAR plate solver was dead code.** Since PLATESOLVE-2026-10 every MASTERSTAR is solved by `astrosolve`
(`masterstar_build`), but `vyvar_platesolver.solve_wcs_with_local_gaia` and its 59 helpers (4 728 lines), the
triangle blind solver `vyvar_blind_solver.py` + `vyvar_blind_series.py` (1 796 lines) and the PKL index builder
`GAIA_DR3/build_blind_index.py` were still in the tree - nothing reached them. Removed, with 29 config keys that
only they read: `blind_*` (15), `blind_index_fine_path` / `_wide_path` / `_select_mode`, `catalog_query_max_rows`,
`debug_platesolver`, `verify_mag_limit`, `masterstar_catalog_recovery_min`, `masterstar_centre_rms_max_px`,
`masterstar_distortion_benign_ratio_max`, `masterstar_min_matched_floor`, `masterstar_platesolve_sip_max_order` /
`_min_order`, `masterstar_quality_crowded_n_cat_min`, `masterstar_detection_cap_*` (4); 7 dead constants of
`pipeline_constants.py`; the DAO-stars UI sliders and the two .pkl path fields in Settings (replaced by
`astrometry_index_dir`, now in config.json). `vyvar_platesolver.py` keeps the pointing hints, proper motion, the
Gaia cone catalogue and the sibling-filter recovery (1 634 lines). New `GAIA_DR3/build_astro_index.py` (thin entry
for `astro_index.build_index`, release tool `build_astro_index`); INSTALL / README / install guide / FLOW / the
installers no longer tell users to build or copy the .pkl indexes (the installers do not copy the 7 GB
`vyvar_astro_index/`; hinted solving works without it).

**Other dead or duplicate keys.**
- `psf_zp_membership`, `psf_zp_for_zp_validated_rigs`: both modes admitted the same epochs since FIT-OK-ADMISSION-01
  (psf_fit_ok and a positive flux) - one rule `psf_internal_lc.psf_usable_mask`, rig-identity plumbing and the two
  header lines removed.
- `calibration_library_native_binning`: ignored on load since CAL-2026-10 (always None; the UI field edited a value
  that was never read back).
- `phase01_flux_col`: Phase 1 runs before RUN ePSF, so `dao_flux` was the only usable value.
- `comp_color_tiers`: 3 rows; the 4th only mirrored `comp_max_delta_bprp`. Old 4-row tables (and their `w`) load.
- `snr_cog_isolation_fwhm` -> `comp_isolation_fwhm` (it is the comparison blend-isolation radius, not a growth-curve
  parameter); the old name loads (`config.RENAMED_KEYS`).
- Legacy aliases no longer migrated (`GAIA_DB_PATH`, `VSX_LOCAL_DB_PATH`, `EXOPLANET_LOCAL_DB_PATH`,
  `aavso_observer_code`, `blind_index_path`, `comp_tier{1..4}_bprp_limit` / `_weight`): KNOWN_REMOVED.
All removed keys are KNOWN_REMOVED_KEYS (INFO on load). `dev/scripts/validate_config.py` reports them as WARN and
`--write-clean OUT` writes a cleaned copy of a config.json (values kept, removed keys dropped, renamed keys
migrated); the input is never changed. The repo config.json is regenerated by the canonical writer (128 keys).

**Dead code found on the way** (no caller outside tests): `aperture_pertarget.normalize_f_grid / abbe_p2p_scatter /
equal_weight_ensemble_delta / pick_f_star`, `utils.maybe_rescale_linear_wcs_cd_to_target_arcsec_per_pixel`,
`wcs_invertibility.ensure_sip_inverse_coefficients`, `lc_outlier.export_keep_mask`,
`psf_chi2_locus.apply_chi2_locus_to_rows` (its tests now run the production `reapply_night_locus_to_proc_dir`),
`pipeline_gate_helpers.py` (whole module), 11 unused dataclasses / exceptions,
`run_draft_ram_calibration_qc_to_obs_files(pipeline_config)`, `write_internal_psf_lightcurves(cfg)`.

**Open for Milan** (kept, they still change behaviour): psf_photometry_enabled, aperture_photometry_enabled,
cog_* (7), aperture_policy_mode alternatives, k2_fit_*, qc_max_hfr / qc_min_stars / qc_max_background_rms,
dao_qc_in_calibrate, gain_container_scale / color_level_k_* as global values of per-rig quantities. -> CFG-REVIEW2-2026-10.

## CFG-REVIEW2-2026-10 (Milan 2026-10-19) - the open keys removed, camera properties moved to EQUIPMENTS

Milan: "co je nutne odstranit, ak to sucasny kod nepotrebuje / 7 zapni / vlastnosti zostavy su v databaze, su tam
tabulky - hlavne camera". 23 config keys removed (config_runtime 133 -> 110; KNOWN_REMOVED, INFO on load):

- `psf_photometry_enabled`: reports and AAVSO / VarAstro exports are aperture photometry (INV-PSF-SUBMIT-01); the
  ePSF light curves are a separate product (ePSF page in the PDF, `epsf_export_auto`). `report_methods` keeps only
  the file layout; the multi-method report builder, `epsf_hooks.py` and `build_epsf_science_set` are removed. The
  PDF methods section cites the PSF literature when the setup has ePSF light curves (`psf_used`).
- `aperture_photometry_enabled`: aperture photometry is the only photometry of the per-frame catalogues.
- `cog_*` (7): the per-frame curve-of-growth correction (default off, never validated) and its night gate
  (`evaluate_cog_night_apcorr_gate`, `use_apcorr_flux`, cog meta / invariants) are removed; k* per target replaces it.
- `aperture_policy_mode`: per_target is the only policy; `f_fixed_night` / `f_per_frame` and
  `normalize_aperture_policy_mode` are removed. Per-frame catalogue radius = aperture_fwhm_factor x frame QC FWHM
  (night median when a frame has none) - what per_target already did.
- `k2_fit_*` (4) + `k2_ceiling`: the night fit of k'' (NIGHT_FIT v2, never enabled) is removed; k2_mode is
  `literature` or `off`.
- `qc_max_hfr`, `qc_min_stars`, `qc_max_background_rms`: they only set a pass/fail flag (VYQCPASS, manifest
  qc.passed, qc_rejected) that nothing read - frames were always kept. The QC measurements (VY_QCHFR / VY_QCNS /
  VY_QCBG / VY_QCRMS, manifest) stay.
- `dao_qc_in_calibrate` (item 7): the DAO QC during calibration (PERF-10) always runs.
- `color_level_k_mag_per_bprp` / `_stderr`: NOT moved to the DB - the Clear level colour correction had been
  unreachable since BANDAUTH-2026-10 (`_compute_group_color_term_fit` returns before it for a non-transformable
  filter; Clear is reported untransformed, TRANS=NO). Its only live effect was `resolve_apply_color_term` = True for
  Clear, so the trust gate believed the colour term was on. Now Clear auto = colour term off (consistent with what
  is applied); `apply_color_term(comp_weights)` and the `clear_level` mode are removed. The trust soft reason
  "T3+ comps with color term off" now applies to standard (transformable) filters only: for an untransformed filter
  no colour term exists and the comps' colour mismatch is already the "ensemble includes T3+ comps" reason (without
  this R CVn on 535 went YELLOW -> RED for the same cause counted twice; Milan's results stay as they were).
- `gain_container_scale` -> `EQUIPMENTS.ADC_BITS` (container scale 2^(16 - ADC_BITS); unset = 1). Seeded once from
  the sensor: IMX492 / IMX294 / IMX533 = 14, IMX571 / IMX455 / IMX411 / QHY600 = 16. A 16-bit camera no longer gets
  the 14-bit scale 4 in the gain fallback (used only when the photon-transfer gain is not accepted).
- `sigma_sys_mag` map -> `EQUIPMENTS.SIGMA_SYS_MAG` (seeded 0.018 for equipment 4, C5A-150M, as the map had it).
  The night-measured floor still takes precedence (SIGMA-SYS-NIGHT-2026-10).
New `rig_photometry.py` reads the two columns read-only from `database_path`; `VyvarDatabase` migration
`_ensure_equipments_rig_photometry_columns`; both columns are editable in Database -> EQUIPMENTS.

## CT-STUB-REMOVED-2026-10 (Milan 2026-10-24) - the CT preselection test machinery is removed

The M67 / chi & h Per colour-term validation runs (2026-09) left a test machinery in the production path:
`photometry_comp._variable_targets_looks_like_ct_presel_stub` + `ensure_full_variable_targets_if_presel_stub`
ran before every photometry (night run and re-run). Besides the explicit markers (notes "CT presel", names
"M67 in-range / red-giant") it took ANY `variable_targets.csv` with < 50 rows and MASTERSTAR > 200 stars for a
stub and regenerated it from the VSX cone: hand-added targets (MANUAL / USER) were lost and
`comparison_stars.csv` / `photometry_plan.json` were rewritten (injection 534: all 45 synthetic targets gone).
Removed: both functions and their call, the `VYVAR_CT_PROTOTYPE` diagnostics (`ct_prototype.csv`,
`_append_ct_prototype_row`, `_color_term_cat_inst_scatter_pair`, `_CT_PROTOTYPE_CSV_FIELDS`), the scripts
`dev/scripts/chiandh_*` (they imported a module that no longer exists), the tests. The colour-term correction
itself (`fit_color_term_c1`, `apply_color_term`) is production and stays.

## BORDER-FOOTPRINT-2026-10 (Milan 2026-10-24) - border filter = overlap of all aligned frames, always

Milan: "vzdy musi urobit tento aligment aby sme nerobili fotometriu hviezd co su na okraji"; AstroImageJ crops
the common area of all frames.
Before: the safe bbox was the intersection of the "finite pixels" of the aligned frames. The aligner fills the
uncovered reference pixels with a constant (the frame median), not NaN, so the bbox was the whole chip on every
night (MN Boo 538: frame 116 shifted -80 px, bbox x=[11,2070] y=[11,1385] = chip - r_out). Stars at the edge were
measured on the frames that cover them and dropped per epoch by the footprint (AP-FOOTPRINT) - light curves
with holes instead of no light curve. In addition the aligned frames were looked up as `proc_*.fits` (three
places) although they keep the raw names: 0 found, so outside the RAM handoff (disk mode, the comparison-pool
refill) the post-alignment filter was silently skipped. With a field jump the bbox used only the "dominant
group" of frames (`detect_field_jumps`).
Now: each aligned frame's data footprint comes from its frame->reference transform `VY_AM0..8`
(`frame_footprint.footprint_bbox`: the corners mapped, the second smallest / largest x and y, 1 px inside); the
filter is the overlap of ALL aligned frames (`common_field_intersection_bbox_px[_from_headers]`, headers only),
shrunk by the sky annulus r_out as before. The aligned frames are taken by their real names. Target and
comparison selection refuse to run without `safe_bbox_px` in `photometry_plan.json` (RuntimeError, no silent
fallback). Removed: the finite-pixel bbox, the dominant-group exception, `detect_field_jumps` (no other user),
`night_run.run_ui_night_photometry` (no caller since the GUI photometry page was removed; GUI RUN VYVAR = `python
-m night_run --job`, one code path).
A/B night 537 (old = s37 code): aligned FITS identical (81/81); overlap x=[8,2076] y=[7,1387] -> safe bbox
x=[20,2064] y=[19,1375] (was [12,2069] x [12,1384]); one target fewer (ASASSN-V J154735.86+510549.5 at x=2067:
its annulus leaves the data on some frames); 7 of 30 light curves identical, the others changed through the
comparison stars dropped at the edge (zero point, scatter of the difference 2-4 mmag) and, where that moved the
per-target aperture choice, by more (AP-CHOICE-STABILITY-01).

## ALIGN-POLY-2026-10 (Milan 2026-10-24) - polynomial registration when the similarity does not verify

MN Boo 538 (QHY294 + Zeiss 200 mm, OpenAstroTracker): the mount jumped at frame 102 (+60..90 px) and drifted
130 px in x; the field rotated 1.46 deg over the night (polar alignment). Against the reference frame 143 the
similarity transform (astroalign / star_similarity) left p90 0.59-0.62 px on frames 1-101 (edge of the field
~1 px) and ALIGN-VERIFY refused them (gate max(0.25 FWHM, 0.5 px)): 101 of 139 frames not measured, only the
rising branch of the MN Boo minimum. Measured on 250 star pairs per frame: affine the same as similarity,
2nd-order polynomial p90 0.12-0.16 px (cubic no better) - the lens distortion of a wide field after the image
moved 100+ px on the chip. The IRAF geomap / geotran "general" geometry (linear part + polynomial distortion)
is the standard registration for this case.
Now: when the similarity does not verify, `fit_polynomial_from_stars` fits a polynomial of order
`ALIGN_POLY_ORDER` = 2 frame -> reference on the mutual star pairs (seeded by the best similarity, refined three
times, at least 3 pairs per coefficient) and its inverse; the SAME gate verifies it (unchanged). The frame is
resampled by `warp_to_reference` exactly as astroalign does (cubic, constant fill = frame median). Method
`star_polynomial`; `VY_AM0..8` = projective matrix closest to the polynomial on a 9x9 grid (`VY_AMERR` = largest
grid difference, MN Boo <= 0.38 px, inside the 1 px footprint margin) for the steps that need a matrix
(footprint, saturation back-mapping, field corners); OSC channel frames reuse the polynomial (`poly_inv`).
Frames that verify with astroalign / similarity are unchanged (537: aligned FITS byte-identical).
A/B MN Boo 538: aligned 39 -> 140 frames (102 star_polynomial, median p90 0.15 px); overlap x=[7,1939]
y=[4,1280]; MN Boo light curve 38 -> 140 points, the whole minimum (9.09 -> 9.36 mag) without a step at the mount
jump; target light curves 36 -> 29 (7 targets outside the common area).

## PROCESSED-FIELD-NAME-2026-10 (Milan 2026-10-24) - a run is a "processed field" in <TARGET>_<YYYYMMDD>_<NNN>

A night run was stored in `Archive/Drafts/draft_NNNNNN` and called a "draft" in the GUI. Now:
- **Folder name** `<TARGET>_<YYYYMMDD>_<NNN>` (e.g. `MN_Boo_20260422_001`): TARGET = the FITS OBJECT of the light
  frames (OBJECT / OBJNAME), else the name of the selected source folder (observers name it after the star),
  else `field`; spaces -> `_`, only [A-Za-z0-9+-_]. YYYYMMDD = the evening the night began: first frame in local
  mean solar time (site longitude) minus 12 h, so frames after midnight keep the evening's date (Milan: numeric
  YYYYMMDD, sorts chronologically). NNN = first free counter for the same target and night (two runs of the
  same data -> `_001`, `_002`).
- **The numeric run id stays the internal key** (database, manifest, job specs): `draft_manifest.json` keeps
  `draft_id` and gains `name` / `target`. The folder name carries no id; every place that composed
  `draft_{id:06d}` resolves the folder by the manifest (`draft_provenance.draft_dir_for_id`), every place that
  parsed `draft_<digits>` out of a path reads the manifest (`processed_field.processed_field_root / _id`).
  Ids are 64-bit integers in SQLite (Milan: a large observatory with many fields every night) and are never
  reused: a new id is above every id under `Archive/Drafts` and above the largest id in the database
  (`OBS_QC_PROCESSING_RUN.DRAFT_ID`), so deleting old folders cannot bring an id back.
- **Legacy folders** `draft_NNNNNN` keep working (resolved by name or manifest); Milan reprocesses and deletes
  them himself.
- **GUI**: "draft" is no longer used - "Load processed field" (name, full path or number), all messages say
  "processed field". The on-disk folder `Archive/Drafts` keeps its name.
- Removed with it: the GUI "Active session" id (`utils.generate_session_id`, `vyvar_varstrem_session_id`) and
  the unused `_current_session_id` / `_ap_session_id` session keys - shown only, never written anywhere.
- Found with it: `record_observer_location_provenance` rewrote `draft_manifest.json` from a fixed key list and
  dropped every other key (`name`, `target`, also `imported_at` written at import). It now merges
  (`patch_draft_manifest`). A/B 537 / MN Boo 538 (s38 vs this): every light curve, per-frame CSV and aligned FITS
  identical; only the folder name, the run id (537 / 538 = above the database's largest id) and the manifest
  keys differ. Two runs of the same MN Boo data: `MN_Boo_20260422_001`, `MN_Boo_20260422_002`.

## ALIGN-MODEL-2026-10 (Milan 2026-10-24) - every frame chooses its registration model on its own stars

Milan: VYVAR should switch between the similarity and the polynomial by itself. "Smaller residual wins"
cannot decide: the polynomial (12 coefficients) always fits a little better than the similarity (4).
Now a VERIFIED similarity (astroalign / star_similarity) is compared with the 2nd-order polynomial on the
SAME star pairs by the Bayesian information criterion, BIC = n ln(RSS/n) + k ln n (n = 2 x pairs); the
polynomial replaces it when BIC_similarity - BIC_polynomial > `ALIGN_MODEL_DBIC` = 10 ("very strong
evidence", Kass & Raftery 1995). An unverified similarity falls back to the polynomial as before (ALIGN-POLY).
A frame identical to the reference (residual ~1e-13 px) is not compared. `alignment_report.csv` records
`align_model_dbic`. WCS-reprojected frames are not compared (the reprojection is not a similarity).
MN Boo 538 frames: near the reference (shift < 25 px) dBIC -57..+8 -> similarity; 135 (13 px) 99, 001-120
(60-160 px) 400-1600 -> polynomial.
A/B (s39 vs this): 537 - every frame keeps the similarity (dBIC <= 3), aligned FITS and light curves
byte-identical (111/111). MN Boo - 129 polynomial + 11 similarity (was 102 + 38), worst p90 0.61 -> 0.23 px,
MN Boo point-to-point scatter 5.67 -> 5.23 mmag; zero point -15 mmag (different comparison choice).

## AP-NIGHT-K-2026-10 (Milan 2026-10-24, option A) - one aperture k per night and group (superseded 2026-10-27 by AP-BRIGHT-K-2026-10)

Closes AP-CHOICE-STABILITY-01. The aperture factor k (r = k x FWHM of each frame) is now ONE value for all
targets of an observation group in a night: for every target the comp leave-one-out scatter curve over the grid
1.0-2.0 is normalised to its minimum; the night curve is the median over targets (a k enters when >= 75 % of the
targets may use it under the BLEND-K limit); k_night = the largest k within 3 % of the curve minimum
(`NIGHT_K_TIE_FRAC`, = the bootstrap standard error of the median curve on 537 / MN Boo 538, 0.02-0.03). Each
target gets the largest allowed k <= k_night, targets without comp scatter included, so target, comps and check
star keep one enclosed-light fraction. The target's own curve minimum stays as `f_own_curve_min` in
`aperture_per_target.json`, with `night_k` (k, curve, number of targets).

**Why.** The per-target minimum was decided by noise: one comp more or less changed k for 45-48 % of the targets
(537, MN Boo 538), which is why two runs of the same night gave 3.0 vs 3.6 px. The night median does not move
(0 % when one comp of one target is dropped, 0 % for one frame).
**Rejected: per-target CCD-equation S/N (Howell 1989).** It drives faint targets to k ~1.0 or below (optimum
~0.67 FWHM, Naylor 1998, Mighell 1999). Full-night A/B: target rms OOE -2..-7 % median but 20-24 % of the targets
worse, check stars +3..+7 %. With k = 0.75 added to the grid: point-to-point noise of faint targets -10 %, but the
comps' slow noise (10-frame bins = drifts) +18-19 % vs k = 1.0, lowest at 1.25-1.75 - small apertures on
undersampled images (FWHM 2.2-2.4 px) lose a seeing / centring / pixel-phase dependent fraction of the light
(Howell: "assure the source is not undersampled"; Mighell; Lauer 1999; Stetson 1990). The comp scatter over the
whole night contains these drifts, the photon model does not. COMP-SELECT-2026-10 had reached the same verdict.
Practice elsewhere: one factor x FWHM for every star (AAVSO VPhot 1.5, AstroImageJ 1.7, HiPERCAM 1.5-2.5,
AutoPhOT 1.6, Siril 2; AAVSO guide: diameter 3-4 FWHM); per-star selection reported as a small gain (VaST,
VAPHOT). k < 1 stays open for well-sampled setups (AP-SMALL-K-01).
**A/B (s40 vs this, full nights).** 537: k_night 1.25 (curve 1.0: 1.098, 1.25: 1.000, 1.5: 1.105, 1.75: 1.186),
30 light curves as before, rms OOE median 1.000 / mean 0.979 (better > 2 % for 20 %, worse for 7 %), check star
median 1.000. MN Boo 538: k_night 1.25 (1.0: 1.104, 1.25: 1.007, 1.5: 1.061), 29 light curves, rms OOE median
1.000 / mean 0.900 (better 34 %, worse 7 %), check star median 1.000 / mean 0.947.

## TRANSIENT-2026-10 (Milan 2026-10-25) - transient search of one night, minor bodies, opt-in

`transient_search_enabled` (default off). When on, the field variability search (same place, GUI and night
run) also:
1. **New sources** (`transient_field.py`; method of Gaia Science Alerts, Hodgkin et al. 2021, and NMW / VaST,
   Sokolovsky et al.): DAOStarFinder on every aligned frame; detections farther than max(1 FWHM, 2 px) from
   every MASTERSTAR source are linked across frames (friends-of-friends, 1 FWHM); a group on >= 3 frames that
   stays put is a source, one that runs with time is a moving object; MASTERSTAR `unmatched_in_range` detections
   are sources too. Every source is measured on every frame at its position (field k, annulus sky, frame zero
   point of the variability search; noise from 300 empty-sky apertures per frame - the aligned frames are
   interpolated). A candidate (T1, T2, ...): S/N >= 5 on >= 3 consecutive frames (ATLAS: >= 3 detections),
   centroid within 0.75 FWHM, not the same detector pixel elsewhere on the sky (hot / warm pixel, reflection:
   the pixel is followed through the inverse alignment; flickering warm pixels too), no Gaia star within
   max(2", 2.5 FWHM) less than 1 mag fainter, not in the halo of a G < 8 star >= 3 mag brighter than the source
   (radius max(10", 4 FWHM x 10^(0.2 (8 - G)))), not on a hot pixel of the master dark of the night (> 50 sigma;
   the only test for a field that does not drift on the detector), aperture and sky annulus inside the frame on
   >= 80 % of the frames (the onset is judged on those frames only), brighter than the catalogue depth - 1 mag, not on an OpenNGC galaxy core (`catalogs/NGC.csv`; off the core
   but inside the galaxy it is kept as a possible supernova). Kind: appeared / faded / present. Online catalogue
   check as for the variability candidates.
2. **Brighter than Gaia**: night magnitude vs Gaia G through the field relation (robust, colour terms) plus the
   Gaia neighbours' light in the aperture; <= -1 mag and <= -3 sigma -> listed (a nova on a faint catalogued star,
   a dwarf nova all night).
3. **Minor planets and comets** (`minor_bodies.py`): MPCORB.DAT.gz and CometEls.txt in `catalogs/` (the user
   downloads them from the MPC), two-body orbits (checked against the astropy ephemeris: 0.2 / 1.2 / 12" for Mars
   2 / 10 / 60 days from the epoch), light time, topocentric, V <= 20 (Milan). Listed with predicted V, motion and
   on how many frames they were detected; moving objects are matched to them.
Outputs: `transient_candidates.csv`, `moving_objects.csv`, `minor_bodies_in_field.csv`, `transient_meta.json`,
`lightcurves/transient_T<n>.csv`; Variability tab section; Summary Measure Report pages after the variability
pages; citations section TRANSIENT SEARCH. Off: no file, no page, outputs byte-identical (MN Boo checked).
**Tests.** MN Boo 538 (140 frames, 9.8"/px, mount drift): 84 sources examined, 6 candidates - compact
G 13.9-14.5 sources all night without a Gaia star (5 after the halo / dark / edge rules; 5 more matched NGC/IC galaxies within 2" and were removed;
the rest are likely fainter galaxies - the online SIMBAD check names them). Hot / warm pixels: 15 of 18 "moving
objects" and 7 stationary candidates were detector pixels. Injection on the same night (20 uncatalogued
sources G 10-14: present, switching on, nova-like rise of 3 mag in 1.5 h, flare with 20 min decay; 2 moving
objects 34 and 68 "/h): 20 / 20 candidates, 2 / 2 moving objects; a switching-on or flaring source is reported
5.5-6.7 min after the onset (3 frames at the 2-min cadence), the rise when it crosses the detection limit;
no false candidate beyond the control night. Dense field V842 Her 537 (Hercules, 40 frames, no mount drift,
so the detector-pixel test cannot run; draft 528 holds the same raw lights): 11 control candidates before the
halo / dark / edge rules (halo of a G 4.6 star, a master-dark hot pixel, a frame-edge artefact, two warm pixels
not hot in the dark, compact faint sources), 4 after. Injection 13 / 20, 2 / 2 moving objects: misses are G 14
(too faint for 3 consecutive S/N 5 frames), the G 12-13 rises that do not reach the limit within the night,
and one G 13 source that the plan put on a master-dark hot pixel. Without drift a warm pixel that is not hot in
the dark cannot be told from a star - such candidates need the cutout check. Not planned (Milan 2026-10-25): a supernova on a galaxy core (image
subtraction) and comparison with earlier nights of the field - VYVAR is for variable-star photometry.

## RESOURCES-2026-10 (Milan 2026-10-26) - VYVAR sizes its parallel work itself: half the RAM, half the CPUs

A run uses at most half of the computer's RAM (on Linux also the container limit) and half of its logical CPUs
(Milan: "50% RAM a polovica CPU - toto je fer"). Every per-frame stage (calibration, QC in-place, alignment,
per-frame catalogs, aperture grid, target field maps, ePSF, transient search) takes
min(CPU budget, items, (RAM budget - main process) / worker cost) workers, worker cost = 30 x one frame as
float32 + 300 MB (measured on Brno 534, 9.5 Mpx frames, one worker: QC 29x, catalogs 27x, alignment 24x,
ePSF 19x, calibration 11x of the frame; the largest is used). Nothing to set: `per_frame_mp_reserve_ram_gb`
and the computed `qc_preprocess_workers` are removed, so are the legacy environment variables and the thread
backend; `VYVAR_PARALLEL_WORKERS` stays only as a test / benchmark override of the CPU budget.
One pool (`parallel_frames.map_frames`) for every stage: results in frame order; at most two frames per worker
in flight; every worker exits by itself when the main process is gone (orphans of a killed run held 2.2 GB);
a pool that breaks (worker killed, MemoryError) continues with half the workers down to one.
**Why.** The old sizing assumed 48 MB per worker whatever the frame size (real: 0.7-1.1 GB for 9.5 Mpx), and
the alignment collected every aligned frame of a setup before writing it (Brno 534: 97 x 38 MB in the main
process) - two runs of 534 were killed out of memory on a 7 GB machine. The aligned frames now stay in memory
(RAM handoff) only when they take at most half of the RAM budget, and the alignment writes each frame as it
arrives. The budget is logged in the infolog ([RESOURCES]) and stored in pipeline_meta (`resources`).

## EPSF-HASH-2026-10 (Milan 2026-10-26) - the ePSF model hash identifies the models, not the build time

`epsf_model_sha256` (ePSF light-curve headers, pipeline_meta) was the sha256 of the manifest file, which carries
`created_utc`: two runs with byte-identical models got different hashes (phase 0 baseline). It is now
`psf_photometry.epsf_models_sha256`: sha256 of the manifest content without `created_utc` (canonical JSON) -
every per-frame model sha256, the night spatial perturbation sha256 and the build parameters. Golden runs A and
B of BO CVn 527 and Brno 534 (g, r) give the same hash.

## EXPORT-LOG-2026-10 (Milan 2026-10-26) - ERROR only when an export failed

Three kinds of "no file", three levels. A rule decides the data are not submitted (too few usable epochs, no
exportable points, magnitudes not on the band's scale, oneRGGB) -> INFO "not exported by rule", counted in
`export_stats["not_exported_by_rule"]`, not an export failure. The data or the setup need the observer's
attention (time base not BJD_TDB, filter without an AAVSO code, no VarAstro field map, no AAVSO row) ->
WARNING, collected. The export itself failed (read / write error, exception, LC file without the needed
columns) -> ERROR, collected. The batch summary takes the highest level of what it collected. Phase 0: three
BO CVn targets with 3-25 of 146 usable epochs were logged as ERROR although the rule was right.

## FITS-KEYS-2026-10 (Milan 2026-10-26: "premenovat, je dolezite dodrziavat standarts")

Every FITS keyword VYVAR writes has at most 8 characters (FITS 4.0); 13 longer keywords (HIERARCH cards with a
warning on every frame) are renamed, `VY_FWHM_GAUSS` (a copy of `VY_FWHM` since FWHM-ONE) is no longer written
and every reader takes `VY_FWHM`. No aliases: old names are not read, old drafts are processed again.
Table: docs/VYVAR_FITS_KEYWORDS.md. Contract: dev/tests/test_fits_keywords.py.

## CORE-NO-UI-2026-10 (Milan 2026-10-26) - the pipeline never touches GUI state

No pipeline module imports Streamlit (the night run logged "missing ScriptRunContext"). Core modules log and
report through `progress_cb`; GUI state (selected camera and telescope, the processed field of the last job)
lives in `ui_session.py` and the ui_* / app modules. Found on the way: `find_photometry_setups` resolved the
draft folder through the GUI session first ("last job output"), so a night run started from the GUI could take
the setups of another processed field than its own; it now uses the given folder or the folder of `draft_id`,
and the night run passes its own. Removed: Streamlit popups of calibration-diagnostic and alignment messages
(the lines stay in the infolog), the session copy of the per-frame store (never read) and the session write of
the field centre (the GUI sets it from the night-run result). Contract: dev/tests/test_core_no_ui.py.

## TEST-ISOLATION-2026-10 (Milan 2026-10-26) - tests never see the user's data

In a git checkout the data root is the code tree, so a test calling `AppConfig()` read the user's config.json and
opened the user's vyvar.sqlite3 (and on Linux created files named after the Windows paths). Every test now runs
with `VYVAR_DATA_DIR` on a fresh data root of the session (code defaults, an empty database); tests that write
their own config.json use `tmp_path_is_data_root`. Opening a database in the code tree outside dev/ fails the
test (dev/tests/conftest.py, contract dev/tests/test_test_isolation.py).

## LOG-NOISE-2026-10 (Milan 2026-10-26) - one line per repeated warning
A target without a comparison star with a valid catalogue magnitude in the band is logged per target at INFO and
once per setup as a WARNING with the count (Brno 534: 49 WARNING lines before). Library warnings that repeat for
every frame or star (astropy FITSFixedWarning for the camera's RADECSYS / MJD-OBS, photutils "star excluded from
ePSF fitting") are shown once per process with the first example (`infolog.quiet_repeated_library_warnings`,
numbers in the text ignored); at the end of the run one INFO line per such warning gives the number of repeats.
Header card comments are at most 47 characters, so value and comment fit one 80-character card (astropy cut
18 comments and warned 296 times per BO CVn night); contract in dev/tests/test_log_noise_disk_2026_10.py.

## DISK-PREFLIGHT-2026-10 (Milan 2026-10-26) - a run that does not fit on the disk stops before the import
After the scan the night run estimates what it will write: the imported copy of every light plus a calibrated and
an aligned float32 frame per light, times 1.10 (`parallel_frames.run_disk_need`), and stops with an OSError naming
the shortfall when the archive's disk has less free space. Found when a run failed in the middle on a full disk.

## MP-CALIB-ON-2026-10 (Milan 2026-10-26) - parallel calibration stays on; supersedes D-MP-CALIB-OFF-01
D-MP-CALIB-OFF-01 kept the calibration serial until a real draft showed parity. Parity is shown: Brno 534 with
2 workers gives the same light curves, tables and exports as with 1 worker, and BO CVn with 2 workers has
pixel-identical calibrated and aligned frames to the phase 0 golden run (2026-10-26). Worker count follows
RESOURCES-2026-10.

## COMP-RMS-NORM-2026-10 (Milan 2026-10-26, D2) - comparison RMS against a normalised reference
The leave-one-out RMS that ranks comparison candidates (COMP-RMS-DEF-01-B) compared each star with the median of
the raw magnitudes of the other pool stars. The pool stays as the rules define it (quality, D3, saturation,
blend); each pool star is now first put on its own night level (median magnitude over the night) and compared
with the median of the other stars' offsets in the same frame. Stars of different brightness that are missing
on some frames (edge, dropout) no longer move the reference: constant stars with 40 % random dropouts gave
0.74-1.11 mag of spurious scatter before and 0 after (dev/tests/test_comp_rms_norm_2026_10.py). Honeycutt (1992,
star levels and frame zero points solved together); Burdanov et al. (2014, Astrokit, ensemble offset against its
own mean).

## SIGMA-SYS-MODEL-2026-10 (Milan 2026-10-26, D3) - the night floor is what the noise model does not explain
SIGMA-SYS-NIGHT-2026-10 took the comps' point-to-point scatter minus their photon noise as the floor. That
remainder also held each comp's scintillation and the noise of its reference (the mean of the other comps), and
the light-curve error then added scintillation and the ensemble error again. The predicted variance now holds
every known term once - photon + sky + read noise of the comp, its scintillation at the frame airmass (same rig
parameters as the LC error) and the reference noise (sum over the other n comps of photon^2 + scintillation^2) /
n^2 - and the floor is the median excess above it, never negative. Simulation (12 comps, 300 frames, 4 mmag
scintillation): no true floor gave 4.2 mmag before, < 1.5 mmag now; 3 and 6 mmag floors are recovered within
1.5 mmag (dev/tests/test_sigma_sys_model_2026_10.py). Burdanov et al. (2014); Everett & Howell (2001).

## K2-ONE-2026-10 (Milan 2026-10-26, D4) - one k'' convention, ensemble colour
Second-order extinction is a differential effect, dm0 = dm - k'' X (C_target - C_comparison) (Henden, AAVSO
forum "Secondary extinction and blue-red pairs"). It was applied to mag_calib only on calibrated light curves,
while delta_mag stayed uncorrected and the PSF light curve subtracted k2_delta_mag - the PSF-aperture strip
of a calibrated g night showed a ~17 mmag airmass trend. Now the same per-epoch term goes into delta_mag always and
into mag_calib when the curve is calibrated (mag_calib - delta_mag stays the zero point). The comparison colour is
the Broeg-weighted mean BP-RP of the ensemble comps the light curve uses (`k2_extinction.ensemble_bp_rp`), the
same weights as the zero point; the plain median of usable comps only when no weights exist.

## COMP-POOL-ONLY-2026-10 (Milan 2026-10-26, D5) - comparison stars only from the rule pool
When the global comparison pool was empty or its build failed, every target took MASTERSTAR stars with a
reduced gate set (no Gaia/VSX variable check of the pool, no blend, saturation-any-frame or D3 gate) - the
"masterstars fallback" (kept in DECISIONS 5013-5015, review G07-01). Milan: VYVAR selects the comparison stars
correctly by its rules; the only real gap is a very red target whose good comps differ too much in colour.
Now: comps come only from the pool that passed the rules; an empty pool gives no comps (the target is not
measured), a failed pool build stops the run. A red target whose comps all lie beyond the colour cap
(`comp_max_delta_bprp`, the T4 fallback inside the pool) keeps its light curve, and trust is RED with "colour
mismatch: comparison stars up to |dBP-RP| x (cap y)" (`classify_warnings`). The per-frame pass that served only
the fallback stays until the dead-code step (T10) removes it.

## SKY-MASK-2026-10 (Milan 2026-10-26, D6) - sky surface from a source mask, not an ADU cut
The preprocess sky surface (T3-PREPROCESS-SKY-SURFACE) was fitted only to pixels within +-100 ADU of the frame's
background median outside the DAO star stamps. On a strong gradient the cut removed the steep part from the fit
(19-37 % of a real gradient stayed in the frame) and the 100 ADU depended on the rig. Now, as the photutils
background guide recommends (mask the sources, `make_source_mask`): pass 1 fits the polynomial outside the DAO
stamps; the sources of its residual (segmentation at 2 x MAD sigma, >= 5 connected pixels, dilated by one FWHM) are
added to the mask and the polynomial is fitted again. No ADU constant, no clipping of fit samples. Synthetic
600 ADU gradient with 400 stars and noise: 39.7 ADU (p1-p99) left before, 0.44 ADU now
(dev/tests/test_preprocess_sky_surface.py). QC key `sky_surface_calm_adu` -> `sky_surface_source_mask_px`.

## EPSF-NOCLIP-2026-10 (Milan 2026-10-26, D1) - the ePSF builder without sigma clipping
photutils EPSFBuilder combined the residuals of every ePSF grid point with a sigma-clipped median (default
SigmaClip 3 sigma, 10 iterations) - the last clipping left in VYVAR (INV-NOCLIP-01). It now runs with
`sigma_clip=None`, a plain median (photutils 3.1.1 source). Comparison on BO CVn (12 frames, FWHM 2.0 px; ePSF built
on 300 build stars, tested by PSF fits on the other 300): fit residual identical (median 1.574 % vs 1.573 % of the
model power), PSF flux / cutout sum 0.961 vs 0.947 (1.4 % closer in every frame), ePSF peak 1.5 % lower, wings
+0.02 % of the peak. Anderson & King (2000) used a mean with 2.5 sigma rejection; Godden & Blundell (2025) show that a
clipped median biases the peak. Plot: dev/results/plots/D1_epsf_noclip_20261026.png.

## CHI2-FLUX-REF-2026-10 (Milan 2026-10-27, A) - psf_fit_ok against stars of similar flux
The chi2 test of the frame PSF fit (EPSF-CHI2-LOCUS-01) compared each star with one straight line
log10 chi2 = a + k log10 flux, fitted with 3 rounds of 3-sigma MAD clipping. chi2 = noise term + PSF-model mismatch
growing ~ flux^2 (DAOPHOT profile-error term, Stetson 1987) is curved; the line, set by the many faint stars, put the
bright stars 4-8 sigma above it. Brno 534 g: OK 99 % below log flux 4, 65 % at 4.5-5.0, 35 % at 5.0-5.5, while the
rejected bright epochs had the same PSF/aperture stability (2.07 vs 2.10 mmag) - and an epoch of a PSF light curve
needs every comp (INV-PSF-LC-PIN-01), so PSF light curves lost up to 80 % of their epochs. Now: night reference = plain
median of log10 chi2 in 0.25 dex flux bins (merged to >= 30 stars), scatter = MAD sigma of the residuals from that
curve per bin, frame offset = plain median, 5 sigma; linear between bins, linear extension outside. No clipping.
Brno 534: PSF epochs g 4079 -> 5494, r 4451 -> 5642; light curves with < 50 % epochs 19 -> 1 (g); precision on common
epochs unchanged (0.999 / 1.000). BO CVn: one light curve regained all 146 epochs, the rest identical. Faint end
stricter: blended stars rejected (r: rejected fits deviate 20 mmag, accepted 2.6 mmag).

## SKY-SRC-MASK-2026-10 (Milan 2026-10-27, B) - local sky without detected sources
The annulus sky was the plain median of all annulus pixels. In a crowded field it holds the wings of neighbours and
of the star itself; with seeing the annulus grows (2.7-5.2 FWHM) and the sky comes out too high by a fixed ADU per
pixel, times the aperture area: faint stars get a magnitude-dependent airmass trend (Brno 534 constant stars r,
aperture 1.0 FWHM: +17 / +39 / +96 mmag per airmass at G 15-16 / 16-17 / 17-18). Now (`sky_estimation.sky_image`):
plain median of the annulus pixels outside the sources of photutils `detect_sources` at 5 x MAD sigma (>= 5 pixels,
grown by one FWHM) - the photutils guide's masking instead of clipping. 5 sigma is the detection limit: resolved stars
leave the sky, the unresolved population stays in sky and aperture alike. Deeper masks (2-3 sigma, catalogue mask)
zero the sky of empty apertures but leave the light of resolved neighbours inside a crowded star's growing aperture
uncompensated (-24 to -114 mmag/airmass); for isolated stars all masks agree. Used by the frame photometry (proc),
the per-target aperture grid, MASTERSTAR S/N and the PSF local sky (photutils MedianBackground removed: its default
is a 3-sigma clip with 10 iterations). Brno 534: trends +4 / +2 / +29 mmag/airmass, check stars g std 17.4 -> 8.1 mmag,
r 12.9 -> 9.8, sigma_sys 2.8 -> 2.0-2.2 mmag. BO CVn: neutral; R CVn (G 7.1) had its own halo in the annulus
(+477 ADU/px): flux +32 %, p2p 85 -> 48 mmag. One function for every annulus (`annulus_sky_medians`): fewer than
30 sky pixels (annulus covered by a saturated star's own halo, CV CVn G 5.9) -> the sky pixels of a box of
half-size 2 x r_out. The mask costs ~0.8 s per frame (statistics on every 4th pixel, exact disk dilation by rows);
BO CVn run 41.4 -> 47.9 min on the same machine. The misnamed `sky_clipped_mean_med_std` (plain statistics) is removed.

## AP-BRIGHT-K-2026-10 (Milan 2026-10-27, option c) - aperture k by the target's brightness
Replaces AP-NIGHT-K-2026-10. With the unbiased sky (B) the night comp-scatter curve was flat and its tie rule chose
k = 1.5 on Brno 534 r from the scatter of the bright comps: faint targets 20-60 % noisier. Now each target takes the k
of its brightness group: constant field stars (no VSX / Gaia variable, never saturated) get the LOO scatter at every k
(`comp_rms_loo`, MAD sigma), are grouped by instrumental magnitude (0.5 mag bins merged to >= 30 stars), and a group
takes the largest k whose median normalised scatter is within one bootstrap standard error of its minimum (larger k:
less sensitive to seeing, centring, pixel phase). The k does not come from the target's own comps, so it is stable
when a comp changes (the reason for AP-NIGHT-K). The target, its comps and check star use the target's k; the blend
limit (BLEND-K) still caps it. Brno 534: bright stars 1.25, fainter 1.0; BO CVn: brightest 1.25-1.5, rest 1.0.
`aperture_per_target.json`: `criterion` brightness_k_field_scatter, `brightness_k` (groups, curves, k per target).

## GAIA-ID-STRICT-2026-10 (code review 2 T07, phase 2) - Gaia ids are text; identity is catalog_id
- `repair_catalog_ids.py` deleted (G04-02/03/04): it gave VSX targets that the cross-match left without a Gaia id
  the nearest Gaia star (no cos dec, no magnitude check) and rewrote variable_targets.csv on every selection; on
  suspected_variables.csv it reported repairs and wrote nothing. Every writer writes ids as text, so the module had no
  legitimate input. The Gaia table name comes from `catalog_db.gaia_table`.
- `normalize_gaia_source_id` is strict (G04-16): int >= 0, ASCII digit strings, exact integer floats < 2**53; a float
  of a 19-digit id, "4.6e+18", bool, bytes, non-ASCII digits, dict / list give "" (one WARNING per kind), names (VSX
  names, DET_ labels) give "" silently. It used to fabricate ids (4.611686018427388e+18 -> ...388000).
- One id dtype map `VYVAR_CSV_DTYPE` (catalog_id, name, target_ / comp_ / check_catalog_id, source_id,
  source_id_gaia, vy_seed_catalog_id); `read_proc_csv` and `read_vyvar_csv` always apply it; the aperture-grid worker,
  comp_blend, masterstar_build and the summary / masterstars / cone readers use them (G05-12, G06-04, G07-13, G04-48).
  An unreadable proc file in the grid is now an error, not a silently missing frame.
- Identity = `catalog_id` only (G04-17, G07-05): MASTERSTAR join key, proc row index and comparison-pool keys no longer
  prefer `name`. Duplicate id helpers in psf_internal_lc / epsf_science_set / dilution removed (G09-23, G07-34).
- Not in this step: one CSV writer (G03-39, T11) and Gaia epoch metadata (G03-25).
- Follow-up (A/B on BO CVn): `name` is a label (VSX name), never run through the id normaliser. Six loops still
  normalised ("catalog_id", "name") together and blanked every VSX name in active_targets.csv ("FY CVn" -> "");
  they and the `catalog_id`-else-`name` fallbacks now read catalog_id only (a missing catalog_id column raises).
  Source-scan test in test_gaia_id_strict_2026_10.py.

## FAIL-LOUD-2026-10 (code review 2 T05, phase 2) - required steps raise; optional diagnostics warn
- Rule: a step whose output the night needs has no try around it; its error stops the stage (or marks the frame
  "error:" through the existing per-frame handler). An optional diagnostic keeps a try, logs WARNING (never DEBUG)
  and says what is missing. Before, about 40 places logged and continued, so a run ended "ok" with a missing result.
- Required now (try removed or error re-raised): VSX export / variable_targets.csv (G03-06), proximity veto and
  safe-bbox border filter (G03-07), plate-solve input bundle metadata (G03-18), missing preprocessed lights folder
  (G03-37), DAO census / plate scale in MASTERSTAR (G04-15), DAO block and identity gate in catalog match (G04-13/14),
  DAO master block, MASTERSTAR join, forced inject / meta / saturation and time columns in pipeline_catalog (G05-08/09),
  aperture grid + AP-BRIGHT-K (G06-01), per-target grid flux application (G06-02), masterstars load in the aperture
  state (G06-05), comparison blend gate (status != ok raises, G07-14), per-target comparison selection (G07-28), PSF
  plate scale and cone load (G09-10), OSC QC verdict replication, Bayer lookup in the calibration library (G14-16).
- pipeline_meta.json (G11-07): an unreadable file is kept as `.json.corrupt` and raised (it was overwritten, losing
  provenance and sigma_sys); numpy values are serialised; tmp + os.replace write.
- catalog_crossmatch (G10-09/10): no process-wide warnings filter at import; the executor is shut down with
  cancel_futures on a timeout.
- HRD SIMBAD (G10-16): a failed query for one star is not cached as "no SIMBAD entry"; it is asked again next run.
- OSC effective gain / read noise (G01-40): unknown (NaN) in gives NaN out instead of stopping the calibration; a known
  invalid value still raises.
- Variability cutouts (G14-30): missing frames.csv / index handled, frame stem without the "proc_" prefix.
- Test fixtures that used a non-celestial MASTERSTAR WCS or no objects DB were relying on the silent VSX skip; they
  now use a celestial WCS and a real objects DB. Tests: dev/tests/test_fail_loud_2026_10.py (6 of 7 fail on the old
  code).

## LAYER-2026-10 (code review 2 T13, phase 2 step 2) - no import cycles; pipeline modules skip the facades
- Before: 12 modules could not be imported on their own (aperture_photometry, photometry_shared, photometry_comp,
  target_comp_selection, ...); they worked only after `photometry_core` was imported, because they took LOGGER and
  constants from the facade that re-exports them, and `aperture_photometry` bound 25 names onto the facade with
  `setattr`. Counting call-time imports, one cycle joined 79 modules.
- Now: every src_py module imports on its own; no cycle among module-level imports; call-time cycles only between a
  driver and its stages / dispatchers (aperture driver + state/target, comp selection pair, night_run <-> run_job,
  astro_index <-> astrosolve, config <-> params_registry, database <-> observer_location), frozen in
  dev/tests/test_layering_2026_10.py (may only shrink; the aperture and comp pairs go with the T12 splits).
- D-FACADE-PERMANENT-01 kept: `pipeline.py`, `photometry_core.py` and `photometry.py` stay as re-export facades for
  dev/ scripts. Pipeline modules import from the defining module (contract test); a facade may still be imported for
  a name it defines itself (AstroPipeline, run_full_photometry_pipeline).
- New leaf `photometry_defs.py`: photometry logger (name "photometry_core", routing unchanged), proc / LC column
  names, labels, FINAL_SKIP_REASONS, aperture-grid names (DEFAULT_APERTURE_F_GRID, flux_k_column). MAD_CONSISTENCY and
  BKG_SCALE_R_CLAMP_* to `pipeline_constants` (D-CONSTANTS-LEAF-01). `_GAIA_ID_DTYPE` copies replaced by
  `VYVAR_CSV_DTYPE`.
- Helpers moved to the layer they belong to (facades still export them): CSV writer `_vyvar_df_to_csv` and
  `_safe_proc_name` -> proc_frame_store; saturation peak helpers -> sat_diag; airmass -> time_utils; cone_sky_box ->
  catalog_db; MASTERSTAR input-path helpers -> pipeline_preprocess; DAO pass-2 -> pipeline_catalog; astrometry
  orchestration -> new `astrometry_orchestration.py`; export row selection and `resolve_lc_time_base` ->
  photometry_lightcurve (`photometry_ui_helpers.py` removed); Howell error model -> photometry_gate_helpers;
  colour-term mode -> band_classify; comp-quality JSON helpers -> comp_table; footprint helpers -> frame_footprint;
  PSF sidecar guard -> new `psf_sidecar_guard.py`; obs-group folder name -> fits_meta; header capture time ->
  calib_library; norm_fits_path_key -> utils; field density -> target_comp_selection; `draft_provenance` imports
  SmartImportPlan under TYPE_CHECKING only.
- Not in this step: G04-38 pass-2 finder cache (kept, SPEED-2026-10; pass-2 is serial), G06-31 / G09-35 (T12 splits),
  G10-36 / G10-38 network helpers (T19), G11-39 porcelain parsed twice (T11).

## CONFIG-ONE-2026-10 (code review 2 T06, phase 2 step 2) - one run config, no second defaults
- One `AppConfig` per run, built by the entry point (night_run, run_job, app / ui_*) and passed down as a required
  `cfg` / `app_config`; no library module builds one (static test). Before: 53 call sites re-read config.json, so
  `--config`, the draft config snapshot of a re-run (INV-CFG-SOURCE-01) and the DAO sigma override did not reach
  catalog_match, the per-frame DAO, the plan writer, the report and others. Per-frame workers get the run config in
  their state (pickled once), not copies of single keys.
- `--config X` loads exactly X (`AppConfig.from_file`; it loaded config.json next to X, G12-03).
- DAO sigma: resolved once in night_run into the run config; the `dao_threshold_sigma` arguments that the stages
  ignored (catalog_match, per-frame DAO) or passed through (astrometry_align, frame_export, orchestration) are gone.
- Gaia / objects DB: an empty or missing key is the default file the builder writes (`pipeline_constants`
  GAIA_DB_DEFAULT / OBJECTS_DB_DEFAULT, Milan 2026-10-27); no file there = no catalogue (run gates check the file).
  Builder magnitude defaults in one place (Gaia 17.5, VSX 18.0; the Gaia CLI said 16.5).
- No fallback that differs from the config default and no rig default: annulus 4.0 / 6.0 and aperture 1.75 / 1.9
  keyword defaults, n_comp_max 7, min_frames_frac 0.3, delta BP-RP 0.5, FWHM 3.7 px, frame 2082 x 1397 px, plate
  scale 1.3 "/px, gain 1 / read noise 10, trust bounds 3 / 12 and 3 / 8, observer site 0 / 0 removed (required
  arguments or the config value). The frame size comes from MASTERSTAR NAXIS (or a light); unknown is an error.
  run_aperture_photometry / prepare state take aperture and annulus factors only from cfg (force_aperture_r_px.py
  measured with annulus 4 / 6 before).
- Calibration workers never open the database of a freshly read config: the flat offset is resolved in the run
  (G01-34); a call without the database is an error.
- `VyvarDatabase.for_config(cfg)` sets the run's Archive root; the path-only fallback parses config.json as JSONC
  (new leaf `jsonc.py`; json.loads always failed on the comment lines, G13-11).
- Loader: rules between keys in one list `config_rule_problems` (loader warns, Settings refuses, validator reports,
  G13-08); a typo in k2_mode / apply_color_term and a truncated whole number warn (G13-35/36); KNOWN_REMOVED_KEYS
  moved to `config_removed_keys.py` (G13-40); alignment_max_stars range 10-200 = the real cap (G02-47).
- Settings Save starts from the file as written (relative catalogue paths stay relative, G14-35); registry marks the
  observer site keys resolved / hidden (G13-26); values_differ(NaN, NaN) is False (G13-25); the report HRD page uses
  the run snapshot (G11-27).
- Tests: dev/tests/test_config_one_2026_10.py (E3 / E4 contracts + behaviour). Tests that relied on the removed
  defaults pass those values explicitly.
- Follow-up (Milan 2026-10-27, approved removal): GUI / CLI parameters catalog_match_max_sep_arcsec, max_catalog_rows,
  min_detected_stars, max_detected_stars removed end to end (app, NightRunParams, night_run, orchestration,
  astrometry_align, masterstar_build, frame_export, per-frame DAO, catalog_match). They had three defaults each
  (2.0 / 25.0 / 12000, 100 / 500) and were floored. The values runs used are now `pipeline_constants`:
  CATALOG_MATCH_SEP_ARCSEC 10" (MASTERSTAR initial match, meta only - the identity radius replaces it; the per-frame
  radius comes from the plate scale as before), MASTERSTAR_MAX_DAO_ROWS 100000, FRAME_MAX_DAO_ROWS 12000,
  ALIGN_MIN_DETECTED_STARS 100, ALIGN_MAX_DETECTED_STARS 500. Whether 12000 suits other rigs is T09.

## AP-SMALL-K-2026-10 (Milan 2026-10-28, phase-1 item O5) - the aperture grid goes down to 0.6 FWHM
DEFAULT_APERTURE_F_GRID = (0.6, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0) (was 1.0 ... 2.0). With the old grid the
field-scatter minimum of most magnitude groups lay at the grid edge k = 1.0 (Brno 534 g 14/16, r 11/13, BO CVn
12/14). Isolated constant field stars, LOO scatter k 0.6 vs 1.0: Brno g G 16-17 40.8 vs 46.8 mmag, G 17-18 71.2
vs 83.8; r 31.3 vs 35.5, 52.4 vs 62.7; BO CVn (FWHM 2.2 px) G 14-15 204 vs 231, G 15-16 329 (0.5) vs 376.
Literature: S/N optimum for sky-limited profiles ~0.6-0.7 FWHM radius (Howell 1989; Naylor 1998).
AP-BRIGHT-K is unchanged (per magnitude group the largest k within one bootstrap SE of the minimum), so bright
groups keep 1.0-1.5. A target without a brightness group keeps k <= BRIGHT_K_NO_GROUP = 1.0 (small apertures
only from measured field scatter). Columns flux_kNNN / blend_kNNN are generated from the grid (3 digits:
flux_k060, flux_k075, flux_k088).

## BLEND-SEEING-2026-10 (Milan 2026-10-28, phase-1 item O4b) - neighbour light judged at bad seeing
The neighbour-light model (comp_blend, Moffat with the MASTERSTAR beta) is evaluated at the night's bad-seeing
FWHM, BLEND_FWHM_QUANTILE = 0.9 of the group's frame FWHM, instead of the median; thresholds unchanged (comps and
check <= 1 %, target trust soft > 1 %, hard > 5 %, variability gate <= 1 %). Reason: a neighbour outside the
aperture at median seeing enters it in bad seeing (Brno 534 FWHM 3.9-8.7 px). Field stars: measured airmass
slopes = 1.01 x (g) / 0.96 x (r) the slopes predicted from the per-frame FWHM (r = 0.74 / 0.79). Brno g: 10
targets with blend <= 1 % at the median FWHM had a predicted night range of 6-102 mmag (6 GREEN), e.g. Gaia DR3
2172312549003414144 (G 13.6, neighbour G 17.1 at 8.6 px; its LC correlates 0.78 with the prediction). The beta is
measured on the MASTERSTAR at the median FWHM (profile_fwhm_px). pipeline_meta target_blend records the FWHM.

## SIGMA-SYS-TOTAL-2026-10 (Milan 2026-10-28, phase-1 item O1 option C) - night floor from the total scatter
night_sigma_sys_from_comps takes the comps' total LOO scatter 1.4826 MAD(r - median r) instead of the
point-to-point scatter. The floor then holds the night's slow systematics and the LC errors describe the
repeatability over the night (AAVSO practice: check-star scatter as the uncertainty). Evidence: Brno 534 constant
check stars p2p / err 0.85-1.0 in every magnitude bin (white noise right) but total / err 1.3-1.9 (slow, mostly
airmass-correlated); 14 (g) / 12 (r) targets were YELLOW only from the check-star test. Supersedes the p2p
choice of SIGMA-SYS-NIGHT-2026-10; the SIGMA-SYS-MODEL-2026-10 variance model (photon + scintillation +
reference) is unchanged.
CHECK-SCINT-2026-10 (code review 2 D10 / G08-10, with O1): the check-star K error adds the epoch's
scintillation, as the target error does.

## SKY-ONE-2026-10 (Milan 2026-10-28, phase-1 item O7) - empty apertures use the science sky
measure_empty_aperture_sigma_bkg takes the source-masked sky image (sky_estimation.sky_image) and the
annulus_sky_medians rule of the science photometry (SKY-SRC-MASK-2026-10) instead of the plain median of the
whole annulus. Effect ~1 % (Brno g sigma_bkg_ap / Howell 1.333 -> 1.321). The spatial excess over Howell (~32 %)
is static field structure: the time-domain noise of constant stars matches the CCD equation (p2p / err
0.85-1.0), so the LC errors correctly keep the CCD equation and do not use sigma_bkg_ap.

## Phase-1 items closed without code change (2026-10-28)
- O2 "Brno errors underestimated": constant check stars p2p / err 0.85-1.0; targets 1.1-1.2 because they vary.
- O3 check star Gaia DR3 2172313747292886528: -0.07 (r) / -0.13 (g) mag vs a field colour fit with 0.05-0.07
  mag scatter, no neighbour within 15"; no longer selected as check; COMP-CATOFF 0.20 / trust 0.10 catch worse.

## SKY-MASK-KEEP-2026-10 (Milan 2026-10-28, phase-1 item O4a) - per-frame source mask kept; residual trend documented
Faint isolated stars in a crowded field fade slightly with airmass: the per-frame source mask (SKY-SRC-MASK-2026-10)
follows the seeing (fewer faint sources detected in bad seeing, more of their light in the sky median). Brno 534,
isolated constant stars, flux-domain slope vs a bright isolated reference, G 16-17 / 17-18: k 1.0 +22/+52 (g),
+16/+39 (r) mmag/X; with the k 0.6 apertures of AP-SMALL-K +19/+38, +13/+28. Tested, not adopted: fixed MASTERSTAR
mask with a constant radius (1.5 x night median FWHM) +12/+23, +7/+19; with a radius growing with the frame FWHM
-16/-22, -14/-19 (masks most of the field in bad seeing); union with the per-frame mask -21/-34; mean / mode
estimators; a per-frame additive sky offset fitted on constant stars (unstable). Bright stars (G < 15) are
unaffected by every variant. No mask removes the effect; it is a limit of aperture photometry in crowded fields.
Faint targets in crowded fields: the ePSF light curve (joint fit with Gaia neighbours) is the reference product.
BO CVn (X 1.01-1.22) cannot test it.
