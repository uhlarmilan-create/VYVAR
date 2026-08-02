# VYVAR -- Known limitations (methods-paper section)

**Date:** 2026-08-02
**Purpose:** Referee-facing summary of documented and deferred audit items with measured
magnitude, fix status, and literature reference. Updated at audit closure batches A and B.

---

## A-1 -- SNR-table differential aperture (no curve-of-growth correction)

The SNR aperture table assigns magnitude-dependent radii (`aperture_r_px`, clamp 1.916 px at
the faint end) with **no curve-of-growth correction**, so the enclosed-flux fraction differs
between target and comparison stars and does not fully cancel in the differential. Verdict
robust (all 5x3 proxy cells exceed the 10 mmag gate, Step 1d). The physics expectation from
`dev/tools/closure_a1_reference_fixture.py` is **~144 mmag** for G 8-9 comparisons over the
anchor r50 span. **Recommended fix:** enable `cog_aperture_correction_enabled` (Stetson 1990
growth-curve aperture correction). Deferred to numeric batch D. Evidence:
`dev/results/CURSOR_RESULT_closure_step1g.md`, register item 16.

## A-9 -- Absolute PSF / FWHM scale unresolved

FWHM estimators disagree (`VY_FWHM_GAUSS` 2.395 px, header `VY_FWHM` 3.207 px, COG identities
4.0-4.9 px) because they measure different quantities (Gaussian core fit vs moment vs
curve-of-growth on a non-Gaussian PSF). **Not blocking** differential photometry. **Required
before any absolute claim** (enclosed-flux fraction, absolute SNR, detection completeness,
D1-2 `fwhm_ratio` test). Use curve-of-growth r50 as scale proxy, not a fitted FWHM. Status:
**DOCUMENTED**. Evidence: register item 31, Step 1f.

## D1-1 -- No cosmic-ray rejection

No cosmic-ray rejection in `src_py` (grep confirms only gain/read-noise "cosmic" parameter names).
Single-frame MASTERSTAR and per-frame photometry limit damage to isolated outlier epochs. Hard
prerequisite for stacking or coaddition (`docs/VYVAR_TODO_MASTERSTAR_REFERENCE.md`); publication
gap vs standard pipelines. **Fix:** L.A.Cosmic (van Dokkum 2001, PASP 113, 1420) or astroscrappy.
Scheduled in implementation batch E. Status: **DOCUMENTED**, CR-1 **QUEUED**.

## D1-2 -- Detector linearity correction

Closure Steps 1k-1n and batch B tested peak-ADU vs flux-deficit correlations. Batch B (B2) with
G 9-11 reference slope (-0.414) gives partial correlation deficit~peak | r50 = **+0.37** on
G < 9 (below the pre-registered +0.4 threshold). Faint-half (G > 10) reference slope is -0.18,
not -0.4, confounding the pre-registered test. **Not confirmed** on the unconfounded criterion.
Status: **DEFERRED**. Literature: Howell (2006) sec 4.4. Evidence:
`dev/results/CURSOR_RESULT_batch_B.md`.

## D5-2 -- Production flux vs catalogue magnitude compression

Pipeline flux does not scale as 10^(-0.4 G). Measured slopes: `flux` **-0.296**, `flux_large`
(fixed 9.58 px) **-0.269** (4058 star-frames, anchor draft_435). Localisation: G 8-9 bin
**-0.258** (sharp break to -0.434 at G 9-10). **Mechanism open** after batch B (B-open): B1
before/after sky test **VOID** (sanity gate); B2 does not meet non-linearity threshold. Status:
**MEASURED**, mechanism **DEFERRED**. Evidence: Steps 1k-1n, `dev/results/CURSOR_RESULT_batch_B.md`.

## I-12 -- Proper motion when pmra/pmdec absent

Gaia proper-motion correction is a no-op when PM columns are missing; logging fixed (WARNING).
Not a photometry bias on the anchor; deferred to Gaia DR4 for full PM coverage. Status: **FIXED**
(logging). Evidence: `CURSOR_RESULT_audit_t2.md`.

## D11-1 -- Dilution / crowding (G proxy)

Crowding and blend dilution are not fully propagated into the reported error budget; Gaia G used
as a proxy where crowding metrics are incomplete. Affects faint-end comp selection context, not
the anchor differential aperture closure. Status: **DOCUMENTED**. Evidence: Stage 3 forensics.

## D12-1 -- Sigma-clip bias in ensemble statistics

Iterative sigma clipping on comparison-star ensembles introduces a small bias toward lower
scatter (standard in Honeycutt 1992 pipelines). Magnitude not re-measured on anchor; acknowledged
for crowded fields. Status: **DOCUMENTED**. Literature: Honeycutt (1992).

## U-09 -- DATE-OBS convention per rig

BO CVn wide rig: DATE-OBS = shutter-open, **verified**. Other rigs: convention not verified per
rig; a +EXPTIME/2 error is invisible in light-curve shape but fatal for times of minimum. QHY294
has no DATE-END/EXPMID header; driver convention **UNVERIFIED** before timing-critical submission.
Status: **MEASURED** (home rig), **DOCUMENTED** (others). Evidence: `CURSOR_RESULT_audit_stage2.md`.

## MASTERSTAR architecture (enhancement thread)

Single-frame MASTERSTAR copy is scientifically usable but non-standard vs stacked reference
(Stetson 1994). Frame-selection metric `I_j`, median stack, provenance (register Steps 1-6),
admission gate C-1/C-2, and proper coaddition (TODO-B) are **enhancements**, not audit-correctness
blockers. See `docs/VYVAR_TODO_MASTERSTAR_REFERENCE.md`, `docs/VYVAR_MASTERSTAR_REFERENCE_ARCHITECTURE.md`.
