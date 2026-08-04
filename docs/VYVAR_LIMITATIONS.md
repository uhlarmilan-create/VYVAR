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

## D1-1 -- Cosmic-ray rejection

**FIXED (batch E, 2026-08-04).** L.A.Cosmic via astroscrappy in preprocessing (`enable_lacosmic`).
Physical re-cut: 365810 pixels cleaned on 150 frames. Evidence:
`dev/results/CURSOR_RESULT_batch_E_physical_recut.md`.

## D1-2 -- Detector linearity correction

**DEFERRED** to observing plan. Per-sensor linearity curve (Howell 2006 sec 4.4) requires a
dome-flat ramp measurement per sensor. **Not** the chosen fix for D5-2.

## D5-2 -- Production flux vs catalogue magnitude compression

**FIXED (2026-08-04, batch E).** Mechanism: saturation / detector non-linearity G 8-9.
Saturation admission gate at **70%** full well (C-1/C-2). Physical re-cut G 8-9 slope
**-0.318 -> -0.491**. Evidence: `dev/results/CURSOR_RESULT_batch_B_revised.md`,
`dev/results/CURSOR_RESULT_batch_E_physical_recut.md`.

## WIDE-ERR -- Wide-rig quoted error underquoted

Wide-rig (equipment_id 1) quoted error underquoted ~2x vs check-star scatter (H1-global).
Affects **error bars only**, not fluxes. Fix routed: Honeycutt 1992 leave-one-out ensemble SEM +
photon-term audit. Must be resolved before a wide-rig submission claims its error bars. Status:
**OPEN** (future thread). Evidence: `dev/results/CURSOR_RESULT_wide_error_diag.md`.

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
