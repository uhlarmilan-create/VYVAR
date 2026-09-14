CURSOR RESULT - 2026-09-14 LEDGER-EPSF-XVAL-DOD-01

Date: 2026-09-14. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 8fee250 (verified HEAD at start).
Class: DOCS-ONLY. No code, no numbers change.

## Premise (Rule 0.1)

Compared: ROADMAP/DECISIONS rows for EPSF-XVAL-01 and
EPSF-SHAPE-01 (and the existing EPSF-CORE-01 row) against
Milan's 2026-09-14 D-EPSF-XVAL-DOD-01 text. They differ.

ROADMAP before this commit:

- EPSF-XVAL-01 OPEN, one-line "method unspecced", blocked-on
  "literature spec". No A2 number, no closure condition.
- EPSF-SHAPE-01 OPEN, one-line "root narrow ePSF core
  (FWHM 2.36 vs 3.30); routed to EPSF-CORE-01". That is the
  2026-08-23 AC-arc row, not the 2026-09-14 measurement.
- EPSF-CORE-01 OPEN, one-line "literature-parameter ePSF
  rebuild", blocked-on FUTURE.

DECISIONS before this commit: no D-EPSF-XVAL-DOD-01 heading.
EPSF-LC-LOG-01 / INV-PSF-SUBMIT-01 already say internal
diagnostic + no AAVSO/VarAstro PSF export; those headings
were left unchanged (sidecar wording already matches).

No dedicated `docs/` architect-error file exists. Numbered
entries currently live in CURSOR_RESULT_* (18-19 in
RED_TARGET_T4_01; 20 in EPSF_XVAL_A2_COMPARE; 21 first
written in EPSF_SHAPE_01). This RESULT is the append
location for 21 (restated) and 22 (new).

Did not invent a second XVAL, SHAPE, or CORE row.

## What I did

Added D-EPSF-XVAL-DOD-01 at the top of DECISIONS. Updated
the three existing ROADMAP rows in place (SHAPE moved to
CLOSED this arc). Appended architect errors 21-22 here.
Ran `--fast --clean`. Committed and pushed
`origin consolidate-01:consolidate-01` only.

## Output / findings

- EPSF-XVAL-01 stays OPEN. Status line is A2 R-A2-3
  (target 10.48 / check 21.41 mmag, deg2, 134 identical
  epochs; PSFEx deg2 vs deg3 ~0.1-0.3 mmag). Closure =
  D-EPSF-XVAL-DOD-01 (R-A2-1 on an unchanged A2-COMPARE
  re-run; no threshold moves).
- EPSF-SHAPE-01 CLOSED (measurement complete, 8fee250).
  Reading R-SH3. Standalone spatial-FWHM finding recorded.
- EPSF-CORE-01 OPEN, next (fit machinery; task issued
  separately). Same row as before; blocked-on FUTURE
  replaced by next.
- PSF path remains internal-diagnostic until XVAL closes.
  Aperture path untouched (1.9503 mmag vs AIJ).

## Architect error ledger

21. SHAPE-01 reader-validation gate specified against header
    PSF_FWHM, which is 4.7 * PSF_SAMP by PSFEx sampling
    design, not a measured width; psfex.stdout empty
    (diagnostic is on stderr, different estimator). Root
    class: carried claim not verified on the governing
    artifact. Cursor's algebraic validation was the correct
    substitute. First written in
    `CURSOR_RESULT_EPSF_SHAPE_01.md`.

22. A2 kit, self-reported: SEEING_FWHM derived as qc
    fwhm_px (calibrated-grid, 5.14-5.30 px) times the
    ALIGNED frames' WCS scale; aligned-grid FWHM is
    ~2.36 px, so the value fed to SExtractor was ~2.2x
    overstated. Assessed harmless for A2: pass1/pass2
    params use no CLASS_STAR (sole SEEING_FWHM consumer);
    PSFEx sample selection is FLUX_RADIUS /
    SAMPLE_FWHMRANGE driven and 2.36 px is in range;
    deg2 vs deg3 stability consistent. Root class: grid of
    a number not named (same family as error 20). Rule
    reinforced: every FWHM number states its grid
    (calibrated vs aligned vs oversampled).

## Errors (if any)

None.

## Files changed

- `docs/VYVAR_ROADMAP.md`
- `docs/VYVAR_DECISIONS.md`
- this file (architect-error ledger append)

## Gates

`--fast --clean` OVERALL PASS (1629 passed, 34 skipped;
clean-tree PASS). HEAD at gate: 8fee250.

## STOP

DOCS only. No measurement, no code. EPSF-CORE-01 runs as
its own task.
