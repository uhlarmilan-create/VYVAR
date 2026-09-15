CURSOR RESULT - 2026-09-15 LEDGER-EPSF-DOD-02

Date: 2026-09-15. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: e910562 (verified HEAD at start).
Class: DOCS-ONLY. No code, no numbers change.

## Premise (Rule 0.1)

Compared: ROADMAP/DECISIONS rows for EPSF-XVAL-01 and
EPSF-CORE-01 (and existing CORE-02..04 / SHAPE closed rows)
against Milan's 2026-09-15 D-EPSF-XVAL-DOD-02 text. They differ.

Before this commit:

- D-EPSF-XVAL-DOD-01 was the live DoD: XVAL closes on R-A2-1
  (<= 3.0 mmag vs PSFEx on an unchanged A2-COMPARE re-run).
  Sequencing note already flagged CORE-04 R-R2 as a possible
  DoD re-decide.
- EPSF-XVAL-01 OPEN, blocked-on D-EPSF-XVAL-DOD-01, one-line
  still the A2 R-A2-3 numbers.
- EPSF-CORE-01 CLOSED as the single CORE-01 measurement
  (R-C0, R-C1); CORE-02..04 each have their own CLOSED rows.
- No EPSF-BUILD-OSAMP-01 row. TODO-A had no dithered-ePSF
  sub-item.
- Architect errors 25-26 not yet in a ledger RESULT
  (25 was noted in CORE-04 RESULT / STATE prose only).

Did not invent a second XVAL or CORE-02..04 row. Updated
EPSF-CORE-01 in place to the investigation-complete summary.
CORE-02..04 CLOSED lines left as the measurement records.

## What I did

Added D-EPSF-XVAL-DOD-02 at the top of DECISIONS (supersedes
DOD-01). Marked DOD-01 SUPERSEDED; kept its historical bar
text. Updated EPSF-XVAL-01 OPEN row to DOD-02 via EPSF-VAL-01
+ 520 fix list. Rewrote EPSF-CORE-01 CLOSED as investigation
complete (CORE-01..04). Added EPSF-BUILD-OSAMP-01 (LOW).
Extended TODO-A with the optional dithered ePSF sub-item.
Appended architect errors 25-26 here. Ran `--fast --clean`.
Committed and pushed `origin consolidate-01:consolidate-01`
only.

## Output / findings

- EPSF-XVAL-01 stays OPEN. Closure = D-EPSF-XVAL-DOD-02
  (PRECISION / ACCURACY / CODE) via EPSF-VAL-01 + the 520
  re-cut fix list. R-A2-3 stays historical (superseded bar).
- New DoD: PRECISION ratio vs aperture on >=4 check-class
  stars; ACCURACY colour/mag slopes; CODE =
  FIT-OK-ADMISSION-01 / GAIN-FALSY-01 / FIXPOS-NOOP-01 gated
  at the era re-cut. Thresholds adjustable until EPSF-VAL-01
  runs, then frozen.
- Aperture remains the science product for bright targets on
  this rig. Phase component deferred to dithered ePSF under
  TODO-A only if crowded/faint PSF use is needed.
- EPSF-CORE-01 CLOSED as the CORE-01..04 investigation
  rollup. EPSF-BUILD-OSAMP-01 OPEN LOW (record only).

## Architect error ledger

25. CORE-03 R-Q4: "slope x observed phase spread" is a
    peak-to-peak quantity, not an RMS. "8.56 of 10.48"
    over-states the phase share. Correct: ptp over the
    ~0.095 px live window ~8 mmag; RMS (uniform phase) =
    ptp/sqrt(12) ~2.3 mmag; empirical share from CORE-01
    target phase rank R^2 0.221 ~4.9 mmag in quadrature.
    Phase is a co-driver, not 82% of the target residual.
    R-Q3 stands unchanged. First written in
    `CURSOR_RESULT_EPSF_CORE_04.md` (task housekeep) /
    CORE-04 framing.

26. DOD-01: validation bar transferred from an
    aperture-vs-aperture precedent (1.9503 mmag vs AIJ) to a
    PSF-vs-PSF comparison without noise accounting;
    unreachable by construction on this set (aperture check
    RMS_med 8.23 mmag; CORE-04 A4 / R-R2). Root class:
    reference point not derived from the comparison's own
    noise floor.

## Errors (if any)

None.

## Files changed

- `docs/VYVAR_ROADMAP.md`
- `docs/VYVAR_DECISIONS.md`
- `docs/VYVAR_STATE.md` / `docs/VYVAR_JOURNAL.md` (status sync)
- this file (architect-error ledger append)

## Gates

`--fast --clean` OVERALL PASS (1629 passed, 34 skipped;
clean-tree PASS). `a2/` never staged.

## STOP

DOCS only. No measurement, no code. EPSF-VAL-01 is its own
task.
