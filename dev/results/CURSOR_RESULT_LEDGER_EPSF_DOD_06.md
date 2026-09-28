CURSOR RESULT - 2026-09-28 LEDGER-EPSF-DOD-06

Date: 2026-09-28. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 672a6a7 (verified HEAD).
Class: DOCS-ONLY. No code, no numbers change.

## Premise (Rule 0.1)

Compared: D-EPSF-XVAL-DOD-05, D-EPSF-XVAL-CLOSE-TEXT-01, and
EPSF-XVAL-01 against the DOD-06 state. They differ.

Before this commit:

- DOD-05 status: criteria 2a/2b PASS; closure = 1+2a/2b+3 then
  enact CLOSE-TEXT-01; phase surface still cited as a driver of
  the per-star residual.
- CLOSE-TEXT-01 was the live (unenacted) close statement.
- EPSF-XVAL-01 blocked on "520 re-cut" -> CLOSE-TEXT-01; no
  ERA-520-RECUT-01 row; no PSF-COMMON-SCALE-RESIDUAL-01.
- Architect error 31 not ledgered.
- CHROMATIC-PSF-01 already carried c = +137.5 +/- 48.3 from AC-02
  (no number change; annotate only).

## What I did

Added D-EPSF-XVAL-DOD-06 (criterion 2 met; closure sequence fixed)
and D-EPSF-XVAL-CLOSE-TEXT-02 (supersedes CLOSE-TEXT-01). Annotated
DOD-05 and CLOSE-TEXT-01 as superseded for closure. Updated
EPSF-XVAL-01; added ERA-520-RECUT-01 and
PSF-COMMON-SCALE-RESIDUAL-01; annotated CHROMATIC-PSF-01.
Appended architect error 31. Ran `--fast --clean`. Commit and push
`origin consolidate-01:consolidate-01` only.

## Output / findings (deltas only)

- Criterion 2 confirmed met on 516 (AC-02 R-AC1/R-AC2/R-AC4).
- Caveat: 2/36 split tests exceed 3x own noise (~10-15 mmag minority
  wander); limitation in CLOSE-TEXT-02, not a FAIL.
- Error 31: CORE-03 T1 does not drive per-star offsets (rho=0.13);
  ~37 mmag residual UNATTRIBUTED (PSF-COMMON-SCALE-RESIDUAL-01).
- Closure sequence: ERA-520-RECUT-01 -> VAL-02/AC-02 re-check on
  520 products -> enact CLOSE-TEXT-02.
- CLOSE-TEXT-01 superseded; not to be enacted.

## Architect error ledger

25-30. See LEDGER-EPSF-DOD-02..05 RESULTs (unchanged).

31. CORE-03 T1 phase surface predicted to drive per-star
    common-scale offsets (+/-36 mmag); AC-02 Part C rho=0.13,
    p=0.61 - NOT confirmed. Root class: a measured mechanism
    (per-epoch, one star) generalized to a different population
    (per-star, 18 stars) without a test. Phase remains established
    only for the target's per-epoch component (~2-5 mmag RMS).

## Errors (if any)

None.

## Files changed

- `docs/VYVAR_DECISIONS.md`
- `docs/VYVAR_ROADMAP.md`
- `docs/VYVAR_STATE.md` / `docs/VYVAR_JOURNAL.md` (status sync)
- this file (architect-error ledger append)

## Gates

`--fast --clean` OVERALL PASS on dirty tree at base `672a6a7`
(1629 passed, 34 skipped; clean-tree PASS). Stamp after commit.

## STOP

DOCS only. ERA-520-RECUT-01 is its own task.
