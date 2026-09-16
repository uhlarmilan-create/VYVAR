CURSOR RESULT - 2026-09-16 LEDGER-EPSF-DOD-05

Date: 2026-09-16. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 093f615 (verified HEAD).
Class: DOCS-ONLY. No code, no numbers change.

## Premise (Rule 0.1)

Compared: D-EPSF-XVAL-DOD-04 and the EPSF-XVAL-01 / D5-1 /
D10-1 rows against the DOD-05 state. They differ.

Before this commit:

- D-EPSF-XVAL-DOD-04 criterion 2 live as a single bar: |b|<=5
  and robust scatter <=15 mmag on uncorrected PSF-vs-CoG d(s).
- EPSF-XVAL-01 OPEN with criterion 2 FAIL on VAL-03 R-X1
  (scatter 58.6 mmag); closure waited Milan sequencing.
- CLOSE-TEXT-01 validated-domain clause lacked per-star offset
  stability / ensemble absorption; limitations lacked the
  ~40-60 mmag common-scale systematic.
- No AC-DESIGN-01 / CHROMATIC-PSF-01; no architect error 30.
- D5-1 lacked the b_ap = +74.2 raw-aperture growth-curve annotate.
- EPSF-AC-02 listed CLOSED (historical AC wiring only).

## What I did

Added D-EPSF-XVAL-DOD-05 (splits DOD-04 criterion 2 into 2a/2b),
AC-DESIGN-01, and CHROMATIC-PSF-01. Amended CLOSE-TEXT-01
validated-domain and limitations clauses. Annotated DOD-04
criterion 2 as historical / superseded by DOD-05. Updated
EPSF-XVAL-01 closure sequence to EPSF-AC-02 + 520; reopened
EPSF-AC-02 for 2a/2b + Part C; annotated VAL-03 and D5-1 /
D10-1. Appended architect error 30. Ran `--fast --clean`.
Commit and push `origin consolidate-01:consolidate-01` only.

## Output / findings (deltas only)

- Criterion 2 live bar: 2a LINEARITY |b|<=5 mmag/mag (simultaneous
  colour; bootstrap std reported; |b|-2*std not required for
  PASS) and 2b STABILITY split-half residual scatter <=10 mmag.
  Per-star offset scatter itself is RECORD only (VAL-03: 42 mmag
  after colour, 59 raw).
- Criteria 1 and 3 unchanged. Status on 516: C1 PASS (VAL-02);
  C2a/C2b pending EPSF-AC-02; C3 pending 520.
- VAL-03 R-X1 FAIL explained (error 30), not a live miss under
  DOD-05.
- AC-DESIGN-01: no per-star AC; `p4_none`; differential-only.
- CHROMATIC-PSF-01: c = +130 mmag/mag BP-RP (bootstrap pending
  EPSF-AC-02); feeds D10-1 / methods paper.
- Close text amended; not enacted until 1+2a/2b+3 hold.

## Architect error ledger

25-29. See LEDGER-EPSF-DOD-02 / DOD-03 / DOD-04 RESULTs
(unchanged).

30. VAL-03 design: criterion 2 evaluated on uncorrected PSF
    magnitudes in the PSF path's weakest domain with a threshold
    (15 mmag) set without predicting the CORE-03 phase-surface
    contribution (72 mmag ptp -> up to +/-36 mmag per star).
    Root class: threshold set without propagating an
    already-measured systematic into the expected floor.

## Errors (if any)

None.

## Files changed

- `docs/VYVAR_DECISIONS.md`
- `docs/VYVAR_ROADMAP.md`
- `docs/VYVAR_AUDIT_2026_REGISTER.md` (D5-1 annotate)
- `docs/VYVAR_STATE.md` / `docs/VYVAR_JOURNAL.md` (status sync)
- this file (architect-error ledger append)

## Gates

`--fast --clean` OVERALL PASS on `13beb3d` (1629 passed, 34
skipped; clean-tree PASS). `a2/` never staged.

## STOP

DOCS only. EPSF-AC-02 is its own task.
