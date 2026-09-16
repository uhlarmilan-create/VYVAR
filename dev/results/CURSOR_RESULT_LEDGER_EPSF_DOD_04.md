CURSOR RESULT - 2026-09-16 LEDGER-EPSF-DOD-04

Date: 2026-09-16. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 27bdac9 (verified HEAD; ancestor
check PASS). Class: DOCS-ONLY. No code, no numbers change.

## Premise (Rule 0.1)

Compared: D-EPSF-XVAL-DOD-03 and EPSF-XVAL-01 against the
literature-backed amendment for criterion 2. They differ.

Before this commit:

- D-EPSF-XVAL-DOD-03 criterion 2 was live: d = m_psf_inst - m_cat
  (Gaia->V), |b|<=5, resid RMS<=25; c recorded. Closure via
  EPSF-VAL-02 + 520.
- EPSF-XVAL-01 OPEN on DOD-03; VAL-02 R-W1/R-W2 PASS, R-W3 FAIL
  (resid RMS 519), R-W4 FAIL - treated as a criterion miss.
- No D-EPSF-XVAL-DOD-04 / CLOSE-TEXT-01; no ENS4-BLEND-01.
- Architect error 29 not ledgered.
- D5-1 / D10-1 lacked the VAL-03 / c=+140 annotate.

## What I did

Added D-EPSF-XVAL-DOD-04 (amends DOD-03 criterion 2 only) and
D-EPSF-XVAL-CLOSE-TEXT-01 (record; enact on closure). Annotated
DOD-03 criterion 2 as amended; VAL-02 R-W3 INCONCLUSIVE (method).
Updated EPSF-XVAL-01 to VAL-03 + 520 -> CLOSE-TEXT. Added
ENS4-BLEND-01 (MED). Annotated D5-1 and D10-1. Appended architect
error 29. Ran `--fast --clean`. Commit and push
`origin consolidate-01:consolidate-01` only.

## Output / findings (deltas only)

- Criterion 2 live bar: CoG-tied large aperture (r_L ~ 4 x FWHM);
  |b|<=5 mmag/mag; robust scatter 1.4826*MAD <= 15 mmag; c vs
  BP-RP recorded only. No catalogue transform; no blends in the
  reference.
- Criteria 1 and 3 unchanged. Status on 516: C1 PASS (VAL-02);
  C2 pending VAL-03; C3 pending 520.
- VAL-02 R-W3 reclassified INCONCLUSIVE (method / error 29), not
  a PSF-path FAIL.
- Close text prepared under CLOSE-TEXT-01; not enacted.

## Architect error ledger

25-28. See LEDGER-EPSF-DOD-02 / DOD-03 RESULTs (unchanged).

29. VAL-02 criterion 2: accuracy reference carried the colour
    transformation and no blend screen; threshold 25 mmag not
    derived from the reference's own scatter. Root class: same as
    26 (reference floor not measured before the bar was set).

## Errors (if any)

None.

## Files changed

- `docs/VYVAR_DECISIONS.md`
- `docs/VYVAR_ROADMAP.md`
- `docs/VYVAR_AUDIT_2026_REGISTER.md` (D5-1 / D10-1 annotate)
- `docs/VYVAR_STATE.md` / `docs/VYVAR_JOURNAL.md` (status sync)
- this file (architect-error ledger append)

## Gates

`--fast --clean` OVERALL PASS on `12c3248` (1629 passed, 34
skipped; clean-tree PASS). `a2/` never staged.

## STOP

DOCS only. EPSF-VAL-03 is its own task.
