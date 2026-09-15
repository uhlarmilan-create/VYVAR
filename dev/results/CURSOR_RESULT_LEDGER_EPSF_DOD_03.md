CURSOR RESULT - 2026-09-15 LEDGER-EPSF-DOD-03

Date: 2026-09-15. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 0825932 (verified).
Class: DOCS-ONLY. No code, no numbers change.

## Premise (Rule 0.1)

Compared: D-EPSF-XVAL-DOD-02, EPSF-XVAL-01, EPSF-CORE-01 /
EPSF-CORE-04 R-P2, EPSF-BUILD-OSAMP-01, and D5-1 against the
literature-backed state for DOD-03. They differ.

Before this commit:

- D-EPSF-XVAL-DOD-02 was the live DoD: PRECISION ratio on all
  bright constants (no G split); ACCURACY PSF-minus-aperture
  slopes with |c| as a criterion; closure via EPSF-VAL-01 + 520.
- EPSF-XVAL-01 OPEN on DOD-02; VAL-01 R-V1/R-V2/R-V3 FAIL on the
  record under that bar.
- EPSF-BUILD-OSAMP-01 OPEN LOW (record only).
- CORE-04 R-P2 read as "sampling alone does not fix phase".
- No D-EPSF-PHASE-ROUTES-01; no IPS-01 / BRIGHTER-FATTER-01;
  D5-1 CLOSED without the CoG common-scale annotate.
- Architect errors 27-28 not yet ledgered.
- No `docs/VYVAR_LITERATURE_CHECK_EPSF_20260915.md`.

## What I did

Filed the architect literature memo. Added D-EPSF-XVAL-DOD-03
(supersedes DOD-02) and D-EPSF-PHASE-ROUTES-01. Marked DOD-02
SUPERSEDED (historical text kept). Updated EPSF-XVAL-01 to
DOD-03 via EPSF-VAL-02 + 520. Raised EPSF-BUILD-OSAMP-01 to MED
(blocks Route A). Annotated CORE-01/04 R-P2 re-read and D5-1.
Added IPS-01 / BRIGHTER-FATTER-01 (LOW). Appended architect
errors 27-28 here. Ran `--fast --clean`. Commit and push
`origin consolidate-01:consolidate-01` only.

## Output / findings (deltas only)

- DoD: PRECISION now G-domain split (D: G>=9.5 ratio bar;
  G<9.5 admission/picker test). ACCURACY rebased on Gaia-
  transformed m_cat; |b|<=5 and resid RMS<=25 mmag; colour
  term c RECORDED only; aperture fit RECORDED vs D5-1.
  CODE unchanged (520 fix list). Thresholds freeze at
  EPSF-VAL-02.
- Phase: D-EPSF-PHASE-ROUTES-01 (Route A osamp>=3 blocked on
  BUILD-OSAMP-01; Route B dithered AK under TODO-A).
- VAL-01 under DOD-02 = historical. EPSF-VAL-02 is the next
  measurement (own task).
- Literature memo referenced from DOD-03.

## Architect error ledger

25-26. See `CURSOR_RESULT_LEDGER_EPSF_DOD_02.md` (unchanged).

27. VAL-01 criterion 2: accuracy criterion built on a
    reference (raw per-star SNR-optimal apertures) that the
    July audit had already flagged as not aperture-corrected
    (D5-1). Root class: reference not checked against the
    audit's own findings.

28. CORE-04 R-P2 reading: "sampling alone does not fix the
    phase component" asserted from a pathological rebuild;
    literature (RASTI 2025) shows osamp 3 should suffice for
    Gaussian-like ePSFs. Root class: builder failure read as
    physics.

## Errors (if any)

None.

## Files changed

- `docs/VYVAR_LITERATURE_CHECK_EPSF_20260915.md` (new)
- `docs/VYVAR_DECISIONS.md`
- `docs/VYVAR_ROADMAP.md`
- `docs/VYVAR_AUDIT_2026_REGISTER.md` (D5-1 annotate)
- `docs/VYVAR_STATE.md` / `docs/VYVAR_JOURNAL.md` (status sync)
- this file (architect-error ledger append)

## Gates

`--fast --clean` PENDING at commit time; stamped after PASS.
`a2/` never staged.

## STOP

DOCS only. EPSF-VAL-02 is its own task.
