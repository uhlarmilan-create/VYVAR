CURSOR RESULT - 2026-08-02 10:45 UTC+2

What I did
Produced batch C decision brief for Milan: four DECISION items with physics, options, literature,
recommendations, and post-implementation confirming numbers. Documentation only; no code, no
measurement, no entry in `VYVAR_DECISIONS.md` (awaiting Milan).

## Deliverable

**Primary:** `docs/VYVAR_DECISION_BRIEF.md`

## One-line recommendations

| item | register | recommendation |
|------|----------|----------------|
| **C.1 I-11** | 21 | **Option 1** -- use pre-subtraction `sky_surface_bg_median_adu` in Howell sky term |
| **C.2 I-04** | 22 | **Option 1** -- NaN + exclude epoch when ensemble scatter unmatched |
| **C.3 P-02/A-6** | 25 (+ A-5, A-6) | **Option 3** -- wire scintillation first, then per-rig `sigma_sys` floor if chi2_red still > ~1.2 |
| **C.4 T4-1** | 10 | **Option B** -- single N_equiv correction; confirm **3.78** vs **4.71** from Part 2b rel_err |

## Confirming numbers (after batch D)

| item | metric |
|------|--------|
| I-11 | chi2_red unchanged on anchor (0 legacy-path epochs); rises toward correct on crowded drafts |
| I-04 | byte-identical anchor (0 unmatched epochs) |
| P-02/A-6 | median check-star chi2_red within ~20% of 1.0 on >= 2 rigs, before/after reported |
| T4-1 | stable effective detection threshold; Milan must choose before re-cut |

## Docs impact

- `docs/VYVAR_DECISION_BRIEF.md` -- **new**
- `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md` -- items 10, 21, 22, 25 annotated
- `docs/VYVAR_STATE.md`, `docs/VYVAR_ROADMAP.md` -- batch C complete
- `docs/VYVAR_DECISIONS.md` -- **no entry** (Milan chooses)

## Files changed

- `docs/VYVAR_DECISION_BRIEF.md` (new)
- `dev/results/CURSOR_RESULT_batch_C.md` (this file)
- `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md`
- `docs/VYVAR_STATE.md`
- `docs/VYVAR_ROADMAP.md`
