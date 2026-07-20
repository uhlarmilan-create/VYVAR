CURSOR RESULT ù 2026-07-20

What I did
Implemented NIGHT_FIT v2 (`fit_k2_night`, `k2_feasibility_pregate`) in
`src_py/k2_extinction.py` per design spec v1.1 S5/S6; wired resolve + Phase 2A /
method LC apply paths for `K2Source.NIGHT_FIT`; synthetic recovery suite + draft-427
REFUSE fixture. Default `k2_fit_enabled=false`. ONE commit; not pushed.

## Scope finding
v2 was never implemented (config stubs only). Remaining activation blocker is
ONLY the B2 data night. C1 synthetic pre-validation is DONE against the real fitter.

## Output / findings

### Spec adherence
- Residuals: flux-derived Honeycutt (never proc catalog `mag`).
- Fit model after CM removal: `k2 * (C-Cref) * dX` (S5-identifiable form).
- Fit-frame subset: READ-ONLY from `align_residual_px` (+ optional quality mask).
- Pre-gate: monotonic refuse; detectability; outer tertile + arc consistency;
  plausibility ceiling + lit factor/sign.
- Refuse ? literature + `k2_fit_refuse_reason` in meta.
- No new config keys.

### Synthetic recovery sweep (accepted = within max(2 ?_boot, 0.005))
Matrix k2_true ù colour_spread ù noise_mmag; non-detectable cells must refuse
`detectability` (never silent wrong accept). High-leverage low-noise cells accept
and recover (e.g. k2=0.08, spread=1.0, noise=5 mmag). Zero-signal: accept?0 or
refuse; no fabricated large detection.

### REFUSE cases
| Case | Expected reason (family) |
|------|--------------------------|
| Monotonic X(t) | `monotonic_airmass` |
| Absurd k2=0.5 | plausibility / consistency |
| Split colour k2 | tertile inconsistent (or related) |
| Zero-signal high noise | `detectability` / plausibility / inconsistent |
| Draft 427 fixture | consistency and/or plausibility (items 3ù4) |

### 427 fixture provenance
`dev/validation/fixtures/k2_draft427_refuse.json` ù
`fixture_source=synthesized_from_decisions` (tmp/k2_fit427_v2 JSON gone; signature
from SPEC S1/S6 + `CURSOR_RESULT_recon_dao435_k2data.md`: ungated +56 mmag, tertile/
arc inconsistency, lit ~?4 mmag).

### Gates
- Suite green; two-run deterministic test PASS
- ruff clean; docs-sync green (FLOW ch 11.8 regenerated; facts unchanged)
- `--fast` OVERALL PASS (1031 passed)
- P1 golden 5/5
- `--full` OVERALL PASS ó BYTE-IDENTICAL (core `3d26f469Ö` n=333; extended `6420f1daÖ` n=499; science-compare n_lc=166 failures=0)

## Docs impact
- DECISIONS: `K2-NIGHT-FIT-V2-IMPLEMENTED`
- ROADMAP: v2 implemented + synthetic validation; activation only on B2; C1 DONE
- STATE: one-liner
- FLOW ch 11.8 + PDF regenerated; `flow_doc_facts` unchanged (`k2_fit_enabled=False`)
- `FLOW_DOC_V3_GAPS.md`: C1 DONE addendum

## Files changed
- `src_py/k2_extinction.py`, `src_py/photometry_core.py`, `src_py/method_lc_output.py`
- `dev/tests/test_k2_night_fit_recovery.py`
- `dev/validation/fixtures/k2_draft427_refuse.json`
- docs / FLOW / GAPS as above
