CURSOR RESULT — 2026-06-05 (verify_mag_limit A/B)

What I did
Added verify instrumentation (catalog_load_s, verify_s, cone_n_cat, max_false_n_matched, early_exit_*), fraction-based early-exit (`blind_verify_early_fraction=0.20`), A/B harness (`--verify-mag-limit`, `scripts/blind_verify_mag_ab.py`), ran 16 vs 14 experiment, adopted **verify_mag_limit=14** as default.

## A/B results (4-field subset)

| id | HIT 16/14 | sep Δ | n_matched 16→14 | max_false 16→14 | total_s 16→14 | early_exit |
|---|---|---|---|---|---|---|
| 365_wide | HIT/HIT | 0 | 86→95 | 24→22 | 268→279 | True/True |
| 380_B | HIT/HIT | 0 | — | 59→59 | 838→363 | False/False |
| 375_B | HIT/HIT | 0 | — | 60→60 | 122→122 | False/False |
| 368_Blue | HIT/HIT | 0 | — | 48→48 | 45→45 | False/False |

Subset totals: **1273 s → 808 s** (−36%). Wide margin truth/false@14: **95/22 ≈ 4.3×** (≥2× required).

## Full battery @ mag 14

| Metric | mag 16 (baseline) | mag 14 (adopted) |
|---|---|---|
| HIT rate | 10/10 | **10/10** |
| median sep | 0.146° | **0.146°** |
| 365_wide sep | 0.1242° | **0.1242°** |
| 365_wide n_matched | 86 | **95** |
| 365_wide max_false | ~31 (prior) / 24 (A/B) | **22** |
| Total battery time | ~2124 s | **~1529 s** (−28%) |
| early_exit wide | True | **True** |

CSV: `validation/blind_solve_rate_mag14.csv`, A/B: `validation/mag_ab/`

## Decision

**Adopt `verify_mag_limit=14`** — all adoption criteria met (parity, margin ≥2×, faster battery). Default updated in `config.py` + PARAMS.

Early-exit now uses **fraction ≥ 0.20** (depth-independent); absolute floor remains fallback when fraction=0.

## Errors (if any)
None in final validation. Tests: 23/23 blind suite green.

## Files changed
- `vyvar_platesolver.py` — instrumentation, fraction early-exit
- `config.py` — `blind_verify_early_fraction`, default `verify_mag_limit=14`
- `scripts/blind_solve_rate.py` — metrics CSV, `--verify-mag-limit`, `--fields`
- `scripts/blind_verify_mag_ab.py` — new A/B runner
- `scripts/_build_vyvar_params.py`, `tests/test_blind_verify.py`, `VYVAR_PARAMS.md`
