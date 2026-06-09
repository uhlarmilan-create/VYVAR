CURSOR RESULT — 2026-06-09

What I did
Implemented sandwich reported uncertainty (`psf_err_mode=sandwich_skyonly`) for sky-only PSF
weights. Flux unchanged; only `psf_flux_err` column changes.

## Change

Var(f_hat) = sum(w_i^2 * sigma_true_i^2 * P_i^2) / (sum(w_i * P_i^2))^2
- w_i = 1/sigma_sky^2 (fit weights)
- sigma_true^2 = sigma_sky^2 + f_hat * P_i / gain

## P3 OLD vs NEW

| mag | OLD P3 | NEW P3 |
|----:|-------:|-------:|
| 12 | 0.563 | **1.070** |
| 13 | 0.714 | 0.978 |
| 14 | 1.009 | 1.092 |
| 15 | 1.105 | 1.042 |
| 16 | 1.137 | 1.137 |
| 17 | 0.942 | 0.942 |

## Bias unchanged (post-AC %)

| mag | before sandwich | after |
|----:|----------------:|------:|
| 12 | +0.80 | +0.80 |
| 16 | +1.75 | +1.75 |

## VERDICT

**Fully publication-grade** at fine scale. V3d **PASS** (accuracy + precision + P3 mag<=17).

## Regression

- SHA `770966c3` unchanged
- pytest: 218 passed
- A9: FAIL-SILENT 0, HV 83.3%

Report: `tests/validation/data/tier_v3d/v3d_sandwich_proof.md`
