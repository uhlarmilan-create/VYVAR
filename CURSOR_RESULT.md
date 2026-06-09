CURSOR RESULT -- 2026-06-08T20:00:00Z

What I did
NEIGHBOR-SUB step 2a: strengthened guards to eliminate FAIL-SILENT under realistic PSF mismatch.
Inclusive sep floor (<=0.8), catalog-anchored neighbor_overfit / target_undershoot / subtract_harmed,
sky-noise SNR check. Re-ran A9 realistic-mismatch diagnostic. No production wiring (2b not started).

## Output / findings

### Guard changes (`psf_neighbor_sub.py`, `config.py`)
- `nn_dist_fwhm <= neighbor_sub_refuse_sep_fwhm` (0.8 inclusive) -- sep 0.5 + 0.8 -> PASS-REFUSE
- `neighbor_overfit`: fitted neighbour aperture mag brighter than `nn_mag` by >0.3 mag
- `target_undershoot`: recovered mag fainter than catalog `target_mag` by >0.2 mag
- `subtract_harmed`: mild contamination + clean < 95% of plain
- `nonphysical_flux`, `low_recovered_snr` (sky+read noise, not residual RMS)
- A9 `measure_cell` passes `target_mag`, `nn_mag`, `flux_zp`

### Realistic mismatch diagnostic (post-2a)

| Metric | Pre-2a | Post-2a |
|--------|--------|---------|
| FAIL-SILENT | 14 | **0** |
| HV PASS-RECOVER | 16.7% | **17.6%** |
| REFUSE correctness | 62.5% | **100%** |
| Verdict | BLOCK_2B_GUARDS | **SAFE_LOW_YIELD** |

HV yield: only dM0 cells at sep 1.0-1.5 recover (faint neighbour); bright-neighbour cells refuse
(target_undershoot / nonphysical). **2b blocked** -- safe but low-yield at coarse bin2; recommend
fine-scale A9 (draft 367) and/or ePSF improvement before pipeline wire.

### Verify
| check | result |
|-------|--------|
| pytest | 203 passed, 6 skipped |
| numeric SHA draft_000366 | 770966c3 unchanged |
| psf_neighbor_sub_enabled | OFF |

## Errors (if any)
None.

## Files changed
psf_neighbor_sub.py, config.py, tests/validation/a9_core.py, tests/test_psf_neighbor_sub.py
tests/validation/data/tier_a9/a9_mismatch_diagnostic.json, .md
docs/VYVAR_NEIGHBOR_SUB_DESIGN.md, VYVAR_VALIDATION.md, VYVAR_JOURNAL.md, CURSOR_RESULT.md

Not committed.
