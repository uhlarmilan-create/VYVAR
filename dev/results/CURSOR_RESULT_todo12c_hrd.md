CURSOR RESULT - 2026-07-10 (TODO-12c-HRD)

What I did
Fixed Stage-2 label priority (RSG/RG before Very cool; giant branch blocks cool fallback), added Stage-1 per-net reserved slots (`hrd_min_per_net=4`), validated on draft_425/424, and closed the TODO-12 HRD arc with Milan-authorized commits.

## Output / findings

### Label priority (pre-12c ? post-12c, draft_425 B)
| Metric | pre-12c | post-12c |
|--------|---------|----------|
| Table rows | 6 | 10 |
| Red supergiant | 2 (+ 1 Very cool mislabeled s*r) | **3 RSG** + 1 Very cool/s*r |
| Very cool (s*r) mislabel | 1 (`Very cool ... SIMBAD: s*r`) | 1 (logg N/A - honest fallback) |
| Binary | 0 | 3 (cap) |

Same pattern on V/R: 3 RSG + 3 Binary + 2 LP Very cool + 1 residual Very cool/s*r.

**Remaining otype conflict (reported, not tuned):** `458407464445792384` - M_G=-5.5, bp_rp=3.43, Gaia logg N/A ? stays Very cool while SIMBAD says s*r. Spec allows Very cool when logg missing.

### Luminous-net enrichment (draft_425 B sample)
| catalog_id (tail) | M_G | BP-RP | teff (Gaia TAP) | in Stage-1 pick |
|-------------------|-----|-------|-----------------|-----------------|
| ...44792 | -5.50 | 3.43 | NaN | yes |
| ...523392 | -5.50 | 0.87 | NaN | yes |
| ...422400 | -5.44 | 0.43 | NaN | yes |
| ...296000 | -5.42 | 0.75 | NaN | yes |

No teff >= 25000 (acceptable - GSP-Phot NaN/underestimation for reddened cluster stars). Luminous-net stars now reach enrichment (`in_stage1_pick: true` for top 4).

### draft_424
Reliable 3515 (unchanged); table 4 ? 8 (per-net reservations expose more categories); no otype conflicts.

## DoD
- pytest: **696 passed**, 15 skipped (+3 tests)
- session_baseline_check --fast: PASS
- PDF overflow draft_425 B: **0 violations**
- Offline run: clean

## Evidence
- `tmp/todo12_hrd/summary.json`, PNGs
- Before: `tmp/todo12_hrd/pre12c/`

## Fixture updates
- `test_stage1_hits_each_criterion`: `min_per_net=0` for cap test (reservations would alter count)
- Added: `test_stage2_supergiant_beats_very_cool`, `test_stage1_luminous_net_reserved_slots`, `test_shrink_net_reservations_round_robin`

## Files changed
- `hrd_analysis.py`, `config.py`, `tests/test_hrd_extreme.py`, `scripts/todo12_hrd_validate.py`
- `docs/VYVAR_PARAMS.md`, `docs/config_schema.md`, ROADMAP/STATE/JOURNAL

## Commit status
Milan-authorized 4-commit series + push (see git log).
