# CURSOR RESULT - GATE-REGIME-01

Date: 2026-08-15
Repo tip at issue: 691d3be
Commit: (filled after local commit)
Type: BOUNDED FIX (failure path + provenance). Happy path science unchanged.
Push: NO

## Premise

Compared: global comparison-pool admission control flow before vs after GATE-REGIME-01.
Difference: explicit `CompPoolRegime` (DERIVED | LEGACY | FAILED), INV-NO-SILENT-EMPTY raise when derived admission empties a non-empty pool, and persisted `comp_pool_admission.json` / `pipeline_meta.comp_pool_admission`. Thresholds and `comp_pool_derived_admission` default are unchanged.

## What I did

1. Replaced the `None` / empty-set / populated-set tri-state with `CompPoolRegime` in `comp_pool_noise.py`.
2. Made legacy RMS prefilter and derived filter mutually exclusive by regime enum (never `not bool(admitted_ids)`).
3. FAILED (exception or missing decisions) raises `CompPoolAdmissionError` instead of silent legacy downgrade.
4. Wired `assert_population_nonempty` / `PopulationEmptiedError` (INV-NO-SILENT-EMPTY) at derived admission.
5. Persist `(n_in, n_out, rule_id, threshold, unit, regime)` plus `reject_reason_counts` to `photometry/comp_pool_admission.json`, mirrored into `pipeline_meta.json`.
6. Caller no longer swallows admission/empty errors into per-target fallback.
7. Documented INV-NO-SILENT-EMPTY in `docs/VYVAR_INVARIANTS.md` with fire-proof test.

## Output / findings

### Happy path (byte-identical pool membership)

Draft 512, same inputs (`masterstars_full_match.csv` + 134 `proc_*.csv`), `comp_pool_derived_admission=True`:

| | SHA-256 of sorted catalog_id set |
|--|--|
| Pre-fix (691d3be) | `cccdda39bd74cfbd58f141bbe602e2eb58e45eea7780f51d54055d5b6598d77b` |
| Post-fix | `cccdda39bd74cfbd58f141bbe602e2eb58e45eea7780f51d54055d5b6598d77b` |
| Pool n | 29 (both) |

Science membership of `build_global_comp_pool` is identical. New artifacts (`comp_pool_admission.json`, meta key) are additive and outside photometry SHA scope (`pipeline_meta.json` already excluded).

Full Phase-1/2A draft rebuild not required to prove this function's happy path; pool ID set is the direct science output of the changed path.

### Failure path

Synthetic case (monkeypatch `analyze_draft_comp_pool` to admit zero stars) calls the **real** `build_global_comp_pool` and raises `PopulationEmptiedError` with `rule_id=COMP_POOL_DERIVED_ADMIT`, `n_in`, threshold, unit, population. Artifact written before raise. FAILED regime (exception in derived) raises `CompPoolAdmissionError` and does not run legacy.

### Downstream empty handling (not duplicated)

Empty global pool previously became `_global_pool_df = None` and silently fell back to per-target masterstars (`photometry_core.py` caller). That is a second silent path, not an attributable empty-pool guard. This task raises at the admission site and re-raises through the caller so the fallback does not hide derived failure. No second empty-guard was added on the same condition.

### Reject reasons

Per-star `reject_reasons` from `admit_pool_stars` are aggregated to `reject_reason_counts` in the sidecar (answers BO-ENSEMBLE / COMP-POOL Stage 2 attribution without logs).

### Provenance location choice

Sidecar next to `comparison_stars_per_target.csv` (`comp_pool_admission.json`) is the primary artifact (readable from output dir alone). Also stamped into `pipeline_meta.json` via existing `merge_photometry_pipeline_meta` (same provenance channel as other run facts; outside science SHA).

### Note on legacy RMS prefilter

`apply_rms_prefilter` remains a no-op inside `comp_pool_rms.py` (intentional `pass`). Regime exclusivity still matters: empty derived set previously ran the derived filter to zero while also requesting legacy; the empty-set bug is fixed regardless of the no-op.

## Tests

- `dev/tests/test_gate_regime_01.py`: 4 passed (fire proof, empty derived raise, FAILED no legacy, passthrough).
- `--fast`: see counts in commit note / companion result.

## Files changed

- `src_py/comp_pool_noise.py` - CompPoolRegime, artifact writer, reject_reason_counts
- `src_py/invariants_runtime.py` - assert_population_nonempty, PopulationEmptiedError
- `src_py/photometry_core.py` - build_global_comp_pool regime + caller re-raise
- `docs/VYVAR_INVARIANTS.md` - INV-NO-SILENT-EMPTY
- `dev/tests/test_gate_regime_01.py`

## Errors

None.
