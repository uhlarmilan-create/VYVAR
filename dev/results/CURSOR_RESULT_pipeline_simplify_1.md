CURSOR RESULT - 2026-07-22 14:35 UTC+2

What I did
Implemented PIPELINE-SIMPLIFY-1 in three commits (A/B/C): permanent skip-only preprocess,
Phase 1 skip for out-of-scope VSX types, and detection-limited VSX scope (mag param removed).
Ran pytest, ruff, `--fast`, and `--full` anchor gate on draft_435. **STOP before push.**

## Part A - SKIPPROC-PERMANENT (`013cb0c`)

- Removed `skip_processed_directory` end-to-end; `KNOWN_REMOVED_KEYS` logs INFO on legacy key.
- Preprocess is always in-place QC via `_qc_enrich_calibrated_in_place`; no `processed/lights` copy tree.
- Alignment QC allowlist + QC-01 are **unconditional** (requires `qc_metrics.csv`).
- **Deleted helpers (A3):** `_skip_processed_directory`, `_get_vy_qc_status`, `_preprocess_calibrated_one`, `_apply_temporal_sigma_clip_in_place`.
- **`_apply_temporal_sigma_clip_in_place`:** copy-mode-only (temporal outlier mask on `processed/` FITS tree); no callers remain after skip-only preprocess -- safe delete.
- **Resume marker (A6):** preprocess complete when `calibrated/lights/qc_metrics.csv` exists under the draft lights root (legacy read fallback: `processed/lights/qc_metrics.csv` for old drafts only).

## Part B - VSX-SCOPE-PHASE1-SKIP (`7b235fe`)

- Phase 1 loop `continue` on `skip_reason == "vsx_type_out_of_scope"` before Gaia enrichment / comp selection.
- Log: `Faza 1: N out-of-scope targets skipped (no comp selection)`; progress denominator excludes OOS targets.
- Saturated / `zone_flag` targets **not** skipped (PER-FRAME-SAT comment at skip site).
- Anchor note: `vsx_out_of_scope_types=[]` in config for gate; byte-identical expected from B alone.

## Part C - VSX-AUTO-MAGLIM (`a0e3431`)

- Removed static `mag_max` pre-filter and `vsx_variable_targets_mag_limit` parameter.
- Diagnostics/logs: `n_with_masterstar_match` / "detection-limited (DAO+Gaia match)".
- Report/UI: VSX scope line replaces numeric mag threshold.
- Consequences: `variable_targets.csv` may grow (all in-frame VSX); comp-pool veto coverage grows; `active_targets` criterion unchanged (DAO+Gaia on MASTERSTAR).

## Parameter counts

| | Registered |
|---|---|
| Before | 272 |
| After | 270 |
| Delta | -2 (`skip_processed_directory`, `vsx_variable_targets_mag_limit`) |

## Adapted tests

- `test_skipproc_qc_allowlist.py` (allowlist unconditional, no-processed-dir pin, KNOWN_REMOVED)
- `test_preprocess_sky_surface.py` (copy-mode tests removed; helper-only)
- `test_masterstar_obs_group.py`, `test_pre_calibrated_run.py` (draft lights root)
- `test_astrometry_fault_isolation.py` (calibrated + qc_metrics.csv fixture)
- `test_auto_vsx_limit_report.py`, `test_wave_a_report_config.py` (detection-limited scope row)
- `test_ui_params_dashboard.py` (owner partition 244->242 config_runtime)
- `test_f428_fixbatch.py`, `test_fail_safety_hygiene.py` (`variability_mag_limit`)
- `test_vsx_out_of_scope_types.py` (explicit `[]` for default equivalence)

## Gates

### pytest + ruff
- `1069 passed, 24 skipped`
- `ruff check src_py dev/tests` - All checks passed

### `--fast` (2026-07-22, head `f7c9278` at run start; tip `a0e3431` after doc amend)
```
OVERALL: PASS
pytest: 1069 passed, 24 skipped
```

### `--full` draft_435 (2026-07-22T13:00Z, ~2113s pipeline)
Pre-registered decision rule:
```
full-snapshot-sha-core       PASS   03d8fb6491bc3c22... n=333
full-photometry-sha-core     PASS   03d8fb6491bc3c22... n=333
full-photometry-sha-extended PASS   bbfcc92e7ac5c4c5... n=499
full-science-compare         PASS   n_lc=166 failures=0
OVERALL: PASS
```
**Verdict:** Byte-identical core+extended SHAs -> **PASS, arc complete.** No diff triage required. Ledger stamped `a0e3431`.

## Docs impact

- `VYVAR_DECISIONS.md`: SKIPPROC-PERMANENT, VSX-AUTO-MAGLIM; QC-ALLOWLIST unconditional
- `VYVAR_INVARIANTS.md`: QC-01 unconditional wording
- `VYVAR_CONFIG_GUIDE_EN/CZ.md`, `VYVAR_PARAMS.md`, `VYVAR_STATE.md`, `flow_doc_facts.py`
- FLOW PDF builder: `vsx_variable_targets_mag_limit` fact removed (regenerate PDF when Milan wants doc refresh)

## Commits (feature stack)

1. `013cb0c` feat(skipproc)!: skip-only preprocess; remove skip_processed_directory param and processed/ tree
2. `7b235fe` perf(phase1): skip comp selection for vsx_type_out_of_scope targets (mask-first listing preserved)
3. `a0e3431` feat(vsx)!: scope by DAO+Gaia detection; remove vsx_variable_targets_mag_limit

Plus bookkeeping commit (ledger + STATE + this result file).

## Push (2026-07-22, Milan authorized)

### Pre-push checks

| Check | Result |
|-------|--------|
| `git fetch origin`; `origin/main` | `8815c45` (unchanged as required) |
| Stack `git log origin/main..HEAD --oneline` | 4 commits - exact match (see below) |
| `git status --short` | Clean; allowlisted untracked only (`dy_peg_night_run_bvr.py`, `qatar8_night_run_v.py`) |
| `session_baseline_check.py --fast` (final HEAD) | **OVERALL PASS** - 1069 passed, 24 skipped |

### Pushed commits (feature stack, newest first)

```
a0e3431 feat(vsx)!: scope by DAO+Gaia detection; remove vsx_variable_targets_mag_limit
7b235fe perf(phase1): skip comp selection for vsx_type_out_of_scope targets (mask-first listing preserved)
013cb0c feat(skipproc)!: skip-only preprocess; remove skip_processed_directory param and processed/ tree
```

Base: `8815c45` -> stack tip: `a0e3431` (`git push origin main` succeeded; bookkeeping commit on top, not listed here).

### Bookkeeping

One docs commit (ledger stamp from `--full`, STATE param counts, this result file including `_apply_temporal_sigma_clip_in_place` confirmation) pushed with the stack.

### Final origin/main tip

Local HEAD matches `origin/main` after push. For current tip: `git rev-parse origin/main`.
