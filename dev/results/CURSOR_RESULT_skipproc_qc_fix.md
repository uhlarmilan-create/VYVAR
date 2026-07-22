CURSOR RESULT - 2026-07-22

What I did

Pre-step housekeeping (two commits on clean tree), then skip-processed QC fix
(design C+A): qc_metrics.csv allowlist gates alignment; full-set in-place QC with
DB prefilter stamping; segmentation-FWHM self-reject removed in skip mode; QC-01
wired gate + docs.

## Pre-step commits

| Commit | Summary |
|--------|---------|
| `cff5502` | fix(invariants): INV-FLUX-02 median + skewed-flat guard test + doc |
| `a5ddfa1` | fix(ui): `:material/telescope:` icon + structured params JSON widgets |

Skewed-flat test indices corrected post-commit (`flat[:15,:]` so median=1.0).

## Implementation summary

### C — qc_metrics.csv allowlist (skip mode only)

- `astrometry_align_and_build_masterstar` (~14368): replaced VY_QC-header filter
  with `filter_files_by_qc_metrics_allowlist` + `find_qc_metrics_csv`.
- Missing CSV ? fail-closed `FileNotFoundError` with actionable message.
- Log: `skip_processed: using X/Y lights FITS (qc_metrics.csv allowlist) ...`
- Path match: normalized resolved absolute paths (casefold).

### A — full-set visitation + prefilter stamping

- `app._vyvar_execute_preprocess_pending` + `night_run._night_run_preprocess`:
  when skip mode, build `prefilter_rejected` via `build_prefilter_rejected_map`,
  pass `only_paths=None`, visit all lights.
- `_qc_enrich_calibrated_in_place`: new `prefilter_rejected` param; stamps
  `VY_QC=rejected_prefilter_fwhm` (+ diagnostics) on excluded frames; CSV row
  for every frame.

### Dual-estimator cleanup (skip mode)

- Removed `rejected_fwhm` / `rejected_elong` from skip-mode in-place QC.
- Statuses: `ok` | `rejected_prefilter_*` | `error:...` only.
- Non-skip `preprocess_calibrated_to_processed` copy path **unchanged**.

### QC-01 wired

- `invariants_runtime.check_qc01_skipproc_alignment` after allowlist join.
- `WIRED_INV_IDS` + `docs/VYVAR_INVARIANTS.md` registry row.
- `_get_vy_qc_status` comment: diagnostic only (DECISIONS QC-ALLOWLIST-AUTHORITY).

## Tests

New: `dev/tests/test_skipproc_qc_allowlist.py` (7 tests)

- Allowlist join: ok included, prefilter rejected excluded, disk-not-in-CSV excluded
- Missing CSV raises
- Full-set visitation + header/CSV stamping
- No `rejected_fwhm` / `rejected_elong` in skip mode
- QC-01 FAIL on violation
- Path key casefold

Updated: `dev/tests/test_invariants_p2.py` (QC-01 registry regex, flux02 skew fix)

## Test counts

```
dev/tests (full): 1067 passed, 24 skipped, 1 failed (pre-existing)
dev/tests/test_docs_sync_guard.py: 4 passed
dev/tests/test_skipproc_qc_allowlist.py: 7 passed
ruff (touched files): clean
```

**Pre-existing failure:** `test_ascii_policy.py::test_tracked_text_files_are_ascii`
(non-ASCII in `dev/results/OSC_GAP_INVENTORY.md`, `docs/VYVAR_ROADMAP.md` — not
introduced by this arc). UI emoji was **not** committed; `:material/telescope:` used.
The guard did **not** flag the transient emoji in the working tree before replacement
(guard scans tracked files only — **not a guard gap** for untracked edits).

## session_baseline_check.py --fast

```
OVERALL: FAIL (pytest exit 1 due to pre-existing ascii_policy failure above)
1067 passed otherwise; git/docs/config/ledger checks PASS/WARN as usual
```

Anchor `--full` re-run **not required**: draft_435 uses `skip_processed_directory=false`.

## Anchor-protection argument

| Function | Change | Skip guard |
|----------|--------|------------|
| `norm_fits_path_key`, `load_qc_metrics_status_by_path`, `filter_files_by_qc_metrics_allowlist`, `build_prefilter_rejected_map` | New helpers | Callers skip-only |
| `astrometry_align_and_build_masterstar` | Allowlist + QC-01 | `if _skip_processed_directory(...)` block only |
| `_qc_enrich_calibrated_in_place` | Full-set + prefilter; no seg reject | Only called from skip branch of `preprocess_calibrated_to_processed` |
| `preprocess_calibrated_to_processed` | `prefilter_rejected` param | Skip branch only; non-skip body untouched |
| `app._vyvar_execute_preprocess_pending` | prefilter map when skip | `_skip_processed_directory(_app_cfg)` |
| `night_run._night_run_preprocess` | same | same |
| `_get_vy_qc_status` | Comment only | N/A |
| `invariants_runtime.check_qc01_skipproc_alignment` | New gate | Invoked only from skip alignment block |

Non-skip copy-mode preprocess, calibration, phase2a, anchor draft_435 path: **byte-identical behavior**.

## Open findings

1. **Non-skip estimator mismatch (recorded, no code change):** legacy
   `preprocess_calibrated_to_processed` copy path still compares segmentation FWHM
   from `_qc_fwhm_elongation` against `reject_fwhm_px` (DAO-scale Auto limit when
   wired through UI). Same dead-check class as the skip-mode bug, but anchor
   draft_435 uses `skip_processed=false` and processed/ copy flow — left unchanged
   per anchor protection.

2. **ASCII policy:** pre-existing non-ASCII bytes in tracked OSC/ROADMAP docs
   (outside this arc).

3. **FLOW PDF:** no existing skip_processed frame-selection section in
   `build_flow_doc.py`; no FLOW builder edit required. DECISIONS + INVARIANTS
   updated instead.

## Docs impact

- `docs/VYVAR_INVARIANTS.md`: QC-01 **[wired]**
- `docs/VYVAR_DECISIONS.md`: QC-ALLOWLIST-AUTHORITY entry
- `docs/VYVAR_STATE.md`: wired-gates bullet (+ QC-01)
- `dev/tests/test_invariants_p2.py`: registry regex for QC-01
- `test_docs_sync_guard`: PASS (no flow_doc_facts count change)

## Acceptance (Milan UI re-run draft_000448)

After deleting stale `detrended_aligned/`, `platesolve/`, photometry outputs:

- `qc_metrics.csv`: 150 rows (130 ok + 20 `rejected_prefilter_fwhm`)
- Infolog: `skip_processed: using 130/150 lights FITS (qc_metrics.csv allowlist)`
- `detrended_aligned/lights/NoFilter_60_2`: 130 light FITS (+ MASTERSTAR)
- Phase 2A: 130 proc CSVs; QC-01 silent

## STOP before push

Milan authorizes push.

## Files changed (this arc)

- `src_py/pipeline.py`
- `src_py/invariants_runtime.py`
- `src_py/app.py`
- `src_py/night_run.py`
- `dev/tests/test_skipproc_qc_allowlist.py` (new)
- `dev/tests/test_invariants_p2.py`
- `docs/VYVAR_INVARIANTS.md`
- `docs/VYVAR_DECISIONS.md`
- `docs/VYVAR_STATE.md`
