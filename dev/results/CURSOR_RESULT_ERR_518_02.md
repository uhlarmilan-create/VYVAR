CURSOR RESULT - 2026-08-21 (ERR-518-02 + ERR-518-02b)

What I did
Implemented Milan-approved ERR-518-01 options 1+2 in two commits; verified on
516 anchor (byte-identical core SHA) and draft 518 re-export (100% empirical).
ERR-518-02b: documented the non-zero `--full` incident, re-ran clean `--full`
(OVERALL PASS exit 0), and recorded same-frame sigma spread.

## Summary

**Commit 1 (`70a6022`):** `_sigma_bkg_r_key()` + INV-ERR-SIGMA-ACCT-01 in
`enhance_catalog_dataframe_aperture_bpm`; unit tests.

**Commit 2 (`30d631d`):** `_finalize_hybrid_bkg_fallback_sidecar()` in
`pipeline.py`; post-flush finalize on RAM handoff; RAM finalize tests.

**Commit 3 (`ea0b4ad`):** JOURNAL, INVARIANTS, DECISIONS; this report.

**Commit 4 (`5f46469`, 02b wiring gap):** add `INV-ERR-SIGMA-ACCT-01` to
`WIRED_INV_IDS` in `invariants_runtime.py` (registry parity test blocked
clean `--full` at `ea0b4ad`).

## Commit 1 - key function + accounting invariant

### Code (`src_py/photometry_core.py`)

| Item | Location |
|------|----------|
| `_sigma_bkg_r_key(r)` | ~81-90 |
| `_assert_inv_err_sigma_acct_01(...)` | ~93-128 |
| Store via `_sigma_bkg_r_key(_r_u)` | ~14340-14342 |
| Lookup via `_sigma_bkg_r_key(...)` | ~14357-14367 |
| Invariant call after assignment | ~14371-14377 |

Exception type: `InvariantViolation` from `invariants_runtime` (same family as
INV-DAG-01 / INV-MS-EXPAND-01).

### Grep proof (zero direct float keys)

```
$ rg "_sigma_by_r\[" src_py/photometry_core.py
  _sigma_by_r[_sigma_bkg_r_key(_r_u)] = (float(_sig), ERR_BKG_SOURCE_EMPIRICAL)
  _sigma_by_r[_sigma_bkg_r_key(_r_u)] = (float("nan"), ERR_BKG_SOURCE_HOWELL_FALLBACK)

$ rg "_sigma_by_r\.get" src_py/photometry_core.py
  _sig_v, _src_v = _sigma_by_r.get(_r_key, ...)   # _r_key = _sigma_bkg_r_key(...)
  _sig_v, _src_v = _sigma_by_r.get(_sigma_bkg_r_key(float(r_ap)), ...)
```

No `_sigma_by_r[float(` or bare `round(float(r_ap` lookup remains (seed RNG
still uses `round(float(r_ap)*10000)` for Labbe determinism only).

### Tests (fail-before-fix record)

| Test | Pre-fix (34f9cf1 / ERR-518-01) | Post-fix |
|------|--------------------------------|----------|
| `test_global_fixed_518_r_ap_emits_empirical_sigma` | **FAIL** - ERR-518-01 replay: enhance on draft 518 aligned FITS returned 62352/62352 `howell_fallback`, 0 finite `sigma_bkg_ap` despite Labbe success (store key 4.35461 vs lookup 4.3546) | **PASS** |
| `test_inv_err_sigma_acct_01_fires_on_desynced_keys` | N/A (invariant absent) | **PASS** |
| `test_snr_table_path_sigma_projection_unchanged` | N/A | **PASS** (identical sigma columns run-to-run) |
| `test_sigma_bkg_r_key_canonicalizes_518_global_aperture` | N/A | **PASS** |

Pre-fix evidence: `CURSOR_RESULT_ERR_518_01.md` + sandbox replay (enhance with
`gaussian_fwhm_px_override=2.2919` -> 100% fallback before fix).

## Commit 2 - finalize on RAM path

### Code (`src_py/pipeline.py`)

| Item | Location |
|------|----------|
| `_finalize_hybrid_bkg_fallback_sidecar()` | ~10633-10658 |
| Direct-write path (unchanged semantics) | ~11685-11693 (`not defer_disk_writes`) |
| RAM post-flush call | ~15842-15851 (gated on `deferred_csv_writes` non-empty) |

**Double-invocation guard:** `export_per_frame_catalogs` calls finalize only when
`not defer_disk_writes`. RAM path uses `defer_disk_writes=True`, so in-export
finalize is skipped; exactly one post-flush call runs. Direct-write never hits
post-flush block.

### Tests

| Test | Result |
|------|--------|
| `test_ram_flush_finalize_scales_fallback_rows` | PASS (1 howell_scaled row) |
| `test_ram_flush_finalize_noop_on_all_fallback` | PASS (r_setup None, rows unchanged) |

## Incident: first `--full` at `30d631d` exited non-zero (2026-08-21)

The first anchor recut during ERR-518-02 ran concurrently with in-flight docs
commit 3 staging. Science gates passed; harness exit was **1**.

| Check | Status | Detail |
|-------|--------|--------|
| git-head | PASS | `30d631d` |
| git-staged | **FAIL** | 4 staged change(s) |
| pytest | **FAIL** | `test_apply_install_config.py::test_end_to_end_write_validates_and_drops_author_paths` - `JSONDecodeError` on repo-root `config.json` |
| full-science-compare | PASS | n_lc=60 failures=0 |
| full-snapshot-sha-core | PASS | **9902d918...** n=121 |
| full-photometry-sha-core | PASS | **9902d918...** n=121 |
| OVERALL | **FAIL** | exit **1** |

### What was staged / conflicted

**Staged (4 paths, per harness):** concurrent `git add` for ERR-518-02 commit 3
while `--full` was still running - `docs/VYVAR_JOURNAL.md`,
`docs/VYVAR_INVARIANTS.md`, `docs/VYVAR_DECISIONS.md`,
`dev/results/CURSOR_RESULT_ERR_518_02.md`.

**Unmerged (`UU`) working-tree paths (same incident window):**
`dev/validation/params_registry.json`, `docs/VYVAR_DECISIONS.md`,
`docs/VYVAR_LIMITATIONS.md`, `docs/VYVAR_PARAMS.md` - overlap between docs
sync edits and the in-flight commit, not caused by ERR-518 code changes.

### Broken `config.json` (repo root)

**File:** `config.json` at repository root (`c:\ASTRO\python\VYVAR\config.json`).

**Symptom:** pytest failed with `json.decoder.JSONDecodeError: Expecting value:
line 1 column 1 (char 0)` - file truncated to **0 bytes** on disk.

**Mechanism:** recovery attempt `git checkout HEAD -- config.json` collided with
`.git/index.lock` held by the overlapping git operations above; checkout did not
atomically restore the file. Valid HEAD content is ~142572 bytes (JSON with
`//` comments, loaded by `AppConfig`).

**Did the recut consume it?** **No.** `session_baseline_check.py --full` calls
`run_full_baseline` after the initial pytest pass/fail record; the frozen 516
snapshot pipeline loads `AppConfig()` once at entry and copies snapshot inputs
into `tmp/session_baseline/-` - it does not re-read or mutate repo `config.json`
mid-run. Evidence: `full-pipeline` PASS (1972 s), core SHA **9902d918**
byte-identical, science compare 60/0. End-of-run tree dirt caused harness FAIL,
not science drift.

Recovery: remove `index.lock`, `git checkout HEAD -- config.json` (+ conflicted
docs/registry paths); subsequent `--fast` OVERALL PASS.

## Verification (ERR-518-02b)

### `--fast`

OVERALL **PASS** (1478 passed, 32 skipped) at `30d631d` and after recovery.

### Clean `--full` re-run

| Run | git-head | git-staged | pytest | core SHA | exit | OVERALL |
|-----|----------|------------|--------|----------|------|---------|
| ea0b4ad (clean tracked tree) | ea0b4ad | PASS none | **FAIL** `test_registry_lists_all_wired_ids` (INV-ERR-SIGMA-ACCT-01 in docs, missing from `WIRED_INV_IDS`) | 9902d918 n=121 (science gates PASS) | **1** | **FAIL** |
| **5f46469 (clean tracked tree)** | **5f46469** | **PASS none** | **PASS 1478** | **9902d918 n=121** | **0** | **PASS** |

Authoritative push gate (2026-08-21T08:53:41Z):

```
git-head                     PASS   5f46469
git-staged                   PASS   none
pytest                       PASS   1478 passed, 32 skipped
full-science-compare         PASS   n_lc=60 failures=0
full-snapshot-sha-core       PASS   9902d918e9f48e0f... n=121
full-photometry-sha-core     PASS   9902d918e9f48e0f... n=121
full-photometry-sha-extended PASS   472bc9e4446f13a8... n=179
OVERALL: PASS
exit_code: 0
```

Log: `dev/results/context/session_20260821_err518_02/full_baseline_clean_5f46469.log`

Validation ledger: `VL-ANCHOR-WCSINV` updated `last_verified=2026-08-21`,
`commit=5f46469`, `core_sha=9902d918-` unchanged.

### Draft 518 before/after (`dev/sandbox/err_518_02_reexport_proc.py`)

Method: re-run `enhance_catalog_dataframe_aperture_bpm` on existing aligned FITS
+ proc rows (strip sigma columns); 71 files.

| | ERR-518-01 (before) | ERR-518-02 (after) |
|---|---------------------|---------------------|
| empirical | 0 | **62352** |
| howell_fallback | 62352 | 0 |
| howell_scaled | 0 | 0 |

First-frame `sigma_bkg_ap`: **51.18 ADU**. Catalog_id `1624628764771224960`:
**INV-ERR-MODE-01 preflight sample PASS**.

### Same-frame sigma check (report only)

**Yes - same frame** `TOI-1131.01.b_2025-04-22_23-05-09_V.fits`: production
proc CSV **51.18 ADU** (content-derived Labbe seed `980296111500101067`) vs
ERR-518-01 sandbox **71.80 ADU** (`seed=42`); spread **~20.6 ADU** - expected
mask/seed sensitivity of the empty-aperture estimator, not a frame mismatch.

### SNR table rejection (report only)

`Archive/Drafts/draft_000518/aperture_snr_table_REJECTED.json`:
`impl02_gates.ok=false`, failures:
**INV-COG-MONOTONE**, **INV-COG-R90**, **INV-APERTURE-BOUND** (`detail`:
"bound binds" / bright mags hit r_max). No accepted `aperture_snr_table.json` ->
export uses **global_fixed** (trigger path for ERR-518-01).

## Docs impact (DOCS-SYNC)

| File | Change |
|------|--------|
| `docs/VYVAR_JOURNAL.md` | ERR-518-02 entry |
| `docs/VYVAR_INVARIANTS.md` | INV-ERR-SIGMA-ACCT-01 row |
| `docs/VYVAR_DECISIONS.md` | ERR-518-02; options 3/5 rejected |
| `docs/VYVAR_ROADMAP.md` | PRECAL-INPUT-CONTRACT-01 (from ERR-518-01, unchanged) |

## Files changed

- `src_py/photometry_core.py` (commit 1)
- `src_py/pipeline.py` (commit 2)
- `src_py/invariants_runtime.py` (commit 4, WIRED_INV_IDS)
- `dev/tests/test_err_background_empirical.py` (commits 1-2)
- `dev/sandbox/err_518_02_reexport_proc.py`
- `dev/results/context/session_20260821_err518_02/reexport_counts.json`
- `dev/validation/VYVAR_VALIDATION_LEDGER.json` (auto-updated on clean `--full` PASS)
- Docs (commit 3)

## ERR-518 series close-out (commit 5, Part 0)

Commit 5 (docs/evidence only): amended this report, ledger stamp
(`VL-ANCHOR-WCSINV` @ `5f46469`), `VYVAR_PROCESS.md` no-git-during-`--full`
rule, `VYVAR_ROADMAP.md` SIGMA-BKG-VAR-01 one-liner.

**Series range:** `70a6022..HEAD` (5 commits: `70a6022` c1, `30d631d` c2,
`ea0b4ad` c3, `5f46469` c4, c5 = this commit, message ERR-518-02b evidence).

STOP - Milan pushes the series (`70a6022..HEAD`). Part A (DOCS-SYNC-517)
waits until `origin/main` includes c5 (`git log -1 --oneline` matches ERR-518-02b
evidence message).
