CURSOR RESULT - 2026-08-21 (ERR-518-02)

What I did
Implemented Milan-approved ERR-518-01 options 1+2 in two commits; verified on
516 anchor (byte-identical core SHA) and draft 518 re-export (100% empirical).

## Summary

**Commit 1 (`70a6022`):** `_sigma_bkg_r_key()` + INV-ERR-SIGMA-ACCT-01 in
`enhance_catalog_dataframe_aperture_bpm`; unit tests.

**Commit 2 (`30d631d`):** `_finalize_hybrid_bkg_fallback_sidecar()` in
`pipeline.py`; post-flush finalize on RAM handoff; RAM finalize tests.

**Docs (`pending commit 3`):** JOURNAL, INVARIANTS, DECISIONS; this report.

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

## Verification

### `--fast`

OVERALL **PASS** (1478 passed, 32 skipped) at tip `30d631d`.

### Anchor recut (516 snapshot, `--full`)

| Check | Result |
|-------|--------|
| full-science-compare | PASS n_lc=60 failures=0 |
| full-snapshot-sha-core | PASS **9902d918e9f48e0f...** n=121 |
| full-photometry-sha-core | PASS **9902d918e9f48e0f...** n=121 (byte-identical) |
| full-photometry-sha-extended | PASS 472bc9e4... n=179 |

Byte-identity **confirmed** on 516 (snr_table path unchanged).

### Draft 518 before/after (`dev/sandbox/err_518_02_reexport_proc.py`)

Method: re-run `enhance_catalog_dataframe_aperture_bpm` on existing aligned FITS
+ proc rows (strip sigma columns); 71 files.

| | ERR-518-01 (before) | ERR-518-02 (after) |
|---|---------------------|---------------------|
| empirical | 0 | **62352** |
| howell_fallback | 62352 | 0 |
| howell_scaled | 0 | 0 |

First-frame `sigma_bkg_ap`: **51.18 ADU** (order of ERR-518-01 sandbox 71.8;
production seeds/mask differ). Catalog_id `1624628764771224960`:
**INV-ERR-MODE-01 preflight sample PASS**.

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
- `dev/tests/test_err_background_empirical.py` (both commits)
- `dev/sandbox/err_518_02_reexport_proc.py`
- `dev/results/context/session_20260821_err518_02/reexport_counts.json`
- Docs (commit 3)

STOP - Milan reviews and pushes.
