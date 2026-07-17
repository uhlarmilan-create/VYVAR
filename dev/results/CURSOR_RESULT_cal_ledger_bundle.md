CURSOR RESULT -- 2026-07-14

What I did
Pushed CAL-DIAG closeout (Part 0). Verified and closed three CAL-DIAG spec section-10
ledger items: centralized mtime-fallback WARN in shared age helper (Part 1), confirmed RN
header passthrough + added missing-header test (Part 2), confirmed dead passthrough removal
(Part 3). Fixed ROADMAP EXCEPT-BULK-2 stale row and marked section-10 RESOLVED (Part 4).

## Output / findings

### Part 0 -- Push
- Pushed `b08f1cc..237dd34` to `origin/main`.
- `session_baseline_check.py --fast` PASS on pushed HEAD **237dd34** (830 passed, 15 skipped).

### Part 1 -- CAL-AGE-CLOCK
- Shared helper: `calibration.resolve_master_age` (header `VY_CDATE` -> `DATE-OBS` -> `DATEOBS`;
  naive -> UTC; mtime fallback with one-time WARN via `reset_master_age_mtime_warnings`).
- Consumers: `importer._age_days`, `get_calibration_status`, `ui_calibration_library.get_master_age_days`.
- Original unify: `5143485`, `ee89de8`. Bundle warn centralization: **`4d00d27`**.
- Tests: `tests/test_cal_age_clock.py` (10 tests): header vs mtime, copy scenario, timezone,
  boundary 90/200 d, UI warn path.
- **Live library verdict changes (report-only):** `CalibrationLibrary` -- 6 FITS paths
  (3 unique masters). **0 validity verdict changes** under unified clock.
  All have `VY_CDATE` 2026-04-22; header age ~82.9 d, mtime ~82.6 d; valid at 90/200 d limits.
- **Calibration selection:** unified clock affects master rejection only when header age and
  mtime disagree across expiry boundary (copy scenario). Current library: no flips; no master
  selection change on existing data.

| Master | kind | age_header | age_mtime | valid (unified) | valid (mtime-only) |
|--------|------|------------|-----------|-----------------|---------------------|
| Dark_120s_..._20260422.fits | dark | 82.9 d | 82.6 d | ok | ok |
| Dark_60s_..._20260422.fits | dark | 82.9 d | 82.6 d | ok | ok |
| Flat_0.15s_..._20260422.fits | flat | 82.9 d | 82.6 d | ok | ok |

### Part 2 -- RN-HEADER-NONE
- Fix already on main: `1830527` -- `precompute_and_save_snr_aperture_table_for_draft` loads
  MASTERSTAR header and passes to `resolve_read_noise` (`photometry_core.py` ~1755-1768).
- Bundle test: `tests/test_snr_table_rn_header.py` (2 tests); commit **`adb3661`**.
- **Impact statement:** Persisted output change limited to **`aperture_snr_table.json`**
  (`read_noise` field, SNR-optimal aperture planning). LC `err` and Phase 2A science columns
  **unchanged** (2026-07-08 draft_424 byte-identical Phase 2A rerun after RN fix; anchor core
  SHA `bf3743a1` unchanged).

### Part 3 -- CAL-PASSTHRU-DEAD
- Removed in **`21c20e3`**. `get_processed_master` raises `MasterResamplingError` if file missing.
- **grep evidence (`.py` only):** zero matches for `allow_passthrough` in production or tests.
  Remaining mentions: audit docs only (`VYVAR_AUDIT_FINDINGS.md`, `VYVAR_FULL_AUDIT_LEDGER.md`).

### Part 4 -- Docs consistency
- Fixed ROADMAP standing row: EXCEPT-BULK-2 **Parked** -> **CLOSED (98/98, `97affe3`)**.
- Updated CAL-DIAG spec section 10, ROADMAP section-10 table, STATE, JOURNAL.
- One-pass sweep: no other standing-table rows contradicted CENSUS/STATE/JOURNAL.
- JOURNAL root-cause note: stale EXCEPT-BULK-2 row survived SPARSE-TRUST arc-close verification
  because Part 2 re-asserted pre-BULK-2 wording without CENSUS cross-check.

## Errors (if any)
None.

## Files changed
- `calibration.py` -- centralized mtime WARN; `reset_master_age_mtime_warnings`
- `importer.py` -- delegate WARN to calibration helper
- `tests/test_cal_age_clock.py` -- UI warn test; ASCII cleanup
- `tests/test_snr_table_rn_header.py` -- header-unavailable regression test
- `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_CAL_DIAG_SPEC.md`, `docs/VYVAR_STATE.md`, `docs/VYVAR_JOURNAL.md`
- `CURSOR_RESULT_cal_ledger_bundle.md`

Commits (local, post-Part-0, NOT PUSHED):
- `4d00d27` fix(cal): centralize master age mtime fallback warning
- `adb3661` test(phot): SNR table RN when MASTERSTAR header missing
- `b268a6c` docs: close CAL-LEDGER-BUNDLE section-10 items and ROADMAP sweep

pytest: **832 passed**, 15 skipped. ruff: clean on touched `.py` files.
