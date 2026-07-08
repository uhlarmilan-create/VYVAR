CURSOR RESULT ù 2026-07-08 (CAL-AGE-CLOCK)

What I did
Unified master validity on header capture date: import scan, `get_calibration_status`, and
library UI now share `resolve_master_age`. Audited header key chain; scanned real
CalibrationLibrary for validity flips; tests + full gate green; pushed.

## Output / findings

### Header age chain (`calibration.resolve_master_age`)
| Priority | Key | Parse |
|----------|-----|-------|
| 1 | `VY_CDATE` | ISO-8601 (`Z`?`+00:00`, space?`T`); `%Y%m%d` / `%Y-%m-%d` fallback |
| 2 | `DATE-OBS` | same |
| 3 | `DATEOBS` | same |
| TZ | naive ? **UTC** | `dt.replace(tzinfo=timezone.utc)` |
| Fallback | filesystem mtime | one warning per file per scan (never silent) |

`VY_CDATE` stamped at master write (`importer.py:1066`).

### Consumers unified (was 3 clocks)
| Consumer | Before | After |
|----------|--------|-------|
| Import scan `_age_days` | mtime only | `resolve_master_age` |
| `get_calibration_status` | mtime only | `resolve_master_age` |
| Library UI `get_master_age_days` | header + mtime (already) | unchanged API, shared core |

No other `validity_days` / master-age consumers found beyond these + config defaults.

### Boundary rule (matches UI `_status_for_age`)
- **Valid:** `age <= validity_days` (90 dark / 200 flat)
- **Expired:** `age > validity_days` (strictly greater; age == limit is OK)

### Tests (`tests/test_cal_age_clock.py`, 9 passed)
- Header old + mtime fresh ? rejected (copy scenario) ?
- Header fresh + mtime old ? accepted ?
- No header date ? mtime fallback + warning ?
- Boundary inclusive / just-over ?

### Local CalibrationLibrary scan
Path: `CalibrationLibrary/` ù **3 masters, 0 validity flips**. No STOP (no in-use master
flips to EXPIRED).

| File | kind | header_key | age_header | age_mtime | limit | valid_old | valid_new | flipped |
|------|------|------------|------------|-----------|-------|-----------|-----------|---------|
| Dark_120s_Dark_0G_-10deg_Bin1_20260422.fits | dark | VY_CDATE | 76.5 d | 76.2 d | 90 | ? | ? | no |
| Dark_60s_Dark_0G_-10deg_Bin1_20260422.fits | dark | VY_CDATE | 76.5 d | 76.2 d | 90 | ? | ? | no |
| Flat_0.15s_NoFilter_0G_-10.5deg_Bin1_20260422.fits | flat | VY_CDATE | 76.5 d | 76.2 d | 200 | ? | ? | no |

Evidence: `tmp/cal_age_clock/library_scan.json`

### Gate
`577 passed`, 15 skipped; ruff BLE001/E722 ù **PASS**

## Errors (if any)
None.

## Files changed
| Commit | Files |
|--------|-------|
| `5143485` | `calibration.py`, `importer.py`, `tests/test_cal_age_clock.py` |
| `11e8e50` | `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_STATE.md`, `docs/VYVAR_JOURNAL.md`, `CURSOR_RESULT_cal_age_clock.md` |
