CURSOR RESULT - 2026-07-15 13:50 UTC+2

What I did
Implemented F-428 fix batch (B1-B7): VSX path resolution, repair-log aggregation, HRD TAP retry,
ePSF skip dedup, UTC infolog, AC/excluded visibility; A1-A3 verification; unit tests; diagnostic script.

## Part A - Verification

### A1 F-428-VSXFLAG (CONFIRMED)
- `Archive/Drafts/draft_000428/platesolve/NoFilter_60_2/variable_targets.csv` - **EXISTS**
- `.../photometry/variable_targets.csv` - **MISSING**
- Pre-fix `variability_candidates.csv`: 8 rows, all `vsx_known_variable=False`, `vsx_match=False`
  (known VSX in `active_targets.csv`: BO CVn, SS CVn, FZ CVn, FY CVn, FU CVn, RX CVn, NSVS 5096293, CSS_J140918.7+423422).
- **Affected entry points:** UI (`ui_aperture_photometry.py` comp CSV under `photometry/`) **and** headless
  (`run_phase2a` / `pipeline.py:6283` writes `variable_targets.csv` in platesolve setup dir).
- **Fix:** `resolve_variable_targets_csv()` probes parent + grandparent + explicit paths; logs INFO/WARNING.

### A2 Gaia vari flag for RRAB (EXPECTED - no fix)
- Local Gaia DB (`vyvar_gaia_dr3.db`) **does not** include `phot_variable_flag` column.
- Sample columns: `source_id, ra, dec, g_mag, bp_mag, rp_mag, bp_rp, ... teff_gspphot, logg_gspphot`.
- `gaia_dr3_variable_catalog=False` for RX CVn / SS CVn on draft_428 is **expected** for G<=16 zaloha DB.

### A3 Masterstar<->Gaia separation diagnostic (READ-ONLY)
Script: `scripts/diag_428_unmatched_sep.py`  
Output: `tmp/f428_unmatched_sep.txt`

| Metric | Value |
|--------|-------|
| Unmatched DET_* rows | 2724 / 6699 |
| FWHM @ bin2 | 6.234 arcsec (2.3976 px x 2.6 arcsec/px) |
| Nearest-Gaia sep p50 | 80.973 arcsec |
| p90 / p95 / max | 132.478 arcsec / 149.915 arcsec / 198.664 arcsec |
| Within 1x / 1.5x / 2x FWHM | 0 / 4 / 5 of 2724 |

**Radius decision:** OPEN (Milan) - diagnostic input only; no radius change applied.

Excluded VSX targets (not in `photometry/active_targets.csv`): **163** rows listed in diagnostic output
(includes 17 `no_dao_gaia_match` class from Phase 0 census; remainder Gaia-placeholder / out-of-frame cases).

## Part B - Fixes

| ID | Status | Summary |
|----|--------|---------|
| F-428-VSXFLAG | **FIXED** | `resolve_variable_targets_csv()` + EXC-0575 logging fix in `variability_detector.py` |
| F-428-REPAIR-FLOOD | **FIXED** | `skip_unmatched_placeholders=True` at pipeline call; aggregated `REPAIR summary:` line; `box_deg` param |
| F-428-TAP | **FIXED** | 3x TAP attempts (5s/15s backoff); `hrd_enrich_tap_timeout_s` in AppConfig/config.json/PARAMS; `_hrd_cache/summary.json` skip metadata; PDF one-line note |
| F-428-EPSF-LOGNOISE | **FIXED** | Once-per-run INFO log with path (`_EPSF_SKIP_LOGGED` set) |
| F-428-LOGTZ | **FIXED** | `Formatter.converter = time.gmtime`; infolog header `# timestamps: UTC` |
| F-428-AC-VISIBILITY | **FIXED** | `ac_applied`, `ac_skip_reason`, optional `ac_delta_m_corr`/`ac_scatter`/`ac_n_ref`; `[AC] run summary:` |
| F-428-EXCLUDED-VISIBILITY | **FIXED** | INFO name list + `excluded_targets.csv` sidecar (`out_of_frame`, `no_catalog_id`, `saturated`, `no_dao_gaia_match`) |

## Part C - Validation

### pytest
**859 passed**, 16 skipped (includes archive-missing skips). New: `tests/test_f428_fixbatch.py` (8 tests).

### Byte-identity / re-export on draft_428
**NOT RUN** - draft_428 on disk lacks `proc_*.csv` per-frame inputs (summary/LC artifacts only post partial export).
Variability re-export attempted -> `FileNotFoundError: No proc_*.csv`.
**Expected diffs when re-run with full proc tree:** `variability_candidates.csv` VSX flags + candidate mask;
`photometry_summary.csv` AC columns; new `excluded_targets.csv`. Proc/LC science columns unchanged by design.

### Candidates re-check
Pre-fix evidence: 8 candidates, all VSX false flags. Post-fix live re-export **pending** full proc inputs.

### PDF
Not regenerated (no full draft re-run). HRD enrichment note path implemented in `photometry_report.py`.

### Provenance (read-only, `pipeline_meta.json`)
```json
"provenance": {
  "git_hash": "31ab762a758aa79f87ba4d363555a0b327aa42df",
  "git_dirty": true,
  "entry_point": "run_phase2a",
  "stamped_at_utc": "2026-07-15T10:37:10.719825+00:00"
}
```
First live PROV-HEADLESS stamp from UI->Phase2A path verified.

## Files changed
- `photometry_core.py` - VSX resolver, AC summary columns, excluded sidecar
- `variability_detector.py` - VSX load WARNING
- `scripts/repair_catalog_ids.py` - placeholder skip + summary
- `pipeline.py` - repair flag, ePSF dedup
- `hrd_enrich.py`, `hrd_analysis.py`, `photometry_report.py`
- `config.py`, `config.json`, `docs/VYVAR_PARAMS.md`
- `infolog.py`
- `scripts/diag_428_unmatched_sep.py`
- `tests/test_f428_fixbatch.py`
- `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_STATE.md`, `CHANGELOG.md`

## Errors (if any)
None in pytest. Draft_428 byte-identity blocked by missing proc CSV inputs.
