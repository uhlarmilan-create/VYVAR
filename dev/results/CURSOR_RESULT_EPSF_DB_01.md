CURSOR RESULT - 2026-08-21 (EPSF-DB-01 diagnosis)

What I did
Read-only localization of "database disk image is malformed" on draft 517 RUN ePSF.
No repair, no `src_py/` changes. Reproduced failure on live DB.

## Part 1 - localize and assess

### 1. Where it died (failure bracket)

**Code path** (`app.py:1788-1826`, tip 69f4f5e): `run_epsf` ->
`build_epsf_model(..., db=pipeline.db, draft_id=517)` ->
`_epsf_prepare_stars` (`psf_photometry.py:801+`) with default `csv_only=False`.

**Reproduced this session:**

| Step | Result |
|------|--------|
| `get_epsf_fwhm_from_context` (FITS header + manifest FWHM) | **OK** (3.301 px) |
| `SELECT COUNT(*) FROM MASTER_SOURCES WHERE DRAFT_ID=517` | **FAIL** `database disk image is malformed` |
| Full `_epsf_prepare_stars(...)` | **FAIL** same `DatabaseError` |

**Expected UI log bracket (no session infolog with traceback found on disk):**

- **Present before failure:** `[ePSF job] Building ePSF model...` (`app.py:1819`)
- **Likely present:** `PSF ePSF: FWHM=3.301 px ...` (`psf_photometry.py:842`) and CSV-filter logs
- **First missing success line:** `PSF ePSF: joined MASTER_SOURCES intersect CSV ...`
  (`psf_photometry.py:1058`) or `[ePSF job] Model built:` (`app.py:1826`)

**Failing call:** `db.conn.execute` on **MASTER_SOURCES** safe-comp query
(`psf_photometry.py:980-998`), after CSV quality filtering, before star cutout extraction.

**Gaia DR3 catalog DB:** not opened on this path. `export_per_frame_catalogs` (Gaia via
catalog match) runs only **after** model build (`app.py:1841+`) and was not reached.

**Dropped tables rule-out:** ePSF uses `fetch_draft_light_rows_for_quality` (manifest JSON,
`draft_provenance.py:510`) for FWHM fallback, not OBS_FILES/OBS_DRAFT. No "no such table"
error observed.

### 2. Per-DB verdict table

Connection mode: **`file:<abs-path>?mode=ro`** (SQLite URI read-only) unless noted.

| DB | Size | mtime | -wal/-shm | quick_check | Verdict |
|----|------|-------|-----------|-------------|---------|
| **vyvar.sqlite3** (app) | 50.3 MB | 2026-08-21 09:19 | wal 0 B, shm 32 KB | **FAIL** (~1.2 s) | **MALFORMED** |
| **GAIA_DR3/vyvar_gaia_dr3.db** | 49.5 GB | 2026-06-15 | wal 0 B, shm 32 KB | **OK** (970 s, read-only) | **OK** |
| **GAIA_DR3/gaia_dr3_local.sqlite3** | 0 B | 2026-07-21 | n/a | skipped | **Empty stub** |

**vyvar.sqlite3 integrity detail:** `PRAGMA quick_check` and `integrity_check` report
corruption in **btrees 11-14** only:

| rootpage | Object |
|----------|--------|
| 11 | `MASTER_SOURCES` table |
| 12 | `IDX_MASTER_SOURCES_DRAFT` |
| 13 | `IDX_MASTER_SOURCES_GAIA` |
| 14 | `IDX_MASTER_SOURCES_PHOTCAT` |

Errors include invalid page numbers 12884-13068, duplicate page refs, and unused pages
12612-12647. **All other app tables readable** (row counts below).

### 3. Environment facts

| Fact | Value |
|------|-------|
| Disk free (C:) | ~56 GB |
| vyvar.sqlite3 last write | **2026-08-21 09:19** (same mtime as -wal/-shm) |
| Concurrent --full | Not running during this diagnosis |
| OneDrive/Dropbox on data root | Not detected (local `C:\ASTRO\python\VYVAR`) |
| Antivirus | Not queried (Windows default unknown) |
| config.json truncation incident | Same morning (2026-08-21); same volume; plausible concurrent DB stress window |

### 4. Journal mode and connection audit

| Item | Evidence |
|------|----------|
| Journal mode | **WAL** set on every open (`database.py:54` `PRAGMA journal_mode = WAL`) |
| Busy timeout | 30 s default (`database.py:38`, `:55`) |
| Concurrent writers | UI Streamlit + `night_run` + `--full` harness all use `VyvarDatabase` |
| Open always writes | `VyvarDatabase.__init__` runs `_create_tables()` + migrations (`database.py:1155-1158`) even for read-mostly callers |
| DB copy while open | No `shutil.copy` of `vyvar.sqlite3` found in codebase; FITS/archive copies only |
| `--fast` DB access | Opens `VyvarDatabase` for manifest-db-parity (`session_baseline_check.py:295-307`) and `--full` photometry (`:625`); **writes** via init/migrations/seeding |

**Note:** `--fast` can still **OVERALL PASS** while `MASTER_SOURCES` is corrupt (parity check
does not query that table; pytest may not touch it).

### 5. Blast assessment (vyvar.sqlite3)

| Table | Rows (readable) | Rebuild source of truth |
|-------|-----------------|-------------------------|
| EQUIPMENTS | 5 | Milan hardware / UI settings; sample rows intact (QHY294MM, etc.) |
| TELESCOPE | 8 | Same |
| LOCATION | 6 | Same + config LOCATION |
| LOCATION_OLD | 6 | Legacy mirror |
| CALIBRATION_LIBRARY | 3 | `CalibrationLibrary/` FITS paths on disk |
| FITS_HEADER_CACHE | 16102 | **Cache** - rebuild on demand |
| OBS_QC_PROCESSING_* | 15 / 1414 | QC history; not needed for ePSF if manifest QC exists |
| FIELD_REGISTRY | 0 | Empty |
| COMP_STAR_LIBRARY | 0 | Empty |
| **MASTER_SOURCES** | **UNREADABLE** | **Per-draft MAKE MASTERSTAR** repopulates via `replace_master_sources_for_draft` (`database.py:1419+`, `pipeline.py` MASTER_SOURCES writer) |

**State with no other copy:** only **MASTER_SOURCES** (+ its indexes) is lost inside the
file. Draft science data (manifests, MASTERSTAR.fits, CSVs, proc CSVs) live on disk under
`Archive/Drafts/`. **Full app DB rebuild from disk is feasible** for equipment/location/cal
library; **MASTER_SOURCES alone can be repopulated** by re-running MAKE MASTERSTAR on affected
drafts (517 at minimum for ePSF).

### 6. Backup inventory

| File | Size | Date | Notes |
|------|------|------|-------|
| **vyvar.sqlite3.bak** | **NOT FOUND** | - | Task-advised backup absent on disk |
| `GAIA_DR3/zaloha/vyvar_gaia_dr3.db` | 10.0 GB | 2026-04-28 | Older **Gaia** catalog backup, not app DB |
| `dev/sandbox/scripts/vyvar.sqlite3` | 184 KB | 2026-05-22 | Sandbox fixture, not production |
| `tmp/fk_test*/vyvar.sqlite3` | 128 KB | 2026-07-23 | Test fixtures |

---

## Part 2 - repair options (NOT executed)

### A. `.recover` into fresh DB (copy workflow)

- Copy malformed file -> run `sqlite3 .recover` -> new DB -> row-count diff vs readable tables.
- **Risk:** low for non-MASTER tables; MASTER_SOURCES recovery uncertain if btree is structurally bad.
- **Verify:** per-table counts, MASTER_SOURCES query for draft 517, `_epsf_prepare_stars` smoke, `--fast`, UI load draft 517.
- **Time:** ~1-2 h including diff and spot checks.

### B. Rebuild from disk sources (preferred if coverage confirmed)

- **Surgical:** DROP/recreate **MASTER_SOURCES** + indexes only (corruption isolated to rootpages 11-14); repopulate draft 517 via MAKE MASTERSTAR (no full DB replace).
- **Full:** New empty schema + re-insert EQUIPMENTS/TELESCOPE/LOCATION/CALIBRATION_LIBRARY from known config + replay MASTERSTAR for drafts needing MASTER_SOURCES.
- **Risk:** lowest for science data (manifests untouched); MASTER_SOURCES for other drafts lost until each draft is re-run.
- **Verify:** same as A.
- **Time:** surgical ~30-60 min; full rebuild ~2-4 h.

### C. Restore from backup + delta

- **Blocked:** no production `vyvar.sqlite3.bak` found.
- Gaia zaloha backup is unrelated to app DB.

---

## Part 3 - ePSF arc preflight (draft 517)

| Prerequisite | Status |
|--------------|--------|
| `platesolve/NoFilter_60_2/MASTERSTAR.fits` | **Present** (11.6 MB) |
| `masterstars_full_match.csv` | **Present** (1.3 MB) |
| Aligned frames / proc CSVs | **134** proc CSVs under `detrended_aligned/lights/NoFilter_60_2` |
| `platesolve/NoFilter_60_2/` | **Present** |
| `masterstar_epsf.fits` | **Absent** (expected) |

**Config state (no changes):**

| Key | Value |
|-----|-------|
| `psf_photometry_enabled` | **false** |
| `photometry_mode` | **both** |
| `psf_spatial_enabled` | **false** |
| `psf_spatial_order` | **0** |
| `epsf_min_stars` | **30** |
| oversampling default in code | **2** (`build_epsf_model` default) |

**ePSF-VALID-01 governing spec (DECISIONS 2026-08-20 readback):** ePSF validation measures
**science set only** (targets + comps + checks + BLENDED sample), never all stars; PSF-star
selection composes existing quality gates; acceptance uses a **split-half self-test certificate**
(not fixed star-count thresholds); spatial/time mode (fixed / FWHM-scaled / per-frame) chosen by
measurement per setup; **mono cameras only** (OSC deferred).

---

## Docs impact

None (diagnosis only).

## Errors

None during diagnosis.

## Files changed

- `dev/results/CURSOR_RESULT_EPSF_DB_01.md` (this file)
- `dev/results/context/session_20260821_epsf_db_01/summary.json`
- `dev/sandbox/epsf_db_01_check.py` (read-only checker; gitignored sandbox)

**STOP** - repair route is Milan's + architect's. No DB-writing jobs until green.

Operator constraint remains: no RUN VYVAR / MASTERSTAR / ePSF until repair; AAVSO/VarAstro
uploads (file-based) unaffected.
