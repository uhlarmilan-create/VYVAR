CURSOR RESULT — 2026-07-07 (CAL-FLOW-MAP)

What I did
Read-only trace of the calibration/master flow from source-folder selection through
calibrated light output. Verified Q1-Q8 against the live tree with file:line citations.

## Output / findings

---

## 1. End-to-end flow diagram (text)

```
[USER] Select source folder + equipment/telescope IDs
    |
    v
app.py: smart_scan_source() ........................ importer.py:1130
    |  Recursive rglob + IMAGETYP classify (light/dark/flat)
    |  Per obs_group (FILTER|EXPTIME|BINNING): library master lookup
    |  Validity: mtime age vs masterdark_validity_days (90) / masterflat_validity_days (200)
    v
SmartImportPlan (dark_master, masterflat_by_obs_key, dark_master_by_obs_key, ...)
    |
    v
app.py: smart_import_session() .................... importer.py:1620
    |  IF quick_look (no library dark AND no raw darks on source):
    |      -> archive/.../non_calibrated/lights only (no calibration)
    |  ELSE:
    |      Copy lights -> Raw/lights (or non_calibrated/lights if flat missing)
    |      Copy session darks/flats -> Raw/darks, Raw/flats (IMAGETYP-sorted; NOT auto-stacked)
    |      Store library master PATHS in draft calib_path (not built from session raw)
    v
app.py: pipeline.quick_calibrate_last_import() .... pipeline.py:16813
    |
    v
calibrate_lights_to_calibrated() .................... pipeline.py:14720
    |  IF no usable dark path -> passthrough copy (VY_CALIB=PASSTHROUGH)
    |  ELSE for each light FITS under Raw/lights or non_calibrated/lights:
    v
_calibrate_one_light_disk() ......................... pipeline.py:14424
    -> _calibrate_one_light_apply_masters_in_ram() ... pipeline.py:14229
        |  Load light; read XBINNING -> light_bx
        |  get_processed_master(dark) ............... calibration.py:414, pipeline.py:14286
        |      master_binning = cfg native (default 1) OR header XBINNING if null
        |      resample: dark=block SUM, flat=block MEAN; shape match / upscale
        |  data = light - dark_resampled
        |  get_processed_master(flat) ............... pipeline.py:14330
        |      normalize_flat_master if VYFLNRD=1 ..... calibration.py:487-498
        |  data = (light - dark) / flat_norm
        |  Optional post-cal QC (HFR/stars/sky) ..... pipeline.py:14472-14494
    v
Write calibrated/lights/<obs_group>/<file>.fits
    |
    v
[DOWNSTREAM] Analyze QC, platesolve, photometry (gain/RN via param_resolver)
```

**Branch summary (library vs session raw):**

| Branch | Trigger | Master used at calibrate time | Auto-build from session raw? |
|--------|---------|------------------------------|------------------------------|
| Library masters | Library match + valid age; stored in plan.*_by_obs_key | CalibrationLibrary FITS path | No |
| Session raw present | IMAGETYP dark/flat in recursive scan | Library path still filled if match (importer.py:1438-1439); raw copied to Raw/darks, Raw/flats but not stacked | No (manual via ui_calibration_library.py:366,398) |
| Quick Look | No library dark AND no raw darks (importer.py:1455-1457) | None — non_calibrated only | No |
| Passthrough | No usable dark at calibrate (pipeline.py:14786-14798) | Copy raw with VY_CALIB=PASSTHROUGH | No |

---

## 2. File:line evidence table (by step)

| Step | Function / artifact | File:line |
|------|---------------------|-----------|
| User scan entry | `smart_scan_source` | importer.py:1130-1480 |
| IMAGETYP recursive classify | `_collect_fits_by_type` | importer.py:500-557 |
| Light IMAGETYP tokens | `_LIGHT_IMAGETYP`, `_classify_imagetyp` | importer.py:407-426 |
| Obs group key | `observation_group_key` usage in scan | importer.py:1250,1272 |
| Library dark lookup per group | `_find_matching_master_in_library` | importer.py:1298-1322 |
| Library flat lookup + validity | `_find_best_masterflat_for_filter` | importer.py:1279-1294,882-987 |
| Dark/flat scan priority | `_scan_cal` | importer.py:1364-1420 |
| Library path backfill when raw on source | `dark_master = str(dark_found)` | importer.py:1438-1439 |
| Import DB write | `smart_import_session` | importer.py:1620-1811 |
| Raw archive layout | `Raw/lights`, `Raw/darks`, `Raw/flats` | importer.py:1727-1784 |
| Calibrate entry | `quick_calibrate_last_import` | pipeline.py:16813-16859 |
| Calibrate loop | `calibrate_lights_to_calibrated` | pipeline.py:14720-14942 |
| Per-light RAM calibrate | `_calibrate_one_light_apply_masters_in_ram` | pipeline.py:14229-14368 |
| Master resample core | `get_processed_master` | calibration.py:414-510 |
| Block SUM/MEAN resample | `resample_master_to_light_binning` | calibration.py:199-255 |
| Flat norm after resample | `normalize_flat_master` | calibration.py:309-411,487-498 |
| Master stack (library build) | `_generate_master_dark` / `_generate_master_flat` | importer.py:1998-2023,1879-1909 |
| Library write + register | `_write_master_to_library`, `_register_master_path_in_calibration_library` | importer.py:990-1087,317-354 |
| DB table | `CALIBRATION_LIBRARY` | database.py:1969-1980,2100-2115 |
| DB scoped lookup | `find_best_calibration_library_path` | database.py:2170-2312 |
| Validity defaults | `masterdark_validity_days=90`, `masterflat_validity_days=200` | config.py:101-102,653-654; database.py:2683-2687 |
| Gain/RN resolve | `resolve_gain`, `resolve_read_noise` | param_resolver.py:462-506 |
| Bin scaling formula | `_scale_bin1_to_binning` | param_resolver.py:154-159 |

---

## 3. Q1 — Library data model

**Registration table:** `CALIBRATION_LIBRARY` with columns
`ID, KIND, FILE_PATH (UNIQUE), XBINNING, EXPTIME, CCD_TEMP, FILTER_NAME, GAIN, NCOMBINE,
REGISTERED_AT, ID_EQUIPMENTS, ID_TELESCOPE` (database.py:1969-1980,1994-1997).

**Insert/update:** `register_calibration_library_entry` (database.py:2044-2132) called from
`_register_master_path_in_calibration_library` (importer.py:317-354). Registration requires
finite `ID_EQUIPMENTS` and `ID_TELESCOPE`; dark rows require finite `CCD_TEMP`
(database.py:2071-2089, importer.py:329-336).

**Equipment linkage:** Scoped matching requires both IDs equal
(`calibration_scopes_match`, database.py:2001-2012; SQL `ID_EQUIPMENTS = ? AND ID_TELESCOPE = ?`,
database.py:2245-2246,2257-2258). Legacy NULL,NULL rows are excluded from scoped lookup
(database.py:2186-2188).

**Validity windows:** Defaults **90 d dark / 200 d flat**
(config.py:101-102; database.py:2683-2687; tests/test_master_validity_days_g6_f002.py:14-15).

**Enforcement points (two different age clocks):**

| Mechanism | Age source | Where enforced |
|-----------|------------|----------------|
| Import scan / `get_calibration_status` | Filesystem **mtime** | importer.py:873-879,1404-1411,2026-2103 (`age_days > validity_days`) |
| CalibrationLibrary UI dashboard | Header `VY_CDATE`/`DATE-OBS` then mtime | `get_master_age_days` calibration.py:79-98; ui_calibration_library.py:96 |

**Note:** `get_master_age_days` (calibration.py:79) is **not** called from `smart_scan_source`;
import validity uses `_age_days` on mtime (importer.py:873-879,1313-1318).

**VY_MBLIB operationally:** Set to `1` on every library master write with comment
"native stack in CalibrationLibrary; resample to light XBINNING at calibrate"
(importer.py:1049-1052). Means: FITS stores native calibration-frame binning (typically 1x1);
software resample to light binning happens only at calibrate time (importer.py:1010-1011,
calibration.py:5-17).

---

## 4. Q2 — Library master selection

**Primary path:** `_find_matching_master_in_library` (importer.py:741-870) delegates to
`find_best_calibration_library_path` when `db` is set (importer.py:776-787), else filesystem scan.

**Keys that must match (DB query, database.py:2235-2258):**

| Parameter | Dark | Flat | Tolerance |
|-----------|------|------|-----------|
| Equipment scope | ID_EQUIPMENTS + ID_TELESCOPE required | same | exact |
| XBINNING | exact, or prefer XBINNING=1 when light>1 | same | exact (with 1x1 preference, database.py:2292-2296) |
| EXPTIME | exact | **not gated** | exact (dark only) |
| GAIN | exact (`COALESCE(GAIN,0)`) | exact | exact integer |
| CCD_TEMP | required, `ABS(CCD_TEMP - light) <= tol` | not used | default **0.5 C** (importer.py:754,1139; database.py:2179,2244) |
| FILTER | empty | normalized filter name | exact string |

**Filesystem fallback** (importer.py:841-858): dark — exact exposure, temp within `temp_tolerance`,
exact gain, binning exact or (light bin>1 and master bin==1); flat — exact normalized filter,
exact gain, same binning rule.

**Missing scope / temp:** Returns `None` with log; no fallback master (importer.py:756-772,
database.py:2211-2220).

**No match behavior:**

- Dark: `dark_missing=True` if `_scan_cal` finds no raw and no library ? **Quick Look**
  (importer.py:1398-1402,1455-1457).
- Flat: `missing_obs_keys` ? those lights go to `non_calibrated` (importer.py:1295-1296,1752-1754).
- Flat expired in `_find_best_masterflat_for_filter`: returns `(None, warning)` (importer.py:928-931).
- **No silent wrong-equipment fallback** (scoped-only model, database.py:2186-2188).

**`_find_existing_master_for_raw_set`** (importer.py:1090-1127): same matcher via sample file metadata;
used before building duplicate masters.

**Scoped-filename conflict rule** (`_master_path_scope_conflicts`, importer.py:287-301):
If target filename exists and DB row belongs to another equipment/telescope set, write under
`_scoped_master_filename` (`..._eq{id}_tel{id}.fits`, importer.py:304-314,1027-1035).

---

## 5. Q3 — Branch decision: build-new vs use-library

**Detection rules (NOT folder-name driven for import scan):**

- `smart_scan_source` uses `_collect_fits_by_type(root, db)` — **recursive rglob**, classify each
  FITS by `IMAGETYP`/`FRAME`/`IMTYPE` (importer.py:1177-1180,500-557).
- `_find_lights_subdirectory` (importer.py:429-436) and `_resolve_session_lights` (importer.py:473-497)
  exist but are **not called** by `smart_scan_source` or `smart_import_session` (dead for import path).

**Priority in `_scan_cal`** (importer.py:1365-1374):

1. Raw calibration frames on source (any_raw in first 3 files) ? status `"raw"`, master path `None`
2. Else master files on source ? use that path
3. Else library match ? `"library"` or `"expired"`
4. Else ? `"missing"`

**When session HAS raw darks/flats AND library HAS valid master:**

- Scan row shows `"raw"` for darks/flats (importer.py:1369-1373).
- Raw files **copied** to archive `Raw/darks`, `Raw/flats` (importer.py:1771-1784).
- **`plan.dark_master` still set from library** if `dark_found` (importer.py:1438-1439).
- Per-group maps `dark_master_by_obs_key` / `masterflat_by_obs_key` come from **library only**
  (importer.py:1279-1322) — session raw is not stacked into masters during import.

**Auto-build from session:** **Not implemented on import path.** Masters are built only via
`generate_master_dark_from_source_dir` / `generate_master_flat_from_source_dir`
(importer.py:1483-1608), invoked from CalibrationLibrary UI (ui_calibration_library.py:366,398).
Those call `_write_master_to_library` + `_register_master_path_in_calibration_library`
(importer.py:1534-1542,1598-1606,1079-1086).

**Reuse next time:** `_find_existing_master_for_raw_set` skips rebuild if library match exists
(importer.py:1503-1529,1567-1593). New masters are registered for reuse (importer.py:1079-1086).

**Discrepancy vs intended design:** Milan's "session folder with lights/darks/flats builds new
masters" is **not** automatic on import; only manual/UI generation + library registration.

---

## 6. Q4 — Master stacking radiometry

**Dark:** Per-pixel **mean** (`nanmean` axis 0), no sigma-clipping
(importer.py:1838-1839,1879-1892,1998-2009; header `VYSTKMOD=MEAN` importer.py:1991-1992).

**Flat:** Per-pixel **median** (`nanmedian` axis 0)
(importer.py:1841-1842,1895-1909,2012-2023; header `VYSTKMOD=MEDIAN` importer.py:1993-1994).

**Saved units:** Raw ADU at **native calibration-frame binning** (no bin-down to match lights;
`target_binning` ignored, importer.py:1012-1013). Flat **not** median-normalized at stack time;
`VYFLNRD=1` defers norm to calibrate (importer.py:1057-1061,1842-1843).

**Headers written:** `XBINNING`/`YBINNING`/`BINNING` = raw frame binning; `VY_MBLIB=1`
(importer.py:1046-1052). Data written as `float32` ADU (importer.py:1068).

---

## 7. Q5 — Calibrate-time resampling

**Confirmed in `resample_master_to_light_binning`** (calibration.py:199-255):

- Dark: `np.sum` over block (calibration.py:250-251)
- Flat: `np.mean` over block (calibration.py:252-254)
- Trailing rows/cols clipped when not multiple of block_factor (calibration.py:238-247)
- `MasterResamplingError` if light binning < master or non-integer ratio (calibration.py:222-229)

**`get_processed_master`** (calibration.py:414-510):

- `master_binning`: if passed and >0, used; else read from header (calibration.py:458-461)
- Production default: `CALIBRATION_LIBRARY_NATIVE_BINNING = 1` via `_cfg_calibration_library_native_binning`
  (calibration.py:23; pipeline.py:660-668,14810)
- `infer_spatial_block_factor`: shape-based override increases effective light binning
  (calibration.py:115-135,464-467)
- `allow_passthrough`: synthetic zeros (dark) / ones (flat) when file missing
  (calibration.py:441-452) — **no production callers pass `allow_passthrough=True`**
  (grep: only calibration.py definition; pipeline always raises on missing master via normal path)
- Flat: `normalize_flat_master` after resample when `VYFLNRD=1` (calibration.py:487-498)

**Production call sites of `get_processed_master`:**

| Site | File:line | kind | master_binning | light_shape |
|------|-----------|------|----------------|-------------|
| Dark cache helper | pipeline.py:683-688 | dark | `_native_b` from config (default 1) or None | not passed |
| Per-light calibrate (dark) | pipeline.py:14286-14293 | dark | `_mb_lib` (default 1 or None) | `data.shape` |
| Per-light calibrate (flat) | pipeline.py:14330-14338 | flat | `_mb_lib` | `data.shape` |
| Validation only | tests/validation/recover.py:675-676 | both | not set (header) | `(ny,nx)` |

---

## 8. Q6 — Radiometric checks today

**Confirmed: no calibrate-time radiometric sanity gate** comparing light vs resampled dark
medians, and no SUM-vs-MEAN dark resample cross-check.

Existing guards are **geometric/resampling**:

- Shape match / upscale / `MasterResamplingError` (calibration.py:473-483)
- Binning ratio rules (calibration.py:222-229)
- Flat pixel floor before divide (pipeline.py:14346,14358)
- Flat skipped if no dark subtracted (pipeline.py:14327,14365-14367)

**Post-calibration QC (not pre-subtraction radiometry):**

- `_post_calibration_qc_eval`: sigma-clipped sky stats, HFR, star count on **already calibrated**
  array (pipeline.py:14078-14148,14472-14494) ? headers `VYQCPASS`, `VY_QCHFR`, `VY_QCBG`, `VY_QCRMS`
- `_quality_inspection_dao_metrics_array` when `dao_qc_in_calibrate` (pipeline.py:14515-14522)
- RAM QC path in draft workflow (pipeline.py:1986) — post-cal sky/stars

**No check** that `median(light) - median(resampled_dark) > 0` or that dark SUM resample
matches expected charge scaling. docs/VYVAR_DECISIONS.md:346 and docs/VYVAR_ROADMAP.md:79
document this gap (not re-cited as code evidence).

---

## 9. Q7 — CAL-DIAG gate hook points

**Recommended minimal hook (once per master + obs_group, amortized across frames):**

1. **Primary:** `calibrate_lights_to_calibrated` ? `_one_sequential` inner loop
   (pipeline.py:14903-14942), immediately after resolving `md_use` / `_ok` obs_group key
   (pipeline.py:14912-14920) and before `_calibrate_one_light_disk`.
   Maintain a `checked: set[tuple[str,str]]` keyed by `(obs_group_key, str(md_use.resolve()))`.
   On first sighting: load one representative light header/binning, call `get_processed_master`
   for dark (reuse `_dark_np_for_calibration_path` cache at pipeline.py:671-690), compute
   `median(light)` vs `median(dark_resampled)` and optional post-subtraction sky median on a
   small crop.

2. **Secondary (shared dark load):** Extend `_dark_np_for_calibration_path` (pipeline.py:671-690)
   to run the same check once when a new cache key is created (key already encodes
   path|light_binning|master_binning).

3. **Flat norm sanity (optional, same cadence):** First flat cache miss in
   `_calibrate_one_light_apply_masters_in_ram` (pipeline.py:14329-14339) — keyed by
   `(obs_group, mf_path, light_bx)`.

**Why not inside `get_processed_master` alone:** Called per flat/dark load without obs_group
context; hook at calibrate loop preserves once-per-group semantics.

---

## 10. Q8 — Gain/RN provenance (F-BINGAIN-1 RN sub-question)

**Gain resolution** (`resolve_gain`, param_resolver.py:462-479):

1. Header e-/ADU (`EGAIN`/`GAIN`) if valid
2. Header setting index ? `GAIN_SETTING_INDEX_MAP` ? source `header_index_mapped`
   (param_resolver.py:403-412,85-87)
3. DB `EQUIPMENTS.GAIN_ADU` scaled by binning (exponent **2**)
4. Config fallback

**Read noise resolution** (`resolve_read_noise`, param_resolver.py:482-506):

1. DB `READNOISE_E` from `get_equipment_cosmic_params` (database.py:2901-2920), **scaled first**
   via `_scale_bin1_db_for_header(..., exponent=1)` (param_resolver.py:493-498)
2. Then `_resolve_equipment_intrinsic` — DB wins over header (param_resolver.py:244-257)
3. Header keys: `RDNOISE`, `READNOISE`, `RDNOISEE`, `RN` (param_resolver.py:69)

**EQUIP-BINNING scaling formula** (treats DB values as **bin1 per-pixel intrinsics**):

```python
# param_resolver.py:154-159
eff = value * (binning ** exponent)   # gain: exponent=2; RN: exponent=1
```

Example at bin 2x2: RN 1.3 -> 2.6 (tests/test_param_resolver_binning_scale.py:94-102).
For RN 7.6 -> 15.2: `7.6 * 2^1 = 15.2` (exponent=1).

**Documentation of DB semantics:** Explicit "bin1 per-pixel" comment at
param_resolver.py:155. `set_equipment_cosmic_params` docstring says "read noise [e-]" only
(database.py:2928) — does **not** state bin mode; scaling policy implies bin1 store.

**Production call sites (gain / RN):**

| Path | File:line | Header passed? |
|------|-----------|----------------|
| Per-frame error map | pipeline.py:248-253 | yes (light hdr) |
| Phase 2A photometry | photometry_core.py:6664-6665 | yes (masterstar hdr) |
| SNR aperture table helper | photometry_core.py:1210-1211 | gain: yes; RN: **None** (no bin scaling from light hdr unless DB path gets header in resolve_read_noise) |
| PSF photometry | psf_photometry.py:682-687 | yes |
| Crowding index | crowding_index.py:90-91 | RN: None |

**RN bin-scaling caveat:** `resolve_read_noise` scales DB using `header` binning when header
is provided (param_resolver.py:494-496). photometry_core.py:1211 passes `header=None` for RN —
scaling uses unscaled DB unless caller passes light header. Phase 2A passes `_ms_header` (6665).

**Decision for CAL-DIAG spec:** Code treats `READNOISE_E` as bin1 per-pixel with `RN_eff = RN_db * bin`
(param_resolver.py:155,494-496). No code comment says DB is pre-binned effective RN.

---

## 11. Discrepancies / surprises

1. **Session raw does not auto-build masters on import.** Intended "build from session darks/flats"
   is manual (`generate_master_*_from_source_dir` + UI). Import copies raw to archive and uses
   library paths when available (importer.py:1771-1784,1438-1439).

2. **When both session raw AND library match exist, library path wins for calibration** despite
   scan row status `"raw"` (importer.py:1369-1373 vs 1438-1439).

3. **Folder names `lights`/`darks`/`flats` are not the import branch trigger.** Classification is
   recursive IMAGETYP (importer.py:500-557). `_find_lights_subdirectory` is unused in import flow.

4. **Two validity age clocks:** import scan uses **mtime** (`_age_days`, importer.py:873-879);
   library UI uses **header date** (`get_master_age_days`, calibration.py:79-98).

5. **`get_master_age_days` not used in import validity** despite task referencing calibration.py:79
   — enforcement at import is mtime-based via `get_calibration_status` (importer.py:1405-1406).

6. **`allow_passthrough` synthetic master is dead code in production** (calibration.py:441-452;
   no callers with `allow_passthrough=True`).

7. **`AstroPipeline.calibrate()` is a stub** returning None (pipeline.py:16804-16811); real path is
   `quick_calibrate_last_import` / `calibrate_lights_to_calibrated`.

8. **Flat EXPTIME not matched** in library flat lookup (database.py:2188,2250-2258) — filter+gain+bin
   only; differs from dark.

9. **RN scaling depends on caller passing header** — inconsistent between photometry_core SNR helper
   (RN header=None, photometry_core.py:1211) and Phase 2A (header set, photometry_core.py:6665).

---

## Errors (if any)

None — read-only investigation completed successfully.

## Files changed

None (read-only task). Deliverable: `CURSOR_RESULT_caldiag_flow.md`.
