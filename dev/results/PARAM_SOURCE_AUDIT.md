# PARAM-SOURCE-AUDIT - where every VYVAR parameter comes from

Date: 2026-07-17
Arc: UI block - parameters, redesign grounding
Scope: ANALYSIS ONLY. No code/config/registry changed. Paths are post-reorg
(`src_py/` = production code, `dev/` = development material).
Companion machine-readable file: `dev/results/param_source_audit.csv` (304 rows).

All claims below are cited as `path:line`. Where a flow could not be
established from code it is marked UNKNOWN.

---

## Executive summary

There are effectively **FIVE live parameter sources**, but only **three that
matter for the redesign**:

1. `config.json` (repo root, git-tracked) - the single authoritative store for
   286 of 304 AppConfig keys. Loaded by `AppConfig.__post_init__`
   (`src_py/config.py:715-716`).
2. The SQLite reference tables (`EQUIPMENTS`, `TELESCOPE`, `LOCATION`,
   `SCANNING`) joined per-draft via `OBS_DRAFT`/`OBS_FILES` - the authoritative
   store for detector/site/optics FACTS at science time.
3. FITS headers - authoritative at runtime for gain, saturation, frame
   dimensions, plate scale (via WCS), binning, exptime, filter, and time.

The remaining two are minor:
4. Environment variables (`VYVAR_*`) and CLI flags of headless entries.
5. Hardcoded module-level constants (density/crowding overrides, literature
   tables, threshold banks).

Key finding: the DB `SETTINGS` table is **vestigial** - it is seeded on every
DB open but has **zero production readers**. The registry's `kind`
(static/resolved) axis describes AppConfig coercion behavior only and does NOT
describe storage/ownership; Milan's storage-and-ownership model is the correct
target and is not represented anywhere in the current data model.

---

## STEP 1 - Parameter store inventory

### 1.1 Config files

| File | Tracked? | Read as config by | Notes |
|------|----------|-------------------|-------|
| `config.json` (repo root) | YES | `load_config_json` (`src_py/config.py:20-27`), called from `AppConfig.__post_init__` (`src_py/config.py:716`) | The one real config file. 292 keys today. Also holds infra paths (`archive_root`, `calibration_library_root`, `database_path`, `gaia_db_path`, blind index / VSX / exoplanet DB paths). |
| `pyproject.toml` | YES | tooling only (ruff, pytest) - NOT read by the pipeline | Not a science-parameter store. |
| `.streamlit/*` | n/a | none found | No `.streamlit/config.toml` is read by VYVAR code. |
| per-equipment / per-rig config files | n/a | none | No per-rig config file exists; rig facts live in DB tables. |

There is exactly ONE pipeline config file: `config.json`. `save_config_json`
(`src_py/config.py:57-59`) is the only writer, invoked by the Settings UI save
handlers.

Night-run dev scripts **patch `config.json` in place** before a run and restore
it afterwards (e.g. `dev/scripts/dy_peg_night_run_bvr.py:61-80,442-444`;
`dev/scripts/qatar8_night_run_v.py`). This is a transient mutation of the one
config file, not a separate store.

### 1.2 SQLite tables that parameterize behavior

DB path resolved from `config.json` -> `AppConfig.database_path`
(`src_py/config.py:722`); default `vyvar.sqlite3` at repo root. Schema created
in `VyvarDatabase._create_tables` (`src_py/database.py:891-981`).

| Table | Columns that parameterize | Definition | Runtime role |
|-------|---------------------------|------------|--------------|
| `SETTINGS` | `KEY`, `VALUE` | `src_py/database.py:2665-2668` | VESTIGIAL. Seeded with `masterdark_validity_days=90`, `masterflat_validity_days=200` (`src_py/database.py:2673-2682`). Reader `get_setting_int` (`:2684`) and writer `set_setting` (`:2694`) have ZERO callers outside `database.py` (grep confirmed; only `dev/tests/test_master_validity_days_g6_f002.py` reads it directly). |
| `EQUIPMENTS` | `PIXELSIZE`, `SATURATE_ADU`, `GAIN_ADU`, `READNOISE_E`, `FOCAL` | `src_py/database.py:894-902` + migrations (`:2644-2660`) | Detector facts. Read via `get_equipment_*` (`:2827,2849,2879,2897`) and `get_combined_metadata` (`:2971-3103`). Feeds `param_resolver` gain/RN/pixel/saturation. |
| `TELESCOPE` | `DIAMETER`, `FOCAL` | `src_py/database.py:904-911` | Optics facts. Focal feeds plate-scale; diameter feeds `sigma_budget` (`src_py/sigma_budget.py:231-245`). |
| `LOCATION` | `PLACENAME`, `LATITUDE`, `LONGITUDE`, `ALTITUDE` | `src_py/database.py:913-919` | Observatory site. Read via `get_observer_location_by_id` (`:425`) and per-draft join (`:646`). |
| `SCANNING` | `EXPTIME`, `FILTERS`, `BINNING`, `SENSORTEMP`, `GAIN` | `src_py/database.py:921-928` | Import-time FK snapshot for grouping (`find_or_create_scanning_id` `:3157-3196`). NOT a runtime photometry source (FITS headers dominate). `resolve_binning`/`resolve_exptime` exist (`src_py/param_resolver.py:624-633`) but have zero callers. |
| `CALIBRATION_LIBRARY` | equipment+telescope+binning+exptime/gain/filter match keys | `src_py/database.py:1959+`, lookup `:2165-2245` | Master dark/flat selection; validity days come from `cfg`, not SETTINGS. |
| `FIELD_REGISTRY`, `COMP_STAR_LIBRARY` | field reuse + comp-star persistence | `src_py/database.py:3989+,4006+` | Not detector/optics params; finalization bookkeeping (`src_py/ui_finalization.py:317-402`). |

`SETTINGS` key namespace actually stored today: **only** `masterdark_validity_days`
and `masterflat_validity_days`, and both are ignored at runtime.

### 1.3 FITS-header-derived values

Central resolution lives in `src_py/param_resolver.py` (`HEADER_KEYS` at
`:67-78`, `SANITY` bounds at `:52`). Full keyword inventory (100+ keywords) is
in the CSV notes; the science-critical families:

| Value | Header keywords | Lands in | Overrides config? |
|-------|-----------------|----------|-------------------|
| Gain | `EGAIN`, `GAIN` (+ QHY index map) | per-frame error map, SNR, PSF (`src_py/pipeline.py:268-285`, `photometry_core.py:7876-7887`) | YES - header wins over DB and `cfg.gain` (`param_resolver.py:375-451`) |
| Read noise | `RDNOISE`, `READNOISE`, `RDNOISEE`, `RN` | same error paths | Partly - DB `READNOISE_E` wins, then header, then `cfg.read_noise` (`param_resolver.py:490-514`) |
| Saturation | `SATURATE`, `MAXLIN`, `DATAMAX`, `BITPIX`, ... | saturation ceiling (`pipeline.py:5317-5356`) | YES - FITS first, then DB `SATURATE_ADU` |
| Frame W x H | `NAXIS1`, `NAXIS2` | frame geometry everywhere | YES - `NAXIS` on MASTERSTAR overrides `cfg.frame_width_px`/`frame_height_px` (`photometry_core.py:11681-11692`) |
| Plate scale | `CD1_1..CD2_2`, `CDELT*`, `PIXSCALE`, `SECPIX`, `VY_PLTS` | arcsec/px (`photometry_core.py:10326-10410`, `psf_photometry.py:266-328`) | YES - WCS/CD beats DB-derived beats `cfg.*plate_scale*` |
| Binning | `XBINNING`, `YBINNING`, `BINNING` | metadata, master matching (`utils.py:280-285`) | YES - header authoritative |
| Exptime | `EXPTIME`, `EXPOSURE` | JD mid, metadata | header authoritative |
| Filter | `FILTER`, `FILT` | classification, grouping, export | header authoritative |
| Time | `DATE-OBS`, `TIME-OBS`, `JD`, `MJD-OBS` | `mid_exposure_jd`, BJD/HJD (`time_utils.py:63-110`) | primary time anchor; no config equivalent |
| Airmass | `AIRMASS`, `SECZ`, else AltAz from site+coords | per-frame `airmass` (`pipeline.py:9275-9294`) | header wins over computed |
| Site | `SITELAT/LONG/ELEV` families | `resolve_site` fallback (`param_resolver.py:726-793`) | only if no draft LOCATION; then flagged `cfg.observer_*` |
| Pointing | `OBJCTRA/DEC`, `RA/DEC`, `CRVAL1/2`, `VYTARGRA/DE` | plate-solve hints, BJD target | header/WCS |

VYVAR also writes and re-reads its own `VY_*` stamp keywords (e.g. `VY_FWHM`,
`VY_NDAO`, `VY_PLTS`, `VY_REF`) for cross-stage handoff; these are provenance,
not user parameters.

### 1.4 Environment variables, CLI flags, session-state

Environment variables read in production (`src_py/`):

| Env var | Controls | Citation |
|---------|----------|----------|
| `VYVAR_PARALLEL_WORKERS` | unified worker count (overrides legacy env + config/host auto) | `src_py/pipeline.py:16201-16204` |
| `VYVAR_QC_PREPROCESS_WORKERS`, `VYVAR_PER_FRAME_CSV_WORKERS` | legacy worker overrides | `src_py/pipeline.py:16208-16216` |
| `VYVAR_PARALLEL_BACKEND` | `process` vs `thread` | `src_py/pipeline.py:9000-9001` |
| `VYVAR_CALIBRATE_MP` | enable calibration multiprocessing | `src_py/pipeline.py:688-689` |
| `VYVAR_SKIP_SOLVE_FIELD`, `VYVAR_SOLVE_FIELD_EXE`, `VYVAR_PLATE_SOLVE_NO_SIP` | local solve-field control | `src_py/pipeline.py:3939-3945,4362,4377` |
| `ASTROMETRY_NET_API_KEY` | online WCS solve key | `src_py/pipeline.py:4058-4060` |
| `VYVAR_ASTROMETRY_TWEAK_ORDER` | SIP tweak order (clamp 3-6) | `src_py/utils.py:321-328` |
| `VYVAR_LABBE_DEBUG_DUMP[_PATH]` | Labbe empty-aperture debug dump | `src_py/photometry_core.py:762-776` |
| `VYVAR_CT_PROTOTYPE` | color-term prototype diagnostics | `src_py/photometry_core.py:4138` |
| `PYTHONIOENCODING` | set to utf-8 if unset | `src_py/psf_runner.py:84` |

Note: `VYVAR_RANDOM_SEED` is NOT an env var - it is the module constant `42`
(`src_py/utils.py:25`). `VYVAR_PER_FRAME_MASTER_FAST` appears only in a
docstring, never read.

Headless CLI entry points with flags: `src_py/simulate_night_run.py`
(`--source/--eq/--tel/--config/--dry-run/--no-sysrem/--sysrem-iter/--log`,
lines 49-93), `dev/scripts/session_baseline_check.py` (`--fast/--full`,
`:459-468`), plus anchor/cohort runners in `dev/scripts/`. Full list in the
env/CLI appendix of the CSV notes.

Streamlit session-state that persists parameter values: the Parameters
dashboard uses prefix `vyvar_pd_{key}` for 128 auto widgets and writes them
back via `setattr(cfg, key, ...)` + `save_config_json`
(`src_py/ui_params_dashboard.py:123,180-193,254-255`). The Settings tab writes
~50 fields plus per-equipment gain/RN to the DB
(`src_py/ui_settings.py:1056-1139,561-575`). Draft center/optics widgets write
to `OBS_DRAFT` in the DB (`src_py/ui_components.py:23-80`).

### 1.5 Hardcoded module-level constants acting as de-facto parameters

Highest-impact (full list ~35 in the CSV notes):

| Constant | Controls | Citation |
|----------|----------|----------|
| `DENSITY_OVERRIDES` | delta-adjust comp/annulus params by field density class | `src_py/config.py:2445` |
| `CROWDING_LOOSEN_OVERRIDES` / `CROWDING_TIGHTEN_OVERRIDES` | crowding-classifier deltas | `src_py/config.py:2472,2478` |
| `GAIN_SETTING_INDEX_MAP` | QHY294MM gain index -> e-/ADU (`{1:{0:3.17}}`) | `src_py/param_resolver.py:85` |
| `SANITY`, `CROSS_CHECK_RTOL`, `NULL_ISLAND_...` | header/DB accept bounds + cross-check | `src_py/param_resolver.py:52,81,118` |
| `DAO_STAR_FINDER_NO_ROUNDNESS_FILTER` + astrometry defaults | detection shape cuts, tweak order, cpulimit | `src_py/utils.py:304-339` |
| `_PSF_QUALITY_THRESH` | PSF fit grading bands | `src_py/psf_photometry.py:2480` |
| `SparseTrustConfig` defaults, `_CHECK_SOFT_LO/_HARD_LO` | trust bands | `src_py/sparse_trust_core.py:18-23`, `trust_flag_core.py:39-42` |
| K2 literature tables (`SLOPE_*`, `SMITH_K2_NATIVE`, ...) | extinction coefficients | `src_py/k2_extinction.py:36-99` |
| `dao_reconcile` match/blend constants | completeness + matching | `src_py/dao_reconcile.py:23-30` |
| `crowding_index` GAIN/RN fallbacks (3.17 / 7.6) | limiting-mag model | `src_py/crowding_index.py:47-48` |

---

## STEP 2 - Per-key source map (304 keys)

The complete row-per-key table is `dev/results/param_source_audit.csv` with
columns: `key, kind_registry, phase_registry, tier_registry, primary_source,
other_sources, representative_reader, fits_override, db_duplicate,
proposed_owner, in_config_json`.

Writers (uniform, so not repeated per row in the CSV): every key is loaded by
`AppConfig.__post_init__` (`src_py/config.py:716+`, per-key `data.get(...)`
with clamping) and written by `save_config_json` (`src_py/config.py:57`) via a
Settings UI handler - the auto dashboard (`src_py/ui_params_dashboard.py:180-193`)
for `widget=auto` keys, or a dedicated widget in `src_py/ui_settings.py` /
`ui_dao_stars.py` / `ui_photometry.py` for the rest. The precedence-bearing
keys have their exact runtime writer/override chain in the `other_sources`
column.

### 2.1 Distribution

| primary_source | count |
|----------------|-------|
| `config.json` | 286 |
| `code-default-only` (AppConfig default; not in config.json) | 10 |
| `FITS` (gain, frame_width_px, frame_height_px) | 3 |
| `computed(WCS)` (plate_scale_arcsec_per_px, phase01_plate_scale_arcsec_per_px) | 2 |
| `computed(FITS+DB optics)` (plate_solve_fov_deg) | 1 |
| `DB.EQUIPMENTS` (read_noise) | 1 |
| `env/host` (qc_preprocess_workers) | 1 |

| proposed_owner (mechanical seed) | count |
|----------------------------------|-------|
| `config_runtime` | 277 |
| `internal` (paths, worker counts, project_root) | 11 |
| `db_static` (observer site + identity) | 9 |
| `fits_dynamic` (gain, read_noise, frame dims, plate scales, fov) | 7 |

### 2.2 The 14 keys that are NOT plain config.json (the interesting rows)

| key | primary_source | db_duplicate? | fits_override? | proposed_owner | representative reader |
|-----|----------------|---------------|----------------|----------------|-----------------------|
| `gain` | FITS | YES (EQUIPMENTS.GAIN_ADU) | yes (EGAIN/GAIN) | fits_dynamic | `src_py/pipeline.py:309` |
| `read_noise` | DB.EQUIPMENTS.READNOISE_E | YES | yes (RDNOISE) | fits_dynamic | `src_py/pipeline.py:310` |
| `frame_width_px` | FITS.NAXIS1 | no (FITS only) | yes (NAXIS1) | fits_dynamic | `src_py/photometry_core.py:14712` |
| `frame_height_px` | FITS.NAXIS2 | no (FITS only) | yes (NAXIS2) | fits_dynamic | `src_py/photometry_core.py:14713` |
| `plate_scale_arcsec_per_px` | computed(WCS) | YES (PIXELSIZE/FOCAL) | yes (CD/PIXSCALE) | fits_dynamic | `src_py/photometry_core.py:9934` |
| `phase01_plate_scale_arcsec_per_px` | computed(WCS) | YES | yes | fits_dynamic | `src_py/photometry_core.py:10063` |
| `plate_solve_fov_deg` | computed(FITS+DB optics) | NO | yes | fits_dynamic | `src_py/app.py:2155` |
| `export_arcsec_per_px` | config.json | YES (derivable from optics) | no | config_runtime | `src_py/config.py:1253` |
| `observer_lat` | config.json | YES (LOCATION.LATITUDE) | yes (SITELAT) | db_static | `src_py/app.py:2043` |
| `observer_lon` | config.json | YES (LOCATION.LONGITUDE) | yes (SITELONG) | db_static | `src_py/app.py:2044` |
| `observer_alt_m` | config.json | YES (LOCATION.ALTITUDE) | yes (SITEELEV) | db_static | `src_py/app.py:2045` |
| `observer_location_name` | config.json | YES (LOCATION.PLACENAME) | no | db_static | `src_py/app.py:2046` |
| `observer_location_id` | config.json | YES (LOCATION fk) | no | db_static | `src_py/app.py:1965` |
| `masterdark_validity_days` / `masterflat_validity_days` | config.json | YES (SETTINGS, vestigial) | no | config_runtime | `src_py/app.py:1853-1854` |

Everything else (277 keys of algorithm behavior + 8 infra path/worker keys) is
config.json-only with no DB or FITS competitor.

---

## STEP 3 - Precedence and conflict analysis

### 3.1 Load order (config assembly)

1. **Code defaults**: dataclass field defaults in `AppConfig`
   (`src_py/config.py:100-708`).
2. **config.json**: `AppConfig.__post_init__` overwrites each field with
   `data.get(key, default)` plus per-field clamping/validation
   (`src_py/config.py:715-2433`). This is the only file merge.
3. **DB SETTINGS**: NOT merged (vestigial; zero readers).
4. **DB LOCATION hydrate** (observer block only): if
   `observer_location_id > 0`, read `get_observer_location_by_id`; fill
   `observer_location_name` if empty; fill lat/lon/alt ONLY if lat==lon==0.0
   (`src_py/config.py:1231-1244`).
5. **Per-run overrides**: `apply_density_overrides` returns a COPY of cfg with
   density-class deltas applied (`src_py/config.py:2503+`);
   crowding-classifier overrides similarly. UI RUN VYVAR job dict can override
   `sips_dao_*` and FWHM limits at pipeline entry (`src_py/app.py:2141-2142`,
   `pipeline.py:1850-1857`).
6. **FITS-derived** (per frame/draft): `param_resolver.resolve_*` and
   `extract_fits_metadata` compute gain/RN/saturation/plate-scale/frame-dims/
   time; these bypass or override the cfg values as tabulated below.
7. **Environment**: worker counts and solver toggles read last at execution
   time (`src_py/pipeline.py:16201+`).

### 3.2 Where two stores can disagree, and what wins

| Fact | Stores | Winner + citation | Silent or loud |
|------|--------|-------------------|----------------|
| Gain | FITS `EGAIN/GAIN`; DB `GAIN_ADU`; `cfg.gain` | Header e-/ADU if valid, else index-map, else DB, else config (`param_resolver.py:375-451`) | LOUD when header disagrees with DB > 5% (`CROSS_CHECK_RTOL`) - warns "using header (session truth)" (`param_resolver.py:400-406`) |
| Read noise | DB `READNOISE_E`; FITS; `cfg.read_noise` | DB first (equipment-intrinsic), then header, then config (`param_resolver.py:490-514`) | Cross-check warn on header/DB mismatch |
| Saturation | FITS `SATURATE/...`; DB `SATURATE_ADU`; config | FITS keywords, then DB, then DATAMAX/BITPIX (`pipeline.py:5317-5356`) | mostly silent |
| Frame W x H | FITS `NAXIS1/2`; `cfg.frame_width_px/height_px` | NAXIS on MASTERSTAR wins (`photometry_core.py:11681-11692`) | silent |
| Plate scale | WCS CD-matrix; DB `PIXELSIZE*bin/FOCAL`; `cfg.*plate_scale*` | WCS, then DB-derived, then config fallback (`photometry_core.py:10060-10169`) | silent |
| **Observer site** | draft `ID_LOCATION`->LOCATION; FITS `SITELAT/...`; `cfg.observer_*` | **Science (BJD/airmass/lunar/Phase2A): draft LOCATION first, FITS cross-check, config only as flagged fallback** (`param_resolver.py:639-793`, `photometry_core.py:8096-8101`). **AppConfig session: config.json, DB hydrate only if 0,0** (`config.py:1231-1244`) | flagged fallback is loud; the config-vs-draft split is SILENT (see 3.3) |
| Master validity days | DB SETTINGS; config.json | config.json always (SETTINGS unused) (`config.py:724-725`) | silent (SETTINGS simply ignored) |
| Binning/exptime/filter | FITS; DB SCANNING | FITS (SCANNING is import FK snapshot) | silent |
| Focal length | FITS `FOCALLEN`; DB EQUIPMENTS.FOCAL / TELESCOPE.FOCAL | For enriched metadata: DB focal BEFORE header (`pipeline.py:3592-3617`); `resolve_focal_mm`: header then DB (`param_resolver.py:535-560`) | INCONSISTENT direction between the two call paths - see 3.3 |
| Telescope diameter | DB `TELESCOPE.DIAMETER`; hardcoded draft fallback | DB, else literal (`sigma_budget.py:257-267`) | silent fallback |
| Worker count | `VYVAR_PARALLEL_WORKERS`; legacy env; config/host | env first, then legacy env, then config/host auto (`pipeline.py:16201-16229`) | logged |

### 3.3 UI-edits-one-store, pipeline-reads-another (the "why did it run differently" class)

1. **Observer coordinates (highest risk).** The Settings UI location picker
   writes the full LOCATION row into `cfg.observer_*` AND `save_config_json`
   (`src_py/app.py:2039-2047`), so config.json tracks the CURRENTLY selected
   location. But per-draft science reads `OBS_DRAFT.ID_LOCATION` -> LOCATION
   directly (`param_resolver.py:639-647`), NOT `cfg.observer_*`. If a draft was
   imported with a different `ID_LOCATION` than the current config location,
   the light curve uses the draft's site while the UI/export state shows the
   config site. The `config.json` lat/lon can silently drift from the draft's
   actual observing site. History: `dev/scripts/_fix_location_jirny.py` renamed
   LOCATION id=2 to "Jirny" to align the DB row with config
   (`observer_location_id: 2`).

2. **Gain / read noise.** Edited in the Settings UI, they are written to the
   DB `EQUIPMENTS` row via `set_equipment_cosmic_params`
   (`src_py/ui_settings.py:575-576`), NOT to config.json. But `cfg.gain` /
   `cfg.read_noise` still exist in config.json (1.0 / 10.0) and are the LAST
   fallback. A user editing "gain" in Settings changes the DB; the config.json
   values are effectively dead unless FITS and DB both fail to resolve.

3. **Frame dimensions.** `cfg.frame_width_px/height_px` (2082/1397) are
   overridden by FITS `NAXIS` at science time, so editing them in config has no
   effect on a real run.

4. **Plate scale.** Three config keys
   (`plate_scale_arcsec_per_px`, `phase01_plate_scale_arcsec_per_px`,
   `export_arcsec_per_px`, all 1.3) are superseded by WCS/DB-derived scale for
   the actual photometry; they act as fallbacks or the export label only.

5. **Focal-length direction inconsistency.** `_enrich_calibration_metadata_from_header`
   prefers DB focal then header (`pipeline.py:3592-3617`), while
   `resolve_focal_mm` prefers header then DB (`param_resolver.py:535-560`). Two
   code paths can pick different focal values from the same inputs.

---

## STEP 4 - Answers for Milan

**1. How many config files exist and which is authoritative for what?**
One: `config.json` at the repo root. It is authoritative for 286 of 304
parameters - essentially all algorithm-behavior knobs (photometry, comp
selection, variability, HRD, blind solve, calibration diagnostics, trust) plus
the infrastructure paths (archive, calibration library, database, Gaia/VSX/
exoplanet DBs). It is NOT authoritative for detector facts (gain, read noise,
saturation, pixel size, focal), frame geometry, plate scale, or the observing
site at science time - those come from FITS headers and DB tables.

**2. What does the DB SETTINGS table store today, and does it overlap config.json?**
It stores exactly two keys (`masterdark_validity_days=90`,
`masterflat_validity_days=200`), seeded on every DB open. It overlaps
config.json for those two keys, but the overlap is dead: nothing reads SETTINGS
at runtime, and the pipeline uses the config.json values. SETTINGS is
effectively unused scaffolding.

**3. Where do static observatory facts live today, and how fragmented is that?**
Fragmented across FOUR places: (a) DB `EQUIPMENTS` (camera pixel size,
saturation, gain, read noise, per-camera focal); (b) DB `TELESCOPE` (diameter,
focal); (c) DB `LOCATION` (site lat/lon/alt/name); (d) `config.json`
`observer_*` fields that DUPLICATE the LOCATION row, plus `gain`/`read_noise`/
`frame_*`/`plate_scale_*` that duplicate EQUIPMENTS/FITS facts. So a single
"observatory fact" like the site latitude exists in up to three stores
(LOCATION, config.json, FITS header) with a non-obvious precedence.

**4. What is genuinely FITS-derived at runtime?**
Gain (header-first), read noise (header as second choice), saturation ceiling,
frame width/height (NAXIS), plate scale (WCS/CD matrix), binning, exposure
time, filter, observation time (DATE-OBS/JD -> BJD/HJD), airmass, and pointing/
WCS. These are the values that legitimately change per frame or per session and
should be considered dynamic, not user-config.

**5. Top 5 duplications/conflicts to resolve, ranked by risk.**
1. **Observer site: config.json `observer_*` vs DB `LOCATION` vs per-draft
   `ID_LOCATION`.** Science uses the draft's LOCATION; UI/export uses config.
   Highest risk of a silent "ran with the wrong coordinates" outcome.
2. **Gain / read noise: config.json vs DB `EQUIPMENTS` vs FITS.** UI edits go to
   the DB; config values are dead fallbacks; header can override the DB. Users
   cannot tell which value was used without reading the resolve log.
3. **Plate scale: three config keys vs WCS vs DB optics.** Config keys are
   fallbacks/labels only; editing them rarely does what a user expects.
4. **Frame dimensions: config.json vs FITS NAXIS.** Config values are inert on
   real runs.
5. **Master validity days: config.json vs vestigial DB SETTINGS.** Low risk
   (SETTINGS unused) but it is a live source of confusion and should be removed
   or wired, not left half-built.

---

## STEP 5 - Redesign options (sketch, no recommendation)

Milan's target model: STATIC facts -> DB-table management UI; DYNAMIC ->
computed/FITS; everything else -> config file UI. The full SUMMARY MEASURE
REPORT Configuration page must carry a full snapshot of "the config file(s)"
as-run. The audit shows "the config file" today already means config.json PLUS
the DB reference tables PLUS the FITS-resolved values - the snapshot must
capture all three to be honest.

### Option A - Minimal: relabel, do not move storage

Add an ownership axis (`db_static` / `config_runtime` / `fits_dynamic` /
`internal`) to `params_registry.json` (the CSV `proposed_owner` column is the
seed). The Parameters dashboard groups by ownership; `db_static` and
`fits_dynamic` rows are shown READ-ONLY with a link to the DB management UI or
a note "resolved from FITS at runtime". Storage is unchanged.
- Moves: nothing physical; one registry field + dashboard grouping.
- Migration cost: low (registry + UI only; no science-path change; anchor-safe).
- Provenance/report impact: the config snapshot must additionally serialize the
  resolved DB/FITS values (gain, RN, site, plate scale, frame dims) as-run, not
  just config.json, or it will misrepresent the run. Requires extending the
  provenance block to dump `param_resolver` outputs.

### Option B - Consolidate site/detector facts into the DB, thin config.json

Remove the duplicated observatory facts from config.json
(`observer_lat/lon/alt/name`, `gain`, `read_noise`, `frame_width_px/height_px`,
`plate_scale_*`, `export_arcsec_per_px`) and make the DB tables the single
source; config.json keeps only algorithm-behavior knobs + infra paths. The
observer block hydrate in `config.py:1231-1244` becomes the mandatory path.
- Moves: ~14 keys out of config.json into DB-owned status; UI edits for them go
  to DB only (gain/RN already do).
- Migration cost: medium. Touches `AppConfig.__post_init__`, the Settings UI,
  and every reader that used `cfg.gain`/`cfg.observer_*` as a fallback. This IS
  a science-path change (fallback removal) and MUST be anchor-gated. Kill the
  vestigial SETTINGS table in the same pass.
- Provenance/report impact: cleaner - the snapshot is config.json (behavior) +
  a DB facts block (site/detector/optics) + FITS-resolved block. Clear
  three-part story matching Milan's model.

### Option C - Two explicit config files + DB facts

Split config.json into `config_runtime.json` (algorithm behavior, user-tunable)
and keep infra paths in a small `config_paths.json`; DB owns all static facts;
FITS owns dynamics. The report page snapshots runtime config verbatim plus the
resolved facts block.
- Moves: config.json split in two; DB unchanged from Option B; requires a
  loader that merges both files.
- Migration cost: medium-high. New loader + migration of existing config.json +
  every test/fixture/night-run script that patches config.json
  (e.g. `dev/scripts/*night_run*.py`). Anchor-gated.
- Provenance/report impact: the "full copy of the config file(s)" requirement
  becomes literally two files + facts block; most explicit, most churn.

Trade-off summary: A is safe and fast but leaves the duplication in place
(relabels it); B removes the real duplication at the cost of a science-path
fallback change (anchor gate mandatory); C is the cleanest conceptual
separation but the highest migration and test-churn cost. All three require the
provenance snapshot to include the resolved DB + FITS values, not just
config.json, to satisfy the full-config report requirement honestly.

---

## Appendix - method and caveats

- Store inventory and citations gathered by targeted reads of `src_py/config.py`,
  `src_py/database.py`, `src_py/param_resolver.py`, plus four exploration passes
  over `src_py/` for FITS reads, DB call sites, env/CLI/session-state, and
  module-level constants.
- The 304-key CSV was generated by a gitignored scratch script
  (`tmp/build_param_source_audit.py`) that reads the registry + config.json +
  all `src_py/*.py`, indexes the first representative reader per key, and
  applies the mechanical classification rules documented above. No repo code,
  config, or registry was modified.
- `representative_reader` is the first attribute-style read (`cfg.key`) found
  outside `config.py`; a few keys are read via `getattr` by name and may have
  additional readers not surfaced by the first-hit index. Zero keys were left
  without a reader.
- The `proposed_owner` column is a MECHANICAL PROPOSAL seeded by rules
  (site/equipment facts -> db_static; header/WCS-derived -> fits_dynamic; paths/
  workers -> internal; algorithm behavior -> config_runtime). It is input for
  Milan's decision, not a decision.
