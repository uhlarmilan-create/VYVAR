# CURSOR RESULT - Task B: Calibration paths audit

**Date:** 2026-08-05  
**Type:** Read-only audit (no code changes)  
**Base:** `c:\ASTRO\python\VYVAR` repo HEAD  

---

## Active data-corruption check

No evidence of **silent wrong-pixel math** (calibration applied twice or to the wrong array) on the three paths audited. The highest-severity issue found is **provenance falsehood**: draft-level `CALIBRATION_MODE=vyvar_calibrated` and PDF/infolog text claiming *"VYVAR bias/dark/flat applied"* on PASSTHROUGH runs (mechanism 2). That misstates science metadata in JAAVSO/AAVSO-bound deliverables; it does not rewrite pixel values.

---

## 1. Premise check (restated from code)

**What is being compared.** The intended design compares (a) VYVAR performs calibration end-to-end vs (b) VYVAR consumes existing lights as-is.

**What the code actually implements.** The tree contains **three** distinct mechanisms, not two. Workflow **(b) maps onto two of them**, not one:

| Mechanism | Maps to intent (b)- | Role |
|-----------|---------------------|------|
| 1 - VYVAR calibration | No (this is (a)) | Default path when usable MasterDark exists |
| 2 - PASSTHROUGH | **Partial (b)** - implicit, accidental | Fires when calibration step runs but no usable dark (`pipeline.py:15997-16011`) |
| 3 - `pre_calibrated_mode` | **Partial (b)** - explicit, deliberate | User opt-in; calibration step skipped entirely (`night_run.py:688-698`, `app.py:355-363`) |

**Key divergence.** Mechanisms 2 and 3 both mean "VYVAR applied no dark/flat math," but they differ in:

- **Output tree:** 2 - `calibrated/lights` (copy); 3 - `non_calibrated/lights` (in-place import, no `calibrated/` tree)
- **FITS provenance:** 2 writes `VY_CALIB=PASSTHROUGH`, `VY_CFLAG=P`, `VY_CALST=0`, `VYVARCAL=True` (`pipeline.py:15898-15905`); 3 writes **no VYVAR calibration headers** on import (`importer.py:1803-1804` copies only)
- **Draft provenance:** 2 records `OBS_DRAFT.CALIBRATION_MODE=vyvar_calibrated` (`night_run.py:665-669`); 3 records `pre_calibrated` (`night_run.py:669`, `draft_provenance.py:14`)
- **Import path overlap:** Both can use the `quick_look` import branch into `non_calibrated/lights` (`importer.py:1547-1548`, `1785-1834`), but mechanism 3 **always** forces `quick_look=True` via `apply_pre_calibrated_import_plan` (`draft_provenance.py:52`) whereas mechanism 2's quick_look is triggered by **missing MasterDark at scan time** (`importer.py:1547`).

**Conclusion for Milan:** Supported workflow (b) is **two code paths**, not one. Whether that split is intentional (accidental-missing-dark vs deliberate-external-cal) is a product decision, not established by code alone.

---

## 2. Path inventory (B1)

### Mechanism 1 - VYVAR calibration

| Field | Evidence |
|-------|----------|
| **Entry condition** | `pre_calibrated_mode=False` (default); usable MasterDark at calibrate time (`pipeline.py:15993-15996`, `_has_usable_master_dark`) |
| **UI entry** | `app.py:2235-2240` - `! RUN VYVAR`; `_run_vyvar_full_pipeline(..., pre_calibrated_mode=False)` default (`app.py:139`) |
| **CLI entry** | `night_run.run_night_pipeline(NightRunParams(pre_calibrated_mode=False))` default (`night_run.py:63`); invoked from `simulate_night_run.py:111-120` (no `--pre-calibrated` flag) |
| **Call chain** | Scan: `smart_scan_source` (`importer.py:1544-1564`) - Import to `Raw/lights` when dark found (`importer.py:1836-1843`) - `record_draft_calibration_provenance(..., vyvar_calibrated)` (`app.py:331-335`, `night_run.py:665-669`) - `quick_calibrate_last_import` (`pipeline.py:18271-18329`) - `calibrate_lights_to_calibrated` (`pipeline.py:15932+`) - `_calibrate_one_light_apply_masters_in_ram` / disk (`pipeline.py:15308-15510`, `15565-15675`) - RAM QC / platesolve / MASTERSTAR / photometry via `resolve_draft_lights_root(..., draft_id, db)` - `calibrated/lights` (`draft_provenance.py:47`) |
| **Lights source for downstream** | `archive/<draft>/calibrated/lights/<setup>/` |
| **FITS headers written** | `VYVARCAL`, `VY_DARK`, `VY_FLAT`, `VY_CFLAG` (DF/D/FS), `VY_CALIB` (DARK+FLAT / DARK_ONLY / -), `VY_CALST` (2/1/0), `VY_CLBX`, `VY_CLBY`, `VY_MBNC`, `VY_COSM`, `VY_OBSG`, optional `VY_MDP`, `VY_MFP`, `VY_FLATM`, `VY_FLFL`, `VY_WARN` (FS), CAL-DIAG via `apply_cal_diag_headers` (`pipeline.py:15508`, `cal_diag.py:380-383`), post-cal QC keys `VYQCPASS`, `VY_QCHFR`, etc. (`pipeline.py:15630-15640`) |
| **DB - OBS_FILES** | `IS_CALIBRATED=1` if dark applied (`"D" in flags`), else 0 (`pipeline.py:16223-16225`); `CALIB_TYPE` from `_calibration_type_from_flags`; `CALIB_FLAGS` = VY_CFLAG |
| **DB - OBS_DRAFT** | `CALIBRATION_MODE='vyvar_calibrated'` (`database.py:2778-2780`, `draft_provenance.py:154-155`) |
| **draft_manifest.json** | `{"draft_id", "calibration_mode": "vyvar_calibrated", "updated_utc", ...}` (`draft_provenance.py:80-88`) |

### Mechanism 2 - PASSTHROUGH (no usable MasterDark at calibrate)

| Field | Evidence |
|-------|----------|
| **Entry condition** | Calibration step invoked (`pre_calibrated_mode=False`) but `_has_dark_any` false (`pipeline.py:15993-16011`) |
| **UI entry** | Same as mechanism 1 - user clicks `! RUN VYVAR`; scan may warn "No suitable MasterDark - Quick Look" (`importer.py:1548-1549`) but run continues into calibrate |
| **CLI entry** | Same as mechanism 1 - `NightRunParams(pre_calibrated_mode=False)` + missing/expired dark |
| **Call chain** | Scan may set `quick_look=True` if dark missing at scan (`importer.py:1547`) - import to `non_calibrated/lights` (`importer.py:1785-1834`) **or** `Raw/lights` if dark was found then lost - `quick_calibrate_last_import` (`pipeline.py:18300-18314`) - `calibrate_lights_to_calibrated` early exit - `_passthrough_lights_to_calibrated` (`pipeline.py:16003-16011`, `15870-15929`) - downstream reads `calibrated/lights` |
| **Lights source for downstream** | `archive/<draft>/calibrated/lights/<setup>/` (copied from `non_calibrated/lights` or `Raw/lights`) |
| **FITS headers written** | `VYVARCAL=True`, `VY_DARK=False`, `VY_FLAT=False`, `VY_CFLAG=P`, `VY_CALIB=PASSTHROUGH`, `VY_CALST=0`, history "No calibration frames applied." (`pipeline.py:15898-15904`); CAL-DIAG stub via `passthrough_cal_diag_headers`: `VY_DKRSMP=PASSTHROUGH`, `VY_CDSTAT=PASS` when gate enabled (`cal_diag.py:386-390`, `pipeline.py:15905`) |
| **DB - OBS_FILES** | `IS_CALIBRATED=0`, `CALIB_TYPE=PASSTHROUGH`, `CALIB_FLAGS=P` (`pipeline.py:15884-15891`, `15911-15917`) |
| **DB - OBS_DRAFT** | **`CALIBRATION_MODE='vyvar_calibrated'`** - same as mechanism 1 (`night_run.py:669`, `app.py:335`) |
| **draft_manifest.json** | `"calibration_mode": "vyvar_calibrated"` (written at import, before passthrough is known) |

### Mechanism 3 - `pre_calibrated_mode`

| Field | Evidence |
|-------|----------|
| **Entry condition** | Explicit `pre_calibrated_mode=True` (`app.py:139`, `night_run.py:63`) |
| **UI entry** | `[folder] RUN VYVAR (non-cal)` (`app.py:2242-2252`, `2284`) |
| **CLI entry** | **Not reachable via argparse.** `NightRunParams.pre_calibrated_mode` exists (`night_run.py:63`) but `simulate_night_run.py:48-120` has no flag; dev scripts set it in Python (`dev/scripts/toi1131_night_run_v.py:192`, `qatar8_night_run_v.py:625`, etc.) |
| **Call chain** | Scan (masters may be found) - `apply_pre_calibrated_import_plan` clears masters, forces `quick_look=True` (`draft_provenance.py:50-62`, `night_run.py:606-614`, `app.py:294`) - import to `non_calibrated/lights` (`importer.py:1785-1834`) - `record_draft_calibration_provenance(..., pre_calibrated)` (`night_run.py:665-669`, `app.py:331-335`) - **calibration step skipped** (`night_run.py:688-698`, `app.py:355-363`) - `resolve_draft_lights_root` - `non_calibrated/lights` - platesolve / MASTERSTAR / photometry |
| **Lights source for downstream** | `archive/<draft>/non_calibrated/lights/<setup>/` (`draft_provenance.py:45-46`) |
| **FITS headers written by VYVAR** | **None** on import (file copy only, `importer.py:1803-1804`). Preprocess/QC may add `VY_SKYSF`, `VY_FWHM`, etc. later (`pipeline.py:17248+`) |
| **DB - OBS_FILES** | At import: `is_calibrated=0`, `calib_type=RAW_NON_CALIBRATED` (`importer.py:1811-1812`). No calibrate-time sync |
| **DB - OBS_DRAFT** | `CALIBRATION_MODE='pre_calibrated'` (`draft_provenance.py:14`, `154-155`) |
| **draft_manifest.json** | `"calibration_mode": "pre_calibrated"` |

---

## 3. B2 - Overlap and ambiguity

### `pre_calibrated_mode=True` + usable MasterDark exists

- **Dark silently ignored:** Yes. `apply_pre_calibrated_import_plan` sets `plan.dark_master = None` and clears per-key maps (`draft_provenance.py:53-57`). After import, UI and night_run explicitly null masters again (`app.py:355-359`, `night_run.py:688-692`).
- **User told:** Partially. Warning appended to plan: *"Pre-calibrated mode: bias/dark/flat skipped-"* (`draft_provenance.py:60-62`). UI logs `calibration_mode_report_line(CALIBRATION_MODE_PRE)` (`app.py:241-242`). Scan may still show MasterDark "found" from pre-mutation scan (`importer.py:1542`) - **stale/conflicting signal**.

### `pre_calibrated_mode=False` + no usable MasterDark - PASSTHROUGH

- PASSTHROUGH fires (`pipeline.py:15997-16011`).
- **`OBS_DRAFT.CALIBRATION_MODE` set to `vyvar_calibrated`:** Confirmed (`night_run.py:669`, `app.py:335`).

### Suspected defect - `calibration_mode_report_line` falsehood

**CONFIRMED - severity HIGH.**

```65:68:src_py/draft_provenance.py
def calibration_mode_report_line(mode: str | None) -> str:
    if str(mode or "").strip() == CALIBRATION_MODE_PRE:
        return "Calibration: skipped - source treated as pre-calibrated"
    return "Calibration: VYVAR bias/dark/flat applied"
```

- PASSTHROUGH drafts resolve as `vyvar_calibrated` (DB/manifest) - report line claims full VYVAR calibration.
- **PDF:** `photometry_report.py:2762-2767` draws that line on the Summary Measure Report cover.
- **pipeline_meta:** `photometry_core.py:11025-11028` writes the same line via `resolve_calibration_mode(draft_id, db)` - also returns `vyvar_calibrated` for PASSTHROUGH runs.
- **UI dashboard** correctly warns on per-file `CALIB_TYPE=PASSTHROUGH` (`ui_quality_dashboard.py:470-471`) but PDF uses draft-level mode, not per-file state.

---

## 4. B3 - Source-of-truth resolution

### `resolve_calibration_mode` (`draft_provenance.py:105-124`)

Priority: DB (`OBS_DRAFT.CALIBRATION_MODE`) - `draft_manifest.json` - default `vyvar_calibrated`.

**Silent DB downgrade:** `except Exception: pass` at `draft_provenance.py:118-119` - any DB failure skips to manifest/default. If manifest absent or corrupt (`load_draft_manifest` returns `{}` on error, `draft_provenance.py:100-102`), mode becomes **`vyvar_calibrated`** even for pre-cal drafts.

### Manifest-only `resolve_draft_lights_root` call sites

| Site | Function | Draft states that reach it | Manifest guaranteed- | Wrong-branch consequence |
|------|----------|---------------------------|----------------------|--------------------------|
| `pipeline.py:1221` | `draft_obs_group_count` | Any draft path passed to multi-group logic (`night_run.py:826`, `pipeline.py:11429`) | **No** - manifest may be missing on legacy drafts | Pre-cal draft: defaults to `calibrated/lights` (empty) - fallback `processed/lights` (`1222-1223`) - **count 0** if only `non_calibrated/lights` exists - `draft_is_multi_group_obs` false when true - **wrong MASTERSTAR single-group vs per-group strategy** (`night_run.py:826-833`) |
| `pipeline.py:2601` | `estimate_archive_memory_profile` | UI/night_run memory preflight (`app.py:397`, `night_run.py:741`) | **No** | Pre-cal draft: RAM estimate uses empty/wrong tree - **incorrect memory guidance** (operational, not pixel math) |

Most science-path call sites pass `draft_id` + `db` (e.g. `resolve_masterstar_input_root` `pipeline.py:1255`, night_run `718`).

---

## 5. B4 - Downstream consumers

### Header read/write census

| Header | Writers | Readers (science path) |
|--------|---------|------------------------|
| `VYVARCAL` | `pipeline.py:15460`, `15898` | **None found** |
| `VY_CALIB` | `pipeline.py:15473`, `15902` | **None found** (UI reads DB `CALIB_TYPE` instead, `ui_quality_dashboard.py:463-471`) |
| `VY_CALST` | `pipeline.py:15474-15477`, `15903` | **None found** |
| `VY_CFLAG` | `pipeline.py:15469`, `15901` | `_hdr_vy_cflag_str` internally (`15264-15270`); copied to `masterstar_context` snippet for display only (`masterstar_context.py:153-155`) - **not used for branching** |

**Confirmed:** `VY_CALST` / `VY_CALIB` / `VYVARCAL` are **write-only** on the science path. Branching uses draft-level `resolve_calibration_mode` / `is_pre_calibrated_draft` and DB `CALIB_TYPE` / `IS_CALIBRATED`.

### Stage assumptions (load-bearing)

| Stage | Assumes VYVAR-calibrated- | Evidence |
|-------|---------------------------|----------|
| **Sky-surface preprocess** | **No explicit gate.** Runs on whatever file path QC touches (`pipeline.py:17211-17235`). Uses `VY_SKYSF` idempotency guard (`17177-17187`) - assumes VYVAR preprocess owns the frame, not that dark was applied. Pre-cal / PASSTHROUGH frames get sky subtraction if order > 0. |
| **CAL-DIAG radiometry gate** | **Only during mechanism 1 calibrate.** Pre-gate requires dark path (`cal_diag.py:429-431` - `if dark_p is None: continue`). Mechanism 2: `passthrough_cal_diag_headers` writes PASS without checks (`cal_diag.py:386-390`). **Mechanism 3: no equivalent** - calibration skipped; no `cal_diag.json` (`cal_diag.py:447-450` requires `session.gate_results`). |
| **Bad-pixel masking (`bpm_dark_mad_sigma`)** | **Optional dark BPM sidecar.** Used when `master_dark_path` resolves to `*_dark_bpm.json` (`pipeline.py:12951-12953`, `13714-13716`). Pre-cal / PASSTHROUGH: masters nulled - **BPM not applied** (graceful skip, not error). |
| **Master validity clocks** | **Scan-time only.** `masterdark_validity_days` / `masterflat_validity_days` evaluated in `smart_scan_source` (`importer.py:1489-1495`, `1538-1540`). Pre-cal mode runs scan **before** clearing masters (`night_run.py:581-614`) - may emit expired/found warnings that are then discarded. **No warning that masters were intentionally ignored** beyond plan warning string. |
| **L.A.Cosmic** | **Runs regardless of calibration state** when `enable_lacosmic=True` (`pipeline.py:17239-17240`, `config.py:759`). Applied to pre-cal and PASSTHROUGH frames in QC in-place path - **may be suboptimal on non-dark-subtracted data**; not blocked. |

---

## 6. B5 - Noise model and units

| Question | Finding |
|----------|---------|
| **Validate pre-cal units/gain vs DB equipment-** | **No pre-cal-specific gate.** `param_resolver.py:7-12` resolves gain (header-DB-config) and read_noise (DB-authoritative) with sanity ranges and cross-check warnings - applies uniformly; **does not verify** external calibration state, electron vs ADU units, or stacked-frame semantics. |
| **Labbe empty-aperture term on externally pedestal-removed frames-** | **No calibration-mode guard.** `_measure_empty_aperture_scatter` documents inclusion of "pedestal offsets" (`photometry_core.py:865-871`). `_sky_pp_for_photometric_error` prefers `sky_surface_bg_median_adu` for Howell term (`photometry_core.py:1272-1274`). If external pipeline already removed pedestal and VYVAR also subtracts sky surface, **error budget assumptions may be wrong** - gap, not guarded. |
| **Double-processing check (dark-subtracted + VYVAR cal)-** | **None for mechanism 3.** Mechanism 1 would apply dark if masters exist; mechanism 3 prevents that. **No check** that imported frames aren't already bias-subtracted when user selects non-cal mode. |

---

## 7. B6 - Report and export truthfulness

| Deliverable | Mechanism 1 | Mechanism 2 (PASSTHROUGH) | Mechanism 3 (pre-cal) |
|-------------|-------------|---------------------------|------------------------|
| **PDF cover calibration line** | Correct: "VYVAR bias/dark/flat applied" | **FALSE** - same line (`photometry_report.py:2767`) | Correct: "skipped - pre-calibrated" |
| **pipeline_meta.json `calibration_mode`** | `vyvar_calibrated` via Phase 2A merge (`photometry_core.py:11025-11028`) | **`vyvar_calibrated`** (false) | `pre_calibrated` |
| **AAVSO / VarAstro exports** | No calibration-state line in export body (`export_reports.py:303-330`, `citations.py:623-645` - algorithm citations only) | Same - **no PASSTHROUGH disclosure** | Same - **no pre-cal disclosure** |
| **UI quality dashboard** | No special warning | **Warns** "uncalibrated data (Passthrough Mode)" (`ui_quality_dashboard.py:470-471`) | No PASSTHROUGH-style warning (DB `CALIB_TYPE=RAW_NON_CALIBRATED`) |

**When is `calibration_mode` missing from pipeline_meta first-** Phase 2A merge is primary writer (`photometry_core.py:11044-11059`). PDF back-fills from manifest-only `resolve_calibration_mode(archive_path=...)` without `draft_id` (`photometry_report.py:661-668`) - works if manifest present; fails silently to default if manifest missing (`except` at 669-671).

---

## 8. B7 - Test coverage

### `dev/tests/test_pre_calibrated_run.py`

**10 tests** - all helper/unit level:

- Plan mutation, manifest round-trip, DB provenance, `resolve_draft_lights_root`, path mapping for MASTERSTAR (`test_apply_pre_calibrated_import_plan_*`, `test_resolve_*`, etc.)

**Not covered:**

- End-to-end mechanism 3: UI/CLI - import - skip cal - platesolve - photometry - light curve
- `calibration_mode_report_line` truth on PDF output
- Multi-group pre-cal + `draft_is_multi_group_obs` manifest-only bug

### Mechanism 2 (PASSTHROUGH)

**No dedicated E2E test** through `quick_calibrate_last_import` / night_run with missing dark.

`dev/tests/test_cal_diag_gate.py`:

- `test_passthrough_provenance_headers` - unit test for `passthrough_cal_diag_headers` only (`334-341`)
- All `calibrate_lights_to_calibrated` integration tests supply a dark master (`152+`, `171+`, `396+`)

### Related

- `dev/tests/test_pre_cal_proc_csv_naming_e2e.py` - export naming on existing draft fixture; **not** a full non-cal pipeline run

---

## 9. B8 - UI/CLI parity

| Surface | Mechanism 3 reachable- | Evidence |
|---------|------------------------|----------|
| **Streamlit UI** | Yes | `app.py:2242-2284` - dedicated button |
| **`simulate_night_run.py` CLI** | **No** | `argparse` lacks `--pre-calibrated` (`simulate_night_run.py:48-120`); `NightRunParams` built without flag |
| **`night_run.py` module CLI** | **No** | No `if __name__` / argparse in `night_run.py` |
| **Dev scripts** | Yes (Python API) | `toi1131_night_run_v.py`, `qatar8_night_run_v.py`, `chiandh_allfilters_overnight.py`, etc. |

**Parity break confirmed:** mechanism 3 is GUI-first unless callers construct `NightRunParams(pre_calibrated_mode=True)` manually.

---

## 10. B9 - Naming (`non_calibrated` vs `calibrated`)

| Directory | Actual contents in code |
|-----------|-------------------------|
| `non_calibrated/lights` | Mechanism 3 **pre-calibrated** frames; also mechanism 2 **pre-passthrough** import staging; partial-flat missing imports (`importer.py:1845-1847`) |
| `calibrated/lights` | Mechanism 1 VYVAR-calibrated frames **and** mechanism 2 **uncalibrated PASSTHROUGH copies** |

**Comprehensibility finding:** Directory names invert user mental model - "calibrated" can hold raw passthrough; "non_calibrated" holds deliberately pre-calibrated science frames.

**Blast radius of rename (count only, no proposal):**

| Scope | `non_calibrated` occurrences |
|-------|------------------------------|
| `src_py/` (production) | ~51 references across 8 files (`pipeline.py` 25, `app.py` 8, `night_run.py` 6, `draft_provenance.py` 5, `importer.py` 4, `ui_calibration.py` 2, `ui_quality_dashboard.py` 1) |
| `dev/tests/` | 10 |
| `dev/scripts/` | ~15 across 8 scripts |
| `dev/results/`, `docs/` | ~15 (documentation only) |

**On-disk drafts:** Existing `Archive/Drafts/draft_*/non_calibrated/` trees would break unless symlink migration or compatibility shim reads old paths. Path strings also stored in DB `FILE_PATH`, import logs (`importer.py:1819`), and manifest - rename is **high blast radius** for historical archives.

---

## 11. Findings table

| ID | Severity | file:line | Evidence | Consequence | Proposed fix |
|----|----------|-----------|----------|-------------|--------------|
| F-B01 | **HIGH** | `draft_provenance.py:65-68`, `night_run.py:669`, `photometry_report.py:2767` | PASSTHROUGH runs record `vyvar_calibrated`; report line says "VYVAR bias/dark/flat applied" | **False calibration claim in PDF** destined for archives / JAAVSO | Add `passthrough` calibration mode (DB + manifest); extend `calibration_mode_report_line`; backfill from `OBS_FILES.CALIB_TYPE` when draft mode is ambiguous |
| F-B02 | **HIGH** | `photometry_core.py:11025-11028` | Same mode resolution for `pipeline_meta` | Downstream exports inherit false provenance | Same as F-B01; optionally embed per-file PASSTHROUGH counts in meta |
| F-B03 | **MED** | `pipeline.py:1221-1225`, `night_run.py:826` | `draft_is_multi_group_obs` without `draft_id`/`db` | Pre-cal multi-group drafts misclassified - wrong MASTERSTAR selection strategy | Pass `draft_id`/`db` into `draft_obs_group_count` |
| F-B04 | **MED** | `draft_provenance.py:118-119` | DB read failure - silent default `vyvar_calibrated` | Pre-cal draft misrouted to `calibrated/lights` if manifest also missing | Log ERROR; fail closed or require manifest for pre-cal |
| F-B05 | **MED** | `importer.py:1547-1549` vs `draft_provenance.py:50-62` | Missing-dark quick_look and pre_cal both land in `non_calibrated/` | Same tree, different semantics - operator confusion | Document; long-term unify or tag manifest with `import_reason` |
| F-B06 | **MED** | `cal_diag.py:386-390` vs (none for mech 3) | PASSTHROUGH gets CAL-DIAG PASS stub; pre-cal gets nothing | Pre-cal drafts lack radiometry gate record | Explicit `pre_calibrated` CAL-DIAG skip record in manifest or `cal_diag.json` |
| F-B07 | **MED** | `pipeline.py:17239-17240` | L.A.Cosmic on all frames | CR cleaning on raw/bias-dominated frames may artifact | Gate L.A.Cosmic on calibration state or sky level |
| F-B08 | **LOW** | `pipeline.py:2601-2603` | Memory profile manifest-only | Wrong RAM estimate for pre-cal | Pass `draft_id`/`db` |
| F-B09 | **LOW** | `simulate_night_run.py:48-120` | No `--pre-calibrated` | CLI parity break | Add flag wired to `NightRunParams.pre_calibrated_mode` |
| F-B10 | **LOW** | `export_reports.py:303-330` | No calibration line in AAVSO/VarAstro | External reviewers cannot see PASSTHROUGH/pre-cal from export files alone | Add `# CALIBRATION:` comment line from `pipeline_meta` |
| F-B11 | **LOW** | B9 naming | `non_calibrated` holds pre-cal; `calibrated` holds passthrough raw | Operator mis-reads disk layout | Document; defer rename (large blast radius) |

---

## 12. Gap list

### Workflow (b) - does not exist yet (vs exists but wrong)

| Gap | Type |
|-----|------|
| Single unified "no VYVAR calibration" mode | **Does not exist** - two mechanisms |
| Explicit PASSTHROUGH opt-in / fail-loud when dark missing in vyvar mode | **Does not exist** - implicit fallback |
| Pre-cal units/gain/e-/ADU validation gate | **Does not exist** |
| Double-subtraction / already-calibrated frame detector | **Does not exist** |
| CAL-DIAG / radiometry record for mechanism 3 | **Does not exist** |
| AAVSO/VarAstro calibration disclosure | **Does not exist** |
| CLI flag for `pre_calibrated_mode` | **Does not exist** |
| E2E tests for mechanisms 2 and 3 through light curve | **Does not exist** |
| PASSTHROUGH draft-level provenance in PDF/meta | **Exists but wrong** (F-B01, F-B02) |
| UI dashboard PASSTHROUGH warning | **Exists** (`ui_quality_dashboard.py:470-471`) but not wired to PDF |

---

## 13. Decision list for Milan

1. **Unify mechanisms 2 and 3-**
   - **A:** Keep distinct - accidental missing-dark (PASSTHROUGH - `calibrated/`) vs deliberate external-cal (`pre_calibrated` - `non_calibrated/`).
   - **B:** Unify under one "no VYVAR calibration" mode with explicit user intent at scan time.
   - *Trade-off:* A preserves backward-compatible trees; B reduces operator confusion and false PDF lines.

2. **PASSTHROUGH: implicit vs explicit-**
   - **A:** Keep implicit fallback when dark missing (current `pipeline.py:15997-16011`).
   - **B:** Fail loud in `vyvar_calibrated` mode; require explicit opt-in for passthrough.
   - *Trade-off:* A keeps partial nights alive; B prevents silent uncalibrated science with false provenance.

3. **Pre-cal units/gain sanity gate before v1.0-**
   - **A:** Document assumption ("operator confirms frames match EQUIPMENTS gain/RN").
   - **B:** Implement gate (header EGAIN vs DB, median ADU range, optional `IMAGETYP` check).
   - *Trade-off:* A is fast; B protects SNR/error budget on Telescope Live-style imports.

4. **PDF calibration line source of truth-**
   - **A:** Draft-level `CALIBRATION_MODE` only.
   - **B:** Derive from per-file `OBS_FILES.CALIB_TYPE` aggregate (PASSTHROUGH count > 0 - passthrough line).
   - *Trade-off:* B is truthful for mixed drafts; A is simpler but wrong today for mechanism 2.

5. **Rename `non_calibrated/` directory-**
   - **A:** Keep names; document inversion (B9).
   - **B:** Rename with on-disk migration (e.g. `source_lights/`).
   - *Trade-off:* B ~50+ production references + all historical drafts; high cost.

---

## 14. Recommended fix order

1. **F-B01 / F-B02 - Provenance falsehood (HIGH)**  
   Introduce distinguishable draft/file calibration state for PASSTHROUGH; fix `calibration_mode_report_line` and PDF/meta writers. **Reason:** Science deliverable integrity; smallest conceptual fix with highest user-visible impact.

2. **F-B03 - Multi-group MASTERSTAR routing (MED)**  
   Pass `draft_id`/`db` to `draft_obs_group_count`. **Reason:** Pre-cal multi-filter nights may silently use wrong MASTERSTAR strategy.

3. **F-B09 - CLI `--pre-calibrated` (LOW effort, parity)**  
   Wire `simulate_night_run.py` (and any canonical CLI entry). **Reason:** Unblocks headless regression for mechanism 3.

4. **E2E tests for mechanisms 2 and 3 (B7 gaps)**  
   After provenance fix, lock behavior with night_run-level tests. **Reason:** Prevent regression on F-B01.

5. **F-B04 - Fail-closed mode resolution (MED)**  
   Replace silent `except: pass` on DB read. **Reason:** Prevents wrong lights root on DB glitch.

6. **F-B10 - Export calibration comment (LOW)**  
   AAVSO/VarAstro `# CALIBRATION:` line from truthful meta. **Reason:** External reviewer visibility.

7. **Product decisions (Milan -13 items 1-3)** before deeper refactors (unify modes, PASSTHROUGH policy, units gate, rename).

8. **F-B06, F-B07, F-B08** - CAL-DIAG pre-cal stub, L.A.Cosmic gating, memory profile - as follow-on hardening.

---

## 15. Audit question index

| Q | Section |
|---|---------|
| B1 | -2 Path inventory |
| B2 | -3 |
| B3 | -4 |
| B4 | -5 |
| B5 | -6 |
| B6 | -7 |
| B7 | -8 |
| B8 | -9 |
| B9 | -10 |

---

*End of Task B audit. No code changes made.*
