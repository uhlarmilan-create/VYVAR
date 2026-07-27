# CURSOR RESULT - POST-453 CONSOLIDATION (2026-07-27)

Performance, entry-point equivalence, infolog durability, and doc hygiene after draft 453 analysis.

Session data: `dev/results/context/session_20260727_post453/` (pushed with commits).

---

## Rule 0.2

`e406e09` (draft453 session data) was already on origin at task start. New context under
`session_20260727_post453/` committed and pushed with this arc.

---

## Part 1 - Preprocess performance

### 1.1 Profile (one frame, BO_CVn_Light_001, draft 452 field geometry)

| Step | seconds | notes |
|------|--------:|-------|
| FITS read | 0.008 | memmap |
| **Source masking (DAOStarFinder)** | **1.35** | **dominant**; bbox-limited stamp loop |
| Sigma-clipped polynomial fit | 0.055 | negligible |
| Surface evaluation (full grid) | 0.22 | negligible |
| FITS write-back + QC headers | 0.064 | |
| QC metrics (FWHM/elong) | ~0.7 | photutils segmentation |
| **other** | -- | fit is NOT the bottleneck |

**Stop condition check:** subsampled-fit / numerics change is **not** warranted; masking dominates.

### 1.2 Optimisation (numerically neutral)

- Bbox-limited star-mask stamps (same algorithm, local patches only).
- Parallel `_qc_enrich_calibrated_in_place` via `_qc_enrich_one_frame` worker (picklable top-level).
- Single FITS update pass per frame (sky + QC headers merged).

### 1.3 Timing and acceptance

| Metric | draft 452 (old) | POST-453 (new) |
|--------|----------------:|---------------:|
| per-frame preprocess | 18.3 s | **1.23 s** sequential (10f sample) |
| 150 frames projected | 2743 s | **~120 s** (8 workers) |
| byte-identity vs 452 | -- | **max abs diff 0.0** on 9/10 frames |

Method: run new preprocess on **draft 451 calibrated** input (no prior VYSKYORD) and compare
data arrays to **draft 452** output (same frame names). Frame 001 excluded: 451/452 calibrated
inputs differ by ~660 ADU before preprocess (different calibration pass), not a preprocess regression.

Parallel vs sequential on the same 451 inputs: identical.

Anchor `--full`: not re-run this session (interpreted-mode gate green via `--fast`; full anchor
scheduled before reference cut).

---

## Part 2 - Unaccounted UI time (measurement only)

Headless draft 452 (night_run log + folder times):

| Phase | seconds | notes |
|-------|--------:|-------|
| smart_scan_source | 0.6 | |
| smart_import_session | 1.1 | 150 FITS, **0.81 GB** draft 451 Raw => **~0.74 GB/s** |
| calibration (first calibrated lights) | 149.2 | dominates pre-platesolve |
| preprocess (old) | 2743.1 | 38% of run |

UI draft 453 folder CreationTime: Raw -> calibrated **~7 s** once import starts.

**Gap Milan observed (~45 min):** completed draft 453 folder span is **~76 min**, not >2 h.
Pre-first-artifact UI idle (user setup, Streamlit rerun, run not yet started) is **not logged** today;
infolog ring buffer (pre-fix) dropped calibration/preprocess so UI runs could not reconstruct early
phases from saved infolog alone.

No optimisation in this part (per task).

---

## Part 3 - Entry-point equivalence

### 3.1 Observer location

**Root cause:** `NightRunParams.location_id` defaulted to **1** (Dablice). Headless import passed
`id_location=1` into `smart_import_session`; UI used `config.json` `observer_location_id: 2` (Jirny).

**Fixes:**
- `location_id: int | None = None`; headless import uses config when unset.
- Fail-loud if `observer_location_id` unset or LOCATION row missing (names config key).
- `resolve_import_location_id`: removed MIN(ID) / default-ID silent fallbacks.
- Hydrate `cfg.observer_*` from resolved LOCATION after import; `[SITE]` milestone + draft
  `ID_LOCATION` drive BJD via `resolve_site`.
- `_phase2a_observer_location_dict`: `location_id` from OBS_DRAFT when site source is `draft`.

**Prague default path:** config.py default `observer_location_id=1` + old `NightRunParams.location_id=1`
+ DB `resolve_import_location_id` MIN(ID) fallback -- all removed/bypassed for headless.

**Exports affected:** draft 452 pipeline_meta carried Jirny `location_id=2` with Dablice lat/lon
(inconsistent). Anchor `--full` runs headless; AAVSO/VarAstro site coords on anchor exports should be
audited. No evidence in-repo of AAVSO/VarAstro submission from sky-surface-inflated drafts 438-451.

### 3.2 delta_mag_sysrem schema

`save_lightcurve_csv()` always writes `delta_mag_sysrem` (NaN when SysRem off).
Test: `dev/tests/test_post453_equivalence.py`.

### 3.3 Re-verify

Full fresh headless + UI runs on BO CVn **not executed** this session (runtime). Code + unit tests
landed; byte-identical re-check is the next operator step before reference cut.

---

## Part 4 - Infolog durability

- New `src_py/infolog_session.py` (pure Python; complements compiled `infolog.pyd`).
- `start_infolog_session` / `log_milestone` / milestone-first `save_infolog_to_disk`.
- Wired from UI (`app.py`) and headless (`night_run.py`).
- `[PREPROCESS] start` milestone via `log_milestone` in pipeline.

**INV-PREP-01 on real run:** pending next full UI/headless run with new infolog session file.

---

## Part 5 - Documentation

- Removed disproven photutils non-repro claim from `VYVAR_STATE.md`.
- Unified `sigma_pp` convention (46.90 ADU unmasked MAD; ~45.03 masked).
- Production-path harness lesson + `dev/results/context/` retention rule in `VYVAR_PROCESS.md`.

---

## Part 6 - Ledger (ROADMAP)

Confirmed/added in `VYVAR_ROADMAP.md`: CATALOG-PROVENANCE (HIGH), GAIA-PM deferral, BORDER-MARGIN,
SKY-SURFACE-BLAST-RADIUS export audit, R-CVN empty-comp.

---

## Path to reference cut

1. Part 3 closed -- fresh headless + UI equivalence byte-identical (pending runs).
2. Two agreeing full BO CVn runs.
3. Milan decision -> ledger cut.

Performance (Parts 1-2) does not block reference cut (numerics neutral).

---

## Gates

| Gate | Status |
|------|--------|
| `session_baseline_check.py --fast` | PASS after conftest prefers `src_py/*.py` over stale `.pyd` |
| `ruff` (touched files) | clean |
| anchor `--full` | not re-run this session |

## Files changed (summary)

- `src_py/pipeline.py` - preprocess perf + INV-PREP-01 milestone
- `src_py/infolog_session.py`, `src_py/infolog.py`, `src_py/app.py`, `src_py/night_run.py`
- `src_py/database.py`, `src_py/photometry_core.py`, `src_py/simulate_night_run.py`
- `dev/tests/conftest.py`, `dev/tests/test_post453_equivalence.py`, `dev/tests/test_database_fk_draft.py`
- `dev/scripts/post453_*.py`, `dev/results/context/session_20260727_post453/`
- `docs/VYVAR_STATE.md`, `docs/VYVAR_PROCESS.md`, `docs/VYVAR_ROADMAP.md`
