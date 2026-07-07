CURSOR RESULT — 2026-07-07 18:45 UTC+2

What I did
Implemented CAL-DIAG radiometry gate per `docs/VYVAR_CAL_DIAG_SPEC.md` v1.1 (8 stages): dark resample
mode override, shared gate helper + three-path wiring (sequential, RAM-QC, MP parent pre-gate), provenance
headers/json, config keys, D3 comments, 14 pytest tests, draft_424 regression PASS, docs close.

## Output / findings

### Stage 1 — resample core override
- `calibration.py:30` `DarkResampleMode`; `calibration.py:201-261` `dark_resample_mode` on
  `resample_master_to_light_binning` (MEAN reuses flat block-mean branch); `calibration.py:431+`
  routed through `get_processed_master`.
- Default `mode="sum"` byte-identical when gate OFF or PASS/SUM.
- Commit: `0268547`

### Stage 2 — gate helper + cache + three-path wiring
- New `cal_diag.py`: `_cal_diag_gate_for_obs_group` (Check A/B), `ensure_cal_diag_gate`,
  `run_cal_diag_pregate`, `dark_np_for_cal_diag`, session export for MP workers.
- `pipeline.py`: pregate in `calibrate_lights_to_calibrated` (~15129+), RAM-QC (~2011+),
  `AstroPipeline.calibrate_batch` (~17287+); `_calibrate_batch_process_one` (~14724+);
  FAIL-CLOSED skip + partial delete; `stats["cal_diag_aborted_groups"]`.
- Commit: `6ecdc4d`

### Stage 3 — provenance
- Headers: `apply_cal_diag_headers` / `passthrough_cal_diag_headers` (`cal_diag.py:372+`);
  wired in `_calibrate_one_light_apply_masters_in_ram` / passthrough path.
- `write_cal_diag_json` at end of calibrate (`pipeline.py:15400-15402`).
- Phase 2A merge: `photometry_core.py:8621+` `load_cal_diag_json_for_meta`.
- Commit: `fe794bd`

### Stage 4 — config
- `config.py:117-121` defaults + clamps; `ui_settings.py` exposes `cal_diag_gate_enabled`;
  `docs/VYVAR_PARAMS.md` CAL-DIAG table (1 exposed, 4 hidden).
- Commit: `dc8d977` (combined with Stage 5 in same commit)

### Stage 5 — D3 comments
- `param_resolver.py:155-157` READNOISE_E bin1 semantic; `database.py:2928-2930` docstring mirror.
- Commit: `dc8d977`

### Stage 6 — tests
- `tests/test_cal_diag_gate.py`: **14 passed** — SUM PASS, AUTO-CORRECT, FAIL-CLOSED (garbage + bf1),
  near-zero sky WARN, PASSTHROUGH headers, path-coverage pregate, byte-identity gate off/on,
  cal_diag.json write, fail-closed sibling groups.
- Full suite: **549 passed**, 15 skipped.
- `ruff check . --select BLE001,E722`: clean.
- Commit: `de05011`

### Stage 7 — draft_424 regression
- Harness: `sandbox/caldiag_d424_regression.py`; report `tmp/caldiag_d424_regression/stage7_report.json`.
- Gate ON: **150/150** `VY_DKRSMP=SUM`, **0** WARN/FAIL, `cal_diag_keys=1`, `cal_diag_aborted_groups=0`;
  calibrated arrays byte-identical to baseline (`arrays_ok: true`);
  photometry `science_failures: 0` (`compare_photometry_science_meaningful`, setup `NoFilter_60_2`).
- Gate OFF: calibrated arrays byte-identical; **zero** `VY_DKRSMP`/`VY_CDSKY`/`VY_CDSTAT` on outputs.

### Stage 8 — docs + push
- ROADMAP CAL-DIAG -> implemented/validated; ledger rows CAL-AGE-CLOCK / RN-HEADER-NONE / CAL-PASSTHRU-DEAD.
- STATE + JOURNAL session entry.
- Commit: `3d1508b`; pushed after Stage 7 PASS.

## Errors (if any)
None blocking. Fixed invalid UTF-8 byte in `cal_diag.py:268` (saturation warn message).

## Files changed
- `calibration.py`, `cal_diag.py` (new), `pipeline.py`, `photometry_core.py`
- `config.py`, `ui_settings.py`, `param_resolver.py`, `database.py`
- `tests/test_cal_diag_gate.py` (new)
- `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_STATE.md`, `docs/VYVAR_JOURNAL.md`, `docs/VYVAR_PARAMS.md`
- `docs/VYVAR_CAL_DIAG_SPEC.md` (binding spec, prior session)

Commit range: `0268547` .. `3d1508b` (7 commits)
