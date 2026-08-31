# CURSOR RESULT - EPSF-CHAIN-01B

Date: 2026-08-31. Branch: consolidate-01. English. ASCII.

Base: origin/consolidate-01 @ da84d62. Live draft 516 was not written.
No refit. Work-copy product `tmp/epsf_chain_m2_era04` was still on disk.

## Premise (Rule 0.1)

What is compared: (1) automatic ePSF night-run vs config default OFF;
(2) G3 residual (psf_delta - ap_delta) vs the 746db0f raw psf_delta_mag
meter (145.917 / 14.557) and vs live regenerate 8.495 / 5.218.
How they differ: raw BO RMS is target variability; residual cancels it.
Refs 12.505 / 4.629 are measured on the existing work-copy LCs, not a
gate-fit. Live 8.495/5.218 is a different pool (67 vs 63).

## What I did

Closed EPSF-CHAIN-01 product rules: config default OFF, CLI requires
the same three inputs as UI RUN VYVAR, G3 residual meter, epsf01 ledger
accepted as the first ePSF repeatability anchor.

## A. Product rules

### A1 config key

No existing automatic ePSF night-run key. `psf_photometry_enabled`
(config.py:642, default False, config.json) is the Phase 2A psf_*
column switch, not the post-aperture stage.

Added `epsf_auto_run: false`.

| Where | Location |
| --- | --- |
| AppConfig field | src_py/config.py:646 default False |
| load | config.py:1649 |
| persist | config.py:2764 to_json |
| config.json | key with comment, value false |
| registry | dev/validation/params_registry.json `epsf_auto_run` |

`NightRunParams.epsf` is `bool | None = None` (night_run.py:78).
None = read `cfg.epsf_auto_run`. Explicit True/False overrides.
`run_night_pipeline` passes `epsf=params.epsf` (not `bool()`, which
would turn None into False without reading the key).
`be0ebfc` default ON is removed.

RUN VYVAR (app.py:230) does not pass `epsf=`, so it stays None and
follows the key (default OFF).

### A2 UI buttons (verify only, no rewire)

| Path | Call | Behaviour |
| --- | --- | --- |
| RUN ePSF job | app.py:1395 `run_epsf_stage(params=None, ...)` | always runs |
| Dashboard write LCs | ui_epsf_dashboard.py:393 `params=None` | always runs |
| C3 aperture | run_ui_night_photometry, epsf default None | follows config |

Fire proof: `test_run_epsf_stage_params_none_runs_when_config_off` and
`test_a2_explicit_true_forces_stage_when_key_off`.

### A3 CLI three inputs

`night_run.py` now has `parse_night_run_cli` / `main`. Required:
`--camera/--eq`, `--telescope/--tel`, `--site/--location` (or draft
manifest rig; site may also come from `observer_location_id`).
Refuse message names the missing input: camera / telescope /
observing site.

`simulate_night_run.py` silent 1/1 defaults removed; same helper.

### A4 gate cadence

`--full` / `--parity`: ePSF OFF; aperture era04_aperture d55fcc9d n=53
/ ext cc8b532e n=157.
`--full-epsf` / `--parity-epsf`: run stage, gate epsf01 c743b8ba n=53
and G3 residual. Lock-time. No `--parity-epsf` run this task.

## B. G3 residual

B1 landed in `src_py/epsf_zp_ok.py`: `residual_stats` is the single
implementation (census `meters_for_target` calls it;
`session_baseline_check` G3 calls `residual_meters_from_lightcurves`).

B2: existing work copy `tmp/epsf_chain_m2_era04`. Not cleaned. No
refit. Hours billed for B2: 0 (product already on disk).

| Target | n_full | coverage | level_offset_mmag | rms_mmag | demeaned_rms_mmag | vs live |
| --- | --- | --- | --- | --- | --- | --- |
| BO 1498613634033133184 | 134/134 | 1.0 | -9.454 | 14.142 | **12.505** | 1.47x of 8.495 |
| FW 1497343732462852864 | 134/134 | 1.0 | 10.349 | 11.019 | **4.629** | 0.89x of 5.218 |

Neither above ~3x live. No STOP. No tune. n_full=134 hard (B4).

B3 replaced 746db0f refs 145.917 / 14.557 with 12.505 / 4.629
(session_baseline_check.py G3_BO_REF_MMAG / G3_FW_REF_MMAG).

Raw zp_ok_current on this product is not used: one pinned comp is
fit_ok=False with finite flux, so strict pred yields n_full=0. LC
writer uses fit_ok_for_zp; residual from LCs matches census with
zp_ok_conv_finite.

## C. Ledger

C1 VL-ANCHOR-EPSF01 notes now include: "science validation pending;
external independent gate for ePSF (AIJ-class) on ROADMAP".
DECISIONS one line, date 2026-08-28, same wording.

C2 ROADMAP carry list: EPSF-XVAL-01 (MED) and EPSF-PERF-01 (FUTURE).
Neither implemented.

## G5 fire proofs

| Proof | Test | Result |
| --- | --- | --- |
| A1 default OFF | test_a1_default_off_no_stage_no_psf_completeness | PASS |
| A1 key ON | test_a1_key_on_runs_stage | PASS |
| A2 button | test_run_epsf_stage_params_none_runs_when_config_off | PASS |
| A3 camera | test_a3_missing_camera | PASS |
| A3 telescope | test_a3_missing_telescope | PASS |
| A3 observing site | test_a3_missing_observing_site | PASS |

## Gates

| Gate | Status | Detail |
| --- | --- | --- |
| G1 before | --clean PASS at da84d62. Dirty pytest FAIL was mid-edit race (config.json gained epsf_auto_run before registry). Not a product fail. | |
| G1 after | PASS 1628 passed, 32 skipped (dirty tree, then again on --full). | |
| G2 --full | PASS era04_aperture d55fcc9d n=53 / ext cc8b532e n=157. Pipeline 1975s. ePSF OFF (no full-epsf-stage / no G3). | |
| G3 residual refs | 12.505 / 4.629 from B2; n_full=134. Computed on work copy, not this --full tree. | |
| G4 live 516 | PASS csv bfa24039778f437b / fits 13e77cf8a1dcb4e7 / epsf 172f95403beae36d (re-checked after --full) | |
| G5 | PASS (table above) | |
| --parity-epsf | not required | |

## STOPs

None. Residual BO 12.505 is 1.47x live 8.495 (below 3x).

## Files changed

New: `src_py/epsf_zp_ok.py`, `dev/tests/test_epsf_chain_01b.py`,
`dev/results/context/session_20260828_epsf_chain_b/`.
Also config/registry/guides, night_run CLI, session_baseline_check A4/G3,
ledger/DECISIONS/ROADMAP.

Product commit on consolidate-01: `362043f`.
Push: `git push origin consolidate-01:consolidate-01`.
