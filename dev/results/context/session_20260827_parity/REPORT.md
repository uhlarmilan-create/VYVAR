# EXPORT-PARITY-01 + FRAME-QC-PARITY-2 -- STOP

Date: 2026-08-27. Base SHA `45f3e5f`. Branch `consolidate-01` created
from that tip. CONSOLIDATE-01A is not in this tree (no branch, no
result file, G1..G6 not on disk). Section 0 re-verified at `45f3e5f`.

No production code changed. No photometry run. Frozen era04 and live
516 were not written.

## Premise (Rule 0.1)

Compared: C3 (Aperture Photometry page RUN) as it is today
(`run_full_photometry_pipeline` without `db`/`draft_id`) on era04
snapshot *inputs* versus the era04 freeze (core `9367f998` n=160,
134 QC-ok frames). Differ: P1 predicts 150 epochs because C3 omits
`draft_id`. The governing inputs for C3 are the existing
`proc_*.csv` files, not the 150-row `qc_metrics.csv`.

## Prerequisite

CONSOLIDATE-01A closed green (G1..G6): **ABSENT**.
`origin/consolidate-01`: **ABSENT** (only `main` `7c086e8` and
`sel-ghost-01` `45f3e5f`). This branch is a new line from `45f3e5f`
so the STOP report can be pushed as specified (`git push origin
consolidate-01`). 01A was not invented.

## 0. Callers re-verified at 45f3e5f

| ID | Site | draft_id / db |
|----|------|----------------|
| C1 | `app.py:651` RUN VYVAR | `db=pipeline.db`, `draft_id=int(_did)` |
| C2 | `night_run.py:1056` | `db=pipeline.db`, `draft_id=draft_id` |
| C3 | `ui_aperture_photometry.py:1340` | neither; `cfg` only |

`run_full_photometry_pipeline` def: `photometry_core.py:18562`.
Forwards `draft_id` at 18586, 18632, 18651, 18707, 18754
(plate scale, frame HW, phase0+1, phase2a). That forwarding is
**not** the frame-QC limit.

W1 `app.py:119` `_run_vyvar_full_pipeline`. W2 `night_run.py:495`
`run_night_pipeline` (docstring: mirrors W1). Module header
`night_run.py:6`: "app.py UI wrapper - deferred". `app.py` does not
import `night_run`. Confirmed.

## M1 -- P1 FAIL (STOP)

File: `m1_c3_inputs.json`.

era04 `qc_metrics.csv`: n=150
(`ok=134`, `rejected_prefilter_fwhm=16`).
era04 `proc_*.csv`: **134**. Aligned FITS: 135.

C3 discovers frames via `_find_phase2a_paths`
(`ui_aperture_photometry.py:133`) then globs `proc_*.csv` under
`detrended_aligned/lights/{setup}/`. Probe with
`draft_dir_override=era04`: setup `NoFilter_60_2`, **n_proc=134**.

Phase 2A enumerates those CSVs (`photometry_core.py:1216`, `9590`,
`17783`). It does not re-open the 16 FWHM-rejected lights.

P1 said 150 epochs. Measured input cardinality is 134. **P1 FAIL.**
Core hash was not re-run: 150 epochs is already impossible, and a
hash run would write into a work copy (task: measure first; STOP if
P1 fails).

### Refute: where frame QC actually lives

Not in `run_full_photometry_pipeline` from the DB draft record.

Preprocess, when `pending["quality_filter_draft_id"]` is set:

- W1 `app.py:816-821` `calibrated_paths_for_draft_apply_filters(..., fwhm_max_px=_fwhm_lim)`
- W2 `night_run.py:241-246` same

When `_dqf is None`, both wrappers skip the DB filter and enrich
all FITS in `source_dir`:

- W1 `app.py:870-882` (logs "draft_id chyba - FWHM limit sa neaplikuje z DB")
- W2 `night_run.py:280-289` (no equivalent log)

That is the "150 frames" mechanism: a **full-chain preprocess**
without `quality_filter_draft_id`, not C3 on an already-cut
snapshot. The 2026-08-27 handoff line about `draft_id=None`
admitting 150 frames applies to that preprocess else-branch, not
to C3's photometry call.

`pending["fwhm_limit_px"]` is computed in the wrappers
(`compute_auto_fwhm_limit` + DB light rows) and stuffed into the
MAKE MASTERSTAR job dict (`app.py:556`, `night_run.py:929`). It is
not a photometry-core cfg key read inside C3.

FRAME-QC-PARITY-2 as specified (`draft_id=None` raises inside the
photometry core) would not invert P1 on a snapshot copy (already
134 CSVs) and would not be the 150-frame full-chain bug (that is
preprocess `_dqf is None`).

C3's page **has** `draft_id` (`render_aperture_photometry` arg,
`ui_aperture_photometry.py:1101`) and uses it for path lookup.
The defect is that the RUN call at 1340 does not forward it. That
can still change plate-scale / site / crowding / calibration_mode
inside phase2a. It does not change epoch count on era04 inputs.

## M2 -- W1 vs W2 stage table (spec for a future `run_night`)

W1: `app.py:119` `_run_vyvar_full_pipeline`.
W2: `night_run.py:495` `run_night_pipeline`.
Existing name is `run_night_pipeline`, not `run_night`.

| Stage | W1 | W2 | Args |
|-------|----|----|------|
| optics resolve | `app.py:244` `resolve_working_optics` + `sync_optics_session` | absent | present-in-W1-only |
| scan | `app.py:263` `smart_scan_source` | `night_run.py:575` same | same shape; W1 validity days from UI args, W2 from cfg |
| flat fallbacks | `app.py:292` `_vyvar_apply_smart_plan_flat_fallbacks` | `night_run.py:595` `params.manual_flat_map` only | differs |
| observer site | `app.py:301` `cfg.observer_location_id`, hint `ui_selection` | `night_run.py:617` `params.location_id`, hint `cli_arg` | differs |
| import | `app.py:308` `smart_import_session` | `night_run.py:625` same | same |
| cal provenance | `app.py:331` | `night_run.py:659` | same |
| calibration | `app.py:366` `quick_calibrate_last_import` (no roundness kw) | `night_run.py:696` + `roundness_reject_above=params.roundness_reject_above` | differs |
| RAM QC | `app.py:399` always `run_draft_ram_calibration_qc_to_obs_files` | `night_run.py:738-763` skipped when PERF-10 `dao_qc_in_calibrate` | differs / present-in-W1-always |
| pointing | `app.py:412` | `night_run.py:767` | same |
| auto FWHM | `app.py:462-479` | `night_run.py:792-813` | same idea; W1 logs via `log_event` |
| MASTERSTAR TOP1 | `app.py:482` always global TOP1 | `night_run.py:815-878` skips global TOP1 if `draft_is_multi_group_obs` | differs |
| coords + hash | `app.py:529-550` | `night_run.py:881-905` | same |
| preprocess | `app.py:587` `_vyvar_execute_preprocess_pending` | `night_run.py:961` `_night_run_preprocess` | mirrors; W1 writes Streamlit session; W2 does not |
| platesolve | `app.py:593` `_vyvar_execute_platesolve_pending` | `night_run.py:967` `_night_run_platesolve` | mirrors; W2 takes `plan` |
| post-platesolve hook | absent | `night_run.py:981` `params.post_platesolve_hook` | present-in-W2-only |
| discover groups | `app.py:604` `_find_phase2a_paths` | `night_run.py:998` same | same |
| photometry | `app.py:651` C1 with db+draft_id | `night_run.py:1056` C2 with db+draft_id | same call; W2 wraps exceptions into list |
| completeness gate | absent | `night_run.py:1091` `audit_photometry_completeness` can FAIL the run | present-in-W2-only |
| PDF | `app.py:678` `generate_all_method_photometry_reports` | `night_run.py:1116` same | same |
| Streamlit state | many `st.session_state` writes | none | allowed wrapper delta |

C3 is not a wrapper of W1/W2. It is a third photometry entry
(`ui_aperture_photometry.py:1340`) on already-exported catalogs.

## M3 -- proc sidecar writers / readers

`_vyvar_df_to_csv` def: `pipeline.py:5934`.

### Writers of `proc_*.csv` (production)

| Site | When | INV-EXPORT-READ-ONLY |
|------|------|----------------------|
| `pipeline.py:10704` `_export_per_frame_run_catalog_core` | pipeline catalog export | pipeline-once (allowed) |
| `pipeline.py:11616` alternate export path | pipeline catalog export | pipeline-once (allowed) |
| `pipeline.py:16088` / `16146` `export_per_frame_catalogs` | `generate_masterstar_and_catalog` only callers in `src_py/` | pipeline-once (allowed) |
| `pipeline.py:13436..14728` later catalog stamp/fill in the same MASTERSTAR run | pipeline | pipeline-once (allowed) |
| `photometry_core.py:1290` sigma_bkg scaled rewrite | during photometry | pipeline photometry (allowed if counted as the run) |
| `epsf_psf_merge.py:179` `stamp_p4_none_sidecar` | F6 P4 stamp | **non-pipeline rewrite** |
| `epsf_psf_merge.py:304` `merge_psf_into_sidecar` | F6 PSF column merge | **non-pipeline rewrite** (additive psf_* only; still a writer) |

R5 / H2: `_export_per_frame_run_catalog_core` is `pipeline.py:10331`.
Today `export_per_frame_catalogs` is not called from UI pages; only
from `generate_masterstar_and_catalog`. The live non-pipeline writer
surface is **ePSF merge**, not a second full catalog export.

### Readers (production, not exhaustive tests)

Phase 0+1 / 2A glob `proc_*.csv` (`photometry_core.py`).
`ProcFrameStore`. Comp QA. PSF internal LC. AAVSO/VarAstro export
reads lightcurves, not a catalog rebuild. PDF reads products.

## Gates

| Gate | Status |
|------|--------|
| G1 --fast --clean | not re-run this STOP (no code); last PASS was APERTURE-01d lock at `45f3e5f` |
| G2 --full era04 | not run (no code) |
| G3 ePSF ZP-OK | not run |
| G4 live 516 | untouched (no writes) |
| G7 --parity | not added |
| G8 C3 n=134 + era04 hash | not run; n=134 is the C3 *input* count; hash not measured |

## Commits

This STOP: report + `m1_c3_inputs.json` only. 3a-3e not started.
Follow-up on this branch: APERTURE-01d `--full` logs were UTF-16-LE
(PowerShell redirect BOM 0xFF). Rewritten as ASCII so `--fast` can
pass; content unchanged.

## What a rewritten spec must use

1. Frame-QC authority is preprocess `quality_filter_draft_id` +
   `fwhm_limit_px` (`app.py:816`, `night_run.py:241`), not photometry
   `draft_id`. Raising in the core on `draft_id=None` does not restore
   134-vs-150 on C3-from-snapshot.
2. C3 still must not be a third photometry entry: it should call
   `run_night_pipeline` (or a photometry-only slice of it) with the
   page `draft_id` forwarded. That is INV-ONE-ENTRY-01, independent
   of P1.
3. INV-EXPORT-READ-ONLY-01: guard `epsf_psf_merge` writers (or
   classify PSF-column additive merge as in-run). Do not hunt a
   second `export_per_frame_catalogs` UI caller; there is none.
4. W1 vs W2 are not argument-identical (RAM QC skip, roundness,
   multi-group MASTERSTAR, completeness gate, optics resolve). 3a
   byte-identity of W2 before/after is the right first commit; W1
   cannot become a thin wrapper without those deltas being decided.

## Errors

P1 FAIL. CONSOLIDATE-01A missing. No 3a-3e.
