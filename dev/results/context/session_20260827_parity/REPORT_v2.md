# EXPORT-PARITY-01 v2 REPORT

Date: 2026-08-27. Architect: Claude. Implementer: Cursor.
Base: origin/consolidate-01 @ d9448e3 then 3a-1 9d19b6c.
Branch: consolidate-01. ASCII.
Tip at close: see git log; last code commit 793c4a5.

v1 STOP (b2aca4c) remains on the record. This v2 is control-flow +
config provenance, not physics. REPORT.md (v1) is the source for the
facts in the spec; no STOP line vs that report.

## Commits

| SHA | What |
|-----|------|
| 9d19b6c | 3a: NightRunParams fields; W2 defaults unchanged |
| 39b7f7f | 3a-2/3b/3c/3d: run_night_pipeline is the only night-run entry; FRAME-QC raise; photometry slice; snapshot cfg |
| 3fe1eb6 | 3a-2: W1 RUN VYVAR is a NightRunParams wrapper |
| 093ae22 | 3c: C3 calls run_ui_night_photometry with draft_id/db |
| af023a2 | 3e: ePSF merge restore-if-non-psf-changed + fire-proofs |
| 18c5840 | 3f: session_baseline_check --parity (G7) |
| d9eaba9 | 3e follow-up: stable non-psf hash across CSV dtype round-trip |
| 793c4a5 | 3f follow-up: sequential one-copy G7 (WinError 112 on dual freeze copy) |

## Invariants

INV-ONE-ENTRY-01: `night_run.run_night_pipeline` is the production
night-run entry. W1 (`app._run_vyvar_full_pipeline`) resolves optics,
location, validity days, and flat-fallback choices into
`NightRunParams`, calls `run_night_pipeline`, writes Streamlit state
from `NightRunResult`. C3 calls `run_ui_night_photometry` (same module)
with the page `draft_id` and `pipeline.db`.

INV-FRAME-QC-01: `_night_run_preprocess` else-branch (`quality_filter_draft_id`
is None) raises `RuntimeError` matching INV-FRAME-QC-01.
`stamp_frame_qc_provenance` writes `fwhm_limit_px` + `fwhm_limit_source`
to `night_run_qc_provenance.json`. Fire-proof:
`test_frame_qc_raises_without_quality_filter_draft_id` PASS.

INV-CFG-SOURCE-01: `resolve_cfg_for_photometry(..., existing_draft=True)`
overlays `provenance.config_snapshot`. Changed keys logged and stamped.
Fire-proof: `test_cfg_source_changed_key_on_rerun` PASS.
New draft (`existing_draft=False`) stays live cfg.

INV-EXPORT-READ-ONLY-01: `guarded_psf_sidecar_write` on
`stamp_p4_none_sidecar` and `merge_psf_into_sidecar`. Hash is pandas
column values (quoted header `flux,adu` fire-proof), not naive split.
Mismatch restores pre-image and raises. First `hash_pandas_object`
false-tripped valid merges after to_csv/read_csv (d9eaba9).

## M2 dispositions (code vs Milan table)

No STOP. W2 remains reference:

- optics: W1 resolves, `params.optics` (None = current W2)
- scan validity days: params or cfg default
- flat fallbacks: in `run_night_pipeline` when
  `apply_smart_plan_flat_fallbacks` (W1 True; headless default False)
- location: `params.location_id` + `location_source_hint`
- `roundness_reject_above`: shared, W1 from session / cfg default 1.25
- RAM QC PERF-10 skip: W2 logic unchanged
- MASTERSTAR TOP1 multi-group: W2 logic unchanged
- post-platesolve hook: params, default None
- completeness gate: shared in `run_night_photometry`; UI can FAIL loudly

## Fire-proof outputs

```
dev/tests/test_export_parity_01.py  9 tests PASS (plus ePSF merge 9 PASS)
FRAME-QC: call without quality_filter_draft_id -> RuntimeError INV-FRAME-QC-01
CFG: auto_fwhm_k_factor changed key in log and night_run_qc_provenance.json
C3 context: draft_id=516 plate_scale/site/calibration_mode logged before
            (draft_id=None) and after; C3 path equals C1 path (same triple)
EXPORT: quoted header hash; mutated dao_flux raises and restores bytes
```

C3 fire-proof three values (live DB 516, no MASTERSTAR in the unit test
so plate_scale may be cfg/DB-derived rather than WCS):

- BEFORE draft_id=None and AFTER draft_id=516 are logged by
  `resolve_photometry_context_triple`
- AFTER is what C3 now forwards (same function C1 uses)

G7 stamped params (identical W1 and W2):

```
cfg_source=live_no_snapshot
cfg_changed_keys=[]
plate_scale=9.773972549787624
site=draft:50.1121658:14.6982547:275.0
calibration_mode=vyvar_calibrated
n_lightcurves=253 n_frames=134
```

live_no_snapshot: `_copy_frozen_anchor_inputs` ignores photometry/, so
the freeze `pipeline_meta.json` config_snapshot is not in the G7 work
tree. Overlay is covered by the unit fire-proof, not by G7. W1 and W2
stamped the same source. Files: parity_stamped_params_w1.json /
parity_stamped_params_w2.json.

## Gates

| Gate | Status | Numbers |
|------|--------|---------|
| G1 before | PASS | d9448e3 --fast --clean 1588 passed, 32 skipped (v1 STOP tip) |
| G1 after | PASS | d9eaba9 --fast --clean 1597 passed, 32 skipped; clean-tree PASS. Log: g1_after2.txt. First after-hash-fix attempt flaked on test_database_sqlite_threading (isolated re-run 13 passed); not this diff. 793c4a5 is gate-script only; --full pytest at 793c4a5: 1597 passed, 32 skipped. |
| G2 after 3a | PARTIAL | 9d19b6c --full. Funnel PASS active_targets 253 skip_photometry 197. Counters phase2a_empty_comp_drop=3. Science-compare PASS n_lc=53. Aperture LC bytes 53/53 identical to era04. Core SHA FAIL: run de99975ae78994ee n=160 vs snap 9367f998 n=160. Diff set is exactly 53 PSF LC files (git_hash/git_dirty headers). Log: g2_after_3a_ascii.txt. |
| G2 after 3f | PARTIAL | 793c4a5 --full. pytest 1597 passed, 32 skipped. full-pipeline 1314s -> tmp/session_baseline/20260827T104844Z. PSF wrote 53 in 22s. Funnel PASS active_targets 253 skip_photometry 197. Counters phase2a_empty_comp_drop=3. Science-compare PASS n_lc=53 failures=0. Snapshot core SHA PASS 9367f998 n=160. Run core FAIL e31da59ecf2d1ac4 n=160 vs 9367f998 n=160. Extended FAIL 5f942cab9ca6f2f6 vs d3cefff3. File diff: 107/160 identical, 53/53 differ are lightcurve_*_psf.csv. Aperture LC bytes 53/53 identical. Do not retune 9367f998. Log: g2_after_3f.txt. Diff: g2_after_3f_sha_diff.json. |
| G3 ePSF ZP-OK | not re-run | product numbers unchanged by this task: BO 134/134 8.495 mmag, FW 134/134 5.218 mmag (EPSF-ZP-OK-01 WIRE v2 / era04 lock). No ePSF science path edit except merge write guard. |
| G4 live 516 | PASS | re-checked after G7: csv bfa24039778f437b / fits 13e77cf8a1dcb4e7 / epsf 172f95403beae36d (masterstars_full_match.csv, MASTERSTAR.fits, masterstar_epsf.fits). No live 516 writes. |
| G7 --parity | PASS | 793c4a5 sequential one-copy. W2 1319s, W1 1336s. core=bcfb64bc21083183... n=107 ext=cc8b532ee668b9b3... n=157. W1==W2. n=107 is aperture core (era04 160 minus 53 PSF LCs; G7 does not write internal PSF LCs). First dual-copy attempt FAIL WinError 112 (g7_parity.txt then rewritten by PASS run). Stamped params identical; no tune. |

## STOPs

None vs REPORT.md facts. None vs Milan M2 table.

G2 SHA vs 9367f998 cannot hold on a later git commit while
`write_internal_psf_lightcurves` embeds HEAD into `lightcurve_*_psf.csv`
and `compute_photometry_sha` globs `lightcurve_*.csv`. That is a
pre-existing gate/PSF-header coupling, not a 3a/3f W2 behaviour change.
Aperture LCs (the era04 science freeze) were byte-identical after 3a
and after 3f.

G7 first attempt STOP-for-disk (not a product STOP): dual freeze copy
hit WinError 112. Fixed by sequential one copy (793c4a5). Retry PASS.

## Files changed (production)

- src_py/night_run.py
- src_py/app.py
- src_py/ui_aperture_photometry.py
- src_py/epsf_psf_merge.py
- dev/scripts/session_baseline_check.py
- dev/tests/test_export_parity_01.py
- dev/results/context/session_20260827_parity/ (this report + gate logs)
