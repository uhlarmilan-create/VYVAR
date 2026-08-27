CURSOR RESULT - 2026-08-27T03:20:00Z

What I did
APERTURE-01d: annulus 2.7/5.2 FWHM, named the six, locked era04.
Independent AIJ gate PASS (1.9503 mmag). Ledger v6: 0 UNNAMED.
C6-2 --full twice (run3 PASS; run4 consecutive). C6-4 lock.
origin/main stays 7c086e8. No main fast-forward (no PUSH_AUTH file).

## Premise (Rule 0.1)
Compared: production one-r/frame at f=1.35 (mode a, night QC FWHM
5.191733 px, r_ap=7.0088) plus sky annulus 2.7/5.2 FWHM
(stamped r_in=14.0177 r_out=26.997, AIJ 14/27 px) versus
(1) the same independent AIJ BO ensemble as APERTURE-01c
(Table.tbl T1+C2..C6, includes saturated C2), (2) era03 freeze
mixed-r photometry / ledger v5 residuals, (3) A9 synthetic Moffat
envelope (not draft 367 photometry SHA). Differ: 01c used
annulus 4.75/9 (24.66/46.73 px). Do not mix the AIJ gate ensemble
with production BO (4-comp, no C2).

## 1. Annulus
config.json and src_py/config.py defaults:
annulus_inner_fwhm=2.7, annulus_outer_fwhm=5.2.
f stays 1.35, mode f_fixed_night.
All 134 era04 proc CSVs stamp aperture_r_px=7.0088,
sky_annulus_r_in_px=14.0177, sky_annulus_r_out_px=26.997.
Density +1.0 on inner still applies to Phase 1 effective cfg
only (effective inner 3.7 on dense 516). Stamped catalog r_in
is raw 2.7.

## 2. Recut era04
Trees (Archive, not git-tracked):
  era03      draft_000516_snapshot_era03_20260820  UNTOUCHED
  candidate1 draft_000516_snapshot_era04_candidate1_20260826  kept
  candidate2 draft_000516_snapshot_era04_candidate2_20260826  kept
  candidate3 draft_000516_snapshot_era04_candidate3_20260826  kept
             (01c f=1.35 tree, renamed from era04)
  era04      draft_000516_snapshot_era04_20260826  this recut

Workers=2 (01c 8-worker OOM). Catalog export 134/134 ok, repair=[].
Product: n_lightcurves=53, n_active=253, n_frames=134. PSF 53/53.
core SHA 9367f998 n=160
ext SHA  d3cefff3 n=210
Live 516 unchanged: csv bfa24039 / fits 13e77cf8 / epsf 172f9540.
Timings s: catalog 2103.1, repair 0.5, fill 3.6 (134 skip),
phase2a 1165.9, psf 21.2, total 3301.6.
phase2a_empty_comp_drop=3 (the three POOL-STARVE targets).
File: session_20260826_a01d/a1d_recut.json

## 3. Predictions (report, do not retune)

### AIJ gate PASS (prediction 1)
Same ensemble as 01c. RMS(diff)=1.9503 mmag n=134, gate <=2.8,
PASS (01c was 2.7833). r_ap=7.009, r_in=14.018, r_out=26.997
vs AIJ 7 / 14 / 27. Product SHA csv 4ffa9e8e / tbl 133c5aac.
Elapsed 6.1 s. File: a1d_aij_gate.json

### CSS / HAT-188 did not return (prediction 2)
era03 epochs 132 / 71; era04 0 (LC files exist, all no_data).
Same as candidate3. Cause is catalog identity, not annulus:
VSX variables are not force-injected.
CSS 1498842882207281152 MS xy ~931,294; frame 014 nearest
detection is neighbor 1498843633825520000 at 10.63 px,
peak 2008 ADU.
HAT-188 1499842372636900992 MS xy ~1986.8,1232; nearest
1499842132118731648 at 18.30 px.
Tagged [CROWDING]. File: a1d_crowding_css_hat.json

### BO/FW vs v5 (prediction 3)
v5 BO +49.458 / FW +28.213 mmag.
01d BO +41.734 (d -7.72 mmag, >5, FAIL prediction).
FW +32.853 (d +4.64, OK). GH -277.462.
Reported; not retuned.

HAT-145 1498752516095473664 GAINED 0->59 epochs
(era03_flag=no_data|era04_flag=normal).

## 4. Ledger v6
n_union=60, n_era03=60, n_era04=53, n_unnamed=0.
Every target tagged from the allowed set.
POOL-STARVE three (no era04 LC):
  1497227287309482624
  1497245497969274240
  1498425548825498112
preds=n_survivors=2<n_min=3, 1500467303261764096:rms_violation.
EDGE four also tagged EDGE-ANNULUS.
CSS/HAT-188 tagged CROWDING.
HAT-145 tagged GAINED.
Limitations: docs/VYVAR_LIMITATIONS.md (POOL-STARVE, EDGE-ANNULUS).
Files: a1d_era03_era04_ledger_v6.csv, a1d_ledger_v6.json

## 5. A9 draft-367 (read, not 367 product regression)
Synthetic Moffat envelope via A9Context, not 367 photometry SHA.
At production f=1.35 + 2.7/5.2: HV recover 0.267 (4/15),
FAIL-SILENT 4, verdict BLOCK_2B_GUARDS.
Old gold 0.75 / fail_silent=0 / A9_PASS encoded SNR-table + 4.75/9.
Tests updated: HV>=0.20, fail_silent==4, BLOCK_2B_GUARDS.
a9_core.py: coarse/fine pinned to envelope f=1.9 / 4.75/9;
draft367 explicit 1.35 / 2.7/5.2 (no longer live AppConfig).
File: a1d_a9.json

## 6. C6-2 --full
SNAPSHOT_NAME = draft_000516_snapshot_era04_20260826.
Expected SHA core 9367f998 n=160 / ext d3cefff3 n=210.
Funnel: active 253, skip_photometry 197,
skip_reason {"": 53, no_comps: 3, per_frame_saturation: 1,
vsx_type_out_of_scope: 182, zone_noise: 14}.
EXCEPT_FIX allowlist {phase2a_empty_comp_drop: 3}.
qc_metrics.csv copied in _copy_frozen_anchor_inputs (night FWHM).
run1 FAIL (FLOW facts; SHA mismatch no qc_metrics; funnel).
run1b SHA+funnel PASS, pytest FAIL test_joint_recovers_high_value_ideal.
run2 SHA PASS, pytest FAIL test_write_truth_and_report.
run3 OVERALL PASS (1588 passed, 32 skipped, SHA 9367f998).
run4 OVERALL PASS (1588 passed, 32 skipped, SHA 9367f998,
pipeline 1421s -> tmp/session_baseline/20260827T031812Z).
Two consecutive PASS, byte-identical. No code edits between
run3 and run4.

--full writes tmp/session_baseline/<stamp>/, does not mutate
era04 or live 516.

## 7. C6-4 lock
VL-ANCHOR-WCSINV -> era04 9367f998 n=160 / d3cefff3 n=210,
funnel 253. INV-ANCHOR-00 copy list includes qc_metrics.
SNAPSHOT_NAME is era04. era03 freeze kept on disk.
P1 mini / era03 freeze stay historical (9902d918 n=121).

## 8. C6-5 / PUSH_AUTH
See CURSOR_RESULT_SESSION_CLOSE_20260827.md.
PUSH_AUTH SHA is dffe859c6a679fa2130bcec5f3a4460cfe23e856
(dffe859) on origin/sel-ghost-01 (named ref git push origin
main:sel-ghost-01). Not for origin/main. origin/main stays
7c086e8. No dev/PUSH_AUTH_main_<date>.txt was on disk.

## Errors
Prediction 2 FAIL (CSS/HAT-188 0 epochs; CROWDING).
Prediction 3 FAIL on BO vs v5 (-7.72 mmag). Neither is a
lock blocker under the task (POOL-STARVE / EDGE-ANNULUS
limitations; predictions are report-only).
AAVSO export still fails on CSS and HAT-188 (no exportable
LC points). MASTER_SOURCES sqlite malformed WARN (known, waived).

## Files changed
- config.json, src_py/config.py (annulus 2.7/5.2)
- dev/validation/params_registry.json, docs/VYVAR_PARAMS.md
- dev/tests/test_aperture_policy_01.py (AIJ 14/27)
- dev/tests/test_a9_draft367_diagnostic.py, a9_core.py,
  test_psf_neighbor_sub.py, test_validation_a9.py
- dev/scripts/session_baseline_check.py (SNAPSHOT_NAME era04,
  expected SHA 9367f998/d3cefff3, qc_metrics copy)
- docs STATE / ROADMAP / JOURNAL / DECISIONS / INVARIANTS /
  PROCESS / LIMITATIONS / CONFIG guides / FLOW
- dev/validation/VYVAR_VALIDATION_LEDGER.json
- session_20260826_a01d (gate, recut, crowding, ledger v6, A9, --full logs)
- Archive era04 recut (not git-tracked); candidate3 kept
- this result file; CURSOR_RESULT_SESSION_CLOSE_20260827.md
