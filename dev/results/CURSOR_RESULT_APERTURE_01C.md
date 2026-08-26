CURSOR RESULT - 2026-08-26T21:15:00Z

What I did
APERTURE-01c (Milan GO f=1.35): set config, independent AIJ
gate PASS, recut era04 at f=1.35. Ledger v5 has 6 UNNAMED.
STOP. era04 not locked. C6-2 skipped. origin/main stays 7c086e8.

## Premise (Rule 0.1)
Compared: production one-r/frame at Milan GO f=1.35 (mode a,
night QC FWHM 5.191733 px, r_ap=7.0088) versus (1) AIJ BO
ensemble at the same r=7 with comps read from
XVAL_AIJ_01_Table.tbl, and (2) era03 freeze mixed-r photometry.
Differ: independent gate pins the AIJ C2-C6 set (includes
saturated C2); production BO is the 4-comp ensemble without C2.
APERTURE-01b pool-ZP AIJ RMS at f=1.35 (14.04 mmag) is not this
gate. Residual after the unconfounded gate is sky annulus +
centring + calibration. Derived expectation RMS(diff) <= 4 mmag.

## 1. Config
aperture_fwhm_factor = 1.35. aperture_policy_mode = f_fixed_night.
Scatter ladder is diagnostic only; r_min = 0.75 FWHM
(DEFAULT_R_MIN_FWHM / LadderSpec.radii_from_fwhm). Production
radii remain f x FWHM. test_scatter_ladder_r_min_is_075_fwhm added.

## 2. Independent AIJ gate PASS
XVAL_AIJ_01_bo_compare.csv has no comp list (csv_has_comp_list
false). STOP was not taken: XVAL_AIJ_01_Table.tbl has T1 + C2..C6
(RA hours). Matched to masterstars (max sep 5.819"):

  T1  1498613634033133184
  C2  1500748301498613248  (SAT-LIMIT C2)
  C3  1497771992240531712
  C4  1499200223486564608
  C5  1497974027502858240
  C6  1497368849430107904

rel_flux = T1 / sum(C2..C6), median-normalized mag RMS vs AIJ
rel_flux_T1. r_ap=7.0088 vs AIJ Source_Radius=7. Production
annulus 4.75/9 FWHM (24.66/46.73 px); AIJ sky 14/27 px is not
matched (part of the residual).

RMS(diff) = 2.7833 mmag  n=134  gate <= 4  PASS
median_diff = -1.2209 mmag; max_abs = 11.2977 mmag; elapsed 6.43 s.
Product SHA 4ffa9e8e (csv) / 133c5aac (tbl).
File: session_20260826_a01c/a1c_aij_gate.json

## 3. Recut era04 (candidate2 renamed, not deleted)
Trees:
  era03     draft_000516_snapshot_era03_20260820  UNTOUCHED
  cand1     draft_000516_snapshot_era04_candidate1_20260826  kept
  cand2     draft_000516_snapshot_era04_candidate2_20260826  kept
            (APERTURE-01 f=0.385 recut, renamed not deleted)
  era04     draft_000516_snapshot_era04_20260826  this recut

Catalog export under 8 workers left 3 frames without aperture_r_px
(001 enhance-fail restore; 004/012 29-col stubs; MemoryError on
annulus to_image). Repair copied those three proc CSVs from
candidate2 and re-enhanced sequentially. Guard: missing flux /
peak_max_adu no longer calls .to_numpy on a scalar nan.

All 134 proc_*.csv: unique aperture_r_px = 7.0088, f=1.35,
mode f_fixed_night, r_in=24.6607, r_out=46.7256.
fill_sigma: 134 skip (already empirical). Phase 2A 1143.6 s,
n_lightcurves=53, n_active=253, n_frames=134. PSF LC 53/53.
core SHA 988a2e13 n=160; ext e8fed401 n=210.
Live 516 SHA unchanged (csv bfa24039 / fits 13e77cf8 /
epsf 172f9540). Timings: repair 57.8 s, fill 3.4 s,
photometry 1143.6 s, psf 21.3 s, total 1226.3 s.

## Ledger v5 vs era03  STOP (6 UNNAMED)
n_union=60; n_era03=60; n_era04=53.
Named tags present: [APERTURE-01 f=1.35] with residual mmag;
[CROWDING] on GH with D11-1 pointer
(docs/VYVAR_LIMITATIONS.md D11-1 Dilution / crowding G proxy);
EDGE / FRAME-29 / D3-D5 / NAME-FIX / C3-K5 / CT-REF kept.
EDGE is only the C6-3c out_of_frame four, not every missing era04 LC.

Level residual (median mag_calib era04-era03):
  BO  1498613634033133184   +49.458 mmag  [CT-REF;APERTURE-01 f=1.35]
  FW  1497343732462852864   +28.213 mmag  [APERTURE-01 f=1.35]
  GH  1498804639818507904  -243.322 mmag  [APERTURE-01 f=1.35;CROWDING]

UNNAMED (no existing named tag applies; none invented):
  1498752516095473664 HAT-145-0004529
    era03 mag 0/134 (all no_data); era04 mag 59/134. dmag undefined.
  1498842882207281152 CSS_J135929.8+421520
    era03 mag 132/134; era04 mag 0/134 (all no_data). Lost LC at r=7.
  1499842372636900992 HAT-188-0003359
    era03 mag 71/134; era04 mag 0/134. Lost LC at r=7.
  1497227287309482624 Gaia DR3 ...
    era03 LC 134/134; era04 no LC (Phase1 n_survivors=2 < n_min=3,
    rms_violation). Not EDGE (annulus fits).
  1497245497969274240 HAT-188-0000323  same (no era04 LC).
  1498425548825498112 ASASSN-V J140619.34+422109.5  same.

Anything without a tag = STOP. C6-2 not run (would freeze a
failing ledger). SNAPSHOT_NAME remains era03
(9902d918 n=121 / 472bc9e4 n=179).

## 4. Lock / C6-5 / PUSH_AUTH
Lock skipped. C6-5 inventory:
CURSOR_RESULT_SESSION_CLOSE_20260826.md (this STOP; era04
products 988a2e13 / e8fed401 not the --full gate).
PUSH_AUTH SHA is the STOP commit on origin/sel-ghost-01
(named ref git push origin main:sel-ghost-01). Not for
origin/main. origin/main stays 7c086e8.

config f stays 1.35 (Milan GO). --full still photometers era03.

## Errors
Catalog export 3-frame miss under parallel MemoryError / missing
column .to_numpy. Repaired sequentially. AAVSO export failed on
CSS and HAT-188 (no exportable LC points). MASTER_SOURCES sqlite
malformed WARN (known, waived).

## Files changed
- config.json, src_py/config.py (default f=1.35)
- src_py/aperture_scatter_select.py (diagnostic r_min 0.75 FWHM)
- src_py/photometry_core.py (annulus from cfg into phase2a;
  scalar to_numpy guards)
- dev/tests/test_aperture_policy_01.py
- dev/validation/params_registry.json, docs/VYVAR_PARAMS.md
- session_20260826_a01c (gate, recut/repair, ledger v5)
- Archive era04 recut (not git-tracked); candidate2 kept
- docs STATE / ROADMAP / JOURNAL / DECISIONS
- this result file
