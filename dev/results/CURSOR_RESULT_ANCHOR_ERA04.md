CURSOR RESULT - 2026-08-26T17:10:00Z

What I did
C6 ANCHOR RE-CUT era04 through C6-3d. Named [APERTURE-RADIUS]
from SNR-table artifacts (not inferred). P1 FAIL / P2 FAIL /
P3 PASS on BO/FW/GH (delta 0.0 mmag). Literature CONFIRMS
D5-1. Fix direction for Milan GO, not wired. era04 NOT locked.
era03 untouched. origin/main stays 7c086e8. No PUSH_AUTH SHA.

## Premise (Rule 0.1)
Compared: (C6-0) isolated c592ecf + A files + one declared 3-tuple
shim versus T3 R2. (C6-1/C6-3) HEAD full-chain 516 products in
draft_000516_snapshot_era04_20260826 versus frozen era03
draft_000516_snapshot_era03_20260820 (ad19e14; core 9902d918 n=121;
ext 472bc9e4 n=179). Differ: era04 is HEAD MASTERSTAR + photometry
on the live 134-frame QC allowlist; era03 is the 2026-08-20 freeze.

## C6-0 R1'' (informational; not a gate)

Label: "pre-B1 control, reconstructed with declared shim"

Shim, only in worktree copy of dao_gaia_stage_01_iter4.py
(.worktrees/c6_r1p_c592ecf), 24571 -> 24590 bytes:

old:
        det_to_g, gaia_owner_p1, _, _ = lock_existing_and_leftover_assign(
            dx, dy, gaia_g18, locked_pairs=None, leftover_radius_px=match_radius_px
        )

new:
        res = lock_existing_and_leftover_assign(
            dx, dy, gaia_g18, locked_pairs=None, leftover_radius_px=match_radius_px
        )
        det_to_g, gaia_owner_p1, _ = res[:3]

JSON: dev/results/context/session_20260826_c6/c6_0_r1pp_shim.json

R1'' then FAIL: KeyError 'frame' in
dao_gaia_stage_01.g2_empty_false_accept (empty_df has no frame
column; worktree lacks empty_positions_main.csv layout). No 60-row
table. C6-0-P1 untested. Skipped; did not STOP on C6-0.
JSON: c6_0_r1pp_skip.json. elapsed_s 49.9.

## C6-1 Full chain into era04

Path: Archive/Drafts/draft_000516_snapshot_era04_20260826
era03 still present. Live 516 SHA unchanged
(csv bfa24039 / fits 13e77cf8 / epsf 172f9540).

Calibration: production dark+flat, draft_id=None (live OBS_FILES
not mutated). TOP1 MASTERSTAR BO_CVn_Light_109.fits (same as live
manifest). Certificate passed.

First photometry FAIL INV-CAL-01 (cal_diag only at draft root).
Copied cal_diag/sat_diag/draft_manifest to platesolve/ and
platesolve/NoFilter_60_2/.

Preprocess with draft_id=None left qc_metrics status=ok for all 150
(fwhm_limit_px=0). Live/era03 admit 134 via rejected_prefilter_fwhm.
Stamped the same 16 live basenames onto era04 qc_metrics and dropped
those aligned products. MASTERSTAR kept. Live DB not mutated.
JSON: c6_1_qc_allowlist.json. Photometry then loaded 134 frames.

Phot-only: photometry 1446.2 s; PSF LC 19.8 s; 56 aperture LCs + 56
internal PSF LCs. INV-CAL-01 PASS after sidecar copy.

sha_core 961d590fd957a39ab4837ba82a2160a036d231dc5a288eefaf37bd3b2ac3cd32 n=169
sha_ext  59206a24319a796f5501de2eda14d8977981bf18046f53e821cd364fe296a5f1 n=222

INV counters:
- identity gate CSV vy_identity_gate: n=3600 ok=2623 warn=7 fail=0
- identity gate MS pipeline_meta (before phot-only overwrite):
  passes=2 ok=5238 warn=14 fail=0 n_lock_geometry_reject=0
- optimizer refit: rejected=false reason=ok n=2630
  p95_entry=1.3066 p95_candidate=1.2277
- D3 global pool drops: state=766 gate=0 resid=0 snr_ap_pixscaled=1094
  (n_in=2927 n_out=1067)
- zone_saturated_n=24
- suspected_variables_n=163

JSON: context/session_20260826_c6/c6_1_summary.json

## C6-3 Delta ledger era03 -> era04  STOP (no lock)

n_targets_union=60. n_unchanged_all_files=0. n_changed=60.

Named (closed list):
  [FRAME-29] 1
  [D3-D5]    2  (1498964240802993408 / 1500579870061241088 path)
  [C3-K5]    1  (1496315070616056064)
  [NAME-FIX] 1  (CSS_J134925 1497169940906156032)
  [DEPTH]    0  (1500387696044768384 absent on both freezes)
  [ZONE]     0  (zone_new_saturated empty)
  [ZP-OK]    0  (aperture LCs also changed)
  [D1-AC]    0  (not a constant 2.4 mmag overlay)

UNNAMED: 55 targets. Sanity FAIL: same-ensemble per-epoch mag_calib
is not 0 and not the AC overlay constant.

BO CVn 1498613634033133184 (AAVSO candidate):
  files aperture_lc;psf_lc; n_comps 4/4; ids_swapped empty;
  source files identical (134 frames, Light_002 etc. excluded both);
  mag_calib median +2.787 mmag std 1.555 mmag range -2.822..+6.887;
  mag_calib_final median +59.389 mmag; ac_correction delta 0.
  No closed-list tag.

FW CVn 1497343732462852864: 8/8 comps; median_dmag 10.836 mmag UNNAMED
GH CVn 1498804639818507904: 8/8 comps; median_dmag 6.814 mmag UNNAMED

CSV: c63_era03_era04_ledger.csv
JSON: c6_3_stop.json

Did not invent a tag. era04 not locked.

## C6-2 Determinism  PASS (does not lock without C6-3)

Two consecutive `--full` OVERALL PASS on HEAD d75440f against the
era04 tree (temporary SNAPSHOT_NAME point; reverted after C6-3 STOP
so the committed gate stays era03).

Run1b: full-pipeline 1352s; PSF 56 files 21s; core 961d590f n=169;
ext 59206a24 n=222; science-compare n_lc=56 failures=0.
Run2:  full-pipeline 1355s; PSF 56 files 21s; same SHAs.

First --full attempt crashed: science comparator read
lightcurve_*_psf.csv comment headers (ParserError). Comparator now
skips _psf/_adaptive (photometry_sha._lc_map). Not a science-file
diff.

C6-2 PASS. C6-3 STOP. Both gates required. era04 not locked.

Summaries: c62_full_run1b_summary.txt, c62_full_run2_summary.txt

## C6-3b Name the 55  STOP (measure only; no lock)

Premise (Rule 0.1): compared era04 tree
draft_000516_snapshot_era04_20260826 (HEAD 65eed01; core 961d590f)
versus live draft_000516 aligned FITS (M1 pixels/masters) and versus
era03 freeze draft_000516_snapshot_era03_20260820 (M2 pool / LC).
Differ: era04 photometry is HEAD; live aligned FITS mtime 2026-08-17;
era03 photometry is the 2026-08-20 freeze. M1 probe frames 010, 050,
109 (TOP1). No era04 lock. No config change.

### M1 PREPROC  elapsed_s 7.6

Frames 010 / 050 / 109 (era04 aligned FITS vs live aligned FITS):

  SHA equal: N / N / N
  max|dpix|: 0.0 / 0.0 / 0.0
  median dpix: 0.0 / 0.0 / 0.0
  aperture stamp around BO (x=957.66 y=822.44 r=5.499 px): max|dpix|=0

SHA differs on six WCS cards only (CRPIX1/2, PC1_1, PC1_2, PC2_1,
PC2_2); same six cards on all three frames. era04 vs era03 aligned
pixels also max|dpix|=0 (SHA unequal, same header-only class).

Masters era04 vs live cal_diag / draft_manifest / library:
  dark  CalibrationLibrary/Dark_60s_Dark_0G_-10deg_Bin1_20260422.fits
        SHA 525daf5c... library ID=86 REGISTERED_AT 2026-08-12T10:10:40Z
  flat  CalibrationLibrary/Flat_0.15s_NoFilter_0G_-10.5deg_Bin1_20260422.fits
        SHA 0667ed76... library ID=88 REGISTERED_AT 2026-08-12T10:12:13Z
  cal_diag dark_path identical. Library did not change since 2026-08-20.

Masters equal AND science pixels equal: no alignment/resample commit
to name. (a) is not pixels. BO per-epoch dmag vs max|dpix| correlation
undefined (dpix=0 every epoch). mag_calib +2.787 mmag vs era03 is not
an aligned-FITS pixel difference.

[PREPROC-REBUILD] REFUSED (BO, FW, GH, unnamed 55). Stays UNNAMED;
recut waits.

JSON: c63b_m1.json, c63b_m1_headers.json
CSV: c63b_m1_frames.csv, c63b_m1_bo_dmag_dpix.csv

### M2 POOL  elapsed_s 22.7

AC/ZP export comparison_stars.csv:
  era03 n=2240  G p10/p50/p90 = 11.19 / 13.33 / 14.40  BP-RP p50=0.864
  era04 n=150   G p10/p50/p90 =  8.84 / 10.35 / 11.94  BP-RP p50=0.854
  removed from era03 file: 2091 (G p50=13.43, faint). added: 1.
  Global BP-RP p50 shift -0.010: new pool did not introduce a colour
  bias at the field-pool level.

D3 1860 (governing C6-1 log, not the bbox-free reconstruction):
  n_in=2927 n_out=1067 removed=1860
  source_state=766
  snr_ap_pixscaled<10=1094  (reconstruction G p50=14.04, faint)
  vy_identity_gate=0
  gaia_dao_resid=0 (reconstruction found 3; log says 0)
  k*photon of the 1860: 0 (k*photon is after D3)

Reconstruction without variable-target/bbox cut: n_in=3291 n_out=1177
drops state=875 snr=1236 resid=3. Same shape (faint SNR + unmatched
state). source_state rows have no phot_g (n_G=0).

Then derived admission 1067 -> 534 (era04 comp_pool_admission.json;
faint_limit_g=10.66). era03 derived was 3114 -> 1352, same G~10.65
ceiling, no D3 in front.

k*photon among D3 survivors: 90 fail (G p10/p50/p90 = 9.21 / 13.29 /
14.13). Bright-decile median r = 4.749 (does not exceed 5).
G at which median r crosses 5: 9.516 (bright end, G 8.5-9.5 bins
have r_p50 9.0 and 6.6; then r falls below 5). BO four ensemble
members all r<3, none fail k*photon.

Recompute era04 mag_calib_final with era03 pool membership (script:
era04 mag_calib + era03 ct_bp_rp_comp_med * c1 + era04 ac):
  BO  obs +59.389 mmag -> replay +2.787 mmag  COLLAPSES
  FW  obs -10.617 mmag -> replay -10.836 mmag  does not collapse
  GH  obs  -8.238 mmag -> replay  -6.814 mmag  does not collapse

BO mechanism: ct_c1=-0.373 unchanged. ac_correction unchanged
(-0.129885). ct_bp_rp_comp_med 0.454638 -> 0.606386.
era03 comparison_stars.csv contains all 4 BO ensemble IDs; era04
file contains only 1499200223486564608 (bp_rp=0.606386), the reddest.
Weighted CT ref collapsed to that one star. Same 4-star ensemble
still used for mag_calib (ids_swapped empty).

+59 mmag cause = pool gate (CT membership), residual = mag_calib
+2.787 mmag (UNNAMED; not pixels).

JSON: c63b_m2.json  CSV: c63b_m2_drop_split.csv, c63b_m2_r_vs_g.csv,
c63b_m2_replay.csv, c63b_lc_census.csv

### M3 tags  (assigned or refused per target class)

[C3-FLOOR]     BO REFUSED  FW REFUSED  GH REFUSED  unnamed55 REFUSED
               Bright stars were not the 1860. No fix, no re-run.

[POOL-SNR]     BO ASSIGNED
               FW REFUSED   GH REFUSED
               unnamed55 PARTIAL: 12/56 aperture LCs are CT-dominated
               (|dct|>20 mmag) and take [POOL-SNR] for mag_calib_final.
               Field median dmag_calib=-10.3 mmag (FW-like) stays
               UNNAMED. 59 mmag goes to the ledger and methods paper
               as measured CT-ref offset of the old n=2240 pool vs
               the n=150 pool, not as a mag_calib ZP of the 4-star
               ensemble. Global BP-RP p50 did not shift; BO's
               effective CT ref did (1-star overlap).

[PREPROC-REBUILD]  all classes REFUSED
               Pixels equal; masters equal; no named input.

era04 still not locked. 55 unnamed not fully named: mag_calib
motion with identical pixels remains UNNAMED and the recut waits.

JSON: c63b_m3_tags.json  harness: c63b_measure.py
elapsed_s total 30.3 (M1 7.6 + M2 22.7).

## C6-3c measure X1/X3, fix X2  STOP (no lock)

HEAD cfffa82 (CT-REF helper). era04 tree on disk, not locked.
Photometry-only rerun wrote new LCs (core 233fce2e n=169) then
INV-DAG-01 at phase2a stamp (seq=6 vs already-stamped seq=7).
Science LCs + 56 PSF LCs are on disk. live 516 SHA unchanged
(csv bfa24039 / fits 13e77cf8 / epsf 172f9540). era03 present.

### X1 [WCS-APERTURE]  REFUSED

a) Governing: aligned frames (VY_ALGN). Call pipeline.py:8401-8410
   `_lock_matched_centroids_to_master_grid` :7491. DAO detect, match
   to MASTERSTAR, snap matched stars to MASTERSTAR x,y, then brightest
   pixel in ~2.5 FWHM window. Not per-frame WCS world2pix of Gaia.
   Unaligned: `_apply_dao_centroid_wcs_guard` :7455. Aperture photometry
   uses catalog x,y already written
   (photometry_core.py:enhance_catalog_dataframe_aperture_bpm:14136).

   Six WCS cards (CRPIX/PC) on era04 MASTERSTAR and aligned lights
   differ from era03/live. Origin: D2 accepted optimizer refit on
   era04 MS (c6_1_summary.json rejected=false, p95 1.307->1.228).
   Not S3 match.

b) BO+4 comps, FW+8, GH+8, all 134 frames: proc CSV dx=dy=|d|=0
   exactly. MS x,y differ ~0.001 px. |d| vs CRPIX distance: undefined
   (zero variance).

c) Predicted Gaussian flux-loss (FWHM 5.15, r=5.499)=0. Correlation
   dmag vs d_diff = nan. Does not explain 2.8/10.8/6.8 mmag.

d) COM in 3xFWHM box on era04 aligned pixels (identical to era03):
   res_e3_p50 = res_e4_p50 = 0.797 px, p95=1.66. Equal, not a
   regression. No D2/S3 WCS regression on aperture centres.

JSON: c63c_x1_x3.json  CSV: c63c_x1_xy_per_star.csv,
c63c_x1_xy_per_frame.csv, c63c_x1_dmag_vs_dxy.csv,
c63c_x1_centroid_truth.csv, c63c_x1_centroid_truth_per_star.csv

Measured residual (not named; not WCS-APERTURE): aperture_r_px
BO 5.999->5.499 all 134 frames, mag_inst +18.62 mmag, mag_calib
+2.787 mmag (ensemble absorbs most). FW/GH target r and mag_inst
unchanged; several comps r/flux changed -> mag_calib -10.836 /
-6.814. CSV: c63c_aperture_r.csv

### X2 [CT-REF]  ASSIGNED (fix + photometry-only)

ct_ensemble_reference_maps: per-target CT colour-ref is the ZP
ensemble, never comparison_stars.csv. Test
test_ct_ref_uses_full_ensemble_not_export_pool PASSED.
Commit cfffa82.

After phot-only: BO ct_bp_rp_comp_med 0.454638 (was 0.606);
ct_correction -0.001305 (era03 match). mag_calib_final median
+2.7875 mmag = mag_calib +2.787 mmag (predicted ~+2.8).
12 CT-dominated LCs: 10 collapse (dct=0 vs era03). Remaining 2:
1499209638054824320 D3-D5 CT skipped; 1500410236033012352 dct
-25 mmag (ensemble colour changed, n 2229->149). ct_n_comp column
still stores the field-fit n (2345); the colour-ref value is the
ensemble (log: comp_wmean bp_rp=0.455).

JSON: c63c_x2_rerun.json  CSV: c63c_x2_ct_collapse.csv

### X3  [EDGE]  (not STOP)

era03 n_lc=60, era04 n_lc=56, only_era03=4, only_era04=0.
Funnel: select_active_targets (photometry_core.py:14936). All four
in variable_targets; excluded_targets.csv reason=out_of_frame
(era04 out_of_frame n=176 vs era03 82). Chip enlarge then safe_bbox
annulus-aware intersection. y~1376-1394 on NAXIS2=1397.

  1485560025830226432  Gaia DR3 ... VAR   G=13.61  [EDGE]
  1496037650087948160  TIC 23815847 EXOPLANET G=13.98  [EDGE]
  1496733984545821696  FR CVn BY          G=11.14  [EDGE]
  1497491273179203456  Gaia DR3 ... VAR   G=13.99  [EDGE]

CSV: c63c_x3_era03_only.csv  JSON: c63c_x3.json

### X4 ledger v2  STOP

[WCS-APERTURE] REFUSED
[CT-REF]       ASSIGNED (mag_calib_final; 10/12 collapse)
[EDGE]         ASSIGNED (4 era03-only)
UNNAMED        42 including FW/GH mag_calib and field median;
               BO mag_calib +2.8 still UNNAMED (CT-REF named the
               +59, not the +2.8)

X1d not a regression. mag_calib 2.8/10.8/6.8 remain UNNAMED.
C6-4 lock skipped. C6-2 twice and C6-5 inventory not run.

CSV: c63c_era03_era04_ledger_v2.csv  JSON: c63c_x4.json

## C6-3d [APERTURE-RADIUS]  STOP (named; not locked; not wired)

Premise (Rule 0.1): era04 snapshot_20260826 photometry after
CT-REF vs era03 freeze snapshot_20260820. Aligned pixels identical
(C6-3b). Centres identical (C6-3c). Compared: aperture_snr_table.json
both trees, proc aperture_r_px, qc_metrics fwhm_px (era04; era03
file absent), photometry_plan.safe_bbox_px.

### X6a Governing  elapsed_s 0.18

BO mag=9.720 both eras, bin 9.5 both. Bin did not move. The table
at 9.5 moved 5.999 -> 5.499. proc_r matches table. Same for the
two bright BO comps (bins 10.0 and 9.5). Two faint BO comps
(bins 11.0, 11.5) r unchanged 3.999 / 3.499.

SNR-table inputs (artifacts, not inferred):

  fwhm_px            5.19465 = 5.19465  (dao_moment_median;
                     fwhm_px_scope=masterstar_header_vy_fwhm_dao_preferred)
  vy_fwhm_gauss      3.3014 = 3.3014
  gain / RN          0.7925 / 15.2 both
  zero_point         25.0 = 25.0; zp_cal ok=false reason=no_proc_csv both
  ee_path            measured_growth_curve both
  bound_hit 9.5      none / none
  sky_adu_per_px     1919.18 -> 2002.64
  bkg_var_adu2       1872.76 -> 1969.59
  ee_n_cog           59 -> 58
  ee_r90_px          6.0 -> 5.5

Ablation of table[9.5]:
  e3 EE + e4 sky/bkg     5.999  (sky/bkg alone do not move the bin)
  e4 EE + e3 sky/bkg     5.649  (EE curve is the primary mover)
  e4 full                5.499  (sky+bkg supply the remaining 0.15 px)

Named mover: measured growth curve (ee_r90 6.0->5.5, n_cog 59->58),
with a secondary sky/bkg_var term. NOT FWHM-AUTH-01 (fwhm_px identical).
NOT D1/D3-pool ZP (calibration failed both; ZP=25 both).

FW target bin 9.0 r stays 6.999; several FW/GH comps sit in bins
whose table r moved (8.5: 7.999->7.549; 9.5: 5.999->5.499). That
is the FW/GH mag_calib path (target mag_inst unchanged).

CSV: c63d_x6a_stars.csv  JSON: c63d_x6.json

### X6b Pre-registered tests  (no tuning)

P1 FAIL. Gaussian EE 2.5*log10(EE(6.0)/EE(5.5)) at qc_metrics
fwhm_px p50=5.192 px = +22.35 mmag. Obs mag_inst +18.62 mmag.
|pred-18.62|=3.73 > 3 mmag. Architect 21 mmag at FWHM 5.15 is
the same Gaussian model; production used measured EE (era04
interp EE(6)/EE(5.5) = +19.93 mmag, 1.31 from obs). Verdict:
Gaussian overpredicts; test as written FAIL.
elapsed in P1+P2 <1 s. CSV: c63d_p1_per_epoch.csv

P2 FAIL. Spearman dmag_calib vs qc fwhm_px:
  BO  rho=+0.026 (p=0.76)
  FW  rho=-0.277 (p=0.001)
  GH  rho=-0.124 (p=0.16)
All |rho|<0.5. Matches D5-1 Q1: radii are draft-constant, so the
2.8/10.8/6.8 mmag is a night-level offset, not per-epoch seeing.

P3 PASS on BO/FW/GH. Recompute era04 fluxes at era03 per-star
radii (photutils circular + era04 annulus sky; n_meas=13149;
33.4 s). Differential mag_calib vs era03:
  BO  obs +2.787 -> p3 0.000 mmag
  FW  obs -10.836 -> p3 0.000 mmag
  GH  obs -6.814 -> p3 0.000 mmag
34/40 of the v2 UNNAMED list collapse to ~0. Remaining 6: two
are 2.8 mag / 0.46 mag (not a radius effect);
1485534187306501376, 1485987151737107200; four have n=0
(missing proc rows). Cause of the 2.8/10.8/6.8 mmag class is
fully named [APERTURE-RADIUS].
CSV: c63d_p3_recompute.csv

### X6c Literature vs D5-1  CONFIRM

Howell 1989 PASP 101, 616: SNR-optimal radius is a SINGLE-STAR
S/N argument (CCD equation; max S/N near ~FWHM). The same paper
introduces CCD growth curves "as a means of correcting the flux
from faint or crowded sources, when measured with very small
(optimum) apertures." Optimum r WITHOUT COG is not what Howell
prescribed.

Stetson 1990 PASP 102, 932 (DAOGROW): growth curves are how
different apertures are put on one total-flux scale.

Everett & Howell 2001 PASP 113, 1428: survey LCs; "the aperture
size and corresponding light curve we choose to analyze depends
on the brightness of the star of interest." That is per-target
choice of which reduction to keep, not mixed radii inside one
ensemble ZP without COG.

Collins et al. 2017 AJ 153, 77 (AstroImageJ): one radius per
frame for all apertures -- fixed, or (FWHM factor) x (average
FWHM of all apertures in that image). Same r for target and
comps. Variable-r warning: do not use in crowded fields.

Naylor 1998 MNRAS 296, 339: optimal extraction is PSF-weighted,
not per-magnitude circular apertures.

Mighell 1999 (ASP CCD photometry): DAOPHOT-style growth curves
when apertures vary.

Zackay & Ofek 2017 ApJ: matched-filter / same PSF kernel across
the field. Cross-check: SNR-optimal measurement is a common
kernel, not a per-star radius table.

None of these supports per-magnitude-bin radii with COG off in
an ensemble. D5-1 (VYVAR_AUDIT_FINAL.md): radii track magnitude
bins from a draft-constant FWHM table; they do not track
per-frame seeing; COG default OFF. CONFIRM with the citations
above. VYVAR implemented Howell's SNR table and left Howell's
growth-curve correction off (`cog_aperture_correction_enabled=False`).

### X6d Fix direction  (NOT wired; Milan GO)

(i) One radius per frame for all stars = f x FWHM(frame), same f
    for target and comps (Collins 2017 / one-r reading of
    differential practice). f from the existing scatter ladder.
    Predict: mag_calib LEVEL 2.8/10.8/6.8 -> 0 (P3). Demeaned RMS
    almost unchanged (BO 145.7->146.0 mmag already; FW 13.78 vs
    13.82; GH 107.97 vs 107.93) because the defect is a constant
    offset (P2 |rho|<0.3). After (i) EE fraction is constant vs
    seeing; predict |Spearman rho| remains <0.3.

(ii) Keep per-star SNR radii and enable cog_aperture_correction
    with a measured growth curve per frame (Stetson 1990; Howell
    1989 as written). Predict: same level collapse if COG is
    complete; RMS unchanged at the 0.2 mmag level; residual
    seeing correlation only from COG-fit noise. Mixed-frame
    all-or-nothing gate already exists (APCORR-MIXEDFRAME).

Milan decides. era04 lock happens AFTER that change (one recut,
not two). Do not wire in this task.

### X6e [EDGE] rule

Input that changed: photometry_plan.json safe_bbox_px
  era03: null  (select_active_targets enlarges chip from max x,y)
  era04: [46.75, 46.75, 2034.25, 1349.25]
safe_bbox_r_out_px = 46.75 both. NAXIS 2082 x 1397 both.
era04 bbox = full chip minus r_out (not a tighter alignment
intersection; not MASTERSTAR WCS millipixels; not edge_margin=50).
HEAD writes/requires safe_bbox (pipeline.py require_safe_bbox).
era03 freeze plan left it null.

FR CVn y=1393.7, x=2078.8: outside y and x.
The other three: y=1376-1382 > 1349.25, x in-bounds.

Rule: a named variable whose sky annulus cannot fit on-chip
(y > NAXIS2 - r_out, or x similar) is out_of_frame when
safe_bbox_px is set. That is a product rule, not only a tag.
CSV: c63d_x6e_edge.csv

### Ledger v3

[APERTURE-RADIUS] assigned to BO/FW/GH and 34 mmag-class LCs
(P3 collapse). [EDGE] 4. [CT-REF] mag_calib_final. 6 remain
UNNAMED (two ~mag-class, four n=0). C6-4 lock still skipped:
Milan has not chosen (i) vs (ii); recut waits that change.

CSV: c63d_era03_era04_ledger_v3.csv
elapsed_s total 33.6 (X6a 0.18, P3 33.4).

## C6-4 / C6-5
Lock skipped (C6-3c STOP; C6-3d named APERTURE-RADIUS but fix
not wired). SNAPSHOT_NAME remains era03 (9902d918 / 472bc9e4).
INV-ANCHOR-00 pointer unchanged. No PUSH_AUTH SHA.

## Errors
C6-0 R1'' skipped (KeyError frame). C6-3 STOP unnamed + sanity.
C6-3b: D3 reconstruction n_in=3291 != log 2927 (bbox/variable-target
static filters omitted); governing 1860 split is the C6-1 log.
C6-1 first photometry INV-CAL-01 (fixed by copying cal_diag).
C6-3c phot-only: INV-DAG-01 phase2a seq backwards (max stamped
seq=7). LCs+PSF written; stamp failed at end of finalize_exports.
C6-3d P1 FAIL as written (Gaussian EE); P2 FAIL (D5-1 Q1).

## Files changed
- src_py/photometry_core.py (ct_ensemble_reference_maps; cfffa82)
- dev/tests/test_photometry_core.py (CT-REF test; cfffa82)
- dev/results/CURSOR_RESULT_ANCHOR_ERA04.md
- docs STATE / ROADMAP / JOURNAL / DECISIONS
- C6-3c artifacts (prior)
- C6-3d: c63d_measure.py, c63d_x6.json, c63d_x6a_stars.csv,
  c63d_p1_per_epoch.csv, c63d_p3_recompute.csv, c63d_x6e_edge.csv,
  c63d_era03_era04_ledger_v3.csv
- Archive era04 snapshot photometry (not git-tracked)
