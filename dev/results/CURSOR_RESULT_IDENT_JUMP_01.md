CURSOR RESULT - 2026-10-01 IDENT-JUMP-01
Commit: c834c81

What I did
Census + root cause + fix for catalog stars jumping onto
neighbours on aligned frames. Root cause unambiguous:
master-grid peak refine within ~2.5 FWHM (~14 px). Fix
ships on alpha-fixes-01. Default photometry path still needs
proc re-export to rewrite existing drafts.

## M1 - census
Artifacts: session_20261001_ident_jump_01/

### Draft 521 (131 frames, 611k star-frames)
median frac offset>3 px = 0.310; >8 px = 0.187;
frac x integer = 0.888.
Worst frame 036: 1558 >3 px, 1002 >8 px (matches architect).
By G (all rows): G<10 gt8=1.8%; 12-13=20%; 13-14=31%; 14-15=31%.
Forced rows: gt8=0 (bound still jumped some at 3-8 px before fix).
Targets with any gt8: 122. Comps with gt8: 2.
V1023 Her 1403049512185012992: 3 epochs jumped 14.65 px
(036, 110, 137) to (1456,995) from ref (1441.42,996.42).

### Draft 516 live (era06 product host; no era06 snapshot dir)
median gt3=0.328; gt8=0.255; x integer=0.993.
Targets with gt8: 129. Comps with gt8: 6.
Anchors BO / FW / GH: n_gt3=0, max offset 0.87 / 0.21 / 0.64 px.
AIJ-gated BO ensemble was NOT contaminated by this defect
(bright enough that peak lock stayed on target).

### Draft 516 era05 snapshot (era06 stand-in; no era06_* folder)
median gt3=0.251; gt8=0.148. Anchors likewise clean.

## M2 - root cause (unambiguous)
file:line:
  src_py/pipeline_catalog.py:_lock_matched_centroids_to_master_grid
  (introduced bad8c4b 2026-08-12 "master-grid centroid lock with
  local peak search"; extracted e803655).
Behaviour: on VY_ALGN frames, after DAO<->master match, every
matched row is moved to brightest pixel in a box of radius
  ceil(FWHM * search_fwhm) with search_fwhm=2.5
  -> ~14 px at FWHM 5.4 (exact V1023 jump).
Integer x,y: peak index is integer pixel.
Forced path: forced_photometry._bounded_peak_refine used the
same FWHM-scaled radius (default bound_fwhm=2.5).
gaia_dao_resid_px / vy_identity_gate: JOIN-copied from
MASTERSTAR in pipeline_catalog.py (~3565), not evaluated per
frame -> identical 0.5791 on every V1023 row.
Match radius (sky NN) is a separate step; the 14 px jump is
the peak refine, not the catalog match.

Present when AIJ gate measured (APERTURE-01c 2026-08-26): YES
(bad8c4b is earlier). Gate survived because BO/comps are bright.

Existing test test_master_grid_photometry.py previously
ASSERTED jumping to a neighbour peak (encoded the bug).

## M3 - impact (fixed master xy remasure, same aperture/comps)
m3_lc_before_after.csv:
  V1023 Her: RMS 99.6 -> 78.4 mmag; p2p 84.8 -> 61.5 mmag
    (3 jump events). Mag on jumped epochs was bright neighbour.
  Some always-jumped faint targets get noisier when fixed
    (honest: prior LC was the neighbour). Example NSVS
    J1548510: p2p 59 -> 24 mmag (improves).

## F - fix (shipped)
1. _lock_matched_centroids_to_master_grid: position = master
   float (x,y); optional peak refine bound = 1.0 px (cap 1.5),
   NOT 2.5*FWHM. Stated bound: align_residual_px on 521 is
   recorded as 0.0 (circular with jumped positions); bright-star
   anchors show <1 px, so refine_bound_px=1.0.
2. forced _bounded_peak_refine: same 1.0 px bound.
3. Unaligned WCS guard: cap allowed DAO shift at 1.5 px.
4. Per-frame gaia_dao_resid_px / vy_identity_gate stamped from
   offset vs master; MASTERSTAR join no longer overwrites them.
5. Tests: neighbour 14 px away -> stay on target (pre-fail of
   old test); sub-pixel within 1 px still allowed.

## LC-OUTLIER-01 reflag on 521
Full night class-total reflag requires proc CSV re-export with
the new lock (not run here - hours of catalog export).
V1023 simulation (fixed-pos remasure + evidence at master xy):
  before: normal=127, artifact=3, spike_unconfirmed=1
  after:  normal=130, artifact=0, spike_unconfirmed=1
  Jump epochs 036/110/137: artifact -> normal.
  Live LC on disk unchanged until re-export.
File: m3_v1023_reflag_sim.json

## era06 anchors
BO/FW/GH positions never jumped (>3 px). Era product LCs for
those anchors are not moved by this defect. Field-wide faint
stars on 516 ARE affected; a future era re-cut should re-export
proc after this fix. No LOCK in this task.

## Gates
--fast --clean: see session artifact.
Tests: test_master_grid_photometry.py PASS.

## STOP
Root cause fixed in code. Existing draft proc_*.csv still carry
jumped positions until re-export. Milan: schedule 521 (and 516
field) catalog re-export + LC rebuild + LC-OUTLIER reflag when
ready; AAVSO/VarAstro blocked on jumped faint targets until then.
