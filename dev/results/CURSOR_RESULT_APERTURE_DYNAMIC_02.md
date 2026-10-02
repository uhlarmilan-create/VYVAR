CURSOR RESULT - 2026-10-02 APERTURE-DYNAMIC-02

Architect: Claude. Implementer: Claude Code.
Branch: alpha-fixes-01. Base: a42959e.
Class: FIX + validation. Dynamic aperture stays default.
No LOCK (era07 separate after Milan review).

==============================================================================
What I did
==============================================================================
Replaced Abbe p2p argmin with Howell S/N argmax for per_target f*
selection. Fine sampling grid 0.4..3.0 step 0.05. Flat top within 1%
of max S/N -> largest f. Documented fallback f=0.70 (Howell/Naylor
sky-limited midband) replaces undocumented grid-midpoint fallback
that caused the 521 spike of 32 stars at exactly 1.35. Abbe p2p kept
as a reported diagnostic only. Validated drafts 516 and 521.

## Design
- criterion: night-median growth F(f) / sqrt(Howell var)
- noise model: mirror of photometry_phase2a._howell_variance_adu2
  (src_py/photometry_phase2a.py ~L378-401): flux/g + sky_pp/g*area
  + (RN/g)^2*area; local copy in aperture_pertarget.howell_variance_adu2
  to avoid circular import
- DEFAULT_APERTURE_F_GRID = 0.4..3.0 / 0.05 (53 points)
- FALLBACK_F_STAR = 0.70 (documented; was len(grid)//2 -> 1.35)
- sky_pp stored per (frame, cid, f) from annulus
- gain/RN from draft gain_photon_transfer authority + RN=10 default
- write_per_target_choices policy=APERTURE-DYNAMIC-02 records
  snr_by_f, growth_F_by_f, sky_pp_by_f, reason, snr_max
- p2p_by_f retained as diagnostic

## Explain: 521 f*=1.35 spike (32 stars)
Undocumented fallback in choose_f_for_target when all p2p were
non-finite: f_star = f_grid[len//2]. On the DYNAMIC-01 grid that
index is 1.35. Removed. New fallback is FALLBACK_F_STAR=0.70 with
reason string. On 521 remeasure: n_fallback_f070=23 (at 0.70);
only 1 star at 1.35 (real S/N pick, not fallback).

## Explain: no_data=1885 (draft 521 class totals)
Census artifact: no_data_census_draft521.json
- 1885 epochs across 30 targets; flag_reason uniformly "no_data"
  (no finer edge/saturation taxonomy recorded)
- 9 targets all-NaN (1179 epochs = 9x131): present in
  active_targets/MASTERSTAR but ABSENT from every proc_*.csv
  (forced photometry never extracted) -> scaffold LCs with
  mag=NaN, flag=no_data. Expected for unrecovered extractions,
  not primarily edge/saturation.
- 21 targets partial (706 epochs): intermittent missing photometry,
  same flag_reason.

## Tests
dev/tests/test_aperture_pertarget_01.py: 11/11 PASS
  sky-limited Gaussian -> f* in 0.55-0.85, not edge
  photon-limited (brighter/lower sky) -> larger f*, not edge
  flat-top -> largest f within 1%
  fallback is 0.70 not 1.35
  default fine grid 0.4..3.0 step 0.05

## Validation
Artifacts: dev/results/context/session_20261002_aperture_dynamic_02/
  validate_aperture_dynamic_02.py
  validation_summary.json
  aperture_per_target_draft{516,521}.json
  fstar_table_draft{516,521}.csv
  fstar_vs_mag_draft{516,521}.png
  p2p_gain_vs_mag_draft{516,521}.png
  rms_gain_vs_mag_draft{516,521}.png
  no_data_census_draft521.json
  aij_equal_radius_settings_for_milan.json
  fast_clean.txt / full_epsf.txt

### Runtime (fine grid 53 f)
| draft | stars | frames | grid_s | n_f_edge | n_fallback |
| 516   | 143   | 134    | 5406   | 42       | 1          |
| 521   | 214   | 131    | 4997   | 75       | 23         |

### f* vs G (516) - expect rising with brightness
| mag_bin | n  | median f* | n_f_edge | med p2p gain | med RMS gain |
| G<12    | 14 | 0.575     | 0        | 0.96 mmag    | 0.23 mmag    |
| 12-14   | 33 | 0.40      | 26       | 77.8         | 57.9         |
| 14-16   | 28 | 0.40      | 16       | 241.8        | 253.3        |
Bright G<12 interior (product rule direction: larger than faint).
Faint stars still pile on lower edge 0.4: their Howell S/N(f) is
monotone decreasing from 0.4 (no interior max on this grid). Not the
old p2p "always follow the edge" failure mode for bright stars, but
n_f_edge is NOT ~0 overall. Report for Milan.

### f* vs G (521)
| mag_bin | n  | median f* | n_f_edge | med p2p gain | med RMS gain |
| G<12    | 21 | 0.70      | 1        | 7.9 mmag     | 3.9 mmag     |
| 12-14   | 59 | 0.45      | 27       | 138.3        | 137.9        |
| 14-16   | 81 | 0.40      | 47       | 586.0        | 584.4        |
Hist: 0.4:66, 0.45:32, 0.7:23 (fallback), 3.0:9; 1.35 spike gone.

### BO CVn (516)
f*=0.65 (interior; flat-top largest of n=4 within 1% of peak at 0.55);
p2p diagnostic gain vs f=1.35 = 0.18 mmag;
dlevel vs f=2.5 = -5.24 mmag; rho(resid, seeing)=0.057.

### D5-1 |rho|
|rho| median 0.080 (516) / 0.117 (521);
frac |rho|<0.3 = 0.81 / 0.68.

### Fair AIJ comparison (BO, pinned C2..C6)
AIJ gate table used Source_Radius=7 px fixed
(~f=1.35 at night FWHM~5.19). Unequal-radius was unfair.

| compare                              | RMS(diff) mmag |
| VYVAR f=1.35 night-r vs AIJ r=7      | 2.887          |
| VYVAR f*=0.65 per-frame-r vs AIJ r=7 | 6.844          |
| EQUAL: VYVAR forced r=7 vs AIJ r=7   | 2.889          |

Equal-radius recovers ~2.89 mmag. AIJ remasure at VYVAR f*=0.65
NOT run here (Claude Code cannot run AIJ). Settings for Milan:
  aij_equal_radius_settings_for_milan.json
  Source_Radius = 0.65 x FWHM_night ~= 3.37 px (or variable
  aperture factor 0.65); Sky keep 14/27 (or 2.7/5.2 x FWHM);
  same pinned comps C2..C6. STOP for that item.

## Gate
--fast --clean OVERALL PASS (1678 passed, 34 skipped).
Artifact: session_20261002_aperture_dynamic_02/fast_clean.txt

--full-epsf OVERALL FAIL (expected; S/N criterion moved anchors). NO LOCK.
Artifact: session_20261002_aperture_dynamic_02/full_epsf.txt
Key hashes for era07 planning (report only):
  full-pipeline PASS 9522s
  full-epsf-stage PASS n_stars=64 wrote 53 PSF LCs in 13914s
  era05_aperture snap (unchanged reference): 87197716af167132... n=53
  run core aperture: 3663adefb7be83eb... n=53  (MISMATCH vs snap)
  run ext_aperture:  a5b975b1b073b564... n=157 (MISMATCH)
  run core psf:      60e4f11755a53620... n=53  (MISMATCH vs epsf01)
  full-sha-v1-identity FAIL non-PSF v1 diffs n=103
  full-science-compare FAIL science_failures=50
  full-g3-residual FAIL (BO rms shifted under S/N aperture)

## Files changed
- src_py/aperture_pertarget.py (S/N criterion, fine grid, sky_pp,
  FALLBACK_F_STAR=0.70, snr records)
- src_py/photometry_phase2a.py (pass gain/RN into choose_f)
- src_py/config.py / config.json (fine aperture_f_grid default)
- src_py/ui_settings.py (help text Howell S/N)
- dev/validation/params_registry.json
- docs/VYVAR_PARAMS.md, CONFIG_GUIDE_EN/CZ.md
- dev/tests/test_aperture_pertarget_01.py
- session_20261002_aperture_dynamic_02/* (validation)

## Errors (if any)
None blocking. n_f_edge still high for faint stars because their
predicted S/N peaks at the lower sampling bound 0.4 (monotone fall
with f) - open question for Milan whether to extend below 0.4,
regularise growth curves, or accept small-f for faint targets.
Bright-star behaviour and 1.35 fallback spike are fixed as specified.

## STOP
Push alpha-fixes-01. Era07 LOCK is a separate step after Milan
reviews this result (and optionally remasures AIJ at f*=0.65).
