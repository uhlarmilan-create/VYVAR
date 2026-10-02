CURSOR RESULT - 2026-10-02 ALPHA-FIXES-02

Architect: Claude. Implementer: Claude Code.
Branch: alpha-fixes-01. Base: 630a51b.
Two commits: A = APERTURE-DYNAMIC-01; B = LC-FLAG-ERR-01.

==============================================================================
PART A - APERTURE-DYNAMIC-01
==============================================================================

What I did
Default aperture is now per_target (dynamic). UI toggle "Same aperture
for all targets and comps" defaults OFF. Production radius is
r = f* x FWHM_frame (FWHM-AUTH-01); night median only as fallback when
a frame has no VY_FWHM (logged). Grid extended with 0.5 and 0.6.
aperture_f_edge recorded per target; n_f_edge logged. Fixed phase2a
overwrite that collapsed per-frame r to night-median scale.

## Design
- aperture_policy_mode default = per_target (config.py, config.json)
- aperture_f_grid = [0.5, 0.6, 0.75, 1.0, 1.25, 1.35, 1.5, 1.75, 2.0, 2.5]
- fwhm_for_radius(per_target) uses frame FWHM like f_per_frame
- measure_night_grid / apply_grid_fluxes_to_frames: r = f* x FWHM_frame
- write_per_target_choices: policy APERTURE-DYNAMIC-01, n_f_edge,
  aperture_f_edge per target
- UI help updated; PARAMS + CONFIG guides + registry

## Tests
dev/tests/test_aperture_pertarget_01.py: 7/7 PASS
(+ aperture_policy_01 still PASS; 20 total with policy suite)

## Validation
Artifacts: dev/results/context/session_20261002_alpha_fixes_02/
  validate_aperture_dynamic_01.py
  validation_summary.json
  aperture_per_target_draft{516,521}.json
  fstar_table_draft{516,521}.csv
  fstar_vs_mag_draft{516,521}.png
  p2p_gain_vs_mag_draft{516,521}.png

### Runtime (grid)
| draft | stars | frames | grid_s | n_f_edge | n_fwhm_fallback |
| 516   | 143   | 134    | 1085.2 | 46       | 0               |
| 521   | 214   | 131    | 943.2  | 98       | 0               |

### f* vs G (516) - IDENT-JUMP-clean remeasure
| mag_bin | n  | median f* | median p2p gain mmag | n_f_edge |
| G<12    | 14 | 0.80      | 4.24                 | 6        |
| 12-14   | 33 | 0.50      | 80.47                | 24       |
| 14-16   | 28 | 0.55      | 301.17               | 16       |
f* hist: 0.5:42, 0.6:16, 0.75:3, 1.0:3, 1.25:1, 1.35:1, 1.5:1,
1.75:1, 2.0:3, 2.5:4. Median f*=0.5 (was stuck at 0.75 edge before
0.5/0.6 extension). Still 46/75 on a grid edge (mostly 0.5).

### f* vs G (521)
| mag_bin | n  | median f* | median p2p gain mmag | n_f_edge |
| G<12    | 21 | 0.75      | 8.69                 | 4        |
| 12-14   | 59 | 0.50      | 137.60               | 42       |
| 14-16   | 81 | 0.50      | 560.17               | 52       |
f* hist: 0.5:96, 0.6:11, 0.75:4, 1.0:5, 1.25:1, 1.35:32, 1.5:2,
1.75:2, 2.0:6, 2.5:2. n_f_edge=98.

### BO CVn (516)
f*=1.0 (not edge); p2p gain vs f=1.35 = 0.80 mmag;
dlevel vs f=2.5 = -4.14 mmag; rho(resid, seeing)=0.060.

### D5-1 (seeing-correlated residual)
|rho| median 0.087 (516) / 0.115 (521);
frac |rho|<0.3 = 0.79 / 0.65.

### AIJ gate on BO (pinned C2..C6, annulus 2.7/5.2)
| mode                         | f    | RMS(diff) mmag |
| f_fixed_night                | 1.35 | 2.887          |
| DYNAMIC default (per-frame r)| 1.0  | 5.245          |
Plain value for Milan: 5.245 mmag (prev night-FWHM f*=1.0 was 5.255).
No tolerance decision here.

### 516 product hashes (pre re-export; for era07 planning)
Do NOT LOCK. Anchors will move on IDENT-JUMP + DYNAMIC re-export.
  photometry_summary.csv
    sha256=92dc193d84ba520c7d7c7d6a5c7559fa7eb8c7d9cf307bb4fbf88be223d2fc72
  active_targets.csv
    sha256=c207f939a47a13f6edb8633b3a6dd7eaf1c8f2c544890da4753f3c584bc56e59
  comparison_stars_per_target.csv
    sha256=657edb2f39dd847bcbe16d9ea0b2aa6a49b5928748cea6ba9fc009ea2b24c632
  lightcurve_BO (1498613634033133184).csv
    sha256=a5ea3980cc7c978566ed1ad4f97cc3025c4dc80817c86ed4503ad1af7f90cabd

## Files changed (Part A)
- src_py/aperture_pertarget.py
- src_py/aperture_policy.py
- src_py/config.py
- src_py/photometry_phase2a.py
- src_py/phase2a_target.py
- src_py/ui_settings.py
- config.json
- dev/validation/params_registry.json
- docs/VYVAR_PARAMS.md, CONFIG_GUIDE_EN/CZ.md
- dev/tests/test_aperture_pertarget_01.py
- session_20261002_alpha_fixes_02/* (validation)

## Errors (if any)
None blocking. Many f* still land on the new lower edge (0.5);
n_f_edge reported for Milan (516=46, 521=98).

==============================================================================
PART B - LC-FLAG-ERR-01
==============================================================================
(pending after Part A commit)

## Gate
(pending --fast --clean / --full-epsf after Part B)

## STOP
Part A numbers above for Milan. Part B follows in next commit.
