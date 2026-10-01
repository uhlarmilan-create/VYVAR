CURSOR RESULT - 2026-10-01 APERTURE-PERTARGET-01
Commit: 18257e1

What I did
Optional per_target aperture mode (brightness-dependent f*
per target; comps at the same f*). Default remains
f_fixed_night. UI toggle "Same aperture for all targets and
comps". Validation on drafts 516 and 521. STOP for Milan on
whether per_target becomes the default (era re-cut separate).

## Design (shipped)
- aperture_policy_mode += "per_target" (existing modes stay).
- aperture_f_grid default [0.75, 1.0, 1.25, 1.35, 1.5, 1.75,
  2.0, 2.5] (APERTURE-01b sampling).
- Shared core: src_py/aperture_pertarget.py
  measure_flux_on_f_grid / measure_night_grid /
  choose_f_for_target (Abbe p2p = std(diff)/sqrt(2);
  ties -> larger f).
- Phase 2A: when mode=per_target, measure grid once, pick f*
  per target, rebuild mag_inst for target+comps at that f;
  write aperture_per_target.json. Default path untouched.
- Physics D5-1: target and its comps always share one f*
  inside one differential LC.

## Tests
- T1 default mode unchanged: PASS
- T2 target+comps share r_ap: PASS
- T3 synthetic bright/faint/eclipse (p2p): PASS
- T4 config<->UI parity (registry + to_json): PASS
- test_aperture_pertarget_01.py: 5/5 PASS

## Validation (no default change; live LCs not rewritten)
Artifacts:
  dev/results/context/session_20261001_aperture_pertarget_01/

### Runtime vs f_fixed_night
| draft | stars | frames | grid_s |
| 516   | 143   | 134    | 1277.8 |
| 521   | 213   | 131    | 1141.7 |
f_fixed_night has zero grid overhead.

### f* vs magnitude (516)
| mag_bin | n  | median f* | median p2p gain mmag |
| G<12    | 14 | 1.00      | 3.72                 |
| 12-14   | 33 | 0.75      | 63.25                |
| 14-16   | 28 | 0.75      | 286.21               |
f* hist: 0.75:55, 1.0:4, 1.25:2, 1.35:1, 1.5:5, 1.75:1,
2.0:4, 2.5:3. Plots: fstar_vs_mag_draft516.png,
p2p_gain_vs_mag_draft516.png.

### f* vs magnitude (521)
| mag_bin | n  | median f* | median p2p gain mmag |
| G<12    | 21 | 0.75      | 8.83                 |
| 12-14   | 59 | 0.75      | 90.58                |
| 14-16   | 81 | 1.00      | 161.67               |
f* hist: 0.75:88, 1.0:19, 1.25:2, 1.35:3, 1.5:33, 1.75:1,
2.0:7, 2.5:8.

### BO CVn (516)
f*=1.0 (r_ap=5.192 px); p2p gain vs f=1.35 = 0.78 mmag;
dlevel vs f=2.5 = -4.67 mmag; rho(resid, seeing)=0.060.

### D5-1 (seeing-correlated residual)
Same-f construction. |rho| median 0.082 (516) / 0.098 (521);
frac |rho|<0.3 = 0.89 / 0.71. No systematic seeing leak
into the differential LC from mismatched radii.

### AIJ gate on BO (pinned C2..C6, annulus 2.7/5.2)
| mode            | f    | RMS(diff) mmag | gate 2.8 |
| f_fixed_night   | 1.35 | 2.887          | FAIL*    |
| per_target f*   | 1.0  | 5.255          | FAIL     |
*Remeasure on live 516; APERTURE-01d era04 gate was 1.95 /
~2.78. Relative: per_target f* worsens AIJ agreement on BO.

## Reading (for Milan)
per_target improves Abbe p2p by tens to hundreds of mmag
median for G>12 by preferring small f (~0.75-1.0); bright
G<12 gain is small (~4-9 mmag). This is the opposite of a
classic "brighter = larger aperture" ladder under the p2p
criterion (sky/contamination noise wins). AIJ accuracy on
BO degrades at the p2p-chosen f* (5.26 vs 2.89 mmag).
D5-1 residual vs seeing is small. Cost ~20 min grid/night
on these drafts. Recommendation: keep default
f_fixed_night; leave per_target optional until Milan
decides. If defaulted, era re-cut is a separate LOCK.

## Files changed
- src_py/aperture_pertarget.py (NEW)
- src_py/aperture_policy.py (MODE_PER_TARGET)
- src_py/config.py (aperture_f_grid + mode whitelist)
- src_py/photometry_phase2a.py / phase2a_target.py (wire)
- src_py/ui_settings.py (toggle)
- config.json (aperture_f_grid)
- dev/validation/params_registry.json
- docs/VYVAR_PARAMS.md, CONFIG_GUIDE_EN/CZ.md
- dev/tests/test_aperture_pertarget_01.py (NEW)
- dev/tests/test_ui_params_dashboard.py (owner 274)

## Errors (if any)
None blocking. AIJ remasure at f=1.35 on live 516 sits
0.09 mmag above the 2.8 gate (not a regression of this
task's default path). Converted PowerShell UTF-16
Tee/redirect artifacts to ASCII for ENCODING-POLICY.

## Gate
--fast --clean OVERALL PASS (1668 passed, 34 skipped).
Artifact: session_20261001_aperture_pertarget_01/fast_clean.txt

## STOP
Numbers above. Milan decides whether per_target becomes
the default; if yes, era re-cut is a separate LOCK.
