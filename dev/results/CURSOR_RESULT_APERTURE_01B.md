CURSOR RESULT - 2026-08-26T19:50:00Z

What I did
APERTURE-01b: measured f by accuracy (colour-flat EE + COG
flatness + AIJ), not scatter. Mode (a) f_fixed_night, 516
frame set, night QC FWHM = 5.191733 px (FWHM-AUTH-01).
Harness only; no recut; no lock. f* empty; rule not relaxed.

## Premise (Rule 0.1)
Compared: enclosed energy and photometric accuracy as a
function of f (r = f x night-median qc_metrics.fwhm_px) vs
the APERTURE-01 scatter-ladder default f=0.385228.
Differ: B3 selects the smallest f with dEE<0.01 AND
|level(f)-level(2.5)|<3 mmag (BO, FW, GH) AND AIJ RMS(diff)
<=3.3 mmag on BO; RMS is a tie-break only. Architect
outside-harness expectation: f* in 1.25-1.5 (AIJ 1.35).

## B1 Growth curves by colour
Comp pool n=149, BP-RP quartiles 0.736 / 0.854 / 1.185.
Q1 n=38 (iso 15), Q4 n=38 (iso 14); isolation 3 FWHM =
15.58 px. EE(r) = flux(r)/flux(3.0 FWHM), median of
per-frame quartile medians. Continuous f=0.50..3.00 step
0.05 (pixel step 0.260 px, not a 0.5-px grid). Annulus
4.75/9 x night FWHM (24.66 / 46.73 px). 134 frames.
Elapsed B1+B2: 143.8 s.

f     r_px    EE_Q1    EE_Q4    dEE      dEE<0.01
0.75  3.894   0.8279   0.7503   0.07764  NO
1.00  5.192   0.8998   0.8441   0.05571  NO
1.25  6.490   0.9427   0.9023   0.04042  NO
1.35  7.009   0.9548   0.9191   0.03567  NO
1.50  7.788   0.9683   0.9399   0.02842  NO
1.75  9.086   0.9832   0.9646   0.01860  NO
2.00  10.383  0.9918   0.9801   0.01178  NO
2.50  12.979  0.9985   0.9948   0.00372  YES

dEE<0.01 only at f=2.5. At the expected 1.25-1.5 band,
dEE is 0.040-0.028 (3-4x the gate). Full curve:
session_20260826_a01b/b1b_measure.json B1_curve.

## B2 Harness (BO/FW/GH, no lock)
Level = median mag_calib (mag_inst + frame ZP from pool
G - mag_inst). dlevel vs f=2.5 in mmag. AIJ product on
disk: dev/results/XVAL_AIJ_01_bo_compare.csv SHA
4ffa9e8e43b0736809eff132db959e399fed53a8ccc6b6006c9eb6c2660c7fc1
(AIJ one aperture r=7). FW AIJ not on disk; NOT MEASURED.

f    dBO    dFW    dGH     RMS_jnt  AIJ   col_b   |rho| BO/FW/GH
0.75 -57.1  -8.8   -544.0  93.18    10.45 414.8   0.088/0.213/0.098
1.00 -35.0  -11.4  -475.2  95.67    12.56 374.3   0.093/0.185/0.134
1.25 -19.4  -8.6   -443.3  100.64   14.16 360.4   0.088/0.214/0.132
1.35 -16.2  -7.1   -421.5  103.54   14.04 359.9   0.088/0.191/0.137
1.50 -15.7  -5.1   -377.4  108.53   11.05 362.3   0.099/0.137/0.165
1.75 -8.9   +0.5   -341.8  119.86   12.76 380.0   0.101/0.150/0.199
2.00 -9.2   +0.6   -227.0  136.74   12.66 413.4   0.095/0.193/0.199
2.50  0.0    0.0    0.0    208.88   15.36 468.4   0.082/0.243/0.174

Units: d* mmag; RMS_jnt = mean of BO/FW/GH demeaned RMS
mmag; AIJ = RMS(diff) mmag n=134; col_b = OLS slope of
(mag_inst - G) vs BP-RP, mmag per mag BP-RP (D10-1 style
resid vs colour; G = phot_g_mean_mag).

No f has AIJ RMS <= 3.3 (best 10.45 at f=0.75). No f
except 2.5 has |dlevel|<3 mmag on all three stars; GH
never flattens on this grid (crowding: GH RMS 119 -> 461
mmag as f grows, same direction as APERTURE-01 ladder).

## B3 Selection
Rule (written before the numbers): f* = smallest f with
ALL of dEE<0.01; |level-f2.5|<3 mmag BO and FW and GH;
AIJ RMS(diff)<=3.3 mmag on BO. Among survivors, lowest
joint demeaned RMS. Not relaxed in-session.

f* = NONE. Zero survivors.

## B4
Skipped (no f*). config aperture_fwhm_factor stays
0.385228. SNAPSHOT_NAME stays era03 (9902d918 / 472bc9e4).
C6-2 / ledger v5 / C6-4 lock not run. era03 untouched.
origin/main stays 7c086e8.

P-B1 / P-B2 / P-B3 were predictions at f*; not measured
(no f*). P-B2 honesty: FW AIJ still not on disk.

## Errors
None. Harness completed 134/134 frames.

## Files
- dev/results/context/session_20260826_a01b/b1b_measure.py
- b1b_measure.json, b1_dee_grid.csv, b2_f_grid.csv,
  b3_selection.csv
- this result file; STATE / ROADMAP / JOURNAL / DECISIONS
