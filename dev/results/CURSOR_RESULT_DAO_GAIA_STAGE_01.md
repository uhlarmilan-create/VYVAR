CURSOR RESULT - 2026-08-19 12:58 UTC+2

What I did
Built sandbox harness `tmp/dao_gaia_stage_01.py` (not imported by
production). Single-pass DAOStarFinder on star-masked sky-sigma
threshold, FWHM=5.3 px (header), opened sharpness (0,10), 3 px
greedy match, overlay PNGs + G1-G4 metrics. Ran full threshold
sweep {3.0, 3.5, 3.8, 4.5, 5.0} x sky sigma on draft 516
MASTERSTAR + Light_001/076/148. No src_py or config edits.

## Output / findings

### Part A - harness

| Artifact | Path |
|----------|------|
| Harness | `tmp/dao_gaia_stage_01.py` |
| Session ctx | `dev/results/context/session_20260819_daostage01/` |
| Full sweep log | `.../iteration_log_full_sweep.csv` (20 rows) |
| Best-config overlays | `.../thr4.5_sharp_open/{MASTERSTAR,Light_*}/overlay_*.png` |
| All iteration PNGs | `.../thr{3.0,3.5,3.8,4.5,5.0}_sharp_open/` |
| Metrics JSON | `.../metrics_all.json`, `.../best_config.json` |
| Sharpness diagnostic | `.../sharpness_kill_report.json` |

Runtime (Rule 0.3): full 5-threshold x 4-frame sweep **325.8 s**
wall; per-frame **6-28 s** depending on detection count.

### Part B - sky sigma vs GAIA-00 local annulus

MASTERSTAR (representative):

| Quantity | Value (ADU) | GAIA-00 reference |
|----------|-------------|-------------------|
| Star-masked sigma-clipped sky sigma | **39.8** | (new axis) |
| Local annulus std p50 | **41.8** | ~41.4 p50 |
| sky_mad_sigma | 23.7 | SNR-gate scale |
| Production rms_conv | 648.2 | 635.4 |
| thr @ 4.5 sigma | 179 | pass1 thr ~2402 |

Sky-sigma threshold is **~0.06 x rms_conv**, matching GAIA-00
diagnosis that pass1 uses star-contaminated convolved RMS while
pass2 uses sky-like local annulus (~40 ADU).

### Part B - sharpness kill (3c)

At thr=3.8 sigma, production sharpness (0,2) vs opened (0,10):
**0 / 1026** bright Gaia (G<=13) lost to sharpness cuts.
Missing bright circles are **not** explained by roundness/sharpness
filtering on this field.

### Part B - threshold sweep (MASTERSTAR)

| thr x sky | G1 <=13 | G1 <=14.5 | G2 | G3 | n_det | Verdict |
|-----------|---------|-----------|----|----|-------|---------|
| 3.0 | 94.7% | 92.6% | 2.9% | 35.1% | 6069 | FAIL all |
| 3.5 | 94.7% | 91.3% | 0.8% | 11.5% | 3665 | FAIL G1,G3 |
| 3.8 | 94.7% | 90.5% | 0.3% | 5.1% | 3128 | FAIL G1,G3 |
| **4.5** | **94.7%** | **87.5%** | **0.09%** | **1.64%** | 2633 | **FAIL G1,G3** |
| 5.0 | 94.7% | 83.4% | 0.09% | 1.63% | 2390 | FAIL G1,G3 |

G4: **0 unnamed** at all thresholds (every G<=15 star gets
DETECTED / BLENDED / EDGE / TOO_FAINT / SATURATED).

G2 empty-sky (INV-DET-FALSEFILL-01 main set, n=2200): PASS at
thr >= 3.5 on MASTERSTAR; Light frames not in empty_positions
CSV (G2 blank in log).

Best compromise on MASTERSTAR: **thr=4.5 x sky sigma** (G2 green,
G3 barely over 1%, G1 still short).

### Part B - G1 completeness curve (thr=4.5, MASTERSTAR)

| G bin | Complete | G bin | Complete |
|-------|----------|-------|----------|
| 8-10 | ~96-100% | 13-13.5 | ~85% |
| 10-11 | ~94-95% | 13.5-14 | ~78% |
| 11-12 | ~95% | 14-14.5 | **70%** |
| 12-13 | ~94% | 14.5-16 | 0.8-21% |

**G<=13 stuck at 94.7%** across all thresholds (54/1026 holes).
Plateau is not threshold-sensitive -> not fixable by sigma sweep
alone.

thr=4.5 state census (G<=16 on-chip n=6168): DETECTED 2238,
BLENDED 164, EDGE 714, TOO_FAINT 3052.

### Part B - pass 2 necessity (item 4)

**Single-pass does NOT meet Milan targets.** No sweep point reaches
G1 >= 99% (G<=13) or >= 95% (G<=14.5) with G2/G3 green.

Magnitude range needing local second pass:
- **G <= 13**: ~5% holes (BLEND/SAT/EDGE/TOO_FAINT mix; not
  threshold-limited).
- **13 < G <= 14.5**: primary gap (**70-85%** bin completeness);
  faint-end detection misses dominate.
- **G > 14.5**: intentionally below single-pass depth.

Pass 2 remains **necessary** (at least for G 13-14.5); deleting
it would sacrifice the 13-14.5 band where single-pass tops out
at 87.5% even before G3 is tightened to <= 1%.

### Part C - deliverables for Milan eye loop

Best-config overlays ready for review:
- `session_20260819_daostage01/thr4.5_sharp_open/MASTERSTAR/overlay_full.png`
- Same dir: `overlay_crop_{center,mid,corner}.png`
- Light_001/076/148 under `thr4.5_sharp_open/Light_*/`

Legend: green circle = DAO detection; blue dot = Gaia (G<=16);
red X = Gaia G<=14 without detection; violet = blend pair member.

**Verdict: sandbox harness PASS; no production config accepted.**
Iterate on Milan feedback (likely need targeted pass-2 / seed
for G 13-14.5 band while holding sky-sigma single-pass for
bright end).

## Errors (if any)

Initial run used too-narrow Gaia cone (4131 on-chip); fixed to
WCS-corner radius (~4.6 deg) -> 12611 on-chip / 6168 G<=16.
Full sweep log preserved as `iteration_log_full_sweep.csv` after
single-threshold overlay refresh overwrote `iteration_log.csv`.

## Files changed

- `dev/results/CURSOR_TASK_DAO_GAIA_STAGE_01.md`
- `dev/results/CURSOR_RESULT_DAO_GAIA_STAGE_01.md`
- `dev/results/context/session_20260819_daostage01/` (CSV/JSON/PNG)
- sandbox only: `tmp/dao_gaia_stage_01.py`

Push not authorized.

## Runtime (Rule 0.3)

| Part | seconds |
|------|---------|
| Harness build | (included) |
| Full sweep 5x4 | 325.8 |
| Best-config overlay refresh | 49.8 |
| **Wall** | **~376** |

Per-iteration MASTERSTAR: 6-18 s (thr 5.0-3.0).
