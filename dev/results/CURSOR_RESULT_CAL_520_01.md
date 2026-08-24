CURSOR RESULT - 2026-08-24 21:40 UTC+2

What I did
Measured why draft 520 g_60_4 V0612 Cam photometry is a wreck versus
the 2026-06-16 run of the same Brno AZ800 / C5A-150M observation.
Measurement only: no reclassification, no recalibration, no gate or
selection changes. HEAD `505fa1334fa7be7fa1dc49611c49e29972e0320b`
(is the W6 tip; descendant of itself). Declared dirt OK (AC-02 wiring
unstaged, unrelated). Not pushed.

## Premise (Rule 0.1)

**What is compared:** today's draft 520 `g_60_4` V0612 Cam
(`catalog_id` `1111749368289526912`) aperture photometry versus the
operator's 2026-06-16 UI snapshot of the same rig+observation
(drafts 399/400/402/407/409 products are gone).

**How they differ:** today `lc_rms=0.394883` and `lc_rms_ooe=0.080252`
(`photometry_summary.csv`), comps G 14.48-16.86 with `comp_rms`
0.0165-0.0339, `comp_path=sparse_fallback`. June snapshot: `lc_rms`
0.0622, comps G 11.6-13.9 with `comp_rms` 0.005-0.012, time axis
"BJD (TDB)". JOURNAL 2026-06-16 draft_409 (same field, products gone)
recorded `lc_rms_ooe ~0.006`, pre-eclipse RMS ~0.010, 8 comps, trust
GREEN. Full `lc_rms` includes the EA eclipse on both reductions; the
operator pair is 0.3949 vs 0.0622. Like-for-like ooe, if June ~0.006
is used, is 0.080 vs 0.006.

## Gates

| Gate | Status | Evidence |
|------|--------|----------|
| G1 tip descendant of `505fa13` | PASS | HEAD **is** `505fa13` |
| G2 516 production ePSF SHA | PASS | `172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20` unchanged (`masterstar_epsf.fits`) |
| `--fast` | see end | run after this write-up |

## M1 - classification chain (file:line)

**520 is `pre_calibrated` because the UI button `[folder] RUN VYVAR (non-cal)` was clicked.** Not folder layout, not a FITS header card. Folder layout is the *consequence*.

1. `src_py/app.py:2244-2256` second button; help text says it skips bias/dark/flat and treats source frames as already calibrated.
2. `src_py/app.py:2256` `_pre_cal = bool(run_vyvar_nc_clicked)`.
3. `src_py/app.py:2290` `pre_calibrated_mode=_pre_cal`.
4. `src_py/app.py:229` `_calibration_mode = CALIBRATION_MODE_PRE` (`"pre_calibrated"`, `draft_provenance.py:16`).
5. `src_py/app.py:291-294` skips `_vyvar_apply_smart_plan_flat_fallbacks`; calls `apply_pre_calibrated_import_plan(plan)` (`draft_provenance.py:733-745`: `quick_look=True`, dark/flat masters cleared).
6. `src_py/app.py:355-362` **skips** `pipeline.quick_calibrate_last_import`. Log: `"Pre-calibrated mode: skipping calibration - downstream reads {non_calibrated/lights}"`.
7. Downstream lights root: `draft_provenance.py:720-730` `resolve_draft_lights_root` returns `non_calibrated/lights` when `is_pre_calibrated_draft` (`:710-717`) reads manifest `calibration_mode` via `resolve_calibration_mode` (`:1134-1156`).

Live 520 `draft_manifest.json`: `"calibration_mode": "pre_calibrated"`, `"is_calibrated": 0`, files `calib_type: RAW_NON_CALIBRATED`, `paths.lights` under `non_calibrated/lights`. Infolog `infolog_20260824_204055.txt` 18:40:53: `[PHASE] Scan Source + Import (pre-cal passthrough)`, `[RUN VYVAR (non-cal)]`, `Pre-calibrated mode: skipping calibration`. Site resolved (Zdanice id=5); not a location miss.

**June-era key (partially recoverable):** drafts 399/400/402/407/409 are gone. JOURNAL 2026-06-14 draft_400 `g_60_4` **production-path PASS** is the `! RUN VYVAR` button (calibrate + analyze), not `(non-cal)`. JOURNAL 2026-06-16 draft_409 is the V0612 UI run with GREEN trust. Whether that production path applied VYVAR dark/flat or ingested already-calibrated lights is **not recoverable** from remaining logs/library (M3: no AZ800 masters exist now). Do not invent June Gaia IDs; none remain in-repo.

## M2 - radiometric evidence

Artifacts: `dev/results/session_20260824_cal_520_01/m2_radiometric.json`, `m2_large_scale_compact_mask.json`, `two_light_ratio_g0000_g0096.png`.

### M2.1 large-scale structure (star-masked, 50/80/150 px)

Compact-core star mask (high-pass 2 px, 8-sigma, dilate 8). 520 g lights are raw C5A-150M bin4, 3552x2664, pedestal ~3.3e4-3.6e4 ADU. 516 reference is a **different camera/FOV** (QHY294PROM bin2, VYVAR-calibrated `BO_CVn_Light_001.fits`).

| Frame | sky median ADU | p01 ADU | lp80 p99-p1 ADU | p99-p1 / median | p99-p1 / (p50-p01) |
|-------|----------------|---------|-----------------|-----------------|---------------------|
| 520 g_0000 (20:02 UTC) | 35746.6 | 35709.4 | 27.0 | 0.00075 | 0.727 |
| 520 g_mid | 33317.0 | 33311.8 | 1.39 | 0.000042 | 0.269 |
| 520 g_last (22:13 UTC) | 33305.5 | 33300.9 | 1.02 | 0.000031 | 0.222 |
| 516 Light_001 (calibrated) | 2410.7 | 2283.1 | 68.2 | 0.0283 | 0.535 |

Reading, not adjectives:
- Relative to the raw pedestal, 520 looks *flatter* than 516 (0.075% vs 2.8%). 516 is not a same-rig control; it still carries 68 ADU of 80-px structure on a 2411 ADU sky.
- 520 structure **tracks illumination**: 27 ADU on the twilight-bright first frame, ~1 ADU after sky drops ~2400 ADU. That is the multiplicative-flat signature (dust/vignetting lit by sky, not an additive bias pattern).
- p50-p01 on g_0000 is only 37 ADU; the 27 ADU low-pass is 73% of that illumination range.

### M2.2 two-light ratio (small dither)

Measured star xcorr shift last minus first: **dx=+1 px, dy=-2 px** (donut scale 50-200 px, so "double at offset" is not testable). Median-scaled pixel ratio g_0096/g_0000: lp80 rms **0.000134**, p99-p1 **0.000674**. Star-aligned ratio is the same (0.000133 / 0.000672) as expected at 2 px. Plot: `two_light_ratio_g0000_g0096.png`. Number: remaining ratio structure ~9e-4 of unity after scale; dither too small to separate pixel-fixed donuts from a slowly changing sky.

### M2.3 INV-PREP-01 as recorded in the run

INV-PREP-01 **did run** on this 520 preprocess (not skipped). Infolog 18:43:19-18:43:25:

- g_60_4 `large_small_ratio=0.02x (warn>10)`
- i_70_4 `0.02x`
- r_60_4 `0.01x`
- z_90_4 `0.06x`

This-measure median on all 25 g lights: **0.0165x** (matches the log). `invariants_runtime.py:477-512` / `pipeline.py:18673-18704`, policy WARN at 10. Stars dominate `var_small`, so donut-scale power never trips. **The existing metric saw the frames and stayed informational.** PRECAL-INPUT-CONTRACT-01 cannot reuse INV-PREP-01 as the loud gate; it needs a star-masked, bias-aware flatness check (M2.1).

## M3 - CalibrationLibrary for equipment 4 (2026-06-08)

Light match keys: C5A-150M / AZ800, bin4, 60 s, FILTER=g, CCD-TEMP=-15.02 C, GAIN header 12.48, DATE-OBS 2026-06-08. Validity windows 90 d dark / 200 d flat.

On disk, only three FITS, all **WIDE QHY294PROM + Carl-Zeiss 200mm**, Bin1, GAIN 0, ~-10 C, 2026-04-22 (47 d before the night, inside the windows, **wrong rig**):

| File | SHA256 | DATE-OBS | DB scope |
|------|--------|----------|----------|
| `Dark_60s_Dark_0G_-10deg_Bin1_20260422.fits` | `525daf5c56b78ece...` | 2026-04-22T18:19:42 | eq=1 tel=1 |
| `Dark_120s_Dark_0G_-10deg_Bin1_20260422.fits` | `fc165ce74dfdbc77...` | 2026-04-22T18:37:10 | eq=1 tel=1 |
| `Flat_0.15s_NoFilter_0G_-10.5deg_Bin1_20260422.fits` | `0667ed7662e62fd9...` | 2026-04-22T18:14:30 | eq=1 tel=1 FILTER=NoFilter |

`CALIBRATION_LIBRARY` has those three rows only. EQUIPMENTS id=4 is C5A-150M; TELESCOPE id=6 is AZ 800. Scoped lookup `find_best_calibration_library_path` (`database.py:2343-2487`) for eq=4 tel=6 returns **None** for dark and flat (bin4 and prefer-unbinned bin1).

**VYVAR cannot calibrate 520 from the current library.** Milan must supply AZ800 / C5A-150M darks+flats (g at least; ideally g/i/r/z, bin4, ~-15 C), then a raw reclassify + cal rerun is possible. June's calibration provenance is the open M1 question (production-path button, masters not in the library now).

## M4 - downstream confirmation (read-only)

### M4.1 comp forensics

June Gaia IDs from the June table: **not recovered**. Proxy: every g_60_4 masterstar with Gaia G in [11.6, 13.9], excluding the target itself.

CSV: `m4_comp_forensics.csv`.

| Set | n | G range | today's rms |
|-----|---|---------|-------------|
| June-band field stars (excl. V0612) | 47 in field; 8 with fieldwide `comp_rms` | 11.61-13.90 | median **0.236**, min 0.156, max 0.299 |
| Selected V0612 comps today | 8 | 14.48-16.86 | photometry `comp_rms` median **0.0263**, min 0.0165, max 0.0339 |

Direction matches H-CAL-MISCLASS: the bright June-class stars are ~9x noisier today than the faint set Layer 2 actually kept (`sparse_fallback` from a 254-star pool). Selected G starts at 14.48 (one star brighter than the 15.2-16.9 UI note).

### M4.2 time axis "time (unknown)"

**Independent UI defect, not a missing pre_cal time stamp.**

On-disk LC `lightcurve_1111749368289526912.csv` has `time_base=BJD_TDB` on every row; BJD/HJD/JD populated; observer site Zdanice (infolog 18:40:55). `_recompute_bjd_hjd_with_status` succeeded.

UI: `src_py/ui_aperture_photometry.py:32-36` `_lc_time_axis_title` catches `ValueError` from `resolve_lc_time_base` and returns `"time (unknown)"`. `_LC_OVERVIEW_COLS` (`:40-56`) **omits** `time_base`. `_cached_read_csv` (`:67-76`) `usecols=lambda c: c in _LC_OVERVIEW_COLS` drops the column. `photometry_core.py:137-152` then raises `"time_base column absent"`.

June "BJD (TDB)" was the same `lc_time_axis_short_label` path (`photometry_core.py:156-159`) when the column was present in the loaded frame. Fix scope: add `time_base` (and keep catalog id cols) to `_LC_OVERVIEW_COLS`. Not executed here.

## M5 - STOP menu (nothing executed)

Evidence-ranked options for Milan:

**(b) first.** Library has no eq=4/tel=6 masters. Milan supplies AZ800 / C5A-150M calibration frames (dark 60 s bin4 ~-15 C, flats in g and the other filters used), then (a).

**(a) blocked until (b).** If those masters exist, reclassify 520 input as raw (`! RUN VYVAR`, not non-cal) and run VYVAR calibration, full rerun. Predicted: June-class photometry if the June production path was in fact a calibrated reduction (JOURNAL draft_409 ooe ~0.006 / pre-eclipse 0.010). Not a promise of 0.0622 until the cal frames exist and a rerun is measured.

**(c) PRECAL-INPUT-CONTRACT-01** (separate wiring task, stays ROADMAP MED). `pre_calibrated` input must pass a radiometric flatness check using a star-masked, bias-aware metric (M2.1), with the threshold derived from VYVAR-calibrated references of the **same class** (not a magic 10x INV-PREP-01, which already ran and stayed at 0.02x). Fail loud naming the structure. Third evidence instance: 518 saturate CONFLICT, 518 `is_calibrated=0`, 520 donuts + non-cal button on `RAW_NON_CALIBRATED` lights.

**(d) time-axis.** One-line UI preload: put `time_base` in `_LC_OVERVIEW_COLS`. Independent of H-CAL-MISCLASS.

z_90_4 solve reject remains parked (MULTIFILTER-WCS-01). No gate/selection change in this task.

## `--fast`

`python dev/scripts/session_baseline_check.py --fast` after this measure:
**OVERALL PASS** (pytest 1530 passed, 32 skipped). Expected WARNs:
untracked (AC-02 + Archive + this session dir until commit), origin/main
`b1af049`, db-quick-check waived, ledger-todo, deps-outdated. Wall ~10 min.

## Files changed

Measure / docs only (no src_py, no AC-02, no 516 ePSF):

- `dev/results/CURSOR_RESULT_CAL_520_01.md` (this file)
- `dev/results/session_20260824_cal_520_01/` (metrics, ratio PNG, comp CSV, scripts)
- `dev/results/context/session_20260824_cal_520_01/` (Rule 0.2 copy)
- `docs/VYVAR_JOURNAL.md`, `docs/VYVAR_STATE.md`, `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_DECISIONS.md` (one-liners)

## Errors

None. `--fast` recorded below.
