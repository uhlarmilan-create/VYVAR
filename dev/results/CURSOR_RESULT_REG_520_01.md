CURSOR RESULT - 2026-08-24 22:20 UTC+2

What I did
Measured why the same SS Cam 2026-06-08 night, same
`RUN VYVAR (non-cal)` button, yields V0612 Cam lc_rms 0.39 today vs
0.06 on 2026-06-16. Measurement only: no wiring, no 516 writes, live
520 draft untouched. HEAD `92361a353823566fb9130632a57b1003a9fa76f5`
(descendant of `505fa13`). Declared dirt OK (AC-02 wiring unstaged,
unrelated). Not pushed.

Rig: Brno AZ800 / C5A-150M, non-cal path. Artifacts:
`dev/results/session_20260824_reg_520_01/` and Rule 0.2 copy
`dev/results/context/session_20260824_reg_520_01/`.

## Premise (Rule 0.1)

**What is compared:** today's draft 520 `g_60_4` V0612 Cam
(`catalog_id` `1111749368289526912`) aperture photometry versus the
operator's 2026-06-16 UI snapshot of the same non-cal button on the
same night. June products (drafts 399/400/402/407/409) are gone; June
Gaia IDs are a G 11.6-13.9 band proxy, not recovered IDs.

**How they differ:** today `lc_rms=0.394883` / `lc_rms_ooe=0.080252`,
comps G 14.48-16.86, `comp_rms` 0.0165-0.0339, `sparse_fallback`,
UI axis "time (unknown)". June snapshot: `lc_rms` 0.0622, comps
G 11.6-13.9, `comp_rms` 0.005-0.012, axis "BJD (TDB)". Full lc_rms
includes the EA eclipse on both reductions. CAL-520-01 classified this
as H-CAL-MISCLASS (wrong button / no masters). Milan testimony on
record: June used the same non-cal button and VYVAR has no AZ800
masters; donuts are a per-star constant under stable pointing. The
regression under test is therefore inside VYVAR between 2026-06-16 and
today, on the non-cal path.

Replay LCs below use existing per-frame `dao_flux` (24 epochs with
target flux). They are comparable to today's 0.3949 (reproduced at
0.3914). They are not a re-extraction, so they cannot claim June's
aperture/detrend bit-identity; they can claim whether today's
photometry of a given ensemble already contains June-class scatter.

## Gates

| Gate | Status | Evidence |
|------|--------|----------|
| G1 tip descendant of `505fa13` | PASS | HEAD `92361a3`; `git merge-base --is-ancestor 505fa13 HEAD` exit 0 |
| G2 516 production ePSF SHA | PASS | `172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20` unchanged |
| no 516 writes / live 520 untouched | PASS | outputs only under session dir |
| `--fast` | see end | run after this write-up |

Wall for M1/M1b/M2/M3 measure scripts: ~5 s + 1.4 s + 2.4 s.

## M1 - A/B/C matching replay (S1)

Success criterion, stated up front: if B (2.0 px) or C (2.5 px)
restores June-class comps (G 11.6-13.9) and V0612 lc_rms ~0.06, S1 is
the regression and S2 is exonerated on this draft.

Certificate `Archive/Drafts/draft_000520/platesolve/g_60_4/dao_gaia_calibration.json`:
`pass2_center_tol_px=1.0`, `forced_seed_centroid_max_px=1.0` (floor
`DEFAULT_CENTROID_FLOOR_PX=1.0` at `dao_gaia_calibration.py:36`;
derive `dao_gaia_calibration.py:721-776`). Pass1 `match_radius_px=3.5`
(1.7 x identity p95 2.10). Solve rms g=1.44 px. The 1.0 px number is
the pass2/seed centroid clamp, not the pass1 pairing radius.

Existing census (already DETECTED_P1, independent of A/B/C):

| Band | n Gaia | DETECTED_P1 |
|------|--------|-------------|
| G<12 | 11 | 11 |
| G<14 | 57 | 57 |
| G 11.6-13.9 | 44 | 44 |

Pass2/seed replay on unmatched Gaia G<=17.5 (n=78 tried):

| Variant | tol px | pass2 accept | forced seed | centroid reject | n DETECTED G<12 | n DETECTED G<14 |
|---------|--------|--------------|-------------|-----------------|-----------------|-----------------|
| A today | 1.0 | 0 | 25 | 10 | 11 | 57 |
| B hand class | 2.0 | 0 | 28 | 3 | 11 | 57 |
| C ERA-03 class | 2.5 | 0 | 28 | 3 | 11 | 57 |

Table: `m1_abc_table.csv`. Bright G<14 pass2 IDs: empty at every
tol. **n matched at G<12 / G<14 is unchanged across A/B/C.** June-band
stars are already in the catalog as DETECTED_P1. Widening the 1.0 px
floor does not bring them back because they were never lost to
pass2/seed starvation.

LC replay (same `dao_flux`, no re-extraction):

| Ensemble | n epochs | lc_rms | lc_rms_ooe | mean n_comp |
|----------|----------|--------|------------|-------------|
| A = today's 8 selected | 24 | 0.3914 | 0.0735 | 8.00 |
| First-8 June-band DETECTED (incomplete) | 24 | 0.182 | 0.030 | 3.08 |
| All June-band DETECTED | 24 | 0.468 | 0.534 | 10.42 |

**Success criterion NOT met by matching.** S1 as "tolerance below
astrometric rms starves bright-star pairing, so the pool degrades to
faint survivors" is **false on this draft for catalog membership**.
The June G 11.6-13.9 stars are present. They are not selected. Plot:
`lc_v0612_ensembles.png`.

S2 therefore remains in play (M3).

Today's 8 selected Gaia IDs (none in the June band):

- `1112112413285008896` G=16.19
- `1112115024625070720` G=15.23
- `1111930718988511616` G=15.32
- `1112119250872867200` G=16.50
- `1112110042463052928` G=16.40
- `1111931371821079552` G=15.56
- `1111737823417422464` G=16.86
- `1111922300852743808` G=14.48

June-band fieldwide `comp_rms` today (CAL-520-01 CSV, 8 stars with a
value): median 0.236, min 0.156, max 0.299. All fail
`phase01_comparison_max_comp_rms=0.1` (`config.json`; default
`config.py:883`). That is why they cannot enter
`_select_comps_by_rms_then_color`.

## M1b - provenance, saturation, visibility

Stored columns used: masterstar `source_state`, `source_type`,
`forced_photometry`, `peak_dao`, `peak_max_adu`, `zone`; per-frame
proc `forced_photometry`, `source_type`, `x`, `y`, `dao_flux`.
**Per-frame `source_state` is not persisted**
(`per_frame_source_state_column_present: false`). Persistence-gap
finding: detection provenance is MASTERSTAR/census only.

Today's 8 selected:

- `source_state=DETECTED_P1`, `source_type=GAIA_MATCHED`, zone linear,
  all 8/8.
- `forced_photometry` fraction = 1.0 on every photometered selected
  star (placed-aperture path, not a unique ghost flag: the same flag
  is 1.0 on many real bright stars too).
- MASTERSTAR `peak_dao` 114-898 ADU on a ~36k pedestal; SNR_ap median
  ~4-22.
- `sat80_clipped_frac=0` on the selected set.

June-band G 11.6-13.9 (C5A bin4, 0.8 m, 60 s):

- 43/47 DETECTED_P1 in the MS join.
- `sat80_clipped_frac=0`. They are **not** clipped/nonlinear at 80% of
  65535. Peak above sky is hundreds to ~1290 ADU. Honest large
  `comp_rms` (0.16-0.30) is scatter, not a saturation flag.
- Completeness is poor for several: `n_frames` 0 or 1 for a subset
  (edge / lock), which is why the naive first-8 June LC had
  mean_n_comp 3.08.

Visibility cutouts (MASTERSTAR + aligned mid-night):

- `cutouts_selected_comps_masterstar.png` /
  `cutouts_selected_comps_aligned_mid.png`: selected comps at
  **masterstar x,y** are visually empty / noise. Milan's "invisible on
  the frame" claim is true at the photometry position.
- `cutouts_june_band_masterstar.png` /
  `cutouts_june_band_aligned_mid.png`: June-band G~12 stars are
  obvious bright blobs at their MS x,y.
- `cutouts_selected_at_gaia_xy.png` /
  `cutouts_selected_at_gaia_xy_aligned_mid.png`: same 8 IDs at
  **census Gaia x,y**.

Gaia-DAO residual (MS x,y vs census x_gaia,y_gaia):

| Set | n | median d px | p95 px | n within 2 px |
|-----|---|-------------|--------|---------------|
| today's 8 selected | 8 | **59.0** | 140 | **0** |
| June-band | 43 | 1.18 | 16.5 | 36 |
| G<12 | 11 | 1.74 | 3.94 | 7 |

Per selected ID, proc-frame x,y sits on the MS detection
(median 0.47-0.99 px) and **19-151 px from Gaia**. Photometry is not
at the Gaia star. CSV: `m1b_selected_xy_vs_gaia.csv`,
`m1b_selected_proc_xy.json`, `june_comp_forensics.csv`.

Ghost hypothesis, refined: the 8 selected are **not** `catalog_only`
forced apertures at Gaia/WCS. They are labeled DETECTED_P1 /
GAIA_MATCHED, but the aperture follows a DAO peak tens of pixels from
that Gaia ID. Sky-vs-sky (or noise-peak-vs-noise-peak) differential
then yields deceptively low `comp_rms` 0.016-0.034, which the rms
ceiling prefers. The Gaia G=15-17 label is a catalog identity, not a
star under the aperture.

**Verdict:** 0/8 of today's ensemble is `catalog_only`. 8/8 is
DETECTED_P1 with a **false lock geometry** (aperture != Gaia).
Fraction forced-photometry flag = 1.0, but that flag does not mean
"ghost"; it means placed aperture. Comp selection **does not gate on
`source_state` at all**:

- `photometry_core.py:15853-15860` `cand_mask` = not saturated / not
  VSX / not likely_saturated. No `source_state`, no Gaia-DAO residual,
  no SNR floor.
- `comp_pool_noise.py:1002-1046` `admit_pool_stars` = VSX / Gaia
  variable only.
- `_select_comps_by_rms_then_color` `photometry_core.py:15392-15462`
  hard ceiling `phase01_comparison_max_comp_rms=0.1`.
- `comp_selection_per_target.py` `cand_mask` (~396+) likewise has no
  `source_state` (grep: zero hits in that file).

That is a selection-input defect independent of S1, and a fourth
instance of "the statistic under the gate stops measuring what the
gate thinks": `DETECTED_P1` / low `comp_rms` is treated as a real
comparison star. Menu item: comps must be DAO-detected **on the Gaia
position** (residual comparable to solve rms / FWHM), or forced /
large-residual comps must carry an rms penalty / exclusion. Not
wired here.

## M2 - tolerance vs astrometry (generic principle)

Curve: nearest DETECTED xy vs Gaia (this is a geometric match-loss
curve, not the census DETECTED_P1 at match_radius 3.5). File
`m2_tolerance_curve.json`, plot `m2_match_vs_radius.png`.

g_60_4, 400 Gaia in cone:

| radius px | all Gaia | G<12 n=11 | G<14 n=57 | June-band n=45 |
|-----------|----------|-----------|-----------|----------------|
| 1.0 | 8.8% | 27% | 35% | 36% |
| 1.44 (solve rms) | 13% | 36% | 63% | 69% |
| 2.0 | 17% | 64% | 81% | 84% |
| 2.5 | 18% | 91% | 88% | 87% |
| 3.5 (live match_radius) | 19% | 91% | 93% | 93% |

A 1.0 px **match** radius would starve bright pairing. Live pass1 does
not use 1.0; it uses 3.5. The 1.0 floor still governs pass2/seed
(`centroid_raw = clamp(seed_p95, floor=1.0, cap)`).

Per-set derived vs solve rms:

| set | solve rms | fwhm | pass2/seed tol | match_radius | tol - rms |
|-----|-----------|------|----------------|--------------|-----------|
| g_60_4 | 1.44 | 1.25 | 1.0 | 3.5 | -0.44 |
| i_70_4 | 2.98 | 1.25 | 1.0 | 4.0 | -1.98 |
| r_60_4 | 1.49 | 1.25 | 1.5 | 5.0 | +0.01 |

Candidate fix to characterize, not wire: derived matching / centroid
tolerance floor must be a function of measured astrometric rms and
FWHM, never a fixed 1.0 px clamp. Worst live gap is i_70_4. This
curve is what a later fix task should cite. It is **not** why June
G 11.6-13.9 are absent from the selected ensemble.

## M3 - S2 (run because M1 B/C did not restore)

Not a git checkout of 2026-08-15 Layer 1/2/3 vs June-era source.
Bisect used: **identical** matched pool and identical `dao_flux`,
different ensemble membership. That is the selection question.

Completeness filter: `dao_flux` finite and >0 in >=20 of 25 frames.

| Ensemble | IDs | G range | lc_rms | lc_rms_ooe | mean n_comp |
|----------|-----|---------|--------|------------|-------------|
| today selected 8 | 8 | 14.48-16.86 | 0.3914 | 0.0735 | 8.00 |
| June-band complete (7; all that pass >=20) | 7 | 11.61-13.9 class | 0.189 | 0.131 | 6.88 |
| G<14 DETECTED complete 8 (brightest with >=20) | 8 | 7.63-12.07 | **0.0682** | 0.0240 | 7.92 |
| G<14 DETECTED complete all 13 | 13 | G<14 | 0.116 | 0.0158 | 12.88 |

G<14 complete8 Gaia IDs (brightest first):

- `1112113680298377344` G=7.63  d=0.27 px  n=25  (peak_max 88781 > 65535; clipped, still zone=linear)
- `1111931371823539456` G=9.23  d=5.54 px  n=25
- `1111920204908702336` G=10.07 d=1.41 px  n=25
- `1112110695298081664` G=10.90 d=2.34 px  n=25
- `1112121862213003648` G=11.10 d=1.86 px  n=25
- `1111749157833870208` G=11.23 d=0.68 px  n=25
- `1112121067641532160` G=11.61 d=2.22 px  n=25  (June-band)
- `1111750360424713344` G=12.07 d=1.26 px  n=22  (June-band)

Plot: `lc_v0612_today_vs_g14_complete8.png`. JSON: `m3_bright_ensemble.json`.

Reading, not a slogan:

- Forcing a **bright, complete, well-centered** ensemble from today's
  photometry recovers headline lc_rms **0.068 vs June 0.0622**. That
  is June-class on the number the operator quoted.
- That 8 is **brighter** than June's G 11.6-13.9 (it includes G=7.63,
  which is clipped). It is not a recovered June table.
- Forcing the June-band complete stars only recovers **0.189, not
  0.06**. So S2 explains *which* comps are selected (faint 59-px
  offsets with pretty rms, instead of bright stars). A residual ~3x
  vs June remains on the June-band-only subset: completeness,
  aperture/detrend, and/or the unrecoverable June 8-star mix. Do not
  claim bit-identity with 2026-06-16.

S2 mechanism on the live path, file:line: rms-then-color
(`photometry_core.py:15392-15462`) with hard ceiling 0.1 drops every
June-band star that has fieldwide rms 0.16-0.30, then `head(n_comp_max)`
keeps the lowest-rms survivors, which on this draft are the 59-px
offset G~15-17 IDs. Layer 1/2/3 (2026-08-15/16) replaced what June
ran. Matching (S1) did not.

## M4 - "time (unknown)" (S3, independent)

Not a missing non-cal header stamp. Site Zdanice resolved. On-disk
`platesolve/g_60_4/photometry/lightcurves/lightcurve_1111749368289526912.csv`
has `time_base=BJD_TDB` on every row; `bjd` / `hjd` / `jd` populated
(CAL-520-01 re-confirmed this measure: columns include `time_base`).

UI: `src_py/ui_aperture_photometry.py:32-36` `_lc_time_axis_title`
catches `ValueError` from `resolve_lc_time_base` and returns
`"time (unknown)"`. `_LC_OVERVIEW_COLS` (`:40-56`) **omits**
`time_base`. `_cached_read_csv` (`:67-76`) `usecols` drops the
column. `photometry_core.py:137-152` then raises
`"time_base column absent"`.

June "BJD (TDB)" is `lc_time_axis_short_label`
(`photometry_core.py:156-159`) when the column is present in the
loaded frame. Fix scope: add `time_base` to `_LC_OVERVIEW_COLS`.
Not executed here.

## M5 - STOP menu (nothing executed)

Evidence-ranked after M1/M2/M3. CAL-520-01 H-CAL-MISCLASS is
**superseded as the causal story** for 0.39 vs 0.06. Library facts
from CAL-520-01 remain true (no eq=4/tel=6 masters; donuts real;
INV-PREP-01 0.02x informational) and do not explain the June-vs-today
gap on the same non-cal button.

**1. S2 selection-input (this draft's lc_rms).** Hard
`phase01_comparison_max_comp_rms=0.1` plus rms-then-color, with no
gate on `source_state`, Gaia-DAO residual, or visibility. Comps must
be DAO-detected on the Gaia position (residual ~ solve rms / FWHM),
or large-residual / forced comps excluded or penalised. Fourth
"statistic under the gate" instance. Also: G=7.63 is zone=linear with
`peak_max_adu=88781 > 65535` (clipped) -- zone is not measuring
saturation either.

**2. (a) production tolerance floor from solve rms / FWHM (S1 hygiene,
generic, all rigs).** Cite the M2 curve. Live pass1 match_radius is
already 3.5 on g; the defect is the 1.0 px pass2/seed clamp,
especially i_70_4 (tol 1.0 vs rms 2.98). XFER-01 fixed the SANDBOX
gate only. This is real and should be wired later. It did **not**
cause today's missing June comps.

**3. (b) RAW / NON-CAL declared mode (Milan product direction).**
Provenance stamp `calibration_mode=non_cal_declared`, UI banner,
cautious LC quality class, submit lock analogous to
INV-PSF-SUBMIT-01. The path is legitimate and first-class; it must
be self-aware, never silent. Do not treat non-cal as a misclick.

**4. (c) S3 time-axis.** Add `time_base` to `_LC_OVERVIEW_COLS`
(`ui_aperture_photometry.py:40-56`). Independent of S1/S2.

**5. (d) PRECAL radiometric metric demoted from gate to informative
stamp.** Non-cal is a first-class route by design; INV-PREP-01 0.02x
already informed and did not block. The metric informs, does not
block. PRECAL-INPUT-CONTRACT-01 as a **blocking** gate is the wrong
product direction given this testimony.

z_90_4 solve reject remains parked (MULTIFILTER-WCS-01). No
threshold, selection, or tolerance wiring in this task.

## `--fast`

`python dev/scripts/session_baseline_check.py --fast` after this
measure on HEAD `92361a3` (REG-520 artifacts still dirty):
**OVERALL PASS**. pytest 1530 passed, 32 skipped. Expected WARNs:
untracked, origin/main `b1af049`, db-quick-check waived, ledger-todo,
deps-outdated. Wall ~8 min.

## Files changed

Measure / docs only (no src_py, no AC-02, no 516 ePSF, live 520
untouched):

- `dev/results/CURSOR_RESULT_REG_520_01.md` (this file)
- `dev/results/session_20260824_reg_520_01/` (A/B/C tables, LC plots,
  M2 curve, June-comp forensics, cutouts, scripts)
- `dev/results/context/session_20260824_reg_520_01/` (Rule 0.2 copy)
- `docs/VYVAR_JOURNAL.md`, `docs/VYVAR_STATE.md`, `docs/VYVAR_ROADMAP.md`,
  `docs/VYVAR_DECISIONS.md` (one-liners)

## Errors

None.
