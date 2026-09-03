CURSOR RESULT - 2026-08-18 22:20 UTC+2

What I did
DAO-COMPLETENESS-01 measurement only (no production code/config edits).
Product: draft **516**, setup **NoFilter_60_2**, 134 aligned 60 s lights,
MASTERSTAR.fits + masterstars_full_match.csv from the ANCHOR-516-04
clean rebuild. Frames sampled across the night (not assumed a priori):

| stem | DATE-OBS |
|------|----------|
| BO_CVn_Light_001 | 2026-04-23T19:35:20 (twilight) |
| BO_CVn_Light_037 | 2026-04-23T20:47:56 |
| BO_CVn_Light_076 | 2026-04-23T22:06:35 |
| BO_CVn_Light_109 | 2026-04-23T23:13:08 |
| BO_CVn_Light_148 | 2026-04-24T00:31:47 (dark) |

Pixel scale 9.774 arcsec/px, 2082 x 1397. Gaia set for accounting =
local SQLite cone at chip circumradius 3.554 deg, G<=17.5, then pixel
footprint: **11953** on-chip stars (query n=20083). This is not the
100k `field_catalog_cone.csv` (4131 on-chip). Raw pipeline completeness
3.55% uses the 100k cone as denominator and is not a FOV completeness.

Harnesses (tmp/, not tracked): `dao_completeness_01.py`,
`dao_completeness_01b.py`. Raw numbers under
`dev/results/context/session_20260818_dao_comp_01/`.

Premise (Rule 0.1): overlay census on Light_001 proc CSV is
GAIA_MATCHED 3500, FORCED_APERTURE 0, DAO_ONLY 0, with
forced_photometry=True on 360 rows. UI 3516 is the same layer on a
slightly different row set (MS 3546 GAIA_MATCHED). Comparable. The
visible Gaia-without-DAO marks are the cone overlay plus stars that
never entered MASTERSTAR, not a UI count of FORCED_APERTURE.

## Pre-registered expectations

| ID | State | Evidence |
|----|-------|----------|
| E1 | DEVIATE | Bright-end (G <= edge50-2 = 13.0) DAO match rate **90.8-91.3%**, not ~100%. 55 on-chip stars with G<=13 are not in MASTERSTAR. Isolated bright misses are not rare. Pipeline Fleming reconcile on this product: completeness_50=91.3%, G_lim_50=15.02. |
| E2 | PASS | Night-median 50% DAO-match edge = **15.0** (all 5 frames). `target_depth_g` = **15.0**. Delta **0.0**. Note: target_depth is last 0.5-mag bin with zone linear_frac>=0.5, not match_frac; the two rules land on the same mag_hi here. Match_frac in 14.5-15.0 is 0.69 median; 15.0-15.5 is 0.25. T4-1 "sigma 52->30" is not this product: `sigma_bkg_ap` median 169->139 then 159; DAO `rms_conv` ~630 ADU, flat; edge50 did not move. |
| E3 | DEVIATE | Bright unmatched (G<=13.5, ~124-134/frame): BLENDED (nn<2 px) is **31-34** (~26%), not a majority. DETECTION_HOLE **63-67**, TOO_FAINT **24-26**, EDGE **2-8**, SATURATED **2**. At this plate scale 2 px = 19.5 arcsec is tighter than FWHM (5.30 px). Most "holes" have a Gaia neighbour inside the 10 px catalog-match floor. |
| E4 | DEVIATE | Pass 2 is wired, runs, and is the **primary** catalog-filling path. Production infolog `infolog_20260817_222127.txt`: `[DAO pass 1] 307 detections, 3681 Gaia unmatched`; `[DAO pass 2] 3314 additional detections from 3681 targeted positions`; `[DAO total] 3621`. Replicated on MASTERSTAR.fits vs cone-on-chip: p2=3315/3681. Not disabled, not gated to ~0. Local threshold = `masterstar_dao_pass2_sigma` (1.9) x annulus std; `center_tol=5` px; pass-2 exempt from prematch k-gate (`prematch_exempt_pass2=True`). |

## Output / findings

### Part A - two-pass accounting

**Why overlay shows FORCED_APERTURE 0 (numbers, not UI):**
`forced_photometry.py` sets `FORCED_SOURCE_TYPE = "GAIA_MATCHED"` and
marks `forced_photometry=True`. `ui_masterstar_qa.py` counts
`source_type == "FORCED_APERTURE"`. Writer and display disagree.
Light_001 proc: 3500 GAIA_MATCHED, 0 FORCED_APERTURE, 0 DAO_ONLY,
360 forced_photometry=True. Forced photometry **did run**. Display
filter, not "ran and rejected all".

**Pass 2 implementation** (`pipeline.py` `_dao_targeted_pass2_unmatched_gaia`):
- Entry: Gaia in FOV of `cat_df` with no pass-1 DAO within
  `_catalog_match_radius_px` (sep 25 arcsec, **floor 10 px**).
- Cutout 21x21 (hw=10). Local annulus on full `data0` (`_dao_pass2_annulus_stats`).
- Threshold: max(1.9, min(20, pass2_sigma)) x local_std.
- Accept nearest peak with d(center)<5 px; merge dedup 3 px; `vy_dao_pass=2`.
- Call sites: MASTERSTAR build (`detect_stars_and_match_catalog` ~8835,
  cat = field cone) and per-frame `detect_stars_match_master_reference`
  ~8021 (cat = MASTERSTAR members only).

**Production vs harness:** targeting all 11953 on-chip Gaia inflates
unmatched to ~11270 and pass2 raw to ~8500. Production MASTERSTAR build
targets cone-on-chip (~4131), unmatched 3681, pass2 **3314**. Per-frame
path vs MS members: unmatched 3235, pass2 raw 3107, catalog-matched
pass1=310 -> merged=3460 (pass2-only matches +3150).

**Consumers of unmatched Gaia (not in MASTERSTAR):**

| Consumer | Behaviour on unmatched (no MS row) |
|----------|-------------------------------------|
| MASTERSTAR admission | Excluded. Need DAO (pass1 or pass2) + Gaia 1-1 match. 75 DAO_ONLY kept on MS, no catalog_id. |
| Per-frame proc CSV | `_proc_drop_unmatched_dao_rows` then `_proc_catalog_keep_matched_rows_only` (GAIA_MATCHED + catalog_id). DAO_ONLY dropped. Forced injects **force-eligible MS members only** (not sat/nonlinear, not VSX/variable/NSS, not QSO/GAL). Unmatched Gaia never injected. |
| Comp pool (`comparison_stars.csv`) | From MASTERSTAR; require catalog match / zone gates. 0 DAO_ONLY in pool (n=2356). Unmatched Gaia cannot be comps. |
| CT / ensemble | Same pool. Per-frame, 182/2356 comps on Light_001 are `forced_photometry=True` (INV-COMP-MEMBERSHIP). |
| Target photometry | VT needs catalog_id. `below_target_depth` uses `target_depth_g=15` on planned targets, not on unmatched field stars. |
| Dilution neighbours | `gs11_dilution_enabled` default False. When on, queries Gaia DB (not MS), so unmatched FOV Gaia would count. Currently off. |
| ePSF isolation | Uses full `field_catalog_cone.csv` (not MS), so unmatched cone stars can veto ePSF candidates. |

### Part B - completeness curve

Per-frame and night-median 0.5-mag bins:
`completeness_bins_per_frame.csv`, `completeness_bins_night_median.csv`.
Edge50 per frame: all **15.0**. DAO-matched (forced_photometry=False).

Night median match_frac (n_gaia median):

| G bin | n | match_frac |
|-------|---|------------|
| 8.0-8.5 | 13 | 1.00 |
| 8.5-9.0 | 14 | 1.00 |
| 9.0-9.5 | 27 | 0.96 |
| 9.5-10.0 | 35 | 1.00 |
| 10.0-10.5 | 39 | 0.97 |
| 10.5-11.0 | 59 | 0.97 |
| 11.0-11.5 | 102 | 0.95 |
| 11.5-12.0 | 150 | 0.89 |
| 12.0-12.5 | 223 | 0.87 |
| 12.5-13.0 | 290 | 0.89 |
| 13.0-13.5 | 397 | 0.89 |
| 13.5-14.0 | 505 | 0.87 |
| 14.0-14.5 | 652 | 0.82 |
| 14.5-15.0 | 856 | 0.69 |
| 15.0-15.5 | 1111 | 0.25 |
| 15.5-16.0 | 1360 | 0.0007 |

Delta vs E2 reference: **0.0 mag**. Sky median 2416 -> 1346 ADU across
the five frames; 50% edge did not move.

### Part C - unmatched classification (G <= edge50-1.5 = 13.5)

Counts per frame (`frame_meta.csv`):

| frame | n_unm | TOO_FAINT | BLENDED | HOLE | EDGE | SAT |
|-------|-------|-----------|---------|------|------|-----|
| 001 | 133 | 26 | 34 | 63 | 8 | 2 |
| 037 | 134 | 24 | 34 | 67 | 7 | 2 |
| 076 | 124 | 24 | 31 | 65 | 2 | 2 |
| 109 | 126 | 24 | 33 | 65 | 2 | 2 |
| 148 | 128 | 26 | 33 | 65 | 2 | 2 |

Full c-list: `detection_holes.csv` (325 rows). Unique IDs: 69.
62/69 persist on all 5 frames. Spatial: median (x,y)=(950,880),
near-center; edge<50 px = 2; corners not over-represented
(8/7/7/8 in 15% strips). Not vignetting-clustered.

**Governing mechanism for holes (not an empty-sky miss):**
Pass 2 only searches Gaia with **no DAO within 10 px**. Catalog match
is 1-1. A neighbour DAO inside the 10 px floor marks the star
"matched" for targeting; greedy assignment then gives the detection
to the other Gaia; the loser never enters MS and is not force-injected.
Of 69 unique holes, **nn_px > 10: 1**. PSF-isolated (nn >= FWHM 5.30)
and not in MS: **26** (25/26 on all 5 frames). List:
`detection_holes_isolated_not_ms.csv`. Brightest: G=11.39
`1485618815342037504` peak~3060 ADU, SNR_ap~16, nn=9.46 px.

**Reverse (production proc DAO, not harness pass1):** Light_001
unforced x,y vs on-chip Gaia within 10 px: 3140 DAO rows, **199** with
no Gaia (6.3%), **170/199 in corners**. MS DAO_ONLY=75. Pattern:
edge/corner artifacts plus the 75 DAO_ONLY kept on MS and dropped
from proc. Not a uniform extra-galactic population.

### Part D - ladder decision input (no implementation)

Rung occupancy, Light_001, on-chip G<=17.5 (n=11953)
(`ladder_rungs_frame001.csv`):

| Rung | n/frame | Notes |
|------|---------|-------|
| DETECTED | 2992 | proc, not forced |
| FORCED_OK | 336 | proc, forced; MS members missed this frame |
| MS_INELIGIBLE | 42 | in MS but not injected (sat/var/extended) |
| TOO_FAINT (G>15) | ~8184 | below 50% edge; legitimate non-detection |
| UNACCOUNTED G<=15 | 399 | blend / match-floor collision / hole / edge / sat |

Architect rungs DETECTED / FORCED_OK / TOO_FAINT / BLENDED cover the
mass. SATURATED+EDGE are small (2+2 to 8 per frame at G<=13.5).
FORCED_OK today is **only MS members**. The 399 G<=15 not in MS would
need a new admission/forced path to become FORCED_OK.

**Position-QA for FORCED_OK:** identity residuals on 3546 matched MS
rows: p50=0.54 px, p95=1.78 px, p99=2.91 px, max=4.92 px.
Current refine bound is `forced_photometry_centroid_bound_fwhm=2.5` x
FWHM = **13.3 px**, far looser than astrometric p95. A centroid check
at ~2 px (p95) is feasible; 13 px is not a useful QA.

**Comp pool = DETECTED only:**
- Admission: **confirm** (0 DAO_ONLY in comparison_stars; need Gaia
  catalog_id from MS). "DETECTED" here includes pass-2 recoveries
  (3314 of 3621 MS detections).
- Per-frame presence: **refute** as a hard DETECTED-this-frame rule.
  182/2356 comps on Light_001 are FORCED_OK. That is the
  INV-COMP-MEMBERSHIP design. Dropping FORCED_OK from the ensemble
  would punch 8% of the pool every frame.

**Pass 2: repair vs absorb.** Do not treat as dead code. It already
**is** the ladder's DETECTED rung for ~90% of MS rows. Absorb into
the ladder as DETECTED (targeted recovery), not FORCED_OK (aperture
at catalog XY with no DAO peak). Remaining gap is the 10 px
any-neighbour skip + 1-1 assignment, which Pass 2 cannot see by
construction. Repairing Pass 2 without changing the match floor
will not admit the 26 FWHM-isolated holes. Overlay FORCED_APERTURE
label is a display bug; do not fix in this task.

## Errors (if any)

None. Measurement-only constraint held.

## Files changed

No production code, no config, no docs except JOURNAL (category-c
confirmed). Push not authorized.

- `dev/results/CURSOR_TASK_DAO_COMPLETENESS_01.md`
- `dev/results/CURSOR_RESULT_DAO_COMPLETENESS_01.md`
- `docs/VYVAR_JOURNAL.md` (category-c note)
- `dev/results/context/session_20260818_dao_comp_01/` (CSV/JSON)

## Runtime (Rule 0.3)

| Part | seconds |
|------|---------|
| A overlay census + code read | <1 (I/O) |
| B+C 5-frame curve + class + all-Gaia pass2 | 132.1 |
| A/C follow-up (MS-path pass2, reverse proc, hole enrich) | 40.3 |
| Gaia query | 0.5 |
| **Total measurement** | **~173** |

Pass 2 cost on production path: 6.9 s/frame vs MS members, 7.9 s on
MASTERSTAR vs cone-on-chip, 23.8 s if pointed at all 11953 on-chip
Gaia. MASTERSTAR build already pays the cone-on-chip cost (infolog
20:25:44 to 20:25:52, ~8 s).
