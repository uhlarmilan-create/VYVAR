CURSOR RESULT - 2026-08-05 (DRAFT-501 DAO detection census)

What I did
Read-only grep of draft infologs; masterstars CSV analysis; pipeline_meta
Gaia/DAO metrics; 20-row DAO-only sample; threshold arithmetic. No pipeline
runs. No config changes.

## 1 -- Per-frame DAO detection census

Infolog paths:
  draft_501: Archive/Drafts/draft_000501/infolog_20260805_113441.txt
  draft_435: Archive/Drafts/draft_000435_snapshot_skysurface_20260716/infolog_20260716_123126.txt
  draft_500: no infolog under Archive/Drafts/draft_000500 (pipeline_meta only)

MASTERSTAR DAO detection runs on ONE reference frame (not summed over 70 frames).
Alignment-stage DAO counts are per science frame (cap=200).

| draft | plate scale | MASTERSTAR DAO (ref frame) | align DAO/frame (median) | align min/max | ms rows | dao_only_fraction |
|-------|-------------|----------------------------|--------------------------|---------------|---------|-------------------|
| 501 Newton V | 1.30"/px | 1668 (after SNR filter; 1670 raw) | 200 | 200/200 | 1668 | 0.417 |
| 500 wide | 9.77"/px | (no infolog; ms=4122, n_stars_dao=2894) | n/a | n/a | 4122 | 0.136 |
| 435 wide anchor | 9.77"/px | 2951 (3777 raw -> 2951 SNR filter) | 200 | 200/200 | 2951 | 0.037 |

draft_501 MASTERSTAR log excerpts:
  [DAO pass 1] 1654 detections, 16 Gaia unmatched
  [DAO pass 2] 16 additional detections
  [DAO total] 1670 detections after merge
  DAO po SNR filtri: 1668/1670 (noise_floor~33493.9 ADU)
  detected_stars=1668, catalog_matched=972

draft_435 MASTERSTAR log excerpts:
  [DAO total] 3777 detections after merge
  DAO po SNR filtri: 2951/3777 (noise_floor~2105.9 ADU)
  detected_stars=2951, catalog_matched=2842

Interpretation:
  - Alignment per-frame DAO count is identical (median 200, hard cap) on Newton
    and wide anchor -- Newton does NOT detect more stars per frame during align.
  - MASTERSTAR ref-frame detections on Newton (1668) are FEWER than wide rig
    (2951-4122), not higher.
  - High dao_only_fraction on 501 is NOT from excess detections; it is from
    low match purity (972/1668 = 58% vs 2842/2951 = 96% on 435).

Supports mechanism (b) partially (match/Gaia rejection) over (a) on detection
count alone. Spurious DAO-only population examined in section 3.

## 2 -- Per-frame sigma / noise_floor distribution

No infolog lines matched "sigma_clipped_stats", "background std", "sky_sigma",
or "background_median" per frame on any draft searched.

Available noise_floor markers (MASTERSTAR ref frame, pre-catalog-match):

| draft | noise_floor (ADU) | sky_adu_per_px (meta) | sigma estimate | k=1.8 margin (k*sigma) |
|-------|-------------------|----------------------|----------------|------------------------|
| 501 | 33493.9 | 33482.2 | 8.65 (FITS sample) | **15.6 ADU** |
| 435 | 2105.9 | 1570.0 | ~297 implied from nf-sky | ~536 ADU |
| 500 | n/a (no infolog) | 1550.9 | 55.94 (FITS sample) | **100.7 ADU** (est.) |

Formula: noise_floor = median + k*sigma (masterstar_prematch_peak_sigma_floor=1.8).

501: median~33487 ADU, std_clip 8.65 ADU -> margin above pedestal only ~12 ADU
absolute (33494 - 33482).

500 estimated k*sigma / 501 k*sigma = 100.7 / 15.6 = **6.5x** (>5x).

Threshold mechanism is not adapting the sigma margin to pre-calibrated high
pedestal + tiny RMS. Absolute noise_floor tracks sky level, but the 1.8*sigma
acceptance band is far narrower on Newton pre-cal data, admitting low-significance
peaks. Log confirms: "687 DAO-only stars below 50sigma (kept in CSV by design)."

Supports mechanism (a).

## 3 -- Sample of 696 DAO-only rows (draft_501)

Random seed 501, n=20 from catalog_id-empty rows.

| x | y | peak_max_adu | peak_dao | flux | match_sep | edge_safe | snr50_ok |
|---|----|--------------|----------|------|-----------|-----------|----------|
| 924.4 | 4.5 | 33523.1 | 17.5 | -11.2 | (blank) | false | false |
| 2594.0 | 592.1 | 33504.1 | 15.0 | 9.0 | blank | true | false |
| 2652.5 | 1600.9 | 33511.7 | 16.3 | 37.3 | blank | true | false |
| 1058.9 | 171.1 | 33505.9 | 16.8 | 39.5 | blank | true | false |
| 1512.3 | 1605.0 | 33502.9 | 16.7 | -9.2 | blank | true | false |
| 456.3 | 92.7 | 33527.1 | 32.7 | 81.1 | blank | true | false |
| 373.8 | 1386.0 | 33516.1 | 17.4 | 53.6 | blank | true | false |
| 2114.6 | 1447.3 | 33496.8 | 13.2 | -23.2 | blank | true | false |
| 160.8 | 1932.3 | 33520.0 | 17.4 | 36.1 | blank | true | false |
| 981.2 | 222.0 | 33503.0 | 17.4 | 12.7 | blank | true | false |
| 3044.6 | 1139.5 | 33504.9 | 20.3 | 68.6 | blank | true | false |
| 2422.8 | 1364.1 | 33514.8 | 23.7 | 41.1 | blank | true | false |
| 3121.0 | 1904.7 | 33502.0 | 14.1 | -108.8 | blank | false | false |
| 2997.1 | 2022.1 | 33543.1 | 21.7 | -27.8 | blank | true | false |
| 1654.4 | 434.5 | 33521.5 | 20.6 | 77.4 | blank | true | false |
| 1585.3 | 742.4 | 33550.0 | 17.2 | -14.6 | blank | true | false |
| 285.2 | 90.7 | 33510.1 | 25.5 | 53.3 | blank | true | false |
| 3092.9 | 237.9 | 33509.2 | 22.4 | 48.3 | blank | true | false |
| 865.9 | 1628.0 | 33505.0 | 17.7 | -13.0 | blank | true | false |
| 1389.8 | 614.5 | 33512.6 | 22.0 | 75.0 | blank | true | false |

Aggregate (all 696 DAO-only vs 972 matched):

| metric | DAO-only | matched |
|--------|----------|---------|
| peak_dao median | 19.2 ADU | 105.8 ADU |
| peak_max_adu median | 33520.6 | 33641.3 |
| flux median | 27.2 | 286.5 |
| peak_dao < 50 ADU | 683/696 (98%) | (few) |
| flux < 100 ADU | 643/696 (92%) | -- |
| negative flux | 142/696 (20%) | -- |
| snr50_ok true | 9/696 (1.3%) | 382/972 (39%) |
| edge_safe_10px false | 25/696 (3.6%) | 16/972 (1.6%) |
| likely_saturated | 1/696 | 10/972 |
| phot_g_mean_mag (matched only) | n/a | median 16.0, p95 17.4 |

DAO-only rows sit at sky pedestal (peak_max_adu ~33500) with tiny peak_dao
(13-33 ADU), low/negative flux, almost all snr50_ok=false. They do NOT resemble
normal stars just below Gaia G=17.5. Pattern matches noise/hot-pixel artifacts
just above the narrow sigma margin, not faint real sources.

## 4 -- Gaia catalog depth (draft_501)

From pipeline_meta:

| metric | value |
|--------|-------|
| catalog_rows (Gaia in cone) | 26504 |
| n_gaia_detected (DAO matched) | 972 |
| n_gaia_undetected | 25532 |
| n_dao_unmatched (DAO-only) | 696 |
| masterstars rows | 1668 |
| gaia_dao_completeness_pct | 98.17 (of Gaia sources in frame, DAO hit rate) |
| gaia_dao_completeness_raw_pct | 3.67 (DAO detections as fraction of Gaia cone) |
| match_depth | 18.0 |
| g_lim_50 | 17.5 (censored at DB max_g) |

Derived ratios:
  catalog coverage   = 972/26504  = 3.67%  (DAO reaches few Gaia sources)
  detection purity   = 972/1668   = 58.3%  (many DAO detections unmatched)
  Gaia missed by DAO = 25532/26504 = 96.3%

The two completeness metrics are NOT inconsistent:
  - gaia_dao_completeness_pct ~98%: of Gaia sources bright enough for DAO to
    detect, almost all get matched (fleming footprint reconcile).
  - gaia_dao_completeness_raw_pct ~3.7%: DAO overall depth is shallow vs full
    Gaia cone to G=17.5.

696 DAO-only are NOT the 25532 missing Gaia sources; they are extra detections
above sky with no catalog counterpart (section 3 artifact profile).

## 5 -- Wide rig comparison (draft_500, draft_435)

| metric | draft_501 Newton | draft_500 wide | draft_435 wide |
|--------|------------------|----------------|----------------|
| catalog_rows | 26504 | 100000 | 100000 |
| n_gaia_detected | 972 | 3554 | 2842 |
| n_gaia_undetected | 25532 | 96446 | 97158 |
| n_dao_unmatched | 696 | 561 | 109 |
| ms rows | 1668 | 4122 | 2951 |
| gaia_dao_completeness_pct | 98.17 | 90.07 | (not in meta; log 96.3% match) |
| gaia_dao_completeness_raw_pct | 3.67 | 3.55 | 2.84 |
| catalog coverage (n_gaia_det/cat) | 3.67% | 3.55% | 2.84% |
| detection purity (n_gaia_det/ms) | 58.3% | 86.2% | 96.3% |
| dao_only_fraction | 0.417 | 0.136 | 0.037 |
| sky_adu_per_px | 33482 | 1551 | 1570 |

All rigs show ~3-4% Gaia cone coverage by DAO (similar raw completeness).
Newton differs sharply in detection purity (58% vs 86-96%) and dao_only_fraction.

draft_500 has MORE ms rows and MORE Gaia matches but similar raw Gaia coverage;
its 561 DAO-only are a smaller fraction. Newton's 696 DAO-only dominate purity
loss despite fewer total detections.

## Files changed

None (read-only).

DRAFT501-BOTH -- evidence for both mechanisms
