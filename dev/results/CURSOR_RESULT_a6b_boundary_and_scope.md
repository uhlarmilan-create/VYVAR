CURSOR RESULT - 2026-08-06 (A-6b confirmable depth and installation scope)

What I did
Replaced A-6 single boundary (max_g_mag - population margin) with derived
confirmable_depth_g, per-row sigma_g margin, renamed classes, added cap-relative
columns, split-detection measurement, ASCII sanitisation of five result docs.

## 1. confirmable_depth_g per draft

Formula: min(gaia_db_max_g_mag, effective_match_depth, cone_query_mag_limit).

| draft | confirmable_depth_g | winner | gaia_db_max_g_mag | match_depth | cone_query_mag_limit |
|-------|---------------------|--------|-------------------|-------------|----------------------|
| 501 | 17.50 | gaia_db_max_g_mag | 17.5 | 18.0 | n/a in meta |
| 435 | 17.50 | gaia_db_max_g_mag | 17.5 | 18.0 | n/a |
| 500 | 17.50 | gaia_db_max_g_mag | 17.5 | 18.0 | n/a |

Gaia DB identity (all drafts): fingerprint 921ecb430eabd2f5..., row_count 211712600,
max_g_mag 17.5. Phantom band 17.5-18.0 from MASTERSTAR floor no longer drives
classification because DB cap wins min().

## 2. Margin distribution (per-row sigma_g)

Formula: sigma_g(row) = hypot(zp_residual_rms, 1.0857 / SNR(row)); SNR = flux / noise_adu.
fleming_sigma_mag retained as population diagnostic only.

| draft | zp RMS | sigma_g median | sigma_g p90 | A-6 margin (ref) |
|-------|--------|----------------|-------------|------------------|
| 501 | 0.431 | 0.80 mag | 3.19 | 1.22 |
| 435 | 0.837 | 1.59 mag | 3.42 | 0.81 |
| 500 | 0.946 | 1.30 mag | 2.64 | 0.81 |

Per-row margin is ~0.5-1.6 mag (median), not ~1.2 mag flat. Effect: A-6 band
16.28-17.5 on draft_501 becomes mostly ambiguous_depth (310) instead of false
below_catalogue (525).

## 3. Corrected counts and delta vs A-6

| draft | class | A-6 | A-6b | delta |
|-------|-------|-----|------|-------|
| 501 | artifact_negative | 142 | 142 | 0 |
| 501 | below/unmatched_in_range | 525 / 14 | 203 / 26 | -322 / +12 |
| 501 | unconfirmed/ambiguous | 14 / 0 | 310 / - | +310 new band |
| 501 | indeterminate | 15 | 15 | 0 |
| 435 | artifact_negative | 8 | 8 | 0 |
| 435 | below/unmatched | 0 / 98 | 0 / 81 | -17 to ambiguous |
| 435 | ambiguous_depth | - | 17 | +17 |
| 500 | artifact_negative | 48 | 48 | 0 |
| 500 | below/unmatched | 8 / 496 | 0 / 455 | -8 / -41 |
| 500 | ambiguous_depth | - | 49 | +49 |

artifact_negative on draft_501: 142 (control, unchanged).

## 4. Split-detection measurement (unmatched_in_range)

| draft | n_in_range | median dist to nearest matched (px) | within 5 px | within 5 px (neighbour G<16.3) | control matched median |
|-------|------------|--------------------------------------|-------------|--------------------------------|------------------------|
| 435 | 81 | 15.05 | 4.9% | 5.0% | 16.64 |
| 500 | 455 | 10.76 | 17.1% | 16.7% | 14.54 |
| 501 | 26 | n/a | n/a | n/a | n/a |

Verdict: wide-rig in-range unmatched rows sit far from bright Gaia matches (median
10-16 px vs FWHM ~3 px). Only ~5-17% within 5 px; not consistent with dominant
on-star split detections at 1-2 px. More likely deblended structure / spurious
detections at larger separation, or ambiguous photometry at the depth boundary.

## 5. SNR-floor confound (final CSV snr50_ok on DAO_ONLY)

| draft | DAO_ONLY | snr50_ok=False | fraction |
|-------|----------|----------------|----------|
| 435 | 109 | 93 | 85.3% |
| 500 | 561 | 248 | 44.2% |
| 501 | 696 | 687 | 98.7% |

Pre-match detection SNR floor removed ~826 rows (21.9%) on draft_435 upstream;
draft_501 faint population largely retained. Cross-draft class comparisons are
confounded by this asymmetry.

## 6. Local A/B and tests

P1 mini headless core SHA (with A-6b, frozen mini inputs): unchanged from A-6 local
A/B at f5e69aa baseline:

`aa72e97979a74d5b8297c6bc3624bee668d8bd5f28624de0a708149e286c2636` (n=325)

Photometry path unchanged; classification is additive on masterstars only.

ASCII policy: PASS (5 result docs sanitised).
Remaining non-slow suite failures (4):
- test_invariants_p1_golden.py::test_headless_chain_sha (stale P1 ledger fingerprint)
- test_invariants_p1_seed.py::test_p1_snapshot_sha_matches_registered
- test_invariants_p1_seed.py::test_p1_census_fingerprint_in_meta
- test_params_registry.py::test_generated_params_md_is_fresh (generated docs stale)

## 7. Undecidable / installation scope

Class counts depend on local Gaia DB max_g_mag and noise estimate; do not compare
across installations or wire to gates. Comparable quantity: implied-G distribution
(implied_g_mag, implied_g_minus_depth per row; deciles in pipeline_meta).

Whether unmatched_in_range rows are real uncatalogued sources vs artifacts remains
undecidable at detection stage (DAO-PHYS campaign).

## Files changed

- src_py/dao_reconcile.py (core)
- src_py/pipeline.py, photometry_report.py, ui_masterstar_qa.py
- dev/tests/test_dao_reconcile.py
- dev/tools/a6_classify_offline.py, dev/tools/a6b_split_detection_measure.py
- docs/VYVAR_LIMITATIONS.md
- dev/results/CURSOR_RESULT_*.md (ASCII sanitise)
