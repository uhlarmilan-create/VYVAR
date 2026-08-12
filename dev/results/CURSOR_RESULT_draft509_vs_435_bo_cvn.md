CURSOR RESULT - 2026-08-12

What I did
Read-only investigation of why draft 509 BO CVn does not replicate draft 435. Parts 1-4 from archived drafts; Part 5 already satisfied by draft 509 itself (byte-identical raw re-run on HEAD `682f40c`). No code fixes.

## Output / findings

### Part 1 - Comparability (SAME NIGHT)

**Drafts 435 and 509 are the same night and the same raw files** (DATE-OBS 2026-04-23, MD5-identical sample FITS, identical 150-name set). Differences below are processing/code/config, not sky.

| item | draft 435 | draft 509 |
|---|---|---|
| target / rig / telescope / camera | BO CVn; equip 1 / tel 1 / loc 2; Sample Primary 200@F/2.9; QHY294PROM | same |
| date of observation (UT) | 2026-04-23T19:35 ? 2026-04-24T00:35 | same (byte-identical RAW) |
| filter / exposure / binning | NoFilter / 60 s / 2x2 | same |
| n frames imported / to photometry | 150 / 139 | 150 / 134 |
| airmass range | 1.013-1.219 (from LC) | 1.013-1.219 |
| median sky (inspection / dyn) | insp ~2428 ADU; dyn sky_adu 1570 | insp ~2429; dyn 1553 |
| moon | ~48% illum, alt ~27 deg, sep ~54 deg, MEDIUM | same |
| per-frame FWHM insp min/med/max | 4.95 / 5.14 / 5.82 | 5.20 / 5.31 / 6.19 |
| MASTERSTAR VY_FWHM / VY_FWHM_GAUSS | 3.207 / **2.395** | 5.195 / **3.301** |
| aperture target | 2.716 px | 4.141 px |
| aperture comps | 2.566, 2.716, 2.316 | 3.941, 4.141, 3.541, 3.541, 3.341 |
| comp catalog_ids | 3x TIER1: ...531712, ...564608, ...698240 | 5x TIER1: same 3 + ...2858240 + ...0107904 |
| check star / scatter | ...1001088 @ K~8.79 / **0.0079** | ...4892800 @ K~10.12 / **0.0252** |
| aperture corr ?M | yes, ?0.222 | yes, ?0.136 |
| git | `10d610c` (2026-07-16) | `682f40c` (2026-08-12) |

OOE definition (`compute_lc_rms_ooe`): **brightest tertile** of mag_calib (quantile 0.33), not a phase window. For EW BO CVn this includes rising/falling branch near max ? `lc_rms_ooe?0.05` is partly metric artifact; do not drive diagnosis.

### Part 2 - Bimodality

- Target `mag_inst` / `delta_mag` residuals: **unimodal** (~0.012-0.013).
- Target `mag_calib_raw` / `mag_calib_final` residuals: **bimodal** (KDE peaks ? ?0.037 and +0.011; branch offset **?54 mmag**).
- Check-star `kmag` residuals: **bimodal** (peaks ? ?0.038 / +0.015; offset **?49 mmag**).
- Clean separator: **per-frame ZP sigma-clip rejection of bright TIER1 comp `1497771992240531712`** (37/134 frames). Crosstab vs low residual mode: 37/37 rejected frames in low branch; 95/97 kept frames in high branch.
- Always-drop-faint-only (no intermittent bright rejection) ? unimodal, std ? 0.013.
- Frame parity / FWHM / sky / airmass / aperture / centroid integer hop: do **not** cleanly separate.

### Part 3 - Centroids

| star | 509 dist?1/2/3 px | 435 dist?1/2/3 px |
|---|---|---|
| target + 5 comps / check | max 1.4 px; **0 frames ?2 px** | max ~0.17 px; **0 ?1 px** |

- 509: integer-pixel coords on **all 134** proc frames ? `_lock_matched_centroids_to_master_grid` fired (master_reference_sky). Lock moves centroids to brightest pixel in search window; not logged as a draft-level total, but n_locked = matched stars every frame.
- 435: sub-pixel DAO centroids (lock did not exist). No neighbour-star hijack signature (?2-3 px) on either draft.

### Part 4 - Config / aperture chain

Source: `pipeline_meta.json` ? `provenance.config_snapshot`.

84 differing keys. Flux-path highlights:
- `phase01_comparison_max_mag_diff`: 1.5 ? **2.0** (admits fainter comps)
- `phase01_comparison_n_comp_min`: 3 ? 2
- `comp_max_delta_bprp`: 0.79 ? 0.99
- `masterstar_dao_threshold_sigma`: 2.1 ? 3.8
- `aperture_snr_sizing`: missing ? `{small:1.5, large:4.0}`
- size-class `aperture_fwhm_factor_{small,medium,large}` removed in 509
- `dao_centroid_max_shift_fwhm`: missing ? 1.0
- `comp_clip_sigma` / several frame_quality keys present only in 435

Aperture 4.1 px: sizing consumed **`VY_FWHM_GAUSS = 3.3014`**, not insp FWHM ~5.3.
- Resolver: `photometry_core.py:1801-1821` / Phase 2A `8416-8437`
- Runtime stamp: proc `fwhm_px_for_aperture=3.3014`, `fwhm_px_scope=per_draft_gaussian_override`, `snr_aperture_mode=snr_table`
- 4.1/3.30 ? SNR-table factor; **not** an implied 2.2 px moment FWHM bug.

Instrumental check-star scatter (flux ? 3-comp ensemble) is **~0.008-0.009 on both drafts** for both check stars. Official 0.025 is post-ensemble.

### Part 5 - Controlled experiment

**Already executed as draft 509:** same RAW as 435, full pipeline on HEAD `682f40c`, new MASTERSTAR (not frozen).

**Result B** - re-run produces degraded official LC/check metrics on the same raw data. Code/config regression vs 435 (`10d610c`), not night quality.

(Bisect not run; Parts 2+4 already name the mechanism. First candidate for a later bisect if needed: whatever widened comp admission to 5 stars + retained ZP sigma-clip at `ensemble_normalize` when `len(z) >= 4`.)

### Cause (chain)

Same raw ? HEAD admits **5** comps (mag-diff 2.0), including faint `...0107904` (G=11.52, rms=0.021) ? `ensemble_normalize` ZP path (`photometry_core.py:3461-3479`) runs **3? MAD clip only when ?4 comps** (435's 3 comps never entered) ? faint comp rejected every frame; bright TIER1 `...531712` rejected on **37/134** frames ? **~50 mmag two-state zeropoint** ? parallel tracks on calibrated LC and check-star scatter 0.025 (vs MERR 0.007). Instrumental photometry and centroids are not the defect.

## Errors (if any)
None.

## Files changed
None (investigation only). Scratch under `tmp/_cmp_*.py`, `tmp/_sim_*.py`, `tmp/_confirm_*.py`.
