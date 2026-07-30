# CARRY-01 — Per-frame diff table (new profiler `byte_compare`)

**Generated:** 2026-07-30 (CARRY-01 Part A3)  
**Method:** `dev/scripts/post453_preprocess_bench.py` ? `byte_compare()`  
**Comparison:** draft **452** calibrated FITS (reference, untouched) vs same frames after in-place re-run of `_qc_enrich_calibrated_in_place` on a bench copy.

## Live recomputation status: BLOCKED

`Archive/Drafts/draft_000452/` is **not present** on this machine (`Archive/` is gitignored; only `draft_000435` and snapshots remain). Attempted run exited immediately (`total_s ? 0.01`, no frames processed).

**To unblock:** restore `draft_000451` and `draft_000452` under `Archive/Drafts/`, then:

```powershell
cd C:\ASTRO\python\VYVAR
python dev/scripts/post453_preprocess_bench_per_frame.py   # to be added in CARRY-02
```

---

## Per-frame table (new profiler comparison)

| frame | max_abs_diff | mean_abs_diff | n_pixels_nonzero |
|-------|-------------:|--------------:|-----------------:|
| *pending live recompute* | — | — | — |

*(10 frames: `BO_CVn_Light_001.fits` … `BO_CVn_Light_010.fits`)*

---

## Aggregates (new profiler comparison)

| Aggregate | Value | Status |
|-----------|------:|--------|
| (i) max over all 10 frames | **508.969482421875** | from committed `session_20260730_preprocess_profile/preprocess_profile.csv` (aggregate only) |
| (ii) max excluding frame001 | *pending* | requires live recompute |
| (iii) frame001 alone | *pending* | requires live recompute |

---

## Related evidence — OLD comparison (451 input + preprocess vs 452 output)

From `dev/results/context/session_20260728_post453_fixes/frame001_investigation.csv` (commit `6c0a524`, measured when drafts were on disk). **Not the same comparison as the new profiler**, but documents frame001 dominance for the cross-draft identity check the old profile CSV describes.

| frame | max_abs_diff (ADU) | note |
|-------|-------------------:|------|
| BO_CVn_Light_001.fits | 533.450 | identity: 451-cal + new preprocess vs 452-cal output |
| BO_CVn_Light_002.fits | 0.0 | |
| BO_CVn_Light_003.fits | 0.0 | |
| BO_CVn_Light_004.fits | 0.0 | |
| BO_CVn_Light_005.fits | 0.0 | |
| BO_CVn_Light_006.fits | 0.0 | |
| BO_CVn_Light_007.fits | 0.0 | |
| BO_CVn_Light_008.fits | 0.0 | |
| BO_CVn_Light_009.fits | 0.0 | |
| BO_CVn_Light_010.fits | 0.0 | |

Old-comparison aggregates (from investigation CSV):

| Aggregate | Value |
|-----------|------:|
| (i) max all frames | 533.450 |
| (ii) max excl frame001 | **0.0** |
| (iii) frame001 alone | 533.450 |

Pre-preprocess 451-vs-452 calibrated input diff (frame001): **659.646 ADU** (same CSV, column `max_abs_diff_451_vs_452_calibrated_ADU`).
