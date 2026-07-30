> **PROVENANCE WARNING (added 2026-07-30).** The numbers in this document derive from **draft_000450** (comparative tables also use pre-013cb0c **draft_000435**), produced after the in-place preprocess architecture landed (`013cb0c`, 2026-07-22) and before the sky-surface idempotency guard (`84174ae`, 2026-07-30). During that window a repeated preprocess pass could subtract the sky surface twice, at a measured cost of order 500 ADU. **draft_000450** is no longer available, so its status is UNKNOWN, not clean. Treat these numbers as indicative, not validated.

# CURSOR RESULT - MASTERSTAR-COUNT-DIAG (2026-07-26)

Read-only diagnostic. No code changes, no commits. Scratch: `tmp/masterstar_dao_replay.py`.

---

## Executive summary

The **2.3x CSV gap** (2951 vs 6698) is real and **physical**, not a labelling artefact. It traces to
**different MASTERSTAR.fits pixel data** between drafts, not to the VSX matcher arc.

| Stage | draft_435 (anchor snapshot) | draft_450 (Friday re-run) | Ratio |
|-------|---------------------------:|--------------------------:|------:|
| **Pass-1 raw DAO** (`VY_NDAO` / pass 1) | **2552** | **8926** | **3.49x** |
| After pass-2 merge | 3777 | (not logged) | -- |
| After prematch SNR filter (1.8 sigma) | **2951** | (inferred ~6698 export) | **2.27x** |
| Gaia catalog-matched | **2842** | **3993** | 1.41x |
| CSV rows | **2951** | **6698** | 2.27x |

**Root cause (observed):** calibrated light frames are **byte-identical** between drafts (same dark
master `Dark_60s_Dark_0G_-10deg_Bin1_20260422.fits`), but **139/139 detrended aligned frames differ**
(max abs delta **3337 ADU**). The MASTERSTAR stack built from those frames differs (mean abs delta
**23 ADU**), producing **3.5x more pass-1 DAO detections** on the 450 image.

**Photutils 3.0 migration (`f2e7ce4`) is not the primary cause.** Re-running pass-1 `DAOStarFinder`
(photutils **3.0.0**, astropy **8.0.1**, sigma=2.1, FWHM=2.5 px) on the **frozen** FITS files:

| MASTERSTAR.fits | Stamped `VY_NDAO` | Current replay pass-1 |
|-----------------|------------------:|----------------------:|
| draft_435 | 2552 | **2579** (+1.1%) |
| draft_450 | 8926 | **9115** (+2.1%) |

Same detector on the 435 image stays near 2552; on the 450 image stays near 8926.

**Anchor gate blind spot confirmed:** `--full` copies frozen `MASTERSTAR.fits` and
`masterstars_full_match.csv` from the snapshot and **never rebuilds** detection or the stack
(`session_baseline_check.py` ~461-498). A DAO/detrending regression would not be visible there.

**Milan STOP implication:** Re-cutting the anchor today without understanding this would freeze either
(435) a shallow catalogue or (450) a noisier one -- neither is reproducible from the other under
current code without a deliberate MASTERSTAR rebuild policy.

---

## E1 - Reconcile the three counts

### What each number means (draft_435, from infolog + ledger)

| Label | Value | Definition |
|-------|------:|------------|
| `n_raw_dao` / pass 1 | **2552** | `DAOStarFinder` peaks on MASTERSTAR after background clip; stamped `VY_NDAO` |
| Pass-2 merge total | **3777** | Pass 1 + **1225** targeted cutouts at unmatched Gaia positions |
| After prematch SNR filter | **2951** | `median + 1.8 x sigma` noise floor (`masterstar_prematch_peak_sigma_floor`) |
| `catalog_matched` / ledger `matched` | **2842** | Rows with Gaia `source_id` after astrometry optimizer |
| CSV export rows | **2951** | `masterstars_full_match.csv` (= post-SNR detections incl. 109 `DAO_ONLY`) |

Ledger `VL-ANCHOR-WCSINV` fingerprint (`n_raw_dao=2552`, `matched=2842`) matches the **435** build
log and `MASTERSTAR.fits` header exactly.

### draft_450 funnel (partial -- build log absent from infolog)

| Label | Value | Source |
|-------|------:|--------|
| Pass-1 raw DAO | **8926** | `VY_NDAO` header; `pipeline_meta.json` / `field_density.json` |
| CSV export | **6698** | `masterstars_full_match.csv` |
| Gaia catalog-matched | **3993** | CSV `source_type=GAIA_MATCHED`; meta `n_gaia_detected=3992` |
| `DAO_ONLY` | **2705** | Unmatched detections kept for QA |

Pass-2 / SNR-filter line items were **not logged** in the surviving `draft_450` infolog (run starts
mid-pipeline with MASTERSTAR already present).

### Like-for-like comparison

**Compare at pass-1 raw DAO:** **8926 / 2552 = 3.49x** -- this is the physical detection increase.

**CSV ratio 6698 / 2951 = 2.27x** is smaller because both pipelines apply pass-2 merge and prematch
filtering after raw DAO; the 435 funnel removes more relative to raw (2552 -> 2951 is +16% via pass-2
net; 8926 -> 6698 is **-25%** filtering).

**Definitional vs physical split (approximate):**

- **~3.5x** of the gap is **different MASTERSTAR image / detrending** (physical).
- **~0.65x** remaining ratio compression is **post-detection filtering/export** (definitional funnel
  differences; exact 450 funnel not logged).

---

## E2 - Config delta (provenance snapshots)

Git commits at run time: **435 = `10d610c0`**, **450 = `cb78b25`** (current HEAD).

**Masterstar detection parameters -- no meaningful difference:**

| Parameter | draft_435 | draft_450 |
|-----------|----------:|----------:|
| `masterstar_dao_threshold_sigma` | 2.1 | 2.1 |
| `masterstar_prematch_peak_sigma_floor` | 1.8 | 1.8 |
| `masterstar_best_of_n` | 10 | 10 |
| `masterstar_detection_cap_adaptive` | true | true |
| `masterstar_detection_cap_k` | 0.08 | 0.08 |
| `masterstar_detection_cap_max` | 800 | 800 |
| `masterstar_detection_cap_min` | 250 | 250 |
| `masterstar_catalog_recovery_min` | 0.65 | 0.65 |
| `masterstar_min_matched_floor` | 40 | 40 |
| `catalog_query_max_rows` | 15000 | 15000 |

**Recorded differences (snapshot serialization / pipeline context, not detection sigma):**

| Parameter | draft_435 | draft_450 |
|-----------|----------|----------|
| `vsx_variable_targets_mag_limit` | **14.5** | **absent** (removed by `a0e3431`) |
| `skip_processed_directory` | false | absent (param removed `013cb0c`) |
| `masterstar_use_best_frame_fwhm` | absent | **true** |
| `osc_channel_binning` | absent | 2 |
| `qc_preprocess_workers` | absent | 8 |
| Several `masterstar_odds_*` / SIP guard keys | present | absent from snapshot blob |

**Conclusion:** The 2.3x CSV gap is **not explained by a changed DAO threshold or detection cap**.
Config snapshots agree on all detection-sigma knobs.

---

## E3 - Input delta

### Frames in the MASTERSTAR stack

| Item | draft_435 | draft_450 |
|------|-----------|-----------|
| Detrended light count | 139 (+ `MASTERSTAR.fits`) | 139 (+ `MASTERSTAR.fits`) |
| Naming | `proc_BO_CVn_Light_NNN.fits` | `BO_CVn_Light_NNN.fits` |
| Same underlying indices | 001-150 (139 frames) | 001-150 (139 frames) |
| Pairwise pixel match (435 proc vs 450 plain) | **0/139 identical** | max delta **3337 ADU** |

Same frame **list**; **different detrended pixel values** on every frame.

### Calibration masters

Both drafts -- **identical** from `cal_diag.json` and calibrated FITS headers:

| Master | Value |
|--------|-------|
| Dark | `CalibrationLibrary/Dark_60s_Dark_0G_-10deg_Bin1_20260422.fits` |
| Flat median | `VY_FLATM = 32975.0` |
| `m_S`, `sigma_r`, `sky_median` | identical numerically |

**Fresh darks around 2026-07-21 (STATE note) do not explain the gap** -- both builds used the
**20260422** dark.

### Exposure / binning / temperature

From `MASTERSTAR.fits` headers (both drafts): `EXPTIME=60`, `XBINNING=YBINNING=2`, `CCD-TEMP=-10`,
`NAXIS=2082x1397`. Identical.

---

## E4 - Code delta and detection replay

### Commits between `10d610c0` (435) and `cb78b25` (450) touching detection/calibration path

| Priority | Commit | Date | Suspect rationale |
|----------|--------|------|-------------------|
| **1 (tested -- not primary)** | `f2e7ce4` | 2026-07-21 | photutils 3.0 + astropy 8.0; `--full` used frozen MASTERSTAR so gate could not see DAO drift |
| **2 (high)** | `013cb0c` / `263c6e7` | 2026-07-22 | SKIPPROC: skip-only preprocess, QC allowlist gating -- **all detrended frames differ** |
| **3 (medium)** | `0f1c07f` / `224c442` | 2026-07-22 | OSC arc: shared calibration/registration touched even for mono |
| 4 | `12887bc` | 2026-07-21 | calibration import fix |
| 5 | `a0e3431` | between | VSX scope by DAO+Gaia; mag limit removed -- explains 450 **re-run intent**, not raw DAO |
| 6 | `07108b8` | earlier | out-of-scope VSX filter |

### Decisive test (E4 requirement)

**Setup:** frozen `draft_435` `MASTERSTAR.fits`, current env, pass-1 only (no stack rebuild).

| Quantity | Value |
|----------|------:|
| photutils | **3.0.0** |
| astropy | **8.0.1** |
| `masterstar_dao_threshold_sigma` | **2.1** |
| FWHM passed to finder | **2.5 px** (config default) |
| Threshold ADU | **176.0** (= 2.1 x bg_std 83.8) |
| **n_raw_dao replay** | **2579** |
| Stamped `VY_NDAO` on FITS | **2552** |

**Verdict on `f2e7ce4`:** +27 sources (+1.1%) on the **same** 435 image -- negligible vs 8926/2552 gap.
**Primary suspect moves to detrending/stack rebuild (SKIPPROC era)**, which changed every aligned frame
and the MASTERSTAR pixel array.

**Missing on disk:** `draft_450` infolog contains **no MASTERSTAR build section** (no `DAO pass 1`
lines). The 8926-stamp build occurred in an unlogged or truncated session. Milan may need to supply
that log segment or re-run with full infolog capture.

---

## E5 - Which count is right?

**Mixed -- not a clean "deeper is better" outcome.**

| Metric | draft_435 | draft_450 | Reading |
|--------|----------:|----------:|---------|
| Pass-1 raw DAO | 2552 | 8926 | 450 finds many more peaks |
| Gaia-matched in CSV | 2842 (96.3% of rows) | 3993 (**59.6%** of rows) | 450 adds **+1151** real Gaia IDs but **match purity per row drops sharply** |
| `DAO_ONLY` (no Gaia counterpart) | 109 (3.7%) | **2705 (40.4%)** | 450 exports **25x more** unmatched peaks |
| Gaia->DAO completeness (raw) | **2.84%** | **3.99%** | Slight field-level completeness gain |
| Matched mag p50 | **13.52** | **14.05** | 450 goes slightly fainter |
| HRD flux filter | n/a | **6087/6698** (91%) DAO-like flux | 911 rows fail even flux-based DAO check |

**Reading:**

- **450 is deeper** in the sense of recovering **+1151 more Gaia counterparts** (including the 77
  VSX rows only present in the 450 catalogue -- see MATCHER-FIX-3-DIAG).
- **450 is noisier:** 40% of exported rows have **no Gaia ID** vs 4% in 435; raw completeness only
  rose 2.84% -> 3.99% despite 3.5x more pass-1 peaks. A detector that finds 3.5x more sources but
  only 1.4x more Gaia matches is picking substantial **non-astrophysical structure** from the changed
  detrended background.

**For the identity gate:** draft_435 anchor snapshot (2552/2951) is **conservative and Gaia-pure**;
draft_450 (8926/6698) is **more complete for VSX cross-identification** but **not reproducible** from
435 inputs without rebuilding the entire detrended + MASTERSTAR chain.

**Neither count should be treated as "the" target until Milan picks a rebuild policy** (re-cut anchor
from a fresh MASTERSTAR build with current code, or freeze 450 as the new reference night).

---

## `--full` anchor gate blind spot

**Confirmed correct.** `session_baseline_check.py --full`:

1. Requires existing snapshot artefacts (`MASTERSTAR.fits`, `masterstars_full_match.csv`,
   `variable_targets.csv`, aligned lights).
2. Calls `run_full_photometry_pipeline(...)` with those frozen paths.
3. Compares photometry SHA to the snapshot.

It does **not** call `generate_masterstar_and_catalog`, stack frames, or run `DAOStarFinder` on raw/
detrended data. Detection and stack rebuild sit **outside** the anchor gate -- same class of blind
spot as plan-time VSX matching before B3.

---

## What is missing / Milan action

1. **MASTERSTAR build infolog for draft_450** (DAO pass 1/2 lines, stack attempt label, best-of-N
   frame list) -- not in `infolog_20260725_002337.txt`.
2. **Explicit policy:** which MASTERSTAR catalogue is authoritative for BO CVn anchor night going
   forward (435 snapshot vs 450 re-run).
3. **Optional:** re-run MASTERSTAR build once with current code + full logging into a scratch draft,
   to see whether 6698/8926 is stable or code still drifting.

---

## Files referenced (read-only)

| Path | Role |
|------|------|
| `Archive/Drafts/draft_000435/platesolve/NoFilter_60_2/MASTERSTAR.fits` | Anchor snapshot stack (`VY_NDAO=2552`) |
| `Archive/Drafts/draft_000450/platesolve/NoFilter_60_2/MASTERSTAR.fits` | Friday stack (`VY_NDAO=8926`) |
| `Archive/Drafts/draft_000435/infolog_20260716_123126.txt` | Full 435 DAO funnel log |
| `Archive/Drafts/draft_000450/platesolve/NoFilter_60_2/photometry/pipeline_meta.json` | 450 provenance + `n_stars_dao=8926` |
| `dev/validation/VYVAR_VALIDATION_LEDGER.json` | `VL-ANCHOR-WCSINV` census fingerprint |
| `tmp/masterstar_dao_replay.py` | Pass-1 DAO replay script |

**No source files modified. No commits. STOP remains in force.**
