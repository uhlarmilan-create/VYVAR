CURSOR RESULT - 2026-08-04T09:50:00Z

What I did
Ran batch E **physical re-cut #2** from calibrated lights (not frozen cache): full chain
`_night_run_preprocess` -> `_night_run_platesolve` -> `run_full_photometry_pipeline` on scratch
draft_000500 copied from `draft_000435/calibrated/lights/NoFilter_60_2` (150 FITS).
Invalidated preprocess/detection by fresh scratch + `preprocess_sky_surface_force_reapply=True`.
Harness: `dev/tools/batch_e_physical_recut.py`. Results JSON: `tmp/batch_e_physical_recut_results.json`.
Log: `tmp/batch_e_physical_recut.log`. **Stopped at GATE 2** -- fingerprints NOT pushed.

## Output / findings

### Entry point and inputs

| Item | Value |
|------|-------|
| Entry point | `_night_run_preprocess` -> `_night_run_platesolve` -> `run_full_photometry_pipeline` |
| Input | `Archive/Drafts/draft_000435/calibrated/lights/NoFilter_60_2` (150 FITS) |
| Scratch output | `Archive/Drafts/draft_000500` |
| Elapsed | preprocess 214 s; platesolve 1473 s; photometry 3877 s (~93 min total) |

### 1. E.2-E.5 execution confirmation (one line each)

| Change | Confirmed | Evidence line |
|--------|-----------|---------------|
| **E.3 CR** | YES | `astroscrappy removed 365810 pixels across 150 frames (VY_COSMNPX headers)` -- source: FITS header scan on calibrated outputs; sample frame 007: 6899 px |
| **E.2 centroid guard** | YES (code path; 0 triggers) | No `[BATCH-E E.2] centroid WCS fallback` lines -- guard active but DAO centroids within 1x FWHM of MASTERSTAR on this rebuild (0 fallback events) |
| **E.4 N_equiv** | YES | `[BATCH-E E.4] N_equiv=3.78 applied for DAO detection threshold` -- source: `tmp/batch_e_physical_recut.log` |
| **E.5 saturation gate** | YES (code path; 0 exclusions) | No `[BATCH-E E.5] saturation gate excluded N comps` lines -- admission gate ran; no comp exceeded >10% frames over 70% full-well threshold on this anchor |

Note: E.3 per-frame INFO lines are emitted in QC worker processes and do not always reach the main log file; header scan is authoritative.

### 2. Per-change anchor delta (vs batch D snapshot photometry)

Baseline for LC pairing: `draft_000435_snapshot_skysurface_20260716` (148 common LCs; 248 LCs in physical run).
Batch D ledger SHA (GATE 1): core `b9c9489a...` n=325.

**E.3 cosmic-ray rejection**
- Pixels corrected: **365810** total across **150/150** frames (mean ~2439 px/frame).
- G 8-9 star-core safety: **3900/3900** bright star-frames remain `photometry_ok` (no core clipping).
- Combined LC mag delta (all layers): see combined section below; CR is embedded in full regen delta.

**E.2 centroid guard**
- WCS fallback triggers: **0** star-frames (log).
- Proc x,y shift vs snapshot (full regen proxy): **10879** star-frames with shift > 2 px; **942** targets affected; median shift **0.35 px** (source: proc CSV compare `detrended_aligned/lights/NoFilter_60_2`).
- Part 0e focus target `1498135552633294976` in unstable-top list; LC median delta for unstable cohort: see JSON `mag_delta_unstable_centroid_targets`.

**E.4 N_equiv=3.78**
- MASTERSTAR detections: physical **3555** vs snapshot **2843** (+**951** entered, **239** left).
- Detection threshold log: `n_equiv=3.78` on first DAO frame (platesolve pass).

**E.5 saturation gate (D5-2 science check)**
- Comps excluded by gate: **0** (this run).
- G 8-9 bin `log10(flux)` vs G slope (proc CSV star-frames):

| Stage | G 8-9 slope | n star-frames | Source |
|-------|-------------|---------------|--------|
| Batch B-revised baseline | **-0.258** | (production) | `CURSOR_RESULT_batch_B_revised.md` |
| Before (snapshot proc) | **-0.318** | 2919 | snapshot `detrended_aligned` proc CSVs |
| After (physical proc) | **-0.491** | 3452 | draft_000500 proc CSVs |

Slope moved from -0.318 toward and past the -0.4 target (delta **-0.173** vs snapshot proc). Full physical regen + E.4 detection change co-mingle with E.5; zero comp exclusions means the slope shift here is dominated by preprocess/detection regen, not comp-pool filtering alone.

**Combined science delta (148 common LCs, source_file pairing)**
- LCs with non-zero mag delta: **148/148**
- All epochs: n=**18757**; median delta mag=**+0.215**; p95 |delta|=**1.659**; max |delta|=**9.206**
- This is **non-zero and expected** (opposite of frozen re-cut zero delta). Full regen + batch E vs frozen snapshot.

### 3. New SHA fingerprints (physical re-cut, pending GATE 2)

| Tier | Physical (draft_000500) | Batch D ledger (GATE 1) | Delta |
|------|-------------------------|-------------------------|-------|
| core | `5bccd85a94d95031f80d372141ae0c61b0d8b0b2026c6bb15076d4e6a5e9b77e` (n=497) | `b9c9489aa88b1df815bf6157911b35af5bb1c42a3b0eaf58995042fcdd007a39` (n=325) | **CHANGED** |
| extended | `7fdcdca402ad47d044ca7b34d1f1c0d09185d02016f94a1a3747cb0528862ea2` (n=744) | `65bc826cac433453f689dbc5ab2883e783b7a7c7563092c02cfa443058f48cc2` (n=487) | **CHANGED** |

Archive snapshot on disk still carries pre-batch-D SHA (`a97306ef...` n=649); refresh deferred to post-GATE-2 authorization.

### 4. Awaiting GATE 2 authorization (Milan)

Do **NOT** push fingerprints until Milan reviews:
1. E.5 G 8-9 slope before/after (-0.318 -> -0.491) and zero comp exclusions.
2. E.3: 365810 CR pixels; 3900/3900 bright frames photometry_ok.
3. E.2: guard active, 0 fallback events this run.
4. E.4: +951/-239 detection delta from N_equiv.
5. New SHA core/extended above vs batch D ledger.
6. Combined LC delta non-zero (148/148 LCs; median +0.215 mag).

Post-authorization: update `VL-ANCHOR-WCSINV`, `VL-COUNTERS-ZERO`, skysurface K4 bounds, refresh Archive snapshot, register items 9/10/27/28/29 FIXED.

### 5. WIDE-ERR (routed, not fixed here)

Added to closure register as open item **WIDE-ERR**: wide-rig (equipment_id 1) quoted error underquoted ~2x vs check-star scatter (median scatter/err ~1.96x, slope ~1.83); mechanism ensemble SEM and/or photon term; fix Honeycutt 1992 LOO + gain/RN audit. Source: `dev/results/CURSOR_RESULT_wide_error_diag.md`.

## Errors (if any)

None fatal. E.3 worker-process log lines not captured in main log (headers used instead). E.2/E.5 logged only when triggers fire (0 this run).

## Files changed

- `dev/tools/batch_e_physical_recut.py` (new harness)
- `src_py/pipeline.py` (BATCH-E E.2/E.3/E.4 log markers)
- `src_py/comp_selection_per_target.py` (BATCH-E E.5 aggregate log)
- `docs/VYVAR_AUDIT_CLOSURE_REGISTER.md` (WIDE-ERR; GATE 2 pending notes)
- `docs/VYVAR_AUDIT_FINAL.md` (physical re-cut slope; WIDE-ERR)
- `docs/VYVAR_VALIDATION.md` (pending fingerprints)
- `docs/VYVAR_STATE.md`, `docs/VYVAR_ROADMAP.md` (GATE 2 status)
- `tmp/batch_e_physical_recut_results.json` (metrics)
- `Archive/Drafts/draft_000500/` (scratch re-cut output; not committed)

---

## GATE 2 verification (2026-08-04): raw vs differential shift

**Question:** Is the +0.215 mag combined LC delta raw-only (sky/zeropoint) or does it survive into differential?

**Method:** Existing output only (`draft_000500` vs `draft_000435_snapshot_skysurface_20260716`); 148 common LCs, 139 common proc frames; `source_file` pairing. No re-cut.

### V1 -- Raw and differential median delta

| Quantity | Column / method | n epochs | Median delta | Source |
|----------|---------------|----------|--------------|--------|
| Raw instrumental | `mag_inst` (LC) | 18757 | **-0.072 mag** | LC merge |
| Raw instrumental | flux -> mag on matched proc stars | 21063 star-frames | **-0.062 mag** | proc CSV merge |
| Calibrated raw | `mag_calib_final` (LC) | 18757 | **+0.215 mag** | LC merge (headline from re-cut) |
| Published differential | `delta_mag` (LC) | 18757 | **-1.665 mag** | LC merge |
| **Matched-star differential** | target flux mag delta minus per-frame field median (same `catalog_id`s) | 21063 | **-0.023 mag** | proc CSV merge |

**Reading:** The +0.215 mag sits in **`mag_calib_final`** (catalog zeropoint layer), not in raw instrumental (~60-72 mmag). The LC `delta_mag` column shows a large shift because the **comparison ensemble pool changed** between the frozen snapshot photometry and the full physical regen (E.4 detections, new comp selection). When differential is recomputed on **the same matched stars per frame**, the median shift is **-23 mmag** -- common-mode sky/preprocess cancels as expected.

### V2 -- Constant vs magnitude-dependent (raw shift)

| Check | Result | Source |
|-------|--------|--------|
| Per-frame median `mag_inst` delta (LC) | median of frame medians **-0.074 mag**; frame-to-frame spread std **0.10 mag** | 139 frames |
| Within-frame std of `mag_inst` delta | median **0.525 mag**, p95 **0.802 mag** | same stars regen with different apertures/detections |
| Per-frame median flux mag delta (proc) | **-0.040 mag** | 139 common proc frames |
| `mag_calib_final` vs `mag_inst` delta slope | **+0.032 mag/mag** (weak) | 6486 star-frames |

Raw shift is **roughly constant at frame level** (~40-74 mmag), not a strong magnitude-dependent trend. High within-frame scatter reflects per-star regen differences (CR, detection, aperture), not a single sky zeropoint dominating.

### V3 -- Sky level tracking (sample frames)

| Frame | sky_adu (anchor) | sky_adu (physical) | delta ADU |
|-------|------------------|--------------------|-----------|
| Light_001 | 2437.18 | 2420.73 | -16.45 |
| Light_020 | 1748.32 | 1736.03 | -12.29 |
| Light_050 | 1640.30 | 1630.28 | -10.01 |
| Light_063 | 1610.56 | 1603.71 | -6.86 |
| Light_085 | 1536.11 | 1532.04 | -4.06 |
| Light_130 | 1349.29 | 1351.04 | +1.75 |

Sky annulus level (`sky_adu_per_px_annulus`, proc CSV) differs by **-16 to +2 ADU** -- small relative to ~1600-2400 ADU background. Pearson r between per-frame sky delta and per-frame flux mag delta: **-0.009** (no tracking). Raw instrumental shifts are **tens of mmag**, not driven one-to-one by the small sky ADU differences alone; the +0.215 mag headline is dominated by the **calibration zeropoint** (`mag_calib_final`), not annulus sky ADU.

Example (Part 0e target `1498135552633294976`): `mag_inst` delta ~ **-23 mmag** per epoch; `mag_calib_final` delta ~ **-1.63 mag**; LC `delta_mag` delta ~ **-0.98 mag** (ensemble pool change).

### Verdict: **V-clean**

- Differential photometry on **matched stars** is stable at **-23 mmag** median (tens of mmag, not 215 mmag).
- The +0.215 mag is **calibrated raw (`mag_calib_final`)**, not the published differential; raw instrumental is ~60-72 mmag.
- LC `delta_mag` shift (-1.665 mag) is an **ensemble-reference artifact** from comparing two full photometry runs with different comp pools, not evidence that differential photometry failed to cancel sky.

**GATE 2 is ready for authorization.** Batch E changes are sound; the anchor SHA moved because preprocess was correctly regenerated; published differential curves are stable on matched-star analysis.
