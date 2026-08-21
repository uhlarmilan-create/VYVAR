CURSOR RESULT - 2026-08-21 (ERR-518-01)

What I did
Investigation-only measurement of INV-ERR-MODE-01 on draft 518 (Newton /
pre-calibrated TOI-1131). No `src_py/` changes. Sandbox:
`dev/sandbox/err_518_01_labbe_measure.py`;
`dev/results/context/session_20260821_err518/labbe_measurements.json`.

## Named mechanism (one paragraph)

**Branch D (code-path bug, not pre-registered A/B/C as stated):** Labbe
empty-aperture measurement **succeeds** on draft 518 frames (sandbox:
`sigma_bkg_ap=71.8`, `n_valid=64`, reason empty on aligned FITS), but
`enhance_catalog_dataframe_aperture_bpm` in **global_fixed** aperture mode
(`snr_aperture_table=None`; draft 518 has only
`aperture_snr_table_REJECTED.json`, no `aperture_snr_table.json`) stores
Labbe results in `_sigma_by_r` under the **unrounded** key `float(r_ap)`
(e.g. `4.35461`) while row assignment looks up `round(float(r_ap), 4)`
(e.g. `4.3546`) - keys do not match (`dict.get` returns miss). Every row
is written `err_bkg_source=howell_fallback`, `sigma_bkg_ap=NaN` despite
successful Labbe. **Secondary Branch B:** RAM handoff calls
`export_per_frame_catalogs(..., defer_disk_writes=True)` (`pipeline.py:15797`);
`finalize_hybrid_bkg_fallback_proc_dir` is gated on `not defer_disk_writes`
(`pipeline.py:11657-11660`) and is **not** re-invoked after the batch CSV
flush (`pipeline.py:15815-15824`), so even a partial empirical population
would not be howell_scaled on this path. With 100% raw fallback,
`r_setup=NaN` and finalize is a no-op anyway (measurement:
`n_ratio_samples=0`). INV-ERR-MODE-01 firing is **correct** guard behaviour.

**Rejected pre-registered branches:**
- **A (Labbe failed every frame):** REJECTED. Direct rerun of
  `measure_empty_aperture_sigma_bkg` on aligned FITS with proc star list
  succeeds (71.8 px, 64 valid). No `err_bkg empirical fallback` infolog
  lines because Labbe did **not** fail (log only fires on measurement
  failure, `photometry_core.py:14291-14300`).
- **C (finalize NaN inputs):** REJECTED as primary cause. `_sky_pp_for_photometric_error`
  returns finite `33488.88` ADU/px from `noise_floor_adu` /
  `sky_adu_per_px_annulus` (no `sky_surface_bg_median_adu` column in proc
  CSV). Simulated `scaled_sigma_bkg_ap_from_howell` with `r_setup=1.0`
  yields finite sigma - finalize would work if empirical ratios existed.

## 4.1 Logs

### 4.1.1 `err_bkg empirical fallback`
**Absent** (0 lines) in `Archive/Drafts/draft_000518/infolog_20260821_065225.txt`.
Consistent with Labbe measurement succeeding inside workers (failure log
not emitted).

### 4.1.2 `err_bkg howell_scaled setup`
**Absent** (0 lines). Finalize did not log (no empirical rows -> early return
`photometry_core.py:1184-1185`; and finalize skipped on defer path).

### 4.1.3 `hybrid bkg fallback finalize skipped`
**Absent** (0 lines). No swallowed exception; finalize simply did not run.

### 4.1.4 Preflight log (`logs/run_preflight_error_20260821_050500.log`)
Full content beyond invariant:
- Step: Phase 0+1 + Phase 2A
- DB: EQUIPMENTS ids 1-5, config `observer_location_id=1` exists
- Traceback: `read_flux_from_csv` -> `_phase2a_empirical_sigma_bkg_ap`
  (`photometry_core.py:1388`)

Run path: UI **RUN VYVAR (non-cal)** per infolog (pre-calibrated draft).

## 4.2 Proc CSVs

### 4.2.5 `err_bkg_source` counts (71 proc files, 62352 rows)

| source | rows |
|--------|-----:|
| howell_fallback | 62352 |
| empirical | 0 |
| howell_scaled | 0 |

Per-file pattern identical (878 rows/file x 71 files).

### 4.2.6 Target row catalog_id=1624628764771224960 (+ 2 peers)

| catalog_id | sigma_bkg_ap | aperture_r_px | sky_adu_per_px_annulus | noise_floor_adu | err_bkg_source |
|------------|---------------:|--------------:|-----------------------:|------------------:|----------------|
| 1624628764771224960 | NaN | 4.35461 | 33488.8828125 | 33488.8828125 | howell_fallback |
| 1497613731286514432 | NaN | 4.35461 | 33488.8828125 | 33488.8828125 | howell_fallback |
| 1500803208360486144 | NaN | 4.35461 | 33488.8828125 | 33488.8828125 | howell_fallback |

Columns **absent** from proc CSV: `aperture_area_px`, `sky_surface_bg_median_adu`
(I-11 pre-subtraction sky not stamped on this export).

### 4.2.7 Unique `aperture_r_px`
Single value **4.35461** px (global_fixed, `aperture_factor_applied=global_1.900x`,
`fwhm_px_for_aperture=2.2919`, `snr_aperture_mode=global_fixed`). All rows
fallback - not radius-selective.

## 4.3 Execution-path facts (Branch B discriminators)

| Setting | Value | Evidence |
|---------|-------|----------|
| `err_background_mode` | `empirical` | `config.json:437`; infolog non-cal run |
| `write_sidecar_csv_next_to_fits` | **true** (implicit default) | Sidecar proc CSVs written under `detrended_aligned/lights/V_60_2/` |
| `defer_disk_writes` | **true** during catalog export | `pipeline.py:15797`; infolog `RAM handoff: ano` (`infolog:2475`) |
| finalize at export | **skipped** | Gate `not defer_disk_writes` false at `pipeline.py:11660`; no second call after flush |
| `gaussian_fwhm_px_override` | **2.2919 px** | infolog line 2400 |

**Provenance gap:** runtime trio not stamped into `pipeline_meta.json` for
draft 518 (unrecorded in artifact; inferred from code + infolog only).

## 4.4 Frames (physical cause)

### FITS summary (first frame, sandbox measurement)

| | aligned | non_calibrated (ingest) |
|---|---------|-------------------------|
| shape | 2088 x 3126 | same |
| dtype | >f4 | >f4 |
| NaN fraction | 0.0 | 0.0 |
| median ADU | 33487.4 | 33487.3 |
| min / max ADU | 33416 / 92818 | 33416 / 98232 |
| `VYVARPR` / sky surface | applied | applied |
| FWHM DAO median | NaN (flat field) | NaN |

External pre-calibration leaves a **uniform ~33.5k ADU pedestal**; annulus
sky estimates match pedestal (`noise_floor_adu` ~33489). Not an edge-NaN /
crowding failure on disk.

### Decisive Labbe rerun (`measure_empty_aperture_sigma_bkg`)

Setup: `r_ap=4.35461`, `r_in=10.887`, `r_out=20.627` px (annulus 4.75/9.0 x
FWHM 2.2919); 878 proc stars; seed=42.

| image | sigma_bkg_ap | n_valid | reason |
|-------|-------------:|--------:|--------|
| aligned | 71.80 | 64 | (empty) |
| non_calibrated | 99.93 | 64 | (empty) |

### Pre-calibrated path facts
- `draft_manifest.json`: `calibration_mode: "pre_calibrated"`, `is_calibrated: 0`,
  lights under `non_calibrated/lights/` (VYVAR convention for ingest-only lights).
- VYVAR **did** run preprocess QC in-place on those lights (`infolog:108-112`,
  `VYVARPR=true`, sky-surface order 2 on headers).
- Alignment + per-frame catalog ran on RAM handoff buffers then flushed to disk
  (`infolog:2474-2475`).

### Key mismatch proof (code, not measurement)

```python
r_ap = max(0.5, 1.9 * 2.2919)  # 4.35461
{float(r_ap): 'stored'}.get(round(float(r_ap), 4))  # -> None (MISS)
```

Storage: `_sigma_by_r[float(_r_u)]` with `_r_u` from unrounded `r_ap`
(`photometry_core.py:14258-14290`). Lookup (global mode): `round(float(r_ap), 4)`
(`photometry_core.py:14315-14317`). SNR-table path uses rounded keys throughout
- explains draft 516 (`snr_aperture_mode=snr_table`, 100% empirical on frame 001).

## Fix options (no recommendation)

1. **Unify `_sigma_by_r` lookup key** (always `round(r, 4)` on store and fetch
   in global_fixed branch). Consequence: empirical sigma propagates; finalize
   may still be skipped on RAM path unless option 2 also applied.

2. **Call `finalize_hybrid_bkg_fallback_proc_dir` after RAM batch CSV flush**
   (`pipeline.py` post-15824). Consequence: raw fallbacks become howell_scaled
   when any empirical rows exist; does not help 518 until option 1 fixed.

3. **Block empirical mode without accepted `aperture_snr_table.json`** (fail loud
   at plan time). Consequence: forces SNR-table path where keys already round;
   Newton drafts need SNR gate to pass first.

4. **PRECAL-INPUT-CONTRACT-01**: document/stamp pre-subtraction sky (`sky_surface_bg_median_adu`)
   and pedestal semantics for externally calibrated lights. Consequence: better
   Howell fallback scaling and I-11 compliance; does not alone fix NaN sigma.

5. **Temporary ops workaround:** regenerate accepted `aperture_snr_table.json`
   (not REJECTED-only) so export uses snr_table path. Consequence: immediate
   empirical rows on re-export; depends on SNR gate passing on high-pedestal data.

## Docs impact (DOCS-SYNC)

| Item | Action |
|------|--------|
| `dev/results/CURSOR_RESULT_ERR_518_01.md` | This report |
| ROADMAP | Suggest **PRECAL-INPUT-CONTRACT-01** (externally pre-calibrated frame contract: pedestal, sky stamp, SNR-table requirement for empirical err mode) - not written here |
| `docs/VYVAR_ROADMAP.md` | Not edited in this task (investigation STOP) |

## Errors (if any)
None (investigation only).

## Files changed
- `dev/results/CURSOR_RESULT_ERR_518_01.md` (this file)
- `dev/sandbox/err_518_01_labbe_measure.py`
- `dev/results/context/session_20260821_err518/labbe_measurements.json`

STOP - Milan reviews; code fix is a separate task after decision.
