# VYVAR Bug Hunt Report — 2026-05-27

Systematic static review of scoped VYVAR Python modules (no code changes). Patterns from task categories 1–5; line numbers refer to workspace state at review time.

## Summary

| Category | HIGH | MEDIUM | LOW | Total |
|----------|------|--------|-----|-------|
| 1 Type errors | 0 | 2 | 1 | 3 |
| 2 State deps | 1 | 4 | 5 | 10 |
| 3 Off-by-one / bounds | 2 | 5 | 3 | 10 |
| 4 Astro edge | 0 | 4 | 4 | 8 |
| 5 Resources | 0 | 4 | 2 | 6 |
| **TOTAL** | **3** | **19** | **15** | **37** |

**Category 1a (NumPy ← pandas mask):** No remaining `np_array[pd.Series]` indexing found in scoped modules. `variability_detector.py:664` uses `.to_numpy(dtype=bool)` (post-fix 874a36d).

---

## Category 1 — Type errors

### BUG-001 [MEDIUM] photometry_core.py:1824–1827
**Pattern:** WCS used without `has_celestial` guard before `all_world2pix`  
**Code:**
```python
wcs = WCS(hdr)
xy = wcs.all_world2pix(np.array([[float(ra_deg), float(dec_deg)]], dtype=np.float64), 0)
```
**Risk:** Invalid or plate-solve-incomplete headers may raise or return non-finite pixels; outer `except` masks target as `no_data`.  
**Fix:** Guard with `fits_header_has_celestial(hdr)` (or `w.has_celestial`) before constructing transforms.

### BUG-002 [MEDIUM] photometry_core.py:9038–9068
**Pattern:** Same — `WCS(hdul[0].header)` then `all_world2pix` inside broad `try`  
**Code:**
```python
wcs_m = WCS(hdul[0].header)
# ...
xy_part = wcs_m.all_world2pix(pts, 0)
```
**Risk:** CATALOG_ONLY VT placement silently skipped on bad WCS; harder to diagnose than explicit guard.  
**Fix:** Check `has_celestial` before transform; log distinct reason when WCS missing.

### BUG-003 [LOW] photometry_core.py:2055–2056
**Pattern:** Division by `weights.sum()` without zero check (PyTICS loop)  
**Code:**
```python
weights = 1.0 / (rms_arr**2)
weights /= weights.sum()
```
**Risk:** If all weights are NaN/0 (pathological `rms_arr`), `ZeroDivisionError`.  
**Fix:** `s = weights.sum(); if s <= 0 or not np.isfinite(s): break` before normalize.

---

## Category 2 — State dependencies and missing guards

### BUG-004 [HIGH] ui_variability.py:932
**Pattern:** Streamlit `st.session_state["tess_results"]` without prior `setdefault` in dialog path  
**Code:**
```python
tess_store: dict[str, TessResult] = st.session_state["tess_results"]
```
**Risk:** `KeyError` if user opens catalog crossmatch dialog (TESS tab) before `render_variability_dashboard()` runs (`setdefault` only at line 1213).  
**Fix:** `tess_store = st.session_state.setdefault("tess_results", {})` at start of `_variability_crossmatch_dialog_body()`.

### BUG-005 [MEDIUM] photometry_core.py:5177–5183
**Pattern:** `pd.read_csv` on `active_targets_csv` / `comparison_stars_csv` with no `Path.is_file()` check  
**Code:**
```python
at_df = pd.read_csv(active_targets_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
comp_df = pd.read_csv(comparison_stars_csv, ...)
```
**Risk:** `FileNotFoundError` aborts full Phase 2A if UI/pipeline passes stale paths.  
**Fix:** Validate paths at `run_full_photometry_pipeline` entry; return structured error.

### BUG-006 [MEDIUM] ui_aperture_photometry.py:37–38
**Pattern:** Cached CSV reader has no existence guard  
**Code:**
```python
def _cached_read_csv(path_s: str) -> pd.DataFrame:
    return pd.read_csv(path_s, low_memory=False, dtype=_GAIA_ID_DTYPE)
```
**Risk:** Stale cache key or race → `FileNotFoundError` on rerun.  
**Fix:** Check `Path(path_s).is_file()` inside cached function or at call sites.

### BUG-007 [MEDIUM] photometry_core.py:1456–1462
**Pattern:** `pd.read_csv(frame_csv_path)` when `csv_df is None` — no existence check  
**Code:**
```python
csv_df = pd.read_csv(frame_csv_path, low_memory=False, dtype=_GAIA_ID_DTYPE)
```
**Risk:** Missing per-frame proc CSV returns empty via `except`; caller may not distinguish missing vs corrupt.  
**Fix:** `if not Path(frame_csv_path).is_file(): return pd.DataFrame()` before read.

### BUG-008 [LOW] vyvar_platesolver.py:1808, 2502, 2565, 2618, 2885
**Pattern:** `getattr(AppConfig(), ...)` / fresh `AppConfig()` inside solver (not injected `cfg`)  
**Code:**
```python
tri_cap = float(getattr(AppConfig(), "platesolve_triangle_mag_cap", tri_cap_default))
```
**Risk:** Ignores draft-specific or UI-overridden config if added later; extra disk reads of `config.json`.  
**Fix:** Thread `app_config` parameter from pipeline/UI (same pattern as `pipeline.py` `app_config or AppConfig()`).

### BUG-009 [LOW] vyvar_blind_solver.py:119
**Pattern:** `getattr(AppConfig(), "debug_platesolver", False)`  
**Fix:** Pass `cfg` from caller.

### BUG-010 [LOW] ui_aperture_photometry.py:1390
**Pattern:** `getattr(AppConfig(), "vsx_local_db_path", "")` in autorun path  
**Fix:** Use `cfg` already available in render scope.

### BUG-011 [LOW] photometry_core.py:4623, 5161, 8836, 9826, 10370, 10871, 11316
**Pattern:** `AppConfig()` instantiated inside hot photometry paths  
**Risk:** Repeated JSON load; inconsistent with injected config in tests.  
**Fix:** Require `cfg` argument; default only at public API boundary.

### BUG-012 [LOW] pipeline.py:9385
**Pattern:** `extract_fits_metadata(fp, app_config=AppConfig())` inside per-file loop  
**Risk:** Performance + config drift in long comp-RMS scans.  
**Fix:** Hoist `cfg_for_workers = app_config or AppConfig()` once outside loop.

### BUG-013 [LOW] utils.py:325, 332
**Pattern:** `header["VY_FWHM"]` / `header["VY_FWHM_GAUSS"]` after `in header` check — OK; document as safe pattern.

---

## Category 3 — Off-by-one and boundary errors

### BUG-014 [HIGH] photometry_core.py:3644–3645
**Pattern:** `np.percentile` on possibly empty finite slice  
**Code:**
```python
vmin = float(np.percentile(data[np.isfinite(data)], percentile_lo))
vmax = float(np.percentile(data[np.isfinite(data)], percentile_hi))
```
**Risk:** All-NaN / constant blank MASTERSTAR → `ValueError: zero-size array` in `save_field_map_png`.  
**Fix:** `finite = data[np.isfinite(data)]; if finite.size == 0: return` or use nan-safe defaults.

### BUG-015 [HIGH] photometry_core.py:3588–3590
**Pattern:** `np.percentile(cutout, …)` when cutout can be empty  
**Code:**
```python
cutout = data[y0:y1, x0:x1]
vmin = float(np.percentile(cutout, 5))
```
**Risk:** Degenerate crop (`x1<=x0`, tiny image, bad coordinates) crashes cutout PNG export.  
**Fix:** `if cutout.size == 0: return` before percentiles.

### BUG-016 [MEDIUM] photometry_core.py:3748–3749
**Pattern:** Duplicate empty-finite percentile issue in `save_target_field_map_png`  
**Fix:** Same guard as BUG-014.

### BUG-017 [MEDIUM] variability_detector.py:679–681
**Pattern:** Division `n_frames_used_clean / n_frames_used` when denominator can be 0  
**Code:**
```python
ratio = pd.to_numeric(out["n_frames_used_clean"], errors="coerce") / pd.to_numeric(out["n_frames_used"], errors="coerce")
ok_clip = ratio >= float(clip_ratio_min)
```
**Risk:** `inf`/`NaN` clip_ratio for stars with zero frames; filter behavior undefined.  
**Fix:** `den = ...; ratio = np.where(den > 0, num/den, np.nan)` and treat non-finite as fail clip.

### BUG-018 [MEDIUM] variability_detector.py:495
**Pattern:** `np.percentile(mag_arr, 95/5)` only when `flux_arr.size >= 10` — OK; stars with 3–9 frames get `amplitude=nan` (documented).

### BUG-019 [MEDIUM] ui_variability.py:1890
**Pattern:** `.iloc[0]` after filter without `empty` check  
**Code:**
```python
row = results_df[results_df["catalog_id"].astype(str).map(lambda x: str(x).strip()) == cid_key].iloc[0]
```
**Risk:** `IndexError` if normalization mismatch between `cid_key` and results table.  
**Fix:** `sub = ...; if sub.empty: continue` before `iloc[0]`.

### BUG-020 [MEDIUM] vyvar_blind_solver.py:170–187
**Pattern:** `dao_stars["x"].max()` / `["y"].max()` without empty-DataFrame guard  
**Code:**
```python
x_max = float(dao_stars["x"].max())
```
**Risk:** Empty input → NaN centers; downstream triangle matching wastes work (no crash). Missing `x`/`y` → `KeyError`.  
**Fix:** Early return if `dao_stars.empty` or required columns missing.

### BUG-021 [LOW] ui_variability.py:2031
**Pattern:** `.iloc[0]` for field map — wrapped in `try/except` (acceptable but hides data bugs).  
**Fix:** Explicit empty check + user-visible warning.

### BUG-022 [LOW] export_reports.py:433
**Pattern:** `return df.iloc[0]` as last resort for check star — only when `len(df) > n_comp_min` earlier; low risk.

### BUG-023 [LOW] photometry_core.py:1944–1949
**Pattern:** Rolling median loop `for i in range(half, len(arr) - half)` — guarded by `w < 3 or w > len(arr)`; safe.

---

## Category 4 — Astronomy-specific edge cases

### BUG-024 [MEDIUM] photometry_core.py:9039–9076
**Pattern:** WCS pixel conversion for CATALOG_ONLY without validating solution quality  
**Risk:** Wrong x,y for forced aperture on mis-solved MASTERSTAR; photometry on blank sky.  
**Fix:** Require `VY_PSOLV` / SIP keywords or RMS guard before using `wcs_m`.

### BUG-025 [MEDIUM] variability_detector.py:478
**Pattern:** RMS% uses `sig / mu` with `mu != 0` guard — good; `mu` near zero still explodes RMS%  
**Code:**
```python
rms_map[str(cid)] = (sig / mu) * 100.0 if (math.isfinite(mu) and mu != 0) else float("nan")
```
**Risk:** Low-flux comps → huge RMS%; propagates to envelope.  
**Fix:** Minimum `|mu|` floor or flux-based exclusion.

### BUG-026 [MEDIUM] photometry_core.py:2504–2506
**Pattern:** `ensemble_normalize` with zero selected comps returns all-NaN arrays (by design)  
**Code:**
```python
if not good_ids:
    return mag_calib, delta_mag, ensemble_scatter
```
**Risk:** Downstream may not check all-NaN before export.  
**Fix:** Document contract; assert/log at LC export when `good_ids` empty.

### BUG-027 [MEDIUM] comp_pool_rms.py:222
**Pattern:** `_rel = _raw_flux / _norm_med` — rows with `_norm_med` 0 filtered by `_rel_ok` — OK.

### BUG-028 [LOW] comp_selection_per_target.py:1776, 2326
**Pattern:** `1.0 / (rms_f**2)` guarded by `rms_f > 1e-6` — OK.

### BUG-029 [LOW] gaia_catalog_id.py:80–118
**Pattern:** `normalize_gaia_source_id` handles NaN, empty, scientific notation — robust; non-numeric strings returned as-is (callers must validate).

### BUG-030 [LOW] variability_detector.py:692
**Pattern:** `mean_flux_norm_clean > 0.001` — stars below threshold excluded; borderline values sensitive to units.

### BUG-031 [LOW] export_reports.py:738–742
**Pattern:** `comp_ids.append(v2[:18])` — truncates Gaia IDs in AAVSO notes (intentional length cap; verify 18 digits sufficient).

### BUG-032 [LOW] pipeline.py:8010–8021
**Pattern:** Airmass from `CRVAL1/2` or `OBJCTRA/DEC` — guarded; returns `nan` if missing — OK.

---

## Category 5 — Resource and performance risks

### BUG-033 [MEDIUM] photometry_core.py:7668–7676 (approx.)
**Pattern:** `pd.read_csv(sidecar)` inside loop over frame FITS paths without cross-invocation cache  
**Risk:** Re-reads same sidecar CSV many times in comp-pool / frame aggregation paths.  
**Fix:** Dict cache keyed by `sidecar` path (same pattern as `_phase2a_csv_cache`).

### BUG-034 [MEDIUM] ui_finalization.py:329–338
**Pattern:** `pd.concat([pd.read_csv(f) for f in _all_csv])` over all `*_catalog.csv` under archive  
**Risk:** Large drafts → high RAM and slow finalization FIELD_REGISTRY step.  
**Fix:** Streaming RMS via `groupby` per file or `read_csv(..., usecols=[...])`.

### BUG-035 [MEDIUM] ui_aperture_photometry.py:1607–1614
**Pattern:** Optional “Load all light curves into memory” loads every `lightcurve_*.csv` into `@st.cache_data`  
**Risk:** Session memory growth; Streamlit cache retains large DataFrames across reruns.  
**Fix:** Cap count, lazy-load per selected target, or disk-backed cache with TTL.

### BUG-036 [MEDIUM] photometry_report.py:1561–1576
**Pattern:** `pd.read_csv(lp, nrows=100000)` inside `for _, sr in self.summary_df.iterrows()`  
**Risk:** N_targets × disk I/O for PDF airmass summary.  
**Fix:** Read one representative LC or precomputed QC column only.

### BUG-037 [LOW] comp_selection_per_target.py:874–919
**Pattern:** Builds `csv_cache` on first call — good pattern when cache passed through; duplicate full reads if multiple code paths pass `csv_cache=None`.  
**Fix:** Centralize cache at Phase 1 entry.

### BUG-038 [LOW] vyvar_blind_solver.py:41
**Pattern:** `pickle.load` inside `with open` — file handle OK. Module-level `_CACHED_INDEX` unbounded if many index paths — edge case.

### BUG-039 [INFO] vyvar_platesolver.py / vyvar_blind_solver.py
**Pattern:** No `subprocess.run` — plate solve is in-process; category 5d (subprocess timeout) **not applicable** to these modules.

---

## Findings requiring immediate fix (HIGH risk)

| ID | File | Line | Issue |
|----|------|------|-------|
| BUG-004 | ui_variability.py | 932 | `KeyError` on `tess_results` in crossmatch dialog |
| BUG-014 | photometry_core.py | 3644 | `np.percentile` on empty finite data (field map PNG) |
| BUG-015 | photometry_core.py | 3588 | `np.percentile` on empty cutout (target PNG) |

---

## Modules reviewed (no HIGH findings, notes only)

| Module | Notes |
|--------|--------|
| `photometry_phase2a.py` | Re-export shim only |
| `comp_pool_rms.py` | Division guards present; detrend uses `safe_trend` |
| `config.py` | Defensive `.get()` on JSON load |
| `calibration.py` | FITS writes use `with`; header writes intentional |
| `importer.py` | Uses `with fits.open` |
| `database.py` | `iloc[0]` after `hits.empty` check |
| `ui_calibration.py` | DB paths guarded |
| `ui_settings.py` | Standard session writes |
| `ui_masterstar_qa.py` | CSV reads mostly behind `is_file()` |
| `ui_quality_dashboard.py` | TOP1 path after non-empty eligible filter |
| `psf_photometry.py` | Raises if masterstars CSV missing before read |
| `pipeline.py` | Extensive `with fits.open`; WCS helpers use try/except |
| `utils.py` | WCS helpers return `False`/`None` on failure |
| `vyvar_alignment_frame.py` | `ms_csv.is_file()` before read |
| `app.py` | `vyvar_footer_state` initialized via `vyvar_init_footer_state_if_missing()` before read |

---

## Recommended fix order (for tomorrow’s triage)

1. **BUG-004** — one-line `setdefault` in dialog (user-facing crash).  
2. **BUG-014 / BUG-015 / BUG-016** — percentile guards on PNG exporters.  
3. **BUG-017 / BUG-019** — variability clip ratio and VT export `iloc`.  
4. **BUG-001 / BUG-002 / BUG-024** — explicit WCS validity checks.  
5. **BUG-033–BUG-036** — performance caches (non-blocking).

---

*End of report — documentation only; no source files modified.*
