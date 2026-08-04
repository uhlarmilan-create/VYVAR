CURSOR RESULT - AUDIT-T1 REMEDIATION (2026-07-30)

Base: `8c55020`. Groups A and B implemented; C and D not implemented (C gated; D measure-only).

---

## Group A - no numeric change

| Item | Change |
|------|--------|
| A1 | `_extract_airmass_from_header` docstring: Kasten & Young (1989), priority AIRMASS ? ALT ? AltAz |
| A2 | `compute_hjd_bjd` except: ERROR log + counter **`hjd_bjd_compute_fail`** |
| A3 | `kastenyoung1989` wired in citations CORE when `RunCitationContext.use_airmass` (set on export when LC has finite airmass) |
| A4 | `_howell_variance_adu2` docstring lists implemented vs omitted Howell terms |

### Acceptance A (byte-identity)

Headless regen on commit `06ed950` (`tmp/session_baseline/20260730T102630Z`, provenance `git_hash=06ed950...`) vs `draft_000435_snapshot_skysurface_20260716`:

| Gate | Result |
|------|--------|
| Photometry SHA core | **PASS** `b7f980c0...` n=325 (byte-identical) |
| Photometry SHA extended | **PASS** `2c43bbbf...` n=487 (byte-identical) |
| Science compare (162 LCs, incl. `err` in file hash) | **PASS** 0 science/time failures |
| `except_fix_counters` | `phase2a_empty_comp_drop=1` (expected for R CVn on 435) |

Structural note: Group A touches no photometry math, export row values, or success-path time conversion.

---

## Group B - export truth

### B1 policy (implemented)

**REFUSE** AAVSO (and VarAstro, same gate) when:
- `time_base == JD_FALLBACK`
- `time_base` column absent (unknown)
- mixed `time_base` within one LC

Log ERROR via `record_export_failure`; count `time_base_refused` in export batch summary. **No file written** that declares `#DATE=BJD`.

When `time_base == BJD_TDB` throughout: `#DATE=BJD` written (unchanged healthy path).

### B2 surfaces updated

| Surface | Behaviour |
|---------|-----------|
| `export_reports.py` | `#DATE=BJD` only after BJD_TDB guard passes |
| `ui_aperture_photometry.py` | axis title from `time_base` via `resolve_lc_time_base` |
| `photometry_report.py` | PDF LC x-axis label from `time_base` when column present |
| VarAstro | declares `# TIME SYSTEM: BJD(TDB)`; same export gate - refused when not BJD_TDB |

### B3 `err_scatter_unmatched`

Surfaced only (err unchanged): batch summary + run summary log `err_scatter_unmatched_epochs`; per-target INFO log.

### Acceptance B

Re-export vs anchor AAVSO (20 targets sampled, 13 with ref files): **13/13 byte-identical**, 0 diff.

Test: `dev/tests/test_export_time_base_refusal.py` - **4 passed** (JD_FALLBACK / absent / mixed ? no file, no `#DATE=BJD`).

---

## Group C - STOP (not implemented)

Gate not cleared. Findings for Milan:

**C1 legacy Howell:** anchor snapshot 162 LCs, **0** epochs with `err_method=howell`; production default is **empirical (Labbe)**. `(1+n_pix/n_B)` factor would need annulus pixel count `n_B` at `_howell_variance_adu2` call sites - available as `aperture_area_px` / annulus geometry in per-frame rows but not wired into variance helper today.

**C2 ensemble scatter:** anchor **0** epochs with `err_scatter_unmatched=True`. Options (choose before implement): (a) NaN err + exclude from export, or (b) flag + inflated err.

---

## Group D - measurements only

### D1 Scintillation / variance budget (sample targets, NoFilter_60_2)

| target_id | n_epochs | err_median_mag | chi2_reduced | scint_would_be_rel |
|-----------|----------|----------------|--------------|-------------------|
| 148554061257 | 139 | 0.0736 | 0.577 | 0.000826 |
| 148555232924 | 139 | 0.0490 | 1.265 | 0.000826 |
| 148557489929 | 139 | 0.0276 | 1.345 | 0.000826 |
| 148560953821 | 37 | 0.0965 | 1.578 | 0.000826 |
| 148591382805 | 139 | 0.0480 | 1.477 | 0.000826 |

chi2_reduced < 1 on several targets ? budget already over-quoted; adding scintillation would worsen. LC does not decompose photon vs ensemble vs sigma_sys terms.

### D2 Airmass provenance (30 frames, header AIRMASS vs AltAz K&Y)

| metric | value |
|--------|------:|
| n_frames | 30 |
| delta_median | 0.00052 |
| delta_p95 | 0.0007 |
| delta_max | 0.00073 |
| pct < 0.01 | 100% |

Current header priority order is fine for this rig (BO CVn Newton).

### D3 DATE-OBS convention (NoFilter_60_2 sample)

| rig | sample | DATE-OBS | DATE-END | EXPMID | EXPTIME |
|-----|--------|----------|----------|--------|---------|
| NoFilter_60_2 | BO_CVn_Light_001.fits | 2026-04-23T19:35:20.355 | - | - | 60.0 |

No DATE-END / EXPMID / MIDPOINT on sample; shutter-open vs mid-exposure not established from one frame - needs rig-specific sweep.

---

## Commits

| Group | Hash | Message |
|-------|------|---------|
| A | `e8afd8e` | fix(audit-t1): Group A docstrings, HJD/BJD error logging, Kasten-Young citation |
| B | `06ed950` | fix(audit-t1): Group B export time_base truth and err_scatter surfacing |

Pushed to `origin/main` after `git pull --rebase && git push`.

---

## Files changed

Group A: `src_py/pipeline.py`, `src_py/time_utils.py`, `src_py/except_fix_counters.py`, `src_py/citations.py`, `src_py/photometry_core.py` (docstring + time_base helpers)

Group B: `src_py/export_reports.py`, `src_py/ui_aperture_photometry.py`, `src_py/photometry_report.py`, `src_py/photometry_core.py` (export stats), `dev/tests/test_export_time_base_refusal.py`
