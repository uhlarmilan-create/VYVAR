CURSOR RESULT — 2026-06-22 (G5-F003 fix)

What I did
Diagnosed `_generate_candidate_lc_png` column selection vs export/main plots; implemented calibrated-mag precedence mirroring `_select_export_lc_rows`; added tests; updated ledger. Isolated commit (no push).

## Diagnosis

### `_generate_candidate_lc_png` (before fix)

```3306:3320:photometry_report.py
            mag_col = next((c for c in df.columns if "mag" in str(c).lower()), None)
            ...
            ax.set_ylabel("mag_inst")
```

- Magnitude: **first column whose name contains `"mag"`** → typically `mag_inst` (column order in LC CSV).
- Y-label: hardcoded **`mag_inst`** regardless of plotted column.
- Y-axis inverted (brighter up) — already matched main plots; kept.

### Export + main per-star plots (reference)

| Path | Magnitude precedence |
|------|----------------------|
| **Export** (`_select_export_lc_rows`) | `mag_calib`; when `ac_ok` + finite `mag_calib_ac` → AC values (export column conceptually `mag_calib`) |
| **Main LC plots** (`_plot_lightcurve_to_jpeg`, `_load_lc_xy_from_csv`) | `mag_calib_ct` if present, else `mag_calib`; filter `flag=normal`; y-label = column name |

Task scope: candidate figures mirror **export** AC precedence (`mag_calib_ac` when AC on, else `mag_calib`), not `mag_calib_ct` (main plots use CT column when present; export does not).

## Fix

- **`_resolve_candidate_lc_mag_for_plot(df)`** — export-mirror: `mag_calib_ac` when `ac_ok` & finite; else `mag_calib`; explicit `mag_inst` only when no calibrated columns.
- **`_generate_candidate_lc_png`** — uses helper; `flag=normal` filter; time cols `bjd_tdb`/`bjd`/`hjd`/`jd`; y-label = resolved column name; `invert_yaxis()` unchanged.

**Not touched:** main per-star plot path (`_plot_lightcurve_to_jpeg`, `_load_lc_xy_from_csv`).

## Tests (`tests/test_g5_f003_candidate_lc.py`)

| Case | Result |
|------|--------|
| `mag_calib` + `mag_inst` | Uses `mag_calib`, ylab `mag_calib` |
| AC on (`mag_calib_ac`, `ac_ok`) | Uses AC mags, ylab `mag_calib_ac` |
| Mixed AC | AC values where `ac_ok`, ylab `mag_calib` |
| No calib columns | Explicit `mag_inst` fallback |
| PNG integration | Figure ylab + scatter y-data verified |

**387 passed**, 15 skipped; ruff clean.

## PDF regression

Fix scoped strictly to `_generate_candidate_lc_png` (candidate detail pages only). Main per-star LC JPEG path, tables, headers, glossary unchanged — no code overlap. Full-draft PDF byte-identity not run (would require golden PDF fixture); collateral risk limited to candidate LC PNG cache files (`lc_*.png` in `_report_cache`).

## Ledger

| Finding | Status |
|---------|--------|
| **G5-F003** | **FIXED** — fix-log step 9 |

## Commit

`fix(report): candidate LC figures use calibrated mag, not mag_inst (G5-F003)` — see hash below.

**Not pushed** — stop for Claude review.

## Files changed

- `photometry_report.py`, `tests/test_g5_f003_candidate_lc.py`
- `docs/VYVAR_FULL_AUDIT_LEDGER.md`, `CURSOR_RESULT.md`
