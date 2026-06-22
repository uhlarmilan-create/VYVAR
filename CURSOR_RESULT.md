CURSOR RESULT — 2026-06-22 (G5-F011 Path A: mag_calib_final)

What I did
Implemented canonical `mag_calib_final` = `mag_calib` + CT + AC (per-target/night constants) in LC CSV; routed export + all publication LC figures to it; left `lc_rms`/trust on `mag_calib`. Tests + ledger. Isolated commit (no push).

## Canonical column

**Name:** `mag_calib_final`

**Formula** (`compute_mag_calib_final`, `save_lightcurve_csv` after SG-aligned `mag_calib` + `mag_calib_ac`):

`mag_calib_final` = `mag_calib` + (`ct_correction` if `ct_ok` else 0) + (`delta_m_corr` if `ac_ok` else 0)

When **CT off** and AC on: copies `mag_calib_ac` array → **byte-identical** to pre-fix export MAG (rounded 6 dp).

Provenance columns retained: `mag_calib`, `mag_calib_raw`, `mag_calib_ct`, `mag_calib_ac`.

## Consumers updated

| Consumer | Now uses |
|----------|----------|
| AAVSO export (`_select_export_lc_rows`) | `mag_calib_final` → copied to export `mag_calib` column |
| VarAstro body | same (via `_select_export_lc_rows`) |
| Main per-star PDF (`_plot_lightcurve_to_jpeg`, `_load_lc_xy_from_csv`) | `mag_calib_final` via `_publication_lc_mag_column` |
| LC overlay PDF | same |
| Candidate figures | `mag_calib_final` via `_resolve_candidate_lc_mag_for_plot` |

**`delta_mag`:** left as ensemble differential (vs AIJ flux sum) — **not** CT/AC adjusted (distinct quantity).

**Unchanged:** `lc_rms`, `lc_rms_ooe`, comp_qa, trust → `mag_calib` (CT/AC are constants → scatter invariant; test added).

**Legacy CSVs** without `mag_calib_final`: export falls back to AC precedence (`mag_calib_ac`).

## Tests

| Case | Result |
|------|--------|
| Neither gate | `mag_calib_final` == `mag_calib` |
| AC only | == `mag_calib_ac` |
| CT only / both | additive |
| CSV rounding CT off | `mag_calib_final` == `mag_calib_ac` exactly |
| Export rows | use `mag_calib_final` |
| `lc_rms` invariance | PASS |
| CT-on synthetic export | `mag_calib_final` flows to rows |
| G5-F003 / G5-F007 suites | PASS |

**12** new tests in `tests/test_mag_calib_final_g5_f011.py`; updated `test_g5_f003_candidate_lc.py`.

## Re-validation (draft 419/420)

No local draft LC files found under `c:\ASTRO`. **Do-no-harm gate** covered by:

- `test_save_lc_csv_rounding_byte_identity_ct_off` — rounded CSV `mag_calib_final` == `mag_calib_ac`
- `test_export_byte_identical_legacy_ac_path_when_no_final_column` — legacy path unchanged

With **tracked config** (CT off, AC on): export AAVSO MAG **byte-identical** to pre-fix AC export.

**Intended main-plot change:** main PDF y-data shifts by **`ac_correction`** (`delta_m_corr`, typically mmag–tens of mmag) vs pre-fix (`mag_calib_ct` ≈ `mag_calib` without AC). Candidate + export already matched AC; now all three match `mag_calib_final`.

**CT-on:** synthetic `test_export_ct_on_flows_to_rows` confirms CT+AC in export MAG.

## Ledger

- **G5-F011** added **FIXED** (fix-log step 10)
- **G5-F003** note: superseded by canonical column

## Commit

`fix(calib): canonical mag_calib_final (CT+AC) used by export and all figures (G5-F011)` — see hash below.

**Not pushed** — stop for Claude review. **Next:** documentation pass (column lineage + consumer table for manual).

## Files changed

- `photometry_core.py`, `export_reports.py`, `photometry_report.py`
- `tests/test_mag_calib_final_g5_f011.py`, `tests/test_g5_f003_candidate_lc.py`, `tests/photometry_sha.py`
- `docs/VYVAR_FULL_AUDIT_LEDGER.md`, `CURSOR_RESULT.md`
