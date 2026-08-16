# CURSOR RESULT - IMPL-05 Item A (draft 515 dedupe crash)

Date: 2026-08-16
Baseline: 20ced6a (COMP-ASSIGN-02, stamp 977d9f5)
Tip: b2ae3b7
Push: NO

## What I did

Fixed `_proc_deduplicate_matched_catalog_rows` so the score column is always a
`pd.Series` before `.fillna`, including the flux fallback when
`peak_max_adu` / `dao_flux` are absent.

## Why the fallback fired on draft 515 (not on 514)

Draft 514 per-frame proc CSVs carry `peak_max_adu` from the saturation block
written with every non-empty DAO table - the `peak_max_adu` branch always ran.

Draft 515 crashed mid `export_per_frame_catalogs` (infolog `05:57:17`) with
no proc CSVs written. The traceback is the flux fallback. That branch runs only
when **neither** `peak_max_adu` nor `dao_flux` is present. That happens when
`detect_stars_match_master_reference` returns an **empty DAO table**
(`tbl is None or len(tbl)==0` -> empty DataFrame) and
`inject_forced_masterstar_rows` then supplies catalog rows that, on the
geometry-ok path, omit `peak_max_adu`, `dao_flux`, and `flux`. With `flux`
absent, `out.get("flux")` is `None`, `pd.to_numeric(None)` is
`numpy.float64`, and `.fillna` raises.

So the missing column is not a silent schema drift between drafts: on the
crashing 515 frame(s) the peak column never existed because DAO found no
stars; forced-phot stubs filled catalog_ids without the saturation/flux
columns 514's detections always had. Separate finding: empty-DAO frames still
reach dedupe with forced-only rows - the dtype crash was the blocker; empty-DAO
rate on 515 remains a field/QC question.

## Tests

- `test_proc_deduplicate_fallback_without_peak_or_dao_flux` (one-row, no
  peak/dao_flux/flux)
- existing brightest-peak test unchanged

## Files

- `src_py/pipeline.py`
- `dev/tests/test_proc_catalog_dedupe.py`
- this result
