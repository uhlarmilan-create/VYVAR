CURSOR RESULT - 2026-07-10 12:22 UTC+2

What I did
Implemented TODO-12d-HRD: NSS category config flag (default off), hardened annotated field image with
MASTERSTAR pixel-scale guard + short labels/legend, PDF HRD page wiring, validation + alignment check.

## Output / findings

### Item A - NSS category off (default)
- New key `hrd_nss_category_enabled=False` in AppConfig, config_schema, VYVAR_PARAMS.
- Stage-1/2 exclude NSS net and Binary label when off; tests parametrize ON for NSS-specific cases.

### Table composition (draft_425 B, online enrich)
| | pre-12d (12c) | post-12d |
|---|---|---|
| rows | 10 | 7 |
| Binary (Gaia NSS) | 3 | 0 |
| categories | WD + 3 Binary + 3 Very cool + 3 RSG | WD + 3 Very cool + 3 RSG |

pre-12d snapshot: `tmp/todo12_hrd/pre12d/summary.json`
post-12d: `tmp/todo12_hrd/summary.json`

### Item B - Annotated field image
- Background: skips `field_map.png` for annotation (matplotlib `bbox_inches=tight` breaks uniform scale);
  renders `MASTERSTAR.fits` ? `hrd_field_from_fits.png` at 1:1 (3126-2088).
- Labels: WD, RSG, RG, HOT, HOT-LUM, COOL (+ NSS when flag ON); SIMBAD main_id under label; legend strip.
- PDF: annotated image right of HRD plot; caption "Extreme objects marked on the MASTERSTAR field".
- Cache: `Archive/Drafts/draft_000425/platesolve/B_20_2/photometry/_report_cache/hrd_field_annotated_report.png`

### Alignment check (brightest RSG, draft_425 B - catalog_id 458357741598022528)

| state | background | scale | peak/bg ratio | aligned |
|---|---|---|---|---|
| before scale guard + field_map | field_map.png 1492-963 | 0.477-0.461 | 1.90 | FAIL |
| after FITS 1:1 background | hrd_field_from_fits.png 3126-2088 | 1.0-1.0 | 6.375 | PASS |

Evidence crops: `tmp/todo12_hrd/draft425_B_rsg_align_crop.png`
Annotated PNGs: `tmp/todo12_hrd/draft425_{B,V,R}_field_annotated.png`

Also aligned: draft_425 V ratio 15.94, R ratio 8.50.

### PDF (draft_425 B_20_2)
- `VYVAR_report_B_20_2_overflow_verify.pdf`: 389 pages, **overflow_violations: 0**
- Spot pages: `Archive/Drafts/draft_000425/platesolve/B_20_2/_overflow_spot/page_*.png`

### Validation suite
- pytest: **698 passed**, 15 skipped
- `session_baseline_check.py --fast`: PASS
- `scripts/todo12_hrd_validate.py`: all setups OK; binary_rows=0 everywhere

## Errors (if any)
None - alignment FAIL on field_map-only path fixed by FITS 1:1 background (within pixel-scale scope).

## Files changed
- Code commit `d060323`: config.py, hrd_analysis.py, ui_hrd.py, photometry_report.py, scripts/todo12_hrd_validate.py, tests/test_hrd_extreme.py
- Docs commit `0e960a6`: docs/config_schema.md, docs/VYVAR_PARAMS.md, docs/VYVAR_STATE.md, docs/VYVAR_JOURNAL.md, docs/VYVAR_ROADMAP.md, CURSOR_RESULT_todo12d_hrd.md
- Pushed to origin/main
