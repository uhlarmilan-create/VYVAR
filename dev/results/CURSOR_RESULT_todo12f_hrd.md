CURSOR RESULT ù 2026-07-10 15:30 UTC+2

What I did
Extended HRD row payload with catalog/astrophysics detail fields; added PDF follow-on
"Extreme objects -- details" page(s) and UI expander; stamped validate summary.json.

## Output / findings

### Row extensions (`_make_row`)
New fields: `dist_pc`, `parallax_mas`, `parallax_snr`, `sp_type_raw`, `otype_raw`, `dsc_wd_p`,
`teff_source`, `ra_dec_sex`, `simbad_main_id`. Compact overview table unchanged.

### PDF (draft_425 B)
- Pages: **390** (was 389); details section on **page 25**
- **overflow_violations: 0**
- Sample rendered blocks:

**V* RS Per** (`458407464445792384`):
```
RA/Dec (J2000): 02:22:24.3 +57:06:34.1 | G=6.56 | M_G=-5.50 | BP-RP=3.427 |
dist=2581.1 pc (plx 0.39 mas, SNR 8.61) | Teff=N/A K (n/a) | logg=N/A (simbad_lumclass) |
SpT=M3.5IabFe-1 | otype=s*r | DSC WD p=2e-12 | pix=(680, 985)
```

**LAWD 12** (`458558784733311232`):
```
RA/Dec (J2000): 02:17:34.2 +57:06:46.8 | G=13.64 | M_G=10.64 | BP-RP=-0.308 |
dist=39.7 pc (plx 25.16 mas, SNR 889.32) | Teff=N/A K (n/a) | logg=N/A (n/a) |
SpT=DA2.3 | otype=WD* | DSC WD p=1 | pix=(2495, 966)
```

### Validate hygiene
`tmp/todo12_hrd/summary.json` now includes:
- `generated_at_utc`: e.g. `2026-07-10T13:24:27+00:00`
- `git_head`: short hash at run time
- Pre-change snapshot: `tmp/todo12_hrd/pre12f/`

### Tests / CI
- pytest: **723 passed**, 15 skipped (+6 new in `test_hrd_details.py`)
- `session_baseline_check.py --fast`: PASS
- UI expander: no automated UI test (Streamlit tab untested; expander wired in `ui_hrd.py`)

## Errors (if any)
None.

## Files changed
- Code commit `51e5d64`: hrd_analysis.py, photometry_report.py, ui_hrd.py, scripts/todo12_hrd_validate.py, tests/test_hrd_details.py
- Docs commit `eb8a381`: docs + CURSOR_RESULT_todo12f_hrd.md
- Pushed to origin/main
