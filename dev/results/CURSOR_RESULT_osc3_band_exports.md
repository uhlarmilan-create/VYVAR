CURSOR RESULT - 2026-07-22 22:15 UTC+2

What I did
Implemented OSC-3 (phase 3/3): TR/TG/TB AAVSO/VarAstro band mapping, Gaia G+BP-RP ->
Johnson/Cousins comp mags via gaia_johnson.py, OSC-03 wired export gate, report
multi-band summary, tests, docs, FLOW + parameter handbook PDF regen. **STOP before push.**

## Implementation summary

### New module `src_py/gaia_johnson.py`
- **Coefficient source:** Gaia DR3 documentation CU5 Table 5.9 (GBP-GRP -> G-V/G-B/G-R)
- **Scatter sigmas:** same Table 5.9 sigma column (0.03017 / 0.0633 / 0.03167 mag)
- **Validation reference:** Ruelas-Mayorga et al. 2025 RASTI 4:37 doi:10.1093/rasti/rzaf037
  (Crossref-verified; Landolt independent fit - NOT the coefficient source)

### Export path (`export_reports.py`)
- `resolve_aavso_filt_from_obs_group`: R/G/B -> TR/TG/TB; oneRGGB -> skip (E1)
- `_prepare_osc_comp_df_for_export`: Johnson comp mags for VarAstro + AAVSO notes
- OSC-03 pre-write gate via `check_osc03_export_eligibility`
- Methods matrix OSC rows (channel extraction, binning, transform citation)
- `_guess_setup_info_from_obs_group` strips OSC channel suffix before parse

### Other wiring
- `osc_align.py`: `is_onerggb_internal_obs_group`, `osc_multiband_summary_rows`
- `invariants_runtime.py`: OSC-03 [wired]
- `band_classify.guess_aavso_code_from_obs_group` -> OSC-aware resolver
- `citations.py`: RunCitationContext OSC fields + matrix rows
- `photometry_report.py`: OSC multi-band summary PDF page

## Transformation set + citation

| Item | Value |
|------|-------|
| Coefficients | Gaia DR3 CU5 Table 5.9 (Johnson-Cousins, X=GBP-GRP) |
| Sigmas | Table 5.9 sigma column: V 0.03017, B 0.0633, R 0.03167 mag |
| Validation | Ruelas-Mayorga et al. 2025 RASTI doi:10.1093/rasti/rzaf037 |
| Test fallback | Riello et al. 2021 EDR3 Table C.2-style (`RIELLO_EDR3_FALLBACK`) |

## Test evidence

```
dev/tests/test_osc3_band_exports.py: 17 passed
  - resolve_aavso_filt_from_obs_group (R/G/B/oneRGGB/mono)
  - k2 TR/TG/TB none-token path
  - gaia_johnson transform grid + out-of-validity exclusion
  - comp Johnson plumbing + OSC-03 gates
  - methods matrix OSC rows
  - oneRGGB export skip + TG in AAVSO file

Full dev/tests: 1117 passed, 24 skipped
ruff: clean on touched files
--fast: OVERALL PASS
--full: running (see Gates)
```

## Anchor enumeration (shared export/report functions touched)

| Function | OSC guard | Mono path |
|----------|-----------|-----------|
| `export_lightcurve_reports` | oneRGGB early return; TR/TG/TB FILT; Johnson comp prep | Unchanged when obs_group has no channel suffix |
| `export_all_method_lightcurve_reports` | (calls above) | Unchanged |
| `_format_varastro_comp_table` | optional `use_johnson_mag` + validity filter | Default Gaia G mag column |
| `_guess_setup_info_from_obs_group` | Strips `_R/_G/_B/_oneRGGB` before parse | Same parse on base name |
| `resolve_aavso_filt_from_obs_group` | TR/TG/TB / oneRGGB block | `_resolve_aavso_filter` on setup token |
| `_vyvar_export_citation_lines` | OSC ctx via obs_group | Unchanged |
| `build_run_citation_context` | Optional obs_group OSC flags | Defaults false |
| `build_methods_matrix_lines` | +3 OSC rows when export eligible | No extra rows |
| `guess_aavso_code_from_obs_group` | Delegates to OSC resolver | Mono tokens unchanged |
| `_PhotometryReportBuilder.build_pdf` | +OSC multiband summary page | Skipped when not OSC folder |
| `check_osc03_export_eligibility` | FAIL on oneRGGB / wrong FILT | N/A pass |

## Gates

### pytest + ruff
- `1117 passed, 24 skipped`
- `ruff check` on touched files - clean

### `--fast`
```
OVERALL: PASS (1117 passed, 24 skipped)
```

### `--full` draft_435 (mono anchor)
```
full-snapshot-sha-core       PASS   03d8fb6491bc3c22... n=333
full-photometry-sha-core     PASS   03d8fb6491bc3c22... n=333
full-photometry-sha-extended PASS   bbfcc92e7ac5c4c5... n=499
full-science-compare         PASS   n_lc=166 failures=0
OVERALL: PASS (~2549s)
```
Mono anchor byte-identical preserved.

## Push (2026-07-22, Milan authorized)

### Pre-push checks

| Check | Result |
|-------|--------|
| `git fetch origin`; `origin/main` before push | `ad35345` |
| Stack `git log origin/main..HEAD --oneline` | 2 commits - exact match (see below) |
| Citation provenance | Coefficients: Gaia DR3 CU5 Table 5.9; sigmas: same table sigma column; validation: Ruelas-Mayorga et al. 2025 RASTI doi:10.1093/rasti/rzaf037 (Crossref-verified) |
| `git status --short` | Clean; allowlisted untracked only |
| `session_baseline_check.py --fast` (final HEAD `44b22f6`) | **OVERALL PASS** - 1117 passed, 24 skipped |
| `--full` draft_435 (pre-push) | **OVERALL PASS** - byte-identical core+extended |

### Commit inventory (`origin/main..HEAD`, newest first)

```
44b22f6 docs(osc): OSC-3 invariants, decisions, FLOW/handbook PDFs, gate record
859e776 feat(osc): TR/TG/TB exports, Gaia Table 5.9 Johnson comps, OSC-03 (phase 3)
```

Base: `ad35345` -> stack tip: `44b22f6` (`git push origin main` succeeded).

### Final origin/main tip

Local HEAD matches `origin/main` at `44b22f6` after push. For current tip: `git rev-parse origin/main`.
