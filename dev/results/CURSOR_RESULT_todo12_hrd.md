CURSOR RESULT - 2026-07-10

What I did
Implemented TODO-12-HRD: two-stage absolute extreme-object selection, online Gaia TAP / SIMBAD enrichment (`hrd_enrich.py`), session-aware HRD titles/captions/tables in PDF and UI, config keys, citations, tests, and real-data validation on draft_425 (B/V/R) + draft_424.

## Output / findings

### Part 0 (pre-change verification)
| Setup | masterstars rows | DAO-filtered HRD rows | reliable parallax |
|-------|------------------|----------------------|-------------------|
| draft_425 B_20_2 | 19251 | 14574 | 1015 |
| draft_425 V_20_2 | 8381 | 7847 | 795 |
| draft_425 R_20_2 | 17268 | 12474 | 1021 |
| draft_424 NoFilter_60_2 | 6699 | 6088 | 2474 |

- DAO filter uses `flux` column (not fallback branch). Reductions: B 14574/19251, V 7847/8381, R 12474/17268.
- Local Gaia SQLite (`vyvar_gaia_dr3.db`): `COUNT(*)=211712600`, `COUNT(teff_gspphot)=0`, `COUNT(logg_gspphot)=0`.

### Real-data validation (`tmp/todo12_hrd/`)
| Label | obs_group | HRD rows | candidates | notes |
|-------|-----------|----------|------------|-------|
| draft425_B | B_20_2 | 14574 | 19 | 1 WD + 9 binary + 9 very cool |
| draft425_V | V_20_2 | 7847 | 20 | +1 binary vs B (detection selection) |
| draft425_R | R_20_2 | 12474 | 19 | similar mix to B |
| draft424 | NoFilter_60_2 | 6088 | 5 | mostly very cool + 1 binary |
| draft425_B_offline | B_20_2 | 14574 | 19 | enrich off; teff/logg show N/A |

Evidence PNGs: `tmp/todo12_hrd/draft425_B_hrd.png`, `_V_`, `_R_`, `draft424_hrd.png`, `draft425_B_offline_hrd.png`.
Summary JSON: `tmp/todo12_hrd/summary.json`.

### DoD checks
- pytest: **691 passed**, 15 skipped (+10 new HRD tests).
- `session_baseline_check.py --fast`: **OVERALL PASS**.
- PDF overflow (draft_425 B_20_2): **0 violations** (389 pages).
- Offline run: enrich flags off; PDF path validated via validation script (no errors, N/A teff/logg).

## Errors (if any)
None blocking. Gaia TAP returns masked null teff for some candidates (expected at color extremes per Andrae et al. 2023); handled as N/A.

## Files changed
- `hrd_enrich.py` (new)
- `hrd_analysis.py` - Stage 1/2 selection, classification, plot title, caption constant, category colors
- `ui_hrd.py` - cfg/cache wiring, table columns, empty-field info, caption
- `photometry_report.py` - HRD page table/caption/enrichment wiring
- `config.py` - `hrd_online_enrich_enabled`, `hrd_simbad_enrich_enabled`, `hrd_enrich_max_candidates`
- `citations.py`, `CITATIONS.bib` - Pecaut & Mamajek (2013), Andrae et al. (2023)
- `docs/config_schema.md`, `docs/VYVAR_PARAMS.md`, `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_STATE.md`, `docs/VYVAR_JOURNAL.md`
- `tests/test_hrd_extreme.py` (new)
- `scripts/todo12_hrd_validate.py` (validation helper)

Base commit (pre-change): `f4c7c83ce36e1d60f58170ecaf88c02fb1b4a425` - changes not committed (awaiting Milan).
