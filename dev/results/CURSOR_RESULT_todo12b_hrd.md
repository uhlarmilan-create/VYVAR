CURSOR RESULT - 2026-07-10 (TODO-12b-HRD)

What I did
Follow-up to TODO-12-HRD: lowered/config-driven parallax gate (0.15 mas + SNR 5), per-category table cap (default 3) with NSS deprioritized in the enrich budget, and honest apparent-G legend/caption in PDF + UI.

## Output / findings

### Before/after reliable parallax counts
| Setup | pre-12b reliable | post-12b reliable | candidates (pre ? post) |
|-------|------------------|-------------------|-------------------------|
| draft_425 B_20_2 | 1015 | **7989** | 19 ? **6** |
| draft_425 V_20_2 | 795 | **6651** | 20 ? **7** |
| draft_425 R_20_2 | 1021 | **8011** | 19 ? **6** |
| draft_424 NoFilter_60_2 | 2474 | **3515** | 5 ? **4** |

Pre-12b snapshots preserved under `tmp/todo12_hrd/pre12b/`.

### draft_425 table composition (post-12b, online enrich)
**B_20_2 (6 rows):** 1 White dwarf, 3 Very cool, **2 Red supergiant** (chi Per M supergiants via Gaia TAP logg). No Binary rows (was 9; cap + physics-first budget).

**V_20_2 (7 rows):** 1 WD, 3 Very cool, **3 Red supergiant**.

**R_20_2 (6 rows):** 1 WD, 3 Very cool, **2 Red supergiant**.

**Not observed:** `Very hot` / `Hot luminous` rows on draft_425 - no threshold tuning applied (per spec). Cluster B stars sit on the main sequence in BP-RP/M_G; hot luminous requires bp_rp < -0.1 AND M_G < 0 with reliable parallax.

**draft_424:** Very cool + 1 Binary; modest reliable increase only (field stars, as expected).

### Fixture / test updates
- Added `test_parallax_gate_default_and_clamps` - no changes to existing TODO-12 tests.
- Added `test_category_cap_and_nss_deprioritized` - synthetic giants use bp_rp >= 2.6 (Stage-1 red-extreme net) so Stage-2 giant labels fire; no existing test expectations altered.

## DoD
- pytest: **693 passed**, 15 skipped (+2).
- `session_baseline_check.py --fast`: PASS (see terminal).
- PDF overflow draft_425 B_20_2: **0 violations**.
- Offline run: clean (4 rows, N/A teff/logg, no errors).

## Evidence
- PNGs: `tmp/todo12_hrd/draft425_{B,V,R}_hrd.png`, `draft424_hrd.png`
- Before: `tmp/todo12_hrd/pre12b/*.png`
- Summary: `tmp/todo12_hrd/summary.json`

## Errors (if any)
None.

## Files changed
- `hrd_analysis.py` - gate defaults 0.15 mas, `hrd_parallax_params_from_cfg`, NSS deprioritization, category cap, legend/caption
- `config.py` - 3 new keys
- `ui_hrd.py`, `photometry_report.py` - wire parallax params
- `citations.py`, `CITATIONS.bib` - Bailer-Jones et al. 2021 (+ Lindegren in HRD section)
- `docs/*`, `tests/test_hrd_extreme.py`, `scripts/todo12_hrd_validate.py`

Base commit: `f4c7c83` (TODO-12 + TODO-12b uncommitted; Milan review pending).
