CURSOR RESULT — 2026-06-22 (magnitude-calibration manual)

What I did
Read-only verification of post-`be3e193` code paths; wrote Czech ASCII calibration data-flow manual with file:line citations. **Doc only — no code change.**

## Doc choice

**New file:** `docs/VYVAR_CALIBRATION.md` (dedicated section "Kalibrace magnitud — datovy tok").

**Linked from:** `docs/VYVAR_PIPELINE_CZ.md` §7, `docs/VYVAR_STATE.md`, `docs/VYVAR_DECISIONS.md` (Path A decision block added).

Matches existing manual style: Czech, ASCII, code-anchored (same family as `VYVAR_PIPELINE_CZ.md`).

## Section outline (`VYVAR_CALIBRATION.md`)

1. Column lineage — ASCII diagram + CSV column table
2. Corrections — ensemble ZP, GS11, Savitzky-Golay, CT, AC, `mag_calib_final` (formula, gates, config defaults, file:line)
3. Consumer table — export, PDF, `lc_rms`, comp_qa, trust, variability
4. `err` column — photon + ensemble SEM, invariance under CT/AC, `ac_scatter` not in `err`
5. Cross-links — DECISIONS, ledger G5-F011, CITATIONS, canonical combination doc

## Source verification (post-`be3e193`)

| Item | Verified location |
|------|-------------------|
| `ensemble_normalize` / `mag_calib` formula | `photometry_core.py:2254-2452` |
| GS11 dilution gate | `dilution.py:346-359`, `config.py:485` |
| AC Method B `delta_m_corr` | `photometry_core.py:2075-2206`, gates ~7173-7180 |
| CT `apply_color_term` | `photometry_core.py:2580-2613`, per-target ~7394-7405 |
| `apply_reporting_postprocess` / `mag_calib_raw` | `photometry_core.py:3422-3474` |
| Savitzky-Golay + AC recompute | `photometry_core.py:7563-7576` |
| `compute_mag_calib_final` + CSV write | `photometry_core.py:3698-3735`, `save_lightcurve_csv` ~3809-3854 |
| Export `mag_calib_final` | `export_reports.py:645-686` |
| PDF `_publication_lc_mag_column` | `photometry_report.py:129-138` |
| `lc_rms` on `mag_calib` | `photometry_core.py:7843-7844` |
| Variability `dao_flux` | `variability_detector.py:277` |
| `err` assembly | `photometry_core.py:628-649`, ~7502-7516 |

**Note documented:** `mag_calib_ct` in CSV is CT on pre-SG `mag_calib`; `mag_calib_final` applies scalar `ct_correction` to post-SG `mag_calib` — canonical export uses `mag_calib_final`.

**VarAstro `delta_mag`:** left as ensemble differential (`export_reports.py:1057`) — documented explicitly.

## Commit

`docs: magnitude-calibration data-flow (lineage, corrections, consumers) in manual` — **`1da84a8`**.

**Not pushed** — stop for Claude review.

## Files changed

- `docs/VYVAR_CALIBRATION.md` (new)
- `docs/VYVAR_PIPELINE_CZ.md`, `docs/VYVAR_DECISIONS.md`, `docs/VYVAR_STATE.md`
- `CURSOR_RESULT.md`
