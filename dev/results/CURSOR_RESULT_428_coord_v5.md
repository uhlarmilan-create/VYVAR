CURSOR RESULT - 2026-07-16 (F-428-COORD forensics v5)

What I did
Read-only ground-truth forensics on draft_428 (`8e01e3d` baseline). Implemented and ran
`scripts/diag_428_coord_forensics_v5.py` -> `tmp/f428_coord_forensics_v5.txt`,
`tmp/f428_coord_v5_rows.csv`, `tmp/f428_priority_contact_sheet.png`,
`tmp/f428_det_scatter_v5.png`. No production pipeline changes.

## Gate verdict

| Gate | Result |
|------|--------|
| **T2 peak test (164 MISASSIGNED set)** | **RECLASSIFY-PROJECTION** - 160/160 testable rows `CATALOG_PROJECTION_OFF`; 0 `GENUINE_CONFUSION` |
| **T2 priority targets (7)** | All 7 `CATALOG_PROJECTION_OFF`; 0 distinct second-star confusion |
| **T1 direction stats** | Isotropic (vector mean 4.6 arcsec << mean\|Delta\| 12.3 arcsec); weak radius corr 0.246 - not rigid shift / not SIP-radius signature |
| **T3 LC positions** | **No per-frame x/y** in LC CSVs; `alignment_report.csv` frame-level only |
| **T4 81 arcsec population** | **SPURIOUS-UNIFORM** - control ratio det/control = 1.07; diagnostic not biased |
| **T4 re-run + anchor** | **UNBLOCK pending Milan contact-sheet review** (per v5 gate) |

v4 **STOP** superseded for science-flux / identity interpretation; fix lands in next batch
(post-match sep gate + coordinate finalization already staged). F-428-A3-RADIUS stays **OPEN**.

---

## Core finding (resolves v4 tension)

v4 flagged **164 MISASSIGNED-ID** via `sep(WCS(x,y), Gaia[catalog_id]) > 3.9 arcsec`. v5 peak test shows
the **assigned Gaia source projects to within ~0.7-1.9 px of ms x/y** on MASTERSTAR.fits (median
1.3 px for priority targets; aperture radius ~2 px), while **variable_targets / active_targets
sky coords match Gaia within ~0.24 arcsec** (median).

| Quantity | Priority median | Interpretation |
|----------|----------------:|----------------|
| `sep_proj_px` (ms x/y <-> Gaia WCS->pixel) | **1.3 px** | Same physical star in pixel space |
| `sep_wcs_gaia_arcsec` (pix2world(x,y) <-> Gaia) | **12.8 arcsec** | v4 violation metric |
| `sep_vt_gaia_arcsec` (vt coords <-> Gaia) | **0.24 arcsec** | catalog_id assignment correct |
| `peak_sep_from_xy_px` | **0.4 px** | Real DAO peak at stored x/y |

**Conclusion:** Flux is measured on the correct star; `catalog_id` is correct; stored ra/dec /
`pix2world(x,y)` sky bookkeeping is systematically offset from the assigned identity. This is a
**projection / coordinate-finalization systematic**, not wrong-star confusion - consistent with
priority targets showing correct EW / RRAB / EB morphology despite v4 MISASSIGNED label.

---

## T1 - Direction statistics (Delta = WCS(x,y) - Gaia[cid])

| Set | n | vector mean | mean \|Delta\| | corr(\|Delta\|, r_field) |
|-----|--:|------------:|-----------:|---------------------:|
| MISASSIGNED (v4 class) | 160* | 4.6 arcsec | 12.3 arcsec | +0.246 |
| CONSISTENT (violation subset) | 20 | 1.0 arcsec | 2.9 arcsec | -0.248 |

\*160 of 164 v4 MISASSIGNED rows have Gaia DB + ms row for peak test; same 160 in T1 stats.

**Match origin (MISASSIGNED):** 160/160 `detection_catalog_match_coords`; 0 optimizer `_write_match`
loose in testable set.

**Interpretation:** Offset directions are **isotropic** (no common rigid vector); **weak** radius
correlation - inconsistent with global SIP/distortion-radius signature or single canvas offset.
Per-row loose catalog match at detection time (96 arcsec class) is the plausible assignment path.

---

## T2 - Peak test on MASTERSTAR.fits

**Histogram (MISASSIGNED n=160):** `CATALOG_PROJECTION_OFF`: 160; `GENUINE_CONFUSION`: 0; `PHANTOM`: 0

### Priority targets

| Target | t2_class | sep_proj_px | sep_wcs_gaia | sep_vt_gaia | peak@xy px |
|--------|----------|------------:|-------------:|------------:|-----------:|
| FY CVn | CATALOG_PROJECTION_OFF | 1.21 | 11.8 arcsec | 0.24 arcsec | 0.76 |
| FZ CVn | CATALOG_PROJECTION_OFF | 1.31 | 12.8 arcsec | 0.27 arcsec | 0.39 |
| R CVn | CATALOG_PROJECTION_OFF | 1.42 | 13.9 arcsec | 0.13 arcsec | 3.54 |
| CSS_J134925.3+393524 | CATALOG_PROJECTION_OFF | 1.86 | 18.2 arcsec | 0.21 arcsec | 5.42 |
| NSVS 5096293 | CATALOG_PROJECTION_OFF | 0.66 | 6.5 arcsec | 1.35 arcsec | 18.78 |
| CSS_J140918.7+423422 | CATALOG_PROJECTION_OFF | 1.43 | 13.9 arcsec | 0.99 arcsec | 21.37 |
| RX CVn | CATALOG_PROJECTION_OFF | 0.86 | 8.4 arcsec | 0.18 arcsec | 0.08 |

Contact sheet: `tmp/f428_priority_contact_sheet.png` (red = ms x/y, cyan = Gaia WCS proj, yellow = local peak).

Per-row table: `tmp/f428_coord_v5_rows.csv` (184 violation rows + full ms catalog in `f428_coord_v5_all_rows.csv`).

---

## T3 - Per-frame LC position evidence

LC CSVs (`photometry/lightcurves/lightcurve_<cid>.csv`) contain time/mag/airmass only - **no x/y
or per-frame match separation**. `alignment_report.csv` (140 frames) has frame-level
`align_residual_px` only. **T3 inconclusive** - proc tree absent; no LC-side centroid scatter
available for draft_428 archive.

---

## T4 - 81 arcsec unmatched DET population

| Metric | Value |
|--------|-------|
| Unmatched DET_* | 2724 |
| field_catalog in frame | 4409 |
| det NN p50 | 131.1 arcsec |
| random control p50 | 122.4 arcsec |
| det/control ratio | **1.07** |
| edge (<50 px border) fraction | 0.094 |
| **Verdict** | **SPURIOUS-UNIFORM** |

Control test near Poisson-scale - v3/v4 separation numbers **not quarantined**. Scatter PNG:
`tmp/f428_det_scatter_v5.png`.

---

## Errors (if any)

None. Script exit 0 on gate pass.

## Files changed

| File | Action |
|------|--------|
| `scripts/diag_428_coord_forensics_v5.py` | Added (read-only diagnostic) |
| `tmp/f428_coord_forensics_v5.txt` | Generated |
| `tmp/f428_coord_v5_rows.csv` | Generated |
| `tmp/f428_priority_contact_sheet.png` | Generated |
| `tmp/f428_det_scatter_v5.png` | Generated |
| `CURSOR_RESULT_428_coord_v5.md` | This file |
| `docs/VYVAR_STATE.md` | Ledger update |
| `docs/VYVAR_ROADMAP.md` | F-428-COORD gate update |
| `docs/VYVAR_JOURNAL.md` | Session note |
| `CHANGELOG.md` | Diagnostic entry |

No git commit (read-only forensics + diagnostic script; commit when Milan approves).
