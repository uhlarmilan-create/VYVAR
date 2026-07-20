CURSOR RESULT - 2026-07-16 (F-428-COORD forensics v4)

What I did
Read-only identity-integrity forensics on draft_428 (`8e01e3d` baseline). Ran
`scripts/diag_428_coord_forensics_v4.py` -> `tmp/f428_coord_forensics_v4.txt`.
No production code changes.

## Verdict gates

| Gate | Result |
|------|--------|
| **T1 STALE-COORDS (all 184)** | **FAIL** - 0 STALE-COORDS, **164 MISASSIGNED-ID**, 20 CONSISTENT |
| **T1 any MISASSIGNED-ID** | **STOP** - T4 re-run + anchor path **BLOCKED** pending Milan |
| **T2 coverage (81 arcsec unmatched DET)** | **CONFIRMED** (spurious/unmatched DAO; not stale coords; not radius-fixable) |
| **T3 mag backstop** | 71/184 violation rows \|mag resid\| > 0.5 mag (ambiguous; does not overturn T1) |
| **T4 science consumers** | Photometry/LC use **x/y**; GS11 dilution uses ra/dec - see table below |

---

## v3 interpretation corrections (carried forward)

1. **Isotropic, not rigid shift:** vector_mean ~ 0.1 arcsec vs mean|Delta| ~ 85 arcsec for unmatched DET_*; corr(|Delta|, r) ~ 0.003; recompute-invariance -> unmatched DET sky coords are **fine**; 81 arcsec median = no nearby catalog row, not WCS staleness.
2. **Two matched populations:** (a) ms-gaia ~ 0 arcsec rows with detection-time Gaia coords (e.g. FW CVn stored); (b) 164 rows where **stored ~ WCS(x,y)** but **both** are 4-90 arcsec from assigned `catalog_id` Gaia position - **catalog_id <-> pixel sky mismatch**, not stale-metadata-only.

---

## T1 - Pixel-space identity test

**Population:** 184 rows with vt<->ms sep > 2 arcsec (179 from v3 + 5 active-only without vt row).

| Class | Count | Meaning |
|-------|------:|---------|
| **MISASSIGNED-ID** | **164** | sep(WCS(x,y), Gaia[catalog_id]) > 3.9 arcsec |
| CONSISTENT | 20 | sep(WCS(x,y), Gaia) <= 3.9 arcsec |
| STALE-COORDS | 0 | - |

**Subclass (184 violations):**
| Subclass | Count | Interpretation |
|----------|------:|----------------|
| wcs_stored <= 1 arcsec | 177 | Stored ra/dec **tracks** final WCS(x,y) - **not** initial-WCS staleness |
| stored_gaia <= 1 arcsec | 3 | Metadata forced to Gaia; pixels elsewhere (FW CVn, HAT-188-0000203, ...) |
| sep_wcs ~ sep_stored vs Gaia | 177 | Offset is in **pixel/catalog identity**, not stale sky columns |

### Priority active targets

| Target | Class | sep(WCS,Gaia) | sep(WCS,VSX) | sep(vt,ms) | LC (draft_428) |
|--------|-------|--------------:|-------------:|-----------:|----------------|
| FY CVn | MISASSIGNED | 11.8 arcsec | ~12.0 arcsec | 12.9 arcsec | EW, lc_rms=10.4%, dilution=1.0 |
| FZ CVn | MISASSIGNED | 12.8 arcsec | ~12.7 arcsec | 13.5 arcsec | EW, lc_rms=19.0%, dilution=1.0 |
| CSS_J134925.3+393524 | MISASSIGNED | 18.2 arcsec | ~18.3 arcsec | 19.3 arcsec | RRAB, lc_rms=23.0% |
| CSS_J140918.7+423422 | MISASSIGNED | 13.9 arcsec | - | 13.8 arcsec | RRAB, lc_rms=22.9% |
| NSVS 5096293 | MISASSIGNED | 6.5 arcsec | ~7.8 arcsec | 7.7 arcsec | EB, lc_rms=17.6% |
| RX CVn | MISASSIGNED | 8.4 arcsec | ~8.5 arcsec | 9.0 arcsec | RRAB, lc_rms=6.6% |
| R CVn | MISASSIGNED | 13.9 arcsec | - | 12.9 arcsec | (excluded from active) |

**LC corroboration:** Active targets show **expected VSX types** and plausible variability amplitude - **does not** resolve MISASSIGNED-ID under the task gate (pixel position is 6-19 arcsec from VSX **and** Gaia for the assigned cid). Milan should visually overlay x/y on MASTERSTAR vs VSX/Gaia before accepting science.

**MISASSIGNED-ID explicit list:** all 164 rows in `tmp/f428_coord_forensics_v4.txt` S MISASSIGNED-ID explicit list.

---

## T2 - Gaia coverage vs frame (81 arcsec unmatched DET)

| Item | Value |
|------|-------|
| Global SQLite DB | 211.7M rows; bbox essentially all-sky (not field-scoped) |
| Field catalog cone | r ~ 13.6 deg, center (209.49 deg, 41.16 deg); export cap 100k, G <= 15.26 |
| Sources in frame footprint | **4379** in `field_catalog_cone.csv` |
| Unmatched DET_* | 2724 |
| NN p50 (sqlite field slice) | 81.053 arcsec |
| NN p50 (field_catalog in frame) | **131.1 arcsec** |
| Within 20 arcsec of field_catalog | **11 / 2724** |

**COVERAGE VERDICT (81 arcsec population): CONFIRMED** - unmatched DET are **spurious/faint DAO detections without catalog association** in the loaded cone; **not** fixable by match-radius tuning. Actionable item remains **deeper/wider field DB** (GAIA-DR4 build - Milan decision; out of scope).

**Separate from T1:** the 164 MISASSIGNED matched rows are **not** the 81 arcsec unmatched population.

---

## T3 - Magnitude-consistency backstop

- Reference fit: 246 rows with ms-gaia <= 0.5 arcsec
- Violation rows with \|mag_resid\| > 0.5 mag: **71 / 184**
- Does **not** clear T1 STOP (many mismatches have consistent instrumental vs Gaia G brightness - consistent with nearby confusion, not identity proof)

---

## T4 - Science-consumer audit (masterstars `ra_deg`/`dec_deg`)

| Consumer | Path | Uses ra/dec? | draft_428 affected? | Evidence |
|----------|------|--------------|---------------------|----------|
| **Aperture photometry / LC extraction** | `photometry_core.py` ~5007, ~8415+ | **x/y primary** | **No** (flux at DAO centroid) | proc/LC built from masterstars x,y by catalog_id |
| **GS11 dilution cone query** | `dilution.py`, `photometry_core.py` ~8482 | **ra/dec** | **Low** for 177/184 (stored ~ WCS xy); **Yes** for 3 stored_gaia<=1 arcsec | FY/FZ/RX dilution_factor=**1.0** in photometry_summary |
| **Comp spatial pool** | `comp_selection_per_target.py` ~351-364 | **x/y when available** | **No** for draft_428 (pixel dist used) | BO CVn debug path uses pixel mode |
| **Comp color lookup** | `comp_selection_per_target.py` ~134 | ra/dec for fallback | **No** (catalog_id->Gaia DB primary) | bp_rp from SQLite by cid |
| **VSX proximity veto (comps)** | `pipeline.py` ~5968+ | variable_targets coords | **No** | Veto uses vt coords, not ms |
| **Variability detector field mask** | `variability_detector.py` ~570 | ms ra/dec in export | **No** science | Masks vsx_known_variable from envelope |
| **HRD / report rendering** | `hrd_analysis.py` | display only | **No** | PDF labels |
| **UI MASTERSTAR QA** | `ui_masterstar_qa.py` | overlay display | **No** science | Diagnostic |
| **k2 cohort / airmass** | frame-level | not ms ra/dec | **No** | - |
| **Lunar context** | `lunar_context.py` | frame-level moon | **No** | - |

**Summary:** Core photometry/LC/proc columns are **not** driven by stale ms ra/dec for the 177-row wcs_stored subclass. **Identity risk** is **catalog_id labeling a DAO centroid that is arcseconds from that Gaia source** - science validity depends on whether the centroid is the intended optical variable (LC morphology suggests yes for priority CVn targets; T1 gate still STOP per task spec).

---

## Recommended Milan actions (no code in this arc)

1. **STOP:** Do not treat draft_428 as anchor-valid until T1 MISASSIGNED-ID resolved or waived with visual proof.
2. **Visual overlay:** MASTERSTAR + x/y markers vs Gaia/VSX for FY CVn, FZ CVn, RX CVn (6-13 arcsec offsets).
3. **If waived:** root cause likely **optimizer loose catalog match** assigning Gaia IDs without pixel-Gaia agreement check - fix in next batch (post-match sep(WCS(x,y), Gaia[cid]) gate).
4. **81 arcsec unmatched DET:** accept as spurious-detection population; radius decision stays OPEN; field DB depth is separate track.

## Files

- `scripts/diag_428_coord_forensics_v4.py` (read-only diagnostic)
- `tmp/f428_coord_forensics_v4.txt`
- `CURSOR_RESULT_428_coord_v4.md`
- `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_STATE.md` (ledger)

## Errors

None (script Unicode print fixed; output written to tmp).
