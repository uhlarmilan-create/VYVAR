CURSOR RESULT — 2026-07-15 (F-428-MS-STAMP + F-428-COORD pre-T4)

What I did
- **T1:** Extended `scripts/diag_428_unmatched_sep.py` to v3 (`--forensics`); ran on draft_428 → `tmp/f428_coord_forensics.txt`.
- **T2:** Implemented `stamp_vsx_known_variable_on_masterstars()` (catalog_id join primary; positional fallback for vt rows without Gaia ID); wired post-match in `detect_stars_and_match_catalog` and post-optimizer in pipeline; unit tests in `tests/test_f428_msstamp_coord.py`.
- **T3:** Forensics **rejected** stale-WCS pass condition for unmatched DET_* (recomputed p50 81.141″ ≈ stored 81.053″). **No production coord fix** per task gate — Milan decision on optional matched-row Gaia coord refresh.
- **T4 gate:** pytest **866 passed**, 16 skipped; committed + pushed.

## Output / findings

### T1 — F-428-COORD forensics (`tmp/f428_coord_forensics.txt`)

**Provenance (T1.1)**
| Population | ra_deg/dec_deg source | file:line |
|------------|----------------------|-----------|
| Unmatched DET_* | Initial DAO WCS pix2world at detection | `pipeline.py:8245` |
| Matched rows | Same detection coords carried into df_out (not Gaia catalog coords) | `pipeline.py:8498-8500` |
| Optimizer | Updates catalog_id/mag only; does not refresh ra/dec | `astrometry_optimizer.py:488-503`, export ~1173 |
| WCS rescale | Recomputes all coords from x/y | `pipeline.py:3894-3907` |

**Systematic-offset test (T1.2) — unmatched DET_* (n=2724)**
| Metric | Stored coords | Recomputed (final MASTERSTAR WCS + x/y) |
|--------|---------------|------------------------------------------|
| vector-mean magnitude | 0.114″ | 0.226″ |
| mean \|Δ\| to nearest Gaia | 84.771″ | 84.777″ |
| **p50 \|Δ\|** | **81.053″** | **81.141″** |
| corr(\|Δ\|, field-center dist) | 0.0034 | — |

**DIAG SELF-CHECK FAIL (forensics):** recomputed p50 ≥ 20″ → **stale-WCS hypothesis NOT confirmed** for unmatched DET_*. Large p50 reflects unmatched DAO detections without nearby Gaia (not a fixable global WCS offset).

**vt↔ms violation source (T1.3):** 179 violations analyzed → **ms_deviates=176**, vt_deviates=3, tie=0. vt positions align with Gaia DB (VSX catalog coords); ms ra/dec retain detection-time WCS (5–60″ off Gaia for matched IDs).

### T2 — F-428-MS-STAMP dry-run (draft_428 archive)

| Metric | Value |
|--------|------:|
| Current `vsx_known_variable=True` | 46 |
| **Predicted after fix (catalog_id join)** | **197** unique vt IDs in masterstars |
| Simulated stamp total | **207** (197 id_join + 10 pre-existing positional) |
| Positional fallback (vt without Gaia ID) | 0 |

**8 known VSX candidates:** BO CVn, SS CVn, FZ CVn, FY CVn, FU CVn, RX CVn, NSVS 5096293 → all stamp **True** after fix (were False). R CVn has no masterstars row at vt catalog_id on this draft.

### T3 — F-428-COORD

**No code change** (T1.2 pass condition failed). Optional follow-up for Milan: matched-row `ra_deg/dec_deg` ← Gaia DB by `catalog_id` would resolve 176/179 vt↔ms violations; would **not** change unmatched DET p50.

**Science audit:** No consumer uses DET_* `ra_deg/dec_deg` for photometry/LC/proc — photometry uses pixel x/y; comp selection uses `catalog_id` + `variable_target_catalog_ids`; radius diagnostic uses metadata coords only.

### Downstream note (documented, no change)

`variability_detector.py:~565-570` hockey-stick field mask excludes `vsx_known_variable` rows — MS-STAMP fix slightly cleans stable-field envelope fit; not science-output-affecting for LC/proc columns.

## T4 checklist for Milan UI re-run

After single draft_428 re-run:
- [ ] `vsx_known_variable=True` count ≈ **197–207** (not 46); 7/8 known VSX candidates True at masterstar level
- [ ] diag v3 on fresh masterstars: unmatched DET p50 still ~81″ (expected — not a coord bug); vt↔ms violations unchanged until optional COORD follow-up
- [ ] Radius decision input: re-run `--forensics` for within 1×/1.5×/2× FWHM counts; Milan decides radius — **no radius change in this arc**
- [ ] Previous closeout T4 checklist unchanged (byte-identity proc/LC where applicable)

## Errors (if any)

None.

## Files changed

- `photometry_core.py` — `stamp_vsx_known_variable_on_masterstars()`
- `pipeline.py` — post-match + post-optimizer VSX stamp hooks
- `scripts/diag_428_unmatched_sep.py` — v3 forensics (`--forensics`)
- `tests/test_f428_msstamp_coord.py` — new
- `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_STATE.md`, `CHANGELOG.md`
- `CURSOR_RESULT_428_msstamp_coord.md`

Commit: `8e01e3d` (includes prior F-428 batch `d3ca223`); pushed to origin/main.
